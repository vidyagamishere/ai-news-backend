#!/usr/bin/env python3
"""
WebSocket router for real-time admin updates
Provides live updates for scraping progress, source changes, and article updates
"""

import os
import logging
import json
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for admin clients"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, dict] = {}
    
    async def connect(self, websocket: WebSocket, admin_key: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "admin_key": admin_key,
            "connected_at": datetime.utcnow().isoformat(),
            "messages_sent": 0
        }
        logger.info(f"✅ Admin WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        logger.info(f"❌ Admin WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        try:
            await websocket.send_json(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["messages_sent"] += 1
        except Exception as e:
            logger.error(f"❌ Failed to send personal message: {str(e)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected admin clients"""
        if not self.active_connections:
            return
        
        # Add timestamp to message
        message["timestamp"] = datetime.utcnow().isoformat()
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
                if connection in self.connection_metadata:
                    self.connection_metadata[connection]["messages_sent"] += 1
            except Exception as e:
                logger.error(f"❌ Broadcast failed for connection: {str(e)}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
        
        if self.active_connections:
            logger.debug(f"📡 Broadcast to {len(self.active_connections)} admin clients")
    
    def get_stats(self) -> dict:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "connections": [
                {
                    "connected_at": meta["connected_at"],
                    "messages_sent": meta["messages_sent"]
                }
                for meta in self.connection_metadata.values()
            ]
        }


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws/admin")
async def admin_websocket_endpoint(
    websocket: WebSocket,
    admin_key: str = Query(None)
):
    """
    WebSocket endpoint for real-time admin updates
    
    Events emitted:
    - scraping_started: { type, job_id, content_type, llm_model, frequency }
    - scraping_progress: { type, job_id, progress, current, total, message }
    - scraping_completed: { type, job_id, success, articles_inserted, errors }
    - source_updated: { type, source_id, action, data }
    - article_deleted: { type, article_id }
    - connection_info: { type, message, stats }
    """
    
    # Validate admin key
    ADMIN_API_KEY = os.getenv('ADMIN_API_KEY')
    if not admin_key or admin_key != ADMIN_API_KEY:
        await websocket.close(code=1008, reason="Invalid admin API key")
        return
    
    # Connect client
    await manager.connect(websocket, admin_key)
    
    try:
        # Send connection confirmation
        await manager.send_personal_message({
            "type": "connection_info",
            "message": "Connected to admin WebSocket",
            "stats": manager.get_stats()
        }, websocket)
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            
            # Handle ping/pong for keep-alive
            if data == "ping":
                await manager.send_personal_message({"type": "pong"}, websocket)
            else:
                # Echo back for debugging
                try:
                    message = json.loads(data)
                    logger.debug(f"📥 Received from admin: {message}")
                    await manager.send_personal_message({
                        "type": "echo",
                        "received": message
                    }, websocket)
                except json.JSONDecodeError:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "Invalid JSON"
                    }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Admin WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {str(e)}")
        manager.disconnect(websocket)


# Export manager for use in other modules
def get_websocket_manager() -> ConnectionManager:
    """Get the global WebSocket connection manager"""
    return manager