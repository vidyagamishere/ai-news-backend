#!/usr/bin/env python3
"""
Health and status router for modular FastAPI architecture
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import HealthResponse, DatabaseStats
from app.dependencies.database import get_db
from db_service import PostgreSQLService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
@router.get("/", response_model=HealthResponse)
async def health_check(db: PostgreSQLService = Depends(get_db)):
    """
    Health check endpoint - compatible with existing frontend
    Endpoint: GET /health or GET /
    """
    try:
        logger.info("🏥 Health check requested")
        
        # Test database connection
        article_count = db.execute_query("SELECT COUNT(*) as count FROM articles", fetch_one=True)
        user_count = db.execute_query("SELECT COUNT(*) as count FROM users", fetch_one=True)
        topic_count = db.execute_query("SELECT COUNT(*) as count FROM ai_topics", fetch_one=True)
        
        database_stats = DatabaseStats(
            articles=article_count['count'] if article_count else 0,
            users=user_count['count'] if user_count else 0,
            ai_topics=topic_count['count'] if topic_count else 0,
            connection_pool="active"
        )
        
        response = HealthResponse(
            status="healthy",
            version="3.0.0-postgresql-modular",
            database="postgresql",
            migration_source="/app/ai_news.db",
            timestamp=datetime.utcnow(),
            database_stats=database_stats
        )
        
        logger.info("✅ Health check completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'status': 'unhealthy',
                'error': str(e),
                'database': 'postgresql',
                'timestamp': datetime.utcnow().isoformat()
            }
        )