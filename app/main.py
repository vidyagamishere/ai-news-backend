#!/usr/bin/env python3
"""
Main FastAPI application for AI News Scraper
Modular architecture with PostgreSQL-only backend
Maintains exact API compatibility with existing frontend
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import modular routers
from app.routers import health, auth, content
from db_service import initialize_database, close_database_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Starting AI News Scraper API with PostgreSQL")
    logger.info("🐘 Initializing PostgreSQL database and migration from SQLite...")
    
    try:
        # Initialize database and run SQLite migration if needed
        initialize_database()
        logger.info("✅ Database initialization completed")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI News Scraper API")
    try:
        close_database_service()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Database shutdown error: {str(e)}")


# Create FastAPI application with lifespan events
app = FastAPI(
    title="AI News Scraper API",
    description="Modular FastAPI backend for AI news aggregation with PostgreSQL",
    version="3.0.0-postgresql-modular-deployed",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://ai-news-react.vercel.app",
    "https://www.vidyagam.com",
]

# Add any additional origins from environment
env_origins = os.getenv('ALLOWED_ORIGINS', '')
if env_origins:
    allowed_origins.extend([origin.strip() for origin in env_origins.split(',')])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with exact same endpoints as before for frontend compatibility
# Note: Using include_router without prefix to maintain exact API paths
app.include_router(health.router, tags=["health"])
app.include_router(auth.router, tags=["authentication"])
app.include_router(content.router, tags=["content"])

# Additional endpoints for admin and utilities
@app.get("/sources")
async def get_sources():
    """Get all content sources - maintained for frontend compatibility"""
    try:
        from db_service import get_database_service
        db = get_database_service()
        
        sources_query = """
            SELECT name, rss_url, website, content_type, category, priority, enabled
            FROM ai_sources
            WHERE enabled = TRUE
            ORDER BY priority ASC, name ASC
        """
        
        sources = db.execute_query(sources_query)
        
        processed_sources = []
        for source in sources:
            processed_sources.append({
                'name': source['name'],
                'rss_url': source['rss_url'],
                'website': source.get('website', ''),
                'content_type': source['content_type'],
                'category': source.get('category', 'general'),
                'priority': source['priority'],
                'enabled': source['enabled']
            })
        
        return {
            'sources': processed_sources,
            'total_count': len(processed_sources),
            'enabled_count': len([s for s in processed_sources if s['enabled']]),
            'database': 'postgresql'
        }
        
    except Exception as e:
        logger.error(f"❌ Sources endpoint failed: {str(e)}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get sources',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@app.post("/admin/cleanup-test-users")
async def cleanup_test_users(request: dict):
    """Admin endpoint to cleanup test users"""
    try:
        admin_key = request.get('admin_key')
        if admin_key != 'test-cleanup-key-2025':
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Invalid admin key")
        
        from db_service import get_database_service
        db = get_database_service()
        
        # Delete test users (emails containing 'test', 'example', or 'debug')
        cleanup_query = """
            DELETE FROM users 
            WHERE email LIKE '%test%' 
               OR email LIKE '%example%' 
               OR email LIKE '%debug%'
               OR email LIKE '%@test.com'
               OR email LIKE '%@example.com'
        """
        
        db.execute_query(cleanup_query, fetch_results=False)
        
        logger.info("✅ Test users cleaned up successfully")
        return {
            'success': True,
            'message': 'Test users cleaned up successfully',
            'database': 'postgresql'
        }
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {str(e)}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Cleanup failed',
                'message': str(e)
            }
        )


# Archive endpoint for newsletter functionality
@app.get("/archive")
async def get_archive():
    """Get newsletter archive - maintained for frontend compatibility"""
    try:
        from db_service import get_database_service
        db = get_database_service()
        
        # Get archived newsletters
        archive_query = """
            SELECT id, title, content, created_at, sent_at
            FROM newsletters
            WHERE sent_at IS NOT NULL
            ORDER BY sent_at DESC
            LIMIT 50
        """
        
        archives = db.execute_query(archive_query)
        
        processed_archives = []
        for archive in archives:
            archive_dict = dict(archive)
            # Convert timestamps to ISO format
            for field in ['created_at', 'sent_at']:
                if archive_dict.get(field):
                    archive_dict[field] = archive_dict[field].isoformat() if hasattr(archive_dict[field], 'isoformat') else str(archive_dict[field])
            processed_archives.append(archive_dict)
        
        return {
            'archives': processed_archives,
            'total_archives': len(processed_archives),
            'database': 'postgresql'
        }
        
    except Exception as e:
        logger.error(f"❌ Archive endpoint failed: {str(e)}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get archive',
                'message': str(e),
                'database': 'postgresql'
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )