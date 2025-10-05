#!/usr/bin/env python3
"""
Main entry point for AI News Scraper API
Modular FastAPI application with PostgreSQL-only backend for Railway deployment
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Clear Python cache on startup to fix Railway caching issues
if hasattr(sys, '_getframe'):
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('app.'):
            del sys.modules[module_name]

# Add current directory to Python path for Railway
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Configure logging with DEBUG support
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if DEBUG else getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log the debug mode status
if DEBUG:
    logger.debug("🐛 DEBUG mode enabled - verbose logging activated")
else:
    logger.info(f"📊 Log level set to: {logging.getLevelName(log_level)}")

# Import modular components
from app.routers import health, auth, content, admin, personalized_feed
from app.routers import enhanced_auth
from db_service import initialize_database, close_database_service


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
    version="3.0.0-postgresql-modular-railway-fixed",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://www.vidyagam.com",
    "https://ai-news-react-theta.vercel.app",
    "https://ai-news-react.vercel.app",
    
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
app.include_router(health.router, tags=["health"])
app.include_router(auth.router, tags=["authentication"])  # Legacy auth endpoints
app.include_router(enhanced_auth.router, prefix="/api/v2", tags=["enhanced-auth"])  # New auth endpoints
app.include_router(content.router, tags=["content"])
app.include_router(admin.router, tags=["admin"])
app.include_router(personalized_feed.router, prefix="/api/v1", tags=["personalized-feed"])

# Additional endpoints for compatibility
@app.get("/sources")
async def get_sources():
    """Get all content sources - maintained for frontend compatibility"""
    try:
        from db_service import get_database_service
        db = get_database_service()
        
        sources_query = """
            SELECT s.name, s.rss_url, s.website, s.content_type, 
                   COALESCE(c.name, 'general') as category, 
                   s.priority, s.enabled
            FROM ai_sources s
            LEFT JOIN ai_categories_master c ON s.category_id = c.id
            WHERE s.enabled = TRUE
            ORDER BY s.priority ASC, s.name ASC
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

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting AI News Scraper API (Modular PostgreSQL Railway) on port {port}")
    uvicorn.run(
        "main:app",  # Use main:app for Railway
        host="0.0.0.0",
        port=port,
        reload=False
    )