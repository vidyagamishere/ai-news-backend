#!/usr/bin/env python3
"""
Railway/Vercel compatible entry point for AI News Scraper API
Uses modular FastAPI architecture with PostgreSQL backend
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add current directory and parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Import modular components
    from app.routers import health, auth, content
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
            # Don't raise to allow startup even with DB issues
            logger.warning("⚠️ Continuing startup without database...")
        
        yield
        
        # Shutdown
        logger.info("🛑 Shutting down AI News Scraper API")
        try:
            close_database_service()
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"❌ Database shutdown error: {str(e)}")

    # Create FastAPI application
    app = FastAPI(
        title="AI News Scraper API",
        description="Modular FastAPI backend for AI news aggregation with PostgreSQL",
        version="3.0.0-postgresql-modular-railway-api",
        lifespan=lifespan
    )

    # Configure CORS
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173", 
        "https://ai-news-react.vercel.app",
        "https://www.vidyagam.com",
    ]

    # Add environment origins
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

    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(auth.router, tags=["authentication"])
    app.include_router(content.router, tags=["content"])

    # Additional endpoints for compatibility
    @app.get("/sources")
    async def get_sources():
        """Get all content sources"""
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

    logger.info("✅ FastAPI app created successfully with modular architecture")

except ImportError as e:
    logger.error(f"❌ Import error: {str(e)}")
    logger.info("🔄 Creating fallback minimal FastAPI app...")
    
    # Fallback minimal app if imports fail
    app = FastAPI(
        title="AI News Scraper API", 
        description="Fallback API while deployment resolves",
        version="3.0.0-fallback"
    )
    
    @app.get("/")
    async def root():
        return {
            "message": "AI News Scraper API - Modular Architecture",
            "status": "fallback_mode",
            "version": "3.0.0-fallback",
            "note": "Import error occurred, using fallback app"
        }
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "version": "3.0.0-fallback",
            "database": "not_connected",
            "mode": "fallback"
        }

except Exception as e:
    logger.error(f"❌ Unexpected error: {str(e)}")
    # Last resort fallback
    app = FastAPI(title="AI News Scraper API", version="3.0.0-error")
    
    @app.get("/")
    async def root():
        return {"message": "Error during startup", "version": "3.0.0-error"}

# Export for Railway/Vercel
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting Railway deployment on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)