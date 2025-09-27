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

# Add current directory to Python path for Railway
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Configure logging
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
app.include_router(health.router, tags=["health"])
app.include_router(auth.router, tags=["authentication"])
app.include_router(content.router, tags=["content"])

# Additional endpoints for compatibility
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


@app.post("/admin/setup-foreign-keys")
async def setup_foreign_keys(request: dict):
    """Admin endpoint to setup foreign key relationships"""
    try:
        admin_key = request.get('admin_key')
        if admin_key != 'setup-foreign-keys-2025':
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Invalid admin key")
        
        from db_service import get_database_service
        db = get_database_service()
        
        # Ensure ai_topics table exists with sample data
        ai_topics_exists_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'ai_topics'
        );
        """
        
        table_exists = db.execute_query(ai_topics_exists_query)[0]['exists']
        
        if not table_exists:
            # Create ai_topics table
            create_ai_topics_query = """
            CREATE TABLE ai_topics (
                id SERIAL PRIMARY KEY,
                category VARCHAR(50) UNIQUE NOT NULL,
                description TEXT,
                keywords TEXT,
                priority INTEGER DEFAULT 1,
                enabled BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            db.execute_query(create_ai_topics_query, fetch_results=False)
            
            # Insert sample topics
            sample_topics = [
                ('research', 'AI Research and Academic Papers', 'research, academic, papers, studies, experiments', 1),
                ('business', 'AI in Business and Industry', 'business, industry, enterprise, commercial, market', 2),
                ('technical', 'Technical AI Development and Tools', 'development, programming, tools, frameworks, libraries', 3),
                ('education', 'AI Education and Learning Resources', 'education, learning, courses, tutorials, training', 4),
                ('platform', 'AI Platforms and Services', 'platform, service, cloud, API, infrastructure', 5),
                ('robotics', 'Robotics and Automation', 'robotics, automation, robots, mechanical, hardware', 6),
                ('healthcare', 'AI in Healthcare and Medicine', 'healthcare, medicine, medical, health, diagnosis', 7),
                ('automotive', 'AI in Automotive and Transportation', 'automotive, transportation, vehicles, autonomous, driving', 8)
            ]
            
            for category, description, keywords, priority in sample_topics:
                insert_query = """
                INSERT INTO ai_topics (category, description, keywords, priority)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (category) DO NOTHING;
                """
                db.execute_query(insert_query, (category, description, keywords, priority), fetch_results=False)
        
        # Check and add ai_topics_id column to ai_sources
        column_exists_query = """
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'ai_sources' AND column_name = 'ai_topics_id' AND table_schema = 'public';
        """
        column_exists = db.execute_query(column_exists_query)
        
        if not column_exists:
            # Add ai_topics_id column
            db.execute_query("ALTER TABLE ai_sources ADD COLUMN ai_topics_id INTEGER;", fetch_results=False)
            
            # Add foreign key constraint for ai_topics_id
            try:
                db.execute_query("""
                    ALTER TABLE ai_sources 
                    ADD CONSTRAINT fk_ai_sources_ai_topics_id 
                    FOREIGN KEY (ai_topics_id) REFERENCES ai_topics(id) ON DELETE SET NULL;
                """, fetch_results=False)
            except Exception as fk_error:
                logger.warning(f"⚠️ Could not add ai_topics_id foreign key: {str(fk_error)}")
        
        # Add foreign key constraint for category if not exists
        try:
            constraint_query = """
            SELECT constraint_name FROM information_schema.table_constraints 
            WHERE table_name = 'ai_sources' AND constraint_type = 'FOREIGN KEY' 
            AND constraint_name = 'fk_ai_sources_category';
            """
            constraint_exists = db.execute_query(constraint_query)
            
            if not constraint_exists:
                db.execute_query("""
                    ALTER TABLE ai_sources 
                    ADD CONSTRAINT fk_ai_sources_category 
                    FOREIGN KEY (category) REFERENCES ai_topics(category) ON DELETE SET NULL;
                """, fetch_results=False)
        except Exception as cat_fk_error:
            logger.warning(f"⚠️ Could not add category foreign key: {str(cat_fk_error)}")
        
        logger.info("✅ Foreign key setup completed successfully")
        return {
            'success': True,
            'message': 'Foreign key relationships setup completed',
            'database': 'postgresql',
            'foreign_keys_added': [
                'ai_sources.ai_topics_id → ai_topics.id',
                'ai_sources.category → ai_topics.category'
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Foreign key setup failed: {str(e)}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Foreign key setup failed',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@app.get("/admin/foreign-key-options/{table_name}/{column_name}")
async def get_foreign_key_options(table_name: str, column_name: str):
    """Get foreign key options for dropdown menus"""
    try:
        from db_service import get_database_service
        db = get_database_service()
        
        options = []
        
        # Handle ai_sources table foreign keys
        if table_name == 'ai_sources':
            if column_name == 'ai_topics_id':
                # Get options from ai_topics.id
                query = "SELECT id, category, description FROM ai_topics WHERE enabled = true ORDER BY priority;"
                options = db.execute_query(query)
            elif column_name == 'category':
                # Get options from ai_topics.category
                query = "SELECT category, description FROM ai_topics WHERE enabled = true ORDER BY priority;"
                categories = db.execute_query(query)
                options = [{'category': row['category'], 'description': row['description']} for row in categories]
        
        return {
            'success': True,
            'options': [dict(option) for option in options],
            'table': table_name,
            'column': column_name,
            'database': 'postgresql'
        }
        
    except Exception as e:
        logger.error(f"❌ Foreign key options failed: {str(e)}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get foreign key options',
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