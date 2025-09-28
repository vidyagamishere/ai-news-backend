#!/usr/bin/env python3
"""
PostgreSQL Database Service for AI News Scraper
Single database backend using psycopg2 with connection pooling
"""

import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
import psycopg2.sql

logger = logging.getLogger(__name__)

class PostgreSQLService:
    def __init__(self):
        """Initialize PostgreSQL connection pool"""
        self.database_url = os.getenv('POSTGRES_URL') or os.getenv('DATABASE_URL')
        self.skip_schema_init = os.getenv('SKIP_SCHEMA_INIT', 'false').lower() == 'true'
        
        if not self.database_url:
            raise ValueError("POSTGRES_URL or DATABASE_URL environment variable is required")
        
        logger.info(f"🐘 Initializing PostgreSQL service")
        logger.info(f"📊 Database URL configured: {self.database_url[:50]}...")
        logger.info(f"⚙️ Skip schema initialization: {self.skip_schema_init}")
        
        # Create connection pool
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20,  # min and max connections
                self.database_url,
                cursor_factory=RealDictCursor
            )
            logger.info("✅ PostgreSQL connection pool created successfully")
            
            # Initialize database schema only if not skipped
            if not self.skip_schema_init:
                self.initialize_database()
            else:
                logger.info("⏭️ Skipping database schema initialization (existing database)")
            
        except Exception as e:
            logger.error(f"❌ Failed to create PostgreSQL connection pool: {e}")
            raise e
    
    @contextmanager
    def get_db_connection(self):
        """Get database connection from pool with automatic cleanup"""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ Database connection error: {e}")
            raise e
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = True) -> Optional[Any]:
        """Execute a query with automatic connection management"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
                
                if fetch_one:
                    return cursor.fetchone()
                elif fetch_all:
                    return cursor.fetchall()
                else:
                    return cursor.rowcount
    
    def initialize_database(self):
        """Initialize PostgreSQL database schema"""
        logger.info("🏗️ Initializing PostgreSQL database schema...")
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    
                    # Create content_types table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS content_types (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(50) UNIQUE NOT NULL,
                            display_name VARCHAR(100) NOT NULL,
                            description TEXT,
                            frontend_section VARCHAR(50),
                            icon VARCHAR(10),
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Create ai_topics table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS ai_topics (
                            id VARCHAR(100) PRIMARY KEY,
                            name VARCHAR(200) NOT NULL,
                            description TEXT,
                            category VARCHAR(100) NOT NULL,
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Create articles table with foreign keys
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS articles (
                            id SERIAL PRIMARY KEY,
                            source VARCHAR(255),
                            title TEXT,
                            url TEXT UNIQUE,
                            published_at TIMESTAMP,
                            description TEXT,
                            content TEXT,
                            significance_score INTEGER DEFAULT 5,
                            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            category VARCHAR(100) DEFAULT 'general',
                            reading_time INTEGER DEFAULT 3,
                            image_url TEXT,
                            keywords TEXT,
                            content_type_id INTEGER REFERENCES content_types(id),
                            ai_topic_id VARCHAR(100) REFERENCES ai_topics(id),
                            processing_status VARCHAR(50) DEFAULT 'pending',
                            content_hash VARCHAR(64),
                            audio_url TEXT,
                            video_url TEXT,
                            thumbnail_url TEXT,
                            view_count INTEGER DEFAULT 0,
                            duration_minutes INTEGER
                        );
                    """)
                    
                    # Create article_topics junction table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS article_topics (
                            id SERIAL PRIMARY KEY,
                            article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
                            topic_id VARCHAR(100) REFERENCES ai_topics(id) ON DELETE CASCADE,
                            relevance_score FLOAT DEFAULT 1.0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(article_id, topic_id)
                        );
                    """)
                    
                    # Create users table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id VARCHAR(255) PRIMARY KEY,
                            email VARCHAR(255) UNIQUE NOT NULL,
                            name VARCHAR(255),
                            profile_image TEXT,
                            subscription_tier VARCHAR(50) DEFAULT 'free',
                            preferences JSONB DEFAULT '{}',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_login_at TIMESTAMP,
                            verified_email BOOLEAN DEFAULT FALSE,
                            is_admin BOOLEAN DEFAULT FALSE
                        );
                    """)
                    
                    # Create user_preferences table for detailed preferences as per functional requirements
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS user_preferences (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(255) REFERENCES users(id) ON DELETE CASCADE,
                            experience_level VARCHAR(50) DEFAULT 'beginner',
                            professional_roles TEXT[],
                            newsletter_frequency VARCHAR(20) DEFAULT 'weekly',
                            email_notifications BOOLEAN DEFAULT TRUE,
                            breaking_news_alerts BOOLEAN DEFAULT FALSE,
                            push_notifications BOOLEAN DEFAULT FALSE,
                            mobile_number VARCHAR(20),
                            onboarding_completed BOOLEAN DEFAULT FALSE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(user_id)
                        );
                    """)
                    
                    # Create user_topic_preferences table for AI topic selections
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS user_topic_preferences (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(255) REFERENCES users(id) ON DELETE CASCADE,
                            topic_id VARCHAR(100) REFERENCES ai_topics(id) ON DELETE CASCADE,
                            selected BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(user_id, topic_id)
                        );
                    """)
                    
                    # Create user_content_type_preferences table for content type selections
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS user_content_type_preferences (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(255) REFERENCES users(id) ON DELETE CASCADE,
                            content_type_id INTEGER REFERENCES content_types(id) ON DELETE CASCADE,
                            selected BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(user_id, content_type_id)
                        );
                    """)
                    
                    # Create user_passwords table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS user_passwords (
                            user_id VARCHAR(255) PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                            password_hash TEXT NOT NULL,
                            salt TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Create user_sessions table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS user_sessions (
                            id VARCHAR(255) PRIMARY KEY,
                            user_id VARCHAR(255) REFERENCES users(id) ON DELETE CASCADE,
                            token_hash TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP,
                            last_used_at TIMESTAMP
                        );
                    """)
                    
                    # Create daily_archives table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS daily_archives (
                            id SERIAL PRIMARY KEY,
                            archive_date DATE UNIQUE,
                            digest_data JSONB,
                            article_count INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB DEFAULT '{}'
                        );
                    """)
                    
                    # Create ai_sources table (consolidated sources table)
                    # First, ensure the table has all required columns
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS ai_sources (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(255) NOT NULL,
                            rss_url TEXT NOT NULL,
                            website TEXT,
                            content_type VARCHAR(50) NOT NULL,
                            category VARCHAR(100),
                            enabled BOOLEAN DEFAULT TRUE,
                            priority INTEGER DEFAULT 5,
                            ai_topic_id VARCHAR(100),
                            meta_tags TEXT,
                            description TEXT,
                            last_scraped TIMESTAMP,
                            scrape_frequency_hours INTEGER DEFAULT 6,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Add foreign key constraint separately (in case ai_topics doesn't exist yet)
                    try:
                        cursor.execute("""
                            ALTER TABLE ai_sources 
                            ADD CONSTRAINT fk_ai_sources_topic 
                            FOREIGN KEY (ai_topic_id) REFERENCES ai_topics(id);
                        """)
                    except Exception as fk_error:
                        logger.warning(f"⚠️ Foreign key constraint already exists or ai_topics not ready: {fk_error}")
                    
                    # Ensure enabled column exists (in case of partial table creation)
                    try:
                        cursor.execute("""
                            ALTER TABLE ai_sources 
                            ADD COLUMN IF NOT EXISTS enabled BOOLEAN DEFAULT TRUE;
                        """)
                    except Exception as col_error:
                        logger.warning(f"⚠️ Column enabled already exists: {col_error}")
                    
                    # Create indexes for performance with error handling
                    index_queries = [
                        ("idx_articles_published_at", "CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at DESC);"),
                        ("idx_articles_content_type", "CREATE INDEX IF NOT EXISTS idx_articles_content_type ON articles(content_type_id);"),
                        ("idx_articles_topic", "CREATE INDEX IF NOT EXISTS idx_articles_topic ON articles(ai_topic_id);"),
                        ("idx_article_topics_article", "CREATE INDEX IF NOT EXISTS idx_article_topics_article ON article_topics(article_id);"),
                        ("idx_article_topics_topic", "CREATE INDEX IF NOT EXISTS idx_article_topics_topic ON article_topics(topic_id);"),
                        ("idx_users_email", "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);"),
                        ("idx_ai_sources_enabled", "CREATE INDEX IF NOT EXISTS idx_ai_sources_enabled ON ai_sources(enabled);"),
                        ("idx_ai_sources_topic", "CREATE INDEX IF NOT EXISTS idx_ai_sources_topic ON ai_sources(ai_topic_id);"),
                        ("idx_user_preferences_user", "CREATE INDEX IF NOT EXISTS idx_user_preferences_user ON user_preferences(user_id);"),
                        ("idx_user_topic_prefs_user", "CREATE INDEX IF NOT EXISTS idx_user_topic_prefs_user ON user_topic_preferences(user_id);"),
                        ("idx_user_content_prefs_user", "CREATE INDEX IF NOT EXISTS idx_user_content_prefs_user ON user_content_type_preferences(user_id);")
                    ]
                    
                    for index_name, index_query in index_queries:
                        try:
                            cursor.execute(index_query)
                            logger.info(f"✅ Index {index_name} created successfully")
                        except Exception as idx_error:
                            logger.warning(f"⚠️ Index {index_name} creation failed: {idx_error}")
                            # Continue with other indexes
                    
                    # Consolidate sources tables - migrate any data from old 'sources' table to 'ai_sources'
                    cursor.execute("""
                        DO $$
                        BEGIN
                            -- Check if old 'sources' table exists and migrate data
                            IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'sources') THEN
                                -- Migrate data from sources to ai_sources if ai_sources is empty
                                INSERT INTO ai_sources (name, rss_url, website, content_type, category, enabled, priority)
                                SELECT 
                                    name, 
                                    COALESCE(rss_url, url) as rss_url,
                                    website,
                                    COALESCE(content_type, 'articles') as content_type,
                                    COALESCE(category, 'general') as category,
                                    COALESCE(enabled, true) as enabled,
                                    COALESCE(priority, 5) as priority
                                FROM sources
                                WHERE NOT EXISTS (SELECT 1 FROM ai_sources WHERE ai_sources.name = sources.name);
                                
                                -- Drop the old sources table
                                DROP TABLE sources CASCADE;
                                
                                RAISE NOTICE 'Migrated data from sources table to ai_sources and dropped old table';
                            END IF;
                        END $$;
                    """)
                    
                    # Create optimized database views
                    self.create_database_views(cursor)
                    
                    # Populate master data
                    self.populate_content_types(cursor)
                    self.populate_ai_topics(cursor)
                    self.populate_ai_sources(cursor)
                    
                    conn.commit()
                    logger.info("✅ PostgreSQL database schema initialized successfully")
                    
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise e
    
    def create_database_views(self, cursor):
        """Create optimized database views for content delivery"""
        logger.info("📊 Creating optimized database views...")
        
        # Enhanced articles view with topic information
        cursor.execute("""
            CREATE OR REPLACE VIEW articles_with_topics AS
            SELECT 
                a.*,
                ct.name as content_type_name,
                ct.display_name as content_type_display,
                STRING_AGG(DISTINCT at2.name, ', ') as topic_names,
                STRING_AGG(DISTINCT at2.category, ', ') as topic_categories,
                COALESCE(
                    JSON_AGG(
                        DISTINCT jsonb_build_object(
                            'id', at2.id,
                            'name', at2.name,
                            'category', at2.category
                        )
                    ) FILTER (WHERE at2.id IS NOT NULL),
                    '[]'::json
                ) as topics,
                COUNT(DISTINCT att.topic_id) as topic_count
            FROM articles a
            LEFT JOIN content_types ct ON a.content_type_id = ct.id
            LEFT JOIN article_topics att ON a.id = att.article_id
            LEFT JOIN ai_topics at2 ON att.topic_id = at2.id
            GROUP BY a.id, ct.name, ct.display_name;
        """)
        
        # Optimized digest view
        cursor.execute("""
            CREATE OR REPLACE VIEW digest_articles AS
            SELECT 
                awt.*,
                CASE 
                    WHEN awt.published_at > NOW() - INTERVAL '24 hours' THEN 'today'
                    WHEN awt.published_at > NOW() - INTERVAL '7 days' THEN 'week'
                    ELSE 'older'
                END as recency_category
            FROM articles_with_topics awt
            WHERE awt.significance_score >= 6
            ORDER BY awt.published_at DESC, awt.significance_score DESC;
        """)
        
        logger.info("✅ Database views created successfully")
    
    def populate_content_types(self, cursor):
        """Populate content_types table with master data"""
        logger.info("📋 Populating content_types table...")
        
        # Check if already populated
        cursor.execute("SELECT COUNT(*) FROM content_types")
        count = cursor.fetchone()['count']
        
        if count > 0:
            logger.info(f"📊 Found {count} existing content types, skipping population")
            return
        
        content_types = [
            ("blogs", "Blog Articles", "News articles, analysis pieces, and written content", "blog", "📝"),
            ("podcasts", "Podcasts", "Audio content and podcast episodes", "audio", "🎧"),
            ("videos", "Videos", "Video content and tutorials", "video", "📹"),
            ("events", "Events", "Conferences, webinars, and industry events", "events", "📅"),
            ("learning", "Learning Resources", "Courses, tutorials, and educational content", "learning", "📚"),
            ("demos", "Demos & Tools", "Interactive demonstrations and AI tools", "demos", "🛠️")
        ]
        
        for name, display_name, description, section, icon in content_types:
            cursor.execute("""
                INSERT INTO content_types (name, display_name, description, frontend_section, icon)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING
            """, (name, display_name, description, section, icon))
        
        logger.info("✅ Content types populated successfully")
    
    def populate_ai_topics(self, cursor):
        """Populate ai_topics table with comprehensive AI topics"""
        logger.info("📋 Populating ai_topics table...")
        
        # Check if already populated
        cursor.execute("SELECT COUNT(*) FROM ai_topics")
        count = cursor.fetchone()['count']
        
        if count > 0:
            logger.info(f"📊 Found {count} existing AI topics, skipping population")
            return
        
        ai_topics = [
            ("ml_foundations", "Machine Learning", "Core ML algorithms, techniques, and foundations", "research"),
            ("deep_learning", "Deep Learning", "Neural networks, deep learning research and applications", "research"),
            ("nlp_llm", "Natural Language Processing", "Language models, NLP, and conversational AI", "language"),
            ("computer_vision", "Computer Vision", "Image recognition, visual AI, and computer vision", "research"),
            ("ai_tools", "AI Tools & Platforms", "New AI tools and platforms for developers", "platform"),
            ("ai_research", "AI Research Papers", "Latest academic research and scientific breakthroughs", "research"),
            ("ai_ethics", "AI Ethics & Safety", "Responsible AI, safety research, and ethical considerations", "policy"),
            ("robotics", "Robotics & Automation", "Physical AI, robotics, and automation systems", "robotics"),
            ("ai_business", "AI in Business", "Enterprise AI and industry applications", "company"),
            ("ai_startups", "AI Startups & Funding", "New AI companies and startup ecosystem", "startup"),
            ("ai_regulation", "AI Policy & Regulation", "Government policies and AI governance", "policy"),
            ("ai_hardware", "AI Hardware & Computing", "AI chips and hardware innovations", "hardware"),
            ("ai_automotive", "AI in Automotive", "Self-driving cars and automotive AI", "automotive"),
            ("ai_healthcare", "AI in Healthcare", "Medical AI applications and healthcare tech", "healthcare"),
            ("ai_finance", "AI in Finance", "Financial AI, trading, and fintech applications", "finance"),
            ("ai_gaming", "AI in Gaming", "Game AI, procedural generation, and gaming tech", "gaming"),
            ("ai_creative", "AI in Creative Arts", "AI for art, music, design, and creative content", "creative"),
            ("ai_cloud", "AI Cloud Services", "Cloud-based AI services and infrastructure", "cloud"),
            ("ai_events", "AI Events & Conferences", "AI conferences, workshops, and industry events", "events"),
            ("ai_learning", "AI Learning & Education", "AI courses, tutorials, and educational content", "learning"),
            ("ai_news", "AI News & Updates", "Latest AI news and industry updates", "news"),
            ("ai_international", "AI International", "Global AI developments and international news", "international"),
        ]
        
        for topic_id, name, description, category in ai_topics:
            cursor.execute("""
                INSERT INTO ai_topics (id, name, description, category, is_active)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (topic_id, name, description, category, True))
        
        logger.info("✅ AI topics populated successfully")
    
    def populate_ai_sources(self, cursor):
        """Populate ai_sources table with comprehensive AI news sources"""
        # Check if ai_sources already has data
        cursor.execute("SELECT COUNT(*) FROM ai_sources")
        existing_count = cursor.fetchone()['count']
        
        if existing_count > 0:
            logger.info(f"📊 Found {existing_count} existing AI sources, skipping population")
            return
            
        logger.info("📋 Populating ai_sources table...")
        
        # Comprehensive AI news sources
        ai_sources = [
            # AI Research & Papers
            ("OpenAI Blog", "https://openai.com/blog/rss/", "https://openai.com/blog", "blogs", "research", True, 1),
            ("Anthropic Blog", "https://www.anthropic.com/news", "https://www.anthropic.com", "blogs", "research", True, 1),
            ("DeepMind Blog", "https://deepmind.google/discover/blog/", "https://deepmind.google", "blogs", "research", True, 1),
            ("AI Research", "https://ai.googleblog.com/feeds/posts/default", "https://ai.googleblog.com", "blogs", "research", True, 2),
            ("Meta AI Blog", "https://ai.meta.com/blog/", "https://ai.meta.com", "blogs", "research", True, 2),
            
            # Industry & Business
            ("VentureBeat AI", "https://venturebeat.com/ai/feed/", "https://venturebeat.com/ai", "blogs", "business", True, 2),
            ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "https://techcrunch.com", "blogs", "business", True, 2),
            ("The Information AI", "https://www.theinformation.com/topics/artificial-intelligence", "https://www.theinformation.com", "blogs", "business", True, 3),
            ("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/", "https://www.technologyreview.com", "blogs", "research", True, 2),
            
            # Technical News
            ("Towards Data Science", "https://towardsdatascience.com/feed", "https://towardsdatascience.com", "blogs", "technical", True, 3),
            ("AI News", "https://artificialintelligence-news.com/feed/", "https://artificialintelligence-news.com", "blogs", "technical", True, 3),
            ("Machine Learning Mastery", "https://machinelearningmastery.com/feed/", "https://machinelearningmastery.com", "blogs", "education", True, 4),
            ("Analytics Vidhya", "https://www.analyticsvidhya.com/blog/feed/", "https://www.analyticsvidhya.com", "blogs", "education", True, 4),
            
            # Podcasts
            ("Lex Fridman Podcast", "https://lexfridman.com/podcast/", "https://lexfridman.com", "podcasts", "interviews", True, 2),
            ("AI Podcast", "https://blogs.nvidia.com/ai-podcast/", "https://blogs.nvidia.com", "podcasts", "technical", True, 3),
            ("The AI Podcast by NVIDIA", "https://soundcloud.com/theaipodcast", "https://soundcloud.com/theaipodcast", "podcasts", "technical", True, 3),
            
            # Videos & YouTube
            ("Two Minute Papers", "https://www.youtube.com/c/K%C3%A1rolyZsolnai", "https://www.youtube.com/c/K%C3%A1rolyZsolnai", "videos", "education", True, 3),
            ("AI Explained", "https://www.youtube.com/c/AIExplained-Official", "https://www.youtube.com/c/AIExplained-Official", "videos", "education", True, 4),
            ("Yannic Kilcher", "https://www.youtube.com/c/YannicKilcher", "https://www.youtube.com/c/YannicKilcher", "videos", "research", True, 4),
            
            # Learning Resources
            ("Coursera AI Blog", "https://blog.coursera.org/tag/artificial-intelligence/", "https://blog.coursera.org", "learning", "education", True, 4),
            ("edX AI News", "https://blog.edx.org/tag/artificial-intelligence", "https://blog.edx.org", "learning", "education", True, 4),
            ("Udacity AI Blog", "https://blog.udacity.com/tag/artificial-intelligence", "https://blog.udacity.com", "learning", "education", True, 4),
            
            # Demonstrations & Tools
            ("Hugging Face Blog", "https://huggingface.co/blog", "https://huggingface.co", "demos", "platform", True, 2),
            ("OpenAI Platform", "https://platform.openai.com/docs", "https://platform.openai.com", "demos", "platform", True, 2),
            ("Papers with Code", "https://paperswithcode.com/", "https://paperswithcode.com", "demos", "research", True, 3),
            
            # Events & Conferences
            ("NeurIPS", "https://neurips.cc/", "https://neurips.cc", "events", "research", True, 3),
            ("ICML", "https://icml.cc/", "https://icml.cc", "events", "research", True, 3),
            ("ICLR", "https://iclr.cc/", "https://iclr.cc", "events", "research", True, 3)
        ]
        
        for name, rss_url, website, content_type, category, enabled, priority in ai_sources:
            cursor.execute("""
                INSERT INTO ai_sources (name, rss_url, website, content_type, category, enabled, priority)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING
            """, (name, rss_url, website, content_type, category, enabled, priority))
        
        logger.info("✅ AI sources populated successfully")
    
    
    
    def get_ai_sources(self) -> List[Dict[str, Any]]:
        """Get all AI sources for scraping"""
        try:
            query = """
                SELECT id, name, rss_url, website, category, priority, is_active
                FROM ai_sources 
                WHERE is_active = TRUE 
                ORDER BY priority DESC, name
            """
            sources = self.execute_query(query, fetch_all=True)
            
            if not sources:
                # Return default sources if none exist
                logger.warning("⚠️ No AI sources found in database, using defaults")
                return [
                    {'id': 1, 'name': 'OpenAI Blog', 'url': 'https://openai.com/blog/', 'website': 'https://openai.com/blog/', 'category': 'company', 'priority': 10, 'is_active': True},
                    {'id': 2, 'name': 'Google AI Blog', 'url': 'https://ai.googleblog.com/', 'website': 'https://ai.googleblog.com/', 'category': 'company', 'priority': 10, 'is_active': True},
                    {'id': 3, 'name': 'Anthropic News', 'url': 'https://www.anthropic.com/news', 'website': 'https://www.anthropic.com/news', 'category': 'company', 'priority': 9, 'is_active': True},
                    {'id': 4, 'name': 'DeepMind Blog', 'url': 'https://deepmind.google/discover/blog/', 'website': 'https://deepmind.google/discover/blog/', 'category': 'research', 'priority': 9, 'is_active': True},
                ]
            
            return [dict(source) for source in sources]
            
        except Exception as e:
            logger.error(f"❌ Failed to get AI sources: {e}")
            return []
    
    def insert_article(self, article_data: Dict[str, Any]) -> bool:
        """Insert scraped article into articles table"""
        try:
            # Check if article already exists
            existing_query = "SELECT id FROM articles WHERE url = %s"
            existing = self.execute_query(existing_query, (article_data['url'],), fetch_one=True)
            
            if existing:
                logger.info(f"📄 Article already exists: {article_data.get('title', 'Unknown')}")
                return False
            
            # Insert new article
            insert_query = """
                INSERT INTO articles (
                    id, title, description, content_summary, url, source, 
                    category, content_type, significance_score, published_date, 
                    scraped_date, llm_processed, is_current_day
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            values = (
                article_data.get('id'),
                article_data.get('title'),
                article_data.get('description'),
                article_data.get('content_summary'),
                article_data.get('url'),
                article_data.get('source'),
                article_data.get('category', 'ai_news'),
                article_data.get('content_type', 'article'),
                article_data.get('significance_score', 5.0),
                article_data.get('published_date'),
                article_data.get('scraped_date'),
                article_data.get('llm_processed', False),
                True  # is_current_day
            )
            
            self.execute_query(insert_query, values, fetch_all=False)
            logger.info(f"✅ Article inserted: {article_data.get('title', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert article: {e}")
            logger.error(f"Article data: {article_data}")
            return False
    
    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Save user preferences according to functional requirements"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Insert or update user preferences
                    cursor.execute("""
                        INSERT INTO user_preferences (
                            user_id, experience_level, professional_roles, 
                            newsletter_frequency, email_notifications, breaking_news_alerts,
                            push_notifications, mobile_number, onboarding_completed, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (user_id) 
                        DO UPDATE SET
                            experience_level = EXCLUDED.experience_level,
                            professional_roles = EXCLUDED.professional_roles,
                            newsletter_frequency = EXCLUDED.newsletter_frequency,
                            email_notifications = EXCLUDED.email_notifications,
                            breaking_news_alerts = EXCLUDED.breaking_news_alerts,
                            push_notifications = EXCLUDED.push_notifications,
                            mobile_number = EXCLUDED.mobile_number,
                            onboarding_completed = EXCLUDED.onboarding_completed,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        user_id,
                        preferences.get('experience_level', 'beginner'),
                        preferences.get('user_roles', []),
                        preferences.get('newsletter_frequency', 'weekly'),
                        preferences.get('email_notifications', True),
                        preferences.get('breaking_news_alerts', False),
                        preferences.get('push_notifications', False),
                        preferences.get('mobile_number'),
                        preferences.get('onboarding_completed', True)
                    ))
                    
                    # Clear existing topic preferences
                    cursor.execute("DELETE FROM user_topic_preferences WHERE user_id = %s", (user_id,))
                    
                    # Insert selected topics
                    topics = preferences.get('topics', [])
                    for topic in topics:
                        if topic.get('selected', False):
                            cursor.execute("""
                                INSERT INTO user_topic_preferences (user_id, topic_id, selected)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (user_id, topic_id) DO UPDATE SET selected = EXCLUDED.selected
                            """, (user_id, topic['id'], True))
                    
                    # Clear existing content type preferences
                    cursor.execute("DELETE FROM user_content_type_preferences WHERE user_id = %s", (user_id,))
                    
                    # Insert selected content types
                    content_types = preferences.get('content_types', [])
                    for content_type in content_types:
                        # Get content type ID
                        cursor.execute("SELECT id FROM content_types WHERE name = %s", (content_type,))
                        ct_result = cursor.fetchone()
                        if ct_result:
                            cursor.execute("""
                                INSERT INTO user_content_type_preferences (user_id, content_type_id, selected)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (user_id, content_type_id) DO UPDATE SET selected = EXCLUDED.selected
                            """, (user_id, ct_result['id'], True))
                    
                    conn.commit()
                    logger.info(f"✅ User preferences saved for user: {user_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Failed to save user preferences: {e}")
            return False
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user preferences with topics and content types"""
        try:
            query = """
                SELECT 
                    up.*,
                    COALESCE(
                        JSON_AGG(
                            DISTINCT jsonb_build_object(
                                'id', at.id,
                                'name', at.name,
                                'category', at.category,
                                'selected', utp.selected
                            )
                        ) FILTER (WHERE at.id IS NOT NULL),
                        '[]'::json
                    ) as topics,
                    COALESCE(
                        JSON_AGG(
                            DISTINCT jsonb_build_object(
                                'id', ct.id,
                                'name', ct.name,
                                'display_name', ct.display_name,
                                'selected', uctp.selected
                            )
                        ) FILTER (WHERE ct.id IS NOT NULL),
                        '[]'::json
                    ) as content_types
                FROM user_preferences up
                LEFT JOIN user_topic_preferences utp ON up.user_id = utp.user_id
                LEFT JOIN ai_topics at ON utp.topic_id = at.id
                LEFT JOIN user_content_type_preferences uctp ON up.user_id = uctp.user_id
                LEFT JOIN content_types ct ON uctp.content_type_id = ct.id
                WHERE up.user_id = %s
                GROUP BY up.id
            """
            
            result = self.execute_query(query, (user_id,), fetch_one=True)
            
            if result:
                return dict(result)
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get user preferences: {e}")
            return None

    def close_connections(self):
        """Close all connections in the pool"""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()
            logger.info("🔌 PostgreSQL connection pool closed")

# Global database service instance
db_service = None

def get_database_service() -> PostgreSQLService:
    """Get global database service instance"""
    global db_service
    if db_service is None:
        db_service = PostgreSQLService()
    return db_service

def initialize_database():
    """Initialize the database service and perform migration if needed"""
    global db_service
    if db_service is None:
        db_service = PostgreSQLService()
        logger.info("✅ Database service initialized successfully")
    else:
        logger.info("ℹ️ Database service already initialized")
    return db_service


def close_database_service():
    """Close global database service"""
    global db_service
    if db_service:
        db_service.close_connections()
        db_service = None