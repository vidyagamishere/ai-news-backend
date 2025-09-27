#!/usr/bin/env python3
"""
PostgreSQL Database Service for AI News Scraper
Single database backend using psycopg2 with connection pooling and SQLite migration
"""

import os
import json
import logging
import traceback
import sqlite3
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
        """Initialize PostgreSQL connection pool and migrate from SQLite if needed"""
        self.database_url = os.getenv('POSTGRES_URL') or os.getenv('DATABASE_URL')
        self.sqlite_path = '/app/ai_news.db'  # Docker path to latest SQLite database
        
        if not self.database_url:
            raise ValueError("POSTGRES_URL or DATABASE_URL environment variable is required")
        
        logger.info(f"🐘 Initializing PostgreSQL service")
        logger.info(f"📊 Database URL configured: {self.database_url[:50]}...")
        logger.info(f"📁 SQLite migration source: {self.sqlite_path}")
        
        # Create connection pool
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20,  # min and max connections
                self.database_url,
                cursor_factory=RealDictCursor
            )
            logger.info("✅ PostgreSQL connection pool created successfully")
            
            # Initialize database schema and migrate data
            self.initialize_database()
            
            # Migrate data from SQLite if available
            self.migrate_from_sqlite()
            
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
                            verified_email BOOLEAN DEFAULT FALSE
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
                        ("idx_ai_sources_topic", "CREATE INDEX IF NOT EXISTS idx_ai_sources_topic ON ai_sources(ai_topic_id);")
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
    
    def migrate_from_sqlite(self):
        """Migrate data from SQLite to PostgreSQL"""
        logger.info("🔄 Starting SQLite to PostgreSQL migration...")
        
        # Check if SQLite database exists
        if not os.path.exists(self.sqlite_path):
            logger.info(f"📁 SQLite database not found at {self.sqlite_path}, skipping migration")
            return
        
        try:
            # Connect to SQLite database
            sqlite_conn = sqlite3.connect(self.sqlite_path)
            sqlite_conn.row_factory = sqlite3.Row
            sqlite_cursor = sqlite_conn.cursor()
            
            logger.info("✅ Connected to SQLite database for migration")
            
            # Get PostgreSQL connection
            with self.get_db_connection() as pg_conn:
                with pg_conn.cursor() as pg_cursor:
                    
                    # Check if migration already done
                    pg_cursor.execute("SELECT COUNT(*) FROM articles")
                    existing_articles = pg_cursor.fetchone()['count']
                    
                    if existing_articles > 0:
                        logger.info(f"📊 PostgreSQL already has {existing_articles} articles, skipping migration")
                        return
                    
                    # Migrate articles table
                    self.migrate_table(sqlite_cursor, pg_cursor, 'articles', [
                        'source', 'title', 'url', 'published_at', 'description', 'content',
                        'significance_score', 'scraped_at', 'category', 'reading_time',
                        'image_url', 'keywords'
                    ])
                    
                    # Migrate users table
                    self.migrate_table(sqlite_cursor, pg_cursor, 'users', [
                        'id', 'email', 'name', 'profile_image', 'subscription_tier',
                        'preferences', 'created_at', 'last_login_at', 'verified_email'
                    ])
                    
                    # Migrate user_passwords table if exists
                    try:
                        self.migrate_table(sqlite_cursor, pg_cursor, 'user_passwords', [
                            'user_id', 'password_hash', 'salt'
                        ])
                    except Exception as e:
                        logger.info(f"⚠️ Skipping user_passwords migration: {e}")
                    
                    # Migrate user_sessions table if exists
                    try:
                        self.migrate_table(sqlite_cursor, pg_cursor, 'user_sessions', [
                            'id', 'user_id', 'token_hash', 'created_at', 'expires_at', 'last_used_at'
                        ])
                    except Exception as e:
                        logger.info(f"⚠️ Skipping user_sessions migration: {e}")
                    
                    # Migrate daily_archives table if exists
                    try:
                        self.migrate_table(sqlite_cursor, pg_cursor, 'daily_archives', [
                            'archive_date', 'digest_data', 'article_count', 'created_at', 'metadata'
                        ])
                    except Exception as e:
                        logger.info(f"⚠️ Skipping daily_archives migration: {e}")
                    
                    # Commit all migrations
                    pg_conn.commit()
                    
                    logger.info("✅ SQLite to PostgreSQL migration completed successfully")
            
            sqlite_conn.close()
            
        except Exception as e:
            logger.error(f"❌ SQLite migration failed: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # Don't raise error - continue with empty PostgreSQL database
    
    def migrate_table(self, sqlite_cursor, pg_cursor, table_name, columns):
        """Migrate a specific table from SQLite to PostgreSQL"""
        logger.info(f"📦 Migrating table: {table_name}")
        
        try:
            # Check if table exists in SQLite
            sqlite_cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if sqlite_cursor.fetchone()[0] == 0:
                logger.info(f"⚠️ Table {table_name} not found in SQLite, skipping")
                return
            
            # Get data from SQLite
            sqlite_cursor.execute(f"SELECT * FROM {table_name}")
            rows = sqlite_cursor.fetchall()
            
            if not rows:
                logger.info(f"📊 No data in {table_name} table")
                return
            
            # Get actual columns from SQLite table
            sqlite_cursor.execute(f"PRAGMA table_info({table_name})")
            sqlite_columns_info = sqlite_cursor.fetchall()
            sqlite_columns = [col[1] for col in sqlite_columns_info]
            
            # Filter columns to only those that exist in both databases
            available_columns = [col for col in columns if col in sqlite_columns]
            
            if table_name in ['articles', 'daily_archives']:
                # Skip 'id' column for SERIAL columns
                available_columns = [col for col in available_columns if col != 'id']
            
            if not available_columns:
                logger.warning(f"⚠️ No compatible columns found for {table_name}")
                return
            
            # Prepare insert statement
            placeholders = ', '.join(['%s'] * len(available_columns))
            columns_str = ', '.join(available_columns)
            insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
            
            # Migrate each row
            migrated_count = 0
            for row in rows:
                row_dict = dict(row)
                
                # Extract values for available columns
                values = []
                for col in available_columns:
                    value = row_dict.get(col)
                    
                    # Handle JSON fields for PostgreSQL
                    if table_name in ['users', 'daily_archives'] and col in ['preferences', 'digest_data', 'metadata']:
                        if isinstance(value, str):
                            try:
                                # Parse JSON string and convert to dict
                                parsed_value = json.loads(value)
                                values.append(json.dumps(parsed_value))  # Store as JSON string for JSONB
                            except (json.JSONDecodeError, TypeError):
                                values.append('{}')  # Default empty JSON
                        else:
                            values.append(json.dumps(value) if value else '{}')
                    else:
                        values.append(value)
                
                # Insert row
                try:
                    pg_cursor.execute(insert_query, values)
                    migrated_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to migrate row in {table_name}: {e}")
                    continue
            
            logger.info(f"✅ Migrated {migrated_count} rows from {table_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate table {table_name}: {e}")
            raise e
    
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