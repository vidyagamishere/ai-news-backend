#!/usr/bin/env python3
"""
PostgreSQL Database Service for AI News Scraper
Single database backend using psycopg2 with connection pooling
"""

import os
import json
import logging
import traceback
from datetime import datetime, timedelta, timezone
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
        import os
        DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        
        if DEBUG:
            logger.debug(f"🔍 Executing query: {query[:200]}{'...' if len(query) > 200 else ''}")
            if params:
                logger.debug(f"🔍 Query parameters: {params}")
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    
                    if fetch_one:
                        result = cursor.fetchone()
                        if DEBUG:
                            logger.debug(f"🔍 Query returned one result: {result}")
                        return result
                    elif fetch_all:
                        results = cursor.fetchall()
                        if DEBUG:
                            logger.debug(f"🔍 Query returned {len(results)} results")
                        return results
                    else:
                        rowcount = cursor.rowcount
                        if DEBUG:
                            logger.debug(f"🔍 Query affected {rowcount} rows")
                        return rowcount
                        
        except Exception as e:
            logger.error(f"❌ Query execution failed: {str(e)}")
            if DEBUG:
                logger.debug(f"🔍 Failed query: {query}")
                logger.debug(f"🔍 Failed params: {params}")
            raise e
    
    def initialize_database(self):
        """Initialize PostgreSQL database schema - COMMENTED OUT (handled manually)"""
        logger.info("🏗️ Database schema initialization skipped - handled manually by admin")
        logger.info("✅ PostgreSQL database schema initialized successfully")
        # All DDL operations (CREATE TABLE, ALTER TABLE, CREATE INDEX, CREATE VIEW) are handled manually
    
    def create_database_views(self, cursor):
        """Create optimized database views for content delivery - COMMENTED OUT (handled manually)"""
        logger.info("📊 Database view creation skipped - handled manually by admin")
        logger.info("✅ Database views creation process completed")
        # All view creation handled manually
    
    def populate_content_types(self, cursor):
        """Populate content_types table with master data - COMMENTED OUT (handled manually)"""
        logger.info("📋 Content types population skipped - handled manually by admin")
        logger.info("📊 Found existing content types, skipping population")
        # All data population handled manually
    
    def populate_ai_topics(self, cursor):
        """Populate ai_topics table with comprehensive AI topics - COMMENTED OUT (handled manually)"""
        logger.info("📋 AI topics population skipped - handled manually by admin")
        logger.info("📊 Found existing AI topics, skipping population")
        # All data population handled manually
    
    def populate_ai_sources(self, cursor):
        """Populate ai_sources table with comprehensive AI news sources - COMMENTED OUT (handled manually)"""
        logger.info("📊 AI sources population skipped - handled manually by admin")
        logger.info("📊 Found existing AI sources, skipping population")
        # All data population handled manually
    
    def get_ai_sources(self) -> List[Dict[str, Any]]:
        """Get all AI sources for scraping"""
        try:
            query = """
                SELECT s.id, s.publisher_id, s.name, s.rss_url, s.website, COALESCE(c.name, 'general') as category, s.priority, s.enabled  
                FROM ai_sources s LEFT JOIN ai_categories_master c ON s.category_id = c.id WHERE s.enabled = TRUE and s.is_active = TRUE
                ORDER BY s.priority DESC, s.name
            """
            sources = self.execute_query(query, fetch_all=True)
            
            if not sources:
                # Return default sources if none exist
                logger.warning("⚠️ No AI sources found in database, using defaults")
                return [
                    {'id': 1, 'publisher_id': 1, 'name': 'OpenAI Blog', 'url': 'https://openai.com/blog/', 'website': 'https://openai.com/blog/', 'category': 'company', 'priority': 10, 'is_active': True},
                    {'id': 2, 'publisher_id': 2, 'name': 'Google AI Blog', 'url': 'https://ai.googleblog.com/', 'website': 'https://ai.googleblog.com/', 'category': 'company', 'priority': 10, 'is_active': True},
                    {'id': 3, 'publisher_id': 3, 'name': 'Anthropic News', 'url': 'https://www.anthropic.com/news', 'website': 'https://www.anthropic.com/news', 'category': 'company', 'priority': 9, 'is_active': True},
                    {'id': 4, 'publisher_id': 4, 'name': 'DeepMind Blog', 'url': 'https://deepmind.google/discover/blog/', 'website': 'https://deepmind.google/discover/blog/', 'category': 'research', 'priority': 9, 'is_active': True},
                ]
            
            return [dict(source) for source in sources]
            
        except Exception as e:
            logger.error(f"❌ Failed to get AI sources: {e}")
            return []
    
    def insert_article(self, article_data: Dict[str, Any]) -> bool:
        # Insert scraped article into articles table after mapping LLM labels to integer foreign keys (content_type_id, ai_topic_id).
        # Define fallback IDs in case the LLM returns an invalid label
        # These should correspond to 'Articles/Blogs' and 'AI News & Updates' in your master tables.
        FALLBACK_CONTENT_TYPE_ID = 1  
        FALLBACK_TOPIC_ID = 21       

        try:
            url = article_data.get('url')
            if not url:
                logger.error("❌ Article data missing URL.")
                return False

            # --- 1. Check if article already exists ---
            existing_query = "SELECT id FROM articles WHERE url = %s AND scraped_date > %s"
            eight_hours_ago = datetime.now(timezone.utc) - timedelta(hours=8)
            existing = self.execute_query(existing_query, (url, eight_hours_ago), fetch_one=True)

            if existing:
                logger.info(f"📄 Article already exists: {article_data.get('title', 'Unknown')}")
                return False
            
            # --- 2. Get Content Type ID (Label-to-ID Mapping) ---
            final_content_type_id = FALLBACK_CONTENT_TYPE_ID
            content_type_label = article_data.get('content_type_label')
            
            if content_type_label:
                # Try both display_name and name for flexible matching  
                content_type_query = "SELECT id FROM content_types WHERE display_name = %s OR name = %s OR LOWER(display_name) = LOWER(%s)"
                content_type_result = self.execute_query(content_type_query, (content_type_label, content_type_label.lower(), content_type_label), fetch_one=True)
                
                if content_type_result:
                    # Safely extract the ID from the result set (which could be a dict or a tuple)
                    final_content_type_id = content_type_result['id'] if isinstance(content_type_result, dict) else content_type_result[0]
                    logger.debug(f"✅ Found content type '{content_type_label}' with ID: {final_content_type_id}")
                else:
                    logger.warning(f"⚠️ Content type '{content_type_label}' not found, using default ID: {FALLBACK_CONTENT_TYPE_ID}")

            
            # --- 3. Get Category ID with Hybrid Fallback (Claude AI > RSS Source > Hardcoded) ---
            topic_category_label = article_data.get('topic_category_label')  # Claude AI category
            source_category = article_data.get('source_category')  # RSS source category
            
            final_ai_topic_id = FALLBACK_TOPIC_ID
            category_source = "hardcoded_fallback"
            
            # Priority 1: Try Claude AI category first
            if topic_category_label:
                topic_query = "SELECT id FROM ai_categories_master WHERE name = %s"
                ai_topic_result = self.execute_query(topic_query, (topic_category_label,), fetch_one=True)
                
                if ai_topic_result:
                    final_ai_topic_id = ai_topic_result['id'] if isinstance(ai_topic_result, dict) else ai_topic_result[0]
                    category_source = f"claude_ai({topic_category_label})"
                    logger.debug(f"✅ Found Claude AI category '{topic_category_label}' with ID: {final_ai_topic_id}")
                else:
                    logger.warning(f"⚠️ Claude AI category '{topic_category_label}' not found, trying RSS source fallback...")
            
            # Priority 2: If Claude AI failed, try RSS source category
            if category_source == "hardcoded_fallback" and source_category and source_category != 'general':
                source_topic_query = "SELECT id FROM ai_categories_master WHERE name = %s OR LOWER(name) = LOWER(%s)"
                source_topic_result = self.execute_query(source_topic_query, (source_category, source_category), fetch_one=True)
                
                if source_topic_result:
                    final_ai_topic_id = source_topic_result['id'] if isinstance(source_topic_result, dict) else source_topic_result[0]
                    category_source = f"rss_source({source_category})"
                    logger.info(f"✅ Using RSS source category '{source_category}' with ID: {final_ai_topic_id}")
                else:
                    logger.warning(f"⚠️ RSS source category '{source_category}' not found, using hardcoded fallback ID: {FALLBACK_TOPIC_ID}")
            
            # Priority 3: Final fallback to hardcoded ID (already set above)
            if category_source == "hardcoded_fallback":
                logger.info(f"📋 Using hardcoded fallback category ID: {FALLBACK_TOPIC_ID}")
            
            logger.info(f"🏷️ Final category assignment: ID {final_ai_topic_id} from {category_source}")

            #if the rss feed  already exists, keep the date as is
            if article_data.get('published_date') is None:
                article_data['published_date'] = datetime.now(timezone.utc) - timedelta(days=1)
            if article_data.get('scraped_date') is None:
                article_data['scraped_date'] = datetime.now(timezone.utc)
            if article_data.get('created_date') is None:
                article_data['created_date'] = datetime.now(timezone.utc)
            
            
            # --- 4. Insert New Article (Fixed Query) ---
            insert_query = """
                INSERT INTO articles (
                    content_hash, title, summary, url, source, significance_score, published_date, scraped_date, 
                    llm_processed, content_type_id, category_id, reading_time, author, complexity_level, updated_date, created_date, keywords, publisher_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) ON CONFLICT (url) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    source = EXCLUDED.source,
                    significance_score = EXCLUDED.significance_score,
                    published_date = EXCLUDED.published_date,
                    scraped_date = EXCLUDED.scraped_date,
                    llm_processed = EXCLUDED.llm_processed,
                    content_type_id = EXCLUDED.content_type_id,
                    category_id = EXCLUDED.category_id,
                    reading_time = EXCLUDED.reading_time,
                    author = EXCLUDED.author,
                    complexity_level = EXCLUDED.complexity_level,
                    updated_date = EXCLUDED.updated_date,
                    created_date = EXCLUDED.created_date,
                    keywords = EXCLUDED.keywords,
                    publisher_id = EXCLUDED.publisher_id
            """ # 18 columns, 18 placeholders (%s)
            
            values = (
                article_data.get('content_hash'),
                article_data.get('title'),  # Using 'title' from LLM output
                article_data.get('summary'),   # Using 'summary' from LLM output
                url,
                article_data.get('source'),
                article_data.get('significance_score'),
                article_data.get('published_date'),     # Using 'published_date' from LLM output
                article_data.get('scraped_date'),
                True,  # Assuming if we reached here, LLM processed is True
                final_content_type_id,
                final_ai_topic_id,
                article_data.get('reading_time', 1),
                article_data.get('author', 'Unknown'),
                article_data.get('complexity_level', 'Medium'),
                article_data.get('updated_date', datetime.now(timezone.utc)),
                article_data.get('created_date', datetime.now(timezone.utc)),
                article_data.get('keywords', ['generative AI']),
                article_data.get('publisher_id')  # Include publisher_id from ai_sources mapping
            )
            
            # Debug logging for publisher_id and category
            publisher_id_val = article_data.get('publisher_id')
            if publisher_id_val:
                logger.info(f"📎 Including publisher_id {publisher_id_val} for article: {article_data.get('title', url)[:50]}...")
            else:
                logger.warning(f"⚠️ No publisher_id provided for article: {article_data.get('title', url)[:50]}...")
            
            logger.info(f"📂 Category assignment summary for {article_data.get('title', url)[:50]}:")
            logger.info(f"   • Claude AI category: '{topic_category_label}' → {'✅ Found' if topic_category_label and category_source.startswith('claude_ai') else '❌ Not found/used'}")
            logger.info(f"   • RSS source category: '{source_category}' → {'✅ Used as fallback' if category_source.startswith('rss_source') else '❌ Not used'}")
            logger.info(f"   • Final category_id: {final_ai_topic_id} (from {category_source})")
            
            self.execute_query(insert_query, values, fetch_all=False)
            logger.info(f"✅ Article inserted with publisher_id {publisher_id_val}: {article_data.get('title', url)[:50]}...")
            return True
                
        except Exception as e:
            logger.error(f"❌ Failed to insert article: {e}")
            logger.error(f"Article data (pre-insert): {article_data}")
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

# Backward compatibility alias
DatabaseService = PostgreSQLService