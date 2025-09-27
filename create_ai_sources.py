#!/usr/bin/env python3
"""
Create ai_sources table and populate with essential AI news sources
For Railway PostgreSQL database
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ai_sources_table():
    """Create and populate ai_sources table in Railway PostgreSQL"""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        logger.error("psycopg2 not available. This script needs to run in Railway environment.")
        return False
    
    # Use Railway PostgreSQL URL
    postgres_url = os.getenv('POSTGRES_URL')
    if not postgres_url:
        logger.error("POSTGRES_URL environment variable not set")
        return False
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(postgres_url, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        logger.info("🐘 Connected to Railway PostgreSQL database")
        
        # Create ai_sources table
        logger.info("🏗️ Creating ai_sources table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_sources (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                rss_url TEXT NOT NULL,
                website TEXT,
                content_type VARCHAR(50) NOT NULL,
                category VARCHAR(100) DEFAULT 'general',
                enabled BOOLEAN DEFAULT TRUE,
                priority INTEGER DEFAULT 5,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Clear existing sources
        cursor.execute("DELETE FROM ai_sources;")
        
        # Insert essential AI sources
        logger.info("📚 Inserting AI news sources...")
        sources = [
            ("OpenAI Blog", "https://openai.com/blog/rss.xml", "https://openai.com", "blogs", "research", True, 1, "Official OpenAI blog and announcements"),
            ("Anthropic", "https://www.anthropic.com/news", "https://www.anthropic.com", "blogs", "research", True, 1, "Anthropic AI safety and research updates"),
            ("Google AI Blog", "https://ai.googleblog.com/feeds/posts/default", "https://ai.googleblog.com", "blogs", "research", True, 1, "Google AI research and developments"),
            ("DeepMind", "https://deepmind.google/discover/blog/", "https://deepmind.google", "blogs", "research", True, 1, "DeepMind research and breakthroughs"),
            ("Meta AI", "https://ai.meta.com/blog/", "https://ai.meta.com", "blogs", "research", True, 1, "Meta AI research and product updates"),
            ("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed/", "https://www.technologyreview.com", "blogs", "business", True, 2, "MIT Technology Review AI coverage"),
            ("VentureBeat AI", "https://venturebeat.com/ai/feed/", "https://venturebeat.com", "blogs", "business", True, 2, "AI business news and trends"),
            ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "https://techcrunch.com", "blogs", "business", True, 2, "AI startup and business news"),
            ("Towards Data Science", "https://towardsdatascience.com/feed", "https://towardsdatascience.com", "blogs", "technical", True, 3, "Data science and ML tutorials"),
            ("AI News", "https://www.artificialintelligence-news.com/feed/", "https://www.artificialintelligence-news.com", "blogs", "technical", True, 3, "AI industry news and analysis"),
        ]
        
        for source in sources:
            cursor.execute("""
                INSERT INTO ai_sources (name, rss_url, website, content_type, category, enabled, priority, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, source)
        
        # Create indexes
        logger.info("🔗 Creating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_sources_enabled ON ai_sources(enabled);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_sources_priority ON ai_sources(priority);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_sources_category ON ai_sources(category);")
        
        # Commit changes
        conn.commit()
        
        # Verify creation
        cursor.execute("SELECT COUNT(*) as count FROM ai_sources WHERE enabled = TRUE;")
        count = cursor.fetchone()['count']
        
        logger.info(f"✅ ai_sources table created successfully with {count} enabled sources")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create ai_sources table: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting ai_sources table creation for Railway PostgreSQL...")
    success = create_ai_sources_table()
    
    if success:
        logger.info("🎉 ai_sources table creation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ ai_sources table creation failed")
        sys.exit(1)