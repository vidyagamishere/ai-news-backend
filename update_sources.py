#!/usr/bin/env python3
"""
Update AI sources with correct RSS feeds for proper scraping
"""

import os
import sys
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_ai_sources():
    """Update ai_sources table with working RSS feeds"""
    try:
        # Use Railway PostgreSQL URL
        postgres_url = os.getenv('POSTGRES_URL')
        if not postgres_url:
            logger.error("POSTGRES_URL environment variable not set")
            return False
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(postgres_url, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        logger.info("🐘 Connected to Railway PostgreSQL database")
        
        # Clear existing sources and add working ones
        logger.info("🧹 Clearing existing sources...")
        cursor.execute("DELETE FROM ai_sources;")
        
        # Insert working AI sources with correct RSS feeds
        logger.info("📚 Inserting working AI news sources...")
        sources = [
            # Verified high-priority research sources
            ("OpenAI Blog", "https://openai.com/blog/rss.xml", "https://openai.com", "blogs", "research", True, 1, "Official OpenAI blog and announcements"),
            ("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed/", "https://www.technologyreview.com", "blogs", "research", True, 1, "MIT Technology Review AI coverage"),
            ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "https://techcrunch.com", "blogs", "business", True, 1, "AI startup and business news"),
            ("Towards Data Science", "https://towardsdatascience.com/feed", "https://towardsdatascience.com", "blogs", "technical", True, 2, "Data science and ML tutorials"),
            ("Machine Learning Mastery", "https://machinelearningmastery.com/feed/", "https://machinelearningmastery.com", "blogs", "education", True, 2, "ML tutorials and guides"),
            ("Anthropic Blog", "https://www.anthropic.com/news/rss", "https://www.anthropic.com", "blogs", "research", True, 1, "Anthropic AI safety and research updates"),
            
            # Alternative verified sources
            ("AI News", "https://www.artificialintelligence-news.com/feed/", "https://www.artificialintelligence-news.com", "blogs", "technical", True, 2, "AI industry news and analysis"),
            ("Analytics Vidhya", "https://www.analyticsvidhya.com/feed/", "https://www.analyticsvidhya.com", "blogs", "technical", True, 2, "Data science and AI tutorials"),
            ("KDnuggets", "https://www.kdnuggets.com/feed", "https://www.kdnuggets.com", "blogs", "technical", True, 2, "Data science and machine learning news"),
            ("The Batch by DeepLearning.AI", "https://www.deeplearning.ai/the-batch/feed/", "https://www.deeplearning.ai", "blogs", "education", True, 2, "Weekly AI news and insights"),
            
            # Podcast sources
            ("Lex Fridman Podcast", "https://lexfridman.com/feed/podcast/", "https://lexfridman.com", "podcasts", "education", True, 1, "Long-form conversations about AI"),
            ("AI Podcast by NVIDIA", "https://feeds.soundcloud.com/users/soundcloud:users:264034133/sounds.rss", "https://blogs.nvidia.com", "podcasts", "technical", True, 2, "NVIDIA AI discussions"),
            ("The TWIML AI Podcast", "https://twimlai.com/feed/podcast/", "https://twimlai.com", "podcasts", "technical", True, 2, "This Week in Machine Learning and AI"),
            
            # Video sources (YouTube RSS)
            ("Two Minute Papers", "https://www.youtube.com/feeds/videos.xml?channel_id=UCbfYPyITQ-7l4upoX8nvctg", "https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg", "videos", "education", True, 1, "AI research paper explanations"),
            ("AI Explained", "https://www.youtube.com/feeds/videos.xml?channel_id=UCNJ1Ymd5yFuUPtn21xtRbbw", "https://www.youtube.com/channel/UCNJ1Ymd5yFuUPtn21xtRbbw", "videos", "education", True, 2, "AI concepts explained"),
            ("Yannic Kilcher", "https://www.youtube.com/feeds/videos.xml?channel_id=UCZHmQk67mSJgfCCTn7xBfew", "https://www.youtube.com/channel/UCZHmQk67mSJgfCCTn7xBfew", "videos", "research", True, 2, "AI research and paper discussions"),
            ("3Blue1Brown", "https://www.youtube.com/feeds/videos.xml?channel_id=UCYO_jab_esuFRV4b17AJtAw", "https://www.youtube.com/c/3blue1brown", "videos", "education", True, 1, "Mathematical explanations of ML concepts"),
            
            # Learning resources
            ("Distill", "https://distill.pub/rss.xml", "https://distill.pub", "learning", "research", True, 1, "Visual explanations of ML concepts"),
            ("Papers With Code Blog", "https://paperswithcode.com/feed.xml", "https://paperswithcode.com", "learning", "research", True, 2, "Latest ML papers and implementations"),
            ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml", "https://huggingface.co", "learning", "technical", True, 2, "Hugging Face ML and AI tutorials"),
            
            # Events and demos
            ("AI Research Events", "https://airesearch.com/feed/", "https://airesearch.com", "events", "research", True, 3, "AI conference and event updates"),
            ("ML Conferences", "https://mlconf.com/feed/", "https://mlconf.com", "events", "business", True, 3, "Machine learning conference news"),
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_sources_content_type ON ai_sources(content_type);")
        
        # Commit changes
        conn.commit()
        
        # Verify creation
        cursor.execute("SELECT COUNT(*) as count FROM ai_sources WHERE enabled = TRUE;")
        count = cursor.fetchone()['count']
        
        logger.info(f"✅ ai_sources table updated successfully with {count} enabled sources")
        
        # Show content type distribution
        cursor.execute("SELECT content_type, COUNT(*) as count FROM ai_sources GROUP BY content_type ORDER BY count DESC;")
        content_types = cursor.fetchall()
        
        logger.info("📊 Content type distribution:")
        for ct in content_types:
            logger.info(f"  - {ct['content_type']}: {ct['count']} sources")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update ai_sources table: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting ai_sources table update with working RSS feeds...")
    success = update_ai_sources()
    
    if success:
        logger.info("🎉 ai_sources table update completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ ai_sources table update failed")
        sys.exit(1)