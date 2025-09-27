#!/usr/bin/env python3
"""
External script to add new AI resources to Railway PostgreSQL database
Run this script from anywhere to add new RSS sources for scraping
"""

import os
import sys
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Railway PostgreSQL connection URL
POSTGRES_URL = "postgresql://postgres:FgvftzrGueiGipLiRRMKMElppasuzBjptZlwPL@autorack.proxy.rlwy.net:51308/railway"

def connect_to_database():
    """Connect to Railway PostgreSQL database"""
    try:
        conn = psycopg2.connect(POSTGRES_URL, cursor_factory=RealDictCursor)
        logger.info("✅ Connected to Railway PostgreSQL database")
        return conn
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {str(e)}")
        return None

def add_ai_sources(new_sources):
    """Add new AI sources to the database"""
    conn = connect_to_database()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        logger.info(f"📚 Adding {len(new_sources)} new AI sources...")
        
        for source in new_sources:
            cursor.execute("""
                INSERT INTO ai_sources (name, rss_url, website, content_type, category, enabled, priority, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (rss_url) DO UPDATE SET
                    name = EXCLUDED.name,
                    website = EXCLUDED.website,
                    content_type = EXCLUDED.content_type,
                    category = EXCLUDED.category,
                    enabled = EXCLUDED.enabled,
                    priority = EXCLUDED.priority,
                    description = EXCLUDED.description
            """, source)
        
        conn.commit()
        
        # Verify addition
        cursor.execute("SELECT COUNT(*) as count FROM ai_sources WHERE enabled = TRUE;")
        count = cursor.fetchone()['count']
        
        logger.info(f"✅ Successfully added sources. Total enabled sources: {count}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to add sources: {str(e)}")
        if conn:
            conn.rollback()
            conn.close()
        return False

def view_current_sources():
    """View current sources in the database"""
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name, content_type, category, enabled, priority, rss_url
            FROM ai_sources 
            ORDER BY priority, content_type, name
        """)
        sources = cursor.fetchall()
        
        logger.info(f"📊 Current sources in database ({len(sources)} total):")
        logger.info("-" * 80)
        
        for source in sources:
            status = "✅" if source['enabled'] else "❌"
            logger.info(f"{status} [{source['content_type']}/{source['category']}] {source['name']} (Priority: {source['priority']})")
            logger.info(f"   URL: {source['rss_url']}")
            logger.info("")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to view sources: {str(e)}")
        if conn:
            conn.close()

def main():
    """Main function - customize this with your new sources"""
    
    # Example: Add new AI sources here
    # Format: (name, rss_url, website, content_type, category, enabled, priority, description)
    new_sources = [
        # Add your new sources here - examples:
        
        # Research sources
        ("Anthropic Blog", "https://www.anthropic.com/news/rss", "https://www.anthropic.com", "blogs", "research", True, 1, "Anthropic AI safety and research updates"),
        ("Meta AI Blog", "https://ai.meta.com/blog/rss/", "https://ai.meta.com", "blogs", "research", True, 1, "Meta AI research and development"),
        ("DeepMind Blog", "https://deepmind.google/discover/blog/rss.xml", "https://deepmind.google", "blogs", "research", True, 1, "Google DeepMind research blog"),
        
        # Business sources
        ("AI Business", "https://aibusiness.com/rss.xml", "https://aibusiness.com", "blogs", "business", True, 2, "AI business news and analysis"),
        ("Forbes AI", "https://www.forbes.com/ai/feed/", "https://www.forbes.com", "blogs", "business", True, 2, "Forbes AI business coverage"),
        
        # Technical sources
        ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml", "https://huggingface.co", "blogs", "technical", True, 2, "Hugging Face ML and AI tutorials"),
        ("PyTorch Blog", "https://pytorch.org/blog/feed.xml", "https://pytorch.org", "blogs", "technical", True, 2, "PyTorch framework updates and tutorials"),
        
        # Educational content
        ("Coursera AI Blog", "https://blog.coursera.org/feed/", "https://blog.coursera.org", "learning", "education", True, 3, "Online AI and ML education content"),
        ("Andrew Ng Blog", "https://www.andrewng.org/feed/", "https://www.andrewng.org", "learning", "education", True, 1, "Andrew Ng's insights on AI education"),
        
        # Podcasts
        ("The AI Podcast", "https://blogs.nvidia.com/ai-podcast/feed/", "https://blogs.nvidia.com", "podcasts", "technical", True, 2, "NVIDIA's AI podcast series"),
        ("Machine Learning Street Talk", "https://anchor.fm/s/443ced78/podcast/rss", "https://anchor.fm/machine-learning-street-talk", "podcasts", "technical", True, 2, "Technical ML discussions and interviews"),
        
        # Video content
        ("3Blue1Brown", "https://www.youtube.com/feeds/videos.xml?channel_id=UCYO_jab_esuFRV4b17AJtAw", "https://www.youtube.com/c/3blue1brown", "videos", "education", True, 1, "Mathematical explanations of ML concepts"),
        ("Computerphile", "https://www.youtube.com/feeds/videos.xml?channel_id=UC9-y-6csu5WGm29I7JiwpnA", "https://www.youtube.com/c/Computerphile", "videos", "education", True, 2, "Computer science and AI explanations"),
        
        # Events and conferences
        ("NeurIPS", "https://nips.cc/feed.xml", "https://neurips.cc", "events", "research", True, 1, "Neural Information Processing Systems conference"),
        ("ICML Blog", "https://icml.cc/feed/", "https://icml.cc", "events", "research", True, 1, "International Conference on Machine Learning"),
    ]
    
    logger.info("🚀 AI Resources Database Manager")
    logger.info("=" * 50)
    
    # Show current sources
    logger.info("📋 Current sources in database:")
    view_current_sources()
    
    # Add new sources if any are defined
    if new_sources:
        logger.info(f"➕ Adding {len(new_sources)} new sources...")
        success = add_ai_sources(new_sources)
        
        if success:
            logger.info("🎉 New sources added successfully!")
            logger.info("📋 Updated source list:")
            view_current_sources()
        else:
            logger.error("❌ Failed to add new sources")
    else:
        logger.info("ℹ️ No new sources to add. Customize the new_sources list in main() to add sources.")
    
    logger.info("✅ Script completed!")

if __name__ == "__main__":
    main()