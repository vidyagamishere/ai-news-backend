#!/usr/bin/env python3
"""
Simple template script to add custom AI sources
Customize the sources list below and run this script
"""

import psycopg2
from psycopg2.extras import RealDictCursor

# Railway PostgreSQL connection
POSTGRES_URL = "postgresql://postgres:FgvftzrGueiGipLiRRMKMElppasuzBjptZlwPL@autorack.proxy.rlwy.net:51308/railway"

def add_sources():
    """Add your custom sources here"""
    
    # CUSTOMIZE THIS LIST - Add your own AI sources
    # Format: (name, rss_url, website, content_type, category, enabled, priority, description)
    custom_sources = [
        # Example entries - replace with your sources
        ("Your AI Blog", "https://your-site.com/rss.xml", "https://your-site.com", "blogs", "research", True, 2, "Description of your source"),
        ("Your Podcast", "https://your-podcast.com/feed.xml", "https://your-podcast.com", "podcasts", "education", True, 2, "Your podcast description"),
        
        # Content types: blogs, podcasts, videos, learning, demos, events
        # Categories: research, business, technical, education, platform
        # Priority: 1 (highest) to 5 (lowest)
        # Enabled: True/False
    ]
    
    try:
        # Connect to database
        conn = psycopg2.connect(POSTGRES_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        print(f"Adding {len(custom_sources)} custom sources...")
        
        for source in custom_sources:
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
            print(f"✅ Added: {source[0]}")
        
        conn.commit()
        
        # Show total count
        cursor.execute("SELECT COUNT(*) as count FROM ai_sources WHERE enabled = TRUE;")
        count = cursor.fetchone()['count']
        print(f"🎉 Success! Total enabled sources: {count}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    add_sources()