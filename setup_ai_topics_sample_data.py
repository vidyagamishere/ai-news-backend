#!/usr/bin/env python3
"""
Setup AI Topics Sample Data
Add sample topics to the ai_topics table for foreign key relationships
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor

# PostgreSQL connection
POSTGRES_URL = os.getenv('POSTGRES_URL', "postgresql://postgres:FgvftzrGueiGipLiRRMKMElppasuzBjptZlwPL@autorack.proxy.rlwy.net:51308/railway")

def setup_sample_ai_topics():
    """Add sample AI topics data to the database"""
    try:
        conn = psycopg2.connect(POSTGRES_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        # Sample AI topics data
        sample_topics = [
            {
                'category': 'research', 
                'description': 'AI Research and Academic Papers',
                'keywords': 'research, academic, papers, studies, experiments',
                'priority': 1,
                'enabled': True
            },
            {
                'category': 'business', 
                'description': 'AI in Business and Industry',
                'keywords': 'business, industry, enterprise, commercial, market',
                'priority': 2,
                'enabled': True
            },
            {
                'category': 'technical', 
                'description': 'Technical AI Development and Tools',
                'keywords': 'development, programming, tools, frameworks, libraries',
                'priority': 3,
                'enabled': True
            },
            {
                'category': 'education', 
                'description': 'AI Education and Learning Resources',
                'keywords': 'education, learning, courses, tutorials, training',
                'priority': 4,
                'enabled': True
            },
            {
                'category': 'platform', 
                'description': 'AI Platforms and Services',
                'keywords': 'platform, service, cloud, API, infrastructure',
                'priority': 5,
                'enabled': True
            },
            {
                'category': 'robotics', 
                'description': 'Robotics and Automation',
                'keywords': 'robotics, automation, robots, mechanical, hardware',
                'priority': 6,
                'enabled': True
            },
            {
                'category': 'healthcare', 
                'description': 'AI in Healthcare and Medicine',
                'keywords': 'healthcare, medicine, medical, health, diagnosis',
                'priority': 7,
                'enabled': True
            },
            {
                'category': 'automotive', 
                'description': 'AI in Automotive and Transportation',
                'keywords': 'automotive, transportation, vehicles, autonomous, driving',
                'priority': 8,
                'enabled': True
            }
        ]
        
        # Check if ai_topics table exists, create if not
        cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'ai_topics'
        );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("Creating ai_topics table...")
            cursor.execute("""
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
            """)
            conn.commit()
            print("✅ Created ai_topics table")
        else:
            print("ℹ️ ai_topics table already exists")
        
        # Insert sample topics (skip if already exist)
        for topic in sample_topics:
            try:
                cursor.execute("""
                INSERT INTO ai_topics (category, description, keywords, priority, enabled)
                VALUES (%(category)s, %(description)s, %(keywords)s, %(priority)s, %(enabled)s)
                ON CONFLICT (category) DO NOTHING;
                """, topic)
                
                if cursor.rowcount > 0:
                    print(f"✅ Added topic: {topic['category']}")
                else:
                    print(f"ℹ️ Topic already exists: {topic['category']}")
                    
            except Exception as e:
                print(f"⚠️ Error adding topic {topic['category']}: {str(e)}")
        
        conn.commit()
        
        # Show current topics
        cursor.execute("SELECT id, category, description FROM ai_topics ORDER BY priority;")
        topics = cursor.fetchall()
        
        print(f"\n📊 Current AI Topics ({len(topics)} total):")
        for topic in topics:
            print(f"  {topic['id']:2d}. {topic['category']:<12} - {topic['description']}")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 AI topics sample data setup completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up AI topics: {str(e)}")
        return False

if __name__ == '__main__':
    print("🔧 Setting up AI Topics Sample Data...")
    print(f"🔗 Database: {POSTGRES_URL.split('@')[1] if '@' in POSTGRES_URL else 'Unknown'}")
    print()
    
    success = setup_sample_ai_topics()
    
    if success:
        print("\n💡 You can now:")
        print("  • Run the admin interface: python run_admin.py")
        print("  • View ai_sources table to see new foreign key columns")
        print("  • Add records with ai_topics_id references")
        print("  • Use category dropdown with ai_topics categories")
    else:
        print("\n⚠️ Setup failed. Check your database connection.")