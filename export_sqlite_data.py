#!/usr/bin/env python3
"""
Export SQLite data to SQL format for Neon Postgres import
"""
import sqlite3
import json
import os
from datetime import datetime

SQLITE_PATH = '/Users/vijayansubramaniyan/Desktop/AI-ML/Projects/ai-news-scraper/ai_news.db'
OUTPUT_FILE = '/Users/vijayansubramaniyan/Desktop/AI-ML/Projects/ai-news-scraper/neon_migration.sql'

# Postgres table schemas
POSTGRES_SCHEMAS = {
    'articles': """
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            source TEXT,
            title TEXT,
            url TEXT UNIQUE,
            published_at TIMESTAMP,
            description TEXT,
            content TEXT,
            significance_score INTEGER DEFAULT 5,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            category TEXT DEFAULT 'general',
            reading_time INTEGER DEFAULT 3,
            image_url TEXT,
            keywords TEXT
        );
    """,
    'audio_content': """
        CREATE TABLE IF NOT EXISTS audio_content (
            id SERIAL PRIMARY KEY,
            source TEXT,
            title TEXT,
            url TEXT UNIQUE,
            published_at TIMESTAMP,
            description TEXT,
            content TEXT,
            significance_score INTEGER DEFAULT 5,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            category TEXT DEFAULT 'general',
            duration_minutes INTEGER,
            image_url TEXT,
            keywords TEXT,
            audio_url TEXT
        );
    """,
    'video_content': """
        CREATE TABLE IF NOT EXISTS video_content (
            id SERIAL PRIMARY KEY,
            source TEXT,
            title TEXT,
            url TEXT UNIQUE,
            published_at TIMESTAMP,
            description TEXT,
            content TEXT,
            significance_score INTEGER DEFAULT 5,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            category TEXT DEFAULT 'general',
            duration_minutes INTEGER,
            image_url TEXT,
            keywords TEXT,
            video_url TEXT,
            thumbnail_url TEXT,
            view_count INTEGER
        );
    """,
    'users': """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            profile_image TEXT,
            subscription_tier TEXT DEFAULT 'free',
            preferences JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login_at TIMESTAMP,
            email_verified BOOLEAN DEFAULT FALSE
        );
    """,
    'user_passwords': """
        CREATE TABLE IF NOT EXISTS user_passwords (
            user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        );
    """,
    'ai_topics': """
        CREATE TABLE IF NOT EXISTS ai_topics (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            category TEXT NOT NULL,
            is_active BOOLEAN DEFAULT TRUE
        );
    """,
    'user_sessions': """
        CREATE TABLE IF NOT EXISTS user_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            last_used_at TIMESTAMP
        );
    """,
    'daily_archives': """
        CREATE TABLE IF NOT EXISTS daily_archives (
            id SERIAL PRIMARY KEY,
            archive_date DATE UNIQUE,
            digest_data JSONB,
            article_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'
        );
    """
}

def escape_sql_string(value):
    """Escape string for SQL"""
    if value is None:
        return 'NULL'
    if isinstance(value, str):
        # Escape single quotes and backslashes
        escaped = value.replace("'", "''").replace("\\", "\\\\")
        return f"'{escaped}'"
    elif isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, dict):
        # Convert dict to JSON string
        json_str = json.dumps(value).replace("'", "''")
        return f"'{json_str}'"
    else:
        return f"'{str(value)}'"

def export_table_data(conn, table_name):
    """Export table data as INSERT statements"""
    print(f"📦 Exporting {table_name}...")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    if not rows:
        return [f"-- No data in {table_name}\\n"]
    
    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    columns = [col[1] for col in columns_info]
    
    sql_lines = [f"-- Data for {table_name} ({len(rows)} rows)"]
    
    for row in rows:
        values = []
        for i, value in enumerate(row):
            column_name = columns[i]
            
            # Handle special cases
            if table_name in ['articles', 'audio_content', 'video_content', 'daily_archives'] and column_name == 'id':
                continue  # Skip auto-increment IDs
            
            # Convert JSON fields
            if table_name in ['users', 'daily_archives'] and column_name in ['preferences', 'digest_data', 'metadata']:
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        values.append(f"'{json.dumps(parsed)}'::jsonb")
                    except:
                        values.append("'{}'::jsonb")
                else:
                    values.append(escape_sql_string(value))
            else:
                values.append(escape_sql_string(value))
        
        # Filter columns for tables with SERIAL primary keys
        if table_name in ['articles', 'audio_content', 'video_content', 'daily_archives']:
            filtered_columns = [col for col in columns if col != 'id']
            insert_columns = ', '.join(filtered_columns)
        else:
            filtered_columns = columns
            insert_columns = ', '.join(columns)
        
        if values:  # Only if we have values after filtering
            values_str = ', '.join(values)
            sql_lines.append(f"INSERT INTO {table_name} ({insert_columns}) VALUES ({values_str}) ON CONFLICT DO NOTHING;")
    
    sql_lines.append("")  # Empty line
    return sql_lines

def create_new_ai_topics():
    """Create INSERT statements for new AI topics"""
    topics = [
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
    
    sql_lines = [
        "-- Clear and update AI topics with new categories",
        "DELETE FROM ai_topics;",
        ""
    ]
    
    for topic_id, name, description, category in topics:
        sql_lines.append(f"INSERT INTO ai_topics (id, name, description, category, is_active) VALUES ('{topic_id}', '{name}', '{description}', '{category}', TRUE);")
    
    sql_lines.append("")
    return sql_lines

def main():
    """Export SQLite data to SQL file"""
    print("🚀 Exporting SQLite data to SQL for Neon Postgres...")
    
    if not os.path.exists(SQLITE_PATH):
        print(f"❌ SQLite database not found: {SQLITE_PATH}")
        return False
    
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"-- Neon Postgres Migration Script\\n")
            f.write(f"-- Generated: {datetime.now().isoformat()}\\n")
            f.write(f"-- Source: {SQLITE_PATH}\\n\\n")
            
            # Write table schemas
            f.write("-- ============================================\\n")
            f.write("-- CREATE TABLES\\n")
            f.write("-- ============================================\\n\\n")
            
            for table_name, schema in POSTGRES_SCHEMAS.items():
                f.write(f"-- Table: {table_name}\\n")
                f.write(schema)
                f.write("\\n")
            
            # Get tables to export
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Write data
            f.write("-- ============================================\\n")
            f.write("-- INSERT DATA\\n")
            f.write("-- ============================================\\n\\n")
            
            for table in tables:
                if table in POSTGRES_SCHEMAS:
                    sql_lines = export_table_data(conn, table)
                    for line in sql_lines:
                        f.write(line + "\\n")
            
            # Write new AI topics
            f.write("-- ============================================\\n")
            f.write("-- UPDATE AI TOPICS WITH NEW CATEGORIES\\n")
            f.write("-- ============================================\\n\\n")
            
            ai_topic_lines = create_new_ai_topics()
            for line in ai_topic_lines:
                f.write(line + "\\n")
            
            f.write("-- Migration complete\\n")
        
        conn.close()
        
        print(f"✅ Export completed successfully!")
        print(f"📄 SQL file: {OUTPUT_FILE}")
        print(f"💡 You can now run this SQL file against your Neon Postgres database")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)