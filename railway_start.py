#!/usr/bin/env python3
"""
Railway deployment starter script for AI News Scraper
Forces the use of clean_main.py with PostgreSQL-only backend
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_correct_main_file():
    """Ensure clean_main.py is used as the main application file"""
    try:
        # Check if clean_main.py exists
        if not os.path.exists('clean_main.py'):
            logger.error("❌ clean_main.py not found!")
            sys.exit(1)
        
        logger.info("✅ clean_main.py found - using COMPREHENSIVE modular PostgreSQL backend")
        
        # Check file content to verify we have the comprehensive version
        with open('clean_main.py', 'r') as f:
            content = f.read()
            
        # Check for key indicators of comprehensive version
        has_content_types = '/content-types' in content
        has_multimedia = '/multimedia/audio' in content
        has_comprehensive_version = '4.0.0-complete-modular-postgresql' in content
        has_topics_endpoint = '@app.get("/topics")' in content
        has_auth_profile = '/auth/profile' in content
        
        logger.info(f"📊 Comprehensive version indicators:")
        logger.info(f"  - Content-types endpoint: {'✅' if has_content_types else '❌'}")
        logger.info(f"  - Multimedia endpoints: {'✅' if has_multimedia else '❌'}")
        logger.info(f"  - Topics endpoint: {'✅' if has_topics_endpoint else '❌'}")
        logger.info(f"  - Auth profile endpoint: {'✅' if has_auth_profile else '❌'}")
        logger.info(f"  - Comprehensive version string: {'✅' if has_comprehensive_version else '❌'}")
        
        if not (has_content_types and has_multimedia and has_comprehensive_version and has_topics_endpoint and has_auth_profile):
            logger.error("❌ clean_main.py does not contain comprehensive endpoints!")
            logger.error("❌ This appears to be the basic version, not the comprehensive one")
            logger.error("❌ Railway will not start with incomplete backend")
            return False
        
        logger.info("✅ VERIFIED: clean_main.py contains comprehensive endpoint implementation")
        
        # Log environment check
        postgres_url = os.getenv('POSTGRES_URL')
        if postgres_url:
            # Extract host without credentials for logging
            if '@' in postgres_url:
                host_part = postgres_url.split('@')[1].split('/')[0]
                logger.info(f"🐘 PostgreSQL configured: {host_part}")
            else:
                logger.info("🐘 PostgreSQL environment detected")
        else:
            logger.warning("⚠️ POSTGRES_URL not found in environment variables")
        
        # Check other required environment variables
        jwt_secret = os.getenv('JWT_SECRET')
        google_client_id = os.getenv('GOOGLE_CLIENT_ID')
        
        logger.info(f"🔐 JWT_SECRET: {'✅ Set' if jwt_secret else '❌ Missing'}")
        logger.info(f"🔐 GOOGLE_CLIENT_ID: {'✅ Set' if google_client_id else '❌ Missing'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Startup check failed: {str(e)}")
        return False

def disable_sqlite_files():
    """Disable any SQLite files to force PostgreSQL usage"""
    try:
        sqlite_patterns = ['*.db', 'ai_news*.db']
        disabled_count = 0
        
        for pattern in sqlite_patterns:
            for sqlite_file in Path('.').glob(pattern):
                if sqlite_file.name.startswith('ai_news') or sqlite_file.suffix == '.db':
                    try:
                        backup_name = f"{sqlite_file}.disabled_railway"
                        sqlite_file.rename(backup_name)
                        logger.info(f"🗃️ Disabled SQLite file: {sqlite_file} → {backup_name}")
                        disabled_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Could not disable {sqlite_file}: {str(e)}")
        
        if disabled_count > 0:
            logger.info(f"✅ Disabled {disabled_count} SQLite files - PostgreSQL will be used exclusively")
        else:
            logger.info("ℹ️ No SQLite files found to disable")
            
    except Exception as e:
        logger.warning(f"⚠️ SQLite file disabling failed: {str(e)}")

def start_application():
    """Start the FastAPI application using clean_main.py"""
    try:
        logger.info("🚀 Starting AI News Scraper with modular PostgreSQL backend...")
        
        # Use uvicorn to run clean_main.py
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "clean_main:app",
            "--host", "0.0.0.0",
            "--port", str(os.getenv('PORT', 8000)),
            "--log-level", "info"
        ]
        
        logger.info(f"🔧 Running command: {' '.join(cmd)}")
        
        # Execute the command
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Application failed to start: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("🚂 Railway deployment starting...")
    logger.info("📋 Using COMPLETE modular PostgreSQL backend with comprehensive endpoints")
    logger.info("🎯 Version: 4.0.0-complete-modular-postgresql")
    
    # Perform startup checks
    if not ensure_correct_main_file():
        sys.exit(1)
    
    # Disable SQLite files
    disable_sqlite_files()
    
    # Start the application
    start_application()