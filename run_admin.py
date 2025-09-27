#!/usr/bin/env python3
"""
Admin Interface Runner for AI News Database Management
"""

import os
import sys
from admin_interface import app, DatabaseManager

def setup_environment():
    """Setup environment variables for admin interface"""
    # Set default PostgreSQL connection if not provided
    if not os.getenv('POSTGRES_URL'):
        os.environ['POSTGRES_URL'] = 'postgresql://postgres:FgvftzrGueiGipLiRRMKMElppasuzBjptZlwPL@autorack.proxy.rlwy.net:51308/railway'
    
    # Set secret key for sessions
    if not os.getenv('SECRET_KEY'):
        os.environ['SECRET_KEY'] = 'ai-news-admin-secret-key-2025'
    
    # Set Brevo API key (optional for email service)
    if not os.getenv('BREVO_API_KEY'):
        print("⚠️  BREVO_API_KEY not set - OTP emails will show debug mode")
        print("   To enable email service, set: export BREVO_API_KEY=your_key")

def test_database_connection():
    """Test PostgreSQL database connection"""
    try:
        db_manager = DatabaseManager()
        connection = db_manager.get_connection()
        
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                print(f"✅ Database connected: {version[0]}")
        
        connection.close()
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        print("   Check your POSTGRES_URL environment variable")
        return False

def main():
    """Main runner function"""
    print("🚀 Starting AI News Admin Interface")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Test database connection
    if not test_database_connection():
        print("\n💡 To fix database connection:")
        print("   export POSTGRES_URL='your_postgresql_connection_string'")
        sys.exit(1)
    
    # Print access information
    print(f"\n📋 Admin Interface Information:")
    print(f"   🔗 URL: http://localhost:5001")
    print(f"   👤 Admin Email: admin@vidyagam.com")
    print(f"   🔑 Authentication: Email OTP")
    print(f"   🗄️  Database: PostgreSQL on Railway")
    
    print(f"\n🎯 Features Available:")
    print(f"   • View all database tables")
    print(f"   • Add, edit, delete records")
    print(f"   • Download tables as CSV")
    print(f"   • Bulk upload via CSV")
    print(f"   • Trigger RSS scraping")
    print(f"   • Update news sources")
    
    print(f"\n🔧 Starting Flask development server...")
    print(f"   Press Ctrl+C to stop")
    print("=" * 50)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        use_reloader=False  # Avoid double startup messages
    )

if __name__ == '__main__':
    main()