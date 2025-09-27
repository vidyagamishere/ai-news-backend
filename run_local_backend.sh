#!/bin/bash

# Local Backend Development Server
echo "🚀 Starting AI News Scraper Backend (Local Development)"

# Check if we're in the right directory
if [ ! -f "api/main_with_auth.py" ]; then
    echo "❌ Error: main_with_auth.py not found. Please run this from the ai-news-scraper directory."
    exit 1
fi

# Create local environment if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.local" ]; then
        cp .env.local .env
        echo "📝 Created .env from .env.local template"
        echo "⚠️  Please update .env with your actual API keys before continuing"
        echo ""
        read -p "Press Enter after updating .env file..."
    else
        echo "❌ No .env.local template found. Please create .env file with required variables."
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
if [ -f "api/requirements_auth.txt" ]; then
    pip install -r api/requirements_auth.txt
else
    pip install -r requirements.txt
fi

# Install additional development dependencies
pip install python-multipart uvicorn[standard]

# Check if database exists, if not initialize it
if [ ! -f "ai_news_local.db" ]; then
    echo "🗄️ Initializing local database..."
    # The database will be created automatically when the app starts
fi

# Start the development server
echo "🌟 Starting development server at http://localhost:8000"
echo "📖 API Documentation available at http://localhost:8000/docs"
echo "🔍 Health check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Change to api directory and run the server
cd api
python -m uvicorn main_with_auth:app --reload --host 0.0.0.0 --port 8000