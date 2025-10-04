#!/bin/bash

# Quick Local Development Setup Script
echo "🚀 Setting up Vidyagam Local Development Environment"
echo "=============================================="

# Check if we're in the right directory
if [ ! -d "ai-news-backend" ] || [ ! -d "ai-news-frontend" ]; then
    echo "❌ Error: Please run this script from the Vidyagam root directory"
    echo "Expected directories: ai-news-backend, ai-news-frontend"
    exit 1
fi

echo "📁 Current directory: $(pwd)"

# Function to prompt for input with default
prompt_with_default() {
    local prompt=$1
    local default=$2
    local response
    
    read -p "$prompt [$default]: " response
    echo "${response:-$default}"
}

echo ""
echo "🔧 Backend Environment Setup"
echo "----------------------------"

# Get Railway database URL
echo "📊 Please provide your Railway PostgreSQL connection string:"
echo "Format: postgresql://username:password@host:port/database"
echo "You can find this in your Railway project dashboard."
DATABASE_URL=$(prompt_with_default "Database URL" "postgresql://username:password@host:port/database")

# Get other credentials
JWT_SECRET=$(prompt_with_default "JWT Secret Key" "your-super-secret-jwt-key-$(date +%s)")
GOOGLE_CLIENT_ID=$(prompt_with_default "Google Client ID" "your-google-client-id")
SENDGRID_API_KEY=$(prompt_with_default "SendGrid API Key (optional)" "your-sendgrid-api-key")
ANTHROPIC_API_KEY=$(prompt_with_default "Anthropic API Key (optional)" "your-anthropic-api-key")

# Create backend .env file
cat > ai-news-backend/.env << EOF
# Local Development Environment - Generated $(date)
# Railway PostgreSQL Database
DATABASE_URL=$DATABASE_URL
POSTGRES_URL=$DATABASE_URL

# Authentication
JWT_SECRET_KEY=$JWT_SECRET
GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID

# Email Service
SENDGRID_API_KEY=$SENDGRID_API_KEY

# AI Service
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY

# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
SKIP_SCHEMA_INIT=true

# CORS Settings for Local Development
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Admin Settings
ADMIN_API_KEY=admin-api-key-2024
EOF

echo "✅ Backend .env created"

echo ""
echo "🌐 Frontend Environment Setup"
echo "----------------------------"

# Update frontend .env.local to point to localhost
cat > ai-news-frontend/.env.local << EOF
# Local Development Environment - Generated $(date)
# API Configuration - Points to LOCAL backend
VITE_API_BASE=http://localhost:8000

# Enable debug mode for development
VITE_DEBUG_MODE=true

# Development flags
VITE_ENABLE_ADMIN_FEATURES=true
VITE_ENABLE_BETA_FEATURES=true

# Google OAuth Configuration
VITE_GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID

# Admin API Key
VITE_ADMIN_API_KEY=admin-api-key-2024
EOF

echo "✅ Frontend .env.local updated"

echo ""
echo "📦 Installing Dependencies"
echo "-------------------------"

# Install backend dependencies
echo "🐍 Installing backend dependencies..."
cd ai-news-backend
if command -v python3 &> /dev/null; then
    python3 -m pip install -r requirements.txt
elif command -v python &> /dev/null; then
    python -m pip install -r requirements.txt
else
    echo "❌ Python not found. Please install Python and try again."
    exit 1
fi
cd ..

# Install frontend dependencies
echo "📱 Installing frontend dependencies..."
cd ai-news-frontend
if command -v npm &> /dev/null; then
    npm install
else
    echo "❌ npm not found. Please install Node.js and try again."
    exit 1
fi
cd ..

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo "1. Start the backend server:"
echo "   cd ai-news-backend && python main.py"
echo ""
echo "2. Start the frontend server (in a new terminal):"
echo "   cd ai-news-frontend && npm run dev"
echo ""
echo "3. Open your browser to:"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo ""
echo "4. Test admin login:"
echo "   Go to: http://localhost:5173/admin/login"
echo "   Username: admin@vidyagam.com"
echo "   Password: Vidyagam@Success"
echo ""
echo "🔗 For detailed instructions, see: LOCAL_DEVELOPMENT_SETUP.md"
echo ""
echo "🚨 Important: Make sure your Railway PostgreSQL database is accessible!"