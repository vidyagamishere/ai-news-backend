# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the backend component of Vidyagam, an AI news aggregation platform. It's a FastAPI application with PostgreSQL database deployed on Railway.

## Development Commands

### Running the Application
```bash
python main.py                          # Start development server (localhost:8000)
uvicorn main:app --reload --port 8000   # Alternative dev start with auto-reload
```

### Dependencies and Setup
```bash
pip install -r requirements.txt         # Install all dependencies
```

### Testing
```bash
python test_database_fixes.py           # Test database functionality
python test_railway_fix.py              # Test Railway deployment fixes
python test_crawl4ai_features.py        # Test content scraping features
```

## Architecture Overview

### Core Structure
- **main.py**: Application entry point with CORS, router registration, and lifespan management
- **db_service.py**: PostgreSQL connection pooling and database operations using psycopg2
- **crawl4ai_scraper.py**: AI-powered content scraping using Crawl4AI and Mistral AI
- **app/**: Modular application structure following FastAPI best practices

### Modular Components
- **app/routers/**: API endpoints organized by functionality
  - health.py: Health checks and status monitoring
  - auth.py: Authentication with Google OAuth and email verification
  - content.py: Content delivery and digest generation
  - admin.py: Administrative functionality
- **app/services/**: Business logic layer
  - auth_service.py: Authentication and user management
  - content_service.py: Content aggregation and personalization
- **app/models/schemas.py**: Pydantic models for request/response validation
- **app/dependencies/**: Dependency injection for auth and database

### Database Architecture
- **PostgreSQL**: Primary database with connection pooling
- **Migration Support**: Automatic schema initialization and SQLite migration
- **Key Tables**:
  - users: User accounts with authentication and preferences
  - ai_sources: News sources with RSS feeds and categorization
  - ai_topics: Topic categories with keywords and priorities
  - articles: Scraped content with AI-generated summaries and significance scoring
  - user_preferences: User topic preferences for personalization

### Content Pipeline
1. **RSS Feed Monitoring**: Automated scraping from configured AI news sources
2. **Crawl4AI Processing**: Advanced web scraping with JavaScript execution
3. **Claude AI Integration**: Content summarization, categorization, and significance scoring
4. **PostgreSQL Storage**: Structured content storage with full-text search capabilities
5. **Personalization Engine**: Topic-based content filtering and recommendation

## Key Features

### Authentication System
- Google OAuth integration with JWT tokens
- Email verification with OTP support
- User preference management and onboarding
- Admin role-based access control

### Content Aggregation
- **Automated RSS Scraping**: 8-hour scheduled scraping from AI news sources
- **8-Hour Duplicate Prevention**: Prevents re-scraping URLs within 8 hours
- **AI-powered Content Processing**: Claude AI for summarization and categorization
- **Topic Classification**: Automatic categorization into 10 AI topic areas
- **Significance Scoring**: 1-10 importance ranking for content prioritization
- **Keywords Extraction**: Automatic keyword extraction and storage for search

### API Endpoints
- `/health`: System health and database status
- `/digest`: Personalized content digest generation
- `/content/{type}`: Content filtering by type
- `/auth/*`: Authentication and user management
- `/admin/*`: Administrative functions
- `/admin/scheduler/status`: Auto-scraping scheduler status monitoring
- `/content/admin/scrape`: Manual trigger for immediate RSS scraping (existing)

## Environment Configuration

### Required Environment Variables
- `POSTGRES_URL` or `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET_KEY`: JWT token signing secret
- `GOOGLE_CLIENT_ID`: Google OAuth client ID
- `SENDGRID_API_KEY`: Email service for notifications
- `ANTHROPIC_API_KEY`: Claude AI API access

### Optional Configuration
- `DEBUG`: Enable debug logging (default: false)
- `LOG_LEVEL`: Logging level (default: INFO)
- `ALLOWED_ORIGINS`: Additional CORS origins
- `SKIP_SCHEMA_INIT`: Skip database schema initialization (for existing databases)

### Auto-Scraping Scheduler Configuration
- `AUTO_SCRAPING_ENABLED`: Enable/disable automated RSS scraping (default: true)
- `SCRAPING_INTERVAL_HOURS`: Scraping interval in hours (default: 8)
- `RUN_INITIAL_SCRAPING`: Run scraping immediately on startup (default: false)

## Railway Deployment

### Configuration Files
- **railway.json**: Deployment configuration with health checks
- **railway.toml**: Build and deploy settings
- **Dockerfile**: Container configuration
- **requirements.txt**: Python dependencies optimized for Railway

### Deployment Features
- Automatic PostgreSQL database provisioning
- Health check endpoint monitoring
- Connection pooling for database efficiency
- Environment-specific configuration handling
- Automatic schema migration and initialization

## Development Guidelines

### Database Operations
- Use `db_service.py` for all database interactions
- Connection pooling is handled automatically
- Use parameterized queries to prevent SQL injection
- Database schema is initialized automatically on first startup

### Content Processing
- **Automated Scraping**: Runs every 8 hours via APScheduler with RSS feed parsing
- **8-Hour Duplicate Check**: Prevents re-processing URLs scraped within 8 hours
- **AI Processing**: Requires valid Anthropic API key for Claude content analysis
- **Significance Scoring**: 1-10 ranking system for content importance
- **Keywords Support**: Automatic extraction and database storage for search functionality
- **Manual Triggers**: Admin can trigger immediate scraping via `/admin/scheduler/trigger`

### API Development
- Follow FastAPI router pattern in `app/routers/`
- Use Pydantic schemas in `app/models/schemas.py` for validation
- Implement business logic in `app/services/`
- Include proper error handling and logging

### Testing
- Test files are prefixed with `test_`
- Database tests use separate test databases
- Content scraping tests may require API keys
- Railway deployment tests validate production readiness