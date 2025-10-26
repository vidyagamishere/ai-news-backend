#!/usr/bin/env python3
"""
Content router for modular FastAPI architecture
Maintains compatibility with existing frontend API endpoints
"""

import os
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.models.schemas import DigestResponse, ContentByTypeResponse, UserResponse
from app.dependencies.auth import get_current_user_optional, get_current_user
from app.services.content_service import ContentService

logger = logging.getLogger(__name__)

# Get DEBUG mode
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

router = APIRouter()

if DEBUG:
    logger.debug("🔍 Content router initialized in DEBUG mode")


def get_content_service() -> ContentService:
    """Dependency to get ContentService instance"""
    return ContentService()


@router.get("/digest", response_model=DigestResponse)
async def get_digest(
    current_user: Optional[UserResponse] = Depends(get_current_user_optional),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get general digest - compatible with existing frontend
    Endpoint: GET /digest (same as before)
    """
    try:
        logger.info(f"📊 Digest requested - User: {current_user.email if current_user else 'anonymous'}")
        
        is_personalized = bool(current_user)
        digest = content_service.get_digest(
            user_id=current_user.id if current_user else None,
            personalized=is_personalized
        )
        
        logger.info("✅ Digest generated successfully")
        return digest
        
    except Exception as e:
        logger.error(f"❌ Digest endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get digest',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/personalized-digest", response_model=DigestResponse)
async def get_personalized_digest(
    current_user: UserResponse = Depends(get_current_user_optional),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get personalized digest - requires authentication
    Endpoint: GET /personalized-digest (same as before)
    """
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail={
                'error': 'Authentication required',
                'message': 'Please log in to access personalized content'
            }
        )
    
    try:
        logger.info(f"📊 Personalized digest requested - User: {current_user.email}")
        
        digest = content_service.get_digest(
            user_id=current_user.id,
            personalized=True
        )
        
        logger.info("✅ Personalized digest generated successfully")
        return digest
        
    except Exception as e:
        logger.error(f"❌ Personalized digest endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get personalized digest',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/content/{content_type}", response_model=ContentByTypeResponse)
async def get_content_by_type(
    content_type: str,
    limit: int = Query(20, ge=1, le=100),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get content by type - compatible with existing frontend
    Endpoint: GET /content/{type} (same as before)
    """
    try:
        logger.info(f"📄 Content by type requested - Type: {content_type}, Limit: {limit}")
        
        articles = content_service.get_content_by_type(content_type, limit)
        
        response = ContentByTypeResponse(
            articles=articles,
            content_type=content_type,
            count=len(articles),
            database="postgresql"
        )
        
        logger.info(f"✅ Content by type generated successfully - {len(articles)} articles")
        return response
        
    except Exception as e:
        logger.error(f"❌ Content by type endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get content',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/ai-topics")
async def get_ai_topics(
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get all AI topics and content types
    Endpoint: GET /ai-topics (new endpoint for frontend)
    """
    try:
        logger.info("📑 AI categories and content types requested")

        categories = content_service.get_ai_categories()
        content_types = content_service.get_content_types()

        logger.info(f"✅ AI categories retrieved successfully - {len(categories)} categories")
        logger.info(f"✅ Content types retrieved successfully - {len(content_types)} types")
        
        return {
            'categories': categories,
            'content_types': content_types,
            'count': len(categories),
            'database': 'postgresql'
        }
        
    except Exception as e:
        logger.error(f"❌ AI categories endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get AI categories',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/content-types")
async def get_content_types(
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get all content types
    Endpoint: GET /content-types (new endpoint for frontend)
    """
    try:
        logger.info("📋 Content types requested")
        
        content_types = content_service.get_content_types()
        
        logger.info(f"✅ Content types retrieved successfully - {len(content_types)} types")
        return {
            'content_types': {ct['name']: ct for ct in content_types},
            'count': len(content_types),
            'database': 'postgresql'
        }
        
    except Exception as e:
        logger.error(f"❌ Content types endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get content types',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/content-counts")
async def get_content_counts(
    category_id: Optional[str] = Query(None, description="Category ID or 'all' for all categories"),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get content counts by category and content type
    Endpoint: GET /content-counts?category_id=all or /content-counts?category_id=category_name
    Returns: {
        'total_articles': int,
        'total_podcasts': int, 
        'total_videos': int,
        'by_category': {...}
    }
    """
    try:
        logger.info(f"📊 Content counts requested for category: {category_id or 'all'}")
        
        counts = content_service.get_content_counts(category_id)
        
        logger.info(f"✅ Content counts retrieved successfully")
        return counts
        
    except Exception as e:
        logger.error(f"❌ Content counts endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get content counts',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/breaking-news")
async def get_breaking_news_alerts(
    limit: int = Query(5, ge=1, le=10)
):
    """
    Get breaking news alerts for pre-login landing page
    Returns high significance score Generative AI articles from last 24 hours
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /breaking-news
    Query params: limit (default: 5, max: 10)
    """
    try:
        logger.info(f"🚨 Breaking news requested - Limit: {limit}")
        
        from db_service import get_database_service
        db = get_database_service()
        
        # Use the existing breaking_news_alerts view and filter for Generative AI
        query = """
            SELECT title, summary, url, source, significance_score, 
                   published_date, name as content_type_name
            FROM breaking_news_alerts
            JOIN ai_categories_master ON breaking_news_alerts.category_id = ai_categories_master.id
            ORDER BY significance_score DESC, published_date DESC
            LIMIT %s
        """
        
        articles = db.execute_query(query, (limit,), fetch_all=True)
        
        result = []
        for article in articles:
            result.append({
                'title': article['title'],
                'summary': article['summary'] or '',
                'url': article['url'],
                'source': article['source'],
                'significanceScore': float(article['significance_score']) if article['significance_score'] else 8.5,
                'published_date': article['published_date'].isoformat() if article['published_date'] else None,
                'content_type': article['content_type_name'],
                'category': 'Generative AI'
            })
        
        logger.info(f"✅ Breaking news retrieved: {len(result)} articles")
        return {
            'articles': result,
            'count': len(result),
            'type': 'breaking_news'
        }
        
    except Exception as e:
        logger.error(f"❌ Breaking news endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get breaking news',
                'message': str(e)
            }
        )


@router.get("/generative-ai-content")
async def get_generative_ai_stories(
    limit: int = Query(3, ge=1, le=20)
):
    """
    Get Generative AI category stories for pre-login landing page
    Returns curated AI articles focused on OpenAI, ChatGPT, Claude, etc.
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /generative-ai-content  
    Query params: limit (default: 3, max: 20)
    """
    try:
        logger.info(f"🤖 Generative AI content requested - Limit: {limit}")
        
        from db_service import get_database_service
        db = get_database_service()
        
        # Get articles focused on Generative AI, OpenAI, etc.
        query = """
            SELECT a.title, a.summary, a.url, a.source, a.significance_score, 
                   a.published_date, a.author, c.name as category
            FROM articles a
            LEFT JOIN ai_categories_master c ON a.category_id = c.id
            WHERE (c.name = 'Generative AI')
            ORDER BY a.significance_score DESC, a.scraped_date DESC
            LIMIT %s
        """
        
        articles = db.execute_query(query, (limit,), fetch_all=True)
        
        result = []
        for article in articles:
            result.append({
                'title': article['title'],
                'summary': article['summary'] or '',
                'url': article['url'],
                'source': article['source'],
                'significanceScore': float(article['significance_score']) if article['significance_score'] else 7.0,
                'published_date': article['published_date'].isoformat() if article['published_date'] else None,
                'author': article['author'],
                'category': article['category'] or 'Generative AI'
            })
        
        logger.info(f"✅ Generative AI content retrieved: {len(result)} articles")
        return {
            'articles': result,
            'count': len(result),
            'type': 'generative_ai'
        }
        
    except Exception as e:
        logger.error(f"❌ Generative AI content endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get Generative AI content',
                'message': str(e)
            }
        )


@router.get("/ai-applications-content")
async def get_ai_applications_stories(
    limit: int = Query(3, ge=1, le=20)
):
    """
    Get AI Applications category stories for pre-login landing page
    Returns curated AI articles focused on enterprise use cases, industry solutions
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /ai-applications-content  
    Query params: limit (default: 3, max: 20)
    """
    try:
        logger.info(f"🏢 AI Applications content requested - Limit: {limit}")
        
        from db_service import get_database_service
        db = get_database_service()
        
    
        query = """
            SELECT a.title, a.summary, a.url, a.source, a.significance_score, 
                   a.published_date, a.author, 'AI Applications' as category
            FROM articles a
            WHERE a.category_id = 2
            ORDER BY a.significance_score DESC, a.scraped_date DESC
            LIMIT %s
            """
        articles = db.execute_query(query, (limit,), fetch_all=True)
        
        result = []
        for article in articles:
            result.append({
                'title': article['title'],
                'summary': article['summary'] or '',
                'url': article['url'],
                'source': article['source'],
                'significanceScore': float(article['significance_score']) if article['significance_score'] else 7.0,
                'published_date': article['published_date'].isoformat() if article['published_date'] else None,
                'author': article.get('author', ''),
                'category': article.get('category', 'AI Applications')
            })
        
        logger.info(f"✅ AI Applications content retrieved: {len(result)} articles")
        return {
            'articles': result,
            'count': len(result),
            'type': 'ai_applications'
        }
        
    except Exception as e:
        logger.error(f"❌ AI Applications content endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get AI Applications content',
                'message': str(e)
            }
        )


@router.get("/ai-startups-content")
async def get_ai_startups_stories(
    limit: int = Query(3, ge=1, le=20)
):
    """
    Get AI Startups category stories for pre-login landing page
    Returns curated AI articles focused on funding, M&A, emerging companies
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /ai-startups-content  
    Query params: limit (default: 3, max: 20)
    """
    try:
        logger.info(f"🚀 AI Startups content requested - Limit: {limit}")
        
        from db_service import get_database_service
        db = get_database_service()
        
        query = """
            SELECT a.title, a.summary, a.url, a.source, a.significance_score, 
                   a.published_date, a.author, 'AI Startups' as category
            FROM articles a
            WHERE a.category_id = 3
            ORDER BY a.scraped_date DESC
            LIMIT %s
            """
        articles = db.execute_query(query, (limit,), fetch_all=True)
        
        result = []
        for article in articles:
            result.append({
                'title': article['title'],
                'summary': article['summary'] or '',
                'url': article['url'],
                'source': article['source'],
                'significanceScore': float(article['significance_score']) if article['significance_score'] else 7.0,
                'published_date': article['published_date'].isoformat() if article['published_date'] else None,
                'author': article.get('author', ''),
                'category': article.get('category', 'AI Startups')
            })
        
        logger.info(f"✅ AI Startups content retrieved: {len(result)} articles")
        return {
            'articles': result,
            'count': len(result),
            'type': 'ai_startups'
        }
        
    except Exception as e:
        logger.error(f"❌ AI Startups content endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get AI Startups content',
                'message': str(e)
            }
        )


@router.get("/landing-content")
async def get_landing_content(
    limit_per_type: int = Query(3, ge=1, le=10)
):
    """
    Get all categories and content types for landing page
    Returns content organized by category (Generative AI, AI Applications, AI Startups)
    Each category contains content types (blogs, podcasts, videos) with max 3 items per type
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /landing-content
    Query params: limit_per_type (default: 3, max: 10)
    """
    try:
        logger.info(f"🏠 Landing content requested - Limit per type: {limit_per_type}")
        
        from db_service import get_database_service
        db = get_database_service()
        
        # Get all categories sorted by priority
        categories_query = """
            SELECT id, name, priority, description
            FROM ai_categories_master
            ORDER BY priority ASC
        """
        categories = db.execute_query(categories_query, fetch_all=True)
        
        # If no categories, create default ones
        if not categories:
            categories = [
                {'id': 1, 'name': 'Generative AI', 'priority': 1, 'description': 'LLMs, GPT, Claude, and AI Generation'},
                {'id': 2, 'name': 'AI Applications', 'priority': 2, 'description': 'Enterprise Use Cases & Industry Solutions'},
                {'id': 3, 'name': 'AI Startups', 'priority': 3, 'description': 'Funding, M&A & Emerging Companies'}
            ]
        
        # Get content types
        content_types_query = """
            SELECT id, name, display_name, frontend_section
            FROM content_types
            WHERE is_active = TRUE
        """
        content_types = db.execute_query(content_types_query, fetch_all=True)
        if not content_types:
            content_types = [
                {'id': 1, 'name': 'blogs', 'display_name': 'Blogs', 'frontend_section': 'blog'},
                {'id': 2, 'name': 'podcasts', 'display_name': 'Podcasts', 'frontend_section': 'podcast'},
                {'id': 3, 'name': 'videos', 'display_name': 'Videos', 'frontend_section': 'video'}
            ]
        
        result = {
            'categories': [],
            'total_categories': len(categories)
        }
        
        for category in categories:
            category_data = {
                'id': category['id'],
                'name': category['name'],
                'priority': category['priority'],
                'description': category.get('description', ''),
                'content': {}
            }
            logger.info(f"🔍 Processing category: {category['name']}")
            
            # For each content type, get articles
            for content_type in content_types:
                # Get articles for this category and content type
                logger.info(f"🔍 Fetching articles for type: {content_type['name']} in category: {category['name']}")
                articles_query = """
                    SELECT a.title, a.summary, a.url, a.source, a.significance_score, 
                           a.published_date, a.author, ct.name as content_type_name,
                           cm.name as category_name
                    FROM articles a
                    LEFT JOIN content_types ct ON a.content_type_id = ct.id
                    LEFT JOIN ai_categories_master cm ON a.category_id = cm.id
                    WHERE ct.name = %s 
                    AND cm.id = %s
                    ORDER BY a.significance_score DESC, a.scraped_date DESC
                    LIMIT %s
                """
                articles = db.execute_query(articles_query, (content_type['name'], category['id'], limit_per_type), fetch_all=True)

                # Format articles
                formatted_articles = []
                for article in articles:
                    formatted_articles.append({
                        'title': article['title'],
                        'summary': article['summary'] or '',
                        'url': article['url'],
                        'source': article['source'],
                        'significanceScore': float(article['significance_score']) if article['significance_score'] else 7.0,
                        'published_date': article['published_date'].isoformat() if article['published_date'] else None,
                        'author': article.get('author', ''),
                        'category': category['name'],
                        'content_type': content_type['name']
                    })
                logger.info(f"🔍 Articles fetched for type: {content_type['name']} in category: {category['name']}")
                category_data['content'][content_type['name']] = formatted_articles

            result['categories'].append(category_data)
        
        logger.info(f"✅ Landing content retrieved: {len(categories)} categories")
        return result
        
    except Exception as e:
        logger.error(f"❌ Landing content endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get landing content',
                'message': str(e)
            }
        )


@router.post("/admin/scrape")
async def admin_initiate_scraping(
    request: Request,
    llm_model: str = Query('claude', description="LLM model to use: 'claude', 'gemini', or 'huggingface'"),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Admin-only endpoint to initiate AI news scraping process
    Uses Content Service for scraping operations
    
    Query Parameters:
    - llm_model: 'claude' | 'gemini' | 'huggingface' (default: 'claude')
    """
    try:
        # Check for admin API key authentication
        admin_api_key = request.headers.get('X-Admin-API-Key')
        expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')
        
        if not admin_api_key or admin_api_key != expected_api_key:
            logger.warning(f"⚠️ Unauthorized admin scraping attempt")
            raise HTTPException(
                status_code=403,
                detail={
                    'error': 'Admin access required',
                    'message': 'Valid admin API key required for scraping'
                }
            )
        
        logger.info(f"🔧 Admin scraping initiated with API key authentication")
        logger.info(f"🤖 LLM MODEL SELECTED: '{llm_model}'")
        
        if DEBUG:
            logger.debug(f"🔍 Admin scrape request with valid API key")
            logger.debug(f"🔍 Admin permissions verified")
            logger.debug(f"🔍 LLM model parameter: {llm_model}")
        
        # Trigger scraping operation with LLM model parameter
        try:
            logger.info(f"🔍 About to call content_service.scrape_content(llm_model='{llm_model}')")
            result = await content_service.scrape_content(llm_model=llm_model)
            logger.info(f"🔍 Successfully called content_service.scrape_content() with {llm_model}")
        except NameError as ne:
            logger.error(f"❌ NameError in scrape_content: {str(ne)}")
            raise ne
        except Exception as se:
            logger.error(f"❌ Other error in scrape_content: {str(se)}")
            raise se
        
        if DEBUG:
            logger.debug(f"🔍 Scraping completed with result: {result}")
        
        logger.info(f"✅ Admin scraping completed successfully with {llm_model} model")
        return {
            'success': True,
            'message': f'Content scraping completed successfully using {llm_model}',
            'llm_model_used': llm_model,
            'data': result,
            'database': 'postgresql'
        }
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"❌ Admin scraping endpoint failed: {str(e)}")
        logger.error(f"❌ Full traceback: {error_traceback}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Admin scraping failed',
                'message': str(e),
                'traceback': error_traceback,
                'database': 'postgresql'
            }
        )


# ✅ NEW: Manual Scheduler Trigger Endpoint
@router.post("/admin/trigger-scheduler")
async def admin_trigger_scheduler(
    request: Request
):
    """
    Admin-only endpoint to manually trigger the scheduled scraping job
    This bypasses the 12-hour schedule and runs the job immediately
    Useful for testing and on-demand content updates
    
    Returns:
    - success: boolean
    - message: status message
    - llm_model: model that will be used (gemini)
    """
    try:
        # Check for admin API key authentication
        admin_api_key = request.headers.get('X-Admin-API-Key')
        expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')
        
        if not admin_api_key or admin_api_key != expected_api_key:
            logger.warning(f"⚠️ Unauthorized scheduler trigger attempt")
            raise HTTPException(
                status_code=403,
                detail={
                    'error': 'Admin access required',
                    'message': 'Valid admin API key required for scheduler trigger'
                }
            )
        
        logger.info(f"🔧 Admin triggered manual scheduler execution")
        
        # Import scheduler service
        from app.services.scheduler_service import scheduler_service
        
        # Trigger the scheduler manually
        success = scheduler_service.trigger_now()
        
        if success:
            logger.info(f"✅ Scheduler triggered successfully")
            return {
                'success': True,
                'message': 'Scheduled scraping job triggered successfully. Job will run immediately.',
                'llm_model_used': 'gemini',
                'schedule': '12 hours interval',
                'database': 'postgresql'
            }
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    'error': 'Scheduler trigger failed',
                    'message': 'Failed to trigger scheduler. Check server logs.'
                }
            )
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"❌ Scheduler trigger endpoint failed: {str(e)}")
        logger.error(f"❌ Full traceback: {error_traceback}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Scheduler trigger failed',
                'message': str(e),
                'traceback': error_traceback,
                'database': 'postgresql'
            }
        )