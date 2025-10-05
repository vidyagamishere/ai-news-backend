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
    Get all AI topics
    Endpoint: GET /ai-topics (new endpoint for frontend)
    """
    try:
        logger.info("📑 AI topics requested")
        
        topics = content_service.get_ai_topics()
        
        logger.info(f"✅ AI topics retrieved successfully - {len(topics)} topics")
        return {
            'topics': topics,
            'count': len(topics),
            'database': 'postgresql'
        }
        
    except Exception as e:
        logger.error(f"❌ AI topics endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get AI topics',
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
            SELECT title, description as summary, url, source, significance_score, 
                   published_at as published_date, content_type_name
            FROM breaking_news_alerts
            WHERE (title ILIKE '%AI%' OR title ILIKE '%GPT%' OR title ILIKE '%ChatGPT%' 
                   OR title ILIKE '%OpenAI%' OR title ILIKE '%generative%' OR title ILIKE '%Claude%')
            ORDER BY significance_score DESC, published_at DESC
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
    limit: int = Query(6, ge=1, le=20)
):
    """
    Get Generative AI category stories for pre-login landing page
    Returns curated AI articles focused on OpenAI, ChatGPT, Claude, etc.
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /generative-ai-content  
    Query params: limit (default: 6, max: 20)
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
            LEFT JOIN ai_categories_master c ON a.category_id = c.priority
            WHERE (c.name = 'Generative AI' 
                   OR a.title ILIKE '%OpenAI%' 
                   OR a.title ILIKE '%ChatGPT%' 
                   OR a.title ILIKE '%GPT%'
                   OR a.title ILIKE '%Claude%'
                   OR a.title ILIKE '%Gemini%'
                   OR a.title ILIKE '%generative%'
                   OR a.source ILIKE '%openai%'
                   OR a.source ILIKE '%anthropic%')
            AND a.significance_score >= 7.0
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


@router.post("/admin/scrape")
async def admin_initiate_scraping(
    request: Request,
    content_service: ContentService = Depends(get_content_service)
):
    """
    Admin-only endpoint to initiate AI news scraping process
    Uses Content Service for scraping operations
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
        
        if DEBUG:
            logger.debug(f"🔍 Admin scrape request with valid API key")
            logger.debug(f"🔍 Admin permissions verified")
        
        
        # Trigger scraping operation with detailed error handling
        try:
            logger.info("🔍 About to call content_service.scrape_content()")
            result = await content_service.scrape_content()
            logger.info("🔍 Successfully called content_service.scrape_content()")
        except NameError as ne:
            logger.error(f"❌ NameError in scrape_content: {str(ne)}")
            raise ne
        except Exception as se:
            logger.error(f"❌ Other error in scrape_content: {str(se)}")
            raise se
        
        if DEBUG:
            logger.debug(f"🔍 Scraping completed with result: {result}")
        
        logger.info(f"✅ Admin scraping completed successfully with API key authentication")
        return {
            'success': True,
            'message': 'Content scraping completed successfully',
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