#!/usr/bin/env python3
"""
Content router for modular FastAPI architecture
Maintains compatibility with existing frontend API endpoints
"""

import os
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request,BackgroundTasks

from app.models.schemas import DigestResponse, ContentByTypeResponse, UserResponse
from app.dependencies.auth import get_current_user_optional, get_current_user
from app.services.content_service import ContentService
from app.services.pagination_service import pagination_service
from app.models.pagination import PaginationParams, PaginationMeta
import uuid
from datetime import datetime, timezone

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

@router.get("/content/paginated")
async def get_paginated_content(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page (max 50)"),
    content_type: Optional[str] = Query(None, description="Filter: blogs, podcasts, videos"),
    category_id: Optional[int] = Query(None, description="Filter by category ID"),
    sort_by: str = Query("published_date", description="Sort field"),
    sort_order: str = Query("desc", description="asc or desc")
):
    """
    Get paginated content with filtering and sorting.
    
    Steve Jobs Strategy:
    - Fast: Returns only 10 items per request
    - Smart: Prefetch triggers when user scrolls to item 7
    - Smooth: No loading spinners, seamless infinite scroll
    
    Example:
        GET /api/content/paginated?page=1&page_size=10&content_type=blogs
        
    Returns:
        {
            "success": true,
            "items": [...10 articles...],
            "meta": {
                "current_page": 1,
                "page_size": 10,
                "total_items": 245,
                "total_pages": 25,
                "has_next": true,
                "has_prev": false,
                "next_page": 2,
                "prev_page": null
            }
        }
    """
    try:
        logger.info(f"📄 Paginated request: page={page}, size={page_size}, type={content_type}")
        
        from db_service import get_database_service
        db = get_database_service()
        
        # Build WHERE clause
        
        where_conditions = [] 
        params = []
        
        if content_type:
            where_conditions.append("ct.name = %s")
            params.append(content_type)
        
        if category_id:
            where_conditions.append("a.category_id = %s")
            params.append(category_id)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
    
        logger.info(f"🔍 DEBUG: WHERE clause = {where_clause}, params = {params}")  # ✅ ADD DEBUG
        
        # Validate sort_order
        if sort_order.lower() not in ['asc', 'desc']:
            sort_order = 'desc'
        
        # Validate sort_by
        valid_sort_fields = ['published_date', 'significance_score', 'title', 'source']
        if sort_by not in valid_sort_fields:
            sort_by = 'published_date'
        
        # Base query
        base_query = f"""
            SELECT 
                a.id,
                a.title,
                a.summary,
                a.url,
                a.source,
                a.published_date,
                a.significance_score,
                a.image_url,
                a.image_source,
                a.reading_time,
                a.author,
                ct.name as content_type,
                ct.display_name as content_type_display,
                cm.name as category_name,
                cm.id as category_id,
                a.llm_processed
            FROM articles a
            LEFT JOIN content_types ct ON a.content_type_id = ct.id
            LEFT JOIN ai_categories_master cm ON a.category_id = cm.id
            WHERE {where_clause}
            ORDER BY a.{sort_by} {sort_order}
        """
        
        # Count query
        count_query = f"""
            SELECT COUNT(*) as count
            FROM articles a
            LEFT JOIN content_types ct ON a.content_type_id = ct.id
            WHERE {where_clause}
        """
        
        # Execute paginated query using pagination service
        items, meta = await pagination_service.paginate_query(
            db_service=db,
            base_query=base_query,
            count_query=count_query,
            params=tuple(params),
            page=page,
            page_size=page_size
        )
        
        logger.info(f"✅ Returned {len(items)} items (page {page}/{meta['total_pages']})")
        
        return {
            "success": True,
            "items": items,
            "meta": meta,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Pagination failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
    category_id: str = Query("all", description="Category ID or 'all' for all categories"),
    time_filter: str = Query("All Time", description="Time filter: 'Last 24 Hours', 'Last Week', 'Last Month', 'This Year', 'All Time'")
):
    """
    Get content counts by category and content type
    ✅ NOW SUPPORTS TIME FILTERING - returns counts matching the selected time filter
    """
    try:
        logger.info(f"📊 Content counts requested for category: {category_id}, time_filter: {time_filter}")
        
        # Get counts from content service WITH time filtering
        content_service = ContentService()
        counts = content_service.get_content_counts(category_id, time_filter)
        
        logger.info(f"✅ Content counts retrieved successfully for time filter: {time_filter}")
        return counts
        
    except Exception as e:
        logger.error(f"❌ Failed to get content counts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get content counts: {str(e)}"
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

# FIND LINE 457 and REPLACE the entire function with:

@router.get("/landing-content")
async def get_landing_content(
    page: int = Query(1, ge=1),                      # ✅ NEW: Page number
    page_size: int = Query(10, ge=1, le=50)          # ✅ NEW: Items per page
):
    """
    Get all categories and content types for landing page with PAGINATION.
    Returns content organized by category (Generative AI, AI Applications, AI Startups)
    Each category contains content types (blogs, podcasts, videos) with paginated items
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /landing-content
    Query params: 
        - page: Page number (default: 1)
        - page_size: Items per content type (default: 10, max: 50)
    
    Example:
        GET /api/landing-content?page=1&page_size=10
    
    Returns:
        {
            "success": true,
            "categories": [...],
            "meta": {
                "page": 1,
                "page_size": 10,
                "total_blogs": 150,
                "total_podcasts": 45,
                "total_videos": 50,
                "has_next": true,
                "total_pages": 15
            }
        }
    """
    try:
        logger.info(f"🏠 Landing content requested - Page: {page}, Size: {page_size}")
        
        from db_service import get_database_service
        db = get_database_service()
        
        # Get all categories sorted by priority
        categories_query = """
            SELECT id, name, priority, description, default_image_url
            FROM ai_categories_master
            WHERE is_active = TRUE
            ORDER BY priority ASC
        """
        categories = db.execute_query(categories_query, fetch_all=True)
        
        # If no categories, create default ones
        if not categories:
            categories = [
                {'id': 1, 'name': 'Generative AI', 'priority': 1, 'description': 'LLMs, GPT, Claude, and AI Generation', 'default_image_url': ''},
                {'id': 2, 'name': 'AI Applications', 'priority': 2, 'description': 'Enterprise Use Cases & Industry Solutions', 'default_image_url': ''},
                {'id': 3, 'name': 'AI Startups', 'priority': 3, 'description': 'Funding, M&A & Emerging Companies', 'default_image_url': ''}
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
        
        result = []
        offset = (page - 1) * page_size  # ✅ Calculate offset for pagination
        
        for category in categories:
            category_data = {
                'id': category['id'],
                'name': category['name'],
                'priority': category['priority'],
                'description': category.get('description', ''),
                'default_image_url': category.get('default_image_url', ''),
                'content': {}
            }
            
            logger.info(f"🔍 Processing category: {category['name']}")
            
            # For each content type, get PAGINATED articles
            for content_type in content_types:
                logger.info(f"🔍 Fetching articles for type: {content_type['name']} in category: {category['name']}")
                
                # ✅ PAGINATED QUERY with LIMIT and OFFSET
                articles_query = """
                    SELECT a.title, a.summary, a.url, a.source, a.significance_score, 
                           a.published_date, a.author, ct.name as content_type_name,
                           cm.name as category_name, a.image_url, a.image_source
                    FROM articles a
                    LEFT JOIN content_types ct ON a.content_type_id = ct.id
                    LEFT JOIN ai_categories_master cm ON a.category_id = cm.id
                    WHERE ct.name = %s 
                    AND cm.id = %s
                    AND a.published_date >= NOW() - INTERVAL '30 days'
                    ORDER BY a.published_date DESC, a.source, a.significance_score DESC
                    LIMIT %s OFFSET %s
                """
                
                # ✅ Pass page_size and offset for pagination
                articles = db.execute_query(
                    articles_query, 
                    (content_type['name'], category['id'], page_size, offset), 
                    fetch_all=True
                )
                
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
                        'content_type': content_type['name'],
                        'image_url': article.get('image_url', ''),
                        'image_source': article.get('image_source', '')
                    })
                
                logger.info(f"✅ Fetched {len(formatted_articles)} articles for type: {content_type['name']}")
                category_data['content'][content_type['name']] = formatted_articles
            
            result.append(category_data)
        
        # ✅ Get total counts for pagination metadata
        count_query = """
            SELECT 
                ct.name as content_type,
                COUNT(*) as count
            FROM articles a
            LEFT JOIN content_types ct ON a.content_type_id = ct.id
            WHERE a.published_date >= NOW() - INTERVAL '30 days'
            GROUP BY ct.name
        """
        counts = db.execute_query(count_query, fetch_all=True)
        
        total_blogs = next((c['count'] for c in counts if c['content_type'] == 'blogs'), 0)
        total_podcasts = next((c['count'] for c in counts if c['content_type'] == 'podcasts'), 0)
        total_videos = next((c['count'] for c in counts if c['content_type'] == 'videos'), 0)
        
        # ✅ Calculate pagination metadata
        max_total = max(total_blogs, total_podcasts, total_videos)
        has_next = (page * page_size) < max_total
        total_pages = (max_total + page_size - 1) // page_size  # Ceiling division
        
        logger.info(f"✅ Landing content retrieved: {len(categories)} categories, page {page}/{total_pages}")
        
        return {
            'success': True,
            'categories': result,
            'meta': {
                'page': page,
                'page_size': page_size,
                'total_blogs': total_blogs,
                'total_podcasts': total_podcasts,
                'total_videos': total_videos,
                'has_next': has_next,
                'total_pages': total_pages
            },
            'total_categories': len(categories),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Landing content endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get landing content',
                'message': str(e)
            }
        )
        
# Global scraping job tracker
scraping_jobs = {}

def track_scraping_job(job_id: str, llm_model: str, scrape_frequency: int, content_service: ContentService):
    """Background task for scraping - runs async"""
    try:
        scraping_jobs[job_id]['status'] = 'running'
        scraping_jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        logger.info(f"🔄 Background scraping job {job_id} started")
        
        # Run the actual scraping
        import asyncio
        result = asyncio.run(content_service.scrape_with_frequency(
            llm_model=llm_model,
            scrape_frequency=scrape_frequency
        ))
        
        scraping_jobs[job_id]['status'] = 'completed'
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        scraping_jobs[job_id]['result'] = result
        
        logger.info(f"✅ Background scraping job {job_id} completed")
        
    except Exception as e:
        scraping_jobs[job_id]['status'] = 'failed'
        scraping_jobs[job_id]['error'] = str(e)
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        logger.error(f"❌ Background scraping job {job_id} failed: {str(e)}")

@router.post("/admin/scrape")
async def admin_initiate_scraping(
    request: Request,
    background_tasks: BackgroundTasks,  # ✅ ADD THIS
    llm_model: str = Query('claude', description="LLM model to use"),
    scrape_frequency: int = Query(1, description="Scrape frequency in days (1, 7, or 30)"),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Admin-only endpoint to initiate AI news scraping process (ASYNC)
    Returns immediately with job_id, use /admin/scrape-status/{job_id} to check progress
    """
    try:
        # Check for admin API key authentication
        admin_api_key = request.headers.get('X-Admin-API-Key')
        expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')
        
        if not admin_api_key or admin_api_key != expected_api_key:
            raise HTTPException(status_code=403, detail='Admin access required')
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        scraping_jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'llm_model': llm_model,
            'scrape_frequency': scrape_frequency,
            'created_at': datetime.now().isoformat()
        }
        
        # ✅ Start background task (returns immediately)
        background_tasks.add_task(
            track_scraping_job,
            job_id=job_id,
            llm_model=llm_model,
            scrape_frequency=scrape_frequency,
            content_service=content_service
        )
        
        logger.info(f"✅ Scraping job {job_id} queued successfully")
        
        return {
            'success': True,
            'job_id': job_id,
            'message': f'Scraping job started for {scrape_frequency}-day frequency using {llm_model}',
            'status': 'queued',
            'check_status_url': f'/api/content/admin/scrape-status/{job_id}',
            'estimated_duration_minutes': '3-5'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start scraping job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/scrape-status/{job_id}")
async def get_scraping_status(
    request: Request,
    job_id: str
):
    """
    Check status of a scraping job
    Returns: queued | running | completed | failed
    """
    try:
        # Check for admin API key
        admin_api_key = request.headers.get('X-Admin-API-Key')
        expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')
        
        if not admin_api_key or admin_api_key != expected_api_key:
            raise HTTPException(status_code=403, detail='Admin access required')
        
        if job_id not in scraping_jobs:
            raise HTTPException(status_code=404, detail='Job not found')
        
        job = scraping_jobs[job_id]
        
        return {
            'job_id': job_id,
            'status': job['status'],
            'llm_model': job['llm_model'],
            'scrape_frequency': job['scrape_frequency'],
            'created_at': job['created_at'],
            'started_at': job.get('started_at'),
            'completed_at': job.get('completed_at'),
            'result': job.get('result'),
            'error': job.get('error')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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