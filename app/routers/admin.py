#!/usr/bin/env python3
"""
Admin router for modular FastAPI architecture
Handles admin-only endpoints for source management and system control
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any
from fastapi import Body, APIRouter, Depends, HTTPException, Request, Query, Header

# Add parent directory to path to import scheduler_service
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models.schemas import UserResponse
from app.dependencies.auth import get_current_user
from app.services.content_service import ContentService
from db_service import get_database_service
from scheduler_service import get_scheduler
from app.services.shorts_service import create_shorts_for_articles
from crawl4ai_scraper import Crawl4AIScraper


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin")


def get_content_service() -> ContentService:
    """Dependency to get ContentService instance"""
    return ContentService()


def require_admin_access(request: Request) -> bool:
    """Dependency to ensure admin access via API key or JWT token"""
    # Check for admin API key first
    admin_api_key = request.headers.get('X-Admin-API-Key')
    expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')

    # 🐛 DEBUG: Log what we're receiving
    #logger.debug(f"🔑 Admin access check:")
    #logger.debug(f"   Received API key: {admin_api_key}")
    #logger.debug(f"   Expected API key: {expected_api_key}")
    #logger.debug(f"   Match: {admin_api_key == expected_api_key}")
    #logger.debug(f"   All headers: {dict(request.headers)}")
    
    if admin_api_key and admin_api_key == expected_api_key:
        return True
    
    # Fallback to JWT token authentication
    try:
        from app.dependencies.auth import get_current_user_from_token
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if token:
            current_user = get_current_user_from_token(token)
            if current_user and current_user.is_admin:
                return True
    except:
        pass
    
    raise HTTPException(
        status_code=403,
        detail={
            'error': 'Admin access required',
            'message': 'Admin API key or valid admin JWT token required'
        }
    )

def require_admin_user(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Dependency to ensure user is admin (JWT token only)"""
    if not current_user or not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail={
                'error': 'Admin access required',
                'message': 'Only admin users can access this endpoint'
            }
        )
    return current_user

@router.get("/articles")
async def get_articles(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    admin_access: bool = Depends(require_admin_access)
):
    try:
        db = get_database_service()
        articles = db.get_articles_paginated(page, page_size)
        total = db.get_articles_count()
        return {
            "articles": articles,
            "total": total,
            "page": page,
            "page_size": page_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-shorts")
async def generate_shorts(
    article_ids: list[int] = Body(...),
    request: Request = None,
    admin_access: bool = Depends(require_admin_access)
):
    """
    Generate shorts for selected articles (admin only)
    """
    try:
        result = await create_shorts_for_articles(article_ids)
        return {"status": "success", "processed": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
async def get_admin_sources(
    request: Request,
    content_service: ContentService = Depends(get_content_service),
    admin_access: bool = Depends(require_admin_access)
):
    """
    Get all AI sources for admin management
    Endpoint: GET /admin/sources
    """
    try:
        logger.info(f"🔧 Admin sources requested with API key authentication")
        
        db = get_database_service()
        
        # Get all sources with category via JOIN relationship (including inactive ones for admin view)
        sources_query = """
            SELECT s.id, s.name, s.rss_url, s.website, 
                   COALESCE(c.name, 'general') as category, 
                   s.priority, s.enabled, s.is_active,
                   s.created_date, s.updated_date, s.category_id
            FROM ai_sources s 
            LEFT JOIN ai_categories_master c ON s.category_id = c.id
            ORDER BY s.priority ASC, s.name ASC
        """
        
        sources = db.execute_query(sources_query)
        
        processed_sources = []
        for source in sources:
            source_dict = dict(source)
            # Convert timestamps to ISO format
            for field in ['created_date', 'updated_date']:
                if source_dict.get(field):
                    source_dict[field] = source_dict[field].isoformat() if hasattr(source_dict[field], 'isoformat') else str(source_dict[field])
            processed_sources.append(source_dict)
        
        logger.info(f"✅ Admin sources retrieved: {len(processed_sources)} sources")
        return {
            'sources': processed_sources,
            'total_count': len(processed_sources),
            'enabled_count': len([s for s in processed_sources if s.get('enabled', False)]),
            'active_count': len([s for s in processed_sources if s.get('is_active', False)]),
            'database': 'postgresql',
            'admin': 'api_key_authenticated'
        }
        
    except Exception as e:
        logger.error(f"❌ Admin sources endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get admin sources',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.post("/sources/add")
async def add_admin_source(
    source_data: Dict[str, Any],
    request: Request,
    admin_access: bool = Depends(require_admin_access)
):
    """
    Add new AI source
    Endpoint: POST /admin/sources/add
    """
    try:
        logger.info(f"🔧 Admin adding source with API key authentication")
        
        db = get_database_service()
        
        # Insert new source using category_id relationship
        insert_query = """
            INSERT INTO ai_sources (name, rss_url, website, category_id, priority, enabled, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        values = (
            source_data.get('name'),
            source_data.get('rss_url'),
            source_data.get('website', ''),
            source_data.get('category_id'),
            source_data.get('priority', 1),
            source_data.get('enabled', True),
            source_data.get('is_active', True)
        )
        
        result = db.execute_query(insert_query, values, fetch_one=True)
        
        logger.info(f"✅ Admin source added: ID {result['id']}")
        return {
            'success': True,
            'message': 'Source added successfully',
            'source_id': result['id'],
            'admin': 'api_key_authenticated'
        }
        
    except Exception as e:
        logger.error(f"❌ Admin add source failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to add source',
                'message': str(e)
            }
        )


@router.post("/sources/update")
async def update_admin_source(
    update_data: Dict[str, Any],
    request: Request,
    admin_access: bool = Depends(require_admin_access)
):
    """
    Update existing AI source
    Endpoint: POST /admin/sources/update
    """
    try:
        logger.info(f"🔧 Admin updating source with API key authentication")
        
        db = get_database_service()
        source_id = update_data.get('id')
        
        if not source_id:
            raise HTTPException(status_code=400, detail="Source ID required")
        
        # Update source using category_id relationship
        update_query = """
            UPDATE ai_sources 
            SET name = %s, rss_url = %s, website = %s, 
                category_id = %s, priority = %s, enabled = %s, is_active = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        
        values = (
            update_data.get('name'),
            update_data.get('rss_url'),
            update_data.get('website', ''),
            update_data.get('category_id'),
            update_data.get('priority', 1),
            update_data.get('enabled', True),
            update_data.get('is_active', True),
            source_id
        )
        
        db.execute_query(update_query, values, fetch_all=False)
        
        logger.info(f"✅ Admin source updated: ID {source_id}")
        return {
            'success': True,
            'message': 'Source updated successfully',
            'source_id': source_id,
            'admin': 'api_key_authenticated'
        }
        
    except Exception as e:
        logger.error(f"❌ Admin update source failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to update source',
                'message': str(e)
            }
        )


@router.post("/sources/delete")
async def delete_admin_source(
    delete_data: Dict[str, Any],
    request: Request,
    admin_access: bool = Depends(require_admin_access)
):
    """
    Delete AI source
    Endpoint: POST /admin/sources/delete
    """
    try:
        logger.info(f"🔧 Admin deleting source with API key authentication")
        
        db = get_database_service()
        source_id = delete_data.get('id')
        
        if not source_id:
            raise HTTPException(status_code=400, detail="Source ID required")
        
        # Delete source
        delete_query = "DELETE FROM ai_sources WHERE id = %s"
        db.execute_query(delete_query, (source_id,), fetch_all=False)
        
        logger.info(f"✅ Admin source deleted: ID {source_id}")
        return {
            'success': True,
            'message': 'Source deleted successfully',
            'source_id': source_id,
            'admin': 'api_key_authenticated'
        }
        
    except Exception as e:
        logger.error(f"❌ Admin delete source failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to delete source',
                'message': str(e)
            }
        )


@router.post("/validate-all-feeds")
async def validate_all_feeds(
    request: Request,
    admin_access: bool = Depends(require_admin_access)
):
    """
    Validate all RSS feeds
    Endpoint: POST /admin/validate-all-feeds
    """
    try:
        logger.info(f"🔧 Admin validating all feeds with API key authentication")
        
        db = get_database_service()
        
        # Get all active sources
        sources_query = """
            SELECT id, name, rss_url
            FROM ai_sources
            WHERE is_active = TRUE AND enabled = TRUE
        """
        
        sources = db.execute_query(sources_query)
        
        validation_results = []
        for source in sources:
            # Simple validation (can be enhanced with actual RSS validation)
            result = {
                'id': source['id'],
                'name': source['name'],
                'rss_url': source['rss_url'],
                'status': 'valid' if source['rss_url'] and source['rss_url'].startswith('http') else 'invalid',
                'message': 'URL format check passed' if source['rss_url'] and source['rss_url'].startswith('http') else 'Invalid URL format'
            }
            validation_results.append(result)
        
        logger.info(f"✅ Feed validation completed: {len(validation_results)} sources checked")
        return {
            'success': True,
            'message': 'Feed validation completed',
            'results': validation_results,
            'total_checked': len(validation_results),
            'admin': 'api_key_authenticated'
        }
        
    except Exception as e:
        logger.error(f"❌ Admin feed validation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to validate feeds',
                'message': str(e)
            }
        )


@router.post("/validate-feed")
async def validate_single_feed(
    feed_data: Dict[str, Any],
    request: Request,
    admin_access: bool = Depends(require_admin_access)
):
    """
    Validate single RSS feed
    Endpoint: POST /admin/validate-feed
    """
    try:
        logger.info(f"🔧 Admin validating single feed with API key authentication")
        
        feed_url = feed_data.get('feed_url')
        if not feed_url:
            raise HTTPException(status_code=400, detail="Feed URL required")
        
        # Simple validation (can be enhanced with actual RSS validation)
        is_valid = feed_url.startswith('http') and ('rss' in feed_url.lower() or 'feed' in feed_url.lower() or 'xml' in feed_url.lower())
        
        result = {
            'feed_url': feed_url,
            'status': 'valid' if is_valid else 'invalid',
            'message': 'Feed URL appears valid' if is_valid else 'Feed URL format may be invalid'
        }
        
        logger.info(f"✅ Single feed validation completed: {feed_url}")
        return {
            'success': True,
            'message': 'Feed validation completed',
            'result': result,
            'admin': 'api_key_authenticated'
        }
        
    except Exception as e:
        logger.error(f"❌ Admin single feed validation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to validate feed',
                'message': str(e)
            }
        )


@router.get("/scheduler/status")
async def get_scheduler_status(
    request: Request,
    admin_access: bool = Depends(require_admin_access)
):
    """
    Get auto-scraping scheduler status
    Endpoint: GET /admin/scheduler/status
    """
    try:
        logger.info("🔧 Admin requesting scheduler status")
        
        scheduler = get_scheduler()
        status = scheduler.get_scheduler_status()
        
        logger.info(f"✅ Scheduler status retrieved: {status.get('status', 'unknown')}")
        return {
            'success': True,
            'scheduler_status': status,
            'admin': 'api_key_authenticated'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get scheduler status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get scheduler status',
                'message': str(e)
            }
        )


@router.post("/scrape")
async def trigger_scraping(
    llm_model: str = Query('claude', description="LLM model to use: claude, gemini, or huggingface"),
    scrape_frequency: int = Query(1, description="Scrape frequency in days: 1 (daily), 7 (weekly), 30 (monthly)"),
    admin_access: bool = Depends(require_admin_access)
):
    """
    Trigger admin-initiated scraping with LLM model selection and frequency filtering
    """
    # Validate admin authentication
    ADMIN_API_KEY = os.getenv('ADMIN_API_KEY')
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    logger.info(f"🔧 Admin scraping triggered with model: {llm_model}, frequency: {scrape_frequency} days")
    
    # Validate frequency parameter
    if scrape_frequency not in [1, 7, 30]:
        raise HTTPException(
            status_code=400,
            detail="Invalid scrape_frequency. Must be 1 (daily), 7 (weekly), or 30 (monthly)"
        )
    
    # Validate LLM model parameter
    if llm_model not in ['claude', 'gemini', 'huggingface','ollama']:
        raise HTTPException(
            status_code=400,
            detail="Invalid llm_model. Must be 'claude', 'gemini', 'huggingface', or 'ollama'"
        )
    
    try:
        # Get database service
        from db_service import get_database_service
        db = get_database_service()
        
        # Initialize scraper and admin interface
        from crawl4ai_scraper import AdminScrapingInterface
        admin_interface = AdminScrapingInterface(db)
        
        # Run scraping with selected model and frequency
        result = await admin_interface.initiate_scraping(
            admin_email="admin@vidyagam.com",
            llm_model=llm_model,
            scrape_frequency=scrape_frequency
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Admin scraping failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ENHANCED ARTICLE MANAGEMENT (NEW) ====================

@router.get("/articles/filtered")
async def get_filtered_articles(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category_id: Optional[int] = Query(None),
    publisher_id: Optional[int] = Query(None),
    llm_model: Optional[str] = Query(None),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    content_type_id: Optional[int] = Query(None),
    search_query: Optional[str] = Query(None),
    admin_access: bool = Depends(require_admin_access)
):
    """
    Get filtered articles with advanced search options
    """
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    try:
        db = get_database_service()
        
        # Build dynamic WHERE clause
        where_clauses = []
        params = []
        
        if category_id:
            where_clauses.append("a.category_id = %s")
            params.append(category_id)
        
        if publisher_id:
            where_clauses.append("a.publisher_id = %s")
            params.append(publisher_id)
        
        if llm_model:
            where_clauses.append("a.llm_processed = %s")
            params.append(llm_model)
        
        if from_date:
            where_clauses.append("a.scraped_date >= %s")
            params.append(from_date)
        
        if to_date:
            where_clauses.append("a.scraped_date <= %s")
            params.append(to_date)
        
        if content_type_id:
            where_clauses.append("a.content_type_id = %s")
            params.append(content_type_id)
        
        if search_query:
            where_clauses.append("(a.title ILIKE %s OR a.summary ILIKE %s)")
            search_pattern = f"%{search_query}%"
            params.extend([search_pattern, search_pattern])
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Get total count
        count_query = f"""
            SELECT COUNT(*) as total
            FROM articles a
            WHERE {where_sql}
        """
        count_result = db.execute_query(count_query, tuple(params), fetch_all=True)
        total = count_result[0]['total'] if count_result and len(count_result) > 0 else 0
        
        # Get paginated results with joins
        offset = (page - 1) * page_size
        params.extend([page_size, offset])
        
        articles_query = f"""
            SELECT 
                a.id, a.title, a.url, a.summary, a.author, a.source,
                a.significance_score, a.complexity_level, a.reading_time,
                a.published_date, a.scraped_date, CAST(a.keywords AS TEXT), a.llm_processed,
                a.content_type_id, a.category_id, a.publisher_id,
                a.image_url, a.is_yt_shorts, a.is_insta_reels,
                c.name as category_name,
                p.publisher_name,
                ct.name as content_type_name
            FROM articles a
            LEFT JOIN ai_categories_master c ON a.category_id = c.id
            LEFT JOIN publishers_master p ON a.publisher_id = p.id
            LEFT JOIN content_types ct ON a.content_type_id = ct.id
            WHERE {where_sql}
            ORDER BY a.scraped_date DESC
            LIMIT %s OFFSET %s
        """
        
        articles = db.execute_query(articles_query, tuple(params), fetch_all=True)
        
        return {
            "success": True,
            "articles": articles or [],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch filtered articles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/articles/{article_id}")
async def delete_article(
    article_id: int,
    admin_access: bool = Depends(require_admin_access)
):
    """
    Delete a single article by ID
    """
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    try:
        db = get_database_service()
        
        # Check if article exists
        check_query = "SELECT id FROM articles WHERE id = %s"
        article = db.execute_query(check_query, (article_id,), fetch_all=True)
        
        if not article or len(article) == 0:
            raise HTTPException(status_code=404, detail=f"Article {article_id} not found")
        
        # Delete article
        delete_query = "DELETE FROM articles WHERE id = %s"
        db.execute_query(delete_query, (article_id,), fetch_all=True)
        
        logger.info(f"✅ Article {article_id} deleted by admin")
        
        return {
            "success": True,
            "message": f"Article {article_id} deleted successfully",
            "article_id": article_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete article {article_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/articles/bulk-delete")
async def bulk_delete_articles(
    article_ids: List[int] = Body(..., embed=True),
    admin_access: bool = Depends(require_admin_access)
):
    """
    Delete multiple articles by IDs
    """
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    try:
        if not article_ids:
            raise HTTPException(status_code=400, detail="No article IDs provided")
        
        db = get_database_service()
        
        # Use parameterized query with IN clause
        placeholders = ','.join(['%s'] * len(article_ids))
        delete_query = f"DELETE FROM articles WHERE id IN ({placeholders}) RETURNING id"
        
        deleted = db.execute_query(delete_query, tuple(article_ids), fetch_all=True)
        deleted_count = len(deleted) if deleted else 0
        
        logger.info(f"✅ Bulk deleted {deleted_count} articles by admin")
        
        return {
            "success": True,
            "message": f"{deleted_count} articles deleted successfully",
            "deleted_count": deleted_count,
            "requested_count": len(article_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Bulk delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENHANCED SOURCE MANAGEMENT (NEW) ====================

@router.get("/sources/by-type")
async def get_sources_by_type(
    content_type: str = Query(..., description="blogs, podcasts, or videos"),
    admin_access: bool = Depends(require_admin_access)
):
    """
    Get sources filtered by content type (blogs/podcasts/videos)
    Returns from ai_sources (blogs), ai_podcasts, or ai_videos table
    """
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    try:
        # Map content type to table
        if content_type.lower() == "podcasts":
            table = "ai_podcasts"
            url_field = "url"
        elif content_type.lower() == "videos":
            table = "ai_videos"
            url_field = "url"
        elif content_type.lower() == "blogs":
            table = "ai_sources"
            url_field = "rss_url"
        else:
            raise HTTPException(status_code=400, detail="Invalid content_type. Use: blogs, podcasts, or videos")
        
        db = get_database_service()
        
        # Query structure varies slightly by table
        if table == "ai_sources":
            query = f"""
                SELECT 
                    s.id, s.name, s.rss_url as url, s.website, s.category_id, 
                    s.priority, s.enabled, s.is_active, 
                    s.scrape_frequency_days, s.updated_date as last_scraped_date,
                    c.name as category_name
                FROM {table} s
                LEFT JOIN ai_categories_master c ON s.category_id = c.id
                ORDER BY s.priority DESC, s.name ASC
            """
        else:
            query = f"""
                SELECT 
                    s.id, s.name, s.{url_field} as url, s.category_id, s.priority,
                    s.is_active, s.scraped_status, s.last_scraped_date,
                    c.name as category_name
                FROM {table} s
                LEFT JOIN ai_categories_master c ON s.category_id = c.id
                ORDER BY s.priority DESC, s.name ASC
            """
        
        sources = db.execute_query(query, fetch_all=True)
        
        return {
            "success": True,
            "content_type": content_type,
            "sources": sources or [],
            "count": len(sources) if sources else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to fetch sources by type: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sources/bulk-update")
async def bulk_update_sources(
    source_ids: List[int] = Body(...),
    enabled: Optional[bool] = Body(None),
    priority: Optional[int] = Body(None),
    scrape_frequency_days: Optional[int] = Body(None),
    content_type: str = Body("blogs", description="blogs, podcasts, or videos"),
    admin_access: bool = Depends(require_admin_access)
):
    """
    Bulk update sources (enable/disable, change priority, etc.)
    Works with ai_sources, ai_podcasts, or ai_videos tables
    """
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    try:
        if not source_ids:
            raise HTTPException(status_code=400, detail="No source IDs provided")
        
        # Map content type to table
        if content_type.lower() == "podcasts":
            table = "ai_podcasts"
        elif content_type.lower() == "videos":
            table = "ai_videos"
        else:
            table = "ai_sources"
        
        # Build dynamic UPDATE clause
        update_fields = []
        params = []
        
        if enabled is not None:
            update_fields.append("is_active = %s")
            params.append(enabled)
            # For ai_sources, also update enabled field
            if table == "ai_sources":
                update_fields.append("enabled = %s")
                params.append(enabled)
        
        if priority is not None:
            update_fields.append("priority = %s")
            params.append(priority)
        
        if scrape_frequency_days is not None and table == "ai_sources":
            update_fields.append("scrape_frequency_days = %s")
            params.append(scrape_frequency_days)
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="No update fields provided")
        
        # Add updated timestamp
        if table == "ai_sources":
            update_fields.append("updated_date = CURRENT_TIMESTAMP")
        
        db = get_database_service()
        
        # Update query
        placeholders = ','.join(['%s'] * len(source_ids))
        params.extend(source_ids)
        
        update_query = f"""
            UPDATE {table}
            SET {', '.join(update_fields)}
            WHERE id IN ({placeholders})
            RETURNING id
        """
        
        updated = db.execute_query(update_query, tuple(params), fetch_all=True)
        updated_count = len(updated) if updated else 0
        
        logger.info(f"✅ Bulk updated {updated_count} sources in {table} by admin")
        
        return {
            "success": True,
            "message": f"{updated_count} sources updated successfully",
            "updated_count": updated_count,
            "requested_count": len(source_ids),
            "table": table
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Bulk update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sources/bulk-delete")
async def bulk_delete_sources(
    source_ids: List[int] = Body(..., embed=True),
    content_type: str = Body("blogs", description="blogs, podcasts, or videos"),
    admin_access: bool = Depends(require_admin_access)
):
    """
    Bulk delete sources (soft delete by setting is_active=false)
    Works with ai_sources, ai_podcasts, or ai_videos tables
    """
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    try:
        if not source_ids:
            raise HTTPException(status_code=400, detail="No source IDs provided")
        
        # Map content type to table
        if content_type.lower() == "podcasts":
            table = "ai_podcasts"
        elif content_type.lower() == "videos":
            table = "ai_videos"
        else:
            table = "ai_sources"
        
        db = get_database_service()
        
        # Soft delete (set is_active = false)
        placeholders = ','.join(['%s'] * len(source_ids))
        
        if table == "ai_sources":
            delete_query = f"""
                UPDATE {table}
                SET is_active = false, updated_date = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
                RETURNING id
            """
        else:
            delete_query = f"""
                UPDATE {table}
                SET is_active = false
                WHERE id IN ({placeholders})
                RETURNING id
            """
        
        deleted = db.execute_query(delete_query, tuple(source_ids), fetch_all=True)
        deleted_count = len(deleted) if deleted else 0
        
        logger.info(f"✅ Bulk deleted {deleted_count} sources from {table} by admin")
        
        return {
            "success": True,
            "message": f"{deleted_count} sources deleted successfully",
            "deleted_count": deleted_count,
            "requested_count": len(source_ids),
            "table": table
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Bulk delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CATEGORIES MANAGEMENT (NEW) ====================

@router.get("/categories/all")
async def get_all_categories(
    admin_access: bool = Depends(require_admin_access)
):
    """
    Get all categories for admin management with article/source counts
    """
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    try:
        db = get_database_service()
        
        query = """
            SELECT 
                c.id, c.name, c.description, c.priority, 
                c.category_label, c.is_active,
                COUNT(DISTINCT a.id) as article_count,
                COUNT(DISTINCT s.id) as source_count
            FROM ai_categories_master c
            LEFT JOIN articles a ON c.id = a.category_id
            LEFT JOIN ai_sources s ON c.id = s.category_id
            GROUP BY c.id, c.name, c.description, c.priority, c.category_label, c.is_active
            ORDER BY c.priority DESC, c.name ASC
        """
        
        categories = db.execute_query(query, fetch_all=True)
        
        return {
            "success": True,
            "categories": categories or [],
            "count": len(categories) if categories else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SCRAPING JOB MONITORING (NEW) ====================

@router.get("/scraping/active-jobs")
async def get_active_scraping_jobs(
    admin_access: bool = Depends(require_admin_access)
):
    """
    Get currently active scraping jobs from scheduler
    """
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    try:
        # Get scheduler status
        from scheduler_service import get_scheduler
        scheduler = get_scheduler()
        
        if not scheduler:
            return {
                "success": True,
                "active_jobs": [],
                "scheduler_running": False
            }
        
        # Get job information
        jobs = []
        for job in scheduler.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
                "args": str(job.args) if job.args else None
            })
        
        return {
            "success": True,
            "active_jobs": jobs,
            "scheduler_running": scheduler.is_running and scheduler.scheduler.running,
            "job_count": len(jobs)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch active jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tavily/search")
async def trigger_tavily_search(
    query: str = Query(..., description="Search query for Tavily"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum results (1-50)"),
    enrich_with_llm: bool = Query(False, description="Enhance with LLM processing"),
    llm_model: str = Query('gemini', description="LLM model: claude, gemini, or ollama"),
    admin_access: bool = Depends(require_admin_access)
):
    """
    Trigger Tavily search for AI articles with LLM enrichment option
    
    Free tier: 1,000 searches/month
    """
    # Validate admin authentication
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    logger.info(f"🔧 Admin Tavily search triggered: '{query}' with model: {llm_model}, enrich: {enrich_with_llm}")
    
    # Validate LLM model parameter
    if llm_model not in ['claude', 'gemini', 'ollama']:
        raise HTTPException(
            status_code=400,
            detail="Invalid llm_model. Must be 'claude', 'gemini', or 'ollama'"
        )
    
    try:
        # Initialize scraper directly (no AdminScrapingInterface needed for Tavily)
        scraper = Crawl4AIScraper()
        
        # Run Tavily search with selected model
        result = await scraper.search_and_insert_tavily_articles(
            query=query,
            max_results=max_results,
            enrich_with_llm=enrich_with_llm,
            llm_model=llm_model,
            include_domains=None,
            exclude_domains=None
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Admin Tavily search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tavily/searches")
async def get_tavily_search_history(
    limit: int = Query(50, description="Number of recent searches to retrieve"),
    admin_access: bool = Depends(require_admin_access)
):
    """
    Get history of Tavily searches from tavily_searches table
    """
    # Validate admin authentication
    if not admin_access:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    try:
        from db_service import get_database_service
        db = get_database_service()
        
        searches = db.execute_query(
            """
            SELECT id, query, max_results, enrich_with_llm, llm_model,
                   articles_found, articles_inserted, articles_skipped,
                   created_at, completed_at, error_message,
                   search_params
            FROM tavily_searches
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
            fetch_all=True
        )
        
        return {
            "success": True,
            "searches": searches or [],
            "count": len(searches) if searches else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch search history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))