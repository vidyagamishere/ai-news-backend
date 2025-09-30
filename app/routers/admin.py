#!/usr/bin/env python3
"""
Admin router for modular FastAPI architecture
Handles admin-only endpoints for source management and system control
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import UserResponse
from app.dependencies.auth import get_current_user
from app.services.content_service import ContentService
from db_service import get_database_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin")


def get_content_service() -> ContentService:
    """Dependency to get ContentService instance"""
    return ContentService()


def require_admin_user(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Dependency to ensure user is admin"""
    if not current_user or not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail={
                'error': 'Admin access required',
                'message': 'Only admin users can access this endpoint'
            }
        )
    return current_user


@router.get("/sources")
async def get_admin_sources(
    admin_user: UserResponse = Depends(require_admin_user),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get all AI sources for admin management
    Endpoint: GET /admin/sources
    """
    try:
        logger.info(f"🔧 Admin sources requested by: {admin_user.email}")
        
        db = get_database_service()
        
        # Get all sources with category via JOIN relationship (including inactive ones for admin view)
        sources_query = """
            SELECT s.id, s.name, s.rss_url, s.website, s.content_type, 
                   COALESCE(c.name, 'general') as category, 
                   s.priority, s.enabled, s.is_active,
                   s.created_at, s.updated_at, s.ai_topic_id
            FROM ai_sources s 
            LEFT JOIN ai_topics t ON s.ai_topic_id = t.id 
            LEFT JOIN ai_categories_master c ON t.category_id = c.id
            ORDER BY s.priority ASC, s.name ASC
        """
        
        sources = db.execute_query(sources_query)
        
        processed_sources = []
        for source in sources:
            source_dict = dict(source)
            # Convert timestamps to ISO format
            for field in ['created_at', 'updated_at']:
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
            'admin': admin_user.email
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
    admin_user: UserResponse = Depends(require_admin_user)
):
    """
    Add new AI source
    Endpoint: POST /admin/sources/add
    """
    try:
        logger.info(f"🔧 Admin adding source: {admin_user.email}")
        
        db = get_database_service()
        
        # Insert new source using ai_topic_id relationship
        insert_query = """
            INSERT INTO ai_sources (name, rss_url, website, content_type, ai_topic_id, priority, enabled, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        values = (
            source_data.get('name'),
            source_data.get('rss_url'),
            source_data.get('website', ''),
            source_data.get('content_type', 'blogs'),
            source_data.get('ai_topic_id'),
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
            'admin': admin_user.email
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
    admin_user: UserResponse = Depends(require_admin_user)
):
    """
    Update existing AI source
    Endpoint: POST /admin/sources/update
    """
    try:
        logger.info(f"🔧 Admin updating source: {admin_user.email}")
        
        db = get_database_service()
        source_id = update_data.get('id')
        
        if not source_id:
            raise HTTPException(status_code=400, detail="Source ID required")
        
        # Update source using ai_topic_id relationship
        update_query = """
            UPDATE ai_sources 
            SET name = %s, rss_url = %s, website = %s, content_type = %s, 
                ai_topic_id = %s, priority = %s, enabled = %s, is_active = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        
        values = (
            update_data.get('name'),
            update_data.get('rss_url'),
            update_data.get('website', ''),
            update_data.get('content_type', 'blogs'),
            update_data.get('ai_topic_id'),
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
            'admin': admin_user.email
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
    admin_user: UserResponse = Depends(require_admin_user)
):
    """
    Delete AI source
    Endpoint: POST /admin/sources/delete
    """
    try:
        logger.info(f"🔧 Admin deleting source: {admin_user.email}")
        
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
            'admin': admin_user.email
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
    admin_user: UserResponse = Depends(require_admin_user)
):
    """
    Validate all RSS feeds
    Endpoint: POST /admin/validate-all-feeds
    """
    try:
        logger.info(f"🔧 Admin validating all feeds: {admin_user.email}")
        
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
            'admin': admin_user.email
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
    admin_user: UserResponse = Depends(require_admin_user)
):
    """
    Validate single RSS feed
    Endpoint: POST /admin/validate-feed
    """
    try:
        logger.info(f"🔧 Admin validating single feed: {admin_user.email}")
        
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
            'admin': admin_user.email
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