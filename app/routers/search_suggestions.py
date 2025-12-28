#!/usr/bin/env python3
"""
Search Suggestions & Autocomplete Router
Provides:
- Curated search questions per category
- Autocomplete suggestions
- Trending searches
- Search history tracking
"""

import os
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from datetime import datetime, timedelta

from app.dependencies.auth import get_current_user_optional
from app.models.schemas import UserResponse

logger = logging.getLogger(__name__)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

router = APIRouter()

# =====================================================
# REQUEST/RESPONSE MODELS
# =====================================================

class SearchQuestion(BaseModel):
    """Curated search question"""
    id: int
    question_text: str
    question_type: str  # beginner, advanced, trending, featured
    category_name: Optional[str] = None
    keywords: List[str] = []
    priority: int = 0
    click_count: int = 0

class AutocompleteSuggestion(BaseModel):
    """Autocomplete suggestion"""
    text: str
    type: str  # curated, trending, popular, related
    category: Optional[str] = None
    relevance: float = 1.0

class TrendingSearch(BaseModel):
    """Trending search term"""
    term: str
    category: Optional[str] = None
    search_count: int
    trending_score: float
    last_searched: datetime

class SearchHistoryCreate(BaseModel):
    """Log search query"""
    query: str
    category_id: Optional[int] = None
    results_count: int = 0
    session_id: Optional[str] = None

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_db():
    """Get database service"""
    from db_service import get_database_service
    return get_database_service()

# =====================================================
# ENDPOINTS
# =====================================================

@router.get("/search-suggestions/questions")
async def get_search_questions(
    category_id: Optional[int] = Query(None, description="Filter by category ID"),
    question_type: Optional[str] = Query(None, description="Filter by type: beginner, advanced, trending, featured"),
    limit: int = Query(10, ge=1, le=50, description="Max questions to return"),
    current_user: Optional[UserResponse] = Depends(get_current_user_optional)
):
    """
    Get curated search questions to guide users
    
    Usage:
    - Show on empty search bar as suggestions
    - Display as "Popular Searches" or "Try Searching For"
    - Category-specific quick searches
    
    Returns questions sorted by priority and popularity
    """
    try:
        logger.info(f"📋 Getting search questions - Category: {category_id}, Type: {question_type}")
        
        db = get_db()
        
        # Build query
        query = """
            SELECT 
                sq.id,
                sq.question_text,
                sq.question_type,
                sq.priority,
                sq.click_count,
                sq.keywords,
                c.name as category_name
            FROM search_questions sq
            LEFT JOIN ai_categories_master c ON sq.category_id = c.id
            WHERE sq.is_active = true
        """
        
        params = []
        
        # Add filters
        if category_id is not None:
            query += " AND sq.category_id = %s"
            params.append(category_id)
        
        if question_type:
            query += " AND sq.question_type = %s"
            params.append(question_type)
        
        # Order by priority (high first) and popularity
        query += """
            ORDER BY 
                sq.priority DESC,
                sq.click_count DESC,
                sq.last_used_at DESC NULLS LAST
            LIMIT %s
        """
        params.append(limit)
        
        results = db.execute_query(query, tuple(params), fetch_all=True)
        
        questions = [
            SearchQuestion(
                id=r['id'],
                question_text=r['question_text'],
                question_type=r['question_type'],
                category_name=r.get('category_name'),
                keywords=r.get('keywords', []) or [],
                priority=r.get('priority', 0),
                click_count=r.get('click_count', 0)
            )
            for r in results
        ]
        
        logger.info(f"✅ Returned {len(questions)} search questions")
        
        return {
            'questions': [q.dict() for q in questions],
            'count': len(questions),
            'category_id': category_id,
            'question_type': question_type
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get search questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-suggestions/autocomplete")
async def get_autocomplete_suggestions(
    query: str = Query(..., min_length=2, description="Search query prefix"),
    category_id: Optional[int] = Query(None, description="Filter by category"),
    limit: int = Query(8, ge=1, le=20, description="Max suggestions"),
    current_user: Optional[UserResponse] = Depends(get_current_user_optional)
):
    """
    Get autocomplete suggestions as user types
    
    Combines:
    - Curated questions matching the query
    - Popular searches from user history
    - Trending searches
    
    Returns suggestions sorted by relevance
    """
    try:
        logger.info(f"🔍 Autocomplete for: '{query}' in category {category_id}")
        
        db = get_db()
        
        # Use trigram similarity for fuzzy matching
        search_pattern = f"%{query.lower()}%"
        
        query_sql = """
            WITH combined_suggestions AS (
                -- Curated questions
                SELECT 
                    question_text as text,
                    'curated' as type,
                    c.name as category,
                    (sq.priority + sq.click_count * 0.1) as relevance,
                    1 as source_priority
                FROM search_questions sq
                LEFT JOIN ai_categories_master c ON sq.category_id = c.id
                WHERE sq.is_active = true
                AND LOWER(sq.question_text) LIKE %s
                {category_filter}
                
                UNION ALL
                
                -- Autocomplete suggestions
                SELECT 
                    suggestion_text as text,
                    suggestion_type as type,
                    c.name as category,
                    relevance_score as relevance,
                    2 as source_priority
                FROM autocomplete_suggestions acs
                LEFT JOIN ai_categories_master c ON acs.category_id = c.id
                WHERE acs.is_active = true
                AND LOWER(acs.suggestion_text) LIKE %s
                {category_filter}
                
                UNION ALL
                
                -- Popular searches
                SELECT 
                    search_term as text,
                    'popular' as type,
                    c.name as category,
                    trending_score as relevance,
                    3 as source_priority
                FROM popular_searches ps
                LEFT JOIN ai_categories_master c ON ps.category_id = c.id
                WHERE LOWER(ps.search_term) LIKE %s
                AND ps.last_searched > NOW() - INTERVAL '30 days'
                {category_filter}
            )
            SELECT DISTINCT ON (LOWER(text))
                text,
                type,
                category,
                relevance
            FROM combined_suggestions
            ORDER BY 
                LOWER(text),
                source_priority,
                relevance DESC
            LIMIT %s
        """
        
        # Add category filter if provided
        category_filter = "AND sq.category_id = %s" if category_id else ""
        query_sql = query_sql.replace("{category_filter}", category_filter)
        
        # Build params
        params = [search_pattern, search_pattern, search_pattern]
        if category_id:
            params.extend([category_id, category_id, category_id])
        params.append(limit)
        
        results = db.execute_query(query_sql, tuple(params), fetch_all=True)
        
        suggestions = [
            AutocompleteSuggestion(
                text=r['text'],
                type=r['type'],
                category=r.get('category'),
                relevance=float(r.get('relevance', 1.0))
            )
            for r in results
        ]
        
        logger.info(f"✅ Returned {len(suggestions)} autocomplete suggestions")
        
        return {
            'suggestions': [s.dict() for s in suggestions],
            'count': len(suggestions),
            'query': query
        }
        
    except Exception as e:
        logger.error(f"❌ Autocomplete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-suggestions/trending")
async def get_trending_searches(
    category_id: Optional[int] = Query(None, description="Filter by category"),
    days: int = Query(7, ge=1, le=30, description="Look back period in days"),
    limit: int = Query(10, ge=1, le=50),
    current_user: Optional[UserResponse] = Depends(get_current_user_optional)
):
    """
    Get trending searches based on recent activity
    
    Shows what other users are searching for
    Can be displayed as "Trending Now" or "Popular This Week"
    """
    try:
        logger.info(f"📈 Getting trending searches for last {days} days")
        
        db = get_db()
        
        query = """
            SELECT 
                ps.search_term as term,
                c.name as category,
                ps.search_count,
                ps.trending_score,
                ps.last_searched
            FROM popular_searches ps
            LEFT JOIN ai_categories_master c ON ps.category_id = c.id
            WHERE ps.last_searched > NOW() - INTERVAL '%s days'
        """
        
        params = [days]
        
        if category_id is not None:
            query += " AND ps.category_id = %s"
            params.append(category_id)
        
        query += """
            ORDER BY ps.trending_score DESC, ps.search_count DESC
            LIMIT %s
        """
        params.append(limit)
        
        results = db.execute_query(query, tuple(params), fetch_all=True)
        
        trending = [
            TrendingSearch(
                term=r['term'],
                category=r.get('category'),
                search_count=r['search_count'],
                trending_score=float(r['trending_score']),
                last_searched=r['last_searched']
            )
            for r in results
        ]
        
        logger.info(f"✅ Returned {len(trending)} trending searches")
        
        return {
            'trending': [t.dict() for t in trending],
            'count': len(trending),
            'period_days': days
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get trending searches: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-suggestions/track")
async def track_search(
    search_data: SearchHistoryCreate,
    request: Request,
    current_user: Optional[UserResponse] = Depends(get_current_user_optional)
):
    """
    Track a search query for analytics and trending
    
    Call this endpoint whenever a user performs a search
    Updates:
    - search_history table
    - popular_searches (via trigger)
    - Trending scores
    """
    try:
        db = get_db()
        
        # Get client info
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get('user-agent', '')
        
        query = """
            INSERT INTO search_history (
                user_id,
                search_query,
                category_id,
                results_count,
                session_id,
                ip_address,
                user_agent,
                search_timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING id
        """
        
        params = (
            current_user.id if current_user else None,
            search_data.query,
            search_data.category_id,
            search_data.results_count,
            search_data.session_id,
            client_ip,
            user_agent
        )
        
        result = db.execute_query(query, params, fetch_one=True)
        
        if DEBUG:
            logger.info(f"📊 Tracked search: '{search_data.query}' - ID: {result['id']}")
        
        return {
            'success': True,
            'search_id': result['id'],
            'message': 'Search tracked successfully'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to track search: {str(e)}")
        # Don't fail the search if tracking fails
        return {
            'success': False,
            'message': 'Search tracking failed',
            'error': str(e)
        }


@router.post("/search-suggestions/track-question-click")
async def track_question_click(
    question_id: int,
    session_id: Optional[str] = None,
    category_context: Optional[str] = None,
    current_user: Optional[UserResponse] = Depends(get_current_user_optional)
):
    """
    Track when a user clicks on a curated search question
    
    Used for analytics and improving question rankings
    """
    try:
        db = get_db()
        
        query = """
            INSERT INTO search_question_analytics (
                question_id,
                user_id,
                session_id,
                category_context,
                clicked_at
            ) VALUES (%s, %s, %s, %s, NOW())
            RETURNING id
        """
        
        params = (
            question_id,
            current_user.id if current_user else None,
            session_id,
            category_context
        )
        
        result = db.execute_query(query, params, fetch_one=True)
        
        logger.info(f"📊 Tracked question click: {question_id}")
        
        return {
            'success': True,
            'analytics_id': result['id']
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to track question click: {str(e)}")
        return {
            'success': False,
            'message': 'Click tracking failed'
        }


@router.get("/search-suggestions/analytics")
async def get_search_analytics(
    days: int = Query(7, ge=1, le=90),
    category_id: Optional[int] = Query(None),
    current_user: Optional[UserResponse] = Depends(get_current_user_optional)
):
    """
    Get search analytics (admin/analytics endpoint)
    
    Returns:
    - Total searches
    - Unique users
    - Popular queries
    - Trending topics
    """
    try:
        db = get_db()
        
        # Basic analytics
        analytics_query = """
            SELECT 
                COUNT(*) as total_searches,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT search_query) as unique_queries,
                AVG(results_count) as avg_results_per_search
            FROM search_history
            WHERE search_timestamp > NOW() - INTERVAL '%s days'
        """
        
        params = [days]
        if category_id:
            analytics_query += " AND category_id = %s"
            params.append(category_id)
        
        analytics = db.execute_query(analytics_query, tuple(params), fetch_one=True)
        
        # Top searches
        top_searches_query = """
            SELECT search_query, COUNT(*) as count
            FROM search_history
            WHERE search_timestamp > NOW() - INTERVAL '%s days'
        """
        
        if category_id:
            top_searches_query += " AND category_id = %s"
        
        top_searches_query += """
            GROUP BY search_query
            ORDER BY count DESC
            LIMIT 20
        """
        
        top_searches = db.execute_query(top_searches_query, tuple(params), fetch_all=True)
        
        return {
            'period_days': days,
            'analytics': analytics,
            'top_searches': top_searches
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-suggestions/admin/refresh-autocomplete")
async def refresh_autocomplete(
    current_user: Optional[UserResponse] = Depends(get_current_user_optional)
):
    """
    Admin endpoint: Refresh autocomplete suggestions
    
    Regenerates suggestions from popular searches
    Run this periodically (e.g., daily via cron)
    """
    try:
        db = get_db()
        
        # Call the stored procedure
        query = "SELECT refresh_autocomplete_suggestions()"
        db.execute_query(query)
        
        logger.info("✅ Autocomplete suggestions refreshed")
        
        return {
            'success': True,
            'message': 'Autocomplete suggestions refreshed successfully'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to refresh autocomplete: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
