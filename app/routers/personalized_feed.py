#!/usr/bin/env python3
"""
Enhanced Personalized feed endpoints for mobile-first interface
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from ..models.schemas import (
    ContentFilterRequest, 
    PersonalizedFeedResponse, 
    GroupedContentResponse,
    Article,
    UserPreferences
)
from ..dependencies.database import get_database_service
from .enhanced_auth import get_current_user
from ..services.content_filtering_service import content_filtering_service, FilterCriteria
from db_service import DatabaseService

logger = logging.getLogger(__name__)
router = APIRouter()


def parse_time_filter(time_filter: str) -> Optional[datetime]:
    """Convert time filter string to datetime"""
    if time_filter == "All Time":
        return None
    
    now = datetime.now()
    if time_filter == "Last 24 hours":
        return now - timedelta(days=1)
    elif time_filter == "Last Week":
        return now - timedelta(days=7)
    elif time_filter == "Last Month":
        return now - timedelta(days=30)
    elif time_filter == "Last Year":
        return now - timedelta(days=365)
    else:
        return now - timedelta(days=7)  # Default to last week


def filter_content_by_criteria(
    db: DatabaseService,
    filter_request: ContentFilterRequest
) -> List[Article]:
    """Filter content based on user criteria"""
    
    # Base query for articles
    base_query = """
        SELECT DISTINCT a.*,
               ct.name as content_type_label,
               c.name as category_label,
               s.name as source_name
        FROM articles a
        LEFT JOIN content_types ct ON a.content_type_id = ct.id
        LEFT JOIN ai_categories_master c ON a.category_id = c.id
        LEFT JOIN ai_sources s ON a.source = s.name
        WHERE 1=1
    """
    
    params = []
    
    # Time filter
    if filter_request.time_filter and filter_request.time_filter != "All Time":
        time_threshold = parse_time_filter(filter_request.time_filter)
        if time_threshold:
            base_query += " AND a.published_date >= %s"
            params.append(time_threshold)
    
    # Content type filter
    if filter_request.content_types:
        placeholders = ",".join(["%s"] * len(filter_request.content_types))
        base_query += f" AND ct.name IN ({placeholders})"
        params.extend(filter_request.content_types)
    
    # Publisher filter
    if filter_request.publishers:
        placeholders = ",".join(["%s"] * len(filter_request.publishers))
        base_query += f" AND s.name IN ({placeholders})"
        params.extend(filter_request.publishers)
    
    # Interest/topic filter
    if filter_request.interests:
        placeholders = ",".join(["%s"] * len(filter_request.interests))
        base_query += f" AND at.name IN ({placeholders})"
        params.extend(filter_request.interests)
    
    # Search query filter
    if filter_request.search_query:
        search_term = f"%{filter_request.search_query.lower()}%"
        base_query += """
            AND (
                LOWER(a.title) LIKE %s OR 
                LOWER(a.description) LIKE %s OR
                LOWER(at.name) LIKE %s OR
                LOWER(s.name) LIKE %s
            )
        """
        params.extend([search_term, search_term, search_term, search_term])
    
    # Order by significance and date
    base_query += " ORDER BY a.significance_score DESC, a.published_date DESC"
    
    # Limit
    if filter_request.limit:
        base_query += " LIMIT %s"
        params.append(filter_request.limit)
    
    try:
        results = db.execute_query(base_query, params)
        
        articles = []
        for row in results:
            article = Article(
                id=row['id'],
                source=row.get('source'),
                title=row.get('title'),
                url=row.get('url'),
                published_date=row.get('published_date'),
                description=row.get('description'),
                content_hash=row.get('content_hash'),
                significance_score=row.get('significance_score', 5),
                scraped_date=row.get('scraped_date'),
                reading_time=row.get('reading_time', 3),
                image_url=row.get('image_url'),
                keywords=row.get('keywords'),
                content_type_id=row.get('content_type_id'),
                content_type_label=row.get('content_type_label'),
                category_id=row.get('category_id'),
                category_label=row.get('category_label')
            )
            articles.append(article)
        
        return articles
        
    except Exception as e:
        logger.error(f"Error filtering content: {str(e)}")
        return []


def group_content_by_interests(
    articles: List[Article], 
    user_interests: List[str],
    include_uncategorized: bool = False
) -> List[GroupedContentResponse]:
    """Group articles by user interests"""
    
    grouped = {}
    uncategorized = []
    
    # Group by interests
    for interest in user_interests:
        interest_articles = [
            article for article in articles 
            if article.ai_topic_label == interest
        ]
        if interest_articles:
            grouped[interest] = interest_articles
    
    # Handle uncategorized content for search results
    if include_uncategorized:
        uncategorized = [
            article for article in articles 
            if not article.ai_topic_label or article.ai_topic_label not in user_interests
        ]
        if uncategorized:
            grouped['Other Relevant Results'] = uncategorized
    
    # Convert to response format
    response_groups = []
    for category, items in grouped.items():
        response_groups.append(GroupedContentResponse(
            category=category,
            items=items,
            count=len(items)
        ))
    
    return response_groups


@router.post("/personalized-feed", response_model=PersonalizedFeedResponse)
async def get_personalized_feed(
    filter_request: ContentFilterRequest,
    db: DatabaseService = Depends(get_database_service),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get enhanced personalized content feed with advanced filtering"""
    
    try:
        # Create filter criteria
        criteria = FilterCriteria(
            interests=filter_request.interests or [],
            content_types=filter_request.content_types or [],
            publishers=filter_request.publishers or [],
            time_filter=filter_request.time_filter or "Last Week",
            search_query=filter_request.search_query or "",
            limit=filter_request.limit or 50,
            user_id=current_user.get('id') if current_user else None
        )
        
        # Get filtered content using enhanced service
        filtered_articles = content_filtering_service.get_filtered_content(db, criteria)
        
        # Determine if search is active
        search_active = bool(criteria.search_query.strip())
        
        # Group content by interests
        grouped_content_dict = content_filtering_service.group_content_by_interests(
            filtered_articles, 
            criteria.interests,
            include_uncategorized=search_active
        )
        
        # Convert to response format
        grouped_content = []
        for category, items in grouped_content_dict.items():
            # Convert articles to Article objects
            article_objects = []
            for item in items:
                article = Article(
                    id=item['id'],
                    source=item.get('source'),
                    title=item.get('title'),
                    url=item.get('url'),
                    published_date=item.get('published_date'),
                    description=item.get('description'),
                    content_hash=item.get('content_hash'),
                    significance_score=item.get('significance_score', 5),
                    scraped_date=item.get('scraped_date'),
                    reading_time=item.get('reading_time', 3),
                    image_url=item.get('image_url'),
                    keywords=item.get('keywords'),
                    content_type_id=item.get('content_type_id'),
                    content_type_label=item.get('content_type_label'),
                    category_id=item.get('category_id'),
                    category_label=item.get('category_label')
                )
                article_objects.append(article)
            
            grouped_content.append(GroupedContentResponse(
                category=category,
                items=article_objects,
                count=len(article_objects)
            ))
        
        # Generate welcome message
        if search_active:
            welcome_message = f"Showing results for: '{criteria.search_query}'"
        else:
            user_name = current_user.get('name', 'User') if current_user else 'User'
            welcome_message = f"Welcome back, {user_name}! Showing: {criteria.time_filter}"
        
        # User profile info
        user_profile = {
            "interests_count": len(criteria.interests),
            "content_types_count": len(criteria.content_types),
            "publishers_count": len(criteria.publishers),
            "time_filter": criteria.time_filter,
            "user_authenticated": current_user is not None
        }
        
        # Filters applied
        filters_applied = {
            "interests": criteria.interests,
            "content_types": criteria.content_types,
            "publishers": criteria.publishers,
            "time_filter": criteria.time_filter,
            "search_query": criteria.search_query
        }
        
        return PersonalizedFeedResponse(
            welcome_message=welcome_message,
            user_profile=user_profile,
            grouped_content=grouped_content,
            search_active=search_active,
            search_query=criteria.search_query if search_active else None,
            total_items=len(filtered_articles),
            filters_applied=filters_applied
        )
        
    except Exception as e:
        logger.error(f"Error generating personalized feed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate personalized feed",
                "message": str(e)
            }
        )


@router.get("/available-interests")
async def get_available_interests(
    db: DatabaseService = Depends(get_database_service)
):
    """Get list of available interests/topics"""
    try:
        query = """
            SELECT DISTINCT name 
            FROM ai_topics 
            WHERE is_active = TRUE 
            ORDER BY name
        """
        results = db.execute_query(query)
        interests = [row['name'] for row in results]
        
        return {
            "interests": interests,
            "count": len(interests)
        }
    except Exception as e:
        logger.error(f"Error getting interests: {str(e)}")
        return {"interests": [], "count": 0}


@router.get("/available-publishers")
async def get_available_publishers(
    db: DatabaseService = Depends(get_database_service)
):
    """Get list of available publishers"""
    try:
        query = """
            SELECT DISTINCT name 
            FROM ai_sources 
            WHERE enabled = TRUE 
            ORDER BY name
        """
        results = db.execute_query(query)
        publishers = [row['name'] for row in results]
        
        return {
            "publishers": publishers,
            "count": len(publishers)
        }
    except Exception as e:
        logger.error(f"Error getting publishers: {str(e)}")
        return {"publishers": [], "count": 0}


@router.get("/available-content-types")
async def get_available_content_types(
    db: DatabaseService = Depends(get_database_service)
):
    """Get list of available content types"""
    try:
        query = """
            SELECT DISTINCT name, display_name 
            FROM content_types 
            ORDER BY name
        """
        results = db.execute_query(query)
        content_types = [
            {"name": row['name'], "display_name": row['display_name']} 
            for row in results
        ]
        
        return {
            "content_types": content_types,
            "count": len(content_types)
        }
    except Exception as e:
        logger.error(f"Error getting content types: {str(e)}")
        return {"content_types": [], "count": 0}


@router.get("/recommendations")
async def get_recommendations(
    limit: int = Query(10, ge=1, le=50),
    current_user: Dict = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service)
):
    """Get personalized content recommendations"""
    try:
        recommendations = content_filtering_service.get_content_recommendations(
            db, current_user['id'], limit
        )
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations),
            "user_id": current_user['id']
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return {"recommendations": [], "count": 0}


@router.get("/trending")
async def get_trending_topics(
    time_filter: str = Query("Last Week", description="Time filter for trending analysis"),
    db: DatabaseService = Depends(get_database_service)
):
    """Get trending topics and content"""
    try:
        trending = content_filtering_service.get_trending_topics(db, time_filter)
        
        return {
            "trending_topics": trending,
            "count": len(trending),
            "time_filter": time_filter
        }
    except Exception as e:
        logger.error(f"Error getting trending topics: {str(e)}")
        return {"trending_topics": [], "count": 0}


@router.get("/search-suggestions")
async def get_search_suggestions(
    query: str = Query(..., min_length=2, description="Partial search query"),
    limit: int = Query(10, ge=1, le=20),
    db: DatabaseService = Depends(get_database_service)
):
    """Get search suggestions based on partial query"""
    try:
        # Get suggestions from titles, topics, and sources
        suggestions_query = """
            (SELECT DISTINCT title as suggestion, 'article' as type 
             FROM articles 
             WHERE LOWER(title) LIKE %s 
             ORDER BY significance_score DESC 
             LIMIT %s)
            UNION
            (SELECT DISTINCT name as suggestion, 'topic' as type 
             FROM ai_topics 
             WHERE LOWER(name) LIKE %s 
             LIMIT %s)
            UNION
            (SELECT DISTINCT name as suggestion, 'source' as type 
             FROM ai_sources 
             WHERE LOWER(name) LIKE %s 
             LIMIT %s)
            ORDER BY suggestion
            LIMIT %s
        """
        
        search_pattern = f"%{query.lower()}%"
        per_category_limit = max(1, limit // 3)
        
        results = db.execute_query(
            suggestions_query, 
            (search_pattern, per_category_limit, search_pattern, per_category_limit, 
             search_pattern, per_category_limit, limit)
        )
        
        suggestions = [
            {"suggestion": row['suggestion'], "type": row['type']}
            for row in results
        ]
        
        return {
            "suggestions": suggestions,
            "query": query,
            "count": len(suggestions)
        }
    except Exception as e:
        logger.error(f"Error getting search suggestions: {str(e)}")
        return {"suggestions": [], "query": query, "count": 0}


@router.get("/stats")
async def get_content_stats(
    time_filter: str = Query("Last Week", description="Time filter for stats"),
    db: DatabaseService = Depends(get_database_service)
):
    """Get content statistics"""
    try:
        time_threshold = content_filtering_service.parse_time_filter(time_filter)
        
        # Overall stats
        stats_query = """
            SELECT 
                COUNT(*) as total_articles,
                COUNT(DISTINCT category_id) as unique_categories,
                COUNT(DISTINCT source) as unique_sources,
                AVG(significance_score) as avg_significance,
                MAX(published_date) as latest_article
            FROM articles
            WHERE published_date >= %s
        """
        
        params = [time_threshold] if time_threshold else [datetime.now() - timedelta(days=7)]
        stats = db.execute_query(stats_query, params, fetch_one=True)
        
        # Content type breakdown
        type_query = """
            SELECT 
                ct.name as content_type,
                COUNT(*) as count,
                AVG(a.significance_score) as avg_significance
            FROM articles a
            JOIN content_types ct ON a.content_type_id = ct.id
            WHERE a.published_date >= %s
            GROUP BY ct.name
            ORDER BY count DESC
        """
        
        type_stats = db.execute_query(type_query, params)
        
        return {
            "overall": {
                "total_articles": stats.get('total_articles', 0),
                "unique_topics": stats.get('unique_topics', 0),
                "unique_sources": stats.get('unique_sources', 0),
                "avg_significance": float(stats.get('avg_significance', 0)) if stats.get('avg_significance') else 0,
                "latest_article": stats.get('latest_article')
            },
            "by_content_type": [
                {
                    "content_type": row['content_type'],
                    "count": row['count'],
                    "avg_significance": float(row['avg_significance']) if row['avg_significance'] else 0
                }
                for row in type_stats
            ],
            "time_filter": time_filter
        }
    except Exception as e:
        logger.error(f"Error getting content stats: {str(e)}")
        return {"overall": {}, "by_content_type": [], "time_filter": time_filter}