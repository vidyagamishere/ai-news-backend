#!/usr/bin/env python3
"""
Enhanced Content Filtering Service
Provides advanced search, filtering, and recommendation capabilities
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from db_service import get_database_service

logger = logging.getLogger(__name__)


@dataclass
class FilterCriteria:
    """Data class for content filtering criteria"""
    interests: List[str] = None  # Category names (for backward compatibility)
    content_types: List[str] = None  # Content type names (for backward compatibility)
    publishers: List[str] = None  # Publisher names (for backward compatibility)
    
    # New ID-based fields (preferred for accurate filtering)
    category_ids: List[int] = None  # Category IDs from ai_categories_master
    content_type_ids: List[int] = None  # Content type IDs from content_types
    publisher_ids: List[int] = None  # Publisher IDs from publishers_master
    
    time_filter: str = "Last Week"
    search_query: str = ""
    limit: int = 50
    significance_threshold: int = 1
    user_id: Optional[str] = None


class ContentFilteringService:
    """Enhanced content filtering and search service"""
    
    def __init__(self):
        # ✅ FIX: Initialize db attribute with database service
        self.db = get_database_service()
        logger.info("✅ ContentFilteringService initialized with database connection")
    
        self.search_weights = {
            'title': 3.0,
            'summary': 2.0,
            'keywords': 1.5,
            'source': 1.0
        }
    
    def get_category_ids_from_names(self, category_names: List[str]) -> List[int]:
        """Convert category names to IDs for accurate filtering"""
        if not category_names:
            return []
        
        try:
            placeholders = ','.join(['%s'] * len(category_names))
            query = f"""
                SELECT id FROM ai_categories_master 
                WHERE name IN ({placeholders})
            """
            results = self.db.execute_query(query, tuple(category_names))
            return [row['id'] for row in results]
        except Exception as e:
            logger.error(f"❌ Failed to get category IDs: {str(e)}")
            return []
    
    def get_content_type_ids_from_names(self, content_type_names: List[str]) -> List[int]:
        """Convert content type names to IDs for accurate filtering"""
        if not content_type_names:
            return []
        
        try:
            # ✅ FIX: Correct variable name from 'content_typeNames' to 'content_type_names'
            placeholders = ','.join(['%s'] * len(content_type_names))
            query = f"""
                SELECT id FROM content_types 
                WHERE LOWER(name) IN ({','.join(['LOWER(%s)'] * len(content_type_names))})
            """
            results = self.db.execute_query(query, tuple(content_type_names))
            return [row['id'] for row in results]
        except Exception as e:
            logger.error(f"❌ Failed to get content type IDs: {str(e)}")
            return []
    
    def get_publisher_ids_from_names(self, publisher_names: List[str]) -> List[int]:
        """Convert publisher names to IDs for accurate filtering"""
        if not publisher_names:
            return []
        
        try:
            placeholders = ','.join(['LOWER(%s)'] * len(publisher_names))
            query = f"""
                SELECT id FROM publishers_master 
                WHERE LOWER(publisher_name) IN ({placeholders})
            """
            results = self.db.execute_query(query, tuple(publisher_names))
            return [row['id'] for row in results]
        except Exception as e:
            logger.error(f"❌ Failed to get publisher IDs: {str(e)}")
            return []
    
    def parse_time_filter(self, time_filter: str) -> Optional[datetime]:
        """Convert time filter string to datetime threshold with timezone awareness"""
        if not time_filter or time_filter == "All Time":
            return None
        
        # Use timezone-aware datetime (UTC)
        now = datetime.now(timezone.utc)
        
        # ✅ VERIFIED: Time filter mapping matches frontend exactly
        time_mappings = {
            # Frontend values (MUST match exactly) - ✅ CORRECT
            "Last 24 Hours": timedelta(hours=24),  # ✅ THIS IS CORRECT
            "Last Week": timedelta(days=7),
            "Last Month": timedelta(days=30),
            "This Year": timedelta(days=365),
            # Legacy values
            "Last 24 hours": timedelta(hours=24),
            "Today": timedelta(hours=24),
            "Yesterday": timedelta(days=1),
            "This Week": timedelta(days=7),
            "This Month": timedelta(days=30),
            "Last Year": timedelta(days=365)
        }
        
        # Exact match (preferred)
        if time_filter in time_mappings:
            delta = time_mappings[time_filter]
            threshold = now - delta
            # ✅ ENHANCED LOGGING
            logger.info(f"⏰ [parse_time_filter] Matched '{time_filter}':")
            logger.info(f"   📅 Current UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   📅 Threshold UTC: {threshold.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   ⏳ Delta: {delta}")
            logger.info(f"   ✅ Returning articles published AFTER {threshold.isoformat()}")
            return threshold
        
        # Case-insensitive fallback
        for key, delta in time_mappings.items():
            if time_filter.lower() == key.lower():
                threshold = now - delta
                logger.info(f"⏰ [parse_time_filter] Case-insensitive match: '{time_filter}' -> '{key}', threshold: {threshold.isoformat()}")
                return threshold
        
        # Default fallback
        logger.warning(f"⚠️ [parse_time_filter] Unknown filter '{time_filter}', using Last Week")
        return now - timedelta(days=7)

    def build_search_query(self, criteria: FilterCriteria) -> Tuple[str, List]:
        """Build optimized SQL query with search and filters"""
        
        # Base query with joins for metadata
        base_query = """
            SELECT DISTINCT a.id, a.title, a.summary, a.url, a.source,
                   a.published_date, a.scraped_date, a.significance_score,
                   a.reading_time, a.image_url, a.keywords, a.content_hash,
                   ct.name as content_type_label,
                   ct.display_name as content_type_display,
                   c.name as category_label,
                   s.name as source_name,
                   s.website as source_website
            FROM articles a
            LEFT JOIN content_types ct ON a.content_type_id = ct.id
            LEFT JOIN ai_categories_master c ON a.category_id = c.id
            LEFT JOIN ai_sources s ON a.source = s.name
            WHERE 1=1
        """
        
        params = []
        
        # Time filter - ✅ FIXED: Include NULL published_dates to match landing-content behavior
        time_threshold = self.parse_time_filter(criteria.time_filter)
        if time_threshold:
            base_query += " AND (a.published_date >= %s OR a.published_date IS NULL)"
            params.append(time_threshold)
        
        # Content type filter - prioritize ID-based filtering
        if criteria.content_type_ids:
            # Use ID-based filtering (preferred)
            placeholders = ",".join(["%s"] * len(criteria.content_type_ids))
            base_query += " AND ct.id IN (" + placeholders + ")"
            params.extend(criteria.content_type_ids)
        elif criteria.content_types:
            # Fallback to name-based filtering (handle both uppercase and lowercase)
            placeholders = ",".join(["%s"] * len(criteria.content_types))
            base_query += " AND (UPPER(ct.name) IN (" + placeholders + ") OR LOWER(ct.name) IN (" + placeholders + "))"
            # Add both uppercase and lowercase versions
            params.extend([ct.upper() for ct in criteria.content_types])
            params.extend([ct.lower() for ct in criteria.content_types])
        
        # Category filter - prioritize ID-based filtering
        if criteria.category_ids:
            # Use ID-based filtering (preferred)
            placeholders = ",".join(["%s"] * len(criteria.category_ids))
            base_query += " AND c.id IN (" + placeholders + ")"
            params.extend(criteria.category_ids)
        elif criteria.interests:
            # Fallback to name-based filtering
            placeholders = ",".join(["%s"] * len(criteria.interests))
            base_query += " AND c.name IN (" + placeholders + ")"
            params.extend(criteria.interests)
        
        # Publisher filter - prioritize ID-based filtering
        if criteria.publisher_ids:
            # Use ID-based filtering (preferred) - direct match with articles.publisher_id
            placeholders = ",".join(["%s"] * len(criteria.publisher_ids))
            base_query += " AND a.publisher_id IN (" + placeholders + ")"
            params.extend(criteria.publisher_ids)
        elif criteria.publishers:
            # Fallback to name-based filtering with smart domain matching
            publisher_conditions = []
            for publisher in criteria.publishers:
                # Match exact publisher name in ai_sources table
                publisher_conditions.append("LOWER(s.name) = LOWER(%s)")
                params.append(publisher)
                
                # Match publisher in article source field (primary check)
                # This handles cases like 'techcrunch' matching 'techcrunch.com' in article.source
                publisher_conditions.append("LOWER(a.source) LIKE LOWER(%s)")
                params.append(f"%{publisher}%")
                
                # Match publisher in source website field  
                publisher_conditions.append("LOWER(s.website) LIKE LOWER(%s)")
                params.append(f"%{publisher}%")
                
                # Additional check for exact domain matches (e.g., 'arxiv' -> 'arxiv.org')
                publisher_conditions.append("LOWER(a.source) LIKE LOWER(%s)")
                params.append(f"%{publisher}.%")  # Matches techcrunch.com, arxiv.org, etc.
            
            if publisher_conditions:
                base_query += " AND (" + " OR ".join(publisher_conditions) + ")"
        
        # Significance threshold
        if criteria.significance_threshold > 1:
            base_query += " AND a.significance_score >= %s"
            params.append(criteria.significance_threshold)
        
        # Advanced search query processing
        if criteria.search_query:
            search_conditions = self._build_search_conditions(criteria.search_query)
            if search_conditions:
                base_query += " AND (" + search_conditions['query'] + ")"
                params.extend(search_conditions['params'])
        
        # Order by significance score and date
        base_query += " ORDER BY a.significance_score DESC, a.published_date DESC"
       
        logger.debug(f"Constructed Query: {base_query} with params {params}")
       
        # Limit results
        if criteria.limit:
            base_query += " LIMIT %s"
            params.append(criteria.limit)

        logger.debug(f"Constructed Query with LIMIT: {base_query} with params {params}")
        return base_query, params
    
    def _build_search_conditions(self, search_query: str) -> Dict[str, Any]:
        """Build advanced search conditions with weights"""
        if not search_query.strip():
            return None
        
        # Clean and tokenize search query
        query_terms = self._tokenize_search_query(search_query)
        if not query_terms:
            return None
        
        conditions = []
        params = []
        
        for term in query_terms:
            term_pattern = f"%{term.lower()}%"
            
            # Search in multiple fields with weights
            field_conditions = [
                "LOWER(a.title) LIKE %s",
                "LOWER(a.summary) LIKE %s",
                "LOWER(a.keywords) LIKE %s",
                "LOWER(a.source) LIKE %s",
                "LOWER(c.name) LIKE %s",
                "LOWER(s.name) LIKE %s"
            ]
            
            # Add parameters for each field
            params.extend([term_pattern] * 6)
            
            # Combine field conditions with OR
            term_condition = "(" + " OR ".join(field_conditions) + ")"
            conditions.append(term_condition)
        
        # Combine all term conditions with AND (all terms must match)
        final_query = " AND ".join(conditions)
        
        return {
            'query': final_query,
            'params': params
        }
    
    def _tokenize_search_query(self, query: str) -> List[str]:
        """Tokenize search query into meaningful terms"""
        # Remove special characters and split
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        terms = [term.strip() for term in cleaned.split() if len(term.strip()) > 2]
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'day'}
        terms = [term for term in terms if term.lower() not in stop_words]
        
        return terms[:10]  # Limit to 10 terms for performance
    
    def get_filtered_content(self, criteria: FilterCriteria) -> List[Dict[str, Any]]:
        """Get filtered content based on criteria with MANDATORY time filtering"""
        try:
            # ✅ CRITICAL FIX: Parse time filter FIRST
            time_threshold = self.parse_time_filter(criteria.time_filter) if criteria.time_filter else None
            
            # ✅ FIX: Convert timezone-aware datetime to naive datetime for database comparison
            if time_threshold and time_threshold.tzinfo:
                # Remove timezone info but keep the UTC time value
                time_threshold_naive = time_threshold.replace(tzinfo=None)
                logger.info(f"🔧 Converted timezone-aware threshold to naive: {time_threshold_naive.isoformat()}")
                time_threshold = time_threshold_naive
            
            # ✅ ENHANCED LOGGING
            logger.info(f"🔍 FILTERING CONTENT:")
            logger.info(f"   📅 Time filter: '{criteria.time_filter}'")
            if time_threshold:
                logger.info(f"   📅 Time threshold (naive UTC): {time_threshold.isoformat()}")
                logger.info(f"   📊 SQL will filter: published_date >= '{time_threshold.isoformat()}'")
            else:
                logger.info(f"   📊 No time filter - returning ALL articles (not recommended)")
            
            # Build base query with composite ranking score
            query = """
                SELECT DISTINCT
                    a.id,
                    a.title,
                    a.summary,
                    a.content_hash,
                    a.url,
                    a.source,
                    a.author,
                    a.published_date,
                    a.scraped_date,
                    a.significance_score,
                    a.complexity_level,
                    a.reading_time,
                    a.keywords,
                    a.content_type_id,
                    a.category_id,
                    a.publisher_id,
                    ct.name as content_type_label,
                    c.name as category_label,
                    -- User interaction flags (personal state)
                    CASE WHEN ai_like.id IS NOT NULL THEN TRUE ELSE FALSE END as has_liked,
                    CASE WHEN ai_bookmark.id IS NOT NULL THEN TRUE ELSE FALSE END as has_bookmarked,
                    CASE WHEN ai_view.id IS NOT NULL THEN TRUE ELSE FALSE END as has_viewed,
                    -- Total interaction counts from article_stats (visible to all users)
                    COALESCE(ast.likes_count, 0) as total_likes,
                    COALESCE(ast.bookmarks_count, 0) as total_bookmarks,
                    COALESCE(ast.views_count, 0) as total_views,
                    COALESCE(ast.shares_count, 0) as total_shares,
                    COALESCE(ast.comments_count, 0) as total_comments,
                    COALESCE(ast.engagement_score, 0) as engagement_score,
                    -- DYNAMIC RANKING SCORE (hybrid approach)
                        (
                            -- Static component (pre-computed publisher + significance)
                            COALESCE(a.static_score_component, 0.375) +
                            
                            -- Dynamic recency component (calculated in real-time with NOW())
                            (
                                CASE 
                                    WHEN a.published_date >= NOW() - INTERVAL '1 day' THEN 1.0
                                    WHEN a.published_date >= NOW() - INTERVAL '3 days' THEN 0.8
                                    WHEN a.published_date >= NOW() - INTERVAL '7 days' THEN 0.6
                                    WHEN a.published_date >= NOW() - INTERVAL '14 days' THEN 0.4
                                    WHEN a.published_date >= NOW() - INTERVAL '30 days' THEN 0.2
                                    ELSE 0.1
                                END
                            ) * 0.25
                        ) as ranking_score
                FROM articles a
                LEFT JOIN content_types ct ON a.content_type_id = ct.id
                LEFT JOIN ai_categories_master c ON a.category_id = c.id
                LEFT JOIN article_stats ast ON a.id = ast.article_id
                LEFT JOIN article_interactions ai_like ON a.id = ai_like.article_id AND ai_like.action_type_id = 1 AND ai_like.user_id = %s
                LEFT JOIN article_interactions ai_bookmark ON a.id = ai_bookmark.article_id AND ai_bookmark.action_type_id = 3 AND ai_bookmark.user_id = %s
                LEFT JOIN article_interactions ai_view ON a.id = ai_view.article_id AND ai_view.action_type_id = 4 AND ai_view.user_id = %s
                WHERE 1=1
            """
            
            params = []
            # Add user_id parameters for the JOINs (3 times for like, bookmark, view)
            user_id_param = criteria.user_id if criteria.user_id else None
            logger.info(f"👤 Filtering with user_id: {user_id_param} (type: {type(user_id_param).__name__})")
            params.extend([user_id_param, user_id_param, user_id_param])
            
            # ✅ CRITICAL: Apply time filter FIRST and ALWAYS if provided
            if time_threshold:
                query += " AND a.published_date >= %s"
                params.append(time_threshold)
                logger.info(f"✅ SQL WHERE clause added: AND a.published_date >= %s (param: {time_threshold.isoformat()})")
            
            # Apply category filter (ID-based preferred)
            if criteria.category_ids:
                placeholders = ','.join(['%s'] * len(criteria.category_ids))
                query += f" AND a.category_id IN ({placeholders})"
                params.extend(criteria.category_ids)
                logger.info(f"✅ SQL WHERE clause added: AND a.category_id IN ({criteria.category_ids})")
        
            # Apply content type filter (ID-based preferred)
            if criteria.content_type_ids:
                placeholders = ','.join(['%s'] * len(criteria.content_type_ids))
                query += f" AND a.content_type_id IN ({placeholders})"
                params.extend(criteria.content_type_ids)
                logger.info(f"✅ SQL WHERE clause added: AND a.content_type_id IN ({criteria.content_type_ids})")
        
            # Apply publisher filter (ID-based preferred)
            if criteria.publisher_ids:
                placeholders = ','.join(['%s'] * len(criteria.publisher_ids))
                query += f" AND a.publisher_id IN ({placeholders})"
                params.extend(criteria.publisher_ids)
                logger.info(f"✅ SQL WHERE clause added: AND a.publisher_id IN ({criteria.publisher_ids})")
        
            # Apply search query filter
            if criteria.search_query and criteria.search_query.strip():
                search_term = f"%{criteria.search_query.lower()}%"
                query += """
                    AND (
                        LOWER(a.title) LIKE %s OR 
                        LOWER(a.summary) LIKE %s OR
                        LOWER(c.name) LIKE %s
                    )
                """
                params.extend([search_term, search_term, search_term])
                logger.info(f"✅ SQL WHERE clause added: search query '{criteria.search_query}'")
        
            # Order by significance score and date
            query += " ORDER BY ranking_score DESC, a.published_date DESC"
        
            # Apply limit
            if criteria.limit:
                query += " LIMIT %s"
                params.append(criteria.limit)
                logger.info(f"✅ SQL LIMIT added: {criteria.limit}")
        
            # ✅ LOG FINAL QUERY
            logger.info(f"🔍 FINAL SQL QUERY:")
            logger.info(f"   Query: {query}...")
            logger.info(f"   Params: {params}")
        
            # Execute query
            db = get_database_service()
            results = db.execute_query(query, tuple(params) if params else None)
            
            logger.info(f"✅ Query returned {len(results)} articles")
            
            # Log sample of returned dates to verify filtering
            if results and len(results) > 0:
                sample_dates = [r.get('published_date').isoformat() if r.get('published_date') else 'NULL' for r in results[:5]]
                logger.info(f"📅 Sample published dates (first 5): {sample_dates}")
                
                # ✅ FIX: Compare naive datetimes (remove timezone awareness before comparison)
                if time_threshold:
                    outside_filter = []
                    for r in results:
                        pub_date = r.get('published_date')
                        if pub_date:
                            # Ensure both datetimes are naive for comparison
                            if hasattr(pub_date, 'tzinfo') and pub_date.tzinfo:
                                pub_date_naive = pub_date.replace(tzinfo=None)
                            else:
                                pub_date_naive = pub_date
                            
                            if pub_date_naive < time_threshold:
                                outside_filter.append(r)
                    
                    if outside_filter:
                        logger.warning(f"⚠️ FILTER NOT WORKING: {len(outside_filter)} articles are BEFORE threshold!")
                        logger.warning(f"   Threshold: {time_threshold.isoformat()}")
                        logger.warning(f"   Sample old dates: {[r.get('published_date').isoformat() if hasattr(r.get('published_date'), 'isoformat') else str(r.get('published_date')) for r in outside_filter[:3]]}")
                    else:
                        logger.info(f"✅ Time filter verified: ALL {len(results)} articles are after {time_threshold.isoformat()}")
            
            # ✅ Log interaction states for debugging
            if results and criteria.user_id:
                liked_count = sum(1 for r in results if r.get('has_liked'))
                bookmarked_count = sum(1 for r in results if r.get('has_bookmarked'))
                viewed_count = sum(1 for r in results if r.get('has_viewed'))
                logger.info(f"💙 Interaction states for user {criteria.user_id}: {liked_count} liked, {bookmarked_count} bookmarked, {viewed_count} viewed out of {len(results)} articles")
        
            return [dict(row) for row in results]
        
        except Exception as e:
            logger.error(f"❌ Error filtering content: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def group_content_by_interests(self, articles: List[Dict], user_interests: List[str], include_uncategorized: bool = False) -> Dict[str, List[Dict]]:
        """Group articles by user interests"""
        grouped = {}
        uncategorized = []
        
        # Group by interests
        for interest in user_interests:
            interest_articles = [
                article for article in articles 
                if article.get('category_label') == interest
            ]
            if interest_articles:
                grouped[interest] = interest_articles
        
        # Handle uncategorized content for search results
        if include_uncategorized:
            uncategorized = [
                article for article in articles 
                if not article.get('category_label') or article.get('category_label') not in user_interests
            ]
            if uncategorized:
                grouped['Other Relevant Results'] = uncategorized
        
        return grouped
    
    def get_content_recommendations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get personalized content recommendations based on user behavior"""
        try:
            # This would typically use ML algorithms, but for now we'll use simple heuristics
            
            db = get_database_service()
            
            # Get user's reading history and preferences
            user_query = """
                SELECT * FROM user_preferences WHERE user_id = %s
            """
            user_result = db.execute_query(user_query, (user_id,), fetch_one=True)
            
            if not user_result:
                return []
            
            # Build criteria from user preferences
            criteria = FilterCriteria(
                interests=user_result.get('categories_selected', []),
                content_types=user_result.get('content_types_selected', ['BLOGS', 'VIDEOS', 'PODCASTS']),
                publishers=user_result.get('publishers_selected', []),
                time_filter='Last Week',
                significance_threshold=6,  # Higher significance for recommendations
                limit=limit
            )
            
            return self.get_filtered_content(criteria)
            
        except Exception as e:
            logger.error(f"❌ Failed to get recommendations: {str(e)}")
            return []
    
    def get_trending_topics(self, time_filter: str = "Last Week") -> List[Dict]:
        """Get trending topics based on content volume and significance"""
        try:
            db = get_database_service()
            time_threshold = self.parse_time_filter(time_filter)
            
            query = """
                SELECT 
                    c.name as topic,
                    COUNT(*) as article_count,
                    AVG(a.significance_score) as avg_significance,
                    MAX(a.published_date) as latest_article
                FROM articles a
                JOIN ai_categories_master c ON a.category_id = c.id
                WHERE a.published_date >= %s
                GROUP BY c.name
                HAVING COUNT(*) >= 3
                ORDER BY article_count DESC, avg_significance DESC
                LIMIT 20
            """
            
            params = [time_threshold] if time_threshold else [datetime.now() - timedelta(days=7)]
            results = db.execute_query(query, params)
            
            trending = []
            for row in results:
                trending.append({
                    'topic': row['topic'],
                    'article_count': row['article_count'],
                    'avg_significance': float(row['avg_significance']) if row['avg_significance'] else 0,
                    'latest_article': row['latest_article']
                })
            
            return trending
            
        except Exception as e:
            logger.error(f"❌ Failed to get trending topics: {str(e)}")
            return []


# Global instance
content_filtering_service = ContentFilteringService()