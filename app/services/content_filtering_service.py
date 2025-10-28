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
                   s.website as source_website,
                   p.publisher_name,
                   p.id as publisher_id
            FROM articles a
            LEFT JOIN content_types ct ON a.content_type_id = ct.id
            LEFT JOIN ai_categories_master c ON a.category_id = c.id
            LEFT JOIN ai_sources s ON a.source = s.name
            LEFT JOIN publishers_master p ON a.publisher_id = p.id
            WHERE 1=1
        """
        
        params = []
        
        # Time filter
        time_threshold = self.parse_time_filter(criteria.time_filter)
        if time_threshold:
            base_query += " AND a.published_date >= %s"
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
        
        # Order by relevance and significance
        if criteria.search_query:
            base_query += " ORDER BY a.significance_score DESC, a.published_date DESC"
        else:
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
        """Get filtered content based on criteria with enhanced logging"""
        
        try:
            # Parse time filter FIRST to get the datetime threshold
            time_threshold = self.parse_time_filter(criteria.time_filter)
            
            # Log the threshold being used
            if time_threshold:
                logger.info(f"⏰ Using time threshold: {time_threshold.isoformat()} for filter '{criteria.time_filter}'")
            else:
                logger.info(f"⏰ No time filter applied (showing all time)")
            
            # Build base query
            query = """
                SELECT DISTINCT
                    a.id,
                    a.title,
                    a.summary,
                    a.url,
                    a.source,
                    a.published_date,
                    a.scraped_date,
                    a.significance_score,
                    a.reading_time,
                    a.author,
                    a.content_hash,
                    ct.name as content_type_label,
                    ct.display_name as content_type_display_name,
                    c.name as category_label,
                    c.id as category_id,
                    a.complexity_level,
                    a.keywords
                FROM articles a
                LEFT JOIN content_types ct ON a.content_type_id = ct.id
                LEFT JOIN ai_categories_master c ON a.category_id = c.id
                WHERE 1=1
            """
            
            params = []
            
            # CRITICAL: Apply time filter FIRST (most restrictive)
            if time_threshold:
                query += " AND a.published_date >= %s"
                params.append(time_threshold)
                logger.info(f"🔍 Time filter applied: published_date >= {time_threshold.isoformat()}")
            
            # Apply category filter (ID-based preferred, fallback to name-based)
            if criteria.category_ids:
                placeholders = ','.join(['%s'] * len(criteria.category_ids))
                query += f" AND c.id IN ({placeholders})"
                params.extend(criteria.category_ids)
                logger.info(f"🔍 Category ID filter applied: {criteria.category_ids}")
            elif criteria.interests:
                placeholders = ','.join(['%s'] * len(criteria.interests))
                query += f" AND c.name IN ({placeholders})"
                params.extend(criteria.interests)
                logger.info(f"🔍 Category name filter applied: {criteria.interests}")
            
            # Apply content type filter (ID-based preferred, fallback to name-based)
            if criteria.content_type_ids:
                placeholders = ','.join(['%s'] * len(criteria.content_type_ids))
                query += f" AND ct.id IN ({placeholders})"
                params.extend(criteria.content_type_ids)
                logger.info(f"🔍 Content type ID filter applied: {criteria.content_type_ids}")
            elif criteria.content_types:
                placeholders = ','.join(['%s'] * len(criteria.content_types))
                query += f" AND ct.name IN ({placeholders})"
                params.extend(criteria.content_types)
                logger.info(f"🔍 Content type name filter applied: {criteria.content_types}")
            
            # Apply publisher filter (ID-based preferred, fallback to name-based)
            if criteria.publisher_ids:
                placeholders = ','.join(['%s'] * len(criteria.publisher_ids))
                query += f" AND a.publisher_id IN ({placeholders})"
                params.extend(criteria.publisher_ids)
                logger.info(f"🔍 Publisher ID filter applied: {criteria.publisher_ids}")
            elif criteria.publishers and criteria.publishers != ['all']:
                # Get publisher IDs from names
                publisher_names = criteria.publishers
                publisher_ids_query = f"SELECT id FROM publishers_master WHERE publisher_name IN ({','.join(['%s'] * len(publisher_names))})"
                publisher_results = self.db.execute_query(publisher_ids_query, tuple(publisher_names))
                publisher_ids = [pub['id'] for pub in publisher_results]
                
                if publisher_ids:
                    placeholders = ','.join(['%s'] * len(publisher_ids))
                    query += f" AND a.publisher_id IN ({placeholders})"
                    params.extend(publisher_ids)
                    logger.info(f"🔍 Publisher name filter converted to IDs and applied: {publisher_ids}")
            
            # Apply search query filter
            if criteria.search_query:
                search_term = f"%{criteria.search_query.lower()}%"
                query += """
                    AND (
                        LOWER(a.title) LIKE %s 
                        OR LOWER(a.summary) LIKE %s
                        OR LOWER(c.name) LIKE %s
                    )
                """
                params.extend([search_term, search_term, search_term])
                logger.info(f"🔍 Search query filter applied: '{criteria.search_query}'")
            
            # Order by significance and recency
            query += " ORDER BY a.significance_score DESC, a.published_date DESC"
            
            # Apply limit
            if criteria.limit:
                query += " LIMIT %s"
                params.append(criteria.limit)
            
            # Log final query for debugging
            logger.info(f"🔍 Final query parameters: {params}")
            logger.debug(f"🔍 Final SQL query: {query}")
            
            # Execute query
            results = self.db.execute_query(query, tuple(params))
            
            # ✅ ENHANCED: Log actual date range of returned results
            if results:
                dates = [a.get('published_date') for a in results if a.get('published_date')]
                if dates:
                    oldest = min(dates)
                    newest = max(dates)
                    logger.info(f"📊 [get_filtered_content] Date range of results:")
                    logger.info(f"   Oldest article: {oldest}")
                    logger.info(f"   Newest article: {newest}")
                    logger.info(f"   Total articles: {len(results)}")
            
            logger.info(f"✅ Content filtering returned {len(results)} results")
            return [dict(row) for row in results]
        
        except Exception as e:
            logger.error(f"❌ [get_filtered_content] Error: {str(e)}")
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