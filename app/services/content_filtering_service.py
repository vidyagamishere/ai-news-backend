#!/usr/bin/env python3
"""
Enhanced Content Filtering Service
Provides advanced search, filtering, and recommendation capabilities
"""

import logging
import re
from datetime import datetime, timedelta
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
            db = get_database_service()
            placeholders = ",".join(["%s"] * len(category_names))
            query = f"""
                SELECT id FROM ai_categories_master 
                WHERE name IN ({placeholders})
            """
            results = db.execute_query(query, category_names)
            return [row['id'] for row in results]
        except Exception as e:
            logger.error(f"❌ Failed to get category IDs: {str(e)}")
            return []
    
    def get_content_type_ids_from_names(self, content_type_names: List[str]) -> List[int]:
        """Convert content type names to IDs for accurate filtering"""
        if not content_type_names:
            return []
        
        try:
            db = get_database_service()
            placeholders = ",".join(["%s"] * len(content_type_names))
            query = f"""
                SELECT id FROM content_types 
                WHERE LOWER(name) IN ({','.join(['LOWER(%s)'] * len(content_type_names))})
                   OR UPPER(name) IN ({','.join(['UPPER(%s)'] * len(content_type_names))})
            """
            # Prepare parameters for both lower and upper case matching
            params = content_type_names + content_type_names
            results = db.execute_query(query, params)
            return [row['id'] for row in results]
        except Exception as e:
            logger.error(f"❌ Failed to get content type IDs: {str(e)}")
            return []
    
    def get_publisher_ids_from_names(self, publisher_names: List[str]) -> List[int]:
        """Convert publisher names to IDs for accurate filtering"""
        if not publisher_names:
            return []
        
        try:
            db = get_database_service()
            placeholders = ",".join(["%s"] * len(publisher_names))
            query = f"""
                SELECT id FROM publishers_master 
                WHERE LOWER(publisher_name) IN ({','.join(['LOWER(%s)'] * len(publisher_names))})
                   OR UPPER(publisher_name) IN ({','.join(['UPPER(%s)'] * len(publisher_names))})
                   AND is_active = TRUE
            """
            # Prepare parameters for both lower and upper case matching
            params = publisher_names + publisher_names
            results = db.execute_query(query, params)
            return [row['id'] for row in results]
        except Exception as e:
            logger.error(f"❌ Failed to get publisher IDs: {str(e)}")
            return []
    
    def parse_time_filter(self, time_filter: str) -> Optional[datetime]:
        """Convert time filter string to datetime"""
        if time_filter == "All Time":
            return None
        
        now = datetime.now()
        filters = {
            "Last 24 hours": timedelta(days=1),
            "Last Week": timedelta(days=7),
            "Last Month": timedelta(days=30),
            "Last Year": timedelta(days=365)
        }
        
        delta = filters.get(time_filter, timedelta(days=7))
        return now - delta
    
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
        """Get filtered content based on criteria with enhanced error handling"""
        try:
            db = get_database_service()
            
            # Validate database connection
            if not db or not hasattr(db, 'execute_query'):
                raise RuntimeError("Database service is not available or improperly initialized")
            
            # Build query with validation
            try:
                query, params = self.build_search_query(criteria)
            except Exception as query_error:
                logger.error(f"❌ Failed to build search query: {str(query_error)}")
                raise RuntimeError(f"Query construction failed: {str(query_error)}")
            
            # Validate query parameters
            if not query or not isinstance(params, list):
                raise ValueError("Invalid query or parameters generated")
            
            logger.info(f"🔍 Content filtering - Query length: {len(query)}, Params: {len(params)}")
            if criteria.search_query:
                logger.info(f"🔍 Search query: '{criteria.search_query}'")
            
            # Execute query with timeout protection
            try:
                results = db.execute_query(query, params)
            except Exception as db_error:
                logger.error(f"❌ Database query execution failed: {str(db_error)}")
                # Try a simplified query as fallback
                logger.info(f"🔄 Attempting simplified query fallback...")
                simple_query = "SELECT DISTINCT a.* FROM articles a WHERE a.published_date >= NOW() - INTERVAL '7 days' ORDER BY a.published_date DESC LIMIT %s"
                try:
                    results = db.execute_query(simple_query, [min(criteria.limit or 50, 100)])
                    logger.info(f"✅ Simplified query fallback successful")
                except Exception as simple_error:
                    logger.error(f"❌ Even simplified query failed: {str(simple_error)}")
                    raise RuntimeError(f"All database queries failed: {str(simple_error)}")
            
            # Convert to standardized format
            articles = []
            for row in results:
                article = {
                    'id': row['id'],
                    'title': row.get('title'),
                    'summary': row.get('summary'),
                    'url': row.get('url'),
                    'source': row.get('source_name') or row.get('source'),
                    'published_date': row.get('published_date'),
                    'scraped_date': row.get('scraped_date'),
                    'significance_score': row.get('significance_score', 5),
                    'reading_time': row.get('reading_time', 3),
                    'image_url': row.get('image_url'),
                    'keywords': row.get('keywords'),
                    'content_hash': row.get('content_hash'),
                    'content_type_label': row.get('content_type_label'),
                    'content_type_display': row.get('content_type_display'),
                    'category_label': row.get('category_label'),
                    'source_website': row.get('source_website'),
                    'publisher_name': row.get('publisher_name'),
                    'publisher_id': row.get('publisher_id')
                }
                articles.append(article)
            
            logger.info(f"✅ Found {len(articles)} articles matching criteria")
            
            # Final validation
            if not articles:
                logger.warning(f"⚠️ No articles found with current criteria - this may indicate data issues or overly restrictive filters")
            
            return articles
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Content filtering failed: {error_msg}")
            
            # Provide helpful error context
            if "database" in error_msg.lower():
                logger.error(f"📎 Database-related error - check connection and query syntax")
            elif "query" in error_msg.lower():
                logger.error(f"📎 Query-related error - check filter criteria and SQL syntax")
            elif "timeout" in error_msg.lower():
                logger.error(f"📎 Timeout error - query may be too complex or database overloaded")
            else:
                logger.error(f"📎 Unknown error type - check system resources and logs")
            
            # Return empty list but log the failure for monitoring
            logger.info(f"📎 Returning empty result set due to filtering failure")
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