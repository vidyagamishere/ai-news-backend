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
    interests: List[str] = None
    content_types: List[str] = None
    publishers: List[str] = None
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
            'description': 2.0,
            'keywords': 1.5,
            'source': 1.0
        }
    
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
            SELECT DISTINCT a.id, a.title, a.description, a.url, a.source,
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
        
        # Time filter
        time_threshold = self.parse_time_filter(criteria.time_filter)
        if time_threshold:
            base_query += " AND a.published_date >= %s"
            params.append(time_threshold)
        
        # Content type filter
        if criteria.content_types:
            placeholders = ",".join(["%s"] * len(criteria.content_types))
            base_query += f" AND UPPER(ct.name) IN ({placeholders})"
            params.extend([ct.upper() for ct in criteria.content_types])
        
        # Publisher/Source filter
        if criteria.publishers:
            placeholders = ",".join(["%s"] * len(criteria.publishers))
            base_query += f" AND (s.name IN ({placeholders}) OR a.source IN ({placeholders}))"
            params.extend(criteria.publishers * 2)
        
        # Interest/Topic filter
        if criteria.interests:
            placeholders = ",".join(["%s"] * len(criteria.interests))
            base_query += f" AND at.name IN ({placeholders})"
            params.extend(criteria.interests)
        
        # Significance threshold
        if criteria.significance_threshold > 1:
            base_query += " AND a.significance_score >= %s"
            params.append(criteria.significance_threshold)
        
        # Advanced search query processing
        if criteria.search_query:
            search_conditions = self._build_search_conditions(criteria.search_query)
            if search_conditions:
                base_query += f" AND ({search_conditions['query']})"
                params.extend(search_conditions['params'])
        
        # Order by relevance and significance
        if criteria.search_query:
            base_query += " ORDER BY a.significance_score DESC, a.published_date DESC"
        else:
            base_query += " ORDER BY a.significance_score DESC, a.published_date DESC"
        
        # Limit results
        if criteria.limit:
            base_query += " LIMIT %s"
            params.append(criteria.limit)
        
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
                "LOWER(a.description) LIKE %s",
                "LOWER(a.keywords) LIKE %s",
                "LOWER(a.source) LIKE %s",
                "LOWER(at.name) LIKE %s",
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
        """Get filtered content based on criteria"""
        try:
            db = get_database_service()
            query, params = self.build_search_query(criteria)
            
            logger.info(f"🔍 Content filtering - Query length: {len(query)}, Params: {len(params)}")
            if criteria.search_query:
                logger.info(f"🔍 Search query: '{criteria.search_query}'")
            
            results = db.execute_query(query, params)
            
            # Convert to standardized format
            articles = []
            for row in results:
                article = {
                    'id': row['id'],
                    'title': row.get('title'),
                    'description': row.get('description'),
                    'url': row.get('url'),
                    'source': row.get('source') or row.get('source_name'),
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
                    'source_website': row.get('source_website')
                }
                articles.append(article)
            
            logger.info(f"✅ Found {len(articles)} articles matching criteria")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Content filtering failed: {str(e)}")
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
            
            # Get user's reading history and preferences
            user_query = """
                SELECT preferences FROM users WHERE id = %s
            """
            user_result = db.execute_query(user_query, (user_id,), fetch_one=True)
            
            if not user_result or not user_result.get('preferences'):
                return []
            
            preferences = user_result['preferences']
            
            # Build criteria from user preferences
            criteria = FilterCriteria(
                interests=preferences.get('interests', []),
                content_types=preferences.get('selected_content_types', ['ARTICLE', 'VIDEO', 'AUDIO']),
                publishers=preferences.get('selected_publishers', []),
                time_filter=preferences.get('time_filter', 'Last Week'),
                significance_threshold=6,  # Higher significance for recommendations
                limit=limit
            )
            
            return self.get_filtered_content(db, criteria)
            
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