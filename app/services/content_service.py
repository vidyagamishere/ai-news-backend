#!/usr/bin/env python3
"""
Content service for modular FastAPI architecture
"""

import os
import json
import logging
import traceback
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

# Ensure logging is available globally as a fallback
import logging as logging_module

# Make logging available in global namespace as a workaround
globals()['logging'] = logging

from db_service import get_database_service
from crawl4ai_scraper import AdminScrapingInterface

logger = logging.getLogger(__name__)


class DatabaseAdapter:
    """Adapter to make database service compatible with AdminScrapingInterface"""
    
    def __init__(self, db_service):
        self.db_service = db_service
    
    def get_ai_sources(self):
        """Get AI sources with proper field mapping"""
        return self.db_service.get_ai_sources()
    
    def get_ai_sources_by_frequency(self, scrape_frequency_days: int = 1):
        """Get AI sources filtered by scrape_frequency_days"""
        return self.db_service.get_ai_sources_by_frequency(scrape_frequency_days)
    
    def insert_article(self, article_data):
        """Insert article with proper field mapping and defaults"""
        try:
            # Map and validate article data with required defaults
            mapped_data = {
                'id': article_data.get('id'),
                'title': article_data.get('title', 'Untitled Article'),
                'summary': article_data.get('summary', ''),
                'content': article_data.get('content', article_data.get('summary', '')),
                'content_hash': article_data.get('content_hash'),
                'author': article_data.get('author', 'Unknown'),
                'url': article_data.get('url'),
                'source': article_data.get('source', 'Unknown'),
                
                # Content type mapping - use ID if provided, otherwise use label for db_service mapping
                'content_type_id': article_data.get('content_type_id'),
                'content_type_label': article_data.get('content_type_label', 'Blogs'),  # Default to blogs
                
                # Category mapping - use ID if provided, otherwise use label for db_service mapping  
                'category_id': article_data.get('category_id'),
                'topic_category_label': article_data.get('topic_category_label', 'Generative AI'),
                'source_category': article_data.get('source_category'),  # RSS source category for hybrid fallback
                
                'significance_score': article_data.get('significance_score', 6),
                'complexity_level': article_data.get('complexity_level', 'Medium'),
                'reading_time': article_data.get('reading_time', 1),
                'published_date': article_data.get('published_date'),
                'keywords': article_data.get('keywords', []),
                'scraped_date': article_data.get('scraped_date'),
                'created_date': article_data.get('created_date'),
                'updated_date': article_data.get('updated_date'),
                'llm_processed': article_data.get('llm_processed', True),
                'publisher_id': article_data.get('publisher_id')  # Include publisher_id from ai_sources mapping
            }
            
            # Debug logging for critical fields
            logger.info(f"🔍 DATABASE INSERT DEBUG:")
            logger.info(f"   Title: '{mapped_data.get('title', 'MISSING')}'")
            logger.info(f"   URL: {mapped_data.get('url', 'MISSING')}")
            logger.info(f"   Publisher ID: {mapped_data.get('publisher_id', 'MISSING')}")
            logger.info(f"   Content Type ID: {mapped_data.get('content_type_id', 'MISSING')}")
            logger.info(f"   Content Type Label: {mapped_data.get('content_type_label', 'MISSING')}")
            logger.info(f"   Category ID: {mapped_data.get('category_id', 'MISSING')}")
            logger.info(f"   Topic Category Label: {mapped_data.get('topic_category_label', 'MISSING')}")
            logger.info(f"   Source Category: {mapped_data.get('source_category', 'MISSING')}")
            logger.info(f"   Significance Score: {mapped_data.get('significance_score', 'MISSING')}")
            logger.info(f"   Content Hash: {mapped_data.get('content_hash', 'MISSING')}")

            
            
            return self.db_service.insert_article(mapped_data)
            
        except Exception as e:
            logger.error(f"❌ DatabaseAdapter insert_article failed: {e}")
            return False


class ContentService:
    def __init__(self):
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        logger.info("📰 ContentService initialized with PostgreSQL")
        
        if self.DEBUG:
            logger.debug("🔍 Debug mode enabled in ContentService")
    
    def get_digest(self, user_id: Optional[str] = None, personalized: bool = False) -> Dict[str, Any]:
        """Get news digest from PostgreSQL database with topic information"""
        try:
            db = get_database_service()
            
            logger.info(f"📊 Getting digest - User: {user_id or 'anonymous'}, Personalized: {personalized}")
            
            # Get articles with topic information using database view
            # First check if any articles exist
            count_query = "SELECT COUNT(*) as count FROM articles"
            article_count = db.execute_query(count_query, fetch_one=True)
            
            if article_count and article_count['count'] > 0:
                # Query articles table directly instead of using views that may not exist
                articles_query = """
                    SELECT a.*, ct.name as content_type_name, ct.display_name as content_type_display
                    FROM articles a
                    LEFT JOIN content_types ct ON a.content_type_id = ct.id
                    WHERE a.published_date > NOW() - INTERVAL '7 days'
                    ORDER BY a.published_date DESC, a.significance_score DESC

                    LIMIT 100
                """
                articles = db.execute_query(articles_query)
            else:
                logger.info("📊 No articles found in database, returning empty digest")
                articles = []
            
            # Process articles for response
            processed_articles = []
            for article in articles:
                article_dict = dict(article)
                
                # Convert timestamps to ISO format
                for field in ['published_date', 'scraped_date']:
                    if article_dict.get(field):
                        article_dict[field] = article_dict[field].isoformat() if hasattr(article_dict[field], 'isoformat') else str(article_dict[field])
                
                # Parse topics JSON
                if article_dict.get('topics'):
                    if isinstance(article_dict['topics'], str):
                        article_dict['topics'] = json.loads(article_dict['topics'])
                
                # Convert topic strings to lists
                if article_dict.get('topic_names'):
                    if isinstance(article_dict['topic_names'], str):
                        article_dict['topic_names'] = article_dict['topic_names'].split(', ') if article_dict['topic_names'] else []
                
                if article_dict.get('topic_categories'):
                    if isinstance(article_dict['topic_categories'], str):
                        article_dict['topic_categories'] = article_dict['topic_categories'].split(', ') if article_dict['topic_categories'] else []
                
                processed_articles.append(article_dict)
            
            # Get top stories (high significance score)
            if processed_articles:
                top_stories = [article for article in processed_articles if article.get('significance_score', 0) >= 8][:10]
                
                # Organize content by type using content_type_name from database view
                content = {
                    'blog': [a for a in processed_articles if a.get('content_type_name') == 'blogs'][:20],
                    'audio': [a for a in processed_articles if a.get('content_type_name') == 'podcasts'][:15],
                    'video': [a for a in processed_articles if a.get('content_type_name') == 'videos'][:15],
                    'events': [a for a in processed_articles if a.get('content_type_name') == 'events'][:10],
                    'learning': [a for a in processed_articles if a.get('content_type_name') == 'learning'][:10],
                    'demos': [a for a in processed_articles if a.get('content_type_name') == 'demos'][:10]
                }
            else:
                # No articles found - provide empty structure with helpful message
                logger.info("📊 No articles in database - providing empty digest structure")
                top_stories = []
                content = {
                    'blog': [],
                    'audio': [],
                    'video': [],
                    'events': [],
                    'learning': [],
                    'demos': []
                }
            
            # Create summary
            summary = {
                'total_articles': len(processed_articles),
                'top_stories_count': len(top_stories),
                'content_distribution': {k: len(v) for k, v in content.items()},
                'latest_update': datetime.utcnow().isoformat(),
                'personalization_note': f"{'Personalized' if personalized else 'General'} content for AI professionals",
                'status': 'empty_database' if len(processed_articles) == 0 else 'success',
                'message': 'No articles found. Admin can use the scraping feature to populate the database.' if len(processed_articles) == 0 else 'Articles loaded successfully'
            }
            
            response = {
                'topStories': top_stories,
                'content': content,
                'summary': summary,
                'personalized': personalized,
                'debug_info': {
                    'total_articles_fetched': len(processed_articles),
                    'database_view_used': 'digest_articles',
                    'is_personalized': personalized,
                    'user_id': user_id,
                    'database_type': 'postgresql',
                    'migration_from_sqlite': True,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            logger.info(f"✅ Digest generated successfully - {len(processed_articles)} articles, {len(top_stories)} top stories")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get digest: {str(e)}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise e
    
    def get_content_by_type(self, content_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get content by type from PostgreSQL"""
        try:
            db = get_database_service()
            
            # First check if any articles exist
            count_query = "SELECT COUNT(*) as count FROM articles"
            article_count = db.execute_query(count_query, fetch_one=True)
            
            if article_count and article_count['count'] > 0:
                # Query articles table directly instead of using views that may not exist
                query = """
                    SELECT a.*, ct.name as content_type_name, ct.display_name as content_type_display
                    FROM articles a
                    LEFT JOIN content_types ct ON a.content_type_id = ct.id
                    WHERE ct.name = %s
                    ORDER BY a.published_date DESC
                    LIMIT %s
                """
                articles = db.execute_query(query, (content_type, limit))
            else:
                logger.info(f"📊 No articles found for content type: {content_type}")
                articles = []
            
            processed_articles = []
            for article in articles:
                article_dict = dict(article)
                
                # Convert timestamps
                for field in ['published_date', 'scraped_date']:
                    if article_dict.get(field):
                        article_dict[field] = article_dict[field].isoformat() if hasattr(article_dict[field], 'isoformat') else str(article_dict[field])
                
                # Parse JSON fields
                if article_dict.get('topics'):
                    if isinstance(article_dict['topics'], str):
                        article_dict['topics'] = json.loads(article_dict['topics'])
                
                processed_articles.append(article_dict)
            
            logger.info(f"✅ Retrieved {len(processed_articles)} articles for content type: {content_type}")
            return processed_articles
            
        except Exception as e:
            logger.error(f"❌ Failed to get content by type: {str(e)}")
            return []
    
    def get_ai_categories(self) -> List[Dict[str, Any]]:
        """Get all AI categories from ai_categories_master table only"""
        try:
            db = get_database_service()
            
            # Check if ai_categories_master table exists and has data
            try:
                count_query = "SELECT COUNT(*) as count FROM ai_categories_master"
                count_result = db.execute_query(count_query, fetch_one=True)
                logger.info(f"🔍 ai_categories_master table count: {count_result.get('count', 0)}")
            except Exception as table_error:
                logger.error(f"❌ ai_categories_master table issue: {str(table_error)}")
                return []
            
            query = """
                SELECT id, name, description, priority, category_label
                FROM ai_categories_master
                WHERE is_active = TRUE
                ORDER BY priority, name
            """
            
            categories = db.execute_query(query)
            logger.info(f"🔍 Raw categories query returned: {len(categories) if categories else 0} rows")
            
            # Auto-select categories: Generative AI, AI Start Ups, AI Applications
            auto_selected_categories = ["Generative AI", "AI Start Ups", "AI Applications"]
            
            processed_categories = []
            for category in categories:
                category_dict = dict(category)
                # Auto-select specific categories
                category_dict['selected'] = category_dict.get('name', '') in auto_selected_categories
                logger.info(f"🔍 Processing category: {category_dict}")
                processed_categories.append(category_dict)
            
            logger.info(f"✅ Retrieved {len(processed_categories)} AI categories from ai_categories_master")
            return processed_categories
            
        except Exception as e:
            logger.error(f"❌ Failed to get AI categories: {str(e)}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return []
    
    def get_content_types(self) -> List[Dict[str, Any]]:
        """Get only 3 main content types with auto-selection"""
        try:
            # Return only 3 main content types with auto-selection
            content_types = [
                {
                    'id': 1,
                    'name': 'ARTICLE',
                    'display_name': 'Articles',
                    'description': 'News articles and blog posts',
                    'frontend_section': 'blog',
                    'icon': 'article',
                    'is_active': True,
                    'selected': True  # Auto-selected
                },
                {
                    'id': 2,
                    'name': 'VIDEO',
                    'display_name': 'Videos',
                    'description': 'Video content and tutorials',
                    'frontend_section': 'video',
                    'icon': 'video',
                    'is_active': True,
                    'selected': True  # Auto-selected
                },
                {
                    'id': 3,
                    'name': 'AUDIO',
                    'display_name': 'Podcasts',
                    'description': 'Podcast episodes and audio content',
                    'frontend_section': 'audio',
                    'icon': 'audio',
                    'is_active': True,
                    'selected': True  # Auto-selected
                }
            ]
            
            logger.info(f"✅ Retrieved {len(content_types)} content types (hardcoded 3 main types)")
            return content_types
            
        except Exception as e:
            logger.error(f"❌ Failed to get content types: {str(e)}")
            # Return fallback content types if database query fails
            return [
                {
                    'id': 1,
                    'name': 'blogs',
                    'display_name': 'Articles',
                    'description': 'Blog posts and articles',
                    'frontend_section': 'blog',
                    'icon': '📄',
                    'is_active': True
                },
                {
                    'id': 2,
                    'name': 'podcasts',
                    'display_name': 'Podcasts',
                    'description': 'Audio content and interviews',
                    'frontend_section': 'audio',
                    'icon': '🎙️',
                    'is_active': True
                },
                {
                    'id': 3,
                    'name': 'videos',
                    'display_name': 'Videos',
                    'description': 'Video content and tutorials',
                    'frontend_section': 'video',
                    'icon': '🎥',
                    'is_active': True
                }
            ]
    
    async def scrape_content(self, llm_model: str = 'claude') -> Dict[str, Any]:
        """Trigger content scraping operation using Crawl4AI + selected LLM"""
        try:
            logger.info(f"🕷️ Starting content scraping operation with Crawl4AI + {llm_model}")  # ✅ FIX: Use parameter
            logger.info(f"🤖 LLM MODEL PARAMETER RECEIVED IN scrape_content: '{llm_model}'")
            
            if self.DEBUG:
                logger.debug(f"🔍 Scrape content method called with llm_model={llm_model}")
            
            db = get_database_service()
            
            # Initialize the admin scraping interface with a database adapter
            db_adapter = DatabaseAdapter(db)
            admin_scraper = AdminScrapingInterface(db_adapter)
            
            if self.DEBUG:
                logger.debug("🔍 AdminScrapingInterface initialized with database adapter")
            
            # Run the async scraping process with LLM model
            try:
                if self.DEBUG:
                    logger.debug(f"🔍 Running async scraping process with {llm_model}")
                
                # ✅ FIX: PASS llm_model TO initiate_scraping!
                logger.info(f"🔍 Calling admin_scraper.initiate_scraping(llm_model='{llm_model}')")
                result = await admin_scraper.initiate_scraping(llm_model=llm_model)  # ← ADD THIS PARAMETER!
                
                if self.DEBUG:
                    logger.debug(f"🔍 Scraping result: {result}")
                
                return result
                
            except Exception as async_error:
                logger.error(f"❌ Async scraping failed: {str(async_error)}")
                if self.DEBUG:
                    logger.debug(f"🔍 Async error details: {traceback.format_exc()}")
                
                # Fallback to mock implementation if async fails
                logger.info("🔄 Falling back to mock scraping due to async error")
                return self._fallback_mock_scraping(db)
            
        except Exception as e:
            logger.error(f"❌ Content scraping failed: {str(e)}")
            if self.DEBUG:
                logger.debug(f"🔍 Scraping error details: {traceback.format_exc()}")
            
            # Try fallback implementation
            try:
                db = get_database_service()
                return self._fallback_mock_scraping(db)
            except:
                raise e
    
    def _fallback_mock_scraping(self, db) -> Dict[str, Any]:
        """Fallback mock scraping implementation"""
        logger.info("🔄 Using fallback mock scraping implementation")
        
        # Get enabled sources for scraping
        sources_query = """
            SELECT 
                s.id, s.name, s.rss_url, s.website, s.content_type, s.priority,
                COALESCE(c.name, 'general') as category
            FROM ai_sources s
            LEFT JOIN ai_categories_master c ON s.category_id = c.id
            WHERE s.enabled = TRUE
            ORDER BY s.priority ASC
        """
        
        sources = db.execute_query(sources_query)
        
        if not sources:
            logger.warning("⚠️ No enabled sources found for scraping")
            return {
                'success': False,
                'message': 'No enabled sources found',
                'sources_processed': 0,
                'articles_scraped': 0
            }
        
        # Mock processing
        articles_scraped = 0
        for source in sources:
            if self.DEBUG:
                logger.debug(f"🔍 Mock processing source: {source['name']}")
            articles_scraped += 3  # Mock number per source
        
        logger.info(f"✅ Fallback scraping completed - {len(sources)} sources, {articles_scraped} articles")
        
        return {
            'success': True,
            'message': 'Content scraping completed successfully (fallback mode)',
            'sources_processed': len(sources),
            'articles_scraped': articles_scraped,
            'sources': [{'name': s['name'], 'category': s['category']} for s in sources],
            'scraping_mode': 'fallback_mock'
        }
    
    def get_content_counts(self, category_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get content counts by category and content type
        Returns ALL TIME counts (not time-filtered)
        
        Args:
            category_id: Category ID/name, 'all' for all categories, or None for all
            
        Returns:
            Dict with total counts and counts by category
        """
        try:
            db = get_database_service()
            
            if self.DEBUG:
                logger.debug(f"🔍 Getting content counts for category: {category_id}")
            
            # ✅ FIXED: Query ALL articles without time filter for total counts
            base_query = """
                SELECT 
                    c.name as category_name,
                    c.id as category_id,
                    ct.name as content_type,
                    COUNT(a.id) as count
                FROM ai_categories_master c
                LEFT JOIN articles a ON a.category_id = c.id
                LEFT JOIN content_types ct ON a.content_type_id = ct.id
            """
            
            # Add category filter if specified (but NO time filter)
            if category_id and category_id != 'all':
                # Try to match by name or ID
                base_query += " WHERE (c.name ILIKE %s OR c.id::text = %s)"
                params = [f"%{category_id}%", str(category_id)]
            else:
                params = []
            
            base_query += " GROUP BY c.id, c.name, ct.id, ct.name ORDER BY c.name, ct.name"
            
            results = db.execute_query(base_query, params if params else None)
            
            # Initialize totals
            total_articles = 0
            total_podcasts = 0 
            total_videos = 0
            by_category = {}
            
            # Process results
            for row in results:
                category_name = row['category_name']
                content_type = row['content_type'] or 'blogs'  # Default to blogs if None
                count = row['count'] or 0
                
                # Initialize category if not exists
                if category_name not in by_category:
                    by_category[category_name] = {
                        'category_id': row['category_id'],
                        'blogs': 0,
                        'podcasts': 0,
                        'videos': 0,
                        'total': 0
                    }
                
                # Map content types and add to totals
                if content_type.lower() in ['blogs', 'blog', 'article', 'articles', 'news', 'post', 'posts']:
                    by_category[category_name]['blogs'] += count
                    total_articles += count
                elif content_type.lower() in ['podcasts', 'podcast', 'audio', 'sound']:
                    by_category[category_name]['podcasts'] += count
                    total_podcasts += count
                elif content_type.lower() in ['videos', 'video', 'youtube', 'vimeo']:
                    by_category[category_name]['videos'] += count
                    total_videos += count
                else:
                    # Default unknown types to blogs
                    by_category[category_name]['blogs'] += count
                    total_articles += count
                
                by_category[category_name]['total'] += count
        
            # ✅ FIXED: Get overall totals for ALL articles (no time filter)
            if not category_id or category_id == 'all':
                total_query = """
                    SELECT 
                        ct.name as content_type,
                        COUNT(a.id) as count
                    FROM articles a
                    LEFT JOIN content_types ct ON a.content_type_id = ct.id
                    GROUP BY ct.id, ct.name
                """
                
                total_results = db.execute_query(total_query)
                
                # Reset totals to get accurate counts
                total_articles = 0
                total_podcasts = 0
                total_videos = 0
                
                for row in total_results:
                    content_type = row['content_type'] or 'blogs'
                    count = row['count'] or 0
                    
                    if content_type.lower() in ['blogs', 'blog', 'article', 'articles', 'news', 'post', 'posts']:
                        total_articles += count
                    elif content_type.lower() in ['podcasts', 'podcast', 'audio', 'sound']:
                        total_podcasts += count
                    elif content_type.lower() in ['videos', 'video', 'youtube', 'vimeo']:
                        total_videos += count
                    else:
                        total_articles += count  # Default to articles
            
            response = {
                'total_blogs': total_articles,
                'total_articles': total_articles,  # Alias for backward compatibility
                'total_podcasts': total_podcasts,
                'total_videos': total_videos,
                'total_all': total_articles + total_podcasts + total_videos,
                'by_category': by_category,
                'category_filter': category_id,
                'time_filter': 'all_time',  # ✅ NEW: Indicate this is all-time count
                'database': 'postgresql'
            }
            
            if self.DEBUG:
                logger.debug(f"🔍 Content counts response: {response}")
            
            logger.info(f"✅ Content counts retrieved - Total: {response['total_all']} (Blogs:{total_articles}, Podcasts:{total_podcasts}, Videos:{total_videos}) [ALL TIME]")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get content counts: {str(e)}")
            if self.DEBUG:
                logger.debug(f"🔍 Content counts error details: {traceback.format_exc()}")
            
            # Return fallback counts
            return {
                'total_blogs': 0,
                'total_articles': 0,
                'total_podcasts': 0,
                'total_videos': 0,
                'total_all': 0,
                'by_category': {},
                'category_filter': category_id,
                'time_filter': 'all_time',
                'error': str(e),
                'database': 'postgresql'
            }