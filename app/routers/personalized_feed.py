#!/usr/bin/env python3
"""
Enhanced Personalized feed endpoints for mobile-first interface
"""

import logging
from datetime import datetime, timedelta, timezone
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
    """Convert time filter string to datetime with enhanced validation and timezone awareness"""
    if not time_filter or time_filter == "All Time":
        return None
    
    # Use timezone-aware datetime (UTC)
    now = datetime.now(timezone.utc)
    
    # ✅ VERIFIED: Time filter mapping is correct - "Last 24 Hours" uses timedelta(hours=24)
    time_mappings = {
        # Frontend values (case-sensitive) - ✅ CORRECT
        "Last 24 Hours": timedelta(hours=24),  # ✅ THIS IS CORRECT - returns content from last 24 hours
        "Last Week": timedelta(days=7),
        "Last Month": timedelta(days=30),
        "This Year": timedelta(days=365),
        # Legacy values (for backward compatibility)
        "Last 24 hours": timedelta(hours=24),
        "Last Year": timedelta(days=365),
        "Today": timedelta(hours=24),
        "Yesterday": timedelta(days=1),
        "This Week": timedelta(days=7),
        "This Month": timedelta(days=30)
    }
    
    # Try exact match first
    if time_filter in time_mappings:
        time_threshold = now - time_mappings[time_filter]
        # ✅ ENHANCED LOGGING: Show both threshold and current time for debugging
        logger.info(f"⏰ Time filter matched: '{time_filter}'")
        logger.info(f"   📅 Current time (UTC): {now.isoformat()}")
        logger.info(f"   📅 Time threshold (UTC): {time_threshold.isoformat()}")
        logger.info(f"   ⏳ Delta: {time_mappings[time_filter]}")
        logger.info(f"   📊 Will return articles published AFTER: {time_threshold.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return time_threshold
    
    # Try case-insensitive match
    for key, delta in time_mappings.items():
        if time_filter.lower() == key.lower():
            time_threshold = now - delta
            logger.info(f"⏰ Time filter matched (case-insensitive): '{time_filter}' -> {delta}, threshold: {time_threshold.isoformat()}")
            return time_threshold
    
    # Default fallback with logging
    logger.warning(f"⚠️ Unknown time filter '{time_filter}', defaulting to Last Week")
    time_threshold = now - timedelta(days=7)
    logger.info(f"⏰ Using default threshold: {time_threshold.isoformat()}")
    return time_threshold


def validate_user_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize user preferences with legacy data conversion"""
    validated = {
        'categories_selected': [],
        'content_types_selected': [],
        'publishers_selected': [],
        'category_ids_selected': [],
        'content_type_ids_selected': [],
        'publisher_ids_selected': []
    }
    
    try:
        # Handle category preferences with mixed data type support
        if 'categories_selected' in preferences:
            cats = preferences['categories_selected']
            if isinstance(cats, list):
                # Check for string IDs that should be converted to integers
                if cats and all(str(cat).isdigit() for cat in cats if cat not in ['all']):
                    validated['category_ids_selected'] = [int(cat) for cat in cats if str(cat).isdigit()]
                    validated['categories_selected'] = [cat for cat in cats if not str(cat).isdigit()]
                    logger.info(f"🔧 Converted mixed category data: {len(validated['category_ids_selected'])} IDs, {len(validated['categories_selected'])} names")
                else:
                    validated['categories_selected'] = [str(cat) for cat in cats if cat]
        
        # Handle category IDs
        if 'category_ids_selected' in preferences:
            cat_ids = preferences['category_ids_selected']
            if isinstance(cat_ids, list):
                validated['category_ids_selected'].extend([int(cid) for cid in cat_ids if str(cid).isdigit()])
        
        # Handle content type preferences with mixed data type support
        if 'content_types_selected' in preferences:
            cts = preferences['content_types_selected']
            if isinstance(cts, list):
                if cts and all(str(ct).isdigit() for ct in cts if ct not in ['all']):
                    validated['content_type_ids_selected'] = [int(ct) for ct in cts if str(ct).isdigit()]
                    validated['content_types_selected'] = [ct for ct in cts if not str(ct).isdigit()]
                    logger.info(f"🔧 Converted mixed content type data: {len(validated['content_type_ids_selected'])} IDs, {len(validated['content_types_selected'])} names")
                else:
                    validated['content_types_selected'] = [str(ct) for ct in cts if ct]
        
        # Handle content type IDs
        if 'content_type_ids_selected' in preferences:
            ct_ids = preferences['content_type_ids_selected']
            if isinstance(ct_ids, list):
                validated['content_type_ids_selected'].extend([int(ctid) for ctid in ct_ids if str(ctid).isdigit()])
        
        # Handle publisher preferences
        if 'publishers_selected' in preferences:
            pubs = preferences['publishers_selected']
            if isinstance(pubs, list):
                if pubs and all(str(pub).isdigit() for pub in pubs if pub not in ['all']):
                    validated['publisher_ids_selected'] = [int(pub) for pub in pubs if str(pub).isdigit()]
                    validated['publishers_selected'] = [pub for pub in pubs if not str(pub).isdigit()]
                    logger.info(f"🔧 Converted mixed publisher data: {len(validated['publisher_ids_selected'])} IDs, {len(validated['publishers_selected'])} names")
                else:
                    validated['publishers_selected'] = [str(pub) for pub in pubs if pub]
        
        # Handle publisher IDs
        if 'publisher_ids_selected' in preferences:
            pub_ids = preferences['publisher_ids_selected']
            if isinstance(pub_ids, list):
                validated['publisher_ids_selected'].extend([int(pid) for pid in pub_ids if str(pid).isdigit()])
        
        # Remove duplicates while preserving order
        for key in ['category_ids_selected', 'content_type_ids_selected', 'publisher_ids_selected']:
            validated[key] = list(dict.fromkeys(validated[key]))  # Remove duplicates
        
        logger.info(f"✅ Validated preferences: {sum(len(v) for v in validated.values())} total items")
        return validated
        
    except Exception as e:
        logger.error(f"❌ Error validating user preferences: {str(e)}")
        return validated  # Return empty but valid structure


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
                LOWER(a.summary) LIKE %s OR
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
                description=row.get('summary'),
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
        # Extract and validate user preferences with enhanced error handling
        user_categories = []
        user_content_types = []
        user_publishers = []
        user_category_ids = []
        user_content_type_ids = []
        user_publisher_ids = []
        
        if current_user and current_user.get('preferences'):
            try:
                # Validate and sanitize preferences
                raw_prefs = current_user['preferences']
                prefs = validate_user_preferences(raw_prefs)
                logger.info(f"🔧 User preferences validated successfully")
            except Exception as pref_error:
                logger.error(f"❌ Error processing user preferences: {str(pref_error)}")
                prefs = {}  # Use empty preferences as fallback
            # Prioritize ID-based preferences when available
            user_category_ids = prefs.get('category_ids_selected', [])
            user_content_type_ids = prefs.get('content_type_ids_selected', [])
            user_publisher_ids = prefs.get('publisher_ids_selected', [])
            
            # Handle "all" selections - expand to include all available IDs
            if "all" in user_category_ids:
                # Get all category IDs from ai_categories_master
                all_categories = db.execute_query("SELECT id FROM ai_categories_master ORDER BY priority ASC")
                user_category_ids = [cat['id'] for cat in all_categories]
                logger.info(f"📂 Expanded 'all' categories to {len(user_category_ids)} category IDs")
            
            if "all" in user_content_type_ids:
                # Get all content type IDs from content_types
                all_content_types = db.execute_query("SELECT id FROM content_types WHERE is_active = TRUE ORDER BY name")
                user_content_type_ids = [ct['id'] for ct in all_content_types]
                logger.info(f"📄 Expanded 'all' content types to {len(user_content_type_ids)} content type IDs")
            
            if "all" in user_publisher_ids:
                # Get all publisher IDs from publishers_master
                all_publishers = db.execute_query("SELECT id FROM publishers_master WHERE is_active = TRUE ORDER BY priority ASC, publisher_name ASC")
                user_publisher_ids = [pub['id'] for pub in all_publishers]
                logger.info(f"📰 Expanded 'all' publishers to {len(user_publisher_ids)} publisher IDs")
            
            # Fallback to name-based preferences for backward compatibility
            user_categories = prefs.get('categories_selected', [])
            user_content_types = prefs.get('content_types_selected', [])
            user_publishers = prefs.get('publishers_selected', [])
            
            # ENHANCED FIX: Robust conversion of string IDs to integer IDs for legacy data
            # Handle mixed string/integer data with error recovery
            if user_categories:
                try:
                    # Check if all values are numeric strings
                    if all(str(cat).isdigit() for cat in user_categories if cat not in ['all']):
                        user_category_ids = [int(cat) for cat in user_categories if str(cat).isdigit()]
                        user_categories = [cat for cat in user_categories if not str(cat).isdigit()]  # Keep non-numeric
                        logger.info(f"🔧 Converted {len(user_category_ids)} legacy category strings to IDs: {user_category_ids}")
                        if user_categories:
                            logger.info(f"🔧 Retained {len(user_categories)} non-numeric category names: {user_categories}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Error converting category strings to IDs: {str(e)} - using as names")
            
            # Enhanced content type conversion with error handling
            if user_content_types:
                try:
                    # Check if all values are numeric strings
                    if all(str(ct).isdigit() for ct in user_content_types if ct not in ['all']):
                        user_content_type_ids = [int(ct) for ct in user_content_types if str(ct).isdigit()]
                        user_content_types = [ct for ct in user_content_types if not str(ct).isdigit()]  # Keep non-numeric
                        logger.info(f"🔧 Converted {len(user_content_type_ids)} legacy content type strings to IDs: {user_content_type_ids}")
                        if user_content_types:
                            logger.info(f"🔧 Retained {len(user_content_types)} non-numeric content type names: {user_content_types}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Error converting content type strings to IDs: {str(e)} - using as names")
                    
            # Enhanced publisher conversion with error handling
            if user_publishers:
                try:
                    # Check if all values are numeric strings
                    if all(str(pub).isdigit() for pub in user_publishers if pub not in ['all']):
                        user_publisher_ids = [int(pub) for pub in user_publishers if str(pub).isdigit()]
                        user_publishers = [pub for pub in user_publishers if not str(pub).isdigit()]  # Keep non-numeric
                        logger.info(f"🔧 Converted {len(user_publisher_ids)} legacy publisher strings to IDs: {user_publisher_ids}")
                        if user_publishers:
                            logger.info(f"🔧 Retained {len(user_publishers)} non-numeric publisher names: {user_publishers}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Error converting publisher strings to IDs: {str(e)} - using as names")
            
            # Handle "all" in name-based preferences as well
            if "all" in user_categories:
                all_category_names = db.execute_query("SELECT name FROM ai_categories_master ORDER BY priority ASC")
                user_categories = [cat['name'] for cat in all_category_names]
                logger.info(f"📂 Expanded 'all' category names to {len(user_categories)} categories")
            
            if "all" in user_content_types:
                all_content_type_names = db.execute_query("SELECT name FROM content_types WHERE is_active = TRUE ORDER BY name")
                user_content_types = [ct['name'] for ct in all_content_type_names]
                logger.info(f"📄 Expanded 'all' content type names to {len(user_content_types)} content types")
            
            if "all" in user_publishers:
                all_publisher_names = db.execute_query("SELECT publisher_name FROM publishers_master WHERE is_active = TRUE ORDER BY priority ASC, publisher_name ASC")
                user_publishers = [pub['publisher_name'] for pub in all_publisher_names]
                logger.info(f"📰 Expanded 'all' publisher names to {len(user_publishers)} publishers")
        
        # Determine which filters to use - prioritize user preferences for personalized feed
        # Only use request filters if they are explicitly provided and meaningful
        use_categories = user_categories  # Always use user preferences for categories
        use_content_types = user_content_types  # Always use user preferences for content types
        use_publishers = user_publishers  # Always use user preferences for publishers
        
        # Allow request to override only for search functionality
        if filter_request.search_query and filter_request.search_query.strip():
            # If there's a search query, use any explicit filters from request
            use_categories = filter_request.interests if filter_request.interests else user_categories
            use_content_types = filter_request.content_types if filter_request.content_types else user_content_types
            use_publishers = filter_request.publishers if filter_request.publishers else user_publishers
        
        # Convert names to IDs for more accurate filtering
        category_ids = []
        content_type_ids = []
        publisher_ids = []
        
        # Prioritize ID-based filtering with multiple fallback strategies
        if filter_request.category_ids:
            # Request has category IDs - use them (handle "all" case)
            category_ids = filter_request.category_ids
            if "all" in category_ids:
                all_categories = db.execute_query("SELECT id FROM ai_categories_master ORDER BY priority ASC")
                category_ids = [cat['id'] for cat in all_categories]
                logger.info(f"🔍 Request: Expanded 'all' categories to {len(category_ids)} category IDs")
        elif user_category_ids:
            # User preferences have category IDs - use them (preferred)
            category_ids = user_category_ids
            
            # CRITICAL FIX: Fetch category names for grouping when we have IDs but no names
            if not use_categories and category_ids:
                try:
                    id_placeholders = ','.join(['%s'] * len(category_ids))
                    category_names_query = f"SELECT name FROM ai_categories_master WHERE id IN ({id_placeholders}) ORDER BY priority ASC"
                    category_results = db.execute_query(category_names_query, tuple(category_ids))
                    use_categories = [cat['name'] for cat in category_results]
                    logger.info(f"🔧 CRITICAL FIX: Fetched {len(use_categories)} category names from IDs for grouping: {use_categories}")
                except Exception as e:
                    logger.error(f"❌ Error fetching category names from IDs: {str(e)}")
                    use_categories = []
        elif use_categories:
            # Convert category names to IDs (fallback)
            category_ids = content_filtering_service.get_category_ids_from_names(use_categories)
            
        if filter_request.content_type_ids:
            # Request has content type IDs - use them (handle "all" case)
            content_type_ids = filter_request.content_type_ids
            if "all" in content_type_ids:
                all_content_types = db.execute_query("SELECT id FROM content_types WHERE is_active = TRUE ORDER BY name")
                content_type_ids = [ct['id'] for ct in all_content_types]
                logger.info(f"🔍 Request: Expanded 'all' content types to {len(content_type_ids)} content type IDs")
        elif user_content_type_ids:
            # User preferences have content type IDs - use them (preferred)
            content_type_ids = user_content_type_ids
            
            # CRITICAL FIX: Fetch content type names for grouping when we have IDs but no names
            if not use_content_types and content_type_ids:
                try:
                    id_placeholders = ','.join(['%s'] * len(content_type_ids))
                    content_type_names_query = f"SELECT name FROM content_types WHERE id IN ({id_placeholders}) AND is_active = TRUE ORDER BY name"
                    content_type_results = db.execute_query(content_type_names_query, tuple(content_type_ids))
                    use_content_types = [ct['name'] for ct in content_type_results]
                    logger.info(f"🔧 CRITICAL FIX: Fetched {len(use_content_types)} content type names from IDs for grouping: {use_content_types}")
                except Exception as e:
                    logger.error(f"❌ Error fetching content type names from IDs: {str(e)}")
                    use_content_types = []
        elif use_content_types:
            # Convert content type names to IDs (fallback)
            content_type_ids = content_filtering_service.get_content_type_ids_from_names(use_content_types)
            
        if filter_request.publisher_ids:
            # Request has publisher IDs - use them (handle "all" case)
            publisher_ids = filter_request.publisher_ids
            if "all" in publisher_ids:
                all_publishers = db.execute_query("SELECT id FROM publishers_master WHERE is_active = TRUE ORDER BY priority ASC, publisher_name ASC")
                publisher_ids = [pub['id'] for pub in all_publishers]
                logger.info(f"🔍 Request: Expanded 'all' publishers to {len(publisher_ids)} publisher IDs")
        elif user_publisher_ids:
            # User preferences have publisher IDs - use them (preferred)
            publisher_ids = user_publisher_ids
            
            # CRITICAL FIX: Fetch publisher names for grouping when we have IDs but no names
            if not use_publishers and publisher_ids:
                try:
                    id_placeholders = ','.join(['%s'] * len(publisher_ids))
                    publisher_names_query = f"SELECT publisher_name FROM publishers_master WHERE id IN ({id_placeholders}) AND is_active = TRUE ORDER BY priority ASC, publisher_name ASC"
                    publisher_results = db.execute_query(publisher_names_query, tuple(publisher_ids))
                    use_publishers = [pub['publisher_name'] for pub in publisher_results]
                    logger.info(f"🔧 CRITICAL FIX: Fetched {len(use_publishers)} publisher names from IDs for grouping: {use_publishers}")
                except Exception as e:
                    logger.error(f"❌ Error fetching publisher names from IDs: {str(e)}")
                    use_publishers = []
        elif use_publishers:
            # Convert publisher names to IDs (fallback)
            publisher_ids = content_filtering_service.get_publisher_ids_from_names(use_publishers)
        
        # Create filter criteria with both name and ID-based filtering
        criteria = FilterCriteria(
            interests=use_categories,  # Keep for backward compatibility
            content_types=use_content_types,  # Keep for backward compatibility
            publishers=use_publishers,  # Keep for backward compatibility
            category_ids=category_ids,  # Preferred ID-based filtering
            content_type_ids=content_type_ids,  # Preferred ID-based filtering
            publisher_ids=publisher_ids,  # Preferred ID-based filtering
            time_filter=filter_request.time_filter or "Last Week",
            search_query=filter_request.search_query or "",
            limit=filter_request.limit or 50,
            user_id=current_user.get('id') if current_user else None
        )
        
        logger.info(f"📱 Personalized feed criteria:")
        logger.info(f"   📂 Categories: {criteria.interests} (IDs: {criteria.category_ids})")
        logger.info(f"   📄 Content Types: {criteria.content_types} (IDs: {criteria.content_type_ids})")
        logger.info(f"   📰 Publishers: {criteria.publishers} (IDs: {criteria.publisher_ids})")
        logger.info(f"📱 Using user preferences: categories={len(user_categories)}, content_types={len(user_content_types)}, publishers={len(user_publishers)}")
        
        # Get filtered content using enhanced service
        filtered_articles = content_filtering_service.get_filtered_content(criteria)
        
        # If no results found with strict filtering, try progressively less restrictive approaches
        if not filtered_articles and (criteria.interests or criteria.publishers):
            logger.info(f"📱 No results with strict filtering, trying fallback approaches...")
            
            # Try without publisher filter first (keep categories and content types)
            if criteria.publishers or criteria.publisher_ids:
                fallback_criteria = FilterCriteria(
                    interests=criteria.interests,
                    content_types=criteria.content_types,
                    publishers=[],  # Remove publisher restriction
                    category_ids=criteria.category_ids,  # Keep category IDs
                    content_type_ids=criteria.content_type_ids,  # Keep content type IDs
                    publisher_ids=[],  # Remove publisher ID restriction
                    time_filter=criteria.time_filter,
                    search_query=criteria.search_query,
                    limit=criteria.limit,
                    user_id=criteria.user_id
                )
                logger.info(f"📱 Trying without publisher filter: categories={fallback_criteria.interests}, category_ids={fallback_criteria.category_ids}")
                try:
                    filtered_articles = content_filtering_service.get_filtered_content(fallback_criteria)
                    if filtered_articles:
                        logger.info(f"✅ Publisher removal fallback successful - retrieved {len(filtered_articles)} articles")
                except Exception as fallback_error:
                    logger.error(f"❌ Publisher removal fallback failed: {str(fallback_error)}")
                    filtered_articles = []
            
            # If still no results, try with only content types (no category or publisher filter)
            if not filtered_articles:
                minimal_criteria = FilterCriteria(
                    interests=[],  # Remove category restriction
                    content_types=criteria.content_types,
                    publishers=[],  # Remove publisher restriction
                    category_ids=[],  # Remove category ID restriction
                    content_type_ids=criteria.content_type_ids,  # Keep content type IDs
                    publisher_ids=[],  # Remove publisher ID restriction
                    time_filter=criteria.time_filter,
                    search_query=criteria.search_query,
                    limit=criteria.limit,
                    user_id=criteria.user_id
                )
                logger.info(f"📱 Trying with minimal filtering: content_types={minimal_criteria.content_types}, content_type_ids={minimal_criteria.content_type_ids}")
                try:
                    filtered_articles = content_filtering_service.get_filtered_content(minimal_criteria)
                    if filtered_articles:
                        logger.info(f"✅ Minimal filtering successful - retrieved {len(filtered_articles)} articles")
                except Exception as minimal_error:
                    logger.error(f"❌ Minimal filtering failed: {str(minimal_error)}")
                    filtered_articles = []
        
        # Final safety net: If still no results after all fallbacks, get recent content
        if not filtered_articles:
            logger.warning(f"⚠️ All filtering strategies failed, applying emergency fallback...")
            try:
                emergency_criteria = FilterCriteria(
                    time_filter="Last Week",
                    limit=min(filter_request.limit or 20, 50),
                    significance_threshold=1
                )
                filtered_articles = content_filtering_service.get_filtered_content(emergency_criteria)
                if filtered_articles:
                    logger.info(f"🆘 Emergency fallback retrieved {len(filtered_articles)} articles")
                else:
                    logger.error(f"💥 Even emergency fallback returned no results")
            except Exception as emergency_error:
                logger.error(f"💥 Emergency fallback failed: {str(emergency_error)}")
                filtered_articles = []
        
        # Log final result summary
        if filtered_articles:
            logger.info(f"🎯 Personalized feed completed successfully with {len(filtered_articles)} articles")
        else:
            logger.warning(f"⚠️ Personalized feed returned empty - user may need to adjust preferences")
        
        # Determine if search is active
        search_active = bool(criteria.search_query and criteria.search_query.strip())
        
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
                    description=item.get('summary'),
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


@router.get("/available-categories")
async def get_available_categories(
    db: DatabaseService = Depends(get_database_service)
):
    """Get list of available categories with IDs for onboarding"""
    try:
        query = """
            SELECT id, name, description, priority 
            FROM ai_categories_master 
            WHERE name IS NOT NULL AND name != ''
            ORDER BY priority ASC, name ASC
        """
        results = db.execute_query(query)
        categories = []
        
        for row in results:
            try:
                categories.append({
                    "id": int(row['id']) if row['id'] is not None else 0,
                    "name": str(row['name']).strip() if row['name'] else "Unknown Category",
                    "description": str(row.get('description', '')).strip(),
                    "priority": int(row.get('priority', 999)) if row.get('priority') is not None else 999
                })
            except (ValueError, TypeError) as val_error:
                logger.warning(f"⚠️ Skipping invalid category: {row} - {str(val_error)}")
        
        return {
            "categories": categories,
            "count": len(categories)
        }
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        return {"categories": [], "count": 0}

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
    """Get list of available publishers with IDs for onboarding"""
    try:
        query = """
            SELECT id, publisher_name, category_id, priority, is_active
            FROM publishers_master 
            WHERE is_active = TRUE 
            ORDER BY priority ASC, publisher_name ASC
        """
        results = db.execute_query(query)
        publishers = []
        
        for row in results:
            try:
                publishers.append({
                    "id": int(row['id']) if row['id'] is not None else 0,
                    "name": str(row['publisher_name']).strip() if row['publisher_name'] else "Unknown Publisher",
                    "category_id": int(row['category_id']) if row['category_id'] is not None else None,
                    "priority": int(row.get('priority', 999)) if row.get('priority') is not None else 999
                })
            except (ValueError, TypeError) as val_error:
                logger.warning(f"⚠️ Skipping invalid publisher: {row} - {str(val_error)}")
        
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
    """Get list of available content types with IDs for onboarding"""
    try:
        query = """
            SELECT id, name, display_name, description, is_active
            FROM content_types 
            WHERE is_active = TRUE
            ORDER BY name
        """
        results = db.execute_query(query)
        content_types = []
        
        for row in results:
            try:
                content_types.append({
                    "id": int(row['id']) if row['id'] is not None else 0,
                    "name": str(row['name']).strip() if row['name'] else "Unknown Type",
                    "display_name": str(row.get('display_name', row['name'])).strip() if row.get('display_name') or row['name'] else "Unknown Type",
                    "description": str(row.get('description', '')).strip()
                })
            except (ValueError, TypeError) as val_error:
                logger.warning(f"⚠️ Skipping invalid content type: {row} - {str(val_error)}")
        
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
            current_user['id'], limit
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
        trending = content_filtering_service.get_trending_topics(time_filter)
        
        return {
            "trending_topics": trending,
            "count": len(trending),
            "time_filter": time_filter
        }
    except Exception as e:
        logger.error(f"Error getting trending topics: {str(e)}")
        return {"trending_topics": [], "count": 0}


@router.get("/metadata")
async def get_filtering_metadata(
    db: DatabaseService = Depends(get_database_service)
):
    """Get enhanced metadata for frontend filtering with validation and error handling"""
    try:
        # Get all categories with IDs and enhanced validation
        categories = []
        try:
            categories_query = """
                SELECT id, name, description, priority 
                FROM ai_categories_master 
                WHERE name IS NOT NULL AND name != ''
                ORDER BY priority ASC
            """
            categories_raw = db.execute_query(categories_query)
            
            for row in categories_raw:
                try:
                    # Validate and sanitize each category
                    category = {
                        "id": int(row['id']) if row['id'] is not None else 0,
                        "name": str(row['name']).strip() if row['name'] else "Unknown Category",
                        "description": str(row.get('description', '')).strip(),
                        "priority": int(row.get('priority', 999)) if row.get('priority') is not None else 999
                    }
                    categories.append(category)
                except (ValueError, TypeError) as val_error:
                    logger.warning(f"⚠️ Skipping invalid category: {row} - {str(val_error)}")
            
            logger.info(f"✅ Successfully validated {len(categories)} categories")
        except Exception as cat_error:
            logger.error(f"❌ Failed to retrieve categories: {str(cat_error)}")
        
        # Get all content types with IDs and enhanced validation
        content_types = []
        try:
            content_types_query = """
                SELECT id, name, display_name, frontend_section 
                FROM content_types 
                WHERE is_active = TRUE AND name IS NOT NULL AND name != ''
                ORDER BY name
            """
            content_types_raw = db.execute_query(content_types_query)
            
            for row in content_types_raw:
                try:
                    # Validate and sanitize each content type
                    content_type = {
                        "id": int(row['id']) if row['id'] is not None else 0,
                        "name": str(row['name']).strip() if row['name'] else "Unknown Type",
                        "display_name": str(row.get('display_name', row['name'])).strip() if row.get('display_name') or row['name'] else "Unknown Type",
                        "frontend_section": str(row.get('frontend_section', '')).strip()
                    }
                    content_types.append(content_type)
                except (ValueError, TypeError) as val_error:
                    logger.warning(f"⚠️ Skipping invalid content type: {row} - {str(val_error)}")
            
            logger.info(f"✅ Successfully validated {len(content_types)} content types")
        except Exception as ct_error:
            logger.error(f"❌ Failed to retrieve content types: {str(ct_error)}")
        
        # Get publishers with validation
        publishers = []
        try:
            publishers_query = """
                SELECT id, publisher_name, category_id, priority 
                FROM publishers_master 
                WHERE is_active = TRUE AND publisher_name IS NOT NULL AND publisher_name != ''
                ORDER BY priority ASC, publisher_name ASC
            """
            publishers_raw = db.execute_query(publishers_query)
            
            for row in publishers_raw:
                try:
                    # Validate and sanitize each publisher
                    publisher = {
                        "id": int(row['id']) if row['id'] is not None else 0,
                        "name": str(row['publisher_name']).strip() if row['publisher_name'] else "Unknown Publisher",
                        "category_id": int(row.get('category_id')) if row.get('category_id') is not None else None,
                        "priority": int(row.get('priority', 999)) if row.get('priority') is not None else 999
                    }
                    publishers.append(publisher)
                except (ValueError, TypeError) as val_error:
                    logger.warning(f"⚠️ Skipping invalid publisher: {row} - {str(val_error)}")
            
            logger.info(f"✅ Successfully validated {len(publishers)} publishers")
        except Exception as pub_error:
            logger.error(f"❌ Failed to retrieve publishers: {str(pub_error)}")
        
        # Data integrity check
        data_quality = "high" if len(categories) > 0 and len(content_types) > 0 else "low" if len(categories) == 0 and len(content_types) == 0 else "medium"
        
        response_data = {
            "categories": categories,
            "content_types": content_types,
            "publishers": publishers,
            "meta": {
                "categories_count": len(categories),
                "content_types_count": len(content_types),
                "publishers_count": len(publishers),
                "data_quality": data_quality,
                "supports_id_filtering": True,
                "supports_name_filtering": True  # Backward compatibility
            }
        }
        
        logger.info(f"🎯 Metadata endpoint returning: {len(categories)} categories, {len(content_types)} content types, {len(publishers)} publishers")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error getting filtering metadata: {str(e)}")
        # Return fallback data to prevent frontend breakage
        return {
            "categories": [],
            "content_types": [
                {"id": 1, "name": "BLOGS", "display_name": "Blog Articles", "frontend_section": ""},
                {"id": 2, "name": "VIDEOS", "display_name": "Videos", "frontend_section": ""},
                {"id": 3, "name": "PODCASTS", "display_name": "Podcasts", "frontend_section": ""}
            ],
            "publishers": [],
            "meta": {
                "error": str(e),
                "fallback_mode": True,
                "data_quality": "fallback"
            }
        }


@router.get("/publishers")
async def get_publishers(
    category_id: Optional[int] = Query(None, description="Filter publishers by category_id"),
    db: DatabaseService = Depends(get_database_service)
):
    """Get unique publishers from publishers_master table with optional category filtering"""
    try:
        # Build query with optional category filtering
        if category_id:
            publishers_query = """
                SELECT id, publisher_name, category_id, is_active, priority 
                FROM publishers_master 
                WHERE is_active = TRUE AND category_id = %s
                ORDER BY priority ASC, publisher_name ASC
            """
            publishers = db.execute_query(publishers_query, (category_id,))
        else:
            publishers_query = """
                SELECT id, publisher_name, category_id, is_active, priority 
                FROM publishers_master 
                WHERE is_active = TRUE
                ORDER BY priority ASC, publisher_name ASC
            """
            publishers = db.execute_query(publishers_query)
        
        return {
            "publishers": [
                {
                    "id": row['id'], 
                    "publisher_name": row['publisher_name'],
                    "category_id": row.get('category_id'),
                    "is_active": row.get('is_active', True),
                    "priority": row.get('priority', 1)
                } 
                for row in publishers
            ],
            "total_count": len(publishers)
        }
        
    except Exception as e:
        logger.error(f"Error getting publishers: {str(e)}")
        return {"publishers": [], "total_count": 0}


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