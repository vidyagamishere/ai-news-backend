#!/usr/bin/env python3
"""
Content router for modular FastAPI architecture
Maintains compatibility with existing frontend API endpoints
"""

import os
import logging
import re
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Request,BackgroundTasks

from app.models.schemas import DigestResponse, ContentByTypeResponse, UserResponse
from app.dependencies.auth import get_current_user_optional, get_current_user
from app.services.content_service import ContentService
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# Get environment variables
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
IS_SUMMARY = os.getenv("IS_SUMMARY", "false").lower() == "true"

# Common stopwords to exclude from search
STOPWORDS = {
    'what', 'is', 'are', 'was', 'were', 'in', 'the', 'a', 'an', 'and', 'or', 'but',
    'how', 'why', 'when', 'where', 'who', 'which', 'this', 'that', 'these', 'those',
    'do', 'does', 'did', 'can', 'could', 'should', 'would', 'will', 'has', 'have',
    'had', 'be', 'been', 'being', 'for', 'with', 'about', 'as', 'by', 'from', 'of',
    'to', 'at', 'on', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'all', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'me', 'my', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'we', 'us', 'our', 'they', 'them', 'their'
}

# Common AI/Tech phrases to keep together (phrase detection)
IMPORTANT_PHRASES = [
    'machine learning', 'deep learning', 'neural network', 'neural networks',
    'artificial intelligence', 'natural language processing', 'computer vision',
    'large language model', 'large language models', 'generative ai',
    'retrieval augmented generation', 'reinforcement learning',
    'transfer learning', 'federated learning', 'edge computing', 'quantum computing',
    'cloud computing', 'data science', 'big data', 'data analytics',
    'business intelligence', 'internet of things', 'cyber security',
    'blockchain technology', 'augmented reality', 'virtual reality',
    'foundation model', 'foundation models', 'prompt engineering',
    'fine tuning', 'model training', 'attention mechanism'
]

# Acronym expansion map for better search coverage
ACRONYM_MAP = {
    'llm': 'large language model',
    'llms': 'large language models',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'nlp': 'natural language processing',
    'cv': 'computer vision',
    'ai': 'artificial intelligence',
    'rag': 'retrieval augmented generation',
    'gpt': 'generative pretrained transformer',
    'bert': 'bidirectional encoder representations transformers',
    'gan': 'generative adversarial network',
    'gans': 'generative adversarial networks',
    'rnn': 'recurrent neural network',
    'cnn': 'convolutional neural network',
    'lstm': 'long short term memory',
    'api': 'application programming interface',
    'iot': 'internet of things',
    'ar': 'augmented reality',
    'vr': 'virtual reality',
    'gpu': 'graphics processing unit',
    'tpu': 'tensor processing unit'
}

# Simple stemming rules (basic suffix removal)
STEMMING_RULES = [
    ('ing', ''),      # running -> run
    ('ed', ''),       # trained -> train
    ('s', ''),        # models -> model
    ('ly', ''),       # quickly -> quick
    ('er', ''),       # faster -> fast
    ('est', ''),      # fastest -> fast
    ('tion', 'te'),   # generation -> generate
    ('sion', 'de'),   # decision -> decide
]

def simple_stem(word: str) -> str:
    """Apply simple stemming rules to reduce words to root form"""
    if len(word) <= 3:  # Don't stem very short words
        return word
    
    for suffix, replacement in STEMMING_RULES:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)] + replacement
    return word

def extract_search_keywords(query: str) -> List[str]:
    """
    Enhanced keyword extraction with:
    1. Phrase detection - Keeps important multi-word phrases together
    2. Acronym expansion - Expands known acronyms for better coverage
    3. Stemming - Reduces words to root form for broader matching
    4. Stopword filtering - Removes common non-meaningful words
    """
    query_lower = query.lower()
    keywords = []
    
    # Step 1: Detect and extract important phrases
    for phrase in IMPORTANT_PHRASES:
        if phrase in query_lower:
            # Add phrase as single keyword (with underscores for SQL pattern matching)
            keywords.append(phrase.replace(' ', '_'))
            # Remove phrase from query to avoid duplicate processing
            query_lower = query_lower.replace(phrase, '')
            logger.info(f"   📍 Found phrase: '{phrase}'")
    
    # Step 2: Extract individual words from remaining query
    words = re.findall(r'\w+', query_lower)
    
    # Step 3: Process each word
    for word in words:
        # Skip if already processed as part of a phrase
        if any(word in phrase.replace(' ', '_') for phrase in keywords):
            continue
        
        # Skip stopwords and very short words
        if word in STOPWORDS or len(word) < 2:
            continue
        
        # Step 4: Check for acronym expansion
        if word in ACRONYM_MAP:
            expanded = ACRONYM_MAP[word]
            keywords.append(word)  # Keep original acronym
            keywords.append(expanded.replace(' ', '_'))  # Add expanded form
            logger.info(f"   🔤 Expanded acronym: '{word}' → '{expanded}'")
        else:
            # Step 5: Apply stemming for better matching
            stemmed = simple_stem(word)
            keywords.append(stemmed)
            
            # Also keep original if different from stemmed
            if stemmed != word and len(word) > 4:
                keywords.append(word)
    
    # Step 6: Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    logger.info(f"🔑 Extracted {len(unique_keywords)} keywords from '{query}': {unique_keywords}")
    return unique_keywords

router = APIRouter()

if DEBUG:
    logger.debug("🔍 Content router initialized in DEBUG mode")
    logger.debug(f"📝 IS_SUMMARY environment variable: {IS_SUMMARY}")


def get_content_service() -> ContentService:
    """Dependency to get ContentService instance"""
    return ContentService()


@router.get("/digest", response_model=DigestResponse)
async def get_digest(
    current_user: Optional[UserResponse] = Depends(get_current_user_optional),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get general digest - compatible with existing frontend
    Endpoint: GET /digest (same as before)
    """
    try:
        logger.info(f"📊 Digest requested - User: {current_user.email if current_user else 'anonymous'}")
        
        is_personalized = bool(current_user)
        digest = content_service.get_digest(
            user_id=current_user.id if current_user else None,
            personalized=is_personalized
        )
        
        logger.info("✅ Digest generated successfully")
        return digest
        
    except Exception as e:
        logger.error(f"❌ Digest endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get digest',
                'message': str(e),
                'database': 'postgresql'
            }
        )

@router.get("/search-content")
async def search_content(
    query: str = Query(..., min_length=1, description="Search query string"),
    category_id: Optional[int] = Query(None, description="Category ID filter (None = All categories)"),
    days_filter: int = Query(7, ge=1, le=365, description="Number of days to look back"),
    limit_per_type: int = Query(20, ge=1, le=50, description="Max results per content type")
):
    """
    Search content across all types (blogs, podcasts, videos) with filters
    - Searches in: title, keywords, summary, author, source
    - Weighted ranking: title (3x), keywords (2x), summary (1x)
    - Respects category and time filters
    - Returns all content types with counts for tab badges
    
    Endpoint: GET /search-content
    Query params: 
      - query: search term (required)
      - category_id: filter by category (optional, None = All)
      - days_filter: time range in days (default: 7)
      - limit_per_type: max results per type (default: 20)
    """
    try:
        logger.info(f"🔍 Search requested - Query: '{query}', Category: {category_id}, Days: {days_filter}")
        
        from db_service import get_database_service
        from datetime import datetime, timedelta
        db = get_database_service()
        
        # Extract keywords from query for better matching
        search_keywords = extract_search_keywords(query)
        
        # If no keywords extracted, fall back to original query
        if not search_keywords:
            search_keywords = [query]
            logger.warning(f"⚠️ No keywords extracted, using full query: '{query}'")
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_filter)
        
        # Get category name if filtering by category
        category_name = "All Categories"
        if category_id:
            cat_query = "SELECT name FROM ai_categories_master WHERE id = %s"
            cat_result = db.execute_query(cat_query, (category_id,), fetch_one=True)
            if cat_result:
                category_name = cat_result['name']
        
        # Build dynamic WHERE clause for multiple keywords
        # Each keyword searches across all fields with OR logic
        keyword_conditions = []
        for _ in search_keywords:
            keyword_conditions.append("""(
                LOWER(a.title) LIKE LOWER(%s)
                OR LOWER(a.summary) LIKE LOWER(%s)
                OR (a.keywords IS NOT NULL AND LOWER(a.keywords) LIKE LOWER(%s))
                OR LOWER(a.author) LIKE LOWER(%s)
                OR LOWER(a.source) LIKE LOWER(%s)
            )""")
        
        where_clause = " OR ".join(keyword_conditions)
        
        # Build relevance scoring that counts keyword matches
        # More keywords matched = higher score
        relevance_parts = []
        for i in range(len(search_keywords)):
            offset = i * 5  # 5 params per keyword in SELECT
            relevance_parts.append(f"""(
                CASE WHEN LOWER(a.title) LIKE LOWER(%s) THEN 3.0 ELSE 0.0 END +
                CASE WHEN a.keywords IS NOT NULL AND LOWER(a.keywords) LIKE LOWER(%s) THEN 2.0 ELSE 0.0 END +
                CASE WHEN LOWER(a.summary) LIKE LOWER(%s) THEN 1.0 ELSE 0.0 END +
                CASE WHEN LOWER(a.author) LIKE LOWER(%s) THEN 0.5 ELSE 0.0 END +
                CASE WHEN LOWER(a.source) LIKE LOWER(%s) THEN 0.3 ELSE 0.0 END
            )""")
        
        relevance_score = " + ".join(relevance_parts) + " + (a.significance_score / 100.0)"
        
        # Build base search query with dynamic keyword matching and ranking score
        base_search_query = f"""
            SELECT 
                a.id,
                a.title,
                a.summary,
                a.url,
                a.source,
                a.significance_score,
                a.published_date,
                a.author,
                c.name as category,
                c.category_label,
                ct.name as content_type,
                ct.display_name as content_type_display,
                p.publisher_name,
                p.priority as publisher_priority,
                ({relevance_score}) AS relevance_score,
                
                -- DYNAMIC RANKING SCORE (hybrid approach)
                (
                    COALESCE(a.static_score_component, 0.375) +
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
            LEFT JOIN ai_categories_master c ON a.category_id = c.id
            LEFT JOIN content_types ct ON a.content_type_id = ct.id
            LEFT JOIN publishers_master p ON a.publisher_id = p.id
            WHERE 
                ({where_clause})
                AND a.published_date >= %s
                {{category_filter}}
                AND a.content_type_id = %s
            ORDER BY relevance_score DESC, ranking_score DESC, a.published_date DESC
            LIMIT %s
        """
        
        # Category filter clause
        category_filter = "AND a.category_id = %s" if category_id else ""
        
        # Build the complete SQL query with category filter
        final_search_query = base_search_query.format(category_filter=category_filter)
        
        # Helper function to execute search for a specific content type
        def execute_search_for_type(content_type_id: int, content_type_name: str):
            """Execute search query for a specific content type"""
            # Build parameter list dynamically based on keywords
            params = []
            
            # Add parameters for relevance scoring (SELECT clause)
            for keyword in search_keywords:
                # Convert underscores back to spaces for phrase matching
                search_term = keyword.replace('_', ' ')
                pattern = f"%{search_term}%"
                params.extend([pattern, pattern, pattern, pattern, pattern])  # title, keywords, summary, author, source
            
            # Add parameters for WHERE clause
            for keyword in search_keywords:
                # Convert underscores back to spaces for phrase matching
                search_term = keyword.replace('_', ' ')
                pattern = f"%{search_term}%"
                params.extend([pattern, pattern, pattern, pattern, pattern])  # title, summary, keywords, author, source
            
            # Add date filter
            params.append(cutoff_date)
            
            # Add category_id if filtering by category
            if category_id:
                params.append(category_id)
            
            # Add content_type_id and limit
            params.append(content_type_id)
            params.append(limit_per_type)
            
            logger.info(f"🔍 Searching content_type_id={content_type_id} ({content_type_name}) with {len(search_keywords)} keywords, {len(params)} params")
            
            results = db.execute_query(final_search_query, tuple(params), fetch_all=True)
            
            # Log what was actually returned
            if results:
                returned_types = set(r.get('content_type') for r in results)
                logger.info(f"   → Got {len(results)} results, content_types returned: {returned_types}")
                if len(returned_types) > 1 or (returned_types and list(returned_types)[0] != content_type_name):
                    logger.warning(f"   ⚠️ WARNING: Expected '{content_type_name}', but got: {returned_types}")
            else:
                logger.info(f"   → No results found for {content_type_name}")
            
            return [{
                'id': r['id'],
                'title': r['title'],
                'summary': (r['summary'] or '') if IS_SUMMARY else '',
                'url': r['url'],
                'source': r['source'],
                'significanceScore': float(r['significance_score']) if r['significance_score'] else 7.0,
                'published_date': r['published_date'].isoformat() if r['published_date'] else None,
                'author': r['author'] or '',
                'category': r['category'] or 'General',
                'category_label': r['category_label'] or 'general',
                'content_type': r['content_type'],
                'content_type_display': r['content_type_display'],
                'relevance_score': float(r['relevance_score']),
                'ranking_score': float(r['ranking_score']) if r.get('ranking_score') else 0.0
            } for r in results]
        
        # Execute searches for each content type separately
        blogs = execute_search_for_type(1, 'blogs')
        podcasts = execute_search_for_type(3, 'podcasts')
        videos = execute_search_for_type(2, 'videos')
        
        # Validate results - ensure each array only contains its own type
        blogs = [b for b in blogs if b.get('content_type') == 'blogs']
        podcasts = [p for p in podcasts if p.get('content_type') == 'podcasts']
        videos = [v for v in videos if v.get('content_type') == 'videos']
        
        logger.info(f"✅ Search complete - Found: {len(blogs)} blogs, {len(podcasts)} podcasts, {len(videos)} videos")
        
        return {
            'query': query,
            'category': category_name,
            'category_id': category_id,
            'days_filter': days_filter,
            'results': {
                'blogs': blogs,
                'podcasts': podcasts,
                'videos': videos
            },
            'counts': {
                'blogs': len(blogs),
                'podcasts': len(podcasts),
                'videos': len(videos),
                'total': len(blogs) + len(podcasts) + len(videos)
            },
            'metadata': {
                'search_type': 'weighted_fulltext',
                'filters_applied': {
                    'category': category_name,
                    'time_range_days': days_filter
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Search endpoint failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Search failed',
                'message': str(e)
            }
        )

@router.get("/personalized-digest", response_model=DigestResponse)
async def get_personalized_digest(
    current_user: UserResponse = Depends(get_current_user_optional),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get personalized digest - requires authentication
    Endpoint: GET /personalized-digest (same as before)
    """
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail={
                'error': 'Authentication required',
                'message': 'Please log in to access personalized content'
            }
        )
    
    try:
        logger.info(f"📊 Personalized digest requested - User: {current_user.email}")
        
        digest = content_service.get_digest(
            user_id=current_user.id,
            personalized=True
        )
        
        logger.info("✅ Personalized digest generated successfully")
        return digest
        
    except Exception as e:
        logger.error(f"❌ Personalized digest endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get personalized digest',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/content/{content_type}", response_model=ContentByTypeResponse)
async def get_content_by_type(
    content_type: str,
    limit: int = Query(20, ge=1, le=100),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get content by type - compatible with existing frontend
    Endpoint: GET /content/{type} (same as before)
    """
    try:
        logger.info(f"📄 Content by type requested - Type: {content_type}, Limit: {limit}")
        
        articles = content_service.get_content_by_type(content_type, limit)
        
        response = ContentByTypeResponse(
            articles=articles,
            content_type=content_type,
            count=len(articles),
            database="postgresql"
        )
        
        logger.info(f"✅ Content by type generated successfully - {len(articles)} articles")
        return response
        
    except Exception as e:
        logger.error(f"❌ Content by type endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get content',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/ai-topics")
async def get_ai_topics(
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get all AI topics and content types
    Endpoint: GET /ai-topics (new endpoint for frontend)
    """
    try:
        logger.info("📑 AI categories and content types requested")

        categories = content_service.get_ai_categories()
        content_types = content_service.get_content_types()

        logger.info(f"✅ AI categories retrieved successfully - {len(categories)} categories")
        logger.info(f"✅ Content types retrieved successfully - {len(content_types)} types")
        
        return {
            'categories': categories,
            'content_types': content_types,
            'count': len(categories),
            'database': 'postgresql'
        }
        
    except Exception as e:
        logger.error(f"❌ AI categories endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get AI categories',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/content-types")
async def get_content_types(
    content_service: ContentService = Depends(get_content_service)
):
    """
    Get all content types
    Endpoint: GET /content-types (new endpoint for frontend)
    """
    try:
        logger.info("📋 Content types requested")
        
        content_types = content_service.get_content_types()
        
        logger.info(f"✅ Content types retrieved successfully - {len(content_types)} types")
        return {
            'content_types': {ct['name']: ct for ct in content_types},
            'count': len(content_types),
            'database': 'postgresql'
        }
        
    except Exception as e:
        logger.error(f"❌ Content types endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get content types',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/content-counts")
async def get_content_counts(
    category_id: str = Query("all", description="Category ID or 'all' for all categories"),
    time_filter: str = Query("All Time", description="Time filter: 'Last 24 Hours', 'Last Week', 'Last Month', 'This Year', 'All Time'")
):
    """
    Get content counts by category and content type
    ✅ NOW SUPPORTS TIME FILTERING - returns counts matching the selected time filter
    """
    try:
        logger.info(f"📊 Content counts requested for category: {category_id}, time_filter: {time_filter}")
        
        # Get counts from content service WITH time filtering
        content_service = ContentService()
        counts = content_service.get_content_counts(category_id, time_filter)
        
        logger.info(f"✅ Content counts retrieved successfully for time filter: {time_filter}")
        return counts
        
    except Exception as e:
        logger.error(f"❌ Failed to get content counts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get content counts: {str(e)}"
        )


@router.get("/breaking-news")
async def get_breaking_news_alerts(
    limit: int = Query(5, ge=1, le=10)
):
    """
    Get breaking news alerts for pre-login landing page
    Returns high significance score Generative AI articles from last 24 hours
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /breaking-news
    Query params: limit (default: 5, max: 10)
    """
    try:
        logger.info(f"🚨 Breaking news requested - Limit: {limit}")
        
        from db_service import get_database_service
        db = get_database_service()
        
        # Use articles with high significance scores for breaking news
        query = """
            SELECT 
                a.title, 
                a.summary, 
                a.url, 
                a.source, 
                a.significance_score, 
                a.published_date, 
                ct.name as content_type_name, 
                cm.category_label,
                p.publisher_name,
                p.priority as publisher_priority,
                
                -- DYNAMIC RANKING SCORE
                (
                    COALESCE(a.static_score_component, 0.375) +
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
            LEFT JOIN ai_categories_master cm ON a.category_id = cm.id
            LEFT JOIN publishers_master p ON a.publisher_id = p.id
            WHERE a.significance_score >= 8.0
            AND a.published_date >= NOW() - INTERVAL '3 days'
            ORDER BY ranking_score DESC, a.published_date DESC
            LIMIT %s
        """
        
        articles = db.execute_query(query, (limit,), fetch_all=True)
        
        result = []
        for article in articles:
            result.append({
                'title': article['title'],
                'summary': (article['summary'] or '') if IS_SUMMARY else '',
                'url': article['url'],
                'source': article['source'],
                'significanceScore': float(article['significance_score']) if article['significance_score'] else 8.5,
                'published_date': article['published_date'].isoformat() if article['published_date'] else None,
                'content_type': article['content_type_name'],
                'category': 'Generative AI',
                'category_label': article.get('category_label', 'generative_ai'),
                'ranking_score': float(article.get('ranking_score', 0.0))
            })
        
        logger.info(f"✅ Breaking news retrieved: {len(result)} articles")
        return {
            'articles': result,
            'count': len(result),
            'type': 'breaking_news'
        }
        
    except Exception as e:
        logger.error(f"❌ Breaking news endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get breaking news',
                'message': str(e)
            }
        )


@router.get("/generative-ai-content")
async def get_generative_ai_stories(
    limit: int = Query(10, ge=1, le=20)
):
    """
    Get Generative AI category stories for pre-login landing page
    Returns curated AI articles focused on OpenAI, ChatGPT, Claude, etc.
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /generative-ai-content  
    Query params: limit (default: 10, max: 20)
    """
    try:
        logger.info(f"🤖 Generative AI content requested - Limit: {limit}")
        
        from db_service import get_database_service
        db = get_database_service()
        
        # Get articles focused on Generative AI with composite ranking
        query = """
            SELECT 
                a.title, 
                a.summary, 
                a.url, 
                a.source, 
                a.significance_score, 
                a.published_date, 
                a.author, 
                c.name as category, 
                c.category_label,
                p.publisher_name,
                p.priority as publisher_priority,
                
                -- DYNAMIC RANKING SCORE
                (
                    COALESCE(a.static_score_component, 0.375) +
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
            LEFT JOIN ai_categories_master c ON a.category_id = c.id
            LEFT JOIN publishers_master p ON a.publisher_id = p.id
            WHERE (c.name = 'Generative AI')
            ORDER BY ranking_score DESC, a.published_date DESC
            LIMIT %s
        """
        
        articles = db.execute_query(query, (limit,), fetch_all=True)
        
        result = []
        for article in articles:
            result.append({
                'title': article['title'],
                'summary': (article['summary'] or '') if IS_SUMMARY else '',
                'url': article['url'],
                'source': article['source'],
                'significanceScore': float(article['significance_score']) if article['significance_score'] else 7.0,
                'published_date': article['published_date'].isoformat() if article['published_date'] else None,
                'author': article['author'],
                'category': article['category'] or 'Generative AI',
                'ranking_score': float(article.get('ranking_score', 0.0))
            })
        
        logger.info(f"✅ Generative AI content retrieved: {len(result)} articles")
        return {
            'articles': result,
            'count': len(result),
            'type': 'generative_ai'
        }
        
    except Exception as e:
        logger.error(f"❌ Generative AI content endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get Generative AI content',
                'message': str(e)
            }
        )


@router.get("/ai-applications-content")
async def get_ai_applications_stories(
    limit: int = Query(10, ge=1, le=20)
):
    """
    Get AI Applications category stories for pre-login landing page
    Returns curated AI articles focused on enterprise use cases, industry solutions
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /ai-applications-content  
    Query params: limit (default: 10, max: 20)
    """
    try:
        logger.info(f"AI Applications content requested - Limit: {limit}")
        
        from db_service import get_database_service
        db = get_database_service()
        
    
        query = """
            SELECT 
                a.title, 
                a.summary, 
                a.url, 
                a.source, 
                a.significance_score, 
                a.published_date, 
                a.author, 
                'AI Applications' as category,
                p.publisher_name,
                p.priority as publisher_priority,
                
                -- DYNAMIC RANKING SCORE
                (
                    COALESCE(a.static_score_component, 0.375) +
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
            LEFT JOIN publishers_master p ON a.publisher_id = p.id
            WHERE a.category_id = 2
            ORDER BY ranking_score DESC, a.published_date DESC
            LIMIT %s
            """
        articles = db.execute_query(query, (limit,), fetch_all=True)
        
        result = []
        for article in articles:
            result.append({
                'title': article['title'],
                'summary': (article['summary'] or '') if IS_SUMMARY else '',
                'url': article['url'],
                'source': article['source'],
                'significanceScore': float(article['significance_score']) if article['significance_score'] else 7.0,
                'published_date': article['published_date'].isoformat() if article['published_date'] else None,
                'author': article.get('author', ''),
                'category': article.get('category', 'AI Applications'),
                'ranking_score': float(article.get('ranking_score', 0.0))
            })
        
        logger.info(f"✅ AI Applications content retrieved: {len(result)} articles")
        return {
            'articles': result,
            'count': len(result),
            'type': 'ai_applications'
        }
        
    except Exception as e:
        logger.error(f"❌ AI Applications content endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get AI Applications content',
                'message': str(e)
            }
        )


@router.get("/ai-startups-content")
async def get_ai_startups_stories(
    limit: int = Query(10, ge=1, le=20)
):
    """
    Get AI Startups category stories for pre-login landing page
    Returns curated AI articles focused on funding, M&A, emerging companies
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /ai-startups-content  
    Query params: limit (default: 3, max: 20)
    """
    try:
        logger.info(f"🚀 AI Startups content requested - Limit: {limit}")
        
        from db_service import get_database_service
        db = get_database_service()
        
        query = """
            SELECT 
                a.title, 
                a.summary, 
                a.url, 
                a.source, 
                a.significance_score, 
                a.published_date, 
                a.author, 
                'AI Startups' as category,
                p.publisher_name,
                p.priority as publisher_priority,
                
                -- DYNAMIC RANKING SCORE
                (
                    COALESCE(a.static_score_component, 0.375) +
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
            LEFT JOIN publishers_master p ON a.publisher_id = p.id
            WHERE a.category_id = 3
            ORDER BY ranking_score DESC, a.published_date DESC
            LIMIT %s
            """
        articles = db.execute_query(query, (limit,), fetch_all=True)
        
        result = []
        for article in articles:
            result.append({
                'title': article['title'],
                'summary': (article['summary'] or '') if IS_SUMMARY else '',
                'url': article['url'],
                'source': article['source'],
                'significanceScore': float(article['significance_score']) if article['significance_score'] else 7.0,
                'published_date': article['published_date'].isoformat() if article['published_date'] else None,
                'author': article.get('author', ''),
                'category': article.get('category', 'AI Startups'),
                'ranking_score': float(article.get('ranking_score', 0.0))
            })
        
        logger.info(f"✅ AI Startups content retrieved: {len(result)} articles")
        return {
            'articles': result,
            'count': len(result),
            'type': 'ai_startups'
        }
        
    except Exception as e:
        logger.error(f"❌ AI Startups content endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get AI Startups content',
                'message': str(e)
            }
        )


@router.get("/landing-content")
async def get_landing_content(
    limit_per_type: int = Query(50, ge=1, le=100, description="Max results per content type"),
    days_filter: int = Query(7, ge=1, le=365, description="Filter articles by days (1, 7, 30, 365)"),
    category_id: Optional[int] = Query(None, description="Filter by specific category (None = all categories)"),
    content_type_id: Optional[int] = Query(None, description="Filter by specific content type (1=blogs, 2=videos, 3=podcasts, 4=posts, 5=learning)")
):
    """
    Get all categories and content types for landing page
    Returns content organized by category (Generative AI, AI Applications, AI Startups)
    Each category contains content types (blogs, podcasts, videos) with max 100 items per type
    No authentication required - public endpoint for landing page
    
    Endpoint: GET /landing-content
    Query params: 
        - limit_per_type (default: 50, max: 100)
        - days_filter (default: 7, filter by published date in last N days)
        - category_id (optional: filter by specific category, None = all)
        - content_type_id (optional: filter by specific content type, None = all)
    """
    try:
        logger.info(f"🏠 Landing content requested - Limit: {limit_per_type}, Days: {days_filter}, Category: {category_id}, ContentType: {content_type_id}")
        
        from db_service import get_database_service
        db = get_database_service()
        
        # Get all categories sorted by priority (or filter by specific category)
        if category_id is not None:
            categories_query = """
                SELECT id, name, priority, description
                FROM ai_categories_master
                WHERE id = %s
                ORDER BY priority ASC
            """
            categories = db.execute_query(categories_query, (category_id,), fetch_all=True)
        else:
            categories_query = """
                SELECT id, name, priority, description
                FROM ai_categories_master
                ORDER BY priority ASC
            """
            categories = db.execute_query(categories_query, fetch_all=True)
        
        # If no categories, create default ones
        if not categories:
            categories = [
                {'id': 1, 'name': 'Generative AI', 'priority': 1, 'description': 'LLMs, GPT, Claude, and AI Generation'},
                {'id': 2, 'name': 'AI Applications', 'priority': 2, 'description': 'Enterprise Use Cases & Industry Solutions'},
                {'id': 3, 'name': 'AI Startups', 'priority': 3, 'description': 'Funding, M&A & Emerging Companies'}
            ]
        
        # Get content types (or filter by specific content type)
        if content_type_id is not None:
            content_types_query = """
                SELECT id, name, display_name, frontend_section
                FROM content_types
                WHERE is_active = TRUE AND id = %s
                ORDER BY display_order ASC
            """ 
            content_types = db.execute_query(content_types_query, (content_type_id,), fetch_all=True)
        else:
            content_types_query = """
                SELECT id, name, display_name, frontend_section
                FROM content_types
                WHERE is_active = TRUE
                ORDER BY display_order ASC
            """
            content_types = db.execute_query(content_types_query, fetch_all=True)
        if not content_types:
            content_types = [
                {'id': 1, 'name': 'blogs', 'display_name': 'Blogs', 'frontend_section': 'blog'},
                {'id': 2, 'name': 'videos', 'display_name': 'Videos', 'frontend_section': 'video'},
                {'id': 3, 'name': 'podcasts', 'display_name': 'Podcasts', 'frontend_section': 'podcast'}
            ]
        
        logger.info(f"📊 Processing {len(categories)} categories and {len(content_types)} content types")
        logger.info(f"📋 Content types to fetch: {[ct['name'] for ct in content_types]}")
        
        result = {
            'categories': [],
            'total_categories': len(categories)
        }
        
        for category in categories:
            # Always initialize all three content type arrays for consistent response structure
            category_data = {
                'id': category['id'],
                'name': category['name'],
                'priority': category['priority'],
                'description': category.get('description', ''),
                'content': {
                    'blogs': [],
                    'podcasts': [],
                    'videos': []
                }
            }
            logger.info(f"🔍 Processing category: {category['name']} (ID: {category['id']})")
            
            # For each content type, get articles
            for content_type in content_types:
                # Get articles for this category and content type
                logger.info(f"🔍 Fetching articles for type: {content_type['name']} (ID: {content_type['id']}) in category: {category['name']} (ID: {category['id']})")
                
                # Calculate date filter - ✅ FIXED: Use timezone-aware datetime for consistency
                from datetime import datetime, timedelta, timezone
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_filter)
                
                articles_query = """
                    SELECT 
                        a.title, 
                        a.summary, 
                        a.url, 
                        a.source, 
                        a.significance_score, 
                        a.published_date, 
                        a.author, 
                        ct.name as content_type_name,
                        cm.name as category_name,
                        p.publisher_name,
                        p.priority as publisher_priority,
                        
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
                    LEFT JOIN ai_categories_master cm ON a.category_id = cm.id
                    LEFT JOIN publishers_master p ON a.publisher_id = p.id
                    WHERE ct.id = %s 
                    AND cm.id = %s
                    AND (a.published_date >= %s OR a.published_date IS NULL)
                    ORDER BY ranking_score DESC, a.published_date DESC
                    LIMIT %s
                """
                articles = db.execute_query(articles_query, (content_type['id'], category['id'], cutoff_date, limit_per_type), fetch_all=True)
                logger.info(f"✅ Query executed for type: {content_type['name']} in category: {category['name']} is {articles_query} ")
                # Format articles
                formatted_articles = []
                for article in articles:
                    formatted_articles.append({
                        'title': article['title'],
                        'summary': (article['summary'] or '') if IS_SUMMARY else '',
                        'url': article['url'],
                        'source': article['source'],
                        'significanceScore': float(article['significance_score']) if article['significance_score'] else 7.0,
                        'published_date': article['published_date'].isoformat() if article['published_date'] else None,
                        'author': article.get('author', ''),
                        'category': category['name'],
                        'content_type': content_type['name'],
                        'ranking_score': float(article.get('ranking_score', 0.0))
                    })
                logger.info(f"✅ Found {len(formatted_articles)} articles for type: {content_type['name']} in category: {category['name']}")
                category_data['content'][content_type['name']] = formatted_articles

            result['categories'].append(category_data)
        
        logger.info(f"✅ Landing content retrieved: {len(result['categories'])} categories, Total available: {result['total_categories']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Landing content endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get landing content',
                'message': str(e)
            }
        )

# Global scraping job tracker
scraping_jobs = {}

def track_scraping_job(job_id: str, llm_model: str, scrape_frequency: int, content_service: ContentService):
    """Background task for scraping - runs async"""
    try:
        logger.info(f"🔄 Background scraping job {job_id} starting - Setting status to 'running'")
        scraping_jobs[job_id]['status'] = 'running'
        scraping_jobs[job_id]['started_at'] = datetime.now().isoformat()
        logger.info(f"🔍 DEBUG: Status after setting to running: {scraping_jobs[job_id].get('status')}")
        
        logger.info(f"🔄 Background scraping job {job_id} started")
        
        # Run the actual scraping
        import asyncio
        result = asyncio.run(content_service.scrape_with_frequency(
            llm_model=llm_model,
            scrape_frequency=scrape_frequency
        ))
        
        logger.info(f"🔍 DEBUG: Scraping function completed, now updating status to 'completed'")
        scraping_jobs[job_id]['status'] = 'completed'
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        scraping_jobs[job_id]['result'] = result
        
        logger.info(f"✅ Background scraping job {job_id} completed")
        logger.info(f"🔍 DEBUG: Job status after completion: {scraping_jobs[job_id].get('status')}")
        logger.info(f"🔍 DEBUG: Full job data: {scraping_jobs[job_id]}")
        logger.info(f"🔍 DEBUG: Job {job_id} is in scraping_jobs dict: {job_id in scraping_jobs}")
        
    except Exception as e:
        logger.error(f"❌ Background scraping job {job_id} failed with error: {str(e)}")
        scraping_jobs[job_id]['status'] = 'failed'
        scraping_jobs[job_id]['error'] = str(e)
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        logger.error(f"❌ Background scraping job {job_id} failed: {str(e)}")

def track_podcast_scraping_job(job_id: str, llm_model: str):
    """Background task for podcast scraping - runs async"""
    try:
        scraping_jobs[job_id]['status'] = 'running'
        scraping_jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        logger.info(f"🎧 Background podcast scraping job {job_id} started")
        
        # Import and run scraper
        from crawl4ai_scraper import Crawl4AIScraper
        scraper = Crawl4AIScraper()
        
        import asyncio
        result = asyncio.run(scraper.scrape_pending_podcasts(llm_model=llm_model))
        
        scraping_jobs[job_id]['status'] = 'completed'
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        scraping_jobs[job_id]['result'] = result
        
        logger.info(f"✅ Background podcast scraping job {job_id} completed")
        
    except Exception as e:
        scraping_jobs[job_id]['status'] = 'failed'
        scraping_jobs[job_id]['error'] = str(e)
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        logger.error(f"❌ Background podcast scraping job {job_id} failed: {str(e)}")

def track_video_scraping_job(job_id: str, llm_model: str):
    """Background task for video scraping - runs async"""
    try:
        scraping_jobs[job_id]['status'] = 'running'
        scraping_jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        logger.info(f"🎥 Background video scraping job {job_id} started")
        
        # Import and run scraper
        from crawl4ai_scraper import Crawl4AIScraper
        scraper = Crawl4AIScraper()
        
        import asyncio
        result = asyncio.run(scraper.scrape_pending_videos(llm_model=llm_model))
        
        scraping_jobs[job_id]['status'] = 'completed'
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        scraping_jobs[job_id]['result'] = result
        
        logger.info(f"✅ Background video scraping job {job_id} completed")
        
    except Exception as e:
        scraping_jobs[job_id]['status'] = 'failed'
        scraping_jobs[job_id]['error'] = str(e)
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        logger.error(f"❌ Background video scraping job {job_id} failed: {str(e)}")

def track_podcast_scraping_job(job_id: str, llm_model: str):
    """Background task for podcast scraping - runs async"""
    try:
        scraping_jobs[job_id]['status'] = 'running'
        scraping_jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        logger.info(f"🎧 Background podcast scraping job {job_id} started")
        
        # Import and run scraper
        from crawl4ai_scraper import Crawl4AIScraper
        scraper = Crawl4AIScraper()
        
        import asyncio
        result = asyncio.run(scraper.scrape_pending_podcasts(llm_model=llm_model))
        
        scraping_jobs[job_id]['status'] = 'completed'
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        scraping_jobs[job_id]['result'] = result
        
        logger.info(f"✅ Background podcast scraping job {job_id} completed")
        
    except Exception as e:
        scraping_jobs[job_id]['status'] = 'failed'
        scraping_jobs[job_id]['error'] = str(e)
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        logger.error(f"❌ Background podcast scraping job {job_id} failed: {str(e)}")

def track_video_scraping_job(job_id: str, llm_model: str):
    """Background task for video scraping - runs async"""
    try:
        scraping_jobs[job_id]['status'] = 'running'
        scraping_jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        logger.info(f"🎥 Background video scraping job {job_id} started")
        
        # Import and run scraper
        from crawl4ai_scraper import Crawl4AIScraper
        scraper = Crawl4AIScraper()
        
        import asyncio
        result = asyncio.run(scraper.scrape_pending_videos(llm_model=llm_model))
        
        scraping_jobs[job_id]['status'] = 'completed'
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        scraping_jobs[job_id]['result'] = result
        
        logger.info(f"✅ Background video scraping job {job_id} completed")
        
    except Exception as e:
        scraping_jobs[job_id]['status'] = 'failed'
        scraping_jobs[job_id]['error'] = str(e)
        scraping_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        logger.error(f"❌ Background video scraping job {job_id} failed: {str(e)}")

@router.post("/admin/scrape")
async def admin_initiate_scraping(
    request: Request,
    background_tasks: BackgroundTasks,
    llm_model: str = Query('claude', description="LLM model to use"),
    scrape_frequency: int = Query(1, description="Scrape frequency in days (1, 7, or 30)"),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Admin-only endpoint to initiate AI news scraping process (ASYNC)
    Uses FREQUENCY-BASED scraping from ai_sources table with scrape_frequency_days filter
    
    Returns immediately with job_id, use /admin/scrape-status/{job_id} to check progress
    """
    try:
        # Check for admin API key authentication
        admin_api_key = request.headers.get('X-Admin-API-Key')
        expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')
        
        if not admin_api_key or admin_api_key != expected_api_key:
            raise HTTPException(status_code=403, detail='Admin access required')
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # ✅ FIX: Use helper method to create complete job status with frequency-based mode
        from db_service import get_database_service
        db = get_database_service()
        scraping_jobs[job_id] = db.create_scrape_job_status(
            job_id=job_id,
            scrape_frequency=scrape_frequency,  # 1, 7, or 30 for frequency-based scraping
            content_type='articles'
        )
        scraping_jobs[job_id]['llm_model'] = llm_model
        
        # Start background task (returns immediately)
        background_tasks.add_task(
            track_scraping_job,
            job_id=job_id,
            llm_model=llm_model,
            scrape_frequency=scrape_frequency,
            content_service=content_service
        )
        
        logger.info(f"✅ Scraping job {job_id} queued successfully (frequency-based: {scrape_frequency}-day sources)")
        
        return {
            'success': True,
            'job_id': job_id,
            'message': f'Scraping job started for {scrape_frequency}-day frequency using {llm_model}',
            'status': 'queued',
            'scraping_mode': 'frequency_based',
            'check_status_url': f'/api/content/admin/scrape-status/{job_id}',
            'estimated_duration_minutes': '3-5'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start scraping job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/scrape-status/{job_id}")
async def get_scraping_status(
    request: Request,
    job_id: str
):
    """
    Check status of a scraping job
    Returns: queued | running | completed | failed
    """
    try:
        # Check for admin API key
        admin_api_key = request.headers.get('X-Admin-API-Key')
        expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')
        
        if not admin_api_key or admin_api_key != expected_api_key:
            raise HTTPException(status_code=403, detail='Admin access required')
        
        if job_id not in scraping_jobs:
            raise HTTPException(status_code=404, detail='Job not found')
        
        # ✅ DEBUG: Log the raw job data from scraping_jobs
        raw_job_data = scraping_jobs.get(job_id)
        logger.info(f"🔍 DEBUG: Raw job data for {job_id}: status={raw_job_data.get('status')}, completed_at={raw_job_data.get('completed_at')}")
        
        # ✅ FIX: Use safe job status retrieval
        from db_service import get_database_service
        db = get_database_service()
        job = db.get_safe_job_status(raw_job_data)
        
        # Add user-friendly description based on scraping mode
        scraping_description = ""
        if job['scraping_mode'] == 'one_time_pending':
            scraping_description = f"One-time scraping of pending {job['content_type']}"
        elif job['scraping_mode'] == 'frequency_based':
            scraping_description = f"Frequency-based scraping ({job['scrape_frequency']}-day sources)"
        
        return {
            'job_id': job_id,
            'status': job['status'],
            'llm_model': job.get('llm_model', 'unknown'),
            'scrape_frequency': job['scrape_frequency'],
            'content_type': job['content_type'],
            'scraping_mode': job['scraping_mode'],
            'scraping_description': scraping_description,
            'created_at': job.get('created_at'),
            'started_at': job.get('started_at'),
            'completed_at': job.get('completed_at'),
            'total_sources': job.get('total_sources', 0),
            'processed_sources': job.get('processed_sources', 0),
            'total_items': job.get('total_items', 0),
            'new_items': job.get('new_items', 0),
            'duplicate_items': job.get('duplicate_items', 0),
            'error_count': job.get('error_count', 0),
            'result': job.get('result'),
            'error': job.get('error'),
            'progress_percentage': job.get('progress_percentage', 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ NEW: Manual Scheduler Trigger Endpoint
@router.post("/admin/trigger-scheduler")
async def admin_trigger_scheduler(
    request: Request
):
    """
    Admin-only endpoint to manually trigger the scheduled scraping job
    This bypasses the 12-hour schedule and runs the job immediately
    Useful for testing and on-demand content updates
    
    Returns:
    - success: boolean
    - message: status message
    - llm_model: model that will be used (gemini)
    """
    try:
        # Check for admin API key authentication
        admin_api_key = request.headers.get('X-Admin-API-Key')
        expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')
        
        if not admin_api_key or admin_api_key != expected_api_key:
            logger.warning(f"⚠️ Unauthorized scheduler trigger attempt")
            raise HTTPException(
                status_code=403,
                detail={
                    'error': 'Admin access required',
                    'message': 'Valid admin API key required for scheduler trigger'
                }
            )
        
        logger.info(f"🔧 Admin triggered manual scheduler execution")
        
        # Import scheduler service
        from app.services.scheduler_service import scheduler_service
        
        # Trigger the scheduler manually
        success = scheduler_service.trigger_now()
        
        if success:
            logger.info(f"✅ Scheduler triggered successfully")
            return {
                'success': True,
                'message': 'Scheduled scraping job triggered successfully. Job will run immediately.',
                'llm_model_used': 'gemini',
                'schedule': '12 hours interval',
                'database': 'postgresql'
            }
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    'error': 'Scheduler trigger failed',
                    'message': 'Failed to trigger scheduler. Check server logs.'
                }
            )
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"❌ Scheduler trigger endpoint failed: {str(e)}")
        logger.error(f"❌ Full traceback: {error_traceback}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Scheduler trigger failed',
                'message': str(e),
                'traceback': error_traceback,
                'database': 'postgresql'
            }
        )


@router.post("/admin/scrape-pending-podcasts")
async def scrape_pending_podcasts(
    request: Request,
    background_tasks: BackgroundTasks,
    llm_model: str = Query('gemini', description="LLM model to use: 'claude', 'gemini', or 'ollama'")
):
    """
    Admin-only endpoint to scrape pending podcasts (one-time scraping based on scraped_status='pending')
    Returns immediately with job_id, use /admin/scrape-status/{job_id} to check progress
    
    Endpoint: POST /admin/scrape-pending-podcasts
    Headers: X-Admin-API-Key (required)
    Query params: llm_model (default: gemini)
    """
    try:
        # Check for admin API key authentication
        admin_api_key = request.headers.get('X-Admin-API-Key')
        expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')
        
        if not admin_api_key or admin_api_key != expected_api_key:
            logger.warning(f"⚠️ Unauthorized podcast scraping attempt")
            raise HTTPException(status_code=403, detail='Admin access required')
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # ✅ FIX: Use helper method with scrape_frequency=0 for one-time scraping
        from db_service import get_database_service
        db = get_database_service()
        scraping_jobs[job_id] = db.create_scrape_job_status(
            job_id=job_id,
            scrape_frequency=0,  # 0 = one-time scraping (not frequency-based)
            content_type='podcasts'
        )
        scraping_jobs[job_id]['llm_model'] = llm_model
        scraping_jobs[job_id]['type'] = 'podcast'
        
        # Start background task (returns immediately)
        background_tasks.add_task(
            track_podcast_scraping_job,
            job_id=job_id,
            llm_model=llm_model
        )
        
        logger.info(f"🎧 Podcast scraping job {job_id} queued with LLM: {llm_model} (one-time pending scraping)")
        
        return {
            'success': True,
            'job_id': job_id,
            'message': f'Podcast scraping job started using {llm_model} (one-time pending items)',
            'status': 'queued',
            'scraping_mode': 'one_time_pending',
            'check_status_url': f'/api/content/admin/scrape-status/{job_id}',
            'estimated_duration': 'Varies by podcast count'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"❌ Podcast scraping endpoint failed: {str(e)}")
        logger.error(f"❌ Full traceback: {error_traceback}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Podcast scraping failed',
                'message': str(e),
                'traceback': error_traceback,
                'database': 'postgresql'
            }
        )


@router.post("/admin/scrape-pending-videos")
async def scrape_pending_videos(
    request: Request,
    background_tasks: BackgroundTasks,
    llm_model: str = Query('gemini', description="LLM model to use: 'claude', 'gemini', or 'ollama'")
):
    """
    Admin-only endpoint to scrape pending videos (one-time scraping based on scraped_status='pending')
    Returns immediately with job_id, use /admin/scrape-status/{job_id} to check progress
    
    Endpoint: POST /admin/scrape-pending-videos
    Headers: X-Admin-API-Key (required)
    Query params: llm_model (default: gemini)
    """
    try:
        # Check for admin API key authentication
        admin_api_key = request.headers.get('X-Admin-API-Key')
        expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')
        
        if not admin_api_key or admin_api_key != expected_api_key:
            logger.warning(f"⚠️ Unauthorized video scraping attempt")
            raise HTTPException(status_code=403, detail='Admin access required')
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # ✅ FIX: Use helper method with scrape_frequency=0 for one-time scraping
        from db_service import get_database_service
        db = get_database_service()
        scraping_jobs[job_id] = db.create_scrape_job_status(
            job_id=job_id,
            scrape_frequency=0,  # 0 = one-time scraping (not frequency-based)
            content_type='videos'
        )
        scraping_jobs[job_id]['llm_model'] = llm_model
        scraping_jobs[job_id]['type'] = 'video'
        
        # Start background task (returns immediately)
        background_tasks.add_task(
            track_video_scraping_job,
            job_id=job_id,
            llm_model=llm_model
        )
        
        logger.info(f"🎥 Video scraping job {job_id} queued with LLM: {llm_model} (one-time pending scraping)")
        
        return {
            'success': True,
            'job_id': job_id,
            'message': f'Video scraping job started using {llm_model} (one-time pending items)',
            'status': 'queued',
            'scraping_mode': 'one_time_pending',
            'check_status_url': f'/api/content/admin/scrape-status/{job_id}',
            'estimated_duration': 'Varies by video count'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"❌ Video scraping endpoint failed: {str(e)}")
        logger.error(f"❌ Full traceback: {error_traceback}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Video scraping failed',
                'message': str(e)
            }
        )
