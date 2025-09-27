#!/usr/bin/env python3
"""
Content router for modular FastAPI architecture
Maintains compatibility with existing frontend API endpoints
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from app.models.schemas import DigestResponse, ContentByTypeResponse, UserResponse
from app.dependencies.auth import get_current_user_optional
from app.services.content_service import ContentService

logger = logging.getLogger(__name__)

router = APIRouter()


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
    Get all AI topics
    Endpoint: GET /ai-topics (new endpoint for frontend)
    """
    try:
        logger.info("📑 AI topics requested")
        
        topics = content_service.get_ai_topics()
        
        logger.info(f"✅ AI topics retrieved successfully - {len(topics)} topics")
        return {
            'topics': topics,
            'count': len(topics),
            'database': 'postgresql'
        }
        
    except Exception as e:
        logger.error(f"❌ AI topics endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get AI topics',
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


@router.post("/admin/scrape")
async def admin_initiate_scraping(
    current_user: UserResponse = Depends(get_current_user_optional),
    content_service: ContentService = Depends(get_content_service)
):
    """
    Admin-only endpoint to initiate AI news scraping process
    Uses Crawl4AI + Mistral-Small-3 as specified in functional requirements
    """
    # Check if user is admin
    if not current_user or not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail={
                'error': 'Admin access required',
                'message': 'Only admin@vidyagam.com can initiate scraping'
            }
        )
    
    try:
        logger.info(f"🔧 Admin scraping initiated by: {current_user.email}")
        
        # Import the admin scraping interface
        from crawl4ai_scraper import AdminScrapingInterface
        from db_service import get_database_service
        
        # Initialize scraping interface
        db_service = get_database_service()
        admin_scraper = AdminScrapingInterface(db_service)
        
        # Start scraping process
        result = await admin_scraper.initiate_scraping(current_user.email)
        
        if result['success']:
            logger.info(f"✅ Admin scraping completed: {result['articles_processed']} articles processed")
            return {
                'success': True,
                'message': 'Scraping process completed successfully',
                'data': result,
                'admin': current_user.email
            }
        else:
            logger.error(f"❌ Admin scraping failed: {result['message']}")
            raise HTTPException(
                status_code=500,
                detail={
                    'error': 'Scraping process failed',
                    'message': result['message'],
                    'admin': current_user.email
                }
            )
            
    except Exception as e:
        logger.error(f"❌ Admin scraping endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to initiate scraping',
                'message': str(e),
                'admin': current_user.email if current_user else 'unknown'
            }
        )