#!/usr/bin/env python3
"""
Content router for modular FastAPI architecture
Maintains compatibility with existing frontend API endpoints
"""

import os
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.models.schemas import DigestResponse, ContentByTypeResponse, UserResponse
from app.dependencies.auth import get_current_user_optional, get_current_user
from app.services.content_service import ContentService

logger = logging.getLogger(__name__)

# Get DEBUG mode
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

router = APIRouter()

if DEBUG:
    logger.debug("🔍 Content router initialized in DEBUG mode")


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
    request: Request,
    content_service: ContentService = Depends(get_content_service)
):
    """
    Admin-only endpoint to initiate AI news scraping process
    Uses Content Service for scraping operations
    """
    try:
        # Check for admin API key authentication
        admin_api_key = request.headers.get('X-Admin-API-Key')
        expected_api_key = os.getenv('VITE_ADMIN_API_KEY', 'admin-api-key-2024')
        
        if not admin_api_key or admin_api_key != expected_api_key:
            logger.warning(f"⚠️ Unauthorized admin scraping attempt")
            raise HTTPException(
                status_code=403,
                detail={
                    'error': 'Admin access required',
                    'message': 'Valid admin API key required for scraping'
                }
            )
        
        logger.info(f"🔧 Admin scraping initiated with API key authentication")
        
        if DEBUG:
            logger.debug(f"🔍 Admin scrape request with valid API key")
            logger.debug(f"🔍 Admin permissions verified")
        
        
        # Trigger scraping operation with detailed error handling
        try:
            logger.info("🔍 About to call content_service.scrape_content()")
            result = await content_service.scrape_content()
            logger.info("🔍 Successfully called content_service.scrape_content()")
        except NameError as ne:
            logger.error(f"❌ NameError in scrape_content: {str(ne)}")
            raise ne
        except Exception as se:
            logger.error(f"❌ Other error in scrape_content: {str(se)}")
            raise se
        
        if DEBUG:
            logger.debug(f"🔍 Scraping completed with result: {result}")
        
        logger.info(f"✅ Admin scraping completed successfully with API key authentication")
        return {
            'success': True,
            'message': 'Content scraping completed successfully',
            'data': result,
            'database': 'postgresql'
        }
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"❌ Admin scraping endpoint failed: {str(e)}")
        logger.error(f"❌ Full traceback: {error_traceback}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Admin scraping failed',
                'message': str(e),
                'traceback': error_traceback,
                'database': 'postgresql'
            }
        )