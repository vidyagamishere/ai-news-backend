#!/usr/bin/env python3
"""
Pydantic models for request/response validation
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field


# =============================================================================
# USER MODELS
# =============================================================================

class UserBase(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    profile_image: Optional[str] = None


class UserCreate(UserBase):
    password: Optional[str] = None


class UserResponse(UserBase):
    id: str
    subscription_tier: str = "free"
    preferences: Dict[str, Any] = {}
    created_at: datetime
    verified_email: bool = False
    is_admin: bool = False
    
    class Config:
        from_attributes = True


class UserPreferences(BaseModel):
    topics: Optional[List[str]] = []
    user_roles: Optional[List[str]] = []
    role_type: Optional[str] = None
    experience_level: Optional[str] = None
    content_types: Optional[List[str]] = []
    newsletter_frequency: Optional[str] = "weekly"
    email_notifications: Optional[bool] = True
    breaking_news_alerts: Optional[bool] = False
    newsletter_subscribed: Optional[bool] = True
    onboarding_completed: Optional[bool] = False


# =============================================================================
# AUTHENTICATION MODELS
# =============================================================================

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class GoogleAuthRequest(BaseModel):
    credential: str


class OTPRequest(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    auth_mode: Optional[str] = "signin"


class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str
    userData: Optional[Dict[str, Any]] = {}


# =============================================================================
# CONTENT MODELS
# =============================================================================

class AITopic(BaseModel):
    id: str
    name: str
    category: str
    description: Optional[str] = None
    is_active: bool = True


class ContentType(BaseModel):
    id: int
    name: str
    display_name: str
    description: Optional[str] = None
    frontend_section: Optional[str] = None
    icon: Optional[str] = None


class Article(BaseModel):
    id: int
    source: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    description: Optional[str] = None
    content: Optional[str] = None
    significance_score: int = 5
    scraped_at: Optional[datetime] = None
    category: str = "general"
    reading_time: int = 3
    image_url: Optional[str] = None
    keywords: Optional[str] = None
    content_type_name: Optional[str] = None
    content_type_display: Optional[str] = None
    topics: Optional[List[AITopic]] = []
    topic_names: Optional[List[str]] = []
    topic_categories: Optional[List[str]] = []
    topic_count: int = 0
    
    class Config:
        from_attributes = True


class DigestResponse(BaseModel):
    topStories: List[Article]
    content: Dict[str, List[Article]]
    summary: Dict[str, Any]
    personalized: bool = False
    debug_info: Optional[Dict[str, Any]] = None


class ContentByTypeResponse(BaseModel):
    articles: List[Article]
    content_type: str
    count: int
    database: str = "postgresql"


# =============================================================================
# HEALTH AND STATUS MODELS
# =============================================================================

class DatabaseStats(BaseModel):
    articles: int = 0
    users: int = 0
    ai_topics: int = 0
    connection_pool: str = "active"


class HealthResponse(BaseModel):
    status: str
    version: str = "3.0.0-postgresql"
    database: str = "postgresql"
    migration_source: str = "/app/ai_news.db"
    timestamp: datetime
    database_stats: DatabaseStats


# =============================================================================
# ERROR MODELS
# =============================================================================

class ErrorResponse(BaseModel):
    error: str
    message: str
    database: str = "postgresql"
    timestamp: Optional[datetime] = None


class NotFoundResponse(BaseModel):
    error: str
    available_endpoints: List[str]
    database: str = "postgresql"
    message: str