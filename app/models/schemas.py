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
    # Core user_preferences table fields
    experience_level: Optional[str] = "intermediate"  # beginner, intermediate, advanced, expert
    professional_roles: Optional[List[str]] = []  # developer, researcher, student, etc.
    categories_selected: Optional[List[str]] = []  # AI category names (not IDs)
    content_types_selected: Optional[List[str]] = ["ARTICLE", "VIDEO", "AUDIO"]  # Content type names
    publishers_selected: Optional[List[str]] = []  # Publisher names
    
    # Additional preference fields
    newsletter_frequency: Optional[str] = "weekly"
    email_notifications: Optional[bool] = True
    breaking_news_alerts: Optional[bool] = False
    onboarding_completed: Optional[bool] = False


class PasswordAuthRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    auth_mode: str = "signin"  # signin or signup


class UserSignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    confirm_password: str


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
    category_id: str
    description: Optional[str] = None
    is_active: bool = True


class ContentType(BaseModel):
    id: int
    name: str
    display_name: str
    frontend_section: Optional[str] = None
    icon: Optional[str] = None


class Article(BaseModel):
    id: int
    source: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    published_date: Optional[datetime] = None
    description: Optional[str] = None
    content_hash: Optional[str] = None
    significance_score: int = 5
    scraped_date: Optional[datetime] = None
    reading_time: int = 3
    image_url: Optional[str] = None
    keywords: Optional[str] = None
    content_type_id: Optional[int] = None
    content_type_label: Optional[str] = None
    category_id: Optional[int] = None
    category_label: Optional[str] = None

    

class ContentFilterRequest(BaseModel):
    """Request model for filtering content"""
    interests: Optional[List[str]] = []
    content_types: Optional[List[str]] = []
    publishers: Optional[List[str]] = []
    time_filter: Optional[str] = "Last Week"
    search_query: Optional[str] = ""
    limit: Optional[int] = 50


class GroupedContentResponse(BaseModel):
    """Response model for grouped content by category"""
    category: str
    items: List[Article]
    count: int


class PersonalizedFeedResponse(BaseModel):
    """Enhanced personalized feed response"""
    welcome_message: str
    user_profile: Dict[str, Any]
    grouped_content: List[GroupedContentResponse]
    search_active: bool = False
    search_query: Optional[str] = None
    total_items: int
    filters_applied: Dict[str, Any]


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