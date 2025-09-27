"""
Authentication and user management models for AI News Scraper
"""
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class SubscriptionTier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"

class ContentType(str, Enum):
    ARTICLES = "articles"
    PODCASTS = "podcasts"
    VIDEOS = "videos"
    EVENTS = "events"
    LEARNING = "learning"

class TopicCategory(str, Enum):
    TECHNOLOGY = "technology"
    RESEARCH = "research"
    INDUSTRY = "industry"
    ETHICS = "ethics"
    TOOLS = "tools"

class AITopic(BaseModel):
    id: str
    name: str
    description: str
    category: TopicCategory
    selected: bool = False

class UserPreferences(BaseModel):
    topics: List[AITopic] = []
    newsletter_frequency: str = "weekly"  # "daily" or "weekly"
    email_notifications: bool = True
    content_types: List[ContentType] = [ContentType.ARTICLES]

class User(BaseModel):
    id: str
    email: EmailStr
    name: str
    profile_image: Optional[str] = None
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    preferences: UserPreferences = UserPreferences()
    created_at: datetime
    last_login_at: Optional[datetime] = None
    is_active: bool = True

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str

class PreferencesUpdate(BaseModel):
    topics: Optional[List[AITopic]] = None
    newsletter_frequency: Optional[str] = None
    email_notifications: Optional[bool] = None
    content_types: Optional[List[ContentType]] = None

class AuthResponse(BaseModel):
    user: User
    token: str
    message: str = "Authentication successful"

class PasswordHash(BaseModel):
    password_hash: str
    salt: str

class UserSession(BaseModel):
    user_id: str
    email: str
    name: str
    subscription_tier: SubscriptionTier
    created_at: datetime
    expires_at: datetime