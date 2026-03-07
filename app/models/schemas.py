#!/usr/bin/env python3
"""
Pydantic models for request/response validation
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field
from dataclasses import dataclass, asdict
import json


# =============================================================================
# USER MODELS
# =============================================================================

class UserBase(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    profile_image: Optional[str] = None


class UserCreate(UserBase):
    password: Optional[str] = None


class UserSignIn(BaseModel):
    """Schema for user sign-in"""
    email: EmailStr
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123"
            }
        }

# ...existing code...

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
    
    # Name-based fields (for backward compatibility)
    categories_selected: Optional[List[str]] = []  # AI category names
    content_types_selected: Optional[List[str]] = ["BLOGS", "VIDEOS", "PODCASTS"]  # Content type names
    publishers_selected: Optional[List[str]] = []  # Publisher names
    
    # ID-based fields (preferred for new implementations)
    category_ids_selected: Optional[List[int]] = []  # AI category IDs from ai_categories_master
    content_type_ids_selected: Optional[List[int]] = []  # Content type IDs from content_types
    publisher_ids_selected: Optional[List[int]] = []  # Publisher IDs from publishers_master
    
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


class SendOTPRequest(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    auth_mode: str = "signup"  # signin or signup


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
    has_liked: Optional[bool] = False
    has_bookmarked: Optional[bool] = False
    has_viewed: Optional[bool] = False
    # Total counts from article_stats (visible to all users)
    total_likes: Optional[int] = 0
    total_bookmarks: Optional[int] = 0
    total_views: Optional[int] = 0
    total_shares: Optional[int] = 0
    total_comments: Optional[int] = 0
    engagement_score: Optional[float] = 0.0
    is_trending: Optional[bool] = False
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    is_remote: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = {}
    

class ContentFilterRequest(BaseModel):
    """Request model for filtering content"""
    interests: Optional[List[str]] = []  # Category names (for backward compatibility)
    content_types: Optional[List[str]] = []  # Content type names (for backward compatibility)
    publishers: Optional[List[str]] = []  # Publisher names (for backward compatibility)
    
    # New ID-based fields (preferred)
    category_ids: Optional[List[int]] = []  # Category IDs from ai_categories_master
    content_type_ids: Optional[List[int]] = []  # Content type IDs from content_types
    publisher_ids: Optional[List[int]] = []  # Publisher IDs from publishers_master
    
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

# ✅ NEW: Content-type specific metadata schemas for articles.metadata JSONB column

@dataclass
class CourseMetadata:
    """
    Metadata schema for courses/learning resources (content_type_id = 5).
    Stored in articles.metadata JSONB column.
    """
    # Core course details
    course_title: str
    platform: str  # "Coursera", "Udemy", "DeepLearning.AI", "YouTube", "edX"
    instructor: Optional[str] = None
    provider: Optional[str] = None  # "Stanford", "Google", "MIT"
    course_url: Optional[str] = None  # Direct enrollment/access URL
    
    # Duration & difficulty
    duration_hours: Optional[float] = None
    duration_weeks: Optional[int] = None
    difficulty: str = "Intermediate"  # "Beginner", "Intermediate", "Advanced"
    
    # Pricing
    price: Optional[float] = None
    currency: str = "USD"
    is_free: bool = False
    has_certificate: bool = False
    course_type: str = "Free"  # "Free", "Paid", "Certification", "Audit"
    
    # Course structure
    modules: Optional[List[str]] = None  # ["Intro to ML", "Neural Networks", ...]
    prerequisites: Optional[List[str]] = None
    learning_outcomes: Optional[List[str]] = None
    topics_covered: Optional[List[str]] = None
    
    # Ratings & popularity
    rating: Optional[float] = None  # 4.8/5.0
    num_reviews: Optional[int] = None
    num_students: Optional[int] = None  # Completed learners
    completion_rate: Optional[float] = None  # Percentage
    
    # Enrollment
    enrollment_url: Optional[str] = None
    start_date: Optional[str] = None
    is_self_paced: bool = True
    enrollment_open: bool = True
    
    # LLM-enriched fields
    ai_topics: Optional[List[str]] = None  # ["Deep Learning", "Computer Vision"]
    recommended_for: Optional[str] = None  # "ML Engineers", "Data Scientists"
    skill_level_required: Optional[str] = None  # "Basic Python", "Advanced Math"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CourseMetadata':
        """Create instance from dictionary"""
        return cls(**data)


@dataclass
class JobMetadata:
    """
    Metadata schema for AI/ML job listings (content_type_id = 6).
    Stored in articles.metadata JSONB column.
    Only for Gen AI, Machine Learning, Deep Learning, Neural Networks,
    Cloud Computing, Quantum Computing, AI Infrastructure roles.
    """
    # Core job details
    job_title: str
    company: str
    job_location: Optional[str] = None  # "San Francisco, CA" or "Remote"
    is_remote: bool = False
    employment_type: str = "Full-time"  # "Full-time", "Part-time", "Contract", "Internship"
    job_type: Optional[str] = None  # "Engineering", "Research", "Data Science", etc.

    # Compensation
    salary_range: Optional[str] = None  # "$120K - $180K" or None if not disclosed
    currency: str = "USD"

    # Requirements
    experience_level: Optional[str] = None  # "Entry", "Mid", "Senior", "Staff", "Principal"
    skills_required: Optional[List[str]] = None  # ["Python", "PyTorch", "LLMs", ...]
    education_required: Optional[str] = None  # "BS/MS in CS", "PhD preferred", etc.

    # Application info
    application_url: Optional[str] = None  # Direct apply link
    application_deadline: Optional[str] = None  # "YYYY-MM-DD" or None if rolling

    # AI-domain categorization
    ai_domain: Optional[str] = None  # "Generative AI", "Machine Learning", "MLOps", "AI Infrastructure"
    company_size: Optional[str] = None  # "Startup", "Series B", "Enterprise"
    posted_date: Optional[str] = None  # "YYYY-MM-DD"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobMetadata':
        """Create instance from dictionary"""
        return cls(**data)


@dataclass
class EventMetadata:
    """
    Metadata schema for AI/ML/Cloud events (content_type_id = 7).
    Stored in articles.metadata JSONB column.
    Only for AI, cloud computing, machine learning related events globally.
    """
    # Core event details
    event_name: str
    event_date: Optional[str] = None      # "YYYY-MM-DD" or date range start
    event_end_date: Optional[str] = None  # "YYYY-MM-DD" or None for single-day
    event_location: Optional[str] = None  # "San Francisco, CA" or "Online"
    is_virtual: bool = False
    event_type: Optional[str] = None  # "Conference", "Workshop", "Meetup", "Hackathon", "Webinar", "Summit"
    event_format: Optional[str] = None  # "In-person", "Virtual", "Hybrid"

    # Hosts & organizers
    event_hosts: Optional[List[str]] = None  # ["Google", "DeepMind", "Hugging Face"]
    speakers: Optional[List[str]] = None  # Notable speaker names

    # Registration
    registration_url: Optional[str] = None  # Direct registration link
    ticket_price: Optional[str] = None  # "Free", "$299", "$299 - $999"
    registration_deadline: Optional[str] = None  # "YYYY-MM-DD" or None

    # Themes
    ai_topics: Optional[List[str]] = None  # ["LLMs", "Computer Vision", "MLOps"]
    target_audience: Optional[str] = None  # "Researchers", "Practitioners", "Developers", "Everyone"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventMetadata':
        """Create instance from dictionary"""
        return cls(**data)


def get_metadata_schema(content_type_id: int) -> Optional[type]:
    """
    Get metadata dataclass for content type.
    Returns appropriate metadata schema based on content type ID.
    
    Args:
        content_type_id: Content type identifier (1=Blogs, 5=Courses, 6=Jobs, 7=Events)
    
    Returns:
        Metadata dataclass or None if no specific schema exists
    """
    schemas = {
        5: CourseMetadata,   # Courses/Learning
        6: JobMetadata,      # AI/ML Jobs
        7: EventMetadata,    # AI/ML Events
    }
    return schemas.get(content_type_id)


def validate_metadata(content_type_id: int, metadata: Dict[str, Any]) -> bool:
    """
    Validate metadata against schema for content type.
    
    Args:
        content_type_id: Content type identifier
        metadata: Metadata dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    schema = get_metadata_schema(content_type_id)
    if not schema:
        return True  # No validation needed for content types without metadata
    
    try:
        # Try to instantiate the dataclass (will fail if required fields missing)
        schema(**metadata)
        return True
    except (TypeError, ValueError) as e:
        logger.warning(f"⚠️ Metadata validation failed: {e}")
        return False


# ✅ NEW: Pydantic schemas for API requests/responses

class CourseFilterRequest(BaseModel):
    """Request schema for filtering courses"""
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    is_free: Optional[bool] = None
    platform: Optional[str] = None
    min_rating: Optional[float] = None
    limit: int = 50
    offset: int = 0


class CourseResponse(BaseModel):
    """Response schema for course data"""
    id: int
    title: str
    instructor: str
    summary: str
    url: str
    platform: Optional[str]
    difficulty: Optional[str]
    duration_hours: Optional[float]
    price: Optional[float]
    is_free: bool
    has_certificate: bool
    course_type: Optional[str]
    rating: Optional[float]
    num_reviews: Optional[int]
    num_students: Optional[int]
    topics_covered: List[str] = []
    prerequisites: List[str] = []
    learning_outcomes: List[str] = []
    enrollment_url: Optional[str]
    ai_topics: List[str] = []
    image_url: Optional[str]
    significance_score: float
    created_date: datetime
    updated_date: datetime


class CoursesListResponse(BaseModel):
    """Response schema for courses list"""
    courses: List[CourseResponse]
    count: int
    filters: Dict[str, Any]
    pagination: Dict[str, Any]


