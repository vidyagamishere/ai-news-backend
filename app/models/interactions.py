"""
Pydantic models for user interactions, gamification, and social features
Maps to existing database schema with tables:
- article_interactions, user_actions, user_reading_history
- user_streaks, user_points, achievements, user_achievements
- comments, user_follows, expert_users, user_notifications
- article_collections, collection_articles
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum

# ==========================================
# ENUMS
# ==========================================

class InteractionType(str, Enum):
    """Article interaction types"""
    LIKE = "like"
    BOOKMARK = "bookmark"
    SHARE = "share"
    VIEW = "view"

class ActionType(str, Enum):
    """User action types for points/tracking"""
    LIKE = "like"
    BOOKMARK = "bookmark"
    SHARE = "share"
    READ = "read"
    COMMENT = "comment"
    FOLLOW = "follow"

class NotificationType(str, Enum):
    """Notification types"""
    COMMENT = "comment"
    LIKE = "like"
    FOLLOW = "follow"
    ACHIEVEMENT = "achievement"
    MENTION = "mention"

class AchievementCategory(str, Enum):
    """Achievement categories"""
    READING = "reading"
    ENGAGEMENT = "engagement"
    LEARNING = "learning"
    SOCIAL = "social"
    STREAK = "streak"

class LeaderboardPeriod(str, Enum):
    """Leaderboard time periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ALL_TIME = "all_time"

# ==========================================
# ARTICLE INTERACTION MODELS
# ==========================================

class ArticleInteractionCreate(BaseModel):
    """Create article interaction (like, bookmark, share, view)"""
    article_id: int
    interaction_type: InteractionType
    metadata: Optional[Dict[str, Any]] = None

class ArticleInteractionResponse(BaseModel):
    """Article interaction response"""
    id: int
    user_id: int
    article_id: int
    interaction_type: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ArticleStatsResponse(BaseModel):
    """Article statistics"""
    article_id: int
    views_count: int
    likes_count: int
    bookmarks_count: int
    shares_count: int
    comments_count: int
    engagement_score: float
    last_updated: datetime

# ==========================================
# USER ACTION MODELS (for points tracking)
# ==========================================

class UserActionCreate(BaseModel):
    """Create user action for points"""
    action_type: ActionType
    article_id: Optional[int] = None
    target_user_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class UserActionResponse(BaseModel):
    """User action response"""
    id: int
    user_id: int
    action_type: str
    article_id: Optional[int]
    points_earned: int
    created_at: datetime

# ==========================================
# READING HISTORY MODELS
# ==========================================

class ReadingProgressUpdate(BaseModel):
    """Update reading progress"""
    article_id: int
    read_percentage: int = Field(..., ge=0, le=100)
    time_spent_seconds: int = Field(..., ge=0)
    completed: bool = False

class ReadingHistoryResponse(BaseModel):
    """Reading history item"""
    id: int
    article_id: int
    read_percentage: int
    time_spent_seconds: int
    completed: bool
    created_at: datetime
    updated_at: datetime

# ==========================================
# BOOKMARK MODELS
# ==========================================

class BookmarkArticle(BaseModel):
    """Article in bookmark list"""
    id: int
    title: str
    summary: Optional[str]
    url: str
    source_name: str
    published_date: Optional[str]
    content_type_name: str
    thumbnail_url: Optional[str]
    category_name: Optional[str]
    significance: Optional[int]
    bookmarked_at: datetime
    engagement_score: Optional[float] = 0

class BookmarksListResponse(BaseModel):
    """List of bookmarked articles"""
    articles: List[BookmarkArticle]
    count: int

# ==========================================
# READING STREAK MODELS
# ==========================================

class ReadingStreakResponse(BaseModel):
    """User reading streak stats"""
    current_streak: int
    longest_streak: int
    total_days_active: int
    last_active_date: Optional[str]
    streak_freeze_count: int

# ==========================================
# GAMIFICATION MODELS
# ==========================================

class AchievementResponse(BaseModel):
    """Achievement details"""
    id: int
    name: str
    description: Optional[str]
    icon: Optional[str]
    category: str
    tier: str
    threshold: Optional[int]
    points: int
    badge_color: Optional[str]
    unlocked: bool = False
    unlocked_at: Optional[datetime] = None
    progress: int = 0

class UserPointsResponse(BaseModel):
    """User points/XP details"""
    total_points: int
    level: int
    current_level_points: int
    next_level_threshold: int
    lifetime_points: int
    rank: Optional[int]

class LeaderboardEntry(BaseModel):
    """Leaderboard entry"""
    rank: int
    user_id: int
    username: str
    avatar_url: Optional[str]
    total_points: int
    level: int
    achievements_count: int

class LeaderboardResponse(BaseModel):
    """Leaderboard response"""
    period: str
    entries: List[LeaderboardEntry]
    user_rank: Optional[int]
    total_users: int

class DailyChallengeResponse(BaseModel):
    """Daily challenge details"""
    id: int
    title: str
    description: Optional[str]
    challenge_type: str
    target_value: int
    points_reward: int
    progress: int = 0
    completed: bool = False
    expires_at: Optional[datetime]

class GamificationResponse(BaseModel):
    """Complete gamification stats"""
    points: UserPointsResponse
    achievements: List[AchievementResponse]
    current_streak: int
    challenges: List[DailyChallengeResponse]
    leaderboard_rank: Optional[int]

# ==========================================
# SWIPEABLE FEED MODELS
# ==========================================

class SwipeableArticle(BaseModel):
    """Article for swipeable feed"""
    id: int
    title: str
    summary: Optional[str]
    url: str
    source_name: str
    published_date: Optional[str]
    content_type_name: str
    thumbnail_url: Optional[str]
    category_name: Optional[str]
    significance: Optional[int]
    read_time: Optional[str]
    engagement_score: Optional[float] = 0
    is_bookmarked: bool = False
    is_liked: bool = False
    recommendation_score: Optional[float] = None

class SwipeableFeedResponse(BaseModel):
    """Swipeable feed response"""
    articles: List[SwipeableArticle]
    has_more: bool
    next_page: int
    total_count: int
    session_id: str

# ==========================================
# COMMENT MODELS
# ==========================================

class CommentCreate(BaseModel):
    """Create comment"""
    article_id: int
    content: str = Field(..., min_length=1, max_length=5000)
    parent_comment_id: Optional[int] = None

class CommentUpdate(BaseModel):
    """Update comment"""
    content: str = Field(..., min_length=1, max_length=5000)

class CommentResponse(BaseModel):
    """Comment details"""
    id: int
    user_id: int
    username: str
    user_avatar: Optional[str]
    article_id: int
    parent_comment_id: Optional[int]
    content: str
    likes_count: int
    replies_count: int
    is_edited: bool
    is_deleted: bool
    user_liked: bool = False
    created_at: datetime
    updated_at: datetime

class CommentsListResponse(BaseModel):
    """List of comments"""
    comments: List[CommentResponse]
    total_count: int
    page: int
    has_more: bool

# ==========================================
# SOCIAL MODELS
# ==========================================

class UserFollowCreate(BaseModel):
    """Follow a user"""
    following_id: int

class UserProfileResponse(BaseModel):
    """User profile"""
    id: int
    username: str
    email: str
    avatar_url: Optional[str]
    bio: Optional[str]
    total_points: int
    level: int
    current_streak: int
    achievements_count: int
    followers_count: int = 0
    following_count: int = 0
    is_following: bool = False
    is_expert: bool = False

class ExpertUserResponse(BaseModel):
    """Expert/influencer user"""
    id: int
    user_id: int
    username: str
    avatar_url: Optional[str]
    expertise_areas: List[str]
    verification_status: str
    bio: Optional[str]
    credentials: Optional[str]
    social_links: Optional[Dict[str, str]]
    followers_count: int
    articles_contributed: int
    total_engagement: int

# ==========================================
# COLLECTION MODELS
# ==========================================

class CollectionCreate(BaseModel):
    """Create article collection"""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str]
    is_public: bool = False
    cover_image_url: Optional[str]

class CollectionUpdate(BaseModel):
    """Update collection"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str]
    is_public: Optional[bool]
    cover_image_url: Optional[str]

class CollectionArticleAdd(BaseModel):
    """Add article to collection"""
    article_id: int
    note: Optional[str]

class CollectionResponse(BaseModel):
    """Collection details"""
    id: int
    user_id: int
    name: str
    description: Optional[str]
    is_public: bool
    cover_image_url: Optional[str]
    articles_count: int
    followers_count: int
    is_following: bool = False
    created_at: datetime
    updated_at: datetime

class CollectionArticleResponse(BaseModel):
    """Article in collection"""
    collection_id: int
    article_id: int
    title: str
    summary: Optional[str]
    url: str
    thumbnail_url: Optional[str]
    note: Optional[str]
    position: int
    added_at: datetime

# ==========================================
# NOTIFICATION MODELS
# ==========================================

class NotificationResponse(BaseModel):
    """User notification"""
    id: int
    notification_type: str
    title: str
    message: Optional[str]
    related_user_id: Optional[int]
    related_user_name: Optional[str]
    related_article_id: Optional[int]
    related_article_title: Optional[str]
    read: bool
    action_url: Optional[str]
    created_at: datetime

class NotificationsListResponse(BaseModel):
    """List of notifications"""
    notifications: List[NotificationResponse]
    unread_count: int
    total_count: int

# ==========================================
# ANALYTICS MODELS
# ==========================================

class UserEngagementMetrics(BaseModel):
    """User engagement metrics for a day"""
    date: date
    articles_read: int
    time_spent_minutes: int
    likes_given: int
    comments_posted: int
    shares_made: int
    achievements_unlocked: int
    points_earned: int

class UserEngagementSummary(BaseModel):
    """User engagement summary"""
    total_articles_read: int
    total_time_spent_hours: int
    total_interactions: int
    favorite_category: Optional[str]
    reading_pace: str  # "slow", "moderate", "fast"
    weekly_metrics: List[UserEngagementMetrics]