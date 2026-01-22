"""
API endpoints for social features
Uses tables: comments, comment_likes, user_follows, expert_users,
user_notifications, article_collections, collection_articles
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime
import logging

from app.dependencies.auth import get_current_user
from app.models.interactions import (
    CommentCreate, CommentUpdate, CommentResponse, CommentsListResponse,
    UserFollowCreate, UserProfileResponse, ExpertUserResponse,
    NotificationResponse, NotificationsListResponse,
    CollectionCreate, CollectionUpdate, CollectionResponse,
    CollectionArticleAdd, CollectionArticleResponse
)

from app.routers.interactions import award_points, ActionTypeId
from app.dependencies.database import get_database_service


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/social", tags=["social"])

# ==========================================
# COMMENTS ENDPOINTS
# ==========================================

@router.post("/comments", response_model=CommentResponse)
def create_comment(
    comment: CommentCreate,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Create a comment on an article"""
    try:
        user_id = current_user.id
        
        # Insert comment
        result = db.execute_query(
            """
            INSERT INTO comments (user_id, article_id, parent_comment_id, content, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, created_at, updated_at
            """,
            (user_id, comment.article_id, comment.parent_comment_id, 
             comment.content, datetime.now(), datetime.now())
        )
        comment_id = result[0]['id']
        
        award_points(user_id, ActionTypeId.COMMENT, db)
        
        # Get user details
        user_row = db.execute_query(
            "SELECT first_name, last_name, profile_image FROM users WHERE id = %s",
            (user_id,)
        )
        
        
        return CommentResponse(
            id=comment_id,
            user_id=user_id,
            username=user_row[0]['first_name'] + ' ' + user_row[0]['last_name'],
            user_avatar=user_row[0]['profile_image'].tobytes().decode('utf-8') if isinstance(user_row[0]['profile_image'], memoryview) else user_row[0].get('profile_image', ''),
            article_id=comment.article_id,
            parent_comment_id=comment.parent_comment_id,
            content=comment.content,
            likes_count=0,
            replies_count=0,
            is_edited=False,
            is_deleted=False,
            user_liked=False,
            created_at=result[0]['created_at'],
            updated_at=result[0]['updated_at']
        )
        
    except Exception as e:
        logger.error(f"Error creating comment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create comment: {str(e)}"
        )

@router.get("/comments/article/{article_id}", response_model=CommentsListResponse)
def get_article_comments(
    article_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Get comments for an article"""
    try:
        user_id = current_user.id
        offset = (page - 1) * limit
        
        # Get ALL comments (including nested replies) for the article
        rows = db.execute_query(
            """
            SELECT 
                c.id, c.user_id, CONCAT(u.first_name, ' ', u.last_name) AS username, u.profile_image,
                c.article_id, c.parent_comment_id, c.content,
                c.likes_count, c.replies_count, c.is_edited, c.is_deleted,
                EXISTS(SELECT 1 FROM comment_likes WHERE comment_id = c.id AND user_id = %s) as user_liked,
                c.created_at, c.updated_at
            FROM comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.article_id = %s AND c.is_deleted = false
            ORDER BY COALESCE(c.parent_comment_id, c.id), c.created_at ASC
            LIMIT %s OFFSET %s
            """,
            (user_id, article_id, limit + 1, offset)
        )
        
        has_more = len(rows) > limit
        comments_data = rows[:limit]
        
        comments = []
        for row in comments_data:
            comments.append(CommentResponse(
                id=row['id'],
                user_id=row['user_id'],
                username=row['username'],
                user_avatar=row['profile_image'].tobytes().decode('utf-8') if isinstance(row['profile_image'], memoryview) else row.get('profile_image', ''),
                article_id=row['article_id'],
                parent_comment_id=row['parent_comment_id'],
                content=row['content'],
                likes_count=row['likes_count'],
                replies_count=row['replies_count'],
                is_edited=row['is_edited'],
                is_deleted=row['is_deleted'],
                user_liked=row['user_liked'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            ))
        
        # Get total count
        total_count_row = db.execute_query(
            "SELECT COUNT(*) FROM comments WHERE article_id = %s AND parent_comment_id IS NULL AND is_deleted = false",
            (article_id,)
        )
        total_count = total_count_row[0]['count']
        
        return CommentsListResponse(
            comments=comments,
            total_count=total_count,
            page=page,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"Error fetching comments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch comments: {str(e)}"
        )

@router.put("/comments/{comment_id}", response_model=CommentResponse)
def update_comment(
    comment_id: int,
    comment_update: CommentUpdate,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Update a comment"""
    try:
        user_id = current_user.id
        
        # Check ownership

        row = db.execute_query(
            "SELECT user_id FROM comments WHERE id = %s",
            (comment_id,)
        )
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Comment not found"
            )
        
        if row[0]['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to edit this comment"
            )
        
        # Update comment
        row = db.execute_query(
            """
            UPDATE comments 
            SET content = %s, is_edited = true, edited_at = %s, updated_at = %s
            WHERE id = %s
            RETURNING user_id, article_id, parent_comment_id, likes_count, replies_count, 
                      is_deleted, created_at, updated_at
            """,
            (comment_update.content, datetime.now(), datetime.now(), comment_id)
        )
        
        # Get user details
        user_row = db.execute_query(
            "SELECT first_name, last_name, profile_image FROM users WHERE id = %s",
            (user_id,)
        )
        
        return CommentResponse(
            id=comment_id,
            user_id=user_id,
            username=user_row[0]['first_name'] + ' ' + user_row[0]['last_name'],
            user_avatar=user_row[0]['profile_image'].tobytes().decode('utf-8') if isinstance(user_row[0]['profile_image'], memoryview) else user_row[0].get('profile_image', ''),
            article_id=row[0]['article_id'],
            parent_comment_id=row[0]['parent_comment_id'],
            content=comment_update.content,
            likes_count=row[0]['likes_count'],
            replies_count=row[0]['replies_count'],
            is_edited=True,
            is_deleted=row[0]['is_deleted'],
            user_liked=False,
            created_at=row[0]['created_at'],
            updated_at=row[0]['updated_at']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating comment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update comment: {str(e)}"
        )

@router.delete("/comments/{comment_id}", response_model=dict)
def delete_comment(
    comment_id: int,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Delete a comment (soft delete)"""
    try:
        user_id = current_user.id
        
        # Check ownership
        row = db.execute_query(
            "SELECT user_id FROM comments WHERE id = %s",
            (comment_id,)
        )
    
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Comment not found"
            )
        
        if row[0]['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this comment"
            )
        
        # Soft delete
        db.execute_query(
            """
            UPDATE comments 
            SET is_deleted = true, deleted_at = %s, updated_at = %s
            WHERE id = %s
            """,
            (datetime.now(), datetime.now(), comment_id)
        )
        
        return {"success": True, "message": "Comment deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete comment: {str(e)}"
        )

@router.post("/comments/{comment_id}/like", response_model=dict)
def like_comment(
    comment_id: int,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Like a comment"""
    try:
        user_id = current_user.id
        
        # Check if already liked
        result = db.execute_query(
            "SELECT id FROM comment_likes WHERE user_id = %s AND comment_id = %s",
            (user_id, comment_id)
        )
        
        if result:
            
            return {"success": True, "message": "Comment already liked"}
        
        # Add like
        db.execute_query(
            "INSERT INTO comment_likes (user_id, comment_id, created_at) VALUES (%s, %s, %s)",
            (user_id, comment_id, datetime.now())
        )
        # Update likes count
        db.execute_query(
            "UPDATE comments SET likes_count = likes_count + 1 WHERE id = %s",
            (comment_id,)
        )
        return {"success": True, "message": "Comment liked successfully"}
        
    except Exception as e:
        logger.error(f"Error liking comment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to like comment: {str(e)}"
        )

@router.delete("/comments/{comment_id}/like", response_model=dict)
def unlike_comment(
    comment_id: int,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Unlike a comment"""
    try:
        user_id = current_user.id
        
        result = db.execute_query(
            "DELETE FROM comment_likes WHERE user_id = %s AND comment_id = %s",
            (user_id, comment_id)
        )
        # Check if delete was successful and update count
         
        if result:
            db.execute_query(
                "UPDATE comments SET likes_count = likes_count - 1 WHERE id = %s",
                (comment_id,)
            )
        
        if not result:
            return {"success": False, "message": "Like not found"}
        
        return {"success": True, "message": "Comment unliked successfully"}
        
    except Exception as e:
        logger.error(f"Error unliking comment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unlike comment: {str(e)}"
        )

# ==========================================
# USER FOLLOW ENDPOINTS
# ==========================================

@router.post("/follow", response_model=dict)
def follow_user(
    follow: UserFollowCreate,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Follow a user"""
    try:
        user_id = current_user.id
        
        if user_id == follow.following_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot follow yourself"
            )
        
        # Check if already following
        result = db.execute_query(
            "SELECT id FROM user_follows WHERE follower_id = %s AND following_id = %s",
            (user_id, follow.following_id)
        )
        
        if result:
            
            return {"success": True, "message": "Already following this user"}
        
        # Add follow
        result = db.execute_query(
            """
            INSERT INTO user_follows (follower_id, following_id, followed_at, notification_enabled)
            VALUES (%s, %s, %s, true)
            """,
            (user_id, follow.following_id, datetime.now())
        )
        
        # Award points        
        award_points(user_id, ActionTypeId.FOLLOW, db)
        
        
        return {"success": True, "message": "User followed successfully", "points_earned": 5}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error following user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to follow user: {str(e)}"
        )

@router.delete("/follow/{following_id}", response_model=dict)
def unfollow_user(
    following_id: int,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Unfollow a user"""
    try:
        user_id = current_user.id

        result = db.execute_query(
            "DELETE FROM user_follows WHERE follower_id = %s AND following_id = %s",
            (user_id, following_id)
        )

        if not result:
            return {"success": False, "message": "Not following this user"}
        
        return {"success": True, "message": "User unfollowed successfully"}
        
    except Exception as e:
        logger.error(f"Error unfollowing user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unfollow user: {str(e)}"
        )

@router.get("/profile/{user_id}", response_model=UserProfileResponse)
def get_user_profile(
    user_id: int,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Get user profile"""
    try:
        current_user_id = current_user.id
        
        # Get user details
        row = db.execute_query(
            """
            SELECT 
                u.id, CONCAT(u.first_name, ' ', u.last_name) AS username, u.email, u.profile_image,
                eu.bio,
                COALESCE(up.total_points, 0) as total_points,
                COALESCE(up.level, 1) as level,
                COALESCE(us.current_streak, 0) as current_streak,
                (SELECT COUNT(*) FROM user_achievements WHERE user_id = u.id) as achievements_count,
                COALESCE(eu.followers_count, 0) as followers_count,
                COALESCE(eu.following_count, 0) as following_count,
                EXISTS(SELECT 1 FROM user_follows WHERE follower_id = %s AND following_id = u.id) as is_following,
                EXISTS(SELECT 1 FROM expert_users WHERE user_id = u.id AND verification_status = 'verified') as is_expert
            FROM users u
            LEFT JOIN expert_users eu ON u.id = eu.user_id
            LEFT JOIN user_points up ON u.id = up.user_id
            LEFT JOIN user_streaks us ON u.id = us.user_id
            WHERE u.id = %s
            """,
            (current_user_id, user_id)
        )
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserProfileResponse(
            id=row[0]['id'],
            username=row[0]['first_name'] + ' ' + user_row[0]['last_name'],
            email=row[0]['email'],
            avatar_url=row[0]['profile_image'],
            bio=row[0]['bio'],
            total_points=row[0]['total_points'],
            level=row[0]['level'],
            current_streak=row[0]['current_streak'],
            achievements_count=row[0]['achievements_count'],
            followers_count=row[0]['followers_count'],
            following_count=row[0]['following_count'],
            is_following=row[0]['is_following'],
            is_expert=row[0]['is_expert']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch user profile: {str(e)}"
        )

# ==========================================
# NOTIFICATIONS ENDPOINTS
# ==========================================

@router.get("/notifications", response_model=NotificationsListResponse)
def get_notifications(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    unread_only: bool = False,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Get user notifications"""
    try:
        user_id = current_user.id
        offset = (page - 1) * limit
        
        where_clause = "AND n.read = false" if unread_only else ""
        
        rows = db.execute_query(
            f"""
            SELECT 
                n.id, n.notification_type, n.title, n.message,
                n.related_user_id, CONCAT(u.first_name, ' ', u.last_name) AS username as related_user_name,
                n.related_article_id, a.title as related_article_title,
                n.read, n.action_url, n.created_at
            FROM user_notifications n
            LEFT JOIN users u ON n.related_user_id = u.id
            LEFT JOIN articles a ON n.related_article_id = a.id
            WHERE n.user_id = %s {where_clause}
            ORDER BY n.created_at DESC
            LIMIT %s OFFSET %s
            """,
            (user_id, limit, offset)
        )
        
        notifications = []
        for row in rows:
            notifications.append(NotificationResponse(
                id=row[0]['id'],
                notification_type=row[0]['notification_type'],
                title=row[0]['title'],
                message=row[0]['message'],
                related_user_id=row[0]['related_user_id'],
                related_user_name=row[0]['related_user_name'],
                related_article_id=row[0]['related_article_id'],
                related_article_title=row[0]['related_article_title'],
                read=row[0]['read'],
                action_url=row[0]['action_url'],
                created_at=row[0]['created_at']
            ))
        
        # Get unread count
        rows = db.execute_query(
            "SELECT COUNT(*) FROM user_notifications WHERE user_id = %s AND read = false",
            (user_id,)
        )
        unread_count = rows[0][0] if rows else 0
        
        # Get total count
        rows = db.execute_query(
            "SELECT COUNT(*) FROM user_notifications WHERE user_id = %s",
            (user_id,)
        )
        total_count = rows[0][0] if rows else 0
        
        
        
        return NotificationsListResponse(
            notifications=notifications,
            unread_count=unread_count,
            total_count=total_count
        )
        
    except Exception as e:
        logger.error(f"Error fetching notifications: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch notifications: {str(e)}"
        )

@router.post("/notifications/{notification_id}/read", response_model=dict)
def mark_notification_read(
    notification_id: int,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database_service)
):
    """Mark notification as read"""
    try:
        user_id = current_user.id
        
        result = db.execute_query(
            """
            UPDATE user_notifications 
            SET read = true, read_at = %s
            WHERE id = %s AND user_id = %s
            """,
            (datetime.now(), notification_id, user_id)
        )
        
        if not result:
            return {"success": False, "message": "Notification not found"}
        
        return {"success": True, "message": "Notification marked as read"}
        
    except Exception as e:
        logger.error(f"Error marking notification as read: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark notification as read: {str(e)}"
        )