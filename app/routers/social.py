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
from db_service import get_db_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/social", tags=["social"])

# ==========================================
# COMMENTS ENDPOINTS
# ==========================================

@router.post("/comments", response_model=CommentResponse)
async def create_comment(
    comment: CommentCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a comment on an article"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        # Insert comment
        cursor.execute(
            """
            INSERT INTO comments (user_id, article_id, parent_comment_id, content, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, created_at, updated_at
            """,
            (user_id, comment.article_id, comment.parent_comment_id, 
             comment.content, datetime.now(), datetime.now())
        )
        row = cursor.fetchone()
        comment_id = row[0]
        
        # Award points
        cursor.execute(
            """
            INSERT INTO user_actions (user_id, article_id, action_type, points_earned, created_at)
            VALUES (%s, %s, 'comment', 20, %s)
            """,
            (user_id, comment.article_id, datetime.now())
        )
        
        cursor.execute(
            """
            UPDATE user_points 
            SET total_points = total_points + 20,
                current_level_points = current_level_points + 20,
                lifetime_points = lifetime_points + 20,
                updated_at = %s
            WHERE user_id = %s
            """,
            (datetime.now(), user_id)
        )
        
        conn.commit()
        
        # Get user details
        cursor.execute(
            "SELECT username, avatar_url FROM users WHERE id = %s",
            (user_id,)
        )
        user_row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return CommentResponse(
            id=comment_id,
            user_id=user_id,
            username=user_row[0],
            user_avatar=user_row[1],
            article_id=comment.article_id,
            parent_comment_id=comment.parent_comment_id,
            content=comment.content,
            likes_count=0,
            replies_count=0,
            is_edited=False,
            is_deleted=False,
            user_liked=False,
            created_at=row[1],
            updated_at=row[2]
        )
        
    except Exception as e:
        logger.error(f"Error creating comment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create comment: {str(e)}"
        )

@router.get("/comments/article/{article_id}", response_model=CommentsListResponse)
async def get_article_comments(
    article_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Get comments for an article"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        offset = (page - 1) * limit
        
        # Get comments
        cursor.execute(
            """
            SELECT 
                c.id, c.user_id, u.username, u.avatar_url,
                c.article_id, c.parent_comment_id, c.content,
                c.likes_count, c.replies_count, c.is_edited, c.is_deleted,
                EXISTS(SELECT 1 FROM comment_likes WHERE comment_id = c.id AND user_id = %s) as user_liked,
                c.created_at, c.updated_at
            FROM comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.article_id = %s AND c.parent_comment_id IS NULL AND c.is_deleted = false
            ORDER BY c.created_at DESC
            LIMIT %s OFFSET %s
            """,
            (user_id, article_id, limit + 1, offset)
        )
        rows = cursor.fetchall()
        
        has_more = len(rows) > limit
        comments_data = rows[:limit]
        
        comments = []
        for row in comments_data:
            comments.append(CommentResponse(
                id=row[0],
                user_id=row[1],
                username=row[2],
                user_avatar=row[3],
                article_id=row[4],
                parent_comment_id=row[5],
                content=row[6],
                likes_count=row[7],
                replies_count=row[8],
                is_edited=row[9],
                is_deleted=row[10],
                user_liked=row[11],
                created_at=row[12],
                updated_at=row[13]
            ))
        
        # Get total count
        cursor.execute(
            "SELECT COUNT(*) FROM comments WHERE article_id = %s AND parent_comment_id IS NULL AND is_deleted = false",
            (article_id,)
        )
        total_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
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
async def update_comment(
    comment_id: int,
    comment_update: CommentUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update a comment"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        # Check ownership
        cursor.execute(
            "SELECT user_id FROM comments WHERE id = %s",
            (comment_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Comment not found"
            )
        
        if row[0] != user_id:
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to edit this comment"
            )
        
        # Update comment
        cursor.execute(
            """
            UPDATE comments 
            SET content = %s, is_edited = true, edited_at = %s, updated_at = %s
            WHERE id = %s
            RETURNING user_id, article_id, parent_comment_id, likes_count, replies_count, 
                      is_deleted, created_at, updated_at
            """,
            (comment_update.content, datetime.now(), datetime.now(), comment_id)
        )
        row = cursor.fetchone()
        conn.commit()
        
        # Get user details
        cursor.execute(
            "SELECT username, avatar_url FROM users WHERE id = %s",
            (user_id,)
        )
        user_row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return CommentResponse(
            id=comment_id,
            user_id=user_id,
            username=user_row[0],
            user_avatar=user_row[1],
            article_id=row[1],
            parent_comment_id=row[2],
            content=comment_update.content,
            likes_count=row[3],
            replies_count=row[4],
            is_edited=True,
            is_deleted=row[5],
            user_liked=False,
            created_at=row[6],
            updated_at=row[7]
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
async def delete_comment(
    comment_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete a comment (soft delete)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        # Check ownership
        cursor.execute(
            "SELECT user_id FROM comments WHERE id = %s",
            (comment_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Comment not found"
            )
        
        if row[0] != user_id:
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this comment"
            )
        
        # Soft delete
        cursor.execute(
            """
            UPDATE comments 
            SET is_deleted = true, deleted_at = %s, updated_at = %s
            WHERE id = %s
            """,
            (datetime.now(), datetime.now(), comment_id)
        )
        conn.commit()
        
        cursor.close()
        conn.close()
        
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
async def like_comment(
    comment_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Like a comment"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        # Check if already liked
        cursor.execute(
            "SELECT id FROM comment_likes WHERE user_id = %s AND comment_id = %s",
            (user_id, comment_id)
        )
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return {"success": True, "message": "Comment already liked"}
        
        # Add like
        cursor.execute(
            "INSERT INTO comment_likes (user_id, comment_id, created_at) VALUES (%s, %s, %s)",
            (user_id, comment_id, datetime.now())
        )
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {"success": True, "message": "Comment liked successfully"}
        
    except Exception as e:
        logger.error(f"Error liking comment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to like comment: {str(e)}"
        )

@router.delete("/comments/{comment_id}/like", response_model=dict)
async def unlike_comment(
    comment_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Unlike a comment"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        cursor.execute(
            "DELETE FROM comment_likes WHERE user_id = %s AND comment_id = %s",
            (user_id, comment_id)
        )
        deleted = cursor.rowcount
        conn.commit()
        
        cursor.close()
        conn.close()
        
        if deleted == 0:
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
async def follow_user(
    follow: UserFollowCreate,
    current_user: dict = Depends(get_current_user)
):
    """Follow a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        if user_id == follow.following_id:
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot follow yourself"
            )
        
        # Check if already following
        cursor.execute(
            "SELECT id FROM user_follows WHERE follower_id = %s AND following_id = %s",
            (user_id, follow.following_id)
        )
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return {"success": True, "message": "Already following this user"}
        
        # Add follow
        cursor.execute(
            """
            INSERT INTO user_follows (follower_id, following_id, followed_at, notification_enabled)
            VALUES (%s, %s, %s, true)
            """,
            (user_id, follow.following_id, datetime.now())
        )
        
        # Award points
        cursor.execute(
            """
            INSERT INTO user_actions (user_id, target_user_id, action_type, points_earned, created_at)
            VALUES (%s, %s, 'follow', 5, %s)
            """,
            (user_id, follow.following_id, datetime.now())
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
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
async def unfollow_user(
    following_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Unfollow a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        cursor.execute(
            "DELETE FROM user_follows WHERE follower_id = %s AND following_id = %s",
            (user_id, following_id)
        )
        deleted = cursor.rowcount
        conn.commit()
        
        cursor.close()
        conn.close()
        
        if deleted == 0:
            return {"success": False, "message": "Not following this user"}
        
        return {"success": True, "message": "User unfollowed successfully"}
        
    except Exception as e:
        logger.error(f"Error unfollowing user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unfollow user: {str(e)}"
        )

@router.get("/profile/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get user profile"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        current_user_id = current_user['id']
        
        # Get user details
        cursor.execute(
            """
            SELECT 
                u.id, u.username, u.email, u.avatar_url,
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
        row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserProfileResponse(
            id=row[0],
            username=row[1],
            email=row[2],
            avatar_url=row[3],
            bio=row[4],
            total_points=row[5],
            level=row[6],
            current_streak=row[7],
            achievements_count=row[8],
            followers_count=row[9],
            following_count=row[10],
            is_following=row[11],
            is_expert=row[12]
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
async def get_notifications(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    unread_only: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """Get user notifications"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        offset = (page - 1) * limit
        
        where_clause = "AND n.read = false" if unread_only else ""
        
        cursor.execute(
            f"""
            SELECT 
                n.id, n.notification_type, n.title, n.message,
                n.related_user_id, u.username as related_user_name,
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
        rows = cursor.fetchall()
        
        notifications = []
        for row in rows:
            notifications.append(NotificationResponse(
                id=row[0],
                notification_type=row[1],
                title=row[2],
                message=row[3],
                related_user_id=row[4],
                related_user_name=row[5],
                related_article_id=row[6],
                related_article_title=row[7],
                read=row[8],
                action_url=row[9],
                created_at=row[10]
            ))
        
        # Get unread count
        cursor.execute(
            "SELECT COUNT(*) FROM user_notifications WHERE user_id = %s AND read = false",
            (user_id,)
        )
        unread_count = cursor.fetchone()[0]
        
        # Get total count
        cursor.execute(
            "SELECT COUNT(*) FROM user_notifications WHERE user_id = %s",
            (user_id,)
        )
        total_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
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
async def mark_notification_read(
    notification_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Mark notification as read"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        cursor.execute(
            """
            UPDATE user_notifications 
            SET read = true, read_at = %s
            WHERE id = %s AND user_id = %s
            """,
            (datetime.now(), notification_id, user_id)
        )
        updated = cursor.rowcount
        conn.commit()
        
        cursor.close()
        conn.close()
        
        if updated == 0:
            return {"success": False, "message": "Notification not found"}
        
        return {"success": True, "message": "Notification marked as read"}
        
    except Exception as e:
        logger.error(f"Error marking notification as read: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark notification as read: {str(e)}"
        )