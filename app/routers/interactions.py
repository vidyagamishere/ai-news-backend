"""
API endpoints for article interactions, bookmarks, and feed
Uses tables: article_interactions, user_actions, user_reading_history,
article_stats, user_feed_state, user_recommendations
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime
import logging
import uuid

from app.dependencies.auth import get_current_user, get_current_user_optional
from app.models.interactions import (
    ArticleInteractionCreate, ArticleInteractionResponse,
    ArticleStatsResponse, ReadingProgressUpdate, ReadingHistoryResponse,
    BookmarkArticle, BookmarksListResponse,
    SwipeableArticle, SwipeableFeedResponse,
    UserActionCreate, UserActionResponse
)
from db_service import get_database_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/interactions", tags=["interactions"])

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_db():
    """Get database service"""
    return get_database_service()

def award_points(user_id: int, action_type: str, cursor, conn) -> int:
    """Award points for user actions and update user_points table"""
    points_map = {
        'read': 10,
        'bookmark': 5,
        'like': 3,
        'share': 15,
        'comment': 20,
        'follow': 5,
        'view': 1
    }
    
    points = points_map.get(action_type, 0)
    
    if points > 0:
        # Update user_points table
        cursor.execute(
            """
            INSERT INTO user_points (user_id, total_points, current_level_points, lifetime_points)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                total_points = user_points.total_points + EXCLUDED.total_points,
                current_level_points = user_points.current_level_points + EXCLUDED.current_level_points,
                lifetime_points = user_points.lifetime_points + EXCLUDED.lifetime_points,
                updated_at = %s
            """,
            (user_id, points, points, points, datetime.now())
        )
        
        # Check for level up
        cursor.execute(
            """
            SELECT current_level_points, next_level_threshold, level 
            FROM user_points WHERE user_id = %s
            """,
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row and row[0] >= row[1]:
            new_level = row[2] + 1
            new_threshold = int(row[1] * 1.5)
            cursor.execute(
                """
                UPDATE user_points 
                SET level = %s,
                    current_level_points = 0,
                    next_level_threshold = %s,
                    updated_at = %s
                WHERE user_id = %s
                """,
                (new_level, new_threshold, datetime.now(), user_id)
            )
        
        conn.commit()
    
    return points

def update_reading_streak(user_id: int, cursor, conn):
    """Update user reading streak"""
    today = datetime.now().date()
    
    cursor.execute(
        """
        SELECT current_streak, longest_streak, last_active_date, total_days_active 
        FROM user_streaks WHERE user_id = %s
        """,
        (user_id,)
    )
    row = cursor.fetchone()
    
    if row:
        current_streak, longest_streak, last_active_date, total_days = row
        
        if last_active_date:
            days_diff = (today - last_active_date).days
            
            if days_diff == 0:
                return
            elif days_diff == 1:
                current_streak += 1
            else:
                current_streak = 1
        else:
            current_streak = 1
        
        longest_streak = max(longest_streak, current_streak)
        total_days += 1
        
        cursor.execute(
            """
            UPDATE user_streaks 
            SET current_streak = %s,
                longest_streak = %s,
                last_active_date = %s,
                total_days_active = %s,
                updated_at = %s
            WHERE user_id = %s
            """,
            (current_streak, longest_streak, today, total_days, datetime.now(), user_id)
        )
    else:
        cursor.execute(
            """
            INSERT INTO user_streaks 
            (user_id, current_streak, longest_streak, last_active_date, total_days_active)
            VALUES (%s, 1, 1, %s, 1)
            """,
            (user_id, today)
        )
    
    conn.commit()

# ==========================================
# ARTICLE INTERACTION ENDPOINTS
# ==========================================

@router.post("/article", response_model=dict)
async def create_article_interaction(
    interaction: ArticleInteractionCreate,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Create article interaction (like, bookmark, share, view)
    Inserts into article_interactions table
    Optional authentication - view tracking works for anonymous users
    """
    try:
        db = get_db()
        user_id = current_user['id'] if current_user else None
        
        # For authenticated users, check if interaction already exists
        if user_id:
            existing = db.execute_query(
                """
                SELECT id FROM article_interactions 
                WHERE user_id = %s AND article_id = %s AND interaction_type = %s
                """,
                (user_id, interaction.article_id, interaction.interaction_type.value),
                fetch_one=True
            )
            
            if existing:
                return {"success": True, "message": f"Article already {interaction.interaction_type.value}d"}
        
        # Create interaction (user_id can be NULL for anonymous views)
        db.execute_query(
            """
            INSERT INTO article_interactions 
            (user_id, article_id, interaction_type, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_id, interaction.article_id, interaction.interaction_type.value, 
             None if not interaction.metadata else str(interaction.metadata), datetime.now()),
            fetch_all=False
        )
        
        # Award points only for authenticated users
        points = 0
        if user_id:
            points_map = {
                'read': 10, 'bookmark': 5, 'like': 3, 'share': 15,
                'comment': 20, 'follow': 5, 'view': 1
            }
            points = points_map.get(interaction.interaction_type.value, 0)
            
            if points > 0:
                db.execute_query(
                    """
                    INSERT INTO user_actions 
                    (user_id, article_id, action_type, points_earned, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, interaction.article_id, interaction.interaction_type.value, points, datetime.now()),
                    fetch_all=False
                )
        
        # Update view_count in articles table if interaction is 'view'
        if interaction.interaction_type.value == 'view':
            db.execute_query(
                """
                UPDATE articles 
                SET view_count = COALESCE(view_count, 0) + 1
                WHERE id = %s
                """,
                (interaction.article_id,),
                fetch_all=False
            )
        
        logger.info(f"User {user_id or 'anonymous'} {interaction.interaction_type.value}d article {interaction.article_id}")
        return {
            "success": True, 
            "message": f"Article {interaction.interaction_type.value}d successfully",
            "points_earned": points
        }
        
    except Exception as e:
        logger.error(f"Error creating interaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create interaction: {str(e)}"
        )

@router.delete("/article", response_model=dict)
async def remove_article_interaction(
    article_id: int = Query(...),
    interaction_type: str = Query(..., regex="^(like|bookmark|share)$"),
    current_user: dict = Depends(get_current_user)
):
    """Remove article interaction"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        cursor.execute(
            """
            DELETE FROM article_interactions 
            WHERE user_id = %s AND article_id = %s AND interaction_type = %s
            """,
            (user_id, article_id, interaction_type)
        )
        deleted = cursor.rowcount
        conn.commit()
        
        cursor.close()
        conn.close()
        
        if deleted == 0:
            return {"success": False, "message": "Interaction not found"}
        
        return {"success": True, "message": f"{interaction_type} removed successfully"}
        
    except Exception as e:
        logger.error(f"Error removing interaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove interaction: {str(e)}"
        )

@router.get("/bookmarks", response_model=BookmarksListResponse)
async def get_bookmarks(
    current_user: dict = Depends(get_current_user)
):
    """Get all bookmarked articles"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        cursor.execute(
            """
            SELECT 
                a.id, a.title, a.summary, a.url,
                s.name as source_name,
                a.published_date,
                ct.name as content_type_name,
                a.thumbnail_url,
                c.name as category_name,
                a.significance,
                ai.created_at as bookmarked_at,
                COALESCE(ast.engagement_score, 0) as engagement_score
            FROM article_interactions ai
            JOIN articles a ON ai.article_id = a.id
            LEFT JOIN sources s ON a.source_id = s.id
            LEFT JOIN content_types ct ON a.content_type_id = ct.id
            LEFT JOIN categories c ON a.category_id = c.id
            LEFT JOIN article_stats ast ON a.id = ast.article_id
            WHERE ai.user_id = %s AND ai.interaction_type = 'bookmark'
            ORDER BY ai.created_at DESC
            """,
            (user_id,)
        )
        
        rows = cursor.fetchall()
        articles = []
        
        for row in rows:
            articles.append(BookmarkArticle(
                id=row[0],
                title=row[1],
                summary=row[2],
                url=row[3],
                source_name=row[4] or "Unknown",
                published_date=str(row[5]) if row[5] else None,
                content_type_name=row[6] or "BLOGS",
                thumbnail_url=row[7],
                category_name=row[8] or "AI News",
                significance=row[9],
                bookmarked_at=row[10],
                engagement_score=float(row[11]) if row[11] else 0
            ))
        
        cursor.close()
        conn.close()
        
        return BookmarksListResponse(articles=articles, count=len(articles))
        
    except Exception as e:
        logger.error(f"Error fetching bookmarks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch bookmarks: {str(e)}"
        )

# ==========================================
# READING PROGRESS ENDPOINTS
# ==========================================

@router.post("/reading-progress", response_model=dict)
async def update_reading_progress(
    progress: ReadingProgressUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update reading progress for an article"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        
        cursor.execute(
            """
            INSERT INTO user_reading_history 
            (user_id, article_id, read_percentage, time_spent_seconds, completed, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, article_id) DO UPDATE SET
                read_percentage = EXCLUDED.read_percentage,
                time_spent_seconds = user_reading_history.time_spent_seconds + EXCLUDED.time_spent_seconds,
                completed = EXCLUDED.completed,
                updated_at = EXCLUDED.updated_at
            """,
            (user_id, progress.article_id, progress.read_percentage, 
             progress.time_spent_seconds, progress.completed, datetime.now())
        )
        
        # If completed, track as read action
        if progress.completed:
            cursor.execute(
                """
                INSERT INTO user_actions (user_id, article_id, action_type, points_earned, created_at)
                VALUES (%s, %s, 'read', 10, %s)
                ON CONFLICT DO NOTHING
                """,
                (user_id, progress.article_id, datetime.now())
            )
            award_points(user_id, 'read', cursor, conn)
            update_reading_streak(user_id, cursor, conn)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {"success": True, "message": "Reading progress updated"}
        
    except Exception as e:
        logger.error(f"Error updating reading progress: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update reading progress: {str(e)}"
        )

# ==========================================
# SWIPEABLE FEED ENDPOINT
# ==========================================

@router.get("/swipeable-feed", response_model=SwipeableFeedResponse)
async def get_swipeable_feed(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    feed_type: str = Query('personalized', regex="^(personalized|trending|following)$"),
    category: Optional[str] = None,
    content_type: Optional[str] = None,
    exclude_viewed: bool = Query(True),
    current_user: dict = Depends(get_current_user)
):
    """
    Get personalized swipeable feed
    Uses: user_recommendations, user_feed_state, article_interactions
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = current_user['id']
        offset = (page - 1) * limit
        
        # Get or create session
        cursor.execute(
            """
            SELECT session_id FROM user_feed_state 
            WHERE user_id = %s AND feed_type = %s
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id, feed_type)
        )
        row = cursor.fetchone()
        session_id = str(row[0]) if row else str(uuid.uuid4())
        
        # Build exclusion list
        exclude_ids = []
        if exclude_viewed:
            cursor.execute(
                """
                SELECT DISTINCT article_id FROM article_interactions 
                WHERE user_id = %s AND interaction_type IN ('view', 'read')
                """,
                (user_id,)
            )
            exclude_ids = [row[0] for row in cursor.fetchall()]
        
        # Build query
        where_clauses = []
        params = []
        
        if exclude_ids:
            placeholders = ','.join(['%s'] * len(exclude_ids))
            where_clauses.append(f"a.id NOT IN ({placeholders})")
            params.extend(exclude_ids)
        
        if category:
            where_clauses.append("c.name = %s")
            params.append(category)
        
        if content_type:
            where_clauses.append("ct.name = %s")
            params.append(content_type)
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Get articles
        query = f"""
            SELECT DISTINCT
                a.id, a.title, a.summary, a.url,
                s.name as source_name,
                a.published_date,
                ct.name as content_type_name,
                a.thumbnail_url,
                c.name as category_name,
                a.significance,
                a.read_time,
                COALESCE(ast.engagement_score, 0) as engagement_score,
                COALESCE(ur.recommendation_score, 0) as rec_score,
                EXISTS(SELECT 1 FROM article_interactions WHERE user_id = %s AND article_id = a.id AND interaction_type = 'bookmark') as is_bookmarked,
                EXISTS(SELECT 1 FROM article_interactions WHERE user_id = %s AND article_id = a.id AND interaction_type = 'like') as is_liked
            FROM articles a
            LEFT JOIN sources s ON a.source_id = s.id
            LEFT JOIN content_types ct ON a.content_type_id = ct.id
            LEFT JOIN categories c ON a.category_id = c.id
            LEFT JOIN article_stats ast ON a.id = ast.article_id
            LEFT JOIN user_recommendations ur ON ur.user_id = %s AND ur.article_id = a.id
            WHERE {where_sql}
            ORDER BY rec_score DESC, ast.engagement_score DESC, a.published_date DESC
            LIMIT %s OFFSET %s
        """
        
        params_query = [user_id, user_id, user_id] + params + [limit + 1, offset]
        cursor.execute(query, params_query)
        rows = cursor.fetchall()
        
        has_more = len(rows) > limit
        articles_data = rows[:limit]
        
        articles = []
        for row in articles_data:
            articles.append(SwipeableArticle(
                id=row[0],
                title=row[1],
                summary=row[2],
                url=row[3],
                source_name=row[4] or "Unknown",
                published_date=str(row[5]) if row[5] else None,
                content_type_name=row[6] or "BLOGS",
                thumbnail_url=row[7],
                category_name=row[8] or "AI News",
                significance=row[9],
                read_time=row[10] or "5 min",
                engagement_score=float(row[11]) if row[11] else 0,
                recommendation_score=float(row[12]) if row[12] else None,
                is_bookmarked=row[13],
                is_liked=row[14]
            ))
        
        # Update feed state
        if articles:
            cursor.execute(
                """
                INSERT INTO user_feed_state 
                (user_id, feed_type, last_article_id, offset_position, session_id, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, feed_type) DO UPDATE SET
                    last_article_id = EXCLUDED.last_article_id,
                    offset_position = EXCLUDED.offset_position,
                    session_id = EXCLUDED.session_id,
                    updated_at = EXCLUDED.updated_at
                """,
                (user_id, feed_type, articles[-1].id, offset + len(articles), session_id, datetime.now())
            )
            conn.commit()
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM articles a WHERE {where_sql}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return SwipeableFeedResponse(
            articles=articles,
            has_more=has_more,
            next_page=page + 1 if has_more else page,
            total_count=total_count,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error fetching swipeable feed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch swipeable feed: {str(e)}"
        )

@router.get("/article-stats/{article_id}", response_model=ArticleStatsResponse)
async def get_article_stats(
    article_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get statistics for an article"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT article_id, views_count, likes_count, bookmarks_count, 
                   shares_count, comments_count, engagement_score, last_updated
            FROM article_stats WHERE article_id = %s
            """,
            (article_id,)
        )
        row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not row:
            # Return zeros if no stats yet
            return ArticleStatsResponse(
                article_id=article_id,
                views_count=0,
                likes_count=0,
                bookmarks_count=0,
                shares_count=0,
                comments_count=0,
                engagement_score=0.0,
                last_updated=datetime.now()
            )
        
        return ArticleStatsResponse(
            article_id=row[0],
            views_count=row[1],
            likes_count=row[2],
            bookmarks_count=row[3],
            shares_count=row[4],
            comments_count=row[5],
            engagement_score=float(row[6]),
            last_updated=row[7]
        )
        
    except Exception as e:
        logger.error(f"Error fetching article stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch article stats: {str(e)}"
        )