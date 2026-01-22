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
from enum import IntEnum
import json


from app.dependencies.auth import get_current_user, get_current_user_optional
from app.dependencies.database import get_db
from app.models.interactions import (
    ArticleInteractionCreate, ArticleInteractionResponse,
    ArticleStatsResponse, ReadingProgressUpdate, ReadingHistoryResponse,
    BookmarkArticle, BookmarksListResponse,
    SwipeableArticle, SwipeableFeedResponse,
    UserActionCreate, UserActionResponse
)
from db_service import PostgreSQLService

# Action Type ID Enum (matches user_action_types table)
class ActionTypeId(IntEnum):
    LIKE = 1
    COMMENT = 2
    BOOKMARK = 3
    VIEW = 4
    SHARE = 5
    FOLLOW = 6

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/interactions", tags=["interactions"])

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def award_points(user_id: int, action_type_id: int, db: PostgreSQLService) -> int:
    """Award points for user actions and update user_points table"""
    # Fetch points from config table
    points_query = """
        SELECT points 
        FROM action_points_config 
        WHERE action_type_id = %s AND is_active = TRUE
    """
    result = db.execute_query(points_query, (action_type_id,))
    
    if not result or len(result) == 0:
        return 0 # No points configured for this action
    
    points = result[0]['points']
    
    # Insert into user_actions
    action_query = """
        INSERT INTO user_actions (user_id, action_type_id, points_earned, created_at)
        VALUES (%s, %s, %s, NOW())
        RETURNING id
    """
    db.execute_query(action_query, (user_id, action_type_id, points))
    
    # Update user_points
    update_points_query = """
        INSERT INTO user_points (user_id, total_points, current_level_points, lifetime_points, level, next_level_threshold)
        VALUES (%s, %s, %s, %s, 1, 100)
        ON CONFLICT (user_id) DO UPDATE SET
            total_points = user_points.total_points + %s,
            current_level_points = user_points.current_level_points + %s,
            lifetime_points = user_points.lifetime_points + %s,
            updated_at = NOW()
        RETURNING user_id
    """
    db.execute_query(
        update_points_query, 
        (user_id, points, points, points, points, points, points)
    )
        
    # Check for level up
    level_check_query = """
        SELECT current_level_points, next_level_threshold, level 
        FROM user_points WHERE user_id = %s
    """
    level_result = db.execute_query(level_check_query, (user_id,))
    
    if level_result and len(level_result) > 0:
        row = level_result[0]
        current_level_points = row['current_level_points']
        next_level_threshold = row['next_level_threshold']
        level = row['level']
        
        if current_level_points >= next_level_threshold:
            new_level = level + 1
            new_threshold = int(next_level_threshold * 1.5)
            level_up_query = """
                UPDATE user_points 
                SET level = %s,
                    current_level_points = 0,
                    next_level_threshold = %s,
                    updated_at = NOW()
                WHERE user_id = %s
                RETURNING level
            """
            db.execute_query(level_up_query, (new_level, new_threshold, user_id))
    
    return points

def update_reading_streak(user_id: int, db: PostgreSQLService):
    """Update user reading streak"""
    from datetime import date
    today = date.today()
    
    # Get current streak data
    query = """
        SELECT current_streak, longest_streak, last_active_date, total_days_active 
        FROM user_streaks WHERE user_id = %s
    """
    result = db.execute_query(query, (user_id,))
    
    if result and len(result) > 0:
        row = result[0]
        current_streak = row['current_streak']
        longest_streak = row['longest_streak']
        last_active_date = row['last_active_date']
        total_days = row['total_days_active']
        
        if last_active_date:
            days_diff = (today - last_active_date).days
            
            if days_diff == 0:
                logger.info(f"User {user_id} already active today. No streak update.")
            elif days_diff == 1:
                current_streak += 1
            else:
                current_streak = 1
        else:
            current_streak = 1
        
        longest_streak = max(longest_streak, current_streak)
        total_days += 1
        
        update_query = """
            UPDATE user_streaks 
            SET current_streak = %s,
                longest_streak = %s,
                last_active_date = %s,
                total_days_active = %s,
                updated_at = NOW()
            WHERE user_id = %s
        """
        db.execute_query(
            update_query, 
            (current_streak, longest_streak, today, total_days, user_id)
        )
    else:
        insert_query = """
            INSERT INTO user_streaks 
            (user_id, current_streak, longest_streak, last_active_date, total_days_active)
            VALUES (%s, 1, 1, %s, 1)
        """
        db.execute_query(insert_query, (user_id, today))

# ==========================================
# ARTICLE INTERACTION ENDPOINTS
# ==========================================

@router.post("/article", response_model=dict)
def create_article_interaction(
    interaction: dict,  # { article_id, action_type_id, metadata? }
    current_user: Optional[dict] = Depends(get_current_user_optional),
    db: PostgreSQLService = Depends(get_db)
):
    """Create article interaction using action_type_id"""
    user_id = current_user.id if current_user else None
    article_id = interaction.get('article_id')
    action_type_id = interaction.get('action_type_id')  # Changed from interaction_type
    metadata = interaction.get('metadata', {})
    
    logger.info(f"📥 CREATE INTERACTION - Request from frontend: user_id={user_id}, article_id={article_id}, action_type_id={action_type_id}, metadata={metadata}")

    # Validate action_type_id
    if action_type_id not in [1, 2, 3, 4, 5, 6]:
        logger.error(f"❌ Invalid action_type_id: {action_type_id}")
        raise HTTPException(status_code=400, detail="Invalid action_type_id")
    
    logger.info(f"🔄 Inserting interaction: user_id={user_id}, article_id={article_id}, action_type_id={action_type_id}")
    
    # Insert interaction
    query = """
        INSERT INTO article_interactions (user_id, article_id, action_type_id, metadata, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (user_id, article_id, action_type_id) DO NOTHING
        RETURNING id
    """
    result = db.execute_query(
        query, 
        (user_id, article_id, action_type_id, json.dumps(metadata))
    )
    logger.info(f"✅ Interaction inserted successfully: result={result}")
    
    # Note: article_stats are now automatically updated by the database trigger
    # Award points if user is authenticated
    if result and len(result) > 0 and user_id:
        award_points(user_id, action_type_id, db)
    
    return {"success": True, "message": "Interaction created"}

@router.delete("/article", response_model=dict)
def remove_article_interaction(
    article_id: int,
    action_type_id: int,  # Changed from interaction_type
    current_user: dict = Depends(get_current_user),
    db: PostgreSQLService = Depends(get_db)
):
    """Remove interaction using action_type_id"""
    user_id = current_user.id
    logger.info(f"🗑️ DELETE INTERACTION - Request from frontend: user_id={user_id}, article_id={article_id}, action_type_id={action_type_id}")
    
    query = """
        DELETE FROM article_interactions 
        WHERE user_id = %s AND article_id = %s AND action_type_id = %s
        RETURNING id
    """
    result = db.execute_query(query, (user_id, article_id, action_type_id))
    logger.info(f"✅ Interaction deleted: result={result}")
    
    # Note: article_stats are automatically updated by the database trigger on DELETE
    return {"success": True, "message": "Interaction removed"}

@router.get("/bookmarks", response_model=BookmarksListResponse)
def get_bookmarks(
    content_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db: PostgreSQLService = Depends(get_db)
):
    """Get user's bookmarked articles (action_type_id = 3)"""
    user_id = current_user.id
    logger.info(f"📚 GET BOOKMARKS - Request from frontend: user_id={user_id}, content_type={content_type}, limit={limit}, offset={offset}")
    
    content_type_filter = ""
    params = [user_id, ActionTypeId.BOOKMARK]  # Use ID 3 for bookmark
    
    if content_type:
        content_type_filter = "AND ct.name = %s"
        params.append(content_type)
    
    query = f"""
        SELECT 
            a.id, a.title, a.url, a.summary, a.published_date,
            a.thumbnail_url, a.source,
            ct.name as content_type, ct.display_name as content_type_label,
            cat.name as category_name,
            ast.likes_count, ast.comments_count, ast.bookmarks_count, 
            ast.shares_count  -- ✅ Correct column name
        FROM article_interactions ai
        JOIN articles a ON ai.article_id = a.id
        LEFT JOIN content_types ct ON a.content_type_id = ct.id
        LEFT JOIN ai_categories_master cat ON a.category_id = cat.id
        LEFT JOIN article_stats ast ON a.id = ast.article_id
        WHERE ai.user_id = %s 
        AND ai.action_type_id = %s  -- Changed from interaction_type = 'bookmark'
        {content_type_filter}
        ORDER BY ai.created_at DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])
    
    results = db.execute_query(query, tuple(params))
    logger.info(f"✅ Bookmarks fetched: count={len(results)}")
    
    return {
        "articles": results,
        "total": len(results),
        "content_type_filter": content_type
    }
# ==========================================
# READING PROGRESS ENDPOINTS
# ==========================================
@router.post("/reading-progress", response_model=dict)
def update_reading_progress(
    progress: ReadingProgressUpdate,
    current_user: dict = Depends(get_current_user),
    db: PostgreSQLService = Depends(get_db)
):
    """Update reading progress for an article"""
    try:
        user_id = current_user.id
        logger.info(f"📖 UPDATE READING PROGRESS - Request from frontend: user_id={user_id}, article_id={progress.article_id}, read_percentage={progress.read_percentage}, time_spent={progress.time_spent_seconds}, completed={progress.completed}")
        
        # Insert/update reading progress
        db.execute_query(
            """
            INSERT INTO user_reading_history 
            (user_id, article_id, read_percentage, time_spent_seconds, completed, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id, article_id) DO UPDATE SET
                read_percentage = EXCLUDED.read_percentage,
                time_spent_seconds = user_reading_history.time_spent_seconds + EXCLUDED.time_spent_seconds,
                completed = EXCLUDED.completed,
                updated_at = NOW()
            """,
            (user_id, progress.article_id, progress.read_percentage, 
             progress.time_spent_seconds, progress.completed)
        )
        
        # If completed, award points and update streak
        if progress.completed:
            # Award points for reading (use action_type_id = 4 for VIEW, or add 7 to database)
            award_points(user_id, ActionTypeId.VIEW, db)
            update_reading_streak(user_id, db)
        
        # ❌ REMOVED: conn.commit(), cursor.close(), conn.close()
        # These don't exist when using db.execute_query()
        
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
def get_swipeable_feed(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    feed_type: str = Query('personalized', regex="^(personalized|trending|following)$"),
    category: Optional[str] = None,
    content_type: Optional[str] = None,
    exclude_viewed: bool = Query(True),
    current_user: dict = Depends(get_current_user),
    db: PostgreSQLService = Depends(get_db)
):
    """
    Get personalized swipeable feed
    Uses: user_recommendations, user_feed_state, article_interactions
    """
    try:
     
        user_id = current_user.id
        offset = (page - 1) * limit
        
        # Get or create session
        db.execute_query(
            """
            SELECT session_id FROM user_feed_state 
            WHERE user_id = %s AND feed_type = %s
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id, feed_type)
        )
        row = db.fetchone()
        session_id = str(row[0]) if row else str(uuid.uuid4())
        
        # Build exclusion list
        exclude_ids = []
        if exclude_viewed:
            db.execute_query(
                """
                SELECT DISTINCT article_id FROM article_interactions 
                WHERE user_id = %s AND action_type_id IN (4, 10)  -- 4=view, assuming 10=read
                """,
                (user_id,)
            )
            rows = db.fetchall()
            exclude_ids = [row[0] for row in rows]
        
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
                EXISTS(SELECT 1 FROM article_interactions WHERE user_id = %s AND article_id = a.id AND action_type_id = 3) as is_bookmarked,
                EXISTS(SELECT 1 FROM article_interactions WHERE user_id = %s AND article_id = a.id AND action_type_id = 1) as is_liked
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
        db.execute_query(query, params_query)
        rows = db.fetchall()
        
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
            db.execute_query(
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
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM articles a WHERE {where_sql}"
        db.execute_query(count_query, params)
        total_count = (db.fetchone())[0]
        
        
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

# ✅ UPDATED: Share Tracking Endpoint
@router.post("/share/track")
def track_share_platform(
    share_data: dict,  # { article_id, platform }
    current_user: dict = Depends(get_current_user),
    db: PostgreSQLService = Depends(get_db)
):
    """Track share with platform and update stats"""
    user_id = current_user.id
    article_id = share_data['article_id']
    platform = share_data['platform']
    
    # Insert into share_tracking
    query = """
        INSERT INTO share_tracking (user_id, article_id, platform, created_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (user_id, article_id, platform) DO NOTHING
        RETURNING id
    """
    result = db.execute_query(query, (user_id, article_id, platform))
    
    # Update article_stats.shares_count (correct column name)
    update_stats_query = """
        UPDATE article_stats 
        SET shares_count = (
            SELECT COUNT(DISTINCT user_id) 
            FROM share_tracking 
            WHERE article_id = %s
        ),
        last_updated = NOW()
        WHERE article_id = %s
    """
    db.execute_query(update_stats_query, (article_id, article_id))
    
    # Award points (action_type_id = 5 for share)
    if result and len(result) > 0:
        award_points(user_id, ActionTypeId.SHARE, db)
    
    return {"success": True, "message": "Share tracked"}


# ✅ UPDATED: User Stats Endpoint
@router.get("/user/stats")
def get_user_stats(
    current_user: dict = Depends(get_current_user),
    db: PostgreSQLService = Depends(get_db)
):
    """Get user stats with points breakdown by action type"""
    user_id = current_user.id
    
    # Get points and level
    points_query = """
        SELECT total_points, level, current_level_points, next_level_threshold
        FROM user_points
        WHERE user_id = %s
    """
    points_data = db.execute_query(points_query, (user_id,))
    
    # Get streak
    streak_query = """
        SELECT current_streak, longest_streak, total_days_active, last_active_date
        FROM user_streaks
        WHERE user_id = %s
    """
    streak_data = db.execute_query(streak_query, (user_id,))
    
    # Get actions breakdown with action type names
    actions_query = """
        SELECT 
            uat.name as action_type,
            COUNT(*) as count,
            SUM(ua.points_earned) as total_points
        FROM user_actions ua
        JOIN user_action_types uat ON ua.action_type_id = uat.id
        WHERE ua.user_id = %s
        GROUP BY uat.name, uat.id
        ORDER BY uat.id
    """
    actions_data = db.execute_query(actions_query, (user_id,))
    
    # Get points config
    config_query = """
        SELECT uat.name as action_type, apc.points
        FROM action_points_config apc
        JOIN user_action_types uat ON apc.action_type_id = uat.id
        WHERE apc.is_active = TRUE
    """
    config_data = db.execute_query(config_query)
    points_config = {row['action_type']: row['points'] for row in config_data}
    
    return {
        "points": points_data[0] if points_data else {
            "total_points": 0, "level": 1, "current_level_points": 0, "next_level_threshold": 100
        },
        "streak": streak_data[0] if streak_data else {
            "current_streak": 0, "longest_streak": 0, "total_days_active": 0
        },
        "actions_breakdown": actions_data,
        "points_config": points_config
    }


@router.get("/article-stats/{article_id}", response_model=ArticleStatsResponse)
def get_article_stats(
    article_id: int,
    current_user: dict = Depends(get_current_user),
    db: PostgreSQLService = Depends(get_db)
):
    """Get statistics for an article"""
    try:
        
        db.execute_query(
            """
            SELECT article_id, views_count, likes_count, bookmarks_count, 
                   shares_count, comments_count, engagement_score, last_updated
            FROM article_stats WHERE article_id = %s
            """,
            (article_id,)
        )
        row = db.fetchone()
        
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