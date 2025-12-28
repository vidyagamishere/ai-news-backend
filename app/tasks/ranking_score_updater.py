import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import async_session_maker
from app.services.ranking_score_service import RankingScoreService
import logging

logger = logging.getLogger(__name__)


async def update_static_score_components_task():
    """
    Background task to update static score components periodically.
    Static components = publisher priority + significance score
    Recency is calculated dynamically in queries, so it doesn't need periodic updates.
    """
    logger.info("Starting static score component update task")
    
    async with async_session_maker() as db:
        try:
            # Update static components for articles from last 30 days
            rows_updated = await RankingScoreService.update_static_score_components_bulk(
                db=db,
                days_back=30
            )
            logger.info(f"Static score component update completed. Updated {rows_updated} articles.")
            
        except Exception as e:
            logger.error(f"Error in static score component update task: {str(e)}")


async def start_ranking_score_updater():
    """
    Start the background task that runs every hour.
    Since recency is calculated dynamically, we only need to update static components
    when publisher priorities or significance scores change.
    """
    while True:
        try:
            await update_static_score_components_task()
        except Exception as e:
            logger.error(f"Error in ranking score updater: {str(e)}")
        
        # Wait 1 hour before next update (static components don't change often)
        # Publisher priorities and significance scores are relatively stable
        await asyncio.sleep(3600)  # 3600 seconds = 1 hour


async def initialize_static_score_components():
    """One-time initialization of static score components for all articles"""
    logger.info("Initializing static score components for all articles")
    
    async with async_session_maker() as db:
        try:
            rows_updated = await RankingScoreService.update_static_score_components_bulk(
                db=db,
                days_back=365  # Update all articles from last year
            )
            logger.info(f"Initialized {rows_updated} article static score components")
            
        except Exception as e:
            logger.error(f"Error initializing static score components: {str(e)}")
