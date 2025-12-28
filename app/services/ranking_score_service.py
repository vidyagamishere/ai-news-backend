from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
import logging

logger = logging.getLogger(__name__)


class RankingScoreService:
    """Service to calculate and update article ranking scores"""
    
    # Tunable weights
    PUBLISHER_WEIGHT = 0.40
    SIGNIFICANCE_WEIGHT = 0.35
    RECENCY_WEIGHT = 0.25
    
    @staticmethod
    def get_dynamic_ranking_query_fragment() -> str:
        """
        Returns SQL fragment to calculate ranking score dynamically with current recency.
        This should be used in SELECT queries to get real-time ranking.
        """
        return f"""
        (
            -- Static component (publisher + significance) - pre-computed
            COALESCE(a.static_score_component, 0.375) +
            
            -- Dynamic recency component (calculated in real-time based on NOW())
            (
                CASE 
                    WHEN a.published_date >= NOW() - INTERVAL '1 day' THEN 1.0
                    WHEN a.published_date >= NOW() - INTERVAL '3 days' THEN 0.8
                    WHEN a.published_date >= NOW() - INTERVAL '7 days' THEN 0.6
                    WHEN a.published_date >= NOW() - INTERVAL '14 days' THEN 0.4
                    WHEN a.published_date >= NOW() - INTERVAL '30 days' THEN 0.2
                    ELSE 0.1
                END
            ) * {RankingScoreService.RECENCY_WEIGHT}
        ) as ranking_score
        """
    
    @staticmethod
    async def update_static_score_components_bulk(
        db: AsyncSession, 
        article_ids: list[int] = None, 
        days_back: int = 30
    ):
        """
        Update only the static components (publisher + significance) that don't change over time.
        Recency will be calculated dynamically in queries.
        
        Args:
            db: Database session
            article_ids: Optional list of specific article IDs to update
            days_back: Number of days to look back (default 30)
        """
        try:
            # Build the WHERE clause
            where_clause = f"WHERE a.published_date >= NOW() - INTERVAL '{days_back} days'"
            if article_ids:
                article_ids_str = ','.join(map(str, article_ids))
                where_clause += f" AND a.id IN ({article_ids_str})"
            
            # Update only static components (publisher + significance)
            update_query = text(f"""
                UPDATE articles a
                SET 
                    static_score_component = (
                        -- Publisher priority weight (40%)
                        ((11 - COALESCE(p.priority, 10)) / 10.0) * {RankingScoreService.PUBLISHER_WEIGHT} +
                        
                        -- Significance score weight (35%)
                        (COALESCE(a.significance_score, 5.0) / 10.0) * {RankingScoreService.SIGNIFICANCE_WEIGHT}
                    ),
                    last_score_update = NOW()
                FROM publishers_master p
                WHERE a.publisher_id = p.id
                {where_clause}
            """)
            
            result = await db.execute(update_query)
            await db.commit()
            
            rows_updated = result.rowcount
            logger.info(f"Updated static score components for {rows_updated} articles")
            return rows_updated
            
        except Exception as e:
            logger.error(f"Error updating static score components: {str(e)}")
            await db.rollback()
            raise
    
    @staticmethod
    async def update_single_article_score(db: AsyncSession, article_id: int):
        """Update static score component for a single article (for new articles)"""
        try:
            query = text("""
                UPDATE articles a
                SET 
                    static_score_component = (
                        ((11 - COALESCE(p.priority, 10)) / 10.0) * :pub_weight +
                        (COALESCE(a.significance_score, 5.0) / 10.0) * :sig_weight
                    ),
                    last_score_update = NOW()
                FROM publishers_master p
                WHERE a.publisher_id = p.id AND a.id = :article_id
            """)
            
            await db.execute(query, {
                'article_id': article_id,
                'pub_weight': RankingScoreService.PUBLISHER_WEIGHT,
                'sig_weight': RankingScoreService.SIGNIFICANCE_WEIGHT
            })
            await db.commit()
            logger.info(f"Updated static score component for article {article_id}")
            
        except Exception as e:
            logger.error(f"Error updating single article score: {str(e)}")
            await db.rollback()
            raise
    
    @staticmethod
    def get_example_query() -> str:
        """Returns an example query showing how to use dynamic ranking"""
        return f"""
        SELECT 
            a.*,
            ct.name as content_type_label,
            c.name as category_label,
            p.publisher_name,
            p.priority as publisher_priority,
            {RankingScoreService.get_dynamic_ranking_query_fragment()}
        FROM articles a
        LEFT JOIN content_types ct ON a.content_type_id = ct.id
        LEFT JOIN ai_categories_master c ON a.category_id = c.id
        LEFT JOIN publishers_master p ON a.publisher_id = p.id
        WHERE a.published_date >= NOW() - INTERVAL '7 days'
        ORDER BY ranking_score DESC, a.published_date DESC
        LIMIT 50;
        """
