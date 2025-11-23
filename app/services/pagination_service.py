"""
Pagination service - Handles all pagination logic.
Steve Jobs: "Think Different, Load Smart"
"""

from typing import List, Dict, Any, Optional, Tuple
from math import ceil
import logging

logger = logging.getLogger(__name__)

class PaginationService:
    """
    Centralized pagination logic for all endpoints.
    
    Philosophy:
    - Backend does heavy lifting
    - Frontend just displays
    - User sees content in < 1 second
    """
    
    @staticmethod
    def calculate_offset(page: int, page_size: int) -> int:
        """Calculate SQL OFFSET from page number"""
        return (page - 1) * page_size
    
    @staticmethod
    def create_meta(
        current_page: int,
        page_size: int,
        total_items: int
    ) -> Dict[str, Any]:
        """Create pagination metadata"""
        total_pages = ceil(total_items / page_size) if page_size > 0 else 0
        
        return {
            "current_page": current_page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": current_page < total_pages,
            "has_prev": current_page > 1,
            "next_page": current_page + 1 if current_page < total_pages else None,
            "prev_page": current_page - 1 if current_page > 1 else None
        }
    
    @staticmethod
    async def paginate_query(
        db_service,
        base_query: str,
        count_query: str,
        params: Tuple,
        page: int,
        page_size: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute paginated query and return items + metadata.
        
        Returns:
            Tuple of (items, pagination_meta)
        """
        # Get total count
        count_result = db_service.execute_query(count_query, params)
        total_items = count_result[0]['count'] if count_result else 0
        
        # Calculate offset
        offset = PaginationService.calculate_offset(page, page_size)
        
        # Add LIMIT and OFFSET
        paginated_query = f"{base_query} LIMIT %s OFFSET %s"
        paginated_params = params + (page_size, offset)
        
        # Execute
        items = db_service.execute_query(paginated_query, paginated_params)
        
        # Create metadata
        meta = PaginationService.create_meta(page, page_size, total_items)
        
        logger.info(
            f"📄 Paginated: page={page}, size={page_size}, "
            f"total={total_items}, returned={len(items)}"
        )
        
        return items, meta

# Singleton
pagination_service = PaginationService()