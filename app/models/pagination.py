"""
Pagination models and schemas.
Steve Jobs principle: "Simple is sophisticated"
"""

from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Optional, List
from datetime import datetime

T = TypeVar('T')

class PaginationParams(BaseModel):
    """Request parameters for pagination"""
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(10, ge=1, le=50, description="Items per page (max 50)")
    content_type: Optional[str] = Field(None, description="Filter by content type")
    category_id: Optional[int] = Field(None, description="Filter by category")
    sort_by: str = Field("published_date", description="Sort field")
    sort_order: str = Field("desc", description="asc or desc")

class PaginationMeta(BaseModel):
    """Pagination metadata"""
    current_page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_prev: bool
    next_page: Optional[int]
    prev_page: Optional[int]

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response"""
    success: bool = True
    items: List[T]
    meta: PaginationMeta
    timestamp: datetime = Field(default_factory=datetime.utcnow)