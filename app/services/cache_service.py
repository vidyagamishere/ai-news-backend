"""
Optional caching service for improved performance.
Use Redis or in-memory caching.
"""

from typing import Optional, Any
import json
import hashlib
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# Simple in-memory cache (replace with Redis in production)
_cache: dict = {}

class CacheService:
    """
    Simple caching for pagination results.
    Steve Jobs: "Performance is a feature"
    """
    
    @staticmethod
    def generate_key(endpoint: str, params: dict) -> str:
        """Generate cache key from endpoint and params"""
        params_str = json.dumps(params, sort_keys=True)
        key = f"{endpoint}:{params_str}"
        return hashlib.md5(key.encode()).hexdigest()
    
    @staticmethod
    def get(key: str) -> Optional[Any]:
        """Get cached value"""
        value = _cache.get(key)
        if value:
            logger.info(f"✅ Cache HIT: {key[:8]}...")
        else:
            logger.info(f"❌ Cache MISS: {key[:8]}...")
        return value
    
    @staticmethod
    def set(key: str, value: Any, ttl: int = 300):
        """Set cached value (ttl in seconds)"""
        _cache[key] = value
        logger.info(f"💾 Cached: {key[:8]}... (TTL: {ttl}s)")
        # TODO: Implement TTL with asyncio.sleep or Redis
    
    @staticmethod
    def invalidate(pattern: str = None):
        """Invalidate cache (all or by pattern)"""
        if pattern:
            keys_to_delete = [k for k in _cache.keys() if pattern in k]
            for key in keys_to_delete:
                del _cache[key]
            logger.info(f"🗑️ Invalidated {len(keys_to_delete)} cache entries")
        else:
            _cache.clear()
            logger.info("🗑️ Cleared entire cache")

cache_service = CacheService()

def cache_response(ttl: int = 300):
    """Decorator to cache endpoint responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_service.generate_key(
                func.__name__,
                {**kwargs}
            )
            
            # Check cache
            cached = cache_service.get(cache_key)
            if cached:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache_service.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator