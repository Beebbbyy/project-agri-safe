import redis.asyncio as redis
from typing import Optional, Any
import json
from src.api.config import settings

# Create Redis client
redis_client = redis.from_url(
    settings.REDIS_URL,
    encoding="utf-8",
    decode_responses=True
)


class CacheService:
    """Redis caching service"""

    def __init__(self):
        self.client = redis_client

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = await self.client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache (will be JSON-serialized)
            ttl: Time-to-live in seconds
        """
        serialized = json.dumps(value) if not isinstance(value, str) else value

        if ttl:
            return await self.client.setex(key, ttl, serialized)
        else:
            return await self.client.set(key, serialized)

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        return await self.client.delete(key) > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return await self.client.exists(key) > 0

    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern

        Args:
            pattern: Redis pattern (e.g., "forecast:*")

        Returns:
            Number of keys deleted
        """
        keys = []
        async for key in self.client.scan_iter(match=pattern):
            keys.append(key)

        if keys:
            return await self.client.delete(*keys)
        return 0
