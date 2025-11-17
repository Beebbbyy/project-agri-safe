"""
Redis Feature Caching Layer

Provides caching for computed features and aggregations to improve
API response times and reduce database load.
"""

import json
import os
from typing import Any, Optional, Union
from datetime import timedelta
import redis
from loguru import logger


class RedisFeatureCache:
    """
    Redis cache manager for weather features and aggregations.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True
    ):
        """
        Initialize Redis connection.

        Args:
            host: Redis host (default from env: REDIS_HOST)
            port: Redis port (default from env: REDIS_PORT)
            db: Redis database number
            password: Redis password (if required)
            decode_responses: Automatically decode byte responses to strings
        """
        self.host = host or os.getenv("REDIS_HOST", "redis")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.db = db
        self.password = password or os.getenv("REDIS_PASSWORD")

        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _make_key(self, namespace: str, identifier: str) -> str:
        """
        Create a namespaced cache key.

        Args:
            namespace: Cache namespace (e.g., 'weather', 'risk', 'features')
            identifier: Unique identifier within namespace

        Returns:
            str: Formatted cache key
        """
        return f"agrisafe:{namespace}:{identifier}"

    def set(
        self,
        namespace: str,
        identifier: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """
        Set a cache value with optional TTL.

        Args:
            namespace: Cache namespace
            identifier: Unique identifier
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds or timedelta

        Returns:
            bool: True if successful
        """
        key = self._make_key(namespace, identifier)

        try:
            # Serialize value to JSON
            serialized = json.dumps(value, default=str)

            if ttl:
                if isinstance(ttl, timedelta):
                    ttl = int(ttl.total_seconds())
                result = self.client.setex(key, ttl, serialized)
            else:
                result = self.client.set(key, serialized)

            logger.debug(f"Cached: {key} (TTL: {ttl})")
            return bool(result)

        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            return False

    def get(
        self,
        namespace: str,
        identifier: str,
        default: Any = None
    ) -> Any:
        """
        Get a cached value.

        Args:
            namespace: Cache namespace
            identifier: Unique identifier
            default: Default value if not found

        Returns:
            Cached value or default
        """
        key = self._make_key(namespace, identifier)

        try:
            value = self.client.get(key)

            if value is None:
                logger.debug(f"Cache miss: {key}")
                return default

            # Deserialize JSON
            logger.debug(f"Cache hit: {key}")
            return json.loads(value)

        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            return default

    def delete(self, namespace: str, identifier: str) -> bool:
        """
        Delete a cached value.

        Args:
            namespace: Cache namespace
            identifier: Unique identifier

        Returns:
            bool: True if key was deleted
        """
        key = self._make_key(namespace, identifier)

        try:
            result = self.client.delete(key)
            logger.debug(f"Deleted cache key: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., 'agrisafe:weather:*')

        Returns:
            int: Number of keys deleted
        """
        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Deleted {deleted} keys matching: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {e}")
            return 0

    def exists(self, namespace: str, identifier: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            namespace: Cache namespace
            identifier: Unique identifier

        Returns:
            bool: True if key exists
        """
        key = self._make_key(namespace, identifier)
        return bool(self.client.exists(key))

    def get_ttl(self, namespace: str, identifier: str) -> int:
        """
        Get remaining TTL for a key.

        Args:
            namespace: Cache namespace
            identifier: Unique identifier

        Returns:
            int: Remaining seconds (-1 if no expiry, -2 if not exists)
        """
        key = self._make_key(namespace, identifier)
        return self.client.ttl(key)

    def cache_weather_stats(
        self,
        region_id: int,
        stats: dict,
        ttl: int = 3600
    ):
        """
        Cache weather statistics for a region.

        Args:
            region_id: Region ID
            stats: Statistics dictionary
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.set("weather:stats", str(region_id), stats, ttl)

    def get_weather_stats(self, region_id: int) -> Optional[dict]:
        """
        Get cached weather statistics for a region.

        Args:
            region_id: Region ID

        Returns:
            Statistics dictionary or None
        """
        return self.get("weather:stats", str(region_id))

    def cache_rolling_features(
        self,
        region_id: int,
        window_days: int,
        features: dict,
        ttl: int = 3600
    ):
        """
        Cache rolling window features.

        Args:
            region_id: Region ID
            window_days: Window size (7, 14, or 30 days)
            features: Features dictionary
            ttl: Time to live in seconds
        """
        identifier = f"{region_id}:{window_days}d"
        self.set("features:rolling", identifier, features, ttl)

    def get_rolling_features(
        self,
        region_id: int,
        window_days: int
    ) -> Optional[dict]:
        """
        Get cached rolling window features.

        Args:
            region_id: Region ID
            window_days: Window size

        Returns:
            Features dictionary or None
        """
        identifier = f"{region_id}:{window_days}d"
        return self.get("features:rolling", identifier)

    def cache_risk_indicators(
        self,
        region_id: int,
        indicators: dict,
        ttl: int = 1800
    ):
        """
        Cache flood risk indicators for a region.

        Args:
            region_id: Region ID
            indicators: Risk indicators dictionary
            ttl: Time to live in seconds (default: 30 minutes)
        """
        self.set("risk:indicators", str(region_id), indicators, ttl)

    def get_risk_indicators(self, region_id: int) -> Optional[dict]:
        """
        Get cached flood risk indicators.

        Args:
            region_id: Region ID

        Returns:
            Risk indicators dictionary or None
        """
        return self.get("risk:indicators", str(region_id))

    def invalidate_region_cache(self, region_id: int):
        """
        Invalidate all cached data for a region.

        Args:
            region_id: Region ID
        """
        patterns = [
            f"agrisafe:weather:*:{region_id}",
            f"agrisafe:features:*:{region_id}*",
            f"agrisafe:risk:*:{region_id}",
        ]

        total_deleted = 0
        for pattern in patterns:
            total_deleted += self.delete_pattern(pattern)

        logger.info(f"Invalidated {total_deleted} cache entries for region {region_id}")

    def flush_all(self):
        """
        Flush all cache data (use with caution!).
        """
        logger.warning("Flushing all Redis cache data")
        self.client.flushdb()

    def get_info(self) -> dict:
        """
        Get Redis server info.

        Returns:
            dict: Redis server information
        """
        return self.client.info()

    def close(self):
        """
        Close Redis connection.
        """
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")
