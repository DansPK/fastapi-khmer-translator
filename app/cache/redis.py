import json
import logging

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: Redis | None = None


async def get_redis_client() -> Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(
            get_settings().redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
    return _client


async def close_redis_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


class RedisCache:
    """
    Translation result cache.

    All operations fail-open: a Redis outage degrades to no caching,
    never to a 500 error for the caller.
    """

    def __init__(self, client: Redis, ttl: int) -> None:
        self._client = client
        self._ttl = ttl

    async def get(self, key: str) -> list[str] | None:
        try:
            value = await self._client.get(key)
        except Exception as exc:
            logger.warning("cache get failed key=%s err=%s", key, exc)
            return None

        if value is None:
            logger.debug("cache miss key=%s", key)
            return None

        try:
            result: list[str] = json.loads(value)
            logger.info("cache hit  key=%s items=%d", key, len(result))
            return result
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("cache corrupt key=%s err=%s", key, exc)
            return None

    async def set(self, key: str, value: list[str]) -> None:
        try:
            await self._client.setex(
                key, self._ttl, json.dumps(value, ensure_ascii=False)
            )
            logger.debug("cache set  key=%s items=%d ttl=%ds", key, len(value), self._ttl)
        except Exception as exc:
            logger.warning("cache set failed key=%s err=%s", key, exc)

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception as exc:
            logger.warning("redis ping failed err=%s", exc)
            return False
