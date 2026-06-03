"""Tests for app.cache.redis — RedisCache."""

import json
from unittest.mock import AsyncMock

import pytest

from app.cache.redis import RedisCache


@pytest.fixture()
def cache() -> RedisCache:
    client = AsyncMock()
    client.get.return_value = None
    client.setex.return_value = True
    client.ping.return_value = True
    return RedisCache(client, ttl=3600)


class TestRedisCache:
    @pytest.mark.asyncio
    async def test_get_miss(self, cache):
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_hit(self, cache):
        cache._client.get.return_value = json.dumps(["hello", "world"])
        result = await cache.get("key")
        assert result == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_get_corrupt_json(self, cache):
        cache._client.get.return_value = "not valid json{{"
        result = await cache.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_redis_error(self, cache):
        cache._client.get.side_effect = ConnectionError("redis down")
        result = await cache.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_stores_value(self, cache):
        await cache.set("key", ["translated"])
        cache._client.setex.assert_called_once_with(
            "key", 3600, json.dumps(["translated"], ensure_ascii=False)
        )

    @pytest.mark.asyncio
    async def test_set_redis_error_swallowed(self, cache):
        cache._client.setex.side_effect = ConnectionError("redis down")
        # Should not raise
        await cache.set("key", ["value"])

    @pytest.mark.asyncio
    async def test_ping_success(self, cache):
        assert await cache.ping() is True

    @pytest.mark.asyncio
    async def test_ping_failure(self, cache):
        cache._client.ping.side_effect = ConnectionError("down")
        assert await cache.ping() is False
