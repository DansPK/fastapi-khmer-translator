"""Tests for app.services.translation — TranslationService and _cache_key."""

from unittest.mock import AsyncMock, PropertyMock

import pytest

from app.cache.redis import RedisCache
from app.providers.base import TranslationProvider
from app.services.translation import TranslationService, _cache_key


class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key("en", "kh", ["Hello"])
        k2 = _cache_key("en", "kh", ["Hello"])
        assert k1 == k2

    def test_different_texts_different_keys(self):
        k1 = _cache_key("en", "kh", ["Hello"])
        k2 = _cache_key("en", "kh", ["World"])
        assert k1 != k2

    def test_different_langs_different_keys(self):
        k1 = _cache_key("en", "kh", ["Hello"])
        k2 = _cache_key("en", "fr", ["Hello"])
        assert k1 != k2

    def test_order_sensitive(self):
        k1 = _cache_key("en", "kh", ["A", "B"])
        k2 = _cache_key("en", "kh", ["B", "A"])
        assert k1 != k2

    def test_format(self):
        key = _cache_key("en", "kh", ["test"])
        assert key.startswith("translate:en:kh:")
        # Hash part is a 64-char hex string
        hash_part = key.split(":")[-1]
        assert len(hash_part) == 64


class TestTranslationService:
    def _make_service(self):
        provider = AsyncMock(spec=TranslationProvider)
        type(provider).name = PropertyMock(return_value="mock")
        provider.translate.return_value = ["translated"]

        cache = AsyncMock(spec=RedisCache)
        cache.get.return_value = None
        cache.set.return_value = None

        return TranslationService(provider, cache), provider, cache

    @pytest.mark.asyncio
    async def test_cache_miss_calls_provider(self):
        service, provider, cache = self._make_service()
        result = await service.translate(["hello"], "en", "kh")
        assert result == ["translated"]
        provider.translate.assert_called_once_with(["hello"], "en", "kh")
        cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_provider(self):
        service, provider, cache = self._make_service()
        cache.get.return_value = ["cached_result"]

        result = await service.translate(["hello"], "en", "kh")
        assert result == ["cached_result"]
        provider.translate.assert_not_called()
        cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_is_cached_after_provider_call(self):
        service, provider, cache = self._make_service()
        await service.translate(["hello"], "en", "kh")
        cache.set.assert_called_once()
        # The first positional arg to set() is the key
        call_args = cache.set.call_args
        assert call_args[0][1] == ["translated"]
