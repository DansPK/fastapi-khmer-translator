"""Tests for app.providers.stub — StubProvider."""

import pytest

from app.providers.stub import StubProvider


@pytest.fixture()
def provider() -> StubProvider:
    return StubProvider()


class TestStubProvider:
    @pytest.mark.asyncio
    async def test_name(self, provider):
        assert provider.name == "stub"

    @pytest.mark.asyncio
    async def test_translate_single(self, provider):
        result = await provider.translate(["Hello"], "en", "kh")
        assert result == ["[kh:en] Hello"]

    @pytest.mark.asyncio
    async def test_translate_batch(self, provider):
        result = await provider.translate(["A", "B", "C"], "en", "kh")
        assert len(result) == 3
        assert result[0] == "[kh:en] A"
        assert result[2] == "[kh:en] C"

    @pytest.mark.asyncio
    async def test_translate_preserves_order(self, provider):
        texts = [f"text_{i}" for i in range(5)]
        result = await provider.translate(texts, "eng", "kh")
        for i, r in enumerate(result):
            assert f"text_{i}" in r

    @pytest.mark.asyncio
    async def test_health_check(self, provider):
        assert await provider.health_check() is True
