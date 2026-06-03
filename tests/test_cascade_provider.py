"""Tests for app.providers.cascade — CascadeProvider."""

import asyncio
from unittest.mock import AsyncMock, PropertyMock

import pytest

from app.providers.base import TranslationProvider
from app.providers.cascade import CascadeProvider


def _make_provider(name: str, result: list[str] | Exception) -> TranslationProvider:
    """Create a mock TranslationProvider."""
    mock = AsyncMock(spec=TranslationProvider)
    type(mock).name = PropertyMock(return_value=name)
    if isinstance(result, Exception):
        mock.translate.side_effect = result
    else:
        mock.translate.return_value = result
    mock.health_check.return_value = True
    return mock


class TestCascadeProvider:
    def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError, match="at least one"):
            CascadeProvider([])

    def test_name_concatenation(self):
        p1 = _make_provider("a", ["x"])
        p2 = _make_provider("b", ["y"])
        cascade = CascadeProvider([p1, p2])
        assert cascade.name == "a>b"

    @pytest.mark.asyncio
    async def test_returns_first_successful(self):
        p1 = _make_provider("first", ["translated"])
        p2 = _make_provider("second", ["fallback"])
        cascade = CascadeProvider([p1, p2])

        result = await cascade.translate(["hello"], "en", "kh")
        assert result == ["translated"]
        p1.translate.assert_called_once()
        p2.translate.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_on_exception(self):
        p1 = _make_provider("first", RuntimeError("fail"))
        p2 = _make_provider("second", ["fallback"])
        cascade = CascadeProvider([p1, p2])

        result = await cascade.translate(["hello"], "en", "kh")
        assert result == ["fallback"]

    @pytest.mark.asyncio
    async def test_all_fail_raises(self):
        p1 = _make_provider("a", RuntimeError("fail1"))
        p2 = _make_provider("b", RuntimeError("fail2"))
        cascade = CascadeProvider([p1, p2])

        with pytest.raises(RuntimeError, match="All providers failed"):
            await cascade.translate(["hello"], "en", "kh")

    @pytest.mark.asyncio
    async def test_timeout_triggers_fallback(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(60)
            return ["never"]

        p1 = AsyncMock(spec=TranslationProvider)
        type(p1).name = PropertyMock(return_value="slow")
        p1.translate.side_effect = slow

        p2 = _make_provider("fast", ["ok"])
        cascade = CascadeProvider([p1, p2])

        result = await cascade.translate(["hello"], "en", "kh")
        assert result == ["ok"]

    @pytest.mark.asyncio
    async def test_health_check_any_healthy(self):
        p1 = _make_provider("a", ["x"])
        p1.health_check.return_value = False
        p2 = _make_provider("b", ["y"])
        p2.health_check.return_value = True
        cascade = CascadeProvider([p1, p2])
        assert await cascade.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_all_unhealthy(self):
        p1 = _make_provider("a", ["x"])
        p1.health_check.return_value = False
        p2 = _make_provider("b", ["y"])
        p2.health_check.return_value = False
        cascade = CascadeProvider([p1, p2])
        assert await cascade.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_exception_treated_as_unhealthy(self):
        p1 = _make_provider("a", ["x"])
        p1.health_check.side_effect = RuntimeError("boom")
        cascade = CascadeProvider([p1])
        assert await cascade.health_check() is False
