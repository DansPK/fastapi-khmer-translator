"""Shared fixtures for the test suite."""

import os
from unittest.mock import AsyncMock

import pytest

# Ensure pydantic-settings picks up test-friendly defaults and doesn't
# require a real .env file or real API keys.
os.environ.setdefault("TRANSLATION_PROVIDER", "stub")
os.environ.setdefault("API_USERNAME", "testuser")
os.environ.setdefault("API_PASSWORD", "testpass")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from app.config import Settings, get_settings  # noqa: E402


@pytest.fixture()
def settings() -> Settings:
    """Return a fresh Settings instance with test defaults (bypasses lru_cache)."""
    return Settings(
        translation_provider="stub",
        api_username="testuser",
        api_password="testpass",
        redis_url="redis://localhost:6379/0",
        rate_limit_enabled=False,
    )


@pytest.fixture()
def mock_redis() -> AsyncMock:
    """AsyncMock that behaves like a ``redis.asyncio.Redis`` client."""
    client = AsyncMock()
    client.get.return_value = None
    client.setex.return_value = True
    client.ping.return_value = True
    client.pipeline.return_value.__aenter__ = AsyncMock(return_value=client)
    client.pipeline.return_value.__aexit__ = AsyncMock(return_value=False)
    return client
