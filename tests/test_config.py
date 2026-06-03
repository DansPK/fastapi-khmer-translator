"""Tests for app.config — Settings defaults and get_settings."""

import os

import pytest

from app.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings(
            _env_file=None,  # don't read .env during tests
        )
        assert s.app_name == "TranslateKH Compatible API"
        assert s.app_version == "1.0.0"
        assert s.debug is False
        assert s.environment == "development"
        assert s.host == "0.0.0.0"
        assert s.port == 8000
        assert s.workers == 1
        assert s.redis_cache_ttl == 86400
        assert s.rate_limit_enabled is True
        assert s.rate_limit_per_minute == 20

    def test_provider_default_is_stub(self):
        s = Settings(_env_file=None)
        assert s.translation_provider == "stub"

    def test_custom_values(self):
        s = Settings(
            app_name="Custom",
            debug=True,
            port=9000,
            translation_provider="gemini",
            _env_file=None,
        )
        assert s.app_name == "Custom"
        assert s.debug is True
        assert s.port == 9000
        assert s.translation_provider == "gemini"

    def test_invalid_provider_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(translation_provider="invalid", _env_file=None)

    def test_invalid_environment_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(environment="unknown", _env_file=None)

    def test_api_credentials_from_env(self, monkeypatch):
        monkeypatch.delenv("API_USERNAME", raising=False)
        monkeypatch.delenv("API_PASSWORD", raising=False)
        s = Settings(_env_file=None)
        assert s.api_username == ""
        assert s.api_password == ""

    def test_provider_credentials_default_empty(self):
        s = Settings(_env_file=None)
        assert s.google_translate_api_key == ""
        assert s.gemini_api_key == ""
        assert s.openrouter_api_key == ""
