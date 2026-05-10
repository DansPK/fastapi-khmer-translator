from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "TranslateKH Compatible API"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 86400  # 24 hours

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 20

    # HTTP Basic Auth — credentials clients must send (like TranslateKH)
    api_username: str = ""
    api_password: str = ""

    # Provider selection
    # "google"     — GoogleTranslate → Gemini → OpenRouter cascade (recommended)
    # "gemini"     — Gemini only
    # "openrouter" — OpenRouter free with Gemini fallback
    # "stub"       — returns bracketed placeholders (dev/testing only)
    translation_provider: Literal["stub", "gemini", "openrouter", "google"] = "stub"

    # Provider credentials — keys are never logged
    google_translate_api_key: str = ""   # Cloud Translation API key (not Gemini)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    openrouter_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
