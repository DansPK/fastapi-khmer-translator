import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from redis.asyncio import Redis

from app.cache.redis import RedisCache, get_redis_client
from app.config import Settings, get_settings
from app.providers.base import TranslationProvider
from app.services.translation import TranslationService

# Registers "basicAuth" in the OpenAPI securitySchemes and shows the
# Authorize button in Swagger UI automatically.
_basic_auth = HTTPBasic()


def verify_credentials(
    credentials: Annotated[HTTPBasicCredentials, Depends(_basic_auth)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """
    HTTP Basic Auth guard — mirrors TranslateKH authentication style.

    Uses secrets.compare_digest for constant-time comparison to prevent
    timing attacks. Raises 401 with WWW-Authenticate: Basic on failure.
    """
    expected_user = settings.api_username.encode("utf-8")
    expected_pass = settings.api_password.encode("utf-8")

    user_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"), expected_user
    )
    pass_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"), expected_pass
    )

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


def get_provider(request: Request) -> TranslationProvider:
    return request.app.state.provider


async def get_cache(
    redis: Annotated[Redis, Depends(get_redis_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RedisCache:
    return RedisCache(redis, settings.redis_cache_ttl)


async def get_translation_service(
    provider: Annotated[TranslationProvider, Depends(get_provider)],
    cache: Annotated[RedisCache, Depends(get_cache)],
) -> TranslationService:
    return TranslationService(provider, cache)
