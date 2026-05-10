from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.cache.redis import RedisCache, get_redis_client
from app.config import Settings, get_settings
from app.core.dependencies import get_provider
from app.schemas.health import HealthResponse, ServiceStatus

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns liveness status of the API, Redis, and the active translation provider.",
)
async def health(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    provider = get_provider(request)

    redis_client = await get_redis_client()
    cache = RedisCache(redis_client, settings.redis_cache_ttl)
    redis_ok = await cache.ping()
    provider_ok = await provider.health_check()

    overall = "healthy" if (redis_ok and provider_ok) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        services={
            "redis": ServiceStatus(status="up" if redis_ok else "down"),
            "provider": ServiceStatus(
                status="up" if provider_ok else "down",
                detail=provider.name,
            ),
        },
    )
