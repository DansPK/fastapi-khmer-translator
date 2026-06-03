import logging
import time

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.cache.redis import get_redis_client
from app.config import get_settings

logger = logging.getLogger(__name__)

# Paths exempt from rate limiting
_EXCLUDED = frozenset({"/health", "/health/detailed", "/docs", "/redoc", "/openapi.json"})


def _client_ip(request: Request) -> str:
    """
    Extract the client IP from the direct TCP connection.

    Proxy headers (X-Forwarded-For, X-Real-IP) are intentionally ignored
    because they can be spoofed by any client to bypass rate limiting.
    When deployed behind a trusted reverse proxy, configure the proxy to
    set the REMOTE_ADDR / client.host correctly (e.g. Nginx real_ip module,
    or Uvicorn --proxy-headers with --forwarded-allow-ips).
    """
    if request.client:
        return request.client.host
    return "unknown"


async def _increment(ip: str) -> tuple[int, int]:
    """
    Increment the fixed-window counter for (ip, current_minute).

    Returns (count_this_window, seconds_until_window_reset).

    Uses a single pipeline transaction (MULTI/EXEC) for atomicity.
    The key TTL is set to 120 s so stale keys from previous windows
    are cleaned up automatically without blocking the current window.
    """
    now = time.time()
    window = int(now // 60)
    key = f"rate:{ip}:{window}"
    retry_after = max(1, int(60 - (now % 60)))

    client = await get_redis_client()
    async with client.pipeline(transaction=True) as pipe:
        pipe.incr(key)
        pipe.expire(key, 120)
        results = await pipe.execute()

    return int(results[0]), retry_after


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Fixed-window rate limiter: 20 req/min per IP by default.

    Fail-open: if Redis is unavailable the request is allowed through
    and a warning is emitted. This avoids a Redis outage taking down
    the translation service.

    Response headers on every non-limited request:
      X-RateLimit-Limit     — configured maximum
      X-RateLimit-Remaining — requests left in the current window
      X-RateLimit-Reset     — Unix timestamp of the next window start

    On HTTP 429:
      Retry-After           — seconds until the window resets
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _EXCLUDED:
            return await call_next(request)

        settings = get_settings()

        if not settings.rate_limit_enabled:
            return await call_next(request)

        ip = _client_ip(request)
        limit = settings.rate_limit_per_minute
        window_reset = (int(time.time() // 60) + 1) * 60

        try:
            count, retry_after = await _increment(ip)
        except Exception as exc:
            logger.warning("rate-limit: Redis unavailable, allowing ip=%s err=%s", ip, exc)
            return await call_next(request)

        if count > limit:
            logger.warning(
                "rate-limit: exceeded ip=%s count=%d limit=%d",
                ip,
                count,
                limit,
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again in the next minute."},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(window_reset),
                },
            )

        response = await call_next(request)
        remaining = max(0, limit - count)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(window_reset)
        return response
