import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            ms = (time.perf_counter() - start) * 1000
            logger.error(
                "%s %s → unhandled exception (%s) (%.1fms)",
                request.method,
                request.url.path,
                exc,
                ms,
            )
            raise
        ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            ms,
        )
        return response
