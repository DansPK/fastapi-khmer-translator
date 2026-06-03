import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.router import router
from app.cache.redis import close_redis_client, get_redis_client
from app.config import get_settings
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.providers.base import TranslationProvider
from app.providers.cascade import CascadeProvider
from app.providers.gemini import GeminiProvider
from app.providers.google_translate import GoogleTranslateProvider
from app.providers.openrouter import OpenRouterProvider
from app.providers.registry import register, resolve
from app.providers.stub import StubProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------

def _register_providers() -> None:
    register("stub", StubProvider)
    register("gemini", GeminiProvider)
    register("openrouter", OpenRouterProvider)
    register("google", _build_google_cascade)


def _build_google_cascade() -> TranslationProvider:
    """
    Build the production cascade: Google Translate → Gemini → OpenRouter.

    Each provider is constructed independently. If a provider's API key is
    missing or its __init__ raises, that provider is skipped with a warning
    rather than crashing startup. The cascade degrades gracefully to whatever
    providers are available.
    """
    candidates: list[tuple[type[TranslationProvider], str]] = [
        (GoogleTranslateProvider, "google_translate"),
        (GeminiProvider, "gemini"),
        (OpenRouterProvider, "openrouter"),
    ]

    providers: list[TranslationProvider] = []
    for factory, label in candidates:
        try:
            providers.append(factory())
            logger.info("cascade provider=%s status=registered", label)
        except Exception as exc:
            logger.warning(
                "cascade provider=%s status=skipped reason=%s", label, exc
            )

    if not providers:
        raise RuntimeError(
            "No translation providers configured. "
            "Set at least one of: GOOGLE_TRANSLATE_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY"
        )

    if len(providers) == 1:
        logger.info("cascade single_provider=%s", providers[0].name)
        return providers[0]

    return CascadeProvider(providers)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    if not settings.api_username or not settings.api_password:
        logger.warning(
            "API_USERNAME or API_PASSWORD is not set — "
            "every request will return 401. Set both in .env to enable access."
        )

    _register_providers()
    app.state.provider = resolve(settings.translation_provider)

    try:
        await get_redis_client()
        logger.info("redis connection established url=%s", settings.redis_url)
    except Exception as exc:
        logger.error(
            "redis connection failed url=%s err=%s — "
            "the app will start but caching and rate-limiting will be unavailable",
            settings.redis_url,
            exc,
        )

    yield
    await close_redis_client()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Drop-in replacement for the TranslateKH API.\n\n"
            "Change only the **base URL** and **credentials** — "
            "no frontend code changes required.\n\n"
            "**Endpoint**: `POST /api`\n\n"
            "**Authentication**: HTTP Basic Auth — "
            "set `API_USERNAME` and `API_PASSWORD` in the environment, "
            "then pass them with `-u username:password`.\n\n"
            "**Provider cascade** (transparent to clients):\n"
            "1. Google Cloud Translation — deterministic NMT\n"
            "2. Gemini Flash Lite — LLM fallback\n"
            "3. OpenRouter free — last resort"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Starlette wraps in reverse add order: last-added = outermost.
    # Desired request path: RequestLogging → RateLimit → route
    app.add_middleware(RateLimitMiddleware)       # inner — checked after logging starts
    app.add_middleware(RequestLoggingMiddleware)  # outer — logs ALL requests incl. 429s
    app.include_router(router)

    return app


app = create_app()
