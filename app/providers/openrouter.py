"""
OpenRouter translation provider with Gemini fallback.

Response parsing:
  choices[0].message.content  (OpenAI-compatible schema)

Retry policy (per HTTP attempt):
  429, 5xx, timeout → retry with exponential back-off (max 2 attempts)
  401, 403, 404     → fail-fast, no retry
  parse errors      → fail-fast, no retry

Fallback chain:
  1. openrouter/free  — 5 s per-provider hard cap
  2. GeminiProvider   — 5 s per-provider hard cap (requires GEMINI_API_KEY)

Total budget: 10 s across both providers.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.providers.base import TranslationProvider
from app.services.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

_prompt_builder = PromptBuilder()

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODELS_URL = "https://openrouter.ai/api/v1/models"

_MODEL = "openrouter/free"

# HTTP-level timeout: connect + read total ≈ 5 s
_HTTP_TIMEOUT = httpx.Timeout(connect=3.0, read=4.5, write=2.0, pool=2.0)

_PROVIDER_TIMEOUT = 5.0   # asyncio hard cap per provider (seconds)
_TOTAL_TIMEOUT    = 10.0  # hard cap across all providers (seconds)

# 429, 5xx, timeout → retry with back-off
# 401, 403, 404, others → fail-fast (not in set → _is_retryable returns False → no retry)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)


class _FallbackError(Exception):
    """OpenRouter failed — Gemini fallback should be attempted."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


def _init_gemini_fallback() -> "TranslationProvider | None":
    """
    Lazily import and instantiate GeminiProvider to avoid circular imports.
    Returns None if GEMINI_API_KEY is missing or GeminiProvider fails to init.
    """
    try:
        from app.providers.gemini import GeminiProvider  # noqa: PLC0415
        provider = GeminiProvider()
        logger.info("provider=gemini status=fallback_ready")
        return provider
    except Exception as exc:
        logger.warning("provider=gemini status=fallback_unavailable reason=%s", exc)
        return None


class OpenRouterProvider(TranslationProvider):

    def __init__(self) -> None:
        api_key = get_settings().openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "TranslationAPI",
        }
        self._gemini: TranslationProvider | None = _init_gemini_fallback()

    @property
    def name(self) -> str:
        return "openrouter"

    async def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        deadline = time.perf_counter() + _TOTAL_TIMEOUT

        # --- Provider 1: OpenRouter free ---
        try:
            return await asyncio.wait_for(
                self._call_openrouter(texts, src_lang, tgt_lang),
                timeout=_PROVIDER_TIMEOUT,
            )
        except (asyncio.TimeoutError, _FallbackError) as exc:
            logger.warning(
                "provider=openrouter status=fallback reason=%s gemini=%s",
                exc,
                "available" if self._gemini is not None else "unavailable",
            )
        # auth errors and other hard failures propagate — no fallback

        # --- Provider 2: Gemini fallback ---
        if self._gemini is None:
            raise RuntimeError(
                "OpenRouter failed and Gemini fallback is not configured "
                "(set GEMINI_API_KEY to enable it)"
            )

        remaining = deadline - time.perf_counter()
        if remaining < 0.5:
            raise RuntimeError("Total timeout budget exhausted before Gemini fallback")

        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._gemini.translate(texts, src_lang, tgt_lang),
                timeout=min(remaining, _PROVIDER_TIMEOUT),
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Gemini fallback timed out after {_PROVIDER_TIMEOUT:.0f}s"
            ) from exc

        logger.info(
            "provider=gemini status=ok texts=%d latency=%.0fms source=fallback",
            len(texts),
            (time.perf_counter() - t0) * 1000,
        )
        return result

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(_MODELS_URL, headers=self._headers)
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("health_check provider=openrouter status=error reason=%s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _call_openrouter(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        payload = _build_payload(texts, src_lang, tgt_lang)
        t0 = time.perf_counter()

        try:
            raw = await self._post(payload)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning(
                "provider=openrouter model=%s status=http_%d latency=%.0fms",
                _MODEL, status, (time.perf_counter() - t0) * 1000,
            )
            if status in (401, 403):
                raise  # auth failures propagate — skip fallback
            raise _FallbackError(f"HTTP {status}") from exc
        except httpx.TimeoutException as exc:
            logger.warning(
                "provider=openrouter model=%s status=timeout latency=%.0fms",
                _MODEL, (time.perf_counter() - t0) * 1000,
            )
            raise _FallbackError("http_timeout") from exc
        except Exception as exc:
            raise _FallbackError(str(exc)) from exc

        try:
            results = _parse(raw, expected=len(texts))
        except ValueError as exc:
            logger.warning(
                "provider=openrouter model=%s status=parse_error reason=%s",
                _MODEL, exc,
            )
            raise _FallbackError(str(exc)) from exc

        logger.info(
            "provider=openrouter model=%s status=ok texts=%d latency=%.0fms",
            _MODEL, len(texts), (time.perf_counter() - t0) * 1000,
        )
        return results

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            reraise=True,
        ):
            with attempt:
                n = attempt.retry_state.attempt_number
                if n > 1:
                    logger.warning(
                        "provider=openrouter model=%s retry attempt=%d",
                        payload.get("model"), n,
                    )
                async with httpx.AsyncClient(
                    timeout=_HTTP_TIMEOUT,
                    headers=self._headers,
                ) as client:
                    resp = await client.post(_API_URL, json=payload)
                    resp.raise_for_status()
                    return resp.json()

        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Pure functions — no instance state required
# ---------------------------------------------------------------------------

def _build_payload(
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
) -> dict[str, Any]:
    pair = _prompt_builder.build(texts, src_lang, tgt_lang)
    return {
        "model": _MODEL,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": pair.system},
            {"role": "user",   "content": pair.user},
        ],
    }


def _extract_json(content: str) -> dict[str, Any]:
    """
    Three-pass extraction for models that ignore response_format:
      1. Direct JSON parse
      2. Markdown code block  ```json ... ```
      3. Brace-span scan  { ... }
    """
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    m = _JSON_BLOCK_RE.search(content)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot extract JSON from model output: {content[:200]!r}")


def _parse(data: dict[str, Any], expected: int) -> list[str]:
    """Parse OpenRouter response: choices[0].message.content → translate_text array."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected response envelope: {exc}") from exc

    finish_reason = choice.get("finish_reason")
    if finish_reason not in ("stop", "end_turn", None):
        raise ValueError(f"Model stopped early: finish_reason={finish_reason!r}")

    try:
        content: str = choice["message"]["content"]
    except KeyError as exc:
        raise ValueError(f"Missing message.content: {exc}") from exc

    parsed = _extract_json(content)

    if "translate_text" not in parsed:
        raise ValueError(f"Missing 'translate_text' key. Got keys: {list(parsed)}")

    results = parsed["translate_text"]

    if not isinstance(results, list):
        raise ValueError(
            f"'translate_text' must be an array, got {type(results).__name__}"
        )

    if any(not isinstance(r, str) for r in results):
        raise ValueError("'translate_text' items must all be strings")

    if len(results) != expected:
        raise ValueError(
            f"Count mismatch: expected {expected} translations, got {len(results)}"
        )

    return results
