"""
Google Cloud Translation API v2 (Neural Machine Translation) provider.

This is the dedicated Google Cloud Translation REST service — NOT Gemini AI Studio.
Endpoint: https://translation.googleapis.com/language/translate/v2

Characteristics vs. LLM providers:
  - Deterministic: same input always produces same output
  - Fast: sub-second latency for typical batches
  - No hallucination: purpose-built NMT, not a chat model
  - Handles technical terms natively — proper nouns, brand names, and
    established technical vocabulary are preserved without prompt engineering

Auth: Cloud Console API key with "Cloud Translation API" enabled.
Env:  GOOGLE_TRANSLATE_API_KEY

Retry policy:
  429, 5xx, timeout → retry with exponential back-off (max 3 attempts)
  400, 401, 403, 404 → fail-fast (no retry; cascade will try next provider)
"""

import logging
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
from app.services.validator import TranslationValidator

logger = logging.getLogger(__name__)

_API_URL = "https://translation.googleapis.com/language/translate/v2"

_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_validator = TranslationValidator()

# ---------------------------------------------------------------------------
# Language-code normalization
# TranslateKH uses non-standard codes ("kh", "eng", "viet" …).
# Google Translate expects ISO 639-1 / BCP 47 codes.
# ---------------------------------------------------------------------------

_LANG_MAP: dict[str, str] = {
    # Khmer
    "kh": "km",   "khm": "km",  "km": "km",
    # English
    "eng": "en",  "en": "en",
    # South-East Asia
    "vi": "vi",   "viet": "vi",
    "th": "th",
    "id": "id",
    "ms": "ms",
    "my": "my",
    "lo": "lo",
    # East Asia
    "zh": "zh",   "zh-cn": "zh-CN",  "zh-tw": "zh-TW",
    "ja": "ja",
    "ko": "ko",
    # European
    "fr": "fr",   "de": "de",   "es": "es",   "pt": "pt",
    "ru": "ru",   "it": "it",   "nl": "nl",
    # Other
    "ar": "ar",   "hi": "hi",   "tr": "tr",
}


def _to_google_lang(code: str) -> str | None:
    """
    Convert a TranslateKH-style language code to a Google Translate code.
    Returns None for "auto" (omitting the source param triggers auto-detection).
    Falls back to the raw code if unknown (Google may still accept it).
    """
    lower = code.lower()
    if lower == "auto":
        return None
    return _LANG_MAP.get(lower, lower)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class GoogleTranslateProvider(TranslationProvider):
    """
    Google Cloud Translation API v2 provider.

    Language codes are normalized from TranslateKH format (kh, eng, viet …)
    to Google format (km, en, vi …) before each request.

    Output is validated with TranslationValidator before being returned.
    Redis caching is handled upstream by TranslationService.
    """

    def __init__(self) -> None:
        api_key = get_settings().google_translate_api_key
        if not api_key:
            raise ValueError("GOOGLE_TRANSLATE_API_KEY is not set")
        self._api_key = api_key  # never logged

    @property
    def name(self) -> str:
        return "google_translate"

    async def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        t0 = time.perf_counter()

        raw = await self._post(texts, src_lang, tgt_lang)
        results = _parse(raw, expected=len(texts))

        check = _validator.validate_batch(texts, results, tgt_lang)
        if not check.valid:
            raise ValueError(
                f"Google Translate output failed validation: {check.reason}"
            )

        logger.info(
            "provider=google_translate status=ok texts=%d latency=%.0fms",
            len(texts),
            (time.perf_counter() - t0) * 1000,
        )
        return results

    async def health_check(self) -> bool:
        try:
            await self.translate(["ok"], "en", "fr")
            return True
        except Exception as exc:
            logger.warning("health_check provider=google_translate status=error reason=%s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> dict[str, Any]:
        payload = _build_payload(texts, src_lang, tgt_lang)

        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                n = attempt.retry_state.attempt_number
                if n > 1:
                    logger.warning(
                        "provider=google_translate retry attempt=%d", n
                    )
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.post(
                        _API_URL,
                        params={"key": self._api_key},  # key in query param, not header
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()

        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _build_payload(texts: list[str], src_lang: str, tgt_lang: str) -> dict[str, Any]:
    tgt = _to_google_lang(tgt_lang)
    payload: dict[str, Any] = {
        "q": texts,
        "target": tgt or "en",  # fallback guard; tgt should always be valid
        "format": "text",
    }
    src = _to_google_lang(src_lang)
    if src:
        payload["source"] = src
    # omitting "source" triggers Google's auto-detection
    return payload


def _parse(data: dict[str, Any], expected: int) -> list[str]:
    """
    Extract translated strings from Google Translate v2 response.

    Response shape:
      {"data": {"translations": [{"translatedText": "..."}]}}
    """
    try:
        translations: list[dict[str, Any]] = data["data"]["translations"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unexpected Google Translate response shape: {exc}"
        ) from exc

    if not isinstance(translations, list):
        raise ValueError(
            f"'translations' must be an array, got {type(translations).__name__}"
        )

    results = []
    for i, item in enumerate(translations):
        text = item.get("translatedText")
        if not isinstance(text, str):
            raise ValueError(
                f"'translatedText' missing or non-string at index {i}: {item!r}"
            )
        results.append(text)

    if len(results) != expected:
        raise ValueError(
            f"Count mismatch: expected {expected} translations, got {len(results)}"
        )

    return results
