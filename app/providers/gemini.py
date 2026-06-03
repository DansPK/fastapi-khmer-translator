import json
import logging
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
from app.services.validator import TranslationValidator

logger = logging.getLogger(__name__)

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

# Module-level singletons — both are stateless
_prompt_builder = PromptBuilder()
_validator = TranslationValidator()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


class GeminiProvider(TranslationProvider):

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.gemini_api_key
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._model = settings.gemini_model
        self._generate_url = f"{_API_BASE}/models/{self._model}:generateContent"
        self._model_url = f"{_API_BASE}/models/{self._model}"

    @property
    def name(self) -> str:
        return "gemini"

    async def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        for strict in (False, True):
            payload = _build_payload(texts, src_lang, tgt_lang, strict=strict)
            raw = await self._post(payload)
            results = _parse(raw, expected=len(texts))
            check = _validator.validate_batch(texts, results, tgt_lang)
            if check.valid:
                return results
            logger.warning(
                "gemini validation fail strict=%s reason=%s",
                strict,
                check.reason,
            )

        raise ValueError(f"Gemini output failed validation after strict retry: {check.reason}")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                resp = await client.get(
                    self._model_url,
                    params={"key": self._api_key},
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                attempt_num = attempt.retry_state.attempt_number
                if attempt_num > 1:
                    logger.warning(
                        "Gemini retry attempt %d for model %s",
                        attempt_num,
                        self._model,
                    )
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.post(
                        self._generate_url,
                        params={"key": self._api_key},
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()

        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Pure functions — no instance state needed
# ---------------------------------------------------------------------------

def _build_payload(texts: list[str], src_lang: str, tgt_lang: str, *, strict: bool = False) -> dict[str, Any]:
    pair = _prompt_builder.build(texts, src_lang, tgt_lang, strict=strict)
    return {
        # system_instruction keeps role separation clean for Gemini models
        "system_instruction": {
            "parts": [{"text": pair.system}]
        },
        "contents": [
            {"parts": [{"text": pair.user}]}
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "translate_text": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    }
                },
                "required": ["translate_text"],
            },
        },
    }


def _parse(data: dict[str, Any], expected: int) -> list[str]:
    candidate = data["candidates"][0]
    finish_reason = candidate.get("finishReason", "STOP")
    if finish_reason not in ("STOP", "END_OF_TURN"):
        raise ValueError(f"Gemini stopped early: finishReason={finish_reason!r}")

    try:
        text: str = candidate["content"]["parts"][0]["text"]
        results: list[str] = json.loads(text)["translate_text"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unexpected Gemini response shape: {exc}") from exc

    if len(results) != expected:
        raise ValueError(
            f"Count mismatch: expected {expected} translations, got {len(results)}"
        )
    return results
