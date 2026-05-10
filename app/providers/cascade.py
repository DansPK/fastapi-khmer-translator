"""
CascadeProvider — tries providers in priority order, falls back on any failure.

Each provider gets an independent asyncio timeout. If a provider raises any
exception (HTTP error, parse error, validation error, timeout), the cascade
logs a warning and immediately tries the next provider without waiting.

Provider name format: "a>b>c" — reflects the active fallback chain in logs.
"""

import asyncio
import logging
import time

from app.providers.base import TranslationProvider

logger = logging.getLogger(__name__)

_PROVIDER_TIMEOUT = 15.0  # hard cap per provider attempt (seconds)


class CascadeProvider(TranslationProvider):
    """
    Wraps an ordered list of TranslationProvider instances.
    Returns the first successful result; raises only when all providers fail.
    """

    def __init__(self, providers: list[TranslationProvider]) -> None:
        if not providers:
            raise ValueError("CascadeProvider requires at least one provider")
        self._providers = providers

    @property
    def name(self) -> str:
        return ">".join(p.name for p in self._providers)

    async def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        last_exc: BaseException | None = None

        for idx, provider in enumerate(self._providers):
            t0 = time.perf_counter()
            next_name = (
                self._providers[idx + 1].name
                if idx + 1 < len(self._providers)
                else "none"
            )
            try:
                result = await asyncio.wait_for(
                    provider.translate(texts, src_lang, tgt_lang),
                    timeout=_PROVIDER_TIMEOUT,
                )
                logger.info(
                    "cascade provider=%s status=ok texts=%d latency=%.0fms",
                    provider.name,
                    len(texts),
                    (time.perf_counter() - t0) * 1000,
                )
                return result

            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    "cascade provider=%s status=timeout latency=%.0fms next=%s",
                    provider.name,
                    (time.perf_counter() - t0) * 1000,
                    next_name,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "cascade provider=%s status=error latency=%.0fms reason=%s next=%s",
                    provider.name,
                    (time.perf_counter() - t0) * 1000,
                    exc,
                    next_name,
                )

        raise RuntimeError(
            f"All providers failed ({self.name}). Last error: {last_exc}"
        ) from last_exc

    async def health_check(self) -> bool:
        for provider in self._providers:
            try:
                if await provider.health_check():
                    return True
            except Exception:
                pass
        return False
