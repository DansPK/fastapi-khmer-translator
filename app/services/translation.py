import hashlib
import json
import logging

from app.cache.redis import RedisCache
from app.providers.base import TranslationProvider

logger = logging.getLogger(__name__)


class TranslationService:
    def __init__(self, provider: TranslationProvider, cache: RedisCache) -> None:
        self._provider = provider
        self._cache = cache

    async def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        key = _cache_key(src_lang, tgt_lang, texts)

        cached = await self._cache.get(key)
        if cached is not None:
            return cached

        logger.info(
            "cache miss key=%s provider=%s texts=%d",
            key,
            self._provider.name,
            len(texts),
        )

        results = await self._provider.translate(texts, src_lang, tgt_lang)
        await self._cache.set(key, results)
        return results


def _cache_key(src_lang: str, tgt_lang: str, texts: list[str]) -> str:
    """
    Deterministic cache key — provider-agnostic, order-sensitive.
    Format: translate:{src}:{tgt}:{sha256_of_texts}
    """
    text_hash = hashlib.sha256(
        json.dumps(texts, ensure_ascii=False).encode()
    ).hexdigest()
    return f"translate:{src_lang}:{tgt_lang}:{text_hash}"
