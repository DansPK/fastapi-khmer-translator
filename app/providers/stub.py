from app.providers.base import TranslationProvider


class StubProvider(TranslationProvider):
    """Development stub. Returns bracketed placeholders — no external calls."""

    @property
    def name(self) -> str:
        return "stub"

    async def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        return [f"[{tgt_lang}:{src_lang}] {text}" for text in texts]

    async def health_check(self) -> bool:
        return True
