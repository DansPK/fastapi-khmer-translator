from abc import ABC, abstractmethod


class TranslationProvider(ABC):
    """
    Contract all translation backends must satisfy.
    Input and output are parallel lists — index i in output is the translation of index i in input.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]: ...

    async def health_check(self) -> bool:
        return True
