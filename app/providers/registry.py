from typing import Callable

from app.providers.base import TranslationProvider

_registry: dict[str, Callable[[], TranslationProvider]] = {}


def register(name: str, factory: Callable[[], TranslationProvider]) -> None:
    _registry[name] = factory


def resolve(name: str) -> TranslationProvider:
    factory = _registry.get(name)
    if factory is None:
        raise ValueError(
            f"Provider '{name}' is not registered. Available: {sorted(_registry)}"
        )
    return factory()


def available() -> list[str]:
    return sorted(_registry)
