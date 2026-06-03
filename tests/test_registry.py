"""Tests for app.providers.registry — register / resolve / available."""

import pytest

from app.providers.base import TranslationProvider
from app.providers.registry import _registry, available, register, resolve
from app.providers.stub import StubProvider


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure tests don't leak registrations into each other."""
    saved = dict(_registry)
    _registry.clear()
    yield
    _registry.clear()
    _registry.update(saved)


class TestRegister:
    def test_register_and_resolve(self):
        register("stub", StubProvider)
        provider = resolve("stub")
        assert isinstance(provider, StubProvider)
        assert provider.name == "stub"

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            resolve("nonexistent")

    def test_available_returns_sorted(self):
        register("b_provider", StubProvider)
        register("a_provider", StubProvider)
        assert available() == ["a_provider", "b_provider"]

    def test_register_overwrite(self):
        register("test", StubProvider)
        register("test", StubProvider)
        assert resolve("test").name == "stub"
