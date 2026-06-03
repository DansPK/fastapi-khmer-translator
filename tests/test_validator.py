"""Tests for app.services.validator — TranslationValidator and helpers."""

import pytest

from app.services.validator import (
    TranslationValidator,
    ValidationResult,
    _check_khmer,
    _has_translatable_content,
)


class TestHasTranslatableContent:
    """Unit tests for the _has_translatable_content helper."""

    def test_plain_english_sentence(self):
        assert _has_translatable_content("Login to your account") is True

    def test_only_technical_terms(self):
        # All tokens are in _TECH_STRIP → no translatable words remain
        assert _has_translatable_content("API SDK JWT") is False

    def test_technical_with_surrounding_words(self):
        assert _has_translatable_content("Connect to the Redis server") is True

    def test_short_words_only(self):
        # Words shorter than 3 chars don't count
        assert _has_translatable_content("go to it") is False

    def test_numbers_and_punctuation(self):
        assert _has_translatable_content("123 !!!") is False

    def test_empty_string(self):
        assert _has_translatable_content("") is False

    def test_version_string(self):
        assert _has_translatable_content("v1.0.3") is False

    def test_mixed_tech_and_natural(self):
        assert _has_translatable_content("Please configure your Docker container") is True


class TestCheckKhmer:
    """Unit tests for the _check_khmer validation function."""

    def test_valid_khmer_output(self):
        result = _check_khmer(0, "Hello", "សួស្តី")
        assert result.valid is True

    def test_cjk_contamination(self):
        result = _check_khmer(0, "Hello", "你好")
        assert result.valid is False
        assert "CJK" in result.reason

    def test_japanese_hiragana_contamination(self):
        result = _check_khmer(0, "Hello", "こんにちは")
        assert result.valid is False
        assert "CJK/Japanese/Korean" in result.reason

    def test_korean_hangul_contamination(self):
        result = _check_khmer(0, "Hello", "안녕하세요")
        assert result.valid is False
        assert "CJK/Japanese/Korean" in result.reason

    def test_no_khmer_chars_when_expected(self):
        result = _check_khmer(0, "Login to your account", "Login to your account")
        assert result.valid is False
        assert "no Khmer Unicode" in result.reason

    def test_pure_tech_term_passes(self):
        # Source is only tech terms → no Khmer expected
        result = _check_khmer(0, "API SDK", "API SDK")
        assert result.valid is True

    def test_empty_translation(self):
        # _check_khmer doesn't check empty — that's _validate_item's job
        # but khmer check itself would pass because there's no CJK and
        # source has no translatable content either if it's short
        result = _check_khmer(0, "ok", "ok")
        assert result.valid is True


class TestTranslationValidator:
    """Tests for the TranslationValidator.validate_batch method."""

    def setup_method(self):
        self.validator = TranslationValidator()

    def test_valid_batch(self):
        result = self.validator.validate_batch(
            ["Hello", "World"],
            ["សួស្តី", "ពិភពលោក"],
            "kh",
        )
        assert result.valid is True

    def test_count_mismatch(self):
        result = self.validator.validate_batch(
            ["Hello", "World"],
            ["សួស្តី"],
            "kh",
        )
        assert result.valid is False
        assert "count mismatch" in result.reason

    def test_empty_translation_in_batch(self):
        result = self.validator.validate_batch(
            ["Hello", "World"],
            ["សួស្តី", "  "],
            "kh",
        )
        assert result.valid is False
        assert "empty translation" in result.reason

    def test_non_khmer_target_passes_without_script_check(self):
        result = self.validator.validate_batch(
            ["Hello"],
            ["Bonjour"],
            "fr",
        )
        assert result.valid is True

    def test_khmer_aliases(self):
        """All Khmer codes (kh, khm, km) should trigger Khmer validation."""
        for code in ("kh", "khm", "km"):
            result = self.validator.validate_batch(
                ["Hello"], ["Hello"], code
            )
            assert result.valid is False, f"Should fail for code={code}"

    def test_first_failure_returned(self):
        result = self.validator.validate_batch(
            ["Hello", "World", "Test"],
            ["សួស្តី", "你好", "ពិភពលោក"],
            "kh",
        )
        assert result.valid is False
        assert "item[1]" in result.reason


class TestValidationResult:
    def test_ok_singleton(self):
        from app.services.validator import _OK
        assert _OK.valid is True
        assert _OK.reason is None

    def test_frozen(self):
        vr = ValidationResult(valid=True)
        with pytest.raises(AttributeError):
            vr.valid = False
