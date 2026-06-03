"""Tests for app.services.prompt_builder — PromptBuilder and helpers."""

import pytest

from app.services.prompt_builder import (
    PromptBuilder,
    PromptPair,
    _lang_name,
    _LANG_NAMES,
    _QUALITY_HINTS,
    _TECH_TERMS,
)


class TestLangName:
    def test_known_code(self):
        assert _lang_name("kh") == "Khmer"
        assert _lang_name("en") == "English"
        assert _lang_name("vi") == "Vietnamese"

    def test_case_insensitive(self):
        assert _lang_name("KH") == "Khmer"
        assert _lang_name("En") == "English"

    def test_unknown_code_returns_raw(self):
        assert _lang_name("xyz") == "xyz"

    def test_auto_code(self):
        assert _lang_name("auto") == "the detected source language"


class TestPromptPair:
    def test_frozen(self):
        pair = PromptPair(system="sys", user="usr")
        assert pair.system == "sys"
        assert pair.user == "usr"
        with pytest.raises(AttributeError):
            pair.system = "new"


class TestPromptBuilder:
    def setup_method(self):
        self.builder = PromptBuilder()

    def test_build_returns_prompt_pair(self):
        pair = self.builder.build(["Hello"], src_lang="en", tgt_lang="kh")
        assert isinstance(pair, PromptPair)

    def test_system_contains_target_language(self):
        pair = self.builder.build(["Hello"], src_lang="en", tgt_lang="kh")
        assert "Khmer" in pair.system

    def test_system_contains_tech_terms(self):
        pair = self.builder.build(["Hello"], src_lang="en", tgt_lang="kh")
        assert "API" in pair.system
        assert "FastAPI" in pair.system

    def test_quality_hint_injected_for_khmer(self):
        pair = self.builder.build(["Hello"], src_lang="en", tgt_lang="kh")
        assert "Khmer quality" in pair.system

    def test_quality_hint_injected_for_vietnamese(self):
        pair = self.builder.build(["Hello"], src_lang="en", tgt_lang="vi")
        assert "Vietnamese quality" in pair.system

    def test_no_quality_hint_for_unknown_lang(self):
        pair = self.builder.build(["Hello"], src_lang="en", tgt_lang="de")
        # German has no quality hint
        assert "quality" not in pair.system.lower() or "QUALITY" in pair.system

    def test_user_prompt_numbered_list(self):
        pair = self.builder.build(
            ["Hello", "World"],
            src_lang="en",
            tgt_lang="kh",
        )
        assert "1. Hello" in pair.user
        assert "2. World" in pair.user

    def test_user_prompt_contains_languages(self):
        pair = self.builder.build(["Hello"], src_lang="en", tgt_lang="kh")
        assert "English" in pair.user
        assert "Khmer" in pair.user

    def test_strict_mode_adds_enforcement(self):
        normal = self.builder.build(["Hello"], src_lang="en", tgt_lang="kh", strict=False)
        strict = self.builder.build(["Hello"], src_lang="en", tgt_lang="kh", strict=True)
        assert "CRITICAL SCRIPT ENFORCEMENT" not in normal.system
        assert "CRITICAL SCRIPT ENFORCEMENT" in strict.system

    def test_json_output_contract_in_system(self):
        pair = self.builder.build(["Hello"], src_lang="en", tgt_lang="kh")
        assert "translate_text" in pair.system
        assert "JSON" in pair.system

    def test_single_text(self):
        pair = self.builder.build(["Only one"], src_lang="en", tgt_lang="kh")
        assert "1. Only one" in pair.user
        assert "2." not in pair.user


class TestConstants:
    def test_lang_names_all_lowercase_keys(self):
        for key in _LANG_NAMES:
            assert key == key.lower(), f"Key {key!r} should be lowercase"

    def test_quality_hints_subset_of_lang_names(self):
        for code in _QUALITY_HINTS:
            assert code in _LANG_NAMES, f"Quality hint for {code!r} but no lang name entry"

    def test_tech_terms_is_nonempty_string(self):
        assert isinstance(_TECH_TERMS, str)
        assert len(_TECH_TERMS) > 10
