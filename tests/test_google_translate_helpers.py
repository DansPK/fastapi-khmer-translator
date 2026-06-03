"""Tests for pure functions in app.providers.google_translate."""

import httpx
import pytest

from app.providers.google_translate import _is_retryable, _to_google_lang


class TestToGoogleLang:
    def test_khmer_codes(self):
        assert _to_google_lang("kh") == "km"
        assert _to_google_lang("khm") == "km"
        assert _to_google_lang("km") == "km"

    def test_english_codes(self):
        assert _to_google_lang("eng") == "en"
        assert _to_google_lang("en") == "en"

    def test_vietnamese(self):
        assert _to_google_lang("viet") == "vi"
        assert _to_google_lang("vi") == "vi"

    def test_case_insensitive(self):
        assert _to_google_lang("KH") == "km"
        assert _to_google_lang("ENG") == "en"

    def test_auto_returns_none(self):
        assert _to_google_lang("auto") is None
        assert _to_google_lang("AUTO") is None

    def test_unknown_code_passthrough(self):
        assert _to_google_lang("xyz") == "xyz"

    def test_chinese_variants(self):
        assert _to_google_lang("zh") == "zh"
        assert _to_google_lang("zh-cn") == "zh-CN"
        assert _to_google_lang("zh-tw") == "zh-TW"


class TestIsRetryable:
    def test_timeout_retryable(self):
        assert _is_retryable(httpx.ConnectTimeout("timeout")) is True

    def test_502_retryable(self):
        resp = httpx.Response(502, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("err", request=resp.request, response=resp)
        assert _is_retryable(exc) is True

    def test_401_not_retryable(self):
        resp = httpx.Response(401, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("err", request=resp.request, response=resp)
        assert _is_retryable(exc) is False

    def test_generic_not_retryable(self):
        assert _is_retryable(ValueError("bad")) is False
