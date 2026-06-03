"""Tests for pure functions in app.providers.gemini — _build_payload, _parse, _is_retryable."""

import json

import httpx
import pytest

from app.providers.gemini import _build_payload, _is_retryable, _parse


class TestBuildPayload:
    def test_structure(self):
        payload = _build_payload(["Hello"], "en", "kh")
        assert "system_instruction" in payload
        assert "contents" in payload
        assert "generationConfig" in payload
        assert payload["generationConfig"]["temperature"] == 0.0
        assert payload["generationConfig"]["responseMimeType"] == "application/json"

    def test_system_instruction_contains_text(self):
        payload = _build_payload(["Hello"], "en", "kh")
        system_text = payload["system_instruction"]["parts"][0]["text"]
        assert "Khmer" in system_text

    def test_contents_contain_user_text(self):
        payload = _build_payload(["Hello", "World"], "en", "kh")
        user_text = payload["contents"][0]["parts"][0]["text"]
        assert "1. Hello" in user_text
        assert "2. World" in user_text

    def test_response_schema(self):
        payload = _build_payload(["Hello"], "en", "kh")
        schema = payload["generationConfig"]["responseSchema"]
        assert schema["type"] == "OBJECT"
        assert "translate_text" in schema["properties"]


class TestParse:
    def _wrap(self, translate_text: list[str], finish_reason: str = "STOP"):
        return {
            "candidates": [
                {
                    "finishReason": finish_reason,
                    "content": {
                        "parts": [
                            {"text": json.dumps({"translate_text": translate_text})}
                        ]
                    },
                }
            ]
        }

    def test_valid_response(self):
        data = self._wrap(["សួស្តី"])
        result = _parse(data, expected=1)
        assert result == ["សួស្តី"]

    def test_multiple_items(self):
        data = self._wrap(["a", "b", "c"])
        result = _parse(data, expected=3)
        assert result == ["a", "b", "c"]

    def test_count_mismatch_raises(self):
        data = self._wrap(["only_one"])
        with pytest.raises(ValueError, match="Count mismatch"):
            _parse(data, expected=2)

    def test_early_stop_raises(self):
        data = self._wrap(["x"], finish_reason="SAFETY")
        with pytest.raises(ValueError, match="stopped early"):
            _parse(data, expected=1)

    def test_end_of_turn_accepted(self):
        data = self._wrap(["ok"], finish_reason="END_OF_TURN")
        result = _parse(data, expected=1)
        assert result == ["ok"]

    def test_malformed_json_raises(self):
        data = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "not json"}]},
                }
            ]
        }
        with pytest.raises(ValueError, match="Unexpected Gemini response"):
            _parse(data, expected=1)

    def test_missing_candidates_raises(self):
        with pytest.raises((KeyError, IndexError)):
            _parse({}, expected=1)


class TestIsRetryable:
    def test_timeout_is_retryable(self):
        assert _is_retryable(httpx.ReadTimeout("timeout")) is True

    def test_500_is_retryable(self):
        resp = httpx.Response(500, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("err", request=resp.request, response=resp)
        assert _is_retryable(exc) is True

    def test_429_is_retryable(self):
        resp = httpx.Response(429, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("err", request=resp.request, response=resp)
        assert _is_retryable(exc) is True

    def test_400_not_retryable(self):
        resp = httpx.Response(400, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("err", request=resp.request, response=resp)
        assert _is_retryable(exc) is False

    def test_generic_exception_not_retryable(self):
        assert _is_retryable(RuntimeError("boom")) is False
