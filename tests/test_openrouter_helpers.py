"""Tests for pure functions in app.providers.openrouter — _build_payload, _parse, _extract_json."""

import json

import pytest

from app.providers.openrouter import (
    _build_payload,
    _extract_json,
    _is_retryable,
    _parse,
)


class TestBuildPayload:
    def test_structure(self):
        payload = _build_payload(["Hello"], "en", "kh")
        assert payload["model"] == "openrouter/free"
        assert payload["temperature"] == 0.0
        assert payload["response_format"] == {"type": "json_object"}
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    def test_system_message_content(self):
        payload = _build_payload(["Hello"], "en", "kh")
        assert "Khmer" in payload["messages"][0]["content"]

    def test_user_message_content(self):
        payload = _build_payload(["A", "B"], "en", "kh")
        user_msg = payload["messages"][1]["content"]
        assert "1. A" in user_msg
        assert "2. B" in user_msg


class TestExtractJson:
    def test_direct_json(self):
        data = json.dumps({"translate_text": ["ok"]})
        assert _extract_json(data) == {"translate_text": ["ok"]}

    def test_markdown_code_block(self):
        content = '```json\n{"translate_text": ["ok"]}\n```'
        assert _extract_json(content) == {"translate_text": ["ok"]}

    def test_brace_span_fallback(self):
        content = 'Some text before {"translate_text": ["ok"]} and after'
        assert _extract_json(content) == {"translate_text": ["ok"]}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="Cannot extract JSON"):
            _extract_json("no json here at all")

    def test_whitespace_handling(self):
        data = '  \n  {"translate_text": ["ok"]}  \n  '
        assert _extract_json(data) == {"translate_text": ["ok"]}


class TestParse:
    def _wrap(self, translate_text: list[str], finish_reason: str = "stop"):
        return {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "content": json.dumps({"translate_text": translate_text})
                    },
                }
            ]
        }

    def test_valid_response(self):
        result = _parse(self._wrap(["translated"]), expected=1)
        assert result == ["translated"]

    def test_count_mismatch(self):
        with pytest.raises(ValueError, match="Count mismatch"):
            _parse(self._wrap(["a", "b"]), expected=1)

    def test_early_stop(self):
        with pytest.raises(ValueError, match="stopped early"):
            _parse(self._wrap(["x"], finish_reason="length"), expected=1)

    def test_end_turn_accepted(self):
        result = _parse(self._wrap(["ok"], finish_reason="end_turn"), expected=1)
        assert result == ["ok"]

    def test_none_finish_reason_accepted(self):
        result = _parse(self._wrap(["ok"], finish_reason=None), expected=1)
        assert result == ["ok"]

    def test_missing_translate_text_key(self):
        data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"wrong_key": ["x"]})},
                }
            ]
        }
        with pytest.raises(ValueError, match="Missing 'translate_text'"):
            _parse(data, expected=1)

    def test_non_list_translate_text(self):
        data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"translate_text": "not a list"})},
                }
            ]
        }
        with pytest.raises(ValueError, match="must be an array"):
            _parse(data, expected=1)

    def test_non_string_items(self):
        data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"translate_text": [123]})},
                }
            ]
        }
        with pytest.raises(ValueError, match="must all be strings"):
            _parse(data, expected=1)

    def test_missing_choices(self):
        with pytest.raises(ValueError, match="Unexpected response"):
            _parse({}, expected=1)

    def test_missing_message_content(self):
        data = {"choices": [{"finish_reason": "stop", "message": {}}]}
        with pytest.raises(ValueError, match="Missing message.content"):
            _parse(data, expected=1)
