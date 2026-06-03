"""Tests for middleware helpers — _client_ip extraction."""

from unittest.mock import MagicMock

import pytest

from app.middleware.rate_limit import _client_ip


def _make_request(headers=None, client_host="127.0.0.1"):
    """Build a minimal mock Request."""
    req = MagicMock()
    req.headers = headers or {}
    req.client = MagicMock()
    req.client.host = client_host
    return req


class TestClientIp:
    def test_x_real_ip_takes_priority(self):
        req = _make_request(
            headers={"X-Real-IP": "1.2.3.4", "X-Forwarded-For": "5.6.7.8"},
        )
        assert _client_ip(req) == "1.2.3.4"

    def test_x_forwarded_for_leftmost(self):
        req = _make_request(
            headers={"X-Forwarded-For": "10.0.0.1, 10.0.0.2, 10.0.0.3"},
        )
        assert _client_ip(req) == "10.0.0.1"

    def test_direct_connection(self):
        req = _make_request(client_host="192.168.1.1")
        assert _client_ip(req) == "192.168.1.1"

    def test_no_client_returns_unknown(self):
        req = MagicMock()
        req.headers = {}
        req.client = None
        assert _client_ip(req) == "unknown"

    def test_x_real_ip_stripped(self):
        req = _make_request(headers={"X-Real-IP": "  1.2.3.4  "})
        assert _client_ip(req) == "1.2.3.4"
