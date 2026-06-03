"""Tests for app.core.dependencies — auth and DI helpers."""

import secrets

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from app.config import Settings
from app.core.dependencies import verify_credentials


class TestVerifyCredentials:
    def _settings(self, user="admin", password="secret"):
        return Settings(
            api_username=user,
            api_password=password,
            _env_file=None,
        )

    def test_valid_credentials(self):
        creds = HTTPBasicCredentials(username="admin", password="secret")
        # Should not raise
        verify_credentials(creds, self._settings())

    def test_invalid_username(self):
        creds = HTTPBasicCredentials(username="wrong", password="secret")
        with pytest.raises(HTTPException) as exc_info:
            verify_credentials(creds, self._settings())
        assert exc_info.value.status_code == 401

    def test_invalid_password(self):
        creds = HTTPBasicCredentials(username="admin", password="wrong")
        with pytest.raises(HTTPException) as exc_info:
            verify_credentials(creds, self._settings())
        assert exc_info.value.status_code == 401

    def test_both_invalid(self):
        creds = HTTPBasicCredentials(username="wrong", password="wrong")
        with pytest.raises(HTTPException) as exc_info:
            verify_credentials(creds, self._settings())
        assert exc_info.value.status_code == 401

    def test_www_authenticate_header(self):
        creds = HTTPBasicCredentials(username="wrong", password="wrong")
        with pytest.raises(HTTPException) as exc_info:
            verify_credentials(creds, self._settings())
        assert exc_info.value.headers["WWW-Authenticate"] == "Basic"

    def test_empty_credentials_rejected(self):
        creds = HTTPBasicCredentials(username="", password="")
        settings = self._settings(user="admin", password="secret")
        with pytest.raises(HTTPException):
            verify_credentials(creds, settings)
