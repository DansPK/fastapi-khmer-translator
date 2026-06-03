"""Tests for app.schemas — Pydantic request/response models."""

import pytest
from pydantic import ValidationError

from app.schemas.health import HealthResponse, ServiceStatus
from app.schemas.translate import TranslateRequest, TranslateResponse


class TestTranslateRequest:
    def test_valid_request(self):
        req = TranslateRequest(
            input_text=["Hello"],
            src_lang="en",
            tgt_lang="kh",
        )
        assert req.input_text == ["Hello"]
        assert req.src_lang == "en"
        assert req.tgt_lang == "kh"

    def test_multiple_texts(self):
        req = TranslateRequest(
            input_text=["A", "B", "C"],
            src_lang="en",
            tgt_lang="kh",
        )
        assert len(req.input_text) == 3

    def test_empty_input_text_rejected(self):
        with pytest.raises(ValidationError):
            TranslateRequest(input_text=[], src_lang="en", tgt_lang="kh")

    def test_too_many_texts_rejected(self):
        with pytest.raises(ValidationError):
            TranslateRequest(
                input_text=[f"text_{i}" for i in range(11)],
                src_lang="en",
                tgt_lang="kh",
            )

    def test_max_texts_accepted(self):
        req = TranslateRequest(
            input_text=[f"text_{i}" for i in range(10)],
            src_lang="en",
            tgt_lang="kh",
        )
        assert len(req.input_text) == 10

    def test_missing_fields_rejected(self):
        with pytest.raises(ValidationError):
            TranslateRequest(input_text=["Hello"])


class TestTranslateResponse:
    def test_valid_response(self):
        resp = TranslateResponse(translate_text=["សួស្តី"])
        assert resp.translate_text == ["សួស្តី"]


class TestServiceStatus:
    def test_with_detail(self):
        s = ServiceStatus(status="up", detail="stub")
        assert s.status == "up"
        assert s.detail == "stub"

    def test_without_detail(self):
        s = ServiceStatus(status="down")
        assert s.detail is None


class TestHealthResponse:
    def test_valid(self):
        resp = HealthResponse(
            status="healthy",
            version="1.0.0",
            environment="development",
            services={
                "redis": ServiceStatus(status="up"),
                "provider": ServiceStatus(status="up", detail="stub"),
            },
        )
        assert resp.status == "healthy"
        assert resp.services["redis"].status == "up"
