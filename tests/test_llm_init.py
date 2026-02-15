import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from app.llm import ensure_api_key, to_data_url, safe_provider_error


# --- ensure_api_key ---

class TestEnsureApiKey:
    def test_returns_key_when_set(self):
        with patch.dict("os.environ", {"MY_KEY": "sk-123"}):
            assert ensure_api_key("MY_KEY", "TestProvider") == "sk-123"

    def test_raises_when_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                ensure_api_key("MISSING_KEY", "TestProvider")
            assert exc_info.value.status_code == 400
            assert "TestProvider" in exc_info.value.detail

    def test_raises_when_empty(self):
        with patch.dict("os.environ", {"EMPTY_KEY": ""}):
            with pytest.raises(HTTPException):
                ensure_api_key("EMPTY_KEY", "TestProvider")


# --- to_data_url ---

class TestToDataUrl:
    def test_default_mime(self):
        result = to_data_url("abc123")
        assert result == "data:image/png;base64,abc123"

    def test_custom_mime(self):
        result = to_data_url("abc123", "image/jpeg")
        assert result == "data:image/jpeg;base64,abc123"

    def test_svg_mime(self):
        result = to_data_url("svgdata", "image/svg+xml")
        assert result == "data:image/svg+xml;base64,svgdata"


# --- safe_provider_error ---

class TestSafeProviderError:
    def _mock_response(self, status_code, json_data=None, json_raises=False):
        resp = MagicMock()
        resp.status_code = status_code
        if json_raises:
            resp.json.side_effect = ValueError("not json")
        else:
            resp.json.return_value = json_data or {}
        return resp

    def test_non_json_response(self):
        resp = self._mock_response(500, json_raises=True)
        exc = safe_provider_error("OpenAI", resp)
        assert exc.status_code == 502
        assert "unexpected error" in exc.detail

    def test_moderation_code(self):
        resp = self._mock_response(400, {"error": {"message": "blocked", "code": "moderation_blocked"}})
        exc = safe_provider_error("OpenAI", resp)
        assert exc.status_code == 422
        assert "safety filter" in exc.detail

    def test_safety_in_message(self):
        resp = self._mock_response(400, {"error": {"message": "Content Safety violation", "code": ""}})
        exc = safe_provider_error("Google", resp)
        assert exc.status_code == 422

    def test_generic_error_message(self):
        resp = self._mock_response(500, {"error": {"message": "Rate limit exceeded", "code": "rate_limit"}})
        exc = safe_provider_error("OpenAI", resp)
        assert exc.status_code == 502
        assert "Rate limit exceeded" in exc.detail

    def test_no_error_object(self):
        resp = self._mock_response(500, {"something": "else"})
        exc = safe_provider_error("Google", resp)
        assert exc.status_code == 502
        assert "500" in exc.detail

    def test_error_not_dict(self):
        resp = self._mock_response(500, {"error": "string error"})
        exc = safe_provider_error("OpenAI", resp)
        assert exc.status_code == 502

    def test_provider_name_in_detail(self):
        resp = self._mock_response(500, {"error": {"message": "oops", "code": ""}})
        exc = safe_provider_error("Anthropic", resp)
        assert "Anthropic" in exc.detail
