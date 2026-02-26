import base64
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.llm import google as llm_google
from tests.conftest import SAMPLE_PNG_B64


def _google_image_response(b64: str = SAMPLE_PNG_B64, mime: str = "image/png") -> dict:
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "inlineData": {"mimeType": mime, "data": b64},
                }]
            }
        }]
    }


def _mock_post(status_code: int = 200, json_data: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}

    client = AsyncMock()
    client.post.return_value = resp
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client, resp


class TestGenerateImage:
    @pytest.mark.asyncio
    async def test_success(self):
        client, _ = _mock_post(200, _google_image_response())
        with patch("app.llm.google.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"}):
            result = await llm_google.generate_image("a dog", "1024x1024", "standard", "1:1", [])
        assert result.startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_with_references(self):
        client, _ = _mock_post(200, _google_image_response())
        refs = [("ref.jpg", b"jpgdata", "image/jpeg")]
        with patch("app.llm.google.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"}):
            result = await llm_google.generate_image("a dog", "1024x1024", "standard", "1:1", refs)
        assert result.startswith("data:image/")
        call_json = client.post.call_args[1]["json"]
        parts = call_json["contents"][0]["parts"]
        assert len(parts) == 2  # text + inlineData

    @pytest.mark.asyncio
    async def test_svg_sources_in_prompt(self):
        client, _ = _mock_post(200, _google_image_response())
        with patch("app.llm.google.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"}):
            await llm_google.generate_image("a dog", "1024x1024", "hd", "1:1", [], svg_sources=["<svg/>"])
        call_json = client.post.call_args[1]["json"]
        text_part = call_json["contents"][0]["parts"][0]["text"]
        assert "Reference SVG 1" in text_part

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        client, _ = _mock_post(500, {"error": {"message": "Quota exceeded", "code": ""}})
        with patch("app.llm.google.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"}):
            with pytest.raises(HTTPException) as exc_info:
                await llm_google.generate_image("a dog", "1024x1024", "standard", "1:1", [])
            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_no_image_in_response_raises(self):
        client, _ = _mock_post(200, {"candidates": [{"content": {"parts": [{"text": "sorry"}]}}]})
        with patch("app.llm.google.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"}):
            with pytest.raises(HTTPException) as exc_info:
                await llm_google.generate_image("a dog", "1024x1024", "standard", "1:1", [])
            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_empty_candidates_raises(self):
        client, _ = _mock_post(200, {"candidates": []})
        with patch("app.llm.google.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"}):
            with pytest.raises(HTTPException):
                await llm_google.generate_image("a dog", "1024x1024", "standard", "1:1", [])

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                await llm_google.generate_image("a dog", "1024x1024", "standard", "1:1", [])
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_snake_case_inline_data_key(self):
        """Google sometimes uses snake_case keys in responses."""
        payload = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "inline_data": {"mime_type": "image/jpeg", "data": SAMPLE_PNG_B64},
                    }]
                }
            }]
        }
        client, _ = _mock_post(200, payload)
        with patch("app.llm.google.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"}):
            result = await llm_google.generate_image("a dog", "1024x1024", "standard", "1:1", [])
        assert result.startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_default_uses_flash_model(self):
        """Default generate_image uses the Flash model URL."""
        client, _ = _mock_post(200, _google_image_response())
        with patch("app.llm.google.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"}):
            await llm_google.generate_image("a dog", "1024x1024", "standard", "1:1", [])
        url = client.post.call_args[0][0]
        assert "gemini-3.1-flash-image-preview" in url

    @pytest.mark.asyncio
    async def test_pro_uses_pro_model(self):
        """generate_image_pro uses the Pro model URL."""
        client, _ = _mock_post(200, _google_image_response())
        with patch("app.llm.google.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"}):
            await llm_google.generate_image_pro("a dog", "1024x1024", "standard", "1:1", [])
        url = client.post.call_args[0][0]
        assert "gemini-3-pro-image-preview" in url
