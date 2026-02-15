import base64
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.llm import anthropic as llm_anthropic
from tests.conftest import SAMPLE_SVG


def _anthropic_response(text: str) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 50, "output_tokens": 100},
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


class TestGenerateSvg:
    @pytest.mark.asyncio
    async def test_success(self):
        client, _ = _mock_post(200, _anthropic_response(SAMPLE_SVG))
        with patch("app.llm.anthropic.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ak-test"}):
            result = await llm_anthropic.generate_svg("a flower", "1024x1024", "medium", "1:1", [])
        assert result.startswith("data:image/svg+xml;base64,")
        decoded = base64.b64decode(result.split(",")[1]).decode()
        assert "<svg" in decoded

    @pytest.mark.asyncio
    async def test_max_tokens_fixed(self):
        client, _ = _mock_post(200, _anthropic_response(SAMPLE_SVG))
        with patch("app.llm.anthropic.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ak-test"}):
            await llm_anthropic.generate_svg("a flower", "1024x1024", "low", "1:1", [])
        call_json = client.post.call_args[1]["json"]
        assert call_json["max_tokens"] == 16000

    @pytest.mark.asyncio
    async def test_reference_images_filtered_by_mime(self):
        """Anthropic only accepts jpeg/png/gif/webp — others should be skipped."""
        client, _ = _mock_post(200, _anthropic_response(SAMPLE_SVG))
        refs = [
            ("photo.png", b"pngdata", "image/png"),
            ("doc.pdf", b"pdfdata", "application/pdf"),
            ("photo.jpg", b"jpgdata", "image/jpeg"),
        ]
        with patch("app.llm.anthropic.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ak-test"}):
            await llm_anthropic.generate_svg("a flower", "1024x1024", "medium", "1:1", refs)
        call_json = client.post.call_args[1]["json"]
        user_content = call_json["messages"][0]["content"]
        image_blocks = [b for b in user_content if b["type"] == "image"]
        assert len(image_blocks) == 2  # png and jpg, not pdf

    @pytest.mark.asyncio
    async def test_svg_sources_included(self):
        client, _ = _mock_post(200, _anthropic_response(SAMPLE_SVG))
        with patch("app.llm.anthropic.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ak-test"}):
            await llm_anthropic.generate_svg("a flower", "1024x1024", "medium", "1:1", [],
                                             svg_sources=["<svg>ref</svg>"])
        call_json = client.post.call_args[1]["json"]
        user_content = call_json["messages"][0]["content"]
        text_block = [b for b in user_content if b["type"] == "text"][0]
        assert "Reference SVG 1" in text_block["text"]

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        client, _ = _mock_post(400, {"error": {"message": "Invalid key", "code": ""}})
        with patch("app.llm.anthropic.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ak-test"}):
            with pytest.raises(HTTPException) as exc_info:
                await llm_anthropic.generate_svg("a flower", "1024x1024", "medium", "1:1", [])
            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_no_svg_in_response_raises(self):
        client, _ = _mock_post(200, _anthropic_response("I cannot generate images"))
        with patch("app.llm.anthropic.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ak-test"}):
            with pytest.raises(HTTPException) as exc_info:
                await llm_anthropic.generate_svg("a flower", "1024x1024", "medium", "1:1", [])
            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                await llm_anthropic.generate_svg("a flower", "1024x1024", "medium", "1:1", [])
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_uses_correct_model(self):
        client, _ = _mock_post(200, _anthropic_response(SAMPLE_SVG))
        with patch("app.llm.anthropic.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ak-test"}):
            await llm_anthropic.generate_svg("a flower", "1024x1024", "medium", "1:1", [])
        call_json = client.post.call_args[1]["json"]
        assert call_json["model"] == "claude-opus-4-6"

    @pytest.mark.asyncio
    async def test_uses_correct_headers(self):
        client, _ = _mock_post(200, _anthropic_response(SAMPLE_SVG))
        with patch("app.llm.anthropic.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ak-test"}):
            await llm_anthropic.generate_svg("a flower", "1024x1024", "medium", "1:1", [])
        call_headers = client.post.call_args[1]["headers"]
        assert call_headers["x-api-key"] == "ak-test"
        assert call_headers["anthropic-version"] == "2023-06-01"
