from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.llm import azure_openai as llm_azure
from tests.conftest import SAMPLE_PNG_B64


AZURE_ENV = {
    "AZURE_OPENAI_API_KEY": "test-key-123",
    "AZURE_OPENAI_ENDPOINT": "https://lechler-ai-openai.openai.azure.com",
    "AZURE_OPENAI_API_VERSION": "2025-04-01-preview",
    "AZURE_OPENAI_DEPLOYMENT_IMAGE": "gpt-image-1",
}


def _image_response(b64: str) -> dict:
    return {"data": [{"b64_json": b64}]}


def _mock_post(status_code: int = 200, json_data: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}

    client = AsyncMock()
    client.post.return_value = resp
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client, resp


# --- generate_image ---

class TestAzureGenerateImage:
    @pytest.mark.asyncio
    async def test_success_no_references(self):
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", AZURE_ENV):
            result = await llm_azure.generate_image("a cat", "1024x1024", "auto", [])
        assert result.startswith("data:image/png;base64,")
        url = client.post.call_args[0][0]
        assert "generations" in url
        assert "gpt-image-1" in url

    @pytest.mark.asyncio
    async def test_success_with_references(self):
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        refs = [("ref.png", b"imgdata", "image/png")]
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", AZURE_ENV):
            result = await llm_azure.generate_image("a cat", "1024x1024", "auto", refs)
        assert result.startswith("data:image/png;base64,")
        url = client.post.call_args[0][0]
        assert "edits" in url

    @pytest.mark.asyncio
    async def test_uses_azure_endpoint_and_deployment(self):
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", AZURE_ENV):
            await llm_azure.generate_image("a cat", "1024x1024", "medium", [])
        url = client.post.call_args[0][0]
        assert "lechler-ai-openai.openai.azure.com" in url
        assert "gpt-image-1" in url
        assert "api-version=2025-04-01-preview" in url

    @pytest.mark.asyncio
    async def test_uses_api_key_header(self):
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", AZURE_ENV):
            await llm_azure.generate_image("a cat", "1024x1024", "medium", [])
        headers = client.post.call_args[1]["headers"]
        assert headers["api-key"] == "test-key-123"

    @pytest.mark.asyncio
    async def test_svg_sources_appended_to_prompt(self):
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", AZURE_ENV):
            await llm_azure.generate_image("a cat", "1024x1024", "medium", [], svg_sources=["<svg/>"])
        call_json = client.post.call_args[1]["json"]
        assert "Reference SVG 1" in call_json["prompt"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("quality", ["low", "medium", "high", "auto"])
    async def test_quality_passed_through(self, quality):
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", AZURE_ENV):
            result = await llm_azure.generate_image("a cat", "1024x1024", quality, [])
        assert result.startswith("data:image/png;base64,")
        call_json = client.post.call_args[1]["json"]
        assert call_json["quality"] == quality

    @pytest.mark.asyncio
    @pytest.mark.parametrize("size", ["1024x1024", "1536x1024", "1024x1536"])
    async def test_size_passed_through(self, size):
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", AZURE_ENV):
            await llm_azure.generate_image("a cat", size, "medium", [])
        call_json = client.post.call_args[1]["json"]
        assert call_json["size"] == size

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        client, _ = _mock_post(500, {"error": {"message": "Server error", "code": ""}})
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", AZURE_ENV):
            with pytest.raises(HTTPException) as exc_info:
                await llm_azure.generate_image("a cat", "1024x1024", "medium", [])
            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_safety_filter_raises_422(self):
        client, _ = _mock_post(400, {"error": {"message": "content safety policy", "code": "moderation"}})
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", AZURE_ENV):
            with pytest.raises(HTTPException) as exc_info:
                await llm_azure.generate_image("something bad", "1024x1024", "medium", [])
            assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_no_image_payload_raises(self):
        client, _ = _mock_post(200, {"data": [{}]})
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", AZURE_ENV):
            with pytest.raises(HTTPException) as exc_info:
                await llm_azure.generate_image("a cat", "1024x1024", "medium", [])
            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                await llm_azure.generate_image("a cat", "1024x1024", "medium", [])
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_endpoint(self):
        env = {**AZURE_ENV}
        del env["AZURE_OPENAI_ENDPOINT"]
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                await llm_azure.generate_image("a cat", "1024x1024", "medium", [])
            assert exc_info.value.status_code == 400
            assert "AZURE_OPENAI_ENDPOINT" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_missing_deployment(self):
        env = {**AZURE_ENV}
        del env["AZURE_OPENAI_DEPLOYMENT_IMAGE"]
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                await llm_azure.generate_image("a cat", "1024x1024", "medium", [])
            assert exc_info.value.status_code == 400
            assert "AZURE_OPENAI_DEPLOYMENT_IMAGE" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_endpoint_trailing_slash_stripped(self):
        env = {**AZURE_ENV, "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/"}
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        with patch("app.llm.azure_openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", env):
            await llm_azure.generate_image("a cat", "1024x1024", "medium", [])
        url = client.post.call_args[0][0]
        assert "//openai" not in url
