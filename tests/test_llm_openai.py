import base64
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.llm import openai as llm_openai
from tests.conftest import SAMPLE_SVG, SAMPLE_SVG_B64, SAMPLE_PNG_B64


def _chat_response(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 100},
    }


def _image_response(b64: str) -> dict:
    return {"data": [{"b64_json": b64}]}


def _mock_post(status_code: int = 200, json_data: dict | None = None, content: bytes = b""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.content = content

    client = AsyncMock()
    client.post.return_value = resp
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client, resp


# --- generate_image ---

class TestGenerateImage:
    @pytest.mark.asyncio
    async def test_success_no_references(self):
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = await llm_openai.generate_image("a cat", "1024x1024", "auto", [])
        assert result.startswith("data:image/png;base64,")
        call_args = client.post.call_args
        assert "generations" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_success_with_references(self):
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        refs = [("ref.png", b"imgdata", "image/png")]
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = await llm_openai.generate_image("a cat", "1024x1024", "auto", refs)
        assert result.startswith("data:image/png;base64,")
        call_args = client.post.call_args
        assert "edits" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_svg_sources_appended_to_prompt(self):
        client, _ = _mock_post(200, _image_response(SAMPLE_PNG_B64))
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            await llm_openai.generate_image("a cat", "1024x1024", "auto", [], svg_sources=["<svg/>"])
        call_json = client.post.call_args[1]["json"]
        assert "Reference SVG 1" in call_json["prompt"]

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        client, _ = _mock_post(500, {"error": {"message": "Server error", "code": ""}})
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with pytest.raises(HTTPException) as exc_info:
                await llm_openai.generate_image("a cat", "1024x1024", "auto", [])
            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_no_image_payload_raises(self):
        client, _ = _mock_post(200, {"data": [{}]})
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with pytest.raises(HTTPException) as exc_info:
                await llm_openai.generate_image("a cat", "1024x1024", "auto", [])
            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                await llm_openai.generate_image("a cat", "1024x1024", "auto", [])
            assert exc_info.value.status_code == 400


# --- generate_svg ---

class TestGenerateSvg:
    @pytest.mark.asyncio
    async def test_success(self):
        client, _ = _mock_post(200, _chat_response(SAMPLE_SVG))
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = await llm_openai.generate_svg("a star", "1024x1024", "medium", "1:1", [])
        assert result.startswith("data:image/svg+xml;base64,")
        decoded = base64.b64decode(result.split(",")[1]).decode()
        assert "<svg" in decoded

    @pytest.mark.asyncio
    async def test_max_tokens_fixed(self):
        client, _ = _mock_post(200, _chat_response(SAMPLE_SVG))
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            await llm_openai.generate_svg("a star", "1024x1024", "low", "1:1", [])
        call_json = client.post.call_args[1]["json"]
        assert call_json["max_completion_tokens"] == 16000

    @pytest.mark.asyncio
    async def test_reference_images_included(self):
        client, _ = _mock_post(200, _chat_response(SAMPLE_SVG))
        refs = [("ref.png", b"imgdata", "image/png")]
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            await llm_openai.generate_svg("a star", "1024x1024", "medium", "1:1", refs)
        call_json = client.post.call_args[1]["json"]
        user_msg = call_json["messages"][1]["content"]
        assert any(item["type"] == "image_url" for item in user_msg)

    @pytest.mark.asyncio
    async def test_svg_sources_included(self):
        client, _ = _mock_post(200, _chat_response(SAMPLE_SVG))
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            await llm_openai.generate_svg("a star", "1024x1024", "medium", "1:1", [], svg_sources=["<svg/>"])
        call_json = client.post.call_args[1]["json"]
        user_msg = call_json["messages"][1]["content"]
        text_block = [item for item in user_msg if item["type"] == "text"][0]
        assert "Reference SVG 1" in text_block["text"]

    @pytest.mark.asyncio
    async def test_no_svg_in_response_raises(self):
        client, _ = _mock_post(200, _chat_response("I can't do that"))
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with pytest.raises(HTTPException) as exc_info:
                await llm_openai.generate_svg("a star", "1024x1024", "medium", "1:1", [])
            assert exc_info.value.status_code == 502


# --- suggest_filename ---

class TestSuggestFilename:
    @pytest.mark.asyncio
    async def test_success(self):
        client, _ = _mock_post(200, _chat_response("golden-sunset-cat"))
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = await llm_openai.suggest_filename("a golden sunset with a cat", "fallback")
        assert result == "golden-sunset-cat"

    @pytest.mark.asyncio
    async def test_returns_fallback_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await llm_openai.suggest_filename("a cat", "my-fallback")
        assert result == "my-fallback"

    @pytest.mark.asyncio
    async def test_returns_fallback_on_api_error(self):
        client, _ = _mock_post(500, {})
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = await llm_openai.suggest_filename("a cat", "my-fallback")
        assert result == "my-fallback"

    @pytest.mark.asyncio
    async def test_returns_fallback_on_empty_content(self):
        client, _ = _mock_post(200, {"choices": [{"message": {"content": ""}}]})
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = await llm_openai.suggest_filename("a cat", "my-fallback")
        assert result == "my-fallback"


# --- describe_image ---

class TestDescribeImage:
    @pytest.mark.asyncio
    async def test_success(self):
        client, _ = _mock_post(200, _chat_response("A lovely cat painting!"))
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = await llm_openai.describe_image("data:image/png;base64,abc", "a cat", "en")
        assert result == "A lovely cat painting!"

    @pytest.mark.asyncio
    async def test_uses_language_instruction(self):
        client, _ = _mock_post(200, _chat_response("Ein schönes Bild!"))
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            await llm_openai.describe_image("data:image/png;base64,abc", "katze", "de")
        call_json = client.post.call_args[1]["json"]
        system_content = call_json["messages"][0]["content"]
        assert "de" in system_content

    @pytest.mark.asyncio
    async def test_empty_response_raises(self):
        client, _ = _mock_post(200, _chat_response(""))
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with pytest.raises(HTTPException) as exc_info:
                await llm_openai.describe_image("data:image/png;base64,abc", "a cat", "en")
            assert exc_info.value.status_code == 502


# --- transcribe_audio ---

class TestTranscribeAudio:
    @pytest.mark.asyncio
    async def test_success(self):
        client, _ = _mock_post(200, {"text": "Hello world", "duration": 3.5})
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = await llm_openai.transcribe_audio(b"audiodata", "voice.webm", "audio/webm")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_empty_text_raises(self):
        client, _ = _mock_post(200, {"text": "", "duration": 1.0})
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with pytest.raises(HTTPException) as exc_info:
                await llm_openai.transcribe_audio(b"audiodata", "voice.webm", "audio/webm")
            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        client, _ = _mock_post(400, {"error": {"message": "Bad audio", "code": ""}})
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with pytest.raises(HTTPException):
                await llm_openai.transcribe_audio(b"audiodata", "voice.webm", "audio/webm")


# --- synthesize_speech ---

class TestSynthesizeSpeech:
    @pytest.mark.asyncio
    async def test_success(self):
        client, _ = _mock_post(200, content=b"mp3audiobytes")
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = await llm_openai.synthesize_speech("Hello", "en")
        assert result == b"mp3audiobytes"

    @pytest.mark.asyncio
    async def test_empty_audio_raises(self):
        client, _ = _mock_post(200, content=b"")
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with pytest.raises(HTTPException) as exc_info:
                await llm_openai.synthesize_speech("Hello", "en")
            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        client, _ = _mock_post(500, {"error": {"message": "Overloaded", "code": ""}})
        with patch("app.llm.openai.httpx.AsyncClient", return_value=client), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with pytest.raises(HTTPException):
                await llm_openai.synthesize_speech("Hello", "en")
