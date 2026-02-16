"""Tests for the WebSocket-based API (formerly REST routes).

These tests use the WebSocket endpoint to verify all actions work correctly,
including validation, provider dispatch, and response formats.
"""

import base64
from unittest.mock import patch, AsyncMock

import pytest
from starlette.testclient import TestClient

from app.main import app
from app.providers import PROVIDER_CAPABILITIES
from tests.conftest import SAMPLE_SVG_B64, SAMPLE_PNG_B64, ws_connect


client = TestClient(app)


def _ws_send(ws, action, payload=None):
    """Helper: send a WS action and return the response."""
    ws.send_json({"id": "t1", "action": action, "payload": payload or {}})
    return ws.receive_json()


# --- providers ---

class TestGetProviders:
    def test_returns_all_providers(self):
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "sk-test",
            "GOOGLE_API_KEY": "",
            "ANTHROPIC_API_KEY": "ak-test",
        }):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "providers")
        assert resp["ok"] is True
        data = resp["result"]["providers"]
        assert "openai" in data
        assert "google" in data
        assert "anthropic" in data

    def test_has_key_flags(self):
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "sk-test",
            "GOOGLE_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
        }):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "providers")
        data = resp["result"]["providers"]
        assert data["openai"]["hasKey"] is True
        assert data["google"]["hasKey"] is False

    def test_includes_capabilities(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": ""}):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "providers")
        data = resp["result"]["providers"]
        assert "qualities" in data["openai"]
        assert "ratios" in data["openai"]
        assert "formats" in data["openai"]


# --- PROVIDER_CAPABILITIES structure ---

class TestProviderCapabilities:
    def test_all_providers_have_required_keys(self):
        required = {"label", "qualities", "ratios", "ratioSizes", "formats", "requiresKey"}
        for provider_id, caps in PROVIDER_CAPABILITIES.items():
            assert required.issubset(caps.keys()), f"{provider_id} missing keys"

    def test_ratio_sizes_match_ratios(self):
        for provider_id, caps in PROVIDER_CAPABILITIES.items():
            for ratio in caps["ratios"]:
                assert ratio in caps["ratioSizes"], f"{provider_id}: ratio {ratio} missing from ratioSizes"

    def test_format_qualities_subset_of_qualities(self):
        for provider_id, caps in PROVIDER_CAPABILITIES.items():
            fq = caps.get("formatQualities", {})
            all_quals = set()
            for fmt_quals in fq.values():
                all_quals.update(fmt_quals)
            if all_quals:
                assert all_quals.issubset(set(caps["qualities"]) | all_quals)


# --- suggest-filename ---

class TestSuggestFilename:
    def test_empty_description(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "suggest-filename", {"description": ""})
        assert resp["ok"] is True
        assert resp["result"]["filename"] == "generated-image"

    def test_calls_openai(self):
        with patch("app.handlers.llm_openai.suggest_filename", new_callable=AsyncMock, return_value="golden-cat"):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "suggest-filename", {"description": "a golden cat"})
        assert resp["ok"] is True
        assert resp["result"]["filename"] == "golden-cat"


# --- transcribe ---

class TestTranscribe:
    def test_success(self):
        with patch("app.handlers.llm_openai.transcribe_audio", new_callable=AsyncMock, return_value="Hello"):
            audio_b64 = base64.b64encode(b"audiodata").decode()
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "transcribe", {
                    "audio_b64": audio_b64,
                    "filename": "voice.webm",
                    "content_type": "audio/webm",
                })
        assert resp["ok"] is True
        assert resp["result"]["text"] == "Hello"

    def test_empty_audio(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "transcribe", {"audio_b64": ""})
        assert resp["ok"] is False
        assert resp["code"] == 400

    def test_too_large(self):
        big_b64 = base64.b64encode(b"x" * (8 * 1024 * 1024 + 1)).decode()
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "transcribe", {"audio_b64": big_b64})
        assert resp["ok"] is False
        assert resp["code"] == 400


# --- describe-image ---

class TestDescribeImage:
    def test_success(self):
        with patch("app.handlers.llm_openai.describe_image", new_callable=AsyncMock, return_value="Nice cat!"):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "describe-image", {
                    "image_data_url": "data:image/png;base64,abc",
                    "source_text": "a cat",
                    "language": "en",
                })
        assert resp["ok"] is True
        assert resp["result"]["description"] == "Nice cat!"

    def test_invalid_data_url(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "describe-image", {
                "image_data_url": "not-a-data-url",
                "source_text": "",
                "language": "",
            })
        assert resp["ok"] is False
        assert resp["code"] == 400


# --- tts ---

class TestTts:
    def test_success(self):
        with patch("app.handlers.llm_openai.synthesize_speech", new_callable=AsyncMock, return_value=b"mp3data"):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "tts", {"text": "Hello world"})
        assert resp["ok"] is True
        assert base64.b64decode(resp["result"]["audio_b64"]) == b"mp3data"

    def test_empty_text(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "tts", {"text": ""})
        assert resp["ok"] is False
        assert resp["code"] == 400


# --- generate ---

class TestGenerate:
    def test_unsupported_provider(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "generate", {
                "provider": "nonexistent",
                "description": "test",
                "quality": "auto",
                "ratio": "1:1",
            })
        assert resp["ok"] is False
        assert resp["code"] == 400

    def test_unsupported_quality(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "generate", {
                "provider": "openai",
                "description": "test",
                "quality": "ultra",
                "ratio": "1:1",
            })
        assert resp["ok"] is False
        assert resp["code"] == 400

    def test_unsupported_ratio(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "generate", {
                "provider": "openai",
                "description": "test",
                "quality": "auto",
                "ratio": "99:1",
            })
        assert resp["ok"] is False
        assert resp["code"] == 400

    def test_unsupported_format(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "generate", {
                "provider": "google",
                "description": "test",
                "quality": "standard",
                "ratio": "1:1",
                "format": "Vector",
            })
        assert resp["ok"] is False
        assert resp["code"] == 400

    def test_format_quality_validation(self):
        """OpenAI Vector doesn't accept 'auto' quality."""
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "generate", {
                "provider": "openai",
                "description": "test",
                "quality": "auto",
                "ratio": "1:1",
                "format": "Vector",
            })
        assert resp["ok"] is False
        assert resp["code"] == 400

    def test_openai_photo_success(self):
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"
        with patch("app.handlers.llm_openai.generate_image", new_callable=AsyncMock, return_value=data_url):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "generate", {
                    "provider": "openai",
                    "description": "a sunset",
                    "quality": "auto",
                    "ratio": "1:1",
                    "format": "Photo",
                    "reference_images": [],
                })
        assert resp["ok"] is True
        body = resp["result"]
        assert body["provider"] == "openai"
        assert body["format"] == "Photo"
        assert body["image_data_url"].startswith("data:image/")

    def test_openai_vector_success(self):
        data_url = f"data:image/svg+xml;base64,{SAMPLE_SVG_B64}"
        with patch("app.handlers.llm_openai.generate_svg", new_callable=AsyncMock, return_value=data_url):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "generate", {
                    "provider": "openai",
                    "description": "a star",
                    "quality": "medium",
                    "ratio": "1:1",
                    "format": "Vector",
                    "reference_images": [],
                })
        assert resp["ok"] is True
        assert resp["result"]["format"] == "Vector"

    def test_google_photo_success(self):
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"
        with patch("app.handlers.llm_google.generate_image", new_callable=AsyncMock, return_value=data_url):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "generate", {
                    "provider": "google",
                    "description": "a mountain",
                    "quality": "standard",
                    "ratio": "1:1",
                    "format": "Photo",
                    "reference_images": [],
                })
        assert resp["ok"] is True
        assert resp["result"]["provider"] == "google"

    def test_anthropic_vector_success(self):
        data_url = f"data:image/svg+xml;base64,{SAMPLE_SVG_B64}"
        with patch("app.handlers.llm_anthropic.generate_svg", new_callable=AsyncMock, return_value=data_url):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "generate", {
                    "provider": "anthropic",
                    "description": "a tree",
                    "quality": "high",
                    "ratio": "1:1",
                    "format": "Vector",
                    "reference_images": [],
                })
        assert resp["ok"] is True
        body = resp["result"]
        assert body["provider"] == "anthropic"
        assert body["format"] == "Vector"

    def test_size_derived_from_ratio(self):
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"
        with patch("app.handlers.llm_openai.generate_image", new_callable=AsyncMock, return_value=data_url) as mock_gen:
            with ws_connect(client) as ws:
                _ws_send(ws, "generate", {
                    "provider": "openai",
                    "description": "wide shot",
                    "quality": "auto",
                    "ratio": "3:2",
                    "format": "Photo",
                    "reference_images": [],
                })
        _, kwargs = mock_gen.call_args
        assert kwargs["size"] == "1536x1024"

    def test_response_includes_all_fields(self):
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"
        with patch("app.handlers.llm_openai.generate_image", new_callable=AsyncMock, return_value=data_url):
            with ws_connect(client) as ws:
                resp = _ws_send(ws, "generate", {
                    "provider": "openai",
                    "description": "test",
                    "quality": "auto",
                    "ratio": "1:1",
                    "format": "Photo",
                    "reference_images": [],
                })
        body = resp["result"]
        assert all(k in body for k in ["provider", "size", "quality", "ratio", "format", "image_data_url", "used_reference_images"])


# --- GET / ---

class TestRoot:
    def test_serves_index_with_token(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert 'bs-token' in resp.text
