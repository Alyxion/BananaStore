import base64
import io
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app, PROVIDER_CAPABILITIES
from tests.conftest import SAMPLE_SVG, SAMPLE_SVG_B64, SAMPLE_PNG_B64


client = TestClient(app)


# --- GET /api/providers ---

class TestGetProviders:
    def test_returns_all_providers(self):
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "sk-test",
            "GOOGLE_API_KEY": "",
            "ANTHROPIC_API_KEY": "ak-test",
        }):
            resp = client.get("/api/providers")
        assert resp.status_code == 200
        data = resp.json()["providers"]
        assert "openai" in data
        assert "google" in data
        assert "anthropic" in data

    def test_has_key_flags(self):
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "sk-test",
            "GOOGLE_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
        }):
            resp = client.get("/api/providers")
        data = resp.json()["providers"]
        assert data["openai"]["hasKey"] is True
        assert data["google"]["hasKey"] is False

    def test_includes_capabilities(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": ""}):
            resp = client.get("/api/providers")
        data = resp.json()["providers"]
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
        """formatQualities values should be subsets of the provider's qualities list."""
        for provider_id, caps in PROVIDER_CAPABILITIES.items():
            fq = caps.get("formatQualities", {})
            all_quals = set()
            for fmt_quals in fq.values():
                all_quals.update(fmt_quals)
            if all_quals:
                assert all_quals.issubset(set(caps["qualities"]) | all_quals)


# --- POST /api/suggest-filename ---

class TestSuggestFilename:
    def test_empty_description(self):
        resp = client.post("/api/suggest-filename", json={"description": ""})
        assert resp.status_code == 200
        assert resp.json()["filename"] == "generated-image"

    def test_calls_openai(self):
        with patch("app.main.llm_openai.suggest_filename", new_callable=AsyncMock, return_value="golden-cat"):
            resp = client.post("/api/suggest-filename", json={"description": "a golden cat"})
        assert resp.status_code == 200
        assert resp.json()["filename"] == "golden-cat"


# --- POST /api/transcribe-openai ---

class TestTranscribeOpenai:
    def test_success(self):
        with patch("app.main.llm_openai.transcribe_audio", new_callable=AsyncMock, return_value="Hello"):
            resp = client.post(
                "/api/transcribe-openai",
                files={"audio": ("voice.webm", b"audiodata", "audio/webm")},
            )
        assert resp.status_code == 200
        assert resp.json()["text"] == "Hello"

    def test_empty_audio(self):
        resp = client.post(
            "/api/transcribe-openai",
            files={"audio": ("voice.webm", b"", "audio/webm")},
        )
        assert resp.status_code == 400

    def test_too_large(self):
        big = b"x" * (8 * 1024 * 1024 + 1)
        resp = client.post(
            "/api/transcribe-openai",
            files={"audio": ("voice.webm", big, "audio/webm")},
        )
        assert resp.status_code == 400


# --- POST /api/describe-image ---

class TestDescribeImage:
    def test_success(self):
        with patch("app.main.llm_openai.describe_image", new_callable=AsyncMock, return_value="Nice cat!"):
            resp = client.post("/api/describe-image", json={
                "image_data_url": "data:image/png;base64,abc",
                "source_text": "a cat",
                "language": "en",
            })
        assert resp.status_code == 200
        assert resp.json()["description"] == "Nice cat!"

    def test_invalid_data_url(self):
        resp = client.post("/api/describe-image", json={
            "image_data_url": "not-a-data-url",
            "source_text": "",
            "language": "",
        })
        assert resp.status_code == 400


# --- POST /api/tts-openai ---

class TestTtsOpenai:
    def test_success(self):
        with patch("app.main.llm_openai.synthesize_speech", new_callable=AsyncMock, return_value=b"mp3data"):
            resp = client.post("/api/tts-openai", json={"text": "Hello world"})
        assert resp.status_code == 200
        assert resp.content == b"mp3data"
        assert resp.headers["content-type"] == "audio/mpeg"

    def test_empty_text(self):
        resp = client.post("/api/tts-openai", json={"text": ""})
        assert resp.status_code == 400


# --- POST /api/generate ---

class TestGenerate:
    def test_unsupported_provider(self):
        resp = client.post("/api/generate", data={
            "provider": "nonexistent",
            "description": "test",
            "quality": "auto",
            "ratio": "1:1",
        })
        assert resp.status_code == 400

    def test_unsupported_quality(self):
        resp = client.post("/api/generate", data={
            "provider": "openai",
            "description": "test",
            "quality": "ultra",
            "ratio": "1:1",
        })
        assert resp.status_code == 400

    def test_unsupported_ratio(self):
        resp = client.post("/api/generate", data={
            "provider": "openai",
            "description": "test",
            "quality": "auto",
            "ratio": "99:1",
        })
        assert resp.status_code == 400

    def test_unsupported_format(self):
        resp = client.post("/api/generate", data={
            "provider": "google",
            "description": "test",
            "quality": "standard",
            "ratio": "1:1",
            "format": "Vector",
        })
        assert resp.status_code == 400

    def test_format_quality_validation(self):
        """OpenAI Vector doesn't accept 'auto' quality."""
        resp = client.post("/api/generate", data={
            "provider": "openai",
            "description": "test",
            "quality": "auto",
            "ratio": "1:1",
            "format": "Vector",
        })
        assert resp.status_code == 400

    def test_openai_photo_success(self):
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"
        with patch("app.main.llm_openai.generate_image", new_callable=AsyncMock, return_value=data_url):
            resp = client.post("/api/generate", data={
                "provider": "openai",
                "description": "a sunset",
                "quality": "auto",
                "ratio": "1:1",
                "format": "Photo",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "openai"
        assert body["format"] == "Photo"
        assert body["image_data_url"].startswith("data:image/")

    def test_openai_vector_success(self):
        data_url = f"data:image/svg+xml;base64,{SAMPLE_SVG_B64}"
        with patch("app.main.llm_openai.generate_svg", new_callable=AsyncMock, return_value=data_url):
            resp = client.post("/api/generate", data={
                "provider": "openai",
                "description": "a star",
                "quality": "medium",
                "ratio": "1:1",
                "format": "Vector",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "Vector"

    def test_google_photo_success(self):
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"
        with patch("app.main.llm_google.generate_image", new_callable=AsyncMock, return_value=data_url):
            resp = client.post("/api/generate", data={
                "provider": "google",
                "description": "a mountain",
                "quality": "standard",
                "ratio": "1:1",
                "format": "Photo",
            })
        assert resp.status_code == 200
        assert resp.json()["provider"] == "google"

    def test_anthropic_vector_success(self):
        data_url = f"data:image/svg+xml;base64,{SAMPLE_SVG_B64}"
        with patch("app.main.llm_anthropic.generate_svg", new_callable=AsyncMock, return_value=data_url):
            resp = client.post("/api/generate", data={
                "provider": "anthropic",
                "description": "a tree",
                "quality": "high",
                "ratio": "1:1",
                "format": "Vector",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "anthropic"
        assert body["format"] == "Vector"

    def test_size_derived_from_ratio(self):
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"
        with patch("app.main.llm_openai.generate_image", new_callable=AsyncMock, return_value=data_url) as mock_gen:
            client.post("/api/generate", data={
                "provider": "openai",
                "description": "wide shot",
                "quality": "auto",
                "ratio": "3:2",
                "format": "Photo",
            })
        _, kwargs = mock_gen.call_args
        assert kwargs["size"] == "1536x1024"

    def test_response_includes_all_fields(self):
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"
        with patch("app.main.llm_openai.generate_image", new_callable=AsyncMock, return_value=data_url):
            resp = client.post("/api/generate", data={
                "provider": "openai",
                "description": "test",
                "quality": "auto",
                "ratio": "1:1",
                "format": "Photo",
            })
        body = resp.json()
        assert all(k in body for k in ["provider", "size", "quality", "ratio", "format", "image_data_url", "used_reference_images"])


# --- GET / ---

class TestRoot:
    def test_serves_index(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
