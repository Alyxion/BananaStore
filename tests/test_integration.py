"""Integration tests that hit real APIs. Skipped unless API keys are set."""

import base64
import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app.main import app
from tests.conftest import ws_connect


client = TestClient(app)

FIXTURES = Path(__file__).parent / "fixtures"
HAVE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
HAVE_GOOGLE = bool(os.getenv("GOOGLE_API_KEY"))
HAVE_ANTHROPIC = bool(os.getenv("ANTHROPIC_API_KEY"))


def _ws_send(ws, action, payload=None, timeout=30):
    ws.send_json({"id": "t1", "action": action, "payload": payload or {}})
    return ws.receive_json()


@pytest.mark.skipif(not HAVE_OPENAI, reason="OPENAI_API_KEY not set")
class TestOpenaiIntegration:
    def test_suggest_filename_records_cost(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "suggest-filename", {"description": "a cute puppy on a hill"})
            assert resp["ok"] is True
            costs = _ws_send(ws, "costs")
            assert costs["result"]["total_usd"] > 0
            assert "prompt" in costs["result"]["by_category"]

    def test_generate_image_low_1024(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "generate", {
                "provider": "openai",
                "description": "a single yellow banana on white background",
                "quality": "low",
                "ratio": "1:1",
                "format": "Photo",
                "reference_images": [],
            })
        assert resp["ok"] is True
        assert resp["result"]["cost_usd"] is not None
        assert abs(resp["result"]["cost_usd"] - 0.011) < 0.001

    def test_generate_svg_records_cost(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "generate", {
                "provider": "openai",
                "description": "simple circle icon",
                "quality": "low",
                "ratio": "1:1",
                "format": "Vector",
                "reference_images": [],
            })
            assert resp["ok"] is True
            costs = _ws_send(ws, "costs")
            assert costs["result"]["total_usd"] > 0

    def test_synthesize_speech_records_cost(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "tts", {"text": "Hi"})
            assert resp["ok"] is True
            costs = _ws_send(ws, "costs")
            assert costs["result"]["total_usd"] > 0
            assert "voice_output" in costs["result"]["by_category"]

    def test_transcribe_records_cost(self):
        with ws_connect(client) as ws:
            # Generate a tiny audio clip first
            tts_resp = _ws_send(ws, "tts", {"text": "Hello"})
            assert tts_resp["ok"] is True
            audio_b64 = tts_resp["result"]["audio_b64"]

            resp = _ws_send(ws, "transcribe", {
                "audio_b64": audio_b64,
                "filename": "test.mp3",
                "content_type": "audio/mpeg",
            })
            assert resp["ok"] is True
            costs = _ws_send(ws, "costs")
            assert costs["result"]["total_usd"] > 0


    def test_transcribe_german_voice_fixture(self):
        audio_path = FIXTURES / "voice-test-german.webm"
        if not audio_path.exists():
            pytest.skip("voice-test-german.webm fixture not found")
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "transcribe", {
                "audio_b64": audio_b64,
                "filename": "voice-test-german.webm",
                "content_type": "audio/webm",
            }, timeout=30)
        assert resp["ok"] is True
        text = resp["result"]["text"].lower()
        assert "geil" in text or "scheiß" in text or "scheiss" in text

    def test_tts_then_stt_roundtrip(self):
        """Text → Speech → Text: the transcript should resemble the original."""
        original = "Das ist ein einfacher Test für die Sprachsynthese."
        with ws_connect(client) as ws:
            tts_resp = _ws_send(ws, "tts", {"text": original, "language": "de"}, timeout=30)
            assert tts_resp["ok"] is True
            audio_b64 = tts_resp["result"]["audio_b64"]

            stt_resp = _ws_send(ws, "transcribe", {
                "audio_b64": audio_b64,
                "filename": "roundtrip.mp3",
                "content_type": "audio/mpeg",
            }, timeout=30)
            assert stt_resp["ok"] is True
            transcript = stt_resp["result"]["text"].lower()
            # Key words should survive the round-trip
            assert "test" in transcript
            assert "sprachsynthese" in transcript or "sprach" in transcript


@pytest.mark.skipif(not HAVE_GOOGLE, reason="GOOGLE_API_KEY not set")
class TestGoogleIntegration:
    def test_generate_image_records_cost(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "generate", {
                "provider": "google",
                "description": "a single yellow banana on white background",
                "quality": "standard",
                "ratio": "1:1",
                "format": "Photo",
                "reference_images": [],
            })
            assert resp["ok"] is True
            costs = _ws_send(ws, "costs")
            assert abs(costs["result"]["total_usd"] - 0.04) < 0.001


@pytest.mark.skipif(not HAVE_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
class TestAnthropicIntegration:
    def test_generate_svg_records_cost(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "generate", {
                "provider": "anthropic",
                "description": "simple red square icon",
                "quality": "low",
                "ratio": "1:1",
                "format": "Vector",
                "reference_images": [],
            })
            assert resp["ok"] is True
            costs = _ws_send(ws, "costs")
            assert costs["result"]["total_usd"] > 0


@pytest.mark.skipif(not HAVE_OPENAI, reason="OPENAI_API_KEY not set")
class TestSpendingLimitIntegration:
    def test_limit_enforcement_returns_429(self):
        with ws_connect(client) as ws:
            # Set a very low limit
            resp = _ws_send(ws, "costs-limit", {"limit_usd": 0.001})
            assert resp["ok"] is True

            # First cheap call to consume the limit
            _ws_send(ws, "suggest-filename", {"description": "a dog"})

            # Second call should hit the limit
            resp = _ws_send(ws, "suggest-filename", {"description": "another dog"})
            assert resp["ok"] is False
            assert resp["code"] == 429


# --- Cost endpoint tests (no real API keys needed) ---

class TestCostEndpoints:
    def test_get_costs_empty(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "costs")
        assert resp["ok"] is True
        assert resp["result"]["total_usd"] == 0.0
        assert resp["result"]["entry_count"] == 0

    def test_get_costs_history_empty(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "costs-history")
        assert resp["ok"] is True
        assert resp["result"] == []

    def test_set_and_get_limit(self):
        with ws_connect(client) as ws:
            resp = _ws_send(ws, "costs-limit", {"limit_usd": 5.0})
            assert resp["ok"] is True
            assert resp["result"]["limit_usd"] == 5.0

            resp = _ws_send(ws, "costs")
            assert resp["result"]["limit_usd"] == 5.0

    def test_remove_limit(self):
        with ws_connect(client) as ws:
            _ws_send(ws, "costs-limit", {"limit_usd": 10.0})
            resp = _ws_send(ws, "costs-limit", {"limit_usd": None})
            assert resp["ok"] is True
            assert resp["result"]["limit_usd"] is None
