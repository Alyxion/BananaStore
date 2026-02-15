"""Integration tests that hit real APIs. Skipped unless API keys are set."""

import os

import pytest
from fastapi.testclient import TestClient

from app.costs import tracker
from app.main import app


client = TestClient(app)

HAVE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
HAVE_GOOGLE = bool(os.getenv("GOOGLE_API_KEY"))
HAVE_ANTHROPIC = bool(os.getenv("ANTHROPIC_API_KEY"))


@pytest.mark.skipif(not HAVE_OPENAI, reason="OPENAI_API_KEY not set")
class TestOpenaiIntegration:
    def test_suggest_filename_records_cost(self):
        resp = client.post("/api/suggest-filename", json={"description": "a cute puppy on a hill"})
        assert resp.status_code == 200
        assert tracker.total_usd > 0
        by_cat = tracker.totals_by_category()
        assert "prompt" in by_cat

    def test_generate_image_low_1024(self):
        resp = client.post("/api/generate", data={
            "provider": "openai",
            "description": "a single yellow banana on white background",
            "quality": "low",
            "ratio": "1:1",
            "format": "Photo",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["cost_usd"] is not None
        assert abs(body["cost_usd"] - 0.011) < 0.001

    def test_generate_svg_records_cost(self):
        resp = client.post("/api/generate", data={
            "provider": "openai",
            "description": "simple circle icon",
            "quality": "low",
            "ratio": "1:1",
            "format": "Vector",
        })
        assert resp.status_code == 200
        assert tracker.total_usd > 0
        by_cat = tracker.totals_by_category()
        assert "prompt" in by_cat

    def test_synthesize_speech_records_cost(self):
        resp = client.post("/api/tts-openai", json={"text": "Hi"})
        assert resp.status_code == 200
        assert tracker.total_usd > 0
        by_cat = tracker.totals_by_category()
        assert "voice_output" in by_cat

    def test_transcribe_records_cost(self):
        # Generate a tiny audio clip first
        tts_resp = client.post("/api/tts-openai", json={"text": "Hello"})
        assert tts_resp.status_code == 200
        tracker.reset()

        resp = client.post(
            "/api/transcribe-openai",
            files={"audio": ("test.mp3", tts_resp.content, "audio/mpeg")},
        )
        assert resp.status_code == 200
        assert tracker.total_usd > 0
        by_cat = tracker.totals_by_category()
        assert "voice_input" in by_cat


@pytest.mark.skipif(not HAVE_GOOGLE, reason="GOOGLE_API_KEY not set")
class TestGoogleIntegration:
    def test_generate_image_records_cost(self):
        resp = client.post("/api/generate", data={
            "provider": "google",
            "description": "a single yellow banana on white background",
            "quality": "standard",
            "ratio": "1:1",
            "format": "Photo",
        })
        assert resp.status_code == 200
        assert abs(tracker.total_usd - 0.04) < 0.001


@pytest.mark.skipif(not HAVE_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
class TestAnthropicIntegration:
    def test_generate_svg_records_cost(self):
        resp = client.post("/api/generate", data={
            "provider": "anthropic",
            "description": "simple red square icon",
            "quality": "low",
            "ratio": "1:1",
            "format": "Vector",
        })
        assert resp.status_code == 200
        assert tracker.total_usd > 0
        by_cat = tracker.totals_by_category()
        assert "prompt" in by_cat


@pytest.mark.skipif(not HAVE_OPENAI, reason="OPENAI_API_KEY not set")
class TestSpendingLimitIntegration:
    def test_limit_enforcement_returns_429(self):
        # Set a very low limit
        resp = client.post("/api/costs/limit", json={"limit_usd": 0.001})
        assert resp.status_code == 200

        # First cheap call to consume the limit
        client.post("/api/suggest-filename", json={"description": "a dog"})

        # Second call should hit the limit
        resp = client.post("/api/suggest-filename", json={"description": "another dog"})
        assert resp.status_code == 429
        body = resp.json()
        assert "limit" in body
        assert "current" in body


# --- Cost API endpoint tests (no real API keys needed) ---


class TestCostEndpoints:
    def test_get_costs_empty(self):
        resp = client.get("/api/costs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_usd"] == 0.0
        assert body["entry_count"] == 0

    def test_get_costs_history_empty(self):
        resp = client.get("/api/costs/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_set_and_get_limit(self):
        resp = client.post("/api/costs/limit", json={"limit_usd": 5.0})
        assert resp.status_code == 200
        assert resp.json()["limit_usd"] == 5.0

        resp = client.get("/api/costs")
        assert resp.json()["limit_usd"] == 5.0

    def test_remove_limit(self):
        tracker.limit_usd = 10.0
        resp = client.post("/api/costs/limit", json={"limit_usd": None})
        assert resp.status_code == 200
        assert resp.json()["limit_usd"] is None
