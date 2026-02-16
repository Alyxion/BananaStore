import base64
from unittest.mock import patch, AsyncMock

import pytest
from starlette.testclient import TestClient

from app.main import app
from app.session import registry
from tests.conftest import SAMPLE_PNG_B64, SAMPLE_SVG_B64, ws_connect


client = TestClient(app)


class TestWsConnection:
    def test_connect_no_token_rejected(self):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()

    def test_connect_with_valid_token(self):
        import asyncio
        token = asyncio.get_event_loop().run_until_complete(registry.create_session()).token
        with client.websocket_connect(f"/ws?token={token}") as ws:
            auth = ws.receive_json()
            assert auth["type"] == "auth"
            assert auth["token"] == token

    def test_connect_with_invalid_token(self):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws?token=bad-token-123") as ws:
                ws.receive_json()


class TestWsProviders:
    def test_providers_action(self):
        with ws_connect(client) as ws:
            ws.send_json({"id": "r1", "action": "providers"})
            resp = ws.receive_json()
            assert resp["id"] == "r1"
            assert resp["ok"] is True
            assert "providers" in resp["result"]
            assert "openai" in resp["result"]["providers"]


class TestWsSuggestFilename:
    def test_empty_description(self):
        with ws_connect(client) as ws:
            ws.send_json({"id": "r1", "action": "suggest-filename", "payload": {"description": ""}})
            resp = ws.receive_json()
            assert resp["ok"] is True
            assert resp["result"]["filename"] == "generated-image"

    def test_with_description(self):
        with patch("app.handlers.llm_openai.suggest_filename", new_callable=AsyncMock, return_value="golden-cat"):
            with ws_connect(client) as ws:
                ws.send_json({"id": "r1", "action": "suggest-filename", "payload": {"description": "a golden cat"}})
                resp = ws.receive_json()
        assert resp["ok"] is True
        assert resp["result"]["filename"] == "golden-cat"


class TestWsTranscribe:
    def test_success(self):
        with patch("app.handlers.llm_openai.transcribe_audio", new_callable=AsyncMock, return_value="Hello"):
            audio_b64 = base64.b64encode(b"audiodata").decode()
            with ws_connect(client) as ws:
                ws.send_json({"id": "r1", "action": "transcribe", "payload": {
                    "audio_b64": audio_b64,
                    "filename": "voice.webm",
                    "content_type": "audio/webm",
                }})
                resp = ws.receive_json()
        assert resp["ok"] is True
        assert resp["result"]["text"] == "Hello"

    def test_empty_audio(self):
        with ws_connect(client) as ws:
            ws.send_json({"id": "r1", "action": "transcribe", "payload": {"audio_b64": ""}})
            resp = ws.receive_json()
        assert resp["ok"] is False
        assert resp["code"] == 400


class TestWsDescribeImage:
    def test_success(self):
        with patch("app.handlers.llm_openai.describe_image", new_callable=AsyncMock, return_value="Nice!"):
            with ws_connect(client) as ws:
                ws.send_json({"id": "r1", "action": "describe-image", "payload": {
                    "image_data_url": "data:image/png;base64,abc",
                    "source_text": "a cat",
                    "language": "en",
                }})
                resp = ws.receive_json()
        assert resp["ok"] is True
        assert resp["result"]["description"] == "Nice!"

    def test_invalid_data_url(self):
        with ws_connect(client) as ws:
            ws.send_json({"id": "r1", "action": "describe-image", "payload": {
                "image_data_url": "not-valid",
                "source_text": "",
                "language": "",
            }})
            resp = ws.receive_json()
        assert resp["ok"] is False
        assert resp["code"] == 400


class TestWsTts:
    def test_success(self):
        with patch("app.handlers.llm_openai.synthesize_speech", new_callable=AsyncMock, return_value=b"mp3data"):
            with ws_connect(client) as ws:
                ws.send_json({"id": "r1", "action": "tts", "payload": {"text": "Hello", "language": "en"}})
                resp = ws.receive_json()
        assert resp["ok"] is True
        assert resp["result"]["audio_b64"]
        assert base64.b64decode(resp["result"]["audio_b64"]) == b"mp3data"

    def test_empty_text(self):
        with ws_connect(client) as ws:
            ws.send_json({"id": "r1", "action": "tts", "payload": {"text": "", "language": "en"}})
            resp = ws.receive_json()
        assert resp["ok"] is False
        assert resp["code"] == 400


class TestWsGenerate:
    def test_openai_photo(self):
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"
        with patch("app.handlers.llm_openai.generate_image", new_callable=AsyncMock, return_value=data_url):
            with ws_connect(client) as ws:
                ws.send_json({"id": "r1", "action": "generate", "payload": {
                    "provider": "openai",
                    "description": "a sunset",
                    "quality": "auto",
                    "ratio": "1:1",
                    "format": "Photo",
                    "reference_images": [],
                }})
                resp = ws.receive_json()
        assert resp["ok"] is True
        assert resp["result"]["provider"] == "openai"
        assert resp["result"]["image_data_url"].startswith("data:image/")

    def test_unsupported_provider(self):
        with ws_connect(client) as ws:
            ws.send_json({"id": "r1", "action": "generate", "payload": {
                "provider": "nonexistent",
                "description": "test",
                "quality": "auto",
                "ratio": "1:1",
            }})
            resp = ws.receive_json()
        assert resp["ok"] is False
        assert resp["code"] == 400

    def test_with_reference_images(self):
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"
        with patch("app.handlers.llm_openai.generate_image", new_callable=AsyncMock, return_value=data_url):
            with ws_connect(client) as ws:
                ws.send_json({"id": "r1", "action": "generate", "payload": {
                    "provider": "openai",
                    "description": "a sunset",
                    "quality": "auto",
                    "ratio": "1:1",
                    "format": "Photo",
                    "reference_images": [
                        {"name": "ref.png", "data_b64": SAMPLE_PNG_B64, "content_type": "image/png"},
                    ],
                }})
                resp = ws.receive_json()
        assert resp["ok"] is True
        assert resp["result"]["used_reference_images"] == 1


class TestWsCosts:
    def test_costs_action(self):
        with ws_connect(client) as ws:
            ws.send_json({"id": "r1", "action": "costs"})
            resp = ws.receive_json()
        assert resp["ok"] is True
        assert "total_usd" in resp["result"]

    def test_costs_history(self):
        with ws_connect(client) as ws:
            ws.send_json({"id": "r1", "action": "costs-history"})
            resp = ws.receive_json()
        assert resp["ok"] is True
        assert isinstance(resp["result"], list)

    def test_costs_limit(self):
        with ws_connect(client) as ws:
            ws.send_json({"id": "r1", "action": "costs-limit", "payload": {"limit_usd": 5.0}})
            resp = ws.receive_json()
        assert resp["ok"] is True
        assert resp["result"]["limit_usd"] == 5.0


class TestWsUnknownAction:
    def test_unknown_action(self):
        with ws_connect(client) as ws:
            ws.send_json({"id": "r1", "action": "nonexistent"})
            resp = ws.receive_json()
        assert resp["ok"] is False
        assert resp["code"] == 400
