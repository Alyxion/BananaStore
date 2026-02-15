import base64
import io
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile

from app.costs import tracker


SAMPLE_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="red"/></svg>'
SAMPLE_SVG_B64 = base64.b64encode(SAMPLE_SVG.encode()).decode()
SAMPLE_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakedata").decode()


@pytest.fixture(autouse=True)
def _reset_cost_tracker():
    tracker.reset()
    yield
    tracker.reset()


def make_upload_file(content: bytes, filename: str = "test.png", content_type: str = "image/png") -> UploadFile:
    """Create a mock UploadFile with the given content."""
    file_obj = io.BytesIO(content)
    return UploadFile(file=file_obj, filename=filename, headers={"content-type": content_type})


def mock_httpx_response(status_code: int = 200, json_data: dict | None = None, content: bytes = b"") -> AsyncMock:
    """Create a mock httpx.Response."""
    resp = AsyncMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.content = content
    return resp
