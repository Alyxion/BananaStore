"""Standalone BananaStore — serves index.html with anonymous sessions.

Run with:
    poetry run uvicorn app.standalone:app --host 0.0.0.0 --port 8070 --reload
"""

from app.main import app, enable_standalone

enable_standalone()

__all__ = ["app"]
