"""Session registry with UUID token auth for WebSocket connections."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from starlette.websockets import WebSocket

from app.costs import CostTracker


@dataclass
class Session:
    token: str
    created_at: float
    last_active: float
    tracker: CostTracker = field(default_factory=CostTracker)
    websocket: WebSocket | None = None


class SessionRegistry:
    """Manages sessions and exposes lifecycle hooks for host integration.

    Lifecycle callbacks (all optional, all async):
        on_connect(session, websocket) -> bool
            Called after token validation, before the message loop.
            Return False to reject the connection.

        on_disconnect(session)
            Called when the WebSocket disconnects (clean or abrupt).
            Use this for cleanup, budget return, usage logging, etc.
    """

    def __init__(self, idle_timeout: int = 1800) -> None:
        self._sessions: dict[str, Session] = {}
        self._idle_timeout = idle_timeout
        self._cleanup_task: asyncio.Task | None = None

        # Lifecycle hooks — set by the host application
        self.on_connect: Callable[[Session, WebSocket], Awaitable[bool]] | None = None
        self.on_disconnect: Callable[[Session], Awaitable[Any]] | None = None

    async def create_session(self) -> Session:
        token = str(uuid.uuid4())
        now = time.time()
        session = Session(token=token, created_at=now, last_active=now)
        self._sessions[token] = session
        return session

    async def get_session(self, token: str) -> Session | None:
        session = self._sessions.get(token)
        if session:
            session.last_active = time.time()
        return session

    async def remove_session(self, token: str) -> None:
        self._sessions.pop(token, None)

    async def cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            token for token, session in self._sessions.items()
            if now - session.last_active > self._idle_timeout
        ]
        for token in expired:
            self._sessions.pop(token, None)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self.cleanup_expired()

    def start_cleanup(self) -> None:
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop_cleanup(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    @property
    def session_count(self) -> int:
        return len(self._sessions)


# Module-level singleton
registry = SessionRegistry()
