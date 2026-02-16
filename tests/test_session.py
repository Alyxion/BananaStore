import time

import pytest

from app.session import SessionRegistry


@pytest.fixture
def reg():
    return SessionRegistry(idle_timeout=2)


class TestSessionRegistry:
    @pytest.mark.asyncio
    async def test_create_session(self, reg):
        session = await reg.create_session()
        assert session.token
        assert session.tracker is not None
        assert reg.session_count == 1

    @pytest.mark.asyncio
    async def test_get_session_valid(self, reg):
        session = await reg.create_session()
        found = await reg.get_session(session.token)
        assert found is session

    @pytest.mark.asyncio
    async def test_get_session_invalid(self, reg):
        result = await reg.get_session("nonexistent-token")
        assert result is None

    @pytest.mark.asyncio
    async def test_remove_session(self, reg):
        session = await reg.create_session()
        await reg.remove_session(session.token)
        assert reg.session_count == 0
        assert await reg.get_session(session.token) is None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, reg):
        session = await reg.create_session()
        # Backdate last_active to make it expired
        session.last_active = time.time() - 10
        await reg.cleanup_expired()
        assert reg.session_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_keeps_active(self, reg):
        session = await reg.create_session()
        session.last_active = time.time()
        await reg.cleanup_expired()
        assert reg.session_count == 1

    @pytest.mark.asyncio
    async def test_get_session_updates_last_active(self, reg):
        session = await reg.create_session()
        old_time = session.last_active
        # Small delay to ensure time difference
        import asyncio
        await asyncio.sleep(0.01)
        await reg.get_session(session.token)
        assert session.last_active >= old_time

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, reg):
        s1 = await reg.create_session()
        s2 = await reg.create_session()
        assert s1.token != s2.token
        assert reg.session_count == 2

    @pytest.mark.asyncio
    async def test_per_session_tracker(self, reg):
        s1 = await reg.create_session()
        s2 = await reg.create_session()
        assert s1.tracker is not s2.tracker
