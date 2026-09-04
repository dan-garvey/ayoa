"""Single-writer locks for durable session aggregates."""

from __future__ import annotations

import asyncio


class SessionWriterLock:
    """Task-reentrant lock so bridge helpers can share one session writer."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None
        self._depth = 0

    async def acquire(self) -> bool:
        current = asyncio.current_task()
        if current is not None and current is self._owner:
            self._depth += 1
            return True
        await self._lock.acquire()
        self._owner = current
        self._depth = 1
        return True

    def release(self) -> None:
        current = asyncio.current_task()
        if current is not self._owner or self._depth <= 0:
            raise RuntimeError("session writer lock released by a non-owner")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    async def __aenter__(self) -> "SessionWriterLock":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.release()


class SessionWriterLocks:
    def __init__(self) -> None:
        self._locks: dict[str, SessionWriterLock] = {}
        self._guard = asyncio.Lock()

    async def for_session(self, session_id: str) -> SessionWriterLock:
        if not session_id.strip():
            raise ValueError("session id must not be blank")
        async with self._guard:
            return self._locks.setdefault(session_id, SessionWriterLock())

    async def get(self, session_id: str) -> SessionWriterLock:
        return await self.for_session(session_id)
