"""SQLite mapping from Discord channels to engine sessions.

One row per channel that has an active story. Deleting a row detaches the
channel from its story (the checkpoint files on disk stay intact).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiosqlite

DEFAULT_DB_PATH = Path("app/storage/bot/sessionmap.db")


@dataclass
class SessionRow:
    channel_id: int
    guild_id: Optional[int]
    session_id: str
    owner_user_id: int
    story_id: str
    created_at: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    channel_id    INTEGER PRIMARY KEY,
    guild_id      INTEGER,
    session_id    TEXT NOT NULL,
    owner_user_id INTEGER NOT NULL,
    story_id      TEXT NOT NULL,
    created_at    INTEGER NOT NULL
);

-- ux-threads-7: per-user private POV threads. One row per (channel,
-- user) pair; the thread persists across re-binds so the player's
-- narrative log stays continuous even if they /leave and /join a
-- different character. character_id stores whoever they were bound
-- to when the thread was last touched, but it's diagnostic — the
-- (channel_id, user_id) pair is the lookup key.
CREATE TABLE IF NOT EXISTS pov_threads (
    channel_id   INTEGER NOT NULL,
    user_id      INTEGER NOT NULL,
    thread_id    INTEGER NOT NULL,
    character_id TEXT,
    created_at   INTEGER NOT NULL,
    PRIMARY KEY (channel_id, user_id)
);
"""


class SessionMap:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def get(self, channel_id: int) -> Optional[SessionRow]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT channel_id, guild_id, session_id, owner_user_id, "
                "story_id, created_at FROM sessions WHERE channel_id = ?",
                (channel_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return SessionRow(*row)

    async def upsert(
        self,
        *,
        channel_id: int,
        guild_id: Optional[int],
        session_id: str,
        owner_user_id: int,
        story_id: str,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO sessions "
                "(channel_id, guild_id, session_id, owner_user_id, story_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(channel_id) DO UPDATE SET "
                "guild_id=excluded.guild_id, "
                "session_id=excluded.session_id, "
                "owner_user_id=excluded.owner_user_id, "
                "story_id=excluded.story_id, "
                "created_at=excluded.created_at",
                (channel_id, guild_id, session_id, owner_user_id, story_id, int(time.time())),
            )
            await db.commit()

    async def delete(self, channel_id: int) -> bool:
        """Detach this channel from its session AND drop every cached
        POV thread row keyed to it.

        Without the `pov_threads` cleanup, a `/session end` followed by
        a fresh `/session start` + `/story start` in the same channel
        would inherit the previous story's thread ids and silently
        send POV output to dead threads (or, worse, threads from an
        unrelated session). The thread objects themselves are left
        alive in Discord — they're just forgotten by the bot, so the
        next `/join` lazily creates fresh ones.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM sessions WHERE channel_id = ?", (channel_id,)
            )
            await db.execute(
                "DELETE FROM pov_threads WHERE channel_id = ?", (channel_id,),
            )
            await db.commit()
            return cursor.rowcount > 0

    # ---- pov threads (ux-threads-7) -----------------------------------------

    async def get_pov_thread_id(
        self, channel_id: int, user_id: int,
    ) -> Optional[int]:
        """Return the cached private-thread id for this (channel, user),
        or None if no thread has been opened yet (or it was forgotten
        by clear_pov_thread)."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT thread_id FROM pov_threads "
                "WHERE channel_id = ? AND user_id = ?",
                (channel_id, user_id),
            ) as cursor:
                row = await cursor.fetchone()
        return int(row[0]) if row else None

    async def set_pov_thread(
        self,
        *,
        channel_id: int,
        user_id: int,
        thread_id: int,
        character_id: str,
    ) -> None:
        """Cache (or overwrite) the private-thread id for this user. The
        thread persists across re-binds; character_id is diagnostic."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO pov_threads "
                "(channel_id, user_id, thread_id, character_id, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(channel_id, user_id) DO UPDATE SET "
                "thread_id=excluded.thread_id, "
                "character_id=excluded.character_id",
                (channel_id, user_id, thread_id, character_id, int(time.time())),
            )
            await db.commit()

    async def clear_pov_thread(
        self, channel_id: int, user_id: int,
    ) -> None:
        """Forget the cached thread id (e.g. after the bot couldn't post
        to it — the next ensure call will create a fresh one)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM pov_threads "
                "WHERE channel_id = ? AND user_id = ?",
                (channel_id, user_id),
            )
            await db.commit()
