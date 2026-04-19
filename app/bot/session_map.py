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
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM sessions WHERE channel_id = ?", (channel_id,)
            )
            await db.commit()
            return cursor.rowcount > 0
