"""SQLite mapping from Discord channels to engine sessions.

One row per channel that has an active story. Deleting a row detaches the
channel from its story (the checkpoint files on disk stay intact).
"""

from __future__ import annotations

import time
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_DB_PATH = Path("app/storage/bot/sessionmap.db")


@dataclass
class SessionRow:
    channel_id: int
    guild_id: Optional[int]
    session_id: str
    owner_user_id: int
    story_id: str
    created_at: int


@dataclass(frozen=True)
class TurnMessageRef:
    channel_id: int
    session_id: str
    turn_index: int
    discord_channel_id: int
    message_id: int
    delivery: str
    recipient_user_id: Optional[int]
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

CREATE TABLE IF NOT EXISTS turn_messages (
    channel_id          INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    turn_index          INTEGER NOT NULL,
    discord_channel_id  INTEGER NOT NULL,
    message_id          INTEGER NOT NULL,
    delivery            TEXT NOT NULL,
    recipient_user_id   INTEGER,
    created_at          INTEGER NOT NULL,
    PRIMARY KEY (
        channel_id,
        session_id,
        turn_index,
        discord_channel_id,
        message_id
    )
);

CREATE INDEX IF NOT EXISTS turn_messages_rewind_idx
ON turn_messages(channel_id, session_id, turn_index);

CREATE TABLE IF NOT EXISTS media_cleanup_outbox (
    channel_id          INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    turn_index          INTEGER NOT NULL,
    discord_channel_id  INTEGER NOT NULL,
    message_id          INTEGER NOT NULL,
    delivery            TEXT NOT NULL,
    recipient_user_id   INTEGER,
    created_at          INTEGER NOT NULL,
    PRIMARY KEY (discord_channel_id, message_id)
);
"""


class SessionMap:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.executescript(_SCHEMA)

    async def get(self, channel_id: int) -> Optional[SessionRow]:
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                "SELECT channel_id, guild_id, session_id, owner_user_id, "
                "story_id, created_at FROM sessions WHERE channel_id = ?",
                (channel_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return SessionRow(*row)

    async def get_by_session(self, session_id: str) -> Optional[SessionRow]:
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                "SELECT channel_id, guild_id, session_id, owner_user_id, "
                "story_id, created_at FROM sessions WHERE session_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            )
            row = cursor.fetchone()
        return SessionRow(*row) if row is not None else None

    async def upsert(
        self,
        *,
        channel_id: int,
        guild_id: Optional[int],
        session_id: str,
        owner_user_id: int,
        story_id: str,
    ) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
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
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                "DELETE FROM sessions WHERE channel_id = ?", (channel_id,)
            )
            db.execute(
                "DELETE FROM pov_threads WHERE channel_id = ?", (channel_id,),
            )
            db.execute(
                "DELETE FROM turn_messages WHERE channel_id = ?",
                (channel_id,),
            )
            return cursor.rowcount > 0

    # ---- pov threads (ux-threads-7) -----------------------------------------

    async def get_pov_thread_id(
        self, channel_id: int, user_id: int,
    ) -> Optional[int]:
        """Return the cached private-thread id for this (channel, user),
        or None if no thread has been opened yet (or it was forgotten
        by clear_pov_thread)."""
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                "SELECT thread_id FROM pov_threads "
                "WHERE channel_id = ? AND user_id = ?",
                (channel_id, user_id),
            )
            row = cursor.fetchone()
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
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT INTO pov_threads "
                "(channel_id, user_id, thread_id, character_id, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(channel_id, user_id) DO UPDATE SET "
                "thread_id=excluded.thread_id, "
                "character_id=excluded.character_id",
                (channel_id, user_id, thread_id, character_id, int(time.time())),
            )

    async def clear_pov_thread(
        self, channel_id: int, user_id: int,
    ) -> None:
        """Forget the cached thread id (e.g. after the bot couldn't post
        to it — the next ensure call will create a fresh one)."""
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "DELETE FROM pov_threads "
                "WHERE channel_id = ? AND user_id = ?",
                (channel_id, user_id),
            )

    async def clear_all_pov_threads(self, channel_id: int) -> int:
        """Forget every cached POV-thread row for `channel_id` without
        touching the session binding itself.

        Used by `/clear`: after deleting the actual Discord thread
        objects we drop the SQL cache so the next `/join` lazily
        creates fresh threads instead of trying to send to dead ids.
        Distinct from `delete()`, which also nukes the `sessions` row
        (the engine session stays bound across `/clear`).

        Returns the number of rows deleted.
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                "DELETE FROM pov_threads WHERE channel_id = ?",
                (channel_id,),
            )
            return cursor.rowcount

    # ---- turn messages ------------------------------------------------------

    async def record_turn_message(
        self,
        *,
        channel_id: int,
        session_id: str,
        turn_index: int,
        discord_channel_id: int,
        message_id: int,
        delivery: str,
        recipient_user_id: Optional[int] = None,
    ) -> None:
        """Remember one Discord message that rendered an engine turn.

        `/rewind` deletes checkpoints by turn index. Discord messages live
        outside those checkpoint files, so the bot records every successful
        turn render as it posts it and later uses these refs to delete or
        hide only the messages from rewound turns.
        """
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT OR IGNORE INTO turn_messages "
                "(channel_id, session_id, turn_index, discord_channel_id, "
                "message_id, delivery, recipient_user_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    channel_id,
                    session_id,
                    turn_index,
                    discord_channel_id,
                    message_id,
                    delivery,
                    recipient_user_id,
                    int(time.time()),
                ),
            )

    async def list_turn_messages(
        self,
        *,
        channel_id: int,
        session_id: str,
        turns: Iterable[int],
    ) -> list[TurnMessageRef]:
        turn_list = sorted({int(t) for t in turns})
        if not turn_list:
            return []
        placeholders = ",".join("?" for _ in turn_list)
        params = (channel_id, session_id, *turn_list)
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                "SELECT channel_id, session_id, turn_index, "
                "discord_channel_id, message_id, delivery, "
                "recipient_user_id, created_at "
                "FROM turn_messages "
                f"WHERE channel_id = ? AND session_id = ? "
                f"AND turn_index IN ({placeholders}) "
                "ORDER BY turn_index, created_at, "
                "discord_channel_id, message_id",
                params,
            )
            rows = cursor.fetchall()
        return [TurnMessageRef(*row) for row in rows]

    async def has_turn_delivery(
        self,
        *,
        channel_id: int,
        session_id: str,
        turn_index: int,
        recipient_user_id: int,
        delivery_suffix: str,
    ) -> bool:
        """Whether this private turn artifact was already sent and tracked."""

        suffix = str(delivery_suffix or "").strip()
        if not suffix:
            return False
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute(
                """
                SELECT delivery FROM turn_messages
                WHERE channel_id = ? AND session_id = ? AND turn_index = ?
                  AND recipient_user_id = ?
                """,
                (
                    channel_id,
                    session_id,
                    turn_index,
                    recipient_user_id,
                ),
            ).fetchall()
        return any(str(row[0]).endswith(f"_{suffix}") for row in rows)

    async def forget_turn_messages(
        self,
        refs: Iterable[TurnMessageRef],
    ) -> int:
        ref_list = list(refs)
        if not ref_list:
            return 0
        deleted = 0
        with sqlite3.connect(self.db_path) as db:
            for ref in ref_list:
                cursor = db.execute(
                    "DELETE FROM turn_messages "
                    "WHERE channel_id = ? AND session_id = ? "
                    "AND turn_index = ? AND discord_channel_id = ? "
                    "AND message_id = ?",
                    (
                        ref.channel_id,
                        ref.session_id,
                        ref.turn_index,
                        ref.discord_channel_id,
                        ref.message_id,
                    ),
                )
                deleted += cursor.rowcount
        return deleted

    async def clear_all_turn_messages(self, channel_id: int) -> int:
        """Drop all remembered Discord turn-message refs for a channel.

        Used when `/clear` deletes the underlying channel/thread content.
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                "DELETE FROM turn_messages WHERE channel_id = ?",
                (channel_id,),
            )
            return cursor.rowcount

    async def record_media_cleanup(self, ref: TurnMessageRef) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                INSERT OR REPLACE INTO media_cleanup_outbox (
                    channel_id, session_id, turn_index, discord_channel_id,
                    message_id, delivery, recipient_user_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ref.channel_id,
                    ref.session_id,
                    ref.turn_index,
                    ref.discord_channel_id,
                    ref.message_id,
                    ref.delivery,
                    ref.recipient_user_id,
                    int(time.time()),
                ),
            )

    async def list_media_cleanup(self) -> list[TurnMessageRef]:
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute(
                """
                SELECT channel_id, session_id, turn_index,
                    discord_channel_id, message_id, delivery,
                    recipient_user_id, created_at
                FROM media_cleanup_outbox
                ORDER BY created_at, discord_channel_id, message_id
                """
            ).fetchall()
        return [TurnMessageRef(*row) for row in rows]

    async def forget_media_cleanup(
        self,
        refs: Iterable[TurnMessageRef],
    ) -> int:
        deleted = 0
        with sqlite3.connect(self.db_path) as db:
            for ref in refs:
                cursor = db.execute(
                    """
                    DELETE FROM media_cleanup_outbox
                    WHERE discord_channel_id = ? AND message_id = ?
                    """,
                    (ref.discord_channel_id, ref.message_id),
                )
                deleted += cursor.rowcount
        return deleted
