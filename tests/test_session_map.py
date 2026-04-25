"""Tests for SessionMap — the SQLite mapping from Discord channel/user
keys to engine sessions and per-user POV threads.

The session-row CRUD is well-trodden; this file exists primarily to lock
in the new pov_threads table introduced by ux-threads-7 (per-user
private threads in the story channel) so the schema and the three
helpers (get/set/clear) don't regress.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.bot.session_map import SessionMap


@pytest.fixture
async def smap(tmp_path: Path) -> SessionMap:
    """Fresh SessionMap on a temp DB file. `init()` runs the schema
    creation idempotently; calling it twice should be a no-op."""
    sm = SessionMap(db_path=tmp_path / "sessionmap.db")
    await sm.init()
    return sm


# ---- session row CRUD (smoke) -------------------------------------------


class TestSessionRowCrud:
    """Basic smoke coverage for the existing sessions table — the bot
    relies on these and there were no tests anywhere in the suite."""

    async def test_get_missing_returns_none(self, smap: SessionMap):
        assert await smap.get(channel_id=12345) is None

    async def test_upsert_then_get(self, smap: SessionMap):
        await smap.upsert(
            channel_id=12345,
            guild_id=999,
            session_id="sess-a",
            owner_user_id=42,
            story_id="story-1",
        )
        row = await smap.get(channel_id=12345)
        assert row is not None
        assert row.channel_id == 12345
        assert row.guild_id == 999
        assert row.session_id == "sess-a"
        assert row.owner_user_id == 42
        assert row.story_id == "story-1"
        assert row.created_at > 0

    async def test_upsert_overwrites(self, smap: SessionMap):
        """ON CONFLICT clause must replace all mutable fields, not just
        bump created_at — a /story start on a channel that already had
        a session must repoint everything."""
        await smap.upsert(
            channel_id=12345, guild_id=999, session_id="old",
            owner_user_id=1, story_id="story-old",
        )
        await smap.upsert(
            channel_id=12345, guild_id=999, session_id="new",
            owner_user_id=2, story_id="story-new",
        )
        row = await smap.get(channel_id=12345)
        assert row is not None
        assert row.session_id == "new"
        assert row.owner_user_id == 2
        assert row.story_id == "story-new"

    async def test_delete_returns_true_when_present(self, smap: SessionMap):
        await smap.upsert(
            channel_id=12345, guild_id=999, session_id="x",
            owner_user_id=1, story_id="s",
        )
        assert await smap.delete(12345) is True
        assert await smap.get(12345) is None

    async def test_delete_returns_false_when_absent(self, smap: SessionMap):
        assert await smap.delete(99999) is False

    async def test_delete_also_drops_pov_threads(self, smap: SessionMap):
        """`/session end` must clean up cached POV threads for the
        channel; otherwise a fresh `/story start` in the same channel
        would inherit thread ids from the previous (now-detached)
        session and silently send POV output to dead threads."""
        await smap.upsert(
            channel_id=12345, guild_id=999, session_id="old",
            owner_user_id=1, story_id="s",
        )
        await smap.set_pov_thread(
            channel_id=12345, user_id=42, thread_id=777, character_id="hero",
        )
        await smap.set_pov_thread(
            channel_id=12345, user_id=43, thread_id=888, character_id="rival",
        )
        # Sanity: threads cached for THIS channel and a different one.
        await smap.set_pov_thread(
            channel_id=99999, user_id=42, thread_id=111, character_id="other",
        )

        await smap.delete(12345)

        assert await smap.get(12345) is None
        # Both pov_thread rows for channel 12345 are gone…
        assert await smap.get_pov_thread_id(12345, 42) is None
        assert await smap.get_pov_thread_id(12345, 43) is None
        # …but rows keyed to a DIFFERENT channel survive.
        assert await smap.get_pov_thread_id(99999, 42) == 111


# ---- pov threads (ux-threads-7) -----------------------------------------


class TestPovThreads:
    async def test_get_missing_returns_none(self, smap: SessionMap):
        """Unknown (channel, user) pairs return None — the bot uses this
        as the signal to lazily create a fresh thread."""
        assert await smap.get_pov_thread_id(channel_id=1, user_id=2) is None

    async def test_set_then_get(self, smap: SessionMap):
        await smap.set_pov_thread(
            channel_id=10, user_id=20, thread_id=999, character_id="hero",
        )
        assert await smap.get_pov_thread_id(channel_id=10, user_id=20) == 999

    async def test_set_overwrites_existing(self, smap: SessionMap):
        """A second call for the same (channel, user) must replace the
        cached thread id — used when a stale thread is detected and a
        fresh one gets opened."""
        await smap.set_pov_thread(
            channel_id=10, user_id=20, thread_id=111, character_id="hero",
        )
        await smap.set_pov_thread(
            channel_id=10, user_id=20, thread_id=222, character_id="hero",
        )
        assert await smap.get_pov_thread_id(channel_id=10, user_id=20) == 222

    async def test_set_keyed_by_channel_and_user(self, smap: SessionMap):
        """The PRIMARY KEY is (channel, user) — the same user in a
        different channel gets a separate thread, and two users in the
        same channel each get their own."""
        await smap.set_pov_thread(
            channel_id=10, user_id=20, thread_id=111, character_id="a",
        )
        await smap.set_pov_thread(
            channel_id=10, user_id=21, thread_id=222, character_id="b",
        )
        await smap.set_pov_thread(
            channel_id=11, user_id=20, thread_id=333, character_id="c",
        )
        assert await smap.get_pov_thread_id(channel_id=10, user_id=20) == 111
        assert await smap.get_pov_thread_id(channel_id=10, user_id=21) == 222
        assert await smap.get_pov_thread_id(channel_id=11, user_id=20) == 333

    async def test_clear_drops_row(self, smap: SessionMap):
        await smap.set_pov_thread(
            channel_id=10, user_id=20, thread_id=999, character_id="hero",
        )
        await smap.clear_pov_thread(channel_id=10, user_id=20)
        assert await smap.get_pov_thread_id(channel_id=10, user_id=20) is None

    async def test_clear_missing_is_noop(self, smap: SessionMap):
        """Clearing a row that was never set must not raise — the
        bot calls clear unconditionally on certain failure paths."""
        await smap.clear_pov_thread(channel_id=999, user_id=999)

    async def test_init_is_idempotent(self, tmp_path: Path):
        """Re-running init on an existing DB must not blow up; the
        bot's startup path may call it on every connection."""
        sm = SessionMap(db_path=tmp_path / "sm.db")
        await sm.init()
        await sm.init()
        await sm.set_pov_thread(
            channel_id=1, user_id=1, thread_id=42, character_id="x",
        )
        assert await sm.get_pov_thread_id(channel_id=1, user_id=1) == 42
