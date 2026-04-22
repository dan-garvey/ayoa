"""Focused unit tests for bot-layer internals that the code review
flagged as having no coverage. Covers:

- F3.9: best-effort parsing of DISCORD_ADMIN_USER_IDS (_is_admin).
- F3.8: EngineBridge.import_story invokes on_analysis_complete with the
  right (analysis, error) tuple in both success and failure paths.

Discord-interaction-heavy paths (F3.7 defer, F3.10 orphan purge) aren't
covered here — they need a full discord.py mock harness we don't have.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.bot import commands as bot_commands
from app.bot.engine_bridge import EngineBridge
from app.schemas.checkpoint import CheckpointFile, ImportAnalysis
from app.schemas.state import SessionState, WorldState


# ---- F3.9: admin env parsing ------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_admin_cache():
    """_is_admin memoizes per env-value. Clear between tests so each test
    sees a fresh parse."""
    bot_commands._ADMIN_CACHE = None
    yield
    bot_commands._ADMIN_CACHE = None


class TestAdminEnvParsing:
    def test_unset_env_denies_everyone(self, monkeypatch):
        monkeypatch.delenv("DISCORD_ADMIN_USER_IDS", raising=False)
        assert bot_commands._is_admin(12345) is False

    def test_empty_env_denies_everyone(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "")
        assert bot_commands._is_admin(12345) is False

    def test_single_valid_id(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "12345")
        assert bot_commands._is_admin(12345) is True
        assert bot_commands._is_admin(99999) is False

    def test_comma_list(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "111, 222 , 333")
        for uid in (111, 222, 333):
            assert bot_commands._is_admin(uid) is True
        assert bot_commands._is_admin(444) is False

    def test_bad_entries_skipped_not_rejecting_whole_list(self, monkeypatch, caplog):
        """Core F3.9 fix: one bad entry doesn't nuke all admin access."""
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "111,notanumber,222")
        with caplog.at_level(logging.WARNING):
            assert bot_commands._is_admin(111) is True
            assert bot_commands._is_admin(222) is True
        assert any("notanumber" in r.message for r in caplog.records)

    def test_warning_logged_once_per_unique_env_value(self, monkeypatch, caplog):
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "111,nope")
        with caplog.at_level(logging.WARNING):
            bot_commands._is_admin(111)
            bot_commands._is_admin(111)
            bot_commands._is_admin(111)
        bad_warnings = [r for r in caplog.records if "nope" in r.message]
        assert len(bad_warnings) == 1

    def test_warning_re_fires_when_env_changes(self, monkeypatch, caplog):
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "111,nope")
        with caplog.at_level(logging.WARNING):
            bot_commands._is_admin(111)
            monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "222,stillbad")
            bot_commands._is_admin(222)
        bad_warnings = [r for r in caplog.records if "stillbad" in r.message]
        assert len(bad_warnings) == 1


# ---- F3.8: import_story callback --------------------------------------------


@pytest.fixture
def mock_bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return EngineBridge(saves_dir=str(tmp_path), prompts_dir="app/prompts")


def _minimal_ckpt(story_id: str) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(session_id=story_id, story_id=story_id),
        world_state=WorldState(),
        characters=[],
    )


def _minimal_combined_result(story_id: str):
    """Match the shape `run_import_combined` returns: checkpoint +
    priming_messages + assistant_text, so the test can mock the new
    combined-import path without importing the private result class."""
    from app.engine.story_importer import _CombinedImportResult
    return _CombinedImportResult(
        checkpoint=_minimal_ckpt(story_id),
        priming_messages=[
            {"role": "system", "content": "fake system"},
            {"role": "user", "content": "fake user"},
        ],
        assistant_text='{"fake": "assistant echo"}',
    )


class TestImportAnalysisCallback:
    """EngineBridge.import_story fires on_analysis_complete with
    (analysis, None) on success and (None, exception) on failure.

    Both paths mock out run_import and run_preservation_analysis so the
    test doesn't touch the LLM. The callback is awaited by the
    background task that the bridge schedules; we use asyncio.Event
    plumbing to wait for it deterministically."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_callback_fires_with_analysis_on_success(self, mock_bridge):
        story_id = "cb_success"
        analysis = ImportAnalysis(
            source_chars=100,
            source_words=20,
            output_chars=80,
            output_words=15,
            coverage_rating="high",
            dropped_topics=[],
            compressed_topics=[],
            preservation_notes="fine",
            duration_s=1.0,
            model="claude-sonnet-4-6",
        )

        received: dict = {}
        done = asyncio.Event()

        async def _cb(a, e):
            received["analysis"] = a
            received["err"] = e
            done.set()

        async def run():
            with patch(
                "app.bot.engine_bridge.run_import_two_call",
                new=AsyncMock(return_value=_minimal_combined_result(story_id)),
            ), patch(
                "app.bot.engine_bridge.run_preservation_analysis_continuation",
                new=AsyncMock(return_value=analysis),
            ):
                await mock_bridge.import_story(
                    "source text here", story_id,
                    on_analysis_complete=_cb,
                )
                # Background task is scheduled but not awaited — wait for
                # the callback to fire.
                await asyncio.wait_for(done.wait(), timeout=2.0)

        self._run(run())
        assert received["err"] is None
        assert received["analysis"] is analysis

    def test_callback_fires_with_error_on_failure(self, mock_bridge):
        story_id = "cb_failure"
        boom = RuntimeError("analysis exploded")

        received: dict = {}
        done = asyncio.Event()

        async def _cb(a, e):
            received["analysis"] = a
            received["err"] = e
            done.set()

        async def run():
            with patch(
                "app.bot.engine_bridge.run_import_two_call",
                new=AsyncMock(return_value=_minimal_combined_result(story_id)),
            ), patch(
                "app.bot.engine_bridge.run_preservation_analysis_continuation",
                new=AsyncMock(side_effect=boom),
            ):
                await mock_bridge.import_story(
                    "source text here", story_id,
                    on_analysis_complete=_cb,
                )
                await asyncio.wait_for(done.wait(), timeout=2.0)

        self._run(run())
        assert received["analysis"] is None
        assert received["err"] is boom

    def test_no_callback_is_fine(self, mock_bridge):
        """Omitting the callback is the CLI/legacy path — shouldn't error."""
        story_id = "no_cb"
        analysis = ImportAnalysis(
            coverage_rating="high", duration_s=0.0,
        )

        async def run():
            with patch(
                "app.bot.engine_bridge.run_import_two_call",
                new=AsyncMock(return_value=_minimal_combined_result(story_id)),
            ), patch(
                "app.bot.engine_bridge.run_preservation_analysis_continuation",
                new=AsyncMock(return_value=analysis),
            ):
                ckpt = await mock_bridge.import_story("src", story_id)
                # Give the background task a moment to complete; success
                # path just logs.
                await asyncio.sleep(0.05)
                assert ckpt.session.story_id == story_id

        self._run(run())
