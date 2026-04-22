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


# ---- v11-A5: sweep hook + purge wire-up --------------------------------------


class TestEngineBridgeSweepHook:
    """EngineBridge exposes a sweep-stale-pins hook the /act hot path can
    call before running the orchestrator. Invocation happens inside
    run_turn; tests here drive the primitive directly.
    """

    def test_sweep_stale_pins_method_exists(self, mock_bridge):
        """EngineBridge exposes a sweep method that callers can invoke."""
        assert hasattr(mock_bridge, "sweep_stale_pins")
        assert callable(mock_bridge.sweep_stale_pins)

    def test_sweep_resolves_expired_pins(self, mock_bridge):
        """When a session has a Cat II event older than the timeout, the
        sweep marks its stale responders as swept (via structured list)."""
        from datetime import datetime, timedelta, timezone

        from app.schemas.checkpoint import CheckpointFile
        from app.schemas.state import OpenCatIIEvent, SessionState, WorldState

        session_id = "swtest"
        ckpt = CheckpointFile(
            session=SessionState(
                session_id=session_id,
                character_bindings={"alice": "1"},
            ),
            world_state=WorldState(),
            characters=[],
        )
        ckpt.session.config.settings.cat_ii_human_timeout_seconds = 1
        evt = OpenCatIIEvent(
            event_id="evt_a",
            scene_id="gate",
            initiator_id="pip",
            initiator_intention="punch",
            required_responders=["alice"],
        )
        evt.opened_at = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).isoformat()
        ckpt.session.open_cat_ii_events.append(evt)

        mock_bridge.checkpoint_mgr.save(ckpt)

        swept = mock_bridge.sweep_stale_pins(session_id)
        assert "evt_a" in swept

        reloaded = mock_bridge.load_latest(session_id)
        evt_live = next(
            e for e in reloaded.session.open_cat_ii_events
            if e.event_id == "evt_a"
        )
        assert "alice" in evt_live.swept_responders

    def test_sweep_noop_when_nothing_stale(self, mock_bridge):
        """Sweep on a session with no open events returns [] and doesn't
        touch the checkpoint."""
        from app.schemas.checkpoint import CheckpointFile
        from app.schemas.state import SessionState, WorldState

        ckpt = CheckpointFile(
            session=SessionState(session_id="noop_sw"),
            world_state=WorldState(),
            characters=[],
        )
        mock_bridge.checkpoint_mgr.save(ckpt)
        assert mock_bridge.sweep_stale_pins("noop_sw") == []


class TestPurgeOnUnbind:
    """unbind_user must call purge_character_state so a /leave mid-beat
    doesn't strand slot pins, responder entries, or render buffers."""

    def test_unbind_user_calls_purge(self, mock_bridge):
        from app.schemas.characters import CharacterRecord, PublicSheet
        from app.schemas.checkpoint import CheckpointFile
        from app.schemas.state import (
            OpenCatIIEvent,
            RenderBufferEntry,
            SessionState,
            SlotEntry,
            WorldState,
        )

        session_id = "purgetest"
        ckpt = CheckpointFile(
            session=SessionState(
                session_id=session_id,
                character_bindings={"bob": "77"},
            ),
            world_state=WorldState(),
            characters=[
                CharacterRecord(
                    character_id="bob",
                    name="Bob",
                    public_sheet=PublicSheet(role="player"),
                    location="gate",
                    is_player=True,
                ),
            ],
        )
        evt = OpenCatIIEvent(
            event_id="evt_b",
            scene_id="gate",
            initiator_id="pip",
            initiator_intention="punch",
            required_responders=["bob"],
        )
        ckpt.session.open_cat_ii_events.append(evt)
        ckpt.session.active_act_slots["gate"] = {
            "bob": SlotEntry(
                reason="cat_ii_responder",
                cat_ii_event_id="evt_b",
                claimed_at="",
            ),
        }
        ckpt.session.render_buffers["bob"] = [
            RenderBufferEntry(event_id="evt_prior", observation_level="direct"),
        ]
        mock_bridge.checkpoint_mgr.save(ckpt)

        freed = mock_bridge.unbind_user(session_id, 77)
        assert freed == "bob"

        reloaded = mock_bridge.load_latest(session_id)
        # Binding gone.
        assert "bob" not in reloaded.session.character_bindings
        # Slot pin gone.
        assert "bob" not in reloaded.session.active_act_slots.get("gate", {})
        # Bob removed from the open event's required_responders.
        assert not any(
            "bob" in e.required_responders
            for e in reloaded.session.open_cat_ii_events
        )
        # Render buffer swept.
        assert "bob" not in reloaded.session.render_buffers


class TestApplyRosterUpdatesPurgesCulled:
    """Culling a character should purge their v11 slot/event state too."""

    def test_cull_triggers_purge(self):
        from app.engine.character_manager import CharacterManager
        from app.schemas.characters import CharacterRecord, PublicSheet
        from app.schemas.checkpoint import CheckpointFile
        from app.schemas.event_router import EventRouterOutput
        from app.schemas.events import (
            CanonicalEvent,
            SceneDelta,
            WorldAdjudication,
        )
        from app.schemas.state import OpenCatIIEvent, SessionState, WorldState

        ckpt = CheckpointFile(
            session=SessionState(session_id="culltest"),
            world_state=WorldState(),
            characters=[
                CharacterRecord(
                    character_id="villain",
                    name="Villain",
                    public_sheet=PublicSheet(role="npc"),
                    location="gate",
                    is_player=False,
                ),
            ],
        )
        ckpt.session.open_cat_ii_events.append(
            OpenCatIIEvent(
                event_id="evt_c",
                scene_id="gate",
                initiator_id="villain",
                initiator_intention="swing",
                required_responders=["hero"],
            )
        )

        routed = EventRouterOutput(
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="x",
                    feasible=True,
                    resolved_outcome="x",
                ),
                scene_delta=SceneDelta(
                    time_advanced_seconds=0, new_scene_id=""
                ),
                observable_facts=[],
            ),
            cull=["villain"],
        )
        mgr = CharacterManager()
        mgr.apply_roster_updates(ckpt, routed)
        # Event initiated by the culled character is abandoned.
        assert not any(
            e.initiator_id == "villain"
            for e in ckpt.session.open_cat_ii_events
        )


# ---- v11-r6b: sweep drives re-adjudication -----------------------------------


class TestSweepDrivesReadjudication:
    """EngineBridge.run_turn must close out Cat II events that sweep_
    stale_pins populated with AFK intentions BEFORE running the
    player's /act. Without this, a scene pinned on an AFK human sits
    open and every subsequent /act bounces off the pin."""

    def test_sweep_returns_event_ids_triggers_resolve_cat_ii(
        self, mock_bridge,
    ):
        """When sweep_stale_pins returns event ids, run_turn awaits
        orchestrator.resolve_cat_ii(session_id, event_id) for each
        before invoking process_turn."""
        from app.schemas.responses import TurnResponse

        # Stub sweep_stale_pins to return a single stale event id.
        mock_bridge.sweep_stale_pins = MagicMock(return_value=["evt_x"])
        # Mock both orchestrator entry points as AsyncMocks; the test
        # only cares about the call order + arguments.
        mock_bridge.orchestrator.resolve_cat_ii = AsyncMock(
            return_value=TurnResponse(
                session_id="session",
                beat_ended_reason="cat_ii_resolution",
            )
        )
        mock_bridge.orchestrator.process_turn = AsyncMock(
            return_value=TurnResponse(
                session_id="session",
                beat_ended_reason="directed_at_player",
            )
        )

        async def run():
            return await mock_bridge.run_turn(
                session_id="session",
                user_input="I look around",
                acting_character_id="alice",
            )

        result = asyncio.run(run())

        # resolve_cat_ii was awaited exactly once with the swept event id.
        mock_bridge.orchestrator.resolve_cat_ii.assert_awaited_once_with(
            "session", "evt_x",
        )
        # process_turn still ran after the re-adjudication completed;
        # the caller's /act should never be silently dropped.
        assert mock_bridge.orchestrator.process_turn.await_count == 1
        # run_turn returns the process_turn result, not resolve_cat_ii's.
        assert result.beat_ended_reason == "directed_at_player"

    def test_resolve_cat_ii_failure_does_not_block_current_act(
        self, mock_bridge,
    ):
        """If resolve_cat_ii raises, the /act still proceeds — one
        wedged stale event should never permanently block the session."""
        from app.schemas.responses import TurnResponse

        mock_bridge.sweep_stale_pins = MagicMock(return_value=["evt_bad"])
        mock_bridge.orchestrator.resolve_cat_ii = AsyncMock(
            side_effect=RuntimeError("boom"),
        )
        mock_bridge.orchestrator.process_turn = AsyncMock(
            return_value=TurnResponse(
                session_id="session",
                beat_ended_reason="directed_at_player",
            )
        )

        async def run():
            return await mock_bridge.run_turn(
                session_id="session",
                user_input="hi",
                acting_character_id="alice",
            )

        result = asyncio.run(run())
        mock_bridge.orchestrator.resolve_cat_ii.assert_awaited_once()
        assert mock_bridge.orchestrator.process_turn.await_count == 1
        assert result.beat_ended_reason == "directed_at_player"
