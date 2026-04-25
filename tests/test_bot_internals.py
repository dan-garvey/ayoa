"""Focused unit tests for bot-layer internals that the code review
flagged as having no coverage. Covers:

- F3.9: best-effort parsing of DISCORD_ADMIN_USER_IDS (_is_admin).
- F3.8: EngineBridge.import_story invokes on_analysis_complete with the
  right (analysis, error) tuple in both success and failure paths.
- briefing copy: render_briefing must not mention `/describe` (legacy
  command renamed in the join-overhaul) and must point at `/join`.
- POV-thread cascade: `_post_actor_render` falls thread → DM → none.

Heavier discord-interaction paths (full /act and /join harness, orphan
thread purge) still aren't covered — they'd need a discord.py mock
infrastructure we don't have.
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
                    is_playable=True,
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
                    is_playable=False,
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
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="x",
                    feasible=True,
                    resolved_outcome="x",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[],
            ),
            observers=[],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="",
            spawn=[],
            dormant=[],
            cull=["villain"],
            roster_moves=[],
            scenes_created=[],
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


# ---- briefing copy: /describe demoted, /join is the canonical opener ---


class TestBriefingCopy:
    """The /story start briefing used to point players at /describe as the
    next-step command. Under the new /join flow `/describe` is an
    advanced-only mid-game tweak, and the briefing should funnel
    everyone through /join (which also exposes custom-create)."""

    def _ckpt_with_primer(self, primer: str) -> CheckpointFile:
        return CheckpointFile(
            session=SessionState(session_id="briefing_test"),
            world_state=WorldState(),
            player_primer=primer,
        )

    def _embed_text(self, embed) -> str:
        """Concatenate description + every field value so the assertion
        catches mentions wherever the renderer puts them."""
        parts = [embed.description or ""]
        for field in embed.fields:
            parts.append(field.value or "")
        return "\n".join(parts)

    def test_briefing_does_not_mention_describe(self):
        from app.bot.embed import render_briefing
        ckpt = self._ckpt_with_primer(
            "You wake up in a sun-drenched villa. You don't remember "
            "the cameras or the roses. You suspect both are imminent."
        )
        embed = render_briefing(ckpt, story_id="dating_villa_s1")

        text = self._embed_text(embed)
        assert "/describe" not in text, text
        # Sanity: still funnels players to the canonical entry command.
        assert "/join" in text, text

    def test_briefing_falls_back_to_stub_without_describe(self):
        """Pre-v8 / hand-built checkpoints with no primer also must not
        leak a /describe mention via the fallback copy."""
        from app.bot.embed import render_briefing
        ckpt = CheckpointFile(
            session=SessionState(session_id="briefing_fallback"),
            world_state=WorldState(),
            player_primer="",
        )
        embed = render_briefing(ckpt, story_id="legacy_story")
        text = self._embed_text(embed)
        assert "/describe" not in text, text
        assert "/join" in text, text


# ---- _post_actor_render: thread → DM → public cascade -----------------


class TestPostActorRenderCascade:
    """The actor's narrative is delivered POV-thread-first, with DM and
    public-channel fallbacks. This unifies solo and multi-player UX —
    every bound human reads their beat in a private thread; the public
    channel is lobby/acks only. Tests stub the discord layer at the
    helper boundary (`_session_text_channel`, `_ensure_pov_thread`)
    so we can drive every venue branch deterministically without a
    full discord.py harness."""

    def _make_env(
        self, monkeypatch, tmp_path: Path,
        *, thread_send_behavior, dm_succeeds: bool,
    ):
        """Patch the two discord-touching helpers used by
        `_post_actor_render` and return (inter, user, smap, embeds,
        captured) so the test can drive every venue branch.

        `thread_send_behavior` is one of:
          * `None` — `_ensure_pov_thread` returns None (no thread).
          * `"ok"` — thread exists; `.send` succeeds and captures.
          * `"raise"` — thread exists; `.send` raises RuntimeError.
        """
        from app.bot import commands as bot_commands
        from app.bot.session_map import SessionMap

        monkeypatch.setattr(
            bot_commands, "_session_text_channel",
            lambda inter: object(),  # non-None sentinel
        )

        captured: dict = {"thread_sends": [], "dm_sends": []}
        thread_obj = None
        if thread_send_behavior is not None:
            thread_obj = MagicMock()
            thread_obj.id = 999
            thread_obj.mention = "<#999>"
            if thread_send_behavior == "ok":
                async def _thread_send(*args, **kwargs):
                    captured["thread_sends"].append((args, kwargs))
                thread_obj.send = AsyncMock(side_effect=_thread_send)
            elif thread_send_behavior == "raise":
                thread_obj.send = AsyncMock(
                    side_effect=RuntimeError("server hates us"),
                )
            else:
                raise ValueError(
                    f"unknown thread_send_behavior: {thread_send_behavior!r}"
                )

        async def _ensure(**kwargs):
            return thread_obj

        monkeypatch.setattr(
            bot_commands, "_ensure_pov_thread", _ensure,
        )

        async def _user_send(*args, **kwargs):
            captured["dm_sends"].append((args, kwargs))
            if not dm_succeeds:
                raise RuntimeError("simulated DM failure")

        user = MagicMock()
        user.id = 42
        user.send = AsyncMock(side_effect=_user_send)

        inter = MagicMock()
        inter.channel = MagicMock()
        inter.channel.id = 777
        inter.channel_id = 777
        inter.user = user

        smap = SessionMap(db_path=tmp_path / "actor_render.db")
        asyncio.run(smap.init())

        import discord as _discord
        embeds = [MagicMock(spec=_discord.Embed)]
        return inter, user, smap, embeds, captured, thread_obj

    def test_thread_success_returns_thread_and_skips_dm(
        self, monkeypatch, tmp_path: Path,
    ):
        """Happy path — thread.send works; DM is never attempted."""
        from app.bot.commands import _post_actor_render

        inter, user, smap, embeds, captured, thread = self._make_env(
            monkeypatch, tmp_path,
            thread_send_behavior="ok", dm_succeeds=True,
        )

        venue, returned_thread = asyncio.run(_post_actor_render(
            inter=inter, smap=smap, user=user,
            character_id="alice", char_name="Alice",
            embeds=embeds, intro_content="**Alice** acted.",
        ))
        assert venue == "thread"
        assert returned_thread is thread
        assert len(captured["thread_sends"]) == 1
        assert captured["dm_sends"] == []
        _, kwargs = captured["thread_sends"][0]
        assert kwargs.get("content") == "**Alice** acted."
        assert kwargs.get("embeds") is embeds

    def test_thread_send_failure_falls_back_to_dm(
        self, monkeypatch, tmp_path: Path,
    ):
        """thread.send raising → cached id is cleared and DM is tried."""
        from app.bot.commands import _post_actor_render

        inter, user, smap, embeds, captured, _ = self._make_env(
            monkeypatch, tmp_path,
            thread_send_behavior="raise", dm_succeeds=True,
        )

        venue, returned_thread = asyncio.run(_post_actor_render(
            inter=inter, smap=smap, user=user,
            character_id="alice", char_name="Alice",
            embeds=embeds, intro_content="x",
        ))
        assert venue == "dm"
        assert returned_thread is None
        assert len(captured["dm_sends"]) == 1

    def test_no_thread_available_uses_dm(
        self, monkeypatch, tmp_path: Path,
    ):
        """`_ensure_pov_thread` returning None (no perms etc.) → DM only."""
        from app.bot.commands import _post_actor_render

        inter, user, smap, embeds, captured, _ = self._make_env(
            monkeypatch, tmp_path,
            thread_send_behavior=None, dm_succeeds=True,
        )
        venue, returned_thread = asyncio.run(_post_actor_render(
            inter=inter, smap=smap, user=user,
            character_id="alice", char_name="Alice",
            embeds=embeds,
        ))
        assert venue == "dm"
        assert returned_thread is None
        assert len(captured["dm_sends"]) == 1

    def test_both_paths_fail_returns_none(
        self, monkeypatch, tmp_path: Path,
    ):
        """Neither thread nor DM works → caller must fall back to public."""
        from app.bot.commands import _post_actor_render

        inter, user, smap, embeds, captured, _ = self._make_env(
            monkeypatch, tmp_path,
            thread_send_behavior=None, dm_succeeds=False,
        )
        venue, returned_thread = asyncio.run(_post_actor_render(
            inter=inter, smap=smap, user=user,
            character_id="alice", char_name="Alice",
            embeds=embeds,
        ))
        assert venue == "none"
        assert returned_thread is None
