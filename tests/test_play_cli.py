"""Smoke tests for the interactive CLI (scripts/play.py).

Exercises CLIState command dispatch with a mocked EngineBridge — the goal
is to catch regressions in how commands map to bridge calls, claim state,
and current-actor updates. End-to-end engine behavior is covered by the
regular test suite."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.play import CLIState
from app.bot.engine_bridge import CharacterSummary
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import SessionState, WorldState


SESSION_ID = "cli_test"
STORY_ID = "test_story"


def _empty_ckpt(bindings: dict[str, str] | None = None) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            player_character_id="aldric",
            character_bindings=bindings or {},
        ),
        world_state=WorldState(),
        characters=[],
    )


def _mock_engine(bindings: dict[str, str] | None = None) -> MagicMock:
    engine = MagicMock()
    engine.load_latest.return_value = _empty_ckpt(bindings)
    engine.list_session_characters.return_value = [
        CharacterSummary(
            character_id="aldric", name="Aldric", role="wanderer",
            faction="", appearance="tall",
            status="active", is_playable=True,
            bound_user_id=(bindings or {}).get("aldric", ""),
        ),
        CharacterSummary(
            character_id="sera", name="Sera Vance", role="thief",
            faction="", appearance="wiry",
            status="active", is_playable=False,
            bound_user_id=(bindings or {}).get("sera", ""),
        ),
    ]
    engine.takeover = MagicMock()
    engine.unbind_user = MagicMock()
    engine.build_character_dossier = MagicMock(return_value="# Dossier · Sera")
    engine.set_character_identity = MagicMock(return_value=_empty_ckpt(bindings))
    engine.run_turn = AsyncMock()
    return engine


@pytest.fixture
def run():
    """Run an async coroutine in a sync test."""
    return asyncio.run


class TestInitialization:
    def test_adopts_existing_bindings_on_resume(self):
        engine = _mock_engine(bindings={"aldric": "1", "sera": "2"})
        state = CLIState(engine, SESSION_ID, STORY_ID)
        assert state.claims == {"aldric": 1, "sera": 2}
        # Current actor auto-picks the first adopted claim.
        assert state.current_actor == "aldric"
        # Next uid is one past the max seen so we don't collide.
        assert state._next_user_id == 3

    def test_fresh_session_has_no_claims(self):
        engine = _mock_engine(bindings=None)
        state = CLIState(engine, SESSION_ID, STORY_ID)
        assert state.claims == {}
        assert state.current_actor is None

    def test_skips_non_integer_bindings(self):
        # Discord user_ids fit in int, but corrupted data shouldn't crash.
        engine = _mock_engine(bindings={"aldric": "not-a-number"})
        state = CLIState(engine, SESSION_ID, STORY_ID)
        assert state.claims == {}


class TestJoinLeave:
    def test_join_binds_and_sets_current(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join sera"))
        engine.takeover.assert_called_once_with(SESSION_ID, "sera", 1)
        assert state.claims == {"sera": 1}
        assert state.current_actor == "sera"

    def test_second_join_does_not_steal_current_actor(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        assert state.current_actor == "aldric"
        assert state.claims == {"aldric": 1, "sera": 2}

    def test_join_refuses_duplicate_claim(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join sera"))
        engine.takeover.reset_mock()
        run(state.handle_line("/join sera"))
        engine.takeover.assert_not_called()

    def test_leave_default_is_current_actor(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join sera"))
        run(state.handle_line("/leave"))
        engine.unbind_user.assert_called_once_with(SESSION_ID, 1)
        assert state.claims == {}
        assert state.current_actor is None

    def test_leave_switches_current_to_remaining_claim(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        run(state.handle_line("/leave aldric"))
        assert state.current_actor == "sera"


class TestAs:
    def test_switches_current_when_claimed(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        run(state.handle_line("/as sera"))
        assert state.current_actor == "sera"

    def test_refuses_unclaimed_target(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/as sera"))
        assert state.current_actor == "aldric"  # unchanged


class TestDeferCommand:
    def test_defer_submits_null_turn_for_current_actor(self, run):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(return_value=SimpleNamespace(
            beat_ended_reason="state_change",
            turn_index=3,
            output_text="",
            pre_turn_resolutions=[],
            per_player_renders={},
        ))

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/defer"))

        engine.run_turn.assert_awaited_once_with(
            session_id=SESSION_ID,
            user_input="(defer)",
            acting_character_id="aldric",
        )


@pytest.mark.skip(reason="v11: legacy v8 pipeline path; re-port against run_beat.")
class TestActingDescribe:
    def test_plain_text_acts_as_current(self, run):
        engine = _mock_engine()
        response = MagicMock()
        response.turn_index = 2
        response.output_text = "narration"
        engine.run_turn = AsyncMock(return_value=response)

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("I look around"))
        engine.run_turn.assert_awaited_once()
        call_kwargs = engine.run_turn.await_args.kwargs
        assert call_kwargs["user_input"] == "I look around"
        assert call_kwargs["acting_character_id"] == "aldric"

    def test_act_refused_without_actor(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("I look around"))
        engine.run_turn.assert_not_called()

    def test_describe_writes_to_current_actor(self, run, monkeypatch):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        # /describe is now interactive (two prompts). Stub the input
        # path so the test doesn't hang on stdin.
        inputs = iter(["Aldric", "tall and weary"])
        monkeypatch.setattr(
            "builtins.input", lambda prompt="": next(inputs),
        )
        run(state.handle_line("/describe"))
        engine.set_character_identity.assert_called_once_with(
            SESSION_ID, "aldric", name="Aldric", appearance="tall and weary",
        )

    def test_describe_preplay_auto_begins(self, run, monkeypatch):
        engine = _mock_engine()
        response = MagicMock()
        response.turn_index = 1
        response.output_text = "opening narration"
        engine.run_turn = AsyncMock(return_value=response)

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        inputs = iter(["", "tall and weary"])
        monkeypatch.setattr(
            "builtins.input", lambda prompt="": next(inputs),
        )
        run(state.handle_line("/describe"))
        engine.run_turn.assert_awaited_once()
        assert engine.run_turn.await_args.kwargs["user_input"] == "(begin)"


class TestQuit:
    def test_quit_clears_running(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/quit"))
        assert state.running is False

    def test_unknown_command_does_not_exit(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/nope"))
        assert state.running is True


class TestRewindCommand:
    """CLI dispatch tests for /rewind. The engine layer is stubbed to
    isolate the command-handler logic — actual rewind behavior is
    covered by tests/test_rewind.py."""

    def _make_engine_with_rewind(
        self, *, turns: list[int],
    ) -> MagicMock:
        engine = _mock_engine()
        engine.list_checkpoint_turns = MagicMock(return_value=turns)

        # `preview_rewind` and `rewind_session` mirror each other's
        # validation. The mocks here re-implement the bare-minimum
        # validation so the CLI handler exercises the branches it
        # needs to handle (unknown turn, target == latest, etc.) the
        # same way the real bridge would.
        from app.bot.engine_bridge import RewindResult

        def _preview(_session_id: str, target: int) -> RewindResult:
            if not turns:
                raise FileNotFoundError("no checkpoints")
            if target < 0:
                raise ValueError("must be >= 0")
            if target not in turns:
                raise ValueError(
                    f"Turn {target} has no checkpoint."
                )
            if target >= turns[-1]:
                raise ValueError(
                    f"already the current state (latest is turn {turns[-1]})"
                )
            return RewindResult(
                session_id=SESSION_ID,
                target_turn=target,
                previous_latest=turns[-1],
                new_latest=target,
                deleted_turns=[t for t in turns if t > target],
                location="hall",
                actor_character_id="aldric",
            )

        async def _rewind(_session_id: str, target: int) -> RewindResult:
            return _preview(_session_id, target)

        engine.preview_rewind = MagicMock(side_effect=_preview)
        engine.rewind_session = AsyncMock(side_effect=_rewind)
        return engine

    def test_rewind_no_arg_lists_range(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2, 3])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind"))

        out = capsys.readouterr().out
        assert "available turns: 0..3" in out
        assert "current latest: turn 3" in out
        # Bare /rewind should NOT mutate.
        engine.rewind_session.assert_not_called()

    def test_rewind_with_target_invokes_engine(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2, 3, 4])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind 2"))

        engine.preview_rewind.assert_called_once_with(SESSION_ID, 2)
        engine.rewind_session.assert_awaited_once_with(SESSION_ID, 2)
        out = capsys.readouterr().out
        assert "rewinding" in out
        assert "turn 4 → turn 2" in out
        assert "rewound to turn 2" in out
        assert "deleted 2 checkpoint(s)" in out

    def test_rewind_invalid_target_does_not_commit(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind 5"))

        engine.preview_rewind.assert_called_once_with(SESSION_ID, 5)
        engine.rewind_session.assert_not_called()
        out = capsys.readouterr().out
        assert "error" in out.lower()

    def test_rewind_to_latest_rejected_before_commit(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind 2"))

        engine.preview_rewind.assert_called_once_with(SESSION_ID, 2)
        engine.rewind_session.assert_not_called()
        out = capsys.readouterr().out
        assert "already the current state" in out

    def test_rewind_non_integer_arg_complains(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind hello"))

        engine.preview_rewind.assert_not_called()
        engine.rewind_session.assert_not_called()
        out = capsys.readouterr().out
        assert "integer" in out.lower()

    def test_rewind_empty_session_says_so(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind"))

        out = capsys.readouterr().out
        assert "no checkpoints" in out
