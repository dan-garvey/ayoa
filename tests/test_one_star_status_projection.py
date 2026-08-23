from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.bot.commands import _render_session_status_body_lines
from app.bot.engine_bridge import EngineBridge
from app.engine.frontend_views import SessionActivityView
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import SessionState
from scripts.play import CLIState


def _view(*, ruleset_lines: tuple[str, ...] = ()) -> SessionActivityView:
    return SessionActivityView(
        session_id="status",
        story_id="story",
        turn_index=4,
        state="Open table.",
        viewpoint_name="Seat",
        location="Lobby",
        ruleset_lines=ruleset_lines,
    )


def test_cli_and_discord_render_the_same_ruleset_status_lines(
    capsys,
) -> None:
    ruleset_lines = (
        "Wallet: Gold 37; Gems 4; Building Resources 2",
        "Hero capacity: 12; stamina 3/5",
    )
    engine = MagicMock()
    engine.session_activity.return_value = _view(ruleset_lines=ruleset_lines)
    cli = CLIState(engine=engine, session_id="status", story_id="story")

    cli.cmd_status("")
    cli_output = capsys.readouterr().out.splitlines()
    discord_output = _render_session_status_body_lines(
        engine.session_activity.return_value
    )

    for line in ruleset_lines:
        assert cli_output.count(line) == 1
        assert discord_output.count(line) == 1


def test_ruleset_status_defaults_empty_for_other_rulesets(capsys) -> None:
    engine = MagicMock()
    engine.session_activity.return_value = _view()
    cli = CLIState(engine=engine, session_id="status", story_id="story")

    cli.cmd_status("")
    output = capsys.readouterr().out

    assert "Wallet:" not in output
    assert all(
        "Wallet:" not in line
        for line in _render_session_status_body_lines(
            engine.session_activity.return_value
        )
    )


def test_engine_bridge_populates_shared_ruleset_lines(monkeypatch) -> None:
    checkpoint = CheckpointFile(
        session=SessionState(session_id="status", story_id="story")
    )
    checkpoint.session.config.settings.ruleset_id = "one_star_ascension"
    bridge = object.__new__(EngineBridge)
    bridge.checkpoint_mgr = MagicMock()
    bridge.checkpoint_mgr.load_latest.return_value = checkpoint
    expected = ("projected line one", "projected line two")
    monkeypatch.setattr(
        "app.engine.one_star_projection.one_star_status_lines",
        lambda received, viewpoint: (
            expected
            if received is checkpoint and viewpoint == ""
            else ()
        ),
    )

    activity = bridge.session_activity("status")

    assert activity.ruleset_lines == expected


def test_engine_bridge_uses_shared_master_command_projection(monkeypatch) -> None:
    checkpoint = CheckpointFile(
        session=SessionState(session_id="status", story_id="story")
    )
    bridge = object.__new__(EngineBridge)
    bridge.checkpoint_mgr = MagicMock()
    bridge.checkpoint_mgr.load_latest.return_value = checkpoint
    expected = ("Wallet: Gold 37", "highest cleared floor 4")
    projection = MagicMock(return_value=expected)
    monkeypatch.setattr(
        "app.engine.one_star_projection.one_star_master_command_lines",
        projection,
    )

    result = bridge.one_star_master_command(
        "status",
        "owner",
        "hero",
        hero_ref="Tired Baker",
    )

    assert result == expected
    projection.assert_called_once_with(
        checkpoint,
        "owner",
        "hero",
        hero_ref="Tired Baker",
    )


def test_engine_bridge_synthesis_command_uses_normal_locked_turn(monkeypatch) -> None:
    checkpoint = CheckpointFile(
        session=SessionState(session_id="status", story_id="story")
    )
    bridge = object.__new__(EngineBridge)
    bridge.checkpoint_mgr = MagicMock()
    bridge.checkpoint_mgr.load_latest.return_value = checkpoint
    bridge._session_locks = {}
    bridge._locks_mutex = asyncio.Lock()
    expected_response = object()
    bridge._run_turn_locked = AsyncMock(return_value=expected_response)
    projection = MagicMock(return_value="exact synthesis selection")
    monkeypatch.setattr(
        "app.engine.one_star_projection.one_star_synthesis_command_intention",
        projection,
    )

    result = asyncio.run(bridge.run_one_star_synthesis_command(
        "status",
        "owner",
        target_ref="Tired Baker",
        source_refs=("Edric", "Pip"),
    ))

    assert result is expected_response
    projection.assert_called_once_with(
        checkpoint,
        "owner",
        target_ref="Tired Baker",
        source_refs=("Edric", "Pip"),
    )
    bridge._run_turn_locked.assert_awaited_once_with(
        session_id="status",
        user_input="exact synthesis selection",
        acting_character_id="owner",
    )
