"""Tests for user-tunable settings + tick-driven scene creation behind
the agents_can_create_scenes flag."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.bot.engine_bridge import EngineBridge
from app.engine.orchestrator import Orchestrator
from app.engine.settings import (
    SETTINGS,
    SETTINGS_BY_KEY,
    UnknownSettingError,
    _parse_bool,
    get_setting,
    list_settings_view,
    set_setting,
)
from app.llm.config import LLMConfig
from app.schemas.agents import (
    CharacterAgentOutput,
    DirectiveSend,
    PrivateUpdates,
    PublicResponse,
)
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import SceneCreation
from app.schemas.state import LocationState, SessionState, WorldState


# ---- fixtures ---------------------------------------------------------------


SESSION_ID = "test_settings"


def _ckpt() -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            player_character_id="aldric",
        ),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="hall",
                scene_graph={
                    "hall": {
                        "name": "Great Hall",
                        "description": "",
                        "connected_to": [],
                        "properties": {},
                    },
                },
            ),
        ),
        characters=[
            CharacterRecord(
                character_id="aldric",
                name="Aldric",
                is_player=True,
                location="hall",
                public_sheet=PublicSheet(role="envoy"),
            ),
            CharacterRecord(
                character_id="regent",
                name="The Regent",
                location="hall",
                public_sheet=PublicSheet(role="regent"),
                private_state=PrivateState(intentions_enabled=True),
            ),
        ],
    )


@pytest.fixture
def bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    b = EngineBridge(saves_dir=str(tmp_path), prompts_dir="app/prompts")
    ckpt = _ckpt()
    ckpt.session.turn_index = 1
    b.checkpoint_mgr.save(ckpt)
    return b


# ---- registry internals -----------------------------------------------------


class TestSettingsRegistry:
    def test_parse_bool_accepts_common_spellings(self):
        for raw in ("true", "True", "TRUE", "yes", "on", "1", "enabled"):
            assert _parse_bool(raw) is True
        for raw in ("false", "no", "off", "0", "disabled"):
            assert _parse_bool(raw) is False

    def test_parse_bool_rejects_garbage(self):
        with pytest.raises(ValueError, match="boolean"):
            _parse_bool("maybe")

    def test_registry_is_non_empty(self):
        # If this breaks the /settings surface is empty — likely a refactor
        # removed the only setting without replacing it. Add one or remove
        # the UI.
        assert SETTINGS, "no settings registered"
        assert "agents_can_create_scenes" in SETTINGS_BY_KEY

    def test_every_spec_round_trips_its_default(self):
        """Every setting's default value should render cleanly back through
        its parser (proves parse + render are consistent)."""
        for spec in SETTINGS:
            rendered = spec.render(spec.default)
            reparsed = spec.parse(rendered)
            assert reparsed == spec.default, (
                f"{spec.key}: default {spec.default!r} renders as "
                f"{rendered!r} but re-parsing yields {reparsed!r}"
            )


class TestSettingsHelpers:
    def test_get_setting_returns_default_on_fresh_ckpt(self):
        ckpt = _ckpt()
        assert get_setting(ckpt, "agents_can_create_scenes") is False

    def test_get_unknown_raises(self):
        ckpt = _ckpt()
        with pytest.raises(UnknownSettingError):
            get_setting(ckpt, "not_a_real_setting")

    def test_set_setting_applies_parsed_value(self):
        ckpt = _ckpt()
        new = set_setting(ckpt, "agents_can_create_scenes", "on")
        assert new is True
        assert ckpt.session.config.settings.agents_can_create_scenes is True

    def test_set_setting_unknown_raises(self):
        ckpt = _ckpt()
        with pytest.raises(UnknownSettingError):
            set_setting(ckpt, "phantom", "true")

    def test_set_setting_bad_value_raises(self):
        ckpt = _ckpt()
        with pytest.raises(ValueError):
            set_setting(ckpt, "agents_can_create_scenes", "kinda")

    def test_list_view_shape(self):
        ckpt = _ckpt()
        view = list_settings_view(ckpt)
        assert len(view) == len(SETTINGS)
        keys = {row["key"] for row in view}
        assert "agents_can_create_scenes" in keys
        row = next(r for r in view if r["key"] == "agents_can_create_scenes")
        for field in (
            "key", "value", "rendered_value",
            "default", "rendered_default", "description",
        ):
            assert field in row


# ---- EngineBridge passthrough ----------------------------------------------


class TestEngineBridgeSettings:
    def test_get_returns_default(self, bridge: EngineBridge):
        assert bridge.get_setting(SESSION_ID, "agents_can_create_scenes") is False

    def test_set_persists_across_reloads(self, bridge: EngineBridge):
        bridge.set_setting(SESSION_ID, "agents_can_create_scenes", "true")
        # New handle, fresh disk read.
        assert bridge.get_setting(SESSION_ID, "agents_can_create_scenes") is True

    def test_list_has_rows(self, bridge: EngineBridge):
        rows = bridge.list_settings(SESSION_ID)
        assert len(rows) >= 1
        assert all("key" in r and "value" in r for r in rows)

    def test_known_setting_keys(self, bridge: EngineBridge):
        keys = bridge.known_setting_keys()
        assert "agents_can_create_scenes" in keys


# ---- tick-path scene creation ----------------------------------------------


def _orch() -> Orchestrator:
    """Orchestrator wired to mocks. Only state-mutation helpers are
    exercised; no LLM calls."""
    client = MagicMock()
    client.complete = AsyncMock()
    client.config = LLMConfig()
    checkpoint_mgr = MagicMock()
    prompt_mgr = MagicMock()
    return Orchestrator(client, checkpoint_mgr, prompt_mgr)


def _tick_output(character_id: str, scenes, moved_to: str = "") -> CharacterAgentOutput:
    return CharacterAgentOutput(
        character_id=character_id,
        public_response=PublicResponse(),
        private_updates=PrivateUpdates(
            current_objectives=[],
            scenes_created=list(scenes),
            moved_to=moved_to,
        ),
    )


def _fake_tick_result(regent, scenes, moved_to=""):
    """Build the (char, output, usage) triple _run_ticks assembles when
    a tick succeeds."""
    return (regent, _tick_output(regent.character_id, scenes, moved_to), {})


@pytest.mark.skip(reason="v11: legacy v8 pipeline path; re-port against run_beat.")
class TestTickSceneCreationGate:
    """The agents_can_create_scenes flag gates whether tick outputs'
    scenes_created are applied to the scene graph. We drive _run_ticks
    via a mocked agent_engine.tick to avoid an LLM call."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _arm_tick(self, orch, regent, scenes, moved_to=""):
        """Wire agent_engine.tick to return a preset output for regent."""
        output = _tick_output(regent.character_id, scenes, moved_to)

        async def _fake(char, checkpoint, acting_character_id=""):
            return output

        orch.agent_engine.tick = AsyncMock(side_effect=_fake)
        orch.agent_engine.last_usage = {}

    def test_flag_off_drops_scenes_with_warning(self, caplog):
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 4
        ckpt.session.tick_cadence = 5
        ckpt.session.tick_last_scene_id = "hall"
        # Flag default is False.
        orch = _orch()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        self._arm_tick(orch, regent, [
            SceneCreation(scene_id="study", name="Study", connected_to=["hall"]),
        ])

        with caplog.at_level(logging.WARNING):
            self._run(orch._run_ticks(ckpt, set(), "aldric"))

        # Scene NOT created.
        assert "study" not in ckpt.world_state.locations.scene_graph
        # Warning was logged.
        assert any(
            "agents_can_create_scenes" in r.message for r in caplog.records
        )

    def test_flag_on_creates_scenes_before_moved_to(self):
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 4
        ckpt.session.tick_cadence = 5
        ckpt.session.tick_last_scene_id = "hall"
        ckpt.session.config.settings.agents_can_create_scenes = True

        orch = _orch()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        # Agent creates "study" AND moves into it on the same tick. The
        # move should succeed because scene is created first.
        self._arm_tick(
            orch, regent,
            scenes=[SceneCreation(
                scene_id="study", name="Study", connected_to=["hall"],
            )],
            moved_to="study",
        )

        self._run(orch._run_ticks(ckpt, set(), "aldric"))

        assert "study" in ckpt.world_state.locations.scene_graph
        assert ckpt.world_state.locations.scene_graph["study"]["name"] == "Study"
        # Regent moved into the new scene.
        moved = next(c for c in ckpt.characters if c.character_id == "regent")
        assert moved.location == "study"
        # Bidirectional edge added.
        hall = ckpt.world_state.locations.scene_graph["hall"]
        assert "study" in hall["connected_to"]

    def test_flag_on_with_no_scenes_emitted_is_noop(self):
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 4
        ckpt.session.tick_cadence = 5
        ckpt.session.tick_last_scene_id = "hall"
        ckpt.session.config.settings.agents_can_create_scenes = True

        orch = _orch()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        self._arm_tick(orch, regent, scenes=[])

        graph_before = dict(ckpt.world_state.locations.scene_graph)
        self._run(orch._run_ticks(ckpt, set(), "aldric"))
        assert ckpt.world_state.locations.scene_graph.keys() == graph_before.keys()
