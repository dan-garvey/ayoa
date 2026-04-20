"""Tests for directives, objectives, and the off-stage tick engine.

Exercises orchestrator helpers directly rather than driving the full
turn pipeline — most of what we want to verify is state-transformation
logic (depth stamping, list replacement, eligibility filtering) that
doesn't need an LLM in the loop.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.orchestrator import (
    DIRECTIVE_DEPTH_CAP,
    DIRECTIVE_DEPTH_WARN,
    Orchestrator,
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
    IncomingDirective,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import LocationState, SessionState, WorldState


def _ckpt() -> CheckpointFile:
    """Checkpoint with two NPCs (regent, lira) in a two-scene world."""
    return CheckpointFile(
        session=SessionState(
            session_id="test",
            player_character_id="johnny",
            turn_index=5,
        ),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="hall",
                scene_graph={
                    "hall": {"name": "Hall", "description": "", "connected_to": ["garden"]},
                    "garden": {"name": "Garden", "description": "", "connected_to": ["hall"]},
                },
            ),
        ),
        characters=[
            CharacterRecord(
                character_id="johnny",
                name="Johnny",
                is_player=True,
                location="hall",
                public_sheet=PublicSheet(role="envoy"),
            ),
            CharacterRecord(
                character_id="regent",
                name="The Regent",
                location="garden",
                public_sheet=PublicSheet(role="regent"),
                private_state=PrivateState(intentions_enabled=True),
            ),
            CharacterRecord(
                character_id="lira",
                name="Lira",
                location="hall",
                public_sheet=PublicSheet(role="liaison"),
                private_state=PrivateState(intentions_enabled=True),
            ),
            CharacterRecord(
                character_id="pageboy",
                name="Pageboy",
                location="hall",
                public_sheet=PublicSheet(role="page"),
                # intentions_enabled defaults to False — not tick-eligible
            ),
        ],
    )


def _orch() -> Orchestrator:
    """Orchestrator wired to mocks. Only the state-mutation helpers are
    exercised in these tests, so the client/prompts never get called."""
    client = MagicMock()
    client.complete = AsyncMock()
    client.config = LLMConfig()
    checkpoint_mgr = MagicMock()
    prompt_mgr = MagicMock()
    return Orchestrator(client, checkpoint_mgr, prompt_mgr)


def _agent_output(
    character_id: str,
    *,
    objectives: list[str] | None = None,
    directives: list[DirectiveSend] | None = None,
    moved_to: str = "",
) -> CharacterAgentOutput:
    return CharacterAgentOutput(
        character_id=character_id,
        public_response=PublicResponse(),
        private_updates=PrivateUpdates(
            current_objectives=list(objectives or []),
            directives_sent=list(directives or []),
            moved_to=moved_to,
        ),
    )


class TestObjectivesReplacement:
    def test_full_list_replaces(self):
        orch = _orch()
        ckpt = _ckpt()
        lira = next(c for c in ckpt.characters if c.character_id == "lira")
        lira.private_state.current_objectives = ["old A", "old B", "old C"]

        output = _agent_output("lira", objectives=["new X", "new Y"])
        orch._apply_agent_private_updates(ckpt, lira, output, base_depth=0)

        assert lira.private_state.current_objectives == ["new X", "new Y"]

    def test_empty_list_clears(self):
        orch = _orch()
        ckpt = _ckpt()
        lira = next(c for c in ckpt.characters if c.character_id == "lira")
        lira.private_state.current_objectives = ["stale"]

        output = _agent_output("lira", objectives=[])
        orch._apply_agent_private_updates(ckpt, lira, output, base_depth=0)

        assert lira.private_state.current_objectives == []


class TestDirectiveRouting:
    def test_routes_to_target_queue(self):
        orch = _orch()
        ckpt = _ckpt()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        output = _agent_output(
            "regent",
            directives=[DirectiveSend(to="lira", content="redirect the envoy")],
        )
        orch._apply_agent_private_updates(ckpt, regent, output, base_depth=0)

        lira = next(c for c in ckpt.characters if c.character_id == "lira")
        assert len(lira.incoming_directives) == 1
        d = lira.incoming_directives[0]
        assert d.from_character_id == "regent"
        assert d.content == "redirect the envoy"
        assert d.turn == ckpt.session.turn_index
        assert d.depth == 1  # base_depth 0 + 1

    def test_depth_increments_from_base(self):
        orch = _orch()
        ckpt = _ckpt()
        lira = next(c for c in ckpt.characters if c.character_id == "lira")
        # lira just flushed an incoming directive at depth 3; her outgoing
        # directives should stamp at depth 4.
        output = _agent_output(
            "lira",
            directives=[DirectiveSend(to="pageboy", content="fetch the letter")],
        )
        orch._apply_agent_private_updates(ckpt, lira, output, base_depth=3)

        pageboy = next(c for c in ckpt.characters if c.character_id == "pageboy")
        assert pageboy.incoming_directives[0].depth == 4

    def test_warns_above_depth_warn(self, caplog):
        orch = _orch()
        ckpt = _ckpt()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        output = _agent_output(
            "regent",
            directives=[DirectiveSend(to="lira", content="relay")],
        )
        with caplog.at_level(logging.WARNING):
            orch._apply_agent_private_updates(
                ckpt, regent, output, base_depth=DIRECTIVE_DEPTH_WARN,
            )
        assert any("Deep directive chain" in r.message for r in caplog.records)
        lira = next(c for c in ckpt.characters if c.character_id == "lira")
        assert len(lira.incoming_directives) == 1  # still routed

    def test_drops_above_depth_cap(self, caplog):
        orch = _orch()
        ckpt = _ckpt()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        output = _agent_output(
            "regent",
            directives=[DirectiveSend(to="lira", content="too deep")],
        )
        with caplog.at_level(logging.WARNING):
            orch._apply_agent_private_updates(
                ckpt, regent, output, base_depth=DIRECTIVE_DEPTH_CAP,
            )
        assert any("exceeded depth cap" in r.message for r in caplog.records)
        lira = next(c for c in ckpt.characters if c.character_id == "lira")
        assert lira.incoming_directives == []  # dropped

    def test_unknown_target_dropped(self, caplog):
        orch = _orch()
        ckpt = _ckpt()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        output = _agent_output(
            "regent",
            directives=[DirectiveSend(to="ghost_char", content="hi")],
        )
        with caplog.at_level(logging.WARNING):
            orch._apply_agent_private_updates(ckpt, regent, output, base_depth=0)
        assert any("unknown character" in r.message for r in caplog.records)


class TestSelfMovement:
    def test_valid_scene_applies(self):
        orch = _orch()
        ckpt = _ckpt()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        assert regent.location == "garden"
        output = _agent_output("regent", moved_to="hall")
        orch._apply_agent_private_updates(ckpt, regent, output, base_depth=0)
        assert regent.location == "hall"

    def test_invalid_scene_ignored(self, caplog):
        orch = _orch()
        ckpt = _ckpt()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        output = _agent_output("regent", moved_to="nonexistent_scene")
        with caplog.at_level(logging.WARNING):
            orch._apply_agent_private_updates(ckpt, regent, output, base_depth=0)
        assert regent.location == "garden"  # unchanged
        assert any("unknown scene" in r.message for r in caplog.records)

    def test_empty_moved_to_is_noop(self):
        orch = _orch()
        ckpt = _ckpt()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        output = _agent_output("regent", moved_to="")
        orch._apply_agent_private_updates(ckpt, regent, output, base_depth=0)
        assert regent.location == "garden"


class TestTickScheduler:
    @pytest.fixture
    def orch_with_mock_tick(self):
        """Orchestrator whose agent_engine.tick() returns a preset output so
        we can verify what _run_ticks did without touching the LLM."""
        orch = _orch()

        async def _fake_tick(char, checkpoint, acting_character_id=""):
            return _agent_output(char.character_id)

        orch.agent_engine.tick = AsyncMock(side_effect=_fake_tick)
        orch.agent_engine.last_usage = {}
        return orch

    def _run(self, coro):
        return asyncio.run(coro)

    def test_no_tick_below_cadence(self, orch_with_mock_tick):
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 0
        ckpt.session.tick_cadence = 5
        ckpt.session.tick_last_scene_id = "hall"
        ckpt.world_state.locations.current_scene_id = "hall"

        usages = self._run(
            orch_with_mock_tick._run_ticks(ckpt, set(), "johnny"),
        )
        assert usages == []
        assert ckpt.session.tick_turn_counter == 1  # incremented, not fired
        orch_with_mock_tick.agent_engine.tick.assert_not_called()

    def test_fires_on_cadence(self, orch_with_mock_tick):
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 4  # this call makes it 5
        ckpt.session.tick_cadence = 5
        ckpt.session.tick_last_scene_id = "hall"
        ckpt.world_state.locations.current_scene_id = "hall"

        usages = self._run(
            orch_with_mock_tick._run_ticks(ckpt, set(), "johnny"),
        )
        # regent + lira are both intentions_enabled and not acting this turn
        assert len(usages) == 2
        assert ckpt.session.tick_turn_counter == 0  # reset

    def test_fires_on_scene_change(self, orch_with_mock_tick):
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 1
        ckpt.session.tick_cadence = 5
        ckpt.session.tick_last_scene_id = "hall"
        ckpt.world_state.locations.current_scene_id = "garden"  # changed

        usages = self._run(
            orch_with_mock_tick._run_ticks(ckpt, set(), "johnny"),
        )
        assert len(usages) == 2
        assert ckpt.session.tick_last_scene_id == "garden"  # tracks new scene
        assert ckpt.session.tick_turn_counter == 0

    def test_skips_player_bound(self, orch_with_mock_tick):
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 4
        ckpt.session.tick_last_scene_id = "hall"
        ckpt.world_state.locations.current_scene_id = "hall"
        # Bind lira to a human — she should NOT tick even though she's
        # intentions_enabled.
        ckpt.session.character_bindings = {"lira": "42"}

        self._run(orch_with_mock_tick._run_ticks(ckpt, set(), "johnny"))
        calls = orch_with_mock_tick.agent_engine.tick.call_args_list
        ticked_ids = {call.args[0].character_id for call in calls}
        assert "lira" not in ticked_ids
        assert "regent" in ticked_ids

    def test_skips_already_acted(self, orch_with_mock_tick):
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 4
        ckpt.session.tick_last_scene_id = "hall"
        ckpt.world_state.locations.current_scene_id = "hall"

        self._run(
            orch_with_mock_tick._run_ticks(ckpt, {"lira"}, "johnny"),
        )
        calls = orch_with_mock_tick.agent_engine.tick.call_args_list
        ticked_ids = {call.args[0].character_id for call in calls}
        assert "lira" not in ticked_ids
        assert "regent" in ticked_ids

    def test_skips_not_intentions_enabled(self, orch_with_mock_tick):
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 4
        ckpt.session.tick_last_scene_id = "hall"
        ckpt.world_state.locations.current_scene_id = "hall"

        self._run(orch_with_mock_tick._run_ticks(ckpt, set(), "johnny"))
        calls = orch_with_mock_tick.agent_engine.tick.call_args_list
        ticked_ids = {call.args[0].character_id for call in calls}
        assert "pageboy" not in ticked_ids  # intentions_enabled=False

    def test_skips_dormant(self, orch_with_mock_tick):
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 4
        ckpt.session.tick_last_scene_id = "hall"
        ckpt.world_state.locations.current_scene_id = "hall"
        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        regent.status = CharacterStatus.dormant

        self._run(orch_with_mock_tick._run_ticks(ckpt, set(), "johnny"))
        calls = orch_with_mock_tick.agent_engine.tick.call_args_list
        ticked_ids = {call.args[0].character_id for call in calls}
        assert "regent" not in ticked_ids

    def test_seeds_tick_last_scene_on_first_call(self, orch_with_mock_tick):
        """When tick_last_scene_id is empty (brand new session), the first
        sub-cadence call should seed it so future scene-change checks work."""
        ckpt = _ckpt()
        ckpt.session.tick_turn_counter = 0
        ckpt.session.tick_last_scene_id = ""
        ckpt.world_state.locations.current_scene_id = "hall"

        self._run(orch_with_mock_tick._run_ticks(ckpt, set(), "johnny"))
        assert ckpt.session.tick_last_scene_id == "hall"
