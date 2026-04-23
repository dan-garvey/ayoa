"""Tests for the opening-turn / scene-population pipeline fixes:

- Player character's location follows the scene transition emitted by
  the router (previously only current_scene_id moved).
- Router receives an opening_directive only on the very first turn
  (session_conversation empty).

The broader semantic fix — narrator no longer transcribing dialogue
from the opening prose, router now responsible for placing characters
the opening names — is prompt-level and hard to unit-test without an
LLM. Covered by playtest observation instead.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.orchestrator import Orchestrator
from app.llm.config import LLMConfig
from app.schemas.agents import CharacterAgentOutput, PrivateUpdates, PublicResponse
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import CanonicalEvent, SceneDelta, WorldAdjudication
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.requests import TurnRequest
from app.schemas.state import LocationState, SessionState, WorldState

def _ckpt(*, player_location: str = "hall") -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id="test_pop",
            player_character_id="aldric",
        ),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="hall",
                scene_graph={
                    "hall": {"name": "Hall", "description": "", "connected_to": ["garden"], "properties": {}},
                    "garden": {"name": "Garden", "description": "", "connected_to": ["hall"], "properties": {}},
                },
            ),
        ),
        characters=[
            CharacterRecord(
                character_id="aldric",
                name="Aldric",
                is_player=True,
                location=player_location,
                public_sheet=PublicSheet(role="envoy"),
            ),
        ],
    )

def _llm_response(parsed):
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "{}"
    text_block.model_dump = lambda: {"type": "text", "text": "{}"}
    raw.content = [text_block]
    raw.model = "claude-sonnet-4-6"
    from app.llm.client import LLMResponse
    return LLMResponse(parsed=parsed, raw_response=raw, content="{}", model="claude-sonnet-4-6")

@pytest.mark.skip(reason="v11: legacy v8 pipeline path; re-port against run_beat.")
class TestPlayerLocationFollowsScene:
    """Core fix: when scene_delta.new_scene_id fires, the acting
    character's location updates to match. Without this, the player is
    structurally stranded at their starting location while the world's
    current_scene_id moves on."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_valid_transition_updates_player_location(self):
        from app.engine.prompt_manager import PromptManager
        ckpt = _ckpt(player_location="hall")
        mock_client = MagicMock()
        mock_client.complete = AsyncMock()
        mock_client.config = LLMConfig()
        mock_mgr = MagicMock()
        mock_mgr.load_latest.return_value = ckpt
        mock_mgr.save.return_value = None

        merged = EventRouterOutput(
            event_id="",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="go to the garden",
                    feasible=True,
                    resolved_outcome="Aldric steps out into the garden.",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0, new_scene_id="garden"),
                observable_facts=["Aldric steps into the garden."],
            ),
            observers=[],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="",
            spawn=[],
            dormant=[],
            cull=[],
            roster_moves=[],
            scenes_created=[],
        )
        narrator_out = NarratorFinalOutput(
            final_text="You walk into the garden.",
            transcript_entry=TranscriptEntry(
                user="go to garden", assistant="You walk into the garden.",
            ),
        )
        mock_client.complete.side_effect = [
            _llm_response(merged), _llm_response(narrator_out),
        ]

        orch = Orchestrator(mock_client, mock_mgr, PromptManager("app/prompts"))
        self._run(orch.process_turn(TurnRequest(
            session_id="test_pop", user_input="go to garden",
        )))

        # Both the world scene AND the player's location should be garden.
        assert ckpt.world_state.locations.current_scene_id == "garden"
        aldric = next(c for c in ckpt.characters if c.character_id == "aldric")
        assert aldric.location == "garden"

    def test_invalid_transition_leaves_location_alone(self):
        """When the router emits an unknown scene_id that wasn't created,
        neither the world scene nor the player location changes."""
        from app.engine.prompt_manager import PromptManager
        ckpt = _ckpt(player_location="hall")
        mock_client = MagicMock()
        mock_client.complete = AsyncMock()
        mock_client.config = LLMConfig()
        mock_mgr = MagicMock()
        mock_mgr.load_latest.return_value = ckpt
        mock_mgr.save.return_value = None

        merged = EventRouterOutput(
            event_id="",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="teleport",
                    feasible=False,
                    resolved_outcome="Nothing happens.",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0, new_scene_id="phantom_place"),
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
            cull=[],
            roster_moves=[],
            scenes_created=[],
        )
        narrator_out = NarratorFinalOutput(
            final_text="Nothing happens.",
            transcript_entry=TranscriptEntry(user="teleport", assistant="Nothing happens."),
        )
        mock_client.complete.side_effect = [
            _llm_response(merged), _llm_response(narrator_out),
        ]

        orch = Orchestrator(mock_client, mock_mgr, PromptManager("app/prompts"))
        self._run(orch.process_turn(TurnRequest(
            session_id="test_pop", user_input="teleport",
        )))

        assert ckpt.world_state.locations.current_scene_id == "hall"
        aldric = next(c for c in ckpt.characters if c.character_id == "aldric")
        assert aldric.location == "hall"

@pytest.mark.skip(reason="v11: legacy v8 pipeline path; re-port against run_beat.")
class TestOpeningDirectivePresence:
    """The router's opening_directive template var is populated only on
    the first turn — when session_conversation is empty."""

    def test_opening_block_built_when_session_conversation_empty(self):
        from app.engine.prompt_manager import PromptManager
        # We render the template directly rather than drive the full
        # router (which calls the LLM). Simulate both states.
        pm = PromptManager("app/prompts")
        # With opening directive
        rendered = pm.render(
            "event_router",
            setting_summary="x", world_lore="x", world_rules="x",
            scene_graph="x", current_scene="x", characters_present="x",
            world_facts="x", hidden_lore="x", hidden_facts="x",
            character_registry="x", user_input="(begin)",
            acting_character_name="Aldric", player_characters_block="x",
            since_last_turn_block="",
            recent_turn_recap="",
            opening_directive=(
                "## Author's Opening Scene Guidance\n"
                "Aldric enters the hall. The Regent is already inside.\n\n"
            ),
        )
        assert "Author's Opening Scene Guidance" in rendered
        assert "Regent is already inside" in rendered

    def test_opening_block_omittable(self):
        """Non-opening turn: the directive block is empty and shouldn't
        pollute the prompt with any marker content. The phrase 'Author's
        Opening Scene Guidance' appears in rule 13 as a reference; we
        assert the block's distinctive opening-only instruction is
        absent, not the rule text."""
        from app.engine.prompt_manager import PromptManager
        pm = PromptManager("app/prompts")
        rendered = pm.render(
            "event_router",
            setting_summary="x", world_lore="x", world_rules="x",
            scene_graph="x", current_scene="x", characters_present="x",
            world_facts="x", hidden_lore="x", hidden_facts="x",
            character_registry="x", user_input="look around",
            acting_character_name="Aldric", player_characters_block="x",
            since_last_turn_block="",
            recent_turn_recap="",
            opening_directive="",
        )
        # The distinctive directive text ("This is the first turn.") only
        # appears inside a populated opening_directive block. An empty
        # directive means no such text.
        assert "This is the first turn." not in rendered
