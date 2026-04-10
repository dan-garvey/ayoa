"""Tests for Narrator Phase 1 (adjudication)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.narrator import Narrator
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.checkpoint import CheckpointFile
from app.schemas.characters import CharacterRecord, PublicSheet, PrivateState
from app.schemas.events import CanonicalEvent, WorldAdjudication, SceneDelta
from app.schemas.narrator import TranscriptEntry
from app.schemas.state import (
    LocationState,
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)


# --- Fixtures ---

@pytest.fixture
def prompt_manager():
    return PromptManager("app/prompts")


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock()
    return client


@pytest.fixture
def sample_checkpoint():
    """A minimal but realistic checkpoint for testing."""
    return CheckpointFile(
        session=SessionState(session_id="test", turn_index=3),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="courtyard",
                scene_graph={
                    "courtyard": {
                        "name": "Estate Courtyard",
                        "description": "A wide stone courtyard with a fountain in the center.",
                        "connected_to": ["great_hall", "stables"],
                    },
                    "great_hall": {
                        "name": "The Great Hall",
                        "description": "A grand hall with high ceilings.",
                        "connected_to": ["courtyard"],
                    },
                    "stables": {
                        "name": "The Stables",
                        "description": "Hay and horses.",
                        "connected_to": ["courtyard"],
                    },
                },
            ),
            facts=[
                "The courtyard fountain is currently dry.",
                "A storm is approaching from the west.",
            ],
            physics_ruleset=PhysicsRuleset(
                strength_limits="human_baseline",
                magic_enabled=False,
            ),
            setting=StorySetting(
                genre="fantasy",
                era="medieval",
                tone="dark political intrigue",
                premise="A young lord navigates court politics.",
            ),
            lore="The kingdom has been at war for decades. Magic was banned after the Sundering.",
        ),
        characters=[
            CharacterRecord(
                character_id="guard_17",
                name="Captain Vero",
                location="courtyard",
                public_sheet=PublicSheet(
                    role="guard captain",
                    traits=["disciplined", "dry humor", "loyal"],
                ),
            ),
            CharacterRecord(
                character_id="stable_boy",
                name="Tom",
                location="stables",
                public_sheet=PublicSheet(
                    role="stable hand",
                    traits=["nervous", "young"],
                ),
            ),
        ],
        transcript=[
            TranscriptEntry(user="I enter the courtyard.", assistant="You step into the wide courtyard..."),
        ],
        config=SessionConfig(narrative_rules="Write concise prose. Never break the fourth wall."),
    )


@pytest.fixture
def feasible_event():
    return CanonicalEvent(
        event_id="evt_0003",
        user_intent="Look around the courtyard",
        world_adjudication=WorldAdjudication(
            attempted_action="Player surveys the courtyard",
            feasible=True,
            resolved_outcome="The player scans the courtyard, noting the dry fountain and the approaching storm clouds.",
        ),
        scene_delta=SceneDelta(time_advanced_seconds=5),
        observable_facts=[
            "The player looks around the courtyard slowly.",
            "The player glances at the dry fountain.",
            "The player looks up at the darkening sky.",
        ],
    )


@pytest.fixture
def infeasible_event():
    return CanonicalEvent(
        event_id="evt_0003",
        user_intent="Lift the fountain with bare hands",
        world_adjudication=WorldAdjudication(
            attempted_action="Player attempts to lift the stone fountain barehanded",
            feasible=False,
            resolved_outcome="The player grips the fountain basin and heaves. The stone does not budge. Mortar grinds under fingertips but the structure is far too heavy for any human.",
        ),
        scene_delta=SceneDelta(time_advanced_seconds=8),
        observable_facts=[
            "The player grips the fountain basin and strains upward.",
            "The fountain does not move.",
            "The player's face reddens with effort.",
            "The player releases the fountain and steps back, breathing hard.",
        ],
    )


# --- Unit tests ---

class TestNarratorContextBuilders:
    """Test the context-building helper methods."""

    def test_setting_summary(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        summary = narrator._build_setting_summary(sample_checkpoint)
        assert "fantasy" in summary
        assert "medieval" in summary
        assert "dark political intrigue" in summary

    def test_world_rules(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        rules = narrator._build_world_rules(sample_checkpoint)
        assert "human_baseline" in rules
        assert "disabled" in rules

    def test_scene_context(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        context = narrator._build_scene_context(sample_checkpoint)
        assert "Estate Courtyard" in context
        assert "fountain" in context
        assert "Great Hall" in context
        assert "Stables" in context

    def test_scene_context_empty(self, mock_client, prompt_manager):
        narrator = Narrator(mock_client, prompt_manager)
        checkpoint = CheckpointFile(session=SessionState(session_id="test"))
        context = narrator._build_scene_context(checkpoint)
        assert "No scene" in context

    def test_characters_present(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        present = narrator._build_characters_present(sample_checkpoint)
        assert "Captain Vero" in present
        # Tom is in stables, not courtyard
        assert "Tom" not in present

    def test_no_characters_present(self, mock_client, prompt_manager):
        narrator = Narrator(mock_client, prompt_manager)
        checkpoint = CheckpointFile(
            session=SessionState(session_id="test"),
            world_state=WorldState(
                locations=LocationState(current_scene_id="empty_room"),
            ),
        )
        present = narrator._build_characters_present(checkpoint)
        assert "No other characters" in present

    def test_recent_transcript(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        transcript = narrator._build_recent_transcript(sample_checkpoint)
        assert "I enter the courtyard" in transcript
        assert "You step into" in transcript

    def test_empty_transcript(self, mock_client, prompt_manager):
        narrator = Narrator(mock_client, prompt_manager)
        checkpoint = CheckpointFile(session=SessionState(session_id="test"))
        transcript = narrator._build_recent_transcript(checkpoint)
        assert "beginning" in transcript.lower()

    def test_world_facts(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        facts = narrator._build_world_facts(sample_checkpoint)
        assert "fountain is currently dry" in facts
        assert "storm" in facts


class TestNarratorPhase1:
    """Test the phase_1 method with mocked LLM."""

    @pytest.mark.asyncio
    async def test_feasible_action(
        self, mock_client, prompt_manager, sample_checkpoint, feasible_event
    ):
        mock_client.complete.return_value = LLMResponse(parsed=feasible_event)
        narrator = Narrator(mock_client, prompt_manager)

        result = await narrator.phase_1("I look around.", sample_checkpoint)

        assert result.world_adjudication.feasible is True
        assert len(result.observable_facts) == 3
        mock_client.complete.assert_called_once()

        # Verify prompt was constructed with correct template variables
        call_kwargs = mock_client.complete.call_args
        assert call_kwargs.kwargs["role"] == "narrator"
        assert call_kwargs.kwargs["response_model"] == CanonicalEvent

    @pytest.mark.asyncio
    async def test_infeasible_action(
        self, mock_client, prompt_manager, sample_checkpoint, infeasible_event
    ):
        mock_client.complete.return_value = LLMResponse(parsed=infeasible_event)
        narrator = Narrator(mock_client, prompt_manager)

        result = await narrator.phase_1(
            "I lift the fountain with my bare hands.", sample_checkpoint
        )

        assert result.world_adjudication.feasible is False
        assert "fountain" in result.world_adjudication.resolved_outcome.lower()
        assert len(result.observable_facts) >= 2

    @pytest.mark.asyncio
    async def test_event_id_assigned(
        self, mock_client, prompt_manager, sample_checkpoint
    ):
        """If LLM returns empty event_id, narrator assigns one."""
        event = CanonicalEvent(
            event_id="",
            user_intent="test",
            world_adjudication=WorldAdjudication(
                attempted_action="test",
                feasible=True,
                resolved_outcome="test",
            ),
        )
        mock_client.complete.return_value = LLMResponse(parsed=event)
        narrator = Narrator(mock_client, prompt_manager)

        result = await narrator.phase_1("test", sample_checkpoint)

        assert result.event_id == "evt_0003"  # turn_index is 3

    @pytest.mark.asyncio
    async def test_prompt_contains_world_context(
        self, mock_client, prompt_manager, sample_checkpoint, feasible_event
    ):
        """Verify the prompt sent to LLM contains relevant world context."""
        mock_client.complete.return_value = LLMResponse(parsed=feasible_event)
        narrator = Narrator(mock_client, prompt_manager)

        await narrator.phase_1("I look around.", sample_checkpoint)

        call_args = mock_client.complete.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert "fantasy" in prompt_content
        assert "Estate Courtyard" in prompt_content
        assert "Captain Vero" in prompt_content
        assert "fountain is currently dry" in prompt_content
        assert "I look around." in prompt_content
        assert "human_baseline" in prompt_content

    @pytest.mark.asyncio
    async def test_prompt_contains_narrative_rules(
        self, mock_client, prompt_manager, sample_checkpoint, feasible_event
    ):
        mock_client.complete.return_value = LLMResponse(parsed=feasible_event)
        narrator = Narrator(mock_client, prompt_manager)

        await narrator.phase_1("test", sample_checkpoint)

        call_args = mock_client.complete.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert "fourth wall" in prompt_content

    @pytest.mark.asyncio
    async def test_prompt_contains_lore(
        self, mock_client, prompt_manager, sample_checkpoint, feasible_event
    ):
        mock_client.complete.return_value = LLMResponse(parsed=feasible_event)
        narrator = Narrator(mock_client, prompt_manager)

        await narrator.phase_1("test", sample_checkpoint)

        call_args = mock_client.complete.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert "Sundering" in prompt_content
