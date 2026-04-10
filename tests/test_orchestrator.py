"""Tests for the Orchestrator — turn pipeline integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.engine.character_manager import CharacterManager
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.agents import CharacterAgentOutput, PublicResponse, PrivateUpdates
from app.schemas.characters import CharacterRecord, PublicSheet, PrivateState
from app.schemas.checkpoint import CheckpointFile
from app.schemas.discriminator import DiscriminatorOutput, ObserverEntry
from app.schemas.events import CanonicalEvent, WorldAdjudication, SceneDelta
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.requests import TurnRequest
from app.schemas.state import LocationState, SessionState, WorldState


# --- Fixtures ---

@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock()
    return client


@pytest.fixture
def sample_checkpoint():
    return CheckpointFile(
        session=SessionState(session_id="test-session", turn_index=0),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="courtyard",
                scene_graph={
                    "courtyard": {
                        "name": "Courtyard",
                        "description": "A stone courtyard.",
                        "connected_to": [],
                    },
                },
            ),
        ),
        characters=[
            CharacterRecord(
                character_id="guard_17",
                name="Captain Vero",
                location="courtyard",
                public_sheet=PublicSheet(role="guard captain"),
                private_state=PrivateState(attitudes={"user": 0.0}),
            ),
        ],
    )


@pytest.fixture
def mock_checkpoint_mgr(sample_checkpoint):
    mgr = MagicMock()
    mgr.load_latest.return_value = sample_checkpoint
    mgr.save.return_value = None
    return mgr


@pytest.fixture
def prompt_manager():
    return PromptManager("app/prompts")


# --- Tests ---

class TestCharacterManager:
    def test_get_character(self, sample_checkpoint):
        mgr = CharacterManager()
        char = mgr.get_character(sample_checkpoint, "guard_17")
        assert char is not None
        assert char.name == "Captain Vero"

    def test_get_missing_character(self, sample_checkpoint):
        mgr = CharacterManager()
        char = mgr.get_character(sample_checkpoint, "nonexistent")
        assert char is None

    def test_apply_attitude_delta(self, sample_checkpoint):
        mgr = CharacterManager()
        output = CharacterAgentOutput(
            character_id="guard_17",
            private_updates=PrivateUpdates(attitude_delta={"user": -0.1}),
        )
        mgr.apply_agent_output(sample_checkpoint, output)
        char = mgr.get_character(sample_checkpoint, "guard_17")
        assert char.private_state.attitudes["user"] == pytest.approx(-0.1)

    def test_apply_attitude_clamped(self, sample_checkpoint):
        mgr = CharacterManager()
        # Set initial attitude to 0.9
        char = mgr.get_character(sample_checkpoint, "guard_17")
        char.private_state.attitudes["user"] = 0.9
        # Apply delta that would exceed 1.0
        output = CharacterAgentOutput(
            character_id="guard_17",
            private_updates=PrivateUpdates(attitude_delta={"user": 0.5}),
        )
        mgr.apply_agent_output(sample_checkpoint, output)
        assert char.private_state.attitudes["user"] == 1.0

    def test_apply_memory_writes(self, sample_checkpoint):
        mgr = CharacterManager()
        output = CharacterAgentOutput(
            character_id="guard_17",
            memory_writes=["Saw the player looking around."],
        )
        mgr.apply_agent_output(sample_checkpoint, output)
        char = mgr.get_character(sample_checkpoint, "guard_17")
        assert "looking around" in char.memory.episodic[0]

    def test_apply_roster_dormant(self, sample_checkpoint):
        mgr = CharacterManager()
        disc = DiscriminatorOutput(dormant=["guard_17"])
        mgr.apply_roster_updates(sample_checkpoint, disc)
        char = mgr.get_character(sample_checkpoint, "guard_17")
        assert char.status.value == "dormant"


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_full_turn(self, mock_client, mock_checkpoint_mgr, prompt_manager):
        """Test a complete turn with mocked LLM responses."""
        # Set up sequential mock responses for the 4 LLM calls:
        # NP1, Discriminator, Agent, NP2
        event = CanonicalEvent(
            event_id="evt_0000",
            user_intent="look around",
            world_adjudication=WorldAdjudication(
                attempted_action="survey area",
                feasible=True,
                resolved_outcome="Player looks around.",
            ),
            observable_facts=["Player looks around."],
        )

        disc = DiscriminatorOutput(
            event_id="evt_0000",
            observers=[
                ObserverEntry(
                    character_id="guard_17",
                    facts=["Player looks around."],
                    should_respond=True,
                ),
            ],
        )

        agent_out = CharacterAgentOutput(
            character_id="guard_17",
            public_response=PublicResponse(
                dialogue=["Everything alright?"],
            ),
            private_updates=PrivateUpdates(attitude_delta={"user": 0.05}),
            memory_writes=["The player looked around."],
        )

        narrator_out = NarratorFinalOutput(
            final_text="You look around. \"Everything alright?\" Captain Vero asks.",
            transcript_entry=TranscriptEntry(
                user="I look around.",
                assistant="You look around. \"Everything alright?\" Captain Vero asks.",
            ),
            turn_summary="Player surveyed area; guard responded.",
        )

        mock_client.complete.side_effect = [
            LLMResponse(parsed=event),
            LLMResponse(parsed=disc),
            LLMResponse(parsed=agent_out),
            LLMResponse(parsed=narrator_out),
        ]

        orchestrator = Orchestrator(mock_client, mock_checkpoint_mgr, prompt_manager)
        request = TurnRequest(session_id="test-session", user_input="I look around.")

        response = await orchestrator.process_turn(request)

        assert response.session_id == "test-session"
        assert "look around" in response.output_text.lower()
        assert response.turn_index == 1
        assert response.debug is None
        assert mock_client.complete.call_count == 4
        mock_checkpoint_mgr.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_turn_with_no_responders(
        self, mock_client, mock_checkpoint_mgr, prompt_manager
    ):
        """Test turn where no characters respond."""
        event = CanonicalEvent(
            event_id="evt_0000",
            user_intent="think quietly",
            world_adjudication=WorldAdjudication(
                attempted_action="internal reflection",
                feasible=True,
                resolved_outcome="Player thinks.",
            ),
        )

        disc = DiscriminatorOutput(event_id="evt_0000", observers=[])

        narrator_out = NarratorFinalOutput(
            final_text="You stand quietly, collecting your thoughts.",
            transcript_entry=TranscriptEntry(
                user="I think about things.",
                assistant="You stand quietly, collecting your thoughts.",
            ),
            turn_summary="Player reflected.",
        )

        mock_client.complete.side_effect = [
            LLMResponse(parsed=event),
            LLMResponse(parsed=disc),
            LLMResponse(parsed=narrator_out),
        ]

        orchestrator = Orchestrator(mock_client, mock_checkpoint_mgr, prompt_manager)
        request = TurnRequest(session_id="test-session", user_input="I think about things.")

        response = await orchestrator.process_turn(request)

        assert "quietly" in response.output_text.lower()
        # Only 3 LLM calls: NP1, Disc, NP2 (no agents)
        assert mock_client.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_debug_mode(self, mock_client, mock_checkpoint_mgr, prompt_manager):
        """Test that debug mode includes internal artifacts."""
        event = CanonicalEvent(
            event_id="evt_0000",
            user_intent="test",
            world_adjudication=WorldAdjudication(
                attempted_action="test",
                feasible=True,
                resolved_outcome="test",
            ),
        )

        disc = DiscriminatorOutput(event_id="evt_0000")

        narrator_out = NarratorFinalOutput(
            final_text="Test.",
            transcript_entry=TranscriptEntry(user="test", assistant="Test."),
            turn_summary="test",
        )

        mock_client.complete.side_effect = [
            LLMResponse(parsed=event),
            LLMResponse(parsed=disc),
            LLMResponse(parsed=narrator_out),
        ]

        orchestrator = Orchestrator(mock_client, mock_checkpoint_mgr, prompt_manager)
        request = TurnRequest(
            session_id="test-session", user_input="test", debug=True
        )

        response = await orchestrator.process_turn(request)

        assert response.debug is not None
        assert "event_id" in response.debug.canonical_event
        assert "observers" in response.debug.discriminator
