"""Tests for Narrator Phase 2 (final composition)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.narrator import Narrator
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.agents import CharacterAgentOutput, PublicResponse, PrivateUpdates
from app.schemas.checkpoint import CheckpointFile
from app.schemas.events import CanonicalEvent, WorldAdjudication, SceneDelta
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import (
    LocationState,
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


def _llm_response(parsed) -> LLMResponse:
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "{}"
    text_block.model_dump = lambda: {"type": "text", "text": "{}"}
    raw.content = [text_block]
    raw.model = "claude-sonnet-4-6"
    return LLMResponse(parsed=parsed, raw_response=raw, content="{}", model="claude-sonnet-4-6")


@pytest.fixture
def sample_checkpoint():
    return CheckpointFile(
        session=SessionState(session_id="test"),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="courtyard",
                scene_graph={
                    "courtyard": {
                        "name": "Estate Courtyard",
                        "description": "A wide stone courtyard.",
                        "connected_to": ["great_hall"],
                    },
                },
            ),
            setting=StorySetting(genre="fantasy", tone="dark intrigue"),
        ),
        config=SessionConfig(narrative_rules="Concise prose. No clichés."),
    )


@pytest.fixture
def sample_event():
    return CanonicalEvent(
        event_id="evt_0003",
        user_intent="Look around the courtyard",
        world_adjudication=WorldAdjudication(
            attempted_action="Player surveys the courtyard",
            feasible=True,
            resolved_outcome="Player scans the area.",
        ),
        scene_delta=SceneDelta(time_advanced_seconds=5),
        observable_facts=[
            "The player looks around the courtyard.",
            "The player glances at the dry fountain.",
        ],
    )


@pytest.fixture
def sample_agent_outputs():
    return [
        CharacterAgentOutput(
            character_id="guard_17",
            public_response=PublicResponse(
                actions=["steps closer"],
                dialogue=["Storm's coming. Best head inside."],
                expression="eyes narrow, scanning the perimeter",
            ),
            private_updates=PrivateUpdates(attitude_delta={"user": -0.05}),
            memory_writes=["Saw the player looking around."],
        ),
    ]


@pytest.fixture
def sample_narrator_output():
    return NarratorFinalOutput(
        final_text=(
            "You scan the courtyard, noting the dry fountain and darkening sky. "
            'Captain Vero steps closer, eyes narrowing. "Storm\'s coming. Best head inside."'
        ),
        world_updates={},
        transcript_entry=TranscriptEntry(
            user="I look around.",
            assistant="You scan the courtyard...",
        ),
        turn_summary="Player surveyed the courtyard; Captain Vero warned about approaching storm.",
    )


# --- Tests ---

class TestNarratorPhase2:
    @pytest.mark.asyncio
    async def test_basic_composition(
        self, mock_client, prompt_manager, sample_checkpoint,
        sample_event, sample_agent_outputs, sample_narrator_output
    ):
        mock_client.complete.return_value = _llm_response(sample_narrator_output)
        narrator = Narrator(mock_client, prompt_manager)

        result = await narrator.compose(
            "I look around.",
            sample_event,
            sample_agent_outputs,
            sample_checkpoint,
        )

        assert "courtyard" in result.final_text.lower()
        assert result.transcript_entry.user == "I look around."
        assert result.turn_summary != ""

    @pytest.mark.asyncio
    async def test_prompt_contains_event(
        self, mock_client, prompt_manager, sample_checkpoint,
        sample_event, sample_agent_outputs, sample_narrator_output
    ):
        mock_client.complete.return_value = _llm_response(sample_narrator_output)
        narrator = Narrator(mock_client, prompt_manager)

        await narrator.compose(
            "I look around.", sample_event, sample_agent_outputs, sample_checkpoint
        )

        call_args = mock_client.complete.call_args
        prompt = "\n".join(
            m["content"] for m in call_args.kwargs["messages"]
            if isinstance(m["content"], str)
        )
        assert "evt_0003" in prompt
        assert "dry fountain" in prompt

    @pytest.mark.asyncio
    async def test_prompt_contains_agent_outputs(
        self, mock_client, prompt_manager, sample_checkpoint,
        sample_event, sample_agent_outputs, sample_narrator_output
    ):
        mock_client.complete.return_value = _llm_response(sample_narrator_output)
        narrator = Narrator(mock_client, prompt_manager)

        await narrator.compose(
            "I look around.", sample_event, sample_agent_outputs, sample_checkpoint
        )

        call_args = mock_client.complete.call_args
        prompt = "\n".join(
            m["content"] for m in call_args.kwargs["messages"]
            if isinstance(m["content"], str)
        )
        assert "Storm's coming" in prompt
        assert "steps closer" in prompt
        assert "guard_17" in prompt

    @pytest.mark.asyncio
    async def test_prompt_contains_narrative_rules(
        self, mock_client, prompt_manager, sample_checkpoint,
        sample_event, sample_agent_outputs, sample_narrator_output
    ):
        mock_client.complete.return_value = _llm_response(sample_narrator_output)
        narrator = Narrator(mock_client, prompt_manager)

        await narrator.compose(
            "test", sample_event, sample_agent_outputs, sample_checkpoint
        )

        call_args = mock_client.complete.call_args
        prompt = "\n".join(
            m["content"] for m in call_args.kwargs["messages"]
            if isinstance(m["content"], str)
        )
        assert "No clichés" in prompt

    @pytest.mark.asyncio
    async def test_no_agent_outputs(
        self, mock_client, prompt_manager, sample_checkpoint,
        sample_event, sample_narrator_output
    ):
        """NP2 should work with no character responses."""
        mock_client.complete.return_value = _llm_response(sample_narrator_output)
        narrator = Narrator(mock_client, prompt_manager)

        result = await narrator.compose(
            "I look around.", sample_event, [], sample_checkpoint
        )

        call_args = mock_client.complete.call_args
        prompt = "\n".join(
            m["content"] for m in call_args.kwargs["messages"]
            if isinstance(m["content"], str)
        )
        assert "No characters responded" in prompt

    @pytest.mark.asyncio
    async def test_uses_narrator_role(
        self, mock_client, prompt_manager, sample_checkpoint,
        sample_event, sample_agent_outputs, sample_narrator_output
    ):
        mock_client.complete.return_value = _llm_response(sample_narrator_output)
        narrator = Narrator(mock_client, prompt_manager)

        await narrator.compose(
            "test", sample_event, sample_agent_outputs, sample_checkpoint
        )

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["role"] == "narrator"
        assert call_args.kwargs["response_model"] == NarratorFinalOutput


class TestFormatAgentOutputs:
    def test_single_agent(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        output = CharacterAgentOutput(
            character_id="guard_17",
            public_response=PublicResponse(
                actions=["steps closer"],
                dialogue=["Watch yourself."],
                expression="stern face",
            ),
        )
        formatted = narrator._format_agent_outputs([output], sample_checkpoint)
        assert "guard_17" in formatted
        assert "steps closer" in formatted
        assert "Watch yourself" in formatted

    def test_multiple_agents(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        outputs = [
            CharacterAgentOutput(
                character_id="guard_17",
                public_response=PublicResponse(dialogue=["First."]),
            ),
            CharacterAgentOutput(
                character_id="servant_01",
                public_response=PublicResponse(dialogue=["Second."]),
            ),
        ]
        formatted = narrator._format_agent_outputs(outputs, sample_checkpoint)
        assert "First" in formatted
        assert "Second" in formatted

    def test_empty_agents(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        formatted = narrator._format_agent_outputs([], sample_checkpoint)
        assert "No characters responded" in formatted
