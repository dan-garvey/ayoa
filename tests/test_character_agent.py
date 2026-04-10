"""Tests for the Character Agent engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.character_agent import CharacterAgent
from app.engine.context_builder import (
    build_character_packet,
    build_scene_context,
    build_world_context,
    format_observed_facts,
    _attitude_label,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.agents import CharacterAgentOutput, PublicResponse, PrivateUpdates
from app.schemas.characters import (
    CharacterRecord,
    CharacterMemory,
    PublicSheet,
    PrivateState,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import (
    LocationState,
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
def guard_character():
    return CharacterRecord(
        character_id="guard_17",
        name="Captain Vero",
        location="courtyard",
        public_sheet=PublicSheet(
            role="guard captain",
            traits=["disciplined", "dry humor", "loyal"],
            voice="clipped and formal",
            appearance="Tall, scarred, in polished armor",
            faction="City Watch",
        ),
        private_state=PrivateState(
            goals=["maintain order", "protect the estate"],
            attitudes={"user": -0.1, "servant_01": 0.3},
            secrets=["knows about the hidden passage"],
            intentions_enabled=True,
        ),
        memory=CharacterMemory(
            episodic=["Saw the player enter the courtyard earlier."],
        ),
        backstory="Served the estate for twenty years. Rose from foot soldier to captain.",
        personality="Stoic exterior hiding genuine care for those under his protection.",
        narrative_notes="His right hand twitches when he's lying. Respects competence.",
    )


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
                        "description": "A wide stone courtyard with a dry fountain.",
                    },
                },
            ),
            setting=StorySetting(
                genre="fantasy",
                tone="dark intrigue",
                premise="A young lord navigates court politics.",
            ),
            facts=["The courtyard fountain is dry.", "A storm approaches."],
        ),
    )


@pytest.fixture
def sample_agent_output():
    return CharacterAgentOutput(
        character_id="guard_17",
        public_response=PublicResponse(
            actions=["steps closer, hand resting on sword pommel"],
            dialogue=["You'll want to head inside. Storm's coming."],
            expression="eyes narrow slightly, scanning the perimeter",
        ),
        private_updates=PrivateUpdates(
            intentions=["watch this newcomer more closely"],
            attitude_delta={"user": -0.05},
        ),
        memory_writes=["The player looked around the courtyard. Seemed lost."],
    )


# --- Context builder tests ---

class TestContextBuilder:
    def test_build_character_packet(self, guard_character):
        packet = build_character_packet(guard_character)
        assert packet["character_id"] == "guard_17"
        assert packet["character_name"] == "Captain Vero"
        assert "disciplined" in packet["character_traits"]
        assert "clipped" in packet["character_voice"]
        assert "hidden passage" in packet["character_secrets"]
        assert "user" in packet["character_attitudes"]
        assert "twenty years" in packet["character_backstory"]
        assert "right hand twitches" in packet["character_narrative_notes"]

    def test_build_character_packet_with_memories(self, guard_character):
        packet = build_character_packet(guard_character)
        assert "enter the courtyard" in packet["character_memories"]

    def test_build_character_packet_empty(self):
        char = CharacterRecord(character_id="minimal", name="Nobody")
        packet = build_character_packet(char)
        assert packet["character_goals"] == "None specified."
        assert packet["character_secrets"] == "None."
        assert "No prior memories" in packet["character_memories"]

    def test_build_scene_context(self, sample_checkpoint):
        context = build_scene_context(sample_checkpoint)
        assert "Estate Courtyard" in context
        assert "dry fountain" in context

    def test_build_world_context(self, sample_checkpoint):
        context = build_world_context(sample_checkpoint)
        assert "fantasy" in context
        assert "fountain is dry" in context

    def test_format_observed_facts(self):
        facts = ["Player looks around.", "Player touches the wall."]
        formatted = format_observed_facts(facts)
        assert "Player looks around" in formatted
        assert "Player touches the wall" in formatted

    def test_format_observed_facts_empty(self):
        formatted = format_observed_facts([])
        assert "nothing unusual" in formatted

    def test_attitude_labels(self):
        assert _attitude_label(0.8) == "strong affinity"
        assert _attitude_label(0.5) == "positive"
        assert _attitude_label(0.15) == "mildly positive"
        assert _attitude_label(0.0) == "neutral"
        assert _attitude_label(-0.2) == "mildly negative"
        assert _attitude_label(-0.5) == "negative"
        assert _attitude_label(-0.8) == "hostile"


# --- Character agent tests ---

class TestCharacterAgent:
    @pytest.mark.asyncio
    async def test_basic_response(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output
    ):
        mock_client.complete.return_value = LLMResponse(parsed=sample_agent_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        result = await agent.respond(
            guard_character,
            ["Player looks around the courtyard."],
            sample_checkpoint,
        )

        assert result.character_id == "guard_17"
        assert len(result.public_response.dialogue) == 1
        assert "storm" in result.public_response.dialogue[0].lower()

    @pytest.mark.asyncio
    async def test_character_id_enforced(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint
    ):
        """Agent should enforce character_id even if LLM returns wrong one."""
        wrong_output = CharacterAgentOutput(
            character_id="wrong_id",
            public_response=PublicResponse(dialogue=["test"]),
        )
        mock_client.complete.return_value = LLMResponse(parsed=wrong_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        result = await agent.respond(
            guard_character, ["fact"], sample_checkpoint
        )
        assert result.character_id == "guard_17"

    @pytest.mark.asyncio
    async def test_prompt_contains_character_context(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output
    ):
        mock_client.complete.return_value = LLMResponse(parsed=sample_agent_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(
            guard_character,
            ["Player looks around."],
            sample_checkpoint,
        )

        call_args = mock_client.complete.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Captain Vero" in prompt
        assert "guard captain" in prompt
        assert "clipped and formal" in prompt
        assert "hidden passage" in prompt
        assert "twenty years" in prompt
        assert "right hand twitches" in prompt

    @pytest.mark.asyncio
    async def test_prompt_contains_observed_facts(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output
    ):
        mock_client.complete.return_value = LLMResponse(parsed=sample_agent_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(
            guard_character,
            ["Player picks up a rock.", "Player throws the rock at the wall."],
            sample_checkpoint,
        )

        call_args = mock_client.complete.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "picks up a rock" in prompt
        assert "throws the rock" in prompt

    @pytest.mark.asyncio
    async def test_uses_agent_role(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output
    ):
        mock_client.complete.return_value = LLMResponse(parsed=sample_agent_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, ["fact"], sample_checkpoint)

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["role"] == "agent"
        assert call_args.kwargs["response_model"] == CharacterAgentOutput
