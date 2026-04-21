"""Tests for the Character Agent engine (rolling-conversation architecture)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.character_agent import CharacterAgent
from app.engine.context_builder import (
    build_character_packet,
    build_character_state,
    build_scene_context,
    build_world_context,
    format_observed_facts,
    format_pending_observations_block,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.agents import CharacterAgentOutput, PrivateUpdates, PublicResponse
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
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


def _llm_response(parsed) -> LLMResponse:
    """Build the LLMResponse shape the new engine expects (includes raw_response.content)."""
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "{}"
    text_block.model_dump = lambda: {"type": "text", "text": "{}"}
    raw.content = [text_block]
    raw.model = "claude-haiku-4-5"
    return LLMResponse(parsed=parsed, raw_response=raw, content="{}", model="claude-haiku-4-5")


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
            secrets=["knows about the hidden passage"],
            intentions_enabled=True,
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
            current_objectives=["watch this newcomer more closely"],
        ),
    )


# --- Context builder tests ---

class TestContextBuilder:
    def test_build_character_packet_identity(self, guard_character):
        packet = build_character_packet(guard_character)
        assert packet["character_id"] == "guard_17"
        assert packet["character_name"] == "Captain Vero"
        assert "disciplined" in packet["character_traits"]
        assert "clipped" in packet["character_voice"]
        assert "twenty years" in packet["character_backstory"]
        assert "right hand twitches" in packet["character_narrative_notes"]

    def test_build_character_state_dynamic(self, guard_character):
        state = build_character_state(guard_character)
        assert "maintain order" in state["character_goals"]
        assert "hidden passage" in state["character_secrets"]

    def test_build_character_state_empty(self):
        char = CharacterRecord(character_id="minimal", name="Nobody")
        state = build_character_state(char)
        assert state["character_goals"] == "None specified."
        assert state["character_secrets"] == "None."

    def test_build_scene_context(self, sample_checkpoint):
        context = build_scene_context(sample_checkpoint)
        assert "Estate Courtyard" in context
        assert "dry fountain" in context

    def test_build_world_context_legacy_fallback(self, sample_checkpoint, guard_character):
        """Pre-v2 characters (known_context=="") fall back to global lore/facts."""
        assert guard_character.known_context == ""
        context = build_world_context(guard_character, sample_checkpoint)
        assert "fantasy" in context
        assert "fountain is dry" in context

    def test_build_world_context_uses_envelope(self, sample_checkpoint, guard_character):
        """When the character carries a known_context envelope, that IS the
        world context — global lore doesn't bleed in."""
        guard_character.known_context = "The courtyard is wet. You heard shouting earlier."
        context = build_world_context(guard_character, sample_checkpoint)
        assert context == guard_character.known_context
        # Global lore/premise absent
        assert "fantasy" not in context

    def test_format_observed_facts(self):
        facts = ["Player looks around.", "Player touches the wall."]
        formatted = format_observed_facts(facts)
        assert "Player looks around" in formatted
        assert "Player touches the wall" in formatted

    def test_format_observed_facts_empty(self):
        formatted = format_observed_facts([])
        assert "nothing unusual" in formatted

    def test_format_pending_empty(self, guard_character):
        assert format_pending_observations_block(guard_character) == ""

    def test_format_pending_nonempty(self, guard_character):
        guard_character.pending_observations = [
            "[Turn 3] A shout in the hall.",
            "[Turn 4] Footsteps receding.",
        ]
        block = format_pending_observations_block(guard_character)
        assert "Since your last response" in block
        assert "shout in the hall" in block
        assert "Footsteps receding" in block


# --- Character agent tests ---

class TestCharacterAgent:
    @pytest.mark.asyncio
    async def test_basic_response(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_output)
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
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        wrong_output = CharacterAgentOutput(
            character_id="wrong_id",
            public_response=PublicResponse(dialogue=["test"]),
        )
        mock_client.complete.return_value = _llm_response(wrong_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        result = await agent.respond(
            guard_character, ["fact"], sample_checkpoint,
        )
        assert result.character_id == "guard_17"

    @pytest.mark.asyncio
    async def test_prompt_contains_character_context(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(
            guard_character, ["Player looks around."], sample_checkpoint,
        )

        call_args = mock_client.complete.call_args
        prompt = "\n".join(
            m["content"] for m in call_args.kwargs["messages"]
            if isinstance(m["content"], str)
        )
        assert "Captain Vero" in prompt
        assert "guard captain" in prompt
        assert "clipped and formal" in prompt
        assert "hidden passage" in prompt
        assert "twenty years" in prompt
        assert "right hand twitches" in prompt

    @pytest.mark.asyncio
    async def test_prompt_contains_observed_facts(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(
            guard_character,
            ["Player picks up a rock.", "Player throws the rock at the wall."],
            sample_checkpoint,
        )

        call_args = mock_client.complete.call_args
        prompt = "\n".join(
            m["content"] for m in call_args.kwargs["messages"]
            if isinstance(m["content"], str)
        )
        assert "picks up a rock" in prompt
        assert "throws the rock" in prompt

    @pytest.mark.asyncio
    async def test_uses_agent_role(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, ["fact"], sample_checkpoint)

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["role"] == "agent"
        assert call_args.kwargs["response_model"] == CharacterAgentOutput
        assert call_args.kwargs["compact"] is True

    @pytest.mark.asyncio
    async def test_appends_to_rolling_conversation(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        assert sample_checkpoint.character_conversations == {}

        await agent.respond(
            guard_character, ["fact1"], sample_checkpoint,
        )

        convo = sample_checkpoint.character_conversations["guard_17"]
        assert len(convo) == 2
        assert convo[0].role == "user"
        assert convo[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_pending_observations_flushed_and_cleared(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output,
    ):
        guard_character.pending_observations = [
            "[Turn 2] A door slammed.",
            "[Turn 3] Footsteps in the hall.",
        ]
        mock_client.complete.return_value = _llm_response(sample_agent_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(
            guard_character, ["current fact"], sample_checkpoint,
        )

        # User message should contain the flushed observations.
        call_args = mock_client.complete.call_args
        user_msg = call_args.kwargs["messages"][-1]["content"]
        user_text = user_msg if isinstance(user_msg, str) else user_msg[0]["text"]
        assert "door slammed" in user_text
        assert "Footsteps" in user_text
        # And the pending list is cleared on the character.
        assert guard_character.pending_observations == []

    @pytest.mark.asyncio
    async def test_sends_prior_conversation_as_history(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_output,
    ):
        sample_checkpoint.character_conversations["guard_17"] = [
            ConversationMessage(role="user", content="prior user content"),
            ConversationMessage(
                role="assistant",
                content=[{"type": "text", "text": "prior response"}],
            ),
        ]
        mock_client.complete.return_value = _llm_response(sample_agent_output)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, ["fact"], sample_checkpoint)

        # Expect: system, prior user, prior assistant, current user
        messages = mock_client.complete.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "prior user content"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
