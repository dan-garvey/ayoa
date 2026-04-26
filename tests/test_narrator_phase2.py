"""Tests for Narrator Phase 2 (final composition)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.narrator import Narrator
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.agents import CharacterAgentOutput
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
        world_adjudication=WorldAdjudication(
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
            public_text=(
                'He steps closer, eyes narrowing as he scans the perimeter. '
                "\"Storm's coming. Best head inside.\""
            ),
            intent="Push the visitor inside; weather is a convenient cover.",
        ),
    ]

@pytest.fixture
def sample_narrator_output():
    return NarratorFinalOutput(
        final_text=(
            "You scan the courtyard, noting the dry fountain and darkening sky. "
            'Captain Vero steps closer, eyes narrowing. "Storm\'s coming. Best head inside."'
        ),
    )

# --- Tests ---

@pytest.mark.skip(reason="v11: legacy v8 pipeline path; re-port against run_beat.")
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
    """Legacy v8 shim — the narrator's `_format_agent_outputs` is dead
    code in v11 (the run_beat path folds agent prose into the canonical
    event's resolved_outcome via the router). These tests pin the shim's
    behavior so a future cleanup doesn't quietly change what the helper
    returns and trip a stale call site."""

    def test_single_agent_renders_public_text_only(
        self, mock_client, prompt_manager, sample_checkpoint,
    ):
        narrator = Narrator(mock_client, prompt_manager)
        output = CharacterAgentOutput(
            character_id="guard_17",
            public_text='He steps closer, stern-faced. "Watch yourself."',
            intent="Warning, not threatening — yet.",
        )
        formatted = narrator._format_agent_outputs([output], sample_checkpoint)
        assert "guard_17" in formatted
        assert "steps closer" in formatted
        assert "Watch yourself" in formatted
        # Critical: the trailing parenthetical (intent) is private to the
        # agent + engine and must NEVER reach the narrator's input. This
        # is one of three chokepoints enforcing that contract.
        assert "Warning" not in formatted

    def test_multiple_agents(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        outputs = [
            CharacterAgentOutput(
                character_id="guard_17",
                public_text='"First."',
                intent="",
            ),
            CharacterAgentOutput(
                character_id="servant_01",
                public_text='"Second."',
                intent="",
            ),
        ]
        formatted = narrator._format_agent_outputs(outputs, sample_checkpoint)
        assert "First" in formatted
        assert "Second" in formatted

    def test_empty_agents(self, mock_client, prompt_manager, sample_checkpoint):
        narrator = Narrator(mock_client, prompt_manager)
        formatted = narrator._format_agent_outputs([], sample_checkpoint)
        assert "No characters responded" in formatted

    def test_silent_beat_fallback(
        self, mock_client, prompt_manager, sample_checkpoint,
    ):
        """If an agent emitted only a parenthetical (no public prose),
        public_text is empty. The shim renders a `(silent beat)` token
        rather than an empty section — so a downstream prompt doesn't
        contain a malformed empty character header."""
        narrator = Narrator(mock_client, prompt_manager)
        outputs = [
            CharacterAgentOutput(
                character_id="guard_17",
                public_text="",
                intent="Saying nothing on purpose.",
            ),
        ]
        formatted = narrator._format_agent_outputs(outputs, sample_checkpoint)
        assert "(silent beat)" in formatted
