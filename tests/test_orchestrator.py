"""Tests for the Orchestrator — turn pipeline integration.

v11-A3c: the TestOrchestrator class tests reach into the v8
process_turn body (EventRouter/Agent/Narrator LLM call sequence,
per-phase latencies, pending_recap, turn_recap summarizer) that the
v11 rewrite deleted. Coverage for the new thin-wrapper shape lives in
tests/test_orchestrator_v11.py. The TestCharacterManager and
TestCharacterSpawn classes survive — those exercise CharacterManager
directly and don't depend on the orchestrator path.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.engine.character_manager import CharacterManager
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.engine.turn_recap import _TurnRecapOutput
from app.llm.client import LLMClient, LLMResponse
from app.llm.config import LLMConfig
from app.schemas.agents import CharacterAgentOutput, PublicResponse, PrivateUpdates
from app.schemas.characters import CharacterRecord, PublicSheet, PrivateState
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry, SpawnRequest
from app.schemas.events import CanonicalEvent, WorldAdjudication, SceneDelta
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.requests import TurnRequest
from app.schemas.state import LocationState, SessionState, WorldState

# --- Fixtures ---

@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock()
    client.config = LLMConfig()
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
                private_state=PrivateState(),
            ),
        ],
    )

@pytest.fixture
def mock_checkpoint_mgr(sample_checkpoint):
    mgr = MagicMock()
    mgr.load_latest.return_value = sample_checkpoint
    mgr.save.return_value = None
    return mgr

def _llm_response(parsed) -> LLMResponse:
    """Shape the LLMResponse. Character spawn parses JSON from
    response.content (structured output disabled — see benchmark), so
    content round-trips through the parsed model."""
    text = parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else "{}"
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    text_block.model_dump = lambda: {"type": "text", "text": text}
    raw.content = [text_block]
    raw.model = "claude-sonnet-4-6"
    return LLMResponse(parsed=parsed, raw_response=raw, content=text, model="claude-sonnet-4-6")

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

    def test_apply_roster_dormant(self, sample_checkpoint):
        mgr = CharacterManager()
        routed = EventRouterOutput(
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="wait", feasible=True, resolved_outcome="",
                ),
                scene_delta=SceneDelta(),
            ),
            dormant=["guard_17"],
        )
        mgr.apply_roster_updates(sample_checkpoint, routed)
        char = mgr.get_character(sample_checkpoint, "guard_17")
        assert char.status.value == "dormant"

@pytest.mark.skip(
    reason="v11: legacy orchestrator path superseded; rewrite coverage in "
    "test_orchestrator_v11.py"
)
class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_full_turn(self, mock_client, mock_checkpoint_mgr, prompt_manager):
        """Test a complete turn with mocked LLM responses (canonical merged path)."""
        # 3 LLM calls: EventRouter (merged NP1+Disc), Agent, NP2.
        merged = EventRouterOutput(
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="survey area",
                    feasible=True,
                    resolved_outcome="Player looks around.",
                ),
                observable_facts=["Player looks around."],
            ),
            observers=[
                ObserverEntry(
                    character_id="guard_17",
                    response_priority=5,
                ),
            ],
        )

        agent_out = CharacterAgentOutput(
            character_id="guard_17",
            public_response=PublicResponse(
                dialogue=["Everything alright?"],
            ),
            private_updates=PrivateUpdates(),
        )

        narrator_out = NarratorFinalOutput(
            final_text="You look around. \"Everything alright?\" Captain Vero asks.",
            transcript_entry=TranscriptEntry(
                user="I look around.",
                assistant="You look around. \"Everything alright?\" Captain Vero asks.",
            ),
        )

        mock_client.complete.side_effect = [
            _llm_response(merged),
            _llm_response(agent_out),
            _llm_response(narrator_out),
            _llm_response(_TurnRecapOutput(note="Captain Vero watched.")),
        ]

        orchestrator = Orchestrator(mock_client, mock_checkpoint_mgr, prompt_manager)
        request = TurnRequest(session_id="test-session", user_input="I look around.")

        response = await orchestrator.process_turn(request)

        assert response.session_id == "test-session"
        assert "look around" in response.output_text.lower()
        assert response.turn_index == 1
        assert response.debug is None
        # 4 LLM calls: router + agent + narrator + turn-recap summarizer.
        assert mock_client.complete.call_count == 4
        mock_checkpoint_mgr.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_turn_with_no_responders(
        self, mock_client, mock_checkpoint_mgr, prompt_manager
    ):
        """Turn where no characters respond — EventRouter returns no observers."""
        merged = EventRouterOutput(
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="internal reflection",
                    feasible=True,
                    resolved_outcome="Player thinks.",
                ),
            ),
            observers=[],
        )

        narrator_out = NarratorFinalOutput(
            final_text="You stand quietly, collecting your thoughts.",
            transcript_entry=TranscriptEntry(
                user="I think about things.",
                assistant="You stand quietly, collecting your thoughts.",
            ),
        )

        mock_client.complete.side_effect = [
            _llm_response(merged),
            _llm_response(narrator_out),
            _llm_response(_TurnRecapOutput(note="")),
        ]

        orchestrator = Orchestrator(mock_client, mock_checkpoint_mgr, prompt_manager)
        request = TurnRequest(session_id="test-session", user_input="I think about things.")

        response = await orchestrator.process_turn(request)

        assert "quietly" in response.output_text.lower()
        # 2 LLM calls: EventRouter, NP2 (no agents)
        # router + narrator + turn-recap (no agents this turn).
        assert mock_client.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_debug_mode(self, mock_client, mock_checkpoint_mgr, prompt_manager):
        """Debug mode includes internal artifacts on the merged path."""
        merged = EventRouterOutput(
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="test",
                    feasible=True,
                    resolved_outcome="test",
                ),
            ),
            observers=[],
        )

        narrator_out = NarratorFinalOutput(
            final_text="Test.",
            transcript_entry=TranscriptEntry(user="test", assistant="Test."),
        )

        mock_client.complete.side_effect = [
            _llm_response(merged),
            _llm_response(narrator_out),
            _llm_response(_TurnRecapOutput(note="")),
        ]

        orchestrator = Orchestrator(mock_client, mock_checkpoint_mgr, prompt_manager)
        request = TurnRequest(
            session_id="test-session", user_input="test", debug=True
        )

        response = await orchestrator.process_turn(request)

        assert response.debug is not None
        assert "world_adjudication" in response.debug.canonical_event
        assert "observers" in response.debug.router_output
        assert response.debug.total_duration_ms > 0
        assert any(lat.phase == "event_router" for lat in response.debug.latencies)
        assert "event_router" in response.debug.models_used
        assert isinstance(response.debug.validations, list)

class TestCharacterSpawn:
    @pytest.mark.asyncio
    async def test_spawn_character(self, mock_client, sample_checkpoint):
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Tom the Stablehand",
            location="courtyard",
            role="stablehand",
            appearance="", faction="", backstory="",
            personality="Nervous, avoids eye contact.",
            known_context="", goals=[], current_objectives=[],
            secrets=[], intentions_enabled=False,
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="stablehand_01", seed={"role": "stablehand"})],
        )

        assert len(spawned) == 1
        assert spawned[0].name == "Tom the Stablehand"
        # Should be added to checkpoint
        assert mgr.get_character(sample_checkpoint, "stablehand_01") is not None

    @pytest.mark.asyncio
    async def test_skip_existing_character(self, mock_client, sample_checkpoint):
        mock_client.complete = AsyncMock()
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        # guard_17 already exists
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="guard_17", seed={"role": "guard"})],
        )

        assert len(spawned) == 0
        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_spawn_limit(self, mock_client, sample_checkpoint):
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="NPC", location="courtyard",
            role="", appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False,
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        requests = [
            SpawnRequest(character_id=f"npc_{i}", seed={"role": f"npc{i}"})
            for i in range(5)
        ]
        spawned = await mgr.spawn_characters(sample_checkpoint, requests)

        # Capped at MAX_SPAWNS_PER_TURN = 3
        assert len(spawned) == 3

    def test_spawn_without_client(self, sample_checkpoint):
        mgr = CharacterManager()
        import asyncio
        spawned = asyncio.run(
            mgr.spawn_characters(
                sample_checkpoint,
                [SpawnRequest(character_id="test", seed={})],
            )
        )
        assert len(spawned) == 0
