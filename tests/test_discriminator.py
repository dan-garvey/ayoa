"""Tests for the Discriminator (perception gating)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.discriminator import Discriminator
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.checkpoint import CheckpointFile
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.discriminator import DiscriminatorOutput, ObserverEntry, SpawnRequest
from app.schemas.events import CanonicalEvent, WorldAdjudication, SceneDelta
from app.schemas.state import (
    LocationState,
    PhysicsRuleset,
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
    """Checkpoint with characters at different locations."""
    return CheckpointFile(
        session=SessionState(session_id="test", turn_index=3),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="courtyard",
                scene_graph={
                    "courtyard": {
                        "name": "Estate Courtyard",
                        "description": "A wide stone courtyard.",
                        "connected_to": ["great_hall", "stables"],
                    },
                    "great_hall": {
                        "name": "The Great Hall",
                        "description": "A grand hall.",
                        "connected_to": ["courtyard"],
                    },
                    "stables": {
                        "name": "The Stables",
                        "description": "Hay and horses.",
                        "connected_to": ["courtyard"],
                    },
                    "tower": {
                        "name": "The Tower",
                        "description": "A distant watchtower.",
                        "connected_to": ["great_hall"],
                    },
                },
            ),
            setting=StorySetting(genre="fantasy", tone="dark intrigue"),
        ),
        characters=[
            CharacterRecord(
                character_id="guard_17",
                name="Captain Vero",
                location="courtyard",
                public_sheet=PublicSheet(role="guard captain", traits=["disciplined"]),
            ),
            CharacterRecord(
                character_id="servant_01",
                name="Elena",
                location="great_hall",
                public_sheet=PublicSheet(role="head servant", traits=["quiet"]),
            ),
            CharacterRecord(
                character_id="merchant_01",
                name="Gruff Aldric",
                location="tower",
                public_sheet=PublicSheet(role="visiting merchant", traits=["shrewd"]),
            ),
        ],
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
            "The player looks around the courtyard slowly.",
            "The player glances at the dry fountain.",
            "The player looks up at the darkening sky.",
        ],
    )


@pytest.fixture
def sample_discriminator_output():
    return DiscriminatorOutput(
        event_id="evt_0003",
        observers=[
            ObserverEntry(
                character_id="guard_17",
                observation_level="direct",
                facts=[
                    "The player looks around the courtyard slowly.",
                    "The player glances at the dry fountain.",
                    "The player looks up at the darkening sky.",
                ],
                response_priority=2,
            ),
        ],
        spawn=[],
        dormant=[],
        cull=[],
    )


# --- Tests ---

class TestDiscriminatorContextBuilders:
    """Test context-building helpers."""

    def test_character_registry_proximity(
        self, mock_client, prompt_manager, sample_checkpoint
    ):
        disc = Discriminator(mock_client, prompt_manager)
        registry = disc._build_character_registry(sample_checkpoint)

        assert "SAME LOCATION" in registry  # guard in courtyard
        assert "ADJACENT" in registry  # servant in great_hall
        assert "DISTANT" in registry  # merchant in tower

    def test_character_registry_includes_all(
        self, mock_client, prompt_manager, sample_checkpoint
    ):
        disc = Discriminator(mock_client, prompt_manager)
        registry = disc._build_character_registry(sample_checkpoint)

        assert "Captain Vero" in registry
        assert "Elena" in registry
        assert "Gruff Aldric" in registry
        assert "guard_17" in registry

    def test_scene_context(self, mock_client, prompt_manager, sample_checkpoint):
        disc = Discriminator(mock_client, prompt_manager)
        context = disc._build_scene_context(sample_checkpoint)

        assert "Estate Courtyard" in context
        assert "courtyard" in context
        assert "Great Hall" in context
        assert "Stables" in context

    def test_format_event(self, mock_client, prompt_manager, sample_event):
        disc = Discriminator(mock_client, prompt_manager)
        formatted = disc._format_event(sample_event)

        assert "evt_0003" in formatted
        assert "Look around the courtyard" in formatted
        assert "observable_facts" in formatted

    def test_empty_registry(self, mock_client, prompt_manager):
        disc = Discriminator(mock_client, prompt_manager)
        checkpoint = CheckpointFile(session=SessionState(session_id="test"))
        registry = disc._build_character_registry(checkpoint)
        assert "No characters" in registry


class TestDiscriminatorRun:
    """Test the run method with mocked LLM."""

    @pytest.mark.asyncio
    async def test_basic_run(
        self, mock_client, prompt_manager, sample_checkpoint,
        sample_event, sample_discriminator_output
    ):
        mock_client.complete.return_value = LLMResponse(
            parsed=sample_discriminator_output
        )
        disc = Discriminator(mock_client, prompt_manager)

        result = await disc.run(sample_event, sample_checkpoint)

        assert len(result.observers) == 1
        assert result.observers[0].character_id == "guard_17"
        assert result.observers[0].observation_level == "direct"
        assert result.event_id == "evt_0003"

    @pytest.mark.asyncio
    async def test_event_id_propagated(
        self, mock_client, prompt_manager, sample_checkpoint, sample_event
    ):
        output = DiscriminatorOutput(event_id="", observers=[])
        mock_client.complete.return_value = LLMResponse(parsed=output)
        disc = Discriminator(mock_client, prompt_manager)

        result = await disc.run(sample_event, sample_checkpoint)
        assert result.event_id == "evt_0003"

    @pytest.mark.asyncio
    async def test_spawn_requests(
        self, mock_client, prompt_manager, sample_checkpoint, sample_event
    ):
        output = DiscriminatorOutput(
            event_id="evt_0003",
            observers=[],
            spawn=[
                SpawnRequest(
                    character_id="stablehand_01",
                    seed={"role": "stablehand", "location": "stables"},
                )
            ],
        )
        mock_client.complete.return_value = LLMResponse(parsed=output)
        disc = Discriminator(mock_client, prompt_manager)

        result = await disc.run(sample_event, sample_checkpoint)
        assert len(result.spawn) == 1
        assert result.spawn[0].character_id == "stablehand_01"

    @pytest.mark.asyncio
    async def test_prompt_contains_context(
        self, mock_client, prompt_manager, sample_checkpoint,
        sample_event, sample_discriminator_output
    ):
        mock_client.complete.return_value = LLMResponse(
            parsed=sample_discriminator_output
        )
        disc = Discriminator(mock_client, prompt_manager)

        await disc.run(sample_event, sample_checkpoint)

        call_args = mock_client.complete.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert "Captain Vero" in prompt_content
        assert "SAME LOCATION" in prompt_content
        assert "DISTANT" in prompt_content
        assert "evt_0003" in prompt_content
        assert "courtyard" in prompt_content

    @pytest.mark.asyncio
    async def test_uses_discriminator_role(
        self, mock_client, prompt_manager, sample_checkpoint,
        sample_event, sample_discriminator_output
    ):
        mock_client.complete.return_value = LLMResponse(
            parsed=sample_discriminator_output
        )
        disc = Discriminator(mock_client, prompt_manager)

        await disc.run(sample_event, sample_checkpoint)

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["role"] == "discriminator"
        assert call_args.kwargs["response_model"] == DiscriminatorOutput
