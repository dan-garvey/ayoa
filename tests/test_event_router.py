"""Tests for the merged event router prototype."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.event_router import EventRouter
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.checkpoint import CheckpointFile
from app.schemas.characters import CharacterRecord, PublicSheet, PrivateState
from app.schemas.event_router import EventRouterOutput
from app.schemas.events import CanonicalEvent, WorldAdjudication, SceneDelta
from app.schemas.state import (
    LocationState,
    PhysicsRuleset,
    SessionState,
    StorySetting,
    WorldState,
)


@pytest.fixture
def prompt_manager():
    return PromptManager("app/prompts")


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock()
    return client


def _llm_response(parsed) -> LLMResponse:
    """Build an LLMResponse with a mocked raw_response.content so engine code
    that persists the raw content blocks can call .model_dump() on each block."""
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
                },
            ),
            setting=StorySetting(genre="fantasy", tone="dark intrigue"),
            physics_ruleset=PhysicsRuleset(magic_enabled=False),
            facts=["The courtyard fountain is dry."],
        ),
        characters=[
            CharacterRecord(
                character_id="guard_17",
                name="Captain Vero",
                location="courtyard",
                public_sheet=PublicSheet(role="guard captain", traits=["disciplined"]),
                private_state=PrivateState(attitudes={"user": 0.0}),
            ),
            CharacterRecord(
                character_id="servant_01",
                name="Elena",
                location="great_hall",
                public_sheet=PublicSheet(role="head servant", traits=["quiet"]),
            ),
        ],
    )


@pytest.fixture
def sample_router_output():
    return EventRouterOutput(
        canonical_event=CanonicalEvent(
            event_id="",
            user_intent="Look around the courtyard",
            world_adjudication=WorldAdjudication(
                attempted_action="Player surveys the courtyard",
                feasible=True,
                resolved_outcome="You scan the courtyard.",
            ),
            scene_delta=SceneDelta(time_advanced_seconds=5),
            observable_facts=[
                "The player looks around the courtyard slowly.",
                "The player glances at the dry fountain.",
            ],
        ),
    )


class TestEventRouterRun:
    @pytest.mark.asyncio
    async def test_basic_run(
        self, mock_client, prompt_manager, sample_checkpoint, sample_router_output
    ):
        mock_client.complete.return_value = _llm_response(sample_router_output)
        router = EventRouter(mock_client, prompt_manager)

        result = await router.run("I look around.", sample_checkpoint)

        assert result.canonical_event.event_id == "evt_0003"
        assert result.canonical_event.world_adjudication.feasible is True

    @pytest.mark.asyncio
    async def test_prompt_contains_routing_context(
        self, mock_client, prompt_manager, sample_checkpoint, sample_router_output
    ):
        mock_client.complete.return_value = _llm_response(sample_router_output)
        router = EventRouter(mock_client, prompt_manager)

        await router.run("I look around.", sample_checkpoint)

        call_args = mock_client.complete.call_args
        prompt = "\n".join(
            m["content"] for m in call_args.kwargs["messages"]
            if isinstance(m["content"], str)
        )
        assert "Captain Vero" in prompt
        assert "great_hall" in prompt
        assert "Estate Courtyard" in prompt
        assert "courtyard fountain" in prompt

    @pytest.mark.asyncio
    async def test_uses_event_router_role(
        self, mock_client, prompt_manager, sample_checkpoint, sample_router_output
    ):
        mock_client.complete.return_value = _llm_response(sample_router_output)
        router = EventRouter(mock_client, prompt_manager)

        await router.run("I look around.", sample_checkpoint)

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["role"] == "event_router"
        assert call_args.kwargs["response_model"] == EventRouterOutput
