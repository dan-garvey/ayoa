"""Tests for character-manager behavior used by the orchestrator path."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.character_manager import CharacterManager
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.llm.config import LLMConfig
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    PublicSheet,
    PrivateState,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, SpawnRequest
from app.schemas.events import CanonicalEvent, WorldAdjudication
from app.schemas.state import SessionState, WorldState

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
        world_state=WorldState(),
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
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[],
            ),
            observers=[],
            requires_responders=False,
            required_responders=[],
            ends_beat=True,
            ends_beat_reason="",
            spawn=[],
            dormant=["guard_17"],
            cull=[],
        )
        mgr.apply_roster_updates(sample_checkpoint, routed)
        char = mgr.get_character(sample_checkpoint, "guard_17")
        assert char.status.value == "dormant"

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
            router_summary="Nervous stablehand at the courtyard, watching the gate.",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="stablehand_01", seed={"role": "stablehand"})],
        )

        assert len(spawned) == 1
        assert spawned[0].name == "Tom the Stablehand"
        assert spawned[0].agent_tier == CharacterAgentTier.utility
        assert mock_client.complete.call_args.kwargs["role"] == "agent_convenience"
        # Should be added to checkpoint
        assert mgr.get_character(sample_checkpoint, "stablehand_01") is not None

    @pytest.mark.asyncio
    async def test_existing_character_spawn_raises(
        self, mock_client, sample_checkpoint,
    ):
        mock_client.complete = AsyncMock()
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        with pytest.raises(ValueError, match="existing ids"):
            await mgr.spawn_characters(
                sample_checkpoint,
                [SpawnRequest(character_id="guard_17", seed={"role": "guard"})],
            )

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
            router_summary="Generic walk-on at the courtyard.",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        requests = [
            SpawnRequest(character_id=f"npc_{i}", seed={"role": f"npc{i}"})
            for i in range(5)
        ]
        with pytest.raises(ValueError, match="max per turn"):
            await mgr.spawn_characters(sample_checkpoint, requests)

        mock_client.complete.assert_not_called()

    def test_spawn_without_client(self, sample_checkpoint):
        mgr = CharacterManager()
        import asyncio
        with pytest.raises(RuntimeError, match="no LLM client"):
            asyncio.run(
                mgr.spawn_characters(
                    sample_checkpoint,
                    [SpawnRequest(character_id="test", seed={})],
                )
            )

    @pytest.mark.asyncio
    async def test_router_spawn_does_not_queue_engine_update(
        self, mock_client, sample_checkpoint,
    ):
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Sera the Cartographer",
            location="courtyard",
            role="cartographer",
            appearance="", faction="", backstory="",
            personality="Quiet, watchful.",
            known_context="",
            goals=[], current_objectives=[],
            secrets=[], intentions_enabled=False,
            router_summary=(
                "Exiled cartographer who has just stepped into the courtyard, "
                "looking for the steward to plead her family's case."
            ),
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="sera_01", seed={"role": "cartographer"})],
        )

        assert len(spawned) == 1
        assert sample_checkpoint.session.pending_engine_state_updates == []

    def test_router_summary_normalizes_newlines_and_truncates(self):
        from app.engine.character_manager import (
            ROUTER_SUMMARY_MAX_CHARS,
            _normalize_router_summary,
        )

        long_summary = (
            "First line.\nSecond line with extra   whitespace.\n\n"
            + ("padding " * 200)
        )
        cleaned = _normalize_router_summary(long_summary)

        assert "\n" not in cleaned
        assert len(cleaned) <= ROUTER_SUMMARY_MAX_CHARS
        assert "  " not in cleaned

    @pytest.mark.asyncio
    async def test_spawn_seeds_location_in_pending_observations(
        self, mock_client, sample_checkpoint,
    ):
        """A freshly-spawned NPC's `pending_observations` must
        carry a `[your own action] <Name> at <location>.` seed. Same
        shape the importer writes for author-seeded NPCs at session
        start. Without this seed, the spawn's first agent dispatch
        arrives with no location signal once the on-stage agent body's
        historical `## Scene` block is gone (also r10) — the inbox is
        the only channel left."""
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Tom the Stablehand",
            location="courtyard",
            role="stablehand",
            appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False,
            router_summary="Stablehand at the courtyard.",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="stablehand_01", seed={"role": "stablehand"})],
        )

        assert len(spawned) == 1
        assert spawned[0].pending_observations == [
            "[your own action] Tom the Stablehand at courtyard.",
        ]

    @pytest.mark.asyncio
    async def test_duplicate_spawn_ids_raise(
        self, mock_client, sample_checkpoint,
    ):
        mock_client.complete = AsyncMock()
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        with pytest.raises(ValueError, match="duplicate character spawns"):
            await mgr.spawn_characters(
                sample_checkpoint,
                [
                    SpawnRequest(
                        character_id="runner_01",
                        seed={"role": "runner"},
                    ),
                    SpawnRequest(
                        character_id="runner_01",
                        seed={"role": "runner"},
                    ),
                ],
            )

        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_spawn_duplicate_ids_raise_before_cap(
        self, mock_client, sample_checkpoint,
    ):
        mock_client.complete = AsyncMock()
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        with pytest.raises(ValueError, match="duplicate character spawns"):
            await mgr.spawn_characters(
                sample_checkpoint,
                [
                    SpawnRequest(character_id="runner_01", seed={"role": "r"}),
                    SpawnRequest(character_id="runner_01", seed={"role": "r"}),
                    SpawnRequest(character_id="runner_01", seed={"role": "r"}),
                    SpawnRequest(
                        character_id="unique_villain",
                        seed={"role": "v"},
                    ),
                ],
            )

        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_spawn_uses_acting_actor_location_when_seed_omits(
        self, mock_client, sample_checkpoint,
    ):
        """v11-r7i: when the router omits `seed.location`, the new
        character materializes at the acting actor's scene (passed by
        the orchestrator). Pre-r7i the fallback was the global
        global location pivot; new behavior drops the spawn at the actor's
        opaque location label."""
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        # The LLM authors location="ignored" — the engine should override
        # this with the supplied acting_actor_location, not the LLM's choice.
        authored = AuthoredCharacter(
            name="Courier",
            location="ignored_by_engine",
            role="messenger",
            appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False,
            router_summary="A courier.",
        )
        mock_client.complete.return_value = _llm_response(authored)
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="courier_01", seed={})],
            acting_actor_location="library",
        )
        assert len(spawned) == 1
        assert spawned[0].location == "library"

    @pytest.mark.asyncio
    async def test_spawn_seed_location_beats_actor_location(
        self, mock_client, sample_checkpoint,
    ):
        """When the router DOES specify `seed.location`, that wins over
        the acting actor's location — the router knows where the spawn
        should appear (e.g. recipient's scene for a courier, witness
        location for a bystander). Actor-location is only the
        fallback when seed is silent."""
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Steward", location="ignored_by_engine", role="steward",
            appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False, router_summary="Steward.",
        )
        mock_client.complete.return_value = _llm_response(authored)
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(
                character_id="steward_01",
                seed={"location": "chapel", "role": "steward"},
            )],
            acting_actor_location="library",
        )
        assert spawned[0].location == "chapel"
