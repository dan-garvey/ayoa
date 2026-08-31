"""Tests for character-manager behavior used by the orchestrator path."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.character_manager import (
    CastingBrief,
    CastingPlan,
    CharacterManager,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.llm.config import LLMConfig
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterStatus,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import SpawnRequest
from app.schemas.state import (
    CharacterGenerationGuidance,
    KnowledgeTier,
    SessionState,
    WorldState,
)
from tests.support.factories import router_output

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
    raw.model = "claude-sonnet-5"
    return LLMResponse(parsed=parsed, raw_response=raw, content=text, model="claude-sonnet-5")


def _messages_text(messages: list[dict]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def _dnd_statblock(**overrides):
    data = {
        "size": "Medium",
        "creature_type": "humanoid",
        "alignment": "neutral",
        "armor_class": 12,
        "hit_points": 17,
        "hit_dice": "3d8+3",
        "speed": "30 ft.",
        "ability_scores": {
            "strength": 10,
            "dexterity": 12,
            "constitution": 12,
            "intelligence": 11,
            "wisdom": 13,
            "charisma": 10,
        },
        "proficiency_bonus": 2,
        "skills": [{"name": "perception", "value": 3}],
        "senses": ["passive Perception 13"],
        "passive_perception": 13,
        "languages": ["Common"],
        "challenge_rating": "",
        "xp": 0,
        "traits": [],
        "actions": [
            {
                "action_id": "ledger_cudgel",
                "name": "Ledger Cudgel",
                "attack_bonus": 2,
                "reach_ft": 5,
                "range_normal_ft": 0,
                "range_long_ft": 0,
                "target": "one target",
                "damage": "1d4 bludgeoning",
                "damage_type": "bludgeoning",
                "description": "Swings a heavy ledger as an improvised club.",
            }
        ],
    }
    data.update(overrides)
    return data


def _spawn_request(
    character_id: str,
    *,
    seed: dict | None = None,
) -> SpawnRequest:
    seed_data = {
        "role": "",
        "reason": "",
        "location": "",
        "objectives": [],
        "knowledge_tier": 0,
    }
    seed_data.update(seed or {})
    return SpawnRequest(character_id=character_id, seed=seed_data)

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
        routed = router_output(facts=[], dormant=["guard_17"])
        mgr.apply_roster_updates(sample_checkpoint, routed)
        char = mgr.get_character(sample_checkpoint, "guard_17")
        assert char.status.value == "dormant"

    def test_dormant_cannot_downgrade_terminal_cull(self, sample_checkpoint):
        mgr = CharacterManager()
        char = mgr.get_character(sample_checkpoint, "guard_17")
        assert char is not None
        char.status = CharacterStatus.culled

        mgr.apply_roster_updates(
            sample_checkpoint,
            router_output(facts=[], dormant=["guard_17"]),
        )

        assert char.status == CharacterStatus.culled

class TestCharacterSpawn:
    @pytest.mark.asyncio
    async def test_spawn_wave_plans_once_and_authors_in_parallel(
        self, mock_client, sample_checkpoint,
    ):
        from app.schemas.takeover import AuthoredCharacter

        requests = [
            _spawn_request(character_id="first_arrival", seed={"role": "scout"}),
            _spawn_request(character_id="second_arrival", seed={"role": "porter"}),
        ]
        plan = CastingPlan(
            briefs=[
                CastingBrief(character_id="first_arrival", brief="A wary scout with a blue scarf."),
                CastingBrief(character_id="second_arrival", brief="A broad porter with a brass tally."),
            ]
        )
        authored_by_id = {
            request.character_id: AuthoredCharacter(
                name=request.character_id,
                location="courtyard",
                role=request.seed.role,
                appearance="Distinct appearance.",
                public_context="",
                default_loadout="Distinct loadout.",
                faction="",
                actor={"may_act_offstage": False, "facts": []},
                router_summary="",
            )
            for request in requests
        }
        entered: list[str] = []
        all_entered = asyncio.Event()
        generation_messages: dict[str, list[dict]] = {}

        async def complete(**kwargs):
            if kwargs["response_model"] is CastingPlan:
                return _llm_response(plan)
            text = kwargs["messages"][-1]["content"]
            character_id = next(
                request.character_id
                for request in requests
                if f"Character ID: {request.character_id}" in text
            )
            generation_messages[character_id] = kwargs["messages"]
            entered.append(character_id)
            if len(entered) == len(requests):
                all_entered.set()
            await asyncio.wait_for(all_entered.wait(), timeout=1)
            return _llm_response(authored_by_id[character_id])

        mock_client.complete.side_effect = complete
        manager = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await manager.spawn_characters(sample_checkpoint, requests)

        assert [character.character_id for character in spawned] == [
            "first_arrival",
            "second_arrival",
        ]
        assert [character.character_id for character in sample_checkpoint.characters] == [
            "guard_17",
            "first_arrival",
            "second_arrival",
        ]
        assert mock_client.complete.await_count == 3
        assert mock_client.complete.await_args_list[0].kwargs["response_model"] is CastingPlan
        for messages in generation_messages.values():
            prompt = messages[-1]["content"]
            assert "first_arrival: A wary scout with a blue scarf." in prompt
            assert "second_arrival: A broad porter with a brass tally." in prompt

    @pytest.mark.asyncio
    async def test_spawn_wave_failure_keeps_live_roster_unchanged(
        self, mock_client, sample_checkpoint,
    ):
        requests = [
            _spawn_request(character_id="successful_arrival", seed={"role": "scout"}),
            _spawn_request(character_id="failed_arrival", seed={"role": "porter"}),
        ]
        plan = CastingPlan(
            briefs=[
                CastingBrief(character_id=request.character_id, brief="Distinct arrival.")
                for request in requests
            ]
        )
        from app.schemas.takeover import AuthoredCharacter

        authored = AuthoredCharacter(
            name="Successful",
            location="courtyard",
            role="scout",
            appearance="Distinct appearance.",
            public_context="",
            default_loadout="Distinct loadout.",
            faction="",
            actor={"may_act_offstage": False, "facts": []},
            router_summary="",
        )

        async def complete(**kwargs):
            if kwargs["response_model"] is CastingPlan:
                return _llm_response(plan)
            if "failed_arrival" in kwargs["messages"][-1]["content"]:
                raise RuntimeError("generation failed")
            return _llm_response(authored)

        mock_client.complete.side_effect = complete
        manager = CharacterManager(mock_client, PromptManager("app/prompts"))

        with pytest.raises(RuntimeError, match="generation failed"):
            await manager.spawn_characters(sample_checkpoint, requests)

        assert [character.character_id for character in sample_checkpoint.characters] == [
            "guard_17",
        ]

    @pytest.mark.asyncio
    async def test_spawn_character(self, mock_client, sample_checkpoint):
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Tom the Stablehand",
            location="courtyard",
            role="stablehand",
            appearance="", public_context="", default_loadout="", faction="",
            actor={"may_act_offstage": False, "facts": []},
            router_summary="Nervous stablehand at the courtyard, watching the gate.",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [_spawn_request(character_id="stablehand_01", seed={"role": "stablehand"})],
        )

        assert len(spawned) == 1
        assert spawned[0].name == "Tom the Stablehand"
        assert spawned[0].agent_tier == CharacterAgentTier.utility
        kwargs = mock_client.complete.call_args.kwargs
        assert kwargs["role"] == "character_manager"
        assert kwargs["response_model"] is AuthoredCharacter
        assert "dnd_statblock" not in _messages_text(kwargs["messages"])
        assert "## Tier Authoring Guidance" not in kwargs["messages"][1]["content"]
        assert "## Knowledge Boundary" not in kwargs["messages"][1]["content"]
        assert "gacha" not in _messages_text(kwargs["messages"]).lower()
        # Should be added to checkpoint
        assert mgr.get_character(sample_checkpoint, "stablehand_01") is not None

    @pytest.mark.asyncio
    async def test_story_authoring_guidance_renders_in_user_tail(
        self, mock_client, sample_checkpoint,
    ):
        from app.schemas.takeover import AuthoredCharacter

        sample_checkpoint.world_state.knowledge_tiers = [
            KnowledgeTier(
                tier=2,
                label="court initiate",
                personal_depth="Remembers a household apprenticeship.",
                world_knowledge="Knows the public etiquette of the river court.",
                generation_guidance=CharacterGenerationGuidance(
                    actor_fact_guidance=(
                        "Use only a few concrete facts justified by this rung."
                    ),
                    public_visual_detail="A stable face, build, palette, and silhouette.",
                    loadout_detail="Weathered silk with one carefully finished tool.",
                    visual_salience="A clear secondary figure in ensemble scenes.",
                    presentation_guidance="Story-local mature romantic-drama casting.",
                ),
                agent_tier=CharacterAgentTier.standard,
            )
        ]
        authored = AuthoredCharacter(
            name="Tarin Vale",
            location="courtyard",
            role="court initiate",
            appearance="",
            public_context="",
            default_loadout="",
            faction="",
            actor={"may_act_offstage": False, "facts": []},
            router_summary="",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [
                _spawn_request(
                    character_id="tarin_vale",
                    seed={
                        "role": "court initiate",
                        "knowledge_tier": 2,
                    },
                )
            ],
        )

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_text = messages[0]["content"]
        user_text = messages[1]["content"]
        assert "## Tier Authoring Guidance (authoritative)" in user_text
        assert "Use only a few concrete facts justified by this rung." in user_text
        assert "Weathered silk with one carefully finished tool." in user_text
        assert "Use only a few concrete facts justified by this rung." not in system_text
        assert "Weathered silk with one carefully finished tool." not in system_text
        assert spawned[0].agent_tier == CharacterAgentTier.standard

    @pytest.mark.asyncio
    async def test_dnd_spawn_character_uses_authored_statblock(
        self, mock_client, sample_checkpoint,
    ):
        from app.schemas.dnd_character_gen import AuthoredDndCharacter

        sample_checkpoint.session.config.settings.ruleset_id = "dnd5e_basic"
        mock_client.complete = AsyncMock()
        authored = AuthoredDndCharacter(
            name="Meris Venn",
            location="courtyard",
            role="custodian",
            appearance="",
            public_context="",
            default_loadout="",
            faction="",
            actor={"may_act_offstage": False, "facts": []},
            router_summary="Custodian at the courtyard.",
            dnd_statblock=_dnd_statblock(),
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [_spawn_request(character_id="npc_meris", seed={"role": "custodian"})],
        )

        assert len(spawned) == 1
        kwargs = mock_client.complete.call_args.kwargs
        assert kwargs["response_model"] is AuthoredDndCharacter
        assert "dnd_statblock" in _messages_text(kwargs["messages"])
        mechanics = spawned[0].mechanics
        assert mechanics["ruleset_id"] == "dnd5e_basic"
        assert mechanics["source"] == "character_gen_dnd_statblock"
        assert mechanics["armor_class"] == 12
        assert mechanics["hit_points"] == {
            "current": 17,
            "max": 17,
            "temporary": 0,
            "formula": "3d8+3",
        }
        assert mechanics["ability_scores"]["dex"] == 12
        assert mechanics["dnd5e_sheet"]["statblock"]["actions"][0]["name"] == (
            "Ledger Cudgel"
        )
        assert "default_combatant_profile" not in mechanics

    @pytest.mark.asyncio
    async def test_existing_character_spawn_raises(
        self, mock_client, sample_checkpoint,
    ):
        mock_client.complete = AsyncMock()
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        with pytest.raises(ValueError, match="existing ids"):
            await mgr.spawn_characters(
                sample_checkpoint,
                [_spawn_request(character_id="guard_17", seed={"role": "guard"})],
            )

        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_spawn_limit(self, mock_client, sample_checkpoint):
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="NPC", location="courtyard",
            role="", appearance="", public_context="", default_loadout="", faction="",
            actor={"may_act_offstage": False, "facts": []},
            router_summary="Generic walk-on at the courtyard.",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        requests = [
            _spawn_request(character_id=f"npc_{i}", seed={"role": f"npc{i}"})
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
                    [_spawn_request(character_id="test", seed={})],
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
            appearance="", public_context="", default_loadout="", faction="",
            actor={"may_act_offstage": False, "facts": []},
            router_summary=(
                "Exiled cartographer who has just stepped into the courtyard, "
                "looking for the steward to plead her family's case."
            ),
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [_spawn_request(character_id="sera_01", seed={"role": "cartographer"})],
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
        shape the story seed writes for author-seeded NPCs at session
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
            appearance="", public_context="", default_loadout="", faction="",
            actor={"may_act_offstage": False, "facts": []},
            router_summary="Stablehand at the courtyard.",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [_spawn_request(character_id="stablehand_01", seed={"role": "stablehand"})],
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
                    _spawn_request(
                        character_id="runner_01",
                        seed={"role": "runner"},
                    ),
                    _spawn_request(
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
                    _spawn_request(character_id="runner_01", seed={"role": "r"}),
                    _spawn_request(character_id="runner_01", seed={"role": "r"}),
                    _spawn_request(character_id="runner_01", seed={"role": "r"}),
                    _spawn_request(
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
            appearance="", public_context="", default_loadout="", faction="",
            actor={"may_act_offstage": False, "facts": []},
            router_summary="A courier.",
        )
        mock_client.complete.return_value = _llm_response(authored)
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [_spawn_request(character_id="courier_01", seed={})],
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
            appearance="", public_context="", default_loadout="", faction="",
            actor={"may_act_offstage": False, "facts": []}, router_summary="Steward.",
        )
        mock_client.complete.return_value = _llm_response(authored)
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [_spawn_request(
                character_id="steward_01",
                seed={"location": "chapel", "role": "steward"},
            )],
            acting_actor_location="library",
        )
        assert spawned[0].location == "chapel"
