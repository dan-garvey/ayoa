"""Rendered character-generation knowledge-boundary contracts."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.character_agent import model_role_for_character
from app.engine.character_manager import CharacterManager
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterStatus,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import SpawnRequest
from app.schemas.one_star import ONE_STAR_HERO_KEY
from app.schemas.one_star_character_gen import AuthoredOneStarCharacter
from app.schemas.state import (
    CharacterGenerationGuidance,
    KnowledgeTier,
    PhysicsRuleset,
    SessionState,
    StorySetting,
    WorldState,
)
from app.schemas.takeover import AuthoredCharacter


def _authored(
    *,
    known_context: str = "Knows only the granted public duty",
    **overrides: object,
) -> AuthoredCharacter:
    fields: dict[str, object] = {
        "name": "Mara Venn",
        "location": "lower_hall",
        "role": "scout",
        "appearance": "A wiry adult with cropped black hair.",
        "default_loadout": "Plain wool layers and a worn walking staff.",
        "faction": "",
        "backstory": "She remembers waking in the lower hall.",
        "personality": "Wary and direct; counts exits before speaking.",
        "known_context": known_context,
        "goals": ["Survive without abandoning the people beside her."],
        "current_objectives": ["Understand the sealed gate."],
        "secrets": [],
        "intentions_enabled": True,
        "router_summary": "",
    }
    fields.update(overrides)
    return AuthoredCharacter.model_validate(fields)


def _response(authored: AuthoredCharacter) -> LLMResponse:
    content = authored.model_dump_json()
    raw = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = content
    block.model_dump = lambda: {"type": "text", "text": content}
    raw.content = [block]
    raw.model = "offline-fixture"
    return LLMResponse(
        parsed=authored,
        raw_response=raw,
        content=content,
        model="offline-fixture",
    )


def _client(*responses: AuthoredCharacter) -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(side_effect=[_response(item) for item in responses])
    return client


def _checkpoint(*, tiered: bool = True) -> CheckpointFile:
    tiers = (
        [
            KnowledgeTier(
                tier=1,
                label="new arrival",
                personal_depth="No history before waking; one ordinary hand-skill.",
                world_knowledge=(
                    "They serve the unseen Regent, must climb the Spire, and "
                    "know the sealed brass gate does not reopen from inside."
                ),
                generation_guidance=CharacterGenerationGuidance(
                    backstory_depth="One sentence beginning with this arrival.",
                    personality_depth="One temperament and one survival strategy.",
                    public_visual_detail="Two ordinary first-look details.",
                    loadout_detail="Plain cloth and one worn tool.",
                    visual_salience="Low ensemble salience.",
                    presentation_guidance="An ordinary adult supporting figure.",
                ),
            ),
            KnowledgeTier(
                tier=2,
                label="archive initiate",
                personal_depth="Remembers an apprenticeship and a sibling.",
                world_knowledge=(
                    "They know the Glass Synthesis Chamber consumes identities "
                    "to strengthen its chosen subject."
                ),
            ),
        ]
        if tiered
        else []
    )
    checkpoint = CheckpointFile(
        session=SessionState(session_id="knowledge-test"),
        world_state=WorldState(
            setting=StorySetting(
                genre="Gothic fantasy",
                era="An iron-age city beneath an enchanted sky",
                tone="Tense, humane, and strange",
                premise=(
                    "The Regent secretly feeds memories through a credit "
                    "refinery beneath the Spire."
                ),
            ),
            lore=(
                "The Glass Synthesis Chamber and the later credit refinery "
                "power the hidden economy."
            ),
            hidden_lore="The Argent Cartel owns every recovered identity.",
            hidden_facts=["The final ascent awakens the buried king."],
            physics_ruleset=PhysicsRuleset(
                strength_limits=(
                    "Power comes from the Mnemonic Forge and its secret seals."
                ),
                magic_enabled=True,
            ),
            knowledge_tiers=tiers,
        ),
        characters=[
            CharacterRecord(
                character_id="local_guide",
                name="Orra",
                location="lower_hall",
                public_sheet=PublicSheet(role="hall guide"),
            ),
            CharacterRecord(
                character_id="remote_plotter",
                name="The Cartel Prince",
                location="upper_vault",
                public_sheet=PublicSheet(role="Argent Cartel heir"),
            ),
            CharacterRecord(
                character_id="dormant_archivist",
                name="Vey",
                status=CharacterStatus.dormant,
                location="lower_hall",
                public_sheet=PublicSheet(role="credit refinery architect"),
            ),
        ],
    )
    checkpoint.session.config.narrative_rules = (
        "Never reveal that the Glass Synthesis Chamber funds the hidden economy."
    )
    return checkpoint


def _request(*, tier: int, reason: str = "A new scout wakes in the lower hall") -> SpawnRequest:
    return SpawnRequest(
        character_id="mara_venn",
        seed={
            "role": "scout",
            "reason": reason,
            "location": "lower_hall",
            "objectives": ["Learn what the sealed gate permits"],
            "knowledge_tier": tier,
        },
    )


def _rendered_call(client: MagicMock, index: int = 0) -> tuple[str, str]:
    messages = client.complete.await_args_list[index].kwargs["messages"]
    return messages[0]["content"], messages[1]["content"]


@pytest.mark.asyncio
async def test_tier_one_receives_grant_without_broader_story_truth() -> None:
    checkpoint = _checkpoint()
    client = _client(_authored())

    spawned = await CharacterManager(
        client, PromptManager("app/prompts"),
    ).spawn_characters(checkpoint, [_request(tier=1)])

    system, user = _rendered_call(client)
    rendered = f"{system}\n{user}".casefold()
    assert all(term in rendered for term in ("unseen regent", "spire", "sealed brass gate"))
    assert all(
        term not in rendered
        for term in (
            "glass synthesis chamber",
            "credit refinery",
            "hidden economy",
            "mnemonic forge",
            "argent cartel",
        )
    )
    assert all(term in system for term in ("Gothic fantasy", "iron-age", "Tense"))
    assert "knowledge tier 1" not in system.casefold()
    assert "Character ID: mara_venn" not in system
    assert "Orra" in user
    assert "The Cartel Prince" not in user
    assert "Vey" not in user
    assert spawned[0].knowledge_tier == 1


@pytest.mark.asyncio
async def test_tiered_generation_does_not_inspect_or_retry_model_output() -> None:
    checkpoint = _checkpoint()
    client = _client(
        _authored(
            known_context=(
                "The Glass Synthesis Chamber consumes identities."
            ),
        ),
    )

    spawned = await CharacterManager(
        client, PromptManager("app/prompts"),
    ).spawn_characters(checkpoint, [_request(tier=1)])

    assert client.complete.await_count == 1
    assert spawned[0].known_context == (
        "The Glass Synthesis Chamber consumes identities."
    )


@pytest.mark.asyncio
async def test_higher_tier_and_explicit_spawn_facts_remain_available() -> None:
    higher_checkpoint = _checkpoint()
    higher_client = _client(
        _authored(known_context="The Glass Synthesis Chamber consumes identities."),
    )
    await CharacterManager(
        higher_client, PromptManager("app/prompts"),
    ).spawn_characters(higher_checkpoint, [_request(tier=2)])
    _, higher_user = _rendered_call(higher_client)
    assert "Glass Synthesis Chamber" in higher_user
    assert higher_client.complete.await_count == 1

    explicit_checkpoint = _checkpoint()
    explicit_client = _client(
        _authored(known_context="The Glass Synthesis Chamber needs a scout."),
    )
    await CharacterManager(
        explicit_client, PromptManager("app/prompts"),
    ).spawn_characters(
        explicit_checkpoint,
        [_request(
            tier=1,
            reason="The Glass Synthesis Chamber explicitly summoned this scout",
        )],
    )
    assert explicit_client.complete.await_count == 1


@pytest.mark.asyncio
async def test_tier_zero_in_tiered_story_gets_no_ladder_knowledge() -> None:
    checkpoint = _checkpoint()
    client = _client(_authored(known_context="Only this unfamiliar hall is apparent."))

    await CharacterManager(
        client, PromptManager("app/prompts"),
    ).spawn_characters(checkpoint, [_request(tier=0)])

    system, user = _rendered_call(client)
    rendered = f"{system}\n{user}".casefold()
    assert "knowledge tier 0" in user.casefold()
    assert "magic exists in this story: yes" in user.casefold()
    assert all(
        term not in rendered
        for term in ("unseen regent", "sealed brass gate", "glass synthesis chamber")
    )


@pytest.mark.asyncio
async def test_untiered_story_retains_full_generation_context() -> None:
    checkpoint = _checkpoint(tiered=False)
    client = _client(_authored())

    await CharacterManager(
        client, PromptManager("app/prompts"),
    ).spawn_characters(checkpoint, [_request(tier=0)])

    system, user = _rendered_call(client)
    rendered = f"{system}\n{user}".casefold()
    assert "credit refinery" in rendered
    assert "glass synthesis chamber" in rendered
    assert "mnemonic forge" in rendered
    assert "## Knowledge Budget (authoritative)" not in user
    assert "The Cartel Prince" in user


@pytest.mark.asyncio
async def test_one_star_tier_one_render_excludes_later_system_concepts() -> None:
    checkpoint_path = Path(
        "app/storage/stories/one_star_ascension_s1/ckpt_0000.json"
    )
    checkpoint = CheckpointFile.model_validate_json(checkpoint_path.read_text())
    client = _client(
        _authored(
            known_context="Only cold light and an unfamiliar hall are apparent.",
        ),
    )
    request = SpawnRequest(
        character_id="knowledge_boundary_fixture",
        seed={
            "role": "elderly baker with no combat training",
            "reason": "cold light left a stranger in an unfamiliar stone hall",
            "location": "niflheim_lobby",
            "objectives": ["Find out where they are and whether there is an exit"],
            "knowledge_tier": 1,
        },
    )

    spawned = await CharacterManager(
        client, PromptManager("app/prompts"),
    ).spawn_characters(
        checkpoint,
        [request],
        one_star_hero_ids=set(),
    )

    system, user = _rendered_call(client)
    rendered = f"{system}\n{user}".casefold()
    assert all(
        term not in rendered
        for term in (
            "unseen master",
            "the master",
            "tower",
            "climb",
            "deployment",
            "hero duty",
            "synthesis",
            "promotion chamber",
            "daily-dungeon",
            "hidden economy",
            "moebius",
            "the fade",
        )
    )
    assert ONE_STAR_HERO_KEY not in spawned[0].mechanics
    assert client.complete.await_args.kwargs["response_model"] is AuthoredCharacter


@pytest.mark.asyncio
async def test_one_star_summon_membership_selects_hero_generation_overlay() -> None:
    checkpoint_path = Path(
        "app/storage/stories/one_star_ascension_s1/ckpt_0000.json"
    )
    checkpoint = CheckpointFile.model_validate_json(checkpoint_path.read_text())
    authored_data = _authored(
        known_context="Only cold light and an unfamiliar hall are apparent.",
    ).model_dump(mode="json")
    authored_data["one_star_hero"] = {
        "level": 1,
        "experience_points": 0,
        "hp_current": 7,
        "hp_max": 7,
        "stats": {"vitality": 1},
        "equipment": [],
        "skills": [],
        "conditions": [],
        "persistent_injuries": [],
        "innate_system_sight": False,
        "hidden_capabilities": {},
        "private_potential": "",
    }
    authored = AuthoredOneStarCharacter.model_validate(authored_data)
    client = _client(authored)
    request = SpawnRequest(
        character_id="summoned_hero_fixture",
        seed={
            "role": "elderly baker with no combat training",
            "reason": "cold light left a stranger in an unfamiliar stone hall",
            "location": "niflheim_lobby",
            "objectives": ["Find safety and an exit"],
            "knowledge_tier": 1,
        },
    )

    spawned = await CharacterManager(
        client, PromptManager("app/prompts"),
    ).spawn_characters(
        checkpoint,
        [request],
        one_star_hero_ids={request.character_id},
    )

    assert ONE_STAR_HERO_KEY in spawned[0].mechanics
    assert client.complete.await_args.kwargs["response_model"] is AuthoredOneStarCharacter
    assert client.complete.await_args.kwargs["role"] == "character_manager"
    assert client.complete.await_args.kwargs["max_tokens"] == 8_000
    assert spawned[0].agent_tier is CharacterAgentTier.standard
    assert model_role_for_character(spawned[0]) == "agent_standard"
    assert spawned[0].mechanics[ONE_STAR_HERO_KEY]["generated_for_summon"] is True
