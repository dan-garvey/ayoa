from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.engine.character_agent import CharacterAgent
from app.engine.one_star_projection import (
    one_star_agent_state_block,
    visible_equipped_item_description,
)
from app.engine.one_star_adapter import OneStarTransactionError
from app.engine.prompt_manager import PromptManager
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    ONE_STAR_RULESET_ID,
    OneStarAccountEnvelope,
    OneStarAccountState,
    OneStarCost,
    OneStarEquipmentEntry,
    OneStarHeroConstraints,
    OneStarHeroState,
    OneStarMissionCounter,
    OneStarMissionState,
    OneStarPendingOperation,
    OneStarResources,
    OneStarRulesConfig,
    OneStarSkillEntry,
    OneStarSummonPool,
)
from app.schemas.state import SessionConfig, SessionSettings, SessionState
from tests.support.factories import text_llm_response


def _hero_state(*, innate_system_sight: bool = False) -> OneStarHeroState:
    return OneStarHeroState(
        birth_stars=1,
        current_stars=2,
        level=7,
        experience_points=901,
        hp_current=17,
        hp_max=53,
        stats={"power": 23, "agility": 19},
        equipment=[
            OneStarEquipmentEntry(
                item_id="knife",
                name="Notched Kitchen Knife",
                slot="hand",
                quantity=1,
                durability_current=4,
                durability_max=11,
                tags=[],
                visible=True,
            ),
            OneStarEquipmentEntry(
                item_id="token",
                name="Hidden Token",
                slot="pocket",
                quantity=1,
                durability_current=1,
                durability_max=1,
                tags=[],
                visible=False,
            ),
        ],
        skills=[
            OneStarSkillEntry(
                skill_id="knead",
                name="Dough-Hardened Grip",
                rank=3,
                capability="Can keep hold of a struggling object.",
                tags=[],
                visible=True,
            ),
            OneStarSkillEntry(
                skill_id="sealed",
                name="Sealed Art",
                rank=9,
                capability="spoiler skill",
                tags=[],
                visible=False,
            ),
        ],
        conditions=["bleeding"],
        persistent_injuries=["bad knee"],
        innate_system_sight=innate_system_sight,
        owner_lobby_id="cold_lobby",
        hidden_capabilities={"awakening": "world-ending spoiler"},
        private_potential="secret six-star potential",
    )


def _rules_config() -> OneStarRulesConfig:
    zero = OneStarCost(
        gold=0,
        gems=0,
        building_resources=0,
    )
    return OneStarRulesConfig(
        starting_resources=OneStarResources(
            gold=40,
            gems=5,
            building_resources=3,
        ),
        lobby_id="cold_lobby",
        lobby_location_label="cold_lobby_hall",
        catalogue={},
        summon_pools={
            "basic": OneStarSummonPool(
                cost=OneStarCost(
                    gold=2,
                    gems=0,
                    building_resources=0,
                ),
                minimum_birth_stars=1,
                maximum_birth_stars=3,
                fresh_generation_allowed=True,
                usage="standard",
            ),
        },
        star_level_caps={1: 10, 2: 20},
        starting_lobby_floor=1,
        starting_capacity=10,
        maximum_stamina=5,
        stamina_recovery_seconds=300,
        deployment_stamina_cost=1,
        max_summon_batch=10,
        hero_constraints=OneStarHeroConstraints(
            minimum_hp_max=1,
            maximum_hp_max=1000,
            maximum_xp=10000,
            maximum_stat_value=100,
            maximum_equipment_entries=20,
            maximum_skill_entries=20,
        ),
        floor_rewards={},
        repeat_gold_numerator=1,
        repeat_gold_denominator=2,
        repeat_gold_minimum=1,
        promotion_cost=zero,
        operation_requirements={
            "deployment": {
                "facility_id": "tower_gate",
                "required_location": "",
            },
            "synthesis": {
                "facility_id": "synthesis_chamber",
                "required_location": "synthesis_room",
            },
            "promotion": {
                "facility_id": "promotion_chamber",
                "required_location": "promotion_room",
            },
        },
        lobby_return_healing=True,
        hero_system_visibility_research_key="hero_reaction",
    )


def _checkpoint(
    *,
    research_level: int = 0,
    innate_system_sight: bool = False,
    system_observer: bool = False,
    ruleset_id: str = ONE_STAR_RULESET_ID,
) -> tuple[CheckpointFile, CharacterRecord, CharacterRecord, CharacterRecord]:
    config = _rules_config()
    account = OneStarAccountEnvelope(
        config=config,
        state=OneStarAccountState(
            resources=OneStarResources(
                gold=37,
                gems=4,
                building_resources=2,
                materials={"slime_residue": 6},
            ),
            inventory={"iron_sword": 2},
            facilities={
                "armory": 1,
                "tower_gate": 1,
                "synthesis_chamber": 1,
                "promotion_chamber": 1,
            },
            lobby_floor=2,
            capacity=12,
            highest_unlocked_floor=2,
            stamina_current=3,
            active_master_feed_id="feed-7",
            guide_character_ids=["guide"],
            system_observer_ids=["hero"] if system_observer else [],
            research_levels={
                "hero_reaction": research_level
            } if research_level else {},
            tutorial_deliveries={"summoning": ["owner"]},
            pending_operation=OneStarPendingOperation(
                operation_id="op-1",
                kind="synthesis",
                participant_ids=["hero", "donor"],
                target_id="hero",
                destination="synthesis_room",
                opened_at_s=42,
            ),
        ),
    )
    owner = CharacterRecord(
        character_id="owner",
        name="Account Seat",
        public_sheet=PublicSheet(role="account"),
        private_state=PrivateState(secrets=["owner-secret"]),
        mechanics={ONE_STAR_ACCOUNT_KEY: account.model_dump(mode="json")},
    )
    hero = CharacterRecord(
        character_id="hero",
        name="Tired Baker",
        public_sheet=PublicSheet(role="baker"),
        private_state=PrivateState(secrets=["hero-secret"]),
        mechanics={
            ONE_STAR_HERO_KEY: _hero_state(
                innate_system_sight=innate_system_sight
            ).model_dump(mode="json")
        },
    )
    guide = CharacterRecord(
        character_id="guide",
        name="Guide",
        public_sheet=PublicSheet(role="guide"),
    )
    ckpt = CheckpointFile(
        session=SessionState(
            session_id="projection",
            config=SessionConfig(
                settings=SessionSettings(ruleset_id=ruleset_id)
            ),
        ),
        characters=[owner, hero, guide],
    )
    return ckpt, owner, hero, guide


def test_account_projection_is_exact_but_excludes_private_mechanics() -> None:
    checkpoint, owner, _, _ = _checkpoint()

    block = one_star_agent_state_block(checkpoint, owner)

    assert "Gold 37" in block
    assert "Gems 4" in block
    assert "Building Resources 2" in block
    assert "armory 1" in block
    assert "Hero capacity: 12" in block
    assert "basic: 2 Gold" in block
    assert "feed-7" in block
    assert "Tired Baker: 2-star" in block
    assert "HP 17/53" in block
    assert "owner-secret" not in block
    assert "hero-secret" not in block
    assert "secret six-star potential" not in block
    assert "world-ending spoiler" not in block
    assert "Hidden Token" not in block
    assert "Sealed Art" not in block


def test_ordinary_hero_gets_embodied_state_without_system_numbers_or_lore() -> None:
    checkpoint, _, hero, _ = _checkpoint()

    block = one_star_agent_state_block(checkpoint, hero)

    assert "badly hurt" in block
    assert "bleeding" in block
    assert "bad knee" in block
    assert "Notched Kitchen Knife" in block
    assert "Dough-Hardened Grip" in block
    assert "17" not in block
    assert "53" not in block
    assert "901" not in block
    assert "Stats:" not in block
    assert "Gold" not in block
    assert "Master" not in block
    assert "Tower" not in block
    assert "duty" not in block.lower()
    assert "Hidden Token" not in block
    assert "Sealed Art" not in block
    assert "six-star" not in block


@pytest.mark.parametrize(
    ("research_level", "innate_system_sight", "system_observer"),
    [(1, False, False), (0, True, False), (0, False, True)],
)
def test_research_or_system_sight_reveals_exact_own_sheet(
    research_level: int,
    innate_system_sight: bool,
    system_observer: bool,
) -> None:
    checkpoint, _, hero, _ = _checkpoint(
        research_level=research_level,
        innate_system_sight=innate_system_sight,
        system_observer=system_observer,
    )

    block = one_star_agent_state_block(checkpoint, hero)

    assert "level 7" in block
    assert "XP 901" in block
    assert "HP 17/53" in block
    assert "power 23" in block
    assert "rank 3" in block
    assert "durability 4/11" in block
    assert "secret six-star potential" not in block
    assert "world-ending spoiler" not in block


def test_guide_gets_management_channel_without_tactical_or_hero_omniscience() -> None:
    checkpoint, _, _, guide = _checkpoint()

    block = one_star_agent_state_block(checkpoint, guide)

    assert "Authored System Channel" in block
    assert "Gold 37" in block
    assert "synthesis" in block
    assert "summoning" in block
    assert "Active mission:" not in block
    assert "feed-7" not in block
    assert "HP 17/53" not in block
    assert "hero-secret" not in block


def test_active_mission_is_visible_to_owner_but_not_guide() -> None:
    checkpoint, owner, _, guide = _checkpoint()
    account = OneStarAccountEnvelope.model_validate(
        owner.mechanics[ONE_STAR_ACCOUNT_KEY]
    )
    account.state.pending_operation = None
    account.state.active_mission = OneStarMissionState(
        mission_id="floor_two",
        floor=2,
        party_ids=["hero"],
        formation_labels={"hero": "front"},
        destination="tower_floor_2",
        completion_declaration="the last enemy falls",
        failure_declaration="the party is destroyed",
        counters={"enemies": OneStarMissionCounter(current=2, target=5)},
        started_at_s=40,
        deadline_at_s=340,
    )
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")

    owner_block = one_star_agent_state_block(checkpoint, owner)
    guide_block = one_star_agent_state_block(checkpoint, guide)

    assert "Active mission: floor_two" in owner_block
    assert "enemies 2/5" in owner_block
    assert "Active mission:" not in guide_block
    assert "enemies 2/5" not in guide_block


def test_projection_is_binding_invariant_and_non_one_star_is_empty() -> None:
    checkpoint, _, hero, _ = _checkpoint()
    before = one_star_agent_state_block(checkpoint, hero)
    checkpoint.session.character_bindings = {
        "hero": "controller-a",
        "owner": "controller-b",
    }
    after = one_star_agent_state_block(checkpoint, hero)
    assert before == after

    narrative, _, narrative_hero, _ = _checkpoint(ruleset_id="narrative")
    assert one_star_agent_state_block(narrative, narrative_hero) == ""


def test_active_ruleset_with_missing_account_fails_loudly() -> None:
    checkpoint = CheckpointFile(
        session=SessionState(
            session_id="broken",
            config=SessionConfig(
                settings=SessionSettings(ruleset_id=ONE_STAR_RULESET_ID)
            ),
        ),
    )

    with pytest.raises(OneStarTransactionError):
        one_star_agent_state_block(
            checkpoint,
            CharacterRecord(character_id="nobody", name="Nobody"),
        )


def test_foreign_lobby_hero_stays_out_of_account_but_keeps_own_body_view() -> None:
    checkpoint, owner, hero, _ = _checkpoint()
    foreign = _hero_state()
    foreign.owner_lobby_id = "different_lobby"
    hero.mechanics[ONE_STAR_HERO_KEY] = foreign.model_dump(mode="json")

    owner_block = one_star_agent_state_block(checkpoint, owner)
    own_block = one_star_agent_state_block(checkpoint, hero)

    assert "Tired Baker" not in owner_block
    assert "Your bodily condition" in own_block
    assert "Notched Kitchen Knife" in own_block
    assert "HP 7/10" not in own_block


def test_image_equipment_projection_only_includes_visible_current_items() -> None:
    _, _, hero, guide = _checkpoint()

    description = visible_equipped_item_description(hero)

    assert "Notched Kitchen Knife" in description
    assert "hand slot" in description
    assert "Hidden Token" not in description
    assert "bleeding" not in description
    assert visible_equipped_item_description(guide) == ""


@pytest.mark.asyncio
async def test_dynamic_mechanics_stay_in_user_tail_and_out_of_cached_addon() -> None:
    checkpoint, _, hero, _ = _checkpoint()
    client = AsyncMock()
    client.complete.return_value = text_llm_response(
        "She grips the counter. (I will not be volunteered.)"
    )
    agent = CharacterAgent(client, PromptManager())

    first_draft = await agent.draft_turn(hero, checkpoint)
    first_messages = client.complete.await_args.kwargs["messages"]

    hero_state = _hero_state()
    hero_state.hp_current = 3
    hero.mechanics[ONE_STAR_HERO_KEY] = hero_state.model_dump(mode="json")
    await agent.draft_turn(hero, checkpoint)
    second_messages = client.complete.await_args.kwargs["messages"]

    assert first_messages[0]["content"] == second_messages[0]["content"]
    assert "badly hurt" not in first_messages[0]["content"]
    assert "badly hurt" in first_messages[-1]["content"]
    assert "critically hurt" in second_messages[-1]["content"]
    assert "badly hurt" not in first_draft.user_message.content
    assert "Current Mechanics" not in first_draft.user_message.content
