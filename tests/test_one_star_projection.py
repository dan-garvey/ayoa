from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.engine.character_agent import CharacterAgent
from app.engine.one_star_projection import (
    one_star_agent_state_block,
    one_star_master_command_lines,
    one_star_synthesis_authoritative_plan,
    visible_equipped_item_description,
)
from app.engine.one_star_adapter import OneStarTransactionError
from app.engine.one_star_progression import rebalance_hero
from app.engine.prompt_manager import PromptManager
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_COMBATANT_KEY,
    ONE_STAR_HERO_KEY,
    ONE_STAR_RULESET_ID,
    OneStarAccountEnvelope,
    OneStarAccountState,
    OneStarCatalogueEntry,
    OneStarCombatantState,
    OneStarCost,
    OneStarEquipmentEntry,
    OneStarGemPurchaseConfig,
    OneStarHeroState,
    OneStarMissionCounter,
    OneStarMissionState,
    OneStarPendingOperation,
    OneStarResources,
    OneStarRulesConfig,
    OneStarProgressionConfig,
    OneStarSkillEntry,
    OneStarSynthesisPreview,
    OneStarSummonPool,
)
from app.schemas.state import SessionConfig, SessionSettings, SessionState
from tests.support.factories import text_llm_response


def _hero_state(*, innate_system_sight: bool = False) -> OneStarHeroState:
    hero = OneStarHeroState(
        birth_stars=1,
        current_stars=2,
        level=7,
        experience_points=2_100,
        hp_current=1,
        hp_max=1,
        stats={"power": 1, "agility": 1, "resilience": 1},
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
        terminal_event_id="",
        progression_seed="projection-test-hero",
        strong_stat_id="power",
        weak_stat_id="agility",
        potential_grade=1,
    )
    rebalance_hero(hero=hero, config=_rules_config(), restore_full_hp=True)
    hero.hp_current = hero.hp_max // 2
    return hero


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
                star_weights={1: 8_000, 2: 1_800, 3: 200},
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
        progression=OneStarProgressionConfig(
            stat_ids=["power", "agility", "resilience"],
            grade_multiplier_milli=1250,
            birth_stat_total=15,
            birth_hp_max=8,
            variance_basis_points=500,
            stat_growth_per_level_milli=1000,
            hp_growth_per_level_milli=500,
            xp_threshold_factor=50,
            floor_xp_per_floor=100,
            overlevel_xp_percentages=[100, 75, 50, 25, 10, 5, 0],
            cap_bank_extra_levels=1,
            synthesis_source_base_xp=100,
            synthesis_skill_chance_basis_points=500,
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
        gem_purchase=OneStarGemPurchaseConfig(
            funds_label="$",
            starting_funds=200,
            periodic_income=100,
            income_interval_seconds=604_800,
            funds_cost=100,
            gems_granted=20,
        ),
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
            stored_equipment=[
                OneStarEquipmentEntry(
                    item_id="stored_bronze_shield",
                    name="Stored Bronze Shield",
                    slot="offhand",
                    quantity=1,
                    durability_current=7,
                    durability_max=9,
                    tags=["storage"],
                    visible=True,
                ),
            ],
            facilities={
                "armory": 1,
                "tower_gate": 1,
                "synthesis_chamber": 1,
                "promotion_chamber": 1,
            },
            lobby_floor=2,
            capacity=12,
            highest_cleared_floor=1,
            highest_unlocked_floor=2,
            stamina_current=3,
            discretionary_funds=200,
            funds_accrual_anchor_s=0,
            active_master_feed_id="feed-7",
            guide_character_ids=["guide"],
            system_observer_ids=["hero"] if system_observer else [],
            research_levels={"hero_reaction": research_level} if research_level else {},
            tutorial_deliveries={"summoning": ["owner"]},
            pending_operation=OneStarPendingOperation(
                operation_id="op-1",
                kind="synthesis",
                participant_ids=["hero", "donor"],
                target_id="hero",
                destination="synthesis_room",
                opened_at_s=42,
                synthesis_preview=OneStarSynthesisPreview(
                    offered_xp=100,
                    applied_xp=100,
                    wasted_xp=0,
                    returned_equipment=[
                        OneStarEquipmentEntry(
                            item_id="donor_knife",
                            name="Donor Knife",
                            slot="hand",
                            quantity=1,
                            durability_current=3,
                            durability_max=3,
                            tags=[],
                            visible=True,
                        ),
                    ],
                    skill_transfer_chance_basis_points=500,
                    input_state_fingerprint="private-preview-fingerprint",
                ),
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
        mechanics={
            ONE_STAR_COMBATANT_KEY: OneStarCombatantState(
                hp_current=2_000,
                hp_max=2_000,
                stats={"power": 200, "agility": 200, "resilience": 200},
            ).model_dump(mode="json")
        },
    )
    ckpt = CheckpointFile(
        session=SessionState(
            session_id="projection",
            config=SessionConfig(settings=SessionSettings(ruleset_id=ruleset_id)),
        ),
        characters=[owner, hero, guide],
    )
    return ckpt, owner, hero, guide


def test_account_projection_is_exact_but_excludes_private_mechanics() -> None:
    checkpoint, owner, _, _ = _checkpoint()

    block = one_star_agent_state_block(checkpoint, owner)

    assert "Gold 37" in block
    assert "Gems 4" in block
    assert "Discretionary funds: $200" in block
    assert "Building Resources 2" in block
    assert "armory 1" in block
    assert "Hero capacity: 12" in block
    assert "basic: 2 Gold" in block
    assert "1-star 80%, 2-star 18%, 3-star 2%" in block
    assert "feed-7" in block
    assert "Tired Baker: 2-star" in block
    assert "HP 5/10" in block
    assert "Stored Bronze Shield [stored_bronze_shield] (offhand)" in block
    assert "owner-secret" not in block
    assert "hero-secret" not in block
    assert "potential_grade" not in block
    assert "progression_seed" not in block
    assert "xp_threshold_factor" not in block
    assert '"item_id"' not in block
    assert "world-ending spoiler" not in block
    assert "Hidden Token" not in block
    assert "Sealed Art" not in block


def test_guide_receives_exact_non_hero_combat_authority() -> None:
    checkpoint, _, _, guide = _checkpoint()

    block = one_star_agent_state_block(checkpoint, guide)

    assert "exact non-Hero combat authority" in block
    assert "HP 2000/2000" in block
    assert "agility 200, power 200, resilience 200" in block
    assert ONE_STAR_HERO_KEY not in guide.mechanics


def test_master_commands_split_account_roster_and_hero_sheet() -> None:
    checkpoint, owner, hero_character, _ = _checkpoint()

    status = one_star_master_command_lines(
        checkpoint,
        owner.character_id,
        "status",
    )
    roster = one_star_master_command_lines(
        checkpoint,
        owner.character_id,
        "heroes",
    )
    hero_lines = one_star_master_command_lines(
        checkpoint,
        owner.character_id,
        "hero",
        hero_ref="#1",
    )

    status_text = "\n".join(status)
    assert "Gold 37; Gems 4; Building Resources 2" in status_text
    assert "slime_residue x6" in status_text
    assert "iron_sword x2" in status_text
    assert "Stored Bronze Shield [stored_bronze_shield] (offhand)" in status_text
    assert "durability 7/9" in status_text
    assert "highest cleared floor 1" in status_text
    assert "highest unlocked floor 2" in status_text
    assert "Hero capacity: 12; stamina 3/5" in status_text
    assert "Occupied Hero slots: 1/12" in status_text
    assert "armory 1" in status_text
    assert "feed-7" in status_text
    assert "Pending management operation: synthesis" in status_text
    assert "potential_grade" not in status_text
    assert "progression_seed" not in status_text
    assert "xp_threshold_factor" not in status_text
    assert '"item_id"' not in status_text

    roster_text = "\n".join(roster)
    assert "Tired Baker: 2-star" in roster_text
    assert "level 7" in roster_text
    assert "XP 2100/2800 to level 8" in roster_text
    assert "HP 5/10" in roster_text
    assert "Stats: agility 5, power 9, resilience 7" in roster_text

    hero_text = "\n".join(hero_lines)
    assert "Tired Baker [hero]" in hero_text
    assert "XP 2100/2800 to level 8" in hero_text
    assert "Dough-Hardened Grip (rank 3)" in hero_text
    assert "Notched Kitchen Knife (hand)" in hero_text
    assert "bleeding, bad knee" in hero_text
    assert "Hidden Token" not in hero_text
    assert "Sealed Art" not in hero_text
    assert "potential_grade" not in hero_text
    assert "progression_seed" not in hero_text

    hero_state = _hero_state()
    hero_state.level = 20
    hero_state.experience_points = 21_000
    hero_character.mechanics[ONE_STAR_HERO_KEY] = hero_state.model_dump(mode="json")
    cap_roster = one_star_master_command_lines(
        checkpoint,
        owner.character_id,
        "heroes",
    )
    assert "XP 21000; cap-bank 2000/2000" in "\n".join(cap_roster)


def test_master_status_projects_weekly_funds_without_mutating_the_ledger() -> None:
    checkpoint, owner, _, _ = _checkpoint()
    checkpoint.session.leading_at_s = 604_800

    status_text = "\n".join(
        one_star_master_command_lines(
            checkpoint,
            owner.character_id,
            "status",
        )
    )

    assert "Discretionary funds: $300" in status_text
    assert "income $100 every 604800s" in status_text
    account = OneStarAccountEnvelope.model_validate(
        owner.mechanics[ONE_STAR_ACCOUNT_KEY]
    )
    assert account.state.discretionary_funds == 200
    assert account.state.funds_accrual_anchor_s == 0


def test_master_commands_reject_non_owner_and_other_rulesets() -> None:
    checkpoint, owner, hero, _ = _checkpoint()

    with pytest.raises(ValueError, match="only to the character that owns"):
        one_star_master_command_lines(
            checkpoint,
            hero.character_id,
            "status",
        )

    narrative, narrative_owner, _, _ = _checkpoint(ruleset_id="narrative")
    with pytest.raises(ValueError, match="only in a One-Star Ascension"):
        one_star_master_command_lines(
            narrative,
            narrative_owner.character_id,
            "status",
        )


def test_master_hero_command_resolves_name_and_rejects_unknown() -> None:
    checkpoint, owner, _, _ = _checkpoint()

    by_name = one_star_master_command_lines(
        checkpoint,
        owner.character_id,
        "hero",
        hero_ref="tired baker",
    )
    assert "Tired Baker [hero]" in by_name

    with pytest.raises(ValueError, match="No owned Hero matches"):
        one_star_master_command_lines(
            checkpoint,
            owner.character_id,
            "hero",
            hero_ref="stranger",
        )


def test_synthesis_command_resolves_exact_heroes_without_mutating_state() -> None:
    checkpoint, owner, target, _guide = _checkpoint()
    account = OneStarAccountEnvelope.model_validate(
        owner.mechanics[ONE_STAR_ACCOUNT_KEY]
    )
    account.state.pending_operation = None
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    source = CharacterRecord(
        character_id="donor",
        name="Edric",
        public_sheet=PublicSheet(role="guard"),
        private_state=PrivateState(),
        mechanics={ONE_STAR_HERO_KEY: _hero_state().model_dump(mode="json")},
    )
    checkpoint.characters.append(source)
    before = checkpoint.model_dump(mode="json")

    plan = one_star_synthesis_authoritative_plan(
        checkpoint,
        owner.character_id,
        target_ref=target.name,
        source_refs=("Edric",),
    )

    assert plan.viewpoint_character_id == owner.character_id
    assert plan.ruleset_actor_id == owner.character_id
    assert plan.location_updates == (
        ("donor", "synthesis_room"),
        ("hero", "synthesis_room"),
        ("guide", "synthesis_room"),
    )
    assert [update["kind"] for update in plan.state_updates] == [
        "pending_open",
        "pending_resolve",
    ]
    assert plan.state_updates[0]["details"] == [
        "participant=donor",
        "target_id=hero",
        "destination=synthesis_room",
    ]
    assert [request.character_id for request in plan.contribution_requests] == [
        "donor",
    ]
    assert checkpoint.model_dump(mode="json") == before


def test_synthesis_command_rejects_duplicate_or_unavailable_selections() -> None:
    checkpoint, owner, target, _guide = _checkpoint()
    account = OneStarAccountEnvelope.model_validate(
        owner.mechanics[ONE_STAR_ACCOUNT_KEY]
    )
    account.state.pending_operation = None
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    source = CharacterRecord(
        character_id="donor",
        name="Edric",
        public_sheet=PublicSheet(role="guard"),
        mechanics={ONE_STAR_HERO_KEY: _hero_state().model_dump(mode="json")},
    )
    checkpoint.characters.append(source)

    with pytest.raises(ValueError, match="cannot also be a source"):
        one_star_synthesis_authoritative_plan(
            checkpoint,
            owner.character_id,
            target_ref=target.name,
            source_refs=(target.name,),
        )
    with pytest.raises(ValueError, match="only once"):
        one_star_synthesis_authoritative_plan(
            checkpoint,
            owner.character_id,
            target_ref=target.name,
            source_refs=("Edric", "donor"),
        )

    account = OneStarAccountEnvelope.model_validate(
        owner.mechanics[ONE_STAR_ACCOUNT_KEY]
    )
    account.config.catalogue["synthesis_chamber_1"] = OneStarCatalogueEntry(
        kind="facility_build",
        cost=OneStarCost(gold=1, gems=0, building_resources=1),
        facility_id="synthesis_chamber",
        target_level=1,
    )
    account.state.facilities.pop("synthesis_chamber")
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    with pytest.raises(ValueError, match="not operational"):
        one_star_synthesis_authoritative_plan(
            checkpoint,
            owner.character_id,
            target_ref=target.name,
            source_refs=("Edric",),
        )


def test_synthesis_after_first_resolution_omits_character_contributions() -> None:
    checkpoint, owner, target, _guide = _checkpoint()
    account = OneStarAccountEnvelope.model_validate(
        owner.mechanics[ONE_STAR_ACCOUNT_KEY]
    )
    account.state.pending_operation = None
    account.state.synthesis_resolution_count = 1
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    checkpoint.characters.append(
        CharacterRecord(
            character_id="donor",
            name="Edric",
            public_sheet=PublicSheet(role="guard"),
            private_state=PrivateState(),
            mechanics={ONE_STAR_HERO_KEY: _hero_state().model_dump(mode="json")},
        )
    )

    plan = one_star_synthesis_authoritative_plan(
        checkpoint,
        owner.character_id,
        target_ref=target.name,
        source_refs=("Edric",),
    )

    assert plan.contribution_requests == ()


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
    assert "potential" not in block


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
    assert "XP 2100/2800 to level 8" in block
    assert "HP 5/10" in block
    assert "power 9" in block
    assert "rank 3" in block
    assert "durability 4/11" in block
    assert "potential_grade" not in block
    assert "progression_seed" not in block
    assert "world-ending spoiler" not in block


def test_guide_gets_management_channel_without_tactical_or_hero_omniscience() -> None:
    checkpoint, _, _, guide = _checkpoint()

    block = one_star_agent_state_block(checkpoint, guide)

    assert "Authored System Channel" in block
    assert "Gold 37" in block
    assert "synthesis" in block
    assert "summoning" in block
    assert "1-star level 10, 2-star level 20" in block
    assert "exactly one star rank" in block
    assert "no resources" in block
    assert "promotion_room" in block
    assert "Selecting a promotion spends nothing" in block
    assert "retained experience" in block
    assert "Active mission:" not in block
    assert "feed-7" not in block
    assert "HP 5/10" not in block
    assert "hero-secret" not in block
    assert "Stored Bronze Shield" not in block
    assert "potential_grade" not in block
    assert "progression_seed" not in block


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
        formation_labels=[{"character_id": "hero", "label": "front"}],
        destination="tower_floor_2",
        completion_declaration="the last enemy falls",
        failure_declaration="the party is destroyed",
        counters=[
            OneStarMissionCounter(counter_id="enemies", current=2, target=5),
        ],
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
    hero_state.hp_current = 2
    hero.mechanics[ONE_STAR_HERO_KEY] = hero_state.model_dump(mode="json")
    await agent.draft_turn(hero, checkpoint)
    second_messages = client.complete.await_args.kwargs["messages"]

    assert first_messages[0]["content"] == second_messages[0]["content"]
    assert "badly hurt" not in first_messages[0]["content"]
    assert "badly hurt" in first_messages[-1]["content"]
    assert "critically hurt" in second_messages[-1]["content"]
    assert "badly hurt" not in first_draft.user_message.content
    assert "Current Mechanics" not in first_draft.user_message.content
