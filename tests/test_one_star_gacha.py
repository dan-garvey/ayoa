"""Offline contracts for adapter-owned One-Star summon draws."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.engine.action_rejection import PlayerActionRejected
from app.engine.one_star_adapter import (
    OneStarTransactionError,
    apply_one_star_prepared_mutation,
    load_one_star_account,
    one_star_birth_stars_for_ticket,
    one_star_standard_summon_lifecycle,
    one_star_state_updates_to_transaction,
    one_star_summon_draw_preview,
    preflight_one_star_account_updates,
    prepare_one_star_transaction,
)
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    OneStarCost,
    OneStarSummonPool,
    OneStarStateUpdate,
    OneStarTransaction,
)
from app.schemas.state import SessionConfig, SessionSettings, SessionState


def _pool(
    minimum: int,
    maximum: int,
    weights: dict[int, int],
) -> OneStarSummonPool:
    return OneStarSummonPool(
        cost=OneStarCost(
            gold=0,
            gems=0,
            building_resources=0,
        ),
        minimum_birth_stars=minimum,
        maximum_birth_stars=maximum,
        star_weights=weights,
        fresh_generation_allowed=True,
        usage="standard",
    )


def _hero(
    character_id: str,
    *,
    birth_stars: int = 1,
    status: CharacterStatus = CharacterStatus.dormant,
    owner: str = "",
) -> CharacterRecord:
    stat_totals = {
        1: {"power": 6, "agility": 5, "resilience": 4},
        2: {"power": 8, "agility": 6, "resilience": 5},
        3: {"power": 9, "agility": 8, "resilience": 6},
        4: {"power": 11, "agility": 10, "resilience": 8},
        5: {"power": 14, "agility": 12, "resilience": 11},
    }
    hp_by_grade = {1: 5, 2: 6, 3: 8, 4: 10, 5: 12}
    return CharacterRecord(
        character_id=character_id,
        name=character_id.replace("_", " ").title(),
        status=status,
        location="lobby",
        mechanics={
            ONE_STAR_HERO_KEY: {
                "birth_stars": birth_stars,
                "current_stars": birth_stars,
                "hp_current": hp_by_grade[birth_stars],
                "hp_max": hp_by_grade[birth_stars],
                "stats": stat_totals[birth_stars],
                "owner_lobby_id": owner,
                "acquisition_event_id": "seed" if owner else "",
                "terminal_event_id": "",
                "progression_seed": f"{character_id}_progression_seed",
                "strong_stat_id": "power",
                "weak_stat_id": "resilience",
                "potential_grade": birth_stars,
            }
        },
    )


def _checkpoint() -> CheckpointFile:
    config = {
        "starting_resources": {
            "gold": 100,
            "gems": 100,
            "building_resources": 0,
            "materials": {},
        },
        "lobby_id": "lobby_a",
        "lobby_location_label": "lobby",
        "catalogue": {},
        "summon_pools": {
            "basic": {
                "cost": {
                    "gold": 2,
                    "gems": 0,
                    "building_resources": 0,
                    "materials": {},
                },
                "minimum_birth_stars": 1,
                "maximum_birth_stars": 1,
                "star_weights": {1: 10_000},
                "eligible_existing_ids": ["reserve_a", "reserve_b"],
                "fresh_generation_allowed": True,
                "usage": "standard",
            },
            "premium": {
                "cost": {
                    "gold": 0,
                    "gems": 5,
                    "building_resources": 0,
                    "materials": {},
                },
                "minimum_birth_stars": 2,
                "maximum_birth_stars": 5,
                "star_weights": {2: 7500, 3: 2300, 4: 175, 5: 25},
                "eligible_existing_ids": [],
                "fresh_generation_allowed": True,
                "usage": "standard",
            },
        },
        "star_level_caps": {1: 10, 2: 20, 3: 40, 4: 60, 5: 80},
        "starting_lobby_floor": 1,
        "starting_capacity": 20,
        "maximum_stamina": 5,
        "stamina_recovery_seconds": 1800,
        "deployment_stamina_cost": 1,
        "max_summon_batch": 5,
        "progression": {
            "stat_ids": ["power", "agility", "resilience"],
            "grade_multiplier_milli": 1250,
            "birth_stat_total": 15,
            "birth_hp_max": 5,
            "variance_basis_points": 0,
            "stat_growth_per_level_milli": 1000,
            "hp_growth_per_level_milli": 500,
            "xp_threshold_factor": 50,
            "floor_xp_per_floor": 100,
            "overlevel_xp_percentages": [100, 75, 50, 25, 10, 5, 0],
            "cap_bank_extra_levels": 1,
            "synthesis_source_base_xp": 100,
            "synthesis_skill_chance_basis_points": 500,
        },
        "floor_rewards": {},
        "repeat_gold_numerator": 0,
        "repeat_gold_denominator": 1,
        "repeat_gold_minimum": 0,
        "promotion_cost": {
            "gold": 0,
            "gems": 0,
            "building_resources": 0,
            "materials": {},
        },
        "operation_requirements": {
            "deployment": {"facility_id": "tower_gate", "required_location": ""},
            "synthesis": {
                "facility_id": "synthesis_chamber",
                "required_location": "synthesis_room",
            },
            "promotion": {
                "facility_id": "promotion_chamber",
                "required_location": "promotion_room",
            },
        },
        "gem_purchase": {
            "funds_label": "$",
            "starting_funds": 200,
            "periodic_income": 100,
            "income_interval_seconds": 604_800,
            "funds_cost": 100,
            "gems_granted": 20,
        },
        "lobby_return_healing": True,
        "hero_system_visibility_research_key": "",
    }
    state = {
        "resources": config["starting_resources"],
        "facilities": {
            "tower_gate": 1,
            "synthesis_chamber": 1,
            "promotion_chamber": 1,
        },
        "lobby_floor": 1,
        "capacity": 20,
        "highest_unlocked_floor": 1,
        "highest_cleared_floor": 0,
        "stamina_current": 5,
        "discretionary_funds": 200,
        "funds_accrual_anchor_s": 0,
        "summon_draw_counters": {},
        "stored_equipment": [],
    }
    owner = CharacterRecord(
        character_id="account_owner",
        name="Account Owner",
        mechanics={ONE_STAR_ACCOUNT_KEY: {"config": config, "state": state}},
    )
    return CheckpointFile(
        session=SessionState(
            session_id="gacha_session_a",
            config=SessionConfig(
                settings=SessionSettings(ruleset_id="one_star_ascension")
            ),
        ),
        characters=[owner, _hero("reserve_a"), _hero("reserve_b")],
    )


def _summon_transaction(
    *,
    pool_id: str,
    hero_ids: list[str],
    birth_stars: list[int],
) -> OneStarTransaction:
    return OneStarTransaction.model_validate({
        "present": True,
        "operations": [{
            "operation": "summon",
            "pool_id": pool_id,
            "hero_ids": hero_ids,
            "birth_stars": birth_stars,
        }],
    })


@pytest.mark.parametrize(
    ("pool", "ticket", "expected_stars"),
    [
        (_pool(1, 3, {1: 8000, 2: 1800, 3: 200}), 0, 1),
        (_pool(1, 3, {1: 8000, 2: 1800, 3: 200}), 7999, 1),
        (_pool(1, 3, {1: 8000, 2: 1800, 3: 200}), 8000, 2),
        (_pool(1, 3, {1: 8000, 2: 1800, 3: 200}), 9799, 2),
        (_pool(1, 3, {1: 8000, 2: 1800, 3: 200}), 9800, 3),
        (_pool(1, 3, {1: 8000, 2: 1800, 3: 200}), 9999, 3),
        (_pool(2, 5, {2: 7500, 3: 2300, 4: 175, 5: 25}), 7499, 2),
        (_pool(2, 5, {2: 7500, 3: 2300, 4: 175, 5: 25}), 7500, 3),
        (_pool(2, 5, {2: 7500, 3: 2300, 4: 175, 5: 25}), 9799, 3),
        (_pool(2, 5, {2: 7500, 3: 2300, 4: 175, 5: 25}), 9800, 4),
        (_pool(2, 5, {2: 7500, 3: 2300, 4: 175, 5: 25}), 9974, 4),
        (_pool(2, 5, {2: 7500, 3: 2300, 4: 175, 5: 25}), 9975, 5),
        (_pool(2, 5, {2: 7500, 3: 2300, 4: 175, 5: 25}), 9999, 5),
    ],
)
def test_weighted_ticket_boundaries_are_exact(
    pool: OneStarSummonPool,
    ticket: int,
    expected_stars: int,
) -> None:
    assert one_star_birth_stars_for_ticket(pool, ticket) == expected_stars


@pytest.mark.parametrize(
    "weights",
    [
        {1: 8000, 2: 2000},
        {1: 8000, 2: 1800, 3: 199},
        {1: 8000, 2: 2000, 3: 0},
    ],
)
def test_pool_rejects_missing_inexact_or_zero_weight_authority(
    weights: dict[int, int],
) -> None:
    with pytest.raises(ValidationError, match="star_weights|weights"):
        _pool(1, 3, weights)


def test_preview_is_replay_stable_and_exhausts_reserves_without_duplication() -> None:
    checkpoint = _checkpoint()
    preview = one_star_summon_draw_preview(checkpoint, "basic", count=3)
    replayed = CheckpointFile.model_validate_json(checkpoint.model_dump_json())

    assert one_star_summon_draw_preview(replayed, "basic", count=3) == preview
    reserve_ids = [draw.existing_character_id for draw in preview[:2]]
    assert set(reserve_ids) == {"reserve_a", "reserve_b"}
    assert preview[2].existing_character_id == ""
    assert [draw.birth_stars for draw in preview] == [1, 1, 1]


def test_compact_standard_summon_update_derives_hidden_draw_and_lifecycle() -> None:
    checkpoint = _checkpoint()
    update = OneStarStateUpdate(
        kind="summon",
        target_id="basic",
        value="3",
        details=[],
    )
    preview = one_star_summon_draw_preview(checkpoint, "basic", count=3)
    transaction = one_star_state_updates_to_transaction(
        checkpoint,
        [update],
        canonical_at_s=0,
    )
    operation = transaction.operations[0]
    spawns, wakes = one_star_standard_summon_lifecycle(checkpoint, [update])

    assert operation.birth_stars == [draw.birth_stars for draw in preview]
    assert set(operation.hero_ids[:2]) == {"reserve_a", "reserve_b"}
    assert operation.hero_ids[2] == "lobby_a_basic_0003"
    assert {wake.character_id for wake in wakes} == {"reserve_a", "reserve_b"}
    assert [spawn.character_id for spawn in spawns] == ["lobby_a_basic_0003"]
    assert update.model_dump() == {
        "kind": "summon",
        "target_id": "basic",
        "value": "3",
        "details": [],
    }


def test_unaffordable_summon_is_rejected_without_mutating_draw_state() -> None:
    checkpoint = _checkpoint()
    owner, account = load_one_star_account(checkpoint)
    account.state.resources.gems = 5
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    before = checkpoint.model_dump_json()

    with pytest.raises(
        PlayerActionRejected,
        match=(
            r"Premium summon rejected: 5 pulls cost 25 Gems, but only "
            r"5 Gems are available.*Nothing was spent"
        ),
    ):
        preflight_one_star_account_updates(
            checkpoint,
            [OneStarStateUpdate(
                kind="summon",
                target_id="premium",
                value="5",
                details=[],
            )],
            initiating_actor_id="account_owner",
        )

    assert checkpoint.model_dump_json() == before
    assert load_one_star_account(checkpoint)[1].state.summon_draw_counters == {}


def test_oversized_summon_is_rejected_before_drawing() -> None:
    checkpoint = _checkpoint()
    _owner, account = load_one_star_account(checkpoint)

    with pytest.raises(
        PlayerActionRejected,
        match=(
            rf"allow at most {account.config.max_summon_batch} pulls at once"
        ),
    ):
        preflight_one_star_account_updates(
            checkpoint,
            [OneStarStateUpdate(
                kind="summon",
                target_id="premium",
                value=str(account.config.max_summon_batch + 1),
                details=[],
            )],
            initiating_actor_id="account_owner",
        )


def test_same_event_gem_purchase_is_available_to_summon_preflight() -> None:
    checkpoint = _checkpoint()
    owner, account = load_one_star_account(checkpoint)
    account.state.resources.gems = 5
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")

    preflight_one_star_account_updates(
        checkpoint,
        [
            OneStarStateUpdate(
                kind="gem_purchase",
                target_id="gems",
                value="20",
                details=[],
            ),
            OneStarStateUpdate(
                kind="summon",
                target_id="premium",
                value="5",
                details=[],
            ),
        ],
        initiating_actor_id="account_owner",
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            [OneStarStateUpdate(
                kind="inventory_delta",
                target_id="gems",
                value="100",
                details=[],
            )],
            "sold only in packs of 20",
        ),
        (
            [OneStarStateUpdate(
                kind="gem_purchase",
                target_id="gems",
                value="25",
                details=[],
            )],
            "25 Gems cannot be purchased",
        ),
        (
            [OneStarStateUpdate(
                kind="gem_purchase",
                target_id="gems",
                value="60",
                details=[],
            )],
            "60 Gems cost",
        ),
    ],
)
def test_invalid_gem_acquisition_is_player_rejected_without_mutation(
    updates: list[OneStarStateUpdate],
    message: str,
) -> None:
    checkpoint = _checkpoint()
    before = checkpoint.model_dump_json()

    with pytest.raises(PlayerActionRejected, match=message):
        preflight_one_star_account_updates(
            checkpoint,
            updates,
            initiating_actor_id="account_owner",
        )

    assert checkpoint.model_dump_json() == before


def test_counter_advancement_exposes_the_unconsumed_weighted_suffix() -> None:
    checkpoint = _checkpoint()
    preview = one_star_summon_draw_preview(checkpoint, "premium", count=5)
    owner = checkpoint.characters[0]
    owner.mechanics[ONE_STAR_ACCOUNT_KEY]["state"]["summon_draw_counters"] = {
        "premium": 2
    }

    suffix = one_star_summon_draw_preview(checkpoint, "premium", count=3)
    assert [draw.birth_stars for draw in suffix] == [
        draw.birth_stars for draw in preview[2:]
    ]


def test_exact_prefix_commits_once_and_failed_substitution_cannot_reroll() -> None:
    checkpoint = _checkpoint()
    preview = one_star_summon_draw_preview(checkpoint, "basic", count=2)
    exact_ids = [draw.existing_character_id for draw in preview]
    exact_stars = [draw.birth_stars for draw in preview]
    activation_locations = {hero_id: "lobby" for hero_id in exact_ids}

    with pytest.raises(OneStarTransactionError, match="exact reserve"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="substitute",
            transaction=_summon_transaction(
                pool_id="basic",
                hero_ids=list(reversed(exact_ids)),
                birth_stars=exact_stars,
            ),
            activated_character_ids=exact_ids,
            activated_character_locations=activation_locations,
            initiating_actor_id="account_owner",
        )
    assert load_one_star_account(checkpoint)[1].state.summon_draw_counters == {}
    assert one_star_summon_draw_preview(checkpoint, "basic", count=2) == preview

    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="valid_pull",
        transaction=_summon_transaction(
            pool_id="basic",
            hero_ids=exact_ids,
            birth_stars=exact_stars,
        ),
        activated_character_ids=exact_ids,
        activated_character_locations=activation_locations,
        initiating_actor_id="account_owner",
    )
    assert load_one_star_account(checkpoint)[1].state.summon_draw_counters == {}
    prepared_account = load_one_star_account(prepared.after_checkpoint)[1]
    assert prepared_account.state.summon_draw_counters == {"basic": 2}
    assert prepared_account.state.resources.gold == 96

    assert apply_one_star_prepared_mutation(checkpoint, prepared) is True
    assert load_one_star_account(checkpoint)[1].state.summon_draw_counters == {
        "basic": 2
    }
    assert apply_one_star_prepared_mutation(checkpoint, prepared) is False
    assert load_one_star_account(checkpoint)[1].state.summon_draw_counters == {
        "basic": 2
    }
    replay = prepare_one_star_transaction(
        checkpoint,
        event_id="valid_pull",
        transaction=_summon_transaction(
            pool_id="basic",
            hero_ids=exact_ids,
            birth_stars=exact_stars,
        ),
        activated_character_ids=exact_ids,
        activated_character_locations=activation_locations,
        initiating_actor_id="account_owner",
    )
    assert replay.already_applied is True
    assert load_one_star_account(checkpoint)[1].state.summon_draw_counters == {
        "basic": 2
    }


def test_exhausted_reserve_grade_falls_back_to_fresh_generated_hero() -> None:
    checkpoint = _checkpoint()
    initial = one_star_summon_draw_preview(checkpoint, "basic", count=2)
    reserve_ids = [draw.existing_character_id for draw in initial]
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="consume_reserves",
        transaction=_summon_transaction(
            pool_id="basic",
            hero_ids=reserve_ids,
            birth_stars=[1, 1],
        ),
        activated_character_ids=reserve_ids,
        activated_character_locations={hero_id: "lobby" for hero_id in reserve_ids},
        initiating_actor_id="account_owner",
    )
    apply_one_star_prepared_mutation(checkpoint, prepared)
    next_draw = one_star_summon_draw_preview(checkpoint, "basic", count=1)[0]
    assert next_draw.existing_character_id == ""

    fresh_id = "fresh_roll"
    checkpoint.characters.append(_hero(
        fresh_id,
        status=CharacterStatus.active,
    ))
    fresh = prepare_one_star_transaction(
        checkpoint,
        event_id="fresh_pull",
        transaction=_summon_transaction(
            pool_id="basic",
            hero_ids=[fresh_id],
            birth_stars=[next_draw.birth_stars],
        ),
        spawned_character_ids=[fresh_id],
        initiating_actor_id="account_owner",
    )
    assert load_one_star_account(fresh.after_checkpoint)[1].state.summon_draw_counters == {
        "basic": 3
    }


def test_wrong_weighted_birth_grade_is_rejected_without_consumption() -> None:
    checkpoint = _checkpoint()
    draw = one_star_summon_draw_preview(checkpoint, "premium", count=1)[0]
    fresh_id = "premium_fresh"
    checkpoint.characters.append(_hero(
        fresh_id,
        birth_stars=draw.birth_stars + 1 if draw.birth_stars < 5 else 4,
        status=CharacterStatus.active,
    ))
    with pytest.raises(OneStarTransactionError, match="exact next weighted"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="wrong_grade",
            transaction=_summon_transaction(
                pool_id="premium",
                hero_ids=[fresh_id],
                birth_stars=[
                    draw.birth_stars + 1 if draw.birth_stars < 5 else 4
                ],
            ),
            spawned_character_ids=[fresh_id],
            initiating_actor_id="account_owner",
        )
    assert load_one_star_account(checkpoint)[1].state.summon_draw_counters == {}
