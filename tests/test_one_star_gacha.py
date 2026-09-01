"""Offline contracts for adapter-owned One-Star summon draws."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from app.engine.action_rejection import PlayerActionRejected
from app.engine.one_star_adapter import (
    OneStarTransactionError,
    apply_one_star_prepared_mutation,
    load_one_star_account,
    one_star_birth_stars_for_ticket,
    one_star_opening_roster_preview,
    one_star_state_updates_to_transaction,
    one_star_summon_lifecycle,
    one_star_summon_draw_preview,
    preflight_one_star_account_updates,
    prepare_one_star_transaction,
)
from app.schemas.characters import CharacterRecord, CharacterStatus, PlayerSlotKind
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    OneStarCost,
    OneStarSummonPool,
    OneStarStandardSummonPool,
    OneStarStateUpdate,
    OneStarTransaction,
)
from app.schemas.state import SessionConfig, SessionSettings, SessionState


def _pool(
    minimum: int,
    maximum: int,
    weights: dict[int, int],
) -> OneStarStandardSummonPool:
    return OneStarStandardSummonPool(
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
        "floor_rewards": {
            "1": {
                "gold": 0,
                "gems": 0,
                "building_resources": 0,
                "materials": {},
            },
        },
        "floor_scenarios": {
            "1": {
                "mission_id": "floor_1_goblin_ambush",
                "destination": "tower_floor_1_goblin_ambush",
                "premise": "Survive the goblin ambush and reach the exit.",
                "completion_declaration": (
                    "At least one Hero survived the goblin ambush and reached "
                    "the exit."
                ),
                "failure_declaration": (
                    "No Hero remains alive and able to reach the exit."
                ),
                "counters": [
                    {
                        "counter_id": "survivor_reaches_exit",
                        "current": 0,
                        "target": 1,
                    },
                ],
                "pressure_beats": ["Armed goblins attack immediately."],
            },
        },
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


def _direct_opening_updates(
    checkpoint: CheckpointFile,
    *,
    pool_id: str,
    party_ids: list[str],
) -> list[OneStarStateUpdate]:
    _owner, account = load_one_star_account(checkpoint)
    scenario = account.config.floor_scenarios[1]
    return [
        OneStarStateUpdate(
            kind="summon",
            target_id=pool_id,
            value=str(len(party_ids)),
            details=[],
        ),
        OneStarStateUpdate(
            kind="mission_start",
            target_id=scenario.mission_id,
            value="1",
            details=[
                *(f"party={hero_id}" for hero_id in party_ids),
                f"destination={scenario.destination}",
                f"completion={scenario.completion_declaration}",
                f"failure={scenario.failure_declaration}",
                *(
                    f"counter.{counter.counter_id}="
                    f"{counter.current}/{counter.target}"
                    for counter in scenario.counters
                ),
            ],
        ),
    ]


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
    pool: OneStarStandardSummonPool,
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
    spawns, wakes = one_star_summon_lifecycle(checkpoint, [update])

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


def test_legacy_opening_wave_pool_shape_is_retired() -> None:
    with pytest.raises(ValidationError, match="opening_wave"):
        TypeAdapter(OneStarSummonPool).validate_python({
            "usage": "opening_wave",
            "cost": {
                "gold": 0,
                "gems": 0,
                "building_resources": 0,
                "materials": {},
            },
            "minimum_birth_stars": 1,
            "maximum_birth_stars": 1,
            "star_weights": {1: 10_000},
            "eligible_existing_ids": [],
            "fresh_generation_allowed": True,
        })


def test_opening_roster_preview_is_stable_ordered_and_uses_all_authored_heroes() -> None:
    checkpoint = _checkpoint()
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["summon_pools"]["opening"] = {
        "usage": "opening_roster",
        "slots": [
            {"kind": "fixed", "character_id": "edren"},
            {"kind": "fixed", "character_id": "renna"},
            {"kind": "random_existing_grade", "birth_stars": 3},
        ],
    }
    edren = _hero("edren")
    renna = _hero("renna")
    wren = _hero("wren", birth_stars=3)
    mirelle = _hero("mirelle", birth_stars=3)
    player_authored = _hero("blank_player", birth_stars=3)
    player_authored.player_slot_kind = PlayerSlotKind.player_authored
    active = _hero("active_three", birth_stars=3, status=CharacterStatus.active)
    owned = _hero("owned_three", birth_stars=3, owner="another_lobby")
    terminal = _hero("terminal_three", birth_stars=3)
    terminal.mechanics[ONE_STAR_HERO_KEY]["terminal_event_id"] = "death"
    checkpoint.characters.extend([
        edren,
        renna,
        wren,
        mirelle,
        player_authored,
        active,
        owned,
        terminal,
    ])

    preview = one_star_opening_roster_preview(checkpoint, "opening")
    replay = CheckpointFile.model_validate_json(checkpoint.model_dump_json())

    assert one_star_opening_roster_preview(replay, "opening") == preview
    assert [draw.existing_character_id for draw in preview[:2]] == [
        "edren",
        "renna",
    ]
    assert preview[2].existing_character_id in {"wren", "mirelle"}
    assert [draw.birth_stars for draw in preview] == [1, 1, 3]


def test_opening_roster_future_authored_grade_is_eligible_without_allowlist() -> None:
    checkpoint = _checkpoint()
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["summon_pools"]["opening"] = {
        "usage": "opening_roster",
        "slots": [{"kind": "random_existing_grade", "birth_stars": 3}],
    }
    checkpoint.characters.append(_hero("future_authored_three", birth_stars=3))

    assert one_star_opening_roster_preview(
        checkpoint,
        "opening",
    )[0].existing_character_id == "future_authored_three"


def test_opening_roster_missing_grade_fails_loudly_without_mutation() -> None:
    checkpoint = _checkpoint()
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["summon_pools"]["opening"] = {
        "usage": "opening_roster",
        "slots": [{"kind": "random_existing_grade", "birth_stars": 3}],
    }
    before = checkpoint.model_dump_json()

    with pytest.raises(OneStarTransactionError, match="no eligible.*birth-3"):
        one_star_opening_roster_preview(checkpoint, "opening")

    assert checkpoint.model_dump_json() == before


def test_opening_roster_compact_update_owns_transaction_and_wake_identities() -> None:
    checkpoint = _checkpoint()
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["summon_pools"]["opening"] = {
        "usage": "opening_roster",
        "slots": [
            {"kind": "fixed", "character_id": "edren"},
            {"kind": "random_existing_grade", "birth_stars": 3},
        ],
    }
    checkpoint.characters.extend([
        _hero("edren"),
        _hero("authored_three", birth_stars=3),
    ])
    preview = one_star_opening_roster_preview(checkpoint, "opening")
    party_ids = [draw.existing_character_id for draw in preview]
    updates = _direct_opening_updates(
        checkpoint,
        pool_id="opening",
        party_ids=party_ids,
    )

    transaction = one_star_state_updates_to_transaction(
        checkpoint,
        updates,
        canonical_at_s=0,
    )
    spawns, wakes = one_star_summon_lifecycle(checkpoint, updates)
    operation = transaction.operations[0]

    assert operation.hero_ids == party_ids
    assert operation.birth_stars == [draw.birth_stars for draw in preview]
    assert spawns == ()
    assert [wake.character_id for wake in wakes] == operation.hero_ids
    assert all(
        wake.location_label == "tower_floor_1_goblin_ambush"
        for wake in wakes
    )


def test_opening_roster_cannot_stop_before_the_direct_first_mission() -> None:
    checkpoint = _checkpoint()
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["summon_pools"]["opening"] = {
        "usage": "opening_roster",
        "slots": [{"kind": "fixed", "character_id": "edren"}],
    }
    checkpoint.characters.append(_hero("edren"))
    update = OneStarStateUpdate(
        kind="summon",
        target_id="opening",
        value="1",
        details=[],
    )

    with pytest.raises(
        OneStarTransactionError,
        match="must be followed by exactly one direct mission start",
    ):
        one_star_summon_lifecycle(checkpoint, [update])


def test_bound_player_actor_roster_activates_its_exact_bound_slot() -> None:
    checkpoint = _checkpoint()
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["summon_pools"]["newcomer_opening"] = {
        "usage": "opening_roster",
        "slots": [{"kind": "bound_player_actor", "character_id": "newcomer"}],
    }
    newcomer = _hero("newcomer")
    newcomer.player_slot_kind = PlayerSlotKind.player_authored
    checkpoint.characters.append(newcomer)
    checkpoint.session.character_bindings["newcomer"] = "player-1"
    updates = _direct_opening_updates(
        checkpoint,
        pool_id="newcomer_opening",
        party_ids=["newcomer"],
    )

    transaction = one_star_state_updates_to_transaction(
        checkpoint,
        updates,
        canonical_at_s=0,
    )
    spawns, wakes = one_star_summon_lifecycle(checkpoint, updates)
    operation = transaction.operations[0]

    assert operation.hero_ids == ["newcomer"]
    assert operation.birth_stars == [1]
    assert spawns == ()
    assert [(wake.character_id, wake.location_label) for wake in wakes] == [
        ("newcomer", "tower_floor_1_goblin_ambush"),
    ]


def test_direct_opening_atomically_acquires_roster_and_starts_floor_one() -> None:
    checkpoint = _checkpoint()
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["summon_pools"]["opening"] = {
        "usage": "opening_roster",
        "slots": [
            {"kind": "fixed", "character_id": "renna"},
            {"kind": "fixed", "character_id": "edren"},
        ],
    }
    checkpoint.characters.extend([_hero("renna"), _hero("edren")])
    updates = _direct_opening_updates(
        checkpoint,
        pool_id="opening",
        party_ids=["renna", "edren"],
    )
    transaction = one_star_state_updates_to_transaction(
        checkpoint,
        updates,
        canonical_at_s=0,
    )
    _spawns, wakes = one_star_summon_lifecycle(checkpoint, updates)
    activation_locations = {
        wake.character_id: wake.location_label for wake in wakes
    }

    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="direct_opening",
        transaction=transaction,
        activated_character_ids=list(activation_locations),
        activated_character_locations=activation_locations,
        initiating_actor_id="account_owner",
    )
    prepared_account = load_one_star_account(prepared.after_checkpoint)[1]

    assert prepared_account.state.pending_operation is None
    assert prepared_account.state.active_mission is not None
    assert prepared_account.state.active_mission.mission_id == (
        "floor_1_goblin_ambush"
    )
    assert prepared_account.state.active_mission.party_ids == ["renna", "edren"]
    assert prepared_account.state.stamina_current == 4
    for hero_id in ("renna", "edren"):
        hero = next(
            character
            for character in prepared.after_checkpoint.characters
            if character.character_id == hero_id
        )
        assert hero.mechanics[ONE_STAR_HERO_KEY]["owner_lobby_id"] == "lobby_a"
        assert hero.mechanics[ONE_STAR_HERO_KEY]["acquisition_event_id"] == (
            "direct_opening"
        )


@pytest.mark.parametrize(
    ("player_authored", "bound", "error"),
    [
        (False, True, "not a player-authored slot"),
        (True, False, "no live player binding"),
    ],
)
def test_bound_player_actor_roster_requires_exact_authored_live_binding(
    player_authored: bool,
    bound: bool,
    error: str,
) -> None:
    checkpoint = _checkpoint()
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["summon_pools"]["newcomer_opening"] = {
        "usage": "opening_roster",
        "slots": [{"kind": "bound_player_actor", "character_id": "newcomer"}],
    }
    newcomer = _hero("newcomer")
    if player_authored:
        newcomer.player_slot_kind = PlayerSlotKind.player_authored
    checkpoint.characters.append(newcomer)
    if bound:
        checkpoint.session.character_bindings["newcomer"] = "player-1"
    before = checkpoint.model_dump_json()

    with pytest.raises(OneStarTransactionError, match=error):
        one_star_opening_roster_preview(checkpoint, "newcomer_opening")

    assert checkpoint.model_dump_json() == before


def test_every_summon_rejects_router_authored_identity_details() -> None:
    checkpoint = _checkpoint()
    update = OneStarStateUpdate(
        kind="summon",
        target_id="basic",
        value="1",
        details=["hero_id=reserve_a"],
    )

    with pytest.raises(OneStarTransactionError, match="unsupported details"):
        one_star_state_updates_to_transaction(
            checkpoint,
            [update],
            canonical_at_s=0,
        )


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

    with pytest.raises(OneStarTransactionError, match="exact adapter preview"):
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

    update = OneStarStateUpdate(
        kind="summon",
        target_id="basic",
        value="1",
        details=[],
    )
    transaction = one_star_state_updates_to_transaction(
        checkpoint,
        [update],
        canonical_at_s=0,
    )
    fresh_id = transaction.operations[0].hero_ids[0]
    assert fresh_id == "lobby_a_basic_0003"
    checkpoint.characters.append(_hero(
        fresh_id,
        status=CharacterStatus.active,
    ))
    fresh = prepare_one_star_transaction(
        checkpoint,
        event_id="fresh_pull",
        transaction=transaction,
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
    with pytest.raises(OneStarTransactionError, match="exact adapter preview"):
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
