from __future__ import annotations

import pytest

from app.engine.one_star_adapter import (
    OneStarTransactionError,
    apply_one_star_prepared_mutation,
    load_one_star_account,
    load_one_star_hero,
    one_star_state_updates_to_transaction,
    prepare_one_star_transaction,
)
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterStatus,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import (
    OneStarRulesConfig,
    OneStarStateUpdate,
    OneStarTransaction,
)
from app.schemas.state import OpenCatIIEvent, SessionConfig, SessionSettings, SessionState


def _config() -> dict:
    return {
        "lobby_id": "lobby_a",
        "lobby_location_label": "lobby",
        "starting_resources": {
            "gold": 10,
            "gems": 0,
            "building_resources": 0,
            "materials": {},
        },
        "catalogue": {
            "synthesis_chamber_1": {
                "kind": "facility_build",
                "cost": {"gold": 2, "gems": 0, "building_resources": 0, "materials": {}},
                "facility_id": "synthesis_chamber",
                "target_level": 1,
            },
            "promotion_chamber_1": {
                "kind": "facility_build",
                "cost": {"gold": 2, "gems": 0, "building_resources": 0, "materials": {}},
                "facility_id": "promotion_chamber",
                "target_level": 1,
            },
        },
        "summon_pools": {
            "basic": {
                "cost": {"gold": 2, "gems": 0, "building_resources": 0, "materials": {}},
                "minimum_birth_stars": 1,
                "maximum_birth_stars": 1,
                "star_weights": {1: 10_000},
                "eligible_existing_ids": ["reserve"],
                "fresh_generation_allowed": True,
                "usage": "standard",
            }
        },
        "star_level_caps": {"1": 10, "2": 20, "3": 40},
        "starting_lobby_floor": 1,
        "starting_capacity": 5,
        "maximum_stamina": 5,
        "stamina_recovery_seconds": 30,
        "deployment_stamina_cost": 1,
        "max_summon_batch": 5,
        "progression": {
            "stat_ids": ["power", "agility", "resilience"],
            "grade_multiplier_milli": 1250,
            "birth_stat_total": 15,
            "birth_hp_max": 7,
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
            "1": {"gold": 4, "gems": 0, "building_resources": 1, "materials": {}}
        },
        "floor_scenarios": {
            "1": {
                "mission_id": "mission_1",
                "destination": "tower_floor_1",
                "premise": "Clear the first floor.",
                "completion_declaration": "the floor is cleared",
                "failure_declaration": "the party is broken",
                "counters": [{"counter_id": "clear", "current": 0, "target": 1}],
                "pressure_beats": ["The floor presses the party forward."],
            }
        },
        "repeat_gold_numerator": 1,
        "repeat_gold_denominator": 4,
        "repeat_gold_minimum": 1,
        "promotion_cost": {"gold": 3, "gems": 0, "building_resources": 0, "materials": {}},
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
        "hero_system_visibility_research_key": "hero_system",
    }


def _checkpoint() -> CheckpointFile:
    master = CharacterRecord(
        character_id="account_owner",
        name="Account Owner",
        mechanics={
            "one_star_account": {
                "config": _config(),
                "state": {
                    "resources": {"gold": 10, "gems": 0, "building_resources": 0, "materials": {}},
                    "lobby_floor": 1,
                    "capacity": 5,
                    "stamina_current": 5,
                    "discretionary_funds": 200,
                    "funds_accrual_anchor_s": 0,
                    "facilities": {"tower_gate": 1},
                    "stored_equipment": [],
                },
            }
        },
    )
    reserve = CharacterRecord(
        character_id="reserve",
        name="Reserve",
        status=CharacterStatus.dormant,
        agent_tier=CharacterAgentTier.utility,
        mechanics={
            "one_star_hero": {
                "birth_stars": 1,
                "current_stars": 1,
                "level": 1,
                "experience_points": 0,
                "hp_current": 7,
                "hp_max": 7,
                "stats": {"power": 6, "agility": 5, "resilience": 4},
                "terminal_event_id": "",
                "progression_seed": "reserve_progression_seed",
                "strong_stat_id": "power",
                "weak_stat_id": "resilience",
                "potential_grade": 1,
            }
        },
    )
    return CheckpointFile(
        session=SessionState(
            session_id="one_star_test",
            config=SessionConfig(settings=SessionSettings(ruleset_id="one_star_ascension")),
        ),
        characters=[master, reserve],
    )


def test_floor_scenarios_may_be_a_reviewed_subset_of_rewarded_floors() -> None:
    config = _config()
    config["floor_rewards"]["2"] = {
        "gold": 6,
        "gems": 0,
        "building_resources": 1,
        "materials": {},
    }

    parsed = OneStarRulesConfig.model_validate(config)

    assert set(parsed.floor_rewards) == {1, 2}
    assert set(parsed.floor_scenarios) == {1}


def test_floor_scenario_requires_a_matching_reward() -> None:
    config = _config()
    config["floor_scenarios"]["2"] = {
        "mission_id": "mission_2",
        "destination": "tower_floor_2",
        "premise": "Clear the second floor.",
        "completion_declaration": "the second floor is cleared",
        "failure_declaration": "the party is broken",
        "counters": [{"counter_id": "clear", "current": 0, "target": 1}],
        "pressure_beats": ["The second floor presses the party forward."],
    }

    with pytest.raises(ValueError, match="must have a configured floor reward"):
        OneStarRulesConfig.model_validate(config)


def test_multiple_opening_rosters_may_require_guide_handoff() -> None:
    config = _config()
    config["summon_pools"].update({
        "master_opening": {
            "usage": "opening_roster",
            "slots": [{"kind": "fixed", "character_id": "master_hero"}],
            "initial_deployment_requires_guide_handoff": True,
        },
        "duo_opening": {
            "usage": "opening_roster",
            "slots": [{"kind": "fixed", "character_id": "duo_hero"}],
            "initial_deployment_requires_guide_handoff": True,
        },
    })

    parsed = OneStarRulesConfig.model_validate(config)

    assert parsed.summon_pools["master_opening"].usage == "opening_roster"
    assert parsed.summon_pools["duo_opening"].usage == "opening_roster"


def test_transaction_requires_exact_present_shape() -> None:
    with pytest.raises(ValueError, match="present"):
        OneStarTransaction.model_validate({
            "present": False,
            "operations": [{
                "operation": "inventory_delta",
                "item_id": "gold",
                "quantity_delta": 1,
            }],
        })


def test_retired_active_feed_is_not_a_router_update() -> None:
    with pytest.raises(ValueError):
        OneStarStateUpdate.model_validate({
            "kind": "active_feed",
            "target_id": "hero",
            "value": "",
            "details": [],
        })


def test_automatic_stamina_recovery_is_returned_as_one_time_history_state() -> None:
    ckpt = _checkpoint()
    _owner, account = load_one_star_account(ckpt)
    account.state.stamina_current = 3
    account.state.stamina_recovery_anchor_s = 0
    ckpt.characters[0].mechanics["one_star_account"] = account.model_dump(mode="json")
    ckpt.session.leading_at_s = 60

    prepared = prepare_one_star_transaction(
        ckpt,
        event_id="evt_stamina_recovery",
        transaction=OneStarTransaction(present=False, operations=[]),
    )

    assert prepared.engine_history_updates == (
        "stamina_recovered current=5 recovery_anchor_s=60",
    )

    full = _checkpoint()
    full.session.leading_at_s = 60
    no_recovery = prepare_one_star_transaction(
        full,
        event_id="evt_no_stamina_recovery",
        transaction=OneStarTransaction(present=False, operations=[]),
    )
    assert no_recovery.engine_history_updates == ()


def test_existing_reserve_keeps_authored_mechanics_and_acquires_atomically() -> None:
    checkpoint = _checkpoint()
    transaction = OneStarTransaction.model_validate(
        {
            "present": True,
            "operations": [
                {
                    "operation": "summon",
                    "pool_id": "basic",
                    "hero_ids": ["reserve"],
                    "birth_stars": [1],
                }
            ],
        }
    )

    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="evt_summon",
        transaction=transaction,
        activated_character_ids=["reserve"],
        activated_character_locations={"reserve": "lobby"},
        initiating_actor_id="account_owner",
    )

    reserve = next(c for c in prepared.after_checkpoint.characters if c.character_id == "reserve")
    hero = load_one_star_hero(reserve)
    assert hero is not None
    assert (hero.hp_current, hero.hp_max, hero.owner_lobby_id, hero.acquisition_event_id) == (7, 7, "lobby_a", "evt_summon")
    _owner, account = load_one_star_account(prepared.after_checkpoint)
    assert account.state.resources.gold == 8
    assert prepared.newly_acquired_hero_ids == ("reserve",)
    assert hero.generated_for_summon is False
    assert reserve.agent_tier is CharacterAgentTier.utility
    assert reserve.private_state.intentions_enabled is True


def test_inventory_delta_routes_account_currencies_to_resources() -> None:
    checkpoint = _checkpoint()
    updates = [
        OneStarStateUpdate(
            kind="inventory_delta",
            target_id="gold",
            value="-3",
            details=[],
        ),
        OneStarStateUpdate(
            kind="inventory_delta",
            target_id="building_resources",
            value="2",
            details=[],
        ),
        OneStarStateUpdate(
            kind="inventory_delta",
            target_id="healing_draught",
            value="4",
            details=[],
        ),
    ]
    transaction = one_star_state_updates_to_transaction(
        checkpoint,
        updates,
        canonical_at_s=0,
    )

    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="evt_resource_grant",
        transaction=transaction,
        initiating_actor_id="account_owner",
    )
    _owner, account = load_one_star_account(prepared.after_checkpoint)

    assert account.state.resources.model_dump() == {
        "gold": 7,
        "gems": 0,
        "building_resources": 2,
        "materials": {},
    }
    assert account.state.inventory == {"healing_draught": 4}


def test_positive_gem_inventory_delta_is_not_an_acquisition_path() -> None:
    checkpoint = _checkpoint()
    transaction = one_star_state_updates_to_transaction(
        checkpoint,
        [OneStarStateUpdate(
            kind="inventory_delta",
            target_id="gems",
            value="100",
            details=[],
        )],
        canonical_at_s=0,
    )

    with pytest.raises(OneStarTransactionError, match="positive Gem changes"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="evt_unbacked_gems",
            transaction=transaction,
            initiating_actor_id="account_owner",
        )


def test_gem_purchase_and_weekly_funds_accrual_are_exact_and_idempotent() -> None:
    checkpoint = _checkpoint()
    purchase = one_star_state_updates_to_transaction(
        checkpoint,
        [OneStarStateUpdate(
            kind="gem_purchase",
            target_id="gems",
            value="20",
            details=[],
        )],
        canonical_at_s=0,
    )
    prepared_purchase = prepare_one_star_transaction(
        checkpoint,
        event_id="evt_gem_purchase",
        transaction=purchase,
        canonical_at_s=0,
        initiating_actor_id="account_owner",
    )
    _owner, purchased = load_one_star_account(
        prepared_purchase.after_checkpoint
    )
    assert purchased.state.discretionary_funds == 100
    assert purchased.state.resources.gems == 20
    assert apply_one_star_prepared_mutation(checkpoint, prepared_purchase)

    weekly_accrual = prepare_one_star_transaction(
        checkpoint,
        event_id="evt_weekly_accrual",
        transaction=OneStarTransaction(present=False, operations=[]),
        canonical_at_s=604_800,
    )
    _owner, accrued = load_one_star_account(weekly_accrual.after_checkpoint)
    assert accrued.state.discretionary_funds == 200
    assert accrued.state.funds_accrual_anchor_s == 604_800
    assert weekly_accrual.engine_history_updates == (
        "discretionary_funds_accrued current=200 "
        "accrual_anchor_s=604800",
    )
    assert apply_one_star_prepared_mutation(checkpoint, weekly_accrual)

    no_duplicate = prepare_one_star_transaction(
        checkpoint,
        event_id="evt_same_week",
        transaction=OneStarTransaction(present=False, operations=[]),
        canonical_at_s=604_800,
    )
    assert no_duplicate.engine_history_updates == ()
    _owner, unchanged = load_one_star_account(no_duplicate.after_checkpoint)
    assert unchanged.state.discretionary_funds == 200


@pytest.mark.parametrize(
    ("resource_id", "delta"),
    [("gold", -11), ("gems", -1), ("building_resources", -1)],
)
def test_inventory_delta_rejects_account_currency_underflow(
    resource_id: str,
    delta: int,
) -> None:
    checkpoint = _checkpoint()
    transaction = one_star_state_updates_to_transaction(
        checkpoint,
        [OneStarStateUpdate(
            kind="inventory_delta",
            target_id=resource_id,
            value=str(delta),
            details=[],
        )],
        canonical_at_s=0,
    )

    with pytest.raises(OneStarTransactionError, match="resource delta"):
        prepare_one_star_transaction(
            checkpoint,
            event_id=f"evt_underflow_{resource_id}",
            transaction=transaction,
            initiating_actor_id="account_owner",
        )


def test_preparation_uses_durable_copy_and_apply_preserves_transients() -> None:
    class Uncopyable:
        def __deepcopy__(self, memo):  # pragma: no cover - must never be called.
            raise AssertionError("deep copy touched live runtime")

    checkpoint = _checkpoint()
    open_event = OpenCatIIEvent(
        event_id="cat_ii_open",
        initiator_id="account_owner",
        initiator_intention="I select the reserve.",
        required_responders=["reserve"],
    )
    checkpoint.session.open_cat_ii_events = [open_event]
    live_session = checkpoint.session
    sentinel = Uncopyable()
    checkpoint.__dict__["_closed_event_runtime"] = sentinel
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="evt_empty",
        transaction=OneStarTransaction.model_validate({"present": False, "operations": []}),
    )

    assert apply_one_star_prepared_mutation(checkpoint, prepared) is True
    assert checkpoint.__dict__["_closed_event_runtime"] is sentinel
    assert checkpoint.session is live_session
    assert checkpoint.session.open_cat_ii_events[0] is open_event


def test_summon_rejects_batch_larger_than_seed_limit() -> None:
    checkpoint = _checkpoint()
    transaction = OneStarTransaction.model_validate(
        {
            "present": True,
            "operations": [
                {
                    "operation": "summon",
                    "pool_id": "basic",
                    "hero_ids": ["a", "b", "c", "d", "e", "f"],
                    "birth_stars": [1, 1, 1, 1, 1, 1],
                }
            ],
        }
    )
    with pytest.raises(OneStarTransactionError, match="maximum batch"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="evt_too_many",
            transaction=transaction,
            spawned_character_ids=["a", "b", "c", "d", "e", "f"],
            initiating_actor_id="account_owner",
        )
