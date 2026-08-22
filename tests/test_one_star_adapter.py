from __future__ import annotations

import pytest

from app.engine.one_star_adapter import (
    OneStarTransactionError,
    apply_one_star_prepared_mutation,
    load_one_star_account,
    load_one_star_hero,
    prepare_one_star_transaction,
)
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import OneStarTransaction
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
        "hero_constraints": {
            "minimum_hp_max": 1,
            "maximum_hp_max": 50,
            "maximum_xp": 1000,
            "maximum_stat_value": 20,
            "maximum_equipment_entries": 5,
            "maximum_skill_entries": 5,
        },
        "floor_rewards": {
            "1": {"gold": 4, "gems": 0, "building_resources": 1, "materials": {}}
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
                    "facilities": {"tower_gate": 1},
                },
            }
        },
    )
    reserve = CharacterRecord(
        character_id="reserve",
        name="Reserve",
        status=CharacterStatus.dormant,
        mechanics={
            "one_star_hero": {
                "birth_stars": 1,
                "current_stars": 1,
                "level": 1,
                "experience_points": 0,
                "hp_current": 7,
                "hp_max": 7,
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


def test_transaction_requires_exact_present_shape() -> None:
    with pytest.raises(ValueError, match="present"):
        OneStarTransaction.model_validate({"present": False, "operations": [{"operation": "active_feed", "hero_id": ""}]})


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
