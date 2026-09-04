"""Small offline One-Star fixtures shared by adapter contract tests."""

from __future__ import annotations

from copy import deepcopy

from app.engine.one_star_progression import rebalance_hero
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    OneStarHeroState,
    OneStarMissionCounter,
    OneStarMissionState,
    OneStarPendingOperation,
    OneStarRulesConfig,
    OneStarTransaction,
)
from app.schemas.state import (
    KnowledgeTier,
    SessionConfig,
    SessionSettings,
    SessionState,
    WorldState,
)


def one_star_config() -> dict:
    return {
        "lobby_id": "lobby_a",
        "lobby_location_label": "lobby",
        "starting_resources": {
            "gold": 20,
            "gems": 0,
            "building_resources": 0,
            "materials": {},
        },
        "catalogue": {
            "synthesis_chamber": {
                "kind": "facility_build",
                "cost": {
                    "gold": 1,
                    "gems": 0,
                    "building_resources": 0,
                    "materials": {},
                },
                "facility_id": "synthesis_chamber",
                "target_level": 1,
            },
            "promotion_chamber": {
                "kind": "facility_build",
                "cost": {
                    "gold": 1,
                    "gems": 0,
                    "building_resources": 0,
                    "materials": {},
                },
                "facility_id": "promotion_chamber",
                "target_level": 1,
            },
        },
        "summon_pools": {
            "basic": {
                "cost": {
                    "gold": 1,
                    "gems": 0,
                    "building_resources": 0,
                    "materials": {},
                },
                "minimum_birth_stars": 1,
                "maximum_birth_stars": 1,
                "star_weights": {1: 10_000},
                "eligible_existing_ids": ["reserve"],
                "fresh_generation_allowed": False,
                "usage": "standard",
            }
        },
        "star_level_caps": {"1": 10, "2": 20},
        "starting_lobby_floor": 1,
        "starting_capacity": 10,
        "maximum_stamina": 5,
        "stamina_recovery_seconds": 30,
        "deployment_stamina_cost": 1,
        "max_summon_batch": 5,
        "progression": {
            "stat_ids": ["power", "agility", "spirit"],
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
            "1": {
                "gold": 4,
                "gems": 0,
                "building_resources": 1,
                "materials": {},
            },
        },
        "floor_scenarios": {
            "1": {
                "mission_id": "mission_1",
                "destination": "tower_floor_1",
                "premise": "Clear the first floor.",
                "completion_declaration": "the floor is cleared",
                "failure_declaration": "the party is broken",
                "counters": [
                    {"counter_id": "clear", "current": 0, "target": 1}
                ],
                "pressures": ["The floor presses the party forward."],
            }
        },
        "repeat_gold_numerator": 1,
        "repeat_gold_denominator": 4,
        "repeat_gold_minimum": 1,
        "promotion_cost": {
            "gold": 2,
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
        "lobby_return_healing": True,
        "hero_system_visibility_research_key": "",
    }


def one_star_hero(
    *,
    status: CharacterStatus = CharacterStatus.active,
    location: str = "lobby",
    owner: str = "lobby_a",
    level: int = 1,
    xp: int = 0,
) -> CharacterRecord:
    character = CharacterRecord(
        character_id="hero",
        name="Hero",
        status=status,
        location=location,
        mechanics={
            ONE_STAR_HERO_KEY: {
                "birth_stars": 1,
                "current_stars": 1,
                "level": level,
                "experience_points": xp,
                "hp_current": 7,
                "hp_max": 7,
                "stats": {"power": 6, "agility": 5, "spirit": 4},
                "owner_lobby_id": owner,
                "acquisition_event_id": "seed" if owner else "",
                "terminal_event_id": "",
                "progression_seed": "atomicity_hero",
                "strong_stat_id": "power",
                "weak_stat_id": "spirit",
                "potential_grade": 1,
            }
        },
    )
    raw_config = one_star_config()
    raw_config["star_level_caps"]["1"] = max(10, level)
    config = OneStarRulesConfig.model_validate(raw_config)
    hero = OneStarHeroState.model_validate(character.mechanics[ONE_STAR_HERO_KEY])
    rebalance_hero(hero=hero, config=config, restore_full_hp=True)
    character.mechanics[ONE_STAR_HERO_KEY] = hero.model_dump(mode="json")
    return character


def one_star_mission(
    *,
    destination: str = "tower_floor_1",
    party: list[str] | None = None,
) -> OneStarMissionState:
    return OneStarMissionState(
        mission_id="mission_1",
        floor=1,
        party_ids=party or ["hero"],
        destination=destination,
        completion_declaration="the floor is cleared",
        failure_declaration="the party is broken",
        counters=[OneStarMissionCounter(counter_id="clear", current=0, target=1)],
        started_at_s=0,
        deadline_at_s=0,
    )


def one_star_checkpoint(
    *,
    heroes: list[CharacterRecord] | None = None,
    active_mission: OneStarMissionState | None = None,
    pending_operation: OneStarPendingOperation | None = None,
    stamina_current: int = 5,
    stamina_anchor: int = 0,
    knowledge_tiers: list[KnowledgeTier] | None = None,
) -> CheckpointFile:
    state = {
        "resources": deepcopy(one_star_config()["starting_resources"]),
        "stored_equipment": [],
        "lobby_floor": 1,
        "capacity": 10,
        "highest_unlocked_floor": 1,
        "highest_cleared_floor": 0,
        "stamina_current": stamina_current,
        "stamina_recovery_anchor_s": stamina_anchor,
        "facilities": {
            "tower_gate": 1,
            "synthesis_chamber": 1,
            "promotion_chamber": 1,
        },
    }
    if active_mission is not None:
        state["active_mission"] = active_mission.model_dump(mode="json")
    if pending_operation is not None:
        state["pending_operation"] = pending_operation.model_dump(mode="json")
    owner = CharacterRecord(
        character_id="account_owner",
        name="Account Owner",
        mechanics={
            ONE_STAR_ACCOUNT_KEY: {
                "config": one_star_config(),
                "state": state,
            }
        },
    )
    return CheckpointFile(
        session=SessionState(
            session_id="one_star_atomicity",
            config=SessionConfig(
                settings=SessionSettings(ruleset_id="one_star_ascension")
            ),
        ),
        world_state=WorldState(knowledge_tiers=knowledge_tiers or []),
        characters=[owner, *(heroes or [one_star_hero()])],
    )


def one_star_transaction(*operations: dict) -> OneStarTransaction:
    return OneStarTransaction.model_validate({
        "present": bool(operations),
        "operations": list(operations),
    })


def marked_mission_update(*, current: int) -> dict:
    return {
        "operation": "mission_update",
        "mission_id": "mission_1",
        "counters": [{"counter_id": "clear", "current": current, "target": 1}],
    }


def mission_end(*, outcome: str, escape_authority_id: str = "") -> dict:
    return {
        "operation": "mission_end",
        "mission_id": "mission_1",
        "outcome": outcome,
        "return_destination": "lobby" if outcome != "failed" else "",
        "escape_authority_id": escape_authority_id,
    }
