from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.engine.one_star_progression import (
    apply_experience,
    apply_promotion_banked_experience,
    banked_experience_at_current_cap,
    birth_hp_mean,
    birth_stat_total_mean,
    build_generated_hero,
    experience_to_reach_level,
    rebalance_hero,
    remaining_experience_capacity,
)
from app.engine.one_star_adapter import (
    OneStarTransactionError,
    prepare_one_star_transaction,
)
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import (
    OneStarCost,
    OneStarOperationRequirement,
    OneStarProgressionConfig,
    OneStarResources,
    OneStarRulesConfig,
    OneStarTransaction,
)
from app.schemas.state import SessionConfig, SessionSettings, SessionState
from app.schemas.one_star_character_gen import (
    AuthoredOneStarHiddenCapability,
    AuthoredOneStarHeroMechanics,
)
from app.schemas.takeover import AuthoredCharacter


def _config(*, first_star_cap: int = 4) -> OneStarRulesConfig:
    return OneStarRulesConfig(
        starting_resources=OneStarResources(
            gold=0,
            gems=0,
            building_resources=0,
        ),
        lobby_id="lobby",
        lobby_location_label="lobby",
        catalogue={},
        summon_pools={
            "basic": {
                "cost": {"gold": 0, "gems": 0, "building_resources": 0},
                "minimum_birth_stars": 1,
                "maximum_birth_stars": 1,
                "star_weights": {1: 10_000},
                "usage": "standard",
            }
        },
        star_level_caps={
            1: first_star_cap,
            2: 4,
            3: 6,
            4: 8,
            5: 10,
            6: 12,
            7: 14,
        },
        starting_lobby_floor=1,
        starting_capacity=1,
        maximum_stamina=0,
        stamina_recovery_seconds=1,
        deployment_stamina_cost=0,
        max_summon_batch=1,
        progression=OneStarProgressionConfig(
            stat_ids=["power", "agility", "resilience"],
            grade_multiplier_milli=1_250,
            birth_stat_total=15,
            birth_hp_max=8,
            variance_basis_points=500,
            stat_growth_per_level_milli=1_000,
            hp_growth_per_level_milli=500,
            xp_threshold_factor=50,
            floor_xp_per_floor=100,
            overlevel_xp_percentages=[100, 75, 50, 25, 10, 5, 0],
            cap_bank_extra_levels=1,
            synthesis_source_base_xp=100,
            synthesis_skill_chance_basis_points=500,
        ),
        floor_rewards={},
        floor_scenarios={},
        repeat_gold_numerator=0,
        repeat_gold_denominator=1,
        repeat_gold_minimum=0,
        promotion_cost=OneStarCost(gold=0, gems=0, building_resources=0),
        operation_requirements={
            "deployment": OneStarOperationRequirement(
                facility_id="gate", required_location=""
            ),
            "synthesis": OneStarOperationRequirement(
                facility_id="forge", required_location="forge"
            ),
            "promotion": OneStarOperationRequirement(
                facility_id="altar", required_location="altar"
            ),
        },
        lobby_return_healing=True,
    )


def _generated(
    *, strong_stat_id: str = "power", weak_stat_id: str = "resilience"
) -> AuthoredOneStarHeroMechanics:
    return AuthoredOneStarHeroMechanics.model_validate(
        {
            "strong_stat_id": strong_stat_id,
            "weak_stat_id": weak_stat_id,
            "equipment": [
                {
                    "item_id": "iron blade",
                    "name": "Iron Blade",
                    "slot": "hand",
                    "quantity": 1,
                    "durability_current": 0,
                    "durability_max": 0,
                    "tags": [],
                    "visible": True,
                },
                {
                    "item_id": "iron blade",
                    "name": "Spare Iron Blade",
                    "slot": "pack",
                    "quantity": 1,
                    "durability_current": 0,
                    "durability_max": 0,
                    "tags": [],
                    "visible": False,
                },
            ],
            "skills": [],
            "conditions": [],
            "persistent_injuries": [],
            "innate_system_sight": False,
            "hidden_capabilities": [
                AuthoredOneStarHiddenCapability(
                    capability_id="quiet_step",
                    description="Moves quietly over old stone.",
                )
            ],
        }
    )


def _hero(*, config: OneStarRulesConfig, character_id: str = "hero"):
    return build_generated_hero(
        character_id=character_id,
        generated=_generated(),
        birth_stars=1,
        config=config,
    )


@pytest.mark.parametrize(
    ("grade", "stat_total", "hp_max"),
    [
        (1, 15, 8),
        (2, 19, 10),
        (3, 23, 13),
        (4, 29, 16),
        (5, 37, 20),
        (6, 46, 24),
        (7, 57, 31),
    ],
)
def test_birth_grade_means_use_fixed_point_half_up(
    grade: int, stat_total: int, hp_max: int
) -> None:
    config = _config()

    assert birth_stat_total_mean(grade, config) == stat_total
    assert birth_hp_mean(grade, config) == hp_max


def test_generated_hero_is_replay_stable_and_normalises_equipment_ids() -> None:
    config = _config()
    first = _hero(config=config, character_id="hero.alpha")
    replay = _hero(config=config, character_id="hero.alpha")

    assert first.model_dump(mode="json") == replay.model_dump(mode="json")
    assert [item.item_id for item in first.equipment] == [
        "hero_alpha_iron_blade",
        "hero_alpha_iron_blade_2",
    ]
    assert first.hidden_capabilities == {"quiet_step": "Moves quietly over old stone."}
    assert first.stats
    assert first.hp_current == first.hp_max > 0


def test_birth_allocation_keeps_strong_high_and_weak_low() -> None:
    hero = _hero(config=_config())

    assert hero.stats[hero.strong_stat_id] > hero.stats["agility"]
    assert hero.stats["agility"] > hero.stats[hero.weak_stat_id]


def test_experience_replay_is_order_independent_and_rebalances_existing_level() -> None:
    config = _config()
    incrementally = _hero(config=config)
    in_one_award = _hero(config=config)

    apply_experience(hero=incrementally, experience_delta=125, config=config)
    apply_experience(hero=incrementally, experience_delta=475, config=config)
    apply_experience(hero=in_one_award, experience_delta=600, config=config)

    assert incrementally.model_dump(mode="json") == in_one_award.model_dump(mode="json")
    existing = _hero(config=config)
    existing.level = 2
    existing.experience_points = 0
    rebalance_hero(hero=existing, config=config, restore_full_hp=True)
    assert existing.experience_points == experience_to_reach_level(2, config)
    assert existing.hp_current == existing.hp_max


def test_hidden_potential_changes_growth_not_the_level_one_birth_sheet() -> None:
    config = _config()
    ordinary = _hero(config=config, character_id="renna")
    exceptional = ordinary.model_copy(deep=True, update={"potential_grade": 4})

    assert exceptional.stats == ordinary.stats
    assert exceptional.hp_max == ordinary.hp_max
    apply_experience(hero=ordinary, experience_delta=600, config=config)
    apply_experience(hero=exceptional, experience_delta=600, config=config)

    assert sum(exceptional.stats.values()) > sum(ordinary.stats.values())
    assert exceptional.hp_max > ordinary.hp_max


def test_xp_application_rejects_inconsistent_replay_state() -> None:
    config = _config()
    hero = _hero(config=config)
    hero.experience_points = experience_to_reach_level(2, config)

    with pytest.raises(ValueError, match="unrealized next level"):
        apply_experience(hero=hero, experience_delta=0, config=config)

    hero.level = 2
    hero.experience_points = 0
    with pytest.raises(ValueError, match="below its reached level"):
        apply_experience(hero=hero, experience_delta=0, config=config)


def test_model_overlay_cannot_author_numeric_progression_or_arbitrary_maps() -> None:
    schema = AuthoredOneStarHeroMechanics.model_json_schema()
    properties = schema["properties"]

    assert not {
        "level",
        "experience_points",
        "hp_current",
        "hp_max",
        "stats",
        "private_potential",
        "potential_grade",
    } & set(properties)
    assert properties["hidden_capabilities"]["type"] == "array"
    with pytest.raises(ValidationError):
        AuthoredOneStarHeroMechanics.model_validate(
            {
                **_generated().model_dump(mode="json"),
                "potential_grade": 7,
            }
        )


def test_cap_banking_and_promotion_carryover_are_deterministic() -> None:
    config = _config(first_star_cap=2)
    hero = _hero(config=config)

    report = apply_experience(hero=hero, experience_delta=1_000, config=config)

    assert (report.offered_xp, report.applied_xp, report.wasted_xp) == (1_000, 300, 700)
    assert hero.level == 2
    assert banked_experience_at_current_cap(hero, config) == 200
    assert remaining_experience_capacity(hero, config) == 0
    potential_grade = hero.potential_grade
    hero.current_stars = 2
    carryover = apply_promotion_banked_experience(hero=hero, config=config)

    assert carryover.levels_gained == 1
    assert hero.level == 3
    assert hero.experience_points == experience_to_reach_level(3, config)
    assert hero.potential_grade == potential_grade


def test_generic_character_generation_remains_ruleset_free() -> None:
    generic = AuthoredCharacter.model_validate(
        {
            "name": "Mara",
            "location": "hall",
            "role": "scout",
            "appearance": "Plain travel clothes.",
            "default_loadout": "A walking staff.",
            "faction": "",
            "backstory": "A traveller.",
            "personality": "Cautious.",
            "known_context": "Only the hall.",
            "goals": [],
            "current_objectives": [],
            "secrets": [],
            "intentions_enabled": True,
            "router_summary": "",
        }
    )

    assert generic.to_record(character_id="mara").mechanics == {}


def test_progression_config_rejects_totals_that_variation_can_make_unallocatable() -> None:
    values = _config().progression.model_dump(mode="python")
    values.update(birth_stat_total=6, variance_basis_points=1)

    with pytest.raises(ValidationError, match="strict strong, middle, and weak"):
        OneStarProgressionConfig.model_validate(values)


def test_hero_potential_cannot_be_below_birth_grade() -> None:
    values = _hero(config=_config()).model_dump(mode="python")
    values.update(birth_stars=2, current_stars=2, potential_grade=1)

    with pytest.raises(ValidationError, match="potential_grade"):
        type(_hero(config=_config())).model_validate(values)


@pytest.mark.parametrize("field", ["stats", "hp_max"])
def test_prepare_rejects_corrupted_deterministic_hero_sheet(field: str) -> None:
    config = _config()
    hero = _hero(config=config)
    hero_values = hero.model_dump(mode="python")
    if field == "stats":
        hero_values["stats"]["power"] += 1
    else:
        hero_values["hp_max"] += 1
        hero_values["hp_current"] += 1
    owner = CharacterRecord(
        character_id="owner",
        name="Owner",
        mechanics={
            "one_star_account": {
                "config": config.model_dump(mode="python"),
                "state": {
                    "resources": config.starting_resources.model_dump(mode="python"),
                    "stored_equipment": [],
                    "facilities": {"gate": 1, "forge": 1, "altar": 1},
                    "lobby_floor": 1,
                    "capacity": 1,
                    "stamina_current": 0,
                },
            }
        },
    )
    character = CharacterRecord(
        character_id="hero",
        name="Hero",
        mechanics={"one_star_hero": hero_values},
    )
    checkpoint = CheckpointFile(
        session=SessionState(
            session_id="corrupt_progression",
            config=SessionConfig(
                settings=SessionSettings(ruleset_id="one_star_ascension")
            ),
        ),
        characters=[owner, character],
    )

    with pytest.raises(OneStarTransactionError, match="deterministic progression"):
        prepare_one_star_transaction(
            checkpoint,
            event_id=f"evt_corrupt_{field}",
            transaction=OneStarTransaction(present=False, operations=[]),
        )
