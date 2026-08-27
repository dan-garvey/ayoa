"""Offline reducer tests for the One-Star Ascension adapter.

These tests exercise the adapter's durable transaction boundary.  They do not
call a provider, render a prompt, or start a live harness.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.engine.one_star_adapter import (
    OneStarTransactionError,
    apply_one_star_prepared_mutation,
    effective_one_star_stamina,
    load_one_star_account,
    load_one_star_hero,
    prepare_one_star_transaction,
)
from app.engine.one_star_progression import rebalance_hero
from app.schemas.characters import CharacterAgentTier, CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.events import CanonicalEvent
from app.schemas.event_router import CommitmentOpenSignal
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    OneStarMissionCounter,
    OneStarMissionState,
    OneStarPendingOperation,
    OneStarPendingOperationSelection,
    OneStarTransaction,
    OneStarEventRouterOutput,
    OneStarHeroState,
    OneStarRulesConfig,
)
from app.schemas.state import KnowledgeTier, SessionConfig, SessionSettings, SessionState, WorldState


def _config() -> dict:
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
                "cost": {"gold": 1, "gems": 0, "building_resources": 0, "materials": {}},
                "facility_id": "synthesis_chamber",
                "target_level": 1,
            },
            "promotion_chamber": {
                "kind": "facility_build",
                "cost": {"gold": 1, "gems": 0, "building_resources": 0, "materials": {}},
                "facility_id": "promotion_chamber",
                "target_level": 1,
            },
        },
        "summon_pools": {
            "basic": {
                "cost": {"gold": 1, "gems": 0, "building_resources": 0, "materials": {}},
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
            "1": {"gold": 4, "gems": 0, "building_resources": 1, "materials": {}},
        },
        "repeat_gold_numerator": 1,
        "repeat_gold_denominator": 4,
        "repeat_gold_minimum": 1,
        "promotion_cost": {"gold": 2, "gems": 0, "building_resources": 0, "materials": {}},
        "operation_requirements": {
            "deployment": {"facility_id": "tower_gate", "required_location": ""},
            "synthesis": {"facility_id": "synthesis_chamber", "required_location": "synthesis_room"},
            "promotion": {"facility_id": "promotion_chamber", "required_location": "promotion_room"},
        },
        "lobby_return_healing": True,
        "hero_system_visibility_research_key": "",
    }


def _hero(*, status: CharacterStatus = CharacterStatus.active, location: str = "lobby", owner: str = "lobby_a", level: int = 1, xp: int = 0) -> CharacterRecord:
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
    raw_config = _config()
    raw_config["star_level_caps"]["1"] = max(10, level)
    config = OneStarRulesConfig.model_validate(raw_config)
    hero = OneStarHeroState.model_validate(character.mechanics[ONE_STAR_HERO_KEY])
    rebalance_hero(hero=hero, config=config, restore_full_hp=True)
    character.mechanics[ONE_STAR_HERO_KEY] = hero.model_dump(mode="json")
    return character


def _mission(*, destination: str = "tower_floor_1", party: list[str] | None = None) -> OneStarMissionState:
    return OneStarMissionState(
        mission_id="mission_1",
        floor=1,
        party_ids=party or ["hero"],
        formation_labels=[
            {"character_id": cid, "label": "front"}
            for cid in (party or ["hero"])
        ],
        destination=destination,
        completion_declaration="the floor is cleared",
        failure_declaration="the party is broken",
        counters=[OneStarMissionCounter(counter_id="clear", current=0, target=1)],
        started_at_s=0,
        deadline_at_s=0,
    )


def _checkpoint(
    *,
    heroes: list[CharacterRecord] | None = None,
    active_mission: OneStarMissionState | None = None,
    pending_operation: OneStarPendingOperation | None = None,
    stamina_current: int = 5,
    stamina_anchor: int = 0,
    knowledge_tiers: list[KnowledgeTier] | None = None,
) -> CheckpointFile:
    state = {
        "resources": deepcopy(_config()["starting_resources"]),
        "stored_equipment": [],
        "lobby_floor": 1,
        "capacity": 10,
        "highest_unlocked_floor": 1,
        "highest_cleared_floor": 0,
        "stamina_current": stamina_current,
        "stamina_recovery_anchor_s": stamina_anchor,
        "facilities": {"tower_gate": 1, "synthesis_chamber": 1, "promotion_chamber": 1},
    }
    if active_mission is not None:
        state["active_mission"] = active_mission.model_dump(mode="json")
    if pending_operation is not None:
        state["pending_operation"] = pending_operation.model_dump(mode="json")
    owner = CharacterRecord(
        character_id="account_owner",
        name="Account Owner",
        mechanics={ONE_STAR_ACCOUNT_KEY: {"config": _config(), "state": state}},
    )
    return CheckpointFile(
        session=SessionState(
            session_id="one_star_atomicity",
            config=SessionConfig(settings=SessionSettings(ruleset_id="one_star_ascension")),
        ),
        world_state=WorldState(knowledge_tiers=knowledge_tiers or []),
        characters=[owner, *(heroes or [_hero()])],
    )


def _transaction(*operations: dict) -> OneStarTransaction:
    return OneStarTransaction.model_validate({"present": bool(operations), "operations": list(operations)})


def _hero_delta(hero_id: str = "hero", **overrides: object) -> dict:
    operation = {
        "operation": "hero_delta",
        "hero_id": hero_id,
        "hp_current": None,
        "equipment_add": [],
        "equipment_remove_ids": [],
        "skills_add": [],
        "skills_remove_ids": [],
        "equipment_durability": [],
        "skill_rank_updates": [],
        "conditions": None,
        "persistent_injuries": None,
        "terminal_action": "none",
        "death_cause": "",
    }
    operation.update(overrides)
    return operation


def test_ordinary_non_hero_spawn_does_not_require_a_summon_transaction() -> None:
    checkpoint = _checkpoint()
    recurring_npc = CharacterRecord(
        character_id="recurring_rival",
        name="Recurring Rival",
        location="tower_antechamber",
    )
    checkpoint.characters.append(recurring_npc)

    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="ordinary_non_hero_spawn",
        transaction=_transaction(),
        spawned_character_ids=[recurring_npc.character_id],
    )

    spawned = next(
        character
        for character in prepared.after_checkpoint.characters
        if character.character_id == recurring_npc.character_id
    )
    assert ONE_STAR_HERO_KEY not in spawned.mechanics
    assert (
        "ordinary_non_hero_spawn"
        in load_one_star_account(prepared.after_checkpoint)[1]
        .state.applied_event_fingerprints
    )


def _pending(
    kind: str,
    *,
    target: str = "",
    participants: list[str] | None = None,
    destination: str,
) -> OneStarPendingOperationSelection:
    return OneStarPendingOperationSelection(
        operation_id=f"{kind}_1",
        kind=kind,
        participant_ids=participants or ([target] if target else ["hero"]),
        target_id=target,
        destination=destination,
        opened_at_s=0,
    )


@pytest.mark.parametrize("destination", ["lobby", "synthesis_room", "promotion_room"])
def test_active_mission_party_cannot_cross_sealed_boundary_with_location_update(destination: str) -> None:
    checkpoint = _checkpoint(
        heroes=[_hero(location="tower_floor_1")],
        active_mission=_mission(),
    )

    with pytest.raises(OneStarTransactionError, match="sealed Tower boundary"):
        prepare_one_star_transaction(
            checkpoint,
            event_id=f"bad_return_{destination}",
            transaction=_transaction(),
            location_updates={"hero": destination},
        )


def test_active_mission_rejects_pending_open_and_completed_end_returns_survivor() -> None:
    pending = _pending("synthesis", target="hero", destination="synthesis_room")
    checkpoint = _checkpoint(
        heroes=[_hero(location="tower_floor_1")],
        active_mission=_mission(),
    )
    returning_sheet = checkpoint.characters[1].mechanics[ONE_STAR_HERO_KEY]
    returning_sheet["hp_current"] = 2
    returning_sheet["conditions"] = ["bleeding", "poisoned", "exhausted"]
    returning_sheet["persistent_injuries"] = ["missing fingertip"]
    owner, account_before = load_one_star_account(checkpoint)
    account_before.state.active_master_feed_id = "hero"
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account_before.model_dump(mode="json")
    with pytest.raises(OneStarTransactionError, match="active"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="bad_open",
            transaction=_transaction({
                "operation": "pending_open",
                "pending": pending.model_dump(mode="json"),
            }),
            initiating_actor_id="account_owner",
        )

    with pytest.raises(OneStarTransactionError, match="declared counter"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="premature_mission_complete",
            transaction=_transaction({
                "operation": "mission_end",
                "mission_id": "mission_1",
                "outcome": "completed",
                "return_destination": "lobby",
                "escape_authority_id": "",
            }),
            canonical_at_s=1,
        )

    completed = prepare_one_star_transaction(
        checkpoint,
        event_id="mission_complete",
        transaction=_transaction(
            {
                "operation": "mission_update",
                "mission_id": "mission_1",
                "counters": [
                    {"counter_id": "clear", "current": 1, "target": 1},
                ],
            },
            {
                "operation": "mission_end",
                "mission_id": "mission_1",
                "outcome": "completed",
                "return_destination": "lobby",
                "escape_authority_id": "",
            },
        ),
        canonical_at_s=1,
    )
    character = next(item for item in completed.after_checkpoint.characters if item.character_id == "hero")
    account = load_one_star_account(completed.after_checkpoint)[1]
    returned_hero = load_one_star_hero(character)
    assert returned_hero is not None
    assert character.location == "lobby"
    assert returned_hero.hp_current == returned_hero.hp_max
    assert returned_hero.conditions == []
    assert returned_hero.persistent_injuries == ["missing fingertip"]
    assert account.state.active_mission is None
    assert account.state.highest_cleared_floor == 1
    assert account.state.highest_unlocked_floor == 1
    assert "active_master_feed_id=none previous=hero" in completed.engine_history_updates


def test_active_mission_rejects_undeployed_reinforcement_and_party_dormancy() -> None:
    party_hero = _hero(location="tower_floor_1")
    reinforcement = _hero(location="lobby")
    reinforcement.character_id = "reinforcement"
    checkpoint = _checkpoint(
        heroes=[party_hero, reinforcement],
        active_mission=_mission(),
    )

    with pytest.raises(OneStarTransactionError, match="cannot join"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="undeployed_reinforcement",
            transaction=_transaction(),
            location_updates={"reinforcement": "tower_floor_1"},
        )
    with pytest.raises(OneStarTransactionError, match="party lifecycle"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="dormant_escape",
            transaction=_transaction(),
            dormant_character_ids=["hero"],
        )


@pytest.mark.parametrize("character_id", ["missing", "hero"])
def test_dormant_requires_a_live_unpinned_character(character_id: str) -> None:
    checkpoint = _checkpoint()
    if character_id == "hero":
        checkpoint.characters[1].status = CharacterStatus.culled
    with pytest.raises(
        OneStarTransactionError,
        match="cannot become dormant|unknown character id",
    ):
        prepare_one_star_transaction(
            checkpoint,
            event_id=f"dormant_{character_id}",
            transaction=_transaction(),
            dormant_character_ids=[character_id],
        )


def test_culled_character_cannot_be_activated() -> None:
    checkpoint = _checkpoint()
    checkpoint.characters[1].status = CharacterStatus.culled
    with pytest.raises(OneStarTransactionError, match="cannot be activated"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="activate_culled",
            transaction=_transaction(),
            activated_character_ids=["hero"],
            activated_character_locations={"hero": "lobby"},
        )


def test_active_mission_rejects_lifecycle_and_summon_reinforcements() -> None:
    party_hero = _hero(location="tower_floor_1")
    local_reserve = _hero(status=CharacterStatus.dormant, location="lobby")
    local_reserve.character_id = "local_reserve"
    checkpoint = _checkpoint(
        heroes=[party_hero, local_reserve],
        active_mission=_mission(),
    )
    with pytest.raises(OneStarTransactionError, match="cannot activate"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="activate_reinforcement",
            transaction=_transaction(),
            activated_character_ids=["local_reserve"],
            activated_character_locations={"local_reserve": "tower_floor_1"},
        )

    unowned_reserve = _hero(
        status=CharacterStatus.dormant,
        location="lobby",
        owner="",
    )
    unowned_reserve.character_id = "reserve"
    reserve_checkpoint = _checkpoint(
        heroes=[party_hero.model_copy(deep=True), unowned_reserve],
        active_mission=_mission(),
    )
    with pytest.raises(OneStarTransactionError, match="active mission"):
        prepare_one_star_transaction(
            reserve_checkpoint,
            event_id="summon_reinforcement",
            transaction=_transaction({
                "operation": "summon",
                "pool_id": "basic",
                "hero_ids": ["reserve"],
                "birth_stars": [1],
            }),
            activated_character_ids=["reserve"],
            activated_character_locations={"reserve": "tower_floor_1"},
            initiating_actor_id="account_owner",
        )


def test_account_owner_actor_label_does_not_block_router_world_hero_delta() -> None:
    checkpoint = _checkpoint(
        heroes=[_hero(location="tower_floor_1")],
        active_mission=_mission(),
    )
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="owner_observes_environmental_harm",
        transaction=_transaction(_hero_delta(hp_current=5)),
        initiating_actor_id="account_owner",
    )
    hero = next(
        character
        for character in prepared.after_checkpoint.characters
        if character.character_id == "hero"
    )
    assert load_one_star_hero(hero).hp_current == 5


def test_resolved_cat_ii_can_kill_despite_original_master_initiator() -> None:
    checkpoint = _checkpoint(
        pending_operation=_pending(
            "deployment",
            participants=["hero"],
            destination="tower_floor_1",
        )
    )
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="warden_kills_resisting_hero",
        transaction=_transaction(
            _hero_delta(
                hp_current=0,
                terminal_action="death",
                death_cause="killed by the lobby's defender",
            ),
            {
                "operation": "pending_cancel",
                "operation_id": "deployment_1",
            },
        ),
        initiating_actor_id="account_owner",
    )
    hero = next(
        character
        for character in prepared.after_checkpoint.characters
        if character.character_id == "hero"
    )
    assert hero.status is CharacterStatus.culled


def test_guide_tutorial_delivery_records_active_recipient_exactly_once() -> None:
    guide = CharacterRecord(character_id="guide", name="Guide")
    checkpoint = _checkpoint(heroes=[_hero(), guide])
    account_raw = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]
    account_raw["state"]["guide_character_ids"] = ["guide"]
    operation = {
        "operation": "tutorial_delivery",
        "tutorial_key": "tower_gate",
        "delivered_to_ids": ["hero"],
    }
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="tutorial_once",
        transaction=_transaction(operation),
        initiating_actor_id="guide",
    )
    account = load_one_star_account(prepared.after_checkpoint)[1]
    assert account.state.tutorial_deliveries == {"tower_gate": ["hero"]}

    with pytest.raises(OneStarTransactionError, match="exactly once"):
        prepare_one_star_transaction(
            prepared.after_checkpoint,
            event_id="tutorial_duplicate",
            transaction=_transaction(operation),
            initiating_actor_id="guide",
        )

    fresh = _hero(location="tower_floor_1", owner="")
    fresh.character_id = "fresh"
    fresh_checkpoint = _checkpoint(heroes=[fresh])
    fresh_checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"][
        "summon_pools"
    ]["basic"].update(
        {
            "eligible_existing_ids": [],
            "fresh_generation_allowed": True,
        }
    )
    with pytest.raises(OneStarTransactionError, match="configured lobby"):
        prepare_one_star_transaction(
            fresh_checkpoint,
            event_id="fresh_wrong_location",
            transaction=_transaction({
                "operation": "summon",
                "pool_id": "basic",
                "hero_ids": ["fresh"],
                "birth_stars": [1],
            }),
            spawned_character_ids=["fresh"],
            initiating_actor_id="account_owner",
        )


def test_delayed_full_stamina_anchor_does_not_instantly_refill_after_spend() -> None:
    checkpoint = _checkpoint(stamina_current=4, stamina_anchor=0)
    account = load_one_star_account(checkpoint)[1]
    recovered, anchor = effective_one_star_stamina(account.state, account.config, 60)
    assert (recovered, anchor) == (5, 60)

    account.state.stamina_current = recovered - 1
    account.state.stamina_recovery_anchor_s = anchor
    assert effective_one_star_stamina(account.state, account.config, 60) == (4, 60)


@pytest.mark.parametrize("hp_current", [0, 8])
def test_malformed_hp_requires_valid_bounds_or_explicit_terminal_cull(hp_current: int) -> None:
    checkpoint = _checkpoint()
    with pytest.raises(OneStarTransactionError):
        prepare_one_star_transaction(
            checkpoint,
            event_id=f"bad_hp_{hp_current}",
            transaction=_transaction(_hero_delta(hp_current=hp_current)),
        )


def test_existing_reserve_must_be_dormant_before_summon_activation() -> None:
    reserve = _hero(status=CharacterStatus.active, owner="", location="lobby")
    reserve.character_id = "reserve"
    checkpoint = _checkpoint(heroes=[reserve])
    transaction = _transaction({
        "operation": "summon",
        "pool_id": "basic",
        "hero_ids": ["reserve"],
        "birth_stars": [1],
    })
    with pytest.raises(OneStarTransactionError, match="no eligible reserve"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="summon_active_reserve",
            transaction=transaction,
            activated_character_ids=["reserve"],
            activated_character_locations={"reserve": "lobby"},
            initiating_actor_id="account_owner",
        )


def test_terminal_hero_cull_cannot_be_overwritten_by_generic_dormancy() -> None:
    checkpoint = _checkpoint()
    with pytest.raises(OneStarTransactionError, match="terminal.*dormancy"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="death_and_dormancy",
            transaction=_transaction(
                _hero_delta(
                    hp_current=0,
                    terminal_action="death",
                    death_cause="killed in the Tower",
                )
            ),
            dormant_character_ids=["hero"],
        )


def test_exact_event_replay_is_idempotent_but_payload_drift_is_rejected() -> None:
    checkpoint = _checkpoint()
    transaction = _transaction()
    prepared = prepare_one_star_transaction(checkpoint, event_id="same", transaction=transaction)
    assert apply_one_star_prepared_mutation(checkpoint, prepared) is True

    replay = prepare_one_star_transaction(checkpoint, event_id="same", transaction=transaction)
    assert replay.already_applied is True
    with pytest.raises(OneStarTransactionError, match="reused"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="same",
            transaction=_transaction({"operation": "active_feed", "hero_id": "hero"}),
        )


def test_synthesis_resolve_derives_selected_source_culls() -> None:
    source_a = _hero(location="synthesis_room", owner="lobby_a")
    source_a.character_id = "source_a"
    source_b = _hero(location="synthesis_room", owner="lobby_a")
    source_b.character_id = "source_b"
    target = _hero(location="synthesis_room", owner="lobby_a")
    target.character_id = "target"
    checkpoint = _checkpoint(heroes=[source_a, source_b, target])
    opened = prepare_one_star_transaction(
        checkpoint,
        event_id="synthesis_open",
        transaction=_transaction({
            "operation": "pending_open",
            "pending": {
                "operation_id": "synthesis_1",
                "kind": "synthesis",
                "participant_ids": ["source_a", "source_b"],
                "target_id": "target",
                "destination": "synthesis_room",
                "opened_at_s": 0,
            },
        }),
        initiating_actor_id="account_owner",
    )
    improved = prepare_one_star_transaction(
        opened.after_checkpoint,
        event_id="synthesis_resolve",
        transaction=_transaction({
            "operation": "pending_resolve",
            "operation_id": "synthesis_1",
        }),
    )
    by_id = {item.character_id: item for item in improved.after_checkpoint.characters}
    assert by_id["source_a"].status is CharacterStatus.culled
    assert by_id["source_b"].status is CharacterStatus.culled
    assert load_one_star_hero(by_id["target"]).experience_points > 0


@pytest.mark.parametrize(
    ("generated_for_summon", "starting_tier", "expected_tier"),
    [
        (True, CharacterAgentTier.standard, CharacterAgentTier.standard),
        (False, CharacterAgentTier.utility, CharacterAgentTier.premium),
    ],
)
def test_promotion_preserves_level_xp_and_restores_reviewed_knowledge(
    generated_for_summon: bool,
    starting_tier: CharacterAgentTier,
    expected_tier: CharacterAgentTier,
) -> None:
    target = _hero(location="promotion_room", level=10, xp=4_500)
    target.agent_tier = starting_tier
    target.mechanics[ONE_STAR_HERO_KEY]["generated_for_summon"] = (
        generated_for_summon
    )
    pending = _pending("promotion", target="hero", destination="promotion_room")
    checkpoint = _checkpoint(
        heroes=[target],
        pending_operation=pending,
        knowledge_tiers=[KnowledgeTier(
            tier=2,
            personal_depth="a buried memory",
            world_knowledge="the gate's truth",
            agent_tier=CharacterAgentTier.premium,
        )],
    )
    account = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]
    account["config"]["visual_novel_presentation"] = {
        "veiled_sprite_set_ids": {
            "masculine": "veiled-masculine",
            "feminine": "veiled-feminine",
        },
        "seeded_birth_one_reveal_stars": 2,
        "generated_birth_one_reveal_stars": 3,
    }
    checkpoint.session.visual_introductions = {
        "account_owner": ["hero"],
    }
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="promotion",
        transaction=_transaction({
            "operation": "pending_resolve",
            "operation_id": "promotion_1",
        }),
    )
    promoted = next(item for item in prepared.after_checkpoint.characters if item.character_id == "hero")
    hero = load_one_star_hero(promoted)
    assert hero is not None
    assert (hero.current_stars, hero.level, hero.experience_points) == (2, 10, 4_500)
    assert promoted.knowledge_tier == 2
    assert promoted.agent_tier is expected_tier
    assert "a buried memory" in promoted.known_context
    assert "the gate's truth" in promoted.known_context
    assert (
        "hero"
        in prepared.after_checkpoint.session.visual_introductions.get(
            "account_owner",
            [],
        )
    ) is generated_for_summon


def test_pending_open_is_the_only_operation_and_cannot_change_its_heroes() -> None:
    pending = _pending("deployment", participants=["hero"], destination="tower_floor_1")
    opening = {
        "operation": "pending_open",
        "pending": pending.model_dump(mode="json"),
    }
    cases = [
        ({"operations": [opening, {"operation": "active_feed", "hero_id": ""}]}, {}, {}, ["hero"], [], []),
        ({"operations": [opening]}, {"hero": "lobby"}, {}, [], [], []),
        ({"operations": [opening]}, {}, {"hero": "lobby"}, [], [], []),
        ({"operations": [opening]}, {}, {}, [], ["hero"], []),
        ({"operations": [opening]}, {}, {}, [], [], ["hero"]),
    ]
    for index, (raw, locations, activation_locations, activated, dormant, spawned) in enumerate(cases):
        transaction = OneStarTransaction.model_validate({
            "present": True,
            "operations": raw["operations"],
        })
        with pytest.raises(OneStarTransactionError):
            prepare_one_star_transaction(
                _checkpoint(),
                event_id=f"pending_open_invalid_{index}",
                transaction=transaction,
                location_updates=locations,
                activated_character_ids=activated,
                activated_character_locations=activation_locations,
                dormant_character_ids=dormant,
                spawned_character_ids=spawned,
            )


def test_deployment_pending_open_rejects_a_separate_target_hero() -> None:
    pending = _pending(
        "deployment",
        target="hero",
        participants=["hero"],
        destination="tower_floor_1",
    )

    with pytest.raises(
        OneStarTransactionError,
        match="deployment has no separate target Hero",
    ):
        prepare_one_star_transaction(
            _checkpoint(),
            event_id="deployment_with_target",
            transaction=_transaction({
                "operation": "pending_open",
                "pending": pending.model_dump(mode="json"),
            }),
            initiating_actor_id="account_owner",
        )


def test_deployment_gate_crossing_requires_atomic_resolution_and_mission_start() -> None:
    pending = _pending("deployment", participants=["hero"], destination="tower_floor_1")
    opened = prepare_one_star_transaction(
        _checkpoint(),
        event_id="pending_open",
        transaction=_transaction({
            "operation": "pending_open",
            "pending": pending.model_dump(mode="json"),
        }),
        initiating_actor_id="account_owner",
    )
    checkpoint = opened.after_checkpoint
    assert load_one_star_account(checkpoint)[1].state.pending_operation is not None

    with pytest.raises(OneStarTransactionError, match="same canonical event"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="crosses_without_resolution",
            transaction=_transaction(),
            location_updates={"hero": "tower_floor_1"},
        )

    resolved = prepare_one_star_transaction(
        checkpoint,
        event_id="crosses_and_deploys",
        transaction=_transaction(
            {
                "operation": "pending_resolve",
                "operation_id": pending.operation_id,
            },
            {
                "operation": "mission_start",
                "mission": _mission().model_dump(mode="json"),
                "pending_operation_id": pending.operation_id,
            },
        ),
        location_updates={"hero": "tower_floor_1"},
    )
    account = load_one_star_account(resolved.after_checkpoint)[1]
    assert account.state.pending_operation is None
    assert account.state.active_mission is not None
    assert next(
        item
        for item in resolved.after_checkpoint.characters
        if item.character_id == "hero"
    ).location == "tower_floor_1"


def test_crossed_pending_deployment_cannot_cancel_or_return() -> None:
    pending = _pending("deployment", participants=["hero"], destination="tower_floor_1")
    checkpoint = _checkpoint(
        heroes=[_hero(location="tower_floor_1")],
        pending_operation=pending,
    )

    with pytest.raises(OneStarTransactionError, match="same canonical event"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="cancel_after_crossing",
            transaction=_transaction({
                "operation": "pending_cancel",
                "operation_id": pending.operation_id,
            }),
        )
    with pytest.raises(OneStarTransactionError, match="cannot return"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="return_after_crossing",
            transaction=_transaction(),
            location_updates={"hero": "lobby"},
        )


def test_unselected_local_hero_cannot_preposition_beyond_pending_gate() -> None:
    pending = _pending("deployment", participants=["hero"], destination="tower_floor_1")
    other = _hero()
    other.character_id = "other"
    checkpoint = _checkpoint(
        heroes=[_hero(), other],
        pending_operation=pending,
    )

    with pytest.raises(OneStarTransactionError, match="unselected local Hero"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="preposition_unselected",
            transaction=_transaction(),
            location_updates={"other": "tower_floor_1"},
        )

    other.location = "tower_floor_1"
    with pytest.raises(OneStarTransactionError, match="selected participant"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="start_with_prepositioned_unselected",
            transaction=_transaction(
                {
                    "operation": "pending_resolve",
                    "operation_id": pending.operation_id,
                },
                {
                    "operation": "mission_start",
                    "mission": _mission().model_dump(mode="json"),
                    "pending_operation_id": pending.operation_id,
                },
            ),
            location_updates={"hero": "tower_floor_1"},
        )


@pytest.mark.parametrize("operation", ["catalogue", "summon"])
def test_owner_lobby_controls_are_unavailable_during_active_mission(
    operation: str,
) -> None:
    checkpoint = _checkpoint(
        heroes=[_hero(location="tower_floor_1")],
        active_mission=_mission(),
    )
    if operation == "catalogue":
        transaction = _transaction({
            "operation": "catalogue_apply",
            "catalogue_id": "synthesis_chamber",
            "quantity": 1,
        })
        kwargs: dict[str, object] = {}
    else:
        transaction = _transaction({
            "operation": "summon",
            "pool_id": "basic",
            "hero_ids": ["reserve"],
            "birth_stars": [1],
        })
        reserve = _hero(status=CharacterStatus.dormant, owner="")
        reserve.character_id = "reserve"
        checkpoint.characters.append(reserve)
        kwargs = {
            "activated_character_ids": ["reserve"],
            "activated_character_locations": {"reserve": "lobby"},
        }
    with pytest.raises(OneStarTransactionError, match="active mission"):
        prepare_one_star_transaction(
            checkpoint,
            event_id=f"mission_blocks_{operation}",
            transaction=transaction,
            initiating_actor_id="account_owner",
            **kwargs,
        )

    feed = prepare_one_star_transaction(
        checkpoint,
        event_id=f"mission_allows_feed_{operation}",
        transaction=_transaction({
            "operation": "active_feed",
            "hero_id": "hero",
        }),
        initiating_actor_id="account_owner",
    )
    assert load_one_star_account(feed.after_checkpoint)[1].state.active_master_feed_id == "hero"


def test_lobby_control_cannot_follow_mission_start_in_same_transaction() -> None:
    pending = _pending("deployment", participants=["hero"], destination="tower_floor_1")
    checkpoint = _checkpoint(pending_operation=pending)
    with pytest.raises(OneStarTransactionError, match="active mission"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="deploy_then_shop",
            transaction=_transaction(
                {
                    "operation": "pending_resolve",
                    "operation_id": pending.operation_id,
                },
                {
                    "operation": "mission_start",
                    "mission": _mission().model_dump(mode="json"),
                    "pending_operation_id": pending.operation_id,
                },
                {
                    "operation": "catalogue_apply",
                    "catalogue_id": "synthesis_chamber",
                    "quantity": 1,
                },
            ),
            location_updates={"hero": "tower_floor_1"},
            initiating_actor_id="account_owner",
        )


def _add_escape_skill() -> dict:
    return {
        "skill_id": "escape_skill",
        "name": "Escape Skill",
        "rank": 1,
        "capability": "opens a way out of the Tower",
        "tags": ["tower_escape"],
        "visible": True,
    }


def test_escape_authority_added_in_same_mission_end_event_cannot_authorize_escape() -> None:
    checkpoint = _checkpoint(heroes=[_hero(location="tower_floor_1")], active_mission=_mission())
    with pytest.raises(OneStarTransactionError, match="escape authority"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="escape_with_new_skill",
            transaction=_transaction(
                _hero_delta(skills_add=[_add_escape_skill()]),
                {
                    "operation": "mission_end",
                    "mission_id": "mission_1",
                    "outcome": "escaped",
                    "return_destination": "lobby",
                    "escape_authority_id": "escape_skill",
                },
            ),
        )


def test_preexisting_escape_authority_allows_escaped_mission_return() -> None:
    hero = _hero(location="tower_floor_1")
    hero.mechanics[ONE_STAR_HERO_KEY]["skills"] = [_add_escape_skill()]
    checkpoint = _checkpoint(heroes=[hero], active_mission=_mission())
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="escape_with_old_skill",
        transaction=_transaction({
            "operation": "mission_end",
            "mission_id": "mission_1",
            "outcome": "escaped",
            "return_destination": "lobby",
            "escape_authority_id": "escape_skill",
        }),
    )
    assert next(item for item in prepared.after_checkpoint.characters if item.character_id == "hero").location == "lobby"


def test_dispatcher_passes_event_end_time_to_ruleset_adapter_for_delayed_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fiction time stays pinned while the adapter keeps its ledger monotonic."""

    from app.engine import one_star_adapter
    from app.engine.turn_loop_dispatcher import LLMDispatcher

    checkpoint = _checkpoint()
    checkpoint.session.leading_at_s = 100
    result = OneStarEventRouterOutput.model_construct(
        event_id="delayed_cat_ii",
        effective_at_s=10,
        duration_s=0,
        decision_rationale="",
        canonical_event=CanonicalEvent.model_construct(world_adjudication=None, observable_facts=[]),
        event_kind="cat_i",
        requires_responders=False,
        required_responders=[],
        observers=[],
        spawn=[],
        dormant=[],
        cull=[],
        commitment_open=CommitmentOpenSignal.model_construct(
            present=False,
            actor_ids=[],
            description="",
            expected_duration_s=0,
            max_duration_s=0,
            location_label="",
        ),
        commitment_resolutions=[],
        commitment_interrupts=[],
        location_updates=[],
        activate=[],
        state_updates=[],
    )
    captured: dict[str, int] = {}

    monkeypatch.setattr(one_star_adapter, "one_star_event_already_applied", lambda *args, **kwargs: False)
    monkeypatch.setattr(one_star_adapter, "one_star_event_fingerprint", lambda payload: "fingerprint")

    def fake_prepare(*args, **kwargs):
        captured["canonical_at_s"] = kwargs["canonical_at_s"]
        return SimpleNamespace(
            already_applied=False,
            culled_character_ids=(),
            newly_acquired_hero_ids=(),
            engine_history_updates=(),
            system_consequences=(),
        )

    monkeypatch.setattr(one_star_adapter, "prepare_one_star_transaction", fake_prepare)
    monkeypatch.setattr(
        one_star_adapter,
        "apply_one_star_prepared_mutation",
        lambda *_args, **_kwargs: True,
    )
    dispatcher = LLMDispatcher(client=None, prompt_mgr=None)
    asyncio.run(dispatcher.prepare_ruleset_event(ckpt=checkpoint, result=result, actor_id="account_owner"))
    assert captured["canonical_at_s"] == 10

    checkpoint = _checkpoint(stamina_current=4, stamina_anchor=100)
    checkpoint.session.leading_at_s = 100
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="delayed_cat_ii_ledger",
        transaction=_transaction(),
        canonical_at_s=10,
    )
    state = load_one_star_account(prepared.after_checkpoint)[1].state
    assert state.stamina_current == 4
    assert state.stamina_recovery_anchor_s == 100


def test_promotion_rejects_a_target_below_the_current_star_level_cap() -> None:
    target = _hero(location="promotion_room", level=9)
    checkpoint = _checkpoint(
        heroes=[target],
        pending_operation=_pending(
            "promotion",
            target="hero",
            destination="promotion_room",
        ),
        knowledge_tiers=[KnowledgeTier(tier=2)],
    )
    with pytest.raises(OneStarTransactionError, match="level cap"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="premature_promotion",
            transaction=_transaction({
            "operation": "pending_resolve",
            "operation_id": "promotion_1",
        }),
        )


def test_six_to_seven_star_promotion_preserves_last_reviewed_knowledge_tier() -> None:
    target = _hero(location="promotion_room", level=99, xp=485_100)
    mechanics = target.mechanics[ONE_STAR_HERO_KEY]
    mechanics["current_stars"] = 6
    checkpoint = _checkpoint(
        heroes=[target],
        pending_operation=_pending(
            "promotion",
            target="hero",
            destination="promotion_room",
        ),
        knowledge_tiers=[
            KnowledgeTier(tier=6, personal_depth="everything remembered")
        ],
    )
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["star_level_caps"].update({"3": 40, "4": 60, "5": 80, "6": 99, "7": 999999})
    target.knowledge_tier = 6
    target.known_context = "everything remembered"
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="promotion_seven",
        transaction=_transaction({
            "operation": "pending_resolve",
            "operation_id": "promotion_1",
        }),
    )
    promoted = next(
        character
        for character in prepared.after_checkpoint.characters
        if character.character_id == "hero"
    )
    assert load_one_star_hero(promoted).current_stars == 7
    assert promoted.knowledge_tier == 6
    assert promoted.known_context == "everything remembered"


@pytest.mark.parametrize(
    "operation",
    [
        {"operation": "catalogue_apply", "catalogue_id": "synthesis_chamber", "quantity": 1},
        {
            "operation": "pending_open",
            "pending": {
                "operation_id": "deployment_rival",
                "kind": "deployment",
                "participant_ids": ["hero"],
                "target_id": "",
                "destination": "tower_floor_1",
                "opened_at_s": 0,
            },
        },
        {"operation": "active_feed", "hero_id": "hero"},
    ],
)
def test_non_owner_cannot_initiate_master_control_operations(operation: dict) -> None:
    checkpoint = _checkpoint()
    with pytest.raises(OneStarTransactionError, match="account owner"):
        prepare_one_star_transaction(
            checkpoint,
            event_id=f"unauthorized_{operation['operation']}",
            transaction=_transaction(operation),
            initiating_actor_id="hero",
        )


def test_opening_actor_pool_remains_available_to_the_exact_newcomer() -> None:
    newcomer = _hero(status=CharacterStatus.dormant, location="not_yet", owner="")
    newcomer.character_id = "newcomer"
    checkpoint = _checkpoint(heroes=[newcomer])
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["summon_pools"]["opening"] = {
        "cost": {"gold": 0, "gems": 0, "building_resources": 0, "materials": {}},
        "minimum_birth_stars": 1,
        "maximum_birth_stars": 1,
        "star_weights": {1: 10_000},
        "eligible_existing_ids": ["newcomer"],
        "fresh_generation_allowed": False,
        "usage": "opening_actor",
    }
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="newcomer_opening",
        transaction=_transaction({
            "operation": "summon",
            "pool_id": "opening",
            "hero_ids": ["newcomer"],
            "birth_stars": [1],
        }),
        activated_character_ids=["newcomer"],
        activated_character_locations={"newcomer": "lobby"},
        initiating_actor_id="newcomer",
    )
    acquired = next(
        character
        for character in prepared.after_checkpoint.characters
        if character.character_id == "newcomer"
    )
    assert load_one_star_hero(acquired).owner_lobby_id == "lobby_a"
    assert (
        load_one_star_account(prepared.after_checkpoint)[1]
        .state.summon_draw_counters
        == {}
    )


def test_authored_master_opening_wave_does_not_consume_standard_draws() -> None:
    fresh_heroes = []
    for index in range(3):
        hero = _hero(status=CharacterStatus.active, owner="")
        hero.character_id = f"opening_{index}"
        fresh_heroes.append(hero)
    checkpoint = _checkpoint(heroes=fresh_heroes)
    config = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"]
    config["summon_pools"]["opening_wave"] = {
        "cost": {"gold": 1, "gems": 0, "building_resources": 0, "materials": {}},
        "minimum_birth_stars": 1,
        "maximum_birth_stars": 1,
        "star_weights": {1: 10_000},
        "eligible_existing_ids": [],
        "fresh_generation_allowed": True,
        "usage": "opening_wave",
    }
    ids = [hero.character_id for hero in fresh_heroes]
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="master_opening_wave",
        transaction=_transaction({
            "operation": "summon",
            "pool_id": "opening_wave",
            "hero_ids": ids,
            "birth_stars": [1, 1, 1],
        }),
        spawned_character_ids=ids,
        initiating_actor_id="account_owner",
    )
    account = load_one_star_account(prepared.after_checkpoint)[1]
    assert account.state.resources.gold == 17
    assert account.state.summon_draw_counters == {}


def test_deployment_selection_cannot_target_the_lobby_as_its_floor() -> None:
    checkpoint = _checkpoint()
    with pytest.raises(OneStarTransactionError, match="beyond"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="lobby_deployment",
            transaction=_transaction({
                "operation": "pending_open",
                "pending": {
                    "operation_id": "deployment_lobby",
                    "kind": "deployment",
                    "participant_ids": ["hero"],
                    "target_id": "",
                    "destination": "lobby",
                    "opened_at_s": 0,
                },
            }),
            initiating_actor_id="account_owner",
        )


def test_catalogue_purchase_obeys_authored_progression_prerequisites() -> None:
    checkpoint = _checkpoint()
    envelope = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]
    envelope["config"]["catalogue"]["locked_item"] = {
        "kind": "purchase",
        "cost": {"gold": 1, "gems": 0, "building_resources": 0, "materials": {}},
        "inventory_item_id": "locked_item",
        "required_cleared_floor": 2,
    }
    with pytest.raises(OneStarTransactionError, match="cleared Tower floor"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="locked_purchase",
            transaction=_transaction({
                "operation": "catalogue_apply",
                "catalogue_id": "locked_item",
                "quantity": 1,
            }),
            initiating_actor_id="account_owner",
        )


@pytest.mark.parametrize(("minimum", "expected_gold"), [(0, 20), (1, 21)])
def test_repeat_reward_minimum_is_seed_authored(
    minimum: int,
    expected_gold: int,
) -> None:
    checkpoint = _checkpoint(
        heroes=[_hero(location="tower_floor_1")],
        active_mission=_mission(),
    )
    envelope = checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]
    envelope["config"]["repeat_gold_numerator"] = 0
    envelope["config"]["repeat_gold_minimum"] = minimum
    envelope["state"]["highest_cleared_floor"] = 1
    envelope["state"]["active_mission"]["counters"][0]["current"] = 1
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id=f"repeat_minimum_{minimum}",
        transaction=_transaction({
            "operation": "mission_end",
            "mission_id": "mission_1",
            "outcome": "completed",
            "return_destination": "lobby",
            "escape_authority_id": "",
        }),
        canonical_at_s=1,
    )
    account = load_one_star_account(prepared.after_checkpoint)[1]
    assert account.state.resources.gold == expected_gold
