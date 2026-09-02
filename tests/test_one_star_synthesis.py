"""Offline contracts for deterministic One-Star synthesis and mission XP."""

from __future__ import annotations

from copy import deepcopy

import pytest

from app.engine.one_star_adapter import (
    OneStarTransactionError,
    apply_one_star_prepared_mutation,
    load_one_star_account,
    load_one_star_hero,
    one_star_transaction_cull_ids,
    one_star_state_updates_to_transaction,
    prepare_one_star_transaction,
)
from app.engine.one_star_progression import experience_to_reach_level
from app.engine.one_star_progression import apply_experience, rebalance_hero
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.one_star import (
    ONE_STAR_HERO_KEY,
    OneStarHeroState,
    OneStarRulesConfig,
    OneStarStateUpdate,
)
from tests.test_one_star_atomicity import (
    _checkpoint,
    _config,
    _hero,
    _marked_mission_update,
    _mission,
    _mission_end,
    _transaction,
)


def _rebalance_character(
    character: CharacterRecord,
    *,
    raw_config: dict | None = None,
) -> None:
    config = OneStarRulesConfig.model_validate(raw_config or _config())
    hero = OneStarHeroState.model_validate(character.mechanics[ONE_STAR_HERO_KEY])
    rebalance_hero(hero=hero, config=config, restore_full_hp=True)
    character.mechanics[ONE_STAR_HERO_KEY] = hero.model_dump(mode="json")


def _set_total_xp(
    character: CharacterRecord,
    *,
    experience_points: int,
    raw_config: dict | None = None,
) -> None:
    config = OneStarRulesConfig.model_validate(raw_config or _config())
    hero = OneStarHeroState.model_validate(character.mechanics[ONE_STAR_HERO_KEY])
    hero.level = 1
    hero.experience_points = 0
    rebalance_hero(hero=hero, config=config, restore_full_hp=True)
    apply_experience(
        hero=hero,
        experience_delta=experience_points,
        config=config,
    )
    character.mechanics[ONE_STAR_HERO_KEY] = hero.model_dump(mode="json")


def _set_level(character, *, level: int, current_stars: int = 1) -> None:
    hero = character.mechanics["one_star_hero"]
    hero["level"] = level
    hero["current_stars"] = current_stars
    hero["experience_points"] = 50 * level * (level - 1)
    _rebalance_character(character)


def _item(item_id: str, name: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "name": name,
        "slot": "hand",
        "quantity": 1,
        "durability_current": 4,
        "durability_max": 5,
        "tags": ["kept"],
        "visible": True,
    }


def _skill(skill_id: str, name: str, *, visible: bool = True) -> dict[str, object]:
    return {
        "skill_id": skill_id,
        "name": name,
        "rank": 3,
        "capability": f"{skill_id} capability",
        "tags": ["source"],
        "visible": visible,
    }


def _open_synthesis(checkpoint, *, operation_id: str = "synth"):
    return prepare_one_star_transaction(
        checkpoint,
        event_id=f"{operation_id}_open",
        transaction=_transaction({
            "operation": "pending_open",
            "pending": {
                "operation_id": operation_id,
                "kind": "synthesis",
                "participant_ids": ["source_a", "source_b"],
                "target_id": "target",
                "destination": "synthesis_room",
                "opened_at_s": 0,
            },
        }),
        initiating_actor_id="account_owner",
    )


def _synthesis_checkpoint(*, chance_basis_points: int = 500):
    source_a = _hero(location="synthesis_room")
    source_a.character_id = "source_a"
    source_a.mechanics["one_star_hero"].update({
        "birth_stars": 3,
        "current_stars": 3,
        "potential_grade": 3,
        "level": 2,
        "experience_points": 101,
        "equipment": [_item("source_a_blade", "Bronze Blade")],
        "skills": [_skill("a_visible", "Guard Break")],
    })
    source_b = _hero(location="synthesis_room")
    source_b.character_id = "source_b"
    source_b.mechanics["one_star_hero"].update({
        "equipment": [_item("source_b_charm", "Ash Charm")],
        "skills": [_skill("b_hidden", "Veiled Step", visible=False)],
    })
    target = _hero(location="synthesis_room")
    target.character_id = "target"
    checkpoint = _checkpoint(heroes=[source_a, source_b, target])
    raw_config = checkpoint.characters[0].mechanics["one_star_account"]["config"]
    raw_config["star_level_caps"].update({"3": 40})
    raw_config["progression"]["synthesis_skill_chance_basis_points"] = chance_basis_points
    source_a = next(
        item for item in checkpoint.characters if item.character_id == "source_a"
    )
    _rebalance_character(source_a, raw_config=raw_config)
    return checkpoint


def test_synthesis_preview_uses_exact_source_formula_and_rejects_no_capacity() -> None:
    checkpoint = _synthesis_checkpoint()
    opened = _open_synthesis(checkpoint)
    account = load_one_star_account(opened.after_checkpoint)[1]
    pending = account.state.pending_operation
    assert pending is not None and pending.synthesis_preview is not None
    preview = pending.synthesis_preview

    # source_a: floor(101 / 2) + round-half-up(100 * 1.25**2) = 206;
    # source_b: 0 + 100.
    assert (preview.offered_xp, preview.applied_xp, preview.wasted_xp) == (306, 306, 0)
    assert [item.item_id for item in preview.returned_equipment] == [
        "source_a_blade",
        "source_b_charm",
    ]
    assert preview.skill_transfer_chance_basis_points == 500
    assert any("306 XP offered" in item.text for item in opened.system_consequences)

    capped = _synthesis_checkpoint()
    target = next(item for item in capped.characters if item.character_id == "target")
    target_sheet = target.mechanics["one_star_hero"]
    target_sheet["level"] = 10
    target_sheet["experience_points"] = 5_500
    _rebalance_character(target)
    with pytest.raises(OneStarTransactionError, match="cannot accept any offered experience"):
        _open_synthesis(capped, operation_id="capped")


def test_synthesis_moves_exact_gear_culls_sources_and_transfers_multiple_skills() -> None:
    checkpoint = _synthesis_checkpoint(chance_basis_points=10_000)
    opened = _open_synthesis(checkpoint)
    resolved = prepare_one_star_transaction(
        opened.after_checkpoint,
        event_id="synth_resolve",
        transaction=_transaction({
            "operation": "pending_resolve",
            "operation_id": "synth",
        }),
    )
    account = load_one_star_account(resolved.after_checkpoint)[1]
    characters = {item.character_id: item for item in resolved.after_checkpoint.characters}
    target = load_one_star_hero(characters["target"])
    assert target is not None
    assert [item.model_dump(mode="json") for item in account.state.stored_equipment] == [
        _item("source_a_blade", "Bronze Blade"),
        _item("source_b_charm", "Ash Charm"),
    ]
    assert [skill.skill_id for skill in target.skills] == ["a_visible", "b_hidden"]
    assert all(skill.rank == 1 for skill in target.skills)
    assert all(characters[source_id].status is CharacterStatus.culled for source_id in ("source_a", "source_b"))
    assert all(
        load_one_star_hero(characters[source_id]).equipment == []
        for source_id in ("source_a", "source_b")
    )
    assert all(
        load_one_star_hero(characters[source_id]).terminal_event_id == "synth_resolve"
        for source_id in ("source_a", "source_b")
    )
    assert set(one_star_transaction_cull_ids(
        resolved.after_checkpoint,
        event_id="synth_resolve",
    )) == {"source_a", "source_b"}
    consequence_text = "\n".join(item.text for item in resolved.system_consequences)
    assert "Guard Break" in consequence_text
    assert "Veiled Step" not in consequence_text
    authority_updates = "\n".join(resolved.engine_history_updates)
    assert '"target_character_id":"target"' in authority_updates
    assert '"source_character_id":"source_b"' in authority_updates
    assert '"name":"Veiled Step"' in authority_updates
    assert '"capability":"b_hidden capability"' in authority_updates

    assert apply_one_star_prepared_mutation(opened.after_checkpoint, resolved) is True
    replay = prepare_one_star_transaction(
        opened.after_checkpoint,
        event_id="synth_resolve",
        transaction=_transaction({
            "operation": "pending_resolve",
            "operation_id": "synth",
        }),
    )
    assert replay.already_applied is True


def test_authoritative_synthesis_atomically_opens_moves_and_resolves_once() -> None:
    checkpoint = _synthesis_checkpoint(chance_basis_points=0)
    for character in checkpoint.characters:
        if character.character_id in {"source_a", "source_b", "target"}:
            character.location = "lobby"
    transaction = _transaction(
        {
            "operation": "pending_open",
            "pending": {
                "operation_id": "system_synth",
                "kind": "synthesis",
                "participant_ids": ["source_a", "source_b"],
                "target_id": "target",
                "destination": "synthesis_room",
                "opened_at_s": 0,
            },
        },
        {
            "operation": "pending_resolve",
            "operation_id": "system_synth",
        },
    )
    locations = {
        "source_a": "synthesis_room",
        "source_b": "synthesis_room",
        "target": "synthesis_room",
    }

    with pytest.raises(OneStarTransactionError, match="only One-Star operation"):
        prepare_one_star_transaction(
            checkpoint,
            event_id="system_synthesis",
            transaction=transaction,
            location_updates=locations,
            initiating_actor_id="account_owner",
        )

    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="system_synthesis",
        transaction=transaction,
        location_updates=locations,
        initiating_actor_id="account_owner",
        authoritative_system_result=True,
    )
    account = load_one_star_account(prepared.after_checkpoint)[1]
    by_id = {
        character.character_id: character
        for character in prepared.after_checkpoint.characters
    }
    assert account.state.pending_operation is None
    assert account.state.synthesis_resolution_count == 1
    assert by_id["source_a"].status is CharacterStatus.culled
    assert by_id["source_b"].status is CharacterStatus.culled
    assert by_id["target"].status is CharacterStatus.active

    assert apply_one_star_prepared_mutation(checkpoint, prepared) is True
    replay = prepare_one_star_transaction(
        checkpoint,
        event_id="system_synthesis",
        transaction=transaction,
        location_updates=locations,
        initiating_actor_id="account_owner",
        authoritative_system_result=True,
    )
    assert replay.already_applied is True
    assert load_one_star_account(checkpoint)[1].state.synthesis_resolution_count == 1


@pytest.mark.parametrize(
    "drift",
    ["equipment", "source_skill", "target_skill", "target_xp"],
)
def test_synthesis_stale_preview_rolls_back_all_sources(drift: str) -> None:
    checkpoint = _synthesis_checkpoint(chance_basis_points=0)
    opened = _open_synthesis(checkpoint)
    stale = opened.after_checkpoint
    source = next(item for item in stale.characters if item.character_id == "source_a")
    if drift == "equipment":
        source.mechanics["one_star_hero"]["equipment"][0]["durability_current"] = 1
    elif drift == "source_skill":
        source.mechanics["one_star_hero"]["skills"][0]["rank"] = 4
    elif drift == "target_skill":
        target = next(item for item in stale.characters if item.character_id == "target")
        target.mechanics["one_star_hero"]["skills"] = [_skill("target_skill", "Target Skill")]
    else:
        target = next(item for item in stale.characters if item.character_id == "target")
        target.mechanics["one_star_hero"]["experience_points"] = 1
    before = deepcopy(stale.model_dump(mode="json"))

    with pytest.raises(OneStarTransactionError, match="changed since selection"):
        prepare_one_star_transaction(
            stale,
            event_id="stale_resolve",
            transaction=_transaction({
                "operation": "pending_resolve",
                "operation_id": "synth",
            }),
        )
    assert stale.model_dump(mode="json") == before


def test_synthesis_allows_embodied_injury_while_selected_heroes_respond() -> None:
    opened = _open_synthesis(_synthesis_checkpoint(chance_basis_points=0))
    checkpoint = opened.after_checkpoint
    source = next(
        item for item in checkpoint.characters if item.character_id == "source_a"
    )
    source.mechanics["one_star_hero"]["hp_current"] = 2
    source.mechanics["one_star_hero"]["conditions"] = ["bruised while resisting"]
    target = next(
        item for item in checkpoint.characters if item.character_id == "target"
    )
    target.mechanics["one_star_hero"]["hp_current"] = 4

    resolved = prepare_one_star_transaction(
        checkpoint,
        event_id="injured_resolve",
        transaction=_transaction({
            "operation": "pending_resolve",
            "operation_id": "synth",
        }),
    )

    characters = {
        item.character_id: item for item in resolved.after_checkpoint.characters
    }
    assert characters["source_a"].status is CharacterStatus.culled
    resolved_target = load_one_star_hero(characters["target"])
    assert resolved_target is not None
    assert resolved_target.hp_current >= 4


def test_synthesis_cap_banks_once_then_promotion_releases_reachable_xp() -> None:
    checkpoint = _synthesis_checkpoint(chance_basis_points=0)
    target = next(item for item in checkpoint.characters if item.character_id == "target")
    target_sheet = target.mechanics["one_star_hero"]
    target_sheet["level"] = 10
    target_sheet["experience_points"] = 4_500
    _rebalance_character(target)
    for source_id in ("source_a", "source_b"):
        source = next(item for item in checkpoint.characters if item.character_id == source_id)
        raw_config = checkpoint.characters[0].mechanics["one_star_account"]["config"]
        _set_total_xp(source, experience_points=1_000, raw_config=raw_config)
    opened = _open_synthesis(checkpoint)
    preview = load_one_star_account(opened.after_checkpoint)[1].state.pending_operation.synthesis_preview
    assert (preview.offered_xp, preview.applied_xp, preview.wasted_xp) == (1_256, 1_000, 256)
    resolved = prepare_one_star_transaction(
        opened.after_checkpoint,
        event_id="banked_synthesis",
        transaction=_transaction({"operation": "pending_resolve", "operation_id": "synth"}),
    )
    promoted_checkpoint = resolved.after_checkpoint
    promoted_target = next(item for item in promoted_checkpoint.characters if item.character_id == "target")
    promoted_target.location = "promotion_room"
    assert load_one_star_hero(promoted_target).experience_points == 5_500
    opened_promotion = prepare_one_star_transaction(
        promoted_checkpoint,
        event_id="promotion_open",
        transaction=_transaction({
            "operation": "pending_open",
            "pending": {
                "operation_id": "promotion_1",
                "kind": "promotion",
                "participant_ids": ["target"],
                "target_id": "target",
                "destination": "promotion_room",
                "opened_at_s": 0,
            },
        }),
        initiating_actor_id="account_owner",
    )
    promoted = prepare_one_star_transaction(
        opened_promotion.after_checkpoint,
        event_id="promotion_resolve",
        transaction=_transaction({"operation": "pending_resolve", "operation_id": "promotion_1"}),
    )
    hero = load_one_star_hero(next(
        item for item in promoted.after_checkpoint.characters if item.character_id == "target"
    ))
    assert (hero.current_stars, hero.level, hero.experience_points) == (2, 11, 5_500)


def test_synthesis_zero_transfer_chance_never_copies_source_skills() -> None:
    checkpoint = _synthesis_checkpoint(chance_basis_points=0)
    opened = _open_synthesis(checkpoint)
    resolved = prepare_one_star_transaction(
        opened.after_checkpoint,
        event_id="no_transfer",
        transaction=_transaction({"operation": "pending_resolve", "operation_id": "synth"}),
    )
    target = load_one_star_hero(next(
        item for item in resolved.after_checkpoint.characters if item.character_id == "target"
    ))
    assert target.skills == []


def test_synthesis_five_percent_skill_rolls_are_independent_and_replay_stable() -> None:
    def resolve_once() -> list[str]:
        opened = _open_synthesis(_synthesis_checkpoint(), operation_id="prob_6")
        resolved = prepare_one_star_transaction(
            opened.after_checkpoint,
            event_id="prob_6_resolve",
            transaction=_transaction({
                "operation": "pending_resolve",
                "operation_id": "prob_6",
            }),
        )
        target = load_one_star_hero(next(
            item for item in resolved.after_checkpoint.characters if item.character_id == "target"
        ))
        return [skill.skill_id for skill in target.skills]

    # The fixed session/op/source digest makes source_a miss and source_b hit.
    assert resolve_once() == ["b_hidden"]
    assert resolve_once() == ["b_hidden"]


def test_completed_mission_awards_each_living_survivor_with_overlevel_multiplier() -> None:
    heroes = []
    for level in range(1, 8):
        hero = _hero(location="tower_floor_1")
        hero.character_id = f"hero_{level}"
        _set_level(hero, level=level)
        heroes.append(hero)
    dead = _hero(status=CharacterStatus.culled, location="tower_floor_1")
    dead.character_id = "dead"
    _set_level(dead, level=1)
    mission = _mission(party=[*(hero.character_id for hero in heroes), "dead"])
    mission.counters[0].current = mission.counters[0].target
    checkpoint = _checkpoint(heroes=[*heroes, dead], active_mission=mission)
    checkpoint.characters.append(
        CharacterRecord(
            character_id="guide",
            name="Guide",
        )
    )
    account_state = checkpoint.characters[0].mechanics["one_star_account"]["state"]
    account_state["guide_character_ids"] = ["guide"]
    account_state["system_observer_ids"] = ["guide"]

    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="floor_one_complete",
        transaction=_transaction(
            _marked_mission_update(
                current=1,
                credited_id="hero_1",
            ),
            _mission_end(
                event_id="floor_one_complete",
                outcome="completed",
                mvp_character_id="hero_1",
            ),
        ),
    )
    by_id = {item.character_id: item for item in prepared.after_checkpoint.characters}
    multipliers = [100, 75, 50, 25, 10, 5, 0]
    config = load_one_star_account(prepared.after_checkpoint)[1].config
    for level, multiplier in enumerate(multipliers, start=1):
        hero = load_one_star_hero(by_id[f"hero_{level}"])
        assert hero.experience_points == experience_to_reach_level(level, config) + multiplier
    assert load_one_star_hero(by_id["dead"]).experience_points == 0
    assert any(
        "guide" in consequence.recipient_character_ids
        and "Mission report MVP" in consequence.text
        for consequence in prepared.system_consequences
    )
    assert all(
        "guide" not in consequence.recipient_character_ids
        for consequence in prepared.system_consequences
        if " mission reward:" in consequence.text
    )


def test_first_and_repeat_completion_use_the_same_floor_xp_authority() -> None:
    def complete(*, already_cleared: bool) -> int:
        hero = _hero(location="tower_floor_1")
        mission = _mission()
        mission.counters[0].current = mission.counters[0].target
        checkpoint = _checkpoint(heroes=[hero], active_mission=mission)
        if already_cleared:
            checkpoint.characters[0].mechanics["one_star_account"]["state"]["highest_cleared_floor"] = 1
        prepared = prepare_one_star_transaction(
            checkpoint,
            event_id=f"complete_{already_cleared}",
            transaction=_transaction(
                _marked_mission_update(current=1),
                _mission_end(
                    event_id=f"complete_{already_cleared}",
                    outcome="completed",
                ),
            ),
        )
        return load_one_star_hero(next(
            item for item in prepared.after_checkpoint.characters if item.character_id == "hero"
        )).experience_points

    assert complete(already_cleared=False) == complete(already_cleared=True) == 100


@pytest.mark.parametrize("obsolete", ["hp_max=9", "level=2", "experience_delta=100", "stat.power=1"])
def test_router_cannot_author_progression_numbers(obsolete: str) -> None:
    checkpoint = _checkpoint()
    with pytest.raises(OneStarTransactionError, match="unsupported details"):
        one_star_state_updates_to_transaction(
            checkpoint,
            [OneStarStateUpdate(
                kind="hero_delta",
                target_id="hero",
                value="",
                details=[obsolete],
            )],
            canonical_at_s=0,
        )
