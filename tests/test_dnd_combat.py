import pytest

from app.engine import dice
from app.engine.dnd_combat import (
    add_combatant,
    advance_turn,
    apply_damage,
    apply_healing,
    current_combatant,
    end_effect,
    end_combat,
    private_status,
    public_status,
    remove_combatant,
    roll_death_save,
    start_effect,
    start_combat,
    update_effect,
)
from app.schemas.characters import CharacterRecord, CharacterStatus, PublicSheet
from app.schemas.state import (
    CatIIRollTransaction,
    DndCombatantState,
    DndCombatState,
    DndEffectRecurringSave,
    DndRuntimeEffect,
    SessionState,
    SlotEntry,
)


def _character(
    character_id: str,
    name: str,
    *,
    dex: int = 10,
    hp: int = 10,
    temp: int = 0,
    ac: int = 10,
    status: CharacterStatus = CharacterStatus.active,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        status=status,
        public_sheet=PublicSheet(role="combatant"),
        mechanics={
            "ruleset_id": "dnd5e_basic",
            "ability_scores": {
                "str": 10,
                "dex": dex,
                "con": 10,
                "int": 10,
                "wis": 10,
                "cha": 10,
            },
            "armor_class": ac,
            "hit_points": {
                "current": hp,
                "max": hp,
                "temporary": temp,
            },
            "conditions": [],
        },
    )


def test_recurring_save_end_synonyms_normalize_to_engine_values():
    assert DndEffectRecurringSave(ends_on="succeed").ends_on == "success"
    assert DndEffectRecurringSave(ends_on="fails").ends_on == "failure"
    assert DndEffectRecurringSave(ends_on="unexpected").ends_on == "success"


def test_start_combat_builds_snapshots_rolls_initiative_and_persists(monkeypatch):
    values = iter([9, 9, 19])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    session = SessionState(
        session_id="s",
        turn_index=7,
        character_bindings={"alice": "discord_1"},
    )
    alice = _character("alice", "Alice", dex=14, hp=12, ac=16)
    bob = _character("bob", "Bob", dex=14, hp=9, ac=13)
    pip = _character("pip", "Pip", dex=8, hp=7, ac=15)

    combat = start_combat(session, [alice, bob, pip], combat_id="ambush")

    assert session.active_combat is combat
    assert combat.status == "active"
    assert combat.started_at_turn_index == 7
    assert combat.round_number == 1
    assert [c.combatant_id for c in combat.combatants] == ["pip", "alice", "bob"]
    assert [c.initiative_total for c in combat.combatants] == [19, 12, 12]
    assert [c.initiative_roll for c in combat.combatants] == [20, 10, 10]
    assert [c.initiative_order for c in combat.combatants] == [1, 2, 3]
    assert current_combatant(session).combatant_id == "pip"
    assert combat.combatants[1].player_controlled is True
    assert SessionState(**session.model_dump()).active_combat.combat_id == "ambush"


def test_start_combat_assigns_default_stats_to_story_character(monkeypatch):
    values = iter([9])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    session = SessionState(session_id="s")
    meris = CharacterRecord(
        character_id="npc_meris",
        name="Meris Venn",
        public_sheet=PublicSheet(role="custodian"),
    )

    combat = start_combat(session, [meris], combat_id="ambush")

    assert meris.mechanics["ruleset_id"] == "dnd5e_basic"
    assert meris.mechanics["source"] == "dnd_default_combatant_profile"
    assert meris.mechanics["hit_points"] == {
        "current": 4,
        "max": 4,
        "temporary": 0,
    }
    assert meris.mechanics["dnd5e_sheet"]["statblock"]["xp"] == 0
    combatant = combat.combatants[0]
    assert combatant.character_id == "npc_meris"
    assert combatant.hit_points_current == 4
    assert combatant.hit_points_max == 4


def test_start_combat_rejects_invalid_authoritative_spawn_stats():
    session = SessionState(session_id="s")
    broken = CharacterRecord(
        character_id="broken_spawn",
        name="Broken Spawn",
        public_sheet=PublicSheet(role="spawned monster"),
        mechanics={
            "ruleset_id": "dnd5e_basic",
            "source": "router_combatant_spawn",
            "hit_points": {"current": 0, "max": 0, "temporary": 0},
        },
    )

    with pytest.raises(ValueError, match="invalid hit points"):
        start_combat(session, [broken], combat_id="ambush")


def test_turn_advancement_skips_defeated_and_removed_and_wraps_round(monkeypatch):
    values = iter([14, 13, 12])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    session = SessionState(session_id="s")
    combat = start_combat(
        session,
        [
            _character("alice", "Alice", dex=10),
            _character("bob", "Bob", dex=10),
            _character("pip", "Pip", dex=10),
        ],
    )
    assert [c.combatant_id for c in combat.combatants] == ["alice", "bob", "pip"]

    combat.combatants[0].reaction_available = False
    apply_damage(session, "bob", 99)
    remove_combatant(session, "pip")

    advanced = advance_turn(session)
    assert advanced.combatant_id == "alice"
    assert advanced.reaction_available is True
    assert combat.round_number == 2


def test_damage_and_healing_use_hp_snapshot_without_touching_character_mechanics(
    monkeypatch,
):
    values = iter([4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    hero = _character("hero", "Hero", hp=10, temp=3)
    session = SessionState(session_id="s")
    start_combat(session, [hero])

    damaged = apply_damage(session, "hero", 8)

    assert damaged.hit_points_temporary == 0
    assert damaged.hit_points_current == 5
    assert hero.mechanics["hit_points"] == {
        "current": 10,
        "max": 10,
        "temporary": 3,
    }

    apply_damage(session, "hero", 99)
    assert damaged.hit_points_current == 0
    assert damaged.defeat_state == "defeated"

    healed = apply_healing(session, "hero", 4)
    assert healed.hit_points_current == 4
    assert healed.defeat_state == "active"


def test_damage_and_healing_sync_to_character_mechanics_when_records_available(
    monkeypatch,
):
    values = iter([4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    hero = _character("hero", "Hero", hp=10, temp=3)
    hero.mechanics["dnd5e_sheet"] = {
        "statblock": {
            "defenses": {
                "hit_points": {"current": 10, "max": 10, "temporary": 3}
            }
        }
    }
    session = SessionState(session_id="s", character_bindings={"hero": "u1"})
    start_combat(session, [hero])

    apply_damage(session, "hero", 8, characters=[hero])

    assert hero.mechanics["hit_points"] == {
        "current": 5,
        "max": 10,
        "temporary": 0,
    }
    assert (
        hero.mechanics["dnd5e_sheet"]["statblock"]["defenses"]["hit_points"]
        == {"current": 5, "max": 10, "temporary": 0}
    )

    apply_damage(session, "hero", 6, characters=[hero])

    assert hero.mechanics["hit_points"]["current"] == 0
    assert "unconscious" in hero.mechanics["conditions"]

    apply_healing(session, "hero", 4, characters=[hero])

    assert hero.mechanics["hit_points"]["current"] == 4
    assert "unconscious" not in hero.mechanics["conditions"]
    assert (
        hero.mechanics["dnd5e_sheet"]["statblock"]["defenses"]["hit_points"]
        ["current"]
        == 4
    )


def test_legacy_defeated_field_migrates_to_defeat_state():
    combatant = DndCombatantState.model_validate({
        "combatant_id": "goblin",
        "character_id": "goblin",
        "hit_points_current": 0,
        "hit_points_max": 7,
        "defeated": True,
    })

    assert combatant.defeat_state == "defeated"


def test_player_controlled_combatant_goes_down_and_healing_recovers(
    monkeypatch,
):
    values = iter([4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    hero = _character("hero", "Hero", hp=10)
    session = SessionState(
        session_id="s",
        character_bindings={"hero": "discord_1"},
    )
    start_combat(session, [hero])

    damaged = apply_damage(session, "hero", 10)

    assert damaged.hit_points_current == 0
    assert damaged.defeat_state == "down"
    assert damaged.death_save_successes == 0
    assert damaged.death_save_failures == 0
    assert "unconscious" in damaged.conditions

    apply_damage(session, "hero", 1)
    assert damaged.defeat_state == "down"
    assert damaged.death_save_failures == 1

    healed = apply_healing(session, "hero", 4)
    assert healed.hit_points_current == 4
    assert healed.defeat_state == "active"
    assert healed.death_save_successes == 0
    assert healed.death_save_failures == 0
    assert "unconscious" not in healed.conditions


def test_death_saves_stabilize_or_kill_player_controlled_combatant(
    monkeypatch,
):
    values = iter([4, 9, 9, 9, 0])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    hero = _character("hero", "Hero", hp=10)
    session = SessionState(
        session_id="s",
        character_bindings={"hero": "discord_1"},
    )
    start_combat(session, [hero])
    apply_damage(session, "hero", 10)

    roll_death_save(session, "hero")
    roll_death_save(session, "hero")
    roll_death_save(session, "hero")

    combatant = session.active_combat.combatants[0]
    assert combatant.defeat_state == "stable"
    assert combatant.death_save_successes == 0
    assert combatant.death_save_failures == 0
    assert "third success; they are stable" in session.active_combat.audit_lines[-1]
    assert "Hero stabilizes." in session.active_combat.pending_visible_facts

    apply_damage(session, "hero", 1)
    assert combatant.defeat_state == "down"
    assert combatant.death_save_successes == 0
    assert combatant.death_save_failures == 1

    roll_death_save(session, "hero")
    assert combatant.defeat_state == "dead"
    assert combatant.death_save_failures == 3
    assert "Hero dies." in session.active_combat.pending_visible_facts


def test_advance_turn_rolls_death_save_for_down_combatant(monkeypatch):
    values = iter([14, 9, 19])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    hero = _character("hero", "Hero", hp=10)
    goblin = _character("goblin", "Goblin", hp=7)
    session = SessionState(
        session_id="s",
        character_bindings={"hero": "discord_1"},
    )
    combat = start_combat(session, [hero, goblin])
    assert [c.combatant_id for c in combat.combatants] == ["hero", "goblin"]
    combat.turn_index = 1
    apply_damage(session, "hero", 10)

    advanced = advance_turn(session)

    assert advanced.combatant_id == "hero"
    assert advanced.defeat_state == "active"
    assert advanced.hit_points_current == 1
    assert "natural 20; they regain 1 HP" in combat.audit_lines[-1]
    assert combat.pending_visible_facts == ["Hero regains consciousness."]


def test_advance_turn_persists_death_save_when_everyone_is_down(monkeypatch):
    values = iter([4, 4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    hero = _character("hero", "Hero", hp=10)
    session = SessionState(
        session_id="s",
        character_bindings={"hero": "discord_1"},
    )
    combat = start_combat(session, [hero])
    apply_damage(session, "hero", 10)

    advanced = advance_turn(session)

    assert advanced.combatant_id == "hero"
    assert advanced.defeat_state == "down"
    assert advanced.death_save_failures == 1
    assert "failure (1/3)" in combat.audit_lines[-1]


def test_add_and_remove_combatants_preserve_current_turn(monkeypatch):
    values = iter([9, 4, 19, 7])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    session = SessionState(session_id="s")
    combat = start_combat(
        session,
        [
            _character("alice", "Alice", dex=10),
            _character("bob", "Bob", dex=10),
        ],
    )
    assert current_combatant(combat).combatant_id == "alice"

    added = add_combatant(session, _character("pip", "Pip", dex=10))

    assert added.initiative_total == 20
    assert [c.combatant_id for c in combat.combatants] == ["pip", "alice", "bob"]
    assert current_combatant(combat).combatant_id == "alice"

    removed = remove_combatant(session, "alice")
    assert removed.removed is True
    assert current_combatant(combat).combatant_id == "bob"

    duplicate = add_combatant(
        session,
        _character("summon", "Summon", dex=10),
        combatant_id="summon_1",
    )
    assert duplicate.character_id == "summon"
    hard_removed = remove_combatant(session, "summon", hard=True)
    assert hard_removed.combatant_id == "summon_1"
    assert all(c.combatant_id != "summon_1" for c in combat.combatants)


def test_public_and_private_statuses_are_concise_and_private_has_roll_details(
    monkeypatch,
):
    values = iter([9])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    session = SessionState(session_id="s")
    start_combat(session, [_character("alice", "Alice", dex=14, hp=12, ac=16)])

    public = public_status(session)
    private = private_status(session)

    assert public == {
        "combat_id": "combat",
        "round": 1,
        "current": {
            "combatant_id": "alice",
            "character_id": "alice",
            "name": "Alice",
            "hp": {"current": 12, "max": 12, "temporary": 0},
            "defeat_state": "active",
            "death_saves": {"successes": 0, "failures": 0},
            "active_effects": [],
            "removed": False,
            "pending_initiating_action": "",
            "pending_initiating_event_id": "",
        },
        "turn_order": [
            {
                "combatant_id": "alice",
                "character_id": "alice",
                "name": "Alice",
                "hp": {"current": 12, "max": 12, "temporary": 0},
                "defeat_state": "active",
                "death_saves": {"successes": 0, "failures": 0},
                "active_effects": [],
                "removed": False,
                "pending_initiating_action": "",
                "pending_initiating_event_id": "",
            }
        ],
    }
    assert private["current"]["armor_class"] == 16
    assert private["current"]["initiative"]["roll"] == 10
    assert private["current"]["initiative"]["modifier"] == 2
    assert private["current"]["initiative"]["total"] == 12


def test_runtime_effects_seed_combat_and_sync_back(monkeypatch):
    values = iter([9])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    alice = _character("alice", "Alice")
    alice.mechanics["dnd5e_runtime"] = {
        "active_effects": [
            {
                "effect_id": "eff_bless",
                "name": "Bless",
                "slug": "bless",
                "target_id": "alice",
                "originator_id": "cleric",
                "conditions": ["blessed"],
                "concentration": True,
                "duration_kind": "minutes",
                "duration_amount": 1,
                "remaining_rounds": 10,
            }
        ]
    }
    session = SessionState(session_id="s")

    combat = start_combat(session, [alice])

    combatant = combat.combatants[0]
    assert combatant.active_effects[0].name == "Bless"
    assert "blessed" in combatant.conditions

    combatant.active_effects[0].remaining_rounds = 8
    end_combat(session, characters=[alice])

    stored = alice.mechanics["dnd5e_runtime"]["active_effects"]
    assert stored[0]["effect_id"] == "eff_bless"
    assert stored[0]["remaining_rounds"] == 8


def test_start_effect_skips_invalid_target_and_humanizes_reason(monkeypatch):
    values = iter([9, 9])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    alice = _character("alice", "Alice")
    bob = _character("bob", "Bob")
    session = SessionState(session_id="s")
    combat = start_combat(session, [alice, bob])

    skipped = start_effect(session, DndRuntimeEffect(
        effect_id="eff_missing",
        name="Bless",
        slug="bless",
        target_id="missing",
        originator_id="alice",
        conditions=["blessed"],
        metadata={"reason": "failed initial save"},
    ))
    assert skipped.effect_id == "eff_missing"
    assert all(c.active_effects == [] for c in combat.combatants)
    assert combat.pending_visible_facts == []
    assert "Effect start skipped" in combat.audit_lines[-1]

    bob_effect = start_effect(session, DndRuntimeEffect(
        effect_id="eff_hold",
        name="Hold Person",
        slug="hold_person",
        target_id="bob",
        originator_id="alice",
        conditions=["paralyzed"],
        metadata={"reason": "failed initial save"},
    ))

    bob_state = next(c for c in combat.combatants if c.character_id == "bob")
    assert bob_effect in bob_state.active_effects
    assert "paralyzed" in bob_state.conditions
    assert (
        "Hold Person takes hold on Bob after the initial save fails."
        in combat.pending_visible_facts
    )


def test_start_effect_replacement_reconciles_old_conditions(monkeypatch):
    values = iter([9, 9])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    alice = _character("alice", "Alice")
    bob = _character("bob", "Bob")
    session = SessionState(session_id="s")
    combat = start_combat(session, [alice, bob])

    start_effect(session, DndRuntimeEffect(
        effect_id="eff_hold",
        name="Hold Person",
        slug="hold_person",
        target_id="bob",
        originator_id="alice",
        conditions=["paralyzed"],
    ))
    start_effect(session, DndRuntimeEffect(
        effect_id="eff_hold",
        name="Hold Person",
        slug="hold_person",
        target_id="bob",
        originator_id="alice",
        conditions=["restrained"],
    ))

    bob_state = next(c for c in combat.combatants if c.character_id == "bob")
    assert [effect.effect_id for effect in bob_state.active_effects] == ["eff_hold"]
    assert "restrained" in bob_state.conditions
    assert "paralyzed" not in bob_state.conditions

    alice_state = next(c for c in combat.combatants if c.character_id == "alice")
    start_effect(session, DndRuntimeEffect(
        effect_id="eff_shift",
        name="Shifting Curse",
        slug="shifting_curse",
        target_id="alice",
        originator_id="bob",
        conditions=["frightened"],
    ))
    assert "frightened" in alice_state.conditions

    start_effect(session, DndRuntimeEffect(
        effect_id="eff_shift",
        name="Shifting Curse",
        slug="shifting_curse",
        target_id="bob",
        originator_id="bob",
        conditions=["grappled"],
    ))

    assert all(effect.effect_id != "eff_shift" for effect in alice_state.active_effects)
    assert "frightened" not in alice_state.conditions
    assert any(effect.effect_id == "eff_shift" for effect in bob_state.active_effects)
    assert "grappled" in bob_state.conditions


def test_end_effect_uses_owning_combatant_and_preserves_overlapping_conditions(
    monkeypatch,
):
    values = iter([9, 9])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    alice = _character("alice", "Alice")
    bob = _character("bob", "Bob")
    session = SessionState(session_id="s")
    combat = start_combat(session, [alice, bob])
    bob_state = next(c for c in combat.combatants if c.character_id == "bob")
    bob_state.active_effects.extend([
        DndRuntimeEffect(
            effect_id="eff_bad_target",
            name="Hold Person",
            slug="hold_person",
            target_id="missing",
            originator_id="alice",
            conditions=["paralyzed"],
        ),
        DndRuntimeEffect(
            effect_id="eff_other_hold",
            name="Hold Person",
            slug="hold_person",
            target_id="bob",
            originator_id="cleric",
            conditions=["paralyzed"],
        ),
    ])
    bob_state.conditions.append("paralyzed")

    ended = end_effect(
        session,
        effect_id="eff_bad_target",
        reason="duration expired",
    )

    assert [effect.effect_id for effect in ended] == ["eff_bad_target"]
    assert [effect.effect_id for effect in bob_state.active_effects] == [
        "eff_other_hold"
    ]
    assert "paralyzed" in bob_state.conditions
    assert (
        "Hold Person ends on Bob as its duration runs out."
        in combat.pending_visible_facts
    )

    end_effect(session, effect_id="eff_other_hold")
    assert bob_state.active_effects == []
    assert "paralyzed" not in bob_state.conditions


def test_slug_only_end_skips_multiple_originators(monkeypatch):
    values = iter([9, 9])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    alice = _character("alice", "Alice")
    bob = _character("bob", "Bob")
    session = SessionState(session_id="s")
    combat = start_combat(session, [alice, bob])
    alice_state = next(c for c in combat.combatants if c.character_id == "alice")
    bob_state = next(c for c in combat.combatants if c.character_id == "bob")
    alice_state.active_effects.append(DndRuntimeEffect(
        effect_id="eff_bless_a",
        name="Bless",
        slug="bless",
        target_id="alice",
        originator_id="cleric_a",
        conditions=["blessed"],
    ))
    bob_state.active_effects.append(DndRuntimeEffect(
        effect_id="eff_bless_b",
        name="Bless",
        slug="bless",
        target_id="bob",
        originator_id="cleric_b",
        conditions=["blessed"],
    ))
    alice_state.conditions.append("blessed")
    bob_state.conditions.append("blessed")

    ended = end_effect(session, slug="bless")

    assert ended == []
    assert [effect.effect_id for effect in alice_state.active_effects] == [
        "eff_bless_a"
    ]
    assert [effect.effect_id for effect in bob_state.active_effects] == [
        "eff_bless_b"
    ]
    assert combat.pending_visible_facts == []
    assert "slug-only selector is ambiguous" in combat.audit_lines[-1]


def test_update_effect_reconciles_conditions_and_ignores_bad_target_when_exact(
    monkeypatch,
):
    values = iter([9, 9])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    alice = _character("alice", "Alice")
    bob = _character("bob", "Bob")
    session = SessionState(session_id="s")
    combat = start_combat(session, [alice, bob])
    bob_state = next(c for c in combat.combatants if c.character_id == "bob")
    bob_state.active_effects.append(DndRuntimeEffect(
        effect_id="eff_hold",
        name="Hold Person",
        slug="hold_person",
        target_id="bob",
        originator_id="alice",
        conditions=["paralyzed"],
    ))
    bob_state.conditions.append("paralyzed")

    updated = update_effect(
        session,
        effect_id="eff_hold",
        target_id="missing",
        conditions=["restrained"],
        reason="the spell shifts",
    )

    assert updated is bob_state.active_effects[0]
    assert bob_state.active_effects[0].conditions == ["restrained"]
    assert "restrained" in bob_state.conditions
    assert "paralyzed" not in bob_state.conditions
    assert (
        "Hold Person changes on Bob because the spell shifts."
        in combat.pending_visible_facts
    )


def test_update_effect_skips_ambiguous_and_target_only_selectors(monkeypatch):
    values = iter([9, 9])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    alice = _character("alice", "Alice")
    bob = _character("bob", "Bob")
    session = SessionState(session_id="s")
    combat = start_combat(session, [alice, bob])
    bob_state = next(c for c in combat.combatants if c.character_id == "bob")
    bob_state.active_effects.extend([
        DndRuntimeEffect(
            effect_id="eff_bless_a",
            name="Bless",
            slug="bless",
            target_id="bob",
            originator_id="cleric_a",
            conditions=["blessed"],
            remaining_rounds=8,
        ),
        DndRuntimeEffect(
            effect_id="eff_bless_b",
            name="Bless",
            slug="bless",
            target_id="bob",
            originator_id="cleric_b",
            conditions=["blessed"],
            remaining_rounds=9,
        ),
    ])
    bob_state.conditions.append("blessed")

    updated = update_effect(
        session,
        target_id="bob",
        slug="bless",
        remaining_rounds=3,
    )

    assert updated is None
    assert [effect.remaining_rounds for effect in bob_state.active_effects] == [
        8,
        9,
    ]
    assert "selector is ambiguous" in combat.audit_lines[-1]

    updated = update_effect(
        session,
        target_id="bob",
        remaining_rounds=1,
    )

    assert updated is None
    assert [effect.remaining_rounds for effect in bob_state.active_effects] == [
        8,
        9,
    ]
    assert "selector is ambiguous" in combat.audit_lines[-1]


def test_advance_turn_runs_recurring_save_and_ends_effect(monkeypatch):
    values = iter([9, 9, 14])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    alice = _character("alice", "Alice")
    bob = _character("bob", "Bob")
    session = SessionState(session_id="s")
    combat = start_combat(session, [alice, bob])
    bob_state = next(c for c in combat.combatants if c.character_id == "bob")
    bob_state.active_effects.append(DndRuntimeEffect(
        effect_id="eff_hold",
        name="Hold Person",
        slug="hold_person",
        target_id="bob",
        originator_id="alice",
        conditions=["paralyzed"],
        concentration=True,
        duration_kind="minutes",
        duration_amount=1,
        remaining_rounds=10,
        recurring_save=DndEffectRecurringSave(
            ability="wis",
            dc=10,
            timing="end_of_turn",
            ends_on="success",
        ),
    ))
    bob_state.conditions.append("paralyzed")
    combat.turn_index = combat.combatants.index(bob_state)

    advance_turn(session)

    assert bob_state.active_effects == []
    assert "paralyzed" not in bob_state.conditions
    assert (
        "Hold Person ends on Bob after a successful Wisdom saving throw."
        in combat.pending_visible_facts
    )


def test_advance_turn_failed_recurring_save_does_not_emit_remains_fact(
    monkeypatch,
):
    values = iter([9, 9, 4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    alice = _character("alice", "Alice")
    bob = _character("bob", "Bob")
    session = SessionState(session_id="s")
    combat = start_combat(session, [alice, bob])
    bob_state = next(c for c in combat.combatants if c.character_id == "bob")
    bob_state.active_effects.append(DndRuntimeEffect(
        effect_id="eff_hold",
        name="Hold Person",
        slug="hold_person",
        target_id="bob",
        originator_id="alice",
        conditions=["paralyzed"],
        concentration=True,
        duration_kind="minutes",
        duration_amount=1,
        remaining_rounds=10,
        recurring_save=DndEffectRecurringSave(
            ability="wis",
            dc=15,
            timing="end_of_turn",
            ends_on="success",
        ),
    ))
    bob_state.conditions.append("paralyzed")
    combat.turn_index = combat.combatants.index(bob_state)

    advance_turn(session)

    assert [effect.effect_id for effect in bob_state.active_effects] == ["eff_hold"]
    assert "paralyzed" in bob_state.conditions
    assert combat.pending_visible_facts == []
    assert "continues" in combat.audit_lines[-1]


def test_advance_turn_expires_readied_action_at_start_of_next_turn():
    session = SessionState(session_id="s")
    alice_state = DndCombatantState(
        combatant_id="alice",
        character_id="alice",
        name="Alice",
        active_effects=[
            DndRuntimeEffect(
                effect_id="ready_magic_missile",
                name="Readied Magic Missile",
                slug="readied_spell",
                source_type="spell",
                source_id="magic_missile",
                originator_id="alice",
                target_id="alice",
                concentration=True,
                duration_kind="rounds",
                duration_amount=1,
                remaining_rounds=1,
                metadata={
                    "readied_action": {
                        "source_id": "magic_missile",
                        "source_type": "spell",
                        "readying_actor_id": "alice",
                        "trigger_text": "when Bob opens the door",
                        "created_round": 1,
                        "created_turn_index": 0,
                        "requires_reaction": True,
                        "expires_at_start_of_actor_turn": True,
                    },
                },
            )
        ],
    )
    bob_state = DndCombatantState(
        combatant_id="bob",
        character_id="bob",
        name="Bob",
    )
    combat = DndCombatState(
        combat_id="combat",
        round_number=1,
        turn_index=1,
        combatants=[alice_state, bob_state],
    )
    session.active_combat = combat

    advance_turn(session)

    assert current_combatant(session).character_id == "alice"
    assert combat.round_number == 2
    assert alice_state.active_effects == []
    assert (
        "Readied Magic Missile ends on Alice as the readying turn begins."
        in combat.pending_visible_facts
    )


def test_damage_can_break_concentration(monkeypatch):
    values = iter([9, 9, 4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    alice = _character("alice", "Alice")
    bob = _character("bob", "Bob")
    session = SessionState(session_id="s")
    combat = start_combat(session, [alice, bob])
    bob_state = next(c for c in combat.combatants if c.character_id == "bob")
    bob_state.active_effects.append(DndRuntimeEffect(
        effect_id="eff_hold",
        name="Hold Person",
        slug="hold_person",
        target_id="bob",
        originator_id="bob",
        conditions=["paralyzed"],
        concentration=True,
        duration_kind="minutes",
        duration_amount=1,
        remaining_rounds=10,
    ))
    bob_state.conditions.append("paralyzed")

    apply_damage(session, "bob", 3, characters=[alice, bob])

    assert bob_state.active_effects == []
    assert "paralyzed" not in bob_state.conditions
    assert (
        "Hold Person ends on Bob because concentration breaks."
        in combat.pending_visible_facts
    )


def test_lifecycle_and_roster_validation(monkeypatch):
    values = iter([1])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    session = SessionState(session_id="s")

    with pytest.raises(ValueError, match="without combatants"):
        start_combat(
            session,
            [_character("ghost", "Ghost", status=CharacterStatus.culled)],
        )

    combat = start_combat(session, [_character("alice", "Alice")])
    with pytest.raises(ValueError, match="already active"):
        start_combat(session, [_character("bob", "Bob")])
    combat.pending_advance_actor_id = "alice"
    session.active_act_slots["alice"] = SlotEntry(
        reason="combat_reaction",
        trigger_event_id="evt_react",
    )
    session.active_act_slots["bob"] = SlotEntry(
        reason="combat_blocked",
        trigger_event_id="evt_blocked",
    )
    session.active_act_slots["pip"] = SlotEntry(
        reason="cat_ii_roll",
        cat_ii_event_id="cmb_1",
    )
    session.cat_ii_roll_transactions.append(CatIIRollTransaction(
        transaction_id="rolltxn_1",
        event_id="cmb_1",
        source="combat",
        actor_id="pip",
        status="awaiting_player_rolls",
    ))

    ended = end_combat(session)
    assert ended is combat
    assert ended.status == "ended"
    assert ended.pending_advance_actor_id == ""
    assert session.active_combat is None
    assert session.active_act_slots == {}
    assert session.cat_ii_roll_transactions[0].status == "cancelled"
    with pytest.raises(ValueError, match="not active"):
        current_combatant(session)
