import pytest

from app.engine import dice
from app.engine.dnd_combat import (
    add_combatant,
    advance_turn,
    apply_damage,
    apply_healing,
    current_combatant,
    end_combat,
    private_status,
    public_status,
    remove_combatant,
    roll_death_save,
    start_combat,
)
from app.schemas.characters import CharacterRecord, CharacterStatus, PublicSheet
from app.schemas.state import CatIIRollTransaction, SessionState, SlotEntry


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
