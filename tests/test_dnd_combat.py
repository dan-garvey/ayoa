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
    start_combat,
)
from app.schemas.characters import CharacterRecord, CharacterStatus, PublicSheet
from app.schemas.state import SessionState, SlotEntry


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
    assert damaged.defeated is True

    healed = apply_healing(session, "hero", 4)
    assert healed.hit_points_current == 4
    assert healed.defeated is False


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
            "defeated": False,
            "removed": False,
        },
        "turn_order": [
            {
                "combatant_id": "alice",
                "character_id": "alice",
                "name": "Alice",
                "hp": {"current": 12, "max": 12, "temporary": 0},
                "defeated": False,
                "removed": False,
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

    ended = end_combat(session)
    assert ended is combat
    assert ended.status == "ended"
    assert ended.pending_advance_actor_id == ""
    assert session.active_combat is None
    assert session.active_act_slots == {}
    with pytest.raises(ValueError, match="not active"):
        current_combatant(session)
