from __future__ import annotations

import pytest

from app.bot.engine_bridge import EngineBridge
from app.engine import dnd_combat
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import SessionState, WorldState


SESSION_ID = "combat_session"


@pytest.fixture
def bridge(tmp_path, monkeypatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return EngineBridge(saves_dir=str(tmp_path), prompts_dir="app/prompts")


def _char(
    character_id: str,
    *,
    name: str,
    location: str = "Hall",
    active: bool = True,
    hp_current: int = 10,
    hp_max: int = 10,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        status="active" if active else "dormant",
        location=location,
        mechanics={
            "armor_class": 12,
            "hit_points": {
                "current": hp_current,
                "max": hp_max,
                "temporary": 0,
            },
        },
    )


def _seed(bridge: EngineBridge) -> None:
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            story_id="story",
            turn_index=1,
            character_bindings={"alice": "111", "bob": "222"},
        ),
        world_state=WorldState(),
        characters=[
            _char("alice", name="Alice", location="Hall"),
            _char("bob", name="Bob", location="Hall"),
            _char("guard", name="Guard", location="Hall", hp_current=7, hp_max=7),
            _char("distant", name="Distant", location="Tower"),
            _char("sleeping", name="Sleeping", location="Hall", active=False),
        ],
    )
    bridge.checkpoint_mgr.save(ckpt)


def test_begin_combat_defaults_to_bound_party_and_same_location_npcs(
    bridge: EngineBridge,
    monkeypatch,
):
    _seed(bridge)
    original_start = dnd_combat.start_combat
    calls: dict[str, object] = {}

    def start_spy(session, characters, **kwargs):
        selected = list(characters)
        calls["begin_participant_ids"] = [c.character_id for c in selected]
        return original_start(session, selected, **kwargs)

    monkeypatch.setattr(dnd_combat, "start_combat", start_spy)

    view = bridge.begin_combat(SESSION_ID)

    assert calls["begin_participant_ids"] == ["alice", "bob", "guard"]
    assert view.active is True
    assert {p.character_id for p in view.participants} == {"alice", "bob", "guard"}
    assert all(p.hp_current is not None for p in view.participants)


def test_combat_status_and_mutations_delegate_and_persist(
    bridge: EngineBridge,
    monkeypatch,
):
    _seed(bridge)
    calls: dict[str, object] = {}
    original_private_status = dnd_combat.private_status

    def private_status_spy(combat):
        calls["status_private"] = True
        return original_private_status(combat)

    monkeypatch.setattr(dnd_combat, "private_status", private_status_spy)
    bridge.begin_combat(SESSION_ID, ["alice", "guard"])

    status = bridge.combat_status(SESSION_ID, private=True)
    assert calls["status_private"] is True
    assert status.round_number == 1
    assert status.turn_number >= 1
    assert all(p.initiative is not None for p in status.participants)

    next_view = bridge.combat_next(SESSION_ID)
    assert next_view.current_participant_id in {"alice", "guard"}

    added = bridge.combat_add(SESSION_ID, "bob")
    assert "bob" in {p.character_id for p in added.participants}
    removed = bridge.combat_remove(SESSION_ID, "bob")
    bob = next(p for p in removed.participants if p.character_id == "bob")
    assert bob.character_id == "bob"

    damaged = bridge.combat_damage(SESSION_ID, "guard", 3)
    guard = next(p for p in damaged.participants if p.character_id == "guard")
    assert guard.hp_current == 4

    healed = bridge.combat_heal(SESSION_ID, "guard", 2)
    guard = next(p for p in healed.participants if p.character_id == "guard")
    assert guard.hp_current == 6
    status_after_heal = bridge.combat_status(SESSION_ID, private=True)
    guard = next(
        p for p in status_after_heal.participants if p.character_id == "guard"
    )
    assert guard.hp_current == 6

    reloaded = bridge.load_latest(SESSION_ID)
    combatant = next(
        c for c in reloaded.session.active_combat.combatants
        if c.combatant_id == "guard"
    )
    assert combatant.hit_points_current == 6

    ended = bridge.combat_end(SESSION_ID)
    assert ended.active is False
    assert ended.message == "Combat ended."


def test_manual_combat_end_broadcasts_observable_event(bridge: EngineBridge):
    _seed(bridge)
    bridge.begin_combat(SESSION_ID, ["alice", "guard"])

    ended = bridge.combat_end(SESSION_ID)

    assert ended.active is False
    reloaded = bridge.load_latest(SESSION_ID)
    assert reloaded.session.active_combat is None
    assert reloaded.canonical_events
    facts = [
        fact.text
        for fact in reloaded.canonical_events[-1].canonical_event.observable_facts
    ]
    assert "D&D combat ends." in facts
    assert reloaded.session.render_buffers["alice"]
    guard = next(c for c in reloaded.characters if c.character_id == "guard")
    assert "D&D combat ends." in guard.pending_observations


def test_combat_bridge_reports_missing_core(bridge: EngineBridge, monkeypatch):
    _seed(bridge)

    def missing_core():
        raise RuntimeError("expected app.engine.dnd_combat")

    monkeypatch.setattr(bridge, "_dnd_combat_module", missing_core)

    with pytest.raises(RuntimeError, match="app.engine.dnd_combat"):
        bridge.combat_status(SESSION_ID)
