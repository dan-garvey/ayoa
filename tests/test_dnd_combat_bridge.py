from __future__ import annotations

import asyncio

import pytest

from app.bot.engine_bridge import EngineBridge
from app.engine import dnd_combat
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import SessionState, SlotEntry, WorldState


SESSION_ID = "combat_session"


@pytest.fixture
def bridge(tmp_path, monkeypatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )


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
    original_public_status = dnd_combat.public_status
    original_private_status = dnd_combat.private_status
    original_advance = dnd_combat.advance_turn_with_effects
    original_end = dnd_combat.end_combat

    def public_status_spy(combat):
        calls["status_public"] = True
        return original_public_status(combat)

    def private_status_spy(combat):
        calls["status_private"] = True
        return original_private_status(combat)

    def advance_spy(combat, **kwargs):
        characters = list(kwargs.get("characters") or [])
        calls["next_character_ids"] = [c.character_id for c in characters]
        return original_advance(combat, **kwargs)

    def end_spy(session, **kwargs):
        characters = list(kwargs.get("characters") or [])
        calls["end_character_ids"] = [c.character_id for c in characters]
        return original_end(session, **kwargs)

    monkeypatch.setattr(dnd_combat, "public_status", public_status_spy)
    monkeypatch.setattr(dnd_combat, "private_status", private_status_spy)
    monkeypatch.setattr(dnd_combat, "advance_turn_with_effects", advance_spy)
    monkeypatch.setattr(dnd_combat, "end_combat", end_spy)
    bridge.begin_combat(SESSION_ID, ["alice", "guard"])

    public_status = bridge.combat_status(SESSION_ID)
    assert calls["status_public"] is True
    assert public_status.round_number == 1

    status = bridge.combat_status(SESSION_ID, private=True)
    assert calls["status_private"] is True
    assert status.round_number == 1
    assert status.turn_number >= 1
    assert all(p.initiative is not None for p in status.participants)

    next_view = bridge.combat_next(SESSION_ID)
    assert calls["next_character_ids"] == [
        "alice",
        "bob",
        "guard",
        "distant",
        "sleeping",
    ]
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
    assert calls["end_character_ids"] == [
        "alice",
        "bob",
        "guard",
        "distant",
        "sleeping",
    ]
    assert ended.active is False
    assert ended.message == "Combat ended."


def test_manual_combat_end_broadcasts_observable_event(bridge: EngineBridge):
    _seed(bridge)
    bridge.begin_combat(SESSION_ID, ["alice", "guard"])
    ckpt = bridge.load_latest(SESSION_ID)
    ckpt.session.active_act_slots["distant"] = SlotEntry(
        reason="combat_blocked",
        trigger_event_id="evt_blocked",
    )
    bridge.checkpoint_mgr.save(ckpt)

    ended = bridge.combat_end(SESSION_ID)

    assert ended.active is False
    reloaded = bridge.load_latest(SESSION_ID)
    assert reloaded.session.active_combat is None
    assert reloaded.session.active_act_slots == {}
    assert reloaded.canonical_events
    event = reloaded.canonical_events[-1]
    assert "distant" not in {observer.character_id for observer in event.observers}
    facts = [
        fact.text
        for fact in event.canonical_event.observable_facts
    ]
    assert "D&D combat ends." in facts
    assert all("You may act again" not in fact for fact in facts)
    assert reloaded.session.render_buffers["alice"]
    assert "distant" not in reloaded.session.render_buffers
    guard = next(c for c in reloaded.characters if c.character_id == "guard")
    assert "D&D combat ends." in guard.pending_observations


@pytest.mark.asyncio
async def test_locked_combat_damage_waits_for_session_lock(bridge: EngineBridge):
    _seed(bridge)
    bridge.begin_combat(SESSION_ID, ["alice", "guard"])
    lock = await bridge._lock_for(SESSION_ID)
    await lock.acquire()
    task = asyncio.create_task(
        bridge.combat_damage_locked(SESSION_ID, "guard", 3)
    )
    try:
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        lock.release()
    damaged = await task
    guard = next(p for p in damaged.participants if p.character_id == "guard")
    assert guard.hp_current == 4


@pytest.mark.asyncio
async def test_locked_combat_damage_waits_for_orchestrator_lock(
    bridge: EngineBridge,
):
    _seed(bridge)
    bridge.begin_combat(SESSION_ID, ["alice", "guard"])
    lock = await bridge.orchestrator.session_locks.get(SESSION_ID)
    await lock.acquire()
    task = asyncio.create_task(
        bridge.combat_damage_locked(SESSION_ID, "guard", 3)
    )
    try:
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        lock.release()
    damaged = await task
    guard = next(p for p in damaged.participants if p.character_id == "guard")
    assert guard.hp_current == 4


def test_combat_bridge_reports_missing_core(bridge: EngineBridge, monkeypatch):
    _seed(bridge)

    def missing_core():
        raise RuntimeError("expected app.engine.dnd_combat")

    monkeypatch.setattr(bridge, "_dnd_combat_module", missing_core)

    with pytest.raises(RuntimeError, match="app.engine.dnd_combat"):
        bridge.combat_status(SESSION_ID)
