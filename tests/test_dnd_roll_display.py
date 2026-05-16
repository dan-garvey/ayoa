from __future__ import annotations

from app.engine.dnd_roll_display import (
    completed_automatic_roll_keys,
    dice_roll_displays_since,
)
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import (
    CatIIRollDamageRecord,
    CatIIRollRecord,
    CatIIRollTransaction,
    DndCombatantState,
    DndCombatState,
    SessionState,
    WorldState,
)


def _completed_attack_record(*, completed_by_user_id: str = "engine"):
    return CatIIRollRecord(
        roll_id="attack_alice",
        actor_id="alice",
        actor_control="agent",
        status="completed",
        request={
            "roll_id": "attack_alice",
            "actor_id": "alice",
            "kind": "attack_roll",
            "ability": "str",
            "skill": "",
            "dc": 0,
            "opposed_by": "",
            "advantage_state": "normal",
            "reason": "Alice swings at the rat.",
            "action_id": "shortsword",
            "target_id": "rat",
            "damage_adjustments": [],
        },
        modifier=4,
        label="Attack (Shortsword)",
        reason="Alice swings at the rat.",
        result={
            "roll_id": "attack_alice",
            "expression": "1d20+4",
            "total": 20,
            "detail": "1d20 (16) + 4 = `20`",
            "crit": "none",
            "dice": [
                {"size": 20, "values": [16], "kept": True, "total": 16},
            ],
        },
        completed_by_user_id=completed_by_user_id,
        completed_at="2026-05-15T12:00:00Z",
    )


def _checkpoint(record: CatIIRollRecord) -> CheckpointFile:
    ckpt = CheckpointFile(
        session=SessionState(session_id="s"),
        world_state=WorldState(),
        characters=[
            CharacterRecord(character_id="alice", name="Alice"),
        ],
    )
    ckpt.session.active_combat = DndCombatState(
        combatants=[
            DndCombatantState(
                combatant_id="alice",
                character_id="alice",
                name="Alice",
                armor_class=14,
            ),
            DndCombatantState(
                combatant_id="rat",
                character_id="rat",
                name="Giant Rat",
                armor_class=12,
            ),
        ]
    )
    ckpt.session.cat_ii_roll_transactions.append(
        CatIIRollTransaction(
            transaction_id="rolltxn_1",
            event_id="evt_1",
            source="combat",
            actor_id="alice",
            status="finalized",
            rolls=[record],
            damage_records=[
                CatIIRollDamageRecord(
                    roll_id="attack_alice",
                    target_id="rat",
                    raw_amount=7,
                    amount=7,
                    damage_type="piercing",
                    detail="1d6 (3) + 4",
                    applied=True,
                )
            ],
        )
    )
    return ckpt


def test_dice_roll_displays_since_projects_completed_engine_roll():
    ckpt = _checkpoint(_completed_attack_record())

    displays = dice_roll_displays_since(ckpt, before=set())

    assert len(displays) == 1
    display = displays[0]
    assert display.roll_id == "attack_alice"
    assert display.actor_name == "Alice"
    assert display.target_name == "Giant Rat"
    assert display.die_values == [16]
    assert display.modifier == 4
    assert display.total == 20
    assert display.dc == 12
    assert display.outcome == "hit"
    assert display.damage_total == 7
    assert display.damage_type == "piercing"


def test_dice_roll_displays_since_ignores_seen_and_player_rolls():
    engine_ckpt = _checkpoint(_completed_attack_record())
    before = completed_automatic_roll_keys(engine_ckpt)
    assert dice_roll_displays_since(engine_ckpt, before=before) == []

    player_ckpt = _checkpoint(_completed_attack_record(
        completed_by_user_id="123",
    ))
    assert dice_roll_displays_since(player_ckpt, before=set()) == []
