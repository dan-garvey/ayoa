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


def test_opposed_npc_rolls_wait_for_player_resolution_before_display():
    npc_record = CatIIRollRecord(
        roll_id="roll_pip",
        actor_id="pip",
        actor_control="agent",
        status="completed",
        request={
            "roll_id": "roll_pip",
            "actor_id": "pip",
            "kind": "skill_check",
            "ability": "wis",
            "skill": "insight",
            "dc": 0,
            "opposed_by": "roll_alice",
            "advantage_state": "normal",
            "reason": "Pip reads Alice's bluff.",
        },
        modifier=2,
        label="Insight",
        reason="Pip reads Alice's bluff.",
        result={
            "roll_id": "roll_pip",
            "expression": "1d20+2",
            "total": 15,
            "detail": "1d20 (13) + 2 = `15`",
            "crit": "none",
            "dice": [
                {"size": 20, "values": [13], "kept": True, "total": 13},
            ],
        },
        completed_by_user_id="engine",
    )
    player_record = CatIIRollRecord(
        roll_id="roll_alice",
        actor_id="alice",
        actor_control="player",
        status="pending",
        request={
            "roll_id": "roll_alice",
            "actor_id": "alice",
            "kind": "skill_check",
            "ability": "cha",
            "skill": "deception",
            "dc": 0,
            "opposed_by": "roll_pip",
            "advantage_state": "normal",
            "reason": "Alice keeps her story straight.",
        },
        modifier=5,
        label="Deception",
        reason="Alice keeps her story straight.",
    )
    ckpt = CheckpointFile(
        session=SessionState(session_id="s"),
        world_state=WorldState(),
        characters=[
            CharacterRecord(character_id="alice", name="Alice"),
            CharacterRecord(character_id="pip", name="Pip"),
        ],
    )
    transaction = CatIIRollTransaction(
        transaction_id="rolltxn_social",
        event_id="evt_social",
        source="cat_ii",
        actor_id="alice",
        status="awaiting_player_rolls",
        rolls=[player_record, npc_record],
    )
    ckpt.session.cat_ii_roll_transactions.append(transaction)

    before = completed_automatic_roll_keys(ckpt)

    assert before == set()
    assert dice_roll_displays_since(ckpt, before=set()) == []

    player_record.status = "completed"
    player_record.completed_by_user_id = "123"
    transaction.status = "ready_to_finalize"
    assert completed_automatic_roll_keys(ckpt) == set()

    transaction.status = "finalized"
    displays = dice_roll_displays_since(ckpt, before=before)

    assert len(displays) == 1
    assert displays[0].roll_id == "roll_pip"
    assert displays[0].actor_name == "Pip"
    assert displays[0].total == 15
