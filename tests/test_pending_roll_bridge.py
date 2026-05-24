from __future__ import annotations

import pytest

from app.bot.commands import _dice_roll_content, _roll_result_line
from app.bot.engine_bridge import EngineBridge
from app.engine.frontend_views import CompletedPendingRoll
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.responses import DiceRollDisplay
from app.schemas.state import (
    CatIIRollRecord,
    CatIIRollTransaction,
    DndCombatantState,
    DndCombatState,
    OpenCatIIEvent,
    SessionState,
    SlotEntry,
    WorldState,
)


def _pending_roll_checkpoint() -> CheckpointFile:
    ckpt = CheckpointFile(
        session=SessionState(
            session_id="roll_session",
            story_id="story",
            turn_index=4,
        ),
        world_state=WorldState(),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                public_sheet=PublicSheet(role="duelist"),
            ),
            CharacterRecord(
                character_id="pip",
                name="Pip",
                public_sheet=PublicSheet(role="duelist"),
            ),
        ],
    )
    ckpt.session.character_bindings["alice"] = "123"
    ckpt.session.active_act_slots["alice"] = SlotEntry(
        reason="cat_ii_roll",
        cat_ii_event_id="evt_open",
    )
    ckpt.session.open_cat_ii_events.append(
        OpenCatIIEvent(
            event_id="evt_open",
            initiator_id="pip",
            initiator_intention="Pip sweeps at Alice's feet.",
            required_responders=["alice"],
            collected_intentions={"alice": "Alice tumbles aside."},
            opening_observer_ids=["alice", "pip"],
            opening_observable_facts=[
                "Pip hooks a practice blade toward Alice's ankles.",
            ],
            roll_transaction_id="rolltxn_1",
        )
    )
    ckpt.session.cat_ii_roll_transactions.append(
        CatIIRollTransaction(
            transaction_id="rolltxn_1",
            event_id="evt_open",
            ruleset_id="dnd5e_basic",
            status="awaiting_player_rolls",
            plan={
                "needs_rolls": True,
                "roll_requests": [
                    {
                        "roll_id": "roll_alice",
                        "actor_id": "alice",
                        "kind": "skill_check",
                        "ability": "dex",
                        "skill": "acrobatics",
                        "dc": 0,
                        "opposed_by": "",
                        "advantage_state": "normal",
                        "reason": "Alice tries to stay on her feet.",
                    },
                ],
                "no_roll_reason": "",
            },
            rolls=[
                CatIIRollRecord(
                    roll_id="roll_alice",
                    actor_id="alice",
                    actor_control="player",
                    request={
                        "roll_id": "roll_alice",
                        "actor_id": "alice",
                        "kind": "skill_check",
                        "ability": "dex",
                        "skill": "acrobatics",
                        "dc": 0,
                        "opposed_by": "",
                        "advantage_state": "normal",
                        "reason": "Alice tries to stay on her feet.",
                    },
                    modifier=3,
                    label="Acrobatics",
                    reason="Alice tries to stay on her feet.",
                ),
            ],
            ledger_lines=[],
        )
    )
    return ckpt


def _combat_pending_roll_checkpoint() -> CheckpointFile:
    ckpt = _pending_roll_checkpoint()
    ckpt.session.open_cat_ii_events = []
    ckpt.session.active_act_slots["alice"] = SlotEntry(
        reason="cat_ii_roll",
        cat_ii_event_id="cmb_open",
    )
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    transaction.event_id = "cmb_open"
    transaction.source = "combat"
    transaction.actor_id = "pip"
    ckpt.session.active_combat = DndCombatState(
        turn_index=1,
        combatants=[
            DndCombatantState(
                combatant_id="pip",
                character_id="pip",
                name="Pip",
            ),
            DndCombatantState(
                combatant_id="alice",
                character_id="alice",
                name="Alice",
                player_controlled=True,
            ),
        ],
    )
    return ckpt


@pytest.mark.asyncio
async def test_complete_pending_roll_saves_dice_before_router_finalize(
    tmp_path,
    monkeypatch,
):
    from app.engine import dice

    bridge = EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )
    bridge.checkpoint_mgr.save(_pending_roll_checkpoint())
    monkeypatch.setattr(dice.d20.expression.random, "randrange", lambda _: 15)

    result = await bridge.complete_pending_roll(
        session_id="roll_session",
        event_id="evt_open",
        roll_id="roll_alice",
        user_id=123,
    )

    assert result.total == 19
    assert result.detail == "1d20 (16) + 3 = `19`"
    assert result.remaining_pending_rolls == 0

    latest = bridge.checkpoint_mgr.load_latest("roll_session")
    transaction = latest.session.cat_ii_roll_transactions[0]
    assert latest.session.turn_index == 4
    assert latest.canonical_events == []
    assert latest.session.open_cat_ii_events[0].event_id == "evt_open"
    assert latest.session.active_act_slots["alice"].reason == "cat_ii_roll"
    assert transaction.status == "ready_to_finalize"
    assert transaction.final_event_id == ""
    assert transaction.rolls[0].status == "completed"
    assert transaction.rolls[0].result["total"] == 19


@pytest.mark.asyncio
async def test_complete_pending_roll_accepts_combat_transaction_without_cat_ii(
    tmp_path,
    monkeypatch,
):
    from app.engine import dice

    bridge = EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )
    bridge.checkpoint_mgr.save(_combat_pending_roll_checkpoint())
    monkeypatch.setattr(dice.d20.expression.random, "randrange", lambda _: 15)

    result = await bridge.complete_pending_roll(
        session_id="roll_session",
        event_id="cmb_open",
        roll_id="roll_alice",
        user_id=123,
    )

    assert result.total == 19
    latest = bridge.checkpoint_mgr.load_latest("roll_session")
    assert latest.session.open_cat_ii_events == []
    transaction = latest.session.cat_ii_roll_transactions[0]
    assert transaction.source == "combat"
    assert transaction.status == "ready_to_finalize"
    assert transaction.rolls[0].status == "completed"


@pytest.mark.asyncio
async def test_complete_pending_roll_rejects_stale_combat_transaction(tmp_path):
    bridge = EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )
    ckpt = _combat_pending_roll_checkpoint()
    ckpt.session.active_combat = None
    bridge.checkpoint_mgr.save(ckpt)

    with pytest.raises(ValueError, match="combat roll is no longer active"):
        await bridge.complete_pending_roll(
            session_id="roll_session",
            event_id="cmb_open",
            roll_id="roll_alice",
            user_id=123,
        )


def test_roll_result_line_surfaces_total_for_discord_ui():
    line = _roll_result_line(
        CompletedPendingRoll(
            session_id="s",
            event_id="evt",
            roll_id="roll_1",
            actor_id="alice",
            user_id="123",
            label="Attack",
            reason="",
            expression="1d20+4",
            total=24,
            detail="1d20 (**20**) + 4 = `24`",
            crit="crit",
            remaining_pending_rolls=0,
        )
    )

    assert line == (
        "**Rolled Attack:** 1d20 (**20**) + 4 = `24`. Critical success."
    )


def test_dice_roll_content_formats_d20_equation_for_discord_ui():
    content = _dice_roll_content(
        CompletedPendingRoll(
            session_id="s",
            event_id="evt",
            roll_id="roll_1",
            actor_id="alice",
            user_id="123",
            label="Attack",
            reason="",
            expression="1d20+4",
            total=19,
            detail="1d20 (15) + 4 = `19`",
            crit="none",
            remaining_pending_rolls=0,
            dc=12,
            outcome="hit",
        )
    )

    assert "D&D Roll: alice - Attack" in content
    assert "`d20 15 + 4 = 19 vs DC 12`" in content
    assert "**Hit**" in content


def test_dice_roll_content_formats_damage_roll_without_d20_for_discord_ui():
    content = _dice_roll_content(
        DiceRollDisplay(
            actor_id="mon_mountain_lion_1",
            actor_name="Mountain Lion",
            target_id="pc_expedition_leader",
            target_name="Demo Expedition Leader",
            label="Damage (Claw)",
            kind="damage_roll",
            total=4,
            damage_raw_total=4,
            damage_total=4,
            damage_type="slashing",
            damage_expression="1d4+2",
            damage_detail="1d4 (2) + 2 = `4`",
            target_hp_before=33,
            target_hp_after=29,
            target_hp_max=38,
            target_defeat_state="active",
        )
    )

    assert "D&D Damage: Mountain Lion - Damage (Claw)" in content
    assert "d20 ?" not in content
    assert "`Damage: 1d4 (2) + 2 = 4 slashing`" in content
    assert "Target HP: Demo Expedition Leader 33/38 -> 29/38" in content
