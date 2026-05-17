import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.engine.dnd_cat_ii import (
    DndCatIIRollsPending,
    DndCatIIResolver,
    complete_pending_player_roll,
)
from app.llm.client import LLMResponse
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_cat_ii import PlannedRoll, RollPlan, RulesAdjudication
from app.schemas.state import OpenCatIIEvent, SessionState, WorldState


def _ckpt() -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(session_id="s"),
        world_state=WorldState(),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                public_sheet=PublicSheet(role="fighter"),
                mechanics={
                    "ruleset_id": "dnd5e_basic",
                    "ability_scores": {"str": 16, "dex": 12},
                    "proficiency_bonus": 2,
                    "skill_proficiencies": ["athletics"],
                },
            ),
            CharacterRecord(
                character_id="pip",
                name="Pip",
                public_sheet=PublicSheet(role="goblin"),
                mechanics={
                    "ruleset_id": "dnd5e_basic",
                    "ability_scores": {"str": 8, "dex": 14},
                    "proficiency_bonus": 2,
                    "skill_proficiencies": ["acrobatics"],
                },
            ),
        ],
    )


def _llm_response(parsed) -> LLMResponse:
    raw = MagicMock()
    raw.content = []
    raw.model = "gpt-5.2"
    return LLMResponse(
        parsed=parsed,
        raw_response=raw,
        content="{}",
        model="gpt-5.2",
    )


def _opposed_plan() -> RollPlan:
    return RollPlan(
        needs_rolls=True,
        roll_requests=[
            PlannedRoll(
                roll_id="roll_alice",
                actor_id="alice",
                kind="skill_check",
                ability="str",
                skill="athletics",
                dc=0,
                opposed_by="roll_pip",
                advantage_state="normal",
                reason="Alice tries to shove Pip away from the door.",
            ),
            PlannedRoll(
                roll_id="roll_pip",
                actor_id="pip",
                kind="skill_check",
                ability="dex",
                skill="acrobatics",
                dc=0,
                opposed_by="roll_alice",
                advantage_state="normal",
                reason="Pip tries to keep his feet.",
            ),
        ],
        no_roll_reason="",
    )


def _open_event() -> OpenCatIIEvent:
    return OpenCatIIEvent(
        event_id="evt_open",
        initiator_id="alice",
        initiator_intention="I shove Pip away from the door",
        required_responders=["pip"],
        collected_intentions={"pip": "I twist aside"},
        opening_observer_ids=["alice", "pip"],
        opening_observable_facts=[
            "Alice lunges toward Pip at the doorway.",
        ],
    )


def test_dnd_cat_ii_executes_roll_plan_and_compiles_router_output(monkeypatch):
    from app.engine import dice

    ckpt = _ckpt()
    values = iter([9, 12])
    monkeypatch.setattr(
        dice.d20.expression.random, "randrange", lambda _: next(values)
    )

    adjudication = RulesAdjudication(
        feasible=True,
        mechanical_summary="Alice beats Pip's opposed check.",
        visible_outcome_facts=[
            "Alice drives Pip back from the doorway.",
        ],
        state_deltas=[],
        rules_notes=["Opposed Athletics versus Acrobatics."],
        fallback_reason="",
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_opposed_plan()),
        _llm_response(adjudication),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCatIIResolver(client, prompt_mgr).resolve_cat_ii(
            ckpt=ckpt,
            cat_ii_event=_open_event(),
        )
    )

    assert [call.kwargs["role"] for call in client.complete.await_args_list] == [
        "event_router",
        "event_router",
    ]
    assert routed.requires_responders is False
    assert routed.ends_beat is True
    assert routed.ends_beat_reason == "cat_ii_resolution"
    assert routed.next_output_character_ids == []
    assert [o.character_id for o in routed.observers] == ["alice", "pip"]
    assert routed.canonical_event.observable_facts[0].text == (
        "Alice drives Pip back from the doorway."
    )
    assert "roll_alice" not in routed.decision_rationale
    assert "roll_pip" not in routed.decision_rationale

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.status == "finalized"
    assert [r.roll_id for r in transaction.rolls] == ["roll_alice", "roll_pip"]
    assert "roll_alice" in transaction.ledger_lines[0]
    assert "D&D Cat II resolved" in ckpt.session.pending_router_state_changes[0]
    assert ckpt.session_conversation == []


def test_dnd_cat_ii_interactive_player_roll_pauses_until_roll_submitted(
    monkeypatch,
):
    from app.engine import dice

    ckpt = _ckpt()
    ckpt.session.character_bindings["alice"] = "discord_1"
    ckpt.session.config.settings.player_roll_mode = "interactive"
    values = iter([12, 9])
    monkeypatch.setattr(
        dice.d20.expression.random, "randrange", lambda _: next(values)
    )

    adjudication = RulesAdjudication(
        feasible=False,
        mechanical_summary="Pip beats Alice's opposed check.",
        visible_outcome_facts=["Alice fails to move Pip from the doorway."],
        state_deltas=[],
        rules_notes=[],
        fallback_reason="",
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_opposed_plan()),
        _llm_response(adjudication),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]
    evt = _open_event()
    resolver = DndCatIIResolver(client, prompt_mgr)

    try:
        asyncio.run(resolver.resolve_cat_ii(ckpt=ckpt, cat_ii_event=evt))
    except DndCatIIRollsPending as exc:
        transaction = exc.transaction
    else:
        raise AssertionError("interactive player roll did not pause")

    assert transaction.status == "awaiting_player_rolls"
    assert transaction.rolls[0].status == "pending"
    assert transaction.rolls[1].status == "completed"
    assert ckpt.session.active_act_slots["alice"].reason == "cat_ii_roll"
    assert client.complete.await_count == 1

    complete_pending_player_roll(
        ckpt,
        event_id=evt.event_id,
        roll_id="roll_alice",
        completed_by_user_id="discord_1",
    )
    routed = asyncio.run(resolver.resolve_cat_ii(ckpt=ckpt, cat_ii_event=evt))

    assert routed.canonical_event.observable_facts[0].text == (
        "Alice fails to move Pip from the doorway."
    )
    assert transaction.status == "finalized"
    assert ckpt.session.active_act_slots == {}
    assert [call.kwargs["role"] for call in client.complete.await_args_list] == [
        "event_router",
        "event_router",
    ]
