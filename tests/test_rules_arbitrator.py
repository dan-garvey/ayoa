import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.engine.rules_arbitrator import RulesArbitrator
from app.llm.client import LLMResponse
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.rules_arbitrator import PlannedRoll, RollPlan, RulesAdjudication
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


def test_rules_arbitrator_executes_roll_plan_and_compiles_router_output(
    monkeypatch,
):
    from app.engine import dice

    ckpt = _ckpt()
    values = iter([9, 12])
    monkeypatch.setattr(dice.d20.expression.random, "randrange", lambda _: next(values))

    plan = RollPlan(
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
        _llm_response(plan),
        _llm_response(adjudication),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]
    evt = OpenCatIIEvent(
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

    routed = asyncio.run(
        RulesArbitrator(client, prompt_mgr).resolve_cat_ii(
            ckpt=ckpt,
            cat_ii_event=evt,
        )
    )

    assert [call.kwargs["role"] for call in client.complete.await_args_list] == [
        "rules_arbitrator",
        "rules_arbitrator",
    ]
    assert routed.requires_responders is False
    assert routed.ends_beat is True
    assert routed.ends_beat_reason == "cat_ii_resolution"
    assert routed.agent_responder_picks == []
    assert [o.character_id for o in routed.observers] == ["alice", "pip"]
    assert routed.canonical_event.observable_facts[0].text == (
        "Alice drives Pip back from the doorway."
    )
    assert "roll_alice" in routed.decision_rationale
    assert "roll_pip" in routed.decision_rationale
    assert "Rules adjudication resolved Cat II" in (
        ckpt.session.pending_router_state_changes[0]
    )
