"""Orchestrator integration tests for the v11 turn path."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.engine import dice
from app.engine.dnd_combat import current_combatant
from app.engine.orchestrator import Orchestrator
from app.engine.turn_loop import BeatResult
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import (
    DndEventRouterOutput,
    EventRouterOutput,
    ObserverEntry,
)
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    WorldAdjudication,
)
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.requests import TurnRequest
from app.schemas.state import (
    CatIIRollTransaction,
    DndCombatantState,
    DndCombatState,
    LocationState,
    OpenCatIIEvent,
    SessionState,
    SlotEntry,
    WorldState,
)


def _ckpt(bindings: dict[str, str] | None = None) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id="s",
            turn_index=0,
            player_character_id="alice",
            character_bindings=bindings or {"alice": "u1"},
        ),
        world_state=WorldState(locations=LocationState()),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                public_sheet=PublicSheet(role="player"),
                location="gatehouse",
                is_playable=True,
            ),
            CharacterRecord(
                character_id="bob",
                name="Bob",
                public_sheet=PublicSheet(role="player"),
                location="gatehouse",
                is_playable=True,
            ),
            CharacterRecord(
                character_id="pip",
                name="Pip",
                public_sheet=PublicSheet(role="guard"),
                location="gatehouse",
                is_playable=False,
            ),
        ],
    )


def _router_out(
    *,
    requires_responders: bool = False,
    required_responders: list[str] | None = None,
    agent_picks: list[str] | None = None,
    ends_beat: bool = True,
    ends_beat_reason: str = "directed_at_player",
    facts: list[ObservableFact | str] | None = None,
) -> EventRouterOutput:
    picks = agent_picks or []
    required = required_responders or []
    observer_ids = ["alice", *picks, *required]
    observers: list[ObserverEntry] = []
    seen: set[str] = set()
    for cid in observer_ids:
        if cid in seen:
            continue
        seen.add(cid)
        observers.append(
            ObserverEntry(
                character_id=cid,
                observation_level="d",
                response_priority=5 if cid in required else 3,
            )
        )
    if not ends_beat:
        ends_beat_reason = ""
    return EventRouterOutput(
        event_id="",
        decision_rationale="test fixture",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(feasible=True),
            observable_facts=facts if facts is not None else [
                ObservableFact.all("Something happens.")
            ],
        ),
        observers=observers,
        requires_responders=requires_responders,
        required_responders=required,
        agent_responder_picks=picks,
        ends_beat=ends_beat,
        ends_beat_reason=ends_beat_reason,
        spawn=[],
        dormant=[],
        cull=[],
    )


def _dnd_router_out(
    *,
    interaction_mode: str,
    combatant_ids: list[str] | None = None,
    **kwargs,
) -> DndEventRouterOutput:
    data = _router_out(**kwargs).model_dump()
    data["interaction_mode"] = interaction_mode
    data["combatant_ids"] = combatant_ids or []
    return DndEventRouterOutput(**data)


class FakeDispatcher:
    _route_responses: list[EventRouterOutput] = []
    _agent_responses: list[str] = []
    _narrator_text: str = "POV_RENDER"
    route_calls: list[dict] = []
    agent_calls: list[dict] = []
    narrator_calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def reset(cls) -> None:
        cls._route_responses = []
        cls._agent_responses = []
        cls._narrator_text = "POV_RENDER"
        cls.route_calls = []
        cls.agent_calls = []
        cls.narrator_calls = []

    @classmethod
    def queue_route(cls, response: EventRouterOutput) -> None:
        cls._route_responses.append(response)

    @classmethod
    def queue_agent(cls, intention: str) -> None:
        cls._agent_responses.append(intention)

    async def route_intention(self, **kw) -> EventRouterOutput:
        type(self).route_calls.append(kw)
        return type(self)._route_responses.pop(0)

    async def route_continuation(self, **kw) -> EventRouterOutput:
        type(self).route_calls.append(kw)
        return type(self)._route_responses.pop(0)

    async def route_combat_action(self, **kw) -> EventRouterOutput:
        type(self).route_calls.append(kw)
        return type(self)._route_responses.pop(0)

    async def continue_combat_transaction(self, **kw) -> EventRouterOutput:
        type(self).route_calls.append(kw)
        return type(self)._route_responses.pop(0)

    async def agent_intend(self, **kw) -> str:
        type(self).agent_calls.append(kw)
        return type(self)._agent_responses.pop(0)

    async def narrator_compose(self, **kw):
        type(self).narrator_calls.append(kw)
        envelope = NarratorFinalOutput(final_text=type(self)._narrator_text)
        entry = TranscriptEntry(
            user=kw.get("user_input", ""),
            assistant=type(self)._narrator_text,
        )
        return envelope, entry


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeDispatcher.reset()
    yield
    FakeDispatcher.reset()


@pytest.fixture
def patched_orchestrator(monkeypatch):
    monkeypatch.setattr("app.engine.orchestrator.LLMDispatcher", FakeDispatcher)
    client = MagicMock()
    client.config = MagicMock()
    prompt_mgr = MagicMock()

    def _factory(ckpt: CheckpointFile):
        mgr = MagicMock()
        mgr.load_latest.return_value = ckpt
        mgr.save = MagicMock()
        return Orchestrator(client, mgr, prompt_mgr), mgr

    return _factory


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_cat_i_ends_beat_populates_renders_and_saves(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(ends_beat=True))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I look around",
            acting_character_id="alice",
        ))

        assert response.per_player_renders["alice"] == "POV_RENDER"
        assert response.output_text == "POV_RENDER"
        assert response.beat_ended_reason == "directed_at_player"
        assert response.turn_index == 1
        assert mgr.save.call_count == 1
        saved = mgr.save.call_args[0][0]
        assert len(saved.canonical_events) == 1
        assert saved.session.active_act_slots == {}
        assert len(saved.transcript) == 1
        assert saved.transcript[0].assistant == "POV_RENDER"


class TestSlotRejection:
    @pytest.mark.asyncio
    async def test_second_act_against_held_slot_rejected_without_save(
        self, patched_orchestrator,
    ):
        from app.engine.turn_loop import claim_initiator_slot

        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        claim_initiator_slot(ckpt, "alice")
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I speak up",
            acting_character_id="bob",
        ))

        assert "didn't go through" in response.output_text
        assert response.per_player_renders == {}
        assert response.beat_ended_reason == "slot_rejected"
        assert mgr.save.call_count == 0
        assert FakeDispatcher.route_calls == []


class TestPendingCombatRolls:
    @pytest.mark.asyncio
    async def test_continue_roll_without_open_cat_ii_finalizes_combat(
        self,
        patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.turn_index = 4
        ckpt.session.active_combat = DndCombatState(
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="cat_ii_roll",
            cat_ii_event_id="cmb_1",
        )
        ckpt.session.cat_ii_roll_transactions.append(
            CatIIRollTransaction(
                transaction_id="rolltxn_1",
                event_id="cmb_1",
                source="combat",
                actor_id="alice",
                status="ready_to_finalize",
                plan={"needs_rolls": True, "roll_requests": []},
                ledger_lines=["attack_alice: alice attack_roll rolled 18"],
            )
        )
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(
            _router_out(ends_beat_reason="ruleset_resolution")
        )

        response = await orch.continue_cat_ii_after_roll(
            session_id="s",
            event_id="cmb_1",
            actor_id="alice",
        )

        assert response.beat_ended_reason == "ruleset_resolution"
        assert response.per_player_renders["alice"] == "POV_RENDER"
        assert response.turn_index == 5
        assert FakeDispatcher.route_calls[0]["event_id"] == "cmb_1"
        assert ckpt.session.active_act_slots == {}
        assert len(ckpt.canonical_events) == 1
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_cancelled_combat_roll_clears_slot_without_dispatch(
        self,
        patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="cat_ii_roll",
            cat_ii_event_id="cmb_cancelled",
        )
        ckpt.session.cat_ii_roll_transactions.append(
            CatIIRollTransaction(
                transaction_id="rolltxn_cancelled",
                event_id="cmb_cancelled",
                source="combat",
                actor_id="alice",
                status="cancelled",
            )
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.continue_cat_ii_after_roll(
            session_id="s",
            event_id="cmb_cancelled",
            actor_id="alice",
        )

        assert response.beat_ended_reason == "cat_ii_stale"
        assert response.output_text == "That combat roll is no longer active."
        assert ckpt.session.active_act_slots == {}
        assert FakeDispatcher.route_calls == []
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_submit_cancelled_combat_roll_returns_stale_without_dispatch(
        self,
        patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="cat_ii_roll",
            cat_ii_event_id="cmb_cancelled",
        )
        ckpt.session.cat_ii_roll_transactions.append(
            CatIIRollTransaction(
                transaction_id="rolltxn_cancelled",
                event_id="cmb_cancelled",
                source="combat",
                actor_id="alice",
                status="cancelled",
            )
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.submit_cat_ii_roll(
            session_id="s",
            event_id="cmb_cancelled",
            roll_id="attack_alice",
            actor_id="alice",
            user_id="u1",
        )

        assert response.beat_ended_reason == "cat_ii_stale"
        assert response.output_text == "That combat roll is no longer active."
        assert ckpt.session.active_act_slots == {}
        assert FakeDispatcher.route_calls == []
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_submit_non_pending_cat_ii_roll_returns_friendly_stale(
        self,
        patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.open_cat_ii_events.append(OpenCatIIEvent(
            event_id="evt_open",
            initiator_id="bob",
            initiator_intention="Bob trips Alice.",
            required_responders=["alice"],
            collected_intentions={"alice": "Alice keeps her feet."},
        ))
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.submit_cat_ii_roll(
            session_id="s",
            event_id="evt_open",
            roll_id="roll_alice",
            actor_id="alice",
            user_id="u1",
        )

        assert response.beat_ended_reason == "cat_ii_stale"
        assert response.output_text == (
            "That roll is no longer pending for your character."
        )
        assert "alice" not in response.output_text
        assert "roll_alice" not in response.output_text
        assert FakeDispatcher.route_calls == []
        assert mgr.save.call_count == 0


class TestCombatTurnGating:
    @pytest.mark.asyncio
    async def test_dnd_combat_start_signal_starts_initiative_without_advancing(
        self, patched_orchestrator, monkeypatch,
    ):
        values = iter([19, 0, 0])
        monkeypatch.setattr(
            dice.d20.expression.random,
            "randrange",
            lambda _: next(values),
        )
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice", "bob"],
            facts=["Alice commits to an attack against Bob."],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I attack Bob",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_started"
        assert ckpt.session.active_combat is not None
        assert current_combatant(ckpt.session).character_id == "alice"
        assert ckpt.session.open_cat_ii_events == []
        assert ckpt.session.active_act_slots == {}
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_non_current_human_combatant_is_rejected_without_save(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = {
            "status": "active",
            "round_number": 1,
            "turn_index": 0,
            "combatants": [
                {
                    "character_id": "alice",
                    "name": "Alice",
                    "player_controlled": True,
                },
                {
                    "character_id": "bob",
                    "name": "Bob",
                    "player_controlled": True,
                },
            ],
        }
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I rush in anyway",
            acting_character_id="bob",
        ))

        assert response.beat_ended_reason == "combat_turn_rejected"
        assert "Alice" in response.output_text
        assert "initiative turn" in response.output_text
        assert response.per_player_renders == {}
        assert mgr.save.call_count == 0
        assert FakeDispatcher.route_calls == []

    @pytest.mark.asyncio
    async def test_bound_human_outside_combat_can_act_while_combat_exists(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(ends_beat=True))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I search the library",
            acting_character_id="bob",
        ))

        assert response.beat_ended_reason == "directed_at_player"
        assert FakeDispatcher.route_calls[0]["actor_id"] == "bob"
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_npc_initiative_turn_runs_agent_then_advances_to_human(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            actor_id = kw["actor_id"]
            return BeatResult(
                renders={"alice": f"{actor_id} acts."},
                events_closed=0,
                ended_reason="directed_at_player",
                transcript_entries={},
                event_actor_ids=[actor_id],
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        FakeDispatcher.queue_agent("Pip attacks.")
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I strike the goblin",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "directed_at_player"
        assert ckpt.session.active_combat.turn_index == 2
        assert ckpt.session.active_combat.round_number == 1
        assert response.output_text == "alice acts.\n\npip acts."
        assert FakeDispatcher.agent_calls[0]["character_id"] == "pip"
        assert any(
            "Initiative advanced to Pip" in line
            for line in ckpt.session.active_combat.audit_lines
        )
        assert any(
            "Initiative advanced to Bob" in line
            for line in ckpt.session.active_combat.audit_lines
        )
        assert mgr.save.call_count == 2

    @pytest.mark.asyncio
    async def test_npc_automation_agent_failure_skips_to_next_turn(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            return BeatResult(
                renders={"alice": "Alice acts."},
                events_closed=0,
                ended_reason="directed_at_player",
                transcript_entries={},
                event_actor_ids=[kw["actor_id"]],
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I strike",
            acting_character_id="alice",
        ))

        assert response.output_text == "Alice acts."
        assert ckpt.session.active_combat.turn_index == 2
        assert any(
            "failed before an intention was produced" in line
            for line in ckpt.session.active_combat.audit_lines
        )
        assert mgr.save.call_count == 2

    @pytest.mark.asyncio
    async def test_npc_automation_run_beat_failure_aborts_and_advances(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            if kw["actor_id"] == "pip":
                from app.engine.turn_loop import claim_initiator_slot

                claim_initiator_slot(kw["ckpt"], "pip")
                raise RuntimeError("simulated route outage")
            return BeatResult(
                renders={"alice": "Alice acts."},
                events_closed=0,
                ended_reason="directed_at_player",
                transcript_entries={},
                event_actor_ids=[kw["actor_id"]],
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        FakeDispatcher.queue_agent("Pip attacks.")
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I strike",
            acting_character_id="alice",
        ))

        assert response.output_text == "Alice acts."
        assert ckpt.session.active_combat.turn_index == 2
        assert ckpt.session.active_act_slots == {}
        assert any(
            "failed during resolution" in line
            for line in ckpt.session.active_combat.audit_lines
        )
        assert mgr.save.call_count == 2

    @pytest.mark.asyncio
    async def test_pending_cat_ii_render_does_not_advance_combat(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            return BeatResult(
                renders={"alice": "The exchange hangs unresolved."},
                events_closed=0,
                ended_reason="cat_ii_pending",
                transcript_entries={},
                event_actor_ids=["alice"],
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = {
            "status": "active",
            "round_number": 1,
            "turn_index": 0,
            "combatants": [
                {
                    "character_id": "alice",
                    "name": "Alice",
                    "player_controlled": True,
                },
                {
                    "character_id": "bob",
                    "name": "Bob",
                    "player_controlled": True,
                },
            ],
        }
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I shove Bob",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "cat_ii_pending"
        assert ckpt.session.active_combat["turn_index"] == 0
        assert "audit_lines" not in ckpt.session.active_combat
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_reaction_prompt_delays_initiative_advance(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            return BeatResult(
                renders={"alice": "Alice moves.", "bob": "Bob can react."},
                events_closed=1,
                ended_reason="combat_reaction_pending",
                transcript_entries={},
                event_actor_ids=["alice"],
                reaction_prompts={"bob": "evt_react"},
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I move away",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_reaction_pending"
        assert response.reaction_prompts == {"bob": "evt_react"}
        assert ckpt.session.active_combat.turn_index == 0
        assert ckpt.session.active_combat.pending_advance_actor_id == "alice"
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_ruleset_resolution_advances_initiative(
        self,
        patched_orchestrator,
        monkeypatch,
    ):
        async def fake_run_beat(**kw):
            return BeatResult(
                renders={"alice": "Alice resolves a combat action."},
                events_closed=1,
                ended_reason="ruleset_resolution",
                transcript_entries={},
                event_actor_ids=["alice"],
                reaction_prompts={},
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I attack",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "ruleset_resolution"
        assert ckpt.session.active_combat.turn_index == 1
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_non_current_reaction_slot_bypasses_combat_turn_rejection(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                    reaction_available=True,
                ),
            ],
            pending_advance_actor_id="alice",
        )
        ckpt.session.active_act_slots["bob"] = SlotEntry(
            reason="combat_reaction",
            trigger_event_id="evt_react",
        )
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(EventRouterOutput(
            event_id="",
            decision_rationale="reaction act",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[ObservableFact.all("Bob reacts.")],
            ),
            observers=[
                ObserverEntry(
                    character_id="bob",
                    observation_level="d",
                    response_priority=3,
                )
            ],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="directed_at_player",
            spawn=[],
            dormant=[],
            cull=[],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I make an opportunity attack",
            acting_character_id="bob",
        ))

        assert response.beat_ended_reason == "directed_at_player"
        assert FakeDispatcher.route_calls[0]["actor_id"] == "bob"
        assert "Combat reaction" in FakeDispatcher.route_calls[0]["intention"]
        assert ckpt.session.active_act_slots == {}
        assert ckpt.session.active_combat.turn_index == 1
        assert ckpt.session.active_combat.pending_advance_actor_id == ""
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_defer_clears_blocked_combat_start_without_llm(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
            ],
        )
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="combat_blocked",
            trigger_event_id="evt_blocked",
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="(defer)",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_start_blocked_deferred"
        assert "dropped" in response.output_text
        assert ckpt.session.active_act_slots == {}
        assert FakeDispatcher.route_calls == []
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_normal_act_abandons_blocked_combat_start(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
            ],
        )
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="combat_blocked",
            trigger_event_id="evt_blocked",
        )
        FakeDispatcher.queue_route(_router_out(
            ends_beat=True,
            facts=[ObservableFact.only("Alice steps back.", ["alice"])],
        ))
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I step back and watch.",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "directed_at_player"
        assert ckpt.session.active_act_slots == {}
        assert FakeDispatcher.route_calls[0]["actor_id"] == "alice"
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_defer_combat_reaction_clears_slot_without_llm_and_advances(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                    reaction_available=False,
                ),
            ],
            pending_advance_actor_id="alice",
        )
        ckpt.session.active_act_slots["bob"] = SlotEntry(
            reason="combat_reaction",
            trigger_event_id="evt_react",
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.defer_combat_reaction(
            session_id="s",
            character_id="bob",
            event_id="evt_react",
        )

        assert response.beat_ended_reason == "combat_reaction_deferred"
        assert "Initiative advances to **Bob**" in response.output_text
        assert ckpt.session.active_act_slots == {}
        assert ckpt.session.active_combat.turn_index == 1
        assert ckpt.session.active_combat.combatants[1].reaction_available is True
        assert FakeDispatcher.route_calls == []
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_cat_ii_resolution_resumes_pending_combat_advance(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                    reaction_available=False,
                ),
            ],
            pending_advance_actor_id="alice",
        )
        from app.engine.turn_loop import open_cat_ii, pin_cat_ii_responder

        opened = open_cat_ii(
            ckpt,
            initiator_id="alice",
            initiator_intention="Alice's interrupted action",
            required_responders=["bob"],
        )
        pin_cat_ii_responder(ckpt, "bob", opened.event_id)
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(EventRouterOutput(
            event_id="",
            decision_rationale="cat ii closes after reaction",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[ObservableFact.all("The exchange resolves.")],
            ),
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    response_priority=3,
                ),
                ObserverEntry(
                    character_id="bob",
                    observation_level="d",
                    response_priority=3,
                ),
            ],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="cat_ii_resolution",
            spawn=[],
            dormant=[],
            cull=[],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I block",
            acting_character_id="bob",
        ))

        assert response.beat_ended_reason == "cat_ii_resolution"
        assert ckpt.session.active_act_slots == {}
        assert ckpt.session.active_combat.turn_index == 1
        assert ckpt.session.active_combat.pending_advance_actor_id == ""
        assert ckpt.session.active_combat.combatants[1].reaction_available is True
        assert mgr.save.call_count == 1


class TestCatIIPending:
    @pytest.mark.asyncio
    async def test_cat_ii_against_human_pauses_and_persists_open_event(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(
            requires_responders=True,
            required_responders=["bob"],
            ends_beat=False,
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I attack Bob",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "cat_ii_pending"
        assert "alice" in response.per_player_renders
        assert "bob" in response.per_player_renders
        saved = mgr.save.call_args[0][0]
        assert len(saved.session.open_cat_ii_events) == 1
        assert saved.session.active_act_slots["bob"].reason == "cat_ii_responder"


class TestResolveCatII:
    @pytest.mark.asyncio
    async def test_ready_event_closes_and_returns_render(
        self, patched_orchestrator,
    ):
        from app.engine.turn_loop import open_cat_ii

        ckpt = _ckpt(bindings={"alice": "u1"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="pip",
            initiator_intention="pip swings at alice",
            required_responders=["alice"],
        )
        evt.collected_intentions["alice"] = "[AFK-swept: no player intention]"
        evt.swept_responders.append("alice")
        orch, mgr = patched_orchestrator(ckpt)

        resolution = _router_out(
            ends_beat=True,
            facts=[
                ObservableFact.all("Alice keeps her guard up."),
                ObservableFact.all("Pip ends the exchange checked."),
            ],
        )
        resolution.observers = [
            ObserverEntry(
                character_id="alice",
                observation_level="d",
                response_priority=5,
            ),
            ObserverEntry(
                character_id="pip",
                observation_level="d",
                response_priority=5,
            ),
        ]
        FakeDispatcher.queue_route(resolution)
        FakeDispatcher.queue_agent("Pip releases the angle.")
        followup = _router_out(ends_beat=True, agent_picks=[])
        followup.observers = [
            ObserverEntry(
                character_id="alice",
                observation_level="d",
                response_priority=5,
            ),
            ObserverEntry(
                character_id="pip",
                observation_level="d",
                response_priority=1,
            ),
        ]
        FakeDispatcher.queue_route(followup)

        response = await orch.resolve_cat_ii("s", evt.event_id)

        assert response.beat_ended_reason == "directed_at_player"
        assert response.per_player_renders["alice"] == "POV_RENDER"
        saved = mgr.save.call_args[0][0]
        assert all(e.event_id != evt.event_id for e in saved.session.open_cat_ii_events)
        assert len(saved.canonical_events) == 2
        assert FakeDispatcher.agent_calls[0]["character_id"] == "pip"
        assert FakeDispatcher.route_calls[1]["actor_id"] == "pip"

    @pytest.mark.asyncio
    async def test_stale_event_returns_noop(self, patched_orchestrator):
        ckpt = _ckpt(bindings={"alice": "u1"})
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.resolve_cat_ii("s", "missing_evt")

        assert response.beat_ended_reason == "cat_ii_stale"
        assert response.per_player_renders == {}
        assert mgr.save.call_count == 0
