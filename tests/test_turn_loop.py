"""Turn-loop state-machine tests.

These exercise the orchestrator-independent pieces: session-wide act slots,
Cat II collection, beat cascade endings, observer-driven broadcast, and
error-message formatting. A fake dispatcher stands in for LLM calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.engine.turn_loop import (
    SessionLockManager,
    SlotConflict,
    abort_beat,
    append_to_render_buffer,
    broadcast_event,
    check_act_slot,
    claim_initiator_slot,
    format_slot_rejection,
    open_cat_ii,
    pin_cat_ii_responder,
    pin_combat_reaction,
    purge_character_state,
    release_beat_slots,
    run_beat,
    sweep_stale_cat_ii_pins,
)
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    WorldAdjudication,
)
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import (
    DndCombatantState,
    DndCombatState,
    RenderBufferEntry,
    SessionState,
    SlotEntry,
    WorldState,
)


def _ckpt(bindings: dict[str, str] | None = None) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id="s",
            character_bindings=bindings or {},
        ),
        world_state=WorldState(),
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
                public_sheet=PublicSheet(role="npc"),
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
        ends_beat_reason="directed_at_player" if ends_beat else "",
        spawn=[],
        dormant=[],
        cull=[],
    )


class FakeDispatcher:
    def __init__(self):
        self.route_calls: list[dict] = []
        self.continuation_calls: list[dict] = []
        self.agent_calls: list[dict] = []
        self.narrator_calls: list[dict] = []
        self.harvest_calls: list[dict] = []
        self.combat_calls: list[dict] = []
        self._route_responses: list[EventRouterOutput] = []
        self._combat_responses: list[EventRouterOutput] = []
        self._agent_responses: list[str] = []
        self._harvest_responses: list[list[str]] = []
        self._narrator_response = "RENDER"

    def queue_route(self, response: EventRouterOutput) -> None:
        self._route_responses.append(response)

    def queue_combat(self, response: EventRouterOutput) -> None:
        self._combat_responses.append(response)

    def queue_agent(self, intention: str) -> None:
        self._agent_responses.append(intention)

    def queue_harvest(self, fragments: list[str]) -> None:
        self._harvest_responses.append(fragments)

    async def route_intention(self, **kw) -> EventRouterOutput:
        self.route_calls.append(kw)
        return self._route_responses.pop(0)

    async def route_continuation(self, **kw) -> EventRouterOutput:
        self.continuation_calls.append(kw)
        return self._route_responses.pop(0)

    async def route_combat_action(self, **kw) -> EventRouterOutput:
        self.combat_calls.append(kw)
        return self._combat_responses.pop(0)

    async def continue_combat_transaction(self, **kw) -> EventRouterOutput:
        self.combat_calls.append(kw)
        return self._combat_responses.pop(0)

    async def agent_intend(self, **kw) -> str:
        self.agent_calls.append(kw)
        return self._agent_responses.pop(0)

    async def harvest_perceptions(self, **kw) -> list[str]:
        self.harvest_calls.append(kw)
        if self._harvest_responses:
            return self._harvest_responses.pop(0)
        return ["" for _ in kw.get("character_ids", [])]

    async def narrator_compose(self, **kw):
        self.narrator_calls.append(kw)
        envelope = NarratorFinalOutput(final_text=self._narrator_response)
        entry = TranscriptEntry(
            user=kw.get("user_input", ""),
            assistant=self._narrator_response,
        )
        return envelope, entry


class TestCheckActSlot:
    def test_free_when_empty(self):
        assert check_act_slot(_ckpt({"alice": "1"}), "alice").conflict == (
            SlotConflict.FREE
        )

    def test_initiator_held_by_other(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        claim_initiator_slot(ckpt, "alice")
        check = check_act_slot(ckpt, "bob")
        assert check.conflict == SlotConflict.INITIATOR_HELD
        assert check.holder_id == "alice"

    def test_self_busy_when_same_user_double_acts(self):
        ckpt = _ckpt({"alice": "1"})
        claim_initiator_slot(ckpt, "alice")
        assert check_act_slot(ckpt, "alice").conflict == SlotConflict.SELF_BUSY

    def test_cat_ii_responder_acceptance_path(self):
        ckpt = _ckpt({"alice": "1"})
        pin_cat_ii_responder(ckpt, "alice", "evt_xyz")
        check = check_act_slot(ckpt, "alice")
        assert check.conflict == SlotConflict.CAT_II_SELF_RESPONDER
        assert check.cat_ii_event_id == "evt_xyz"

    def test_cat_ii_other_held_rejects_bystander(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        pin_cat_ii_responder(ckpt, "alice", "evt_xyz")
        check = check_act_slot(ckpt, "bob")
        assert check.conflict == SlotConflict.CAT_II_OTHER_HELD
        assert check.holder_id == "alice"

    def test_cat_ii_roll_pin_rejects_act_until_roll_ui(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="cat_ii_roll",
            cat_ii_event_id="evt_xyz",
        )
        check = check_act_slot(ckpt, "alice")
        assert check.conflict == SlotConflict.CAT_II_SELF_ROLL
        assert check.cat_ii_event_id == "evt_xyz"

    def test_combat_reaction_slot_accepts_holder_act(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.active_combat = DndCombatState(
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                )
            ]
        )
        assert pin_combat_reaction(ckpt, "alice", "evt_react") is True

        check = check_act_slot(ckpt, "alice")

        assert check.conflict == SlotConflict.COMBAT_REACTION_SELF
        assert check.trigger_event_id == "evt_react"

    def test_combat_reaction_slot_blocks_bystander(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        ckpt.session.active_combat = DndCombatState(
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
            ]
        )
        pin_combat_reaction(ckpt, "alice", "evt_react")

        check = check_act_slot(ckpt, "bob")

        assert check.conflict == SlotConflict.COMBAT_REACTION_OTHER_HELD
        assert check.holder_id == "alice"


class TestBeatCascade:
    def test_single_cat_i_ends_beat_and_renders(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="look around",
        ))

        assert result.events_closed == 1
        assert result.ended_reason == "directed_at_player"
        assert "alice" in result.renders
        assert ckpt.session.active_act_slots == {}

    def test_combat_high_priority_observer_gets_reaction_prompt(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
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
        fake = FakeDispatcher()
        fake.queue_route(EventRouterOutput(
            event_id="",
            decision_rationale="reaction prompt",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[ObservableFact.all("Alice rushes past Bob.")],
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
                    response_priority=5,
                ),
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

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I rush past Bob",
        ))

        event_id = ckpt.canonical_events[0].event_id
        assert result.ended_reason == "combat_reaction_pending"
        assert result.reaction_prompts == {"bob": event_id}
        assert ckpt.session.active_act_slots["bob"].reason == "combat_reaction"
        assert ckpt.session.active_act_slots["bob"].trigger_event_id == event_id
        assert "bob" in result.renders

    def test_combat_low_priority_observer_does_not_get_reaction_prompt(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
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
        fake = FakeDispatcher()
        fake.queue_route(EventRouterOutput(
            event_id="",
            decision_rationale="no reaction prompt",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[ObservableFact.all("Alice shifts her stance.")],
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
                    response_priority=4,
                ),
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

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I shift my stance",
        ))

        assert result.ended_reason == "directed_at_player"
        assert result.reaction_prompts == {}
        assert "bob" not in ckpt.session.active_act_slots
        assert ckpt.session.render_buffers.get("bob") == []

    def test_dnd_combat_reaction_uses_combat_resolver_with_trigger_context(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
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
        trigger = EventRouterOutput(
            event_id="evt_trigger",
            decision_rationale="trigger",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[
                    ObservableFact.all("Alice exposes herself as she moves.")
                ],
            ),
            observers=[
                ObserverEntry(
                    character_id="bob",
                    observation_level="d",
                    response_priority=5,
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
        )
        ckpt.canonical_events.append(trigger)
        ckpt.session.active_act_slots["bob"] = SlotEntry(
            reason="combat_reaction",
            trigger_event_id="evt_trigger",
        )
        fake = FakeDispatcher()
        combat_out = _router_out(ends_beat=True)
        combat_out.ends_beat_reason = "ruleset_resolution"
        fake.queue_combat(combat_out)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="bob",
            intention="I make an opportunity attack",
            combat_reaction_event_id="evt_trigger",
        ))

        assert fake.route_calls == []
        routed_intention = fake.combat_calls[0]["intention"]
        assert "Combat reaction to this event" in routed_intention
        assert "Alice exposes herself" in routed_intention
        assert "I make an opportunity attack" in routed_intention
        assert result.ended_reason == "ruleset_resolution"
        assert result.reaction_prompts == {}

    def test_query_response_harvests_private_perception_targets(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(EventRouterOutput(
            event_id="",
            decision_rationale="private query answer",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[
                    ObservableFact.only("You focus on Pip.", ["alice"]),
                ],
            ),
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    response_priority=1,
                ),
            ],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=["pip"],
            ends_beat=True,
            ends_beat_reason="query_response",
            spawn=[],
            dormant=[],
            cull=[],
        ))
        fake.queue_harvest(["Pip wears a red coat."])

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="(query: what does Pip look like?)",
        ))

        assert result.events_closed == 1
        assert result.ended_reason == "query_response"
        assert fake.harvest_calls[0]["character_ids"] == ["pip"]
        facts = ckpt.canonical_events[0].canonical_event.observable_facts
        assert [fact.text for fact in facts] == [
            "You focus on Pip.",
            "[loadout — Pip] Pip wears a red coat.",
        ]
        assert "alice" in result.renders

    def test_cat_i_cascades_through_agent_pick(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(agent_picks=["pip"], ends_beat=False))
        fake.queue_agent("Pip polishes the bell")
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert result.events_closed == 2
        assert fake.agent_calls[0]["character_id"] == "pip"

    def test_false_endbeat_with_no_picks_routes_continuation(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(ends_beat=False))
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert result.ended_reason == "directed_at_player"
        assert len(fake.continuation_calls) == 1
        assert result.events_closed == 2

    def test_repeated_false_endbeat_without_picks_errors(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(ends_beat=False))
        fake.queue_route(_router_out(ends_beat=False))

        with pytest.raises(RuntimeError, match="without a dispatchable"):
            asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="wait",
            ))


class TestCatIIBeat:
    def test_cat_ii_with_agent_responder_resolves_inline(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip"],
            ends_beat=False,
        ))
        fake.queue_agent("Pip dodges")
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I attack Pip",
        ))

        assert result.ended_reason == "cat_ii_resolution"
        assert result.events_closed == 2
        assert ckpt.canonical_events[0].ends_beat_reason == "cat_ii_open"
        assert ckpt.session.open_cat_ii_events == []

    def test_cat_ii_inline_overrun_logs_cap_telemetry(self, caplog):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.max_events_per_beat = 1
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip"],
            ends_beat=False,
        ))
        fake.queue_agent("Pip dodges")
        fake.queue_route(_router_out(ends_beat=True))

        with caplog.at_level("WARNING", logger="app.engine.turn_loop"):
            result = asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="I attack Pip",
            ))

        assert result.events_closed == 2
        assert result.ended_reason == "cat_ii_resolution"
        message = caplog.messages[-1]
        assert "Beat cap overrun" in message
        assert "configured_cap=1" in message
        assert "events_rendered=2" in message
        assert "ended_reason=cat_ii_resolution" in message
        assert "cat_ii_open_likely=True" in message
        assert "cat_ii_resolution_likely=True" in message
        assert "cat_ii_followup_likely=False" in message

    def test_cat_ii_with_human_responder_pauses_and_renders_partial(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["bob"],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I attack Bob",
        ))

        assert result.ended_reason == "cat_ii_pending"
        assert "alice" in result.renders
        assert "bob" in result.renders
        assert ckpt.session.active_act_slots["bob"].reason == "cat_ii_responder"
        open_evt = ckpt.session.open_cat_ii_events[0]
        assert open_evt.opening_observer_ids == ["alice", "bob"]
        assert open_evt.opening_observable_facts == ["Something happens."]
        assert fake.narrator_calls
        assert all(c.get("partial_mode_override") is True for c in fake.narrator_calls)

    def test_active_combat_suppresses_generic_cat_ii_open(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
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
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["bob"],
            ends_beat=False,
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I attack Bob",
        ))

        assert result.ended_reason == "ruleset_cat_ii_suppressed"
        assert ckpt.session.open_cat_ii_events == []
        assert ckpt.session.active_act_slots == {}
        assert result.reaction_prompts == {}
        assert ckpt.canonical_events[0].ends_beat_reason == (
            "ruleset_cat_ii_suppressed"
        )
        assert fake.agent_calls == []

    def test_dnd_active_combat_uses_combat_resolver_instead_of_generic_router(
        self,
    ):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
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
        fake = FakeDispatcher()
        combat_out = _router_out(
            requires_responders=False,
            facts=["Alice's strike resolves against Bob."],
        )
        combat_out.ends_beat_reason = "ruleset_resolution"
        fake.queue_combat(combat_out)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I attack Bob",
        ))

        assert result.ended_reason == "ruleset_resolution"
        assert fake.route_calls == []
        assert fake.combat_calls[0]["actor_id"] == "alice"
        assert ckpt.canonical_events[0].ends_beat_reason == (
            "ruleset_resolution"
        )

    def test_high_priority_combat_observer_gets_render_on_npc_turn(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.session.active_combat = DndCombatState(
            turn_index=2,
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
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
            ],
        )
        fake = FakeDispatcher()
        combat_out = EventRouterOutput(
            event_id="",
            decision_rationale="Pip hits Bob.",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[ObservableFact.all("Pip cuts Bob.")],
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
                    response_priority=5,
                ),
            ],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="ruleset_resolution",
            spawn=[],
            dormant=[],
            cull=[],
        )
        fake.queue_combat(combat_out)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="pip",
            intention="I attack Bob",
        ))

        assert result.renders == {"bob": "RENDER"}
        assert ckpt.session.render_buffers.get("alice") == []

    def test_noncombat_actor_can_open_cat_ii_while_combat_exists(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
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
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["alice"],
            ends_beat=False,
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="pip",
            intention="I shove Alice",
        ))

        assert result.ended_reason == "cat_ii_pending"
        assert len(ckpt.session.open_cat_ii_events) == 1
        assert ckpt.session.active_act_slots["alice"].reason == (
            "cat_ii_responder"
        )

    def test_cat_ii_responder_intention_closes_event(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="alice",
            initiator_intention="I attack Bob",
            required_responders=["bob"],
        )
        pin_cat_ii_responder(ckpt, "bob", evt.event_id)

        fake = FakeDispatcher()
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="bob",
            intention="I block",
            cat_ii_event_id=evt.event_id,
        ))

        assert result.ended_reason == "cat_ii_resolution"
        assert ckpt.session.open_cat_ii_events == []
        assert ckpt.session.active_act_slots == {}

    def test_cat_ii_multi_responder_pauses_until_all_intentions_in(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="pip",
            initiator_intention="Pip throws a punch",
            required_responders=["alice", "bob"],
        )
        pin_cat_ii_responder(ckpt, "alice", evt.event_id)
        pin_cat_ii_responder(ckpt, "bob", evt.event_id)

        fake = FakeDispatcher()
        result1 = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="bob",
            intention="I lunge to block",
            cat_ii_event_id=evt.event_id,
        ))
        assert result1.ended_reason == "cat_ii_pending"
        assert "bob" not in ckpt.session.active_act_slots
        assert "alice" in ckpt.session.active_act_slots

        fake.queue_route(_router_out(ends_beat=True))
        fake.queue_agent("")
        result2 = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I duck",
            cat_ii_event_id=evt.event_id,
        ))
        assert result2.ended_reason == "cat_ii_resolution"
        assert ckpt.session.open_cat_ii_events == []


class TestObservationHarvest:
    def test_harvest_appends_loadout_fragments(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.characters.append(CharacterRecord(
            character_id="vex",
            name="Vex",
            public_sheet=PublicSheet(role="npc"),
            location="gatehouse",
        ))
        fake = FakeDispatcher()
        out = _router_out(ends_beat=True, agent_picks=["pip", "vex"])
        out.ends_beat_reason = "observation_harvest"
        fake.queue_route(out)
        fake.queue_harvest([
            "Pip in patched leathers.",
            "Vex in midnight silk.",
        ])

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I look them over",
        ))

        assert result.ended_reason == "observation_harvest"
        assert fake.harvest_calls[0]["character_ids"] == ["pip", "vex"]
        facts = ckpt.canonical_events[-1].canonical_event.observable_facts
        assert any("[loadout" in str(f) and "Pip" in str(f) for f in facts)
        assert any("[loadout" in str(f) and "Vex" in str(f) for f in facts)
        assert fake.agent_calls == []

    def test_harvest_drops_human_picks(self):
        ckpt = _ckpt({"alice": "1"})
        out = _router_out(ends_beat=True, agent_picks=["alice"])
        out.ends_beat_reason = "observation_harvest"
        fake = FakeDispatcher()
        fake.queue_route(out)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I look at myself",
        ))

        assert result.ended_reason == "observation_harvest"
        assert fake.harvest_calls == []


class TestBroadcastEvent:
    def _event(
        self,
        observer_ids: list[str],
        facts: list[ObservableFact | str] | None = None,
    ) -> EventRouterOutput:
        return EventRouterOutput(
            event_id="",
            decision_rationale="test fixture",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=facts if facts is not None else [
                    ObservableFact.all("Alice sets down a glass.")
                ],
            ),
            observers=[
                ObserverEntry(
                    character_id=cid,
                    observation_level="d",
                    response_priority=3,
                )
                for cid in observer_ids
            ],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="directed_at_player",
            spawn=[],
            dormant=[],
            cull=[],
        )

    def test_same_location_without_observer_entry_gets_nothing(self):
        ckpt = _ckpt()
        event = self._event(observer_ids=["alice"])
        broadcast_event(ckpt, event, actor_id="alice")
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == []

    def test_all_observers_fact_with_empty_observers_delivers_to_no_one(self):
        ckpt = _ckpt({"alice": "1"})
        event = self._event(
            observer_ids=[],
            facts=[ObservableFact.all("A bell rings across the estate.")],
        )

        visible = broadcast_event(ckpt, event, actor_id="pip")

        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        bob = next(c for c in ckpt.characters if c.character_id == "bob")
        assert visible == []
        assert ckpt.session.render_buffers == {}
        assert pip.pending_observations == []
        assert alice.pending_observations == []
        assert bob.pending_observations == []

    def test_npc_observer_gets_visible_facts(self):
        ckpt = _ckpt()
        event = self._event(observer_ids=["pip"])
        broadcast_event(ckpt, event, actor_id="alice")
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == ["Alice sets down a glass."]

    def test_actor_excluded_from_own_inbox(self):
        ckpt = _ckpt()
        event = self._event(observer_ids=["pip"])
        broadcast_event(ckpt, event, actor_id="pip")
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == []

    def test_human_observer_gets_render_buffer_not_inbox(self):
        ckpt = _ckpt({"alice": "1"})
        event = self._event(observer_ids=["alice"])
        visible = broadcast_event(ckpt, event, actor_id="pip")
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        assert visible == ["alice"]
        assert alice.pending_observations == []
        assert ckpt.session.render_buffers["alice"][0].event_id == event.event_id

    def test_scoped_fact_only_reaches_visible_recipient(self):
        ckpt = _ckpt()
        ckpt.characters.extend([
            CharacterRecord(
                character_id="ashara",
                name="Ashara",
                public_sheet=PublicSheet(role="heir"),
                location="gatehouse",
            ),
            CharacterRecord(
                character_id="aldric",
                name="Aldric",
                public_sheet=PublicSheet(role="heir"),
                location="gatehouse",
            ),
        ])
        event = self._event(
            observer_ids=["ashara", "aldric"],
            facts=[
                ObservableFact.only(
                    "Dan's foot touches Ashara's boot under the table.",
                    ["ashara"],
                ),
                ObservableFact.all(
                    "Dan asks Thessaly whether she knows curses.",
                ),
            ],
        )

        broadcast_event(ckpt, event, actor_id="alice")

        ashara = next(c for c in ckpt.characters if c.character_id == "ashara")
        aldric = next(c for c in ckpt.characters if c.character_id == "aldric")
        assert "foot touches" in ashara.pending_observations[0]
        assert "knows curses" in ashara.pending_observations[0]
        assert aldric.pending_observations == [
            "Dan asks Thessaly whether she knows curses."
        ]

    def test_empty_observable_facts_means_no_push(self):
        ckpt = _ckpt()
        event = self._event(observer_ids=["pip"], facts=[])
        broadcast_event(ckpt, event, actor_id="alice")
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == []


class TestBookkeeping:
    def test_release_beat_slots_preserves_open_cat_ii_pins(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="pip",
            initiator_intention="punch",
            required_responders=["alice"],
        )
        pin_cat_ii_responder(ckpt, "alice", evt.event_id)
        claim_initiator_slot(ckpt, "bob")

        release_beat_slots(ckpt)

        assert "bob" not in ckpt.session.active_act_slots
        assert ckpt.session.active_act_slots["alice"].reason == (
            "cat_ii_responder"
        )

    def test_abort_clears_all_pins_events_and_buffers(self):
        ckpt = _ckpt({"alice": "1"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="pip",
            initiator_intention="punch",
            required_responders=["alice"],
        )
        pin_cat_ii_responder(ckpt, "alice", evt.event_id)
        ckpt.session.render_buffers["alice"] = [
            RenderBufferEntry(event_id="evt_x", observation_level="direct"),
        ]

        dropped = abort_beat(ckpt)

        assert dropped == 1
        assert ckpt.session.active_act_slots == {}
        assert ckpt.session.open_cat_ii_events == []
        assert ckpt.session.render_buffers["alice"] == []

    def test_purge_drops_character_pin_and_required_responder(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="pip",
            initiator_intention="punch",
            required_responders=["alice", "bob"],
        )
        pin_cat_ii_responder(ckpt, "alice", evt.event_id)
        pin_cat_ii_responder(ckpt, "bob", evt.event_id)

        purge_character_state(ckpt, "bob")

        assert "bob" not in ckpt.session.active_act_slots
        assert "alice" in ckpt.session.active_act_slots
        assert "bob" not in ckpt.session.open_cat_ii_events[0].required_responders

    def test_sweep_fills_stale_pin_as_afk_swept(self):
        from datetime import datetime, timedelta, timezone

        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.cat_ii_human_timeout_seconds = 1
        evt = open_cat_ii(
            ckpt,
            initiator_id="pip",
            initiator_intention="punch",
            required_responders=["alice"],
        )
        evt.opened_at = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).isoformat()

        ids = sweep_stale_cat_ii_pins(ckpt)

        assert ids == [evt.event_id]
        assert "alice" in evt.swept_responders
        assert "AFK-swept" in evt.collected_intentions["alice"]


class TestSessionLockManager:
    def test_same_session_returns_same_lock(self):
        mgr = SessionLockManager()

        async def run():
            a = await mgr.get("s")
            b = await mgr.get("s")
            return a is b

        assert asyncio.run(run())

    def test_different_sessions_distinct_locks(self):
        mgr = SessionLockManager()

        async def run():
            a = await mgr.get("s1")
            b = await mgr.get("s2")
            return a is not b

        assert asyncio.run(run())


class TestFilterPicksForDispatch:
    def test_remote_npc_pick_passes_through(self):
        from app.engine.turn_loop import _filter_picks_for_dispatch

        ckpt = _ckpt({"alice": "1"})
        watcher = CharacterRecord(
            character_id="watcher",
            name="Watcher",
            public_sheet=PublicSheet(role="operator"),
            location="control_room",
        )
        ckpt.characters.append(watcher)

        assert _filter_picks_for_dispatch(ckpt, ["watcher"]) == ["watcher"]

    def test_human_pick_filtered(self):
        from app.engine.turn_loop import _filter_picks_for_dispatch

        ckpt = _ckpt({"alice": "1", "bob": "2"})
        assert _filter_picks_for_dispatch(ckpt, ["alice", "bob", "pip"]) == [
            "pip"
        ]


class TestSchemaValidators:
    def test_requires_responders_without_list_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _router_out(requires_responders=True, required_responders=[])

    def test_picks_not_in_observers_dropped(self):
        out = EventRouterOutput(
            event_id="",
            decision_rationale="test fixture",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[],
            ),
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    response_priority=3,
                )
            ],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=["alice", "pip"],
            ends_beat=True,
            ends_beat_reason="directed_at_player",
            spawn=[],
            dormant=[],
            cull=[],
        )
        assert out.agent_responder_picks == ["alice"]

    def test_observation_harvest_picks_do_not_need_to_observe(self):
        out = EventRouterOutput(
            event_id="",
            decision_rationale="test fixture",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[
                    ObservableFact.only("Alice studies Pip.", ["alice"]),
                ],
            ),
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    response_priority=1,
                )
            ],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=["pip"],
            ends_beat=True,
            ends_beat_reason="observation_harvest",
            spawn=[],
            dormant=[],
            cull=[],
        )
        assert out.agent_responder_picks == ["pip"]

    def test_off_stage_tick_is_valid_end_reason(self):
        out = _router_out(ends_beat=True)
        data = out.model_dump()
        data["ends_beat_reason"] = "off_stage_tick"
        rebuilt = EventRouterOutput.model_validate(data)
        assert rebuilt.ends_beat_reason == "off_stage_tick"

    def test_unknown_ends_beat_reason_coerced_to_empty(self):
        out = _router_out(ends_beat=True)
        data = out.model_dump()
        data["ends_beat_reason"] = "location-transition"
        rebuilt = EventRouterOutput.model_validate(data)
        assert rebuilt.ends_beat_reason == ""

    def test_observation_harvest_coerces_ends_beat_true(self):
        out = _router_out(ends_beat=True, agent_picks=["pip"])
        data = out.model_dump()
        data["ends_beat_reason"] = "observation_harvest"
        data["ends_beat"] = False
        rebuilt = EventRouterOutput.model_validate(data)
        assert rebuilt.ends_beat is True


class TestEndBeatFanout:
    def test_buffered_humans_render_in_parallel(self):
        import time
        from app.engine.turn_loop import _end_beat

        ckpt = _ckpt({"alice": "1", "bob": "2"})
        event = _router_out(ends_beat=True)
        ckpt.canonical_events.append(event)
        append_to_render_buffer(ckpt, "alice", event.event_id, "direct")
        append_to_render_buffer(ckpt, "bob", event.event_id, "direct")

        class SlowDispatcher(FakeDispatcher):
            async def narrator_compose(self, **kw):
                await asyncio.sleep(0.1)
                return await super().narrator_compose(**kw)

        fake = SlowDispatcher()
        t0 = time.monotonic()
        result = asyncio.run(_end_beat(
            ckpt,
            fake,
            ended_reason="directed_at_player",
            events_closed=1,
            event_actor_ids=["alice"],
        ))
        elapsed = time.monotonic() - t0

        assert set(result.renders) == {"alice", "bob"}
        assert len(fake.narrator_calls) == 2
        assert elapsed < 0.18


class TestRejectionFormatting:
    def test_mentions_holder_name_on_initiator_held(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        claim_initiator_slot(ckpt, "alice")
        check = check_act_slot(ckpt, "bob")
        assert "Alice" in format_slot_rejection(check, ckpt)

    def test_echoes_attempted_text(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        claim_initiator_slot(ckpt, "alice")
        check = check_act_slot(ckpt, "bob")
        msg = format_slot_rejection(
            check,
            ckpt,
            attempted_text="I walk outside and look at the stars",
        )
        assert "I walk outside" in msg
