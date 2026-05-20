"""Turn-loop state-machine tests.

These exercise the orchestrator-independent pieces: session-wide act slots,
Cat II collection, beat cascade endings, observer-driven broadcast, and
error-message formatting. A fake dispatcher stands in for LLM calls.
"""

from __future__ import annotations

import asyncio

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
    _end_dnd_combat_from_router_signal,
)
from app.schemas.characters import CharacterRecord, CharacterVisuals, PublicSheet
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import (
    DndObserverEntry,
    EventRouterOutput,
    ObserverEntry,
    SpawnRequest,
    empty_commitment_open_signal,
)
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    WorldAdjudication,
    visible_fact_texts,
)
from app.schemas.router_targets import targets_from_router_output
from app.schemas.state import (
    DndCombatantState,
    DndCombatState,
    DndRouterObservedFact,
    OpenCommitment,
    RenderBufferEntry,
    SlotEntry,
)
from tests.support.factories import (
    InstanceFakeDispatcher as FakeDispatcher,
    dnd_router_output as _dnd_router_out,
    gatehouse_checkpoint,
    router_output as _router_out,
)


def _ckpt(bindings: dict[str, str] | None = None):
    return gatehouse_checkpoint(bindings=bindings)


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

    def test_combat_dnd_reaction_observer_gets_reaction_prompt(self):
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
            effective_at_s=0,
            duration_s=0,
            decision_rationale="reaction prompt",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[ObservableFact.all("Alice rushes past Bob.")],
            ),
            event_kind="directed_at_player",
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    routing_role="observe_only",
                ),
                DndObserverEntry(
                    character_id="bob",
                    observation_level="d",
                    routing_role="dnd_reaction",
                ),
            ],
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
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

    def test_combat_observe_only_observer_does_not_get_reaction_prompt(self):
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
            effective_at_s=0,
            duration_s=0,
            decision_rationale="no reaction prompt",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[ObservableFact.all("Alice shifts her stance.")],
            ),
            event_kind="directed_at_player",
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    routing_role="observe_only",
                ),
                ObserverEntry(
                    character_id="bob",
                    observation_level="d",
                    routing_role="observe_only",
                ),
            ],
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
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
            effective_at_s=0,
            duration_s=0,
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
                    routing_role="observe_only",
                )
            ],
            event_kind="directed_at_player",
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
        )
        ckpt.canonical_events.append(trigger)
        ckpt.session.active_act_slots["bob"] = SlotEntry(
            reason="combat_reaction",
            trigger_event_id="evt_trigger",
        )
        fake = FakeDispatcher()
        combat_out = _router_out(ends_beat=True)
        combat_out.event_kind = "ruleset_resolution"
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
            effective_at_s=0,
            duration_s=0,
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
                    routing_role="observe_only",
                ),
                ObserverEntry(
                    character_id="pip",
                    observation_level="f",
                    routing_role="perception_enrichment",
                ),
            ],
            event_kind="query_response",
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
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
            "[loadout — Pip] Pip wears a red coat.",
        ]
        assert "alice" in result.renders

    def test_query_harvest_suppresses_router_authored_visual_guess(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(EventRouterOutput(
            event_id="",
            effective_at_s=0,
            duration_s=0,
            decision_rationale="private query answer",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[
                    ObservableFact.only(
                        "Pip has invented gray-green eyes.", ["alice"],
                    ),
                ],
            ),
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    routing_role="observe_only",
                ),
                ObserverEntry(
                    character_id="pip",
                    observation_level="f",
                    routing_role="perception_enrichment",
                ),
            ],
            event_kind="query_response",
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
        ))
        fake.queue_harvest(["Pip has brown eyes."])

        asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="(query: what does Pip look like?)",
        ))

        facts = ckpt.canonical_events[0].canonical_event.observable_facts
        assert [fact.text for fact in facts] == [
            "[loadout — Pip] Pip has brown eyes.",
        ]

    def test_query_harvest_refreshes_existing_router_history(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session_conversation.append(ConversationMessage(
            role="assistant",
            content=(
                "prior_event evt_query @0+0 source=alice mode=intention "
                "end=query_response\n"
                "fact only[alice] @0+0: Pip has invented gray-green eyes."
            ),
        ))
        fake = FakeDispatcher()
        fake.queue_route(EventRouterOutput(
            event_id="evt_query",
            effective_at_s=0,
            duration_s=0,
            decision_rationale="private query answer",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[
                    ObservableFact.only(
                        "Pip has invented gray-green eyes.", ["alice"],
                    ),
                ],
            ),
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    routing_role="observe_only",
                ),
                ObserverEntry(
                    character_id="pip",
                    observation_level="f",
                    routing_role="perception_enrichment",
                ),
            ],
            event_kind="query_response",
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
        ))
        fake.queue_harvest(["Pip has brown eyes."])

        asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="(query: what does Pip look like?)",
        ))

        stored = ckpt.session_conversation[-1].content
        assert "Pip has brown eyes" in stored
        assert "gray-green" not in stored

    def test_cat_i_cascades_through_agent_pick(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        prior = _router_out(agent_ids=["pip"], ends_beat=False)
        fake.queue_route(prior)
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
        assert fake.agent_calls[0]["frame"] == "foreground"
        assert len(fake.agent_output_calls) == 1
        agent_output = fake.agent_output_calls[0]
        assert agent_output["character_id"] == "pip"
        assert agent_output["public_text"] == "Pip polishes the bell"
        assert "prior_result" not in agent_output

    def test_public_fact_observe_only_delivers_inbox_without_cascade(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fact = ObservableFact.all(
            "Official criers announce the cohort has been summoned."
        )
        fake.queue_route(_router_out(
            event_kind="public_fact",
            observer_ids=["alice", "pip"],
            facts=[fact],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert result.events_closed == 1
        assert result.ended_reason == "public_fact"
        assert pip.pending_observations == [
            "Official criers announce the cohort has been summoned."
        ]
        assert fake.agent_calls == []
        assert fake.agent_output_calls == []
        assert fake.continuation_calls == []

    def test_public_fact_next_output_dispatches_background_turn(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        prior = _router_out(
            event_kind="public_fact",
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
            facts=[ObservableFact.all("A courier reaches Pip with the news.")],
        )
        fake.queue_route(prior)
        fake.queue_agent("Pip sends a runner to the archive")
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert result.events_closed == 2
        assert fake.agent_calls[0]["character_id"] == "pip"
        assert fake.agent_calls[0]["frame"] == "background"
        assert "Location: gatehouse" in fake.agent_calls[0]["local_context"]
        assert "Alice (alice)" in fake.agent_calls[0]["local_context"]
        agent_output = fake.agent_output_calls[0]
        assert agent_output["character_id"] == "pip"
        assert "prior_result" not in agent_output

    def test_background_thread_cap_limits_public_fact_cascades(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            event_kind="public_fact",
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
        ))
        for i in range(4):
            fake.queue_agent(f"Pip advances background thread {i}")
            fake.queue_route(_router_out(
                event_kind="public_fact",
                agent_ids=["pip"],
                observer_ids=["alice", "pip"],
            ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert result.ended_reason == "cascade_cap"
        assert result.events_closed == 5
        assert len(fake.agent_calls) == 4
        assert len(fake.agent_output_calls) == 4

    def test_agent_pick_without_bound_player_observer_uses_private_frame(self):
        ckpt = _ckpt({})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(agent_ids=["pip"], ends_beat=False))
        fake.queue_agent("Pip lowers his voice")
        fake.queue_route(_router_out(ends_beat=True))

        asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert fake.agent_calls[0]["character_id"] == "pip"
        assert fake.agent_calls[0]["frame"] == "private"
        agent_output = fake.agent_output_calls[0]
        assert agent_output["character_id"] == "pip"

    def test_agent_cascade_cap_forces_beat_end(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.max_agent_cascades_per_beat = 1
        fake = FakeDispatcher()
        fake.queue_route(_router_out(agent_ids=["pip"], ends_beat=False))
        fake.queue_agent("Pip polishes the bell")
        fake.queue_route(_router_out(agent_ids=["pip"], ends_beat=False))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert result.ended_reason == "cascade_cap"
        assert result.events_closed == 2
        assert len(fake.route_calls) == 1
        assert len(fake.agent_output_calls) == 1
        assert len(fake.agent_calls) == 1

    def test_cat_i_dispatches_only_first_agent_pick_before_router_roundtrip(
        self,
    ):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(agent_ids=["pip", "bob"], ends_beat=False))
        fake.queue_agent("Pip polishes the bell")
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert result.events_closed == 2
        assert [call["character_id"] for call in fake.agent_calls] == [
            "pip",
        ]
        assert len(fake.agent_output_calls) == 1
        assert fake.agent_output_calls[0]["character_id"] == "pip"
        assert fake.agent_output_calls[0]["public_text"] == "Pip polishes the bell"

    def test_router_can_continue_to_next_agent_pick_after_canonicalizing_first(
        self,
    ):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(agent_ids=["pip", "bob"], ends_beat=False))
        fake.queue_agent("Pip polishes the bell")
        fake.queue_route(_router_out(agent_ids=["bob"], ends_beat=False))
        fake.queue_agent("Bob studies the latch")
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert result.events_closed == 3
        assert [call["character_id"] for call in fake.agent_calls] == [
            "pip",
            "bob",
        ]
        assert len(fake.agent_output_calls) == 2
        assert [
            call["character_id"]
            for call in fake.agent_output_calls
        ] == ["pip", "bob"]

    def test_false_endbeat_with_no_next_output_routes_continuation(self):
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

    def test_repeated_false_endbeat_without_next_output_errors(self):
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
            effective_at_s=100,
            duration_s=30,
        ))
        fake.queue_agent("Pip dodges")
        fake.queue_route(_router_out(
            ends_beat=True,
            effective_at_s=500,
            duration_s=5,
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I attack Pip",
        ))

        assert result.ended_reason == "cat_ii_resolution"
        assert result.events_closed == 2
        assert ckpt.canonical_events[0].event_kind == "cat_ii_open"
        assert ckpt.canonical_events[0].effective_at_s == 100
        assert ckpt.canonical_events[0].duration_s == 0
        assert ckpt.canonical_events[1].effective_at_s == 100
        assert ckpt.session.open_cat_ii_events == []

    def test_cat_ii_materializes_spawned_responder_before_dispatch(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["hidden_lookout"],
            observer_ids=["alice", "hidden_lookout"],
            ends_beat=False,
            spawn=[
                SpawnRequest(
                    character_id="hidden_lookout",
                    seed={
                        "role": "hidden lookout",
                        "reason": "spotted the stealth attempt",
                        "location": "mill hedge",
                        "objectives": ["track intruders"],
                    },
                ),
            ],
        ))
        fake.queue_agent("The lookout freezes and signals.")
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I creep toward the mill.",
        ))

        assert result.ended_reason == "cat_ii_resolution"
        assert fake.materialize_calls
        assert fake.agent_calls[0]["character_id"] == "hidden_lookout"
        assert fake.agent_character_exists == [True]
        assert any(c.character_id == "hidden_lookout" for c in ckpt.characters)
        assert ckpt.canonical_events[0].spawn == []

    def test_cat_ii_unknown_required_responder_without_spawn_errors(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["ghost_responder"],
            observer_ids=["alice", "ghost_responder"],
            ends_beat=False,
        ))

        with pytest.raises(RuntimeError, match="not in the roster"):
            asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="I creep toward the mill.",
            ))

        assert fake.materialize_calls == []
        assert fake.agent_calls == []

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
        assert ckpt.canonical_events[0].event_kind == (
            "ruleset_cat_ii_suppressed"
        )
        assert fake.agent_calls == []

    def test_dnd_combat_start_signal_starts_combat_tracker(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        fake = FakeDispatcher()
        fake.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice", "bob"],
            battle_map_seed={
                "present": True,
                "map_name": "Gatehouse",
                "width": 8,
                "height": 6,
                "square_size_ft": 5,
                "tokens": [
                    {
                        "token_id": "alice",
                        "character_id": "alice",
                        "label": "Alice",
                        "x": 1,
                        "y": 2,
                        "size_squares": 1,
                    },
                    {
                        "token_id": "stray",
                        "character_id": "stray",
                        "label": "Stray",
                        "x": 7,
                        "y": 5,
                        "size_squares": 1,
                    },
                ],
                "terrain": [
                    {
                        "zone_id": "crate",
                        "label": "Crate",
                        "x": 3,
                        "y": 2,
                        "width": 1,
                        "height": 1,
                        "blocks_movement": True,
                        "blocks_line_of_sight": False,
                        "cover": "half",
                        "notes": "",
                    }
                ],
                "areas": [],
                "notes": "",
            },
            facts=[ObservableFact.all("Alice commits to an attack against Bob.")],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I attack Bob",
        ))

        assert result.ended_reason == "combat_started"
        assert ckpt.session.active_combat is not None
        assert ckpt.session.open_cat_ii_events == []
        assert ckpt.session.active_act_slots == {}
        assert fake.agent_calls == []
        assert {c.character_id for c in ckpt.session.active_combat.combatants} == {
            "alice", "bob",
        }
        battle_map = ckpt.session.active_combat.battle_map
        assert battle_map is not None
        assert battle_map.map_name == "Gatehouse"
        assert battle_map.width == 8
        assert {token.character_id for token in battle_map.tokens} == {
            "alice", "bob",
        }
        assert next(
            token for token in battle_map.tokens if token.character_id == "alice"
        ).x == 1
        assert battle_map.terrain[0].cover == "half"
        assert ckpt.canonical_events[0].requires_responders is False
        facts = [
            fact.text
            for fact in ckpt.canonical_events[0].canonical_event.observable_facts
        ]
        assert any("D&D combat begins" in fact for fact in facts)
        assert all("does not resolve before initiative" not in fact for fact in facts)
        assert all("Initiative order" not in fact for fact in facts)
        assert {observer.character_id for observer in ckpt.canonical_events[0].observers} >= {
            "alice", "bob",
        }
        alice = next(
            c for c in ckpt.session.active_combat.combatants
            if c.character_id == "alice"
        )
        assert alice.pending_initiating_action == "I attack Bob"

    def test_underpopulated_dnd_combat_start_signal_closes_without_crashing(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        fake = FakeDispatcher()
        fake.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice"],
            facts=[ObservableFact.all("Alice raises a blade with no clear opponent.")],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I attack.",
        ))

        assert result.ended_reason == "state_change"
        assert ckpt.session.active_combat is None
        assert ckpt.session.active_act_slots == {}

    def test_dnd_social_cat_ii_does_not_start_combat(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        fake = FakeDispatcher()
        fake.queue_route(_dnd_router_out(
            interaction_mode="cat_ii",
            requires_responders=True,
            required_responders=["bob"],
            facts=[
                ObservableFact.all(
                    "Alice leans toward Bob without completing contact."
                )
            ],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I try to kiss Bob",
        ))

        assert result.ended_reason == "cat_ii_pending"
        assert ckpt.session.active_combat is None
        assert [evt.initiator_id for evt in ckpt.session.open_cat_ii_events] == [
            "alice",
        ]
        assert ckpt.session.open_cat_ii_events[0].required_responders == ["bob"]
        assert ckpt.session.active_act_slots["bob"].reason == "cat_ii_responder"

    def test_second_dnd_combat_start_pins_only_the_blocked_action(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.session.active_combat = DndCombatState(
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
        fake = FakeDispatcher()
        fake.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice", "pip"],
            facts=[ObservableFact.all("Alice raises a blade in another room.")],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I attack Pip.",
        ))

        assert result.ended_reason == "combat_start_blocked"
        assert ckpt.session.active_combat is not None
        assert ckpt.session.active_act_slots["alice"].reason == "combat_blocked"
        assert check_act_slot(ckpt, "alice").conflict == SlotConflict.FREE
        facts = [
            fact.text
            for fact in ckpt.canonical_events[0].canonical_event.observable_facts
        ]
        assert any("raises a blade" in fact for fact in facts)
        assert any("no attack, spell, or injury takes effect" in fact for fact in facts)
        assert all("already in initiative" not in fact for fact in facts)
        assert {observer.character_id for observer in ckpt.canonical_events[0].observers} >= {
            "alice",
        }

    def test_dnd_combat_end_does_not_broadcast_to_blocked_outsider(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="combat_blocked",
            trigger_event_id="evt_blocked",
        )
        ckpt.session.active_combat = DndCombatState(
            turn_index=0,
            router_observed_facts=[
                DndRouterObservedFact(
                    fact="Bob secured Pip's surrender without killing him.",
                    salience="notable",
                    reason="This mercy should carry into the post-combat scene.",
                ),
            ],
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
        out = _dnd_router_out(
            interaction_mode="dnd_combat_end",
            facts=[ObservableFact.all("Bob and Pip lower their weapons.")],
        )
        out.observers = [
            ObserverEntry(character_id="bob", observation_level="d", routing_role="observe_only"),
            ObserverEntry(character_id="pip", observation_level="d", routing_role="observe_only"),
        ]

        ended = _end_dnd_combat_from_router_signal(
            ckpt,
            out,
            actor_id="bob",
        )

        assert ended is True
        assert ckpt.session.active_combat is None
        assert ckpt.session.active_act_slots == {}
        assert ckpt.session.pending_engine_state_updates == [
            "Combat continuity [notable]: Bob secured Pip's surrender without "
            "killing him. Reason: This mercy should carry into the post-combat "
            "scene."
        ]
        event = out
        assert "alice" not in {observer.character_id for observer in event.observers}
        facts = [fact.text for fact in event.canonical_event.observable_facts]
        assert all("You may act again" not in fact for fact in facts)

    def test_dnd_combat_end_signal_from_outsider_does_not_clear_active_combat(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.session.active_combat = DndCombatState(
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=False,
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
        fake.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_end",
            facts=[ObservableFact.all("Bob and Pip lower their weapons.")],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="The fight is over.",
        ))

        assert result.ended_reason == "state_change"
        assert ckpt.session.active_combat is not None
        assert ckpt.canonical_events[0].requires_responders is False
        facts = [
            fact.text
            for fact in ckpt.canonical_events[0].canonical_event.observable_facts
        ]
        assert not any("D&D combat ends" in fact for fact in facts)

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
            facts=[ObservableFact.all("Alice's strike resolves against Bob.")],
        )
        combat_out.event_kind = "ruleset_resolution"
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
        assert ckpt.canonical_events[0].event_kind == (
            "ruleset_resolution"
        )

    def test_combat_observers_keep_visible_events_on_npc_turn(self):
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
            effective_at_s=0,
            duration_s=0,
            decision_rationale="Pip hits Bob.",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[ObservableFact.all("Pip cuts Bob.")],
            ),
            event_kind="ruleset_resolution",
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    routing_role="observe_only",
                ),
                ObserverEntry(
                    character_id="bob",
                    observation_level="d",
                    routing_role="observe_only",
                ),
            ],
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
        )
        fake.queue_combat(combat_out)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="pip",
            intention="I attack Bob",
        ))

        assert result.renders == {"alice": "RENDER", "bob": "RENDER"}
        assert ckpt.session.render_buffers.get("alice") == []
        assert ckpt.session.render_buffers.get("bob") == []

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

    def test_cat_ii_resolution_routes_explicit_next_output_before_initiator(self):
        ckpt = _ckpt({"alice": "1"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="alice",
            initiator_intention="I warn Pip away from the door",
            required_responders=["pip"],
        )

        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
            ends_beat=False,
        ))
        fake.queue_agent("Pip answers the warning.")
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="pip",
            intention="I try not to show fear",
            cat_ii_event_id=evt.event_id,
        ))

        assert result.ended_reason == "directed_at_player"
        assert ckpt.session.open_cat_ii_events == []
        assert fake.agent_calls[0]["character_id"] == "pip"
        assert fake.agent_output_calls[0]["character_id"] == "pip"
        assert fake.route_calls[0]["actor_id"] == "alice"

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
        out = _router_out(ends_beat=True, agent_ids=["pip", "vex"])
        out.event_kind = "observation_harvest"
        for observer in out.observers:
            if observer.character_id in {"pip", "vex"}:
                observer.routing_role = "perception_enrichment"
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

    def test_harvest_drops_human_targets(self):
        ckpt = _ckpt({"alice": "1"})
        out = _router_out(ends_beat=True, agent_ids=["alice"])
        out.event_kind = "observation_harvest"
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
        facts: list[ObservableFact] | None = None,
        effective_at_s: int = 0,
        duration_s: int = 0,
    ) -> EventRouterOutput:
        return EventRouterOutput(
            event_id="",
            effective_at_s=effective_at_s,
            duration_s=duration_s,
            decision_rationale="test fixture",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=facts if facts is not None else [
                    ObservableFact.all("Alice sets down a glass.")
                ],
            ),
            event_kind="directed_at_player",
            observers=[
                ObserverEntry(
                    character_id=cid,
                    observation_level="d",
                    routing_role="observe_only",
                )
                for cid in observer_ids
            ],
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
        )

    def _with_updates(self, event: EventRouterOutput, **updates) -> EventRouterOutput:
        data = event.model_dump()
        data.update(updates)
        return EventRouterOutput.model_validate(data)

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

    def test_npc_observer_gets_first_meeting_loadout_once(self):
        ckpt = _ckpt()
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        alice.visuals = CharacterVisuals(
            default_loadout="Blue travel cloak, rain-dark hair, silver pin.",
        )
        event = self._event(
            observer_ids=["pip"],
            facts=[ObservableFact.all("Alice says, 'Here.'")],
        )

        broadcast_event(ckpt, event, actor_id="alice")
        broadcast_event(ckpt, event, actor_id="alice")

        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == [
            "Alice says, 'Here.'",
            (
                "Newly introduced character context:\n"
                "- Alice: first visible impression: Blue travel cloak, rain-dark hair, silver pin."
            ),
            "Alice says, 'Here.'",
        ]
        assert ckpt.session.visual_introductions["pip"] == ["alice"]

    def test_indirect_observation_does_not_add_first_meeting_loadout(self):
        ckpt = _ckpt()
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        alice.visuals = CharacterVisuals(default_loadout="Blue travel cloak.")
        event = self._event(observer_ids=["pip"])
        event.observers[0].observation_level = "i"

        broadcast_event(ckpt, event, actor_id="alice")

        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == ["Alice sets down a glass."]
        assert ckpt.session.visual_introductions == {}

    def test_npc_observer_sees_names_when_router_fact_uses_ids(self):
        ckpt = _ckpt()
        event = self._event(
            observer_ids=["pip"],
            facts=[ObservableFact.all("alice sets bob's cup by pip.")],
        )

        broadcast_event(ckpt, event, actor_id="alice")

        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == ["Alice sets Bob's cup by Pip."]

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

    def test_private_channel_fact_reaches_bound_pov_without_npc_leak(self):
        ckpt = _ckpt({"alice": "1"})
        event = self._event(
            observer_ids=["alice", "bob"],
            facts=[
                ObservableFact.only(
                    "A Message spell whispers only to Alice.",
                    ["alice"],
                ),
                ObservableFact.all("Bob keeps watch at the arch."),
            ],
        )

        broadcast_event(ckpt, event, actor_id="pip")

        bob = next(c for c in ckpt.characters if c.character_id == "bob")
        assert bob.pending_observations == ["Bob keeps watch at the arch."]
        assert ckpt.session.render_buffers["alice"][0].event_id == event.event_id
        assert visible_fact_texts(
            event.canonical_event.observable_facts,
            "alice",
            include_all_observers=True,
        ) == [
            "A Message spell whispers only to Alice.",
            "Bob keeps watch at the arch.",
        ]
        assert visible_fact_texts(
            event.canonical_event.observable_facts,
            "bob",
            include_all_observers=True,
        ) == ["Bob keeps watch at the arch."]

    def test_empty_observable_facts_means_no_push(self):
        ckpt = _ckpt()
        event = self._event(observer_ids=["pip"], facts=[])
        broadcast_event(ckpt, event, actor_id="alice")
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == []

    def test_broadcast_applies_relative_time_to_clocks_and_buffers(self):
        ckpt = _ckpt({"alice": "1"})
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        pip.clock_at_s = 10
        event = self._with_updates(
            self._event(
                observer_ids=["alice", "pip"],
                facts=[
                    ObservableFact.all(
                        "Pip opens the west door.",
                        at_offset_s=2,
                        duration_s=3,
                    )
                ],
                effective_at_s=4,
                duration_s=10,
            ),
        )

        visible = broadcast_event(ckpt, event, actor_id="pip")

        assert visible == ["alice"]
        assert event.effective_at_s == 10
        assert pip.clock_at_s == 20
        assert alice.clock_at_s == 15
        assert pip.last_agent_turn_at_s == 20
        assert alice.last_agent_turn_at_s is None
        assert ckpt.session.leading_at_s == 20
        entry = ckpt.session.render_buffers["alice"][0]
        assert entry.visible_at_s == 15
        assert entry.event_sequence == 0

    def test_observing_event_does_not_reset_last_agent_turn_time(self):
        ckpt = _ckpt({"alice": "1"})
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        pip.last_agent_turn_at_s = 5
        event = self._with_updates(
            self._event(
                observer_ids=["pip"],
                facts=[ObservableFact.all("Alice sets down a glass.")],
                effective_at_s=25,
            ),
        )

        broadcast_event(ckpt, event, actor_id="alice")

        assert pip.clock_at_s == 25
        assert pip.last_agent_turn_at_s == 5

    def test_open_commitment_is_private_and_does_not_render_without_facts(self):
        ckpt = _ckpt({"alice": "1"})
        event = self._with_updates(
            self._event(observer_ids=["alice"], facts=[]),
            commitment_open={
                "present": True,
                "actor_ids": ["alice"],
                "description": "Alice searches the cabinet.",
                "expected_duration_s": 60,
                "max_duration_s": 180,
                "location_label": "gatehouse",
            },
        )

        visible = broadcast_event(ckpt, event, actor_id="alice")

        assert visible == []
        assert ckpt.session.render_buffers == {}
        assert len(ckpt.session.open_commitments) == 1
        assert ckpt.session.open_commitments[0].description == (
            "Alice searches the cabinet."
        )

    def test_commitment_interrupt_creates_revision_without_act_slot(self):
        ckpt = _ckpt({"alice": "1"})
        open_event = self._with_updates(
            self._event(observer_ids=["alice"], facts=[]),
            commitment_open={
                "present": True,
                "actor_ids": ["alice"],
                "description": "Alice searches the cabinet.",
                "expected_duration_s": 60,
                "max_duration_s": 180,
                "location_label": "gatehouse",
            },
        )
        broadcast_event(ckpt, open_event, actor_id="alice")
        commitment_id = ckpt.session.open_commitments[0].commitment_id

        interrupt_event = self._with_updates(
            self._event(
                observer_ids=["alice"],
                facts=[ObservableFact.all("The cabinet door swings shut.")],
            ),
            commitment_interrupts=[
                {
                    "commitment_id": commitment_id,
                    "actor_ids": ["alice"],
                    "observed_at_offset_s": 0,
                    "reason": "the cabinet changed while Alice searched it",
                }
            ],
        )

        broadcast_event(ckpt, interrupt_event, actor_id="pip")

        prompt = ckpt.session.pending_commitment_revisions["alice"]
        assert prompt.commitment_id == commitment_id
        assert prompt.trigger_event_id == interrupt_event.event_id
        assert ckpt.session.active_act_slots == {}
        assert len(ckpt.session.open_commitments) == 1

    def test_commitment_id_takes_precedence_over_actor_ids(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        ckpt.session.open_commitments = [
            OpenCommitment(
                commitment_id="commit_alice",
                actor_ids=["alice"],
                description="Alice searches.",
            ),
            OpenCommitment(
                commitment_id="commit_bob",
                actor_ids=["bob"],
                description="Bob waits.",
            ),
        ]
        event = self._with_updates(
            self._event(observer_ids=[], facts=[]),
            commitment_resolutions=[
                {
                    "commitment_id": "commit_alice",
                    "actor_ids": ["bob"],
                    "reason": "resolved",
                    "resolved_at_offset_s": 0,
                }
            ],
        )

        broadcast_event(ckpt, event, actor_id="pip")

        assert [c.commitment_id for c in ckpt.session.open_commitments] == [
            "commit_bob"
        ]

    def test_commitment_resolution_offset_advances_committed_actor_clock(self):
        ckpt = _ckpt({"alice": "1"})
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        ckpt.session.open_commitments = [
            OpenCommitment(
                commitment_id="commit_alice",
                actor_ids=["alice"],
                description="Alice waits.",
            )
        ]
        event = self._event(
            observer_ids=[],
            facts=[],
            effective_at_s=20,
            duration_s=30,
        )
        event = self._with_updates(
            event,
            commitment_resolutions=[
                {
                    "commitment_id": "commit_alice",
                    "actor_ids": [],
                    "reason": "resolved",
                    "resolved_at_offset_s": 15,
                }
            ],
        )

        broadcast_event(ckpt, event, actor_id="pip")

        assert ckpt.session.open_commitments == []
        assert alice.clock_at_s == 35

    def test_visible_event_auto_interrupts_human_open_commitment(self):
        ckpt = _ckpt({"alice": "1"})
        open_event = self._with_updates(
            self._event(observer_ids=["alice"], facts=[]),
            commitment_open={
                "present": True,
                "actor_ids": ["alice"],
                "description": "Alice searches the cabinet.",
                "expected_duration_s": 60,
                "max_duration_s": 180,
                "location_label": "gatehouse",
            },
        )
        broadcast_event(ckpt, open_event, actor_id="alice")
        commitment_id = ckpt.session.open_commitments[0].commitment_id

        visible_event = self._event(
            observer_ids=["alice"],
            facts=[ObservableFact.all("Pip drops a key beside Alice.")],
        )
        broadcast_event(ckpt, visible_event, actor_id="pip")

        prompt = ckpt.session.pending_commitment_revisions["alice"]
        assert prompt.commitment_id == commitment_id
        assert prompt.trigger_event_id == visible_event.event_id
        assert "new visible information" in prompt.reason
        assert ckpt.session.active_act_slots == {}


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
        from app.engine.turn_loop import _filter_routed_agents_for_dispatch

        ckpt = _ckpt({"alice": "1"})
        watcher = CharacterRecord(
            character_id="watcher",
            name="Watcher",
            public_sheet=PublicSheet(role="operator"),
            location="control_room",
        )
        ckpt.characters.append(watcher)

        assert _filter_routed_agents_for_dispatch(ckpt, ["watcher"]) == ["watcher"]

    def test_human_pick_filtered(self):
        from app.engine.turn_loop import _filter_routed_agents_for_dispatch

        ckpt = _ckpt({"alice": "1", "bob": "2"})
        assert _filter_routed_agents_for_dispatch(ckpt, ["alice", "bob", "pip"]) == [
            "pip"
        ]


class TestSchemaValidators:
    def test_requires_responders_without_list_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _router_out(requires_responders=True, required_responders=[])

    def test_next_output_observers_drive_router_targets(self):
        out = EventRouterOutput(
            event_id="",
            effective_at_s=0,
            duration_s=0,
            decision_rationale="test fixture",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[],
            ),
            event_kind="beat_continues",
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    routing_role="observe_only",
                ),
                ObserverEntry(
                    character_id="pip",
                    observation_level="d",
                    routing_role="next_output",
                ),
            ],
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
        )
        assert out.next_output_character_ids == ["pip"]

    def test_public_fact_event_kind_is_accepted(self):
        out = _router_out(event_kind="public_fact", observer_ids=["pip"])
        assert out.event_kind == "public_fact"
        assert out.event_kind != "beat_continues"

    def test_observation_harvest_uses_perception_enrichment_targets(self):
        out = EventRouterOutput(
            event_id="",
            effective_at_s=0,
            duration_s=0,
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
                    routing_role="observe_only",
                ),
                ObserverEntry(
                    character_id="pip",
                    observation_level="f",
                    routing_role="perception_enrichment",
                ),
            ],
            event_kind="observation_harvest",
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
        )
        assert out.perception_enrichment_character_ids == ["pip"]

    def test_legacy_agent_responder_picks_field_is_rejected(self):
        out = _router_out(agent_ids=["pip"], ends_beat=False)
        data = out.model_dump()
        data["agent_responder_picks"] = ["pip", "offstage_npc"]
        with pytest.raises(ValueError, match="Extra inputs"):
            EventRouterOutput.model_validate(data)

    def test_router_target_projection_uses_runtime_frame_semantics(self):
        out = _router_out(agent_ids=["pip"], ends_beat=False)
        assert targets_from_router_output(
            out,
            player_ids={"alice"},
            agent_ids=["pip"],
        ).targets[0].frame == "foreground"
        assert targets_from_router_output(
            out,
            player_ids=set(),
            agent_ids=["pip"],
        ).targets[0].frame == "private"

        offstage = _router_out(ends_beat=False)
        assert targets_from_router_output(
            offstage,
            player_ids={"alice"},
            agent_ids=["offstage_npc"],
        ).targets[0].frame == "background"

        public_fact = _router_out(
            event_kind="public_fact",
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
        )
        assert targets_from_router_output(
            public_fact,
            player_ids={"alice"},
            agent_ids=["pip"],
        ).targets[0].frame == "background"

        departing = _router_out(
            agent_ids=["pip"],
            ends_beat=False,
            observer_ids=["alice", "pip"],
            location_updates=[
                {"character_id": "pip", "location_label": "archive"}
            ],
        )
        assert targets_from_router_output(
            departing,
            player_ids={"alice"},
            agent_ids=["pip"],
        ).targets[0].frame == "background"

    def test_unknown_event_kind_coerced_to_terminal_kind(self):
        out = _router_out(ends_beat=True)
        data = out.model_dump()
        data["event_kind"] = "location-transition"
        rebuilt = EventRouterOutput.model_validate(data)
        assert rebuilt.event_kind == "directed_at_player"

    def test_observation_harvest_is_terminal(self):
        out = _router_out(ends_beat=True, agent_ids=["pip"])
        data = out.model_dump()
        data["event_kind"] = "observation_harvest"
        for observer in data["observers"]:
            if observer["character_id"] == "pip":
                observer["routing_role"] = "perception_enrichment"
        rebuilt = EventRouterOutput.model_validate(data)
        assert rebuilt.event_kind != "beat_continues"


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
