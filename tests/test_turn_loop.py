"""Turn-loop state-machine tests.

These exercise the orchestrator-independent pieces: session-wide act slots,
Cat II collection, beat cascade endings, observer-driven broadcast, and
error-message formatting. A fake dispatcher stands in for LLM calls.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.engine.action_rejection import PlayerActionRejected
from app.engine.closed_event_runtime import (
    ClosedEventRuntime,
    install_closed_event_runtime,
)
from app.engine.content_resolver import append_pending_router_content_records
from app.engine.spawn_authoring import SpawnAuthoringCoordinator
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
from app.schemas.characters import (
    ActorRecord,
    CharacterRecord,
    CharacterVisuals,
    PlayerSlotKind,
    PublicSheet,
)
from app.schemas.conversation import ConversationMessage
from app.schemas.content import ContentPackState
from app.schemas.content_pack import FrontDossierRecord
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
from app.schemas.narrator import (
    NarratorFinalOutput,
    TranscriptEntry,
    VisualNovelBeatPages,
    VisualNovelNarratorOutput,
    VisualNovelPage,
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


def _queue_narrator_continue(fake, count: int = 1) -> None:
    for _ in range(count):
        fake.queue_narrator(
            handoff="continue",
            reason="The current motion still needs to advance.",
            text="DISCARDED CANDIDATE",
        )


class _SpeculativeBranchFake(FakeDispatcher):
    """Expose branch-local mutations so commit/discard is observable."""

    def __init__(self):
        super().__init__()
        self.branch_ready = asyncio.Event()

    async def agent_intend(self, **kw) -> str:
        output = await super().agent_intend(**kw)
        speculative_ckpt = kw["ckpt"]
        character_id = kw["character_id"]
        speculative_ckpt.character_conversations.setdefault(
            character_id, []
        ).append(ConversationMessage(
            role="assistant",
            content="speculative agent memory",
        ))
        character = next(
            c for c in speculative_ckpt.characters
            if c.character_id == character_id
        )
        character.pending_observations.clear()
        return output

    async def route_intention(self, **kw) -> EventRouterOutput:
        result = await super().route_intention(**kw)
        if kw["actor_id"] == "pip":
            kw["ckpt"].session_conversation.append(ConversationMessage(
                role="assistant",
                content="speculative router history",
            ))
            self.branch_ready.set()
        return result

    async def narrator_compose(self, **kw):
        await asyncio.wait_for(self.branch_ready.wait(), timeout=1)
        return await super().narrator_compose(**kw)


class _ConcurrentCatIIResponderFake(FakeDispatcher):
    """Pause autonomous responders until every isolated branch has started."""

    def __init__(self, *, failing_id: str = ""):
        super().__init__()
        self.failing_id = failing_id
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.branch_event_ids: dict[str, list[str]] = {}
        self.branch_collected_intentions: dict[str, dict[str, str]] = {}

    async def agent_intend(self, **kw) -> str:
        self.agent_calls.append(kw)
        character_id = kw["character_id"]
        branch = kw["ckpt"]
        self.branch_event_ids[character_id] = [
            event.event_id for event in branch.canonical_events
        ]
        self.branch_collected_intentions[character_id] = dict(
            branch.session.open_cat_ii_events[0].collected_intentions
        )
        if len(self.agent_calls) == 2:
            self.started.set()
        if character_id == self.failing_id:
            self.release.set()
            raise RuntimeError(f"responder failed: {character_id}")
        await self.release.wait()
        branch.character_conversations.setdefault(character_id, []).append(
            ConversationMessage(
                role="assistant",
                content=f"draft from {character_id}",
            )
        )
        return f"{character_id} commits to a distinct response."


def _hidden_front_dossier(
    private_dossier_text: str,
    hidden_plan: str,
) -> FrontDossierRecord:
    return FrontDossierRecord(
        ref="front.strahd",
        content_hash="hash-front-strahd",
        title="Strahd Front",
        summary=private_dossier_text,
        review_status="approved",
        gate_status="runtime_ready",
        villain_refs=["npc.strahd"],
        goals=[hidden_plan],
        initial_knowledge=["The tavern sheltering Ireena matters."],
        action_palette=[
            {
                "action_id": "test_tavern",
                "action_kind": "spy",
                "priority": 5,
                "trigger": "Ireena is sheltered in public.",
                "summary": "Summon wolves to test the tavern road.",
            }
        ],
    )


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
    def test_bound_side_effect_free_infeasible_submission_is_rejected(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        rejected = _router_out(observer_ids=[], facts=[])
        rejected.canonical_event.world_adjudication.feasible = False
        fake.queue_route(rejected)

        with pytest.raises(
            PlayerActionRejected,
            match="Nothing changed",
        ):
            asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="I buy an unavailable quantity of Gems.",
            ))

        assert ckpt.canonical_events == []
        assert fake.narrator_calls == []

    def test_factful_failed_attempt_remains_canonical(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        failed_attempt = _router_out(
            observer_ids=["alice"],
            facts=[ObservableFact.all(
                "Alice pulls the locked handle, but it does not move."
            )],
        )
        failed_attempt.canonical_event.world_adjudication.feasible = False
        fake.queue_route(failed_attempt)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I wrench the sealed door open.",
        ))

        assert result.events_closed == 1
        assert len(ckpt.canonical_events) == 1
        assert fake.narrator_calls

    def test_autonomous_side_effect_free_infeasible_result_is_not_ui_rejected(
        self,
    ):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        rejected = _router_out(observer_ids=[], facts=[])
        rejected.canonical_event.world_adjudication.feasible = False
        fake.queue_route(rejected)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="pip",
            intention="I attempt something impossible.",
        ))

        assert result.events_closed == 1
        assert len(ckpt.canonical_events) == 1

    def test_single_cat_i_closes_and_renders(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="look around",
        ))

        assert result.events_closed == 1
        assert result.ended_reason == "cascade_exhausted"
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
            event_kind="cascade_exhausted",
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
            event_kind="cascade_exhausted",
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

        assert result.ended_reason == "cascade_exhausted"
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
            event_kind="cascade_exhausted",
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
        combat_out = _router_out(event_kind="cascade_exhausted")
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
        prior = _router_out(agent_ids=["pip"], event_kind="beat_continues")
        fake.queue_route(prior)
        fake.queue_agent("Pip polishes the bell")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert result.events_closed == 2
        assert fake.agent_calls[0]["character_id"] == "pip"
        assert fake.agent_calls[0]["frame"] == "foreground"
        assert len(fake.route_calls) == 2
        agent_submission = fake.route_calls[1]
        assert agent_submission["actor_id"] == "pip"
        assert agent_submission["intention"] == "Pip polishes the bell"
        assert agent_submission["cat_ii_event"] is None

    def test_narrator_render_discards_completed_speculative_agent_branch(self):
        ckpt = _ckpt({"alice": "1"})
        fake = _SpeculativeBranchFake()
        fake.queue_route(_router_out(
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
            event_kind="beat_continues",
            facts=[ObservableFact.all("Alice asks Pip to answer.")],
        ))
        fake.queue_agent("Pip begins an answer.")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="Pip, answer me.",
        ))

        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert result.renders == {"alice": "RENDER"}
        assert result.events_closed == 1
        assert len(fake.agent_calls) == 1
        assert len(fake.route_calls) == 2
        assert fake.agent_calls[0]["ckpt"] is not ckpt
        assert ckpt.character_conversations.get("pip", []) == []
        assert "Alice asks Pip to answer." in pip.pending_observations
        assert all(
            message.content != "speculative router history"
            for message in ckpt.session_conversation
        )

    def test_narrator_continue_commits_completed_speculative_agent_branch(self):
        ckpt = _ckpt({"alice": "1"})
        pip_before = next(c for c in ckpt.characters if c.character_id == "pip")
        fake = _SpeculativeBranchFake()
        fake.queue_route(_router_out(
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
            event_kind="beat_continues",
            facts=[ObservableFact.all("Alice asks Pip to answer.")],
        ))
        fake.queue_agent("Pip gives his answer.")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="Pip, answer me.",
        ))

        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip is pip_before
        assert result.events_closed == 2
        assert len(fake.agent_calls) == 1
        assert len(fake.route_calls) == 2
        assert [
            message.content
            for message in ckpt.character_conversations["pip"]
        ] == ["speculative agent memory"]
        assert pip.pending_observations == []
        assert any(
            message.content == "speculative router history"
            for message in ckpt.session_conversation
        )

    def test_retried_narrator_continue_resumes_saved_next_output_target(self):
        ckpt = _ckpt({"alice": "1"})
        prior = _router_out(
            event_id="evt_waiting_for_pip",
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
            event_kind="beat_continues",
            facts=[ObservableFact.all("Alice waits for Pip's answer.")],
        )
        broadcast_event(ckpt, prior, actor_id="alice")
        fake = FakeDispatcher()
        fake.queue_agent("Pip finally answers.")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="Pip, answer me.",
            resume_after_handoff=prior,
            resume_events_closed=1,
            resume_event_actor_ids=["alice"],
        ))

        assert result.events_closed == 2
        assert fake.continuation_calls == []
        assert fake.agent_calls[0]["character_id"] == "pip"
        assert fake.route_calls[0]["actor_id"] == "pip"
        assert len(ckpt.canonical_events) == 2
        assert ckpt.canonical_events[0].event_id == "evt_waiting_for_pip"

    def test_parenthesized_agent_silence_is_routed_as_actor_submission(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
            event_kind="beat_continues",
        ))
        fake.queue_agent("(remains silent)")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I wait for Pip's answer.",
        ))

        assert result.ended_reason == "cascade_exhausted"
        assert len(fake.route_calls) == 2
        assert fake.route_calls[1]["actor_id"] == "pip"
        assert fake.route_calls[1]["intention"] == "(remains silent)"
        assert fake.continuation_calls == []

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
        assert len(fake.route_calls) == 1
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
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
        _queue_narrator_continue(fake)

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
        agent_submission = fake.route_calls[1]
        assert agent_submission["actor_id"] == "pip"
        assert agent_submission["intention"] == "Pip sends a runner to the archive"

    def test_public_front_signal_does_not_leak_hidden_plan_to_surfaces(
        self, caplog,
    ):
        private_dossier_text = (
            "PRIVATE DOSSIER: Strahd prepares the dinner ambush."
        )
        hidden_plan = "HIDDEN PLAN: abduct Ireena before dawn."
        public_fact = "A courier announces Ireena is sheltered in the tavern."
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.content_state = {
            "curse": ContentPackState(
                pack_id="curse",
                metadata={
                    "domain_catalog": {
                        "pack_id": "curse",
                        "front_dossiers": [
                            _hidden_front_dossier(
                                private_dossier_text,
                                hidden_plan,
                            ).model_dump(mode="json")
                        ],
                    }
                },
            )
        }
        fake = FakeDispatcher()
        fake.queue_route(
            _router_out(
                event_kind="public_fact",
                agent_ids=["pip"],
                observer_ids=["alice", "pip"],
                facts=[ObservableFact.all(public_fact)],
            )
        )
        fake.queue_agent("Pip sends a public warning.")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
        _queue_narrator_continue(fake)

        with caplog.at_level("INFO", logger="app.engine.turn_loop"):
            result = asyncio.run(
                run_beat(
                    ckpt=ckpt,
                    dispatcher=fake,
                    actor_id="alice",
                    intention="wait",
                )
            )

        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        joined_inbox = "\n".join(pip.pending_observations)
        joined_renders = "\n".join(result.renders.values())
        agent_context = "\n".join(
            str(call.get("local_context", ""))
            for call in fake.agent_calls
        )
        for hidden_text in (private_dossier_text, hidden_plan):
            assert hidden_text not in joined_inbox
            assert hidden_text not in joined_renders
            assert hidden_text not in agent_context
            assert hidden_text not in caplog.text
        assert public_fact in joined_inbox
        assert result.renders == {"alice": "RENDER"}

        records = append_pending_router_content_records(ckpt)

        assert len(records) == 1
        assert 'pressure="Summon wolves to test the tavern road."' in records[0]
        assert private_dossier_text not in records[0]
        assert hidden_plan not in records[0]

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
        _queue_narrator_continue(fake, count=5)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert result.ended_reason == "cascade_cap"
        assert result.events_closed == 5
        assert len(fake.agent_calls) == 4
        assert len(fake.route_calls) == 5

    def test_agent_pick_without_bound_player_observer_uses_private_frame(self):
        ckpt = _ckpt({})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(agent_ids=["pip"], event_kind="beat_continues"))
        fake.queue_agent("Pip lowers his voice")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
        _queue_narrator_continue(fake)

        asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert fake.agent_calls[0]["character_id"] == "pip"
        assert fake.agent_calls[0]["frame"] == "private"
        assert fake.route_calls[1]["actor_id"] == "pip"

    def test_agent_cascade_cap_forces_beat_end(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.max_agent_cascades_per_beat = 1
        fake = FakeDispatcher()
        fake.queue_route(_router_out(agent_ids=["pip"], event_kind="beat_continues"))
        fake.queue_agent("Pip polishes the bell")
        fake.queue_route(_router_out(agent_ids=["pip"], event_kind="beat_continues"))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="wait",
        ))

        assert result.ended_reason == "cascade_cap"
        assert result.events_closed == 2
        assert len(fake.route_calls) == 2
        assert len(fake.agent_calls) == 1

    def test_one_star_mission_batch_runs_to_existing_agent_cap(
        self,
        monkeypatch,
    ):
        from app.engine import one_star_adapter

        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.max_agent_cascades_per_beat = 2
        fake = FakeDispatcher()
        for _ in range(3):
            fake.queue_route(_router_out(
                agent_ids=["pip"],
                event_kind="beat_continues",
            ))
        fake.queue_agent("Pip advances the floor objective once.")
        fake.queue_agent("Pip advances the floor objective twice.")
        account = SimpleNamespace(
            state=SimpleNamespace(active_mission=object()),
        )
        monkeypatch.setattr(
            one_star_adapter,
            "one_star_should_autonomous_mission_batch_after_result",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(
            one_star_adapter,
            "load_one_star_account",
            lambda _ckpt: (object(), account),
        )

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="(defer)",
        ))

        assert result.ended_reason == "cascade_cap"
        assert result.events_closed == 3
        assert [
            call["character_id"] for call in fake.agent_calls
        ] == ["pip", "pip"]
        assert fake.continuation_calls == []
        assert len(fake.narrator_calls) == 1
        assert (
            fake.narrator_calls[0]["narration_mode"]
            == "compressed_sequence"
        )

    def test_one_star_mission_batch_is_one_grouped_visual_novel_segment(
        self,
        monkeypatch,
    ):
        from app.engine import one_star_adapter

        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.presentation_mode = "visual_novel"
        ckpt.session.config.settings.max_agent_cascades_per_beat = 2

        class VisualNovelDispatcher(FakeDispatcher):
            async def narrator_compose(self, **kwargs):
                envelope, entry = await super().narrator_compose(**kwargs)
                return VisualNovelNarratorOutput(
                    handoff=envelope.handoff,
                    handoff_reason=envelope.handoff_reason,
                    beats=[VisualNovelBeatPages(pages=[VisualNovelPage(
                        kind="narration",
                        text="The ambush resolves in one violent rush.",
                    )])],
                ), entry

        fake = VisualNovelDispatcher()
        fake.queue_route(_router_out(
            event_id="evt_ambush_opens",
            agent_ids=["pip"],
            event_kind="beat_continues",
        ))
        fake.queue_route(_router_out(
            event_id="evt_ambush_turns",
            agent_ids=["pip"],
            event_kind="beat_continues",
        ))
        fake.queue_route(_router_out(
            event_id="evt_ambush_closes",
            agent_ids=["pip"],
            event_kind="beat_continues",
        ))
        fake.queue_agent("Pip makes one scene-sized survival move.")
        fake.queue_agent("Pip finishes the initial attackers.")
        account = SimpleNamespace(
            state=SimpleNamespace(active_mission=object()),
        )
        monkeypatch.setattr(
            one_star_adapter,
            "one_star_should_autonomous_mission_batch_after_result",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(
            one_star_adapter,
            "load_one_star_account",
            lambda _ckpt: (object(), account),
        )

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="(defer)",
        ))

        render = result.visual_novel_renders["alice"]
        assert len(render.segments) == 1
        assert render.segments[0].rendered_event_ids == [
            "evt_ambush_opens",
            "evt_ambush_turns",
            "evt_ambush_closes",
        ]
        assert fake.narrator_calls[0]["narration_mode"] == (
            "compressed_sequence"
        )

    def test_one_star_mission_batch_tries_one_targetless_continuation(
        self,
        monkeypatch,
    ):
        from app.engine import one_star_adapter

        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(event_kind="state_change"))
        fake.queue_route(_router_out(event_kind="state_change"))
        account = SimpleNamespace(
            state=SimpleNamespace(active_mission=object()),
        )
        monkeypatch.setattr(
            one_star_adapter,
            "one_star_should_autonomous_mission_batch_after_result",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(
            one_star_adapter,
            "load_one_star_account",
            lambda _ckpt: (object(), account),
        )

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I watch the floor team.",
        ))

        assert result.events_closed == 2
        assert len(fake.route_calls) == 1
        assert len(fake.continuation_calls) == 1
        assert fake.continuation_calls[0]["actor_id"] == ""
        assert result.event_actor_ids == ["alice", ""]
        assert fake.agent_calls == []
        assert len(fake.narrator_calls) == 1

    def test_cat_i_dispatches_only_first_agent_pick_before_router_roundtrip(
        self,
    ):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(agent_ids=["pip", "bob"], event_kind="beat_continues"))
        fake.queue_agent("Pip polishes the bell")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
        _queue_narrator_continue(fake)

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
        assert fake.route_calls[1]["actor_id"] == "pip"
        assert fake.route_calls[1]["intention"] == "Pip polishes the bell"

    def test_router_can_continue_to_next_agent_pick_after_canonicalizing_first(
        self,
    ):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(agent_ids=["pip", "bob"], event_kind="beat_continues"))
        fake.queue_agent("Pip polishes the bell")
        fake.queue_route(_router_out(agent_ids=["bob"], event_kind="beat_continues"))
        fake.queue_agent("Bob studies the latch")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
        _queue_narrator_continue(fake, count=2)

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
        assert [
            call["actor_id"]
            for call in fake.route_calls[1:]
        ] == ["pip", "bob"]

    def test_mutated_targetless_beat_continues_fails_before_broadcast(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        invalid = _router_out(event_kind="cascade_exhausted")
        invalid.event_kind = "beat_continues"
        fake.queue_route(invalid)

        with pytest.raises(
            RuntimeError,
            match="beat_continues requires at least one next_output",
        ):
            asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="wait",
            ))

        assert fake.continuation_calls == []
        assert ckpt.canonical_events == []

    def test_next_output_spawn_materializes_before_broadcast_and_dispatch(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            event_id="evt_scout_arrives",
            agent_ids=["new_scout"],
            observer_ids=["alice", "new_scout"],
            event_kind="beat_continues",
            facts=[ObservableFact.all("A scout steps through the gate.")],
            spawn=[
                SpawnRequest(
                    character_id="new_scout",
                    seed={
                        "role": "rain-soaked scout",
                        "reason": "the gate needs a warning",
                        "location": "gatehouse",
                        "objectives": ["warn Alice"],
                        "knowledge_tier": 0,
                    },
                ),
            ],
        ))
        fake.queue_agent("The scout delivers the warning.")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I wait for news.",
        ))

        scout = next(c for c in ckpt.characters if c.character_id == "new_scout")
        assert result.ended_reason == "cascade_exhausted"
        assert fake.continuation_calls == []
        assert fake.materialize_calls[0]["character_ids"] == ["new_scout"]
        assert fake.agent_calls[0]["character_id"] == "new_scout"
        assert fake.agent_character_exists == [True]
        assert "A scout steps through the gate." in scout.pending_observations

    def test_continuation_still_cannot_open_cat_ii(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(event_kind="state_change"))
        fake.queue_narrator(
            handoff="continue",
            reason="The visible sequence still has established motion.",
            text="DISCARDED CANDIDATE",
        )
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip"],
            observer_ids=["alice", "pip"],
            event_kind="cat_ii_open",
        ))

        with pytest.raises(RuntimeError, match="continuation opened Cat II"):
            asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="wait",
            ))

        assert fake.materialize_calls == []
        assert fake.agent_calls == []


class TestCatIIBeat:
    def test_cat_ii_open_rejects_durable_participant_outcome(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip"],
            observer_ids=["alice", "pip"],
            event_kind="cat_ii_open",
            location_updates=[{
                "character_id": "alice",
                "location_label": "past_pip",
            }],
        ))

        with pytest.raises(
            ValueError,
            match="durable state change.*alice",
        ):
            asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="I shove past Pip.",
            ))

        alice = next(
            character for character in ckpt.characters
            if character.character_id == "alice"
        )
        assert alice.location == "gatehouse"
        assert ckpt.session.open_cat_ii_events == []
        assert ckpt.canonical_events == []

    def test_direct_and_autonomous_submissions_share_cat_ii_open_contract(self):
        submission = "Bob reaches for Alice's letter."

        direct_ckpt = _ckpt({"alice": "1", "bob": "2"})
        direct = FakeDispatcher()
        direct.queue_route(_router_out(
            requires_responders=True,
            required_responders=["alice"],
            observer_ids=["alice", "bob"],
            event_kind="cat_ii_open",
        ))
        direct_result = asyncio.run(run_beat(
            ckpt=direct_ckpt,
            dispatcher=direct,
            actor_id="bob",
            intention=submission,
        ))

        autonomous_ckpt = _ckpt({"alice": "1"})
        autonomous = FakeDispatcher()
        autonomous.queue_route(_router_out(
            agent_ids=["bob"],
            observer_ids=["alice", "bob"],
            event_kind="beat_continues",
        ))
        autonomous.queue_agent(submission)
        autonomous.queue_route(_router_out(
            requires_responders=True,
            required_responders=["alice"],
            observer_ids=["alice", "bob"],
            event_kind="cat_ii_open",
        ))
        _queue_narrator_continue(autonomous)
        autonomous_result = asyncio.run(run_beat(
            ckpt=autonomous_ckpt,
            dispatcher=autonomous,
            actor_id="alice",
            intention="I wait with the letter in hand.",
        ))

        assert direct_result.ended_reason == autonomous_result.ended_reason == (
            "cat_ii_pending"
        )
        direct_open = direct_ckpt.session.open_cat_ii_events[0]
        autonomous_open = autonomous_ckpt.session.open_cat_ii_events[0]
        assert (
            direct_open.initiator_id,
            direct_open.initiator_intention,
            direct_open.required_responders,
        ) == (
            autonomous_open.initiator_id,
            autonomous_open.initiator_intention,
            autonomous_open.required_responders,
        ) == (
            "bob",
            submission,
            ["alice"],
        )
        assert autonomous.route_calls[1]["cat_ii_event"] is None

    def test_agent_submission_opens_cat_ii_with_exact_submission_text(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            agent_ids=["bob"],
            observer_ids=["alice", "bob"],
            event_kind="beat_continues",
        ))
        agent_submission = "Bob reaches for Alice's letter."
        fake.queue_agent(agent_submission)
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["alice"],
            observer_ids=["alice", "bob"],
            event_kind="cat_ii_open",
        ))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I wait with the letter in hand.",
        ))

        assert result.ended_reason == "cat_ii_pending"
        assert [call["actor_id"] for call in fake.route_calls] == [
            "alice",
            "bob",
        ]
        assert fake.route_calls[1]["intention"] == agent_submission
        open_event = ckpt.session.open_cat_ii_events[0]
        assert open_event.initiator_id == "bob"
        assert open_event.initiator_intention == agent_submission
        assert open_event.required_responders == ["alice"]
        assert ckpt.session.active_act_slots["alice"].reason == (
            "cat_ii_responder"
        )
        assert [event.event_kind for event in ckpt.canonical_events] == [
            "beat_continues",
            "cat_ii_open",
        ]

    def test_agent_submission_cat_ii_collects_agent_and_resolves_once(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            agent_ids=["bob"],
            observer_ids=["alice", "bob"],
            event_kind="beat_continues",
        ))
        initiator_submission = "Bob reaches for Pip's letter."
        responder_submission = "Pip pulls the letter against his chest."
        fake.queue_agent(initiator_submission)
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip"],
            observer_ids=["alice", "bob", "pip"],
            event_kind="cat_ii_open",
        ))
        fake.queue_agent(responder_submission)
        fake.queue_route(_router_out(
            observer_ids=["alice"],
            event_kind="cascade_exhausted",
        ))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I watch Bob and Pip.",
        ))

        assert result.ended_reason == "cat_ii_resolution"
        assert ckpt.session.open_cat_ii_events == []
        assert [call["actor_id"] for call in fake.route_calls] == [
            "alice",
            "bob",
            "bob",
        ]
        resolution_call = fake.route_calls[2]
        assert resolution_call["intention"] == initiator_submission
        assert resolution_call["cat_ii_event"].initiator_intention == (
            initiator_submission
        )
        assert resolution_call["cat_ii_event"].collected_intentions == {
            "pip": responder_submission,
        }
        assert [event.event_kind for event in ckpt.canonical_events] == [
            "beat_continues",
            "cat_ii_open",
            "cascade_exhausted",
        ]

    @pytest.mark.parametrize("ruleset_id", ["", "dnd5e_basic"])
    def test_all_agent_cat_ii_resolution_lets_human_observer_continue(
        self, ruleset_id,
    ):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.ruleset_id = ruleset_id
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            event_id="evt_agent_contest_open",
            requires_responders=True,
            required_responders=["pip"],
            observer_ids=["alice", "bob", "pip"],
            event_kind="cat_ii_open",
        ))
        fake.queue_agent("Pip braces against Bob's shove.")
        fake.queue_route(_router_out(
            event_id="evt_agent_contest_resolution",
            observer_ids=["alice", "bob", "pip"],
            event_kind="cat_ii_resolution",
        ))
        fake.queue_route(_router_out(
            event_id="evt_after_agent_contest",
            observer_ids=["alice", "bob", "pip"],
            event_kind="cascade_exhausted",
        ))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="bob",
            intention="Bob shoves Pip away from the door.",
        ))

        assert result.ended_reason == "cascade_exhausted"
        assert result.events_closed == 3
        assert len(fake.continuation_calls) == 1
        assert (
            fake.continuation_calls[0]["prior_result"].event_kind
            == "cat_ii_resolution"
        )
        assert [
            call["handoff_policy"] for call in fake.narrator_calls
        ] == ["candidate", "candidate"]
        assert [
            len(call["buffered_events"]) for call in fake.narrator_calls
        ] == [2, 3]

    def test_cat_ii_with_agent_responder_resolves_inline(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip"],
            event_kind="cat_ii_open",
            effective_at_s=100,
            duration_s=30,
        ))
        fake.queue_agent("Pip dodges")
        fake.queue_route(_router_out(
            event_kind="cat_ii_resolution",
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
        assert fake.narrator_calls[0]["handoff_policy"] == "forced"
        assert fake.continuation_calls == []

    def test_cat_ii_autonomous_responders_use_same_snapshot_and_order(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.characters.append(
            CharacterRecord(
                character_id="quill",
                name="Quill",
                location="gatehouse",
                public_sheet=PublicSheet(role="scribe"),
            )
        )
        fake = _ConcurrentCatIIResponderFake()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip", "quill"],
            observer_ids=["alice", "pip", "quill"],
            event_kind="cat_ii_open",
        ))
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))

        async def exercise():
            task = asyncio.create_task(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="I challenge both witnesses.",
            ))
            await asyncio.wait_for(fake.started.wait(), timeout=1)
            assert [call["character_id"] for call in fake.agent_calls] == [
                "pip",
                "quill",
            ]
            assert fake.branch_event_ids["pip"]
            assert fake.branch_event_ids["quill"] == fake.branch_event_ids[
                "pip"
            ]
            assert fake.branch_collected_intentions == {
                "pip": {},
                "quill": {},
            }
            fake.release.set()
            return await task

        result = asyncio.run(exercise())

        assert result.ended_reason == "cat_ii_resolution"
        assert ckpt.session.open_cat_ii_events == []
        assert [
            message.content
            for message in ckpt.character_conversations["pip"]
        ] == ["draft from pip"]
        assert [
            message.content
            for message in ckpt.character_conversations["quill"]
        ] == ["draft from quill"]
        assert fake.route_calls[1]["cat_ii_event"].collected_intentions == {
            "pip": "pip commits to a distinct response.",
            "quill": "quill commits to a distinct response.",
        }

    def test_cat_ii_autonomous_responder_failure_rolls_back_collection(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.characters.append(
            CharacterRecord(
                character_id="quill",
                name="Quill",
                location="gatehouse",
                public_sheet=PublicSheet(role="scribe"),
            )
        )
        fake = _ConcurrentCatIIResponderFake(failing_id="quill")
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip", "quill"],
            observer_ids=["alice", "pip", "quill"],
            event_kind="cat_ii_open",
        ))

        with pytest.raises(RuntimeError, match="responder failed: quill"):
            asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="I challenge both witnesses.",
            ))

        assert ckpt.canonical_events == []
        assert ckpt.session.open_cat_ii_events == []
        assert "pip" not in ckpt.session.active_act_slots
        assert "quill" not in ckpt.session.active_act_slots
        assert ckpt.character_conversations == {}

    def test_all_agent_cat_ii_resolution_still_yields_to_bound_next_output(
        self,
    ):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip"],
            observer_ids=["alice", "bob", "pip"],
            event_kind="cat_ii_open",
        ))
        fake.queue_agent("Pip steps back from Bob.")
        fake.queue_route(_router_out(
            agent_ids=["alice"],
            observer_ids=["alice", "bob", "pip"],
            event_kind="cat_ii_resolution",
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="bob",
            intention="Bob orders Pip away, then looks to Alice.",
        ))

        assert result.ended_reason == "awaiting_player_turn"
        assert ckpt.session.open_cat_ii_events == []
        assert ckpt.canonical_events[-1].next_output_character_ids == [
            "alice"
        ]
        assert [call["character_id"] for call in fake.agent_calls] == ["pip"]
        assert fake.narrator_calls[0]["handoff_policy"] == "forced"
        assert fake.continuation_calls == []

    def test_inline_cat_ii_resolution_yields_to_bound_semantic_next_output(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip"],
            observer_ids=["alice", "bob", "pip"],
            event_kind="cat_ii_open",
        ))
        fake.queue_agent("Pip pulls the letter against his chest.")
        fake.queue_route(_router_out(
            agent_ids=["bob"],
            observer_ids=["alice", "bob", "pip"],
            event_kind="beat_continues",
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I reach for Pip's letter.",
        ))

        assert result.ended_reason == "awaiting_player_turn"
        assert ckpt.session.open_cat_ii_events == []
        assert ckpt.canonical_events[-1].next_output_character_ids == ["bob"]
        assert [call["character_id"] for call in fake.agent_calls] == ["pip"]
        assert len(fake.route_calls) == 2

    def test_cat_ii_materializes_spawned_responder_before_dispatch(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["hidden_lookout"],
            observer_ids=["alice", "hidden_lookout"],
            event_kind="cat_ii_open",
            spawn=[
                SpawnRequest(
                    character_id="hidden_lookout",
                    seed={
                        "role": "hidden lookout",
                        "reason": "spotted the stealth attempt",
                        "location": "mill hedge",
                        "objectives": ["track intruders"],
                        "knowledge_tier": 0,
                    },
                ),
            ],
        ))
        fake.queue_agent("The lookout freezes and signals.")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))

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
            event_kind="cat_ii_open",
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
            event_kind="cat_ii_open",
        ))
        fake.queue_agent("Pip dodges")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))

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
            event_kind="cat_ii_open",
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

    def test_autonomous_actor_submission_can_start_dnd_combat(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            agent_ids=["bob"],
            observer_ids=["alice", "bob"],
            event_kind="beat_continues",
        ))
        submission = "Bob lunges at Alice with a knife."
        fake.queue_agent(submission)
        fake.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["bob", "alice"],
            facts=[ObservableFact.all("Bob commits to an attack against Alice.")],
        ))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I watch Bob's hands.",
        ))

        assert result.ended_reason == "combat_started"
        assert ckpt.session.active_combat is not None
        assert [call["actor_id"] for call in fake.route_calls] == [
            "alice",
            "bob",
        ]
        bob = next(
            combatant
            for combatant in ckpt.session.active_combat.combatants
            if combatant.character_id == "bob"
        )
        assert bob.pending_initiating_action == submission

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
            interaction_mode="narrative",
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
            event_kind="cat_ii_open",
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
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))

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
            event_kind="beat_continues",
        ))
        fake.queue_agent("Pip answers the warning.")
        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="pip",
            intention="I try not to show fear",
            cat_ii_event_id=evt.event_id,
        ))

        assert result.ended_reason == "cascade_exhausted"
        assert ckpt.session.open_cat_ii_events == []
        assert fake.agent_calls[0]["character_id"] == "pip"
        assert fake.route_calls[0]["actor_id"] == "alice"
        assert fake.route_calls[1]["actor_id"] == "pip"
        assert fake.route_calls[1]["intention"] == "Pip answers the warning."

    def test_cat_ii_resolution_yields_to_bound_semantic_next_output(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="pip",
            initiator_intention="Pip asks Alice to choose a route",
            required_responders=["alice"],
        )
        pin_cat_ii_responder(ckpt, "alice", evt.event_id)

        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            agent_ids=["bob"],
            observer_ids=["alice", "bob", "pip"],
            event_kind="beat_continues",
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I ask Bob to make the final call.",
            cat_ii_event_id=evt.event_id,
        ))

        assert result.ended_reason == "awaiting_player_turn"
        assert ckpt.session.open_cat_ii_events == []
        assert ckpt.canonical_events[-1].next_output_character_ids == ["bob"]
        assert fake.agent_calls == []
        assert len(fake.route_calls) == 1

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

        fake.queue_route(_router_out(event_kind="cascade_exhausted"))
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
        presented_facts: list[list[str]] = []

        class RecordingImageSink:
            async def start_render_candidate(self, **kwargs):
                checkpoint = kwargs["checkpoint"]
                event_ids = {
                    entry.event_id
                    for entries in kwargs["buffered_events_by_pov"].values()
                    for entry in entries
                }
                presented_facts.append([
                    fact.text
                    for event in checkpoint.canonical_events
                    if event.event_id in event_ids
                    for fact in event.canonical_event.observable_facts
                ])
                return None

            async def cancel_transaction(self, *_args, **_kwargs):
                return None

            async def commit_transaction(self, *_args, **_kwargs):
                return None

        install_closed_event_runtime(
            ckpt,
            ClosedEventRuntime(
                transaction_id="tx_harvest",
                source_turn_index=1,
                spawn_authoring=SpawnAuthoringCoordinator(object()),
                image_sink=RecordingImageSink(),
            ),
        )
        ckpt.characters.append(CharacterRecord(
            character_id="vex",
            name="Vex",
            public_sheet=PublicSheet(role="npc"),
            location="gatehouse",
        ))
        fake = FakeDispatcher()
        out = _router_out(event_kind="cascade_exhausted", agent_ids=["pip", "vex"])
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
        assert len(presented_facts) == 1
        assert any("[loadout" in fact for fact in presented_facts[0])
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        vex = next(c for c in ckpt.characters if c.character_id == "vex")
        for observer in (pip, vex):
            received = "\n".join(observer.pending_observations)
            assert "[loadout — Pip] Pip in patched leathers." in received
            assert "[loadout — Vex] Vex in midnight silk." in received
        assert fake.agent_calls == []

    def test_harvest_drops_human_targets(self):
        ckpt = _ckpt({"alice": "1"})
        out = _router_out(event_kind="cascade_exhausted", agent_ids=["alice"])
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
            event_kind="cascade_exhausted",
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

    def test_unclaimed_player_authored_reference_fails_before_mutation(self):
        ckpt = _ckpt()
        ckpt.characters.append(CharacterRecord(
            character_id="blank_arrival",
            name="the Newcomer",
            is_playable=True,
            player_slot_kind=PlayerSlotKind.player_authored,
        ))
        before = ckpt.model_dump()
        event = self._event(observer_ids=["blank_arrival"])

        with pytest.raises(
            RuntimeError,
            match="unclaimed player-authored seat.*observer=blank_arrival",
        ):
            broadcast_event(ckpt, event, actor_id="alice")

        assert ckpt.model_dump() == before

    def test_unclaimed_player_authored_id_cannot_enter_canonical_fact_text(self):
        ckpt = _ckpt()
        ckpt.characters.append(CharacterRecord(
            character_id="blank_arrival",
            name="the Newcomer",
            is_playable=True,
            player_slot_kind=PlayerSlotKind.player_authored,
        ))
        event = self._event(
            observer_ids=["alice"],
            facts=[ObservableFact.all("blank_arrival steps into the room.")],
        )

        with pytest.raises(
            RuntimeError,
            match="fact text=blank_arrival",
        ):
            broadcast_event(ckpt, event, actor_id="alice")

        assert ckpt.canonical_events == []

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

    def test_npc_actor_receives_visible_canonical_outcome(self):
        ckpt = _ckpt()
        event = self._event(observer_ids=["pip"])
        broadcast_event(ckpt, event, actor_id="pip")
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == ["Alice sets down a glass."]

    def test_npc_actor_receives_external_consequences_in_mixed_event(self):
        ckpt = _ckpt()
        event = self._event(
            observer_ids=["pip"],
            facts=[
                ObservableFact.all("Pip holds position in silence."),
                ObservableFact.all(
                    "A goblin breaks cover and throws a spear past Pip."
                ),
                ObservableFact.all(
                    "A second goblin closes and cuts Pip across the ribs."
                ),
            ],
        )

        broadcast_event(ckpt, event, actor_id="pip")

        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == [
            "  - Pip holds position in silence.\n"
            "  - A goblin breaks cover and throws a spear past Pip.\n"
            "  - A second goblin closes and cuts Pip across the ribs."
        ]

    def test_human_observer_gets_render_buffer_not_inbox(self):
        ckpt = _ckpt({"alice": "1"})
        event = self._event(observer_ids=["alice"])
        visible = broadcast_event(ckpt, event, actor_id="pip")
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        assert visible == ["alice"]
        assert alice.pending_observations == []
        assert ckpt.session.render_buffers["alice"][0].event_id == event.event_id

    def test_render_buffer_freezes_character_display_for_each_event(self):
        ckpt = _ckpt({"alice": "1"})
        pip = next(
            character
            for character in ckpt.characters
            if character.character_id == "pip"
        )
        pip.visuals.visual_novel_presentation.current_variant_key = "skeptical"
        first = self._with_updates(
            self._event(observer_ids=["alice"]),
            event_id="evt_first_display",
        )

        broadcast_event(ckpt, first, actor_id="pip")
        pip.visuals.visual_novel_presentation.current_variant_key = "happy"
        second = self._with_updates(
            self._event(observer_ids=["alice"]),
            event_id="evt_second_display",
        )
        broadcast_event(ckpt, second, actor_id="pip")

        buffered = ckpt.session.render_buffers["alice"]
        assert [entry.event_id for entry in buffered] == [
            "evt_first_display",
            "evt_second_display",
        ]
        assert [
            entry.sprite_variant_keys_by_character_id["pip"]
            for entry in buffered
        ] == ["skeptical", "happy"]

    def test_canonical_location_change_resets_display_but_same_location_does_not(
        self,
    ):
        ckpt = _ckpt()
        pip = next(
            character
            for character in ckpt.characters
            if character.character_id == "pip"
        )
        presentation = pip.visuals.visual_novel_presentation
        presentation.current_variant_key = "angry"
        presentation.scene_location = pip.location
        move = self._with_updates(
            self._event(observer_ids=["pip"]),
            location_updates=[{
                "character_id": "pip",
                "location_label": "archive",
            }],
        )

        broadcast_event(ckpt, move, actor_id="alice")

        assert pip.location == "archive"
        assert presentation.scene_location == "archive"
        assert presentation.current_variant_key == "neutral"

        presentation.current_variant_key = "happy"
        same_place = self._with_updates(
            self._event(observer_ids=["pip"]),
            location_updates=[{
                "character_id": "pip",
                "location_label": "archive",
            }],
        )
        broadcast_event(ckpt, same_place, actor_id="alice")
        assert presentation.current_variant_key == "happy"

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

    def test_sweep_resolves_event_with_empty_opened_at(self):
        # A hand-edited or migration-authored open Cat II with no opened_at
        # used to raise in _parse_iso("") and get swallowed, wedging every
        # other player's /act behind the pin forever. It must auto-resolve.
        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.cat_ii_human_timeout_seconds = 60
        evt = open_cat_ii(
            ckpt,
            initiator_id="pip",
            initiator_intention="punch",
            required_responders=["alice"],
        )
        evt.opened_at = ""

        ids = sweep_stale_cat_ii_pins(ckpt)

        assert ids == [evt.event_id]
        assert "alice" in evt.swept_responders

    def test_combat_reaction_pin_auto_passed_after_timeout(self):
        # An AFK human holding an optional combat-reaction pin must not
        # wedge initiative forever (the only prior sweep handled Cat II
        # only). The stale pin is released so the table can advance.
        from datetime import datetime, timedelta, timezone
        from app.engine.turn_loop import (
            pin_combat_reaction,
            sweep_stale_combat_reaction_pins,
        )

        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.cat_ii_human_timeout_seconds = 1
        assert pin_combat_reaction(ckpt, "alice", "evt_trigger")
        ckpt.session.active_act_slots["alice"].claimed_at = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).isoformat()

        released = sweep_stale_combat_reaction_pins(ckpt)

        assert released == ["alice"]
        assert "alice" not in ckpt.session.active_act_slots

    def test_fresh_combat_reaction_pin_not_swept(self):
        from app.engine.turn_loop import (
            pin_combat_reaction,
            sweep_stale_combat_reaction_pins,
        )

        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.cat_ii_human_timeout_seconds = 3600
        assert pin_combat_reaction(ckpt, "alice", "evt_trigger")

        released = sweep_stale_combat_reaction_pins(ckpt)

        assert released == []
        assert "alice" in ckpt.session.active_act_slots

    def test_combat_reaction_pin_with_empty_claimed_at_is_swept(self):
        from app.engine.turn_loop import (
            pin_combat_reaction,
            sweep_stale_combat_reaction_pins,
        )

        ckpt = _ckpt({"alice": "1"})
        ckpt.session.config.settings.cat_ii_human_timeout_seconds = 3600
        assert pin_combat_reaction(ckpt, "alice", "evt_trigger")
        ckpt.session.active_act_slots["alice"].claimed_at = ""

        released = sweep_stale_combat_reaction_pins(ckpt)

        assert released == ["alice"]
        assert "alice" not in ckpt.session.active_act_slots


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

    def test_unobserved_remote_pick_requires_actor_offstage_policy(self):
        from app.engine.turn_loop import _filter_routed_agents_for_dispatch

        ckpt = _ckpt({"alice": "1"})
        ckpt.characters.extend(
            [
                CharacterRecord(
                    character_id="offstage_allowed",
                    name="Offstage Allowed",
                    location="distant_archive",
                    public_sheet=PublicSheet(role="remote watcher"),
                    actor=ActorRecord(may_act_offstage=True),
                ),
                CharacterRecord(
                    character_id="offstage_blocked",
                    name="Offstage Blocked",
                    location="distant_archive",
                    public_sheet=PublicSheet(role="local guard"),
                ),
            ]
        )
        event = _router_out(event_kind="state_change", observer_ids=[])

        assert _filter_routed_agents_for_dispatch(
            ckpt,
            ["offstage_allowed", "offstage_blocked"],
            event=event,
        ) == ["offstage_allowed"]


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
        out = _router_out(agent_ids=["pip"], event_kind="beat_continues")
        data = out.model_dump()
        data["agent_responder_picks"] = ["pip", "offstage_npc"]
        with pytest.raises(ValueError, match="Extra inputs"):
            EventRouterOutput.model_validate(data)

    def test_router_target_projection_uses_runtime_frame_semantics(self):
        out = _router_out(agent_ids=["pip"], event_kind="beat_continues")
        assert targets_from_router_output(
            out,
            player_ids={"alice"},
            agent_ids=["pip"],
        )[0].frame == "foreground"
        assert targets_from_router_output(
            out,
            player_ids=set(),
            agent_ids=["pip"],
        )[0].frame == "private"

        offstage = _router_out(event_kind="state_change")
        assert targets_from_router_output(
            offstage,
            player_ids={"alice"},
            agent_ids=["offstage_npc"],
        )[0].frame == "background"

        public_fact = _router_out(
            event_kind="public_fact",
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
        )
        assert targets_from_router_output(
            public_fact,
            player_ids={"alice"},
            agent_ids=["pip"],
        )[0].frame == "background"

        departing = _router_out(
            agent_ids=["pip"],
            event_kind="beat_continues",
            observer_ids=["alice", "pip"],
            location_updates=[
                {"character_id": "pip", "location_label": "archive"}
            ],
        )
        assert targets_from_router_output(
            departing,
            player_ids={"alice"},
            agent_ids=["pip"],
        )[0].frame == "background"

    def test_unknown_event_kind_coerced_to_cascade_exhausted(self):
        out = _router_out(event_kind="cascade_exhausted")
        data = out.model_dump()
        data["event_kind"] = "location-transition"
        rebuilt = EventRouterOutput.model_validate(data)
        assert rebuilt.event_kind == "cascade_exhausted"

    def test_observation_harvest_is_not_an_open_cascade(self):
        out = _router_out(event_kind="cascade_exhausted", agent_ids=["pip"])
        data = out.model_dump()
        data["event_kind"] = "observation_harvest"
        for observer in data["observers"]:
            if observer["character_id"] == "pip":
                observer["routing_role"] = "perception_enrichment"
        rebuilt = EventRouterOutput.model_validate(data)
        assert rebuilt.event_kind != "beat_continues"


class TestEndBeatFanout:
    @pytest.mark.parametrize("presentation_mode", ["prose", "visual_novel"])
    @pytest.mark.parametrize(
        "pov_ids",
        [("alice",), ("alice", "bob")],
        ids=["single_pov", "multi_pov"],
    )
    def test_forced_continue_rolls_back_entire_render_batch_atomically(
        self,
        presentation_mode,
        pov_ids,
    ):
        from app.engine.turn_loop import _end_beat

        ckpt = _ckpt({
            character_id: str(index)
            for index, character_id in enumerate(pov_ids, start=1)
        })
        ckpt.session.config.settings.presentation_mode = presentation_mode
        event = _router_out(
            event_id="evt_forced_boundary",
            event_kind="cascade_exhausted",
            observer_ids=list(pov_ids),
            facts=[ObservableFact.all("The gate locks open.")],
        )
        ckpt.canonical_events.append(event)
        for character_id in pov_ids:
            append_to_render_buffer(
                ckpt,
                character_id,
                event.event_id,
                "direct",
            )
            ckpt.narrator_conversations[character_id] = [ConversationMessage(
                role="assistant",
                content=f"prior {character_id} render",
            )]
        ckpt.session.visual_introductions = {
            character_id: ["existing_npc"] for character_id in pov_ids
        }
        expected_history = {
            character_id: [
                message.model_dump(mode="json")
                for message in ckpt.narrator_conversations[character_id]
            ]
            for character_id in pov_ids
        }
        expected_introductions = {
            character_id: ["existing_npc"] for character_id in pov_ids
        }

        cancelled_transactions: list[str] = []

        class RecordingImageSink:
            async def start_render_candidate(self, **_kwargs):
                return "imgtx_forced_boundary"

            async def cancel_transaction(self, transaction_id, **_kwargs):
                cancelled_transactions.append(transaction_id)

            async def commit_transaction(self, *_args, **_kwargs):
                return None

        image_runtime = ClosedEventRuntime(
            transaction_id="tx_forced_boundary",
            source_turn_index=1,
            spawn_authoring=SpawnAuthoringCoordinator(object()),
            image_sink=RecordingImageSink(),
        )
        install_closed_event_runtime(ckpt, image_runtime)

        class MutatingDispatcher(FakeDispatcher):
            async def narrator_compose(self, **kwargs):
                self.narrator_calls.append(kwargs)
                character_id = kwargs["character_id"]
                kwargs["ckpt"].narrator_conversations[character_id].append(
                    ConversationMessage(
                        role="assistant",
                        content="speculative narrator mutation",
                    )
                )
                kwargs["ckpt"].session.visual_introductions[
                    character_id
                ].append("pip")
                handoff = "continue" if character_id == "alice" else "render"
                if presentation_mode == "visual_novel":
                    envelope = VisualNovelNarratorOutput(
                        handoff=handoff,
                        handoff_reason="Forced-boundary contract test.",
                        beats=(
                            []
                            if handoff == "continue"
                            else [VisualNovelBeatPages(pages=[
                                VisualNovelPage(
                                    kind="narration",
                                    text="The gate locks open.",
                                )
                            ])]
                        ),
                    )
                else:
                    envelope = NarratorFinalOutput(
                        handoff=handoff,
                        handoff_reason="Forced-boundary contract test.",
                        final_text=(
                            "" if handoff == "continue" else "The gate locks open."
                        ),
                    )
                return envelope, TranscriptEntry(
                    user=kwargs.get("user_input", ""),
                    assistant=(
                        "" if handoff == "continue" else "The gate locks open."
                    ),
                )

        fake = MutatingDispatcher()

        with pytest.raises(ValueError, match="forced handoff policy"):
            asyncio.run(_end_beat(
                ckpt,
                fake,
                ended_reason="cascade_exhausted",
                events_closed=1,
                event_actor_ids=["alice"],
            ))

        assert {
            character_id: [
                message.model_dump(mode="json")
                for message in ckpt.narrator_conversations[character_id]
            ]
            for character_id in pov_ids
        } == expected_history
        assert ckpt.session.visual_introductions == expected_introductions
        assert all(
            [entry.event_id for entry in ckpt.session.render_buffers[character_id]]
            == [event.event_id]
            for character_id in pov_ids
        )
        assert cancelled_transactions == ["imgtx_forced_boundary"]
        assert image_runtime.accepted_image_transaction_ids == set()

    def test_buffered_humans_render_in_parallel(self):
        import time
        from app.engine.turn_loop import _end_beat

        ckpt = _ckpt({"alice": "1", "bob": "2"})
        event = _router_out(event_kind="cascade_exhausted")
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
            ended_reason="cascade_exhausted",
            events_closed=1,
            event_actor_ids=["alice"],
        ))
        elapsed = time.monotonic() - t0

        assert set(result.renders) == {"alice", "bob"}
        assert len(fake.narrator_calls) == 2
        assert elapsed < 0.18

    def test_only_acting_pov_retains_the_raw_player_submission(self):
        from app.engine.turn_loop import _end_beat

        ckpt = _ckpt({"alice": "1", "bob": "2"})
        event = _router_out(
            event_kind="cascade_exhausted",
            observer_ids=["alice", "bob"],
        )
        ckpt.canonical_events.append(event)
        append_to_render_buffer(ckpt, "alice", event.event_id, "direct")
        append_to_render_buffer(ckpt, "bob", event.event_id, "direct")

        asyncio.run(_end_beat(
            ckpt,
            FakeDispatcher(),
            ended_reason="cascade_exhausted",
            events_closed=1,
            event_actor_ids=["alice"],
            acting_player_id="alice",
            acting_player_input="I whisper the password.",
        ))

        assert [
            (message.role, message.content)
            for message in ckpt.narrator_conversations["alice"]
            if message.role == "user"
        ] == [("user", "I whisper the password.")]
        assert all(
            message.role == "assistant"
            for message in ckpt.narrator_conversations["bob"]
        )


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


class TestHumanNextOutputYield:
    """Multiplayer turn ownership: when the router wants a co-present human to
    respond next, the beat must yield to that human's own /act instead of an
    agent turn or a narrator continuation voicing them."""

    def test_beat_yields_when_router_routes_a_human_next(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        fake = FakeDispatcher()
        # Alice (human) addresses Bob (human); router keeps the beat open with
        # Bob as next output. Bob must NOT be voiced by the engine.
        fake.queue_route(_router_out(
            agent_ids=["bob"],
            event_kind="beat_continues",
            facts=[ObservableFact.all("Alice grins and nudges Bob's elbow.")],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="I nudge Bob's elbow and grin.",
        ))

        assert result.ended_reason == "awaiting_player_turn"
        # The beat yielded: no narrator continuation and no agent turn for Bob.
        assert fake.continuation_calls == []
        assert fake.agent_calls == []
        # Only Alice's own action was canonicalized; Bob was not spoken for.
        assert len(ckpt.canonical_events) == 1
        assert "bob" in result.renders  # Bob still perceives Alice's action.

    def test_response_requested_dispatches_npc_when_narrator_defers(self):
        ckpt = _ckpt({"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            event_kind="response_requested",
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
        ))
        fake.queue_agent("I step through the gate.")
        fake.queue_route(_router_out(
            event_kind="cascade_exhausted",
            observer_ids=["alice", "pip"],
        ))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="Pip, go first.",
        ))

        assert result.ended_reason == "cascade_exhausted"
        assert [call["character_id"] for call in fake.agent_calls] == ["pip"]
        assert len(ckpt.canonical_events) == 2

    def test_response_requested_yields_to_a_bound_human(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            event_kind="response_requested",
            agent_ids=["bob"],
            observer_ids=["alice", "bob"],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="Bob, are you ready?",
        ))

        assert result.ended_reason == "awaiting_player_turn"
        assert fake.agent_calls == []
        assert fake.continuation_calls == []

    def test_first_bound_human_target_wins_over_later_npc(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            event_kind="response_requested",
            agent_ids=["bob", "pip"],
            observer_ids=["alice", "bob", "pip"],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="Bob and Pip, decide who enters.",
        ))

        assert result.ended_reason == "awaiting_player_turn"
        assert fake.agent_calls == []

    def test_first_npc_target_runs_before_later_bound_human(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            event_kind="response_requested",
            agent_ids=["pip", "bob"],
            observer_ids=["alice", "pip", "bob"],
        ))
        fake.queue_agent("I will enter.")
        fake.queue_route(_router_out(
            event_kind="cascade_exhausted",
            observer_ids=["alice", "pip", "bob"],
        ))
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="Pip and Bob, decide who enters.",
        ))

        assert result.ended_reason == "cascade_exhausted"
        assert [call["character_id"] for call in fake.agent_calls] == ["pip"]

    def test_silent_first_npc_falls_through_to_later_bound_human(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            event_kind="response_requested",
            agent_ids=["pip", "bob"],
            observer_ids=["alice", "pip", "bob"],
        ))
        fake.queue_agent("")
        _queue_narrator_continue(fake)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="Pip and Bob, tell me what you decide.",
        ))

        assert result.ended_reason == "awaiting_player_turn"
        assert [call["character_id"] for call in fake.agent_calls] == ["pip"]
        assert len(fake.route_calls) == 1
        assert fake.continuation_calls == []

    def test_unbound_player_authored_target_fails_loudly(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.characters.append(CharacterRecord(
            character_id="newcomer",
            name="the Newcomer",
            is_playable=True,
            player_slot_kind=PlayerSlotKind.player_authored,
        ))
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            event_kind="response_requested",
            agent_ids=["newcomer"],
            observer_ids=["alice", "newcomer"],
        ))

        with pytest.raises(
            RuntimeError,
            match="unclaimed player-authored seat",
        ):
            asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="Newcomer, answer me.",
            ))

        assert fake.agent_calls == []


class TestNarratorHandoff:
    def test_candidate_context_separates_defer_frequency_from_involvement(self):
        ckpt = _ckpt({"alice": "1"})
        ckpt.narrator_conversations["alice"] = [
            ConversationMessage(role="user", content="I ask Pip for the map."),
            ConversationMessage(role="assistant", content="Pip looks up."),
            ConversationMessage(role="user", content="(defer)"),
            ConversationMessage(role="assistant", content="Pip considers."),
            ConversationMessage(role="user", content="(defer)"),
            ConversationMessage(role="assistant", content="Pip reaches inside."),
        ]
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            event_id="evt_pip_next",
            event_kind="response_requested",
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
            facts=[ObservableFact.all("Alice asks Pip for a direct answer.")],
        ))
        fake.queue_agent("Pip answers plainly.")
        fake.queue_route(_router_out(
            event_id="evt_pip_answer",
            event_kind="cascade_exhausted",
            observer_ids=["alice", "pip"],
            facts=[ObservableFact.all("Pip answers plainly.")],
        ))
        fake.queue_narrator(
            handoff="render",
            reason="The question is a useful boundary.",
            text="Pip looks ready to answer.",
        )

        asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="alice",
            intention="Pip, answer me plainly.",
        ))

        context = fake.narrator_calls[0]["handoff_context"]
        assert "Meaningful player-owned response available now: no." in context
        assert "Autonomous character response selected next: yes." in context
        assert "2 of the last 4 player submissions were (defer)" in context
        assert (
            "1 of the last 3 prior player submissions were substantive" in context
        )
        assert "I ask Pip for the map" not in context

    def test_pending_retry_anchor_uses_the_selected_gate_pov(self):
        ckpt = _ckpt({"alice": "1", "bob": "2"})
        prior = _router_out(
            event_id="evt_bob_buffer",
            event_kind="cascade_exhausted",
            observer_ids=["bob"],
            facts=[ObservableFact.only("Bob hears a distant bell.", ["bob"])],
        )
        broadcast_event(ckpt, prior, actor_id="pip")

        class FailingGateDispatcher(FakeDispatcher):
            async def narrator_compose(self, **kwargs):
                if kwargs["character_id"] == "alice":
                    self.narrator_calls.append(kwargs)
                    raise RuntimeError("alice narrator offline")
                return await super().narrator_compose(**kwargs)

        fake = FailingGateDispatcher()
        fake.queue_route(_router_out(
            event_id="evt_alice_gate",
            event_kind="cascade_exhausted",
            observer_ids=["alice"],
            facts=[ObservableFact.only("Alice reaches the lift.", ["alice"])],
        ))
        persisted = []
        fake.persist_pending_narrator_render = lambda value: persisted.append(
            value.session.pending_narrator_render
        )

        with pytest.raises(RuntimeError, match="alice narrator offline"):
            asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="I take the lift.",
            ))

        pending = ckpt.session.pending_narrator_render
        assert pending is not None
        assert persisted == [pending]
        assert list(ckpt.session.render_buffers) == ["bob", "alice"]
        assert pending.soft_handoff_candidate is True
        assert pending.handoff_event_id == "evt_alice_gate"

    def test_continue_retains_full_batch_and_discards_candidate_prose(
        self, caplog,
    ):
        ckpt = _ckpt({"alice": "1"})
        image_batches: list[list[str]] = []
        cancelled_transactions: list[str] = []

        class RecordingImageSink:
            async def start_render_candidate(self, **kwargs):
                image_batches.append([
                    entry.event_id
                    for entry in kwargs["buffered_events_by_pov"]["alice"]
                ])
                return f"imgtx_{len(image_batches)}"

            async def cancel_transaction(self, transaction_id, **_kwargs):
                cancelled_transactions.append(transaction_id)

            async def commit_transaction(self, *_args, **_kwargs):
                return None

        image_runtime = ClosedEventRuntime(
            transaction_id="tx_handoff",
            source_turn_index=1,
            spawn_authoring=SpawnAuthoringCoordinator(object()),
            image_sink=RecordingImageSink(),
        )
        install_closed_event_runtime(ckpt, image_runtime)
        ckpt.session.open_commitments = [OpenCommitment(
            commitment_id="commit_wait",
            description="Alice waits until the gate finishes opening.",
            actor_ids=["alice"],
            opened_event_id="evt_prior",
        )]

        class VisualNovelDispatcher(FakeDispatcher):
            async def narrator_compose(self, **kwargs):
                envelope, entry = await super().narrator_compose(**kwargs)
                return VisualNovelNarratorOutput(
                    handoff=envelope.handoff,
                    handoff_reason=envelope.handoff_reason,
                    beats=(
                        [
                            VisualNovelBeatPages(pages=[
                                VisualNovelPage(
                                    kind="narration",
                                    text=(
                                        "The gate starts to rise."
                                        if index == 0
                                        else envelope.final_text
                                    ),
                                )
                            ])
                            for index, _buffered in enumerate(
                                kwargs["buffered_events"]
                            )
                        ]
                        if envelope.handoff == "render"
                        else []
                    ),
                ), entry

        fake = VisualNovelDispatcher()
        fake.queue_route(_router_out(
            event_id="evt_motion",
            event_kind="cascade_exhausted",
            observer_ids=["alice"],
            facts=[ObservableFact.all("The gate starts to rise.")],
        ))
        fake.queue_route(_router_out(
            event_id="evt_arrival",
            event_kind="cascade_exhausted",
            observer_ids=["alice"],
            facts=[ObservableFact.all("The gate locks open.")],
        ))
        fake.queue_narrator(
            handoff="continue",
            reason="The submitted wait condition is still pending.",
            text="REJECTED CANDIDATE",
        )
        fake.queue_narrator(
            handoff="render",
            reason="The gate has finished opening.",
            text="ACCEPTED BATCH",
        )

        with caplog.at_level("INFO", logger="app.engine.turn_loop"):
            result = asyncio.run(run_beat(
                ckpt=ckpt,
                dispatcher=fake,
                actor_id="alice",
                intention="I wait until the gate finishes opening.",
            ))

        assert result.renders == {
            "alice": "The gate starts to rise.\n\nACCEPTED BATCH"
        }
        render = result.visual_novel_renders["alice"]
        assert len(render.segments) == 2
        assert [segment.rendered_event_ids for segment in render.segments] == [
            ["evt_motion"],
            ["evt_arrival"],
        ]
        assert [
            page.text
            for segment in render.segments
            for page in segment.pages
        ] == [
            "The gate starts to rise.",
            "ACCEPTED BATCH",
        ]
        assert [
            len(call["buffered_events"]) for call in fake.narrator_calls
        ] == [1, 2]
        assert fake.narrator_calls[0]["handoff_policy"] == "candidate"
        assert "gate finishes opening" in (
            fake.narrator_calls[0]["handoff_context"]
        )
        assert fake.continuation_calls[0]["original_action"] == (
            "I wait until the gate finishes opening."
        )
        assert "handoff_reason" not in fake.continuation_calls[0]
        assert any(
            "The submitted wait condition is still pending." in message
            for message in caplog.messages
        )
        history = ckpt.narrator_conversations["alice"]
        assert "REJECTED CANDIDATE" not in str(history)
        assert "ACCEPTED BATCH" in str(history)
        assert ckpt.session.render_buffers["alice"] == []
        assert image_batches == [
            ["evt_motion"],
            ["evt_motion", "evt_arrival"],
        ]
        assert cancelled_transactions == ["imgtx_1"]
        assert image_runtime.accepted_image_transaction_ids == {"imgtx_2"}
