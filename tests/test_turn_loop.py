"""v11 turn-loop state-machine tests.

Covers the orchestrator-independent pieces: slot validation, Cat II event
collection, beat cascade ends, and error-message formatting. A fake
`Dispatcher` stands in for the real router/agent/narrator so the state
machine can be exercised without LLM calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.engine import turn_loop
from app.engine.turn_loop import (
    SceneLockManager,
    SlotCheck,
    SlotConflict,
    abort_scene,
    check_act_slot,
    claim_initiator_slot,
    collect_cat_ii_intention,
    format_slot_rejection,
    open_cat_ii,
    pin_cat_ii_responder,
    purge_character_state,
    release_character_slot,
    release_scene_slots,
    run_beat,
    sweep_stale_cat_ii_pins,
)
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    SceneDelta,
    WorldAdjudication,
)
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import SessionState, WorldState


# ---- helpers ---------------------------------------------------------------


def _ckpt(bindings: dict[str, str] | None = None) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id="s",
            character_bindings=bindings or {},
        ),
        world_state=WorldState(),
        characters=[
            CharacterRecord(
                character_id="alice", name="Alice",
                public_sheet=PublicSheet(role="player"),
                location="gatehouse",
                is_playable=True,
            ),
            CharacterRecord(
                character_id="bob", name="Bob",
                public_sheet=PublicSheet(role="player"),
                location="gatehouse",
                is_playable=True,
            ),
            CharacterRecord(
                character_id="pip", name="Pip",
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
    scene: str = "gatehouse",
) -> EventRouterOutput:
    # v11-r5: the schema validator clamps picks to observers. Auto-add
    # every pick to observers here so tests can specify picks without
    # manually curating the observer list — matches what the router
    # prompt's "picks ⊆ observers" invariant requires in production.
    picks = agent_picks or []
    observers = [
        ObserverEntry(character_id="alice", observation_level="d", response_priority=3),
    ]
    existing = {o.character_id for o in observers}
    for p in picks:
        if p not in existing:
            observers.append(
                ObserverEntry(character_id=p, observation_level="d", response_priority=3)
            )
            existing.add(p)
    return EventRouterOutput(
        event_id="",
        decision_rationale="(test fixture)",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                feasible=True,
                resolved_outcome="something happens",
            ),
            scene_delta=SceneDelta(time_advanced_seconds=0),
            observable_facts=[],
        ),
        observers=observers,
        requires_responders=requires_responders,
        required_responders=required_responders or [],
        agent_responder_picks=picks,
        ends_beat=ends_beat,
        ends_beat_reason="directed_at_player" if ends_beat else "",
        spawn=[],
        dormant=[],
        cull=[],
        roster_moves=[],
        scenes_created=[],
    )


class FakeDispatcher:
    """Stubs for Router / Agent / Narrator calls. Each call records its
    inputs and returns a pre-canned response."""

    def __init__(self):
        self.route_calls: list[dict] = []
        self.agent_calls: list[dict] = []
        self.narrator_calls: list[dict] = []
        self.harvest_calls: list[dict] = []
        self._route_responses: list[EventRouterOutput] = []
        self._agent_responses: list[str] = []
        self._narrator_response: str = "RENDER"
        # Observation-harvest fork: queue lists-of-fragments aligned with
        # the order of expected harvest_perceptions calls. If the queue
        # is empty when a call lands, the fake returns one empty
        # fragment per requested character (the same shape the engine
        # would see if every perception failed) — keeps tests that
        # don't exercise harvest from having to seed it.
        self._harvest_responses: list[list[str]] = []

    def queue_route(self, response: EventRouterOutput) -> None:
        self._route_responses.append(response)

    def queue_agent(self, intention: str) -> None:
        self._agent_responses.append(intention)

    def queue_harvest(self, fragments: list[str]) -> None:
        self._harvest_responses.append(fragments)

    async def route_intention(self, **kw) -> EventRouterOutput:
        self.route_calls.append(kw)
        return self._route_responses.pop(0)

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


# ---- r6a partial-mode tests -----------------------------------------------
# Commit 1 deleted `_serialize_agent_intention` — the agent now emits
# prose directly and the dispatcher passes `output.public_text.strip()`
# straight to the router. The serializer's tests went with it. The
# parenthetical-extraction contract is exercised in
# tests/test_character_agent.py instead.


class TestCatIIOpenPartialRender:
    """v11-r6a: Cat II open with pinned humans renders a PARTIAL-mode
    cliffhanger for the pinned humans (and the initiator if human),
    instead of returning renders={} and leaving them blind."""

    def test_cat_ii_open_with_human_responder_renders_partial(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["bob"],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I attack Bob",
            scene_id="gatehouse",
        ))

        assert result.ended_reason == "cat_ii_pending"
        # Bob (pinned) + Alice (initiator, human) both get a render.
        assert "bob" in result.renders
        assert "alice" in result.renders
        # Pin still intact (release_slots=False on the open path).
        slot = ckpt.session.active_act_slots.get("gatehouse", {})
        assert "bob" in slot
        assert slot["bob"].reason == "cat_ii_responder"
        # The router's real Cat II-open event was appended to
        # canonical_events so the narrator's event-id lookup can resolve
        # it without replacing the router's observer/fact payload.
        assert any(
            evt.ends_beat_reason == "cat_ii_open"
            for evt in ckpt.canonical_events
        )
        assert not any(
            "Synthetic open-attempt event" in evt.decision_rationale
            for evt in ckpt.canonical_events
        )
        # Every narrator call used partial_mode_override=True.
        assert fake.narrator_calls, "expected at least one narrator call"
        for call in fake.narrator_calls:
            assert call.get("partial_mode_override") is True

    def test_cat_ii_open_broadcasts_router_facts_to_npc_observers(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        fake = FakeDispatcher()
        open_event = _router_out(
            requires_responders=True,
            required_responders=["bob"],
        )
        open_event.observers = [
            ObserverEntry(
                character_id="alice",
                observation_level="d",
                response_priority=5,
            ),
            ObserverEntry(
                character_id="bob",
                observation_level="d",
                response_priority=5,
            ),
            ObserverEntry(
                character_id="pip",
                observation_level="d",
                response_priority=2,
            ),
        ]
        open_event.canonical_event.observable_facts = [
            ObservableFact.all("Alice extends a hand and says: 'Hold.'")
        ]
        fake.queue_route(open_event)

        asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I grab Bob's sleeve and say hold",
            scene_id="gatehouse",
        ))

        assert ckpt.canonical_events[0] is open_event
        assert ckpt.canonical_events[0].ends_beat_reason == "cat_ii_open"
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert any(
            "Alice extends a hand and says: 'Hold.'" in obs
            for obs in pip.pending_observations
        )

    def test_cat_ii_open_with_only_agent_responder_no_partial_render(self):
        """Cat II that resolves inline with only agent responders should
        NOT take the partial-render path — no pins held. The router's
        actual open-attempt facts are still broadcast before the agent
        responder is asked for an intention."""
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        open_event = _router_out(
            requires_responders=True,
            required_responders=["pip"],
            ends_beat=False,
        )
        open_event.observers = [
            ObserverEntry(
                character_id="alice",
                observation_level="d",
                response_priority=3,
            ),
            ObserverEntry(
                character_id="pip",
                observation_level="d",
                response_priority=5,
            ),
        ]
        open_event.canonical_event.observable_facts = [
            ObservableFact.all("Alice swings at Pip.")
        ]
        fake.queue_route(open_event)
        fake.queue_agent("Pip dodges")
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I attack Pip",
            scene_id="gatehouse",
        ))
        # Resolved inline.
        assert result.ended_reason == "cat_ii_resolution"
        assert len(ckpt.canonical_events) == 2
        assert ckpt.canonical_events[0] is open_event
        assert ckpt.canonical_events[0].ends_beat_reason == "cat_ii_open"
        # No partial_mode_override on any render (should be None / default).
        for call in fake.narrator_calls:
            assert call.get("partial_mode_override") in (None, False)

    def test_end_beat_release_slots_false_keeps_pins_intact(self):
        """v11-r6a: _end_beat(release_slots=False) does NOT clear pins
        — used by the Cat II-open render path to compose renders while
        leaving responder pins alive until they /act."""
        from app.engine.turn_loop import _end_beat
        ckpt = _ckpt(bindings={"alice": "1"})
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", "evt_xyz")
        fake = FakeDispatcher()
        # No renders needed — just verify the pin survives.
        asyncio.run(_end_beat(
            ckpt, fake, "gatehouse",
            ended_reason="cat_ii_pending",
            events_closed=0,
            event_actor_ids=[],
            release_slots=False,
        ))
        slot = ckpt.session.active_act_slots.get("gatehouse", {})
        assert "alice" in slot
        assert slot["alice"].reason == "cat_ii_responder"


# ---- slot-check tests ------------------------------------------------------


class TestCheckActSlot:
    def test_free_when_scene_empty(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        check = check_act_slot(ckpt, "gatehouse", "alice")
        assert check.conflict == SlotConflict.FREE

    def test_initiator_held_by_other(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        claim_initiator_slot(ckpt, "gatehouse", "alice")
        check = check_act_slot(ckpt, "gatehouse", "bob")
        assert check.conflict == SlotConflict.INITIATOR_HELD
        assert check.holder_id == "alice"

    def test_self_busy_when_same_user_double_acts(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        claim_initiator_slot(ckpt, "gatehouse", "alice")
        check = check_act_slot(ckpt, "gatehouse", "alice")
        assert check.conflict == SlotConflict.SELF_BUSY

    def test_cat_ii_self_responder_pins_to_event(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", "evt_xyz")
        check = check_act_slot(ckpt, "gatehouse", "alice")
        assert check.conflict == SlotConflict.CAT_II_SELF_RESPONDER
        assert check.cat_ii_event_id == "evt_xyz"

    def test_cat_ii_other_held_rejects_bystander(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", "evt_xyz")
        check = check_act_slot(ckpt, "gatehouse", "bob")
        assert check.conflict == SlotConflict.CAT_II_OTHER_HELD
        assert check.holder_id == "alice"

    def test_cat_ii_takes_precedence_over_initiator_in_mixed_slot(self):
        # Contrived but could arise — slot has both reasons.
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2", "pip": "3"})
        claim_initiator_slot(ckpt, "gatehouse", "pip")
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", "evt_xyz")
        check = check_act_slot(ckpt, "gatehouse", "bob")
        # Reports Cat II (more specific) rather than the initiator pin.
        assert check.conflict == SlotConflict.CAT_II_OTHER_HELD


class TestRejectionFormatting:
    def test_mentions_holder_name_on_initiator_held(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        claim_initiator_slot(ckpt, "gatehouse", "alice")
        check = check_act_slot(ckpt, "gatehouse", "bob")
        msg = format_slot_rejection(check, ckpt)
        assert "Alice" in msg

    def test_cat_ii_other_message_is_distinct(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", "evt_xyz")
        check = check_act_slot(ckpt, "gatehouse", "bob")
        msg = format_slot_rejection(check, ckpt)
        # v11 phrasing drops jargon; look for the pause framing.
        assert "paused on" in msg.lower() or "waiting on" in msg.lower()


# ---- beat cascade tests ----------------------------------------------------


class TestCatIBeat:
    def test_single_cat_i_ends_beat_and_renders(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="look around",
            scene_id="gatehouse",
        ))

        assert result.events_closed == 1
        assert "alice" in result.renders
        assert result.ended_reason == "directed_at_player"
        assert "gatehouse" not in ckpt.session.active_act_slots

    def test_cat_i_cascades_through_agent_pick(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        # First event: Alice acts, router says don't end, pick Pip.
        fake.queue_route(_router_out(
            agent_picks=["pip"], ends_beat=False,
        ))
        # Agent Pip intends.
        fake.queue_agent("Pip polishes the bell")
        # Router routes Pip's intention, ends beat.
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="wait",
            scene_id="gatehouse",
        ))

        assert result.events_closed == 2
        assert len(fake.agent_calls) == 1
        assert fake.agent_calls[0]["character_id"] == "pip"

    def test_max_events_per_beat_forces_end(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        ckpt.session.config.settings.max_events_per_beat = 2
        fake = FakeDispatcher()
        # Queue an infinite agent cascade that always wants to continue.
        for _ in range(5):
            fake.queue_route(_router_out(
                agent_picks=["pip"], ends_beat=False,
            ))
            fake.queue_agent("Pip does something")

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="wait",
            scene_id="gatehouse",
        ))

        assert result.events_closed == 2  # capped
        assert result.ended_reason == "max_events_cap"


class TestCatIIBeat:
    def test_cat_ii_with_agent_responder_resolves_inline(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        # Step 1: Alice initiates Cat II targeting Pip (agent).
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip"],
            ends_beat=False,
        ))
        # Pip's responder intention.
        fake.queue_agent("Pip dodges")
        # Router composes the resolved Cat II event.
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I attack Pip",
            scene_id="gatehouse",
        ))

        # Cat II now records the visible open attempt and the final
        # resolution as separate canonical events, even when the only
        # responder is an inline NPC.
        assert result.events_closed == 2
        assert result.ended_reason == "cat_ii_resolution"
        assert ckpt.canonical_events[0].ends_beat_reason == "cat_ii_open"
        assert len(ckpt.session.open_cat_ii_events) == 0

    def test_cat_ii_with_human_responder_pauses_beat(self):
        # Alice attacks Bob (human). Beat should pause pending Bob's /act.
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["bob"],
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I attack Bob",
            scene_id="gatehouse",
        ))

        assert result.ended_reason == "cat_ii_pending"
        # v11-r6a: pinned humans + initiator (if human) now get PARTIAL-
        # mode cliffhanger renders on Cat II open. Previously returned
        # an empty dict and left them waiting blind.
        assert "bob" in result.renders
        assert "alice" in result.renders
        # Bob should now be pinned.
        slot = ckpt.session.active_act_slots.get("gatehouse", {})
        assert "bob" in slot
        assert slot["bob"].reason == "cat_ii_responder"
        # The open event is tracked.
        assert len(ckpt.session.open_cat_ii_events) == 1

    def test_cat_ii_responder_intention_closes_event_and_renders(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        # Pre-state: the open Cat II event exists, Bob is pinned.
        evt = open_cat_ii(
            ckpt,
            scene_id="gatehouse",
            initiator_id="alice",
            initiator_intention="I attack Bob",
            required_responders=["bob"],
        )
        pin_cat_ii_responder(ckpt, "gatehouse", "bob", evt.event_id)

        fake = FakeDispatcher()
        # Resolution routes the Cat II event (single responder filled).
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="bob", intention="I block",
            scene_id="gatehouse",
            cat_ii_event_id=evt.event_id,
        ))

        assert result.events_closed == 1
        assert result.ended_reason == "cat_ii_resolution"
        assert len(ckpt.session.open_cat_ii_events) == 0
        # Slot released.
        assert "gatehouse" not in ckpt.session.active_act_slots

    def test_cat_ii_resolution_reaches_npc_initiator_after_human_response(self):
        ckpt = _ckpt(bindings={"bob": "2"})
        evt = open_cat_ii(
            ckpt,
            scene_id="gatehouse",
            initiator_id="pip",
            initiator_intention="Pip throws Bob",
            required_responders=["bob"],
        )
        pin_cat_ii_responder(ckpt, "gatehouse", "bob", evt.event_id)

        resolution = _router_out(ends_beat=True)
        resolution.observers = [
            ObserverEntry(
                character_id="bob",
                observation_level="d",
                response_priority=5,
            ),
            ObserverEntry(
                character_id="pip",
                observation_level="d",
                response_priority=5,
            ),
        ]
        resolution.canonical_event.observable_facts = [
            ObservableFact.all("Bob pivots through the throw."),
            ObservableFact.all("Pip ends the exchange with his grip broken."),
        ]

        fake = FakeDispatcher()
        fake.queue_route(resolution)

        result = asyncio.run(run_beat(
            ckpt=ckpt,
            dispatcher=fake,
            actor_id="bob",
            intention="I pivot out",
            scene_id="gatehouse",
            cat_ii_event_id=evt.event_id,
        ))

        assert result.ended_reason == "cat_ii_resolution"
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert len(pip.pending_observations) == 1
        assert "Bob pivots through the throw." in pip.pending_observations[0]
        assert "Pip ends the exchange" in pip.pending_observations[0]

    def test_cat_ii_multi_responder_pauses_until_all_intentions_in(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        evt = open_cat_ii(
            ckpt,
            scene_id="gatehouse",
            initiator_id="pip",
            initiator_intention="Pip throws a punch at Alice",
            required_responders=["alice", "bob"],
        )
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", evt.event_id)
        pin_cat_ii_responder(ckpt, "gatehouse", "bob", evt.event_id)

        fake = FakeDispatcher()
        # First Bob /acts as intercept. This should NOT resolve yet —
        # Alice's intention is still missing.
        result1 = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="bob", intention="I lunge to block",
            scene_id="gatehouse",
            cat_ii_event_id=evt.event_id,
        ))
        assert result1.ended_reason == "cat_ii_pending"
        assert result1.renders == {}
        # Bob's pin released, Alice's still held.
        slot = ckpt.session.active_act_slots.get("gatehouse", {})
        assert "bob" not in slot
        assert "alice" in slot

        # Now Alice /acts. This closes the event.
        fake.queue_route(_router_out(ends_beat=True))
        result2 = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I duck",
            scene_id="gatehouse",
            cat_ii_event_id=evt.event_id,
        ))
        assert result2.ended_reason == "cat_ii_resolution"
        assert len(ckpt.session.open_cat_ii_events) == 0


class TestSchemaValidators:
    def test_requires_responders_without_list_rejected(self):
        """Schema enforces Cat II cannot have empty required_responders."""
        with pytest.raises(ValueError, match="empty"):
            _router_out(
                requires_responders=True,
                required_responders=[],
            )

    def test_required_responders_duplicates_rejected(self):
        with pytest.raises(ValueError, match="duplicates"):
            _router_out(
                requires_responders=True,
                required_responders=["bob", "bob"],
            )

    def test_event_id_auto_populated(self):
        out = _router_out()
        assert out.event_id.startswith("evt_")
        assert len(out.event_id) > 4


class TestPurgeCharacterState:
    def test_purge_drops_pin_and_render_buffer(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        evt = open_cat_ii(
            ckpt, scene_id="gatehouse",
            initiator_id="pip", initiator_intention="punch",
            required_responders=["alice", "bob"],
        )
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", evt.event_id)
        pin_cat_ii_responder(ckpt, "gatehouse", "bob", evt.event_id)
        # Fake a render buffer entry for bob.
        from app.schemas.state import RenderBufferEntry
        ckpt.session.render_buffers["bob"] = [
            RenderBufferEntry(event_id="evt_x", observation_level="direct"),
        ]

        purge_character_state(ckpt, "bob")

        # Bob's pin gone, Alice's intact.
        slot = ckpt.session.active_act_slots.get("gatehouse", {})
        assert "bob" not in slot
        assert "alice" in slot
        # Bob's render buffer cleared.
        assert "bob" not in ckpt.session.render_buffers
        # Bob removed from required responders.
        live_evts = ckpt.session.open_cat_ii_events
        assert len(live_evts) == 1
        assert "bob" not in live_evts[0].required_responders
        assert "alice" in live_evts[0].required_responders

    def test_purge_initiator_abandons_event(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        open_cat_ii(
            ckpt, scene_id="gatehouse",
            initiator_id="pip", initiator_intention="punch",
            required_responders=["alice"],
        )
        purge_character_state(ckpt, "pip")
        # Event abandoned entirely because initiator is gone.
        assert ckpt.session.open_cat_ii_events == []


class TestReleaseSceneSlotsPreservesOpenCatII:
    def test_release_leaves_cat_ii_pins_intact(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        # A Cat II is open for alice.
        evt = open_cat_ii(
            ckpt, scene_id="gatehouse",
            initiator_id="pip", initiator_intention="punch",
            required_responders=["alice"],
        )
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", evt.event_id)
        # AND bob has somehow initiated a separate beat (would only
        # happen through a bug, but we're testing the defensive path).
        claim_initiator_slot(ckpt, "gatehouse", "bob")

        # Beat ending for bob shouldn't blow away alice's pin.
        release_scene_slots(ckpt, "gatehouse")
        slot = ckpt.session.active_act_slots.get("gatehouse", {})
        assert "bob" not in slot
        assert "alice" in slot
        assert slot["alice"].reason == "cat_ii_responder"


class TestAbortScene:
    def test_abort_clears_pins_and_events(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        evt = open_cat_ii(
            ckpt, scene_id="gatehouse",
            initiator_id="pip", initiator_intention="punch",
            required_responders=["alice"],
        )
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", evt.event_id)

        dropped = abort_scene(ckpt, "gatehouse")
        assert dropped == 1
        assert "gatehouse" not in ckpt.session.active_act_slots
        assert ckpt.session.open_cat_ii_events == []


class TestSweepStaleCatIIPins:
    def test_sweep_disabled_when_timeout_zero(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        ckpt.session.config.settings.cat_ii_human_timeout_seconds = 0
        open_cat_ii(
            ckpt, scene_id="gatehouse",
            initiator_id="pip", initiator_intention="punch",
            required_responders=["alice"],
        )
        assert sweep_stale_cat_ii_pins(ckpt) == []

    def test_sweep_fills_stale_pin_as_does_not_act(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        ckpt.session.config.settings.cat_ii_human_timeout_seconds = 1
        evt = open_cat_ii(
            ckpt, scene_id="gatehouse",
            initiator_id="pip", initiator_intention="punch",
            required_responders=["alice"],
        )
        # Backdate the open time.
        from datetime import datetime, timedelta, timezone
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        evt.opened_at = past.isoformat()

        ids = sweep_stale_cat_ii_pins(ckpt)
        assert ids == [evt.event_id]
        # Structured marker — router prompt skips these rather than
        # parsing the intention text.
        assert "alice" in evt.swept_responders
        # Intention sentinel is still stored (for debug) but no longer
        # the human-leaking "does not act — away from the scene" text.
        assert "AFK-swept" in evt.collected_intentions["alice"]


class TestSceneLockManager:
    def test_same_scene_returns_same_lock(self):
        mgr = SceneLockManager()
        async def run():
            a = await mgr.get("s", "gatehouse")
            b = await mgr.get("s", "gatehouse")
            return a is b
        assert asyncio.run(run())

    def test_different_scenes_distinct_locks(self):
        mgr = SceneLockManager()
        async def run():
            a = await mgr.get("s", "gatehouse")
            b = await mgr.get("s", "threshold")
            return a is not b
        assert asyncio.run(run())


class TestRejectionEchoesText:
    def test_echo_present_on_initiator_held(self):
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        claim_initiator_slot(ckpt, "gatehouse", "alice")
        check = check_act_slot(ckpt, "gatehouse", "bob")
        msg = format_slot_rejection(
            check, ckpt, attempted_text="I walk outside and look at the stars",
        )
        assert "I walk outside" in msg


class TestValidationHardening:
    def test_unknown_ends_beat_reason_coerced_to_empty(self):
        # ends_beat_reason typo: should be clamped to "" with warn-log,
        # not raise ValidationError that would crash a beat.
        out = _router_out(ends_beat=True)
        out_dict = out.model_dump()
        out_dict["ends_beat_reason"] = "scene-transition"  # typo
        from app.schemas.event_router import EventRouterOutput
        rebuilt = EventRouterOutput.model_validate(out_dict)
        assert rebuilt.ends_beat_reason == ""

    def test_empty_agent_intention_drops_pick_and_ends_beat(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            agent_picks=["pip"], ends_beat=False,
        ))
        # Agent returns an empty string — should be treated as failure,
        # beat ends cascade_exhausted rather than routing the empty
        # intention through adjudication.
        fake.queue_agent("")

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="wait",
            scene_id="gatehouse",
        ))
        assert result.ended_reason == "cascade_exhausted"

    def test_refusal_text_no_longer_drops_pick(self):
        """v11-r5: refusal detection moved to the prompt (agent
        prompt rule against refusals / frame-breaks). If a misbehaving
        model returns refusal text anyway, the
        engine does NOT quietly swallow it — the text is routed through
        the adjudicator like any other intention, and the bug surfaces
        visibly in the rendered scene rather than as a mystery
        cascade_exhausted.
        """
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            agent_picks=["pip"], ends_beat=False,
        ))
        fake.queue_agent("I cannot comply with that request.")
        # Router will see the refusal as Pip's "intention" and classify
        # it; for the test we just need to confirm the pick is NOT
        # dropped by the engine itself.
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="wait",
            scene_id="gatehouse",
        ))
        # Two events closed: Alice's Cat I + Pip's (refusal-shaped) Cat I.
        # Not cascade_exhausted — the engine routed the intention.
        assert result.events_closed == 2


class TestObservationHarvestSchema:
    """v11-r8a: schema invariants on the observation_harvest reason.

    These pin the contract that lets the engine fork branchlessly:
    if the router emits `ends_beat_reason="observation_harvest"`, the
    schema must clamp `ends_beat=true` so the harvest fork in
    run_beat doesn't have to second-guess it; the picks list may be
    empty (will fall through as a sparse Cat I close — engine-side
    warning); and the value must round-trip through model_validate
    without crashing legacy callers that haven't seen the new
    literal yet.
    """

    def test_observation_harvest_is_valid_ends_beat_reason(self):
        # The literal must round-trip cleanly. Pre-r8a this raised
        # ValidationError because the enum didn't include it.
        out = _router_out(ends_beat=True, agent_picks=["pip"])
        out_dict = out.model_dump()
        out_dict["ends_beat_reason"] = "observation_harvest"
        from app.schemas.event_router import EventRouterOutput
        rebuilt = EventRouterOutput.model_validate(out_dict)
        assert rebuilt.ends_beat_reason == "observation_harvest"

    def test_observation_harvest_coerces_ends_beat_true_when_false(self):
        # Router prompt says ends_beat MUST be true on harvest; if
        # the model emits false anyway, the validator clamps to true
        # so the engine doesn't loop.
        out = _router_out(ends_beat=True, agent_picks=["pip"])
        out_dict = out.model_dump()
        out_dict["ends_beat_reason"] = "observation_harvest"
        out_dict["ends_beat"] = False
        from app.schemas.event_router import EventRouterOutput
        rebuilt = EventRouterOutput.model_validate(out_dict)
        assert rebuilt.ends_beat is True

    def test_observation_harvest_with_empty_picks_warns_but_validates(
        self, caplog
    ):
        # Empty picks is malformed (nothing to harvest) but the
        # validator clamps-not-raises so a one-off prompt drift
        # doesn't crash a session. Engine side falls through as a
        # sparse Cat I close — see test_harvest_skips_when_no_picks.
        import logging
        out = _router_out(ends_beat=True)  # picks default to []
        out_dict = out.model_dump()
        out_dict["ends_beat_reason"] = "observation_harvest"
        out_dict["agent_responder_picks"] = []
        from app.schemas.event_router import EventRouterOutput
        with caplog.at_level(logging.WARNING):
            rebuilt = EventRouterOutput.model_validate(out_dict)
        assert rebuilt.ends_beat_reason == "observation_harvest"
        assert any(
            "observation_harvest" in r.message and "empty" in r.message
            for r in caplog.records
        )


class TestObservationHarvestFork:
    """v11-r8a: the run_beat fork that fires Dispatcher.harvest_perceptions
    in parallel and folds the fragments into the canonical event's
    observable_facts BEFORE the narrator composes the render.
    """

    def test_harvest_fires_perceive_and_appends_fragments(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        # Add a second NPC so the picks list has multiple targets.
        ckpt.characters.append(
            CharacterRecord(
                character_id="vex", name="Vex",
                public_sheet=PublicSheet(role="npc"),
                location="gatehouse",
                is_playable=False,
            ),
        )
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            ends_beat=True,
            agent_picks=["pip", "vex"],
        ))
        # Override ends_beat_reason to harvest (the helper defaults to
        # directed_at_player).
        fake._route_responses[0].ends_beat_reason = "observation_harvest"
        fake.queue_harvest([
            "Pip in his usual leathers, hands quiet at his sides.",
            "Vex in midnight silk, eyes tracking every doorway.",
        ])

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I look at each of them",
            scene_id="gatehouse",
        ))

        # Exactly one harvest call, with both targets in order.
        assert len(fake.harvest_calls) == 1
        assert fake.harvest_calls[0]["character_ids"] == ["pip", "vex"]
        # Fragments landed on the canonical event's observable_facts
        # with name-tagged loadout markers the narrator prompt knows
        # how to render.
        appended = ckpt.canonical_events[-1].canonical_event.observable_facts
        assert any("[loadout — Pip]" in f for f in appended)
        assert any("[loadout — Vex]" in f for f in appended)
        # Beat ended on the harvest reason, not directed_at_player.
        assert result.ended_reason == "observation_harvest"
        # No cascade fired (no agent_intend calls — this is the whole
        # point of the harvest fork).
        assert fake.agent_calls == []

    def test_harvest_drops_empty_fragments_silently(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            ends_beat=True, agent_picks=["pip"],
        ))
        fake._route_responses[0].ends_beat_reason = "observation_harvest"
        # Perception failed for the only pick — empty fragment.
        fake.queue_harvest([""])

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I study Pip",
            scene_id="gatehouse",
        ))
        # No loadout markers in observable_facts — empty fragment
        # was dropped, not rendered as "[loadout — Pip] ".
        appended = ckpt.canonical_events[-1].canonical_event.observable_facts
        assert not any("loadout" in f for f in appended)
        # But the beat still closes cleanly on the harvest reason.
        assert result.ended_reason == "observation_harvest"

    def test_harvest_drops_reflective_simile_sentences_before_append(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            ends_beat=True, agent_picks=["pip"],
        ))
        fake._route_responses[0].ends_beat_reason = "observation_harvest"
        fake.queue_harvest([
            (
                "Pip wears patched leather, hands quiet at her sides. "
                "Her eyes track the room with the precision of someone "
                "cataloging which conversations matter. "
                "A brass key hangs at her throat."
            ),
        ])

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I study Pip",
            scene_id="gatehouse",
        ))

        appended = ckpt.canonical_events[-1].canonical_event.observable_facts
        loadouts = [f for f in appended if "[loadout — Pip]" in f]
        assert len(loadouts) == 1
        assert "patched leather" in loadouts[0]
        assert "brass key" in loadouts[0]
        assert "precision of someone" not in loadouts[0]
        assert result.ended_reason == "observation_harvest"

    def test_harvest_skips_when_picks_filter_to_empty(self):
        # Router picks a HUMAN (drift / bug). The engine's
        # _filter_picks_for_dispatch strips humans before harvest
        # fires; with no picks left, no harvest call is made, and
        # the beat closes as a sparse Cat I.
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            ends_beat=True, agent_picks=["alice"],  # alice is human
        ))
        fake._route_responses[0].ends_beat_reason = "observation_harvest"

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I look at myself",
            scene_id="gatehouse",
        ))
        # No harvest call fired (no valid picks after filter).
        assert fake.harvest_calls == []
        # No loadout fragments on the event.
        appended = ckpt.canonical_events[-1].canonical_event.observable_facts
        assert not any("loadout" in f for f in appended)
        # Beat still closes on the harvest reason — fall-through
        # is graceful, not a crash.
        assert result.ended_reason == "observation_harvest"

    def test_harvest_filters_off_scene_picks(self):
        # Router picks an off-scene NPC; the same engine filter
        # strips them. Only the in-scene pick gets a perception.
        ckpt = _ckpt(bindings={"alice": "1"})
        ckpt.characters.append(
            CharacterRecord(
                character_id="nyx", name="Nyx",
                public_sheet=PublicSheet(role="npc"),
                location="library",  # not the gatehouse
                is_playable=False,
            ),
        )
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            ends_beat=True, agent_picks=["pip", "nyx"],
        ))
        fake._route_responses[0].ends_beat_reason = "observation_harvest"
        fake.queue_harvest(["Pip silent."])  # one entry, for pip only

        asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="look",
            scene_id="gatehouse",
        ))
        # Harvest fired for in-scene pick only.
        assert len(fake.harvest_calls) == 1
        assert fake.harvest_calls[0]["character_ids"] == ["pip"]

    def test_harvest_passes_acting_character_id_to_dispatcher(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            ends_beat=True, agent_picks=["pip"],
        ))
        fake._route_responses[0].ends_beat_reason = "observation_harvest"
        fake.queue_harvest(["pip's loadout text"])

        asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="size them up",
            scene_id="gatehouse",
        ))
        # The looker (alice) is plumbed for the dispatcher's
        # context-builder helpers; the perception itself is
        # observer-agnostic.
        assert fake.harvest_calls[0]["acting_character_id"] == "alice"


class TestSweepStructuredMarker:
    def test_sweep_records_swept_responders_list(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        ckpt.session.config.settings.cat_ii_human_timeout_seconds = 1
        evt = open_cat_ii(
            ckpt, scene_id="gatehouse",
            initiator_id="pip", initiator_intention="punch",
            required_responders=["alice"],
        )
        from datetime import datetime, timedelta, timezone
        evt.opened_at = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).isoformat()

        sweep_stale_cat_ii_pins(ckpt)
        assert "alice" in evt.swept_responders
        # Sentinel kept in collected_intentions for debug, but the
        # structured marker is what the router prompt reads.
        assert "AFK-swept" in evt.collected_intentions["alice"]


class TestTimezoneAwareTimestamps:
    def test_claimed_at_is_tz_aware(self):
        ckpt = _ckpt(bindings={"alice": "1"})
        claim_initiator_slot(ckpt, "gatehouse", "alice")
        entry = ckpt.session.active_act_slots["gatehouse"]["alice"]
        # ISO string contains TZ info ("+00:00" or "Z").
        assert "+" in entry.claimed_at or entry.claimed_at.endswith("Z")


class TestInitiatorExcludedFromCatIIResponders:
    def test_self_responder_is_filtered_out(self):
        """If router lists the initiator in required_responders, the loop
        drops them and proceeds as Cat I (or cascades with remaining)."""
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["alice"],  # self-pin attempt
            ends_beat=True,
        ))

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I punch myself",
            scene_id="gatehouse",
        ))

        # No open Cat II — collapsed to Cat I.
        assert len(ckpt.session.open_cat_ii_events) == 0
        assert result.events_closed == 1
        # Slot released at beat end.
        assert "gatehouse" not in ckpt.session.active_act_slots


class TestPicksSubsetOfObservers:
    """v11-r5: the `picks ⊆ observers` invariant is enforced at schema
    layer by clamping (dropping picks not in observers with a warn)."""

    def test_picks_in_observers_preserved(self):
        out = _router_out(agent_picks=["alice"])  # alice is default observer
        assert out.agent_responder_picks == ["alice"]

    def test_picks_not_in_observers_dropped(self):
        from app.schemas.event_router import EventRouterOutput, ObserverEntry
        from app.schemas.events import CanonicalEvent, SceneDelta, WorldAdjudication
        # observers = [alice]; picks = [alice, pip]; pip should be dropped.
        out = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    feasible=True, resolved_outcome="x",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[],
            ),
            observers=[ObserverEntry(character_id="alice", observation_level="d", response_priority=3)],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=["alice", "pip"],
            ends_beat=True,
            ends_beat_reason="directed_at_player",
            spawn=[],
            dormant=[],
            cull=[],
            roster_moves=[],
            scenes_created=[],
        )
        assert out.agent_responder_picks == ["alice"]  # pip clamped


class TestPinnedGuards:
    """v11-r5: the router cannot dormant/cull/roster_move a character
    who is currently pinned (initiator or Cat II responder)."""

    def test_dormant_on_pinned_is_skipped(self):
        from app.engine.character_manager import CharacterManager
        ckpt = _ckpt(bindings={"alice": "1"})
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", "evt_x")
        # Build a router output saying alice goes dormant.
        routed = _router_out()
        routed.dormant = ["alice"]
        CharacterManager().apply_roster_updates(ckpt, routed)
        # Alice's status unchanged; pin still intact.
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        assert alice.status.value == "active"
        assert "alice" in ckpt.session.active_act_slots.get("gatehouse", {})

    def test_cull_on_pinned_proceeds_and_purges(self):
        """Cull is terminal — unlike dormant, it proceeds even on a
        pinned character, with purge_character_state cleaning up the
        pin. The alternative (skipping) would leave the character
        dead-in-fiction but perpetually pinned."""
        from app.engine.character_manager import CharacterManager
        ckpt = _ckpt(bindings={"alice": "1"})
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", "evt_x")
        routed = _router_out()
        routed.cull = ["alice"]
        CharacterManager().apply_roster_updates(ckpt, routed)
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        assert alice.status.value == "culled"
        # Pin released by purge.
        assert "alice" not in ckpt.session.active_act_slots.get("gatehouse", {})


class TestFilterPicksForDispatch:
    """v11-r7g: engine-level filter applied to agent_responder_picks
    before dispatching agent_intend.

    Two filters: (1) drop human-bound characters (humans only enter via
    /act, never via cascade), (2) drop characters not currently in the
    beat's scene (pre-r7g picks at other locations were dispatched and
    routinely returned empty/refusal intentions, producing WARNING
    noise; the playtest log showed `pip` and `nyx` at bell_of_arrivals
    being picked while the player /act'd at archive_main_hall).
    """

    def _ckpt_with_chars(
        self, chars: list[tuple[str, str]],
        bindings: dict[str, str] | None = None,
    ):
        ckpt = _ckpt(bindings=bindings or {})
        # Replace default character roster with our test set; relies on
        # CharacterRecord defaults for everything but id/name/location.
        ckpt.characters = [
            CharacterRecord(
                character_id=cid,
                name=cid.title(),
                location=loc,
                is_playable=cid in (bindings or {}),
            )
            for cid, loc in chars
        ]
        return ckpt

    def test_in_scene_npc_passes_through(self):
        from app.engine.turn_loop import _filter_picks_for_dispatch
        ckpt = self._ckpt_with_chars(
            [("alice", "gatehouse"), ("npc_a", "gatehouse")],
            bindings={"alice": "1"},
        )
        assert _filter_picks_for_dispatch(
            ckpt, "gatehouse", ["npc_a"],
        ) == ["npc_a"]

    def test_out_of_scene_npc_filtered(self):
        from app.engine.turn_loop import _filter_picks_for_dispatch
        ckpt = self._ckpt_with_chars(
            [
                ("alice", "gatehouse"),
                ("npc_in", "gatehouse"),
                ("npc_far", "bell_tower"),
            ],
            bindings={"alice": "1"},
        )
        # npc_far gets dropped; npc_in survives.
        assert _filter_picks_for_dispatch(
            ckpt, "gatehouse", ["npc_in", "npc_far"],
        ) == ["npc_in"]

    def test_human_pick_filtered_even_in_scene(self):
        from app.engine.turn_loop import _filter_picks_for_dispatch
        ckpt = self._ckpt_with_chars(
            [("alice", "gatehouse"), ("bob", "gatehouse")],
            bindings={"alice": "1", "bob": "2"},
        )
        # Both humans, both in scene — both filtered (cascade is NPC-only).
        assert _filter_picks_for_dispatch(
            ckpt, "gatehouse", ["alice", "bob"],
        ) == []

    def test_preserves_router_order(self):
        from app.engine.turn_loop import _filter_picks_for_dispatch
        ckpt = self._ckpt_with_chars(
            [
                ("alice", "gatehouse"),
                ("npc_a", "gatehouse"),
                ("npc_b", "gatehouse"),
                ("npc_c", "gatehouse"),
            ],
            bindings={"alice": "1"},
        )
        assert _filter_picks_for_dispatch(
            ckpt, "gatehouse", ["npc_c", "npc_a", "npc_b"],
        ) == ["npc_c", "npc_a", "npc_b"]

    def test_creator_player_character_filtered_without_binding(self):
        """fix-9 regression: `session.player_character_id` may name the
        creator's character without a corresponding row in
        `character_bindings` (older saves; CLI single-player flows).
        Pre-fix the dispatch filter only consulted `bindings`, so the
        creator's character could slip into NPC dispatch and produce
        the "router tried to make my own character speak" symptom.
        Now the filter routes through `collect_player_ids`, which
        unions bindings AND `player_character_id`."""
        from app.engine.turn_loop import _filter_picks_for_dispatch
        ckpt = self._ckpt_with_chars(
            [("hero", "gatehouse"), ("npc_a", "gatehouse")],
            bindings={},  # explicitly empty
        )
        # Creator binding lives only on the session field.
        ckpt.session.player_character_id = "hero"
        # Hero must be filtered even though they're not in `bindings`.
        assert _filter_picks_for_dispatch(
            ckpt, "gatehouse", ["hero", "npc_a"],
        ) == ["npc_a"]


class TestAgentEmptyGuard:
    """v11-r5: refusal detection moved from pattern matching to
    prompt-level (rule in the agent prompt). The engine only guards literal
    empty/whitespace output now. Legitimate in-character lines
    containing "I can't" / "I cannot" / "As an AI" (rare but possible
    when a character IS an AI in-fiction) all pass through.
    """

    def test_empty_is_treated_as_refusal(self):
        from app.engine.turn_loop import _is_agent_refusal
        assert _is_agent_refusal("")
        assert _is_agent_refusal("   ")
        assert _is_agent_refusal("\n\n")

    def test_in_character_lines_pass_through(self):
        from app.engine.turn_loop import _is_agent_refusal
        # Legitimate terse in-character dialogue that old regex wrongly dropped.
        assert not _is_agent_refusal("I can't see them from here.")
        assert not _is_agent_refusal("I cannot allow that, my lord.")
        # Real refusal content — the PROMPT is supposed to prevent this,
        # but if the model emits it anyway, the engine does NOT quietly
        # drop it; the beat will run and the bug will surface visibly.
        assert not _is_agent_refusal("I can't help with that request.")
        assert not _is_agent_refusal("As an AI, I cannot generate this.")


class TestAbortSceneFlushesBuffers:
    def test_abort_clears_in_scene_humans_buffers(self):
        from app.engine.turn_loop import abort_scene, open_cat_ii, pin_cat_ii_responder
        from app.schemas.state import RenderBufferEntry
        # Use the same _ckpt helper pattern the rest of the file uses.
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        evt = open_cat_ii(
            ckpt, scene_id="gatehouse",
            initiator_id="pip", initiator_intention="punch",
            required_responders=["alice"],
        )
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", evt.event_id)
        # Queue buffered events for both in-scene humans.
        ckpt.session.render_buffers["alice"] = [
            RenderBufferEntry(event_id="evt_x", observation_level="direct"),
        ]
        ckpt.session.render_buffers["bob"] = [
            RenderBufferEntry(event_id="evt_y", observation_level="direct"),
        ]

        abort_scene(ckpt, "gatehouse")

        # Both in-scene humans' buffers cleared.
        assert ckpt.session.render_buffers.get("alice", []) == []
        assert ckpt.session.render_buffers.get("bob", []) == []


class TestRejectionTruncationBumpAndPhrasing:
    def test_rejection_truncates_at_1500_not_500(self):
        from app.engine.turn_loop import check_act_slot, claim_initiator_slot, format_slot_rejection
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        claim_initiator_slot(ckpt, "gatehouse", "alice")
        check = check_act_slot(ckpt, "gatehouse", "bob")
        long_text = "I walk " * 300  # ~2100 chars
        msg = format_slot_rejection(check, ckpt, attempted_text=long_text)
        # The echoed text should preserve more than 500 chars now.
        assert "I walk " * 100 in msg  # ~700 chars worth
        # Still truncates somewhere below the full length.
        assert len(msg) < 2500

    def test_rejection_phrasing_not_gaslighty(self):
        from app.engine.turn_loop import check_act_slot, claim_initiator_slot, format_slot_rejection
        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        claim_initiator_slot(ckpt, "gatehouse", "alice")
        check = check_act_slot(ckpt, "gatehouse", "bob")
        msg = format_slot_rejection(check, ckpt)
        assert "didn't go through" in msg
        assert "wasn't submitted" not in msg  # Old phrasing gone.


class TestParallelNarratorFanOut:
    """v11-r6c: `_end_beat` fans out narrator_compose via asyncio.gather
    so multi-human scenes don't pay serial latency. With two buffered
    humans each waiting 0.1s, the total elapsed time should be ~0.1s
    (parallel) not ~0.2s (serial)."""

    def test_two_humans_renders_in_parallel_not_serial(self):
        import time
        from app.engine.turn_loop import _end_beat, append_to_render_buffer
        from app.schemas.event_router import EventRouterOutput
        from app.schemas.events import (
            CanonicalEvent, SceneDelta, WorldAdjudication,
        )

        ckpt = _ckpt(bindings={"alice": "1", "bob": "2"})
        # Seed a canonical event and buffer it for both humans so each has
        # a non-empty render buffer.
        event = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    feasible=True, resolved_outcome="y",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[],
            ),
            observers=[],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="directed_at_player",
            spawn=[],
            dormant=[],
            cull=[],
            roster_moves=[],
            scenes_created=[],
        )
        ckpt.canonical_events.append(event)
        append_to_render_buffer(ckpt, "alice", event.event_id, "direct")
        append_to_render_buffer(ckpt, "bob", event.event_id, "direct")

        class SlowDispatcher(FakeDispatcher):
            async def narrator_compose(self, **kw):
                # Each render waits 0.1s before returning. Two serial
                # calls would take ~0.2s; parallel gather should finish
                # in ~0.1s.
                await asyncio.sleep(0.1)
                self.narrator_calls.append(kw)
                envelope = NarratorFinalOutput(final_text="RENDER")
                entry = TranscriptEntry(
                    user=kw.get("user_input", ""),
                    assistant="RENDER",
                )
                return envelope, entry

        fake = SlowDispatcher()

        t0 = time.monotonic()
        result = asyncio.run(_end_beat(
            ckpt, fake, "gatehouse",
            ended_reason="directed_at_player",
            events_closed=1,
            event_actor_ids=["alice"],
        ))
        elapsed = time.monotonic() - t0

        # Both POVs rendered.
        assert "alice" in result.renders
        assert "bob" in result.renders
        # Two calls were made.
        assert len(fake.narrator_calls) == 2
        # Parallel: under 0.18s (well below 0.2s serial lower bound).
        # Some headroom for CI jitter; the serial path is 0.2s+.
        assert elapsed < 0.18, (
            f"expected parallel fan-out (<0.18s), got {elapsed:.3f}s — "
            f"likely reverted to serial loop"
        )

    def test_single_human_still_renders(self):
        """Sanity check: the asyncio.gather path handles the 1-human case
        without degrading. Previously the loop was `for h in humans`; now
        `asyncio.gather(*())` on an empty tuple with a single-element
        tuple must still produce a render."""
        from app.engine.turn_loop import _end_beat, append_to_render_buffer
        from app.schemas.event_router import EventRouterOutput
        from app.schemas.events import (
            CanonicalEvent, SceneDelta, WorldAdjudication,
        )

        ckpt = _ckpt(bindings={"alice": "1"})
        event = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    feasible=True, resolved_outcome="y",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[],
            ),
            observers=[],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="directed_at_player",
            spawn=[],
            dormant=[],
            cull=[],
            roster_moves=[],
            scenes_created=[],
        )
        ckpt.canonical_events.append(event)
        append_to_render_buffer(ckpt, "alice", event.event_id, "direct")

        fake = FakeDispatcher()
        result = asyncio.run(_end_beat(
            ckpt, fake, "gatehouse",
            ended_reason="directed_at_player",
            events_closed=1,
            event_actor_ids=["alice"],
        ))
        assert "alice" in result.renders
        assert len(fake.narrator_calls) == 1


class TestBroadcastEventNpcObservers:
    """`broadcast_event` pushes the canonical event's observable_facts
    onto every NPC observer who is co-located with the broadcast
    `scene_id`, and only those. The actor of the event is excluded;
    their own action lives in their rolling history.

    History note:
      - Pre-r8b only off-scene observers were pushed; in-scene NPCs
        were silently skipped on the assumption they'd read the
        event "live" via the cascade dispatcher's user-message
        block. That assumption was wrong (the live channel carried
        `observed_facts=[]`) — cascade NPCs reacted to stale queue
        entries and the router fabricated dialogue to fit.
      - r8b restored symmetric push (in-scene + off-scene).
      - r9b deletes the off-scene channel entirely. The router's
        per-event `resolved_outcome` regularly fused private and
        public sub-beats into one omniscient summary; piping that
        to off-scene observers (e.g. Ashara at the dining hall
        receiving Dan Garvey's bedroom-wardrobe choice as an
        `[off-scene perception]`) was a privacy leak and an
        attribution-confusion seed. Cross-scene awareness now
        requires a separate event whose `scene_id` is wherever the
        news lands — not piggy-backed on an in-scene event's summary.
      - r10 (Option B) replaces the `resolved_outcome` payload with
        the full `observable_facts` list. The outcome string regularly
        wove interpretive interior into surface beats ("the strain of
        speaking close to the edge of what she is permitted"), and
        broadcasting that to in-scene NPCs as their own perception
        leaked author-voice interior into agent-facing context. The
        observable_facts list is the router's surface-grade enumeration
        — verbatim dialogue, visible gestures, ambient shifts — and
        is the only thing observers receive now.
      - r10 also drops the `[in-scene perception]` tag from the
        push. The on-stage agent body's `## Scene` / `## What You
        Observe This Turn` blocks are gone, so the inbox IS the
        live sensorium for the scene; tagging entries adds noise
        without routing value."""

    def _event(
        self,
        *,
        observer_ids: list[str],
        outcome: str = "Jordan hands a note to a runner.",
    ) -> EventRouterOutput:
        observers = [
            ObserverEntry(
                character_id=cid, observation_level="d", response_priority=3,
            )
            for cid in observer_ids
        ]
        return EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    feasible=True,
                    resolved_outcome=outcome,
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[outcome],
            ),
            observers=observers,
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True, ends_beat_reason="directed_at_player",
            spawn=[], dormant=[], cull=[], roster_moves=[], scenes_created=[],
        )

    def test_off_scene_npc_observer_NOT_pushed(self):
        """v11-r9b: off-scene observers are no longer pushed. Pre-r9b
        marcus (at citrus_garden, not gatehouse) would have received
        the event summary tagged `[off-scene perception]` because
        the router listed him as an observer; r9b drops the off-scene
        channel entirely so off-scene NPCs only learn about events
        whose own `scene_id` they actually are in."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()
        marcus = CharacterRecord(
            character_id="marcus", name="Marcus",
            public_sheet=PublicSheet(role="contestant"),
            location="citrus_garden", is_playable=False,
        )
        ckpt.characters.append(marcus)
        event = self._event(observer_ids=["marcus"])

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert marcus.pending_observations == []

    def test_in_scene_npc_observer_gets_in_scene_inbox_line(self):
        """v11-r8b regression guard: in-scene NPCs DO get pushed.
        Pre-r8b they were silently skipped, breaking the cascade."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()  # `pip` is at gatehouse
        event = self._event(observer_ids=["pip"])

        broadcast_event(ckpt, event, scene_id="gatehouse")
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert len(pip.pending_observations) == 1
        # No `[in-scene perception]` tag (r10) — the inbox IS the live
        # sensorium for the scene, no routing label needed.
        assert "note" in pip.pending_observations[0]

    def test_actor_excluded_from_inbox_push(self):
        """The actor of the event is the character whose intention the
        router just adjudicated. They don't need their own action
        echoed back into their inbox — their character_conversations
        history already carries the full assistant message they
        produced, and the in-scene perception of "you observed
        yourself doing the thing you just did" would be noise on
        their next on-stage turn."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()  # `pip` is at gatehouse
        nyx = CharacterRecord(
            character_id="nyx", name="Nyx",
            public_sheet=PublicSheet(role="npc"),
            location="gatehouse", is_playable=False,
        )
        ckpt.characters.append(nyx)
        # pip is the actor, both pip and nyx observe.
        event = self._event(observer_ids=["pip", "nyx"])

        broadcast_event(ckpt, event, scene_id="gatehouse", actor_id="pip")
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        # Actor (pip) skipped.
        assert pip.pending_observations == []
        # Non-actor in-scene observer (nyx) got the push.
        assert len(nyx.pending_observations) == 1

    def test_actor_off_scene_gets_no_push(self):
        """Off-scene observers receive nothing (r9b), so an off-scene
        actor — who would already be excluded by the actor-exclusion
        branch — has its own queue stay empty for two independent
        reasons. The combined assertion makes the failure mode
        clear: marcus is at citrus_garden, the event broadcasts at
        gatehouse, marcus is the actor; queue stays empty either
        way."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()
        marcus = CharacterRecord(
            character_id="marcus", name="Marcus",
            public_sheet=PublicSheet(role="contestant"),
            location="citrus_garden", is_playable=False,
        )
        ckpt.characters.append(marcus)
        event = self._event(observer_ids=["marcus"])

        broadcast_event(
            ckpt, event, scene_id="gatehouse", actor_id="marcus",
        )
        assert marcus.pending_observations == []

    def test_human_observer_NOT_pushed(self):
        """Humans get render buffer entries, not inbox lines. The
        narrator surfaces canonical events to humans through the
        rendered buffer; pushing onto a human's inbox would have no
        consumer (humans don't run agent prompts). True for both
        in-scene humans and off-scene humans (the latter just don't
        see the event at all this beat — they'll see it via narrator
        on their next /act if still bound)."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt(bindings={"alice": "1"})
        # Move alice off-scene so the only thing tested is the
        # human-vs-NPC branch, not the in-scene short-circuit.
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        alice.location = "remote_room"
        event = self._event(observer_ids=["alice"])

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert alice.pending_observations == []

    def test_in_scene_human_observer_NOT_pushed(self):
        """In-scene human is bound — gets render buffer, not inbox."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt(bindings={"alice": "1"})
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        alice.location = "gatehouse"
        event = self._event(observer_ids=["alice"])

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert alice.pending_observations == []

    def test_culled_observer_skipped(self):
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()
        ghost = CharacterRecord(
            character_id="ghost", name="Ghost",
            public_sheet=PublicSheet(role="ex"),
            location="citrus_garden", is_playable=False,
        )
        from app.schemas.characters import CharacterStatus
        ghost.status = CharacterStatus.culled
        ckpt.characters.append(ghost)
        event = self._event(observer_ids=["ghost"])

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert ghost.pending_observations == []

    def test_single_observable_fact_renders_inline(self):
        """One-fact event renders as the bare fact on a single line
        (no `[in-scene perception]` tag — r10). Marcus is in-scene at
        the gatehouse so the push fires."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()
        marcus = CharacterRecord(
            character_id="marcus", name="Marcus",
            public_sheet=PublicSheet(role="contestant"),
            location="gatehouse", is_playable=False,
        )
        ckpt.characters.append(marcus)

        observers = [
            ObserverEntry(
                character_id="marcus", observation_level="i",
                response_priority=2,
            ),
        ]
        event = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    feasible=True,
                    resolved_outcome="A bell rings twice in the distance.",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=["A distant bell rings twice."],
            ),
            observers=observers,
            requires_responders=False, required_responders=[],
            agent_responder_picks=[],
            ends_beat=True, ends_beat_reason="directed_at_player",
            spawn=[], dormant=[], cull=[], roster_moves=[], scenes_created=[],
        )

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert len(marcus.pending_observations) == 1
        assert marcus.pending_observations[0] == (
            "A distant bell rings twice."
        )

    def test_multiple_observable_facts_render_as_bulleted_block(self):
        """When an event carries multiple observable_facts, the inbox
        entry is a single multi-line bulleted block — one bullet per
        fact, no header line (r10 dropped the `[in-scene perception]`
        tag). Keeps each fact addressable while still landing as one
        queue entry per event."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()
        marcus = CharacterRecord(
            character_id="marcus", name="Marcus",
            public_sheet=PublicSheet(role="contestant"),
            location="gatehouse", is_playable=False,
        )
        ckpt.characters.append(marcus)

        observers = [
            ObserverEntry(
                character_id="marcus", observation_level="d",
                response_priority=3,
            ),
        ]
        event = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    feasible=True,
                    resolved_outcome="(audit summary)",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[
                    "Pip raises her glass to the table.",
                    "Pip says: 'To the new arrival.'",
                    "the room quiets and several glasses lift in response",
                ],
            ),
            observers=observers,
            requires_responders=False, required_responders=[],
            agent_responder_picks=[],
            ends_beat=True, ends_beat_reason="directed_at_player",
            spawn=[], dormant=[], cull=[], roster_moves=[], scenes_created=[],
        )

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert len(marcus.pending_observations) == 1
        entry = marcus.pending_observations[0]
        # No `[in-scene perception]` tag (r10) — entry is just the
        # bulleted facts, one per line.
        assert entry.startswith("  - Pip raises her glass to the table.")
        assert "  - Pip says: 'To the new arrival.'" in entry
        assert "  - the room quiets and several glasses lift in response" in entry

    def test_scoped_observable_fact_only_reaches_visible_recipients(self):
        """A mixed public/private event must not leak a scoped fact to
        every observer in the same scene."""
        from app.engine.turn_loop import broadcast_event
        from app.schemas.events import ObservableFact

        ckpt = _ckpt()
        ashara = CharacterRecord(
            character_id="ashara", name="Ashara",
            public_sheet=PublicSheet(role="heir"),
            location="gatehouse", is_playable=False,
        )
        aldric = CharacterRecord(
            character_id="aldric", name="Aldric",
            public_sheet=PublicSheet(role="heir"),
            location="gatehouse", is_playable=False,
        )
        ckpt.characters.extend([ashara, aldric])

        observers = [
            ObserverEntry(
                character_id="ashara", observation_level="d",
                response_priority=4,
            ),
            ObserverEntry(
                character_id="aldric", observation_level="d",
                response_priority=3,
            ),
        ]
        event = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    feasible=True,
                    resolved_outcome="Dan questions Thessaly and signals Ashara.",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[
                    ObservableFact.only(
                        "Dan's foot touches Ashara's boot under the table.",
                        ["ashara"],
                    ),
                    ObservableFact.all(
                        "Dan asks Thessaly whether she knows curses.",
                    ),
                ],
            ),
            observers=observers,
            requires_responders=False, required_responders=[],
            agent_responder_picks=[],
            ends_beat=True, ends_beat_reason="directed_at_player",
            spawn=[], dormant=[], cull=[], roster_moves=[], scenes_created=[],
        )

        broadcast_event(ckpt, event, scene_id="gatehouse", actor_id="alice")

        assert len(ashara.pending_observations) == 1
        assert "foot touches Ashara's boot" in ashara.pending_observations[0]
        assert "knows curses" in ashara.pending_observations[0]
        assert aldric.pending_observations == [
            "Dan asks Thessaly whether she knows curses."
        ]

    def test_resolved_outcome_does_NOT_leak_when_facts_differ(self):
        """Option B regression guard. When `resolved_outcome` carries
        narrator-grade interpretive prose ("the strain of speaking
        close to the edge of what she is permitted") and
        `observable_facts` carries the surface beats, NPCs must only
        receive the facts. The interpretive interior is the omniscient
        author's voice; pushing it as an agent's own perception was
        the t8 plague-verse leak this fix closes."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()
        marcus = CharacterRecord(
            character_id="marcus", name="Marcus",
            public_sheet=PublicSheet(role="contestant"),
            location="gatehouse", is_playable=False,
        )
        ckpt.characters.append(marcus)

        observers = [
            ObserverEntry(
                character_id="marcus", observation_level="d",
                response_priority=3,
            ),
        ]
        event = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    feasible=True,
                    resolved_outcome=(
                        "Seraphel recites a fractured plague verse, the "
                        "strain of speaking close to the edge of what she "
                        "is permitted showing in her wings."
                    ),
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[
                    "Seraphel recites: 'The plague that fell on human "
                    "ground / Killed only those who could be found'",
                    "her wings draw tight against her back, then flutter "
                    "sharply at the verse's end",
                ],
            ),
            observers=observers,
            requires_responders=False, required_responders=[],
            agent_responder_picks=[],
            ends_beat=True, ends_beat_reason="directed_at_player",
            spawn=[], dormant=[], cull=[], roster_moves=[], scenes_created=[],
        )

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert len(marcus.pending_observations) == 1
        entry = marcus.pending_observations[0]
        assert "wings draw tight" in entry
        assert "what she is permitted" not in entry
        assert "the strain of speaking" not in entry

    def test_empty_observable_facts_means_no_push(self):
        """An event with an empty `observable_facts` list produces no
        inbox push — silence is correct, and we explicitly do NOT
        fall back to `resolved_outcome`. A router that fails to emit
        any observable surface for a beat has a bug; this test
        guards against the engine masking that bug by piping the
        audit string."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()
        marcus = CharacterRecord(
            character_id="marcus", name="Marcus",
            public_sheet=PublicSheet(role="contestant"),
            location="gatehouse", is_playable=False,
        )
        ckpt.characters.append(marcus)

        observers = [
            ObserverEntry(
                character_id="marcus", observation_level="d",
                response_priority=3,
            ),
        ]
        event = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    feasible=True,
                    resolved_outcome=(
                        "A whole interpretive paragraph the engine must NOT "
                        "leak, because the router emitted no observable "
                        "facts to back it."
                    ),
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[],
            ),
            observers=observers,
            requires_responders=False, required_responders=[],
            agent_responder_picks=[],
            ends_beat=True, ends_beat_reason="directed_at_player",
            spawn=[], dormant=[], cull=[], roster_moves=[], scenes_created=[],
        )

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert marcus.pending_observations == []

    def test_cascade_pick_sees_just_broadcast_event_in_inbox(self):
        """End-to-end semantic guard: when a cascade pick is selected
        for `agent_intend`, their `pending_observations` already has
        the just-broadcast event as the most recent entry. This is
        what the on-stage `respond` call drains and reads as the
        live event.

        Pre-r8b this didn't work — in-scene NPCs were skipped, the
        dispatcher passed `observed_facts=[]`, and the cascade pick
        had nothing to react to. The downstream symptom was the
        router fabricating dialogue to fit a slot the agent never
        actually filled."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()  # alice (player), bob (player), pip (npc) at gatehouse
        nyx = CharacterRecord(
            character_id="nyx", name="Nyx",
            public_sheet=PublicSheet(role="npc"),
            location="gatehouse", is_playable=False,
        )
        ckpt.characters.append(nyx)
        # Player /act'd; their event observed by both pip and nyx.
        event = self._event(
            observer_ids=["pip", "nyx"],
            outcome="Alice says: 'Would anyone else like to introduce themselves?'",
        )

        broadcast_event(ckpt, event, scene_id="gatehouse", actor_id="alice")
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert "introduce themselves" in pip.pending_observations[-1]
        assert "introduce themselves" in nyx.pending_observations[-1]

    def test_multi_event_beat_accumulates_for_unpicked_npcs(self):
        """v11-r8b: NPCs in scene who AREN'T picked in a cascade
        accumulate every event in the beat. When they're eventually
        picked (this beat or a later one), they drain the whole
        accumulated set in one user message — no per-event LLM call
        wasted on agents who had no opening to speak.

        Models the real cascade flow: player /act → broadcast(p) →
        pick npc1 → broadcast(npc1) → pick npc2 → broadcast(npc2).
        Throughout, the unpicked NPC nyx silently witnesses all
        three events; her queue grows. When SHE is finally picked
        on the NEXT beat, she sees everything she missed."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()  # alice (player), bob (player), pip (npc) at gatehouse
        nyx = CharacterRecord(
            character_id="nyx", name="Nyx",
            public_sheet=PublicSheet(role="npc"),
            location="gatehouse", is_playable=False,
        )
        ckpt.characters.append(nyx)

        # Beat 1: player → cascade → cascade
        e_player = self._event(
            observer_ids=["pip", "nyx"],
            outcome="Alice raises a glass and toasts the table.",
        )
        broadcast_event(ckpt, e_player, scene_id="gatehouse", actor_id="alice")
        e_pip = self._event(
            observer_ids=["pip", "nyx"],
            outcome="Pip lifts her own glass with a smirk.",
        )
        broadcast_event(ckpt, e_pip, scene_id="gatehouse", actor_id="pip")
        # Pip was picked + acted, so her queue was drained between her
        # respond and the broadcast of her own event. We model that
        # explicitly here: her queue was empty going into her broadcast,
        # and her own event is excluded from her push by actor_id.
        # In production this is `clear_character_inbox` in respond().

        # nyx wasn't picked — she's been silently observing.
        assert len(nyx.pending_observations) == 2
        assert "toasts the table" in nyx.pending_observations[0]
        assert "lifts her own glass" in nyx.pending_observations[1]
        # When nyx is finally picked next beat, her respond() call
        # sees BOTH events in one user message via
        # format_pending_observations_block.
