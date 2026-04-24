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
from app.schemas.events import CanonicalEvent, SceneDelta, WorldAdjudication
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
                is_player=True,
            ),
            CharacterRecord(
                character_id="bob", name="Bob",
                public_sheet=PublicSheet(role="player"),
                location="gatehouse",
                is_player=True,
            ),
            CharacterRecord(
                character_id="pip", name="Pip",
                public_sheet=PublicSheet(role="npc"),
                location="gatehouse",
                is_player=False,
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
                attempted_action="something",
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
        self._route_responses: list[EventRouterOutput] = []
        self._agent_responses: list[str] = []
        self._narrator_response: str = "RENDER"

    def queue_route(self, response: EventRouterOutput) -> None:
        self._route_responses.append(response)

    def queue_agent(self, intention: str) -> None:
        self._agent_responses.append(intention)

    async def route_intention(self, **kw) -> EventRouterOutput:
        self.route_calls.append(kw)
        return self._route_responses.pop(0)

    async def agent_intend(self, **kw) -> str:
        self.agent_calls.append(kw)
        return self._agent_responses.pop(0)

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
        # Synthetic open-attempt event was appended to canonical_events
        # so the narrator's event-id lookup can resolve it.
        assert any(
            evt.ends_beat_reason == "cat_ii_open"
            for evt in ckpt.canonical_events
        )
        # Every narrator call used partial_mode_override=True.
        assert fake.narrator_calls, "expected at least one narrator call"
        for call in fake.narrator_calls:
            assert call.get("partial_mode_override") is True

    def test_cat_ii_open_with_only_agent_responder_no_partial_render(self):
        """Cat II that resolves inline with only agent responders should
        NOT take the partial-render path — no pins held, no open_attempt
        synthesized."""
        ckpt = _ckpt(bindings={"alice": "1"})
        fake = FakeDispatcher()
        fake.queue_route(_router_out(
            requires_responders=True,
            required_responders=["pip"],
            ends_beat=False,
        ))
        fake.queue_agent("Pip dodges")
        fake.queue_route(_router_out(ends_beat=True))

        result = asyncio.run(run_beat(
            ckpt=ckpt, dispatcher=fake,
            actor_id="alice", intention="I attack Pip",
            scene_id="gatehouse",
        ))
        # Resolved inline.
        assert result.ended_reason == "cat_ii_resolution"
        # No open_attempt synthesis.
        assert not any(
            evt.ends_beat_reason == "cat_ii_open"
            for evt in ckpt.canonical_events
        )
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

        assert result.events_closed == 1
        assert result.ended_reason == "cat_ii_resolution"
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
        """v11-r5: refusal detection moved to the prompt (agent_v9 rule
        18). If a misbehaving model returns refusal text anyway, the
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
                    attempted_action="x", feasible=True, resolved_outcome="x",
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
                is_player=cid in (bindings or {}),
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


class TestAgentEmptyGuard:
    """v11-r5: refusal detection moved from pattern matching to
    prompt-level (agent_v9 rule 21). The engine only guards literal
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
                    attempted_action="x", feasible=True, resolved_outcome="y",
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
                    attempted_action="x", feasible=True, resolved_outcome="y",
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


class TestBroadcastEventOffSceneObservers:
    """v11-r7j: `broadcast_event` pushes a `[off-scene perception]`
    line onto every NPC observer who is NOT in the broadcast scene.
    This is the engine implementation of router rule 13's cross-scene
    perception channel — pre-r7j, declaring an off-scene recipient as
    an `observer` was decorative; the recipient agent never saw a
    mechanical signal."""

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
                    attempted_action="hand off note",
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

    def test_off_scene_npc_observer_gets_inbox_line(self):
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()
        # marcus is an off-scene NPC
        marcus = CharacterRecord(
            character_id="marcus", name="Marcus",
            public_sheet=PublicSheet(role="contestant"),
            location="citrus_garden", is_player=False,
        )
        ckpt.characters.append(marcus)
        event = self._event(observer_ids=["marcus"])

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert len(marcus.pending_observations) == 1
        assert marcus.pending_observations[0].startswith(
            "[off-scene perception]"
        )
        assert "note" in marcus.pending_observations[0]

    def test_in_scene_npc_observer_NOT_pushed(self):
        """In-scene NPC observers read the canonical event live via
        their normal context block when picked as responders. Pushing
        onto their inbox would double-count."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()  # `pip` is at gatehouse
        event = self._event(observer_ids=["pip"])

        broadcast_event(ckpt, event, scene_id="gatehouse")
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert pip.pending_observations == []

    def test_human_observer_NOT_pushed(self):
        """Humans get render buffer entries, not inbox lines. The
        narrator surfaces canonical events to humans through the
        rendered buffer; pushing onto a human's inbox would have no
        consumer (humans don't run agent prompts)."""
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt(bindings={"alice": "1"})
        # Move alice off-scene so the only thing tested is the
        # human-vs-NPC branch, not the in-scene short-circuit.
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        alice.location = "remote_room"
        event = self._event(observer_ids=["alice"])

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert alice.pending_observations == []

    def test_culled_observer_skipped(self):
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()
        ghost = CharacterRecord(
            character_id="ghost", name="Ghost",
            public_sheet=PublicSheet(role="ex"),
            location="citrus_garden", is_player=False,
        )
        from app.schemas.characters import CharacterStatus
        ghost.status = CharacterStatus.culled
        ckpt.characters.append(ghost)
        event = self._event(observer_ids=["ghost"])

        broadcast_event(ckpt, event, scene_id="gatehouse")
        assert ghost.pending_observations == []

    def test_no_resolved_outcome_falls_back_to_first_fact(self):
        from app.engine.turn_loop import broadcast_event
        ckpt = _ckpt()
        marcus = CharacterRecord(
            character_id="marcus", name="Marcus",
            public_sheet=PublicSheet(role="contestant"),
            location="citrus_garden", is_player=False,
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
                    attempted_action="echo",
                    feasible=True, resolved_outcome="",
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
        assert "bell rings" in marcus.pending_observations[0]
