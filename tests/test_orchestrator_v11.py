"""v11 orchestrator integration tests.

Covers the thin-wrapper layer between `Orchestrator.process_turn` and
`turn_loop.run_beat`: slot conflicts turn into rejection responses,
happy path populates per_player_renders, Cat II pending surfaces as an
empty-render / beat_ended_reason="cat_ii_pending" response, and
roster-move guards still fire after the beat closes.

The LLM layer is stubbed by monkeypatching `LLMDispatcher` in the
orchestrator module with a deterministic `FakeDispatcher`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.orchestrator import Orchestrator
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry, RosterMove
from app.schemas.events import CanonicalEvent, SceneDelta, WorldAdjudication
from app.schemas.requests import TurnRequest
from app.schemas.state import LocationState, SessionState, WorldState


# ---- helpers ---------------------------------------------------------------


def _ckpt(bindings: dict[str, str] | None = None) -> CheckpointFile:
    """Build a minimal v11 checkpoint: gatehouse scene, alice+bob+pip roster,
    optional player bindings."""
    return CheckpointFile(
        session=SessionState(
            session_id="s",
            turn_index=0,
            player_character_id="alice",
            character_bindings=bindings or {"alice": "u1"},
        ),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="gatehouse",
                scene_graph={
                    "gatehouse": {
                        "name": "Gatehouse",
                        "description": "A stone gatehouse.",
                        "connected_to": [],
                    },
                    "threshold": {
                        "name": "Threshold",
                        "description": "An archway.",
                        "connected_to": [],
                    },
                },
            ),
        ),
        characters=[
            CharacterRecord(
                character_id="alice", name="Alice",
                public_sheet=PublicSheet(role="player"),
                location="gatehouse", is_player=True,
            ),
            CharacterRecord(
                character_id="bob", name="Bob",
                public_sheet=PublicSheet(role="player"),
                location="gatehouse", is_player=True,
            ),
            CharacterRecord(
                character_id="pip", name="Pip",
                public_sheet=PublicSheet(role="guard"),
                location="gatehouse", is_player=False,
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
    roster_moves: list[RosterMove] | None = None,
) -> EventRouterOutput:
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
    if not ends_beat:
        # Only the ends_beat=True path supplies a reason; the router's
        # "keep going" output should carry an empty reason.
        ends_beat_reason = ""
    return EventRouterOutput(
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                attempted_action="something",
                feasible=True,
                resolved_outcome="something happens",
            ),
            scene_delta=SceneDelta(time_advanced_seconds=0, new_scene_id=""),
            observable_facts=[],
        ),
        observers=observers,
        requires_responders=requires_responders,
        required_responders=required_responders or [],
        agent_responder_picks=picks,
        ends_beat=ends_beat,
        ends_beat_reason=ends_beat_reason,
        roster_moves=roster_moves or [],
    )


class FakeDispatcher:
    """Queueable Router / Agent / Narrator stub. The orchestrator's
    LLMDispatcher symbol is monkeypatched with this class so
    `run_beat` receives deterministic outputs.

    Accepts (client, prompt_mgr) in __init__ to match the real
    dispatcher's constructor signature; the orchestrator constructs it
    via `LLMDispatcher(self.client, self.prompt_mgr)`.
    """

    # Class-level queues so the monkeypatched constructor (which takes
    # the orchestrator's client + prompt_mgr, not our test args) can be
    # primed before `process_turn` is awaited.
    _route_responses: list = []
    _agent_responses: list = []
    _narrator_text: str = "POV_RENDER"
    # Call logs.
    route_calls: list = []
    agent_calls: list = []
    narrator_calls: list = []

    def __init__(self, *args, **kwargs):
        # Ignore the orchestrator's (client, prompt_mgr) — tests drive
        # state via class-level queues/logs instead.
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

    async def agent_intend(self, **kw) -> str:
        type(self).agent_calls.append(kw)
        return type(self)._agent_responses.pop(0)

    async def narrator_compose(self, **kw) -> str:
        type(self).narrator_calls.append(kw)
        return type(self)._narrator_text


# ---- fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeDispatcher.reset()
    yield
    FakeDispatcher.reset()


@pytest.fixture
def patched_orchestrator(monkeypatch):
    """Factory that returns a new Orchestrator with:
      - a mocked LLMClient (never actually called),
      - an in-memory CheckpointManager stub that returns whatever ckpt
        is passed to .set_ckpt() and records .save() calls,
      - LLMDispatcher patched to FakeDispatcher.
    """
    monkeypatch.setattr(
        "app.engine.orchestrator.LLMDispatcher", FakeDispatcher,
    )

    client = MagicMock()
    client.config = MagicMock()
    prompt_mgr = MagicMock()

    def _factory(ckpt: CheckpointFile):
        mgr = MagicMock()
        mgr.load_latest.return_value = ckpt
        # Record what .save() received so tests can inspect the saved
        # checkpoint's mutations.
        mgr.save = MagicMock()
        orch = Orchestrator(client, mgr, prompt_mgr)
        return orch, mgr

    return _factory


# ---- tests -----------------------------------------------------------------


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

        # Render fan-out hit alice.
        assert "alice" in response.per_player_renders
        assert response.per_player_renders["alice"] == "POV_RENDER"
        # Back-compat field mirrors the actor's own render.
        assert response.output_text == "POV_RENDER"
        assert response.beat_ended_reason == "directed_at_player"
        # Turn index incremented from 0 to 1.
        assert response.turn_index == 1
        # Save called exactly once (no short-circuit).
        assert mgr.save.call_count == 1
        # The saved checkpoint has the canonical event appended.
        saved = mgr.save.call_args[0][0]
        assert len(saved.canonical_events) == 1
        # Scene slot released at beat end.
        assert "gatehouse" not in saved.session.active_act_slots


class TestSlotRejection:
    @pytest.mark.asyncio
    async def test_second_act_against_held_slot_rejected_without_save(
        self, patched_orchestrator,
    ):
        # Pre-state: alice holds the initiator slot. Bob /acts — gets
        # rejected.
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        from app.engine.turn_loop import claim_initiator_slot
        claim_initiator_slot(ckpt, "gatehouse", "alice")

        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I speak up",
            acting_character_id="bob",
        ))

        # Rejection message present.
        assert "didn't go through" in response.output_text
        # No renders produced.
        assert response.per_player_renders == {}
        assert response.beat_ended_reason == "slot_rejected"
        # Crucially: NO checkpoint mutation / save.
        assert mgr.save.call_count == 0
        # FakeDispatcher's route_intention was never called.
        assert FakeDispatcher.route_calls == []


class TestCatIIPending:
    @pytest.mark.asyncio
    async def test_cat_ii_against_human_pauses_beat_and_persists_open_event(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        orch, mgr = patched_orchestrator(ckpt)

        # Alice attacks Bob (human); Cat II opens, beat pauses.
        FakeDispatcher.queue_route(_router_out(
            requires_responders=True,
            required_responders=["bob"],
            ends_beat=False,
            ends_beat_reason="",
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I attack Bob",
            acting_character_id="alice",
        ))

        # Beat paused on the open event.
        assert response.per_player_renders == {}
        assert response.output_text == ""
        assert response.beat_ended_reason == "cat_ii_pending"
        # Checkpoint saved even on pause (the open event + pin must
        # persist across process restarts).
        assert mgr.save.call_count == 1
        saved = mgr.save.call_args[0][0]
        assert len(saved.session.open_cat_ii_events) == 1
        evt = saved.session.open_cat_ii_events[0]
        assert evt.initiator_id == "alice"
        assert evt.required_responders == ["bob"]
        # Bob is pinned as a Cat II responder.
        slot = saved.session.active_act_slots.get("gatehouse", {})
        assert "bob" in slot
        assert slot["bob"].reason == "cat_ii_responder"


class TestRosterMoveGuard:
    @pytest.mark.asyncio
    async def test_pinned_character_not_moved_by_router_output(
        self, patched_orchestrator,
    ):
        """A pinned character's roster_move is skipped: the post-beat
        apply pass walks each closed event's roster_moves but the
        pinned-character guard in `_apply_roster_moves` prevents the
        move. Use a Cat II path so Bob ends up pinned while still
        having a closed event whose roster_moves need applying.

        Flow:
          1. Alice attacks Bob — Cat II opens, Bob pinned (no event
             closed yet).
          2. Bob /acts his response — the Cat II resolves to a single
             canonical event that (maliciously) carries a roster_move
             for an OTHER pinned responder (we reuse alice as pinned
             via a second open event we seed by hand to keep the test
             hermetic). Since that character is pinned, the move is
             skipped and their location stays unchanged.
        """
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})

        # Seed a pre-existing open Cat II event that pins alice as a
        # responder — independent of the beat we run here. `alice`'s
        # location stays "gatehouse"; any roster_move the router emits
        # against her should be dropped by the guard.
        from app.engine.turn_loop import open_cat_ii, pin_cat_ii_responder
        side_evt = open_cat_ii(
            ckpt, scene_id="threshold",
            initiator_id="pip", initiator_intention="pip glares at alice",
            required_responders=["alice"],
        )
        pin_cat_ii_responder(ckpt, "threshold", "alice", side_evt.event_id)

        orch, mgr = patched_orchestrator(ckpt)

        # Bob /acts (fresh initiator in gatehouse). Router returns a
        # Cat I event whose roster_moves try to relocate pinned alice.
        FakeDispatcher.queue_route(_router_out(
            ends_beat=True,
            roster_moves=[
                RosterMove(
                    character_id="alice",
                    to_scene="threshold",
                    reason="router tried to move her",
                ),
            ],
        ))

        await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I pace.",
            acting_character_id="bob",
        ))

        saved = mgr.save.call_args[0][0]
        alice = next(c for c in saved.characters if c.character_id == "alice")
        # Guard fired: alice is still in gatehouse despite the
        # router_move the event carried.
        assert alice.location == "gatehouse"
