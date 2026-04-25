"""Background tick scheduler — Commit 5 unit coverage.

Exercises `Orchestrator._eligible_for_tick` and `Orchestrator._run_ticks`
in isolation: trigger logic (scene-change vs stagnation), eligibility
filter, and concurrency-cap math. The actual `CharacterAgent.tick()`
call is monkeypatched so these tests don't hit the LLM.

Commit 6 will plumb the returned `(char, output)` list into a unified
router fan-in; that integration is covered separately. Here we only
care that the scheduler:

  - increments / resets `turns_since_last_tick` correctly,
  - fires on scene change after cooldown,
  - fires on stagnation regardless of scene,
  - filters out players, dormant, pinned, already-acted, and
    intentions-disabled NPCs,
  - honors `tick_concurrency` clamped against the engine hard cap,
  - persists `tick_last_scene_id` even when no fire happens.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.orchestrator import (
    Orchestrator,
    TICK_CONCURRENCY_HARD_CAP,
)
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import (
    EventRouterOutput,
    RosterMove,
    SceneCreation,
)
from app.schemas.events import CanonicalEvent, SceneDelta, WorldAdjudication
from app.schemas.state import (
    LocationState,
    OpenCatIIEvent,
    SessionState,
    SlotEntry,
    WorldState,
)


# ---- helpers ---------------------------------------------------------------


def _npc(
    char_id: str,
    *,
    intentions_enabled: bool = True,
    status: CharacterStatus = CharacterStatus.active,
    location: str = "courtyard",
    is_playable: bool = False,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=char_id,
        name=char_id.title(),
        status=status,
        location=location,
        is_playable=is_playable,
        public_sheet=PublicSheet(role="npc"),
        private_state=PrivateState(intentions_enabled=intentions_enabled),
    )


def _ckpt(
    *,
    bindings: dict[str, str] | None = None,
    player_character_id: str = "alice",
    characters: list[CharacterRecord] | None = None,
    turns_since_last_tick: int = 0,
    tick_last_scene_id: str = "",
    cooldown: int = 5,
    stagnation: int = 15,
    ticks_on_scene_change: bool = True,
    tick_concurrency: int = 4,
    ticks_enabled: bool = True,
) -> CheckpointFile:
    """Build a minimal v11 checkpoint with one courtyard scene plus
    optional adjacent rooms; tick scheduler defaults match the
    `SessionSettings` defaults but are overridable for tighter trigger
    tests."""
    sess = SessionState(
        session_id="s",
        player_character_id=player_character_id,
        character_bindings=bindings or {player_character_id: "u1"},
        turns_since_last_tick=turns_since_last_tick,
        tick_last_scene_id=tick_last_scene_id,
    )
    sess.config.settings.tick_scene_change_cooldown = cooldown
    sess.config.settings.tick_stagnation_max = stagnation
    sess.config.settings.ticks_on_scene_change = ticks_on_scene_change
    sess.config.settings.tick_concurrency = tick_concurrency
    sess.config.settings.ticks_enabled = ticks_enabled

    return CheckpointFile(
        session=sess,
        world_state=WorldState(
            locations=LocationState(
                scene_graph={
                    "courtyard": {
                        "name": "Courtyard",
                        "description": "Stone yard.",
                        "connected_to": ["hall", "library"],
                    },
                    "hall": {
                        "name": "Great Hall",
                        "description": "Long hall.",
                        "connected_to": ["courtyard"],
                    },
                    "library": {
                        "name": "Library",
                        "description": "Dusty stacks.",
                        "connected_to": ["courtyard"],
                    },
                },
            ),
        ),
        characters=characters or [
            CharacterRecord(
                character_id="alice", name="Alice", is_playable=True,
                location="courtyard",
                public_sheet=PublicSheet(role="player"),
            ),
            # Default NPCs live in `hall` (off-stage from the player's
            # courtyard) so the tick scheduler's "exclude characters
            # in the acting player's scene" filter doesn't drop them
            # by default. Tests that specifically want to exercise
            # the in-scene exclusion override `characters=` with NPCs
            # placed in `courtyard`.
            _npc("regent", location="hall"),
            _npc("scribe", location="hall"),
        ],
    )


def _orchestrator() -> Orchestrator:
    """Bare orchestrator with mocked client / prompt manager. No real
    LLM, no checkpoint manager I/O — tests call `_run_ticks` and
    `_eligible_for_tick` directly."""
    return Orchestrator(
        client=MagicMock(),
        checkpoint_mgr=MagicMock(),
        prompt_mgr=MagicMock(),
    )


def _patch_semaphore_recorder(monkeypatch) -> list[int]:
    """Replace `asyncio.Semaphore` with a recording proxy and return
    the list that captures every `__init__(value)` size.

    Captures a reference to the real Semaphore class FIRST, then swaps
    the module attribute — otherwise the proxy's own `Semaphore(value)`
    call would resolve back into itself and infinite-loop.
    """
    real_semaphore = asyncio.Semaphore
    sizes: list[int] = []

    class _ProbeSemaphore:
        def __init__(self, value):
            sizes.append(value)
            self._inner = real_semaphore(value)

        async def __aenter__(self):
            await self._inner.acquire()
            return self

        async def __aexit__(self, *exc):
            self._inner.release()

    monkeypatch.setattr(
        "app.engine.orchestrator.asyncio.Semaphore", _ProbeSemaphore,
    )
    return sizes


def _stub_character_agent(monkeypatch, recorder: list[str] | None = None):
    """Replace `app.engine.orchestrator.CharacterAgent` with a stub
    whose `tick()` returns a deterministic output and (optionally)
    records the called character_ids on `recorder`.

    Returns the stub class so tests can inspect call counts.
    """

    class _StubAgent:
        def __init__(self, client, prompt_mgr):
            self.client = client
            self.prompt_manager = prompt_mgr
            self.last_usage = {"input_tokens": 1, "output_tokens": 2}

        async def tick(self, *, character, checkpoint, acting_character_id):
            if recorder is not None:
                recorder.append(character.character_id)
            return CharacterAgentOutput(
                character_id=character.character_id,
                public_text=f"{character.name} paces.",
                intent="watch the gate",
            )

    monkeypatch.setattr(
        "app.engine.orchestrator.CharacterAgent", _StubAgent,
    )
    return _StubAgent


# ---- eligibility filter ----------------------------------------------------


class TestEligibility:
    def test_dormant_npc_excluded(self):
        ckpt = _ckpt(characters=[
            _npc("alice", is_playable=True),
            _npc("regent", status=CharacterStatus.dormant),
        ])
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(ckpt, acted_this_turn=set())
        assert [c.character_id for c in eligible] == []

    def test_culled_npc_excluded(self):
        ckpt = _ckpt(characters=[
            _npc("alice", is_playable=True),
            _npc("regent", status=CharacterStatus.culled),
        ])
        orch = _orchestrator()
        assert orch._eligible_for_tick(ckpt, acted_this_turn=set()) == []

    def test_intentions_disabled_excluded(self):
        ckpt = _ckpt(characters=[
            _npc("alice", is_playable=True),
            _npc("regent", intentions_enabled=False),
        ])
        orch = _orchestrator()
        assert orch._eligible_for_tick(ckpt, acted_this_turn=set()) == []

    def test_player_bound_npc_excluded(self):
        # Bind a non-player-flagged NPC to a user (e.g. /takeover) — even
        # though `is_playable=False`, character_bindings contains them, so
        # the scheduler treats them as a human and skips ticking.
        ckpt = _ckpt(
            bindings={"alice": "u1", "regent": "u2"},
            characters=[
                _npc("alice", is_playable=True),
                _npc("regent"),
                _npc("scribe"),
            ],
        )
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(ckpt, acted_this_turn=set())
        assert [c.character_id for c in eligible] == ["scribe"]

    def test_player_character_id_alone_excludes(self):
        # No character_bindings, only player_character_id set — still
        # excluded.
        ckpt = _ckpt(
            bindings={},
            player_character_id="alice",
            characters=[
                _npc("alice", is_playable=True),
                _npc("regent"),
            ],
        )
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(ckpt, acted_this_turn=set())
        assert [c.character_id for c in eligible] == ["regent"]

    def test_acted_this_turn_excluded(self):
        ckpt = _ckpt()
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(
            ckpt, acted_this_turn={"regent"},
        )
        assert [c.character_id for c in eligible] == ["scribe"]

    def test_pinned_npc_excluded(self):
        # Pin Regent into an initiator slot to simulate mid-action;
        # scheduler must skip them so the tick doesn't race the
        # pending Cat II resolution.
        ckpt = _ckpt()
        ckpt.session.active_act_slots["courtyard"] = {
            "regent": SlotEntry(reason="initiator"),
        }
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(ckpt, acted_this_turn=set())
        assert [c.character_id for c in eligible] == ["scribe"]

    def test_required_responder_excluded(self):
        # A character pinned as a Cat II responder is also "pinned" and
        # must be skipped.
        ckpt = _ckpt()
        ckpt.session.open_cat_ii_events.append(OpenCatIIEvent(
            event_id="evt1",
            scene_id="courtyard",
            initiator_id="alice",
            initiator_intention="(test)",
            required_responders=["regent"],
        ))
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(ckpt, acted_this_turn=set())
        assert [c.character_id for c in eligible] == ["scribe"]

    def test_active_eligible_npcs_in_roster_order(self):
        ckpt = _ckpt()
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(ckpt, acted_this_turn=set())
        assert [c.character_id for c in eligible] == ["regent", "scribe"]

    def test_in_scene_npcs_excluded_when_active_scene_set(self):
        # NPCs in the acting player's current scene already had their
        # opportunity via the on-stage cascade. Scribe is in "hall"
        # (off-stage); Regent is in "courtyard" (the active scene)
        # and must be filtered out.
        ckpt = _ckpt(characters=[
            _npc("alice", is_playable=True, location="courtyard"),
            _npc("regent", location="courtyard"),
            _npc("scribe", location="hall"),
        ])
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(
            ckpt, acted_this_turn=set(), active_scene="courtyard",
        )
        assert [c.character_id for c in eligible] == ["scribe"]

    def test_no_active_scene_means_no_scene_filter(self):
        # Defaulting active_scene to "" preserves the pre-filter
        # behavior — every test in this file that omits the arg should
        # see all eligible NPCs regardless of their location.
        ckpt = _ckpt()
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(
            ckpt, acted_this_turn=set(), active_scene="",
        )
        assert [c.character_id for c in eligible] == ["regent", "scribe"]


# ---- master kill switch ----------------------------------------------------


class TestTicksEnabledKillSwitch:
    """`SessionSettings.ticks_enabled = False` must short-circuit the
    scheduler entirely. No counter mutation, no eligibility filter, no
    fan-out, no router fan-in, no canonical-event append. The trigger
    state must freeze so flipping back on later resumes from where the
    model left off rather than firing a backlog.
    """

    @pytest.mark.asyncio
    async def test_disabled_does_not_fan_out_even_when_triggers_would_fire(
        self, monkeypatch,
    ):
        # Counter at stagnation cap AND scene change satisfied — both
        # branches would normally fire. With ticks_enabled=False the
        # scheduler must return [] without invoking the agent at all.
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            cooldown=5, stagnation=15,
            ticks_enabled=False,
        )
        orch = _orchestrator()
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="library",
        )

        assert result == []
        assert recorder == []

    @pytest.mark.asyncio
    async def test_disabled_freezes_trigger_state(self, monkeypatch):
        # The whole point of "freeze, don't drain" is that re-enabling
        # later resumes the model. The counter and last-scene id MUST
        # NOT mutate while disabled — otherwise flipping back on after
        # 50 disabled turns would either fire a backlog (counter
        # incremented) or amnesia the scene baseline (last-scene
        # cleared).
        ckpt = _ckpt(
            turns_since_last_tick=7, tick_last_scene_id="courtyard",
            ticks_enabled=False,
        )
        orch = _orchestrator()
        _stub_character_agent(monkeypatch)

        await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="hall",
        )

        # Frozen — neither counter incremented nor scene baseline
        # advanced.
        assert ckpt.session.turns_since_last_tick == 7
        assert ckpt.session.tick_last_scene_id == "courtyard"

    @pytest.mark.asyncio
    async def test_disabled_does_not_invoke_router_fan_in(
        self, monkeypatch,
    ):
        # Commit 6 fan-in is downstream of the gate; the kill switch
        # firing means LLMDispatcher.route_tick_intentions never runs.
        # Pin this with a recording stub on the dispatcher so a future
        # refactor that moves the gate after fan-in (silently keeping
        # the LLM call alive) trips the test.
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            cooldown=5, stagnation=15,
            ticks_enabled=False,
        )
        orch = _orchestrator()
        _stub_character_agent(monkeypatch)

        called: list[dict] = []

        async def _spy(self, **kwargs):
            called.append(kwargs)
            return None

        monkeypatch.setattr(
            "app.engine.turn_loop_dispatcher.LLMDispatcher."
            "route_tick_intentions",
            _spy,
        )

        await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="library",
        )

        assert called == []

    @pytest.mark.asyncio
    async def test_re_enabling_resumes_from_frozen_state(
        self, monkeypatch,
    ):
        # Disabled tick during turn N: counter stays at 14 (one short
        # of stagnation). Re-enable on turn N+1 with same counter and
        # baseline; now stagnation crosses the threshold and the fan-
        # out fires normally — proving the freeze preserved the
        # trigger model rather than draining it.
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="hall",
            cooldown=5, stagnation=15,
            ticks_enabled=False,
        )
        orch = _orchestrator()
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        # Disabled turn — no fire, state frozen.
        await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )
        assert recorder == []
        assert ckpt.session.turns_since_last_tick == 14

        # Re-enable; counter ticks to 15, hits stagnation, fires.
        ckpt.session.config.settings.ticks_enabled = True
        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )
        assert sorted(c.character_id for c, _ in result) == [
            "regent", "scribe",
        ]
        assert ckpt.session.turns_since_last_tick == 0


# ---- trigger logic ---------------------------------------------------------


class TestTriggerLogic:
    @pytest.mark.asyncio
    async def test_first_call_no_prior_scene_does_not_fire(
        self, monkeypatch,
    ):
        # tick_last_scene_id == "" means we have no baseline; can't
        # detect a "change" yet. Stagnation hasn't accrued either.
        # Counter must still increment, scene tracked.
        ckpt = _ckpt(turns_since_last_tick=0, tick_last_scene_id="")
        orch = _orchestrator()
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        assert result == []
        assert recorder == []
        assert ckpt.session.turns_since_last_tick == 1
        assert ckpt.session.tick_last_scene_id == "courtyard"

    @pytest.mark.asyncio
    async def test_same_scene_no_stagnation_does_not_fire(
        self, monkeypatch,
    ):
        ckpt = _ckpt(
            turns_since_last_tick=2, tick_last_scene_id="courtyard",
        )
        orch = _orchestrator()
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        assert result == []
        assert recorder == []
        assert ckpt.session.turns_since_last_tick == 3

    @pytest.mark.asyncio
    async def test_scene_change_under_cooldown_does_not_fire(
        self, monkeypatch,
    ):
        # Counter at 2 (incremented to 3 inside _run_ticks); cooldown is
        # 5 so 3 < 5 — scene change ignored.
        ckpt = _ckpt(
            turns_since_last_tick=2, tick_last_scene_id="courtyard",
            cooldown=5,
        )
        orch = _orchestrator()
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="hall",
        )

        assert result == []
        assert recorder == []
        assert ckpt.session.turns_since_last_tick == 3
        assert ckpt.session.tick_last_scene_id == "hall"

    @pytest.mark.asyncio
    async def test_scene_change_with_cooldown_satisfied_fires(
        self, monkeypatch,
    ):
        # Counter at 4 (incremented to 5); cooldown is 5 — fires.
        # Player moves to library; default NPCs (regent, scribe) are
        # in hall (off-stage from library) so the in-scene filter
        # doesn't drop them and both tick.
        ckpt = _ckpt(
            turns_since_last_tick=4, tick_last_scene_id="courtyard",
            cooldown=5,
        )
        orch = _orchestrator()
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="library",
        )

        assert sorted(c.character_id for c, _ in result) == [
            "regent", "scribe",
        ]
        assert sorted(recorder) == ["regent", "scribe"]
        assert ckpt.session.turns_since_last_tick == 0
        assert ckpt.session.tick_last_scene_id == "library"

    @pytest.mark.asyncio
    async def test_stagnation_fires_even_without_scene_change(
        self, monkeypatch,
    ):
        # Stagnation cap=3; counter=2 (becomes 3) — stagnation triggers
        # despite same scene.
        ckpt = _ckpt(
            turns_since_last_tick=2, tick_last_scene_id="courtyard",
            cooldown=5, stagnation=3,
        )
        orch = _orchestrator()
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        assert len(result) == 2
        assert ckpt.session.turns_since_last_tick == 0

    @pytest.mark.asyncio
    async def test_ticks_on_scene_change_disabled_only_stagnation_fires(
        self, monkeypatch,
    ):
        # Scene change WOULD fire but the toggle is off; counter not at
        # stagnation cap yet — no fire.
        ckpt = _ckpt(
            turns_since_last_tick=10, tick_last_scene_id="courtyard",
            cooldown=5, stagnation=15,
            ticks_on_scene_change=False,
        )
        orch = _orchestrator()
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="hall",
        )

        assert result == []
        assert recorder == []
        assert ckpt.session.turns_since_last_tick == 11
        assert ckpt.session.tick_last_scene_id == "hall"

    @pytest.mark.asyncio
    async def test_fire_with_no_eligible_resets_counter(
        self, monkeypatch,
    ):
        # Trigger fires (stagnation), but every NPC is excluded (e.g.
        # all dormant). Without the reset, the next turn would fire
        # again immediately.
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15,
            characters=[
                _npc("alice", is_playable=True),
                _npc("regent", status=CharacterStatus.dormant),
            ],
        )
        orch = _orchestrator()
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        assert result == []
        assert recorder == []
        assert ckpt.session.turns_since_last_tick == 0


# ---- concurrency / fan-out --------------------------------------------------


class TestFanOut:
    @pytest.mark.asyncio
    async def test_fans_out_one_call_per_eligible_npc(self, monkeypatch):
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15,
        )
        orch = _orchestrator()
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        # Both NPCs ticked exactly once.
        assert sorted(recorder) == ["regent", "scribe"]
        assert len(result) == 2
        # Each result entry is (CharacterRecord, CharacterAgentOutput).
        for char, out in result:
            assert isinstance(char, CharacterRecord)
            assert isinstance(out, CharacterAgentOutput)
            assert out.character_id == char.character_id

    @pytest.mark.asyncio
    async def test_concurrency_cap_respects_settings(self, monkeypatch):
        sizes = _patch_semaphore_recorder(monkeypatch)
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15, tick_concurrency=2,
        )
        orch = _orchestrator()
        _stub_character_agent(monkeypatch, recorder=[])
        await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )
        assert sizes == [2]

    @pytest.mark.asyncio
    async def test_concurrency_clamped_by_hard_cap(self, monkeypatch):
        sizes = _patch_semaphore_recorder(monkeypatch)
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15, tick_concurrency=999,
        )
        orch = _orchestrator()
        _stub_character_agent(monkeypatch, recorder=[])
        await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )
        # User asked for 999; engine clamps to TICK_CONCURRENCY_HARD_CAP.
        assert sizes == [TICK_CONCURRENCY_HARD_CAP]

    @pytest.mark.asyncio
    async def test_concurrency_floor_is_one(self, monkeypatch):
        # A misconfigured 0 (or negative) shouldn't deadlock with
        # `Semaphore(0)`. The scheduler clamps to a minimum of 1.
        sizes = _patch_semaphore_recorder(monkeypatch)
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15, tick_concurrency=0,
        )
        orch = _orchestrator()
        _stub_character_agent(monkeypatch, recorder=[])
        await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )
        assert sizes == [1]

    @pytest.mark.asyncio
    async def test_per_tick_failure_does_not_drop_others(self, monkeypatch):
        # Force one NPC's tick to raise; the other should still complete.
        class _PartialFailureAgent:
            def __init__(self, client, prompt_mgr):
                self.last_usage = {}

            async def tick(self, *, character, checkpoint, acting_character_id):
                if character.character_id == "regent":
                    raise RuntimeError("simulated API hiccup")
                return CharacterAgentOutput(
                    character_id=character.character_id,
                    public_text=f"{character.name} watches.",
                    intent="stay quiet",
                )

        monkeypatch.setattr(
            "app.engine.orchestrator.CharacterAgent", _PartialFailureAgent,
        )

        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15,
        )
        orch = _orchestrator()
        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        assert [c.character_id for c, _ in result] == ["scribe"]
        # Counter still resets — we DID fire, scribe just was the only
        # survivor. Otherwise a flapping NPC would lock the scheduler
        # into firing every turn forever.
        assert ckpt.session.turns_since_last_tick == 0


# ---- Commit 6: tick fan-in to unified router --------------------------------


def _tick_router_output(
    *,
    roster_moves: list[RosterMove] | None = None,
    scenes_created: list[SceneCreation] | None = None,
) -> EventRouterOutput:
    """Build a tick-mode router output with the invariants the prompt
    pins (no responders, no picks, ends_beat=true with ambient_pause).
    """
    return EventRouterOutput(
        event_id="",
        decision_rationale="(tick fan-in test fixture)",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                attempted_action="(off-stage tick)",
                feasible=True,
                resolved_outcome=(
                    "While the player rested, the regent paced and the "
                    "scribe copied another folio."
                ),
            ),
            scene_delta=SceneDelta(time_advanced_seconds=120),
            observable_facts=[],
        ),
        observers=[],
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="ambient_pause",
        spawn=[],
        dormant=[],
        cull=[],
        roster_moves=roster_moves or [],
        scenes_created=scenes_created or [],
    )


def _stub_dispatcher(
    monkeypatch,
    routed: EventRouterOutput | None,
    recorder: list[dict] | None = None,
    *,
    raise_on_call: Exception | None = None,
):
    """Replace `app.engine.orchestrator.LLMDispatcher` with a stub whose
    `route_tick_intentions` returns `routed` (or raises). When
    `recorder` is provided, every fan-in call appends a dict of the
    kwargs so tests can inspect what the orchestrator actually
    bundled.

    Returns the stub class for any extra introspection.
    """

    class _StubDispatcher:
        def __init__(self, client, prompt_mgr):
            self.client = client
            self.prompt_mgr = prompt_mgr

        async def route_tick_intentions(
            self, *, ckpt, tick_outputs, acting_character_id="",
        ):
            if recorder is not None:
                recorder.append({
                    "tick_outputs": list(tick_outputs),
                    "acting_character_id": acting_character_id,
                })
            if raise_on_call is not None:
                raise raise_on_call
            return routed

    monkeypatch.setattr(
        "app.engine.orchestrator.LLMDispatcher", _StubDispatcher,
    )
    return _StubDispatcher


class TestTickFanIn:
    @pytest.mark.asyncio
    async def test_fan_in_called_with_public_text_and_locations(
        self, monkeypatch,
    ):
        """The orchestrator must hand the dispatcher a list of
        `(name, character_id, location, public_text)` tuples — one
        per successful tick — and crucially must pass `public_text`
        only, NEVER `intent`. Information-asymmetry guard: the
        router must not see any agent's interior."""
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15,
        )
        _stub_character_agent(monkeypatch, recorder=[])
        recorded: list[dict] = []
        _stub_dispatcher(monkeypatch, _tick_router_output(), recorded)

        orch = _orchestrator()
        await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        assert len(recorded) == 1
        call = recorded[0]
        assert call["acting_character_id"] == "alice"
        # Two NPCs ticked → two fan-in entries.
        assert len(call["tick_outputs"]) == 2
        # Stub agent emits f"{name} paces." as public_text and a
        # non-empty intent. The intent ("watch the gate") MUST NOT
        # appear in any of the bundled entries.
        for name, char_id, location, public_text in call["tick_outputs"]:
            assert "paces" in public_text
            assert "watch the gate" not in public_text
            # Location is the off-stage scene the NPCs live in (`hall`
            # in the default fixture), not the player's `courtyard`.
            assert location == "hall"
            assert name in ("Regent", "Scribe")
            assert char_id in ("regent", "scribe")

    @pytest.mark.asyncio
    async def test_fan_in_skipped_when_no_ticks_succeed(
        self, monkeypatch,
    ):
        """If every per-tick attempt fails, there's nothing to bundle.
        The orchestrator must NOT spend a router call on an empty
        payload."""

        class _AlwaysFailAgent:
            def __init__(self, client, prompt_mgr):
                self.last_usage = {}

            async def tick(self, *, character, checkpoint, acting_character_id):
                raise RuntimeError("every tick fails this beat")

        monkeypatch.setattr(
            "app.engine.orchestrator.CharacterAgent", _AlwaysFailAgent,
        )
        recorded: list[dict] = []
        _stub_dispatcher(monkeypatch, _tick_router_output(), recorded)

        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15,
        )
        orch = _orchestrator()
        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        assert result == []
        assert recorded == []
        # Counter still reset — the fire happened, just no survivors
        # to bundle.
        assert ckpt.session.turns_since_last_tick == 0

    @pytest.mark.asyncio
    async def test_fan_in_applies_roster_moves_from_router(
        self, monkeypatch,
    ):
        """Off-stage ticks unfold via the same movement primitive the
        on-stage cascade uses (`roster_moves`). When the router
        returns one, the orchestrator must apply it to character
        locations on the checkpoint."""
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15,
        )
        # Regent starts in `hall` (default off-stage fixture). The
        # tick router relocates them to `library`.
        routed = _tick_router_output(
            roster_moves=[
                RosterMove(
                    character_id="regent", to_scene="library",
                    reason="walked to fetch a folio",
                ),
            ],
        )
        _stub_character_agent(monkeypatch, recorder=[])
        _stub_dispatcher(monkeypatch, routed)

        orch = _orchestrator()
        await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        regent = next(c for c in ckpt.characters if c.character_id == "regent")
        assert regent.location == "library"

    @pytest.mark.asyncio
    async def test_fan_in_applies_scene_creations_from_router(
        self, monkeypatch,
    ):
        """An off-stage tick can imply a brand-new scene (an NPC walks
        into a side room the importer didn't enumerate). The
        orchestrator must grow the scene_graph before applying any
        moves that target the new scene."""
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15,
        )
        routed = _tick_router_output(
            scenes_created=[
                SceneCreation(
                    scene_id="archive", name="Archive",
                    description="A dusty room of records.",
                    connected_to=["hall"],
                ),
            ],
        )
        _stub_character_agent(monkeypatch, recorder=[])
        _stub_dispatcher(monkeypatch, routed)

        orch = _orchestrator()
        await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        assert "archive" in ckpt.world_state.locations.scene_graph
        # Reverse edge added automatically by `_apply_scene_creations`
        # so the graph stays bidirectionally traversable.
        assert (
            "archive"
            in ckpt.world_state.locations.scene_graph["hall"]["connected_to"]
        )

    @pytest.mark.asyncio
    async def test_fan_in_appends_canonical_event_to_world_log(
        self, monkeypatch,
    ):
        """The off-stage canonical event the router authors lands in
        `ckpt.canonical_events` so future router calls + recap
        passes see it as part of session truth, even though no
        narrator render happens (the player wasn't there)."""
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15,
        )
        before = len(ckpt.canonical_events)
        routed = _tick_router_output()
        _stub_character_agent(monkeypatch, recorder=[])
        _stub_dispatcher(monkeypatch, routed)

        orch = _orchestrator()
        await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        assert len(ckpt.canonical_events) == before + 1
        appended = ckpt.canonical_events[-1]
        assert appended.canonical_event.world_adjudication.resolved_outcome \
            .startswith("While the player rested")

    @pytest.mark.asyncio
    async def test_fan_in_router_failure_swallowed_no_state_corruption(
        self, monkeypatch,
    ):
        """Router fan-in failure (network, schema, anything) MUST
        NOT crash the beat — the on-stage render already landed for
        the player. The per-character tick outputs are still in the
        agents' rolling histories (interior continuity preserved
        for next call); we just don't get a canonical event for
        the off-stage developments this turn."""
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15,
        )
        before = len(ckpt.canonical_events)
        _stub_character_agent(monkeypatch, recorder=[])
        _stub_dispatcher(
            monkeypatch, None,
            raise_on_call=RuntimeError("simulated fan-in API hiccup"),
        )

        orch = _orchestrator()
        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        # Per-tick outputs survive — the orchestrator's return value
        # still reflects what the agents produced before the fan-in
        # failed.
        assert sorted(c.character_id for c, _ in result) == [
            "regent", "scribe",
        ]
        # No canonical event appended (the router never returned).
        assert len(ckpt.canonical_events) == before
        # Counter still reset — the FIRE happened, even though the
        # downstream router call failed.
        assert ckpt.session.turns_since_last_tick == 0

    @pytest.mark.asyncio
    async def test_fan_in_router_returns_none_no_mutations(
        self, monkeypatch,
    ):
        """Defensive: if the dispatcher returns None (legacy path or
        future change), the orchestrator must skip mutation
        application without crashing on `routed.scenes_created` /
        `routed.roster_moves` access."""
        ckpt = _ckpt(
            turns_since_last_tick=14, tick_last_scene_id="courtyard",
            stagnation=15,
        )
        before_events = len(ckpt.canonical_events)
        before_scene_graph = dict(ckpt.world_state.locations.scene_graph)
        _stub_character_agent(monkeypatch, recorder=[])
        _stub_dispatcher(monkeypatch, None)  # router returns None

        orch = _orchestrator()
        result = await orch._run_ticks(
            ckpt, acted_this_turn=set(),
            acting_id="alice", current_scene="courtyard",
        )

        assert len(result) == 2  # fan-out succeeded
        assert len(ckpt.canonical_events) == before_events  # no append
        assert ckpt.world_state.locations.scene_graph == before_scene_graph
