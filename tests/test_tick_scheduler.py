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
    is_player: bool = False,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=char_id,
        name=char_id.title(),
        status=status,
        location=location,
        is_player=is_player,
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

    return CheckpointFile(
        session=sess,
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="courtyard",
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
                character_id="alice", name="Alice", is_player=True,
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
            _npc("alice", is_player=True),
            _npc("regent", status=CharacterStatus.dormant),
        ])
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(ckpt, acted_this_turn=set())
        assert [c.character_id for c in eligible] == []

    def test_culled_npc_excluded(self):
        ckpt = _ckpt(characters=[
            _npc("alice", is_player=True),
            _npc("regent", status=CharacterStatus.culled),
        ])
        orch = _orchestrator()
        assert orch._eligible_for_tick(ckpt, acted_this_turn=set()) == []

    def test_intentions_disabled_excluded(self):
        ckpt = _ckpt(characters=[
            _npc("alice", is_player=True),
            _npc("regent", intentions_enabled=False),
        ])
        orch = _orchestrator()
        assert orch._eligible_for_tick(ckpt, acted_this_turn=set()) == []

    def test_player_bound_npc_excluded(self):
        # Bind a non-player-flagged NPC to a user (e.g. /takeover) — even
        # though `is_player=False`, character_bindings contains them, so
        # the scheduler treats them as a human and skips ticking.
        ckpt = _ckpt(
            bindings={"alice": "u1", "regent": "u2"},
            characters=[
                _npc("alice", is_player=True),
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
                _npc("alice", is_player=True),
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
            _npc("alice", is_player=True, location="courtyard"),
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
                _npc("alice", is_player=True),
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
