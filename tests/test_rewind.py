"""Tests for EngineBridge.rewind_session and preview_rewind.

The storage primitive (CheckpointManager.delete_checkpoints_after) is
covered separately in tests/test_checkpoint.py — this file focuses on
the bridge-level orchestration: validation, preview vs. commit semantics,
result metadata, the per-session lock, and the round-trip property
that loading the post-rewind ckpt produces the exact same state as the
target ckpt did before the deleted turns ran.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.bot.engine_bridge import EngineBridge, RewindResult
from app.schemas.characters import (
    CharacterRecord,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import (
    EventRouterOutput,
    ObserverEntry,
)
from app.schemas.events import (
    CanonicalEvent,
    SceneDelta,
    WorldAdjudication,
)
from app.schemas.state import (
    LocationState,
    SessionState,
    WorldState,
)


SESSION_ID = "rewind_test"


def _make_ckpt(
    *,
    turn_index: int,
    canonical_event_count: int = 0,
    scene_id: str = "hall",
    actor_id: str = "aldric",
    extra_chars: list[CharacterRecord] | None = None,
) -> CheckpointFile:
    """Build a single checkpoint at a given turn_index. Each canonical
    event is unique enough that round-trip checks can verify the right
    snapshot was preserved."""
    chars = [
        CharacterRecord(
            character_id=actor_id,
            name="Aldric Verantus",
            is_playable=True,
            location=scene_id,
            public_sheet=PublicSheet(role="envoy"),
        ),
    ]
    if extra_chars:
        chars.extend(extra_chars)

    events = []
    for i in range(canonical_event_count):
        events.append(
            EventRouterOutput(
                event_id=f"evt_t{turn_index}_{i}",
                decision_rationale="(test fixture)",
                canonical_event=CanonicalEvent(
                    world_adjudication=WorldAdjudication(
                        feasible=True,
                        resolved_outcome=f"Turn {turn_index} event {i}.",
                    ),
                    scene_delta=SceneDelta(time_advanced_seconds=0),
                    observable_facts=[],
                ),
                observers=[
                    ObserverEntry(
                        character_id=actor_id,
                        observation_level="d",
                        response_priority=3,
                    ),
                ],
                requires_responders=False,
                required_responders=[],
                agent_responder_picks=[],
                ends_beat=True,
                ends_beat_reason="cascade_exhausted",
                spawn=[],
                dormant=[],
                cull=[],
                roster_moves=[],
                scenes_created=[],
            )
        )

    facts = [f"fact at turn {turn_index}"]
    return CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            turn_index=turn_index,
            player_character_id=actor_id,
            character_bindings={actor_id: "12345"},
            # Pre-populate surfaced_world_facts to match the facts
            # list, so the on-load backfill in checkpoint_manager
            # (which fires when surfaced is empty AND facts is not)
            # is a no-op. Lets the round-trip equality test be a
            # true round-trip.
            surfaced_world_facts=list(facts),
        ),
        world_state=WorldState(
            facts=facts,
            locations=LocationState(
                scene_graph={
                    scene_id: {
                        "name": "Great Hall",
                        "description": "",
                        "connected_to": [],
                        "properties": {},
                    },
                },
            ),
        ),
        characters=chars,
        canonical_events=events,
    )


@pytest.fixture
def bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return EngineBridge(saves_dir=str(tmp_path), prompts_dir="app/prompts")


def _seed_session(
    bridge: EngineBridge, *, last_turn: int,
) -> list[int]:
    """Save checkpoints for turns 0..last_turn inclusive, with a
    growing canonical_events list so round-trips can verify the exact
    state that was preserved. Returns the list of saved turn indices."""
    saved = []
    for t in range(last_turn + 1):
        bridge.checkpoint_mgr.save(
            _make_ckpt(turn_index=t, canonical_event_count=t),
        )
        saved.append(t)
    return saved


# ---- preview_rewind ---------------------------------------------------------


class TestPreviewRewind:
    """Validation-and-summary path. Should never mutate disk."""

    def test_preview_returns_expected_result(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=5)

        result = bridge.preview_rewind(SESSION_ID, target_turn=2)

        assert isinstance(result, RewindResult)
        assert result.session_id == SESSION_ID
        assert result.target_turn == 2
        assert result.previous_latest == 5
        assert result.new_latest == 2
        assert result.deleted_turns == [3, 4, 5]
        assert result.actor_character_id == "aldric"
        assert result.scene_id == "hall"

    def test_preview_does_not_touch_disk(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=4)

        bridge.preview_rewind(SESSION_ID, target_turn=1)

        # Nothing deleted.
        assert bridge.list_checkpoint_turns(SESSION_ID) == [0, 1, 2, 3, 4]

    def test_preview_empty_session_raises(self, bridge: EngineBridge):
        # No checkpoints saved at all.
        with pytest.raises(FileNotFoundError, match="no checkpoints"):
            bridge.preview_rewind(SESSION_ID, target_turn=0)

    def test_preview_negative_target_raises(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=3)
        with pytest.raises(ValueError, match="must be >= 0"):
            bridge.preview_rewind(SESSION_ID, target_turn=-1)

    def test_preview_unknown_target_raises(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=3)
        with pytest.raises(ValueError, match="has no checkpoint"):
            bridge.preview_rewind(SESSION_ID, target_turn=99)

    def test_preview_target_at_latest_raises(self, bridge: EngineBridge):
        # "Rewind to current state" is meaningless — refuse with a
        # clear message rather than execute a no-op cull.
        _seed_session(bridge, last_turn=3)
        with pytest.raises(ValueError, match="already the current state"):
            bridge.preview_rewind(SESSION_ID, target_turn=3)

    def test_preview_target_above_latest_raises(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=3)
        with pytest.raises(ValueError, match="has no checkpoint"):
            bridge.preview_rewind(SESSION_ID, target_turn=10)


# ---- rewind_session ---------------------------------------------------------


class TestRewindSessionCull:
    """The actual cull. Should produce on-disk state matching the
    preview, and the resulting session should load_latest cleanly to
    the target checkpoint."""

    def test_rewind_culls_after_target(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=5)

        result = asyncio.run(bridge.rewind_session(SESSION_ID, 2))

        assert result.target_turn == 2
        assert result.deleted_turns == [3, 4, 5]
        assert result.new_latest == 2
        assert bridge.list_checkpoint_turns(SESSION_ID) == [0, 1, 2]

    def test_rewind_preserves_target_checkpoint_intact(
        self, bridge: EngineBridge,
    ):
        # The target checkpoint MUST survive byte-for-byte. Round-trip
        # via load_latest after the cull and verify the canonical
        # events match what was at turn 2 before any subsequent turns
        # ran.
        _seed_session(bridge, last_turn=5)
        ckpt_before = bridge.checkpoint_mgr.load(SESSION_ID, "ckpt_0002")

        asyncio.run(bridge.rewind_session(SESSION_ID, 2))

        ckpt_after = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        assert ckpt_after.session.turn_index == 2
        assert (
            ckpt_after.canonical_events
            == ckpt_before.canonical_events
        )
        assert ckpt_after.world_state.facts == ["fact at turn 2"]

    def test_rewind_to_zero_keeps_origin(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=4)

        result = asyncio.run(bridge.rewind_session(SESSION_ID, 0))

        assert result.deleted_turns == [1, 2, 3, 4]
        assert bridge.list_checkpoint_turns(SESSION_ID) == [0]

    def test_rewind_state_round_trip_unchanged(self, bridge: EngineBridge):
        # Comprehensive snapshot equality: serializing the loaded
        # post-rewind ckpt should match the target's saved JSON
        # exactly. This is the "perfect state replication" the user
        # asked for — if we miss a field, this test catches it.
        _seed_session(bridge, last_turn=4)
        target_path = (
            Path(bridge.checkpoint_mgr.save_dir)
            / SESSION_ID
            / "ckpt_0002.json"
        )
        original_bytes = target_path.read_bytes()

        asyncio.run(bridge.rewind_session(SESSION_ID, 2))

        # Target file untouched on disk.
        assert target_path.read_bytes() == original_bytes
        # And reading via the bridge gives the same shape.
        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        loaded_dump = json.loads(loaded.model_dump_json())
        original = json.loads(original_bytes)
        assert loaded_dump == original


class TestRewindSessionValidation:
    """Validation parity with preview_rewind. Anything that would
    error in preview MUST also error here."""

    def test_rewind_empty_session_raises(self, bridge: EngineBridge):
        with pytest.raises(FileNotFoundError, match="no checkpoints"):
            asyncio.run(bridge.rewind_session(SESSION_ID, 0))

    def test_rewind_unknown_target_raises(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=3)
        with pytest.raises(ValueError, match="has no checkpoint"):
            asyncio.run(bridge.rewind_session(SESSION_ID, 99))

    def test_rewind_to_latest_raises(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=3)
        with pytest.raises(ValueError, match="already the current state"):
            asyncio.run(bridge.rewind_session(SESSION_ID, 3))

    def test_rewind_negative_raises(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=3)
        with pytest.raises(ValueError, match="must be >= 0"):
            asyncio.run(bridge.rewind_session(SESSION_ID, -2))

    def test_failed_rewind_leaves_ckpts_intact(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=3)
        before = bridge.list_checkpoint_turns(SESSION_ID)
        with pytest.raises(ValueError):
            asyncio.run(bridge.rewind_session(SESSION_ID, 99))
        after = bridge.list_checkpoint_turns(SESSION_ID)
        assert after == before


class TestRewindLockSerialization:
    """The cull holds the per-session lock. Concurrent rewinds on the
    same session must serialize, and the second one (whose target
    might have just been deleted by the first) gets a clean
    "no longer available" error rather than half-finishing."""

    def test_concurrent_rewinds_serialize(self, bridge: EngineBridge):
        # Two rewinds racing on the same session. The first deletes
        # turns 4-5; the second tries to rewind to turn 4 (which the
        # first one just deleted). Expectation: first succeeds, second
        # raises a clear validation error and disk reflects the first
        # rewind only.
        _seed_session(bridge, last_turn=5)

        async def _race():
            return await asyncio.gather(
                bridge.rewind_session(SESSION_ID, 3),
                bridge.rewind_session(SESSION_ID, 4),
                return_exceptions=True,
            )

        results = asyncio.run(_race())
        successes = [r for r in results if isinstance(r, RewindResult)]
        errors = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 1
        assert len(errors) == 1
        assert successes[0].new_latest == 3
        # Either ordering is valid — validate end state instead of
        # which goroutine "won."
        assert bridge.list_checkpoint_turns(SESSION_ID) == [0, 1, 2, 3]


class TestRewindMetadata:
    """Metadata fields on RewindResult that drive the confirmation
    embed text. A regression here makes the user-facing message
    cryptic."""

    def test_actor_and_scene_recovered(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=4)

        result = asyncio.run(bridge.rewind_session(SESSION_ID, 1))

        assert result.actor_character_id == "aldric"
        assert result.scene_id == "hall"

    def test_no_bound_actor_yields_empty_strings(
        self, bridge: EngineBridge,
    ):
        # A session with no player_character_id (pristine, before any
        # /join) should still rewind cleanly; the actor/scene fields
        # just come back empty so the embed can omit those lines.
        for t in range(3):
            ckpt = _make_ckpt(turn_index=t)
            ckpt.session.player_character_id = ""
            ckpt.session.character_bindings = {}
            bridge.checkpoint_mgr.save(ckpt)

        result = asyncio.run(bridge.rewind_session(SESSION_ID, 1))

        assert result.actor_character_id == ""
        assert result.scene_id == ""
        assert bridge.list_checkpoint_turns(SESSION_ID) == [0, 1]
