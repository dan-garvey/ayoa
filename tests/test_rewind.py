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

from app.bot.engine_bridge import EngineBridge
from app.engine.frontend_views import RewindResult
from app.schemas.characters import (
    CharacterRecord,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content import (
    ContentCombatMapOverlayState,
    ContentOverlayState,
    ContentPackState,
    ContentPovRevealState,
)
from app.schemas.narrator import TranscriptEntry
from app.schemas.state import (
    SessionState,
    WorldState,
)
from tests.support.factories import router_output


SESSION_ID = "rewind_test"


def _make_ckpt(
    *,
    turn_index: int,
    canonical_event_count: int = 0,
    location: str = "hall",
    actor_id: str = "aldric",
    extra_chars: list[CharacterRecord] | None = None,
    content_state: dict[str, ContentPackState] | None = None,
) -> CheckpointFile:
    """Build a single checkpoint at a given turn_index. Each canonical
    event is unique enough that round-trip checks can verify the right
    snapshot was preserved."""
    chars = [
        CharacterRecord(
            character_id=actor_id,
            name="Aldric Verantus",
            is_playable=True,
            location=location,
            public_sheet=PublicSheet(role="envoy"),
        ),
    ]
    if extra_chars:
        chars.extend(extra_chars)

    events = []
    for i in range(canonical_event_count):
        events.append(
            router_output(
                event_id=f"evt_t{turn_index}_{i}",
                observer_ids=[actor_id],
                facts=[],
                event_kind="cascade_exhausted",
            )
        )

    facts = [f"fact at turn {turn_index}"]
    return CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            turn_index=turn_index,
            player_character_id=actor_id,
            character_bindings={actor_id: "12345"},
            content_state=content_state or {},
        ),
        world_state=WorldState(
            facts=facts,
        ),
        characters=chars,
        canonical_events=events,
    )


def _pov_content_state(
    *,
    handouts: list[str],
    revealed_areas: list[str],
    fogged_areas: list[str],
    assets: list[str] | None = None,
    reveal_refs: list[str] | None = None,
) -> dict[str, ContentPackState]:
    return {
        "pack": ContentPackState(
            pack_id="pack",
            overlay=ContentOverlayState(
                pov_reveals={
                    "alice": ContentPovRevealState(
                        viewer_id="alice",
                        revealed_handout_ref_ids=handouts,
                        revealed_asset_ids=assets or [],
                        reveal_ref_ids=reveal_refs or [],
                        map_overlays={
                            "map": ContentCombatMapOverlayState(
                                map_id="map/crypt",
                                content_hash="sha256:map",
                                fog_of_war=True,
                                fully_revealed=False,
                                revealed_area_ref_ids=revealed_areas,
                                fogged_area_ref_ids=fogged_areas,
                            )
                        },
                    )
                }
            ),
        )
    }


@pytest.fixture
def bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )


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
        assert result.location == "hall"

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


# ---- turn_history -----------------------------------------------------------


class TestTurnHistory:
    def test_history_entries_use_checkpoint_turns(self, bridge: EngineBridge):
        first = TranscriptEntry(user="first", assistant="First render.")
        second = TranscriptEntry(user="", assistant="Automated render.")

        ckpt0 = _make_ckpt(turn_index=0)
        bridge.checkpoint_mgr.save(ckpt0)

        ckpt1 = _make_ckpt(turn_index=1)
        ckpt1.transcript = [first]
        bridge.checkpoint_mgr.save(ckpt1)

        ckpt2 = _make_ckpt(turn_index=2)
        ckpt2.transcript = [first]
        bridge.checkpoint_mgr.save(ckpt2)

        ckpt3 = _make_ckpt(turn_index=3)
        ckpt3.transcript = [first, second]
        bridge.checkpoint_mgr.save(ckpt3)

        history = bridge.turn_history(SESSION_ID)

        assert [(item.turn_index, item.entry) for item in history] == [
            (1, first),
            (3, second),
        ]


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

    def test_rewind_restores_per_pov_reveal_and_fog_state(
        self, bridge: EngineBridge,
    ):
        bridge.checkpoint_mgr.save(_make_ckpt(turn_index=0))
        bridge.checkpoint_mgr.save(_make_ckpt(
            turn_index=1,
            content_state=_pov_content_state(
                handouts=["handout/entry-letter"],
                assets=["asset.handout.entry-letter"],
                reveal_refs=["reveal.entry-letter"],
                revealed_areas=["area/entry"],
                fogged_areas=["area/altar", "area/balcony"],
            ),
        ))
        bridge.checkpoint_mgr.save(_make_ckpt(
            turn_index=2,
            content_state=_pov_content_state(
                handouts=["handout/entry-letter", "handout/altar-note"],
                assets=["asset.handout.entry-letter", "asset.map.crypt"],
                reveal_refs=["reveal.entry-letter", "reveal.altar-map"],
                revealed_areas=["area/entry", "area/altar"],
                fogged_areas=["area/balcony"],
            ),
        ))
        target = bridge.checkpoint_mgr.load(SESSION_ID, "ckpt_0001")
        target_state = target.session.content_state["pack"].overlay.pov_reveals[
            "alice"
        ].model_dump(mode="json")

        asyncio.run(bridge.rewind_session(SESSION_ID, 1))

        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        restored = loaded.session.content_state["pack"].overlay.pov_reveals[
            "alice"
        ]
        assert restored.model_dump(mode="json") == target_state
        assert restored.revealed_handout_ref_ids == ["handout/entry-letter"]
        assert restored.revealed_asset_ids == ["asset.handout.entry-letter"]
        assert restored.reveal_ref_ids == ["reveal.entry-letter"]
        map_state = restored.map_overlays["map/crypt::sha256:map"]
        assert map_state.revealed_area_ref_ids == ["area/entry"]
        assert map_state.fogged_area_ref_ids == ["area/altar", "area/balcony"]


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

    def test_actor_and_location_recovered(self, bridge: EngineBridge):
        _seed_session(bridge, last_turn=4)

        result = asyncio.run(bridge.rewind_session(SESSION_ID, 1))

        assert result.actor_character_id == "aldric"
        assert result.location == "hall"

    def test_no_bound_actor_yields_empty_strings(
        self, bridge: EngineBridge,
    ):
        # A session with no player_character_id (pristine, before any
        # /join) should still rewind cleanly; the actor/location fields
        # just come back empty so the embed can omit those lines.
        for t in range(3):
            ckpt = _make_ckpt(turn_index=t)
            ckpt.session.player_character_id = ""
            ckpt.session.character_bindings = {}
            bridge.checkpoint_mgr.save(ckpt)

        result = asyncio.run(bridge.rewind_session(SESSION_ID, 1))

        assert result.actor_character_id == ""
        assert result.location == ""
        assert bridge.list_checkpoint_turns(SESSION_ID) == [0, 1]
