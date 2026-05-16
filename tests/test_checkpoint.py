"""Tests for the checkpoint manager — save/load round-trip, load-latest, corruption handling."""


import pytest

from app.engine.checkpoint_manager import CheckpointManager
from app.schemas.checkpoint import CheckpointFile
from app.schemas.characters import CharacterRecord
from app.schemas.dnd_inventory import DndLootOffer
from app.schemas.event_router import DndEventRouterOutput
from app.schemas.narrator import TranscriptEntry
from app.schemas.state import SessionState, WorldState


def _make_checkpoint(session_id: str = "test-session", turn_index: int = 0) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(session_id=session_id, turn_index=turn_index),
        world_state=WorldState(
            facts=["The courtyard is wet."],
        ),
        characters=[
            CharacterRecord(character_id="guard_17", name="Captain Vero"),
        ],
        transcript=[
            TranscriptEntry(user="I look around.", assistant="You see a courtyard."),
        ] if turn_index > 0 else [],
    )


class TestCheckpointSaveLoad:
    def test_save_and_load(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        ckpt = _make_checkpoint(turn_index=0)

        checkpoint_id = mgr.save(ckpt)
        assert checkpoint_id == "ckpt_0000"

        loaded = mgr.load("test-session", checkpoint_id)
        assert loaded.session.session_id == "test-session"
        assert loaded.session.turn_index == 0
        assert len(loaded.characters) == 1
        assert loaded.characters[0].name == "Captain Vero"

    def test_round_trip_preserves_data(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        ckpt = _make_checkpoint(turn_index=5)

        mgr.save(ckpt)
        loaded = mgr.load("test-session", "ckpt_0005")

        assert loaded.session.turn_index == 5
        assert loaded.world_state.facts == ["The courtyard is wet."]
        assert len(loaded.transcript) == 1
        assert loaded.transcript[0].user == "I look around."

    def test_dnd_router_event_and_loot_offer_round_trip(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        ckpt = _make_checkpoint(turn_index=1)
        ckpt.session.character_bindings = {"guard_17": "42"}
        ckpt.canonical_events.append(DndEventRouterOutput(
            event_id="evt_loot",
            decision_rationale="test",
            canonical_event={
                "world_adjudication": {"feasible": True},
                "observable_facts": [
                    {
                        "text": "Captain Vero opens the chest.",
                        "audience": "all_observers",
                        "visible_to": [],
                    }
                ],
            },
            observers=[
                {
                    "character_id": "guard_17",
                    "observation_level": "d",
                    "response_priority": 1,
                }
            ],
            spawn=[],
            dormant=[],
            cull=[],
            interaction_mode="cat_i",
            combatant_ids=[],
            loot_offer={
                "present": True,
                "source_kind": "container",
                "source_label": "iron chest",
                "visibility": "table",
                "eligible_character_ids": ["guard_17"],
                "items": [],
                "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 5, "pp": 0},
                "notes": "",
            },
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="state_change",
        ))
        ckpt.session.dnd_inventory_offers.append(DndLootOffer(
            offer_id="loot_evt_loot",
            source_event_id="evt_loot",
            source_kind="container",
            source_label="iron chest",
            eligible_character_ids=["guard_17"],
            currency={"gp": 5},
        ))

        mgr.save(ckpt)
        loaded = mgr.load("test-session", "ckpt_0001")

        assert len(loaded.canonical_events) == 1
        event = loaded.canonical_events[0]
        assert isinstance(event, DndEventRouterOutput)
        assert event.event_id == "evt_loot"
        assert event.interaction_mode == "cat_i"
        assert event.loot_offer.present is True
        assert event.loot_offer.currency.gp == 5
        assert event.battle_map_seed.present is False
        assert len(loaded.session.dnd_inventory_offers) == 1
        offer = loaded.session.dnd_inventory_offers[0]
        assert offer.offer_id == "loot_evt_loot"
        assert offer.currency.gp == 5

    def test_save_creates_session_directory(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        ckpt = _make_checkpoint(session_id="new-session")

        mgr.save(ckpt)
        assert (tmp_path / "new-session").is_dir()
        assert (tmp_path / "new-session" / "ckpt_0000.json").exists()


class TestCheckpointLoadLatest:
    def test_load_latest(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))

        for i in range(4):
            mgr.save(_make_checkpoint(turn_index=i))

        latest = mgr.load_latest("test-session")
        assert latest.session.turn_index == 3

    def test_load_with_no_checkpoint_id_uses_latest(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        mgr.save(_make_checkpoint(turn_index=0))
        mgr.save(_make_checkpoint(turn_index=1))

        loaded = mgr.load("test-session", None)
        assert loaded.session.turn_index == 1


class TestCheckpointListCheckpoints:
    def test_list_checkpoints(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        mgr.save(_make_checkpoint(turn_index=0))
        mgr.save(_make_checkpoint(turn_index=1))
        mgr.save(_make_checkpoint(turn_index=2))

        ids = mgr.list_checkpoints("test-session")
        assert ids == ["ckpt_0000", "ckpt_0001", "ckpt_0002"]

    def test_list_empty_session(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        assert mgr.list_checkpoints("nonexistent") == []


class TestCheckpointErrors:
    def test_load_missing_session(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            mgr.load("nonexistent", "ckpt_0000")

    def test_load_missing_checkpoint(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        mgr.save(_make_checkpoint(turn_index=0))
        with pytest.raises(FileNotFoundError):
            mgr.load("test-session", "ckpt_9999")

    def test_load_latest_empty_session(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="No checkpoints found"):
            mgr.load_latest("nonexistent")

    def test_load_corrupt_json(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        session_dir = tmp_path / "test-session"
        session_dir.mkdir()
        (session_dir / "ckpt_0000.json").write_text("{{not json}}")

        with pytest.raises(ValueError, match="Corrupt checkpoint"):
            mgr.load("test-session", "ckpt_0000")

    def test_load_invalid_schema(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        session_dir = tmp_path / "test-session"
        session_dir.mkdir()
        # v11 hard-break: stale schema now rejected with explicit
        # "hard break" message.
        (session_dir / "ckpt_0000.json").write_text('{"schema_version": "1.0"}')

        with pytest.raises(ValueError, match="hard break|schema_version"):
            mgr.load("test-session", "ckpt_0000")

    def test_save_rejects_stale_schema(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        ckpt = _make_checkpoint(turn_index=0)
        ckpt.schema_version = "3.0"

        with pytest.raises(ValueError, match="Refusing to save|schema_version"):
            mgr.save(ckpt)

    def test_invalid_checkpoint_id_format(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Invalid checkpoint_id"):
            mgr.load("test-session", "bad_format")


class TestListTurnIndices:
    """list_turn_indices is the integer-typed sibling of list_checkpoints,
    used by the /rewind validator. Tests live alongside the storage
    primitive so any drift in checkpoint naming gets caught here before
    the bridge layer."""

    def test_list_turn_indices_empty(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        assert mgr.list_turn_indices("nonexistent") == []

    def test_list_turn_indices_sorted(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        for i in [3, 1, 5, 0, 2]:
            mgr.save(_make_checkpoint(turn_index=i))
        assert mgr.list_turn_indices("test-session") == [0, 1, 2, 3, 5]

    def test_list_turn_indices_skips_garbage(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        mgr.save(_make_checkpoint(turn_index=0))
        mgr.save(_make_checkpoint(turn_index=1))
        # A stray non-checkpoint file should not poison the listing.
        (tmp_path / "test-session" / "ckpt_garbage.json").write_text("{}")
        assert mgr.list_turn_indices("test-session") == [0, 1]


class TestDeleteCheckpointsAfter:
    """The /rewind storage primitive. Validates the cull semantics
    (target preserved, only ckpt_>target removed) and the edge cases
    (target == latest is a no-op, target negative raises, missing dir
    returns empty)."""

    def test_culls_only_after_target(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        for i in range(6):
            mgr.save(_make_checkpoint(turn_index=i))

        deleted = mgr.delete_checkpoints_after("test-session", target_turn=2)

        assert deleted == [3, 4, 5]
        assert mgr.list_turn_indices("test-session") == [0, 1, 2]

    def test_target_itself_is_preserved(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        for i in range(4):
            mgr.save(_make_checkpoint(turn_index=i))

        mgr.delete_checkpoints_after("test-session", target_turn=2)

        loaded = mgr.load("test-session", "ckpt_0002")
        assert loaded.session.turn_index == 2

    def test_target_at_latest_is_noop(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        for i in range(3):
            mgr.save(_make_checkpoint(turn_index=i))

        deleted = mgr.delete_checkpoints_after("test-session", target_turn=2)

        assert deleted == []
        assert mgr.list_turn_indices("test-session") == [0, 1, 2]

    def test_target_above_latest_is_noop(self, tmp_path):
        # "Rewind to turn 99" when only turn 2 exists shouldn't delete
        # anything — defensive against a frontend that fails to validate.
        mgr = CheckpointManager(save_dir=str(tmp_path))
        for i in range(3):
            mgr.save(_make_checkpoint(turn_index=i))

        deleted = mgr.delete_checkpoints_after("test-session", target_turn=99)

        assert deleted == []
        assert mgr.list_turn_indices("test-session") == [0, 1, 2]

    def test_target_zero_keeps_only_origin(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        for i in range(5):
            mgr.save(_make_checkpoint(turn_index=i))

        deleted = mgr.delete_checkpoints_after("test-session", target_turn=0)

        assert deleted == [1, 2, 3, 4]
        assert mgr.list_turn_indices("test-session") == [0]

    def test_negative_target_raises(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        mgr.save(_make_checkpoint(turn_index=0))
        with pytest.raises(ValueError, match="must be >= 0"):
            mgr.delete_checkpoints_after("test-session", target_turn=-1)

    def test_missing_session_returns_empty(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        # No session dir created at all — should return empty list,
        # not raise. The rewind validator above this layer is what
        # enforces "session must exist."
        assert mgr.delete_checkpoints_after("ghost", target_turn=0) == []

    def test_idempotent(self, tmp_path):
        # Running cull twice with the same target leaves the second
        # call returning an empty deletion list — important if a
        # rewind is retried after a partial crash.
        mgr = CheckpointManager(save_dir=str(tmp_path))
        for i in range(5):
            mgr.save(_make_checkpoint(turn_index=i))

        first = mgr.delete_checkpoints_after("test-session", target_turn=2)
        second = mgr.delete_checkpoints_after("test-session", target_turn=2)

        assert first == [3, 4]
        assert second == []
        assert mgr.list_turn_indices("test-session") == [0, 1, 2]
