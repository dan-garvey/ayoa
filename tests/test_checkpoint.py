"""Tests for the checkpoint manager — save/load round-trip, load-latest, corruption handling."""

import json

import pytest

from app.engine.checkpoint_manager import CheckpointManager
from app.schemas.checkpoint import CheckpointFile
from app.schemas.characters import CharacterRecord
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
        # v11 hard-break: pre-3.0 schema now rejected with explicit
        # "hard break" message.
        (session_dir / "ckpt_0000.json").write_text('{"schema_version": "1.0"}')

        with pytest.raises(ValueError, match="hard break|schema_version"):
            mgr.load("test-session", "ckpt_0000")

    def test_invalid_checkpoint_id_format(self, tmp_path):
        mgr = CheckpointManager(save_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Invalid checkpoint_id"):
            mgr.load("test-session", "bad_format")
