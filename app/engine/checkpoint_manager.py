from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from app.schemas.checkpoint import CheckpointFile

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages versioned JSON checkpoint save/load."""

    def __init__(self, save_dir: str = "app/storage/saves"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.save_dir / session_id

    def _checkpoint_path(self, session_id: str, turn_index: int) -> Path:
        return self._session_dir(session_id) / f"ckpt_{turn_index:04d}.json"

    def save(self, state: CheckpointFile) -> str:
        """Save a checkpoint. Returns the checkpoint_id."""
        session_id = state.session.session_id
        turn_index = state.session.turn_index
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self._checkpoint_path(session_id, turn_index)
        checkpoint_id = f"ckpt_{turn_index:04d}"

        # Atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(dir=session_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(state.model_dump_json(indent=2))
            os.replace(tmp_path, checkpoint_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        logger.info(f"Saved checkpoint {checkpoint_id} for session {session_id}")
        return checkpoint_id

    def load(self, session_id: str, checkpoint_id: str | None = None) -> CheckpointFile:
        """Load a specific checkpoint, or the latest if checkpoint_id is None.

        Raises FileNotFoundError if session or checkpoint doesn't exist.
        Raises ValueError if the checkpoint file is corrupt.
        """
        if checkpoint_id is None:
            return self.load_latest(session_id)

        # Extract turn index from checkpoint_id like "ckpt_0042"
        try:
            turn_index = int(checkpoint_id.replace("ckpt_", ""))
        except ValueError:
            raise FileNotFoundError(f"Invalid checkpoint_id format: {checkpoint_id}")

        path = self._checkpoint_path(session_id, turn_index)
        return self._load_file(path)

    def load_latest(self, session_id: str) -> CheckpointFile:
        """Load the most recent checkpoint for a session.

        Raises FileNotFoundError if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints(session_id)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for session {session_id}")

        latest_id = checkpoints[-1]
        return self.load(session_id, latest_id)

    def list_checkpoints(self, session_id: str) -> list[str]:
        """List checkpoint IDs for a session, sorted by turn index."""
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return []

        checkpoints = []
        for f in sorted(session_dir.glob("ckpt_*.json")):
            checkpoints.append(f.stem)
        return checkpoints

    def _load_file(self, path: Path) -> CheckpointFile:
        """Load and validate a checkpoint file."""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            raw = path.read_text()
            data = json.loads(raw)
            _migrate_legacy_attitudes(data)
            return CheckpointFile.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt checkpoint file {path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Invalid checkpoint file {path}: {e}") from e


def _migrate_legacy_attitudes(data: dict) -> None:
    """Rewrite character.private_state.attitudes["user"] → the bound player
    character_id, for checkpoints produced before attitudes were keyed by
    character_id throughout.

    No-op when no legacy keys are present or no player binding is set.
    Safe to run on every load; O(n) over the roster with zero allocations
    in the clean case.
    """
    session = data.get("session") or {}
    player_id = session.get("player_character_id")
    if not player_id:
        return
    migrated = 0
    for char in data.get("characters", []) or []:
        ps = char.get("private_state") or {}
        attitudes = ps.get("attitudes") or {}
        if "user" not in attitudes:
            continue
        # Prefer existing char_id-keyed value if both somehow exist;
        # otherwise lift "user" into the char_id slot.
        existing = attitudes.get(player_id)
        legacy = attitudes.pop("user")
        if existing is None:
            attitudes[player_id] = legacy
        migrated += 1
    if migrated:
        logger.info(
            "Migrated legacy 'user' attitude key on %d character(s) to %r",
            migrated, player_id,
        )
