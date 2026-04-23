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
        """Load and validate a checkpoint file.

        v11 hard-break: checkpoints with schema_version < "3.0" fail with
        an explicit ValueError pointing the user at /story start. No
        automatic migration.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            raw = path.read_text()
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt checkpoint file {path}: {e}") from e

        # Schema version gate — hard break for pre-v11 checkpoints.
        from app.schemas.checkpoint import CURRENT_SCHEMA_VERSION
        version = str(data.get("schema_version", "")).strip()
        if version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint {path} has schema_version={version!r}, expected "
                f"{CURRENT_SCHEMA_VERSION!r}. The v11 turn pipeline is a "
                f"hard break — old sessions cannot be resumed. Run "
                f"/story start on a fresh session to continue."
            )

        # Commit-2 deprecation guard: pre-Commit-2 saves persisted
        # `incoming_directives` on every CharacterRecord. The field is
        # gone and Pydantic v2's default is to silently drop unknown
        # keys (CharacterRecord doesn't enforce extra="forbid"). That's
        # the desired behavior for forward-compat — but if a save
        # actually had queued messages, those messages disappear without
        # a trace, which can leave mid-arc threads dangling. Detect and
        # log so the operator at least sees the loss.
        for c in data.get("characters", []) or []:
            queued = c.get("incoming_directives") or []
            if queued:
                logger.warning(
                    "Checkpoint %s: dropping %d legacy incoming_directives "
                    "from character %s on load (Commit 2 removed the "
                    "inter-character message queue). The narrative may "
                    "lose threads that depended on these messages flushing.",
                    path.name, len(queued), c.get("character_id", "<unknown>"),
                )

        try:
            ckpt = CheckpointFile.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid checkpoint file {path}: {e}") from e

        # Commit-3 backfill: `surfaced_world_facts` is the bookkeeping
        # for the new world-facts-delta block. On a save written before
        # Commit 3, this list is empty and the next router call would
        # treat EVERY existing world fact as "new" and dump them all
        # into the user message — defeating the trim. Backfill the
        # list with whatever facts are already on the world so the
        # delta block stays empty until something actually changes.
        # Only fires when the field is empty AND there are facts; an
        # explicit empty queue on a fresh session is a separate state.
        if (
            ckpt.session.turn_index > 0
            and not ckpt.session.surfaced_world_facts
            and ckpt.world_state.facts
        ):
            ckpt.session.surfaced_world_facts = list(ckpt.world_state.facts)
            logger.info(
                "Checkpoint %s: backfilled surfaced_world_facts with %d "
                "pre-existing world facts on load (Commit 3 trim).",
                path.name, len(ckpt.world_state.facts),
            )

        return ckpt
