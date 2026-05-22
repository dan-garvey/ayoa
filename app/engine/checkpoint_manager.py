from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from app.schemas.checkpoint import CURRENT_SCHEMA_VERSION, CheckpointFile
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages versioned JSON checkpoint save/load."""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.save_dir / session_id

    def _checkpoint_path(self, session_id: str, turn_index: int) -> Path:
        return self._session_dir(session_id) / f"ckpt_{turn_index:04d}.json"

    def save(self, state: CheckpointFile) -> str:
        """Save a checkpoint. Returns the checkpoint_id."""
        if str(state.schema_version).strip() != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Refusing to save checkpoint with schema_version="
                f"{state.schema_version!r}; expected "
                f"{CURRENT_SCHEMA_VERSION!r}."
            )
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
                f.write(
                    state.model_dump_json(
                        indent=2,
                        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
                    )
                )
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

    def list_turn_indices(self, session_id: str) -> list[int]:
        """Same as list_checkpoints but returns the integer turn_index of
        each checkpoint, sorted ascending. Used by the /rewind command to
        reason about which targets are reachable without re-parsing names.
        """
        out: list[int] = []
        for cid in self.list_checkpoints(session_id):
            try:
                out.append(int(cid.replace("ckpt_", "")))
            except ValueError:
                continue
        return out

    def delete_checkpoints_after(
        self, session_id: str, target_turn: int,
    ) -> list[int]:
        """Cull every ckpt_NNNN.json with NNNN > target_turn from the
        session directory. Returns the sorted list of turn indices that
        were removed. The target itself is preserved.

        This is the storage primitive behind `/rewind`. The delete is
        intentionally destructive (no .bak, no quarantine dir) — the
        rewind UX is gated by an explicit confirmation in the
        frontends, and recovery via cp from a backup is the operator's
        responsibility. Keeping the on-disk shape clean means
        `list_checkpoints()` immediately reflects the new latest with
        no skip-the-tombstones logic.

        Atomicity note: this is NOT atomic across files. A crash mid-loop
        leaves a partial cull, and re-running the same rewind fixes it
        (the loop is idempotent — already-deleted files are skipped).
        Callers should treat partial success as success for the targets
        that did get removed.
        """
        if target_turn < 0:
            raise ValueError(
                f"target_turn must be >= 0, got {target_turn}"
            )
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return []

        deleted: list[int] = []
        for f in sorted(session_dir.glob("ckpt_*.json")):
            try:
                idx = int(f.stem.replace("ckpt_", ""))
            except ValueError:
                continue
            if idx <= target_turn:
                continue
            try:
                f.unlink()
                deleted.append(idx)
            except FileNotFoundError:
                continue
        if deleted:
            logger.info(
                "Culled %d checkpoint(s) from session %s after turn %d: %s",
                len(deleted), session_id, target_turn, deleted,
            )
        return deleted

    def _load_file(self, path: Path) -> CheckpointFile:
        """Load and validate a checkpoint file.

        v11 hard-break: checkpoints with stale schema_version fail with
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

        return ckpt
