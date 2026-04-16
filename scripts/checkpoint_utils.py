"""Helpers for locating a checkpoint for manual scripts."""

from __future__ import annotations

from pathlib import Path

SAVES_DIR = Path("app/storage/saves")


def resolve_checkpoint_path(path_arg: str | None = None) -> str:
    """Resolve an explicit checkpoint path or discover a recent checkpoint."""
    if path_arg:
        return path_arg
    return str(find_latest_checkpoint())


def find_latest_checkpoint(save_root: Path = SAVES_DIR) -> Path:
    """Return the most recently modified checkpoint under the saves tree."""
    if not save_root.exists():
        raise FileNotFoundError(
            f"Saves directory not found: {save_root}"
        )

    checkpoints: list[Path] = []
    for session_dir in sorted(save_root.iterdir()):
        if not session_dir.is_dir():
            continue
        checkpoints.extend(sorted(session_dir.glob("ckpt_*.json")))

    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints found under app/storage/saves. "
            "Run the story importer first: "
            ".venv/bin/python scripts/import_story.py <story_file.txt>"
        )

    return max(checkpoints, key=lambda path: path.stat().st_mtime)
