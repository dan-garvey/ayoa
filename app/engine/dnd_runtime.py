from __future__ import annotations

from typing import Any, MutableMapping

from app.engine.dnd_constants import DND_RUNTIME_KEY


def get_dnd_runtime(mechanics: object) -> dict[str, Any]:
    """Return the D&D runtime mechanics overlay, if present."""

    if not isinstance(mechanics, dict):
        return {}
    runtime = mechanics.get(DND_RUNTIME_KEY)
    return runtime if isinstance(runtime, dict) else {}


def has_dnd_runtime(mechanics: object) -> bool:
    if not isinstance(mechanics, dict):
        return False
    return isinstance(mechanics.get(DND_RUNTIME_KEY), dict)


def ensure_dnd_runtime(
    mechanics: MutableMapping[str, Any],
) -> dict[str, Any]:
    runtime = mechanics.get(DND_RUNTIME_KEY)
    if not isinstance(runtime, dict):
        runtime = {}
        mechanics[DND_RUNTIME_KEY] = runtime
    return runtime
