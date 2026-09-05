"""Shared character identity helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable


def pick_unused_character_id(name: str, taken_ids: Iterable[str]) -> str:
    """Return a snake-case id derived from ``name`` and unique in ``taken_ids``."""

    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "character"
    taken = set(taken_ids)
    if base not in taken:
        return base
    suffix = 2
    while f"{base}_{suffix}" in taken:
        suffix += 1
    return f"{base}_{suffix}"
