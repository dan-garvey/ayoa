"""User-tunable settings registry and helpers.

Settings live on `checkpoint.session.config.settings` (a `SessionSettings`
Pydantic model). This module defines the *surface* — which settings the
user-facing /settings command exposes, their descriptions, and how to
parse string input from a Discord slash command or CLI arg.

Adding a new setting:
  1. Add the field to `SessionSettings` in `app/schemas/state.py`.
  2. Add a `SettingDef` entry in `SETTINGS` below with a short
     description and a parser suitable for user input.
  3. Wire whatever runtime branch the flag toggles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.schemas.checkpoint import CheckpointFile


@dataclass(frozen=True)
class SettingDef:
    key: str
    default: Any
    description: str
    # Parse raw user input (string from Discord/CLI) into the setting's
    # runtime type. Must raise ValueError on invalid input.
    parse: Callable[[str], Any]
    # Render the current runtime value as a human-readable string.
    # Default handles bools/ints/strings well enough.
    render: Callable[[Any], str] = str


def _parse_int_nonneg(raw: str) -> int:
    """Parse a non-negative integer. Rejects floats and negatives."""
    s = raw.strip()
    try:
        n = int(s)
    except ValueError as e:
        raise ValueError(f"Cannot interpret {raw!r} as an integer.") from e
    if n < 0:
        raise ValueError(f"Value must be zero or positive, got {n}.")
    return n


def _parse_int_positive(raw: str) -> int:
    """Parse a positive (>0) integer."""
    n = _parse_int_nonneg(raw)
    if n == 0:
        raise ValueError("Value must be at least 1.")
    return n


def _parse_nonempty_str(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("Value cannot be empty.")
    return value


def _parse_player_roll_mode(raw: str) -> str:
    value = raw.strip().lower()
    valid = {"auto", "interactive"}
    if value not in valid:
        raise ValueError(
            "Player roll mode must be one of: " + ", ".join(sorted(valid))
        )
    return value


def _parse_presentation_mode(raw: str) -> str:
    value = raw.strip().lower().replace("-", "_")
    aliases = {
        "prose": "prose",
        "text": "prose",
        "visual_novel": "visual_novel",
        "vn": "visual_novel",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            "Presentation mode must be prose or visual_novel."
        ) from exc


SETTINGS: list[SettingDef] = [
    SettingDef(
        key="max_router_batches_without_player_input",
        default=12,
        description=(
            "Max autonomous router batches between player submissions. "
            "The durable causal frontier remains ready when the limit is reached. "
            "Must be >= 1."
        ),
        parse=_parse_int_positive,
    ),
    SettingDef(
        key="ruleset_id",
        default="narrative",
        description=(
            "Rules/content mode label for the session. Default narrative "
            "keeps Ayoa rules-neutral; dnd5e_basic enables D&D-facing "
            "helpers, combat resolution, and code-owned dice for features "
            "that explicitly opt into them."
        ),
        parse=_parse_nonempty_str,
    ),
    SettingDef(
        key="player_roll_mode",
        default="auto",
        description=(
            "How D&D player-character dice are handled. auto rolls them in "
            "code immediately; interactive pauses for Discord roll UI. "
            "NPC/agent rolls are always automatic."
        ),
        parse=_parse_player_roll_mode,
    ),
    SettingDef(
        key="presentation_mode",
        default="prose",
        description=(
            "Player-facing narrator presentation. prose is the original text "
            "surface; visual_novel renders ordered ADV-style cards over one "
            "safe reusable scene plate."
        ),
        parse=_parse_presentation_mode,
    ),
]

SETTINGS_BY_KEY: dict[str, SettingDef] = {s.key: s for s in SETTINGS}


class UnknownSettingError(KeyError):
    """Raised when a user-facing /settings command references a key not
    in the registry. Different from KeyError so callers can distinguish
    'bad input' from 'missing field on model'."""


def get_setting(ckpt: CheckpointFile, key: str) -> Any:
    if key not in SETTINGS_BY_KEY:
        raise UnknownSettingError(key)
    return getattr(ckpt.session.config.settings, key)


def set_setting(ckpt: CheckpointFile, key: str, raw_value: str) -> Any:
    """Parse and apply a setting on the checkpoint in place.

    Returns the parsed value. Caller is responsible for persisting the
    checkpoint (typically via CheckpointManager.save).
    """
    if key not in SETTINGS_BY_KEY:
        raise UnknownSettingError(key)
    spec = SETTINGS_BY_KEY[key]
    parsed = spec.parse(raw_value)
    setattr(ckpt.session.config.settings, key, parsed)
    return parsed


def list_settings_view(ckpt: CheckpointFile) -> list[dict[str, Any]]:
    """Return a list of {key, value, rendered_value, default, description}
    dicts — everything the /settings list surface needs."""
    out: list[dict[str, Any]] = []
    settings_obj = ckpt.session.config.settings
    for spec in SETTINGS:
        value = getattr(settings_obj, spec.key)
        out.append({
            "key": spec.key,
            "value": value,
            "rendered_value": spec.render(value),
            "default": spec.default,
            "rendered_default": spec.render(spec.default),
            "description": spec.description,
        })
    return out
