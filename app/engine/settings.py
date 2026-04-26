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


def _parse_bool(raw: str) -> bool:
    """Accept common user spellings for true/false."""
    s = raw.strip().lower()
    if s in ("true", "t", "yes", "y", "on", "1", "enable", "enabled"):
        return True
    if s in ("false", "f", "no", "n", "off", "0", "disable", "disabled"):
        return False
    raise ValueError(
        f"Cannot interpret {raw!r} as a boolean. "
        f"Try true/false, on/off, yes/no, or 1/0."
    )


def _render_bool(value: bool) -> str:
    return "on" if value else "off"


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


SETTINGS: list[SettingDef] = [
    SettingDef(
        key="agents_can_create_scenes",
        default=False,
        description=(
            "Allow off-stage NPC ticks to invent new scenes via their "
            "scenes_created output. Default off; the router owns world "
            "topology in the baseline pipeline. Flip on to experiment "
            "with more emergent world-building from character-level intent."
        ),
        parse=_parse_bool,
        render=_render_bool,
    ),
    SettingDef(
        key="tick_concurrency",
        default=4,
        description=(
            "Max parallel off-stage tick calls. Ticks are independent so "
            "the only cost of raising this is API rate usage; lowering "
            "trades latency for safety on small quotas. Must be >= 1."
        ),
        parse=_parse_int_positive,
    ),
    SettingDef(
        key="ticks_on_scene_change",
        default=True,
        description=(
            "Whether the off-stage tick pass fires on scene change in "
            "addition to the cadence counter. On = extra tick every time "
            "the active scene shifts (more world liveness, more cost). "
            "Off = only the cadence counter triggers ticks."
        ),
        parse=_parse_bool,
        render=_render_bool,
    ),
    SettingDef(
        key="ticks_enabled",
        default=True,
        description=(
            "Master kill switch for the off-stage tick scheduler. Off = "
            "no background NPC ticks at all (no scene-change fires, no "
            "stagnation fires, no fan-out). Useful for token-budget runs "
            "and for isolating on-stage behavior in diagnostics. The "
            "tick counters freeze in disabled mode, so flipping back on "
            "resumes the trigger model from where it left off rather "
            "than firing a backlog."
        ),
        parse=_parse_bool,
        render=_render_bool,
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
