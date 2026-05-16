from __future__ import annotations

from typing import Any

from app.schemas.checkpoint import CheckpointFile


def obj_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def obj_set(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def session_active_combat(session: Any) -> Any | None:
    combat = obj_get(session, "active_combat")
    if combat is None:
        return None
    status = obj_get(combat, "status")
    if status is not None:
        return combat if status == "active" else None
    if combatants(combat) and obj_get(combat, "ended_at_turn_index") is None:
        return combat
    return None


def checkpoint_active_combat(ckpt: CheckpointFile) -> Any | None:
    return session_active_combat(ckpt.session)


def combatants(combat: Any) -> list[Any]:
    return list(obj_get(combat, "combatants", []) or [])


def combat_turn_index(combat: Any, participants: list[Any] | None = None) -> int:
    ordered = participants if participants is not None else combatants(combat)
    if not ordered:
        return 0
    try:
        return int(obj_get(combat, "turn_index", 0) or 0) % len(ordered)
    except (TypeError, ValueError):
        return 0


def combatant_character_id(combatant: Any) -> str:
    return (
        str(obj_get(combatant, "character_id", "") or "")
        or str(obj_get(combatant, "combatant_id", "") or "")
    )


def combatant_ids(combatant: Any) -> set[str]:
    return {
        value for value in (
            str(obj_get(combatant, "combatant_id", "") or ""),
            str(obj_get(combatant, "character_id", "") or ""),
        ) if value
    }


def combatant_name(combatant: Any) -> str:
    return str(obj_get(combatant, "name", "") or "") or combatant_character_id(
        combatant
    )


def combatant_defeat_state(combatant: Any) -> str:
    state = str(obj_get(combatant, "defeat_state", "") or "")
    return state or "active"


def combatant_defeated(combatant: Any) -> bool:
    return bool(
        combatant_defeat_state(combatant) in {"down", "stable", "dead", "defeated"}
        or obj_get(combatant, "removed", False)
    )


def current_combatant(combat: Any, *, skip_defeated: bool = False) -> Any | None:
    ordered = combatants(combat)
    if not ordered:
        return None
    start = combat_turn_index(combat, ordered)
    if not skip_defeated:
        return ordered[start]
    for offset in range(len(ordered)):
        candidate = ordered[(start + offset) % len(ordered)]
        if not combatant_defeated(candidate):
            return candidate
    return None


def combatant_for_character(combat: Any, character_id: str) -> Any | None:
    if not character_id:
        return None
    for combatant in combatants(combat):
        if combatant_character_id(combatant) == character_id:
            return combatant
    return None


def target_armor_class(
    combat: Any,
    target_id: str,
    *,
    default: int = 10,
) -> int:
    if not target_id:
        return default
    for combatant in combatants(combat):
        if target_id in combatant_ids(combatant):
            try:
                return int(obj_get(combatant, "armor_class", default) or default)
            except (TypeError, ValueError):
                return default
    return default
