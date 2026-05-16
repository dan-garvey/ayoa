from __future__ import annotations

import re
from typing import Any

from app.engine.dnd_combat_access import target_armor_class
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_cat_ii import PlannedRoll
from app.schemas.responses import DiceRollDisplay
from app.schemas.state import CatIIRollRecord, CatIIRollTransaction


RollKey = tuple[str, str]


def completed_automatic_roll_keys(ckpt: CheckpointFile) -> set[RollKey]:
    return {
        (txn.transaction_id, record.roll_id)
        for txn, record in _iter_completed_rolls(ckpt)
        if record.completed_by_user_id == "engine"
    }


def dice_roll_displays_since(
    ckpt: CheckpointFile,
    before: set[RollKey],
) -> list[DiceRollDisplay]:
    displays: list[DiceRollDisplay] = []
    for txn, record in _iter_completed_rolls(ckpt):
        key = (txn.transaction_id, record.roll_id)
        if key in before or record.completed_by_user_id != "engine":
            continue
        displays.append(dice_roll_display_for_record(ckpt, txn, record))
    return displays


def dice_roll_display_for_record(
    ckpt: CheckpointFile,
    transaction: CatIIRollTransaction,
    record: CatIIRollRecord,
) -> DiceRollDisplay:
    request = _planned_roll(record.request)
    result = record.result or {}
    dice = _d20_values(result)
    kept = _kept_d20_values(result) or dice
    target_id = request.target_id if request is not None else _request_text(
        record.request, "target_id"
    )
    dc = _roll_dc(ckpt, transaction, request, target_id)
    total = _int(result.get("total"))
    crit = str(result.get("crit") or "none")
    kind = str(getattr(request, "kind", "") or "")
    damage_total, damage_type, damage_detail = _damage_summary(
        transaction, record.roll_id,
    )
    return DiceRollDisplay(
        transaction_id=transaction.transaction_id,
        event_id=transaction.event_id,
        source=transaction.source,
        roll_id=record.roll_id,
        actor_id=record.actor_id,
        actor_name=_character_name(ckpt, record.actor_id),
        target_id=target_id,
        target_name=_target_name(ckpt, transaction, target_id),
        label=record.label or _fallback_label(request),
        reason=record.reason,
        kind=kind,
        ability=str(getattr(request, "ability", "") or ""),
        skill=str(getattr(request, "skill", "") or ""),
        expression=str(result.get("expression") or ""),
        detail=str(result.get("detail") or ""),
        die_faces=20,
        die_values=dice,
        kept_die_values=kept,
        modifier=int(record.modifier or 0),
        total=total,
        dc=dc,
        outcome=_roll_outcome(kind=kind, total=total, dc=dc, crit=crit),
        crit=crit,
        damage_total=damage_total,
        damage_type=damage_type,
        damage_detail=damage_detail,
        automatic=record.completed_by_user_id == "engine",
    )


def _iter_completed_rolls(
    ckpt: CheckpointFile,
) -> list[tuple[CatIIRollTransaction, CatIIRollRecord]]:
    completed: list[tuple[CatIIRollTransaction, CatIIRollRecord]] = []
    for txn in ckpt.session.cat_ii_roll_transactions or []:
        for record in txn.rolls or []:
            if record.status == "completed" and record.roll_id:
                completed.append((txn, record))
    return completed


def _planned_roll(value: dict[str, Any]) -> PlannedRoll | None:
    try:
        return PlannedRoll.model_validate(value or {})
    except Exception:
        return None


def _request_text(value: dict[str, Any], key: str) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get(key) or "").strip()


def _d20_values(result: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for die in result.get("dice") or []:
        if not isinstance(die, dict) or str(die.get("size")) != "20":
            continue
        values.extend(_int(v) for v in die.get("values") or [])
    return values or _d20_values_from_detail(str(result.get("detail") or ""))


def _kept_d20_values(result: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for die in result.get("dice") or []:
        if (
            not isinstance(die, dict)
            or str(die.get("size")) != "20"
            or not bool(die.get("kept", True))
        ):
            continue
        values.extend(_int(v) for v in die.get("values") or [])
    return values


def _d20_values_from_detail(detail: str) -> list[int]:
    match = re.search(r"1d20\s*\((?:\*\*)?(\d+)", detail)
    if match:
        return [_int(match.group(1))]
    return []


def _roll_dc(
    ckpt: CheckpointFile,
    transaction: CatIIRollTransaction,
    request: PlannedRoll | None,
    target_id: str,
) -> int:
    if request is not None and request.dc:
        return int(request.dc)
    if request is not None and request.kind == "attack_roll" and target_id:
        return _target_ac(ckpt, transaction, target_id)
    return 0


def _target_ac(
    ckpt: CheckpointFile,
    transaction: CatIIRollTransaction,
    target_id: str,
) -> int:
    combat = getattr(ckpt.session, "active_combat", None)
    ac = target_armor_class(combat, target_id, default=0)
    if ac:
        return ac
    for item in _context_combatants(transaction):
        ids = {
            str(item.get("combatant_id") or ""),
            str(item.get("character_id") or ""),
        }
        if target_id in ids:
            return _int(item.get("armor_class"), default=10)
    return 0


def _roll_outcome(*, kind: str, total: int, dc: int, crit: str) -> str:
    if kind == "attack_roll":
        if crit == "crit":
            return "critical hit"
        if crit == "fail":
            return "miss"
        if dc:
            return "hit" if total >= dc else "miss"
        return ""
    if crit == "crit":
        return "critical success"
    if crit == "fail":
        return "critical failure"
    if dc:
        return "success" if total >= dc else "failure"
    return ""


def _damage_summary(
    transaction: CatIIRollTransaction,
    roll_id: str,
) -> tuple[int, str, str]:
    matching = [
        damage for damage in transaction.damage_records or []
        if damage.roll_id == roll_id
    ]
    if not matching:
        return 0, "", ""
    total = sum(int(damage.amount or 0) for damage in matching)
    types = sorted({
        damage.damage_type for damage in matching if damage.damage_type
    })
    details = [
        damage.detail for damage in matching if damage.detail
    ]
    return total, ", ".join(types), "; ".join(details)


def _character_name(ckpt: CheckpointFile, character_id: str) -> str:
    for character in ckpt.characters or []:
        if character.character_id == character_id:
            return character.name or character_id
    combat = getattr(ckpt.session, "active_combat", None)
    for combatant in list(getattr(combat, "combatants", []) or []):
        ids = {
            str(getattr(combatant, "combatant_id", "") or ""),
            str(getattr(combatant, "character_id", "") or ""),
        }
        if character_id in ids:
            return str(getattr(combatant, "name", "") or character_id)
    return character_id


def _target_name(
    ckpt: CheckpointFile,
    transaction: CatIIRollTransaction,
    target_id: str,
) -> str:
    if not target_id:
        return ""
    name = _character_name(ckpt, target_id)
    if name != target_id:
        return name
    for item in _context_combatants(transaction):
        ids = {
            str(item.get("combatant_id") or ""),
            str(item.get("character_id") or ""),
        }
        if target_id in ids:
            return str(item.get("name") or target_id)
    return target_id


def _context_combatants(
    transaction: CatIIRollTransaction,
) -> list[dict[str, Any]]:
    combatants = transaction.context.get("combatants") or []
    return [item for item in combatants if isinstance(item, dict)]


def _fallback_label(request: PlannedRoll | None) -> str:
    if request is None:
        return "Roll"
    if request.kind == "attack_roll" and request.action_id:
        return f"Attack ({request.action_id.replace('_', ' ').title()})"
    if request.kind == "skill_check" and request.skill:
        return request.skill.title()
    if request.kind == "saving_throw":
        return f"{request.ability.upper()} Save"
    if request.kind == "attack_roll":
        return "Attack"
    return f"{request.ability.upper()} Check"


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
