from __future__ import annotations

import re
from typing import Any

from app.schemas.characters import CharacterRecord
from app.schemas.dnd_cat_ii import PlannedRoll
from app.engine import dnd_inventory, dnd_runtime


_SKILL_ABILITIES = {
    "acrobatics": "dex",
    "animal handling": "wis",
    "arcana": "int",
    "athletics": "str",
    "deception": "cha",
    "history": "int",
    "insight": "wis",
    "intimidation": "cha",
    "investigation": "int",
    "medicine": "wis",
    "nature": "int",
    "perception": "wis",
    "performance": "cha",
    "persuasion": "cha",
    "religion": "int",
    "sleight of hand": "dex",
    "stealth": "dex",
    "survival": "wis",
}


def ability_modifier(score: int) -> int:
    return (score - 10) // 2


def roll_modifier(character: CharacterRecord | None, request: PlannedRoll) -> int:
    if character is None:
        return 0
    mechanics = character.mechanics or {}
    detailed = _detailed_roll_modifier(mechanics, request)
    if detailed is not None:
        return detailed

    ability = request.ability
    if request.kind == "skill_check" and request.skill:
        ability = _SKILL_ABILITIES.get(request.skill, ability)

    score = _ability_score(mechanics, ability)
    total = ability_modifier(score)

    if _is_proficient(mechanics, request):
        total += _proficiency_bonus(mechanics)
    return total


def mechanics_summary(
    character: CharacterRecord,
    *,
    include_inventory_resources: bool = True,
) -> dict[str, Any]:
    mechanics = character.mechanics or {}
    statblock = (mechanics.get("dnd5e_sheet") or {}).get("statblock") or {}
    defenses = statblock.get("defenses", {}) if isinstance(statblock, dict) else {}
    runtime = dnd_runtime.get_dnd_runtime(mechanics)
    summary: dict[str, Any] = {
        "ruleset_id": str(mechanics.get("ruleset_id", "")),
        "ability_scores": mechanics.get("ability_scores", {}),
        "proficiency_bonus": mechanics.get("proficiency_bonus", 0),
        "skill_proficiencies": mechanics.get("skill_proficiencies", []),
        "saving_throw_proficiencies": mechanics.get(
            "saving_throw_proficiencies", []
        ),
        "armor_class": mechanics.get("armor_class", 0),
        "hit_points": mechanics.get("hit_points", {}),
        "conditions": mechanics.get("conditions", []),
        "active_effects": runtime.get("active_effects", []),
        "defenses": defenses if isinstance(defenses, dict) else {},
    }
    if include_inventory_resources:
        inventory = dnd_inventory.inventory_view(character)
        summary["resources"] = mechanics.get("resources", {})
        summary["inventory"] = _inventory_summary(inventory)
        summary["raw"] = mechanics.get("raw", {})
    return summary


def _inventory_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    items = [
        item for item in (inventory.get("items") or [])
        if isinstance(item, dict)
    ]
    return {
        "currency": inventory.get("currency") or {},
        "items": [
            {
                "id": str(item.get("id") or item.get("item_id") or ""),
                "name": str(item.get("name") or "Item"),
                "kind": str(item.get("kind") or "gear"),
                "quantity": item.get("quantity") or 1,
                "equipped": bool(item.get("equipped")),
                "attuned": bool(item.get("attuned")),
                "identified": bool(item.get("identified", True)),
            }
            for item in items[:40]
        ],
    }


def _ability_score(mechanics: dict[str, Any], ability: str) -> int:
    scores = mechanics.get("ability_scores") or {}
    if not isinstance(scores, dict):
        return 10
    raw = scores.get(ability, 10)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 10


def _proficiency_bonus(mechanics: dict[str, Any]) -> int:
    try:
        return int(mechanics.get("proficiency_bonus", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _is_proficient(mechanics: dict[str, Any], request: PlannedRoll) -> bool:
    if request.kind == "skill_check" and request.skill:
        profs = mechanics.get("skill_proficiencies") or []
        return request.skill in {str(p).strip().lower() for p in profs}
    if request.kind == "saving_throw":
        profs = mechanics.get("saving_throw_proficiencies") or []
        return request.ability in {str(p).strip().lower() for p in profs}
    return False


def _detailed_roll_modifier(
    mechanics: dict[str, Any],
    request: PlannedRoll,
) -> int | None:
    sheet = mechanics.get("dnd5e_sheet") or {}
    statblock = sheet.get("statblock") or {}
    if not isinstance(statblock, dict):
        return None

    if request.kind == "skill_check" and request.skill:
        skills = statblock.get("skills") or {}
        skill = _lookup_roll_bonus(skills, request.skill)
        if skill is not None:
            return skill

    if request.kind == "saving_throw":
        saves = statblock.get("saves") or {}
        save = _lookup_roll_bonus(saves, request.ability)
        if save is not None:
            return save

    if request.kind == "attack_roll":
        action_key = request.action_id or request.skill
        action_bonus = _lookup_action_attack_bonus(
            statblock,
            action_key,
            reason=request.reason,
        )
        if action_bonus is not None:
            return action_bonus

    if request.kind == "ability_check":
        ability_scores = statblock.get("ability_scores") or {}
        score = ability_scores.get(request.ability) or {}
        if isinstance(score, dict) and "modifier" in score:
            try:
                return int(score.get("modifier") or 0)
            except (TypeError, ValueError):
                return None
    return None


def _lookup_roll_bonus(source: dict[str, Any], key: str) -> int | None:
    normalized = str(key).strip().lower().replace("_", " ")
    keys = {
        str(key).strip().lower(),
        normalized,
        normalized.replace("-", " "),
        normalized.replace(" ", "-"),
    }
    for candidate in keys:
        entry = source.get(candidate)
        if isinstance(entry, dict) and "value" in entry:
            try:
                return int(entry.get("value") or 0)
            except (TypeError, ValueError):
                return None
    return None


def _lookup_action_attack_bonus(
    statblock: dict[str, Any],
    key: str,
    *,
    reason: str = "",
) -> int | None:
    wanted = _normalize_action_text(key)
    reason_text = _normalize_action_text(reason)
    first_bonus: int | None = None
    bonus_count = 0
    for action in _statblock_actions(statblock):
        bonus = _action_attack_bonus(action)
        if bonus is None:
            continue
        bonus_count += 1
        if first_bonus is None:
            first_bonus = bonus
        names = _action_names(action)
        if wanted and wanted in names:
            return bonus
        if reason_text and any(
            _contains_action_name(reason_text, name) for name in names
        ):
            return bonus
    if not wanted and bonus_count == 1:
        return first_bonus
    return None


def _statblock_actions(statblock: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for action in statblock.get("actions") or []:
        if not isinstance(action, dict):
            continue
        actions.append(action)
    return actions


def _action_attack_bonus(action: dict[str, Any]) -> int | None:
    attack = action.get("attack") or {}
    if not isinstance(attack, dict):
        return None
    bonus = attack.get("bonus")
    if bonus is None:
        return None
    try:
        return int(bonus)
    except (TypeError, ValueError):
        return None


def _action_names(action: dict[str, Any]) -> set[str]:
    names = {
        _normalize_action_text(action.get("id") or ""),
        _normalize_action_text(action.get("name") or ""),
    }
    return {name for name in names if name}


def _normalize_action_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _contains_action_name(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return re.search(rf"(^|\s){re.escape(needle)}($|\s)", haystack) is not None
