from __future__ import annotations

from typing import Any

from app.schemas.characters import CharacterRecord
from app.schemas.dnd_cat_ii import PlannedRoll


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


def mechanics_summary(character: CharacterRecord) -> dict[str, Any]:
    mechanics = character.mechanics or {}
    return {
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
        "resources": mechanics.get("resources", {}),
        "raw": mechanics.get("raw", {}),
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

    if request.kind == "attack_roll" and request.skill:
        action_bonus = _lookup_action_attack_bonus(statblock, request.skill)
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
) -> int | None:
    wanted = str(key).strip().lower()
    for action in statblock.get("actions") or []:
        if not isinstance(action, dict):
            continue
        names = {
            str(action.get("id") or "").lower(),
            str(action.get("name") or "").lower(),
        }
        if wanted not in names:
            continue
        attack = action.get("attack") or {}
        bonus = attack.get("bonus")
        if bonus is None:
            continue
        try:
            return int(bonus)
        except (TypeError, ValueError):
            return None
    return None
