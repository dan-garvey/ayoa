from __future__ import annotations

from typing import Any

from app.schemas.characters import CharacterRecord
from app.schemas.rules_arbitrator import PlannedRoll


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
