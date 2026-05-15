from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from app.schemas.characters import CharacterRecord


DND_RUNTIME_KEY = "dnd5e_runtime"

# D&D 5e cumulative experience thresholds for character levels 1-20.
DND5E_XP_THRESHOLDS: dict[int, int] = {
    1: 0,
    2: 300,
    3: 900,
    4: 2700,
    5: 6500,
    6: 14000,
    7: 23000,
    8: 34000,
    9: 48000,
    10: 64000,
    11: 85000,
    12: 100000,
    13: 120000,
    14: 140000,
    15: 165000,
    16: 195000,
    17: 225000,
    18: 265000,
    19: 305000,
    20: 355000,
}

DND5E_CHALLENGE_XP: dict[str, int] = {
    "0": 10,
    "1/8": 25,
    "1/4": 50,
    "1/2": 100,
    "1": 200,
    "2": 450,
    "3": 700,
    "4": 1100,
    "5": 1800,
    "6": 2300,
    "7": 2900,
    "8": 3900,
    "9": 5000,
    "10": 5900,
    "11": 7200,
    "12": 8400,
    "13": 10000,
    "14": 11500,
    "15": 13000,
    "16": 15000,
    "17": 18000,
    "18": 20000,
    "19": 22000,
    "20": 25000,
    "21": 33000,
    "22": 41000,
    "23": 50000,
    "24": 62000,
    "25": 75000,
    "26": 90000,
    "27": 105000,
    "28": 120000,
    "29": 135000,
    "30": 155000,
}


def has_dnd_mechanics(character: CharacterRecord) -> bool:
    mechanics = character.mechanics if isinstance(character.mechanics, dict) else {}
    if mechanics.get("ruleset_id") == "dnd5e_basic":
        return True
    sheet = mechanics.get("dnd5e_sheet")
    return isinstance(sheet, dict) and bool(sheet)


def experience_points(character: CharacterRecord) -> int:
    mechanics = character.mechanics if isinstance(character.mechanics, dict) else {}
    runtime = mechanics.get(DND_RUNTIME_KEY)
    if isinstance(runtime, dict) and "experience_points" in runtime:
        return _safe_nonnegative_int(runtime.get("experience_points"))
    if "experience_points" in mechanics:
        return _safe_nonnegative_int(mechanics.get("experience_points"))

    identity = _sheet_identity(mechanics)
    return _safe_nonnegative_int(identity.get("experience_points"))


def total_level(character: CharacterRecord) -> int:
    mechanics = character.mechanics if isinstance(character.mechanics, dict) else {}
    identity = _sheet_identity(mechanics)
    level = _safe_nonnegative_int(identity.get("total_level"))
    if level:
        return level
    classes = identity.get("classes") or []
    if not isinstance(classes, list):
        return 0
    return sum(
        _safe_nonnegative_int(entry.get("level"))
        for entry in classes
        if isinstance(entry, dict)
    )


def experience_view(character: CharacterRecord) -> dict[str, Any]:
    xp = experience_points(character)
    level = total_level(character)
    earned_level = earned_level_for_xp(xp)
    eligible_level = earned_level if level and earned_level > level else 0
    next_level = level + 1 if 0 < level < 20 else 0
    next_threshold = DND5E_XP_THRESHOLDS.get(next_level, 0)
    xp_to_next = max(0, next_threshold - xp) if next_threshold else 0
    return {
        "experience_points": xp,
        "total_level": level,
        "earned_level": earned_level,
        "eligible_level": eligible_level,
        "level_available": bool(eligible_level),
        "next_level": next_level,
        "next_level_threshold": next_threshold,
        "xp_to_next_level": xp_to_next,
    }


def encounter_xp_value(character: CharacterRecord | None) -> int:
    if character is None:
        return 0
    mechanics = character.mechanics if isinstance(character.mechanics, dict) else {}
    direct = _first_int(
        mechanics,
        ("xp_value", "experience_award", "encounter_xp", "defeat_xp", "challenge_xp"),
    )
    if direct:
        return direct

    sheet = mechanics.get("dnd5e_sheet")
    sheet = sheet if isinstance(sheet, dict) else {}
    statblock = sheet.get("statblock")
    statblock = statblock if isinstance(statblock, dict) else {}
    direct = _first_int(
        statblock,
        ("xp", "xp_value", "experience_award", "encounter_xp", "defeat_xp"),
    )
    if direct:
        return direct

    challenge = statblock.get("challenge")
    if not isinstance(challenge, dict):
        challenge = sheet.get("challenge")
    challenge = challenge if isinstance(challenge, dict) else {}
    direct = _first_int(challenge, ("xp", "xp_value", "experience_award"))
    if direct:
        return direct

    for source, keys in (
        (mechanics, ("challenge_rating", "cr")),
        (sheet, ("challenge_rating", "cr")),
        (statblock, ("challenge_rating", "cr")),
        (challenge, ("rating", "challenge_rating", "cr")),
    ):
        rating = _first_challenge_rating(source, keys)
        if rating:
            return DND5E_CHALLENGE_XP.get(rating, 0)
    return 0


def award_experience(
    character: CharacterRecord,
    amount: int,
    *,
    source: str = "",
    turn_index: int = 0,
    awarded_at: str | None = None,
) -> dict[str, Any]:
    if amount <= 0:
        raise ValueError("Experience award must be positive.")
    if not has_dnd_mechanics(character):
        raise ValueError(
            f"{character.character_id!r} does not have D&D mechanics attached."
        )

    before = experience_points(character)
    after = before + amount
    _set_experience_points(character, after)
    _append_award_log(
        character,
        amount=amount,
        before=before,
        after=after,
        source=source,
        turn_index=turn_index,
        awarded_at=awarded_at,
    )

    view = experience_view(character)
    view.update({
        "amount": amount,
        "before": before,
        "after": after,
    })
    return view


def earned_level_for_xp(xp: int) -> int:
    total = max(0, int(xp))
    earned = 1
    for level, threshold in sorted(DND5E_XP_THRESHOLDS.items()):
        if total >= threshold:
            earned = level
    return earned


def format_experience_progress(view: dict[str, Any]) -> str:
    total = _safe_nonnegative_int(view.get("experience_points"))
    level_available = bool(view.get("level_available"))
    eligible_level = _safe_nonnegative_int(view.get("eligible_level"))
    next_level = _safe_nonnegative_int(view.get("next_level"))
    xp_to_next = _safe_nonnegative_int(view.get("xp_to_next_level"))
    total_level_value = _safe_nonnegative_int(view.get("total_level"))

    line = f"XP {total:,}"
    if level_available and eligible_level:
        line += f" (eligible for level {eligible_level})"
    elif next_level:
        line += f" ({xp_to_next:,} XP to level {next_level})"
    elif total_level_value >= 20:
        line += " (level 20)"
    return line


def _set_experience_points(character: CharacterRecord, value: int) -> None:
    mechanics = character.mechanics
    if not isinstance(mechanics, dict):
        mechanics = {}
        character.mechanics = mechanics

    total = max(0, int(value))
    mechanics["experience_points"] = total
    runtime = mechanics.get(DND_RUNTIME_KEY)
    if not isinstance(runtime, dict):
        runtime = {}
        mechanics[DND_RUNTIME_KEY] = runtime
    runtime["experience_points"] = total

    sheet = mechanics.get("dnd5e_sheet")
    if isinstance(sheet, dict):
        identity = sheet.get("identity")
        if not isinstance(identity, dict):
            identity = {}
            sheet["identity"] = identity
        identity["experience_points"] = total


def _append_award_log(
    character: CharacterRecord,
    *,
    amount: int,
    before: int,
    after: int,
    source: str,
    turn_index: int,
    awarded_at: str | None,
) -> None:
    mechanics = character.mechanics
    if not isinstance(mechanics, dict):
        return
    runtime = mechanics.get(DND_RUNTIME_KEY)
    if not isinstance(runtime, dict):
        runtime = {}
        mechanics[DND_RUNTIME_KEY] = runtime
    log = runtime.get("experience_awards")
    if not isinstance(log, list):
        log = []
        runtime["experience_awards"] = log
    timestamp = awarded_at or datetime.now(timezone.utc).isoformat()
    log.append({
        "amount": amount,
        "before": before,
        "after": after,
        "source": source.strip()[:500],
        "turn_index": max(0, int(turn_index or 0)),
        "awarded_at": timestamp,
    })
    if len(log) > 100:
        del log[:-100]


def _sheet_identity(mechanics: dict[str, Any]) -> dict[str, Any]:
    sheet = mechanics.get("dnd5e_sheet")
    if not isinstance(sheet, dict):
        return {}
    identity = sheet.get("identity")
    return identity if isinstance(identity, dict) else {}


def _first_int(source: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if key in source:
            value = _safe_nonnegative_int(source.get(key))
            if value:
                return value
    return 0


def _first_challenge_rating(
    source: dict[str, Any],
    keys: tuple[str, ...],
) -> str:
    for key in keys:
        rating = _normalize_challenge_rating(source.get(key))
        if rating:
            return rating
    return ""


def _normalize_challenge_rating(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(max(0, value))
    if isinstance(value, float):
        fractions = {
            0.125: "1/8",
            0.25: "1/4",
            0.5: "1/2",
        }
        for numeric, label in fractions.items():
            if abs(value - numeric) < 0.001:
                return label
        if value.is_integer():
            return str(max(0, int(value)))
        return ""

    text = str(value).strip().lower()
    if not text:
        return ""
    fraction = re.search(r"\b(1\s*/\s*[248])\b", text)
    if fraction:
        return fraction.group(1).replace(" ", "")
    number = re.search(r"\b\d+(?:\.\d+)?\b", text)
    if not number:
        return ""
    numeric = float(number.group(0))
    return _normalize_challenge_rating(numeric)


def _safe_nonnegative_int(value: Any) -> int:
    if isinstance(value, str):
        match = re.search(r"\d[\d,]*", value)
        if not match:
            return 0
        value = match.group(0).replace(",", "")
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
