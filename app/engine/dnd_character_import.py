from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.characters import (
    ActorFact,
    ActorRecord,
    CharacterRecord,
    CharacterVisuals,
    PublicSheet,
)


ABILITY_BY_DDB_ID = {
    1: "str",
    2: "dex",
    3: "con",
    4: "int",
    5: "wis",
    6: "cha",
}

ABILITY_NAME_TO_ID = {
    "strength": "str",
    "dexterity": "dex",
    "constitution": "con",
    "intelligence": "int",
    "wisdom": "wis",
    "charisma": "cha",
}

SKILL_ABILITIES = {
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

RESET_TYPES = {
    1: "short_rest",
    2: "long_rest",
    3: "long_rest",
    4: "none",
}


class DndCharacterImportError(ValueError):
    pass


def load_dndbeyond_export(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DndCharacterImportError("D&D Beyond export must be a JSON object.")
    return data


def normalize_dndbeyond_export(
    export: dict[str, Any],
    *,
    include_raw_source: bool = True,
) -> dict[str, Any]:
    """Normalize a user-supplied D&D Beyond browser export.

    The input is the envelope emitted by the local browser helper:
    `{exporter, source, raw}`. The returned dict follows
    `dnd_character_snapshot.schema.json`.
    """

    char = _character_payload(export)
    source = _source_block(export, char)
    abilities = _ability_scores(char)
    proficiency_bonus = _proficiency_bonus(char)
    modifiers = _flatten_modifiers(char)

    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "ruleset_id": _ruleset_id(char),
        "source": source,
        "identity": _identity(char),
        "statblock": {
            "ability_scores": abilities,
            "proficiency_bonus": proficiency_bonus,
            "skills": _skills(abilities, proficiency_bonus, modifiers),
            "saves": _saves(abilities, proficiency_bonus, modifiers),
            "defenses": _defenses(char, abilities, modifiers),
            "resources": _resources(char),
            "actions": _actions(char),
            "spellcasting": _spellcasting(char, abilities, proficiency_bonus),
            "inventory": _inventory(char),
            "features": _features(char),
            "effects": _effects(modifiers),
            "proficiencies": _proficiencies(modifiers),
            "languages": _languages(modifiers),
            "related_actors": _related_actors(char),
            "raw": {
                "activeSourceCategories": char.get("activeSourceCategories"),
                "configuration": char.get("configuration"),
                "preferences": char.get("preferences"),
            },
        },
        "import_report": {
            "status": "ok",
            "warnings": [],
            "losses": [],
            "notes": (
                "DDB browser export normalized locally. Full source payload is "
                "stored only when raw_source is present."
            ),
        },
    }
    if include_raw_source:
        snapshot["raw_source"] = export.get("raw", {})
    return _clean(snapshot)


def mechanics_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    statblock = snapshot.get("statblock") or {}
    abilities = statblock.get("ability_scores") or {}
    skills = statblock.get("skills") or {}
    saves = statblock.get("saves") or {}
    defenses = statblock.get("defenses") or {}
    hp = defenses.get("hit_points") or {}
    resources = {
        res.get("id"): {
            "name": res.get("name", ""),
            "kind": res.get("kind", ""),
            "current": res.get("current"),
            "max": res.get("max"),
            "reset": (res.get("reset") or {}).get("type", ""),
        }
        for res in statblock.get("resources", [])
        if isinstance(res, dict) and res.get("id")
    }

    return {
        "ruleset_id": "dnd5e_basic",
        "ability_scores": {
            ability: int((value or {}).get("score", 10) or 10)
            for ability, value in abilities.items()
            if ability in ABILITY_BY_DDB_ID.values()
        },
        "proficiency_bonus": statblock.get("proficiency_bonus", 0),
        "skill_proficiencies": [
            name
            for name, bonus in skills.items()
            if (bonus or {}).get("proficiency_multiplier", 0) > 0
        ],
        "saving_throw_proficiencies": [
            ability
            for ability, bonus in saves.items()
            if (bonus or {}).get("proficiency_multiplier", 0) > 0
        ],
        "armor_class": (defenses.get("armor_class") or {}).get("value", 0),
        "experience_points": (snapshot.get("identity") or {}).get(
            "experience_points", 0
        ),
        "hit_points": {
            "current": hp.get("current", 0),
            "max": hp.get("max", 0),
            "temporary": hp.get("temporary", 0),
        },
        "conditions": [
            cond.get("name", cond.get("id", ""))
            for cond in defenses.get("conditions", [])
            if isinstance(cond, dict)
        ],
        "resources": resources,
        "raw": {
            "source": snapshot.get("source", {}),
            "import_report": snapshot.get("import_report", {}),
        },
        "dnd5e_sheet": snapshot,
    }


def character_record_from_snapshot(
    snapshot: dict[str, Any],
    *,
    character_id: str | None = None,
    location: str = "",
    is_playable: bool = True,
) -> CharacterRecord:
    identity = snapshot.get("identity") or {}
    char_id = character_id or _slug(identity.get("name") or "dnd_character")
    appearance = identity.get("appearance", "")
    actor_facts = _actor_facts_from_identity(identity)
    return CharacterRecord(
        character_id=char_id,
        name=identity.get("name") or char_id,
        location=location,
        is_playable=is_playable,
        public_sheet=PublicSheet(
            role=_role(identity),
            appearance=appearance,
            faction="",
        ),
        visuals=CharacterVisuals(default_loadout=appearance),
        actor=ActorRecord(facts=actor_facts) if actor_facts else None,
        mechanics=mechanics_from_snapshot(snapshot),
    )


def _actor_facts_from_identity(identity: dict[str, Any]) -> list[ActorFact]:
    """Compile reviewed D&D source identity notes into sparse actor facts."""

    facts: list[ActorFact] = []
    seen: set[str] = set()
    prefixes = {
        "backstory": "You remember this account as your own history: ",
        "personality": "You recognize this description in yourself: ",
    }
    for field_name in ("backstory", "personality"):
        text = str(identity.get(field_name) or "").strip()
        normalized = " ".join(text.casefold().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        facts.append(
            ActorFact(
                origin="lived",
                text=f"{prefixes[field_name]}{text}",
            )
        )
    return facts


def _character_payload(export: dict[str, Any]) -> dict[str, Any]:
    raw = export.get("raw", export)
    if isinstance(raw, dict) and isinstance(raw.get("data"), dict):
        return raw["data"]
    if isinstance(export.get("data"), dict):
        return export["data"]
    if isinstance(raw, dict):
        return raw
    raise DndCharacterImportError("No D&D Beyond character payload found.")


def _source_block(export: dict[str, Any], char: dict[str, Any]) -> dict[str, Any]:
    raw_source = export.get("source") or {}
    raw_payload = export.get("raw", export)
    exported_at = raw_source.get("exported_at") or datetime.now(
        timezone.utc
    ).isoformat()
    source: dict[str, Any] = {
        "type": raw_source.get("type") or "dndbeyond_browser_export",
        "source_character_id": str(
            raw_source.get("character_id") or char.get("id") or ""
        ),
        "source_url": raw_source.get("url") or char.get("readonlyUrl") or "",
        "source_hash": _stable_hash(raw_payload),
        "exported_at": exported_at,
        "exporter": export.get("exporter") or {},
    }
    campaign = char.get("campaign") or {}
    if campaign.get("id") is not None:
        source["campaign_id"] = str(campaign["id"])
    if char.get("username"):
        source["owner_label"] = str(char["username"])
    return source


def _ruleset_id(char: dict[str, Any]) -> str:
    categories = {
        str(cat).lower()
        for cat in (char.get("activeSourceCategories") or [])
    }
    if any("2024" in cat for cat in categories):
        return "dnd5e_2024"
    return "dnd5e_2014"


def _identity(char: dict[str, Any]) -> dict[str, Any]:
    race = char.get("race") or {}
    background = char.get("background") or {}
    bg_definition = background.get("definition") or {}
    notes = char.get("notes") or {}
    traits = char.get("traits") or {}
    appearance_parts = [
        _label_value("Age", char.get("age")),
        _label_value("Height", char.get("height")),
        _label_value("Weight", char.get("weight")),
        _label_value("Eyes", char.get("eyes")),
        _label_value("Hair", char.get("hair")),
        _label_value("Skin", char.get("skin")),
    ]
    appearance = "; ".join(part for part in appearance_parts if part)

    identity: dict[str, Any] = {
        "name": char.get("name") or "Unnamed Character",
        "display_name": char.get("name") or "",
        "species": race.get("fullName") or race.get("baseName") or "",
        "background": bg_definition.get("name") or background.get("name") or "",
        "classes": _classes(char),
        "total_level": _total_level(char),
        "experience_points": char.get("currentXp") or 0,
        "size": _size_name(char),
        "creature_type": race.get("creatureTypeName") or "",
        "appearance": appearance,
        "portrait_url": race.get("portraitAvatarUrl") or race.get("avatarUrl") or "",
        "backstory": _first_string(
            notes.get("backstory"),
            notes.get("organizations"),
            notes.get("allies"),
        ),
        "personality": _first_string(
            traits.get("personalityTraits"),
            traits.get("ideals"),
            traits.get("bonds"),
            traits.get("flaws"),
        ),
        "source_refs": _source_refs("character", char),
        "raw": {
            "alignmentId": char.get("alignmentId"),
            "gender": char.get("gender"),
            "faith": char.get("faith"),
            "lifestyle": char.get("lifestyle"),
        },
    }
    return identity


def _classes(char: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for entry in char.get("classes") or []:
        definition = entry.get("definition") or {}
        subclass = entry.get("subclassDefinition") or {}
        out.append({
            "id": str(entry.get("id") or definition.get("id") or ""),
            "name": definition.get("name") or entry.get("name") or "",
            "subclass": subclass.get("name") or "",
            "level": entry.get("level") or 0,
            "hit_die": definition.get("hitDice") or 6,
            "spellcasting_ability": ABILITY_BY_DDB_ID.get(
                definition.get("spellCastingAbilityId")
                or subclass.get("spellCastingAbilityId")
            ),
            "source_refs": _source_refs("class", definition),
            "raw": {
                "characterClassId": entry.get("id"),
                "definitionId": definition.get("id"),
                "subclassDefinitionId": subclass.get("id"),
                "isStartingClass": entry.get("isStartingClass"),
                "hitDiceUsed": entry.get("hitDiceUsed"),
            },
        })
    return out


def _ability_scores(char: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_scores = _stat_value_map(char.get("stats"))
    bonus_scores = _stat_value_map(char.get("bonusStats"))
    override_scores = _stat_value_map(char.get("overrideStats"))
    modifier_bonuses = {ability: 0 for ability in ABILITY_BY_DDB_ID.values()}

    for _, mod in _flatten_modifiers(char):
        if mod.get("type") != "bonus":
            continue
        ability = _ability_from_score_subtype(mod.get("subType"))
        if ability:
            modifier_bonuses[ability] += _modifier_value(mod)

    out: dict[str, dict[str, Any]] = {}
    for ddb_id, ability in ABILITY_BY_DDB_ID.items():
        base = int(base_scores.get(ability, 10) or 10)
        score = (
            base
            + int(bonus_scores.get(ability) or 0)
            + modifier_bonuses[ability]
        )
        if override_scores.get(ability) is not None:
            score = int(override_scores[ability] or 10)
        out[ability] = {
            "score": score,
            "modifier": ability_modifier(score),
            "base_score": base,
            "override_score": override_scores.get(ability),
            "bonus": score - base,
            "sources": _ability_sources(char, ability),
            "raw": {"ddb_stat_id": ddb_id},
        }
    return out


def ability_modifier(score: int) -> int:
    return (int(score) - 10) // 2


def _stat_value_map(stats: Any) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for item in stats or []:
        ability = ABILITY_BY_DDB_ID.get(item.get("id"))
        if ability:
            out[ability] = item.get("value")
    return out


def _ability_from_score_subtype(subtype: Any) -> str | None:
    text = str(subtype or "").lower().replace("-", " ")
    for name, ability in ABILITY_NAME_TO_ID.items():
        if text == f"{name} score":
            return ability
    return None


def _ability_sources(char: dict[str, Any], ability: str) -> list[dict[str, Any]]:
    sources = []
    for bucket, mod in _flatten_modifiers(char):
        if _ability_from_score_subtype(mod.get("subType")) == ability:
            sources.append(_modifier_source_ref(bucket, mod))
    return sources


def _proficiency_bonus(char: dict[str, Any]) -> int:
    level = _total_level(char)
    if level <= 0:
        return 0
    return 2 + ((level - 1) // 4)


def _total_level(char: dict[str, Any]) -> int:
    return sum((entry.get("level") or 0) for entry in char.get("classes") or [])


def _skills(
    abilities: dict[str, dict[str, Any]],
    proficiency_bonus: int,
    modifiers: list[tuple[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    profs = _proficiency_multipliers(modifiers, set(SKILL_ABILITIES))
    bonuses = _roll_bonuses(modifiers, set(SKILL_ABILITIES))
    out = {}
    for skill, ability in SKILL_ABILITIES.items():
        mult = profs.get(skill, 0)
        bonus = bonuses.get(skill, 0)
        value = (
            (abilities.get(ability) or {}).get("modifier", 0)
            + int(proficiency_bonus * mult)
            + bonus
        )
        out[skill] = {
            "ability": ability,
            "value": value,
            "proficiency_multiplier": mult,
            "bonus": bonus,
            "advantage_state": _advantage_state(modifiers, skill),
            "sources": _roll_sources(modifiers, skill),
        }
    return out


def _saves(
    abilities: dict[str, dict[str, Any]],
    proficiency_bonus: int,
    modifiers: list[tuple[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    out = {}
    for ability in ABILITY_BY_DDB_ID.values():
        key = f"{_ability_name(ability)} saving throws"
        mult = _save_proficiency_multiplier(modifiers, ability)
        bonus = _roll_bonuses(modifiers, {key, "saving throws"}).get(key, 0)
        global_bonus = _roll_bonuses(modifiers, {key, "saving throws"}).get(
            "saving throws", 0
        )
        value = (
            (abilities.get(ability) or {}).get("modifier", 0)
            + int(proficiency_bonus * mult)
            + bonus
            + global_bonus
        )
        out[ability] = {
            "ability": ability,
            "value": value,
            "proficiency_multiplier": mult,
            "bonus": bonus + global_bonus,
            "advantage_state": _advantage_state(modifiers, key),
            "sources": _roll_sources(modifiers, key),
        }
    return out


def _defenses(
    char: dict[str, Any],
    abilities: dict[str, dict[str, Any]],
    modifiers: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "armor_class": _armor_class(char, abilities, modifiers),
        "hit_points": _hit_points(char),
        "initiative": _initiative(abilities, modifiers),
        "movement": _movement(char),
        "senses": _senses(char, modifiers),
        "damage_resistances": _typed_tags(modifiers, "resistance"),
        "damage_immunities": _typed_tags(modifiers, "immunity"),
        "damage_vulnerabilities": _typed_tags(modifiers, "vulnerability"),
        "condition_immunities": _typed_tags(modifiers, "condition-immunity"),
        "conditions": _conditions(char),
        "exhaustion_level": _exhaustion_level(char),
        "death_saves": _death_saves(char),
        "raw": {
            "customDefenseAdjustments": char.get("customDefenseAdjustments"),
            "customSenses": char.get("customSenses"),
            "customSpeeds": char.get("customSpeeds"),
        },
    }


def _armor_class(
    char: dict[str, Any],
    abilities: dict[str, dict[str, Any]],
    modifiers: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    dex_mod = (abilities.get("dex") or {}).get("modifier", 0)
    armor_base = 10 + dex_mod
    components = [{"name": "base", "value": 10}, {"name": "dex", "value": dex_mod}]
    shield_bonus = 0

    for item in char.get("inventory") or []:
        if not item.get("equipped"):
            continue
        definition = item.get("definition") or {}
        if definition.get("filterType") != "Armor":
            continue
        ac_value = definition.get("armorClass")
        if ac_value is None:
            continue
        armor_type = definition.get("armorTypeId")
        name = definition.get("baseArmorName") or definition.get("name") or "armor"
        if armor_type == 4 or str(name).lower() == "shield":
            shield_bonus += int(ac_value)
            components.append({"name": name, "value": int(ac_value)})
            continue
        armor_dex = _armor_dex_bonus(armor_type, dex_mod)
        candidate = int(ac_value) + armor_dex
        if candidate > armor_base:
            armor_base = candidate
            components = [
                {"name": name, "value": int(ac_value)},
                {"name": "dex", "value": armor_dex},
            ]

    bonus = sum(
        _modifier_value(mod)
        for _, mod in modifiers
        if mod.get("type") == "bonus"
        and _canonical_subtype(mod.get("subType")) == "armor class"
    )
    if bonus:
        components.append({"name": "armor class bonus", "value": bonus})
    value = armor_base + shield_bonus + bonus
    return {
        "value": value,
        "calculation": "ddb_import_approximation",
        "components": components,
        "sources": [
            _modifier_source_ref(bucket, mod)
            for bucket, mod in modifiers
            if _canonical_subtype(mod.get("subType")) == "armor class"
        ],
    }


def _armor_dex_bonus(armor_type: Any, dex_mod: int) -> int:
    if armor_type == 2:
        return min(dex_mod, 2)
    if armor_type == 3:
        return 0
    return dex_mod


def _hit_points(char: dict[str, Any]) -> dict[str, Any]:
    maximum = (
        char.get("overrideHitPoints")
        if char.get("overrideHitPoints") is not None
        else (char.get("baseHitPoints") or 0) + (char.get("bonusHitPoints") or 0)
    )
    removed = char.get("removedHitPoints") or 0
    return {
        "current": max(0, int(maximum or 0) - int(removed)),
        "max": int(maximum or 0),
        "temporary": int(char.get("temporaryHitPoints") or 0),
        "hit_dice": _hit_dice(char),
        "raw": {
            "baseHitPoints": char.get("baseHitPoints"),
            "bonusHitPoints": char.get("bonusHitPoints"),
            "overrideHitPoints": char.get("overrideHitPoints"),
            "removedHitPoints": char.get("removedHitPoints"),
        },
    }


def _hit_dice(char: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for entry in char.get("classes") or []:
        definition = entry.get("definition") or {}
        level = entry.get("level") or 0
        used = entry.get("hitDiceUsed") or 0
        out.append({
            "die": definition.get("hitDice") or 6,
            "current": max(0, level - used),
            "max": level,
            "class_name": definition.get("name") or "",
        })
    return out


def _initiative(
    abilities: dict[str, dict[str, Any]],
    modifiers: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    bonus = _roll_bonuses(modifiers, {"initiative"}).get("initiative", 0)
    return {
        "ability": "dex",
        "value": (abilities.get("dex") or {}).get("modifier", 0) + bonus,
        "proficiency_multiplier": 0,
        "bonus": bonus,
        "advantage_state": _advantage_state(modifiers, "initiative"),
        "sources": _roll_sources(modifiers, "initiative"),
    }


def _movement(char: dict[str, Any]) -> dict[str, Any]:
    race = char.get("race") or {}
    speeds = race.get("weightSpeeds") or {}
    custom = char.get("customSpeeds") or {}
    out: dict[str, Any] = {}
    for key in ("walk", "fly", "swim", "climb", "burrow"):
        value = custom.get(key) or speeds.get(key)
        if value is not None:
            out[key] = {"value": int(value), "unit": "ft"}
    if not out:
        out["walk"] = {"value": 30, "unit": "ft"}
    out["raw"] = {"raceWeightSpeeds": speeds, "customSpeeds": custom}
    return out


def _senses(
    char: dict[str, Any],
    modifiers: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    special = []
    for bucket, mod in modifiers:
        if mod.get("type") in {"set-base", "set"} and "vision" in str(
            mod.get("subType") or ""
        ):
            special.append({
                "id": _slug(mod.get("subType")),
                "name": mod.get("friendlySubtypeName") or mod.get("subType"),
                "value": str(mod.get("fixedValue") or ""),
                "sources": [_modifier_source_ref(bucket, mod)],
            })
    return {
        "special": special,
        "raw": {"customSenses": char.get("customSenses")},
    }


def _typed_tags(
    modifiers: list[tuple[str, dict[str, Any]]],
    modifier_type: str,
) -> list[dict[str, Any]]:
    out = []
    for bucket, mod in modifiers:
        if str(mod.get("type") or "").lower() != modifier_type:
            continue
        out.append({
            "id": _slug(mod.get("subType")),
            "name": mod.get("friendlySubtypeName") or mod.get("subType") or "",
            "condition": mod.get("restriction") or "",
            "sources": [_modifier_source_ref(bucket, mod)],
            "raw": {"modifier_id": mod.get("id")},
        })
    return out


def _conditions(char: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for cond in char.get("conditions") or []:
        if isinstance(cond, dict):
            out.append({
                "id": str(cond.get("id") or cond.get("definitionKey") or ""),
                "name": cond.get("name") or cond.get("definitionName") or "",
                "raw": cond,
            })
    return out


def _exhaustion_level(char: dict[str, Any]) -> int:
    for cond in char.get("conditions") or []:
        name = str((cond or {}).get("name") or "").lower()
        if "exhaust" in name:
            return int((cond or {}).get("level") or 1)
    return 0


def _death_saves(char: dict[str, Any]) -> dict[str, int]:
    death = char.get("deathSaves") or {}
    return {
        "successes": int(death.get("successCount") or 0),
        "failures": int(death.get("failCount") or 0),
    }


def _resources(char: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for bucket, action in _iter_bucketed(char.get("actions")):
        resource = _limited_use_resource(action, f"action:{bucket}")
        if resource and resource["id"] not in seen:
            seen.add(resource["id"])
            out.append(resource)

    for bucket, spell in _iter_bucketed(char.get("spells")):
        resource = _limited_use_resource(spell, f"spell:{bucket}")
        if resource and resource["id"] not in seen:
            seen.add(resource["id"])
            out.append(resource)

    for item in char.get("inventory") or []:
        resource = _limited_use_resource(item, "item")
        if resource and resource["id"] not in seen:
            seen.add(resource["id"])
            out.append(resource)

    for level, slot in _spell_slots(char).items():
        out.append({
            "id": f"spell_slot_{level}",
            "name": f"Level {level} Spell Slot",
            "kind": "spell_slot",
            "current": slot["current"],
            "max": slot["max"],
            "unit": "slot",
            "reset": {"type": "long_rest"},
        })

    pact = _pact_slots(char)
    if pact:
        out.append({
            "id": "pact_slot",
            "name": f"Level {pact['level']} Pact Slot",
            "kind": "pact_slot",
            "current": pact["current"],
            "max": pact["max"],
            "unit": "slot",
            "reset": {"type": "short_rest"},
        })

    return out


def _limited_use_resource(obj: dict[str, Any], source_kind: str) -> dict[str, Any] | None:
    limited = obj.get("limitedUse")
    if not isinstance(limited, dict):
        return None
    max_uses = limited.get("maxUses")
    if max_uses in (None, 0):
        return None
    used = limited.get("numberUsed") or 0
    name = obj.get("name") or (obj.get("definition") or {}).get("name") or "Limited Use"
    res_id = f"{_slug(source_kind)}_{_slug(name)}_{obj.get('id') or ''}".strip("_")
    return {
        "id": res_id,
        "name": limited.get("name") or name,
        "kind": _resource_kind(source_kind),
        "current": max(0, int(max_uses) - int(used)),
        "max": int(max_uses),
        "spent": int(used),
        "unit": "use",
        "reset": {
            "type": RESET_TYPES.get(limited.get("resetType"), "special"),
            "text": str(limited.get("resetType") or ""),
        },
        "source_refs": _source_refs(source_kind, obj),
        "raw": {"limitedUse": limited},
    }


def _resource_kind(source_kind: str) -> str:
    if source_kind.startswith("item"):
        return "item_charge"
    if source_kind.startswith("spell"):
        return "limited_use"
    if source_kind.startswith("action:class"):
        return "class_feature"
    if source_kind.startswith("action:race"):
        return "species_feature"
    if source_kind.startswith("action:feat"):
        return "feat"
    return "other"


def _actions(char: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bucket, action in _iter_bucketed(char.get("actions")):
        name = action.get("name") or "Action"
        out.append({
            "id": f"ddb_action_{bucket}_{action.get('id') or _slug(name)}",
            "name": name,
            "kind": _action_kind(bucket, action),
            "activation": _activation(action),
            "range": _range(action.get("range")),
            "target": _target(action.get("range")),
            "attack": _attack_profile(action),
            "save": _save_profile(action),
            "damage": _damage_components(action),
            "consumes": _resource_costs(action, f"action:{bucket}"),
            "automation": {"format": "ddb_definition", "coverage": "partial", "data": action},
            "description": _description(action),
            "source_refs": _source_refs(f"action:{bucket}", action),
            "raw": _raw_without_text(action),
        })

    for item in char.get("inventory") or []:
        definition = item.get("definition") or {}
        if definition.get("filterType") != "Weapon":
            continue
        name = definition.get("name") or "Weapon"
        out.append({
            "id": f"ddb_item_attack_{item.get('id') or _slug(name)}",
            "name": name,
            "kind": "attack",
            "activation": {"type": "action"},
            "attack": _weapon_attack_profile(item, char),
            "damage": _item_damage_components(item),
            "automation": {"format": "ddb_definition", "coverage": "partial", "data": item},
            "description": _description(definition),
            "source_refs": _source_refs("item", item),
            "raw": _raw_without_text(item),
        })
    return out


def _action_kind(bucket: str, action: dict[str, Any]) -> str:
    if bucket == "item":
        return "item"
    if bucket == "feat":
        return "feature"
    activation = str((action.get("activationType") or "")).lower()
    if "reaction" in activation:
        return "reaction"
    if "bonus" in activation:
        return "bonus_action"
    return "feature"


def _spellcasting(
    char: dict[str, Any],
    abilities: dict[str, dict[str, Any]],
    proficiency_bonus: int,
) -> dict[str, Any]:
    profiles = []
    for entry in char.get("classes") or []:
        definition = entry.get("definition") or {}
        if not definition.get("canCastSpells"):
            continue
        ability = ABILITY_BY_DDB_ID.get(definition.get("spellCastingAbilityId"))
        if not ability:
            continue
        mod = (abilities.get(ability) or {}).get("modifier", 0)
        class_id = str(entry.get("id") or definition.get("id"))
        profiles.append({
            "id": f"class_{class_id}",
            "name": definition.get("name") or "Spellcasting",
            "ability": ability,
            "spell_attack_bonus": proficiency_bonus + mod,
            "spell_save_dc": 8 + proficiency_bonus + mod,
            "source_refs": _source_refs("class", definition),
            "raw": {"characterClassId": entry.get("id")},
        })

    return {
        "profiles": profiles,
        "slots": _spell_slots(char),
        "pact_slots": _pact_slots(char),
        "spells": _spells(char),
        "raw": {
            "spellSlots": char.get("spellSlots"),
            "pactMagic": char.get("pactMagic"),
        },
    }


def _spell_slots(char: dict[str, Any]) -> dict[str, dict[str, int]]:
    out = {}
    for slot in char.get("spellSlots") or []:
        level = slot.get("level")
        if not level:
            continue
        available = int(slot.get("available") or 0)
        used = int(slot.get("used") or 0)
        maximum = max(available, used + available)
        if maximum:
            out[str(level)] = {
                "current": max(0, maximum - used),
                "max": maximum,
            }
    return out


def _pact_slots(char: dict[str, Any]) -> dict[str, int] | None:
    best = None
    for slot in char.get("pactMagic") or []:
        available = int(slot.get("available") or 0)
        used = int(slot.get("used") or 0)
        maximum = max(available, used + available)
        if maximum:
            best = {
                "level": int(slot.get("level") or 1),
                "current": max(0, maximum - used),
                "max": maximum,
            }
    return best


def _spells(char: dict[str, Any]) -> list[dict[str, Any]]:
    spells: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for bucket, spell in _iter_bucketed(char.get("spells")):
        normalized = _spell(spell, bucket)
        key = (normalized["name"].lower(), str(normalized["level"]))
        if key not in seen:
            seen.add(key)
            spells.append(normalized)

    for class_spells in char.get("classSpells") or []:
        for spell in class_spells.get("spells") or []:
            normalized = _spell(spell, "class")
            key = (normalized["name"].lower(), str(normalized["level"]))
            if key not in seen:
                seen.add(key)
                spells.append(normalized)
    return spells


def _spell(spell: dict[str, Any], bucket: str) -> dict[str, Any]:
    definition = spell.get("definition") or {}
    name = definition.get("name") or spell.get("name") or "Spell"
    out = {
        "id": f"ddb_spell_{bucket}_{spell.get('id') or definition.get('id') or _slug(name)}",
        "name": name,
        "level": definition.get("level") or 0,
        "school": definition.get("school") or "",
        "prepared": bool(spell.get("prepared")),
        "always_prepared": bool(spell.get("alwaysPrepared")),
        "ritual": bool(definition.get("ritual")),
        "concentration": bool(definition.get("concentration")),
        "activation": _activation(definition),
        "range": _range(definition.get("range")),
        "target": _target(definition.get("range")),
        "components": _spell_components(definition),
        "duration": _duration(definition.get("duration")),
        "attack": _attack_profile(definition),
        "save": _save_profile(definition),
        "damage": _damage_components(definition),
        "healing": _healing_components(definition),
        "consumes": _resource_costs(spell, f"spell:{bucket}"),
        "automation": {"format": "ddb_definition", "coverage": "partial", "data": spell},
        "description": _description(definition),
        "source_refs": _source_refs(f"spell:{bucket}", definition),
        "raw": _raw_without_text(spell),
    }
    return out


def _inventory(char: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": [_inventory_item(item) for item in char.get("inventory") or []],
        "currency": char.get("currencies") or {},
        "raw": {
            "customItems": char.get("customItems"),
            "inventoryCount": len(char.get("inventory") or []),
        },
    }


def _inventory_item(item: dict[str, Any]) -> dict[str, Any]:
    definition = item.get("definition") or {}
    name = definition.get("name") or "Item"
    return {
        "id": f"ddb_item_{item.get('id') or definition.get('id') or _slug(name)}",
        "name": name,
        "kind": _item_kind(definition),
        "quantity": item.get("quantity") or 0,
        "equipped": bool(item.get("equipped")),
        "attuned": bool(item.get("isAttuned")),
        "requires_attunement": bool(definition.get("requiresAttunement")),
        "weight": definition.get("weight") or 0,
        "value_gp": _value_gp(definition),
        "armor_class": _item_armor_class(definition),
        "attack": _weapon_attack_profile(item, {}) if definition.get("filterType") == "Weapon" else None,
        "damage": _item_damage_components(item),
        "properties": [prop.get("name") for prop in definition.get("properties") or [] if prop.get("name")],
        "description": _description(definition),
        "source_refs": _source_refs("item", item),
        "raw": _raw_without_text(item),
    }


def _item_kind(definition: dict[str, Any]) -> str:
    filter_type = str(definition.get("filterType") or "").lower()
    item_type = str(definition.get("type") or "").lower()
    if filter_type == "weapon":
        return "weapon"
    if filter_type == "armor":
        if definition.get("armorTypeId") == 4 or "shield" in str(definition.get("name", "")).lower():
            return "shield"
        return "armor"
    if "wondrous" in filter_type or "wondrous" in item_type:
        return "wondrous_item"
    if "consumable" in filter_type or "potion" in item_type:
        return "consumable"
    if "tool" in filter_type or "tool" in item_type:
        return "tool"
    return "gear"


def _features(char: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for feat in char.get("feats") or []:
        definition = feat.get("definition") or {}
        if definition:
            out.append(_feature(definition, "feat"))
    for feature in char.get("features") or []:
        out.append(_feature(feature, "class"))
    for trait in (char.get("race") or {}).get("racialTraits") or []:
        definition = trait.get("definition") or trait
        out.append(_feature(definition, "species"))
    return out


def _feature(definition: dict[str, Any], kind: str) -> dict[str, Any]:
    name = definition.get("name") or "Feature"
    return {
        "id": f"ddb_{kind}_{definition.get('id') or _slug(name)}",
        "name": name,
        "kind": kind,
        "level": definition.get("requiredLevel") or definition.get("level") or 0,
        "description": _description(definition),
        "automation": {
            "format": "ddb_definition",
            "coverage": "partial",
            "data": definition,
        },
        "source_refs": _source_refs(kind, definition),
        "raw": _raw_without_text(definition),
    }


def _effects(modifiers: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for bucket, mod in modifiers:
        mtype = str(mod.get("type") or "")
        subtype = _canonical_subtype(mod.get("subType"))
        if mtype == "bonus" and (
            subtype.endswith("score")
            or subtype in SKILL_ABILITIES
            or subtype == "armor class"
        ):
            continue
        if mtype == "proficiency":
            continue
        out.append({
            "id": f"ddb_modifier_{bucket}_{mod.get('id') or _slug(subtype)}",
            "name": mod.get("friendlySubtypeName") or subtype or mtype,
            "kind": _effect_kind(mtype),
            "target": subtype,
            "operation": mtype,
            "value": mod.get("fixedValue"),
            "description": {
                "snippet": mod.get("restriction") or "",
                "prompt_safe": False,
            },
            "source_refs": [_modifier_source_ref(bucket, mod)],
            "raw": {"modifier": mod},
        })
    return out


def _effect_kind(modifier_type: str) -> str:
    if modifier_type in {"advantage", "disadvantage", "bonus"}:
        return "passive_bonus"
    if modifier_type in {"resistance", "immunity", "vulnerability"}:
        return "damage_adjustment"
    if modifier_type == "condition-immunity":
        return "condition"
    return "other"


def _proficiencies(modifiers: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    armor = []
    weapons = []
    tools = []
    other = []
    for _, mod in modifiers:
        if mod.get("type") != "proficiency":
            continue
        subtype = _canonical_subtype(mod.get("subType"))
        if subtype in SKILL_ABILITIES or subtype.endswith("saving throws"):
            continue
        name = mod.get("friendlySubtypeName") or subtype
        if any(word in subtype for word in ("armor", "shield")):
            armor.append(name)
        elif any(word in subtype for word in ("weapon", "sword", "dagger", "bow", "staff", "sling")):
            weapons.append(name)
        elif any(word in subtype for word in ("tools", "kit", "supplies")):
            tools.append(name)
        else:
            other.append(name)
    return {
        "armor": sorted(set(armor)),
        "weapons": sorted(set(weapons)),
        "tools": sorted(set(tools)),
        "other": sorted(set(other)),
    }


def _languages(modifiers: list[tuple[str, dict[str, Any]]]) -> list[str]:
    languages = []
    for _, mod in modifiers:
        if mod.get("type") == "language":
            languages.append(mod.get("friendlySubtypeName") or mod.get("subType") or "")
    return sorted({lang for lang in languages if lang})


def _related_actors(char: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for creature in char.get("creatures") or []:
        name = creature.get("name") or (creature.get("definition") or {}).get("name")
        if name:
            out.append({
                "id": f"ddb_creature_{creature.get('id') or _slug(name)}",
                "name": name,
                "relationship": creature.get("creatureType") or "linked creature",
                "raw": creature,
            })
    return out


def _flatten_modifiers(char: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    modifiers = char.get("modifiers") or {}
    if isinstance(modifiers, dict):
        for bucket, values in modifiers.items():
            for mod in values or []:
                if isinstance(mod, dict):
                    out.append((str(bucket), mod))
    return out


def _proficiency_multipliers(
    modifiers: list[tuple[str, dict[str, Any]]],
    keys: set[str],
) -> dict[str, float]:
    out = {key: 0.0 for key in keys}
    for _, mod in modifiers:
        subtype = _canonical_subtype(mod.get("subType"))
        if subtype not in keys:
            continue
        mtype = mod.get("type")
        if mtype == "expertise":
            out[subtype] = max(out[subtype], 2.0)
        elif mtype == "half-proficiency":
            out[subtype] = max(out[subtype], 0.5)
        elif mtype == "proficiency":
            out[subtype] = max(out[subtype], 1.0)
    return out


def _save_proficiency_multiplier(
    modifiers: list[tuple[str, dict[str, Any]]],
    ability: str,
) -> float:
    target = f"{_ability_name(ability)} saving throws"
    return _proficiency_multipliers(modifiers, {target}).get(target, 0)


def _roll_bonuses(
    modifiers: list[tuple[str, dict[str, Any]]],
    keys: set[str],
) -> dict[str, int]:
    out = {key: 0 for key in keys}
    for _, mod in modifiers:
        if mod.get("type") != "bonus":
            continue
        subtype = _canonical_subtype(mod.get("subType"))
        if subtype in keys:
            out[subtype] += _modifier_value(mod)
    return out


def _advantage_state(
    modifiers: list[tuple[str, dict[str, Any]]],
    key: str,
) -> str:
    canonical = _canonical_subtype(key)
    found_adv = False
    found_dis = False
    situational = False
    for _, mod in modifiers:
        subtype = _canonical_subtype(mod.get("subType"))
        if subtype not in {canonical, "saving throws"}:
            continue
        if mod.get("restriction"):
            situational = True
        if mod.get("type") == "advantage":
            found_adv = True
        if mod.get("type") == "disadvantage":
            found_dis = True
    if situational:
        return "situational"
    if found_adv and not found_dis:
        return "advantage"
    if found_dis and not found_adv:
        return "disadvantage"
    return "normal"


def _roll_sources(
    modifiers: list[tuple[str, dict[str, Any]]],
    key: str,
) -> list[dict[str, Any]]:
    canonical = _canonical_subtype(key)
    return [
        _modifier_source_ref(bucket, mod)
        for bucket, mod in modifiers
        if _canonical_subtype(mod.get("subType")) in {canonical, "saving throws"}
        and mod.get("type") in {"proficiency", "expertise", "half-proficiency", "bonus", "advantage", "disadvantage"}
    ]


def _modifier_value(mod: dict[str, Any]) -> int:
    value = mod.get("fixedValue")
    if value is None:
        value = mod.get("value")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _canonical_subtype(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", " ")


def _ability_name(ability: str) -> str:
    return {
        "str": "strength",
        "dex": "dexterity",
        "con": "constitution",
        "int": "intelligence",
        "wis": "wisdom",
        "cha": "charisma",
    }[ability]


def _activation(obj: dict[str, Any]) -> dict[str, Any]:
    activation_type = obj.get("activationType")
    activation = {
        "type": _activation_type(activation_type),
        "cost": obj.get("activationTime"),
    }
    if obj.get("activationCondition"):
        activation["condition"] = obj["activationCondition"]
    return activation


def _activation_type(value: Any) -> str:
    if isinstance(value, str):
        text = value.lower().replace(" ", "_")
        if text in {
            "action",
            "bonus_action",
            "reaction",
            "free",
            "no_action",
            "minute",
            "hour",
            "day",
            "legendary_action",
            "lair_action",
            "mythic_action",
            "passive",
            "special",
        }:
            return text
    return {
        1: "action",
        2: "no_action",
        3: "bonus_action",
        4: "reaction",
        5: "free",
        6: "minute",
        7: "hour",
        8: "special",
    }.get(value, "other")


def _range(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    normal = raw.get("range")
    if normal is None:
        normal = raw.get("rangeValue")
    out: dict[str, Any] = {
        "normal": normal,
        "long": raw.get("longRange"),
        "unit": "ft",
        "shape": raw.get("aoeType") or raw.get("origin") or "",
    }
    if raw.get("aoeSize") or raw.get("aoeValue"):
        out["text"] = f"AOE {raw.get('aoeSize') or raw.get('aoeValue')}"
    return {k: v for k, v in out.items() if v not in (None, "")}


def _target(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        "type": raw.get("aoeType") or raw.get("origin") or "",
        "size": raw.get("aoeSize") or raw.get("aoeValue"),
        "unit": "ft",
    }


def _attack_profile(obj: dict[str, Any]) -> dict[str, Any] | None:
    ability = ABILITY_BY_DDB_ID.get(obj.get("abilityModifierStatId"))
    attack_type = obj.get("attackType")
    if not ability and attack_type is None:
        return None
    return {
        "ability": ability,
        "bonus": obj.get("toHitBonus") or obj.get("attackBonus"),
        "to_hit_formula": obj.get("attackRoll") or "",
        "crit_on": obj.get("critRange") or 20,
        "raw": {
            "attackType": attack_type,
            "abilityModifierStatId": obj.get("abilityModifierStatId"),
        },
    }


def _weapon_attack_profile(item: dict[str, Any], char: dict[str, Any]) -> dict[str, Any]:
    definition = item.get("definition") or {}
    ability = "str" if definition.get("attackType") == 1 else "dex"
    return {
        "ability": ability,
        "proficiency_multiplier": 1,
        "bonus": None,
        "crit_on": 20,
        "properties": [
            prop.get("name")
            for prop in definition.get("properties") or []
            if prop.get("name")
        ],
        "raw": {
            "attackType": definition.get("attackType"),
            "characterId": char.get("id"),
        },
    }


def _save_profile(obj: dict[str, Any]) -> dict[str, Any] | None:
    ability = ABILITY_BY_DDB_ID.get(obj.get("saveDcAbilityId"))
    dc = obj.get("fixedSaveDc")
    if not ability and dc is None:
        return None
    return {
        "ability": ability,
        "dc": dc,
        "success": obj.get("saveSuccessDescription") or "",
        "failure": obj.get("saveFailDescription") or "",
    }


def _damage_components(obj: dict[str, Any]) -> list[dict[str, Any]]:
    damage = obj.get("damage") or obj.get("damageEffect")
    if isinstance(damage, dict):
        formula = damage.get("diceString") or _dice_formula(damage)
        if formula:
            return [{
                "formula": formula,
                "damage_type": obj.get("damageType") or damage.get("damageType") or "",
                "raw": {"damage": damage},
            }]
    return []


def _item_damage_components(item: dict[str, Any]) -> list[dict[str, Any]]:
    return _damage_components(item.get("definition") or {})


def _healing_components(obj: dict[str, Any]) -> list[dict[str, Any]]:
    healing = obj.get("healing")
    if isinstance(healing, dict):
        formula = healing.get("diceString") or _dice_formula(healing)
        if formula:
            return [{"formula": formula, "raw": {"healing": healing}}]
    return []


def _dice_formula(dice: dict[str, Any]) -> str:
    count = dice.get("diceCount")
    value = dice.get("diceValue")
    fixed = dice.get("fixedValue")
    if count and value:
        formula = f"{count}d{value}"
        if fixed:
            formula += f"+{fixed}"
        return formula
    if fixed:
        return str(fixed)
    return ""


def _resource_costs(obj: dict[str, Any], source_kind: str) -> list[dict[str, Any]]:
    res = _limited_use_resource(obj, source_kind)
    if not res:
        return []
    return [{
        "resource_id": res["id"],
        "amount": 1,
        "consume_on": "use",
        "optional": False,
    }]


def _spell_components(definition: dict[str, Any]) -> dict[str, Any]:
    components = definition.get("components") or []
    if isinstance(components, list):
        component_ids = set(components)
    else:
        component_ids = set()
    return {
        "verbal": 1 in component_ids or bool(definition.get("verbal")),
        "somatic": 2 in component_ids or bool(definition.get("somatic")),
        "material": 3 in component_ids or bool(definition.get("material")),
        "material_text": definition.get("componentsDescription") or "",
        "material_consumed": bool(definition.get("requiresMaterialConsumption")),
    }


def _duration(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    dtype = str(raw.get("durationType") or "").lower()
    unit = raw.get("durationUnit")
    amount = raw.get("durationInterval")
    if "concentration" in dtype:
        kind = "minutes" if str(unit).lower().startswith("minute") else "special"
    elif "instant" in dtype:
        kind = "instant"
    else:
        kind = {
            "round": "rounds",
            "minute": "minutes",
            "hour": "hours",
            "day": "days",
        }.get(str(unit or "").lower(), "special")
    return {
        "kind": kind,
        "amount": amount,
        "unit": unit or "",
        "concentration": "concentration" in dtype,
        "text": dtype,
    }


def _description(obj: dict[str, Any]) -> dict[str, Any]:
    snippet = obj.get("snippet") or obj.get("shortDescription") or ""
    full_text = obj.get("description") or ""
    return {
        "snippet": snippet,
        "summary": snippet,
        "full_text": full_text,
        "source_url": obj.get("moreDetailsUrl") or "",
        "prompt_safe": False,
    }


def _item_armor_class(definition: dict[str, Any]) -> dict[str, Any] | None:
    if definition.get("armorClass") is None:
        return None
    return {
        "value": definition.get("armorClass") or 0,
        "calculation": definition.get("baseArmorName") or definition.get("name") or "",
    }


def _value_gp(definition: dict[str, Any]) -> float:
    cost = definition.get("cost")
    if isinstance(cost, dict):
        return float(cost.get("quantity") or 0)
    try:
        return float(cost or 0)
    except (TypeError, ValueError):
        return 0


def _source_refs(source_type: str, obj: dict[str, Any]) -> list[dict[str, Any]]:
    source_id = obj.get("id") or obj.get("definitionId") or obj.get("entityId")
    source_name = obj.get("name") or (obj.get("definition") or {}).get("name") or ""
    refs = []
    if source_id is not None or source_name:
        refs.append({
            "source_type": source_type,
            "source_id": str(source_id or ""),
            "source_name": source_name,
            "ddb_definition_id": obj.get("definitionId") or obj.get("id"),
            "url": obj.get("moreDetailsUrl") or "",
        })
    for source in obj.get("sources") or []:
        if isinstance(source, dict):
            refs.append({
                "source_type": "ddb_source",
                "source_id": str(source.get("sourceId") or source.get("id") or ""),
                "source_name": source.get("sourceName") or source.get("name") or "",
                "book": source.get("description") or "",
                "page": source.get("pageNumber"),
                "raw": source,
            })
    return refs


def _modifier_source_ref(bucket: str, mod: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_type": f"modifier:{bucket}",
        "source_id": str(mod.get("id") or ""),
        "source_name": mod.get("friendlySubtypeName") or mod.get("subType") or "",
        "ddb_definition_id": mod.get("entityId"),
        "raw": {
            "entityTypeId": mod.get("entityTypeId"),
            "componentId": mod.get("componentId"),
            "componentTypeId": mod.get("componentTypeId"),
        },
    }


def _iter_bucketed(raw: Any) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(raw, dict):
        return out
    for bucket, items in raw.items():
        for item in items or []:
            if isinstance(item, dict):
                out.append((str(bucket), item))
    return out


def _raw_without_text(obj: dict[str, Any]) -> dict[str, Any]:
    text_keys = {"description", "snippet", "shortDescription"}
    return {k: v for k, v in obj.items() if k not in text_keys}


def _size_name(char: dict[str, Any]) -> str:
    race = char.get("race") or {}
    size = race.get("size")
    if isinstance(size, str):
        return size
    size_id = race.get("sizeId")
    return {1: "Tiny", 2: "Small", 3: "Medium", 4: "Large"}.get(size_id, "")


def _role(identity: dict[str, Any]) -> str:
    classes = identity.get("classes") or []
    class_text = "/".join(
        f"{entry.get('name')} {entry.get('level')}"
        for entry in classes
        if entry.get("name")
    )
    bits = [
        identity.get("species", ""),
        class_text,
        identity.get("background", ""),
    ]
    return ", ".join(bit for bit in bits if bit)


def _label_value(label: str, value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{label}: {value}"


def _first_string(*values: Any) -> str:
    return "\n\n".join(str(v).strip() for v in values if isinstance(v, str) and v.strip())


def _stable_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _clean(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_clean(item) for item in value if item is not None]
    return value
