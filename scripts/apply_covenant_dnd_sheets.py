#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SYNTHETIC_SOURCE_TYPE = "synthetic_ayoa_covenant_thrones"
RULESET_ID = "dnd5e_basic"
ABILITY_ORDER = ("str", "dex", "con", "int", "wis", "cha")
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
HIT_DIE_BY_CLASS = {
    "Bard": 8,
    "Cleric": 8,
    "Druid": 8,
    "Fighter": 10,
    "Monk": 8,
    "Paladin": 10,
    "Ranger": 10,
    "Rogue": 8,
    "Sorcerer": 6,
    "Warlock": 8,
    "Wizard": 6,
    "Expert": 8,
    "Aristocrat": 8,
}


@dataclass(frozen=True)
class Profile:
    species: str
    background: str
    classes: tuple[tuple[str, str, int], ...]
    abilities: dict[str, int]
    saving_throws: tuple[str, ...]
    skill_multipliers: dict[str, int]
    armor_class: int
    hit_points: int
    movement: dict[str, int] = field(default_factory=lambda: {"walk": 30})
    languages: tuple[str, ...] = ("Common",)
    resistances: tuple[str, ...] = ()
    condition_immunities: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    resources: tuple[str, ...] = ("Resolve",)
    spellcasting_ability: str = ""
    spells: tuple[tuple[str, int], ...] = ()


PROFILES: dict[str, Profile] = {
    "player_garvey": Profile(
        species="Human",
        background="Displaced Noble",
        classes=(("Expert", "Garvey Scion", 3),),
        abilities={"str": 11, "dex": 12, "con": 12, "int": 14, "wis": 13, "cha": 14},
        saving_throws=("int", "cha"),
        skill_multipliers={
            "history": 1,
            "insight": 1,
            "investigation": 1,
            "persuasion": 1,
        },
        armor_class=12,
        hit_points=24,
        languages=("Common",),
        features=("Unexpected Heir", "Compressed Legal Education"),
        resources=("Garvey composure",),
    ),
    "ashara_vel_kothren": Profile(
        species="Demon",
        background="Seat Heiress Duelist",
        classes=(("Fighter", "Battle Master", 8),),
        abilities={"str": 18, "dex": 14, "con": 16, "int": 12, "wis": 13, "cha": 15},
        saving_throws=("str", "con"),
        skill_multipliers={
            "athletics": 2,
            "intimidation": 1,
            "insight": 1,
            "perception": 1,
        },
        armor_class=17,
        hit_points=76,
        languages=("Common", "Abyssal"),
        resistances=("fire",),
        features=("Top-ranked duelist", "Demonic heat", "Controlled tail"),
        resources=("Superiority dice", "Composure"),
    ),
    "ysolde_thornmantle": Profile(
        species="Dragon",
        background="Dragon Seat Heiress",
        classes=(("Sorcerer", "Draconic Bloodline", 7),),
        abilities={"str": 13, "dex": 14, "con": 18, "int": 14, "wis": 15, "cha": 16},
        saving_throws=("con", "cha"),
        skill_multipliers={
            "history": 1,
            "insight": 1,
            "intimidation": 1,
            "perception": 1,
        },
        armor_class=16,
        hit_points=68,
        movement={"walk": 30, "fly": 40},
        languages=("Common", "Draconic"),
        resistances=("cold",),
        features=("Dragon poise", "Cold presence", "Court stillness"),
        resources=("Draconic composure",),
        spellcasting_ability="cha",
        spells=(("Frost Ward", 1), ("Glacial Rebuke", 2), ("Winged Passage", 3)),
    ),
    "caelindra_vaeyn": Profile(
        species="Elf",
        background="Amber Court Diplomat",
        classes=(("Bard", "Eloquence", 6),),
        abilities={"str": 8, "dex": 16, "con": 12, "int": 16, "wis": 13, "cha": 17},
        saving_throws=("dex", "cha"),
        skill_multipliers={
            "deception": 1,
            "history": 1,
            "insight": 1,
            "perception": 1,
            "persuasion": 2,
        },
        armor_class=14,
        hit_points=44,
        languages=("Common", "Elvish", "Sylvan"),
        features=("Amber Court polish", "Social detachment", "Elven memory"),
        resources=("Court poise", "Bardic inspiration"),
        spellcasting_ability="cha",
        spells=(("Cutting Remark", 0), ("Courtly Veil", 1), ("Compel Courtesy", 2)),
    ),
    "seraphel_dawnquill": Profile(
        species="Angel",
        background="Cursed Dawnquill Scion",
        classes=(("Cleric", "Light", 6),),
        abilities={"str": 9, "dex": 14, "con": 12, "int": 13, "wis": 17, "cha": 16},
        saving_throws=("wis", "cha"),
        skill_multipliers={
            "insight": 2,
            "performance": 1,
            "persuasion": 1,
            "religion": 1,
        },
        armor_class=14,
        hit_points=42,
        movement={"walk": 30, "fly": 30},
        languages=("Common", "Celestial"),
        resistances=("radiant",),
        features=("Radiant presence", "Verse-bound speech", "Expressive wings"),
        resources=("Radiance",),
        spellcasting_ability="wis",
        spells=(("Guiding Motif", 0), ("Radiant Sign", 1), ("Winged Blessing", 2)),
    ),
    "thessaly_morrow": Profile(
        species="Fae",
        background="Bound Court Observer",
        classes=(("Warlock", "Archfey", 6),),
        abilities={"str": 7, "dex": 17, "con": 11, "int": 14, "wis": 15, "cha": 18},
        saving_throws=("wis", "cha"),
        skill_multipliers={
            "deception": 2,
            "insight": 1,
            "perception": 1,
            "performance": 1,
            "stealth": 1,
        },
        armor_class=15,
        hit_points=38,
        languages=("Common", "Sylvan"),
        condition_immunities=("charmed",),
        features=("Fae timing", "Contract sense", "Unsettling smile"),
        resources=("Glamour", "Contract leverage"),
        spellcasting_ability="cha",
        spells=(("Glamour Thread", 0), ("Borrowed Face", 1), ("Unnatural Pause", 2)),
    ),
    "lysara_vane": Profile(
        species="Half-demon half-elf",
        background="Exceptional Case Observer",
        classes=(("Rogue", "Inquisitive", 5),),
        abilities={"str": 10, "dex": 16, "con": 13, "int": 15, "wis": 15, "cha": 12},
        saving_throws=("dex", "int"),
        skill_multipliers={
            "acrobatics": 1,
            "deception": 1,
            "insight": 1,
            "perception": 2,
            "stealth": 1,
        },
        armor_class=15,
        hit_points=38,
        languages=("Common", "Abyssal", "Elvish"),
        resistances=("fire",),
        features=("Article Nineteen exception", "Categoryless read", "Quiet anger"),
        resources=("Patience", "Edge"),
    ),
    "rashid_vel_amara": Profile(
        species="Beastkin",
        background="Seat Heir Duelist",
        classes=(("Monk", "Kensei", 7),),
        abilities={"str": 14, "dex": 18, "con": 14, "int": 15, "wis": 16, "cha": 14},
        saving_throws=("str", "dex"),
        skill_multipliers={
            "acrobatics": 1,
            "athletics": 1,
            "insight": 1,
            "perception": 2,
            "persuasion": 1,
        },
        armor_class=17,
        hit_points=59,
        movement={"walk": 45},
        languages=("Common", "Beastkin cant"),
        features=("Martial ranking", "Predatory senses", "Impeccable presentation"),
        resources=("Focus", "Ki"),
    ),
    "aldric_verantus": Profile(
        species="Angel",
        background="Angel Seat Heir",
        classes=(("Paladin", "Devotion", 6),),
        abilities={"str": 15, "dex": 12, "con": 14, "int": 13, "wis": 16, "cha": 17},
        saving_throws=("wis", "cha"),
        skill_multipliers={
            "insight": 1,
            "persuasion": 1,
            "religion": 1,
        },
        armor_class=16,
        hit_points=54,
        movement={"walk": 30, "fly": 30},
        languages=("Common", "Celestial"),
        resistances=("radiant",),
        features=("Angel-heir bearing", "Formal discipline", "Manifest wings"),
        resources=("Lay on hands", "Conviction"),
        spellcasting_ability="cha",
        spells=(("Oathmark", 1), ("Sanctified Rebuke", 1), ("Zone of Candor", 2)),
    ),
    "prof_kira_vel_shaan": Profile(
        species="Demon",
        background="Practical Rhetoric Professor",
        classes=(("Bard", "Eloquence", 8),),
        abilities={"str": 10, "dex": 14, "con": 13, "int": 16, "wis": 15, "cha": 18},
        saving_throws=("dex", "cha"),
        skill_multipliers={
            "deception": 1,
            "insight": 2,
            "intimidation": 1,
            "persuasion": 2,
        },
        armor_class=14,
        hit_points=52,
        languages=("Common", "Abyssal"),
        resistances=("fire",),
        features=("Practical rhetoric", "Handler discipline", "Tail control"),
        resources=("Instructional pressure", "Bardic inspiration"),
        spellcasting_ability="cha",
        spells=(("Measured Phrase", 0), ("Disarming Premise", 1), ("Commanding Thesis", 2)),
    ),
    "prof_elara_windwhisper": Profile(
        species="Elf",
        background="Covenant Law Professor",
        classes=(("Wizard", "Divination", 10),),
        abilities={"str": 7, "dex": 13, "con": 10, "int": 20, "wis": 17, "cha": 12},
        saving_throws=("int", "wis"),
        skill_multipliers={
            "arcana": 1,
            "history": 2,
            "insight": 1,
            "investigation": 2,
            "perception": 1,
        },
        armor_class=13,
        hit_points=55,
        languages=("Common", "Elvish", "Draconic", "Celestial"),
        features=("Living precedent", "Amber Court channel", "Legal silence"),
        resources=("Portent", "Institutional memory"),
        spellcasting_ability="int",
        spells=(("Precedent Sense", 0), ("Arcane Seal", 1), ("Witness Thread", 2), ("Memory Vault", 4)),
    ),
    "prof_vex_thorn": Profile(
        species="Demon",
        background="War Historian",
        classes=(("Fighter", "Champion", 8),),
        abilities={"str": 17, "dex": 12, "con": 16, "int": 16, "wis": 13, "cha": 14},
        saving_throws=("str", "con"),
        skill_multipliers={
            "athletics": 1,
            "history": 2,
            "insight": 1,
            "intimidation": 1,
        },
        armor_class=15,
        hit_points=72,
        languages=("Common", "Abyssal"),
        resistances=("fire",),
        features=("Lecture-hall voice", "Comparative war record", "Reformist contacts"),
        resources=("Second wind", "Argumentative force"),
    ),
    "prof_gareth_stone": Profile(
        species="Human",
        background="Frightened Historian",
        classes=(("Expert", "Sage", 5),),
        abilities={"str": 6, "dex": 9, "con": 9, "int": 18, "wis": 16, "cha": 11},
        saving_throws=("int", "wis"),
        skill_multipliers={
            "history": 2,
            "insight": 1,
            "investigation": 2,
            "perception": 1,
        },
        armor_class=10,
        hit_points=24,
        languages=("Common", "Elvish"),
        features=("Forbidden documents", "Plausible deniability", "Academic caution"),
        resources=("Hidden archive",),
    ),
    "chancellor_ashworth": Profile(
        species="Angel",
        background="Academy Chancellor",
        classes=(("Cleric", "Order", 10),),
        abilities={"str": 9, "dex": 10, "con": 14, "int": 18, "wis": 19, "cha": 17},
        saving_throws=("wis", "cha"),
        skill_multipliers={
            "history": 1,
            "insight": 2,
            "persuasion": 1,
            "religion": 1,
        },
        armor_class=14,
        hit_points=70,
        movement={"walk": 25, "fly": 30},
        languages=("Common", "Celestial", "Elvish"),
        resistances=("radiant",),
        features=("Chancellor's authority", "Institution-first instincts", "Transparent wings"),
        resources=("Office authority", "Radiant reserve"),
        spellcasting_ability="wis",
        spells=(("Administrative Seal", 0), ("Command Silence", 1), ("Institutional Ward", 3)),
    ),
    "lord_verantus": Profile(
        species="Angel",
        background="Council Seat Holder",
        classes=(("Paladin", "Crown", 12),),
        abilities={"str": 16, "dex": 10, "con": 15, "int": 17, "wis": 18, "cha": 20},
        saving_throws=("wis", "cha"),
        skill_multipliers={
            "deception": 1,
            "insight": 2,
            "intimidation": 1,
            "persuasion": 2,
            "religion": 1,
        },
        armor_class=18,
        hit_points=100,
        movement={"walk": 30, "fly": 40},
        languages=("Common", "Celestial", "Elvish"),
        resistances=("radiant",),
        features=("Council authority", "Ceremonial wings", "Conspiracy discipline"),
        resources=("Command presence", "Lay on hands"),
        spellcasting_ability="cha",
        spells=(("Edict", 0), ("Compelled Oath", 2), ("Seal Testimony", 3)),
    ),
    "lady_ashira_vel_kothren": Profile(
        species="Demon",
        background="Demon Matriarch",
        classes=(("Warlock", "Great Old One", 12),),
        abilities={"str": 7, "dex": 9, "con": 13, "int": 18, "wis": 19, "cha": 20},
        saving_throws=("wis", "cha"),
        skill_multipliers={
            "deception": 1,
            "history": 2,
            "insight": 2,
            "intimidation": 1,
        },
        armor_class=13,
        hit_points=66,
        movement={"walk": 20},
        languages=("Common", "Abyssal", "Infernal"),
        resistances=("fire",),
        features=("Old conspiracy memory", "Matriarchal gravity", "Weathered horns"),
        resources=("Family authority", "Old magic"),
        spellcasting_ability="cha",
        spells=(("Withering Glance", 0), ("Memory Lock", 2), ("Matriarch's Geas", 5)),
    ),
    "toxicia_vaeyn": Profile(
        species="Elf",
        background="Silence Faction Leader",
        classes=(("Rogue", "Mastermind", 12),),
        abilities={"str": 8, "dex": 18, "con": 13, "int": 20, "wis": 17, "cha": 18},
        saving_throws=("dex", "int"),
        skill_multipliers={
            "deception": 2,
            "history": 1,
            "insight": 2,
            "investigation": 1,
            "persuasion": 1,
            "stealth": 1,
        },
        armor_class=16,
        hit_points=78,
        languages=("Common", "Elvish", "Sylvan", "Draconic"),
        features=("Silence network", "Amber Court command", "Patient secrecy"),
        resources=("Court leverage", "Operational silence"),
    ),
    "lady_coldpeak": Profile(
        species="Dragon",
        background="Dragon Court Rival",
        classes=(("Sorcerer", "Draconic Bloodline", 12),),
        abilities={"str": 17, "dex": 12, "con": 20, "int": 16, "wis": 18, "cha": 19},
        saving_throws=("con", "cha"),
        skill_multipliers={
            "deception": 1,
            "insight": 2,
            "intimidation": 2,
            "perception": 1,
        },
        armor_class=18,
        hit_points=110,
        movement={"walk": 30, "fly": 50},
        languages=("Common", "Draconic"),
        resistances=("cold",),
        features=("Dragon court pressure", "Cold plotting", "Humanoid dragon form"),
        resources=("Draconic presence", "Cold reserve"),
        spellcasting_ability="cha",
        spells=(("Rime Lash", 0), ("Ice Mirror", 2), ("Court of Winter", 4), ("Dominating Frost", 5)),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply synthetic D&D 5e sheets to Covenant of Thrones checkpoints."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--overwrite-non-synthetic",
        action="store_true",
        help="Replace existing non-synthetic sheets. Default preserves them.",
    )
    parser.add_argument(
        "--no-enable-settings",
        action="store_true",
        help="Do not switch the checkpoint settings to D&D mode.",
    )
    args = parser.parse_args()

    for path in args.paths:
        result = apply_file(
            path,
            preserve_non_synthetic=not args.overwrite_non_synthetic,
            enable_settings=not args.no_enable_settings,
        )
        print(
            f"{path}: updated={result['updated']} preserved={result['preserved']} "
            f"settings={result['settings']}"
        )
    return 0


def apply_file(
    path: Path,
    *,
    preserve_non_synthetic: bool = True,
    enable_settings: bool = True,
) -> dict[str, int | bool]:
    data = json.loads(path.read_text(encoding="utf-8"))
    characters = data.get("characters")
    if not isinstance(characters, list):
        raise ValueError(f"{path} does not look like an Ayoa checkpoint")

    missing = {
        str(char.get("character_id") or "")
        for char in characters
        if char.get("character_id") not in PROFILES
    }
    missing.discard("")
    if missing:
        raise ValueError(
            f"{path} has Covenant characters without synthetic profiles: "
            + ", ".join(sorted(missing))
        )

    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    updated = 0
    preserved = 0
    for char in characters:
        cid = str(char.get("character_id") or "")
        profile = PROFILES.get(cid)
        if profile is None:
            continue

        mechanics = char.get("mechanics") or {}
        source_type = (
            ((mechanics.get("dnd5e_sheet") or {}).get("source") or {}).get("type")
            if isinstance(mechanics, dict)
            else ""
        )
        if (
            preserve_non_synthetic
            and mechanics
            and source_type != SYNTHETIC_SOURCE_TYPE
        ):
            preserved += 1
            continue

        char["mechanics"] = build_mechanics(
            character_id=cid,
            name=str(char.get("name") or cid),
            role=str((char.get("public_sheet") or {}).get("role") or ""),
            profile=profile,
            generated_at=now,
        )
        updated += 1

    settings_enabled = False
    if enable_settings:
        session = data.setdefault("session", {})
        config = session.setdefault("config", {})
        settings = config.setdefault("settings", {})
        settings["ruleset_id"] = RULESET_ID
        settings.setdefault("player_roll_mode", "auto")
        settings_enabled = True

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "updated": updated,
        "preserved": preserved,
        "settings": settings_enabled,
    }


def build_mechanics(
    *,
    character_id: str,
    name: str,
    role: str,
    profile: Profile,
    generated_at: str,
) -> dict[str, Any]:
    snapshot = build_snapshot(
        character_id=character_id,
        name=name,
        role=role,
        profile=profile,
        generated_at=generated_at,
    )
    statblock = snapshot["statblock"]
    hp = statblock["defenses"]["hit_points"]
    resources = {
        res["id"]: {
            "name": res["name"],
            "kind": res["kind"],
            "current": res["current"],
            "max": res["max"],
            "reset": (res.get("reset") or {}).get("type", ""),
        }
        for res in statblock.get("resources", [])
    }
    return {
        "ruleset_id": RULESET_ID,
        "ability_scores": {
            ability: statblock["ability_scores"][ability]["score"]
            for ability in ABILITY_ORDER
        },
        "proficiency_bonus": statblock["proficiency_bonus"],
        "skill_proficiencies": [
            skill for skill, mult in profile.skill_multipliers.items() if mult > 0
        ],
        "saving_throw_proficiencies": list(profile.saving_throws),
        "armor_class": profile.armor_class,
        "hit_points": {
            "current": hp["current"],
            "max": hp["max"],
            "temporary": hp["temporary"],
        },
        "conditions": [],
        "resources": resources,
        "raw": {
            "source": snapshot["source"],
            "import_report": {
                "synthetic": True,
                "warnings": [
                    "Synthetic Covenant of Thrones profile; replace with a real sheet when available."
                ],
            },
        },
        "dnd5e_sheet": snapshot,
    }


def build_snapshot(
    *,
    character_id: str,
    name: str,
    role: str,
    profile: Profile,
    generated_at: str,
) -> dict[str, Any]:
    level = total_level(profile)
    pb = proficiency_bonus(level)
    abilities = {
        ability: {
            "score": profile.abilities[ability],
            "modifier": ability_modifier(profile.abilities[ability]),
        }
        for ability in ABILITY_ORDER
    }
    return {
        "ruleset_id": RULESET_ID,
        "source": {
            "type": SYNTHETIC_SOURCE_TYPE,
            "character_id": character_id,
            "generated_at": generated_at,
        },
        "identity": {
            "name": name,
            "species": profile.species,
            "background": profile.background,
            "role": role,
            "classes": class_entries(profile),
        },
        "statblock": {
            "ability_scores": abilities,
            "proficiency_bonus": pb,
            "skills": skill_entries(profile, abilities, pb),
            "saves": save_entries(profile, abilities, pb),
            "defenses": defense_block(profile, abilities),
            "actions": action_entries(profile, abilities, pb),
            "spellcasting": spellcasting_block(profile, abilities, pb),
            "inventory": {"items": [], "currency": {}},
            "features": feature_entries(profile),
            "resources": resource_entries(profile),
            "proficiencies": proficiency_entries(profile),
            "languages": list(profile.languages),
        },
        "import_report": {
            "synthetic": True,
            "warnings": [
                "Hand-authored synthetic profile for Covenant of Thrones D&D arbitration."
            ],
        },
    }


def class_entries(profile: Profile) -> list[dict[str, Any]]:
    entries = []
    for class_name, subclass, level in profile.classes:
        class_id = slug(f"{subclass} {class_name}") if subclass else slug(class_name)
        entries.append({
            "id": class_id,
            "name": class_name,
            "subclass": subclass,
            "level": level,
            "hit_die": HIT_DIE_BY_CLASS.get(class_name, 8),
            "source_refs": [{"source_type": "synthetic", "source_id": class_id}],
        })
    return entries


def skill_entries(
    profile: Profile,
    abilities: dict[str, dict[str, int]],
    pb: int,
) -> dict[str, dict[str, Any]]:
    out = {}
    for skill, ability in SKILL_ABILITIES.items():
        mult = int(profile.skill_multipliers.get(skill, 0))
        value = abilities[ability]["modifier"] + (pb * mult)
        out[skill] = {
            "ability": ability,
            "value": value,
            "proficiency_multiplier": mult,
            "passive": 10 + value,
        }
    return out


def save_entries(
    profile: Profile,
    abilities: dict[str, dict[str, int]],
    pb: int,
) -> dict[str, dict[str, Any]]:
    out = {}
    for ability in ABILITY_ORDER:
        mult = 1 if ability in profile.saving_throws else 0
        value = abilities[ability]["modifier"] + (pb * mult)
        out[ability] = {
            "ability": ability,
            "value": value,
            "proficiency_multiplier": mult,
        }
    return out


def defense_block(profile: Profile, abilities: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        "armor_class": {"value": profile.armor_class},
        "hit_points": {
            "current": profile.hit_points,
            "max": profile.hit_points,
            "temporary": 0,
        },
        "initiative": {"value": abilities["dex"]["modifier"]},
        "movement": {
            kind: {"value": value, "unit": "ft"}
            for kind, value in profile.movement.items()
        },
        "conditions": [],
        "exhaustion_level": 0,
        "damage_resistances": [
            {"id": slug(name), "name": name} for name in profile.resistances
        ],
        "damage_immunities": [],
        "damage_vulnerabilities": [],
        "condition_immunities": [
            {"id": slug(name), "name": name} for name in profile.condition_immunities
        ],
    }


def action_entries(
    profile: Profile,
    abilities: dict[str, dict[str, int]],
    pb: int,
) -> list[dict[str, Any]]:
    attack_ability = "dex" if abilities["dex"]["modifier"] >= abilities["str"]["modifier"] else "str"
    attack_mod = abilities[attack_ability]["modifier"]
    actions = [{
        "id": "measured-strike",
        "name": "Measured Strike",
        "kind": "weapon_attack",
        "attack": {"bonus": attack_mod + pb, "ability": attack_ability},
        "damage": [{
            "formula": f"1d8+{attack_mod}" if attack_mod >= 0 else f"1d8{attack_mod}",
            "damage_type": "bludgeoning",
        }],
    }]
    if profile.spellcasting_ability:
        spell_mod = abilities[profile.spellcasting_ability]["modifier"]
        actions.append({
            "id": "arcane-pressure",
            "name": "Arcane Pressure",
            "kind": "spell_attack",
            "attack": {
                "bonus": spell_mod + pb,
                "ability": profile.spellcasting_ability,
            },
            "damage": [{"formula": "1d10", "damage_type": "force"}],
        })
    return actions


def spellcasting_block(
    profile: Profile,
    abilities: dict[str, dict[str, int]],
    pb: int,
) -> dict[str, Any]:
    if not profile.spellcasting_ability:
        return {"profiles": [], "slots": {}, "spells": []}
    mod = abilities[profile.spellcasting_ability]["modifier"]
    level = total_level(profile)
    max_slot = min(5, max(1, (level + 1) // 2))
    return {
        "profiles": [{
            "name": "Synthetic spellcasting",
            "ability": profile.spellcasting_ability,
            "spell_attack_bonus": mod + pb,
            "spell_save_dc": 8 + mod + pb,
        }],
        "slots": {
            str(slot): {"current": max(1, 4 - slot), "max": max(1, 4 - slot)}
            for slot in range(1, max_slot + 1)
        },
        "spells": [
            {"id": slug(name), "name": name, "level": level}
            for name, level in profile.spells
        ],
    }


def feature_entries(profile: Profile) -> list[dict[str, Any]]:
    return [
        {"id": slug(name), "name": name, "type": "synthetic"}
        for name in profile.features
    ]


def resource_entries(profile: Profile) -> list[dict[str, Any]]:
    return [
        {
            "id": slug(name),
            "name": name,
            "kind": "synthetic",
            "current": 1,
            "max": 1,
            "reset": {"type": "long_rest"},
        }
        for name in profile.resources
    ]


def proficiency_entries(profile: Profile) -> dict[str, list[str]]:
    return {
        "armor": ["light armor"] if profile.armor_class <= 14 else ["light armor", "medium armor"],
        "weapons": ["simple weapons"],
        "tools": [],
        "other": [
            f"{skill.title()} expertise" if mult > 1 else f"{skill.title()} proficiency"
            for skill, mult in profile.skill_multipliers.items()
            if mult > 0
        ],
    }


def total_level(profile: Profile) -> int:
    return sum(level for _, _, level in profile.classes)


def proficiency_bonus(level: int) -> int:
    if level >= 17:
        return 6
    if level >= 13:
        return 5
    if level >= 9:
        return 4
    if level >= 5:
        return 3
    return 2


def ability_modifier(score: int) -> int:
    return (score - 10) // 2


def slug(value: str) -> str:
    return (
        value.strip().lower()
        .replace("'", "")
        .replace("/", "-")
        .replace(" ", "-")
        .replace("_", "-")
    )


if __name__ == "__main__":
    raise SystemExit(main())
