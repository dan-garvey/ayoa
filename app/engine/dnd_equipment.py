from __future__ import annotations

import re
from dataclasses import dataclass

from app.engine import dnd_inventory, mechanics


@dataclass(frozen=True)
class RuntimeWeaponProfile:
    name: str
    expression: str
    damage_type: str
    ability: str = "str"
    weapon_class: str = "simple"


_RUNTIME_WEAPON_PROFILES: tuple[tuple[str, RuntimeWeaponProfile], ...] = (
    ("greatsword", RuntimeWeaponProfile(
        "Greatsword",
        "2d6",
        "slashing",
        weapon_class="martial",
    )),
    ("longsword", RuntimeWeaponProfile(
        "Longsword",
        "1d8",
        "slashing",
        weapon_class="martial",
    )),
    ("shortsword", RuntimeWeaponProfile(
        "Shortsword",
        "1d6",
        "piercing",
        "dex",
        "martial",
    )),
    ("rapier", RuntimeWeaponProfile(
        "Rapier",
        "1d8",
        "piercing",
        "dex",
        "martial",
    )),
    ("scimitar", RuntimeWeaponProfile(
        "Scimitar",
        "1d6",
        "slashing",
        "dex",
        "martial",
    )),
    ("longbow", RuntimeWeaponProfile(
        "Longbow",
        "1d8",
        "piercing",
        "dex",
        "martial",
    )),
    ("shortbow", RuntimeWeaponProfile("Shortbow", "1d6", "piercing", "dex")),
    ("handaxe", RuntimeWeaponProfile("Handaxe", "1d6", "slashing")),
    ("spear", RuntimeWeaponProfile("Spear", "1d6", "piercing")),
    ("mace", RuntimeWeaponProfile("Mace", "1d6", "bludgeoning")),
    ("quarterstaff", RuntimeWeaponProfile("Quarterstaff", "1d6", "bludgeoning")),
    ("dagger", RuntimeWeaponProfile("Dagger", "1d4", "piercing", "dex")),
    ("club", RuntimeWeaponProfile("Club", "1d4", "bludgeoning")),
)


def runtime_weapon_actions(character: object) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for item in (dnd_inventory.inventory_view(character).get("items") or []):
        if not isinstance(item, dict) or not item.get("source_offer_id"):
            continue
        profile = runtime_weapon_profile(item)
        if profile is None:
            continue
        name = str(item.get("name") or profile.name).strip() or profile.name
        item_key = (
            item.get("source_item_id")
            or item.get("id")
            or item.get("item_id")
            or name
        )
        actions.append({
            "id": f"runtime_item_attack_{_slug(item_key)}",
            "name": name,
            "kind": "attack",
            "attack": {
                "ability": profile.ability,
                "bonus": runtime_weapon_attack_bonus(character, profile),
                "damage": f"{profile.expression} {profile.damage_type}",
            },
            "damage": [{
                "formula": profile.expression,
                "damage_type": profile.damage_type,
            }],
            "runtime_inventory_item_id": str(
                item.get("id") or item.get("item_id") or ""
            ),
        })
    return actions


def runtime_weapon_profile(
    item: dict[str, object],
) -> RuntimeWeaponProfile | None:
    name = _normalize_action_text(item.get("name") or item.get("id") or "")
    kind = _normalize_action_text(item.get("kind") or "")
    for needle, profile in _RUNTIME_WEAPON_PROFILES:
        if needle in name:
            return profile
    if "weapon" not in kind:
        return None
    return RuntimeWeaponProfile("Improvised Weapon", "1d4", "bludgeoning")


def runtime_weapon_attack_bonus(
    character: object,
    profile: RuntimeWeaponProfile,
) -> int:
    mechanics_state = getattr(character, "mechanics", None) or {}
    scores = mechanics_state.get("ability_scores") or {}
    try:
        score = int(scores.get(profile.ability, 10) or 10)
    except (TypeError, ValueError):
        score = 10
    bonus = mechanics.ability_modifier(score)
    if runtime_weapon_is_proficient(mechanics_state, profile):
        try:
            bonus += int(mechanics_state.get("proficiency_bonus", 0) or 0)
        except (TypeError, ValueError):
            pass
    return bonus


def runtime_weapon_is_proficient(
    mechanics_state: dict[str, object],
    profile: RuntimeWeaponProfile,
) -> bool:
    statblock = (mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {}
    proficiencies = statblock.get("proficiencies") or {}
    weapons = [
        _normalize_action_text(value)
        for value in proficiencies.get("weapons", []) or []
    ]
    if any(_normalize_action_text(profile.name) in weapon for weapon in weapons):
        return True
    if profile.weapon_class == "martial" and "martial weapons" in weapons:
        return True
    return profile.weapon_class == "simple" and (
        "simple weapons" in weapons or "martial weapons" in weapons
    )


def _normalize_action_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
