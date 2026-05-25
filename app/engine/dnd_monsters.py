from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from app.schemas.characters import (
    CharacterAgentTier,
    CharacterDescriptions,
    CharacterRecord,
    CharacterVisuals,
    PublicSheet,
)
from app.schemas.dnd_monsters import (
    DndCombatantSpawn,
    DndMonsterAction,
    DndMonsterStatBlock,
)

logger = logging.getLogger(__name__)


class StatblockResolutionError(ValueError):
    """Raised when an authoritative statblock resolver must block combat."""


StatblockOverrideProvider = Callable[
    [DndCombatantSpawn],
    DndMonsterStatBlock | Mapping[str, Any] | None,
]

_STATBLOCK_OVERRIDE_PROVIDERS: list[StatblockOverrideProvider] = []


def register_statblock_override_provider(
    provider: StatblockOverrideProvider,
) -> None:
    """Register a D&D statblock correction provider.

    Providers are adapter-local hooks for plugins or tests that can supply a
    more authoritative stat block than the router fallback. Later registrations
    win, so a session/plugin-specific provider can override a broader catalog.
    """

    if provider not in _STATBLOCK_OVERRIDE_PROVIDERS:
        _STATBLOCK_OVERRIDE_PROVIDERS.append(provider)


def clear_statblock_override_providers() -> None:
    _STATBLOCK_OVERRIDE_PROVIDERS.clear()


def resolve_statblock(spawn: DndCombatantSpawn) -> DndMonsterStatBlock:
    """Return the best stat block for a router combatant spawn."""

    required_ref = str(getattr(spawn, "statblock_ref", "") or "").strip()
    for provider in reversed(_STATBLOCK_OVERRIDE_PROVIDERS):
        try:
            candidate = provider(spawn)
        except StatblockResolutionError:
            raise
        except Exception:
            logger.exception(
                "D&D statblock override provider failed for %s",
                spawn.monster_key or spawn.character_id,
            )
            continue
        if candidate is None:
            continue
        try:
            return DndMonsterStatBlock.model_validate(candidate)
        except Exception:
            logger.exception(
                "D&D statblock override provider returned invalid data for %s",
                required_ref or spawn.monster_key or spawn.character_id,
            )
            if required_ref:
                raise StatblockResolutionError(
                    "D&D combatant spawn statblock_ref could not be "
                    f"resolved safely: {required_ref}"
                )
    if required_ref:
        raise StatblockResolutionError(
            "D&D combatant spawn statblock_ref is not available in a "
            f"reviewed runtime catalog: {required_ref}"
        )
    if spawn.statblock is None:
        raise StatblockResolutionError(
            "D&D combatant spawn requires either a resolved statblock_ref "
            "or an inline fallback statblock."
        )
    return spawn.statblock


def character_from_combatant_spawn(
    spawn: DndCombatantSpawn,
    *,
    default_location: str = "",
) -> CharacterRecord:
    """Build a non-playable combat character from a D&D router spawn."""

    statblock = resolve_statblock(spawn)
    location = spawn.location or default_location
    role = " ".join(
        part for part in (statblock.size, statblock.creature_type) if part
    ).strip()
    visible_description = spawn.description or role or statblock.creature_type
    return CharacterRecord(
        character_id=spawn.character_id,
        name=spawn.name or spawn.monster_key or spawn.character_id,
        location=location,
        is_playable=False,
        agent_tier=CharacterAgentTier.utility,
        public_sheet=PublicSheet(role=role),
        descriptions=CharacterDescriptions(public=visible_description),
        visuals=CharacterVisuals(default_loadout=visible_description),
        mechanics=mechanics_from_statblock(
            statblock,
            monster_key=spawn.monster_key,
            source="router_combatant_spawn",
        ),
    )


def mechanics_from_statblock(
    statblock: DndMonsterStatBlock,
    *,
    monster_key: str = "",
    source: str = "router_combatant_spawn",
) -> dict[str, Any]:
    scores = _flat_scores(statblock)
    detailed_scores = {
        key: {"score": value, "modifier": _ability_modifier(value)}
        for key, value in scores.items()
    }
    hit_points = {
        "current": statblock.hit_points,
        "max": statblock.hit_points,
        "temporary": 0,
        "formula": statblock.hit_dice,
    }
    skills = {
        skill.name: {"value": skill.value}
        for skill in statblock.skills
        if skill.name
    }
    actions = [_action_to_mechanics(action) for action in statblock.actions]
    sheet = {
        "ruleset_id": "dnd5e_basic",
        "identity": {
            "name": "",
            "total_level": 0,
            "experience_points": 0,
            "classes": [],
        },
        "statblock": {
            "size": statblock.size,
            "creature_type": statblock.creature_type,
            "alignment": statblock.alignment,
            "speed": statblock.speed,
            "ability_scores": detailed_scores,
            "proficiency_bonus": statblock.proficiency_bonus,
            "skills": skills,
            "saves": {},
            "defenses": {
                "armor_class": {"value": statblock.armor_class},
                "hit_points": hit_points,
                "initiative": {
                    "value": detailed_scores["dex"]["modifier"],
                    "advantage_state": "normal",
                },
                "conditions": [],
            },
            "senses": statblock.senses,
            "passive_perception": statblock.passive_perception,
            "languages": statblock.languages,
            "challenge": {
                "rating": statblock.challenge_rating,
                "xp": statblock.xp,
            },
            "challenge_rating": statblock.challenge_rating,
            "xp": statblock.xp,
            "features": [
                {"name": trait.name, "description": trait.description}
                for trait in statblock.traits
                if trait.name or trait.description
            ],
            "actions": actions,
        },
    }
    return {
        "ruleset_id": "dnd5e_basic",
        "source": source,
        "monster_key": monster_key,
        "ability_scores": scores,
        "proficiency_bonus": statblock.proficiency_bonus,
        "armor_class": statblock.armor_class,
        "hit_points": hit_points,
        "challenge_rating": statblock.challenge_rating,
        "xp_value": statblock.xp,
        "dnd5e_sheet": sheet,
    }


def _flat_scores(statblock: DndMonsterStatBlock) -> dict[str, int]:
    scores = statblock.ability_scores
    return {
        "str": scores.strength,
        "dex": scores.dexterity,
        "con": scores.constitution,
        "int": scores.intelligence,
        "wis": scores.wisdom,
        "cha": scores.charisma,
    }


def _action_to_mechanics(action: DndMonsterAction) -> dict[str, Any]:
    attack = {
        "bonus": action.attack_bonus,
        "damage": _damage_text(action),
        "reach_ft": action.reach_ft,
        "range_normal_ft": action.range_normal_ft,
        "range_long_ft": action.range_long_ft,
        "target": action.target,
    }
    damage_components = []
    formula = _damage_formula(action.damage)
    if formula:
        damage_components.append({
            "formula": formula,
            "damage_type": action.damage_type,
        })
    return {
        "id": action.action_id,
        "name": action.name,
        "description": action.description,
        "attack": attack,
        "damage": damage_components,
    }


def _damage_text(action: DndMonsterAction) -> str:
    damage = action.damage.strip()
    if not damage:
        return ""
    if action.damage_type and action.damage_type not in damage.lower():
        return f"{damage} {action.damage_type}"
    return damage


def _damage_formula(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    import re

    match = re.search(r"\d+d\d+(?:\s*[+-]\s*\d+)?|\d+", text)
    if not match:
        return ""
    return re.sub(r"\s+", "", match.group(0))


def _ability_modifier(score: int) -> int:
    return (int(score) - 10) // 2
