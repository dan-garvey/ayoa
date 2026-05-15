from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


def _clean_id(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


class DndMonsterAbilityScores(BaseModel):
    """Router-emitted fallback ability scores for a D&D combatant spawn."""

    model_config = ConfigDict(extra="forbid")

    strength: int
    dexterity: int
    constitution: int
    intelligence: int
    wisdom: int
    charisma: int

    @model_validator(mode="after")
    def _clean(self) -> "DndMonsterAbilityScores":
        for key in (
            "strength",
            "dexterity",
            "constitution",
            "intelligence",
            "wisdom",
            "charisma",
        ):
            value = int(getattr(self, key, 10) or 10)
            setattr(self, key, min(30, max(1, value)))
        return self


class DndMonsterSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: int

    @model_validator(mode="after")
    def _clean(self) -> "DndMonsterSkill":
        self.name = self.name.strip().lower().replace("_", " ")
        return self


class DndMonsterTrait(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str

    @model_validator(mode="after")
    def _clean(self) -> "DndMonsterTrait":
        self.name = self.name.strip()
        self.description = self.description.strip()
        return self


class DndMonsterAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    name: str
    attack_bonus: int
    reach_ft: int
    range_normal_ft: int
    range_long_ft: int
    target: str
    damage: str
    damage_type: str
    description: str

    @model_validator(mode="after")
    def _clean(self) -> "DndMonsterAction":
        self.name = self.name.strip()
        self.action_id = _clean_id(self.action_id or self.name)
        self.reach_ft = max(0, int(self.reach_ft or 0))
        self.range_normal_ft = max(0, int(self.range_normal_ft or 0))
        self.range_long_ft = max(0, int(self.range_long_ft or 0))
        self.target = self.target.strip()
        self.damage = self.damage.strip()
        self.damage_type = self.damage_type.strip().lower()
        self.description = self.description.strip()
        return self


class DndMonsterStatBlock(BaseModel):
    """Compact D&D stat block emitted for a just-in-time combat spawn.

    This schema is intentionally limited to combat surfaces the adapter can
    consume: defenses, ability scores, CR/XP, visible senses/languages, traits,
    and actions.
    """

    model_config = ConfigDict(extra="forbid")

    size: str
    creature_type: str
    alignment: str
    armor_class: int
    hit_points: int
    hit_dice: str
    speed: str
    ability_scores: DndMonsterAbilityScores
    proficiency_bonus: int
    skills: list[DndMonsterSkill]
    senses: list[str]
    passive_perception: int
    languages: list[str]
    challenge_rating: str
    xp: int
    traits: list[DndMonsterTrait]
    actions: list[DndMonsterAction]

    @model_validator(mode="after")
    def _clean(self) -> "DndMonsterStatBlock":
        self.size = self.size.strip()
        self.creature_type = self.creature_type.strip().lower()
        self.alignment = self.alignment.strip().lower()
        self.armor_class = max(0, int(self.armor_class or 0))
        self.hit_points = max(1, int(self.hit_points or 1))
        self.hit_dice = self.hit_dice.strip()
        self.speed = self.speed.strip()
        self.proficiency_bonus = int(self.proficiency_bonus or 0)
        self.senses = [sense.strip() for sense in self.senses if sense.strip()]
        self.passive_perception = max(0, int(self.passive_perception or 0))
        self.languages = [
            language.strip()
            for language in self.languages
            if language.strip() and language.strip() not in {"-", "—"}
        ]
        self.challenge_rating = self.challenge_rating.strip()
        self.xp = max(0, int(self.xp or 0))
        return self


class DndCombatantSpawn(BaseModel):
    """D&D-only router request to create a combat-ready NPC from a stat block."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    monster_key: str
    name: str
    location: str
    description: str
    statblock: DndMonsterStatBlock

    @model_validator(mode="after")
    def _clean(self) -> "DndCombatantSpawn":
        self.character_id = _clean_id(self.character_id)
        self.monster_key = _clean_id(self.monster_key or self.name)
        self.name = self.name.strip()
        self.location = self.location.strip()
        self.description = self.description.strip()
        return self


def empty_monster_statblock() -> dict[str, Any]:
    return {
        "size": "",
        "creature_type": "",
        "alignment": "",
        "armor_class": 10,
        "hit_points": 1,
        "hit_dice": "",
        "speed": "",
        "ability_scores": {
            "strength": 10,
            "dexterity": 10,
            "constitution": 10,
            "intelligence": 10,
            "wisdom": 10,
            "charisma": 10,
        },
        "proficiency_bonus": 0,
        "skills": [],
        "senses": [],
        "passive_perception": 10,
        "languages": [],
        "challenge_rating": "",
        "xp": 0,
        "traits": [],
        "actions": [],
    }
