from __future__ import annotations

from app.engine import dnd_monsters
from app.schemas.dnd_monsters import DndCombatantSpawn


def _rat_spawn(**overrides) -> DndCombatantSpawn:
    data = {
        "character_id": "rat_1",
        "monster_key": "rat",
        "name": "Rat",
        "location": "",
        "description": "A small rat snaps at exposed ankles.",
        "statblock": {
            "size": "Tiny",
            "creature_type": "beast",
            "alignment": "unaligned",
            "armor_class": 10,
            "hit_points": 1,
            "hit_dice": "1d4 - 1",
            "speed": "20 ft.",
            "ability_scores": {
                "strength": 2,
                "dexterity": 11,
                "constitution": 9,
                "intelligence": 2,
                "wisdom": 10,
                "charisma": 4,
            },
            "proficiency_bonus": 2,
            "skills": [],
            "senses": ["darkvision 30 ft."],
            "passive_perception": 10,
            "languages": [],
            "challenge_rating": "0",
            "xp": 10,
            "traits": [],
            "actions": [
                {
                    "action_id": "bite",
                    "name": "Bite",
                    "attack_bonus": 0,
                    "reach_ft": 5,
                    "range_normal_ft": 0,
                    "range_long_ft": 0,
                    "target": "one target",
                    "damage": "1 piercing",
                    "damage_type": "piercing",
                    "description": (
                        "Melee Weapon Attack: +0 to hit, reach 5 ft., one "
                        "target. Hit: 1 piercing damage."
                    ),
                }
            ],
        },
    }
    data.update(overrides)
    return DndCombatantSpawn(**data)


def test_router_fallback_statblock_becomes_dnd_mechanics():
    dnd_monsters.clear_statblock_override_providers()
    spawn = _rat_spawn()

    rat = dnd_monsters.character_from_combatant_spawn(
        spawn,
        default_location="cellar",
    )

    assert rat.character_id == "rat_1"
    assert rat.location == "cellar"
    assert rat.is_playable is False
    assert rat.public_sheet.role == "Tiny beast"
    mechanics = rat.mechanics
    assert mechanics["ruleset_id"] == "dnd5e_basic"
    assert mechanics["armor_class"] == 10
    assert mechanics["hit_points"]["max"] == 1
    assert mechanics["challenge_rating"] == "0"
    assert mechanics["xp_value"] == 10
    statblock = mechanics["dnd5e_sheet"]["statblock"]
    assert statblock["defenses"]["initiative"]["value"] == 0
    assert statblock["actions"][0]["attack"]["damage"] == "1 piercing"
    assert statblock["actions"][0]["damage"] == [
        {"formula": "1", "damage_type": "piercing"}
    ]


def test_registered_statblock_provider_overrides_router_fallback():
    dnd_monsters.clear_statblock_override_providers()
    spawn = _rat_spawn()

    def corrected(candidate: DndCombatantSpawn):
        assert candidate.monster_key == "rat"
        return {
            **candidate.statblock.model_dump(),
            "armor_class": 13,
            "hit_points": 2,
            "challenge_rating": "1/8",
            "xp": 25,
        }

    try:
        dnd_monsters.register_statblock_override_provider(corrected)
        rat = dnd_monsters.character_from_combatant_spawn(spawn)
    finally:
        dnd_monsters.clear_statblock_override_providers()

    assert rat.mechanics["armor_class"] == 13
    assert rat.mechanics["hit_points"]["max"] == 2
    assert rat.mechanics["challenge_rating"] == "1/8"
    assert rat.mechanics["xp_value"] == 25
