from __future__ import annotations

import json

import pytest

from app.engine import dnd_monsters, mechanics
from app.engine.imported_statblocks import (
    ImportedStatBlockCatalog,
    ImportedStatBlockNotFoundError,
    ImportedStatBlockSpawnSpec,
    ImportedStatBlockValidationError,
    resolve_spawn_character_from_content_state,
    statblock_override_provider,
)
from app.schemas.content_pack import ContentPackDomainCatalog
from app.schemas.content import ContentPackState
from app.schemas.dnd_cat_ii import PlannedRoll
from app.schemas.dnd_monsters import DndCombatantSpawn


def test_imported_statblock_ref_resolves_combat_ready_character_and_combatant():
    domain_catalog = ContentPackDomainCatalog(
        pack_id="synthetic-pack",
        statblocks=[_statblock()],
    )
    catalog = ImportedStatBlockCatalog.from_domain_catalog(domain_catalog)
    spec = ImportedStatBlockSpawnSpec(
        statblock_ref="stat.guardian",
        character_id="guardian_1",
        name="Reviewed Guardian",
        location="entry hall",
        description="/private/source.pdf raw_ocr=PROTECTED_SOURCE_EXCERPT",
        tactics_refs=("tactics.guardian", "tactics.guardian"),
    )

    character = catalog.resolve_character(spec)
    combatant = catalog.resolve_combatant(spec)

    assert character.character_id == "guardian_1"
    assert character.location == "entry hall"
    assert character.descriptions.public == "A combat-ready synthetic guardian."
    assert combatant.armor_class == 15
    assert combatant.hit_points_current == 33
    assert combatant.hit_points_max == 33
    assert combatant.initiative_modifier == 2

    sheet_statblock = character.mechanics["dnd5e_sheet"]["statblock"]
    assert sheet_statblock["speed"] == "walk 30 ft., climb 20 ft."
    assert sheet_statblock["saves"]["dex"]["value"] == 4
    assert sheet_statblock["skills"]["perception"]["value"] == 5
    assert sheet_statblock["defenses"]["damage_resistances"] == [
        {"id": "cold", "name": "cold"}
    ]
    assert sheet_statblock["defenses"]["damage_immunities"] == [
        {"id": "poison", "name": "poison"}
    ]
    assert sheet_statblock["defenses"]["damage_vulnerabilities"] == [
        {"id": "thunder", "name": "thunder"}
    ]
    assert sheet_statblock["defenses"]["condition_immunities"] == [
        {"id": "charmed", "name": "charmed"}
    ]
    assert sheet_statblock["traits"][0]["name"] == "Stone Camouflage"
    assert sheet_statblock["reactions"][0]["activation"]["type"] == "reaction"
    assert sheet_statblock["legendary_actions"][0]["name"] == "Detect"
    assert sheet_statblock["lair_actions"][0]["name"] == "Falling Stones"
    assert sheet_statblock["spellcasting"]["profiles"][0] == {
        "id": "imported_spellcasting",
        "name": "Spellcasting",
        "ability": "wisdom",
        "spell_attack_bonus": 5,
        "spell_save_dc": 13,
        "caster_level": 4,
    }
    assert sheet_statblock["spellcasting"]["slots"] == {
        "1": {"current": 3, "max": 3},
        "2": {"current": 2, "max": 2},
    }
    assert {
        spell["name"] for spell in sheet_statblock["spellcasting"]["spells"]
    } == {"Shield", "Magic Missile", "Mage Hand"}
    assert character.mechanics["imported_statblock"]["tactics_refs"] == [
        "tactics.guardian"
    ]

    assert mechanics.roll_modifier(character, _saving_throw("dex")) == 4
    assert mechanics.roll_modifier(character, _skill_check("perception")) == 5
    assert mechanics.roll_modifier(character, _attack_roll("slam")) == 5

    dumped = json.dumps(character.model_dump(mode="json"), sort_keys=True)
    for forbidden in (
        "/private/source.pdf",
        "PROTECTED_SOURCE_EXCERPT",
        "raw_ocr",
        "source_path",
    ):
        assert forbidden not in dumped


def test_imported_statblock_validation_blocks_missing_combat_fields():
    data = _statblock()
    data.pop("armor_class")
    catalog = ImportedStatBlockCatalog([data])

    with pytest.raises(ImportedStatBlockValidationError, match="armor_class"):
        catalog.resolve_character(_spawn_spec())


def test_imported_statblock_validation_blocks_noncombat_scope():
    data = {**_statblock(), "automation_scope": "noncombat_lookup"}
    catalog = ImportedStatBlockCatalog([data])

    with pytest.raises(ImportedStatBlockValidationError, match="automation_scope"):
        catalog.resolve_character(_spawn_spec())


def test_imported_statblock_allows_explicit_stationary_combatant():
    data = _statblock()
    data["ref"] = "stat.stationary_brain"
    data["title"] = "Stationary Brain"
    data["speed_ft_by_mode"] = {"walk": 0}
    catalog = ImportedStatBlockCatalog([data])
    spec = ImportedStatBlockSpawnSpec(
        statblock_ref="stat.stationary_brain",
        character_id="brain_1",
        name="Stationary Brain",
    )

    character = catalog.resolve_character(spec)
    combatant = catalog.resolve_combatant(spec)
    monster = catalog.resolve_monster_statblock("stat.stationary_brain")

    sheet_statblock = character.mechanics["dnd5e_sheet"]["statblock"]
    assert sheet_statblock["speed"] == "walk 0 ft."
    assert sheet_statblock["speed_ft_by_mode"] == {"walk": 0}
    assert sheet_statblock["defenses"]["movement"] == {
        "walk": {"value": 0, "unit": "ft"}
    }
    assert character.mechanics["defenses"]["movement"] == {
        "walk": {"value": 0, "unit": "ft"}
    }
    assert monster.speed == "walk 0 ft."
    assert combatant.character_id == "brain_1"


def test_imported_statblock_validation_blocks_empty_speed_field():
    data = _statblock()
    data["speed_ft_by_mode"] = {}
    catalog = ImportedStatBlockCatalog([data])

    with pytest.raises(ImportedStatBlockValidationError, match="speed_ft_by_mode"):
        catalog.resolve_character(_spawn_spec())


def test_spawn_override_provider_resolves_ref_and_propagates_blockers():
    catalog = ImportedStatBlockCatalog([_statblock()])
    spawn = _router_spawn(monster_key="stat.guardian")

    try:
        dnd_monsters.clear_statblock_override_providers()
        dnd_monsters.register_statblock_override_provider(
            statblock_override_provider(catalog)
        )
        resolved = dnd_monsters.resolve_statblock(spawn)
    finally:
        dnd_monsters.clear_statblock_override_providers()

    assert resolved.armor_class == 15
    assert resolved.hit_points == 33
    assert resolved.speed == "walk 30 ft., climb 20 ft."

    blocked = _statblock()
    blocked.pop("speed_ft_by_mode")
    blocked_catalog = ImportedStatBlockCatalog([blocked])
    try:
        dnd_monsters.register_statblock_override_provider(
            statblock_override_provider(blocked_catalog)
        )
        with pytest.raises(ImportedStatBlockValidationError, match="speed_ft_by_mode"):
            dnd_monsters.resolve_statblock(spawn)
    finally:
        dnd_monsters.clear_statblock_override_providers()


def test_ref_only_spawn_resolves_from_content_state_catalog():
    spawn = DndCombatantSpawn(
        character_id="guardian_1",
        statblock_ref="stat.guardian",
        description="/private/source.pdf raw_ocr=PROTECTED_SOURCE_EXCERPT",
    )
    content_state = {
        "synthetic-pack": ContentPackState(
            pack_id="synthetic-pack",
            metadata={"statblocks": [_statblock()]},
        )
    }

    character = resolve_spawn_character_from_content_state(
        spawn,
        content_state=content_state,
        default_location="entry hall",
    )

    assert character is not None
    assert character.character_id == "guardian_1"
    assert character.name == "Synthetic Guardian"
    assert character.location == "entry hall"
    assert character.descriptions.public == "A combat-ready synthetic guardian."
    assert character.mechanics["source"] == "imported_statblock_catalog"
    assert character.mechanics["armor_class"] == 15
    assert character.mechanics["imported_statblock"]["ref"] == "stat.guardian"
    dumped = json.dumps(character.model_dump(mode="json"), sort_keys=True)
    for forbidden in (
        "/private/source.pdf",
        "PROTECTED_SOURCE_EXCERPT",
        "raw_ocr",
        "source_path",
    ):
        assert forbidden not in dumped


def test_ref_only_spawn_blocks_missing_and_unreviewed_refs():
    missing = DndCombatantSpawn(
        character_id="guardian_1",
        statblock_ref="stat.missing",
    )
    content_state = {
        "synthetic-pack": ContentPackState(
            pack_id="synthetic-pack",
            metadata={"statblocks": [_statblock()]},
        )
    }

    with pytest.raises(ImportedStatBlockNotFoundError, match="stat.missing"):
        resolve_spawn_character_from_content_state(
            missing,
            content_state=content_state,
        )

    unreviewed = _statblock()
    unreviewed["review_status"] = "needs_review"
    unreviewed["gate_status"] = "flagged"
    unsafe_state = {
        "synthetic-pack": ContentPackState(
            pack_id="synthetic-pack",
            metadata={"statblocks": [unreviewed]},
        )
    }
    with pytest.raises(ImportedStatBlockValidationError, match="review_status"):
        resolve_spawn_character_from_content_state(
            DndCombatantSpawn(
                character_id="guardian_1",
                statblock_ref="stat.guardian",
            ),
            content_state=unsafe_state,
        )


def _spawn_spec() -> ImportedStatBlockSpawnSpec:
    return ImportedStatBlockSpawnSpec(
        statblock_ref="stat.guardian",
        character_id="guardian_1",
    )


def _saving_throw(ability: str) -> PlannedRoll:
    return _roll(kind="saving_throw", ability=ability)


def _skill_check(skill: str) -> PlannedRoll:
    return _roll(kind="skill_check", ability="wis", skill=skill)


def _attack_roll(action_id: str) -> PlannedRoll:
    return _roll(kind="attack_roll", ability="str", action_id=action_id)


def _roll(
    *,
    kind: str,
    ability: str,
    skill: str = "",
    action_id: str = "",
) -> PlannedRoll:
    return PlannedRoll(
        roll_id="roll_1",
        actor_id="guardian_1",
        kind=kind,
        ability=ability,
        skill=skill,
        dc=0,
        opposed_by="",
        advantage_state="normal",
        reason="Slam",
        action_id=action_id,
        target_id="",
        effect_id="",
    )


def _router_spawn(*, monster_key: str) -> DndCombatantSpawn:
    return DndCombatantSpawn(
        character_id="guardian_1",
        monster_key=monster_key,
        name="Router Fallback Guardian",
        location="entry hall",
        description="A fallback statblock that should be ignored by ref.",
        statblock={
            "size": "Medium",
            "creature_type": "construct",
            "alignment": "unaligned",
            "armor_class": 10,
            "hit_points": 1,
            "hit_dice": "1d4",
            "speed": "30 ft.",
            "ability_scores": {
                "strength": 10,
                "dexterity": 10,
                "constitution": 10,
                "intelligence": 10,
                "wisdom": 10,
                "charisma": 10,
            },
            "proficiency_bonus": 2,
            "skills": [],
            "senses": [],
            "passive_perception": 10,
            "languages": [],
            "challenge_rating": "0",
            "xp": 0,
            "traits": [],
            "actions": [
                {
                    "action_id": "fallback",
                    "name": "Fallback",
                    "attack_bonus": 0,
                    "reach_ft": 5,
                    "range_normal_ft": 0,
                    "range_long_ft": 0,
                    "target": "one target",
                    "damage": "1 bludgeoning",
                    "damage_type": "bludgeoning",
                    "description": "Fallback attack.",
                }
            ],
        },
    )


def _statblock() -> dict:
    return {
        "ref": "stat.guardian",
        "content_hash": "sha256:stat-guardian",
        "title": "Synthetic Guardian",
        "summary": "A combat-ready synthetic guardian.",
        "body": "Private source notes at /private/source.pdf must not be copied.",
        "confidence": 0.98,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "automation_scope": "combat",
        "size": "Medium",
        "creature_type": "Construct",
        "alignment": "Unaligned",
        "armor_class": 15,
        "hit_points": 33,
        "hit_dice": "6d8+6",
        "speed_ft_by_mode": {"walk": 30, "climb": 20},
        "ability_scores": {
            "strength": 16,
            "dexterity": 14,
            "constitution": 13,
            "intelligence": 8,
            "wisdom": 12,
            "charisma": 7,
        },
        "proficiency_bonus": 2,
        "saves": [{"name": "dexterity", "value": 4}],
        "skills": [{"name": "perception", "value": 5}],
        "senses": ["darkvision 60 ft."],
        "passive_perception": 15,
        "languages": ["understands its creator"],
        "challenge_rating": "2",
        "xp": 450,
        "damage_resistances": ["cold"],
        "damage_immunities": ["poison"],
        "damage_vulnerabilities": ["thunder"],
        "condition_immunities": ["charmed"],
        "traits": [
            {
                "feature_id": "stone_camouflage",
                "name": "Stone Camouflage",
                "description": "Advantage on checks to hide among stone.",
            }
        ],
        "actions": [
            {
                "feature_id": "slam",
                "name": "Slam",
                "economy": "action",
                "attack_bonus": 5,
                "reach_ft": 5,
                "target": "one target",
                "damage": [
                    {"expression": "1d8+3", "damage_type": "bludgeoning"}
                ],
                "description": "Melee Weapon Attack.",
            }
        ],
        "bonus_actions": [
            {
                "feature_id": "guarded_step",
                "name": "Guarded Step",
                "economy": "bonus_action",
                "description": "Shift without dropping its guard.",
            }
        ],
        "reactions": [
            {
                "feature_id": "parry",
                "name": "Parry",
                "economy": "reaction",
                "description": "Adds 2 to AC against one melee attack.",
            }
        ],
        "legendary_actions": [
            {
                "feature_id": "detect",
                "name": "Detect",
                "economy": "legendary_action",
                "description": "Makes a Wisdom check.",
            }
        ],
        "lair_actions": [
            {
                "feature_id": "falling_stones",
                "name": "Falling Stones",
                "economy": "lair_action",
                "save_dc": 13,
                "save_ability": "dexterity",
                "damage": [{"expression": "2d6", "damage_type": "bludgeoning"}],
                "description": "Loose stones fall in the chamber.",
            }
        ],
        "spellcasting": {
            "ability": "wisdom",
            "save_dc": 13,
            "attack_bonus": 5,
            "caster_level": 4,
            "spell_slots_by_level": {"1": 3, "2": 2},
            "spells": ["Shield", "Magic Missile"],
            "at_will": ["Mage Hand"],
            "limited_uses": {"Invisibility": "1/day"},
        },
    }
