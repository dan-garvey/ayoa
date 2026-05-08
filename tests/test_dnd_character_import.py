import json
from pathlib import Path

from app.engine import mechanics
from app.engine.dnd_character_import import (
    character_record_from_snapshot,
    load_dndbeyond_export,
    mechanics_from_snapshot,
    normalize_dndbeyond_export,
)
from app.schemas.characters import CharacterRecord
from app.schemas.dnd_cat_ii import PlannedRoll


def _synthetic_ddb_export():
    return {
        "exporter": {"name": "ayoa-ddb-json-exporter", "version": "0.1.0"},
        "source": {
            "type": "dndbeyond_browser_export",
            "character_id": "123",
            "url": "https://www.dndbeyond.com/characters/123",
            "exported_at": "2026-05-05T00:00:00Z",
        },
        "raw": {
            "success": True,
            "data": {
                "id": 123,
                "name": "Test Hexblade",
                "currentXp": 14000,
                "stats": [
                    {"id": 1, "value": 8},
                    {"id": 2, "value": 14},
                    {"id": 3, "value": 14},
                    {"id": 4, "value": 8},
                    {"id": 5, "value": 12},
                    {"id": 6, "value": 15},
                ],
                "bonusStats": [{"id": i, "value": None} for i in range(1, 7)],
                "overrideStats": [{"id": i, "value": None} for i in range(1, 7)],
                "baseHitPoints": 28,
                "bonusHitPoints": None,
                "overrideHitPoints": None,
                "removedHitPoints": 5,
                "temporaryHitPoints": 3,
                "deathSaves": {"successCount": 1, "failCount": 2},
                "conditions": [],
                "currencies": {"cp": 0, "sp": 1, "ep": 0, "gp": 34, "pp": 0},
                "race": {
                    "fullName": "Custom Lineage",
                    "baseName": "Custom Lineage",
                    "sizeId": 3,
                    "weightSpeeds": {"walk": 30},
                    "racialTraits": [],
                },
                "background": {"definition": {"name": "Custom Background"}},
                "classes": [
                    {
                        "id": 10,
                        "level": 4,
                        "isStartingClass": True,
                        "hitDiceUsed": 1,
                        "definition": {
                            "id": 6,
                            "name": "Sorcerer",
                            "hitDice": 6,
                            "canCastSpells": True,
                            "spellCastingAbilityId": 6,
                        },
                        "subclassDefinition": {
                            "id": 99,
                            "name": "Clockwork Soul",
                        },
                    },
                    {
                        "id": 11,
                        "level": 2,
                        "isStartingClass": False,
                        "hitDiceUsed": 0,
                        "definition": {
                            "id": 7,
                            "name": "Warlock",
                            "hitDice": 8,
                            "canCastSpells": True,
                            "spellCastingAbilityId": 6,
                        },
                        "subclassDefinition": {"id": 115, "name": "Hexblade"},
                    },
                ],
                "modifiers": {
                    "race": [
                        {
                            "id": "race-cha",
                            "type": "bonus",
                            "subType": "charisma-score",
                            "friendlySubtypeName": "Charisma Score",
                            "fixedValue": 2,
                        },
                        {
                            "id": "darkvision",
                            "type": "set-base",
                            "subType": "darkvision",
                            "friendlySubtypeName": "Darkvision",
                            "fixedValue": 60,
                        },
                    ],
                    "class": [
                        {
                            "id": "con-save",
                            "type": "proficiency",
                            "subType": "constitution-saving-throws",
                            "friendlySubtypeName": "Constitution Saving Throws",
                        },
                        {
                            "id": "cha-save",
                            "type": "proficiency",
                            "subType": "charisma-saving-throws",
                            "friendlySubtypeName": "Charisma Saving Throws",
                        },
                        {
                            "id": "arcana",
                            "type": "proficiency",
                            "subType": "arcana",
                            "friendlySubtypeName": "Arcana",
                        },
                    ],
                    "background": [
                        {
                            "id": "perception",
                            "type": "proficiency",
                            "subType": "perception",
                            "friendlySubtypeName": "Perception",
                        }
                    ],
                    "item": [
                        {
                            "id": "armor-plus-one",
                            "type": "bonus",
                            "subType": "armor-class",
                            "friendlySubtypeName": "Armor Class",
                            "fixedValue": 1,
                        },
                        {
                            "id": "save-adv",
                            "type": "advantage",
                            "subType": "saving-throws",
                            "friendlySubtypeName": "Saving Throws",
                            "restriction": "against dragon breath weapons",
                        },
                    ],
                    "feat": [
                        {
                            "id": "feat-cha",
                            "type": "bonus",
                            "subType": "charisma-score",
                            "friendlySubtypeName": "Charisma Score",
                            "fixedValue": 1,
                        }
                    ],
                    "condition": [],
                },
                "actions": {
                    "class": [
                        {
                            "id": "hexblade-curse",
                            "name": "Hexblade's Curse",
                            "snippet": "Curse a target.",
                            "limitedUse": {
                                "numberUsed": 0,
                                "maxUses": 1,
                                "resetType": 1,
                            },
                            "range": {"range": 30},
                        }
                    ],
                    "race": [],
                    "background": [],
                    "item": [],
                    "feat": [],
                },
                "spells": {
                    "class": [
                        {
                            "id": 1,
                            "prepared": True,
                            "alwaysPrepared": False,
                            "usesSpellSlot": True,
                            "definition": {
                                "id": 100,
                                "name": "Test Bolt",
                                "level": 0,
                                "school": "Evocation",
                                "range": {"origin": "Ranged", "rangeValue": 120},
                                "duration": {
                                    "durationInterval": 0,
                                    "durationType": "Instantaneous",
                                },
                                "components": [1, 2],
                            },
                        }
                    ],
                    "race": [],
                    "background": [],
                    "item": [],
                    "feat": [],
                },
                "classSpells": [],
                "spellSlots": [{"level": 1, "used": 1, "available": 3}],
                "pactMagic": [{"level": 1, "used": 0, "available": 1}],
                "inventory": [
                    {
                        "id": 1,
                        "quantity": 1,
                        "equipped": True,
                        "isAttuned": False,
                        "definition": {
                            "id": 1,
                            "name": "Breastplate, +1",
                            "filterType": "Armor",
                            "armorClass": 14,
                            "armorTypeId": 2,
                            "baseArmorName": "Breastplate",
                            "grantedModifiers": [],
                        },
                    },
                    {
                        "id": 2,
                        "quantity": 1,
                        "equipped": True,
                        "isAttuned": False,
                        "definition": {
                            "id": 2,
                            "name": "Shield",
                            "filterType": "Armor",
                            "armorClass": 2,
                            "armorTypeId": 4,
                            "baseArmorName": "Shield",
                        },
                    },
                    {
                        "id": 3,
                        "quantity": 1,
                        "equipped": True,
                        "isAttuned": False,
                        "definition": {
                            "id": 3,
                            "name": "Longsword, +1",
                            "filterType": "Weapon",
                            "attackType": 1,
                            "damage": {"diceString": "1d8"},
                            "damageType": "Slashing",
                        },
                    },
                ],
                "features": [],
                "feats": [
                    {
                        "definition": {
                            "id": 29,
                            "name": "Lucky",
                            "description": "Spend luck points.",
                        }
                    }
                ],
                "creatures": [],
            },
        },
    }


def test_dndbeyond_export_normalizes_to_full_snapshot():
    snapshot = normalize_dndbeyond_export(_synthetic_ddb_export())
    statblock = snapshot["statblock"]

    assert snapshot["identity"]["name"] == "Test Hexblade"
    assert snapshot["identity"]["total_level"] == 6
    assert statblock["ability_scores"]["cha"]["score"] == 18
    assert statblock["proficiency_bonus"] == 3
    assert statblock["skills"]["arcana"]["value"] == 2
    assert statblock["skills"]["perception"]["value"] == 4
    assert statblock["saves"]["con"]["value"] == 5
    assert statblock["defenses"]["armor_class"]["value"] == 19
    assert statblock["defenses"]["hit_points"]["current"] == 23
    assert statblock["defenses"]["hit_points"]["temporary"] == 3
    assert any(r["name"] == "Hexblade's Curse" for r in statblock["resources"])
    assert any(a["name"] == "Longsword, +1" for a in statblock["actions"])
    assert any(s["name"] == "Test Bolt" for s in statblock["spellcasting"]["spells"])
    assert "raw_source" in snapshot


def test_snapshot_projects_to_existing_mechanics_shape_without_prompt_raw_bloat():
    snapshot = normalize_dndbeyond_export(_synthetic_ddb_export())
    projected = mechanics_from_snapshot(snapshot)

    assert projected["ruleset_id"] == "dnd5e_basic"
    assert projected["ability_scores"]["cha"] == 18
    assert projected["armor_class"] == 19
    assert projected["hit_points"] == {
        "current": 23,
        "max": 28,
        "temporary": 3,
    }
    assert "dnd5e_sheet" in projected
    assert "raw_source" not in projected["raw"]


def test_character_record_from_snapshot_and_detailed_roll_modifiers():
    snapshot = normalize_dndbeyond_export(_synthetic_ddb_export())
    record = character_record_from_snapshot(snapshot, character_id="test_hexblade")

    assert record.character_id == "test_hexblade"
    assert record.is_playable is True
    assert "Warlock 2" in record.public_sheet.role

    arcana = PlannedRoll(
        roll_id="roll_1",
        actor_id="test_hexblade",
        kind="skill_check",
        ability="int",
        skill="arcana",
        dc=0,
        opposed_by="",
        advantage_state="normal",
        reason="test",
    )
    con_save = PlannedRoll(
        roll_id="roll_2",
        actor_id="test_hexblade",
        kind="saving_throw",
        ability="con",
        skill="",
        dc=0,
        opposed_by="",
        advantage_state="normal",
        reason="test",
    )

    assert mechanics.roll_modifier(record, arcana) == 2
    assert mechanics.roll_modifier(record, con_save) == 5


def test_detailed_roll_modifiers_accept_underscore_skill_aliases():
    record = CharacterRecord(
        character_id="synthetic",
        name="Synthetic",
        mechanics={
            "ruleset_id": "dnd5e_basic",
            "ability_scores": {"dex": 12},
            "proficiency_bonus": 2,
            "dnd5e_sheet": {
                "statblock": {
                    "skills": {
                        "sleight of hand": {
                            "value": 7,
                            "proficiency_multiplier": 2,
                        },
                    },
                },
            },
        },
    )
    sleight = PlannedRoll(
        roll_id="roll_1",
        actor_id="synthetic",
        kind="skill_check",
        ability="dex",
        skill="sleight_of_hand",
        dc=0,
        opposed_by="",
        advantage_state="normal",
        reason="test",
    )

    assert mechanics.roll_modifier(record, sleight) == 7


def test_load_dndbeyond_export_reads_envelope(tmp_path: Path):
    path = tmp_path / "ddb.json"
    path.write_text(json.dumps(_synthetic_ddb_export()), encoding="utf-8")

    loaded = load_dndbeyond_export(path)

    assert loaded["source"]["character_id"] == "123"
