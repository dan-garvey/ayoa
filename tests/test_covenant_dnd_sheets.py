from __future__ import annotations

from app.engine import mechanics
from app.schemas.dnd_cat_ii import PlannedRoll
from scripts.apply_covenant_dnd_sheets import PROFILES, build_mechanics


COVENANT_CHARACTER_IDS = {
    "player_garvey",
    "ashara_vel_kothren",
    "ysolde_thornmantle",
    "caelindra_vaeyn",
    "seraphel_dawnquill",
    "thessaly_morrow",
    "lysara_vane",
    "rashid_vel_amara",
    "aldric_verantus",
    "prof_kira_vel_shaan",
    "prof_elara_windwhisper",
    "prof_vex_thorn",
    "prof_gareth_stone",
    "chancellor_ashworth",
    "lord_verantus",
    "lady_ashira_vel_kothren",
    "toxicia_vaeyn",
    "lady_coldpeak",
}


def test_covenant_synthetic_profiles_cover_every_seed_character():
    assert set(PROFILES) == COVENANT_CHARACTER_IDS

    for character_id, profile in PROFILES.items():
        built = build_mechanics(
            character_id=character_id,
            name=character_id,
            role="test",
            profile=profile,
            generated_at="2026-05-05T00:00:00+00:00",
        )
        sheet = built.get("dnd5e_sheet") or {}
        assert built.get("ruleset_id") == "dnd5e_basic"
        assert sheet.get("source", {}).get("type") == (
            "synthetic_ayoa_covenant_thrones"
        )
        assert set(built.get("ability_scores") or {}) >= {
            "str",
            "dex",
            "con",
            "int",
            "wis",
            "cha",
        }
        assert sheet.get("statblock", {}).get("skills")


def test_covenant_synthetic_sheets_are_used_for_roll_modifiers():
    by_id = {
        character_id: _character_record(character_id)
        for character_id in COVENANT_CHARACTER_IDS
    }

    assert _modifier(by_id["ashara_vel_kothren"], "str", "athletics") >= 8
    assert _modifier(by_id["rashid_vel_amara"], "wis", "perception") >= 8
    assert _modifier(by_id["lysara_vane"], "wis", "perception") >= 6


def _modifier(character, ability: str, skill: str) -> int:
    roll = PlannedRoll(
        roll_id=f"roll_{character.character_id}_{skill}",
        actor_id=character.character_id,
        kind="skill_check",
        ability=ability,
        skill=skill,
        dc=0,
        opposed_by="",
        advantage_state="normal",
        reason="test",
    )
    return mechanics.roll_modifier(character, roll)


def _character_record(character_id: str):
    from app.schemas.characters import CharacterRecord

    return CharacterRecord(
        character_id=character_id,
        name=character_id,
        mechanics=build_mechanics(
            character_id=character_id,
            name=character_id,
            role="test",
            profile=PROFILES[character_id],
            generated_at="2026-05-05T00:00:00+00:00",
        ),
    )
