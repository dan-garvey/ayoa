from __future__ import annotations

import pytest

from app.bot.engine_bridge import EngineBridge
from app.engine import dnd_experience
from app.engine.dnd_combat import (
    apply_damage,
    drain_pending_experience_awards,
    start_combat,
)
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import SessionState, WorldState


SESSION_ID = "dnd_xp"


def _character(
    character_id: str = "hero",
    *,
    xp: int = 2600,
    level: int = 3,
    hp: int = 10,
    challenge_rating: str = "",
) -> CharacterRecord:
    mechanics = {
        "ruleset_id": "dnd5e_basic",
        "experience_points": xp,
        "armor_class": 12,
        "hit_points": {
            "current": hp,
            "max": hp,
            "temporary": 0,
        },
        "dnd5e_sheet": {
            "identity": {
                "name": character_id.title(),
                "total_level": level,
                "experience_points": xp,
                "classes": [{"name": "Fighter", "level": level}],
            },
            "statblock": {},
        },
    }
    if challenge_rating:
        mechanics["challenge_rating"] = challenge_rating
    return CharacterRecord(
        character_id=character_id,
        name=character_id.title(),
        status="active",
        is_playable=True,
        mechanics=mechanics,
    )


def test_award_experience_updates_runtime_overlay_and_sheet_snapshot():
    hero = _character()

    result = dnd_experience.award_experience(
        hero,
        100,
        source="goblin ambush",
        turn_index=7,
        awarded_at="2026-05-15T00:00:00+00:00",
    )

    assert result["before"] == 2600
    assert result["after"] == 2700
    assert result["level_available"] is True
    assert result["eligible_level"] == 4
    assert hero.mechanics["experience_points"] == 2700
    assert hero.mechanics["dnd5e_runtime"]["experience_points"] == 2700
    assert (
        hero.mechanics["dnd5e_sheet"]["identity"]["experience_points"]
        == 2700
    )
    assert hero.mechanics["dnd5e_runtime"]["experience_awards"] == [
        {
            "amount": 100,
            "before": 2600,
            "after": 2700,
            "source": "goblin ambush",
            "turn_index": 7,
            "awarded_at": "2026-05-15T00:00:00+00:00",
        }
    ]


def test_award_experience_rejects_non_dnd_characters():
    character = CharacterRecord(character_id="plain", name="Plain")

    with pytest.raises(ValueError, match="D&D mechanics"):
        dnd_experience.award_experience(character, 50)


def test_experience_view_reports_xp_to_next_level():
    hero = _character(xp=900, level=3)

    view = dnd_experience.experience_view(hero)

    assert view["experience_points"] == 900
    assert view["next_level"] == 4
    assert view["xp_to_next_level"] == 1800
    assert view["level_available"] is False


def test_bridge_awards_xp_to_all_bound_dnd_player_characters(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    bridge = EngineBridge(saves_dir=str(tmp_path), prompts_dir="app/prompts")
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            story_id="story",
            turn_index=5,
            character_bindings={"hero": "111", "cleric": "222"},
        ),
        world_state=WorldState(),
        characters=[
            _character("hero", xp=900, level=3),
            _character("cleric", xp=14000, level=6),
        ],
    )
    bridge.checkpoint_mgr.save(ckpt)

    results = bridge.award_dnd_experience(
        SESSION_ID,
        "all",
        150,
        source="session reward",
    )

    assert [result.character_id for result in results] == ["hero", "cleric"]
    assert [result.after for result in results] == [1050, 14150]

    loaded = bridge.load_latest(SESSION_ID)
    by_id = {character.character_id: character for character in loaded.characters}
    assert dnd_experience.experience_points(by_id["hero"]) == 1050
    assert (
        by_id["hero"].mechanics["dnd5e_runtime"]["experience_awards"][-1]
        ["source"]
        == "session reward"
    )


def test_defeated_enemy_awards_split_xp_once_to_player_combatants():
    session = SessionState(
        session_id="combat_xp",
        turn_index=9,
        character_bindings={"hero": "111", "cleric": "222"},
    )
    hero = _character("hero", xp=0, level=1, hp=10)
    cleric = _character("cleric", xp=0, level=1, hp=10)
    goblin = _character("goblin", xp=0, level=1, hp=4, challenge_rating="1/4")
    start_combat(session, [hero, cleric, goblin])

    apply_damage(session, "goblin", 99, characters=[hero, cleric, goblin])

    assert dnd_experience.experience_points(hero) == 25
    assert dnd_experience.experience_points(cleric) == 25
    assert session.active_combat.xp_awarded_combatant_ids == ["goblin"]
    awards = drain_pending_experience_awards(session)
    assert sorted((award.character_id, award.amount) for award in awards) == [
        ("cleric", 25),
        ("hero", 25),
    ]
    assert awards[0].source == "Defeated Goblin"
    assert awards[0].experience_points == 25
    audit = "\n".join(session.active_combat.audit_lines)
    assert "XP awarded for defeating Goblin: 50 split as" in audit
    assert "Hero +25" in audit
    assert "Cleric +25" in audit

    apply_damage(session, "goblin", 99, characters=[hero, cleric, goblin])

    assert dnd_experience.experience_points(hero) == 25
    assert dnd_experience.experience_points(cleric) == 25
