from __future__ import annotations

from pathlib import Path

import pytest

from app.bot.commands import DND_SHEET_PAGES, _DndSheetView, _render_dnd_sheet_page
from app.bot.engine_bridge import EngineBridge
from app.schemas.characters import CharacterRecord, CharacterStatus, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import SessionState, WorldState


SESSION_ID = "dnd_sheet_attach"


def _ddb_export(name: str = "DDB Sheet Name") -> dict:
    stats = [{"id": i, "value": 10} for i in range(1, 7)]
    empty_buckets = {
        "class": [],
        "race": [],
        "background": [],
        "item": [],
        "feat": [],
    }
    return {
        "exporter": {"name": "ayoa-ddb-json-exporter", "version": "0.1.0"},
        "source": {
            "type": "dndbeyond_browser_export",
            "character_id": "ddb-1",
            "url": "https://www.dndbeyond.com/characters/ddb-1",
            "exported_at": "2026-05-05T00:00:00Z",
        },
        "raw": {
            "success": True,
            "data": {
                "id": 1,
                "name": name,
                "currentXp": 900,
                "stats": stats,
                "bonusStats": [{"id": i, "value": None} for i in range(1, 7)],
                "overrideStats": [
                    {"id": i, "value": None} for i in range(1, 7)
                ],
                "baseHitPoints": 22,
                "bonusHitPoints": None,
                "overrideHitPoints": None,
                "removedHitPoints": 3,
                "temporaryHitPoints": 4,
                "currencies": {"gp": 12, "sp": 3},
                "race": {
                    "fullName": "Human",
                    "baseName": "Human",
                    "sizeId": 3,
                    "weightSpeeds": {"walk": 30},
                    "racialTraits": [],
                },
                "background": {"definition": {"name": "Soldier"}},
                "classes": [
                    {
                        "id": 10,
                        "level": 3,
                        "definition": {
                            "id": 1,
                            "name": "Fighter",
                            "hitDice": 10,
                            "canCastSpells": False,
                        },
                    }
                ],
                "modifiers": {
                    **empty_buckets,
                    "class": [
                        {
                            "id": "athletics-prof",
                            "type": "proficiency",
                            "subType": "athletics",
                            "friendlySubtypeName": "Athletics",
                        }
                    ],
                },
                "actions": empty_buckets,
                "spells": empty_buckets,
                "classSpells": [],
                "spellSlots": [],
                "pactMagic": [],
                "inventory": [
                    {
                        "id": 1,
                        "quantity": 1,
                        "equipped": True,
                        "isAttuned": False,
                        "definition": {
                            "id": 1,
                            "name": "Longsword",
                            "filterType": "Weapon",
                            "attackType": 1,
                            "damage": {"diceString": "1d8"},
                        },
                    }
                ],
                "conditions": [],
            },
        },
    }


def _checkpoint(bindings: dict[str, str] | None = None) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            turn_index=1,
            character_bindings=bindings or {"hero": "42"},
        ),
        world_state=WorldState(),
        characters=[
            CharacterRecord(
                character_id="hero",
                name="Story Name",
                status=CharacterStatus.active,
                is_playable=True,
                public_sheet=PublicSheet(role="protagonist"),
            ),
            CharacterRecord(
                character_id="rival",
                name="Rival",
                status=CharacterStatus.active,
                is_playable=True,
            ),
        ],
    )


@pytest.fixture
def bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return EngineBridge(saves_dir=str(tmp_path), prompts_dir="app/prompts")


def _seed(bridge: EngineBridge, ckpt: CheckpointFile) -> None:
    bridge.checkpoint_mgr.save(ckpt)


@pytest.mark.asyncio
async def test_attach_preserves_story_identity_without_override(
    bridge: EngineBridge,
):
    _seed(bridge, _checkpoint())

    summary = await bridge.attach_dndbeyond_character_export(
        SESSION_ID,
        42,
        _ddb_export(),
    )

    assert summary.character_id == "hero"
    assert summary.character_name == "Story Name"
    assert summary.imported_name == "DDB Sheet Name"
    assert summary.name_overridden is False
    assert summary.total_level == 3
    assert summary.armor_class == 10
    assert summary.hit_points_current == 19
    assert summary.hit_points_max == 22
    assert summary.hit_points_temporary == 4
    assert summary.session_ruleset_id == "dnd5e_basic"
    assert summary.cat_ii_resolution_mode == "dnd5e_router"
    assert summary.player_roll_mode == "auto"

    loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
    settings = loaded.session.config.settings
    assert settings.ruleset_id == "dnd5e_basic"
    assert settings.cat_ii_resolution_mode == "dnd5e_router"
    assert settings.player_roll_mode == "auto"
    hero = next(c for c in loaded.characters if c.character_id == "hero")
    assert hero.name == "Story Name"
    assert hero.mechanics["dnd5e_sheet"]["identity"]["name"] == "DDB Sheet Name"
    assert "raw_source" in hero.mechanics["dnd5e_sheet"]
    assert "D&D sheet attached" in loaded.session.pending_router_state_changes[-1]
    assert (
        "D&D session settings enabled"
        in loaded.session.pending_router_state_changes[-1]
    )


@pytest.mark.asyncio
async def test_attach_enables_dnd_without_overwriting_player_roll_mode(
    bridge: EngineBridge,
):
    ckpt = _checkpoint()
    ckpt.session.config.settings.player_roll_mode = "interactive"
    _seed(bridge, ckpt)

    summary = await bridge.attach_dndbeyond_character_export(
        SESSION_ID,
        42,
        _ddb_export(),
    )

    assert summary.session_ruleset_id == "dnd5e_basic"
    assert summary.cat_ii_resolution_mode == "dnd5e_router"
    assert summary.player_roll_mode == "interactive"

    loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
    settings = loaded.session.config.settings
    assert settings.ruleset_id == "dnd5e_basic"
    assert settings.cat_ii_resolution_mode == "dnd5e_router"
    assert settings.player_roll_mode == "interactive"


@pytest.mark.asyncio
async def test_attach_name_override_is_explicit(bridge: EngineBridge):
    _seed(bridge, _checkpoint())

    summary = await bridge.attach_dndbeyond_character_export(
        SESSION_ID,
        42,
        _ddb_export(),
        name_override="Table Name",
    )

    assert summary.character_name == "Table Name"
    assert summary.name_overridden is True
    loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
    hero = next(c for c in loaded.characters if c.character_id == "hero")
    assert hero.name == "Table Name"


@pytest.mark.asyncio
async def test_attach_rejects_unbound_target(bridge: EngineBridge):
    _seed(bridge, _checkpoint(bindings={"hero": "42", "rival": "99"}))

    with pytest.raises(ValueError, match="currently control"):
        await bridge.attach_dndbeyond_character_export(
            SESSION_ID,
            42,
            _ddb_export(),
            character_id="rival",
        )


@pytest.mark.asyncio
async def test_sheet_renderer_uses_snapshot_not_raw_source(bridge: EngineBridge):
    _seed(bridge, _checkpoint())
    await bridge.attach_dndbeyond_character_export(
        SESSION_ID,
        42,
        _ddb_export(),
    )
    hero = bridge.get_bound_character_record(SESSION_ID, 42)

    embed = _render_dnd_sheet_page(hero, "overview")
    rendered = "\n".join(
        [embed.description or ""]
        + [f"{field.name}\n{field.value}" for field in embed.fields]
    )

    assert "Combat" in rendered
    assert "AC 10" in rendered
    assert "HP 19/22 (+4 temp)" in rendered
    assert "raw_source" not in rendered


@pytest.mark.asyncio
async def test_sheet_view_pages_with_buttons_and_wraps(bridge: EngineBridge):
    _seed(bridge, _checkpoint())
    await bridge.attach_dndbeyond_character_export(
        SESSION_ID,
        42,
        _ddb_export(),
    )

    view = _DndSheetView(
        engine=bridge,
        session_id=SESSION_ID,
        user_id=42,
        character_id="hero",
        page="overview",
    )

    assert view.page == "overview"
    assert view.page_button.label == f"Overview 1/{len(DND_SHEET_PAGES)}"

    view._advance(1)
    assert view.page == "abilities"
    assert view.page_button.label == f"Abilities 2/{len(DND_SHEET_PAGES)}"

    view._advance(-2)
    assert view.page == "features"
    assert view.page_button.label == f"Features 6/{len(DND_SHEET_PAGES)}"


@pytest.mark.asyncio
async def test_sheet_view_reload_reads_latest_checkpoint(bridge: EngineBridge):
    _seed(bridge, _checkpoint())
    await bridge.attach_dndbeyond_character_export(
        SESSION_ID,
        42,
        _ddb_export(),
    )
    view = _DndSheetView(
        engine=bridge,
        session_id=SESSION_ID,
        user_id=42,
        character_id="hero",
        page="overview",
    )

    ckpt = bridge.checkpoint_mgr.load_latest(SESSION_ID)
    hero = next(c for c in ckpt.characters if c.character_id == "hero")
    hp = (
        hero.mechanics["dnd5e_sheet"]["statblock"]
        ["defenses"]["hit_points"]
    )
    hp["current"] = 7
    hp["temporary"] = 0
    bridge.checkpoint_mgr.save(ckpt)

    embed = view._render_current()
    rendered = "\n".join(
        [embed.description or ""]
        + [f"{field.name}\n{field.value}" for field in embed.fields]
    )

    assert "HP 7/22" in rendered
    assert "HP 19/22" not in rendered
