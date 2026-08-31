"""Smoke tests for the hand-authored Spring Rain Second Chorus story seed."""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.characters import CharacterAgentTier
from app.schemas.checkpoint import CheckpointFile


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "storage"
    / "stories"
    / "spring_rain_second_chorus"
    / "ckpt_0000.json"
)


def _load_checkpoint() -> CheckpointFile:
    raw = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return CheckpointFile.model_validate(raw)


def test_checkpoint_loads_as_rules_neutral_story() -> None:
    checkpoint = _load_checkpoint()

    assert checkpoint.schema_version == "5.0"
    assert checkpoint.session.story_id == "spring_rain_second_chorus"
    assert checkpoint.session.session_id == "spring_rain_second_chorus"
    assert checkpoint.session.config.settings.ruleset_id == "narrative"
    assert checkpoint.world_state.physics_ruleset.magic_enabled is False
    assert checkpoint.session.active_combat is None


def test_story_authors_intended_love_triangle() -> None:
    checkpoint = _load_checkpoint()
    by_id = {character.character_id: character for character in checkpoint.characters}

    assert set(by_id) == {
        "ren_sato",
        "mio_tachibana",
        "hanae_morikawa",
        "yui_kisaragi",
        "kenta_fujiwara",
        "naomi_kurata",
        "sora_minazuki",
        "professor_aya_shimizu",
        "daichi_okumura",
        "chika_enomoto",
        "subaru_amari",
    }

    # Every named character is technically claimable via /join.
    for cid, character in by_id.items():
        assert character.is_playable is True, cid
    assert by_id["mio_tachibana"].agent_tier == CharacterAgentTier.premium
    assert by_id["hanae_morikawa"].agent_tier == CharacterAgentTier.premium

    hidden = "\n".join(checkpoint.world_state.hidden_facts).lower()
    assert "ren is in love with mio" in hidden
    assert "mio is in love with hanae" in hidden
    assert "hanae is developing serious feelings for ren" in hidden


def test_story_driver_scaffolding_is_present() -> None:
    checkpoint = _load_checkpoint()

    facts = "\n".join(checkpoint.world_state.facts).lower()
    # Festival timeline anchor so the router can pace deadlines.
    assert "monday" in facts
    assert "sunday-evening closing act" in facts
    # External romantic/competitive pressure and the fixed leak source exist.
    assert any("lantern nine" in fact.lower() for fact in checkpoint.world_state.facts)

    hidden = "\n".join(checkpoint.world_state.hidden_facts).lower()
    # Previously dead-end props now have an access path / defined status.
    assert "cabinet" in hidden and "key" in hidden
    assert "rina aketa is alive" in hidden
    assert "chika enomoto is the unwitting source" in hidden

    rules = checkpoint.session.config.narrative_rules.lower()
    assert "festival timeline" in rules
    assert "story-driver toolkit" in rules


def test_story_seed_has_depth_without_dnd_mechanics() -> None:
    checkpoint = _load_checkpoint()

    assert len(checkpoint.player_primer) > 500
    assert len(checkpoint.world_state.lore) > 3000
    assert len(checkpoint.world_state.hidden_lore) > 2000
    assert len(checkpoint.world_state.facts) >= 12
    assert len(checkpoint.world_state.hidden_facts) >= 20
    assert "story driver" in checkpoint.session.config.narrative_rules

    for character in checkpoint.characters:
        assert character.actor is not None, character.character_id
        assert character.actor.facts, character.character_id
        assert character.public_sheet.public_context.strip(), character.character_id
        assert character.visuals.default_loadout.strip(), character.character_id
        assert character.mechanics == {}, character.character_id
