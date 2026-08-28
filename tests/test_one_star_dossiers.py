"""Rendered player-dossier audience contracts for One-Star Ascension."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.bot.engine_bridge import EngineBridge
from app.engine.context_builder import (
    build_character_packet,
    build_character_state,
    build_world_context,
    format_pending_observations_block,
)
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import AGENT_TURN_HEADER, format_agent_turn_body
from app.schemas.checkpoint import CheckpointFile


SEED_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "storage"
    / "stories"
    / "one_star_ascension_s1"
    / "ckpt_0000.json"
)
SESSION_ID = "one_star_dossier_contract"


def _seed_checkpoint() -> CheckpointFile:
    checkpoint = CheckpointFile.model_validate(
        json.loads(SEED_PATH.read_text(encoding="utf-8"))
    )
    checkpoint.session.session_id = SESSION_ID
    # Dossier projection does not exercise reviewed image artifacts. Keep this
    # fixture self-contained instead of copying the story's frozen image store
    # into every temporary EngineBridge runtime.
    checkpoint.visual_novel_onboarding = None
    checkpoint.reviewed_visual_references = []
    checkpoint.reviewed_visual_novel_sprite_sets = []
    checkpoint.location_visual_reference_ids = {}
    for character in checkpoint.characters:
        character.visuals.identity_reference_id = ""
        character.visuals.sprite_set_id = ""
    return checkpoint


@pytest.fixture
def bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    engine = EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )
    engine.checkpoint_mgr.save(_seed_checkpoint())
    return engine


def test_claimed_newcomer_dossier_keeps_authored_identity_and_control_contract(
    bridge: EngineBridge,
) -> None:
    checkpoint = bridge.load_latest(SESSION_ID)
    newcomer = next(
        character
        for character in checkpoint.characters
        if character.character_id == "one_star_newcomer"
    )
    assert not newcomer.backstory
    assert not newcomer.personality
    assert not newcomer.known_context
    assert newcomer.private_state.secrets == []

    asyncio.run(bridge.claim_player_character(
        SESSION_ID,
        "one_star_newcomer",
        42,
        name="Mara Vale",
        appearance="a scarlet coat and iron-gray braid",
    ))

    dossier = bridge.build_character_dossier(
        SESSION_ID,
        "one_star_newcomer",
    )

    assert "# Dossier · Mara Vale" in dossier
    assert "a scarlet coat and iron-gray braid" in dossier
    assert "## Your Control & Perspective" in dossier
    assert newcomer.player_guidance in dossier
    assert "## How You Think & Feel" not in dossier
    assert "## Your Backstory" not in dossier


def test_playable_dossiers_exclude_agent_portrayal_and_keep_known_interior(
    bridge: EngineBridge,
) -> None:
    checkpoint = bridge.load_latest(SESSION_ID)
    by_id = {
        character.character_id: character
        for character in checkpoint.characters
    }

    for character_id in (
        "the_master",
        "halcyon_of_the_gilded_march",
    ):
        character = by_id[character_id]
        dossier = bridge.build_character_dossier(SESSION_ID, character_id)

        assert character.player_guidance in dossier
        assert character.personality not in dossier
        assert "## How You Think & Feel" not in dossier
        if character.backstory:
            assert character.backstory in dossier
        if character.known_context:
            assert character.known_context in dossier
        for secret in character.private_state.secrets:
            assert secret in dossier

    master_dossier = bridge.build_character_dossier(SESSION_ID, "the_master")
    assert "## Your Backstory" not in master_dossier
    assert "## What You Keep To Yourself" not in master_dossier
    assert by_id["the_master"].descriptions.private not in master_dossier

    halcyon_dossier = bridge.build_character_dossier(
        SESSION_ID,
        "halcyon_of_the_gilded_march",
    )
    assert "Do not play him" not in halcyon_dossier


def test_seed_fields_respect_character_awareness_boundaries() -> None:
    checkpoint = _seed_checkpoint()
    by_id = {
        character.character_id: character
        for character in checkpoint.characters
    }
    for character in checkpoint.characters:
        known = character.known_context.lower()
        assert "never invent" not in known
        assert "do not " not in known
        assert "should never" not in known
        assert "unless such a means" not in known

    master = by_id["the_master"]
    assert master.backstory == ""
    assert master.private_state.secrets == []

    renna = by_id["renna_holt"]
    assert not any(
        "potential" in secret.lower() or "talent" in secret.lower()
        for secret in renna.private_state.secrets
    )
    assert renna.descriptions.private

    iselle = by_id["iselle_the_guide"]
    assert not any(
        "not a person" in secret.lower()
        or "scripted interface" in secret.lower()
        for secret in iselle.private_state.secrets
    )

    warden = by_id["warden_of_the_eighth"]
    assert warden.entity_kind.value == "hazard"
    assert warden.is_playable is False
    assert warden.player_guidance == ""
    assert warden.backstory == ""
    assert warden.personality == ""
    assert warden.private_state.goals == []
    assert warden.private_state.current_objectives == []
    assert warden.private_state.secrets == []
    assert warden.private_state.intentions_enabled is False
    assert warden.known_context == ""


def _render_foreground_agent(
    checkpoint: CheckpointFile,
    character_id: str,
) -> list[dict[str, str]]:
    character = next(
        item for item in checkpoint.characters
        if item.character_id == character_id
    )
    return PromptManager("app/prompts").render_conversation(
        "agent",
        history=[],
        agent_ruleset_system_addon="",
        **build_character_packet(character, checkpoint),
        **build_character_state(character, checkpoint),
        world_context=build_world_context(character, checkpoint),
        pending_observations_block=format_pending_observations_block(character),
        mode_header=AGENT_TURN_HEADER,
        mode_block=format_agent_turn_body(frame="foreground"),
    )


def test_birth_one_star_learns_tutorial_only_from_witnessed_dialogue() -> None:
    checkpoint = _seed_checkpoint()
    renna = next(
        character for character in checkpoint.characters
        if character.character_id == "renna_holt"
    )

    initial = _render_foreground_agent(checkpoint, renna.character_id)
    assert [message["role"] for message in initial] == ["system", "user"]
    initial_system = initial[0]["content"]
    initial_user = initial[1]["content"]
    for pre_tutorial_leak in (
        "master",
        "tower",
        "climb",
        "deployment",
        "summoned hero",
        "one-star",
    ):
        assert pre_tutorial_leak not in initial_system.casefold()
        assert pre_tutorial_leak not in initial_user.casefold()

    renna.pending_observations.append(
        "Iselle says, \"An unseen Master brought you here. You are expected "
        "to climb the Tower, and entering the deployment gate commits you "
        "until the objective is cleared.\""
    )
    instructed = _render_foreground_agent(checkpoint, renna.character_id)
    assert instructed[0]["content"] == initial_system
    instructed_user = instructed[-1]["content"].casefold()
    assert "since your last response" in instructed_user
    for witnessed_fact in ("unseen master", "climb the tower", "deployment gate"):
        assert witnessed_fact in instructed_user
