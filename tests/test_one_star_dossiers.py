"""Rendered player-dossier audience contracts for One-Star Ascension."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.bot.engine_bridge import EngineBridge
from app.engine.context_builder import (
    build_character_turn_identity_seed,
    build_character_turn_request_packet,
    format_pending_observations_block,
)
from app.engine.one_star_adapter import load_one_star_account
from app.engine.prompt_manager import PromptManager
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
    owner, account = load_one_star_account(checkpoint)
    account.config.visual_novel_presentation = None
    owner.mechanics["one_star_account"] = account.model_dump(mode="json")
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
    assert newcomer.actor is None
    assert newcomer.public_sheet.public_context == ""

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


def test_playable_dossiers_render_their_own_actor_facts(
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
        assert character.actor is not None
        assert "## What You Know Of Yourself" in dossier
        for fact in character.actor.facts:
            assert fact.text in dossier

    master_dossier = bridge.build_character_dossier(SESSION_ID, "the_master")
    assert "## Your Backstory" not in master_dossier
    assert "## What You Keep To Yourself" not in master_dossier

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
        if character.actor is None:
            continue
        actor_text = "\n".join(fact.text for fact in character.actor.facts).lower()
        assert "never invent" not in actor_text
        assert "should never" not in actor_text

    master = by_id["the_master"]
    assert master.actor is not None

    renna = by_id["renna_holt"]
    assert renna.actor is not None
    assert renna.actor.facts

    iselle = by_id["iselle_the_guide"]
    assert iselle.actor is not None
    assert iselle.actor.facts

    warden = by_id["warden_of_the_eighth"]
    assert warden.entity_kind.value == "hazard"
    assert warden.is_playable is False
    assert warden.player_guidance == ""
    assert warden.actor is None


def _render_foreground_agent(
    checkpoint: CheckpointFile,
    character_id: str,
) -> list[dict[str, str]]:
    character = next(
        item for item in checkpoint.characters
        if item.character_id == character_id
    )
    return PromptManager("app/prompts").render_conversation(
        "agent_turn",
        history=[],
        ruleset_guidance="",
        request_packet=build_character_turn_request_packet(
            format_pending_observations_block(character),
            identity_seed=build_character_turn_identity_seed(character, checkpoint),
        ),
    )


@pytest.mark.parametrize("character_id", ("renna_holt", "edren_marr"))
def test_birth_one_star_learns_tutorial_only_from_witnessed_dialogue(
    character_id: str,
) -> None:
    checkpoint = _seed_checkpoint()
    character = next(
        character for character in checkpoint.characters
        if character.character_id == character_id
    )

    initial = _render_foreground_agent(checkpoint, character.character_id)
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

    character.pending_observations.append(
        "Iselle says, \"An unseen Master brought you here. You are expected "
        "to climb the Tower, and entering the deployment gate commits you "
        "until the objective is cleared.\""
    )
    instructed = _render_foreground_agent(checkpoint, character.character_id)
    assert instructed[0]["content"] == initial_system
    instructed_user = instructed[-1]["content"].casefold()
    for witnessed_fact in ("unseen master", "climb the tower", "deployment gate"):
        assert witnessed_fact in instructed_user
