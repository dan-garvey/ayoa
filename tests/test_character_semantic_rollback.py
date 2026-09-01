"""Contracts for restoring authored character material into actor facts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.engine.context_builder import (
    build_character_self_packet,
    build_visible_self_packet,
)
from app.schemas.characters import (
    ActorRecord,
    CharacterRecord,
    is_non_social_hazard,
    is_player_authored_slot,
)
from app.schemas.checkpoint import CheckpointFile


ROOT = Path(__file__).resolve().parent.parent
STORY_CHECKPOINT_PATHS = tuple(
    sorted(ROOT.glob("app/storage/stories/*/ckpt_0000.json"))
)
SOURCE_CHECKPOINT_PATH = (
    ROOT / "app/storage/stories/one_star_ascension_s1/ckpt_0000.json"
)
PROMOTION_CHECKPOINT_PATH = (
    ROOT / "app/storage/stories/one_star_ascension_s1_promotion_playtest/ckpt_0000.json"
)
RETIRED_CONTROL_MARKERS = (
    "private_carry",
    "protect the opening's exact authored cast",
    "response opportunity is already spent",
    "the engine should render",
    "the protagonist's ethical project",
)


def _load(path: Path) -> CheckpointFile:
    return CheckpointFile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _tracked_actors(checkpoint: CheckpointFile) -> list[CharacterRecord]:
    return [
        character
        for character in checkpoint.characters
        if (
            character.actor is not None
            and not is_player_authored_slot(character)
            and not is_non_social_hazard(character)
        )
    ]


def test_seeded_actor_material_reaches_only_its_owner_packet() -> None:
    """Rich facts stay private to the actor turn, never exterior perception."""
    for path in STORY_CHECKPOINT_PATHS:
        checkpoint = _load(path)
        actors = _tracked_actors(checkpoint)
        if not actors:
            assert not checkpoint.characters, path
            continue
        assert actors, path
        for character in actors:
            assert character.actor is not None
            facts = [" ".join(fact.text.split()) for fact in character.actor.facts]
            assert facts, (path, character.character_id)
            owner_packet = build_character_self_packet(character, checkpoint)
            visible_packet = build_visible_self_packet(character, checkpoint)
            owner_fact_lines = [
                line.removeprefix("- ")
                for line in owner_packet.splitlines()
                if line.startswith("- ")
            ]
            assert len(owner_fact_lines) == len(facts)
            assert all(fact not in visible_packet for fact in facts)
            assert all(fact not in visible_packet for fact in owner_fact_lines)
            assert all(
                marker not in owner_packet.casefold()
                for marker in RETIRED_CONTROL_MARKERS
            )


@pytest.mark.parametrize(
    "retired_field",
    ("backstory", "personality", "known_context", "private_state", "private_carry"),
)
def test_current_actor_schema_refuses_retired_profile_and_carry_fields(
    retired_field: str,
) -> None:
    payload = CharacterRecord(
        character_id="actor",
        name="Actor",
        actor=ActorRecord(),
    ).model_dump(mode="json")
    payload[retired_field] = "retired"

    with pytest.raises(ValidationError):
        CharacterRecord.model_validate(payload)


def test_promotion_fixture_extends_every_restored_source_actor_record() -> None:
    source = _load(SOURCE_CHECKPOINT_PATH)
    promotion = _load(PROMOTION_CHECKPOINT_PATH)
    promotion_by_id = {
        character.character_id: character for character in promotion.characters
    }

    for source_character in _tracked_actors(source):
        promotion_character = promotion_by_id[source_character.character_id]
        assert source_character.actor is not None
        assert promotion_character.actor is not None
        source_facts = source_character.actor.facts
        assert promotion_character.actor.facts[: len(source_facts)] == source_facts
