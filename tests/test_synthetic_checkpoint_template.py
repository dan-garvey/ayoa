from __future__ import annotations

import json
from pathlib import Path

from app.schemas.checkpoint import CheckpointFile


TEMPLATE_DIR = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "storage"
    / "story_templates"
    / "synthetic_checkpoint"
)
TEMPLATE_PATH = TEMPLATE_DIR / "ckpt_0000.json"
NOTES_PATH = TEMPLATE_DIR / "DESIGN_NOTES.md"
GITIGNORE_PATH = Path(__file__).resolve().parent.parent / ".gitignore"

SHIPPED_STORY_FIXTURES = (
    "dating_villa_s1",
    "the_unblessed_summon",
)


def test_synthetic_checkpoint_template_validates() -> None:
    raw = json.loads(TEMPLATE_PATH.read_text())
    checkpoint = CheckpointFile.model_validate(raw)

    assert checkpoint.schema_version == "6.0"
    assert "config" not in raw
    assert not hasattr(checkpoint, "config")
    assert checkpoint.session.session_id == "synthetic_checkpoint_template"
    assert checkpoint.session.story_id == "synthetic_checkpoint_template"
    assert checkpoint.characters == []
    assert checkpoint.canonical_events == []
    assert checkpoint.session_conversation == []
    assert checkpoint.world_state.facts == []
    assert checkpoint.world_state.hidden_facts == []
    assert checkpoint.session.config.settings.ruleset_id == "narrative"
    assert "discriminator" not in checkpoint.session.config.models.model_dump()
    assert not hasattr(checkpoint, "importer_version")
    assert not hasattr(checkpoint, "import_analysis")


def test_synthetic_checkpoint_notes_cover_authoring_contract() -> None:
    notes = NOTES_PATH.read_text()

    assert "app/storage/stories/<story_id>/ckpt_0000.json" in notes
    assert "the_unblessed_summon" in notes
    assert ".gitignore" in notes
    assert "player_primer" in notes
    assert "importer_version" in notes


def test_storage_fixture_gitignore_policy() -> None:
    ignore_lines = {
        line.strip()
        for line in GITIGNORE_PATH.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "app/storage/sessions/*" in ignore_lines
    assert "app/storage/playtest_reports/" in ignore_lines
    assert "app/storage/stories/*" in ignore_lines
    assert "tests/test_unblessed_summon_checkpoint.py" not in ignore_lines

    for story_id in SHIPPED_STORY_FIXTURES:
        assert f"!app/storage/stories/{story_id}/" in ignore_lines
        assert f"!app/storage/stories/{story_id}/ckpt_0000.json" in ignore_lines
