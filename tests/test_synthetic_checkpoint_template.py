from __future__ import annotations

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


def test_synthetic_checkpoint_template_validates() -> None:
    checkpoint = CheckpointFile.model_validate_json(TEMPLATE_PATH.read_text())

    assert checkpoint.schema_version == "4.0"
    assert checkpoint.session.session_id == "synthetic_checkpoint_template"
    assert checkpoint.session.story_id == "synthetic_checkpoint_template"
    assert checkpoint.characters == []
    assert checkpoint.canonical_events == []
    assert checkpoint.session_conversation == []
    assert checkpoint.world_state.facts == []
    assert checkpoint.world_state.hidden_facts == []
    assert checkpoint.session.config.settings.ruleset_id == "narrative"
    assert checkpoint.config.settings.ruleset_id == "narrative"
    assert not hasattr(checkpoint, "importer_version")
    assert not hasattr(checkpoint, "import_analysis")


def test_synthetic_checkpoint_notes_cover_authoring_contract() -> None:
    notes = NOTES_PATH.read_text()

    assert "app/storage/stories/<story_id>/ckpt_0000.json" in notes
    assert "the_unblessed_summon" in notes
    assert "player_primer" in notes
    assert "importer_version" in notes
