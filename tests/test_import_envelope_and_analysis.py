"""Tests for the v2 importer additions: knowledge envelopes and
preservation analysis. These exercise the pure-Python bits — LLM calls
are mocked or skipped."""

from __future__ import annotations

import pytest

from app.engine.story_importer import (
    IMPORTER_VERSION,
    _serialize_checkpoint_for_analysis,
    build_checkpoint,
)
from app.schemas.checkpoint import CheckpointFile, ImportAnalysis
from app.schemas.import_extraction import (
    CharacterExtraction,
    CharacterKnowledgeEnvelope,
    CharacterKnowledgeListExtraction,
    CharacterListExtraction,
    LocationsExtraction,
    OpeningExtraction,
    PrivateStateExtraction,
    PublicSheetExtraction,
    SceneExtraction,
    SettingExtraction,
    WorldExtraction,
)


from app.schemas.import_extraction import PhysicsRulesetExtraction


def _world() -> WorldExtraction:
    return WorldExtraction(
        setting=SettingExtraction(
            genre="political intrigue",
            era="",
            tone="tense",
            premise="Full authorial premise including the Aetheri conspiracy.",
        ),
        lore="Public lore describing the known world.",
        facts=["The borders are closed.", "The Athenaeum is sealed."],
        physics_ruleset=PhysicsRulesetExtraction(
            strength_limits="human_baseline", magic_enabled=False,
        ),
        narrative_rules="",
        hidden_lore="The Aetheri were deliberately erased.",
        hidden_facts=["Regent knows the truth.", "Nessa is a Keeper operative."],
        locations=LocationsExtraction(
            current_scene_id="hall",
            scene_graph=[SceneExtraction(
                scene_id="hall", name="Hall",
                description="", connected_to=[],
            )],
        ),
    )


def _roster() -> CharacterListExtraction:
    return CharacterListExtraction(characters=[
        CharacterExtraction(
            character_id="regent",
            name="Emeric Hale",
            status="active",
            location="",
            is_player=False,
            public_sheet=PublicSheetExtraction(
                role="Regent", appearance="", faction="",
            ),
            private_state=PrivateStateExtraction(
                goals=["preserve Mirenza"],
                current_objectives=[],
                secrets=["knows the Aetheri truth"],
                intentions_enabled=True,
            ),
            backstory="Twenty-five years governing the sealed state.",
            personality="",
        ),
        CharacterExtraction(
            character_id="lira",
            name="Lira Fontaine",
            status="active",
            location="",
            is_player=False,
            public_sheet=PublicSheetExtraction(
                role="Court Liaison", appearance="", faction="",
            ),
            private_state=PrivateStateExtraction(
                goals=["see the world outside"],
                current_objectives=[],
                secrets=["never leaves Mirenza"],
                intentions_enabled=False,
            ),
            backstory="",
            personality="",
        ),
    ])


def _opening() -> OpeningExtraction:
    return OpeningExtraction(text="You cross into Mirenza...")


def _envelopes() -> CharacterKnowledgeListExtraction:
    return CharacterKnowledgeListExtraction(envelopes=[
        CharacterKnowledgeEnvelope(
            character_id="regent",
            known_context="As Regent, you read the intelligence reports nobody else sees. You know the Aetheri testimony crystals are real.",
        ),
        CharacterKnowledgeEnvelope(
            character_id="lira",
            known_context="You know what every court-raised Mirenzan knows: the Closure began forty years ago, the Athenaeum was once great, the Isolation Mandate hangs over every foreign encounter. Nothing deeper.",
        ),
    ])


class TestBuildCheckpointEnvelope:
    def test_envelope_lands_on_character(self):
        ckpt = build_checkpoint(
            _world(), _roster(), _opening(), _envelopes(), "test_story",
        )
        by_id = {c.character_id: c for c in ckpt.characters}
        assert "testimony crystals" in by_id["regent"].known_context
        assert "testimony crystals" not in by_id["lira"].known_context

    def test_missing_envelope_falls_back_to_empty(self, caplog):
        """If the envelope extraction silently omits a character, the
        importer should still produce a record — with empty known_context —
        and log a warning."""
        roster = _roster()
        # envelopes list has regent but not lira
        partial = CharacterKnowledgeListExtraction(envelopes=[
            CharacterKnowledgeEnvelope(
                character_id="regent",
                known_context="regent knows things",
            ),
        ])
        with caplog.at_level("WARNING"):
            ckpt = build_checkpoint(
                _world(), roster, _opening(), partial, "test_story",
            )
        by_id = {c.character_id: c for c in ckpt.characters}
        assert by_id["lira"].known_context == ""
        assert any("No knowledge envelope for lira" in r.message for r in caplog.records)

    def test_stamps_current_version(self):
        ckpt = build_checkpoint(
            _world(), _roster(), _opening(), _envelopes(), "test_story",
        )
        assert ckpt.importer_version == IMPORTER_VERSION
        # Any bump here means the extraction contract changed — re-import
        # in-flight stories if you rely on a new field.
        assert IMPORTER_VERSION == "v7"


class TestImportAnalysisSchema:
    def test_defaults_to_none_on_checkpoint(self):
        ckpt = build_checkpoint(
            _world(), _roster(), _opening(), _envelopes(), "test_story",
        )
        assert ckpt.import_analysis is None

    def test_round_trips(self):
        analysis = ImportAnalysis(
            source_chars=1000,
            source_words=150,
            output_chars=800,
            output_words=110,
            coverage_rating="high",
            dropped_topics=["thing A"],
            compressed_topics=["thing B"],
            preservation_notes="looks good",
            duration_s=45.2,
            model="claude-sonnet-4-6",
        )
        serialized = analysis.model_dump_json()
        restored = ImportAnalysis.model_validate_json(serialized)
        assert restored == analysis

    def test_coverage_rating_accepts_unknown(self):
        # Safety: when the LLM returns something weird, the coercion in
        # run_preservation_analysis maps it to "unknown" — the schema must
        # allow that value.
        a = ImportAnalysis(coverage_rating="unknown")
        assert a.coverage_rating == "unknown"


class TestSerializeForAnalysis:
    def test_includes_main_text_fields(self):
        ckpt = build_checkpoint(
            _world(), _roster(), _opening(), _envelopes(), "test_story",
        )
        rendered = _serialize_checkpoint_for_analysis(ckpt)
        assert "political intrigue" in rendered  # setting.genre
        assert "Public lore describing" in rendered  # lore
        assert "Aetheri were deliberately" in rendered  # hidden_lore
        assert "borders are closed" in rendered  # facts
        assert "Regent knows the truth" in rendered  # hidden_facts
        assert "Emeric Hale" in rendered  # character name
        assert "testimony crystals" in rendered  # known_context
        assert "You cross into Mirenza" in rendered  # opening

    def test_missing_fields_are_silently_skipped(self):
        empty_world = WorldExtraction(
            setting=SettingExtraction(genre="", era="", tone="", premise=""),
            lore="",
            facts=[],
            physics_ruleset=PhysicsRulesetExtraction(
                strength_limits="human_baseline", magic_enabled=False,
            ),
            narrative_rules="",
            hidden_lore="",
            hidden_facts=[],
            locations=LocationsExtraction(
                current_scene_id="hall",
                scene_graph=[SceneExtraction(
                    scene_id="hall", name="Hall",
                    description="", connected_to=[],
                )],
            ),
        )
        empty_roster = CharacterListExtraction(characters=[])
        empty_opening = OpeningExtraction(text="")
        empty_envelopes = CharacterKnowledgeListExtraction(envelopes=[])
        ckpt = build_checkpoint(
            empty_world, empty_roster, empty_opening, empty_envelopes, "empty",
        )
        # Should not raise, should produce a (possibly sparse) string
        rendered = _serialize_checkpoint_for_analysis(ckpt)
        assert isinstance(rendered, str)


class TestCharacterRecordEnvelopeDefault:
    def test_known_context_defaults_empty(self):
        from app.schemas.characters import CharacterRecord, PublicSheet

        c = CharacterRecord(
            character_id="x", name="X",
            public_sheet=PublicSheet(role="r"),
        )
        assert c.known_context == ""
