"""Tests for the v2 importer additions: knowledge envelopes and
preservation analysis. These exercise the pure-Python bits — LLM calls
are mocked or skipped."""

from __future__ import annotations


from app.engine.story_importer import (
    IMPORTER_VERSION,
    _serialize_checkpoint_for_analysis,
    build_checkpoint,
)
from app.schemas.checkpoint import ImportAnalysis
from app.schemas.import_extraction import (
    CharacterExtraction,
    CharacterDescriptionsExtraction,
    CharacterKnowledgeEnvelope,
    CharacterKnowledgeListExtraction,
    CharacterListExtraction,
    PrivateStateExtraction,
    PublicSheetExtraction,
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
    )


def _roster() -> CharacterListExtraction:
    return CharacterListExtraction(characters=[
        CharacterExtraction(
            character_id="regent",
            name="Emeric Hale",
            status="active",
            location="",
            is_playable=False,
            public_sheet=PublicSheetExtraction(
                role="Regent", appearance="", faction="",
            ),
            descriptions=CharacterDescriptionsExtraction(
                public="Emeric Hale is the publicly recognized Regent of Mirenza.",
                private="Emeric knows the Aetheri conspiracy and keeps it hidden.",
            ),
            private_state=PrivateStateExtraction(
                goals=["preserve Mirenza"],
                current_objectives=[],
                secrets=["knows the Aetheri truth"],
                intentions_enabled=True,
                tick_cues=[],
            ),
            backstory="Twenty-five years governing the sealed state.",
            personality="",
        ),
        CharacterExtraction(
            character_id="lira",
            name="Lira Fontaine",
            status="active",
            location="",
            is_playable=False,
            public_sheet=PublicSheetExtraction(
                role="Court Liaison", appearance="", faction="",
            ),
            descriptions=CharacterDescriptionsExtraction(
                public="Lira Fontaine is a court liaison assigned to foreign guests.",
                private="Lira privately longs to leave Mirenza.",
            ),
            private_state=PrivateStateExtraction(
                goals=["see the world outside"],
                current_objectives=[],
                secrets=["never leaves Mirenza"],
                intentions_enabled=False,
                tick_cues=[],
            ),
            backstory="",
            personality="",
        ),
    ])


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
            _world(), _roster(), _envelopes(), "test_story",
        )
        by_id = {c.character_id: c for c in ckpt.characters}
        assert "testimony crystals" in by_id["regent"].known_context
        assert "testimony crystals" not in by_id["lira"].known_context
        assert by_id["regent"].descriptions.public.startswith(
            "Emeric Hale is the publicly recognized Regent"
        )
        assert "Aetheri conspiracy" in by_id["regent"].descriptions.private

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
                _world(), roster, partial, "test_story",
            )
        by_id = {c.character_id: c for c in ckpt.characters}
        assert by_id["lira"].known_context == ""
        assert any("No knowledge envelope for lira" in r.message for r in caplog.records)

    def test_stamps_current_version(self):
        ckpt = build_checkpoint(
            _world(), _roster(), _envelopes(), "test_story",
        )
        assert ckpt.importer_version == IMPORTER_VERSION
        # Any bump here means the extraction contract changed — re-import
        # in-flight stories if you rely on a new field.
        assert IMPORTER_VERSION == "v11"


class TestImportAnalysisSchema:
    def test_defaults_to_none_on_checkpoint(self):
        ckpt = build_checkpoint(
            _world(), _roster(), _envelopes(), "test_story",
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
            _world(), _roster(), _envelopes(), "test_story",
        )
        rendered = _serialize_checkpoint_for_analysis(ckpt)
        assert "political intrigue" in rendered  # setting.genre
        assert "Public lore describing" in rendered  # lore
        assert "Aetheri were deliberately" in rendered  # hidden_lore
        assert "borders are closed" in rendered  # facts
        assert "Regent knows the truth" in rendered  # hidden_facts
        assert "regent" in rendered  # character id
        assert "testimony crystals" in rendered  # known_context

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
        )
        empty_roster = CharacterListExtraction(characters=[])
        empty_envelopes = CharacterKnowledgeListExtraction(envelopes=[])
        ckpt = build_checkpoint(
            empty_world, empty_roster, empty_envelopes, "empty",
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


class TestLocationSeedPush:
    """v11-r10: every author-seeded NPC with a known starting
    location gets a `[your own action] <Name> at <location>.`
    push into `pending_observations`. This is the inbox-level
    counterpart to the importer-side `is_playable=true` path (player
    characters are skipped because humans don't read pending_observations
    through an LLM).

    Without this seed, an NPC's very first agent dispatch arrives
    with no location signal once the on-stage agent body's
    historical location block is gone — the inbox is
    the only channel left."""

    def _placed_roster(self) -> CharacterListExtraction:
        return CharacterListExtraction(characters=[
            CharacterExtraction(
                character_id="regent",
                name="Emeric Hale",
                status="active",
                location="Regent's Study",
                is_playable=False,
                public_sheet=PublicSheetExtraction(
                    role="Regent", appearance="", faction="",
                ),
                descriptions=CharacterDescriptionsExtraction(
                    public="Emeric Hale is the publicly recognized Regent of Mirenza.",
                    private="Emeric is the private keeper of sealed intelligence.",
                ),
                private_state=PrivateStateExtraction(
                    goals=[], current_objectives=[], secrets=[],
                    intentions_enabled=False,
                    tick_cues=[],
                ),
                backstory="", personality="",
            ),
            CharacterExtraction(
                character_id="lira",
                name="Lira Fontaine",
                status="active",
                location="Great Hall",
                is_playable=False,
                public_sheet=PublicSheetExtraction(
                    role="Liaison", appearance="", faction="",
                ),
                descriptions=CharacterDescriptionsExtraction(
                    public="Lira Fontaine is a court liaison assigned to guests.",
                    private="Lira privately doubts the court's isolation.",
                ),
                private_state=PrivateStateExtraction(
                    goals=[], current_objectives=[], secrets=[],
                    intentions_enabled=False,
                    tick_cues=[],
                ),
                backstory="", personality="",
            ),
        ])

    def test_npc_with_location_gets_seed_push(self):
        ckpt = build_checkpoint(
            _world(),
            self._placed_roster(),
            _envelopes(),
            "test_story",
        )
        by_id = {c.character_id: c for c in ckpt.characters}
        regent = by_id["regent"]
        # Single seed entry with the location label.
        assert regent.pending_observations == [
            "[your own action] Emeric Hale at Regent's Study.",
        ]

    def test_seed_uses_location_label_verbatim(self):
        ckpt = build_checkpoint(
            _world(),
            self._placed_roster(),
            _envelopes(),
            "test_story",
        )
        by_id = {c.character_id: c for c in ckpt.characters}
        lira = by_id["lira"]
        assert lira.pending_observations == [
            "[your own action] Lira Fontaine at Great Hall.",
        ]

    def test_player_character_is_not_seeded(self):
        """Players never read pending_observations through an LLM —
        seeding their inbox would surface engine-internal text on a
        future flow that ever queried it for a player render."""
        roster = self._placed_roster()
        roster.characters[0].is_playable = True  # regent is now player
        ckpt = build_checkpoint(
            _world(), roster, _envelopes(),
            "test_story",
        )
        by_id = {c.character_id: c for c in ckpt.characters}
        assert by_id["regent"].pending_observations == []
        # NPC sibling still seeded.
        assert by_id["lira"].pending_observations == [
            "[your own action] Lira Fontaine at Great Hall.",
        ]

    def test_npc_without_location_is_not_seeded(self):
        """An author-seeded NPC with no resolvable location (legacy
        importer path, schema-default empty string) gets no seed —
        we don't want to interpolate `at .` or `at unknown.`. They'll
        get a `[your own action]` push the first time they actually
        move."""
        roster = self._placed_roster()
        roster.characters[0].location = ""  # regent unsited
        ckpt = build_checkpoint(
            _world(), roster, _envelopes(),
            "test_story",
        )
        by_id = {c.character_id: c for c in ckpt.characters}
        assert by_id["regent"].pending_observations == []
