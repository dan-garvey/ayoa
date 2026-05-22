from __future__ import annotations

import sqlite3

import pytest

from app.engine.content_pack_compiler import (
    DEFAULT_PRIVATE_PACK_DIR,
    CompiledContentPackReader,
    CompiledContentPackWriter,
    assess_card_runtime_gate,
    default_compiled_pack_path,
)


PROTECTED_EXCERPT = "PROTECTED MODULE EXCERPT"
RAW_SOURCE_PATH = "/private/table/curse_of_strahd.pdf"


def test_compiled_pack_writer_persists_inventory_cards_provenance_and_manifest(
    tmp_path,
):
    db_path = tmp_path / "synthetic_pack.sqlite"
    writer = CompiledContentPackWriter(
        db_path,
        pack_id="synthetic-curse",
        pack_version="0.0.1",
        source_fingerprint="sha256:redacted-source",
        importer_version="test-redacted-v1",
    )

    manifest = writer.write_pack(
        pages=[
            {
                "page_id": "page-001",
                "source_asset_id": "src-page-001",
                "pdf_page_index": 0,
                "printed_page_label": "1",
                "source_sha256": "hash-page-001",
                "alignment_status": "reviewed",
                "confidence": 0.98,
                "review_status": "approved",
            }
        ],
        cards=[
            {
                "ref": "area.entry",
                "card_kind": "location_card",
                "visibility": "hidden",
                "title": "Entry Gallery",
                "summary": "A cold gallery with a visible north door.",
                "body": "Reviewed redacted room notes for runtime lookup.",
                "spoiler_class": "low",
                "confidence": 0.95,
                "review_status": "approved",
                "aliases": ["entry", "gallery"],
                "provenance": [
                    {
                        "source_asset_id": "src-page-001",
                        "page_id": "page-001",
                        "span_id": "span-001",
                        "bbox": [0, 0, 10, 10],
                        "method": "synthetic-fixture",
                        "confidence": 0.95,
                        "human_review_status": "approved",
                    }
                ],
                "field_provenance": {
                    "summary": [
                        {
                            "source_asset_id": "src-page-001",
                            "page_id": "page-001",
                            "span_id": "span-summary",
                            "method": "human-review",
                            "confidence": 0.99,
                            "human_review_status": "approved",
                        }
                    ]
                },
            }
        ],
        aliases=[
            {
                "alias": "front door",
                "ref": "area.entry",
                "kind": "navigation",
                "confidence": 0.90,
                "review_status": "approved",
            }
        ],
        source_page_count=1,
    )

    assert manifest.pack_id == "synthetic-curse"
    assert manifest.source_page_count == 1
    assert manifest.compiled_page_count == 1
    assert manifest.card_count == 1
    assert manifest.ready_count == 1
    assert manifest.blocked_count == 0

    reader = CompiledContentPackReader(db_path)
    assert reader.manifest() == manifest
    assert [page.page_id for page in reader.list_pages()] == ["page-001"]
    assert sorted(alias.alias for alias in reader.list_aliases(ref="area.entry")) == [
        "entry",
        "front door",
        "gallery",
    ]
    cards = reader.load_cards()
    assert len(cards) == 1
    assert cards[0].ref == "area.entry"
    assert cards[0].gate_status == "runtime_ready"
    assert cards[0].provenance[0].source_asset_id == "src-page-001"
    assert cards[0].field_provenance["summary"][0].span_id == "span-summary"

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "pack_metadata",
        "page_inventory",
        "content_cards",
        "card_provenance",
        "card_field_provenance",
        "content_aliases",
        "coverage_manifest",
    } <= tables
    with sqlite3.connect(db_path) as conn:
        field_rows = conn.execute(
            "SELECT field_name, span_id FROM card_field_provenance"
        ).fetchall()
    assert field_rows == [("summary", "span-summary")]


def test_coverage_gates_block_low_confidence_and_high_spoiler_runtime_cards(
    tmp_path,
):
    db_path = tmp_path / "synthetic_pack.sqlite"
    writer = CompiledContentPackWriter(db_path, pack_id="synthetic")

    manifest = writer.write_pack(
        pages=[
            {
                "page_id": "page-001",
                "source_asset_id": "src-page-001",
                "review_status": "approved",
            }
        ],
        cards=[
            {
                "ref": "safe.ready",
                "summary": "Ready redacted card.",
                "body": "Safe reviewed text.",
                "confidence": 0.91,
                "review_status": "approved",
                "spoiler_class": "none",
            },
            {
                "ref": "review.flagged",
                "summary": "Needs human review but is not blocked.",
                "body": "Synthetic redacted text.",
                "confidence": 0.93,
                "review_status": "needs_review",
                "spoiler_class": "low",
            },
            {
                "ref": "ocr.low",
                "summary": "Low-confidence OCR candidate.",
                "body": "Synthetic redacted text.",
                "confidence": 0.42,
                "review_status": "approved",
                "spoiler_class": "low",
            },
            {
                "ref": "future.secret",
                "summary": "Future reveal without an approved trigger.",
                "body": "Synthetic redacted text.",
                "confidence": 0.96,
                "review_status": "approved",
                "spoiler_class": "high",
            },
        ],
    )

    assert manifest.ready_count == 1
    assert manifest.flagged_count == 1
    assert manifest.blocked_count == 2
    assert manifest.low_confidence_count == 1
    assert manifest.high_spoiler_count == 1

    reader = CompiledContentPackReader(db_path)
    assert [card.ref for card in reader.load_cards()] == ["safe.ready"]
    assert [card.ref for card in reader.load_cards(include_flagged=True)] == [
        "review.flagged",
        "safe.ready",
    ]
    assert reader.load_cards(refs=["ocr.low"], include_flagged=True) == []
    assert reader.load_cards(refs=["future.secret"], include_flagged=True) == []


def test_high_spoiler_with_reviewed_reveal_trigger_can_pass_gate():
    gate = assess_card_runtime_gate(
        {
            "ref": "foreshadowed.secret",
            "summary": "A reviewed secret with a concrete reveal condition.",
            "body": "Synthetic redacted text.",
            "confidence": 0.88,
            "review_status": "approved",
            "spoiler_class": "high",
            "reveal_trigger": "Reveal only after the marked handout is found.",
            "content_hash": "hash-secret",
        }
    )

    assert gate.status == "runtime_ready"
    assert gate.allowed is True
    assert gate.reasons == []


def test_coverage_manifest_reports_expected_domains_and_blocking_issues(tmp_path):
    db_path = tmp_path / "synthetic_pack.sqlite"
    writer = CompiledContentPackWriter(db_path, pack_id="synthetic")

    manifest = writer.write_pack(
        pages=[
            {
                "page_id": "page-001",
                "source_asset_id": "src-page-001",
                "review_status": "approved",
            }
        ],
        cards=[
            {
                "ref": "area.entry",
                "card_kind": "keyed_area",
                "summary": "Reviewed keyed area.",
                "body": "Synthetic redacted text.",
                "confidence": 0.90,
                "review_status": "approved",
                "metadata": {"domain": "keyed_areas"},
            },
            {
                "ref": "front.strahd",
                "card_kind": "front_dossier",
                "summary": "Reviewed front dossier.",
                "body": "Synthetic redacted text.",
                "confidence": 0.90,
                "review_status": "approved",
                "metadata": {"domain": "fronts"},
            },
        ],
        expected_domain_counts={
            "sections": 1,
            "keyed_areas": 2,
            "maps": 1,
            "tables": 1,
            "statblocks": 1,
            "traps": 1,
            "loot": 1,
            "handouts": 1,
            "fronts": 1,
        },
        coverage_issues=[
            {
                "issue_kind": "unresolved_ref",
                "domain": "keyed_areas",
                "ref": "area.entry",
                "target_ref": "area.missing",
                "message": "Synthetic missing exit.",
            },
            {
                "issue_kind": "malformed_table",
                "domain": "tables",
                "ref": "table.bad",
                "severity": "warning",
                "message": "Synthetic table row has a bad range.",
            },
            {
                "issue_kind": "blocked_section",
                "domain": "sections",
                "ref": "sec.blocked",
                "message": "Synthetic blocked section.",
            },
            {
                "issue_kind": "invalid_statblock",
                "domain": "statblocks",
                "ref": "stat.bad",
                "message": "Synthetic statblock lacks actions.",
            },
            {
                "issue_kind": "unreviewed_topology",
                "domain": "maps",
                "ref": "map.unreviewed",
                "message": "Synthetic map topology is unreviewed.",
            },
        ],
    )

    assert manifest.domain_coverage["keyed_areas"].expected_count == 2
    assert manifest.domain_coverage["keyed_areas"].found_count == 1
    assert manifest.domain_coverage["keyed_areas"].missing_count == 1
    assert manifest.domain_coverage["fronts"].ready_count == 1
    assert manifest.domain_coverage["tables"].warning_count == 1
    assert manifest.domain_coverage["sections"].blocking_issue_count == 1
    assert manifest.unresolved_ref_count == 1
    assert manifest.malformed_table_count == 1
    assert manifest.blocked_section_count == 1
    assert manifest.invalid_statblock_count == 1
    assert manifest.unreviewed_topology_count == 1
    assert "blocking_coverage_issues_present" in manifest.warnings

    assert CompiledContentPackReader(db_path).manifest() == manifest


def test_compiler_defaults_are_private_and_sanitize_forbidden_source_material(
    tmp_path,
):
    assert DEFAULT_PRIVATE_PACK_DIR == default_compiled_pack_path(
        "demo"
    ).parent
    assert str(default_compiled_pack_path("Curse Pack 01")).endswith(
        "private_extractions/compiled/Curse-Pack-01.sqlite"
    )

    db_path = tmp_path / "synthetic_pack.sqlite"
    writer = CompiledContentPackWriter(
        db_path,
        pack_id="synthetic",
        protected_terms=[PROTECTED_EXCERPT],
    )
    writer.write_pack(
        pages=[
            {
                "page_id": "page-001",
                "source_asset_id": "src-page-001",
                "source_sha256": "hash-page-001",
                "review_status": "approved",
            }
        ],
        cards=[
            {
                "ref": "safe.redacted",
                "summary": "A redacted synthetic card.",
                "body": "Only fixture-safe paraphrase is persisted.",
                "confidence": 0.90,
                "review_status": "approved",
                "spoiler_class": "none",
                "metadata": {
                    "safe_tag": "fixture",
                    "source_path": RAW_SOURCE_PATH,
                    "protected_excerpt": PROTECTED_EXCERPT,
                    "nested": {"raw_text": PROTECTED_EXCERPT},
                },
            }
        ],
    )

    raw_db = db_path.read_bytes()
    assert RAW_SOURCE_PATH.encode() not in raw_db
    assert PROTECTED_EXCERPT.encode() not in raw_db

    card = CompiledContentPackReader(db_path).load_cards()[0]
    assert card.metadata == {"safe_tag": "fixture", "nested": {}}


def test_compiler_rejects_protected_excerpts_in_persisted_card_text(tmp_path):
    writer = CompiledContentPackWriter(
        tmp_path / "synthetic_pack.sqlite",
        pack_id="synthetic",
        protected_terms=[PROTECTED_EXCERPT],
    )

    with pytest.raises(ValueError, match="protected source excerpt"):
        writer.write_pack(
            pages=[
                {
                    "page_id": "page-001",
                    "source_asset_id": "src-page-001",
                    "review_status": "approved",
                }
            ],
            cards=[
                {
                    "ref": "unsafe.raw",
                    "summary": PROTECTED_EXCERPT,
                    "body": "Synthetic body.",
                    "confidence": 0.90,
                    "review_status": "approved",
                }
            ],
        )
