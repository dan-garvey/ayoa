from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.schemas.content_pack import (
    CompiledContentCard,
    ContentAliasRecord,
    ContentProvenance,
    CoverageGateResult,
    CoverageManifest,
    PageInventoryRecord,
)


DEFAULT_PRIVATE_PACK_DIR = Path("private_extractions/compiled")
SCHEMA_VERSION = "content-pack-v1"
APPROVED_REVIEW_STATUSES = {"reviewed", "approved"}
BLOCKED_REVIEW_STATUSES = {"blocked", "rejected"}
FORBIDDEN_METADATA_KEYS = {
    "dm_notes",
    "file_path",
    "local_path",
    "path",
    "protected_excerpt",
    "raw_bytes",
    "raw_ocr",
    "raw_source_path",
    "raw_text",
    "source_path",
}
_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^/\s]+/)+[^/\s]+")


@dataclass(frozen=True)
class CoverageGatePolicy:
    min_runtime_confidence: float = 0.70
    approved_review_statuses: set[str] = field(
        default_factory=lambda: set(APPROVED_REVIEW_STATUSES)
    )
    blocked_review_statuses: set[str] = field(
        default_factory=lambda: set(BLOCKED_REVIEW_STATUSES)
    )
    high_spoiler_requires_reveal_trigger: bool = True


def default_compiled_pack_path(
    pack_id: str,
    *,
    base_dir: str | Path = DEFAULT_PRIVATE_PACK_DIR,
) -> Path:
    safe_pack_id = _safe_token(pack_id)
    return Path(base_dir) / f"{safe_pack_id}.sqlite"


def assess_card_runtime_gate(
    card: CompiledContentCard | Mapping[str, Any],
    *,
    policy: CoverageGatePolicy | None = None,
) -> CoverageGateResult:
    gate_policy = policy or CoverageGatePolicy()
    record = _coerce_card(card, pack_id="")
    reasons: list[str] = []
    blocked = False

    if record.review_status in gate_policy.blocked_review_statuses:
        reasons.append(f"review_status:{record.review_status}")
        blocked = True

    if record.confidence < gate_policy.min_runtime_confidence:
        reasons.append("low_confidence")
        blocked = True

    if (
        gate_policy.high_spoiler_requires_reveal_trigger
        and record.spoiler_class == "high"
        and not record.reveal_trigger
    ):
        reasons.append("high_spoiler_without_reveal_trigger")
        blocked = True

    if record.review_status not in gate_policy.approved_review_statuses:
        reasons.append(f"review_required:{record.review_status}")

    if not record.content_hash:
        reasons.append("missing_content_hash")

    if blocked:
        return CoverageGateResult(status="blocked", allowed=False, reasons=reasons)
    if reasons:
        return CoverageGateResult(status="flagged", allowed=True, reasons=reasons)
    return CoverageGateResult(status="runtime_ready", allowed=True, reasons=[])


class CompiledContentPackWriter:
    """Write redacted synthetic import records to a private SQLite pack."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        pack_id: str,
        pack_version: str = "0.1.0",
        source_fingerprint: str = "synthetic-redacted",
        importer_version: str = "manual-redacted-v1",
        policy: CoverageGatePolicy | None = None,
        protected_terms: Sequence[str] = (),
    ) -> None:
        self.db_path = Path(db_path)
        self.pack_id = pack_id.strip()
        self.pack_version = pack_version.strip()
        self.source_fingerprint = source_fingerprint.strip()
        self.importer_version = importer_version.strip()
        self.policy = policy or CoverageGatePolicy()
        self.protected_terms = tuple(term for term in protected_terms if term)

    def write_pack(
        self,
        *,
        pages: Iterable[PageInventoryRecord | Mapping[str, Any]],
        cards: Iterable[CompiledContentCard | Mapping[str, Any]],
        aliases: Iterable[ContentAliasRecord | Mapping[str, Any]] = (),
        source_page_count: int | None = None,
    ) -> CoverageManifest:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        page_records = [_coerce_page(page, pack_id=self.pack_id) for page in pages]
        card_records = [self._prepared_card(card) for card in cards]
        alias_records = [
            _coerce_alias(alias, pack_id=self.pack_id) for alias in aliases
        ]
        alias_records.extend(self._aliases_from_cards(card_records))
        alias_records = _dedupe_aliases(alias_records)
        manifest = self._build_manifest(
            page_records,
            card_records,
            alias_records,
            source_page_count=source_page_count,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            _create_schema(conn)
            _replace_pack(conn, self.pack_id)
            self._write_metadata(conn, manifest)
            for page in page_records:
                self._write_page(conn, page)
            for card in card_records:
                self._write_card(conn, card)
            for alias in alias_records:
                self._write_alias(conn, alias)
            self._write_manifest(conn, manifest)
            conn.commit()

        return manifest

    def _prepared_card(
        self,
        card: CompiledContentCard | Mapping[str, Any],
    ) -> CompiledContentCard:
        record = _coerce_card(card, pack_id=self.pack_id)
        self._assert_safe_record_text(record)
        record.metadata = _sanitize_metadata(record.metadata, self.protected_terms)
        if not record.content_hash:
            record.content_hash = _content_hash(record)
        gate = assess_card_runtime_gate(record, policy=self.policy)
        record.gate_status = gate.status
        record.gate_reasons = gate.reasons
        return record

    def _build_manifest(
        self,
        pages: Sequence[PageInventoryRecord],
        cards: Sequence[CompiledContentCard],
        aliases: Sequence[ContentAliasRecord],
        *,
        source_page_count: int | None,
    ) -> CoverageManifest:
        warnings: list[str] = []
        low_confidence = sum(
            1 for card in cards if card.confidence < self.policy.min_runtime_confidence
        )
        high_spoiler = sum(1 for card in cards if card.spoiler_class == "high")
        blocked = sum(1 for card in cards if card.gate_status == "blocked")
        flagged = sum(1 for card in cards if card.gate_status == "flagged")
        ready = sum(1 for card in cards if card.gate_status == "runtime_ready")
        compiled_page_count = len(pages)
        expected_pages = (
            source_page_count if source_page_count is not None else compiled_page_count
        )
        if expected_pages != compiled_page_count:
            warnings.append("page_inventory_count_mismatch")
        if blocked:
            warnings.append("blocked_runtime_records_present")
        if low_confidence:
            warnings.append("low_confidence_records_present")
        if high_spoiler:
            warnings.append("high_spoiler_records_present")
        return CoverageManifest(
            pack_id=self.pack_id,
            pack_version=self.pack_version,
            source_fingerprint=self.source_fingerprint,
            importer_version=self.importer_version,
            schema_version=SCHEMA_VERSION,
            source_page_count=max(0, expected_pages),
            compiled_page_count=compiled_page_count,
            card_count=len(cards),
            alias_count=len(aliases),
            ready_count=ready,
            flagged_count=flagged,
            blocked_count=blocked,
            low_confidence_count=low_confidence,
            high_spoiler_count=high_spoiler,
            warnings=warnings,
        )

    def _write_metadata(
        self,
        conn: sqlite3.Connection,
        manifest: CoverageManifest,
    ) -> None:
        rows = {
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "source_fingerprint": self.source_fingerprint,
            "importer_version": self.importer_version,
            "schema_version": SCHEMA_VERSION,
            "manifest_json": manifest.model_dump_json(),
        }
        conn.executemany(
            """
            INSERT INTO pack_metadata (pack_id, key, value)
            VALUES (?, ?, ?)
            """,
            [(self.pack_id, key, value) for key, value in rows.items()],
        )

    def _write_page(self, conn: sqlite3.Connection, page: PageInventoryRecord) -> None:
        conn.execute(
            """
            INSERT INTO page_inventory (
                pack_id, page_id, source_asset_id, pdf_page_index,
                printed_page_label, source_sha256, section_id, alignment_status,
                confidence, review_status, coverage_status, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.pack_id,
                page.page_id,
                page.source_asset_id,
                page.pdf_page_index,
                page.printed_page_label,
                page.source_sha256,
                page.section_id,
                page.alignment_status,
                page.confidence,
                page.review_status,
                page.coverage_status,
                page.notes,
            ),
        )

    def _write_card(self, conn: sqlite3.Connection, card: CompiledContentCard) -> None:
        conn.execute(
            """
            INSERT INTO content_cards (
                pack_id, ref, content_hash, kind, card_kind, visibility,
                title, summary, body, spoiler_class, reveal_trigger,
                confidence, review_status, gate_status, gate_reasons_json,
                provenance_json, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.pack_id,
                card.ref,
                card.content_hash,
                card.card_kind,
                card.card_kind,
                card.visibility,
                card.title,
                card.summary,
                card.body,
                card.spoiler_class,
                card.reveal_trigger,
                card.confidence,
                card.review_status,
                card.gate_status,
                json.dumps(card.gate_reasons, sort_keys=True),
                json.dumps(
                    [item.model_dump(mode="json") for item in card.provenance],
                    sort_keys=True,
                ),
                json.dumps(card.metadata, sort_keys=True),
            ),
        )
        for provenance in card.provenance:
            self._write_provenance(conn, card.ref, provenance)

    def _write_provenance(
        self,
        conn: sqlite3.Connection,
        ref: str,
        provenance: ContentProvenance,
    ) -> None:
        conn.execute(
            """
            INSERT INTO card_provenance (
                pack_id, ref, source_asset_id, page_id, span_id, image_id,
                bbox_json, section_id, method, confidence, importer_version,
                human_review_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.pack_id,
                ref,
                provenance.source_asset_id,
                provenance.page_id,
                provenance.span_id,
                provenance.image_id,
                json.dumps(provenance.bbox, sort_keys=True),
                provenance.section_id,
                provenance.method,
                provenance.confidence,
                provenance.importer_version,
                provenance.human_review_status,
            ),
        )

    def _write_alias(self, conn: sqlite3.Connection, alias: ContentAliasRecord) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO content_aliases (
                pack_id, alias, ref, kind, confidence, review_status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.pack_id,
                alias.alias,
                alias.ref,
                alias.kind,
                alias.confidence,
                alias.review_status,
            ),
        )

    def _write_manifest(
        self,
        conn: sqlite3.Connection,
        manifest: CoverageManifest,
    ) -> None:
        conn.execute(
            """
            INSERT INTO coverage_manifest (pack_id, manifest_json)
            VALUES (?, ?)
            """,
            (self.pack_id, manifest.model_dump_json()),
        )

    def _aliases_from_cards(
        self,
        cards: Sequence[CompiledContentCard],
    ) -> list[ContentAliasRecord]:
        aliases: list[ContentAliasRecord] = []
        for card in cards:
            for alias in card.aliases:
                aliases.append(
                    ContentAliasRecord(
                        pack_id=self.pack_id,
                        alias=alias,
                        ref=card.ref,
                        kind="card_alias",
                        confidence=card.confidence,
                        review_status=card.review_status,
                    )
                )
        return aliases

    def _assert_safe_record_text(self, record: CompiledContentCard) -> None:
        for field_name in ("title", "summary", "body", "reveal_trigger"):
            _assert_safe_text(
                getattr(record, field_name),
                protected_terms=self.protected_terms,
                field_name=field_name,
            )


class CompiledContentPackReader:
    """Read compiled pack records without exposing blocked runtime content."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def manifest(self) -> CoverageManifest | None:
        if not self.db_path.exists():
            return None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT manifest_json
                FROM coverage_manifest
                ORDER BY rowid DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return CoverageManifest.model_validate_json(row[0])

    def list_pages(self) -> list[PageInventoryRecord]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM page_inventory
                ORDER BY pdf_page_index, page_id
                """
            ).fetchall()
        return [_page_from_row(row) for row in rows]

    def list_aliases(self, *, ref: str | None = None) -> list[ContentAliasRecord]:
        if not self.db_path.exists():
            return []
        where: list[str] = []
        params: list[Any] = []
        if ref:
            where.append("ref = ?")
            params.append(ref)
        sql = "SELECT * FROM content_aliases"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY alias"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [_alias_from_row(row) for row in rows]

    def load_cards(
        self,
        *,
        refs: Iterable[str] | None = None,
        include_flagged: bool = False,
    ) -> list[CompiledContentCard]:
        if not self.db_path.exists():
            return []
        wanted_refs = [ref.strip() for ref in refs or () if ref.strip()]
        gate_statuses = ["runtime_ready"]
        if include_flagged:
            gate_statuses.append("flagged")

        where = [f"gate_status IN ({','.join('?' for _ in gate_statuses)})"]
        params: list[Any] = list(gate_statuses)
        if wanted_refs:
            where.append(f"ref IN ({','.join('?' for _ in wanted_refs)})")
            params.extend(wanted_refs)

        sql = "SELECT * FROM content_cards WHERE " + " AND ".join(where)
        sql += " ORDER BY ref"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [_card_from_row(row) for row in rows]


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pack_metadata (
            pack_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (pack_id, key)
        );
        CREATE TABLE IF NOT EXISTS page_inventory (
            pack_id TEXT NOT NULL,
            page_id TEXT NOT NULL,
            source_asset_id TEXT NOT NULL,
            pdf_page_index INTEGER NOT NULL,
            printed_page_label TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            section_id TEXT NOT NULL,
            alignment_status TEXT NOT NULL,
            confidence REAL NOT NULL,
            review_status TEXT NOT NULL,
            coverage_status TEXT NOT NULL,
            notes TEXT NOT NULL,
            PRIMARY KEY (pack_id, page_id)
        );
        CREATE TABLE IF NOT EXISTS content_cards (
            pack_id TEXT NOT NULL,
            ref TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            kind TEXT NOT NULL,
            card_kind TEXT NOT NULL,
            visibility TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            body TEXT NOT NULL,
            spoiler_class TEXT NOT NULL,
            reveal_trigger TEXT NOT NULL,
            confidence REAL NOT NULL,
            review_status TEXT NOT NULL,
            gate_status TEXT NOT NULL,
            gate_reasons_json TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (pack_id, ref)
        );
        CREATE TABLE IF NOT EXISTS card_provenance (
            pack_id TEXT NOT NULL,
            ref TEXT NOT NULL,
            source_asset_id TEXT NOT NULL,
            page_id TEXT NOT NULL,
            span_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            bbox_json TEXT NOT NULL,
            section_id TEXT NOT NULL,
            method TEXT NOT NULL,
            confidence REAL NOT NULL,
            importer_version TEXT NOT NULL,
            human_review_status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS content_aliases (
            pack_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            ref TEXT NOT NULL,
            kind TEXT NOT NULL,
            confidence REAL NOT NULL,
            review_status TEXT NOT NULL,
            PRIMARY KEY (pack_id, alias, ref)
        );
        CREATE TABLE IF NOT EXISTS coverage_manifest (
            pack_id TEXT NOT NULL,
            manifest_json TEXT NOT NULL
        );
        """
    )


def _replace_pack(conn: sqlite3.Connection, pack_id: str) -> None:
    for table in (
        "pack_metadata",
        "page_inventory",
        "content_cards",
        "card_provenance",
        "content_aliases",
        "coverage_manifest",
    ):
        conn.execute(f"DELETE FROM {table} WHERE pack_id = ?", (pack_id,))


def _coerce_page(
    page: PageInventoryRecord | Mapping[str, Any],
    *,
    pack_id: str,
) -> PageInventoryRecord:
    if isinstance(page, PageInventoryRecord):
        values = page.model_dump()
    else:
        values = dict(page)
    values["pack_id"] = values.get("pack_id") or pack_id
    return PageInventoryRecord(**values)


def _coerce_card(
    card: CompiledContentCard | Mapping[str, Any],
    *,
    pack_id: str,
) -> CompiledContentCard:
    if isinstance(card, CompiledContentCard):
        values = card.model_dump()
    else:
        values = dict(card)
    values["pack_id"] = values.get("pack_id") or pack_id
    return CompiledContentCard(**values)


def _coerce_alias(
    alias: ContentAliasRecord | Mapping[str, Any],
    *,
    pack_id: str,
) -> ContentAliasRecord:
    if isinstance(alias, ContentAliasRecord):
        values = alias.model_dump()
    else:
        values = dict(alias)
    values["pack_id"] = values.get("pack_id") or pack_id
    return ContentAliasRecord(**values)


def _dedupe_aliases(aliases: Iterable[ContentAliasRecord]) -> list[ContentAliasRecord]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[ContentAliasRecord] = []
    for alias in aliases:
        key = (alias.pack_id, alias.alias.casefold(), alias.ref)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(alias)
    return deduped


def _content_hash(card: CompiledContentCard) -> str:
    payload = {
        "pack_id": card.pack_id,
        "ref": card.ref,
        "card_kind": card.card_kind,
        "visibility": card.visibility,
        "title": card.title,
        "summary": card.summary,
        "body": card.body,
        "spoiler_class": card.spoiler_class,
        "reveal_trigger": card.reveal_trigger,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sanitize_metadata(
    value: Any,
    protected_terms: Sequence[str],
) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.strip().lower() in FORBIDDEN_METADATA_KEYS:
                continue
            sanitized_value = _sanitize_metadata(item, protected_terms)
            if sanitized_value is not None:
                sanitized[key_text] = sanitized_value
        return sanitized
    if isinstance(value, list):
        return [
            sanitized
            for item in value
            if (sanitized := _sanitize_metadata(item, protected_terms)) is not None
        ]
    if isinstance(value, str):
        if _ABSOLUTE_PATH_RE.search(value):
            return None
        if any(term in value for term in protected_terms):
            return None
    return value


def _assert_safe_text(
    value: str,
    *,
    protected_terms: Sequence[str],
    field_name: str,
) -> None:
    if _ABSOLUTE_PATH_RE.search(value):
        raise ValueError(f"{field_name} contains a raw source path")
    for term in protected_terms:
        if term in value:
            raise ValueError(f"{field_name} contains a protected source excerpt")


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    if not token:
        raise ValueError("pack_id must contain at least one safe path character")
    return token


def _page_from_row(row: sqlite3.Row) -> PageInventoryRecord:
    return PageInventoryRecord(
        pack_id=row["pack_id"],
        page_id=row["page_id"],
        source_asset_id=row["source_asset_id"],
        pdf_page_index=row["pdf_page_index"],
        printed_page_label=row["printed_page_label"],
        source_sha256=row["source_sha256"],
        section_id=row["section_id"],
        alignment_status=row["alignment_status"],
        confidence=row["confidence"],
        review_status=row["review_status"],
        coverage_status=row["coverage_status"],
        notes=row["notes"],
    )


def _alias_from_row(row: sqlite3.Row) -> ContentAliasRecord:
    return ContentAliasRecord(
        pack_id=row["pack_id"],
        alias=row["alias"],
        ref=row["ref"],
        kind=row["kind"],
        confidence=row["confidence"],
        review_status=row["review_status"],
    )


def _card_from_row(row: sqlite3.Row) -> CompiledContentCard:
    provenance = [
        ContentProvenance(**item)
        for item in _json_list(row["provenance_json"])
        if isinstance(item, Mapping)
    ]
    metadata = _json_mapping(row["metadata_json"])
    gate_reasons = [
        str(item) for item in _json_list(row["gate_reasons_json"]) if str(item)
    ]
    return CompiledContentCard(
        pack_id=row["pack_id"],
        ref=row["ref"],
        content_hash=row["content_hash"],
        card_kind=row["card_kind"],
        visibility=row["visibility"],
        title=row["title"],
        summary=row["summary"],
        body=row["body"],
        spoiler_class=row["spoiler_class"],
        reveal_trigger=row["reveal_trigger"],
        confidence=row["confidence"],
        review_status=row["review_status"],
        gate_status=row["gate_status"],
        gate_reasons=gate_reasons,
        provenance=provenance,
        metadata=dict(metadata),
    )


def _json_list(value: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _json_mapping(value: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}
