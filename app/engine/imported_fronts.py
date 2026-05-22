from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from app.schemas.content import ContentPackState, PendingContentSignal
from app.schemas.content_pack import ContentPackDomainCatalog, FrontDossierRecord
from app.schemas.content_privacy import contains_imported_asset_sentinel


APPROVED_REVIEW_STATUSES = {"reviewed", "approved"}


class ImportedFrontError(ValueError):
    """Base error for reviewed imported front resolution."""


class ImportedFrontNotFoundError(ImportedFrontError):
    """Raised when a requested front dossier is not authored."""


class ImportedFrontValidationError(ImportedFrontError):
    """Raised when a front dossier is not router-ready."""


class ImportedFrontDossierCatalog:
    """In-memory index of reviewed front/villain pressure dossiers."""

    def __init__(
        self,
        records: Iterable[FrontDossierRecord | Mapping[str, Any]],
    ) -> None:
        self._records: dict[str, FrontDossierRecord] = {}
        self._aliases: dict[str, str] = {}
        for raw in records:
            record = _validated_record(raw)
            if record.ref in self._records:
                raise ImportedFrontValidationError(
                    f"Duplicate imported front dossier ref: {record.ref}"
                )
            self._records[record.ref] = record
            for alias in _record_aliases(record):
                existing = self._aliases.get(alias)
                if existing and existing != record.ref:
                    raise ImportedFrontValidationError(
                        "Imported front dossier aliases collide: "
                        f"{alias} maps to both {existing} and {record.ref}."
                    )
                self._aliases[alias] = record.ref

    @classmethod
    def from_domain_catalog(
        cls,
        catalog: ContentPackDomainCatalog | Mapping[str, Any],
    ) -> "ImportedFrontDossierCatalog":
        if isinstance(catalog, Mapping):
            pack_id = str(catalog.get("pack_id") or "").strip()
            records = []
            for raw in catalog.get("front_dossiers") or ():
                if isinstance(raw, Mapping) and pack_id and not raw.get("pack_id"):
                    records.append({**raw, "pack_id": pack_id})
                else:
                    records.append(raw)
            return cls(records)
        return cls(catalog.front_dossiers)

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(self._records)

    def get(self, ref: str) -> FrontDossierRecord:
        cleaned = str(ref or "").strip()
        canonical_ref = self._aliases.get(cleaned) or self._aliases.get(
            _clean_id(cleaned)
        )
        if not cleaned or canonical_ref not in self._records:
            raise ImportedFrontNotFoundError(
                f"Imported front dossier ref is not authored: {cleaned or '<empty>'}"
            )
        return self._records[canonical_ref]


def catalog_from_pack_state(
    pack_key: Any,
    pack_state: Any,
) -> ImportedFrontDossierCatalog | None:
    records: list[FrontDossierRecord | Mapping[str, Any]] = []
    pack_id = _pack_id(pack_key, pack_state)
    metadata = _pack_metadata(pack_state)
    for key in ("domain_catalog", "content_pack_domain_catalog"):
        raw_catalog = metadata.get(key)
        if raw_catalog is None:
            continue
        if isinstance(raw_catalog, ContentPackDomainCatalog):
            records.extend(raw_catalog.front_dossiers)
        elif isinstance(raw_catalog, Mapping):
            records.extend(
                _records_with_pack(raw_catalog.get("front_dossiers") or (), pack_id)
            )
    for key in (
        "front_dossiers",
        "front_dossier_records",
        "imported_front_dossiers",
        "imported_fronts",
    ):
        records.extend(_records_with_pack(metadata.get(key) or (), pack_id))
    if not records:
        return None
    return ImportedFrontDossierCatalog(records)


def front_dossier_router_payload(
    record: FrontDossierRecord | Mapping[str, Any],
    *,
    pack_id: str = "",
) -> dict[str, Any]:
    """Project a reviewed front dossier into router-only adjudication facts."""

    front = _validated_record(record)
    villain_refs = _safe_items(front.villain_refs)
    payload: dict[str, Any] = {
        "kind": "front_signal",
        "ref": front.ref,
        "pack_id": pack_id or front.pack_id,
        "content_hash": front.content_hash,
        "visibility": "hidden",
        "actor": villain_refs[0] if villain_refs else "",
        "villains": villain_refs,
        "knows": _safe_items(front.initial_knowledge),
        "summary": _safe_text(front.summary),
        "goals": _safe_items(front.goals),
        "constraints": _safe_items(front.constraints),
        "knowledge_channels": _safe_items(front.knowledge_channels),
        "resources": _safe_items(front.resources),
        "minions": _safe_items(front.minion_refs),
        "escalation_thresholds": _safe_items(front.escalation_thresholds),
        "cooldowns": _safe_items(front.cooldowns),
        "restraints": _safe_items(front.restraints),
        "actions": _front_actions(front),
    }
    return {
        key: value
        for key, value in payload.items()
        if value not in ("", [], {}, None)
    }


def queue_front_dossier_signal(
    pack_state: ContentPackState | dict[str, Any],
    record: FrontDossierRecord | Mapping[str, Any],
    *,
    pack_id: str = "",
    reason: str = "",
    priority: int = 0,
) -> PendingContentSignal:
    """Queue one front_signal record for the omniscient router only."""

    payload = front_dossier_router_payload(record, pack_id=pack_id)
    front_ref = str(payload["ref"])
    resolved_pack_id = str(payload.get("pack_id") or pack_id or "").strip()
    signal = PendingContentSignal(
        signal_id=_signal_id(resolved_pack_id, front_ref, str(payload["content_hash"])),
        pack_id=resolved_pack_id,
        ref_id=front_ref,
        content_hash=str(payload["content_hash"]),
        reason=_safe_text(reason) or _safe_text(payload.get("summary", "")),
        status="pending",
        priority=max(0, int(priority or 0)),
        requested_fields=["front_signal"],
        metadata=payload,
    )
    pending = (
        pack_state.setdefault("pending_signals", {})
        if isinstance(pack_state, dict)
        else pack_state.pending_signals
    )
    pending[signal.signal_id] = signal
    return signal


def _front_actions(front: FrontDossierRecord) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for action in sorted(
        front.action_palette,
        key=lambda item: (-item.priority, item.action_id),
    ):
        payload = {
            "id": _safe_text(action.action_id),
            "kind": _safe_text(action.action_kind),
            "priority": str(max(0, int(action.priority or 0))),
            "trigger": _safe_text(action.trigger),
            "cooldown": _safe_text(action.cooldown),
            "target_scope": _safe_text(action.target_scope),
            "summary": _safe_text(action.summary),
            "resources": _safe_items(action.resource_refs),
            "minions": _safe_items(action.minion_refs),
            "restraints": _safe_items(action.restraints),
            "consequences": _safe_items(action.consequence_refs),
            "encounters": _safe_items(action.encounter_template_refs),
            "statblocks": _safe_items(action.statblock_refs),
        }
        cleaned = {
            key: value
            for key, value in payload.items()
            if value not in ("", [], None)
        }
        if cleaned:
            actions.append(cleaned)
    return actions


def _validated_record(
    record: FrontDossierRecord | Mapping[str, Any],
) -> FrontDossierRecord:
    front = (
        record
        if isinstance(record, FrontDossierRecord)
        else _coerce_record(record)
    )
    if not front.ref:
        raise ImportedFrontValidationError("Imported front dossier is missing ref.")
    if not front.content_hash:
        raise ImportedFrontValidationError(
            f"Imported front dossier {front.ref!r} is missing content_hash."
        )
    if front.review_status not in APPROVED_REVIEW_STATUSES:
        raise ImportedFrontValidationError(
            f"Imported front dossier {front.ref!r} is not reviewed."
        )
    if front.gate_status != "runtime_ready":
        raise ImportedFrontValidationError(
            f"Imported front dossier {front.ref!r} is not runtime-ready."
        )
    return front


def _coerce_record(record: Mapping[str, Any]) -> FrontDossierRecord:
    try:
        return FrontDossierRecord(**dict(record))
    except ValidationError as exc:
        raise ImportedFrontValidationError(str(exc)) from exc


def _record_aliases(record: FrontDossierRecord) -> tuple[str, ...]:
    aliases = [record.ref, _clean_id(record.ref), record.title, _clean_id(record.title)]
    aliases.extend(record.villain_refs)
    return tuple(alias for alias in dict.fromkeys(aliases) if alias)


def _records_with_pack(records: Iterable[Any], pack_id: str) -> list[Any]:
    values: list[Any] = []
    for raw in records or ():
        if isinstance(raw, Mapping) and pack_id and not raw.get("pack_id"):
            values.append({**raw, "pack_id": pack_id})
        else:
            values.append(raw)
    return values


def _safe_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text or contains_imported_asset_sentinel(text):
        return ""
    return text


def _safe_items(values: Iterable[Any]) -> list[str]:
    return [text for value in values or () if (text := _safe_text(value))]


def _signal_id(pack_id: str, ref: str, content_hash: str) -> str:
    payload = json.dumps(
        {
            "pack_id": pack_id,
            "ref": ref,
            "content_hash": content_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "front-dossier-" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _pack_metadata(pack_state: Any) -> Mapping[str, Any]:
    raw = (
        pack_state.get("metadata")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "metadata", None)
    )
    return raw if isinstance(raw, Mapping) else {}


def _pack_id(pack_key: Any, pack_state: Any) -> str:
    raw = (
        pack_state.get("pack_id")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "pack_id", "")
    )
    return str(raw or pack_key or "").strip()


def _clean_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
