from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from app.schemas.content import (
    ContentPackState,
    ContentTrapOverlayState,
    content_overlay_key,
    content_ref_key,
)
from app.schemas.content_pack import (
    ContentPackDomainCatalog,
    DndDamageExpression,
    TrapHazardMechanics,
    TrapHazardPlacement,
    TrapHazardRecord,
)
from app.schemas.content_privacy import (
    contains_imported_asset_sentinel,
    redact_imported_asset_text,
    sanitize_player_safe_text,
)


APPROVED_REVIEW_STATUSES = {"reviewed", "approved"}
ADJUDICATION_AUDIENCES = {"router", "dnd_combat_resolver"}
TrapHazardAdjudicationAudience = Literal["router", "dnd_combat_resolver"]


class ImportedTrapHazardError(ValueError):
    """Base error for reviewed imported trap/hazard resolution."""


class ImportedTrapHazardNotFoundError(ImportedTrapHazardError):
    """Raised when a requested trap/hazard ref is not in the catalog."""


class ImportedTrapHazardValidationError(ImportedTrapHazardError):
    """Raised when an imported trap/hazard is not safe for adjudication."""


class ImportedTrapHazardCatalog:
    """In-memory index of reviewed D&D trap and hazard pack records."""

    def __init__(
        self,
        records: Iterable[TrapHazardRecord | Mapping[str, Any]],
    ) -> None:
        self._records: dict[str, TrapHazardRecord] = {}
        self._aliases: dict[str, str] = {}
        self._explicit_fields: dict[str, frozenset[str]] = {}
        for raw in records:
            record, explicit_fields = _coerce_record_with_fields(raw)
            if not record.ref:
                raise ImportedTrapHazardValidationError(
                    "Imported trap/hazard record is missing ref."
                )
            key = _record_key(record)
            if key in self._records:
                raise ImportedTrapHazardValidationError(
                    f"Duplicate imported trap/hazard ref: {key}"
                )
            self._records[key] = record
            self._explicit_fields[key] = explicit_fields
            for alias in _record_aliases(record):
                existing = self._aliases.get(alias)
                if existing and existing != key:
                    raise ImportedTrapHazardValidationError(
                        "Imported trap/hazard aliases collide: "
                        f"{alias} maps to both {existing} and {key}."
                    )
                self._aliases[alias] = key

    @classmethod
    def from_domain_catalog(
        cls,
        catalog: ContentPackDomainCatalog | Mapping[str, Any],
    ) -> "ImportedTrapHazardCatalog":
        if isinstance(catalog, Mapping):
            pack_id = str(catalog.get("pack_id") or "").strip()
            return cls(
                _records_with_pack(
                    catalog.get("trap_hazards") or (),
                    pack_id=pack_id,
                )
            )
        return cls(catalog.trap_hazards)

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(record.ref for record in self._records.values())

    def get(self, ref: str, *, pack_id: str = "") -> TrapHazardRecord:
        key = self._canonical_key(ref, pack_id=pack_id)
        return self._records[key]

    def resolve(self, ref: str, *, pack_id: str = "") -> TrapHazardRecord:
        key = self._canonical_key(ref, pack_id=pack_id)
        return validate_trap_hazard_record(
            self._records[key],
            explicit_fields=self._explicit_fields[key],
        )

    def refs_for_location(self, location_ref: str) -> tuple[str, ...]:
        cleaned = _clean_text(location_ref)
        if not cleaned:
            return ()
        return tuple(
            record.ref
            for record in self._records.values()
            if cleaned in record.linked_location_refs
            or any(placement.location_ref == cleaned for placement in record.placements)
        )

    def refs_for_map_feature(self, map_feature_ref: str) -> tuple[str, ...]:
        cleaned = _clean_text(map_feature_ref)
        if not cleaned:
            return ()
        return tuple(
            record.ref
            for record in self._records.values()
            if cleaned in record.linked_map_feature_refs
            or any(
                placement.map_feature_ref == cleaned
                for placement in record.placements
            )
        )

    def router_context_records(
        self,
        refs: Iterable[str] | None = None,
        *,
        overlays: Any = None,
    ) -> list[str]:
        return self._adjudication_context_records(
            refs,
            overlays=overlays,
            audience="router",
        )

    def combat_context_records(
        self,
        refs: Iterable[str] | None = None,
        *,
        overlays: Any = None,
    ) -> list[str]:
        return self._adjudication_context_records(
            refs,
            overlays=overlays,
            audience="dnd_combat_resolver",
        )

    def player_safe_records(
        self,
        refs: Iterable[str] | None = None,
        *,
        overlays: Any = None,
    ) -> list[str]:
        records: list[str] = []
        for record in self._iter_records(refs):
            rendered = format_player_safe_trap_hazard_record(
                validate_trap_hazard_record(
                    record,
                    explicit_fields=self._explicit_fields[_record_key(record)],
                ),
                overlay=_overlay_for_record(record, overlays),
            )
            if rendered:
                records.append(rendered)
        return records

    def ordinary_log_records(
        self,
        refs: Iterable[str] | None = None,
        *,
        overlays: Any = None,
    ) -> list[str]:
        return [
            format_trap_hazard_log_record(
                record,
                overlay=_overlay_for_record(record, overlays),
            )
            for record in self._iter_records(refs)
        ]

    def _adjudication_context_records(
        self,
        refs: Iterable[str] | None,
        *,
        overlays: Any,
        audience: TrapHazardAdjudicationAudience,
    ) -> list[str]:
        records: list[str] = []
        for record in self._iter_records(refs):
            key = _record_key(record)
            records.append(
                format_trap_hazard_context_record(
                    validate_trap_hazard_record(
                        record,
                        explicit_fields=self._explicit_fields[key],
                    ),
                    overlay=_overlay_for_record(record, overlays),
                    audience=audience,
                )
            )
        return records

    def _iter_records(
        self,
        refs: Iterable[str] | None,
    ) -> list[TrapHazardRecord]:
        if refs is None:
            return list(self._records.values())
        return [self.resolve(ref) for ref in refs]

    def _canonical_key(self, ref: str, *, pack_id: str = "") -> str:
        cleaned = str(ref or "").strip()
        if not cleaned:
            raise ImportedTrapHazardNotFoundError(
                "Imported trap/hazard ref is not authored: <empty>"
            )
        candidates = [cleaned, _clean_id(cleaned)]
        if pack_id:
            scoped = f"{pack_id.strip()}::{cleaned}"
            candidates.extend([scoped, _clean_id(scoped)])
        for candidate in candidates:
            canonical = self._aliases.get(candidate)
            if canonical in self._records:
                return canonical
        raise ImportedTrapHazardNotFoundError(
            f"Imported trap/hazard ref is not authored: {cleaned}"
        )


def catalog_from_content_state(
    content_state: Any,
) -> ImportedTrapHazardCatalog | None:
    """Build a trap/hazard catalog from checkpoint content-pack metadata."""

    if not isinstance(content_state, Mapping):
        return None
    records: list[TrapHazardRecord | Mapping[str, Any]] = []
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        metadata = _pack_metadata(pack_state)
        records.extend(_metadata_trap_hazard_records(metadata, pack_id=pack_id))
    if not records:
        return None
    return ImportedTrapHazardCatalog(records)


def trap_overlay_from_content_state(
    content_state: Any,
    record: TrapHazardRecord | Mapping[str, Any],
) -> ContentTrapOverlayState | None:
    """Resolve mutable reveal/depletion state for one imported trap/hazard."""

    if not isinstance(content_state, Mapping):
        return None
    trap = _coerce_record(record)
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        if trap.pack_id and pack_id and trap.pack_id != pack_id:
            continue
        overlay = _overlay_for_record(trap, pack_state)
        if overlay is not None:
            return overlay
    return None


def validate_trap_hazard_record(
    record: TrapHazardRecord | Mapping[str, Any],
    *,
    explicit_fields: set[str] | frozenset[str] | None = None,
) -> TrapHazardRecord:
    trap_hazard = _coerce_record(record)
    problems = _runtime_validation_problems(
        trap_hazard,
        explicit_fields=explicit_fields,
    )
    if problems:
        joined = ", ".join(problems)
        raise ImportedTrapHazardValidationError(
            f"Imported trap/hazard {trap_hazard.ref!r} is not runtime-ready: "
            f"{joined}."
        )
    return trap_hazard


def trap_hazard_runtime_state(
    record: TrapHazardRecord | Mapping[str, Any],
    *,
    overlay: ContentTrapOverlayState | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trap_hazard = _coerce_record(record)
    state = _coerce_overlay(overlay)
    revealed = bool(getattr(state, "revealed", False)) if state else False
    disabled = bool(getattr(state, "disabled", False)) if state else False
    sprung = bool(getattr(state, "sprung", False)) if state else False
    depleted = bool(getattr(state, "depleted", False)) if state else False
    if depleted:
        status = "depleted"
    elif disabled:
        status = "disabled"
    elif sprung:
        status = "sprung"
    elif revealed:
        status = "revealed"
    else:
        status = "hidden"
    mechanics = trap_hazard.mechanics
    reset_policy = mechanics.reset_policy if mechanics is not None else ""
    return {
        "status": status,
        "revealed": revealed,
        "disabled": disabled,
        "sprung": sprung,
        "depleted": depleted,
        "armed": not disabled and not depleted and not sprung,
        "reset_policy": reset_policy,
        "depletion_ref": mechanics.depletion_ref if mechanics is not None else "",
    }


def trap_hazard_adjudication_payload(
    record: TrapHazardRecord | Mapping[str, Any],
    *,
    overlay: ContentTrapOverlayState | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return omniscient router/combat context for a reviewed trap/hazard."""

    trap_hazard = validate_trap_hazard_record(record)
    payload = {
        "ref": trap_hazard.ref,
        "pack_id": trap_hazard.pack_id,
        "content_hash": trap_hazard.content_hash,
        "kind": trap_hazard.trap_hazard_kind,
        "visibility": trap_hazard.visibility,
        "state": trap_hazard_runtime_state(trap_hazard, overlay=overlay),
        "title": _context_text(trap_hazard.title),
        "summary": _context_text(trap_hazard.summary),
        "trigger": _context_text(trap_hazard.trigger),
        "detection": _context_text(trap_hazard.detection),
        "countermeasures": _context_texts(trap_hazard.countermeasures),
        "linked_location_refs": trap_hazard.linked_location_refs,
        "linked_map_feature_refs": trap_hazard.linked_map_feature_refs,
        "placements": [
            _placement_payload(placement)
            for placement in trap_hazard.placements
        ],
        "runtime_consequences": _context_texts(
            trap_hazard.runtime_consequences
        ),
        "mechanics": _mechanics_payload(trap_hazard.mechanics),
    }
    return _drop_empty(payload)


def format_trap_hazard_context_record(
    record: TrapHazardRecord | Mapping[str, Any],
    *,
    overlay: ContentTrapOverlayState | Mapping[str, Any] | None = None,
    audience: TrapHazardAdjudicationAudience = "router",
) -> str:
    """Format a compact omniscient context record for router/combat only."""

    if audience not in ADJUDICATION_AUDIENCES:
        raise ValueError(f"Unsupported trap/hazard context audience: {audience}")
    payload = trap_hazard_adjudication_payload(record, overlay=overlay)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"trap_hazard_context audience={audience} payload={encoded}"


def format_player_safe_trap_hazard_record(
    record: TrapHazardRecord | Mapping[str, Any],
    *,
    overlay: ContentTrapOverlayState | Mapping[str, Any] | None = None,
) -> str | None:
    """Format a player-safe record only after the trap/hazard is revealed."""

    trap_hazard = validate_trap_hazard_record(record)
    state = trap_hazard_runtime_state(trap_hazard, overlay=overlay)
    if not _player_can_see_trap(trap_hazard, state):
        return None
    payload = {
        "ref": trap_hazard.ref,
        "pack_id": trap_hazard.pack_id,
        "content_hash": trap_hazard.content_hash,
        "kind": trap_hazard.trap_hazard_kind,
        "state": state["status"],
        "title": sanitize_player_safe_text(trap_hazard.title),
        "summary": sanitize_player_safe_text(trap_hazard.summary),
        "detection": sanitize_player_safe_text(trap_hazard.detection),
        "countermeasures": [
            cleaned
            for text in trap_hazard.countermeasures
            if (cleaned := sanitize_player_safe_text(text))
        ],
    }
    encoded = json.dumps(
        _drop_empty(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"trap_hazard_visible payload={encoded}"


def format_trap_hazard_log_record(
    record: TrapHazardRecord | Mapping[str, Any],
    *,
    overlay: ContentTrapOverlayState | Mapping[str, Any] | None = None,
) -> str:
    """Return a log-safe state line without trigger/mechanics/secret text."""

    trap_hazard = _coerce_record(record)
    state = trap_hazard_runtime_state(trap_hazard, overlay=overlay)
    parts = [
        "trap_hazard_state",
        f"ref={_log_token(trap_hazard.ref)}",
        f"kind={_log_token(trap_hazard.trap_hazard_kind)}",
        f"state={_log_token(state['status'])}",
    ]
    if trap_hazard.pack_id:
        parts.append(f"pack={_log_token(trap_hazard.pack_id)}")
    if trap_hazard.content_hash:
        parts.append(f"hash={_log_token(trap_hazard.content_hash)}")
    for key in ("revealed", "disabled", "sprung", "depleted", "armed"):
        parts.append(f"{key}={str(bool(state[key])).lower()}")
    return " ".join(part for part in parts if part)


def _runtime_validation_problems(
    trap_hazard: TrapHazardRecord,
    *,
    explicit_fields: set[str] | frozenset[str] | None,
) -> list[str]:
    problems: list[str] = []
    if trap_hazard.gate_status != "runtime_ready":
        problems.append(f"gate_status:{trap_hazard.gate_status}")
    if trap_hazard.review_status not in APPROVED_REVIEW_STATUSES:
        problems.append(f"review_status:{trap_hazard.review_status}")
    if not trap_hazard.content_hash:
        problems.append("content_hash")
    if not trap_hazard.trigger:
        problems.append("trigger")
    if (
        not trap_hazard.detection
        and (trap_hazard.mechanics is None or trap_hazard.mechanics.detection_dc is None)
    ):
        problems.append("detection")
    if (
        not trap_hazard.countermeasures
        and (trap_hazard.mechanics is None or trap_hazard.mechanics.disarm_dc is None)
    ):
        problems.append("disarm_or_countermeasure")
    if (
        not trap_hazard.linked_location_refs
        and not trap_hazard.linked_map_feature_refs
        and not trap_hazard.placements
    ):
        problems.append("placement")
    if trap_hazard.mechanics is None:
        problems.append("mechanics")
    else:
        problems.extend(_mechanics_validation_problems(trap_hazard.mechanics))
        if (
            not trap_hazard.mechanics.reset_policy
            and not trap_hazard.mechanics.depletion_ref
        ):
            problems.append("reset_or_depletion")
        if not _has_runtime_resolution(trap_hazard):
            problems.append("save_attack_damage_or_effect")
    problems.extend(_unsafe_text_problems(trap_hazard))
    if explicit_fields is not None and "mechanics" not in explicit_fields:
        problems.append("mechanics")
    return list(dict.fromkeys(problems))


def _mechanics_validation_problems(
    mechanics: TrapHazardMechanics,
) -> list[str]:
    problems: list[str] = []
    if mechanics.detection_dc is not None and mechanics.detection_dc <= 0:
        problems.append("detection_dc")
    if mechanics.disarm_dc is not None and mechanics.disarm_dc <= 0:
        problems.append("disarm_dc")
    if mechanics.save_dc is not None and mechanics.save_dc <= 0:
        problems.append("save_dc")
    if mechanics.save_dc is not None and not mechanics.save_ability:
        problems.append("save_ability")
    if mechanics.save_ability and mechanics.save_dc is None:
        problems.append("save_dc")
    return problems


def _has_runtime_resolution(trap_hazard: TrapHazardRecord) -> bool:
    mechanics = trap_hazard.mechanics
    if mechanics is None:
        return False
    return any(
        (
            mechanics.save_dc is not None and bool(mechanics.save_ability),
            mechanics.attack_bonus is not None,
            bool(mechanics.damage),
            bool(mechanics.conditions),
            bool(mechanics.effects),
            bool(trap_hazard.runtime_consequences),
        )
    )


def _unsafe_text_problems(trap_hazard: TrapHazardRecord) -> list[str]:
    problems: list[str] = []
    text_fields: list[tuple[str, str]] = [
        ("ref", trap_hazard.ref),
        ("title", trap_hazard.title),
        ("summary", trap_hazard.summary),
        ("trigger", trap_hazard.trigger),
        ("detection", trap_hazard.detection),
    ]
    text_fields.extend(
        (f"countermeasures[{index}]", text)
        for index, text in enumerate(trap_hazard.countermeasures)
    )
    text_fields.extend(
        (f"runtime_consequences[{index}]", text)
        for index, text in enumerate(trap_hazard.runtime_consequences)
    )
    if trap_hazard.mechanics is not None:
        text_fields.extend(_mechanics_text_fields(trap_hazard.mechanics))
    for index, placement in enumerate(trap_hazard.placements):
        text_fields.extend(_placement_text_fields(placement, index=index))
    return [
        f"unsafe:{field}"
        for field, text in text_fields
        if contains_imported_asset_sentinel(text)
    ]


def _mechanics_text_fields(
    mechanics: TrapHazardMechanics,
) -> list[tuple[str, str]]:
    fields = [
        ("mechanics.target", mechanics.target),
        ("mechanics.save_success", mechanics.save_success),
        ("mechanics.save_failure", mechanics.save_failure),
        ("mechanics.reset_policy", mechanics.reset_policy),
        ("mechanics.depletion_ref", mechanics.depletion_ref),
    ]
    fields.extend(
        (f"mechanics.conditions[{index}]", text)
        for index, text in enumerate(mechanics.conditions)
    )
    fields.extend(
        (f"mechanics.effects[{index}]", text)
        for index, text in enumerate(mechanics.effects)
    )
    return fields


def _placement_text_fields(
    placement: TrapHazardPlacement,
    *,
    index: int,
) -> list[tuple[str, str]]:
    return [
        (f"placements[{index}].placement_id", placement.placement_id),
        (f"placements[{index}].location_ref", placement.location_ref),
        (f"placements[{index}].map_template_ref", placement.map_template_ref),
        (f"placements[{index}].map_feature_ref", placement.map_feature_ref),
        (f"placements[{index}].area_ref", placement.area_ref),
        (f"placements[{index}].floor_id", placement.floor_id),
        (f"placements[{index}].label", placement.label),
        (f"placements[{index}].reveal_trigger", placement.reveal_trigger),
    ]


def _mechanics_payload(
    mechanics: TrapHazardMechanics | None,
) -> dict[str, Any]:
    if mechanics is None:
        return {}
    save = _drop_empty({
        "dc": mechanics.save_dc,
        "ability": mechanics.save_ability,
        "success": _context_text(mechanics.save_success),
        "failure": _context_text(mechanics.save_failure),
    })
    attack = _drop_empty({
        "bonus": mechanics.attack_bonus,
        "target": _context_text(mechanics.target),
    })
    return _drop_empty({
        "ruleset_id": mechanics.ruleset_id,
        "target": _context_text(mechanics.target),
        "detection_dc": mechanics.detection_dc,
        "disarm_dc": mechanics.disarm_dc,
        "save": save,
        "attack": attack,
        "damage": [
            _damage_payload(damage)
            for damage in mechanics.damage
        ],
        "conditions": _context_texts(mechanics.conditions),
        "effects": _context_texts(mechanics.effects),
        "reset_policy": _context_text(mechanics.reset_policy),
        "depletion_ref": mechanics.depletion_ref,
    })


def _damage_payload(damage: DndDamageExpression) -> dict[str, Any]:
    return _drop_empty({
        "expression": damage.expression,
        "damage_type": damage.damage_type,
        "condition": _context_text(damage.condition),
    })


def _placement_payload(placement: TrapHazardPlacement) -> dict[str, Any]:
    return _drop_empty({
        "placement_id": placement.placement_id,
        "location_ref": placement.location_ref,
        "map_template_ref": placement.map_template_ref,
        "map_feature_ref": placement.map_feature_ref,
        "area_ref": placement.area_ref,
        "floor_id": placement.floor_id,
        "cells": [
            cell.model_dump(mode="json")
            for cell in placement.cells
        ],
        "bounds": (
            placement.bounds.model_dump(mode="json")
            if placement.bounds is not None else None
        ),
        "label": _context_text(placement.label),
        "hidden": placement.hidden,
        "reveal_trigger": _context_text(placement.reveal_trigger),
    })


def _player_can_see_trap(
    trap_hazard: TrapHazardRecord,
    state: Mapping[str, Any],
) -> bool:
    if trap_hazard.visibility == "player_safe":
        return True
    if trap_hazard.visibility == "player_visible_after_reveal":
        return bool(state.get("revealed") or state.get("sprung") or state.get("disabled"))
    return bool(
        state.get("revealed")
        or state.get("sprung")
        or state.get("disabled")
        or state.get("depleted")
    )


def _overlay_for_record(
    record: TrapHazardRecord,
    overlays: Any,
) -> ContentTrapOverlayState | None:
    if overlays is None:
        return None
    if isinstance(overlays, ContentPackState):
        return _overlay_for_record(record, overlays.overlay.traps)
    if hasattr(overlays, "overlay"):
        return _overlay_for_record(record, getattr(overlays, "overlay"))
    if hasattr(overlays, "traps"):
        return _overlay_for_record(record, getattr(overlays, "traps"))
    if isinstance(overlays, Mapping) and "overlay" in overlays:
        return _overlay_for_record(record, overlays.get("overlay"))
    if isinstance(overlays, Mapping) and "traps" in overlays:
        return _overlay_for_record(record, overlays.get("traps"))
    if not isinstance(overlays, Mapping):
        return _coerce_overlay(overlays)
    candidates = [
        content_overlay_key(record.ref, record.content_hash),
        content_ref_key(record.pack_id, record.ref, record.content_hash),
        record.ref,
    ]
    for candidate in candidates:
        raw = overlays.get(candidate)
        if raw is not None:
            return _coerce_overlay(raw)
    return None


def _coerce_overlay(
    overlay: ContentTrapOverlayState | Mapping[str, Any] | None,
) -> ContentTrapOverlayState | None:
    if overlay is None:
        return None
    if isinstance(overlay, ContentTrapOverlayState):
        return overlay
    if isinstance(overlay, Mapping):
        return ContentTrapOverlayState.model_validate(overlay)
    return None


def _metadata_trap_hazard_records(
    metadata: Mapping[str, Any],
    *,
    pack_id: str,
) -> list[TrapHazardRecord | Mapping[str, Any]]:
    records: list[TrapHazardRecord | Mapping[str, Any]] = []
    for key in ("domain_catalog", "content_pack_domain_catalog"):
        raw_catalog = metadata.get(key)
        if raw_catalog is None:
            continue
        if isinstance(raw_catalog, ContentPackDomainCatalog):
            records.extend(raw_catalog.trap_hazards)
        elif isinstance(raw_catalog, Mapping):
            records.extend(
                _records_with_pack(
                    raw_catalog.get("trap_hazards") or (),
                    pack_id=pack_id,
                )
            )
    for key in (
        "trap_hazards",
        "dnd_trap_hazards",
        "imported_trap_hazards",
    ):
        records.extend(_records_with_pack(metadata.get(key) or (), pack_id=pack_id))
    return records


def _records_with_pack(
    raw_records: Any,
    *,
    pack_id: str,
) -> list[TrapHazardRecord | Mapping[str, Any]]:
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        return []
    records: list[TrapHazardRecord | Mapping[str, Any]] = []
    for raw in raw_records:
        if isinstance(raw, Mapping) and pack_id and not raw.get("pack_id"):
            records.append({**raw, "pack_id": pack_id})
        elif isinstance(raw, (TrapHazardRecord, Mapping)):
            records.append(raw)
    return records


def _pack_metadata(pack_state: Any) -> Mapping[str, Any]:
    metadata = (
        pack_state.get("metadata")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "metadata", None)
    )
    return metadata if isinstance(metadata, Mapping) else {}


def _pack_id(pack_key: Any, pack_state: Any) -> str:
    raw = (
        pack_state.get("pack_id")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "pack_id", "")
    )
    return str(raw or pack_key or "").strip()


def _coerce_record(
    record: TrapHazardRecord | Mapping[str, Any],
) -> TrapHazardRecord:
    if isinstance(record, TrapHazardRecord):
        return record
    return TrapHazardRecord.model_validate(record)


def _coerce_record_with_fields(
    record: TrapHazardRecord | Mapping[str, Any],
) -> tuple[TrapHazardRecord, frozenset[str]]:
    if isinstance(record, TrapHazardRecord):
        return record, frozenset(record.model_fields_set)
    return (
        TrapHazardRecord.model_validate(record),
        frozenset(str(key) for key in record),
    )


def _record_key(record: TrapHazardRecord) -> str:
    return content_ref_key(record.pack_id, record.ref, record.content_hash)


def _record_aliases(record: TrapHazardRecord) -> list[str]:
    aliases = [
        _record_key(record),
        record.ref,
        _clean_id(record.ref),
    ]
    if record.pack_id:
        scoped = f"{record.pack_id}::{record.ref}"
        aliases.extend([scoped, _clean_id(scoped)])
    return [alias for alias in dict.fromkeys(aliases) if alias]


def _context_text(value: str) -> str:
    return redact_imported_asset_text(value)


def _context_texts(values: Sequence[str]) -> list[str]:
    return [
        cleaned
        for value in values
        if (cleaned := _context_text(value))
    ]


def _drop_empty(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _drop_empty(item)) not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := _drop_empty(item)) not in (None, "", [], {})
        ]
    return value


def _log_token(value: str) -> str:
    text = _clean_text(redact_imported_asset_text(value))
    if not text:
        return "<empty>"
    return re.sub(r"[^A-Za-z0-9_.:/@+-]+", "_", text)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_id(value: str) -> str:
    text = _clean_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")
