from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.engine import (
    imported_statblocks,
    imported_trap_hazards,
    imported_treasure,
    tactical_map_templates,
)
from app.schemas.content_pack import (
    ContentPackDomainCatalog,
    EncounterTemplateRecord,
    TacticalMapTemplateRecord,
)
from app.schemas.dnd_monsters import DndCombatantSpawn
from app.schemas.dnd_spatial import DndBattleMapState


APPROVED_REVIEW_STATUSES = {"reviewed", "approved"}


class ImportedEncounterError(ValueError):
    """Base error for reviewed imported encounter resolution."""


class ImportedEncounterNotFoundError(ImportedEncounterError):
    """Raised when a requested imported encounter template is not authored."""


class ImportedEncounterValidationError(ImportedEncounterError):
    """Raised when an imported encounter template is not combat-start ready."""


@dataclass(frozen=True, slots=True)
class ResolvedImportedEncounter:
    """Reviewed encounter seed projected to combat-start inputs."""

    encounter_ref: str
    pack_id: str
    content_hash: str
    combatant_spawns: tuple[DndCombatantSpawn, ...]
    battle_map: DndBattleMapState | None
    trap_refs: tuple[str, ...]
    treasure_refs: tuple[str, ...]


class ImportedEncounterCatalog:
    """In-memory index of reviewed D&D encounter template records."""

    def __init__(
        self,
        records: Iterable[EncounterTemplateRecord | Mapping[str, Any]],
    ) -> None:
        self._records: dict[str, EncounterTemplateRecord] = {}
        self._aliases: dict[str, str] = {}
        for raw in records:
            record = _coerce_record(raw)
            if not record.ref:
                raise ImportedEncounterValidationError(
                    "Imported encounter template is missing ref."
                )
            key = _record_key(record)
            if key in self._records:
                raise ImportedEncounterValidationError(
                    f"Duplicate imported encounter template ref: {key}"
                )
            self._records[key] = record
            for alias in _record_aliases(record):
                existing = self._aliases.get(alias)
                if existing and existing != key:
                    raise ImportedEncounterValidationError(
                        "Imported encounter aliases collide: "
                        f"{alias} maps to both {existing} and {key}."
                    )
                self._aliases[alias] = key

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(record.ref for record in self._records.values())

    def get(self, ref: str, *, pack_id: str = "") -> EncounterTemplateRecord:
        return self._records[self._canonical_key(ref, pack_id=pack_id)]

    def resolve_for_location(
        self,
        location_ref: str,
    ) -> EncounterTemplateRecord | None:
        location = str(location_ref or "").strip()
        if not location:
            return None
        matches = [
            record
            for record in self._records.values()
            if location in record.location_refs
        ]
        if len(matches) > 1:
            refs = ", ".join(record.ref for record in matches)
            raise ImportedEncounterValidationError(
                "Multiple imported encounter templates match location "
                f"{location!r}: {refs}."
            )
        return matches[0] if matches else None

    def _canonical_key(self, ref: str, *, pack_id: str = "") -> str:
        cleaned = str(ref or "").strip()
        if not cleaned:
            raise ImportedEncounterNotFoundError(
                "Imported encounter template ref is not authored: <empty>"
            )
        candidates = [cleaned, _clean_id(cleaned)]
        if pack_id:
            scoped = f"{pack_id.strip()}::{cleaned}"
            candidates.extend([scoped, _clean_id(scoped)])
        for candidate in candidates:
            canonical = self._aliases.get(candidate)
            if canonical in self._records:
                return canonical
        raise ImportedEncounterNotFoundError(
            f"Imported encounter template ref is not authored: {cleaned}"
        )


def resolve_combat_start_from_content_state(
    content_state: Any,
    *,
    location_ref: str,
) -> ResolvedImportedEncounter | None:
    """Resolve the reviewed encounter template for the active module location."""

    catalog = catalog_from_content_state(content_state)
    if catalog is None:
        return None
    encounter = catalog.resolve_for_location(location_ref)
    if encounter is None:
        return None
    return resolve_encounter_template(
        encounter,
        content_state=content_state,
    )


def catalog_from_content_state(content_state: Any) -> ImportedEncounterCatalog | None:
    if not isinstance(content_state, Mapping):
        return None
    records: list[EncounterTemplateRecord | Mapping[str, Any]] = []
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        metadata = _pack_metadata(pack_state)
        records.extend(_metadata_encounter_records(metadata, pack_id=pack_id))
    if not records:
        return None
    return ImportedEncounterCatalog(records)


def resolve_encounter_template(
    encounter: EncounterTemplateRecord | Mapping[str, Any],
    *,
    content_state: Any,
) -> ResolvedImportedEncounter:
    record = _validated_encounter_record(encounter)
    statblock_catalog = imported_statblocks.catalog_from_content_state(content_state)
    if statblock_catalog is None:
        raise ImportedEncounterValidationError(
            "Imported encounter requires statblock catalog."
        )
    spawns: list[DndCombatantSpawn] = []
    for participant in record.participants:
        if not participant.statblock_ref:
            raise ImportedEncounterValidationError(
                f"Imported encounter {record.ref!r} participant "
                f"{participant.participant_id!r} has no statblock_ref."
            )
        statblock_catalog.resolve_monster_statblock(participant.statblock_ref)
        base_id = _clean_id(participant.participant_id or participant.statblock_ref)
        for index in range(1, participant.count + 1):
            character_id = base_id if participant.count == 1 else f"{base_id}_{index}"
            spawns.append(DndCombatantSpawn(
                character_id=character_id,
                monster_key="",
                statblock_ref=participant.statblock_ref,
                name="",
                location="",
                description=participant.tactics,
                statblock=None,
            ))

    battle_map = _resolved_battle_map(record, content_state)
    _validate_trap_refs(record, content_state)
    _validate_treasure_refs(record, content_state)
    return ResolvedImportedEncounter(
        encounter_ref=record.ref,
        pack_id=record.pack_id,
        content_hash=record.content_hash,
        combatant_spawns=tuple(spawns),
        battle_map=battle_map,
        trap_refs=tuple(record.trap_refs),
        treasure_refs=tuple(record.treasure_refs),
    )


def apply_resolved_encounter_to_router_output(
    result: Any,
    resolved: ResolvedImportedEncounter | None,
) -> None:
    """Merge reviewed encounter combat-start inputs into router output."""

    if resolved is None:
        return
    existing_spawns = list(getattr(result, "combatant_spawns", []) or [])
    seen_spawn_refs = {
        (
            str(getattr(spawn, "character_id", "") or ""),
            str(getattr(spawn, "statblock_ref", "") or ""),
        )
        for spawn in existing_spawns
    }
    for spawn in resolved.combatant_spawns:
        key = (spawn.character_id, spawn.statblock_ref)
        if key in seen_spawn_refs:
            continue
        existing_spawns.append(spawn)
        seen_spawn_refs.add(key)
    result.combatant_spawns = existing_spawns
    combatant_ids = list(getattr(result, "combatant_ids", []) or [])
    combatant_ids.extend(spawn.character_id for spawn in resolved.combatant_spawns)
    result.combatant_ids = [cid for cid in dict.fromkeys(combatant_ids) if cid]
    if resolved.battle_map is not None:
        result.battle_map_seed = resolved.battle_map


def _validated_encounter_record(
    record: EncounterTemplateRecord | Mapping[str, Any],
) -> EncounterTemplateRecord:
    encounter = _coerce_record(record)
    problems: list[str] = []
    if encounter.gate_status != "runtime_ready":
        problems.append(f"gate_status:{encounter.gate_status}")
    if encounter.review_status not in APPROVED_REVIEW_STATUSES:
        problems.append(f"review_status:{encounter.review_status}")
    if not encounter.content_hash:
        problems.append("content_hash")
    if not encounter.location_refs:
        problems.append("location_refs")
    if not encounter.participants:
        problems.append("participants")
    if not encounter.trigger:
        problems.append("trigger")
    if problems:
        raise ImportedEncounterValidationError(
            f"Imported encounter {encounter.ref!r} is not combat-start ready: "
            + ", ".join(dict.fromkeys(problems))
        )
    return encounter


def _resolved_battle_map(
    encounter: EncounterTemplateRecord,
    content_state: Any,
) -> DndBattleMapState | None:
    if not encounter.map_template_refs:
        return None
    map_records = _map_templates_from_content_state(content_state)
    authored_refs = _authored_refs_from_content_state(content_state)
    first_map: DndBattleMapState | None = None
    for map_ref in encounter.map_template_refs:
        template = _map_template_by_ref(map_records, map_ref)
        if template is None:
            raise ImportedEncounterValidationError(
                f"Imported encounter {encounter.ref!r} references missing "
                f"tactical map template {map_ref!r}."
            )
        compiled = tactical_map_templates.compile_tactical_map_template(
            template,
            authored_refs=authored_refs or None,
            required_layers=("map_ref", "spawn_anchors"),
        )
        if first_map is None:
            first_map = compiled.to_battle_map_state()
    return first_map


def _validate_trap_refs(
    encounter: EncounterTemplateRecord,
    content_state: Any,
) -> None:
    if not encounter.trap_refs:
        return
    catalog = imported_trap_hazards.catalog_from_content_state(content_state)
    if catalog is None:
        raise ImportedEncounterValidationError(
            f"Imported encounter {encounter.ref!r} requires trap/hazard catalog."
        )
    for trap_ref in encounter.trap_refs:
        catalog.resolve(trap_ref, pack_id=encounter.pack_id)


def _validate_treasure_refs(
    encounter: EncounterTemplateRecord,
    content_state: Any,
) -> None:
    if not encounter.treasure_refs:
        return
    for treasure_ref in encounter.treasure_refs:
        if _find_treasure_record(content_state, treasure_ref, pack_id=encounter.pack_id):
            continue
        raise ImportedEncounterValidationError(
            f"Imported encounter {encounter.ref!r} references missing treasure "
            f"{treasure_ref!r}."
        )


def _find_treasure_record(
    content_state: Any,
    treasure_ref: str,
    *,
    pack_id: str,
) -> bool:
    if not isinstance(content_state, Mapping):
        return False
    for pack_key, pack_state in content_state.items():
        state_pack_id = _pack_id(pack_key, pack_state)
        if pack_id and state_pack_id and pack_id != state_pack_id:
            continue
        catalog = imported_treasure.catalog_from_pack_state(pack_key, pack_state)
        if catalog is None:
            continue
        try:
            record = catalog.get(treasure_ref)
            imported_treasure.loot_offer_from_treasure_record(
                record,
                pack_id=state_pack_id,
            )
        except imported_treasure.ImportedTreasureError as exc:
            raise ImportedEncounterValidationError(str(exc)) from exc
        return True
    return False


def _map_templates_from_content_state(
    content_state: Any,
) -> list[TacticalMapTemplateRecord]:
    if not isinstance(content_state, Mapping):
        return []
    records: list[TacticalMapTemplateRecord] = []
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        metadata = _pack_metadata(pack_state)
        for raw in _metadata_map_template_records(metadata, pack_id=pack_id):
            records.append(
                raw if isinstance(raw, TacticalMapTemplateRecord)
                else TacticalMapTemplateRecord.model_validate(raw)
            )
    return records


def _map_template_by_ref(
    records: Sequence[TacticalMapTemplateRecord],
    ref: str,
) -> TacticalMapTemplateRecord | None:
    cleaned = str(ref or "").strip()
    cleaned_id = _clean_id(cleaned)
    for record in records:
        if cleaned in {record.ref, _record_key(record)}:
            return record
        if cleaned_id and cleaned_id == _clean_id(record.ref):
            return record
    return None


def _authored_refs_from_content_state(content_state: Any) -> set[str]:
    refs: set[str] = set()
    if not isinstance(content_state, Mapping):
        return refs
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        metadata = _pack_metadata(pack_state)
        for records in _all_domain_record_groups(metadata, pack_id=pack_id):
            for raw in records:
                ref = (
                    str(raw.get("ref") or "").strip()
                    if isinstance(raw, Mapping)
                    else str(getattr(raw, "ref", "") or "").strip()
                )
                if ref:
                    refs.add(ref)
    return refs


def _all_domain_record_groups(
    metadata: Mapping[str, Any],
    *,
    pack_id: str,
) -> list[list[Any]]:
    groups: list[list[Any]] = []
    for key in ("domain_catalog", "content_pack_domain_catalog"):
        raw_catalog = metadata.get(key)
        if isinstance(raw_catalog, ContentPackDomainCatalog):
            groups.extend([
                list(raw_catalog.locations),
                list(raw_catalog.keyed_areas),
                list(raw_catalog.tactical_map_templates),
                list(raw_catalog.statblocks),
                list(raw_catalog.trap_hazards),
                list(raw_catalog.treasures),
                list(raw_catalog.encounter_templates),
            ])
        elif isinstance(raw_catalog, Mapping):
            for record_key in _DOMAIN_RECORD_KEYS:
                groups.append(_records_with_pack(
                    raw_catalog.get(record_key) or (),
                    pack_id=pack_id,
                ))
    for record_key in _DOMAIN_RECORD_KEYS:
        groups.append(_records_with_pack(metadata.get(record_key) or (), pack_id=pack_id))
    return groups


def _metadata_encounter_records(
    metadata: Mapping[str, Any],
    *,
    pack_id: str,
) -> list[EncounterTemplateRecord | Mapping[str, Any]]:
    records: list[EncounterTemplateRecord | Mapping[str, Any]] = []
    for key in ("domain_catalog", "content_pack_domain_catalog"):
        raw_catalog = metadata.get(key)
        if isinstance(raw_catalog, ContentPackDomainCatalog):
            records.extend(raw_catalog.encounter_templates)
        elif isinstance(raw_catalog, Mapping):
            records.extend(_records_with_pack(
                raw_catalog.get("encounter_templates") or (),
                pack_id=pack_id,
            ))
    for key in ("encounter_templates", "dnd_encounters", "imported_encounters"):
        records.extend(_records_with_pack(metadata.get(key) or (), pack_id=pack_id))
    return records


def _metadata_map_template_records(
    metadata: Mapping[str, Any],
    *,
    pack_id: str,
) -> list[TacticalMapTemplateRecord | Mapping[str, Any]]:
    records: list[TacticalMapTemplateRecord | Mapping[str, Any]] = []
    for key in ("domain_catalog", "content_pack_domain_catalog"):
        raw_catalog = metadata.get(key)
        if isinstance(raw_catalog, ContentPackDomainCatalog):
            records.extend(raw_catalog.tactical_map_templates)
        elif isinstance(raw_catalog, Mapping):
            records.extend(_records_with_pack(
                raw_catalog.get("tactical_map_templates") or (),
                pack_id=pack_id,
            ))
    for key in ("tactical_map_templates", "map_templates", "dnd_maps"):
        records.extend(_records_with_pack(metadata.get(key) or (), pack_id=pack_id))
    return records


def _records_with_pack(raw_records: Any, *, pack_id: str) -> list[Any]:
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        return []
    records: list[Any] = []
    for raw in raw_records:
        if isinstance(raw, Mapping) and pack_id and not raw.get("pack_id"):
            records.append({**raw, "pack_id": pack_id})
        else:
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
    record: EncounterTemplateRecord | Mapping[str, Any],
) -> EncounterTemplateRecord:
    if isinstance(record, EncounterTemplateRecord):
        return record
    return EncounterTemplateRecord.model_validate(record)


def _record_key(record: Any) -> str:
    pack_id = str(getattr(record, "pack_id", "") or "").strip()
    ref = str(getattr(record, "ref", "") or "").strip()
    content_hash = str(getattr(record, "content_hash", "") or "").strip()
    return "::".join(part for part in (pack_id, ref, content_hash) if part)


def _record_aliases(record: EncounterTemplateRecord) -> list[str]:
    aliases = [_record_key(record), record.ref, _clean_id(record.ref)]
    if record.pack_id:
        scoped = f"{record.pack_id}::{record.ref}"
        aliases.extend([scoped, _clean_id(scoped)])
    return [alias for alias in dict.fromkeys(aliases) if alias]


def _clean_id(value: str) -> str:
    text = " ".join(str(value or "").strip().split()).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


_DOMAIN_RECORD_KEYS = (
    "locations",
    "keyed_areas",
    "tactical_map_templates",
    "statblocks",
    "trap_hazards",
    "treasures",
    "encounter_templates",
)
