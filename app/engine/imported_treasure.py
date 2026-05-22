from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.schemas.checkpoint import CheckpointFile
from app.schemas.content import (
    ContentOverlayState,
    ContentPackState,
    ContentTreasureOverlayState,
    content_overlay_key,
)
from app.schemas.content_pack import (
    ContentPackDomainCatalog,
    TreasureRecord,
)
from app.schemas.content_privacy import sanitize_player_safe_text
from app.schemas.dnd_inventory import DndCurrency, DndLootOffer, DndLootOfferItem


APPROVED_REVIEW_STATUSES = {"reviewed", "approved"}
_COIN_KEYS = ("cp", "sp", "ep", "gp", "pp")
_CURRENCY_ALIASES = {
    "copper": "cp",
    "copper_piece": "cp",
    "copper_pieces": "cp",
    "cp": "cp",
    "silver": "sp",
    "silver_piece": "sp",
    "silver_pieces": "sp",
    "sp": "sp",
    "electrum": "ep",
    "electrum_piece": "ep",
    "electrum_pieces": "ep",
    "ep": "ep",
    "gold": "gp",
    "gold_piece": "gp",
    "gold_pieces": "gp",
    "gp": "gp",
    "platinum": "pp",
    "platinum_piece": "pp",
    "platinum_pieces": "pp",
    "pp": "pp",
}


class ImportedTreasureError(ValueError):
    """Base error for reviewed imported treasure resolution."""


class ImportedTreasureNotFoundError(ImportedTreasureError):
    """Raised when a revealed treasure overlay references no authored record."""


class ImportedTreasureValidationError(ImportedTreasureError):
    """Raised when a revealed imported treasure record is not offer-ready."""


class ImportedTreasureCatalog:
    """In-memory index of reviewed D&D treasure pack records."""

    def __init__(
        self,
        records: Iterable[TreasureRecord | Mapping[str, Any]],
    ) -> None:
        self._records: dict[str, TreasureRecord] = {}
        self._aliases: dict[str, str] = {}
        for raw in records:
            record = _coerce_record(raw)
            ref = record.ref.strip()
            if not ref:
                raise ImportedTreasureValidationError(
                    "Imported D&D treasure record is missing ref."
                )
            if ref in self._records:
                raise ImportedTreasureValidationError(
                    f"Duplicate imported D&D treasure ref: {ref}"
                )
            self._records[ref] = record
            for alias in (ref, _clean_id(ref)):
                if not alias:
                    continue
                existing = self._aliases.get(alias)
                if existing and existing != ref:
                    raise ImportedTreasureValidationError(
                        "Imported D&D treasure aliases collide: "
                        f"{alias} maps to both {existing} and {ref}."
                    )
                self._aliases[alias] = ref

    @classmethod
    def from_domain_catalog(
        cls,
        catalog: ContentPackDomainCatalog | Mapping[str, Any],
    ) -> "ImportedTreasureCatalog":
        if isinstance(catalog, Mapping):
            pack_id = str(catalog.get("pack_id") or "").strip()
            records = []
            for raw in catalog.get("treasures") or ():
                if isinstance(raw, Mapping) and pack_id and not raw.get("pack_id"):
                    records.append({**raw, "pack_id": pack_id})
                else:
                    records.append(raw)
            return cls(records)
        return cls(catalog.treasures)

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(self._records)

    def get(self, ref: str) -> TreasureRecord:
        cleaned = str(ref or "").strip()
        canonical_ref = self._aliases.get(cleaned) or self._aliases.get(
            _clean_id(cleaned)
        )
        if not cleaned or canonical_ref not in self._records:
            raise ImportedTreasureNotFoundError(
                f"Imported D&D treasure ref is not authored: {cleaned or '<empty>'}"
            )
        return self._records[canonical_ref]


def apply_revealed_treasure_offers(
    ckpt: CheckpointFile,
) -> dict[str, list[str]]:
    """Create pending loot offers for explicitly revealed imported treasure."""

    if not _ruleset_is_dnd(ckpt):
        return {}
    content_state = getattr(ckpt.session, "content_state", None)
    if not isinstance(content_state, Mapping):
        return {}

    existing_source_ids = {
        offer.source_event_id
        for offer in getattr(ckpt.session, "dnd_inventory_offers", []) or []
    }
    prompts: dict[str, list[str]] = {}
    for pack_key, pack_state in content_state.items():
        catalog = catalog_from_pack_state(pack_key, pack_state)
        if catalog is None:
            continue
        overlay = _pack_overlay(pack_state)
        if overlay is None:
            continue
        pack_id = _pack_id(pack_key, pack_state)
        for treasure_state in list(overlay.treasures.values()):
            if not treasure_state.revealed or treasure_state.depleted:
                continue
            record = catalog.get(treasure_state.treasure_id)
            if treasure_state.content_hash and (
                treasure_state.content_hash != record.content_hash
            ):
                raise ImportedTreasureValidationError(
                    "Revealed imported treasure overlay hash does not match "
                    f"record {record.ref!r}."
                )
            source_event_id = treasure_source_event_id(record, pack_id=pack_id)
            if source_event_id in existing_source_ids:
                continue
            eligible = _eligible_character_ids(ckpt, record)
            if not eligible:
                continue
            offer = loot_offer_from_treasure_record(
                record,
                pack_id=pack_id,
                overlay_state=treasure_state,
                eligible_character_ids=eligible,
            )
            if not offer.items and offer.currency.is_empty():
                continue
            ckpt.session.dnd_inventory_offers.append(offer)
            existing_source_ids.add(source_event_id)
            _mark_offer_refs_remaining(overlay, treasure_state, offer)
            for cid in offer.eligible_character_ids:
                prompts.setdefault(cid, []).append(offer.offer_id)
    return prompts


def catalog_from_pack_state(
    pack_key: Any,
    pack_state: Any,
) -> ImportedTreasureCatalog | None:
    records: list[TreasureRecord | Mapping[str, Any]] = []
    pack_id = _pack_id(pack_key, pack_state)
    metadata = _pack_metadata(pack_state)
    for key in ("domain_catalog", "content_pack_domain_catalog"):
        raw_catalog = metadata.get(key)
        if raw_catalog is None:
            continue
        if isinstance(raw_catalog, ContentPackDomainCatalog):
            records.extend(raw_catalog.treasures)
        elif isinstance(raw_catalog, Mapping):
            raw_records = raw_catalog.get("treasures") or ()
            records.extend(_records_with_pack(raw_records, pack_id=pack_id))
    for key in ("treasures", "dnd_treasures", "imported_treasures"):
        records.extend(_records_with_pack(metadata.get(key) or (), pack_id=pack_id))
    if not records:
        return None
    return ImportedTreasureCatalog(records)


def loot_offer_from_treasure_record(
    record: TreasureRecord | Mapping[str, Any],
    *,
    pack_id: str = "",
    overlay_state: ContentTreasureOverlayState | None = None,
    eligible_character_ids: Sequence[str] = (),
) -> DndLootOffer:
    treasure = _validated_offer_record(record)
    state = overlay_state or ContentTreasureOverlayState(
        treasure_id=treasure.ref,
        content_hash=treasure.content_hash,
        revealed=True,
    )
    available_refs = _available_refs_for_state(treasure, state)
    currency_refs = set(_currency_refs(treasure))
    items = [
        offer_item_from_treasure_item(item, idx=idx)
        for idx, item in enumerate(treasure.items, start=1)
        if _treasure_item_ref(item, idx=idx) in available_refs
    ]
    currency = _offer_currency_from_record(
        treasure,
        available_currency_refs=currency_refs.intersection(available_refs),
    )
    return DndLootOffer(
        offer_id=treasure_offer_id(treasure, pack_id=pack_id),
        source_event_id=treasure_source_event_id(treasure, pack_id=pack_id),
        source_kind=_source_kind(treasure),
        source_label=_source_label(treasure),
        source_pack_id=pack_id or treasure.pack_id,
        source_ref=treasure.ref,
        source_content_hash=treasure.content_hash,
        source_depletion_ref=treasure.depletion_ref,
        visibility="table",
        eligible_character_ids=list(dict.fromkeys(eligible_character_ids)),
        items=_dedupe_items(items),
        currency=currency,
        notes=sanitize_player_safe_text(treasure.summary),
    )


def offer_item_from_treasure_item(item: Any, *, idx: int) -> DndLootOfferItem:
    item_id = _treasure_item_ref(item, idx=idx)
    name = sanitize_player_safe_text(getattr(item, "name", "") or "") or item_id
    return DndLootOfferItem(
        item_id=item_id,
        name=name,
        kind=(getattr(item, "item_type", "") or "gear"),
        quantity=max(1, int(getattr(item, "quantity", 1) or 1)),
        identified=bool(getattr(item, "identified", True)),
        requires_identification=not bool(getattr(item, "identified", True)),
        requires_attunement=bool(getattr(item, "requires_attunement", False)),
        consumable=bool(getattr(item, "consumable", False)),
        value_gp=max(0.0, float(getattr(item, "value_gp", 0) or 0)),
        weight=max(0.0, float(getattr(item, "weight_lb", 0) or 0)),
        notes=sanitize_player_safe_text(getattr(item, "rarity", "") or ""),
    )


def update_treasure_claim_state_from_offer(
    ckpt: CheckpointFile,
    offer: DndLootOffer,
    *,
    claimed_item_ids: Iterable[str] = (),
    currency_claimed: bool = False,
) -> None:
    """Mirror imported loot claims into checkpoint content overlay state."""

    if not offer.source_ref:
        return
    treasure_state = _overlay_state_for_offer(ckpt, offer)
    if treasure_state is None:
        return

    all_refs = _offer_takeable_refs(offer)
    remaining = list(treasure_state.remaining_ref_ids or all_refs)
    claimed = list(treasure_state.claimed_ref_ids)
    claim_set = set(str(item_id).strip() for item_id in claimed_item_ids if item_id)
    if currency_claimed:
        claim_set.update(_offer_currency_refs(offer))
    for ref in all_refs:
        if ref in claim_set and ref not in claimed:
            claimed.append(ref)
    remaining = [ref for ref in remaining if ref not in set(claimed)]
    if not remaining:
        remaining = [ref for ref in all_refs if ref not in set(claimed)]

    treasure_state.revealed = True
    treasure_state.looted = bool(claimed)
    treasure_state.claimed_ref_ids = claimed
    treasure_state.remaining_ref_ids = remaining
    treasure_state.depleted = bool(all_refs) and not remaining
    _rekey_offer_overlay(ckpt, offer, treasure_state)


def treasure_offer_id(record: TreasureRecord, *, pack_id: str = "") -> str:
    parts = [
        "loot",
        "imported",
        _slug(pack_id or record.pack_id),
        _slug(record.ref),
        _slug(record.content_hash)[-12:],
    ]
    return "_".join(part for part in parts if part)


def treasure_source_event_id(record: TreasureRecord, *, pack_id: str = "") -> str:
    return "::".join(
        (
            "imported_treasure",
            pack_id or record.pack_id,
            record.ref,
            record.content_hash,
        )
    )


def _validated_offer_record(
    record: TreasureRecord | Mapping[str, Any],
) -> TreasureRecord:
    treasure = _coerce_record(record)
    problems: list[str] = []
    if treasure.gate_status != "runtime_ready":
        problems.append(f"gate_status:{treasure.gate_status}")
    if treasure.review_status not in APPROVED_REVIEW_STATUSES:
        problems.append(f"review_status:{treasure.review_status}")
    if not treasure.content_hash:
        problems.append("content_hash")
    if not (treasure.items or treasure.currency):
        problems.append("contents")
    for currency in treasure.currency:
        if currency.amount and _currency_key(currency.denomination) not in _COIN_KEYS:
            problems.append(f"currency:{currency.denomination}")
    if problems:
        raise ImportedTreasureValidationError(
            f"Imported D&D treasure {treasure.ref!r} is not offer-ready: "
            + ", ".join(dict.fromkeys(problems))
        )
    return treasure


def _eligible_character_ids(
    ckpt: CheckpointFile,
    record: TreasureRecord,
) -> list[str]:
    active_ids = {char.character_id for char in ckpt.characters}
    bound_ids = set((ckpt.session.character_bindings or {}).keys())
    explicit = [
        cid
        for cid in record.eligible_character_refs
        if cid in active_ids and cid in bound_ids
    ]
    if record.eligible_character_refs:
        return explicit
    return [
        char.character_id
        for char in ckpt.characters
        if char.character_id in active_ids and char.character_id in bound_ids
    ]


def _ruleset_is_dnd(ckpt: CheckpointFile) -> bool:
    settings = getattr(getattr(ckpt.session, "config", None), "settings", None)
    ruleset_id = str(getattr(settings, "ruleset_id", "") or "").lower()
    return ruleset_id.startswith("dnd")


def _mark_offer_refs_remaining(
    overlay: ContentOverlayState,
    treasure_state: ContentTreasureOverlayState,
    offer: DndLootOffer,
) -> None:
    refs = _offer_takeable_refs(offer)
    if refs and not treasure_state.remaining_ref_ids:
        treasure_state.remaining_ref_ids = refs
    treasure_state.revealed = True
    key = treasure_state.overlay_key()
    if key:
        overlay.treasures[key] = treasure_state


def _overlay_state_for_offer(
    ckpt: CheckpointFile,
    offer: DndLootOffer,
) -> ContentTreasureOverlayState | None:
    content_state = getattr(ckpt.session, "content_state", None)
    if not isinstance(content_state, Mapping):
        return None
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        if offer.source_pack_id and offer.source_pack_id != pack_id:
            continue
        overlay = _pack_overlay(pack_state)
        if overlay is None:
            continue
        keys = [
            content_overlay_key(offer.source_ref, offer.source_content_hash),
            content_overlay_key(offer.source_ref),
        ]
        for key in keys:
            if key in overlay.treasures:
                return overlay.treasures[key]
        treasure_state = ContentTreasureOverlayState(
            treasure_id=offer.source_ref,
            content_hash=offer.source_content_hash,
            revealed=True,
        )
        key = treasure_state.overlay_key()
        if key:
            overlay.treasures[key] = treasure_state
            return treasure_state
    return None


def _rekey_offer_overlay(
    ckpt: CheckpointFile,
    offer: DndLootOffer,
    treasure_state: ContentTreasureOverlayState,
) -> None:
    content_state = getattr(ckpt.session, "content_state", None)
    if not isinstance(content_state, Mapping):
        return
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        if offer.source_pack_id and offer.source_pack_id != pack_id:
            continue
        overlay = _pack_overlay(pack_state)
        if overlay is None:
            continue
        for key in (
            content_overlay_key(offer.source_ref, offer.source_content_hash),
            content_overlay_key(offer.source_ref),
        ):
            overlay.treasures.pop(key, None)
        overlay.treasures[treasure_state.overlay_key()] = treasure_state


def _available_refs_for_state(
    record: TreasureRecord,
    state: ContentTreasureOverlayState,
) -> set[str]:
    all_refs = _record_takeable_refs(record)
    if state.depleted:
        return set()
    if state.remaining_ref_ids:
        return set(state.remaining_ref_ids).intersection(all_refs)
    if state.claimed_ref_ids:
        return all_refs - set(state.claimed_ref_ids)
    if state.looted:
        return set()
    return all_refs


def _record_takeable_refs(record: TreasureRecord) -> set[str]:
    return set(_item_refs(record)).union(_currency_refs(record))


def _offer_takeable_refs(offer: DndLootOffer) -> list[str]:
    refs = [item.item_id for item in offer.items if item.item_id]
    refs.extend(_offer_currency_refs(offer))
    return list(dict.fromkeys(refs))


def _offer_currency_refs(offer: DndLootOffer) -> list[str]:
    return [
        f"currency/{key}"
        for key in _COIN_KEYS
        if int(getattr(offer.currency, key, 0) or 0) > 0
    ]


def _item_refs(record: TreasureRecord) -> list[str]:
    return [
        _treasure_item_ref(item, idx=idx)
        for idx, item in enumerate(record.items, start=1)
    ]


def _currency_refs(record: TreasureRecord) -> list[str]:
    currency = _currency_dict(record)
    return [f"currency/{key}" for key, amount in currency.items() if amount > 0]


def _offer_currency_from_record(
    record: TreasureRecord,
    *,
    available_currency_refs: set[str],
) -> DndCurrency:
    currency = {
        key: amount
        for key, amount in _currency_dict(record).items()
        if f"currency/{key}" in available_currency_refs
    }
    return DndCurrency(**{key: currency.get(key, 0) for key in _COIN_KEYS})


def _currency_dict(record: TreasureRecord) -> dict[str, int]:
    out = {key: 0 for key in _COIN_KEYS}
    for currency in record.currency:
        key = _currency_key(currency.denomination)
        if key in out:
            out[key] += max(0, int(currency.amount or 0))
    return out


def _currency_key(denomination: str) -> str:
    clean = re.sub(r"[^a-z]+", "_", denomination.strip().lower()).strip("_")
    return _CURRENCY_ALIASES.get(clean, clean)


def _treasure_item_ref(item: Any, *, idx: int) -> str:
    item_ref = str(getattr(item, "item_ref", "") or "").strip()
    if item_ref:
        return item_ref
    name = str(getattr(item, "name", "") or "").strip()
    return _slug(name) or f"item_{idx}"


def _source_kind(record: TreasureRecord) -> str:
    if record.treasure_kind == "reward":
        return "reward"
    if _structured_ref_starts_with_body(record.ref) or _structured_ref_starts_with_body(
        record.container_ref
    ):
        return "body"
    if record.treasure_kind == "container":
        return "container"
    return "other"


def _source_label(record: TreasureRecord) -> str:
    return (
        sanitize_player_safe_text(record.title)
        or sanitize_player_safe_text(record.ref)
        or "Imported treasure"
    )


def _structured_ref_starts_with_body(value: str) -> bool:
    return bool(re.match(r"^(?:body|corpse|remains)(?:[./:_-]|$)", value.strip()))


def _dedupe_items(items: list[DndLootOfferItem]) -> list[DndLootOfferItem]:
    out: list[DndLootOfferItem] = []
    seen: set[str] = set()
    for idx, item in enumerate(items, start=1):
        item_id = item.item_id or f"item_{idx}"
        base = item_id
        suffix = 2
        while item_id in seen:
            item_id = f"{base}_{suffix}"
            suffix += 1
        seen.add(item_id)
        if item_id != item.item_id:
            item = DndLootOfferItem(**{**item.model_dump(), "item_id": item_id})
        out.append(item)
    return out


def _coerce_record(record: TreasureRecord | Mapping[str, Any]) -> TreasureRecord:
    if isinstance(record, TreasureRecord):
        return record
    return TreasureRecord.model_validate(record)


def _records_with_pack(
    raw_records: Any,
    *,
    pack_id: str,
) -> list[TreasureRecord | Mapping[str, Any]]:
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        return []
    records: list[TreasureRecord | Mapping[str, Any]] = []
    for raw in raw_records:
        if isinstance(raw, Mapping) and pack_id and not raw.get("pack_id"):
            records.append({**raw, "pack_id": pack_id})
        elif isinstance(raw, (TreasureRecord, Mapping)):
            records.append(raw)
    return records


def _pack_metadata(pack_state: Any) -> Mapping[str, Any]:
    metadata = (
        pack_state.get("metadata")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "metadata", None)
    )
    return metadata if isinstance(metadata, Mapping) else {}


def _pack_overlay(pack_state: Any) -> ContentOverlayState | None:
    overlay = (
        pack_state.get("overlay")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "overlay", None)
    )
    return overlay if isinstance(overlay, ContentOverlayState) else None


def _pack_id(pack_key: Any, pack_state: Any) -> str:
    raw = (
        pack_state.get("pack_id")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "pack_id", "")
    )
    return str(raw or pack_key or "").strip()


def _clean_id(value: str) -> str:
    return _slug(value)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug[:64]
