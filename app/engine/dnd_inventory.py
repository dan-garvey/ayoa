from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Iterable

from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_inventory import (
    DndLootOffer,
    DndLootOfferItem,
    DndLootOfferSignal,
)
from app.schemas.event_router import EventRouterOutput


_COIN_KEYS = ("cp", "sp", "ep", "gp", "pp")
_CLOSED_OFFER_RETAIN = 25
_DND5E_BASIC_RULESET_ID = "dnd5e_basic"
_NARRATIVE_LOOT_NOTES = "Inferred from a visible mundane gear handoff."
_NARRATIVE_HANDOFF_PATTERNS = (
    re.compile(
        r"\b(?:sets?|puts?|places?)\b.{0,120}\b(?:within reach|down|on top)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:hands?|passes?|gives?|issues?|offers?|supplies?|provides?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:returns?|comes back)\b.{0,120}\b(?:carrying|with)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bfolds?\b.{0,120}\b(?:parcels?|wraps?)\b.{0,120}\b(?:counter|hand|hands)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bholding\b.{0,80}\b(?:waterskins?|water\s+skins?)\b.{0,80}\bopen\b",
        re.IGNORECASE | re.DOTALL,
    ),
)
_NARRATIVE_GEAR_PATTERNS: tuple[tuple[re.Pattern[str], dict[str, Any]], ...] = (
    (
        re.compile(r"\bhealer(?:'s|s)?\s+kit\b", re.IGNORECASE),
        {
            "item_id": "healers_kit",
            "name": "Healer's Kit",
            "kind": "tool",
            "quantity": 1,
            "consumable": True,
            "weight": 3.0,
        },
    ),
    (
        re.compile(r"\b(?:coiled\s+)?rope\b", re.IGNORECASE),
        {
            "item_id": "rope",
            "name": "Rope",
            "kind": "gear",
            "quantity": 1,
            "weight": 10.0,
        },
    ),
    (
        re.compile(r"\bbedrolls?\b", re.IGNORECASE),
        {
            "item_id": "bedroll",
            "name": "Bedroll",
            "kind": "gear",
            "quantity": 1,
            "weight": 7.0,
        },
    ),
    (
        re.compile(r"\bration(?:s|\s+wraps?)?\b", re.IGNORECASE),
        {
            "item_id": "rations",
            "name": "Rations",
            "kind": "consumable",
            "quantity": 1,
            "consumable": True,
            "weight": 2.0,
        },
    ),
    (
        re.compile(r"\b(?:torches?|torch\s+bundle)\b", re.IGNORECASE),
        {
            "item_id": "torch_bundle",
            "name": "Torch Bundle",
            "kind": "gear",
            "quantity": 1,
            "weight": 5.0,
        },
    ),
    (
        re.compile(r"\b(?:rough\s+map|map\s+sheet|folded\s+map|map\s+fold)\b", re.IGNORECASE),
        {
            "item_id": "rough_map",
            "name": "Rough Map",
            "kind": "gear",
            "quantity": 1,
            "weight": 0.0,
        },
    ),
    (
        re.compile(r"\b(?:waterskins?|water\s+skins?)\b", re.IGNORECASE),
        {
            "item_id": "waterskin",
            "name": "Waterskin",
            "kind": "gear",
            "quantity": 1,
            "weight": 5.0,
        },
    ),
    (
        re.compile(
            r"\b(?:cloth-wrapped\s+parcels?|wrapped\s+(?:meals?|food)|"
            r"stew|bread|cheese|apples)\b",
            re.IGNORECASE,
        ),
        {
            "item_id": "wrapped_meal",
            "name": "Wrapped Meal",
            "kind": "consumable",
            "quantity": 1,
            "consumable": True,
            "weight": 1.0,
        },
    ),
)


def inventory_view(character: CharacterRecord) -> dict[str, Any]:
    """Return the live D&D inventory for display.

    Imported D&D Beyond inventory is the baseline. The first mutation creates a
    `mechanics.dnd5e_runtime.inventory` overlay so imports remain rewindable.
    """

    runtime = _runtime_inventory(character)
    if runtime is not None:
        return deepcopy(runtime)
    return _imported_inventory(character)


def apply_loot_offers_from_events(
    ckpt: CheckpointFile,
    events: Iterable[EventRouterOutput],
) -> dict[str, list[str]]:
    """Copy router-authored D&D loot signals into pending checkpoint offers.

    Returns character_id -> newly available offer ids for frontend prompts.
    Re-running with the same source event is idempotent.
    """

    prune_inventory_offers(ckpt)
    prompts: dict[str, list[str]] = {}
    existing_event_ids = {
        offer.source_event_id
        for offer in getattr(ckpt.session, "dnd_inventory_offers", [])
    }
    bindings = ckpt.session.character_bindings or {}
    for event in events:
        signal = getattr(event, "loot_offer", None)
        if signal is None:
            signal = _narrative_loot_offer_signal(ckpt, event)
        if isinstance(signal, dict):
            signal = DndLootOfferSignal(**signal)
        if isinstance(signal, DndLootOfferSignal) and not signal.present:
            inferred_signal = _narrative_loot_offer_signal(ckpt, event)
            if inferred_signal is not None:
                signal = inferred_signal
        if not isinstance(signal, DndLootOfferSignal) or not signal.present:
            continue
        if event.event_id in existing_event_ids:
            continue
        if not signal.items and signal.currency.is_empty():
            continue

        eligible = _eligible_ids_for_signal(ckpt, event, signal)
        if not eligible:
            continue
        offer = DndLootOffer(
            offer_id=_offer_id(event.event_id),
            source_event_id=event.event_id,
            source_kind=signal.source_kind,
            source_label=signal.source_label,
            visibility=signal.visibility,
            eligible_character_ids=eligible,
            items=_normalized_offer_items(signal.items),
            currency=signal.currency,
            notes=signal.notes,
            created_turn_index=ckpt.session.turn_index,
        )
        ckpt.session.dnd_inventory_offers.append(offer)
        existing_event_ids.add(event.event_id)
        for cid in offer.eligible_character_ids:
            if cid in bindings:
                prompts.setdefault(cid, []).append(offer.offer_id)
    return prompts


def open_loot_offers_for_character(
    ckpt: CheckpointFile,
    character_id: str,
) -> list[DndLootOffer]:
    prune_inventory_offers(ckpt)
    return [
        offer
        for offer in ckpt.session.dnd_inventory_offers
        if _offer_is_available_to(ckpt, offer, character_id)
    ]


def claim_loot(
    ckpt: CheckpointFile,
    *,
    character_id: str,
    offer_id: str,
    item_ids: Iterable[str],
    take_currency: bool = False,
    take_all_available: bool = False,
) -> dict[str, Any]:
    prune_inventory_offers(ckpt)
    offer = _find_open_offer(ckpt, offer_id)
    if not _character_is_eligible(ckpt, offer, character_id):
        raise ValueError("That loot offer is not available to this character.")

    available_items = {
        item.item_id: item
        for item in offer.items
        if item.item_id not in set(offer.claimed_item_ids)
    }
    selected = (
        list(available_items)
        if take_all_available
        else [iid.strip() for iid in item_ids if iid.strip()]
    )
    selected_set = set(selected)
    if selected_set - set(available_items):
        missing = ", ".join(sorted(selected_set - set(available_items)))
        raise ValueError(
            "Item(s) already claimed or no longer available: "
            f"{missing}. Use /loot list to see what remains."
        )
    if take_currency and not offer.has_available_currency():
        if take_all_available:
            take_currency = False
        else:
            raise ValueError(
                "Currency is no longer available in that offer. "
                "Use /loot list to see what remains."
            )
    if not selected and not (take_currency and offer.has_available_currency()):
        if take_all_available:
            raise ValueError(
                "That loot offer has nothing left to claim. "
                "Use /loot list to see open offers."
            )
        raise ValueError("Choose at least one item or currency to claim.")

    character = _character_by_id(ckpt, character_id)
    inventory = _ensure_runtime_inventory(character)

    claimed_items: list[dict[str, Any]] = []
    for item_id in selected:
        item = available_items[item_id]
        inventory_item = _inventory_item_from_offer(item, offer)
        inventory_item["id"] = _unique_inventory_item_id(
            inventory,
            inventory_item["id"],
        )
        inventory_item["item_id"] = inventory_item["id"]
        inventory.setdefault("items", []).append(inventory_item)
        offer.claimed_item_ids.append(item.item_id)
        claimed_items.append(inventory_item)

    claimed_currency: dict[str, int] = {}
    if take_currency:
        currency = _currency_dict(inventory.get("currency") or {})
        for key in _COIN_KEYS:
            amount = int(getattr(offer.currency, key))
            if amount:
                currency[key] = int(currency.get(key, 0)) + amount
                claimed_currency[key] = amount
        inventory["currency"] = currency
        offer.currency_claimed = True

    _close_offer_if_empty_or_declined(ckpt, offer)
    prune_inventory_offers(ckpt)
    return {
        "offer_id": offer.offer_id,
        "character_id": character_id,
        "claimed_items": claimed_items,
        "claimed_currency": claimed_currency,
        "offer_closed": offer.status == "closed",
    }


def split_loot_currency(
    ckpt: CheckpointFile,
    *,
    offer_id: str,
    actor_id: str,
) -> dict[str, Any]:
    prune_inventory_offers(ckpt)
    offer = _find_open_offer(ckpt, offer_id)
    if not _character_is_eligible(ckpt, offer, actor_id):
        raise ValueError("That loot offer is not available to this character.")
    if not offer.has_available_currency():
        raise ValueError("Currency is no longer available in that offer.")

    recipients = _currency_split_recipients(ckpt, offer, actor_id)
    if not recipients:
        raise ValueError("No eligible characters can receive that currency.")

    shares = {cid: {key: 0 for key in _COIN_KEYS} for cid in recipients}
    ordered = [actor_id, *[cid for cid in recipients if cid != actor_id]]
    for key in _COIN_KEYS:
        amount = int(getattr(offer.currency, key))
        if amount <= 0:
            continue
        base, remainder = divmod(amount, len(recipients))
        for cid in recipients:
            shares[cid][key] = base
        for cid in ordered[:remainder]:
            shares[cid][key] += 1

    for cid, share in shares.items():
        character = _character_by_id(ckpt, cid)
        inventory = _ensure_runtime_inventory(character)
        currency = _currency_dict(inventory.get("currency") or {})
        for key, amount in share.items():
            if amount:
                currency[key] = int(currency.get(key, 0)) + amount
        inventory["currency"] = currency

    offer.currency_claimed = True
    _close_offer_if_empty_or_declined(ckpt, offer)
    prune_inventory_offers(ckpt)
    return {
        "offer_id": offer.offer_id,
        "shares": shares,
        "offer_closed": offer.status == "closed",
    }


def decline_loot(
    ckpt: CheckpointFile,
    *,
    character_id: str,
    offer_id: str,
) -> dict[str, Any]:
    prune_inventory_offers(ckpt)
    offer = _find_open_offer(ckpt, offer_id)
    if not _character_is_eligible(ckpt, offer, character_id):
        raise ValueError("That loot offer is not available to this character.")
    if character_id not in offer.declined_by_character_ids:
        offer.declined_by_character_ids.append(character_id)
    _close_offer_if_empty_or_declined(ckpt, offer)
    prune_inventory_offers(ckpt)
    return {"offer_id": offer.offer_id, "offer_closed": offer.status == "closed"}


def remove_character_from_loot_offers(
    ckpt: CheckpointFile,
    character_id: str,
) -> int:
    """Remove a departed player character from pending offer eligibility."""

    changed = 0
    for offer in ckpt.session.dnd_inventory_offers:
        before = list(offer.eligible_character_ids)
        offer.eligible_character_ids = [
            cid for cid in offer.eligible_character_ids if cid != character_id
        ]
        if before != offer.eligible_character_ids:
            changed += 1
        _close_offer_if_empty_or_declined(ckpt, offer)
    changed += prune_inventory_offers(ckpt)
    return changed


def prune_inventory_offers(
    ckpt: CheckpointFile,
    *,
    max_closed: int = _CLOSED_OFFER_RETAIN,
) -> int:
    """Close orphaned offers and retain only a bounded closed-offer tail."""

    offers = list(getattr(ckpt.session, "dnd_inventory_offers", []) or [])
    changed = 0
    for offer in offers:
        before = (
            offer.status,
            tuple(offer.eligible_character_ids),
            tuple(offer.declined_by_character_ids),
        )
        _close_offer_if_empty_or_declined(ckpt, offer)
        after = (
            offer.status,
            tuple(offer.eligible_character_ids),
            tuple(offer.declined_by_character_ids),
        )
        if after != before:
            changed += 1

    closed = [offer for offer in offers if offer.status == "closed"]
    if max_closed < 0 or len(closed) <= max_closed:
        return changed

    keep_closed = {id(offer) for offer in closed[-max_closed:]}
    kept = [
        offer
        for offer in offers
        if offer.status != "closed" or id(offer) in keep_closed
    ]
    removed = len(offers) - len(kept)
    ckpt.session.dnd_inventory_offers = kept
    return changed + removed


def available_item_ids(offer: DndLootOffer) -> list[str]:
    return offer.available_item_ids()


def available_currency_dict(offer: DndLootOffer) -> dict[str, int]:
    if not offer.has_available_currency():
        return {key: 0 for key in _COIN_KEYS}
    return {key: int(getattr(offer.currency, key)) for key in _COIN_KEYS}


def _runtime_inventory(character: CharacterRecord) -> dict[str, Any] | None:
    mechanics = character.mechanics or {}
    runtime = mechanics.get("dnd5e_runtime")
    if not isinstance(runtime, dict):
        return None
    inventory = runtime.get("inventory")
    if not isinstance(inventory, dict):
        return None
    return inventory


def _ensure_runtime_inventory(character: CharacterRecord) -> dict[str, Any]:
    mechanics = character.mechanics
    if not isinstance(mechanics, dict):
        character.mechanics = {}
        mechanics = character.mechanics
    runtime = mechanics.get("dnd5e_runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        mechanics["dnd5e_runtime"] = runtime
    inventory = runtime.get("inventory")
    if not isinstance(inventory, dict):
        inventory = _imported_inventory(character)
        runtime["inventory"] = inventory
    inventory.setdefault("items", [])
    inventory["currency"] = _currency_dict(inventory.get("currency") or {})
    return inventory


def _imported_inventory(character: CharacterRecord) -> dict[str, Any]:
    mechanics = character.mechanics or {}
    sheet = mechanics.get("dnd5e_sheet") or {}
    statblock = sheet.get("statblock") or {}
    inventory = statblock.get("inventory") if isinstance(statblock, dict) else {}
    if not isinstance(inventory, dict):
        inventory = {}
    return {
        "items": deepcopy(inventory.get("items") or []),
        "currency": _currency_dict(inventory.get("currency") or {}),
    }


def _currency_dict(raw: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in _COIN_KEYS:
        try:
            out[key] = max(0, int(raw.get(key, 0) or 0))
        except (TypeError, ValueError):
            out[key] = 0
    return out


def _eligible_ids_for_signal(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    signal: DndLootOfferSignal,
) -> list[str]:
    active_ids = {char.character_id for char in ckpt.characters}
    bound = ckpt.session.character_bindings or {}
    eligible = [
        cid
        for cid in signal.eligible_character_ids
        if cid in active_ids and cid in bound
    ]
    if eligible:
        return list(dict.fromkeys(eligible))
    observer_ids = [obs.character_id for obs in event.observers]
    return [
        cid
        for cid in dict.fromkeys(observer_ids)
        if cid in bound and cid in active_ids
    ]


def _narrative_loot_offer_signal(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
) -> DndLootOfferSignal | None:
    if not _dnd_inventory_enabled(ckpt):
        return None
    text = _event_visible_text(event)
    if not text or not _looks_like_narrative_handoff(text):
        return None
    items = _narrative_handoff_items(text)
    if not items:
        return None
    return DndLootOfferSignal(
        present=True,
        source_kind="handoff",
        source_label=_narrative_handoff_source_label(text),
        visibility="table",
        eligible_character_ids=_bound_player_character_ids(ckpt),
        items=items,
        currency={key: 0 for key in _COIN_KEYS},
        notes=_NARRATIVE_LOOT_NOTES,
    )


def _dnd_inventory_enabled(ckpt: CheckpointFile) -> bool:
    settings = getattr(getattr(ckpt.session, "config", None), "settings", None)
    return getattr(settings, "ruleset_id", "") == _DND5E_BASIC_RULESET_ID


def _event_visible_text(event: EventRouterOutput) -> str:
    canonical = getattr(event, "canonical_event", None)
    facts = getattr(canonical, "observable_facts", []) if canonical is not None else []
    parts: list[str] = []
    for fact in facts:
        if isinstance(fact, dict):
            text = str(fact.get("text") or "")
        else:
            text = str(getattr(fact, "text", "") or "")
        if text:
            parts.append(text)
    return "\n".join(parts)


def _bound_player_character_ids(ckpt: CheckpointFile) -> list[str]:
    active_ids = {char.character_id for char in ckpt.characters}
    return [
        cid
        for cid in dict.fromkeys(ckpt.session.character_bindings or {})
        if cid in active_ids
    ]


def _looks_like_narrative_handoff(text: str) -> bool:
    return any(pattern.search(text) for pattern in _NARRATIVE_HANDOFF_PATTERNS)


def _narrative_handoff_items(text: str) -> list[DndLootOfferItem]:
    items: list[DndLootOfferItem] = []
    for pattern, template in _NARRATIVE_GEAR_PATTERNS:
        if not pattern.search(text):
            continue
        quantity = _narrative_item_quantity(text, template["item_id"])
        items.append(DndLootOfferItem(
            item_id=str(template["item_id"]),
            name=str(template["name"]),
            kind=str(template.get("kind") or "gear"),
            quantity=quantity,
            identified=True,
            requires_identification=False,
            requires_attunement=False,
            consumable=bool(template.get("consumable", False)),
            value_gp=0,
            weight=float(template.get("weight", 0)),
            notes=_NARRATIVE_LOOT_NOTES,
        ))
    return items


def _narrative_item_quantity(text: str, item_id: object) -> int:
    lowered = text.lower()
    item = str(item_id)
    if item in {"bedroll", "wrapped_meal"} and re.search(
        r"\b(?:two|2)\b.{0,40}\b(?:bedrolls?|parcels?|wraps?|meals?)\b",
        lowered,
    ):
        return 2
    if item == "waterskin" and re.search(r"\b(?:two|2)\b.{0,40}\bwater\s*skins?\b", lowered):
        return 2
    return 1


def _narrative_handoff_source_label(text: str) -> str:
    lowered = text.lower()
    if "hall-stock" in lowered or "hall stock" in lowered:
        return "Hall stock"
    if "stable-hand" in lowered or "stables" in lowered:
        return "stable hand"
    if "tavern counter" in lowered or "wrapped" in lowered:
        return "tavern counter"
    return "narrative handoff"


def _normalized_offer_items(
    items: list[DndLootOfferItem],
) -> list[DndLootOfferItem]:
    out: list[DndLootOfferItem] = []
    seen: set[str] = set()
    for idx, item in enumerate(items, start=1):
        item_id = _slug(item.item_id) or _slug(item.name) or f"item_{idx}"
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


def _offer_is_available_to(
    ckpt: CheckpointFile,
    offer: DndLootOffer,
    character_id: str,
) -> bool:
    return (
        offer.status == "open"
        and _character_is_eligible(ckpt, offer, character_id)
        and character_id not in set(offer.declined_by_character_ids)
        and (offer.available_item_ids() or offer.has_available_currency())
    )


def _character_is_eligible(
    ckpt: CheckpointFile,
    offer: DndLootOffer,
    character_id: str,
) -> bool:
    if not character_id:
        return False
    if character_id not in (ckpt.session.character_bindings or {}):
        return False
    eligible = _effective_eligible_ids(ckpt, offer)
    if not eligible:
        return False
    return character_id in eligible


def _find_open_offer(ckpt: CheckpointFile, offer_id: str) -> DndLootOffer:
    wanted = offer_id.strip()
    for offer in ckpt.session.dnd_inventory_offers:
        if offer.offer_id == wanted:
            if offer.status != "open":
                raise ValueError("That loot offer is already closed.")
            return offer
    raise ValueError(f"No open loot offer '{offer_id}'.")


def _character_by_id(ckpt: CheckpointFile, character_id: str) -> CharacterRecord:
    character = next(
        (char for char in ckpt.characters if char.character_id == character_id),
        None,
    )
    if character is None:
        raise ValueError(f"No character '{character_id}' in this session.")
    return character


def _inventory_item_from_offer(
    item: DndLootOfferItem,
    offer: DndLootOffer,
) -> dict[str, Any]:
    item_id = f"{offer.offer_id}_{item.item_id}"
    return {
        "id": item_id,
        "item_id": item_id,
        "source_item_id": item.item_id,
        "source_offer_id": offer.offer_id,
        "name": item.name,
        "kind": item.kind,
        "quantity": item.quantity,
        "equipped": False,
        "attuned": False,
        "identified": item.identified,
        "requires_identification": item.requires_identification,
        "requires_attunement": item.requires_attunement,
        "consumable": item.consumable,
        "weight": item.weight,
        "value_gp": item.value_gp,
        "notes": item.notes,
    }


def _unique_inventory_item_id(
    inventory: dict[str, Any],
    base_item_id: str,
) -> str:
    existing = {
        str(item.get("id") or item.get("item_id") or "")
        for item in inventory.get("items") or []
        if isinstance(item, dict)
    }
    item_id = base_item_id
    suffix = 2
    while item_id in existing:
        item_id = f"{base_item_id}_{suffix}"
        suffix += 1
    return item_id


def _currency_split_recipients(
    ckpt: CheckpointFile,
    offer: DndLootOffer,
    actor_id: str,
) -> list[str]:
    eligible = _effective_eligible_ids(ckpt, offer)
    if actor_id not in eligible and _character_is_eligible(ckpt, offer, actor_id):
        eligible.insert(0, actor_id)
    return list(dict.fromkeys(eligible))


def _close_offer_if_empty_or_declined(
    ckpt: CheckpointFile,
    offer: DndLootOffer,
) -> None:
    if not offer.available_item_ids() and not offer.has_available_currency():
        offer.status = "closed"
        return
    eligible = _effective_eligible_ids(ckpt, offer)
    offer.eligible_character_ids = eligible
    offer.declined_by_character_ids = [
        cid for cid in offer.declined_by_character_ids if cid in set(eligible)
    ]
    if not eligible:
        offer.status = "closed"
        return
    if eligible and set(eligible).issubset(set(offer.declined_by_character_ids)):
        offer.status = "closed"


def _effective_eligible_ids(
    ckpt: CheckpointFile,
    offer: DndLootOffer,
) -> list[str]:
    bindings = ckpt.session.character_bindings or {}
    active_ids = {char.character_id for char in ckpt.characters}
    seed = offer.eligible_character_ids
    return [
        cid
        for cid in dict.fromkeys(seed)
        if cid in bindings and cid in active_ids
    ]


def _offer_id(event_id: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", event_id.strip()).strip("_")
    return f"loot_{clean or 'offer'}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug[:48]
