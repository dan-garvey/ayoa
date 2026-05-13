from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Iterable

from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_inventory import (
    DndCurrency,
    DndLootOffer,
    DndLootOfferItem,
    DndLootOfferSignal,
)
from app.schemas.event_router import EventRouterOutput


_COIN_KEYS = ("cp", "sp", "ep", "gp", "pp")


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

    prompts: dict[str, list[str]] = {}
    existing_event_ids = {
        offer.source_event_id
        for offer in getattr(ckpt.session, "dnd_inventory_offers", [])
    }
    bindings = ckpt.session.character_bindings or {}
    for event in events:
        signal = getattr(event, "loot_offer", None)
        if signal is None:
            continue
        if isinstance(signal, dict):
            signal = DndLootOfferSignal(**signal)
        if not isinstance(signal, DndLootOfferSignal) or not signal.present:
            continue
        if event.event_id in existing_event_ids:
            continue
        if not signal.items and signal.currency.is_empty():
            continue

        eligible = _eligible_ids_for_signal(ckpt, event, signal)
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
) -> dict[str, Any]:
    offer = _find_open_offer(ckpt, offer_id)
    if not _character_is_eligible(ckpt, offer, character_id):
        raise ValueError("That loot offer is not available to this character.")

    selected = [iid.strip() for iid in item_ids if iid.strip()]
    selected_set = set(selected)
    available_items = {
        item.item_id: item
        for item in offer.items
        if item.item_id not in set(offer.claimed_item_ids)
    }
    if selected_set - set(available_items):
        missing = ", ".join(sorted(selected_set - set(available_items)))
        raise ValueError(f"Item(s) no longer available in that offer: {missing}")
    if take_currency and not offer.has_available_currency():
        raise ValueError("Currency is no longer available in that offer.")
    if not selected and not take_currency:
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
    offer = _find_open_offer(ckpt, offer_id)
    if not _character_is_eligible(ckpt, offer, character_id):
        raise ValueError("That loot offer is not available to this character.")
    if character_id not in offer.declined_by_character_ids:
        offer.declined_by_character_ids.append(character_id)
    _close_offer_if_empty_or_declined(ckpt, offer)
    return {"offer_id": offer.offer_id, "offer_closed": offer.status == "closed"}


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
    eligible = [cid for cid in signal.eligible_character_ids if cid in active_ids]
    if eligible:
        return list(dict.fromkeys(eligible))
    observer_ids = [obs.character_id for obs in event.observers]
    bound = ckpt.session.character_bindings or {}
    return [
        cid
        for cid in dict.fromkeys(observer_ids)
        if cid in bound and cid in active_ids
    ]


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
    eligible = offer.eligible_character_ids
    if not eligible:
        return character_id in (ckpt.session.character_bindings or {})
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
    bindings = ckpt.session.character_bindings or {}
    if offer.eligible_character_ids:
        eligible = [cid for cid in offer.eligible_character_ids if cid in bindings]
    else:
        eligible = list(bindings)
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
    eligible = offer.eligible_character_ids or list(ckpt.session.character_bindings)
    if eligible and set(eligible).issubset(set(offer.declined_by_character_ids)):
        offer.status = "closed"


def _offer_id(event_id: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", event_id.strip()).strip("_")
    return f"loot_{clean or 'offer'}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug[:48]
