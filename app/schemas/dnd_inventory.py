from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DndLootSourceKind = Literal[
    "body",
    "container",
    "reward",
    "handoff",
    "vendor",
    "other",
]
DndLootVisibility = Literal["table", "private"]
DndLootOfferStatus = Literal["open", "closed"]
_VALID_SOURCE_KINDS = {"body", "container", "reward", "handoff", "vendor", "other"}
_VALID_VISIBILITIES = {"table", "private"}


class DndCurrency(BaseModel):
    """D&D coin pouch.

    Defaults are fine for checkpoint/runtime state; the LLM-facing loot signal
    still requires the object itself to be present on `DndCanonicalEventRecord`.
    """

    model_config = ConfigDict(extra="forbid")

    cp: int
    sp: int
    ep: int
    gp: int
    pp: int

    @model_validator(mode="before")
    @classmethod
    def _fill_missing(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {key: data.get(key, 0) for key in ("cp", "sp", "ep", "gp", "pp")}

    @model_validator(mode="after")
    def _clean(self) -> "DndCurrency":
        for key in ("cp", "sp", "ep", "gp", "pp"):
            value = getattr(self, key)
            if value < 0:
                setattr(self, key, 0)
        return self

    def is_empty(self) -> bool:
        return not any(getattr(self, key) for key in ("cp", "sp", "ep", "gp", "pp"))


class DndLootOfferItem(BaseModel):
    """One router-visible item candidate in a loot/reward offer."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    name: str
    kind: str
    quantity: int
    identified: bool
    requires_identification: bool
    requires_attunement: bool
    consumable: bool
    value_gp: float
    weight: float
    notes: str

    @model_validator(mode="after")
    def _clean(self) -> "DndLootOfferItem":
        self.item_id = self.item_id.strip()
        self.name = self.name.strip() or "Item"
        self.kind = self.kind.strip().lower() or "gear"
        if self.quantity < 1:
            self.quantity = 1
        if self.value_gp < 0:
            self.value_gp = 0
        if self.weight < 0:
            self.weight = 0
        self.notes = self.notes.strip()
        return self


class DndLootOfferSignal(BaseModel):
    """LLM-facing loot signal on D&D fresh-router outputs.

    All fields are required in the JSON schema. When no offer exists, the
    router emits `present=false` and empty lists/strings/zero currency.
    """

    model_config = ConfigDict(extra="forbid")

    present: bool
    source_kind: DndLootSourceKind
    source_label: str
    visibility: DndLootVisibility
    eligible_character_ids: list[str]
    items: list[DndLootOfferItem]
    currency: DndCurrency
    notes: str

    @model_validator(mode="before")
    @classmethod
    def _clamp_literals(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("source_kind") not in _VALID_SOURCE_KINDS:
            data["source_kind"] = "other"
        if data.get("visibility") not in _VALID_VISIBILITIES:
            data["visibility"] = "table"
        return data

    @model_validator(mode="after")
    def _clean(self) -> "DndLootOfferSignal":
        self.source_label = self.source_label.strip()
        self.eligible_character_ids = [
            cid.strip()
            for cid in dict.fromkeys(self.eligible_character_ids)
            if cid.strip()
        ]
        self.notes = self.notes.strip()
        return self


class DndLootOffer(BaseModel):
    """Checkpoint-persistent pending loot/reward offer.

    Offers are prompts, not mutations. Claim commands move selected items and
    currency into a character's runtime inventory overlay.
    """

    model_config = ConfigDict(extra="forbid")

    offer_id: str
    source_event_id: str
    source_kind: DndLootSourceKind = "other"
    source_label: str = ""
    source_pack_id: str = ""
    source_ref: str = ""
    source_content_hash: str = ""
    source_depletion_ref: str = ""
    visibility: DndLootVisibility = "table"
    eligible_character_ids: list[str] = Field(default_factory=list)
    items: list[DndLootOfferItem] = Field(default_factory=list)
    currency: DndCurrency = Field(default_factory=lambda: DndCurrency(
        cp=0,
        sp=0,
        ep=0,
        gp=0,
        pp=0,
    ))
    notes: str = ""
    claimed_item_ids: list[str] = Field(default_factory=list)
    currency_claimed: bool = False
    declined_by_character_ids: list[str] = Field(default_factory=list)
    status: DndLootOfferStatus = "open"
    created_turn_index: int = 0

    @model_validator(mode="before")
    @classmethod
    def _clamp_literals(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("source_kind") not in _VALID_SOURCE_KINDS:
            data["source_kind"] = "other"
        if data.get("visibility") not in _VALID_VISIBILITIES:
            data["visibility"] = "table"
        if data.get("status") not in {"open", "closed", None}:
            data["status"] = "open"
        return data

    @model_validator(mode="after")
    def _clean(self) -> "DndLootOffer":
        self.offer_id = self.offer_id.strip()
        self.source_event_id = self.source_event_id.strip()
        self.source_label = self.source_label.strip()
        self.source_pack_id = self.source_pack_id.strip()
        self.source_ref = self.source_ref.strip()
        self.source_content_hash = self.source_content_hash.strip()
        self.source_depletion_ref = self.source_depletion_ref.strip()
        self.eligible_character_ids = [
            cid.strip()
            for cid in dict.fromkeys(self.eligible_character_ids)
            if cid.strip()
        ]
        self.claimed_item_ids = [
            iid.strip()
            for iid in dict.fromkeys(self.claimed_item_ids)
            if iid.strip()
        ]
        self.declined_by_character_ids = [
            cid.strip()
            for cid in dict.fromkeys(self.declined_by_character_ids)
            if cid.strip()
        ]
        self.notes = self.notes.strip()
        return self

    def available_item_ids(self) -> list[str]:
        claimed = set(self.claimed_item_ids)
        return [item.item_id for item in self.items if item.item_id not in claimed]

    def has_available_currency(self) -> bool:
        return not self.currency_claimed and not self.currency.is_empty()


def empty_loot_offer_signal() -> dict[str, Any]:
    return {
        "present": False,
        "source_kind": "other",
        "source_label": "",
        "visibility": "table",
        "eligible_character_ids": [],
        "items": [],
        "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 0, "pp": 0},
        "notes": "",
    }
