"""Durable, player-safe narrator delivery records.

All successful narrator renders use this single per-POV outbox, whether they
were produced while handling a player request or by autonomous work. Frontends
claim with a lease and acknowledge by stable delivery id, giving retries
at-least-once semantics without duplicating canonical fiction.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.content_pack import SafeAssetRevealPayload
from app.schemas.narrator import (
    VisualNovelPage,
    visual_novel_pages_contain_source_identifiers,
)


class NarratorEventRef(BaseModel):
    """One canonical event projected into a particular narrator POV."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    observation_level: Literal["direct", "indirect", "inferred"]
    visible_at_s: int
    event_sequence: int
    sprite_variant_keys_by_character_id: dict[str, str]

    @model_validator(mode="after")
    def _validate_ref(self) -> "NarratorEventRef":
        if not self.event_id.strip():
            raise ValueError("narrator event id must not be blank")
        if self.visible_at_s < 0 or self.event_sequence < 0:
            raise ValueError("narrator event time and sequence cannot be negative")
        return self


class DeliveryVisualNovelSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pages: list[VisualNovelPage] = Field(min_length=1)
    rendered_event_ids: list[str] = Field(min_length=1)
    sprite_variant_keys_by_label: dict[str, str] = Field(default_factory=dict)

    @field_validator("rendered_event_ids")
    @classmethod
    def _validate_event_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("rendered event ids cannot contain blanks")
        if len(values) != len(set(values)):
            raise ValueError("rendered event ids cannot contain duplicates")
        return values

    @field_validator("sprite_variant_keys_by_label")
    @classmethod
    def _validate_variant_keys(cls, values: dict[str, str]) -> dict[str, str]:
        for label, key in values.items():
            if not label.strip() or not key.strip():
                raise ValueError("sprite variant snapshots cannot contain blanks")
            if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", key) is None:
                raise ValueError("sprite variant snapshot key is invalid")
        return values

    @model_validator(mode="after")
    def _validate_pages(self) -> "DeliveryVisualNovelSegment":
        if visual_novel_pages_contain_source_identifiers(self.pages):
            raise ValueError("visual-novel delivery cannot expose source ids")
        labels = {label for page in self.pages for label in page.sprites}
        if set(self.sprite_variant_keys_by_label) - labels:
            raise ValueError("sprite variants may name only depicted labels")
        return self


class DeliveryVisualNovelRender(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[DeliveryVisualNovelSegment] = Field(min_length=1)


class DeliveryPayload(BaseModel):
    """One POV's complete standard frontend payload."""

    model_config = ConfigDict(extra="forbid")

    prose: str
    visual_novel: DeliveryVisualNovelRender | None
    asset_reveals: list[SafeAssetRevealPayload]
    reaction_prompt_event_id: str
    loot_offer_ids: list[str]
    commitment_revision_ids: list[str]
    dice_rolls: list[dict[str, Any]]
    experience_awards: list[dict[str, Any]]
    owner_error: str


class DeliveryOutboxEntry(BaseModel):
    """Lease-claimed at-least-once delivery for one bound POV."""

    model_config = ConfigDict(extra="forbid")

    delivery_id: str
    pov_character_id: str
    source_event_ids: list[str]
    highest_event_sequence: int
    created_revision: int
    payload: DeliveryPayload
    status: Literal["pending", "claimed", "acknowledged"]
    claim_token: str
    claimed_by: str
    claimed_at: str
    attempts: int
    acknowledged_at: str

    @model_validator(mode="after")
    def _validate_entry(self) -> "DeliveryOutboxEntry":
        if not self.delivery_id.strip() or not self.pov_character_id.strip():
            raise ValueError("delivery and POV ids must not be blank")
        if any(not value.strip() for value in self.source_event_ids):
            raise ValueError("delivery source event ids cannot contain blanks")
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("delivery source event ids cannot be duplicated")
        if self.highest_event_sequence < -1 or self.created_revision < 0:
            raise ValueError("delivery sequence and revision are invalid")
        if self.attempts < 0:
            raise ValueError("delivery attempts cannot be negative")
        if self.status == "pending" and any((
            self.claim_token,
            self.claimed_by,
            self.claimed_at,
            self.acknowledged_at,
        )):
            raise ValueError("pending delivery cannot carry claim or ack state")
        if self.status == "claimed" and not all((
            self.claim_token,
            self.claimed_by,
            self.claimed_at,
        )):
            raise ValueError("claimed delivery requires a complete lease")
        if self.status == "acknowledged" and not self.acknowledged_at:
            raise ValueError("acknowledged delivery requires its timestamp")
        return self


class NarratorRenderJob(BaseModel):
    """Per-POV retryable render work keyed by canonical event lineage."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    lane_id: str
    pov_character_id: str
    source_event_ids: list[str]
    event_refs: list[NarratorEventRef]
    highest_event_sequence: int
    created_revision: int
    user_input: str
    partial_mode: bool
    narration_mode: Literal["event_aligned", "compressed_sequence"]
    dice_rolls: list[dict[str, Any]] = Field(default_factory=list)
    experience_awards: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["pending", "failed"]
    attempts: int
    last_error: str

    @model_validator(mode="after")
    def _validate_job(self) -> "NarratorRenderJob":
        if (
            not self.job_id.strip()
            or not self.lane_id.strip()
            or not self.pov_character_id.strip()
        ):
            raise ValueError("narrator job, lane, and POV ids must not be blank")
        if not self.source_event_ids or any(
            not value.strip() for value in self.source_event_ids
        ):
            raise ValueError("narrator job requires source event ids")
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("narrator job source ids cannot be duplicated")
        if [item.event_id for item in self.event_refs] != self.source_event_ids:
            raise ValueError("narrator job refs must align with source event ids")
        if self.highest_event_sequence < 0 or self.created_revision < 0:
            raise ValueError("narrator job sequence and revision are invalid")
        if self.attempts < 0:
            raise ValueError("narrator job attempts cannot be negative")
        return self
