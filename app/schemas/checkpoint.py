from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)

from app.schemas.characters import CharacterRecord
from app.schemas.conversation import ConversationMessage
from app.schemas.content_privacy import should_include_private_runtime_metadata
from app.schemas.event_router import DndEventRouterOutput, EventRouterOutput
from app.schemas.one_star import OneStarEventRouterOutput
from app.schemas.state import SessionState, WorldState
from app.schemas.visual_references import ReviewedVisualReference


CURRENT_SCHEMA_VERSION = "5.0"


class CheckpointFile(BaseModel):
    # Schema 5.0 removes the session-global transcript. Player-visible history
    # is reconstructed from the per-POV narrator conversations below. Older
    # checkpoints hard-break on load: checkpoint_manager raises with a
    # message pointing the user at /story start. No migration shim.
    schema_version: str = CURRENT_SCHEMA_VERSION
    session: SessionState
    # 1-2 paragraph player-facing world primer: the first thing a fresh
    # player sees after /story start. Distinct from the omniscient dossier
    # (which leaked spoilers). Synthetic story checkpoints should author
    # this directly. Empty checkpoints render a fallback stub.
    #
    # Note: there is no longer an authored `opening_narrative` field —
    # the opening beat is composed at runtime by the router (using
    # world_state, character_records, and the `(begin)` OOC directive)
    # and rendered per-POV by the narrator on the first turn. This
    # keeps every turn on a single code path and avoids the POV-binding
    # and race-window problems of an authored opener; see commit log
    # for the rationale.
    player_primer: str = ""
    world_state: WorldState = Field(default_factory=WorldState)
    characters: list[CharacterRecord] = Field(default_factory=list)
    # Engine-only catalog of manually reviewed source images. The files remain
    # outside checkpoint JSON; only hash-pinned metadata and authored bindings
    # are durable. Default serialization redacts both fields so generic prompt
    # snapshots cannot accidentally carry paths, hashes, or reference handles.
    reviewed_visual_references: list[ReviewedVisualReference] = Field(
        default_factory=list
    )
    location_visual_reference_ids: dict[str, list[str]] = Field(
        default_factory=dict
    )
    # Rolling conversation histories: each role sees the full prior exchange
    # on every call, so continuity and caching both work.
    session_conversation: list[ConversationMessage] = Field(default_factory=list)
    # v11: per-character narrator rolling history. Each human (by their
    # bound character_id) has their own stream; the narrator_phase2 call
    # reads character_id-keyed history. The old session-wide
    # narrator_conversation is gone in the v11 pipeline.
    narrator_conversations: dict[str, list[ConversationMessage]] = Field(default_factory=dict)
    character_conversations: dict[str, list[ConversationMessage]] = Field(default_factory=dict)
    # v11: the canonical event log. Every closed canonical event appended
    # here. Source of truth for rendering, replay, and debug.
    canonical_events: list[
        DndEventRouterOutput | OneStarEventRouterOutput | EventRouterOutput
    ] = Field(
        default_factory=list
    )
    visibility_log: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_reviewed_visual_bindings(self) -> "CheckpointFile":
        references: dict[str, ReviewedVisualReference] = {}
        for reference in self.reviewed_visual_references:
            if reference.reference_id in references:
                raise ValueError(
                    "reviewed visual reference ids must be unique"
                )
            references[reference.reference_id] = reference

        character_ids = {character.character_id for character in self.characters}
        for reference in references.values():
            if (
                reference.scope == "character"
                and reference.scope_id not in character_ids
            ):
                raise ValueError(
                    f"reviewed identity reference {reference.reference_id!r} "
                    f"targets unknown character {reference.scope_id!r}"
                )

        if len(self.location_visual_reference_ids) > 512:
            raise ValueError(
                "location visual reference mapping exceeds 512 labels"
            )
        normalized_locations: dict[str, list[str]] = {}
        for raw_label, raw_ids in self.location_visual_reference_ids.items():
            label = " ".join(str(raw_label or "").split())
            if (
                not label
                or len(label) > 200
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in label
                )
            ):
                raise ValueError(
                    "location visual reference labels must be safe, non-empty, "
                    "and at most 200 characters"
                )
            reference_ids = [
                reference_id
                for reference_id in dict.fromkeys(
                    str(value or "").strip() for value in raw_ids
                )
                if reference_id
            ]
            for reference_id in reference_ids:
                reference = references.get(reference_id)
                if reference is None:
                    raise ValueError(
                        f"location {label!r} selects unknown reviewed visual "
                        f"reference {reference_id!r}"
                    )
                if (
                    reference.scope != "location"
                    or reference.purpose not in {"environment", "style"}
                    or reference.scope_id != label
                ):
                    raise ValueError(
                        f"location {label!r} selects non-location reference "
                        f"{reference_id!r}"
                    )
                if not reference.diffusion_authorized:
                    raise ValueError(
                        f"location {label!r} selects reference "
                        f"{reference_id!r} without diffusion authorization"
                    )
            if reference_ids:
                if label in normalized_locations:
                    raise ValueError(
                        f"duplicate normalized location visual label {label!r}"
                    )
                normalized_locations[label] = reference_ids
        self.location_visual_reference_ids = normalized_locations

        identity_owners: dict[str, str] = {}
        for character in self.characters:
            reference_id = character.visuals.identity_reference_id.strip()
            character.visuals.identity_reference_id = reference_id
            reference = references.get(reference_id)
            if reference is None:
                if reference_id and not reference_id.startswith("imgref_"):
                    raise ValueError(
                        f"character {character.character_id!r} selects unknown "
                        f"authored identity reference {reference_id!r}"
                    )
                # Generated runtime candidates use the reserved imgref_ prefix.
                # Story loading separately rejects them in source seeds.
                continue
            if reference.scope != "character" or reference.purpose != "identity":
                raise ValueError(
                    f"character {character.character_id!r} selects "
                    f"non-identity reference {reference_id!r}"
                )
            if reference.scope_id != character.character_id:
                raise ValueError(
                    f"character {character.character_id!r} selects identity "
                    f"reference owned by {reference.scope_id!r}"
                )
            if not reference.diffusion_authorized:
                raise ValueError(
                    f"character {character.character_id!r} selects reference "
                    f"{reference_id!r} without diffusion authorization"
                )
            prior_owner = identity_owners.get(reference_id)
            if prior_owner is not None:
                raise ValueError(
                    f"reviewed identity reference {reference_id!r} is selected "
                    f"by both {prior_owner!r} and {character.character_id!r}"
                )
            identity_owners[reference_id] = character.character_id
        return self

    @field_serializer("reviewed_visual_references")
    def _serialize_reviewed_visual_references(
        self,
        value: list[ReviewedVisualReference],
        info: SerializationInfo,
    ) -> list[ReviewedVisualReference]:
        if should_include_private_runtime_metadata(info.context):
            return value
        return []

    @field_serializer("location_visual_reference_ids")
    def _serialize_location_visual_reference_ids(
        self,
        value: dict[str, list[str]],
        info: SerializationInfo,
    ) -> dict[str, list[str]]:
        if should_include_private_runtime_metadata(info.context):
            return value
        return {}
