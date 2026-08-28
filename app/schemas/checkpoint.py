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
from app.schemas.onboarding import VisualNovelOnboarding
from app.schemas.state import SessionState, WorldState
from app.schemas.narrator import (
    visual_novel_pages_contain_source_identifiers,
    visual_novel_text_contains_source_identifiers,
)
from app.schemas.visual_references import (
    ReviewedVisualNovelSpriteSet,
    ReviewedVisualReference,
)


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
    # Optional deterministic, player-facing VN tutorial. Its opaque asset and
    # roster handles are private runtime metadata and are never model input.
    visual_novel_onboarding: VisualNovelOnboarding | None = None
    world_state: WorldState = Field(default_factory=WorldState)
    characters: list[CharacterRecord] = Field(default_factory=list)
    # Engine-only catalog of manually reviewed source images. The files remain
    # outside checkpoint JSON; only hash-pinned metadata and authored bindings
    # are durable. Default serialization redacts both fields so generic prompt
    # snapshots cannot accidentally carry paths, hashes, or reference handles.
    reviewed_visual_references: list[ReviewedVisualReference] = Field(
        default_factory=list
    )
    reviewed_visual_novel_sprite_sets: list[ReviewedVisualNovelSpriteSet] = Field(
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
            fixed_reference_ids = [
                reference_id
                for reference_id in reference_ids
                if references[reference_id].fixed_stage
            ]
            if len(fixed_reference_ids) > 1:
                raise ValueError(
                    f"location {label!r} selects more than one fixed stage"
                )
            if reference_ids:
                if label in normalized_locations:
                    raise ValueError(
                        f"duplicate normalized location visual label {label!r}"
                    )
                normalized_locations[label] = reference_ids
        self.location_visual_reference_ids = normalized_locations

        for reference in references.values():
            if not reference.fixed_stage:
                continue
            selected_ids = normalized_locations.get(reference.scope_id, [])
            if reference.reference_id not in selected_ids:
                raise ValueError(
                    f"fixed stage {reference.reference_id!r} is not selected "
                    f"by location {reference.scope_id!r}"
                )

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

        sprite_sets: dict[str, ReviewedVisualNovelSpriteSet] = {}
        sprite_reference_owners: dict[str, str] = {}
        for sprite_set in self.reviewed_visual_novel_sprite_sets:
            if sprite_set.sprite_set_id in sprite_sets:
                raise ValueError("reviewed sprite set ids must be unique")
            sprite_sets[sprite_set.sprite_set_id] = sprite_set
            owner_id = sprite_set.owner_character_id
            if owner_id and owner_id not in character_ids:
                raise ValueError(
                    f"reviewed sprite set {sprite_set.sprite_set_id!r} "
                    f"targets unknown character {owner_id!r}"
                )
            expected_scope = "character" if owner_id else "presentation"
            expected_scope_id = owner_id or sprite_set.sprite_set_id
            for reference_id in sprite_set.variant_reference_ids.values():
                reference = references.get(reference_id)
                if reference is None:
                    raise ValueError(
                        f"reviewed sprite set {sprite_set.sprite_set_id!r} "
                        f"selects unknown reference {reference_id!r}"
                    )
                if (
                    reference.purpose != "sprite"
                    or reference.scope != expected_scope
                    or reference.scope_id != expected_scope_id
                ):
                    raise ValueError(
                        f"reviewed sprite set {sprite_set.sprite_set_id!r} "
                        f"selects invalid sprite reference {reference_id!r}"
                    )
                if reference.diffusion_authorized:
                    raise ValueError(
                        "reviewed sprite references cannot be offered to "
                        "runtime diffusion direction"
                    )
                prior_set = sprite_reference_owners.get(reference_id)
                if prior_set is not None:
                    raise ValueError(
                        f"reviewed sprite reference {reference_id!r} is used "
                        f"by both {prior_set!r} and {sprite_set.sprite_set_id!r}"
                    )
                sprite_reference_owners[reference_id] = sprite_set.sprite_set_id

        for character in self.characters:
            sprite_set_id = character.visuals.sprite_set_id.strip()
            character.visuals.sprite_set_id = sprite_set_id
            if not sprite_set_id:
                continue
            sprite_set = sprite_sets.get(sprite_set_id)
            if sprite_set is None:
                raise ValueError(
                    f"character {character.character_id!r} selects unknown "
                    f"sprite set {sprite_set_id!r}"
                )
            if sprite_set.owner_character_id != character.character_id:
                raise ValueError(
                    f"character {character.character_id!r} selects sprite set "
                    f"owned by {sprite_set.owner_character_id or 'presentation'}"
                )

        onboarding = self.visual_novel_onboarding
        if onboarding is not None:
            stage = references.get(onboarding.stage_reference_id)
            if stage is None:
                raise ValueError("visual-novel onboarding selects an unknown stage")
            if not (
                stage.scope == "location"
                and stage.purpose == "environment"
                and stage.diffusion_authorized
            ):
                raise ValueError(
                    "visual-novel onboarding requires an authorized location stage"
                )

            active_labels: dict[str, list[CharacterRecord]] = {}
            for character in self.characters:
                if character.status.value == "culled":
                    continue
                label = " ".join((character.name or "").split()).strip()
                if label:
                    active_labels.setdefault(label, []).append(character)
            pages = [authored.page for authored in onboarding.pages]
            if visual_novel_pages_contain_source_identifiers(
                pages,
                source_ids=tuple(character_ids),
            ):
                raise ValueError(
                    "visual-novel onboarding pages expose a source identifier"
                )
            for authored in onboarding.pages:
                for label in authored.page.sprites:
                    matches = active_labels.get(label, [])
                    if len(matches) != 1:
                        raise ValueError(
                            "visual-novel onboarding sprite labels must resolve "
                            "to exactly one active character"
                        )
                    character = matches[0]
                    sprite_set = sprite_sets.get(character.visuals.sprite_set_id)
                    if sprite_set is None:
                        raise ValueError(
                            "visual-novel onboarding characters require a "
                            "reviewed sprite set"
                        )
                    variant_key = authored.sprite_variant_keys_by_label.get(
                        label,
                        "neutral",
                    )
                    if variant_key not in sprite_set.variant_reference_ids:
                        raise ValueError(
                            "visual-novel onboarding selects an unavailable "
                            "sprite variant"
                        )

            characters_by_id = {
                character.character_id: character
                for character in self.characters
            }
            for choice in onboarding.join_choices:
                target = characters_by_id.get(choice.character_id)
                if (
                    target is None
                    or not target.is_playable
                    or target.status.value == "culled"
                ):
                    raise ValueError(
                        "visual-novel onboarding choices require playable seats"
                    )
                if visual_novel_text_contains_source_identifiers(
                    choice.label,
                    source_ids=tuple(character_ids),
                ):
                    raise ValueError(
                        "visual-novel onboarding choice labels expose a source id"
                    )
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

    @field_serializer("reviewed_visual_novel_sprite_sets")
    def _serialize_reviewed_visual_novel_sprite_sets(
        self,
        value: list[ReviewedVisualNovelSpriteSet],
        info: SerializationInfo,
    ) -> list[ReviewedVisualNovelSpriteSet]:
        if should_include_private_runtime_metadata(info.context):
            return value
        return []

    @field_serializer("visual_novel_onboarding")
    def _serialize_visual_novel_onboarding(
        self,
        value: VisualNovelOnboarding | None,
        info: SerializationInfo,
    ) -> VisualNovelOnboarding | None:
        if should_include_private_runtime_metadata(info.context):
            return value
        return None
