"""Resolve private sprite-set handles into deterministic VN placements.

Narrator pages carry only player-safe character labels. This boundary maps
those labels back to checkpoint identities, applies the character-authored
event-relative variant snapshot,
viewpoint-specific adapter policy, and resolves immutable PNG bytes without
putting image metadata into any model context.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from app.engine.one_star_visuals import sprite_set_id_for_viewer
from app.engine.reviewed_visual_references import (
    ReviewedVisualReferenceError,
    resolve_frozen_visual_reference_media,
)
from app.engine.visual_novel_presentation import VisualNovelSpritePlacement
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.narrator import (
    VisualNovelPage,
    visual_novel_text_contains_source_identifiers,
)
if TYPE_CHECKING:
    from app.engine.image_generation import ImageGenerationCoordinator
    from app.engine.player_media import ResolvedPlayerMedia


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ResolvedCue:
    subject_handle: str
    sprite_set_id: str
    variant_handle: str
    media: "ResolvedPlayerMedia"
    source_facing: str


@dataclass(frozen=True)
class VisualNovelSpriteIdentityTransition:
    """One viewer-visible sprite-set change across committed checkpoints."""

    character_id: str
    character_name: str
    before_sprite_set_id: str
    after_sprite_set_id: str


def resolve_visual_novel_sprite_placements(
    *,
    checkpoint: CheckpointFile,
    viewer_character_id: str,
    page: VisualNovelPage,
    generation: "ImageGenerationCoordinator",
    sprite_set_id_overrides: Mapping[str, str] | None = None,
    variant_keys_by_label: Mapping[str, str] | None = None,
) -> tuple[VisualNovelSpritePlacement, ...]:
    """Resolve zero, one, or two safe page cues into immutable placements."""

    characters_by_label = _unique_character_labels(checkpoint)
    overrides = sprite_set_id_overrides or {}
    variant_keys = variant_keys_by_label or {}
    resolved: list[_ResolvedCue] = []
    for label in page.sprites:
        character = characters_by_label.get(label)
        if character is None or character.character_id == viewer_character_id:
            continue
        sprite_set_id = overrides.get(character.character_id) or (
            sprite_set_id_for_viewer(
                checkpoint,
                viewer_character_id=viewer_character_id,
                character=character,
            )
        )
        if not sprite_set_id:
            continue
        candidate = _resolve_cue(
            checkpoint=checkpoint,
            generation=generation,
            character=character,
            subject_handle=_subject_handle(
                checkpoint=checkpoint,
                viewer_character_id=viewer_character_id,
                character=character,
            ),
            sprite_set_id=sprite_set_id,
            variant_key=variant_keys.get(label, "neutral"),
        )
        if candidate is not None:
            resolved.append(candidate)

    return _placements_for_resolved_cues(resolved)


def visual_novel_sprite_identity_transitions(
    *,
    before_checkpoint: CheckpointFile,
    after_checkpoint: CheckpointFile,
    viewer_character_id: str,
    pages: Sequence[VisualNovelPage],
) -> tuple[VisualNovelSpriteIdentityTransition, ...]:
    """Find depicted identities whose viewer-scoped sprite set just changed.

    The checkpoint boundary is authoritative. Page text is used only as a
    visibility guard so an unrelated off-camera state change cannot create a
    reveal card for this viewer.
    """

    if (
        before_checkpoint.session.session_id
        != after_checkpoint.session.session_id
    ):
        raise ValueError("VN identity transitions require one session")

    before_by_id = {
        character.character_id: character
        for character in before_checkpoint.characters
    }
    visible_cues = {
        " ".join(label.split()).casefold()
        for page in pages
        for label in page.sprites
        if " ".join(label.split()).strip()
    }
    visible_speakers = {
        " ".join(page.speaker.split()).casefold()
        for page in pages
        if " ".join(page.speaker.split()).strip()
    }
    visible_text = "\n".join(page.text for page in pages)
    after_labels = _unique_character_labels(after_checkpoint)

    transitions: list[VisualNovelSpriteIdentityTransition] = []
    for character_name, after_character in after_labels.items():
        if (
            after_character.character_id == viewer_character_id
            or after_character.status == CharacterStatus.culled
            or visual_novel_text_contains_source_identifiers(character_name)
        ):
            continue
        before_character = before_by_id.get(after_character.character_id)
        if before_character is None:
            continue
        normalized_name = character_name.casefold()
        if (
            normalized_name not in visible_cues
            and normalized_name not in visible_speakers
            and re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(character_name)}"
                rf"(?![A-Za-z0-9_])",
                visible_text,
                re.IGNORECASE,
            )
            is None
        ):
            continue
        before_sprite_set_id = sprite_set_id_for_viewer(
            before_checkpoint,
            viewer_character_id=viewer_character_id,
            character=before_character,
        )
        after_sprite_set_id = sprite_set_id_for_viewer(
            after_checkpoint,
            viewer_character_id=viewer_character_id,
            character=after_character,
        )
        if (
            not before_sprite_set_id
            or not after_sprite_set_id
            or before_sprite_set_id == after_sprite_set_id
        ):
            continue
        transitions.append(
            VisualNovelSpriteIdentityTransition(
                character_id=after_character.character_id,
                character_name=character_name,
                before_sprite_set_id=before_sprite_set_id,
                after_sprite_set_id=after_sprite_set_id,
            )
        )
    return tuple(transitions)


def resolve_visual_novel_identity_transition_placement(
    *,
    checkpoint: CheckpointFile,
    viewer_character_id: str,
    character_id: str,
    sprite_set_id: str,
    generation: "ImageGenerationCoordinator",
    variant_key: str = "neutral",
) -> VisualNovelSpritePlacement | None:
    """Resolve one centered character-owned sprite for a reveal card."""

    character = next(
        (
            item
            for item in checkpoint.characters
            if item.character_id == character_id
            and item.status != CharacterStatus.culled
        ),
        None,
    )
    if character is None or character.character_id == viewer_character_id:
        return None
    resolved = _resolve_cue(
        checkpoint=checkpoint,
        generation=generation,
        character=character,
        subject_handle=_subject_handle(
            checkpoint=checkpoint,
            viewer_character_id=viewer_character_id,
            character=character,
        ),
        sprite_set_id=sprite_set_id,
        variant_key=variant_key,
    )
    if resolved is None:
        return None
    return _placements_for_resolved_cues([resolved])[0]


def _placements_for_resolved_cues(
    resolved: Sequence[_ResolvedCue],
) -> tuple[VisualNovelSpritePlacement, ...]:
    count = len(resolved)
    if count == 0:
        return ()
    if count == 1:
        slots = (("center", "right", (512, 565), 98),)
    elif count == 2:
        slots = (
            ("left", "right", (292, 565), 92),
            ("right", "left", (732, 565), 92),
        )
    else:
        raise ValueError("visual-novel pages accept at most two sprite cues")
    return tuple(
        VisualNovelSpritePlacement(
            subject_handle=item.subject_handle,
            identity_handle=item.sprite_set_id,
            variant_handle=item.variant_handle,
            media=item.media,
            slot=slot,
            source_facing=item.source_facing,
            facing=facing,
            anchor=anchor,
            scale_percent=scale_percent,
        )
        for item, (slot, facing, anchor, scale_percent) in zip(
            resolved,
            slots,
            strict=True,
        )
    )


def _unique_character_labels(
    checkpoint: CheckpointFile,
) -> dict[str, CharacterRecord]:
    active = [
        character
        for character in checkpoint.characters
        if character.status != CharacterStatus.culled
        and " ".join((character.name or "").split()).strip()
    ]
    counts: dict[str, int] = {}
    for character in active:
        label = " ".join(character.name.split()).strip()
        counts[label] = counts.get(label, 0) + 1
    return {
        " ".join(character.name.split()).strip(): character
        for character in active
        if counts[" ".join(character.name.split()).strip()] == 1
    }


def _resolve_cue(
    *,
    checkpoint: CheckpointFile,
    generation: "ImageGenerationCoordinator",
    character: CharacterRecord,
    subject_handle: str,
    sprite_set_id: str,
    variant_key: str,
) -> _ResolvedCue | None:
    authored = next(
        (
            sprite_set
            for sprite_set in checkpoint.reviewed_visual_novel_sprite_sets
            if sprite_set.sprite_set_id == sprite_set_id
        ),
        None,
    )
    variant_key = variant_key.strip().lower() or "neutral"

    def _resolve_authored(key: str) -> _ResolvedCue | None:
        if authored is None:
            return None
        reference_id = authored.variant_reference_ids.get(key, "")
        if not reference_id:
            return None
        frozen = generation.store.reviewed_reference(
            session_id=checkpoint.session.session_id,
            reference_id=reference_id,
        )
        if frozen is None:
            return None
        try:
            media = resolve_frozen_visual_reference_media(
                frozen,
                runtime_root=generation.config.runtime_root,
            )
        except ReviewedVisualReferenceError as exc:
            logger.warning(
                "reviewed VN sprite validation failed session=%s set=%s code=%s",
                checkpoint.session.session_id,
                sprite_set_id,
                exc.code,
            )
            return None
        return _ResolvedCue(
            subject_handle=subject_handle,
            sprite_set_id=sprite_set_id,
            variant_handle=reference_id,
            media=media,
            source_facing=authored.source_facing,
        )

    def _resolve_generated(key: str) -> _ResolvedCue | None:
        resolved = generation.resolve_visual_novel_sprite_variant(
            session_id=checkpoint.session.session_id,
            character_id=character.character_id,
            sprite_pack_id=sprite_set_id,
            variant_key=key,
        )
        if resolved is None:
            return None
        variant_handle, media, source_facing = resolved
        return _ResolvedCue(
            subject_handle=subject_handle,
            sprite_set_id=sprite_set_id,
            variant_handle=variant_handle,
            media=media,
            source_facing=source_facing,
        )

    for key in dict.fromkeys((variant_key, "neutral")):
        candidate = _resolve_authored(key) or _resolve_generated(key)
        if candidate is not None:
            return candidate
    logger.warning(
        "VN sprite is unavailable session=%s set=%s variant=%s",
        checkpoint.session.session_id,
        sprite_set_id,
        variant_key,
    )
    return None


def _subject_handle(
    *,
    checkpoint: CheckpointFile,
    viewer_character_id: str,
    character: CharacterRecord,
) -> str:
    """Return a stable opaque subject key without persisting source ids."""

    digest = hashlib.sha256(
        "\0".join(
            (
                checkpoint.session.session_id,
                viewer_character_id,
                character.character_id,
            )
        ).encode("utf-8")
    ).hexdigest()
    return f"vnsubject.{digest[:24]}"
