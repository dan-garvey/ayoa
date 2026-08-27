"""Resolve private sprite-set handles into deterministic VN placements.

Narrator pages carry only player-safe character labels and bounded expression
cues. This boundary maps those labels back to checkpoint identities, applies
viewpoint-specific adapter policy, and resolves immutable PNG bytes without
putting image metadata into any model context.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.engine.one_star_visuals import sprite_set_id_for_viewer
from app.engine.reviewed_visual_references import (
    ReviewedVisualReferenceError,
    resolve_frozen_visual_reference_media,
)
from app.engine.visual_novel_presentation import VisualNovelSpritePlacement
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.narrator import VisualNovelPage
from app.schemas.visual_references import VisualNovelSpriteExpression

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


def resolve_visual_novel_sprite_placements(
    *,
    checkpoint: CheckpointFile,
    viewer_character_id: str,
    page: VisualNovelPage,
    generation: "ImageGenerationCoordinator",
) -> tuple[VisualNovelSpritePlacement, ...]:
    """Resolve zero, one, or two safe page cues into immutable placements."""

    characters_by_label = _unique_character_labels(checkpoint)
    resolved: list[_ResolvedCue] = []
    for cue in page.sprites:
        character = characters_by_label.get(cue.character)
        if character is None or character.character_id == viewer_character_id:
            continue
        sprite_set_id = sprite_set_id_for_viewer(
            checkpoint,
            viewer_character_id=viewer_character_id,
            character=character,
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
            expression=cue.expression,
        )
        if candidate is not None:
            resolved.append(candidate)

    count = len(resolved)
    if count == 0:
        return ()
    if count == 1:
        slots = (("center", "right", (512, 565), 98),)
    else:
        slots = (
            ("left", "right", (292, 565), 92),
            ("right", "left", (732, 565), 92),
        )
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
    expression: VisualNovelSpriteExpression,
) -> _ResolvedCue | None:
    authored = next(
        (
            sprite_set
            for sprite_set in checkpoint.reviewed_visual_novel_sprite_sets
            if sprite_set.sprite_set_id == sprite_set_id
        ),
        None,
    )
    if authored is not None:
        requested_reference_id = authored.variant_reference_ids.get(expression)
        neutral_reference_id = authored.variant_reference_ids["neutral"]
        candidate_reference_ids = tuple(
            dict.fromkeys(
                reference_id
                for reference_id in (
                    requested_reference_id,
                    neutral_reference_id,
                )
                if reference_id
            )
        )
        reference_id = ""
        frozen = None
        for candidate_reference_id in candidate_reference_ids:
            candidate = generation.store.reviewed_reference(
                session_id=checkpoint.session.session_id,
                reference_id=candidate_reference_id,
            )
            if candidate is not None:
                reference_id = candidate_reference_id
                frozen = candidate
                break
        if frozen is None:
            logger.warning(
                "reviewed VN sprite is unavailable session=%s set=%s",
                checkpoint.session.session_id,
                sprite_set_id,
            )
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

    resolved = generation.resolve_visual_novel_sprite_variant(
        session_id=checkpoint.session.session_id,
        character_id=character.character_id,
        sprite_pack_id=sprite_set_id,
        expression=expression,
    )
    if resolved is None and expression != "neutral":
        resolved = generation.resolve_visual_novel_sprite_variant(
            session_id=checkpoint.session.session_id,
            character_id=character.character_id,
            sprite_pack_id=sprite_set_id,
            expression="neutral",
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
