"""Character-owned visual-novel pose-expression presentation state.

This module handles text-only catalog choices. Runtime models receive no image
bytes, paths, hashes, job identifiers, or compositor details.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping

from app.engine.one_star_visuals import generated_sprite_pack_id
from app.engine.text_safety import strip_terminal_control
from app.schemas.agents import CharacterPresentationChoice
from app.schemas.characters import (
    CharacterRecord,
    VisualNovelCustomVariantRequest,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import (
    REDACTED_IMPORT_SENTINEL,
    redact_imported_content_metadata_text,
)
from app.schemas.visual_references import (
    VISUAL_NOVEL_SPRITE_VARIANT_DIRECTIONS,
)


logger = logging.getLogger(__name__)

MAX_CUSTOM_VARIANTS_PER_PACK = 20
MAX_PENDING_CUSTOM_VARIANTS = 2
_VARIANT_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
_PRESENTATION_FOOTER_RE = re.compile(
    r"\s*<presentation>(?P<payload>.*?)</presentation>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_TRAILING_PRESENTATION_START_RE = re.compile(
    r"\s*<presentation>.*$",
    re.DOTALL | re.IGNORECASE,
)


def character_owned_sprite_pack_id(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
) -> str:
    """Return the character's private pack, never a viewer-specific veil."""

    if character.visuals.sprite_set_id:
        return character.visuals.sprite_set_id
    if checkpoint.session.config.settings.ruleset_id == "one_star_ascension":
        from app.engine.one_star_adapter import load_one_star_hero

        if load_one_star_hero(character) is not None:
            return generated_sprite_pack_id(checkpoint, character)
    return ""


def sync_character_presentation_scene(character: CharacterRecord) -> None:
    """Reset a carried display only when canonical location actually changes."""

    presentation = character.visuals.visual_novel_presentation
    location = " ".join((character.location or "").split()).strip()
    if presentation.scene_location and presentation.scene_location != location:
        presentation.current_variant_key = "neutral"
    presentation.scene_location = location


def _sync_character_presentation_pack(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
) -> str:
    presentation = character.visuals.visual_novel_presentation
    sprite_pack_id = character_owned_sprite_pack_id(checkpoint, character)
    if presentation.custom_variant_sprite_pack_id != sprite_pack_id:
        if presentation.current_variant_key.startswith("custom-"):
            presentation.current_variant_key = "neutral"
        presentation.custom_variant_directions = {}
        presentation.pending_requests = []
        presentation.custom_variant_sprite_pack_id = sprite_pack_id
    return sprite_pack_id


def character_presentation_catalog(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
) -> dict[str, str]:
    """Return the stable core catalog followed by current-pack custom variants."""

    _sync_character_presentation_pack(checkpoint, character)
    presentation = character.visuals.visual_novel_presentation
    safe_custom: dict[str, str] = {}
    for key, raw_direction in presentation.custom_variant_directions.items():
        direction = " ".join(
            redact_imported_content_metadata_text(
                strip_terminal_control(raw_direction)
            ).split()
        ).strip()
        if (
            not direction
            or REDACTED_IMPORT_SENTINEL in direction
            or "<" in direction
            or ">" in direction
        ):
            logger.warning(
                "Character %s has an unsafe VN presentation direction %s",
                character.character_id,
                key,
            )
            continue
        safe_custom[key] = direction
    return {
        **VISUAL_NOVEL_SPRITE_VARIANT_DIRECTIONS,
        **safe_custom,
    }


def format_character_presentation_catalog(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
) -> str:
    """Build the volatile text-only choice block for one agent call."""

    if checkpoint.session.config.settings.presentation_mode != "visual_novel":
        return ""
    sync_character_presentation_scene(character)
    presentation = character.visuals.visual_novel_presentation
    sprite_pack_id = _sync_character_presentation_pack(checkpoint, character)
    catalog = character_presentation_catalog(checkpoint, character)
    lines = [
        (
            "<presentation_catalog "
            f'current="{presentation.current_variant_key}">'
        ),
        "Choose the visible pose-expression that honestly matches your outward "
        "presentation in this beat. An empty use value keeps the current choice.",
    ]
    lines.extend(f"- {key}: {direction}" for key, direction in catalog.items())
    if sprite_pack_id:
        remaining = max(
            0,
            MAX_CUSTOM_VARIANTS_PER_PACK
            - len(presentation.custom_variant_directions)
            - len(presentation.pending_requests),
        )
        lines.append(
            "You may request one missing combined pose-expression in plain "
            f"visible terms; {remaining} custom slots and "
            f"{MAX_PENDING_CUSTOM_VARIANTS - len(presentation.pending_requests)} "
            "pending slots remain. Leave request empty unless the listed choices "
            "cannot express the beat."
        )
    else:
        lines.append("Custom presentation requests are unavailable for this body.")
    lines.append("</presentation_catalog>")
    return "\n".join(lines)


def parse_character_presentation_footer(
    text: str,
) -> tuple[str, CharacterPresentationChoice]:
    """Strip and parse one terminal footer; malformed metadata never leaks."""

    raw = text or ""
    match = _PRESENTATION_FOOTER_RE.search(raw)
    if match is None:
        # If the model started the private footer but malformed it, discard the
        # whole attempted suffix rather than exposing it as character prose.
        stripped = _TRAILING_PRESENTATION_START_RE.sub("", raw).rstrip()
        return stripped, CharacterPresentationChoice()
    stripped = raw[: match.start()].rstrip()
    try:
        payload = json.loads(match.group("payload"))
        choice = CharacterPresentationChoice.model_validate(payload)
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Character presentation footer was malformed and ignored")
        choice = CharacterPresentationChoice()
    return stripped, choice


def _custom_variant_key(direction: str) -> str:
    words = re.findall(r"[a-z0-9]+", direction.casefold())
    slug = "-".join(words[:5])[:52].strip("-") or "custom"
    digest = hashlib.sha256(direction.casefold().encode("utf-8")).hexdigest()[:10]
    return f"custom-{slug}-{digest}"


def apply_character_presentation_choice(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
    choice: CharacterPresentationChoice,
) -> None:
    """Commit one validated character choice and optional generation request."""

    if checkpoint.session.config.settings.presentation_mode != "visual_novel":
        return
    sync_character_presentation_scene(character)
    presentation = character.visuals.visual_novel_presentation
    sprite_pack_id = _sync_character_presentation_pack(checkpoint, character)
    catalog = character_presentation_catalog(checkpoint, character)
    use_key = choice.use.strip().lower()
    if use_key:
        if use_key in catalog:
            presentation.current_variant_key = use_key
        else:
            logger.warning(
                "Character %s selected unavailable VN presentation %s",
                character.character_id,
                use_key,
            )

    direction = " ".join(
        redact_imported_content_metadata_text(
            strip_terminal_control(choice.request)
        ).split()
    ).strip()
    if REDACTED_IMPORT_SENTINEL in direction:
        logger.warning(
            "Character %s submitted private metadata as a VN presentation request",
            character.character_id,
        )
        return
    if "<" in direction or ">" in direction:
        logger.warning(
            "Character %s submitted markup as a VN presentation request",
            character.character_id,
        )
        return
    if not direction or not sprite_pack_id:
        return
    variant_key = _custom_variant_key(direction)
    if not _VARIANT_KEY_RE.fullmatch(variant_key):
        return
    existing_directions: Mapping[str, str] = presentation.custom_variant_directions
    if variant_key in existing_directions or any(
        request.variant_key == variant_key
        for request in presentation.pending_requests
    ):
        return
    if (
        len(presentation.pending_requests) >= MAX_PENDING_CUSTOM_VARIANTS
        or len(presentation.custom_variant_directions)
        + len(presentation.pending_requests)
        >= MAX_CUSTOM_VARIANTS_PER_PACK
    ):
        logger.info(
            "Character %s VN custom presentation capacity reached",
            character.character_id,
        )
        return
    presentation.pending_requests.append(
        VisualNovelCustomVariantRequest(
            variant_key=variant_key,
            direction=direction,
            sprite_pack_id=sprite_pack_id,
            requested_turn_index=checkpoint.session.turn_index,
        )
    )


def select_player_character_presentation(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
    variant_key: str,
) -> None:
    """Apply a human's selection from the same existing catalog."""

    if checkpoint.session.config.settings.presentation_mode != "visual_novel":
        raise ValueError("Visual-novel displays require visual-novel presentation mode.")
    key = variant_key.strip().lower()
    if not key or key not in character_presentation_catalog(checkpoint, character):
        raise ValueError("Choose an available visual-novel display key.")
    sync_character_presentation_scene(character)
    character.visuals.visual_novel_presentation.current_variant_key = key
