"""Viewpoint-scoped One-Star visual identity presentation.

This adapter chooses only opaque sprite-set handles and player-safe exterior
vocabulary. It never resolves image bytes and never enters the generic
visual-novel compositor.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from app.engine.one_star_adapter import (
    load_one_star_account,
    load_one_star_hero,
)
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import (
    ONE_STAR_RULESET_ID,
    OneStarVisualNovelPresentationConfig,
)


VEILED_FIRST_LOOK = (
    "A basic Hero-shaped figure in plain beginner clothing. A soft dark "
    "System veil leaves every facial feature and identifying detail unreadable."
)

_FEMININE_PRESENTATION_RE = re.compile(
    r"\b(?:woman|women|female|girl|girls|she|her|hers|herself)\b",
    re.IGNORECASE,
)
_MASCULINE_PRESENTATION_RE = re.compile(
    r"\b(?:man|men|male|boy|boys|he|him|his|himself)\b",
    re.IGNORECASE,
)


def _veiled_presentation(character: CharacterRecord) -> str | None:
    """Use explicit authored exterior language, never inferred identity."""

    player_safe_exterior = " ".join((
        character.public_sheet.role,
        character.public_sheet.appearance,
        character.descriptions.public,
    ))
    feminine = bool(_FEMININE_PRESENTATION_RE.search(player_safe_exterior))
    masculine = bool(_MASCULINE_PRESENTATION_RE.search(player_safe_exterior))
    if feminine == masculine:
        return None
    return "feminine" if feminine else "masculine"


def one_star_visual_novel_config(
    checkpoint: CheckpointFile,
) -> tuple[str, OneStarVisualNovelPresentationConfig] | None:
    if checkpoint.session.config.settings.ruleset_id != ONE_STAR_RULESET_ID:
        return None
    owner, account = load_one_star_account(checkpoint)
    config = account.config.visual_novel_presentation
    if config is None:
        return None
    return owner.character_id, config


def generated_sprite_pack_id(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
) -> str:
    """Return a stable opaque pack id without exposing its identity inputs."""

    payload = {
        "session_id": checkpoint.session.session_id,
        "character_id": character.character_id,
        "name": character.name,
        "appearance": character.public_sheet.appearance,
        "loadout": character.visuals.default_loadout,
    }
    digest = hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return f"imgspritepack_{digest[:32]}"


def one_star_character_has_reviewed_sprite_set(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
) -> bool:
    """Return whether this Hero owns a reviewed, seed-authored sprite set.

    Character provenance and visual provenance are separate contracts. An
    authored dormant Hero may intentionally begin without bespoke art, while a
    generated summon may later carry a generated sprite-pack id. Only the
    checkpoint's reviewed sprite catalogue grants the earlier seeded reveal.
    """

    sprite_set_id = character.visuals.sprite_set_id.strip()
    if not sprite_set_id:
        return False
    return any(
        sprite_set.sprite_set_id == sprite_set_id
        and sprite_set.owner_character_id == character.character_id
        for sprite_set in checkpoint.reviewed_visual_novel_sprite_sets
    )


def one_star_identity_reveal_stars(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
) -> int | None:
    """Return the Master-facing identity-reveal threshold for a birth-one Hero."""

    configured = one_star_visual_novel_config(checkpoint)
    hero = load_one_star_hero(character)
    if configured is None or hero is None or hero.birth_stars != 1:
        return None
    _owner_id, config = configured
    if one_star_character_has_reviewed_sprite_set(checkpoint, character):
        return config.seeded_birth_one_reveal_stars
    return config.generated_birth_one_reveal_stars


def one_star_character_is_veiled_for_viewer(
    checkpoint: CheckpointFile,
    *,
    viewer_character_id: str,
    character: CharacterRecord,
) -> bool:
    configured = one_star_visual_novel_config(checkpoint)
    if configured is None:
        return False
    owner_id, _config = configured
    if viewer_character_id != owner_id:
        return False
    hero = load_one_star_hero(character)
    threshold = one_star_identity_reveal_stars(checkpoint, character)
    if hero is None or threshold is None:
        return False
    return hero.current_stars < threshold


def sprite_set_id_for_viewer(
    checkpoint: CheckpointFile,
    *,
    viewer_character_id: str,
    character: CharacterRecord,
) -> str:
    """Choose an authored, veiled, or generated set for one viewpoint."""

    configured = one_star_visual_novel_config(checkpoint)
    if configured is not None and one_star_character_is_veiled_for_viewer(
        checkpoint,
        viewer_character_id=viewer_character_id,
        character=character,
    ):
        _owner_id, config = configured
        presentation = _veiled_presentation(character)
        index = int.from_bytes(hashlib.sha256(
            (
                checkpoint.session.session_id
                + "\0"
                + character.character_id
            ).encode("utf-8")
        ).digest()[:8], "big") % len(config.veiled_sprite_set_ids)
        if presentation is None:
            presentation = ("masculine", "feminine")[index]
        return config.veiled_sprite_set_ids[presentation]
    if character.visuals.sprite_set_id:
        return character.visuals.sprite_set_id
    if load_one_star_hero(character) is not None:
        return generated_sprite_pack_id(checkpoint, character)
    return ""


def first_look_override_for_viewer(
    checkpoint: CheckpointFile,
    *,
    viewer_character_id: str,
    character: CharacterRecord,
) -> str | None:
    if one_star_character_is_veiled_for_viewer(
        checkpoint,
        viewer_character_id=viewer_character_id,
        character=character,
    ):
        return VEILED_FIRST_LOOK
    return None


def characters_needing_generated_sprite_prewarm(
    checkpoint: CheckpointFile,
    *,
    required_visible_character_ids: Iterable[str] = (),
) -> tuple[CharacterRecord, ...]:
    """Return birth-one Heroes without reviewed art approaching reveal."""

    configured = one_star_visual_novel_config(checkpoint)
    if configured is None:
        return ()

    owner_id, _config = configured
    required_ids = {
        character_id.strip()
        for character_id in required_visible_character_ids
        if character_id.strip()
    }

    def ready_for_prewarm(character: CharacterRecord) -> bool:
        hero = load_one_star_hero(character)
        if hero is None:
            return False
        if (
            character.character_id in required_ids
            and not one_star_character_is_veiled_for_viewer(
                checkpoint,
                viewer_character_id=owner_id,
                character=character,
            )
        ):
            return True
        reveal_stars = one_star_identity_reveal_stars(checkpoint, character)
        if hero.birth_stars != 1 or reveal_stars is None:
            return False
        if one_star_character_has_reviewed_sprite_set(checkpoint, character):
            return False
        # Start the neutral candidate one promotion before the Master-facing
        # reveal so ordinary image latency does not stall that story beat.
        return hero.current_stars >= max(
            1,
            reveal_stars - 1,
        )
    return tuple(
        character
        for character in checkpoint.characters
        if character.status != CharacterStatus.culled
        and load_one_star_hero(character) is not None
        and ready_for_prewarm(character)
        and not character.visuals.sprite_set_id
        and (
            character.player_slot_kind.value != "player_authored"
            or character.character_id
            in checkpoint.session.character_bindings
            or checkpoint.session.player_character_id == character.character_id
        )
    )
