"""Deterministic Master-facing One-Star Hero card boards.

The One-Star adapter owns event selection and rank data.  This module turns
that committed state into ordinary immutable player media; it never sends
visual bytes or presentation provenance to a model.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from app.engine.one_star_adapter import (
    is_one_star_checkpoint,
    load_one_star_account,
    load_one_star_hero,
)
from app.engine.one_star_visuals import (
    one_star_character_has_reviewed_sprite_set,
    one_star_character_is_veiled_for_viewer,
    one_star_visual_novel_config,
    sprite_set_id_for_viewer,
)
from app.engine.player_media import PlayerMediaBytes, ResolvedPlayerMedia
from app.engine.reviewed_visual_references import (
    ReviewedVisualReferenceError,
    resolve_frozen_visual_reference_media,
)
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import (
    OneStarOpeningRosterSummonPool,
    OneStarSummonRevealBand,
)
from app.schemas.responses import VisualNovelRender

if TYPE_CHECKING:
    from app.engine.image_generation import ImageGenerationCoordinator


logger = logging.getLogger(__name__)

CARD_SIZE = (640, 1024)
BOARD_SIZE = (1024, 576)
PORTRAIT_BOX = (70, 135, 570, 842)
NAME_BOX = (108, 872, 532, 959)
NAMEPLATE_FRONT_Y = 832
MAX_HEROES_PER_BOARD = 5

_SERIF = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
_SERIF_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")
_SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
_SANS_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

HeroCardEventKind = Literal["summon", "mission_start"]
HeroCardBoardLayout = Literal["group", "individual"]
PortraitSource = Literal["override", "neutral", "unavailable"]
Color = tuple[int, int, int]

_BOARD_TITLES: dict[HeroCardEventKind, tuple[str, str]] = {
    "summon": ("HEROES ACQUIRED", "Heroes acquired"),
    "mission_start": ("DEPLOYMENT CONFIRMED", "Deployment confirmed"),
}


class OneStarHeroCardError(RuntimeError):
    """A committed Hero-card presentation contract could not be fulfilled."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"One-Star Hero card presentation failed ({code}).")


@dataclass(frozen=True)
class OneStarHeroCardEvent:
    event_id: str
    kind: HeroCardEventKind
    characters: tuple[CharacterRecord, ...]


@dataclass(frozen=True)
class OneStarHeroCardBoard:
    media: ResolvedPlayerMedia
    accessible_text: str
    event_id: str
    kind: HeroCardEventKind
    page_number: int
    page_count: int
    layout: HeroCardBoardLayout = "group"


@dataclass(frozen=True)
class OneStarSummonCardBoards:
    individual_boards: tuple[OneStarHeroCardBoard, ...]
    group_boards: tuple[OneStarHeroCardBoard, ...]


@dataclass(frozen=True)
class OneStarSummonReveal:
    media: ResolvedPlayerMedia
    accessible_text: str
    event_id: str
    band: OneStarSummonRevealBand
    pull_number: int
    pull_count: int


@dataclass(frozen=True)
class _Portrait:
    media: PlayerMediaBytes | None
    source: PortraitSource


@dataclass(frozen=True)
class _RankStyle:
    stars: int
    border_shadow: Color
    border_mid: Color
    border_highlight: Color
    star_fill: Color
    star_highlight: Color
    star_outline: Color
    glow: Color
    tint_strength: float
    frame_glow_alpha: int
    star_glow_alpha: int


_RANK_STYLES = (
    _RankStyle(1, (8, 10, 13), (31, 35, 41), (80, 88, 99), (96, 105, 116), (174, 182, 191), (21, 24, 29), (107, 117, 130), .87, 3, 12),
    _RankStyle(2, (13, 15, 19), (62, 68, 76), (142, 152, 164), (148, 159, 171), (215, 224, 231), (27, 30, 36), (134, 147, 161), .85, 15, 33),
    _RankStyle(3, (15, 18, 22), (79, 86, 95), (177, 187, 198), (183, 194, 206), (235, 241, 246), (31, 34, 40), (155, 169, 183), .84, 26, 53),
    _RankStyle(4, (20, 22, 26), (116, 123, 132), (238, 244, 250), (232, 239, 246), (255, 255, 255), (36, 39, 44), (193, 205, 217), .84, 44, 82),
    _RankStyle(5, (28, 22, 11), (112, 83, 38), (207, 168, 91), (210, 158, 58), (247, 221, 157), (61, 42, 12), (203, 147, 45), .66, 58, 105),
    _RankStyle(6, (39, 29, 9), (174, 128, 35), (255, 222, 122), (248, 194, 62), (255, 244, 184), (75, 50, 9), (245, 183, 51), .80, 86, 143),
    _RankStyle(7, (45, 36, 13), (223, 183, 76), (255, 255, 238), (255, 234, 142), (255, 255, 255), (84, 63, 9), (255, 226, 122), .90, 120, 188),
)
_STYLE_BY_STARS = {style.stars: style for style in _RANK_STYLES}

_SUMMON_REVEAL_ACCESSIBLE_BANDS: dict[OneStarSummonRevealBand, str] = {
    "under_3": "an iron response signals a one- or two-star result",
    "3_to_4": "a silver response signals a three- or four-star result",
    "5_to_6": "a gold response signals a five- or six-star result",
    "7": "a white-gold response signals the highest-rank result",
}


def committed_one_star_hero_card_event(
    checkpoint: CheckpointFile,
    event_id: str,
) -> OneStarHeroCardEvent | None:
    """Resolve one card-worthy event only from committed canonical state."""

    if not is_one_star_checkpoint(checkpoint):
        return None
    clean_event_id = event_id.strip()
    _owner, account = load_one_star_account(checkpoint)
    if clean_event_id not in account.state.applied_event_fingerprints:
        return None
    matches = [
        event
        for event in checkpoint.canonical_events
        if event.event_id == clean_event_id
    ]
    if len(matches) != 1:
        if len(matches) > 1:
            raise OneStarHeroCardError("duplicate_canonical_event")
        return None
    event = matches[0]
    card_updates = [
        update
        for update in getattr(event, "state_updates", ())
        if update.kind in {"summon", "mission_start"}
    ]
    if not card_updates:
        return None
    if len(card_updates) == 1:
        update = card_updates[0]
    else:
        summons = [update for update in card_updates if update.kind == "summon"]
        mission_starts = [
            update for update in card_updates if update.kind == "mission_start"
        ]
        opening_pool = (
            account.config.summon_pools.get(summons[0].target_id.strip())
            if len(summons) == 1
            else None
        )
        if (
            len(card_updates) == 2
            and len(summons) == 1
            and len(mission_starts) == 1
            and card_updates[0] is summons[0]
            and card_updates[1] is mission_starts[0]
            and isinstance(opening_pool, OneStarOpeningRosterSummonPool)
        ):
            # The direct opening acquires and deploys one roster atomically.
            # Its summon reveal is the one Master-facing card for that event.
            update = summons[0]
        else:
            raise OneStarHeroCardError("ambiguous_card_event")
    characters_by_id = {
        character.character_id: character for character in checkpoint.characters
    }
    if update.kind == "summon":
        acquired = {
            character.character_id: character
            for character in checkpoint.characters
            if (
                (hero := load_one_star_hero(character)) is not None
                and hero.acquisition_event_id == clean_event_id
            )
        }
        ordered_ids: list[str] = []
        for character_id in (
            *(
                request.character_id
                for request in getattr(event, "spawn", ())
            ),
            *(
                signal.character_id
                for signal in getattr(event, "activate", ())
            ),
        ):
            if character_id in acquired and character_id not in ordered_ids:
                ordered_ids.append(character_id)
        ordered_ids.extend(
            character.character_id
            for character in checkpoint.characters
            if character.character_id in acquired
            and character.character_id not in ordered_ids
        )
        if not ordered_ids:
            raise OneStarHeroCardError("summon_has_no_acquired_heroes")
    else:
        ordered_ids = []
        for detail in update.details:
            key, separator, value = detail.partition("=")
            if separator and key.strip() == "party" and value.strip():
                ordered_ids.append(value.strip())
        if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
            raise OneStarHeroCardError("mission_party_order_invalid")

    characters: list[CharacterRecord] = []
    for character_id in ordered_ids:
        character = characters_by_id.get(character_id)
        if character is None or load_one_star_hero(character) is None:
            raise OneStarHeroCardError("card_hero_missing")
        characters.append(character)
    return OneStarHeroCardEvent(
        event_id=clean_event_id,
        kind=update.kind,
        characters=tuple(characters),
    )


def new_one_star_hero_card_events(
    checkpoint: CheckpointFile,
    previous_checkpoint: CheckpointFile | None,
) -> tuple[OneStarHeroCardEvent, ...]:
    """Return newly committed card events in canonical order."""

    if not is_one_star_checkpoint(checkpoint):
        return ()
    _owner, account = load_one_star_account(checkpoint)
    previous_ids: set[str] = set()
    if previous_checkpoint is not None and is_one_star_checkpoint(
        previous_checkpoint
    ):
        _previous_owner, previous_account = load_one_star_account(
            previous_checkpoint
        )
        previous_ids = set(previous_account.state.applied_event_fingerprints)
    committed_ids = set(account.state.applied_event_fingerprints) - previous_ids
    result: list[OneStarHeroCardEvent] = []
    seen: set[str] = set()
    for canonical_event in checkpoint.canonical_events:
        event_id = canonical_event.event_id
        if event_id in seen or event_id not in committed_ids:
            continue
        seen.add(event_id)
        event = committed_one_star_hero_card_event(checkpoint, event_id)
        if event is not None:
            result.append(event)
    return tuple(result)


def one_star_hero_card_events_for_render(
    *,
    checkpoint: CheckpointFile,
    previous_checkpoint: CheckpointFile | None,
    viewer_character_id: str,
    render: VisualNovelRender,
) -> tuple[OneStarHeroCardEvent, ...]:
    """Require and order the Master's new cards by rendered segment."""

    configured = one_star_visual_novel_config(checkpoint)
    if configured is None:
        return ()
    owner_id, _config = configured
    if viewer_character_id != owner_id:
        return ()
    required = {
        event.event_id: event
        for event in new_one_star_hero_card_events(
            checkpoint,
            previous_checkpoint,
        )
    }
    rendered_ids = [
        event_id
        for segment in render.segments
        for event_id in segment.rendered_event_ids
    ]
    missing = set(required) - set(rendered_ids)
    if missing:
        raise OneStarHeroCardError("master_render_missing_card_event")
    ordered: list[OneStarHeroCardEvent] = []
    seen: set[str] = set()
    for event_id in rendered_ids:
        if event_id in required and event_id not in seen:
            ordered.append(required[event_id])
            seen.add(event_id)
    return tuple(ordered)


def generated_portrait_prewarm_character_ids(
    *,
    checkpoint: CheckpointFile,
    viewer_character_id: str,
    events: Sequence[OneStarHeroCardEvent],
) -> tuple[str, ...]:
    """Select visible, unreviewed Heroes whose neutral pack may be generated."""

    result: list[str] = []
    for event in events:
        if event.kind != "summon":
            continue
        for character in event.characters:
            hero = load_one_star_hero(character)
            if (
                hero is None
                or not hero.generated_for_summon
                or one_star_character_has_reviewed_sprite_set(
                    checkpoint,
                    character,
                )
                or one_star_character_is_veiled_for_viewer(
                    checkpoint,
                    viewer_character_id=viewer_character_id,
                    character=character,
                )
            ):
                continue
            result.append(character.character_id)
    return tuple(dict.fromkeys(result))


def render_one_star_hero_card_boards(
    *,
    checkpoint: CheckpointFile,
    viewer_character_id: str,
    event: OneStarHeroCardEvent,
    generation: ImageGenerationCoordinator,
) -> tuple[OneStarHeroCardBoard, ...]:
    """Compose deterministic, transport-neutral boards for one event."""

    rendered = _rendered_one_star_hero_cards(
        checkpoint=checkpoint,
        viewer_character_id=viewer_character_id,
        event=event,
        generation=generation,
    )
    if rendered is None:
        return ()
    return _render_one_star_group_boards(event=event, rendered=rendered)


def render_one_star_summon_card_boards(
    *,
    checkpoint: CheckpointFile,
    viewer_character_id: str,
    event: OneStarHeroCardEvent,
    generation: ImageGenerationCoordinator,
) -> OneStarSummonCardBoards:
    """Compose each pull result and the final party summary in one pass."""

    if event.kind != "summon":
        return OneStarSummonCardBoards((), ())
    rendered = _rendered_one_star_hero_cards(
        checkpoint=checkpoint,
        viewer_character_id=viewer_character_id,
        event=event,
        generation=generation,
    )
    if rendered is None:
        return OneStarSummonCardBoards((), ())

    pull_count = len(rendered)
    individual_boards: list[OneStarHeroCardBoard] = []
    for pull_number, (character, portrait, card) in enumerate(
        rendered,
        start=1,
    ):
        board = _render_board(
            (card,),
            kind="summon",
            page_number=pull_number,
            page_count=pull_count,
            layout="individual",
        )
        individual_boards.append(OneStarHeroCardBoard(
            media=_board_media(
                board,
                filename=(
                    f"one-star-hero-pull-{pull_number:02d}-of-"
                    f"{pull_count:02d}.png"
                ),
            ),
            accessible_text=(
                f"System panel — Hero acquired (pull {pull_number} of "
                f"{pull_count}): {_hero_card_accessible_entry(character, portrait)}"
            ),
            event_id=event.event_id,
            kind=event.kind,
            page_number=pull_number,
            page_count=pull_count,
            layout="individual",
        ))
    return OneStarSummonCardBoards(
        individual_boards=tuple(individual_boards),
        group_boards=_render_one_star_group_boards(
            event=event,
            rendered=rendered,
        ),
    )


def _rendered_one_star_hero_cards(
    *,
    checkpoint: CheckpointFile,
    viewer_character_id: str,
    event: OneStarHeroCardEvent,
    generation: ImageGenerationCoordinator,
) -> list[tuple[CharacterRecord, _Portrait, Image.Image]] | None:
    configured = one_star_visual_novel_config(checkpoint)
    if configured is None or configured[0] != viewer_character_id:
        return None
    _owner_id, presentation = configured
    frame_id = presentation.hero_card_frame_reference_id
    if not frame_id:
        raise OneStarHeroCardError("presentation_frame_not_configured")
    frozen_frame = generation.store.reviewed_reference(
        session_id=checkpoint.session.session_id,
        reference_id=frame_id,
    )
    if frozen_frame is None:
        raise OneStarHeroCardError("presentation_frame_unavailable")
    try:
        frame_media = resolve_frozen_visual_reference_media(
            frozen_frame,
            runtime_root=generation.config.runtime_root,
        )
        frame = _decode_rgba(frame_media)
    except (ReviewedVisualReferenceError, OSError, ValueError) as exc:
        raise OneStarHeroCardError("presentation_frame_invalid") from exc
    if frame.size != CARD_SIZE:
        raise OneStarHeroCardError("presentation_frame_dimensions_mismatch")

    rendered: list[tuple[CharacterRecord, _Portrait, Image.Image]] = []
    for character in event.characters:
        portrait = _resolve_portrait(
            checkpoint=checkpoint,
            viewer_character_id=viewer_character_id,
            character=character,
            generation=generation,
        )
        rendered.append((
            character,
            portrait,
            render_one_star_hero_card(
                frame=frame,
                portrait=portrait.media,
                portrait_source=portrait.source,
                display_name=character.name,
                stars=_current_stars(character),
            ),
        ))
    return rendered


def _render_one_star_group_boards(
    *,
    event: OneStarHeroCardEvent,
    rendered: Sequence[tuple[CharacterRecord, _Portrait, Image.Image]],
) -> tuple[OneStarHeroCardBoard, ...]:

    groups = [
        rendered[start:start + MAX_HEROES_PER_BOARD]
        for start in range(0, len(rendered), MAX_HEROES_PER_BOARD)
    ]
    page_count = len(groups)
    boards: list[OneStarHeroCardBoard] = []
    for page_number, group in enumerate(groups, start=1):
        board = _render_board(
            [card for _character, _portrait, card in group],
            kind=event.kind,
            page_number=page_number,
            page_count=page_count,
        )
        title = _BOARD_TITLES[event.kind][1]
        entries = [
            _hero_card_accessible_entry(character, portrait)
            for character, portrait, _card in group
        ]
        page_suffix = (
            f" (page {page_number} of {page_count})" if page_count > 1 else ""
        )
        boards.append(OneStarHeroCardBoard(
            media=_board_media(
                board,
                filename=f"one-star-hero-board-{page_number:02d}.png",
            ),
            accessible_text=(
                f"System panel — {title}{page_suffix}: " + "; ".join(entries)
            ),
            event_id=event.event_id,
            kind=event.kind,
            page_number=page_number,
            page_count=page_count,
        ))
    return tuple(boards)


def _hero_card_accessible_entry(
    character: CharacterRecord,
    portrait: _Portrait,
) -> str:
    stars = _current_stars(character)
    entry = (
        f"{character.name} — {stars} "
        f"{'star' if stars == 1 else 'stars'}"
    )
    if portrait.source == "unavailable":
        entry += " (portrait unavailable)"
    return entry


def _board_media(board: Image.Image, *, filename: str) -> ResolvedPlayerMedia:
    data = _png_bytes(board)
    return ResolvedPlayerMedia(
        filename=filename,
        mime_type="image/png",
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        width=BOARD_SIZE[0],
        height=BOARD_SIZE[1],
    )


def one_star_summon_reveal_band(stars: int) -> OneStarSummonRevealBand:
    """Map the current Hero rank to the approved metal-milestone animation."""

    if stars in {1, 2}:
        return "under_3"
    if stars in {3, 4}:
        return "3_to_4"
    if stars in {5, 6}:
        return "5_to_6"
    if stars == 7:
        return "7"
    raise OneStarHeroCardError("star_rank_out_of_range")


def render_one_star_summon_reveals(
    *,
    checkpoint: CheckpointFile,
    viewer_character_id: str,
    event: OneStarHeroCardEvent,
    generation: ImageGenerationCoordinator,
) -> tuple[OneStarSummonReveal, ...]:
    """Resolve approved pre-result motion for one committed summon event."""

    configured = one_star_visual_novel_config(checkpoint)
    if (
        configured is None
        or configured[0] != viewer_character_id
        or event.kind != "summon"
    ):
        return ()
    _owner_id, presentation = configured
    reference_ids = presentation.summon_reveal_reference_ids
    if not reference_ids:
        return ()

    pull_count = len(event.characters)
    reveals: list[OneStarSummonReveal] = []
    for pull_number, character in enumerate(event.characters, start=1):
        band = one_star_summon_reveal_band(_current_stars(character))
        reference_id = reference_ids[band]
        frozen = generation.store.reviewed_reference(
            session_id=checkpoint.session.session_id,
            reference_id=reference_id,
        )
        if frozen is None:
            raise OneStarHeroCardError("summon_reveal_unavailable")
        try:
            media = resolve_frozen_visual_reference_media(
                frozen,
                runtime_root=generation.config.runtime_root,
            )
        except ReviewedVisualReferenceError as exc:
            raise OneStarHeroCardError("summon_reveal_invalid") from exc
        if (
            media.mime_type != "image/gif"
            or media.width != BOARD_SIZE[0]
            or media.height != BOARD_SIZE[1]
        ):
            raise OneStarHeroCardError("summon_reveal_invalid")
        reveals.append(OneStarSummonReveal(
            media=media,
            accessible_text=(
                f"Summon reveal — pull {pull_number} of {pull_count}: the seal "
                f"strains, recoils, and releases; "
                f"{_SUMMON_REVEAL_ACCESSIBLE_BANDS[band]}. "
                "The Hero identity remains sealed until the next static card."
            ),
            event_id=event.event_id,
            band=band,
            pull_number=pull_number,
            pull_count=pull_count,
        ))
    return tuple(reveals)


def render_one_star_hero_card(
    *,
    frame: Image.Image,
    portrait: PlayerMediaBytes | None,
    portrait_source: PortraitSource,
    display_name: str,
    stars: int,
) -> Image.Image:
    """Render one 640x1024 card with the reviewed Option F treatment."""

    style = _STYLE_BY_STARS.get(stars)
    if style is None:
        raise OneStarHeroCardError("star_rank_out_of_range")
    tinted_frame = _tint_frame(frame, style)
    portrait_layer = _portrait_layer(portrait, portrait_source)
    stars_layer = Image.new("RGBA", CARD_SIZE, (0, 0, 0, 0))
    _draw_rank_stars(stars_layer, style)
    interface = Image.new("RGBA", CARD_SIZE, (0, 0, 0, 0))
    _draw_name(interface, display_name)
    _draw_niflheim_emblem(interface)

    card = _card_plate()
    card = Image.alpha_composite(card, _frame_glow(tinted_frame, style))
    card = Image.alpha_composite(card, tinted_frame)
    # The bust intentionally sits over the rank border, then the lower
    # nameplate and star interface restore the foreground hierarchy.
    card = Image.alpha_composite(card, portrait_layer)
    card = Image.alpha_composite(card, _nameplate_foreground(tinted_frame))
    card = Image.alpha_composite(card, stars_layer)
    card = Image.alpha_composite(card, interface)
    return card


def _resolve_portrait(
    *,
    checkpoint: CheckpointFile,
    viewer_character_id: str,
    character: CharacterRecord,
    generation: ImageGenerationCoordinator,
) -> _Portrait:
    sprite_set_id = sprite_set_id_for_viewer(
        checkpoint,
        viewer_character_id=viewer_character_id,
        character=character,
    )
    authored = next((
        sprite_set
        for sprite_set in checkpoint.reviewed_visual_novel_sprite_sets
        if sprite_set.sprite_set_id == sprite_set_id
    ), None)

    def reviewed(reference_id: str) -> PlayerMediaBytes | None:
        if not reference_id:
            return None
        frozen = generation.store.reviewed_reference(
            session_id=checkpoint.session.session_id,
            reference_id=reference_id,
        )
        if frozen is None:
            return None
        try:
            return resolve_frozen_visual_reference_media(
                frozen,
                runtime_root=generation.config.runtime_root,
            )
        except ReviewedVisualReferenceError as exc:
            logger.warning(
                "One-Star Hero portrait validation failed session=%s code=%s",
                checkpoint.session.session_id,
                exc.code,
            )
            return None

    if authored is not None:
        override = reviewed(authored.portrait_reference_id)
        if override is not None:
            return _Portrait(override, "override")
        neutral = reviewed(authored.variant_reference_ids.get("neutral", ""))
        if neutral is not None:
            return _Portrait(neutral, "neutral")
    generated = generation.resolve_visual_novel_sprite_variant(
        session_id=checkpoint.session.session_id,
        character_id=character.character_id,
        sprite_pack_id=sprite_set_id,
        variant_key="neutral",
    )
    if generated is not None:
        _handle, media, _source_facing = generated
        return _Portrait(media, "neutral")
    return _Portrait(None, "unavailable")


def _current_stars(character: CharacterRecord) -> int:
    hero = load_one_star_hero(character)
    if hero is None:
        raise OneStarHeroCardError("card_character_is_not_hero")
    return hero.current_stars


def _decode_rgba(media: PlayerMediaBytes) -> Image.Image:
    with Image.open(BytesIO(media.data)) as opened:
        image = opened.convert("RGBA")
        image.load()
    return image


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def _gradient(
    size: tuple[int, int],
    top: Color,
    bottom: Color,
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    denominator = max(1, height - 1)
    for y in range(height):
        ratio = y / denominator
        color = tuple(
            round(start * (1 - ratio) + end * ratio)
            for start, end in zip(top, bottom, strict=True)
        )
        draw.line((0, y, width, y), fill=color)
    return image


def _card_plate() -> Image.Image:
    plate = _gradient(CARD_SIZE, (10, 18, 35), (3, 5, 12)).convert("RGBA")
    decoration = Image.new("RGBA", CARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(decoration)
    draw.ellipse((93, 146, 547, 600), outline=(84, 121, 153, 38), width=3)
    draw.ellipse((139, 192, 501, 554), outline=(188, 145, 76, 24), width=2)
    for radius in (175, 225, 275):
        draw.arc(
            (320 - radius, 220 - radius, 320 + radius, 220 + radius),
            195,
            345,
            fill=(117, 150, 175, 28),
            width=2,
        )
    for x in range(98, 550, 42):
        draw.line((x, 170, x + 150, 775), fill=(121, 151, 173, 13), width=2)
    draw.rounded_rectangle((93, 853, 547, 978), radius=18, fill=(1, 3, 8, 238))
    plate = Image.alpha_composite(plate, decoration)
    mask = Image.new("L", CARD_SIZE, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (10, 18, 630, 1014),
        radius=36,
        fill=255,
    )
    plate.putalpha(mask)
    return plate


def _tint_frame(frame: Image.Image, style: _RankStyle) -> Image.Image:
    source = frame.convert("RGBA")
    alpha = source.getchannel("A")
    graded = ImageOps.colorize(
        ImageOps.grayscale(source),
        black=style.border_shadow,
        mid=style.border_mid,
        white=style.border_highlight,
    )
    result = Image.blend(
        source.convert("RGB"),
        graded,
        style.tint_strength,
    ).convert("RGBA")
    result.putalpha(alpha)
    return result


def _frame_glow(frame: Image.Image, style: _RankStyle) -> Image.Image:
    alpha = frame.getchannel("A").filter(
        ImageFilter.GaussianBlur(5 + style.stars * .8)
    )
    alpha = alpha.point(
        lambda value: round(value * style.frame_glow_alpha / 255)
    )
    glow = Image.new("RGBA", CARD_SIZE, (*style.glow, 0))
    glow.putalpha(alpha)
    return glow


def _nameplate_foreground(frame: Image.Image) -> Image.Image:
    region = Image.new("L", CARD_SIZE, 0)
    ImageDraw.Draw(region).rectangle(
        (0, NAMEPLATE_FRONT_Y, CARD_SIZE[0], CARD_SIZE[1]),
        fill=255,
    )
    result = frame.copy()
    result.putalpha(ImageChops.multiply(frame.getchannel("A"), region))
    return result


def _portrait_layer(
    media: PlayerMediaBytes | None,
    source: PortraitSource,
) -> Image.Image:
    layer = Image.new("RGBA", CARD_SIZE, (0, 0, 0, 0))
    target_width = PORTRAIT_BOX[2] - PORTRAIT_BOX[0]
    target_height = PORTRAIT_BOX[3] - PORTRAIT_BOX[1]
    target = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
    if media is None or source == "unavailable":
        _draw_unavailable_portrait(target)
    else:
        try:
            original = _decode_rgba(media)
            alpha = original.getchannel("A")
            bbox = alpha.point(
                lambda value: 255 if value >= 32 else 0
            ).getbbox()
            if bbox is None:
                raise ValueError("portrait has no visible subject")
            x0, y0, x1, y1 = bbox
            if source == "neutral" and (y1 - y0) > (x1 - x0) * 1.1:
                y1 = max(y0 + 1, round(y0 + (y1 - y0) * .66))
            cropped = original.crop((x0, y0, x1, y1))
            scale = min(
                target_width / cropped.width,
                target_height / cropped.height,
            )
            resized = cropped.convert("RGBa").resize(
                (
                    max(1, round(cropped.width * scale)),
                    max(1, round(cropped.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            ).convert("RGBA")
            x = (target_width - resized.width) // 2
            y = target_height - resized.height
            if source == "override":
                y = min(y, 18)
            target.alpha_composite(resized, (x, y))
        except (OSError, ValueError):
            target = Image.new(
                "RGBA",
                (target_width, target_height),
                (0, 0, 0, 0),
            )
            _draw_unavailable_portrait(target)
    layer.alpha_composite(target, (PORTRAIT_BOX[0], PORTRAIT_BOX[1]))
    fade_height = 155
    fade = Image.new("RGBA", (target_width, fade_height))
    fade_draw = ImageDraw.Draw(fade)
    for y in range(fade_height):
        alpha = round(205 * (y / max(1, fade_height - 1)) ** 1.7)
        fade_draw.line((0, y, target_width, y), fill=(2, 4, 10, alpha))
    layer.alpha_composite(fade, (PORTRAIT_BOX[0], PORTRAIT_BOX[3] - fade_height))
    return layer


def _draw_unavailable_portrait(target: Image.Image) -> None:
    draw = ImageDraw.Draw(target)
    center = target.width // 2
    draw.ellipse(
        (center - 86, 120, center + 86, 292),
        fill=(22, 27, 37, 255),
        outline=(113, 124, 139, 210),
        width=4,
    )
    draw.rounded_rectangle(
        (center - 154, 278, center + 154, 655),
        radius=92,
        fill=(17, 21, 30, 255),
        outline=(86, 97, 112, 210),
        width=4,
    )
    label = "PORTRAIT UNAVAILABLE"
    face = _font(_SANS_BOLD, 19)
    box = draw.textbbox((0, 0), label, font=face)
    draw.text(
        ((target.width - (box[2] - box[0])) / 2, 50),
        label,
        font=face,
        fill=(186, 194, 204, 255),
    )


def _star_points(
    center_x: float,
    center_y: float,
    outer: float,
    inner: float,
) -> list[tuple[float, float]]:
    return [
        (
            center_x + math.cos(-math.pi / 2 + index * math.pi / 5) * radius,
            center_y + math.sin(-math.pi / 2 + index * math.pi / 5) * radius,
        )
        for index in range(10)
        for radius in (outer if index % 2 == 0 else inner,)
    ]


def _draw_rank_stars(layer: Image.Image, style: _RankStyle) -> None:
    spacing = 58
    start = 320 - spacing * (style.stars - 1) / 2
    glow = Image.new("RGBA", CARD_SIZE, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    radius = 29 + style.stars * .65
    for index in range(style.stars):
        center_x = start + index * spacing
        glow_draw.ellipse(
            (
                center_x - radius,
                816 - radius,
                center_x + radius,
                816 + radius,
            ),
            fill=(*style.glow, style.star_glow_alpha),
        )
    layer.alpha_composite(
        glow.filter(ImageFilter.GaussianBlur(9 + style.stars * .7))
    )
    draw = ImageDraw.Draw(layer)
    for index in range(style.stars):
        center_x = start + index * spacing
        draw.polygon(
            _star_points(center_x, 816, 25, 10.7),
            fill=(*style.star_fill, 255),
            outline=(*style.star_outline, 255),
            width=3,
        )
        draw.polygon(
            _star_points(center_x - 2.5, 813.5, 13, 5.5),
            fill=(*style.star_highlight, 215),
        )


def _text_size(
    draw: ImageDraw.ImageDraw,
    value: str,
    face: ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), value, font=face)
    return right - left, bottom - top


def _name_lines(
    draw: ImageDraw.ImageDraw,
    value: str,
    max_width: int,
    max_height: int,
) -> tuple[list[str], ImageFont.FreeTypeFont]:
    normalized = " ".join(value.split())
    if not normalized:
        raise OneStarHeroCardError("hero_name_empty")
    candidates: list[list[str]] = [[normalized]]
    candidates.extend(
        [normalized[:split].rstrip(), normalized[split:].lstrip()]
        for split in range(1, len(normalized))
        if normalized[:split].rstrip() and normalized[split:].lstrip()
    )
    for size in range(48, 9, -1):
        face = _font(_SERIF_BOLD, size)
        viable: list[tuple[tuple[int, int, int], list[str]]] = []
        for lines in candidates:
            widths = [_text_size(draw, line, face)[0] for line in lines]
            line_height = _text_size(draw, "Ag", face)[1]
            total_height = line_height * len(lines) + (4 if len(lines) == 2 else 0)
            if max(widths) <= max_width and total_height <= max_height:
                word_breaks = int(
                    len(lines) == 2
                    and normalized[len(lines[0]):len(lines[0]) + 1] != " "
                )
                raggedness = max(widths) - min(widths) if len(widths) == 2 else 0
                viable.append(((len(lines) - 1, word_breaks, raggedness), lines))
        if viable:
            viable.sort(key=lambda item: item[0])
            return viable[0][1], face
    raise OneStarHeroCardError("hero_name_does_not_fit")


def _draw_name(layer: Image.Image, value: str) -> None:
    draw = ImageDraw.Draw(layer)
    lines, face = _name_lines(
        draw,
        value,
        NAME_BOX[2] - NAME_BOX[0],
        NAME_BOX[3] - NAME_BOX[1],
    )
    line_height = _text_size(draw, "Ag", face)[1]
    gap = 4 if len(lines) == 2 else 0
    total_height = line_height * len(lines) + gap
    y = NAME_BOX[1] + (NAME_BOX[3] - NAME_BOX[1] - total_height) / 2
    for line in lines:
        width, _height = _text_size(draw, line, face)
        x = (CARD_SIZE[0] - width) / 2
        draw.text(
            (x + 2, y + 2),
            line,
            font=face,
            fill=(0, 0, 0, 220),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 220),
        )
        draw.text(
            (x, y),
            line,
            font=face,
            fill=(244, 240, 228, 255),
            stroke_width=1,
            stroke_fill=(111, 76, 31, 255),
        )
        y += line_height + gap


def _draw_niflheim_emblem(layer: Image.Image) -> None:
    draw = ImageDraw.Draw(layer)
    center_x, center_y = 320, 73
    draw.ellipse(
        (center_x - 35, center_y - 35, center_x + 35, center_y + 35),
        fill=(4, 8, 15, 238),
    )
    draw.ellipse(
        (center_x - 29, center_y - 29, center_x + 29, center_y + 29),
        outline=(191, 155, 90, 255),
        width=3,
    )
    draw.arc(
        (center_x - 19, center_y - 20, center_x + 19, center_y + 19),
        195,
        345,
        fill=(188, 224, 232, 255),
        width=4,
    )
    for offset, height in ((-13, 20), (0, 28), (13, 20)):
        draw.line(
            (center_x + offset, center_y - 1, center_x + offset, center_y + height),
            fill=(188, 224, 232, 255),
            width=4,
        )
    draw.line(
        (
            center_x - 18,
            center_y + 20,
            center_x,
            center_y + 29,
            center_x + 18,
            center_y + 20,
        ),
        fill=(191, 155, 90, 255),
        width=3,
    )


def _render_board(
    cards: Sequence[Image.Image],
    *,
    kind: HeroCardEventKind,
    page_number: int,
    page_count: int,
    layout: HeroCardBoardLayout = "group",
) -> Image.Image:
    board = _gradient(BOARD_SIZE, (8, 19, 37), (2, 5, 13)).convert("RGBA")
    draw = ImageDraw.Draw(board)
    draw.rectangle((0, 0, 1023, 575), outline=(144, 116, 66, 170), width=3)
    title = "HERO ACQUIRED" if layout == "individual" else _BOARD_TITLES[kind][0]
    draw.text(
        (38, 24),
        title,
        font=_font(_SERIF_BOLD, 29),
        fill=(239, 235, 222, 255),
    )
    if layout == "individual":
        if len(cards) != 1 or not 1 <= page_number <= page_count:
            raise ValueError("individual Hero board requires one valid pull")
        subtitle = f"PULL {page_number} OF {page_count}  •  CURRENT HERO RANK"
    else:
        subtitle = "CURRENT HERO RANK"
        if page_count > 1:
            subtitle += f"  •  PAGE {page_number} OF {page_count}"
    draw.text(
        (40, 64),
        subtitle,
        font=_font(_SANS, 16),
        fill=(153, 177, 195, 255),
    )
    if layout == "individual":
        width, height, gap, y = 269, 430, 0, 108
    else:
        width, height, gap, y = 174, 278, 18, 123
    total = width * len(cards) + gap * max(0, len(cards) - 1)
    x = (BOARD_SIZE[0] - total) // 2
    for card in cards:
        thumbnail = card.convert("RGBa").resize(
            (width, height),
            Image.Resampling.LANCZOS,
        ).convert("RGBA")
        _place_with_shadow(board, thumbnail, (x, y))
        x += width + gap
    return board.convert("RGB")


def _place_with_shadow(
    canvas: Image.Image,
    item: Image.Image,
    position: tuple[int, int],
) -> None:
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_shape = Image.new("RGBA", item.size, (0, 0, 0, 185))
    shadow_shape.putalpha(item.getchannel("A"))
    shadow.alpha_composite(
        shadow_shape,
        (position[0] + 14, position[1] + 16),
    )
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
    canvas.alpha_composite(item, position)


def _png_bytes(image: Image.Image) -> bytes:
    encoded = BytesIO()
    image.convert("RGB").save(encoded, format="PNG", optimize=False)
    return encoded.getvalue()
