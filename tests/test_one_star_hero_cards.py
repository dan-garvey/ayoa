from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from PIL import Image, ImageDraw
import pytest

import app.engine.one_star_hero_cards as hero_cards
from app.bot.engine_bridge import EngineBridge
from app.engine.one_star_adapter import (
    load_one_star_account,
    load_one_star_hero,
)
from app.engine.one_star_hero_cards import (
    OneStarHeroCardError,
    OneStarHeroCardBoard,
    OneStarHeroCardEvent,
    committed_one_star_hero_card_event,
    generated_portrait_prewarm_character_ids,
    new_one_star_hero_card_events,
    one_star_hero_card_events_for_render,
    render_one_star_hero_card,
    render_one_star_hero_card_boards,
)
from app.engine.one_star_visuals import sprite_set_id_for_viewer
from app.engine.player_media import ResolvedPlayerMedia
from app.engine.visual_novel_presentation import VisualNovelDeckSection
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import SpawnRequest
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    OneStarEventRouterOutput,
)
from app.schemas.responses import (
    VisualNovelRender,
    VisualNovelRenderSegment,
    TurnResponse,
)
from tests.support.factories import checkpoint as generic_checkpoint
from tests.support.factories import router_output


_ROOT = Path(__file__).resolve().parent.parent
_STORY = _ROOT / "app/storage/stories/one_star_ascension_s1"
_FRAME = (
    _STORY
    / "visual-references/system-panels/"
    "one_star_hero_card_frame_obsidian_orrery_v1.png"
)
_RENNA_PORTRAIT = (
    _STORY / "visual-references/hero-card-portraits/renna_holt_v1.png"
)


def _seed() -> CheckpointFile:
    return CheckpointFile.model_validate_json(
        (_STORY / "ckpt_0000.json").read_text(encoding="utf-8")
    )


def _character(checkpoint: CheckpointFile, character_id: str) -> CharacterRecord:
    return next(
        character
        for character in checkpoint.characters
        if character.character_id == character_id
    )


def _set_stars(character: CharacterRecord, stars: int) -> None:
    hero = load_one_star_hero(character)
    assert hero is not None
    hero.current_stars = stars
    character.mechanics[ONE_STAR_HERO_KEY] = hero.model_dump(mode="json")


def _commit_fingerprint(checkpoint: CheckpointFile, event_id: str) -> None:
    owner, account = load_one_star_account(checkpoint)
    account.state.applied_event_fingerprints[event_id] = "committed-test-hash"
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")


def _commit_summon(
    checkpoint: CheckpointFile,
    *,
    event_id: str,
    character_ids: list[str],
) -> OneStarEventRouterOutput:
    for character_id in character_ids:
        character = _character(checkpoint, character_id)
        hero = load_one_star_hero(character)
        assert hero is not None
        hero.acquisition_event_id = event_id
        hero.owner_lobby_id = "niflheim_lobby"
        character.mechanics[ONE_STAR_HERO_KEY] = hero.model_dump(mode="json")
    payload = router_output(
        event_id=event_id,
        event_kind="state_change",
        observer_ids=["the_master", *character_ids],
        activate=[
            {"character_id": character_id, "location_label": "Niflheim Lobby"}
            for character_id in character_ids
        ],
    ).model_dump(mode="json")
    payload["state_updates"] = [{
        "kind": "summon",
        "target_id": "basic_summon",
        "value": str(len(character_ids)),
        "details": [],
    }]
    event = OneStarEventRouterOutput.model_validate(payload)
    checkpoint.canonical_events.append(event)
    _commit_fingerprint(checkpoint, event_id)
    return event


def _commit_mission_start(
    checkpoint: CheckpointFile,
    *,
    event_id: str,
    party_ids: list[str],
) -> OneStarEventRouterOutput:
    payload = router_output(
        event_id=event_id,
        event_kind="state_change",
        observer_ids=["the_master", *party_ids],
    ).model_dump(mode="json")
    payload["state_updates"] = [{
        "kind": "mission_start",
        "target_id": "tower_floor_1",
        "value": "1",
        "details": [
            "pending_operation_id=deployment_1",
            *(f"party={character_id}" for character_id in party_ids),
            "destination=Tower Floor 1",
            "completion=the floor is cleared",
            "failure=the party is broken",
        ],
    }]
    event = OneStarEventRouterOutput.model_validate(payload)
    checkpoint.canonical_events.append(event)
    _commit_fingerprint(checkpoint, event_id)
    return event


def _render(*event_ids: str) -> VisualNovelRender:
    return VisualNovelRender(segments=[
        VisualNovelRenderSegment(
            pages=[{"kind": "narration", "text": "A committed event is rendered."}],
            rendered_event_id=event_id,
        )
        for event_id in event_ids
    ])


def _media(image: Image.Image, *, filename: str = "portrait.png") -> ResolvedPlayerMedia:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False)
    data = stream.getvalue()
    return ResolvedPlayerMedia(
        filename=filename,
        mime_type="image/png",
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        width=image.width,
        height=image.height,
    )


def _file_media(path: Path) -> ResolvedPlayerMedia:
    with Image.open(path) as opened:
        width, height = opened.size
    data = path.read_bytes()
    return ResolvedPlayerMedia(
        filename=path.name,
        mime_type="image/png",
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        width=width,
        height=height,
    )


def _connected_components(mask: Image.Image) -> int:
    pixels = mask.load()
    width, height = mask.size
    remaining = {
        (x, y)
        for y in range(height)
        for x in range(width)
        if pixels[x, y]
    }
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            x, y = stack.pop()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


class _FakeStore:
    def __init__(self, *, unavailable_ids: set[str] | None = None) -> None:
        self.unavailable_ids = unavailable_ids or set()

    def reviewed_reference(self, *, session_id: str, reference_id: str):
        del session_id
        if reference_id in self.unavailable_ids:
            return None
        return SimpleNamespace(reference_id=reference_id)


class _FakeGeneration:
    def __init__(
        self,
        *,
        generated_media: ResolvedPlayerMedia | None = None,
        unavailable_ids: set[str] | None = None,
    ) -> None:
        self.store = _FakeStore(unavailable_ids=unavailable_ids)
        self.config = SimpleNamespace(runtime_root=Path("/unused"))
        self.generated_media = generated_media

    def resolve_visual_novel_sprite_variant(self, **_kwargs):
        if self.generated_media is None:
            return None
        return "generated-neutral", self.generated_media, "right"


def test_summon_events_require_commit_and_master_render_and_deduplicate() -> None:
    previous = _seed()
    checkpoint = previous.model_copy(deep=True)
    summon_event = _commit_summon(
        checkpoint,
        event_id="evt_summon_cards",
        character_ids=["halcyon_of_the_gilded_march", "renna_holt"],
    )
    summon_event.spawn.append(SpawnRequest.model_validate({
        "character_id": "halcyon_of_the_gilded_march",
        "seed": {
            "role": "summoned Hero",
            "reason": "exercise spawn and activate deduplication",
            "location": "Niflheim Lobby",
            "objectives": [],
            "knowledge_tier": 3,
        },
    }))

    event = committed_one_star_hero_card_event(
        checkpoint,
        "evt_summon_cards",
    )
    assert event is not None
    assert [character.character_id for character in event.characters] == [
        "halcyon_of_the_gilded_march",
        "renna_holt",
    ]
    assert [item.event_id for item in new_one_star_hero_card_events(
        checkpoint,
        previous,
    )] == ["evt_summon_cards"]
    selected = one_star_hero_card_events_for_render(
        checkpoint=checkpoint,
        previous_checkpoint=previous,
        viewer_character_id="the_master",
        render=_render("evt_summon_cards", "evt_summon_cards"),
    )
    assert [item.event_id for item in selected] == ["evt_summon_cards"]
    assert one_star_hero_card_events_for_render(
        checkpoint=checkpoint,
        previous_checkpoint=previous,
        viewer_character_id="renna_holt",
        render=_render("evt_summon_cards"),
    ) == ()
    with pytest.raises(
        OneStarHeroCardError,
        match="master_render_missing_card_event",
    ):
        one_star_hero_card_events_for_render(
            checkpoint=checkpoint,
            previous_checkpoint=previous,
            viewer_character_id="the_master",
            render=_render("evt_unrelated"),
        )

    assert new_one_star_hero_card_events(
        checkpoint,
        checkpoint.model_copy(deep=True),
    ) == ()
    owner, account = load_one_star_account(checkpoint)
    account.state.applied_event_fingerprints.pop("evt_summon_cards")
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    assert committed_one_star_hero_card_event(
        checkpoint,
        "evt_summon_cards",
    ) is None


def test_non_one_star_checkpoints_never_emit_card_events() -> None:
    checkpoint = generic_checkpoint()

    assert committed_one_star_hero_card_event(checkpoint, "evt_any") is None
    assert new_one_star_hero_card_events(checkpoint, None) == ()


def test_seed_promotes_only_reviewed_frame_and_approved_bust_overrides() -> None:
    checkpoint = _seed()
    _owner, account = load_one_star_account(checkpoint)
    presentation = account.config.visual_novel_presentation
    assert presentation is not None
    assert presentation.hero_card_frame_reference_id == (
        "osa_hero_card_frame_obsidian_orrery_v1"
    )
    references = {
        reference.reference_id: reference
        for reference in checkpoint.reviewed_visual_references
    }
    frame = references[presentation.hero_card_frame_reference_id]
    assert (frame.purpose, frame.scope, frame.scope_id) == (
        "presentation",
        "presentation",
        "one_star_hero_cards",
    )
    sprite_sets = {
        sprite_set.sprite_set_id: sprite_set
        for sprite_set in checkpoint.reviewed_visual_novel_sprite_sets
    }
    assert {
        sprite_set_id: sprite_sets[sprite_set_id].portrait_reference_id
        for sprite_set_id in (
            "osa_vnset_renna_holt_v1",
            "osa_vnset_halcyon_of_the_gilded_march_v1",
            "osa_vnset_veiled_feminine_v1",
            "osa_vnset_veiled_masculine_v1",
        )
    } == {
        "osa_vnset_renna_holt_v1": (
            "osa_hero_card_portrait_renna_holt_v1"
        ),
        "osa_vnset_halcyon_of_the_gilded_march_v1": (
            "osa_hero_card_portrait_halcyon_v1"
        ),
        "osa_vnset_veiled_feminine_v1": (
            "osa_hero_card_portrait_veiled_feminine_v1"
        ),
        "osa_vnset_veiled_masculine_v1": (
            "osa_hero_card_portrait_veiled_masculine_v1"
        ),
    }
    assert (
        sprite_sets["osa_vnset_warden_of_the_eighth_v1"].portrait_reference_id
        == ""
    )
    assert all(
        references[sprite_set.portrait_reference_id].diffusion_authorized is False
        for sprite_set in sprite_sets.values()
        if sprite_set.portrait_reference_id
    )


def test_formation_card_preserves_validated_party_order() -> None:
    checkpoint = _seed()
    _commit_mission_start(
        checkpoint,
        event_id="evt_formation_cards",
        party_ids=["renna_holt", "halcyon_of_the_gilded_march"],
    )

    event = committed_one_star_hero_card_event(
        checkpoint,
        "evt_formation_cards",
    )
    assert event is not None
    assert event.kind == "mission_start"
    assert [character.character_id for character in event.characters] == [
        "renna_holt",
        "halcyon_of_the_gilded_march",
    ]
    assert committed_one_star_hero_card_event(
        checkpoint,
        "evt_not_present",
    ) is None


def test_option_f_draws_exact_star_counts_and_distinct_rank_pixels() -> None:
    layer_hashes: list[str] = []
    for stars in range(1, 8):
        layer = Image.new("RGBA", hero_cards.CARD_SIZE, (0, 0, 0, 0))
        hero_cards._draw_rank_stars(
            layer,
            hero_cards._STYLE_BY_STARS[stars],
        )
        solid = layer.getchannel("A").point(
            lambda alpha: 255 if alpha >= 200 else 0
        )
        assert _connected_components(solid) == stars
        layer_hashes.append(hashlib.sha256(layer.tobytes()).hexdigest())
    assert len(set(layer_hashes)) == 7

    with Image.open(_FRAME) as opened:
        frame = opened.convert("RGBA")
    portrait = _file_media(_RENNA_PORTRAIT)
    cards = [
        render_one_star_hero_card(
            frame=frame,
            portrait=portrait,
            portrait_source="override",
            display_name="Renna Holt",
            stars=stars,
        )
        for stars in range(1, 8)
    ]
    assert len({hashlib.sha256(card.tobytes()).hexdigest() for card in cards}) == 7


def test_unicode_name_fits_without_truncation_and_bust_layering_is_correct() -> None:
    display_name = "Æthelred-Snowfall of Niflheim — 長い英雄名"
    scratch = Image.new("RGBA", hero_cards.CARD_SIZE)
    lines, _face = hero_cards._name_lines(
        ImageDraw.Draw(scratch),
        display_name,
        hero_cards.NAME_BOX[2] - hero_cards.NAME_BOX[0],
        hero_cards.NAME_BOX[3] - hero_cards.NAME_BOX[1],
    )
    assert 1 <= len(lines) <= 2
    assert "".join(lines).replace(" ", "") == display_name.replace(" ", "")

    frame = Image.new("RGBA", hero_cards.CARD_SIZE, (220, 30, 30, 255))
    portrait = _media(Image.new("RGBA", (500, 707), (0, 0, 255, 255)))
    card = render_one_star_hero_card(
        frame=frame,
        portrait=portrait,
        portrait_source="override",
        display_name=display_name,
        stars=3,
    ).convert("RGB")
    # Bust covers the side border in the portrait field.
    assert card.getpixel((90, 500))[2] > card.getpixel((90, 500))[0]
    # The lower nameplate returns above the bust.
    assert card.getpixel((90, 900))[2] < card.getpixel((90, 900))[0]
    # The center star is above the bust as well.
    center = card.getpixel((320, 816))
    assert center[0] > 100


def test_nonhuman_neutral_silhouette_is_preserved_as_portrait_media() -> None:
    creature = Image.new("RGBA", (500, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(creature)
    draw.polygon(
        ((10, 170), (130, 70), (330, 90), (490, 15), (420, 180), (210, 260)),
        fill=(24, 210, 231, 255),
    )
    draw.ellipse((85, 110, 430, 290), fill=(24, 210, 231, 255))

    layer = hero_cards._portrait_layer(_media(creature), "neutral")
    cyan_pixels = sum(
        1
        for red, green, blue, alpha in layer.getdata()
        if alpha > 200 and blue > 180 and green > 170 and red < 80
    )

    assert cyan_pixels > 20_000


def test_visible_generated_summon_prewarm_is_exact_and_summon_only() -> None:
    checkpoint = _seed()
    template = _character(checkpoint, "edren_marr")
    generated = template.model_copy(deep=True)
    generated.character_id = "fresh_visible_generated_hero"
    generated.name = "Fresh Visible Generated Hero"
    generated.status = "active"
    generated.visuals.sprite_set_id = ""
    generated.mechanics[ONE_STAR_HERO_KEY]["generated_for_summon"] = True
    generated.mechanics[ONE_STAR_HERO_KEY]["current_stars"] = 3
    checkpoint.characters.append(generated)
    summon = OneStarHeroCardEvent(
        event_id="evt_generated_summon",
        kind="summon",
        characters=(generated,),
    )

    assert generated_portrait_prewarm_character_ids(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        events=(summon,),
    ) == (generated.character_id,)

    generated.mechanics[ONE_STAR_HERO_KEY]["current_stars"] = 1
    assert generated_portrait_prewarm_character_ids(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        events=(summon,),
    ) == ()

    generated.mechanics[ONE_STAR_HERO_KEY]["current_stars"] = 3
    formation = OneStarHeroCardEvent(
        event_id="evt_generated_formation",
        kind="mission_start",
        characters=(generated,),
    )
    assert generated_portrait_prewarm_character_ids(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        events=(formation,),
    ) == ()


def test_portrait_override_neutral_generated_placeholder_and_veiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _seed()
    renna = _character(checkpoint, "renna_holt")
    _set_stars(renna, 2)
    override_media = _media(Image.new("RGBA", (80, 120), (220, 20, 20, 255)))
    neutral_media = _media(Image.new("RGBA", (80, 120), (20, 220, 20, 255)))
    veiled_media = _media(Image.new("RGBA", (80, 120), (20, 20, 220, 255)))
    generated_media = _media(Image.new("RGBA", (80, 120), (220, 190, 20, 255)))

    sets = {
        sprite_set.sprite_set_id: sprite_set
        for sprite_set in checkpoint.reviewed_visual_novel_sprite_sets
    }
    renna_set = sets[renna.visuals.sprite_set_id]
    media_by_id = {
        renna_set.portrait_reference_id: override_media,
        renna_set.variant_reference_ids["neutral"]: neutral_media,
    }
    for sprite_set in sets.values():
        if not sprite_set.owner_character_id and sprite_set.portrait_reference_id:
            media_by_id[sprite_set.portrait_reference_id] = veiled_media
        media_by_id.setdefault(
            sprite_set.variant_reference_ids["neutral"],
            neutral_media,
        )

    def resolve(reference, *, runtime_root):
        del runtime_root
        return media_by_id[reference.reference_id]

    monkeypatch.setattr(hero_cards, "resolve_frozen_visual_reference_media", resolve)
    generation = _FakeGeneration(generated_media=generated_media)

    portrait = hero_cards._resolve_portrait(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        character=renna,
        generation=generation,
    )
    assert portrait.source == "override"
    assert portrait.media is override_media

    renna_set.portrait_reference_id = ""
    portrait = hero_cards._resolve_portrait(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        character=renna,
        generation=generation,
    )
    assert portrait.source == "neutral"
    assert portrait.media is neutral_media

    renna_set.portrait_reference_id = "osa_hero_card_portrait_renna_holt_v1"
    _set_stars(renna, 1)
    selected_veiled_set = sets[sprite_set_id_for_viewer(
        checkpoint,
        viewer_character_id="the_master",
        character=renna,
    )]
    portrait = hero_cards._resolve_portrait(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        character=renna,
        generation=generation,
    )
    assert selected_veiled_set.owner_character_id == ""
    assert portrait.media is veiled_media
    assert portrait.media is not override_media

    generated = renna.model_copy(deep=True)
    generated.character_id = "fresh_generated_hero"
    generated.name = "Fresh Generated Hero"
    generated.visuals.sprite_set_id = ""
    _set_stars(generated, 3)
    checkpoint.characters.append(generated)
    portrait = hero_cards._resolve_portrait(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        character=generated,
        generation=generation,
    )
    assert portrait.media is generated_media

    unavailable = _FakeGeneration()
    portrait = hero_cards._resolve_portrait(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        character=generated,
        generation=unavailable,
    )
    assert portrait.source == "unavailable"
    assert portrait.media is None


def test_boards_paginate_five_are_deterministic_and_source_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _seed()
    _owner, account = load_one_star_account(checkpoint)
    presentation = account.config.visual_novel_presentation
    assert presentation is not None
    frame_id = presentation.hero_card_frame_reference_id
    frame_media = _file_media(_FRAME)
    portrait_media = _media(Image.new("RGBA", (280, 430), (63, 91, 127, 255)))

    def resolve(reference, *, runtime_root):
        del runtime_root
        return frame_media if reference.reference_id == frame_id else portrait_media

    monkeypatch.setattr(hero_cards, "resolve_frozen_visual_reference_media", resolve)
    generation = _FakeGeneration(generated_media=portrait_media)
    heroes = tuple(
        character
        for character in checkpoint.characters
        if load_one_star_hero(character) is not None
    )[:6]
    event = OneStarHeroCardEvent(
        event_id="evt_six_heroes",
        kind="summon",
        characters=heroes,
    )

    first = render_one_star_hero_card_boards(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        event=event,
        generation=generation,
    )
    second = render_one_star_hero_card_boards(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        event=event,
        generation=generation,
    )
    assert len(first) == 2
    assert [board.media.data for board in first] == [
        board.media.data for board in second
    ]
    assert [board.media.sha256 for board in first] == [
        board.media.sha256 for board in second
    ]
    assert all((board.media.width, board.media.height) == (1024, 576) for board in first)
    assert "page 1 of 2" in first[0].accessible_text
    assert "page 2 of 2" in first[1].accessible_text
    assert sum(character.name in first[0].accessible_text for character in heroes) == 5
    assert sum(character.name in first[1].accessible_text for character in heroes) == 1
    assert all(b"osa_" not in board.media.data for board in first)
    assert all("osa_" not in board.accessible_text for board in first)
    for board in first:
        with Image.open(BytesIO(board.media.data)) as opened:
            assert opened.size == (1024, 576)
            assert opened.info == {}

    assert render_one_star_hero_card_boards(
        checkpoint=checkpoint,
        viewer_character_id="renna_holt",
        event=event,
        generation=generation,
    ) == ()


def test_board_uses_current_rank_after_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _seed()
    hero = _character(checkpoint, "renna_holt")
    _owner, account = load_one_star_account(checkpoint)
    presentation = account.config.visual_novel_presentation
    assert presentation is not None
    frame_id = presentation.hero_card_frame_reference_id
    frame_media = _file_media(_FRAME)
    portrait_media = _file_media(_RENNA_PORTRAIT)

    def resolve(reference, *, runtime_root):
        del runtime_root
        return frame_media if reference.reference_id == frame_id else portrait_media

    monkeypatch.setattr(hero_cards, "resolve_frozen_visual_reference_media", resolve)
    generation = _FakeGeneration(generated_media=portrait_media)
    event = OneStarHeroCardEvent(
        event_id="evt_rank_change",
        kind="summon",
        characters=(hero,),
    )
    _set_stars(hero, 2)
    before = render_one_star_hero_card_boards(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        event=event,
        generation=generation,
    )[0]
    _set_stars(hero, 6)
    after = render_one_star_hero_card_boards(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        event=event,
        generation=generation,
    )[0]

    assert "Renna Holt — 2 stars" in before.accessible_text
    assert "Renna Holt — 6 stars" in after.accessible_text
    assert before.media.sha256 != after.media.sha256


@pytest.mark.asyncio
async def test_bridge_inserts_one_board_immediately_after_matching_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _seed()
    hero = _character(checkpoint, "renna_holt")
    event = OneStarHeroCardEvent(
        event_id="evt_ordered_board",
        kind="summon",
        characters=(hero,),
    )
    panel_media = _media(
        Image.new("RGB", (1024, 576), (11, 27, 49)),
        filename="one-star-board.png",
    )
    board = OneStarHeroCardBoard(
        media=panel_media,
        accessible_text="System panel — Heroes acquired: Renna Holt — 1 star",
        event_id=event.event_id,
        kind="summon",
        page_number=1,
        page_count=1,
    )
    bridge = EngineBridge.__new__(EngineBridge)
    bridge.load_checkpoint = MagicMock(return_value=checkpoint)  # type: ignore[method-assign]
    bridge._previous_visual_novel_checkpoint = MagicMock(  # type: ignore[method-assign]
        return_value=None
    )
    bridge._prewarm_visual_novel_sprites = AsyncMock()  # type: ignore[method-assign]
    bridge.wait_for_visual_novel_stage_work = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    bridge.image_generation = MagicMock()
    bridge.image_generation.resolve_visual_novel_stage.return_value = (
        SimpleNamespace(fallback_reason=""),
        panel_media,
    )
    bridge.visual_novel_renderer = MagicMock()
    bridge.visual_novel_renderer.render_deck.side_effect = tuple
    monkeypatch.setattr(
        "app.bot.engine_bridge.one_star_hero_card_events_for_render",
        lambda **_kwargs: (event,),
    )
    monkeypatch.setattr(
        "app.bot.engine_bridge.generated_portrait_prewarm_character_ids",
        lambda **_kwargs: (),
    )
    render_boards = MagicMock(return_value=(board,))
    monkeypatch.setattr(
        "app.bot.engine_bridge.render_one_star_hero_card_boards",
        render_boards,
    )
    monkeypatch.setattr(
        "app.bot.engine_bridge.resolve_visual_novel_sprite_placements",
        lambda **_kwargs: (),
    )
    render = VisualNovelRender(segments=[
        VisualNovelRenderSegment(
            pages=[{"kind": "narration", "text": "The summons arrive."}],
            rendered_event_id=event.event_id,
        ),
        VisualNovelRenderSegment(
            pages=[{"kind": "narration", "text": "The light settles."}],
            rendered_event_id=event.event_id,
        ),
    ])

    sections = await bridge.prepare_visual_novel_deck(
        session_id=checkpoint.session.session_id,
        checkpoint_id="ckpt_0001",
        pov_character_id="the_master",
        render=render,
    )

    assert isinstance(sections, tuple)
    assert all(isinstance(section, VisualNovelDeckSection) for section in sections)
    assert [section.pages[0].text for section in sections] == [
        "The summons arrive.",
        board.accessible_text,
        "The light settles.",
    ]
    assert [section.card_style for section in sections] == [
        "adv",
        "system_panel",
        "adv",
    ]
    assert sections[1].stage_media is panel_media
    render_boards.assert_called_once()


@pytest.mark.asyncio
async def test_required_generated_prewarm_waits_only_for_exact_neutral_job() -> None:
    checkpoint = _seed()
    bridge = EngineBridge.__new__(EngineBridge)
    bridge.image_generation = MagicMock()
    bridge.image_generation.ensure_visual_novel_sprite_prewarm = AsyncMock(
        return_value=("job-required-neutral", "job-required-happy", "job-other")
    )
    jobs = {
        "job-required-neutral": SimpleNamespace(request=SimpleNamespace(
            character_id="fresh_hero",
            sprite_variant_key="neutral",
        )),
        "job-required-happy": SimpleNamespace(request=SimpleNamespace(
            character_id="fresh_hero",
            sprite_variant_key="happy",
        )),
        "job-other": SimpleNamespace(request=SimpleNamespace(
            character_id="other_hero",
            sprite_variant_key="neutral",
        )),
    }
    bridge.image_generation.store.get.side_effect = jobs.get
    bridge.image_generation.wait_for_terminal = AsyncMock(
        return_value=SimpleNamespace(status="failed")
    )

    await bridge._prewarm_visual_novel_sprites(
        session_id=checkpoint.session.session_id,
        checkpoint=checkpoint,
        required_visible_character_ids=("fresh_hero",),
        await_required=True,
    )

    bridge.image_generation.ensure_visual_novel_sprite_prewarm.assert_awaited_once_with(
        checkpoint,
        required_visible_character_ids=("fresh_hero",),
    )
    bridge.image_generation.wait_for_terminal.assert_awaited_once_with(
        "job-required-neutral"
    )


def test_bridge_fails_loudly_when_master_card_segment_is_omitted() -> None:
    previous = _seed()
    checkpoint = previous.model_copy(deep=True)
    checkpoint.session.session_id = "session"
    previous.session.session_id = "session"
    _commit_summon(
        checkpoint,
        event_id="evt_required_master_card",
        character_ids=["renna_holt"],
    )
    bridge = EngineBridge.__new__(EngineBridge)
    bridge.load_checkpoint = MagicMock(return_value=checkpoint)  # type: ignore[method-assign]
    bridge._previous_visual_novel_checkpoint = MagicMock(  # type: ignore[method-assign]
        return_value=previous
    )
    response = TurnResponse(
        session_id="session",
        checkpoint_id="ckpt_0001",
        per_player_visual_novel_renders={},
    )

    with pytest.raises(
        OneStarHeroCardError,
        match="master_render_missing_card_event",
    ):
        bridge._validate_one_star_hero_card_routing(response)

    response.per_player_visual_novel_renders["the_master"] = _render(
        "evt_required_master_card"
    )
    bridge._validate_one_star_hero_card_routing(response)
