from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.engine.visual_novel_presentation import (
    CARD_HEIGHT,
    CARD_WIDTH,
    VisualNovelCardRenderer,
    VisualNovelDeckSection,
)
from app.schemas.narrator import VisualNovelPage


def _stage(path: Path) -> Path:
    image = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (41, 83, 127))
    image.putpixel((5, 5), (201, 37, 59))
    image.save(path)
    return path


def test_classic_adv_renderer_preserves_stage_above_overlay(tmp_path: Path):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    deck = renderer.render_deck(
        [VisualNovelDeckSection(
            pages=(VisualNovelPage(
                kind="dialogue",
                speaker="Iselle",
                text="You made it. I was beginning to wonder.",
            ),),
            stage_path=_stage(tmp_path / "stage.png"),
        )],
    )

    assert len(deck.cards) == 1
    assert deck.used_neutral_stage is False
    with Image.open(deck.cards[0].image_path) as card:
        assert card.size == (CARD_WIDTH, CARD_HEIGHT)
        assert card.convert("RGB").getpixel((5, 5)) == (201, 37, 59)
        # The selected A treatment is a dark navy lower panel, not baked text
        # or a model-reinterpreted background.
        red, green, blue = card.convert("RGB").getpixel((24, 400))
        assert blue > red
        assert blue > green


def test_long_semantic_page_splits_into_measured_cards(tmp_path: Path):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    deck = renderer.render_deck(
        [VisualNovelDeckSection(pages=(
            VisualNovelPage(
                kind="dialogue",
                speaker="Wren",
                text=" ".join(["carefully measured dialogue"] * 45),
            ),
        ))],
    )

    assert len(deck.cards) > 1
    assert all(card.speaker == "Wren" for card in deck.cards)
    assert [card.index for card in deck.cards] == list(
        range(1, len(deck.cards) + 1)
    )
    assert {card.count for card in deck.cards} == {len(deck.cards)}
    assert deck.used_neutral_stage is True


def test_content_addressed_deck_reloads_same_artifacts(tmp_path: Path):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    pages = [
        VisualNovelPage(
            kind="narration",
            text="Wind moves across the open beginner court.",
        )
    ]

    sections = [VisualNovelDeckSection(pages=tuple(pages))]
    first = renderer.render_deck(sections)
    second = renderer.render_deck(sections)
    loaded = renderer.load_deck(first.deck_id)

    assert second.deck_id == first.deck_id
    assert second.cards[0].image_path == first.cards[0].image_path
    assert loaded == first


def test_corrupt_stage_fails_safe_to_neutral_card(tmp_path: Path):
    bad_stage = tmp_path / "broken.webp"
    bad_stage.write_text("not an image", encoding="utf-8")
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck(
        [VisualNovelDeckSection(
            pages=(VisualNovelPage(
                kind="narration",
                text="You step outside.",
            ),),
            stage_path=bad_stage,
        )],
    )

    assert deck.used_neutral_stage is True
    assert deck.cards[0].image_path.is_file()


def test_ordered_sections_keep_each_paginated_beat_on_its_stage(
    tmp_path: Path,
):
    first_stage = _stage(tmp_path / "first.png")
    second_stage = Image.new(
        "RGB", (CARD_WIDTH, CARD_HEIGHT), (17, 29, 43)
    )
    second_stage.putpixel((5, 5), (13, 211, 97))
    second_stage_path = tmp_path / "second.png"
    second_stage.save(second_stage_path)
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck([
        VisualNovelDeckSection(
            pages=(VisualNovelPage(
                kind="dialogue",
                speaker="Iselle",
                text=" ".join(["The first beat stays here."] * 45),
            ),),
            stage_path=first_stage,
        ),
        VisualNovelDeckSection(
            pages=(VisualNovelPage(
                kind="dialogue",
                speaker="Wren",
                text="The second beat moves outside.",
            ),),
            stage_path=second_stage_path,
        ),
    ])

    iselle_cards = [card for card in deck.cards if card.speaker == "Iselle"]
    wren_cards = [card for card in deck.cards if card.speaker == "Wren"]
    assert len(iselle_cards) > 1
    assert len(wren_cards) == 1
    for card in iselle_cards:
        with Image.open(card.image_path) as image:
            assert image.convert("RGB").getpixel((5, 5)) == (201, 37, 59)
    with Image.open(wren_cards[0].image_path) as image:
        assert image.convert("RGB").getpixel((5, 5)) == (13, 211, 97)
    assert [card.index for card in deck.cards] == list(
        range(1, len(deck.cards) + 1)
    )
    assert {card.count for card in deck.cards} == {len(deck.cards)}
    assert deck.transcript.index("Iselle:") < deck.transcript.index("Wren:")
    assert deck.used_neutral_stage is False

    reloaded = VisualNovelCardRenderer(
        tmp_path / "presentations"
    ).load_deck(deck.deck_id)
    assert reloaded == deck


def test_section_order_and_neutral_fallback_are_part_of_deck_identity(
    tmp_path: Path,
):
    stage = _stage(tmp_path / "stage.png")
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    staged = VisualNovelDeckSection(
        pages=(VisualNovelPage(kind="narration", text="First."),),
        stage_path=stage,
    )
    neutral = VisualNovelDeckSection(
        pages=(VisualNovelPage(kind="narration", text="Second."),),
    )

    forward = renderer.render_deck([staged, neutral])
    reverse = renderer.render_deck([neutral, staged])

    assert forward.deck_id != reverse.deck_id
    assert forward.used_neutral_stage is True
    with Image.open(forward.cards[0].image_path) as image:
        assert image.convert("RGB").getpixel((5, 5)) == (201, 37, 59)
