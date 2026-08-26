from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.engine.visual_novel_presentation import (
    CARD_HEIGHT,
    CARD_WIDTH,
    VisualNovelCardRenderer,
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
        [
            VisualNovelPage(
                kind="dialogue",
                speaker="Iselle",
                text="You made it. I was beginning to wonder.",
            )
        ],
        stage_path=_stage(tmp_path / "stage.png"),
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
        [
            VisualNovelPage(
                kind="dialogue",
                speaker="Wren",
                text=" ".join(["carefully measured dialogue"] * 45),
            )
        ],
        stage_path=None,
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

    first = renderer.render_deck(pages, stage_path=None)
    second = renderer.render_deck(pages, stage_path=None)
    loaded = renderer.load_deck(first.deck_id)

    assert second.deck_id == first.deck_id
    assert second.cards[0].image_path == first.cards[0].image_path
    assert loaded == first


def test_corrupt_stage_fails_safe_to_neutral_card(tmp_path: Path):
    bad_stage = tmp_path / "broken.webp"
    bad_stage.write_text("not an image", encoding="utf-8")
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck(
        [VisualNovelPage(kind="narration", text="You step outside.")],
        stage_path=bad_stage,
    )

    assert deck.used_neutral_stage is True
    assert deck.cards[0].image_path.is_file()
