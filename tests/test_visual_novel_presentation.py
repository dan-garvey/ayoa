from __future__ import annotations

from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw
import pytest

from app.engine.visual_novel_presentation import (
    CARD_HEIGHT,
    CARD_WIDTH,
    _SPEAKER_NAME_MAX_WIDTH,
    _ellipsize,
    _text_width,
    VisualNovelCardRenderer,
    VisualNovelDeckSection,
    VisualNovelSpriteError,
    VisualNovelSpritePlacement,
)
from app.engine.player_media import ResolvedPlayerMedia
from app.schemas.narrator import VisualNovelPage


def _stage(path: Path) -> Path:
    image = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (41, 83, 127))
    image.putpixel((5, 5), (201, 37, 59))
    image.save(path)
    return path


def _sprite_media(
    *,
    color: tuple[int, int, int, int] = (219, 43, 61, 255),
    marker_color: tuple[int, int, int, int] = (247, 239, 91, 255),
    width: int = 80,
    height: int = CARD_HEIGHT,
    edge_alpha: int = 128,
) -> ResolvedPlayerMedia:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    body_top = 80
    for y in range(body_top, height):
        for x in range(8, width - 8):
            alpha = edge_alpha if x == 8 else color[3]
            image.putpixel((x, y), (*color[:3], alpha))
    for y in range(100, 125):
        for x in range(12, 18):
            image.putpixel((x, y), marker_color)
    return _resolved_png(image)


def _resolved_png(image: Image.Image) -> ResolvedPlayerMedia:
    encoded = BytesIO()
    image.save(encoded, format="PNG", optimize=False)
    data = encoded.getvalue()
    sha256 = hashlib.sha256(data).hexdigest()
    return ResolvedPlayerMedia(
        filename=f"sprite-{sha256[:16]}.png",
        mime_type="image/png",
        data=data,
        sha256=sha256,
        byte_count=len(data),
        width=image.width,
        height=image.height,
    )


def _placement(
    *,
    subject_handle: str = "subject.alpha",
    identity_handle: str = "identity.alpha",
    variant_handle: str = "variant.neutral",
    media: object | None = None,
    slot: str = "left",
    source_facing: str = "right",
    facing: str = "right",
    anchor: tuple[int, int] = (240, CARD_HEIGHT),
    scale_percent: int = 100,
) -> VisualNovelSpritePlacement:
    return VisualNovelSpritePlacement(
        subject_handle=subject_handle,
        identity_handle=identity_handle,
        variant_handle=variant_handle,
        media=media if media is not None else _sprite_media(),  # type: ignore[arg-type]
        slot=slot,  # type: ignore[arg-type]
        source_facing=source_facing,  # type: ignore[arg-type]
        facing=facing,  # type: ignore[arg-type]
        anchor=anchor,
        scale_percent=scale_percent,
    )


def _manifest(deck) -> dict:
    return json.loads(deck.manifest_path.read_text(encoding="utf-8"))


def _write_manifest(deck, payload: object) -> None:
    deck.manifest_path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


def _v2_deck_id(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "identity": payload["identity"],
                "card_sha256s": [card["sha256"] for card in payload["cards"]],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _rehome_v2_deck(renderer, deck, payload: dict) -> str:
    new_deck_id = _v2_deck_id(payload)
    payload["deck_id"] = new_deck_id
    new_deck_dir = renderer.deck_root / new_deck_id
    deck.manifest_path.parent.rename(new_deck_dir)
    (new_deck_dir / "manifest.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return new_deck_id


def _single_page_deck(tmp_path: Path):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="narration",
                        text="Wind moves across the open beginner court.",
                    ),
                )
            ),
        ]
    )
    return renderer, deck


def test_system_panel_preserves_exact_bytes_text_hash_and_restart(
    tmp_path: Path,
) -> None:
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    image = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (7, 19, 37))
    image.putpixel((511, 287), (241, 197, 83))
    panel = replace(
        _resolved_png(image),
        filename="reviewed-private-source-id.png",
    )
    accessible = (
        "System panel — Heroes acquired: Renna Holt — 1 star; "
        "Halcyon — 6 stars"
    )

    deck = renderer.render_deck([
        VisualNovelDeckSection(
            pages=(VisualNovelPage(kind="narration", text=accessible),),
            stage_media=panel,
            card_style="system_panel",
        )
    ])

    assert deck.cards[0].image_bytes == panel.data
    assert deck.cards[0].accessible_text == accessible
    assert deck.transcript == accessible
    manifest = _manifest(deck)
    assert manifest["identity"]["sections"][0]["card_style"] == "system_panel"
    assert "reviewed-private-source-id" not in json.dumps(manifest)
    restarted = VisualNovelCardRenderer(tmp_path / "presentations").load_deck(
        deck.deck_id
    )
    assert restarted is not None
    assert restarted.cards[0].image_bytes == panel.data

    changed_text = renderer.render_deck([
        VisualNovelDeckSection(
            pages=(VisualNovelPage(
                kind="narration",
                text=accessible.replace("1 star", "2 stars"),
            ),),
            stage_media=panel,
            card_style="system_panel",
        )
    ])
    assert changed_text.deck_id != deck.deck_id


def test_system_panel_rejects_nonexact_or_annotated_sections(
    tmp_path: Path,
) -> None:
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    panel = _resolved_png(Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT)))

    with pytest.raises(ValueError, match="one media page"):
        renderer.render_deck([
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(kind="narration", text="First"),
                    VisualNovelPage(kind="narration", text="Second"),
                ),
                stage_media=panel,
                card_style="system_panel",
            )
        ])
    bad_panel = replace(panel, sha256="0" * 64)
    with pytest.raises(ValueError, match="exact 1024x576 PNG"):
        renderer.render_deck([
            VisualNovelDeckSection(
                pages=(VisualNovelPage(kind="narration", text="Panel"),),
                stage_media=bad_panel,
                card_style="system_panel",
            )
        ])

    wrong_size = _resolved_png(Image.new("RGB", (CARD_WIDTH - 1, CARD_HEIGHT)))
    with pytest.raises(ValueError, match="exact 1024x576 PNG"):
        renderer.render_deck([
            VisualNovelDeckSection(
                pages=(VisualNovelPage(kind="narration", text="Panel"),),
                stage_media=wrong_size,
                card_style="system_panel",
            )
        ])


def test_classic_adv_renderer_preserves_stage_above_overlay(tmp_path: Path):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Iselle",
                        text="You made it. I was beginning to wonder.",
                    ),
                ),
                stage_path=_stage(tmp_path / "stage.png"),
            )
        ],
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


def test_speaker_nameplate_keeps_ordinary_long_character_name(
    tmp_path: Path,
):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    fonts = renderer._fonts()
    draw = ImageDraw.Draw(Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT)))
    name = "the Warden of the Eighth"

    displayed = _ellipsize(
        draw,
        name,
        fonts.speaker,
        max_width=_SPEAKER_NAME_MAX_WIDTH,
    )

    assert displayed == name
    assert _text_width(draw, displayed, fonts.speaker) > 332
    assert _text_width(draw, displayed, fonts.speaker) <= (_SPEAKER_NAME_MAX_WIDTH)


def test_sprite_alpha_composites_after_exact_stage_and_before_dialogue_ui(
    tmp_path: Path,
):
    stage_path = _stage(tmp_path / "stage.png")
    original_stage_bytes = stage_path.read_bytes()
    sprite = _sprite_media()
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Mirelle",
                        text="Stay behind me.",
                    ),
                ),
                stage_path=stage_path,
                sprite_placements=(_placement(media=sprite),),
            )
        ]
    )

    assert stage_path.read_bytes() == original_stage_bytes
    assert deck.cards[0].accessible_text == "Mirelle: Stay behind me."
    with Image.open(deck.cards[0].image_path) as card:
        pixels = card.convert("RGB")
        # Nothing touched by either foreground or UI remains the exact plate.
        assert pixels.getpixel((5, 5)) == (201, 37, 59)
        # The half-alpha sprite edge blends with, rather than replaces, stage.
        expected_edge = (
            Image.alpha_composite(
                Image.new("RGBA", (1, 1), (41, 83, 127, 255)),
                Image.new("RGBA", (1, 1), (219, 43, 61, 128)),
            )
            .convert("RGB")
            .getpixel((0, 0))
        )
        assert pixels.getpixel((208, 150)) == expected_edge
        assert pixels.getpixel((240, 150)) == (219, 43, 61)
        # The ADV panel is composited last and safely occludes the lower body.
        assert pixels.getpixel((240, 450)) != (219, 43, 61)

    manifest = _manifest(deck)
    assert manifest["identity"]["sections"][0]["stage_sha256"] == (
        hashlib.sha256(original_stage_bytes).hexdigest()
    )


def test_two_sprite_slots_face_inward_without_transferring_identity(
    tmp_path: Path,
):
    left_media = _sprite_media(
        color=(211, 45, 64, 255),
        marker_color=(250, 238, 82, 255),
    )
    right_media = _sprite_media(
        color=(38, 97, 221, 255),
        marker_color=(69, 232, 139, 255),
    )
    left = _placement(
        subject_handle="subject.left",
        identity_handle="identity.left",
        variant_handle="variant.left.tense",
        media=left_media,
        slot="left",
        source_facing="right",
        facing="right",
        anchor=(240, CARD_HEIGHT),
    )
    right = _placement(
        subject_handle="subject.right",
        identity_handle="identity.right",
        variant_handle="variant.right.skeptical",
        media=right_media,
        slot="right",
        source_facing="right",
        facing="left",
        anchor=(784, CARD_HEIGHT),
    )
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="narration",
                        text="They measure one another across the court.",
                    ),
                ),
                stage_path=_stage(tmp_path / "stage.png"),
                sprite_placements=(left, right),
            )
        ]
    )

    with Image.open(deck.cards[0].image_path) as card:
        pixels = card.convert("RGB")
        assert pixels.getpixel((240, 150)) == (211, 45, 64)
        assert pixels.getpixel((784, 150)) == (38, 97, 221)
        # The left source remains right-facing; the right source is mirrored.
        assert pixels.getpixel((214, 110)) == (250, 238, 82)
        assert pixels.getpixel((808, 110)) == (69, 232, 139)

    sprites = _manifest(deck)["identity"]["sections"][0]["sprites"]
    assert [sprite["identity_handle"] for sprite in sprites] == [
        "identity.left",
        "identity.right",
    ]
    assert [(sprite["slot"], sprite["facing"]) for sprite in sprites] == [
        ("left", "right"),
        ("right", "left"),
    ]


def test_distinct_subjects_may_share_one_generic_sprite_pack(tmp_path: Path):
    shared_media = _sprite_media()
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="narration",
                        text="Two veiled figures wait together.",
                    ),
                ),
                stage_path=_stage(tmp_path / "stage.png"),
                sprite_placements=(
                    _placement(
                        subject_handle="subject.renna",
                        identity_handle="identity.veiled.feminine",
                        slot="left",
                    ),
                    _placement(
                        subject_handle="subject.mara",
                        identity_handle="identity.veiled.feminine",
                        media=shared_media,
                        slot="right",
                        facing="left",
                        anchor=(784, CARD_HEIGHT),
                    ),
                ),
            )
        ]
    )

    sprites = _manifest(deck)["identity"]["sections"][0]["sprites"]
    assert [sprite["subject_handle"] for sprite in sprites] == [
        "subject.renna",
        "subject.mara",
    ]
    assert [sprite["identity_handle"] for sprite in sprites] == [
        "identity.veiled.feminine",
        "identity.veiled.feminine",
    ]


def test_later_page_can_swap_resolved_variant_without_redrawing_stage(
    tmp_path: Path,
):
    stage_path = _stage(tmp_path / "stage.png")
    original_stage_bytes = stage_path.read_bytes()
    neutral = _placement(
        identity_handle="identity.mirelle",
        variant_handle="variant.neutral",
        media=_sprite_media(color=(205, 52, 67, 255)),
    )
    concerned = _placement(
        identity_handle="identity.mirelle",
        variant_handle="variant.concerned",
        media=_sprite_media(color=(53, 186, 103, 255)),
    )
    sections = [
        VisualNovelDeckSection(
            pages=(
                VisualNovelPage(
                    kind="dialogue",
                    speaker="Mirelle",
                    text="I understand.",
                ),
            ),
            stage_path=stage_path,
            sprite_placements=(neutral,),
        ),
        VisualNovelDeckSection(
            pages=(
                VisualNovelPage(
                    kind="dialogue",
                    speaker="Mirelle",
                    text="But that does not make it safe.",
                ),
            ),
            stage_path=stage_path,
            sprite_placements=(concerned,),
        ),
    ]
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    first = renderer.render_deck(sections)
    second = renderer.render_deck(sections)

    assert first.deck_id == second.deck_id
    assert [card.image_bytes for card in first.cards] == [
        card.image_bytes for card in second.cards
    ]
    assert stage_path.read_bytes() == original_stage_bytes
    with Image.open(first.cards[0].image_path) as card:
        assert card.convert("RGB").getpixel((240, 150)) == (205, 52, 67)
    with Image.open(first.cards[1].image_path) as card:
        assert card.convert("RGB").getpixel((240, 150)) == (53, 186, 103)
    assert [
        section["sprites"][0]["variant_handle"]
        for section in _manifest(first)["identity"]["sections"]
    ] == ["variant.neutral", "variant.concerned"]
    assert first.transcript == (
        "Mirelle: I understand.\n\nMirelle: But that does not make it safe."
    )


def test_identity_transition_obscures_old_sprite_then_presents_new_sprite(
    tmp_path: Path,
) -> None:
    stage_path = _stage(tmp_path / "stage.png")
    old_placement = _placement(
        subject_handle="subject.renna",
        identity_handle="identity.veiled",
        variant_handle="variant.veiled.neutral",
        media=_sprite_media(color=(202, 48, 66, 255)),
        slot="center",
        anchor=(CARD_WIDTH // 2, CARD_HEIGHT),
        scale_percent=98,
    )
    new_placement = _placement(
        subject_handle="subject.renna",
        identity_handle="identity.renna",
        variant_handle="variant.renna.neutral",
        media=_sprite_media(color=(49, 186, 104, 255)),
        slot="center",
        anchor=(CARD_WIDTH // 2, CARD_HEIGHT),
        scale_percent=98,
    )
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck([
        VisualNovelDeckSection(
            pages=(VisualNovelPage(
                kind="narration",
                text="The chamber seals around the veiled figure.",
            ),),
            stage_path=stage_path,
            sprite_placements=(old_placement,),
        ),
        VisualNovelDeckSection(
            pages=(VisualNovelPage(
                kind="narration",
                text="A new identity comes into focus.",
            ),),
            stage_path=stage_path,
            sprite_placements=(old_placement,),
            card_style="identity_flash",
        ),
        VisualNovelDeckSection(
            pages=(VisualNovelPage(
                kind="narration",
                text="Renna Holt",
            ),),
            stage_path=stage_path,
            sprite_placements=(new_placement,),
            card_style="identity_reveal",
        ),
    ])

    assert len(deck.cards) == 3
    with Image.open(deck.cards[0].image_path) as old_card:
        assert old_card.convert("RGB").getpixel((512, 300)) == (202, 48, 66)
    with Image.open(deck.cards[1].image_path) as flash_card:
        flash_pixel = flash_card.convert("RGB").getpixel((512, 300))
        assert min(flash_pixel) > 220
    with Image.open(deck.cards[2].image_path) as reveal_card:
        assert reveal_card.convert("RGB").getpixel((512, 300)) == (49, 186, 104)

    manifest = _manifest(deck)
    assert [
        section["card_style"]
        for section in manifest["identity"]["sections"]
    ] == ["adv", "identity_flash", "identity_reveal"]
    assert [
        section["sprites"][0]["identity_handle"]
        for section in manifest["identity"]["sections"]
    ] == ["identity.veiled", "identity.veiled", "identity.renna"]
    assert VisualNovelCardRenderer(
        tmp_path / "presentations"
    ).load_deck(deck.deck_id) == deck


def test_sprite_scale_and_bottom_center_anchor_are_deterministic(
    tmp_path: Path,
):
    media = _sprite_media(color=(204, 55, 72, 255))
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    page = (VisualNovelPage(kind="narration", text="A measured shift."),)

    small = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=page,
                sprite_placements=(
                    _placement(
                        media=media,
                        anchor=(240, 500),
                        scale_percent=50,
                    ),
                ),
            )
        ]
    )
    large = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=page,
                sprite_placements=(
                    _placement(
                        media=media,
                        anchor=(240, 500),
                        scale_percent=75,
                    ),
                ),
            )
        ]
    )

    with Image.open(small.cards[0].image_path) as small_card:
        assert small_card.convert("RGB").getpixel((240, 150)) != (204, 55, 72)
        assert small_card.convert("RGB").getpixel((240, 270)) == (204, 55, 72)
    with Image.open(large.cards[0].image_path) as large_card:
        assert large_card.convert("RGB").getpixel((240, 150)) == (204, 55, 72)
    assert small.deck_id != large.deck_id
    small_sprite = _manifest(small)["identity"]["sections"][0]["sprites"][0]
    large_sprite = _manifest(large)["identity"]["sections"][0]["sprites"][0]
    assert small_sprite["anchor"] == large_sprite["anchor"] == [240, 500]
    assert (small_sprite["scale_percent"], large_sprite["scale_percent"]) == (
        50,
        75,
    )


def test_opaque_variant_provenance_is_part_of_deck_identity(tmp_path: Path):
    media = _sprite_media()
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    page = (VisualNovelPage(kind="narration", text="She waits."),)
    neutral = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=page,
                sprite_placements=(
                    _placement(
                        variant_handle="variant.neutral",
                        media=media,
                    ),
                ),
            )
        ]
    )
    alias = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=page,
                sprite_placements=(
                    _placement(
                        variant_handle="variant.reviewed.alias",
                        media=media,
                    ),
                ),
            )
        ]
    )

    assert neutral.cards[0].image_bytes == alias.cards[0].image_bytes
    assert neutral.deck_id != alias.deck_id


def test_long_semantic_page_splits_into_measured_cards(tmp_path: Path):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Wren",
                        text=" ".join(["carefully measured dialogue"] * 45),
                    ),
                )
            )
        ],
    )

    assert len(deck.cards) > 1
    assert all(card.speaker == "Wren" for card in deck.cards)
    assert [card.index for card in deck.cards] == list(range(1, len(deck.cards) + 1))
    assert {card.count for card in deck.cards} == {len(deck.cards)}
    assert all(len(card.text.splitlines()) <= 4 for card in deck.cards)
    assert deck.used_neutral_stage is True


def test_overflow_moves_complete_sentences_to_the_next_card(tmp_path: Path):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    first = (
        "Wren checks each plain timber post before crossing the quiet lobby "
        "floor, keeping her pace measured while afternoon light moves across "
        "the open courtyard and the unadorned railings frame the entrance."
    )
    second = (
        "Iselle watches from beside the modest arch, one hand resting at her "
        "side while her knife remains in the other hand and dust shifts over "
        "the simple paving between them."
    )

    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="narration",
                        text=f"{first} {second}",
                    ),
                )
            ),
        ]
    )

    assert len(deck.cards) == 2
    assert " ".join(deck.cards[0].text.split()) == first
    assert " ".join(deck.cards[1].text.split()) == second
    assert all(len(card.text.splitlines()) <= 4 for card in deck.cards)


def test_adjacent_mid_sentence_model_pages_are_rejoined_before_layout(
    tmp_path: Path,
):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="narration",
                        text=(
                            "Iselle shifts her stance toward Wren, but her knife "
                            "stays tight in her"
                        ),
                    ),
                    VisualNovelPage(
                        kind="narration",
                        text="other hand.",
                    ),
                )
            ),
        ]
    )

    assert len(deck.cards) == 1
    assert " ".join(deck.cards[0].text.split()) == (
        "Iselle shifts her stance toward Wren, but her knife stays tight in "
        "her other hand."
    )


def test_unpunctuated_short_utterance_keeps_its_authored_page(tmp_path: Path):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(kind="dialogue", speaker="Wren", text="No"),
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Wren",
                        text="Try again.",
                    ),
                )
            ),
        ]
    )

    assert [" ".join(card.text.split()) for card in deck.cards] == [
        "No",
        "Try again.",
    ]


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


def test_sprite_deck_restarts_from_manifest_verified_card_bytes(
    tmp_path: Path,
):
    runtime_root = tmp_path / "presentations"
    renderer = VisualNovelCardRenderer(runtime_root)
    media = _sprite_media()
    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Rowan",
                        text="Not yet.",
                    ),
                ),
                stage_path=_stage(tmp_path / "stage.png"),
                sprite_placements=(
                    _placement(
                        identity_handle="identity.rowan",
                        variant_handle="variant.tense",
                        media=media,
                    ),
                ),
            )
        ]
    )
    manifest = _manifest(deck)

    loaded = VisualNovelCardRenderer(runtime_root).load_deck(deck.deck_id)

    assert loaded == deck
    assert loaded is not None
    assert (
        hashlib.sha256(loaded.cards[0].image_bytes).hexdigest()
        == (manifest["cards"][0]["sha256"])
    )
    serialized = json.dumps(manifest, sort_keys=True)
    assert media.filename not in serialized
    assert str(tmp_path) not in serialized
    assert manifest["identity"]["sections"][0]["sprites"] == [
        {
            "subject_handle": "subject.alpha",
            "identity_handle": "identity.rowan",
            "variant_handle": "variant.tense",
            "source_sha256": media.sha256,
            "source_mime_type": "image/png",
            "source_byte_count": media.byte_count,
            "source_size": [media.width, media.height],
            "slot": "left",
            "source_facing": "right",
            "facing": "right",
            "anchor": [240, CARD_HEIGHT],
            "scale_percent": 100,
        }
    ]


def test_new_manifest_binds_v2_identity_and_card_hashes(tmp_path: Path):
    renderer, deck = _single_page_deck(tmp_path)

    payload = _manifest(deck)

    assert payload["version"] == 2
    assert payload["identity"]["renderer"]
    assert payload["identity"]["card_size"] == [CARD_WIDTH, CARD_HEIGHT]
    assert payload["identity"]["sections"][0]["sprites"] == []
    assert payload["identity"]["sections"][0]["pages"] == [
        {
            "kind": "narration",
            "speaker": "",
            "text": "Wind moves across the open beginner court.",
        }
    ]
    assert (
        payload["cards"][0]["sha256"]
        == hashlib.sha256(deck.cards[0].image_path.read_bytes()).hexdigest()
    )
    assert payload["deck_id"] == _v2_deck_id(payload)
    assert renderer.load_deck(deck.deck_id) == deck


def test_loader_rejects_tampered_sprite_provenance_under_original_deck_id(
    tmp_path: Path,
):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(VisualNovelPage(kind="narration", text="She waits."),),
                sprite_placements=(_placement(),),
            )
        ]
    )
    payload = _manifest(deck)
    payload["identity"]["sections"][0]["sprites"][0]["source_sha256"] = "0" * 64
    _write_manifest(deck, payload)

    assert renderer.load_deck(deck.deck_id) is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("anchor", [True, CARD_HEIGHT]),
        ("anchor", [240.0, CARD_HEIGHT]),
        ("scale_percent", True),
        ("scale_percent", 151),
        ("source_size", [80, 0]),
        ("source_byte_count", 0),
        ("slot", "foreground"),
        ("facing", "camera"),
    ),
)
def test_loader_rejects_self_consistent_malformed_sprite_identity(
    tmp_path: Path,
    field: str,
    value: object,
):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(VisualNovelPage(kind="narration", text="She waits."),),
                sprite_placements=(_placement(),),
            )
        ]
    )
    payload = _manifest(deck)
    payload["identity"]["sections"][0]["sprites"][0][field] = value
    new_deck_id = _rehome_v2_deck(renderer, deck, payload)

    assert renderer.load_deck(new_deck_id) is None


def test_loader_rejects_path_or_extra_fields_in_sprite_provenance(
    tmp_path: Path,
):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(VisualNovelPage(kind="narration", text="She waits."),),
                sprite_placements=(_placement(),),
            )
        ]
    )
    payload = _manifest(deck)
    payload["identity"]["sections"][0]["sprites"][0]["source_path"] = str(
        tmp_path / "sprite-link.png"
    )
    new_deck_id = _rehome_v2_deck(renderer, deck, payload)

    assert renderer.load_deck(new_deck_id) is None


def test_loader_rejects_v1_manifest_instead_of_guessing_legacy_identity(
    tmp_path: Path,
):
    renderer, deck = _single_page_deck(tmp_path)
    payload = _manifest(deck)
    payload["version"] = 1
    payload.pop("identity")
    for card in payload["cards"]:
        card.pop("sha256")
    _write_manifest(deck, payload)

    assert renderer.load_deck(deck.deck_id) is None


@pytest.mark.parametrize(
    ("speaker", "text"),
    (
        ("summon_sword", "The newcomer steps into the light."),
        ("Pip", "The summon_sword steps into the light."),
    ),
)
def test_render_rejects_source_shaped_ids_before_writing(
    tmp_path: Path,
    speaker: str,
    text: str,
):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    with pytest.raises(ValueError, match="source-shaped ids"):
        renderer.render_deck(
            [
                VisualNovelDeckSection(
                    pages=(
                        VisualNovelPage(
                            kind="dialogue",
                            speaker=speaker,
                            text=text,
                        ),
                    )
                ),
            ]
        )

    assert list(renderer.deck_root.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    (
        ("speaker", "summon_sword"),
        ("text", "The summon_sword steps into the light."),
    ),
)
def test_v2_loader_rejects_self_consistent_source_shaped_page_fields(
    tmp_path: Path,
    field: str,
    unsafe_value: str,
):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Pip",
                        text="The newcomer steps into the light.",
                    ),
                )
            ),
        ]
    )
    payload = _manifest(deck)
    payload["cards"][0][field] = unsafe_value
    payload["identity"]["sections"][0]["pages"][0][field] = unsafe_value
    page = payload["identity"]["sections"][0]["pages"][0]
    payload["transcript"] = f"{page['speaker']}: {page['text']}"
    new_deck_id = _rehome_v2_deck(renderer, deck, payload)

    assert renderer.load_deck(new_deck_id) is None


def test_loader_rejects_unsupported_manifest_without_raising(tmp_path: Path):
    renderer, deck = _single_page_deck(tmp_path)
    payload = _manifest(deck)
    payload["version"] = 999
    _write_manifest(deck, payload)

    assert renderer.load_deck(deck.deck_id) is None


@pytest.mark.parametrize("serialized", ("{", "[]", '"manifest"'))
def test_loader_returns_none_for_malformed_manifest_payloads(
    tmp_path: Path,
    serialized: str,
):
    renderer, deck = _single_page_deck(tmp_path)
    deck.manifest_path.write_text(serialized, encoding="utf-8")

    assert renderer.load_deck(deck.deck_id) is None


def test_loader_rejects_manifest_whose_identity_no_longer_matches_deck_id(
    tmp_path: Path,
):
    renderer, deck = _single_page_deck(tmp_path)
    payload = _manifest(deck)
    payload["identity"]["sections"][0]["pages"][0]["text"] = "Tampered."
    _write_manifest(deck, payload)

    assert renderer.load_deck(deck.deck_id) is None


def test_v2_loader_rejects_self_consistent_unknown_renderer(tmp_path: Path):
    renderer, deck = _single_page_deck(tmp_path)
    payload = _manifest(deck)
    payload["identity"]["renderer"] = "unknown-renderer"
    new_deck_id = _rehome_v2_deck(renderer, deck, payload)

    assert renderer.load_deck(new_deck_id) is None


@pytest.mark.parametrize(
    "card_size",
    (
        [float(CARD_WIDTH), float(CARD_HEIGHT)],
        [True, CARD_HEIGHT],
        [CARD_WIDTH, False],
    ),
)
def test_v2_loader_rejects_noncanonical_card_size_types(
    tmp_path: Path,
    card_size: list[object],
):
    renderer, deck = _single_page_deck(tmp_path)
    payload = _manifest(deck)
    payload["identity"]["card_size"] = card_size
    new_deck_id = _rehome_v2_deck(renderer, deck, payload)

    assert renderer.load_deck(new_deck_id) is None


def test_v2_loader_keeps_historical_font_digests_as_identity_inputs(
    tmp_path: Path,
):
    renderer, deck = _single_page_deck(tmp_path)
    payload = _manifest(deck)
    payload["identity"]["fonts"]["regular_sha256"] = "0" * 64
    payload["identity"]["fonts"]["bold_sha256"] = "1" * 64
    new_deck_id = _rehome_v2_deck(renderer, deck, payload)

    loaded = renderer.load_deck(new_deck_id)

    assert loaded is not None
    assert loaded.deck_id == new_deck_id


def test_renderer_rejects_symlinked_deck_root_at_construction(tmp_path: Path):
    runtime_root = tmp_path / "presentations"
    runtime_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime_root / "decks").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="deck root"):
        VisualNovelCardRenderer(runtime_root)

    assert list(outside.iterdir()) == []


def test_renderer_rejects_symlinked_runtime_root_at_construction(
    tmp_path: Path,
):
    runtime_root = tmp_path / "presentations"
    outside = tmp_path / "outside"
    outside.mkdir()
    runtime_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="deck root"):
        VisualNovelCardRenderer(runtime_root)

    assert list(outside.iterdir()) == []


def test_render_rejects_symlinked_deck_root_swap_without_writing_outside(
    tmp_path: Path,
):
    runtime_root = tmp_path / "presentations"
    renderer = VisualNovelCardRenderer(runtime_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    renderer.deck_root.rmdir()
    renderer.deck_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="deck root"):
        renderer.render_deck(
            [
                VisualNovelDeckSection(
                    pages=(
                        VisualNovelPage(
                            kind="narration",
                            text="Wind moves across the open beginner court.",
                        ),
                    )
                ),
            ]
        )

    assert list(outside.iterdir()) == []


def test_loader_rejects_symlinked_deck_root_swap(tmp_path: Path):
    renderer, deck = _single_page_deck(tmp_path)
    original_deck_root = tmp_path / "original-decks"
    renderer.deck_root.rename(original_deck_root)
    renderer.deck_root.symlink_to(
        original_deck_root,
        target_is_directory=True,
    )

    assert renderer.load_deck(deck.deck_id) is None


def test_render_rejects_deck_root_swap_after_descriptor_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    moved_deck_root = tmp_path / "moved-decks"
    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir()
    open_deck_directory = renderer._open_render_deck_directory

    def swap_then_open(deck_root_fd: int, deck_id: str) -> int:
        renderer.deck_root.rename(moved_deck_root)
        renderer.deck_root.symlink_to(
            attacker_root,
            target_is_directory=True,
        )
        return open_deck_directory(deck_root_fd, deck_id)

    monkeypatch.setattr(
        renderer,
        "_open_render_deck_directory",
        swap_then_open,
    )

    with pytest.raises(RuntimeError, match="changed after"):
        renderer.render_deck(
            [
                VisualNovelDeckSection(
                    pages=(
                        VisualNovelPage(
                            kind="narration",
                            text="Wind moves across the open beginner court.",
                        ),
                    )
                ),
            ]
        )

    assert list(moved_deck_root.iterdir()) == []
    assert list(attacker_root.iterdir()) == []


def test_loader_rejects_deck_root_swap_after_descriptor_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    renderer, deck = _single_page_deck(tmp_path)
    moved_deck_root = tmp_path / "moved-decks"
    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir()
    shutil.copytree(
        deck.manifest_path.parent,
        attacker_root / deck.deck_id,
    )
    open_deck_directory = renderer._open_load_deck_directory

    def swap_then_open(deck_root_fd: int, deck_id: str) -> int:
        renderer.deck_root.rename(moved_deck_root)
        renderer.deck_root.symlink_to(
            attacker_root,
            target_is_directory=True,
        )
        return open_deck_directory(deck_root_fd, deck_id)

    monkeypatch.setattr(
        renderer,
        "_open_load_deck_directory",
        swap_then_open,
    )

    assert renderer.load_deck(deck.deck_id) is None


def test_render_rejects_replaced_deck_root_directory(tmp_path: Path):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    original_deck_root = tmp_path / "original-decks"
    renderer.deck_root.rename(original_deck_root)
    renderer.deck_root.mkdir()

    with pytest.raises(RuntimeError, match="changed after"):
        renderer.render_deck(
            [
                VisualNovelDeckSection(
                    pages=(
                        VisualNovelPage(
                            kind="narration",
                            text="Wind moves across the open beginner court.",
                        ),
                    )
                ),
            ]
        )

    assert list(renderer.deck_root.iterdir()) == []


def test_render_rejects_runtime_root_symlink_swap(tmp_path: Path):
    runtime_root = tmp_path / "presentations"
    renderer = VisualNovelCardRenderer(runtime_root)
    moved_runtime_root = tmp_path / "moved-presentations"
    runtime_root.rename(moved_runtime_root)
    runtime_root.symlink_to(moved_runtime_root, target_is_directory=True)

    with pytest.raises(RuntimeError, match="changed after"):
        renderer.render_deck(
            [
                VisualNovelDeckSection(
                    pages=(
                        VisualNovelPage(
                            kind="narration",
                            text="Wind moves across the open beginner court.",
                        ),
                    )
                ),
            ]
        )

    assert list((moved_runtime_root / "decks").iterdir()) == []


def test_render_rejects_symlinked_deck_directory_before_writing(tmp_path: Path):
    renderer, deck = _single_page_deck(tmp_path)
    deck_dir = deck.manifest_path.parent
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(deck_dir)
    deck_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="safe directory"):
        renderer.render_deck(
            [
                VisualNovelDeckSection(
                    pages=(
                        VisualNovelPage(
                            kind="narration",
                            text="Wind moves across the open beginner court.",
                        ),
                    )
                ),
            ]
        )

    assert list(outside.iterdir()) == []


def test_render_rejects_symlinked_card_target_before_writing(tmp_path: Path):
    renderer, deck = _single_page_deck(tmp_path)
    victim = tmp_path / "outside.png"
    victim.write_bytes(b"do not overwrite")
    card_path = deck.cards[0].image_path
    card_path.unlink()
    card_path.symlink_to(victim)

    with pytest.raises(RuntimeError, match="safe file"):
        renderer.render_deck(
            [
                VisualNovelDeckSection(
                    pages=(
                        VisualNovelPage(
                            kind="narration",
                            text="Wind moves across the open beginner court.",
                        ),
                    )
                ),
            ]
        )

    assert victim.read_bytes() == b"do not overwrite"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("index", 2),
        ("index", "1"),
        ("count", 0),
        ("count", "1"),
        ("kind", 1),
        ("speaker", ["Iselle"]),
        ("text", 42),
        ("text", " Wind moves across the open beginner court."),
        ("filename", "../page-001.png"),
        ("filename", "page-01.png"),
    ),
)
def test_loader_rejects_noncanonical_card_metadata(
    tmp_path: Path,
    field: str,
    value: object,
):
    renderer, deck = _single_page_deck(tmp_path)
    payload = _manifest(deck)
    payload["cards"][0][field] = value
    _write_manifest(deck, payload)

    assert renderer.load_deck(deck.deck_id) is None


def test_loader_rejects_valid_png_when_card_hash_changes(tmp_path: Path):
    renderer, deck = _single_page_deck(tmp_path)
    Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (90, 10, 40)).save(
        deck.cards[0].image_path
    )

    assert renderer.load_deck(deck.deck_id) is None


def test_loader_rejects_valid_rehashed_png_under_original_deck_id(
    tmp_path: Path,
):
    renderer, deck = _single_page_deck(tmp_path)
    Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (90, 10, 40)).save(
        deck.cards[0].image_path
    )
    payload = _manifest(deck)
    payload["cards"][0]["sha256"] = hashlib.sha256(
        deck.cards[0].image_path.read_bytes()
    ).hexdigest()
    _write_manifest(deck, payload)

    assert renderer.load_deck(deck.deck_id) is None


@pytest.mark.parametrize("replacement", ("invalid", "wrong_size", "animated"))
def test_loader_rejects_noncanonical_png_even_with_matching_manifest_hash(
    tmp_path: Path,
    replacement: str,
):
    renderer, deck = _single_page_deck(tmp_path)
    path = deck.cards[0].image_path
    if replacement == "invalid":
        path.write_bytes(b"not a png")
    elif replacement == "wrong_size":
        Image.new("RGB", (CARD_WIDTH - 1, CARD_HEIGHT), (1, 2, 3)).save(path)
    else:
        first = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (1, 2, 3))
        second = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (4, 5, 6))
        first.save(path, save_all=True, append_images=[second], format="PNG")
    payload = _manifest(deck)
    payload["cards"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    new_deck_id = _rehome_v2_deck(renderer, deck, payload)

    assert renderer.load_deck(new_deck_id) is None


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("missing", "unresolved_media"),
        ("hash", "media_hash_mismatch"),
        ("byte_count", "media_byte_count_mismatch"),
        ("dimensions", "invalid_png"),
        ("mime", "invalid_media_type"),
        ("malformed", "invalid_png"),
        ("opaque", "opaque_png"),
        ("empty", "empty_png"),
    ),
)
def test_renderer_rejects_invalid_resolved_sprite_before_writing(
    tmp_path: Path,
    case: str,
    expected_code: str,
):
    media = _sprite_media()
    invalid: object
    if case == "missing":
        invalid = None
    elif case == "hash":
        invalid = replace(media, sha256="0" * 64)
    elif case == "byte_count":
        invalid = replace(media, byte_count=media.byte_count + 1)
    elif case == "dimensions":
        invalid = replace(media, width=media.width + 1)
    elif case == "mime":
        invalid = replace(media, mime_type="image/webp")
    elif case == "malformed":
        data = b"not a transparent png"
        invalid = replace(
            media,
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            byte_count=len(data),
        )
    elif case == "opaque":
        invalid = _resolved_png(
            Image.new("RGBA", (80, CARD_HEIGHT), (211, 45, 64, 255))
        )
    else:
        invalid = _resolved_png(Image.new("RGBA", (80, CARD_HEIGHT), (0, 0, 0, 0)))
    placement = replace(_placement(), media=invalid)  # type: ignore[arg-type]
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    with pytest.raises(VisualNovelSpriteError) as raised:
        renderer.render_deck(
            [
                VisualNovelDeckSection(
                    pages=(
                        VisualNovelPage(
                            kind="narration",
                            text="She waits.",
                        ),
                    ),
                    sprite_placements=(placement,),
                )
            ]
        )

    assert raised.value.code == expected_code
    assert list(renderer.deck_root.iterdir()) == []


def test_renderer_does_not_accept_path_or_symlink_as_resolved_sprite_media(
    tmp_path: Path,
):
    media = _sprite_media()
    source = tmp_path / "source.png"
    source.write_bytes(media.data)
    link = tmp_path / "sprite-link.png"
    link.symlink_to(source)
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")
    placement = replace(_placement(), media=link)  # type: ignore[arg-type]

    with pytest.raises(VisualNovelSpriteError) as raised:
        renderer.render_deck(
            [
                VisualNovelDeckSection(
                    pages=(
                        VisualNovelPage(
                            kind="narration",
                            text="She waits.",
                        ),
                    ),
                    sprite_placements=(placement,),
                )
            ]
        )

    assert raised.value.code == "unresolved_media"
    assert list(renderer.deck_root.iterdir()) == []


@pytest.mark.parametrize(
    ("placements", "expected_code"),
    (
        (
            (
                _placement(
                    subject_handle="subject.one",
                    identity_handle="identity.one",
                ),
                _placement(
                    subject_handle="subject.two",
                    identity_handle="identity.two",
                ),
            ),
            "duplicate_slot",
        ),
        (
            (
                _placement(identity_handle="identity.same"),
                _placement(
                    subject_handle="subject.alpha",
                    identity_handle="identity.same",
                    slot="right",
                    facing="left",
                    anchor=(784, CARD_HEIGHT),
                ),
            ),
            "duplicate_subject",
        ),
        (
            (
                _placement(slot="center", anchor=(512, CARD_HEIGHT)),
                _placement(
                    subject_handle="subject.two",
                    identity_handle="identity.two",
                    slot="right",
                    facing="left",
                    anchor=(784, CARD_HEIGHT),
                ),
            ),
            "two_sprite_slots_must_be_left_right",
        ),
        (
            (
                _placement(identity_handle="identity.one"),
                _placement(
                    subject_handle="subject.two",
                    identity_handle="identity.two",
                    slot="right",
                    facing="left",
                    anchor=(784, CARD_HEIGHT),
                ),
                _placement(
                    subject_handle="subject.three",
                    identity_handle="identity.three",
                    slot="center",
                    anchor=(512, CARD_HEIGHT),
                ),
            ),
            "too_many_placements",
        ),
    ),
)
def test_renderer_rejects_ambiguous_sprite_layouts(
    tmp_path: Path,
    placements: tuple[VisualNovelSpritePlacement, ...],
    expected_code: str,
):
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    with pytest.raises(VisualNovelSpriteError) as raised:
        renderer.render_deck(
            [
                VisualNovelDeckSection(
                    pages=(
                        VisualNovelPage(
                            kind="narration",
                            text="They wait.",
                        ),
                    ),
                    sprite_placements=placements,
                )
            ]
        )

    assert raised.value.code == expected_code


def test_corrupt_stage_fails_safe_to_neutral_card(tmp_path: Path):
    bad_stage = tmp_path / "broken.webp"
    bad_stage.write_text("not an image", encoding="utf-8")
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="narration",
                        text="You step outside.",
                    ),
                ),
                stage_path=bad_stage,
            )
        ],
    )

    assert deck.used_neutral_stage is True
    assert deck.cards[0].image_path.is_file()


def test_ordered_sections_keep_each_paginated_beat_on_its_stage(
    tmp_path: Path,
):
    first_stage = _stage(tmp_path / "first.png")
    second_stage = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (17, 29, 43))
    second_stage.putpixel((5, 5), (13, 211, 97))
    second_stage_path = tmp_path / "second.png"
    second_stage.save(second_stage_path)
    renderer = VisualNovelCardRenderer(tmp_path / "presentations")

    deck = renderer.render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Iselle",
                        text=" ".join(["The first beat stays here."] * 45),
                    ),
                ),
                stage_path=first_stage,
            ),
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Wren",
                        text="The second beat moves outside.",
                    ),
                ),
                stage_path=second_stage_path,
            ),
        ]
    )

    iselle_cards = [card for card in deck.cards if card.speaker == "Iselle"]
    wren_cards = [card for card in deck.cards if card.speaker == "Wren"]
    assert len(iselle_cards) > 1
    assert len(wren_cards) == 1
    for card in iselle_cards:
        with Image.open(card.image_path) as image:
            assert image.convert("RGB").getpixel((5, 5)) == (201, 37, 59)
    with Image.open(wren_cards[0].image_path) as image:
        assert image.convert("RGB").getpixel((5, 5)) == (13, 211, 97)
    assert [card.index for card in deck.cards] == list(range(1, len(deck.cards) + 1))
    assert {card.count for card in deck.cards} == {len(deck.cards)}
    assert deck.transcript.index("Iselle:") < deck.transcript.index("Wren:")
    assert deck.used_neutral_stage is False

    reloaded = VisualNovelCardRenderer(tmp_path / "presentations").load_deck(
        deck.deck_id
    )
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
