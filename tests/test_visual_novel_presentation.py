from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

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


def _manifest(deck) -> dict:
    return json.loads(deck.manifest_path.read_text(encoding="utf-8"))


def _write_manifest(deck, payload: object) -> None:
    deck.manifest_path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


def _v2_deck_id(payload: dict) -> str:
    return hashlib.sha256(json.dumps(
        {
            "identity": payload["identity"],
            "card_sha256s": [
                card["sha256"] for card in payload["cards"]
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


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
    deck = renderer.render_deck([
        VisualNovelDeckSection(pages=(VisualNovelPage(
            kind="narration",
            text="Wind moves across the open beginner court.",
        ),)),
    ])
    return renderer, deck


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


def test_new_manifest_binds_v2_identity_and_card_hashes(tmp_path: Path):
    renderer, deck = _single_page_deck(tmp_path)

    payload = _manifest(deck)

    assert payload["version"] == 2
    assert payload["identity"]["renderer"]
    assert payload["identity"]["card_size"] == [CARD_WIDTH, CARD_HEIGHT]
    assert payload["identity"]["sections"][0]["pages"] == [{
        "kind": "narration",
        "speaker": "",
        "text": "Wind moves across the open beginner court.",
    }]
    assert payload["cards"][0]["sha256"] == hashlib.sha256(
        deck.cards[0].image_path.read_bytes()
    ).hexdigest()
    assert payload["deck_id"] == _v2_deck_id(payload)
    assert renderer.load_deck(deck.deck_id) == deck


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
        renderer.render_deck([
            VisualNovelDeckSection(pages=(VisualNovelPage(
                kind="dialogue",
                speaker=speaker,
                text=text,
            ),)),
        ])

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
    deck = renderer.render_deck([
        VisualNovelDeckSection(pages=(VisualNovelPage(
            kind="dialogue",
            speaker="Pip",
            text="The newcomer steps into the light.",
        ),)),
    ])
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
        renderer.render_deck([
            VisualNovelDeckSection(pages=(VisualNovelPage(
                kind="narration",
                text="Wind moves across the open beginner court.",
            ),)),
        ])

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
        renderer.render_deck([
            VisualNovelDeckSection(pages=(VisualNovelPage(
                kind="narration",
                text="Wind moves across the open beginner court.",
            ),)),
        ])

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
        renderer.render_deck([
            VisualNovelDeckSection(pages=(VisualNovelPage(
                kind="narration",
                text="Wind moves across the open beginner court.",
            ),)),
        ])

    assert list(renderer.deck_root.iterdir()) == []


def test_render_rejects_runtime_root_symlink_swap(tmp_path: Path):
    runtime_root = tmp_path / "presentations"
    renderer = VisualNovelCardRenderer(runtime_root)
    moved_runtime_root = tmp_path / "moved-presentations"
    runtime_root.rename(moved_runtime_root)
    runtime_root.symlink_to(moved_runtime_root, target_is_directory=True)

    with pytest.raises(RuntimeError, match="changed after"):
        renderer.render_deck([
            VisualNovelDeckSection(pages=(VisualNovelPage(
                kind="narration",
                text="Wind moves across the open beginner court.",
            ),)),
        ])

    assert list((moved_runtime_root / "decks").iterdir()) == []


def test_render_rejects_symlinked_deck_directory_before_writing(tmp_path: Path):
    renderer, deck = _single_page_deck(tmp_path)
    deck_dir = deck.manifest_path.parent
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(deck_dir)
    deck_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="safe directory"):
        renderer.render_deck([
            VisualNovelDeckSection(pages=(VisualNovelPage(
                kind="narration",
                text="Wind moves across the open beginner court.",
            ),)),
        ])

    assert list(outside.iterdir()) == []


def test_render_rejects_symlinked_card_target_before_writing(tmp_path: Path):
    renderer, deck = _single_page_deck(tmp_path)
    victim = tmp_path / "outside.png"
    victim.write_bytes(b"do not overwrite")
    card_path = deck.cards[0].image_path
    card_path.unlink()
    card_path.symlink_to(victim)

    with pytest.raises(RuntimeError, match="safe file"):
        renderer.render_deck([
            VisualNovelDeckSection(pages=(VisualNovelPage(
                kind="narration",
                text="Wind moves across the open beginner court.",
            ),)),
        ])

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
    payload["cards"][0]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    new_deck_id = _rehome_v2_deck(renderer, deck, payload)

    assert renderer.load_deck(new_deck_id) is None


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
