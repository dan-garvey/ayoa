from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from app.engine.player_media import ResolvedPlayerMedia
from app.engine.visual_novel_presentation import (
    VisualNovelCardRenderer,
    VisualNovelDeckSection,
)
from app.schemas.narrator import VisualNovelPage
from scripts.export_vn_playtest_slideshow import (
    export_vn_playtest_slideshow,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_vn_playtest_slideshow_flattens_validated_decks(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "playtest"
    primary = VisualNovelCardRenderer(run_dir / "presentation").render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Mara Venn",
                        text="This face is mine.",
                    ),
                )
            ),
        ]
    )
    probe = VisualNovelCardRenderer(
        run_dir / "narrator-probe/presentation"
    ).render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="narration",
                        speaker="",
                        text="Mara steadies herself.",
                    ),
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Mara Venn",
                        text="I am ready.",
                    ),
                )
            ),
        ]
    )
    original_manifest_hashes = {
        primary.manifest_path: _sha256(primary.manifest_path),
        probe.manifest_path: _sha256(probe.manifest_path),
    }

    slideshow_directory, slideshow_cards = export_vn_playtest_slideshow(run_dir)

    assert [card.name for card in slideshow_cards] == [
        "001__presentation__mara-venn__page-001.png",
        "002__narrator-probe-presentation__narration__page-001.png",
        "003__narrator-probe-presentation__mara-venn__page-002.png",
    ]
    source_cards = (*primary.cards, *probe.cards)
    assert [card.read_bytes() for card in slideshow_cards] == [
        card.image_bytes for card in source_cards
    ]
    assert {
        path: _sha256(path) for path in original_manifest_hashes
    } == original_manifest_hashes

    payload = json.loads(
        (slideshow_directory / "index.json").read_text(encoding="utf-8")
    )
    assert payload["card_count"] == 3
    assert payload["version"] == 2
    assert {card["mime_type"] for card in payload["cards"]} == {"image/png"}
    assert [card["speaker"] for card in payload["cards"]] == [
        "Mara Venn",
        "",
        "Mara Venn",
    ]

    _directory, selected_cards = export_vn_playtest_slideshow(
        run_dir,
        deck_manifest_paths=(probe.manifest_path, primary.manifest_path),
    )
    assert [card.name for card in selected_cards] == [
        "001__narrator-probe-presentation__narration__page-001.png",
        "002__narrator-probe-presentation__mara-venn__page-002.png",
        "003__presentation__mara-venn__page-001.png",
    ]


def test_export_vn_playtest_slideshow_replaces_stale_export(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "playtest"
    VisualNovelCardRenderer(run_dir / "presentation").render_deck(
        [
            VisualNovelDeckSection(
                pages=(
                    VisualNovelPage(
                        kind="narration",
                        speaker="",
                        text="The chamber brightens.",
                    ),
                )
            ),
        ]
    )
    slideshow_directory, _cards = export_vn_playtest_slideshow(run_dir)
    stale = slideshow_directory / "999__stale.png"
    stale.write_bytes(b"stale")

    rebuilt_directory, _cards = export_vn_playtest_slideshow(run_dir)

    assert not stale.exists()
    assert [path.name for path in rebuilt_directory.glob("*.png")] == [
        "001__presentation__narration__page-001.png"
    ]


def test_export_vn_playtest_slideshow_preserves_animated_gif(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "playtest"
    frames = [
        Image.new("RGB", (1024, 576), color)
        for color in ((18, 28, 48), (250, 240, 208))
    ]
    encoded = BytesIO()
    frames[0].save(
        encoded,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=[50, 400],
        disposal=2,
        optimize=False,
    )
    data = encoded.getvalue()
    panel = ResolvedPlayerMedia(
        filename="summon-reveal.gif",
        mime_type="image/gif",
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        width=1024,
        height=576,
    )
    deck = VisualNovelCardRenderer(run_dir / "presentation").render_deck([
        VisualNovelDeckSection(
            pages=(VisualNovelPage(
                kind="narration",
                text="The seal strains, then releases.",
            ),),
            stage_media=panel,
            card_style="system_panel",
        )
    ])

    slideshow_directory, slideshow_cards = export_vn_playtest_slideshow(run_dir)

    assert [card.name for card in slideshow_cards] == [
        "001__presentation__narration__page-001.gif"
    ]
    assert slideshow_cards[0].read_bytes() == deck.cards[0].image_bytes
    payload = json.loads(
        (slideshow_directory / "index.json").read_text(encoding="utf-8")
    )
    assert payload["cards"][0]["mime_type"] == "image/gif"
