from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
