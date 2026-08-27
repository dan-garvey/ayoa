#!/usr/bin/env python3
"""Export every validated VN playtest card into one flat review folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine.visual_novel_presentation import (  # noqa: E402
    VisualNovelCard,
    VisualNovelCardRenderer,
)


DEFAULT_SLIDESHOW_DIRECTORY = "vn-slideshow"


def _slug(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or fallback


def _validated_cards(
    run_dir: Path,
    *,
    deck_manifest_paths: Sequence[str | Path] | None,
) -> list[tuple[str, VisualNovelCard]]:
    manifests: list[tuple[Path, Path]] = []
    candidates = (
        tuple(Path(path).resolve(strict=True) for path in deck_manifest_paths)
        if deck_manifest_paths is not None
        else tuple(run_dir.rglob("manifest.json"))
    )
    for manifest in candidates:
        if manifest.parent.parent.name != "decks":
            continue
        runtime_root = manifest.parent.parent.parent
        try:
            relative_root = runtime_root.relative_to(run_dir)
        except ValueError as exc:
            raise RuntimeError(
                "slideshow deck must belong to the playtest run"
            ) from exc
        manifests.append((relative_root, manifest))
    if deck_manifest_paths is None:
        manifests.sort(
            key=lambda item: (
                len(item[0].parts),
                item[0].as_posix(),
                item[1].parent.name,
            )
        )

    cards: list[tuple[str, VisualNovelCard]] = []
    for relative_root, manifest in manifests:
        deck = VisualNovelCardRenderer(manifest.parent.parent.parent).load_deck(
            manifest.parent.name
        )
        if deck is None:
            raise RuntimeError(
                f"playtest deck failed validation: {manifest.relative_to(run_dir)}"
            )
        label = _slug(relative_root.as_posix(), fallback="presentation")
        cards.extend((label, card) for card in deck.cards)
    if not cards:
        raise RuntimeError("playtest run contains no visual-novel cards")
    return cards


def export_vn_playtest_slideshow(
    run_dir: str | Path,
    *,
    deck_manifest_paths: Sequence[str | Path] | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    """Create a numbered slideshow without altering source decks."""

    run_root = Path(run_dir).resolve(strict=True)
    if not run_root.is_dir():
        raise RuntimeError("playtest run root must be a directory")

    cards = _validated_cards(
        run_root,
        deck_manifest_paths=deck_manifest_paths,
    )
    width = max(3, len(str(len(cards))))
    destination = run_root / DEFAULT_SLIDESHOW_DIRECTORY
    if destination.is_symlink():
        raise RuntimeError("slideshow destination cannot be a symlink")
    if destination.exists() and not destination.is_dir():
        raise RuntimeError("slideshow destination must be a directory")

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{DEFAULT_SLIDESHOW_DIRECTORY}-staging-",
            dir=run_root,
        )
    )
    exported_names: list[str] = []
    index_records: list[dict[str, object]] = []
    try:
        for index, (deck_label, card) in enumerate(cards, start=1):
            subject = _slug(card.speaker or card.kind, fallback="page")
            filename = (
                f"{index:0{width}d}__{deck_label}__{subject}__page-{card.index:03d}.png"
            )
            data = card.image_bytes
            (staging / filename).write_bytes(data)
            exported_names.append(filename)
            index_records.append(
                {
                    "filename": filename,
                    "source": str(card.image_path.relative_to(run_root)),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "speaker": card.speaker,
                    "kind": card.kind,
                }
            )
        (staging / "index.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "card_count": len(index_records),
                    "cards": index_records,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return destination, tuple(destination / name for name in exported_names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    destination, cards = export_vn_playtest_slideshow(args.run_dir)
    print(
        json.dumps(
            {
                "directory": str(destination),
                "card_count": len(cards),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
