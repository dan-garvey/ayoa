#!/usr/bin/env python3
"""Promote reviewed pose-expression cutouts into a story seed.

This is an authoring-time utility. It copies already-reviewed PNGs byte for
byte, records their exact metadata, and binds opaque sprite-set handles. It
never generates or inspects images at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


EXPRESSIONS = (
    "neutral",
    "happy",
    "concerned",
    "tense",
    "skeptical",
    "angry",
    "sad",
    "surprised",
)
GENERIC_FOLDERS = {"veiled_masculine", "veiled_feminine"}
FOLDER_TO_CHARACTER = {"wren": "wren_thelantern"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--full-cast-sprites", type=Path, required=True)
    parser.add_argument("--mirelle-rowan-sprites", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    checkpoint_path = args.checkpoint.resolve()
    story_dir = checkpoint_path.parent
    destination_root = story_dir / "visual-references" / "vn-sprites"
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    sources: dict[str, Path] = {}
    for root in (args.full_cast_sprites, args.mirelle_rowan_sprites):
        for source_dir in sorted(root.resolve().iterdir()):
            if not source_dir.is_dir():
                continue
            identity = FOLDER_TO_CHARACTER.get(source_dir.name, source_dir.name)
            if identity in sources:
                raise ValueError(f"duplicate reviewed sprite folder: {identity}")
            sources[identity] = source_dir

    references: list[dict[str, object]] = []
    sprite_sets: list[dict[str, object]] = []
    character_set_ids: dict[str, str] = {}
    for identity, source_dir in sorted(sources.items()):
        generic = identity in GENERIC_FOLDERS
        sprite_set_id = f"osa_vnset_{identity}_v1"
        variant_ids: dict[str, str] = {}
        for expression in EXPRESSIONS:
            source = source_dir / f"{expression}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGBA":
                    raise ValueError(f"reviewed sprite is not RGBA PNG: {source}")
                width, height = image.size
            if (width, height) != (1100, 1500):
                raise ValueError(f"unexpected reviewed sprite size: {source}")

            destination = destination_root / identity / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if _sha256(source) != _sha256(destination):
                raise RuntimeError(f"sprite copy hash mismatch: {source}")
            reference_id = f"osa_vnsprite_{identity}_{expression}_v1"
            variant_ids[expression] = reference_id
            references.append({
                "reference_id": reference_id,
                "storage_ref": str(destination.relative_to(
                    story_dir / "visual-references"
                )).replace("\\", "/"),
                "mime_type": "image/png",
                "width": width,
                "height": height,
                "byte_count": destination.stat().st_size,
                "sha256": _sha256(destination),
                "purpose": "sprite",
                "scope": "presentation" if generic else "character",
                "scope_id": sprite_set_id if generic else identity,
                "selection_hint": (
                    "Reviewed visual-novel pose-expression cutout."
                ),
                "diffusion_authorized": False,
            })
        sprite_sets.append({
            "sprite_set_id": sprite_set_id,
            "owner_character_id": "" if generic else identity,
            "variant_reference_ids": variant_ids,
            "source_facing": "right",
        })
        if not generic:
            character_set_ids[identity] = sprite_set_id

    known_characters = {
        character["character_id"]: character
        for character in data["characters"]
    }
    missing_characters = sorted(set(character_set_ids) - set(known_characters))
    if missing_characters:
        raise ValueError(
            "reviewed sprite identities are absent from story: "
            + ", ".join(missing_characters)
        )
    for character_id, sprite_set_id in character_set_ids.items():
        known_characters[character_id].setdefault("visuals", {})[
            "sprite_set_id"
        ] = sprite_set_id

    retained = [
        reference
        for reference in data.get("reviewed_visual_references", [])
        if not str(reference.get("reference_id", "")).startswith(
            "osa_vnsprite_"
        )
    ]
    data["reviewed_visual_references"] = retained + references
    data["reviewed_visual_novel_sprite_sets"] = sprite_sets

    account_configs = [
        character["mechanics"]["one_star_account"]["config"]
        for character in data["characters"]
        if "one_star_account" in character.get("mechanics", {})
    ]
    if len(account_configs) != 1:
        raise ValueError("story must contain exactly one One-Star account")
    account_configs[0]["visual_novel_presentation"] = {
        "veiled_sprite_set_ids": {
            "masculine": "osa_vnset_veiled_masculine_v1",
            "feminine": "osa_vnset_veiled_feminine_v1",
        },
        "seeded_birth_one_reveal_stars": 2,
        "generated_birth_one_reveal_stars": 3,
    }

    checkpoint_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
