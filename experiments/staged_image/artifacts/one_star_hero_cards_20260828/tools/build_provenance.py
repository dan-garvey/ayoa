#!/usr/bin/env python3
"""Write auditable provenance records for the isolated hero-card proof set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[3]
VISUALS = WORKSPACE / "app/storage/stories/one_star_ascension_s1/visual-references"
PRIVATE_STYLE_ROOT = (
    WORKSPACE / "private_extractions/pick_me_up_style_lora/raw"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def experiment_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def image_record(path: Path, *, private: bool = False) -> dict[str, Any]:
    with Image.open(path) as image:
        mode = image.mode
        dimensions = [image.width, image.height]
        alpha_extrema = (
            list(image.convert("RGBA").getchannel("A").getextrema())
            if "A" in image.getbands()
            else None
        )
    return {
        "path": workspace_path(path) if private else experiment_path(path),
        "sha256": sha256(path),
        "mode": mode,
        "dimensions": dimensions,
        "alpha_extrema": alpha_extrema,
    }


def source_record(path: Path, role: str) -> dict[str, Any]:
    with Image.open(path) as image:
        dimensions = [image.width, image.height]
        mode = image.mode
    return {
        "path": workspace_path(path),
        "role": role,
        "sha256": sha256(path),
        "mode": mode,
        "dimensions": dimensions,
    }


STYLE_SOURCES = (
    (
        PRIVATE_STYLE_ROOT / "0455_han_isratte_1_star_card_png.png",
        "coding-time style and composition reference 1",
    ),
    (
        PRIVATE_STYLE_ROOT / "0456_jenna_shirai_1_star_card_png.png",
        "coding-time style and composition reference 2",
    ),
    (
        PRIVATE_STYLE_ROOT / "0454_han_isratte_2_star_card_png.png",
        "coding-time style and composition reference 3",
    ),
    (
        PRIVATE_STYLE_ROOT / "0670_han_isratte_3_star_card_png.png",
        "coding-time style and composition reference 4",
    ),
)


PORTRAIT_SOURCES: dict[str, dict[str, Any]] = {
    "renna": {
        "display_name": "Renna Holt",
        "neutral": VISUALS / "vn-sprites/renna_holt/neutral.png",
        "locked": VISUALS / "locked/renna_holt/facial_zoom.png",
        "generation_inputs": (
            (
                VISUALS / "vn-sprites/renna_holt/neutral.png",
                "input 1: primary identity, silhouette, and outfit anchor",
            ),
            (
                VISUALS / "locked/renna_holt/facial_zoom.png",
                "input 2: reviewed facial identity anchor",
            ),
            (
                VISUALS / "locked/renna_holt/anatomy.png",
                "input 3: reviewed anatomy and outfit anchor",
            ),
        ),
        "locked_note": "distinct reviewed facial reference",
    },
    "warden": {
        "display_name": "Warden of the Eighth",
        "neutral": VISUALS / "vn-sprites/warden_of_the_eighth/neutral.png",
        "locked": VISUALS / "warden_of_the_eighth.webp",
        "generation_inputs": (
            (
                VISUALS / "vn-sprites/warden_of_the_eighth/neutral.png",
                "input 1: primary nonhuman identity, silhouette, and anatomy anchor",
            ),
            (
                VISUALS / "warden_of_the_eighth.webp",
                "input 2: reviewed identity and material anchor",
            ),
            (
                VISUALS / "locked/warden_of_the_eighth/anatomy.png",
                "input 3: reviewed anatomy anchor; byte-identical to input 2",
            ),
        ),
        "locked_note": "distinct reviewed nonhuman identity reference",
        "scope_note": "visual layout stress test only; not a summonable Hero",
    },
    "halcyon": {
        "display_name": "Halcyon of the Gilded March",
        "neutral": VISUALS / "vn-sprites/halcyon_of_the_gilded_march/neutral.png",
        "locked": VISUALS
        / "locked/halcyon_of_the_gilded_march/facial_zoom.png",
        "generation_inputs": (
            (
                VISUALS / "vn-sprites/halcyon_of_the_gilded_march/neutral.png",
                "input 1: primary identity, silhouette, and outfit anchor",
            ),
            (
                VISUALS
                / "locked/halcyon_of_the_gilded_march/facial_zoom.png",
                "input 2: reviewed facial identity anchor",
            ),
            (
                VISUALS / "locked/halcyon_of_the_gilded_march/anatomy.png",
                "input 3: reviewed anatomy and outfit anchor",
            ),
        ),
        "locked_note": "distinct reviewed facial reference",
    },
    "veiled_feminine": {
        "display_name": "Veiled feminine default",
        "neutral": VISUALS / "vn-sprites/veiled_feminine/neutral.png",
        "locked": VISUALS / "vn-sprites/veiled_feminine/neutral.png",
        "generation_inputs": (
            (
                VISUALS / "vn-sprites/veiled_feminine/neutral.png",
                "input 1: sole reviewed identity-veiling and clothing anchor",
            ),
        ),
        "locked_note": (
            "no independent locked portrait exists; method 2 intentionally uses "
            "the same reviewed neutral generic asset as method 1"
        ),
    },
    "veiled_masculine": {
        "display_name": "Veiled masculine default",
        "neutral": VISUALS / "vn-sprites/veiled_masculine/neutral.png",
        "locked": VISUALS / "vn-sprites/veiled_masculine/neutral.png",
        "generation_inputs": (
            (
                VISUALS / "vn-sprites/veiled_masculine/neutral.png",
                "input 1: sole reviewed identity-veiling and clothing anchor",
            ),
        ),
        "locked_note": (
            "no independent locked portrait exists; method 2 intentionally uses "
            "the same reviewed neutral generic asset as method 1"
        ),
    },
}


FRAME_ASSETS = (
    (
        "obsidian_orrery",
        ROOT / "generated_raw/frames/01_obsidian_orrery_raw.png",
        ROOT / "inputs/masks/01_obsidian_orrery_birefnet_mask.png",
        ROOT / "generated_raw/frames/01_obsidian_orrery_birefnet_rgba.png",
    ),
    (
        "frostbound_archive",
        ROOT / "generated_raw/frames/02_frostbound_archive_raw.png",
        ROOT / "inputs/masks/02_frostbound_archive_birefnet_mask.png",
        ROOT / "generated_raw/frames/02_frostbound_archive_birefnet_rgba.png",
    ),
    (
        "ashen_crown",
        ROOT / "generated_raw/frames/03_ashen_crown_raw.png",
        ROOT / "inputs/masks/03_ashen_crown_birefnet_mask.png",
        ROOT / "generated_raw/frames/03_ashen_crown_birefnet_rgba.png",
    ),
)


def write_json(name: str, value: Any) -> Path:
    output = ROOT / "provenance" / name
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    (ROOT / "provenance").mkdir(parents=True, exist_ok=True)

    private_sources = {
        "schema_version": 1,
        "handling": (
            "Experiment-private coding-time provenance. Do not copy this file, "
            "its paths, or its identifiers into review pixels, the public proof "
            "manifest, transcripts, or production assets."
        ),
        "generation_input_order": "the same ordered set was supplied to all three frame requests",
        "sources": [
            source_record(path, role) for path, role in STYLE_SOURCES
        ],
        "allowed_influence": (
            "high-level vertical card grammar, dark metal material family, empty "
            "portrait aperture, top medallion placement, and lower plaque placement"
        ),
        "excluded_content": [
            "depicted characters or silhouettes",
            "names or readable text",
            "star rows or rank marks",
            "medallion symbols, heraldry, logos, or emblems",
            "exact ornaments, source identifiers, and extraction paths",
        ],
    }
    write_json("private_sources.json", private_sources)

    portrait_records: dict[str, Any] = {}
    generated_records: dict[str, Any] = {}
    for key, spec in PORTRAIT_SOURCES.items():
        raw = ROOT / f"generated_raw/portraits/{key}_reference_guided_raw.png"
        mask = ROOT / f"inputs/masks/{key}_reference_guided_birefnet_mask.png"
        rgba = ROOT / f"generated_raw/portraits/{key}_reference_guided_rgba.png"
        portrait_records[key] = {
            "display_name": spec["display_name"],
            "method_1_neutral_vn_sprite_crop": source_record(
                spec["neutral"], "deterministic neutral VN sprite crop"
            ),
            "method_2_locked_reference_crop": {
                **source_record(
                    spec["locked"], "deterministic locked reference crop"
                ),
                "note": spec["locked_note"],
            },
            "method_3_reference_guided_generation_inputs": [
                source_record(path, role)
                for path, role in spec["generation_inputs"]
            ],
            "method_3_generated_content": image_record(raw),
            "method_3_birefnet_mask": image_record(mask),
            "method_3_accepted_rgba_candidate": image_record(rgba),
            "scope_note": spec.get("scope_note"),
        }
        generated_records[key] = {
            "kind": "reference-guided bust candidate",
            "prompt": "prompts/02_reference_guided_busts.md",
            "raw_generation": image_record(raw),
            "raw_generation_status": (
                "rejected as an overlay because it is RGB with a baked "
                "checkerboard; accepted only as the content source for matting"
            ),
            "matte": image_record(mask),
            "accepted_rgba_candidate": image_record(rgba),
            "production_status": "experiment-only; not promoted or bound",
        }
    write_json(
        "portrait_sources.json",
        {
            "schema_version": 1,
            "method_labels": {
                "method_1": "neutral VN sprite crop",
                "method_2": "locked facial/reference crop",
                "method_3": "reference-guided generated bust plus BiRefNet",
            },
            "subjects": portrait_records,
        },
    )

    frame_records: dict[str, Any] = {}
    for key, raw, mask, rgba in FRAME_ASSETS:
        frame_records[key] = {
            "kind": "blank ornate frame candidate",
            "prompt": "prompts/01_frame_candidates.md",
            "private_style_inputs": "see provenance/private_sources.json",
            "raw_generation": image_record(raw),
            "raw_generation_status": (
                "rejected as an overlay because it is RGB with a baked "
                "checkerboard; accepted only as the content source for matting"
            ),
            "matte": image_record(mask),
            "accepted_rgba_candidate": image_record(rgba),
            "production_status": "experiment-only; not promoted or bound",
        }
    write_json(
        "generated_assets.json",
        {
            "schema_version": 1,
            "frames": frame_records,
            "portraits": generated_records,
        },
    )

    alpha_attempt = ROOT / "generated_raw/frames/01_obsidian_orrery_alpha_attempt.png"
    rejected_attempts = {
        "schema_version": 1,
        "attempts": [
            {
                "artifact": image_record(alpha_attempt),
                "prompt": "prompts/03_alpha_repair_attempt.md",
                "status": "rejected",
                "reason": (
                    "The background-extraction follow-up returned another RGB "
                    "PNG with a baked checkerboard instead of genuine alpha."
                ),
                "preservation": "kept unchanged as failure evidence",
            },
            *[
                {
                    "artifact": image_record(raw),
                    "status": "rejected as final overlay",
                    "reason": (
                        "The image-generation result is RGB with a baked "
                        "checkerboard rather than transparent RGBA."
                    ),
                    "preservation": (
                        "kept unchanged as the content source for the recorded "
                        "BiRefNet matte"
                    ),
                }
                for _key, raw, _mask, _rgba in FRAME_ASSETS
            ],
            *[
                {
                    "artifact": image_record(
                        ROOT
                        / f"generated_raw/portraits/{key}_reference_guided_raw.png"
                    ),
                    "status": "rejected as final overlay",
                    "reason": (
                        "The image-generation result is RGB with a baked "
                        "checkerboard rather than transparent RGBA."
                    ),
                    "preservation": (
                        "kept unchanged as the content source for the recorded "
                        "BiRefNet matte"
                    ),
                }
                for key in PORTRAIT_SOURCES
            ],
        ],
    }
    write_json("rejected_attempts.json", rejected_attempts)

    prompt_records = []
    for path in sorted((ROOT / "prompts").glob("*.md")):
        prompt_records.append(
            {
                "path": experiment_path(path),
                "sha256": sha256(path),
            }
        )
    write_json(
        "exclusions.json",
        {
            "schema_version": 1,
            "prompts": prompt_records,
            "production_exclusions": [
                "No generated frame or portrait is bound to production code, story seed, registry, or environment configuration.",
                "No production asset was modified by this workstream.",
                "The review directory contains proofs only, not source images or private provenance.",
            ],
            "pixel_and_manifest_exclusions": [
                "private extraction paths and numeric source identifiers",
                "source filenames and source hashes",
                "generation prompt text and provider implementation details",
                "baked character names, stars, emblem, logo, or readable text in frame overlays",
            ],
            "generated_asset_constraints": [
                "single isolated subject or blank frame as applicable",
                "no watermark or signature",
                "no invented character class marker",
                "no facial reveal for either veiled default",
            ],
        },
    )

    for output in sorted((ROOT / "provenance").glob("*.json")):
        print(f"{output.relative_to(ROOT)} {sha256(output)}")


if __name__ == "__main__":
    main()
