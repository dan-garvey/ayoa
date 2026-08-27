#!/usr/bin/env python3
"""Normalize, validate, and summarize this experimental sprite review pack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
STORY_ROOT = REPO_ROOT / "app/storage/stories/one_star_ascension_s1"
CHECKPOINT = STORY_ROOT / "ckpt_0000.json"
VARIANTS = [
    "neutral",
    "happy",
    "concerned",
    "tense",
    "skeptical",
    "angry",
    "sad",
    "surprised",
]
CHARACTERS = ["mirelle_voss", "rowan_kest"]
DISPLAY_NAMES = {"mirelle_voss": "Mirelle Voss", "rowan_kest": "Rowan Kest"}
FINAL_SIZE = (1100, 1500)
RAW_BOTTOM_Y = 1480
ALPHA_NOISE_CUTOFF = 48
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_metadata(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    with Image.open(path) as image:
        data: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_path(path),
            "byte_count": path.stat().st_size,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }
        if "A" in image.getbands():
            alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
            total = int(alpha.size)
            foreground = alpha >= 64
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                foreground.astype(np.uint8), connectivity=8
            )
            significant = [
                int(stats[index, cv2.CC_STAT_AREA])
                for index in range(1, component_count)
                if int(stats[index, cv2.CC_STAT_AREA]) >= 100
            ]
            ys, xs = np.where(foreground)
            data["alpha"] = {
                "minimum": int(alpha.min()),
                "maximum": int(alpha.max()),
                "fully_transparent_pixels": int(np.count_nonzero(alpha == 0)),
                "partially_transparent_pixels": int(
                    np.count_nonzero((alpha > 0) & (alpha < 255))
                ),
                "opaque_pixels": int(np.count_nonzero(alpha == 255)),
                "coverage_fraction_alpha_gte_64": round(
                    float(np.count_nonzero(foreground)) / total, 6
                ),
                "bbox_alpha_gte_64": (
                    [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    if xs.size
                    else None
                ),
                "significant_component_count": len(significant),
                "significant_component_areas": sorted(significant, reverse=True),
            }
        return data


def normalize_sprites() -> None:
    for character_id in CHARACTERS:
        output_dir = ROOT / "sprites" / character_id
        output_dir.mkdir(parents=True, exist_ok=True)
        for variant_id in VARIANTS:
            source = ROOT / "candidates" / character_id / f"{variant_id}.png"
            with Image.open(source) as image:
                rgba = image.convert("RGBA")
            alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8).copy()
            alpha[alpha <= ALPHA_NOISE_CUTOFF] = 0
            rgba.putalpha(Image.fromarray(alpha, mode="L"))
            if rgba.width > FINAL_SIZE[0] or rgba.height > RAW_BOTTOM_Y:
                raise RuntimeError(
                    f"{source} does not fit losslessly on {FINAL_SIZE}: {rgba.size}"
                )
            canvas = Image.new("RGBA", FINAL_SIZE, (0, 0, 0, 0))
            x = (FINAL_SIZE[0] - rgba.width) // 2
            y = RAW_BOTTOM_Y - rgba.height
            canvas.alpha_composite(rgba, (x, y))
            canvas.save(output_dir / f"{variant_id}.png", format="PNG")


def split_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    background = Image.new("RGB", size, (235, 238, 242))
    draw = ImageDraw.Draw(background)
    draw.rectangle((0, 0, width // 2, height), fill=(31, 36, 44))
    return background


def contact_sheet(character_id: str) -> Path:
    columns, rows = 4, 2
    cell_width, cell_height = 480, 650
    art_width, art_height = 450, 575
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (18, 22, 28))
    label_font = ImageFont.truetype(str(FONT), 28)
    small_font = ImageFont.truetype(str(FONT), 18)
    for index, variant_id in enumerate(VARIANTS):
        with Image.open(
            ROOT / "sprites" / character_id / f"{variant_id}.png"
        ) as source:
            rgba = source.convert("RGBA")
        rgba.thumbnail((art_width, art_height), Image.Resampling.LANCZOS)
        cell = split_background((cell_width, cell_height))
        x = (cell_width - rgba.width) // 2
        y = 8 + (art_height - rgba.height)
        cell.paste(rgba, (x, y), rgba)
        draw = ImageDraw.Draw(cell)
        draw.rectangle((0, 590, cell_width, cell_height), fill=(12, 15, 20))
        draw.text((16, 598), variant_id.upper(), fill=(255, 255, 255), font=label_font)
        draw.text(
            (16, 631),
            "dark / light alpha check",
            fill=(165, 174, 186),
            font=small_font,
        )
        row, column = divmod(index, columns)
        sheet.paste(cell, (column * cell_width, row * cell_height))
    output = ROOT / "contact_sheets" / f"{character_id}_complete_sweep.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    return output


def source_references(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = {
        f"osa_{character_id}_locked_{view}_v1"
        for character_id in CHARACTERS
        for view in ("active_profile", "facial_zoom", "anatomy", "back_view")
    }
    records = [
        reference
        for reference in checkpoint["reviewed_visual_references"]
        if reference["reference_id"] in wanted
    ]
    if {record["reference_id"] for record in records} != wanted:
        raise RuntimeError(
            "Checkpoint does not contain the complete locked reference set"
        )
    output = []
    for record in sorted(records, key=lambda item: item["reference_id"]):
        path = STORY_ROOT / "visual-references" / record["storage_ref"]
        actual_hash = sha256_path(path)
        if actual_hash != record["sha256"]:
            raise RuntimeError(f"Reference hash mismatch: {record['reference_id']}")
        output.append(
            {
                "reference_id": record["reference_id"],
                "storage_ref": record["storage_ref"],
                "selection_hint": record["selection_hint"],
                "registry_sha256": record["sha256"],
                "actual_sha256": actual_hash,
                "width": record["width"],
                "height": record["height"],
                "byte_count": record["byte_count"],
                "manually_inspected_before_generation": True,
            }
        )
    return output


def build_manifest(contact_sheets: list[Path]) -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    prompts = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_path(path),
            "byte_count": path.stat().st_size,
        }
        for path in sorted((ROOT / "prompts").glob("*/*.txt"))
    ]
    raw_outputs = [
        image_metadata(path)
        for path in sorted((ROOT / "generation_raw").glob("*/*.png"))
    ]
    candidates = [
        image_metadata(path) for path in sorted((ROOT / "candidates").glob("*/*.png"))
    ]
    final_sprites = [
        image_metadata(path) for path in sorted((ROOT / "sprites").glob("*/*.png"))
    ]
    rejected = []
    for path in sorted((ROOT / "rejected").glob("*/*.png")):
        is_edit = "alpha_edit" in path.name
        rejected.append(
            {
                **image_metadata(path),
                "rejection_reasons": [
                    "transparency",
                    (
                        "targeted alpha-only edit still returned RGB with a "
                        "checkerboard painted into pixels"
                        if is_edit
                        else "initial generation returned RGB with a checkerboard painted into pixels"
                    ),
                ],
            }
        )
    review_documents = [
        {
            "path": path.name,
            "sha256": sha256_path(path),
            "byte_count": path.stat().st_size,
        }
        for path in (
            ROOT / "README.md",
            ROOT / "manual_review.md",
            ROOT / "prompt_conflict_audit.md",
            ROOT / "rejection_log.md",
        )
    ]
    manifest = {
        "schema_version": 1,
        "pack_id": "mirelle-rowan-pose-expression-20260826",
        "status": "experimental_unlocked_pending_human_selection",
        "production_bound": False,
        "runtime_changes": False,
        "reference_registry_changes": False,
        "checkpoint_changes": False,
        "core_variants": VARIANTS,
        "execution": {
            "generation_tool": "built-in image_gen",
            "generation_model": "not exposed by the built-in interface",
            "one_generation_call_per_nonpilot_variant": True,
            "pilot_variants": {
                "mirelle_voss": "angry",
                "rowan_kest": "skeptical",
            },
            "alpha_extraction": {
                "script": "imagegen skill scripts/remove_chroma_key.py",
                "arguments": [
                    "--auto-key",
                    "border",
                    "--soft-matte",
                    "--transparent-threshold",
                    "12",
                    "--opaque-threshold",
                    "72",
                    "--despill",
                ],
                "normalization": {
                    "canvas": list(FINAL_SIZE),
                    "raw_canvas_bottom_y": RAW_BOTTOM_Y,
                    "alpha_noise_cutoff": ALPHA_NOISE_CUTOFF,
                    "resampled": False,
                },
            },
        },
        "references": source_references(checkpoint),
        "prompts": prompts,
        "raw_outputs": raw_outputs,
        "alpha_candidates": candidates,
        "final_sprites": final_sprites,
        "rejected_iterations": rejected,
        "contact_sheets": [image_metadata(path) for path in contact_sheets],
        "review_documents": review_documents,
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_final_sprites() -> None:
    failures: list[str] = []
    for character_id in CHARACTERS:
        for variant_id in VARIANTS:
            path = ROOT / "sprites" / character_id / f"{variant_id}.png"
            metadata = image_metadata(path)
            if (metadata["width"], metadata["height"]) != FINAL_SIZE:
                failures.append(f"{path}: wrong size")
            if metadata["mode"] != "RGBA":
                failures.append(f"{path}: mode is not RGBA")
            alpha = metadata.get("alpha", {})
            if alpha.get("minimum") != 0 or alpha.get("maximum") != 255:
                failures.append(f"{path}: alpha lacks transparent or opaque pixels")
            if alpha.get("significant_component_count") != 1:
                failures.append(
                    f"{path}: significant component count "
                    f"{alpha.get('significant_component_count')}"
                )
    if failures:
        raise RuntimeError("Final sprite validation failed:\n" + "\n".join(failures))


def main() -> None:
    normalize_sprites()
    validate_final_sprites()
    sheets = [contact_sheet(character_id) for character_id in CHARACTERS]
    build_manifest(sheets)


if __name__ == "__main__":
    main()
