#!/usr/bin/env python3
"""Assemble Rowan Kest's v2 sweep with the accepted regenerated sad sprite."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
FULL_CAST = REPO / "experiments/staged_image/artifacts/full_cast_sprite_pack_20260827"
sys.path.insert(0, str(FULL_CAST))

from prepare_review_sprites import normalize_candidate, physical_magenta_matte  # noqa: E402


BASE = REPO / "experiments/staged_image/artifacts/mirelle_rowan_sprite_pack_20260826"
CHARACTER = "rowan_kest"
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
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
HIGH_ALPHA_THRESHOLD = 200
EDGE_BAND_DISTANCE = 3.0
HOT_MAGENTA_SATURATION = 112
HOT_MAGENTA_HUE_DEGREES = (275, 335)
HOT_MAGENTA_MIN_VALUE = 16


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hot_magenta(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16) * 2
    return (
        (hsv[..., 1] >= HOT_MAGENTA_SATURATION)
        & (hue >= HOT_MAGENTA_HUE_DEGREES[0])
        & (hue <= HOT_MAGENTA_HUE_DEGREES[1])
        & (hsv[..., 2] >= HOT_MAGENTA_MIN_VALUE)
    )


def prepare_sad() -> dict[str, object]:
    raw = ROOT / "generation_raw/rowan_kest/sad_chroma_v2.png"
    physical = ROOT / "rejected/rowan_kest_sad_physical_unmix_v1.png"
    candidate = ROOT / "candidates/rowan_kest/sad.png"
    sprite = ROOT / "sprites/rowan_kest/sad.png"
    physical_magenta_matte(raw, physical)

    raw_rgb = np.asarray(Image.open(raw).convert("RGB"), dtype=np.uint8)
    physical_rgba = np.asarray(Image.open(physical).convert("RGBA"), dtype=np.uint8)
    alpha = physical_rgba[..., 3]
    distance_inside = distance_transform_edt(alpha > 0)
    partial_boundary = (
        (alpha > 0)
        & (alpha < 255)
        & (distance_inside <= EDGE_BAND_DISTANCE)
    )
    fully_opaque = alpha == 255
    high_alpha_interior = (
        (alpha >= HIGH_ALPHA_THRESHOLD)
        & (alpha < 255)
        & (distance_inside > EDGE_BAND_DISTANCE)
        & ~hot_magenta(raw_rgb)
    )
    hybrid = physical_rgba.copy()
    restore = fully_opaque | high_alpha_interior
    hybrid[..., :3][restore] = raw_rgb[restore]
    if not np.array_equal(hybrid[..., 3], physical_rgba[..., 3]):
        raise RuntimeError("Rowan sad alpha changed during opaque hybrid")
    if not np.array_equal(
        hybrid[..., :3][partial_boundary],
        physical_rgba[..., :3][partial_boundary],
    ):
        raise RuntimeError("Rowan sad partial-edge RGB changed")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hybrid, mode="RGBA").save(candidate, optimize=True)
    normalize = normalize_candidate(candidate, sprite)
    return {
        "raw_sha256": sha256_path(raw),
        "physical_v1_sha256": sha256_path(physical),
        "candidate_sha256": sha256_path(candidate),
        "sprite_sha256": sha256_path(sprite),
        "partial_boundary_pixels": int(partial_boundary.sum()),
        "visible_hot_magenta_pixels": int(
            ((hybrid[..., 3] > 0) & hot_magenta(hybrid[..., :3])).sum()
        ),
        "normalize": normalize,
    }


def copy_unchanged_variants() -> dict[str, dict[str, str]]:
    copied: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        if variant == "sad":
            continue
        copied[variant] = {}
        for tier in ("candidates", "sprites"):
            source = BASE / tier / CHARACTER / f"{variant}.png"
            destination = ROOT / tier / CHARACTER / f"{variant}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256_path(source) != sha256_path(destination):
                raise RuntimeError(f"copy hash mismatch: {destination}")
            copied[variant][f"{tier}_sha256"] = sha256_path(destination)
    return copied


def split_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (235, 238, 242))
    ImageDraw.Draw(image).rectangle((0, 0, width // 2, height), fill=(31, 36, 44))
    return image


def build_sheet() -> Path:
    cell_width, cell_height = 480, 650
    sheet = Image.new("RGB", (1920, 1300), (18, 22, 28))
    label_font = ImageFont.truetype(str(FONT), 28)
    small_font = ImageFont.truetype(str(FONT), 18)
    for index, variant in enumerate(VARIANTS):
        with Image.open(ROOT / "sprites" / CHARACTER / f"{variant}.png") as source:
            rgba = source.convert("RGBA")
        rgba.thumbnail((450, 575), Image.Resampling.LANCZOS)
        cell = split_background((cell_width, cell_height))
        cell.paste(rgba, ((cell_width - rgba.width) // 2, 8 + 575 - rgba.height), rgba)
        draw = ImageDraw.Draw(cell)
        draw.rectangle((0, 590, cell_width, cell_height), fill=(12, 15, 20))
        draw.text((16, 598), variant.upper(), fill="white", font=label_font)
        draw.text(
            (16, 631),
            "dark / light alpha check",
            fill=(165, 174, 186),
            font=small_font,
        )
        row, column = divmod(index, 4)
        sheet.paste(cell, (column * cell_width, row * cell_height))
    output = ROOT / "contact_sheets/rowan_kest_complete_sweep_v2.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    return output


def main() -> None:
    report = {
        "method": "seven_byte_identical_v1_sprites_plus_opaque_hybrid_sad_v2",
        "parameters": {
            "high_alpha_threshold": HIGH_ALPHA_THRESHOLD,
            "edge_band_distance_native_pixels": EDGE_BAND_DISTANCE,
            "hot_magenta_saturation_u8": HOT_MAGENTA_SATURATION,
            "hot_magenta_hue_degrees": list(HOT_MAGENTA_HUE_DEGREES),
            "hot_magenta_min_value_u8": HOT_MAGENTA_MIN_VALUE,
        },
        "unchanged": copy_unchanged_variants(),
        "sad_v2": prepare_sad(),
    }
    sheet = build_sheet()
    report["contact_sheet"] = {
        "path": str(sheet.relative_to(ROOT)),
        "sha256": sha256_path(sheet),
    }
    report_path = ROOT / "matte_reports/rowan_kest_v2.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"sheet={sheet} sha256={sha256_path(sheet)}")
    print(f"report={report_path} sha256={sha256_path(report_path)}")


if __name__ == "__main__":
    main()
