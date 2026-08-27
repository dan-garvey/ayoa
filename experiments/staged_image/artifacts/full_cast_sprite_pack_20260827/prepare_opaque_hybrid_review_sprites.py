#!/usr/bin/env python3
"""Apply the root-approved physical-edge/raw-interior matte to opaque sprites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from prepare_review_sprites import (
    ROOT,
    VARIANTS,
    build_contact_sheet,
    normalize_candidate,
    sha256_path,
)


HIGH_ALPHA_THRESHOLD = 200
EDGE_BAND_DISTANCE = 3.0
HOT_MAGENTA_SATURATION = 112
HOT_MAGENTA_HUE_DEGREES = (275, 335)
HOT_MAGENTA_MIN_VALUE = 16


def hot_magenta(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16) * 2
    return (
        (hsv[..., 1] >= HOT_MAGENTA_SATURATION)
        & (hue >= HOT_MAGENTA_HUE_DEGREES[0])
        & (hue <= HOT_MAGENTA_HUE_DEGREES[1])
        & (hsv[..., 2] >= HOT_MAGENTA_MIN_VALUE)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True)
    parser.add_argument(
        "--raw-override",
        action="append",
        default=[],
        metavar="LABEL=FILENAME",
    )
    parser.add_argument(
        "--physical-root",
        type=Path,
        help="Explicit physical-matte archive; defaults to the character v1 archive.",
    )
    return parser.parse_args()


def overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        label, separator, filename = value.partition("=")
        if not separator or label not in VARIANTS or not filename:
            raise RuntimeError(f"invalid raw override: {value}")
        result[label] = filename
    return result


def build_variant(
    character: str,
    variant: str,
    raw_filename: str,
    physical_root: Path,
) -> dict[str, object]:
    raw_path = ROOT / "generation_raw" / character / raw_filename
    physical_path = physical_root / "candidates" / f"{variant}.png"
    if not raw_path.is_file() or not physical_path.is_file():
        raise RuntimeError(f"missing raw/physical source for {character}/{variant}")
    raw = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.uint8)
    physical = np.asarray(Image.open(physical_path).convert("RGBA"), dtype=np.uint8)
    if physical.shape[:2] != raw.shape[:2]:
        raise RuntimeError(f"raw/physical geometry mismatch for {character}/{variant}")

    alpha = physical[..., 3]
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
        & ~hot_magenta(raw)
    )
    hybrid = physical.copy()
    restore = fully_opaque | high_alpha_interior
    hybrid[..., :3][restore] = raw[restore]
    if not np.array_equal(hybrid[..., 3], physical[..., 3]):
        raise RuntimeError(f"alpha changed for {character}/{variant}")
    if not np.array_equal(
        hybrid[..., :3][partial_boundary], physical[..., :3][partial_boundary]
    ):
        raise RuntimeError(f"partial-edge RGB changed for {character}/{variant}")
    if not np.array_equal(hybrid[..., :3][fully_opaque], raw[fully_opaque]):
        raise RuntimeError(f"opaque RGB differs from raw for {character}/{variant}")

    candidate = ROOT / "candidates" / character / f"{variant}.png"
    sprite = ROOT / "sprites" / character / f"{variant}.png"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hybrid, mode="RGBA").save(candidate, optimize=True)
    normalized = normalize_candidate(candidate, sprite)
    return {
        "accepted_raw": {
            "filename": raw_filename,
            "sha256": sha256_path(raw_path),
        },
        "physical_v1_source_sha256": sha256_path(physical_path),
        "candidate_sha256": sha256_path(candidate),
        "sprite_sha256": sha256_path(sprite),
        "normalize": normalized,
        "invariants": {
            "alpha_byte_identical_to_physical_v1": True,
            "partial_boundary_pixel_count": int(partial_boundary.sum()),
            "partial_boundary_rgb_byte_identical_to_physical_v1": True,
            "fully_opaque_pixel_count": int(fully_opaque.sum()),
            "fully_opaque_rgb_byte_identical_to_raw": True,
            "high_alpha_interior_raw_restored_pixels": int(high_alpha_interior.sum()),
            "visible_hot_magenta_pixels": int(
                ((hybrid[..., 3] > 0) & hot_magenta(hybrid[..., :3])).sum()
            ),
        },
    }


def main() -> None:
    args = parse_args()
    raw_overrides = overrides(args.raw_override)
    physical_root = args.physical_root or (
        ROOT / "rejected" / f"{args.character}_physical_unmix_v1"
    )
    report: dict[str, object] = {
        "method": "opaque_hybrid_v1",
        "character": args.character,
        "physical_root": str(physical_root),
        "parameters": {
            "high_alpha_threshold": HIGH_ALPHA_THRESHOLD,
            "high_alpha_fraction": round(HIGH_ALPHA_THRESHOLD / 255.0, 6),
            "edge_band_distance_native_pixels": EDGE_BAND_DISTANCE,
            "hot_magenta_saturation_u8": HOT_MAGENTA_SATURATION,
            "hot_magenta_hue_degrees_inclusive": list(HOT_MAGENTA_HUE_DEGREES),
            "hot_magenta_min_value_u8": HOT_MAGENTA_MIN_VALUE,
        },
        "variants": {},
    }
    variant_report = report["variants"]
    assert isinstance(variant_report, dict)
    for variant in VARIANTS:
        filename = raw_overrides.get(variant, f"{variant}_chroma_v1.png")
        variant_report[variant] = build_variant(
            args.character, variant, filename, physical_root
        )
        print(f"{variant}: {json.dumps(variant_report[variant])}")
    sheet = build_contact_sheet(args.character)
    report["contact_sheet"] = {"path": str(sheet), "sha256": sha256_path(sheet)}
    report_path = ROOT / "matte_reports" / f"{args.character}_opaque_hybrid_v1.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"contact_sheet: {sheet} sha256={sha256_path(sheet)}")
    print(f"report: {report_path} sha256={sha256_path(report_path)}")


if __name__ == "__main__":
    main()
