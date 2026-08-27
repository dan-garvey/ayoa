#!/usr/bin/env python3
"""Build Castor's approved physical-edge/raw-interior opaque hybrid sweep."""

from __future__ import annotations

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


CHARACTER = "castor_valebrand"
PHYSICAL_V1 = ROOT / "rejected" / "castor_valebrand_physical_unmix_v1"
REPORT = ROOT / "matte_reports" / "castor_valebrand_opaque_hybrid_v1.json"

HIGH_ALPHA_THRESHOLD = 200
EDGE_BAND_DISTANCE = 3.0
HOT_MAGENTA_SATURATION = 112
HOT_MAGENTA_HUE_DEGREES = (275, 335)
HOT_MAGENTA_MIN_VALUE = 16

RAW_FILES = {
    "neutral": "neutral_chroma_v1.png",
    "happy": "happy_chroma_v1.png",
    "concerned": "concerned_chroma_v2.png",
    "tense": "tense_chroma_v1.png",
    "skeptical": "skeptical_chroma_v1.png",
    "angry": "angry_chroma_v1.png",
    "sad": "sad_chroma_v1.png",
    "surprised": "surprised_chroma_v1.png",
}

EXPECTED_RAW_HASHES = {
    "neutral": "0200b16aafa0d13b78f1609eb7b4ecd09ab12bf7ff04761da6397c0da789bb5f",
    "happy": "898ec5114d22d871b97215b40d98043db82f37cb668773b5b2a5ab6a7f5587f3",
    "concerned": "83da5d1695d2afc4b92f4383b7ed47148b572a3100803da3c91e1e07652ac4ee",
    "tense": "bf7afdbaa3db5bed33cdc57e08b87772df6afa8aa083815c2ca5b90c218a61aa",
    "skeptical": "95436faff0dc3f64c175a1243b4b9755f72d0b6457ab27e2c28a2f10d54046e8",
    "angry": "35e69b56300cdb9d1b8abd33e4fc73a6c690b9f727285bce5c432d0974222988",
    "sad": "0be2cf633e5be5ca67ad0ca8fa2c753c0084edb692461d0096f9c9e99f1246a4",
    "surprised": "c1569fe3d065eeb7b868ce412459cee142c35f4e6ce1b5de4618f70e72af601b",
}

EXPECTED_PHYSICAL_V1_HASHES = {
    "neutral": "a29cea0956a1c3c8ffcc08b9732c1009203fd768a46fee6f6142628aa7daeb8e",
    "happy": "6607057e6ad1680accff24268ad2c3b68085cd478d81440618c52a6cdfcdbbab",
    "concerned": "99585240c33825785c0f42254be8c47a8eddd974b03322e1ff11a483698b9f3b",
    "tense": "4d542595edfbf0c8207e50ff309c3d0585dc4d458d3496cc55d2bac1b00fde0c",
    "skeptical": "2bbe449a24a15c1d58166c0e119463a3b214f77de6aec715cfd1b70c7c822225",
    "angry": "79c791e40b2815b5027aeb2c464468ed42eab6ddee1bb5d96229d1e5f17d0480",
    "sad": "334d5fac9233692230c6e1c220420f892865036901b927e264b3dff4dffbd3d5",
    "surprised": "c653aca6bca6439e3f131ba332972f40a03f86af5cf245acc9012c50a614cda7",
}


def hot_magenta(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16) * 2
    return (
        (hsv[..., 1] >= HOT_MAGENTA_SATURATION)
        & (hue >= HOT_MAGENTA_HUE_DEGREES[0])
        & (hue <= HOT_MAGENTA_HUE_DEGREES[1])
        & (hsv[..., 2] >= HOT_MAGENTA_MIN_VALUE)
    )


def burgundy_mask(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = [rgb[..., channel].astype(np.int16) for channel in range(3)]
    return (
        (red >= 45)
        & (red <= 190)
        & (red - green >= 18)
        & (red - blue >= 10)
        & (green <= 110)
        & (blue <= 120)
    )


def build_variant(variant: str) -> dict[str, object]:
    raw_path = ROOT / "generation_raw" / CHARACTER / RAW_FILES[variant]
    physical_path = PHYSICAL_V1 / "candidates" / f"{variant}.png"
    if sha256_path(raw_path) != EXPECTED_RAW_HASHES[variant]:
        raise RuntimeError(f"accepted raw hash drifted: {raw_path}")
    if sha256_path(physical_path) != EXPECTED_PHYSICAL_V1_HASHES[variant]:
        raise RuntimeError(f"physical-v1 candidate hash drifted: {physical_path}")

    raw = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.uint8)
    physical = np.asarray(Image.open(physical_path).convert("RGBA"), dtype=np.uint8)
    if physical.shape[:2] != raw.shape[:2]:
        raise RuntimeError(f"raw/physical geometry mismatch for {variant}")

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
    hybrid[..., :3][fully_opaque | high_alpha_interior] = raw[
        fully_opaque | high_alpha_interior
    ]
    if not np.array_equal(hybrid[..., 3], physical[..., 3]):
        raise RuntimeError(f"alpha changed for {variant}")
    if not np.array_equal(
        hybrid[..., :3][partial_boundary], physical[..., :3][partial_boundary]
    ):
        raise RuntimeError(f"partial edge RGB changed for {variant}")
    if not np.array_equal(hybrid[..., :3][fully_opaque], raw[fully_opaque]):
        raise RuntimeError(f"opaque RGB differs from raw for {variant}")

    candidate = ROOT / "candidates" / CHARACTER / f"{variant}.png"
    sprite = ROOT / "sprites" / CHARACTER / f"{variant}.png"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hybrid, mode="RGBA").save(candidate, optimize=True)
    normalize_metadata = normalize_candidate(candidate, sprite)

    burgundy = burgundy_mask(raw)
    physical_exact = np.all(physical[..., :3] == raw, axis=2)
    hybrid_exact = np.all(hybrid[..., :3] == raw, axis=2)
    return {
        "accepted_raw": {
            "filename": RAW_FILES[variant],
            "sha256": sha256_path(raw_path),
        },
        "physical_v1_source_sha256": sha256_path(physical_path),
        "candidate_sha256": sha256_path(candidate),
        "sprite_sha256": sha256_path(sprite),
        "normalize": normalize_metadata,
        "invariants": {
            "alpha_byte_identical_to_physical_v1": True,
            "partial_boundary_pixel_count": int(partial_boundary.sum()),
            "partial_boundary_rgb_byte_identical_to_physical_v1": True,
            "fully_opaque_pixel_count": int(fully_opaque.sum()),
            "fully_opaque_rgb_byte_identical_to_raw": True,
            "high_alpha_interior_raw_restored_pixels": int(
                high_alpha_interior.sum()
            ),
            "visible_hot_magenta_pixels": int(
                ((hybrid[..., 3] > 0) & hot_magenta(hybrid[..., :3])).sum()
            ),
        },
        "burgundy_evidence": {
            "pixel_count": int(burgundy.sum()),
            "raw_median_rgb": np.median(raw[burgundy], axis=0).astype(int).tolist(),
            "physical_v1_median_rgba": np.median(
                physical[burgundy], axis=0
            ).astype(int).tolist(),
            "hybrid_median_rgba": np.median(hybrid[burgundy], axis=0)
            .astype(int)
            .tolist(),
            "physical_v1_raw_rgb_exact_pixels": int(physical_exact[burgundy].sum()),
            "hybrid_raw_rgb_exact_pixels": int(hybrid_exact[burgundy].sum()),
        },
    }


def main() -> None:
    report = {
        "method": "opaque_hybrid_v1",
        "character": CHARACTER,
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
    for variant in VARIANTS:
        report["variants"][variant] = build_variant(variant)
        print(f"{variant}: {json.dumps(report['variants'][variant])}")
    sheet = build_contact_sheet(CHARACTER)
    report["contact_sheet"] = {
        "path": str(sheet),
        "sha256": sha256_path(sheet),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"contact_sheet: {sheet} sha256={sha256_path(sheet)}")
    print(f"report: {REPORT} sha256={sha256_path(REPORT)}")


if __name__ == "__main__":
    main()
