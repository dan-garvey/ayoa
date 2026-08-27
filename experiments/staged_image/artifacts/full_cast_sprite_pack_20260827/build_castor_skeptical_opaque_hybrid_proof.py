#!/usr/bin/env python3
"""Build the isolated Castor skeptical opaque-interior hybrid proof.

The physical-unmix v1 alpha and edge RGB are authoritative for screen removal.
Raw RGB is restored only for fully opaque pixels and for high-alpha pixels that
are more than three native pixels inside the matte.  Partial-alpha pixels in
the three-pixel edge band therefore remain byte-identical to physical v1.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

from prepare_review_sprites import normalize_candidate, sha256_path


ROOT = Path(__file__).resolve().parent
PROOF = ROOT / "matte_proofs" / "castor_valebrand_skeptical_opaque_hybrid_v1"
RAW = ROOT / "generation_raw" / "castor_valebrand" / "skeptical_chroma_v1.png"
PHYSICAL_V1 = (
    ROOT
    / "rejected"
    / "castor_valebrand_physical_unmix_v1"
    / "candidates"
    / "skeptical.png"
)
CONNECTED_V3 = (
    ROOT
    / "rejected"
    / "castor_valebrand_connected_key_v3"
    / "candidates"
    / "skeptical.png"
)

EXPECTED_HASHES = {
    RAW: "95436faff0dc3f64c175a1243b4b9755f72d0b6457ab27e2c28a2f10d54046e8",
    PHYSICAL_V1: "2bbe449a24a15c1d58166c0e119463a3b214f77de6aec715cfd1b70c7c822225",
    CONNECTED_V3: "e09c2e3278fce672a112c4257179e199431b6f1af35584a6f2275c243de95595",
}

HIGH_ALPHA_THRESHOLD = 200
EDGE_BAND_DISTANCE = 3.0
HOT_MAGENTA_SATURATION = 112
HOT_MAGENTA_HUE_DEGREES = (275, 335)
HOT_MAGENTA_MIN_VALUE = 16

FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
DARK = (31, 36, 44)
LIGHT = (235, 238, 242)

METHODS = [
    ("raw_rgb_v1_alpha", "RAW RGB + V1 ALPHA", "canonical color / untreated spill"),
    ("physical_unmix_v1", "PHYSICAL-UNMIX V1", "clean edge / burgundy shifted"),
    ("connected_key_v3", "CONNECTED-KEY V3", "raw interior / neon contour"),
    ("opaque_hybrid_v1", "OPAQUE HYBRID", "v1 edge / raw high-alpha interior"),
]

DETAIL_REGIONS = [
    ("hair + face edge", (360, 25, 680, 300)),
    ("hand + coat edge", (130, 300, 480, 620)),
    ("burgundy coat lining", (285, 650, 610, 1200)),
    ("two burgundy guard gems", (540, 650, 780, 900)),
]

COLOR_REGIONS = {
    "coat_lining": (285, 650, 610, 1200),
    "upper_guard_gem": (590, 720, 640, 775),
    "lower_guard_gem": (695, 795, 745, 855),
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


def split_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    result = Image.new("RGB", size, LIGHT)
    ImageDraw.Draw(result).rectangle((0, 0, width // 2, height), fill=DARK)
    return result


def crop_visible(rgba: Image.Image) -> Image.Image:
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        raise RuntimeError("empty proof sprite")
    return rgba.crop(bbox)


def fit_rgba(rgba: Image.Image, maximum: tuple[int, int]) -> Image.Image:
    fitted = rgba.copy()
    fitted.thumbnail(maximum, Image.Resampling.LANCZOS)
    return fitted


def build_full_comparison(sprites: dict[str, Path], output: Path) -> None:
    cell_width, cell_height = 600, 1200
    header_height, art_height = 94, 1010
    sheet = Image.new("RGB", (cell_width * len(METHODS), cell_height), (12, 15, 20))
    title_font = ImageFont.truetype(str(FONT_BOLD), 28)
    small_font = ImageFont.truetype(str(FONT_REGULAR), 18)
    for index, (key, title, subtitle) in enumerate(METHODS):
        cell = Image.new("RGB", (cell_width, cell_height), (12, 15, 20))
        art = split_background((cell_width, art_height))
        with Image.open(sprites[key]) as source:
            rgba = fit_rgba(crop_visible(source.convert("RGBA")), (560, 970))
        art.paste(
            rgba,
            ((cell_width - rgba.width) // 2, art_height - rgba.height - 12),
            rgba,
        )
        cell.paste(art, (0, header_height))
        draw = ImageDraw.Draw(cell)
        draw.text((18, 14), title, fill=(255, 255, 255), font=title_font)
        draw.text((18, 52), subtitle, fill=(176, 185, 196), font=small_font)
        draw.text((18, 1120), "dark / light", fill=(200, 207, 216), font=small_font)
        sheet.paste(cell, (index * cell_width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def build_detail_comparison(candidates: dict[str, Image.Image], output: Path) -> None:
    cell_width, header_height, detail_height = 600, 82, 300
    sheet_height = header_height + detail_height * len(DETAIL_REGIONS)
    sheet = Image.new("RGB", (cell_width * len(METHODS), sheet_height), (12, 15, 20))
    title_font = ImageFont.truetype(str(FONT_BOLD), 25)
    label_font = ImageFont.truetype(str(FONT_BOLD), 18)
    small_font = ImageFont.truetype(str(FONT_REGULAR), 15)
    for column, (key, title, _) in enumerate(METHODS):
        x_base = column * cell_width
        ImageDraw.Draw(sheet).text(
            (x_base + 16, 20), title, fill=(255, 255, 255), font=title_font
        )
        for row, (region_name, bbox) in enumerate(DETAIL_REGIONS):
            y_base = header_height + row * detail_height
            panel = split_background((cell_width, detail_height))
            crop = candidates[key].crop(bbox)
            fitted = fit_rgba(crop, (270, 235))
            y = 10 + (235 - fitted.height) // 2
            panel.paste(fitted, ((300 - fitted.width) // 2, y), fitted)
            panel.paste(fitted, (300 + (300 - fitted.width) // 2, y), fitted)
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 250, cell_width, detail_height), fill=(12, 15, 20))
            draw.text((12, 254), region_name, fill=(255, 255, 255), font=label_font)
            draw.text(
                (375, 258), "dark / light", fill=(176, 185, 196), font=small_font
            )
            sheet.paste(panel, (x_base, y_base))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def region_evidence(
    raw: np.ndarray, candidates: dict[str, np.ndarray]
) -> dict[str, object]:
    red, green, blue = [raw[..., channel].astype(np.int16) for channel in range(3)]
    burgundy = (
        (red >= 45)
        & (red <= 190)
        & (red - green >= 18)
        & (red - blue >= 10)
        & (green <= 110)
        & (blue <= 120)
    )
    result: dict[str, object] = {}
    for name, (x0, y0, x1, y1) in COLOR_REGIONS.items():
        roi = np.zeros(burgundy.shape, dtype=bool)
        roi[y0:y1, x0:x1] = True
        mask = burgundy & roi
        if not np.any(mask):
            raise RuntimeError(f"empty burgundy evidence mask: {name}")
        entry: dict[str, object] = {
            "roi_xyxy": [x0, y0, x1, y1],
            "pixel_count": int(mask.sum()),
            "raw_median_rgb": np.median(raw[mask], axis=0).astype(int).tolist(),
        }
        for key, rgba in candidates.items():
            values = rgba[mask]
            entry[key] = {
                "median_rgba": np.median(values, axis=0).astype(int).tolist(),
                "raw_rgb_exact_pixels": int(
                    np.all(values[:, :3] == raw[mask], axis=1).sum()
                ),
            }
        result[name] = entry
    return result


def main() -> None:
    for source, expected in EXPECTED_HASHES.items():
        actual = sha256_path(source)
        if actual != expected:
            raise RuntimeError(f"source hash changed: {source}: {actual} != {expected}")

    raw = np.asarray(Image.open(RAW).convert("RGB"), dtype=np.uint8)
    physical = np.asarray(Image.open(PHYSICAL_V1).convert("RGBA"), dtype=np.uint8)
    connected = np.asarray(Image.open(CONNECTED_V3).convert("RGBA"), dtype=np.uint8)
    if physical.shape[:2] != raw.shape[:2] or connected.shape[:2] != raw.shape[:2]:
        raise RuntimeError("proof sources do not share native geometry")

    alpha = physical[..., 3]
    distance_inside = distance_transform_edt(alpha > 0)
    partial_boundary = (
        (alpha > 0)
        & (alpha < 255)
        & (distance_inside <= EDGE_BAND_DISTANCE)
    )
    raw_hot = hot_magenta(raw)
    fully_opaque = alpha == 255
    high_alpha_interior = (
        (alpha >= HIGH_ALPHA_THRESHOLD)
        & (alpha < 255)
        & (distance_inside > EDGE_BAND_DISTANCE)
        & ~raw_hot
    )
    restore_raw = fully_opaque | high_alpha_interior

    hybrid = physical.copy()
    hybrid[..., :3][restore_raw] = raw[restore_raw]
    if not np.array_equal(hybrid[..., 3], alpha):
        raise RuntimeError("hybrid alpha diverged from physical v1")
    if not np.array_equal(
        hybrid[..., :3][partial_boundary], physical[..., :3][partial_boundary]
    ):
        raise RuntimeError("a partial-alpha boundary RGB pixel changed")
    if not np.array_equal(hybrid[..., :3][fully_opaque], raw[fully_opaque]):
        raise RuntimeError("a fully opaque RGB pixel differs from raw")

    raw_rgba = np.dstack([raw, alpha])
    arrays = {
        "raw_rgb_v1_alpha": raw_rgba,
        "physical_unmix_v1": physical,
        "connected_key_v3": connected,
        "opaque_hybrid_v1": hybrid,
    }

    candidate_dir = PROOF / "candidates"
    sprite_dir = PROOF / "sprites"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    sprite_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, Path] = {}
    sprites: dict[str, Path] = {}
    for key, rgba in arrays.items():
        candidate = candidate_dir / f"{key}.png"
        sprite = sprite_dir / f"{key}.png"
        Image.fromarray(rgba, mode="RGBA").save(candidate, optimize=True)
        normalize_candidate(candidate, sprite)
        candidates[key] = candidate
        sprites[key] = sprite

    full_proof = PROOF / "castor_skeptical_opaque_hybrid_comparison.png"
    detail_proof = PROOF / "castor_skeptical_opaque_hybrid_details.png"
    build_full_comparison(sprites, full_proof)
    build_detail_comparison(
        {key: Image.fromarray(value, mode="RGBA") for key, value in arrays.items()},
        detail_proof,
    )

    visible_hot_counts = {
        key: int(((rgba[..., 3] > 0) & hot_magenta(rgba[..., :3])).sum())
        for key, rgba in arrays.items()
    }
    metadata = {
        "method": "castor_skeptical_opaque_hybrid_v1",
        "parameters": {
            "high_alpha_threshold": HIGH_ALPHA_THRESHOLD,
            "high_alpha_fraction": round(HIGH_ALPHA_THRESHOLD / 255.0, 6),
            "edge_band_distance_native_pixels": EDGE_BAND_DISTANCE,
            "hot_magenta_saturation_u8": HOT_MAGENTA_SATURATION,
            "hot_magenta_hue_degrees_inclusive": list(HOT_MAGENTA_HUE_DEGREES),
            "hot_magenta_min_value_u8": HOT_MAGENTA_MIN_VALUE,
        },
        "invariants": {
            "alpha_byte_identical_to_physical_v1": bool(
                np.array_equal(hybrid[..., 3], physical[..., 3])
            ),
            "partial_boundary_pixel_count": int(partial_boundary.sum()),
            "partial_boundary_rgb_byte_identical_to_physical_v1": bool(
                np.array_equal(
                    hybrid[..., :3][partial_boundary],
                    physical[..., :3][partial_boundary],
                )
            ),
            "fully_opaque_pixel_count": int(fully_opaque.sum()),
            "fully_opaque_rgb_byte_identical_to_raw": bool(
                np.array_equal(hybrid[..., :3][fully_opaque], raw[fully_opaque])
            ),
            "high_alpha_interior_raw_restored_pixels": int(
                high_alpha_interior.sum()
            ),
            "visible_hot_magenta_pixels": visible_hot_counts,
        },
        "color_regions": region_evidence(raw, arrays),
        "files": {
            "source_raw": {"path": str(RAW), "sha256": sha256_path(RAW)},
            "source_physical_v1": {
                "path": str(PHYSICAL_V1),
                "sha256": sha256_path(PHYSICAL_V1),
            },
            "source_connected_v3": {
                "path": str(CONNECTED_V3),
                "sha256": sha256_path(CONNECTED_V3),
            },
            "candidates": {
                key: {"path": str(path), "sha256": sha256_path(path)}
                for key, path in candidates.items()
            },
            "sprites": {
                key: {"path": str(path), "sha256": sha256_path(path)}
                for key, path in sprites.items()
            },
            "full_proof": {
                "path": str(full_proof),
                "sha256": sha256_path(full_proof),
            },
            "detail_proof": {
                "path": str(detail_proof),
                "sha256": sha256_path(detail_proof),
            },
        },
    }
    metadata_path = PROOF / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"metadata: {metadata_path} sha256={sha256_path(metadata_path)}")


if __name__ == "__main__":
    main()
