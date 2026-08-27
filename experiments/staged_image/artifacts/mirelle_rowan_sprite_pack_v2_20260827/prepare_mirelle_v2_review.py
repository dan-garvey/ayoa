#!/usr/bin/env python3
"""Build Mirelle's color-safe RGBA v2 sweep from approved chroma repairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parent
LABELS = [
    "neutral",
    "happy",
    "concerned",
    "tense",
    "skeptical",
    "angry",
    "sad",
    "surprised",
]
SOURCES = {
    "neutral": "grafts/mirelle_voss/neutral_canonical_metal_repair_v1.png",
    "happy": "grafts/mirelle_voss/happy_upper_primary_metal_reference_graft_v2.png",
    "concerned": "generation_raw/mirelle_voss/concerned_chroma_v3.png",
    "tense": "grafts/mirelle_voss/tense_primary_metal_reference_graft_v2.png",
    "skeptical": "grafts/mirelle_voss/skeptical_primary_metal_reference_graft_v1.png",
    "angry": "grafts/mirelle_voss/angry_primary_metal_reference_graft_v1.png",
    "sad": "grafts/mirelle_voss/sad_canonical_metal_repair_v1.png",
    "surprised": "grafts/mirelle_voss/surprised_primary_metal_reference_graft_v1.png",
}
EXPECTED_SOURCE_HASHES = {
    "neutral": "08fc7d8223ed162039d30d7789031748694ec60510921a4c714033adf650a407",
    "happy": "d84dc6d28c8864c3cb7e2dfb7190e5e6460e7b797d1b59dedb246341ca9f48a3",
    "concerned": "7ca5fe3c5e1a21c0c3c5be7069eb9e690951921a51851ae81b8a187e68e84966",
    "tense": "d1a623ef7fb95e28a890fe7c0c4e235f02da7a6cfbbe26c60f3e13c1cb5a54cc",
    "skeptical": "7b827787a0dad57e5e937936559871beaddcfbd7919a9dfab63c1459fb850fb5",
    "angry": "82f3287b4358e6ca06f2dd70aa3c91193f826866415c303c5aacca97dae0d7e2",
    "sad": "512d710eeba7f5accddd73e27dd2801afd4dd6ad2cab070acd763f11bed31ca6",
    "surprised": "1a729f2df227282599fb09225dd71d14e666cd6eefb131366dea4dd462632a33",
}

PHYSICAL_ROOT = ROOT / "rejected/mirelle_voss_physical_unmix_v1"
CANDIDATE_ROOT = ROOT / "candidates/mirelle_voss"
SPRITE_ROOT = ROOT / "sprites/mirelle_voss"
FINAL_SHEET = ROOT / "contact_sheets/mirelle_voss_complete_sweep_v2.png"
PHYSICAL_SHEET = (
    ROOT
    / "contact_sheets/rejected/mirelle_voss_complete_sweep_physical_unmix_v1.png"
)
REPORT = ROOT / "matte_reports/mirelle_voss_v2_opaque_hybrid.json"

CANVAS_SIZE = (1100, 1500)
TARGET_HEIGHT = 1420
MAX_WIDTH = 1060
BASELINE_Y = 1480
ALPHA_NOISE_CUTOFF = 48
HIGH_ALPHA_THRESHOLD = 200
EDGE_BAND_DISTANCE = 3.0
HOT_MAGENTA_SATURATION = 112
HOT_MAGENTA_HUE_DEGREES = (275, 335)
HOT_MAGENTA_MIN_VALUE = 16
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_magenta_matte(source: Path, destination: Path) -> dict[str, object]:
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32)
    border = np.concatenate(
        [
            rgb[:12].reshape(-1, 3),
            rgb[-12:].reshape(-1, 3),
            rgb[:, :12].reshape(-1, 3),
            rgb[:, -12:].reshape(-1, 3),
        ],
        axis=0,
    )
    key = np.median(border, axis=0)
    key_dominance = float((key[0] + key[2]) * 0.5 - key[1])
    if key_dominance < 96:
        raise RuntimeError(f"{source} lacks a strong magenta border: {key}")
    dominance = (rgb[..., 0] + rgb[..., 2]) * 0.5 - rgb[..., 1]
    alpha = np.clip(1.0 - dominance / key_dominance, 0.0, 1.0)
    distance = np.max(np.abs(rgb - key), axis=2)
    alpha[distance <= 3] = 0.0
    alpha[dominance <= 4] = 1.0

    component_mask = (alpha >= 0.035).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        component_mask, connectivity=8
    )
    keep = np.zeros_like(component_mask)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= 100:
            keep[labels == index] = 1
    alpha *= keep
    alpha[alpha < 0.025] = 0.0
    alpha[alpha > 0.985] = 1.0
    denominator = np.maximum(alpha[..., None], 0.06)
    foreground = (rgb - (1.0 - alpha[..., None]) * key) / denominator
    foreground = np.clip(foreground, 0, 255)
    foreground[alpha >= 0.985] = rgb[alpha >= 0.985]
    foreground[alpha <= 0] = 0
    rgba = np.dstack(
        [foreground.astype(np.uint8), np.round(alpha * 255).astype(np.uint8)]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(destination, optimize=True)
    return {
        "key_rgb": [float(value) for value in key],
        "key_dominance": key_dominance,
        "physical_candidate_sha256": sha256(destination),
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


def normalize(source: Path, destination: Path) -> dict[str, object]:
    rgba = Image.open(source).convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8).copy()
    alpha[alpha <= ALPHA_NOISE_CUTOFF] = 0
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    ys, xs = np.where(alpha >= 64)
    if not xs.size:
        raise RuntimeError(f"empty matte: {source}")
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    crop = rgba.crop(bbox)
    scale = min(TARGET_HEIGHT / crop.height, MAX_WIDTH / crop.width)
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    crop = crop.resize(size, Image.Resampling.LANCZOS)
    x = (CANVAS_SIZE[0] - crop.width) // 2
    y = BASELINE_Y - crop.height
    if x < 0 or y < 0 or x + crop.width > CANVAS_SIZE[0] or y + crop.height > CANVAS_SIZE[1]:
        raise RuntimeError(f"normalized sprite does not fit: {source} -> {size}")
    canvas = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    canvas.alpha_composite(crop, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
    return {
        "source_bbox": list(bbox),
        "scale": round(scale, 8),
        "placement": [x, y, crop.width, crop.height],
        "sha256": sha256(destination),
    }


def intentional_red(raw: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    red = raw[..., 0].astype(np.int16)
    green = raw[..., 1].astype(np.int16)
    blue = raw[..., 2].astype(np.int16)
    return (
        (alpha >= HIGH_ALPHA_THRESHOLD)
        & (red >= 40)
        & ((red - green) >= 14)
        & ((red - blue) >= 8)
        & ~hot_magenta(raw)
    )


def median_rgb(rgb: np.ndarray, mask: np.ndarray) -> list[int] | None:
    if not np.any(mask):
        return None
    return [int(value) for value in np.median(rgb[mask], axis=0)]


def build_hybrid(
    raw_path: Path, physical_path: Path, candidate: Path, sprite: Path
) -> dict[str, object]:
    raw = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.uint8)
    physical = np.asarray(Image.open(physical_path).convert("RGBA"), dtype=np.uint8)
    if raw.shape[:2] != physical.shape[:2]:
        raise RuntimeError(f"raw/physical geometry mismatch: {raw_path}")
    alpha = physical[..., 3]
    distance_inside = distance_transform_edt(alpha > 0)
    partial_boundary = (
        (alpha > 0) & (alpha < 255) & (distance_inside <= EDGE_BAND_DISTANCE)
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
    if not np.array_equal(hybrid[..., 3], alpha):
        raise RuntimeError("hybrid alpha changed")
    if not np.array_equal(
        hybrid[..., :3][partial_boundary], physical[..., :3][partial_boundary]
    ):
        raise RuntimeError("hybrid partial-edge RGB changed")
    if not np.array_equal(hybrid[..., :3][fully_opaque], raw[fully_opaque]):
        raise RuntimeError("hybrid opaque RGB differs from raw")
    visible_hot = (alpha > 0) & hot_magenta(hybrid[..., :3])
    if int(visible_hot.sum()) != 0:
        raise RuntimeError(f"visible hot-magenta pixels remain: {visible_hot.sum()}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hybrid, mode="RGBA").save(candidate, optimize=True)
    normalized = normalize(candidate, sprite)
    red_mask = intentional_red(raw, alpha)
    exact_red = np.all(hybrid[..., :3] == raw, axis=2) & red_mask
    return {
        "candidate_sha256": sha256(candidate),
        "sprite_sha256": sha256(sprite),
        "normalize": normalized,
        "invariants": {
            "alpha_byte_identical_to_physical_v1": True,
            "partial_boundary_pixel_count": int(partial_boundary.sum()),
            "partial_boundary_rgb_byte_identical_to_physical_v1": True,
            "fully_opaque_pixel_count": int(fully_opaque.sum()),
            "fully_opaque_rgb_byte_identical_to_raw": True,
            "high_alpha_interior_raw_restored_pixels": int(
                high_alpha_interior.sum()
            ),
            "visible_hot_magenta_pixels": 0,
        },
        "intentional_red_audit": {
            "pixels": int(red_mask.sum()),
            "exact_raw_rgb_pixels": int(exact_red.sum()),
            "exact_raw_rgb_fraction": round(
                float(exact_red.sum()) / max(1, int(red_mask.sum())), 8
            ),
            "raw_median_rgb": median_rgb(raw, red_mask),
            "physical_v1_median_rgb": median_rgb(physical[..., :3], red_mask),
            "hybrid_median_rgb": median_rgb(hybrid[..., :3], red_mask),
        },
    }


def build_sheet(sprite_root: Path, output: Path, title_suffix: str) -> None:
    columns, rows = 4, 2
    cell_width, cell_height = 480, 650
    art_width, art_height = 450, 575
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (18, 22, 28))
    label_font = ImageFont.truetype(str(FONT), 28)
    small_font = ImageFont.truetype(str(FONT), 17)
    for index, label in enumerate(LABELS):
        rgba = Image.open(sprite_root / f"{label}.png").convert("RGBA")
        rgba.thumbnail((art_width, art_height), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (cell_width, cell_height), (235, 238, 242))
        ImageDraw.Draw(cell).rectangle((0, 0, cell_width // 2, cell_height), fill=(31, 36, 44))
        cell.paste(
            rgba,
            ((cell_width - rgba.width) // 2, 8 + (art_height - rgba.height)),
            rgba,
        )
        draw = ImageDraw.Draw(cell)
        draw.rectangle((0, 590, cell_width, cell_height), fill=(12, 15, 20))
        draw.text((16, 598), label.upper(), fill=(255, 255, 255), font=label_font)
        draw.text(
            (16, 632),
            f"dark / light · {title_suffix}",
            fill=(165, 174, 186),
            font=small_font,
        )
        row, column = divmod(index, columns)
        sheet.paste(cell, (column * cell_width, row * cell_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def main() -> None:
    report: dict[str, object] = {
        "method": "physical_alpha_partial_edges_plus_raw_opaque_high_alpha_interiors",
        "character": "mirelle_voss",
        "parameters": {
            "high_alpha_threshold": HIGH_ALPHA_THRESHOLD,
            "edge_band_distance_native_pixels": EDGE_BAND_DISTANCE,
            "hot_magenta_saturation_u8": HOT_MAGENTA_SATURATION,
            "hot_magenta_hue_degrees": list(HOT_MAGENTA_HUE_DEGREES),
            "hot_magenta_min_value_u8": HOT_MAGENTA_MIN_VALUE,
            "canvas": list(CANVAS_SIZE),
            "target_height": TARGET_HEIGHT,
            "maximum_width": MAX_WIDTH,
            "baseline_y": BASELINE_Y,
        },
        "variants": {},
    }
    for label in LABELS:
        raw_path = ROOT / SOURCES[label]
        actual_hash = sha256(raw_path)
        if actual_hash != EXPECTED_SOURCE_HASHES[label]:
            raise RuntimeError(
                f"source hash changed for {label}: {actual_hash} != {EXPECTED_SOURCE_HASHES[label]}"
            )
        physical_candidate = PHYSICAL_ROOT / "candidates" / f"{label}.png"
        physical_sprite = PHYSICAL_ROOT / "sprites" / f"{label}.png"
        physical_report = physical_magenta_matte(raw_path, physical_candidate)
        physical_normalize = normalize(physical_candidate, physical_sprite)
        final_candidate = CANDIDATE_ROOT / f"{label}.png"
        final_sprite = SPRITE_ROOT / f"{label}.png"
        hybrid_report = build_hybrid(
            raw_path, physical_candidate, final_candidate, final_sprite
        )
        physical_alpha = np.asarray(
            Image.open(physical_sprite).convert("RGBA").getchannel("A"), dtype=np.uint8
        )
        final_alpha = np.asarray(
            Image.open(final_sprite).convert("RGBA").getchannel("A"), dtype=np.uint8
        )
        if not np.array_equal(physical_alpha, final_alpha):
            raise RuntimeError(f"normalized alpha changed for {label}")
        report["variants"][label] = {
            "selected_chroma": {
                "path": SOURCES[label],
                "sha256": actual_hash,
                "mode": Image.open(raw_path).mode,
                "size": list(Image.open(raw_path).size),
            },
            "physical_v1": {
                **physical_report,
                "candidate_path": str(physical_candidate.relative_to(ROOT)),
                "sprite_path": str(physical_sprite.relative_to(ROOT)),
                "normalize": physical_normalize,
            },
            "selected_hybrid": {
                "candidate_path": str(final_candidate.relative_to(ROOT)),
                "sprite_path": str(final_sprite.relative_to(ROOT)),
                **hybrid_report,
                "normalized_alpha_byte_identical_to_physical_v1": True,
            },
        }
        print(f"{label}: {json.dumps(report['variants'][label])}")

    build_sheet(PHYSICAL_ROOT / "sprites", PHYSICAL_SHEET, "rejected physical v1")
    build_sheet(SPRITE_ROOT, FINAL_SHEET, "opaque-hybrid v2")
    report["contact_sheets"] = {
        "rejected_physical_v1": {
            "path": str(PHYSICAL_SHEET.relative_to(ROOT)),
            "sha256": sha256(PHYSICAL_SHEET),
        },
        "selected_v2": {
            "path": str(FINAL_SHEET.relative_to(ROOT)),
            "sha256": sha256(FINAL_SHEET),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"selected_sheet={FINAL_SHEET} sha256={sha256(FINAL_SHEET)}")
    print(f"rejected_sheet={PHYSICAL_SHEET} sha256={sha256(PHYSICAL_SHEET)}")
    print(f"report={REPORT} sha256={sha256(REPORT)}")


if __name__ == "__main__":
    main()
