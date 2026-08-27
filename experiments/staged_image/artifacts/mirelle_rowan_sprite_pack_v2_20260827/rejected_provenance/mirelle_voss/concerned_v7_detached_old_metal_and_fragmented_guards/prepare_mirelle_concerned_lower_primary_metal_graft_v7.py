#!/usr/bin/env python3
"""Build Mirelle concerned v7 from the semantic v2 base, with rigid metal only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "generation_raw/mirelle_voss/concerned_chroma_v2.png"
REJECTED_V6 = (
    ROOT / "grafts/mirelle_voss/concerned_lower_spear_reference_graft_v6.png"
)
METAL = ROOT / "components/mirelle_voss/canonical_primary_metal_rgba_v1.png"
DEFERRED_STREAMERS = (
    ROOT / "components/mirelle_voss/canonical_primary_streamers_rgba_v1.png"
)
MATERIAL_SPLIT_METADATA = (
    ROOT / "component_metadata/mirelle_canonical_primary_material_split_v1.json"
)

APPROVED_COMPARISONS = {
    "neutral": {
        "path": ROOT / "grafts/mirelle_voss/neutral_canonical_metal_repair_v1.png",
        "sha256": "08fc7d8223ed162039d30d7789031748694ec60510921a4c714033adf650a407",
        "scale": 0.6657481436152197,
        "shaft_roi": [285, 340, 330, 900],
        "centerline": [0.02423010630888387, 302.9033232661395],
    },
    "happy": {
        "path": ROOT / "grafts/mirelle_voss/happy_upper_primary_metal_reference_graft_v2.png",
        "sha256": "d84dc6d28c8864c3cb7e2dfb7190e5e6460e7b797d1b59dedb246341ca9f48a3",
        "scale": 0.3664104881802468,
        "shaft_roi": [220, 300, 270, 1100],
        "centerline": [0.02811105822639962, 237.63454621788554],
    },
    "skeptical": {
        "path": ROOT / "grafts/mirelle_voss/skeptical_primary_metal_reference_graft_v1.png",
        "sha256": "7b827787a0dad57e5e937936559871beaddcfbd7919a9dfab63c1459fb850fb5",
        "scale": 0.4474436248930962,
        "shaft_roi": [285, 350, 360, 900],
        "centerline": [0.05909481448100204, 295.3394375458492],
    },
    "sad": {
        "path": ROOT / "grafts/mirelle_voss/sad_canonical_metal_repair_v1.png",
        "sha256": "512d710eeba7f5accddd73e27dd2801afd4dd6ad2cab070acd763f11bed31ca6",
        "scale": 0.4403658282992694,
        "shaft_roi": [365, 300, 420, 900],
        "centerline": [0.01831844956702265, 383.0554297681337],
    },
}

OUTPUT = (
    ROOT / "grafts/mirelle_voss/concerned_lower_primary_metal_reference_graft_v7.png"
)
CLEANED = (
    ROOT / "grafts/mirelle_voss/concerned_old_lower_primary_cleaned_base_v7.png"
)
OLD_HEAD_MASK = (
    ROOT / "masks/mirelle_voss/concerned_old_lower_primary_cleanup_mask_v7.png"
)
PATCH_MASK = (
    ROOT / "masks/mirelle_voss/concerned_lower_canonical_metal_patch_mask_v7.png"
)
UPPER_PROP_MASK = (
    ROOT
    / "masks/mirelle_voss/concerned_upper_cap_tassel_shaft_grip_exact_mask_v7.png"
)
BODY_MASK = (
    ROOT / "masks/mirelle_voss/concerned_body_face_costume_exact_mask_v7.png"
)
OCCLUSION_MASK = (
    ROOT / "masks/mirelle_voss/concerned_base_subject_occlusion_restore_mask_v7.png"
)
AGGREGATE_MASK = (
    ROOT / "masks/mirelle_voss/concerned_lower_primary_aggregate_saved_mask_v7.png"
)
CHANGED_MASK = (
    ROOT / "masks/mirelle_voss/concerned_lower_primary_changed_mask_v7.png"
)
FULL_PROOF = ROOT / "component_proofs/mirelle_concerned_lower_primary_full_v7.png"
CLOSE_PROOF = ROOT / "component_proofs/mirelle_concerned_lower_primary_close_v7.png"
MASK_PROOF = ROOT / "mask_proofs/mirelle_concerned_lower_primary_masks_v7.png"
METADATA = (
    ROOT / "component_metadata/mirelle_concerned_lower_primary_metal_graft_v7.json"
)

EXPECTED_HASHES = {
    BASE: "ac5d892873234388c83080cf42bc5885d8944666cb6315df07d0969206fbb517",
    REJECTED_V6: "83ba93aee746c73cb1f0da1c6eba18725bc5de55237320b98a4e31423b0c1c88",
    METAL: "4f4b369d7f445b79a44673b33aee8bcc763f754eb37a0d7bfaee38213bf73b6f",
    DEFERRED_STREAMERS: "9d7ff6a32a4ad46ac491652b40f0fa5f0a2c8fb4f88cd67c1ebdb1a81e6970dd",
    MATERIAL_SPLIT_METADATA: "6729001c5d692c4d7a31087870c7b930c354284535dc05acf8e9c8f460e03d30",
    **{
        value["path"]: value["sha256"]
        for value in APPROVED_COMPARISONS.values()
    },
}

SOURCE_SOCKET = np.array([200.0, 89.0], dtype=np.float64)
SOURCE_SHAFT_POINT = np.array([289.0, 13.0], dtype=np.float64)
MIN_ALPHA = 48
TARGET_SOCKET_Y = 1128.0
SHAFT_SAMPLE_Y = 1010.0
MAGENTA = np.array([255, 0, 255], dtype=np.uint8)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash fence failed for {path}: {actual} != {expected}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        path, optimize=True
    )


def bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if not xs.size:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def polygon_mask(
    size: tuple[int, int], polygons: list[list[tuple[int, int]]]
) -> np.ndarray:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def red_material(rgb: np.ndarray) -> np.ndarray:
    signed = rgb.astype(np.int16)
    return (
        (signed[..., 0] >= 65)
        & (signed[..., 0] >= signed[..., 1] + 22)
        & (signed[..., 0] >= signed[..., 2] + 8)
        & (signed[..., 1] <= 115)
        & (signed[..., 2] <= 135)
    )


def key_like(rgb: np.ndarray) -> np.ndarray:
    return (
        (rgb[..., 0] >= 220)
        & (rgb[..., 1] <= 35)
        & (rgb[..., 2] >= 220)
    )


def contiguous_runs(values: np.ndarray) -> list[np.ndarray]:
    if not values.size:
        return []
    return list(np.split(values, np.where(np.diff(values) > 1)[0] + 1))


def measured_shaft_width(
    image: np.ndarray,
    roi: list[int],
    slope: float,
    intercept: float,
) -> tuple[float, int, list[float]]:
    x0, y0, x1, y1 = roi
    red = red_material(image)
    widths: list[int] = []
    for y in range(y0, y1):
        center = slope * y + intercept
        left = max(x0, int(np.floor(center - 20)))
        right = min(x1, int(np.ceil(center + 21)))
        candidates = np.where(red[y, left:right])[0] + left
        runs = [run for run in contiguous_runs(candidates) if run.size >= 4]
        if not runs:
            continue
        run = min(runs, key=lambda value: abs(float(np.mean(value)) - center))
        if abs(float(np.mean(run)) - center) <= 5.0:
            widths.append(int(run.size))
    if len(widths) < 200:
        raise RuntimeError(f"insufficient isolated shaft rows: {len(widths)}")
    values = np.asarray(widths, dtype=np.float64)
    return (
        float(np.median(values)),
        int(values.size),
        [float(value) for value in np.percentile(values, [25, 75])],
    )


def fit_concerned_shaft(base: np.ndarray) -> dict[str, object]:
    red = red_material(base)
    row_y: list[float] = []
    row_x: list[float] = []
    row_width: list[int] = []
    for y in range(760, 1110):
        expected = 608.0 + (1128.0 - y) * (23.0 / 118.0)
        left = int(np.floor(expected - 16))
        right = int(np.ceil(expected + 17))
        candidates = np.where(red[y, left:right])[0] + left
        runs = [run for run in contiguous_runs(candidates) if run.size >= 4]
        if not runs:
            continue
        run = min(runs, key=lambda value: abs(float(np.mean(value)) - expected))
        row_y.append(float(y))
        row_x.append(float(np.mean(run)))
        row_width.append(int(run.size))
    if len(row_y) < 325:
        raise RuntimeError(f"insufficient concerned shaft rows: {len(row_y)}")

    y_values = np.asarray(row_y)
    x_values = np.asarray(row_x)
    widths = np.asarray(row_width)
    slope, intercept = np.polyfit(y_values, x_values, 1)
    residual = np.abs(x_values - (slope * y_values + intercept))
    keep = residual <= 2.5
    slope, intercept = np.polyfit(y_values[keep], x_values[keep], 1)
    kept_residual = np.abs(
        x_values[keep] - (slope * y_values[keep] + intercept)
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_rows": int(np.count_nonzero(keep)),
        "max_kept_row_residual_pixels": float(np.max(kept_residual)),
        "median_diameter_pixels": float(np.median(widths[keep])),
        "diameter_p25_p75_pixels": [
            float(value) for value in np.percentile(widths[keep], [25, 75])
        ],
        "fit_roi": [592, 760, 665, 1110],
    }


def derive_scale(base: np.ndarray, shaft: dict[str, object]) -> dict[str, object]:
    approved: dict[str, object] = {}
    ratios: list[float] = []
    for label, item in APPROVED_COMPARISONS.items():
        image = np.asarray(Image.open(item["path"]).convert("RGB"), dtype=np.uint8)
        width, rows, quartiles = measured_shaft_width(
            image,
            item["shaft_roi"],
            item["centerline"][0],
            item["centerline"][1],
        )
        ratio = float(item["scale"]) / width
        ratios.append(ratio)
        approved[label] = {
            "path": str(item["path"]),
            "sha256": sha256(item["path"]),
            "accepted_uniform_scale": item["scale"],
            "measured_median_shaft_diameter_pixels": width,
            "measurement_rows": rows,
            "diameter_p25_p75_pixels": quartiles,
            "scale_per_shaft_pixel": ratio,
        }
    robust_ratio = float(np.median(np.asarray(ratios, dtype=np.float64)))
    concerned_width = float(shaft["median_diameter_pixels"])
    scale = concerned_width * robust_ratio
    if not (0.38 <= scale <= 0.43):
        raise RuntimeError(f"concerned physical scale is outside expected range: {scale}")
    return {
        "method": (
            "concerned median shaft diameter multiplied by the robust median "
            "accepted head-scale-per-shaft-pixel ratio"
        ),
        "approved_comparisons": approved,
        "approved_ratio_values": ratios,
        "robust_median_scale_per_shaft_pixel": robust_ratio,
        "concerned_median_shaft_diameter_pixels": concerned_width,
        "derived_uniform_scale": scale,
        "neutral_large-head ratio is retained as an input but cannot dominate the median": True,
    }


def transform_metal(
    metal: np.ndarray,
    size: tuple[int, int],
    shaft: dict[str, object],
    scale: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    slope = float(shaft["slope"])
    intercept = float(shaft["intercept"])
    target_socket = np.array(
        [slope * TARGET_SOCKET_Y + intercept, TARGET_SOCKET_Y],
        dtype=np.float64,
    )
    target_shaft_point = np.array(
        [slope * SHAFT_SAMPLE_Y + intercept, SHAFT_SAMPLE_Y],
        dtype=np.float64,
    )
    source_vector = SOURCE_SHAFT_POINT - SOURCE_SOCKET
    target_vector = target_shaft_point - target_socket
    rotation = float(
        np.degrees(
            np.arctan2(target_vector[1], target_vector[0])
            - np.arctan2(source_vector[1], source_vector[0])
        )
    )
    radians = np.deg2rad(rotation)
    matrix = np.array(
        [
            [np.cos(radians) * scale, -np.sin(radians) * scale, 0.0],
            [np.sin(radians) * scale, np.cos(radians) * scale, 0.0],
        ],
        dtype=np.float64,
    )
    matrix[:, 2] = target_socket - matrix[:, :2] @ SOURCE_SOCKET

    alpha = metal[..., 3].astype(np.float32) / 255.0
    premultiplied = metal[..., :3].astype(np.float32) * alpha[..., None]
    width, height = size
    warped_alpha = cv2.warpAffine(
        alpha,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_pm = cv2.warpAffine(
        premultiplied,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0)
    warped_rgb = np.zeros_like(warped_pm)
    nonzero = warped_alpha > 1e-4
    warped_rgb[nonzero] = warped_pm[nonzero] / warped_alpha[nonzero, None]
    transformed = np.dstack(
        [
            np.clip(np.round(warped_rgb), 0, 255).astype(np.uint8),
            np.clip(np.round(warped_alpha * 255.0), 0, 255).astype(np.uint8),
        ]
    )
    return transformed, rotation, matrix, target_socket, target_shaft_point


def old_lower_head_cleanup(base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = base.shape[:2]
    manual_core = polygon_mask(
        (width, height),
        [
            [
                (603, 1135),
                (574, 1114),
                (556, 1140),
                (538, 1194),
                (568, 1172),
                (591, 1210),
                (612, 1181),
            ],
            [
                (607, 1138),
                (634, 1119),
                (652, 1147),
                (675, 1201),
                (646, 1172),
                (622, 1213),
                (598, 1181),
            ],
            [
                (595, 1160),
                (625, 1197),
                (602, 1280),
                (518, 1425),
                (521, 1302),
                (556, 1204),
            ],
        ],
    )
    manual = cv2.dilate(
        manual_core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ) > 0
    hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV)
    spread = (
        np.max(base, axis=2).astype(np.int16)
        - np.min(base, axis=2).astype(np.int16)
    )
    value = np.max(base, axis=2)
    metal_core = (
        manual_core
        & (hsv[..., 1] <= 105)
        & (spread <= 105)
        & (value >= 24)
        & ~key_like(base)
    )
    near_core = cv2.dilate(
        metal_core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    ) > 0
    key_distance = np.linalg.norm(
        base.astype(np.int16) - MAGENTA.astype(np.int16), axis=2
    )
    antialias_or_outline = (
        manual
        & near_core
        & (key_distance > 4.0)
        & ~red_material(base)
    )
    # The three hand-authored polygons trace the rejected head silhouette,
    # including its internal negative spaces.  Re-keying the complete traced
    # silhouette is necessary because retaining "already magenta" holes leaves
    # the old outline readable after the much smaller canonical head is placed.
    # Only the bounded antialias/outline fringe is selected by color.
    cleanup = manual_core | antialias_or_outline
    cleanup &= manual
    if int(np.count_nonzero(cleanup)) < 10_000:
        raise RuntimeError(
            f"old-head cleanup unexpectedly small: {np.count_nonzero(cleanup)}"
        )
    return cleanup, manual


def preservation_masks(
    base: np.ndarray,
    shaft: dict[str, object],
    cleanup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = base.shape[:2]
    subject = ~key_like(base)
    slope = float(shaft["slope"])
    intercept = float(shaft["intercept"])

    upper_prop_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(upper_prop_image)
    draw.polygon(
        [(780, 75), (870, 75), (885, 310), (790, 310)], fill=255
    )
    draw.line(
        [
            (round(slope * 1098.0 + intercept), 1098),
            (820, 160),
        ],
        fill=255,
        width=34,
    )
    draw.polygon(
        [(645, 600), (748, 600), (756, 770), (650, 780)], fill=255
    )
    upper_prop = (np.asarray(upper_prop_image, dtype=np.uint8) > 0) & subject

    body_manual = polygon_mask(
        (width, height),
        [
            [
                (165, 70),
                (590, 60),
                (740, 200),
                (850, 620),
                (855, 1210),
                (710, 1405),
                (375, 1405),
                (360, 1110),
                (180, 585),
            ],
        ],
    )
    body = body_manual & subject & ~cleanup
    return upper_prop, body


def checker_component(rgba: np.ndarray) -> np.ndarray:
    height, width = rgba.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    checker = ((xx // 20 + yy // 20) % 2)[..., None]
    background = np.repeat(
        np.where(checker, 205, 155).astype(np.uint8), 3, axis=2
    )
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(
        np.round(rgba[..., :3] * alpha + background * (1.0 - alpha)), 0, 255
    ).astype(np.uint8)


def fit_panel(image: Image.Image, size: tuple[int, int], title: str) -> Image.Image:
    scale = min(size[0] / image.width, size[1] / image.height)
    fitted = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.NEAREST,
    )
    panel = Image.new("RGB", (size[0], size[1] + 42), (12, 15, 20))
    panel.paste(
        fitted,
        ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2),
    )
    ImageDraw.Draw(panel).text((8, size[1] + 13), title, fill=(255, 255, 255))
    return panel


def make_proofs(
    base: np.ndarray,
    rejected_v6: np.ndarray,
    cleaned: np.ndarray,
    result: np.ndarray,
    metal: np.ndarray,
    masks: list[tuple[str, np.ndarray, tuple[int, int, int]]],
) -> None:
    full_panels = (
        fit_panel(Image.fromarray(base), (360, 610), "SEMANTIC BASE V2"),
        fit_panel(Image.fromarray(rejected_v6), (360, 610), "REJECTED V6 (NOT SOURCE)"),
        fit_panel(Image.fromarray(result), (360, 610), "CONCERNED V7 RIGID METAL"),
    )
    full_sheet = Image.new("RGB", (1080, 652), (12, 15, 20))
    for index, panel in enumerate(full_panels):
        full_sheet.paste(panel, (index * 360, 0))
    FULL_PROOF.parent.mkdir(parents=True, exist_ok=True)
    full_sheet.save(FULL_PROOF, optimize=True)

    close_crop = (455, 1010, 725, 1455)
    close_panels = (
        fit_panel(
            Image.fromarray(checker_component(metal)),
            (360, 670),
            "FROZEN RIGID METAL",
        ),
        fit_panel(
            Image.fromarray(base).crop(close_crop),
            (360, 670),
            "BASE OVERSIZED LOWER HEAD",
        ),
        fit_panel(
            Image.fromarray(cleaned).crop(close_crop),
            (360, 670),
            "BASE HEAD CLEANED TO CHROMA",
        ),
        fit_panel(
            Image.fromarray(result).crop(close_crop),
            (360, 670),
            "V7 PHYSICAL-SCALE HEAD",
        ),
    )
    close_sheet = Image.new("RGB", (1440, 712), (12, 15, 20))
    for index, panel in enumerate(close_panels):
        close_sheet.paste(panel, (index * 360, 0))
    CLOSE_PROOF.parent.mkdir(parents=True, exist_ok=True)
    close_sheet.save(CLOSE_PROOF, optimize=True)

    panel_width = 280
    panel_height = 620
    sheet = Image.new(
        "RGB", (panel_width * len(masks), panel_height + 42), (12, 15, 20)
    )
    for index, (title, mask, color) in enumerate(masks):
        shown = result.copy()
        shown[mask] = np.clip(
            np.round(
                shown[mask].astype(np.float32) * 0.28
                + np.asarray(color, dtype=np.float32) * 0.72
            ),
            0,
            255,
        ).astype(np.uint8)
        crop = Image.fromarray(shown).crop(close_crop)
        panel = fit_panel(crop, (panel_width, panel_height), title)
        sheet.paste(panel, (index * panel_width, 0))
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(MASK_PROOF, optimize=True)


def record(path: Path, pixels: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": str(path), "sha256": sha256(path)}
    if pixels is not None:
        value["pixels"] = pixels
    return value


def main() -> None:
    check_hashes()
    base = np.asarray(Image.open(BASE).convert("RGB"), dtype=np.uint8)
    rejected_v6 = np.asarray(Image.open(REJECTED_V6).convert("RGB"), dtype=np.uint8)
    metal = np.asarray(Image.open(METAL).convert("RGBA"), dtype=np.uint8)
    if tuple(base.shape) != (1455, 1081, 3):
        raise RuntimeError(f"unexpected base shape: {base.shape}")
    if tuple(rejected_v6.shape) != tuple(base.shape):
        raise RuntimeError("rejected v6 dimensions differ from semantic base")
    if tuple(metal.shape) != (314, 312, 4):
        raise RuntimeError(f"unexpected metal shape: {metal.shape}")

    shaft = fit_concerned_shaft(base)
    scale_evidence = derive_scale(base, shaft)
    scale = float(scale_evidence["derived_uniform_scale"])
    transformed, rotation, matrix, target_socket, target_shaft_point = transform_metal(
        metal, (base.shape[1], base.shape[0]), shaft, scale
    )
    patch = transformed[..., 3] >= MIN_ALPHA
    cleanup, manual_cleanup = old_lower_head_cleanup(base)
    upper_prop, body = preservation_masks(base, shaft, cleanup)

    cleaned = base.copy()
    cleaned[cleanup] = MAGENTA
    result = cleaned.copy()
    result[patch] = transformed[..., :3][patch]
    affected = cleanup | patch
    occlusion = affected & (upper_prop | body)
    result[occlusion] = base[occlusion]

    aggregate = cleanup | patch | occlusion
    changed = np.any(result != base, axis=2)
    outside_saved_delta = int(np.count_nonzero(changed & ~aggregate))
    outside_prop_delta = int(np.count_nonzero(changed & ~(cleanup | patch)))
    upper_prop_delta = int(np.count_nonzero(changed & upper_prop))
    body_delta = int(np.count_nonzero(changed & body))
    if outside_saved_delta or outside_prop_delta:
        raise RuntimeError(
            f"scope failure: outside saved={outside_saved_delta}, "
            f"outside prop={outside_prop_delta}"
        )
    if upper_prop_delta or body_delta:
        raise RuntimeError(
            f"preservation failure: upper prop={upper_prop_delta}, body={body_delta}"
        )

    visible_metal = patch & ~occlusion
    hot_magenta = visible_metal & key_like(result)
    if np.any(hot_magenta):
        raise RuntimeError(
            f"hot-magenta visible metal pixels: {np.count_nonzero(hot_magenta)}"
        )
    new_red = red_material(result) & ~red_material(base)
    red_component_count, _red_labels, red_stats, _red_centroids = (
        cv2.connectedComponentsWithStats(new_red.astype(np.uint8), connectivity=8)
    )
    new_red_component_areas = [
        int(red_stats[index, cv2.CC_STAT_AREA])
        for index in range(1, red_component_count)
    ]
    cloth_like_new_red_components = [
        area for area in new_red_component_areas if area >= 8
    ]
    if cloth_like_new_red_components:
        raise RuntimeError(
            "rigid-only v7 fabricated cloth-like lower red components: "
            f"{cloth_like_new_red_components}"
        )
    cleaned_old_prop_remaining = cleanup & ~key_like(cleaned)
    if np.any(cleaned_old_prop_remaining):
        raise RuntimeError("old-head cleanup did not become exact clean chroma")

    lower_roi = polygon_mask(
        (base.shape[1], base.shape[0]),
        [[(455, 1010), (725, 1010), (725, 1455), (455, 1455)]],
    )
    rejected_v6_streamer_pixels = (
        red_material(rejected_v6) & ~red_material(base) & lower_roi
    )
    v6_streamer_retained_pixels = int(
        np.count_nonzero(
            rejected_v6_streamer_pixels
            & np.all(result == rejected_v6, axis=2)
            & ~patch
            & ~np.all(result == base, axis=2)
        )
    )
    if v6_streamer_retained_pixels:
        raise RuntimeError(
            f"v7 retained {v6_streamer_retained_pixels} rejected-v6 streamer pixels"
        )

    for path in (
        OUTPUT,
        CLEANED,
        OLD_HEAD_MASK,
        PATCH_MASK,
        UPPER_PROP_MASK,
        BODY_MASK,
        OCCLUSION_MASK,
        AGGREGATE_MASK,
        CHANGED_MASK,
        FULL_PROOF,
        CLOSE_PROOF,
        MASK_PROOF,
        METADATA,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(OUTPUT, optimize=True)
    Image.fromarray(cleaned, mode="RGB").save(CLEANED, optimize=True)
    for mask, path in (
        (cleanup, OLD_HEAD_MASK),
        (patch, PATCH_MASK),
        (upper_prop, UPPER_PROP_MASK),
        (body, BODY_MASK),
        (occlusion, OCCLUSION_MASK),
        (aggregate, AGGREGATE_MASK),
        (changed, CHANGED_MASK),
    ):
        save_mask(mask, path)

    make_proofs(
        base,
        rejected_v6,
        cleaned,
        result,
        metal,
        [
            ("OLD LOWER HEAD CLEANUP", cleanup, (255, 220, 0)),
            ("CANONICAL METAL PATCH", patch, (0, 180, 255)),
            ("UPPER PROP EXACT", upper_prop, (255, 70, 90)),
            ("BODY/COSTUME EXACT", body, (255, 150, 0)),
            ("BASE OCCLUSION RESTORE", occlusion, (0, 255, 230)),
            ("ACTUAL CHANGED", changed, (255, 255, 255)),
        ],
    )

    masks: dict[str, dict[str, object]] = {}
    for name, mask, path in (
        ("old_lower_primary_cleanup", cleanup, OLD_HEAD_MASK),
        ("canonical_lower_primary_metal_patch", patch, PATCH_MASK),
        ("upper_cap_tassel_shaft_grip_exact", upper_prop, UPPER_PROP_MASK),
        ("body_face_costume_exact", body, BODY_MASK),
        ("base_subject_occlusion_restore", occlusion, OCCLUSION_MASK),
        ("aggregate_saved", aggregate, AGGREGATE_MASK),
        ("actual_changed", changed, CHANGED_MASK),
    ):
        masks[name] = {
            **record(path, int(np.count_nonzero(mask))),
            "bbox": bbox(mask),
        }

    metadata = {
        "status": "pending_root_review",
        "label": "concerned",
        "version": "deterministic_lower_primary_metal_v7",
        "method": (
            "semantic concerned v2 base plus deterministic lower rigid-metal-only "
            "replacement at a shaft-diameter-derived physical scale"
        ),
        "model_calls": 0,
        "inputs": {
            "semantic_base": record(BASE),
            "frozen_rigid_metal": record(METAL),
            "frozen_streamers_deferred_not_transformed": record(DEFERRED_STREAMERS),
            "material_split_metadata": record(MATERIAL_SPLIT_METADATA),
            "rejected_v6_preserved_not_pixel_source": record(REJECTED_V6),
            "script": record(Path(__file__).resolve()),
        },
        "endpoint_determination": {
            "selected_endpoint": "lower primary blade and guards",
            "opposite_endpoint": "upper small cap with short tassel",
            "reason": (
                "The inspected lower endpoint carries the visually dominant long blade, "
                "paired guards, and collar; the upper endpoint is a small cap. The role "
                "classification is based on relative design and physical size, not screen direction."
            ),
            "upper_cap_and_short_tassel_preserved_exact": True,
            "no_lower_cloth_fabricated": True,
        },
        "shaft_fit": shaft,
        "scale_evidence": scale_evidence,
        "transform": {
            "source_socket": SOURCE_SOCKET.tolist(),
            "source_shaft_point": SOURCE_SHAFT_POINT.tolist(),
            "target_socket": target_socket.tolist(),
            "target_shaft_point": target_shaft_point.tolist(),
            "rotation_degrees": rotation,
            "uniform_scale": scale,
            "matrix": matrix.tolist(),
            "nonuniform_deformation": False,
            "patch_bbox": bbox(patch),
        },
        "cleanup": {
            "manual_old_head_envelope_bbox": bbox(manual_cleanup),
            "fill_rgb": MAGENTA.tolist(),
            "all_cleanup_pixels_exact_clean_chroma": True,
            "rejected_v6_used_as_pixel_source": False,
            "rejected_v6_streamer_candidate_pixels": int(
                np.count_nonzero(rejected_v6_streamer_pixels)
            ),
            "v6_streamer_retained_pixels_outside_new_patch": (
                v6_streamer_retained_pixels
            ),
        },
        "masks": masks,
        "validation": {
            "base_shape_hwc": list(base.shape),
            "output_mode": "RGB",
            "output_size_wh": [base.shape[1], base.shape[0]],
            "outside_saved_mask_delta_pixels": outside_saved_delta,
            "outside_prop_scope_delta_pixels": outside_prop_delta,
            "upper_cap_tassel_shaft_grip_delta_pixels": upper_prop_delta,
            "body_face_costume_delta_pixels": body_delta,
            "hot_magenta_visible_metal_pixels": int(np.count_nonzero(hot_magenta)),
            "new_lower_warm_red_pixels_from_frozen_metal": int(
                np.count_nonzero(new_red)
            ),
            "new_lower_red_component_areas": new_red_component_areas,
            "new_lower_cloth_like_red_components_area_ge_8": len(
                cloth_like_new_red_components
            ),
            "old_cleanup_non_chroma_pixels": int(
                np.count_nonzero(cleaned_old_prop_remaining)
            ),
            "automated": "pass",
        },
        "outputs": {
            "repaired": record(OUTPUT),
            "cleaned_base": record(CLEANED),
            "full_proof": record(FULL_PROOF),
            "close_proof": record(CLOSE_PROOF),
            "mask_proof": record(MASK_PROOF),
        },
        "acceptance": {
            "automated": "pass",
            "visual": "pending_root_review",
            "manual_inspection": "pending native-resolution inspection",
        },
    }
    write_json(METADATA, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
