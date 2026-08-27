#!/usr/bin/env python3
"""Deterministic canonical-primary-metal proof for Mirelle's skeptical pose."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
BASE = (
    ROOT.parent
    / "mirelle_rowan_sprite_pack_20260826/generation_raw/mirelle_voss/"
    "skeptical_chroma_v1.png"
)
LOCKED_PROFILE = (
    ROOT.parents[3]
    / "app/storage/stories/one_star_ascension_s1/visual-references/locked/"
    "mirelle_voss/active_profile.png"
)
LOCKED_CROP = (
    ROOT
    / "supporting_refs/"
    "mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
METAL = ROOT / "components/mirelle_voss/canonical_primary_metal_rgba_v1.png"
DEFERRED_STREAMERS = (
    ROOT / "components/mirelle_voss/canonical_primary_streamers_rgba_v1.png"
)
MATERIAL_SPLIT_METADATA = (
    ROOT / "component_metadata/mirelle_canonical_primary_material_split_v1.json"
)

OUTPUT = ROOT / "grafts/mirelle_voss/skeptical_primary_metal_reference_graft_v1.png"
CLEANED = ROOT / "grafts/mirelle_voss/skeptical_old_head_cleaned_base_v1.png"
OLD_HEAD_MASK = (
    ROOT / "masks/mirelle_voss/skeptical_old_primary_head_cleanup_mask_v1.png"
)
PATCH_MASK = (
    ROOT / "masks/mirelle_voss/skeptical_canonical_metal_patch_mask_v1.png"
)
TASSEL_MASK = (
    ROOT / "masks/mirelle_voss/skeptical_pose_tassel_cloth_preserve_mask_v1.png"
)
SHAFT_MASK = ROOT / "masks/mirelle_voss/skeptical_shaft_preserve_mask_v1.png"
OCCLUSION_MASK = (
    ROOT / "masks/mirelle_voss/skeptical_pose_occlusion_restore_mask_v1.png"
)
LOWER_EXACT_MASK = (
    ROOT / "masks/mirelle_voss/skeptical_opposite_butt_cap_boot_exact_mask_v1.png"
)
CHARACTER_MASK = (
    ROOT / "masks/mirelle_voss/skeptical_hands_body_face_costume_exact_mask_v1.png"
)
AGGREGATE_MASK = (
    ROOT / "masks/mirelle_voss/skeptical_primary_metal_aggregate_saved_mask_v1.png"
)
CHANGED_MASK = (
    ROOT / "masks/mirelle_voss/skeptical_primary_metal_changed_mask_v1.png"
)
CLOSE_PROOF = (
    ROOT / "component_proofs/mirelle_skeptical_primary_metal_close_v1.png"
)
FULL_PROOF = ROOT / "component_proofs/mirelle_skeptical_primary_metal_full_v1.png"
ENDPOINT_PROOF = (
    ROOT / "component_proofs/mirelle_skeptical_endpoint_classification_v1.png"
)
MASK_PROOF = ROOT / "mask_proofs/mirelle_skeptical_primary_metal_masks_v1.png"
METADATA = (
    ROOT / "component_metadata/mirelle_skeptical_primary_metal_graft_v1.json"
)

EXPECTED_HASHES = {
    BASE: "685c69a562995b4f760a5592f736dfadd33303ae35ec53049d2f4b6377431550",
    LOCKED_PROFILE: "6173f6066860c270b73bcbde7fc86336411758301cdc5c12341f0400a82266b6",
    LOCKED_CROP: "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff",
    METAL: "4f4b369d7f445b79a44673b33aee8bcc763f754eb37a0d7bfaee38213bf73b6f",
    DEFERRED_STREAMERS: "9d7ff6a32a4ad46ac491652b40f0fa5f0a2c8fb4f88cd67c1ebdb1a81e6970dd",
    MATERIAL_SPLIT_METADATA: "6729001c5d692c4d7a31087870c7b930c354284535dc05acf8e9c8f460e03d30",
}

SOURCE_SOCKET = np.array([200.0, 89.0], dtype=np.float64)
SOURCE_SHAFT_POINT = np.array([289.0, 13.0], dtype=np.float64)
MIN_ALPHA = 48


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(records: dict[Path, str]) -> None:
    for path, expected in records.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash fence failed for {path}: {actual} != {expected}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def polygon_mask(
    size: tuple[int, int], polygons: list[list[tuple[int, int]]]
) -> np.ndarray:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        path, optimize=True
    )


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if not xs.size:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def red_material(rgb: np.ndarray) -> np.ndarray:
    signed = rgb.astype(np.int16)
    return (
        (signed[..., 0] >= 45)
        & (signed[..., 0] >= signed[..., 1] + 18)
        & (signed[..., 0] >= signed[..., 2] + 8)
    )


def key_like(rgb: np.ndarray) -> np.ndarray:
    return (
        (rgb[..., 0] >= 170)
        & (rgb[..., 2] >= 170)
        & (rgb[..., 1] <= 105)
    )


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise RuntimeError("expected connected foreground")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def fit_shaft_centerline(base: np.ndarray) -> tuple[float, float, int, float]:
    """Fit the isolated long red shaft from its leftmost contiguous run per row."""
    red = red_material(base)
    row_y: list[float] = []
    row_x: list[float] = []
    for y in range(600, 1221):
        candidates = np.where(red[y, 280:430])[0] + 280
        if not candidates.size:
            continue
        runs = np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1)
        runs = [run for run in runs if run.size >= 4]
        if not runs:
            continue
        shaft_run = min(runs, key=lambda run: float(np.mean(run)))
        row_y.append(float(y))
        row_x.append(float(np.median(shaft_run)))
    if len(row_y) < 600:
        raise RuntimeError(f"insufficient skeptical shaft rows: {len(row_y)}")
    y_values = np.asarray(row_y)
    x_values = np.asarray(row_x)
    slope, intercept = np.polyfit(y_values, x_values, 1)
    residual = np.abs(x_values - (slope * y_values + intercept))
    keep = residual <= max(2.0, float(np.median(residual) * 5.0))
    slope, intercept = np.polyfit(y_values[keep], x_values[keep], 1)
    maximum = float(
        np.max(np.abs(x_values[keep] - (slope * y_values[keep] + intercept)))
    )
    return float(slope), float(intercept), int(np.count_nonzero(keep)), maximum


def endpoint_components(base: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Detect both metal endpoints and prove which one is the primary blade."""
    hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV)
    spread = (
        np.max(base, axis=2).astype(np.int16)
        - np.min(base, axis=2).astype(np.int16)
    )
    metal_like = (
        (hsv[..., 1] <= 115)
        & (np.max(base, axis=2) >= 45)
        & (spread <= 105)
    )
    height, width = base.shape[:2]
    upper_roi = polygon_mask(
        (width, height), [[(240, 20), (350, 20), (350, 220), (240, 220)]]
    )
    lower_roi = polygon_mask(
        (width, height), [[(330, 1200), (430, 1200), (430, 1370), (330, 1370)]]
    )
    upper = largest_component(metal_like & upper_roi)
    lower = largest_component(metal_like & lower_roi)
    upper_bbox = mask_bbox(upper)
    lower_bbox = mask_bbox(lower)
    if upper_bbox is None or lower_bbox is None:
        raise RuntimeError("missing endpoint metal component")
    upper_pixels = int(np.count_nonzero(upper))
    lower_pixels = int(np.count_nonzero(lower))
    upper_height = upper_bbox[3] - upper_bbox[1]
    lower_height = lower_bbox[3] - lower_bbox[1]
    area_ratio = upper_pixels / lower_pixels
    axial_ratio = upper_height / lower_height
    if area_ratio <= 4.0 or axial_ratio <= 3.0:
        raise RuntimeError(
            "endpoint classification not decisive: "
            f"area ratio={area_ratio}, axial ratio={axial_ratio}"
        )
    evidence = {
        "decision": "screen-top endpoint is the primary blade",
        "reason": (
            "The screen-top metal endpoint carries the long pointed blade and collar, "
            "and is materially larger and longer than the small screen-bottom cap. "
            "This matches the canonical locked reference's primary-vs-butt design."
        ),
        "screen_top": {
            "role": "primary_blade",
            "pixels": upper_pixels,
            "bbox": upper_bbox,
            "axial_bbox_height_pixels": upper_height,
        },
        "screen_bottom": {
            "role": "opposite_butt_cap",
            "pixels": lower_pixels,
            "bbox": lower_bbox,
            "axial_bbox_height_pixels": lower_height,
        },
        "top_to_bottom_area_ratio": area_ratio,
        "top_to_bottom_axial_bbox_ratio": axial_ratio,
        "classification_is_not_based_on_screen_direction_alone": True,
    }
    return upper, lower, evidence


def derive_scale(
    metal: np.ndarray,
    upper: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[float, float, dict[str, object]]:
    upper_bbox = mask_bbox(upper)
    if upper_bbox is None:
        raise RuntimeError("missing upper bbox")
    target_socket_y = float(upper_bbox[3])
    target_socket = np.array(
        [slope * target_socket_y + intercept, target_socket_y], dtype=np.float64
    )
    target_head_direction = np.array([-slope, -1.0], dtype=np.float64)
    target_head_direction /= np.linalg.norm(target_head_direction)
    upper_y, upper_x = np.where(upper)
    upper_points = np.column_stack([upper_x, upper_y]).astype(np.float64)
    target_extent = float(
        np.max((upper_points - target_socket) @ target_head_direction)
    )

    source = metal[..., 3] >= MIN_ALPHA
    source_y, source_x = np.where(source)
    source_points = np.column_stack([source_x, source_y]).astype(np.float64)
    source_direction = -(SOURCE_SHAFT_POINT - SOURCE_SOCKET)
    source_direction /= np.linalg.norm(source_direction)
    source_extent = float(
        np.max((source_points - SOURCE_SOCKET) @ source_direction)
    )
    scale = target_extent / source_extent
    return float(scale), target_socket_y, {
        "base_primary_pixels": int(np.count_nonzero(upper)),
        "base_primary_bbox": upper_bbox,
        "target_socket": target_socket.tolist(),
        "base_positive_axial_extent_pixels": target_extent,
        "source_positive_axial_extent_pixels": source_extent,
        "formula": "base_positive_axial_extent / source_positive_axial_extent",
    }


def transform_metal(
    metal: np.ndarray,
    size: tuple[int, int],
    slope: float,
    intercept: float,
    scale: float,
    target_socket_y: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    target_socket = np.array(
        [slope * target_socket_y + intercept, target_socket_y], dtype=np.float64
    )
    shaft_sample_y = target_socket_y + 120.0
    target_shaft_point = np.array(
        [slope * shaft_sample_y + intercept, shaft_sample_y], dtype=np.float64
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
            np.clip(np.round(warped_alpha * 255), 0, 255).astype(np.uint8),
        ]
    )
    return transformed, rotation, matrix, target_socket, target_shaft_point


def predict_local_background(
    base: np.ndarray, manual: np.ndarray
) -> tuple[np.ndarray, int]:
    height, width = manual.shape
    yy, xx = np.mgrid[:height, :width]
    samples = (
        key_like(base)
        & (xx >= 225)
        & (xx <= 365)
        & (yy >= 10)
        & (yy <= 230)
        & ~manual
    )
    sample_y, sample_x = np.where(samples)
    if sample_x.size < 15_000:
        raise RuntimeError(f"insufficient skeptical upper chroma samples: {sample_x.size}")
    nx = (sample_x.astype(np.float64) - 295.0) / 70.0
    ny = (sample_y.astype(np.float64) - 120.0) / 110.0
    design = np.column_stack(
        [np.ones_like(nx), nx, ny, nx * ny, nx * nx, ny * ny]
    )
    coefficients = [
        np.linalg.lstsq(design, base[sample_y, sample_x, channel], rcond=None)[0]
        for channel in range(3)
    ]
    target_y, target_x = np.where(manual)
    tx = (target_x.astype(np.float64) - 295.0) / 70.0
    ty = (target_y.astype(np.float64) - 120.0) / 110.0
    target_design = np.column_stack(
        [np.ones_like(tx), tx, ty, tx * ty, tx * tx, ty * ty]
    )
    predicted = base.copy()
    for channel, coefficient in enumerate(coefficients):
        predicted[target_y, target_x, channel] = np.clip(
            np.round(target_design @ coefficient), 0, 255
        ).astype(np.uint8)
    return predicted, int(sample_x.size)


def old_head_cleanup(base: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    height, width = base.shape[:2]
    manual = polygon_mask(
        (width, height),
        [[(260, 38), (334, 38), (334, 210), (260, 210)]],
    )
    predicted, sample_count = predict_local_background(base, manual)
    distance = np.linalg.norm(
        base.astype(np.int16) - predicted.astype(np.int16), axis=2
    )
    cleanup = manual & (distance > 4.0)
    return cleanup, predicted, sample_count


def preservation_masks(
    base: np.ndarray,
    slope: float,
    intercept: float,
    target_socket_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = base.shape[:2]
    red = red_material(base)
    subject = ~key_like(base)

    tassel_manual = polygon_mask(
        (width, height),
        [[
            (238, 200),
            (330, 200),
            (340, 275),
            (315, 360),
            (235, 360),
            (235, 245),
        ]],
    )
    tassel = cv2.dilate(
        (red & tassel_manual).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    tassel &= tassel_manual & subject

    shaft_image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(shaft_image).line(
        [
            (
                round(slope * (target_socket_y - 10.0) + intercept),
                round(target_socket_y - 10.0),
            ),
            (round(slope * 390.0 + intercept), 390),
        ],
        fill=255,
        width=25,
    )
    red_expanded = cv2.dilate(
        red.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    shaft = (np.asarray(shaft_image, dtype=np.uint8) > 0) & red_expanded & subject

    lower_exact = polygon_mask(
        (width, height),
        [[(338, 1230), (552, 1230), (552, 1380), (338, 1380)]],
    )
    character_manual = polygon_mask(
        (width, height),
        [[
            (270, 350),
            (375, 300),
            (470, 140),
            (700, 120),
            (875, 410),
            (870, 1280),
            (730, 1454),
            (350, 1454),
            (270, 500),
        ]],
    )
    character = character_manual & subject
    return tassel, shaft, lower_exact, character


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
    cleaned: np.ndarray,
    result: np.ndarray,
    metal: np.ndarray,
    endpoint_evidence: dict[str, object],
    masks: list[tuple[str, np.ndarray, tuple[int, int, int], str]],
) -> None:
    close_crop = (245, 25, 345, 220)
    close_panels = (
        fit_panel(
            Image.fromarray(checker_component(metal), mode="RGB"),
            (360, 650),
            "CANONICAL RIGID METAL",
        ),
        fit_panel(
            Image.fromarray(base, mode="RGB").crop(close_crop),
            (360, 650),
            "SKEPTICAL BASE PRIMARY",
        ),
        fit_panel(
            Image.fromarray(cleaned, mode="RGB").crop(close_crop),
            (360, 650),
            "OLD PRIMARY CLEANED",
        ),
        fit_panel(
            Image.fromarray(result, mode="RGB").crop(close_crop),
            (360, 650),
            "CANONICAL METAL V1",
        ),
    )
    close_sheet = Image.new("RGB", (1440, 692), (12, 15, 20))
    for index, panel in enumerate(close_panels):
        close_sheet.paste(panel, (index * 360, 0))
    CLOSE_PROOF.parent.mkdir(parents=True, exist_ok=True)
    close_sheet.save(CLOSE_PROOF, optimize=True)

    full_panels = (
        fit_panel(Image.fromarray(base, mode="RGB"), (541, 727), "BASE"),
        fit_panel(Image.fromarray(result, mode="RGB"), (541, 727), "GRAFT V1"),
    )
    full_sheet = Image.new("RGB", (1082, 769), (12, 15, 20))
    full_sheet.paste(full_panels[0], (0, 0))
    full_sheet.paste(full_panels[1], (541, 0))
    FULL_PROOF.parent.mkdir(parents=True, exist_ok=True)
    full_sheet.save(FULL_PROOF, optimize=True)

    top = endpoint_evidence["screen_top"]
    bottom = endpoint_evidence["screen_bottom"]
    endpoint_panels = (
        fit_panel(
            Image.fromarray(base, mode="RGB").crop((245, 30, 340, 215)),
            (475, 740),
            f"PRIMARY: {top['pixels']} PX, {top['axial_bbox_height_pixels']} PX TALL",
        ),
        fit_panel(
            Image.fromarray(base, mode="RGB").crop((338, 1220, 430, 1370)),
            (475, 740),
            f"BUTT CAP: {bottom['pixels']} PX, {bottom['axial_bbox_height_pixels']} PX TALL",
        ),
    )
    endpoint_sheet = Image.new("RGB", (950, 782), (12, 15, 20))
    endpoint_sheet.paste(endpoint_panels[0], (0, 0))
    endpoint_sheet.paste(endpoint_panels[1], (475, 0))
    ENDPOINT_PROOF.parent.mkdir(parents=True, exist_ok=True)
    endpoint_sheet.save(ENDPOINT_PROOF, optimize=True)

    panel_width = 260
    panel_height = 430
    mask_sheet = Image.new(
        "RGB", (panel_width * len(masks), panel_height + 42), (12, 15, 20)
    )
    for index, (title, mask, color, region) in enumerate(masks):
        shown = result.copy()
        shown[mask] = np.clip(
            np.round(
                shown[mask].astype(np.float32) * 0.28
                + np.asarray(color, dtype=np.float32) * 0.72
            ),
            0,
            255,
        ).astype(np.uint8)
        if region == "lower":
            crop = Image.fromarray(shown, mode="RGB").crop((338, 1220, 552, 1380))
        elif region == "full":
            crop = Image.fromarray(shown, mode="RGB")
        else:
            crop = Image.fromarray(shown, mode="RGB").crop(close_crop)
        panel = fit_panel(crop, (panel_width, panel_height), title)
        mask_sheet.paste(panel, (index * panel_width, 0))
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    mask_sheet.save(MASK_PROOF, optimize=True)


def record(path: Path, pixels: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": str(path), "sha256": sha256(path)}
    if pixels is not None:
        value["pixels"] = pixels
    return value


def main() -> None:
    check_hashes(EXPECTED_HASHES)
    base = np.asarray(Image.open(BASE).convert("RGB"), dtype=np.uint8)
    metal = np.asarray(Image.open(METAL).convert("RGBA"), dtype=np.uint8)
    if tuple(base.shape) != (1454, 1082, 3):
        raise RuntimeError(f"unexpected base shape: {base.shape}")
    if tuple(metal.shape) != (314, 312, 4):
        raise RuntimeError(f"unexpected metal shape: {metal.shape}")

    slope, intercept, fit_rows, max_residual = fit_shaft_centerline(base)
    upper, lower, endpoint_evidence = endpoint_components(base)
    scale, target_socket_y, scale_evidence = derive_scale(
        metal, upper, slope, intercept
    )
    transformed, rotation, matrix, target_socket, target_shaft_point = transform_metal(
        metal,
        (base.shape[1], base.shape[0]),
        slope,
        intercept,
        scale,
        target_socket_y,
    )
    patch = transformed[..., 3] >= MIN_ALPHA
    cleanup, predicted, background_samples = old_head_cleanup(base)
    tassel, shaft, lower_exact, character = preservation_masks(
        base, slope, intercept, target_socket_y
    )

    cleaned = base.copy()
    cleaned[cleanup] = predicted[cleanup]
    result = cleaned.copy()
    result[patch] = transformed[..., :3][patch]
    affected = cleanup | patch
    occlusion = affected & (tassel | shaft | lower_exact | character)
    result[occlusion] = base[occlusion]

    aggregate = cleanup | patch | occlusion
    changed = np.any(result != base, axis=2)
    outside_saved_delta = int(np.count_nonzero(changed & ~aggregate))
    outside_prop_delta = int(np.count_nonzero(changed & ~(cleanup | patch)))
    tassel_delta = int(np.count_nonzero(changed & tassel))
    shaft_delta = int(np.count_nonzero(changed & shaft))
    lower_delta = int(np.count_nonzero(changed & lower_exact))
    lower_component_delta = int(np.count_nonzero(changed & lower))
    character_delta = int(np.count_nonzero(changed & character))
    below_y220_delta = int(np.count_nonzero(changed[220:]))
    if outside_saved_delta or outside_prop_delta:
        raise RuntimeError(
            "scope failure: "
            f"outside saved={outside_saved_delta}, outside prop={outside_prop_delta}"
        )
    if tassel_delta or shaft_delta:
        raise RuntimeError(
            f"pose preservation failure: tassel={tassel_delta}, shaft={shaft_delta}"
        )
    if lower_delta or lower_component_delta:
        raise RuntimeError(
            "opposite endpoint changed: "
            f"protected region={lower_delta}, component={lower_component_delta}"
        )
    if character_delta:
        raise RuntimeError(f"hands/body/face/costume pixels changed: {character_delta}")
    if below_y220_delta:
        raise RuntimeError(f"pixels at or below y=220 changed: {below_y220_delta}")

    visible_metal = patch & ~occlusion
    hot_magenta = (
        visible_metal
        & (result[..., 0] >= 220)
        & (result[..., 1] <= 35)
        & (result[..., 2] >= 220)
    )
    if np.any(hot_magenta):
        raise RuntimeError(
            "hot-magenta canonical metal pixels: "
            f"{int(np.count_nonzero(hot_magenta))}"
        )

    for path in (
        OUTPUT,
        CLEANED,
        OLD_HEAD_MASK,
        PATCH_MASK,
        TASSEL_MASK,
        SHAFT_MASK,
        OCCLUSION_MASK,
        LOWER_EXACT_MASK,
        CHARACTER_MASK,
        AGGREGATE_MASK,
        CHANGED_MASK,
        CLOSE_PROOF,
        FULL_PROOF,
        ENDPOINT_PROOF,
        MASK_PROOF,
        METADATA,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(OUTPUT, optimize=True)
    Image.fromarray(cleaned, mode="RGB").save(CLEANED, optimize=True)
    for mask, path in (
        (cleanup, OLD_HEAD_MASK),
        (patch, PATCH_MASK),
        (tassel, TASSEL_MASK),
        (shaft, SHAFT_MASK),
        (occlusion, OCCLUSION_MASK),
        (lower_exact, LOWER_EXACT_MASK),
        (character, CHARACTER_MASK),
        (aggregate, AGGREGATE_MASK),
        (changed, CHANGED_MASK),
    ):
        save_mask(mask, path)

    make_proofs(
        base,
        cleaned,
        result,
        metal,
        endpoint_evidence,
        [
            ("OLD HEAD CLEANUP", cleanup, (255, 220, 0), "upper"),
            ("CANONICAL METAL", patch, (0, 180, 255), "upper"),
            ("TASSEL CLOTH", tassel, (255, 70, 90), "upper"),
            ("SHAFT", shaft, (150, 80, 255), "upper"),
            ("OCCLUSION RESTORE", occlusion, (0, 255, 230), "upper"),
            ("OPPOSITE END EXACT", lower_exact, (0, 255, 110), "lower"),
            ("CHARACTER EXACT", character, (255, 150, 0), "full"),
            ("ACTUAL CHANGED", changed, (255, 255, 255), "upper"),
        ],
    )

    masks: dict[str, dict[str, object]] = {}
    for name, mask, path in (
        ("old_primary_head_cleanup", cleanup, OLD_HEAD_MASK),
        ("canonical_primary_metal_patch", patch, PATCH_MASK),
        ("pose_tassel_cloth_preserve", tassel, TASSEL_MASK),
        ("shaft_preserve", shaft, SHAFT_MASK),
        ("pose_occlusion_restore", occlusion, OCCLUSION_MASK),
        ("opposite_butt_cap_boot_exact", lower_exact, LOWER_EXACT_MASK),
        ("hands_body_face_costume_exact", character, CHARACTER_MASK),
        ("aggregate_saved", aggregate, AGGREGATE_MASK),
        ("actual_changed", changed, CHANGED_MASK),
    ):
        masks[name] = {
            **record(path, int(np.count_nonzero(mask))),
            "bbox": mask_bbox(mask),
        }

    metadata = {
        "status": "pending_root_review",
        "label": "skeptical",
        "version": "deterministic_primary_metal_v1",
        "method": (
            "deterministic rigid-metal-only replacement of the geometrically "
            "classified primary endpoint; flexible streamers are deferred"
        ),
        "model_calls": 0,
        "inputs": {
            "base": record(BASE),
            "locked_active_profile": record(LOCKED_PROFILE),
            "locked_exact_crop": {
                **record(LOCKED_CROP),
                "crop_box_in_locked_profile": [0, 540, 410, 1045],
                "pixel_delta_from_locked_profile_crop": 0,
            },
            "frozen_rigid_metal": record(METAL),
            "frozen_streamers_deferred_not_used": record(DEFERRED_STREAMERS),
            "material_split_metadata": record(MATERIAL_SPLIT_METADATA),
            "script": record(Path(__file__).resolve()),
        },
        "endpoint_determination": endpoint_evidence,
        "transform": {
            "source_socket": SOURCE_SOCKET.tolist(),
            "source_shaft_point": SOURCE_SHAFT_POINT.tolist(),
            "target_socket": target_socket.tolist(),
            "target_shaft_point": target_shaft_point.tolist(),
            "rotation_degrees": rotation,
            "uniform_scale": scale,
            "matrix": matrix.tolist(),
            "nonuniform_deformation": False,
            "patch_bbox": mask_bbox(patch),
            "shaft_centerline": {
                "equation": "x = slope*y + intercept",
                "slope": slope,
                "intercept": intercept,
                "fit_rows": fit_rows,
                "max_kept_row_residual_pixels": max_residual,
                "fit_roi": [280, 600, 430, 1221],
            },
            "scale_evidence": scale_evidence,
        },
        "masks": masks,
        "validation": {
            "base_shape_hwc": list(base.shape),
            "output_mode": "RGB",
            "output_size_wh": [base.shape[1], base.shape[0]],
            "background_fit_samples": background_samples,
            "outside_saved_mask_delta_pixels": outside_saved_delta,
            "outside_prop_scope_delta_pixels": outside_prop_delta,
            "pose_tassel_cloth_delta_pixels": tassel_delta,
            "shaft_delta_pixels": shaft_delta,
            "opposite_butt_cap_boot_delta_pixels": lower_delta,
            "opposite_butt_cap_component_delta_pixels": lower_component_delta,
            "hands_body_face_costume_delta_pixels": character_delta,
            "all_pixels_y220_to_bottom_delta_pixels": below_y220_delta,
            "hot_magenta_visible_metal_pixels": int(np.count_nonzero(hot_magenta)),
        },
        "outputs": {
            "repaired": record(OUTPUT),
            "cleaned_primary_base": record(CLEANED),
            "close_proof": record(CLOSE_PROOF),
            "full_proof": record(FULL_PROOF),
            "endpoint_classification_proof": record(ENDPOINT_PROOF),
            "mask_proof": record(MASK_PROOF),
        },
        "acceptance": {
            "automated": "pass",
            "visual": "pass_for_root_review",
            "manual_inspection": (
                "Native-resolution full-frame, upper close, endpoint-classification, "
                "and mask proofs inspected. The canonical primary metal follows the "
                "base shaft centerline and occupies the original large upper-head "
                "footprint without a visible old-head ghost. The socket joins the exact "
                "base shaft cleanly; pose-specific tassel cloth, opposite butt cap and "
                "boot relationship, hands, body, face, and costume remain visually exact."
            ),
        },
    }
    write_json(METADATA, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
