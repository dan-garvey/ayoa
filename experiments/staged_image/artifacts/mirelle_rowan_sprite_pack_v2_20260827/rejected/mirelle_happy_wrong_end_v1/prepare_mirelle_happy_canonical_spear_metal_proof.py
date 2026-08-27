#!/usr/bin/env python3
"""Deterministic metal-only canonical spear repair proof for Mirelle's happy pose."""

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
    / "mirelle_rowan_sprite_pack_20260826/generation_raw/mirelle_voss/happy_chroma_v1.png"
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
SOURCE_V6 = (
    ROOT
    / "components/mirelle_voss/canonical_lower_spearhead_streamers_rgba_v6.png"
)
METAL = ROOT / "components/mirelle_voss/canonical_primary_metal_rgba_v1.png"
DEFERRED_STREAMERS = (
    ROOT / "components/mirelle_voss/canonical_primary_streamers_rgba_v1.png"
)
MATERIAL_SPLIT_METADATA = (
    ROOT / "component_metadata/mirelle_canonical_primary_material_split_v1.json"
)
MATERIAL_SPLIT_PROOF = (
    ROOT / "component_proofs/mirelle_canonical_primary_material_split_v1.png"
)
REJECTED_FULL_COMPONENT_PREVIEW = (
    ROOT
    / "rejected/mirelle_happy_full_component_rigid_transform_preview/"
    "full_component_rigid_transform_preview.png"
)
REJECTED_FULL_COMPONENT_METADATA = (
    ROOT
    / "rejected/mirelle_happy_full_component_rigid_transform_preview/"
    "rejection.json"
)

OUTPUT = (
    ROOT / "grafts/mirelle_voss/happy_canonical_spear_metal_repair_proof.png"
)
CLEANED_BASE = (
    ROOT / "grafts/mirelle_voss/happy_old_head_cleaned_base_proof.png"
)
OLD_HEAD_MASK = (
    ROOT / "masks/mirelle_voss/happy_canonical_spear_old_head_cleanup_mask.png"
)
PATCH_MASK = (
    ROOT / "masks/mirelle_voss/happy_canonical_spear_rigid_metal_patch_mask.png"
)
OCCLUSION_MASK = (
    ROOT
    / "masks/mirelle_voss/"
    "happy_canonical_spear_hand_body_garment_occlusion_mask.png"
)
AGGREGATE_MASK = (
    ROOT / "masks/mirelle_voss/happy_canonical_spear_aggregate_saved_mask.png"
)
CHANGED_MASK = (
    ROOT / "masks/mirelle_voss/happy_canonical_spear_changed_mask.png"
)
PROOF = (
    ROOT / "component_proofs/mirelle_happy_canonical_spear_metal_close_proof.png"
)
MASK_PROOF = (
    ROOT / "mask_proofs/mirelle_happy_canonical_spear_metal_masks.png"
)
METADATA = (
    ROOT
    / "component_metadata/mirelle_happy_canonical_spear_metal_repair_proof.json"
)

EXPECTED_HASHES = {
    BASE: "12d42a02f1ac49765c3034916937e73380dddd39560764fafee165b794f2cfb2",
    LOCKED_PROFILE: "6173f6066860c270b73bcbde7fc86336411758301cdc5c12341f0400a82266b6",
    LOCKED_CROP: "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff",
    SOURCE_V6: "1285db94cdfd24d346123563227b8627129bc50a26a888bce85f82099defed4e",
    METAL: "4f4b369d7f445b79a44673b33aee8bcc763f754eb37a0d7bfaee38213bf73b6f",
    DEFERRED_STREAMERS: "9d7ff6a32a4ad46ac491652b40f0fa5f0a2c8fb4f88cd67c1ebdb1a81e6970dd",
    MATERIAL_SPLIT_METADATA: "6729001c5d692c4d7a31087870c7b930c354284535dc05acf8e9c8f460e03d30",
    MATERIAL_SPLIT_PROOF: "d22b5e0bc779c4ec1718df490167fb7c8b7ae1f766e0e38089a0df2049ad1961",
    REJECTED_FULL_COMPONENT_PREVIEW: "b5990f5e629fe97d34df4e4b5518fd673b7f8734f299f02dc59c7690f05ce3d4",
}

SOURCE_SOCKET = np.array([200.0, 89.0], dtype=np.float64)
SOURCE_SHAFT_POINT = np.array([289.0, 13.0], dtype=np.float64)
TARGET_SOCKET_Y = 1304.0
TARGET_AXIS_SAMPLE_SPAN = 120.0
MIN_ALPHA = 48


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hash_fences() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash fence failed for {path}: {actual} != {expected}")


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
    if xs.size == 0:
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


def fit_shaft_centerline(base: np.ndarray) -> tuple[float, float, int, float]:
    """Fit x = slope*y + intercept from row medians of the long red shaft."""
    red = red_material(base)
    row_y: list[float] = []
    row_x: list[float] = []
    for y in range(650, 1221):
        xs = np.where(red[y, 220:301])[0] + 220
        if xs.size >= 5:
            row_y.append(float(y))
            row_x.append(float(np.median(xs)))
    if len(row_y) < 500:
        raise RuntimeError(f"insufficient shaft rows: {len(row_y)}")
    y_values = np.asarray(row_y, dtype=np.float64)
    x_values = np.asarray(row_x, dtype=np.float64)
    slope, intercept = np.polyfit(y_values, x_values, 1)
    residual = np.abs(x_values - (slope * y_values + intercept))
    keep = residual <= max(2.0, float(np.median(residual) * 5.0))
    slope, intercept = np.polyfit(y_values[keep], x_values[keep], 1)
    max_residual = float(
        np.max(np.abs(x_values[keep] - (slope * y_values[keep] + intercept)))
    )
    return float(slope), float(intercept), int(np.count_nonzero(keep)), max_residual


def connected_component(mask: np.ndarray, *, largest: bool = True) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise RuntimeError("expected a foreground connected component")
    indices = range(1, count)
    selected = max(indices, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    if not largest:
        raise NotImplementedError
    return labels == selected


def derive_scale(
    base: np.ndarray, metal: np.ndarray, slope: float, intercept: float
) -> tuple[float, float, float, dict[str, object]]:
    """Match the rigid canonical metal to the base upper metal-head footprint."""
    hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV)
    spread = np.max(base, axis=2).astype(np.int16) - np.min(base, axis=2).astype(np.int16)
    region = np.zeros(base.shape[:2], dtype=bool)
    region[10:170, 200:280] = True
    upper_seed = (
        region
        & (hsv[..., 1] <= 115)
        & (np.max(base, axis=2) >= 45)
        & (spread <= 105)
    )
    upper_metal = connected_component(upper_seed)
    base_socket = np.array([slope * 136.0 + intercept, 136.0], dtype=np.float64)
    base_head_direction = np.array([-slope, -1.0], dtype=np.float64)
    base_head_direction /= np.linalg.norm(base_head_direction)
    upper_y, upper_x = np.where(upper_metal)
    upper_points = np.column_stack([upper_x, upper_y]).astype(np.float64)
    base_projection = (upper_points - base_socket) @ base_head_direction
    base_extent = float(np.max(base_projection))

    source_foreground = metal[..., 3] >= MIN_ALPHA
    source_y, source_x = np.where(source_foreground)
    source_points = np.column_stack([source_x, source_y]).astype(np.float64)
    source_head_direction = -(SOURCE_SHAFT_POINT - SOURCE_SOCKET)
    source_head_direction /= np.linalg.norm(source_head_direction)
    source_projection = (source_points - SOURCE_SOCKET) @ source_head_direction
    source_extent = float(np.max(source_projection))
    scale = base_extent / source_extent
    evidence = {
        "base_upper_metal_mask_pixels": int(np.count_nonzero(upper_metal)),
        "base_upper_metal_bbox": mask_bbox(upper_metal),
        "base_socket": base_socket.tolist(),
        "base_positive_axial_extent_pixels": base_extent,
        "source_positive_axial_extent_pixels": source_extent,
        "scale_formula": "base_positive_axial_extent / source_positive_axial_extent",
    }
    return float(scale), base_extent, source_extent, evidence


def transform_metal(
    metal: np.ndarray,
    target_size: tuple[int, int],
    slope: float,
    intercept: float,
    scale: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    target_socket = np.array(
        [slope * TARGET_SOCKET_Y + intercept, TARGET_SOCKET_Y], dtype=np.float64
    )
    target_shaft_point = np.array(
        [
            slope * (TARGET_SOCKET_Y - TARGET_AXIS_SAMPLE_SPAN) + intercept,
            TARGET_SOCKET_Y - TARGET_AXIS_SAMPLE_SPAN,
        ],
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
    width, height = target_size
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


def old_head_cleanup(
    base: np.ndarray, slope: float, intercept: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select the obsolete cap and its antialias fringe, never the red shaft."""
    height, width = base.shape[:2]
    manual = polygon_mask(
        (width, height),
        [[
            (266, 1301),
            (289, 1301),
            (293, 1310),
            (292, 1346),
            (288, 1357),
            (270, 1357),
            (266, 1347),
        ]],
    )
    predicted, fit_samples = reconstruct_key_background(base, manual)
    distance = np.linalg.norm(
        base.astype(np.int16) - predicted.astype(np.int16), axis=2
    )
    yy, xx = np.mgrid[:height, :width]
    shaft_center = slope * yy + intercept
    shaft = (
        manual
        & (yy <= 1307)
        & (np.abs(xx - shaft_center) <= 12.0)
        & ~key_like(base)
    )
    cleanup = manual & (distance > 8.0) & ~shaft
    return cleanup, shaft, fit_samples


def reconstruct_key_background(
    base: np.ndarray, erase: np.ndarray
) -> tuple[np.ndarray, int]:
    """Fit the nearby smooth chroma plate behind the removed old metal cap."""
    height, width = erase.shape
    yy, xx = np.mgrid[0:height, 0:width]
    fit_region = (xx >= 225) & (xx <= 335) & (yy >= 1240) & (yy <= 1435)
    samples = key_like(base) & ~erase & fit_region
    sample_y, sample_x = np.where(samples)
    if sample_x.size < 10_000:
        raise RuntimeError(f"insufficient local chroma samples: {sample_x.size}")
    nx = (sample_x.astype(np.float64) - 280.0) / 55.0
    ny = (sample_y.astype(np.float64) - 1337.5) / 97.5
    design = np.column_stack(
        [np.ones_like(nx), nx, ny, nx * ny, nx * nx, ny * ny]
    )
    coefficients = [
        np.linalg.lstsq(design, base[sample_y, sample_x, channel], rcond=None)[0]
        for channel in range(3)
    ]
    erase_y, erase_x = np.where(erase)
    ex = (erase_x.astype(np.float64) - 280.0) / 55.0
    ey = (erase_y.astype(np.float64) - 1337.5) / 97.5
    erase_design = np.column_stack(
        [np.ones_like(ex), ex, ey, ex * ey, ex * ex, ey * ey]
    )
    cleaned = base.copy()
    for channel, coefficient in enumerate(coefficients):
        values = np.clip(np.round(erase_design @ coefficient), 0, 255).astype(np.uint8)
        cleaned[erase_y, erase_x, channel] = values
    return cleaned, int(sample_x.size)


def body_garment_occlusion(
    base: np.ndarray, affected: np.ndarray
) -> np.ndarray:
    """Preserve visible boot/toe pixels where the replacement crosses them."""
    height, width = base.shape[:2]
    manual = polygon_mask(
        (width, height),
        [[
            (282, 1325),
            (315, 1318),
            (390, 1298),
            (520, 1295),
            (520, 1375),
            (335, 1375),
            (278, 1351),
        ]],
    )
    boot_red = manual & red_material(base) & ~key_like(base)
    boot_and_outline = cv2.dilate(
        boot_red.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    return boot_and_outline & manual & affected & ~key_like(base)


def checker_rgba(rgba: np.ndarray) -> np.ndarray:
    height, width = rgba.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    checker = ((xx // 20 + yy // 20) % 2)[..., None]
    background = np.where(checker, 205, 155).astype(np.uint8)
    background = np.repeat(background, 3, axis=2)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(
        np.round(rgba[..., :3] * alpha + background * (1.0 - alpha)), 0, 255
    ).astype(np.uint8)


def labeled_panel(
    image: Image.Image, size: tuple[int, int], title: str
) -> Image.Image:
    panel = Image.new("RGB", (size[0], size[1] + 42), (12, 15, 20))
    scale = min(size[0] / image.width, size[1] / image.height)
    fitted = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    panel.paste(fitted, (x, y))
    ImageDraw.Draw(panel).text((10, size[1] + 13), title, fill=(255, 255, 255))
    return panel


def make_proofs(
    base: np.ndarray,
    cleaned: np.ndarray,
    result: np.ndarray,
    metal: np.ndarray,
    masks: list[tuple[str, np.ndarray, tuple[int, int, int]]],
) -> None:
    close_crop = (220, 1240, 360, 1455)
    source_panel = labeled_panel(
        Image.fromarray(checker_rgba(metal), mode="RGB"),
        (420, 645),
        "LOCKED CANONICAL METAL",
    )
    base_panel = labeled_panel(
        Image.fromarray(base, mode="RGB").crop(close_crop),
        (420, 645),
        "HAPPY BASE",
    )
    cleaned_panel = labeled_panel(
        Image.fromarray(cleaned, mode="RGB").crop(close_crop),
        (420, 645),
        "OLD HEAD CLEANED",
    )
    repaired_panel = labeled_panel(
        Image.fromarray(result, mode="RGB").crop(close_crop),
        (420, 645),
        "METAL-ONLY REPAIR",
    )
    sheet = Image.new("RGB", (1680, 687), (12, 15, 20))
    for index, panel in enumerate(
        (source_panel, base_panel, cleaned_panel, repaired_panel)
    ):
        sheet.paste(panel, (index * 420, 0))
    PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(PROOF, optimize=True)

    panel_width = 330
    panel_height = 555
    mask_sheet = Image.new(
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
        crop = Image.fromarray(shown, mode="RGB").crop(close_crop)
        scale = min(panel_width / crop.width, panel_height / crop.height)
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
        x = index * panel_width + (panel_width - crop.width) // 2
        y = (panel_height - crop.height) // 2
        mask_sheet.paste(crop, (x, y))
        ImageDraw.Draw(mask_sheet).text(
            (index * panel_width + 8, panel_height + 13),
            title,
            fill=(255, 255, 255),
        )
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    mask_sheet.save(MASK_PROOF, optimize=True)


def artifact_record(path: Path, *, pixels: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {"path": str(path), "sha256": sha256(path)}
    if pixels is not None:
        record["pixels"] = pixels
    return record


def main() -> None:
    check_hash_fences()
    base = np.asarray(Image.open(BASE).convert("RGB"), dtype=np.uint8)
    metal = np.asarray(Image.open(METAL).convert("RGBA"), dtype=np.uint8)
    if tuple(base.shape) != (1455, 1081, 3):
        raise RuntimeError(f"unexpected base shape: {base.shape}")
    if tuple(metal.shape) != (314, 312, 4):
        raise RuntimeError(f"unexpected metal shape: {metal.shape}")

    slope, intercept, fit_rows, max_residual = fit_shaft_centerline(base)
    scale, base_extent, source_extent, scale_evidence = derive_scale(
        base, metal, slope, intercept
    )
    transformed, rotation, matrix, target_socket, target_shaft_point = transform_metal(
        metal, (base.shape[1], base.shape[0]), slope, intercept, scale
    )
    patch = transformed[..., 3] >= MIN_ALPHA
    cleanup, shaft_preserve, cleanup_model_samples = old_head_cleanup(
        base, slope, intercept
    )
    cleaned, fit_samples = reconstruct_key_background(base, cleanup)
    affected = cleanup | patch
    occlusion = body_garment_occlusion(base, affected)

    result = cleaned.copy()
    result[patch] = transformed[..., :3][patch]
    result[occlusion] = base[occlusion]
    aggregate = cleanup | patch | occlusion
    changed = np.any(result != base, axis=2)

    outside_saved_delta = int(np.count_nonzero(changed & ~aggregate))
    outside_prop_delta = int(np.count_nonzero(changed & ~(cleanup | patch)))
    upper_assembly_delta = int(np.count_nonzero(changed[:500]))
    occlusion_delta = int(np.count_nonzero(changed & occlusion))
    if outside_saved_delta:
        raise RuntimeError(f"{outside_saved_delta} pixels changed outside saved mask")
    if outside_prop_delta:
        raise RuntimeError(f"{outside_prop_delta} non-prop pixels changed")
    if upper_assembly_delta:
        raise RuntimeError(f"upper pointed cap/tassel changed: {upper_assembly_delta}")
    if occlusion_delta:
        raise RuntimeError(f"body/garment occlusion is not exact: {occlusion_delta}")
    visible_patch = patch & ~occlusion
    hot_magenta = (
        visible_patch
        & (result[..., 0] >= 220)
        & (result[..., 1] <= 35)
        & (result[..., 2] >= 220)
    )
    if np.any(hot_magenta):
        raise RuntimeError(
            f"{int(np.count_nonzero(hot_magenta))} hot-magenta visible prop pixels"
        )

    for path in (
        OUTPUT,
        CLEANED_BASE,
        OLD_HEAD_MASK,
        PATCH_MASK,
        OCCLUSION_MASK,
        AGGREGATE_MASK,
        CHANGED_MASK,
        PROOF,
        MASK_PROOF,
        METADATA,
        REJECTED_FULL_COMPONENT_METADATA,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(OUTPUT, optimize=True)
    Image.fromarray(cleaned, mode="RGB").save(CLEANED_BASE, optimize=True)
    for mask, path in (
        (cleanup, OLD_HEAD_MASK),
        (patch, PATCH_MASK),
        (occlusion, OCCLUSION_MASK),
        (aggregate, AGGREGATE_MASK),
        (changed, CHANGED_MASK),
    ):
        save_mask(mask, path)
    make_proofs(
        base,
        cleaned,
        result,
        metal,
        [
            ("OLD HEAD CLEANUP", cleanup, (255, 220, 0)),
            ("RIGID METAL PATCH", patch, (0, 180, 255)),
            ("BODY/GARMENT", occlusion, (0, 255, 110)),
            ("AGGREGATE SAVED", aggregate, (255, 105, 0)),
            ("ACTUAL CHANGED", changed, (255, 255, 255)),
        ],
    )

    rejected_metadata = {
        "status": "rejected",
        "path": str(REJECTED_FULL_COMPONENT_PREVIEW),
        "sha256": sha256(REJECTED_FULL_COMPONENT_PREVIEW),
        "reason": (
            "Full canonical metal plus cloth was subjected to one rigid transform. "
            "That incorrectly locked flexible streamers to blade rotation and gravity."
        ),
        "superseded_by": str(OUTPUT),
        "model_call": False,
    }
    REJECTED_FULL_COMPONENT_METADATA.write_text(
        json.dumps(rejected_metadata, indent=2) + "\n", encoding="utf-8"
    )

    mask_records = {}
    for name, mask, path in (
        ("old_head_cleanup", cleanup, OLD_HEAD_MASK),
        ("rigid_metal_patch", patch, PATCH_MASK),
        ("hand_body_garment_occlusion", occlusion, OCCLUSION_MASK),
        ("aggregate_saved", aggregate, AGGREGATE_MASK),
        ("actual_changed", changed, CHANGED_MASK),
    ):
        mask_records[name] = {
            **artifact_record(path, pixels=int(np.count_nonzero(mask))),
            "bbox": mask_bbox(mask),
        }

    metadata = {
        "status": "pending_root_review",
        "label": "happy",
        "method": (
            "deterministic rigid-metal-only canonical repair; base pose cloth is "
            "preserved byte-exact and canonical streamers are deliberately deferred"
        ),
        "model_call": False,
        "inputs": {
            "base": artifact_record(BASE),
            "locked_active_profile": artifact_record(LOCKED_PROFILE),
            "locked_exact_crop": {
                **artifact_record(LOCKED_CROP),
                "crop_box_in_locked_profile": [0, 540, 410, 1045],
                "pixel_delta_from_locked_profile_crop": 0,
            },
            "source_v6": artifact_record(SOURCE_V6),
            "frozen_rigid_metal": artifact_record(METAL),
            "frozen_streamers_deferred_not_used": artifact_record(DEFERRED_STREAMERS),
            "material_split_metadata": artifact_record(MATERIAL_SPLIT_METADATA),
            "material_split_proof": artifact_record(MATERIAL_SPLIT_PROOF),
            "script": artifact_record(Path(__file__).resolve()),
        },
        "transform": {
            "source_socket": SOURCE_SOCKET.tolist(),
            "source_shaft_point": SOURCE_SHAFT_POINT.tolist(),
            "target_socket": target_socket.tolist(),
            "target_shaft_point": target_shaft_point.tolist(),
            "shaft_centerline": {
                "equation": "x = slope*y + intercept",
                "slope": slope,
                "intercept": intercept,
                "fit_rows": fit_rows,
                "max_kept_row_residual_pixels": max_residual,
                "fit_roi": [220, 650, 301, 1221],
            },
            "rotation_degrees": rotation,
            "uniform_scale": scale,
            "matrix": matrix.tolist(),
            "nonuniform_deformation": False,
            "base_metal_footprint_pixels": base_extent,
            "source_metal_footprint_pixels": source_extent,
            "scale_evidence": scale_evidence,
            "patch_bbox": mask_bbox(patch),
        },
        "masks": mask_records,
        "validation": {
            "base_shape_hwc": list(base.shape),
            "output_mode": "RGB",
            "output_size_wh": [base.shape[1], base.shape[0]],
            "background_fit_samples": fit_samples,
            "cleanup_model_fit_samples": cleanup_model_samples,
            "preserved_shaft_pixels_in_old_head_footprint": int(
                np.count_nonzero(shaft_preserve)
            ),
            "outside_saved_mask_delta_pixels": outside_saved_delta,
            "outside_prop_edit_scope_delta_pixels": outside_prop_delta,
            "upper_pointed_cap_and_short_tassel_delta_pixels": upper_assembly_delta,
            "hand_body_garment_occlusion_delta_pixels": occlusion_delta,
            "hot_magenta_visible_prop_pixels": int(np.count_nonzero(hot_magenta)),
            "pose_cloth_strategy": (
                "retain the happy base's upper short tassel exactly; do not place the "
                "canonical long-streamer component in this proof"
            ),
        },
        "outputs": {
            "repaired": artifact_record(OUTPUT),
            "cleaned_base_proof_stage": artifact_record(CLEANED_BASE),
            "close_proof": artifact_record(PROOF),
            "mask_proof": artifact_record(MASK_PROOF),
        },
        "rejected_provenance": {
            "full_component_rigid_preview": artifact_record(
                REJECTED_FULL_COMPONENT_PREVIEW
            ),
            "metadata": artifact_record(REJECTED_FULL_COMPONENT_METADATA),
            "reason": rejected_metadata["reason"],
        },
        "acceptance": {
            "automated": "pass",
            "visual": "pass_original_resolution_manual_inspection",
            "manual_notes": [
                "one coherent lower metal head is aligned to the fitted shaft centerline",
                "the canonical blade, guards, and socket remain legible at the derived scale",
                "the visible boot/toe pixels occlude the replacement without anatomy loss",
                "the removed butt-cap footprint has no detached slivers or cleanup seam",
                "the upper pointed cap and short pose-specific tassel remain unchanged",
            ],
            "decision": "accept_as_deterministic_metal_only_proof_for_root_review",
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
