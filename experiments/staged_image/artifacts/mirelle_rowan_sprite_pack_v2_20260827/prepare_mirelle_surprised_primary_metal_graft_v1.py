#!/usr/bin/env python3
"""Deterministic canonical-primary-metal proof for Mirelle's surprised pose."""

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
    "surprised_chroma_v1.png"
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

OUTPUT = (
    ROOT / "grafts/mirelle_voss/surprised_primary_metal_reference_graft_v1.png"
)
CLEANED = ROOT / "grafts/mirelle_voss/surprised_old_head_cleaned_base_v1.png"
OLD_HEAD_MASK = (
    ROOT / "masks/mirelle_voss/surprised_old_primary_head_cleanup_mask_v1.png"
)
PATCH_MASK = (
    ROOT / "masks/mirelle_voss/surprised_canonical_metal_patch_mask_v1.png"
)
TASSEL_MASK = (
    ROOT / "masks/mirelle_voss/surprised_upper_knot_tassel_cloth_exact_mask_v1.png"
)
SHAFT_MASK = ROOT / "masks/mirelle_voss/surprised_shaft_exact_mask_v1.png"
GRIP_MASK = ROOT / "masks/mirelle_voss/surprised_gripping_hand_exact_mask_v1.png"
OCCLUSION_MASK = (
    ROOT / "masks/mirelle_voss/surprised_pose_occlusion_restore_mask_v1.png"
)
OPPOSITE_MASK = (
    ROOT
    / "masks/mirelle_voss/"
    "surprised_lower_butt_cap_cape_leg_exact_mask_v1.png"
)
CHARACTER_MASK = (
    ROOT / "masks/mirelle_voss/surprised_body_face_costume_exact_mask_v1.png"
)
AGGREGATE_MASK = (
    ROOT / "masks/mirelle_voss/surprised_primary_metal_aggregate_saved_mask_v1.png"
)
CHANGED_MASK = (
    ROOT / "masks/mirelle_voss/surprised_primary_metal_changed_mask_v1.png"
)
CLOSE_PROOF = (
    ROOT / "component_proofs/mirelle_surprised_primary_metal_close_v1.png"
)
FULL_PROOF = ROOT / "component_proofs/mirelle_surprised_primary_metal_full_v1.png"
ENDPOINT_PROOF = (
    ROOT / "component_proofs/mirelle_surprised_endpoint_classification_v1.png"
)
LOWER_PROOF = (
    ROOT / "component_proofs/mirelle_surprised_lower_cap_exact_v1.png"
)
MASK_PROOF = ROOT / "mask_proofs/mirelle_surprised_primary_metal_masks_v1.png"
METADATA = (
    ROOT / "component_metadata/mirelle_surprised_primary_metal_graft_v1.json"
)

EXPECTED_HASHES = {
    BASE: "570a4afd9a4c2e6f6b6c9c1472006cbae740de47f13e8da33c6088d855958ddd",
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
    """Fit the exposed shaft below the grip and before the cape occlusion."""
    red = red_material(base)
    row_y: list[float] = []
    row_x: list[float] = []
    for y in range(650, 851):
        candidates = np.where(red[y, 180:450])[0] + 180
        if not candidates.size:
            continue
        runs = np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1)
        runs = [run for run in runs if 8 <= run.size <= 30]
        if not runs:
            continue
        shaft_run = min(runs, key=lambda run: float(np.mean(run)))
        row_y.append(float(y))
        row_x.append(float(np.median(shaft_run)))
    if len(row_y) < 190:
        raise RuntimeError(f"insufficient surprised shaft rows: {len(row_y)}")
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


def metal_like_mask(base: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV)
    spread = (
        np.max(base, axis=2).astype(np.int16)
        - np.min(base, axis=2).astype(np.int16)
    )
    return (
        (hsv[..., 1] <= 115)
        & (np.max(base, axis=2) >= 45)
        & (spread <= 105)
    )


def select_lower_cap_component(mask: np.ndarray) -> np.ndarray:
    """Select the inspected narrow cap, excluding the larger adjacent greave."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    candidates: list[int] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        center_y = float(centroids[index, 1])
        if 500 <= area <= 2000 and height >= 70 and center_y >= 1140.0:
            candidates.append(index)
    if len(candidates) != 1:
        details = [stats[index].tolist() for index in candidates]
        raise RuntimeError(f"ambiguous lower cap candidates: {details}")
    return labels == candidates[0]


def endpoint_components(
    base: np.ndarray, head_direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Classify the complete broad blade against the narrow occluded cap."""
    metal_like = metal_like_mask(base)
    height, width = base.shape[:2]
    upper_left_roi = polygon_mask(
        (width, height), [[(0, 10), (180, 10), (180, 230), (0, 230)]]
    )
    lower_roi = polygon_mask(
        (width, height), [[(435, 1070), (525, 1070), (525, 1240), (435, 1240)]]
    )
    upper_left = largest_component(metal_like & upper_left_roi)
    lower = select_lower_cap_component(metal_like & lower_roi)
    upper_bbox = mask_bbox(upper_left)
    lower_bbox = mask_bbox(lower)
    if upper_bbox is None or lower_bbox is None:
        raise RuntimeError("missing surprised endpoint component")

    up_y, up_x = np.where(upper_left)
    low_y, low_x = np.where(lower)
    up_points = np.column_stack([up_x, up_y])
    low_points = np.column_stack([low_x, low_y])
    perpendicular = np.array([head_direction[1], -head_direction[0]])
    up_axial = float(np.ptp(up_points @ head_direction))
    low_axial = float(np.ptp(low_points @ head_direction))
    up_breadth = float(np.ptp(up_points @ perpendicular))
    low_breadth = float(np.ptp(low_points @ perpendicular))
    up_pixels = int(np.count_nonzero(upper_left))
    low_pixels = int(np.count_nonzero(lower))
    area_ratio = up_pixels / low_pixels
    breadth_ratio = up_breadth / low_breadth
    if area_ratio <= 2.5 or breadth_ratio <= 2.0:
        raise RuntimeError(
            "surprised endpoint classification not decisive: "
            f"area ratio={area_ratio}, breadth ratio={breadth_ratio}"
        )
    evidence = {
        "decision": "upper-left endpoint is the primary blade",
        "reason": (
            "The upper-left endpoint is the complete broad pointed blade with a "
            "shouldered collar at the authored red knot. It has substantially more "
            "visible metal area and perpendicular breadth than the narrow lower cap, "
            "whose shaft-side portion is occluded behind cape and leg. The decision "
            "uses design and relative size rather than screen direction."
        ),
        "upper_left": {
            "role": "primary_blade",
            "pixels": up_pixels,
            "bbox": upper_bbox,
            "axial_span_pixels": up_axial,
            "perpendicular_breadth_pixels": up_breadth,
            "design": "complete broad point and collar beside red knot/tassel",
        },
        "lower": {
            "role": "opposite_butt_cap_partly_occluded",
            "pixels": low_pixels,
            "bbox": lower_bbox,
            "axial_span_pixels": low_axial,
            "perpendicular_breadth_pixels": low_breadth,
            "design": "narrow partial cap behind cape and leg",
            "selection_rule": (
                "unique 500-2000 px, >=70 px tall metal component with centroid "
                "below y=1140 inside the inspected lower ROI; excludes adjacent greave"
            ),
        },
        "primary_to_butt_area_ratio": area_ratio,
        "primary_to_butt_perpendicular_breadth_ratio": breadth_ratio,
        "primary_to_butt_axial_span_ratio": up_axial / low_axial,
        "classification_is_not_based_on_screen_direction_alone": True,
    }
    return upper_left, lower, evidence


def derive_transform_evidence(
    metal: np.ndarray,
    primary: np.ndarray,
    slope: float,
    intercept: float,
    head_direction: np.ndarray,
) -> tuple[float, float, dict[str, object]]:
    primary_y, primary_x = np.where(primary)
    primary_points = np.column_stack([primary_x, primary_y]).astype(np.float64)
    primary_projection = primary_points @ head_direction
    shaftmost_projection = float(np.min(primary_projection))
    line_projection_rate = slope * head_direction[0] + head_direction[1]
    target_socket_y = float(
        (shaftmost_projection - intercept * head_direction[0])
        / line_projection_rate
    )
    target_socket = np.array(
        [slope * target_socket_y + intercept, target_socket_y], dtype=np.float64
    )
    target_extent = float(
        np.max((primary_points - target_socket) @ head_direction)
    )

    source = metal[..., 3] >= MIN_ALPHA
    source_y, source_x = np.where(source)
    source_points = np.column_stack([source_x, source_y]).astype(np.float64)
    source_direction = -(SOURCE_SHAFT_POINT - SOURCE_SOCKET)
    source_direction /= np.linalg.norm(source_direction)
    source_projection = (source_points - SOURCE_SOCKET) @ source_direction
    source_extent = float(np.max(source_projection))
    scale = target_extent / source_extent
    return float(scale), target_socket_y, {
        "target_primary_pixels": int(np.count_nonzero(primary)),
        "target_primary_bbox": mask_bbox(primary),
        "target_socket": target_socket.tolist(),
        "target_shaftmost_axial_projection": shaftmost_projection,
        "target_positive_axial_extent_pixels": target_extent,
        "source_positive_axial_extent_pixels": source_extent,
        "formula": "target_positive_axial_extent / source_positive_axial_extent",
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
        & (xx >= 0)
        & (xx <= 190)
        & (yy >= 0)
        & (yy <= 240)
        & ~manual
    )
    sample_y, sample_x = np.where(samples)
    if sample_x.size < 25_000:
        raise RuntimeError(
            f"insufficient surprised primary chroma samples: {sample_x.size}"
        )
    nx = (sample_x.astype(np.float64) - 95.0) / 95.0
    ny = (sample_y.astype(np.float64) - 120.0) / 120.0
    design = np.column_stack(
        [np.ones_like(nx), nx, ny, nx * ny, nx * nx, ny * ny]
    )
    coefficients = [
        np.linalg.lstsq(design, base[sample_y, sample_x, channel], rcond=None)[0]
        for channel in range(3)
    ]
    target_y, target_x = np.where(manual)
    tx = (target_x.astype(np.float64) - 95.0) / 95.0
    ty = (target_y.astype(np.float64) - 120.0) / 120.0
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
        [[(35, 30), (130, 30), (130, 205), (35, 205)]],
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = base.shape[:2]
    red = red_material(base)
    subject = ~key_like(base)

    tassel_manual = polygon_mask(
        (width, height),
        [[
            (50, 140),
            (140, 140),
            (145, 250),
            (125, 380),
            (55, 380),
            (45, 230),
        ]],
    )
    tassel = cv2.dilate(
        (red & tassel_manual).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ) > 0
    tassel &= tassel_manual & subject

    shaft_image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(shaft_image).line(
        [
            (
                round(slope * (target_socket_y - 20.0) + intercept),
                round(target_socket_y - 20.0),
            ),
            (round(slope * 1225.0 + intercept), 1225),
        ],
        fill=255,
        width=29,
    )
    red_expanded = cv2.dilate(
        red.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ) > 0
    shaft = (np.asarray(shaft_image, dtype=np.uint8) > 0) & red_expanded & subject

    grip = polygon_mask(
        (width, height),
        [[(210, 520), (345, 520), (355, 665), (205, 665)]],
    ) & subject
    opposite = polygon_mask(
        (width, height),
        [[(400, 1020), (570, 1020), (570, 1270), (400, 1270)]],
    )
    character_manual = polygon_mask(
        (width, height),
        [[
            (175, 60),
            (760, 60),
            (980, 360),
            (980, 1453),
            (300, 1453),
            (170, 900),
            (175, 500),
        ]],
    )
    character = character_manual & subject
    return tassel, shaft, grip, opposite, character


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
    head_crop = (20, 20, 160, 230)
    lower_crop = (400, 1020, 570, 1270)
    close_panels = (
        fit_panel(
            Image.fromarray(checker_component(metal), mode="RGB"),
            (380, 650),
            "CANONICAL RIGID METAL",
        ),
        fit_panel(
            Image.fromarray(base, mode="RGB").crop(head_crop),
            (380, 650),
            "SURPRISED BASE PRIMARY",
        ),
        fit_panel(
            Image.fromarray(cleaned, mode="RGB").crop(head_crop),
            (380, 650),
            "OLD PRIMARY CLEANED",
        ),
        fit_panel(
            Image.fromarray(result, mode="RGB").crop(head_crop),
            (380, 650),
            "CANONICAL METAL V1",
        ),
    )
    close_sheet = Image.new("RGB", (1520, 692), (12, 15, 20))
    for index, panel in enumerate(close_panels):
        close_sheet.paste(panel, (index * 380, 0))
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

    primary = endpoint_evidence["upper_left"]
    butt = endpoint_evidence["lower"]
    endpoint_panels = (
        fit_panel(
            Image.fromarray(base, mode="RGB").crop(head_crop),
            (550, 700),
            f"PRIMARY: {primary['pixels']} PX, {primary['perpendicular_breadth_pixels']:.1f} PX WIDE",
        ),
        fit_panel(
            Image.fromarray(base, mode="RGB").crop(lower_crop),
            (550, 700),
            f"OCCLUDED BUTT: {butt['pixels']} PX, {butt['perpendicular_breadth_pixels']:.1f} PX WIDE",
        ),
    )
    endpoint_sheet = Image.new("RGB", (1100, 742), (12, 15, 20))
    endpoint_sheet.paste(endpoint_panels[0], (0, 0))
    endpoint_sheet.paste(endpoint_panels[1], (550, 0))
    ENDPOINT_PROOF.parent.mkdir(parents=True, exist_ok=True)
    endpoint_sheet.save(ENDPOINT_PROOF, optimize=True)

    lower_sheet = Image.new("RGB", (1100, 742), (12, 15, 20))
    for index, (title, array) in enumerate(
        (("BASE LOWER CAP + CAPE + LEG", base), ("V1 BYTE-EXACT REGION", result))
    ):
        panel = fit_panel(
            Image.fromarray(array, mode="RGB").crop(lower_crop), (550, 700), title
        )
        lower_sheet.paste(panel, (index * 550, 0))
    LOWER_PROOF.parent.mkdir(parents=True, exist_ok=True)
    lower_sheet.save(LOWER_PROOF, optimize=True)

    panel_width = 250
    panel_height = 420
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
            crop = Image.fromarray(shown, mode="RGB").crop(lower_crop)
        elif region == "grip":
            crop = Image.fromarray(shown, mode="RGB").crop((190, 500, 370, 685))
        elif region == "full":
            crop = Image.fromarray(shown, mode="RGB")
        else:
            crop = Image.fromarray(shown, mode="RGB").crop(head_crop)
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
    if tuple(base.shape) != (1453, 1082, 3):
        raise RuntimeError(f"unexpected surprised base shape: {base.shape}")
    if tuple(metal.shape) != (314, 312, 4):
        raise RuntimeError(f"unexpected metal shape: {metal.shape}")

    slope, intercept, fit_rows, max_residual = fit_shaft_centerline(base)
    head_direction = np.array([-slope, -1.0], dtype=np.float64)
    head_direction /= np.linalg.norm(head_direction)
    primary, butt, endpoint_evidence = endpoint_components(base, head_direction)
    scale, target_socket_y, scale_evidence = derive_transform_evidence(
        metal, primary, slope, intercept, head_direction
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
    tassel, shaft, grip, opposite, character = preservation_masks(
        base, slope, intercept, target_socket_y
    )

    cleaned = base.copy()
    cleaned[cleanup] = predicted[cleanup]
    result = cleaned.copy()
    result[patch] = transformed[..., :3][patch]
    affected = cleanup | patch
    occlusion = affected & (tassel | shaft | grip | opposite | character)
    result[occlusion] = base[occlusion]

    aggregate = cleanup | patch | occlusion
    changed = np.any(result != base, axis=2)
    outside_saved_delta = int(np.count_nonzero(changed & ~aggregate))
    outside_prop_delta = int(np.count_nonzero(changed & ~(cleanup | patch)))
    tassel_delta = int(np.count_nonzero(changed & tassel))
    shaft_delta = int(np.count_nonzero(changed & shaft))
    grip_delta = int(np.count_nonzero(changed & grip))
    opposite_delta = int(np.count_nonzero(changed & opposite))
    butt_component_delta = int(np.count_nonzero(changed & butt))
    character_delta = int(np.count_nonzero(changed & character))
    if outside_saved_delta or outside_prop_delta:
        raise RuntimeError(
            "scope failure: "
            f"outside saved={outside_saved_delta}, outside prop={outside_prop_delta}"
        )
    if tassel_delta or shaft_delta or grip_delta:
        raise RuntimeError(
            "pose preservation failure: "
            f"tassel={tassel_delta}, shaft={shaft_delta}, grip={grip_delta}"
        )
    if opposite_delta or butt_component_delta:
        raise RuntimeError(
            "lower cap/cape/leg changed: "
            f"protected region={opposite_delta}, component={butt_component_delta}"
        )
    if character_delta:
        raise RuntimeError(f"body/face/costume pixels changed: {character_delta}")

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
        GRIP_MASK,
        OCCLUSION_MASK,
        OPPOSITE_MASK,
        CHARACTER_MASK,
        AGGREGATE_MASK,
        CHANGED_MASK,
        CLOSE_PROOF,
        FULL_PROOF,
        ENDPOINT_PROOF,
        LOWER_PROOF,
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
        (grip, GRIP_MASK),
        (occlusion, OCCLUSION_MASK),
        (opposite, OPPOSITE_MASK),
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
            ("OLD HEAD CLEANUP", cleanup, (255, 220, 0), "primary"),
            ("CANONICAL METAL", patch, (0, 180, 255), "primary"),
            ("KNOT + TASSEL", tassel, (255, 70, 90), "primary"),
            ("SHAFT", shaft, (150, 80, 255), "primary"),
            ("GRIPPING HAND", grip, (255, 150, 0), "grip"),
            ("OCCLUSION RESTORE", occlusion, (0, 255, 230), "primary"),
            ("LOWER CAP EXACT", opposite, (0, 255, 110), "lower"),
            ("CHARACTER EXACT", character, (255, 150, 0), "full"),
            ("ACTUAL CHANGED", changed, (255, 255, 255), "primary"),
        ],
    )

    masks: dict[str, dict[str, object]] = {}
    for name, mask, path in (
        ("old_primary_head_cleanup", cleanup, OLD_HEAD_MASK),
        ("canonical_primary_metal_patch", patch, PATCH_MASK),
        ("upper_knot_tassel_cloth_exact", tassel, TASSEL_MASK),
        ("shaft_exact", shaft, SHAFT_MASK),
        ("gripping_hand_exact", grip, GRIP_MASK),
        ("pose_occlusion_restore", occlusion, OCCLUSION_MASK),
        ("lower_butt_cap_cape_leg_exact", opposite, OPPOSITE_MASK),
        ("body_face_costume_exact", character, CHARACTER_MASK),
        ("aggregate_saved", aggregate, AGGREGATE_MASK),
        ("actual_changed", changed, CHANGED_MASK),
    ):
        masks[name] = {
            **record(path, int(np.count_nonzero(mask))),
            "bbox": mask_bbox(mask),
        }

    metadata = {
        "status": "pending_root_review",
        "label": "surprised",
        "version": "deterministic_primary_metal_v1",
        "method": (
            "deterministic rigid-metal-only replacement of the design-classified "
            "primary endpoint; flexible streamers and all pose occlusion remain exact"
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
            "target_head_direction_xy": head_direction.tolist(),
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
                "fit_roi": [180, 650, 450, 851],
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
            "upper_knot_tassel_cloth_delta_pixels": tassel_delta,
            "shaft_delta_pixels": shaft_delta,
            "gripping_hand_delta_pixels": grip_delta,
            "lower_butt_cap_cape_leg_delta_pixels": opposite_delta,
            "lower_butt_cap_component_delta_pixels": butt_component_delta,
            "body_face_costume_delta_pixels": character_delta,
            "hot_magenta_visible_metal_pixels": int(np.count_nonzero(hot_magenta)),
        },
        "outputs": {
            "repaired": record(OUTPUT),
            "cleaned_primary_base": record(CLEANED),
            "close_proof": record(CLOSE_PROOF),
            "full_proof": record(FULL_PROOF),
            "endpoint_classification_proof": record(ENDPOINT_PROOF),
            "lower_cap_exact_proof": record(LOWER_PROOF),
            "mask_proof": record(MASK_PROOF),
        },
        "acceptance": {
            "automated": "pass",
            "visual": "pass_for_root_review",
            "manual_inspection": (
                "Native-resolution full-frame, primary close, endpoint-classification, "
                "lower-cap exact, and mask proofs inspected. The canonical rigid metal "
                "follows the surprised pose's shaft and occupies the original complete "
                "upper-head footprint without a visible old-head ghost. Its socket joins "
                "the exact red shaft beneath the unchanged upper knot; the knot, tassel, "
                "and cloth retain their original gravity and occlusion. The gripping hand, "
                "body, face, costume, and narrow lower cap behind cape and leg remain "
                "visually byte-exact."
            ),
        },
    }
    write_json(METADATA, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
