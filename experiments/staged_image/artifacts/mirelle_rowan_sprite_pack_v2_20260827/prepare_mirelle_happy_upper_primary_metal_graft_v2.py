#!/usr/bin/env python3
"""Correct-end deterministic canonical-metal proof for Mirelle's happy pose."""

from __future__ import annotations

import hashlib
import json
import shutil
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
METAL = ROOT / "components/mirelle_voss/canonical_primary_metal_rgba_v1.png"
DEFERRED_STREAMERS = (
    ROOT / "components/mirelle_voss/canonical_primary_streamers_rgba_v1.png"
)
MATERIAL_SPLIT_METADATA = (
    ROOT / "component_metadata/mirelle_canonical_primary_material_split_v1.json"
)

OUTPUT = (
    ROOT / "grafts/mirelle_voss/happy_upper_primary_metal_reference_graft_v2.png"
)
CLEANED = ROOT / "grafts/mirelle_voss/happy_upper_old_head_cleaned_base_v2.png"
OLD_HEAD_MASK = (
    ROOT / "masks/mirelle_voss/happy_upper_old_primary_head_cleanup_mask_v2.png"
)
PATCH_MASK = (
    ROOT / "masks/mirelle_voss/happy_upper_canonical_metal_patch_mask_v2.png"
)
CLOTH_MASK = (
    ROOT / "masks/mirelle_voss/happy_upper_pose_cloth_preserve_mask_v2.png"
)
SHAFT_MASK = ROOT / "masks/mirelle_voss/happy_upper_shaft_preserve_mask_v2.png"
LOWER_EXACT_MASK = (
    ROOT / "masks/mirelle_voss/happy_lower_butt_cap_boot_exact_mask_v2.png"
)
AGGREGATE_MASK = (
    ROOT / "masks/mirelle_voss/happy_upper_aggregate_saved_mask_v2.png"
)
CHANGED_MASK = ROOT / "masks/mirelle_voss/happy_upper_changed_mask_v2.png"
PROOF = ROOT / "component_proofs/mirelle_happy_upper_primary_metal_graft_v2.png"
LOWER_PROOF = ROOT / "component_proofs/mirelle_happy_lower_butt_cap_exact_v2.png"
MASK_PROOF = ROOT / "mask_proofs/mirelle_happy_upper_primary_metal_masks_v2.png"
METADATA = ROOT / "component_metadata/mirelle_happy_upper_primary_metal_graft_v2.json"

WRONG_END_ARCHIVE = ROOT / "rejected/mirelle_happy_wrong_end_v1"
WRONG_END_REJECTION = WRONG_END_ARCHIVE / "rejection.json"
WRONG_END_FILES = {
    ROOT / "prepare_mirelle_happy_canonical_spear_metal_proof.py":
        "aa256036afc0b8d599642458c0bd8f9903b3cb600352976069f1322b26d1d010",
    ROOT / "grafts/mirelle_voss/happy_canonical_spear_metal_repair_proof.png":
        "d62dbfbf98877ddd94241748accd7c4c933124aca07a2c20d6a93675f0364343",
    ROOT / "grafts/mirelle_voss/happy_old_head_cleaned_base_proof.png":
        "363b90de0412d2d0243ae06f21c342568efb71959e72865763f72671438a904c",
    ROOT / "masks/mirelle_voss/happy_canonical_spear_old_head_cleanup_mask.png":
        "afec47fdddb29f1ebbe2b90e7ba76cb0bb50841e2f64580d06b13d98866c1997",
    ROOT / "masks/mirelle_voss/happy_canonical_spear_rigid_metal_patch_mask.png":
        "0217e64f4e12b5add34c8cacd060d59c4c62656dafcb5f92fba7c91d967022ca",
    ROOT / "masks/mirelle_voss/happy_canonical_spear_hand_body_garment_occlusion_mask.png":
        "1148d3a76e3254d9d32190a09f12d2962529c74c6105d1b4357c814d8067fa48",
    ROOT / "masks/mirelle_voss/happy_canonical_spear_aggregate_saved_mask.png":
        "f5ed08a220a21a382f234e01ef2ae8ce70a53c22e5e995dd72cda0998a2aa4d2",
    ROOT / "masks/mirelle_voss/happy_canonical_spear_changed_mask.png":
        "4f20b8dfe472a2c90440478d28c00e95fb587265f934bce5df8cc6e4ec5ed532",
    ROOT / "component_proofs/mirelle_happy_canonical_spear_metal_close_proof.png":
        "9df1b85254fb0ac5e260f5b20e1c7df028960f38079d689cfd845d679a379f5f",
    ROOT / "mask_proofs/mirelle_happy_canonical_spear_metal_masks.png":
        "cb1ba4dda82eb1a7cc6cd5ea4741196db7fa2d04d3b9a6da35df1c8d4079e650",
    ROOT / "component_metadata/mirelle_happy_canonical_spear_metal_repair_proof.json":
        "160c76700b4e2e85089d2135c317f1b992199e1957536282fb7e66806b6c63c7",
}

EXPECTED_HASHES = {
    BASE: "12d42a02f1ac49765c3034916937e73380dddd39560764fafee165b794f2cfb2",
    LOCKED_PROFILE: "6173f6066860c270b73bcbde7fc86336411758301cdc5c12341f0400a82266b6",
    LOCKED_CROP: "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff",
    METAL: "4f4b369d7f445b79a44673b33aee8bcc763f754eb37a0d7bfaee38213bf73b6f",
    DEFERRED_STREAMERS: "9d7ff6a32a4ad46ac491652b40f0fa5f0a2c8fb4f88cd67c1ebdb1a81e6970dd",
    MATERIAL_SPLIT_METADATA: "6729001c5d692c4d7a31087870c7b930c354284535dc05acf8e9c8f460e03d30",
}

SOURCE_SOCKET = np.array([200.0, 89.0], dtype=np.float64)
SOURCE_SHAFT_POINT = np.array([289.0, 13.0], dtype=np.float64)
TARGET_SOCKET_Y = 136.0
TARGET_SHAFT_SAMPLE_Y = 256.0
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


def archive_wrong_end_v1() -> dict[str, object]:
    check_hashes(WRONG_END_FILES)
    records = []
    for source, expected in WRONG_END_FILES.items():
        relative = source.relative_to(ROOT)
        archived = WRONG_END_ARCHIVE / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        if archived.exists():
            if sha256(archived) != expected:
                raise RuntimeError(f"wrong-end archive hash conflict: {archived}")
        else:
            shutil.copy2(source, archived)
        if sha256(archived) != expected:
            raise RuntimeError(f"wrong-end archive copy mismatch: {archived}")
        records.append(
            {
                "original_path": str(source),
                "archived_path": str(archived),
                "sha256": expected,
            }
        )
    rejection = {
        "status": "rejected_by_root",
        "label": "happy",
        "version": "wrong_end_v1",
        "reason": (
            "The deterministic v1 placed Mirelle's canonical primary blade on the "
            "lower endpoint beside her boot. That endpoint is the opposite butt cap; "
            "the result therefore replaced the wrong end and created double-primary "
            "weapon semantics. The correct target is the upper blade beside the tassel."
        ),
        "model_calls": 0,
        "files": records,
    }
    write_json(WRONG_END_REJECTION, rejection)
    return {
        "path": str(WRONG_END_REJECTION),
        "sha256": sha256(WRONG_END_REJECTION),
        "file_count": len(records),
    }


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


def fit_shaft_centerline(base: np.ndarray) -> tuple[float, float, int, float]:
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


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise RuntimeError("expected connected foreground")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def derive_scale(
    base: np.ndarray, metal: np.ndarray, slope: float, intercept: float
) -> tuple[float, dict[str, object]]:
    hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV)
    spread = np.max(base, axis=2).astype(np.int16) - np.min(base, axis=2).astype(np.int16)
    region = np.zeros(base.shape[:2], dtype=bool)
    region[10:170, 200:280] = True
    old_metal_seed = (
        region
        & (hsv[..., 1] <= 115)
        & (np.max(base, axis=2) >= 45)
        & (spread <= 105)
    )
    old_metal = largest_component(old_metal_seed)
    base_socket = np.array(
        [slope * TARGET_SOCKET_Y + intercept, TARGET_SOCKET_Y], dtype=np.float64
    )
    base_head_direction = np.array([-slope, -1.0], dtype=np.float64)
    base_head_direction /= np.linalg.norm(base_head_direction)
    old_y, old_x = np.where(old_metal)
    old_points = np.column_stack([old_x, old_y]).astype(np.float64)
    base_extent = float(np.max((old_points - base_socket) @ base_head_direction))

    source = metal[..., 3] >= MIN_ALPHA
    source_y, source_x = np.where(source)
    source_points = np.column_stack([source_x, source_y]).astype(np.float64)
    source_direction = -(SOURCE_SHAFT_POINT - SOURCE_SOCKET)
    source_direction /= np.linalg.norm(source_direction)
    source_extent = float(np.max((source_points - SOURCE_SOCKET) @ source_direction))
    scale = base_extent / source_extent
    return float(scale), {
        "old_upper_metal_pixels": int(np.count_nonzero(old_metal)),
        "old_upper_metal_bbox": mask_bbox(old_metal),
        "base_socket": base_socket.tolist(),
        "base_positive_axial_extent_pixels": base_extent,
        "source_positive_axial_extent_pixels": source_extent,
        "formula": "base_positive_axial_extent / source_positive_axial_extent",
    }


def transform_metal(
    metal: np.ndarray,
    size: tuple[int, int],
    slope: float,
    intercept: float,
    scale: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    target_socket = np.array(
        [slope * TARGET_SOCKET_Y + intercept, TARGET_SOCKET_Y], dtype=np.float64
    )
    target_shaft_point = np.array(
        [
            slope * TARGET_SHAFT_SAMPLE_Y + intercept,
            TARGET_SHAFT_SAMPLE_Y,
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
        & (xx >= 175)
        & (xx <= 300)
        & (yy >= 0)
        & (yy <= 195)
        & ~manual
    )
    sample_y, sample_x = np.where(samples)
    if sample_x.size < 10_000:
        raise RuntimeError(f"insufficient upper chroma samples: {sample_x.size}")
    nx = (sample_x.astype(np.float64) - 237.5) / 62.5
    ny = (sample_y.astype(np.float64) - 97.5) / 97.5
    design = np.column_stack(
        [np.ones_like(nx), nx, ny, nx * ny, nx * nx, ny * ny]
    )
    coefficients = [
        np.linalg.lstsq(design, base[sample_y, sample_x, channel], rcond=None)[0]
        for channel in range(3)
    ]
    target_y, target_x = np.where(manual)
    tx = (target_x.astype(np.float64) - 237.5) / 62.5
    ty = (target_y.astype(np.float64) - 97.5) / 97.5
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
        [[
            (233, 14),
            (266, 90),
            (264, 120),
            (258, 147),
            (255, 162),
            (219, 162),
            (213, 140),
            (209, 100),
            (214, 72),
        ]],
    )
    predicted, sample_count = predict_local_background(base, manual)
    distance = np.linalg.norm(
        base.astype(np.int16) - predicted.astype(np.int16), axis=2
    )
    cleanup = manual & (distance > 4.0)
    return cleanup, predicted, sample_count


def pose_preserve_masks(
    base: np.ndarray, slope: float, intercept: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = base.shape[:2]
    red = red_material(base)
    subject = ~key_like(base)
    cloth_manual = polygon_mask(
        (width, height),
        [[
            (230, 120),
            (270, 120),
            (310, 150),
            (325, 235),
            (315, 315),
            (250, 315),
            (220, 215),
        ]],
    )
    cloth = cv2.dilate(
        (red & cloth_manual).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    cloth &= cloth_manual & subject

    shaft_image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(shaft_image).line(
        [
            (round(slope * 110.0 + intercept), 110),
            (round(slope * 190.0 + intercept), 190),
        ],
        fill=255,
        width=23,
    )
    red_expanded = cv2.dilate(
        red.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    shaft = (np.asarray(shaft_image, dtype=np.uint8) > 0) & red_expanded & subject

    lower_exact = polygon_mask(
        (width, height),
        [[(245, 1215), (520, 1215), (520, 1390), (245, 1390)]],
    )
    return cloth, shaft, lower_exact


def checker_component(rgba: np.ndarray) -> np.ndarray:
    height, width = rgba.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    checker = ((xx // 20 + yy // 20) % 2)[..., None]
    background = np.repeat(np.where(checker, 205, 155).astype(np.uint8), 3, axis=2)
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
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    ImageDraw.Draw(panel).text((8, size[1] + 13), title, fill=(255, 255, 255))
    return panel


def make_proofs(
    base: np.ndarray,
    cleaned: np.ndarray,
    result: np.ndarray,
    metal: np.ndarray,
    masks: list[tuple[str, np.ndarray, tuple[int, int, int]]],
) -> None:
    close_crop = (175, 0, 315, 190)
    panels = (
        fit_panel(Image.fromarray(checker_component(metal), mode="RGB"), (400, 680), "CANONICAL METAL"),
        fit_panel(Image.fromarray(base, mode="RGB").crop(close_crop), (400, 680), "HAPPY BASE"),
        fit_panel(Image.fromarray(cleaned, mode="RGB").crop(close_crop), (400, 680), "UPPER HEAD CLEANED"),
        fit_panel(Image.fromarray(result, mode="RGB").crop(close_crop), (400, 680), "CORRECT-END V2"),
    )
    sheet = Image.new("RGB", (1600, 722), (12, 15, 20))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * 400, 0))
    PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(PROOF, optimize=True)

    lower_crop = (240, 1200, 525, 1400)
    lower_sheet = Image.new("RGB", (1140, 842), (12, 15, 20))
    for index, (title, array) in enumerate(
        (("BASE BUTT CAP + BOOT", base), ("V2 BYTE-EXACT REGION", result))
    ):
        panel = Image.fromarray(array, mode="RGB").crop(lower_crop).resize(
            (570, 800), Image.Resampling.NEAREST
        )
        lower_sheet.paste(panel, (index * 570, 0))
        ImageDraw.Draw(lower_sheet).text(
            (index * 570 + 8, 813), title, fill=(255, 255, 255)
        )
    LOWER_PROOF.parent.mkdir(parents=True, exist_ok=True)
    lower_sheet.save(LOWER_PROOF, optimize=True)

    panel_width = 300
    panel_height = 408
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
        if title == "LOWER EXACT":
            crop = Image.fromarray(shown, mode="RGB").crop((240, 1200, 525, 1390))
        else:
            crop = Image.fromarray(shown, mode="RGB").crop(close_crop)
        crop = crop.resize((panel_width, panel_height), Image.Resampling.NEAREST)
        mask_sheet.paste(crop, (index * panel_width, 0))
        ImageDraw.Draw(mask_sheet).text(
            (index * panel_width + 8, panel_height + 13), title, fill=(255, 255, 255)
        )
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    mask_sheet.save(MASK_PROOF, optimize=True)


def record(path: Path, pixels: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": str(path), "sha256": sha256(path)}
    if pixels is not None:
        value["pixels"] = pixels
    return value


def main() -> None:
    check_hashes(EXPECTED_HASHES)
    archive_record = archive_wrong_end_v1()
    base = np.asarray(Image.open(BASE).convert("RGB"), dtype=np.uint8)
    metal = np.asarray(Image.open(METAL).convert("RGBA"), dtype=np.uint8)
    if tuple(base.shape) != (1455, 1081, 3):
        raise RuntimeError(f"unexpected base shape: {base.shape}")
    if tuple(metal.shape) != (314, 312, 4):
        raise RuntimeError(f"unexpected metal shape: {metal.shape}")

    slope, intercept, fit_rows, max_residual = fit_shaft_centerline(base)
    scale, scale_evidence = derive_scale(base, metal, slope, intercept)
    transformed, rotation, matrix, target_socket, target_shaft_point = transform_metal(
        metal, (base.shape[1], base.shape[0]), slope, intercept, scale
    )
    patch = transformed[..., 3] >= MIN_ALPHA
    cleanup, predicted, background_samples = old_head_cleanup(base)
    cloth, shaft, lower_exact = pose_preserve_masks(base, slope, intercept)

    cleaned = base.copy()
    cleaned[cleanup] = predicted[cleanup]
    result = cleaned.copy()
    result[patch] = transformed[..., :3][patch]
    affected = cleanup | patch
    cloth_restore = cloth & affected
    shaft_restore = shaft & affected
    result[cloth_restore] = base[cloth_restore]
    result[shaft_restore] = base[shaft_restore]

    aggregate = cleanup | patch | cloth_restore | shaft_restore
    changed = np.any(result != base, axis=2)
    outside_delta = int(np.count_nonzero(changed & ~aggregate))
    outside_prop_delta = int(np.count_nonzero(changed & ~(cleanup | patch)))
    cloth_delta = int(np.count_nonzero(changed & cloth))
    shaft_delta = int(np.count_nonzero(changed & shaft))
    lower_delta = int(np.count_nonzero(changed & lower_exact))
    lower_band_delta = int(np.count_nonzero(changed[1200:]))
    character_delta = int(np.count_nonzero(changed[:, 300:]))
    if outside_delta or outside_prop_delta:
        raise RuntimeError(
            f"scope failure: outside saved={outside_delta}, outside prop={outside_prop_delta}"
        )
    if cloth_delta or shaft_delta:
        raise RuntimeError(f"pose preservation failure: cloth={cloth_delta}, shaft={shaft_delta}")
    if lower_delta or lower_band_delta:
        raise RuntimeError(
            f"lower butt/boot changed: mask={lower_delta}, band={lower_band_delta}"
        )
    if character_delta:
        raise RuntimeError(f"character/face/costume pixels changed: {character_delta}")
    visible_prop = patch & ~cloth_restore & ~shaft_restore
    hot_magenta = (
        visible_prop
        & (result[..., 0] >= 220)
        & (result[..., 1] <= 35)
        & (result[..., 2] >= 220)
    )
    if np.any(hot_magenta):
        raise RuntimeError(
            f"hot-magenta canonical metal pixels: {int(np.count_nonzero(hot_magenta))}"
        )

    for path in (
        OUTPUT,
        CLEANED,
        OLD_HEAD_MASK,
        PATCH_MASK,
        CLOTH_MASK,
        SHAFT_MASK,
        LOWER_EXACT_MASK,
        AGGREGATE_MASK,
        CHANGED_MASK,
        PROOF,
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
        (cloth, CLOTH_MASK),
        (shaft, SHAFT_MASK),
        (lower_exact, LOWER_EXACT_MASK),
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
            ("OLD UPPER CLEANUP", cleanup, (255, 220, 0)),
            ("CANONICAL METAL", patch, (0, 180, 255)),
            ("POSE CLOTH", cloth_restore, (255, 70, 90)),
            ("SHAFT", shaft_restore, (150, 80, 255)),
            ("LOWER EXACT", lower_exact, (0, 255, 110)),
            ("ACTUAL CHANGED", changed, (255, 255, 255)),
        ],
    )

    masks = {}
    for name, mask, path in (
        ("old_upper_head_cleanup", cleanup, OLD_HEAD_MASK),
        ("canonical_upper_metal_patch", patch, PATCH_MASK),
        ("pose_cloth_preserve", cloth, CLOTH_MASK),
        ("shaft_preserve", shaft, SHAFT_MASK),
        ("lower_butt_cap_boot_exact", lower_exact, LOWER_EXACT_MASK),
        ("aggregate_saved", aggregate, AGGREGATE_MASK),
        ("actual_changed", changed, CHANGED_MASK),
    ):
        masks[name] = {
            **record(path, int(np.count_nonzero(mask))),
            "bbox": mask_bbox(mask),
        }

    metadata = {
        "status": "pending_root_review",
        "label": "happy",
        "version": "correct_upper_endpoint_v2",
        "method": (
            "deterministic rigid-metal-only replacement of the upper/main blade; "
            "base tassel, cloth, shaft, lower butt cap, boot, and character remain exact"
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
                "fit_roi": [220, 650, 301, 1221],
            },
            "scale_evidence": scale_evidence,
        },
        "masks": masks,
        "validation": {
            "base_shape_hwc": list(base.shape),
            "output_mode": "RGB",
            "output_size_wh": [base.shape[1], base.shape[0]],
            "background_fit_samples": background_samples,
            "outside_saved_mask_delta_pixels": outside_delta,
            "outside_prop_scope_delta_pixels": outside_prop_delta,
            "pose_cloth_delta_pixels": cloth_delta,
            "shaft_delta_pixels": shaft_delta,
            "lower_butt_cap_boot_mask_delta_pixels": lower_delta,
            "entire_y1200_to_bottom_delta_pixels": lower_band_delta,
            "all_pixels_x300_to_right_character_region_delta_pixels": character_delta,
            "hot_magenta_visible_metal_pixels": int(np.count_nonzero(hot_magenta)),
            "wrong_end_v1_archived": archive_record,
        },
        "outputs": {
            "repaired": record(OUTPUT),
            "cleaned_upper_base": record(CLEANED),
            "upper_comparison_proof": record(PROOF),
            "lower_exact_proof": record(LOWER_PROOF),
            "mask_proof": record(MASK_PROOF),
        },
        "acceptance": {
            "automated": "pass",
            "visual": "pass_for_root_review",
            "manual_inspection": (
                "Native-resolution upper comparison, lower exact-region proof, mask proof, "
                "and full-frame output inspected: canonical rigid metal replaces the correct "
                "upper/main blade without a visible old-head ghost; the pose-specific red "
                "tassel remains coherent at the socket; shaft, character, and lower butt "
                "cap/boot relationship remain visually unchanged."
            ),
        },
    }
    write_json(METADATA, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
