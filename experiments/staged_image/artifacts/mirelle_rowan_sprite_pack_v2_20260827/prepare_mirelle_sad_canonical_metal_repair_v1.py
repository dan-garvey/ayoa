#!/usr/bin/env python3
"""Build Mirelle's sad rigid-metal spear repair from the approved split.

The sad pose keeps its generated, gravity-aware red tassel and its opposite
lower butt cap.  Only the upper/main silver spearhead is normalized.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
CORE_SCRIPT = ROOT / "prepare_mirelle_neutral_canonical_metal_repair_v1.py"
BASE = (
    ROOT.parent
    / "mirelle_rowan_sprite_pack_20260826/generation_raw/mirelle_voss/sad_chroma_v1.png"
)
OUTPUT = ROOT / "grafts/mirelle_voss/sad_canonical_metal_repair_v1.png"
OLD_HEAD_MASK = ROOT / "masks/mirelle_voss/sad_old_primary_head_cleanup_mask_v1.png"
PATCH_MASK = ROOT / "masks/mirelle_voss/sad_canonical_metal_component_patch_mask_v1.png"
TRANSFORMED_ALPHA = ROOT / "masks/mirelle_voss/sad_canonical_metal_transformed_alpha_v1.png"
TASSEL_MASK = ROOT / "masks/mirelle_voss/sad_existing_pose_tassel_occlusion_mask_v1.png"
BODY_MASK = ROOT / "masks/mirelle_voss/sad_hair_body_occlusion_mask_v1.png"
SAVED_MASK = ROOT / "masks/mirelle_voss/sad_canonical_metal_repair_saved_mask_v1.png"
CHANGED_MASK = ROOT / "masks/mirelle_voss/sad_canonical_metal_repair_changed_mask_v1.png"
COMPARISON_PROOF = ROOT / "component_proofs/mirelle_sad_canonical_metal_repair_close_v1.png"
MASK_PROOF = ROOT / "mask_proofs/mirelle_sad_canonical_metal_repair_masks_v1.png"
METADATA = ROOT / "component_metadata/mirelle_sad_canonical_metal_repair_v1.json"
TARGET_SOCKET_Y = 142.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_core():
    spec = importlib.util.spec_from_file_location("mirelle_neutral_core", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_shaft_centerline(base: np.ndarray) -> tuple[float, float, dict[str, float | int]]:
    not_hot_magenta = ~(
        (base[..., 0] >= 220)
        & (base[..., 1] <= 35)
        & (base[..., 2] >= 220)
    )
    centers: list[tuple[int, float]] = []
    widths: list[int] = []
    for y in [*range(340, 431), *range(590, 701), *range(1120, 1260)]:
        xs = np.flatnonzero(not_hot_magenta[y, 350:435]) + 350
        runs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1) if xs.size else []
        expected_x = 381.0 + 0.020 * y
        candidates = [
            run
            for run in runs
            if 10 <= run.size <= 30 and abs(float(np.mean(run)) - expected_x) <= 10.0
        ]
        if not candidates:
            continue
        run = min(candidates, key=lambda values: abs(float(np.mean(values)) - expected_x))
        centers.append((y, float(np.mean(run))))
        widths.append(int(run.size))

    if len(centers) < 250:
        raise RuntimeError(f"insufficient sad shaft samples: {len(centers)}")
    sample_y = np.asarray([sample[0] for sample in centers], dtype=np.float64)
    sample_x = np.asarray([sample[1] for sample in centers], dtype=np.float64)
    slope, intercept = np.polyfit(sample_y, sample_x, 1)
    residual = sample_x - (slope * sample_y + intercept)
    if not (0.010 <= slope <= 0.030):
        raise RuntimeError(f"implausible sad shaft slope: {slope}")
    if float(np.max(np.abs(residual))) > 2.5:
        raise RuntimeError("sad shaft fit has an outlier larger than 2.5 pixels")
    return float(slope), float(intercept), {
        "sample_count": len(centers),
        "row_ranges": [[340, 430], [590, 700], [1120, 1259]],
        "median_run_width_pixels": float(np.median(widths)),
        "maximum_absolute_residual_pixels": float(np.max(np.abs(residual))),
    }


def fit_local_background(base: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = base.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    fit_region = (xx >= 300) & (xx <= 470) & (yy <= 360)
    key_like = (
        (base[..., 0] >= 180)
        & (base[..., 2] >= 180)
        & (base[..., 1] <= 90)
        & fit_region
    )
    sample_y, sample_x = np.where(key_like)
    if sample_x.size < 35_000:
        raise RuntimeError("insufficient sad chroma samples")
    nx = (sample_x.astype(np.float64) - 385.0) / 100.0
    ny = (sample_y.astype(np.float64) - 180.0) / 180.0
    design = np.column_stack([np.ones_like(nx), nx, ny, nx * ny, nx * nx, ny * ny])
    coefficients = [
        np.linalg.lstsq(design, base[sample_y, sample_x, channel], rcond=None)[0]
        for channel in range(3)
    ]
    all_x = (xx.astype(np.float64) - 385.0) / 100.0
    all_y = (yy.astype(np.float64) - 180.0) / 180.0
    terms = [np.ones_like(all_x), all_x, all_y, all_x * all_y, all_x * all_x, all_y * all_y]
    prediction = np.stack(
        [sum(coefficient[i] * terms[i] for i in range(6)) for coefficient in coefficients],
        axis=2,
    )
    return prediction, int(sample_x.size)


def build_old_head_masks(
    base: np.ndarray, background: np.ndarray, core
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    authored = core.polygon_mask(
        (base.shape[1], base.shape[0]),
        [[(349, 8), (390, 8), (414, 86), (404, 148), (367, 148), (350, 84)]],
    )
    residual = np.linalg.norm(base.astype(np.float64) - background, axis=2)
    edge_candidate = authored & (residual > core.EDGE_BACKGROUND_RESIDUAL)
    cleanup = core.component_containing(edge_candidate, (381, 62))
    near = cv2.dilate(
        cleanup.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ) > 0
    cleanup |= edge_candidate & near
    core_candidate = authored & (residual > core.CORE_BACKGROUND_RESIDUAL)
    footprint_core = core.component_containing(core_candidate, (381, 62))
    return cleanup, footprint_core, authored


def derive_transform(
    metal: np.ndarray,
    footprint_core: np.ndarray,
    slope: float,
    intercept: float,
    core,
) -> tuple[np.ndarray, float, float, float, float]:
    target_socket = np.array(
        [slope * TARGET_SOCKET_Y + intercept, TARGET_SOCKET_Y], dtype=np.float64
    )
    source_inward = core.SOURCE_SHAFT_POINT - core.SOURCE_SOCKET
    source_inward /= np.linalg.norm(source_inward)
    target_inward = np.array([slope, 1.0], dtype=np.float64)
    target_inward /= np.linalg.norm(target_inward)
    source_outward = -source_inward
    target_outward = -target_inward
    source_y, source_x = np.where(metal[..., 3] >= core.MIN_COMPONENT_ALPHA)
    source_points = np.column_stack([source_x, source_y]).astype(np.float64)
    source_extent = float(np.max((source_points - core.SOURCE_SOCKET[None, :]) @ source_outward))
    target_y, target_x = np.where(footprint_core)
    target_points = np.column_stack([target_x, target_y]).astype(np.float64)
    target_extent = float(np.max((target_points - target_socket[None, :]) @ target_outward))
    scale = target_extent / source_extent
    if not (0.40 <= scale <= 0.50):
        raise RuntimeError(f"unexpected sad head scale: {scale}")
    rotation = float(
        np.degrees(
            np.arctan2(target_inward[1], target_inward[0])
            - np.arctan2(source_inward[1], source_inward[0])
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
    matrix[:, 2] = target_socket - matrix[:, :2] @ core.SOURCE_SOCKET
    return matrix, rotation, scale, source_extent, target_extent


def build_pose_preservation_masks(
    base: np.ndarray,
    background: np.ndarray,
    affected: np.ndarray,
    core,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.linalg.norm(base.astype(np.float64) - background, axis=2)
    cloth_region = core.polygon_mask(
        (base.shape[1], base.shape[0]),
        [[(338, 135), (418, 135), (427, 328), (335, 330)]],
    )
    red = base[..., 0].astype(np.int16)
    green = base[..., 1].astype(np.int16)
    blue = base[..., 2].astype(np.int16)
    candidates = (
        cloth_region
        & (red >= 48)
        & ((red - green) >= 18)
        & ((red - blue) >= 28)
        & (residual > 8.0)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), 8
    )
    if count <= 1:
        raise RuntimeError("sad pose tassel seed is empty")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    seed = labels == label
    near = cv2.dilate(
        seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ) > 0
    pose_tassel = cloth_region & near & (residual > core.EDGE_BACKGROUND_RESIDUAL)
    body_region = core.polygon_mask(
        (base.shape[1], base.shape[0]),
        [[(410, 90), (650, 90), (660, 420), (410, 420)]],
    )
    hand_body = body_region & affected & (residual > 12.0) & ~pose_tassel
    return pose_tassel, hand_body


def contained_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (12, 15, 20))
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    panel.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return panel


def make_proofs(core, canonical_crop: np.ndarray, base: np.ndarray, result: np.ndarray,
                masks: list[tuple[str, np.ndarray, tuple[int, int, int]]]) -> None:
    entries = [
        ("LOCKED CANONICAL SOURCE", Image.fromarray(canonical_crop).crop((0, 35, 410, 470))),
        ("SAD BASE", Image.fromarray(base).crop((300, 0, 470, 360))),
        ("METAL-ONLY REPAIR V1", Image.fromarray(result).crop((300, 0, 470, 360))),
    ]
    sheet = Image.new("RGB", (1320, 758), (12, 15, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(entries):
        sheet.paste(contained_panel(image, (440, 710)), (index * 440, 0))
        draw.text((index * 440 + 12, 725), title, fill=(255, 255, 255))
    COMPARISON_PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(COMPARISON_PROOF, optimize=True)
    mask_sheet = Image.new("RGB", (310 * len(masks), 772), (12, 15, 20))
    draw = ImageDraw.Draw(mask_sheet)
    for index, (title, mask, color) in enumerate(masks):
        shown = result.copy()
        shown[mask] = np.round(shown[mask] * 0.30 + np.asarray(color) * 0.70).astype(np.uint8)
        panel = contained_panel(Image.fromarray(shown).crop((300, 0, 470, 360)), (310, 724))
        mask_sheet.paste(panel, (index * 310, 0))
        draw.text((index * 310 + 8, 739), title, fill=(255, 255, 255))
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    mask_sheet.save(MASK_PROOF, optimize=True)


def main() -> None:
    core = load_core()
    core.__file__ = str(Path(__file__).resolve())
    core.BASE = BASE
    core.OUTPUT = OUTPUT
    core.OLD_HEAD_MASK = OLD_HEAD_MASK
    core.PATCH_MASK = PATCH_MASK
    core.TRANSFORMED_ALPHA = TRANSFORMED_ALPHA
    core.TASSEL_MASK = TASSEL_MASK
    core.BODY_MASK = BODY_MASK
    core.SAVED_MASK = SAVED_MASK
    core.CHANGED_MASK = CHANGED_MASK
    core.COMPARISON_PROOF = COMPARISON_PROOF
    core.MASK_PROOF = MASK_PROOF
    core.METADATA = METADATA
    core.TARGET_SOCKET_Y = TARGET_SOCKET_Y
    core.EXPECTED_HASHES = {
        BASE: "7c87432d582a188055384e2fa582cbd44bb2991a3e909c45d453e27ac07dbd6b",
        core.LOCKED_ACTIVE_PROFILE: "6173f6066860c270b73bcbde7fc86336411758301cdc5c12341f0400a82266b6",
        core.LOCKED_CROP: "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff",
        core.FULL_COMPONENT: "1285db94cdfd24d346123563227b8627129bc50a26a888bce85f82099defed4e",
        core.METAL_COMPONENT: "4f4b369d7f445b79a44673b33aee8bcc763f754eb37a0d7bfaee38213bf73b6f",
        core.STREAMER_COMPONENT: "9d7ff6a32a4ad46ac491652b40f0fa5f0a2c8fb4f88cd67c1ebdb1a81e6970dd",
        core.MATERIAL_PROOF: "d22b5e0bc779c4ec1718df490167fb7c8b7ae1f766e0e38089a0df2049ad1961",
    }
    core.fit_shaft_centerline = fit_shaft_centerline
    core.fit_local_background = fit_local_background
    core.build_old_head_masks = lambda base, background: build_old_head_masks(base, background, core)
    core.derive_transform = lambda metal, footprint, slope, intercept: derive_transform(
        metal, footprint, slope, intercept, core
    )
    core.build_pose_preservation_masks = lambda base, background, affected: (
        build_pose_preservation_masks(base, background, affected, core)
    )
    core.make_proofs = lambda canonical, base, result, masks: make_proofs(
        core, canonical, base, result, masks
    )
    core.main()

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    metadata["label"] = "sad"
    metadata["method"] = (
        "deterministic upper/main rigid-metal canonical graft; exact base sad-pose "
        "tassel and opposite lower butt cap retained; zero model calls"
    )
    metadata["endpoint_contract"] = {
        "replaced": "upper/main blade adjacent to the existing sad-pose tassel",
        "preserved": "lower/opposite pointed butt cap",
    }
    metadata["hash_fenced_inputs"]["script"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__).resolve()),
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
