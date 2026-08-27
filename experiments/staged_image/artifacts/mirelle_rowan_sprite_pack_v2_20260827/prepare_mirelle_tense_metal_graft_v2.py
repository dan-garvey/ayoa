#!/usr/bin/env python3
"""Metal-only canonical spear repair for Mirelle's tense v3 pose."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parent
V1_SCRIPT = ROOT / "prepare_mirelle_tense_reference_graft_v1.py"
BASE = ROOT / "generation_raw/mirelle_voss/tense_chroma_v3.png"
METAL = ROOT / "components/mirelle_voss/canonical_primary_metal_rgba_v1.png"
OUTPUT = ROOT / "grafts/mirelle_voss/tense_primary_metal_reference_graft_v2.png"
ERASE_MASK = ROOT / "masks/mirelle_voss/tense_old_metal_head_cleanup_mask_v2.png"
PATCH_MASK = ROOT / "masks/mirelle_voss/tense_canonical_metal_patch_mask_v2.png"
CLOTH_MASK = ROOT / "masks/mirelle_voss/tense_pose_cloth_preserve_mask_v2.png"
SHAFT_MASK = ROOT / "masks/mirelle_voss/tense_shaft_preserve_mask_v2.png"
BODY_MASK = ROOT / "masks/mirelle_voss/tense_hand_hair_preserve_mask_v2.png"
BACKGROUND_MASK = ROOT / "masks/mirelle_voss/tense_detached_fragment_background_restore_mask_v2.png"
CHANGED_MASK = ROOT / "masks/mirelle_voss/tense_metal_graft_changed_mask_v2.png"
PROOF = ROOT / "component_proofs/mirelle_tense_metal_graft_v2.png"
MASK_PROOF = ROOT / "mask_proofs/mirelle_tense_metal_graft_masks_v2.png"
METADATA = ROOT / "component_metadata/mirelle_tense_metal_graft_v2.json"

SOURCE_SOCKET = np.array([200.0, 89.0], dtype=np.float64)
SOURCE_SHAFT_POINT = np.array([289.0, 13.0], dtype=np.float64)
TARGET_SOCKET = np.array([383.0, 298.0], dtype=np.float64)
TARGET_SHAFT_POINT = np.array([401.0, 395.0], dtype=np.float64)
UNIFORM_SCALE = 0.78
MIN_ALPHA = 48


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v1():
    spec = importlib.util.spec_from_file_location("mirelle_tense_v1", V1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {V1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        path, optimize=True
    )


def transform_metal(
    component: np.ndarray, target_size: tuple[int, int]
) -> tuple[np.ndarray, float, np.ndarray]:
    source_vector = SOURCE_SHAFT_POINT - SOURCE_SOCKET
    target_vector = TARGET_SHAFT_POINT - TARGET_SOCKET
    rotation = float(
        np.degrees(
            np.arctan2(target_vector[1], target_vector[0])
            - np.arctan2(source_vector[1], source_vector[0])
        )
    )
    radians = np.deg2rad(rotation)
    matrix = np.array(
        [
            [np.cos(radians) * UNIFORM_SCALE, -np.sin(radians) * UNIFORM_SCALE, 0.0],
            [np.sin(radians) * UNIFORM_SCALE, np.cos(radians) * UNIFORM_SCALE, 0.0],
        ],
        dtype=np.float64,
    )
    matrix[:, 2] = TARGET_SOCKET - matrix[:, :2] @ SOURCE_SOCKET
    alpha = component[..., 3].astype(np.float32) / 255.0
    premultiplied = component[..., :3].astype(np.float32) * alpha[..., None]
    width, height = target_size
    warped_alpha = cv2.warpAffine(
        alpha, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_pm = cv2.warpAffine(
        premultiplied, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0)
    warped_rgb = np.zeros_like(warped_pm)
    nonzero = warped_alpha > 1e-4
    warped_rgb[nonzero] = warped_pm[nonzero] / warped_alpha[nonzero, None]
    return np.dstack(
        [
            np.clip(np.round(warped_rgb), 0, 255).astype(np.uint8),
            np.clip(np.round(warped_alpha * 255), 0, 255).astype(np.uint8),
        ]
    ), rotation, matrix


def base_material_masks(base: np.ndarray, v1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manual_head = v1.polygon_mask(
        (base.shape[1], base.shape[0]),
        [[(329, 8), (367, 8), (403, 68), (425, 180), (421, 315),
          (397, 330), (360, 323), (320, 298), (312, 170)]],
    )
    key_distance = np.linalg.norm(
        base.astype(np.int16) - np.array([255, 0, 255], dtype=np.int16), axis=2
    )
    # The manual region contains only the old metal head plus chroma screen;
    # keep every non-red foreground pixel so its black ink contour cannot
    # survive as a ghost silhouette after the silver interior is removed.
    old_metal = manual_head

    cloth_region = v1.polygon_mask(
        (base.shape[1], base.shape[0]),
        [[(330, 275), (416, 275), (431, 430), (330, 443)]],
    )
    red = base[..., 0].astype(np.int16)
    green = base[..., 1].astype(np.int16)
    blue = base[..., 2].astype(np.int16)
    cloth_candidates = (
        cloth_region
        & (red >= 48)
        & ((red - green) >= 18)
        & ((red - blue) >= 30)
        & (key_distance > 24.0)
    )
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(cloth_candidates.astype(np.uint8), 8)
    )
    if component_count <= 1:
        raise RuntimeError("pose-specific red cloth seed is empty")
    largest_component = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
    cloth_seed = component_labels == largest_component
    cloth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cloth_neighborhood = cv2.dilate(cloth_seed.astype(np.uint8), cloth_kernel) > 0
    dark_edge = np.max(base, axis=2) <= 92
    cloth = cloth_seed | (cloth_neighborhood & dark_edge)
    cloth &= cloth_region & (key_distance > 24.0)
    return old_metal, cloth, key_distance > 24.0


def line_mask(
    size: tuple[int, int], points: list[tuple[int, int]], width: int
) -> np.ndarray:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).line(points, fill=255, width=width)
    return np.asarray(image, dtype=np.uint8) > 0


def make_proofs(
    base: np.ndarray,
    output: np.ndarray,
    masks: list[tuple[str, np.ndarray, tuple[int, int, int]]],
) -> None:
    crop = (255, 0, 560, 540)
    comparison = Image.new("RGB", (1120, 1040), (12, 15, 20))
    for index, (title, array) in enumerate((("TENSE V3 BASE", base), ("METAL-ONLY V2", output))):
        image = Image.fromarray(array, mode="RGB").crop(crop)
        image = image.resize((560, 992), Image.Resampling.LANCZOS)
        comparison.paste(image, (index * 560, 0))
        ImageDraw.Draw(comparison).text((index * 560 + 12, 1003), title, fill=(255, 255, 255))
    PROOF.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(PROOF, optimize=True)

    sheet = Image.new("RGB", (len(masks) * 320, 612), (12, 15, 20))
    for index, (title, mask, color) in enumerate(masks):
        shown = output.copy()
        shown[mask] = np.round(
            shown[mask].astype(np.float32) * 0.3
            + np.asarray(color, dtype=np.float32) * 0.7
        ).astype(np.uint8)
        panel = Image.fromarray(shown, mode="RGB").crop(crop)
        panel = panel.resize((320, 566), Image.Resampling.LANCZOS)
        sheet.paste(panel, (index * 320, 0))
        ImageDraw.Draw(sheet).text((index * 320 + 8, 576), title, fill=(255, 255, 255))
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(MASK_PROOF, optimize=True)


def main() -> None:
    v1 = load_v1()
    base = np.asarray(Image.open(BASE).convert("RGB"), dtype=np.uint8)
    metal = np.asarray(Image.open(METAL).convert("RGBA"), dtype=np.uint8)
    transformed, rotation, matrix = transform_metal(metal, (base.shape[1], base.shape[0]))
    patch = transformed[..., 3] >= MIN_ALPHA
    old_metal, cloth, subject = base_material_masks(base, v1)

    result = cv2.inpaint(
        base,
        np.where(old_metal, 255, 0).astype(np.uint8),
        7.0,
        cv2.INPAINT_TELEA,
    )
    result[patch] = transformed[..., :3][patch]
    affected = old_metal | patch
    shaft = line_mask(
        (base.shape[1], base.shape[0]), [(384, 294), (406, 408)], 15
    ) & affected & subject
    body_manual = v1.polygon_mask(
        (base.shape[1], base.shape[0]),
        [
            [(435, 75), (550, 75), (555, 342), (430, 342)],
            [(366, 365), (445, 365), (460, 476), (378, 492), (352, 423)],
        ],
    )
    body = body_manual & affected & subject
    preserved_cloth = cloth & affected
    result[preserved_cloth] = base[preserved_cloth]
    result[shaft] = base[shaft]
    result[body] = base[body]

    # Telea is useful for the large erased silhouette, but it pulled a few
    # gray/pink old-head boundary colors into the tight socket/guard gaps.
    # Replace only that local exposed background with exact clean chroma pixels
    # copied from the nearest qualifying pixel in the original base image.
    yy, xx = np.mgrid[: base.shape[0], : base.shape[1]]
    fragment_region = (
        (xx >= 325) & (xx <= 426) & (yy >= 245) & (yy <= 338)
    )
    background_restore = (
        fragment_region
        & old_metal
        & ~patch
        & ~preserved_cloth
        & ~shaft
        & ~body
    )
    red = base[..., 0].astype(np.int16)
    green = base[..., 1].astype(np.int16)
    blue = base[..., 2].astype(np.int16)
    clean_chroma = (
        (red >= 225)
        & (blue >= 210)
        & (green <= 22)
        & (np.abs(red - blue) <= 35)
    )
    if not np.any(clean_chroma):
        raise RuntimeError("no clean base chroma source pixels")
    _, nearest_chroma = distance_transform_edt(
        ~clean_chroma, return_indices=True
    )
    result[background_restore] = base[
        nearest_chroma[0][background_restore],
        nearest_chroma[1][background_restore],
    ]

    changed = np.any(result != base, axis=2)
    scope = old_metal | patch | preserved_cloth | shaft | body | background_restore
    outside_delta = int(np.count_nonzero(changed & ~scope))
    if outside_delta:
        raise RuntimeError(f"{outside_delta} pixels changed outside saved masks")
    exact_restore = preserved_cloth | shaft | body
    if not np.array_equal(result[exact_restore], base[exact_restore]):
        raise RuntimeError("pose cloth/shaft/body restore is not exact")
    hot_prop = (
        patch
        & ~exact_restore
        & (result[..., 0] >= 220)
        & (result[..., 1] <= 35)
        & (result[..., 2] >= 220)
    )
    if np.any(hot_prop):
        raise RuntimeError(f"{int(np.count_nonzero(hot_prop))} hot-magenta prop pixels")

    for path in (
        OUTPUT, ERASE_MASK, PATCH_MASK, CLOTH_MASK, SHAFT_MASK, BODY_MASK,
        BACKGROUND_MASK, CHANGED_MASK,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(OUTPUT, optimize=True)
    for mask, path in (
        (old_metal, ERASE_MASK), (patch, PATCH_MASK), (preserved_cloth, CLOTH_MASK),
        (shaft, SHAFT_MASK), (body, BODY_MASK),
        (background_restore, BACKGROUND_MASK), (changed, CHANGED_MASK),
    ):
        save_mask(mask, path)
    make_proofs(
        base,
        result,
        [
            ("OLD METAL", old_metal, (255, 220, 0)),
            ("CANONICAL METAL", patch, (0, 180, 255)),
            ("POSE CLOTH", preserved_cloth, (255, 80, 80)),
            ("SHAFT", shaft, (160, 80, 255)),
            ("HAND/HAIR", body, (0, 255, 110)),
            ("CLEAN BACKGROUND", background_restore, (255, 255, 255)),
            ("ALL CHANGED", changed, (255, 255, 255)),
        ],
    )

    ys, xs = np.where(patch)
    metadata = {
        "status": "pending_root_review",
        "label": "tense",
        "supersedes": {
            "path": str(ROOT / "grafts/mirelle_voss/tense_primary_spear_reference_graft_v1.png"),
            "reason": "root rejected rigidly rotated streamers and oversized head",
        },
        "method": "metal-only canonical graft with exact base pose cloth",
        "inputs": {
            "base": {"path": str(BASE), "sha256": sha256(BASE)},
            "metal_component": {"path": str(METAL), "sha256": sha256(METAL)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "transform": {
            "source_socket": SOURCE_SOCKET.tolist(),
            "source_shaft_point": SOURCE_SHAFT_POINT.tolist(),
            "target_socket": TARGET_SOCKET.tolist(),
            "target_shaft_point": TARGET_SHAFT_POINT.tolist(),
            "rotation_degrees": rotation,
            "uniform_scale": UNIFORM_SCALE,
            "matrix": matrix.tolist(),
            "nonuniform_deformation": False,
            "patch_bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        },
        "bounded_changes": {
            "old_metal_cleanup_pixels": int(np.count_nonzero(old_metal)),
            "background_reconstruction": {
                "method": (
                    "OpenCV Telea inpaint over full old-head polygon, then "
                    "nearest exact base chroma copy in the bounded socket/guard gap"
                ),
                "radius": 7.0,
                "clean_chroma_contract": {
                    "red_min": 225,
                    "blue_min": 210,
                    "green_max": 22,
                    "absolute_red_blue_delta_max": 35,
                },
                "bounded_fragment_region_xyxy_inclusive": [325, 245, 426, 338],
                "exact_base_chroma_copy_pixels": int(np.count_nonzero(background_restore)),
            },
            "canonical_metal_patch_pixels": int(np.count_nonzero(patch)),
            "exact_pose_cloth_pixels": int(np.count_nonzero(preserved_cloth)),
            "exact_shaft_pixels": int(np.count_nonzero(shaft)),
            "exact_hand_hair_pixels": int(np.count_nonzero(body)),
            "actual_changed_pixels": int(np.count_nonzero(changed)),
            "outside_saved_masks_delta_pixels": outside_delta,
            "hot_magenta_prop_pixels": 0,
            "masks": {
                "old_metal": {"path": str(ERASE_MASK), "sha256": sha256(ERASE_MASK)},
                "metal_patch": {"path": str(PATCH_MASK), "sha256": sha256(PATCH_MASK)},
                "pose_cloth": {"path": str(CLOTH_MASK), "sha256": sha256(CLOTH_MASK)},
                "shaft": {"path": str(SHAFT_MASK), "sha256": sha256(SHAFT_MASK)},
                "body": {"path": str(BODY_MASK), "sha256": sha256(BODY_MASK)},
                "background_restore": {
                    "path": str(BACKGROUND_MASK),
                    "sha256": sha256(BACKGROUND_MASK),
                },
                "changed": {"path": str(CHANGED_MASK), "sha256": sha256(CHANGED_MASK)},
            },
        },
        "output": {"path": str(OUTPUT), "sha256": sha256(OUTPUT), "dimensions": [base.shape[1], base.shape[0]]},
        "proofs": {
            "comparison": {"path": str(PROOF), "sha256": sha256(PROOF)},
            "masks": {"path": str(MASK_PROOF), "sha256": sha256(MASK_PROOF)},
        },
    }
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
