#!/usr/bin/env python3
"""Deterministic canonical primary-spear graft for Mirelle's tense v3 base."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "generation_raw/mirelle_voss/tense_chroma_v3.png"
COMPONENT = ROOT / "components/mirelle_voss/canonical_lower_spearhead_streamers_rgba_v6.png"
OUTPUT = ROOT / "grafts/mirelle_voss/tense_primary_spear_reference_graft_v1.png"
ERASE_MASK = ROOT / "masks/mirelle_voss/tense_old_primary_head_cleanup_mask_v1.png"
PATCH_MASK = ROOT / "masks/mirelle_voss/tense_canonical_component_patch_mask_v1.png"
SHAFT_MASK = ROOT / "masks/mirelle_voss/tense_original_shaft_preserve_mask_v1.png"
OCCLUSION_MASK = ROOT / "masks/mirelle_voss/tense_hand_hair_occlusion_mask_v1.png"
CHANGED_MASK = ROOT / "masks/mirelle_voss/tense_reference_graft_changed_mask_v1.png"
PROOF = ROOT / "component_proofs/mirelle_tense_reference_graft_v1.png"
MASK_PROOF = ROOT / "mask_proofs/mirelle_tense_reference_graft_masks_v1.png"
METADATA = ROOT / "component_metadata/mirelle_tense_reference_graft_v1.json"

# The cleaned component crop's socket and away-from-primary-head shaft point.
SOURCE_SOCKET = np.array([200.0, 89.0], dtype=np.float64)
SOURCE_SHAFT_POINT = np.array([289.0, 13.0], dtype=np.float64)
# Tense v3 holds the primary head upward.  The shaft continues down-right from
# the socket; the opposite pointed cap is occluded against the lower boot.
TARGET_SOCKET = np.array([383.0, 298.0], dtype=np.float64)
TARGET_SHAFT_POINT = np.array([401.0, 395.0], dtype=np.float64)
UNIFORM_SCALE = 0.90
MIN_ALPHA = 48


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        path, optimize=True
    )


def polygon_mask(size: tuple[int, int], polygons: list[list[tuple[int, int]]]) -> np.ndarray:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def transform_component(
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
        alpha,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_premultiplied = cv2.warpAffine(
        premultiplied,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0)
    warped_rgb = np.zeros_like(warped_premultiplied)
    nonzero = warped_alpha > 1e-4
    warped_rgb[nonzero] = warped_premultiplied[nonzero] / warped_alpha[nonzero, None]
    transformed = np.dstack(
        [
            np.clip(np.round(warped_rgb), 0, 255).astype(np.uint8),
            np.clip(np.round(warped_alpha * 255), 0, 255).astype(np.uint8),
        ]
    )
    return transformed, rotation, matrix


def old_head_cleanup(size: tuple[int, int]) -> np.ndarray:
    return polygon_mask(
        size,
        [
            [(333, 8), (365, 8), (401, 70), (425, 174), (421, 303),
             (404, 326), (368, 326), (326, 301), (316, 180)],
            [(357, 286), (399, 286), (405, 338), (393, 397),
             (379, 435), (343, 435), (338, 375)],
        ],
    )


def shaft_preserve(size: tuple[int, int], affected: np.ndarray, subject: np.ndarray) -> np.ndarray:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    draw.line([(386, 302), (404, 390)], fill=255, width=17)
    return (np.asarray(image, dtype=np.uint8) > 0) & affected & subject


def character_occlusion(
    size: tuple[int, int], affected: np.ndarray, subject: np.ndarray
) -> np.ndarray:
    manual = polygon_mask(
        size,
        [
            # Hair/face sit behind the head spatially but in front of any long
            # streamer that reaches image-right.
            [(435, 70), (545, 70), (553, 335), (430, 339)],
            # Gripping hand and forearm must occlude the ribbons/shaft.
            [(369, 365), (442, 365), (459, 471), (382, 488), (355, 425)],
        ],
    )
    return manual & affected & subject


def reconstruct_key_background(base: np.ndarray, erase: np.ndarray) -> tuple[np.ndarray, int]:
    """Fit the base's smooth pink chroma gradient around the removed prop."""
    height, width = erase.shape
    yy, xx = np.mgrid[0:height, 0:width]
    fit_region = (xx >= 265) & (xx <= 445) & (yy <= 455)
    key_like = (
        (base[..., 0] >= 170)
        & (base[..., 2] >= 170)
        & (base[..., 1] <= 95)
        & ~erase
        & fit_region
    )
    sample_y, sample_x = np.where(key_like)
    if sample_x.size < 1000:
        raise RuntimeError("insufficient surrounding chroma pixels for background fit")
    # Normalize coordinates before the quadratic fit for numerical stability.
    nx = (sample_x.astype(np.float64) - 355.0) / 100.0
    ny = (sample_y.astype(np.float64) - 225.0) / 225.0
    design = np.column_stack(
        [np.ones_like(nx), nx, ny, nx * ny, nx * nx, ny * ny]
    )
    coefficients = [
        np.linalg.lstsq(design, base[sample_y, sample_x, channel], rcond=None)[0]
        for channel in range(3)
    ]
    erase_y, erase_x = np.where(erase)
    ex = (erase_x.astype(np.float64) - 355.0) / 100.0
    ey = (erase_y.astype(np.float64) - 225.0) / 225.0
    erase_design = np.column_stack(
        [np.ones_like(ex), ex, ey, ex * ey, ex * ex, ey * ey]
    )
    reconstructed = base.copy()
    for channel, coefficient in enumerate(coefficients):
        values = np.clip(np.round(erase_design @ coefficient), 0, 255).astype(np.uint8)
        reconstructed[erase_y, erase_x, channel] = values
    return reconstructed, int(sample_x.size)


def build_proofs(
    base: np.ndarray,
    output: np.ndarray,
    erase: np.ndarray,
    patch: np.ndarray,
    shaft: np.ndarray,
    occlusion: np.ndarray,
    changed: np.ndarray,
) -> None:
    crop = (255, 0, 560, 540)
    cells: list[Image.Image] = []
    for title, array in (("TENSE V3 BASE", base), ("DETERMINISTIC REPAIR", output)):
        image = Image.fromarray(array, mode="RGB").crop(crop)
        image = image.resize((560, 992), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (560, 1040), (12, 15, 20))
        cell.paste(image, (0, 0))
        ImageDraw.Draw(cell).text((12, 1003), title, fill=(255, 255, 255))
        cells.append(cell)
    sheet = Image.new("RGB", (1120, 1040), (12, 15, 20))
    sheet.paste(cells[0], (0, 0))
    sheet.paste(cells[1], (560, 0))
    PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(PROOF, optimize=True)

    panels: list[Image.Image] = []
    for title, mask, color in (
        ("OLD HEAD CLEANUP", erase, (255, 220, 0)),
        ("COMPONENT PATCH", patch, (0, 180, 255)),
        ("SHAFT PRESERVE", shaft, (255, 80, 80)),
        ("HAND/HAIR OCCLUSION", occlusion, (0, 255, 110)),
        ("ALL CHANGED", changed, (255, 255, 255)),
    ):
        shown = output.copy()
        shown[mask] = np.round(
            shown[mask].astype(np.float32) * 0.30
            + np.asarray(color, dtype=np.float32) * 0.70
        ).astype(np.uint8)
        panel = Image.fromarray(shown, mode="RGB").crop(crop)
        panel = panel.resize((360, 638), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (360, 684), (12, 15, 20))
        canvas.paste(panel, (0, 0))
        ImageDraw.Draw(canvas).text((9, 649), title, fill=(255, 255, 255))
        panels.append(canvas)
    mask_sheet = Image.new("RGB", (1800, 684), (12, 15, 20))
    for index, panel in enumerate(panels):
        mask_sheet.paste(panel, (index * 360, 0))
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    mask_sheet.save(MASK_PROOF, optimize=True)


def main() -> None:
    base = np.asarray(Image.open(BASE).convert("RGB"), dtype=np.uint8)
    component = np.asarray(Image.open(COMPONENT).convert("RGBA"), dtype=np.uint8)
    transformed, rotation, matrix = transform_component(
        component, (base.shape[1], base.shape[0])
    )
    patch = transformed[..., 3] >= MIN_ALPHA
    hot_magenta = (
        patch
        & (transformed[..., 0] >= 220)
        & (transformed[..., 1] <= 35)
        & (transformed[..., 2] >= 220)
    )
    if np.any(hot_magenta):
        raise RuntimeError(
            f"cleaned canonical component has {int(np.count_nonzero(hot_magenta))} hot-magenta pixels"
        )

    erase = old_head_cleanup((base.shape[1], base.shape[0]))
    result, background_fit_samples = reconstruct_key_background(base, erase)
    result[patch] = transformed[..., :3][patch]

    key_distance = np.linalg.norm(
        base.astype(np.int16) - np.array([255, 0, 255], dtype=np.int16), axis=2
    )
    base_hot_magenta = (
        (base[..., 0] >= 220) & (base[..., 1] <= 35) & (base[..., 2] >= 220)
    )
    subject = (key_distance > 24.0) & ~base_hot_magenta
    affected = erase | patch
    shaft = shaft_preserve((base.shape[1], base.shape[0]), affected, subject)
    occlusion = character_occlusion(
        (base.shape[1], base.shape[0]), affected, subject
    )
    result[shaft] = base[shaft]
    result[occlusion] = base[occlusion]

    changed = np.any(result != base, axis=2)
    scope = erase | patch | shaft | occlusion
    outside_delta = int(np.count_nonzero(changed & ~scope))
    if outside_delta:
        raise RuntimeError(f"{outside_delta} pixels changed outside saved masks")
    if not np.array_equal(result[shaft | occlusion], base[shaft | occlusion]):
        raise RuntimeError("preserved shaft/body pixels are not exact base pixels")
    output_hot_magenta = (
        patch
        & (result[..., 0] >= 220)
        & (result[..., 1] <= 35)
        & (result[..., 2] >= 220)
        & ~(shaft | occlusion)
    )
    if np.any(output_hot_magenta):
        raise RuntimeError(
            f"output has {int(np.count_nonzero(output_hot_magenta))} hot-magenta prop pixels"
        )

    for path in (OUTPUT, ERASE_MASK, PATCH_MASK, SHAFT_MASK, OCCLUSION_MASK, CHANGED_MASK):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(OUTPUT, optimize=True)
    save_mask(erase, ERASE_MASK)
    save_mask(patch, PATCH_MASK)
    save_mask(shaft, SHAFT_MASK)
    save_mask(occlusion, OCCLUSION_MASK)
    save_mask(changed, CHANGED_MASK)
    build_proofs(base, result, erase, patch, shaft, occlusion, changed)

    ys, xs = np.where(patch)
    metadata = {
        "status": "pending_root_review",
        "label": "tense",
        "method": "uniform deterministic canonical-component graft; no model call",
        "inputs": {
            "base": {"path": str(BASE), "sha256": sha256(BASE)},
            "cleaned_component": {"path": str(COMPONENT), "sha256": sha256(COMPONENT)},
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
            "old_head_cleanup_pixels": int(np.count_nonzero(erase)),
            "old_head_background_reconstruction": {
                "method": "quadratic RGB fit over surrounding base key pixels",
                "sample_pixels": background_fit_samples,
                "fit_bbox": [265, 0, 446, 456],
            },
            "component_patch_pixels": int(np.count_nonzero(patch)),
            "exact_original_shaft_pixels": int(np.count_nonzero(shaft)),
            "exact_original_hand_hair_pixels": int(np.count_nonzero(occlusion)),
            "actual_changed_pixels": int(np.count_nonzero(changed)),
            "outside_saved_masks_delta_pixels": outside_delta,
            "hot_magenta_prop_pixels": 0,
            "masks": {
                "old_head_cleanup": {"path": str(ERASE_MASK), "sha256": sha256(ERASE_MASK)},
                "component_patch": {"path": str(PATCH_MASK), "sha256": sha256(PATCH_MASK)},
                "shaft_preserve": {"path": str(SHAFT_MASK), "sha256": sha256(SHAFT_MASK)},
                "body_occlusion": {"path": str(OCCLUSION_MASK), "sha256": sha256(OCCLUSION_MASK)},
                "changed": {"path": str(CHANGED_MASK), "sha256": sha256(CHANGED_MASK)},
            },
        },
        "output": {
            "path": str(OUTPUT),
            "sha256": sha256(OUTPUT),
            "dimensions": [base.shape[1], base.shape[0]],
        },
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
