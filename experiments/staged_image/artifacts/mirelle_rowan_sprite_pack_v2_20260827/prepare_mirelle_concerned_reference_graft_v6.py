#!/usr/bin/env python3
"""Build the deterministic Mirelle concerned spear-graft v6 proof.

This is deliberately a bounded correction of the reviewed v5 graft.  It does
not invoke a model, redraw Mirelle, or alter the upper spear assembly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
V5_SCRIPT = ROOT / "prepare_mirelle_concerned_reference_graft.py"
V5_GRAFT = ROOT / "grafts/mirelle_voss/concerned_lower_spear_reference_graft_v5.png"
TARGET = ROOT / "generation_raw/mirelle_voss/concerned_chroma_v2.png"

COMPONENT = ROOT / "components/mirelle_voss/canonical_lower_spearhead_streamers_rgba_v6.png"
SOURCE_BOUNDARY_MASK = (
    ROOT / "masks/mirelle_voss/canonical_lower_spearhead_boundary_decontam_mask_v6.png"
)
SOURCE_HOLE_FILL_MASK = (
    ROOT / "masks/mirelle_voss/canonical_lower_spearhead_small_hole_fill_mask_v6.png"
)
PATCH_MASK = ROOT / "masks/mirelle_voss/concerned_reference_graft_component_patch_mask_v6.png"
TRANSFORMED_MAGENTA_CLEANUP_MASK = (
    ROOT / "masks/mirelle_voss/concerned_component_boundary_magenta_cleanup_mask_v6.png"
)
SLIVER_MASK = ROOT / "masks/mirelle_voss/concerned_old_head_sliver_cleanup_mask_v6.png"
OCCLUSION_MASK = (
    ROOT / "masks/mirelle_voss/concerned_rear_streamer_body_occlusion_mask_v6.png"
)
CHANGED_MASK = ROOT / "masks/mirelle_voss/concerned_reference_graft_changed_mask_v6.png"
GRAFT = ROOT / "grafts/mirelle_voss/concerned_lower_spear_reference_graft_v6.png"
FULL_PROOF = ROOT / "component_proofs/mirelle_concerned_reference_graft_v6.png"
MASK_PROOF = ROOT / "mask_proofs/mirelle_concerned_reference_graft_masks_v6.png"
METADATA = ROOT / "component_metadata/mirelle_concerned_reference_graft_v6.json"

MIN_COMPONENT_ALPHA = 48
MAX_FILLED_HOLE_AREA = 64
SOURCE_BOUNDARY_WIDTH = 3.0
BOUNDARY_COVERAGE_DISTANCE_LOW = 12.0
BOUNDARY_COVERAGE_DISTANCE_HIGH = 72.0
SLIVER_DILATION_RADIUS = 7
STREAMER_DILATION_RADIUS = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v5_module():
    spec = importlib.util.spec_from_file_location("mirelle_graft_v5", V5_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {V5_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def prepare_clean_component(v5):
    full_rgba, _component, bbox, key = v5.extract_reference()
    rgb = full_rgba[..., :3].copy()
    alpha = full_rgba[..., 3].copy()
    original_alpha = alpha.copy()
    foreground = alpha > 0

    # The v5 near-key rejection left dozens of tiny transparent pinholes in
    # silver bevels.  Fill only enclosed components <=64 px.  The three large
    # canonical negative spaces (177, 223, and 1578 px) remain transparent.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (~foreground).astype(np.uint8), connectivity=8
    )
    hole_fill = np.zeros_like(foreground)
    height, width = foreground.shape
    filled_areas: list[int] = []
    for index in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[index]]
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if not touches_border and area <= MAX_FILLED_HOLE_AREA:
            hole_fill[labels == index] = True
            filled_areas.append(area)
    alpha[hole_fill] = 255

    foreground = alpha > 0
    inside_distance = cv2.distanceTransform(
        foreground.astype(np.uint8), cv2.DIST_L2, 5
    )
    boundary = foreground & (inside_distance <= SOURCE_BOUNDARY_WIDTH)
    distance_from_key = np.linalg.norm(
        rgb.astype(np.float32) - np.asarray(key, dtype=np.float32), axis=2
    )
    decontam = (
        boundary
        & (alpha > 0)
        & (distance_from_key < BOUNDARY_COVERAGE_DISTANCE_HIGH)
        & ~hole_fill
    )

    # Unmix only partial-alpha boundary pixels against the sampled beige key.
    # Every opaque silver/red interior pixel remains byte-identical to the
    # locked crop, including the newly closed highlight pinholes.
    estimated_coverage = np.clip(
        (distance_from_key - BOUNDARY_COVERAGE_DISTANCE_LOW)
        / (BOUNDARY_COVERAGE_DISTANCE_HIGH - BOUNDARY_COVERAGE_DISTANCE_LOW),
        0.0,
        1.0,
    )
    estimated_coverage = estimated_coverage * estimated_coverage * (
        3.0 - 2.0 * estimated_coverage
    )
    coverage = np.minimum(
        alpha.astype(np.float32) / 255.0, estimated_coverage
    )
    coverage = np.maximum(coverage, 0.18)
    key_float = np.asarray(key, dtype=np.float32)
    source = rgb.astype(np.float32)
    unmixed = (
        source - (1.0 - coverage[..., None]) * key_float[None, None, :]
    ) / coverage[..., None]
    rgb[decontam] = np.clip(np.round(unmixed[decontam]), 0, 255).astype(np.uint8)

    cleaned = np.dstack([rgb, alpha])
    cleaned[alpha == 0, :3] = 0
    interior = (original_alpha == 255) & ~boundary & ~hole_fill
    if not np.array_equal(cleaned[..., :3][interior], full_rgba[..., :3][interior]):
        raise RuntimeError("opaque canonical component interiors changed")
    return cleaned, bbox, key, decontam, hole_fill, filled_areas


def body_protection_mask(size: tuple[int, int]) -> np.ndarray:
    # These polygons protect character pixels while the narrow old-head rim is
    # removed.  They do not establish final depth; that is handled separately.
    return polygon_mask(
        size,
        [
            [(438, 875), (546, 875), (560, 1115), (542, 1235), (455, 1235), (438, 1060)],
            [(552, 875), (655, 875), (661, 1192), (575, 1215), (552, 1060)],
            [(640, 760), (843, 760), (843, 1210), (680, 1220), (647, 1110)],
            [(368, 1160), (552, 1150), (566, 1338), (365, 1345)],
            [(560, 1160), (706, 1150), (718, 1390), (550, 1390)],
        ],
    )


def build_sliver_mask(v5, target: np.ndarray, v5_graft: np.ndarray) -> np.ndarray:
    old_erase = v5.old_prop_erase_mask((target.shape[1], target.shape[0])) > 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (SLIVER_DILATION_RADIUS * 2 + 1, SLIVER_DILATION_RADIUS * 2 + 1),
    )
    expanded = cv2.dilate(old_erase.astype(np.uint8), kernel) > 0
    unchanged_from_target = np.all(v5_graft == target, axis=2)
    key_distance = np.linalg.norm(
        target.astype(np.int16) - np.array([255, 0, 255], dtype=np.int16), axis=2
    )
    non_key = key_distance > 24.0
    protected_body = body_protection_mask((target.shape[1], target.shape[0]))
    return expanded & ~old_erase & unchanged_from_target & non_key & ~protected_body


def build_occlusion_mask(
    transformed: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rgb = transformed[..., :3]
    alpha = transformed[..., 3]
    red_fill = (
        (alpha >= MIN_COMPONENT_ALPHA)
        & (rgb[..., 0] >= 60)
        & (rgb[..., 0].astype(np.int16) >= rgb[..., 1].astype(np.int16) + 18)
        & (rgb[..., 0].astype(np.int16) >= rgb[..., 2].astype(np.int16) + 8)
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (STREAMER_DILATION_RADIUS * 2 + 1, STREAMER_DILATION_RADIUS * 2 + 1),
    )
    streamer_with_outline = cv2.dilate(red_fill.astype(np.uint8), kernel) > 0
    streamer_with_outline &= alpha >= MIN_COMPONENT_ALPHA

    # The clipped tails at the top of the reference crop are deliberately sent
    # behind the original greaves/cape.  The visible free tips on image-left and
    # the collar/primary blade remain foreground.  Central x=590..636 is omitted
    # so the original shaft and canonical socket are never occluded.
    depth_regions = polygon_mask(
        (target.shape[1], target.shape[0]),
        [
            [(486, 994), (578, 994), (584, 1096), (565, 1143), (505, 1150), (485, 1080)],
            [(515, 1138), (537, 1138), (537, 1162), (518, 1164)],
            [(614, 995), (646, 995), (646, 1058), (616, 1068)],
            [(638, 995), (704, 995), (711, 1165), (650, 1172), (638, 1100)],
        ],
    )
    key_distance = np.linalg.norm(
        target.astype(np.int16) - np.array([255, 0, 255], dtype=np.int16), axis=2
    )
    target_hot_magenta = (
        (target[..., 0] >= 220)
        & (target[..., 1] <= 35)
        & (target[..., 2] >= 220)
    )
    original_subject = (key_distance > 24.0) & ~target_hot_magenta
    occlusion = streamer_with_outline & depth_regions & original_subject
    return occlusion, streamer_with_outline


def clean_transformed_boundary_magenta(transformed: np.ndarray) -> np.ndarray:
    """Replace rare transform-edge magenta pixels with nearest prop RGB.

    These pixels arise only where Lanczos resamples a physically unmixed source
    boundary.  The search is bounded to 12 target pixels and never considers a
    transparent pixel or another hot-magenta pixel.
    """
    rgb = transformed[..., :3]
    alpha = transformed[..., 3]
    hot = (
        (alpha >= MIN_COMPONENT_ALPHA)
        & (rgb[..., 0] >= 220)
        & (rgb[..., 1] <= 35)
        & (rgb[..., 2] >= 220)
    )
    valid = (alpha >= MIN_COMPONENT_ALPHA) & ~hot
    for y, x in np.argwhere(hot):
        best: tuple[int, int, int] | None = None
        for radius in range(1, 13):
            y0 = max(0, int(y) - radius)
            y1 = min(alpha.shape[0], int(y) + radius + 1)
            x0 = max(0, int(x) - radius)
            x1 = min(alpha.shape[1], int(x) + radius + 1)
            candidates = np.argwhere(valid[y0:y1, x0:x1])
            if not candidates.size:
                continue
            candidates[:, 0] += y0
            candidates[:, 1] += x0
            squared = (candidates[:, 0] - y) ** 2 + (candidates[:, 1] - x) ** 2
            index = int(np.argmin(squared))
            candidate_y, candidate_x = [int(value) for value in candidates[index]]
            best = (int(squared[index]), candidate_y, candidate_x)
            break
        if best is None:
            raise RuntimeError(f"no clean component neighbor for ({x}, {y})")
        _distance, candidate_y, candidate_x = best
        rgb[y, x] = rgb[candidate_y, candidate_x]
    return hot


def make_proofs(
    original: np.ndarray,
    v5_graft: np.ndarray,
    v6_graft: np.ndarray,
    patch: np.ndarray,
    sliver: np.ndarray,
    occlusion: np.ndarray,
    changed: np.ndarray,
) -> None:
    crop = (430, 900, 750, 1455)
    labels = (("ORIGINAL", original), ("V5 INPUT", v5_graft), ("V6 DETERMINISTIC", v6_graft))
    cells: list[Image.Image] = []
    for title, array in labels:
        image = Image.fromarray(array, mode="RGB").crop(crop)
        image = image.resize((480, 832), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (480, 884), (12, 15, 20))
        cell.paste(image, (0, 0))
        ImageDraw.Draw(cell).text((12, 842), title, fill=(255, 255, 255))
        cells.append(cell)
    sheet = Image.new("RGB", (1440, 884), (12, 15, 20))
    for index, cell in enumerate(cells):
        sheet.paste(cell, (index * 480, 0))
    FULL_PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(FULL_PROOF, optimize=True)

    base = Image.fromarray(v6_graft, mode="RGB")
    overlays = []
    for title, mask, color in (
        ("COMPONENT PATCH", patch, (0, 180, 255)),
        ("OLD-SLIVER CLEANUP", sliver, (255, 220, 0)),
        ("BODY OCCLUSION", occlusion, (0, 255, 110)),
        ("ALL CHANGED", changed, (255, 255, 255)),
    ):
        overlay = np.asarray(base).copy()
        overlay[mask] = np.round(
            overlay[mask].astype(np.float32) * 0.30
            + np.asarray(color, dtype=np.float32) * 0.70
        ).astype(np.uint8)
        panel = Image.fromarray(overlay, mode="RGB").crop(crop)
        panel = panel.resize((400, 694), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (400, 742), (12, 15, 20))
        canvas.paste(panel, (0, 0))
        ImageDraw.Draw(canvas).text((10, 706), title, fill=(255, 255, 255))
        overlays.append(canvas)
    mask_sheet = Image.new("RGB", (1600, 742), (12, 15, 20))
    for index, panel in enumerate(overlays):
        mask_sheet.paste(panel, (index * 400, 0))
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    mask_sheet.save(MASK_PROOF, optimize=True)


def main() -> None:
    v5 = load_v5_module()
    cleaned_full, bbox, key, source_boundary, hole_fill, filled_areas = (
        prepare_clean_component(v5)
    )
    target = np.asarray(Image.open(TARGET).convert("RGB"), dtype=np.uint8)
    v5_graft = np.asarray(Image.open(V5_GRAFT).convert("RGB"), dtype=np.uint8)
    if target.shape != v5_graft.shape:
        raise RuntimeError("target/v5 dimensions differ")

    transformed = v5.transform_component(
        cleaned_full, (target.shape[1], target.shape[0])
    )
    transformed_magenta_cleanup = clean_transformed_boundary_magenta(transformed)
    patch = transformed[..., 3] >= MIN_COMPONENT_ALPHA
    sliver = build_sliver_mask(v5, target, v5_graft)
    occlusion, streamer = build_occlusion_mask(transformed, target)

    result = v5_graft.copy()
    result[sliver] = np.array([255, 0, 255], dtype=np.uint8)
    # Hard replacement with clean unpremultiplied component RGB prevents the
    # magenta chroma screen from entering prop edge RGB.  The source alpha is
    # still preserved as a separate proof mask for later production matting.
    result[patch] = transformed[..., :3][patch]
    result[occlusion] = target[occlusion]

    changed = np.any(result != v5_graft, axis=2)
    scope = patch | sliver | occlusion
    outside_scope_delta = int(np.count_nonzero(changed & ~scope))
    if outside_scope_delta:
        raise RuntimeError(f"{outside_scope_delta} pixels changed outside saved masks")
    if not np.array_equal(result[occlusion], target[occlusion]):
        raise RuntimeError("occluded body pixels are not exact target pixels")
    magenta_prop = (
        patch
        & (result[..., 0] >= 220)
        & (result[..., 1] <= 35)
        & (result[..., 2] >= 220)
    )
    if np.any(magenta_prop):
        raise RuntimeError(
            f"component patch retains {int(np.count_nonzero(magenta_prop))} hot-magenta pixels"
        )

    for path in (
        COMPONENT,
        SOURCE_BOUNDARY_MASK,
        SOURCE_HOLE_FILL_MASK,
        PATCH_MASK,
        TRANSFORMED_MAGENTA_CLEANUP_MASK,
        SLIVER_MASK,
        OCCLUSION_MASK,
        CHANGED_MASK,
        GRAFT,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    ys, xs = np.where(cleaned_full[..., 3] > 0)
    component = cleaned_full[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()
    Image.fromarray(component, mode="RGBA").save(COMPONENT, optimize=True)
    save_mask(source_boundary, SOURCE_BOUNDARY_MASK)
    save_mask(hole_fill, SOURCE_HOLE_FILL_MASK)
    save_mask(patch, PATCH_MASK)
    save_mask(transformed_magenta_cleanup, TRANSFORMED_MAGENTA_CLEANUP_MASK)
    save_mask(sliver, SLIVER_MASK)
    save_mask(occlusion, OCCLUSION_MASK)
    save_mask(changed, CHANGED_MASK)
    Image.fromarray(result, mode="RGB").save(GRAFT, optimize=True)
    make_proofs(target, v5_graft, result, patch, sliver, occlusion, changed)

    metadata = {
        "status": "pending_root_review",
        "method": "deterministic v5 boundary cleanup plus exact-original body occlusion; no model call",
        "inputs": {
            "v5_graft": {"path": str(V5_GRAFT), "sha256": sha256(V5_GRAFT)},
            "target_concerned_v2": {"path": str(TARGET), "sha256": sha256(TARGET)},
            "v5_script": {"path": str(V5_SCRIPT), "sha256": sha256(V5_SCRIPT)},
            "v6_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "component_cleanup": {
            "source_bbox": bbox,
            "background_key_rgb": [float(value) for value in key],
            "small_hole_max_area": MAX_FILLED_HOLE_AREA,
            "filled_hole_component_areas": sorted(filled_areas),
            "filled_hole_pixels": int(np.count_nonzero(hole_fill)),
            "boundary_width_source_pixels": SOURCE_BOUNDARY_WIDTH,
            "boundary_coverage_distance_low": BOUNDARY_COVERAGE_DISTANCE_LOW,
            "boundary_coverage_distance_high": BOUNDARY_COVERAGE_DISTANCE_HIGH,
            "boundary_coverage_floor": 0.18,
            "boundary_unmix_formula": "(observed - (1 - coverage) * beige_key) / coverage",
            "boundary_decontam_pixels": int(np.count_nonzero(source_boundary)),
            "opaque_interior_rgb_byte_identical_to_reference": True,
            "component_path": str(COMPONENT),
            "component_sha256": sha256(COMPONENT),
            "boundary_mask": {
                "path": str(SOURCE_BOUNDARY_MASK),
                "sha256": sha256(SOURCE_BOUNDARY_MASK),
            },
            "hole_fill_mask": {
                "path": str(SOURCE_HOLE_FILL_MASK),
                "sha256": sha256(SOURCE_HOLE_FILL_MASK),
            },
        },
        "bounded_changes": {
            "component_patch_pixels": int(np.count_nonzero(patch)),
            "transformed_boundary_hot_magenta_cleanup_pixels": int(
                np.count_nonzero(transformed_magenta_cleanup)
            ),
            "old_head_sliver_cleanup_pixels": int(np.count_nonzero(sliver)),
            "rear_streamer_occlusion_pixels": int(np.count_nonzero(occlusion)),
            "rear_streamer_candidate_pixels": int(np.count_nonzero(streamer)),
            "changed_pixels": int(np.count_nonzero(changed)),
            "outside_saved_masks_delta_pixels": outside_scope_delta,
            "occlusion_pixels_exact_from_target": True,
            "hot_magenta_pixels_inside_component_patch": 0,
            "component_patch_mask": {"path": str(PATCH_MASK), "sha256": sha256(PATCH_MASK)},
            "transformed_boundary_magenta_cleanup_mask": {
                "path": str(TRANSFORMED_MAGENTA_CLEANUP_MASK),
                "sha256": sha256(TRANSFORMED_MAGENTA_CLEANUP_MASK),
            },
            "old_sliver_mask": {"path": str(SLIVER_MASK), "sha256": sha256(SLIVER_MASK)},
            "body_occlusion_mask": {"path": str(OCCLUSION_MASK), "sha256": sha256(OCCLUSION_MASK)},
            "changed_mask": {"path": str(CHANGED_MASK), "sha256": sha256(CHANGED_MASK)},
        },
        "output": {
            "path": str(GRAFT),
            "sha256": sha256(GRAFT),
            "dimensions": [target.shape[1], target.shape[0]],
        },
        "proofs": {
            "full": {"path": str(FULL_PROOF), "sha256": sha256(FULL_PROOF)},
            "masks": {"path": str(MASK_PROOF), "sha256": sha256(MASK_PROOF)},
        },
    }
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
