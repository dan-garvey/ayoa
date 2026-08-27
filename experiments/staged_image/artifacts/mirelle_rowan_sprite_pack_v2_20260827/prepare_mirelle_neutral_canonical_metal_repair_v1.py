#!/usr/bin/env python3
"""Build the bounded deterministic Mirelle neutral metal-only spear proof.

This proof intentionally uses only the frozen rigid silver component.  The
canonical red streamers are hash-fenced provenance but are never transformed
or composited: neutral keeps its generated, gravity-aware tassel exactly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
BASE = (
    ROOT.parent
    / "mirelle_rowan_sprite_pack_20260826/generation_raw/mirelle_voss/neutral_chroma_v1.png"
)
LOCKED_ACTIVE_PROFILE = (
    REPO_ROOT
    / "app/storage/stories/one_star_ascension_s1/visual-references/locked/"
    "mirelle_voss/active_profile.png"
)
LOCKED_CROP = (
    ROOT
    / "supporting_refs/"
    "mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
FULL_COMPONENT = (
    ROOT / "components/mirelle_voss/canonical_lower_spearhead_streamers_rgba_v6.png"
)
METAL_COMPONENT = ROOT / "components/mirelle_voss/canonical_primary_metal_rgba_v1.png"
STREAMER_COMPONENT = (
    ROOT / "components/mirelle_voss/canonical_primary_streamers_rgba_v1.png"
)
MATERIAL_PROOF = ROOT / "component_proofs/mirelle_canonical_primary_material_split_v1.png"

OUTPUT = ROOT / "grafts/mirelle_voss/neutral_canonical_metal_repair_v1.png"
OLD_HEAD_MASK = (
    ROOT / "masks/mirelle_voss/neutral_old_primary_head_cleanup_mask_v1.png"
)
PATCH_MASK = (
    ROOT / "masks/mirelle_voss/neutral_canonical_metal_component_patch_mask_v1.png"
)
TRANSFORMED_ALPHA = (
    ROOT / "masks/mirelle_voss/neutral_canonical_metal_transformed_alpha_v1.png"
)
TASSEL_MASK = (
    ROOT / "masks/mirelle_voss/neutral_existing_pose_tassel_occlusion_mask_v1.png"
)
BODY_MASK = (
    ROOT / "masks/mirelle_voss/neutral_hand_body_garment_occlusion_mask_v1.png"
)
SAVED_MASK = (
    ROOT / "masks/mirelle_voss/neutral_canonical_metal_repair_saved_mask_v1.png"
)
CHANGED_MASK = (
    ROOT / "masks/mirelle_voss/neutral_canonical_metal_repair_changed_mask_v1.png"
)
COMPARISON_PROOF = (
    ROOT / "component_proofs/mirelle_neutral_canonical_metal_repair_close_v1.png"
)
MASK_PROOF = (
    ROOT / "mask_proofs/mirelle_neutral_canonical_metal_repair_masks_v1.png"
)
METADATA = (
    ROOT / "component_metadata/mirelle_neutral_canonical_metal_repair_v1.json"
)

EXPECTED_HASHES = {
    BASE: "7a52d35686265f1e3976fb41cbae25dd04ce29e02051835918c67e0f4dbeebb9",
    LOCKED_ACTIVE_PROFILE: "6173f6066860c270b73bcbde7fc86336411758301cdc5c12341f0400a82266b6",
    LOCKED_CROP: "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff",
    FULL_COMPONENT: "1285db94cdfd24d346123563227b8627129bc50a26a888bce85f82099defed4e",
    METAL_COMPONENT: "4f4b369d7f445b79a44673b33aee8bcc763f754eb37a0d7bfaee38213bf73b6f",
    STREAMER_COMPONENT: "9d7ff6a32a4ad46ac491652b40f0fa5f0a2c8fb4f88cd67c1ebdb1a81e6970dd",
    MATERIAL_PROOF: "d22b5e0bc779c4ec1718df490167fb7c8b7ae1f766e0e38089a0df2049ad1961",
}

# Component coordinates are inherited from the locked active-profile crop:
# crop anchor (248, 174) and crop shaft point (337, 98), minus the extracted
# component origin (48, 85).
SOURCE_SOCKET = np.array([200.0, 89.0], dtype=np.float64)
SOURCE_SHAFT_POINT = np.array([289.0, 13.0], dtype=np.float64)
TARGET_SOCKET_Y = 225.0
MIN_COMPONENT_ALPHA = 48
CORE_BACKGROUND_RESIDUAL = 24.0
EDGE_BACKGROUND_RESIDUAL = 3.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> None:
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"hash fence failed for {path}: expected {expected}, got {actual}"
            )


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


def fit_shaft_centerline(base: np.ndarray) -> tuple[float, float, dict[str, float | int]]:
    """Fit x=m*y+b to the uninterrupted neutral shaft pixels.

    Rows 390..484 and 610..1199 avoid both the tassel and the gripping hand.
    The selected run is the non-hot-magenta run nearest the expected shaft
    corridor; no endpoints are manually supplied to the angle calculation.
    """

    not_hot_magenta = ~(
        (base[..., 0] >= 220)
        & (base[..., 1] <= 35)
        & (base[..., 2] >= 220)
    )
    centers: list[tuple[int, float]] = []
    widths: list[int] = []
    for y in [*range(390, 485), *range(610, 1200)]:
        xs = np.flatnonzero(not_hot_magenta[y, 285:355]) + 285
        runs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1) if xs.size else []
        expected_x = 303.0 + 0.025 * y
        candidates = [
            run
            for run in runs
            if 10 <= run.size <= 30 and abs(float(np.mean(run)) - expected_x) <= 8.0
        ]
        if not candidates:
            raise RuntimeError(f"no shaft run at y={y}")
        run = min(candidates, key=lambda values: abs(float(np.mean(values)) - expected_x))
        centers.append((y, float(np.mean(run))))
        widths.append(int(run.size))

    sample_y = np.asarray([sample[0] for sample in centers], dtype=np.float64)
    sample_x = np.asarray([sample[1] for sample in centers], dtype=np.float64)
    slope, intercept = np.polyfit(sample_y, sample_x, 1)
    residual = sample_x - (slope * sample_y + intercept)
    if not (0.020 <= slope <= 0.030):
        raise RuntimeError(f"implausible neutral shaft slope: {slope}")
    if float(np.max(np.abs(residual))) > 2.0:
        raise RuntimeError("shaft fit has an outlier larger than two pixels")
    return float(slope), float(intercept), {
        "sample_count": len(centers),
        "row_range_1": [390, 484],
        "row_range_2": [610, 1199],
        "median_run_width_pixels": float(np.median(widths)),
        "maximum_absolute_residual_pixels": float(np.max(np.abs(residual))),
    }


def fit_local_background(base: np.ndarray) -> tuple[np.ndarray, int]:
    """Fit the smooth local magenta plate without changing untouched pixels."""

    height, width = base.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    fit_region = (xx >= 230) & (xx <= 400) & (yy <= 430)
    key_like = (
        (base[..., 0] >= 180)
        & (base[..., 2] >= 180)
        & (base[..., 1] <= 90)
        & fit_region
    )
    sample_y, sample_x = np.where(key_like)
    if sample_x.size < 50_000:
        raise RuntimeError("insufficient neutral chroma samples")

    nx = (sample_x.astype(np.float64) - 315.0) / 100.0
    ny = (sample_y.astype(np.float64) - 215.0) / 215.0
    design = np.column_stack(
        [np.ones_like(nx), nx, ny, nx * ny, nx * nx, ny * ny]
    )
    coefficients = [
        np.linalg.lstsq(design, base[sample_y, sample_x, channel], rcond=None)[0]
        for channel in range(3)
    ]

    all_x = (xx.astype(np.float64) - 315.0) / 100.0
    all_y = (yy.astype(np.float64) - 215.0) / 215.0
    terms = [
        np.ones_like(all_x),
        all_x,
        all_y,
        all_x * all_y,
        all_x * all_x,
        all_y * all_y,
    ]
    prediction = np.stack(
        [sum(coefficient[i] * terms[i] for i in range(6)) for coefficient in coefficients],
        axis=2,
    )
    return prediction, int(sample_x.size)


def component_containing(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    seed_x, seed_y = seed
    label = int(labels[seed_y, seed_x])
    if count <= 1 or label == 0:
        raise RuntimeError(f"seed {seed} is not inside the expected old head")
    return labels == label


def build_old_head_masks(
    base: np.ndarray, background: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bounded cleanup, footprint core, and the authored head region."""

    authored = polygon_mask(
        (base.shape[1], base.shape[0]),
        [
            [
                (299, 18),
                (310, 18),
                (340, 174),
                (328, 203),
                (323, 235),
                (293, 235),
                (289, 205),
                (276, 178),
            ]
        ],
    )
    residual = np.linalg.norm(base.astype(np.float64) - background, axis=2)
    edge_candidate = authored & (residual > EDGE_BACKGROUND_RESIDUAL)
    cleanup = component_containing(edge_candidate, (304, 100))

    # Admit only tiny disconnected antialias fragments immediately adjacent to
    # the connected silhouette.  Unchanged surrounding chroma is never erased.
    near = cv2.dilate(
        cleanup.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ) > 0
    cleanup |= edge_candidate & near

    core_candidate = authored & (residual > CORE_BACKGROUND_RESIDUAL)
    footprint_core = component_containing(core_candidate, (304, 100))
    return cleanup, footprint_core, authored


def derive_transform(
    metal: np.ndarray,
    footprint_core: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, float, float, float, float]:
    target_socket = np.array(
        [slope * TARGET_SOCKET_Y + intercept, TARGET_SOCKET_Y], dtype=np.float64
    )
    source_inward = SOURCE_SHAFT_POINT - SOURCE_SOCKET
    source_inward /= np.linalg.norm(source_inward)
    target_inward = np.array([slope, 1.0], dtype=np.float64)
    target_inward /= np.linalg.norm(target_inward)
    source_outward = -source_inward
    target_outward = -target_inward

    source_y, source_x = np.where(metal[..., 3] >= MIN_COMPONENT_ALPHA)
    source_points = np.column_stack([source_x, source_y]).astype(np.float64)
    source_extent = float(
        np.max((source_points - SOURCE_SOCKET[None, :]) @ source_outward)
    )
    target_y, target_x = np.where(footprint_core)
    target_points = np.column_stack([target_x, target_y]).astype(np.float64)
    target_extent = float(
        np.max((target_points - target_socket[None, :]) @ target_outward)
    )
    uniform_scale = target_extent / source_extent
    if not (0.64 <= uniform_scale <= 0.70):
        raise RuntimeError(f"unexpected neutral head scale: {uniform_scale}")

    rotation = float(
        np.degrees(
            np.arctan2(target_inward[1], target_inward[0])
            - np.arctan2(source_inward[1], source_inward[0])
        )
    )
    radians = np.deg2rad(rotation)
    matrix = np.array(
        [
            [
                np.cos(radians) * uniform_scale,
                -np.sin(radians) * uniform_scale,
                0.0,
            ],
            [
                np.sin(radians) * uniform_scale,
                np.cos(radians) * uniform_scale,
                0.0,
            ],
        ],
        dtype=np.float64,
    )
    matrix[:, 2] = target_socket - matrix[:, :2] @ SOURCE_SOCKET
    return matrix, rotation, uniform_scale, source_extent, target_extent


def warp_component(
    component: np.ndarray, matrix: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    alpha = component[..., 3].astype(np.float32) / 255.0
    premultiplied = component[..., :3].astype(np.float32) * alpha[..., None]
    width, height = size
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
    warped_rgb[nonzero] = (
        warped_premultiplied[nonzero] / warped_alpha[nonzero, None]
    )
    return np.dstack(
        [
            np.clip(np.round(warped_rgb), 0, 255).astype(np.uint8),
            np.clip(np.round(warped_alpha * 255), 0, 255).astype(np.uint8),
        ]
    )


def build_pose_preservation_masks(
    base: np.ndarray,
    background: np.ndarray,
    affected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.linalg.norm(base.astype(np.float64) - background, axis=2)
    cloth_region = polygon_mask(
        (base.shape[1], base.shape[0]),
        [
            [
                (306, 239),
                (341, 238),
                (390, 279),
                (384, 386),
                (329, 390),
                (306, 320),
            ]
        ],
    )
    red_seed = (
        cloth_region
        & (base[..., 0] >= 48)
        & (base[..., 0].astype(np.int16) >= base[..., 1].astype(np.int16) + 18)
        & (base[..., 0].astype(np.int16) >= base[..., 2].astype(np.int16) + 8)
        & (base[..., 2] <= 150)
        & (residual > 12.0)
    )
    cloth_neighborhood = cv2.dilate(
        red_seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ) > 0
    pose_tassel = cloth_region & cloth_neighborhood & (residual > EDGE_BACKGROUND_RESIDUAL)

    # These authored regions cover every possible Mirelle hand/body/garment
    # overlap near this end of the spear.  Neutral's transformed top head is
    # spatially disjoint, so a zero mask is the expected and audited result.
    body_region = polygon_mask(
        (base.shape[1], base.shape[0]),
        [
            [(450, 100), (760, 100), (780, 610), (430, 610)],
            [(270, 470), (500, 470), (520, 620), (260, 620)],
        ],
    )
    base_subject = residual > 12.0
    hand_body_garment = body_region & affected & base_subject
    return pose_tassel, hand_body_garment


def contained_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (12, 15, 20))
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    panel.paste(
        contained,
        ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
    )
    return panel


def make_proofs(
    canonical_crop: np.ndarray,
    base: np.ndarray,
    result: np.ndarray,
    masks: list[tuple[str, np.ndarray, tuple[int, int, int]]],
) -> None:
    image_size = (440, 710)
    label_height = 48
    source = Image.fromarray(canonical_crop, mode="RGB").crop((0, 35, 410, 470))
    base_close = Image.fromarray(base, mode="RGB").crop((235, 0, 415, 420))
    repaired_close = Image.fromarray(result, mode="RGB").crop((235, 0, 415, 420))
    entries = [
        ("LOCKED CANONICAL SOURCE", source),
        ("NEUTRAL BASE", base_close),
        ("METAL-ONLY REPAIR V1", repaired_close),
    ]
    sheet = Image.new(
        "RGB", (image_size[0] * len(entries), image_size[1] + label_height), (12, 15, 20)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(entries):
        panel = contained_panel(image, image_size)
        sheet.paste(panel, (index * image_size[0], 0))
        draw.text(
            (index * image_size[0] + 12, image_size[1] + 15),
            title,
            fill=(255, 255, 255),
        )
    COMPARISON_PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(COMPARISON_PROOF, optimize=True)

    crop = (235, 0, 415, 420)
    panel_width = 310
    panel_height = 724
    mask_sheet = Image.new(
        "RGB", (panel_width * len(masks), panel_height + label_height), (12, 15, 20)
    )
    draw = ImageDraw.Draw(mask_sheet)
    for index, (title, mask, color) in enumerate(masks):
        shown = result.copy()
        shown[mask] = np.round(
            shown[mask].astype(np.float32) * 0.30
            + np.asarray(color, dtype=np.float32) * 0.70
        ).astype(np.uint8)
        panel = contained_panel(
            Image.fromarray(shown, mode="RGB").crop(crop),
            (panel_width, panel_height),
        )
        mask_sheet.paste(panel, (index * panel_width, 0))
        draw.text(
            (index * panel_width + 8, panel_height + 15),
            title,
            fill=(255, 255, 255),
        )
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    mask_sheet.save(MASK_PROOF, optimize=True)


def mask_record(path: Path, mask: np.ndarray) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "pixels": int(np.count_nonzero(mask)),
    }


def main() -> None:
    verify_inputs()
    base = np.asarray(Image.open(BASE).convert("RGB"), dtype=np.uint8)
    metal = np.asarray(Image.open(METAL_COMPONENT).convert("RGBA"), dtype=np.uint8)
    canonical_crop = np.asarray(Image.open(LOCKED_CROP).convert("RGB"), dtype=np.uint8)

    slope, intercept, shaft_fit = fit_shaft_centerline(base)
    background, background_fit_samples = fit_local_background(base)
    cleanup, footprint_core, _authored_head = build_old_head_masks(base, background)
    matrix, rotation, scale, source_extent, target_extent = derive_transform(
        metal, footprint_core, slope, intercept
    )
    transformed = warp_component(metal, matrix, (base.shape[1], base.shape[0]))
    patch = transformed[..., 3] >= MIN_COMPONENT_ALPHA

    transformed_hot = (
        patch
        & (transformed[..., 0] >= 220)
        & (transformed[..., 1] <= 35)
        & (transformed[..., 2] >= 220)
    )
    if np.any(transformed_hot):
        raise RuntimeError(
            f"rigid component contains {int(np.count_nonzero(transformed_hot))} hot-magenta pixels"
        )

    result = base.copy()
    cleanup_y, cleanup_x = np.where(cleanup)
    for channel in range(3):
        result[cleanup_y, cleanup_x, channel] = np.clip(
            np.round(background[cleanup_y, cleanup_x, channel]), 0, 255
        ).astype(np.uint8)
    result[patch] = transformed[..., :3][patch]

    affected = cleanup | patch
    pose_tassel, hand_body_garment = build_pose_preservation_masks(
        base, background, affected
    )
    result[pose_tassel] = base[pose_tassel]
    result[hand_body_garment] = base[hand_body_garment]

    changed = np.any(result != base, axis=2)
    saved = cleanup | patch | pose_tassel | hand_body_garment
    outside_saved_delta = int(np.count_nonzero(changed & ~saved))
    if outside_saved_delta:
        raise RuntimeError(f"{outside_saved_delta} pixels changed outside saved mask")
    if not np.array_equal(result[pose_tassel], base[pose_tassel]):
        raise RuntimeError("pose-specific tassel was not restored exactly")
    if not np.array_equal(result[hand_body_garment], base[hand_body_garment]):
        raise RuntimeError("hand/body/garment occlusion was not restored exactly")

    lower_cap_roi = (slice(1260, 1370), slice(305, 355))
    if not np.array_equal(result[lower_cap_roi], base[lower_cap_roi]):
        raise RuntimeError("opposite lower cap changed")

    visible_prop = patch & ~(pose_tassel | hand_body_garment)
    output_hot = (
        visible_prop
        & (result[..., 0] >= 220)
        & (result[..., 1] <= 35)
        & (result[..., 2] >= 220)
    )
    if np.any(output_hot):
        raise RuntimeError(
            f"output contains {int(np.count_nonzero(output_hot))} hot-magenta prop pixels"
        )

    patch_y, patch_x = np.where(patch)
    if patch_y.size == 0:
        raise RuntimeError("empty transformed metal patch")

    for path in (
        OUTPUT,
        OLD_HEAD_MASK,
        PATCH_MASK,
        TRANSFORMED_ALPHA,
        TASSEL_MASK,
        BODY_MASK,
        SAVED_MASK,
        CHANGED_MASK,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(OUTPUT, optimize=True)
    save_mask(cleanup, OLD_HEAD_MASK)
    save_mask(patch, PATCH_MASK)
    Image.fromarray(transformed[..., 3], mode="L").save(TRANSFORMED_ALPHA, optimize=True)
    save_mask(pose_tassel, TASSEL_MASK)
    save_mask(hand_body_garment, BODY_MASK)
    save_mask(saved, SAVED_MASK)
    save_mask(changed, CHANGED_MASK)
    make_proofs(
        canonical_crop,
        base,
        result,
        [
            ("OLD-HEAD CLEANUP", cleanup, (255, 220, 0)),
            ("RIGID METAL PATCH", patch, (0, 180, 255)),
            ("POSE TASSEL", pose_tassel, (255, 80, 80)),
            ("BODY OCCLUSION", hand_body_garment, (0, 255, 110)),
            ("SAVED SCOPE", saved, (160, 80, 255)),
            ("ACTUAL CHANGES", changed, (255, 255, 255)),
        ],
    )

    target_socket = [slope * TARGET_SOCKET_Y + intercept, TARGET_SOCKET_Y]
    metadata = {
        "status": "pending_root_review",
        "label": "neutral",
        "method": (
            "deterministic rigid-metal-only canonical graft; base pose tassel retained "
            "exactly; no generated output used as identity input"
        ),
        "model_calls": 0,
        "hash_fenced_inputs": {
            "base": {"path": str(BASE), "sha256": sha256(BASE)},
            "locked_active_profile": {
                "path": str(LOCKED_ACTIVE_PROFILE),
                "sha256": sha256(LOCKED_ACTIVE_PROFILE),
            },
            "locked_crop": {"path": str(LOCKED_CROP), "sha256": sha256(LOCKED_CROP)},
            "cleaned_combined_component": {
                "path": str(FULL_COMPONENT),
                "sha256": sha256(FULL_COMPONENT),
            },
            "rigid_metal_component_used": {
                "path": str(METAL_COMPONENT),
                "sha256": sha256(METAL_COMPONENT),
            },
            "flexible_streamer_component_not_used": {
                "path": str(STREAMER_COMPONENT),
                "sha256": sha256(STREAMER_COMPONENT),
                "applied_pixels": 0,
            },
            "material_split_proof": {
                "path": str(MATERIAL_PROOF),
                "sha256": sha256(MATERIAL_PROOF),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "shaft_centerline": {
            "equation": "x = slope * y + intercept",
            "slope": slope,
            "intercept": intercept,
            **shaft_fit,
        },
        "transform": {
            "source_socket_component_xy": SOURCE_SOCKET.tolist(),
            "source_shaft_point_component_xy": SOURCE_SHAFT_POINT.tolist(),
            "target_socket_xy": target_socket,
            "rotation_degrees": rotation,
            "uniform_scale": scale,
            "source_rigid_outward_extent_pixels": source_extent,
            "target_existing_head_outward_extent_pixels": target_extent,
            "matrix": matrix.tolist(),
            "nonuniform_deformation": False,
            "patch_bbox_xyxy": [
                int(patch_x.min()),
                int(patch_y.min()),
                int(patch_x.max()) + 1,
                int(patch_y.max()) + 1,
            ],
        },
        "bounded_changes": {
            "background_fit_samples": background_fit_samples,
            "old_head_cleanup_pixels": int(np.count_nonzero(cleanup)),
            "rigid_metal_patch_pixels": int(np.count_nonzero(patch)),
            "pose_tassel_preserve_pixels": int(np.count_nonzero(pose_tassel)),
            "pose_tassel_patch_overlap_pixels": int(np.count_nonzero(pose_tassel & patch)),
            "hand_body_garment_occlusion_pixels": int(
                np.count_nonzero(hand_body_garment)
            ),
            "aggregate_saved_pixels": int(np.count_nonzero(saved)),
            "actual_changed_pixels": int(np.count_nonzero(changed)),
            "outside_saved_mask_delta_pixels": outside_saved_delta,
            "opposite_lower_cap_delta_pixels": 0,
            "hot_magenta_visible_prop_pixels": 0,
            "masks": {
                "old_head_cleanup": mask_record(OLD_HEAD_MASK, cleanup),
                "rigid_metal_patch": mask_record(PATCH_MASK, patch),
                "transformed_alpha": {
                    "path": str(TRANSFORMED_ALPHA),
                    "sha256": sha256(TRANSFORMED_ALPHA),
                    "nonzero_pixels": int(np.count_nonzero(transformed[..., 3])),
                },
                "pose_tassel": mask_record(TASSEL_MASK, pose_tassel),
                "hand_body_garment": mask_record(BODY_MASK, hand_body_garment),
                "saved_scope": mask_record(SAVED_MASK, saved),
                "actual_changed": mask_record(CHANGED_MASK, changed),
            },
        },
        "output": {
            "path": str(OUTPUT),
            "sha256": sha256(OUTPUT),
            "dimensions": [base.shape[1], base.shape[0]],
            "mode": "RGB",
        },
        "proofs": {
            "original_base_repaired_close": {
                "path": str(COMPARISON_PROOF),
                "sha256": sha256(COMPARISON_PROOF),
            },
            "masks": {"path": str(MASK_PROOF), "sha256": sha256(MASK_PROOF)},
        },
        "acceptance_assertions": {
            "outside_saved_mask_delta_zero": True,
            "pose_tassel_exact_base_pixels": True,
            "hand_body_garment_exact_base_pixels": True,
            "opposite_lower_cap_exact_base_pixels": True,
            "hot_magenta_visible_prop_pixels_zero": True,
            "flexible_streamer_component_applied_pixels_zero": True,
        },
    }
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
