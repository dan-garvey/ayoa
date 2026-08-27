#!/usr/bin/env python3
"""Build the base-only, connected rigid-metal Mirelle concerned v8 proof."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
V7_SCRIPT = ROOT / "prepare_mirelle_concerned_lower_primary_metal_graft_v7.py"
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

V7_FILES = {
    "prepare_mirelle_concerned_lower_primary_metal_graft_v7.py": (
        V7_SCRIPT,
        "5356167b07d4f2d586871c6d2370593bcabede08c9470955ff2d2dc7f22e1908",
    ),
    "concerned_lower_primary_metal_reference_graft_v7.png": (
        ROOT / "grafts/mirelle_voss/concerned_lower_primary_metal_reference_graft_v7.png",
        "3b575a0389536d9f8674a04bcb6f5ef6de60c97ee5292a60c60252c9d5210d2c",
    ),
    "concerned_old_lower_primary_cleaned_base_v7.png": (
        ROOT / "grafts/mirelle_voss/concerned_old_lower_primary_cleaned_base_v7.png",
        "f0d754772604fd7d75d145df172e8496842f6f77f61f024f426e6153cade8ff7",
    ),
    "mirelle_concerned_lower_primary_metal_graft_v7.json": (
        ROOT / "component_metadata/mirelle_concerned_lower_primary_metal_graft_v7.json",
        "a373099e4025478b9b5399004cde4bb8b3cddebd3bc0170ade245dcb024808ac",
    ),
    "mirelle_concerned_lower_primary_full_v7.png": (
        ROOT / "component_proofs/mirelle_concerned_lower_primary_full_v7.png",
        "d3830d9d7c11ca01015b2308ae734f2307cf920eb8dbfd90d45cf38ae99b0058",
    ),
    "mirelle_concerned_lower_primary_close_v7.png": (
        ROOT / "component_proofs/mirelle_concerned_lower_primary_close_v7.png",
        "67640ddca785a8e6c572d3efcc44c7b7bcd5b88d0ee717022bfd8a2d9c020279",
    ),
    "mirelle_concerned_lower_primary_masks_v7.png": (
        ROOT / "mask_proofs/mirelle_concerned_lower_primary_masks_v7.png",
        "12442e9942c853fbb0ac74eea2dc636741c00255a82a6a1cbb4c52cfb8875886",
    ),
    "concerned_old_lower_primary_cleanup_mask_v7.png": (
        ROOT / "masks/mirelle_voss/concerned_old_lower_primary_cleanup_mask_v7.png",
        "734ae5940572f8ce040057b341ae14917fde80d6e1647b3fd125f20433474363",
    ),
    "concerned_lower_canonical_metal_patch_mask_v7.png": (
        ROOT / "masks/mirelle_voss/concerned_lower_canonical_metal_patch_mask_v7.png",
        "69db29bcc337bd779f620828f1e04c4d069f6a87d72f00d961bd57ca5e8e8fa5",
    ),
    "concerned_upper_cap_tassel_shaft_grip_exact_mask_v7.png": (
        ROOT / "masks/mirelle_voss/concerned_upper_cap_tassel_shaft_grip_exact_mask_v7.png",
        "a6737b0540ebb771edff1fbb2c44e34ab4b260fd15e4b395d43d43ec6b8b6618",
    ),
    "concerned_body_face_costume_exact_mask_v7.png": (
        ROOT / "masks/mirelle_voss/concerned_body_face_costume_exact_mask_v7.png",
        "b04050799acc1af62f969d090d974dbc235f1802be78b1e0acebcd611af24f4d",
    ),
    "concerned_base_subject_occlusion_restore_mask_v7.png": (
        ROOT / "masks/mirelle_voss/concerned_base_subject_occlusion_restore_mask_v7.png",
        "eb0971e65a30b454170c07bfbd5ed845b6a7b53006cdd3b35226fc224714c181",
    ),
    "concerned_lower_primary_aggregate_saved_mask_v7.png": (
        ROOT / "masks/mirelle_voss/concerned_lower_primary_aggregate_saved_mask_v7.png",
        "7331531d090617cdca4645c5e53ddfb023434abf15e6e65781318c8f3ed56807",
    ),
    "concerned_lower_primary_changed_mask_v7.png": (
        ROOT / "masks/mirelle_voss/concerned_lower_primary_changed_mask_v7.png",
        "499a8d990effa00eb7ca5381ea5fb22c4c44bc192c42924f42a7f4dd579b6334",
    ),
}

ARCHIVE = (
    ROOT
    / "rejected_provenance/mirelle_voss/"
    "concerned_v7_detached_old_metal_and_fragmented_guards"
)
ARCHIVE_MANIFEST = ARCHIVE / "rejection_manifest.json"

OUTPUT = (
    ROOT / "grafts/mirelle_voss/concerned_lower_primary_metal_reference_graft_v8.png"
)
CLEANED = (
    ROOT / "grafts/mirelle_voss/concerned_old_lower_primary_cleaned_base_v8.png"
)
OLD_HEAD_MASK = (
    ROOT / "masks/mirelle_voss/concerned_old_lower_primary_cleanup_mask_v8.png"
)
PATCH_MASK = (
    ROOT / "masks/mirelle_voss/concerned_lower_canonical_metal_patch_mask_v8.png"
)
UPPER_PROP_MASK = (
    ROOT
    / "masks/mirelle_voss/concerned_upper_cap_tassel_shaft_grip_exact_mask_v8.png"
)
BODY_MASK = (
    ROOT / "masks/mirelle_voss/concerned_body_face_costume_exact_mask_v8.png"
)
RESIDUAL_MASK = (
    ROOT / "masks/mirelle_voss/concerned_residual_old_silver_mask_v8.png"
)
CONNECTIVITY_MASK = (
    ROOT / "masks/mirelle_voss/concerned_connected_metal_component_mask_v8.png"
)
LEFT_GUARD_MASK = (
    ROOT / "masks/mirelle_voss/concerned_canonical_left_guard_mask_v8.png"
)
RIGHT_GUARD_MASK = (
    ROOT / "masks/mirelle_voss/concerned_canonical_right_guard_mask_v8.png"
)
AGGREGATE_MASK = (
    ROOT / "masks/mirelle_voss/concerned_lower_primary_aggregate_saved_mask_v8.png"
)
CHANGED_MASK = (
    ROOT / "masks/mirelle_voss/concerned_lower_primary_changed_mask_v8.png"
)

FULL_PROOF = ROOT / "component_proofs/mirelle_concerned_lower_primary_full_v8.png"
CLOSE_PROOF = ROOT / "component_proofs/mirelle_concerned_lower_primary_close_v8.png"
RESIDUAL_PROOF = ROOT / "component_proofs/mirelle_concerned_residual_silver_v8.png"
CONNECTIVITY_PROOF = (
    ROOT / "component_proofs/mirelle_concerned_metal_connectivity_v8.png"
)
MASK_PROOF = ROOT / "mask_proofs/mirelle_concerned_lower_primary_masks_v8.png"
RESIDUAL_REPORT = (
    ROOT / "component_metadata/mirelle_concerned_residual_silver_report_v8.json"
)
METADATA = (
    ROOT / "component_metadata/mirelle_concerned_lower_primary_metal_graft_v8.json"
)

EXPECTED_HASHES = {
    BASE: "ac5d892873234388c83080cf42bc5885d8944666cb6315df07d0969206fbb517",
    REJECTED_V6: "83ba93aee746c73cb1f0da1c6eba18725bc5de55237320b98a4e31423b0c1c88",
    METAL: "4f4b369d7f445b79a44673b33aee8bcc763f754eb37a0d7bfaee38213bf73b6f",
    DEFERRED_STREAMERS: "9d7ff6a32a4ad46ac491652b40f0fa5f0a2c8fb4f88cd67c1ebdb1a81e6970dd",
    MATERIAL_SPLIT_METADATA: "6729001c5d692c4d7a31087870c7b930c354284535dc05acf8e9c8f460e03d30",
    **{path: expected for path, expected in V7_FILES.values()},
}

SOURCE_SOCKET = np.array([200.0, 89.0], dtype=np.float64)
SOURCE_SHAFT_POINT = np.array([289.0, 13.0], dtype=np.float64)
TARGET_SOCKET_Y = 1128.0
SHAFT_SAMPLE_Y = 1010.0
MIN_ALPHA = 48
SUPERSAMPLE = 4
SCALE_CANDIDATES = [0.44, 0.45, 0.46, 0.47, 0.48]
MIN_SMALLER_GUARD_PIXELS = 400
MAX_RESIDUAL_SILVER_COMPONENT_AREA = 6
MAGENTA = np.array([255, 0, 255], dtype=np.uint8)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_v7_module():
    spec = importlib.util.spec_from_file_location("concerned_v7_frozen", V7_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen v7 helpers: {V7_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_hashes() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash fence failed for {path}: {actual} != {expected}")


def archive_v7() -> dict[str, object]:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for archive_name, (source, expected) in V7_FILES.items():
        destination = ARCHIVE / archive_name
        if destination.exists() and sha256(destination) != expected:
            raise RuntimeError(f"conflicting v7 archive file: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        records.append(
            {
                "source_path": str(source),
                "archive_path": str(destination),
                "sha256": sha256(destination),
            }
        )
    manifest = {
        "status": "rejected_preserved",
        "label": "concerned_v7",
        "rejection_slug": "detached_old_metal_and_fragmented_guards",
        "reason": (
            "Root review found detached old lower-head silver outside the new "
            "canonical patch and judged the reduced/occluded lateral guards fragmented."
        ),
        "selected_for_runtime": False,
        "model_calls": 0,
        "files": records,
    }
    write_json(ARCHIVE_MANIFEST, manifest)
    manifest["manifest_path"] = str(ARCHIVE_MANIFEST)
    manifest["manifest_sha256"] = sha256(ARCHIVE_MANIFEST)
    return manifest


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


def source_region(
    metal: np.ndarray, polygons: list[list[tuple[int, int]]]
) -> np.ndarray:
    image = Image.new("L", (metal.shape[1], metal.shape[0]), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    return (np.asarray(image, dtype=np.uint8) > 0) & (metal[..., 3] >= MIN_ALPHA)


def source_semantic_masks(metal: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "left_guard": source_region(
            metal,
            [[(85, 55), (190, 45), (202, 110), (155, 160), (85, 135)]],
        ),
        "right_guard": source_region(
            metal,
            [[(175, 74), (285, 70), (280, 170), (205, 170), (175, 115)]],
        ),
        "blade": source_region(
            metal,
            [[(0, 120), (190, 105), (205, 200), (20, 314)]],
        ),
        "socket": source_region(
            metal,
            [[(175, 45), (235, 45), (240, 120), (180, 125)]],
        ),
    }


def affine_matrix(
    shaft: dict[str, object], scale: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
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
    return matrix, rotation, target_socket, target_shaft_point


def transform_supersampled(
    metal: np.ndarray,
    size: tuple[int, int],
    matrix: np.ndarray,
) -> np.ndarray:
    alpha = metal[..., 3].astype(np.float32) / 255.0
    premultiplied = metal[..., :3].astype(np.float32) * alpha[..., None]
    width, height = size
    high_matrix = matrix.copy()
    high_matrix[:, :2] *= SUPERSAMPLE
    high_matrix[:, 2] *= SUPERSAMPLE
    high_alpha = cv2.warpAffine(
        alpha,
        high_matrix,
        (width * SUPERSAMPLE, height * SUPERSAMPLE),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    high_pm = cv2.warpAffine(
        premultiplied,
        high_matrix,
        (width * SUPERSAMPLE, height * SUPERSAMPLE),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    alpha_down = cv2.resize(
        np.clip(high_alpha, 0.0, 1.0),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    pm_down = cv2.resize(high_pm, (width, height), interpolation=cv2.INTER_AREA)
    rgb = np.zeros_like(pm_down)
    nonzero = alpha_down > 1e-5
    rgb[nonzero] = pm_down[nonzero] / alpha_down[nonzero, None]
    return np.dstack(
        [
            np.clip(np.round(rgb), 0, 255).astype(np.uint8),
            np.clip(np.round(alpha_down * 255.0), 0, 255).astype(np.uint8),
        ]
    )


def transform_binary(
    source: np.ndarray, size: tuple[int, int], matrix: np.ndarray
) -> np.ndarray:
    return cv2.warpAffine(
        source.astype(np.uint8),
        matrix,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0


def connectivity_record(
    patch: np.ndarray,
    transformed_semantics: dict[str, np.ndarray],
) -> tuple[dict[str, object], np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        patch.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise RuntimeError("canonical patch is empty")
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]
    largest_index = 1 + int(np.argmax(np.asarray(areas)))
    largest = labels == largest_index
    guard_records: dict[str, dict[str, object]] = {}
    for name in ("left_guard", "right_guard"):
        visible = transformed_semantics[name] & patch
        in_largest = visible & largest
        guard_records[name] = {
            "visible_pixels": int(np.count_nonzero(visible)),
            "pixels_in_largest_component": int(np.count_nonzero(in_largest)),
            "all_visible_pixels_in_largest_component": bool(
                np.array_equal(visible, in_largest)
            ),
            "bbox": bbox(visible),
        }
    record = {
        "alpha_threshold": MIN_ALPHA,
        "component_count": count - 1,
        "component_areas_descending": sorted(areas, reverse=True),
        "largest_component_pixels": int(np.count_nonzero(largest)),
        "patch_pixels": int(np.count_nonzero(patch)),
        "largest_component_fraction": float(
            np.count_nonzero(largest) / np.count_nonzero(patch)
        ),
        "guards": guard_records,
        "blade_pixels_in_largest": int(
            np.count_nonzero(transformed_semantics["blade"] & largest)
        ),
        "socket_pixels_in_largest": int(
            np.count_nonzero(transformed_semantics["socket"] & largest)
        ),
    }
    return record, largest


def select_scale(
    metal: np.ndarray,
    size: tuple[int, int],
    shaft: dict[str, object],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, object],
    list[dict[str, object]],
]:
    source_masks = source_semantic_masks(metal)
    candidates: list[dict[str, object]] = []
    selected = None
    for scale in SCALE_CANDIDATES:
        matrix, rotation, target_socket, target_shaft_point = affine_matrix(
            shaft, scale
        )
        transformed = transform_supersampled(metal, size, matrix)
        patch = transformed[..., 3] >= MIN_ALPHA
        semantics = {
            name: transform_binary(mask, size, matrix)
            for name, mask in source_masks.items()
        }
        connection, largest = connectivity_record(patch, semantics)
        smaller_guard = min(
            connection["guards"]["left_guard"]["visible_pixels"],
            connection["guards"]["right_guard"]["visible_pixels"],
        )
        passes = bool(
            connection["component_count"] == 1
            and connection["largest_component_fraction"] == 1.0
            and smaller_guard >= MIN_SMALLER_GUARD_PIXELS
            and all(
                value["all_visible_pixels_in_largest_component"]
                for value in connection["guards"].values()
            )
        )
        candidates.append(
            {
                "scale": scale,
                "passes": passes,
                "smaller_guard_visible_pixels": smaller_guard,
                "connectivity": connection,
            }
        )
        if passes and selected is None:
            selected = (
                scale,
                transformed,
                matrix,
                rotation,
                target_socket,
                target_shaft_point,
                semantics,
                connection,
                largest,
            )
    if selected is None:
        raise RuntimeError(f"no scale candidate met connectivity gates: {candidates}")
    (
        scale,
        transformed,
        matrix,
        rotation,
        target_socket,
        target_shaft_point,
        semantics,
        connection,
        largest,
    ) = selected
    return (
        scale,
        transformed,
        matrix,
        rotation,
        target_socket,
        target_shaft_point,
        semantics,
        connection,
        candidates,
    )


def old_head_cleanup(v7, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = base.shape[:2]
    core = v7.polygon_mask(
        (width, height),
        [
            [
                (608, 1128),
                (570, 1106),
                (548, 1114),
                (530, 1134),
                (506, 1174),
                (500, 1205),
                (540, 1190),
                (568, 1222),
                (614, 1186),
            ],
            [
                (603, 1128),
                (638, 1108),
                (660, 1118),
                (680, 1140),
                (701, 1195),
                (676, 1209),
                (650, 1184),
                (622, 1222),
                (594, 1184),
            ],
            [
                (590, 1150),
                (632, 1188),
                (616, 1290),
                (526, 1440),
                (505, 1438),
                (512, 1295),
                (545, 1200),
            ],
        ],
    )
    expanded = cv2.dilate(
        core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    ) > 0
    hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV)
    spread = (
        np.max(base, axis=2).astype(np.int16)
        - np.min(base, axis=2).astype(np.int16)
    )
    value = np.max(base, axis=2)
    key_distance = np.linalg.norm(
        base.astype(np.int16) - MAGENTA.astype(np.int16), axis=2
    )
    silver_or_outline = (
        (hsv[..., 1] <= 120)
        & (spread <= 120)
        & (value >= 18)
        & (key_distance > 3.0)
        & ~v7.red_material(base)
    )
    fringe = expanded & ~core & silver_or_outline
    cleanup = core | fringe
    return cleanup, expanded


def preservation_masks(
    v7,
    base: np.ndarray,
    shaft: dict[str, object],
    cleanup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    upper_prop, body = v7.preservation_masks(base, shaft, cleanup)
    body &= ~cleanup
    return upper_prop, body


def residual_silver_analysis(
    v7,
    result: np.ndarray,
    search_envelope: np.ndarray,
    patch: np.ndarray,
    upper_prop: np.ndarray,
    body: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    spread = (
        np.max(result, axis=2).astype(np.int16)
        - np.min(result, axis=2).astype(np.int16)
    )
    value = np.max(result, axis=2)
    silver_like = (
        (hsv[..., 1] <= 115)
        & (spread <= 110)
        & (value >= 24)
        & ~v7.key_like(result)
    )
    residual = silver_like & search_envelope & ~patch & ~upper_prop & ~body
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8), connectivity=8
    )
    components: list[dict[str, object]] = []
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        components.append(
            {
                "area_pixels": area,
                "bbox_xyxy_exclusive": [x, y, x + width, y + height],
            }
        )
    components.sort(key=lambda item: item["area_pixels"], reverse=True)
    nontrivial = [
        item
        for item in components
        if item["area_pixels"] > MAX_RESIDUAL_SILVER_COMPONENT_AREA
    ]
    report = {
        "search_envelope_bbox": bbox(search_envelope),
        "silver_classifier": {
            "hsv_saturation_max": 115,
            "rgb_spread_max": 110,
            "value_min": 24,
        },
        "excluded_legitimate_regions": [
            "new transformed rigid metal",
            "exact upper cap/tassel/shaft/grip/collar",
            "exact visible body/face/costume/greaves/boots",
        ],
        "tiny_noise_max_area_pixels": MAX_RESIDUAL_SILVER_COMPONENT_AREA,
        "residual_pixel_count": int(np.count_nonzero(residual)),
        "component_count": len(components),
        "largest_component_area_pixels": (
            components[0]["area_pixels"] if components else 0
        ),
        "components": components,
        "nontrivial_components": nontrivial,
        "passes": not nontrivial,
    }
    return residual, report


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


def tinted(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    shown = base.copy()
    shown[mask] = np.clip(
        np.round(
            shown[mask].astype(np.float32) * 0.28
            + np.asarray(color, dtype=np.float32) * 0.72
        ),
        0,
        255,
    ).astype(np.uint8)
    return shown


def make_proofs(
    base: np.ndarray,
    v7_rejected: np.ndarray,
    cleaned: np.ndarray,
    result: np.ndarray,
    metal: np.ndarray,
    transformed: np.ndarray,
    cleanup: np.ndarray,
    patch: np.ndarray,
    upper_prop: np.ndarray,
    body: np.ndarray,
    residual: np.ndarray,
    search_envelope: np.ndarray,
    largest: np.ndarray,
    semantics: dict[str, np.ndarray],
    residual_report: dict[str, object],
) -> None:
    full_panels = (
        fit_panel(Image.fromarray(base), (360, 610), "SEMANTIC BASE V2"),
        fit_panel(Image.fromarray(v7_rejected), (360, 610), "REJECTED V7 ARCHIVED"),
        fit_panel(Image.fromarray(result), (360, 610), "CONCERNED V8 CONNECTED"),
    )
    full_sheet = Image.new("RGB", (1080, 652), (12, 15, 20))
    for index, panel in enumerate(full_panels):
        full_sheet.paste(panel, (index * 360, 0))
    FULL_PROOF.parent.mkdir(parents=True, exist_ok=True)
    full_sheet.save(FULL_PROOF, optimize=True)

    crop = (470, 1040, 720, 1455)
    close_panels = (
        fit_panel(
            Image.fromarray(checker_component(metal)),
            (360, 670),
            "FROZEN RIGID METAL",
        ),
        fit_panel(
            Image.fromarray(v7_rejected).crop(crop),
            (360, 670),
            "V7 DETACHED/FRAGMENTED",
        ),
        fit_panel(
            Image.fromarray(cleaned).crop(crop),
            (360, 670),
            "V8 OLD HEAD CLEANED",
        ),
        fit_panel(
            Image.fromarray(result).crop(crop),
            (360, 670),
            "V8 BLADE + TWO GUARDS",
        ),
    )
    close_sheet = Image.new("RGB", (1440, 712), (12, 15, 20))
    for index, panel in enumerate(close_panels):
        close_sheet.paste(panel, (index * 360, 0))
    CLOSE_PROOF.parent.mkdir(parents=True, exist_ok=True)
    close_sheet.save(CLOSE_PROOF, optimize=True)

    residual_overlay = tinted(result, residual, (255, 40, 40))
    envelope_overlay = tinted(result, search_envelope, (255, 220, 0))
    residual_panels = (
        fit_panel(Image.fromarray(result).crop(crop), (420, 670), "V8 OUTPUT"),
        fit_panel(
            Image.fromarray(envelope_overlay).crop(crop),
            (420, 670),
            "RESIDUAL SEARCH ENVELOPE",
        ),
        fit_panel(
            Image.fromarray(residual_overlay).crop(crop),
            (420, 670),
            "RESIDUAL SILVER: "
            f"MAX {residual_report['largest_component_area_pixels']} PX",
        ),
    )
    residual_sheet = Image.new("RGB", (1260, 712), (12, 15, 20))
    for index, panel in enumerate(residual_panels):
        residual_sheet.paste(panel, (index * 420, 0))
    RESIDUAL_PROOF.parent.mkdir(parents=True, exist_ok=True)
    residual_sheet.save(RESIDUAL_PROOF, optimize=True)

    connectivity_overlay = result.copy()
    connectivity_overlay[largest] = np.clip(
        np.round(
            connectivity_overlay[largest].astype(np.float32) * 0.45
            + np.array([0, 210, 255], dtype=np.float32) * 0.55
        ),
        0,
        255,
    ).astype(np.uint8)
    left = semantics["left_guard"] & patch
    right = semantics["right_guard"] & patch
    connectivity_overlay[left] = np.array([255, 220, 0], dtype=np.uint8)
    connectivity_overlay[right] = np.array([255, 70, 90], dtype=np.uint8)
    alpha_rgb = np.repeat(transformed[..., 3:4], 3, axis=2)
    connectivity_panels = (
        fit_panel(
            Image.fromarray(checker_component(metal)),
            (360, 670),
            "CANONICAL SOURCE",
        ),
        fit_panel(
            Image.fromarray(alpha_rgb).crop(crop),
            (360, 670),
            f"SUPERSAMPLED ALPHA >= {MIN_ALPHA}",
        ),
        fit_panel(
            Image.fromarray(connectivity_overlay).crop(crop),
            (360, 670),
            "ONE CYAN COMPONENT; GUARDS YELLOW/RED",
        ),
        fit_panel(
            Image.fromarray(result).crop(crop),
            (360, 670),
            "NATIVE V8",
        ),
    )
    connectivity_sheet = Image.new("RGB", (1440, 712), (12, 15, 20))
    for index, panel in enumerate(connectivity_panels):
        connectivity_sheet.paste(panel, (index * 360, 0))
    CONNECTIVITY_PROOF.parent.mkdir(parents=True, exist_ok=True)
    connectivity_sheet.save(CONNECTIVITY_PROOF, optimize=True)

    masks = (
        ("OLD HEAD CLEANUP", cleanup, (255, 220, 0)),
        ("CANONICAL PATCH", patch, (0, 180, 255)),
        ("UPPER PROP EXACT", upper_prop, (255, 70, 90)),
        ("BODY/COSTUME EXACT", body, (255, 150, 0)),
        ("RESIDUAL SILVER", residual, (255, 40, 40)),
    )
    mask_sheet = Image.new("RGB", (280 * len(masks), 662), (12, 15, 20))
    for index, (title, mask, color) in enumerate(masks):
        panel = fit_panel(
            Image.fromarray(tinted(result, mask, color)).crop(crop),
            (280, 620),
            title,
        )
        mask_sheet.paste(panel, (index * 280, 0))
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    mask_sheet.save(MASK_PROOF, optimize=True)


def record(path: Path, pixels: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": str(path), "sha256": sha256(path)}
    if pixels is not None:
        value["pixels"] = pixels
    return value


def main() -> None:
    check_hashes()
    archive_record = archive_v7()
    v7 = load_v7_module()
    base = np.asarray(Image.open(BASE).convert("RGB"), dtype=np.uint8)
    rejected_v7 = np.asarray(
        Image.open(V7_FILES["concerned_lower_primary_metal_reference_graft_v7.png"][0]).convert("RGB"),
        dtype=np.uint8,
    )
    rejected_v6 = np.asarray(Image.open(REJECTED_V6).convert("RGB"), dtype=np.uint8)
    metal = np.asarray(Image.open(METAL).convert("RGBA"), dtype=np.uint8)
    if tuple(base.shape) != (1455, 1081, 3):
        raise RuntimeError(f"unexpected base shape: {base.shape}")
    if tuple(metal.shape) != (314, 312, 4):
        raise RuntimeError(f"unexpected metal shape: {metal.shape}")

    shaft = v7.fit_concerned_shaft(base)
    (
        scale,
        transformed,
        matrix,
        rotation,
        target_socket,
        target_shaft_point,
        semantics,
        connection,
        candidates,
    ) = select_scale(metal, (base.shape[1], base.shape[0]), shaft)
    patch = transformed[..., 3] >= MIN_ALPHA
    cleanup, search_envelope = old_head_cleanup(v7, base)
    upper_prop, body = preservation_masks(v7, base, shaft, cleanup)
    body &= ~(cleanup | patch)

    cleaned = base.copy()
    cleaned[cleanup] = MAGENTA
    result = cleaned.copy()
    result[patch] = transformed[..., :3][patch]

    aggregate = cleanup | patch
    changed = np.any(result != base, axis=2)
    outside_saved_delta = int(np.count_nonzero(changed & ~aggregate))
    outside_prop_delta = int(np.count_nonzero(changed & ~(cleanup | patch)))
    upper_prop_delta = int(np.count_nonzero(changed & upper_prop))
    body_delta = int(np.count_nonzero(changed & body))
    if outside_saved_delta or outside_prop_delta:
        raise RuntimeError(
            f"scope failure: saved={outside_saved_delta}, prop={outside_prop_delta}"
        )
    if upper_prop_delta or body_delta:
        raise RuntimeError(
            f"preservation failure: upper prop={upper_prop_delta}, body={body_delta}"
        )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        patch.astype(np.uint8), connectivity=8
    )
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = labels == largest_index
    if connection["component_count"] != 1 or not np.array_equal(largest, patch):
        raise RuntimeError(f"selected canonical metal is disconnected: {connection}")
    for name in ("left_guard", "right_guard"):
        guard = semantics[name] & patch
        if np.count_nonzero(guard) < MIN_SMALLER_GUARD_PIXELS:
            raise RuntimeError(f"{name} is not significant: {np.count_nonzero(guard)}")
        if not np.all(largest[guard]):
            raise RuntimeError(f"{name} is detached from the primary component")

    residual, residual_report = residual_silver_analysis(
        v7, result, search_envelope, patch, upper_prop, body
    )
    if not residual_report["passes"]:
        raise RuntimeError(
            f"nontrivial residual old silver remains: {residual_report}"
        )

    hot_magenta = patch & v7.key_like(result)
    if np.any(hot_magenta):
        raise RuntimeError(
            f"hot-magenta visible metal pixels: {np.count_nonzero(hot_magenta)}"
        )
    new_red = v7.red_material(result) & ~v7.red_material(base)
    red_count, _red_labels, red_stats, _ = cv2.connectedComponentsWithStats(
        new_red.astype(np.uint8), connectivity=8
    )
    red_areas = [
        int(red_stats[index, cv2.CC_STAT_AREA])
        for index in range(1, red_count)
    ]
    cloth_like = [area for area in red_areas if area >= 8]
    if cloth_like:
        raise RuntimeError(f"lower cloth-like red components exist: {cloth_like}")

    rejected_v6_streamer = (
        v7.red_material(rejected_v6)
        & ~v7.red_material(base)
        & search_envelope
    )
    retained_v6_streamer = rejected_v6_streamer & np.all(result == rejected_v6, axis=2)
    if np.any(retained_v6_streamer):
        raise RuntimeError(
            f"retained rejected-v6 streamer pixels: {np.count_nonzero(retained_v6_streamer)}"
        )

    for path in (
        OUTPUT,
        CLEANED,
        OLD_HEAD_MASK,
        PATCH_MASK,
        UPPER_PROP_MASK,
        BODY_MASK,
        RESIDUAL_MASK,
        CONNECTIVITY_MASK,
        LEFT_GUARD_MASK,
        RIGHT_GUARD_MASK,
        AGGREGATE_MASK,
        CHANGED_MASK,
        FULL_PROOF,
        CLOSE_PROOF,
        RESIDUAL_PROOF,
        CONNECTIVITY_PROOF,
        MASK_PROOF,
        RESIDUAL_REPORT,
        METADATA,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(OUTPUT, optimize=True)
    Image.fromarray(cleaned, mode="RGB").save(CLEANED, optimize=True)
    left_guard = semantics["left_guard"] & patch
    right_guard = semantics["right_guard"] & patch
    for mask, path in (
        (cleanup, OLD_HEAD_MASK),
        (patch, PATCH_MASK),
        (upper_prop, UPPER_PROP_MASK),
        (body, BODY_MASK),
        (residual, RESIDUAL_MASK),
        (largest, CONNECTIVITY_MASK),
        (left_guard, LEFT_GUARD_MASK),
        (right_guard, RIGHT_GUARD_MASK),
        (aggregate, AGGREGATE_MASK),
        (changed, CHANGED_MASK),
    ):
        save_mask(mask, path)
    write_json(RESIDUAL_REPORT, residual_report)

    make_proofs(
        base,
        rejected_v7,
        cleaned,
        result,
        metal,
        transformed,
        cleanup,
        patch,
        upper_prop,
        body,
        residual,
        search_envelope,
        largest,
        semantics,
        residual_report,
    )

    masks: dict[str, dict[str, object]] = {}
    for name, mask, path in (
        ("old_lower_primary_cleanup", cleanup, OLD_HEAD_MASK),
        ("canonical_lower_primary_metal_patch", patch, PATCH_MASK),
        ("upper_cap_tassel_shaft_grip_exact", upper_prop, UPPER_PROP_MASK),
        ("body_face_costume_exact", body, BODY_MASK),
        ("residual_old_silver", residual, RESIDUAL_MASK),
        ("connected_metal_component", largest, CONNECTIVITY_MASK),
        ("canonical_left_guard", left_guard, LEFT_GUARD_MASK),
        ("canonical_right_guard", right_guard, RIGHT_GUARD_MASK),
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
        "version": "deterministic_lower_primary_metal_v8",
        "method": (
            "semantic concerned v2 base; full old-head chroma cleanup; 4x "
            "premultiplied-alpha supersampled rigid-metal transform; no model call"
        ),
        "model_calls": 0,
        "archive": archive_record,
        "inputs": {
            "semantic_base": record(BASE),
            "frozen_rigid_metal": record(METAL),
            "frozen_streamers_deferred_not_transformed": record(DEFERRED_STREAMERS),
            "material_split_metadata": record(MATERIAL_SPLIT_METADATA),
            "rejected_v6_preserved_not_pixel_source": record(REJECTED_V6),
            "rejected_v7_preserved_not_pixel_source": record(
                V7_FILES["concerned_lower_primary_metal_reference_graft_v7.png"][0]
            ),
            "script": record(Path(__file__).resolve()),
        },
        "endpoint_determination": {
            "selected_endpoint": "lower primary blade, socket, and paired guards",
            "opposite_endpoint": "upper cap and short tassel",
            "classification_basis": "relative design and physical size, not screen direction",
        },
        "shaft_fit": shaft,
        "transform": {
            "source_socket": SOURCE_SOCKET.tolist(),
            "source_shaft_point": SOURCE_SHAFT_POINT.tolist(),
            "target_socket": target_socket.tolist(),
            "target_shaft_point": target_shaft_point.tolist(),
            "rotation_degrees": rotation,
            "uniform_scale": scale,
            "matrix": matrix.tolist(),
            "nonuniform_deformation": False,
            "supersample_factor": SUPERSAMPLE,
            "method": (
                "warp premultiplied RGB and alpha at 4x target resolution with "
                "Lanczos4, area-downsample, then unpremultiply"
            ),
            "patch_bbox": bbox(patch),
            "scale_selection": {
                "candidate_scales": SCALE_CANDIDATES,
                "policy": (
                    "smallest candidate with one complete component, 100% largest-"
                    "component coverage, both guards connected, and the smaller guard "
                    f"at least {MIN_SMALLER_GUARD_PIXELS} pixels"
                ),
                "candidate_results": candidates,
            },
        },
        "connectivity": connection,
        "residual_silver": {
            **residual_report,
            "report": record(RESIDUAL_REPORT),
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
            "new_lower_warm_red_component_areas": red_areas,
            "new_lower_cloth_like_red_components_area_ge_8": len(cloth_like),
            "rejected_v6_streamer_retained_pixels": int(
                np.count_nonzero(retained_v6_streamer)
            ),
            "automated": "pass",
        },
        "outputs": {
            "repaired": record(OUTPUT),
            "cleaned_base": record(CLEANED),
            "full_proof": record(FULL_PROOF),
            "close_proof": record(CLOSE_PROOF),
            "residual_silver_proof": record(RESIDUAL_PROOF),
            "connectivity_proof": record(CONNECTIVITY_PROOF),
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
