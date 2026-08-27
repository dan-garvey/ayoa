#!/usr/bin/env python3
"""Extract and graft Mirelle's canonical lower spear assembly deterministically."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
REFERENCE = (
    ROOT
    / "supporting_refs/mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
TARGET = ROOT / "generation_raw/mirelle_voss/concerned_chroma_v2.png"
EXTRACTION_MASK = (
    ROOT / "masks/mirelle_voss/canonical_lower_spearhead_extraction_mask_v5.png"
)
COMPONENT = ROOT / "components/mirelle_voss/canonical_lower_spearhead_streamers_rgba_v5.png"
ERASE_MASK = ROOT / "masks/mirelle_voss/concerned_old_lower_spear_erase_mask_v5.png"
GRAFT_MASK = ROOT / "masks/mirelle_voss/concerned_reference_graft_alpha_mask_v5.png"
GRAFT = ROOT / "grafts/mirelle_voss/concerned_lower_spear_reference_graft_v5.png"
EXTRACTION_PROOF = ROOT / "component_proofs/mirelle_canonical_lower_spear_extraction_v5.png"
GRAFT_PROOF = ROOT / "component_proofs/mirelle_concerned_reference_graft_v5.png"
METADATA = ROOT / "component_metadata/mirelle_concerned_reference_graft_v5.json"

BACKGROUND_DISTANCE = 30.0
SOURCE_ANCHOR = np.array([248.0, 174.0], dtype=np.float64)
TARGET_ANCHOR = np.array([608.0, 1128.0], dtype=np.float64)
SOURCE_SHAFT_TOP = np.array([337.0, 98.0], dtype=np.float64)
TARGET_SHAFT_TOP = np.array([631.0, 1010.0], dtype=np.float64)
ROTATION_DEGREES = float(
    np.degrees(
        np.arctan2(*(TARGET_SHAFT_TOP - TARGET_ANCHOR)[::-1])
        - np.arctan2(*(SOURCE_SHAFT_TOP - SOURCE_ANCHOR)[::-1])
    )
)
UNIFORM_SCALE = 0.90


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    value = np.clip((value - low) / (high - low), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def manual_reference_clip(size: tuple[int, int]) -> np.ndarray:
    clip = Image.new("L", size, 0)
    ImageDraw.Draw(clip).polygon(
        [
            (30, 420),
            (30, 175),
            (115, 135),
            (130, 85),
            (330, 85),
            (342, 105),
            (360, 125),
            (360, 220),
            (300, 285),
            (225, 420),
        ],
        fill=255,
    )
    return np.asarray(clip, dtype=np.uint8) > 0


def extract_reference() -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    rgb = np.asarray(Image.open(REFERENCE).convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    border = np.concatenate(
        [
            rgb[:12].reshape(-1, 3),
            rgb[-12:].reshape(-1, 3),
            rgb[:, :12].reshape(-1, 3),
        ],
        axis=0,
    )
    key = np.median(border, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - key, axis=2)
    clip = manual_reference_clip((width, height))
    grab = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    grab[clip] = cv2.GC_PR_BGD
    grab[clip & (distance >= 38.0)] = cv2.GC_PR_FGD
    grab[clip & (distance <= 3.0)] = cv2.GC_BGD

    # Definite-foreground strokes cover each disconnected material family so
    # GrabCut retains silver interiors as well as the fine red ribbons.
    cv2.line(grab, (52, 392), (245, 177), cv2.GC_FGD, 7)
    cv2.line(grab, (248, 174), (338, 98), cv2.GC_FGD, 7)
    cv2.polylines(
        grab,
        [np.asarray([(155, 116), (180, 101), (230, 103), (285, 106), (327, 98)])],
        False,
        cv2.GC_FGD,
        6,
    )
    cv2.polylines(
        grab,
        [np.asarray([(138, 166), (174, 154), (215, 159), (250, 181), (290, 194), (328, 169), (352, 138)])],
        False,
        cv2.GC_FGD,
        6,
    )
    cv2.polylines(
        grab,
        [np.asarray([(250, 177), (278, 211), (316, 205), (350, 158)])],
        False,
        cv2.GC_FGD,
        6,
    )
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        rgb,
        grab,
        None,
        background_model,
        foreground_model,
        8,
        cv2.GC_INIT_WITH_MASK,
    )
    subject = (
        ((grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD))
        & clip
        & (distance > 20.0)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        subject.astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(subject)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= 8:
            kept[labels == index] = True
    inside_distance = cv2.distanceTransform(kept.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[kept] = 255
    boundary = kept & (inside_distance <= 1.5)
    alpha[boundary] = np.maximum(
        48,
        np.round(255.0 * smoothstep(distance[boundary], 2.0, 18.0)),
    ).astype(np.uint8)

    # The target's long red shaft is already canonical and remains byte-derived
    # from the target. Remove only the reference crop's duplicate straight
    # shaft segment; keep the socket, blade, wing guards, and streamers.
    duplicate_shaft = np.zeros((height, width), dtype=np.uint8)
    cv2.line(duplicate_shaft, (343, 91), (282, 143), 255, 14)
    alpha[duplicate_shaft > 0] = 0

    ys, xs = np.where(alpha > 0)
    if not xs.size:
        raise RuntimeError("reference extraction produced an empty component")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    full_rgba = np.dstack([rgb, alpha])
    component = full_rgba[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
    component[component[..., 3] == 0, :3] = 0
    return full_rgba, component, bbox, key


def transform_component(full_rgba: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    radians = np.deg2rad(ROTATION_DEGREES)
    cosine = float(np.cos(radians) * UNIFORM_SCALE)
    sine = float(np.sin(radians) * UNIFORM_SCALE)
    matrix = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
        ],
        dtype=np.float64,
    )
    matrix[:, 2] = TARGET_ANCHOR - matrix[:, :2] @ SOURCE_ANCHOR
    alpha = full_rgba[..., 3].astype(np.float32) / 255.0
    premultiplied = full_rgba[..., :3].astype(np.float32) * alpha[..., None]
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
    warped_rgb[nonzero] = (
        warped_premultiplied[nonzero] / warped_alpha[nonzero, None]
    )
    return np.dstack(
        [
            np.clip(np.round(warped_rgb), 0, 255).astype(np.uint8),
            np.clip(np.round(warped_alpha * 255.0), 0, 255).astype(np.uint8),
        ]
    )


def old_prop_erase_mask(size: tuple[int, int]) -> np.ndarray:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(
        [
            (603, 1135),
            (574, 1114),
            (556, 1140),
            (538, 1194),
            (568, 1172),
            (591, 1210),
            (612, 1181),
        ],
        fill=255,
    )
    draw.polygon(
        [
            (607, 1138),
            (634, 1119),
            (652, 1147),
            (675, 1201),
            (646, 1172),
            (622, 1213),
            (598, 1181),
        ],
        fill=255,
    )
    draw.polygon(
        [
            (595, 1160),
            (625, 1197),
            (602, 1280),
            (518, 1425),
            (521, 1302),
            (556, 1204),
        ],
        fill=255,
    )
    return np.asarray(mask, dtype=np.uint8)


def composite_graft(target: np.ndarray, transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    erase = old_prop_erase_mask((target.shape[1], target.shape[0]))
    base = target.copy()
    magenta = np.array([255, 0, 255], dtype=np.uint8)
    base[erase > 0] = magenta
    alpha = transformed[..., 3].astype(np.float32) / 255.0
    result = np.round(
        transformed[..., :3].astype(np.float32) * alpha[..., None]
        + base.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)
    return result, erase


def dark_light(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size, (235, 238, 242))
    ImageDraw.Draw(image).rectangle((0, 0, size[0] // 2, size[1]), fill=(31, 36, 44))
    return image


def build_proofs(component: np.ndarray, graft: np.ndarray, transformed: np.ndarray) -> None:
    rgba = Image.fromarray(component, mode="RGBA")
    panel = dark_light((700, 700))
    shown = rgba.copy()
    shown.thumbnail((640, 640), Image.Resampling.LANCZOS)
    panel.paste(shown, ((700 - shown.width) // 2, (700 - shown.height) // 2), shown)
    EXTRACTION_PROOF.parent.mkdir(parents=True, exist_ok=True)
    panel.save(EXTRACTION_PROOF, optimize=True)

    with Image.open(TARGET) as image:
        original = image.convert("RGB")
    graft_image = Image.fromarray(graft, mode="RGB")
    transformed_image = Image.fromarray(transformed, mode="RGBA")
    crop = (440, 900, 760, 1455)
    cells = []
    for title, image in (
        ("ORIGINAL", original),
        ("TRANSFORMED COMPONENT", Image.new("RGB", original.size, (255, 0, 255))),
        ("DETERMINISTIC GRAFT", graft_image),
    ):
        if title == "TRANSFORMED COMPONENT":
            image.paste(transformed_image, (0, 0), transformed_image)
        cell = image.crop(crop).resize((480, 832), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (480, 884), (12, 15, 20))
        canvas.paste(cell, (0, 0))
        ImageDraw.Draw(canvas).text((12, 842), title, fill=(255, 255, 255))
        cells.append(canvas)
    sheet = Image.new("RGB", (1440, 884), (12, 15, 20))
    for index, cell in enumerate(cells):
        sheet.paste(cell, (index * 480, 0))
    GRAFT_PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(GRAFT_PROOF, optimize=True)


def main() -> None:
    full_rgba, component, bbox, key = extract_reference()
    for path in (EXTRACTION_MASK, COMPONENT, ERASE_MASK, GRAFT_MASK, GRAFT):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full_rgba[..., 3], mode="L").save(EXTRACTION_MASK, optimize=True)
    Image.fromarray(component, mode="RGBA").save(COMPONENT, optimize=True)

    target = np.asarray(Image.open(TARGET).convert("RGB"), dtype=np.uint8)
    transformed = transform_component(full_rgba, (target.shape[1], target.shape[0]))
    graft, erase = composite_graft(target, transformed)
    Image.fromarray(erase, mode="L").save(ERASE_MASK, optimize=True)
    Image.fromarray(transformed[..., 3], mode="L").save(GRAFT_MASK, optimize=True)
    Image.fromarray(graft, mode="RGB").save(GRAFT, optimize=True)
    build_proofs(component, graft, transformed)

    metadata = {
        "status": "pending_manual_review",
        "reference": {"path": str(REFERENCE), "sha256": sha256(REFERENCE)},
        "target": {"path": str(TARGET), "sha256": sha256(TARGET)},
        "extraction": {
            "method": "manual prop-only polygon plus seeded GrabCut and near-key rejection; exact source RGB retained",
            "near_key_rejection_distance": 20.0,
            "background_key_rgb": [float(value) for value in key],
            "background_distance": BACKGROUND_DISTANCE,
            "manual_polygon": [
                [30, 420], [30, 175], [115, 135], [130, 85], [330, 85],
                [342, 105], [360, 125], [360, 220], [300, 285], [225, 420],
            ],
            "source_bbox": bbox,
            "mask_path": str(EXTRACTION_MASK),
            "mask_sha256": sha256(EXTRACTION_MASK),
            "component_path": str(COMPONENT),
            "component_sha256": sha256(COMPONENT),
        },
        "transform": {
            "source_socket_anchor": SOURCE_ANCHOR.tolist(),
            "target_socket_anchor": TARGET_ANCHOR.tolist(),
            "source_shaft_top": SOURCE_SHAFT_TOP.tolist(),
            "target_shaft_top": TARGET_SHAFT_TOP.tolist(),
            "rotation_degrees": ROTATION_DEGREES,
            "uniform_scale": UNIFORM_SCALE,
            "duplicate_reference_shaft_removed": True,
            "nonuniform_deformation": False,
            "resampling": "premultiplied RGBA Lanczos4",
        },
        "graft": {
            "erase_mask_path": str(ERASE_MASK),
            "erase_mask_sha256": sha256(ERASE_MASK),
            "transformed_alpha_mask_path": str(GRAFT_MASK),
            "transformed_alpha_mask_sha256": sha256(GRAFT_MASK),
            "path": str(GRAFT),
            "sha256": sha256(GRAFT),
            "dimensions": [target.shape[1], target.shape[0]],
        },
        "proofs": {
            "extraction": {"path": str(EXTRACTION_PROOF), "sha256": sha256(EXTRACTION_PROOF)},
            "graft": {"path": str(GRAFT_PROOF), "sha256": sha256(GRAFT_PROOF)},
        },
    }
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
