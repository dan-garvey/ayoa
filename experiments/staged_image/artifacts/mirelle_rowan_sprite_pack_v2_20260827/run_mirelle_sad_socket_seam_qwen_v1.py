#!/usr/bin/env python3
"""Run exactly one bounded Qwen socket-seam experiment for Mirelle sad v1."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
HELPER_PATH = ROOT / "run_mirelle_neutral_socket_seam_qwen_v1.py"
EXPECTED_HELPER_SHA = "9303d43ceab0f7f4ac2d31c6f4f081651cb7fd9353befb1b7209c3c73e75e94a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("mirelle_neutral_qwen_helper", HELPER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import helper {HELPER_PATH}")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

helper.PRIMARY = ROOT / "grafts/mirelle_voss/sad_canonical_metal_repair_v1.png"
helper.REFERENCE = (
    ROOT
    / "supporting_refs/"
    "mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
helper.METAL_PATCH = (
    ROOT / "masks/mirelle_voss/sad_canonical_metal_component_patch_mask_v1.png"
)
POSE_TASSEL = (
    ROOT / "masks/mirelle_voss/sad_existing_pose_tassel_occlusion_mask_v1.png"
)
helper.MASK = ROOT / "masks/mirelle_voss/sad_socket_background_seam_qwen_mask_v1.png"
helper.MASK_PROOF = ROOT / "mask_proofs/mirelle_sad_socket_seam_qwen_mask_v1.png"
helper.PROMPT_PATH = ROOT / "prompts/qwen/mirelle_sad_socket_seam_qwen_v1.txt"
helper.REQUEST_BODY = ROOT / "qwen_requests/mirelle_sad_socket_seam_qwen_v1.json"
helper.PREFLIGHT_HEALTH = (
    ROOT / "qwen_requests/mirelle_sad_socket_seam_qwen_v1_health.json"
)
helper.RAW_OUTPUT = ROOT / "qwen_raw/mirelle_voss/sad_socket_seam_qwen_v1_model_output.png"
helper.ERROR_BODY = ROOT / "qwen_raw/mirelle_voss/sad_socket_seam_qwen_v1_error.bin"
helper.ALIGNED_OUTPUT = (
    ROOT / "qwen_aligned/mirelle_voss/sad_socket_seam_qwen_v1_aligned.png"
)
helper.COMPOSITE = ROOT / "qwen_composited/mirelle_voss/sad_socket_seam_qwen_v1.png"
helper.HEADERS = ROOT / "qwen_headers/mirelle_sad_socket_seam_qwen_v1.json"
helper.METADATA = ROOT / "qwen_metadata/mirelle_sad_socket_seam_qwen_v1.json"
helper.COMPARISON = (
    ROOT / "component_proofs/mirelle_sad_socket_seam_qwen_v1_comparison.png"
)
helper.SEED = 260827332
helper.STEPS = 24
helper.CFG = 3.2
helper.DENOISE = 0.20
helper.EXPECTED_PRIMARY_SHA = (
    "512d710eeba7f5accddd73e27dd2801afd4dd6ad2cab070acd763f11bed31ca6"
)
helper.EXPECTED_REFERENCE_SHA = (
    "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff"
)
helper.EXPECTED_PATCH_SHA = (
    "93cb85aae149fc7fa86e7611b15d0b344a6a6691d24c4b579cbfb317a9eeae93"
)
helper.PROMPT = """Use case: precise-object-edit
Asset type: one-off masked seam experiment for a visual-novel character sprite
Input roles: Image 1 is the approved deterministic Sad-pose edit target. Image 2 is Mirelle's locked canonical spear crop and is supporting evidence only for the silver collar/socket material and edge treatment.
Primary request: Make the smallest possible cleanup only at the silver collar/socket-to-magenta-background junction inside the white mask. Remove any jagged magenta notch, fringe, or pasted edge there and make the cylindrical silver collar meet the existing red shaft cleanly. Preserve the supplied collar size, position, shading, and silhouette.
Preservation contract: Change nothing outside the white mask. Do not redraw, move, resize, simplify, or restyle the central blade, lateral guards, socket, red shaft, or pose-specific red tassel and cloth. Do not copy Image 2's orientation, streamers, body, or background. Preserve Mirelle's sad face, hair, body, pose, hands, clothing, armor, boots, lower butt cap, framing, and magenta screen exactly. Add no weapon parts, streamers, people, text, scenery, shadow, or texture.
Mask semantics: White is the only eligible seam region. Black must remain unchanged."""

base_verify_inputs = helper.verify_inputs


def verify_inputs() -> None:
    actual_helper_sha = sha256(HELPER_PATH)
    if actual_helper_sha != EXPECTED_HELPER_SHA:
        raise RuntimeError(
            f"helper hash fence failed: expected {EXPECTED_HELPER_SHA}, got {actual_helper_sha}"
        )
    base_verify_inputs()
    if sha256(POSE_TASSEL) != (
        "3122b37c0af8cb67fcf92ee7b1d8522ad1ac67c998ddfc213387fb1f4285502f"
    ):
        raise RuntimeError("sad pose-tassel hash fence failed")


def make_mask() -> int:
    primary = np.asarray(Image.open(helper.PRIMARY).convert("RGB"), dtype=np.uint8)
    patch = np.asarray(
        Image.open(helper.METAL_PATCH).convert("L"), dtype=np.uint8
    ) > 0
    pose_tassel = np.asarray(
        Image.open(POSE_TASSEL).convert("L"), dtype=np.uint8
    ) > 0
    outer = cv2.dilate(
        patch.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    ) > 0
    inner = cv2.erode(
        patch.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ) > 0
    boundary = outer & ~inner

    # Sad's head is smaller than Neutral's.  This authored box starts below the
    # two lateral guards and ends before the red shaft/tassel.  It includes only
    # the cylindrical socket perimeter and an adjacent chroma ring.
    zone = np.zeros_like(patch)
    zone[135:158, 372:397] = True
    red = (
        (primary[..., 0] >= 48)
        & (
            primary[..., 0].astype(np.int16)
            >= primary[..., 1].astype(np.int16) + 18
        )
        & (
            primary[..., 0].astype(np.int16)
            >= primary[..., 2].astype(np.int16) + 8
        )
        & (primary[..., 2] <= 150)
    )
    protected_red_and_cloth = cv2.dilate(
        (red | pose_tassel).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ) > 0
    mask = boundary & zone & ~protected_red_and_cloth
    if not (200 <= int(mask.sum()) <= 260):
        raise RuntimeError(f"unexpected sad tiny seam mask size: {int(mask.sum())}")
    ys, xs = np.where(mask)
    if xs.min() < 372 or xs.max() > 396 or ys.min() < 135 or ys.max() > 157:
        raise RuntimeError("sad seam mask escaped its authored socket zone")
    if np.any(mask & protected_red_and_cloth):
        raise RuntimeError("sad seam mask overlaps protected shaft/tassel/cloth")

    helper.MASK.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        helper.MASK, optimize=True
    )
    make_mask_proof(primary, mask)
    return int(mask.sum())


def make_mask_proof(primary: np.ndarray, mask: np.ndarray) -> None:
    crop = (350, 105, 410, 175)
    base_crop = Image.fromarray(primary, mode="RGB").crop(crop)
    tinted = primary.copy()
    tinted[mask] = np.round(
        tinted[mask].astype(np.float32) * 0.25
        + np.asarray([0, 255, 255], dtype=np.float32) * 0.75
    ).astype(np.uint8)
    overlay_crop = Image.fromarray(tinted, mode="RGB").crop(crop)
    binary_crop = Image.fromarray(
        np.where(mask, 255, 0).astype(np.uint8), mode="L"
    ).convert("RGB").crop(crop)
    entries = [
        ("APPROVED SAD V1", base_crop),
        ("TINY SOCKET MASK", overlay_crop),
        ("BINARY MASK", binary_crop),
    ]
    panel_size = (420, 490)
    label_height = 48
    sheet = Image.new(
        "RGB", (panel_size[0] * len(entries), panel_size[1] + label_height), (12, 15, 20)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(entries):
        panel = helper.contained_panel(image, panel_size)
        sheet.paste(panel, (index * panel_size[0], 0))
        draw.text(
            (index * panel_size[0] + 10, panel_size[1] + 15),
            title,
            fill=(255, 255, 255),
        )
    helper.MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(helper.MASK_PROOF, optimize=True)


def exact_request_payload() -> dict[str, Any]:
    return {
        "prompt": helper.PROMPT,
        "image_base64": helper.encode_image(helper.PRIMARY),
        "image2_base64": helper.encode_image(helper.REFERENCE),
        "mask_base64": helper.encode_image(helper.MASK),
        "seed": helper.SEED,
        "steps": helper.STEPS,
        "cfg": helper.CFG,
        "denoise": helper.DENOISE,
        "filename_prefix": "mirelle_sad_socket_seam_qwen_v1",
    }


def make_comparison(
    primary: np.ndarray,
    raw_resized: np.ndarray,
    aligned: np.ndarray,
    composite: np.ndarray,
) -> None:
    crop = (350, 105, 410, 175)
    entries = [
        ("APPROVED SAD V1", Image.fromarray(primary, mode="RGB").crop(crop)),
        ("RAW RESIZED", Image.fromarray(raw_resized, mode="RGB").crop(crop)),
        ("RAW ALIGNED", Image.fromarray(aligned, mode="RGB").crop(crop)),
        ("HARD-MASK COMPOSITE", Image.fromarray(composite, mode="RGB").crop(crop)),
    ]
    panel_size = (330, 385)
    label_height = 46
    sheet = Image.new(
        "RGB", (panel_size[0] * len(entries), panel_size[1] + label_height), (12, 15, 20)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(entries):
        panel = helper.contained_panel(image, panel_size)
        sheet.paste(panel, (index * panel_size[0], 0))
        draw.text(
            (index * panel_size[0] + 8, panel_size[1] + 14),
            title,
            fill=(255, 255, 255),
        )
    helper.COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(helper.COMPARISON, optimize=True)


helper.verify_inputs = verify_inputs
helper.make_mask = make_mask
helper.exact_request_payload = exact_request_payload
helper.make_comparison = make_comparison


def main() -> None:
    prepare_only = "--prepare-only" in sys.argv
    helper.main()
    if prepare_only or not helper.METADATA.exists():
        return
    metadata = json.loads(helper.METADATA.read_text(encoding="utf-8"))
    metadata["pose"] = "sad"
    metadata["mask"]["bbox_xyxy"] = [372, 135, 397, 158]
    metadata["inputs"]["pose_tassel_source"] = {
        "path": str(POSE_TASSEL),
        "sha256": sha256(POSE_TASSEL),
    }
    metadata["request"]["driver_path"] = str(Path(__file__).resolve())
    metadata["request"]["driver_sha256"] = sha256(Path(__file__).resolve())
    metadata["request"]["shared_helper_path"] = str(HELPER_PATH)
    metadata["request"]["shared_helper_sha256"] = sha256(HELPER_PATH)
    helper.METADATA.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
