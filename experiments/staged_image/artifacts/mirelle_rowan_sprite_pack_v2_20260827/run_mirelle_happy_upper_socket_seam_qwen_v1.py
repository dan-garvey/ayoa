#!/usr/bin/env python3
"""Run exactly one bounded Qwen upper-socket seam experiment for Mirelle happy v2."""

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

helper.PRIMARY = (
    ROOT
    / "grafts/mirelle_voss/happy_upper_primary_metal_reference_graft_v2.png"
)
helper.REFERENCE = (
    ROOT
    / "supporting_refs/"
    "mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
helper.METAL_PATCH = (
    ROOT / "masks/mirelle_voss/happy_upper_canonical_metal_patch_mask_v2.png"
)
POSE_CLOTH = (
    ROOT / "masks/mirelle_voss/happy_upper_pose_cloth_preserve_mask_v2.png"
)
SHAFT = ROOT / "masks/mirelle_voss/happy_upper_shaft_preserve_mask_v2.png"
LOWER_CAP_BOOT = (
    ROOT / "masks/mirelle_voss/happy_lower_butt_cap_boot_exact_mask_v2.png"
)
helper.MASK = (
    ROOT
    / "masks/mirelle_voss/happy_upper_socket_background_seam_qwen_mask_v1.png"
)
helper.MASK_PROOF = (
    ROOT / "mask_proofs/mirelle_happy_upper_socket_seam_qwen_mask_v1.png"
)
helper.PROMPT_PATH = (
    ROOT / "prompts/qwen/mirelle_happy_upper_socket_seam_qwen_v1.txt"
)
helper.REQUEST_BODY = (
    ROOT / "qwen_requests/mirelle_happy_upper_socket_seam_qwen_v1.json"
)
helper.PREFLIGHT_HEALTH = (
    ROOT / "qwen_requests/mirelle_happy_upper_socket_seam_qwen_v1_health.json"
)
helper.RAW_OUTPUT = (
    ROOT
    / "qwen_raw/mirelle_voss/happy_upper_socket_seam_qwen_v1_model_output.png"
)
helper.ERROR_BODY = (
    ROOT / "qwen_raw/mirelle_voss/happy_upper_socket_seam_qwen_v1_error.bin"
)
helper.ALIGNED_OUTPUT = (
    ROOT
    / "qwen_aligned/mirelle_voss/happy_upper_socket_seam_qwen_v1_aligned.png"
)
helper.COMPOSITE = (
    ROOT / "qwen_composited/mirelle_voss/happy_upper_socket_seam_qwen_v1.png"
)
helper.HEADERS = (
    ROOT / "qwen_headers/mirelle_happy_upper_socket_seam_qwen_v1.json"
)
helper.METADATA = (
    ROOT / "qwen_metadata/mirelle_happy_upper_socket_seam_qwen_v1.json"
)
helper.COMPARISON = (
    ROOT / "component_proofs/mirelle_happy_upper_socket_seam_qwen_v1_comparison.png"
)
helper.SEED = 260827333
helper.STEPS = 24
helper.CFG = 3.2
helper.DENOISE = 0.20
helper.EXPECTED_PRIMARY_SHA = (
    "d84dc6d28c8864c3cb7e2dfb7190e5e6460e7b797d1b59dedb246341ca9f48a3"
)
helper.EXPECTED_REFERENCE_SHA = (
    "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff"
)
helper.EXPECTED_PATCH_SHA = (
    "0720fdc9010552132a6668a65f77f94aa1aafb5a24f1b55cb3cdde1ffd8b33c9"
)
helper.PROMPT = """Use case: precise-object-edit
Asset type: one-off masked seam experiment for a visual-novel character sprite
Input roles: Image 1 is the approved deterministic Happy-pose v2 edit target. Image 2 is Mirelle's locked canonical spear crop and is supporting evidence only for the silver socket material and clean anime edge treatment.
Primary request: Make the smallest possible cleanup only at the existing upper silver socket/key-to-magenta-background junction inside the white mask. Remove any pasted-looking magenta notch, fringe, or jagged transition there. Preserve the supplied socket size, position, line weight, silver shading, silhouette, and its existing connection to the red shaft.
Preservation contract: Change nothing outside the white mask. Do not redraw, move, resize, simplify, or restyle the central blade, either lateral guard, red tassel or cloth, red shaft, lower butt cap, or boot. Preserve Mirelle's happy face, orange-red hair and braid, hands, body, pose, cream coat, burgundy costume, armor, framing, and flat saturated magenta screen exactly. Do not copy Image 2's orientation, streamers, body, or background. Add no weapon parts, streamers, people, text, scenery, shadow, or texture.
Mask semantics: White is the only eligible upper-socket seam region. Black must remain unchanged."""

base_verify_inputs = helper.verify_inputs


def verify_inputs() -> None:
    actual_helper_sha = sha256(HELPER_PATH)
    if actual_helper_sha != EXPECTED_HELPER_SHA:
        raise RuntimeError(
            f"helper hash fence failed: expected {EXPECTED_HELPER_SHA}, got {actual_helper_sha}"
        )
    base_verify_inputs()
    extra_expected = {
        POSE_CLOTH: "433d5514577d1c725cef2db93729dd42095c1f01460be26dc56106fec5bd91a3",
        SHAFT: "bf0bdfa5c2a8ffa80b8ac012f5bf0bdd86a31d6bb2c598906ee450436dd53721",
        LOWER_CAP_BOOT: "8c4f146a3bb3e62a20b3d301410859887a16b7025a08cd26fdda0be940861741",
    }
    for path, expected in extra_expected.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"happy preservation-mask hash fence failed for {path}: "
                f"expected {expected}, got {actual}"
            )


def make_mask() -> int:
    primary = np.asarray(Image.open(helper.PRIMARY).convert("RGB"), dtype=np.uint8)
    patch = np.asarray(
        Image.open(helper.METAL_PATCH).convert("L"), dtype=np.uint8
    ) > 0
    pose_cloth = np.asarray(
        Image.open(POSE_CLOTH).convert("L"), dtype=np.uint8
    ) > 0
    shaft = np.asarray(Image.open(SHAFT).convert("L"), dtype=np.uint8) > 0
    lower_cap_boot = np.asarray(
        Image.open(LOWER_CAP_BOOT).convert("L"), dtype=np.uint8
    ) > 0

    # The central lower portion of the transformed metal patch is the visible
    # upper socket/key. The central blade and both lateral guards are everything
    # else in the patch and are protected before the seam ring is selected.
    socket_zone = np.zeros_like(patch)
    socket_zone[119:133, 228:247] = True
    socket = patch & socket_zone
    outer = cv2.dilate(
        socket.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    ) > 0
    inner = cv2.erode(
        socket.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ) > 0
    seam_ring = outer & ~inner
    blade_and_guards = patch & ~socket
    protected_blade_and_guards = cv2.dilate(
        blade_and_guards.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    # Dilation may touch the allowed socket; restore only that authored socket
    # subset while keeping all other transformed metal protected.
    protected_blade_and_guards &= ~socket

    body_face_costume = np.zeros_like(patch)
    body_face_costume[:, 260:] = True
    body_face_costume[180:, :] = True
    protected = (
        protected_blade_and_guards
        | pose_cloth
        | shaft
        | lower_cap_boot
        | body_face_costume
    )
    mask = seam_ring & ~protected

    # Keep only the connected collar/key seam. Two detached background flecks
    # sit beside the lateral guards and are outside this authorized edit target.
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if component_count < 2:
        raise RuntimeError("happy seam mask has no connected foreground component")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == largest

    if int(mask.sum()) != 126:
        raise RuntimeError(f"unexpected happy tiny seam mask size: {int(mask.sum())}")
    ys, xs = np.where(mask)
    actual_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if actual_bbox != [228, 119, 248, 129]:
        raise RuntimeError(f"happy seam mask escaped authored zone: {actual_bbox}")
    overlaps = {
        "blade_guards": int(np.count_nonzero(mask & protected_blade_and_guards)),
        "pose_cloth": int(np.count_nonzero(mask & pose_cloth)),
        "shaft": int(np.count_nonzero(mask & shaft)),
        "lower_cap_boot": int(np.count_nonzero(mask & lower_cap_boot)),
        "body_face_costume": int(np.count_nonzero(mask & body_face_costume)),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"happy seam mask overlaps protected content: {overlaps}")

    helper.MASK.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        helper.MASK, optimize=True
    )
    make_mask_proof(primary, mask)
    return int(mask.sum())


def make_mask_proof(primary: np.ndarray, mask: np.ndarray) -> None:
    crop = (205, 85, 270, 155)
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
        ("APPROVED HAPPY V2", base_crop),
        ("126PX UPPER-SOCKET MASK", overlay_crop),
        ("BINARY MASK", binary_crop),
    ]
    panel_size = (420, 455)
    label_height = 48
    sheet = Image.new(
        "RGB",
        (panel_size[0] * len(entries), panel_size[1] + label_height),
        (12, 15, 20),
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
        "filename_prefix": "mirelle_happy_upper_socket_seam_qwen_v1",
    }


def make_comparison(
    primary: np.ndarray,
    raw_resized: np.ndarray,
    aligned: np.ndarray,
    composite: np.ndarray,
) -> None:
    crop = (205, 85, 270, 155)
    entries = [
        ("APPROVED HAPPY V2", Image.fromarray(primary, mode="RGB").crop(crop)),
        ("RAW RESIZED", Image.fromarray(raw_resized, mode="RGB").crop(crop)),
        ("RAW ALIGNED", Image.fromarray(aligned, mode="RGB").crop(crop)),
        ("HARD-MASK COMPOSITE", Image.fromarray(composite, mode="RGB").crop(crop)),
    ]
    panel_size = (330, 355)
    label_height = 46
    sheet = Image.new(
        "RGB",
        (panel_size[0] * len(entries), panel_size[1] + label_height),
        (12, 15, 20),
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
    metadata["pose"] = "happy"
    metadata["version"] = "upper_primary_metal_reference_graft_v2"
    metadata["selection"] = "approved deterministic happy v2 remains selected fallback"
    metadata["mask"]["bbox_xyxy"] = [228, 119, 248, 129]
    metadata["mask"]["socket_material_pixels_inside"] = 73
    metadata["mask"]["background_seam_pixels_inside"] = 53
    metadata["mask"]["blade_guard_tassel_shaft_character_overlap_pixels"] = 0
    metadata["inputs"]["pose_cloth_source"] = {
        "path": str(POSE_CLOTH),
        "sha256": sha256(POSE_CLOTH),
    }
    metadata["inputs"]["shaft_source"] = {
        "path": str(SHAFT),
        "sha256": sha256(SHAFT),
    }
    metadata["inputs"]["lower_cap_boot_source"] = {
        "path": str(LOWER_CAP_BOOT),
        "sha256": sha256(LOWER_CAP_BOOT),
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
