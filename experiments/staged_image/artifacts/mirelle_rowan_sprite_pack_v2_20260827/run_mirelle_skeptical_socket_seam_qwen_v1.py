#!/usr/bin/env python3
"""Run exactly one bounded Qwen socket-seam experiment for Mirelle skeptical v1."""

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
    ROOT / "grafts/mirelle_voss/skeptical_primary_metal_reference_graft_v1.png"
)
helper.REFERENCE = (
    ROOT
    / "supporting_refs/"
    "mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
helper.METAL_PATCH = (
    ROOT / "masks/mirelle_voss/skeptical_canonical_metal_patch_mask_v1.png"
)
POSE_TASSEL_CLOTH = (
    ROOT / "masks/mirelle_voss/skeptical_pose_tassel_cloth_preserve_mask_v1.png"
)
SHAFT = ROOT / "masks/mirelle_voss/skeptical_shaft_preserve_mask_v1.png"
POSE_OCCLUSION = (
    ROOT / "masks/mirelle_voss/skeptical_pose_occlusion_restore_mask_v1.png"
)
CHARACTER = (
    ROOT / "masks/mirelle_voss/skeptical_hands_body_face_costume_exact_mask_v1.png"
)
OPPOSITE_CAP_BOOT = (
    ROOT / "masks/mirelle_voss/skeptical_opposite_butt_cap_boot_exact_mask_v1.png"
)
helper.MASK = (
    ROOT / "masks/mirelle_voss/skeptical_socket_background_seam_qwen_mask_v1.png"
)
helper.MASK_PROOF = (
    ROOT / "mask_proofs/mirelle_skeptical_socket_seam_qwen_mask_v1.png"
)
helper.PROMPT_PATH = (
    ROOT / "prompts/qwen/mirelle_skeptical_socket_seam_qwen_v1.txt"
)
helper.REQUEST_BODY = (
    ROOT / "qwen_requests/mirelle_skeptical_socket_seam_qwen_v1.json"
)
helper.PREFLIGHT_HEALTH = (
    ROOT / "qwen_requests/mirelle_skeptical_socket_seam_qwen_v1_health.json"
)
helper.RAW_OUTPUT = (
    ROOT / "qwen_raw/mirelle_voss/skeptical_socket_seam_qwen_v1_model_output.png"
)
helper.ERROR_BODY = (
    ROOT / "qwen_raw/mirelle_voss/skeptical_socket_seam_qwen_v1_error.bin"
)
helper.ALIGNED_OUTPUT = (
    ROOT / "qwen_aligned/mirelle_voss/skeptical_socket_seam_qwen_v1_aligned.png"
)
helper.COMPOSITE = (
    ROOT / "qwen_composited/mirelle_voss/skeptical_socket_seam_qwen_v1.png"
)
helper.HEADERS = (
    ROOT / "qwen_headers/mirelle_skeptical_socket_seam_qwen_v1.json"
)
helper.METADATA = (
    ROOT / "qwen_metadata/mirelle_skeptical_socket_seam_qwen_v1.json"
)
helper.COMPARISON = (
    ROOT / "component_proofs/mirelle_skeptical_socket_seam_qwen_v1_comparison.png"
)
helper.SEED = 260827334
helper.STEPS = 24
helper.CFG = 3.2
helper.DENOISE = 0.20
helper.EXPECTED_PRIMARY_SHA = (
    "7b827787a0dad57e5e937936559871beaddcfbd7919a9dfab63c1459fb850fb5"
)
helper.EXPECTED_REFERENCE_SHA = (
    "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff"
)
helper.EXPECTED_PATCH_SHA = (
    "af1771920c639851689e314963d82d39a54fb0caa6da90054bf4c960d996c223"
)
helper.PROMPT = """Use case: precise-object-edit
Asset type: one-off masked seam experiment for a visual-novel character sprite
Input roles: Image 1 is the approved deterministic Skeptical-pose v1 edit target. Image 2 is Mirelle's locked canonical spear crop and is supporting evidence only for the existing silver socket material and clean anime edge treatment.
Primary request: Make the smallest possible cleanup only at the existing silver socket/key-to-magenta-background junction inside the white mask. Remove any pasted-looking magenta notch, fringe, or jagged transition there. Preserve the supplied socket size, position, line weight, silver shading, silhouette, and connection to the existing red shaft.
Preservation contract: Change nothing outside the white mask. Do not redraw, move, resize, simplify, or restyle the central blade, either lateral guard, original red tassel or cloth, red shaft, opposite butt cap, or boot. Preserve Mirelle's skeptical face, orange-red hair and braid, hands, body, pose, cream coat, burgundy costume, armor, framing, and flat saturated magenta screen exactly. Do not copy Image 2's orientation, streamers, body, or background. Add no weapon parts, streamers, people, text, scenery, shadow, or texture.
Mask semantics: White is the only eligible socket/key seam region. Black must remain unchanged."""

base_verify_inputs = helper.verify_inputs


def verify_inputs() -> None:
    actual_helper_sha = sha256(HELPER_PATH)
    if actual_helper_sha != EXPECTED_HELPER_SHA:
        raise RuntimeError(
            f"helper hash fence failed: expected {EXPECTED_HELPER_SHA}, got {actual_helper_sha}"
        )
    base_verify_inputs()
    extra_expected = {
        POSE_TASSEL_CLOTH: "26d0393902d0753f91abf511de159dc4a2bb208591df1ed3776a1a73b3c4e436",
        SHAFT: "ffccdd8d8fa56125298d8a680923eeea0456216ced0918bd530a63882cba09ee",
        POSE_OCCLUSION: "cbf2f4b4cfb8fb216f6dea2be4eae05e3a1b92fdec2c76956277be59da1951ef",
        CHARACTER: "a65a32aff5c299c189b0173f6181993ff454bfb115cbb06f2952d2e92347adce",
        OPPOSITE_CAP_BOOT: "f9e70a2f291f7d5de38016e01bcb82e65716fedc3706c387fadf2b726ab99e81",
    }
    for path, expected in extra_expected.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"skeptical preservation-mask hash fence failed for {path}: "
                f"expected {expected}, got {actual}"
            )


def make_mask() -> int:
    primary = np.asarray(Image.open(helper.PRIMARY).convert("RGB"), dtype=np.uint8)
    patch = np.asarray(
        Image.open(helper.METAL_PATCH).convert("L"), dtype=np.uint8
    ) > 0
    pose_tassel_cloth = np.asarray(
        Image.open(POSE_TASSEL_CLOTH).convert("L"), dtype=np.uint8
    ) > 0
    shaft = np.asarray(Image.open(SHAFT).convert("L"), dtype=np.uint8) > 0
    pose_occlusion = np.asarray(
        Image.open(POSE_OCCLUSION).convert("L"), dtype=np.uint8
    ) > 0
    character = np.asarray(
        Image.open(CHARACTER).convert("L"), dtype=np.uint8
    ) > 0
    opposite_cap_boot = np.asarray(
        Image.open(OPPOSITE_CAP_BOOT).convert("L"), dtype=np.uint8
    ) > 0

    # The central lower portion of the transformed metal patch is the visible
    # socket/key. Everything else in the patch is the blade or lateral guards.
    socket_zone = np.zeros_like(patch)
    socket_zone[168:185, 289:312] = True
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
    protected_blade_and_guards &= ~socket

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
    # The original shaft mask overlaps the graft footprint because the
    # deterministic compositor restored pose pixels over the socket. Semantic
    # shaft protection is therefore the authored shaft outside canonical metal
    # plus every visible red/burgundy shaft pixel, dilated five pixels.
    visible_shaft = (shaft & ~patch) | red
    protected_visible_shaft = cv2.dilate(
        visible_shaft.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    protected = (
        protected_blade_and_guards
        | protected_visible_shaft
        | pose_tassel_cloth
        | character
        | opposite_cap_boot
    )
    mask = seam_ring & ~protected

    # Keep only the connected central seam; detached flecks beside guard tips
    # are not part of the authorized socket/key junction.
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if component_count < 2:
        raise RuntimeError("skeptical seam mask has no connected foreground component")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == largest

    if int(mask.sum()) != 174:
        raise RuntimeError(f"unexpected skeptical tiny seam mask size: {int(mask.sum())}")
    ys, xs = np.where(mask)
    actual_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if actual_bbox != [288, 167, 313, 181]:
        raise RuntimeError(f"skeptical seam mask escaped authored zone: {actual_bbox}")
    semantic_shaft = protected_visible_shaft | pose_occlusion
    overlaps = {
        "blade_guards": int(np.count_nonzero(mask & protected_blade_and_guards)),
        "visible_shaft": int(np.count_nonzero(mask & semantic_shaft)),
        "pose_tassel_cloth": int(np.count_nonzero(mask & pose_tassel_cloth)),
        "character": int(np.count_nonzero(mask & character)),
        "opposite_cap_boot": int(np.count_nonzero(mask & opposite_cap_boot)),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"skeptical seam mask overlaps protected content: {overlaps}")

    helper.MASK.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        helper.MASK, optimize=True
    )
    make_mask_proof(primary, mask)
    return int(mask.sum())


def make_mask_proof(primary: np.ndarray, mask: np.ndarray) -> None:
    crop = (255, 125, 340, 215)
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
        ("APPROVED SKEPTICAL V1", base_crop),
        ("174PX SOCKET/KEY MASK", overlay_crop),
        ("BINARY MASK", binary_crop),
    ]
    panel_size = (425, 450)
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
        "filename_prefix": "mirelle_skeptical_socket_seam_qwen_v1",
    }


def make_comparison(
    primary: np.ndarray,
    raw_resized: np.ndarray,
    aligned: np.ndarray,
    composite: np.ndarray,
) -> None:
    crop = (255, 125, 340, 215)
    entries = [
        ("APPROVED SKEPTICAL V1", Image.fromarray(primary, mode="RGB").crop(crop)),
        ("RAW RESIZED", Image.fromarray(raw_resized, mode="RGB").crop(crop)),
        ("RAW ALIGNED", Image.fromarray(aligned, mode="RGB").crop(crop)),
        ("HARD-MASK COMPOSITE", Image.fromarray(composite, mode="RGB").crop(crop)),
    ]
    panel_size = (340, 360)
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
    metadata["pose"] = "skeptical"
    metadata["version"] = "primary_metal_reference_graft_v1"
    metadata["selection"] = "approved deterministic skeptical v1 remains selected fallback"
    metadata["mask"]["bbox_xyxy"] = [288, 167, 313, 181]
    metadata["mask"]["socket_material_pixels_inside"] = 91
    metadata["mask"]["background_seam_pixels_inside"] = 83
    metadata["mask"]["blade_guard_tassel_shaft_character_overlap_pixels"] = 0
    metadata["inputs"]["pose_tassel_cloth_source"] = {
        "path": str(POSE_TASSEL_CLOTH),
        "sha256": sha256(POSE_TASSEL_CLOTH),
    }
    metadata["inputs"]["shaft_source"] = {
        "path": str(SHAFT),
        "sha256": sha256(SHAFT),
    }
    metadata["inputs"]["pose_occlusion_source"] = {
        "path": str(POSE_OCCLUSION),
        "sha256": sha256(POSE_OCCLUSION),
    }
    metadata["inputs"]["character_source"] = {
        "path": str(CHARACTER),
        "sha256": sha256(CHARACTER),
    }
    metadata["inputs"]["opposite_cap_boot_source"] = {
        "path": str(OPPOSITE_CAP_BOOT),
        "sha256": sha256(OPPOSITE_CAP_BOOT),
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
