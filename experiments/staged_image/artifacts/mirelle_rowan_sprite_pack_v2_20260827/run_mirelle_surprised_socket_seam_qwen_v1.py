#!/usr/bin/env python3
"""Run exactly one authorized seam-only Qwen edit for Mirelle's surprised sprite."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
GATEWAY = "http://127.0.0.1:8199"
ENDPOINT = "/prototype/edit/qwen/masked"

PRIMARY = ROOT / "grafts/mirelle_voss/surprised_primary_metal_reference_graft_v1.png"
REFERENCE = (
    ROOT
    / "supporting_refs/"
    "mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
METAL = ROOT / "components/mirelle_voss/canonical_primary_metal_rgba_v1.png"
GRAFT_METADATA = (
    ROOT / "component_metadata/mirelle_surprised_primary_metal_graft_v1.json"
)
METAL_PATCH = (
    ROOT / "masks/mirelle_voss/surprised_canonical_metal_patch_mask_v1.png"
)
POSE_CLOTH = (
    ROOT
    / "masks/mirelle_voss/surprised_upper_knot_tassel_cloth_exact_mask_v1.png"
)
SHAFT = ROOT / "masks/mirelle_voss/surprised_shaft_exact_mask_v1.png"
GRIP = ROOT / "masks/mirelle_voss/surprised_gripping_hand_exact_mask_v1.png"
LOWER = (
    ROOT
    / "masks/mirelle_voss/surprised_lower_butt_cap_cape_leg_exact_mask_v1.png"
)
BODY = ROOT / "masks/mirelle_voss/surprised_body_face_costume_exact_mask_v1.png"

MASK = ROOT / "masks/mirelle_voss/surprised_socket_seam_qwen_mask_v1.png"
MASK_PROOF = ROOT / "mask_proofs/mirelle_surprised_socket_seam_qwen_mask_v1.png"
PROMPT_PATH = ROOT / "prompts/qwen/mirelle_surprised_socket_seam_qwen_v1.txt"
REQUEST_BODY = ROOT / "qwen_requests/mirelle_surprised_socket_seam_qwen_v1.json"
REQUEST_HEADERS = (
    ROOT
    / "qwen_requests/mirelle_surprised_socket_seam_qwen_request_headers_v1.json"
)
HEALTH_BEFORE_CALL = (
    ROOT
    / "qwen_requests/mirelle_surprised_socket_seam_qwen_health_before_call_v1.json"
)
ATTEMPT = ROOT / "qwen_requests/mirelle_surprised_socket_seam_qwen_attempt_v1.json"
RAW_OUTPUT = (
    ROOT
    / "qwen_raw/mirelle_voss/surprised_socket_seam_qwen_v1_model_output.png"
)
ALIGNED_OUTPUT = (
    ROOT
    / "qwen_aligned/mirelle_voss/"
    "surprised_socket_seam_qwen_v1_model_output_aligned.png"
)
COMPOSITE = (
    ROOT / "qwen_composited/mirelle_voss/surprised_socket_seam_qwen_v1.png"
)
RESPONSE_HEADERS = (
    ROOT / "qwen_headers/mirelle_surprised_socket_seam_qwen_response_v1.json"
)
REVIEW_PROOF = ROOT / "qwen_proofs/mirelle_surprised_socket_seam_qwen_v1.png"
METADATA = ROOT / "qwen_metadata/mirelle_surprised_socket_seam_qwen_v1.json"

EXPECTED_HASHES = {
    PRIMARY: "1a729f2df227282599fb09225dd71d14e666cd6eefb131366dea4dd462632a33",
    REFERENCE: "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff",
    METAL: "4f4b369d7f445b79a44673b33aee8bcc763f754eb37a0d7bfaee38213bf73b6f",
    GRAFT_METADATA: "91bf97dba1427788f70f07f7a1f3e81c05a32ced1357c56e81440a3438e4ea1a",
    METAL_PATCH: "979a8c0dec398c5b6fe1cb45d94e71399152ffe28e0b5848c789c0c6ed0b244b",
    POSE_CLOTH: "311cbd4d61cf5ccf1c0687e0b85be6956f4fa7494daa93b183e1fd312741210c",
    SHAFT: "cedc0320353ef2e398369b59a730406df39d01db45947de38214a5433afcbd6f",
    GRIP: "3ff4513497317abe305006e2338a00fdcd1b2dc3ab6ac024d32dc3f4fa7c8705",
    LOWER: "21e982e2e177ea1b4200346fa498438842e7e839562d3f0ddafd503222abcbcc",
    BODY: "2582e4f2debe65285c008638ffd5c868c3fec22c4a7e21b6bf3681acc67fc213",
}

SEED = 260827733
STEPS = 26
CFG = 3.5
DENOISE = 0.25
EXPECTED_MASK_PIXELS = 47

PROMPT = """Use case: precise-object-edit
Asset type: one-call visual-novel sprite seam experiment
Input roles: Image 1 is the root-approved deterministic surprised sprite and is the only edit target. Image 2 is Mirelle's locked canonical spear crop and is supporting evidence for the existing metal rendering only; do not copy its pose, body, ribbons, background, scale, or framing.
Primary request: Harmonize only the exposed silver collar/socket key junction and its immediately adjacent uniform magenta seam inside the supplied tiny white mask. Make this tiny junction read as one clean connection between the already-present canonical metal head and the already-present red shaft. Remove only tiny magenta or edge slivers inside the mask, preserve the existing anime line weight and silver shading, and restore clean saturated magenta wherever the seam should reveal background.
Hard prop invariants: Do not redraw, resize, rotate, simplify, replace, or restyle the spear. The long central blade, both lateral guards, the red knot and tassel cloth, the shaft, the gripping hand, and the lower butt cap are deliberately outside the mask and must stay exact. Do not add, remove, lengthen, or move any blade, guard, shaft, tassel, ribbon, streamer, hand, or cap.
Character invariants: Preserve Mirelle's face, surprised expression, orange-red hair and braid, hands, body, pose, cream coat, burgundy garments, armor, boots, silhouette, framing, and background. Preserve the original red knot/tassel/cloth and shaft exactly. Do not add a person, prop, text, logo, watermark, scenery, floor, shadow, or texture.
Mask semantics: White is the only eligible edit region. Every black pixel must remain unchanged."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def check_hash_fences() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash fence failed for {path}: {actual} != {expected}")


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if not xs.size:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def prepared_geometry() -> dict[str, np.ndarray | list[list[float]]]:
    primary = np.asarray(Image.open(PRIMARY).convert("RGB"), dtype=np.uint8)
    metal = np.asarray(Image.open(METAL).convert("RGBA"), dtype=np.uint8)
    if tuple(primary.shape) != (1453, 1082, 3):
        raise RuntimeError(f"unexpected primary shape: {primary.shape}")
    if tuple(metal.shape) != (314, 312, 4):
        raise RuntimeError(f"unexpected component shape: {metal.shape}")

    graft_metadata = json.loads(GRAFT_METADATA.read_text(encoding="utf-8"))
    matrix = np.asarray(graft_metadata["transform"]["matrix"], dtype=np.float64)
    source_socket_zone = Image.new("L", (metal.shape[1], metal.shape[0]), 0)
    ImageDraw.Draw(source_socket_zone).polygon(
        [
            (190, 65),
            (224, 65),
            (232, 83),
            (225, 104),
            (210, 116),
            (188, 108),
            (179, 92),
        ],
        fill=255,
    )
    source_socket = (
        (np.asarray(source_socket_zone, dtype=np.uint8) > 0)
        & (metal[..., 3] >= 48)
    )
    height, width = primary.shape[:2]
    transformed_socket = cv2.warpAffine(
        source_socket.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    metal_patch = load_mask(METAL_PATCH)
    cloth = load_mask(POSE_CLOTH)
    shaft = load_mask(SHAFT)
    grip = load_mask(GRIP)
    lower = load_mask(LOWER)
    body = load_mask(BODY)
    outer_socket_seam = cv2.dilate(
        transformed_socket.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ) > 0
    blade_and_guards = metal_patch & ~transformed_socket
    protected_blade_and_guards = cv2.dilate(
        blade_and_guards.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    protected_cloth_shaft = cv2.dilate(
        (cloth | shaft).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    yy, xx = np.mgrid[:height, :width]
    tiny_roi = (xx >= 72) & (xx <= 120) & (yy >= 135) & (yy <= 184)
    mask = (
        outer_socket_seam
        & tiny_roi
        & ~protected_blade_and_guards
        & ~protected_cloth_shaft
        & ~grip
        & ~lower
        & ~body
    )
    return {
        "primary": primary,
        "matrix": matrix.tolist(),
        "transformed_socket": transformed_socket,
        "metal_patch": metal_patch,
        "cloth": cloth,
        "shaft": shaft,
        "grip": grip,
        "lower": lower,
        "body": body,
        "protected_blade_and_guards": protected_blade_and_guards,
        "protected_cloth_shaft": protected_cloth_shaft,
        "mask": mask,
    }


def prepare_mask() -> dict[str, object]:
    geometry = prepared_geometry()
    primary = geometry["primary"]
    mask = geometry["mask"]
    assert isinstance(primary, np.ndarray)
    assert isinstance(mask, np.ndarray)
    if not np.any(mask):
        raise RuntimeError("seam mask is empty")

    protected = (
        geometry["protected_blade_and_guards"]
        | geometry["protected_cloth_shaft"]
        | geometry["grip"]
        | geometry["lower"]
        | geometry["body"]
    )
    assert isinstance(protected, np.ndarray)
    forbidden_overlap = int(np.count_nonzero(mask & protected))
    if forbidden_overlap:
        raise RuntimeError(f"mask includes {forbidden_overlap} protected pixels")
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels != EXPECTED_MASK_PIXELS:
        raise RuntimeError(
            f"unexpected seam-mask count: {mask_pixels} != {EXPECTED_MASK_PIXELS}"
        )

    MASK.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        MASK, optimize=True
    )
    close_crop = (20, 20, 180, 240)
    tinted = primary.copy()
    tinted[mask] = np.clip(
        np.round(
            tinted[mask].astype(np.float32) * 0.20
            + np.array([0, 255, 255]) * 0.80
        ),
        0,
        255,
    ).astype(np.uint8)
    panel_size = (640, 880)
    proof = Image.new("RGB", (1280, 930), (12, 15, 20))
    draw = ImageDraw.Draw(proof)
    for index, (title, array) in enumerate(
        (
            ("ROOT-APPROVED SURPRISED V1", primary),
            (f"{mask_pixels}PX SOCKET-KEY MASK", tinted),
        )
    ):
        panel = Image.fromarray(array, mode="RGB").crop(close_crop).resize(
            panel_size, Image.Resampling.NEAREST
        )
        proof.paste(panel, (index * 640, 0))
        draw.text((index * 640 + 10, 896), title, fill=(255, 255, 255))
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    proof.save(MASK_PROOF, optimize=True)

    transformed_socket = geometry["transformed_socket"]
    metal_patch = geometry["metal_patch"]
    assert isinstance(transformed_socket, np.ndarray)
    assert isinstance(metal_patch, np.ndarray)
    return {
        "pixel_count": mask_pixels,
        "bbox_xyxy_exclusive": bbox(mask),
        "socket_material_pixels_inside": int(
            np.count_nonzero(mask & transformed_socket)
        ),
        "background_seam_pixels_inside": int(np.count_nonzero(mask & ~metal_patch)),
        "blade_guard_protected_overlap": int(
            np.count_nonzero(mask & geometry["protected_blade_and_guards"])
        ),
        "cloth_shaft_protected_overlap": int(
            np.count_nonzero(mask & geometry["protected_cloth_shaft"])
        ),
        "grip_protected_overlap": int(np.count_nonzero(mask & geometry["grip"])),
        "lower_protected_overlap": int(np.count_nonzero(mask & geometry["lower"])),
        "body_protected_overlap": int(np.count_nonzero(mask & geometry["body"])),
        "sha256": sha256(MASK),
        "proof_sha256": sha256(MASK_PROOF),
        "transform_matrix": geometry["matrix"],
        "construction": {
            "socket_dilation_kernel": "7x7 ellipse",
            "blade_guard_protection_dilation": "3x3 ellipse",
            "cloth_shaft_protection_dilation": "5x5 ellipse",
            "tiny_roi_xyxy_inclusive": [72, 135, 120, 184],
        },
    }


def make_payload() -> dict[str, object]:
    return {
        "prompt": PROMPT,
        "image_base64": encode(PRIMARY),
        "image2_base64": encode(REFERENCE),
        "mask_base64": encode(MASK),
        "seed": SEED,
        "steps": STEPS,
        "cfg": CFG,
        "denoise": DENOISE,
        "filename_prefix": "mirelle_surprised_socket_seam_qwen_v1",
    }


def health() -> dict[str, object]:
    request = urllib.request.Request(f"{GATEWAY}/health", method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read())
    edit = data.get("pipelines", {}).get("edit", {})
    if not edit.get("available") or "Qwen" not in str(edit.get("model")):
        raise RuntimeError(f"Qwen edit pipeline unavailable: {edit}")
    if data.get("mode") != "qwen":
        raise RuntimeError(f"gateway is not in qwen mode: {data.get('mode')}")
    return data


def boundary_jump_metrics(
    base: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    base_jumps: list[np.ndarray] = []
    candidate_jumps: list[np.ndarray] = []
    edge_count = 0
    for dy, dx in ((0, 1), (1, 0)):
        if dy:
            inside_a = mask[:-1, :]
            inside_b = mask[1:, :]
            base_a, base_b = base[:-1, :], base[1:, :]
            cand_a, cand_b = candidate[:-1, :], candidate[1:, :]
        else:
            inside_a = mask[:, :-1]
            inside_b = mask[:, 1:]
            base_a, base_b = base[:, :-1], base[:, 1:]
            cand_a, cand_b = candidate[:, :-1], candidate[:, 1:]
        boundary = inside_a != inside_b
        edge_count += int(np.count_nonzero(boundary))
        base_delta = base_a.astype(np.float32) - base_b.astype(np.float32)
        cand_delta = cand_a.astype(np.float32) - cand_b.astype(np.float32)
        base_jumps.append(np.linalg.norm(base_delta[boundary], axis=1))
        candidate_jumps.append(np.linalg.norm(cand_delta[boundary], axis=1))

    base_values = np.concatenate(base_jumps)
    candidate_values = np.concatenate(candidate_jumps)

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_rgb_l2": float(np.mean(values)),
            "median_rgb_l2": float(np.median(values)),
            "p95_rgb_l2": float(np.percentile(values, 95)),
            "max_rgb_l2": float(np.max(values)),
        }

    return {
        "four_connected_boundary_edge_count": edge_count,
        "approved_deterministic": summarize(base_values),
        "qwen_hard_composite": summarize(candidate_values),
        "interpretation": (
            "Lower cross-boundary RGB jumps can indicate a smoother seam, but visual "
            "inspection controls acceptance because legitimate linework also raises them."
        ),
    }


def hard_composite() -> dict[str, object]:
    primary_image = Image.open(PRIMARY).convert("RGB")
    raw_image = Image.open(RAW_OUTPUT)
    raw_format = raw_image.format
    raw_mode = raw_image.mode
    raw_size = raw_image.size
    aligned = raw_image.convert("RGB").resize(
        primary_image.size, Image.Resampling.LANCZOS
    )
    ALIGNED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    aligned.save(ALIGNED_OUTPUT, format="PNG", optimize=True)

    primary = np.asarray(primary_image, dtype=np.uint8).copy()
    aligned_pixels = np.asarray(aligned, dtype=np.uint8)
    mask = load_mask(MASK)
    composite = primary.copy()
    composite[mask] = aligned_pixels[mask]
    changed = np.any(composite != primary, axis=2)
    outside_delta = int(np.count_nonzero(changed & ~mask))
    if outside_delta:
        raise RuntimeError(f"hard composite changed {outside_delta} outside-mask pixels")

    geometry = prepared_geometry()
    protected_delta: dict[str, int] = {}
    for name, protected in (
        ("blade_and_guards", geometry["protected_blade_and_guards"]),
        ("knot_tassel_cloth_and_shaft", geometry["protected_cloth_shaft"]),
        ("gripping_hand", geometry["grip"]),
        ("lower_cap_cape_leg", geometry["lower"]),
        ("body_face_costume", geometry["body"]),
    ):
        assert isinstance(protected, np.ndarray)
        protected_delta[name] = int(np.count_nonzero(changed & protected))
    if any(protected_delta.values()):
        raise RuntimeError(f"protected pixels changed: {protected_delta}")

    COMPOSITE.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite, mode="RGB").save(COMPOSITE, optimize=True)
    return {
        "raw_format": raw_format,
        "raw_mode": raw_mode,
        "raw_dimensions": list(raw_size),
        "aligned_dimensions": list(aligned.size),
        "resampling": "Pillow LANCZOS from raw model dimensions to primary dimensions",
        "composite_method": "hard binary replacement where saved mask >= 128",
        "changed_inside_mask_pixels": int(np.count_nonzero(changed & mask)),
        "changed_outside_mask_pixels": outside_delta,
        "protected_delta_pixels": protected_delta,
        "boundary_jump_metrics": boundary_jump_metrics(primary, composite, mask),
    }


def make_review_proof() -> None:
    crop = (20, 20, 180, 240)
    paths = (
        ("APPROVED V1 FALLBACK", PRIMARY),
        ("QWEN HARD COMPOSITE", COMPOSITE),
        ("ALIGNED RAW MODEL", ALIGNED_OUTPUT),
    )
    sheet = Image.new("RGB", (1920, 930), (12, 15, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (title, path) in enumerate(paths):
        panel = Image.open(path).convert("RGB").crop(crop).resize(
            (640, 880), Image.Resampling.NEAREST
        )
        sheet.paste(panel, (index * 640, 0))
        draw.text((index * 640 + 10, 896), title, fill=(255, 255, 255))
    REVIEW_PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(REVIEW_PROOF, optimize=True)


def execute_once(mask_record: dict[str, object]) -> None:
    if ATTEMPT.exists() or RAW_OUTPUT.exists() or METADATA.exists():
        raise RuntimeError(
            "authorized Qwen attempt already exists; refusing a second model call"
        )
    health_record = health()
    write_json(HEALTH_BEFORE_CALL, health_record)
    payload = make_payload()
    request_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    REQUEST_BODY.parent.mkdir(parents=True, exist_ok=True)
    REQUEST_BODY.write_bytes(request_bytes)
    request_headers = {"Content-Type": "application/json"}
    write_json(REQUEST_HEADERS, request_headers)
    attempt_record: dict[str, object] = {
        "status": "dispatching_single_authorized_call",
        "started_at": datetime.now(UTC).isoformat(),
        "gateway": GATEWAY,
        "endpoint": ENDPOINT,
        "request_body_path": str(REQUEST_BODY),
        "request_body_sha256": sha256(REQUEST_BODY),
        "request_headers_path": str(REQUEST_HEADERS),
        "request_headers_sha256": sha256(REQUEST_HEADERS),
        "second_call_permitted": False,
    }
    write_json(ATTEMPT, attempt_record)

    request = urllib.request.Request(
        f"{GATEWAY}{ENDPOINT}",
        data=request_bytes,
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            content = response.read()
            response_record = {
                "status_code": response.status,
                "reason": response.reason,
                "url": response.url,
                "headers": dict(response.headers.items()),
                "received_byte_count": len(content),
            }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        attempt_record.update(
            {
                "status": "single_authorized_call_failed_no_retry_allowed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(ATTEMPT, attempt_record)
        raise

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.write_bytes(content)
    response_record["raw_response_path"] = str(RAW_OUTPUT)
    response_record["raw_response_sha256"] = sha256(RAW_OUTPUT)
    write_json(RESPONSE_HEADERS, response_record)
    composite_record = hard_composite()
    make_review_proof()
    attempt_record.update(
        {
            "status": "single_authorized_call_completed",
            "finished_at": datetime.now(UTC).isoformat(),
            "response_sha256": sha256(RAW_OUTPUT),
            "second_call_permitted": False,
        }
    )
    write_json(ATTEMPT, attempt_record)

    metadata = {
        "status": "pending_manual_review",
        "selection": {
            "selected": str(PRIMARY),
            "selected_sha256": sha256(PRIMARY),
            "reason": (
                "root-approved deterministic v1 remains selected unless Qwen is "
                "visibly better at original resolution"
            ),
        },
        "model_calls": 1,
        "second_call_permitted": False,
        "inputs": {
            "approved_primary": {"path": str(PRIMARY), "sha256": sha256(PRIMARY)},
            "locked_canonical_crop": {
                "path": str(REFERENCE),
                "sha256": sha256(REFERENCE),
            },
            "frozen_metal_component": {
                "path": str(METAL),
                "sha256": sha256(METAL),
            },
            "mask": {"path": str(MASK), **mask_record},
            "prompt": {"path": str(PROMPT_PATH), "sha256": sha256(PROMPT_PATH)},
        },
        "exact_request": {
            "body_path": str(REQUEST_BODY),
            "body_sha256": sha256(REQUEST_BODY),
            "body_byte_count": REQUEST_BODY.stat().st_size,
            "headers_path": str(REQUEST_HEADERS),
            "headers_sha256": sha256(REQUEST_HEADERS),
            "endpoint": ENDPOINT,
            "seed": SEED,
            "steps": STEPS,
            "cfg": CFG,
            "denoise": DENOISE,
        },
        "exact_response": {
            "headers_path": str(RESPONSE_HEADERS),
            "headers_sha256": sha256(RESPONSE_HEADERS),
            "raw_model_output_path": str(RAW_OUTPUT),
            "raw_model_output_sha256": sha256(RAW_OUTPUT),
            "raw_model_output_byte_count": RAW_OUTPUT.stat().st_size,
        },
        "hard_composite": {
            **composite_record,
            "aligned_output_path": str(ALIGNED_OUTPUT),
            "aligned_output_sha256": sha256(ALIGNED_OUTPUT),
            "composite_path": str(COMPOSITE),
            "composite_sha256": sha256(COMPOSITE),
        },
        "proof": {"path": str(REVIEW_PROOF), "sha256": sha256(REVIEW_PROOF)},
        "attempt": {"path": str(ATTEMPT), "sha256": sha256(ATTEMPT)},
    }
    write_json(METADATA, metadata)
    print(json.dumps(metadata, indent=2))


def finalize_review(disposition: str, reason: str) -> None:
    if not METADATA.is_file():
        raise FileNotFoundError("no completed Qwen metadata to review")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("model_calls") != 1:
        raise RuntimeError("review finalization requires exactly one recorded model call")
    if disposition == "reject":
        selected = PRIMARY
        status = "manual_review_complete_qwen_rejected"
    else:
        selected = COMPOSITE
        status = "manual_review_complete_qwen_accepted"
    metadata["status"] = status
    metadata["selection"] = {
        "selected": str(selected),
        "selected_sha256": sha256(selected),
        "qwen_disposition": disposition,
        "reason": reason,
        "reviewed_at_original_resolution": True,
    }
    metadata["model_calls"] = 1
    metadata["second_call_permitted"] = False
    write_json(METADATA, metadata)
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--execute-authorized-once",
        action="store_true",
        help="send the one authorized Qwen request; absence means prepare-only",
    )
    group.add_argument(
        "--finalize-review",
        choices=("accept", "reject"),
        help="record manual disposition without sending any model request",
    )
    parser.add_argument(
        "--review-reason",
        help="required prose reason when --finalize-review is used",
    )
    args = parser.parse_args()
    check_hash_fences()
    mask_record = prepare_mask()
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_PATH.write_text(PROMPT + "\n", encoding="utf-8")
    if args.finalize_review:
        if not args.review_reason:
            parser.error("--review-reason is required with --finalize-review")
        finalize_review(args.finalize_review, args.review_reason)
        return
    if not args.execute_authorized_once:
        print(
            json.dumps(
                {"status": "prepared_no_model_call", "mask": mask_record},
                indent=2,
            )
        )
        return
    execute_once(mask_record)


if __name__ == "__main__":
    main()
