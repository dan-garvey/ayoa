#!/usr/bin/env python3
"""Run exactly one authorized, seam-only Qwen edit for Mirelle's tense sprite."""

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

PRIMARY = ROOT / "grafts/mirelle_voss/tense_primary_metal_reference_graft_v2.png"
REFERENCE = (
    ROOT
    / "supporting_refs/"
    "mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
METAL = ROOT / "components/mirelle_voss/canonical_primary_metal_rgba_v1.png"
GRAFT_METADATA = ROOT / "component_metadata/mirelle_tense_metal_graft_v2.json"
METAL_PATCH = ROOT / "masks/mirelle_voss/tense_canonical_metal_patch_mask_v2.png"
POSE_CLOTH = ROOT / "masks/mirelle_voss/tense_pose_cloth_preserve_mask_v2.png"
SHAFT = ROOT / "masks/mirelle_voss/tense_shaft_preserve_mask_v2.png"
BODY = ROOT / "masks/mirelle_voss/tense_hand_hair_preserve_mask_v2.png"

MASK = ROOT / "masks/mirelle_voss/tense_socket_seam_qwen_mask_v1.png"
MASK_PROOF = ROOT / "mask_proofs/mirelle_tense_socket_seam_qwen_mask_v1.png"
PROMPT_PATH = ROOT / "prompts/qwen/mirelle_tense_socket_seam_qwen_v1.txt"
REQUEST_BODY = ROOT / "qwen_requests/mirelle_tense_socket_seam_qwen_v1.json"
HEALTH_BEFORE_CALL = (
    ROOT / "qwen_requests/mirelle_tense_socket_seam_qwen_health_before_call_v1.json"
)
ATTEMPT = ROOT / "qwen_requests/mirelle_tense_socket_seam_qwen_attempt_v1.json"
RAW_OUTPUT = (
    ROOT / "qwen_raw/mirelle_voss/tense_socket_seam_qwen_v1_model_output.png"
)
ALIGNED_OUTPUT = (
    ROOT
    / "qwen_aligned/mirelle_voss/tense_socket_seam_qwen_v1_model_output_aligned.png"
)
COMPOSITE = ROOT / "qwen_composited/mirelle_voss/tense_socket_seam_qwen_v1.png"
RESPONSE = ROOT / "qwen_headers/mirelle_tense_socket_seam_qwen_v1.json"
REVIEW_PROOF = ROOT / "qwen_proofs/mirelle_tense_socket_seam_qwen_v1.png"
METADATA = ROOT / "qwen_metadata/mirelle_tense_socket_seam_qwen_v1.json"

EXPECTED_HASHES = {
    PRIMARY: "d1a623ef7fb95e28a890fe7c0c4e235f02da7a6cfbbe26c60f3e13c1cb5a54cc",
    REFERENCE: "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff",
    METAL: "4f4b369d7f445b79a44673b33aee8bcc763f754eb37a0d7bfaee38213bf73b6f",
    GRAFT_METADATA: "3a1620ce3f6c2e2635156bb29d8ecef2e0394190b17ecf06620238a642deb0cb",
    METAL_PATCH: "5bb61ababfbc61e8a49aeb1f386a0637fdd8a55ea47acbacc2c39328e0eb6f97",
    POSE_CLOTH: "47f21cea83fcd728e14d927c6fa5f9ee7a01921a592911ab8757b7dcfde51169",
    SHAFT: "c1147b61f0e59e05b91ab0d4f40accae8990461ada27e345e277e9384d21cb12",
    BODY: "3d59983f23d61b7d2b8391c3c8caef2742791d4bcaf8a5a1e6584959abd700cf",
}

SEED = 260827329
STEPS = 26
CFG = 3.5
DENOISE = 0.25

PROMPT = """Use case: precise-object-edit
Asset type: one-call visual-novel sprite seam experiment
Input roles: Image 1 is the approved deterministic tense sprite and is the only edit target. Image 2 is Mirelle's locked canonical spear crop and is supporting evidence for the existing metal rendering only; do not copy its pose, body, ribbons, background, scale, or framing.
Primary request: Harmonize only the existing silver collar/socket junction and its immediately adjacent uniform magenta-background seam inside the supplied white mask. Make the socket read as one clean connection between the already-present canonical metal head and the already-present red shaft. Remove only tiny magenta/edge slivers inside the mask, preserve the existing anime line weight and silver shading, and restore clean saturated magenta wherever the seam should reveal background.
Hard prop invariants: Do not redraw, resize, rotate, simplify, replace, or restyle the spear. The long central blade and both lateral guards are deliberately outside the mask and must stay exact. Do not add, remove, lengthen, or move any blade, guard, shaft, tassel, ribbon, or streamer.
Character invariants: Preserve Mirelle's face, tense expression, orange-red hair and braid, hands, body, pose, cream coat, burgundy garments, armor, boots, silhouette, framing, and background. Preserve the original red tassel/cloth and shaft exactly. Do not add a person, prop, text, logo, watermark, scenery, floor, shadow, or texture.
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


def prepare_mask() -> dict[str, object]:
    primary = np.asarray(Image.open(PRIMARY).convert("RGB"), dtype=np.uint8)
    metal = np.asarray(Image.open(METAL).convert("RGBA"), dtype=np.uint8)
    if tuple(primary.shape) != (1456, 1080, 3):
        raise RuntimeError(f"unexpected primary shape: {primary.shape}")
    if tuple(metal.shape) != (314, 312, 4):
        raise RuntimeError(f"unexpected component shape: {metal.shape}")

    graft_metadata = json.loads(GRAFT_METADATA.read_text(encoding="utf-8"))
    matrix = np.asarray(graft_metadata["transform"]["matrix"], dtype=np.float64)
    source_socket_zone = Image.new("L", (metal.shape[1], metal.shape[0]), 0)
    ImageDraw.Draw(source_socket_zone).polygon(
        [(190, 65), (224, 65), (232, 83), (225, 104),
         (210, 116), (188, 108), (179, 92)],
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
    body = load_mask(BODY)
    outer_socket_seam = cv2.dilate(
        transformed_socket.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
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
    tiny_roi = (xx >= 350) & (xx <= 414) & (yy >= 266) & (yy <= 316)
    mask = (
        outer_socket_seam
        & tiny_roi
        & ~protected_blade_and_guards
        & ~protected_cloth_shaft
        & ~body
    )
    if not np.any(mask):
        raise RuntimeError("seam mask is empty")
    forbidden_overlap = int(
        np.count_nonzero(
            mask & (protected_blade_and_guards | protected_cloth_shaft | body)
        )
    )
    if forbidden_overlap:
        raise RuntimeError(f"mask includes {forbidden_overlap} protected pixels")
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels != 539:
        raise RuntimeError(f"unexpected seam-mask count: {mask_pixels} != 539")

    MASK.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        MASK, optimize=True
    )
    close_crop = (300, 210, 450, 370)
    tinted = primary.copy()
    tinted[mask] = np.clip(
        np.round(tinted[mask].astype(np.float32) * 0.30 + np.array([0, 255, 255]) * 0.70),
        0,
        255,
    ).astype(np.uint8)
    panel_size = (600, 640)
    proof = Image.new("RGB", (1200, 690), (12, 15, 20))
    for index, (title, array) in enumerate(
        (("APPROVED TENSE V2", primary), ("539PX SOCKET-SEAM MASK", tinted))
    ):
        panel = Image.fromarray(array, mode="RGB").crop(close_crop).resize(
            panel_size, Image.Resampling.NEAREST
        )
        proof.paste(panel, (index * 600, 0))
        ImageDraw.Draw(proof).text(
            (index * 600 + 10, 656), title, fill=(255, 255, 255)
        )
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    proof.save(MASK_PROOF, optimize=True)

    return {
        "pixel_count": mask_pixels,
        "bbox_xyxy_exclusive": bbox(mask),
        "socket_material_pixels_inside": int(np.count_nonzero(mask & transformed_socket)),
        "background_seam_pixels_inside": int(np.count_nonzero(mask & ~metal_patch)),
        "blade_guard_protected_overlap": int(
            np.count_nonzero(mask & protected_blade_and_guards)
        ),
        "cloth_shaft_protected_overlap": int(
            np.count_nonzero(mask & protected_cloth_shaft)
        ),
        "body_protected_overlap": int(np.count_nonzero(mask & body)),
        "sha256": sha256(MASK),
        "proof_sha256": sha256(MASK_PROOF),
        "transform_matrix": matrix.tolist(),
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
        "filename_prefix": "mirelle_tense_socket_seam_qwen_v1",
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
    }


def make_review_proof() -> None:
    crop = (300, 210, 450, 370)
    paths = (
        ("APPROVED V2 FALLBACK", PRIMARY),
        ("QWEN HARD COMPOSITE", COMPOSITE),
        ("ALIGNED RAW MODEL", ALIGNED_OUTPUT),
    )
    sheet = Image.new("RGB", (1800, 690), (12, 15, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (title, path) in enumerate(paths):
        panel = Image.open(path).convert("RGB").crop(crop).resize(
            (600, 640), Image.Resampling.NEAREST
        )
        sheet.paste(panel, (index * 600, 0))
        draw.text((index * 600 + 10, 656), title, fill=(255, 255, 255))
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
    attempt_record: dict[str, object] = {
        "status": "dispatching_single_authorized_call",
        "started_at": datetime.now(UTC).isoformat(),
        "gateway": GATEWAY,
        "endpoint": ENDPOINT,
        "request_body_path": str(REQUEST_BODY),
        "request_body_sha256": sha256(REQUEST_BODY),
        "second_call_permitted": False,
    }
    write_json(ATTEMPT, attempt_record)

    request = urllib.request.Request(
        f"{GATEWAY}{ENDPOINT}",
        data=request_bytes,
        headers={"Content-Type": "application/json"},
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
    write_json(RESPONSE, response_record)
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
            "reason": "approved deterministic v2 remains selected unless Qwen is visibly better",
        },
        "model_calls": 1,
        "inputs": {
            "approved_primary": {"path": str(PRIMARY), "sha256": sha256(PRIMARY)},
            "locked_canonical_crop": {"path": str(REFERENCE), "sha256": sha256(REFERENCE)},
            "frozen_metal_component": {"path": str(METAL), "sha256": sha256(METAL)},
            "mask": {"path": str(MASK), **mask_record},
            "prompt": {"path": str(PROMPT_PATH), "sha256": sha256(PROMPT_PATH)},
        },
        "exact_request": {
            "path": str(REQUEST_BODY),
            "sha256": sha256(REQUEST_BODY),
            "byte_count": REQUEST_BODY.stat().st_size,
            "endpoint": ENDPOINT,
            "seed": SEED,
            "steps": STEPS,
            "cfg": CFG,
            "denoise": DENOISE,
        },
        "exact_response": {
            "headers_path": str(RESPONSE),
            "headers_sha256": sha256(RESPONSE),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute-authorized-once",
        action="store_true",
        help="send the one authorized Qwen request; absence means prepare-only",
    )
    args = parser.parse_args()
    check_hash_fences()
    mask_record = prepare_mask()
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_PATH.write_text(PROMPT + "\n", encoding="utf-8")
    if not args.execute_authorized_once:
        print(json.dumps({"status": "prepared_no_model_call", "mask": mask_record}, indent=2))
        return
    execute_once(mask_record)


if __name__ == "__main__":
    main()
