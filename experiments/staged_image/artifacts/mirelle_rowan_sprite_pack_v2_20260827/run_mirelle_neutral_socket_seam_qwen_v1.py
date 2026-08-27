#!/usr/bin/env python3
"""Run exactly one bounded Qwen seam experiment on approved Mirelle neutral v1."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.staged_image.pipeline import GatewayClient, encode_image  # noqa: E402


PRIMARY = ROOT / "grafts/mirelle_voss/neutral_canonical_metal_repair_v1.png"
REFERENCE = (
    ROOT
    / "supporting_refs/"
    "mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
METAL_PATCH = (
    ROOT / "masks/mirelle_voss/neutral_canonical_metal_component_patch_mask_v1.png"
)
MASK = (
    ROOT / "masks/mirelle_voss/neutral_socket_background_seam_qwen_mask_v1.png"
)
MASK_PROOF = ROOT / "mask_proofs/mirelle_neutral_socket_seam_qwen_mask_v1.png"
PROMPT_PATH = ROOT / "prompts/qwen/mirelle_neutral_socket_seam_qwen_v1.txt"
REQUEST_BODY = ROOT / "qwen_requests/mirelle_neutral_socket_seam_qwen_v1.json"
PREFLIGHT_HEALTH = (
    ROOT / "qwen_requests/mirelle_neutral_socket_seam_qwen_v1_health.json"
)
RAW_OUTPUT = (
    ROOT / "qwen_raw/mirelle_voss/neutral_socket_seam_qwen_v1_model_output.png"
)
ERROR_BODY = ROOT / "qwen_raw/mirelle_voss/neutral_socket_seam_qwen_v1_error.bin"
ALIGNED_OUTPUT = (
    ROOT / "qwen_aligned/mirelle_voss/neutral_socket_seam_qwen_v1_aligned.png"
)
COMPOSITE = (
    ROOT / "qwen_composited/mirelle_voss/neutral_socket_seam_qwen_v1.png"
)
HEADERS = ROOT / "qwen_headers/mirelle_neutral_socket_seam_qwen_v1.json"
METADATA = ROOT / "qwen_metadata/mirelle_neutral_socket_seam_qwen_v1.json"
COMPARISON = (
    ROOT / "component_proofs/mirelle_neutral_socket_seam_qwen_v1_comparison.png"
)

GATEWAY = "http://127.0.0.1:8199"
ENDPOINT = "/prototype/edit/qwen/masked"
SEED = 260827331
STEPS = 24
CFG = 3.2
DENOISE = 0.20

EXPECTED_PRIMARY_SHA = "08fc7d8223ed162039d30d7789031748694ec60510921a4c714033adf650a407"
EXPECTED_REFERENCE_SHA = "e4e454f5b8f98546ae5c8fa2fd1fb28dc16f197d7494f7ad16a16c25a362afff"
EXPECTED_PATCH_SHA = "e9db0040cd50d7fa29b7e653f00cdeee9cd992e0f106fb6db32bf9f189dce6bf"

PROMPT = """Use case: precise-object-edit
Asset type: one-off masked seam experiment for a visual-novel character sprite
Input roles: Image 1 is the approved deterministic edit target. Image 2 is Mirelle's locked canonical spear crop and is supporting evidence only for the silver collar/socket material and edge treatment.
Primary request: Make the smallest possible cleanup only at the silver collar/socket-to-magenta-background junction inside the white mask. Remove any jagged magenta notch, fringe, or pasted edge there and make the cylindrical silver collar meet the existing red shaft cleanly. Preserve the supplied collar size, position, shading, and silhouette.
Preservation contract: Change nothing outside the white mask. Do not redraw, move, resize, simplify, or restyle the central blade, lateral guards, socket, red shaft, or pose-specific red tassel. Do not copy Image 2's orientation, streamers, body, or background. Preserve Mirelle's face, hair, body, pose, hand, clothing, armor, boots, opposite lower cap, framing, and magenta screen exactly. Add no weapon parts, streamers, people, text, scenery, shadow, or texture.
Mask semantics: White is the only eligible seam region. Black must remain unchanged."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> None:
    expected = {
        PRIMARY: EXPECTED_PRIMARY_SHA,
        REFERENCE: EXPECTED_REFERENCE_SHA,
        METAL_PATCH: EXPECTED_PATCH_SHA,
    }
    for path, digest in expected.items():
        actual = sha256(path)
        if actual != digest:
            raise RuntimeError(
                f"hash fence failed for {path}: expected {digest}, got {actual}"
            )


def make_mask() -> int:
    primary = np.asarray(Image.open(PRIMARY).convert("RGB"), dtype=np.uint8)
    patch = np.asarray(Image.open(METAL_PATCH).convert("L"), dtype=np.uint8) > 0
    outer = cv2.dilate(
        patch.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    ) > 0
    inner = cv2.erode(
        patch.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    boundary = outer & ~inner

    # The authored rectangle includes only the cylindrical collar/socket and a
    # narrow ring of adjacent chroma.  Blade and lateral guards end above or
    # outside this zone.  The red exclusion is dilated to protect the shaft and
    # pose-specific tassel even at antialiased edges.
    zone = np.zeros_like(patch)
    zone[205:243, 294:324] = True
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
    red_protection = cv2.dilate(
        red.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ) > 0
    mask = boundary & zone & ~red_protection
    if not (250 <= int(mask.sum()) <= 350):
        raise RuntimeError(f"unexpected tiny seam mask size: {int(mask.sum())}")
    ys, xs = np.where(mask)
    if xs.min() < 294 or xs.max() > 323 or ys.min() < 205 or ys.max() > 242:
        raise RuntimeError("seam mask escaped its authored collar/socket zone")
    if np.any(mask & red_protection):
        raise RuntimeError("seam mask overlaps protected red shaft/cloth")

    MASK.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        MASK, optimize=True
    )
    make_mask_proof(primary, mask)
    return int(mask.sum())


def contained_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (12, 15, 20))
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.NEAREST)
    panel.paste(
        contained,
        ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
    )
    return panel


def make_mask_proof(primary: np.ndarray, mask: np.ndarray) -> None:
    crop = (270, 190, 345, 260)
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
        ("APPROVED V1", base_crop),
        ("TINY MASK OVERLAY", overlay_crop),
        ("BINARY MASK", binary_crop),
    ]
    panel_size = (450, 420)
    label_height = 48
    sheet = Image.new(
        "RGB", (panel_size[0] * len(entries), panel_size[1] + label_height), (12, 15, 20)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(entries):
        panel = contained_panel(image, panel_size)
        sheet.paste(panel, (index * panel_size[0], 0))
        draw.text(
            (index * panel_size[0] + 10, panel_size[1] + 15),
            title,
            fill=(255, 255, 255),
        )
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(MASK_PROOF, optimize=True)


def exact_request_payload() -> dict[str, Any]:
    return {
        "prompt": PROMPT,
        "image_base64": encode_image(PRIMARY),
        "image2_base64": encode_image(REFERENCE),
        "mask_base64": encode_image(MASK),
        "seed": SEED,
        "steps": STEPS,
        "cfg": CFG,
        "denoise": DENOISE,
        "filename_prefix": "mirelle_neutral_socket_seam_qwen_v1",
    }


def post_exact(body: bytes) -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(
        f"{GATEWAY}{ENDPOINT}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            content = response.read()
            response_record = {
                "status": response.status,
                "reason": response.reason,
                "headers": list(response.headers.raw_items()),
            }
    except urllib.error.HTTPError as exc:
        content = exc.read()
        ERROR_BODY.parent.mkdir(parents=True, exist_ok=True)
        ERROR_BODY.write_bytes(content)
        response_record = {
            "status": exc.code,
            "reason": exc.reason,
            "headers": list(exc.headers.raw_items()),
            "error_body_path": str(ERROR_BODY),
            "error_body_sha256": sha256(ERROR_BODY),
        }
        HEADERS.parent.mkdir(parents=True, exist_ok=True)
        HEADERS.write_text(
            json.dumps(response_record, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"single authorized Qwen call returned HTTP {exc.code}") from exc
    if not content:
        raise RuntimeError("single authorized Qwen call returned an empty body")
    return content, response_record


def align_model_output(
    primary: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = primary.shape[:2]
    resized = cv2.resize(generated, (width, height), interpolation=cv2.INTER_LANCZOS4)
    primary_gray = cv2.cvtColor(primary, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    resized_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    not_hot_magenta = ~(
        (primary[..., 0] >= 220)
        & (primary[..., 1] <= 35)
        & (primary[..., 2] >= 220)
    )
    protected = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
    ) > 0
    alignment_mask = np.where(not_hot_magenta & ~protected, 255, 0).astype(np.uint8)
    warp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    fallback = False
    try:
        correlation, warp = cv2.findTransformECC(
            primary_gray,
            resized_gray,
            warp,
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-7),
            alignment_mask,
            5,
        )
        aligned = cv2.warpAffine(
            resized,
            warp,
            (width, height),
            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
    except cv2.error:
        correlation = None
        fallback = True
        aligned = resized
        warp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return aligned, {
        "method": "resize_to_primary_then_ecc_translation",
        "fallback_to_resize_only": fallback,
        "ecc_correlation": None if correlation is None else float(correlation),
        "warp_matrix": warp.astype(float).tolist(),
        "alignment_mask_pixels": int(np.count_nonzero(alignment_mask)),
    }


def make_comparison(
    primary: np.ndarray,
    raw_resized: np.ndarray,
    aligned: np.ndarray,
    composite: np.ndarray,
) -> None:
    crop = (270, 190, 345, 260)
    entries = [
        ("APPROVED V1", Image.fromarray(primary, mode="RGB").crop(crop)),
        ("RAW RESIZED", Image.fromarray(raw_resized, mode="RGB").crop(crop)),
        ("RAW ALIGNED", Image.fromarray(aligned, mode="RGB").crop(crop)),
        ("HARD-MASK COMPOSITE", Image.fromarray(composite, mode="RGB").crop(crop)),
    ]
    panel_size = (360, 336)
    label_height = 46
    sheet = Image.new(
        "RGB", (panel_size[0] * len(entries), panel_size[1] + label_height), (12, 15, 20)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(entries):
        panel = contained_panel(image, panel_size)
        sheet.paste(panel, (index * panel_size[0], 0))
        draw.text(
            (index * panel_size[0] + 8, panel_size[1] + 14),
            title,
            fill=(255, 255, 255),
        )
    COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(COMPARISON, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    verify_inputs()
    mask_pixels = make_mask()
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_PATH.write_text(PROMPT + "\n", encoding="utf-8")
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "mask": str(MASK),
                    "mask_sha256": sha256(MASK),
                    "mask_pixels": mask_pixels,
                    "mask_proof": str(MASK_PROOF),
                    "mask_proof_sha256": sha256(MASK_PROOF),
                    "prompt": str(PROMPT_PATH),
                    "prompt_sha256": sha256(PROMPT_PATH),
                },
                indent=2,
            )
        )
        return

    # REQUEST_BODY is the irreversible one-call sentinel.  Once written, this
    # script refuses to submit again even if the prior network call failed.
    if REQUEST_BODY.exists():
        raise RuntimeError(
            f"authorized Qwen request already materialized at {REQUEST_BODY}; refusing retry"
        )

    client = GatewayClient(GATEWAY, timeout_seconds=1800)
    health = client.health()
    edit = health.get("pipelines", {}).get("edit", {})
    if not edit.get("available") or "Qwen" not in str(edit.get("model")):
        raise RuntimeError(f"Qwen edit pipeline unavailable: {edit}")
    PREFLIGHT_HEALTH.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_HEALTH.write_text(json.dumps(health, indent=2) + "\n", encoding="utf-8")

    payload = exact_request_payload()
    request_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    REQUEST_BODY.parent.mkdir(parents=True, exist_ok=True)
    REQUEST_BODY.write_bytes(request_body)

    content, response_record = post_exact(request_body)
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.write_bytes(content)
    HEADERS.parent.mkdir(parents=True, exist_ok=True)
    HEADERS.write_text(
        json.dumps(response_record, indent=2) + "\n", encoding="utf-8"
    )

    with Image.open(RAW_OUTPUT) as image:
        image.load()
        model_size = list(image.size)
        generated = np.asarray(image.convert("RGB"), dtype=np.uint8)
    primary = np.asarray(Image.open(PRIMARY).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(MASK).convert("L"), dtype=np.uint8) > 0
    aligned, alignment = align_model_output(primary, generated, mask)
    ALIGNED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(aligned, mode="RGB").save(ALIGNED_OUTPUT, optimize=True)

    composite = primary.copy()
    composite[mask] = aligned[mask]
    outside_delta = int(np.count_nonzero(np.any(composite != primary, axis=2) & ~mask))
    if outside_delta:
        raise RuntimeError(f"hard composite changed {outside_delta} pixels outside mask")
    COMPOSITE.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite, mode="RGB").save(COMPOSITE, optimize=True)

    raw_resized = cv2.resize(
        generated,
        (primary.shape[1], primary.shape[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    make_comparison(primary, raw_resized, aligned, composite)
    changed = np.any(composite != primary, axis=2)
    inside_difference = np.abs(
        composite[mask].astype(np.int16) - primary[mask].astype(np.int16)
    )
    metadata = {
        "status": "pending_manual_review",
        "selection": "approved deterministic v1 remains selected fallback",
        "model_calls": 1,
        "inputs": {
            "primary": {"path": str(PRIMARY), "sha256": sha256(PRIMARY)},
            "canonical_support": {
                "path": str(REFERENCE),
                "sha256": sha256(REFERENCE),
            },
            "metal_patch_source": {
                "path": str(METAL_PATCH),
                "sha256": sha256(METAL_PATCH),
            },
        },
        "mask": {
            "path": str(MASK),
            "sha256": sha256(MASK),
            "pixel_count": mask_pixels,
            "bbox_xyxy": [294, 205, 324, 243],
            "blade_guard_tassel_shaft_character_overlap_pixels": 0,
            "proof": {"path": str(MASK_PROOF), "sha256": sha256(MASK_PROOF)},
        },
        "request": {
            "gateway": GATEWAY,
            "endpoint": ENDPOINT,
            "seed": SEED,
            "steps": STEPS,
            "cfg": CFG,
            "denoise": DENOISE,
            "prompt_path": str(PROMPT_PATH),
            "prompt_sha256": sha256(PROMPT_PATH),
            "exact_body_path": str(REQUEST_BODY),
            "exact_body_sha256": sha256(REQUEST_BODY),
            "exact_body_bytes": REQUEST_BODY.stat().st_size,
            "preflight_health_path": str(PREFLIGHT_HEALTH),
            "preflight_health_sha256": sha256(PREFLIGHT_HEALTH),
        },
        "response": {
            "headers_path": str(HEADERS),
            "headers_sha256": sha256(HEADERS),
            "raw_model_output_path": str(RAW_OUTPUT),
            "raw_model_output_sha256": sha256(RAW_OUTPUT),
            "raw_model_output_dimensions": model_size,
            "aligned_output_path": str(ALIGNED_OUTPUT),
            "aligned_output_sha256": sha256(ALIGNED_OUTPUT),
            "alignment": alignment,
            "hard_composite_path": str(COMPOSITE),
            "hard_composite_sha256": sha256(COMPOSITE),
            "hard_composite_dimensions": [primary.shape[1], primary.shape[0]],
            "changed_inside_mask_pixels": int(np.count_nonzero(changed & mask)),
            "changed_outside_mask_pixels": outside_delta,
            "mean_absolute_channel_delta_inside_mask": float(
                np.mean(inside_difference)
            ),
            "comparison_path": str(COMPARISON),
            "comparison_sha256": sha256(COMPARISON),
        },
    }
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
