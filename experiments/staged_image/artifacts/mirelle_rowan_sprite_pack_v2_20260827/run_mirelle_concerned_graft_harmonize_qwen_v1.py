#!/usr/bin/env python3
"""Harmonize only the seams of Mirelle's deterministic reference graft."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))

from experiments.staged_image.pipeline import (  # noqa: E402
    GatewayClient,
    blend_masked_result,
    encode_image,
)


PRIMARY = ROOT / "grafts/mirelle_voss/concerned_lower_spear_reference_graft_v5.png"
REFERENCE = (
    ROOT
    / "supporting_refs/mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
GRAFT_ALPHA = ROOT / "masks/mirelle_voss/concerned_reference_graft_alpha_mask_v5.png"
ERASE_MASK = ROOT / "masks/mirelle_voss/concerned_old_lower_spear_erase_mask_v5.png"
MASK = ROOT / "masks/mirelle_voss/concerned_reference_graft_seam_mask_v2.png"
MASK_PROOF = ROOT / "mask_proofs/mirelle_concerned_reference_graft_seam_mask_v2.png"
PROMPT_PATH = ROOT / "prompts/qwen/mirelle_concerned_reference_graft_harmonize_v2.txt"
RAW_OUTPUT = (
    ROOT / "qwen_raw/mirelle_voss/concerned_reference_graft_harmonize_v2_model_output.png"
)
COMPOSITE = (
    ROOT / "qwen_composited/mirelle_voss/concerned_reference_graft_harmonized_v2.png"
)
HEADERS = ROOT / "qwen_headers/mirelle_concerned_reference_graft_harmonize_v2.json"
METADATA = ROOT / "qwen_metadata/mirelle_concerned_reference_graft_harmonize_v2.json"

SEED = 260827325
STEPS = 26
CFG = 3.5
DENOISE = 0.25

PROMPT = """Use case: seam-only harmonization of a deterministic canonical prop graft.
Input roles: Image 1 is the primary edit target. Its lower spearhead, two small lateral wings, socket, and long red streamers are an exact scaled/rotated graft from Mirelle's locked reference and MUST keep their supplied silhouette, count, placement, and design. Image 2 is supporting canonical-reference evidence only.
Primary request: Integrate only the collar/socket junction, cut ribbon roots and ends, tiny old-wing/magenta slivers, and ribbon/greave/boot occlusion crossings inside the white mask. Restore clean local magenta screen wherever a removed old-prop sliver should reveal background. Where a ribbon crosses a greave, boot, or leg, establish a physically coherent front/back occlusion without changing the armor, leg, or ribbon design. The long central blade body is deliberately outside the mask and must remain exact.
Preservation contract: Do not redraw, resize, simplify, replace, or restyle the canonical graft. Preserve the long central silver leaf blade, exactly two small lateral wing guards, socket, red shaft alignment, and all long red streamers. Preserve every unmasked pixel exactly: Mirelle's face, concerned expression, hair, braid, body, pose, hands, clothing, armor, boots, upper pointed cap and short tassel, framing, and uniform magenta background. Do not add or remove weapon parts, people, text, scenery, shadows, or texture.
Mask semantics: White marks only seam pixels eligible for change; black must remain unchanged."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_mask() -> int:
    alpha = np.asarray(Image.open(GRAFT_ALPHA).convert("L"), dtype=np.uint8)
    erase = np.asarray(Image.open(ERASE_MASK).convert("L"), dtype=np.uint8)
    subject = (alpha >= 16).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    outer = cv2.dilate(subject, kernel) > 0
    inner = cv2.erode(subject, kernel) > 0
    boundary = outer & ~inner
    residual = (erase > 0) & (alpha < 200)
    residual = cv2.dilate(
        residual.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ) > 0
    mask_u8 = (boundary | residual).astype(np.uint8)
    cv2.ellipse(mask_u8, (608, 1128), (36, 54), 0, 0, 360, 1, -1)
    mask = mask_u8 > 0
    zone = np.zeros_like(mask)
    zone[965:1236, 470:711] = True
    mask &= zone
    # Add the collar junction explicitly, while keeping the blade body below
    # the wing/socket assembly completely outside edit scope.
    collar = np.zeros_like(mask, dtype=np.uint8)
    cv2.ellipse(collar, (608, 1128), (34, 46), 0, 0, 360, 1, -1)
    mask |= collar > 0
    binary = np.where(mask, 255, 0).astype(np.uint8)
    MASK.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary, mode="L").save(MASK, optimize=True)

    primary = np.asarray(Image.open(PRIMARY).convert("RGB"), dtype=np.uint8)
    tint = primary.astype(np.float32)
    tint[mask] = 0.45 * tint[mask] + 0.55 * np.array([0, 255, 255])
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(tint).astype(np.uint8), mode="RGB").save(
        MASK_PROOF, optimize=True
    )
    return int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for path in (PRIMARY, REFERENCE, GRAFT_ALPHA, ERASE_MASK):
        if not path.is_file():
            raise RuntimeError(f"missing input: {path}")
    mask_pixels = make_mask()
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_PATH.write_text(PROMPT + "\n", encoding="utf-8")
    if args.prepare_only:
        print(f"mask={MASK} sha256={sha256(MASK)} pixels={mask_pixels}")
        return

    client = GatewayClient("http://127.0.0.1:8199", timeout_seconds=1800)
    edit = client.health().get("pipelines", {}).get("edit", {})
    if not edit.get("available") or "Qwen" not in str(edit.get("model")):
        raise RuntimeError(f"Qwen edit pipeline unavailable: {edit}")
    payload = {
        "prompt": PROMPT,
        "image_base64": encode_image(PRIMARY),
        "image2_base64": encode_image(REFERENCE),
        "mask_base64": encode_image(MASK),
        "seed": SEED,
        "steps": STEPS,
        "cfg": CFG,
        "denoise": DENOISE,
        "filename_prefix": "mirelle_concerned_reference_graft_harmonize_v2",
    }
    content, headers = client.post_image("/prototype/edit/qwen/masked", payload)
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.write_bytes(content)
    with Image.open(RAW_OUTPUT) as image:
        model_size = list(image.size)
        image.verify()
    COMPOSITE.parent.mkdir(parents=True, exist_ok=True)
    outside_delta = blend_masked_result(PRIMARY, RAW_OUTPUT, MASK, COMPOSITE)
    HEADERS.parent.mkdir(parents=True, exist_ok=True)
    HEADERS.write_text(json.dumps(headers, indent=2) + "\n", encoding="utf-8")
    with Image.open(COMPOSITE) as image:
        composite_size = list(image.size)
    metadata = {
        "status": "pending_manual_review",
        "primary_graft": {"path": str(PRIMARY), "sha256": sha256(PRIMARY)},
        "canonical_reference": {"path": str(REFERENCE), "sha256": sha256(REFERENCE)},
        "mask": {"path": str(MASK), "sha256": sha256(MASK), "pixel_count": mask_pixels},
        "request": {
            "endpoint": "/prototype/edit/qwen/masked",
            "seed": SEED,
            "steps": STEPS,
            "cfg": CFG,
            "denoise": DENOISE,
            "prompt_path": str(PROMPT_PATH),
            "prompt_sha256": sha256(PROMPT_PATH),
        },
        "response": {
            "headers": headers,
            "model_output_dimensions": model_size,
            "model_output_sha256": sha256(RAW_OUTPUT),
            "composite_dimensions": composite_size,
            "composite_sha256": sha256(COMPOSITE),
            "changed_outside_mask": outside_delta,
        },
    }
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
