#!/usr/bin/env python3
"""Run one bounded Qwen proof for Mirelle's concerned lower spearhead."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.staged_image.pipeline import (
    GatewayClient,
    blend_masked_result,
    encode_image,
)


ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "generation_raw/mirelle_voss/concerned_chroma_v2.png"
REFERENCE = (
    ROOT
    / "supporting_refs/mirelle_canonical_spearhead_crop_active_profile_x0_y540_x410_y1045.png"
)
SOURCE_MASK = ROOT / "masks/mirelle_voss/concerned_spear_mask_v1.png"
MASK = ROOT / "masks/mirelle_voss/concerned_lower_spearhead_mask_v2.png"
PROMPT_PATH = ROOT / "prompts/qwen/mirelle_concerned_lower_spearhead_masked_v2.txt"
RAW_OUTPUT = (
    ROOT / "qwen_raw/mirelle_voss/concerned_lower_spearhead_masked_v2_model_output.png"
)
COMPOSITE = (
    ROOT / "qwen_composited/mirelle_voss/concerned_lower_spearhead_masked_v2.png"
)
HEADER_PATH = ROOT / "qwen_headers/mirelle_concerned_lower_spearhead_masked_v2.json"
METADATA_PATH = ROOT / "qwen_metadata/mirelle_concerned_lower_spearhead_masked_v2.json"
MASK_PROOF = ROOT / "mask_proofs/mirelle_concerned_lower_spearhead_mask_v2.png"

SEED = 260827322
STEPS = 32
CFG = 4.0
DENOISE = 0.42
LOWER_CUTOFF_Y = 1000

PROMPT = """Use case: precise-object-edit
Asset type: experimental visual-novel character sprite prop normalization
Input roles: Image 1 is the primary edit target and must remain the base image. Image 2 is supporting evidence for Mirelle's canonical spear design only; do not copy its pose, body, background, or framing.
Primary request: Replace only the LOWER primary spearhead, socket, and its adjacent ribbons inside the white mask in Image 1. Keep the existing point-down shaft path and hand grip. Match Image 2: exactly one long central silver leaf blade, exactly two small lateral wing guards beside the socket, and long red streamers tied immediately beside that lower spearhead/socket. Remove the current oversized halberd-like lower silhouette. The result remains one coherent double-ended red-shaft spear.
Canonical upper-end rule: Mirelle's small pointed silver cap plus short red tassel at the UPPER end is intentional and canonical. It is outside this mask and must remain exactly unchanged.
Preservation contract: Change only spearhead/socket/ribbon pixels inside the supplied white mask. Preserve Image 1's exact face, concerned expression, orange-red hair and braid, body, pose, hands, cream coat, burgundy clothing, armor, boots, shaft outside the mask, upper cap and tassel, framing, scale, and uniform magenta background. Keep character and costume pixels visible through or beside the prop unchanged. Do not add a second weapon, extra blade, extra person, text, logo, watermark, scenery, floor, shadow, or texture.
Mask semantics: White is the only eligible edit region; black must remain unchanged."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_mask() -> None:
    with Image.open(PRIMARY) as image:
        size = image.size
    with Image.open(SOURCE_MASK) as image:
        source = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    if (source.shape[1], source.shape[0]) != size:
        raise RuntimeError("source mask geometry differs from primary")
    source[:LOWER_CUTOFF_Y] = 0
    binary = np.where(source >= 128, 255, 0).astype(np.uint8)
    MASK.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary, mode="L").save(MASK, optimize=True)

    with Image.open(PRIMARY) as image:
        proof = image.convert("RGB")
    overlay = Image.new("RGBA", proof.size, (0, 0, 0, 0))
    overlay.putalpha(Image.fromarray((binary // 2).astype(np.uint8), mode="L"))
    ImageDraw.Draw(overlay).rectangle((0, 0, 0, 0), fill=(255, 255, 0, 128))
    tint = Image.new("RGBA", proof.size, (0, 255, 255, 0))
    tint.putalpha(Image.fromarray((binary // 2).astype(np.uint8), mode="L"))
    proof = Image.alpha_composite(proof.convert("RGBA"), tint)
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    proof.convert("RGB").save(MASK_PROOF, optimize=True)


def main() -> None:
    for path in (PRIMARY, REFERENCE, SOURCE_MASK):
        if not path.is_file():
            raise RuntimeError(f"missing input: {path}")
    make_mask()
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_PATH.write_text(PROMPT + "\n", encoding="utf-8")

    client = GatewayClient("http://127.0.0.1:8199", timeout_seconds=1800)
    health = client.health()
    edit = health.get("pipelines", {}).get("edit", {})
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
        "filename_prefix": "mirelle_concerned_lower_spearhead_masked_v2",
    }
    content, headers = client.post_image("/prototype/edit/qwen/masked", payload)
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.write_bytes(content)
    with Image.open(RAW_OUTPUT) as image:
        model_size = list(image.size)
        image.verify()
    COMPOSITE.parent.mkdir(parents=True, exist_ok=True)
    changed_outside = blend_masked_result(PRIMARY, RAW_OUTPUT, MASK, COMPOSITE)

    HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEADER_PATH.write_text(json.dumps(headers, indent=2) + "\n", encoding="utf-8")
    with Image.open(MASK) as image:
        mask_pixels = int((np.asarray(image.convert("L")) > 0).sum())
    metadata = {
        "status": "pending_manual_review",
        "input_roles": {"primary": str(PRIMARY), "canonical_spear_support": str(REFERENCE)},
        "inputs": {
            "primary_sha256": sha256(PRIMARY),
            "canonical_spear_support_sha256": sha256(REFERENCE),
            "source_broad_mask_sha256": sha256(SOURCE_MASK),
            "narrow_lower_mask_sha256": sha256(MASK),
        },
        "request": {
            "endpoint": "/prototype/edit/qwen/masked",
            "seed": SEED,
            "steps": STEPS,
            "cfg": CFG,
            "denoise": DENOISE,
            "lower_cutoff_y": LOWER_CUTOFF_Y,
            "prompt_path": str(PROMPT_PATH),
            "prompt_sha256": sha256(PROMPT_PATH),
            "mask_pixel_count": mask_pixels,
        },
        "response": {
            "headers": headers,
            "model_output_dimensions": model_size,
            "model_output_sha256": sha256(RAW_OUTPUT),
            "composite_dimensions": list(Image.open(COMPOSITE).size),
            "composite_sha256": sha256(COMPOSITE),
            "changed_outside_mask": changed_outside,
        },
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
