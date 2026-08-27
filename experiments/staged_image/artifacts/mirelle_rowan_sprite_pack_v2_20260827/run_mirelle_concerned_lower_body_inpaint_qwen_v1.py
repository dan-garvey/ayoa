#!/usr/bin/env python3
"""Run exactly one bounded Qwen inpaint for Mirelle concerned lower body."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.staged_image.pipeline import GatewayClient, encode_image


ROOT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
GATEWAY = "http://127.0.0.1:8199"
ENDPOINT = "/prototype/edit/qwen/masked"

SEMANTIC_BASE = ROOT / "generation_raw/mirelle_voss/concerned_chroma_v2.png"
CLEANED_BASE = ROOT / "grafts/mirelle_voss/concerned_old_lower_primary_cleaned_base_v8.png"
REJECTED_V8 = ROOT / "grafts/mirelle_voss/concerned_lower_primary_metal_reference_graft_v8.png"
SOURCE_MASK = ROOT / "masks/mirelle_voss/concerned_old_lower_primary_cleanup_mask_v8.png"
V8_PATCH_MASK = ROOT / "masks/mirelle_voss/concerned_lower_canonical_metal_patch_mask_v8.png"
ACTIVE_PROFILE = REPO / "app/storage/stories/one_star_ascension_s1/visual-references/locked/mirelle_voss/active_profile.png"
ANATOMY = REPO / "app/storage/stories/one_star_ascension_s1/visual-references/locked/mirelle_voss/anatomy.png"

MASK = ROOT / "masks/mirelle_voss/concerned_former_head_missing_lower_body_inpaint_mask_v1.png"
PROTECTED = ROOT / "masks/mirelle_voss/concerned_visible_unaffected_protected_mask_v1.png"
MASK_PROOF = ROOT / "mask_proofs/mirelle_concerned_lower_body_inpaint_mask_v1.png"
PROMPT_PATH = ROOT / "prompts/qwen/mirelle_concerned_lower_body_inpaint_v1.txt"
HEALTH = ROOT / "qwen_requests/mirelle_concerned_lower_body_inpaint_v1_health.json"
REQUEST = ROOT / "qwen_requests/mirelle_concerned_lower_body_inpaint_v1_request.json"
HEADERS = ROOT / "qwen_headers/mirelle_concerned_lower_body_inpaint_v1_response.json"
RAW = ROOT / "qwen_raw/mirelle_voss/concerned_lower_body_inpaint_v1_model_output.png"
ALIGNED = ROOT / "qwen_aligned/mirelle_voss/concerned_lower_body_inpaint_v1_aligned.png"
COMPOSITE = ROOT / "qwen_composited/mirelle_voss/concerned_lower_body_inpaint_v1.png"
PROOF = ROOT / "qwen_proofs/mirelle_concerned_lower_body_inpaint_v1.png"
METADATA = ROOT / "qwen_metadata/mirelle_concerned_lower_body_inpaint_v1.json"

ARCHIVE = ROOT / "rejected_provenance/mirelle_voss/concerned_v8_old_head_cleanup_erased_occluded_lower_body"
ARCHIVE_MANIFEST = ARCHIVE / "rejection_manifest.json"

SEED = 260827389
STEPS = 32
CFG = 3.5
DENOISE = 0.62
MAGENTA = np.array([255, 0, 255], dtype=np.uint8)

EXPECTED = {
    SEMANTIC_BASE: "ac5d892873234388c83080cf42bc5885d8944666cb6315df07d0969206fbb517",
    CLEANED_BASE: "2f94667a95e6ec02c500fb8f986a8769a76f8a2b76094e55f4db153219639830",
    REJECTED_V8: "07008f7348b27df98f98240447a7fd0033fe275d994af8abbeed57bfdb338e9e",
    SOURCE_MASK: "f4467ba4f0f5150a2a2c12b2e987ea79befe72248beb99a3d54ad63d4508e8cb",
    V8_PATCH_MASK: "d361af448fc9c94baae4826b556bb68f80e0fef0a60a656dbf0cd1bfe45fd401",
    ACTIVE_PROFILE: "6173f6066860c270b73bcbde7fc86336411758301cdc5c12341f0400a82266b6",
    ANATOMY: "a84842f3e16bd650a4b85c02cf67c20ea6d694966ba855c6c2fd5709b5e0aaec",
}

PROMPT = """Use case: precise-object-edit
Asset type: one-shot masked lower-body reconstruction for a visual-novel character sprite
Input images: Image 1 is the only edit target: Mirelle's concerned pose with the obsolete oversized lower spearhead already removed to flat magenta. Image 2 is Mirelle's locked active-profile reference for her exact cream coat tails, burgundy trousers, silver greaves, and brown boots. Image 3 is Mirelle's locked anatomy reference for body proportions only; do not copy its underwear, pose, expression, framing, or gray background.
Primary request: Reconstruct only the portions of Mirelle's two lower trouser legs, two silver greaves, two brown boots, and the cream coat/cape edges that were hidden by the removed spearhead and are now missing inside the white mask. Continue each visible edge naturally from the unchanged pixels immediately outside the mask. Any masked pixels that are not part of those reconstructed body or garment surfaces must be exact flat saturated RGB 255,0,255 magenta background.
Style: Match Image 1 exactly: clean polished anime/manhwa line art, identical scale, lighting, palette, outline weight, and pose.
Preservation contract: Change only white-mask pixels. Preserve every black-mask pixel exactly, including Mirelle's face, concerned expression, orange-red hair and braid, entire upper body, arms, hands, grip, existing red spear shaft, upper silver cap and short tassel, all visible unaffected costume and lower-body pixels, framing, and flat magenta background. Do not draw, repair, extend, move, or add any weapon or prop. Do not add ribbons, streamers, extra limbs, extra feet, scenery, floor, shadows, gradients, text, logo, or watermark.
Mask semantics: White is the former-head/missing-lower-body region and is the only eligible edit area. Black must remain unchanged.
Output intent: one coherent full-body sprite with exactly two anatomically consistent legs, two greaves, and two boots; no weapon changes; solid magenta everywhere else inside the mask."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def archive_v8() -> dict[str, object]:
    files = [
        ROOT / "prepare_mirelle_concerned_lower_primary_metal_graft_v8.py",
        REJECTED_V8,
        CLEANED_BASE,
        ROOT / "component_metadata/mirelle_concerned_lower_primary_metal_graft_v8.json",
        ROOT / "component_metadata/mirelle_concerned_residual_silver_report_v8.json",
        ROOT / "component_proofs/mirelle_concerned_lower_primary_full_v8.png",
        ROOT / "component_proofs/mirelle_concerned_lower_primary_close_v8.png",
        ROOT / "component_proofs/mirelle_concerned_residual_silver_v8.png",
        ROOT / "component_proofs/mirelle_concerned_metal_connectivity_v8.png",
        ROOT / "mask_proofs/mirelle_concerned_lower_primary_masks_v8.png",
        SOURCE_MASK,
        V8_PATCH_MASK,
        ROOT / "masks/mirelle_voss/concerned_upper_cap_tassel_shaft_grip_exact_mask_v8.png",
        ROOT / "masks/mirelle_voss/concerned_body_face_costume_exact_mask_v8.png",
        ROOT / "masks/mirelle_voss/concerned_lower_primary_aggregate_saved_mask_v8.png",
        ROOT / "masks/mirelle_voss/concerned_lower_primary_changed_mask_v8.png",
    ]
    records = []
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = source.relative_to(ROOT)
        destination = ARCHIVE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(source):
            raise RuntimeError(f"conflicting archive file: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        records.append({"source": str(source), "archive": str(destination), "sha256": sha256(destination)})
    manifest = {
        "status": "rejected_preserved",
        "rejection_slug": "old_head_cleanup_erased_occluded_lower_body",
        "reason": "The deterministic old-head cleanup removed pixels where the obsolete source prop occluded the two lower legs, greaves, boots, and coat/cape edges. The resulting magenta holes are not acceptable body reconstruction.",
        "validation_correction": {
            "withdrawn_claim": "Source-hidden lower-body pixels were exact or preserved.",
            "correct_contract": "Only source-visible unaffected pixels outside the former-head mask are exact. Pixels inside that mask have no source ground truth and require visually reviewed reconstruction.",
            "legacy_body_mask_limit": "The archived v8 body mask excludes the cleanup region; it proves only unaffected visible source pixels and cannot validate geometry hidden by the old prop.",
        },
        "selected": False,
        "files": records,
    }
    write_json(ARCHIVE_MANIFEST, manifest)
    return {"path": str(ARCHIVE_MANIFEST), "sha256": sha256(ARCHIVE_MANIFEST)}


def panel(image: Image.Image, title: str, size: tuple[int, int]) -> Image.Image:
    scale = min(size[0] / image.width, size[1] / image.height)
    fitted = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)
    result = Image.new("RGB", (size[0], size[1] + 38), (12, 15, 20))
    result.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    ImageDraw.Draw(result).text((8, size[1] + 11), title, fill=(255, 255, 255))
    return result


def make_mask_and_proof() -> tuple[np.ndarray, dict[str, object]]:
    clean = np.asarray(Image.open(CLEANED_BASE).convert("RGB"), dtype=np.uint8)
    source = np.asarray(Image.open(SOURCE_MASK).convert("L"), dtype=np.uint8) >= 128
    if int(np.count_nonzero(source)) != 33668 or bbox(source) != [495, 1103, 707, 1441]:
        raise RuntimeError(f"unexpected former-head mask geometry: pixels={np.count_nonzero(source)} bbox={bbox(source)}")
    if np.any(clean[source] != MAGENTA):
        raise RuntimeError("cleaned-base pixels inside former-head mask are not uniformly magenta")
    MASK.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(source, 255, 0).astype(np.uint8), mode="L").save(MASK, optimize=True)
    Image.fromarray(np.where(~source, 255, 0).astype(np.uint8), mode="L").save(PROTECTED, optimize=True)

    tinted = clean.copy()
    tinted[source] = np.clip(np.round(tinted[source] * 0.25 + np.array([0, 230, 255]) * 0.75), 0, 255).astype(np.uint8)
    crop = (440, 1030, 760, 1455)
    pieces = [
        panel(Image.fromarray(clean).crop(crop), "CLEANED BASE / SOURCE-HIDDEN HOLE", (420, 620)),
        panel(Image.fromarray(tinted).crop(crop), "ONLY CYAN REGION MAY CHANGE", (420, 620)),
        panel(Image.fromarray(np.repeat((source * 255)[..., None], 3, axis=2).astype(np.uint8)).crop(crop), "33,668-PIXEL TIGHT MASK", (420, 620)),
    ]
    sheet = Image.new("RGB", (1260, 658), (12, 15, 20))
    for index, item in enumerate(pieces):
        sheet.paste(item, (index * 420, 0))
    MASK_PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(MASK_PROOF, optimize=True)
    return source, {"pixels": 33668, "bbox": [495, 1103, 707, 1441]}


def make_result_proof(clean: np.ndarray, aligned: np.ndarray, composite: np.ndarray, mask: np.ndarray) -> None:
    crop = (440, 1030, 760, 1455)
    mask_overlay = clean.copy()
    mask_overlay[mask] = np.clip(np.round(mask_overlay[mask] * 0.25 + np.array([0, 230, 255]) * 0.75), 0, 255).astype(np.uint8)
    pieces = [
        panel(Image.fromarray(clean).crop(crop), "CLEANED TARGET", (360, 620)),
        panel(Image.fromarray(mask_overlay).crop(crop), "AUTHORIZED MASK", (360, 620)),
        panel(Image.fromarray(aligned).crop(crop), "RAW MODEL OUTPUT ALIGNED", (360, 620)),
        panel(Image.fromarray(composite).crop(crop), "HARD COMPOSITE", (360, 620)),
    ]
    sheet = Image.new("RGB", (1440, 658), (12, 15, 20))
    for index, item in enumerate(pieces):
        sheet.paste(item, (index * 360, 0))
    PROOF.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(PROOF, optimize=True)


def main() -> None:
    for path, expected in EXPECTED.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"hash fence failed: {path}")
    archive = archive_v8()
    mask, mask_record = make_mask_and_proof()
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_PATH.write_text(PROMPT + "\n", encoding="utf-8")

    for sentinel in (REQUEST, RAW, HEADERS, ALIGNED, COMPOSITE, METADATA):
        if sentinel.exists():
            raise RuntimeError(f"authorized one-call artifact already exists; refusing retry: {sentinel}")

    client = GatewayClient(GATEWAY, timeout_seconds=1800)
    health = client.health()
    write_json(HEALTH, health)
    edit = health.get("pipelines", {}).get("edit", {})
    if not edit.get("available") or "Qwen" not in str(edit.get("model")):
        raise RuntimeError(f"Qwen edit pipeline unavailable: {edit}")

    payload = {
        "prompt": PROMPT,
        "image_base64": encode_image(CLEANED_BASE),
        "image2_base64": encode_image(ACTIVE_PROFILE),
        "image3_base64": encode_image(ANATOMY),
        "mask_base64": encode_image(MASK),
        "seed": SEED,
        "steps": STEPS,
        "cfg": CFG,
        "denoise": DENOISE,
        "filename_prefix": "mirelle_concerned_lower_body_inpaint_v1",
    }
    request_record = {"method": "POST", "gateway": GATEWAY, "endpoint": ENDPOINT, "payload": payload}
    write_json(REQUEST, request_record)
    content, headers = client.post_image(ENDPOINT, payload)

    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(content)
    write_json(HEADERS, headers)
    with Image.open(RAW) as image:
        raw_format = image.format
        raw_mode = image.mode
        raw_size = image.size
        raw_rgb = image.convert("RGB")
    with Image.open(CLEANED_BASE) as image:
        clean_image = image.convert("RGB")
    aligned_image = raw_rgb.resize(clean_image.size, Image.Resampling.LANCZOS)
    ALIGNED.parent.mkdir(parents=True, exist_ok=True)
    aligned_image.save(ALIGNED, optimize=True)

    clean = np.asarray(clean_image, dtype=np.uint8)
    aligned = np.asarray(aligned_image, dtype=np.uint8)
    composite = clean.copy()
    composite[mask] = aligned[mask]
    outside_delta = int(np.count_nonzero(np.any(composite != clean, axis=2) & ~mask))
    if outside_delta:
        raise RuntimeError(f"hard composite changed {outside_delta} outside-mask pixels")
    COMPOSITE.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite, mode="RGB").save(COMPOSITE, optimize=True)
    make_result_proof(clean, aligned, composite, mask)

    metadata = {
        "status": "pending_native_visual_review_before_metal_overlay",
        "model_calls": 1,
        "archive": archive,
        "validation_contract": {
            "source_hidden_lower_body_not_exact": True,
            "only_visible_unaffected_pixels_outside_mask_are_exact": True,
            "outside_mask_delta_pixels": outside_delta,
        },
        "inputs": {
            "semantic_base": {"path": str(SEMANTIC_BASE), "sha256": sha256(SEMANTIC_BASE)},
            "cleaned_edit_target": {"path": str(CLEANED_BASE), "sha256": sha256(CLEANED_BASE)},
            "active_profile_support": {"path": str(ACTIVE_PROFILE), "sha256": sha256(ACTIVE_PROFILE)},
            "anatomy_support": {"path": str(ANATOMY), "sha256": sha256(ANATOMY)},
        },
        "mask": {"path": str(MASK), "sha256": sha256(MASK), **mask_record},
        "request": {"path": str(REQUEST), "sha256": sha256(REQUEST), "seed": SEED, "steps": STEPS, "cfg": CFG, "denoise": DENOISE},
        "response": {
            "headers_path": str(HEADERS), "headers_sha256": sha256(HEADERS),
            "raw_path": str(RAW), "raw_sha256": sha256(RAW), "raw_format": raw_format, "raw_mode": raw_mode, "raw_dimensions": list(raw_size),
            "aligned_path": str(ALIGNED), "aligned_sha256": sha256(ALIGNED), "aligned_dimensions": list(aligned_image.size),
            "hard_composite_path": str(COMPOSITE), "hard_composite_sha256": sha256(COMPOSITE), "hard_composite_dimensions": list(clean_image.size),
            "changed_inside_mask_pixels": int(np.count_nonzero(np.any(composite != clean, axis=2) & mask)),
            "changed_outside_mask_pixels": outside_delta,
        },
        "proofs": {"mask": {"path": str(MASK_PROOF), "sha256": sha256(MASK_PROOF)}, "result": {"path": str(PROOF), "sha256": sha256(PROOF)}},
        "metal_overlay": "deferred until native visual pass; no model retry allowed",
    }
    write_json(METADATA, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
