#!/usr/bin/env python3
"""Build and validate the experimental Mirelle/Rowan v2 review pack.

This is an authoring-time provenance tool.  It does not bind assets to Ayoa's
runtime or reference registry, and no runtime model reads an image.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
OLD_PACK = ROOT.parent / "mirelle_rowan_sprite_pack_20260826"
FULL_CAST = ROOT.parent / "full_cast_sprite_pack_20260827"
LABELS = [
    "neutral",
    "happy",
    "concerned",
    "tense",
    "skeptical",
    "angry",
    "sad",
    "surprised",
]

CHARACTERS = {
    "mirelle_voss": {
        "name": "Mirelle Voss",
        "selected_raw": {
            "neutral": OLD_PACK / "generation_raw/mirelle_voss/neutral_chroma_v1.png",
            "happy": OLD_PACK / "generation_raw/mirelle_voss/happy_chroma_v1.png",
            "concerned": ROOT / "generation_raw/mirelle_voss/concerned_chroma_v3.png",
            "tense": ROOT / "generation_raw/mirelle_voss/tense_chroma_v3.png",
            "skeptical": OLD_PACK / "generation_raw/mirelle_voss/skeptical_chroma_v1.png",
            "angry": OLD_PACK / "generation_raw/mirelle_voss/angry_chroma_v3.png",
            "sad": OLD_PACK / "generation_raw/mirelle_voss/sad_chroma_v1.png",
            "surprised": OLD_PACK / "generation_raw/mirelle_voss/surprised_chroma_v1.png",
        },
        "selected_prompt": {
            "neutral": ROOT / "prompts/legacy_selected_bases/mirelle_voss/neutral_v1.txt",
            "happy": ROOT / "prompts/legacy_selected_bases/mirelle_voss/happy_v1.txt",
            "concerned": ROOT / "prompts/imagegen/mirelle_concerned_chroma_v3.txt",
            "tense": ROOT / "prompts/recovered_v2/mirelle_voss/tense_v3.txt",
            "skeptical": ROOT / "prompts/legacy_selected_bases/mirelle_voss/skeptical_v1.txt",
            "angry": ROOT / "prompts/legacy_selected_bases/mirelle_voss/angry_chroma_edit_v3.txt",
            "sad": ROOT / "prompts/legacy_selected_bases/mirelle_voss/sad_v1.txt",
            "surprised": ROOT / "prompts/legacy_selected_bases/mirelle_voss/surprised_v1.txt",
        },
        "selected_repair": {
            "neutral": ROOT / "grafts/mirelle_voss/neutral_canonical_metal_repair_v1.png",
            "happy": ROOT / "grafts/mirelle_voss/happy_upper_primary_metal_reference_graft_v2.png",
            "tense": ROOT / "grafts/mirelle_voss/tense_primary_metal_reference_graft_v2.png",
            "skeptical": ROOT / "grafts/mirelle_voss/skeptical_primary_metal_reference_graft_v1.png",
            "angry": ROOT / "grafts/mirelle_voss/angry_primary_metal_reference_graft_v1.png",
            "sad": ROOT / "grafts/mirelle_voss/sad_canonical_metal_repair_v1.png",
            "surprised": ROOT / "grafts/mirelle_voss/surprised_primary_metal_reference_graft_v1.png",
        },
        "contact_sheet": ROOT / "contact_sheets/mirelle_voss_complete_sweep_v2.png",
        "processing_report": ROOT / "matte_reports/mirelle_voss_v2_opaque_hybrid.json",
    },
    "rowan_kest": {
        "name": "Rowan Kest",
        "selected_raw": {
            "neutral": OLD_PACK / "generation_raw/rowan_kest/neutral_chroma_v1.png",
            "happy": OLD_PACK / "generation_raw/rowan_kest/happy_chroma_v1.png",
            "concerned": OLD_PACK / "generation_raw/rowan_kest/concerned_chroma_v1.png",
            "tense": OLD_PACK / "generation_raw/rowan_kest/tense_chroma_v1.png",
            "skeptical": OLD_PACK / "generation_raw/rowan_kest/skeptical_chroma_v3.png",
            "angry": OLD_PACK / "generation_raw/rowan_kest/angry_chroma_v1.png",
            "sad": ROOT / "generation_raw/rowan_kest/sad_chroma_v2.png",
            "surprised": OLD_PACK / "generation_raw/rowan_kest/surprised_chroma_v1.png",
        },
        "selected_prompt": {
            "neutral": ROOT / "prompts/legacy_selected_bases/rowan_kest/neutral_v1.txt",
            "happy": ROOT / "prompts/legacy_selected_bases/rowan_kest/happy_v1.txt",
            "concerned": ROOT / "prompts/legacy_selected_bases/rowan_kest/concerned_v1.txt",
            "tense": ROOT / "prompts/legacy_selected_bases/rowan_kest/tense_v1.txt",
            "skeptical": ROOT / "prompts/legacy_selected_bases/rowan_kest/skeptical_chroma_edit_v3.txt",
            "angry": ROOT / "prompts/legacy_selected_bases/rowan_kest/angry_v1.txt",
            "sad": ROOT / "prompts/recovered_v2/rowan_kest/sad_v2.txt",
            "surprised": ROOT / "prompts/legacy_selected_bases/rowan_kest/surprised_v1.txt",
        },
        "selected_repair": {},
        "contact_sheet": ROOT / "contact_sheets/rowan_kest_complete_sweep_v2.png",
        "processing_report": ROOT / "matte_reports/rowan_kest_v2.json",
    },
}

REFERENCES = {
    character: [
        REPO_ROOT
        / f"app/storage/stories/one_star_ascension_s1/visual-references/locked/{character}/{view}.png"
        for view in ("active_profile", "facial_zoom", "anatomy", "back_view")
    ]
    for character in CHARACTERS
}

RECOVERED_DESTINATIONS = {
    "generation_raw/mirelle_voss/concerned_chroma_v2.png": ROOT
    / "prompts/recovered_v2/mirelle_voss/concerned_v2.txt",
    "generation_raw/mirelle_voss/tense_chroma_v2.png": ROOT
    / "prompts/recovered_v2/mirelle_voss/tense_v2_rejected.txt",
    "generation_raw/mirelle_voss/tense_chroma_v3.png": ROOT
    / "prompts/recovered_v2/mirelle_voss/tense_v3.txt",
    "generation_raw/rowan_kest/sad_chroma_v2.png": ROOT
    / "prompts/recovered_v2/rowan_kest/sad_v2.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def pack_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record: dict[str, Any] = {
        "path": repo_relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            record.update(
                {
                    "format": image.format,
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                }
            )
            if "A" in image.getbands():
                alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
                record["alpha"] = {
                    "minimum": int(alpha.min()),
                    "maximum": int(alpha.max()),
                    "transparent": int(np.count_nonzero(alpha == 0)),
                    "partial": int(np.count_nonzero((alpha > 0) & (alpha < 255))),
                    "opaque": int(np.count_nonzero(alpha == 255)),
                }
                count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                    (alpha >= 64).astype(np.uint8), connectivity=8
                )
                areas = [
                    int(stats[index, cv2.CC_STAT_AREA])
                    for index in range(1, count)
                    if int(stats[index, cv2.CC_STAT_AREA]) >= 100
                ]
                record["significant_alpha_components"] = {
                    "threshold": 64,
                    "minimum_area": 100,
                    "count": len(areas),
                    "areas": sorted(areas, reverse=True),
                }
    except (OSError, ValueError):
        pass
    return record


def recover_exact_v2_prompts() -> list[dict[str, Any]]:
    ledger_path = FULL_CAST / "exact_generation_prompt_ledger.jsonl"
    matched: dict[str, dict[str, Any]] = {}
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        destination = record.get("destination") or ""
        for suffix in RECOVERED_DESTINATIONS:
            if destination.endswith(suffix):
                matched[suffix] = record
    if set(matched) != set(RECOVERED_DESTINATIONS):
        missing = sorted(set(RECOVERED_DESTINATIONS) - set(matched))
        raise RuntimeError(f"missing recovered prompt records: {missing}")

    output_records: list[dict[str, Any]] = []
    for suffix, prompt_path in RECOVERED_DESTINATIONS.items():
        record = matched[suffix]
        prompt = record["prompt"]
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != record["prompt_sha256"]:
            raise RuntimeError(f"recovered prompt hash mismatch: {suffix}")
        destination = ROOT / suffix
        if sha256(destination) != record["destination_metadata"]["sha256"]:
            raise RuntimeError(f"recovered output hash mismatch: {destination}")
        if not record.get("source_destination_hash_match"):
            raise RuntimeError(f"source/destination mismatch in recovered record: {suffix}")
        output_records.append(record)

    # The final concerned v3 fallback was generated after the historical
    # rollout ledger was recovered.  Its call-time metadata records the exact
    # prompt, five input roles/hashes, built-in original, and workspace copy.
    v3_metadata_path = ROOT / "component_metadata/mirelle_concerned_chroma_v3_generation.json"
    v3_metadata = json.loads(v3_metadata_path.read_text(encoding="utf-8"))
    v3_prompt_path = REPO_ROOT / v3_metadata["prompt"]["path"]
    v3_prompt = v3_prompt_path.read_text(encoding="utf-8")
    if hashlib.sha256(v3_prompt.encode("utf-8")).hexdigest() != v3_metadata["prompt"]["sha256"]:
        raise RuntimeError("concerned v3 exact prompt hash mismatch")
    v3_output = REPO_ROOT / v3_metadata["output"]["workspace_path"]
    v3_builtin = Path(v3_metadata["output"]["built_in_original_path"])
    if sha256(v3_output) != v3_metadata["output"]["sha256"]:
        raise RuntimeError("concerned v3 workspace output hash mismatch")
    if not v3_builtin.is_file() or sha256(v3_builtin) != v3_metadata["output"]["sha256"]:
        raise RuntimeError("concerned v3 built-in original is missing or changed")
    output_records.append(
        {
            "association": "call_time_metadata",
            "destination": v3_metadata["output"]["workspace_path"],
            "destination_metadata": {
                "path": v3_metadata["output"]["workspace_path"],
                "sha256": v3_metadata["output"]["sha256"],
                "format": v3_metadata["output"]["format"],
                "mode": v3_metadata["output"]["mode"],
                "width": v3_metadata["output"]["dimensions"][0],
                "height": v3_metadata["output"]["dimensions"][1],
                "bytes": v3_output.stat().st_size,
            },
            "generated_source": v3_metadata["output"]["built_in_original_path"],
            "generated_source_sha256": sha256(v3_builtin),
            "prompt": v3_prompt,
            "prompt_sha256": v3_metadata["prompt"]["sha256"],
            "prompt_exact": True,
            "references": v3_metadata["input_roles"],
            "source_destination_hash_match": True,
            "source_metadata": pack_relative(v3_metadata_path),
        }
    )

    output_records.sort(key=lambda item: item["destination"])
    output = ROOT / "exact_generation_prompt_ledger.jsonl"
    output.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in output_records),
        encoding="utf-8",
    )
    return output_records


def make_overview() -> Path:
    sheets = [Image.open(CHARACTERS[key]["contact_sheet"]).convert("RGB") for key in CHARACTERS]
    width = max(image.width for image in sheets)
    header = 70
    gap = 18
    output = Image.new("RGB", (width, header + sum(i.height for i in sheets) + gap), (9, 12, 17))
    draw = ImageDraw.Draw(output)
    draw.text((22, 20), "MIRELLE VOSS + ROWAN KEST - V2 POSE / EXPRESSION REVIEW", fill=(255, 255, 255))
    y = header
    for index, image in enumerate(sheets):
        output.paste(image, ((width - image.width) // 2, y))
        y += image.height
        if index == 0:
            y += gap
    path = ROOT / "contact_sheets/mirelle_rowan_v2_complete_overview.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path, optimize=True)
    return path


def old_pack_aggregate() -> str:
    digest_lines = []
    for path in sorted(path for path in OLD_PACK.rglob("*") if path.is_file()):
        # Match the frozen shell command exactly: it was run from the repo root
        # with a repo-relative pack argument, so sha256sum emitted relative paths.
        digest_lines.append(f"{sha256(path)}  {repo_relative(path)}\n")
    return hashlib.sha256("".join(digest_lines).encode("utf-8")).hexdigest()


def selected_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for character, config in CHARACTERS.items():
        for label in LABELS:
            raw = config["selected_raw"][label]
            prompt = config["selected_prompt"][label]
            candidate = ROOT / f"candidates/{character}/{label}.png"
            sprite = ROOT / f"sprites/{character}/{label}.png"
            repair_path = config["selected_repair"].get(label)
            repair = file_record(repair_path) if repair_path else None
            model_base = repair_path or raw
            prompt_bytes = prompt.read_bytes()
            record = {
                "character": character,
                "label": label,
                "selected_raw": file_record(raw),
                "selected_prompt": {
                    **file_record(prompt),
                    "prompt_text_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                },
                "selected_deterministic_repair": repair,
                "matte_input": file_record(model_base),
                "candidate": file_record(candidate),
                "sprite": file_record(sprite),
                "manual_disposition": "accepted_experimental_review_candidate",
            }
            sprite_image = record["sprite"]
            if (sprite_image.get("mode"), sprite_image.get("width"), sprite_image.get("height")) != (
                "RGBA",
                1100,
                1500,
            ):
                raise RuntimeError(f"invalid normalized sprite: {sprite}")
            alpha = sprite_image.get("alpha") or {}
            if alpha.get("minimum") != 0 or alpha.get("maximum") != 255 or not alpha.get("partial"):
                raise RuntimeError(f"invalid alpha: {sprite}")
            records.append(record)
    return records


def qwen_records() -> list[dict[str, Any]]:
    records = []
    for request in sorted((ROOT / "qwen_requests").glob("*.json")):
        if request.name.endswith("_health.json") or "health_before" in request.name:
            continue
        stem = request.stem.replace("_attempt_v1", "_v1")
        possible_review = ROOT / "qwen_reviews" / f"{stem}.json"
        possible_metadata = ROOT / "qwen_metadata" / f"{stem}.json"
        records.append(
            {
                "request": file_record(request),
                "review": file_record(possible_review) if possible_review.is_file() else None,
                "metadata": file_record(possible_metadata) if possible_metadata.is_file() else None,
                "selection": "rejected_or_superseded; deterministic repair retained",
            }
        )
    return records


def make_inventory() -> Path:
    inventory = ROOT / "inventory.sha256"
    lines = []
    for path in sorted(path for path in ROOT.rglob("*") if path.is_file() and path != inventory):
        lines.append(f"{sha256(path)}  {pack_relative(path)}\n")
    inventory.write_text("".join(lines), encoding="utf-8")
    return inventory


def main() -> None:
    recovered = recover_exact_v2_prompts()
    overview = make_overview()
    selected = selected_records()

    expected_old = "6c47390015a9cac61d285ece0bbff59bcda7ea21bb1c83758c013f5932370b33"
    actual_old = old_pack_aggregate()
    if actual_old != expected_old:
        raise RuntimeError(f"old pack changed: {actual_old}")

    references = {
        character: [file_record(path) for path in paths]
        for character, paths in REFERENCES.items()
    }
    manifest = {
        "schema_version": 2,
        "pack_id": "mirelle-rowan-pose-expression-review-v2-20260827",
        "status": "complete experimental review pack; not approved or locked for production",
        "production_bound": False,
        "runtime_changes": False,
        "checkpoint_changes": False,
        "reference_registry_changes": False,
        "runtime_llm_image_reading": False,
        "labels": LABELS,
        "characters": {key: value["name"] for key, value in CHARACTERS.items()},
        "references": references,
        "selected_variants": selected,
        "contact_sheets": [
            file_record(CHARACTERS[key]["contact_sheet"]) for key in CHARACTERS
        ]
        + [file_record(overview)],
        "processing_reports": [
            file_record(CHARACTERS[key]["processing_report"]) for key in CHARACTERS
        ],
        "recovered_generation_ledger": {
            "record_count": len(recovered),
            "path": file_record(ROOT / "exact_generation_prompt_ledger.jsonl"),
            "all_source_destination_hashes_match": all(
                item.get("source_destination_hash_match") for item in recovered
            ),
        },
        "qwen_masked_edit_experiments": qwen_records(),
        "frozen_components": [
            file_record(ROOT / "components/mirelle_voss/canonical_primary_metal_rgba_v1.png"),
            file_record(ROOT / "components/mirelle_voss/canonical_primary_streamers_rgba_v1.png"),
        ],
        "old_pack_integrity": {
            "path": repo_relative(OLD_PACK),
            "expected_aggregate_sha256": expected_old,
            "actual_aggregate_sha256": actual_old,
            "byte_identical": True,
        },
        "review_documents": [
            file_record(ROOT / name)
            for name in (
                "README.md",
                "REVIEW_INDEX.md",
                "manual_review.md",
                "rejection_log.md",
                "prompt_conflict_audit.md",
            )
        ],
        "validation": {
            "selected_label_count": len(selected),
            "expected_selected_label_count": 16,
            "all_sprites_rgba_1100x1500": True,
            "all_sprites_have_transparent_partial_and_opaque_pixels": True,
            "all_png_files_readable": True,
            "old_pack_byte_identical": True,
        },
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for path in ROOT.rglob("*.png"):
        with Image.open(path) as image:
            image.verify()
    inventory = make_inventory()
    print(
        json.dumps(
            {
                "manifest": file_record(manifest_path),
                "overview": file_record(overview),
                "inventory": file_record(inventory),
                "selected": len(selected),
                "old_pack_aggregate": actual_old,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
