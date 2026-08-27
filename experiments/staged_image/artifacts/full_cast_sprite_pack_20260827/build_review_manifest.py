#!/usr/bin/env python3
"""Build the full-cast experiment manifest and manual-review tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
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
    "wren": {
        "name": "Wren",
        "kind": "seeded_human",
        "processing": "physical_magenta_matte_v1",
        "references": [
            "app/storage/stories/one_star_ascension_s1/visual-references/wren_thelantern.png"
        ],
        "prohibited": [
            "app/storage/stories/one_star_ascension_s1/visual-references/locked/wren_thelantern/active_profile.png"
        ],
    },
    "veiled_masculine": {
        "name": "Veiled masculine one-star default",
        "kind": "generic_veiled_default",
        "processing": "physical_magenta_matte_v1",
        "references": [],
        "prohibited": [],
    },
    "veiled_feminine": {
        "name": "Veiled feminine one-star default",
        "kind": "generic_veiled_default",
        "processing": "physical_magenta_matte_v1",
        "references": [],
        "prohibited": [],
    },
    "aveline_morcant": {
        "name": "Aveline Morcant",
        "kind": "seeded_human",
        "processing": "physical_magenta_matte_v1",
    },
    "castor_valebrand": {
        "name": "Castor Valebrand",
        "kind": "seeded_human",
        "processing": "opaque_hybrid_v1",
    },
    "halcyon_of_the_gilded_march": {
        "name": "Halcyon of the Gilded March",
        "kind": "seeded_human",
        "processing": "physical_magenta_matte_v1",
    },
    "iselle_the_guide": {
        "name": "Iselle the Guide",
        "kind": "seeded_fairy",
        "processing": "connected_key_v3_translucent_color_safe",
    },
    "liora_fen": {
        "name": "Liora Fen",
        "kind": "seeded_human",
        "processing": "physical_magenta_matte_v1",
    },
    "renna_holt": {
        "name": "Renna Holt",
        "kind": "seeded_human",
        "processing": "physical_magenta_matte_v1",
    },
    "seris_nightglass": {
        "name": "Seris Nightglass",
        "kind": "seeded_human",
        "processing": "opaque_hybrid_v1",
    },
    "soren_ironvow": {
        "name": "Soren Ironvow",
        "kind": "seeded_human",
        "processing": "opaque_hybrid_v1",
        "references": [
            "app/storage/stories/one_star_ascension_s1/visual-references/locked/soren_ironvow/identity_base.png",
            "app/storage/stories/one_star_ascension_s1/visual-references/locked/soren_ironvow/facial_zoom.png",
            "app/storage/stories/one_star_ascension_s1/visual-references/locked/soren_ironvow/anatomy.png",
            "app/storage/stories/one_star_ascension_s1/visual-references/locked/soren_ironvow/back_view.png",
        ],
        "prohibited": [
            "app/storage/stories/one_star_ascension_s1/visual-references/locked/soren_ironvow/active_profile.png",
            "app/storage/stories/one_star_ascension_s1/visual-references/soren_ironvow.webp",
        ],
    },
    "veil_the_unnumbered": {
        "name": "Veil the Unnumbered",
        "kind": "seeded_human",
        "processing": "opaque_hybrid_v1",
    },
    "warden_of_the_eighth": {
        "name": "Warden of the Eighth",
        "kind": "seeded_nonhuman",
        "processing": "opaque_hybrid_v1",
    },
}

RAW_OVERRIDES = {
    ("castor_valebrand", "concerned"): "concerned_chroma_v2.png",
    ("halcyon_of_the_gilded_march", "sad"): "sad_chroma_v2.png",
    ("renna_holt", "happy"): "happy_chroma_v2.png",
    ("renna_holt", "sad"): "sad_chroma_v2.png",
    ("renna_holt", "skeptical"): "skeptical_chroma_v2.png",
    ("warden_of_the_eighth", "surprised"): "surprised_chroma_v2.png",
}

SPECIAL_NOTES = {
    ("castor_valebrand", "concerned"): "Selected targeted v2; v1 guard geometry is preserved as a rejection.",
    ("halcyon_of_the_gilded_march", "sad"): "Selected v2 fixes the rejected v1 hand defect.",
    ("iselle_the_guide", "tense"): "Selected retry replaces the preserved extra-arm generation failure.",
    ("renna_holt", "happy"): "Selected v2 replaces malformed-bow v1.",
    ("renna_holt", "sad"): "Selected v2 replaces malformed anatomy/prop v1.",
    ("renna_holt", "skeptical"): "Selected prop-shape retry; current smooth recurve is accepted at full resolution.",
    ("warden_of_the_eighth", "surprised"): "Selected v2 visibly attaches the single bead-and-orb chain to one raised hand.",
}

REJECTIONS = [
    {
        "path": "generation_raw/iselle_the_guide/rejected/tense_chroma_v1_rejected_extra_arm.png",
        "reason": "Extra articulated arm; regenerated once with the same identity contract.",
    },
    {
        "path": "generation_raw/warden_of_the_eighth/rejected/surprised_chroma_v1_rejected_floating_prop.png",
        "reason": "Orb and chain were completely airborne with no visible retained attachment.",
    },
    {
        "path": "generation_raw/halcyon_of_the_gilded_march/sad_chroma_v1_rejected.png",
        "reason": "Malformed hand; sad v2 selected.",
    },
    {
        "path": "generation_raw/renna_holt/happy_chroma_v1_rejected.png",
        "reason": "Malformed bow geometry; happy v2 selected.",
    },
    {
        "path": "generation_raw/renna_holt/sad_chroma_v1_rejected.png",
        "reason": "Malformed anatomy/prop relationship; sad v2 selected.",
    },
    {
        "path": "generation_raw/renna_holt/skeptical_chroma_v1_preserved_pilot.png",
        "reason": "Pilot bow had an incoherent limb path; targeted skeptical retry selected.",
    },
    {
        "path": "contact_sheets/rejected/iselle_the_guide_complete_sweep_physical_unmix_v1.png",
        "reason": "Global unmix erased intentional red eyes, pink tie, and mauve wing color.",
    },
    {
        "path": "contact_sheets/rejected/iselle_the_guide_complete_sweep_connected_v2.png",
        "reason": "Subject color restored but hot-magenta edge/gap contamination remained.",
    },
    {
        "path": "contact_sheets/rejected/castor_valebrand_complete_sweep_physical_unmix_v1.png",
        "reason": "Global unmix changed canonical burgundy lining and sword-guard gems to brown.",
    },
    {
        "path": "contact_sheets/rejected/castor_valebrand_complete_sweep_connected_key_v3.png",
        "reason": "Interior color restored but neon-magenta edge contours remained.",
    },
    {
        "path": "contact_sheets/rejected/seris_nightglass_complete_sweep_physical_unmix_v1.png",
        "reason": "Global unmix erased canonical purple, lavender, and burgundy subject colors.",
    },
    {
        "path": "contact_sheets/rejected/seris_nightglass_complete_sweep_connected_key_v3.png",
        "reason": "Interior color restored but neon-magenta edge contours remained.",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def image_record(path: Path, analyze_components: bool = False) -> dict[str, object]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        record: dict[str, object] = {
            "path": relative(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
        }
        if "A" in image.getbands():
            alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
            record["alpha"] = {
                "minimum": int(alpha.min()),
                "maximum": int(alpha.max()),
                "transparent_pixels": int(np.count_nonzero(alpha == 0)),
                "opaque_pixels": int(np.count_nonzero(alpha == 255)),
                "partial_pixels": int(np.count_nonzero((alpha > 0) & (alpha < 255))),
            }
            if analyze_components:
                count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                    (alpha >= 64).astype(np.uint8), connectivity=8
                )
                significant = [
                    int(stats[index, cv2.CC_STAT_AREA])
                    for index in range(1, count)
                    if int(stats[index, cv2.CC_STAT_AREA]) >= 100
                ]
                record["significant_alpha_components"] = {
                    "threshold": 64,
                    "minimum_area": 100,
                    "count": len(significant),
                    "areas": sorted(significant, reverse=True),
                }
        return record


def file_record(path: Path) -> dict[str, object]:
    return {"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def reference_record(path_text: str) -> dict[str, object]:
    path = REPO_ROOT / path_text
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return {
            "path": path_text,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
        }


def default_seeded_references(character: str) -> list[str]:
    base = (
        "app/storage/stories/one_star_ascension_s1/visual-references/locked/"
        + character
    )
    return [
        f"{base}/active_profile.png",
        f"{base}/facial_zoom.png",
        f"{base}/anatomy.png",
        f"{base}/back_view.png",
    ]


def load_prompt_ledger() -> dict[str, list[dict[str, object]]]:
    ledger: dict[str, list[dict[str, object]]] = {}
    path = ROOT / "exact_generation_prompt_ledger.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        destination = item.get("destination")
        if destination:
            ledger.setdefault(str(destination), []).append(item)
    return ledger


def prompt_record(raw_path: Path, ledger: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    destination = str(raw_path.relative_to(REPO_ROOT))
    entries = ledger.get(destination, [])
    raw_hash = sha256(raw_path)
    matching = [
        item
        for item in entries
        if item.get("destination_metadata", {}).get("sha256") == raw_hash
        and item.get("prompt_exact") is True
    ]
    if not matching:
        raise RuntimeError(f"no exact prompt-ledger entry for {destination} {raw_hash}")
    item = matching[-1]
    return {
        "prompt_sha256": item["prompt_sha256"],
        "prompt": item["prompt"],
        "references": item["references"],
        "source_rollout": item["source_rollout"],
        "source_line": item["source_line"],
        "generated_source": item["generated_source"],
        "source_destination_hash_match": item["source_destination_hash_match"],
    }


def review_note(character: str, label: str) -> str:
    note = SPECIAL_NOTES.get((character, label))
    if note:
        return note
    if character == "wren":
        return "Top-level Wren source only; no kneepads; one lantern and one sheathed sword."
    if character.startswith("veiled_"):
        return "Generic, prop-free, non-seeded identity; entire face remains unreadable."
    if character == "iselle_the_guide":
        return "Exactly four mauve wings; canonical red eyes and pink tie preserved."
    if character == "soren_ironvow":
        return "Allowed repaired references only; pale scar remains on viewer-right/character-left."
    if character == "warden_of_the_eighth":
        return "Exactly two articulated arms, four load-bearing legs, and one bead-and-orb chain assembly."
    return "Root-reviewed selected experimental candidate; no regeneration selected."


def build_manual_review() -> str:
    columns = [
        "Character",
        "Label",
        "Identity",
        "Face / focal feature",
        "Hair / silhouette",
        "Outfit / body design",
        "Anatomy / hands",
        "Weapon / prop",
        "Transparency / edges",
        "Pose legibility",
        "Framing / baseline",
        "Semantic distinction",
        "Disposition",
        "Notes",
    ]
    lines = [
        "# Full-cast manual sprite review",
        "",
        "`P` means the root reviewer passed the selected experimental candidate at original resolution. "
        "The rubric is human-authored; no runtime or authoring-time vision model supplied these judgments.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for character, spec in CHARACTERS.items():
        for label in LABELS:
            values = [
                str(spec["name"]),
                label,
                "P",
                "P",
                "P",
                "P",
                "P",
                "P",
                "P",
                "P",
                "P",
                "P",
                "accepted experimental candidate",
                review_note(character, label),
            ]
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Pack-level disposition",
            "",
            "All 104 selected variants passed the final original-resolution review. "
            "This approves them only as experimental review candidates. Nothing in this pack is production-bound or locked.",
            "",
        ]
    )
    return "\n".join(lines)


def build_rejection_log(rejections: list[dict[str, object]]) -> str:
    lines = [
        "# Preserved rejection log",
        "",
        "Rejected generation and matte passes remain in the pack so the selected results are auditable.",
        "",
        "| Artifact | SHA-256 | Reason |",
        "| --- | --- | --- |",
    ]
    for item in rejections:
        lines.append(
            f"| `{item['asset']['path']}` | `{item['asset']['sha256']}` | {item['reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ledger = load_prompt_ledger()
    characters: dict[str, object] = {}
    for character, spec in CHARACTERS.items():
        references = spec.get("references")
        if references is None:
            references = default_seeded_references(character)
        prohibited = spec.get("prohibited", [])
        variants: dict[str, object] = {}
        for label in LABELS:
            raw_name = RAW_OVERRIDES.get((character, label), f"{label}_chroma_v1.png")
            raw_path = ROOT / "generation_raw" / character / raw_name
            candidate_path = ROOT / "candidates" / character / f"{label}.png"
            sprite_path = ROOT / "sprites" / character / f"{label}.png"
            for path in (raw_path, candidate_path, sprite_path):
                if not path.exists():
                    raise FileNotFoundError(path)
            variants[label] = {
                "raw": image_record(raw_path),
                "generation": prompt_record(raw_path, ledger),
                "candidate": image_record(candidate_path, analyze_components=True),
                "sprite": image_record(sprite_path, analyze_components=True),
                "manual_review": {
                    "disposition": "accepted_experimental_candidate",
                    "note": review_note(character, label),
                },
            }
        sheet = ROOT / "contact_sheets" / f"{character}_complete_sweep.png"
        characters[character] = {
            "name": spec["name"],
            "kind": spec["kind"],
            "processing_method": spec["processing"],
            "permitted_references": [reference_record(path) for path in references],
            "prohibited_references": [reference_record(path) for path in prohibited],
            "variants": variants,
            "contact_sheet": image_record(sheet),
            "root_disposition": "accepted_8_of_8_experimental_candidates",
        }

    rejection_records: list[dict[str, object]] = []
    for rejection in REJECTIONS:
        path = ROOT / str(rejection["path"])
        if not path.exists():
            continue
        rejection_records.append(
            {"asset": image_record(path), "reason": rejection["reason"]}
        )

    processor_paths = [
        ROOT / "prepare_review_sprites.py",
        ROOT / "prepare_subject_color_review_sprites.py",
        ROOT / "prepare_opaque_hybrid_review_sprites.py",
        ROOT / "prepare_castor_opaque_hybrid_review_sprites.py",
        ROOT / "build_complete_cast_overview.py",
        ROOT / "recover_generation_provenance.py",
    ]
    manifest = {
        "schema_version": 1,
        "pack_id": "full_cast_sprite_pack_20260827",
        "status": "experimental_not_production_bound_or_locked",
        "runtime_image_reading": False,
        "character_count": len(CHARACTERS),
        "labels": LABELS,
        "selected_variant_count": len(CHARACTERS) * len(LABELS),
        "normalized_canvas": {
            "width": 1100,
            "height": 1500,
            "mode": "RGBA",
            "baseline_y": 1480,
            "target_height": 1420,
            "maximum_width": 1060,
        },
        "prompt_ledger": file_record(ROOT / "exact_generation_prompt_ledger.jsonl"),
        "prompt_conflict_audit": file_record(ROOT / "prompt_conflict_audit.md"),
        "overview": image_record(ROOT / "complete_cast_pose_expression_overview.png"),
        "overview_metadata": file_record(ROOT / "complete_cast_pose_expression_overview.json"),
        "processors": [file_record(path) for path in processor_paths if path.exists()],
        "characters": characters,
        "preserved_rejections": rejection_records,
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "manual_review.md").write_text(build_manual_review(), encoding="utf-8")
    (ROOT / "rejection_log.md").write_text(
        build_rejection_log(rejection_records), encoding="utf-8"
    )
    print(f"manifest={ROOT / 'manifest.json'} sha256={sha256(ROOT / 'manifest.json')}")
    print(f"manual_review={ROOT / 'manual_review.md'} sha256={sha256(ROOT / 'manual_review.md')}")
    print(f"rejection_log={ROOT / 'rejection_log.md'} sha256={sha256(ROOT / 'rejection_log.md')}")
    print(f"characters={len(CHARACTERS)} variants={len(CHARACTERS) * len(LABELS)}")


if __name__ == "__main__":
    main()
