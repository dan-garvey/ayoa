#!/usr/bin/env python3
"""Validate the V3 forged-chain One-Star summon animation proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

import build_animation_forged_chain_v3 as builder
import validate_animation_release_v2 as v2_validation


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review_animation_forged_chain_v3"
KEYFRAMES = ROOT / "proofs/animation_forged_chain_v3/keyframes"
PROVENANCE = ROOT / "provenance"
MANIFEST = ROOT / "animation_forged_chain_v3_manifest.json"
BUILD_SCRIPT = ROOT / "tools/build_animation_forged_chain_v3.py"
DEFAULT_WINDOWS_REVIEW = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/"
    "OneStarPullRevealForgedChainV3_20260831"
)

EXPECTED_REVIEW_FILES = {
    "00_START_HERE.txt",
    "01_iron_plain_release_under_3.gif",
    "02_silver_lock_release_3_to_4.gif",
    "03_gold_forged_chainbreak_5_to_6.gif",
    "04_white_gold_forged_chainbreak_7.gif",
    "05_forged_chain_bound_break_flash.png",
    "SHA256SUMS.txt",
    "index.html",
    "manifest.json",
}

FORBIDDEN_PUBLIC_TOKENS = v2_validation.FORBIDDEN_PUBLIC_TOKENS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_hashes() -> dict[str, str]:
    paths = [path for path in sorted(REVIEW.iterdir()) if path.is_file()]
    paths.extend(sorted(KEYFRAMES.glob("*.png")))
    paths.extend(
        (
            MANIFEST,
            PROVENANCE / "animation_forged_chain_v3_source_hashes.json",
            PROVENANCE / "animation_forged_chain_v3_decisions.json",
            PROVENANCE / "animation_forged_chain_v3_SHA256SUMS.txt",
        )
    )
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in paths
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows-review",
        type=Path,
        default=DEFAULT_WINDOWS_REVIEW,
    )
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="Rebuild and require every V3 artifact hash to remain unchanged.",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            failures.append(f"{name}: {detail}")

    manifest = load_json(MANIFEST)
    check(
        "review manifest is canonical manifest",
        (REVIEW / "manifest.json").read_bytes() == MANIFEST.read_bytes(),
        sha256(MANIFEST),
    )
    check(
        "V3 forged-chain review contract",
        manifest["schema_version"] == 3
        and manifest["status"] == "visual_review_only"
        and manifest["direction"] == "outward_release_forged_chain"
        and manifest["supersedes_for_review"]
        == "outward_release_v2_chain_treatment"
        and manifest["production_binding"] is False
        and manifest["production_intent"]
        == "play_once_hold_flash_then_static_result",
        {
            key: manifest[key]
            for key in (
                "schema_version",
                "status",
                "direction",
                "supersedes_for_review",
                "production_binding",
                "production_intent",
            )
        },
    )
    check(
        "no file-size cap",
        manifest["file_size_cap_bytes"] is None,
        manifest["file_size_cap_bytes"],
    )
    chain_revision = manifest["chain_revision"]
    check(
        "forged-chain design contract",
        chain_revision["new_read"] == "heavy forged interlocking restraint"
        and chain_revision["link_orientation"]
        == "alternating face-on and edge-on"
        and chain_revision["surface"]
        == "dark bevel with restrained rank-metal edge"
        and chain_revision["placement"]
        == "taut across card face with contact shadow"
        and chain_revision["release"] == "coherent rigid halves"
        and chain_revision["fracture"] == "explicit open broken-link halves",
        chain_revision,
    )
    invariants = manifest["invariants"]
    check(
        "V2 release semantics preserved",
        invariants["outward_lock_motion"] is True
        and invariants["held_pre_reveal_flash"] is True
        and invariants["identity_before_static_result"] == "withheld"
        and invariants["exact_rank_before_static_result"] == "withheld"
        and invariants["semantic_fallback"] == "existing static result board"
        and invariants["animation_required_for_comprehension"] is False
        and invariants["v1_preserved"] is True
        and invariants["v2_preserved"] is True,
        invariants,
    )
    actual_review_files = {
        path.name for path in REVIEW.iterdir() if path.is_file()
    }
    check(
        "review set is exact",
        actual_review_files == EXPECTED_REVIEW_FILES,
        sorted(actual_review_files),
    )

    tiers = list(builder.TIERS)
    records = manifest["tiers"]
    check("four animation tiers", len(records) == 4, len(records))
    check(
        "tier file order",
        [record["filename"] for record in records]
        == [tier.filename for tier in tiers],
        [record["filename"] for record in records],
    )
    check(
        "duration and frame count rise with rarity",
        [record["frame_count"] for record in records]
        == sorted(record["frame_count"] for record in records)
        and [record["duration_ms"] for record in records]
        == sorted(record["duration_ms"] for record in records),
        {
            "frame_counts": [record["frame_count"] for record in records],
            "durations_ms": [record["duration_ms"] for record in records],
        },
    )
    geometries = [record["release_geometry"] for record in records]
    check(
        "chains only on gold and white-gold",
        [geometry["chain_count"] for geometry in geometries] == [0, 0, 2, 3],
        [geometry["chain_count"] for geometry in geometries],
    )
    check(
        "forged chains are heavy and interlocked",
        all(
            geometry["chain_visual_style"]
            == "heavy_forged_interlocking_links"
            and geometry["contact_shadow"] is True
            for geometry in geometries[2:]
        ),
        geometries[2:],
    )
    check(
        "chain halves release coherently",
        all(
            geometry["chain_segment_motion"] == "coherent_rigid_halves"
            and geometry["chain_motion"] == "split_and_eject_outward"
            for geometry in geometries[2:]
        ),
        geometries[2:],
    )
    check(
        "fractures use open snapped links",
        all(
            geometry["fracture_shape"] == "two_open_broken_link_halves"
            for geometry in geometries[2:]
        ),
        geometries[2:],
    )
    check(
        "locks still travel outward",
        all(
            geometry["lock_motion"] == "outward"
            and all(
                distance > geometry["lock_start_distance_px"]
                for distance in geometry["lock_end_distances_px"]
            )
            for geometry in geometries[1:]
        ),
        geometries,
    )
    check(
        "every tier ends on held flash",
        all(record["final_frame"] == "held_pre_reveal_flash" for record in records),
        [record["final_frame"] for record in records],
    )

    decoded: list[dict[str, Any]] = []
    for tier, record in zip(tiers, records, strict=True):
        path = REVIEW / tier.filename
        data = v2_validation.gif_data(path)
        decoded.append(data)
        check(
            f"{tier.key} GIF structure",
            data["format"] == "GIF"
            and data["size"] == [1024, 576]
            and data["animated"]
            and data["frame_count"] == tier.frame_count
            and data["loop"] == 0,
            {
                "format": data["format"],
                "size": data["size"],
                "animated": data["animated"],
                "frame_count": data["frame_count"],
                "loop": data["loop"],
            },
        )
        check(
            f"{tier.key} GIF timing",
            data["durations"][:-1] == [tier.frame_ms] * (tier.frame_count - 1)
            and data["durations"][-1] == tier.final_hold_ms
            and sum(data["durations"]) == record["duration_ms"],
            data["durations"],
        )
        check(
            f"{tier.key} has real motion",
            data["unique_frames"] >= tier.frame_count - 1
            and data["mean_motion"] > 0.2
            and data["max_motion"] > data["mean_motion"],
            {
                "unique_frames": data["unique_frames"],
                "mean_motion": round(data["mean_motion"], 4),
                "max_motion": round(data["max_motion"], 4),
            },
        )
        final_luma = data["frame_luma"][-1]
        check(
            f"{tier.key} final frame is brightness peak",
            final_luma >= max(data["frame_luma"]) - 0.01
            and final_luma > data["frame_luma"][0] + 35,
            {
                "start_luma": round(data["frame_luma"][0], 3),
                "final_luma": round(final_luma, 3),
                "max_luma": round(max(data["frame_luma"]), 3),
            },
        )
        check(
            f"{tier.key} manifest hash and bytes",
            record["sha256"] == sha256(path)
            and record["byte_count"] == path.stat().st_size,
            {
                "manifest_hash": record["sha256"],
                "actual_hash": sha256(path),
                "manifest_bytes": record["byte_count"],
                "actual_bytes": path.stat().st_size,
            },
        )

    v2_review = ROOT / "review_animation_release_v2"
    unchanged_pairs = (
        ("01_iron_plain_release_under_3.gif", "01_iron_plain_release_under_3.gif"),
        ("02_silver_lock_release_3_to_4.gif", "02_silver_lock_release_3_to_4.gif"),
    )
    check(
        "non-chain tiers are byte-identical to V2",
        all(
            (REVIEW / v3_name).read_bytes() == (v2_review / v2_name).read_bytes()
            for v3_name, v2_name in unchanged_pairs
        ),
        {
            v3_name: {
                "v3": sha256(REVIEW / v3_name),
                "v2": sha256(v2_review / v2_name),
            }
            for v3_name, v2_name in unchanged_pairs
        },
    )
    chain_pairs = (
        ("03_gold_forged_chainbreak_5_to_6.gif", "03_gold_chainbreak_5_to_6.gif"),
        ("04_white_gold_forged_chainbreak_7.gif", "04_white_gold_chainbreak_7.gif"),
    )
    check(
        "chain tiers differ from V2",
        all(
            sha256(REVIEW / v3_name) != sha256(v2_review / v2_name)
            for v3_name, v2_name in chain_pairs
        ),
        {
            v3_name: {
                "v3": sha256(REVIEW / v3_name),
                "v2": sha256(v2_review / v2_name),
            }
            for v3_name, v2_name in chain_pairs
        },
    )
    check(
        "flash brightness rises with rarity",
        [data["frame_luma"][-1] for data in decoded]
        == sorted(data["frame_luma"][-1] for data in decoded),
        [round(data["frame_luma"][-1], 3) for data in decoded],
    )

    expected_keyframes = {
        f"{tier.key}_{stage}.png"
        for tier in tiers
        for stage in ("bound", "break", "flash")
    }
    actual_keyframes = {path.name for path in KEYFRAMES.glob("*.png")}
    check(
        "exact bound break flash keyframes",
        actual_keyframes == expected_keyframes,
        sorted(actual_keyframes),
    )
    keyframes_valid = True
    keyframe_details: dict[str, Any] = {}
    for path in sorted(KEYFRAMES.glob("*.png")):
        with Image.open(path) as received:
            keyframe_details[path.name] = {
                "format": received.format,
                "mode": received.mode,
                "size": list(received.size),
            }
            keyframes_valid &= (
                received.format == "PNG"
                and received.mode == "RGB"
                and received.size == (1024, 576)
            )
    check("keyframe format", keyframes_valid, keyframe_details)

    for tier in tiers:
        bound = Image.open(KEYFRAMES / f"{tier.key}_bound.png").convert("L")
        broken = Image.open(KEYFRAMES / f"{tier.key}_break.png").convert("L")
        flash = Image.open(KEYFRAMES / f"{tier.key}_flash.png").convert("L")
        stage_detail = {
            "break_difference": round(
                ImageStat.Stat(ImageChops.difference(bound, broken)).mean[0],
                3,
            ),
            "bound_luma": round(ImageStat.Stat(bound).mean[0], 3),
            "flash_luma": round(ImageStat.Stat(flash).mean[0], 3),
        }
        check(
            f"{tier.key} visual stages differ",
            stage_detail["break_difference"] > 1.0
            and stage_detail["flash_luma"] > stage_detail["bound_luma"] + 35,
            stage_detail,
        )
        bound.close()
        broken.close()
        flash.close()

    storyboard = REVIEW / "05_forged_chain_bound_break_flash.png"
    with Image.open(storyboard) as received:
        storyboard_detail = {
            "format": received.format,
            "mode": received.mode,
            "size": list(received.size),
        }
    check(
        "storyboard format",
        storyboard_detail
        == {"format": "PNG", "mode": "RGB", "size": [1024, 576]},
        storyboard_detail,
    )

    source_hashes = manifest["source_hashes"]
    check(
        "reviewed frame is the only image source",
        source_hashes == {"reviewed_card_frame": sha256(builder.base.FRAME_PATH)},
        source_hashes,
    )
    check(
        "source provenance matches",
        load_json(PROVENANCE / "animation_forged_chain_v3_source_hashes.json")
        == source_hashes,
        load_json(PROVENANCE / "animation_forged_chain_v3_source_hashes.json"),
    )

    checksum_entries = v2_validation.parse_hashes(REVIEW / "SHA256SUMS.txt")
    expected_checksum_names = EXPECTED_REVIEW_FILES - {"SHA256SUMS.txt"}
    check(
        "review checksum set",
        set(checksum_entries) == expected_checksum_names,
        sorted(checksum_entries),
    )
    check(
        "review checksums",
        all(
            checksum_entries.get(name) == sha256(REVIEW / name)
            for name in expected_checksum_names
        ),
        checksum_entries,
    )

    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted(REVIEW.iterdir())
        if path.suffix.lower() in {".txt", ".html", ".json"}
    ).lower()
    metadata_text = "\n".join(str(data["metadata"]) for data in decoded).lower()
    leaks = sorted(
        token
        for token in FORBIDDEN_PUBLIC_TOKENS
        if token in public_text or token in metadata_text
    )
    check("public privacy scan", not leaks, leaks)

    windows_files = {
        path.name for path in args.windows_review.iterdir() if path.is_file()
    }
    check(
        "Windows review set is exact",
        windows_files == EXPECTED_REVIEW_FILES,
        sorted(windows_files),
    )
    parity = {
        name: (sha256(REVIEW / name), sha256(args.windows_review / name))
        for name in sorted(EXPECTED_REVIEW_FILES)
        if (args.windows_review / name).exists()
    }
    check(
        "Windows review is byte-identical",
        len(parity) == len(EXPECTED_REVIEW_FILES)
        and all(left == right for left, right in parity.values()),
        parity,
    )

    check(
        "V1 and V2 comparisons remain present",
        (ROOT / "review_animation/manifest.json").is_file()
        and (ROOT / "review_animation_release_v2/manifest.json").is_file(),
        {
            "v1": (ROOT / "review_animation/manifest.json").is_file(),
            "v2": (ROOT / "review_animation_release_v2/manifest.json").is_file(),
        },
    )

    production_paths = (
        "app/engine/one_star_hero_cards.py",
        "app/engine/visual_novel_presentation.py",
        "app/storage/stories/one_star_ascension_s1/visual-references/"
        "system-panels/one_star_hero_card_frame_obsidian_orrery_v1.png",
    )
    production_diff = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *production_paths],
        cwd=builder.base.WORKSPACE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    check("no production binding changes", not production_diff, production_diff)

    if args.determinism:
        before = artifact_hashes()
        result = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT)],
            cwd=builder.base.WORKSPACE,
            check=False,
            capture_output=True,
            text=True,
        )
        after = artifact_hashes()
        changed = {
            key: {"before": before.get(key), "after": after.get(key)}
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        }
        check(
            "deterministic rebuild",
            result.returncode == 0 and not changed,
            {
                "returncode": result.returncode,
                "changed": changed,
                "stderr": result.stderr[-1000:],
            },
        )

    report = {
        "schema_version": 3,
        "passed": not failures,
        "check_count": len(checks),
        "checks": checks,
        "failures": failures,
    }
    report_path = PROVENANCE / "animation_forged_chain_v3_validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for item in checks:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status}  {item['name']}")
    print(f"{len(checks) - len(failures)}/{len(checks)} checks passed")
    print(report_path)
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
