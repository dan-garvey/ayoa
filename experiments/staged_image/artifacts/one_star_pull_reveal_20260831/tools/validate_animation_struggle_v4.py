#!/usr/bin/env python3
"""Validate V4 escalating-struggle One-Star summon animation proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

import build_animation_struggle_v4 as builder
import validate_animation_release_v2 as release_validation


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review_animation_struggle_v4"
KEYFRAMES = ROOT / "proofs/animation_struggle_v4/keyframes"
PROVENANCE = ROOT / "provenance"
MANIFEST = ROOT / "animation_struggle_v4_manifest.json"
BUILD_SCRIPT = ROOT / "tools/build_animation_struggle_v4.py"
DEFAULT_WINDOWS_REVIEW = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/"
    "OneStarPullRevealStruggleV4_20260831"
)

EXPECTED_REVIEW_FILES = {
    "00_START_HERE.txt",
    "01_iron_plain_strain_release_under_3.gif",
    "02_silver_lock_struggle_3_to_4.gif",
    "03_gold_chain_struggle_5_to_6.gif",
    "04_white_gold_chain_struggle_7.gif",
    "05_bound_strain_break_flash_storyboard.png",
    "SHA256SUMS.txt",
    "index.html",
    "manifest.json",
}

FORBIDDEN_PUBLIC_TOKENS = release_validation.FORBIDDEN_PUBLIC_TOKENS


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
            PROVENANCE / "animation_struggle_v4_source_hashes.json",
            PROVENANCE / "animation_struggle_v4_decisions.json",
            PROVENANCE / "animation_struggle_v4_SHA256SUMS.txt",
        )
    )
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in paths
        if path.is_file()
    }


def local_peak_count(values: list[float], *, threshold: float) -> int:
    return sum(
        value >= threshold and value > values[index - 1] and value >= values[index + 1]
        for index, value in enumerate(values[1:-1], start=1)
    )


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
        help="Rebuild and require every V4 artifact hash to remain unchanged.",
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
        "V4 struggle review contract",
        manifest["schema_version"] == 4
        and manifest["status"] == "visual_review_only"
        and manifest["direction"]
        == "escalating_struggle_then_outward_release"
        and manifest["supersedes_for_review"]
        == "forged_chain_v3_motion_timing"
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
    revision = manifest["motion_revision"]
    check(
        "anticipation design contract",
        revision["principle"] == "anticipation through escalating resistance"
        and revision["pre_release"] == "tug recoil pulse sequence"
        and revision["locks"] == "wiggle and tug without opening"
        and revision["chains"] == "tighten and ripple without breaking"
        and revision["card"] == "small restrained shudder"
        and revision["release"] == "one decisive outward snap"
        and revision["rank_fakeout"] is False,
        revision,
    )
    invariants = manifest["invariants"]
    check(
        "visual and semantic invariants",
        invariants["forged_chain_style"] is True
        and invariants["outward_lock_motion"] is True
        and invariants["held_pre_reveal_flash"] is True
        and invariants["identity_before_static_result"] == "withheld"
        and invariants["exact_rank_before_static_result"] == "withheld"
        and invariants["semantic_fallback"] == "existing static result board"
        and invariants["animation_required_for_comprehension"] is False
        and invariants["fakeouts"] == "none"
        and invariants["v1_v2_v3_preserved"] is True,
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
    frame_counts = [record["frame_count"] for record in records]
    durations = [record["duration_ms"] for record in records]
    check(
        "duration and frame count rise with rarity",
        frame_counts == sorted(frame_counts)
        and len(set(frame_counts)) == 4
        and durations == sorted(durations)
        and len(set(durations)) == 4,
        {"frame_counts": frame_counts, "durations_ms": durations},
    )
    struggles = [record["struggle"] for record in records]
    check(
        "pulse count rises by rarity",
        [struggle["pulse_count"] for struggle in struggles] == [1, 2, 3, 4],
        [struggle["pulse_count"] for struggle in struggles],
    )
    check(
        "pulse tugs escalate within every tier",
        all(
            struggle["pulse_tugs"] == sorted(struggle["pulse_tugs"])
            and len(set(struggle["pulse_tugs"]))
            == len(struggle["pulse_tugs"])
            for struggle in struggles
        ),
        [struggle["pulse_tugs"] for struggle in struggles],
    )
    check(
        "release delay rises by rarity",
        [struggle["release_start"] for struggle in struggles]
        == sorted(struggle["release_start"] for struggle in struggles),
        [struggle["release_start"] for struggle in struggles],
    )
    check(
        "chain and lock struggle rises by rarity",
        [struggle["chain_wiggle"] for struggle in struggles]
        == [0.0, 0.0, 0.72, 1.0]
        and [struggle["lock_wiggle_px"] for struggle in struggles]
        == sorted(struggle["lock_wiggle_px"] for struggle in struggles),
        {
            "chain_wiggle": [struggle["chain_wiggle"] for struggle in struggles],
            "lock_wiggle_px": [struggle["lock_wiggle_px"] for struggle in struggles],
        },
    )

    profile_details: dict[str, Any] = {}
    profiles_valid = True
    for tier in tiers:
        spec = builder.SPECS[tier.key]
        dense_raw = [index / 1000 for index in range(1001)]
        samples = [builder.motion_sample(spec, raw) for raw in dense_raw]
        pre_release = [
            sample
            for raw, sample in zip(dense_raw, samples, strict=True)
            if raw < spec.release_start
        ]
        post_release = [
            sample.motion_progress
            for raw, sample in zip(dense_raw, samples, strict=True)
            if raw >= spec.release_start
        ]
        energies = [sample.energy for sample in pre_release]
        detected_peaks = local_peak_count(energies, threshold=0.28)
        pulse_center_motion = [
            builder.motion_sample(spec, center).motion_progress
            for center in spec.pulse_centers
        ]
        no_early_break = all(
            builder.v2.break_progress(sample.motion_progress) == 0.0
            for sample in pre_release
        )
        post_monotonic = all(
            right >= left
            for left, right in zip(post_release, post_release[1:])
        )
        valid = (
            detected_peaks == len(spec.pulse_centers)
            and all(value > 0.10 for value in pulse_center_motion)
            and no_early_break
            and post_monotonic
            and abs(post_release[-1] - 1.0) < 1e-9
        )
        profiles_valid &= valid
        profile_details[tier.key] = {
            "expected_pulses": len(spec.pulse_centers),
            "detected_pulses": detected_peaks,
            "pulse_center_motion": pulse_center_motion,
            "no_early_break": no_early_break,
            "post_release_monotonic": post_monotonic,
            "final_motion_progress": post_release[-1],
        }
    check("motion profiles tug recoil then release", profiles_valid, profile_details)

    geometries = [record["release_geometry"] for record in records]
    check(
        "forged chains remain on high tiers",
        [geometry["chain_count"] for geometry in geometries] == [0, 0, 2, 3]
        and all(
            geometry["chain_visual_style"]
            == "heavy_forged_interlocking_links"
            and geometry["chain_segment_motion"] == "coherent_rigid_halves"
            for geometry in geometries[2:]
        ),
        geometries,
    )
    check(
        "locks still release outward",
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
        data = release_validation.gif_data(path)
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
        release_frame = max(
            2,
            round(builder.SPECS[tier.key].release_start * (tier.frame_count - 1)),
        )
        pre_release_motion = data["motion"][: release_frame - 1]
        check(
            f"{tier.key} has visible pre-release struggle",
            pre_release_motion
            and max(pre_release_motion) > 0.35
            and sum(value > 0.25 for value in pre_release_motion)
            >= len(builder.SPECS[tier.key].pulse_centers),
            {
                "pre_release_frames": len(pre_release_motion),
                "max_motion": round(max(pre_release_motion), 4),
                "active_transitions": sum(
                    value > 0.25 for value in pre_release_motion
                ),
            },
        )
        check(
            f"{tier.key} final frame is brightness peak",
            data["frame_luma"][-1] >= max(data["frame_luma"]) - 0.01
            and data["frame_luma"][-1] > data["frame_luma"][0] + 35,
            {
                "start_luma": round(data["frame_luma"][0], 3),
                "final_luma": round(data["frame_luma"][-1], 3),
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

    check(
        "flash brightness rises with rarity",
        [data["frame_luma"][-1] for data in decoded]
        == sorted(data["frame_luma"][-1] for data in decoded),
        [round(data["frame_luma"][-1], 3) for data in decoded],
    )

    expected_keyframes = {
        f"{tier.key}_{stage}.png"
        for tier in tiers
        for stage in ("bound", "strain", "break", "flash")
    }
    actual_keyframes = {path.name for path in KEYFRAMES.glob("*.png")}
    check(
        "exact bound strain break flash keyframes",
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
        images = {
            stage: Image.open(KEYFRAMES / f"{tier.key}_{stage}.png").convert("L")
            for stage in ("bound", "strain", "break", "flash")
        }
        stage_detail = {
            "strain_difference": round(
                ImageStat.Stat(
                    ImageChops.difference(images["bound"], images["strain"])
                ).mean[0],
                3,
            ),
            "break_difference": round(
                ImageStat.Stat(
                    ImageChops.difference(images["strain"], images["break"])
                ).mean[0],
                3,
            ),
            "bound_luma": round(ImageStat.Stat(images["bound"]).mean[0], 3),
            "flash_luma": round(ImageStat.Stat(images["flash"]).mean[0], 3),
        }
        check(
            f"{tier.key} visual stages differ",
            stage_detail["strain_difference"] > 0.35
            and stage_detail["break_difference"] > 1.0
            and stage_detail["flash_luma"] > stage_detail["bound_luma"] + 35,
            stage_detail,
        )
        for image in images.values():
            image.close()

    storyboard = REVIEW / "05_bound_strain_break_flash_storyboard.png"
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
        load_json(PROVENANCE / "animation_struggle_v4_source_hashes.json")
        == source_hashes,
        load_json(PROVENANCE / "animation_struggle_v4_source_hashes.json"),
    )

    checksum_entries = release_validation.parse_hashes(REVIEW / "SHA256SUMS.txt")
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
        "V1 through V3 comparisons remain present",
        all(
            (ROOT / directory / "manifest.json").is_file()
            for directory in (
                "review_animation",
                "review_animation_release_v2",
                "review_animation_forged_chain_v3",
            )
        ),
        {
            directory: (ROOT / directory / "manifest.json").is_file()
            for directory in (
                "review_animation",
                "review_animation_release_v2",
                "review_animation_forged_chain_v3",
            )
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
        "schema_version": 4,
        "passed": not failures,
        "check_count": len(checks),
        "checks": checks,
        "failures": failures,
    }
    report_path = PROVENANCE / "animation_struggle_v4_validation_report.json"
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
