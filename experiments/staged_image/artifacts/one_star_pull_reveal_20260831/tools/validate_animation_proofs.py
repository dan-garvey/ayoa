#!/usr/bin/env python3
"""Validate the isolated animated One-Star summon-reveal proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

import build_animation_proofs as builder


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review_animation"
PROOFS = ROOT / "proofs/animation"
KEYFRAMES = PROOFS / "keyframes"
PROVENANCE = ROOT / "provenance"
MANIFEST = ROOT / "animation_proof_manifest.json"
BUILD_SCRIPT = ROOT / "tools/build_animation_proofs.py"
DEFAULT_WINDOWS_REVIEW = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/OneStarPullRevealAnimationReview_20260831"
)

EXPECTED_REVIEW_FILES = {
    "00_START_HERE.txt",
    "01_iron_plain_under_3.gif",
    "02_silver_omen_3_to_4.gif",
    "03_gold_omen_5_to_6.gif",
    "04_white_gold_omen_7.gif",
    "05_start_peak_end_storyboard.png",
    "SHA256SUMS.txt",
    "index.html",
    "manifest.json",
}

EXPECTED_RANK_BANDS = {
    "under_3": "iron_plain",
    "3_to_4": "silver",
    "5_to_6": "gold",
    "7": "white_gold",
}

FORBIDDEN_PUBLIC_TOKENS = (
    "/home/",
    "app/storage",
    "visual-references",
    "private_extractions",
    "reference_id",
    "source_id",
    "osa_",
    "vn-sprites",
    "hero-card-portraits",
    "system-panels",
    "eighth warden",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_hashes(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator:
            parsed[name] = digest
    return parsed


def gif_data(path: Path) -> dict[str, Any]:
    frame_hashes: list[str] = []
    durations: list[int] = []
    frame_luma: list[float] = []
    motion: list[float] = []
    previous: Image.Image | None = None
    with Image.open(path) as received:
        image_format = received.format
        size = received.size
        animated = bool(getattr(received, "is_animated", False))
        frame_count = int(getattr(received, "n_frames", 1))
        loop = received.info.get("loop")
        metadata = dict(received.info)
        for index in range(frame_count):
            received.seek(index)
            frame = received.convert("RGB")
            frame_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            durations.append(int(received.info.get("duration", 0)))
            frame_luma.append(ImageStat.Stat(frame.convert("L")).mean[0])
            if previous is not None:
                difference = ImageChops.difference(frame, previous)
                motion.append(sum(ImageStat.Stat(difference).mean) / 3.0)
            previous = frame
    return {
        "format": image_format,
        "size": list(size),
        "animated": animated,
        "frame_count": frame_count,
        "loop": loop,
        "durations": durations,
        "frame_hashes": frame_hashes,
        "unique_frames": len(set(frame_hashes)),
        "frame_luma": frame_luma,
        "peak_luma": max(frame_luma),
        "motion": motion,
        "mean_motion": sum(motion) / len(motion),
        "max_motion": max(motion),
        "metadata": metadata,
    }


def artifact_hashes() -> dict[str, str]:
    paths = [path for path in sorted(REVIEW.iterdir()) if path.is_file()]
    paths.extend(sorted(KEYFRAMES.glob("*.png")))
    paths.extend(
        (
            MANIFEST,
            PROVENANCE / "animation_source_hashes.json",
            PROVENANCE / "animation_decisions.json",
            PROVENANCE / "animation_SHA256SUMS.txt",
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
        help="Rebuild and require every owned artifact hash to remain unchanged.",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            failures.append(f"{name}: {detail}")

    manifest = load_json(MANIFEST)
    review_manifest = REVIEW / "manifest.json"
    check(
        "review manifest is canonical manifest",
        review_manifest.read_bytes() == MANIFEST.read_bytes(),
        sha256(MANIFEST),
    )
    check(
        "review-only optional-motion contract",
        manifest["schema_version"] == 1
        and manifest["status"] == "visual_review_only"
        and manifest["direction"] == "animated_rarity_crescendo"
        and manifest["production_binding"] is False
        and manifest["transport_contract"]
        == "optional non-semantic motion before static result"
        and manifest["review_encoding"] == "loop"
        and manifest["production_intent"] == "play_once_then_hold",
        {
            key: manifest[key]
            for key in (
                "schema_version",
                "status",
                "direction",
                "production_binding",
                "transport_contract",
                "review_encoding",
                "production_intent",
            )
        },
    )
    check(
        "no file-size cap",
        manifest["file_size_cap_bytes"] is None,
        manifest["file_size_cap_bytes"],
    )
    check(
        "all rarity bands represented",
        manifest["rank_bands"] == EXPECTED_RANK_BANDS,
        manifest["rank_bands"],
    )
    invariants = manifest["invariants"]
    check(
        "animation is non-semantic",
        invariants["first_frame"] == "sealed"
        and invariants["last_frame"] == "sealed"
        and invariants["identity_before_result"] == "withheld"
        and invariants["exact_rank_before_result"] == "withheld"
        and invariants["semantic_fallback"] == "existing static result board"
        and invariants["animation_required_for_comprehension"] is False
        and invariants["fakeouts"] == "none",
        invariants,
    )
    check(
        "review set is exact",
        {path.name for path in REVIEW.iterdir() if path.is_file()}
        == EXPECTED_REVIEW_FILES,
        sorted(path.name for path in REVIEW.iterdir() if path.is_file()),
    )

    records = manifest["tiers"]
    check("four animation tiers", len(records) == 4, len(records))
    expected_tiers = list(builder.TIERS)
    check(
        "tier file order",
        [record["filename"] for record in records]
        == [tier.filename for tier in expected_tiers],
        [record["filename"] for record in records],
    )
    check(
        "containment ladder",
        [record["containment_locks"] for record in records] == [0, 2, 3, 4]
        and [record["visual_intensity"] for record in records] == [1, 2, 3, 4],
        [
            (record["containment_locks"], record["visual_intensity"])
            for record in records
        ],
    )
    frame_counts = [record["frame_count"] for record in records]
    durations = [record["duration_ms"] for record in records]
    byte_counts = [record["byte_count"] for record in records]
    check(
        "duration and frame count rise with rarity",
        frame_counts == sorted(frame_counts)
        and len(set(frame_counts)) == 4
        and durations == sorted(durations)
        and len(set(durations)) == 4,
        {"frame_counts": frame_counts, "durations_ms": durations},
    )
    check(
        "low tier is short and plain",
        records[0]["duration_ms"] < records[1]["duration_ms"]
        and records[0]["containment_locks"] == 0
        and "PLAIN" in records[0]["label"],
        records[0],
    )
    check(
        "size recorded without enforcement cap",
        all(recorded > 0 for recorded in byte_counts),
        byte_counts,
    )

    decoded: list[dict[str, Any]] = []
    for tier, record in zip(expected_tiers, records, strict=True):
        path = REVIEW / tier.filename
        data = gif_data(path)
        decoded.append(data)
        check(
            f"{tier.key} GIF structure",
            data["format"] == "GIF"
            and data["size"] == list(base_size := builder.base.BOARD_SIZE)
            and data["animated"]
            and data["frame_count"] == tier.frame_count
            and data["loop"] == 0,
            {
                "format": data["format"],
                "size": data["size"],
                "expected_size": list(base_size),
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
            f"{tier.key} playback contract",
            record["review_loop"] is True
            and record["production_intent"] == "play_once"
            and "static result follows" in record["accessible_text"],
            {
                key: record[key]
                for key in (
                    "review_loop",
                    "production_intent",
                    "accessible_text",
                )
            },
        )

    peak_luma = [round(data["peak_luma"], 3) for data in decoded]
    check(
        "peak brightness rises with rarity",
        peak_luma == sorted(peak_luma) and len(set(peak_luma)) == 4,
        peak_luma,
    )
    check(
        "plain tier motion remains gentler",
        decoded[0]["max_motion"] < decoded[1]["max_motion"]
        and decoded[0]["peak_luma"] < decoded[1]["peak_luma"],
        {
            "plain_max_motion": decoded[0]["max_motion"],
            "silver_max_motion": decoded[1]["max_motion"],
            "plain_peak_luma": decoded[0]["peak_luma"],
            "silver_peak_luma": decoded[1]["peak_luma"],
        },
    )

    expected_keyframes = {
        f"{tier.key}_{stage}.png"
        for tier in expected_tiers
        for stage in ("start", "peak", "end")
    }
    actual_keyframes = {path.name for path in KEYFRAMES.glob("*.png")}
    check(
        "exact start peak end keyframes",
        actual_keyframes == expected_keyframes,
        sorted(actual_keyframes),
    )
    keyframe_details: dict[str, Any] = {}
    keyframes_valid = True
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
                and received.size == builder.base.BOARD_SIZE
            )
    check("keyframe format", keyframes_valid, keyframe_details)
    storyboard = REVIEW / "05_start_peak_end_storyboard.png"
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
        "source hash provenance matches",
        load_json(PROVENANCE / "animation_source_hashes.json") == source_hashes,
        load_json(PROVENANCE / "animation_source_hashes.json"),
    )

    checksum_entries = parse_hashes(REVIEW / "SHA256SUMS.txt")
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
    metadata_text = "\n".join(
        str(data["metadata"]) for data in decoded
    ).lower()
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
        name: (
            sha256(REVIEW / name),
            sha256(args.windows_review / name),
        )
        for name in sorted(EXPECTED_REVIEW_FILES)
        if (args.windows_review / name).exists()
    }
    check(
        "Windows review is byte-identical",
        len(parity) == len(EXPECTED_REVIEW_FILES)
        and all(left == right for left, right in parity.values()),
        parity,
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
        "schema_version": 1,
        "passed": not failures,
        "check_count": len(checks),
        "checks": checks,
        "failures": failures,
    }
    report_path = PROVENANCE / "animation_validation_report.json"
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
