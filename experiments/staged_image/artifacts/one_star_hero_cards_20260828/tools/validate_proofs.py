#!/usr/bin/env python3
"""Validate deterministic hero-card proofs and write an auditable report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[3]
EXPECTED_LONG_NAME = "Aurelia Æthelwyn of the Sevenfold Winter Gate"
EXPECTED_REVIEW_FILES = (
    "00_START_HERE.txt",
    "01_frame_obsidian_orrery.png",
    "02_frame_frostbound_archive.png",
    "03_frame_ashen_crown.png",
    "04_methods_renna.png",
    "05_methods_warden.png",
    "06_methods_halcyon.png",
    "07_methods_veiled_feminine.png",
    "08_methods_veiled_masculine.png",
    "09_star_count_proofs.png",
    "10_long_unicode_name.png",
    "11_five_card_board_1024x576.png",
    "12_contact_sheet.png",
    "index.html",
)
FORBIDDEN_PUBLIC_TOKENS = (
    "private_extractions",
    "pick_me_up_style_lora",
    "han_isratte",
    "jenna_shirai",
    "0455",
    "0456",
    "0454",
    "0670",
    "app/storage",
    "visual-references",
    "vn-sprites",
    "/home/dan/ayoa",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_record_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {
        "generated_raw",
        "inputs",
        "proofs",
        "review",
        "prompts",
        "provenance",
        "tools",
    }:
        return ROOT / path
    return WORKSPACE / path


def hash_tree(paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(paths)
    }


def deterministic_outputs() -> list[Path]:
    paths = list((ROOT / "proofs").rglob("*.png"))
    paths.extend(sorted((ROOT / "review").glob("[0-9][0-9]_*.png")))
    paths.extend((ROOT / "proof_manifest.json", ROOT / "review/index.html"))
    return paths


def exact_vector_star_count(path: Path) -> tuple[int, list[int]]:
    """Count the large exact-fill components in the deterministic star band."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    pixels = rgb.load()
    remaining = {
        (x, y)
        for y in range(785, 848)
        for x in range(80, 561)
        if pixels[x, y] == (242, 190, 76)
    }
    areas: list[int] = []
    while remaining:
        start = remaining.pop()
        pending = [start]
        area = 1
        while pending:
            x, y = pending.pop()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
                    area += 1
        if area >= 100:
            areas.append(area)
    return len(areas), sorted(areas)


def verify_record_hashes(value: Any, label: str, failures: list[str]) -> int:
    """Recursively verify dictionaries that carry both path and sha256."""
    verified = 0
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = resolve_record_path(value["path"])
            if not path.is_file():
                failures.append(f"{label}: missing recorded file {value['path']}")
            else:
                actual = sha256(path)
                if actual != value["sha256"]:
                    failures.append(
                        f"{label}: hash mismatch for {value['path']}: "
                        f"{actual} != {value['sha256']}"
                    )
                verified += 1
        for key, child in value.items():
            verify_label = f"{label}.{key}"
            verified += verify_record_hashes(child, verify_label, failures)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            verified += verify_record_hashes(child, f"{label}[{index}]", failures)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows-review",
        type=Path,
        help="Optional slideshow directory to compare byte-for-byte.",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            failures.append(f"{name}: {detail}")

    manifest_path = ROOT / "proof_manifest.json"
    manifest = load_json(manifest_path)

    frames_detail = []
    for entry in manifest["frame_candidates"]:
        path = ROOT / entry["artifact"]
        with Image.open(path) as image:
            mode = image.mode
            dimensions = list(image.size)
            alpha = image.getchannel("A") if "A" in image.getbands() else None
            extrema = list(alpha.getextrema()) if alpha else None
            if alpha:
                histogram = alpha.histogram()
                pixels = image.width * image.height
                transparent_fraction = histogram[0] / pixels
                opaque_fraction = histogram[255] / pixels
            else:
                transparent_fraction = 0.0
                opaque_fraction = 0.0
        frames_detail.append(
            {
                "artifact": entry["artifact"],
                "mode": mode,
                "dimensions": dimensions,
                "alpha_extrema": extrema,
                "transparent_fraction": round(transparent_fraction, 6),
                "opaque_fraction": round(opaque_fraction, 6),
                "sha256": sha256(path),
            }
        )
        check(
            f"frame {path.name} real RGBA",
            mode == "RGBA"
            and dimensions == [640, 1024]
            and extrema == [0, 255]
            and transparent_fraction > 0.5
            and opaque_fraction > 0.01,
            frames_detail[-1],
        )
        check(
            f"frame {path.name} manifest hash",
            sha256(path) == entry["sha256"],
            {"actual": sha256(path), "recorded": entry["sha256"]},
        )

    method_count = 0
    for subject, methods in manifest["portrait_methods"].items():
        check(
            f"portrait method labels for {subject}",
            set(methods) == {"neutral", "locked", "generated"},
            sorted(methods),
        )
        for method, metadata in methods.items():
            path = ROOT / f"proofs/method_matrix/{subject}/{method}.png"
            with Image.open(path) as image:
                dimensions = list(image.size)
                mode = image.mode
            check(
                f"portrait proof {subject}/{method}",
                dimensions == [640, 1024]
                and mode == "RGBA"
                and sha256(path) == metadata["sha256"]
                and metadata["portrait_method"] == method,
                {
                    "dimensions": dimensions,
                    "mode": mode,
                    "sha256": sha256(path),
                },
            )
            method_count += 1
    check("portrait method proof count", method_count == 15, method_count)

    expected_star_counts = [1, 3, 5, 7]
    recorded_star_counts = [entry["star_count"] for entry in manifest["star_proofs"]]
    check("star manifest counts", recorded_star_counts == expected_star_counts, recorded_star_counts)
    star_pixel_details = []
    for entry in manifest["star_proofs"]:
        count = entry["star_count"]
        path = ROOT / f"proofs/cards/renna_{count}_stars.png"
        visual_count, component_areas = exact_vector_star_count(path)
        detail = {
            "artifact": path.relative_to(ROOT).as_posix(),
            "expected": count,
            "visual_exact_fill_components": visual_count,
            "component_areas": component_areas,
            "sha256": sha256(path),
        }
        star_pixel_details.append(detail)
        check(
            f"visible vector stars {count}",
            visual_count == count and sha256(path) == entry["sha256"],
            detail,
        )

    long_name = manifest["long_name_proof"]
    long_lines = long_name["name_lines"]
    long_path = ROOT / "proofs/cards/long_unicode_name_7_stars.png"
    check(
        "long Unicode name fit",
        long_name["display_name"] == EXPECTED_LONG_NAME
        and " ".join(long_lines) == EXPECTED_LONG_NAME
        and len(long_lines) == 2
        and all("…" not in line and "..." not in line for line in long_lines)
        and long_name["name_font_size"] >= 14
        and sha256(long_path) == long_name["sha256"],
        {
            "display_name": long_name["display_name"],
            "lines": long_lines,
            "line_count": len(long_lines),
            "font_size": long_name["name_font_size"],
        },
    )

    board = ROOT / manifest["five_card_board"]["artifact"]
    review_board = ROOT / "review/11_five_card_board_1024x576.png"
    with Image.open(board) as image:
        board_dimensions = list(image.size)
    board_names = [card["display_name"] for card in manifest["five_card_board"]["cards"]]
    check(
        "five-card 1024x576 board",
        board_dimensions == [1024, 576]
        and len(board_names) == 5
        and sha256(board) == manifest["five_card_board"]["sha256"]
        and sha256(board) == sha256(review_board),
        {
            "dimensions": board_dimensions,
            "card_count": len(board_names),
            "ordered_names": board_names,
            "sha256": sha256(board),
        },
    )

    private_sources = load_json(ROOT / "provenance/private_sources.json")
    portrait_sources = load_json(ROOT / "provenance/portrait_sources.json")
    generated_assets = load_json(ROOT / "provenance/generated_assets.json")
    rejected_attempts = load_json(ROOT / "provenance/rejected_attempts.json")
    exclusions = load_json(ROOT / "provenance/exclusions.json")
    provenance_failures: list[str] = []
    verified_hash_records = 0
    for label, value in (
        ("private_sources", private_sources),
        ("portrait_sources", portrait_sources),
        ("generated_assets", generated_assets),
        ("rejected_attempts", rejected_attempts),
        ("exclusions", exclusions),
    ):
        verified_hash_records += verify_record_hashes(value, label, provenance_failures)
    check(
        "source and artifact provenance hashes",
        not provenance_failures and verified_hash_records >= 50,
        {
            "verified_records": verified_hash_records,
            "failures": provenance_failures,
        },
    )

    for key in ("veiled_feminine", "veiled_masculine"):
        subject = portrait_sources["subjects"][key]
        neutral = subject["method_1_neutral_vn_sprite_crop"]
        locked = subject["method_2_locked_reference_crop"]
        check(
            f"{key} honest locked-source label",
            neutral["path"] == locked["path"]
            and neutral["sha256"] == locked["sha256"]
            and "no independent locked portrait" in locked["note"],
            {"neutral": neutral["path"], "locked": locked["path"], "note": locked["note"]},
        )

    raw_assets = list((ROOT / "generated_raw/frames").glob("*_raw.png"))
    raw_assets.extend((ROOT / "generated_raw/portraits").glob("*_raw.png"))
    raw_modes = {}
    for path in sorted(raw_assets):
        with Image.open(path) as image:
            raw_modes[path.relative_to(ROOT).as_posix()] = image.mode
    check(
        "raw image-generation transparency failures preserved",
        len(raw_modes) == 8 and all(mode == "RGB" for mode in raw_modes.values()),
        raw_modes,
    )
    attempts = rejected_attempts["attempts"]
    check(
        "rejected attempts complete",
        len(attempts) == 9
        and any(
            attempt["artifact"]["path"].endswith("01_obsidian_orrery_alpha_attempt.png")
            and attempt["status"] == "rejected"
            for attempt in attempts
        ),
        {"count": len(attempts), "statuses": [attempt["status"] for attempt in attempts]},
    )

    masks = sorted((ROOT / "inputs/masks").glob("*.png"))
    mask_details = []
    for path in masks:
        with Image.open(path) as image:
            detail = {
                "artifact": path.relative_to(ROOT).as_posix(),
                "mode": image.mode,
                "dimensions": list(image.size),
                "extrema": list(image.getextrema()),
                "sha256": sha256(path),
            }
        mask_details.append(detail)
    check(
        "BiRefNet masks frozen",
        len(mask_details) == 8
        and all(item["mode"] == "L" and item["extrema"] == [0, 255] for item in mask_details),
        mask_details,
    )

    public_text_paths = (
        ROOT / "proof_manifest.json",
        ROOT / "review/index.html",
        ROOT / "review/00_START_HERE.txt",
    )
    public_hits: list[dict[str, str]] = []
    for path in public_text_paths:
        content = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_PUBLIC_TOKENS:
            if token.lower() in content:
                public_hits.append(
                    {"artifact": path.relative_to(ROOT).as_posix(), "token": token}
                )
    public_images = sorted((ROOT / "proofs").rglob("*.png"))
    public_images.extend(sorted((ROOT / "review").glob("*.png")))
    image_metadata: dict[str, list[str]] = {}
    for path in public_images:
        raw = path.read_bytes().lower()
        for token in FORBIDDEN_PUBLIC_TOKENS:
            if token.lower().encode("utf-8") in raw:
                public_hits.append(
                    {"artifact": path.relative_to(ROOT).as_posix(), "token": token}
                )
        with Image.open(path) as image:
            image_metadata[path.relative_to(ROOT).as_posix()] = sorted(image.info)
    check(
        "no private/source strings in public text or PNG metadata",
        not public_hits and all(not keys for keys in image_metadata.values()),
        {
            "forbidden_hits": public_hits,
            "png_count": len(image_metadata),
            "pngs_with_metadata": {
                path: keys for path, keys in image_metadata.items() if keys
            },
        },
    )

    before = hash_tree(deterministic_outputs())
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/build_proofs.py")],
        cwd=WORKSPACE,
        check=False,
        capture_output=True,
        text=True,
    )
    after = hash_tree(deterministic_outputs())
    changed = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }
    check(
        "deterministic compositor rerun",
        result.returncode == 0 and not changed,
        {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "checked_artifacts": len(after),
            "changed": changed,
        },
    )

    windows_detail: dict[str, Any] | None = None
    if args.windows_review:
        mismatches = []
        for name in EXPECTED_REVIEW_FILES:
            source = ROOT / "review" / name
            destination = args.windows_review / name
            if not destination.is_file():
                mismatches.append({"file": name, "issue": "missing"})
            elif sha256(source) != sha256(destination):
                mismatches.append(
                    {
                        "file": name,
                        "issue": "hash mismatch",
                        "source": sha256(source),
                        "destination": sha256(destination),
                    }
                )
        copied_files = (
            sorted(
                path.name
                for path in args.windows_review.iterdir()
                if path.is_file()
            )
            if args.windows_review.is_dir()
            else []
        )
        unexpected = sorted(set(copied_files) - set(EXPECTED_REVIEW_FILES))
        missing = sorted(set(EXPECTED_REVIEW_FILES) - set(copied_files))
        windows_detail = {
            "path": str(args.windows_review),
            "expected_files": len(EXPECTED_REVIEW_FILES),
            "copied_files": copied_files,
            "mismatches": mismatches,
            "unexpected": unexpected,
            "missing": missing,
        }
        check(
            "Windows slideshow copy",
            args.windows_review.is_dir()
            and not mismatches
            and not unexpected
            and not missing,
            windows_detail,
        )

    manual_visual_checks = [
        {
            "check": "all three final frame overlays have clean aperture and edge behavior on contrasted grounds",
            "status": "passed",
            "evidence": "reviewed at original detail after BiRefNet matting",
        },
        {
            "check": "Renna, Warden, and Halcyon generated busts preserve reviewed identity",
            "status": "passed",
            "evidence": "five raw busts and method slides visually compared with neutral and locked inputs",
        },
        {
            "check": "both veiled defaults remain featureless in all three methods",
            "status": "passed",
            "evidence": "method slides reviewed at original detail",
        },
        {
            "check": "blank frames contain no baked names, stars, portrait, logo, emblem, or readable text",
            "status": "passed",
            "evidence": "all three frame review slides inspected",
        },
        {
            "check": "no private or source-shaped identifiers are visible in review pixels",
            "status": "passed",
            "evidence": "all twelve ordered review slides and contact sheet inspected",
        },
    ]

    report = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "checks": checks,
        "failures": failures,
        "manual_visual_checks": manual_visual_checks,
        "limitations": [
            "The image-generation tool returned RGB checkerboard images; all final overlays use separately frozen BiRefNet masks.",
            "The pixel-content exclusion check is manual visual QA supplemented by automated text/metadata scanning; no OCR runtime was available.",
            "Generated frames and busts are review candidates only and are not production assets.",
            "The Warden is a nonhuman layout stress test and not a canonical party member.",
        ],
        "frame_alpha": frames_detail,
        "star_pixel_counts": star_pixel_details,
        "output_hashes_after_deterministic_rerun": after,
        "windows_review": windows_detail,
    }
    report_path = ROOT / "provenance/validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sums_path = ROOT / "provenance/SHA256SUMS.txt"
    hashable = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path != sums_path
    )
    sums_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
            for path in hashable
        ),
        encoding="utf-8",
    )

    print(f"status={report['status']}")
    print(f"checks={len(checks)}")
    print(f"failures={len(failures)}")
    print(f"report={report_path.relative_to(ROOT)}")
    print(f"hashes={sums_path.relative_to(ROOT)}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
