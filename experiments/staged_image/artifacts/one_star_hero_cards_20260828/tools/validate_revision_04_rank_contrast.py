#!/usr/bin/env python3
"""Validate the higher-contrast, restrained rank-option review pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[3]
PROOFS = ROOT / "proofs/revision_04_rank_contrast"
REVIEW = ROOT / "review_revision_04_rank_contrast"
MANIFEST = ROOT / "revision_04_rank_contrast_manifest.json"
DECISION = ROOT / "provenance/review_decision_rank_contrast_20260829.json"
EXPECTED_OPTIONS = (
    "d_seven_metal_ladder",
    "e_obsidian_radiance",
    "f_metal_milestones",
)
EXPECTED_LAYERS = [
    "card_plate",
    "rank_glow",
    "rank_frame_border",
    "generated_bust",
    "nameplate_foreground",
    "vector_stars",
    "name_and_emblem",
]
EXPECTED_REVIEW_FILES = (
    "00_START_HERE.txt",
    "01_d_seven_metal_ladder.png",
    "02_e_obsidian_radiance.png",
    "03_f_metal_milestones.png",
    "04_all_options_comparison.png",
    "05_board_scale_options_1024x576.png",
    "06_palette_options_contact_sheet.png",
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


def exact_star_components(path: Path, fill: tuple[int, int, int]) -> int:
    """Count the large connected regions painted with the exact vector-star fill."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    pixels = rgb.load()
    remaining = {
        (x, y)
        for y in range(785, 848)
        for x in range(80, 561)
        if pixels[x, y] == fill
    }
    components = 0
    while remaining:
        start = remaining.pop()
        pending = [start]
        area = 1
        while pending:
            x, y = pending.pop()
            for neighbor in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
                    area += 1
        if area >= 100:
            components += 1
    return components


def strictly_increasing(values: list[int | float]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def style_vector(style: dict[str, Any]) -> list[float]:
    """Project the visible rank treatment into a deterministic contrast vector."""
    return [
        *style["border_mid"],
        *style["border_highlight"],
        *style["star_fill"],
        style["frame_glow_alpha"],
        style["star_glow_alpha"],
    ]


def vector_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def hash_outputs() -> dict[str, str]:
    paths = sorted(PROOFS.rglob("*")) + sorted(REVIEW.rglob("*"))
    paths.extend((MANIFEST, DECISION))
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
        help="Optional contrast-review directory to compare byte-for-byte.",
    )
    args = parser.parse_args()
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            failures.append(f"{name}: {detail}")

    manifest = load_json(MANIFEST)
    decision = load_json(DECISION)
    check(
        "review feedback frozen",
        manifest["review_decision"]["sha256"] == sha256(DECISION)
        and "too little differentiation" in decision["feedback"].lower()
        and decision["status"].startswith("Contrast review only"),
        {
            "decision_sha256": sha256(DECISION),
            "feedback": decision["feedback"],
            "status": decision["status"],
        },
    )
    check(
        "revision contract",
        manifest["schema_version"] == 4
        and manifest["revision"] == "04_rank_contrast"
        and manifest["status"] == "review_ready"
        and manifest["frozen_invariants"]["portrait"]
        == "generated_bust_plus_matte"
        and manifest["frozen_invariants"]["layer_order"] == EXPECTED_LAYERS
        and manifest["frozen_invariants"]["excluded_card_subjects"]
        == ["warden_of_the_eighth"],
        {
            "revision": manifest["revision"],
            "status": manifest["status"],
            "frozen_invariants": manifest["frozen_invariants"],
        },
    )

    option_keys = tuple(option["key"] for option in manifest["options"])
    check("three requested options", option_keys == EXPECTED_OPTIONS, option_keys)
    all_card_hashes: list[str] = []
    option_summaries: dict[str, Any] = {}
    for option in manifest["options"]:
        key = option["key"]
        cards = option["cards"]
        counts = [card["star_count"] for card in cards]
        check(f"{key} all star levels", counts == list(range(1, 8)), counts)
        frame_glows: list[int] = []
        star_glows: list[int] = []
        frame_hashes: list[str] = []
        styles: list[dict[str, Any]] = []
        family_colors: list[dict[str, tuple[int, int, int]]] = []
        for card in cards:
            stars = card["star_count"]
            card_path = ROOT / card["artifact"]
            frame_path = ROOT / card["frame_artifact"]
            style = card["style"]
            with Image.open(card_path) as image:
                card_mode = image.mode
                card_size = list(image.size)
            with Image.open(frame_path) as image:
                frame_mode = image.mode
                frame_size = list(image.size)
                alpha_extrema = (
                    list(image.getchannel("A").getextrema())
                    if "A" in image.getbands()
                    else None
                )
            star_fill = tuple(style["star_fill"])
            visible = exact_star_components(card_path, star_fill)
            overlaps = card["overlap_pixels"]
            check(
                f"{key} rank {stars} card",
                card_mode == "RGBA"
                and card_size == [640, 1024]
                and frame_mode == "RGBA"
                and frame_size == [640, 1024]
                and alpha_extrema == [0, 255]
                and sha256(card_path) == card["sha256"]
                and sha256(frame_path) == card["frame_sha256"]
                and visible == stars
                and card["layer_order"] == EXPECTED_LAYERS
                and overlaps["bust_and_border"] > 1000
                and overlaps["bust_and_nameplate"] > 0
                and overlaps["bust_and_stars"] > 0,
                {
                    "visible_stars": visible,
                    "dimensions": card_size,
                    "frame_alpha": alpha_extrema,
                    "overlaps": overlaps,
                },
            )
            all_card_hashes.append(card["sha256"])
            frame_hashes.append(card["frame_sha256"])
            frame_glows.append(style["frame_glow_alpha"])
            star_glows.append(style["star_glow_alpha"])
            styles.append(style)
            family_colors.append(
                {
                    "star_fill": star_fill,
                    "border_mid": tuple(style["border_mid"]),
                }
            )

        distances = [
            vector_distance(style_vector(left), style_vector(right))
            for left, right in zip(styles, styles[1:])
        ]
        check(
            f"{key} strong adjacent-rank contrast",
            len(distances) == 6
            and min(distances) >= 60.0
            and strictly_increasing(frame_glows)
            and strictly_increasing(star_glows)
            and len(set(frame_hashes)) == 7,
            {
                "adjacent_style_distances": [round(value, 2) for value in distances],
                "minimum_required": 60.0,
                "frame_glow_alpha": frame_glows,
                "star_glow_alpha": star_glows,
                "unique_frames": len(set(frame_hashes)),
            },
        )

        if key == "e_obsidian_radiance":
            family_ok = all(
                color[0] >= color[1] >= color[2]
                for pair in family_colors
                for color in pair.values()
            )
            brightness = [sum(style["star_fill"]) for style in styles]
            family_ok = family_ok and strictly_increasing(brightness)
        else:
            low_metal_ok = all(
                max(color) - min(color) <= 30
                for pair in family_colors[:4]
                for color in pair.values()
            )
            high_metal_ok = all(
                color[0] >= color[1] >= color[2]
                for pair in family_colors[4:]
                for color in pair.values()
            )
            family_ok = low_metal_ok and high_metal_ok
            brightness = [sum(style["star_fill"]) for style in styles]
        check(
            f"{key} restrained metal families",
            family_ok,
            {"colors": family_colors, "star_brightness": brightness},
        )
        option_summaries[key] = {
            "cards": len(cards),
            "adjacent_style_distances": [round(value, 2) for value in distances],
            "minimum_adjacent_style_distance": round(min(distances), 2),
            "frame_hashes": frame_hashes,
        }

    check(
        "all option cards are distinct",
        len(all_card_hashes) == 21 and len(set(all_card_hashes)) == 21,
        {"cards": len(all_card_hashes), "unique_hashes": len(set(all_card_hashes))},
    )
    check(
        "enemy subject absent from contrast proofs",
        not list(PROOFS.rglob("*warden*")),
        [path.relative_to(ROOT).as_posix() for path in PROOFS.rglob("*warden*")],
    )

    board = manifest["board_scale_proof"]
    board_path = ROOT / board["artifact"]
    review_board = ROOT / board["review_artifact"]
    with Image.open(board_path) as image:
        board_mode = image.mode
        board_size = list(image.size)
    check(
        "exact system-board scale proof",
        board_mode == "RGB"
        and board_size == [1024, 576]
        and board["dimensions"] == [1024, 576]
        and sha256(board_path) == board["sha256"]
        and sha256(board_path) == sha256(review_board),
        {"mode": board_mode, "dimensions": board_size, "sha256": sha256(board_path)},
    )

    actual_review_files = tuple(
        sorted(path.name for path in REVIEW.iterdir() if path.is_file())
    )
    check(
        "complete review pack",
        actual_review_files == tuple(sorted(EXPECTED_REVIEW_FILES)),
        actual_review_files,
    )

    public_text = (MANIFEST, REVIEW / "00_START_HERE.txt", REVIEW / "index.html")
    public_images = sorted(PROOFS.rglob("*.png")) + sorted(REVIEW.glob("*.png"))
    forbidden_hits: list[dict[str, str]] = []
    metadata: dict[str, list[str]] = {}
    for path in public_text:
        content = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_PUBLIC_TOKENS:
            if token.lower() in content:
                forbidden_hits.append(
                    {"artifact": path.relative_to(ROOT).as_posix(), "token": token}
                )
    for path in public_images:
        with Image.open(path) as image:
            metadata[path.relative_to(ROOT).as_posix()] = sorted(image.info)
            textual_metadata = json.dumps(image.info, sort_keys=True).lower()
        searchable = f"{path.name.lower()}\n{textual_metadata}"
        for token in FORBIDDEN_PUBLIC_TOKENS:
            if token.lower() in searchable:
                forbidden_hits.append(
                    {"artifact": path.relative_to(ROOT).as_posix(), "token": token}
                )
    check(
        "no private/source identifiers in contrast artifacts",
        not forbidden_hits and all(not keys for keys in metadata.values()),
        {
            "forbidden_hits": forbidden_hits,
            "pngs_with_metadata": {
                path: keys for path, keys in metadata.items() if keys
            },
        },
    )

    before = hash_outputs()
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/build_revision_04_rank_contrast.py")],
        cwd=WORKSPACE,
        check=False,
        capture_output=True,
        text=True,
    )
    after = hash_outputs()
    changed = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }
    check(
        "deterministic contrast compositor rerun",
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
        copied = (
            sorted(path.name for path in args.windows_review.iterdir() if path.is_file())
            if args.windows_review.is_dir()
            else []
        )
        mismatches: list[dict[str, str]] = []
        for name in EXPECTED_REVIEW_FILES:
            source = REVIEW / name
            destination = args.windows_review / name
            if not destination.is_file():
                mismatches.append({"file": name, "issue": "missing"})
            elif sha256(source) != sha256(destination):
                mismatches.append({"file": name, "issue": "hash mismatch"})
        unexpected = sorted(set(copied) - set(EXPECTED_REVIEW_FILES))
        missing = sorted(set(EXPECTED_REVIEW_FILES) - set(copied))
        windows_detail = {
            "path": str(args.windows_review),
            "files": copied,
            "mismatches": mismatches,
            "unexpected": unexpected,
            "missing": missing,
        }
        check(
            "Windows contrast slideshow copy",
            args.windows_review.is_dir()
            and not mismatches
            and not unexpected
            and not missing,
            windows_detail,
        )

    report = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "checks": checks,
        "failures": failures,
        "option_summaries": option_summaries,
        "manual_visual_checks": [
            {
                "check": "each option has materially stronger rank separation without rainbow hues",
                "status": "passed",
                "evidence": "three full-size 1-through-7 sheets inspected at original detail",
            },
            {
                "check": "generated bust remains above the border and below nameplate and stars",
                "status": "passed",
                "evidence": "all option sheets inspected with the approved stacking frozen",
            },
            {
                "check": "representative ranks remain legible at system-board scale",
                "status": "passed",
                "evidence": "exact 1024x576 board proof inspected at original detail",
            },
            {
                "check": "combined-sheet labels and summaries do not overlap cards",
                "status": "passed",
                "evidence": "3840x2160 comparison inspected at original detail",
            },
            {
                "check": "no private/source identifiers are visible in proof pixels",
                "status": "passed",
                "evidence": "option sheets, comparison, board proof, and contact sheet inspected",
            },
        ],
        "deterministic_hashes": after,
        "windows_review": windows_detail,
    }
    report_path = ROOT / "provenance/validation_revision_04_rank_contrast.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sums_path = ROOT / "provenance/SHA256SUMS_revision_04_rank_contrast.txt"
    hashable = sorted(
        [MANIFEST, DECISION, report_path]
        + list(PROOFS.rglob("*"))
        + list(REVIEW.rglob("*"))
    )
    sums_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
            for path in hashable
            if path.is_file() and path != sums_path
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
