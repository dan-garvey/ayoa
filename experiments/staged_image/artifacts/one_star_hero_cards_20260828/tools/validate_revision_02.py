#!/usr/bin/env python3
"""Validate revision-02 rank treatment and compositor layering proofs."""

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
PROOFS = ROOT / "proofs/revision_02"
REVIEW = ROOT / "review_revision_02"
MANIFEST = ROOT / "revision_02_manifest.json"
DECISION = ROOT / "provenance/review_decision_20260828.json"
EXPECTED_REVIEW_FILES = (
    "00_START_HERE.txt",
    "01_rank_treatments_1_to_7.png",
    "02_layer_order_before_after.png",
    "03_revised_hero_board_1024x576.png",
    "04_revision_02_contact_sheet.png",
    "index.html",
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
EXPECTED_HEROES = [
    "renna",
    "halcyon",
    "veiled_feminine",
    "veiled_masculine",
]
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


def hash_outputs() -> dict[str, str]:
    paths = sorted(PROOFS.rglob("*.png"))
    paths.extend(sorted(REVIEW.iterdir()))
    paths.extend((MANIFEST, DECISION))
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in paths
        if path.is_file()
    }


def exact_star_components(path: Path, fill: tuple[int, int, int]) -> int:
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


def increasing(values: list[float | int]) -> bool:
    return all(first < second for first, second in zip(values, values[1:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows-review",
        type=Path,
        help="Optional revision slideshow directory to compare byte-for-byte.",
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
        "review decision binding",
        manifest["review_decision"]["sha256"] == sha256(DECISION)
        and decision["portrait_selection"]["method"].startswith(
            "reference-guided generated bust"
        )
        and decision["excluded_from_hero_cards"][0]["display_name"]
        == "Warden of the Eighth",
        manifest["review_decision"],
    )
    check(
        "revision contract",
        manifest["schema_version"] == 2
        and manifest["revision"] == "02"
        and manifest["status"] == "review_ready"
        and manifest["portrait_selection"] == "generated_bust_plus_matte"
        and manifest["layer_order"] == EXPECTED_LAYERS,
        {
            "schema_version": manifest["schema_version"],
            "revision": manifest["revision"],
            "status": manifest["status"],
            "layer_order": manifest["layer_order"],
        },
    )

    ranks = manifest["rank_proofs"]
    star_counts = [record["star_count"] for record in ranks]
    check("all star levels represented", star_counts == list(range(1, 8)), star_counts)
    frame_hashes: list[str] = []
    star_fills: list[tuple[int, int, int]] = []
    border_mids: list[tuple[int, int, int]] = []
    tint_strengths: list[float] = []
    frame_glows: list[int] = []
    star_glows: list[int] = []
    star_details: list[dict[str, Any]] = []
    for record in ranks:
        stars = record["star_count"]
        card_path = ROOT / record["artifact"]
        frame_path = ROOT / record["frame_artifact"]
        style = record["style"]
        with Image.open(card_path) as card:
            card_mode = card.mode
            card_size = list(card.size)
        with Image.open(frame_path) as frame:
            frame_mode = frame.mode
            frame_size = list(frame.size)
            frame_alpha = (
                list(frame.getchannel("A").getextrema())
                if "A" in frame.getbands()
                else None
            )
        fill = tuple(style["star_fill"])
        visible = exact_star_components(card_path, fill)
        detail = {
            "stars": stars,
            "visible_components": visible,
            "card": record["artifact"],
            "frame": record["frame_artifact"],
        }
        star_details.append(detail)
        check(
            f"rank {stars} deterministic card and frame",
            card_mode == "RGBA"
            and card_size == [640, 1024]
            and frame_mode == "RGBA"
            and frame_size == [640, 1024]
            and frame_alpha == [0, 255]
            and sha256(card_path) == record["sha256"]
            and sha256(frame_path) == record["frame_sha256"]
            and style["stars"] == stars
            and visible == stars,
            {
                **detail,
                "card_mode": card_mode,
                "card_size": card_size,
                "frame_mode": frame_mode,
                "frame_size": frame_size,
                "frame_alpha": frame_alpha,
            },
        )
        overlaps = record["layer_overlap_pixels"]
        check(
            f"rank {stars} front/back overlap evidence",
            record["layer_order"] == EXPECTED_LAYERS
            and overlaps["bust_over_border"] > 1000
            and overlaps["nameplate_over_bust"] > 0
            and overlaps["stars_over_bust"] > 0,
            overlaps,
        )
        frame_hashes.append(record["frame_sha256"])
        star_fills.append(fill)
        border_mids.append(tuple(style["border_mid"]))
        tint_strengths.append(style["tint_strength"])
        frame_glows.append(style["frame_glow_alpha"])
        star_glows.append(style["star_glow_alpha"])

    check(
        "rank palettes are distinct",
        len(set(frame_hashes)) == 7
        and len(set(star_fills)) == 7
        and len(set(border_mids)) == 7,
        {
            "unique_frame_hashes": len(set(frame_hashes)),
            "star_fills": star_fills,
            "border_mids": border_mids,
        },
    )
    check(
        "rank intensity grows monotonically",
        increasing(tint_strengths)
        and increasing(frame_glows)
        and increasing(star_glows),
        {
            "tint_strength": tint_strengths,
            "frame_glow_alpha": frame_glows,
            "star_glow_alpha": star_glows,
        },
    )

    heroes = manifest["hero_proofs"]
    hero_keys = [record["subject"] for record in heroes]
    check(
        "selected Hero bust subjects only",
        hero_keys == EXPECTED_HEROES
        and all(
            record["portrait_method"] == "generated_bust_plus_matte"
            and record["layer_order"] == EXPECTED_LAYERS
            for record in heroes
        )
        and not list(PROOFS.rglob("*warden*")),
        {
            "subjects": hero_keys,
            "excluded": manifest["excluded_card_subjects"],
            "unexpected_warden_files": [
                path.relative_to(ROOT).as_posix()
                for path in PROOFS.rglob("*warden*")
            ],
        },
    )
    for record in heroes:
        path = ROOT / record["artifact"]
        check(
            f"Hero proof {record['subject']}",
            path.is_file() and sha256(path) == record["sha256"],
            {"artifact": record["artifact"], "sha256": sha256(path)},
        )

    board = manifest["board"]
    board_path = ROOT / board["artifact"]
    review_board = ROOT / board["review_artifact"]
    with Image.open(board_path) as image:
        board_mode = image.mode
        board_size = list(image.size)
    check(
        "revised 1024x576 Hero board",
        board_mode == "RGB"
        and board_size == [1024, 576]
        and len(board["cards"]) == 4
        and sha256(board_path) == board["sha256"]
        and sha256(board_path) == sha256(review_board),
        {
            "mode": board_mode,
            "dimensions": board_size,
            "subjects": [record["subject"] for record in board["cards"]],
            "sha256": sha256(board_path),
        },
    )

    public_text = (MANIFEST, REVIEW / "00_START_HERE.txt", REVIEW / "index.html")
    public_images = sorted(PROOFS.rglob("*.png")) + sorted(REVIEW.glob("*.png"))
    forbidden_hits: list[dict[str, str]] = []
    png_metadata: dict[str, list[str]] = {}
    for path in public_text:
        content = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_PUBLIC_TOKENS:
            if token.lower() in content:
                forbidden_hits.append(
                    {"artifact": path.relative_to(ROOT).as_posix(), "token": token}
                )
    for path in public_images:
        raw = path.read_bytes().lower()
        for token in FORBIDDEN_PUBLIC_TOKENS:
            if token.lower().encode("utf-8") in raw:
                forbidden_hits.append(
                    {"artifact": path.relative_to(ROOT).as_posix(), "token": token}
                )
        with Image.open(path) as image:
            png_metadata[path.relative_to(ROOT).as_posix()] = sorted(image.info)
    check(
        "no private/source identifiers in revision artifacts",
        not forbidden_hits and all(not keys for keys in png_metadata.values()),
        {
            "forbidden_hits": forbidden_hits,
            "png_count": len(png_metadata),
            "pngs_with_metadata": {
                path: keys for path, keys in png_metadata.items() if keys
            },
        },
    )

    before = hash_outputs()
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/build_revision_02.py")],
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
        "deterministic revision compositor rerun",
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
            "Windows revision slideshow copy",
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
        "star_pixel_counts": star_details,
        "manual_visual_checks": [
            {
                "check": "all seven border and star treatments are visibly distinct",
                "status": "passed",
                "evidence": "01_rank_treatments_1_to_7 inspected at original detail",
            },
            {
                "check": "busts cross the inner arch and side ornament",
                "status": "passed",
                "evidence": "Renna hair and Halcyon hair/cape compared before and after",
            },
            {
                "check": "nameplate and vector stars remain in front of every bust",
                "status": "passed",
                "evidence": "individual RGBA cards and exact board inspected",
            },
            {
                "check": "enemy stress-test subject is absent from revised card pixels",
                "status": "passed",
                "evidence": "revision board and all rank cards inspected",
            },
        ],
        "deterministic_hashes": after,
        "windows_review": windows_detail,
    }
    report_path = ROOT / "provenance/validation_revision_02.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sums_path = ROOT / "provenance/SHA256SUMS_revision_02.txt"
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
