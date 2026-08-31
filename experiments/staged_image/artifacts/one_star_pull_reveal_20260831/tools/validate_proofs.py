#!/usr/bin/env python3
"""Validate the isolated One-Star rarity-crescendo visual proof pack."""

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
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from app.engine import one_star_hero_cards as hero_cards  # noqa: E402


REVIEW = ROOT / "review"
PROOFS = ROOT / "proofs"
PROVENANCE = ROOT / "provenance"
MANIFEST = ROOT / "proof_manifest.json"
BUILDER = ROOT / "tools/build_proofs.py"
DEFAULT_WINDOWS_REVIEW = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/OneStarPullRevealReview_20260831"
)

EXPECTED_REVIEW_FILES = (
    "00_START_HERE.txt",
    "01_opening_01_batch_sealed.png",
    "02_opening_02_renna_direct.png",
    "03_opening_03_silver_omen.png",
    "04_opening_04_mirelle_reveal.png",
    "05_opening_05_edren_direct.png",
    "06_opening_06_results.png",
    "07_metal_band_escalation.png",
    "08_rank_ambiguity_contract.png",
    "09_five_pull_pacing.png",
    "10_white_gold_peak_omen.png",
    "11_contact_sheet.png",
    "SHA256SUMS.txt",
    "index.html",
    "manifest.json",
)

EXPECTED_SEQUENCE = [
    "01_opening_01_batch_sealed.png",
    "02_opening_02_renna_direct.png",
    "03_opening_03_silver_omen.png",
    "04_opening_04_mirelle_reveal.png",
    "05_opening_05_edren_direct.png",
    "06_opening_06_results.png",
]

EXPECTED_RANK_OMENS = {
    "1": "direct",
    "2": "direct",
    "3": "silver",
    "4": "silver",
    "5": "gold",
    "6": "gold",
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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_star_components(
    path: Path,
    fill: tuple[int, int, int],
) -> int:
    """Count large exact-fill regions in the card's star strip."""

    with Image.open(path) as received:
        pixels = received.convert("RGB").load()
    remaining = {
        (x, y)
        for y in range(782, 851)
        for x in range(76, 565)
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


def artifact_hashes() -> dict[str, str]:
    paths = sorted(REVIEW.iterdir())
    paths.extend(sorted((PROOFS / "result_cards").glob("*.png")))
    paths.extend(
        PROVENANCE / filename
        for filename in (
            "SHA256SUMS.txt",
            "decisions.json",
            "source_hashes.json",
            "validation_report.json",
        )
    )
    paths.append(MANIFEST)
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in paths
        if path.is_file()
    }


def parse_hash_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        if separator:
            result[filename] = digest
    return result


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
        help="Rebuild once and require every artifact hash to remain unchanged.",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            failures.append(f"{name}: {detail}")

    manifest = load_json(MANIFEST)
    review_manifest = load_json(REVIEW / "manifest.json")
    check(
        "public manifest matches proof manifest",
        (REVIEW / "manifest.json").read_bytes() == MANIFEST.read_bytes(),
        sha256(MANIFEST),
    )
    check(
        "review-only contract",
        manifest["schema_version"] == 1
        and manifest["status"] == "visual_review_only"
        and manifest["direction"] == "rarity_crescendo"
        and manifest["production_binding"] is False
        and manifest["frontend_contract"]
        == "static semantic pages at 1024x576",
        {
            key: manifest[key]
            for key in (
                "schema_version",
                "status",
                "direction",
                "production_binding",
                "frontend_contract",
            )
        },
    )

    invariants = manifest["frozen_invariants"]
    check(
        "frozen approved visual contract",
        invariants["card_frame"] == "approved Obsidian Orrery geometry"
        and invariants["card_rank_style"] == "approved Option F metal milestones"
        and invariants["portrait_layering"]
        == "portrait over border, under nameplate and stars"
        and invariants["identity_rules"] == "existing veils remain active"
        and invariants["batch_rarity_signal"] == "none"
        and invariants["exact_rank_before_result"] == "withheld"
        and invariants["extra_omen_threshold"] == "3 stars and above"
        and invariants["fakeouts"] == "none",
        invariants,
    )
    check(
        "opening sequence order",
        manifest["opening_sequence"] == EXPECTED_SEQUENCE,
        manifest["opening_sequence"],
    )
    check(
        "rank omen ambiguity",
        manifest["rank_to_pre_reveal_omen"] == EXPECTED_RANK_OMENS
        and manifest["five_pull_page_count_with_one_special"] == 8,
        {
            "rank_to_pre_reveal_omen": manifest["rank_to_pre_reveal_omen"],
            "five_pull_pages": manifest["five_pull_page_count_with_one_special"],
        },
    )

    bands = manifest["metal_bands"]
    band_order = ("iron", "silver", "gold", "white_gold")
    containments = [bands[key]["containment"] for key in band_order]
    glow_alphas = [bands[key]["glow_alpha"] for key in band_order]
    check(
        "containment and light escalate",
        containments == [1, 2, 3, 4]
        and all(left < right for left, right in zip(glow_alphas, glow_alphas[1:])),
        {"containment": containments, "glow_alpha": glow_alphas},
    )
    check(
        "band labels",
        [bands[key]["range_label"] for key in band_order]
        == ["1–2 stars", "3–4 stars", "5–6 stars", "7 stars"],
        [bands[key]["range_label"] for key in band_order],
    )

    actual_review_files = tuple(sorted(path.name for path in REVIEW.iterdir()))
    check(
        "exact ordered review set",
        actual_review_files == tuple(sorted(EXPECTED_REVIEW_FILES)),
        actual_review_files,
    )
    review_records = review_manifest["review"]
    check(
        "eleven review images",
        len(review_records) == 11
        and [record["filename"] for record in review_records]
        == list(EXPECTED_REVIEW_FILES[1:12]),
        [record["filename"] for record in review_records],
    )

    for record in review_records:
        path = REVIEW / record["filename"]
        with Image.open(path) as received:
            mode = received.mode
            size = list(received.size)
            metadata = dict(received.info)
        check(
            f"review image {record['filename']}",
            mode == "RGB"
            and size == [1024, 576]
            and record["dimensions"] == [1024, 576]
            and record["sha256"] == sha256(path)
            and bool(record["accessible_text"])
            and bool(record["visible_text"])
            and not metadata,
            {
                "mode": mode,
                "size": size,
                "sha256": sha256(path),
                "metadata": metadata,
            },
        )

    expected_card_stars = {"renna": 1, "mirelle": 3, "edren": 1}
    for record in manifest["result_cards"]:
        role = record["role"]
        stars = record["stars"]
        path = PROOFS / "result_cards" / f"{role}.png"
        style = hero_cards._STYLE_BY_STARS[stars]  # noqa: SLF001
        with Image.open(path) as received:
            mode = received.mode
            size = list(received.size)
        visible_stars = exact_star_components(path, style.star_fill)
        check(
            f"result card {role}",
            expected_card_stars[role] == stars
            and mode == "RGBA"
            and size == [640, 1024]
            and record["dimensions"] == [640, 1024]
            and record["sha256"] == sha256(path)
            and visible_stars == stars,
            {
                "stars": stars,
                "visible_stars": visible_stars,
                "mode": mode,
                "size": size,
            },
        )

    for role, digest in manifest["source_hashes"].items():
        check(
            f"source hash {role}",
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            digest,
        )

    hash_entries = parse_hash_file(REVIEW / "SHA256SUMS.txt")
    expected_hashed_review = {
        path.name: sha256(path)
        for path in REVIEW.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    check(
        "review hash inventory",
        hash_entries == expected_hashed_review,
        {"expected": len(expected_hashed_review), "actual": len(hash_entries)},
    )

    public_paths = [
        REVIEW / "00_START_HERE.txt",
        REVIEW / "index.html",
        REVIEW / "manifest.json",
        REVIEW / "SHA256SUMS.txt",
    ]
    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in public_paths
    ).lower()
    leaks = [token for token in FORBIDDEN_PUBLIC_TOKENS if token in public_text]
    check("no public source identifiers", not leaks, leaks)
    declared_visible = "\n".join(
        text
        for record in review_records
        for text in record["visible_text"]
    ).lower()
    visible_leaks = [
        token for token in FORBIDDEN_PUBLIC_TOKENS if token in declared_visible
    ]
    check("no source identifiers declared in pixels", not visible_leaks, visible_leaks)

    windows_files = (
        tuple(sorted(path.name for path in args.windows_review.iterdir()))
        if args.windows_review.exists()
        else ()
    )
    windows_match = windows_files == tuple(sorted(EXPECTED_REVIEW_FILES))
    windows_hashes_match = windows_match and all(
        (REVIEW / filename).read_bytes()
        == (args.windows_review / filename).read_bytes()
        for filename in EXPECTED_REVIEW_FILES
    )
    check(
        "Windows review copy",
        windows_hashes_match,
        {"directory": str(args.windows_review), "files": windows_files},
    )

    production_paths = (
        WORKSPACE / "app/engine/one_star_hero_cards.py",
        WORKSPACE
        / "app/storage/stories/one_star_ascension_s1/visual-references/system-panels/one_star_hero_card_frame_obsidian_orrery_v1.png",
    )
    diff_result = subprocess.run(
        ["git", "diff", "--name-only", "--", *(str(path) for path in production_paths)],
        cwd=WORKSPACE,
        check=True,
        capture_output=True,
        text=True,
    )
    check(
        "no production binding changes",
        not diff_result.stdout.strip(),
        diff_result.stdout.splitlines(),
    )

    if args.determinism:
        before = artifact_hashes()
        result = subprocess.run(
            [sys.executable, str(BUILDER)],
            cwd=WORKSPACE,
            check=False,
            capture_output=True,
            text=True,
        )
        after = artifact_hashes()
        check(
            "deterministic rebuild",
            result.returncode == 0 and before == after,
            {
                "returncode": result.returncode,
                "before_count": len(before),
                "after_count": len(after),
                "changed": sorted(
                    key
                    for key in set(before) | set(after)
                    if before.get(key) != after.get(key)
                ),
                "stderr": result.stderr,
            },
        )

    report = {
        "schema_version": 1,
        "passed": not failures,
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
    }
    report_path = PROVENANCE / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{sum(check['passed'] for check in checks)}/{len(checks)} checks passed")
    print(report_path)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
