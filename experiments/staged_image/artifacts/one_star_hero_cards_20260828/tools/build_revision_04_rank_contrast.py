#!/usr/bin/env python3
"""Build higher-contrast rank options without returning to a rainbow palette."""

from __future__ import annotations

import html
import json
from pathlib import Path

import build_revision_03_palette_options as revision_03


ROOT = revision_03.ROOT
PROOFS = ROOT / "proofs/revision_04_rank_contrast"
REVIEW = ROOT / "review_revision_04_rank_contrast"
DECISION = ROOT / "provenance/review_decision_rank_contrast_20260829.json"


PALETTES = (
    revision_03.PaletteOption(
        key="d_seven_metal_ladder",
        label="OPTION D — SEVEN-METAL LADDER",
        summary="Black iron, iron, steel, silver, electrum, gold, then white gold",
        styles=(
            revision_03.rank_style(1, (8, 10, 13), (30, 34, 40), (78, 86, 96), (92, 101, 112), (172, 180, 189), (20, 23, 28), (105, 115, 128), 0.88, 3, 12),
            revision_03.rank_style(2, (12, 14, 18), (53, 58, 66), (123, 133, 146), (133, 144, 158), (205, 214, 222), (25, 28, 34), (126, 138, 152), 0.86, 12, 28),
            revision_03.rank_style(3, (14, 17, 21), (76, 84, 94), (169, 181, 194), (177, 190, 204), (230, 237, 243), (29, 33, 39), (148, 164, 180), 0.84, 24, 48),
            revision_03.rank_style(4, (18, 20, 24), (106, 113, 122), (224, 231, 239), (224, 232, 240), (255, 255, 255), (34, 37, 42), (184, 197, 210), 0.83, 40, 74),
            revision_03.rank_style(5, (27, 24, 16), (125, 111, 74), (232, 216, 164), (224, 202, 134), (255, 246, 214), (55, 47, 27), (216, 189, 113), 0.72, 58, 98),
            revision_03.rank_style(6, (37, 28, 10), (173, 128, 39), (255, 221, 126), (248, 194, 62), (255, 244, 184), (74, 50, 10), (245, 182, 50), 0.78, 82, 132),
            revision_03.rank_style(7, (43, 35, 13), (221, 181, 74), (255, 255, 235), (255, 232, 135), (255, 255, 255), (82, 62, 9), (255, 223, 115), 0.88, 112, 175),
        ),
    ),
    revision_03.PaletteOption(
        key="e_obsidian_radiance",
        label="OPTION E — OBSIDIAN RADIANCE",
        summary="One gold hue family with a deliberately steep darkness-to-glow ramp",
        styles=(
            revision_03.rank_style(1, (7, 7, 7), (28, 25, 22), (73, 65, 55), (95, 76, 42), (164, 141, 104), (30, 24, 16), (105, 82, 45), 0.25, 0, 10),
            revision_03.rank_style(2, (10, 9, 8), (44, 37, 28), (110, 91, 67), (125, 97, 47), (191, 161, 111), (38, 29, 16), (130, 96, 44), 0.32, 8, 25),
            revision_03.rank_style(3, (13, 11, 8), (62, 49, 31), (147, 112, 68), (157, 119, 50), (215, 184, 125), (46, 33, 15), (157, 112, 43), 0.40, 20, 48),
            revision_03.rank_style(4, (17, 14, 8), (82, 62, 32), (185, 139, 74), (188, 143, 55), (232, 205, 143), (54, 38, 14), (185, 130, 43), 0.48, 38, 78),
            revision_03.rank_style(5, (21, 17, 8), (108, 79, 31), (222, 172, 85), (217, 169, 62), (245, 225, 165), (63, 44, 13), (214, 151, 47), 0.57, 62, 115),
            revision_03.rank_style(6, (27, 21, 8), (141, 101, 29), (248, 208, 113), (242, 198, 76), (253, 241, 191), (73, 51, 11), (241, 177, 57), 0.67, 92, 155),
            revision_03.rank_style(7, (34, 27, 10), (178, 129, 38), (255, 245, 195), (255, 228, 126), (255, 255, 238), (82, 59, 10), (255, 216, 100), 0.78, 128, 205),
        ),
    ),
    revision_03.PaletteOption(
        key="f_metal_milestones",
        label="OPTION F — METAL MILESTONES",
        summary="Iron at 1–2, silver at 3–4, gold at 5–6, radiant white gold at 7",
        styles=(
            revision_03.rank_style(1, (8, 10, 13), (31, 35, 41), (80, 88, 99), (96, 105, 116), (174, 182, 191), (21, 24, 29), (107, 117, 130), 0.87, 3, 12),
            revision_03.rank_style(2, (13, 15, 19), (62, 68, 76), (142, 152, 164), (148, 159, 171), (215, 224, 231), (27, 30, 36), (134, 147, 161), 0.85, 15, 33),
            revision_03.rank_style(3, (15, 18, 22), (79, 86, 95), (177, 187, 198), (183, 194, 206), (235, 241, 246), (31, 34, 40), (155, 169, 183), 0.84, 26, 53),
            revision_03.rank_style(4, (20, 22, 26), (116, 123, 132), (238, 244, 250), (232, 239, 246), (255, 255, 255), (36, 39, 44), (193, 205, 217), 0.84, 44, 82),
            revision_03.rank_style(5, (28, 22, 11), (112, 83, 38), (207, 168, 91), (210, 158, 58), (247, 221, 157), (61, 42, 12), (203, 147, 45), 0.66, 58, 105),
            revision_03.rank_style(6, (39, 29, 9), (174, 128, 35), (255, 222, 122), (248, 194, 62), (255, 244, 184), (75, 50, 9), (245, 183, 51), 0.80, 86, 143),
            revision_03.rank_style(7, (45, 36, 13), (223, 183, 76), (255, 255, 238), (255, 234, 142), (255, 255, 255), (84, 63, 9), (255, 226, 122), 0.90, 120, 188),
        ),
    ),
)


def configure_shared_renderer() -> None:
    """Point the frozen Revision 03 presentation helpers at this revision."""
    revision_03.PROOFS = PROOFS
    revision_03.REVIEW = REVIEW
    revision_03.DECISION = DECISION
    revision_03.PALETTES = PALETTES


def write_review_files(review_paths: list[Path]) -> tuple[Path, Path]:
    start = REVIEW / "00_START_HERE.txt"
    start.write_text(
        """ONE-STAR HERO CARD VISUAL REVIEW — HIGHER-CONTRAST OPTIONS

01  Option D: Seven-Metal Ladder — a distinct metal finish at every rank
02  Option E: Obsidian Radiance — one gold family with a steep glow ramp
03  Option F: Metal Milestones — iron/silver/gold bands plus rank-seven white gold
04  All three options in one 1-through-7 comparison
05  Representative ranks at exact 1024 x 576 system-board scale
06  Contact sheet

Every option freezes the selected generated bust, Obsidian frame geometry,
exact vector-star counts, and corrected frame → bust → nameplate/stars order.
This pass increases adjacent-rank contrast without restoring the rainbow palette.
No generative image call or production promotion is part of this revision.
""",
        encoding="utf-8",
    )
    figures = []
    for path in review_paths:
        escaped = html.escape(path.name)
        label = html.escape(path.stem.replace("_", " "))
        figures.append(
            f'<figure><a href="{escaped}"><img src="{escaped}" alt="{label}"></a>'
            f"<figcaption>{label}</figcaption></figure>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>One-Star Hero Card — Higher-Contrast Rank Options</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; background: #050a14; color: #eee9db; }}
body {{ margin: 0; padding: 2rem; }}
h1 {{ font-family: Georgia, serif; letter-spacing: .04em; }}
p {{ color: #a9bcc9; max-width: 72rem; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(30rem, 1fr)); gap: 1.25rem; }}
figure {{ margin: 0; border: 1px solid #6f5a36; background: #091225; padding: .65rem; }}
img {{ display: block; width: 100%; height: auto; }}
figcaption {{ padding: .65rem .2rem .1rem; text-transform: capitalize; color: #d7b56e; }}
</style>
</head>
<body>
<h1>One-Star Hero Card — Higher-Contrast Rank Options</h1>
<p>Choose D, E, or F. Portrait, frame geometry, star counts, and layer order are identical across every card.</p>
<main>{''.join(figures)}</main>
</body>
</html>
"""
    index = REVIEW / "index.html"
    index.write_text(document, encoding="utf-8")
    return start, index


def main() -> None:
    configure_shared_renderer()
    PROOFS.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    option_records: dict[str, list[dict[str, object]]] = {}
    review_paths: list[Path] = []
    for index, option in enumerate(PALETTES, start=1):
        records = revision_03.build_palette(option)
        option_records[option.key] = records
        review_paths.append(revision_03.palette_slide(option, records, index))
    comparison = revision_03.all_options_comparison(option_records)
    review_paths.append(comparison)
    board_proof, board_review = revision_03.board_scale_comparison(option_records)
    review_paths.append(board_review)
    contact = revision_03.contact_sheet(review_paths)
    review_paths.append(contact)
    start, index = write_review_files(review_paths)

    manifest = {
        "schema_version": 4,
        "revision": "04_rank_contrast",
        "status": "review_ready",
        "review_decision": {
            "artifact": DECISION.relative_to(ROOT).as_posix(),
            "sha256": revision_03.base.sha256(DECISION),
        },
        "frozen_invariants": {
            "portrait": "generated_bust_plus_matte",
            "frame": "Obsidian Orrery geometry",
            "layer_order": [
                "card_plate",
                "rank_glow",
                "rank_frame_border",
                "generated_bust",
                "nameplate_foreground",
                "vector_stars",
                "name_and_emblem",
            ],
            "excluded_card_subjects": ["warden_of_the_eighth"],
        },
        "options": [
            {
                "key": option.key,
                "label": option.label,
                "summary": option.summary,
                "cards": option_records[option.key],
            }
            for option in PALETTES
        ],
        "board_scale_proof": {
            "artifact": board_proof.relative_to(ROOT).as_posix(),
            "review_artifact": board_review.relative_to(ROOT).as_posix(),
            "dimensions": [1024, 576],
            "sha256": revision_03.base.sha256(board_proof),
        },
        "review": [path.relative_to(ROOT).as_posix() for path in review_paths],
        "review_start": start.relative_to(ROOT).as_posix(),
        "review_html": index.relative_to(ROOT).as_posix(),
    }
    manifest_path = ROOT / "revision_04_rank_contrast_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(index)
    print(board_proof)


if __name__ == "__main__":
    main()
