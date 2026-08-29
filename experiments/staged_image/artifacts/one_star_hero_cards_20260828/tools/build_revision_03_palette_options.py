#!/usr/bin/env python3
"""Build restrained rank-palette options from the frozen Revision 02 layout."""

from __future__ import annotations

import html
import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import build_proofs as base
import build_revision_02 as revision_02


ROOT = base.ROOT
PROOFS = ROOT / "proofs/revision_03_palette_options"
REVIEW = ROOT / "review_revision_03_palette_options"
DECISION = ROOT / "provenance/review_decision_palette_options_20260828.json"
FRAME_PATH = ROOT / "proofs/frames/01_obsidian_orrery_transparent_640x1024.png"
SUBJECT = next(subject for subject in base.SUBJECTS if subject.key == "renna")


@dataclass(frozen=True)
class PaletteOption:
    key: str
    label: str
    summary: str
    styles: tuple[revision_02.RankStyle, ...]


def rank_style(
    stars: int,
    border_shadow: tuple[int, int, int],
    border_mid: tuple[int, int, int],
    border_highlight: tuple[int, int, int],
    star_fill: tuple[int, int, int],
    star_highlight: tuple[int, int, int],
    star_outline: tuple[int, int, int],
    glow: tuple[int, int, int],
    tint_strength: float,
    frame_glow_alpha: int,
    star_glow_alpha: int,
) -> revision_02.RankStyle:
    return revision_02.RankStyle(
        stars=stars,
        border_shadow=border_shadow,
        border_mid=border_mid,
        border_highlight=border_highlight,
        star_fill=star_fill,
        star_highlight=star_highlight,
        star_outline=star_outline,
        glow=glow,
        tint_strength=tint_strength,
        frame_glow_alpha=frame_glow_alpha,
        star_glow_alpha=star_glow_alpha,
    )


PALETTES = (
    PaletteOption(
        key="a_tonal_steel",
        label="OPTION A — TONAL STEEL",
        summary="No hue ladder; neutral metal becomes brighter and more luminous",
        styles=(
            rank_style(1, (15, 17, 21), (51, 55, 62), (112, 118, 126), (119, 125, 133), (194, 199, 205), (28, 31, 36), (132, 139, 147), 0.82, 8, 22),
            rank_style(2, (16, 18, 22), (59, 63, 70), (128, 134, 142), (135, 141, 149), (206, 211, 216), (30, 33, 38), (139, 146, 154), 0.82, 14, 30),
            rank_style(3, (17, 19, 23), (68, 72, 79), (145, 151, 159), (151, 157, 165), (216, 221, 226), (32, 35, 40), (146, 153, 161), 0.82, 21, 40),
            rank_style(4, (18, 20, 24), (77, 81, 88), (163, 169, 177), (168, 174, 182), (226, 231, 235), (34, 37, 42), (153, 160, 168), 0.82, 29, 52),
            rank_style(5, (19, 21, 25), (88, 92, 99), (182, 188, 196), (186, 192, 200), (235, 239, 242), (36, 39, 44), (160, 167, 175), 0.82, 39, 67),
            rank_style(6, (20, 22, 26), (101, 105, 112), (207, 213, 221), (209, 215, 223), (244, 247, 249), (38, 41, 46), (169, 176, 184), 0.82, 51, 86),
            rank_style(7, (22, 24, 28), (118, 122, 129), (235, 240, 246), (235, 240, 246), (255, 255, 255), (41, 44, 49), (181, 188, 196), 0.82, 66, 112),
        ),
    ),
    PaletteOption(
        key="b_obsidian_gold",
        label="OPTION B — OBSIDIAN + GOLD",
        summary="One warm metal family; the original dark frame gains restrained gold",
        styles=(
            rank_style(1, (18, 16, 14), (58, 50, 43), (126, 116, 101), (137, 111, 70), (202, 181, 143), (45, 35, 23), (151, 119, 66), 0.36, 6, 20),
            rank_style(2, (19, 17, 14), (66, 56, 45), (141, 127, 106), (153, 124, 72), (216, 193, 149), (49, 38, 23), (160, 126, 66), 0.39, 11, 28),
            rank_style(3, (20, 18, 14), (74, 62, 46), (157, 139, 109), (169, 137, 75), (228, 204, 154), (53, 41, 23), (169, 133, 65), 0.42, 17, 38),
            rank_style(4, (21, 18, 14), (83, 68, 47), (176, 153, 111), (187, 151, 78), (238, 215, 163), (57, 43, 22), (179, 141, 65), 0.45, 24, 50),
            rank_style(5, (22, 19, 14), (94, 75, 47), (198, 170, 116), (206, 166, 82), (246, 225, 174), (62, 47, 21), (192, 151, 66), 0.48, 33, 65),
            rank_style(6, (23, 20, 14), (108, 84, 46), (224, 193, 128), (228, 185, 91), (252, 236, 193), (68, 51, 20), (210, 166, 71), 0.52, 44, 84),
            rank_style(7, (24, 21, 14), (124, 95, 46), (249, 224, 163), (249, 216, 125), (255, 249, 222), (74, 55, 19), (231, 192, 91), 0.56, 58, 108),
        ),
    ),
    PaletteOption(
        key="c_cold_niflheim",
        label="OPTION C — COLD NIFLHEIM STEEL",
        summary="One blue-black metal family; saturation stays low while frost-light rises",
        styles=(
            rank_style(1, (12, 18, 24), (43, 55, 65), (113, 128, 138), (105, 125, 136), (188, 207, 215), (25, 35, 42), (91, 137, 151), 0.64, 9, 23),
            rank_style(2, (12, 19, 25), (48, 63, 73), (126, 144, 154), (115, 139, 151), (198, 220, 227), (26, 38, 45), (93, 143, 158), 0.66, 15, 32),
            rank_style(3, (12, 20, 27), (54, 72, 83), (140, 161, 171), (126, 154, 167), (208, 231, 237), (27, 41, 49), (95, 149, 165), 0.68, 22, 43),
            rank_style(4, (12, 21, 28), (61, 82, 94), (156, 180, 190), (139, 170, 184), (217, 239, 244), (28, 44, 53), (98, 156, 172), 0.70, 30, 56),
            rank_style(5, (13, 22, 29), (69, 94, 107), (175, 201, 211), (155, 188, 201), (227, 247, 250), (29, 47, 57), (102, 164, 180), 0.72, 40, 71),
            rank_style(6, (13, 23, 30), (79, 108, 121), (197, 224, 233), (175, 211, 222), (237, 253, 255), (31, 51, 61), (108, 174, 190), 0.75, 52, 90),
            rank_style(7, (14, 24, 31), (92, 126, 140), (224, 247, 252), (205, 238, 245), (255, 255, 255), (33, 56, 67), (118, 188, 202), 0.78, 68, 116),
        ),
    ),
)


def centered_text(
    draw: ImageDraw.ImageDraw,
    width: int,
    y: int,
    value: str,
    face: object,
    fill: tuple[int, int, int, int] = (239, 237, 228, 255),
) -> None:
    text_width, _ = base.text_size(draw, value, face)
    draw.text(((width - text_width) / 2, y), value, font=face, fill=fill)


def render_card(
    style: revision_02.RankStyle,
    frame: Image.Image,
) -> tuple[Image.Image, dict[str, object]]:
    portrait = base.portrait_layer(SUBJECT, "generated")
    nameplate = revision_02.nameplate_foreground(frame)
    stars = Image.new("RGBA", base.CARD_SIZE, (0, 0, 0, 0))
    revision_02.draw_rank_stars(stars, style)
    interface = Image.new("RGBA", base.CARD_SIZE, (0, 0, 0, 0))
    name_meta = base.draw_name(interface, SUBJECT.display_name)
    base.draw_niflheim_emblem(interface)

    card = base.card_plate()
    card = Image.alpha_composite(card, revision_02.frame_glow(frame, style))
    card = Image.alpha_composite(card, frame)
    card = Image.alpha_composite(card, portrait)
    card = Image.alpha_composite(card, nameplate)
    card = Image.alpha_composite(card, stars)
    card = Image.alpha_composite(card, interface)
    return card, {
        "display_name": SUBJECT.display_name,
        "subject": SUBJECT.key,
        "star_count": style.stars,
        "portrait_method": "generated_bust_plus_matte",
        "name_lines": name_meta["lines"],
        "name_font_size": name_meta["font_size"],
        "layer_order": [
            "card_plate",
            "rank_glow",
            "rank_frame_border",
            "generated_bust",
            "nameplate_foreground",
            "vector_stars",
            "name_and_emblem",
        ],
        "overlap_pixels": {
            "bust_and_border": revision_02.nonzero_overlap(
                portrait.getchannel("A"), frame.getchannel("A")
            ),
            "bust_and_nameplate": revision_02.nonzero_overlap(
                portrait.getchannel("A"), nameplate.getchannel("A")
            ),
            "bust_and_stars": revision_02.nonzero_overlap(
                portrait.getchannel("A"), stars.getchannel("A")
            ),
        },
    }


def build_palette(option: PaletteOption) -> list[dict[str, object]]:
    option_root = PROOFS / option.key
    frame_dir = option_root / "frames"
    card_dir = option_root / "cards"
    frame_dir.mkdir(parents=True, exist_ok=True)
    card_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(FRAME_PATH) as received:
        source_frame = received.convert("RGBA")

    records: list[dict[str, object]] = []
    for style in option.styles:
        frame = revision_02.tint_frame(source_frame, style)
        frame_path = frame_dir / f"rank_{style.stars}_frame_rgba.png"
        frame.save(frame_path, format="PNG", optimize=True)
        card, metadata = render_card(style, frame)
        card_path = card_dir / f"renna_rank_{style.stars}.png"
        card.save(card_path, format="PNG", optimize=True)
        metadata.update(
            {
                "option": option.key,
                "artifact": card_path.relative_to(ROOT).as_posix(),
                "sha256": base.sha256(card_path),
                "frame_artifact": frame_path.relative_to(ROOT).as_posix(),
                "frame_sha256": base.sha256(frame_path),
                "style": asdict(style),
            }
        )
        records.append(metadata)
    return records


def palette_slide(
    option: PaletteOption,
    records: list[dict[str, object]],
    index: int,
) -> Path:
    slide = base.slide_background((2560, 1440))
    draw = ImageDraw.Draw(slide)
    centered_text(draw, 2560, 32, option.label, base.font(base.SERIF_BOLD, 54))
    centered_text(
        draw,
        2560,
        101,
        option.summary,
        base.font(base.SANS, 26),
        (169, 188, 201, 255),
    )
    by_stars = {record["star_count"]: record for record in records}
    rows = ((1, 2, 3, 4), (5, 6, 7))
    y_positions = (205, 790)
    for row, y in zip(rows, y_positions):
        card_width, card_height = 320, 512
        gap = 138 if len(row) == 4 else 180
        total = card_width * len(row) + gap * (len(row) - 1)
        x = (2560 - total) // 2
        for stars in row:
            record = by_stars[stars]
            with Image.open(ROOT / str(record["artifact"])) as received:
                card = received.convert("RGBA")
            label = f"{stars} STAR" if stars == 1 else f"{stars} STARS"
            label_face = base.font(base.SANS_BOLD, 25)
            label_width, _ = base.text_size(draw, label, label_face)
            draw.text(
                (x + (card_width - label_width) / 2, y - 43),
                label,
                font=label_face,
                fill=(*tuple(record["style"]["star_fill"]), 255),
            )
            thumb = card.resize(
                (card_width, card_height), Image.Resampling.LANCZOS
            )
            base.place_with_shadow(slide, thumb, (x, y))
            x += card_width + gap
    output = REVIEW / f"{index:02d}_{option.key}.png"
    slide.convert("RGB").save(output, format="PNG", optimize=True)
    return output


def all_options_comparison(
    option_records: dict[str, list[dict[str, object]]],
) -> Path:
    slide = base.slide_background((3840, 2160))
    draw = ImageDraw.Draw(slide)
    centered_text(
        draw,
        3840,
        34,
        "RESTRAINED STAR-TIER PALETTE OPTIONS",
        base.font(base.SERIF_BOLD, 66),
    )
    centered_text(
        draw,
        3840,
        115,
        "Identical bust, border geometry, star counts, and corrected layer order",
        base.font(base.SANS, 30),
        (169, 188, 201, 255),
    )
    y_positions = (260, 880, 1500)
    for option, y in zip(PALETTES, y_positions):
        option_number, option_name = option.label.split(" — ", 1)
        draw.text(
            (90, y + 82),
            option_number,
            font=base.font(base.SERIF_BOLD, 38),
            fill=(236, 230, 214, 255),
        )
        draw.text(
            (90, y + 134),
            option_name,
            font=base.font(base.SERIF_BOLD, 31),
            fill=(214, 205, 186, 255),
        )
        draw.multiline_text(
            (90, y + 190),
            textwrap.fill(option.summary, width=40),
            font=base.font(base.SANS, 23),
            fill=(159, 181, 196, 255),
            spacing=8,
        )
        records = option_records[option.key]
        x = 735
        for record in records:
            stars = int(record["star_count"])
            label = str(stars)
            label_face = base.font(base.SANS_BOLD, 24)
            label_width, _ = base.text_size(draw, label, label_face)
            draw.text(
                (x + (285 - label_width) / 2, y - 36),
                label,
                font=label_face,
                fill=(*tuple(record["style"]["star_fill"]), 255),
            )
            with Image.open(ROOT / str(record["artifact"])) as received:
                card = received.convert("RGBA").resize(
                    (285, 456), Image.Resampling.LANCZOS
                )
            base.place_with_shadow(slide, card, (x, y))
            x += 355
    output = REVIEW / "04_all_options_comparison.png"
    slide.convert("RGB").save(output, format="PNG", optimize=True)
    return output


def board_scale_comparison(
    option_records: dict[str, list[dict[str, object]]],
) -> tuple[Path, Path]:
    board = base.gradient((1024, 576), (8, 19, 37), (2, 5, 13)).convert("RGBA")
    draw = ImageDraw.Draw(board)
    draw.rectangle((0, 0, 1023, 575), outline=(144, 116, 66, 170), width=3)
    draw.text(
        (28, 18),
        "PALETTE OPTIONS AT SYSTEM-BOARD SCALE",
        font=base.font(base.SERIF_BOLD, 25),
        fill=(239, 235, 222, 255),
    )
    draw.text(
        (30, 51),
        "Representative 1 / 3 / 5 / 7-star cards",
        font=base.font(base.SANS, 15),
        fill=(153, 177, 195, 255),
    )
    y_positions = (94, 252, 410)
    for option, y in zip(PALETTES, y_positions):
        draw.text(
            (28, y + 44),
            option.label.split(" — ", 1)[0],
            font=base.font(base.SERIF_BOLD, 23),
            fill=(231, 225, 211, 255),
        )
        draw.text(
            (28, y + 76),
            option.label.split(" — ", 1)[1].title(),
            font=base.font(base.SANS, 15),
            fill=(151, 174, 189, 255),
        )
        by_stars = {
            int(record["star_count"]): record
            for record in option_records[option.key]
        }
        x = 345
        for stars in (1, 3, 5, 7):
            with Image.open(ROOT / str(by_stars[stars]["artifact"])) as received:
                card = received.convert("RGBA").resize(
                    (88, 141), Image.Resampling.LANCZOS
                )
            base.place_with_shadow(board, card, (x, y))
            draw.text(
                (x + 38, y + 143),
                str(stars),
                font=base.font(base.SANS_BOLD, 12),
                fill=(*tuple(by_stars[stars]["style"]["star_fill"]), 255),
            )
            x += 145
    proof = PROOFS / "board_scale_options_1024x576.png"
    review = REVIEW / "05_board_scale_options_1024x576.png"
    board.convert("RGB").save(proof, format="PNG", optimize=True)
    board.convert("RGB").save(review, format="PNG", optimize=True)
    return proof, review


def contact_sheet(review_paths: list[Path]) -> Path:
    sheet = base.slide_background((1920, 1740)).convert("RGB")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (34, 20),
        "ONE-STAR HERO CARD — RESTRAINED PALETTE OPTIONS",
        font=base.font(base.SERIF_BOLD, 34),
        fill=(239, 235, 222),
    )
    positions = ((40, 90), (980, 90), (40, 640), (980, 640), (510, 1190))
    for path, position in zip(review_paths, positions):
        with Image.open(path) as received:
            image = ImageOps.fit(
                received.convert("RGB"),
                (900, 506),
                method=Image.Resampling.LANCZOS,
            )
        sheet.paste(image, position)
    output = REVIEW / "06_palette_options_contact_sheet.png"
    sheet.save(output, format="PNG", optimize=True)
    return output


def write_review_files(review_paths: list[Path]) -> tuple[Path, Path]:
    start = REVIEW / "00_START_HERE.txt"
    start.write_text(
        """ONE-STAR HERO CARD VISUAL REVIEW — RESTRAINED PALETTE OPTIONS

01  Option A: Tonal Steel — neutral brightness-only progression
02  Option B: Obsidian + Gold — one restrained warm metal family
03  Option C: Cold Niflheim Steel — one restrained cool metal family
04  All three options in one 1-through-7 comparison
05  Representative ranks at exact 1024 x 576 system-board scale
06  Contact sheet

Every option freezes the selected generated bust, Obsidian frame geometry,
exact vector-star counts, and corrected frame → bust → nameplate/stars order.
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
<title>One-Star Hero Card — Restrained Palette Options</title>
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
<h1>One-Star Hero Card — Restrained Palette Options</h1>
<p>Choose A, B, or C. Portrait, frame geometry, star counts, and layer order are identical across every card.</p>
<main>{''.join(figures)}</main>
</body>
</html>
"""
    index = REVIEW / "index.html"
    index.write_text(document, encoding="utf-8")
    return start, index


def main() -> None:
    PROOFS.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    option_records: dict[str, list[dict[str, object]]] = {}
    review_paths: list[Path] = []
    for index, option in enumerate(PALETTES, start=1):
        records = build_palette(option)
        option_records[option.key] = records
        review_paths.append(palette_slide(option, records, index))
    comparison = all_options_comparison(option_records)
    review_paths.append(comparison)
    board_proof, board_review = board_scale_comparison(option_records)
    review_paths.append(board_review)
    contact = contact_sheet(review_paths)
    review_paths.append(contact)
    start, index = write_review_files(review_paths)

    manifest = {
        "schema_version": 3,
        "revision": "03_palette_options",
        "status": "review_ready",
        "review_decision": {
            "artifact": DECISION.relative_to(ROOT).as_posix(),
            "sha256": base.sha256(DECISION),
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
            "sha256": base.sha256(board_proof),
        },
        "review": [path.relative_to(ROOT).as_posix() for path in review_paths],
        "review_start": start.relative_to(ROOT).as_posix(),
        "review_html": index.relative_to(ROOT).as_posix(),
    }
    manifest_path = ROOT / "revision_03_palette_options_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(index)
    print(board_proof)


if __name__ == "__main__":
    main()
