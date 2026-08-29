#!/usr/bin/env python3
"""Build the rank-color and portrait-layering revision proofs."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

import build_proofs as base


ROOT = base.ROOT
PROOFS = ROOT / "proofs/revision_02"
REVIEW = ROOT / "review_revision_02"
DECISION = ROOT / "provenance/review_decision_20260828.json"
FRAME_PATH = ROOT / "proofs/frames/01_obsidian_orrery_transparent_640x1024.png"
NAMEPLATE_FRONT_Y = 832

Color = tuple[int, int, int]


@dataclass(frozen=True)
class RankStyle:
    stars: int
    border_shadow: Color
    border_mid: Color
    border_highlight: Color
    star_fill: Color
    star_highlight: Color
    star_outline: Color
    glow: Color
    tint_strength: float
    frame_glow_alpha: int
    star_glow_alpha: int


RANK_STYLES = (
    RankStyle(
        1,
        (18, 22, 29),
        (76, 86, 99),
        (173, 185, 197),
        (151, 166, 181),
        (231, 239, 244),
        (31, 38, 47),
        (132, 157, 181),
        0.58,
        18,
        34,
    ),
    RankStyle(
        2,
        (10, 30, 36),
        (50, 108, 120),
        (170, 229, 232),
        (82, 185, 199),
        (224, 255, 255),
        (20, 55, 61),
        (72, 190, 205),
        0.63,
        28,
        48,
    ),
    RankStyle(
        3,
        (8, 29, 53),
        (42, 103, 168),
        (166, 220, 255),
        (65, 157, 231),
        (217, 244, 255),
        (17, 45, 83),
        (55, 145, 235),
        0.68,
        38,
        62,
    ),
    RankStyle(
        4,
        (31, 18, 57),
        (105, 72, 159),
        (220, 193, 255),
        (157, 108, 224),
        (239, 223, 255),
        (55, 29, 82),
        (146, 91, 226),
        0.72,
        48,
        78,
    ),
    RankStyle(
        5,
        (61, 23, 29),
        (163, 65, 68),
        (255, 192, 157),
        (229, 108, 77),
        (255, 226, 194),
        (82, 32, 26),
        (228, 88, 70),
        0.76,
        58,
        94,
    ),
    RankStyle(
        6,
        (66, 45, 11),
        (188, 129, 36),
        (255, 228, 138),
        (241, 180, 60),
        (255, 244, 184),
        (79, 51, 10),
        (244, 174, 45),
        0.82,
        68,
        112,
    ),
    RankStyle(
        7,
        (68, 57, 17),
        (229, 192, 79),
        (255, 255, 239),
        (255, 226, 112),
        (255, 255, 246),
        (84, 62, 8),
        (255, 225, 114),
        0.88,
        80,
        138,
    ),
)

STYLE_BY_STARS = {style.stars: style for style in RANK_STYLES}
HERO_SUBJECTS = tuple(subject for subject in base.SUBJECTS if subject.key != "warden")


def tint_frame(source: Image.Image, style: RankStyle) -> Image.Image:
    frame = source.convert("RGBA")
    alpha = frame.getchannel("A")
    grayscale = ImageOps.grayscale(frame)
    graded = ImageOps.colorize(
        grayscale,
        black=style.border_shadow,
        mid=style.border_mid,
        white=style.border_highlight,
    )
    mixed = Image.blend(frame.convert("RGB"), graded, style.tint_strength)
    result = mixed.convert("RGBA")
    result.putalpha(alpha)
    return result


def frame_glow(frame: Image.Image, style: RankStyle) -> Image.Image:
    alpha = frame.getchannel("A").filter(
        ImageFilter.GaussianBlur(5 + style.stars * 0.8)
    )
    alpha = alpha.point(
        lambda value: round(value * style.frame_glow_alpha / 255)
    )
    glow = Image.new("RGBA", base.CARD_SIZE, (*style.glow, 0))
    glow.putalpha(alpha)
    return glow


def nameplate_foreground(frame: Image.Image) -> Image.Image:
    region = Image.new("L", base.CARD_SIZE, 0)
    ImageDraw.Draw(region).rectangle(
        (0, NAMEPLATE_FRONT_Y, base.CARD_SIZE[0], base.CARD_SIZE[1]),
        fill=255,
    )
    alpha = ImageChops.multiply(frame.getchannel("A"), region)
    foreground = frame.copy()
    foreground.putalpha(alpha)
    return foreground


def draw_rank_stars(layer: Image.Image, style: RankStyle) -> None:
    count = style.stars
    spacing = 58
    start = 320 - spacing * (count - 1) / 2
    glow = Image.new("RGBA", base.CARD_SIZE, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    radius = 29 + count * 0.65
    for index in range(count):
        cx = start + index * spacing
        glow_draw.ellipse(
            (cx - radius, 816 - radius, cx + radius, 816 + radius),
            fill=(*style.glow, style.star_glow_alpha),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(9 + count * 0.7))
    layer.alpha_composite(glow)

    draw = ImageDraw.Draw(layer)
    for index in range(count):
        cx = start + index * spacing
        points = base.star_points(cx, 816, 25, 10.7)
        draw.polygon(
            points,
            fill=(*style.star_fill, 255),
            outline=(*style.star_outline, 255),
            width=3,
        )
        highlight = base.star_points(cx - 2.5, 813.5, 13, 5.5)
        draw.polygon(highlight, fill=(*style.star_highlight, 215))


def nonzero_overlap(first: Image.Image, second: Image.Image) -> int:
    first_binary = first.point(lambda value: 255 if value >= 32 else 0)
    second_binary = second.point(lambda value: 255 if value >= 32 else 0)
    overlap = ImageChops.multiply(first_binary, second_binary)
    histogram = overlap.histogram()
    return sum(histogram[1:])


def render_rank_card(
    subject: base.Subject,
    stars: int,
    frame: Image.Image,
    *,
    display_name: str | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    style = STYLE_BY_STARS[stars]
    portrait = base.portrait_layer(subject, "generated")
    nameplate = nameplate_foreground(frame)
    stars_layer = Image.new("RGBA", base.CARD_SIZE, (0, 0, 0, 0))
    draw_rank_stars(stars_layer, style)
    interface = Image.new("RGBA", base.CARD_SIZE, (0, 0, 0, 0))
    name_meta = base.draw_name(interface, display_name or subject.display_name)
    base.draw_niflheim_emblem(interface)

    card = base.card_plate()
    card = Image.alpha_composite(card, frame_glow(frame, style))
    card = Image.alpha_composite(card, frame)
    card = Image.alpha_composite(card, portrait)
    card = Image.alpha_composite(card, nameplate)
    card = Image.alpha_composite(card, stars_layer)
    card = Image.alpha_composite(card, interface)

    frame_back = frame.getchannel("A").copy()
    ImageDraw.Draw(frame_back).rectangle(
        (0, NAMEPLATE_FRONT_Y, base.CARD_SIZE[0], base.CARD_SIZE[1]),
        fill=0,
    )
    metadata: dict[str, object] = {
        "display_name": display_name or subject.display_name,
        "subject": subject.key,
        "star_count": stars,
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
        "layer_overlap_pixels": {
            "bust_over_border": nonzero_overlap(
                portrait.getchannel("A"), frame_back
            ),
            "nameplate_over_bust": nonzero_overlap(
                portrait.getchannel("A"), nameplate.getchannel("A")
            ),
            "stars_over_bust": nonzero_overlap(
                portrait.getchannel("A"), stars_layer.getchannel("A")
            ),
        },
    }
    return card, metadata


def prepare_rank_frames() -> dict[int, Image.Image]:
    output_dir = PROOFS / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(FRAME_PATH) as received:
        source = received.convert("RGBA")
    frames: dict[int, Image.Image] = {}
    for style in RANK_STYLES:
        frame = tint_frame(source, style)
        output = output_dir / f"rank_{style.stars}_frame_rgba.png"
        frame.save(output, format="PNG", optimize=True)
        frames[style.stars] = frame
    return frames


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


def rank_treatment_proof(
    frames: dict[int, Image.Image],
) -> tuple[Path, list[dict[str, object]]]:
    output_dir = PROOFS / "ranks"
    output_dir.mkdir(parents=True, exist_ok=True)
    subject = next(subject for subject in HERO_SUBJECTS if subject.key == "renna")
    card_records: list[dict[str, object]] = []
    cards: dict[int, Image.Image] = {}
    for style in RANK_STYLES:
        card, metadata = render_rank_card(
            subject,
            style.stars,
            frames[style.stars],
        )
        output = output_dir / f"renna_rank_{style.stars}.png"
        card.save(output, format="PNG", optimize=True)
        metadata.update(
            {
                "artifact": output.relative_to(ROOT).as_posix(),
                "sha256": base.sha256(output),
                "frame_artifact": (
                    PROOFS
                    / "frames"
                    / f"rank_{style.stars}_frame_rgba.png"
                ).relative_to(ROOT).as_posix(),
                "frame_sha256": base.sha256(
                    PROOFS / "frames" / f"rank_{style.stars}_frame_rgba.png"
                ),
                "style": asdict(style),
            }
        )
        card_records.append(metadata)
        cards[style.stars] = card

    slide = base.slide_background((2560, 1440))
    draw = ImageDraw.Draw(slide)
    centered_text(
        draw,
        2560,
        34,
        "STAR LEVEL — BORDER + STAR PROGRESSION",
        base.font(base.SERIF_BOLD, 52),
    )
    centered_text(
        draw,
        2560,
        101,
        "Same generated bust; only current star count and rank treatment change",
        base.font(base.SANS, 26),
        (168, 188, 202, 255),
    )

    rows = ((1, 2, 3, 4), (5, 6, 7))
    y_positions = (205, 790)
    for row, y in zip(rows, y_positions):
        card_width, card_height = 320, 512
        gap = 138 if len(row) == 4 else 180
        total = card_width * len(row) + gap * (len(row) - 1)
        x = (2560 - total) // 2
        for stars in row:
            label = f"{stars} STAR" if stars == 1 else f"{stars} STARS"
            label_width, _ = base.text_size(
                draw, label, base.font(base.SANS_BOLD, 25)
            )
            draw.text(
                (x + (card_width - label_width) / 2, y - 43),
                label,
                font=base.font(base.SANS_BOLD, 25),
                fill=(*STYLE_BY_STARS[stars].star_fill, 255),
            )
            thumb = cards[stars].resize(
                (card_width, card_height), Image.Resampling.LANCZOS
            )
            base.place_with_shadow(slide, thumb, (x, y))
            x += card_width + gap

    output = REVIEW / "01_rank_treatments_1_to_7.png"
    slide.convert("RGB").save(output, format="PNG", optimize=True)
    return output, card_records


def layer_order_proof(
    frames: dict[int, Image.Image],
) -> tuple[Path, list[dict[str, object]]]:
    pairs = (
        (
            "RENNA",
            ROOT / "proofs/method_matrix/renna/generated.png",
            next(subject for subject in HERO_SUBJECTS if subject.key == "renna"),
        ),
        (
            "HALCYON",
            ROOT / "proofs/method_matrix/halcyon/generated.png",
            next(subject for subject in HERO_SUBJECTS if subject.key == "halcyon"),
        ),
    )
    slide = base.slide_background()
    draw = ImageDraw.Draw(slide)
    centered_text(
        draw,
        1920,
        28,
        "PORTRAIT / FRAME LAYER ORDER",
        base.font(base.SERIF_BOLD, 45),
    )
    centered_text(
        draw,
        1920,
        85,
        "Revised busts cross the border, then disappear beneath stars and nameplate",
        base.font(base.SANS, 23),
        (166, 187, 201, 255),
    )
    records: list[dict[str, object]] = []
    x_positions = (90, 500, 1020, 1430)
    column = 0
    for label, before_path, subject in pairs:
        with Image.open(before_path) as received:
            before = received.convert("RGBA")
        revised, metadata = render_rank_card(
            subject,
            subject.stars,
            frames[subject.stars],
        )
        after_path = PROOFS / "heroes" / f"{subject.key}_revised.png"
        after_path.parent.mkdir(parents=True, exist_ok=True)
        revised.save(after_path, format="PNG", optimize=True)
        metadata.update(
            {
                "artifact": after_path.relative_to(ROOT).as_posix(),
                "sha256": base.sha256(after_path),
            }
        )
        records.append(metadata)
        for state, image in (("BEFORE", before), ("REVISED", revised)):
            x = x_positions[column]
            title = f"{label} — {state}"
            title_width, _ = base.text_size(
                draw, title, base.font(base.SANS_BOLD, 21)
            )
            draw.text(
                (x + (300 - title_width) / 2, 142),
                title,
                font=base.font(base.SANS_BOLD, 21),
                fill=(213, 172, 91, 255)
                if state == "REVISED"
                else (151, 169, 181, 255),
            )
            thumb = image.resize((300, 480), Image.Resampling.LANCZOS)
            base.place_with_shadow(slide, thumb, (x, 188))
            column += 1

    draw.text(
        (184, 770),
        "BEFORE  border was the final layer and cut across hair, shoulders, and clothing",
        font=base.font(base.SANS, 22),
        fill=(156, 177, 191, 255),
    )
    draw.text(
        (184, 820),
        "REVISED  frame back → bust → nameplate/stars front",
        font=base.font(base.SANS_BOLD, 24),
        fill=(226, 190, 105, 255),
    )
    draw.text(
        (184, 876),
        "The portrait remains clipped to the card canvas; only its relationship to the ornament changes.",
        font=base.font(base.SANS, 22),
        fill=(156, 177, 191, 255),
    )
    output = REVIEW / "02_layer_order_before_after.png"
    slide.convert("RGB").save(output, format="PNG", optimize=True)
    return output, records


def revised_hero_board(
    frames: dict[int, Image.Image],
    existing_records: list[dict[str, object]],
) -> tuple[Path, dict[str, object]]:
    records_by_subject = {
        str(record["subject"]): record for record in existing_records
    }
    board = base.gradient((1024, 576), (8, 19, 37), (2, 5, 13)).convert("RGBA")
    draw = ImageDraw.Draw(board)
    draw.rectangle((0, 0, 1023, 575), outline=(144, 116, 66, 170), width=3)
    draw.text(
        (34, 22),
        "REVISED HERO-CARD DIRECTION",
        font=base.font(base.SERIF_BOLD, 28),
        fill=(239, 235, 222, 255),
    )
    draw.text(
        (36, 60),
        "Generated busts · rank treatment · corrected layering",
        font=base.font(base.SANS, 16),
        fill=(153, 177, 195, 255),
    )

    width, height, gap = 188, 301, 28
    total = width * len(HERO_SUBJECTS) + gap * (len(HERO_SUBJECTS) - 1)
    x = (1024 - total) // 2
    board_records: list[dict[str, object]] = []
    for subject in HERO_SUBJECTS:
        record = records_by_subject.get(subject.key)
        if record:
            path = ROOT / str(record["artifact"])
            with Image.open(path) as received:
                card = received.convert("RGBA")
            metadata = record
        else:
            card, metadata = render_rank_card(
                subject,
                subject.stars,
                frames[subject.stars],
            )
            path = PROOFS / "heroes" / f"{subject.key}_revised.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            card.save(path, format="PNG", optimize=True)
            metadata.update(
                {
                    "artifact": path.relative_to(ROOT).as_posix(),
                    "sha256": base.sha256(path),
                }
            )
        thumb = card.resize((width, height), Image.Resampling.LANCZOS)
        base.place_with_shadow(board, thumb, (x, 118))
        board_records.append(metadata)
        x += width + gap

    draw.text(
        (34, 506),
        "Hero visual subjects only; the enemy stress-test subject has been removed.",
        font=base.font(base.SANS, 15),
        fill=(142, 164, 180, 255),
    )
    proof_output = PROOFS / "boards/revised_heroes_1024x576.png"
    proof_output.parent.mkdir(parents=True, exist_ok=True)
    board.convert("RGB").save(proof_output, format="PNG", optimize=True)
    review_output = REVIEW / "03_revised_hero_board_1024x576.png"
    board.convert("RGB").save(review_output, format="PNG", optimize=True)
    return proof_output, {
        "artifact": proof_output.relative_to(ROOT).as_posix(),
        "review_artifact": review_output.relative_to(ROOT).as_posix(),
        "sha256": base.sha256(proof_output),
        "dimensions": [1024, 576],
        "cards": board_records,
    }


def contact_sheet(review_paths: list[Path]) -> Path:
    sheet = base.slide_background((1920, 1200)).convert("RGB")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (34, 20),
        "ONE-STAR HERO CARD — REVISION 02",
        font=base.font(base.SERIF_BOLD, 34),
        fill=(239, 235, 222),
    )
    positions = ((40, 90), (980, 90), (510, 640))
    for path, position in zip(review_paths, positions):
        with Image.open(path) as received:
            image = ImageOps.fit(
                received.convert("RGB"),
                (900, 506),
                method=Image.Resampling.LANCZOS,
            )
        sheet.paste(image, position)
    output = REVIEW / "04_revision_02_contact_sheet.png"
    sheet.save(output, format="PNG", optimize=True)
    return output


def write_review_files(review_paths: list[Path]) -> tuple[Path, Path]:
    start = REVIEW / "00_START_HERE.txt"
    start.write_text(
        """ONE-STAR HERO CARD VISUAL REVIEW — REVISION 02

01  Exact 1–7 star-level progression using one unchanged generated bust
02  Before/after layer order for Renna and Halcyon
03  Exact 1024 x 576 board using only the selected Hero bust subjects
04  Contact sheet

Requested compositor order:
  card plate → rank border → generated bust → nameplate and stars

The enemy layout-stress subject is omitted. Obsidian Orrery is retained only
as the frame used to demonstrate this revision; production promotion awaits
confirmation of the revised rank treatment and stacking.
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
<title>One-Star Hero Card Review — Revision 02</title>
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
<h1>One-Star Hero Card Review — Revision 02</h1>
<p>Generated busts are selected. Review the rank-dependent border and star treatment plus the corrected border / bust / foreground stacking.</p>
<main>{''.join(figures)}</main>
</body>
</html>
"""
    index = REVIEW / "index.html"
    index.write_text(document, encoding="utf-8")
    return start, index


def main() -> None:
    for directory in (
        PROOFS / "frames",
        PROOFS / "ranks",
        PROOFS / "heroes",
        PROOFS / "boards",
        REVIEW,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    frames = prepare_rank_frames()
    rank_review, rank_records = rank_treatment_proof(frames)
    layer_review, hero_records = layer_order_proof(frames)
    board_path, board_record = revised_hero_board(frames, hero_records)
    review_paths = [
        rank_review,
        layer_review,
        REVIEW / "03_revised_hero_board_1024x576.png",
    ]
    contact = contact_sheet(review_paths)
    review_paths.append(contact)
    start, index = write_review_files(review_paths)

    manifest = {
        "schema_version": 2,
        "revision": "02",
        "status": "review_ready",
        "review_decision": {
            "artifact": DECISION.relative_to(ROOT).as_posix(),
            "sha256": base.sha256(DECISION),
        },
        "frame": {
            "display_name": "Obsidian Orrery",
            "status": "revision proof only",
        },
        "portrait_selection": "generated_bust_plus_matte",
        "excluded_card_subjects": ["warden_of_the_eighth"],
        "layer_order": [
            "card_plate",
            "rank_glow",
            "rank_frame_border",
            "generated_bust",
            "nameplate_foreground",
            "vector_stars",
            "name_and_emblem",
        ],
        "rank_proofs": rank_records,
        "hero_proofs": board_record["cards"],
        "board": board_record,
        "review": [path.relative_to(ROOT).as_posix() for path in review_paths],
        "review_start": start.relative_to(ROOT).as_posix(),
        "review_html": index.relative_to(ROOT).as_posix(),
    }
    manifest_path = ROOT / "revision_02_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(index)
    print(board_path)


if __name__ == "__main__":
    main()
