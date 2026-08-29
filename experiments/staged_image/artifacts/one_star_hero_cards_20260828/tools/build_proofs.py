#!/usr/bin/env python3
"""Build deterministic One-Star hero-card visual proofs from frozen inputs."""

from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[3]
VISUALS = WORKSPACE / "app/storage/stories/one_star_ascension_s1/visual-references"
SERIF = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
SERIF_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")
SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SANS_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

CARD_SIZE = (640, 1024)
PORTRAIT_BOX = (70, 135, 570, 842)
NAME_BOX = (108, 872, 532, 959)

Method = Literal["neutral", "locked", "generated"]


@dataclass(frozen=True)
class Subject:
    key: str
    display_name: str
    stars: int
    neutral: Path
    locked: Path
    generated: Path
    neutral_crop_fraction: float = 0.62
    locked_note: str = "Reviewed locked reference"


SUBJECTS = (
    Subject(
        key="renna",
        display_name="Renna Holt",
        stars=1,
        neutral=VISUALS / "vn-sprites/renna_holt/neutral.png",
        locked=VISUALS / "locked/renna_holt/facial_zoom.png",
        generated=ROOT / "generated_raw/portraits/renna_reference_guided_rgba.png",
    ),
    Subject(
        key="warden",
        display_name="Warden of the Eighth",
        stars=5,
        neutral=VISUALS / "vn-sprites/warden_of_the_eighth/neutral.png",
        locked=VISUALS / "warden_of_the_eighth.webp",
        generated=ROOT / "generated_raw/portraits/warden_reference_guided_rgba.png",
        neutral_crop_fraction=0.76,
        locked_note="Reviewed nonhuman identity reference",
    ),
    Subject(
        key="halcyon",
        display_name="Halcyon",
        stars=6,
        neutral=VISUALS / "vn-sprites/halcyon_of_the_gilded_march/neutral.png",
        locked=VISUALS / "locked/halcyon_of_the_gilded_march/facial_zoom.png",
        generated=ROOT / "generated_raw/portraits/halcyon_reference_guided_rgba.png",
    ),
    Subject(
        key="veiled_feminine",
        display_name="Identity Veiled",
        stars=1,
        neutral=VISUALS / "vn-sprites/veiled_feminine/neutral.png",
        locked=VISUALS / "vn-sprites/veiled_feminine/neutral.png",
        generated=ROOT
        / "generated_raw/portraits/veiled_feminine_reference_guided_rgba.png",
        neutral_crop_fraction=0.66,
        locked_note="No independent portrait; neutral generic remains locked",
    ),
    Subject(
        key="veiled_masculine",
        display_name="Identity Veiled",
        stars=1,
        neutral=VISUALS / "vn-sprites/veiled_masculine/neutral.png",
        locked=VISUALS / "vn-sprites/veiled_masculine/neutral.png",
        generated=ROOT
        / "generated_raw/portraits/veiled_masculine_reference_guided_rgba.png",
        neutral_crop_fraction=0.66,
        locked_note="No independent portrait; neutral generic remains locked",
    ),
)

FRAME_SOURCES = (
    (
        "obsidian_orrery",
        "Obsidian Orrery",
        ROOT / "generated_raw/frames/01_obsidian_orrery_birefnet_rgba.png",
    ),
    (
        "frostbound_archive",
        "Frostbound Archive",
        ROOT / "generated_raw/frames/02_frostbound_archive_birefnet_rgba.png",
    ),
    (
        "ashen_crown",
        "Ashen Crown",
        ROOT / "generated_raw/frames/03_ashen_crown_birefnet_rgba.png",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def text_size(draw: ImageDraw.ImageDraw, value: str, face: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), value, font=face)
    return right - left, bottom - top


def gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    denominator = max(1, height - 1)
    for y in range(height):
        t = y / denominator
        color = tuple(round(a * (1 - t) + b * t) for a, b in zip(top, bottom))
        draw.line((0, y, width, y), fill=color)
    return image


def prepare_frames() -> dict[str, Path]:
    output_dir = ROOT / "proofs/frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for index, (key, _label, source) in enumerate(FRAME_SOURCES, start=1):
        with Image.open(source) as received:
            image = received.convert("RGBA").resize(CARD_SIZE, Image.Resampling.LANCZOS)
        output = output_dir / f"{index:02d}_{key}_transparent_640x1024.png"
        image.save(output, format="PNG", optimize=True)
        outputs[key] = output
    return outputs


def card_plate() -> Image.Image:
    base = gradient(CARD_SIZE, (10, 18, 35), (3, 5, 12)).convert("RGBA")
    decoration = Image.new("RGBA", CARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(decoration)
    draw.ellipse((93, 146, 547, 600), outline=(84, 121, 153, 38), width=3)
    draw.ellipse((139, 192, 501, 554), outline=(188, 145, 76, 24), width=2)
    for radius in (175, 225, 275):
        draw.arc(
            (320 - radius, 220 - radius, 320 + radius, 220 + radius),
            195,
            345,
            fill=(117, 150, 175, 28),
            width=2,
        )
    for x in range(98, 550, 42):
        draw.line((x, 170, x + 150, 775), fill=(121, 151, 173, 13), width=2)
    draw.rounded_rectangle((93, 853, 547, 978), radius=18, fill=(1, 3, 8, 238))
    base = Image.alpha_composite(base, decoration)
    mask = Image.new("L", CARD_SIZE, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((10, 18, 630, 1014), radius=36, fill=255)
    base.putalpha(mask)
    return base


def threshold_bbox(image: Image.Image, minimum_alpha: int = 32) -> tuple[int, int, int, int]:
    alpha = image.convert("RGBA").getchannel("A")
    bbox = alpha.point(lambda value: 255 if value >= minimum_alpha else 0).getbbox()
    return bbox or (0, 0, image.width, image.height)


def place_portrait(subject: Subject, method: Method) -> Image.Image:
    path = {
        "neutral": subject.neutral,
        "locked": subject.locked,
        "generated": subject.generated,
    }[method]
    with Image.open(path) as received:
        original = received.convert("RGBA")
    target_width = PORTRAIT_BOX[2] - PORTRAIT_BOX[0]
    target_height = PORTRAIT_BOX[3] - PORTRAIT_BOX[1]
    target = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))

    if method == "locked" and subject.key not in {
        "veiled_feminine",
        "veiled_masculine",
    }:
        fitted = ImageOps.fit(
            original.convert("RGB"),
            (target_width, target_height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.33),
        ).convert("RGBA")
        target.alpha_composite(fitted)
        return target

    bbox = threshold_bbox(original)
    if method == "neutral":
        x0, y0, x1, y1 = bbox
        y1 = max(y0 + 1, round(y0 + (y1 - y0) * subject.neutral_crop_fraction))
        bbox = (x0, y0, x1, y1)
    cropped = original.crop(bbox)
    scale = min(target_width / cropped.width, target_height / cropped.height)
    if subject.key == "warden":
        scale = min(scale * 1.03, target_width / cropped.width, target_height / cropped.height)
    resized = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
        Image.Resampling.LANCZOS,
    )
    x = (target_width - resized.width) // 2
    y = target_height - resized.height
    if method == "generated":
        y = min(y, 18)
    target.alpha_composite(resized, (x, y))
    return target


def portrait_layer(subject: Subject, method: Method) -> Image.Image:
    layer = Image.new("RGBA", CARD_SIZE, (0, 0, 0, 0))
    portrait = place_portrait(subject, method)
    layer.alpha_composite(portrait, (PORTRAIT_BOX[0], PORTRAIT_BOX[1]))
    fade_height = 155
    fade = Image.new("RGBA", (PORTRAIT_BOX[2] - PORTRAIT_BOX[0], fade_height))
    fade_draw = ImageDraw.Draw(fade)
    for y in range(fade_height):
        alpha = round(205 * (y / max(1, fade_height - 1)) ** 1.7)
        fade_draw.line((0, y, fade.width, y), fill=(2, 4, 10, alpha))
    layer.alpha_composite(fade, (PORTRAIT_BOX[0], PORTRAIT_BOX[3] - fade_height))
    return layer


def star_points(cx: float, cy: float, outer: float, inner: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = outer if index % 2 == 0 else inner
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    return points


def draw_stars(layer: Image.Image, count: int) -> None:
    if not 1 <= count <= 7:
        raise ValueError(f"star count outside 1..7: {count}")
    glow = Image.new("RGBA", CARD_SIZE, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    spacing = 58
    start = 320 - spacing * (count - 1) / 2
    for index in range(count):
        cx = start + index * spacing
        glow_draw.ellipse((cx - 31, 785, cx + 31, 847), fill=(255, 194, 67, 70))
    glow = glow.filter(ImageFilter.GaussianBlur(13))
    layer.alpha_composite(glow)
    draw = ImageDraw.Draw(layer)
    for index in range(count):
        cx = start + index * spacing
        points = star_points(cx, 816, 25, 10.7)
        draw.polygon(points, fill=(242, 190, 76, 255), outline=(62, 36, 14, 255), width=3)
        highlight = star_points(cx - 2.5, 813.5, 13, 5.5)
        draw.polygon(highlight, fill=(255, 239, 180, 205))


def name_lines(draw: ImageDraw.ImageDraw, value: str, max_width: int, max_height: int) -> tuple[list[str], ImageFont.FreeTypeFont]:
    normalized = " ".join(value.split())
    words = normalized.split(" ")
    candidates: list[list[str]] = [[normalized]]
    if len(words) > 1:
        for split in range(1, len(words)):
            candidates.append([" ".join(words[:split]), " ".join(words[split:])])

    for size in range(48, 13, -1):
        face = font(SERIF_BOLD, size)
        viable: list[tuple[float, list[str]]] = []
        for lines in candidates:
            widths = [text_size(draw, line, face)[0] for line in lines]
            line_height = text_size(draw, "Ag", face)[1]
            total_height = line_height * len(lines) + (4 if len(lines) == 2 else 0)
            if max(widths) <= max_width and total_height <= max_height:
                raggedness = max(widths) - min(widths) if len(widths) == 2 else 0
                score = len(lines) * 1000 + raggedness
                if len(lines) == 1:
                    score = 0
                viable.append((score, lines))
        if viable:
            viable.sort(key=lambda pair: pair[0])
            return viable[0][1], face
    raise ValueError(f"name cannot fit in two lines without truncation: {value!r}")


def draw_name(layer: Image.Image, value: str) -> dict[str, object]:
    draw = ImageDraw.Draw(layer)
    lines, face = name_lines(
        draw,
        value,
        NAME_BOX[2] - NAME_BOX[0],
        NAME_BOX[3] - NAME_BOX[1],
    )
    line_height = text_size(draw, "Ag", face)[1]
    gap = 4 if len(lines) == 2 else 0
    total_height = line_height * len(lines) + gap
    y = NAME_BOX[1] + (NAME_BOX[3] - NAME_BOX[1] - total_height) / 2
    for line in lines:
        width, _height = text_size(draw, line, face)
        x = (CARD_SIZE[0] - width) / 2
        draw.text(
            (x + 2, y + 2),
            line,
            font=face,
            fill=(0, 0, 0, 220),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 220),
        )
        draw.text(
            (x, y),
            line,
            font=face,
            fill=(244, 240, 228, 255),
            stroke_width=1,
            stroke_fill=(111, 76, 31, 255),
        )
        y += line_height + gap
    return {"lines": lines, "font_size": face.size}


def draw_niflheim_emblem(layer: Image.Image) -> None:
    draw = ImageDraw.Draw(layer)
    cx, cy = 320, 73
    draw.ellipse((cx - 35, cy - 35, cx + 35, cy + 35), fill=(4, 8, 15, 238))
    draw.ellipse(
        (cx - 29, cy - 29, cx + 29, cy + 29),
        outline=(191, 155, 90, 255),
        width=3,
    )
    draw.arc((cx - 19, cy - 20, cx + 19, cy + 19), 195, 345, fill=(188, 224, 232, 255), width=4)
    for offset, height in ((-13, 20), (0, 28), (13, 20)):
        draw.line((cx + offset, cy - 1, cx + offset, cy + height), fill=(188, 224, 232, 255), width=4)
    draw.line((cx - 18, cy + 20, cx, cy + 29, cx + 18, cy + 20), fill=(191, 155, 90, 255), width=3)


def render_card(
    subject: Subject,
    method: Method,
    frame_path: Path,
    *,
    stars: int | None = None,
    display_name: str | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    card = card_plate()
    card = Image.alpha_composite(card, portrait_layer(subject, method))
    interface = Image.new("RGBA", CARD_SIZE, (0, 0, 0, 0))
    star_count = subject.stars if stars is None else stars
    draw_stars(interface, star_count)
    name_meta = draw_name(interface, display_name or subject.display_name)
    draw_niflheim_emblem(interface)
    card = Image.alpha_composite(card, interface)
    with Image.open(frame_path) as received:
        frame = received.convert("RGBA")
    card = Image.alpha_composite(card, frame)
    metadata = {
        "display_name": display_name or subject.display_name,
        "star_count": star_count,
        "portrait_method": method,
        "name_lines": name_meta["lines"],
        "name_font_size": name_meta["font_size"],
        "frame": frame_path.name,
    }
    return card, metadata


def slide_background(size: tuple[int, int] = (1920, 1080)) -> Image.Image:
    slide = gradient(size, (8, 17, 32), (2, 5, 12)).convert("RGBA")
    draw = ImageDraw.Draw(slide)
    for x in range(-400, size[0] + 400, 120):
        draw.line((x, 0, x + 600, size[1]), fill=(113, 147, 171, 10), width=2)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(151, 121, 67, 90), width=3)
    return slide


def centered_text(
    draw: ImageDraw.ImageDraw,
    y: float,
    value: str,
    face: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int, int] = (239, 237, 228, 255),
    width: int = 1920,
) -> None:
    text_width, _ = text_size(draw, value, face)
    draw.text(((width - text_width) / 2, y), value, font=face, fill=fill)


def place_with_shadow(canvas: Image.Image, item: Image.Image, xy: tuple[int, int]) -> None:
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    alpha = item.getchannel("A")
    shadow_shape = Image.new("RGBA", item.size, (0, 0, 0, 185))
    shadow_shape.putalpha(alpha)
    shadow.alpha_composite(shadow_shape, (xy[0] + 14, xy[1] + 16))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    canvas.alpha_composite(shadow)
    canvas.alpha_composite(item, xy)


def frame_review_slides(frame_paths: dict[str, Path]) -> list[Path]:
    outputs: list[Path] = []
    reviews = ROOT / "review"
    labels = {key: label for key, label, _source in FRAME_SOURCES}
    for index, key in enumerate(frame_paths, start=1):
        with Image.open(frame_paths[key]) as received:
            frame = received.convert("RGBA")
        slide = slide_background()
        draw = ImageDraw.Draw(slide)
        centered_text(draw, 34, f"FRAME {index} — {labels[key].upper()}", font(SERIF_BOLD, 46))
        centered_text(
            draw,
            94,
            "Same RGBA overlay on dark and light grounds",
            font(SANS, 23),
            fill=(164, 185, 200, 255),
        )
        preview = frame.resize((430, 688), Image.Resampling.LANCZOS)
        for x, color, label in (
            (360, (7, 17, 37, 255), "DARK"),
            (1130, (224, 226, 222, 255), "LIGHT"),
        ):
            panel = Image.new("RGBA", preview.size, color)
            panel.alpha_composite(preview)
            place_with_shadow(slide, panel, (x, 180))
            label_width, _ = text_size(draw, label, font(SANS_BOLD, 22))
            draw.text((x + (430 - label_width) / 2, 895), label, font=font(SANS_BOLD, 22), fill=(192, 159, 92, 255))
        centered_text(
            draw,
            966,
            "Blank frame only: no portrait, name, stars, emblem, logo, or readable text baked in",
            font(SANS, 21),
            fill=(169, 183, 194, 255),
        )
        output = reviews / f"{index:02d}_frame_{key}.png"
        slide.convert("RGB").save(output, format="PNG", optimize=True)
        outputs.append(output)
    return outputs


def method_slides(
    frame_path: Path,
) -> tuple[list[Path], dict[str, dict[str, dict[str, object]]]]:
    outputs: list[Path] = []
    all_metadata: dict[str, dict[str, dict[str, object]]] = {}
    methods: tuple[tuple[Method, str], ...] = (
        ("neutral", "NEUTRAL SPRITE CROP"),
        ("locked", "LOCKED REFERENCE CROP"),
        ("generated", "GENERATED BUST + MATTE"),
    )
    for slide_index, subject in enumerate(SUBJECTS, start=4):
        slide = slide_background()
        draw = ImageDraw.Draw(slide)
        centered_text(draw, 30, f"PORTRAIT METHODS — {subject.display_name.upper()}", font(SERIF_BOLD, 43))
        subject_dir = ROOT / "proofs/method_matrix" / subject.key
        subject_dir.mkdir(parents=True, exist_ok=True)
        subject_meta: dict[str, dict[str, object]] = {}
        for column, (method, label) in enumerate(methods):
            card, metadata = render_card(subject, method, frame_path)
            path = subject_dir / f"{method}.png"
            card.save(path, format="PNG", optimize=True)
            metadata["sha256"] = sha256(path)
            subject_meta[method] = metadata
            thumb = card.resize((360, 576), Image.Resampling.LANCZOS)
            x = 210 + column * 570
            place_with_shadow(slide, thumb, (x, 170))
            label_width, _ = text_size(draw, label, font(SANS_BOLD, 21))
            draw.text((x + (360 - label_width) / 2, 118), label, font=font(SANS_BOLD, 21), fill=(197, 163, 91, 255))
        note = subject.locked_note
        if subject.key == "warden":
            note += "; layout stress test only, not a summonable Hero"
        centered_text(
            draw,
            820,
            note,
            font(SANS, 20),
            fill=(164, 183, 196, 255),
        )
        centered_text(
            draw,
            872,
            "All stars, name text, background, emblem, and frame layering are deterministic",
            font(SANS, 20),
            fill=(164, 183, 196, 255),
        )
        output = ROOT / "review" / f"{slide_index:02d}_methods_{subject.key}.png"
        slide.convert("RGB").save(output, format="PNG", optimize=True)
        outputs.append(output)
        all_metadata[subject.key] = subject_meta
    return outputs, all_metadata


def star_proofs(frame_path: Path) -> tuple[Path, list[dict[str, object]]]:
    subject = SUBJECTS[0]
    slide = slide_background()
    draw = ImageDraw.Draw(slide)
    centered_text(draw, 30, "DETERMINISTIC STAR-COUNT PROOFS", font(SERIF_BOLD, 44))
    metadata: list[dict[str, object]] = []
    for column, count in enumerate((1, 3, 5, 7)):
        card, card_meta = render_card(subject, "generated", frame_path, stars=count)
        path = ROOT / "proofs/cards" / f"renna_{count}_stars.png"
        card.save(path, format="PNG", optimize=True)
        card_meta["sha256"] = sha256(path)
        metadata.append(card_meta)
        thumb = card.resize((300, 480), Image.Resampling.LANCZOS)
        x = 180 + column * 410
        place_with_shadow(slide, thumb, (x, 215))
        label = f"{count} STAR" if count == 1 else f"{count} STARS"
        label_width, _ = text_size(draw, label, font(SANS_BOLD, 24))
        draw.text((x + (300 - label_width) / 2, 745), label, font=font(SANS_BOLD, 24), fill=(229, 190, 99, 255))
    centered_text(
        draw,
        870,
        "One shared portrait and frame isolate the exact 1 / 3 / 5 / 7 vector-star change",
        font(SANS, 22),
        fill=(169, 185, 198, 255),
    )
    output = ROOT / "review/09_star_count_proofs.png"
    slide.convert("RGB").save(output, format="PNG", optimize=True)
    return output, metadata


def long_name_proof(frame_path: Path) -> tuple[Path, dict[str, object]]:
    value = "Aurelia Æthelwyn of the Sevenfold Winter Gate"
    card, metadata = render_card(
        SUBJECTS[2],
        "generated",
        frame_path,
        stars=7,
        display_name=value,
    )
    card_path = ROOT / "proofs/cards/long_unicode_name_7_stars.png"
    card.save(card_path, format="PNG", optimize=True)
    metadata["sha256"] = sha256(card_path)

    slide = slide_background()
    draw = ImageDraw.Draw(slide)
    centered_text(draw, 30, "LONG + UNICODE NAME PROOF", font(SERIF_BOLD, 44))
    thumb = card.resize((430, 688), Image.Resampling.LANCZOS)
    place_with_shadow(slide, thumb, (280, 155))
    crop = card.crop((85, 850, 555, 980)).resize((940, 260), Image.Resampling.LANCZOS)
    place_with_shadow(slide, crop, (835, 335))
    draw.text((930, 650), "NO TRUNCATION", font=font(SANS_BOLD, 27), fill=(224, 187, 98, 255))
    draw.text(
        (835, 704),
        "Maximum two lines\nDeterministic shrink-to-fit\nUnicode preserved",
        font=font(SANS, 25),
        fill=(174, 190, 201, 255),
        spacing=14,
    )
    output = ROOT / "review/10_long_unicode_name.png"
    slide.convert("RGB").save(output, format="PNG", optimize=True)
    return output, metadata


def five_card_board(frame_path: Path) -> tuple[Path, dict[str, object]]:
    board = gradient((1024, 576), (8, 19, 37), (2, 5, 13)).convert("RGBA")
    draw = ImageDraw.Draw(board)
    draw.rectangle((0, 0, 1023, 575), outline=(144, 116, 66, 170), width=3)
    draw.text((38, 24), "FIVE-CARD LAYOUT STRESS TEST", font=font(SERIF_BOLD, 29), fill=(239, 235, 222, 255))
    draw.text((40, 64), "1024 × 576 system-panel proof", font=font(SANS, 16), fill=(153, 177, 195, 255))
    cards: list[dict[str, object]] = []
    order = (SUBJECTS[0], SUBJECTS[2], SUBJECTS[3], SUBJECTS[4], SUBJECTS[1])
    width, height = 174, 278
    gap = 18
    total = width * len(order) + gap * (len(order) - 1)
    x0 = (1024 - total) // 2
    for index, subject in enumerate(order):
        card, metadata = render_card(subject, "generated", frame_path)
        thumb = card.resize((width, height), Image.Resampling.LANCZOS)
        place_with_shadow(board, thumb, (x0 + index * (width + gap), 123))
        cards.append(metadata)
    draw.text(
        (40, 510),
        "Warden exercises the nonhuman silhouette only; this is not a canonical party.",
        font=font(SANS, 15),
        fill=(142, 164, 180, 255),
    )
    output = ROOT / "proofs/boards/five_card_board_1024x576.png"
    board.convert("RGB").save(output, format="PNG", optimize=True)
    review_output = ROOT / "review/11_five_card_board_1024x576.png"
    board.convert("RGB").save(review_output, format="PNG", optimize=True)
    return output, {"cards": cards, "sha256": sha256(output), "dimensions": [1024, 576]}


def contact_sheet(review_paths: list[Path]) -> Path:
    tile = (600, 338)
    columns = 3
    rows = math.ceil(len(review_paths) / columns)
    sheet = Image.new("RGB", (columns * tile[0] + 80, rows * tile[1] + 100), (5, 10, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((34, 20), "ONE-STAR HERO CARD REVIEW", font=font(SERIF_BOLD, 32), fill=(239, 235, 222))
    for index, path in enumerate(review_paths):
        with Image.open(path) as received:
            tile_image = ImageOps.fit(received.convert("RGB"), tile, method=Image.Resampling.LANCZOS)
        x = 20 + (index % columns) * tile[0]
        y = 72 + (index // columns) * tile[1]
        sheet.paste(tile_image, (x, y))
    output = ROOT / "review/12_contact_sheet.png"
    sheet.save(output, format="PNG", optimize=True)
    return output


def write_review_html(review_paths: list[Path]) -> Path:
    cards = []
    for path in review_paths:
        cards.append(
            f'<figure><a href="{html.escape(path.name)}"><img src="{html.escape(path.name)}" '
            f'alt="{html.escape(path.stem.replace("_", " "))}"></a>'
            f'<figcaption>{html.escape(path.stem.replace("_", " "))}</figcaption></figure>'
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>One-Star Hero Card Review</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; background: #050a14; color: #eee9db; }}
body {{ margin: 0; padding: 2rem; }}
h1 {{ font-family: Georgia, serif; letter-spacing: .04em; }}
p {{ color: #a9bcc9; max-width: 72rem; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(28rem, 1fr)); gap: 1.25rem; }}
figure {{ margin: 0; border: 1px solid #6f5a36; background: #091225; padding: .65rem; }}
img {{ display: block; width: 100%; height: auto; }}
figcaption {{ padding: .65rem .2rem .1rem; text-transform: capitalize; color: #d7b56e; }}
</style>
</head>
<body>
<h1>One-Star Hero Card Review</h1>
<p>Generated frames and busts are candidates only. Typography, star counts, emblem, layout, and compositing are deterministic. Review slides contain no private source paths or extraction identifiers.</p>
<main>{''.join(cards)}</main>
</body>
</html>
"""
    output = ROOT / "review/index.html"
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    for directory in (
        ROOT / "proofs/frames",
        ROOT / "proofs/cards",
        ROOT / "proofs/boards",
        ROOT / "proofs/method_matrix",
        ROOT / "review",
        ROOT / "provenance",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    frames = prepare_frames()
    selected_frame = frames["obsidian_orrery"]
    review_paths = frame_review_slides(frames)
    method_paths, method_metadata = method_slides(selected_frame)
    review_paths.extend(method_paths)
    star_path, star_metadata = star_proofs(selected_frame)
    review_paths.append(star_path)
    long_path, long_metadata = long_name_proof(selected_frame)
    review_paths.append(long_path)
    board_path, board_metadata = five_card_board(selected_frame)
    review_paths.append(ROOT / "review/11_five_card_board_1024x576.png")
    contact_path = contact_sheet(review_paths)
    review_paths.append(contact_path)
    html_path = write_review_html(review_paths)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "deterministic_layers": [
            "frame resizing",
            "portrait crop and layering",
            "vector stars",
            "fixed decorative Niflheim gate emblem",
            "two-line shrink-to-fit name typography",
            "card and board layout",
        ],
        "selected_proof_frame": selected_frame.name,
        "frame_candidates": [
            {
                "display_name": label,
                "artifact": frames[key].relative_to(ROOT).as_posix(),
                "sha256": sha256(frames[key]),
                "mode": Image.open(frames[key]).mode,
                "dimensions": list(Image.open(frames[key]).size),
            }
            for key, label, _source in FRAME_SOURCES
        ],
        "portrait_methods": method_metadata,
        "star_proofs": star_metadata,
        "long_name_proof": long_metadata,
        "five_card_board": {
            **board_metadata,
            "artifact": board_path.relative_to(ROOT).as_posix(),
        },
        "review": [path.relative_to(ROOT).as_posix() for path in review_paths],
        "review_html": html_path.relative_to(ROOT).as_posix(),
    }
    manifest_path = ROOT / "proof_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    print(html_path)
    print(board_path)


if __name__ == "__main__":
    main()
