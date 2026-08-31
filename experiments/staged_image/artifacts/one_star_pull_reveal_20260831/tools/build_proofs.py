#!/usr/bin/env python3
"""Build deterministic static proofs for the One-Star summon reveal.

This is an experiment-only compositor.  It reuses the reviewed Hero-card
sources and the production card renderer, but does not register or bind any
new runtime asset.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[3]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from app.engine import one_star_hero_cards as hero_cards  # noqa: E402
from app.engine.player_media import ResolvedPlayerMedia  # noqa: E402


VISUALS = (
    WORKSPACE
    / "app/storage/stories/one_star_ascension_s1/visual-references"
)
PROOFS = ROOT / "proofs"
REVIEW = ROOT / "review"
PROVENANCE = ROOT / "provenance"
WINDOWS_REVIEW = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/OneStarPullRevealReview_20260831"
)

FRAME_PATH = (
    VISUALS / "system-panels/one_star_hero_card_frame_obsidian_orrery_v1.png"
)
PORTRAIT_PATHS = {
    "renna_veiled": VISUALS / "hero-card-portraits/veiled_feminine_v1.png",
    "mirelle": VISUALS / "vn-sprites/mirelle_voss/neutral.png",
    "edren_veiled": VISUALS / "hero-card-portraits/veiled_masculine_v1.png",
}

SERIF = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
SERIF_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")
SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SANS_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

BOARD_SIZE = (1024, 576)


@dataclass(frozen=True)
class BandStyle:
    key: str
    label: str
    range_label: str
    shadow: tuple[int, int, int]
    mid: tuple[int, int, int]
    highlight: tuple[int, int, int]
    glow: tuple[int, int, int]
    glow_alpha: int
    containment: int


BANDS = {
    "neutral": BandStyle(
        "neutral",
        "SEALED",
        "unknown",
        (7, 10, 15),
        (34, 42, 52),
        (101, 115, 127),
        (78, 103, 119),
        24,
        0,
    ),
    "iron": BandStyle(
        "iron",
        "IRON",
        "1–2 stars",
        (7, 9, 12),
        (44, 49, 57),
        (119, 130, 142),
        (105, 118, 132),
        38,
        1,
    ),
    "silver": BandStyle(
        "silver",
        "SILVER",
        "3–4 stars",
        (15, 18, 22),
        (105, 115, 126),
        (235, 244, 250),
        (170, 205, 226),
        108,
        2,
    ),
    "gold": BandStyle(
        "gold",
        "GOLD",
        "5–6 stars",
        (27, 20, 8),
        (145, 101, 28),
        (255, 224, 130),
        (245, 183, 51),
        158,
        3,
    ),
    "white_gold": BandStyle(
        "white_gold",
        "WHITE GOLD",
        "7 stars",
        (39, 31, 10),
        (211, 179, 92),
        (255, 255, 239),
        (255, 233, 155),
        222,
        4,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def text_size(
    draw: ImageDraw.ImageDraw,
    value: str,
    face: ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), value, font=face)
    return right - left, bottom - top


def draw_centered(
    draw: ImageDraw.ImageDraw,
    value: str,
    y: int,
    face: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int, int] = (239, 235, 222, 255),
    width: int = BOARD_SIZE[0],
) -> None:
    label_width, _ = text_size(draw, value, face)
    draw.text(((width - label_width) / 2, y), value, font=face, fill=fill)


def gradient(
    size: tuple[int, int],
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    denominator = max(1, height - 1)
    for y in range(height):
        ratio = y / denominator
        color = tuple(
            round(start * (1 - ratio) + end * ratio)
            for start, end in zip(top, bottom, strict=True)
        )
        draw.line((0, y, width, y), fill=color)
    return image


def system_canvas() -> Image.Image:
    canvas = gradient(BOARD_SIZE, (9, 20, 37), (1, 4, 11)).convert("RGBA")
    decoration = Image.new("RGBA", BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(decoration)
    for x in range(-300, 1250, 76):
        draw.line((x, 0, x + 330, 576), fill=(118, 151, 174, 12), width=1)
    for inset, alpha in ((0, 185), (5, 80), (11, 32)):
        draw.rectangle(
            (inset, inset, 1023 - inset, 575 - inset),
            outline=(149, 119, 67, alpha),
            width=2,
        )
    for radius in range(0, 180, 18):
        alpha = max(0, 36 - radius // 6)
        draw.ellipse(
            (512 - radius * 2, 288 - radius, 512 + radius * 2, 288 + radius),
            outline=(111, 144, 168, alpha),
            width=1,
        )
    canvas = Image.alpha_composite(canvas, decoration)
    vignette = Image.new("L", BOARD_SIZE, 0)
    vignette_draw = ImageDraw.Draw(vignette)
    for inset in range(0, 126, 4):
        alpha = round(190 * (1 - inset / 126) ** 2)
        vignette_draw.rectangle(
            (inset, inset, 1023 - inset, 575 - inset),
            outline=alpha,
            width=5,
        )
    darkness = Image.new("RGBA", BOARD_SIZE, (0, 0, 0, 0))
    darkness.putalpha(vignette)
    return Image.alpha_composite(canvas, darkness)


def add_header(
    canvas: Image.Image,
    *,
    kicker: str,
    title: str,
    progress: str = "",
    accent: tuple[int, int, int] = (154, 180, 197),
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (34, 22),
        kicker,
        font=font(SANS_BOLD, 13),
        fill=(*accent, 255),
    )
    title_size = 30
    title_face = font(SERIF_BOLD, title_size)
    while text_size(draw, title, title_face)[0] > 956 and title_size > 20:
        title_size -= 1
        title_face = font(SERIF_BOLD, title_size)
    draw.text(
        (32, 44),
        title,
        font=title_face,
        fill=(240, 236, 223, 255),
    )
    if progress:
        progress_face = font(SANS_BOLD, 14)
        progress_width, _ = text_size(draw, progress, progress_face)
        draw.text(
            (990 - progress_width, 31),
            progress,
            font=progress_face,
            fill=(*accent, 255),
        )
    draw.line((33, 87, 991, 87), fill=(*accent, 82), width=1)


def media_for(path: Path) -> ResolvedPlayerMedia:
    data = path.read_bytes()
    with Image.open(path) as received:
        width, height = received.size
    return ResolvedPlayerMedia(
        filename=path.name,
        mime_type="image/png",
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        width=width,
        height=height,
    )


def build_cards(frame: Image.Image) -> dict[str, Image.Image]:
    cards = {
        "renna": hero_cards.render_one_star_hero_card(
            frame=frame,
            portrait=media_for(PORTRAIT_PATHS["renna_veiled"]),
            portrait_source="override",
            display_name="Renna Holt",
            stars=1,
        ),
        "mirelle": hero_cards.render_one_star_hero_card(
            frame=frame,
            portrait=media_for(PORTRAIT_PATHS["mirelle"]),
            portrait_source="neutral",
            display_name="Mirelle Voss",
            stars=3,
        ),
        "edren": hero_cards.render_one_star_hero_card(
            frame=frame,
            portrait=media_for(PORTRAIT_PATHS["edren_veiled"]),
            portrait_source="override",
            display_name="Edren Marr",
            stars=1,
        ),
    }
    card_dir = PROOFS / "result_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    for key, card in cards.items():
        card.save(card_dir / f"{key}.png", format="PNG", optimize=True)
    return cards


def tint_frame(frame: Image.Image, band: BandStyle) -> Image.Image:
    source = frame.convert("RGBA")
    alpha = source.getchannel("A")
    graded = ImageOps.colorize(
        ImageOps.grayscale(source),
        black=band.shadow,
        mid=band.mid,
        white=band.highlight,
    )
    result = Image.blend(source.convert("RGB"), graded, 0.88).convert("RGBA")
    result.putalpha(alpha)
    return result


def draw_emblem(layer: Image.Image) -> None:
    draw = ImageDraw.Draw(layer)
    center_x, center_y = 320, 410
    draw.ellipse(
        (center_x - 102, center_y - 102, center_x + 102, center_y + 102),
        fill=(3, 7, 14, 244),
        outline=(168, 138, 82, 255),
        width=7,
    )
    draw.ellipse(
        (center_x - 84, center_y - 84, center_x + 84, center_y + 84),
        outline=(173, 210, 221, 230),
        width=5,
    )
    draw.arc(
        (center_x - 56, center_y - 59, center_x + 56, center_y + 56),
        195,
        345,
        fill=(190, 224, 232, 255),
        width=10,
    )
    for offset, height in ((-38, 58), (0, 84), (38, 58)):
        draw.line(
            (
                center_x + offset,
                center_y - 4,
                center_x + offset,
                center_y + height,
            ),
            fill=(190, 224, 232, 255),
            width=10,
        )
    draw.line(
        (
            center_x - 53,
            center_y + 57,
            center_x,
            center_y + 86,
            center_x + 53,
            center_y + 57,
        ),
        fill=(191, 155, 90, 255),
        width=8,
    )


def sealed_card(frame: Image.Image, band: BandStyle) -> Image.Image:
    back = hero_cards._card_plate()  # noqa: SLF001 - freeze reviewed geometry
    inner = Image.new("RGBA", hero_cards.CARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(inner)
    draw.rounded_rectangle(
        (81, 118, 559, 848),
        radius=30,
        fill=(2, 6, 14, 246),
        outline=(*band.mid, 145),
        width=4,
    )
    for radius in (120, 178, 236):
        draw.ellipse(
            (320 - radius, 410 - radius, 320 + radius, 410 + radius),
            outline=(*band.glow, 36 + band.containment * 10),
            width=3,
        )
    for angle in range(0, 360, 30):
        radians = math.radians(angle)
        x1 = 320 + math.cos(radians) * 115
        y1 = 410 + math.sin(radians) * 115
        x2 = 320 + math.cos(radians) * 235
        y2 = 410 + math.sin(radians) * 235
        draw.line((x1, y1, x2, y2), fill=(*band.glow, 30), width=2)
    draw_emblem(inner)
    sealed_face = Image.alpha_composite(back, inner)
    tinted = tint_frame(frame, band)
    glow_alpha = tinted.getchannel("A").filter(ImageFilter.GaussianBlur(10))
    glow_alpha = glow_alpha.point(
        lambda value: round(value * band.glow_alpha / 255)
    )
    glow = Image.new("RGBA", hero_cards.CARD_SIZE, (*band.glow, 0))
    glow.putalpha(glow_alpha)
    sealed_face = Image.alpha_composite(sealed_face, glow)
    sealed_face = Image.alpha_composite(sealed_face, tinted)
    label_layer = Image.new("RGBA", hero_cards.CARD_SIZE, (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label = "SEALED"
    label_face = font(SERIF_BOLD, 44)
    label_width, _ = text_size(label_draw, label, label_face)
    label_draw.text(
        ((640 - label_width) / 2, 893),
        label,
        font=label_face,
        fill=(235, 231, 217, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )
    return Image.alpha_composite(sealed_face, label_layer)


def place_with_shadow(
    canvas: Image.Image,
    item: Image.Image,
    position: tuple[int, int],
    *,
    blur: int = 14,
) -> None:
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_shape = Image.new("RGBA", item.size, (0, 0, 0, 205))
    shadow_shape.putalpha(item.getchannel("A"))
    shadow.alpha_composite(
        shadow_shape,
        (position[0] + 13, position[1] + 17),
    )
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(blur)))
    canvas.alpha_composite(item, position)


def chain_shadow(size: tuple[int, int], strength: int) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    links = Image.new("RGBA", (900, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(links)
    for x in range(-30, 930, 55):
        draw.ellipse(
            (x, 20, x + 72, 78),
            outline=(0, 0, 0, 75 + strength * 24),
            width=12,
        )
    rotated = links.rotate(-23, resample=Image.Resampling.BICUBIC, expand=True)
    layer.alpha_composite(rotated, (-170, -20))
    layer.alpha_composite(rotated.transpose(Image.Transpose.FLIP_LEFT_RIGHT), (310, 95))
    return layer.filter(ImageFilter.GaussianBlur(2))


def composite_below_header(
    canvas: Image.Image,
    layer: Image.Image,
    *,
    top: int = 91,
) -> None:
    canvas.alpha_composite(layer.crop((0, top, BOARD_SIZE[0], BOARD_SIZE[1])), (0, top))


def add_omen_field(
    canvas: Image.Image,
    *,
    band: BandStyle,
    center: tuple[int, int] = (512, 326),
) -> None:
    center_x, center_y = center
    aura = Image.new("RGBA", BOARD_SIZE, (0, 0, 0, 0))
    aura_draw = ImageDraw.Draw(aura)
    for radius, alpha in (
        (310, band.glow_alpha // 6),
        (230, band.glow_alpha // 4),
        (155, band.glow_alpha // 2),
    ):
        aura_draw.ellipse(
            (
                center_x - radius * 1.45,
                center_y - radius,
                center_x + radius * 1.45,
                center_y + radius,
            ),
            fill=(*band.glow, alpha),
        )
    canvas.alpha_composite(aura.filter(ImageFilter.GaussianBlur(54)))
    composite_below_header(
        canvas,
        chain_shadow(BOARD_SIZE, band.containment),
    )

    geometry = Image.new("RGBA", BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(geometry)
    ring_count = 3 + band.containment * 2
    for index in range(ring_count):
        radius_x = 190 + index * 31
        radius_y = 142 + index * 19
        start = 190 + (index % 3) * 15
        end = 350 - (index % 2) * 13
        draw.arc(
            (
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            ),
            start,
            end,
            fill=(*band.highlight, 96 + band.containment * 25),
            width=2 + band.containment,
        )
        draw.arc(
            (
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            ),
            10,
            165,
            fill=(0, 0, 0, 150 + band.containment * 20),
            width=5 + band.containment,
        )
    spoke_count = 4 + band.containment * 2
    for index in range(spoke_count):
        angle = math.radians(index * (360 / spoke_count) - 90)
        inner_x = center_x + math.cos(angle) * 190
        inner_y = center_y + math.sin(angle) * 133
        outer_x = center_x + math.cos(angle) * 410
        outer_y = center_y + math.sin(angle) * 282
        draw.line(
            (inner_x, inner_y, outer_x, outer_y),
            fill=(0, 0, 0, 120 + band.containment * 25),
            width=4 + band.containment,
        )
        draw.ellipse(
            (inner_x - 8, inner_y - 8, inner_x + 8, inner_y + 8),
            fill=(*band.highlight, 170),
            outline=(0, 0, 0, 255),
            width=3,
        )
    composite_below_header(canvas, geometry)


def add_foreground_locks(
    canvas: Image.Image,
    *,
    band: BandStyle,
    card_box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = card_box
    layer = Image.new("RGBA", BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    lock_count = max(1, band.containment)
    y_positions = [
        round(top + (bottom - top) * (index + 1) / (lock_count + 1))
        for index in range(lock_count)
    ]
    for index, y in enumerate(y_positions):
        side = -1 if index % 2 == 0 else 1
        anchor_x = left - 34 if side < 0 else right + 34
        draw.arc(
            (anchor_x - 23, y - 29, anchor_x + 23, y + 20),
            185,
            355,
            fill=(*band.highlight, 230),
            width=6,
        )
        draw.rounded_rectangle(
            (anchor_x - 20, y, anchor_x + 20, y + 36),
            radius=6,
            fill=(3, 6, 11, 250),
            outline=(*band.mid, 255),
            width=4,
        )
        draw.line(
            (
                anchor_x + side * 18,
                y + 18,
                center_x := (left + right) // 2,
                y + side * 12,
            ),
            fill=(0, 0, 0, 220),
            width=8,
        )
        draw.line(
            (anchor_x + side * 18, y + 18, center_x, y + side * 12),
            fill=(*band.highlight, 92),
            width=2,
        )
    canvas.alpha_composite(layer)


def sealed_batch_page(frame: Image.Image) -> tuple[Image.Image, list[str]]:
    canvas = system_canvas()
    add_header(
        canvas,
        kicker="SYSTEM ACQUISITION",
        title="THREE SIGNATURES SEALED",
        progress="BATCH 01",
    )
    backs: list[Image.Image] = []
    for _ in range(3):
        backs.append(
            sealed_card(frame, BANDS["neutral"]).resize(
                (186, 298),
                Image.Resampling.LANCZOS,
            )
        )
    for index, (back, x, y) in enumerate(
        zip(backs, (197, 419, 641), (151, 123, 151), strict=True),
        start=1,
    ):
        place_with_shadow(canvas, back, (x, y))
        draw = ImageDraw.Draw(canvas)
        label = f"SIGNATURE {index}"
        face = font(SANS_BOLD, 12)
        label_width, _ = text_size(draw, label, face)
        draw.text(
            (x + (186 - label_width) / 2, 469),
            label,
            font=face,
            fill=(144, 169, 186, 255),
        )
    draw = ImageDraw.Draw(canvas)
    draw_centered(
        draw,
        "No batch-level rarity signal",
        523,
        font(SANS, 14),
        fill=(118, 140, 156, 255),
    )
    return canvas, [
        "SYSTEM ACQUISITION",
        "THREE SIGNATURES SEALED",
        "BATCH 01",
        "SIGNATURE 1",
        "SIGNATURE 2",
        "SIGNATURE 3",
        "No batch-level rarity signal",
    ]


def result_page(
    *,
    card: Image.Image,
    name: str,
    stars: int,
    slot: int,
    total: int,
    veiled: bool,
) -> tuple[Image.Image, list[str]]:
    band_key = "iron" if stars <= 2 else "silver" if stars <= 4 else "gold"
    band = BANDS[band_key]
    canvas = system_canvas()
    add_omen_field(canvas, band=band, center=(279, 332))
    add_header(
        canvas,
        kicker="HERO ACQUIRED",
        title=name.upper(),
        progress=f"PULL {slot} / {total}",
        accent=band.highlight,
    )
    thumb = card.convert("RGBa").resize(
        (276, 442),
        Image.Resampling.LANCZOS,
    ).convert("RGBA")
    place_with_shadow(canvas, thumb, (116, 109), blur=18)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (459, 159),
        "RANK RESOLVED",
        font=font(SANS_BOLD, 14),
        fill=(*band.highlight, 255),
    )
    rank_label = f"{stars}-STAR HERO"
    draw.text(
        (454, 187),
        rank_label,
        font=font(SERIF_BOLD, 42),
        fill=(245, 241, 228, 255),
    )
    star_text = "★" * stars
    spacing = 66
    start_x = 481
    for index in range(stars):
        center_x = start_x + index * spacing
        draw.polygon(
            hero_cards._star_points(center_x, 273, 27, 11.5),  # noqa: SLF001
            fill=(*band.highlight, 255),
            outline=(*band.shadow, 255),
            width=3,
        )
        draw.polygon(
            hero_cards._star_points(center_x - 2, 270, 13, 5.5),  # noqa: SLF001
            fill=(255, 255, 245, 180),
        )
    draw.line((456, 313, 926, 313), fill=(*band.highlight, 85), width=1)
    status = "SYSTEM VEIL ACTIVE" if veiled else "IDENTITY REVEALED"
    draw.text(
        (458, 337),
        status,
        font=font(SANS_BOLD, 18),
        fill=(176, 195, 207, 255),
    )
    copy = (
        "The Hero is named. The face remains withheld until promotion."
        if veiled
        else "The contained signature resolves into a reviewed Hero portrait."
    )
    words = copy.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if text_size(draw, candidate, font(SANS, 16))[0] <= 470:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line_index, line in enumerate(lines):
        draw.text(
            (458, 377 + line_index * 25),
            line,
            font=font(SANS, 16),
            fill=(139, 160, 176, 255),
        )
    draw.text(
        (458, 490),
        "NEXT SIGNATURE" if slot < total else "VIEW RESULTS",
        font=font(SANS_BOLD, 15),
        fill=(*band.highlight, 255),
    )
    return canvas, [
        "HERO ACQUIRED",
        name.upper(),
        f"PULL {slot} / {total}",
        "RANK RESOLVED",
        rank_label,
        star_text,
        status,
        copy,
        "NEXT SIGNATURE" if slot < total else "VIEW RESULTS",
    ]


def omen_page(
    *,
    frame: Image.Image,
    band_key: str,
    slot: int,
    total: int,
    title: str = "THE SEAL RESISTS",
) -> tuple[Image.Image, list[str]]:
    band = BANDS[band_key]
    canvas = system_canvas()
    add_omen_field(canvas, band=band)
    add_header(
        canvas,
        kicker="SIGNATURE ANOMALY",
        title=title,
        progress=f"PULL {slot} / {total}",
        accent=band.highlight,
    )
    card = sealed_card(frame, band).resize(
        (264, 422),
        Image.Resampling.LANCZOS,
    )
    position = (380, 105)
    place_with_shadow(canvas, card, position, blur=22)
    add_foreground_locks(
        canvas,
        band=band,
        card_box=(position[0], position[1], position[0] + 264, position[1] + 422),
    )
    draw = ImageDraw.Draw(canvas)
    left_label = "METAL RESPONSE DETECTED"
    draw.text(
        (34, 513),
        left_label,
        font=font(SANS_BOLD, 13),
        fill=(*band.highlight, 255),
    )
    right_label = "EXACT RANK WITHHELD"
    right_face = font(SANS_BOLD, 13)
    right_width, _ = text_size(draw, right_label, right_face)
    draw.text(
        (990 - right_width, 513),
        right_label,
        font=right_face,
        fill=(177, 190, 201, 255),
    )
    return canvas, [
        "SIGNATURE ANOMALY",
        title,
        f"PULL {slot} / {total}",
        "SEALED",
        left_label,
        right_label,
    ]


def results_page(cards: dict[str, Image.Image]) -> tuple[Image.Image, list[str]]:
    board = hero_cards._render_board(  # noqa: SLF001 - exact reviewed board
        [cards["renna"], cards["mirelle"], cards["edren"]],
        kind="summon",
        page_number=1,
        page_count=1,
    ).convert("RGBA")
    return board, [
        "HEROES ACQUIRED",
        "CURRENT HERO RANK",
        "Renna Holt",
        "Mirelle Voss",
        "Edren Marr",
    ]


def mini_band_panel(
    frame: Image.Image,
    band: BandStyle,
    *,
    size: tuple[int, int] = (229, 390),
) -> Image.Image:
    width, height = size
    panel = gradient(size, (9, 17, 28), (1, 3, 8)).convert("RGBA")
    aura = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(aura)
    draw.ellipse(
        (14, 68, width - 14, height - 63),
        fill=(*band.glow, max(18, band.glow_alpha // 2)),
    )
    panel.alpha_composite(aura.filter(ImageFilter.GaussianBlur(28)))
    rings = Image.new("RGBA", size, (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(rings)
    for index in range(2 + band.containment * 2):
        inset = 9 + index * 10
        ring_draw.arc(
            (inset, 62 + inset // 2, width - inset, height - 51 - inset // 2),
            190,
            350,
            fill=(*band.highlight, 80 + 24 * band.containment),
            width=2 + band.containment,
        )
    panel.alpha_composite(rings)
    card = sealed_card(frame, band).resize((126, 202), Image.Resampling.LANCZOS)
    place_with_shadow(panel, card, ((width - 126) // 2, 90), blur=10)
    panel_draw = ImageDraw.Draw(panel)
    label_face = font(SERIF_BOLD, 22)
    label_width, _ = text_size(panel_draw, band.label, label_face)
    panel_draw.text(
        ((width - label_width) / 2, 16),
        band.label,
        font=label_face,
        fill=(*band.highlight, 255),
    )
    range_face = font(SANS_BOLD, 13)
    range_width, _ = text_size(panel_draw, band.range_label, range_face)
    panel_draw.text(
        ((width - range_width) / 2, 324),
        band.range_label,
        font=range_face,
        fill=(198, 205, 209, 255),
    )
    locks = f"CONTAINMENT {band.containment}"
    lock_face = font(SANS, 11)
    lock_width, _ = text_size(panel_draw, locks, lock_face)
    panel_draw.text(
        ((width - lock_width) / 2, 350),
        locks,
        font=lock_face,
        fill=(126, 146, 160, 255),
    )
    panel_draw.rectangle(
        (0, 0, width - 1, height - 1),
        outline=(*band.mid, 190),
        width=2,
    )
    return panel


def band_escalation_page(frame: Image.Image) -> tuple[Image.Image, list[str]]:
    canvas = system_canvas()
    add_header(
        canvas,
        kicker="PROOF  //  PER-CARD OMEN",
        title="ONE PALETTE, FOUR CONTAINMENT STATES",
        progress="EXACT RANK STAYS SEALED",
    )
    x_positions = (28, 274, 520, 766)
    visible = [
        "PROOF  //  PER-CARD OMEN",
        "ONE PALETTE, FOUR CONTAINMENT STATES",
        "EXACT RANK STAYS SEALED",
    ]
    for key, x in zip(("iron", "silver", "gold", "white_gold"), x_positions, strict=True):
        band = BANDS[key]
        panel = mini_band_panel(frame, band)
        canvas.alpha_composite(panel, (x, 119))
        visible.extend((band.label, band.range_label, f"CONTAINMENT {band.containment}"))
    draw = ImageDraw.Draw(canvas)
    draw_centered(
        draw,
        "Hue remains metal-only; geometry and light carry the escalation.",
        528,
        font(SANS, 14),
        fill=(139, 161, 177, 255),
    )
    visible.append(
        "Hue remains metal-only; geometry and light carry the escalation."
    )
    return canvas, visible


def ambiguity_page(frame: Image.Image) -> tuple[Image.Image, list[str]]:
    canvas = system_canvas()
    add_header(
        canvas,
        kicker="PROOF  //  RARITY INFORMATION",
        title="THE OMEN PROMISES A BAND, NOT AN EXACT RESULT",
        progress="NO FAKEOUTS",
    )
    draw = ImageDraw.Draw(canvas)
    panels = (
        (BANDS["silver"], 32, "3★ POSSIBLE", "4★ POSSIBLE"),
        (BANDS["gold"], 524, "5★ POSSIBLE", "6★ POSSIBLE"),
    )
    visible = [
        "PROOF  //  RARITY INFORMATION",
        "THE OMEN PROMISES A BAND, NOT AN EXACT RESULT",
        "NO FAKEOUTS",
    ]
    for band, left, first, second in panels:
        draw.rounded_rectangle(
            (left, 116, left + 468, 515),
            radius=18,
            fill=(3, 7, 14, 215),
            outline=(*band.mid, 190),
            width=2,
        )
        title_face = font(SERIF_BOLD, 24)
        title_width, _ = text_size(draw, f"{band.label} OMEN", title_face)
        draw.text(
            (left + (468 - title_width) / 2, 132),
            f"{band.label} OMEN",
            font=title_face,
            fill=(*band.highlight, 255),
        )
        omen = sealed_card(frame, band).resize((126, 202), Image.Resampling.LANCZOS)
        for index, label in enumerate((first, second)):
            x = left + 62 + index * 220
            glow = Image.new("RGBA", BOARD_SIZE, (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow)
            glow_draw.ellipse(
                (x - 25, 177, x + 151, 411),
                fill=(*band.glow, band.glow_alpha // 2),
            )
            canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(24)))
            place_with_shadow(canvas, omen, (x, 190), blur=10)
            label_face = font(SANS_BOLD, 13)
            label_width, _ = text_size(draw, label, label_face)
            draw.text(
                (x + (126 - label_width) / 2, 420),
                label,
                font=label_face,
                fill=(207, 211, 210, 255),
            )
        shared = "IDENTICAL PRE-REVEAL ART"
        shared_face = font(SANS, 12)
        shared_width, _ = text_size(draw, shared, shared_face)
        draw.text(
            (left + (468 - shared_width) / 2, 474),
            shared,
            font=shared_face,
            fill=(125, 148, 163, 255),
        )
        visible.extend((f"{band.label} OMEN", first, second, shared))
    return canvas, visible


def draw_storyboard_tile(
    canvas: Image.Image,
    *,
    box: tuple[int, int, int, int],
    number: int,
    title: str,
    kind: str,
) -> None:
    left, top, right, bottom = box
    draw = ImageDraw.Draw(canvas)
    accent = (
        BANDS["gold"].highlight
        if kind in {"omen", "special"}
        else (151, 175, 191)
    )
    draw.rounded_rectangle(
        box,
        radius=14,
        fill=(3, 7, 14, 230),
        outline=(*accent, 145),
        width=2,
    )
    draw.ellipse(
        (left + 13, top + 12, left + 41, top + 40),
        fill=(*accent, 220),
    )
    number_face = font(SANS_BOLD, 13)
    number_width, _ = text_size(draw, str(number), number_face)
    draw.text(
        (left + 27 - number_width / 2, top + 18),
        str(number),
        font=number_face,
        fill=(4, 7, 12, 255),
    )
    title_face = font(SANS_BOLD, 12)
    draw.text(
        (left + 52, top + 18),
        title,
        font=title_face,
        fill=(220, 221, 215, 255),
    )
    center_x = (left + right) // 2
    center_y = top + 106
    if kind == "batch":
        for offset in (-34, 0, 34):
            draw.rounded_rectangle(
                (center_x + offset - 24, center_y - 45, center_x + offset + 24, center_y + 43),
                radius=7,
                fill=(5, 11, 21, 255),
                outline=(113, 132, 145, 210),
                width=2,
            )
    elif kind == "omen":
        for radius in (45, 63, 79):
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                outline=(*BANDS["gold"].highlight, 92),
                width=3,
            )
        draw.rounded_rectangle(
            (center_x - 28, center_y - 53, center_x + 28, center_y + 52),
            radius=8,
            fill=(5, 9, 15, 255),
            outline=(*BANDS["gold"].highlight, 255),
            width=3,
        )
    elif kind == "results":
        for offset in (-58, -29, 0, 29, 58):
            draw.rounded_rectangle(
                (center_x + offset - 12, center_y - 39, center_x + offset + 12, center_y + 35),
                radius=4,
                fill=(7, 13, 23, 255),
                outline=(163, 144, 101, 230),
                width=2,
            )
    else:
        draw.rounded_rectangle(
            (center_x - 38, center_y - 54, center_x + 38, center_y + 55),
            radius=8,
            fill=(7, 13, 23, 255),
            outline=(*accent, 220),
            width=3,
        )
        draw.polygon(
            hero_cards._star_points(center_x, center_y + 31, 12, 5),  # noqa: SLF001
            fill=(*accent, 255),
        )


def five_pull_storyboard() -> tuple[Image.Image, list[str]]:
    canvas = system_canvas()
    add_header(
        canvas,
        kicker="PROOF  //  PAGING",
        title="FIVE PULLS, ONE SPECIAL: EIGHT PRESENTATION PAGES",
        progress="FRONTEND-AGNOSTIC",
    )
    titles = (
        ("BATCH SEALED", "batch"),
        ("PULL 1", "direct"),
        ("PULL 2", "direct"),
        ("SPECIAL OMEN", "omen"),
        ("PULL 3 REVEAL", "special"),
        ("PULL 4", "direct"),
        ("PULL 5", "direct"),
        ("RESULTS", "results"),
    )
    positions = [
        (31 + column * 244, 113 + row * 211, 255 + column * 244, 302 + row * 211)
        for row in range(2)
        for column in range(4)
    ]
    for number, ((title, kind), position) in enumerate(
        zip(titles, positions, strict=True),
        start=1,
    ):
        draw_storyboard_tile(
            canvas,
            box=position,
            number=number,
            title=title,
            kind=kind,
        )
    draw = ImageDraw.Draw(canvas)
    draw_centered(
        draw,
        "Only 3★+ inserts an omen page; direct pulls still receive a full result beat.",
        540,
        font(SANS, 14),
        fill=(143, 164, 179, 255),
    )
    return canvas, [
        "PROOF  //  PAGING",
        "FIVE PULLS, ONE SPECIAL: EIGHT PRESENTATION PAGES",
        "FRONTEND-AGNOSTIC",
        *(title for title, _kind in titles),
        "Only 3★+ inserts an omen page; direct pulls still receive a full result beat.",
    ]


def contact_sheet(paths: list[Path]) -> tuple[Image.Image, list[str]]:
    canvas = system_canvas()
    add_header(
        canvas,
        kicker="ONE-STAR SUMMON REVEAL",
        title="REVIEW CONTACT SHEET",
        progress="10 PROOFS",
    )
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths, start=1):
        row, column = divmod(index - 1, 5)
        left = 22 + column * 199
        top = 112 + row * 216
        with Image.open(path) as received:
            thumb = ImageOps.fit(
                received.convert("RGB"),
                (184, 104),
                method=Image.Resampling.LANCZOS,
            )
        canvas.paste(thumb, (left, top))
        draw.rectangle(
            (left, top, left + 183, top + 103),
            outline=(154, 130, 83, 150),
            width=1,
        )
        label = f"{index:02d}  {path.stem.split('_', 1)[1].replace('_', ' ').upper()}"
        label = label[:29]
        draw.text(
            (left, top + 114),
            label,
            font=font(SANS_BOLD, 10),
            fill=(177, 193, 202, 255),
        )
    draw_centered(
        draw,
        "Review the six opening pages in order before the supporting contract sheets.",
        538,
        font(SANS, 13),
        fill=(137, 159, 174, 255),
    )
    return canvas, [
        "ONE-STAR SUMMON REVEAL",
        "REVIEW CONTACT SHEET",
        "10 PROOFS",
        "Review the six opening pages in order before the supporting contract sheets.",
    ]


def save_rgb(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(
        path,
        format="PNG",
        optimize=True,
        compress_level=9,
    )


def write_review_text() -> Path:
    path = REVIEW / "00_START_HERE.txt"
    path.write_text(
        """ONE-STAR RARITY-CRESCENDO PULL REVEAL — VISUAL PROOF ONLY

OPENING SEQUENCE
01  Three neutral sealed signatures; no batch-level rarity clue
02  Renna direct 1-star result; the existing identity veil remains
03  Silver-band omen; exact 3-or-4-star result is still hidden
04  Mirelle 3-star result; card and exact rank appear together
05  Edren direct 1-star result; the existing identity veil remains
06  Existing three-Hero results board

SUPPORTING CONTRACT PROOFS
07  Iron / silver / gold / white-gold containment escalation
08  Identical 3-vs-4 and 5-vs-6 pre-reveal omens
09  Eight-page pacing for a five-pull batch with one special result
10  Maximum white-gold containment treatment
11  Contact sheet

The proposed reveal is static and transport-neutral. Motion, timing, sound,
frontend controls, narrator prose, and production bindings are intentionally
outside this review. Portraits, Option F metal ranks, identity veils, and the
Obsidian Orrery card geometry remain unchanged.
""",
        encoding="utf-8",
    )
    return path


def write_index(records: list[dict[str, object]]) -> Path:
    figures = []
    for record in records:
        filename = html.escape(str(record["filename"]))
        alt = html.escape(str(record["accessible_text"]))
        label = html.escape(str(record["label"]))
        figures.append(
            f'<figure><a href="{filename}"><img src="{filename}" alt="{alt}"></a>'
            f"<figcaption>{label}</figcaption></figure>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>One-Star Pull Reveal — Visual Proofs</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; background: #030711; color: #eee9db; }}
body {{ margin: 0; padding: 2rem; }}
h1 {{ font-family: Georgia, serif; letter-spacing: .035em; }}
p {{ color: #a9bcc9; max-width: 70rem; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(28rem, 1fr)); gap: 1.25rem; }}
figure {{ margin: 0; border: 1px solid #765e36; background: #07101e; padding: .65rem; }}
img {{ display: block; width: 100%; height: auto; }}
figcaption {{ padding: .65rem .2rem .1rem; color: #d5bd87; }}
</style>
</head>
<body>
<h1>One-Star Rarity-Crescendo Pull Reveal</h1>
<p>Visual proof only. Review 01–06 as one opening sequence, then 07–10 as the information and pacing contract.</p>
<main>{''.join(figures)}</main>
</body>
</html>
"""
    path = REVIEW / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def source_hashes() -> dict[str, str]:
    return {
        "reviewed_orrery_frame": sha256(FRAME_PATH),
        "reviewed_veiled_feminine_bust": sha256(PORTRAIT_PATHS["renna_veiled"]),
        "reviewed_mirelle_neutral_sprite": sha256(PORTRAIT_PATHS["mirelle"]),
        "reviewed_veiled_masculine_bust": sha256(PORTRAIT_PATHS["edren_veiled"]),
    }


def public_record(
    path: Path,
    *,
    label: str,
    visible_text: list[str],
) -> dict[str, object]:
    return {
        "filename": path.name,
        "label": label,
        "dimensions": list(BOARD_SIZE),
        "sha256": sha256(path),
        "accessible_text": "; ".join(visible_text),
        "visible_text": visible_text,
    }


def write_manifests(records: list[dict[str, object]]) -> None:
    card_records = []
    for key, stars in (("renna", 1), ("mirelle", 3), ("edren", 1)):
        path = PROOFS / "result_cards" / f"{key}.png"
        card_records.append(
            {
                "role": key,
                "stars": stars,
                "dimensions": list(hero_cards.CARD_SIZE),
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "visual_review_only",
        "direction": "rarity_crescendo",
        "production_binding": False,
        "frontend_contract": "static semantic pages at 1024x576",
        "frozen_invariants": {
            "card_frame": "approved Obsidian Orrery geometry",
            "card_rank_style": "approved Option F metal milestones",
            "portrait_layering": "portrait over border, under nameplate and stars",
            "identity_rules": "existing veils remain active",
            "batch_rarity_signal": "none",
            "exact_rank_before_result": "withheld",
            "extra_omen_threshold": "3 stars and above",
            "fakeouts": "none",
            "excluded_subjects": ["Eighth Warden"],
        },
        "metal_bands": {
            key: asdict(BANDS[key])
            for key in ("iron", "silver", "gold", "white_gold")
        },
        "rank_to_pre_reveal_omen": {
            "1": "direct",
            "2": "direct",
            "3": "silver",
            "4": "silver",
            "5": "gold",
            "6": "gold",
            "7": "white_gold",
        },
        "opening_sequence": [record["filename"] for record in records[:6]],
        "five_pull_page_count_with_one_special": 8,
        "result_cards": card_records,
        "source_hashes": source_hashes(),
        "review": records,
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (ROOT / "proof_manifest.json").write_text(encoded, encoding="utf-8")
    (REVIEW / "manifest.json").write_text(encoded, encoding="utf-8")
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    (PROVENANCE / "source_hashes.json").write_text(
        json.dumps(source_hashes(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (PROVENANCE / "decisions.json").write_text(
        json.dumps(
            {
                "approved_card_treatment": "Option F metal milestones",
                "approved_reveal_direction": "rarity crescendo",
                "batch_signal": "neutral",
                "per_card_omen": True,
                "omen_threshold": 3,
                "horror_motif": "containment geometry",
                "portrait_changes": False,
                "production_binding": False,
                "image_generation_calls": 0,
                "rejected_attempts": [
                    {
                        "artifact_preserved": False,
                        "issue": "font-dependent rank stars rendered as missing-glyph boxes",
                        "resolution": "replaced all large result stars with deterministic vector geometry",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_hashes() -> Path:
    path = REVIEW / "SHA256SUMS.txt"
    entries = [item for item in sorted(REVIEW.iterdir()) if item.name != path.name]
    path.write_text(
        "".join(f"{sha256(item)}  {item.name}\n" for item in entries if item.is_file()),
        encoding="utf-8",
    )
    provenance_path = PROVENANCE / "SHA256SUMS.txt"
    root_entries = sorted(PROOFS.rglob("*.png"))
    root_entries.extend((ROOT / "proof_manifest.json", PROVENANCE / "decisions.json"))
    provenance_path.write_text(
        "".join(
            f"{sha256(item)}  {item.relative_to(ROOT).as_posix()}\n"
            for item in root_entries
        ),
        encoding="utf-8",
    )
    return path


def copy_windows_review() -> None:
    WINDOWS_REVIEW.mkdir(parents=True, exist_ok=True)
    for source in sorted(REVIEW.iterdir()):
        if source.is_file():
            shutil.copyfile(source, WINDOWS_REVIEW / source.name)


def main() -> None:
    PROOFS.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    with Image.open(FRAME_PATH) as received:
        frame = received.convert("RGBA")
    if frame.size != hero_cards.CARD_SIZE:
        raise ValueError(f"unexpected frame size: {frame.size}")
    cards = build_cards(frame)

    pages: list[tuple[str, str, Image.Image, list[str]]] = []
    batch, text = sealed_batch_page(frame)
    pages.append(("01", "opening_01_batch_sealed", batch, text))
    renna, text = result_page(
        card=cards["renna"],
        name="Renna Holt",
        stars=1,
        slot=1,
        total=3,
        veiled=True,
    )
    pages.append(("02", "opening_02_renna_direct", renna, text))
    silver, text = omen_page(
        frame=frame,
        band_key="silver",
        slot=2,
        total=3,
    )
    pages.append(("03", "opening_03_silver_omen", silver, text))
    mirelle, text = result_page(
        card=cards["mirelle"],
        name="Mirelle Voss",
        stars=3,
        slot=2,
        total=3,
        veiled=False,
    )
    pages.append(("04", "opening_04_mirelle_reveal", mirelle, text))
    edren, text = result_page(
        card=cards["edren"],
        name="Edren Marr",
        stars=1,
        slot=3,
        total=3,
        veiled=True,
    )
    pages.append(("05", "opening_05_edren_direct", edren, text))
    results, text = results_page(cards)
    pages.append(("06", "opening_06_results", results, text))
    ladder, text = band_escalation_page(frame)
    pages.append(("07", "metal_band_escalation", ladder, text))
    ambiguity, text = ambiguity_page(frame)
    pages.append(("08", "rank_ambiguity_contract", ambiguity, text))
    storyboard, text = five_pull_storyboard()
    pages.append(("09", "five_pull_pacing", storyboard, text))
    peak, text = omen_page(
        frame=frame,
        band_key="white_gold",
        slot=5,
        total=5,
        title="CONTAINMENT AT ITS LIMIT",
    )
    pages.append(("10", "white_gold_peak_omen", peak, text))

    records: list[dict[str, object]] = []
    page_paths: list[Path] = []
    for number, slug, image, visible_text in pages:
        path = REVIEW / f"{number}_{slug}.png"
        save_rgb(image, path)
        page_paths.append(path)
        records.append(
            public_record(
                path,
                label=slug.replace("_", " ").title(),
                visible_text=visible_text,
            )
        )
    sheet, visible_text = contact_sheet(page_paths)
    sheet_path = REVIEW / "11_contact_sheet.png"
    save_rgb(sheet, sheet_path)
    records.append(
        public_record(
            sheet_path,
            label="Contact Sheet",
            visible_text=visible_text,
        )
    )

    write_review_text()
    write_index(records)
    write_manifests(records)
    write_hashes()
    copy_windows_review()

    for record in records:
        print(f"{record['filename']}  {record['sha256']}")
    print(REVIEW / "index.html")
    print(WINDOWS_REVIEW)


if __name__ == "__main__":
    main()
