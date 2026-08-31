#!/usr/bin/env python3
"""Build deterministic animated proofs for the One-Star summon reveal.

This experiment deliberately keeps every semantic result on the existing
static board.  Each GIF starts and ends with a sealed card, so animation can
be skipped without hiding identity, rank, or acquisition state.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import build_proofs as base


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review_animation"
PROOFS = ROOT / "proofs/animation"
KEYFRAMES = PROOFS / "keyframes"
PROVENANCE = ROOT / "provenance"
MANIFEST = ROOT / "animation_proof_manifest.json"
WINDOWS_REVIEW = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/OneStarPullRevealAnimationReview_20260831"
)


@dataclass(frozen=True)
class MotionTier:
    key: str
    band_key: str
    review_label: str
    filename: str
    kicker: str
    title: str
    footer: str
    frame_count: int
    frame_ms: int
    final_hold_ms: int
    lock_count: int
    flash_time: float
    intensity: int


TIERS = (
    MotionTier(
        key="under_three",
        band_key="iron",
        review_label="UNDER 3 STARS // PLAIN IRON RESPONSE",
        filename="01_iron_plain_under_3.gif",
        kicker="SYSTEM ACQUISITION",
        title="SIGNATURE STABILIZED",
        footer="STANDARD RESPONSE",
        frame_count=16,
        frame_ms=60,
        final_hold_ms=340,
        lock_count=0,
        flash_time=0.62,
        intensity=1,
    ),
    MotionTier(
        key="three_to_four",
        band_key="silver",
        review_label="3-4 STARS // SILVER RESPONSE",
        filename="02_silver_omen_3_to_4.gif",
        kicker="SIGNATURE ANOMALY",
        title="THE SEAL RESISTS",
        footer="METAL RESPONSE DETECTED",
        frame_count=28,
        frame_ms=50,
        final_hold_ms=480,
        lock_count=2,
        flash_time=0.70,
        intensity=2,
    ),
    MotionTier(
        key="five_to_six",
        band_key="gold",
        review_label="5-6 STARS // GOLD RESPONSE",
        filename="03_gold_omen_5_to_6.gif",
        kicker="SIGNATURE ANOMALY",
        title="THE SEAL RESISTS",
        footer="METAL RESPONSE DETECTED",
        frame_count=34,
        frame_ms=50,
        final_hold_ms=560,
        lock_count=3,
        flash_time=0.74,
        intensity=3,
    ),
    MotionTier(
        key="seven",
        band_key="white_gold",
        review_label="7 STARS // WHITE-GOLD RESPONSE",
        filename="04_white_gold_omen_7.gif",
        kicker="SIGNATURE ANOMALY",
        title="CONTAINMENT AT ITS LIMIT",
        footer="METAL RESPONSE DETECTED",
        frame_count=42,
        frame_ms=50,
        final_hold_ms=700,
        lock_count=4,
        flash_time=0.77,
        intensity=4,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smoothstep(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def gaussian(value: float, center: float, width: float) -> float:
    return math.exp(-((value - center) / width) ** 2)


def alpha_composite_below_header(
    canvas: Image.Image,
    layer: Image.Image,
    *,
    top: int = 91,
) -> None:
    canvas.alpha_composite(
        layer.crop((0, top, base.BOARD_SIZE[0], base.BOARD_SIZE[1])),
        (0, top),
    )


def aura_layer(
    tier: MotionTier,
    *,
    progress: float,
    center: tuple[int, int],
) -> Image.Image:
    band = base.BANDS[tier.band_key]
    pulse = 0.55 + 0.45 * math.sin(progress * math.tau * (1 + tier.intensity * 0.22))
    closure = smoothstep((progress - 0.08) / 0.68)
    flash = gaussian(progress, tier.flash_time, 0.10 - tier.intensity * 0.008)
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    center_x, center_y = center
    for index, radius in enumerate((330, 245, 170)):
        alpha = round(
            (7 + tier.intensity * (5 + index * 4))
            * (0.68 + pulse * 0.32)
            * (0.55 + closure * 0.45)
            + flash * tier.intensity * (4 + index * 3)
        )
        draw.ellipse(
            (
                center_x - radius * 1.42,
                center_y - radius,
                center_x + radius * 1.42,
                center_y + radius,
            ),
            fill=(*band.glow, min(120, alpha)),
        )
    return layer.filter(ImageFilter.GaussianBlur(50 - tier.intensity * 3))


def ring_layer(
    tier: MotionTier,
    *,
    progress: float,
    center: tuple[int, int],
) -> Image.Image:
    band = base.BANDS[tier.band_key]
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    center_x, center_y = center
    closure = smoothstep((progress - 0.05) / 0.72)
    ring_count = 1 if tier.intensity == 1 else 2 + tier.intensity
    for index in range(ring_count):
        settled_x = 188 + index * (24 if tier.intensity > 1 else 0)
        settled_y = 143 + index * (16 if tier.intensity > 1 else 0)
        radius_x = settled_x + (1.0 - closure) * (135 + index * 18)
        radius_y = settled_y + (1.0 - closure) * (88 + index * 11)
        spin = progress * (75 + index * 32) * (-1 if index % 2 else 1)
        start = (205 + index * 37 + spin) % 360
        span = 92 if tier.intensity == 1 else 116 + tier.intensity * 8
        alpha = 42 + tier.intensity * 31
        width = 2 if tier.intensity == 1 else 2 + tier.intensity // 2
        box = (
            center_x - radius_x,
            center_y - radius_y,
            center_x + radius_x,
            center_y + radius_y,
        )
        draw.arc(
            box,
            start,
            start + span,
            fill=(*band.highlight, min(238, alpha)),
            width=width,
        )
        if tier.intensity > 1:
            draw.arc(
                box,
                start + 180,
                start + 180 + span * 0.72,
                fill=(*band.mid, min(210, alpha + 18)),
                width=width + 1,
            )
    if tier.intensity > 1:
        spoke_count = tier.intensity * 2
        spoke_alpha = round((35 + tier.intensity * 16) * closure)
        for index in range(spoke_count):
            angle = math.radians(
                index * (360 / spoke_count) - 90 + progress * 12
            )
            inner_x = center_x + math.cos(angle) * 195
            inner_y = center_y + math.sin(angle) * 140
            outer_x = center_x + math.cos(angle) * (315 + tier.intensity * 12)
            outer_y = center_y + math.sin(angle) * (220 + tier.intensity * 8)
            draw.line(
                (inner_x, inner_y, outer_x, outer_y),
                fill=(0, 0, 0, min(190, spoke_alpha + 65)),
                width=2 + tier.intensity,
            )
            draw.ellipse(
                (inner_x - 5, inner_y - 5, inner_x + 5, inner_y + 5),
                fill=(*band.highlight, spoke_alpha),
            )
    return layer


def chain_layer(
    tier: MotionTier,
    *,
    progress: float,
) -> Image.Image:
    if tier.intensity < 3:
        return Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    band = base.BANDS[tier.band_key]
    closure = smoothstep((progress - 0.18) / 0.60)
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    chain = Image.new("RGBA", (910, 76), (0, 0, 0, 0))
    draw = ImageDraw.Draw(chain)
    for x in range(-40, 940, 50 - tier.intensity * 2):
        draw.ellipse(
            (x, 16, x + 65, 60),
            outline=(*band.mid, round(72 + closure * 84)),
            width=7 + tier.intensity,
        )
        draw.ellipse(
            (x + 4, 20, x + 61, 56),
            outline=(0, 0, 0, round(100 + closure * 90)),
            width=3,
        )
    offset = round((1.0 - closure) * 90)
    first = chain.rotate(-21, resample=Image.Resampling.BICUBIC, expand=True)
    second = chain.rotate(21, resample=Image.Resampling.BICUBIC, expand=True)
    layer.alpha_composite(first, (-170 - offset, 54 - offset // 3))
    layer.alpha_composite(second, (302 + offset, 50 - offset // 4))
    return layer.filter(ImageFilter.GaussianBlur(1.2))


def lock_layer(
    tier: MotionTier,
    *,
    progress: float,
    card_box: tuple[int, int, int, int],
) -> Image.Image:
    band = base.BANDS[tier.band_key]
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    if tier.lock_count == 0:
        return layer
    draw = ImageDraw.Draw(layer)
    left, top, right, bottom = card_box
    center_x = (left + right) // 2
    y_positions = [
        round(top + (bottom - top) * (index + 1) / (tier.lock_count + 1))
        for index in range(tier.lock_count)
    ]
    for index, y in enumerate(y_positions):
        start = 0.16 + index * 0.09
        arrival = smoothstep((progress - start) / 0.30)
        side = -1 if index % 2 == 0 else 1
        settled_x = left - 24 if side < 0 else right + 24
        travel = 175 + index * 11
        anchor_x = round(settled_x + side * travel * (1.0 - arrival))
        snap = gaussian(progress, start + 0.30, 0.035)
        tether_end = center_x + side * 7
        draw.line(
            (anchor_x - side * 2, y + 18, tether_end, y + side * 9),
            fill=(0, 0, 0, round(85 + arrival * 150)),
            width=8,
        )
        draw.line(
            (anchor_x - side * 2, y + 18, tether_end, y + side * 9),
            fill=(*band.highlight, round(18 + arrival * 96)),
            width=2,
        )
        draw.arc(
            (anchor_x - 21, y - 28, anchor_x + 21, y + 19),
            185,
            355,
            fill=(*band.highlight, round(80 + arrival * 165)),
            width=5,
        )
        draw.rounded_rectangle(
            (anchor_x - 19, y, anchor_x + 19, y + 34),
            radius=6,
            fill=(3, 6, 11, round(170 + arrival * 80)),
            outline=(*band.mid, round(95 + arrival * 160)),
            width=4,
        )
        if snap > 0.04:
            radius = 15 + round(32 * snap)
            draw.ellipse(
                (
                    anchor_x - radius,
                    y + 17 - radius,
                    anchor_x + radius,
                    y + 17 + radius,
                ),
                outline=(*band.highlight, round(210 * snap)),
                width=2 + tier.intensity,
            )
    return layer


def spark_layer(
    tier: MotionTier,
    *,
    progress: float,
    center: tuple[int, int],
) -> Image.Image:
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    if tier.intensity < 3:
        return layer
    band = base.BANDS[tier.band_key]
    flash = gaussian(progress, tier.flash_time, 0.13)
    if flash < 0.025:
        return layer
    draw = ImageDraw.Draw(layer)
    center_x, center_y = center
    count = 10 + tier.intensity * 5
    for index in range(count):
        angle = math.radians((index * 137.508 + tier.intensity * 17) % 360)
        radial_phase = (index * 0.173 + progress * (1.6 + tier.intensity * 0.12)) % 1
        radius = 105 + radial_phase * (160 + tier.intensity * 24)
        x = center_x + math.cos(angle) * radius
        y = center_y + math.sin(angle) * radius * 0.68
        length = 6 + (index % 5) * 3 + tier.intensity * 2
        alpha = round(flash * (105 + (index % 4) * 32))
        draw.line(
            (
                x,
                y,
                x + math.cos(angle) * length,
                y + math.sin(angle) * length,
            ),
            fill=(*band.highlight, min(245, alpha)),
            width=1 + (index % 3 == 0),
        )
    return layer.filter(ImageFilter.GaussianBlur(0.4))


def flash_layer(tier: MotionTier, *, progress: float) -> Image.Image:
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    if tier.intensity == 1:
        return layer
    band = base.BANDS[tier.band_key]
    peak = gaussian(progress, tier.flash_time, 0.048 + 0.006 * tier.intensity)
    alpha = round(peak * (22 + tier.intensity * 22))
    if alpha:
        wash = Image.new("RGBA", base.BOARD_SIZE, (*band.highlight, alpha))
        layer = Image.alpha_composite(layer, wash)
        center = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(center)
        radius = 118 + tier.intensity * 34
        draw.ellipse(
            (512 - radius * 1.4, 326 - radius, 512 + radius * 1.4, 326 + radius),
            fill=(*band.highlight, min(210, alpha * 2)),
        )
        layer = Image.alpha_composite(
            layer,
            center.filter(ImageFilter.GaussianBlur(50)),
        )
    return layer


def footer_layer(tier: MotionTier) -> Image.Image:
    band = base.BANDS[tier.band_key]
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, 525, 1024, 576), fill=(1, 4, 10, 224))
    draw.line((48, 525, 976, 525), fill=(*band.highlight, 76), width=1)
    draw.text(
        (52, 541),
        tier.footer,
        font=base.font(base.SANS_BOLD, 13),
        fill=(*band.highlight, 255),
    )
    exact = "EXACT RANK WITHHELD"
    face = base.font(base.SANS_BOLD, 13)
    exact_width, _ = base.text_size(draw, exact, face)
    draw.text(
        (972 - exact_width, 541),
        exact,
        font=face,
        fill=(141, 159, 173, 255),
    )
    return layer


def animation_frame(
    tier: MotionTier,
    sealed: Image.Image,
    *,
    progress: float,
) -> Image.Image:
    band = base.BANDS[tier.band_key]
    center = (512, 326)
    canvas = base.system_canvas()
    canvas.alpha_composite(aura_layer(tier, progress=progress, center=center))
    alpha_composite_below_header(
        canvas,
        chain_layer(tier, progress=progress),
    )
    alpha_composite_below_header(
        canvas,
        ring_layer(tier, progress=progress, center=center),
    )

    breath = math.sin(progress * math.tau * 1.35)
    lock_shudder = 0.0
    if tier.intensity > 1:
        for index in range(tier.lock_count):
            lock_shudder += gaussian(progress, 0.46 + index * 0.09, 0.026)
    scale = 0.972 + 0.018 * smoothstep(progress / 0.30) + breath * 0.003
    card_width = round(264 * scale)
    card_height = round(422 * scale)
    card = sealed.resize((card_width, card_height), Image.Resampling.LANCZOS)
    jitter = round(math.sin(progress * 113) * lock_shudder * tier.intensity * 0.65)
    card_left = 512 - card_width // 2 + jitter
    card_top = 316 - card_height // 2
    base.place_with_shadow(canvas, card, (card_left, card_top), blur=18)
    card_box = (card_left, card_top, card_left + card_width, card_top + card_height)

    scan = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    scan_draw = ImageDraw.Draw(scan)
    scan_phase = (progress * (1.05 + tier.intensity * 0.08)) % 1.0
    scan_y = round(card_top + 22 + scan_phase * max(1, card_height - 44))
    scan_alpha = 42 + tier.intensity * 16
    scan_draw.rectangle(
        (card_left + 18, scan_y - 4, card_left + card_width - 18, scan_y + 4),
        fill=(*band.highlight, scan_alpha // 4),
    )
    scan_draw.line(
        (card_left + 18, scan_y, card_left + card_width - 18, scan_y),
        fill=(*band.highlight, scan_alpha),
        width=1 + tier.intensity // 3,
    )
    canvas.alpha_composite(scan.filter(ImageFilter.GaussianBlur(1.5)))
    canvas.alpha_composite(
        lock_layer(tier, progress=progress, card_box=card_box),
    )
    canvas.alpha_composite(spark_layer(tier, progress=progress, center=center))
    canvas.alpha_composite(flash_layer(tier, progress=progress))

    base.add_header(
        canvas,
        kicker=tier.kicker,
        title=tier.title,
        progress="PULL 1 / 1",
        accent=band.highlight,
    )
    canvas.alpha_composite(footer_layer(tier))
    return canvas.convert("RGB")


def global_palette(frames: list[Image.Image]) -> Image.Image:
    samples = [
        frame.resize((256, 144), Image.Resampling.BILINEAR)
        for frame in frames
    ]
    sheet = Image.new("RGB", (256 * len(samples), 144))
    for index, sample in enumerate(samples):
        sheet.paste(sample, (index * 256, 0))
    return sheet.quantize(
        colors=256,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )


def save_gif(
    frames: list[Image.Image],
    *,
    path: Path,
    frame_ms: int,
    final_hold_ms: int,
) -> None:
    palette = global_palette(frames)
    paletted = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]
    durations = [frame_ms] * len(paletted)
    durations[-1] = final_hold_ms
    path.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=paletted[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )


def save_keyframe(
    frame: Image.Image,
    *,
    tier: MotionTier,
    stage: str,
) -> Path:
    path = KEYFRAMES / f"{tier.key}_{stage}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(path, format="PNG", optimize=True, compress_level=9)
    return path


def make_storyboard(
    keyframes: dict[str, dict[str, Image.Image]],
) -> Image.Image:
    canvas = base.system_canvas()
    base.add_header(
        canvas,
        kicker="MOTION REVIEW",
        title="RARITY CRESCENDO — START / PEAK / END",
        accent=(225, 205, 156),
    )
    draw = ImageDraw.Draw(canvas)
    thumb_size = (229, 129)
    stages = ("start", "peak", "end")
    for tier_index, tier in enumerate(TIERS):
        band = base.BANDS[tier.band_key]
        x = 22 + tier_index * 246
        label = tier.review_label.split(" // ")[0]
        draw.text(
            (x, 101),
            label,
            font=base.font(base.SANS_BOLD, 13),
            fill=(*band.highlight, 255),
        )
        for stage_index, stage in enumerate(stages):
            y = 126 + stage_index * 137
            thumb = keyframes[tier.key][stage].resize(
                thumb_size,
                Image.Resampling.LANCZOS,
            )
            canvas.paste(thumb, (x, y))
            draw.rectangle(
                (x, y, x + thumb_size[0], y + thumb_size[1]),
                outline=(*band.mid, 255),
                width=2,
            )
            draw.rectangle((x + 6, y + 6, x + 55, y + 25), fill=(0, 0, 0, 206))
            draw.text(
                (x + 11, y + 8),
                stage.upper(),
                font=base.font(base.SANS_BOLD, 9),
                fill=(239, 235, 222, 255),
            )
    return canvas.convert("RGB")


def source_hashes() -> dict[str, str]:
    return {
        "reviewed_card_frame": sha256(base.FRAME_PATH),
    }


def public_record(tier: MotionTier, path: Path) -> dict[str, object]:
    return {
        "filename": path.name,
        "label": tier.review_label,
        "dimensions": list(base.BOARD_SIZE),
        "sha256": sha256(path),
        "byte_count": path.stat().st_size,
        "frame_count": tier.frame_count,
        "frame_ms": tier.frame_ms,
        "final_hold_ms": tier.final_hold_ms,
        "duration_ms": tier.frame_ms * (tier.frame_count - 1)
        + tier.final_hold_ms,
        "containment_locks": tier.lock_count,
        "visual_intensity": tier.intensity,
        "review_loop": True,
        "production_intent": "play_once",
        "accessible_text": (
            f"{tier.review_label}; sealed acquisition animation; "
            "exact rank withheld; static result follows"
        ),
    }


def write_start_here(records: list[dict[str, object]]) -> Path:
    sizes = "\n".join(
        f"{index:02d}  {record['label']}  "
        f"{record['duration_ms'] / 1000:.2f}s  "
        f"{record['byte_count'] / (1024 * 1024):.2f} MiB"
        for index, record in enumerate(records, start=1)
    )
    path = REVIEW / "00_START_HERE.txt"
    path.write_text(
        f"""ONE-STAR PULL REVEAL — ANIMATION PROOF REVISION

Open the four GIFs side by side or use index.html. They intentionally loop in
this review folder. A production animation would play once, hold its sealed
ending, and hand off to the existing static Hero result board.

{sizes}

READING THE LADDER
- Under 3 stars: one restrained scan and pulse; no anomaly locks or flare.
- 3-4 stars: two silver containment locks and a cold flash.
- 5-6 stars: three gold locks, tightening chains, sparks, and a larger flare.
- 7 stars: four white-gold locks, maximum cage response, and the longest peak.

The GIFs never reveal identity or exact rank. Animation carries no required
state: the following static result board remains authoritative and accessible.
There is deliberately no file-size acceptance cap in this proof revision.
""",
        encoding="utf-8",
    )
    return path


def write_index(records: list[dict[str, object]]) -> Path:
    figures = []
    for record in records:
        filename = html.escape(str(record["filename"]))
        label = html.escape(str(record["label"]))
        alt = html.escape(str(record["accessible_text"]))
        figures.append(
            f'<figure><img src="{filename}" alt="{alt}">'
            f"<figcaption>{label}<br>"
            f"{record['duration_ms'] / 1000:.2f}s · "
            f"{record['byte_count'] / (1024 * 1024):.2f} MiB</figcaption></figure>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>One-Star Pull Reveal — Animation Proofs</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; background: #030711; color: #eee9db; }}
body {{ margin: 0; padding: 2rem; }}
h1 {{ font-family: Georgia, serif; letter-spacing: .035em; }}
p {{ color: #a9bcc9; max-width: 72rem; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(30rem, 1fr)); gap: 1.25rem; }}
figure {{ margin: 0; border: 1px solid #765e36; background: #07101e; padding: .65rem; }}
img {{ display: block; width: 100%; height: auto; }}
figcaption {{ padding: .65rem .2rem .1rem; color: #d5bd87; line-height: 1.5; }}
</style>
</head>
<body>
<h1>One-Star Pull Reveal — Animation Proofs</h1>
<p>Review loops only. Every tier lands on a sealed frame; the next static board owns identity and exact rank.</p>
<main>{''.join(figures)}</main>
</body>
</html>
"""
    path = REVIEW / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def write_manifest(records: list[dict[str, object]], storyboard: Path) -> None:
    manifest = {
        "schema_version": 1,
        "status": "visual_review_only",
        "direction": "animated_rarity_crescendo",
        "production_binding": False,
        "transport_contract": "optional non-semantic motion before static result",
        "review_encoding": "loop",
        "production_intent": "play_once_then_hold",
        "file_size_cap_bytes": None,
        "rank_bands": {
            "under_3": "iron_plain",
            "3_to_4": "silver",
            "5_to_6": "gold",
            "7": "white_gold",
        },
        "invariants": {
            "first_frame": "sealed",
            "last_frame": "sealed",
            "identity_before_result": "withheld",
            "exact_rank_before_result": "withheld",
            "semantic_fallback": "existing static result board",
            "animation_required_for_comprehension": False,
            "fakeouts": "none",
            "sound": "not evaluated",
            "image_generation_calls": 0,
        },
        "tiers": records,
        "storyboard": {
            "filename": storyboard.name,
            "dimensions": list(base.BOARD_SIZE),
            "sha256": sha256(storyboard),
        },
        "source_hashes": source_hashes(),
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    MANIFEST.write_text(encoded, encoding="utf-8")
    (REVIEW / "manifest.json").write_text(encoded, encoding="utf-8")
    (PROVENANCE / "animation_source_hashes.json").write_text(
        json.dumps(source_hashes(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (PROVENANCE / "animation_decisions.json").write_text(
        json.dumps(
            {
                "approved_card_treatment": "Option F metal milestones",
                "animation_scope": "sealed pre-result transition only",
                "low_rarity_motion": "short plain scan and pulse",
                "rank_information_in_animation": False,
                "static_result_authoritative": True,
                "review_loops": True,
                "production_playback": "once then static handoff",
                "file_size_limit": None,
                "production_binding": False,
                "image_generation_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_hashes() -> None:
    review_hashes = REVIEW / "SHA256SUMS.txt"
    review_entries = [
        path for path in sorted(REVIEW.iterdir()) if path.name != review_hashes.name
    ]
    review_hashes.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in review_entries
            if path.is_file()
        ),
        encoding="utf-8",
    )
    provenance_hashes = PROVENANCE / "animation_SHA256SUMS.txt"
    owned = sorted(PROOFS.rglob("*.png"))
    owned.extend(
        (
            MANIFEST,
            PROVENANCE / "animation_source_hashes.json",
            PROVENANCE / "animation_decisions.json",
        )
    )
    provenance_hashes.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
            for path in owned
        ),
        encoding="utf-8",
    )


def copy_windows_review() -> None:
    WINDOWS_REVIEW.mkdir(parents=True, exist_ok=True)
    for path in sorted(REVIEW.iterdir()):
        if path.is_file():
            shutil.copyfile(path, WINDOWS_REVIEW / path.name)


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    KEYFRAMES.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    with Image.open(base.FRAME_PATH) as received:
        frame_overlay = received.convert("RGBA")

    records: list[dict[str, object]] = []
    story_frames: dict[str, dict[str, Image.Image]] = {}
    for tier in TIERS:
        sealed = base.sealed_card(frame_overlay, base.BANDS[tier.band_key])
        frames = [
            animation_frame(
                tier,
                sealed,
                progress=index / (tier.frame_count - 1),
            )
            for index in range(tier.frame_count)
        ]
        path = REVIEW / tier.filename
        save_gif(
            frames,
            path=path,
            frame_ms=tier.frame_ms,
            final_hold_ms=tier.final_hold_ms,
        )
        peak_index = round(tier.flash_time * (tier.frame_count - 1))
        selected = {
            "start": frames[0],
            "peak": frames[peak_index],
            "end": frames[-1],
        }
        for stage, image in selected.items():
            save_keyframe(image, tier=tier, stage=stage)
        story_frames[tier.key] = selected
        records.append(public_record(tier, path))

    storyboard = REVIEW / "05_start_peak_end_storyboard.png"
    make_storyboard(story_frames).save(
        storyboard,
        format="PNG",
        optimize=True,
        compress_level=9,
    )
    write_start_here(records)
    write_index(records)
    write_manifest(records, storyboard)
    write_hashes()
    copy_windows_review()

    for record in records:
        print(
            f"{record['filename']}  {record['frame_count']} frames  "
            f"{record['duration_ms']} ms  {record['byte_count']} bytes"
        )
    print(storyboard)
    print(REVIEW / "index.html")
    print(WINDOWS_REVIEW)


if __name__ == "__main__":
    main()
