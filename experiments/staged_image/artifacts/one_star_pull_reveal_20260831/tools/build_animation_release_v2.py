#!/usr/bin/env python3
"""Build V2 outward-release animation proofs for One-Star summons.

V1 is preserved as the inward-lock comparison.  V2 begins contained, ejects
locks away from the card, breaks any wrapped chains, and holds on a flash that
hands off to the existing static result board.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import build_animation_proofs as v1
import build_proofs as base


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review_animation_release_v2"
PROOFS = ROOT / "proofs/animation_release_v2"
KEYFRAMES = PROOFS / "keyframes"
PROVENANCE = ROOT / "provenance"
MANIFEST = ROOT / "animation_release_v2_manifest.json"
WINDOWS_REVIEW = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/"
    "OneStarPullRevealAnimationReleaseV2_20260831"
)


@dataclass(frozen=True)
class ReleaseTier:
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
    chain_count: int
    flash_start: float
    intensity: int


TIERS = (
    ReleaseTier(
        key="under_three",
        band_key="iron",
        review_label="UNDER 3 STARS // PLAIN RELEASE",
        filename="01_iron_plain_release_under_3.gif",
        kicker="SYSTEM ACQUISITION",
        title="SIGNATURE RELEASING",
        footer="STANDARD RELEASE",
        frame_count=16,
        frame_ms=60,
        final_hold_ms=380,
        lock_count=0,
        chain_count=0,
        flash_start=0.72,
        intensity=1,
    ),
    ReleaseTier(
        key="three_to_four",
        band_key="silver",
        review_label="3-4 STARS // SILVER LOCK RELEASE",
        filename="02_silver_lock_release_3_to_4.gif",
        kicker="SIGNATURE ANOMALY",
        title="CONTAINMENT BREAKS",
        footer="RELEASE CASCADE",
        frame_count=28,
        frame_ms=50,
        final_hold_ms=520,
        lock_count=2,
        chain_count=0,
        flash_start=0.76,
        intensity=2,
    ),
    ReleaseTier(
        key="five_to_six",
        band_key="gold",
        review_label="5-6 STARS // GOLD CHAINBREAK",
        filename="03_gold_chainbreak_5_to_6.gif",
        kicker="SIGNATURE ANOMALY",
        title="CONTAINMENT BREAKS",
        footer="CHAIN SEAL FRACTURED",
        frame_count=36,
        frame_ms=50,
        final_hold_ms=620,
        lock_count=3,
        chain_count=2,
        flash_start=0.79,
        intensity=3,
    ),
    ReleaseTier(
        key="seven",
        band_key="white_gold",
        review_label="7 STARS // WHITE-GOLD CHAINBREAK",
        filename="04_white_gold_chainbreak_7.gif",
        kicker="CONTAINMENT FAILURE",
        title="THE SEAL SHATTERS",
        footer="CRITICAL RELEASE",
        frame_count=44,
        frame_ms=50,
        final_hold_ms=820,
        lock_count=4,
        chain_count=3,
        flash_start=0.82,
        intensity=4,
    ),
)

CARD_CENTER = (512, 316)
CARD_NOMINAL_SIZE = (264, 422)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_progress(progress: float) -> float:
    return v1.smoothstep((progress - 0.10) / 0.60)


def break_progress(progress: float) -> float:
    return v1.smoothstep((progress - 0.27) / 0.43)


def flash_progress(tier: ReleaseTier, progress: float) -> float:
    return v1.smoothstep(
        (progress - tier.flash_start) / (1.0 - tier.flash_start)
    )


def below_header(canvas: Image.Image, layer: Image.Image) -> None:
    canvas.alpha_composite(layer.crop((0, 91, 1024, 576)), (0, 91))


def release_aura(
    tier: ReleaseTier,
    *,
    progress: float,
) -> Image.Image:
    band = base.BANDS[tier.band_key]
    released = release_progress(progress)
    flash = flash_progress(tier, progress)
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    center_x, center_y = CARD_CENTER
    pulse = 0.55 + 0.45 * math.sin(progress * math.tau * (1.1 + tier.intensity * 0.12))
    for index, base_radius in enumerate((142, 206, 278)):
        radius = base_radius + released * (65 + index * 42)
        alpha = round(
            (6 + tier.intensity * (6 + index * 3))
            * (0.72 + 0.28 * pulse)
            * (1.0 - flash * 0.25)
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
    return layer.filter(ImageFilter.GaussianBlur(45 - tier.intensity * 2))


def expanding_rings(
    tier: ReleaseTier,
    *,
    progress: float,
) -> Image.Image:
    band = base.BANDS[tier.band_key]
    released = release_progress(progress)
    flash = flash_progress(tier, progress)
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    center_x, center_y = CARD_CENTER
    ring_count = 1 + tier.intensity
    for index in range(ring_count):
        radius_x = 180 + index * 29 + released * (95 + index * 24)
        radius_y = 132 + index * 18 + released * (67 + index * 14)
        spin = progress * (110 + index * 41) * (-1 if index % 2 else 1)
        start = 205 + index * 39 + spin
        span = 74 + tier.intensity * 14
        alpha = round((44 + tier.intensity * 35) * (1.0 - flash * 0.55))
        bounds = (
            center_x - radius_x,
            center_y - radius_y,
            center_x + radius_x,
            center_y + radius_y,
        )
        draw.arc(
            bounds,
            start,
            start + span,
            fill=(*band.highlight, min(238, alpha)),
            width=2 + tier.intensity // 2,
        )
        if tier.intensity > 1:
            draw.arc(
                bounds,
                start + 180,
                start + 180 + span * 0.68,
                fill=(*band.mid, min(210, alpha + 20)),
                width=3 + tier.intensity // 2,
            )
    return layer


def draw_chain_link(
    layer: Image.Image,
    *,
    center: tuple[float, float],
    angle: float,
    band: base.BandStyle,
    alpha: int,
    scale: float = 1.0,
) -> None:
    width = max(18, round(34 * scale))
    height = max(11, round(20 * scale))
    pad = 8
    link = Image.new("RGBA", (width + pad * 2, height + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(link)
    draw.ellipse(
        (pad, pad, pad + width, pad + height),
        outline=(*band.shadow, min(255, alpha)),
        width=max(3, round(6 * scale)),
    )
    draw.ellipse(
        (pad + 2, pad + 2, pad + width - 2, pad + height - 2),
        outline=(*band.highlight, min(255, alpha)),
        width=max(1, round(2 * scale)),
    )
    rotated = link.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    layer.alpha_composite(
        rotated,
        (
            round(center[0] - rotated.width / 2),
            round(center[1] - rotated.height / 2),
        ),
    )


def wrapped_chains(
    tier: ReleaseTier,
    *,
    progress: float,
    card_box: tuple[int, int, int, int],
) -> tuple[Image.Image, Image.Image, Image.Image]:
    back = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    front = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    fracture = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    if tier.chain_count == 0:
        return back, front, fracture

    band = base.BANDS[tier.band_key]
    broken = break_progress(progress)
    flash = flash_progress(tier, progress)
    left, top, right, bottom = card_box
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    chain_span = right - left + 156
    spacing = 27 if tier.intensity == 3 else 25
    chain_y = (
        (center_y - 86, center_y + 82)
        if tier.chain_count == 2
        else (center_y - 112, center_y, center_y + 112)
    )
    slopes = (-0.23, 0.22) if tier.chain_count == 2 else (-0.25, 0.18, -0.20)
    break_draw = ImageDraw.Draw(fracture)

    for chain_index, (base_y, slope) in enumerate(
        zip(chain_y, slopes, strict=True)
    ):
        link_count = math.floor(chain_span / spacing) + 1
        break_x = center_x + (chain_index - (tier.chain_count - 1) / 2) * 18
        for link_index in range(link_count):
            raw_x = center_x - chain_span / 2 + link_index * spacing
            raw_y = (
                base_y
                + slope * (raw_x - center_x)
                + math.sin(link_index * 0.92 + chain_index) * 3
            )
            side = -1 if raw_x < break_x else 1
            distance_from_break = abs(raw_x - break_x)
            displacement = side * broken**1.25 * (128 + tier.intensity * 19)
            vertical_release = (
                (chain_index - (tier.chain_count - 1) / 2)
                * broken
                * 34
                + side * broken * 10
            )
            x = raw_x + displacement
            y = raw_y + vertical_release
            near_break = distance_from_break < spacing * 0.78
            opacity = round((220 - flash * 105) * (0.25 if near_break and broken > 0.18 else 1.0))
            angle = math.degrees(math.atan(slope))
            angle += 90 if link_index % 2 else 0
            angle += side * broken * (28 + (link_index % 3) * 7)
            target = front if left - 8 <= raw_x <= right + 8 else back
            draw_chain_link(
                target,
                center=(x, y),
                angle=angle,
                band=band,
                alpha=max(0, opacity),
                scale=1.04 if target is front else 0.92,
            )

        fracture_peak = v1.gaussian(progress, 0.49 + chain_index * 0.035, 0.115)
        if fracture_peak > 0.025:
            burst_y = base_y + slope * (break_x - center_x)
            for ray in range(8 + tier.intensity * 2):
                angle = math.radians(ray * (360 / (8 + tier.intensity * 2)) + chain_index * 17)
                length = 12 + (ray % 4) * 8 + fracture_peak * 34
                alpha = round(fracture_peak * (115 + (ray % 3) * 42))
                break_draw.line(
                    (
                        break_x,
                        burst_y,
                        break_x + math.cos(angle) * length,
                        burst_y + math.sin(angle) * length,
                    ),
                    fill=(*band.highlight, min(255, alpha)),
                    width=1 + (ray % 4 == 0),
                )
            radius = 8 + round(fracture_peak * 25)
            break_draw.ellipse(
                (
                    break_x - radius,
                    burst_y - radius,
                    break_x + radius,
                    burst_y + radius,
                ),
                outline=(*band.highlight, round(fracture_peak * 235)),
                width=2 + tier.intensity // 2,
            )
    return back, front, fracture.filter(ImageFilter.GaussianBlur(0.45))


def ejected_locks(
    tier: ReleaseTier,
    *,
    progress: float,
    card_box: tuple[int, int, int, int],
) -> Image.Image:
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    if tier.lock_count == 0:
        return layer
    band = base.BANDS[tier.band_key]
    released = release_progress(progress)
    flash = flash_progress(tier, progress)
    draw = ImageDraw.Draw(layer)
    left, top, right, bottom = card_box
    center_x = (left + right) // 2
    y_positions = [
        round(top + (bottom - top) * (index + 1) / (tier.lock_count + 1))
        for index in range(tier.lock_count)
    ]
    for index, y in enumerate(y_positions):
        side = -1 if index % 2 == 0 else 1
        start_x = left - 18 if side < 0 else right + 18
        travel = 226 + index * 13 + tier.intensity * 10
        x = round(start_x + side * travel * released)
        y_offset = round((index - (tier.lock_count - 1) / 2) * released * 25)
        y += y_offset
        opacity = round(255 * (1.0 - flash * 0.72))

        streak_length = round(22 + released * (78 + tier.intensity * 10))
        draw.line(
            (x - side * streak_length, y + 17, x - side * 22, y + 17),
            fill=(*band.highlight, round(opacity * 0.34)),
            width=2 + tier.intensity // 2,
        )
        if released < 0.72:
            tether_alpha = round((1.0 - released / 0.72) * 150)
            draw.line(
                (x - side * 12, y + 17, center_x + side * 14, y),
                fill=(*band.mid, tether_alpha),
                width=2,
            )

        shackle_lift = round(released * 14)
        shackle_turn = released * side * 16
        draw.arc(
            (
                x - 21 + shackle_turn,
                y - 29 - shackle_lift,
                x + 21 + shackle_turn,
                y + 18 - shackle_lift,
            ),
            185,
            355,
            fill=(*band.highlight, opacity),
            width=5,
        )
        draw.rounded_rectangle(
            (x - 19, y, x + 19, y + 35),
            radius=6,
            fill=(3, 6, 11, opacity),
            outline=(*band.mid, opacity),
            width=4,
        )
        draw.ellipse(
            (x - 3, y + 12, x + 3, y + 21),
            fill=(*band.highlight, opacity),
        )
    return layer


def release_scan(
    tier: ReleaseTier,
    *,
    progress: float,
    card_box: tuple[int, int, int, int],
) -> Image.Image:
    band = base.BANDS[tier.band_key]
    left, top, right, bottom = card_box
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scan_y = round(top + 18 + progress * (bottom - top - 36))
    alpha = 48 + tier.intensity * 18
    draw.rectangle(
        (left + 15, scan_y - 4, right - 15, scan_y + 4),
        fill=(*band.highlight, alpha // 4),
    )
    draw.line(
        (left + 15, scan_y, right - 15, scan_y),
        fill=(*band.highlight, alpha),
        width=1 + tier.intensity // 3,
    )
    return layer.filter(ImageFilter.GaussianBlur(1.3))


def release_sparks(
    tier: ReleaseTier,
    *,
    progress: float,
) -> Image.Image:
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    if tier.intensity < 2:
        return layer
    band = base.BANDS[tier.band_key]
    peak = v1.gaussian(progress, 0.62, 0.18)
    if peak < 0.02:
        return layer
    draw = ImageDraw.Draw(layer)
    count = 8 + tier.intensity * 5
    for index in range(count):
        angle = math.radians((index * 137.508 + tier.intensity * 23) % 360)
        distance = 118 + ((index * 31) % 127) + release_progress(progress) * 80
        x = CARD_CENTER[0] + math.cos(angle) * distance
        y = CARD_CENTER[1] + math.sin(angle) * distance * 0.69
        length = 7 + (index % 5) * 4
        alpha = round(peak * (105 + (index % 4) * 35))
        draw.line(
            (
                x,
                y,
                x + math.cos(angle) * length,
                y + math.sin(angle) * length,
            ),
            fill=(*band.highlight, min(245, alpha)),
            width=1 + (index % 4 == 0),
        )
    return layer.filter(ImageFilter.GaussianBlur(0.35))


def footer_layer(tier: ReleaseTier) -> Image.Image:
    band = base.BANDS[tier.band_key]
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, 525, 1024, 576), fill=(1, 4, 10, 228))
    draw.line((48, 525, 976, 525), fill=(*band.highlight, 82), width=1)
    draw.text(
        (52, 541),
        tier.footer,
        font=base.font(base.SANS_BOLD, 13),
        fill=(*band.highlight, 255),
    )
    exact = "RESULT FOLLOWS"
    face = base.font(base.SANS_BOLD, 13)
    exact_width, _ = base.text_size(draw, exact, face)
    draw.text(
        (972 - exact_width, 541),
        exact,
        font=face,
        fill=(141, 159, 173, 255),
    )
    return layer


def pre_reveal_flash(
    tier: ReleaseTier,
    *,
    progress: float,
) -> Image.Image:
    amount = flash_progress(tier, progress)
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    if amount <= 0:
        return layer
    band = base.BANDS[tier.band_key]
    flash_color = tuple(round(channel * 0.28 + 255 * 0.72) for channel in band.highlight)
    full_alpha = (56, 104, 158, 218)[tier.intensity - 1]
    wash = Image.new(
        "RGBA",
        base.BOARD_SIZE,
        (*flash_color, round(amount * full_alpha)),
    )
    layer = Image.alpha_composite(layer, wash)

    core = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(core)
    radius = round(80 + amount * (170 + tier.intensity * 42))
    core_alpha = round(amount * (125 + tier.intensity * 30))
    draw.ellipse(
        (
            CARD_CENTER[0] - radius * 1.42,
            CARD_CENTER[1] - radius,
            CARD_CENTER[0] + radius * 1.42,
            CARD_CENTER[1] + radius,
        ),
        fill=(*flash_color, min(255, core_alpha)),
    )
    ray_count = 8 + tier.intensity * 4
    for index in range(ray_count):
        angle = math.radians(index * (360 / ray_count) + 7)
        inner = 50 + tier.intensity * 9
        outer = inner + amount * (270 + (index % 4) * 32)
        draw.line(
            (
                CARD_CENTER[0] + math.cos(angle) * inner,
                CARD_CENTER[1] + math.sin(angle) * inner,
                CARD_CENTER[0] + math.cos(angle) * outer,
                CARD_CENTER[1] + math.sin(angle) * outer,
            ),
            fill=(*flash_color, round(amount * (65 + (index % 3) * 30))),
            width=1 + tier.intensity // 2,
        )
    return Image.alpha_composite(
        layer,
        core.filter(ImageFilter.GaussianBlur(34 - tier.intensity * 3)),
    )


def animation_frame(
    tier: ReleaseTier,
    sealed: Image.Image,
    *,
    progress: float,
    chain_renderer: Callable[..., tuple[Image.Image, Image.Image, Image.Image]]
    | None = None,
    lock_renderer: Callable[..., Image.Image] | None = None,
    card_offset: tuple[int, int] = (0, 0),
) -> Image.Image:
    band = base.BANDS[tier.band_key]
    canvas = base.system_canvas()
    canvas.alpha_composite(release_aura(tier, progress=progress))
    below_header(canvas, expanding_rings(tier, progress=progress))

    scale = 0.984 + release_progress(progress) * 0.019
    scale += math.sin(progress * math.tau * 1.2) * 0.0025
    card_width = round(CARD_NOMINAL_SIZE[0] * scale)
    card_height = round(CARD_NOMINAL_SIZE[1] * scale)
    card_left = CARD_CENTER[0] - card_width // 2 + card_offset[0]
    card_top = CARD_CENTER[1] - card_height // 2 + card_offset[1]
    card_box = (card_left, card_top, card_left + card_width, card_top + card_height)

    render_chains = wrapped_chains if chain_renderer is None else chain_renderer
    back_chains, front_chains, fracture = render_chains(
        tier,
        progress=progress,
        card_box=card_box,
    )
    canvas.alpha_composite(back_chains)
    card = sealed.resize((card_width, card_height), Image.Resampling.LANCZOS)
    base.place_with_shadow(canvas, card, (card_left, card_top), blur=18)
    canvas.alpha_composite(front_chains)
    canvas.alpha_composite(release_scan(tier, progress=progress, card_box=card_box))
    canvas.alpha_composite(fracture)
    render_locks = ejected_locks if lock_renderer is None else lock_renderer
    canvas.alpha_composite(
        render_locks(tier, progress=progress, card_box=card_box)
    )
    canvas.alpha_composite(release_sparks(tier, progress=progress))

    base.add_header(
        canvas,
        kicker=tier.kicker,
        title=tier.title,
        progress="PULL 1 / 1",
        accent=band.highlight,
    )
    canvas.alpha_composite(footer_layer(tier))
    canvas.alpha_composite(pre_reveal_flash(tier, progress=progress))
    return canvas.convert("RGB")


def save_keyframe(
    image: Image.Image,
    *,
    tier: ReleaseTier,
    stage: str,
) -> Path:
    path = KEYFRAMES / f"{tier.key}_{stage}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, compress_level=9)
    return path


def make_storyboard(
    keyframes: dict[str, dict[str, Image.Image]],
) -> Image.Image:
    canvas = base.system_canvas()
    base.add_header(
        canvas,
        kicker="MOTION REVIEW V2",
        title="BOUND / OUTWARD RELEASE / PRE-REVEAL FLASH",
        accent=(225, 205, 156),
    )
    draw = ImageDraw.Draw(canvas)
    thumb_size = (229, 129)
    stages = ("bound", "release", "flash")
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
            tag_width = 68 if stage == "release" else 54
            draw.rectangle(
                (x + 6, y + 6, x + 6 + tag_width, y + 25),
                fill=(0, 0, 0, 210),
            )
            draw.text(
                (x + 11, y + 8),
                stage.upper(),
                font=base.font(base.SANS_BOLD, 9),
                fill=(239, 235, 222, 255),
            )
    return canvas.convert("RGB")


def release_geometry(tier: ReleaseTier) -> dict[str, object]:
    start_distance = CARD_NOMINAL_SIZE[0] // 2 + 18
    end_distances = [
        start_distance + 226 + index * 13 + tier.intensity * 10
        for index in range(tier.lock_count)
    ]
    return {
        "lock_motion": "outward",
        "lock_count": tier.lock_count,
        "lock_start_distance_px": start_distance if tier.lock_count else None,
        "lock_end_distances_px": end_distances,
        "chain_count": tier.chain_count,
        "chain_start_layer": "wrapped_across_card_face"
        if tier.chain_count
        else "none",
        "chain_motion": "split_and_eject_outward"
        if tier.chain_count
        else "none",
        "chain_break_displacement_px": 128 + tier.intensity * 19
        if tier.chain_count
        else 0,
    }


def public_record(tier: ReleaseTier, path: Path) -> dict[str, object]:
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
        "visual_intensity": tier.intensity,
        "release_geometry": release_geometry(tier),
        "final_frame": "held_pre_reveal_flash",
        "review_loop": True,
        "production_intent": "play_once_then_static_result",
        "accessible_text": (
            f"{tier.review_label}; contained signature releases outward; "
            "pre-reveal flash; static result follows"
        ),
    }


def write_start_here(records: list[dict[str, object]]) -> None:
    lines = "\n".join(
        f"{index:02d}  {record['label']}  "
        f"{record['duration_ms'] / 1000:.2f}s  "
        f"{record['byte_count'] / (1024 * 1024):.2f} MiB"
        for index, record in enumerate(records, start=1)
    )
    (REVIEW / "00_START_HERE.txt").write_text(
        f"""ONE-STAR PULL REVEAL — OUTWARD RELEASE V2

This revision reverses the prior containment motion. Locks begin attached to
the card and eject outward. Gold and white-gold chains begin visibly wrapped
across the card face, fracture, and travel outward with the locks. Every tier
holds on a pre-reveal flash before the existing static result board.

{lines}

READING THE LADDER
- Under 3 stars: plain scan, outward aura, modest flash; no locks or chains.
- 3-4 stars: two locks pop open and release away from the card.
- 5-6 stars: two wrapped gold chains break as three locks eject.
- 7 stars: three wrapped white-gold chains shatter as four locks eject.

The GIFs loop only for review. Production intent is one play, a held flash,
then the static identity/rank result. Motion carries no required game state.
V1 remains preserved as the rejected inward-lock comparison.
""",
        encoding="utf-8",
    )


def write_index(records: list[dict[str, object]]) -> None:
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
<title>One-Star Pull Reveal — Outward Release V2</title>
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
<h1>One-Star Pull Reveal — Outward Release V2</h1>
<p>Contained start, outward lock release, wrapped-chain fracture where applicable, then a held pre-reveal flash.</p>
<main>{''.join(figures)}</main>
</body>
</html>
"""
    (REVIEW / "index.html").write_text(document, encoding="utf-8")


def write_manifest(records: list[dict[str, object]], storyboard: Path) -> None:
    manifest = {
        "schema_version": 2,
        "status": "visual_review_only",
        "direction": "outward_release",
        "supersedes_for_review": "inward_lock_v1",
        "production_binding": False,
        "transport_contract": "optional motion before authoritative static result",
        "review_encoding": "loop",
        "production_intent": "play_once_hold_flash_then_static_result",
        "file_size_cap_bytes": None,
        "rank_bands": {
            "under_3": "iron_plain",
            "3_to_4": "silver",
            "5_to_6": "gold",
            "7": "white_gold",
        },
        "sequence": [
            "contained_start",
            "locks_eject_outward",
            "wrapped_chains_break_where_present",
            "held_pre_reveal_flash",
            "authoritative_static_result",
        ],
        "invariants": {
            "identity_before_static_result": "withheld",
            "exact_rank_before_static_result": "withheld",
            "semantic_fallback": "existing static result board",
            "animation_required_for_comprehension": False,
            "fakeouts": "none",
            "sound": "not evaluated",
            "image_generation_calls": 0,
            "v1_preserved": True,
        },
        "tiers": records,
        "storyboard": {
            "filename": storyboard.name,
            "dimensions": list(base.BOARD_SIZE),
            "sha256": sha256(storyboard),
        },
        "source_hashes": {
            "reviewed_card_frame": sha256(base.FRAME_PATH),
        },
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    MANIFEST.write_text(encoded, encoding="utf-8")
    (REVIEW / "manifest.json").write_text(encoded, encoding="utf-8")
    (PROVENANCE / "animation_release_v2_source_hashes.json").write_text(
        json.dumps(manifest["source_hashes"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (PROVENANCE / "animation_release_v2_decisions.json").write_text(
        json.dumps(
            {
                "revision": "V2 outward release",
                "v1_disposition": "preserved rejected comparison",
                "lock_motion": "outward",
                "chain_placement": "wrapped across card",
                "chain_motion": "fracture during outward release",
                "final_frame": "held pre-reveal flash",
                "static_result_authoritative": True,
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
    entries = [path for path in sorted(REVIEW.iterdir()) if path != review_hashes]
    review_hashes.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in entries
            if path.is_file()
        ),
        encoding="utf-8",
    )
    provenance_hashes = PROVENANCE / "animation_release_v2_SHA256SUMS.txt"
    owned = sorted(KEYFRAMES.glob("*.png"))
    owned.extend(
        (
            MANIFEST,
            PROVENANCE / "animation_release_v2_source_hashes.json",
            PROVENANCE / "animation_release_v2_decisions.json",
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
        gif_path = REVIEW / tier.filename
        v1.save_gif(
            frames,
            path=gif_path,
            frame_ms=tier.frame_ms,
            final_hold_ms=tier.final_hold_ms,
        )
        release_index = round(0.58 * (tier.frame_count - 1))
        selected = {
            "bound": frames[0],
            "release": frames[release_index],
            "flash": frames[-1],
        }
        for stage, image in selected.items():
            save_keyframe(image, tier=tier, stage=stage)
        story_frames[tier.key] = selected
        records.append(public_record(tier, gif_path))

    storyboard = REVIEW / "05_bound_release_flash_storyboard.png"
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
