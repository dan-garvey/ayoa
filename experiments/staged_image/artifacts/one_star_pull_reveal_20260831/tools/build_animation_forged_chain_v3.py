#!/usr/bin/env python3
"""Build V3 One-Star summon proofs with weightier forged chains."""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import build_animation_proofs as v1
import build_animation_release_v2 as v2
import build_proofs as base


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review_animation_forged_chain_v3"
PROOFS = ROOT / "proofs/animation_forged_chain_v3"
KEYFRAMES = PROOFS / "keyframes"
PROVENANCE = ROOT / "provenance"
MANIFEST = ROOT / "animation_forged_chain_v3_manifest.json"
WINDOWS_REVIEW = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/"
    "OneStarPullRevealForgedChainV3_20260831"
)

FILENAMES = (
    "01_iron_plain_release_under_3.gif",
    "02_silver_lock_release_3_to_4.gif",
    "03_gold_forged_chainbreak_5_to_6.gif",
    "04_white_gold_forged_chainbreak_7.gif",
)

LABELS = (
    "UNDER 3 STARS // PLAIN RELEASE",
    "3-4 STARS // SILVER LOCK RELEASE",
    "5-6 STARS // FORGED GOLD CHAINBREAK",
    "7 STARS // FORGED WHITE-GOLD CHAINBREAK",
)

TIERS = tuple(
    replace(tier, filename=filename, review_label=label)
    for tier, filename, label in zip(v2.TIERS, FILENAMES, LABELS, strict=True)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mix(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    return tuple(
        round(start * (1.0 - amount) + end * amount)
        for start, end in zip(first, second, strict=True)
    )


def draw_forged_link(
    layer: Image.Image,
    *,
    center: tuple[float, float],
    tangent_angle: float,
    band: base.BandStyle,
    alpha: int,
    edge_on: bool,
    scale: float = 1.0,
    shadow_only: bool = False,
) -> None:
    if edge_on:
        width = max(12, round(15 * scale))
        height = max(25, round(34 * scale))
        rotation = tangent_angle + 90
    else:
        width = max(30, round(45 * scale))
        height = max(16, round(25 * scale))
        rotation = tangent_angle
    pad = 12
    sprite = Image.new(
        "RGBA",
        (width + pad * 2, height + pad * 2),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(sprite)
    bounds = (pad, pad, pad + width, pad + height)
    radius = max(5, round(min(width, height) * 0.42))
    if shadow_only:
        draw.rounded_rectangle(
            bounds,
            radius=radius,
            outline=(0, 0, 0, min(220, alpha)),
            width=max(5, round(8 * scale)),
        )
    else:
        dark_metal = mix(band.shadow, band.mid, 0.34)
        body_metal = mix(band.shadow, band.mid, 0.72)
        edge_metal = mix(band.mid, band.highlight, 0.58)
        draw.rounded_rectangle(
            bounds,
            radius=radius,
            outline=(*dark_metal, min(255, alpha)),
            width=max(6, round(9 * scale)),
        )
        inner = max(2, round(3 * scale))
        draw.rounded_rectangle(
            (
                pad + inner,
                pad + inner,
                pad + width - inner,
                pad + height - inner,
            ),
            radius=max(3, radius - inner),
            outline=(*body_metal, min(255, alpha)),
            width=max(3, round(5 * scale)),
        )
        highlight_y = pad + max(2, round(3 * scale))
        draw.line(
            (
                pad + radius,
                highlight_y,
                pad + width - radius,
                highlight_y,
            ),
            fill=(*edge_metal, min(235, alpha)),
            width=max(1, round(2 * scale)),
        )
        draw.line(
            (
                pad + radius,
                pad + height - 2,
                pad + width - radius,
                pad + height - 2,
            ),
            fill=(*band.shadow, min(245, alpha)),
            width=max(1, round(2 * scale)),
        )
    rotated = sprite.rotate(
        rotation,
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


def draw_broken_link_half(
    layer: Image.Image,
    *,
    center: tuple[float, float],
    angle: float,
    side: int,
    band: base.BandStyle,
    alpha: int,
) -> None:
    sprite = Image.new("RGBA", (64, 42), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    bounds = (8, 8, 56, 34)
    start, end = ((90, 270) if side < 0 else (270, 450))
    dark_metal = mix(band.shadow, band.mid, 0.38)
    body_metal = mix(band.shadow, band.mid, 0.76)
    edge_metal = mix(band.mid, band.highlight, 0.62)
    draw.arc(
        bounds,
        start,
        end,
        fill=(*dark_metal, alpha),
        width=10,
    )
    draw.arc(
        bounds,
        start,
        end,
        fill=(*body_metal, alpha),
        width=6,
    )
    draw.arc(
        (10, 10, 54, 32),
        start,
        end,
        fill=(*edge_metal, min(235, alpha)),
        width=2,
    )
    opening_x = 52 if side < 0 else 12
    for endpoint_y in (8, 34):
        draw.line(
            (
                32,
                endpoint_y,
                opening_x,
                endpoint_y + side * 2,
            ),
            fill=(*dark_metal, alpha),
            width=9,
        )
        draw.line(
            (
                32,
                endpoint_y - 1,
                opening_x,
                endpoint_y + side * 2 - 1,
            ),
            fill=(*body_metal, alpha),
            width=5,
        )
        draw.line(
            (
                34,
                endpoint_y - 2,
                opening_x,
                endpoint_y + side * 2 - 2,
            ),
            fill=(*edge_metal, min(235, alpha)),
            width=2,
        )
    rotated = sprite.rotate(
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


def with_contact_shadow(front: Image.Image) -> Image.Image:
    alpha = front.getchannel("A")
    shadow_alpha = alpha.point(lambda value: round(value * 0.72))
    shadow = Image.new("RGBA", front.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha.filter(ImageFilter.GaussianBlur(3.2)))
    combined = Image.new("RGBA", front.size, (0, 0, 0, 0))
    combined.alpha_composite(shadow, (4, 5))
    combined.alpha_composite(front)
    return combined


def forged_chains(
    tier: v2.ReleaseTier,
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
    broken = v2.break_progress(progress)
    flash = v2.flash_progress(tier, progress)
    left, top, right, bottom = card_box
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    chain_span = right - left + 104
    spacing = 30 if tier.intensity == 3 else 28
    chain_y = (
        (center_y - 78, center_y + 76)
        if tier.chain_count == 2
        else (center_y - 102, center_y, center_y + 102)
    )
    slopes = (-0.14, 0.13) if tier.chain_count == 2 else (-0.16, 0.11, -0.14)
    fracture_draw = ImageDraw.Draw(fracture)

    for chain_index, (base_y, slope) in enumerate(
        zip(chain_y, slopes, strict=True)
    ):
        link_count = math.floor(chain_span / spacing) + 1
        break_x = center_x + (chain_index - (tier.chain_count - 1) / 2) * 16
        normalized_break_x = (break_x - center_x) / (chain_span / 2)
        break_y = base_y + slope * (break_x - center_x)
        break_y += (1.0 - normalized_break_x**2) * 4
        for link_index in range(link_count):
            raw_x = center_x - chain_span / 2 + link_index * spacing
            normalized_x = (raw_x - center_x) / (chain_span / 2)
            raw_y = (
                base_y
                + slope * (raw_x - center_x)
                + (1.0 - normalized_x**2) * 4
            )
            side = -1 if raw_x < break_x else 1
            relative_x = raw_x - break_x
            relative_y = raw_y - break_y
            segment_rotation = math.radians(side * broken * (7 + chain_index * 1.5))
            rotated_x = (
                math.cos(segment_rotation) * relative_x
                - math.sin(segment_rotation) * relative_y
            )
            rotated_y = (
                math.sin(segment_rotation) * relative_x
                + math.cos(segment_rotation) * relative_y
            )
            displacement = side * broken**1.18 * (122 + tier.intensity * 18)
            vertical_release = (
                chain_index - (tier.chain_count - 1) / 2
            ) * broken * 27
            x = break_x + rotated_x + displacement
            y = break_y + rotated_y + vertical_release
            distance_from_break = abs(raw_x - break_x)
            removed_for_break = distance_from_break < spacing * 0.82 and broken > 0.10
            if removed_for_break:
                continue
            opacity = round(235 * (1.0 - flash * 0.55))
            tangent = math.degrees(math.atan(slope)) + math.degrees(segment_rotation)
            target = front if left - 9 <= raw_x <= right + 9 else back
            edge_on = link_index % 2 == 1
            draw_forged_link(
                target,
                center=(x, y),
                tangent_angle=tangent,
                band=band,
                alpha=max(0, opacity),
                edge_on=edge_on,
                scale=1.0 if target is front else 0.90,
            )

        fracture_peak = v1.gaussian(
            progress,
            0.48 + chain_index * 0.035,
            0.105,
        )
        if broken > 0.06:
            half_alpha = round((0.52 + fracture_peak * 0.48) * 245 * (1.0 - flash * 0.65))
            for side in (-1, 1):
                relative_x = side * spacing * 0.72
                relative_y = slope * relative_x
                segment_rotation = math.radians(
                    side * broken * (7 + chain_index * 1.5)
                )
                rotated_x = (
                    math.cos(segment_rotation) * relative_x
                    - math.sin(segment_rotation) * relative_y
                )
                rotated_y = (
                    math.sin(segment_rotation) * relative_x
                    + math.cos(segment_rotation) * relative_y
                )
                displacement = (
                    side * broken**1.18 * (122 + tier.intensity * 18)
                )
                vertical_release = (
                    chain_index - (tier.chain_count - 1) / 2
                ) * broken * 27
                draw_broken_link_half(
                    fracture,
                    center=(
                        break_x + rotated_x + displacement,
                        break_y + rotated_y + vertical_release,
                    ),
                    angle=math.degrees(math.atan(slope))
                    + math.degrees(segment_rotation),
                    side=side,
                    band=band,
                    alpha=max(0, half_alpha),
                )
        if fracture_peak > 0.035:
            for ray in range(7 + tier.intensity):
                angle = math.radians(
                    ray * (360 / (7 + tier.intensity)) + chain_index * 19
                )
                length = 11 + (ray % 3) * 8 + fracture_peak * 26
                alpha = round(fracture_peak * (105 + (ray % 3) * 45))
                fracture_draw.line(
                    (
                        break_x,
                        break_y,
                        break_x + math.cos(angle) * length,
                        break_y + math.sin(angle) * length,
                    ),
                    fill=(*band.highlight, min(245, alpha)),
                    width=1 + (ray % 4 == 0),
                )
    return back, with_contact_shadow(front), fracture.filter(
        ImageFilter.GaussianBlur(0.3)
    )


def save_keyframe(
    image: Image.Image,
    *,
    tier: v2.ReleaseTier,
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
        kicker="MOTION REVIEW V3",
        title="FORGED CHAIN — BOUND / BREAK / FLASH",
        accent=(225, 205, 156),
    )
    draw = ImageDraw.Draw(canvas)
    thumb_size = (229, 129)
    stages = ("bound", "break", "flash")
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
            tag_width = 48
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


def public_record(tier: v2.ReleaseTier, path: Path) -> dict[str, object]:
    geometry = v2.release_geometry(tier)
    geometry["chain_visual_style"] = (
        "heavy_forged_interlocking_links" if tier.chain_count else "none"
    )
    geometry["chain_segment_motion"] = (
        "coherent_rigid_halves" if tier.chain_count else "none"
    )
    geometry["fracture_shape"] = (
        "two_open_broken_link_halves" if tier.chain_count else "none"
    )
    geometry["contact_shadow"] = bool(tier.chain_count)
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
        "release_geometry": geometry,
        "final_frame": "held_pre_reveal_flash",
        "review_loop": True,
        "production_intent": "play_once_then_static_result",
        "accessible_text": (
            f"{tier.review_label}; forged chains break outward where present; "
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
        f"""ONE-STAR PULL REVEAL — FORGED CHAIN V3

V3 keeps V2's outward lock ejection and held pre-reveal flash, but replaces
the necklace-like chain treatment. Gold and white-gold now use fewer, heavier
forged links: alternating face-on and edge-on pieces, dark beveled metal,
contact shadows, taut wrapping, coherent broken halves, and open snapped links.

{lines}

Review the bound and break moments closely in 03 and 04. The links should read
as weight on the card, then as two chain segments pulled apart—not a string of
independent glowing ovals. V1 and V2 remain preserved for comparison.

The GIFs loop only for review. Production intent remains one play, a held
flash, then the authoritative static identity/rank result.
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
<title>One-Star Pull Reveal — Forged Chain V3</title>
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
<h1>One-Star Pull Reveal — Forged Chain V3</h1>
<p>V2 outward release with heavier, interlocked, shadowed chains and explicit broken-link halves.</p>
<main>{''.join(figures)}</main>
</body>
</html>
"""
    (REVIEW / "index.html").write_text(document, encoding="utf-8")


def write_manifest(records: list[dict[str, object]], storyboard: Path) -> None:
    manifest = {
        "schema_version": 3,
        "status": "visual_review_only",
        "direction": "outward_release_forged_chain",
        "supersedes_for_review": "outward_release_v2_chain_treatment",
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
        "chain_revision": {
            "old_read": "repeated bright oval links",
            "new_read": "heavy forged interlocking restraint",
            "link_orientation": "alternating face-on and edge-on",
            "surface": "dark bevel with restrained rank-metal edge",
            "placement": "taut across card face with contact shadow",
            "release": "coherent rigid halves",
            "fracture": "explicit open broken-link halves",
        },
        "invariants": {
            "outward_lock_motion": True,
            "held_pre_reveal_flash": True,
            "identity_before_static_result": "withheld",
            "exact_rank_before_static_result": "withheld",
            "semantic_fallback": "existing static result board",
            "animation_required_for_comprehension": False,
            "fakeouts": "none",
            "image_generation_calls": 0,
            "v1_preserved": True,
            "v2_preserved": True,
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
    (PROVENANCE / "animation_forged_chain_v3_source_hashes.json").write_text(
        json.dumps(manifest["source_hashes"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (PROVENANCE / "animation_forged_chain_v3_decisions.json").write_text(
        json.dumps(
            {
                "revision": "V3 forged chain treatment",
                "v1_disposition": "preserved",
                "v2_disposition": "preserved comparison",
                "chain_mass": "heavy dark forged metal",
                "chain_interlock": "alternating face-on and edge-on links",
                "chain_placement": "taut across card with contact shadow",
                "chain_motion": "coherent broken halves",
                "fracture_shape": "open snapped link arcs",
                "outward_locks_unchanged": True,
                "pre_reveal_flash_unchanged": True,
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
    provenance_hashes = PROVENANCE / "animation_forged_chain_v3_SHA256SUMS.txt"
    owned = sorted(KEYFRAMES.glob("*.png"))
    owned.extend(
        (
            MANIFEST,
            PROVENANCE / "animation_forged_chain_v3_source_hashes.json",
            PROVENANCE / "animation_forged_chain_v3_decisions.json",
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
            v2.animation_frame(
                tier,
                sealed,
                progress=index / (tier.frame_count - 1),
                chain_renderer=forged_chains,
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
        break_index = round(0.55 * (tier.frame_count - 1))
        selected = {
            "bound": frames[0],
            "break": frames[break_index],
            "flash": frames[-1],
        }
        for stage, image in selected.items():
            save_keyframe(image, tier=tier, stage=stage)
        story_frames[tier.key] = selected
        records.append(public_record(tier, gif_path))

    storyboard = REVIEW / "05_forged_chain_bound_break_flash.png"
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
