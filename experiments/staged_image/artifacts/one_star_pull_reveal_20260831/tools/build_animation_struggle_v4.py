#!/usr/bin/env python3
"""Build V4 One-Star summon proofs with escalating unsealing struggle."""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import build_animation_forged_chain_v3 as v3
import build_animation_proofs as v1
import build_animation_release_v2 as v2
import build_proofs as base


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review_animation_struggle_v4"
PROOFS = ROOT / "proofs/animation_struggle_v4"
KEYFRAMES = PROOFS / "keyframes"
PROVENANCE = ROOT / "provenance"
MANIFEST = ROOT / "animation_struggle_v4_manifest.json"
WINDOWS_REVIEW = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/"
    "OneStarPullRevealStruggleV4_20260831"
)


@dataclass(frozen=True)
class StruggleSpec:
    pulse_centers: tuple[float, ...]
    pulse_widths: tuple[float, ...]
    pulse_tugs: tuple[float, ...]
    release_start: float
    chain_wiggle: float
    lock_wiggle_px: float
    card_shake_px: float


@dataclass(frozen=True)
class MotionSample:
    motion_progress: float
    strain: float
    energy: float
    phase: float
    releasing: bool


SPECS = {
    "under_three": StruggleSpec(
        pulse_centers=(0.25,),
        pulse_widths=(0.090,),
        pulse_tugs=(0.10,),
        release_start=0.46,
        chain_wiggle=0.0,
        lock_wiggle_px=0.0,
        card_shake_px=1.0,
    ),
    "three_to_four": StruggleSpec(
        pulse_centers=(0.18, 0.40),
        pulse_widths=(0.072, 0.078),
        pulse_tugs=(0.07, 0.135),
        release_start=0.56,
        chain_wiggle=0.0,
        lock_wiggle_px=2.5,
        card_shake_px=1.5,
    ),
    "five_to_six": StruggleSpec(
        pulse_centers=(0.15, 0.34, 0.53),
        pulse_widths=(0.057, 0.062, 0.068),
        pulse_tugs=(0.06, 0.10, 0.15),
        release_start=0.62,
        chain_wiggle=0.72,
        lock_wiggle_px=3.5,
        card_shake_px=2.0,
    ),
    "seven": StruggleSpec(
        pulse_centers=(0.12, 0.29, 0.46, 0.60),
        pulse_widths=(0.046, 0.051, 0.056, 0.061),
        pulse_tugs=(0.05, 0.08, 0.12, 0.16),
        release_start=0.67,
        chain_wiggle=1.0,
        lock_wiggle_px=4.5,
        card_shake_px=2.7,
    ),
}

FRAME_COUNTS = (18, 36, 46, 56)
FINAL_HOLDS = (380, 520, 620, 820)
FILENAMES = (
    "01_iron_plain_strain_release_under_3.gif",
    "02_silver_lock_struggle_3_to_4.gif",
    "03_gold_chain_struggle_5_to_6.gif",
    "04_white_gold_chain_struggle_7.gif",
)
LABELS = (
    "UNDER 3 STARS // ONE RESTRAINED PULSE",
    "3-4 STARS // TWO LOCK TUGS",
    "5-6 STARS // THREE CHAIN STRAINS",
    "7 STARS // FOUR ESCALATING STRAINS",
)
KICKERS = (
    "SYSTEM ACQUISITION",
    "SIGNATURE ANOMALY",
    "SIGNATURE ANOMALY",
    "CONTAINMENT OVERLOAD",
)
TITLES = (
    "SIGNATURE RELEASING",
    "CONTAINMENT RESISTS",
    "CONTAINMENT RESISTS",
    "THE SEAL RESISTS",
)
FOOTERS = (
    "SYSTEM PRESSURE",
    "LOCKS RESISTING",
    "FORGED SEAL UNDER STRAIN",
    "CRITICAL SEAL STRAIN",
)

TIERS = tuple(
    replace(
        tier,
        filename=filename,
        review_label=label,
        frame_count=frame_count,
        final_hold_ms=final_hold,
        kicker=kicker,
        title=title,
        footer=footer,
    )
    for tier, filename, label, frame_count, final_hold, kicker, title, footer in zip(
        v3.TIERS,
        FILENAMES,
        LABELS,
        FRAME_COUNTS,
        FINAL_HOLDS,
        KICKERS,
        TITLES,
        FOOTERS,
        strict=True,
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def motion_sample(spec: StruggleSpec, raw_progress: float) -> MotionSample:
    if raw_progress >= spec.release_start:
        release_fraction = v1.smoothstep(
            (raw_progress - spec.release_start) / (1.0 - spec.release_start)
        )
        return MotionSample(
            motion_progress=0.10 + release_fraction * 0.90,
            strain=0.0,
            energy=0.0,
            phase=raw_progress * math.tau * 5,
            releasing=True,
        )

    tug = 0.0
    signed_strain = 0.0
    energy = 0.0
    for index, (center, width, amplitude) in enumerate(
        zip(
            spec.pulse_centers,
            spec.pulse_widths,
            spec.pulse_tugs,
            strict=True,
        )
    ):
        local = (raw_progress - center) / width
        envelope = math.exp(-(local**2) * 1.55)
        tug += amplitude * envelope
        normalized_amplitude = amplitude / max(spec.pulse_tugs)
        energy = max(energy, envelope * normalized_amplitude)
        signed_strain += (
            envelope
            * normalized_amplitude
            * math.cos(local * math.pi * 1.35 + index * 0.28)
        )
    pressure_ramp = 0.018 * raw_progress / spec.release_start
    return MotionSample(
        motion_progress=min(0.255, 0.10 + pressure_ramp + tug),
        strain=max(-1.0, min(1.0, signed_strain)),
        energy=min(1.0, energy),
        phase=raw_progress * math.tau * (7.0 + len(spec.pulse_centers)),
        releasing=False,
    )


def shifted_layer(layer: Image.Image, offset: tuple[int, int]) -> Image.Image:
    if offset == (0, 0):
        return layer
    shifted = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    shifted.alpha_composite(layer, offset)
    return shifted


def struggle_overlay(
    tier: v2.ReleaseTier,
    *,
    sample: MotionSample,
) -> Image.Image:
    layer = Image.new("RGBA", base.BOARD_SIZE, (0, 0, 0, 0))
    if sample.energy < 0.025 or sample.releasing:
        return layer
    band = base.BANDS[tier.band_key]
    draw = ImageDraw.Draw(layer)
    alpha = round(sample.energy * (40 + tier.intensity * 16))
    extent = 154 + tier.intensity * 7
    for side in (-1, 1):
        x = v2.CARD_CENTER[0] + side * extent
        for index in range(2 + tier.intensity):
            y = 190 + index * (250 / (1 + tier.intensity))
            reach = 8 + index % 2 * 6 + sample.energy * 10
            draw.line(
                (x, y, x + side * reach, y + sample.strain * 5),
                fill=(*band.highlight, min(180, alpha)),
                width=1 + tier.intensity // 3,
            )
    radius_x = 174 + sample.energy * 13
    radius_y = 133 + sample.energy * 9
    draw.arc(
        (
            v2.CARD_CENTER[0] - radius_x,
            v2.CARD_CENTER[1] - radius_y,
            v2.CARD_CENTER[0] + radius_x,
            v2.CARD_CENTER[1] + radius_y,
        ),
        205 + sample.strain * 8,
        335 + sample.strain * 8,
        fill=(*band.highlight, min(170, alpha + 18)),
        width=1 + tier.intensity // 2,
    )
    return layer.filter(ImageFilter.GaussianBlur(0.45))


def animation_frame(
    tier: v2.ReleaseTier,
    sealed: Image.Image,
    *,
    raw_progress: float,
) -> Image.Image:
    spec = SPECS[tier.key]
    sample = motion_sample(spec, raw_progress)
    chain_strain = sample.strain * spec.chain_wiggle

    def render_chains(
        received_tier: v2.ReleaseTier,
        *,
        progress: float,
        card_box: tuple[int, int, int, int],
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        return v3.forged_chains(
            received_tier,
            progress=progress,
            card_box=card_box,
            strain=chain_strain,
            strain_phase=sample.phase,
        )

    def render_locks(
        received_tier: v2.ReleaseTier,
        *,
        progress: float,
        card_box: tuple[int, int, int, int],
    ) -> Image.Image:
        locks = v2.ejected_locks(
            received_tier,
            progress=progress,
            card_box=card_box,
        )
        wiggle_y = round(sample.strain * spec.lock_wiggle_px)
        wiggle_x = round(
            math.sin(sample.phase) * sample.energy * spec.lock_wiggle_px * 0.45
        )
        return shifted_layer(locks, (wiggle_x, wiggle_y))

    card_offset = (
        round(sample.strain * spec.card_shake_px),
        round(
            math.sin(sample.phase * 0.83)
            * sample.energy
            * spec.card_shake_px
            * 0.55
        ),
    )
    frame = v2.animation_frame(
        tier,
        sealed,
        progress=sample.motion_progress,
        chain_renderer=render_chains,
        lock_renderer=render_locks,
        card_offset=card_offset,
    ).convert("RGBA")
    frame.alpha_composite(struggle_overlay(tier, sample=sample))
    return frame.convert("RGB")


def keyframe_raw_progress(tier: v2.ReleaseTier, stage: str) -> float:
    spec = SPECS[tier.key]
    if stage == "bound":
        return 0.0
    if stage == "strain":
        return spec.pulse_centers[-1]
    if stage == "break":
        return spec.release_start + (1.0 - spec.release_start) * 0.50
    if stage == "flash":
        return 1.0
    raise ValueError(stage)


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
        kicker="MOTION REVIEW V4",
        title="BOUND / STRAIN / BREAK / FLASH",
        accent=(225, 205, 156),
    )
    draw = ImageDraw.Draw(canvas)
    thumb_size = (229, 101)
    stages = ("bound", "strain", "break", "flash")
    for tier_index, tier in enumerate(TIERS):
        band = base.BANDS[tier.band_key]
        x = 22 + tier_index * 246
        label = tier.review_label.split(" // ")[0]
        draw.text(
            (x, 98),
            label,
            font=base.font(base.SANS_BOLD, 13),
            fill=(*band.highlight, 255),
        )
        for stage_index, stage in enumerate(stages):
            y = 120 + stage_index * 106
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
            tag_width = 53
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


def spec_record(spec: StruggleSpec) -> dict[str, object]:
    return {
        "pulse_count": len(spec.pulse_centers),
        "pulse_centers": list(spec.pulse_centers),
        "pulse_widths": list(spec.pulse_widths),
        "pulse_tugs": list(spec.pulse_tugs),
        "release_start": spec.release_start,
        "chain_wiggle": spec.chain_wiggle,
        "lock_wiggle_px": spec.lock_wiggle_px,
        "card_shake_px": spec.card_shake_px,
        "pulse_pattern": "tug_recoil_then_stronger_tug",
        "final_release": "single_decisive_outward_snap",
    }


def public_record(tier: v2.ReleaseTier, path: Path) -> dict[str, object]:
    geometry = v2.release_geometry(tier)
    geometry["chain_visual_style"] = (
        "heavy_forged_interlocking_links" if tier.chain_count else "none"
    )
    geometry["chain_segment_motion"] = (
        "coherent_rigid_halves" if tier.chain_count else "none"
    )
    geometry["fracture_shape"] = (
        "open_ends_attached_to_receding_segments" if tier.chain_count else "none"
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
        "struggle": spec_record(SPECS[tier.key]),
        "release_geometry": geometry,
        "final_frame": "held_pre_reveal_flash",
        "review_loop": True,
        "production_intent": "play_once_then_static_result",
        "accessible_text": (
            f"{tier.review_label}; seal strains and recoils before outward release; "
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
        f"""ONE-STAR PULL REVEAL — UNSEALING STRUGGLE V4

V4 preserves V3's forged chains, outward release, and held flash. Before the
final break, the seal now visibly resists: locks tug away and recoil, chains
tighten and ripple, the card shudders, and each pulse escalates by rarity.

{lines}

PULSE LADDER
- Under 3 stars: one restrained system pulse before the plain release.
- 3-4 stars: two lock tugs, with the second stronger than the first.
- 5-6 stars: three escalating strains through the forged chains and locks.
- 7 stars: four increasingly violent strains before one decisive snap.

These are mechanical anticipation beats, not rarity fakeouts. No lock opens
and no chain breaks until the final release. Review GIFs loop; production
intent is one play, held flash, then the authoritative static result.
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
<title>One-Star Pull Reveal — Unsealing Struggle V4</title>
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
<h1>One-Star Pull Reveal — Unsealing Struggle V4</h1>
<p>Escalating tug-and-recoil anticipation before the forged restraints finally release outward.</p>
<main>{''.join(figures)}</main>
</body>
</html>
"""
    (REVIEW / "index.html").write_text(document, encoding="utf-8")


def write_manifest(records: list[dict[str, object]], storyboard: Path) -> None:
    manifest = {
        "schema_version": 4,
        "status": "visual_review_only",
        "direction": "escalating_struggle_then_outward_release",
        "supersedes_for_review": "forged_chain_v3_motion_timing",
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
        "motion_revision": {
            "principle": "anticipation through escalating resistance",
            "pre_release": "tug recoil pulse sequence",
            "locks": "wiggle and tug without opening",
            "chains": "tighten and ripple without breaking",
            "card": "small restrained shudder",
            "release": "one decisive outward snap",
            "rank_fakeout": False,
        },
        "invariants": {
            "forged_chain_style": True,
            "outward_lock_motion": True,
            "held_pre_reveal_flash": True,
            "identity_before_static_result": "withheld",
            "exact_rank_before_static_result": "withheld",
            "semantic_fallback": "existing static result board",
            "animation_required_for_comprehension": False,
            "fakeouts": "none",
            "image_generation_calls": 0,
            "v1_v2_v3_preserved": True,
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
    (PROVENANCE / "animation_struggle_v4_source_hashes.json").write_text(
        json.dumps(manifest["source_hashes"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (PROVENANCE / "animation_struggle_v4_decisions.json").write_text(
        json.dumps(
            {
                "revision": "V4 escalating unsealing struggle",
                "v3_visual_style": "preserved",
                "anticipation": "tug recoil pulses",
                "pulse_count_by_band": [1, 2, 3, 4],
                "pulse_strength": "strictly escalating within and across bands",
                "pre_release_breaks": 0,
                "final_release": "single decisive outward snap",
                "held_flash": True,
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
    provenance_hashes = PROVENANCE / "animation_struggle_v4_SHA256SUMS.txt"
    owned = sorted(KEYFRAMES.glob("*.png"))
    owned.extend(
        (
            MANIFEST,
            PROVENANCE / "animation_struggle_v4_source_hashes.json",
            PROVENANCE / "animation_struggle_v4_decisions.json",
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
                raw_progress=index / (tier.frame_count - 1),
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
        selected: dict[str, Image.Image] = {}
        for stage in ("bound", "strain", "break", "flash"):
            raw = keyframe_raw_progress(tier, stage)
            selected[stage] = animation_frame(
                tier,
                sealed,
                raw_progress=raw,
            )
            save_keyframe(selected[stage], tier=tier, stage=stage)
        story_frames[tier.key] = selected
        records.append(public_record(tier, gif_path))

    storyboard = REVIEW / "05_bound_strain_break_flash_storyboard.png"
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
