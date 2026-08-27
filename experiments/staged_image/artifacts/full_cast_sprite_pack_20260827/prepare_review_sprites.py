#!/usr/bin/env python3
"""Extract magenta mattes and assemble normalized experimental VN sprites."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
VARIANTS = [
    "neutral",
    "happy",
    "concerned",
    "tense",
    "skeptical",
    "angry",
    "sad",
    "surprised",
]
CANVAS_SIZE = (1100, 1500)
TARGET_HEIGHT = 1420
MAX_WIDTH = 1060
BASELINE_Y = 1480
ALPHA_NOISE_CUTOFF = 48
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_magenta_matte(source: Path, destination: Path) -> None:
    """Recover alpha and edge RGB by unmixing the sampled magenta screen.

    The generic chroma helper's single-channel dominance heuristic treats a
    magenta key as red-only and therefore makes ordinary skin translucent.
    Magenta is instead modeled by the combined red/blue dominance over green.
    Solving ``observed = alpha * foreground + (1-alpha) * key`` then removes
    the colored fringe while retaining naturally translucent lantern glow.
    """

    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32)
    border = np.concatenate(
        [
            rgb[:12].reshape(-1, 3),
            rgb[-12:].reshape(-1, 3),
            rgb[:, :12].reshape(-1, 3),
            rgb[:, -12:].reshape(-1, 3),
        ],
        axis=0,
    )
    key = np.median(border, axis=0)
    key_dominance = float((key[0] + key[2]) * 0.5 - key[1])
    if key_dominance < 96:
        raise RuntimeError(f"{source} does not have a strong magenta border: {key}")

    dominance = (rgb[..., 0] + rgb[..., 2]) * 0.5 - rgb[..., 1]
    alpha = np.clip(1.0 - dominance / key_dominance, 0.0, 1.0)
    distance = np.max(np.abs(rgb - key), axis=2)
    alpha[distance <= 3] = 0.0
    alpha[dominance <= 4] = 1.0

    # Drop disconnected dust without erasing connected hair wisps, weapons,
    # lantern lattice, or the glow attached to the held lantern.
    component_mask = (alpha >= 0.035).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        component_mask, connectivity=8
    )
    keep = np.zeros_like(component_mask)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= 100:
            keep[labels == index] = 1
    alpha *= keep
    alpha[alpha < 0.025] = 0.0
    alpha[alpha > 0.985] = 1.0

    denominator = np.maximum(alpha[..., None], 0.06)
    foreground = (rgb - (1.0 - alpha[..., None]) * key) / denominator
    foreground = np.clip(foreground, 0, 255)
    foreground[alpha >= 0.985] = rgb[alpha >= 0.985]
    foreground[alpha <= 0] = 0
    rgba = np.dstack(
        [foreground.astype(np.uint8), np.round(alpha * 255).astype(np.uint8)]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(destination, optimize=True)


def normalize_candidate(source: Path, destination: Path) -> dict[str, object]:
    with Image.open(source) as image:
        rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8).copy()
    alpha[alpha <= ALPHA_NOISE_CUTOFF] = 0
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    ys, xs = np.where(alpha >= 64)
    if not xs.size:
        raise RuntimeError(f"empty matte: {source}")
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    crop = rgba.crop(bbox)
    scale = min(TARGET_HEIGHT / crop.height, MAX_WIDTH / crop.width)
    resized = (
        max(1, round(crop.width * scale)),
        max(1, round(crop.height * scale)),
    )
    crop = crop.resize(resized, Image.Resampling.LANCZOS)
    x = (CANVAS_SIZE[0] - crop.width) // 2
    y = BASELINE_Y - crop.height
    if x < 0 or y < 0 or x + crop.width > CANVAS_SIZE[0] or y + crop.height > CANVAS_SIZE[1]:
        raise RuntimeError(f"normalized sprite does not fit: {source} -> {resized}")
    canvas = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    canvas.alpha_composite(crop, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
    return {
        "source_bbox": list(bbox),
        "scale": round(scale, 8),
        "placement": [x, y, crop.width, crop.height],
        "sha256": sha256_path(destination),
    }


def split_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (235, 238, 242))
    ImageDraw.Draw(image).rectangle((0, 0, width // 2, height), fill=(31, 36, 44))
    return image


def build_contact_sheet(character: str) -> Path:
    columns, rows = 4, 2
    cell_width, cell_height = 480, 650
    art_width, art_height = 450, 575
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (18, 22, 28))
    label_font = ImageFont.truetype(str(FONT), 28)
    small_font = ImageFont.truetype(str(FONT), 18)
    for index, variant in enumerate(VARIANTS):
        with Image.open(ROOT / "sprites" / character / f"{variant}.png") as image:
            rgba = image.convert("RGBA")
        rgba.thumbnail((art_width, art_height), Image.Resampling.LANCZOS)
        cell = split_background((cell_width, cell_height))
        cell.paste(
            rgba,
            ((cell_width - rgba.width) // 2, 8 + (art_height - rgba.height)),
            rgba,
        )
        draw = ImageDraw.Draw(cell)
        draw.rectangle((0, 590, cell_width, cell_height), fill=(12, 15, 20))
        draw.text((16, 598), variant.upper(), fill=(255, 255, 255), font=label_font)
        draw.text(
            (16, 631),
            "dark / light alpha check",
            fill=(165, 174, 186),
            font=small_font,
        )
        row, column = divmod(index, columns)
        sheet.paste(cell, (column * cell_width, row * cell_height))
    output = ROOT / "contact_sheets" / f"{character}_complete_sweep.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True)
    parser.add_argument("--raw-suffix", default="_chroma_v1.png")
    parser.add_argument(
        "--raw-override",
        action="append",
        default=[],
        metavar="LABEL=FILENAME",
    )
    return parser.parse_args()


def raw_overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        label, separator, filename = value.partition("=")
        if not separator or label not in VARIANTS or not filename:
            raise RuntimeError(f"invalid raw override: {value}")
        result[label] = filename
    return result


def main() -> None:
    args = parse_args()
    overrides = raw_overrides(args.raw_override)
    for variant in VARIANTS:
        filename = overrides.get(variant, f"{variant}{args.raw_suffix}")
        raw = ROOT / "generation_raw" / args.character / filename
        candidate = ROOT / "candidates" / args.character / f"{variant}.png"
        sprite = ROOT / "sprites" / args.character / f"{variant}.png"
        if not raw.exists():
            raise RuntimeError(f"missing raw variant: {raw}")
        physical_magenta_matte(raw, candidate)
        metadata = normalize_candidate(candidate, sprite)
        print(f"{variant}: {metadata}")
    sheet = build_contact_sheet(args.character)
    print(f"contact_sheet: {sheet} sha256={sha256_path(sheet)}")


if __name__ == "__main__":
    main()
