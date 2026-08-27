#!/usr/bin/env python3
"""Split Mirelle's approved canonical spear component by material behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "components/mirelle_voss/canonical_lower_spearhead_streamers_rgba_v6.png"
METAL = ROOT / "components/mirelle_voss/canonical_primary_metal_rgba_v1.png"
STREAMERS = ROOT / "components/mirelle_voss/canonical_primary_streamers_rgba_v1.png"
METAL_MASK = ROOT / "masks/mirelle_voss/canonical_primary_metal_mask_v1.png"
STREAMER_MASK = ROOT / "masks/mirelle_voss/canonical_primary_streamer_mask_v1.png"
PROOF = ROOT / "component_proofs/mirelle_canonical_primary_material_split_v1.png"
METADATA = ROOT / "component_metadata/mirelle_canonical_primary_material_split_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        path, optimize=True
    )


def main() -> None:
    source = np.asarray(Image.open(SOURCE).convert("RGBA"), dtype=np.uint8)
    rgb = source[..., :3]
    alpha = source[..., 3]
    foreground = alpha > 0
    red_seed = (
        foreground
        & (rgb[..., 0] >= 52)
        & (rgb[..., 0].astype(np.int16) >= rgb[..., 1].astype(np.int16) + 16)
        & (rgb[..., 0].astype(np.int16) >= rgb[..., 2].astype(np.int16) + 8)
    )
    # Include the dark ink edge adjacent to red fabric without allowing that
    # expansion to walk around the connected metal silhouette.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    red_neighborhood = cv2.dilate(red_seed.astype(np.uint8), kernel) > 0
    dark_edge = foreground & (np.max(rgb, axis=2) <= 92)
    red_tinted_edge = (
        foreground
        & (rgb[..., 0].astype(np.int16) >= rgb[..., 1].astype(np.int16) + 5)
        & (rgb[..., 0].astype(np.int16) >= rgb[..., 2].astype(np.int16) + 3)
    )
    streamer = red_seed | ((dark_edge | red_tinted_edge) & red_neighborhood)

    # Manual material-boundary guards keep socket metal even where it touches
    # red cloth and remove any isolated red pixel from the blade interior.
    socket_guard = Image.new("L", (source.shape[1], source.shape[0]), 0)
    draw = ImageDraw.Draw(socket_guard)
    draw.polygon([(170, 72), (225, 65), (244, 104), (212, 139), (172, 121)], fill=255)
    socket_guard_mask = np.asarray(socket_guard, dtype=np.uint8) > 0
    silver_like = (
        foreground
        & (np.max(rgb, axis=2).astype(np.int16) - np.min(rgb, axis=2).astype(np.int16) <= 32)
    )
    streamer[socket_guard_mask & silver_like] = False
    metal = foreground & ~streamer

    # The rigid assembly is one connected object.  Move any tiny detached
    # antialias fleck to the flexible/rejected material side so transforms can
    # never create a floating metal shard beside the socket.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        metal.astype(np.uint8), connectivity=8
    )
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        detached = metal & (labels != largest)
        metal[detached] = False
        streamer[detached] = True

    # Assign every original foreground pixel exactly once.
    if np.any(metal & streamer) or not np.array_equal(metal | streamer, foreground):
        raise RuntimeError("material masks do not partition the source alpha")
    metal_rgba = source.copy()
    metal_rgba[~metal] = 0
    streamers_rgba = source.copy()
    streamers_rgba[~streamer] = 0

    for path in (METAL, STREAMERS, METAL_MASK, STREAMER_MASK):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(metal_rgba, mode="RGBA").save(METAL, optimize=True)
    Image.fromarray(streamers_rgba, mode="RGBA").save(STREAMERS, optimize=True)
    save_mask(metal, METAL_MASK)
    save_mask(streamer, STREAMER_MASK)

    panel_size = (520, 520)
    cells: list[Image.Image] = []
    for title, rgba in (
        ("APPROVED COMBINED", source),
        ("A: RIGID METAL", metal_rgba),
        ("B: FLEXIBLE STREAMERS", streamers_rgba),
    ):
        canvas = Image.new("RGB", panel_size, (31, 36, 44))
        shown = Image.fromarray(rgba, mode="RGBA")
        shown.thumbnail((470, 450), Image.Resampling.LANCZOS)
        canvas.paste(shown, ((520 - shown.width) // 2, 22), shown)
        ImageDraw.Draw(canvas).text((12, 488), title, fill=(255, 255, 255))
        cells.append(canvas)
    proof = Image.new("RGB", (1560, 520), (12, 15, 20))
    for index, cell in enumerate(cells):
        proof.paste(cell, (index * 520, 0))
    PROOF.parent.mkdir(parents=True, exist_ok=True)
    proof.save(PROOF, optimize=True)

    metadata = {
        "status": "pending_root_review",
        "method": "exact-RGB material partition; red fill plus bounded dark-edge expansion with socket metal guard",
        "source": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
        "partition": {
            "source_foreground_pixels": int(np.count_nonzero(foreground)),
            "metal_pixels": int(np.count_nonzero(metal)),
            "streamer_pixels": int(np.count_nonzero(streamer)),
            "overlap_pixels": int(np.count_nonzero(metal & streamer)),
            "unassigned_pixels": int(np.count_nonzero(foreground & ~(metal | streamer))),
            "source_rgb_preserved_exactly": True,
        },
        "metal": {
            "path": str(METAL),
            "sha256": sha256(METAL),
            "mask_path": str(METAL_MASK),
            "mask_sha256": sha256(METAL_MASK),
        },
        "streamers": {
            "path": str(STREAMERS),
            "sha256": sha256(STREAMERS),
            "mask_path": str(STREAMER_MASK),
            "mask_sha256": sha256(STREAMER_MASK),
        },
        "proof": {"path": str(PROOF), "sha256": sha256(PROOF)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
