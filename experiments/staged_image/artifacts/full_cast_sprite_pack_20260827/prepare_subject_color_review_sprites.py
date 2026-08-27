#!/usr/bin/env python3
"""Build review sprites without recoloring opaque magenta-adjacent details.

The original physical unmix is useful for luminous effects, but it interprets
every magenta-dominant pixel as a foreground/background mixture.  That shifts
opaque red, pink, and mauve subject colors.  This alternate lane first limits
key removal to near-key components connected to the image border.  It preserves
the source RGB exactly for every opaque or substantially visible subject pixel;
only low-alpha boundary pixels receive a softly blended decontamination.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from prepare_review_sprites import (
    ROOT,
    VARIANTS,
    build_contact_sheet,
    normalize_candidate,
    sha256_path,
)


BORDER_SAMPLE = 12
BACKGROUND_ZERO_DISTANCE = 42.0
BACKGROUND_CONNECTION_DISTANCE = 192.0
DECONTAMINATE_BELOW_ALPHA = 160


def _background_connected(
    mask: np.ndarray, definite_key: np.ndarray
) -> np.ndarray:
    """Return near-key components seeded by the screen, including enclosed gaps.

    Border contact identifies the outer screen.  Small spaces enclosed by hair,
    limbs, or crossing wings cannot reach the image border, so a component is
    also background when it contains at least four pixels that are themselves
    within the zero-alpha key distance.  Subject-colored components do not get
    this seed and remain opaque.
    """

    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    border_labels = np.unique(
        np.concatenate(
            [labels[0], labels[-1], labels[:, 0], labels[:, -1]], axis=0
        )
    )
    border_labels = border_labels[border_labels != 0]
    if not border_labels.size:
        raise RuntimeError("near-key mask has no border-connected component")
    definite_counts = np.bincount(
        labels[definite_key].ravel(), minlength=int(labels.max()) + 1
    )
    enclosed_background_labels = np.flatnonzero(definite_counts >= 4)
    seed_labels = np.unique(
        np.concatenate([border_labels, enclosed_background_labels])
    )
    seed_labels = seed_labels[seed_labels != 0]
    return np.isin(labels, seed_labels)


def subject_color_preserving_matte(
    source: Path, destination: Path
) -> dict[str, object]:
    """Extract alpha from connected key colors while preserving subject RGB.

    Euclidean RGB distance is used only to define a permissive near-key region.
    Pixels can receive transparency only when that region connects to an image
    border, so isolated red eyes and pink costume details remain foreground.
    A smoothstep from 42 to 192 RGB-distance units supplies antialiasing.  RGB
    decontamination is restricted to connected boundary pixels with alpha below
    160/255.  Every pixel at or above that alpha keeps its source RGB exactly.
    """

    rgb_u8 = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    rgb = rgb_u8.astype(np.float32)
    border = np.concatenate(
        [
            rgb[:BORDER_SAMPLE].reshape(-1, 3),
            rgb[-BORDER_SAMPLE:].reshape(-1, 3),
            rgb[:, :BORDER_SAMPLE].reshape(-1, 3),
            rgb[:, -BORDER_SAMPLE:].reshape(-1, 3),
        ],
        axis=0,
    )
    key = np.median(border, axis=0)
    key_dominance = float((key[0] + key[2]) * 0.5 - key[1])
    if key_dominance < 96:
        raise RuntimeError(f"{source} does not have a strong magenta border: {key}")

    distance = np.linalg.norm(rgb - key, axis=2)
    definite_key = distance <= BACKGROUND_ZERO_DISTANCE
    connected_key = _background_connected(
        distance <= BACKGROUND_CONNECTION_DISTANCE,
        definite_key,
    )

    transition = np.clip(
        (distance - BACKGROUND_ZERO_DISTANCE)
        / (BACKGROUND_CONNECTION_DISTANCE - BACKGROUND_ZERO_DISTANCE),
        0.0,
        1.0,
    )
    transition = transition * transition * (3.0 - 2.0 * transition)
    alpha = np.ones(distance.shape, dtype=np.float32)
    alpha[connected_key] = transition[connected_key]
    alpha_u8 = np.round(alpha * 255.0).astype(np.uint8)

    foreground = rgb.copy()
    transparent = alpha_u8 == 0
    foreground[transparent] = 0.0

    low_alpha_boundary = (
        connected_key
        & (alpha_u8 > 0)
        & (alpha_u8 < DECONTAMINATE_BELOW_ALPHA)
    )
    if np.any(low_alpha_boundary):
        denominator = np.maximum(alpha[..., None], 0.08)
        unmixed = np.clip(
            (rgb - (1.0 - alpha[..., None]) * key) / denominator,
            0.0,
            255.0,
        )
        foreground[low_alpha_boundary] = unmixed[low_alpha_boundary]

    foreground_u8 = np.round(foreground).astype(np.uint8)
    opaque_or_visible = alpha_u8 >= DECONTAMINATE_BELOW_ALPHA
    changed_visible = np.any(foreground_u8 != rgb_u8, axis=2) & opaque_or_visible
    if np.any(changed_visible):
        raise RuntimeError(
            f"opaque/interior RGB changed for {int(changed_visible.sum())} pixels"
        )

    rgba = np.dstack([foreground_u8, alpha_u8])
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(destination, optimize=True)
    return {
        "sampled_key_rgb": [round(float(value), 3) for value in key],
        "background_zero_distance": BACKGROUND_ZERO_DISTANCE,
        "background_connection_distance": BACKGROUND_CONNECTION_DISTANCE,
        "decontaminate_below_alpha": DECONTAMINATE_BELOW_ALPHA,
        "border_connected_pixels": int(connected_key.sum()),
        "partial_alpha_pixels": int(((alpha_u8 > 0) & (alpha_u8 < 255)).sum()),
        "decontaminated_pixels": int(low_alpha_boundary.sum()),
        "changed_visible_pixels": 0,
        "candidate_sha256": sha256_path(destination),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True)
    parser.add_argument("--raw-suffix", default="_chroma_v1.png")
    parser.add_argument(
        "--raw-override",
        action="append",
        default=[],
        metavar="VARIANT=FILENAME",
        help="select an accepted non-default raw for one variant",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_overrides: dict[str, str] = {}
    for value in args.raw_override:
        variant, separator, filename = value.partition("=")
        if not separator or variant not in VARIANTS or Path(filename).name != filename:
            raise RuntimeError(
                "--raw-override must be VARIANT=FILENAME with a known variant "
                "and a basename-only filename"
            )
        raw_overrides[variant] = filename
    for variant in VARIANTS:
        raw_name = raw_overrides.get(variant, f"{variant}{args.raw_suffix}")
        raw = ROOT / "generation_raw" / args.character / raw_name
        candidate = ROOT / "candidates" / args.character / f"{variant}.png"
        sprite = ROOT / "sprites" / args.character / f"{variant}.png"
        if not raw.exists():
            raise RuntimeError(f"missing raw variant: {raw}")
        matte_metadata = subject_color_preserving_matte(raw, candidate)
        normalize_metadata = normalize_candidate(candidate, sprite)
        print(
            f"{variant}: matte={matte_metadata} normalize={normalize_metadata}"
        )
    sheet = build_contact_sheet(args.character)
    print(f"contact_sheet: {sheet} sha256={sha256_path(sheet)}")


if __name__ == "__main__":
    main()
