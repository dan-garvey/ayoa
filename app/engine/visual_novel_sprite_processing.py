"""Deterministic matte and normalization for generated VN sprite candidates."""

from __future__ import annotations

import hashlib
import os
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from app.engine.player_media import PlayerMediaError, ResolvedPlayerMedia
from app.schemas.image_generation import FrozenReferenceInput


SPRITE_WIDTH = 1100
SPRITE_HEIGHT = 1500
SPRITE_BASELINE_Y = 1480
SPRITE_MAX_SUBJECT_HEIGHT = 1420
SPRITE_MAX_SUBJECT_WIDTH = 1060


def materialize_visual_novel_sprite(
    media: ResolvedPlayerMedia,
    *,
    runtime_root: str | Path,
) -> FrozenReferenceInput:
    """Remove a connected magenta key and normalize one immutable RGBA PNG.

    The key estimate deliberately uses combined red-and-blue dominance. The
    earlier red-only experiment misclassified warm skin, hair, and costume
    pixels; border connectivity plus physical unmixing confines removal to the
    authored screen around the subject.
    """

    try:
        with Image.open(BytesIO(media.data)) as opened:
            opened.load()
            if getattr(opened, "is_animated", False):
                raise PlayerMediaError("sprite_source_animated")
            rgb_image = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise PlayerMediaError("sprite_source_invalid") from exc

    rgb = np.asarray(rgb_image, dtype=np.float32)
    height, width, _channels = rgb.shape
    border_size = max(4, min(16, min(width, height) // 64))
    border = np.concatenate((
        rgb[:border_size].reshape(-1, 3),
        rgb[-border_size:].reshape(-1, 3),
        rgb[:, :border_size].reshape(-1, 3),
        rgb[:, -border_size:].reshape(-1, 3),
    ))
    dominance = (border[:, 0] + border[:, 2]) / 2.0 - border[:, 1]
    likely_key = border[dominance >= 70.0]
    if len(likely_key) < max(64, len(border) // 8):
        raise PlayerMediaError("sprite_key_unavailable")
    key = np.median(likely_key, axis=0)
    if key[0] < 150 or key[2] < 150 or key[1] > 100:
        raise PlayerMediaError("sprite_key_not_magenta")

    distance = np.linalg.norm(rgb - key, axis=2)
    combined_dominance = (rgb[:, :, 0] + rgb[:, :, 2]) / 2.0 - rgb[:, :, 1]
    near_key = (distance < 195.0) & (combined_dominance > 45.0)
    connected = _border_connected_mask(near_key)
    # Arms, capes, weapons, and long hair commonly enclose islands of the
    # physical screen. Those islands do not touch the image border, so a pure
    # flood fill leaves opaque magenta holes in the final sprite. Seed every
    # near-exact key pixel, then grow only a few anti-aliasing pixels through
    # the already color-bounded near-key mask. This keeps warm skin and red
    # costume material outside the matte while admitting enclosed screen.
    enclosed_key = (distance < 64.0) & (combined_dominance > 80.0)
    for _iteration in range(6):
        enclosed_key |= near_key & _dilate_eight(enclosed_key)
    connected |= enclosed_key
    if connected.mean() < 0.08:
        raise PlayerMediaError("sprite_key_area_too_small")

    transition = np.clip((distance - 32.0) / 155.0, 0.0, 1.0)
    transition = transition * transition * (3.0 - 2.0 * transition)
    alpha = np.full((height, width), 255.0, dtype=np.float32)
    alpha[connected] = transition[connected] * 255.0

    foreground = rgb.copy()
    alpha_unit = alpha / 255.0
    partial = connected & (alpha >= 16.0) & (alpha < 208.0)
    if partial.any():
        values = alpha_unit[partial, None]
        foreground[partial] = np.clip(
            (rgb[partial] - (1.0 - values) * key) / values,
            0.0,
            255.0,
        )
    foreground[alpha < 8.0] = 0.0
    rgba = np.dstack((
        np.rint(foreground).astype(np.uint8),
        np.rint(alpha).astype(np.uint8),
    ))

    visible_y, visible_x = np.nonzero(rgba[:, :, 3] >= 32)
    if len(visible_x) < width * height * 0.02:
        raise PlayerMediaError("sprite_subject_too_small")
    if (rgba[:, :, 3] <= 8).mean() < 0.05:
        raise PlayerMediaError("sprite_background_not_transparent")
    left, right = int(visible_x.min()), int(visible_x.max()) + 1
    top, bottom = int(visible_y.min()), int(visible_y.max()) + 1
    subject = Image.fromarray(rgba, mode="RGBA").crop((left, top, right, bottom))

    scale = min(
        SPRITE_MAX_SUBJECT_WIDTH / subject.width,
        SPRITE_MAX_SUBJECT_HEIGHT / subject.height,
    )
    target = (
        max(1, round(subject.width * scale)),
        max(1, round(subject.height * scale)),
    )
    subject = (
        subject.convert("RGBa")
        .resize(target, Image.Resampling.LANCZOS)
        .convert("RGBA")
    )
    canvas = Image.new("RGBA", (SPRITE_WIDTH, SPRITE_HEIGHT), (0, 0, 0, 0))
    x = (SPRITE_WIDTH - subject.width) // 2
    y = SPRITE_BASELINE_Y - subject.height
    canvas.alpha_composite(subject, dest=(x, y))

    # A generated candidate is optional presentation material. Inspect the
    # actual normalized foreground after physical unmixing and resampling, and
    # reject it conservatively instead of surfacing a visible hot-key island
    # or fringe. Applying this gate to the raw keyed image would reject clean
    # candidates merely because their yet-to-be-unmixed edge pixels resemble
    # the screen color.
    canvas_pixels = np.asarray(canvas, dtype=np.uint8)
    final_rgb = canvas_pixels[:, :, :3].astype(np.float32)
    final_distance = np.linalg.norm(final_rgb - key, axis=2)
    final_dominance = (
        (final_rgb[:, :, 0] + final_rgb[:, :, 2]) / 2.0
        - final_rgb[:, :, 1]
    )
    visible_hot_key = (
        (canvas_pixels[:, :, 3] >= 32)
        & (final_distance < 96.0)
        & (final_dominance > 80.0)
    )
    if visible_hot_key.sum() > max(
        64,
        SPRITE_WIDTH * SPRITE_HEIGHT // 20_000,
    ):
        raise PlayerMediaError("sprite_key_residue")

    output = BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    data = output.getvalue()
    sha256 = hashlib.sha256(data).hexdigest()
    root = Path(runtime_root).resolve()
    destination_dir = root / "artifacts" / "sprites"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{sha256}.png"
    if destination.exists():
        if destination.read_bytes() != data:
            raise PlayerMediaError("sprite_content_address_collision")
    else:
        temporary = destination_dir / f".{uuid.uuid4().hex}.tmp"
        temporary.write_bytes(data)
        os.replace(temporary, destination)
        try:
            destination.chmod(0o600)
        except OSError:
            pass
    return FrozenReferenceInput(
        reference_id=f"imgsprite_{sha256[:32]}",
        sha256=sha256,
        mime_type="image/png",
        width=SPRITE_WIDTH,
        height=SPRITE_HEIGHT,
        byte_count=len(data),
        relative_path=str(destination.relative_to(root)).replace("\\", "/"),
        allowed_root="artifacts",
    )


def _border_connected_mask(candidate: np.ndarray) -> np.ndarray:
    height, width = candidate.shape
    padded = Image.new("L", (width + 2, height + 2), 255)
    padded.paste(Image.fromarray(candidate.astype(np.uint8) * 255), (1, 1))
    ImageDraw.floodfill(padded, (0, 0), 128, thresh=0)
    flooded = np.asarray(padded, dtype=np.uint8)[1:-1, 1:-1]
    return flooded == 128


def _dilate_eight(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    height, width = mask.shape
    return np.logical_or.reduce(tuple(
        padded[y : y + height, x : x + width]
        for y in range(3)
        for x in range(3)
    ))
