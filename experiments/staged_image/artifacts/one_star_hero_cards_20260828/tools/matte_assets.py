#!/usr/bin/env python3
"""Freeze BiRefNet masks and RGBA cutouts for the hero-card proof set."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import urllib.request
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matte_one(
    source_path: Path,
    mask_path: Path,
    rgba_path: Path,
    *,
    gateway: str,
    worker: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "image_base64": base64.b64encode(source_path.read_bytes()).decode("ascii"),
        "filename_prefix": f"one-star-card-{source_path.stem}",
        "model_name": "birefnet.safetensors",
    }
    if worker:
        payload["worker"] = worker
    request = urllib.request.Request(
        f"{gateway.rstrip('/')}/prototype/matte/birefnet",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = response.read()
        response_headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower().startswith("x-ayoa-")
        }

    with Image.open(io.BytesIO(body)) as received:
        mask = received.convert("L")
    with Image.open(source_path) as received:
        source = received.convert("RGB")
    if mask.size != source.size:
        mask = mask.resize(source.size, Image.Resampling.LANCZOS)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    rgba_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path, format="PNG", optimize=True)
    rgba = source.convert("RGBA")
    rgba.putalpha(mask)
    rgba.save(rgba_path, format="PNG", optimize=True)

    alpha_extrema = mask.getextrema()
    histogram = mask.histogram()
    pixel_count = mask.width * mask.height
    transparent = histogram[0]
    opaque = histogram[255]
    fringe = pixel_count - transparent - opaque
    bbox = mask.point(lambda value: 255 if value >= 32 else 0).getbbox()
    return {
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "mask": str(mask_path),
        "mask_sha256": sha256(mask_path),
        "rgba": str(rgba_path),
        "rgba_sha256": sha256(rgba_path),
        "dimensions": [source.width, source.height],
        "alpha_extrema": list(alpha_extrema),
        "transparent_fraction": round(transparent / pixel_count, 6),
        "opaque_fraction": round(opaque / pixel_count, 6),
        "fringe_fraction": round(fringe / pixel_count, 6),
        "foreground_bbox_at_32": list(bbox) if bbox else None,
        "model": "BiRefNet",
        "model_file": "birefnet.safetensors",
        "gateway": gateway,
        "response_headers": response_headers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("rgba", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--gateway", default="http://127.0.0.1:8199")
    parser.add_argument("--worker")
    args = parser.parse_args()

    record = matte_one(
        args.source,
        args.mask,
        args.rgba,
        gateway=args.gateway,
        worker=args.worker,
    )
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
