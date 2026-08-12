from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.engine.content_asset_bytes import DEFAULT_MAX_ASSET_BYTES
from app.schemas.image_generation import (
    GeneratedImageArtifact,
    ImageWorkerResult,
)


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class PlayerMediaError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Player media validation failed ({code}).")


@runtime_checkable
class PlayerMediaBytes(Protocol):
    filename: str
    mime_type: str
    data: bytes
    sha256: str
    byte_count: int
    width: int
    height: int


@dataclass(frozen=True)
class ResolvedPlayerMedia:
    filename: str
    mime_type: str
    data: bytes
    sha256: str
    byte_count: int
    width: int
    height: int


def finalize_generated_webp(
    temp_path: str | Path,
    *,
    runtime_root: str | Path,
    worker_result: ImageWorkerResult,
    expected_width: int,
    expected_height: int,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
) -> GeneratedImageArtifact:
    """Validate and atomically content-address one worker-produced WebP."""

    path = Path(temp_path)
    data = _read_limited(path, max_bytes=max_bytes)
    width, height = webp_dimensions(data)
    if (width, height) != (expected_width, expected_height):
        raise PlayerMediaError("dimensions_mismatch")
    if worker_result.width != width or worker_result.height != height:
        raise PlayerMediaError("worker_dimensions_mismatch")
    if worker_result.mime_type.strip().lower() != "image/webp":
        raise PlayerMediaError("worker_mime_mismatch")
    if worker_result.byte_count != len(data):
        raise PlayerMediaError("worker_byte_count_mismatch")

    sha256 = hashlib.sha256(data).hexdigest()
    if worker_result.sha256.strip().lower() != sha256:
        raise PlayerMediaError("worker_hash_mismatch")

    root = Path(runtime_root).resolve()
    artifact_dir = root / "artifacts" / sha256[:2]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _chmod_private(artifact_dir, directory=True)
    destination = artifact_dir / f"{sha256}.webp"
    if destination.exists():
        if _read_limited(destination, max_bytes=max_bytes) != data:
            raise PlayerMediaError("content_address_collision")
        path.unlink(missing_ok=True)
    else:
        os.replace(path, destination)
        _chmod_private(destination, directory=False)

    return GeneratedImageArtifact(
        sha256=sha256,
        relative_path=str(destination.relative_to(root)),
        mime_type="image/webp",
        width=width,
        height=height,
        byte_count=len(data),
    )


def resolve_generated_media(
    artifact: GeneratedImageArtifact,
    *,
    runtime_root: str | Path,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
) -> ResolvedPlayerMedia:
    root = Path(runtime_root).resolve()
    path = (root / artifact.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PlayerMediaError("artifact_path_outside_root") from exc

    data = _read_limited(path, max_bytes=max_bytes)
    sha256 = hashlib.sha256(data).hexdigest()
    if not _SHA256_RE.fullmatch(artifact.sha256) or sha256 != artifact.sha256:
        raise PlayerMediaError("artifact_hash_mismatch")
    if len(data) != artifact.byte_count:
        raise PlayerMediaError("artifact_byte_count_mismatch")
    width, height = webp_dimensions(data)
    if (width, height) != (artifact.width, artifact.height):
        raise PlayerMediaError("artifact_dimensions_mismatch")
    return ResolvedPlayerMedia(
        filename=artifact.filename,
        mime_type=artifact.mime_type,
        data=data,
        sha256=sha256,
        byte_count=len(data),
        width=width,
        height=height,
    )


def webp_dimensions(data: bytes) -> tuple[int, int]:
    """Read static WebP dimensions without importing the GPU worker's Pillow."""

    if len(data) < 20 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise PlayerMediaError("invalid_webp")
    declared_size = int.from_bytes(data[4:8], "little") + 8
    if declared_size != len(data):
        raise PlayerMediaError("invalid_webp_size")

    offset = 12
    canvas_dimensions: tuple[int, int] | None = None
    image_dimensions: tuple[int, int] | None = None
    image_chunks = 0
    while offset + 8 <= len(data):
        chunk_type = data[offset:offset + 4]
        chunk_size = int.from_bytes(data[offset + 4:offset + 8], "little")
        payload_start = offset + 8
        payload_end = payload_start + chunk_size
        if payload_end > len(data):
            raise PlayerMediaError("invalid_webp_chunk")
        payload = data[payload_start:payload_end]

        if chunk_type == b"VP8X":
            if len(payload) < 10:
                raise PlayerMediaError("invalid_webp_vp8x")
            flags = payload[0]
            if flags & 0x02:
                raise PlayerMediaError("animated_webp_not_allowed")
            if flags & (0x20 | 0x08 | 0x04):
                raise PlayerMediaError("webp_metadata_not_allowed")
            width = 1 + int.from_bytes(payload[4:7], "little")
            height = 1 + int.from_bytes(payload[7:10], "little")
            canvas_dimensions = (width, height)
        elif chunk_type in {b"ANIM", b"ANMF"}:
            raise PlayerMediaError("animated_webp_not_allowed")
        elif chunk_type in {b"EXIF", b"XMP ", b"ICCP"}:
            raise PlayerMediaError("webp_metadata_not_allowed")
        elif chunk_type == b"VP8L":
            if len(payload) < 5 or payload[0] != 0x2F:
                raise PlayerMediaError("invalid_webp_vp8l")
            b1, b2, b3, b4 = payload[1:5]
            width = 1 + b1 + ((b2 & 0x3F) << 8)
            height = 1 + ((b2 & 0xC0) >> 6) + (b3 << 2) + ((b4 & 0x0F) << 10)
            image_dimensions = (width, height)
            image_chunks += 1
        elif chunk_type == b"VP8 ":
            if len(payload) < 10 or payload[3:6] != b"\x9d\x01\x2a":
                raise PlayerMediaError("invalid_webp_vp8")
            width = int.from_bytes(payload[6:8], "little") & 0x3FFF
            height = int.from_bytes(payload[8:10], "little") & 0x3FFF
            if width < 1 or height < 1:
                raise PlayerMediaError("invalid_webp_dimensions")
            image_dimensions = (width, height)
            image_chunks += 1
        offset = payload_end + (chunk_size & 1)

    if offset != len(data):
        raise PlayerMediaError("invalid_webp_trailing_bytes")
    if image_chunks != 1 or image_dimensions is None:
        raise PlayerMediaError("missing_or_multiple_webp_image_chunks")
    if canvas_dimensions is not None and canvas_dimensions != image_dimensions:
        raise PlayerMediaError("webp_canvas_dimensions_mismatch")
    return image_dimensions


def _read_limited(path: Path, *, max_bytes: int) -> bytes:
    if max_bytes < 1:
        raise PlayerMediaError("invalid_byte_limit")
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except FileNotFoundError as exc:
        raise PlayerMediaError("missing_artifact") from exc
    if len(data) > max_bytes:
        raise PlayerMediaError("artifact_too_large")
    if not data:
        raise PlayerMediaError("empty_artifact")
    return data


def _chmod_private(path: Path, *, directory: bool) -> None:
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError:
        # Permission semantics can be limited on mounted WSL filesystems.
        return
