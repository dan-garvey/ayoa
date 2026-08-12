from __future__ import annotations

import hashlib

import pytest

from app.engine.player_media import (
    PlayerMediaError,
    finalize_generated_webp,
    resolve_generated_media,
    webp_dimensions,
)
from app.schemas.image_generation import ImageWorkerResult


def _webp(width: int, height: int) -> bytes:
    encoded_width = width - 1
    encoded_height = height - 1
    payload = b"\x2f" + bytes(
        (
            encoded_width & 0xFF,
            ((encoded_width >> 8) & 0x3F) | ((encoded_height & 0x03) << 6),
            (encoded_height >> 2) & 0xFF,
            (encoded_height >> 10) & 0x0F,
        )
    )
    chunk = b"VP8L" + len(payload).to_bytes(4, "little") + payload + b"\x00"
    body = b"WEBP" + chunk
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def test_generated_media_is_content_addressed_and_reverified(tmp_path):
    data = _webp(1024, 1024)
    sha256 = hashlib.sha256(data).hexdigest()
    temp = tmp_path / "temp.webp"
    temp.write_bytes(data)

    artifact = finalize_generated_webp(
        temp,
        runtime_root=tmp_path / "runtime",
        worker_result=ImageWorkerResult(
            ok=True,
            sha256=sha256,
            mime_type="image/webp",
            width=1024,
            height=1024,
            byte_count=len(data),
        ),
        expected_width=1024,
        expected_height=1024,
    )
    media = resolve_generated_media(
        artifact,
        runtime_root=tmp_path / "runtime",
    )

    assert artifact.relative_path == f"artifacts/{sha256[:2]}/{sha256}.webp"
    assert media.filename == f"illustration-{sha256[:16]}.webp"
    assert media.data == data
    assert media.width == 1024
    assert media.height == 1024


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not-webp",
        b"RIFF\x00\x00\x00\x00WEBP",
        _webp(32, 32) + b"trailing",
    ],
)
def test_malformed_webp_is_rejected(data):
    with pytest.raises(PlayerMediaError):
        webp_dimensions(data)


def test_worker_hash_or_dimensions_mismatch_is_rejected(tmp_path):
    data = _webp(256, 256)
    temp = tmp_path / "temp.webp"
    temp.write_bytes(data)
    result = ImageWorkerResult(
        ok=True,
        sha256="0" * 64,
        mime_type="image/webp",
        width=256,
        height=256,
        byte_count=len(data),
    )

    with pytest.raises(PlayerMediaError, match="worker_hash_mismatch"):
        finalize_generated_webp(
            temp,
            runtime_root=tmp_path / "runtime",
            worker_result=result,
            expected_width=256,
            expected_height=256,
        )


def test_animated_or_metadata_bearing_webp_is_rejected():
    width = (256 - 1).to_bytes(3, "little")
    height = (256 - 1).to_bytes(3, "little")
    vp8x = bytes((0x02, 0, 0, 0)) + width + height
    chunk = b"VP8X" + len(vp8x).to_bytes(4, "little") + vp8x
    body = b"WEBP" + chunk
    data = b"RIFF" + len(body).to_bytes(4, "little") + body

    with pytest.raises(PlayerMediaError, match="animated_webp_not_allowed"):
        webp_dimensions(data)
