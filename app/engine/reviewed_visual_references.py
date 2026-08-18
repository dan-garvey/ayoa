from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.schemas.checkpoint import CheckpointFile
from app.schemas.image_generation import FrozenReferenceInput
from app.schemas.visual_references import ReviewedVisualReference


STORY_VISUAL_REFERENCE_DIR = "visual-references"
MAX_REVIEWED_REFERENCE_COUNT = 128
MAX_REVIEWED_REFERENCE_BYTES = 20_000_000
MAX_REVIEWED_REFERENCE_TOTAL_BYTES = 256_000_000
MAX_REVIEWED_REFERENCE_EDGE = 8_192
MAX_REVIEWED_REFERENCE_PIXELS = 40_000_000

_FORMAT_BY_MIME = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_EXTENSIONS_BY_MIME = {
    "image/jpeg": frozenset((".jpg", ".jpeg")),
    "image/png": frozenset((".png",)),
    "image/webp": frozenset((".webp",)),
}
_CANONICAL_EXTENSION_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class ReviewedVisualReferenceError(ValueError):
    def __init__(self, code: str, *, reference_id: str = "") -> None:
        self.code = code
        self.reference_id = reference_id
        suffix = f" ({reference_id})" if reference_id else ""
        super().__init__(f"Reviewed visual reference failed: {code}{suffix}.")


@dataclass(frozen=True)
class _ValidatedReference:
    metadata: ReviewedVisualReference
    data: bytes


def validate_story_visual_references(
    checkpoint: CheckpointFile,
    *,
    story_dir: str | Path,
) -> None:
    """Validate every authored registry file under the story-private root."""

    _validate_story_registry(checkpoint, story_dir=Path(story_dir))


def freeze_story_visual_references(
    checkpoint: CheckpointFile,
    *,
    story_dir: str | Path,
    runtime_root: str | Path,
) -> dict[str, FrozenReferenceInput]:
    """Validate authored files and freeze selected inputs content-addressably."""

    validated = _validate_story_registry(
        checkpoint,
        story_dir=Path(story_dir),
    )
    selected = _selected_reviewed_reference_ids(checkpoint, strict_story=True)
    return {
        reference_id: _freeze_reference(
            validated[reference_id],
            runtime_root=Path(runtime_root),
        )
        for reference_id in selected
    }


def load_frozen_visual_references(
    checkpoint: CheckpointFile,
    *,
    runtime_root: str | Path,
) -> dict[str, FrozenReferenceInput]:
    """Resolve and revalidate selected immutable inputs for a live session."""

    _validate_registry_limits(checkpoint.reviewed_visual_references)
    by_id = {
        reference.reference_id: reference
        for reference in checkpoint.reviewed_visual_references
    }
    selected = _selected_reviewed_reference_ids(checkpoint, strict_story=False)
    resolved: dict[str, FrozenReferenceInput] = {}
    total = 0
    for reference_id in selected:
        metadata = by_id[reference_id]
        path, relative_path = _frozen_path(
            metadata,
            runtime_root=Path(runtime_root),
        )
        data = _read_exact_reference(path, metadata)
        _validate_image_bytes(data, metadata)
        total += len(data)
        if total > MAX_REVIEWED_REFERENCE_TOTAL_BYTES:
            raise ReviewedVisualReferenceError(
                "selected_reference_total_too_large",
                reference_id=reference_id,
            )
        resolved[reference_id] = FrozenReferenceInput(
            reference_id=reference_id,
            sha256=metadata.sha256,
            mime_type=metadata.mime_type,
            width=metadata.width,
            height=metadata.height,
            byte_count=metadata.byte_count,
            relative_path=relative_path,
            allowed_root="artifacts",
        )
    return resolved


def _validate_story_registry(
    checkpoint: CheckpointFile,
    *,
    story_dir: Path,
) -> dict[str, _ValidatedReference]:
    references = checkpoint.reviewed_visual_references
    _validate_registry_limits(references)

    _selected_reviewed_reference_ids(checkpoint, strict_story=True)
    if not references:
        return {}

    story_root = story_dir.resolve()
    authored_root_path = story_dir / STORY_VISUAL_REFERENCE_DIR
    try:
        if authored_root_path.is_symlink():
            raise ValueError("visual reference root may not be a symlink")
        authored_root = authored_root_path.resolve(strict=True)
        authored_root.relative_to(story_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ReviewedVisualReferenceError(
            "story_visual_reference_root_unavailable"
        ) from exc
    if not authored_root.is_dir():
        raise ReviewedVisualReferenceError("story_visual_reference_root_unavailable")

    result: dict[str, _ValidatedReference] = {}
    total = 0
    for metadata in references:
        if (
            Path(metadata.storage_ref).suffix.lower()
            not in _EXTENSIONS_BY_MIME[metadata.mime_type]
        ):
            raise ReviewedVisualReferenceError(
                "reference_extension_mismatch",
                reference_id=metadata.reference_id,
            )
        try:
            candidate = authored_root / metadata.storage_ref
            current = authored_root
            for part in Path(metadata.storage_ref).parts:
                current /= part
                if current.is_symlink():
                    raise ValueError("visual reference path may not contain a symlink")
            path = candidate.resolve(strict=True)
            path.relative_to(authored_root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ReviewedVisualReferenceError(
                "reference_path_unavailable_or_unsafe",
                reference_id=metadata.reference_id,
            ) from exc
        if not path.is_file():
            raise ReviewedVisualReferenceError(
                "reference_path_unavailable_or_unsafe",
                reference_id=metadata.reference_id,
            )
        data = _read_exact_reference(path, metadata)
        _validate_image_bytes(data, metadata)
        total += len(data)
        if total > MAX_REVIEWED_REFERENCE_TOTAL_BYTES:
            raise ReviewedVisualReferenceError(
                "reference_registry_too_large",
                reference_id=metadata.reference_id,
            )
        result[metadata.reference_id] = _ValidatedReference(
            metadata=metadata,
            data=data,
        )
    return result


def _validate_registry_limits(
    references: list[ReviewedVisualReference],
) -> None:
    if len(references) > MAX_REVIEWED_REFERENCE_COUNT:
        raise ReviewedVisualReferenceError("reference_count_exceeded")
    declared_total = 0
    for metadata in references:
        if (
            metadata.width > MAX_REVIEWED_REFERENCE_EDGE
            or metadata.height > MAX_REVIEWED_REFERENCE_EDGE
            or metadata.width * metadata.height > MAX_REVIEWED_REFERENCE_PIXELS
        ):
            raise ReviewedVisualReferenceError(
                "reference_dimensions_exceeded",
                reference_id=metadata.reference_id,
            )
        if metadata.byte_count > MAX_REVIEWED_REFERENCE_BYTES:
            raise ReviewedVisualReferenceError(
                "reference_too_large",
                reference_id=metadata.reference_id,
            )
        declared_total += metadata.byte_count
        if declared_total > MAX_REVIEWED_REFERENCE_TOTAL_BYTES:
            raise ReviewedVisualReferenceError(
                "reference_registry_too_large",
                reference_id=metadata.reference_id,
            )


def _selected_reviewed_reference_ids(
    checkpoint: CheckpointFile,
    *,
    strict_story: bool,
) -> list[str]:
    by_id = {
        reference.reference_id: reference
        for reference in checkpoint.reviewed_visual_references
    }
    selected: list[str] = []
    for character in checkpoint.characters:
        reference_id = character.visuals.identity_reference_id.strip()
        if not reference_id:
            continue
        metadata = by_id.get(reference_id)
        if metadata is None:
            if strict_story or not reference_id.startswith("imgref_"):
                raise ReviewedVisualReferenceError(
                    "selected_identity_reference_missing",
                    reference_id=reference_id,
                )
            continue
        if (
            metadata.purpose != "identity"
            or metadata.scope != "character"
            or not metadata.diffusion_authorized
        ):
            raise ReviewedVisualReferenceError(
                "selected_identity_reference_unauthorized",
                reference_id=reference_id,
            )
        selected.append(reference_id)

    for reference_ids in checkpoint.location_visual_reference_ids.values():
        for reference_id in reference_ids:
            metadata = by_id.get(reference_id)
            if metadata is None:
                raise ReviewedVisualReferenceError(
                    "selected_location_reference_missing",
                    reference_id=reference_id,
                )
            if (
                metadata.purpose not in {"environment", "style"}
                or metadata.scope != "location"
                or not metadata.diffusion_authorized
            ):
                raise ReviewedVisualReferenceError(
                    "selected_location_reference_unauthorized",
                    reference_id=reference_id,
                )
            selected.append(reference_id)
    return list(dict.fromkeys(selected))


def _read_exact_reference(
    path: Path,
    metadata: ReviewedVisualReference,
) -> bytes:
    try:
        with path.open("rb") as handle:
            data = handle.read(metadata.byte_count + 1)
    except OSError as exc:
        raise ReviewedVisualReferenceError(
            "reference_unavailable",
            reference_id=metadata.reference_id,
        ) from exc
    if len(data) != metadata.byte_count:
        raise ReviewedVisualReferenceError(
            "reference_byte_count_mismatch",
            reference_id=metadata.reference_id,
        )
    if hashlib.sha256(data).hexdigest() != metadata.sha256:
        raise ReviewedVisualReferenceError(
            "reference_hash_mismatch",
            reference_id=metadata.reference_id,
        )
    return data


def _validate_image_bytes(
    data: bytes,
    metadata: ReviewedVisualReference,
) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            actual_format = str(image.format or "").upper()
            if actual_format != _FORMAT_BY_MIME[metadata.mime_type]:
                raise ReviewedVisualReferenceError(
                    "reference_mime_mismatch",
                    reference_id=metadata.reference_id,
                )
            if getattr(image, "n_frames", 1) != 1:
                raise ReviewedVisualReferenceError(
                    "animated_reference_not_allowed",
                    reference_id=metadata.reference_id,
                )
            if image.size != (metadata.width, metadata.height):
                raise ReviewedVisualReferenceError(
                    "reference_dimensions_mismatch",
                    reference_id=metadata.reference_id,
                )
            image.verify()
    except ReviewedVisualReferenceError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ReviewedVisualReferenceError(
            "invalid_static_reference_image",
            reference_id=metadata.reference_id,
        ) from exc


def _freeze_reference(
    reference: _ValidatedReference,
    *,
    runtime_root: Path,
) -> FrozenReferenceInput:
    metadata = reference.metadata
    destination, relative_path = _frozen_path(
        metadata,
        runtime_root=runtime_root,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = runtime_root.resolve()
    for directory in (
        root,
        root / "artifacts",
        root / "artifacts" / "references",
        destination.parent,
    ):
        _chmod_private(directory, directory=True)
    if destination.exists():
        existing = _read_exact_reference(destination, metadata)
        _validate_image_bytes(existing, metadata)
    else:
        fd, temporary = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{metadata.sha256}.",
            suffix=".partial",
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(reference.data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    _chmod_private(destination, directory=False)
    return FrozenReferenceInput(
        reference_id=metadata.reference_id,
        sha256=metadata.sha256,
        mime_type=metadata.mime_type,
        width=metadata.width,
        height=metadata.height,
        byte_count=metadata.byte_count,
        relative_path=relative_path,
        allowed_root="artifacts",
    )


def _frozen_path(
    metadata: ReviewedVisualReference,
    *,
    runtime_root: Path,
) -> tuple[Path, str]:
    root = runtime_root.resolve()
    extension = _CANONICAL_EXTENSION_BY_MIME[metadata.mime_type]
    relative = (
        Path("artifacts")
        / "references"
        / metadata.sha256[:2]
        / f"{metadata.sha256}{extension}"
    )
    path = (root / relative).resolve()
    allowed = (root / "artifacts").resolve()
    try:
        path.relative_to(allowed)
    except ValueError as exc:
        raise ReviewedVisualReferenceError(
            "frozen_reference_path_unsafe",
            reference_id=metadata.reference_id,
        ) from exc
    return path, relative.as_posix()


def _chmod_private(path: Path, *, directory: bool) -> None:
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError:
        pass
