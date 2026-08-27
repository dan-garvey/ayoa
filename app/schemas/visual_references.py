from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


VisualReferencePurpose = Literal["identity", "environment", "style"]
VisualReferenceScope = Literal["character", "location"]

_REFERENCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_SUPPORTED_MIME_TYPES = frozenset(("image/jpeg", "image/png", "image/webp"))


class ReviewedVisualReference(BaseModel):
    """Authored metadata for one human-reviewed runtime visual input."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str
    storage_ref: str
    mime_type: str
    width: int
    height: int
    byte_count: int
    sha256: str
    purpose: VisualReferencePurpose
    scope: VisualReferenceScope
    scope_id: str
    selection_hint: str
    diffusion_authorized: bool = False

    @model_validator(mode="after")
    def _validate_reference(self) -> "ReviewedVisualReference":
        self.reference_id = self.reference_id.strip()
        self.storage_ref = self.storage_ref.strip().replace("\\", "/")
        self.mime_type = self.mime_type.strip().lower()
        self.sha256 = self.sha256.strip().lower()
        self.scope_id = " ".join(self.scope_id.split()).strip()
        self.selection_hint = " ".join(self.selection_hint.split()).strip()

        if not _REFERENCE_ID_RE.fullmatch(self.reference_id):
            raise ValueError(
                "reviewed visual reference_id must be an opaque identifier"
            )
        if self.reference_id.startswith("imgref_"):
            raise ValueError(
                "reviewed visual reference_id uses reserved generated prefix"
            )
        relative = PurePosixPath(self.storage_ref)
        if (
            not self.storage_ref
            or len(self.storage_ref) > 500
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in self.storage_ref
            )
            or relative.is_absolute()
            or "." in relative.parts
            or ".." in relative.parts
        ):
            raise ValueError(
                "reviewed visual storage_ref must stay inside its story root"
            )
        if self.mime_type not in _SUPPORTED_MIME_TYPES:
            raise ValueError("unsupported reviewed visual MIME type")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError(
                "reviewed visual sha256 must contain 64 hexadecimal characters"
            )
        if self.width < 1 or self.height < 1 or self.byte_count < 1:
            raise ValueError(
                "reviewed visual dimensions and byte_count must be positive"
            )
        if self.purpose == "identity" and self.scope != "character":
            raise ValueError("identity references require character scope")
        if self.purpose in {"environment", "style"} and self.scope != "location":
            raise ValueError("environment and style references require location scope")
        if not self.scope_id or len(self.scope_id) > 200:
            raise ValueError("reviewed visual scope_id must be a bounded identifier")
        if not self.selection_hint or len(self.selection_hint) > 500:
            raise ValueError(
                "reviewed visual selection_hint must be 1-500 characters"
            )
        return self
