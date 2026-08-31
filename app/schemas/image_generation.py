from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.image_director import ImageDirectionKind, ImageGenerationMode
IMAGE_JOB_SCHEMA_VERSION = "13"


class ImageGenerationStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class ImageDeliveryStatus(str, Enum):
    pending = "pending"
    delivering = "delivering"
    delivered = "delivered"
    cancelled = "cancelled"


class ImageDeliveryKind(str, Enum):
    discord = "discord"
    cli = "cli"


class IdentityReferenceStatus(str, Enum):
    provisional = "provisional"
    locked = "locked"
    retained = "retained"
    retired = "retired"


class FrozenReferenceInput(BaseModel):
    """Hash-pinned private runtime visual reference."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str
    sha256: str
    mime_type: str
    width: int
    height: int
    byte_count: int
    relative_path: str
    allowed_root: str

    @model_validator(mode="after")
    def _validate_reference(self) -> "FrozenReferenceInput":
        self.reference_id = self.reference_id.strip()
        self.sha256 = self.sha256.strip().lower()
        self.mime_type = self.mime_type.strip().lower()
        self.relative_path = self.relative_path.strip().replace("\\", "/")
        self.allowed_root = self.allowed_root.strip()
        if not self.reference_id:
            raise ValueError("reference_id must not be empty")
        if len(self.sha256) != 64:
            raise ValueError("reference sha256 must contain 64 hex characters")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("reference sha256 must be hexadecimal") from exc
        relative = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError("reference relative_path must stay inside its root")
        if not self.allowed_root:
            raise ValueError("reference allowed_root must not be empty")
        if self.mime_type not in {
            "image/gif",
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            raise ValueError("unsupported reference MIME type")
        if self.width < 1 or self.height < 1 or self.byte_count < 1:
            raise ValueError("reference dimensions and byte_count must be positive")
        return self


class ImageGenerationRequest(BaseModel):
    """One event-provenance diffusion request with no delivery metadata."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = IMAGE_JOB_SCHEMA_VERSION
    session_id: str
    transaction_id: str
    source_event_id: str
    source_event_fingerprint: str
    source_event_sequence: int
    source_turn_index: int
    request_ordinal: int
    kind: ImageDirectionKind
    generation_mode: ImageGenerationMode = "compose"
    title: str
    subject_character_ids: list[str]
    prompt: str
    prompt_sha256: str
    model_id: str
    model_revision: str
    width: int
    height: int
    steps: int = 4
    guidance: float = 1.0
    seed: int
    dedupe_key: str
    reference_inputs: list[FrozenReferenceInput] = Field(default_factory=list)
    reroll_of_reference_id: str = ""
    sprite_pack_id: str = ""
    sprite_variant_key: str = ""
    sprite_variant_direction: str = ""
    sprite_generation_round: int = 0
    sprite_character_description: str = ""
    sprite_visual_style: str = ""
    sprite_source_facing: Literal["", "left", "right"] = ""

    @model_validator(mode="after")
    def _validate_request(self) -> "ImageGenerationRequest":
        if self.schema_version != IMAGE_JOB_SCHEMA_VERSION:
            raise ValueError("unsupported image request schema version")
        if any(
            reference.mime_type == "image/gif"
            for reference in self.reference_inputs
        ):
            raise ValueError(
                "animated visual references cannot be image-generation inputs"
            )
        for field_name in (
            "session_id",
            "transaction_id",
            "source_event_id",
            "source_event_fingerprint",
            "title",
            "prompt",
            "prompt_sha256",
            "model_id",
            "model_revision",
            "dedupe_key",
        ):
            value = str(getattr(self, field_name, "") or "").strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            setattr(self, field_name, value)
        if len(self.title) > 80:
            raise ValueError("title must be 80 characters or fewer")
        for hash_field in ("source_event_fingerprint", "prompt_sha256", "dedupe_key"):
            value = str(getattr(self, hash_field))
            if len(value) != 64:
                raise ValueError(f"{hash_field} must contain 64 hex characters")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"{hash_field} must be hexadecimal") from exc
        self.subject_character_ids = [
            character_id.strip()
            for character_id in dict.fromkeys(self.subject_character_ids)
            if character_id.strip()
        ]
        self.reroll_of_reference_id = self.reroll_of_reference_id.strip()
        self.sprite_pack_id = self.sprite_pack_id.strip()
        self.sprite_variant_key = self.sprite_variant_key.strip().lower()
        self.sprite_variant_direction = " ".join(
            self.sprite_variant_direction.split()
        ).strip()
        self.sprite_character_description = (
            " ".join(self.sprite_character_description.split()).strip()
        )
        self.sprite_visual_style = " ".join(
            self.sprite_visual_style.split()
        ).strip()
        sprite_fields = (
            bool(self.sprite_pack_id),
            bool(self.sprite_variant_key),
            bool(self.sprite_variant_direction),
            bool(self.sprite_character_description),
            bool(self.sprite_source_facing),
        )
        if any(sprite_fields) != all(sprite_fields):
            raise ValueError("sprite generation metadata must be all-or-none")
        if self.sprite_pack_id:
            if not re.fullmatch(
                r"[a-z0-9][a-z0-9_.-]{0,79}",
                self.sprite_variant_key,
            ):
                raise ValueError("sprite variant key is invalid")
            if len(self.sprite_variant_direction) > 200:
                raise ValueError("sprite variant direction exceeds limit")
            if (
                self.kind != "portrait"
                or self.generation_mode != "compose"
                or len(self.subject_character_ids) != 1
                or (self.width, self.height) != (1024, 1536)
                or self.reroll_of_reference_id
            ):
                raise ValueError(
                    "sprite generation requires one-subject 1024x1536 compose"
                )
            if len(self.sprite_character_description) > 3_000:
                raise ValueError("sprite character description exceeds limit")
            if len(self.sprite_visual_style) > 800:
                raise ValueError("sprite visual style exceeds limit")
        elif self.sprite_generation_round:
            raise ValueError("sprite generation round requires sprite metadata")
        if self.sprite_generation_round < 0:
            raise ValueError("sprite generation round must be non-negative")
        if self.source_event_sequence < 0 or self.source_turn_index < 0:
            raise ValueError("source sequence and turn must be non-negative")
        if self.request_ordinal < 0:
            raise ValueError("request_ordinal must be non-negative")
        if self.width < 256 or self.height < 256:
            raise ValueError("image dimensions must each be at least 256")
        if self.width % 16 or self.height % 16:
            raise ValueError("image dimensions must be multiples of 16")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        return self


class GeneratedImageArtifact(BaseModel):
    """Validated, content-addressed runtime image metadata.

    `relative_path` is private runtime state. It never belongs in a
    TurnResponse, checkpoint, prompt, Discord filename, or player-facing log.
    """

    sha256: str
    relative_path: str
    mime_type: str = "image/webp"
    width: int
    height: int
    byte_count: int

    @model_validator(mode="after")
    def _validate_artifact(self) -> "GeneratedImageArtifact":
        self.sha256 = self.sha256.strip().lower()
        self.relative_path = self.relative_path.strip()
        self.mime_type = self.mime_type.strip().lower()
        if len(self.sha256) != 64:
            raise ValueError("artifact sha256 must contain 64 hex characters")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("artifact sha256 must be hexadecimal") from exc
        if not self.relative_path or self.relative_path.startswith(("/", "\\")):
            raise ValueError("artifact relative_path must be relative")
        if ".." in self.relative_path.replace("\\", "/").split("/"):
            raise ValueError("artifact relative_path may not escape its root")
        if self.mime_type != "image/webp":
            raise ValueError("generated artifacts must be static WebP images")
        if self.width < 1 or self.height < 1 or self.byte_count < 1:
            raise ValueError("artifact dimensions and byte_count must be positive")
        return self

    @property
    def filename(self) -> str:
        return f"illustration-{self.sha256[:16]}.webp"


class ImageGenerationJob(BaseModel):
    job_id: str
    request: ImageGenerationRequest
    status: ImageGenerationStatus = ImageGenerationStatus.queued
    artifact: GeneratedImageArtifact | None = None
    error_code: str = ""
    attempts: int = 0
    created_at: float
    updated_at: float
    started_at: float | None = None
    completed_at: float | None = None

    @model_validator(mode="after")
    def _validate_job(self) -> "ImageGenerationJob":
        self.job_id = self.job_id.strip()
        self.error_code = self.error_code.strip()
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")
        if self.status == ImageGenerationStatus.succeeded and self.artifact is None:
            raise ValueError(f"{self.status.value} jobs require an artifact")
        return self


class ImageDelivery(BaseModel):
    delivery_id: str
    job_id: str
    session_id: str
    source_turn_index: int
    pov_character_id: str
    delivery_kind: ImageDeliveryKind
    delivery: dict[str, Any] = Field(default_factory=dict)
    status: ImageDeliveryStatus = ImageDeliveryStatus.pending
    attempts: int = 0
    created_at: float
    updated_at: float
    delivered_at: float | None = None


class IdentityReferenceCandidate(BaseModel):
    candidate_id: str
    session_id: str
    character_id: str
    job_id: str
    artifact: GeneratedImageArtifact
    status: IdentityReferenceStatus
    active: bool
    reminder_required: bool
    reroll_of_reference_id: str = ""
    created_at: float
    updated_at: float


class ImageWorkerResult(BaseModel):
    """One response emitted by the isolated JSONL worker."""

    ok: bool
    error_code: str = ""
    sha256: str = ""
    mime_type: str = ""
    width: int = 0
    height: int = 0
    byte_count: int = 0
    generation_seconds: float = 0.0
