from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


IMAGE_JOB_SCHEMA_VERSION = "1"


class ImageGenerationStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    delivering = "delivering"
    delivered = "delivered"
    failed = "failed"
    cancelled = "cancelled"


class ImageDeliveryKind(str, Enum):
    discord = "discord"
    cli = "cli"


class ImageTriggerKind(str, Enum):
    act = "act"
    begin = "begin"
    arrival = "arrival"
    roll_resolution = "roll_resolution"
    render_retry = "render_retry"
    query = "query"


class ImageGenerationRequest(BaseModel):
    """Private, durable request for one output-only POV illustration."""

    schema_version: str = IMAGE_JOB_SCHEMA_VERSION
    session_id: str
    checkpoint_id: str
    checkpoint_sha256: str
    turn_index: int
    actor_character_id: str
    trigger_kind: ImageTriggerKind
    prompt: str
    prompt_sha256: str
    model_id: str
    model_revision: str
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance: float = 1.0
    seed: int
    dedupe_key: str
    delivery_kind: ImageDeliveryKind
    delivery: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_request(self) -> "ImageGenerationRequest":
        for field_name in (
            "session_id",
            "checkpoint_id",
            "checkpoint_sha256",
            "actor_character_id",
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
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")
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
    delivered_at: float | None = None

    @model_validator(mode="after")
    def _validate_job(self) -> "ImageGenerationJob":
        self.job_id = self.job_id.strip()
        self.error_code = self.error_code.strip()
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")
        if self.status in {
            ImageGenerationStatus.succeeded,
            ImageGenerationStatus.delivering,
            ImageGenerationStatus.delivered,
        } and self.artifact is None:
            raise ValueError(f"{self.status.value} jobs require an artifact")
        return self


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
