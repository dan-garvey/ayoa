from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ImageDirectionKind = Literal[
    "portrait",
    "group_portrait",
    "action",
    "establishing",
    "detail",
]
ImageGenerationMode = Literal["compose", "edit"]
ImageStageAction = Literal["independent", "reuse", "replace", "clear"]


class ImageDirection(BaseModel):
    """One aesthetic request authored by the image-director model."""

    model_config = ConfigDict(extra="forbid")

    kind: ImageDirectionKind
    title: str
    subject_character_ids: list[str]
    generation_mode: ImageGenerationMode = "compose"
    reference_ids: list[str] = Field(default_factory=list)
    scene_prompt: str

    @model_validator(mode="after")
    def _clean(self) -> "ImageDirection":
        self.title = " ".join(self.title.split()).strip()
        if not self.title:
            raise ValueError("title must not be empty")
        if len(self.title) > 80:
            raise ValueError("title must be 80 characters or fewer")
        self.subject_character_ids = [
            character_id.strip()
            for character_id in dict.fromkeys(self.subject_character_ids)
            if character_id.strip()
        ]
        self.reference_ids = [
            reference_id.strip()
            for reference_id in dict.fromkeys(self.reference_ids)
            if reference_id.strip()
        ]
        self.scene_prompt = " ".join(self.scene_prompt.split()).strip()
        if not self.scene_prompt:
            raise ValueError("scene_prompt must not be empty")
        return self


class ImageDirectorOutput(BaseModel):
    """Strict provider-facing contract for stage selection or generation."""

    model_config = ConfigDict(extra="forbid")

    # ``independent`` is retained for explicit identity-review jobs that are
    # not a story stage. Visual-novel render direction must choose one of the
    # other three transitions.
    stage_action: ImageStageAction = "independent"
    # A visual-novel replacement may select one reviewed environment plate as
    # the exact stage instead of requesting generation.  This remains an
    # opaque, text-only handle in model context; private bytes and provenance
    # are resolved only after output validation.
    stage_reference_id: str = ""
    requests: list[ImageDirection]

    @model_validator(mode="after")
    def _clean(self) -> "ImageDirectorOutput":
        self.stage_reference_id = self.stage_reference_id.strip()
        return self
