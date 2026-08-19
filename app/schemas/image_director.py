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
    """Strict provider-facing contract; an empty list means no illustration."""

    model_config = ConfigDict(extra="forbid")

    requests: list[ImageDirection]
