from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.narrator import VisualNovelPage


_OPAQUE_HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_VARIANT_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")


class VisualNovelOnboardingPage(BaseModel):
    """One authored tutorial page and its reviewed sprite selections."""

    model_config = ConfigDict(extra="forbid")

    page: VisualNovelPage
    sprite_variant_keys_by_label: dict[str, str] = Field(
        default_factory=dict,
        max_length=2,
    )

    @model_validator(mode="after")
    def _validate_sprite_variants(self) -> "VisualNovelOnboardingPage":
        cleaned: dict[str, str] = {}
        sprite_labels = set(self.page.sprites)
        for raw_label, raw_key in self.sprite_variant_keys_by_label.items():
            label = " ".join(str(raw_label or "").split()).strip()
            key = str(raw_key or "").strip().lower()
            if label not in sprite_labels:
                raise ValueError(
                    "onboarding sprite variants must target a page sprite label"
                )
            if not _VARIANT_KEY_RE.fullmatch(key):
                raise ValueError(
                    "onboarding sprite variants must use bounded opaque keys"
                )
            cleaned[label] = key
        self.sprite_variant_keys_by_label = cleaned
        return self


class VisualNovelOnboardingJoinChoice(BaseModel):
    """One public button label mapped to an authored playable seat."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(min_length=1, max_length=80)
    character_id: str

    @model_validator(mode="after")
    def _validate_choice(self) -> "VisualNovelOnboardingJoinChoice":
        self.label = " ".join(self.label.split()).strip()
        self.character_id = self.character_id.strip()
        if any(
            ord(character) < 32 or ord(character) == 127 for character in self.label
        ):
            raise ValueError("onboarding join labels cannot contain control characters")
        if not _OPAQUE_HANDLE_RE.fullmatch(self.character_id):
            raise ValueError("onboarding join choices require an opaque character id")
        return self


class VisualNovelOnboarding(BaseModel):
    """Optional deterministic VN introduction authored by a story pack.

    The opaque stage and character ids are runtime-only checkpoint metadata.
    Models receive neither this structure nor the referenced image bytes.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    stage_reference_id: str
    pages: list[VisualNovelOnboardingPage] = Field(min_length=1, max_length=4)
    join_choices: list[VisualNovelOnboardingJoinChoice] = Field(
        min_length=1,
        max_length=5,
    )

    @model_validator(mode="after")
    def _validate_onboarding(self) -> "VisualNovelOnboarding":
        self.stage_reference_id = self.stage_reference_id.strip()
        if not _OPAQUE_HANDLE_RE.fullmatch(self.stage_reference_id):
            raise ValueError("onboarding stage reference must be an opaque handle")
        choice_ids = [choice.character_id for choice in self.join_choices]
        if len(choice_ids) != len(set(choice_ids)):
            raise ValueError("onboarding join choices must target unique characters")
        labels = [choice.label.casefold() for choice in self.join_choices]
        if len(labels) != len(set(labels)):
            raise ValueError("onboarding join choice labels must be unique")
        return self
