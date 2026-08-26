from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranscriptEntry(BaseModel):
    user: str
    assistant: str


class NarratorFinalOutput(BaseModel):
    """Produced by Narrator Phase 2. LLM output target.

    The model emits a candidate passage plus its delivery judgment. The engine
    constructs transcript entries from the real player input and commits only
    passages whose handoff is accepted.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    handoff: Literal["render", "continue"]
    handoff_reason: str = Field(min_length=1, max_length=500)
    final_text: str

    @model_validator(mode="after")
    def _require_accepted_render_text(self) -> "NarratorFinalOutput":
        if self.handoff == "render" and not self.final_text:
            raise ValueError("final_text must be non-empty when handoff='render'")
        return self


class VisualNovelPage(BaseModel):
    """One player-visible ADV page authored by the narrator.

    Page boundaries and speaker labels are semantic presentation data. Image
    paths, stage references, layout, and controls remain engine-owned.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["narration", "dialogue"]
    speaker: str = Field(default="", max_length=80)
    text: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def _validate_speaker(self) -> "VisualNovelPage":
        if self.kind == "dialogue" and not self.speaker:
            raise ValueError("dialogue pages require a speaker label")
        if self.kind == "narration":
            self.speaker = ""
        return self


class VisualNovelNarratorOutput(BaseModel):
    """Provider-facing visual-novel narrator contract."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    handoff: Literal["render", "continue"]
    handoff_reason: str = Field(min_length=1, max_length=500)
    pages: list[VisualNovelPage] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def _validate_handoff_pages(self) -> "VisualNovelNarratorOutput":
        if self.handoff == "render" and not self.pages:
            raise ValueError("pages must be non-empty when handoff='render'")
        if self.handoff == "continue" and self.pages:
            raise ValueError("pages must be empty when handoff='continue'")
        return self


NarratorOutput = NarratorFinalOutput | VisualNovelNarratorOutput


_UNDERSCORE_SOURCE_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+(?![A-Za-z0-9_])"
)


def visual_novel_text_contains_source_identifiers(
    value: str,
    *,
    source_ids: Sequence[str] = (),
) -> bool:
    """Return whether one player-visible ADV field exposes an engine id."""

    if _UNDERSCORE_SOURCE_IDENTIFIER_RE.search(value):
        return True
    return any(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(source_id)}(?![A-Za-z0-9_])",
            value,
        )
        for source_id in dict.fromkeys(source_ids)
        if source_id
    )


def visual_novel_pages_contain_source_identifiers(
    pages: Sequence[VisualNovelPage],
    *,
    source_ids: Sequence[str] = (),
) -> bool:
    """Return whether player-visible ADV fields expose engine identifiers.

    Exact source ids are matched case-sensitively at identifier boundaries so
    a display name such as ``Pip`` remains distinct from the source id ``pip``.
    Underscore-form tokens are always source-shaped and do not need a roster
    match to be rejected.
    """

    for page in pages:
        for value in (page.speaker, page.text):
            if visual_novel_text_contains_source_identifiers(
                value,
                source_ids=source_ids,
            ):
                return True
    return False


def narrator_plain_text(result: NarratorOutput) -> str:
    """Deterministic accessible projection of either narrator contract."""

    if isinstance(result, NarratorFinalOutput):
        return result.final_text
    return visual_novel_pages_plain_text(result.pages)


def visual_novel_pages_plain_text(
    pages: Sequence[VisualNovelPage],
) -> str:
    """Project validated ADV pages into the shared text history surface."""

    paragraphs: list[str] = []
    for page in pages:
        if page.kind == "dialogue":
            paragraphs.append(f"{page.speaker}: {page.text}")
        else:
            paragraphs.append(page.text)
    return "\n\n".join(paragraphs)
