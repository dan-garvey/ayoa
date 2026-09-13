from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class ObservableFact(BaseModel):
    """Exactly the information available to the explicitly named recipients.

    Visual subjects are embodied characters pictured in this fact, not merely
    mentioned or heard. Every recipient receives the same text and visual scope.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    visible_to: list[str]
    visual_subject_ids: list[str]
    at_offset_s: int
    duration_s: int

    @model_validator(mode="before")
    @classmethod
    def _fill_timing(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        value.setdefault("visual_subject_ids", [])
        value.setdefault("at_offset_s", 0)
        value.setdefault("duration_s", 0)
        return value

    @classmethod
    def only(
        cls,
        text: str,
        visible_to: Iterable[str],
        *,
        visual_subject_ids: Iterable[str] = (),
        at_offset_s: int = 0,
        duration_s: int = 0,
    ) -> "ObservableFact":
        ids = [cid for cid in visible_to if cid]
        return cls(
            text=text,
            visible_to=ids,
            visual_subject_ids=list(visual_subject_ids),
            at_offset_s=at_offset_s,
            duration_s=duration_s,
        )

    @model_validator(mode="after")
    def _validate_visibility(self) -> "ObservableFact":
        self.text = (self.text or "").strip()
        self.visible_to = list(dict.fromkeys(
            cid.strip() for cid in self.visible_to if cid.strip()
        ))
        self.visual_subject_ids = list(
            dict.fromkeys(cid.strip() for cid in self.visual_subject_ids if cid.strip())
        )
        if self.at_offset_s < 0:
            self.at_offset_s = 0
        if self.duration_s < 0:
            self.duration_s = 0
        if not self.visible_to:
            raise ValueError("ObservableFact requires explicit visible_to recipients")
        return self

    def is_visible_to(self, character_id: str) -> bool:
        return character_id in self.visible_to

    def __str__(self) -> str:
        return self.text

    def __contains__(self, needle: object) -> bool:
        return isinstance(needle, str) and needle in self.text

    def strip(self) -> str:
        return self.text.strip()


def fact_recipient_ids(facts: Iterable[ObservableFact]) -> list[str]:
    """Derive the recipients in first-fact order; never widen any fact."""
    return list(dict.fromkeys(cid for fact in facts for cid in fact.visible_to))


def visible_fact_texts(
    facts: Iterable[ObservableFact],
    character_id: str,
) -> list[str]:
    return [
        fact.text.strip() for fact in facts
        if fact.is_visible_to(character_id) and fact.text.strip()
    ]


def visible_visual_subject_ids(
    facts: Iterable[ObservableFact],
    character_id: str,
) -> set[str]:
    return {
        subject for fact in facts if fact.is_visible_to(character_id)
        for subject in fact.visual_subject_ids
    }
