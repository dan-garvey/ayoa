from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class WorldAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feasible: bool


class ObservableFact(BaseModel):
    """One surface fact plus its fact-level visibility.

    `observation_level` on an observer answers how a character perceived
    the event as a whole (direct / indirect / inferred). This object
    answers a different question: which concrete facts in that event
    were available to that character at all.

    Schema fields are all required for structured-output stability.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    audience: Literal["all_observers", "only"]
    visible_to: list[str]
    at_offset_s: int
    duration_s: int

    @model_validator(mode="before")
    @classmethod
    def _fill_timing(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        value.setdefault("at_offset_s", 0)
        value.setdefault("duration_s", 0)
        return value

    @classmethod
    def all(
        cls,
        text: str,
        *,
        at_offset_s: int = 0,
        duration_s: int = 0,
    ) -> "ObservableFact":
        return cls(
            text=text,
            audience="all_observers",
            visible_to=[],
            at_offset_s=at_offset_s,
            duration_s=duration_s,
        )

    @classmethod
    def only(
        cls,
        text: str,
        visible_to: Iterable[str],
        *,
        at_offset_s: int = 0,
        duration_s: int = 0,
    ) -> "ObservableFact":
        ids = [cid for cid in visible_to if cid]
        return cls(
            text=text,
            audience="only",
            visible_to=ids,
            at_offset_s=at_offset_s,
            duration_s=duration_s,
        )

    @model_validator(mode="after")
    def _validate_visibility(self) -> "ObservableFact":
        self.text = (self.text or "").strip()
        self.visible_to = [cid.strip() for cid in self.visible_to if cid.strip()]
        if self.at_offset_s < 0:
            self.at_offset_s = 0
        if self.duration_s < 0:
            self.duration_s = 0
        if self.audience == "all_observers":
            self.visible_to = []
        elif not self.visible_to:
            raise ValueError("ObservableFact audience='only' requires visible_to")
        return self

    def is_visible_to(self, character_id: str) -> bool:
        return self.audience == "all_observers" or character_id in self.visible_to

    def __str__(self) -> str:
        return self.text

    def __contains__(self, needle: object) -> bool:
        return isinstance(needle, str) and needle in self.text

    def strip(self) -> str:
        return self.text.strip()


def visible_fact_texts(
    facts: Iterable[ObservableFact | str],
    character_id: str = "",
    *,
    include_all_observers: bool = True,
) -> list[str]:
    """Return fact text visible to `character_id`.

    Empty `character_id` is used by legacy/debug formatting paths and
    returns only facts addressed to all observers.

    `include_all_observers=False` is for mediated observers who are not
    physically in the event location: they receive only facts explicitly
    scoped to them by `visible_to`, not broad room facts.
    """
    visible: list[str] = []
    for fact in facts:
        if isinstance(fact, str):
            if not include_all_observers:
                continue
            text = fact.strip()
            if text:
                visible.append(text)
            continue
        if fact.audience == "all_observers":
            if not include_all_observers:
                continue
        elif not (character_id and fact.is_visible_to(character_id)):
            continue
        text = fact.text.strip()
        if text:
            visible.append(text)
    return visible


class CanonicalEvent(BaseModel):
    """Produced by the event router's adjudication pass. LLM output target.

    `user_intent` and `event_id` were dropped; the canonical event now
    carries only feasibility and observable facts. The orchestrator tags
    visibility logs directly from turn_index so there's no need for the
    router to emit an event id inside this nested object.

    All fields REQUIRED — see EventRouterOutput docstring for the
    "Schema is too complex" rationale."""

    model_config = ConfigDict(extra="forbid")

    world_adjudication: WorldAdjudication
    observable_facts: list[ObservableFact]
