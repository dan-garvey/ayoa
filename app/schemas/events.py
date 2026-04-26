from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class WorldAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feasible: bool

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_audit_fields(cls, value: Any) -> Any:
        """Drop retired audit/framing fields from old checkpoints.

        Existing checkpoints may still carry `attempted_action` or the
        older `resolved_outcome` audit-only field.
        Drop it before extra-field validation so old saves load and
        rewrite cleanly under the simplified schema.
        """
        if isinstance(value, dict) and (
            "attempted_action" in value or "resolved_outcome" in value
        ):
            value = dict(value)
            value.pop("attempted_action", None)
            value.pop("resolved_outcome", None)
        return value


class SceneDelta(BaseModel):
    """Per-event time delta. Only carries time advancement; all
    character relocation flows through `EventRouterOutput.roster_moves`
    (see `Orchestrator._apply_roster_moves`).

    All fields REQUIRED — defaults expand the API's grammar compiler
    past the "Schema is too complex" ceiling when this nests inside
    EventRouterOutput. LLM emits 0 for absent values."""

    model_config = ConfigDict(extra="forbid")

    time_advanced_seconds: int


class ObservableFact(BaseModel):
    """One surface fact plus its fact-level visibility.

    `observation_level` on an observer answers how a character perceived
    the event as a whole (direct / indirect / inferred). This object
    answers a different question: which concrete facts in that event
    were available to that character at all.

    Schema fields are all required for structured-output stability.
    Legacy checkpoints/tests that still store bare strings are upgraded
    by `CanonicalEvent._coerce_legacy_facts`.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    audience: Literal["all_observers", "only"]
    visible_to: list[str]

    @classmethod
    def all(cls, text: str) -> "ObservableFact":
        return cls(text=text, audience="all_observers", visible_to=[])

    @classmethod
    def only(cls, text: str, visible_to: Iterable[str]) -> "ObservableFact":
        ids = [cid for cid in visible_to if cid]
        return cls(text=text, audience="only", visible_to=ids)

    @model_validator(mode="after")
    def _validate_visibility(self) -> "ObservableFact":
        self.text = (self.text or "").strip()
        self.visible_to = [cid.strip() for cid in self.visible_to if cid.strip()]
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
) -> list[str]:
    """Return fact text visible to `character_id`.

    Empty `character_id` is used by legacy/debug formatting paths and
    returns only facts addressed to all observers.
    """
    visible: list[str] = []
    for fact in facts:
        if isinstance(fact, str):
            text = fact.strip()
            if text:
                visible.append(text)
            continue
        if fact.audience == "all_observers" or (
            character_id and fact.is_visible_to(character_id)
        ):
            text = fact.text.strip()
            if text:
                visible.append(text)
    return visible


class CanonicalEvent(BaseModel):
    """Produced by the event router's adjudication pass. LLM output target.

    `user_intent`, `world_adjudication.attempted_action`, and
    `event_id` were dropped; the canonical event now carries only
    feasibility, time delta, and observable facts. The orchestrator tags
    visibility logs directly from turn_index so there's no need for the
    router to emit an event id inside this nested object.

    All fields REQUIRED — see EventRouterOutput docstring for the
    "Schema is too complex" rationale."""

    model_config = ConfigDict(extra="forbid")

    world_adjudication: WorldAdjudication
    scene_delta: SceneDelta
    observable_facts: list[ObservableFact]

    @field_validator("observable_facts", mode="before")
    @classmethod
    def _coerce_legacy_facts(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        upgraded = []
        for item in value:
            if isinstance(item, str):
                upgraded.append({
                    "text": item,
                    "audience": "all_observers",
                    "visible_to": [],
                })
            else:
                upgraded.append(item)
        return upgraded
