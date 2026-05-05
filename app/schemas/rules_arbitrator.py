from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


AbilityId = Literal["str", "dex", "con", "int", "wis", "cha"]
AdvantageState = Literal["normal", "advantage", "disadvantage"]
RollKind = Literal[
    "ability_check",
    "skill_check",
    "saving_throw",
    "attack_roll",
]


class PlannedRoll(BaseModel):
    """One rules-arbitrator roll request.

    All fields are required to keep structured-output schemas fixed. Empty
    strings and zeroes are used where a field is not applicable.
    """

    model_config = ConfigDict(extra="forbid")

    roll_id: str
    actor_id: str
    kind: RollKind
    ability: AbilityId
    skill: str
    dc: int
    opposed_by: str
    advantage_state: AdvantageState
    reason: str

    @model_validator(mode="after")
    def _clean(self) -> "PlannedRoll":
        self.roll_id = self.roll_id.strip()
        self.actor_id = self.actor_id.strip()
        self.skill = self.skill.strip().lower()
        self.opposed_by = self.opposed_by.strip()
        self.reason = self.reason.strip()
        if not self.roll_id:
            raise ValueError("roll_id is required")
        if not self.actor_id:
            raise ValueError("actor_id is required")
        if self.dc < 0:
            self.dc = 0
        return self


class RollPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_rolls: bool
    roll_requests: list[PlannedRoll]
    no_roll_reason: str


class RulesAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feasible: bool
    mechanical_summary: str
    visible_outcome_facts: list[str]
    state_deltas: list[str]
    rules_notes: list[str]
    fallback_reason: str

    @model_validator(mode="after")
    def _ensure_visible_fact(self) -> "RulesAdjudication":
        self.visible_outcome_facts = [
            fact.strip() for fact in self.visible_outcome_facts if fact.strip()
        ]
        if not self.visible_outcome_facts:
            text = self.mechanical_summary.strip() or self.fallback_reason.strip()
            if text:
                self.visible_outcome_facts = [text]
        if not self.visible_outcome_facts:
            raise ValueError("Rules adjudication requires a visible outcome fact")
        return self
