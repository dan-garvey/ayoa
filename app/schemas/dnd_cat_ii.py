from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.dnd_spatial import DndSpatialDelta
from app.schemas.state import DndEffectRecurringSave, DndRouterObservedFact


AbilityId = Literal["str", "dex", "con", "int", "wis", "cha"]
AdvantageState = Literal["normal", "advantage", "disadvantage"]
CombatStatus = Literal["ongoing", "ended"]
RollKind = Literal[
    "ability_check",
    "skill_check",
    "saving_throw",
    "attack_roll",
    "damage_roll",
]
DamageAdjustmentKind = Literal[
    "resistance",
    "immunity",
    "vulnerability",
    "halve",
    "double",
]
DamageAdjustmentScope = Literal["component", "attack_total"]
SaveDamageOutcome = Literal["none", "half", "full"]


class PlannedDamageAdjustment(BaseModel):
    """Router-authored D&D damage adjustment for a planned attack roll.

    Sheet-derived adjustments are calculated from durable character data in the
    adapter. This field is for situational facts the router can see in the
    current action context, such as a temporary resistance or vulnerability.
    """

    model_config = ConfigDict(extra="forbid")

    kind: DamageAdjustmentKind
    damage_type: str
    reason: str
    scope: DamageAdjustmentScope = "component"

    @model_validator(mode="before")
    @classmethod
    def _coerce_missing_damage_adjustment_fields(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("damage_type", "")
            data.setdefault("reason", "")
            data.setdefault("scope", "component")
        return data

    @model_validator(mode="after")
    def _clean(self) -> "PlannedDamageAdjustment":
        self.kind = self.kind.strip().lower()
        self.damage_type = self.damage_type.strip().lower()
        self.reason = self.reason.strip()
        self.scope = self.scope.strip().lower()  # type: ignore[assignment]
        return self


class PlannedRoll(BaseModel):
    """One D&D Cat II roll request planned by the event-router role.

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
    # Optional adapter metadata. Empty outside D&D combat. `action_id`
    # names the attack/action profile to use for to-hit and damage lookup;
    # `target_id` names the intended target for AC, saving throws, and damage
    # application. For saving throws, actor_id remains the source/caster while
    # target_id is the creature rolling against dc.
    # `effect_id` links code-owned follow-up saves to a sustained effect.
    action_id: str
    target_id: str
    effect_id: str = ""
    modifier_bonus: int = 0
    modifier_bonus_reason: str = ""
    damage_on_save_success: SaveDamageOutcome = "none"
    damage_adjustments: list[PlannedDamageAdjustment]

    @model_validator(mode="before")
    @classmethod
    def _coerce_missing_adapter_fields(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("action_id", "")
            data.setdefault("target_id", "")
            data.setdefault("effect_id", "")
            data.setdefault("modifier_bonus", 0)
            data.setdefault("modifier_bonus_reason", "")
            data.setdefault("damage_on_save_success", "none")
            data.setdefault("damage_adjustments", [])
        return data

    @model_validator(mode="after")
    def _clean(self) -> "PlannedRoll":
        self.roll_id = self.roll_id.strip()
        self.actor_id = self.actor_id.strip()
        self.skill = self.skill.strip().lower()
        self.opposed_by = self.opposed_by.strip()
        self.reason = self.reason.strip()
        self.action_id = self.action_id.strip().lower()
        self.target_id = self.target_id.strip()
        self.effect_id = self.effect_id.strip()
        self.modifier_bonus_reason = self.modifier_bonus_reason.strip()
        self.damage_on_save_success = (
            self.damage_on_save_success.strip().lower()  # type: ignore[assignment]
            or "none"
        )
        if not self.roll_id:
            raise ValueError("roll_id is required")
        if not self.actor_id:
            raise ValueError("actor_id is required")
        if self.dc < 0:
            self.dc = 0
        if self.damage_on_save_success not in {"none", "half", "full"}:
            self.damage_on_save_success = "none"
        return self


class RollPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_rolls: bool
    roll_requests: list[PlannedRoll]
    no_roll_reason: str


CombatStateDeltaKind = Literal[
    "healing",
    "condition_add",
    "condition_remove",
]


class CombatStateDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: CombatStateDeltaKind
    target_id: str
    amount: int
    condition: str
    reason: str

    @model_validator(mode="after")
    def _clean(self) -> "CombatStateDelta":
        self.target_id = self.target_id.strip()
        self.condition = self.condition.strip()
        self.reason = self.reason.strip()
        if not self.target_id:
            raise ValueError("target_id is required")
        if self.amount < 0:
            self.amount = 0
        return self


EffectDeltaOperation = Literal["start", "end", "update"]
EffectDurationKind = Literal[
    "rounds",
    "minutes",
    "hours",
    "days",
    "special",
    "until_removed",
]


class EffectDelta(BaseModel):
    """Router-authored durable D&D effect mutation.

    This is adapter-only state. Use it for sourced/timed/concentration
    effects; keep one-off condition changes in combat_state_deltas.
    """

    model_config = ConfigDict(extra="forbid")

    operation: EffectDeltaOperation
    target_id: str
    effect_id: str = ""
    name: str = ""
    slug: str = ""
    source_type: str = "custom"
    source_id: str = ""
    originator_id: str = ""
    conditions: list[str] = Field(default_factory=list)
    concentration: bool = False
    duration_kind: EffectDurationKind = "until_removed"
    duration_amount: int = 0
    remaining_rounds: int = 0
    duration_text: str = ""
    break_triggers: list[str] = Field(default_factory=list)
    recurring_save: DndEffectRecurringSave | None = None
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_missing_effect_fields(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("effect_id", "")
            data.setdefault("name", "")
            data.setdefault("slug", "")
            data.setdefault("source_type", "custom")
            data.setdefault("source_id", "")
            data.setdefault("originator_id", "")
            data.setdefault("conditions", [])
            data.setdefault("concentration", False)
            data.setdefault("duration_kind", "until_removed")
            data.setdefault("duration_amount", 0)
            data.setdefault("remaining_rounds", 0)
            data.setdefault("duration_text", "")
            data.setdefault("break_triggers", [])
            data.setdefault("recurring_save", None)
            data.setdefault("reason", "")
        return data

    @model_validator(mode="after")
    def _clean(self) -> "EffectDelta":
        self.target_id = self.target_id.strip()
        self.effect_id = self.effect_id.strip()
        self.name = self.name.strip()
        self.slug = self.slug.strip().lower()
        self.source_type = self.source_type.strip().lower() or "custom"
        self.source_id = self.source_id.strip()
        self.originator_id = self.originator_id.strip()
        self.conditions = [
            condition.strip()
            for condition in self.conditions
            if condition.strip()
        ]
        self.break_triggers = [
            trigger.strip().lower()
            for trigger in self.break_triggers
            if trigger.strip()
        ]
        if self.duration_amount < 0:
            self.duration_amount = 0
        if self.remaining_rounds < 0:
            self.remaining_rounds = 0
        self.duration_text = self.duration_text.strip()
        self.reason = self.reason.strip()
        if not self.target_id:
            raise ValueError("target_id is required")
        return self


class RulesAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feasible: bool
    combat_status: CombatStatus
    mechanical_summary: str
    visible_outcome_facts: list[str]
    state_deltas: list[str]
    combat_state_deltas: list[CombatStateDelta] = Field(default_factory=list)
    effect_deltas: list[EffectDelta] = Field(default_factory=list)
    spatial_deltas: list[DndSpatialDelta]
    rules_notes: list[str]
    fallback_reason: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_missing_combat_fields(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("combat_status", "ongoing")
            data.setdefault("combat_state_deltas", [])
            data.setdefault("effect_deltas", [])
            data.setdefault("spatial_deltas", [])
        return data

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


class DndCombatManagerAdjudication(RulesAdjudication):
    """Per-turn D&D initiative adjudication with post-combat continuity facts."""

    router_observed_facts: list[DndRouterObservedFact] = Field(default_factory=list)
