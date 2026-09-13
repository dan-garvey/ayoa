"""Strict batched routing drafts and durable canonical event records.

Provider outputs contain semantic decisions only. Runtime identity, absolute
time, scheduling persistence, and delivery state live on separate engine-owned
records so they cannot bloat or confuse the router grammar.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from app.schemas.dnd_inventory import DndLootOfferSignal
from app.schemas.dnd_monsters import DndCombatantSpawn
from app.schemas.dnd_spatial import DndBattleMapSeed
from app.schemas.events import ObservableFact, fact_recipient_ids


MAX_ROUTER_BATCH_INPUTS = 5
MAX_ROUTER_BATCH_EVENTS = 5
MAX_ROUTER_NEXT_TURNS = 5

RouterInputKind = Literal[
    "player",
    "character",
    "world",
    "cat_ii_resolution",
    "authoritative_result",
    "query",
    "ruleset",
]
DndInteractionMode = Literal[
    "narrative",
    "dnd_combat_start",
    "dnd_combat_end",
]


def _validate_unique_ids(label: str, values: list[str]) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} cannot contain blank ids")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} cannot contain duplicate ids")


class RouterInputEnvelope(BaseModel):
    """One engine-authored proposal submitted to a batched router call."""

    model_config = ConfigDict(extra="forbid")

    submission_id: str
    input_index: int
    lane_id: str
    kind: RouterInputKind
    actor_ids: list[str]
    participant_ids: list[str]
    source_event_ids: list[str]
    chosen_at_s: int
    observed_through_event_sequence: int
    observed_through_s: int
    payload: str

    @model_validator(mode="after")
    def _validate_envelope(self) -> "RouterInputEnvelope":
        if not self.submission_id.strip():
            raise ValueError("router submission_id must not be blank")
        if not self.lane_id.strip():
            raise ValueError("router lane_id must not be blank")
        if self.input_index < 0 or self.input_index >= MAX_ROUTER_BATCH_INPUTS:
            raise ValueError("router input_index is outside the batch limit")
        for label, values in (
            ("actor_ids", self.actor_ids),
            ("participant_ids", self.participant_ids),
            ("source_event_ids", self.source_event_ids),
        ):
            _validate_unique_ids(label, values)
        if self.kind in {"player", "character"} and len(self.actor_ids) != 1:
            raise ValueError(f"{self.kind} input requires exactly one actor")
        if self.kind == "world" and self.actor_ids:
            raise ValueError("world input cannot name an actor")
        if any(actor not in self.participant_ids for actor in self.actor_ids):
            raise ValueError("every input actor must be one of its participants")
        if self.chosen_at_s < 0 or self.observed_through_s < 0:
            raise ValueError("router input times cannot be negative")
        if self.observed_through_event_sequence < -1:
            raise ValueError("observed event sequence cannot be less than -1")
        if not self.payload.strip():
            raise ValueError("router input payload must not be blank")
        return self


class RouterNextTurn(BaseModel):
    """A model-selected causal frontier entry."""

    model_config = ConfigDict(extra="forbid")

    turn_kind: Literal["character", "world"]
    actor_id: str
    participant_ids: list[str]
    causal_group: int | None

    @model_validator(mode="after")
    def _validate_turn(self) -> "RouterNextTurn":
        _validate_unique_ids("next-turn participants", self.participant_ids)
        if self.causal_group is not None and self.causal_group < 0:
            raise ValueError("next-turn causal_group cannot be negative")
        if self.turn_kind == "world":
            if self.actor_id:
                raise ValueError("world next turns cannot name an actor")
            if self.causal_group is None:
                raise ValueError("world next turns require a causal group")
        elif not self.actor_id.strip() or self.actor_id not in self.participant_ids:
            raise ValueError("next-turn actor must be one of its participants")
        return self


class SpawnSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    reason: str
    location: str
    objectives: list[str]
    knowledge_tier: int


class SpawnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    seed: SpawnSeed

    @model_validator(mode="after")
    def _validate_spawn(self) -> "SpawnRequest":
        if not self.character_id.strip():
            raise ValueError("spawn character id must not be blank")
        if self.seed.knowledge_tier < 0:
            raise ValueError("spawn knowledge tier cannot be negative")
        return self


class CommitmentOpenDirective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor_ids: list[str]
    description: str
    expected_duration_s: int
    max_duration_s: int
    location_label: str

    @model_validator(mode="after")
    def _validate_directive(self) -> "CommitmentOpenDirective":
        _validate_unique_ids("commitment actor ids", self.actor_ids)
        if not self.actor_ids or not self.description.strip():
            raise ValueError("commitment open requires actors and a description")
        if self.expected_duration_s < 0:
            raise ValueError("commitment expected duration cannot be negative")
        if self.max_duration_s < self.expected_duration_s:
            raise ValueError("commitment max duration precedes expected duration")
        return self


class CommitmentResolutionSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor_ids: list[str]
    reason: Literal["resolved", "cancelled", "superseded", "impossible"]
    resolved_at_offset_s: int

    @model_validator(mode="after")
    def _validate_resolution(self) -> "CommitmentResolutionSignal":
        _validate_unique_ids("commitment resolution actor ids", self.actor_ids)
        if not self.actor_ids:
            raise ValueError("commitment resolution requires actors")
        if self.resolved_at_offset_s < 0:
            raise ValueError("commitment resolution offset cannot be negative")
        return self


class CommitmentInterruptSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor_ids: list[str]
    observed_at_offset_s: int
    reason: str

    @model_validator(mode="after")
    def _validate_interrupt(self) -> "CommitmentInterruptSignal":
        _validate_unique_ids("commitment interrupt actor ids", self.actor_ids)
        if not self.actor_ids:
            raise ValueError("commitment interrupt requires actors")
        if self.observed_at_offset_s < 0 or not self.reason.strip():
            raise ValueError("commitment interrupt needs non-negative time and reason")
        return self


class LocationUpdateSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    location_label: str

    @model_validator(mode="after")
    def _validate_location(self) -> "LocationUpdateSignal":
        if not self.character_id.strip() or not self.location_label.strip():
            raise ValueError("location update requires character and location")
        return self


class WakeSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    location_label: str

    @model_validator(mode="after")
    def _validate_wake(self) -> "WakeSignal":
        if not self.character_id.strip():
            raise ValueError("activation requires a character id")
        return self


class RouterEventDraft(BaseModel):
    """One provider-authored event candidate in a router batch."""

    model_config = ConfigDict(extra="forbid")

    feasible_input_indexes: list[int]
    infeasible_input_indexes: list[int]
    duration_s: int
    observable_facts: list[ObservableFact]
    required_responders: list[str]
    appearance_target_ids: list[str]
    spawn: list[SpawnRequest]
    dormant: list[str]
    cull: list[str]
    commitment_opens: list[CommitmentOpenDirective]
    commitment_resolutions: list[CommitmentResolutionSignal]
    commitment_interrupts: list[CommitmentInterruptSignal]
    location_updates: list[LocationUpdateSignal]
    activate: list[WakeSignal]

    def _has_effect(self) -> bool:
        return any((
            self.observable_facts,
            self.spawn,
            self.dormant,
            self.cull,
            self.commitment_opens,
            self.commitment_resolutions,
            self.commitment_interrupts,
            self.location_updates,
            self.activate,
            getattr(self, "state_updates", ()),
            getattr(self, "interaction_mode", "narrative") != "narrative",
            getattr(self, "combatant_ids", ()),
            getattr(self, "combatant_spawns", ()),
            bool(getattr(getattr(self, "loot_offer", None), "present", False)),
            bool(getattr(getattr(self, "battle_map_seed", None), "present", False)),
        ))

    @model_validator(mode="after")
    def _validate_draft(self) -> "RouterEventDraft":
        indexes = [*self.feasible_input_indexes, *self.infeasible_input_indexes]
        if not indexes:
            raise ValueError("router event draft must resolve at least one input")
        if any(index < 0 or index >= MAX_ROUTER_BATCH_INPUTS for index in indexes):
            raise ValueError("router event draft references an invalid input index")
        if len(indexes) != len(set(indexes)):
            raise ValueError("an input index cannot appear twice in one event draft")
        if self.duration_s < 0:
            raise ValueError("event duration cannot be negative")
        for label, values in (
            ("required responders", self.required_responders),
            ("appearance targets", self.appearance_target_ids),
            ("dormant ids", self.dormant),
            ("culled ids", self.cull),
        ):
            _validate_unique_ids(label, values)
        if len(self.commitment_opens) > 1:
            raise ValueError("an event can open at most one commitment")
        for fact in self.observable_facts:
            if fact.at_offset_s + fact.duration_s > self.duration_s:
                raise ValueError("fact timing exceeds its event duration")
        if any(
            signal.resolved_at_offset_s > self.duration_s
            for signal in self.commitment_resolutions
        ):
            raise ValueError("commitment resolution exceeds event duration")
        if any(
            signal.observed_at_offset_s > self.duration_s
            for signal in self.commitment_interrupts
        ):
            raise ValueError("commitment interrupt exceeds event duration")
        if self.required_responders and self.duration_s != 0:
            raise ValueError("an unresolved contested event must have zero duration")
        if set(self.required_responders) - set(self.observer_ids):
            raise ValueError("contested responders must receive the attempt")
        visual_subjects = {
            cid for fact in self.observable_facts for cid in fact.visual_subject_ids
        }
        if set(self.appearance_target_ids) - visual_subjects:
            raise ValueError("appearance targets require visual recipients")
        if not self._has_effect() and (
            self.feasible_input_indexes or not self.infeasible_input_indexes
        ):
            raise ValueError("only infeasible inputs may resolve without an event")
        return self

    @property
    def is_no_event_resolution(self) -> bool:
        return not self._has_effect()

    @property
    def observer_ids(self) -> list[str]:
        return fact_recipient_ids(self.observable_facts)


class RouterBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[RouterEventDraft]
    next_turns: list[RouterNextTurn]

    def validate_for_inputs(
        self,
        inputs: list[RouterInputEnvelope],
    ) -> "RouterBatchOutput":
        if not inputs or len(inputs) > MAX_ROUTER_BATCH_INPUTS:
            raise ValueError("router batch input count must be from one through five")
        if [item.input_index for item in inputs] != list(range(len(inputs))):
            raise ValueError("router input indexes must be contiguous and ordered")
        if len({item.submission_id for item in inputs}) != len(inputs):
            raise ValueError("router submission ids must be unique")
        if len(self.events) > MAX_ROUTER_BATCH_EVENTS:
            raise ValueError("router batch exceeds the event limit")
        if len(self.next_turns) > MAX_ROUTER_NEXT_TURNS:
            raise ValueError("router batch exceeds the next-turn limit")
        accounted = [
            index
            for event in self.events
            for index in (
                *event.feasible_input_indexes,
                *event.infeasible_input_indexes,
            )
        ]
        if sorted(accounted) != list(range(len(inputs))):
            raise ValueError("every input index must appear exactly once across events")

        for event in self.events:
            selected = [
                inputs[index]
                for index in (
                    *event.feasible_input_indexes,
                    *event.infeasible_input_indexes,
                )
            ]
            if any(item.kind == "cat_ii_resolution" for item in selected):
                if event.is_no_event_resolution or event.required_responders:
                    raise ValueError(
                        "a contested-action resolution must close as one event"
                    )

        return self


class DndRouterEventDraft(RouterEventDraft):
    interaction_mode: DndInteractionMode
    combatant_ids: list[str]
    combatant_spawns: list[DndCombatantSpawn]
    loot_offer: DndLootOfferSignal
    battle_map_seed: DndBattleMapSeed
    dnd_reaction_ids: list[str]

    @model_validator(mode="after")
    def _validate_dnd_draft(self) -> "DndRouterEventDraft":
        _validate_unique_ids("D&D reaction ids", self.dnd_reaction_ids)
        if set(self.dnd_reaction_ids) - set(self.observer_ids):
            raise ValueError("D&D reactions require a perceived trigger")
        if self.interaction_mode == "narrative":
            if self.combatant_ids or self.combatant_spawns or self.battle_map_seed.present:
                raise ValueError("narrative events cannot carry combat start fields")
        elif self.interaction_mode == "dnd_combat_start":
            ids = [
                *self.combatant_ids,
                *(spawn.character_id for spawn in self.combatant_spawns),
            ]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("combat start needs unique non-empty combatants")
            if self.required_responders:
                raise ValueError("combat start cannot also open a generic contest")
        elif self.combatant_ids or self.combatant_spawns or self.battle_map_seed.present:
            raise ValueError("combat end cannot carry combat start fields")
        if self.interaction_mode != "narrative" and (
            self.required_responders or self.dnd_reaction_ids
        ):
            raise ValueError(
                "combat lifecycle events cannot open response or reaction work"
            )
        return self


class DndRouterBatchOutput(RouterBatchOutput):
    events: list[DndRouterEventDraft]


class CanonicalEventRecord(BaseModel):
    """Durable materialized fiction, free of transient scheduling decisions."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    causal_lane_id: str
    effective_at_s: int
    duration_s: int
    actor_ids: list[str]
    observable_facts: list[ObservableFact]
    spawn: list[SpawnRequest]
    dormant: list[str]
    cull: list[str]
    commitment_opens: list[CommitmentOpenDirective]
    commitment_resolutions: list[CommitmentResolutionSignal]
    commitment_interrupts: list[CommitmentInterruptSignal]
    location_updates: list[LocationUpdateSignal]
    activate: list[WakeSignal]

    @model_validator(mode="after")
    def _validate_record(self) -> "CanonicalEventRecord":
        if not self.event_id.strip() or not self.causal_lane_id.strip():
            raise ValueError("canonical event and causal lane ids must not be blank")
        if self.effective_at_s < 0 or self.duration_s < 0:
            raise ValueError("canonical event time cannot be negative")
        _validate_unique_ids("actor ids", self.actor_ids)
        for fact in self.observable_facts:
            if fact.at_offset_s + fact.duration_s > self.duration_s:
                raise ValueError("canonical fact timing exceeds event duration")
        return self

    @property
    def observer_ids(self) -> list[str]:
        return fact_recipient_ids(self.observable_facts)


class DndCanonicalEventRecord(CanonicalEventRecord):
    interaction_mode: DndInteractionMode
    combatant_ids: list[str]
    combatant_spawns: list[DndCombatantSpawn]
    loot_offer: DndLootOfferSignal
    battle_map_seed: DndBattleMapSeed


class FrontierTurn(BaseModel):
    """Durable, runnable story work selected by a prior router batch."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    lane_id: str
    turn_kind: Literal["character", "world"]
    actor_id: str
    participant_ids: list[str]
    source_event_ids: list[str]
    gating_pov_ids: list[str]

    @model_validator(mode="after")
    def _validate_frontier_turn(self) -> "FrontierTurn":
        if not self.turn_id.strip() or not self.lane_id.strip():
            raise ValueError("frontier turn and lane ids must not be blank")
        for label, values in (
            ("frontier participants", self.participant_ids),
            ("frontier sources", self.source_event_ids),
            ("frontier POV gates", self.gating_pov_ids),
        ):
            _validate_unique_ids(label, values)
        if self.turn_kind == "world":
            if self.actor_id or not self.source_event_ids:
                raise ValueError("world frontier needs sources and no actor")
        elif not self.actor_id or self.actor_id not in self.participant_ids:
            raise ValueError("character frontier actor must be a participant")
        return self


def materialized_event_id(
    *,
    session_id: str,
    revision: int,
    event_index: int,
    submission_ids: list[str],
) -> str:
    basis = "\x1f".join((
        session_id,
        str(revision),
        str(event_index),
        *submission_ids,
    ))
    return "evt_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]
