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
from app.schemas.events import ObservableFact


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


class ObserverGroups(BaseModel):
    """Event observers grouped once by perceptual directness."""

    model_config = ConfigDict(extra="forbid")

    direct: list[str]
    indirect: list[str]
    inferred: list[str]

    @model_validator(mode="after")
    def _validate_groups(self) -> "ObserverGroups":
        for label, values in (
            ("direct", self.direct),
            ("indirect", self.indirect),
            ("inferred", self.inferred),
        ):
            _validate_unique_ids(f"observer {label}", values)
        flattened = self.all_ids
        if len(flattened) != len(set(flattened)):
            raise ValueError("an observer must appear in exactly one directness group")
        return self

    @property
    def all_ids(self) -> list[str]:
        return [*self.direct, *self.indirect, *self.inferred]

    def level_for(self, character_id: str) -> str:
        if character_id in self.direct:
            return "direct"
        if character_id in self.indirect:
            return "indirect"
        if character_id in self.inferred:
            return "inferred"
        return ""


class RouterNextTurn(BaseModel):
    """A model-selected causal frontier entry."""

    model_config = ConfigDict(extra="forbid")

    turn_kind: Literal["character", "world"]
    actor_id: str
    participant_ids: list[str]
    source_event_index: int

    @model_validator(mode="after")
    def _validate_turn(self) -> "RouterNextTurn":
        _validate_unique_ids("next-turn participants", self.participant_ids)
        if self.source_event_index < -1:
            raise ValueError("next-turn source_event_index cannot be less than -1")
        if self.turn_kind == "world":
            if self.actor_id:
                raise ValueError("world next turns cannot name an actor")
            if self.source_event_index < 0:
                raise ValueError("world next turns require a source event")
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
    observers: ObserverGroups
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
        observer_ids = set(self.observers.all_ids)
        recipients = {
            character_id
            for fact in self.observable_facts
            for character_id in (
                self.observers.all_ids
                if fact.audience == "all_observers"
                else fact.visible_to
            )
        }
        if observer_ids - recipients:
            raise ValueError("every observer must receive at least one fact")
        for fact in self.observable_facts:
            if fact.audience == "only" and set(fact.visible_to) - observer_ids:
                raise ValueError("fact recipients must be members of observer groups")
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
        if set(self.required_responders) - set(self.observers.direct):
            raise ValueError("contested responders require direct observation")
        if self.appearance_target_ids and not self.observers.all_ids:
            raise ValueError("appearance enrichment requires at least one observer")
        if not self._has_effect() and (
            self.feasible_input_indexes or not self.infeasible_input_indexes
        ):
            raise ValueError("only infeasible inputs may resolve without an event")
        return self

    @property
    def is_no_event_resolution(self) -> bool:
        return not self._has_effect()


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

        used_participants: set[str] = set()
        for turn in self.next_turns:
            if turn.source_event_index >= len(self.events):
                raise ValueError("next turn references a missing event draft")
            if turn.source_event_index >= 0:
                source = self.events[turn.source_event_index]
                if source.is_no_event_resolution:
                    raise ValueError("next turn cannot source a no-event resolution")
                if source.required_responders:
                    raise ValueError("an unresolved contest cannot source a next turn")
                if turn.turn_kind == "character":
                    if turn.actor_id not in source.observers.all_ids:
                        raise ValueError("sourced next-turn actor must observe its event")
                    if not any(
                        fact.is_visible_to(turn.actor_id)
                        for fact in source.observable_facts
                    ):
                        raise ValueError("sourced next-turn actor must receive a fact")
            overlap = set(turn.participant_ids) & used_participants
            if overlap:
                raise ValueError(
                    "simultaneously ready next turns share participants: "
                    + ", ".join(sorted(overlap))
                )
            used_participants.update(turn.participant_ids)
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
        if set(self.dnd_reaction_ids) - set(self.observers.direct):
            raise ValueError("D&D reactions require direct observation")
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
    source_submission_ids: list[str]
    feasible_submission_ids: list[str]
    infeasible_submission_ids: list[str]
    observable_facts: list[ObservableFact]
    observers: ObserverGroups
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
        for label, values in (
            ("actor ids", self.actor_ids),
            ("source submission ids", self.source_submission_ids),
            ("feasible submission ids", self.feasible_submission_ids),
            ("infeasible submission ids", self.infeasible_submission_ids),
        ):
            _validate_unique_ids(label, values)
        if set(self.feasible_submission_ids) & set(self.infeasible_submission_ids):
            raise ValueError("a submission cannot be both feasible and infeasible")
        if set(self.source_submission_ids) != (
            set(self.feasible_submission_ids) | set(self.infeasible_submission_ids)
        ):
            raise ValueError("source submissions must match their outcome ids")
        observer_ids = set(self.observers.all_ids)
        recipients = {
            character_id
            for fact in self.observable_facts
            for character_id in (
                self.observers.all_ids
                if fact.audience == "all_observers"
                else fact.visible_to
            )
        }
        if observer_ids - recipients:
            raise ValueError("every canonical observer must receive at least one fact")
        for fact in self.observable_facts:
            if fact.audience == "only" and set(fact.visible_to) - observer_ids:
                raise ValueError("fact recipients must be canonical observers")
            if fact.at_offset_s + fact.duration_s > self.duration_s:
                raise ValueError("canonical fact timing exceeds event duration")
        return self

    @property
    def observer_ids(self) -> list[str]:
        return self.observers.all_ids

    def observation_level_for(self, character_id: str) -> str:
        return self.observers.level_for(character_id)


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
    created_event_sequence: int
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
        if self.created_event_sequence < 0:
            raise ValueError("frontier creation sequence cannot be negative")
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
