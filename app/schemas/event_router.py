from __future__ import annotations

import uuid
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from app.schemas.events import CanonicalEvent
from app.schemas.dnd_inventory import (
    DndLootOfferSignal,
    empty_loot_offer_signal,
)
from app.schemas.dnd_spatial import (
    DndBattleMapState,
    empty_battle_map_state,
)
from app.schemas.dnd_monsters import DndCombatantSpawn


EventKind = Literal[
    # The beat should continue after this event by collecting the next
    # router-selected character output.
    "beat_continues",
    # A public or semi-public information event. The event is still a normal
    # canonical event: observers receive visible facts through the usual inbox
    # path. `next_output` observers may also receive a background turn.
    "public_fact",
    "directed_at_player",
    "state_change",
    "cascade_exhausted",
    "cat_ii_resolution",
    "cat_ii_open",
    "ambient_pause",
    "max_events_cap",
    # Ruleset adapters may resolve a turn/action outside the generic router
    # while still returning the standard canonical event shape.
    "ruleset_resolution",
    # Engine-authored guard: a ruleset-owned mode suppressed the generic Cat II
    # responder flow in favor of adapter-native handling.
    "ruleset_cat_ii_suppressed",
    # query_response: a private /query result. If the answer needs current
    # visual self-presentation from NPCs, observers with
    # routing_role="perception_enrichment" name harvest targets rather than
    # response actors and the harvested loadout is authoritative.
    "query_response",
    # observation_harvest: the actor's intention is pure targeted
    # observation of perceptually available characters (looking, studying, sizing
    # up, scanning) with no dialogue and no physical interaction.
    # The router puts the observation TARGETS (NPCs the actor is looking at)
    # in observers with routing_role="perception_enrichment"; the engine takes
    # a different path on this reason instead of cascading those targets as
    # agents-with-intentions. It fires each target via
    # `Dispatcher.harvest_perceptions` to harvest one self-presentation
    # fragment per character ("what does the world see of me right now"), and
    # appends the fragments to the canonical event's `observable_facts` before
    # the narrator renders. `event_kind` MUST be observation_harvest; enrichment
    # targets MUST be non-empty (no targets = nothing to harvest = router should
    # pick `state_change` or `cascade_exhausted` instead). See the router
    # prompt's "Observation harvest" section for classification guidance.
    "observation_harvest",
]

TERMINAL_EVENT_KINDS = set(EventKind.__args__) - {"beat_continues"}

ObserverRoutingRole = Literal[
    # Receives any visible facts for this event, but no immediate output is
    # requested from this observer.
    "observe_only",
    # The router wants this participant to produce the next live output for
    # this beat. Runtime maps NPC ids to agent turns; human-bound characters
    # render from their buffer when the router emits a terminal event.
    "next_output",
    # The router wants a perception/loadout enrichment target rather than an
    # in-fiction response actor. Used by observation_harvest/query_response.
    "perception_enrichment",
]

DndObserverRoutingRole = Literal[
    "observe_only",
    "next_output",
    "perception_enrichment",
    # D&D adapter extension: a direct observer may need a player-facing combat
    # reaction prompt. This is not generic narrative dispatch.
    "dnd_reaction",
]

FRONTIER_ROUTING_ROLES = {"next_output"}
PERCEPTION_ENRICHMENT_ROUTING_ROLES = {"perception_enrichment"}

DndInteractionMode = Literal[
    "cat_i",
    "cat_ii",
    "dnd_combat_start",
    "dnd_combat_end",
]


def _new_event_id() -> str:
    """Stable event identifier for the canonical event log + render
    buffers. Short enough to be dev-readable; unique across sessions."""
    return f"evt_{uuid.uuid4().hex[:12]}"


def _ascii_id_skeleton(value: str) -> str:
    return "".join(ch for ch in (value or "").strip() if ch.isascii()).lower()


def _unique_match(values: list[str]) -> str:
    unique = list(dict.fromkeys(value for value in values if value))
    return unique[0] if len(unique) == 1 else ""


def _repair_observer_id(value: str, observer_ids: list[str]) -> str:
    """Best-effort repair for model typos in fact-level visibility ids.

    The router sometimes emits a visually close but invalid character id in
    `observable_facts[].visible_to` while the correct id is already present in
    `observers`. Repair only when the observer set makes the intended target
    unambiguous; otherwise return "" and let the caller drop the private
    recipient instead of leaking it.
    """
    raw = (value or "").strip()
    if not raw:
        return ""
    observer_ids = list(dict.fromkeys(cid for cid in observer_ids if cid))
    if raw in observer_ids:
        return raw

    raw_skeleton = _ascii_id_skeleton(raw)
    if raw_skeleton:
        exact = _unique_match(
            [
                observer_id
                for observer_id in observer_ids
                if _ascii_id_skeleton(observer_id) == raw_skeleton
            ]
        )
        if exact:
            return exact

        raw_suffix = raw_skeleton.rsplit("_", 1)[-1]
        raw_first = next((ch for ch in raw_skeleton if ch.isalnum()), "")
        if raw_suffix and raw_first:
            suffix_candidates = []
            for observer_id in observer_ids:
                observer_skeleton = _ascii_id_skeleton(observer_id)
                if not observer_skeleton.startswith(raw_first):
                    continue
                if observer_skeleton.rsplit("_", 1)[-1] != raw_suffix:
                    continue
                if (
                    SequenceMatcher(None, raw_skeleton, observer_skeleton).ratio()
                    < 0.8
                ):
                    continue
                suffix_candidates.append(observer_id)
            suffix_match = _unique_match(suffix_candidates)
            if suffix_match:
                return suffix_match

    basis = raw_skeleton or raw.lower()
    scored = sorted(
        (
            (
                SequenceMatcher(
                    None,
                    basis,
                    _ascii_id_skeleton(observer_id) or observer_id.lower(),
                ).ratio(),
                observer_id,
            )
            for observer_id in observer_ids
        ),
        reverse=True,
    )
    if not scored:
        return ""
    best_score, best_id = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.82 and best_score - second_score >= 0.08:
        return best_id
    return ""


class ObserverEntry(BaseModel):
    """Which characters observed or are otherwise targeted by the event.

    Observation level is a single-char enum: "d" = direct (clear live
    access through presence or a mediated channel), "i" = indirect
    (adjacent, degraded, muffled, or partial spillover), "f" = inferred
    (aftermath or ambient inference only). This is event-level
    visibility, not a guarantee that every fact is shared. Fact-level
    visibility lives on each
    `canonical_event.observable_facts[]` entry; downstream consumers
    select events by observer, then filter that event's facts by
    `audience` / `visible_to`.

    All fields REQUIRED — see EventRouterOutput docstring for the
    "Schema is too complex" rationale behind no defaults anywhere in
    this module."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    observation_level: str
    routing_role: ObserverRoutingRole


class DndObserverEntry(ObserverEntry):
    """D&D observer entry extends generic routing with adapter-owned cases."""

    routing_role: DndObserverRoutingRole


class SpawnSeed(BaseModel):
    """Router-authored direction for character generation.

    OpenAI strict structured outputs require object schemas to declare
    `additionalProperties: false`, so this cannot be a freeform dict.
    Keep the fields broad enough for the router's narrative direction
    while preserving a fixed schema shape.
    """
    model_config = ConfigDict(extra="forbid")

    role: str
    reason: str
    location: str
    objectives: list[str]


class SpawnRequest(BaseModel):
    """Router-directed creation of a new character. `seed` is a fixed
    object consumed by character_gen.

    All fields REQUIRED. The LLM emits `seed={}` for spawns with no
    additional context."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    seed: SpawnSeed


class CommitmentOpenSignal(BaseModel):
    """Router-authored internal long-action directive.

    This is not observable prose. Visible setup, if any, belongs in
    `observable_facts`; the directive only tells runtime state that an actor
    has begun an interruptible activity.
    """

    model_config = ConfigDict(extra="forbid")

    present: bool
    actor_ids: list[str]
    description: str
    expected_duration_s: int
    max_duration_s: int
    location_label: str

    @model_validator(mode="after")
    def _clean(self) -> "CommitmentOpenSignal":
        self.actor_ids = [
            cid.strip() for cid in dict.fromkeys(self.actor_ids) if cid.strip()
        ]
        self.description = self.description.strip()
        self.location_label = self.location_label.strip()
        if self.expected_duration_s < 0:
            self.expected_duration_s = 0
        if self.max_duration_s < 0:
            self.max_duration_s = 0
        return self


class CommitmentResolutionSignal(BaseModel):
    """Router-authored instruction to close an existing commitment."""

    model_config = ConfigDict(extra="forbid")

    commitment_id: str
    actor_ids: list[str]
    reason: str
    resolved_at_offset_s: int

    @model_validator(mode="after")
    def _clean(self) -> "CommitmentResolutionSignal":
        self.commitment_id = self.commitment_id.strip()
        self.actor_ids = [
            cid.strip() for cid in dict.fromkeys(self.actor_ids) if cid.strip()
        ]
        self.reason = self.reason.strip().lower() or "resolved"
        if self.resolved_at_offset_s < 0:
            self.resolved_at_offset_s = 0
        return self


class CommitmentInterruptSignal(BaseModel):
    """Router-authored revision prompt for a still-open commitment."""

    model_config = ConfigDict(extra="forbid")

    commitment_id: str
    actor_ids: list[str]
    observed_at_offset_s: int
    reason: str

    @model_validator(mode="after")
    def _clean(self) -> "CommitmentInterruptSignal":
        self.commitment_id = self.commitment_id.strip()
        self.actor_ids = [
            cid.strip() for cid in dict.fromkeys(self.actor_ids) if cid.strip()
        ]
        self.reason = self.reason.strip()
        if self.observed_at_offset_s < 0:
            self.observed_at_offset_s = 0
        return self


class LocationUpdateSignal(BaseModel):
    """Router-authored durable update to a character's opaque location label."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    location_label: str

    @model_validator(mode="after")
    def _clean(self) -> "LocationUpdateSignal":
        self.character_id = self.character_id.strip()
        self.location_label = self.location_label.strip()
        return self


def empty_commitment_open_signal() -> dict[str, Any]:
    return {
        "present": False,
        "actor_ids": [],
        "description": "",
        "expected_duration_s": 0,
        "max_duration_s": 0,
        "location_label": "",
    }


class EventRouterOutput(BaseModel):
    """Merged adjudication + perception output — AND in v11, the router's
    beat-pacing decision.

    v11 additions:
      - `requires_responders` + `required_responders`: Cat I (self-closing)
        vs Cat II (contested). Cat II events open and collect responder
        intentions before canonicalization closes.
      - `event_kind`: the router's pacing and dispatch signal. The engine
        derives beat closure from this field: `beat_continues` requests the
        next ordered character output, `public_fact` delivers public
        information and can select background output, while terminal event
        kinds render, suspend, or hand off to adapter-owned flows.
      - `observers[].routing_role`: ordered perception/render/agent-routing
        intent. `next_output` entries are the response candidates for a live
        beat; `perception_enrichment` entries are non-response targets for
        observation/query enrichment.

    ## Schema-shape policy: no Pydantic defaults

    Every field is REQUIRED (no `default=`, no `default_factory=`). This
    schema went 12-optionals-deep with default_factory list/dict
    fallbacks, which expanded the API's grammar compiler to 2^12+
    grammar states and tripped Anthropic's "Schema is too complex"
    400 (same failure mode that bit AuthoredCharacter — see
    app/schemas/takeover.py docstring). The compiler would also
    intermittently time out as a 500 ("Internal server error") on the
    same compilation attempt before giving up with the 400.

    All-required collapses the grammar to a single fixed shape. The
    LLM emits explicit `""` / `[]` / `false` for empty content; an
    `event_id` of `""` triggers the `_assign_event_id` validator which
    mints a fresh one.
    """

    model_config = ConfigDict(extra="forbid")

    # Stable event identifier. Populated at parse time — LLM emits
    # "" and the validator below mints a real id, so the schema can
    # be all-required (a default_factory would force this field
    # back into "optional" in the JSON schema and re-explode the
    # grammar).
    event_id: str
    effective_at_s: int
    duration_s: int

    # ---- v11-r7g: TEMPORARY diagnostic — terse justification ------------
    # The router emits a concise diagnostic note explaining its core
    # routing decision this turn (Cat I vs Cat II classification, why this
    # event_kind, why these routing roles). We log it at INFO so playtest transcripts
    # surface the "why" alongside the "what".
    #
    # NON-FREE in tokens — adds ~1 sentence to every router response,
    # and the LLM has to compose it before emitting structural fields.
    # We keep it ONLY while we're solidifying prompt-engineering for
    # the v11 router; it should be removed once the prompt is stable
    # and we trust the routing decisions without inline rationale.
    # When removing: drop this field, drop the prompt's rule + format
    # entry, and drop the engine's INFO log call in turn_loop.
    decision_rationale: str

    canonical_event: CanonicalEvent
    event_kind: EventKind

    # ---- v11: Cat I / Cat II intention classification --------------------
    # When True, this intention is CONTESTED: a canonical event cannot close
    # without the listed required_responders also intending. Examples:
    # violence, consensual physical contact, contested possession, forced
    # movement through a blocker. When False (Cat I), the intention closes
    # immediately — dialogue, passive action, unambiguous movement, OOC
    # directives.
    requires_responders: bool
    # Characters whose intentions must be collected before this Cat II
    # event adjudicates. Includes the direct target plus any plausible
    # intercepter / defender / counter-actor. Empty for Cat I.
    required_responders: list[str]

    # ---- Observation and character lifecycle outputs --------------------
    # `observers` drives render-buffer determination: every
    # character in the observer list gets this event into their render
    # buffer if they're human. Agents get it as observation context for
    # future intend() calls. `routing_role` on each observer is the routing
    # decision for that character.
    observers: list[ObserverEntry]
    spawn: list[SpawnRequest]
    dormant: list[str]
    cull: list[str]
    commitment_open: CommitmentOpenSignal
    commitment_resolutions: list[CommitmentResolutionSignal]
    commitment_interrupts: list[CommitmentInterruptSignal]
    location_updates: list[LocationUpdateSignal]

    @model_validator(mode="before")
    @classmethod
    def _normalize_event_kind(cls, data: Any) -> Any:
        """Pydantic's Literal[] validation rejects unknown values with a
        ValidationError. For a pacing field, a model typo should be a
        warn-log, not a crash. Coerce unknown values to a safe terminal
        event kind and log.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "event_kind" not in data:
            return data
        valid = set(EventKind.__args__)
        raw_kind = data.get("event_kind")
        if not isinstance(raw_kind, str) or raw_kind not in valid:
            import logging
            logging.getLogger(__name__).warning(
                "Unknown event_kind %r coerced to directed_at_player; "
                "valid values: %s", raw_kind, sorted(valid),
            )
            raw_kind = "directed_at_player"
        data["event_kind"] = raw_kind
        return data

    @model_validator(mode="before")
    @classmethod
    def _assign_event_id(cls, data: Any) -> Any:
        """Mint a fresh event_id when the LLM emits "" (the schema-
        defaults policy requires the field to be required, so we
        cannot use Field(default_factory=_new_event_id) — that
        re-introduces the optionality the grammar compiler chokes
        on). The LLM is instructed to emit "" and let the engine
        assign a real one."""
        if isinstance(data, dict):
            eid = data.get("event_id", "")
            if not eid:
                data["event_id"] = _new_event_id()
        return data

    @model_validator(mode="after")
    def _validate_v11_invariants(self) -> "EventRouterOutput":
        """Enforce Cat I / Cat II invariants at the schema boundary so
        the orchestrator never has to defensively re-check them.

        - `requires_responders=true` MUST have at least one
          `required_responders` entry. The empty-set-ready-on-open bug
          (edge-case #2) cannot reach the loop.
        - `required_responders` must be unique; duplicates corrupt the
          collection set semantics in `cat_ii_is_ready`.
        - `observers[].routing_role` may select observer NPCs or explicit
          enrichment targets as ordered response/perception candidates. The
          engine applies hard safety filters before dispatch.
        """
        if self.requires_responders and not self.required_responders:
            raise ValueError(
                "requires_responders=true but required_responders is empty; "
                "an empty Cat II has no one to close it. Treat as Cat I at "
                "the prompt layer."
            )
        if len(self.required_responders) != len(set(self.required_responders)):
            raise ValueError(
                "required_responders contains duplicates; each responder "
                "must appear exactly once."
            )
        observer_ids = [o.character_id for o in self.observers if o.character_id]
        observer_id_set = set(observer_ids)
        visible_facts = []
        repaired_visibility: list[str] = []
        dropped_visibility: list[str] = []
        dropped_fact_count = 0
        for fact in self.canonical_event.observable_facts:
            if fact.audience != "only":
                visible_facts.append(fact)
                continue
            repaired_ids: list[str] = []
            for cid in fact.visible_to:
                if cid in observer_id_set:
                    repaired_ids.append(cid)
                    continue
                repaired = _repair_observer_id(cid, observer_ids)
                if repaired:
                    repaired_ids.append(repaired)
                    repaired_visibility.append(f"{cid}->{repaired}")
                else:
                    dropped_visibility.append(cid)
            fact.visible_to = list(dict.fromkeys(repaired_ids))
            if fact.visible_to:
                visible_facts.append(fact)
            else:
                dropped_fact_count += 1
        self.canonical_event.observable_facts = visible_facts
        if repaired_visibility or dropped_visibility or dropped_fact_count:
            import logging
            logger = logging.getLogger(__name__)
            if repaired_visibility:
                logger.warning(
                    "observable_facts[].visible_to ids repaired against "
                    "observers: %s",
                    sorted(set(repaired_visibility)),
                )
            if dropped_visibility:
                logger.warning(
                    "observable_facts[].visible_to entries not in observers "
                    "were dropped: %s",
                    sorted(set(dropped_visibility)),
                )
            if dropped_fact_count:
                logger.warning(
                    "observable_facts with no valid visible_to recipients "
                    "were dropped: %s",
                    dropped_fact_count,
                )
        if self.event_kind == "observation_harvest":
            # Harvest is a fork in the engine: enrichment roles become
            # perception targets, not cascade actors. These are CLAMP-not-raise
            # checks so prompt drift doesn't crash a session.
            if not self.perception_enrichment_character_ids:
                import logging
                logging.getLogger(__name__).warning(
                    "event_kind='observation_harvest' but "
                    "no perception_enrichment targets were selected; "
                    "nothing to harvest. "
                    "Coercing to cascade_exhausted.",
                )
                self.event_kind = "cascade_exhausted"
        if self.event_kind == "cat_ii_open" or self.requires_responders:
            self.duration_s = 0
        if self.effective_at_s < 0:
            self.effective_at_s = 0
        if self.duration_s < 0:
            self.duration_s = 0
        for fact in self.canonical_event.observable_facts:
            if fact.at_offset_s > self.duration_s:
                fact.at_offset_s = self.duration_s
            if fact.at_offset_s + fact.duration_s > self.duration_s:
                fact.duration_s = max(0, self.duration_s - fact.at_offset_s)
        for signal in self.commitment_resolutions:
            if signal.resolved_at_offset_s > self.duration_s:
                signal.resolved_at_offset_s = self.duration_s
        for signal in self.commitment_interrupts:
            if signal.observed_at_offset_s > self.duration_s:
                signal.observed_at_offset_s = self.duration_s
        self.location_updates = [
            update
            for update in self.location_updates
            if update.character_id and update.location_label
        ]
        return self

    def routed_character_ids(self, *routing_roles: str) -> list[str]:
        roles = set(routing_roles)
        return list(dict.fromkeys(
            observer.character_id
            for observer in self.observers
            if observer.character_id and observer.routing_role in roles
        ))

    @property
    def next_output_character_ids(self) -> list[str]:
        return self.routed_character_ids(*FRONTIER_ROUTING_ROLES)

    @property
    def perception_enrichment_character_ids(self) -> list[str]:
        return self.routed_character_ids(*PERCEPTION_ENRICHMENT_ROUTING_ROLES)

    def clear_routing_roles(self, *routing_roles: str) -> None:
        roles = set(routing_roles or (
            *FRONTIER_ROUTING_ROLES,
            *PERCEPTION_ENRICHMENT_ROUTING_ROLES,
        ))
        for observer in self.observers:
            if observer.routing_role in roles:
                observer.routing_role = "observe_only"  # type: ignore[assignment]


class DndEventRouterOutput(EventRouterOutput):
    """D&D ruleset extension for fresh event-router intentions.

    The generic router's Cat I/Cat II fields stay rules-neutral. In D&D mode,
    fresh intentions add an explicit interaction mode so "contested" and
    "initiative-governed combat" cannot be conflated.
    """

    interaction_mode: DndInteractionMode
    observers: list[DndObserverEntry]
    combatant_ids: list[str]
    combatant_spawns: list[DndCombatantSpawn]
    loot_offer: DndLootOfferSignal
    battle_map_seed: DndBattleMapState

    def clear_routing_roles(self, *routing_roles: str) -> None:
        roles = set(routing_roles or (
            *FRONTIER_ROUTING_ROLES,
            *PERCEPTION_ENRICHMENT_ROUTING_ROLES,
            "dnd_reaction",
        ))
        super().clear_routing_roles(*roles)

    @model_validator(mode="before")
    @classmethod
    def _coerce_interaction_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "loot_offer" not in data:
            data = dict(data)
            data["loot_offer"] = empty_loot_offer_signal()
        if "battle_map_seed" not in data:
            data = dict(data)
            data["battle_map_seed"] = empty_battle_map_state()
        if "combatant_spawns" not in data:
            data = dict(data)
            data["combatant_spawns"] = []
        mode = data.get("interaction_mode")
        if mode in {"cat_i", "dnd_combat_start", "dnd_combat_end"}:
            data = dict(data)
            data["requires_responders"] = False
            data["required_responders"] = []
        elif mode == "cat_ii":
            data = dict(data)
            data["requires_responders"] = True
            data["combatant_ids"] = []
            data["combatant_spawns"] = []
        return data

    @model_validator(mode="after")
    def _validate_interaction_mode(self) -> "DndEventRouterOutput":
        if self.interaction_mode == "cat_ii":
            if not self.required_responders:
                raise ValueError(
                    "interaction_mode='cat_ii' requires required_responders."
                )
            self.requires_responders = True
            self.combatant_ids = []
            self.combatant_spawns = []
            self.battle_map_seed = DndBattleMapState.model_validate(
                empty_battle_map_state()
            )
            return self

        self.requires_responders = False
        self.required_responders = []

        if self.interaction_mode == "dnd_combat_start":
            seen_spawn_ids: set[str] = set()
            combatant_spawns: list[DndCombatantSpawn] = []
            for spawn in self.combatant_spawns:
                if not spawn.character_id or spawn.character_id in seen_spawn_ids:
                    continue
                seen_spawn_ids.add(spawn.character_id)
                combatant_spawns.append(spawn)
            self.combatant_spawns = combatant_spawns
            unique = list(dict.fromkeys([
                *[cid.strip() for cid in self.combatant_ids],
                *[spawn.character_id for spawn in self.combatant_spawns],
            ]))
            self.combatant_ids = [cid for cid in unique if cid]
        elif self.interaction_mode in {"cat_i", "dnd_combat_end"}:
            self.combatant_ids = []
            self.combatant_spawns = []
            self.battle_map_seed = DndBattleMapState.model_validate(
                empty_battle_map_state()
            )

        return self
