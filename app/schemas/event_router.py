from __future__ import annotations

import uuid
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


# Typed enum for `ends_beat_reason`. Keeps the model grammar-constrained
# and makes future branching safe. Adding a new reason requires both a
# schema bump and a prompt update — visible contract.
EndsBeatReason = Literal[
    "",  # free-form / not yet set
    "directed_at_player",
    "state_change",
    "cascade_exhausted",
    "cat_ii_resolution",
    "cat_ii_open",
    "ambient_pause",
    "off_stage_tick",
    "max_events_cap",
    "cat_ii_pending",
    "cat_ii_stale",
    # Ruleset adapters may resolve a turn/action outside the generic router
    # while still returning the standard canonical event shape.
    "ruleset_resolution",
    # Engine-authored guard: a ruleset-owned mode suppressed the generic Cat II
    # responder flow in favor of adapter-native handling.
    "ruleset_cat_ii_suppressed",
    # query_response: a private /query result. The router emits a
    # canonical observable fact scoped to the querying player. If the
    # answer needs current visual self-presentation from NPCs, picks
    # may name harvest targets rather than response actors.
    "query_response",
    # observation_harvest: the actor's intention is pure targeted
    # observation of perceptually available characters (looking, studying, sizing
    # up, scanning) with no dialogue and no physical interaction.
    # The router puts the observation TARGETS (NPCs the actor is
    # looking at) in `agent_responder_picks`; the engine takes a
    # different path on this reason — instead of cascading those
    # picks as agents-with-intentions, it fires each of them in
    # parallel via `Dispatcher.harvest_perceptions` to harvest
    # one self-presentation fragment per character ("what does
    # the world see of me right now"), and appends the fragments
    # to the canonical event's `observable_facts` before the
    # narrator renders. ends_beat MUST be true on this reason;
    # picks MUST be non-empty (no targets = nothing to harvest =
    # router should pick `state_change` or `cascade_exhausted`
    # instead). See the router prompt's "Observation harvest"
    # section for classification guidance.
    "observation_harvest",
]

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


class ObserverEntry(BaseModel):
    """Which characters observed the event and how strongly they should
    respond. Observers at priority 0 should be omitted entirely — they're
    routing noise, not silent responders.

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
    # 1=minimal, 2=low, 3=moderate, 4=high, 5=compelled. Omit 0-priority
    # observers from the output entirely.
    response_priority: int


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

    def get(self, key: str, default: Any = None) -> Any:
        return self.model_dump().get(key, default)

    def items(self):
        return self.model_dump().items()


class SpawnRequest(BaseModel):
    """Router-directed creation of a new character. `seed` is a fixed
    object consumed by character_gen.

    All fields REQUIRED. The LLM emits `seed={}` for spawns with no
    additional context."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    seed: SpawnSeed

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_seed(cls, data: Any) -> Any:
        """Upgrade older checkpoints/tests that stored partial freeform
        seed dicts. The LLM-facing schema remains all-required."""
        if not isinstance(data, dict):
            return data
        seed = data.get("seed")
        if not isinstance(seed, dict):
            return data

        coerced = {
            "role": str(seed.get("role", "") or ""),
            "reason": str(seed.get("reason", "") or ""),
            "location": str(seed.get("location", "") or ""),
            "objectives": seed.get("objectives") or [],
        }
        if not isinstance(coerced["objectives"], list):
            coerced["objectives"] = [str(coerced["objectives"])]

        data = dict(data)
        data["seed"] = coerced
        return data


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
      - `agent_responder_picks`: NPCs the router wants to cascade into the
        current beat. Addressed NPCs (those the player named, asked, or
        answered) are mandatory until each has had a turn this beat.
      - `ends_beat` + `ends_beat_reason`: the router's DM-pacing signal —
        when true, the beat composes its buffered events into a render
        and the active beat slot is released. Cat II adjudication
        always ends the beat (implicit, regardless of this field). For
        Cat I events the router decides explicitly.

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
    # routing decision this turn (Cat I vs Cat II classification, why
    # this ends_beat value and ends_beat_reason, why these picks). We
    # log it at INFO so playtest transcripts surface the "why" alongside
    # the "what".
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

    # ---- v11: post-canonicalization agent cascade ------------------------
    # Router-selected NPC agents to dispatch into the current beat as
    # reactive intentions. No engine-side cap — the router uses
    # `ends_beat` to do the pacing work. Humans are NEVER in this list;
    # humans only enter via /act, gated by active_act_slot. Empty when
    # the router thinks no NPC cascade is warranted. Addressed NPCs
    # (those the player named, asked, or answered this beat) are
    # mandatory and must remain in the picks across cascade calls
    # until each has fired.
    agent_responder_picks: list[str]

    # ---- v11: DM pacing — end beat now, or let the cascade continue? ----
    # True = render now; false = router will pick next actor and continue.
    # Cat II adjudication implicitly ends the beat regardless of this
    # value. For Cat I events this is the router's judgment call.
    ends_beat: bool
    # Typed enum; see EndsBeatReason above. Constrained so the grammar
    # rejects typos and future branching is safe.
    ends_beat_reason: EndsBeatReason

    # ---- Observation and character lifecycle outputs --------------------
    # `observers` drives render-buffer determination: every
    # character in the observer list gets this event into their render
    # buffer if they're human. Agents get it as observation context for
    # future intend() calls. The router_responder_picks above is a
    # DIFFERENT decision — who actually fires next, a subset (or
    # superset) of observers.
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
    def _clamp_unknown_reason(cls, data: Any) -> Any:
        """Pydantic's Literal[] validation rejects unknown values with a
        ValidationError. For a field whose purpose is telemetry, that's
        too harsh — a model typo in `ends_beat_reason`
        should be a warn-log, not a crash. Coerce any unknown string to
        "" and log; the caller will see the beat close correctly.
        """
        if isinstance(data, dict) and "ends_beat_reason" in data:
            valid = set(EndsBeatReason.__args__)
            val = data["ends_beat_reason"]
            if isinstance(val, str) and val not in valid:
                import logging
                logging.getLogger(__name__).warning(
                    "Unknown ends_beat_reason %r coerced to empty; "
                    "valid values: %s", val, sorted(valid),
                )
                data["ends_beat_reason"] = ""
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

    @model_validator(mode="before")
    @classmethod
    def _coerce_new_timing_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.setdefault("effective_at_s", 0)
        data.setdefault("duration_s", 0)
        data.setdefault("commitment_open", empty_commitment_open_signal())
        data.setdefault("commitment_resolutions", [])
        data.setdefault("commitment_interrupts", [])
        data.setdefault("location_updates", [])
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
        - `agent_responder_picks` must be a subset of the `observers`
          list. An agent the router picks for cascade must also be
          perceiving the event — otherwise they're reacting to something
          they couldn't see. The router prompt declares this as an
          INVARIANT; the schema here CLAMPS by silently dropping any
          pick not in observers and logs a warning. Clamp rather than
          raise because this is prompt drift, not a user-facing error —
          the beat should still run. `query_response` and
          `observation_harvest` are exempt because picks there are
          private perception-harvest targets, not actors reacting to
          the event.
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
        if (
            self.agent_responder_picks
            and self.ends_beat_reason not in {
                "query_response",
                "observation_harvest",
            }
        ):
            observer_ids = {o.character_id for o in self.observers}
            dropped = [p for p in self.agent_responder_picks if p not in observer_ids]
            if dropped:
                import logging
                logging.getLogger(__name__).warning(
                    "agent_responder_picks ⊆ observers invariant violated; "
                    "dropping picks not in observers: %s", dropped,
                )
                self.agent_responder_picks = [
                    p for p in self.agent_responder_picks if p in observer_ids
                ]
        observer_ids = {o.character_id for o in self.observers}
        for fact in self.canonical_event.observable_facts:
            if fact.audience != "only":
                continue
            missing_visibility = [
                cid for cid in fact.visible_to if cid not in observer_ids
            ]
            if missing_visibility:
                raise ValueError(
                    "observable_facts[].visible_to entries must also appear "
                    "in observers for the event. Missing from observers: "
                    + ", ".join(sorted(set(missing_visibility)))
                )
        if self.ends_beat_reason == "observation_harvest":
            # Harvest is a fork in the engine: picks become perception
            # targets, not cascade actors. These are CLAMP-not-raise
            # checks so prompt drift doesn't crash a session.
            if not self.agent_responder_picks:
                import logging
                logging.getLogger(__name__).warning(
                    "ends_beat_reason='observation_harvest' but "
                    "agent_responder_picks is empty; nothing to harvest. "
                    "Coercing to cascade_exhausted.",
                )
                self.ends_beat_reason = "cascade_exhausted"
            if not self.ends_beat:
                import logging
                logging.getLogger(__name__).warning(
                    "ends_beat_reason='observation_harvest' but "
                    "ends_beat=false; harvest implies beat-end. "
                    "Coercing ends_beat=true.",
                )
                self.ends_beat = True
        if self.ends_beat_reason == "cat_ii_open" or self.requires_responders:
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


class DndEventRouterOutput(EventRouterOutput):
    """D&D ruleset extension for fresh event-router intentions.

    The generic router's Cat I/Cat II fields stay rules-neutral. In D&D mode,
    fresh intentions add an explicit interaction mode so "contested" and
    "initiative-governed combat" cannot be conflated.
    """

    interaction_mode: DndInteractionMode
    combatant_ids: list[str]
    loot_offer: DndLootOfferSignal
    battle_map_seed: DndBattleMapState

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
        mode = data.get("interaction_mode")
        if mode in {"cat_i", "dnd_combat_start", "dnd_combat_end"}:
            data = dict(data)
            data["requires_responders"] = False
            data["required_responders"] = []
        elif mode == "cat_ii":
            data = dict(data)
            data["requires_responders"] = True
            data["combatant_ids"] = []
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
            return self

        self.requires_responders = False
        self.required_responders = []

        if self.interaction_mode == "dnd_combat_start":
            unique = list(dict.fromkeys(cid.strip() for cid in self.combatant_ids))
            self.combatant_ids = [cid for cid in unique if cid]
        elif self.interaction_mode in {"cat_i", "dnd_combat_end"}:
            self.combatant_ids = []

        return self
