from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from app.schemas.events import CanonicalEvent


# Typed enum for `ends_beat_reason`. Keeps the model grammar-constrained
# and makes future branching safe. Adding a new reason requires both a
# schema bump and a prompt update — visible contract.
EndsBeatReason = Literal[
    "",  # free-form / not yet set
    "directed_at_player",
    "scene_transition",
    "state_change",
    "cascade_exhausted",
    "cat_ii_resolution",
    "cat_ii_open",
    "ambient_pause",
    "max_events_cap",
    "cat_ii_pending",
    "cat_ii_stale",
]


def _new_event_id() -> str:
    """Stable event identifier for the canonical event log + render
    buffers. Short enough to be dev-readable; unique across sessions."""
    return f"evt_{uuid.uuid4().hex[:12]}"


class ObserverEntry(BaseModel):
    """Which characters observed the event and how strongly they should
    respond. Observers at priority 0 should be omitted entirely — they're
    routing noise, not silent responders.

    Observation level is a single-char enum: "d" = direct (in the scene),
    "i" = indirect (adjacent, heard/saw spillover), "f" = inferred
    (aftermath or ambient inference only). Agents see `observable_facts`
    from the canonical event; the level is used downstream to filter that
    set, not to duplicate it per-observer.

    All fields REQUIRED — see EventRouterOutput docstring for the
    "Schema is too complex" rationale behind no defaults anywhere in
    this module."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    observation_level: str
    # 1=minimal, 2=low, 3=moderate, 4=high, 5=compelled. Omit 0-priority
    # observers from the output entirely.
    response_priority: int


class SpawnRequest(BaseModel):
    """Router-directed creation of a new character. `seed` is a freeform
    dict (role, reason, location, objectives, ...) consumed by
    character_gen.

    All fields REQUIRED. The LLM emits `seed={}` for spawns with no
    additional context."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    seed: dict[str, Any]


class RosterMove(BaseModel):
    """Router-directed movement of an existing character between scenes.
    Applied by the orchestrator — updates the target character's
    `location` field. Empty list on turns where nobody's moving.

    All fields REQUIRED. The LLM emits `reason=""` for moves with no
    explanation."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    to_scene: str  # scene_id (must exist in scene_graph)
    reason: str


class SceneCreation(BaseModel):
    """Router-directed creation of a new scene.

    The scene graph is seeded at import time from the master prompt, but
    stories naturally imply spaces the author didn't explicitly enumerate
    — a side room, a vista a balcony overlooks, a shop the player glances
    into. When a player action requires a scene that doesn't exist yet,
    the router creates it here rather than silently dropping the move.

    Applied by the orchestrator BEFORE any movement logic on the same
    turn, so `roster_moves.to_scene` and spawn seed locations may
    reference a scene this list introduces.

    Reverse edges are added automatically — if this scene connects to
    `hallway`, the orchestrator also adds this scene's id to
    `hallway.connected_to`. The graph stays traversable in both
    directions without the router having to spell that out.

    All fields REQUIRED. The LLM emits `description=""` and
    `connected_to=[]` when those are absent."""
    model_config = ConfigDict(extra="forbid")

    scene_id: str  # snake_case, unique across the graph
    name: str
    description: str
    # Scene_ids this new scene is directly reachable from. Entries may
    # reference existing scenes OR other scenes in the same
    # scenes_created batch (for newly-paired adjacent spaces).
    connected_to: list[str]


class EventRouterOutput(BaseModel):
    """Merged adjudication + perception output — AND in v11, the router's
    beat-pacing decision.

    v11 additions:
      - `requires_responders` + `required_responders`: Cat I (self-closing)
        vs Cat II (contested). Cat II events open and collect responder
        intentions before canonicalization closes.
      - `agent_responder_picks`: NPCs the router wants to cascade into the
        current beat. Capped by settings.max_responders.
      - `ends_beat` + `ends_beat_reason`: the router's DM-pacing signal —
        when true, the beat composes its buffered events into a render
        and the scene's active_act_slot is released. Cat II adjudication
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
    mints a fresh one. Downstream code is unchanged — the merged
    object always has the legacy shape regardless of what the LLM
    chose to emit per field.
    """

    model_config = ConfigDict(extra="forbid")

    # Stable event identifier. Populated at parse time — LLM emits
    # "" and the validator below mints a real id, so the schema can
    # be all-required (a default_factory would force this field
    # back into "optional" in the JSON schema and re-explode the
    # grammar).
    event_id: str

    # ---- v11-r7g: TEMPORARY diagnostic — one-sentence justification ------
    # The router emits a single sentence explaining its core routing
    # decision this turn (Cat I vs Cat II classification, why this
    # ends_beat value, why these picks). We log it at INFO so playtest
    # transcripts surface the "why" alongside the "what".
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
    # reactive intentions. Cap: clamped to settings.max_responders at the
    # orchestrator layer. Humans are NEVER in this list; humans only
    # enter via /act, gated by active_act_slot. Empty when the router
    # thinks no NPC cascade is warranted.
    agent_responder_picks: list[str]

    # ---- v11: DM pacing — end beat now, or let the cascade continue? ----
    # True = render now; false = router will pick next actor and continue.
    # Cat II adjudication implicitly ends the beat regardless of this
    # value. For Cat I events this is the router's judgment call.
    ends_beat: bool
    # Typed enum; see EndsBeatReason above. Constrained so the grammar
    # rejects typos and future branching is safe.
    ends_beat_reason: EndsBeatReason

    # ---- Legacy observation / roster plumbing (unchanged) ---------------
    # `observers` is retained for the render-buffer determination: every
    # character in the observer list gets this event into their render
    # buffer if they're human. Agents get it as observation context for
    # future intend() calls. The router_responder_picks above is a
    # DIFFERENT decision — who actually fires next, a subset (or
    # superset) of observers.
    observers: list[ObserverEntry]
    spawn: list[SpawnRequest]
    dormant: list[str]
    cull: list[str]
    roster_moves: list[RosterMove]
    scenes_created: list[SceneCreation]

    @model_validator(mode="before")
    @classmethod
    def _clamp_unknown_reason(cls, data: Any) -> Any:
        """Pydantic's Literal[] validation rejects unknown values with a
        ValidationError. For a field whose purpose is telemetry, that's
        too harsh — a model typo ("scene-transition" vs "scene_transition")
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
          the beat should still run.
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
        if self.agent_responder_picks:
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
        return self
