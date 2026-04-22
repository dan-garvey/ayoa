from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    event_router: str = "claude-sonnet-4-6"
    narrator: str = "claude-sonnet-4-6"
    discriminator: str = "claude-sonnet-4-6"
    agent_default: str = "claude-sonnet-4-6"


class SlotEntry(BaseModel):
    """v11: one participant in a scene's active_act_slot. A scene may have
    zero (free), one (initiator or single Cat II responder), or many
    entries (multi-character Cat II). The reason determines what /act
    from that user is allowed to do.
    """
    reason: str  # "initiator" | "cat_ii_responder"
    # When reason=="cat_ii_responder", the open-event id this slot is
    # pinned to. None for initiator entries.
    cat_ii_event_id: str | None = None
    # ISO-8601 timestamp of when the slot was claimed. Used for debug and
    # for a future "stale slot" cleanup pass.
    claimed_at: str = ""


class OpenCatIIEvent(BaseModel):
    """v11: a Cat II (contested) event that's collecting responder
    intentions and hasn't adjudicated yet. The router opens it when an
    intention classifies as Cat II; the orchestrator closes it when all
    required_responders have intended.
    """
    event_id: str
    scene_id: str
    # Character whose initial intention opened this event.
    initiator_id: str
    # The intention text that opened the event (for the router's
    # composition call when it adjudicates).
    initiator_intention: str
    # Characters whose intentions are required before this closes.
    required_responders: list[str] = Field(default_factory=list)
    # Intentions collected so far: responder_id -> intention text.
    collected_intentions: dict[str, str] = Field(default_factory=dict)
    opened_at: str = ""


class RenderBufferEntry(BaseModel):
    """v11: one canonical event queued for a human's next render. Keyed
    by an event_id stored in canonical_events. The narrator reads these
    entries + the events themselves to compose per-beat prose.
    """
    event_id: str
    # "direct" | "indirect" | "inferred" — observation level for this
    # character, copied from the event's observer list at broadcast
    # time.
    observation_level: str = "direct"


class SessionSettings(BaseModel):
    """User-tunable experimental toggles exposed via /settings.

    Kept separate from the static SessionConfig (models, stream_mode,
    narrative_rules) so the command surface can introspect exactly the
    fields meant for live tuning. Add new toggles here and they become
    automatically available in /settings list / set.
    """
    # Allow off-stage NPC ticks to invent new scenes via their output's
    # scenes_created field. Default off — the router owns world topology
    # in the baseline pipeline. Flip on to experiment with more
    # emergent world-building from character-level intent.
    agents_can_create_scenes: bool = False
    # Max NPC agents that may respond to a single player action. The
    # router's suggested_response_cap is clamped against this; higher
    # values mean richer multi-party scenes at more tokens/latency.
    # 0 is legal (silences all NPC responses — useful for diagnostics).
    max_responders: int = 3
    # Concurrency cap for the off-stage tick pass. Ticks are independent
    # so the only cost of raising this is API rate usage; lowering
    # trades latency for safety on small quotas.
    tick_concurrency: int = 4
    # When True, the tick pass also fires on scene change (in addition to
    # the cadence counter). Scene changes happen frequently in play, so
    # this doubles the tick budget in practice. Default True to preserve
    # the old behavior; flip off to save tokens during playtesting.
    ticks_on_scene_change: bool = True
    # v11: hard cap on how many canonical events the router may chain
    # inside a single beat before the orchestrator forces render + slot
    # release. Prevents runaway agent cascades. 5 is a reasonable
    # starting point for playtest; tune after observing real runs.
    max_events_per_beat: int = 5
    # v11: maximum seconds a Cat II pin may hold a human before the
    # orchestrator's sweep auto-resolves them as "stays out." Default
    # 24 hours — long enough that async multiplayer (play over a day)
    # doesn't time out, short enough that abandoned sessions eventually
    # release. Set to 0 to disable the sweep entirely. The auto-resolve
    # is visible: the rendered outcome notes that the player did not
    # act, so everyone sees the fallback happened.
    cat_ii_human_timeout_seconds: int = 24 * 60 * 60


class SessionConfig(BaseModel):
    models: ModelConfig = Field(default_factory=ModelConfig)
    debug: bool = False
    stream_mode: str = "final_only"
    # Long-form narrator style rules: prose discipline, pacing, subtext philosophy
    narrative_rules: str = ""
    # User-tunable experimental settings. Expanding this is an
    # additive change; do not break existing defaults.
    settings: SessionSettings = Field(default_factory=SessionSettings)


class SessionState(BaseModel):
    session_id: str
    story_id: str = ""
    turn_index: int = 0
    # The creator's binding: populated at /story start by auto-binding the
    # creator to the is_player roster character. Kept for single-player
    # convenience (briefing, legacy prompts). Multi-player bindings live in
    # character_bindings below; player_character_id is one of those keys.
    player_name: str = ""
    player_character_id: str = ""
    # Multi-player bindings: character_id -> discord user id (stringified so
    # the JSON is stable across the int-sized Discord ids). A character with
    # no entry here is AI-driven. Updated by /join, /leave, and /story start.
    character_bindings: dict[str, str] = Field(default_factory=dict)
    # Off-stage tick scheduler state. Counter increments each player turn
    # and fires a tick pass when it hits tick_cadence OR on scene change
    # (whichever comes first). Both cases reset the counter.
    tick_turn_counter: int = 0
    tick_cadence: int = 10
    tick_last_scene_id: str = ""
    # One-slot rolling buffer for the router's missing-narrator-context
    # problem. A Haiku summarizer produces a terse delta note at end of
    # turn N describing what the narrator rendered that the router
    # needs to know. That note is consumed by turn N+1's router call
    # (embedded in its user message, which then archives into
    # session_conversation) and cleared. Empty on a fresh session.
    pending_recap: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    config: SessionConfig = Field(default_factory=SessionConfig)

    # v11: beat-pacing state. One active_act_slot per scene_id; an entry's
    # key within the inner dict is the character_id of the slot-holder.
    # A scene may have zero, one, or many entries (many = multi-responder
    # Cat II). An unmapped scene_id means the scene is free.
    active_act_slots: dict[str, dict[str, SlotEntry]] = Field(default_factory=dict)
    # v11: in-flight Cat II events awaiting responder intentions.
    open_cat_ii_events: list[OpenCatIIEvent] = Field(default_factory=list)
    # v11: per-player queue of canonical events awaiting render. Keyed by
    # character_id (a human's bound character). Cleared after each render
    # fires. An agent's "render buffer" is just its observation context
    # on the next intend() call — no separate store here.
    render_buffers: dict[str, list[RenderBufferEntry]] = Field(default_factory=dict)
    # v11: queued /acts that arrived during a moment where they can't be
    # processed yet — or that may need re-examination once a slot frees.
    # In the reject-on-conflict model we're shipping, this is usually
    # empty; kept for future queueing/inspection.
    pending_intentions: list[dict[str, str]] = Field(default_factory=list)


class TimeState(BaseModel):
    scene_time: datetime = Field(default_factory=datetime.utcnow)
    turn_count: int = 0


class LocationState(BaseModel):
    current_scene_id: str = ""
    scene_graph: dict[str, Any] = Field(default_factory=dict)


class PhysicsRuleset(BaseModel):
    strength_limits: str = "human_baseline"
    magic_enabled: bool = False


class StorySetting(BaseModel):
    """Genre, era, and tone metadata — used to ground character genesis and narrator voice."""
    genre: str = ""
    era: str = ""
    tone: str = ""
    premise: str = ""


class WorldState(BaseModel):
    time: TimeState = Field(default_factory=TimeState)
    locations: LocationState = Field(default_factory=LocationState)
    facts: list[str] = Field(default_factory=list)
    physics_ruleset: PhysicsRuleset = Field(default_factory=PhysicsRuleset)
    global_flags: dict[str, Any] = Field(default_factory=dict)
    setting: StorySetting = Field(default_factory=StorySetting)
    # Long-form world lore: history, factions, laws, magic systems, etc.
    lore: str = ""
    # Hidden lore/facts — available to discriminator and agents for authentic
    # reactions, but NEVER shown to the narrator or the player. These contain
    # spoilers, conspiracy details, and secrets to be discovered through play.
    hidden_lore: str = ""
    hidden_facts: list[str] = Field(default_factory=list)
    # Characters the player has been formally introduced to (by character_id).
    # NP2 uses names only for known characters; others are described by appearance.
    known_characters: list[str] = Field(default_factory=list)
