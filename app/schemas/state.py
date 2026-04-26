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
    # v11-r3a: responder_ids whose intention was SYNTHESIZED by the AFK
    # sweep, not submitted by the human. The Cat II resolution prompt
    # (Part C) skips these responders entirely — they're rendered as
    # present-but-non-reactive rather than having the sentinel string
    # leak into canonical event prose. This is cleaner than parsing a
    # magic-string marker on the intention text.
    swept_responders: list[str] = Field(default_factory=list)
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
    # Concurrency cap for the off-stage tick pass. Ticks are independent
    # so the only cost of raising this is API rate usage; lowering
    # trades latency for safety on small quotas. Hard-capped engine-side
    # by `app.engine.orchestrator.TICK_CONCURRENCY_HARD_CAP` regardless
    # of what's configured here.
    tick_concurrency: int = 4
    # Commit 5: minimum turns between tick-fires triggered by scene
    # change. Without this guard, hopping between two adjacent scenes
    # (very common in early play) would fire a tick on every turn and
    # blow the per-turn API budget. Counted from the LAST tick fire,
    # not from the last scene change. Cooperates with `ticks_on_scene_change`:
    # if that toggle is False, this field is unused. 5 is the v11
    # default — small enough to keep NPCs feeling responsive when the
    # player advances, large enough that quick back-and-forth scenes
    # don't trigger a fan-out per turn.
    tick_scene_change_cooldown: int = 5
    # Commit 5: hard ceiling on consecutive turns with NO tick fire.
    # Even if the player camps in one scene and never triggers the
    # scene-change branch, the world should still advance off-screen
    # after this many turns. Acts as the "world keeps moving" floor
    # so the antagonist's plot doesn't go to sleep just because the
    # player stays in the courtyard. 15 is the v11 default — long
    # enough that idle turns don't burn the budget, short enough that
    # camping doesn't make the world feel frozen.
    tick_stagnation_max: int = 15
    # When True, the tick pass also fires on scene change (in addition to
    # the stagnation counter). Scene changes happen frequently in play,
    # so even with the cooldown above this is the primary tick driver.
    # Default True; flip off to disable scene-change ticks entirely
    # (stagnation-only) for token-budget experiments.
    ticks_on_scene_change: bool = True
    # Master kill switch for the off-stage tick scheduler. When False,
    # `Orchestrator._run_ticks` short-circuits at the top: no eligibility
    # filtering, no agent fan-out, no router fan-in, no canonical-event
    # append. `turns_since_last_tick` and `tick_last_scene_id` are NOT
    # touched in disabled mode either, so flipping this back on later
    # resumes the trigger model from wherever it left off rather than
    # firing a backlog. Useful for token-budget runs, isolating on-stage
    # behavior in playtests, and diagnosing whether a behavior originates
    # from on-stage routing or from background world activity.
    ticks_enabled: bool = True
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
    # The creator's binding: populated at /story start by auto-binding
    # the creator to the first is_playable roster character. Kept for
    # single-player convenience (briefing, legacy prompts). Multi-player
    # bindings live in character_bindings below; player_character_id
    # is one of those keys.
    player_name: str = ""
    player_character_id: str = ""
    # Multi-player bindings: character_id -> discord user id (stringified so
    # the JSON is stable across the int-sized Discord ids). A character with
    # no entry here is AI-driven. Updated by /join, /leave, and /story start.
    character_bindings: dict[str, str] = Field(default_factory=dict)
    # Off-stage tick scheduler state.
    #
    # Commit 5 (v11) replaced the old cadence counter with a turns-since-
    # last-tick counter that the scheduler resets only on a real fire.
    # Trigger model lives in `Orchestrator._run_ticks`; settings are on
    # `SessionSettings.tick_scene_change_cooldown` (gates scene-change
    # fires) and `tick_stagnation_max` (forces a fire after N idle
    # turns). `tick_last_scene_id` still tracks the actor's previous
    # post-beat scene so the scheduler can detect a change.
    turns_since_last_tick: int = 0
    tick_last_scene_id: str = ""
    # DEPRECATED (Commit 5): pre-v11 cadence-based tick scheduler.
    # `tick_turn_counter` is no longer incremented; `tick_cadence` is
    # no longer read. Kept on the schema for one release so old saves
    # written before Commit 5 round-trip cleanly and `/inspect` doesn't
    # show a missing field. Safe to drop in a future migration once
    # no live save references them.
    tick_turn_counter: int = 0
    tick_cadence: int = 10
    # One-slot rolling buffer for the router's missing-narrator-context
    # problem. A Haiku summarizer produces a terse delta note at end of
    # turn N describing what the narrator rendered that the router
    # needs to know. That note is consumed by turn N+1's router call
    # (embedded in its user message, which then archives into
    # session_conversation) and cleared. Empty on a fresh session.
    pending_recap: str = ""
    # Commit-3 (router-trim): which `world_state.facts` entries have
    # already been surfaced to the router in some prior turn's user
    # message. The per-turn user message now carries
    # `world_facts_delta` — only facts NOT in this set — instead of
    # the full list. Updated by the router-context builder at consume
    # time (atomic with the router LLM call). Importer-seeded facts
    # land here on turn 1 and are never re-surfaced; the rare runtime-
    # added fact lands on whatever turn it was added.
    surfaced_world_facts: list[str] = Field(default_factory=list)
    # Router context-trim bookkeeping: scene/location + present-roster
    # blocks are high-bulk grounding that should be surfaced only once
    # for importer-seeded scenes. Router-created scenes are already in
    # router history via `scenes_created`, so they are never listed here.
    surfaced_router_scene_contexts: list[str] = Field(default_factory=list)
    # Commit-3 (router-trim): one-shot lines describing things the
    # engine applied that the router did NOT itself author — spawn
    # outcomes (Commit 4 will populate router_summary into these),
    # /takeover and /join changes, exotic state mutations the operator
    # injected. Drained into the next router call's "## State Changes
    # Since Your Last Call" block, then cleared. Empty in the common
    # case (back-to-back on-stage routing in the same beat with no
    # external mutation).
    pending_router_state_changes: list[str] = Field(default_factory=list)
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
    """Importer-built scene topology. The runtime "where is X" question
    is answered by the per-character `CharacterRecord.location` field,
    NOT by any field here.

    A pre-v11 `current_scene_id` lived on this model as the importer's
    "world pivot" — the scene the story opened in. It was set ONCE at
    import and never updated by any runtime code path, but every
    router/narrator/agent context-builder read it as if it were the
    current scene; that desync was the root cause of "the LLM thinks
    the player is at the bell tower forever even after they moved."
    The field is gone in v11. Old saves with `current_scene_id` in
    their JSON load cleanly because Pydantic v2's default `extra='ignore'`
    silently drops it; new saves serialize without it.
    """
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
