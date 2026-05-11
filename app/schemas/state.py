from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    event_router: str = "gpt-5.2"
    narrator: str = "gpt-5.1"
    discriminator: str = "gpt-5.2"
    agent_default: str = "gpt-5.1"


class SlotEntry(BaseModel):
    """v11: one participant in the session's active act slot. A beat may
    have zero (free), one (initiator or single Cat II responder), or many
    entries (multi-character Cat II). The reason determines what /act
    from that user is allowed to do.
    """
    reason: str  # "initiator" | "cat_ii_responder" | "cat_ii_roll" | "combat_reaction"
    # When reason is tied to an open Cat II event, the open-event id this
    # slot is pinned to. None for initiator entries.
    cat_ii_event_id: str | None = None
    # Neutral trigger link for slots that are tied to a closed canonical
    # event rather than an open Cat II event. Currently used by D&D combat
    # reaction prompts.
    trigger_event_id: str | None = None
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
    # Opening-event context preserved for optional mechanics subflows that
    # specialize the router's final Cat II resolution. Legacy/manual Cat II
    # events may leave these empty; resolution then falls back to the
    # participants as observers.
    opening_event_id: str = ""
    opening_observer_ids: list[str] = Field(default_factory=list)
    opening_observable_facts: list[str] = Field(default_factory=list)
    # Links to the durable roll transaction when D&D Cat II resolution has
    # planned or executed dice for this event. The transaction itself lives on
    # SessionState so it survives after the open Cat II event closes.
    roll_transaction_id: str = ""
    opened_at: str = ""


class CatIIRollRecord(BaseModel):
    """Durable audit record for one planned Cat II roll.

    This is checkpoint state, not prompt history. The router may see a compact
    projection during the immediate D&D finalize call, but normal future LLM
    context receives only the canonical outcome facts.
    """

    roll_id: str
    actor_id: str
    actor_control: str = "agent"  # "agent" | "player"
    status: str = "pending"  # "pending" | "completed"
    request: dict[str, Any] = Field(default_factory=dict)
    modifier: int = 0
    label: str = ""
    reason: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    completed_by_user_id: str = ""
    completed_at: str = ""


class CatIIRollTransaction(BaseModel):
    """Checkpoint-persistent D&D Cat II roll transaction.

    Transactions stay in the checkpoint after their open Cat II event closes so
    rewind/debug can reconstruct the roll plan and dice ledger without putting
    those mechanics details into router or narrator rolling conversations.
    """

    transaction_id: str
    event_id: str
    ruleset_id: str = ""
    status: str = "planning"
    plan: dict[str, Any] = Field(default_factory=dict)
    no_roll_reason: str = ""
    rolls: list[CatIIRollRecord] = Field(default_factory=list)
    ledger_lines: list[str] = Field(default_factory=list)
    final_event_id: str = ""
    created_at: str = ""
    updated_at: str = ""


class DndCombatantState(BaseModel):
    """Checkpoint-persistent snapshot of one active D&D combatant.

    HP and initiative are copied from CharacterRecord mechanics when combat
    starts so turn order and damage remain stable until explicit combat-engine
    mutations change them.
    """

    combatant_id: str
    character_id: str = ""
    name: str = ""
    player_controlled: bool = False
    armor_class: int = 10
    hit_points_current: int = 0
    hit_points_max: int = 0
    hit_points_temporary: int = 0
    initiative_modifier: int = 0
    initiative_advantage_state: str = "normal"
    initiative_roll: int = 0
    initiative_total: int = 0
    initiative_detail: str = ""
    initiative_order: int = 0
    reaction_available: bool = True
    conditions: list[str] = Field(default_factory=list)
    defeated: bool = False
    removed: bool = False
    notes: str = ""


class DndCombatState(BaseModel):
    """Active D&D combat state stored directly on SessionState."""

    combat_id: str = "combat"
    status: str = "active"
    round_number: int = 1
    turn_index: int = 0
    combatants: list[DndCombatantState] = Field(default_factory=list)
    audit_lines: list[str] = Field(default_factory=list)
    started_at_turn_index: int = 0
    ended_at_turn_index: int | None = None
    # When a combatant's completed turn opens reaction prompts, initiative
    # advancement is delayed until those runtime slots clear.
    pending_advance_actor_id: str = ""


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
    # Concurrency cap for the off-stage tick pass. Ticks are independent
    # so the only cost of raising this is API rate usage; lowering
    # trades latency for safety on small quotas. Hard-capped engine-side
    # by `app.engine.orchestrator.TICK_CONCURRENCY_HARD_CAP` regardless
    # of what's configured here.
    tick_concurrency: int = 4
    # Commit 5: hard ceiling on consecutive turns with NO tick fire.
    # Even if the player camps in one conversation, the world should
    # still advance off-screen after this many turns. Acts as the
    # "world keeps moving" floor
    # so the antagonist's plot doesn't go to sleep just because the
    # player stays in the courtyard. 15 is the v11 default — long
    # enough that idle turns don't burn the budget, short enough that
    # camping doesn't make the world feel frozen.
    tick_stagnation_max: int = 15
    # Master kill switch for the off-stage tick scheduler. When False,
    # `Orchestrator._run_ticks` short-circuits at the top: no eligibility
    # filtering, no agent fan-out, no router fan-in, no canonical-event
    # append. `turns_since_last_tick` is NOT touched in disabled mode
    # either, so flipping this back on later
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
    # Ruleset/rules-arbitration toggles. Defaults preserve the existing
    # narrative router-owned Cat II behavior.
    ruleset_id: str = "narrative"
    cat_ii_resolution_mode: str = "router"
    # D&D player roll handling. Agent/NPC rolls are always automatic. Human
    # player rolls are automatic by default for playtest speed, or can pause
    # for Discord UI when set to "interactive".
    player_roll_mode: str = "auto"


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
    # last-tick counter that the scheduler resets only on a real fire.
    # Trigger model lives in `Orchestrator._run_ticks`; `tick_stagnation_max`
    # forces a fire after N idle turns.
    turns_since_last_tick: int = 0
    # DEPRECATED (Commit 5): pre-v11 cadence-based tick scheduler.
    # `tick_turn_counter` is no longer incremented; `tick_cadence` is
    # no longer read. Kept on the schema for one release so old saves
    # written before Commit 5 round-trip cleanly and `/inspect` doesn't
    # show a missing field. Safe to drop in a future migration once
    # no live save references them.
    tick_turn_counter: int = 0
    tick_cadence: int = 10
    # Commit-3 (router-trim): which `world_state.facts` entries have
    # already been surfaced to the router in some prior turn's user
    # message. The per-turn user message now carries
    # `world_facts_delta` — only facts NOT in this set — instead of
    # the full list. Updated by the router-context builder at consume
    # time (atomic with the router LLM call). Importer-seeded facts
    # land here on turn 1 and are never re-surfaced; the rare runtime-
    # added fact lands on whatever turn it was added.
    surfaced_world_facts: list[str] = Field(default_factory=list)
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

    # v11: beat-pacing state. Keyed by slot-holder character_id. The session
    # has one live beat gate; contextual presence is routed through event
    # observers rather than scene-local lock maps.
    active_act_slots: dict[str, SlotEntry] = Field(default_factory=dict)
    # v11: in-flight Cat II events awaiting responder intentions.
    open_cat_ii_events: list[OpenCatIIEvent] = Field(default_factory=list)
    # Durable mechanics audit for D&D Cat II resolution. These records are
    # intentionally not included in normal LLM rolling histories.
    cat_ii_roll_transactions: list[CatIIRollTransaction] = Field(default_factory=list)
    # Checkpoint-persistent active D&D combat state. Combat command wiring is
    # intentionally separate; engine helpers mutate this snapshot directly.
    active_combat: DndCombatState | None = None
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
    story_time: datetime = Field(default_factory=datetime.utcnow)
    turn_count: int = 0


class LocationState(BaseModel):
    """No runtime scene topology.

    CharacterRecord.location remains as an opaque continuity label, but
    perception and participation are determined by router-authored
    observer lists and fact-level visibility packets.
    """
    pass


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
