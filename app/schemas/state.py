from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.characters import CharacterAgentTier, CharacterRecord
from app.schemas.content import ContentPackState
from app.schemas.dnd_inventory import DndLootOffer
from app.schemas.dnd_spatial import DndBattleMapSeed, DndBattleMapState


class ModelConfig(BaseModel):
    event_router: str = "gpt-5.6-terra"
    narrator: str = "gpt-5.6-terra"
    image_director: str = "gpt-5-mini"
    dnd_combat_manager: str = "gpt-5-mini"
    agent_default: str = "gpt-5.6-luna"
    agent_standard: str = "gpt-5.6-luna"
    agent_convenience: str = "gpt-5.6-luna"
    character_manager: str = "gpt-5.6-luna"


class SlotEntry(BaseModel):
    """v11: one participant in the session's active act slot. A beat may
    have zero (free), one (initiator or single Cat II responder), or many
    entries (multi-character Cat II). The reason determines what /act
    from that user is allowed to do.
    """
    reason: str  # "initiator" | "cat_ii_responder" | "cat_ii_roll" | "combat_reaction" | "combat_blocked"
    # When reason is tied to an open Cat II event, the open-event id this
    # slot is pinned to. None for initiator entries.
    cat_ii_event_id: str | None = None
    # Neutral trigger link for slots that are tied to a closed canonical
    # event rather than an open Cat II event. Currently used by D&D combat
    # reaction prompts.
    trigger_event_id: str | None = None
    # ISO-8601 timestamp of when the slot was claimed. Read by
    # `sweep_stale_combat_reaction_pins` to auto-pass `combat_reaction`
    # pins whose human never answered (so AFK reactors cannot wedge
    # initiative). Stamped by every slot-claim helper.
    claimed_at: str = ""


class PendingNarratorRender(BaseModel):
    """A closed beat whose upstream state is durable, but whose POV render
    has not completed yet.

    This lets a narrator-provider failure be retried without replaying the
    router and character-agent calls that already mutated canonical state.
    """

    ended_reason: str = ""
    events_closed: int = 0
    event_actor_ids: list[str] = Field(default_factory=list)
    acting_player_id: str = ""
    acting_player_input: str = ""
    release_slots: bool = True
    force_partial: bool = False
    suppress_reaction_prompts: bool = False
    soft_handoff_candidate: bool = False
    handoff_event_id: str = ""
    roll_keys_before: list[tuple[str, str]] = Field(default_factory=list)
    commitment_revision_character_id: str = ""
    commitment_revision_id: str = ""
    commitment_revision_trigger_id: str = ""
    # Private runtime payload for a rejected render candidate. These authored
    # records are deliberately not active roster entries until a retry render
    # is accepted, but must survive process restart so character generation is
    # never repeated for the already-closed canonical spawn event.
    pending_spawn_records: list[CharacterRecord] = Field(default_factory=list)
    pending_spawn_introductions: dict[str, list[str]] = Field(
        default_factory=dict
    )


class OpenCommitment(BaseModel):
    """Private checkpoint record for an interruptible ongoing activity.

    Open commitments are not canonical observable facts. They summarize an
    activity that can be advanced, interrupted, or resolved by later events;
    only those later canonical events become narrator-visible.
    """

    commitment_id: str
    actor_ids: list[str] = Field(default_factory=list)
    description: str = ""
    trigger_event_id: str = ""
    started_at_s: int = 0
    expected_end_s: int = 0
    max_end_s: int = 0
    location_label: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "OpenCommitment":
        self.commitment_id = self.commitment_id.strip()
        self.actor_ids = [
            cid.strip() for cid in dict.fromkeys(self.actor_ids) if cid.strip()
        ]
        self.description = self.description.strip()
        self.trigger_event_id = self.trigger_event_id.strip()
        self.location_label = self.location_label.strip()
        if self.started_at_s < 0:
            self.started_at_s = 0
        if self.expected_end_s < self.started_at_s:
            self.expected_end_s = self.started_at_s
        if self.max_end_s < self.expected_end_s:
            self.max_end_s = self.expected_end_s
        return self


class CommitmentRevisionPrompt(BaseModel):
    """Private player prompt to revise or continue an open commitment."""

    character_id: str
    commitment_id: str
    trigger_event_id: str = ""
    observed_at_s: int = 0
    reason: str = ""
    previous_description: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "CommitmentRevisionPrompt":
        self.character_id = self.character_id.strip()
        self.commitment_id = self.commitment_id.strip()
        self.trigger_event_id = self.trigger_event_id.strip()
        self.reason = self.reason.strip()
        self.previous_description = self.previous_description.strip()
        if self.observed_at_s < 0:
            self.observed_at_s = 0
        return self


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


class CatIIRollDamageAdjustmentRecord(BaseModel):
    """One deterministic damage adjustment applied to a raw damage roll."""

    source: str = ""
    kind: str = ""
    damage_type: str = ""
    amount_before: int = 0
    amount_after: int = 0
    reason: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "CatIIRollDamageAdjustmentRecord":
        self.source = self.source.strip().lower()
        self.kind = self.kind.strip().lower()
        self.damage_type = self.damage_type.strip().lower()
        self.reason = self.reason.strip()
        if self.amount_before < 0:
            self.amount_before = 0
        if self.amount_after < 0:
            self.amount_after = 0
        return self


class CatIIRollDamageComponentRecord(BaseModel):
    """One typed damage component from a D&D attack damage roll."""

    expression: str = ""
    detail: str = ""
    damage_type: str = ""
    raw_amount: int = 0
    amount: int = 0
    adjustments: list[CatIIRollDamageAdjustmentRecord] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _clean(self) -> "CatIIRollDamageComponentRecord":
        self.expression = self.expression.strip()
        self.detail = self.detail.strip()
        self.damage_type = self.damage_type.strip().lower()
        if self.raw_amount < 0:
            self.raw_amount = 0
        if self.amount < 0:
            self.amount = 0
        return self


class CatIIRollDamageRecord(BaseModel):
    """Structured D&D damage attached to a roll transaction.

    The text ledger remains useful for inspection, but HP mutation reads this
    typed record so damage application is not coupled to ledger prose.
    """

    roll_id: str
    target_id: str
    raw_amount: int = 0
    amount: int = 0
    damage_type: str = ""
    adjustments: list[CatIIRollDamageAdjustmentRecord] = Field(
        default_factory=list
    )
    components: list[CatIIRollDamageComponentRecord] = Field(
        default_factory=list
    )
    expression: str = ""
    detail: str = ""
    target_hp_before: int = 0
    target_hp_after: int = 0
    target_hp_max: int = 0
    target_temp_hp_before: int = 0
    target_temp_hp_after: int = 0
    target_defeat_state_after: str = ""
    applied: bool = False

    @model_validator(mode="before")
    @classmethod
    def _migrate_raw_amount(cls, data: Any) -> Any:
        if isinstance(data, dict) and "raw_amount" not in data:
            data = dict(data)
            data["raw_amount"] = data.get("amount", 0)
        return data

    @model_validator(mode="after")
    def _clean_damage_snapshot(self) -> "CatIIRollDamageRecord":
        self.damage_type = self.damage_type.strip().lower()
        self.expression = self.expression.strip()
        self.detail = self.detail.strip()
        self.target_defeat_state_after = self.target_defeat_state_after.strip()
        if self.raw_amount < 0:
            self.raw_amount = 0
        if self.amount < 0:
            self.amount = 0
        if self.target_hp_before < 0:
            self.target_hp_before = 0
        if self.target_hp_after < 0:
            self.target_hp_after = 0
        if self.target_hp_max < 0:
            self.target_hp_max = 0
        if self.target_temp_hp_before < 0:
            self.target_temp_hp_before = 0
        if self.target_temp_hp_after < 0:
            self.target_temp_hp_after = 0
        return self


class CatIIRollHealingRecord(BaseModel):
    """Structured D&D healing attached to a roll transaction."""

    roll_id: str
    target_id: str
    raw_amount: int = 0
    amount: int = 0
    expression: str = ""
    detail: str = ""
    target_hp_before: int = 0
    target_hp_after: int = 0
    target_hp_max: int = 0
    target_temp_hp_before: int = 0
    target_temp_hp_after: int = 0
    target_defeat_state_after: str = ""
    applied: bool = False

    @model_validator(mode="before")
    @classmethod
    def _migrate_raw_amount(cls, data: Any) -> Any:
        if isinstance(data, dict) and "raw_amount" not in data:
            data = dict(data)
            data["raw_amount"] = data.get("amount", 0)
        return data

    @model_validator(mode="after")
    def _clean_healing_snapshot(self) -> "CatIIRollHealingRecord":
        self.expression = self.expression.strip()
        self.detail = self.detail.strip()
        self.target_defeat_state_after = self.target_defeat_state_after.strip()
        if self.raw_amount < 0:
            self.raw_amount = 0
        if self.amount < 0:
            self.amount = 0
        if self.target_hp_before < 0:
            self.target_hp_before = 0
        if self.target_hp_after < 0:
            self.target_hp_after = 0
        if self.target_hp_max < 0:
            self.target_hp_max = 0
        if self.target_temp_hp_before < 0:
            self.target_temp_hp_before = 0
        if self.target_temp_hp_after < 0:
            self.target_temp_hp_after = 0
        return self


class CatIIRollResourceSpendRecord(BaseModel):
    """One deterministic resource spend attached to a D&D roll transaction."""

    actor_id: str = ""
    resource_id: str = ""
    source_id: str = ""
    amount: int = 0
    applied: bool = False
    reason: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "CatIIRollResourceSpendRecord":
        self.actor_id = self.actor_id.strip()
        self.resource_id = self.resource_id.strip()
        self.source_id = self.source_id.strip()
        self.reason = self.reason.strip()
        if self.amount < 0:
            self.amount = 0
        return self


class CatIIRollTransaction(BaseModel):
    """Checkpoint-persistent D&D Cat II roll transaction.

    Transactions stay in the checkpoint after their open Cat II event closes so
    rewind/debug can reconstruct the roll plan and dice ledger without putting
    those mechanics details into router or narrator rolling conversations.
    """

    transaction_id: str
    event_id: str
    source: Literal["cat_ii", "combat"] = "cat_ii"
    actor_id: str = ""
    intention: str = ""
    ruleset_id: str = ""
    status: str = "planning"
    plan: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    no_roll_reason: str = ""
    rolls: list[CatIIRollRecord] = Field(default_factory=list)
    ledger_lines: list[str] = Field(default_factory=list)
    damage_records: list[CatIIRollDamageRecord] = Field(default_factory=list)
    healing_records: list[CatIIRollHealingRecord] = Field(default_factory=list)
    resource_spends: list[CatIIRollResourceSpendRecord] = Field(
        default_factory=list
    )
    final_event_id: str = ""
    created_at: str = ""
    updated_at: str = ""


DefeatState = Literal["active", "down", "stable", "dead", "defeated"]
DndEffectDurationKind = Literal[
    "rounds",
    "minutes",
    "hours",
    "days",
    "special",
    "until_removed",
]
DndEffectSaveTiming = Literal["start_of_turn", "end_of_turn"]


def _normalized_recurring_save_end(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"success", "succeed", "succeeds", "successful", "save_success"}:
        return "success"
    if text in {"failure", "fail", "fails", "failed", "save_failure"}:
        return "failure"
    return "success"


class DndEffectRecurringSave(BaseModel):
    """A router-authored recurring save attached to a D&D runtime effect.

    The router decides that an effect has a recurring save and supplies the
    spell-specific timing/DC. The combat engine owns the durable countdown and
    executes the save when that timing arrives.
    """

    ability: str = ""
    dc: int = 0
    timing: DndEffectSaveTiming = "end_of_turn"
    ends_on: str = "success"
    repeat: bool = True

    @model_validator(mode="after")
    def _clean(self) -> "DndEffectRecurringSave":
        self.ability = self.ability.strip().lower()
        if self.ability not in {"str", "dex", "con", "int", "wis", "cha"}:
            self.ability = ""
        if self.dc < 0:
            self.dc = 0
        self.ends_on = _normalized_recurring_save_end(self.ends_on)
        return self


class DndRuntimeEffect(BaseModel):
    """D&D-adapter runtime state for a timed or sustained effect."""

    effect_id: str = ""
    name: str = ""
    slug: str = ""
    source_type: str = "custom"
    source_id: str = ""
    originator_id: str = ""
    target_id: str = ""
    conditions: list[str] = Field(default_factory=list)
    concentration: bool = False
    duration_kind: DndEffectDurationKind = "until_removed"
    duration_amount: int = 0
    remaining_rounds: int = 0
    duration_text: str = ""
    break_triggers: list[str] = Field(default_factory=list)
    recurring_save: DndEffectRecurringSave | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "DndRuntimeEffect":
        self.effect_id = self.effect_id.strip()
        self.name = self.name.strip()
        self.slug = self.slug.strip().lower()
        self.source_type = self.source_type.strip().lower() or "custom"
        self.source_id = self.source_id.strip()
        self.originator_id = self.originator_id.strip()
        self.target_id = self.target_id.strip()
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
        return self


class DndExperienceAwardDisplay(BaseModel):
    """Player-facing D&D XP award notice queued by the combat adapter."""

    character_id: str = ""
    character_name: str = ""
    amount: int = 0
    source: str = ""
    experience_points: int = 0
    total_level: int = 0
    eligible_level: int = 0
    next_level: int = 0
    xp_to_next_level: int = 0


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
    active_effects: list[DndRuntimeEffect] = Field(default_factory=list)
    defeat_state: DefeatState = "active"
    death_save_successes: int = 0
    death_save_failures: int = 0
    removed: bool = False
    pending_initiating_action: str = ""
    pending_initiating_event_id: str = ""
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_defeated(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if not data.get("defeated"):
            return data
        state = str(data.get("defeat_state") or "")
        if state in {"", "active"}:
            data = dict(data)
            data["defeat_state"] = "defeated"
        return data


class DndRouterObservedFact(BaseModel):
    """Narrative continuity from active D&D combat worth surfacing later."""

    model_config = ConfigDict(extra="forbid")

    fact: str
    salience: str = "notable"
    reason: str

    @model_validator(mode="after")
    def _clean(self) -> "DndRouterObservedFact":
        self.fact = self.fact.strip()
        self.salience = self.salience.strip().lower() or "notable"
        self.reason = self.reason.strip()
        if not self.fact:
            raise ValueError("router observed fact requires fact")
        if not self.reason:
            raise ValueError("router observed fact requires reason")
        return self


class DndCombatState(BaseModel):
    """Active D&D combat state stored directly on SessionState."""

    combat_id: str = "combat"
    status: str = "active"
    round_number: int = 1
    turn_index: int = 0
    combatants: list[DndCombatantState] = Field(default_factory=list)
    audit_lines: list[str] = Field(default_factory=list)
    # Combatant ids whose defeat XP has already been paid out. Kept on
    # combat state so repeated damage, overkill, or stale roll finalization
    # cannot duplicate rewards.
    xp_awarded_combatant_ids: list[str] = Field(default_factory=list)
    started_at_turn_index: int = 0
    ended_at_turn_index: int | None = None
    # When a combatant's completed turn opens reaction prompts, initiative
    # advancement is delayed until those runtime slots clear.
    pending_advance_actor_id: str = ""
    pending_visible_facts: list[str] = Field(default_factory=list)
    pending_experience_awards: list[DndExperienceAwardDisplay] = Field(
        default_factory=list
    )
    # Adapter-owned tactical map for active D&D combat. Generic narrative
    # location state remains an opaque label and has no topology.
    battle_map: DndBattleMapState | None = None
    # Combat-manager selected narrative continuity to carry back to generic
    # routing when active initiative ends. Routine damage, ammo, and unnamed
    # defeat bookkeeping stay out of this list.
    router_observed_facts: list[DndRouterObservedFact] = Field(default_factory=list)

    @field_validator("battle_map", mode="before")
    @classmethod
    def _coerce_battle_map(
        cls,
        value: DndBattleMapState | DndBattleMapSeed | dict[str, Any] | None,
    ) -> DndBattleMapState | dict[str, Any] | None:
        if value is None or isinstance(value, DndBattleMapState):
            return value
        if isinstance(value, DndBattleMapSeed):
            return DndBattleMapState.model_validate(value.model_dump())
        return value


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
    # Fictional time when this POV has received the visible facts in this
    # event. Narrator composition sorts by this rather than by append order.
    visible_at_s: int = 0
    # Stable event-log order tie-breaker for simultaneous visible events.
    event_sequence: int = 0
    # Character id -> character-owned presentation key at this exact visible
    # event. This stays in checkpoint/runtime state and never enters narrator
    # input or player-visible text.
    sprite_variant_keys_by_character_id: dict[str, str] = Field(
        default_factory=dict
    )


class SessionSettings(BaseModel):
    """User-tunable experimental toggles exposed via /settings.

    Kept separate from the static SessionConfig (models, narrative_rules)
    so the command surface can introspect exactly the
    fields meant for live tuning. Add new toggles here and they become
    automatically available in /settings list / set.
    """
    # v11: hard cap on how many canonical events the router may chain
    # inside a single beat before the orchestrator forces render + slot
    # release. Prevents runaway event growth while allowing large
    # ensemble beats to breathe.
    max_events_per_beat: int = 40
    # v11: hard cap on successful/attempted agent cascade handoffs inside
    # one beat. This is distinct from canonical events because Cat II
    # setup/resolution and router continuations can add events without a
    # normal NPC handoff, while a long NPC-to-NPC chain needs its own
    # budget guard.
    max_agent_cascades_per_beat: int = 35
    # v11: maximum seconds a Cat II pin may hold a human before the
    # orchestrator's sweep auto-resolves them as "stays out." Default
    # 24 hours — long enough that async multiplayer (play over a day)
    # doesn't time out, short enough that abandoned sessions eventually
    # release. Set to 0 to disable the sweep entirely. The auto-resolve
    # is visible: the rendered outcome notes that the player did not
    # act, so everyone sees the fallback happened.
    cat_ii_human_timeout_seconds: int = 24 * 60 * 60
    # Ruleset/rules-arbitration toggles. Defaults preserve the existing
    # narrative behavior.
    ruleset_id: str = "narrative"
    # D&D player roll handling. Agent/NPC rolls are always automatic. Human
    # player rolls are automatic by default for playtest speed, or can pause
    # for Discord UI when set to "interactive".
    player_roll_mode: str = "auto"
    # Player-facing narration surface. ``prose`` preserves the original
    # rules-neutral text delivery. ``visual_novel`` asks the narrator for
    # ordered narration/dialogue pages and lets frontends render those pages
    # over one noncanonical scene plate.
    presentation_mode: Literal["prose", "visual_novel"] = "prose"
    # Imported-content manager cadence. The manager sees recent canonical
    # facts plus compact entity knowledge, so calling it every route cycle is
    # usually redundant. Deterministic pending-content lookup still runs on
    # skipped cycles.
    content_manager_refresh_interval: int = 3
class SessionConfig(BaseModel):
    models: ModelConfig = Field(default_factory=ModelConfig)
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
    # Durable relative story clock. Per-character clocks live on
    # CharacterRecord; this is the maximum known checkpoint time.
    leading_at_s: int = 0
    # Viewer character_id -> character_ids whose stable first-look loadout has
    # already been delivered to that viewer. Overflow targets stay absent so
    # future focused events can introduce them naturally.
    visual_introductions: dict[str, list[str]] = Field(default_factory=dict)
    # One-shot lines for durable state mutations the engine performed outside
    # router-authored canonical events. Drained into an optional
    # engine_state_updates block on the next fresh router call. Router-authored
    # spawn/dormant/cull/location/commitment/time changes belong in compact
    # router history instead, not here.
    pending_engine_state_updates: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
    # D&D-adapter pending loot/reward choices. Generic narrative sessions leave
    # this empty; item/currency claims mutate character mechanics overlays.
    dnd_inventory_offers: list[DndLootOffer] = Field(default_factory=list)
    # Generic adapter/content-pack lookup state. Pack-specific meaning stays in
    # the content scaffold; the narrative engine only persists the slice.
    content_state: dict[str, ContentPackState] = Field(default_factory=dict)
    # Private imported-content preflight cadence state. These counters are not
    # model inputs; they only decide when the content manager gets called.
    content_manager_preflight_cycle: int = 0
    content_manager_last_run_cycle: int = -1
    # Private long-action state. These records are derived from router-authored
    # canonical events and are not narrator-visible facts.
    open_commitments: list[OpenCommitment] = Field(default_factory=list)
    # Non-blocking player revision prompts created when a visible event changes
    # the scene around an open commitment.
    pending_commitment_revisions: dict[str, CommitmentRevisionPrompt] = Field(
        default_factory=dict
    )
    # v11: per-player queue of canonical events awaiting render. Keyed by
    # character_id (a human's bound character). Cleared after each render
    # fires. An agent's "render buffer" is just its observation context
    # on the next intend() call — no separate store here.
    render_buffers: dict[str, list[RenderBufferEntry]] = Field(default_factory=dict)
    # Non-empty only while a user-visible turn has completed router/agent
    # work and is awaiting a successful narrator render. The latest
    # checkpoint can resume the render without rerunning upstream LLM calls.
    pending_narrator_render: PendingNarratorRender | None = None


class PhysicsRuleset(BaseModel):
    strength_limits: str = "human_baseline"
    magic_enabled: bool = False


class StorySetting(BaseModel):
    """Genre, era, and tone metadata — used to ground character genesis and narrator voice."""
    title: str = ""
    recommended_players: str = ""
    play_guidance: str = ""
    discoverable: bool = True
    genre: str = ""
    era: str = ""
    tone: str = ""
    premise: str = ""
    # Player-safe, image-model-facing art direction. This is intentionally
    # separate from narrative_rules, which can contain nonvisual engine rules.
    visual_style: str = ""


class CharacterGenerationGuidance(BaseModel):
    """Story-authored actor and presentation guidance for one generation tier.

    These fields describe a budget, not generic engine rankings. A story may
    use tiers for social station, dramatic importance, supernatural maturity,
    rarity, or any other authored ladder. Omitting the guidance leaves ordinary
    character generation unchanged.
    """

    # One deliberately unstructured instruction for the amount and kind of
    # actor-owned material warranted at this rung. It must not become a fixed
    # checklist: sparse facts and uneven people are valid outcomes.
    actor_fact_guidance: str = ""
    public_visual_detail: str = ""
    loadout_detail: str = ""
    visual_salience: str = ""
    # Contextual casting/art direction, including any intended beauty,
    # age-presentation, or deliberate exception. This is story presentation
    # guidance, not a universal judgment about people or bodies.
    presentation_guidance: str = ""


class KnowledgeTier(BaseModel):
    """One rung of an authored knowledge ladder for router-spawned characters.

    A character spawned at knowledge_tier N is authored (at character_gen) to
    know the cumulative budget of tiers 1..N: the personal depth of their own
    remembered life plus the shared world/plot knowledge unlocked at that rung.
    A rung may also carry a non-cumulative `generation_guidance` target for how
    fully that tier should be authored and presented.
    High rungs may name otherwise-hidden plot facts; the assembled budget
    reaches only the spawned character's own actor facts, never the narrator
    or lower-tier agents.
    `agent_tier`, when set, overrides the default spawn agent tier for this
    rung, so a plot-bearing high-tier summon can be voiced by a stronger model
    than disposable fodder. An empty ladder leaves spawn behavior unchanged.
    """
    tier: int
    label: str = ""
    personal_depth: str = ""
    world_knowledge: str = ""
    generation_guidance: CharacterGenerationGuidance | None = None
    agent_tier: CharacterAgentTier | None = None


class OpeningPolicy(BaseModel):
    """Story-authored constraints for the router's canonical opening.

    The generic runtime does not interpret the story prose. It only enforces
    the explicit spawn authority bit; the router uses ``context`` together
    with the live human bindings to author the opening event.
    """

    allow_spawns: bool = False
    context: str = ""

    @model_validator(mode="after")
    def _validate_spawn_authority(self) -> "OpeningPolicy":
        self.context = self.context.strip()
        if self.allow_spawns and not self.context:
            raise ValueError(
                "opening allow_spawns=true requires authored opening context"
            )
        return self



class WorldState(BaseModel):
    facts: list[str] = Field(default_factory=list)
    physics_ruleset: PhysicsRuleset = Field(default_factory=PhysicsRuleset)
    global_flags: dict[str, Any] = Field(default_factory=dict)
    setting: StorySetting = Field(default_factory=StorySetting)
    # Long-form world lore: history, factions, laws, magic systems, etc.
    lore: str = ""
    # Hidden lore/facts — available to the event router and agents for authentic
    # reactions, but NEVER shown to the narrator or the player. These contain
    # spoilers, conspiracy details, and secrets to be discovered through play.
    hidden_lore: str = ""
    hidden_facts: list[str] = Field(default_factory=list)
    # Optional router-only opening contract. Missing/false spawn authority keeps
    # the generic no-opening-spawns behavior.
    opening: OpeningPolicy | None = None
    # Optional authored knowledge ladder for router-spawned characters. When
    # present, a spawn's seed.knowledge_tier selects cumulative knowledge plus
    # the target rung's optional generation-depth/presentation budget, and may
    # set the spawn's agent tier. Empty (the default) for stories that do not
    # gate character generation by tier.
    knowledge_tiers: list[KnowledgeTier] = Field(default_factory=list)
