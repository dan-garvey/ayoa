from __future__ import annotations

from pydantic import BaseModel, Field


class DiceRollDisplay(BaseModel):
    """Runtime UI payload for an already executed d20 roll.

    This is intentionally presentation-shaped rather than prompt-shaped:
    durable mechanics details stay in checkpoint roll transactions, and LLM
    context continues to receive only canonical outcome facts.
    """

    transaction_id: str = ""
    event_id: str = ""
    source: str = ""
    roll_id: str = ""
    actor_id: str = ""
    actor_name: str = ""
    target_id: str = ""
    target_name: str = ""
    label: str = ""
    reason: str = ""
    kind: str = ""
    ability: str = ""
    skill: str = ""
    expression: str = ""
    detail: str = ""
    die_faces: int = 20
    die_values: list[int] = Field(default_factory=list)
    kept_die_values: list[int] = Field(default_factory=list)
    modifier: int = 0
    total: int = 0
    dc: int = 0
    outcome: str = ""
    crit: str = "none"
    damage_total: int = 0
    damage_type: str = ""
    damage_detail: str = ""
    automatic: bool = True


class TurnResponse(BaseModel):
    session_id: str
    checkpoint_id: str = ""
    turn_index: int = 0
    # Back-compat single-POV render. For v11 beats this mirrors
    # `per_player_renders[acting_character_id]` so legacy callers (the
    # Discord bot's default-channel post, the CLI REPL) keep working
    # unchanged. New multi-POV callers should walk `per_player_renders`
    # directly to deliver each player their own prose.
    output_text: str = ""
    # v11: per-POV beat renders, keyed by character_id. Populated by
    # `run_beat`'s fan-out through `Dispatcher.narrator_compose`; one
    # entry per human POV with at least one observed event this beat,
    # whether local or connected by explicit live perception. Empty when
    # the beat paused mid-Cat-II (see
    # `beat_ended_reason`) or nobody was present to observe.
    per_player_renders: dict[str, str] = Field(default_factory=dict)
    # v11: why the beat stopped. Values come from `BeatResult.ended_reason`
    # (e.g. "directed_at_player", "cat_ii_resolution", "cat_ii_pending",
    # "max_events_cap", "cascade_exhausted"). Callers can detect the
    # Cat II pending state with `not per_player_renders` +
    # `beat_ended_reason == "cat_ii_pending"`.
    beat_ended_reason: str = ""
    # Runtime-only UI affordances keyed by character_id. A value is the
    # canonical event id that opened that character's possible combat reaction
    # window. The event itself stays in checkpoint state; this map tells
    # frontends which rendered POVs should receive a no-LLM "No reaction"
    # button.
    reaction_prompts: dict[str, str] = Field(default_factory=dict)
    # Runtime-only D&D inventory affordances keyed by character_id. Values are
    # pending loot offer ids this character can inspect and claim.
    loot_prompts: dict[str, list[str]] = Field(default_factory=dict)
    # Runtime-only revision affordances keyed by character_id. Values are open
    # commitment ids whose owning player should revise or continue the activity.
    commitment_revision_prompts: dict[str, list[str]] = Field(default_factory=dict)
    # Runtime-only D&D dice-display payloads for rolls completed while building
    # this response. Frontends may animate these before rendering narrator prose.
    dice_rolls: list[DiceRollDisplay] = Field(default_factory=list)
    # v11-r7a+: pre-turn resolutions. Stale Cat II closure and resumed
    # automated combat can each produce TurnResponse objects that should be
    # delivered before the actor's own /act result so display order matches
    # story time. Empty in the common case.
    pre_turn_resolutions: list["TurnResponse"] = Field(default_factory=list)
    # NOTE: a `debug: DebugPayload | None` field lived here through
    # v11-r7i. The orchestrator never wrote it, every consumer
    # (Discord latency log, CLI status, playtest summary) was guarded
    # by `if response.debug is not None:` and silently no-op'd. v11-r7j
    # murdered the field per the vestigial-field destruction policy
    # in DESIGN.md §19.1. Per-turn diagnostics live in the engine
    # logger (`turn_loop.router[route]` lines) and per-turn checkpoint
    # files.
