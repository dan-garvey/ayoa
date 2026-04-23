from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PhaseLatency(BaseModel):
    """Latency + token usage for a single pipeline phase."""
    phase: str
    duration_ms: float
    model: str = ""
    # Token usage for this phase's LLM call. Cache metrics let you verify
    # turn-over-turn that the prior conversation is being read from cache
    # rather than re-processed. Multiple LLM calls in a single phase
    # (e.g. agent fan-out) are summed.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    # Time spent building the prompt (context gathering, template render)
    # before the LLM call fired. Separated from duration_ms so you can
    # spot rendering-heavy turns that weren't caused by slow API time.
    # Summed across calls the same way as token counts.
    prompt_render_ms: float = 0.0


class DebugPayload(BaseModel):
    canonical_event: dict[str, Any] = Field(default_factory=dict)
    # Full router output (observers, spawns, dormant/cull, roster_moves).
    # Historically this was a separate "discriminator" pass — hence the
    # legacy key name, retained to keep downstream log consumers working.
    router_output: dict[str, Any] = Field(default_factory=dict)
    agent_outputs: list[dict[str, Any]] = Field(default_factory=list)
    latencies: list[PhaseLatency] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    models_used: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    validations: list[dict[str, Any]] = Field(default_factory=list)


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
    # entry per in-scene human with at least one observed event this
    # beat. Empty when the beat paused mid-Cat-II (see
    # `beat_ended_reason`) or nobody was present to observe.
    per_player_renders: dict[str, str] = Field(default_factory=dict)
    # v11: why the beat stopped. Values come from `BeatResult.ended_reason`
    # (e.g. "directed_at_player", "cat_ii_resolution", "cat_ii_pending",
    # "max_events_cap", "cascade_exhausted"). Callers can detect the
    # Cat II pending state with `not per_player_renders` +
    # `beat_ended_reason == "cat_ii_pending"`.
    beat_ended_reason: str = ""
    # v11-r7a: pre-turn AFK-sweep resolutions. When the per-session lock
    # holder runs `sweep_stale_pins`, each event the sweep fills is
    # closed by `Orchestrator.resolve_cat_ii`, producing a TurnResponse
    # of its own. Those responses are appended here so the frontend can
    # fan their per-POV renders out before showing the actor their own
    # /act result. Empty in the common case (no stale pins).
    pre_turn_resolutions: list["TurnResponse"] = Field(default_factory=list)
    debug: DebugPayload | None = None
