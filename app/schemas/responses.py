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
    world_updates: dict[str, Any] = Field(default_factory=dict)
    latencies: list[PhaseLatency] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    models_used: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    validations: list[dict[str, Any]] = Field(default_factory=list)


class TurnResponse(BaseModel):
    session_id: str
    checkpoint_id: str = ""
    turn_index: int = 0
    output_text: str
    debug: DebugPayload | None = None
