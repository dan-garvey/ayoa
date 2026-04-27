from __future__ import annotations

from pydantic import BaseModel, Field


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
    # v11-r7a: pre-turn AFK-sweep resolutions. When the per-session lock
    # holder runs `sweep_stale_pins`, each event the sweep fills is
    # closed by `Orchestrator.resolve_cat_ii`, producing a TurnResponse
    # of its own. Those responses are appended here so the frontend can
    # fan their per-POV renders out before showing the actor their own
    # /act result. Empty in the common case (no stale pins).
    pre_turn_resolutions: list["TurnResponse"] = Field(default_factory=list)
    # NOTE: a `debug: DebugPayload | None` field lived here through
    # v11-r7i. The orchestrator never wrote it, every consumer
    # (Discord latency log, CLI status, playtest summary) was guarded
    # by `if response.debug is not None:` and silently no-op'd. v11-r7j
    # murdered the field per the vestigial-field destruction policy
    # in CLAUDE.md. Per-turn diagnostics live in the engine logger
    # (`turn_loop.router[route]` lines) and per-turn checkpoint files.
