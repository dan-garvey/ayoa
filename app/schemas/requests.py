from __future__ import annotations

from pydantic import BaseModel


class TurnRequest(BaseModel):
    session_id: str
    checkpoint_id: str | None = None
    user_input: str
    # Which character is taking this turn. Empty falls back to
    # session.player_character_id so legacy single-player call sites keep
    # working without change. The orchestrator resolves this against the
    # roster each turn for the router/narrator turn context.
    acting_character_id: str = ""
    stream: bool = False
    # NOTE: `debug: bool` and `debug_flags: DebugFlags` lived here through
    # v11-r7i. They were the on/off switch for the also-murdered
    # `TurnResponse.debug` payload — and since nothing in the
    # orchestrator ever populated that payload, neither flag had any
    # observable effect on the turn pipeline. Both gone in v11-r7j per
    # the vestigial-field destruction policy in CLAUDE.md. Old turn
    # requests on the wire that still set `debug=true` load cleanly:
    # Pydantic's default `extra='ignore'` silently drops unknown keys.
