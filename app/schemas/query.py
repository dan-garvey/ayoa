"""Player-facing shape returned by `/query`.

The live `/query` path is router-backed: it enters the normal turn loop
as a private OOC clarification, creates a canonical observable fact for
the asking POV, and returns the narrator render here for Discord/CLI
delivery.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class QueryResponse(BaseModel):
    """Player-facing `/query` response.

    `answer` is what the player sees. `knowledge_gated` remains for
    compatibility and slot/knowledge refusals; the router-backed path
    usually expresses bounded uncertainty inside the rendered answer.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    knowledge_gated: bool = False
    gate_reason: str = ""
