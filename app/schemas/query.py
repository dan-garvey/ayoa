"""Schema for the out-of-character /query handler.

A `/query` is a read-only player consultation: the user asks a question
they cannot ask in fiction ("what do I see right now?", "what was that
NPC's name?", "what day is it?", "have I met this person?") and the
engine consults the asking character's POV envelope to either answer
concisely or refuse in-fiction when the character can't plausibly know.

Read-only by contract: no checkpoint mutation, no broadcast, no turn
advancement. The handler runs as its own short LLM call so /act is
never blocked by an OOC question.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class QueryResponse(BaseModel):
    """Output of the query handler.

    `answer` is what the player sees — second person, OOC, typically
    1-3 sentences. `knowledge_gated` is True when the handler refused
    on knowledge grounds; in that case `answer` is an in-fiction excuse
    ("you can't see — you're blindfolded") and `gate_reason` carries
    a short tag for telemetry (e.g. "blindfolded", "never_met",
    "unconscious", "off_screen", "hidden_info").
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    knowledge_gated: bool = False
    gate_reason: str = ""
