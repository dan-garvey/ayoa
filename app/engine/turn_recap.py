"""Post-turn summarization — a terse delta note the router reads on the
NEXT turn to close the narrator-context gap.

Design context: the router's own rolling session_conversation contains
only its prior adjudications (canonical_event + observer routing). It
never sees the narrator's final prose. So if the narrator rendered
something state-level that wasn't in the canonical event — an agent's
completed action, an environmental change, implicit movement — the
router has no memory of it on the following turn.

This module runs a Haiku call at end of turn N that reads the canonical
event + narrator's final_text and emits a one-to-two-sentence note
capturing ONLY the state-level deltas the narrator added. The note is
stored in SessionState.pending_recap; turn N+1's router consumes it
(embedded into the user message, which archives into
session_conversation) and clears the buffer. Narrator context now
flows to the router one turn delayed but complete enough to adjudicate
against.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.events import CanonicalEvent

logger = logging.getLogger(__name__)


class _TurnRecapOutput(BaseModel):
    """Structured output for the summarizer. Internal — callers receive
    the string note directly."""
    model_config = ConfigDict(extra="forbid")

    note: str = ""


async def summarize_turn(
    client: LLMClient,
    prompt_manager: PromptManager,
    canonical_event: CanonicalEvent,
    narrator_final_text: str,
) -> str:
    """Produce a terse delta note for the next router turn.

    Returns the note string, possibly empty. Errors are caught by
    callers — if summarization fails, the router just runs without
    this extra context (graceful degradation).
    """
    import json

    messages = prompt_manager.render_messages(
        "turn_recap",
        canonical_event_json=json.dumps(
            canonical_event.model_dump(), indent=2, sort_keys=True,
        ),
        narrator_final_text=narrator_final_text,
    )

    response = await client.complete(
        role="summarizer",
        messages=messages,
        response_model=_TurnRecapOutput,
        temperature=0.2,
        max_tokens=400,
    )
    note = (response.parsed.note or "").strip()
    logger.info(
        "Turn recap: %d chars → %s",
        len(narrator_final_text),
        note[:100] if note else "(no delta)",
    )
    return note
