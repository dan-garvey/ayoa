from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TranscriptEntry(BaseModel):
    user: str
    assistant: str


class NarratorFinalOutput(BaseModel):
    """Produced by Narrator Phase 2. LLM output target.

    v11-r7j: `transcript_entry` was removed from the LLM output surface.
    The model emits `final_text` only; the engine constructs the transcript
    entry directly (user = real player input passed into the dispatcher,
    assistant = `final_text`). Pre-r7j the model was asked to echo the
    player's input back into `transcript_entry.user`, but the narrator
    was given `user_input=""` (the per-POV render path doesn't carry the
    original utterance) and the prompt rendered that as `"{name} — "` —
    which the LLM then dutifully echoed into the transcript, breaking
    /history. Engine ownership of the entry eliminates the round-trip,
    saves tokens, and removes a bug surface.
    """

    model_config = ConfigDict(extra="forbid")

    final_text: str
