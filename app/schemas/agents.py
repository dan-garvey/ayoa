"""Character agent output — engine-internal record after parse.

The agent's LLM call no longer uses structured output. The model produces
free-form prose followed by a single trailing parenthetical containing
its private intent. The engine parses that into the two fields below at
`CharacterAgent.turn`/`draft_turn` time:

- `public_text`: everything before the trailing parenthetical. This is
  what flows downstream to the router as `character_id: public_text`.
- `intent`: contents of the trailing parenthetical. Used for
  logging and as a parsed handle on what the model put in the
  parens; NOT mirrored anywhere on the character record. For committed
  agent turns, the parenthetical lives
  verbatim in the agent's own rolling history (so its future self
  carries continuous interior) and is stripped at every cross-agent /
  narrator / router chokepoint. No other actor — not the router, not the
  narrator, not another agent — gets to see this character's
  parenthetical. That asymmetry is load-bearing; if you want to surface
  a character's planning to another LLM, do it through in-fiction
  signals (a courier, a witnessed action, an observable fact) instead.

Why no LLM-target schema:
- The four `private_updates` fields in v9/v10 were schema-required,
  prompt-emitted, and almost entirely unconsumed downstream. Movement
  and objective writeback are now handled by the router, which sees the
  agent's public prose and decides what to canonicalize.
  Inter-character "directives" (a structured message bus for NPC-to-NPC
  coordination) were dropped in Commit 2: characters that need to
  coordinate now do it through normal event prose — a courier walks
  in and speaks, or a note is rendered in observable_facts.
- Free prose lets the model commit to short responses (2-4 sentences
  by default) without negotiating empty list / empty string contracts.
- The agent's own continuity (objectives that mutate over time, plans
  it forms and abandons) lives in its rolling conversation, not in
  fields the engine has to write back. Cheaper, less drift, more
  authorial freedom.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CharacterAgentOutput(BaseModel):
    """Engine-internal record of one parsed agent response.

    Constructed by `CharacterAgent.turn` / `.draft_turn` after extracting
    the trailing parenthetical from the LLM's prose output. NOT an LLM
    target — the LLM emits plain text and we parse it.
    """
    model_config = ConfigDict(extra="forbid")

    character_id: str
    public_text: str
    intent: str
