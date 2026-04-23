"""Character agent output — engine-internal record after parse.

The agent's LLM call no longer uses structured output. The model produces
free-form prose followed by a single trailing parenthetical containing
its private intent. The engine parses that into the two fields below at
`CharacterAgent.respond`/`tick` time:

- `public_text`: everything before the trailing parenthetical. This is
  what flows downstream — narrator phase-1 input, other agents'
  prior-responses cascade input, the router's intention block.
- `intent`: contents of the trailing parenthetical. Engine + router
  see it (the router uses it as freshness signal on the actor's
  CharacterRecord.last_intent); the narrator and other agents NEVER do.
  The parenthetical is in the agent's own rolling history (so its
  future self carries continuous interior); it is stripped at every
  cross-agent / narrator chokepoint.

Why no LLM-target schema:
- The four `private_updates` fields in v9/v10 were schema-required,
  prompt-emitted, and almost entirely unconsumed downstream. Movement,
  scene creation, directives, and objective writeback are now handled
  by the router (on-stage) and the unified tick router (off-stage),
  which see the agent's prose + intent and decide what to canonicalize.
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

    Constructed by `CharacterAgent.respond` / `.tick` after extracting
    the trailing parenthetical from the LLM's prose output. NOT an LLM
    target — the LLM emits plain text and we parse it.
    """
    model_config = ConfigDict(extra="forbid")

    character_id: str
    public_text: str
    intent: str
