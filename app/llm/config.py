from __future__ import annotations

import os

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    api_key: str = ""

    # The discriminator role was merged into the event router in v2; no
    # caller asks for role="discriminator" anymore, so it's omitted here.
    #
    # Agent: Sonnet. Tried Haiku for playtesting cost (most per-turn
    # spend is the agent cascade fan-out) but the playtests caught
    # quality regressions Haiku couldn't carry — attribution drift
    # under multi-NPC cascades, weaker prose, missed subtext on
    # interior beats. The agent is the load-bearing surface for
    # in-character voice consistency, so we eat the cost.
    #
    # Narrator: Haiku as of Option B (v11-r10). The narrator's job
    # contracted sharply once `observable_facts` became the sole render
    # source — it no longer has to interpret narrator-grade interior
    # prose from `resolved_outcome`, infer what the actor was thinking,
    # or compose meaning out of an ambiguous outcome string. The job is
    # now: take 4-6 surface facts, weave them into POV prose, drop
    # facts the observation level forbids, don't editorialize. That's
    # a transformation, not an authorship task. Haiku is well-matched
    # to it and cuts the per-turn narrator cost ~5x. Flip back to
    # Sonnet via config if a playtest shows prose quality regressing.
    role_models: dict[str, str] = Field(default_factory=lambda: {
        "event_router": "claude-sonnet-4-6",
        "narrator": "claude-haiku-4-5",
        "agent": "claude-sonnet-4-6",
        "character_gen": "claude-sonnet-4-6",
        # Terse post-turn summarization (delta notes for the router).
        # Cheap, narrow task — Haiku is plenty.
        "summarizer": "claude-haiku-4-5",
        # Out-of-character /query consultation. Read-only, short
        # answer, single-character POV bound. Latency matters more
        # than depth here — players are staring at Discord waiting
        # for "what do I see?" / "what was her name?" — so Haiku is
        # the default. Flip to Sonnet via env if you want richer
        # in-fiction refusal flavor.
        "query_handler": "claude-haiku-4-5",
    })

    # v11-r9b: per-role extended-thinking budgets (Anthropic Messages
    # API `thinking: {type: "enabled", budget_tokens: N}`). When > 0,
    # the LLMClient enables extended thinking for that role's call;
    # the model spends up to `N` tokens on internal reasoning before
    # composing its visible response. Empty / missing role => no
    # thinking, behave as before.
    #
    # Agent at 2048: the character cascade is where multi-step
    # reasoning matters most — read the perception inbox, weigh
    # attribution carefully, decide whether this beat is the moment
    # to advance a goal, then write the prose. The t8 playtest
    # caught the agent making attribution-inversion mistakes
    # (Ashara claiming Garvey asked Lysara when Lysara made the
    # statement) under the previous one-shot config; thinking
    # tokens give the model room to chew on the inbox before
    # speaking. 2048 is a ceiling, not a floor — typical
    # pre-response planning lands in 500-1500 tokens.
    #
    # Event router at 2048: the router's adjudication call is the
    # other multi-step reasoning surface in the engine — categorize
    # the action (Cat I vs II), pick observers and observation
    # levels, decide whether the beat ends, choose which NPCs to
    # cascade and in what order, honor the addressed-NPC rule,
    # author observable_facts that don't leak interior, etc. The
    # decision_rationale field captures part of this thinking but
    # has to fit in the structured output, so it's terse by
    # construction; an extended-thinking budget lets the model
    # reason freely before composing the JSON. The narrator and
    # other roles still default to 0 — the narrator's job is
    # transformation (facts → POV prose) and the others are narrow
    # enough that a tight system prompt suffices.
    role_thinking_budgets: dict[str, int] = Field(default_factory=lambda: {
        "agent": 2048,
        "event_router": 2048,
    })

    default_model: str = "claude-sonnet-4-6"
    # Retries are for transient API failures: 529 overloaded, 500/503,
    # network blips, streaming disconnects. Anthropic's overload events
    # in particular ride out in 5-30s windows; 4 retries × exp-backoff
    # with jitter clears most of them without dumping the user. The pre-
    # bump default of 2 surfaced "overloaded_error" to the player on the
    # first burst of concurrent /act commands during the playtest.
    max_retries: int = 4
    retry_base_delay: float = 1.0
    # Symmetric jitter band around the exp-backoff delay, expressed as
    # a fraction. 0.3 means each delay is uniformly sampled from
    # [0.7×delay, 1.3×delay]. Prevents thundering-herd lockstep when
    # multiple players /act simultaneously and all hit overloaded at
    # the same wall-clock instant.
    retry_jitter: float = 0.3
    # Hard cap on any single retry sleep. Prevents a 4th retry from
    # waiting 16s+ on a request the user is staring at; we'd rather
    # surface the failure cleanly than hold the slash command for
    # half a minute.
    retry_max_delay: float = 30.0
    # Anthropic's server-side deadline for a single request is 10 minutes.
    # Client timeouts shorter than that truncate long structured-output
    # grammar-compilation passes and surface as "Request timed out or
    # interrupted." See https://docs.anthropic.com/en/api/errors#long-requests
    # — they explicitly recommend using streaming (we do) plus not undercutting
    # the server's own deadline with a shorter client-side one.
    timeout: float = 600.0
    # Compaction trigger (beta). Sonnet 4.6 has a 1M context window with no
    # long-context pricing tier, so we defer compaction until we're within
    # ~100K tokens of the window limit. Tune down for smaller-window models.
    compact_trigger_tokens: int = 900_000

    # Cache TTL for ephemeral prompt-cache blocks. "1h" is the longest
    # Anthropic supports; "5m" is the cheaper write-cost tier. Sessions
    # with long gaps between turns benefit from "1h"; short playtests
    # are indifferent. Can be overridden per-session via SessionSettings
    # (see app/schemas/state.py::SessionSettings) or globally at boot
    # via the ANTHROPIC_CACHE_TTL env var.
    cache_ttl: str = "1h"

    def model_for_role(self, role: str) -> str:
        return self.role_models.get(role, self.default_model)

    def thinking_budget_for_role(self, role: str) -> int:
        """Per-role extended-thinking budget in tokens. 0 means no
        thinking (default behavior)."""
        return self.role_thinking_budgets.get(role, 0)

    @classmethod
    def from_env(cls) -> LLMConfig:
        ttl = os.environ.get("ANTHROPIC_CACHE_TTL", "1h")
        if ttl not in ("5m", "1h"):
            raise ValueError(
                f"ANTHROPIC_CACHE_TTL must be '5m' or '1h', got {ttl!r}"
            )
        return cls(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            cache_ttl=ttl,
        )
