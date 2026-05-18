from __future__ import annotations

import json
import os
import re

from pydantic import BaseModel, Field


_VALID_PROVIDERS = {"anthropic", "openai"}
_VALID_OPENAI_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
}
_VALID_OPENAI_REASONING_SUMMARIES = {
    "auto",
    "concise",
    "detailed",
    "none",
}
_ROUTER_MODEL = "gpt-5.2"
_NARRATOR_MODEL = "gpt-5.2"
_COMBAT_MANAGER_MODEL = "gpt-5.1"
_AGENT_MODEL = "claude-opus-4-6"
_STANDARD_AGENT_MODEL = "claude-haiku-4-5"
_CONVENIENCE_AGENT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MODEL = "gpt-5.1"
_ROLE_ENV_ALIASES = {
    "agent": ("AGENT",),
    "agent_standard": ("AGENT_STANDARD", "STANDARD_AGENT"),
    "agent_convenience": ("AGENT_CONVENIENCE", "CONVENIENCE_AGENT"),
    "character_gen": ("CHARACTER_GEN", "AGENT"),
    "dnd_combat_manager": ("DND_COMBAT_MANAGER", "COMBAT_MANAGER"),
    "event_router": ("ROUTER",),
    "narrator": ("NARRATOR",),
}


def _normalise_provider(provider: str) -> str:
    value = provider.strip().lower()
    if value not in _VALID_PROVIDERS:
        raise ValueError(
            f"LLM provider must be one of {sorted(_VALID_PROVIDERS)}, got {provider!r}"
        )
    return value


def _normalise_openai_reasoning_effort(effort: str) -> str:
    value = effort.strip().lower()
    if value not in _VALID_OPENAI_REASONING_EFFORTS:
        raise ValueError(
            "OpenAI reasoning effort must be one of "
            f"{sorted(_VALID_OPENAI_REASONING_EFFORTS)}, got {effort!r}"
        )
    return value


def _normalise_openai_reasoning_summary(summary: str) -> str:
    value = summary.strip().lower()
    if value not in _VALID_OPENAI_REASONING_SUMMARIES:
        raise ValueError(
            "OpenAI reasoning summary must be one of "
            f"{sorted(_VALID_OPENAI_REASONING_SUMMARIES)}, got {summary!r}"
        )
    return "" if value == "none" else value


def _split_provider_model(model: str) -> tuple[str | None, str]:
    if ":" not in model:
        return None, model
    provider, bare_model = model.split(":", 1)
    return _normalise_provider(provider), bare_model


def _parse_env_map(raw: str | None) -> dict[str, str]:
    """Parse JSON or comma-separated role=value env overrides."""
    if not raw:
        return {}
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM env map must be a JSON object")
        return {str(k): str(v) for k, v in parsed.items()}

    result: dict[str, str] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"LLM env map entries must be role=value, got {item!r}"
            )
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _role_env_suffix(role: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", role.upper()).strip("_")


def _role_env_suffixes(role: str) -> tuple[str, ...]:
    suffixes = [_role_env_suffix(role)]
    suffixes.extend(_ROLE_ENV_ALIASES.get(role, ()))
    return tuple(dict.fromkeys(s for s in suffixes if s))


def _first_env_value(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _first_role_env_value(role: str, prefix: str) -> str:
    return _first_env_value(
        tuple(f"{prefix}_{suffix}" for suffix in _role_env_suffixes(role))
    )


def _openai_role_api_key_env_names(role: str) -> tuple[str, ...]:
    names: list[str] = []

    # Local cost-tracking keys may use short role names:
    # OPEN_AI_AGENT, OPEN_AI_ROUTER, OPEN_AI_NARRATOR.
    for alias in _ROLE_ENV_ALIASES.get(role, ()):
        names.append(f"OPEN_AI_{alias}")

    for suffix in _role_env_suffixes(role):
        names.extend((
            f"OPENAI_API_KEY_{suffix}",
            f"OPEN_AI_API_KEY_{suffix}",
            f"OPENAI_{suffix}_API_KEY",
            f"OPEN_AI_{suffix}_API_KEY",
            f"OPEN_AI_{suffix}",
        ))

    return tuple(dict.fromkeys(names))


class LLMConfig(BaseModel):
    # Back-compat: `api_key` is the Anthropic key used by the original
    # single-provider client. New code should call api_key_for_provider().
    api_key: str = ""
    openai_api_key: str = ""
    openai_role_api_keys: dict[str, str] = Field(default_factory=dict)

    default_provider: str = "openai"
    role_providers: dict[str, str] = Field(default_factory=dict)

    # The discriminator role was merged into the event router in v2; no
    # caller asks for role="discriminator" anymore, so it's omitted here.
    role_models: dict[str, str] = Field(default_factory=lambda: {
        "event_router": _ROUTER_MODEL,
        "narrator": _NARRATOR_MODEL,
        "dnd_combat_manager": _COMBAT_MANAGER_MODEL,
        "agent": _AGENT_MODEL,
        "agent_standard": _STANDARD_AGENT_MODEL,
        "agent_convenience": _CONVENIENCE_AGENT_MODEL,
        "character_gen": _AGENT_MODEL,
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
    # OpenAI reasoning models use effort levels rather than token budgets.
    # Keep medium defaults for roles that run on OpenAI by default or by
    # override, so they get explicit reasoning instead of silently running
    # in fast/no-reasoning mode. Per-role env overrides can lower cheap
    # roles later.
    openai_reasoning_efforts: dict[str, str] = Field(default_factory=lambda: {
        "event_router": "medium",
        "narrator": "medium",
        "dnd_combat_manager": "medium",
        "agent": "medium",
        "character_gen": "medium",
    })
    # Raw OpenAI reasoning tokens are not exposed by the API. This optional
    # per-role setting requests provider-authored summaries instead.
    openai_reasoning_summaries: dict[str, str] = Field(default_factory=lambda: {
        "event_router": "auto",
    })

    default_model: str = _DEFAULT_MODEL
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
    # Keep the client deadline high enough for long structured-output
    # requests and transient provider-side work, especially during imports.
    timeout: float = 600.0
    # Server-side context compaction is opt-in. It is useful for very long
    # Anthropic conversations, but model support varies and prompt-cache plus
    # deterministic history compaction already handle normal sessions.
    enable_anthropic_compaction: bool = False
    # Compaction trigger for providers/models that support server-side
    # context compaction. Tune down for smaller-window models.
    compact_trigger_tokens: int = 900_000

    # Cache TTL for ephemeral prompt-cache blocks. "1h" is the longest
    # Anthropic supports; "5m" is the cheaper write-cost tier. Sessions
    # with long gaps between turns benefit from "1h"; short playtests
    # are indifferent. Can be overridden per-session via SessionSettings
    # (see app/schemas/state.py::SessionSettings) or globally at boot
    # via the ANTHROPIC_CACHE_TTL env var.
    cache_ttl: str = "1h"

    def model_for_role(self, role: str) -> str:
        _, model = _split_provider_model(self.role_models.get(role, self.default_model))
        return model

    def provider_for_role(self, role: str) -> str:
        configured = self.role_providers.get(role)
        if configured:
            return _normalise_provider(configured)

        provider, model = _split_provider_model(
            self.role_models.get(role, self.default_model)
        )
        if provider:
            return provider

        model_lower = model.lower()
        if model_lower.startswith("claude-"):
            return "anthropic"
        if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
            return "openai"
        return _normalise_provider(self.default_provider)

    def roles_for_provider(self, provider: str) -> set[str]:
        provider = _normalise_provider(provider)
        roles = set(self.role_models) | set(self.role_providers)
        return {
            role
            for role in roles
            if self.provider_for_role(role) == provider
        }

    def providers_in_use(self) -> set[str]:
        roles = set(self.role_models) | set(self.role_providers)
        providers = {self.provider_for_role(role) for role in roles}
        if not roles:
            providers.add(_normalise_provider(self.default_provider))
        return providers

    def openai_api_key_for_role(self, role: str) -> str:
        return self.openai_role_api_keys.get(role, "") or self.openai_api_key

    def openai_role_api_key_env_names(self, role: str) -> tuple[str, ...]:
        return _openai_role_api_key_env_names(role)

    def api_key_for_provider(self, provider: str, role: str | None = None) -> str:
        provider = _normalise_provider(provider)
        if provider == "anthropic":
            return self.api_key
        if provider == "openai":
            if role:
                return self.openai_api_key_for_role(role)
            return self.openai_api_key
        raise AssertionError(f"unreachable provider {provider!r}")

    def thinking_budget_for_role(self, role: str) -> int:
        """Per-role extended-thinking budget in tokens. 0 means no
        thinking (default behavior)."""
        return self.role_thinking_budgets.get(role, 0)

    def openai_reasoning_effort_for_role(self, role: str) -> str:
        effort = self.openai_reasoning_efforts.get(role, "")
        return _normalise_openai_reasoning_effort(effort) if effort else ""

    def openai_reasoning_summary_for_role(self, role: str) -> str:
        summary = self.openai_reasoning_summaries.get(role, "")
        return _normalise_openai_reasoning_summary(summary) if summary else ""

    @classmethod
    def from_env(cls) -> LLMConfig:
        ttl = os.environ.get("ANTHROPIC_CACHE_TTL", "1h")
        if ttl not in ("5m", "1h"):
            raise ValueError(
                f"ANTHROPIC_CACHE_TTL must be '5m' or '1h', got {ttl!r}"
            )
        defaults = cls()
        role_models = dict(defaults.role_models)
        role_models.update(_parse_env_map(os.environ.get("LLM_ROLE_MODELS")))
        role_providers = dict(defaults.role_providers)
        role_providers.update(_parse_env_map(os.environ.get("LLM_ROLE_PROVIDERS")))
        openai_reasoning_efforts = dict(defaults.openai_reasoning_efforts)
        openai_reasoning_efforts.update(
            _parse_env_map(os.environ.get("LLM_OPENAI_REASONING_EFFORTS"))
        )
        openai_reasoning_summaries = dict(defaults.openai_reasoning_summaries)
        openai_reasoning_summaries.update(
            _parse_env_map(os.environ.get("LLM_OPENAI_REASONING_SUMMARIES"))
        )

        known_roles = set(role_models) | set(role_providers)
        known_roles.update(openai_reasoning_efforts)
        known_roles.update(openai_reasoning_summaries)
        global_reasoning = os.environ.get("LLM_OPENAI_REASONING_EFFORT", "")
        global_reasoning_summary = os.environ.get(
            "LLM_OPENAI_REASONING_SUMMARY", "",
        )
        for role in known_roles:
            model_override = _first_role_env_value(role, "LLM_MODEL")
            provider_override = _first_role_env_value(role, "LLM_PROVIDER")
            reasoning_override = (
                _first_role_env_value(role, "LLM_OPENAI_REASONING")
                or _first_role_env_value(role, "LLM_REASONING")
            )
            reasoning_summary_override = (
                _first_role_env_value(role, "LLM_OPENAI_REASONING_SUMMARY")
                or _first_role_env_value(role, "LLM_REASONING_SUMMARY")
            )
            if model_override:
                role_models[role] = model_override
            if provider_override:
                role_providers[role] = provider_override
            if reasoning_override:
                openai_reasoning_efforts[role] = reasoning_override
            elif global_reasoning:
                openai_reasoning_efforts[role] = global_reasoning
            if reasoning_summary_override:
                openai_reasoning_summaries[role] = reasoning_summary_override
            elif global_reasoning_summary:
                openai_reasoning_summaries[role] = global_reasoning_summary

        openai_reasoning_efforts = {
            role: _normalise_openai_reasoning_effort(effort)
            for role, effort in openai_reasoning_efforts.items()
            if effort
        }
        openai_reasoning_summaries = {
            role: summary
            for role, summary in (
                (role, _normalise_openai_reasoning_summary(summary))
                for role, summary in openai_reasoning_summaries.items()
                if summary
            )
            if summary
        }

        openai_role_api_keys = _parse_env_map(
            os.environ.get("LLM_OPENAI_ROLE_API_KEYS")
        )
        for role in known_roles:
            role_key = _first_env_value(_openai_role_api_key_env_names(role))
            if role_key:
                openai_role_api_keys[role] = role_key

        compaction_raw = os.environ.get(
            "ANTHROPIC_COMPACTION_ENABLED",
            os.environ.get("LLM_ANTHROPIC_COMPACTION_ENABLED", ""),
        ).strip().lower()
        enable_anthropic_compaction = (
            compaction_raw in {"1", "true", "yes", "on"}
            if compaction_raw
            else defaults.enable_anthropic_compaction
        )

        return cls(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_role_api_keys=openai_role_api_keys,
            default_provider=os.environ.get(
                "LLM_DEFAULT_PROVIDER", defaults.default_provider,
            ),
            role_models=role_models,
            role_providers=role_providers,
            openai_reasoning_efforts=openai_reasoning_efforts,
            openai_reasoning_summaries=openai_reasoning_summaries,
            cache_ttl=ttl,
            enable_anthropic_compaction=enable_anthropic_compaction,
        )
