#!/usr/bin/env python3
"""Live D&D router comparison harness.

This calls the production LLMDispatcher directly with curated synthetic
checkpoints. The cases are based on historical router prompt fixes and recent
D&D playtest regressions, but avoid full CLI/orchestrator setup so model
differences are easy to inspect.

Outputs:
  app/storage/playtest_reports/router_dnd_compare_<timestamp>/report.json
  app/storage/playtest_reports/router_dnd_compare_<timestamp>/report.md
  app/storage/playtest_reports/router_dnd_compare_<timestamp>/raw_calls.jsonl
  app/storage/playtest_reports/router_dnd_compare_<timestamp>/run.log
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.llm.client import LLMClient, LLMResponse
from app.llm.config import LLMConfig
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import DndEventRouterOutput, EventRouterOutput
from app.schemas.state import PhysicsRuleset, SessionConfig, SessionState
from app.schemas.state import StorySetting, WorldState


REPORT_DIR = REPO_ROOT / "app/storage/playtest_reports"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
RUN_DIR = REPORT_DIR / f"router_dnd_compare_{TS}"
JSON_PATH = RUN_DIR / "report.json"
MD_PATH = RUN_DIR / "report.md"
RAW_CALLS_PATH = RUN_DIR / "raw_calls.jsonl"
LOG_PATH = RUN_DIR / "run.log"


@dataclass(frozen=True)
class Candidate:
    label: str
    model: str
    reasoning_effort: str
    provider: str = "openai"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: Any = ""


@dataclass(frozen=True)
class CaseResult:
    name: str
    input_summary: str
    method: str
    output: dict[str, Any]
    checks: list[Check]
    error: str = ""


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    commit_evidence: list[str]
    runner: Callable[[LLMDispatcher], Awaitable[CaseResult]]


class RecordingClient:
    """Wrap LLMClient so each live call is written to raw_calls.jsonl."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        candidate: Candidate,
        raw_path: Path,
        include_prompts: bool = False,
        call_timeout_seconds: float = 180.0,
    ) -> None:
        self.inner = inner
        self.config = inner.config
        self.candidate = candidate
        self.raw_path = raw_path
        self.include_prompts = include_prompts
        self.current_case = ""
        self.call_timeout_seconds = call_timeout_seconds

    async def complete(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        response_model: type[Any] | None = None,
        cache: bool = True,
        compact: bool = False,
        stream: bool = False,
    ) -> LLMResponse:
        started = time.perf_counter()
        response: LLMResponse | None = None
        error = ""
        try:
            response = await asyncio.wait_for(
                self.inner.complete(
                    role,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_model=response_model,
                    cache=cache,
                    compact=compact,
                    stream=stream,
                ),
                timeout=self.call_timeout_seconds,
            )
            return response
        except Exception:
            error = traceback.format_exc()
            raise
        finally:
            elapsed = time.perf_counter() - started
            record: dict[str, Any] = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "candidate": self.candidate.label,
                "candidate_model": self.candidate.model,
                "case": self.current_case,
                "role": role,
                "provider": self.config.provider_for_role(role),
                "configured_model": self.config.model_for_role(role),
                "response_model": response_model.__name__ if response_model else "",
                "elapsed_seconds": round(elapsed, 3),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "cache": cache,
                "compact": compact,
                "usage": response.usage if response else {},
                "returned_model": response.model if response else "",
                "assistant_content": response.content if response else "",
                "reasoning_summaries": (
                    response.reasoning_summaries if response else []
                ),
                "error": error,
                "messages_summary": _message_summary(messages),
            }
            if response is not None and response.parsed is not None:
                parsed = response.parsed
                if hasattr(parsed, "model_dump"):
                    record["parsed"] = parsed.model_dump(mode="json")
                else:
                    record["parsed"] = repr(parsed)
            if self.include_prompts:
                record["messages"] = messages
            with self.raw_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")

    async def close(self) -> None:
        await self.inner.close()


def _message_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=True, sort_keys=True)
        summary.append({
            "role": message.get("role", ""),
            "chars": len(content),
            "approx_tokens": max(1, len(content) // 4),
            "tail": content[-1200:],
        })
    return summary


def _mechanics(
    *,
    level: int = 3,
    armor_class: int = 14,
    hp: int = 22,
    strength: int = 10,
    dexterity: int = 12,
    constitution: int = 12,
    intelligence: int = 10,
    wisdom: int = 12,
    charisma: int = 10,
    classes: str = "",
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "ruleset_id": "dnd5e_basic",
        "level": level,
        "proficiency_bonus": 2,
        "ability_scores": {
            "str": strength,
            "dex": dexterity,
            "con": constitution,
            "int": intelligence,
            "wis": wisdom,
            "cha": charisma,
        },
        "skill_proficiencies": [],
        "saving_throw_proficiencies": [],
        "armor_class": armor_class,
        "hit_points": {"current": hp, "max": hp, "temporary": 0},
        "conditions": [],
        "resources": {},
        "dnd5e_sheet": {
            "identity": {
                "species": "Human",
                "classes": classes,
                "background": "Expedition hire",
            },
            "statblock": {"actions": actions or []},
        },
        "raw": {},
    }


def _char(
    character_id: str,
    name: str,
    role: str,
    *,
    location: str = "kwalish_route",
    appearance: str = "",
    personality: str = "",
    goals: list[str] | None = None,
    objectives: list[str] | None = None,
    playable: bool = False,
    intentions_enabled: bool = False,
    known_context: str = "",
    mechanics: dict[str, Any] | None = None,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        location=location,
        is_playable=playable,
        public_sheet=PublicSheet(role=role, appearance=appearance),
        personality=personality,
        known_context=known_context,
        private_state=PrivateState(
            goals=goals or [],
            current_objectives=objectives or [],
            intentions_enabled=playable or intentions_enabled,
        ),
        mechanics=mechanics or {},
    )


def _base_characters(
    *,
    include_kess: bool = True,
    include_bandit: bool = True,
    include_trainer: bool = True,
) -> list[CharacterRecord]:
    characters = [
        _char(
            "marlowe",
            "Marlowe Vane",
            "level 3 Hexblade warlock expedition face",
            appearance="black traveling coat, pact blade at hip, silver folio case",
            playable=True,
            mechanics=_mechanics(
                level=3,
                armor_class=16,
                hp=27,
                strength=10,
                dexterity=14,
                constitution=14,
                intelligence=12,
                wisdom=10,
                charisma=16,
                classes="Warlock 3",
            ),
        ),
        _char(
            "tavi",
            "Tavi Gearwright",
            "level 3 artificer scout and trap specialist",
            appearance="oil-stained gloves, collapsible tools, bright brass monocle",
            playable=True,
            mechanics=_mechanics(
                level=3,
                armor_class=15,
                hp=24,
                dexterity=14,
                intelligence=16,
                classes="Artificer 3",
            ),
        ),
        _char(
            "ilyra",
            "Ilyra Moss",
            "level 3 Circle of Spores druid",
            appearance="lichen-thread cloak, bone charms, faint mushroom scent",
            playable=True,
            mechanics=_mechanics(
                level=3,
                armor_class=14,
                hp=25,
                wisdom=16,
                classes="Druid 3",
            ),
        ),
        _char(
            "gearbox",
            "Gearbox",
            "clockwork familiar-like expedition device",
            appearance="thumb-sized crystal lens on a brass spider chassis",
            intentions_enabled=False,
            mechanics=_mechanics(level=1, armor_class=13, hp=5),
        ),
    ]
    if include_bandit:
        characters.append(
            _char(
                "bandit_leader",
                "Bandit Leader",
                "armed road bandit leader",
                appearance="scarred leather jack, green scarf, drawn scimitar",
                intentions_enabled=True,
                mechanics=_mechanics(
                    level=2,
                    armor_class=14,
                    hp=18,
                    strength=12,
                    dexterity=14,
                    actions=[{
                        "id": "scimitar",
                        "name": "Scimitar",
                        "attack": {"bonus": 4, "damage": "1d6+2 slashing"},
                    }],
                ),
            )
        )
    if include_trainer:
        characters.append(
            _char(
                "trainer_boros",
                "Boros Flint",
                "veteran caravan guard offering a training drill",
                appearance="battered shield, blunted practice blade, strict posture",
                intentions_enabled=True,
                mechanics=_mechanics(
                    level=3,
                    armor_class=16,
                    hp=30,
                    strength=15,
                    dexterity=12,
                    actions=[{
                        "id": "practice_blade",
                        "name": "Practice Blade",
                        "attack": {"bonus": 4, "damage": "1d6+2 bludgeoning"},
                    }],
                ),
            )
        )
    if include_kess:
        characters.append(
            _char(
                "kess",
                "Kess",
                "nervous route broker with partial expedition knowledge",
                appearance="patched blue cloak, ink-stained fingers, guarded eyes",
                personality="evasive, proud, anxious about whoever owns the road",
                goals=["Keep leverage over the route without provoking the party."],
                objectives=["Deflect direct questions about who controls passage."],
                intentions_enabled=True,
                mechanics=_mechanics(level=2, armor_class=12, hp=13, charisma=13),
            )
        )
    return characters


def _dnd_ckpt(
    session_id: str,
    *,
    facts: list[str] | None = None,
    hidden_facts: list[str] | None = None,
    hidden_lore: str = "",
    lore: str = "",
    include_kess: bool = True,
    include_bandit: bool = True,
    include_trainer: bool = True,
    conversation: list[ConversationMessage] | None = None,
) -> CheckpointFile:
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=session_id,
            story_id="router_dnd_compare",
            player_name="Dan",
            player_character_id="marlowe",
            character_bindings={
                "marlowe": "user-1",
                "tavi": "user-2",
                "ilyra": "user-3",
            },
            config=SessionConfig(),
        ),
        player_primer=(
            "A compact Lost Laboratory-style D&D expedition route test. "
            "The party is level 3 and moving through dangerous frontier terrain."
        ),
        world_state=WorldState(
            facts=facts or _base_world_facts(),
            hidden_facts=hidden_facts or [],
            hidden_lore=hidden_lore,
            lore=lore or (
                "The party is following old Kwalish expedition clues through "
                "broken frontier paths. Reviewed content may authorize route "
                "landmarks, hazards, or creatures, but generic darkness does not "
                "authorize a new speaking faction."
            ),
            physics_ruleset=PhysicsRuleset(
                strength_limits="dnd5e_basic",
                magic_enabled=True,
            ),
            setting=StorySetting(
                genre="D&D 5e expedition fantasy",
                era="pseudo-medieval fantasy",
                tone="concrete, fair, player-facing",
                premise=(
                    "A small adventuring party is testing router decisions "
                    "around exploration, combat starts, and visibility."
                ),
            ),
        ),
        characters=_base_characters(
            include_kess=include_kess,
            include_bandit=include_bandit,
            include_trainer=include_trainer,
        ),
        session_conversation=conversation or [],
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    ckpt.session.config.settings.player_roll_mode = "auto"
    return ckpt


def _base_world_facts() -> list[str]:
    return [
        "Marlowe, Tavi, and Ilyra travel together on a Kwalish expedition route.",
        "The party is not in initiative until the router starts D&D combat.",
        "A cave mouth, broken bridge, old camp, and cliff trail can all matter.",
        "Known companions can answer when directly addressed; unknown darkness is not a character.",
    ]


def _facts(result: EventRouterOutput) -> list[dict[str, Any]]:
    return [
        {
            "text": fact.text,
            "audience": fact.audience,
            "visible_to": fact.visible_to,
            "at_offset_s": fact.at_offset_s,
            "duration_s": fact.duration_s,
        }
        for fact in result.canonical_event.observable_facts
    ]


def _fact_text(result: EventRouterOutput) -> str:
    return "\n".join(fact.text for fact in result.canonical_event.observable_facts)


def _observer_ids(result: EventRouterOutput) -> list[str]:
    return [obs.character_id for obs in result.observers]


def _next_output_ids(result: EventRouterOutput) -> list[str]:
    return list(result.next_output_character_ids)


def _enrichment_ids(result: EventRouterOutput) -> list[str]:
    return list(result.perception_enrichment_character_ids)


def _dnd_mode(result: EventRouterOutput) -> str:
    return str(getattr(result, "interaction_mode", "") or "")


def _combatant_ids(result: EventRouterOutput) -> list[str]:
    return list(getattr(result, "combatant_ids", []) or [])


def _combatant_spawns(result: EventRouterOutput) -> list[dict[str, Any]]:
    return [
        spawn.model_dump(mode="json")
        for spawn in (getattr(result, "combatant_spawns", []) or [])
    ]


def _loot_offer(result: EventRouterOutput) -> dict[str, Any] | None:
    offer = getattr(result, "loot_offer", None)
    return offer.model_dump(mode="json") if offer is not None else None


def _battle_map_seed(result: EventRouterOutput) -> dict[str, Any] | None:
    seed = getattr(result, "battle_map_seed", None)
    return seed.model_dump(mode="json") if seed is not None else None


def _output_dict(result: EventRouterOutput) -> dict[str, Any]:
    output = result.model_dump(mode="json")
    output["schema"] = result.__class__.__name__
    output["interaction_mode"] = _dnd_mode(result)
    output["combatant_ids"] = _combatant_ids(result)
    output["combatant_spawns"] = _combatant_spawns(result)
    output["loot_offer"] = _loot_offer(result)
    output["battle_map_seed"] = _battle_map_seed(result)
    output["fact_texts"] = _facts(result)
    output["observer_ids"] = _observer_ids(result)
    output["next_output_character_ids"] = _next_output_ids(result)
    output["perception_enrichment_character_ids"] = _enrichment_ids(result)
    return output


def _dnd_result_from_output(output: dict[str, Any]) -> DndEventRouterOutput:
    data = {
        key: value
        for key, value in output.items()
        if key in DndEventRouterOutput.model_fields
    }
    return DndEventRouterOutput.model_validate(data)


def _check(name: str, passed: bool, detail: Any = "") -> Check:
    return Check(name=name, passed=bool(passed), detail=detail)


def _contains_any(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


def _has_fact_visible_to(result: EventRouterOutput, character_id: str, terms: list[str]) -> bool:
    for fact in result.canonical_event.observable_facts:
        visible = fact.audience == "all_observers" or character_id in fact.visible_to
        if visible and _contains_any(fact.text, terms):
            return True
    return False


def _visible_fact_snippets(
    result: EventRouterOutput,
    *,
    character_id: str,
    terms: list[str],
) -> list[str]:
    snippets: list[str] = []
    for fact in result.canonical_event.observable_facts:
        visible = fact.audience == "all_observers" or character_id in fact.visible_to
        if visible and _contains_any(fact.text, terms):
            snippets.append(fact.text)
    return snippets


async def _route_fresh(
    dispatcher: LLMDispatcher,
    *,
    ckpt: CheckpointFile,
    actor_id: str,
    intention: str,
    name: str,
) -> CaseResult:
    result = await dispatcher.route_intention(
        ckpt=ckpt,
        actor_id=actor_id,
        intention=intention,
    )
    return CaseResult(
        name=name,
        input_summary=intention,
        method="route_intention",
        output=_output_dict(result),
        checks=[],
    )


async def _case_unknown_space_spawn(dispatcher: LLMDispatcher) -> CaseResult:
    text = (
        "I keep my weapon low and call into the cave: 'We are not raiders. "
        "We seek Kwalish's old trail. What terms allow safe passage?'"
    )
    ckpt = _dnd_ckpt(
        "cmp_unknown_space",
        facts=[
            *_base_world_facts(),
            "The cave mouth is dark; the party has seen old boot tracks but no present occupant.",
            "No reviewed current content authorizes a speaking cave guide here.",
        ],
        hidden_facts=[
            "There may be old route traces near the cave, but no live hidden listener is established.",
        ],
        include_kess=False,
        include_bandit=False,
        include_trainer=False,
    )
    case = await _route_fresh(
        dispatcher,
        ckpt=ckpt,
        actor_id="marlowe",
        intention=text,
        name="unknown_space_gatekeeper_spawn",
    )
    result = _dnd_result_from_output(case.output)
    known_ids = {char.character_id for char in ckpt.characters}
    routed_unknown = [
        cid
        for cid in [*_observer_ids(result), *_next_output_ids(result)]
        if cid not in known_ids
    ]
    facts = _fact_text(result)
    case = CaseResult(
        **{**case.__dict__, "checks": [
            _check("dnd_fresh_schema", case.output["schema"] == "DndEventRouterOutput"),
            _check("ordinary_exploration_mode", _dnd_mode(result) == "cat_i", _dnd_mode(result)),
            _check("no_generic_story_spawn", not result.spawn, result.spawn),
            _check("no_combatant_spawn", not _combatant_spawns(result), _combatant_spawns(result)),
            _check("no_unknown_routing_target", not routed_unknown, routed_unknown),
            _check(
                "no_agentic_hidden_listener_terms",
                not _contains_any(
                    facts,
                    [
                        "gatekeeper",
                        "emissary",
                        "guide answers",
                        "hidden collective",
                        "scout answers",
                        "sentry answers",
                        "negotiator",
                    ],
                ),
                facts,
            ),
        ]},
    )
    return case


async def _case_hostile_attack_starts_combat(dispatcher: LLMDispatcher) -> CaseResult:
    text = (
        "The armed bandits demand the folios. I draw my longsword, step into "
        "reach of the leader, and cut before they can surround us."
    )
    ckpt = _dnd_ckpt(
        "cmp_hostile_attack",
        facts=[
            *_base_world_facts(),
            "Bandit Leader is visible, armed, and close enough to fight Marlowe.",
            "The bandits have demanded the folios and are trying to box the party in.",
        ],
    )
    case = await _route_fresh(
        dispatcher,
        ckpt=ckpt,
        actor_id="marlowe",
        intention=text,
        name="hostile_attack_starts_combat",
    )
    result = _dnd_result_from_output(case.output)
    facts = _fact_text(result)
    checks = [
        _check("dnd_fresh_schema", case.output["schema"] == "DndEventRouterOutput"),
        _check("combat_start_mode", _dnd_mode(result) == "dnd_combat_start", _dnd_mode(result)),
        _check("no_cat_ii_responders", not result.requires_responders and not result.required_responders, result.required_responders),
        _check("actor_in_combatants", "marlowe" in _combatant_ids(result), _combatant_ids(result)),
        _check("target_in_combatants", "bandit_leader" in _combatant_ids(result), _combatant_ids(result)),
        _check(
            "router_does_not_resolve_hit_or_damage",
            not _contains_any(
                facts,
                ["hits", "slashes", "cuts into", "blood", "damage", "wounds", "drops"],
            ),
            facts,
        ),
    ]
    return CaseResult(**{**case.__dict__, "checks": checks})


async def _case_hidden_ambush_before_attack(dispatcher: LLMDispatcher) -> CaseResult:
    text = "I step onto the bridge and ask, 'Ironjaw, do you mean to parley?'"
    ckpt = _dnd_ckpt(
        "cmp_hidden_ambush",
        facts=[
            *_base_world_facts(),
            "The broken bridge is narrow, loud underfoot, and flanked by rock shelves.",
            "Ironjaw is not visible; the party has only heard that name from old warnings.",
        ],
        hidden_lore=(
            "Goblin skirmishers loyal to Ironjaw are already in attack positions "
            "behind the left rock shelf. Their bows are lifting now."
        ),
        hidden_facts=[
            "The goblin ambush is springing as Marlowe steps onto the bridge.",
            "Roll initiative before any parley answer resolves.",
        ],
        include_kess=False,
        include_bandit=False,
        include_trainer=False,
    )
    case = await _route_fresh(
        dispatcher,
        ckpt=ckpt,
        actor_id="marlowe",
        intention=text,
        name="hidden_ambush_before_visible_attack",
    )
    result = _dnd_result_from_output(case.output)
    facts = _fact_text(result)
    checks = [
        _check("dnd_fresh_schema", case.output["schema"] == "DndEventRouterOutput"),
        _check("combat_start_mode", _dnd_mode(result) == "dnd_combat_start", _dnd_mode(result)),
        _check("actor_in_combatants", "marlowe" in _combatant_ids(result), _combatant_ids(result)),
        _check("hidden_enemy_spawned_for_combat", bool(_combatant_spawns(result)), _combatant_spawns(result)),
        _check(
            "visible_facts_do_not_name_hidden_source",
            not _contains_any(
                facts,
                ["goblin", "skirmisher", "ironjaw's loyal", "ambush", "behind the left rock"],
            ),
            facts,
        ),
    ]
    return CaseResult(**{**case.__dict__, "checks": checks})


async def _case_training_spar(dispatcher: LLMDispatcher) -> CaseResult:
    text = (
        "I square up for a training spar with Boros, draw my practice blade, "
        "and make one clean exchange to test my footing."
    )
    ckpt = _dnd_ckpt(
        "cmp_training_spar",
        facts=[
            *_base_world_facts(),
            "Boros and Marlowe are in a clear practice yard beside the caravan.",
            "Boros has agreed this is a training-safe drill, not a lethal duel.",
        ],
        include_kess=False,
        include_bandit=False,
        include_trainer=True,
    )
    case = await _route_fresh(
        dispatcher,
        ckpt=ckpt,
        actor_id="marlowe",
        intention=text,
        name="training_spar_starts_combat",
    )
    result = _dnd_result_from_output(case.output)
    facts = _fact_text(result)
    checks = [
        _check("dnd_fresh_schema", case.output["schema"] == "DndEventRouterOutput"),
        _check("combat_start_mode", _dnd_mode(result) == "dnd_combat_start", _dnd_mode(result)),
        _check("no_cat_ii_responders", not result.requires_responders and not result.required_responders, result.required_responders),
        _check("trainer_in_combatants", "trainer_boros" in _combatant_ids(result), _combatant_ids(result)),
        _check(
            "router_does_not_award_training_result",
            not _contains_any(
                facts,
                ["forces boros", "wins the exchange", "lands cleanly", "boros yields"],
            ),
            facts,
        ),
    ]
    return CaseResult(**{**case.__dict__, "checks": checks})


async def _case_social_cat_ii(dispatcher: LLMDispatcher) -> CaseResult:
    text = (
        "I watch Kess for contradictions while asking who really controls "
        "passage here."
    )
    ckpt = _dnd_ckpt(
        "cmp_social_cat_ii",
        facts=[
            *_base_world_facts(),
            "Kess is present, close enough for Marlowe to read facial tells.",
            "Kess has dodged direct route-control questions twice.",
        ],
    )
    case = await _route_fresh(
        dispatcher,
        ckpt=ckpt,
        actor_id="marlowe",
        intention=text,
        name="social_insight_requires_responder",
    )
    result = _dnd_result_from_output(case.output)
    facts = _fact_text(result)
    checks = [
        _check("dnd_fresh_schema", case.output["schema"] == "DndEventRouterOutput"),
        _check("cat_ii_mode", _dnd_mode(result) == "cat_ii", _dnd_mode(result)),
        _check("requires_kess", "kess" in result.required_responders, result.required_responders),
        _check("no_combat_fields", not _combatant_ids(result) and not _combatant_spawns(result), _combatant_ids(result)),
        _check(
            "does_not_decide_kess_substantive_choice",
            not _contains_any(
                facts,
                [
                    "kess admits",
                    "kess reveals",
                    "kess confesses",
                    "kess agrees",
                    "kess tells marlowe who controls",
                ],
            ),
            facts,
        ),
    ]
    return CaseResult(**{**case.__dict__, "checks": checks})


async def _case_agent_output_not_cat_ii(dispatcher: LLMDispatcher) -> CaseResult:
    text = '"Move!" He draws his scimitar and steps between Marlowe and the cleft.'
    ckpt = _dnd_ckpt(
        "cmp_agent_output",
        facts=[
            *_base_world_facts(),
            "Bandit Leader is already in the live scene and visible to the party.",
            "This is a committed NPC public output from the existing cascade.",
        ],
    )
    result = await dispatcher.route_agent_output(
        ckpt=ckpt,
        character_id="bandit_leader",
        public_text=text,
    )
    facts = _fact_text(result)
    checks = [
        _check("dnd_mode_enabled_on_checkpoint", ckpt.session.config.settings.ruleset_id == "dnd5e_basic"),
        _check("nonfresh_generic_schema", result.__class__.__name__ == "EventRouterOutput", result.__class__.__name__),
        _check("does_not_open_cat_ii", not result.requires_responders and result.event_kind != "cat_ii_open", result.event_kind),
        _check("no_required_responders", not result.required_responders, result.required_responders),
        _check("does_not_spawn_existing_actor", not result.spawn, result.spawn),
        _check("keeps_public_text", "Move" in facts or "scimitar" in facts, facts),
    ]
    return CaseResult(
        name="committed_agent_output_not_cat_ii",
        input_summary=f"bandit_leader: {text}",
        method="route_agent_output",
        output=_output_dict(result),
        checks=checks,
    )


async def _case_mediated_perception(dispatcher: LLMDispatcher) -> CaseResult:
    text = (
        "I lean behind a cracked pillar, point at the glowing rune, and whisper "
        "to Tavi, 'Count to three if it moves.'"
    )
    ckpt = _dnd_ckpt(
        "cmp_mediated_perception",
        facts=[
            *_base_world_facts(),
            "Marlowe and Tavi are beside the cracked pillar and can see each other.",
            "Ilyra hears Marlowe through a crackling sending stone but cannot see Marlowe.",
            "Gearbox sends a silent crystal image to the party but carries no audio.",
        ],
        include_kess=False,
        include_bandit=False,
        include_trainer=False,
    )
    case = await _route_fresh(
        dispatcher,
        ckpt=ckpt,
        actor_id="marlowe",
        intention=text,
        name="mediated_perception_splits_audio_and_visual",
    )
    result = _dnd_result_from_output(case.output)
    audio_terms = ["count to three", "whisper"]
    visual_terms = ["lean", "pillar", "point", "rune"]
    ilyra_visual_leaks = _visible_fact_snippets(
        result,
        character_id="ilyra",
        terms=visual_terms,
    )
    gearbox_audio_leaks = _visible_fact_snippets(
        result,
        character_id="gearbox",
        terms=audio_terms,
    )
    facts = _fact_text(result)
    checks = [
        _check("dnd_fresh_schema", case.output["schema"] == "DndEventRouterOutput"),
        _check("noncombat_mode", _dnd_mode(result) == "cat_i", _dnd_mode(result)),
        _check("audio_recipient_observes", "ilyra" in _observer_ids(result), _observer_ids(result)),
        _check("visual_recipient_observes", "gearbox" in _observer_ids(result), _observer_ids(result)),
        _check("audio_only_does_not_receive_visuals", not ilyra_visual_leaks, ilyra_visual_leaks),
        _check("visual_only_does_not_receive_audio", not gearbox_audio_leaks, gearbox_audio_leaks),
        _check("dialogue_preserved_somewhere", _contains_any(facts, ["count to three"]), facts),
        _check("visual_action_preserved_somewhere", _contains_any(facts, ["rune", "pillar", "point"]), facts),
    ]
    return CaseResult(**{**case.__dict__, "checks": checks})


async def _case_repeated_defer(dispatcher: LLMDispatcher) -> CaseResult:
    text = "(defer)"
    conversation = [
        ConversationMessage(role="user", content="(defer)"),
        ConversationMessage(
            role="assistant",
            content=(
                "prior_event evt_thin @90+5 source=kess mode=agent_output "
                "end=directed_at_player\n"
                "fact all @0+5: Kess says, 'Then say what you are doing next.'\n"
                "obs marlowe:d:observe_only tavi:d:observe_only ilyra:d:observe_only kess:d:observe_only"
            ),
        ),
    ]
    ckpt = _dnd_ckpt(
        "cmp_repeated_defer",
        facts=[
            *_base_world_facts(),
            "The route conversation has stalled at the mouth of the cliff path.",
            "Kess has already put the decision back to the party once.",
        ],
        include_bandit=False,
        include_trainer=False,
        conversation=conversation,
    )
    case = await _route_fresh(
        dispatcher,
        ckpt=ckpt,
        actor_id="marlowe",
        intention=text,
        name="repeated_defer_after_thin_handoff",
    )
    result = _dnd_result_from_output(case.output)
    facts = _fact_text(result).lower()
    invented_player_action = _contains_any(
        facts,
        [
            "marlowe waits",
            "marlowe pauses",
            "marlowe looks",
            "marlowe hesitates",
            "marlowe remains",
        ],
    )
    stronger_affordance = _contains_any(
        facts,
        [
            "path",
            "tracks",
            "wind",
            "stone",
            "obstacle",
            "landmark",
            "arrival",
            "movement",
            "sound",
            "route",
            "choice",
        ],
    )
    thin_direct_return = result.event_kind == "directed_at_player" and not stronger_affordance
    checks = [
        _check("dnd_fresh_schema", case.output["schema"] == "DndEventRouterOutput"),
        _check("does_not_invent_player_action", not invented_player_action, facts),
        _check("does_not_repeat_thin_direct_handoff", not thin_direct_return, {"event_kind": result.event_kind, "facts": facts}),
        _check(
            "open_with_pick_or_terminal_with_affordance",
            (
                result.event_kind == "beat_continues"
                and bool(_next_output_ids(result))
            )
            or (
                result.event_kind != "beat_continues"
                and stronger_affordance
            ),
            {"event_kind": result.event_kind, "next": _next_output_ids(result), "facts": facts},
        ),
    ]
    return CaseResult(**{**case.__dict__, "checks": checks})


async def _case_query_perception(dispatcher: LLMDispatcher) -> CaseResult:
    text = "(query: what does Kess look like right now?)"
    ckpt = _dnd_ckpt(
        "cmp_query_perception",
        facts=[
            *_base_world_facts(),
            "Kess is visible to Marlowe near the route marker.",
            "Kess's public sheet says patched blue cloak, ink-stained fingers, guarded eyes.",
        ],
        include_bandit=False,
        include_trainer=False,
    )
    case = await _route_fresh(
        dispatcher,
        ckpt=ckpt,
        actor_id="marlowe",
        intention=text,
        name="query_appearance_uses_perception_enrichment",
    )
    result = _dnd_result_from_output(case.output)
    facts = _fact_text(result)
    checks = [
        _check("dnd_fresh_schema", case.output["schema"] == "DndEventRouterOutput"),
        _check("query_or_harvest_kind", result.event_kind in {"query_response", "observation_harvest"}, result.event_kind),
        _check("private_or_enrichment_path", "kess" in _enrichment_ids(result) or result.event_kind == "query_response", _enrichment_ids(result)),
        _check("does_not_broadcast_query_to_all", not any(f["audience"] == "all_observers" and "what does kess" in f["text"].lower() for f in _facts(result)), _facts(result)),
        _check("no_procedural_leakage", not _contains_any(facts, ["query_response", "perception_enrichment", "router", "schema"]), facts),
    ]
    return CaseResult(**{**case.__dict__, "checks": checks})


CASES: list[Case] = [
    Case(
        name="unknown_space_gatekeeper_spawn",
        description="Calling into weakly implied darkness must not mint a speaking NPC.",
        commit_evidence=["652d509", "7050f50"],
        runner=_case_unknown_space_spawn,
    ),
    Case(
        name="hostile_attack_starts_combat",
        description="Visible hostile weapon attack starts D&D combat, not Cat II.",
        commit_evidence=["a3aa533", "f313ffa", "7ec235a"],
        runner=_case_hostile_attack_starts_combat,
    ),
    Case(
        name="hidden_ambush_before_visible_attack",
        description="Loaded hidden imminent danger starts initiative without leaking hidden source.",
        commit_evidence=["f313ffa", "7ec235a"],
        runner=_case_hidden_ambush_before_attack,
    ),
    Case(
        name="training_spar_starts_combat",
        description="Training-safe weapon exchange still starts initiative.",
        commit_evidence=["f313ffa", "7ec235a"],
        runner=_case_training_spar,
    ),
    Case(
        name="social_insight_requires_responder",
        description="Insight/social pressure against a present NPC opens Cat II without deciding their choice.",
        commit_evidence=["1099d8d"],
        runner=_case_social_cat_ii,
    ),
    Case(
        name="committed_agent_output_not_cat_ii",
        description="Committed NPC output is canonicalized; it is not a fresh attempted action.",
        commit_evidence=["05aad44"],
        runner=_case_agent_output_not_cat_ii,
    ),
    Case(
        name="mediated_perception_splits_audio_and_visual",
        description="D&D mediated channels must not leak audio-only and visual-only details across recipients.",
        commit_evidence=["f84a835"],
        runner=_case_mediated_perception,
    ),
    Case(
        name="repeated_defer_after_thin_handoff",
        description="Repeated defer after thin handoff should not invent player action or bounce empty control back.",
        commit_evidence=["5bdf10f"],
        runner=_case_repeated_defer,
    ),
    Case(
        name="query_appearance_uses_perception_enrichment",
        description="Appearance query should use query/enrichment flow without public/procedural leakage.",
        commit_evidence=["3f095eb"],
        runner=_case_query_perception,
    ),
]


async def _run_case(
    candidate: Candidate,
    dispatcher: LLMDispatcher,
    recording_client: RecordingClient,
    case: Case,
) -> CaseResult:
    recording_client.current_case = case.name
    try:
        return await case.runner(dispatcher)
    except Exception:
        return CaseResult(
            name=case.name,
            input_summary="",
            method="",
            output={},
            checks=[],
            error=traceback.format_exc(),
        )
    finally:
        recording_client.current_case = ""


def _candidate_config(
    base_config: LLMConfig,
    candidate: Candidate,
    *,
    timeout_seconds: float,
    max_retries: int,
) -> LLMConfig:
    role_models = dict(base_config.role_models)
    role_models["event_router"] = f"{candidate.provider}:{candidate.model}"
    role_providers = dict(base_config.role_providers)
    role_providers["event_router"] = candidate.provider
    reasoning = dict(base_config.openai_reasoning_efforts)
    if candidate.reasoning_effort:
        reasoning["event_router"] = candidate.reasoning_effort
    return base_config.model_copy(update={
        "role_models": role_models,
        "role_providers": role_providers,
        "openai_reasoning_efforts": reasoning,
        "timeout": timeout_seconds,
        "max_retries": max_retries,
        "retry_base_delay": 0.5,
        "retry_max_delay": 4.0,
    })


async def _run_candidate(
    candidate: Candidate,
    *,
    base_config: LLMConfig,
    cases: list[Case],
    include_prompts: bool,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    config = _candidate_config(
        base_config,
        candidate,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    missing = config.missing_credentials({"event_router"})
    if missing:
        names = ", ".join(missing[0].env_names)
        return {
            "label": candidate.label,
            "model": candidate.model,
            "provider": candidate.provider,
            "reasoning_effort": candidate.reasoning_effort,
            "error": f"Missing event_router credentials. Expected one of: {names}",
            "cases": [],
            "summary": {"passed_checks": 0, "total_checks": 0, "failed_cases": []},
        }

    inner = LLMClient(config)
    recording = RecordingClient(
        inner,
        candidate=candidate,
        raw_path=RAW_CALLS_PATH,
        include_prompts=include_prompts,
        call_timeout_seconds=timeout_seconds + 15.0,
    )
    dispatcher = LLMDispatcher(recording, PromptManager(str(REPO_ROOT / "app/prompts")))
    results: list[CaseResult] = []
    try:
        for case in cases:
            print(f"[{candidate.label}] running {case.name}...", flush=True)
            results.append(await _run_case(candidate, dispatcher, recording, case))
    finally:
        await recording.close()

    case_dicts = [_case_result_dict(result) for result in results]
    failed_cases = [
        item["name"]
        for item in case_dicts
        if item["error"] or any(not check["passed"] for check in item["checks"])
    ]
    passed_checks = sum(
        1
        for item in case_dicts
        for check in item["checks"]
        if check["passed"]
    )
    total_checks = sum(len(item["checks"]) for item in case_dicts)
    usage = _candidate_usage(candidate.label)
    return {
        "label": candidate.label,
        "model": candidate.model,
        "provider": candidate.provider,
        "reasoning_effort": candidate.reasoning_effort,
        "cases": case_dicts,
        "summary": {
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "failed_cases": failed_cases,
            "failure_count": len(failed_cases),
            "usage": usage,
        },
    }


def _candidate_usage(label: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    if not RAW_CALLS_PATH.exists():
        return totals
    for line in RAW_CALLS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("candidate") != label:
            continue
        for key, value in (item.get("usage") or {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def _case_result_dict(result: CaseResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "input_summary": result.input_summary,
        "method": result.method,
        "output": result.output,
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in result.checks
        ],
        "error": result.error,
    }


def _default_candidates(config: LLMConfig) -> list[Candidate]:
    configured_provider = config.provider_for_role("event_router")
    configured_model = config.model_for_role("event_router")
    configured_reasoning = (
        config.openai_reasoning_effort_for_role("event_router")
        if configured_provider == "openai"
        else ""
    )
    candidates = [
        Candidate(
            label="current_default",
            model=configured_model,
            provider=configured_provider,
            reasoning_effort=configured_reasoning,
        ),
        Candidate(label="gpt_5_mini", model="gpt-5-mini", reasoning_effort="medium"),
        Candidate(label="gpt_5_4_mini", model="gpt-5.4-mini", reasoning_effort="medium"),
    ]
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = f"{candidate.provider}:{candidate.model}:{candidate.reasoning_effort}"
        if candidate.label != "current_default" and key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _parse_candidate(raw: str) -> Candidate:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "candidate must be label=model or label=provider:model[:reasoning]"
        )
    label, spec = raw.split("=", 1)
    parts = spec.split(":")
    provider = "openai"
    reasoning = "medium"
    if len(parts) == 1:
        model = parts[0]
    elif len(parts) == 2:
        if parts[0] in {"openai", "anthropic"}:
            provider, model = parts
        else:
            model, reasoning = parts
    elif len(parts) == 3:
        provider, model, reasoning = parts
    else:
        raise argparse.ArgumentTypeError(f"invalid candidate spec: {raw!r}")
    if not label.strip() or not model.strip():
        raise argparse.ArgumentTypeError(f"invalid candidate spec: {raw!r}")
    return Candidate(
        label=label.strip(),
        model=model.strip(),
        provider=provider.strip(),
        reasoning_effort=reasoning.strip(),
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D&D Router Model Comparison",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Run directory: `{report['run_dir']}`",
        "",
        "## Candidates",
        "",
    ]
    for item in report["candidates"]:
        summary = item.get("summary") or {}
        lines.append(
            "- "
            f"`{item['label']}`: provider=`{item['provider']}` "
            f"model=`{item['model']}` reasoning=`{item.get('reasoning_effort', '')}` "
            f"checks={summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)} "
            f"failed_cases={summary.get('failed_cases', [])}"
        )
        usage = summary.get("usage") or {}
        if usage:
            lines.append(
                "  "
                f"usage: in={usage.get('prompt_tokens', 0)} "
                f"out={usage.get('completion_tokens', 0)} "
                f"reasoning={usage.get('reasoning_tokens', 0)} "
                f"total={usage.get('total_tokens', 0)}"
            )
    lines.extend(["", "## Case Matrix", ""])

    case_names = [case["name"] for case in report["case_catalog"]]
    header = "| Case | " + " | ".join(item["label"] for item in report["candidates"]) + " |"
    sep = "|---" + "|---" * len(report["candidates"]) + "|"
    lines.extend([header, sep])
    for case_name in case_names:
        row = [f"`{case_name}`"]
        for candidate in report["candidates"]:
            result = next(
                (item for item in candidate.get("cases", []) if item["name"] == case_name),
                None,
            )
            if result is None:
                row.append("not run")
            elif result.get("error"):
                row.append("ERROR")
            else:
                total = len(result["checks"])
                passed = sum(1 for check in result["checks"] if check["passed"])
                row.append(f"{passed}/{total}")
        lines.append("| " + " | ".join(row) + " |")

    for candidate in report["candidates"]:
        lines.extend(["", f"## {candidate['label']}", ""])
        if candidate.get("error"):
            lines.extend(["```text", candidate["error"], "```", ""])
            continue
        for item in candidate.get("cases", []):
            lines.extend([
                f"### {item['name']}",
                "",
                f"Input: {item.get('input_summary') or '(none)'}",
                "",
            ])
            if item.get("error"):
                lines.extend(["```text", item["error"].strip(), "```", ""])
                continue
            output = item["output"]
            lines.extend([
                f"schema=`{output.get('schema')}` "
                f"mode=`{output.get('interaction_mode')}` "
                f"event_kind=`{output.get('event_kind')}` "
                f"requires=`{output.get('required_responders')}`",
                f"next=`{output.get('next_output_character_ids')}` "
                f"combatants=`{output.get('combatant_ids')}` "
                f"spawns={len(output.get('combatant_spawns') or [])}",
                "",
                f"Rationale: {str(output.get('decision_rationale') or '').strip()}",
                "",
                "Checks:",
            ])
            for check in item["checks"]:
                mark = "PASS" if check["passed"] else "FAIL"
                detail = check.get("detail")
                detail_text = "" if detail in ("", [], {}, None) else f" - {detail}"
                lines.append(f"- {mark}: `{check['name']}`{detail_text}")
            lines.extend(["", "Observable facts:"])
            for fact in output.get("fact_texts", []):
                lines.append(
                    f"- [{fact['audience']} {fact['visible_to']}] {fact['text']}"
                )
            lines.append("")
    return "\n".join(lines)


def _select_cases(names: list[str]) -> list[Case]:
    if not names:
        return list(CASES)
    wanted = set(names)
    found = [case for case in CASES if case.name in wanted]
    missing = wanted - {case.name for case in found}
    if missing:
        raise SystemExit(f"Unknown case name(s): {', '.join(sorted(missing))}")
    return found


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv(REPO_ROOT / ".env")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CALLS_PATH.write_text("", encoding="utf-8")
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    base_config = LLMConfig.from_env()
    candidates = args.candidate or _default_candidates(base_config)
    cases = _select_cases(args.case)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(RUN_DIR),
        "raw_calls_path": str(RAW_CALLS_PATH),
        "case_catalog": [
            {
                "name": case.name,
                "description": case.description,
                "commit_evidence": case.commit_evidence,
            }
            for case in cases
        ],
        "candidates": [],
    }

    for candidate in candidates:
        report["candidates"].append(await _run_candidate(
            candidate,
            base_config=base_config,
            cases=cases,
            include_prompts=args.include_prompts,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        ))

    JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    MD_PATH.write_text(_markdown(report), encoding="utf-8")

    print(JSON_PATH)
    print(MD_PATH)
    print(RAW_CALLS_PATH)
    for candidate in report["candidates"]:
        summary = candidate.get("summary") or {}
        print(
            f"{candidate['label']}: "
            f"{summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)} "
            f"failed={summary.get('failed_cases', [])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        help=(
            "Candidate override. Use label=model, label=model:reasoning, "
            "or label=provider:model:reasoning. Defaults to current_default, "
            "gpt-5-mini, and gpt-5.4-mini."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only this case name. Can be repeated.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available cases and exit.",
    )
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include full rendered prompts in raw_calls.jsonl.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-provider request timeout for this harness.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Transient retry count for this harness.",
    )
    args = parser.parse_args()

    if args.list_cases:
        for case in CASES:
            print(f"{case.name}: {case.description}")
        return

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
