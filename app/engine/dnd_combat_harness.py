from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.engine import dnd_combat
from app.engine.dnd_combat_resolution import DndCombatResolver
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_cat_ii import (
    DndCombatManagerAdjudication,
    DndCombatTurnPlan,
)


_GLOBAL_FORBIDDEN_VISIBLE_FACT_TERMS = (
    "saving throw",
    "opportunity attack",
    "hit points",
    "spell slot",
    "concentrating",
    "roll ledger",
    "no attacks",
    "no attack was made",
    "no rolls",
    "no saving throw",
    "no damage",
    "no damage was",
)


@dataclass(frozen=True)
class HarnessReportPaths:
    run_dir: Path
    json_path: Path
    md_path: Path
    log_path: Path
    timestamp: str
    final_checkpoint_path: Path | None = None
    raw_calls_path: Path | None = None


def make_harness_report_paths(
    repo_root: Path,
    run_prefix: str,
    *,
    timestamp: str | None = None,
    include_final_checkpoint: bool = False,
    include_raw_calls: bool = False,
) -> HarnessReportPaths:
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = repo_root / "app/storage/playtest_reports" / f"{run_prefix}_{ts}"
    return HarnessReportPaths(
        run_dir=run_dir,
        json_path=run_dir / "report.json",
        md_path=run_dir / "report.md",
        log_path=run_dir / "run.log",
        timestamp=ts,
        final_checkpoint_path=(
            run_dir / "final_checkpoint.json"
            if include_final_checkpoint else None
        ),
        raw_calls_path=run_dir / "raw_calls.jsonl" if include_raw_calls else None,
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


@dataclass
class CombatCapture:
    actor_id: str
    intention: str
    packet: str = ""
    turn_plan: dict[str, Any] = field(default_factory=dict)
    roll_ledger: list[str] = field(default_factory=list)
    adjudication: dict[str, Any] = field(default_factory=dict)


class CapturingDndCombatResolver(DndCombatResolver):
    def __init__(self, client: LLMClient, prompt_mgr: PromptManager):
        super().__init__(client, prompt_mgr)
        self.captures: list[CombatCapture] = []
        self._active_actor_id = ""
        self._active_intention = ""
        self._active_capture: CombatCapture | None = None

    async def resolve_combat_action(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        intention: str,
    ):
        self._active_actor_id = actor_id
        self._active_intention = intention
        try:
            return await super().resolve_combat_action(
                ckpt=ckpt,
                actor_id=actor_id,
                intention=intention,
            )
        finally:
            self._active_actor_id = ""
            self._active_intention = ""
            self._active_capture = None

    async def _plan_turn(self, packet: str) -> DndCombatTurnPlan:
        plan = await super()._plan_turn(packet)
        self._active_capture = CombatCapture(
            actor_id=self._active_actor_id,
            intention=self._active_intention,
            packet=packet,
            turn_plan=plan.model_dump(mode="json"),
        )
        return plan

    async def _finalize(
        self,
        packet: str,
        ledger_lines: list[str],
        planned_actions_block: str,
    ) -> DndCombatManagerAdjudication:
        adjudication = await super()._finalize(
            packet,
            ledger_lines,
            planned_actions_block,
        )
        capture = self._active_capture or CombatCapture(
            actor_id=self._active_actor_id,
            intention=self._active_intention,
            packet=packet,
        )
        capture.roll_ledger = list(ledger_lines)
        capture.adjudication = adjudication.model_dump(mode="json")
        self.captures.append(capture)
        self._active_capture = None
        return adjudication


def _event_summary(
    event: Any,
    *,
    include_observers: bool = False,
    prefer_ends_beat_reason: bool = False,
) -> dict[str, Any]:
    if event is None:
        return {}
    event_kind = getattr(event, "event_kind", "")
    fact_details: list[dict[str, Any]] = []
    public_facts: list[str] = []
    private_facts: list[dict[str, Any]] = []
    for fact in getattr(event.canonical_event, "observable_facts", []):
        text = str(getattr(fact, "text", "") or "")
        audience = str(getattr(fact, "audience", "") or "all_observers")
        visible_to = [
            str(value)
            for value in getattr(fact, "visible_to", []) or []
            if str(value)
        ]
        detail = {
            "text": text,
            "audience": audience,
            "visible_to": visible_to,
        }
        fact_details.append(detail)
        if audience == "only":
            private_facts.append({"text": text, "visible_to": visible_to})
        else:
            public_facts.append(text)
    summary = {
        "event_id": getattr(event, "event_id", ""),
        "event_kind": event_kind,
        "decision_rationale": getattr(event, "decision_rationale", ""),
        "facts": public_facts,
        "private_facts": private_facts,
        "fact_details": fact_details,
    }
    if include_observers:
        summary["observers"] = [
            {
                "character_id": observer.character_id,
                "observation_level": observer.observation_level,
                "routing_role": observer.routing_role,
            }
            for observer in getattr(event, "observers", [])
        ]
    return summary


def _message_capture(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        content = str(message.get("content") or "")
        captured.append({
            "index": index,
            "role": str(message.get("role") or ""),
            "chars": len(content),
            "content": content,
        })
    return captured


def _capture_dump(capture: CombatCapture | None) -> dict[str, Any]:
    if capture is None:
        return {}
    turn_plan = capture.turn_plan or {}
    return {
        "actor_id": capture.actor_id,
        "intention": capture.intention,
        "packet": json.loads(capture.packet) if capture.packet else {},
        "turn_plan": turn_plan,
        "flattened_rolls": _flatten_turn_rolls(turn_plan),
        "roll_ledger": capture.roll_ledger or [],
        "adjudication": capture.adjudication or {},
    }


def _phase_label(response_model: object) -> str:
    name = getattr(response_model, "__name__", "")
    if name == "DndCombatTurnPlan":
        return "plan_turn"
    if name == "DndCombatManagerAdjudication":
        return "finalize_outcome"
    return str(name or "unstructured")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return repr(value)


def _scenario_cache_key(context: dict[str, Any]) -> str:
    return f"{context.get('index', '')}:{context.get('name', '')}"


def _cache_watch_for_call(
    context: dict[str, Any],
    *,
    phase: str,
    usage: dict[str, Any],
    plan_cache_reads_by_scenario: dict[str, int],
) -> dict[str, Any]:
    scenario_key = _scenario_cache_key(context)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    watch: dict[str, Any] = {
        "cache_read_input_tokens": cache_read,
    }
    if phase == "plan_turn":
        plan_cache_reads_by_scenario[scenario_key] = cache_read
        return watch
    if phase != "finalize_outcome":
        return watch
    plan_cache_read = plan_cache_reads_by_scenario.get(scenario_key)
    if plan_cache_read is None:
        return watch
    watch.update({
        "scenario_plan_cache_read_input_tokens": plan_cache_read,
        "finalize_cache_read_delta_from_plan": cache_read - plan_cache_read,
        "finalize_below_plan_cache_read": cache_read < plan_cache_read,
    })
    return watch


def _combat_summary(ckpt: CheckpointFile) -> dict[str, Any]:
    combat = ckpt.session.active_combat
    if combat is None:
        return {"active": False}
    summary = dnd_combat.public_status(ckpt.session)
    summary["active"] = True
    return summary


def combat_hp_by_id(ckpt: CheckpointFile) -> dict[str, int]:
    combat = ckpt.session.active_combat
    if combat is None:
        return {}
    return {
        combatant.character_id: int(combatant.hit_points_current)
        for combatant in combat.combatants
    }


def combat_reaction_by_id(ckpt: CheckpointFile) -> dict[str, bool]:
    combat = ckpt.session.active_combat
    if combat is None:
        return {}
    return {
        combatant.character_id: bool(combatant.reaction_available)
        for combatant in combat.combatants
    }


def cat_ii_resource_spends(ckpt: CheckpointFile) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for transaction in ckpt.session.cat_ii_roll_transactions:
        for spend in transaction.resource_spends:
            out.append(spend.model_dump(mode="json"))
    return out


def live_all_illegal_stress_turns(
    report: dict[str, Any],
    *,
    illegal_stress_actions: dict[str, str],
    illegal_action_index: int,
) -> list[dict[str, Any]]:
    illegal_turns: list[dict[str, Any]] = []
    for turn in report.get("turns") or []:
        actor_id = str(turn.get("actor_id") or "")
        if actor_id not in illegal_stress_actions:
            continue
        action_index = int(turn.get("actor_action_index") or 0)
        if action_index < 1:
            continue
        capture = turn.get("capture") or {}
        turn_plan = capture.get("turn_plan") or {}
        adjudication = capture.get("adjudication") or {}
        text = "\n".join([
            str(turn.get("intention") or ""),
            str(turn_plan.get("no_action_reason") or ""),
            str(adjudication.get("mechanical_summary") or ""),
            str(adjudication.get("fallback_reason") or ""),
            "\n".join(str(item) for item in capture.get("roll_ledger") or []),
            "\n".join(
                str(item)
                for item in adjudication.get("visible_outcome_facts") or []
            ),
        ])
        lowered = text.lower()
        forbidden_effects = [
            "hold person",
            "teleport",
            "phase",
            "twice",
            "set her shield arm on fire",
            "thirty feet away",
            "bonus spear attack",
            "both bram and aria",
            "heal myself",
            "full health",
            "fireball",
            "invisible",
            "drop her shield",
            "revivify",
            "four copies",
            "legendary actions",
            "mass cure wounds",
            "every cultist",
            "restrain everyone",
            "teleport to safety",
        ]
        rejection_terms = [
            "cannot",
            "unavailable",
            "not listed",
            "not a listed",
            "no roll",
            "no attack",
            "impossible",
            "illegal",
            "blocked",
            "out of range",
            "no effect",
            "not occur",
        ]
        visible_outcome_facts = adjudication.get("visible_outcome_facts") or []
        illegal_turns.append({
            "actor_id": actor_id,
            "turn_number": turn.get("turn_number"),
            "action_index": action_index,
            "is_pure_illegal_probe": action_index >= illegal_action_index,
            "intention": turn.get("intention"),
            "needs_rolls": bool(capture.get("flattened_rolls") or []),
            "flattened_rolls": capture.get("flattened_rolls") or [],
            "rejection_mentions": [
                term for term in rejection_terms if term in lowered
            ],
            "forbidden_mentions": [
                term for term in forbidden_effects if term in lowered
            ],
            "affirmed_illegal_mentions": _live_affirmed_term_mentions(
                "\n".join(str(fact) for fact in visible_outcome_facts),
                forbidden_effects,
            ),
            "visible_outcome_facts": visible_outcome_facts,
            "no_action_reason": turn_plan.get("no_action_reason") or "",
            "mechanical_summary": adjudication.get("mechanical_summary") or "",
            "fallback_reason": adjudication.get("fallback_reason") or "",
            "roll_ledger": capture.get("roll_ledger") or [],
            "stress_profile": illegal_stress_actions[actor_id],
        })
    return illegal_turns


def live_illegal_stress_turns(
    report: dict[str, Any],
    *,
    illegal_action_index: int,
) -> dict[str, dict[str, Any]]:
    illegal_by_actor: dict[str, dict[str, Any]] = {}
    for detail in report.get("all_illegal_stress_turns") or []:
        actor_id = str(detail.get("actor_id") or "")
        action_index = int(detail.get("action_index") or 0)
        if action_index >= illegal_action_index:
            illegal_by_actor[actor_id] = detail
    return illegal_by_actor


def live_report_checks(
    report: dict[str, Any],
    *,
    player_ids: tuple[str, ...],
    agent_monster_id: str,
    dummy_monster_ids: tuple[str, ...],
    illegal_action_index: int,
) -> list[dict[str, Any]]:
    turns = report.get("turns") or []
    role_calls = report.get("role_calls") or []
    max_turns = int(report.get("max_turns") or 0)
    acted_by_source: dict[str, set[str]] = {
        "player_canned": set(),
        "agent_llm": set(),
        "dummy_canned": set(),
    }
    illegal_turns = live_illegal_stress_turns(
        report,
        illegal_action_index=illegal_action_index,
    )
    for turn in turns:
        source = turn.get("source", "")
        if source in acted_by_source:
            acted_by_source[source].add(str(turn.get("actor_id", "")))
    active = bool((report.get("final_combat") or {}).get("active"))
    conversation_len = len(report.get("session_conversation") or [])
    agent_calls = _live_agent_role_calls(report)
    non_enemy_opportunity_attacks = _live_non_enemy_opportunity_attacks(report)
    agent_user_text = "\n\n".join(
        str((call.get("messages") or [{}])[-1].get("content") or "")
        for call in agent_calls
        if call.get("messages")
    )
    agent_turns = [
        turn for turn in turns if turn.get("source") == "agent_llm"
    ]
    expected_agent_turns = (
        3 if max_turns >= 15 else 2 if max_turns >= 9 else 1 if max_turns >= 3 else 0
    )
    return [
        _check("no_harness_error", not report.get("error"), report.get("error")),
        _check(
            "dummy_router_started_combat",
            bool(report.get("dummy_router_start")),
            report.get("dummy_router_start"),
        ),
        _check(
            "complex_battle_map_seeded",
            len((report.get("initial_map") or {}).get("tokens") or []) == 6
            and len((report.get("initial_map") or {}).get("terrain") or []) >= 4,
            report.get("initial_map"),
        ),
        _check(
            "two_player_characters_acted",
            acted_by_source["player_canned"] >= set(player_ids),
            sorted(acted_by_source["player_canned"]),
        ),
        _check(
            "agent_monster_used_llm",
            acted_by_source["agent_llm"] == {agent_monster_id},
            sorted(acted_by_source["agent_llm"]),
        ),
        _check(
            "agent_turn_count_matches_scenario",
            len(agent_turns) >= expected_agent_turns,
            {"expected_at_least": expected_agent_turns, "actual": len(agent_turns)},
        ),
        _check(
            "agent_prompt_captured_for_qa",
            bool(agent_calls and agent_user_text),
            agent_calls,
        ),
        _check(
            "agent_receives_combat_map_actions_and_context",
            all(
                needle in agent_user_text
                for needle in (
                    "## D&D Combat",
                    "Available combat actions:",
                    "Cinder Bolt",
                    "## Tactical Map",
                    "## Local Context",
                )
            ),
            agent_user_text,
        ),
        _check(
            "agent_private_intent_parsed",
            all(
                str((turn.get("source_detail") or {}).get("private_intent") or "")
                for turn in agent_turns
            ),
            [turn.get("source_detail") for turn in agent_turns],
        ),
        _check(
            "three_dummy_monsters_acted",
            acted_by_source["dummy_canned"] >= set(dummy_monster_ids),
            sorted(acted_by_source["dummy_canned"]),
        ),
        _check(
            "illegal_dummy_stress_turns_exercised",
            set(illegal_turns) >= set(dummy_monster_ids),
            illegal_turns,
        ),
        _check(
            "illegal_dummy_stress_rejected",
            _live_illegal_stress_rejected(illegal_turns, dummy_monster_ids),
            illegal_turns,
        ),
        _check(
            "no_non_enemy_opportunity_attacks",
            not non_enemy_opportunity_attacks,
            non_enemy_opportunity_attacks,
        ),
        _check(
            "combat_manager_calls_present",
            any(call.get("role") == "dnd_combat_manager" for call in role_calls),
            role_calls,
        ),
        _check(
            "no_generic_router_or_narrator_calls",
            not any(
                call.get("role") in {"event_router", "narrator"}
                for call in role_calls
            ),
            role_calls,
        ),
        _check(
            "router_history_scoped",
            (conversation_len == 0 if active else conversation_len <= 1),
            report.get("session_conversation"),
        ),
    ]


def live_quality_findings(
    report: dict[str, Any],
    *,
    illegal_action_index: int,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    for turn in report.get("turns") or []:
        adjudication = (turn.get("capture") or {}).get("adjudication") or {}
        for fact in adjudication.get("router_observed_facts") or []:
            observed.append({
                "turn_number": turn.get("turn_number"),
                "actor_id": turn.get("actor_id"),
                "fact": fact,
            })
    if observed:
        findings.append({
            "name": "review_router_observed_facts",
            "severity": "medium",
            "detail": observed,
        })

    for turn in report.get("turns") or []:
        if turn.get("source") != "agent_llm":
            continue
        public_text = str(
            (turn.get("source_detail") or {}).get("public_text") or ""
        )
        if any(
            label in public_text
            for label in ("**Action:**", "**Bonus Action:**", "**Movement:**")
        ):
            findings.append({
                "name": "agent_output_uses_mechanical_markdown",
                "severity": "low",
                "detail": {
                    "turn_number": turn.get("turn_number"),
                    "actor_id": turn.get("actor_id"),
                    "public_text": public_text,
                },
            })

    for turn in report.get("turns") or []:
        capture = turn.get("capture") or {}
        packet = capture.get("packet") or {}
        ac_by_id = {
            item.get("character_id"): item.get("armor_class")
            for item in packet.get("combatants") or []
        }
        for request in capture.get("flattened_rolls") or []:
            if request.get("kind") != "attack_roll":
                continue
            target_id = request.get("target_id")
            dc = request.get("dc")
            base_ac = ac_by_id.get(target_id)
            reason = str(request.get("reason") or "").lower()
            if (
                isinstance(dc, int)
                and isinstance(base_ac, int)
                and dc != base_ac
                and "cover" not in reason
            ):
                findings.append({
                    "name": "attack_dc_differs_without_cover_reason",
                    "severity": "high",
                    "detail": {
                        "turn_number": turn.get("turn_number"),
                        "actor_id": turn.get("actor_id"),
                        "target_id": target_id,
                        "dc": dc,
                        "base_ac": base_ac,
                        "reason": request.get("reason"),
                    },
                })
    illegal_turns = live_illegal_stress_turns(
        report,
        illegal_action_index=illegal_action_index,
    )
    for actor_id, detail in illegal_turns.items():
        if not detail.get("rejection_mentions"):
            findings.append({
                "name": "illegal_dummy_action_not_explicitly_rejected",
                "severity": "high",
                "detail": {"actor_id": actor_id, **detail},
            })
        if detail.get("needs_rolls"):
            findings.append({
                "name": "illegal_dummy_action_requested_rolls",
                "severity": "high",
                "detail": {"actor_id": actor_id, **detail},
            })
        if detail.get("affirmed_illegal_mentions"):
            findings.append({
                "name": "illegal_dummy_effect_leaked_to_outcome",
                "severity": "high",
                "detail": {"actor_id": actor_id, **detail},
            })
    for detail in _live_non_enemy_opportunity_attacks(report):
        findings.append({
            "name": "non_enemy_opportunity_attack_requested",
            "severity": "high",
            "detail": detail,
        })
    return findings


def live_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D&D Combat Manager Live Harness",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Run dir: `{report['run_dir']}`",
        f"Agent: `{report['roles']['agent']}`",
        f"Combat manager: `{report['roles']['dnd_combat_manager']}`",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {mark}: `{check['name']}`")
    lines.extend([
        "",
        "## Quality Findings",
        "",
    ])
    if report.get("quality_findings"):
        for finding in report["quality_findings"]:
            lines.append(
                f"- {finding.get('severity', 'info').upper()}: "
                f"`{finding.get('name')}`"
            )
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Illegal Stress Probes",
        "",
    ])
    for detail in report.get("all_illegal_stress_turns") or []:
        marker = "pure" if detail.get("is_pure_illegal_probe") else "mixed"
        roll_state = "rolls" if detail.get("needs_rolls") else "no rolls"
        reject_state = (
            "rejected" if detail.get("rejection_mentions") else "not rejected"
        )
        leaks = detail.get("affirmed_illegal_mentions") or []
        leak_state = f"; leaks: {', '.join(leaks)}" if leaks else ""
        lines.append(
            f"- Turn {detail.get('turn_number')}: `{detail.get('actor_id')}` "
            f"({marker}, {roll_state}, {reject_state}{leak_state})"
        )
    if not report.get("all_illegal_stress_turns"):
        lines.append("- None.")
    lines.extend([
        "",
        "## Usage",
        "",
        "```json",
        json.dumps(report.get("usage_totals") or {}, indent=2),
        "```",
        "",
        "## Turns",
        "",
    ])
    for turn in report["turns"]:
        lines.extend([
            f"### Turn {turn['turn_number']}: {turn['actor_id']}",
            "",
            f"Source: `{turn['source']}`",
            "",
            f"Intention: {turn['intention']}",
            "",
            "Facts:",
        ])
        for fact in (turn.get("result") or {}).get("facts") or []:
            lines.append(f"- {fact}")
        private_facts = (turn.get("result") or {}).get("private_facts") or []
        if private_facts:
            lines.extend(["", "Private facts:"])
            for fact in private_facts:
                visible_to = ", ".join(fact.get("visible_to") or [])
                lines.append(f"- [{visible_to}] {fact.get('text')}")
        adjudication = (turn.get("capture") or {}).get("adjudication") or {}
        observed = adjudication.get("router_observed_facts") or []
        if observed:
            lines.extend(["", "Router-observed facts:"])
            for fact in observed:
                lines.append(
                    f"- {fact.get('fact')} "
                    f"({fact.get('salience')}: {fact.get('reason')})"
                )
        lines.extend(["", "Roll ledger:"])
        for item in (turn.get("capture") or {}).get("roll_ledger") or []:
            lines.append(f"- {item}")
        if turn.get("source") == "agent_llm":
            detail = turn.get("source_detail") or {}
            lines.extend([
                "",
                "Agent QA:",
                f"- Public text: {detail.get('public_text', '')}",
                f"- Private intent: {detail.get('private_intent', '')}",
            ])
            for call in turn.get("role_calls") or []:
                if call.get("role") not in {
                    "agent",
                    "agent_standard",
                    "agent_convenience",
                }:
                    continue
                messages = call.get("messages") or []
                user_text = (
                    messages[-1].get("content", "") if messages else ""
                )
                lines.extend([
                    "- Agent received:",
                    "```text",
                    user_text,
                    "```",
                ])
        lines.append("")
    if report.get("error"):
        lines.extend(["## Error", "", "```text", report["error"], "```", ""])
    return "\n".join(lines)


def stress_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D&D Combat Manager Targeted Stress Harness",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Run dir: `{report['run_dir']}`",
        f"Raw calls: `{report.get('raw_calls_path', '')}`",
        f"Combat manager: `{report['roles']['dnd_combat_manager']}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks") or []:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {mark}: `{check['name']}`")
    lines.extend(["", "## Cache Watch", ""])
    if report.get("cache_watch_findings"):
        for finding in report["cache_watch_findings"]:
            lines.append(
                "- finalize cache read below same-scenario plan: "
                f"call `{finding.get('call_index')}` / "
                f"`{finding.get('scenario')}` "
                f"({finding.get('finalize_cache_read_input_tokens')} vs "
                f"{finding.get('plan_cache_read_input_tokens')}; "
                f"delta {finding.get('delta')})"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Router Observed Facts By Salience", ""])
    if report.get("router_observed_facts_by_salience"):
        for fact in report["router_observed_facts_by_salience"]:
            lines.append(
                f"- {str(fact.get('salience') or '').upper()}: "
                f"`{fact.get('scenario')}` - {fact.get('fact')} "
                f"({fact.get('reason')})"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Quality Findings", ""])
    if report.get("quality_findings"):
        for finding in report["quality_findings"]:
            lines.append(
                f"- {finding.get('severity', 'info').upper()}: "
                f"`{finding.get('scenario')}` / `{finding.get('name')}`"
            )
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Usage",
        "",
        "```json",
        json.dumps(report.get("usage_totals") or {}, indent=2),
        "```",
        "",
        "## Scenarios",
        "",
    ])
    for scenario in report.get("scenarios") or []:
        lines.extend([
            (
                f"### {scenario.get('index')}. {scenario.get('name')} "
                f"({scenario.get('category', 'active')})"
            ),
            "",
            scenario.get("summary") or "",
            "",
            f"Intention: {scenario.get('intention')}",
            "",
            "Findings:",
        ])
        if scenario.get("findings"):
            for finding in scenario["findings"]:
                lines.append(
                    f"- {finding.get('severity', 'info').upper()}: "
                    f"`{finding.get('name')}`"
                )
        else:
            lines.append("- None.")
        lines.extend(["", "Facts:"])
        for fact in ((scenario.get("event") or {}).get("facts") or []):
            lines.append(f"- {fact}")
        private_facts = (scenario.get("event") or {}).get("private_facts") or []
        if private_facts:
            lines.extend(["", "Private facts:"])
            for fact in private_facts:
                visible_to = ", ".join(fact.get("visible_to") or [])
                lines.append(f"- [{visible_to}] {fact.get('text')}")
        lines.extend(["", "Turn plan:"])
        turn_plan = (scenario.get("capture") or {}).get("turn_plan") or {}
        lines.extend([
            "```json",
            json.dumps(turn_plan, indent=2),
            "```",
            "",
        ])
    if report.get("error"):
        lines.extend(["## Error", "", "```text", report["error"], "```", ""])
    return "\n".join(lines)


def _live_agent_role_calls(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        call for call in report.get("role_calls") or []
        if call.get("role") in {"agent", "agent_standard", "agent_convenience"}
    ]


def _live_affirmed_term_mentions(text: str, terms: list[str]) -> list[str]:
    denial_markers = (
        "cannot",
        "can't",
        "could not",
        "did not",
        "does not",
        "do not",
        "no ",
        "not ",
        "unavailable",
        "lacks",
        "without",
        "not occur",
        "no effect",
        "failed",
        "missed",
    )
    sentences = [
        sentence.strip().lower()
        for sentence in text.replace("\n", ". ").replace(";", ".").split(".")
        if sentence.strip()
    ]
    affirmed: list[str] = []
    for term in terms:
        term_lower = term.lower()
        for sentence in sentences:
            if term_lower not in sentence:
                continue
            if any(marker in sentence for marker in denial_markers):
                continue
            affirmed.append(term)
            break
    return affirmed


def _live_illegal_stress_rejected(
    turns: dict[str, dict[str, Any]],
    dummy_monster_ids: tuple[str, ...],
) -> bool:
    if set(turns) < set(dummy_monster_ids):
        return False
    for detail in turns.values():
        if not detail.get("rejection_mentions"):
            return False
        if detail.get("needs_rolls"):
            return False
        if detail.get("affirmed_illegal_mentions"):
            return False
    return True


def _live_non_enemy_opportunity_attacks(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for turn in report.get("turns") or []:
        capture = turn.get("capture") or {}
        packet = capture.get("packet") or {}
        combatants = {
            str(
                combatant.get("character_id")
                or combatant.get("combatant_id")
                or ""
            ): combatant
            for combatant in packet.get("combatants") or []
        }
        for request in capture.get("flattened_rolls") or []:
            reason = str(request.get("reason") or "")
            roll_id = str(request.get("roll_id") or "")
            opportunity_text = f"{roll_id} {reason}".lower()
            if "opportunity" not in opportunity_text:
                continue
            if any(
                denial in opportunity_text
                for denial in (
                    "no opportunity",
                    "not provoke",
                    "not provoked",
                    "does not provoke",
                    "without provoking",
                )
            ):
                continue
            if not (
                roll_id.lower().startswith(("oa", "opp"))
                or "opportunity attack" in opportunity_text
            ):
                continue
            actor_id = str(request.get("actor_id") or "")
            actor = combatants.get(actor_id) or {}
            relationship = str(
                actor.get("relationship_to_current_actor") or ""
            )
            if relationship == "enemy":
                continue
            findings.append({
                "turn_number": turn.get("turn_number"),
                "current_actor_id": turn.get("actor_id"),
                "opportunity_actor_id": actor_id,
                "target_id": request.get("target_id"),
                "relationship_to_current_actor": relationship or "unknown",
                "reason": reason,
            })
    return findings


def _flattened_rolls(result: dict[str, Any]) -> list[dict[str, Any]]:
    return (result.get("capture") or {}).get("flattened_rolls") or []


def _turn_plan_actions(result: dict[str, Any]) -> list[dict[str, Any]]:
    turn_plan = (result.get("capture") or {}).get("turn_plan") or {}
    return [
        action for action in turn_plan.get("actions") or []
        if isinstance(action, dict)
    ]


def _flatten_turn_rolls(turn_plan: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for action in turn_plan.get("actions") or []:
        if not isinstance(action, dict):
            continue
        for roll in action.get("rolls") or []:
            if not isinstance(roll, dict):
                continue
            flattened = dict(roll)
            flattened["actor_id"] = action.get("actor_id", "")
            flattened["action_id"] = action.get("source_id", "")
            flattened["source_type"] = action.get("source_type", "")
            flattened["source_id"] = action.get("source_id", "")
            flattened["effect_id"] = action.get("effect_id", "")
            flattened["economy"] = action.get("economy", "")
            flattened["use_mode"] = action.get("use_mode", "")
            out.append(flattened)
    return out


def _adjudication(result: dict[str, Any]) -> dict[str, Any]:
    return (result.get("capture") or {}).get("adjudication") or {}


def _packet(result: dict[str, Any]) -> dict[str, Any]:
    return (result.get("capture") or {}).get("packet") or {}


def _visible_facts(result: dict[str, Any]) -> list[str]:
    adjudication = _adjudication(result)
    facts = [
        str(fact)
        for fact in adjudication.get("visible_outcome_facts") or []
        if str(fact).strip()
    ]
    facts.extend(
        str(fact)
        for fact in ((result.get("event") or {}).get("facts") or [])
        if str(fact).strip()
    )
    return facts


def _scenario_findings(result: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    expectations = result.get("expectations") or {}
    requests = _flattened_rolls(result)
    actions = _turn_plan_actions(result)
    adjudication = _adjudication(result)
    packet = _packet(result)
    request_targets = {
        str(request.get("target_id") or "")
        for request in requests
        if str(request.get("target_id") or "")
    }
    save_targets = {
        str(request.get("target_id") or "")
        for request in requests
        if request.get("kind") == "saving_throw"
    }
    request_kinds = {
        str(request.get("kind") or "")
        for request in requests
        if str(request.get("kind") or "")
    }
    attack_requests = [
        request for request in requests if request.get("kind") == "attack_roll"
    ]
    if result.get("error"):
        findings.append({
            "name": "scenario_error",
            "severity": "critical",
            "detail": result.get("error"),
        })
    effect_actions_missing_ids = [
        action for action in actions
        if action.get("source_type") == "effect"
        and not str(action.get("effect_id") or "").strip()
    ]
    if effect_actions_missing_ids:
        findings.append({
            "name": "effect_action_missing_effect_id",
            "severity": "high",
            "detail": effect_actions_missing_ids,
        })
    effect_actions_wrong_mode = [
        action for action in actions
        if action.get("source_type") == "effect"
        and action.get("use_mode") not in {"release", "sustain"}
    ]
    if effect_actions_wrong_mode:
        findings.append({
            "name": "effect_action_wrong_use_mode",
            "severity": "high",
            "detail": effect_actions_wrong_mode,
        })
    forbidden_action_source_ids = {
        str(source_id).strip().lower()
        for source_id in expectations.get("forbid_action_source_ids") or []
        if str(source_id).strip()
    }
    if forbidden_action_source_ids:
        bad_actions = [
            action for action in actions
            if str(action.get("source_id") or "").strip().lower()
            in forbidden_action_source_ids
        ]
        if bad_actions:
            findings.append({
                "name": "forbidden_action_source_used",
                "severity": "high",
                "detail": bad_actions,
            })
    required_action_matches = expectations.get("require_action_matches") or []
    if isinstance(required_action_matches, dict):
        required_action_matches = [required_action_matches]
    for required in required_action_matches:
        if not any(_action_matches_required(action, required) for action in actions):
            findings.append({
                "name": "missing_required_action_match",
                "severity": "high",
                "detail": required,
            })
    expected_status = str(expectations.get("expected_combat_status") or "").strip()
    if expected_status:
        actual_status = str(adjudication.get("combat_status") or "").strip()
        if actual_status != expected_status:
            findings.append({
                "name": "wrong_combat_status",
                "severity": "high",
                "detail": {
                    "expected": expected_status,
                    "actual": actual_status,
                },
            })
    router_facts = [
        fact for fact in adjudication.get("router_observed_facts") or []
        if isinstance(fact, dict) and str(fact.get("fact") or "").strip()
    ]
    if expectations.get("forbid_router_observed_facts") and router_facts:
        findings.append({
            "name": "unexpected_router_observed_facts",
            "severity": "medium",
            "detail": router_facts,
        })
    minimum_router_facts = int(
        expectations.get("minimum_router_observed_facts", 0) or 0
    )
    if len(router_facts) < minimum_router_facts:
        findings.append({
            "name": "missing_router_observed_facts",
            "severity": "high",
            "detail": {
                "minimum": minimum_router_facts,
                "actual": router_facts,
            },
        })
    required_router_terms = [
        str(term).strip().lower()
        for term in expectations.get("router_fact_must_contain_any") or []
        if str(term).strip()
    ]
    if required_router_terms:
        fact_text = " ".join(
            str(fact.get("fact") or "").lower() for fact in router_facts
        )
        if not any(term in fact_text for term in required_router_terms):
            findings.append({
                "name": "router_observed_fact_missing_expected_terms",
                "severity": "medium",
                "detail": {
                    "expected_any": required_router_terms,
                    "actual": router_facts,
                },
            })
    forbidden_router_reason_terms = [
        str(term).strip().lower()
        for term in expectations.get("router_reason_must_not_contain_any") or []
        if str(term).strip()
    ]
    if forbidden_router_reason_terms:
        bad_router_reasons = [
            fact for fact in router_facts
            if any(
                term in str(fact.get("reason") or "").lower()
                for term in forbidden_router_reason_terms
            )
        ]
        if bad_router_reasons:
            findings.append({
                "name": "router_observed_fact_has_tactical_reason",
                "severity": "medium",
                "detail": bad_router_reasons,
            })
    for key in expectations.get("packet_forbidden_keys") or []:
        if _dict_contains_key(packet, str(key)):
            findings.append({
                "name": "packet_forbidden_key_present",
                "severity": "high",
                "detail": {"key": key},
            })
    if expectations.get("forbid_rolls") and requests:
        findings.append({
            "name": "unexpected_rolls_requested",
            "severity": "high",
            "detail": requests,
        })
    missing_kinds = sorted(
        set(expectations.get("must_include_roll_kinds") or []) - request_kinds
    )
    if missing_kinds:
        findings.append({
            "name": "missing_required_roll_kinds",
            "severity": "high",
            "detail": missing_kinds,
        })
    minimum_kind_counts = expectations.get("minimum_roll_kind_count") or {}
    for kind, minimum in minimum_kind_counts.items():
        actual = sum(1 for request in requests if request.get("kind") == kind)
        if actual < int(minimum):
            findings.append({
                "name": "insufficient_required_roll_kind_count",
                "severity": "high",
                "detail": {"kind": kind, "minimum": minimum, "actual": actual},
            })
    any_kinds = set(expectations.get("must_include_any_roll_kinds") or [])
    if any_kinds and not any_kinds.intersection(request_kinds):
        findings.append({
            "name": "missing_any_required_roll_kind",
            "severity": "high",
            "detail": sorted(any_kinds),
        })
    missing_targets = sorted(
        set(expectations.get("must_include_roll_targets") or []) - request_targets
    )
    if missing_targets:
        findings.append({
            "name": "missing_required_roll_targets",
            "severity": "high",
            "detail": missing_targets,
        })
    missing_save_targets = sorted(
        set(expectations.get("must_include_save_targets") or []) - save_targets
    )
    if missing_save_targets:
        findings.append({
            "name": "missing_required_save_targets",
            "severity": "high",
            "detail": missing_save_targets,
        })
    required_effect_id = str(expectations.get("require_roll_effect_id") or "")
    if required_effect_id and not any(
        str(request.get("effect_id") or "") == required_effect_id
        for request in requests
    ):
        findings.append({
            "name": "missing_required_roll_effect_id",
            "severity": "high",
            "detail": required_effect_id,
        })
    forbidden_targets = sorted(
        set(expectations.get("must_exclude_roll_targets") or []) & request_targets
    )
    if forbidden_targets:
        findings.append({
            "name": "forbidden_targets_received_rolls",
            "severity": "high",
            "detail": forbidden_targets,
        })
    if expectations.get("forbid_attack_rolls") and attack_requests:
        findings.append({
            "name": "aoe_spell_requested_attack_rolls",
            "severity": "high",
            "detail": attack_requests,
        })
    if expectations.get("forbid_initial_save_effect_id"):
        bad_effect_save_ids = [
            request for request in requests
            if request.get("kind") == "saving_throw"
            and str(request.get("effect_id") or "").strip()
        ]
        if bad_effect_save_ids:
            findings.append({
                "name": "initial_save_effect_id_present",
                "severity": "medium",
                "detail": bad_effect_save_ids,
            })
    if expectations.get("forbid_attack_modifier_bonus"):
        bad_attack_modifiers = [
            request for request in attack_requests
            if int(request.get("modifier_bonus", 0) or 0) != 0
        ]
        if bad_attack_modifiers:
            findings.append({
                "name": "attack_roll_modifier_bonus_misused",
                "severity": "medium",
                "detail": bad_attack_modifiers,
            })
    expected_advantage = expectations.get("expected_advantage_by_target") or {}
    for target_id, expected_state in expected_advantage.items():
        target_requests = [
            request for request in requests
            if request.get("target_id") == target_id
        ]
        if not target_requests:
            findings.append({
                "name": "missing_advantage_checked_target",
                "severity": "high",
                "detail": {"target_id": target_id, "expected": expected_state},
            })
            continue
        if not any(
            request.get("advantage_state") == expected_state
            for request in target_requests
        ):
            findings.append({
                "name": "wrong_advantage_state",
                "severity": "high",
                "detail": {
                    "target_id": target_id,
                    "expected": expected_state,
                    "actual": [
                        request.get("advantage_state")
                        for request in target_requests
                    ],
                },
            })
    if expectations.get("require_opposed_rolls") and not _has_opposed_rolls(requests):
        findings.append({
            "name": "missing_opposed_roll_structure",
            "severity": "high",
            "detail": requests,
        })
    expected_ability = str(expectations.get("expected_save_ability") or "")
    if expected_ability:
        wrong = [
            request for request in requests
            if request.get("kind") == "saving_throw"
            and request.get("ability") != expected_ability
        ]
        if wrong:
            findings.append({
                "name": "wrong_save_ability",
                "severity": "high",
                "detail": wrong,
            })
    opportunity_from = str(expectations.get("must_include_opportunity_from") or "")
    if opportunity_from:
        if not any(
            request.get("actor_id") == opportunity_from
            and "opportunity" in (
                f"{request.get('roll_id', '')} {request.get('reason', '')}".lower()
            )
            for request in requests
        ):
            findings.append({
                "name": "missing_opportunity_attack",
                "severity": "high",
                "detail": opportunity_from,
            })
    forbidden_opportunity_from = expectations.get("forbid_opportunity_from") or []
    if isinstance(forbidden_opportunity_from, str):
        forbidden_opportunity_from = [forbidden_opportunity_from]
    illegal_opportunity_requests = [
        request for request in requests
        if request.get("actor_id") in set(forbidden_opportunity_from)
        and "opportunity" in (
            f"{request.get('roll_id', '')} {request.get('reason', '')}".lower()
        )
    ]
    if illegal_opportunity_requests:
        findings.append({
            "name": "forbidden_opportunity_attack",
            "severity": "high",
            "detail": illegal_opportunity_requests,
        })
    reason_terms = expectations.get("target_reason_contains") or {}
    for target_id, terms in reason_terms.items():
        target_reasons = " ".join(
            (
                f"{request.get('reason', '')} "
                f"{request.get('modifier_bonus_reason', '')}"
            ).lower()
            for request in requests
            if request.get("target_id") == target_id
        )
        missing_terms = [
            term for term in terms
            if str(term).lower() not in target_reasons
        ]
        if missing_terms:
            findings.append({
                "name": "roll_reason_missing_expected_terms",
                "severity": "medium",
                "detail": {"target_id": target_id, "missing": missing_terms},
            })
    friendly_fire = sorted(
        set(expectations.get("friendly_fire_targets") or []) - request_targets
    )
    if friendly_fire:
        findings.append({
            "name": "friendly_fire_targets_omitted",
            "severity": "high",
            "detail": friendly_fire,
        })
    cover_save_targets = expectations.get("cover_should_matter_for_dex_save_targets")
    for target_id in cover_save_targets or []:
        target_save_requests = [
            request for request in requests
            if request.get("target_id") == target_id
            and request.get("kind") == "saving_throw"
        ]
        target_reasons = " ".join(
            (
                f"{request.get('reason', '')} "
                f"{request.get('modifier_bonus_reason', '')}"
            ).lower()
            for request in target_save_requests
        )
        if not target_reasons:
            continue
        if not any(
            int(request.get("modifier_bonus", 0) or 0) > 0
            for request in target_save_requests
        ):
            findings.append({
                "name": "dex_save_cover_bonus_missing",
                "severity": "high",
                "detail": {"target_id": target_id},
            })
        if "cover" not in target_reasons:
            findings.append({
                "name": "dex_save_cover_not_considered",
                "severity": "medium",
                "detail": {"target_id": target_id},
            })
        elif "does not change" in target_reasons or "not change" in target_reasons:
            findings.append({
                "name": "dex_save_cover_explicitly_ignored",
                "severity": "high",
                "detail": {
                    "target_id": target_id,
                    "reason": target_reasons,
                },
            })
    if expectations.get("expected_save_damage_spell"):
        applied_damage_records = _applied_damage_records(result)
        unapplied_damage_records = _unapplied_damage_records(result)
        hp_changed = result.get("before_hp") != result.get("after_hp")
        if save_targets and not applied_damage_records and not hp_changed:
            findings.append({
                "name": "save_damage_has_no_structured_application",
                "severity": "critical",
                "detail": {
                    "save_targets": sorted(save_targets),
                    "unapplied_damage_records": unapplied_damage_records,
                    "note": (
                        "Save-damage spells should produce structured "
                        "damage_records through the engine damage path."
                    ),
                },
            })
    required_failed_targets = sorted(
        set(expectations.get("must_fail_save_targets") or [])
        - _failed_save_targets(result)
    )
    if required_failed_targets:
        findings.append({
            "name": "required_save_targets_did_not_fail",
            "severity": "high",
            "detail": required_failed_targets,
        })
    if expectations.get("forbid_damage_records") and _damage_records(result):
        findings.append({
            "name": "unexpected_damage_records",
            "severity": "high",
            "detail": _damage_records(result),
        })
    if expectations.get("forbid_hp_change"):
        before_hp = result.get("before_hp") or {}
        after_hp = result.get("after_hp") or {}
        changed = {
            cid: {
                "before": before_hp.get(cid),
                "after": after_hp.get(cid),
            }
            for cid in sorted(set(before_hp.keys()) & set(after_hp.keys()))
            if before_hp.get(cid) != after_hp.get(cid)
        }
        if changed:
            findings.append({
                "name": "unexpected_hp_change",
                "severity": "high",
                "detail": changed,
            })
    required_hp_decrease_if_failed = expectations.get("require_hp_decrease_if_failed")
    if required_hp_decrease_if_failed and _failed_save_targets(result):
        missing_decreases = []
        before_hp = result.get("before_hp") or {}
        after_hp = result.get("after_hp") or {}
        for cid in required_hp_decrease_if_failed:
            before = before_hp.get(cid)
            after = after_hp.get(cid)
            if before is None or after is None or after >= before:
                missing_decreases.append({
                    "character_id": cid,
                    "before": before,
                    "after": after,
                })
        if missing_decreases:
            findings.append({
                "name": "failed_save_missing_expected_hp_decrease",
                "severity": "high",
                "detail": missing_decreases,
            })
    if expectations.get("forbid_resource_spends") and result.get("resource_spends"):
        findings.append({
            "name": "unexpected_resource_spends",
            "severity": "high",
            "detail": result.get("resource_spends"),
        })
    required_reaction_spent = str(expectations.get("require_reaction_spent") or "")
    if required_reaction_spent:
        reactions = result.get("after_reactions") or {}
        if reactions.get(required_reaction_spent) is not False:
            findings.append({
                "name": "required_reaction_not_spent",
                "severity": "high",
                "detail": {
                    "character_id": required_reaction_spent,
                    "after_reactions": reactions,
                },
            })
    visible_text = " ".join(str(fact) for fact in (result.get("event") or {}).get("facts") or [])
    if "concentration shifts to a new effect" in visible_text.lower():
        findings.append({
            "name": "same_spell_concentration_effects_churned",
            "severity": "high",
            "detail": (
                "Multiple target effects from one concentration spell caused "
                "earlier effects to be ended as if concentration shifted."
            ),
        })
    global_bad_visible = [
        {"term": term, "fact": fact}
        for term in _GLOBAL_FORBIDDEN_VISIBLE_FACT_TERMS
        for fact in _visible_facts(result)
        if term in fact.lower()
    ]
    if global_bad_visible:
        findings.append({
            "name": "global_forbidden_visible_fact",
            "severity": "medium",
            "detail": global_bad_visible,
        })
    if expectations.get("require_private_fact_for_failed_save_targets"):
        failed_targets = _failed_save_targets(result)
        required_terms = [
            str(term).strip().lower()
            for term in expectations.get("private_fact_must_contain_any") or []
            if str(term).strip()
        ]
        matching_private = [
            fact for fact in _private_outcome_facts(result)
            if _private_fact_matches_terms(fact, required_terms)
        ]
        if not failed_targets:
            findings.append({
                "name": "no_failed_save_targets_for_private_fact_check",
                "severity": "high",
                "detail": requests,
            })
        if not matching_private:
            findings.append({
                "name": "missing_required_private_outcome_facts",
                "severity": "high",
                "detail": {
                    "failed_targets": sorted(failed_targets),
                    "required_terms": required_terms,
                    "private_outcome_facts": _private_outcome_facts(result),
                },
            })
        missing_private_targets = sorted(
            target_id
            for target_id in failed_targets
            if not any(
                target_id in set(fact.get("visible_to") or [])
                for fact in matching_private
            )
        )
        if missing_private_targets:
            findings.append({
                "name": "failed_save_targets_missing_private_facts",
                "severity": "high",
                "detail": {
                    "missing_targets": missing_private_targets,
                    "private_outcome_facts": matching_private,
                },
            })
        if expectations.get("private_fact_forbid_visible_to_non_failed"):
            bad_private_scope = []
            for fact in matching_private:
                extra = sorted(set(fact.get("visible_to") or []) - failed_targets)
                if extra:
                    bad_private_scope.append({
                        "text": fact.get("text"),
                        "visible_to": fact.get("visible_to"),
                        "non_failed_visible_to": extra,
                    })
            if bad_private_scope:
                findings.append({
                    "name": "private_fact_visible_to_non_failed_targets",
                    "severity": "high",
                    "detail": bad_private_scope,
                })
    required_private_targets = {
        str(target_id).strip()
        for target_id in expectations.get("require_private_fact_for_targets") or []
        if str(target_id).strip()
    }
    if required_private_targets:
        required_terms = [
            str(term).strip().lower()
            for term in expectations.get("private_fact_must_contain_any") or []
            if str(term).strip()
        ]
        matching_private = [
            fact for fact in _private_outcome_facts(result)
            if _private_fact_matches_terms(fact, required_terms)
        ]
        if not matching_private:
            findings.append({
                "name": "missing_required_private_outcome_facts",
                "severity": "high",
                "detail": {
                    "required_targets": sorted(required_private_targets),
                    "required_terms": required_terms,
                    "private_outcome_facts": _private_outcome_facts(result),
                },
            })
        missing_private_targets = sorted(
            target_id
            for target_id in required_private_targets
            if not any(
                target_id in set(fact.get("visible_to") or [])
                for fact in matching_private
            )
        )
        if missing_private_targets:
            findings.append({
                "name": "private_fact_targets_missing",
                "severity": "high",
                "detail": {
                    "missing_targets": missing_private_targets,
                    "private_outcome_facts": matching_private,
                },
            })
        if expectations.get("private_fact_forbid_visible_to_non_targets"):
            bad_private_scope = []
            for fact in matching_private:
                extra = sorted(
                    set(fact.get("visible_to") or []) - required_private_targets
                )
                if extra:
                    bad_private_scope.append({
                        "text": fact.get("text"),
                        "visible_to": fact.get("visible_to"),
                        "unexpected_visible_to": extra,
                    })
            if bad_private_scope:
                findings.append({
                    "name": "private_fact_visible_to_unexpected_targets",
                    "severity": "high",
                    "detail": bad_private_scope,
                })
    for rule in expectations.get("condition_fact_requires_delta") or []:
        condition = str(rule.get("condition") or "").strip().lower()
        target_id = str(rule.get("target_id") or "").strip()
        if not condition or not any(
            _fact_asserts_condition(fact, condition)
            for fact in _visible_facts(result)
        ):
            continue
        if not any(
            delta.get("kind") == "condition_add"
            and delta.get("target_id") == target_id
            and str(delta.get("condition") or "").strip().lower() == condition
            for delta in adjudication.get("combat_state_deltas") or []
        ):
            findings.append({
                "name": "condition_fact_missing_state_delta",
                "severity": "high",
                "detail": {
                    "target_id": target_id,
                    "condition": condition,
                },
            })
    required_spatial = str(expectations.get("require_spatial_delta_kind") or "")
    if required_spatial:
        if not any(
            delta.get("kind") == required_spatial
            for delta in adjudication.get("spatial_deltas") or []
        ):
            findings.append({
                "name": "missing_required_spatial_delta",
                "severity": "high",
                "detail": required_spatial,
            })
    forbidden_spatial_kinds = {
        str(kind).strip()
        for kind in expectations.get("forbid_spatial_delta_kinds") or []
        if str(kind).strip()
    }
    if forbidden_spatial_kinds:
        bad_spatial_deltas = [
            delta for delta in adjudication.get("spatial_deltas") or []
            if str(delta.get("kind") or "") in forbidden_spatial_kinds
        ]
        if bad_spatial_deltas:
            findings.append({
                "name": "forbidden_spatial_delta_kind",
                "severity": "high",
                "detail": bad_spatial_deltas,
            })
    required_spatial_if_failed = str(
        expectations.get("require_spatial_delta_if_failed") or ""
    )
    if required_spatial_if_failed and _failed_save_targets(result):
        if not any(
            delta.get("kind") == required_spatial_if_failed
            for delta in adjudication.get("spatial_deltas") or []
        ):
            findings.append({
                "name": "failed_save_missing_spatial_delta",
                "severity": "high",
                "detail": required_spatial_if_failed,
            })
    expected_move_if_failed = expectations.get("expected_move_if_failed") or {}
    if expected_move_if_failed and _failed_save_targets(result):
        target_id = str(expected_move_if_failed.get("target_id") or "")
        expected_x = expected_move_if_failed.get("x")
        expected_y = expected_move_if_failed.get("y")
        matching_moves = [
            delta for delta in adjudication.get("spatial_deltas") or []
            if delta.get("kind") == "move_token"
            and str(delta.get("target_id") or "") == target_id
        ]
        if not any(
            delta.get("x") == expected_x and delta.get("y") == expected_y
            for delta in matching_moves
        ):
            findings.append({
                "name": "wrong_forced_movement_destination",
                "severity": "high",
                "detail": {
                    "expected": expected_move_if_failed,
                    "actual": matching_moves,
                },
            })
    required_spatial_matches = expectations.get("require_spatial_delta_matches") or []
    if isinstance(required_spatial_matches, dict):
        required_spatial_matches = [required_spatial_matches]
    for required in required_spatial_matches:
        if not any(
            _spatial_delta_matches_required(delta, required)
            for delta in adjudication.get("spatial_deltas") or []
        ):
            findings.append({
                "name": "missing_required_spatial_delta_match",
                "severity": "high",
                "detail": required,
            })
    required_effect_ends = expectations.get("require_effect_end_slug") or []
    if isinstance(required_effect_ends, str):
        required_effect_ends = [required_effect_ends]
    for required_slug in required_effect_ends:
        if not any(
            delta.get("operation") == "end"
            and _effect_delta_contains(delta, str(required_slug))
            for delta in adjudication.get("effect_deltas") or []
        ):
            findings.append({
                "name": "missing_required_effect_end",
                "severity": "high",
                "detail": required_slug,
            })
    known_effect_targets = {
        str(value or "")
        for combatant in packet.get("combatants") or []
        if isinstance(combatant, dict)
        for value in (
            combatant.get("combatant_id"),
            combatant.get("character_id"),
        )
        if str(value or "")
    }
    bad_effect_targets = [
        delta for delta in adjudication.get("effect_deltas") or []
        if str(delta.get("target_id") or "")
        and str(delta.get("target_id") or "") not in known_effect_targets
    ]
    if bad_effect_targets:
        findings.append({
            "name": "effect_delta_targets_non_combatant",
            "severity": "high",
            "detail": bad_effect_targets,
        })
    if expectations.get("forbid_effect_delta_target_ids_from_spatial_areas"):
        area_target_ids = {
            str(delta.get("target_id") or "")
            for delta in adjudication.get("spatial_deltas") or []
            if delta.get("kind") == "add_area"
            and str(delta.get("target_id") or "")
        }
        bad_area_effect_targets = [
            delta for delta in adjudication.get("effect_deltas") or []
            if str(delta.get("target_id") or "") in area_target_ids
        ]
        if bad_area_effect_targets:
            findings.append({
                "name": "effect_delta_targets_spatial_area",
                "severity": "high",
                "detail": bad_area_effect_targets,
            })
    required_effect_delta_matches = (
        expectations.get("require_effect_delta_matches") or []
    )
    if isinstance(required_effect_delta_matches, dict):
        required_effect_delta_matches = [required_effect_delta_matches]
    for required in required_effect_delta_matches:
        if not any(
            _effect_delta_matches_required(delta, required)
            for delta in adjudication.get("effect_deltas") or []
        ):
            findings.append({
                "name": "missing_required_effect_delta_match",
                "severity": "high",
                "detail": required,
            })
    forbidden_effect_source_ids = {
        str(source_id).strip().lower()
        for source_id in expectations.get("forbid_effect_delta_source_ids") or []
        if str(source_id).strip()
    }
    if forbidden_effect_source_ids:
        bad_effect_sources = [
            delta for delta in adjudication.get("effect_deltas") or []
            if str(delta.get("source_id") or "").strip().lower()
            in forbidden_effect_source_ids
        ]
        if bad_effect_sources:
            findings.append({
                "name": "forbidden_effect_delta_source_id",
                "severity": "medium",
                "detail": bad_effect_sources,
            })
    forbidden_recurring_save_sources = {
        str(source_id).strip().lower()
        for source_id in (
            expectations.get("forbid_effect_delta_recurring_save_source_ids") or []
        )
        if str(source_id).strip()
    }
    if forbidden_recurring_save_sources:
        bad_recurring_saves = [
            delta for delta in adjudication.get("effect_deltas") or []
            if str(delta.get("source_id") or "").strip().lower()
            in forbidden_recurring_save_sources
            and delta.get("recurring_save") is not None
        ]
        if bad_recurring_saves:
            findings.append({
                "name": "forbidden_effect_delta_recurring_save",
                "severity": "high",
                "detail": bad_recurring_saves,
            })
    if expectations.get("require_effect_delta_if_failed"):
        failed_targets = _failed_save_targets(result)
        effect_targets = {
            str(delta.get("target_id") or "")
            for delta in adjudication.get("effect_deltas") or []
            if delta.get("operation") == "start"
        }
        missing_effects = sorted(failed_targets - effect_targets)
        if missing_effects:
            findings.append({
                "name": "failed_save_missing_effect_delta",
                "severity": "high",
                "detail": missing_effects,
            })
    forbidden_conditions = expectations.get("forbid_effect_conditions") or []
    if isinstance(forbidden_conditions, str):
        forbidden_conditions = [forbidden_conditions]
    forbidden_condition_keys = {
        str(item).strip().lower() for item in forbidden_conditions
    }
    if forbidden_condition_keys:
        bad_effect_conditions = [
            delta for delta in adjudication.get("effect_deltas") or []
            if forbidden_condition_keys.intersection(
                str(condition).strip().lower()
                for condition in delta.get("conditions") or []
            )
        ]
        if bad_effect_conditions:
            findings.append({
                "name": "forbidden_effect_condition",
                "severity": "medium",
                "detail": bad_effect_conditions,
            })
    forbidden_fact_terms = expectations.get("forbid_fact_contains") or []
    if isinstance(forbidden_fact_terms, str):
        forbidden_fact_terms = [forbidden_fact_terms]
    for term in forbidden_fact_terms:
        text = str(term).strip().lower()
        if not text:
            continue
        matches = [
            fact for fact in _visible_facts(result)
            if text in fact.lower()
        ]
        if matches:
            findings.append({
                "name": "forbidden_visible_fact",
                "severity": "medium",
                "detail": {"term": term, "facts": matches},
            })
    forbidden_private_fact_terms = (
        expectations.get("private_fact_forbid_contains") or []
    )
    if isinstance(forbidden_private_fact_terms, str):
        forbidden_private_fact_terms = [forbidden_private_fact_terms]
    for term in forbidden_private_fact_terms:
        text = str(term).strip().lower()
        if not text:
            continue
        matches = [
            fact for fact in _private_outcome_facts(result)
            if text in str(fact.get("text") or "").lower()
        ]
        if matches:
            findings.append({
                "name": "forbidden_private_fact",
                "severity": "medium",
                "detail": {"term": term, "facts": matches},
            })
    if expectations.get("forbid_visible_damage_numbers"):
        damage_number_pattern = re.compile(
            r"\b\d+\s+(?:acid|bludgeoning|cold|fire|force|lightning|"
            r"necrotic|piercing|poison|psychic|radiant|slashing|thunder)"
            r"\s+damage\b|\b(?:takes?|deals?)\s+\d+\s+damage\b",
            re.IGNORECASE,
        )
        bad_facts = [
            fact for fact in _visible_facts(result)
            if damage_number_pattern.search(fact)
        ]
        if bad_facts:
            findings.append({
                "name": "visible_damage_number",
                "severity": "medium",
                "detail": bad_facts,
            })
    ambiguous_terms = []
    if expectations.get("forbid_ambiguous_visible_outcomes"):
        ambiguous_terms.extend(["stagger or", "or collapse", "falls or", "drops or"])
    ambiguous_terms.extend(expectations.get("forbid_ambiguous_terms") or [])
    bad_ambiguous = [
        {"term": term, "fact": fact}
        for term in ambiguous_terms
        for fact in _visible_facts(result)
        if str(term).lower() in fact.lower()
    ]
    if bad_ambiguous:
        findings.append({
            "name": "ambiguous_visible_outcome",
            "severity": "medium",
            "detail": bad_ambiguous,
        })
    required_spends = expectations.get("require_resource_spends") or []
    if isinstance(required_spends, dict):
        required_spends = [required_spends]
    missing_spends = [
        required for required in required_spends
        if not _has_required_resource_spend(result, required)
    ]
    if missing_spends:
        findings.append({
            "name": "missing_resource_spend",
            "severity": "high",
            "detail": missing_spends,
        })
    return findings


def _fact_asserts_condition(fact: str, condition: str) -> bool:
    lower = " ".join(str(fact or "").lower().split())
    condition = condition.strip().lower()
    if not condition or condition not in lower:
        return False
    negated = re.search(
        rf"\b(?:fail|fails|failed|cannot|can't|does not|doesn't|"
        rf"not|no|neither|nor|without|avoid|avoids|avoided)\b"
        rf"[^.?!;]{{0,80}}\b{re.escape(condition)}\b",
        lower,
    )
    return negated is None


def _has_opposed_rolls(requests: list[dict[str, Any]]) -> bool:
    contest_requests = [
        request for request in requests
        if request.get("kind") in {"ability_check", "skill_check"}
    ]
    if any(str(request.get("opposed_by") or "").strip() for request in contest_requests):
        return True
    actors = {
        str(request.get("actor_id") or "")
        for request in contest_requests
        if str(request.get("actor_id") or "")
    }
    return len(contest_requests) >= 2 and len(actors) >= 2


def _action_matches_required(
    action: dict[str, Any],
    required: dict[str, Any],
) -> bool:
    for key in (
        "actor_id",
        "source_type",
        "source_id",
        "effect_id",
        "use_mode",
        "economy",
    ):
        if key in required and action.get(key) != required[key]:
            return False
    return True


def _spatial_delta_matches_required(
    delta: dict[str, Any],
    required: dict[str, Any],
) -> bool:
    for key in (
        "kind",
        "target_id",
        "character_id",
        "x",
        "y",
        "label",
        "shape",
    ):
        if key in required and delta.get(key) != required[key]:
            return False
    return True


def _effect_delta_contains(delta: dict[str, Any], expected: str) -> bool:
    expected = expected.strip().lower()
    if not expected:
        return False
    haystack = " ".join(
        str(delta.get(key) or "").strip().lower()
        for key in ("effect_id", "name", "slug", "source_id", "reason")
    )
    return expected in haystack


def _effect_delta_matches_required(
    delta: dict[str, Any],
    required: dict[str, Any],
) -> bool:
    for key in ("operation", "target_id", "source_id", "effect_id", "slug"):
        if key in required and delta.get(key) != required[key]:
            return False
    if "concentration" in required:
        if bool(delta.get("concentration")) is not bool(required["concentration"]):
            return False
    if required.get("conditions_empty") and delta.get("conditions"):
        return False
    required_conditions = {
        str(condition).strip().lower()
        for condition in required.get("conditions_include") or []
        if str(condition).strip()
    }
    if required_conditions:
        actual_conditions = {
            str(condition).strip().lower()
            for condition in delta.get("conditions") or []
            if str(condition).strip()
        }
        if not required_conditions.issubset(actual_conditions):
            return False
    return True


def _has_required_resource_spend(
    result: dict[str, Any],
    required: dict[str, Any],
) -> bool:
    for spend in result.get("resource_spends") or []:
        if any(
            key in required and spend.get(key) != required[key]
            for key in ("actor_id", "resource_id", "source_id", "amount", "applied")
        ):
            continue
        return True
    return False


def _failed_save_targets(result: dict[str, Any]) -> set[str]:
    failed: set[str] = set()
    requests_by_roll_id = {
        str(request.get("roll_id") or ""): request
        for request in _flattened_rolls(result)
        if request.get("kind") == "saving_throw"
    }
    for line in ((result.get("capture") or {}).get("roll_ledger") or []):
        lower = str(line).lower()
        if "saving_throw" not in lower:
            continue
        for roll_id, request in requests_by_roll_id.items():
            if roll_id not in line:
                continue
            target_id = str(request.get("target_id") or "")
            if not target_id:
                continue
            if "fail" in lower or "failure" in lower:
                failed.add(target_id)
                continue
            match = re.search(r"=\s*(\d+),\s*DC\s*(\d+)", line)
            if match:
                total = int(match.group(1))
                dc = int(match.group(2))
            else:
                total_match = re.search(r"total\s+(\d+)", lower)
                total = int(total_match.group(1)) if total_match else 0
                dc = int(request.get("dc", 0) or 0)
            if dc and total < dc:
                failed.add(target_id)
    return failed


def _damage_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    # Damage records are not persisted in the capture today. Detect structured
    # attack damage via ledger markers, which is enough for stress reporting.
    records: list[dict[str, Any]] = []
    for line in ((result.get("capture") or {}).get("roll_ledger") or []):
        text = str(line)
        if text.startswith("damage_for="):
            records.append({"ledger": text})
    return records


def _applied_damage_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record for record in _damage_records(result)
        if "no code-readable damage expression" not in record["ledger"].lower()
    ]


def _unapplied_damage_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record for record in _damage_records(result)
        if "no code-readable damage expression" in record["ledger"].lower()
    ]


def _dict_contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value:
            return True
        return any(_dict_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_dict_contains_key(item, key) for item in value)
    return False


def _private_outcome_facts(result: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for raw in _adjudication(result).get("private_outcome_facts") or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        visible_to = [
            str(value).strip()
            for value in raw.get("visible_to") or []
            if str(value).strip()
        ]
        if text and visible_to:
            facts.append({"text": text, "visible_to": visible_to})
    return facts


def _private_fact_matches_terms(
    fact: dict[str, Any],
    terms: list[str],
) -> bool:
    if not terms:
        return True
    text = str(fact.get("text") or "").lower()
    return any(term in text for term in terms)


def _quality_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        for finding in result.get("findings") or []:
            findings.append({
                "scenario": result.get("name"),
                **finding,
            })
    return findings


def _cache_watch_findings(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for call in calls:
        watch = call.get("cache_watch") or {}
        if not watch.get("finalize_below_plan_cache_read"):
            continue
        findings.append({
            "call_index": call.get("raw_call_index"),
            "scenario": ((call.get("scenario") or {}).get("name") or ""),
            "phase": call.get("phase"),
            "plan_cache_read_input_tokens": (
                watch.get("scenario_plan_cache_read_input_tokens")
            ),
            "finalize_cache_read_input_tokens": (
                watch.get("cache_read_input_tokens")
            ),
            "delta": watch.get("finalize_cache_read_delta_from_plan"),
        })
    return findings


_SALIENCE_ORDER = {
    "major": 0,
    "notable": 1,
    "minor": 2,
}


def _router_observed_facts_by_salience(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for result in results:
        adjudication = _adjudication(result)
        for fact in adjudication.get("router_observed_facts") or []:
            if not str(fact.get("fact") or "").strip():
                continue
            facts.append({
                "scenario": result.get("name"),
                "salience": str(fact.get("salience") or "notable").lower(),
                "fact": str(fact.get("fact") or "").strip(),
                "reason": str(fact.get("reason") or "").strip(),
            })
    return sorted(
        facts,
        key=lambda fact: (
            _SALIENCE_ORDER.get(str(fact.get("salience") or ""), 99),
            str(fact.get("scenario") or ""),
            str(fact.get("fact") or ""),
        ),
    )


def _checks(results: list[dict[str, Any]], error: str) -> list[dict[str, Any]]:
    calls = [
        call
        for result in results
        for call in result.get("role_calls") or []
        if call.get("role") == "dnd_combat_manager"
    ]
    cache_check_required = len(results) > 1
    cache_watch_findings = _cache_watch_findings(calls)
    return [
        _check("no_harness_error", not error, error),
        _check(
            "scenarios_completed",
            all(not result.get("error") for result in results),
            [
                {"name": result.get("name"), "error": result.get("error")}
                for result in results
                if result.get("error")
            ],
        ),
        _check(
            "dnd_combat_manager_only",
            all(call.get("role") == "dnd_combat_manager" for call in calls),
            calls,
        ),
        _check(
            "progressive_cache_reads_observed",
            not cache_check_required or any(
                int((call.get("usage") or {}).get("cache_read_input_tokens", 0) or 0)
                > 0
                for call in calls[1:]
            ),
            (
                "not required for a single-scenario run"
                if not cache_check_required else [
                    (call.get("usage") or {}).get("cache_read_input_tokens", 0)
                    for call in calls
                ]
            ),
        ),
        _check(
            "finalize_cache_reads_not_below_plan",
            not cache_watch_findings,
            cache_watch_findings,
        ),
    ]


def _check(name: str, passed: bool, detail: Any = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _usage_totals(calls: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "prompt_tokens",
        "completion_tokens",
        "visible_completion_tokens",
        "reasoning_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "full_input_tokens",
        "total_tokens",
    )
    totals = {key: 0 for key in keys}
    for call in calls:
        usage = call.get("usage") or {}
        for key in keys:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


def _preflight_api_keys(config: LLMConfig, roles: set[str]) -> list[str]:
    missing: list[str] = []
    for role in sorted(roles):
        provider = config.provider_for_role(role)
        if not config.api_key_for_provider(provider, role=role):
            missing.append(f"{role} ({provider})")
    return missing


def _role_label(config: LLMConfig, role: str) -> str:
    return f"{config.provider_for_role(role)}:{config.model_for_role(role)}"
