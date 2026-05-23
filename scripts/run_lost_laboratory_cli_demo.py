#!/usr/bin/env python3
"""Live CLI demo for the Lost Laboratory imported-content startup slice.

The script builds a temporary story from the reviewed private import artifacts,
drives the same CLIState command handlers used by scripts/play.py, and writes a
succinct report under app/storage/playtest_reports. It uses real LLM calls.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import asyncio
import io
import json
import logging
import os
import shutil
import sqlite3
import sys
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.bot.engine_bridge import EngineBridge
from app.llm.config import LIVE_PLAY_REQUIRED_ROLES, LLMConfig
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterStatus,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content import (
    ContentFrontState,
    ContentKnowledgeEntityState,
    ContentPackState,
    PendingContentSignal,
)
from app.engine.content_pack_compiler import (
    SCHEMA_VERSION as CONTENT_PACK_SCHEMA_VERSION,
)
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.state import (
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)
from scripts.play import CLIState


PACK_ARTIFACT_DIR = REPO_ROOT / "private_extractions/lost_laboratory_of_kwalish"
CATALOG_PATH = PACK_ARTIFACT_DIR / "manual_semantic_catalog_startup_reviewed_v1.json"
SEED_INPUTS_PATH = PACK_ARTIFACT_DIR / "startup_checkpoint_seed_inputs_v1.json"
COMPILED_PACK_PATH = (
    REPO_ROOT
    / "private_extractions/compiled/lost_laboratory_kwalish_startup_reviewed_v1.sqlite"
)
REPORT_ROOT = REPO_ROOT / "app/storage/playtest_reports"
STORY_ID = "lost_laboratory_kwalish_cli_demo"
PLAYER_ID = "pc_expedition_leader"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = REPORT_ROOT / f"lost_laboratory_cli_demo_{TS}"
JSON_PATH = RUN_DIR / "report.json"
MD_PATH = RUN_DIR / "report.md"
LOG_PATH = RUN_DIR / "run.log"
ISSUES_FIXED = [
    (
        "Demo runner imported the content-pack schema constant from the wrong "
        "module.",
        "Changed the import to use app.engine.content_pack_compiler, which is "
        "where the compiler/runtime schema version is defined.",
    ),
    (
        "Content-manager live calls were truncated before JSON parsing with "
        "OpenAI incomplete max_output_tokens errors.",
        "Raised the content-manager completion cap to 4000 tokens and set the "
        "content-manager OpenAI reasoning default to low so lookup/ref "
        "selection does not consume router-grade hidden reasoning budget.",
    ),
]


@dataclass(frozen=True)
class RefCard:
    ref: str
    content_hash: str
    kind: str
    summary: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cards() -> dict[str, RefCard]:
    if not COMPILED_PACK_PATH.exists():
        raise FileNotFoundError(f"compiled pack not found: {COMPILED_PACK_PATH}")
    with sqlite3.connect(COMPILED_PACK_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT ref, content_hash, kind, summary
            FROM content_cards
            WHERE gate_status = 'runtime_ready'
            ORDER BY ref
            """
        ).fetchall()
    return {
        row["ref"]: RefCard(
            ref=row["ref"],
            content_hash=row["content_hash"],
            kind=row["kind"],
            summary=row["summary"],
        )
        for row in rows
    }


def _compact_ref(pack_id: str, ref: str, cards: dict[str, RefCard]) -> str:
    card = cards[ref]
    return f"{pack_id}:{ref}@{card.content_hash}"


def _refs_for_actor(
    actor: dict[str, Any],
    context_slice: dict[str, Any],
) -> list[str]:
    refs = [
        actor["ref"],
        actor.get("agent_context_slice_ref", ""),
        *actor.get("home_location_refs", []),
        *actor.get("front_refs", []),
        *actor.get("knowledge_channel_refs", []),
        *context_slice.get("local_context_refs", []),
        *context_slice.get("graph_edge_refs", []),
    ]
    return [ref for ref in dict.fromkeys(refs) if ref]


def _npc_character(
    *,
    actor: dict[str, Any],
    context_slice: dict[str, Any],
    pack_id: str,
    cards: dict[str, RefCard],
) -> CharacterRecord:
    character_id = actor.get("character_id_hint") or actor["ref"].replace(".", "_")
    display_name = {
        "npc_cartophile": "The Cartophile",
        "npc_ctenmiir": "Ctenmiir",
        "npc_garret": "Garret Levistusson",
        "npc_gearbox": "Gearbox",
        "npc_mary": "Mary Greymalkin",
    }.get(character_id, character_id.replace("_", " ").title())
    active = character_id in {"npc_cartophile", "npc_garret", "npc_gearbox"}
    known_refs = [
        _compact_ref(pack_id, ref, cards)
        for ref in _refs_for_actor(actor, context_slice)
        if ref in cards
    ]
    beliefs = "; ".join(context_slice.get("beliefs", []))
    uncertainties = "; ".join(context_slice.get("uncertainties", []))
    boundaries = "; ".join(context_slice.get("hard_boundaries", []))
    known_context = " ".join(
        part
        for part in (
            context_slice.get("known_context", ""),
            f"Beliefs: {beliefs}." if beliefs else "",
            f"Uncertainties: {uncertainties}." if uncertainties else "",
            f"Boundaries: {boundaries}." if boundaries else "",
            f"Reviewed content refs known to this agent: {', '.join(known_refs)}.",
        )
        if part
    )
    secrets = []
    private_text = context_slice.get("private_state", "")
    if private_text:
        secrets.append(private_text)
    constraints = actor.get("constraints", [])
    if constraints:
        secrets.append("Constraints: " + "; ".join(constraints))
    return CharacterRecord(
        character_id=character_id,
        name=display_name,
        status=CharacterStatus.active if active else CharacterStatus.dormant,
        location="loc.cartophile_collection",
        is_playable=False,
        agent_tier=CharacterAgentTier.standard,
        public_sheet=PublicSheet(
            role=actor.get("actor_kind", "npc").replace("_", " "),
            appearance="A reviewed startup NPC from the expedition launch slice.",
            faction="Lost Laboratory expedition",
        ),
        private_state=PrivateState(
            goals=actor.get("goals", []),
            current_objectives=context_slice.get("current_agenda", []),
            secrets=secrets,
            intentions_enabled=character_id in {"npc_cartophile", "npc_garret"},
            tick_cues=actor.get("initiative_triggers", []),
        ),
        backstory=actor.get("summary", ""),
        personality=(
            "Play the reviewed startup role concretely. Keep pressure social "
            "and contractual unless the router frames a contested escalation."
        ),
        known_context=known_context,
        mechanics={},
    )


def _player_character() -> CharacterRecord:
    return CharacterRecord(
        character_id=PLAYER_ID,
        name="Demo Expedition Leader",
        status=CharacterStatus.active,
        location="loc.cartophile_collection",
        is_playable=True,
        agent_tier=CharacterAgentTier.standard,
        public_sheet=PublicSheet(
            role="player character expedition lead",
            appearance="Travel-ready adventurer with notebook, packs, and a cautious eye.",
            faction="Lost Laboratory expedition",
        ),
        private_state=PrivateState(
            goals=["Assemble a viable expedition and keep the party alive."],
            current_objectives=[
                "Understand the patron's terms.",
                "Choose a trustworthy custodian or negotiate an alternative.",
            ],
            secrets=[],
            intentions_enabled=False,
        ),
        known_context=(
            "You are at the expedition sponsor's map collection. You know the "
            "expedition needs route material, terms, and a decision about who "
            "will safeguard the loaned documents."
        ),
        mechanics={
            "ruleset_id": "dnd5e_basic",
            "level": 5,
            "proficiency_bonus": 3,
            "ability_scores": {
                "str": 12,
                "dex": 14,
                "con": 14,
                "int": 13,
                "wis": 12,
                "cha": 10,
            },
            "armor_class": 15,
            "hit_points": {"current": 38, "max": 38, "temporary": 0},
            "conditions": [],
            "resources": {},
            "raw": {},
        },
    )


def _content_state(
    *,
    seed_inputs: dict[str, Any],
    catalog: dict[str, Any],
    cards: dict[str, RefCard],
) -> dict[str, ContentPackState]:
    pack_id = seed_inputs["pack_id"]
    pending: dict[str, PendingContentSignal] = {}
    for index, ref in enumerate(seed_inputs.get("initial_router_lookup_refs", []), start=1):
        if ref not in cards:
            continue
        card = cards[ref]
        pending[f"startup_{index:02d}"] = PendingContentSignal(
            signal_id=f"startup_{index:02d}",
            pack_id=pack_id,
            ref_id=ref,
            content_hash=card.content_hash,
            reason="startup router context",
            priority=10,
            requested_fields=["summary"],
            metadata={
                "kind": card.kind,
                "visibility": "router_hidden",
                "summary": card.summary,
            },
        )

    active_fronts: dict[str, ContentFrontState] = {}
    for front_ref in seed_inputs.get("active_front_refs", []):
        if front_ref in cards:
            active_fronts[front_ref] = ContentFrontState(
                front_id=front_ref,
                label=front_ref,
                status="active",
                notes=cards[front_ref].summary,
            )

    contexts = {
        item["actor_ref"]: item
        for item in catalog.get("agent_context_slices", [])
    }
    knowledge_map: dict[str, ContentKnowledgeEntityState] = {}
    for actor in catalog.get("actor_dossiers", []):
        context_slice = contexts.get(actor["ref"], {})
        character_id = actor.get("character_id_hint")
        if not character_id:
            continue
        known_refs = [
            _compact_ref(pack_id, ref, cards)
            for ref in _refs_for_actor(actor, context_slice)
            if ref in cards
        ]
        knowledge_map[character_id] = ContentKnowledgeEntityState(
            entity_id=character_id,
            known_refs=known_refs,
            notes="Reviewed startup knowledge map seed.",
        )

    return {
        pack_id: ContentPackState(
            pack_id=pack_id,
            pending_signals=pending,
            fronts=active_fronts,
            knowledge_map=knowledge_map,
            metadata={
                "db_path": str(COMPILED_PACK_PATH.relative_to(REPO_ROOT)),
                "pack_version": seed_inputs["pack_version"],
                "source_fingerprint": seed_inputs["source_fingerprint"],
                "schema_version": CONTENT_PACK_SCHEMA_VERSION,
                "active_front_refs": seed_inputs.get("active_front_refs", []),
                "catalog": [
                    {"ref": ref, "aliases": [ref.replace(".", " ")]}
                    for ref in seed_inputs.get("initial_router_lookup_refs", [])
                    if ref in cards
                ],
            },
        )
    }


def _story_checkpoint() -> CheckpointFile:
    catalog = _read_json(CATALOG_PATH)
    seed_inputs = _read_json(SEED_INPUTS_PATH)
    cards = _load_cards()
    context_by_actor = {
        item["actor_ref"]: item
        for item in catalog.get("agent_context_slices", [])
    }
    npcs = [
        _npc_character(
            actor=actor,
            context_slice=context_by_actor.get(actor["ref"], {}),
            pack_id=seed_inputs["pack_id"],
            cards=cards,
        )
        for actor in catalog.get("actor_dossiers", [])
    ]
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=STORY_ID,
            story_id=STORY_ID,
            config=SessionConfig(),
            content_state=_content_state(
                seed_inputs=seed_inputs,
                catalog=catalog,
                cards=cards,
            ),
        ),
        player_primer=(
            "You are opening a Lost Laboratory expedition from the patron's "
            "map collection. The immediate job is to secure terms, understand "
            "the loaned route material, and decide who will accompany or "
            "safeguard the expedition records."
        ),
        world_state=WorldState(
            facts=[
                "The expedition is still in its startup negotiation phase.",
                "The patron can provide route material but expects recovered documentation.",
                "Candidate custodians and specialists may be negotiated before travel.",
            ],
            physics_ruleset=PhysicsRuleset(
                strength_limits="dnd5e_basic",
                magic_enabled=True,
            ),
            setting=StorySetting(
                genre="D&D expedition adventure",
                era="fantasy with strange lost technology",
                tone="concrete, table-play practical, lightly ominous",
                premise=(
                    "A party prepares to follow incomplete route lore toward "
                    "a lost inventor's laboratory."
                ),
            ),
            lore=(
                "Use only reviewed runtime-ready Lost Laboratory startup "
                "content. The downstream monastery, city, final laboratory, "
                "tactical maps, statblocks, and appendices are not available "
                "in this demo slice."
            ),
        ),
        characters=[_player_character(), *npcs],
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    ckpt.session.config.settings.player_roll_mode = "auto"
    ckpt.session.config.narrative_rules = (
        "Keep the startup scene grounded in negotiation, route preparation, "
        "and choosing expedition support. Do not invent downstream locations "
        "or encounters beyond reviewed runtime-ready content."
    )
    return ckpt


def _write_story(stories_dir: Path) -> Path:
    story_dir = stories_dir / STORY_ID
    story_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _story_checkpoint()
    path = story_dir / "ckpt_0000.json"
    path.write_text(
        ckpt.model_dump_json(
            indent=2,
            context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
        ),
        encoding="utf-8",
    )
    CheckpointFile.model_validate_json(path.read_text(encoding="utf-8"))
    return path


async def _run_cli_line(state: CLIState, line: str) -> str:
    output = io.StringIO()
    with redirect_stdout(output):
        await state.handle_line(line)
    return output.getvalue()


def _message_text(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
    return "\n".join(chunks)


def _safe_excerpt(text: str, *, limit: int = 1200) -> str:
    cleaned = "\n".join(line.rstrip() for line in text.strip().splitlines())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "\n...[truncated]"


def _role_label(config: LLMConfig, role: str) -> str:
    return f"{config.provider_for_role(role)}:{config.model_for_role(role)}"


async def _run_demo(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    os.environ.setdefault("LLM_MODEL_AGENT", "claude-haiku-4-5")
    os.environ.setdefault("LLM_MODEL_CHARACTER_GEN", "claude-haiku-4-5")
    os.environ.setdefault("LLM_MODEL_AGENT_CONVENIENCE", "claude-haiku-4-5")

    config = LLMConfig.from_env()
    required = set(LIVE_PLAY_REQUIRED_ROLES) | {"content_manager"}
    missing = config.missing_credentials(required)
    if missing:
        formatted = ", ".join(
            f"{item.role} ({item.provider}; {', '.join(item.env_names)})"
            for item in missing
        )
        raise RuntimeError(f"Missing live credentials: {formatted}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    stories_dir = RUN_DIR / "stories"
    sessions_dir = RUN_DIR / "sessions"
    story_path = _write_story(stories_dir)
    if args.install_story:
        installed_dir = REPO_ROOT / "app/storage/stories" / STORY_ID
        if installed_dir.exists():
            shutil.rmtree(installed_dir)
        shutil.copytree(story_path.parent, installed_dir)

    engine = EngineBridge(
        stories_dir=str(stories_dir),
        sessions_dir=str(sessions_dir),
        prompts_dir=str(REPO_ROOT / "app/prompts"),
        llm_config=config,
    )
    role_calls: list[dict[str, Any]] = []
    real_complete = engine.client.complete

    async def _recording_complete(*call_args, **kwargs):
        role = str(kwargs.get("role") or (call_args[0] if call_args else ""))
        response_model = kwargs.get("response_model")
        messages = kwargs.get("messages") or []
        prompt_text = _message_text(messages)
        record: dict[str, Any] = {
            "role": role,
            "response_model": response_model.__name__ if response_model else "",
            "message_count": len(messages),
            "contains_engine_knowledge_map": "engine_knowledge_map" in prompt_text,
            "contains_knowledge_entity_rows": " entity=npc_" in prompt_text,
            "contains_runtime_content_card": any(
                marker in prompt_text
                for marker in (
                    "location_card ref=loc.cartophile_collection",
                    "front_signal ref=front.expedition_obligation",
                    "content_known ref=handout.cartophile_maps",
                )
            ),
            "contains_turn_hint": "turn_hint " in prompt_text,
            "contains_private_path": "private_extractions/" in prompt_text,
        }
        try:
            response = await real_complete(*call_args, **kwargs)
        except Exception:
            record["error"] = traceback.format_exc()
            role_calls.append(record)
            raise
        record["model"] = response.model
        role_calls.append(record)
        return response

    engine.client.complete = _recording_complete  # type: ignore[method-assign]

    session_id = args.session or f"{STORY_ID}_{TS.lower()}"
    transcript: list[dict[str, str]] = []
    try:
        engine.create_empty_session(session_id)
        state = CLIState(engine, session_id, "")
        lines = [
            f"/story start {STORY_ID}",
            "/characters",
            f"/join {PLAYER_ID}",
            "/begin",
            (
                "I ask the Cartophile what terms he requires before he "
                "releases the maps, and which custodian he recommends."
            ),
            (
                "I turn to Garret and ask him to make his case for joining "
                "as custodian, including what risks he thinks the route hides."
            ),
        ]
        for line in lines:
            transcript.append({
                "input": line,
                "output": await _run_cli_line(state, line),
            })
        ckpt = engine.load_latest(session_id)
        return _build_report(
            args=args,
            config=config,
            session_id=session_id,
            story_path=story_path,
            transcript=transcript,
            role_calls=role_calls,
            checkpoint=ckpt,
            error="",
        )
    except Exception:
        return _build_report(
            args=args,
            config=config,
            session_id=session_id,
            story_path=story_path,
            transcript=transcript,
            role_calls=role_calls,
            checkpoint=None,
            error=traceback.format_exc(),
        )
    finally:
        await engine.close()


def _build_report(
    *,
    args: argparse.Namespace,
    config: LLMConfig,
    session_id: str,
    story_path: Path,
    transcript: list[dict[str, str]],
    role_calls: list[dict[str, Any]],
    checkpoint: CheckpointFile | None,
    error: str,
) -> dict[str, Any]:
    introduced_refs: list[str] = []
    knowledge_entities: list[str] = []
    canonical_fact_count = 0
    turn_index = 0
    if checkpoint is not None:
        turn_index = checkpoint.session.turn_index
        canonical_fact_count = sum(
            len(event.canonical_event.observable_facts)
            for event in checkpoint.canonical_events
        )
        for pack_state in checkpoint.session.content_state.values():
            introduced_refs.extend(sorted(pack_state.introduced_refs))
            knowledge_entities.extend(sorted(pack_state.knowledge_map))

    cli_errors = [
        item
        for item in transcript
        if "error:" in item["output"].lower()
    ]
    router_calls = [call for call in role_calls if call["role"] == "event_router"]
    content_calls = [
        call for call in role_calls if call["role"] == "content_manager"
    ]
    agent_calls = [
        call
        for call in role_calls
        if call["role"] in {"agent", "agent_standard", "agent_convenience"}
    ]
    checks = [
        _check("story_seed_validated", story_path.exists(), str(story_path)),
        _check("cli_completed_without_error_output", not cli_errors, cli_errors),
        _check("content_manager_called", bool(content_calls), content_calls),
        _check(
            "router_never_received_full_knowledge_map",
            all(
                not call["contains_engine_knowledge_map"]
                and not call["contains_knowledge_entity_rows"]
                for call in router_calls
            ),
            router_calls,
        ),
        _check(
            "router_received_projected_runtime_content",
            any(call["contains_runtime_content_card"] for call in router_calls),
            router_calls,
        ),
        _check(
            "character_agents_used_haiku_role",
            bool(agent_calls)
            and all(call["role"] != "agent" for call in agent_calls)
            and all("haiku" in str(call.get("model", "")).lower() for call in agent_calls),
            agent_calls,
        ),
        _check(
            "no_private_pack_paths_in_model_prompts",
            all(not call["contains_private_path"] for call in role_calls),
            role_calls,
        ),
        _check("canonical_events_created", canonical_fact_count > 0, canonical_fact_count),
        _check(
            "startup_content_introduced",
            any("loc.cartophile_collection" in ref for ref in introduced_refs)
            and any("front.expedition_obligation" in ref for ref in introduced_refs),
            introduced_refs,
        ),
        _check(
            "knowledge_map_seed_present",
            "npc_cartophile" in knowledge_entities,
            knowledge_entities,
        ),
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "session_id": session_id,
        "story_id": STORY_ID,
        "story_path": str(story_path.relative_to(REPO_ROOT)),
        "installed_story": bool(args.install_story),
        "roles": {
            role: _role_label(config, role)
            for role in (
                "content_manager",
                "event_router",
                "narrator",
                "agent",
                "agent_standard",
                "agent_convenience",
            )
        },
        "turn_index": turn_index,
        "canonical_fact_count": canonical_fact_count,
        "introduced_refs": introduced_refs,
        "knowledge_entities": knowledge_entities,
        "transcript": transcript,
        "role_calls": role_calls,
        "issues_fixed": [
            {"issue": issue, "fix": fix}
            for issue, fix in ISSUES_FIXED
        ],
        "checks": checks,
        "error": error,
    }


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _markdown(report: dict[str, Any]) -> str:
    passed = sum(1 for check in report["checks"] if check["passed"])
    total = len(report["checks"])
    lines = [
        "# Lost Laboratory CLI Demo Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Run directory: `{report['run_dir']}`",
        f"Story: `{report['story_id']}`",
        f"Session: `{report['session_id']}`",
        f"Checks: `{passed}/{total}`",
        "",
        "## Result",
        "",
    ]
    if report.get("error"):
        lines.extend(["The run ended with an exception:", "", "```text"])
        lines.append(report["error"].strip())
        lines.extend(["```", ""])
    else:
        lines.append("The CLI demo completed and produced canonical events.")
        lines.append("")

    lines.extend(["## Checks", ""])
    for check in report["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {mark}: `{check['name']}`")
    lines.append("")

    lines.extend(["## Issues Hit And Fixed", ""])
    for item in report.get("issues_fixed", []):
        lines.append(f"- Issue: {item['issue']}")
        lines.append(f"  Fix: {item['fix']}")
    lines.append("")

    lines.extend(["## Role Calls", ""])
    for index, call in enumerate(report["role_calls"], start=1):
        lines.append(
            "- "
            f"{index}. role=`{call['role']}` model=`{call.get('model', '-')}` "
            f"schema=`{call['response_model']}` "
            f"knowledge_map={call['contains_engine_knowledge_map']} "
            f"runtime_content={call['contains_runtime_content_card']} "
            f"turn_hint={call['contains_turn_hint']}"
        )
    lines.append("")

    lines.extend(["## CLI Transcript", ""])
    for item in report["transcript"]:
        lines.extend(["```text", f"> {item['input']}"])
        lines.append(_safe_excerpt(item["output"].rstrip(), limit=1800))
        lines.extend(["```", ""])

    lines.extend(["## Content State", ""])
    lines.append(f"Introduced refs: `{len(report['introduced_refs'])}`")
    for ref in report["introduced_refs"]:
        lines.append(f"- `{ref}`")
    lines.append("")
    lines.append("Knowledge-map entities:")
    for entity in report["knowledge_entities"]:
        lines.append(f"- `{entity}`")
    lines.append("")

    lines.extend(["## Notes", ""])
    lines.append(
        "- Character-agent calls are expected to use `agent_standard` with a "
        "Haiku model. The script also sets `LLM_MODEL_AGENT` and "
        "`LLM_MODEL_AGENT_CONVENIENCE` to Haiku unless the environment already "
        "overrides them."
    )
    lines.append(
        "- The reviewed startup slice intentionally excludes downstream sites, "
        "combat statblocks, tactical maps, and player-safe image derivatives."
    )
    lines.append(
        "- The report intentionally records only compact runtime refs and CLI "
        "output, not raw OCR, source PDF text, or private source paths."
    )
    return "\n".join(lines)


async def main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        default="",
        help="Session id for the demo run. Defaults to a timestamped id.",
    )
    parser.add_argument(
        "--install-story",
        action="store_true",
        help=(
            "Also copy the generated story seed into app/storage/stories for "
            "manual scripts/play.py replay."
        ),
    )
    args = parser.parse_args()

    report = await _run_demo(args)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(_markdown(report), encoding="utf-8")
    print(JSON_PATH.relative_to(REPO_ROOT))
    print(MD_PATH.relative_to(REPO_ROOT))
    failed = bool(report.get("error")) or any(
        not check["passed"] for check in report["checks"]
    )
    for check in report["checks"]:
        print(("PASS" if check["passed"] else "FAIL") + f" {check['name']}")
    return 1 if failed else 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
