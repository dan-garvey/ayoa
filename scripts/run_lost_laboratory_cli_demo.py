#!/usr/bin/env python3
"""Live CLI demo for the Lost Laboratory reviewed imported-content pack.

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
import re
import shutil
import sys
import traceback
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine import dnd_combat
from app.engine.content_pack_projections import (
    apply_checkpoint_projection,
    apply_field_start_projection,
    character_record_from_projection,
    content_pack_state_from_projection,
)
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
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.content_projection import (
    ContentCharacterProjection,
    ContentPackProjectionArtifact,
)
from app.schemas.state import (
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)
from scripts.play import CLIState


PACK_ARTIFACT_DIR = REPO_ROOT / "private_extractions/lost_laboratory_of_kwalish"
PROJECTION_PATH = PACK_ARTIFACT_DIR / "semantic_projections_full_reviewed_v1.json"
COMPILED_PACK_PATH = (
    REPO_ROOT
    / "private_extractions/compiled/lost_laboratory_kwalish_full_reviewed_v1.sqlite"
)
REPORT_ROOT = REPO_ROOT / "app/storage/playtest_reports"
STORY_ID = "lost_laboratory_kwalish_cli_demo"
PLAYER_ID = "pc_expedition_leader"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = REPORT_ROOT / f"lost_laboratory_cli_demo_{TS}"
JSON_PATH = RUN_DIR / "report.json"
MD_PATH = RUN_DIR / "report.md"
LOG_PATH = RUN_DIR / "run.log"
SCENARIO_COMMANDS = {
    "cgf": [
        f"/story start {STORY_ID}",
        "/characters",
        f"/join {PLAYER_ID}",
        "/begin",
        (
            "At the wooded foothills, I compare the route folios against the "
            "ridge line and ask Garret whether the maps imply any immediate "
            "route pressure before we commit to the cave approach."
        ),
        (
            "I mark the safest line forward, ask Gearbox what equipment needs "
            "checking against the terrain, and keep the group short of any "
            "ambush point for now."
        ),
    ],
    "startup": [
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
    ],
    "deep": [
        f"/story start {STORY_ID}",
        "/characters",
        f"/join {PLAYER_ID}",
        "/begin",
        (
            "I ask the Cartophile what terms he requires before he "
            "releases the maps, and which custodian he recommends."
        ),
        (
            "I accept the return-documentation terms, appoint Garret as "
            "custodian of the maps, and ask Gearbox to accompany us as a "
            "technical specialist. Then I ask the Cartophile to release "
            "the route materials for the wooded foothills and cave approach."
        ),
        (
            "I confirm the return-documentation terms are fair. We leave at "
            "first light and follow the maps toward the Barrier Peaks route, "
            "comparing ridgelines and watching for the cave approach."
        ),
        (
            "At the wooded foothills, I scout ahead carefully with the map "
            "open, checking loose rock, tracks, and sightlines before we move "
            "toward the cave approach."
        ),
        (
            "The route pressure turns hostile as armed figures move from the "
            "rocks toward the route folios. I warn Garret and Gearbox behind "
            "me, draw my longsword, and hold the line instead of yielding the "
            "documents."
        ),
        "If the outlaws are still fighting, I attack the nearest threat with my longsword.",
    ],
    "field": [
        f"/story start {STORY_ID}",
        "/characters",
        f"/join {PLAYER_ID}",
        "/begin",
        (
            "At the wooded foothills, I scout ahead carefully with the map "
            "open, checking loose rock, tracks, and sightlines before we move "
            "toward the cave approach."
        ),
        (
            "The route pressure turns hostile as armed figures move from the "
            "rocks toward the route folios. I warn Garret and Gearbox behind "
            "me, draw my longsword, and hold the line instead of yielding the "
            "documents."
        ),
        "If the outlaws are still fighting, I attack the nearest threat with my longsword.",
    ],
    "soak": [
        f"/story start {STORY_ID}",
        "/characters",
        f"/join {PLAYER_ID}",
        "/begin",
        (
            "At the wooded foothills, I scout ahead carefully with the map "
            "open, checking loose rock, tracks, and sightlines before we move "
            "toward the cave approach."
        ),
        (
            "The route pressure turns hostile as armed figures move from the "
            "rocks toward the route folios. I warn Garret and Gearbox behind "
            "me, draw my longsword, and hold the line instead of yielding the "
            "documents."
        ),
        "If the outlaws are still fighting, I attack the nearest threat with my longsword.",
    ],
}
FIELD_START_SCENARIOS = {"cgf", "field", "soak"}
ROUTE_EXPLORATION_SCENARIOS = {"cgf", "deep", "field", "soak"}
COMBAT_SCENARIOS = {"deep", "field", "soak"}
SOAK_COMBAT_COMMANDS = (
    (
        "If any outlaw is still standing, I press the attack with my "
        "longsword; otherwise I check Garret and Gearbox for injuries and "
        "secure the route folios."
    ),
    (
        "I keep myself between the outlaws and the expedition, striking one "
        "with my longsword if it is still a threat."
    ),
    (
        "I look for an opening, call the target clearly for Garret and "
        "Gearbox, and attack the outlaws again."
    ),
    (
        "I hold the line at the rocky approach and make another measured "
        "longsword attack if the beast remains in the fight."
    ),
)
SOAK_EXPLORATION_COMMANDS = (
    (
        "I take a minute to check Garret and Gearbox for injuries, then "
        "recover and sort the route folios."
    ),
    (
        "I search the cleft and nearby stones for tracks, scat, claw marks, "
        "or signs that the outlaws had a camp nearby."
    ),
    (
        "I compare the foothill ridges against the Cartophile's notes and "
        "mark the safest path forward."
    ),
    (
        "I ask Garret what dangers on the route match this ambush, and ask "
        "Gearbox what equipment we should recheck before pressing on."
    ),
    (
        "I advance to the best nearby overlook and study the cave approach "
        "before committing the group."
    ),
    (
        "I organize our marching order, with Gearbox protected in the middle "
        "and Garret watching the rear."
    ),
    (
        "I inspect the outlaws' approach path to decide whether this was a "
        "random ambush or a planned attempt to seize the maps."
    ),
    (
        "I listen for water, machinery, wind movement, or voices from deeper "
        "in the rocks."
    ),
    (
        "I pause to update our route notes for the Cartophile's required "
        "return report."
    ),
    (
        "We continue cautiously toward the cave approach, stopping whenever "
        "the map details no longer match the terrain."
    ),
)

def _weapon_action(
    action_id: str,
    name: str,
    *,
    attack_bonus: int,
    damage: str,
    damage_type: str,
    notes: str = "",
    attack_range: str = "5 ft",
) -> dict[str, Any]:
    return {
        "id": action_id,
        "name": name,
        "kind": "attack",
        "attack": {
            "bonus": attack_bonus,
            "damage": f"{damage} {damage_type}",
            "range": attack_range,
        },
        "damage": [{"formula": damage, "damage_type": damage_type}],
        "notes": notes,
    }


def _dnd_mechanics(
    *,
    level: int,
    proficiency_bonus: int,
    ability_scores: dict[str, int],
    armor_class: int,
    hp: int,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "ruleset_id": "dnd5e_basic",
        "level": level,
        "proficiency_bonus": proficiency_bonus,
        "ability_scores": {
            "str": ability_scores.get("str", 10),
            "dex": ability_scores.get("dex", 10),
            "con": ability_scores.get("con", 10),
            "int": ability_scores.get("int", 10),
            "wis": ability_scores.get("wis", 10),
            "cha": ability_scores.get("cha", 10),
        },
        "skill_proficiencies": [],
        "saving_throw_proficiencies": [],
        "armor_class": armor_class,
        "hit_points": {"current": hp, "max": hp, "temporary": 0},
        "conditions": [],
        "resources": {},
        "dnd5e_sheet": {
            "statblock": {
                "actions": actions,
                "spellcasting": {},
                "defenses": {},
            },
        },
        "raw": {},
    }


def _npc_demo_mechanics(character_id: str) -> dict[str, Any]:
    if character_id == "npc_garret":
        return _dnd_mechanics(
            level=3,
            proficiency_bonus=2,
            ability_scores={
                "str": 12,
                "dex": 14,
                "con": 12,
                "int": 11,
                "wis": 14,
                "cha": 10,
            },
            armor_class=13,
            hp=20,
            actions=[
                _weapon_action(
                    "short_blade",
                    "Short Blade",
                    attack_bonus=4,
                    damage="1d6+2",
                    damage_type="piercing",
                    notes=(
                        "Demo combat scaffold for the reviewed module pack; "
                        "represents Garret's practical field knife."
                    ),
                ),
            ],
        )
    if character_id == "npc_gearbox":
        return _dnd_mechanics(
            level=3,
            proficiency_bonus=2,
            ability_scores={
                "str": 13,
                "dex": 12,
                "con": 14,
                "int": 15,
                "wis": 10,
                "cha": 9,
            },
            armor_class=12,
            hp=22,
            actions=[
                _weapon_action(
                    "weighted_wrench",
                    "Weighted Wrench",
                    attack_bonus=3,
                    damage="1d6+1",
                    damage_type="bludgeoning",
                    notes=(
                        "Demo combat scaffold for the reviewed module pack; "
                        "represents Gearbox's tool used defensively."
                    ),
                ),
            ],
        )
    return {}


def _read_projection() -> ContentPackProjectionArtifact:
    if not PROJECTION_PATH.exists():
        raise FileNotFoundError(
            "Lost Laboratory projection artifact is missing. Re-run the reviewed "
            f"import promoter to create {PROJECTION_PATH}."
        )
    return ContentPackProjectionArtifact.model_validate_json(
        PROJECTION_PATH.read_text(encoding="utf-8")
    )


def _npc_character(projection: ContentCharacterProjection) -> CharacterRecord:
    return character_record_from_projection(
        projection,
        mechanics=_npc_demo_mechanics(projection.character_id),
        agent_tier=CharacterAgentTier.standard,
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
        mechanics=_dnd_mechanics(
            level=5,
            proficiency_bonus=3,
            ability_scores={
                "str": 14,
                "dex": 14,
                "con": 14,
                "int": 13,
                "wis": 12,
                "cha": 10,
            },
            armor_class=15,
            hp=38,
            actions=[
                _weapon_action(
                    "longsword",
                    "Longsword",
                    attack_bonus=5,
                    damage="1d8+2",
                    damage_type="slashing",
                    notes=(
                        "Demo combat scaffold for the reviewed module pack; "
                        "the expedition leader is visibly armed with this "
                        "weapon when they hold the line in the field."
                    ),
                ),
            ],
        ),
    )


def _story_checkpoint(*, start_mode: str = "startup") -> CheckpointFile:
    projection = _read_projection()
    npcs = [_npc_character(character) for character in projection.characters]
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=STORY_ID,
            story_id=STORY_ID,
            config=SessionConfig(),
            content_state=content_pack_state_from_projection(
                projection,
                db_path=str(COMPILED_PACK_PATH.relative_to(REPO_ROOT)),
                start_mode=start_mode,
            ),
        ),
        player_primer=projection.checkpoint.player_primer,
        world_state=WorldState(
            facts=list(projection.checkpoint.world_facts),
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
            lore=projection.checkpoint.world_lore,
        ),
        characters=[_player_character(), *npcs],
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    ckpt.session.config.settings.player_roll_mode = "auto"
    apply_checkpoint_projection(ckpt, projection.checkpoint)
    if start_mode == "field":
        apply_field_start_projection(ckpt, projection)
    return ckpt


def _write_story(stories_dir: Path, *, start_mode: str = "startup") -> Path:
    story_dir = stories_dir / STORY_ID
    story_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _story_checkpoint(start_mode=start_mode)
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


def _tag_block(text: str, tag: str) -> str:
    start_marker = f"<{tag}"
    end_marker = f"</{tag}>"
    start = text.find(start_marker)
    if start < 0:
        return ""
    start = text.find(">", start)
    if start < 0:
        return ""
    end = text.find(end_marker, start)
    if end < 0:
        return ""
    return text[start + 1:end].strip()


def _tag_lines(text: str, tag: str, *, limit: int = 24) -> list[str]:
    block = _tag_block(text, tag)
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    return lines[:limit]


def _all_tag_lines(text: str, tag: str, *, limit: int = 500) -> list[str]:
    return _tag_lines(text, tag, limit=limit)


def _nonempty_rows(lines: list[str]) -> list[str]:
    return [line for line in lines if line and line != "-"]


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def _hash_like_count(text: str) -> int:
    return len(re.findall(r"@[A-Fa-f0-9]{16,}\b|\b[A-Fa-f0-9]{64}\b", text))


def _assignment_value(line: str, key: str) -> str:
    marker = f"{key}="
    for part in line.split():
        if part.startswith(marker):
            return part.removeprefix(marker).strip()
    return ""


def _content_dispatch_keys(
    text: str,
    *,
    limit: int = 500,
) -> list[dict[str, str]]:
    keys: list[dict[str, str]] = []
    for line in _all_tag_lines(text, "router_knowledge_dispatch_index", limit=limit):
        item: dict[str, str] = {}
        for key in ("key", "kind", "pack", "priority", "packets"):
            value = _assignment_value(line, key)
            if value:
                item[key] = value
        if item:
            keys.append(item)
    return keys


def _content_manager_prompt_audit(prompt_text: str) -> dict[str, Any]:
    recent_facts = _nonempty_rows(_all_tag_lines(prompt_text, "recent_facts"))
    knowledge_state_rows = _nonempty_rows(
        _all_tag_lines(prompt_text, "router_knowledge_state")
    )
    candidates = _nonempty_rows(
        _all_tag_lines(prompt_text, "candidate_characters")
    )
    dispatch_keys = _content_dispatch_keys(prompt_text)
    kind_counts = Counter(
        item.get("kind", "-") for item in dispatch_keys if item.get("kind")
    )
    return {
        "prompt_chars": len(prompt_text),
        "approx_prompt_tokens": _approx_tokens(prompt_text),
        "hash_like_token_count": _hash_like_count(prompt_text),
        "recent_fact_count": len(recent_facts),
        "router_knowledge_state_rows": len(knowledge_state_rows),
        "candidate_character_count": len(candidates),
        "candidate_character_ids": [
            _assignment_value(line, "character") for line in candidates
        ][:12],
        "dispatch_key_count": len(dispatch_keys),
        "dispatch_key_kind_counts": dict(sorted(kind_counts.items())),
        "dispatch_key_sample": dispatch_keys[:12],
    }


def _content_manager_output_audit(parsed: Any) -> dict[str, Any]:
    if not hasattr(parsed, "model_dump"):
        return {}
    data = parsed.model_dump(mode="json")
    return {
        "knowledge_updates": data.get("knowledge_updates", []),
        "router_required_keys": data.get("router_required_keys", []),
        "router_turn_candidates": data.get("router_turn_candidates", []),
        "agent_context_broadcasts": data.get("agent_context_broadcasts", []),
        "no_update_reason": data.get("no_update_reason", ""),
    }


def _router_output_audit(parsed: Any) -> dict[str, Any]:
    if not hasattr(parsed, "model_dump"):
        return {}
    observers = []
    for observer in getattr(parsed, "observers", []) or []:
        observers.append({
            "character_id": getattr(observer, "character_id", ""),
            "routing_role": getattr(observer, "routing_role", ""),
            "observation_level": getattr(observer, "observation_level", ""),
        })
    return {
        "event_kind": getattr(parsed, "event_kind", ""),
        "interaction_mode": getattr(parsed, "interaction_mode", ""),
        "combatant_ids": list(getattr(parsed, "combatant_ids", []) or []),
        "combatant_spawns": [
            {
                "character_id": getattr(spawn, "character_id", ""),
                "monster_key": getattr(spawn, "monster_key", ""),
                "statblock_ref": getattr(spawn, "statblock_ref", ""),
                "name": getattr(spawn, "name", ""),
            }
            for spawn in getattr(parsed, "combatant_spawns", []) or []
        ],
        "observers": observers,
        "decision_rationale": getattr(parsed, "decision_rationale", ""),
    }


def _combat_state_summary(checkpoint: CheckpointFile | None) -> dict[str, Any]:
    if checkpoint is None:
        return {"active": False}
    combat = getattr(checkpoint.session, "active_combat", None)
    if combat is None:
        return {"active": False}
    public = dnd_combat.public_status(checkpoint.session)
    return {
        "active": True,
        "status": getattr(combat, "status", ""),
        "round_number": getattr(combat, "round_number", 0),
        "turn_index": getattr(combat, "turn_index", 0),
        "public_status": public,
        "combatants": [
            {
                "character_id": getattr(combatant, "character_id", ""),
                "name": getattr(combatant, "name", ""),
                "hp": {
                    "current": getattr(combatant, "hit_points_current", 0),
                    "max": getattr(combatant, "hit_points_max", 0),
                    "temporary": getattr(combatant, "hit_points_temporary", 0),
                },
                "initiative_total": getattr(combatant, "initiative_total", 0),
                "defeat_state": getattr(combatant, "defeat_state", ""),
                "pending_initiating_action": getattr(
                    combatant,
                    "pending_initiating_action",
                    "",
                ),
            }
            for combatant in getattr(combat, "combatants", []) or []
        ],
        "audit_tail": list(getattr(combat, "audit_lines", []) or [])[-12:],
    }


def _log_warning_lines() -> list[str]:
    if not LOG_PATH.exists():
        return []
    warnings: list[str] = []
    for line in LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if "WARNING" in line or "ERROR" in line:
            warnings.append(line)
    return warnings[-40:]


def _imported_encounter_applied(checkpoint: CheckpointFile | None) -> bool:
    combat = (
        getattr(checkpoint.session, "active_combat", None)
        if checkpoint is not None else None
    )
    if combat is None:
        return False
    return any(
        "Imported encounter template applied:" in str(line)
        for line in getattr(combat, "audit_lines", []) or []
    )


def _ad_hoc_spawn_ids_in_imported_combat(
    checkpoint: CheckpointFile | None,
) -> list[str]:
    if checkpoint is None or not _imported_encounter_applied(checkpoint):
        return []
    combat = getattr(checkpoint.session, "active_combat", None)
    if combat is None:
        return []
    characters = {
        getattr(character, "character_id", ""): character
        for character in getattr(checkpoint, "characters", []) or []
    }
    ids: list[str] = []
    for combatant in getattr(combat, "combatants", []) or []:
        character_id = str(getattr(combatant, "character_id", "") or "")
        character = characters.get(character_id)
        mechanics = getattr(character, "mechanics", {}) if character else {}
        if not isinstance(mechanics, dict):
            continue
        combat_spawn = mechanics.get("combat_spawn")
        source = str(mechanics.get("source") or "")
        if isinstance(combat_spawn, dict) and combat_spawn.get("spawned"):
            if source != "imported_statblock_catalog":
                ids.append(character_id)
                continue
        if character_id == "npc_bandit_captain":
            ids.append(character_id)
    return ids


def _usage_totals(role_calls: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for call in role_calls:
        role = str(call.get("role") or "")
        usage = call.get("usage") or {}
        if not role or not isinstance(usage, dict):
            continue
        target = totals.setdefault(role, {})
        for key, value in usage.items():
            if isinstance(value, int):
                target[key] = target.get(key, 0) + value
    return totals


def _content_manager_prompt_budgets(
    role_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    budgets: list[dict[str, Any]] = []
    for call in role_calls:
        if call.get("role") != "content_manager":
            continue
        audit = call.get("content_manager_input") or {}
        if not isinstance(audit, dict):
            continue
        budgets.append({
            "prompt_chars": audit.get("prompt_chars", 0),
            "approx_prompt_tokens": audit.get("approx_prompt_tokens", 0),
            "hash_like_token_count": audit.get("hash_like_token_count", 0),
            "router_knowledge_state_rows": audit.get("router_knowledge_state_rows", 0),
            "candidate_characters": audit.get("candidate_character_count", 0),
            "dispatch_keys": audit.get("dispatch_key_count", 0),
            "usage": call.get("usage") or {},
        })
    return budgets


def _checkpoint_turn_index(checkpoint: CheckpointFile | None) -> int:
    if checkpoint is None:
        return 0
    return int(getattr(checkpoint.session, "turn_index", 0) or 0)


def _next_soak_command(checkpoint: CheckpointFile | None, index: int) -> str:
    combat = (
        getattr(checkpoint.session, "active_combat", None)
        if checkpoint is not None else None
    )
    if combat is not None:
        return SOAK_COMBAT_COMMANDS[index % len(SOAK_COMBAT_COMMANDS)]
    return SOAK_EXPLORATION_COMMANDS[index % len(SOAK_EXPLORATION_COMMANDS)]


def _soak_defeated_spawned_hostiles_active(
    checkpoint: CheckpointFile | None,
) -> bool:
    combat = (
        getattr(checkpoint.session, "active_combat", None)
        if checkpoint is not None else None
    )
    if combat is None:
        return False
    spawned_hostiles = []
    for combatant in getattr(combat, "combatants", []) or []:
        character_id = str(getattr(combatant, "character_id", "") or "")
        combatant_id = str(getattr(combatant, "combatant_id", "") or "")
        if character_id.startswith("mon_") or combatant_id.startswith("mon_"):
            spawned_hostiles.append(combatant)
    if not spawned_hostiles:
        return False
    return all(
        str(getattr(combatant, "defeat_state", "") or "") != "active"
        or int(getattr(combatant, "hit_points_current", 0) or 0) <= 0
        or bool(getattr(combatant, "removed", False))
        for combatant in spawned_hostiles
    )


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
    story_path = _write_story(
        stories_dir,
        start_mode="field" if args.scenario in FIELD_START_SCENARIOS else "startup",
    )
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
            "prompt_chars": len(prompt_text),
            "approx_prompt_tokens": _approx_tokens(prompt_text),
            "contains_engine_knowledge_map": "engine_knowledge_map" in prompt_text,
            "contains_available_catalog": "available_catalog" in prompt_text,
            "contains_dispatch_index": "router_knowledge_dispatch_index" in prompt_text,
            "contains_knowledge_entity_rows": " entity=npc_" in prompt_text,
            "contains_runtime_content_card": any(
                marker in prompt_text
                for marker in (
                    "location_card ref=",
                    "front_signal ref=",
                    "content_known ref=",
                )
            ),
            "contains_turn_hint": "turn_hint " in prompt_text,
            "contains_private_path": "private_extractions/" in prompt_text,
        }
        if role == "content_manager":
            record["content_manager_input"] = _content_manager_prompt_audit(
                prompt_text
            )
        try:
            response = await real_complete(*call_args, **kwargs)
        except Exception:
            record["error"] = traceback.format_exc()
            role_calls.append(record)
            raise
        record["model"] = response.model
        record["usage"] = dict(response.usage or {})
        if role == "content_manager":
            record["content_manager_output"] = _content_manager_output_audit(
                response.parsed
            )
        elif role == "event_router":
            record["router_output"] = _router_output_audit(response.parsed)
        role_calls.append(record)
        return response

    engine.client.complete = _recording_complete  # type: ignore[method-assign]

    session_id = args.session or f"{STORY_ID}_{TS.lower()}"
    transcript: list[dict[str, str]] = []
    try:
        engine.create_empty_session(session_id)
        state = CLIState(engine, session_id, "")
        lines = SCENARIO_COMMANDS[args.scenario]
        for line in lines:
            output = await _run_cli_line(state, line)
            transcript.append({"input": line, "output": output})
            if "error:" in output.lower():
                break
        if args.scenario == "soak" and (
            not transcript or "error:" not in transcript[-1]["output"].lower()
        ):
            soak_index = 0
            while len(transcript) < args.max_soak_commands:
                current_ckpt = engine.load_latest(session_id)
                if _checkpoint_turn_index(current_ckpt) >= args.target_turns:
                    break
                pending_rolls = state._joined_pending_roll_prompts()
                if _soak_defeated_spawned_hostiles_active(current_ckpt):
                    line = "/combat end"
                elif pending_rolls:
                    line = "/roll all"
                else:
                    line = _next_soak_command(current_ckpt, soak_index)
                    soak_index += 1
                output = await _run_cli_line(state, line)
                transcript.append({"input": line, "output": output})
                if "error:" in output.lower():
                    break
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
    content_call_count = len(content_calls)
    router_call_count = len(router_calls)
    router_modes = [
        call.get("router_output", {}).get("interaction_mode", "")
        for call in router_calls
    ]
    content_requested_keys = [
        item.get("key", "")
        for call in content_calls
        for item in call.get("content_manager_output", {}).get(
            "router_required_keys",
            [],
        )
    ]
    agent_calls = [
        call
        for call in role_calls
        if call["role"] in {"agent", "agent_standard", "agent_convenience"}
    ]
    prompt_budgets = _content_manager_prompt_budgets(role_calls)
    log_warnings = _log_warning_lines()
    ad_hoc_imported_combat_ids = _ad_hoc_spawn_ids_in_imported_combat(checkpoint)
    checks = [
        _check("story_seed_validated", story_path.exists(), str(story_path)),
        _check("cli_completed_without_error_output", not cli_errors, cli_errors),
        _check("content_manager_called", bool(content_calls), content_calls),
        _check(
            "content_manager_prompt_budgets_recorded",
            bool(prompt_budgets)
            and all((item.get("usage") or {}) for item in prompt_budgets),
            prompt_budgets,
        ),
        _check(
            "content_manager_dispatch_index_observed",
            bool(prompt_budgets)
            and all(
                0 < int(item.get("dispatch_keys", 0)) <= 40
                for item in prompt_budgets
            ),
            prompt_budgets,
        ),
        _check(
            "content_manager_catalog_not_rendered",
            all(
                not call.get("contains_available_catalog", False)
                for call in content_calls
            ),
            content_calls,
        ),
        _check(
            "content_manager_prompt_hashes_stripped",
            bool(prompt_budgets)
            and all(
                int(item.get("hash_like_token_count", 0)) == 0
                for item in prompt_budgets
            ),
            prompt_budgets,
        ),
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
            all(call["role"] != "agent" for call in agent_calls)
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
            "module_startup_content_introduced",
            any("loc.cartophile_collection" in ref for ref in introduced_refs)
            and any("front.expedition_obligation" in ref for ref in introduced_refs),
            introduced_refs,
        ),
        _check(
            "knowledge_map_seed_present",
            "npc_cartophile" in knowledge_entities,
            knowledge_entities,
        ),
        _check(
            "content_manager_outputs_recorded",
            bool(content_calls)
            and all("content_manager_output" in call for call in content_calls),
            content_calls,
        ),
    ]
    if args.scenario in ROUTE_EXPLORATION_SCENARIOS:
        checks.extend([
            _check(
                "no_unknown_agent_drop_warnings",
                not any(
                    "router picked unknown agent id" in line
                    for line in log_warnings
                ),
                log_warnings,
            ),
            _check(
                "content_manager_throttled_below_router_calls",
                content_call_count < router_call_count,
                {
                    "content_manager_calls": content_call_count,
                    "event_router_calls": router_call_count,
                },
            ),
            _check(
                "route_exploration_content_introduced",
                any("loc.barrier_peaks_route" in ref for ref in introduced_refs),
                introduced_refs,
            ),
        ])
    if args.scenario == "soak":
        checks.extend([
            _check(
                "soak_target_turns_reached",
                turn_index >= args.target_turns,
                {
                    "turn_index": turn_index,
                    "target_turns": args.target_turns,
                    "commands": len(transcript),
                    "max_soak_commands": args.max_soak_commands,
                },
            ),
            _check(
                "soak_not_left_in_defeated_combat_loop",
                not _soak_defeated_spawned_hostiles_active(checkpoint),
                _combat_state_summary(checkpoint),
            ),
        ])
    if args.scenario == "deep":
        checks.extend([
            _check(
                "content_manager_requested_route_context",
                any("route" in key or "barrier_peaks" in key for key in content_requested_keys),
                content_requested_keys,
            ),
        ])
    if args.scenario in COMBAT_SCENARIOS:
        checks.extend([
            _check(
                "combat_path_exercised",
                any(mode == "dnd_combat_start" for mode in router_modes),
                router_modes,
            ),
            _check(
                "imported_encounter_did_not_mix_ad_hoc_spawns",
                not ad_hoc_imported_combat_ids,
                {
                    "ad_hoc_spawn_ids": ad_hoc_imported_combat_ids,
                    "combat": _combat_state_summary(checkpoint),
                },
            ),
        ])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario": args.scenario,
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "session_id": session_id,
        "story_id": STORY_ID,
        "story_path": str(story_path.relative_to(REPO_ROOT)),
        "installed_story": bool(args.install_story),
        "target_turns": args.target_turns if args.scenario == "soak" else None,
        "max_soak_commands": (
            args.max_soak_commands if args.scenario == "soak" else None
        ),
        "command_count": len(transcript),
        "manual_combat_end_used": any(
            item["input"].strip().lower() == "/combat end"
            for item in transcript
        ),
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
        "combat": _combat_state_summary(checkpoint),
        "content_manager_prompt_budgets": prompt_budgets,
        "usage_totals": _usage_totals(role_calls),
        "log_warnings": log_warnings,
        "call_counts": {
            "content_manager": content_call_count,
            "event_router": router_call_count,
            "role_calls_total": len(role_calls),
        },
        "transcript": transcript,
        "role_calls": role_calls,
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
        f"Scenario: `{report['scenario']}`",
        f"Run directory: `{report['run_dir']}`",
        f"Story: `{report['story_id']}`",
        f"Session: `{report['session_id']}`",
    ]
    if report.get("target_turns"):
        lines.append(f"Target turns: `{report['target_turns']}`")
    lines.extend([
        f"Checks: `{passed}/{total}`",
        f"Turn index: `{report.get('turn_index', 0)}`",
        f"CLI commands: `{report.get('command_count', 0)}`",
        f"Manual combat end used: `{report.get('manual_combat_end_used', False)}`",
        "",
        "## Result",
        "",
    ])
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

    lines.extend(["## Role Calls", ""])
    call_counts = report.get("call_counts", {})
    if call_counts:
        lines.append(
            "Call counts: "
            f"content_manager={call_counts.get('content_manager', 0)}, "
            f"event_router={call_counts.get('event_router', 0)}, "
            f"total={call_counts.get('role_calls_total', 0)}."
        )
        lines.append("")
    usage_totals = report.get("usage_totals", {})
    if usage_totals:
        lines.append("Usage totals:")
        for role, usage in sorted(usage_totals.items()):
            lines.append(
                "- "
                f"`{role}` full_in={usage.get('full_input_tokens', 0)} "
                f"prompt={usage.get('prompt_tokens', 0)} "
                f"cache_read={usage.get('cache_read_input_tokens', 0)} "
                f"out={usage.get('completion_tokens', 0)} "
                f"reasoning={usage.get('reasoning_tokens', 0)} "
                f"total={usage.get('total_tokens', 0)}"
            )
        lines.append("")
    for index, call in enumerate(report["role_calls"], start=1):
        usage = call.get("usage") or {}
        lines.append(
            "- "
            f"{index}. role=`{call['role']}` model=`{call.get('model', '-')}` "
            f"schema=`{call['response_model']}` "
            f"full_in={usage.get('full_input_tokens', 0)} "
            f"out={usage.get('completion_tokens', 0)} "
            f"knowledge_map={call['contains_engine_knowledge_map']} "
            f"dispatch_index={call.get('contains_dispatch_index', False)} "
            f"runtime_content={call['contains_runtime_content_card']} "
            f"turn_hint={call['contains_turn_hint']}"
        )
    lines.append("")

    lines.extend(["## Content Manager Audit", ""])
    for index, call in enumerate(
        [
            item for item in report["role_calls"]
            if item["role"] == "content_manager"
        ],
        start=1,
    ):
        output = call.get("content_manager_output", {})
        input_summary = call.get("content_manager_input", {})
        requested = [
            item.get("key", "")
            for item in output.get("router_required_keys", [])
            if item.get("key", "")
        ]
        updates = [
            item.get("ref", "")
            for item in output.get("knowledge_updates", [])
            if item.get("ref", "")
        ]
        candidates = [
            item.get("character_id", "")
            for item in output.get("router_turn_candidates", [])
            if item.get("character_id", "")
        ]
        lines.append(
            "- "
            f"{index}. approx_tokens={input_summary.get('approx_prompt_tokens', 0)} "
            f"hash_like={input_summary.get('hash_like_token_count', 0)} "
            f"recent_facts={input_summary.get('recent_fact_count', 0)} "
            f"router_state_rows={input_summary.get('router_knowledge_state_rows', 0)} "
            f"dispatch_keys={input_summary.get('dispatch_key_count', 0)} "
            f"required_keys={requested or '-'} "
            f"knowledge_updates={updates or '-'} "
            f"turn_candidates={candidates or '-'} "
            f"no_update={output.get('no_update_reason') or '-'}"
        )
    lines.append("")

    lines.extend(["## Combat State", ""])
    combat = report.get("combat", {})
    if not combat.get("active"):
        lines.append("No active D&D combat at final checkpoint.")
    else:
        lines.append(
            f"Active combat: status=`{combat.get('status', '')}` "
            f"round=`{combat.get('round_number', 0)}`."
        )
        for combatant in combat.get("combatants", []):
            hp = combatant.get("hp", {})
            lines.append(
                "- "
                f"`{combatant.get('character_id', '')}` "
                f"hp={hp.get('current', 0)}/{hp.get('max', 0)} "
                f"init={combatant.get('initiative_total', 0)} "
                f"state={combatant.get('defeat_state', '')}"
            )
        if combat.get("audit_tail"):
            lines.append("Audit tail:")
            for audit in combat["audit_tail"]:
                lines.append(f"- {audit}")
    lines.append("")

    if report.get("log_warnings"):
        lines.extend(["## Log Warnings", ""])
        for warning in report["log_warnings"]:
            lines.append(f"- `{_safe_excerpt(warning, limit=260)}`")
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
        "- The reviewed full pack includes downstream semantic records, "
        "encounter templates, statblocks, hazards, treasures, and abstract "
        "theater-map templates; source images and raw OCR stay private."
    )
    lines.append(
        "- The report intentionally records only compact runtime refs and CLI "
        "output, not raw OCR, source PDF text, or private source paths."
    )
    return "\n".join(lines)


async def main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_COMMANDS),
        default="startup",
        help="Which scripted CLI playtest sequence to run.",
    )
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
    parser.add_argument(
        "--target-turns",
        type=int,
        default=35,
        help=(
            "For --scenario soak, continue scripted play until the checkpoint "
            "turn index reaches this value."
        ),
    )
    parser.add_argument(
        "--max-soak-commands",
        type=int,
        default=60,
        help=(
            "For --scenario soak, stop after this many total CLI commands even "
            "if the target turn index has not been reached."
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
