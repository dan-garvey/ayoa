#!/usr/bin/env python3
"""Live harness for the separated D&D combat manager.

The harness avoids a live generic router. It builds a checkpoint with two
human-bound player characters, one LLM-driven monster, and three canned dummy
monsters, then applies a dummy router `dnd_combat_start` output with a simple
battle map. Active initiative turns are resolved by the real
`dnd_combat_manager` role.

Outputs:
  app/storage/playtest_reports/dnd_combat_manager_live_<timestamp>/report.json
  app/storage/playtest_reports/dnd_combat_manager_live_<timestamp>/report.md
  app/storage/playtest_reports/dnd_combat_manager_live_<timestamp>/final_checkpoint.json
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine import dnd_combat
from app.engine.character_agent import CharacterAgent
from app.engine.dnd_combat_resolution import DndCombatResolver
from app.engine.model_config_sync import sync_checkpoint_runtime_models
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop import (
    _start_dnd_combat_from_router_signal,
    broadcast_event,
    flush_combat_visible_facts,
)
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_cat_ii import (
    DndCombatManagerAdjudication,
    RollPlan,
)
from app.schemas.dnd_spatial import (
    DndBattleMapState,
    DndBattleMapToken,
    DndTerrainZone,
)
from app.schemas.event_router import DndEventRouterOutput, DndObserverEntry
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication
from app.schemas.state import (
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)


REPORT_DIR = REPO_ROOT / "app/storage/playtest_reports"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = REPORT_DIR / f"dnd_combat_manager_live_{TS}"
JSON_PATH = RUN_DIR / "report.json"
MD_PATH = RUN_DIR / "report.md"
LOG_PATH = RUN_DIR / "run.log"
FINAL_CHECKPOINT_PATH = RUN_DIR / "final_checkpoint.json"

PLAYER_IDS = ("pc_aria", "pc_bram")
AGENT_MONSTER_ID = "ashbound_warcaller"
DUMMY_MONSTER_IDS = ("ember_grunt_a", "ember_grunt_b", "ember_grunt_c")
ALL_COMBATANT_IDS = (*PLAYER_IDS, AGENT_MONSTER_ID, *DUMMY_MONSTER_IDS)

PLAYER_ACTIONS = {
    "pc_aria": (
        "I advance to hold the centerline and strike the Ashbound Warcaller "
        "with my longsword."
    ),
    "pc_bram": (
        "I duck behind the broken low wall and shoot the Ashbound Warcaller "
        "with my shortbow."
    ),
}
DUMMY_ACTIONS = {
    "ember_grunt_a": "I rush Aria and claw at her shield arm.",
    "ember_grunt_b": "I circle toward Bram and jab him with my spear.",
    "ember_grunt_c": "I throw a hooked net toward Bram from the rubble.",
}


@dataclass
class CombatCapture:
    actor_id: str
    intention: str
    packet: str = ""
    roll_plan: dict[str, Any] | None = None
    roll_ledger: list[str] | None = None
    adjudication: dict[str, Any] | None = None


class CapturingDndCombatResolver(DndCombatResolver):
    def __init__(self, client: LLMClient, prompt_mgr: PromptManager):
        super().__init__(client, prompt_mgr)
        self.captures: list[CombatCapture] = []
        self._active_capture: CombatCapture | None = None
        self._active_actor_id = ""
        self._active_intention = ""

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

    async def _plan_rolls(self, packet: str) -> RollPlan:
        plan = await super()._plan_rolls(packet)
        self._active_capture = CombatCapture(
            actor_id=self._active_actor_id,
            intention=self._active_intention,
            packet=packet,
            roll_plan=plan.model_dump(mode="json"),
        )
        return plan

    async def _finalize(
        self,
        packet: str,
        ledger_lines: list[str],
    ) -> DndCombatManagerAdjudication:
        adjudication = await super()._finalize(packet, ledger_lines)
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


def _mechanics(
    *,
    strength: int,
    dexterity: int,
    constitution: int,
    armor_class: int,
    hp: int,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "ruleset_id": "dnd5e_basic",
        "level": 3,
        "proficiency_bonus": 2,
        "ability_scores": {
            "str": strength,
            "dex": dexterity,
            "con": constitution,
            "int": 10,
            "wis": 10,
            "cha": 10,
        },
        "skill_proficiencies": [],
        "saving_throw_proficiencies": ["str", "con"],
        "armor_class": armor_class,
        "hit_points": {"current": hp, "max": hp, "temporary": 0},
        "conditions": [],
        "resources": {},
        "dnd5e_sheet": {"statblock": {"actions": actions}},
        "raw": {},
    }


def _char(
    *,
    character_id: str,
    name: str,
    role: str,
    appearance: str,
    mechanics: dict[str, Any],
    playable: bool = False,
    intentions_enabled: bool = False,
    objectives: list[str] | None = None,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        location="ember_shrine",
        is_playable=playable,
        public_sheet=PublicSheet(role=role, appearance=appearance),
        private_state=PrivateState(
            goals=["Survive the skirmish and act from the immediate fiction."],
            current_objectives=objectives or ["Win the current fight."],
            secrets=[],
            intentions_enabled=intentions_enabled,
        ),
        known_context=(
            "A D&D 5e initiative scene is active in a small ruined shrine. "
            "The combatants can see and hear each other."
        ),
        mechanics=mechanics,
    )


def _characters() -> list[CharacterRecord]:
    return [
        _char(
            character_id="pc_aria",
            name="Aria Venn",
            role="level 3 human fighter",
            appearance="chain shirt, round shield, longsword, ash-streaked cloak",
            playable=True,
            mechanics=_mechanics(
                strength=16,
                dexterity=12,
                constitution=14,
                armor_class=17,
                hp=32,
                actions=[
                    {
                        "id": "longsword",
                        "name": "Longsword",
                        "attack": {"bonus": 5, "damage": "1d8+3 slashing"},
                    }
                ],
            ),
        ),
        _char(
            character_id="pc_bram",
            name="Bram Flint",
            role="level 3 halfling scout",
            appearance="leather armor, shortbow, dagger, smoke-dark scarf",
            playable=True,
            mechanics=_mechanics(
                strength=10,
                dexterity=16,
                constitution=12,
                armor_class=15,
                hp=24,
                actions=[
                    {
                        "id": "shortbow",
                        "name": "Shortbow",
                        "attack": {"bonus": 5, "damage": "1d6+3 piercing"},
                    },
                    {
                        "id": "dagger",
                        "name": "Dagger",
                        "attack": {"bonus": 5, "damage": "1d4+3 piercing"},
                    },
                ],
            ),
        ),
        _char(
            character_id=AGENT_MONSTER_ID,
            name="Ashbound Warcaller",
            role="ash cult warcaller",
            appearance="charred antler mask, ember staff, ritual bells",
            intentions_enabled=True,
            objectives=[
                "Keep pressure on both intruders.",
                "Use the grunts as a screen while controlling the shrine center.",
            ],
            mechanics=_mechanics(
                strength=14,
                dexterity=12,
                constitution=14,
                armor_class=14,
                hp=38,
                actions=[
                    {
                        "id": "ember_staff",
                        "name": "Ember Staff",
                        "attack": {"bonus": 4, "damage": "1d8+2 bludgeoning"},
                    },
                    {
                        "id": "cinder_bolt",
                        "name": "Cinder Bolt",
                        "attack": {"bonus": 4, "damage": "1d6+2 fire"},
                    },
                ],
            ),
        ),
        _char(
            character_id="ember_grunt_a",
            name="Ember Grunt A",
            role="ash cult bruiser",
            appearance="burned hide vest, hooked claws",
            mechanics=_mechanics(
                strength=14,
                dexterity=12,
                constitution=12,
                armor_class=13,
                hp=18,
                actions=[
                    {
                        "id": "claws",
                        "name": "Claws",
                        "attack": {"bonus": 4, "damage": "1d6+2 slashing"},
                    }
                ],
            ),
        ),
        _char(
            character_id="ember_grunt_b",
            name="Ember Grunt B",
            role="ash cult spear carrier",
            appearance="cracked mask, short spear, soot-wrapped arms",
            mechanics=_mechanics(
                strength=12,
                dexterity=12,
                constitution=12,
                armor_class=13,
                hp=16,
                actions=[
                    {
                        "id": "spear",
                        "name": "Spear",
                        "attack": {"bonus": 3, "damage": "1d6+1 piercing"},
                    }
                ],
            ),
        ),
        _char(
            character_id="ember_grunt_c",
            name="Ember Grunt C",
            role="ash cult net thrower",
            appearance="net bundle, handaxe, scorched leather mask",
            mechanics=_mechanics(
                strength=12,
                dexterity=14,
                constitution=10,
                armor_class=13,
                hp=14,
                actions=[
                    {
                        "id": "hooked_net",
                        "name": "Hooked Net",
                        "attack": {"bonus": 4, "damage": "0 bludgeoning"},
                    },
                    {
                        "id": "handaxe",
                        "name": "Handaxe",
                        "attack": {"bonus": 4, "damage": "1d6+2 slashing"},
                    },
                ],
            ),
        ),
    ]


def _checkpoint(config: LLMConfig) -> CheckpointFile:
    session_config = SessionConfig()
    session_config.settings.ruleset_id = "dnd5e_basic"
    session_config.settings.player_roll_mode = "auto"
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=f"dnd_combat_manager_live_{TS.lower()}",
            story_id="dnd_combat_manager_live",
            player_character_id="pc_aria",
            character_bindings={
                "pc_aria": "player_1",
                "pc_bram": "player_2",
            },
            config=session_config,
        ),
        config=session_config,
        world_state=WorldState(
            facts=[
                "Two adventurers have cornered an ash cult warband inside a "
                "small ruined shrine.",
                "A low broken wall divides the west side from the shrine floor.",
                "All six combatants are visible at the start of initiative.",
            ],
            physics_ruleset=PhysicsRuleset(
                strength_limits="dnd5e_basic",
                magic_enabled=True,
            ),
            setting=StorySetting(
                genre="fantasy adventure",
                era="D&D 5e",
                tone="tactical, concrete, fair",
                premise="Live harness for the separated D&D combat manager.",
            ),
            lore=(
                "The ember shrine is ten squares by eight. A broken waist-high "
                "wall offers cover on the west side; a cracked brazier smolders "
                "in the center."
            ),
        ),
        characters=_characters(),
    )
    sync_checkpoint_runtime_models(ckpt, config)
    return ckpt


def _battle_map_seed() -> DndBattleMapState:
    return DndBattleMapState(
        present=True,
        map_name="Ember Shrine",
        width=10,
        height=8,
        square_size_ft=5,
        tokens=[
            DndBattleMapToken(
                token_id="pc_aria",
                character_id="pc_aria",
                label="Aria",
                x=1,
                y=3,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id="pc_bram",
                character_id="pc_bram",
                label="Bram",
                x=1,
                y=5,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id=AGENT_MONSTER_ID,
                character_id=AGENT_MONSTER_ID,
                label="Warcaller",
                x=6,
                y=3,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id="ember_grunt_a",
                character_id="ember_grunt_a",
                label="Grunt A",
                x=5,
                y=2,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id="ember_grunt_b",
                character_id="ember_grunt_b",
                label="Grunt B",
                x=6,
                y=5,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id="ember_grunt_c",
                character_id="ember_grunt_c",
                label="Grunt C",
                x=8,
                y=4,
                size_squares=1,
            ),
        ],
        terrain=[
            DndTerrainZone(
                zone_id="low_wall",
                label="Broken low wall",
                x=2,
                y=4,
                width=3,
                height=1,
                blocks_movement=False,
                blocks_line_of_sight=False,
                cover="half",
                notes="Waist-high stone; enough for half cover.",
            ),
            DndTerrainZone(
                zone_id="brazier",
                label="Cracked brazier",
                x=5,
                y=4,
                width=1,
                height=1,
                blocks_movement=True,
                blocks_line_of_sight=False,
                cover="none",
                notes="Difficult to pass through; still smoldering.",
            ),
        ],
        areas=[],
        notes="Simple v1 tactical map for live combat-manager testing.",
    )


def _dummy_router_start_output() -> DndEventRouterOutput:
    return DndEventRouterOutput(
        event_id="",
        decision_rationale=(
            "Dummy harness router output: hostile declaration starts initiative."
        ),
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(feasible=True),
            observable_facts=[
                ObservableFact.all(
                    "Aria raises her shield and the ash cult warband surges "
                    "into motion."
                )
            ],
        ),
        event_kind="state_change",
        requires_responders=False,
        required_responders=[],
        observers=[
            DndObserverEntry(
                character_id=cid,
                observation_level="d",
                routing_role="observe_only",
            )
            for cid in ALL_COMBATANT_IDS
        ],
        spawn=[],
        dormant=[],
        cull=[],
        interaction_mode="dnd_combat_start",
        combatant_ids=list(ALL_COMBATANT_IDS),
        combatant_spawns=[],
        battle_map_seed=_battle_map_seed(),
    )


def _force_initiative_order(ckpt: CheckpointFile) -> None:
    combat = ckpt.session.active_combat
    if combat is None:
        return
    initiative = {
        "pc_aria": (18, 16),
        "pc_bram": (15, 13),
        AGENT_MONSTER_ID: (13, 12),
        "ember_grunt_a": (11, 10),
        "ember_grunt_b": (9, 8),
        "ember_grunt_c": (7, 6),
    }
    for combatant in combat.combatants:
        total, roll = initiative.get(combatant.character_id, (5, 5))
        combatant.initiative_total = total
        combatant.initiative_roll = roll
        combatant.initiative_detail = "harness preset"
    dnd_combat.sort_turn_order(combat)
    combat.turn_index = 0


def _start_combat_from_dummy_router(ckpt: CheckpointFile) -> DndEventRouterOutput:
    start = _dummy_router_start_output()
    started = _start_dnd_combat_from_router_signal(
        ckpt,
        start,
        actor_id="pc_aria",
        intention="I raise my shield and challenge the ash cult warband.",
    )
    if not started:
        raise RuntimeError("Dummy router combat-start output did not start combat.")
    _force_initiative_order(ckpt)
    broadcast_event(ckpt, start, actor_id="pc_aria")
    return start


def _current_actor_id(ckpt: CheckpointFile) -> str:
    combat = ckpt.session.active_combat
    if combat is None:
        return ""
    return dnd_combat.current_combatant(combat).character_id


async def _agent_monster_intention(
    agent: CharacterAgent,
    ckpt: CheckpointFile,
) -> tuple[str, dict[str, str]]:
    character = _character_by_id(ckpt, AGENT_MONSTER_ID)
    output = await agent.turn(
        character=character,
        checkpoint=ckpt,
        acting_character_id=AGENT_MONSTER_ID,
        frame="foreground",
        local_context=(
            "It is your initiative turn. Choose one concrete combat action "
            "using your listed combat options or a short command to your grunts."
        ),
    )
    public_text = output.public_text.strip()
    intention = public_text or "(remains silent)"
    return intention, {
        "public_text": public_text,
        "private_intent": output.intent.strip(),
    }


def _character_by_id(ckpt: CheckpointFile, character_id: str) -> CharacterRecord:
    for character in ckpt.characters:
        if character.character_id == character_id:
            return character
    raise KeyError(character_id)


async def _next_intention(
    actor_id: str,
    agent: CharacterAgent,
    ckpt: CheckpointFile,
) -> tuple[str, str, dict[str, Any]]:
    if actor_id in PLAYER_ACTIONS:
        return PLAYER_ACTIONS[actor_id], "player_canned", {}
    if actor_id == AGENT_MONSTER_ID:
        intention, detail = await _agent_monster_intention(agent, ckpt)
        return intention, "agent_llm", detail
    if actor_id in DUMMY_ACTIONS:
        return DUMMY_ACTIONS[actor_id], "dummy_canned", {}
    return "(defer)", "fallback", {"reason": "No scripted action for actor."}


async def _run_harness(max_turns: int) -> dict[str, Any]:
    load_dotenv()
    config = LLMConfig.from_env()
    missing = _preflight_api_keys(config, {"agent", "dnd_combat_manager"})
    if missing:
        raise SystemExit("Missing API key(s) for: " + ", ".join(missing))

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    client = LLMClient(config)
    prompt_mgr = PromptManager(str(REPO_ROOT / "app/prompts"))
    role_calls: list[dict[str, Any]] = []
    real_complete = client.complete

    async def _recording_complete(*call_args, **kwargs):
        role = kwargs.get("role") or (call_args[0] if call_args else "")
        response_model = kwargs.get("response_model")
        entry: dict[str, Any] = {
            "role": str(role),
            "response_model": response_model.__name__ if response_model else "",
        }
        started = time.perf_counter()
        try:
            response = await real_complete(*call_args, **kwargs)
        except Exception as exc:
            entry["elapsed_s"] = round(time.perf_counter() - started, 3)
            entry["error"] = repr(exc)
            role_calls.append(entry)
            raise
        entry["elapsed_s"] = round(time.perf_counter() - started, 3)
        entry["model"] = getattr(response, "model", "") or ""
        entry["usage"] = dict(getattr(response, "usage", {}) or {})
        role_calls.append(entry)
        return response

    client.complete = _recording_complete  # type: ignore[method-assign]

    ckpt = _checkpoint(config)
    dispatcher = LLMDispatcher(client, prompt_mgr)
    resolver = CapturingDndCombatResolver(client, prompt_mgr)
    dispatcher._dnd_combat = resolver
    agent = CharacterAgent(client, prompt_mgr)

    turns: list[dict[str, Any]] = []
    error = ""
    start_output: DndEventRouterOutput | None = None
    try:
        start_output = _start_combat_from_dummy_router(ckpt)
        for turn_number in range(1, max_turns + 1):
            if ckpt.session.active_combat is None:
                break
            actor_id = _current_actor_id(ckpt)
            call_start = len(role_calls)
            capture_start = len(resolver.captures)
            intention, source, source_detail = await _next_intention(
                actor_id,
                agent,
                ckpt,
            )
            result = await dispatcher.route_combat_action(
                ckpt=ckpt,
                actor_id=actor_id,
                intention=intention,
            )
            broadcast_event(ckpt, result, actor_id=actor_id)
            capture = (
                resolver.captures[-1]
                if len(resolver.captures) > capture_start
                else None
            )
            if ckpt.session.active_combat is not None:
                dnd_combat.advance_turn_with_effects(
                    ckpt.session,
                    characters=ckpt.characters,
                )
                dnd_combat.sync_combat_effects_to_characters(
                    ckpt.session.active_combat,
                    ckpt.characters,
                )
                flush_combat_visible_facts(ckpt)
            turns.append({
                "turn_number": turn_number,
                "actor_id": actor_id,
                "source": source,
                "source_detail": source_detail,
                "intention": intention,
                "result": _event_summary(result),
                "capture": _capture_dump(capture),
                "role_calls": role_calls[call_start:],
                "next_actor_id": _current_actor_id(ckpt),
                "session_conversation_len": len(ckpt.session_conversation),
                "pending_engine_state_updates": list(
                    ckpt.session.pending_engine_state_updates
                ),
                "combat_status": _combat_summary(ckpt),
            })
    except Exception:
        error = traceback.format_exc()
    finally:
        FINAL_CHECKPOINT_PATH.write_text(
            ckpt.model_dump_json(indent=2),
            encoding="utf-8",
        )
        await client.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(RUN_DIR),
        "final_checkpoint_path": str(FINAL_CHECKPOINT_PATH),
        "roles": {
            "agent": _role_label(config, "agent"),
            "dnd_combat_manager": _role_label(config, "dnd_combat_manager"),
        },
        "requirements": {
            "player_characters": list(PLAYER_IDS),
            "llm_agent_monster": AGENT_MONSTER_ID,
            "dummy_monsters": list(DUMMY_MONSTER_IDS),
            "dummy_router_outputs": ["dnd_combat_start"],
            "live_roles_expected": ["agent", "dnd_combat_manager"],
        },
        "dummy_router_start": (
            start_output.model_dump(mode="json") if start_output else {}
        ),
        "initial_map": (
            start_output.battle_map_seed.model_dump(mode="json")
            if start_output else {}
        ),
        "turns": turns,
        "role_calls": role_calls,
        "usage_totals": _usage_totals(role_calls),
        "final_combat": _combat_summary(ckpt),
        "session_conversation": [
            message.model_dump(mode="json")
            for message in ckpt.session_conversation
        ],
        "pending_engine_state_updates": list(
            ckpt.session.pending_engine_state_updates
        ),
        "canonical_events": [_event_summary(event) for event in ckpt.canonical_events],
        "error": error,
    }
    report["checks"] = _checks(report)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(_markdown(report), encoding="utf-8")
    return report


def _event_summary(event: Any) -> dict[str, Any]:
    return {
        "event_id": getattr(event, "event_id", ""),
        "event_kind": getattr(event, "ends_beat_reason", "")
        or getattr(event, "event_kind", ""),
        "decision_rationale": getattr(event, "decision_rationale", ""),
        "facts": [
            fact.text
            for fact in getattr(event.canonical_event, "observable_facts", [])
        ],
        "observers": [
            {
                "character_id": observer.character_id,
                "observation_level": observer.observation_level,
                "routing_role": observer.routing_role,
            }
            for observer in getattr(event, "observers", [])
        ],
    }


def _capture_dump(capture: CombatCapture | None) -> dict[str, Any]:
    if capture is None:
        return {}
    return {
        "actor_id": capture.actor_id,
        "intention": capture.intention,
        "packet": json.loads(capture.packet) if capture.packet else {},
        "roll_plan": capture.roll_plan or {},
        "roll_ledger": capture.roll_ledger or [],
        "adjudication": capture.adjudication or {},
    }


def _combat_summary(ckpt: CheckpointFile) -> dict[str, Any]:
    combat = ckpt.session.active_combat
    if combat is None:
        return {"active": False}
    summary = dnd_combat.public_status(ckpt.session)
    summary["active"] = True
    return summary


def _checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    turns = report.get("turns") or []
    role_calls = report.get("role_calls") or []
    acted_by_source: dict[str, set[str]] = {
        "player_canned": set(),
        "agent_llm": set(),
        "dummy_canned": set(),
    }
    for turn in turns:
        source = turn.get("source", "")
        if source in acted_by_source:
            acted_by_source[source].add(str(turn.get("actor_id", "")))
    active = bool((report.get("final_combat") or {}).get("active"))
    conversation_len = len(report.get("session_conversation") or [])
    return [
        _check("no_harness_error", not report.get("error"), report.get("error")),
        _check(
            "dummy_router_started_combat",
            bool(report.get("dummy_router_start")),
            report.get("dummy_router_start"),
        ),
        _check(
            "simple_battle_map_seeded",
            len((report.get("initial_map") or {}).get("tokens") or []) == 6,
            report.get("initial_map"),
        ),
        _check(
            "two_player_characters_acted",
            acted_by_source["player_canned"] >= set(PLAYER_IDS),
            sorted(acted_by_source["player_canned"]),
        ),
        _check(
            "agent_monster_used_llm",
            acted_by_source["agent_llm"] == {AGENT_MONSTER_ID},
            sorted(acted_by_source["agent_llm"]),
        ),
        _check(
            "three_dummy_monsters_acted",
            acted_by_source["dummy_canned"] >= set(DUMMY_MONSTER_IDS),
            sorted(acted_by_source["dummy_canned"]),
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


def _markdown(report: dict[str, Any]) -> str:
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
        lines.append("")
    if report.get("error"):
        lines.extend(["## Error", "", "```text", report["error"], "```", ""])
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()
    if args.max_turns < 1:
        raise SystemExit("--max-turns must be positive")

    report = await _run_harness(args.max_turns)
    print(JSON_PATH)
    print(MD_PATH)
    failed = False
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status}: {check['name']}")
        failed = failed or not check["passed"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
