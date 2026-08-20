#!/usr/bin/env python3
"""Live harness for the separated D&D combat manager.

The harness avoids a live generic router. It builds a checkpoint with two
human-bound player characters, one LLM-driven monster, and three canned dummy
monsters, then applies a dummy router `dnd_combat_start` output with a tactical
battle map. Active initiative turns are resolved by the real
`dnd_combat_manager` role, while the agent monster's exact prompt input and
output are captured for QA.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine import dnd_combat
from app.engine.character_agent import CharacterAgent
from app.engine.model_config_sync import sync_checkpoint_runtime_models
from app.engine.prompt_manager import PromptManager
from app.engine.dnd_combat_harness import (
    CapturingDndCombatResolver,
    _capture_dump,
    _combat_summary,
    _event_summary,
    _message_capture,
    _preflight_api_keys,
    _role_label,
    _usage_totals,
    live_all_illegal_stress_turns,
    live_illegal_stress_turns,
    live_quality_findings,
    live_report_checks,
    live_report_markdown,
    make_harness_report_paths,
)
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
from app.schemas.dnd_spatial import (
    DndBattleMapSeed,
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


REPORT_PATHS = make_harness_report_paths(
    REPO_ROOT,
    "dnd_combat_manager_live",
    include_final_checkpoint=True,
)
TS = REPORT_PATHS.timestamp
RUN_DIR = REPORT_PATHS.run_dir
JSON_PATH = REPORT_PATHS.json_path
MD_PATH = REPORT_PATHS.md_path
LOG_PATH = REPORT_PATHS.log_path
FINAL_CHECKPOINT_PATH = REPORT_PATHS.final_checkpoint_path or (
    RUN_DIR / "final_checkpoint.json"
)

PLAYER_IDS = ("pc_aria", "pc_bram")
AGENT_MONSTER_ID = "ashbound_warcaller"
DUMMY_MONSTER_IDS = ("ember_grunt_a", "ember_grunt_b", "ember_grunt_c")
ALL_COMBATANT_IDS = (*PLAYER_IDS, AGENT_MONSTER_ID, *DUMMY_MONSTER_IDS)
ILLEGAL_STRESS_ACTION_INDEX = 2

ILLEGAL_STRESS_ACTIONS = {
    "ember_grunt_a": (
        "pure illegal: unlisted area spell, invisibility, command effect, "
        "and self-heal with no listed action to salvage"
    ),
    "ember_grunt_b": (
        "pure illegal: unlisted spellcasting, duplication, legendary actions, "
        "and no listed spear attack to salvage"
    ),
    "ember_grunt_c": (
        "pure illegal: invisible potion, unlisted mass heal, global restraint, "
        "and teleport with no listed attack to salvage"
    ),
}

ACTION_SCRIPTS = {
    "pc_aria": (
        "I skirt the cracked brazier to engage the Ashbound Warcaller and "
        "strike with my longsword, keeping the fallen pillar between me and "
        "Grunt B if I can.",
        "If Grunt A is still beside me, I shove it toward the cracked brazier; "
        "otherwise I strike the Warcaller with my longsword.",
    ),
    "pc_bram": (
        "I drop behind the broken low wall, shoot Grunt C with my shortbow, "
        "then stay in cover.",
        "If I am restrained or netted, I cut myself free with my dagger; "
        "otherwise I shoot the Warcaller and keep the low wall between me and "
        "Grunt B.",
    ),
    "ember_grunt_a": (
        "I dart around the fallen pillar to claw Aria and keep her boxed in.",
        "I cast Hold Person on Aria, teleport through the fallen pillar, claw "
        "her twice, and set her shield arm on fire.",
        "I cast Fireball from inside the fallen pillar, turn invisible, command "
        "Aria to drop her shield, and heal myself back to full health.",
    ),
    "ember_grunt_b": (
        "I move to the low-wall gap and jab Bram with my spear from the east "
        "side.",
        "I phase straight through the fallen pillar and stab Bram with my spear "
        "from thirty feet away, then make a bonus spear attack at Aria too.",
        "I cast Revivify on the fallen pillar, split into four copies, and give "
        "the Warcaller legendary actions before anyone else can move.",
    ),
    "ember_grunt_c": (
        "I move fifteen feet toward Bram through the rubble and throw the "
        "hooked net.",
        "I teleport onto the low wall, throw hooked nets through the fallen "
        "pillar at both Bram and Aria, and heal myself back to full health.",
        "I drink an invisible potion, cast Mass Cure Wounds on every cultist, "
        "restrain everyone on the map without throwing a net, and teleport to "
        "safety.",
    ),
}


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
    faction: str = "",
    playable: bool = False,
    intentions_enabled: bool = False,
    objectives: list[str] | None = None,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        location="ember_shrine",
        is_playable=playable,
        public_sheet=PublicSheet(
            role=role,
            appearance=appearance,
            faction=faction,
        ),
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
            faction="adventurers",
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
                        "attack": {
                            "bonus": 5,
                            "damage": "1d8+3 slashing",
                            "range": "5 ft",
                        },
                    }
                ],
            ),
        ),
        _char(
            character_id="pc_bram",
            name="Bram Flint",
            role="level 3 halfling scout",
            appearance="leather armor, shortbow, dagger, smoke-dark scarf",
            faction="adventurers",
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
                        "attack": {
                            "bonus": 5,
                            "damage": "1d6+3 piercing",
                            "range": "80/320 ft",
                        },
                    },
                    {
                        "id": "dagger",
                        "name": "Dagger",
                        "attack": {
                            "bonus": 5,
                            "damage": "1d4+3 piercing",
                            "range": "5 ft melee or 20/60 ft thrown",
                        },
                    },
                ],
            ),
        ),
        _char(
            character_id=AGENT_MONSTER_ID,
            name="Ashbound Warcaller",
            role="ash cult warcaller",
            appearance="charred antler mask, ember staff, ritual bells",
            faction="ash_cult",
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
                        "attack": {
                            "bonus": 4,
                            "damage": "1d8+2 bludgeoning",
                            "range": "5 ft",
                        },
                    },
                    {
                        "id": "cinder_bolt",
                        "name": "Cinder Bolt",
                        "attack": {
                            "bonus": 4,
                            "damage": "1d6+2 fire",
                            "range": "60 ft",
                        },
                        "notes": "Ranged spell attack; no rally or ally-buff feature.",
                    },
                ],
            ),
        ),
        _char(
            character_id="ember_grunt_a",
            name="Ember Grunt A",
            role="ash cult bruiser",
            appearance="burned hide vest, hooked claws",
            faction="ash_cult",
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
                        "attack": {
                            "bonus": 4,
                            "damage": "1d6+2 slashing",
                            "range": "5 ft",
                        },
                    }
                ],
            ),
        ),
        _char(
            character_id="ember_grunt_b",
            name="Ember Grunt B",
            role="ash cult spear carrier",
            appearance="cracked mask, short spear, soot-wrapped arms",
            faction="ash_cult",
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
                        "attack": {
                            "bonus": 3,
                            "damage": "1d6+1 piercing",
                            "range": "5 ft melee or 20/60 ft thrown",
                        },
                    }
                ],
            ),
        ),
        _char(
            character_id="ember_grunt_c",
            name="Ember Grunt C",
            role="ash cult net thrower",
            appearance="net bundle, handaxe, scorched leather mask",
            faction="ash_cult",
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
                        "attack": {
                            "bonus": 4,
                            "damage": "0 bludgeoning",
                            "range": "5/15 ft",
                        },
                        "notes": "On a hit, restrains the target until freed.",
                    },
                    {
                        "id": "handaxe",
                        "name": "Handaxe",
                        "attack": {
                            "bonus": 4,
                            "damage": "1d6+2 slashing",
                            "range": "5 ft melee or 20/60 ft thrown",
                        },
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
                "A fallen pillar blocks part of the central sightline.",
                "A cracked brazier smolders hot enough to make forced movement "
                "dangerous.",
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
                "The ember shrine is twelve squares by ten. A waist-high wall "
                "offers cover on the west side, a fallen pillar blocks the "
                "center, rubble slows the eastern lane, and a cracked brazier "
                "smolders hot in the middle."
            ),
        ),
        characters=_characters(),
    )
    sync_checkpoint_runtime_models(ckpt, config)
    return ckpt


def _battle_map_seed() -> DndBattleMapSeed:
    return DndBattleMapSeed(
        present=True,
        map_name="Ember Shrine",
        width=12,
        height=10,
        square_size_ft=5,
        tokens=[
            DndBattleMapToken(
                token_id="pc_aria",
                character_id="pc_aria",
                label="Aria",
                x=1,
                y=4,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id="pc_bram",
                character_id="pc_bram",
                label="Bram",
                x=1,
                y=7,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id=AGENT_MONSTER_ID,
                character_id=AGENT_MONSTER_ID,
                label="Warcaller",
                x=8,
                y=4,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id="ember_grunt_a",
                character_id="ember_grunt_a",
                label="Grunt A",
                x=7,
                y=3,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id="ember_grunt_b",
                character_id="ember_grunt_b",
                label="Grunt B",
                x=8,
                y=7,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id="ember_grunt_c",
                character_id="ember_grunt_c",
                label="Grunt C",
                x=10,
                y=5,
                size_squares=1,
            ),
        ],
        terrain=[
            DndTerrainZone(
                zone_id="low_wall",
                label="Broken low wall",
                x=2,
                y=6,
                width=4,
                height=1,
                blocks_movement=False,
                blocks_line_of_sight=False,
                cover="half",
                notes="Waist-high stone; enough for half cover.",
            ),
            DndTerrainZone(
                zone_id="fallen_pillar",
                label="Fallen stone pillar",
                x=5,
                y=3,
                width=1,
                height=4,
                blocks_movement=True,
                blocks_line_of_sight=True,
                cover="total",
                notes="Collapsed stone; blocks movement and line of sight.",
            ),
            DndTerrainZone(
                zone_id="cracked_brazier",
                label="Cracked brazier",
                x=6,
                y=5,
                width=1,
                height=1,
                blocks_movement=True,
                blocks_line_of_sight=False,
                cover="none",
                notes="Hot enough that forced contact may hurt.",
            ),
            DndTerrainZone(
                zone_id="rubble",
                label="Rubble",
                x=9,
                y=6,
                width=2,
                height=2,
                blocks_movement=False,
                blocks_line_of_sight=False,
                cover="half",
                notes="Broken masonry; treat as difficult footing.",
            ),
        ],
        areas=[],
        notes="Complex v2 tactical map for live combat-manager testing.",
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
            "using your listed combat options. Include intended movement and "
            "target in character-facing prose. Do not invent named features "
            "that are not listed, and do not use markdown action labels."
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
    action_index: int,
    agent: CharacterAgent,
    ckpt: CheckpointFile,
) -> tuple[str, str, dict[str, Any]]:
    if actor_id == AGENT_MONSTER_ID:
        intention, detail = await _agent_monster_intention(agent, ckpt)
        return intention, "agent_llm", detail
    script = ACTION_SCRIPTS.get(actor_id)
    if script:
        index = min(action_index, len(script) - 1)
        source = "player_canned" if actor_id in PLAYER_IDS else "dummy_canned"
        return script[index], source, {"script_index": index}
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
        messages = kwargs.get("messages") or []
        entry: dict[str, Any] = {
            "role": str(role),
            "response_model": response_model.__name__ if response_model else "",
        }
        if str(role) in {"agent", "agent_standard", "agent_convenience"}:
            entry["messages"] = _message_capture(messages)
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
    actor_turn_counts: dict[str, int] = {}
    error = ""
    start_output: DndEventRouterOutput | None = None
    try:
        start_output = _start_combat_from_dummy_router(ckpt)
        for turn_number in range(1, max_turns + 1):
            if ckpt.session.active_combat is None:
                break
            actor_id = _current_actor_id(ckpt)
            action_index = actor_turn_counts.get(actor_id, 0)
            call_start = len(role_calls)
            capture_start = len(resolver.captures)
            intention, source, source_detail = await _next_intention(
                actor_id,
                action_index,
                agent,
                ckpt,
            )
            actor_turn_counts[actor_id] = action_index + 1
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
                "actor_action_index": action_index,
                "source": source,
                "source_detail": source_detail,
                "intention": intention,
                "result": _event_summary(result, include_observers=True),
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
        "max_turns": max_turns,
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
        "canonical_events": [
            _event_summary(event, include_observers=True)
            for event in ckpt.canonical_events
        ],
        "error": error,
    }
    report["all_illegal_stress_turns"] = live_all_illegal_stress_turns(
        report,
        illegal_stress_actions=ILLEGAL_STRESS_ACTIONS,
        illegal_action_index=ILLEGAL_STRESS_ACTION_INDEX,
    )
    report["illegal_stress_turns"] = live_illegal_stress_turns(
        report,
        illegal_action_index=ILLEGAL_STRESS_ACTION_INDEX,
    )
    report["checks"] = live_report_checks(
        report,
        player_ids=PLAYER_IDS,
        agent_monster_id=AGENT_MONSTER_ID,
        dummy_monster_ids=DUMMY_MONSTER_IDS,
        illegal_action_index=ILLEGAL_STRESS_ACTION_INDEX,
    )
    report["quality_findings"] = live_quality_findings(
        report,
        illegal_action_index=ILLEGAL_STRESS_ACTION_INDEX,
    )
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(live_report_markdown(report), encoding="utf-8")
    return report


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turns", type=int, default=18)
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
