#!/usr/bin/env python3
"""Targeted live stress harness for the D&D combat manager.

This is deliberately not a turn-loop playtest. It runs a short progressive
suite of one-action combat-manager packets in a single process so provider
prompt caching can help while each scenario isolates a difficult rules case.

Outputs:
  app/storage/playtest_reports/dnd_combat_manager_stress_<timestamp>/report.json
  app/storage/playtest_reports/dnd_combat_manager_stress_<timestamp>/report.md
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine import dnd_combat
from app.engine.dnd_constants import DND_RUNTIME_KEY
from app.engine.model_config_sync import sync_checkpoint_runtime_models
from app.engine.prompt_manager import PromptManager
from app.engine.dnd_combat_harness import (
    CapturingDndCombatResolver,
    _cache_watch_findings,
    _cache_watch_for_call,
    _capture_dump,
    _checks,
    _event_summary,
    _jsonable,
    _phase_label,
    _preflight_api_keys,
    _quality_findings,
    _role_label,
    _router_observed_facts_by_salience,
    _scenario_findings,
    _usage_totals,
    append_jsonl,
    cat_ii_resource_spends,
    combat_hp_by_id,
    combat_reaction_by_id,
    make_harness_report_paths,
    stress_report_markdown,
)
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.characters import ActorFact, ActorRecord, CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_spatial import (
    DndAreaTemplate,
    DndBattleMapState,
    DndBattleMapToken,
    DndTerrainZone,
)
from app.schemas.state import (
    DndCombatantState,
    DndCombatState,
    DndRuntimeEffect,
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)


REPORT_PATHS = make_harness_report_paths(
    REPO_ROOT,
    "dnd_combat_manager_stress",
    include_raw_calls=True,
)
TS = REPORT_PATHS.timestamp
RUN_DIR = REPORT_PATHS.run_dir
JSON_PATH = REPORT_PATHS.json_path
MD_PATH = REPORT_PATHS.md_path
LOG_PATH = REPORT_PATHS.log_path
RAW_CALLS_PATH = REPORT_PATHS.raw_calls_path or (RUN_DIR / "raw_calls.jsonl")


@dataclass
class CharacterSpec:
    character_id: str
    name: str
    faction: str
    ac: int = 12
    hp: int = 20
    playable: bool = False
    abilities: dict[str, int] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    spellcasting: dict[str, Any] = field(default_factory=dict)
    reaction_available: bool = True
    conditions: list[str] = field(default_factory=list)
    active_effects: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    summary: str
    actor_id: str
    intention: str
    characters: list[CharacterSpec]
    tokens: list[tuple[str, int, int]]
    terrain: list[DndTerrainZone] = field(default_factory=list)
    areas: list[DndAreaTemplate] = field(default_factory=list)
    expectations: dict[str, Any] = field(default_factory=dict)
    category: str = "active"


def _action(
    action_id: str,
    name: str,
    *,
    bonus: int,
    damage: str,
    range_text: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": action_id,
        "name": name,
        "attack": {
            "bonus": bonus,
            "damage": damage,
            "range": range_text,
        },
        "notes": notes,
    }


def _utility_action(
    action_id: str,
    name: str,
    *,
    range_text: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": action_id,
        "name": name,
        "range": range_text,
        "notes": notes,
    }


def _spell(
    spell_id: str,
    name: str,
    *,
    level: int,
    save_ability: str = "",
    dc: int = 0,
    damage: str = "",
    target_text: str,
    range_text: str,
    concentration: bool = False,
    duration_text: str = "",
    consumes_level: int | None = None,
) -> dict[str, Any]:
    consumes = []
    if consumes_level is not None:
        consumes.append({
            "resource_id": f"spell_slot_{consumes_level}",
            "amount": 1,
        })
    return {
        "id": spell_id,
        "name": name,
        "level": level,
        "prepared": True,
        "always_prepared": False,
        "concentration": concentration,
        "duration": {"text": duration_text} if duration_text else {},
        "range": {"text": range_text},
        "target": {"text": target_text},
        "components": {"text": "V, S, M"},
        "attack": {},
        "save": {"ability": save_ability, "dc": dc} if save_ability else {},
        "damage": [{"formula": damage}] if damage else [],
        "healing": [],
        "consumes": consumes,
    }


def _spellcasting(
    *,
    ability: str = "int",
    attack_bonus: int = 7,
    save_dc: int = 15,
    slots: dict[str, Any] | None = None,
    spells: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "profiles": [{
            "id": "stress_caster",
            "name": "Stress Caster",
            "ability": ability,
            "spell_attack_bonus": attack_bonus,
            "spell_save_dc": save_dc,
        }],
        "slots": slots or {"1": {"current": 4, "max": 4}, "3": {"current": 3, "max": 3}},
        "spells": spells,
    }


def _wall(
    zone_id: str,
    label: str,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    cover: str = "total",
    blocks_movement: bool = True,
    blocks_los: bool = True,
    notes: str = "",
) -> DndTerrainZone:
    return DndTerrainZone(
        zone_id=zone_id,
        label=label,
        x=x,
        y=y,
        width=width,
        height=height,
        blocks_movement=blocks_movement,
        blocks_line_of_sight=blocks_los,
        cover=cover,
        notes=notes or label,
    )


def _area(
    template_id: str,
    label: str,
    *,
    shape: str,
    x: int,
    y: int,
    radius_squares: int = 0,
    width: int = 1,
    height: int = 1,
    notes: str = "",
) -> DndAreaTemplate:
    return DndAreaTemplate(
        template_id=template_id,
        label=label,
        shape=shape,
        x=x,
        y=y,
        radius_squares=radius_squares,
        width=width,
        height=height,
        duration_rounds=0,
        notes=notes,
    )


def _base_abilities(overrides: dict[str, int] | None = None) -> dict[str, int]:
    abilities = {
        "str": 10,
        "dex": 10,
        "con": 10,
        "int": 10,
        "wis": 10,
        "cha": 10,
    }
    abilities.update(overrides or {})
    return abilities


def _mechanics(spec: CharacterSpec) -> dict[str, Any]:
    return {
        "ruleset_id": "dnd5e_basic",
        "level": 7,
        "proficiency_bonus": 3,
        "ability_scores": _base_abilities(spec.abilities),
        "skill_proficiencies": [],
        "saving_throw_proficiencies": [],
        "armor_class": spec.ac,
        "hit_points": {
            "current": spec.hp,
            "max": spec.hp,
            "temporary": 0,
        },
        "conditions": list(spec.conditions),
        "resources": {},
        "dnd5e_sheet": {
            "statblock": {
                "actions": spec.actions,
                "spellcasting": spec.spellcasting,
                "defenses": {},
            }
        },
        DND_RUNTIME_KEY: {
            "active_effects": list(spec.active_effects),
        },
        "raw": {},
    }


def _character(spec: CharacterSpec) -> CharacterRecord:
    return CharacterRecord(
        character_id=spec.character_id,
        name=spec.name,
        location="stress_arena",
        is_playable=spec.playable,
        public_sheet=PublicSheet(
            role="D&D stress combatant",
            appearance="combat-ready",
            faction=spec.faction,
        ),
        actor=ActorRecord(
            facts=[
                ActorFact(
                    text=(
                        "You are in a focused combat stress scenario and want "
                        "to resolve the immediate action."
                    )
                )
            ],
        ),
        mechanics=_mechanics(spec),
    )


def _checkpoint(config: LLMConfig, scenario: Scenario) -> CheckpointFile:
    session_config = SessionConfig()
    session_config.settings.ruleset_id = "dnd5e_basic"
    session_config.settings.player_roll_mode = "auto"
    characters = [_character(spec) for spec in scenario.characters]
    by_id = {spec.character_id: spec for spec in scenario.characters}
    bindings = {
        spec.character_id: f"player_{index}"
        for index, spec in enumerate(scenario.characters, start=1)
        if spec.playable
    }
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=f"dnd_combat_manager_stress_{scenario.name}_{TS.lower()}",
            story_id="dnd_combat_manager_stress",
            player_character_id=scenario.actor_id,
            character_bindings=bindings,
            config=session_config,
        ),
        config=session_config,
        world_state=WorldState(
            facts=[
                "This checkpoint exists only to stress-test one combat-manager packet.",
                "Use D&D 5e rules and the supplied map/spell data exactly.",
            ],
            physics_ruleset=PhysicsRuleset(
                strength_limits="dnd5e_basic",
                magic_enabled=True,
            ),
            setting=StorySetting(
                genre="fantasy tactical combat",
                era="D&D 5e",
                tone="adversarial rules QA",
                premise=scenario.summary,
            ),
            lore=scenario.summary,
        ),
        characters=characters,
    )
    battle_map = DndBattleMapState(
        present=True,
        map_name=f"Stress Arena: {scenario.name}",
        width=12,
        height=10,
        square_size_ft=5,
        tokens=[
            DndBattleMapToken(
                token_id=character_id,
                character_id=character_id,
                label=by_id[character_id].name,
                x=x,
                y=y,
                size_squares=1,
            )
            for character_id, x, y in scenario.tokens
        ],
        terrain=scenario.terrain,
        areas=scenario.areas,
        notes=scenario.summary,
    )
    combatants = [
        DndCombatantState(
            combatant_id=spec.character_id,
            character_id=spec.character_id,
            name=spec.name,
            player_controlled=spec.playable,
            armor_class=spec.ac,
            hit_points_current=spec.hp,
            hit_points_max=spec.hp,
            reaction_available=spec.reaction_available,
            conditions=list(spec.conditions),
            active_effects=[
                DndRuntimeEffect.model_validate(effect)
                for effect in spec.active_effects
            ],
            initiative_total=10 if spec.character_id == scenario.actor_id else 5,
        )
        for spec in scenario.characters
    ]
    actor_index = next(
        index for index, spec in enumerate(scenario.characters)
        if spec.character_id == scenario.actor_id
    )
    ckpt.session.active_combat = DndCombatState(
        combat_id=f"stress_{scenario.name}",
        round_number=1,
        turn_index=actor_index,
        combatants=combatants,
        battle_map=battle_map,
    )
    dnd_combat.sort_turn_order(ckpt.session.active_combat)
    ckpt.session.active_combat.turn_index = next(
        index for index, combatant in enumerate(ckpt.session.active_combat.combatants)
        if combatant.character_id == scenario.actor_id
    )
    sync_checkpoint_runtime_models(ckpt, config)
    return ckpt


def _scenarios() -> list[Scenario]:
    longsword = _action(
        "longsword",
        "Longsword",
        bonus=6,
        damage="1d8+3 slashing",
        range_text="5 ft",
    )
    shortbow = _action(
        "shortbow",
        "Shortbow",
        bonus=7,
        damage="1d6+4 piercing",
        range_text="80/320 ft",
        notes="Ranged weapon attack.",
    )
    dagger = _action(
        "dagger",
        "Dagger",
        bonus=7,
        damage="1d4+4 piercing",
        range_text="5 ft or thrown 20/60 ft",
        notes="Finesse melee weapon attack.",
    )
    disarm = _action(
        "disarm",
        "Disarm",
        bonus=6,
        damage="",
        range_text="5 ft",
        notes=(
            "DMG optional action: make a weapon attack contested by the "
            "target's Strength (Athletics) or Dexterity (Acrobatics). On a "
            "win, the target drops one held item. This action deals no damage."
        ),
    )
    surrender_parley = _utility_action(
        "surrender_parley",
        "Surrender Parley",
        range_text="audible speech",
        notes=(
            "The combatant drops weapons, orders allies to stand down, and "
            "offers a concrete concession in exchange for mercy. No roll is "
            "needed when the concession is voluntary and clearly stated."
        ),
    )
    protective_interposition = _utility_action(
        "protective_interposition",
        "Protective Interposition",
        range_text="movement or teleport placement",
        notes=(
            "The combatant deliberately places themself between a hostile "
            "creature and a protected objective or noncombatant. This is a "
            "narrative protection choice, not an attack."
        ),
    )
    reveal_betrayal = _utility_action(
        "reveal_betrayal",
        "Reveal Betrayal",
        range_text="adjacent object interaction and audible speech",
        notes=(
            "The combatant publicly reveals a hidden allegiance and completes "
            "a simple object interaction that changes the scene's stakes."
        ),
    )
    falling_collision = _action(
        "falling_collision",
        "Falling Collision",
        bonus=0,
        damage="4d6 bludgeoning",
        range_text="vertical fall",
        notes=(
            "Xanathar optional falling onto a creature: if the falling creature "
            "enters another creature's space and neither is Tiny, the lower "
            "creature makes a DC 15 Dexterity saving throw. On a failed save, "
            "the fall impacts that creature, the falling damage is divided "
            "evenly between the two creatures, and both creatures end prone."
        ),
    )
    claws = _action(
        "claws",
        "Claws",
        bonus=5,
        damage="1d6+3 slashing",
        range_text="5 ft",
    )
    thunderwave = _spell(
        "thunderwave",
        "Thunderwave",
        level=1,
        save_ability="con",
        dc=15,
        damage="2d8 thunder",
        range_text="Self",
        target_text=(
            "15-foot cube originating from the caster; creatures take half "
            "damage on a successful Constitution save and are pushed 10 feet "
            "away from the caster on a failed save"
        ),
        consumes_level=1,
    )
    magic_missile = _spell(
        "magic_missile",
        "Magic Missile",
        level=1,
        damage="3 darts, each 1d4+1 force",
        range_text="120 ft",
        target_text=(
            "Three darts hit chosen visible creatures; when readied, the spell "
            "is cast now and held for a perceivable trigger"
        ),
        consumes_level=1,
    )
    misty_step = _spell(
        "misty_step",
        "Misty Step",
        level=2,
        range_text="Self",
        target_text=(
            "Bonus action teleport up to 30 feet to an unoccupied space the "
            "caster can see"
        ),
        duration_text="Instantaneous",
        consumes_level=2,
    )
    fireball = _spell(
        "fireball",
        "Fireball",
        level=3,
        save_ability="dex",
        dc=15,
        damage="8d6 fire",
        range_text="150 ft",
        target_text=(
            "20-foot-radius sphere from a point; spreads around corners; "
            "creatures take half damage on a successful Dexterity save"
        ),
        consumes_level=3,
    )
    burning_hands = _spell(
        "burning_hands",
        "Burning Hands",
        level=1,
        save_ability="dex",
        dc=14,
        damage="3d6 fire",
        range_text="Self",
        target_text=(
            "15-foot cone from the caster; creatures in cone take half damage "
            "on a successful Dexterity save"
        ),
        consumes_level=1,
    )
    lightning_bolt = _spell(
        "lightning_bolt",
        "Lightning Bolt",
        level=3,
        save_ability="dex",
        dc=15,
        damage="8d6 lightning",
        range_text="Self",
        target_text=(
            "100-foot-long, 5-foot-wide line from the caster; creatures in line "
            "take half damage on a successful Dexterity save"
        ),
        consumes_level=3,
    )
    hypnotic_pattern = _spell(
        "hypnotic_pattern",
        "Hypnotic Pattern",
        level=3,
        save_ability="wis",
        dc=15,
        damage="",
        range_text="120 ft",
        target_text=(
            "30-foot cube; each creature in the area who sees the pattern makes "
            "a Wisdom save or is charmed and incapacitated; friendly fire applies"
        ),
        concentration=True,
        duration_text=(
            "Concentration, up to 1 minute; effect ends for a creature that "
            "takes damage or is shaken awake with an action"
        ),
        consumes_level=3,
    )
    fear = _spell(
        "fear",
        "Fear",
        level=3,
        save_ability="wis",
        dc=30,
        damage="",
        range_text="Self",
        target_text=(
            "30-foot cone; each creature in the cone makes a Wisdom save. On "
            "a failed save, the creature privately sees a phantasmal image of "
            "its worst fear, becomes frightened, and drops what it is holding"
        ),
        concentration=True,
        duration_text=(
            "Concentration, up to 1 minute; frightened creatures repeat the "
            "save when they end a turn somewhere they cannot see the caster"
        ),
        consumes_level=3,
    )
    phantasmal_force = _spell(
        "phantasmal_force",
        "Phantasmal Force",
        level=2,
        save_ability="int",
        dc=30,
        damage="",
        range_text="60 ft",
        target_text=(
            "One creature makes an Intelligence save. On a failed save, the "
            "target privately perceives a chosen phantasmal object, creature, "
            "or visible phenomenon as real. The target is not forced to move "
            "or choose a particular response unless later circumstances do so."
        ),
        concentration=True,
        duration_text=(
            "Concentration, up to 1 minute. The target may use a later action "
            "to examine the phenomenon with an Intelligence check against the "
            "spell save DC."
        ),
        consumes_level=2,
    )
    web = _spell(
        "web",
        "Web",
        level=2,
        save_ability="dex",
        dc=15,
        damage="",
        range_text="60 ft",
        target_text=(
            "20-foot cube of webs; difficult terrain; creatures in the area "
            "make Dexterity saves or are restrained; concentration up to 1 hour"
        ),
        concentration=True,
        duration_text="Concentration, up to 1 hour",
        consumes_level=2,
    )
    cloudkill_area = _spell(
        "cloudkill_area",
        "Cloudkill",
        level=5,
        save_ability="con",
        dc=30,
        damage="5d8 poison",
        range_text="active area",
        target_text=(
            "A creature that starts its turn in the cloud makes a Constitution "
            "save; poison damage on a failure and no damage on a success"
        ),
        concentration=True,
        duration_text="Concentration, up to 10 minutes",
    )
    sickening_radiance_area = _spell(
        "sickening_radiance_area",
        "Sickening Radiance",
        level=4,
        save_ability="con",
        dc=30,
        damage="4d10 radiant",
        range_text="active area",
        target_text=(
            "A creature that starts its turn in the radiance makes a "
            "Constitution save; radiant damage and one exhaustion level on a "
            "failure and no damage on a success"
        ),
        concentration=True,
        duration_text="Concentration, up to 10 minutes",
    )

    return [
        Scenario(
            name="opportunity_no_reaction_cost",
            summary=(
                "A player leaves an enemy's reach after attacking. The enemy's "
                "reaction_available flag is false in engine state, but the house "
                "rule says automatic opportunity attacks do not require or spend "
                "combat reactions."
            ),
            actor_id="pc_duelist",
            intention=(
                "I cut the hobgoblin sentinel with my longsword, then run west "
                "out of its reach toward Bram."
            ),
            characters=[
                CharacterSpec(
                    "pc_duelist",
                    "Seren Duelist",
                    "adventurers",
                    ac=16,
                    hp=34,
                    playable=True,
                    abilities={"str": 16, "dex": 14},
                    actions=[longsword],
                ),
                CharacterSpec(
                    "hob_sentinel",
                    "Hobgoblin Sentinel",
                    "hobgoblins",
                    ac=15,
                    hp=22,
                    actions=[longsword],
                    reaction_available=False,
                ),
            ],
            tokens=[
                ("pc_duelist", 4, 4),
                ("hob_sentinel", 5, 4),
            ],
            expectations={
                "packet_forbidden_keys": ["reaction_available"],
                "must_include_roll_targets": ["hob_sentinel"],
                "must_include_opportunity_from": "hob_sentinel",
            },
            category="known_good",
        ),
        Scenario(
            name="fireball_around_corner_friendly_fire",
            summary=(
                "Fireball targets a point behind a fallen pillar. The spell "
                "spreads around corners and should affect both enemies and an "
                "ally inside the radius."
            ),
            actor_id="pc_evoker",
            intention=(
                "I cast Fireball at the point just east of the fallen pillar, "
                "trying to catch both cultists. If Aria is inside the blast, "
                "the spell still detonates there."
            ),
            characters=[
                CharacterSpec(
                    "pc_evoker",
                    "Mira Evoker",
                    "adventurers",
                    ac=13,
                    hp=28,
                    playable=True,
                    abilities={"int": 18, "dex": 14},
                    spellcasting=_spellcasting(spells=[fireball]),
                ),
                CharacterSpec(
                    "pc_aria",
                    "Aria Venn",
                    "adventurers",
                    ac=17,
                    hp=32,
                    playable=True,
                    abilities={"dex": 12},
                    actions=[longsword],
                ),
                CharacterSpec("cult_a", "Cultist A", "ash_cult", ac=12, hp=18),
                CharacterSpec("cult_b", "Cultist B", "ash_cult", ac=12, hp=18),
                CharacterSpec("cult_far", "Far Cultist", "ash_cult", ac=12, hp=18),
            ],
            tokens=[
                ("pc_evoker", 1, 5),
                ("pc_aria", 6, 6),
                ("cult_a", 7, 5),
                ("cult_b", 5, 3),
                ("cult_far", 11, 0),
            ],
            terrain=[
                _wall(
                    "fallen_pillar",
                    "Fallen stone pillar",
                    x=6,
                    y=3,
                    width=1,
                    height=4,
                    cover="none",
                    notes=(
                        "Blocks ordinary sight, but Fireball's spell text says "
                        "the fire spreads around corners."
                    ),
                )
            ],
            areas=[
                _area(
                    "intended_fireball",
                    "Intended Fireball point",
                    shape="circle",
                    x=7,
                    y=5,
                    radius_squares=4,
                    notes="20-foot-radius Fireball blast point.",
                )
            ],
            expectations={
                "forbid_attack_rolls": True,
                "forbid_initial_save_effect_id": True,
                "forbid_visible_damage_numbers": True,
                "forbid_ambiguous_visible_outcomes": True,
                "must_include_save_targets": ["pc_aria", "cult_a", "cult_b"],
                "must_exclude_roll_targets": ["cult_far"],
                "expected_save_damage_spell": True,
                "require_resource_spends": [{
                    "actor_id": "pc_evoker",
                    "resource_id": "spell_slot_3",
                    "source_id": "fireball",
                    "amount": 1,
                    "applied": True,
                }],
            },
        ),
        Scenario(
            name="burning_hands_cone_total_cover",
            summary=(
                "Burning Hands cone catches an enemy, a shielded enemy, and one "
                "cult ally, but a rogue behind a total-cover wall should be "
                "excluded."
            ),
            actor_id="ash_pyro",
            intention=(
                "I fan Burning Hands north through the low-wall gap, not "
                "caring that the ally acolyte is in the cone."
            ),
            characters=[
                CharacterSpec(
                    "ash_pyro",
                    "Ash Pyromancer",
                    "ash_cult",
                    ac=12,
                    hp=24,
                    abilities={"int": 16},
                    spellcasting=_spellcasting(save_dc=14, spells=[burning_hands]),
                ),
                CharacterSpec("pc_aria", "Aria Venn", "adventurers", ac=17, hp=32, playable=True),
                CharacterSpec("pc_bram", "Bram Flint", "adventurers", ac=15, hp=24, playable=True),
                CharacterSpec("pc_rogue", "Rook", "adventurers", ac=14, hp=22, playable=True),
                CharacterSpec("ash_acolyte", "Ash Acolyte", "ash_cult", ac=12, hp=16),
            ],
            tokens=[
                ("ash_pyro", 4, 7),
                ("pc_aria", 5, 6),
                ("pc_bram", 6, 5),
                ("pc_rogue", 4, 3),
                ("ash_acolyte", 5, 5),
            ],
            terrain=[
                _wall(
                    "low_wall",
                    "Broken low wall",
                    x=6,
                    y=5,
                    width=1,
                    height=1,
                    cover="half",
                    blocks_movement=False,
                    blocks_los=False,
                    notes="Half cover against the cone origin.",
                ),
                _wall(
                    "sealed_wall",
                    "Sealed stone wall",
                    x=4,
                    y=4,
                    width=1,
                    height=2,
                    notes="Total cover between the cone origin and Rook.",
                ),
            ],
            areas=[
                _area(
                    "burning_hands_cone",
                    "Burning Hands cone",
                    shape="cone",
                    x=4,
                    y=7,
                    width=3,
                    height=3,
                    notes="15-foot cone aimed north.",
                )
            ],
            expectations={
                "forbid_attack_rolls": True,
                "forbid_initial_save_effect_id": True,
                "forbid_visible_damage_numbers": True,
                "must_include_save_targets": ["pc_aria", "pc_bram", "ash_acolyte"],
                "must_exclude_roll_targets": ["pc_rogue"],
                "expected_save_damage_spell": True,
                "cover_should_matter_for_dex_save_targets": ["pc_bram"],
                "friendly_fire_targets": ["ash_acolyte"],
                "require_resource_spends": [{
                    "actor_id": "ash_pyro",
                    "resource_id": "spell_slot_1",
                    "source_id": "burning_hands",
                    "amount": 1,
                    "applied": True,
                }],
            },
        ),
        Scenario(
            name="lightning_bolt_line_cover",
            summary=(
                "Lightning Bolt runs down a five-foot corridor through a target "
                "with three-quarters cover and a second target behind it. An "
                "adjacent ally is near the line but not inside it."
            ),
            actor_id="pc_evoker",
            intention=(
                "I cast Lightning Bolt straight east down the corridor, through "
                "the arrow-slit guard and into the brute behind him."
            ),
            characters=[
                CharacterSpec(
                    "pc_evoker",
                    "Mira Evoker",
                    "adventurers",
                    ac=13,
                    hp=28,
                    playable=True,
                    abilities={"int": 18, "dex": 14},
                    spellcasting=_spellcasting(spells=[lightning_bolt]),
                ),
                CharacterSpec("pc_aria", "Aria Venn", "adventurers", ac=17, hp=32, playable=True),
                CharacterSpec("slit_guard", "Arrow-Slit Guard", "ash_cult", ac=16, hp=20),
                CharacterSpec("brute", "Ash Brute", "ash_cult", ac=13, hp=32),
            ],
            tokens=[
                ("pc_evoker", 1, 4),
                ("pc_aria", 5, 5),
                ("slit_guard", 5, 4),
                ("brute", 8, 4),
            ],
            terrain=[
                _wall(
                    "arrow_slit",
                    "Arrow-slit barricade",
                    x=4,
                    y=4,
                    width=1,
                    height=1,
                    cover="three_quarters",
                    blocks_movement=True,
                    blocks_los=False,
                    notes="Three-quarters cover against effects crossing it.",
                )
            ],
            areas=[
                _area(
                    "lightning_line",
                    "Lightning Bolt line",
                    shape="line",
                    x=1,
                    y=4,
                    width=12,
                    height=1,
                    notes="100-foot line east from the caster.",
                )
            ],
            expectations={
                "forbid_attack_rolls": True,
                "forbid_initial_save_effect_id": True,
                "forbid_visible_damage_numbers": True,
                "must_include_save_targets": ["slit_guard", "brute"],
                "must_exclude_roll_targets": ["pc_aria"],
                "expected_save_damage_spell": True,
                "cover_should_matter_for_dex_save_targets": ["slit_guard", "brute"],
                "target_reason_contains": {
                    "slit_guard": ["cover"],
                    "brute": ["cover"],
                },
                "require_resource_spends": [{
                    "actor_id": "pc_evoker",
                    "resource_id": "spell_slot_3",
                    "source_id": "lightning_bolt",
                    "amount": 1,
                    "applied": True,
                }],
            },
        ),
        Scenario(
            name="hypnotic_pattern_cube_friendly_fire",
            summary=(
                "Hypnotic Pattern catches two enemies and one ally in a cube. "
                "It should request Wisdom saves, apply no damage, and create "
                "sustained effects only for failed saves."
            ),
            actor_id="pc_bard",
            intention=(
                "I cast Hypnotic Pattern centered over the melee scrum, accepting "
                "that Aria is inside the cube with the cultists."
            ),
            characters=[
                CharacterSpec(
                    "pc_bard",
                    "Ilyra Bard",
                    "adventurers",
                    ac=14,
                    hp=26,
                    playable=True,
                    abilities={"cha": 18},
                    spellcasting=_spellcasting(
                        ability="cha",
                        attack_bonus=7,
                        save_dc=15,
                        spells=[hypnotic_pattern],
                    ),
                ),
                CharacterSpec("pc_aria", "Aria Venn", "adventurers", ac=17, hp=32, playable=True),
                CharacterSpec("cult_a", "Cultist A", "ash_cult", ac=12, hp=18),
                CharacterSpec("cult_b", "Cultist B", "ash_cult", ac=12, hp=18),
                CharacterSpec("cult_outside", "Outside Cultist", "ash_cult", ac=12, hp=18),
            ],
            tokens=[
                ("pc_bard", 2, 5),
                ("pc_aria", 6, 5),
                ("cult_a", 6, 4),
                ("cult_b", 7, 6),
                ("cult_outside", 11, 1),
            ],
            areas=[
                _area(
                    "hypnotic_cube",
                    "Hypnotic Pattern cube",
                    shape="square",
                    x=5,
                    y=4,
                    width=6,
                    height=6,
                    notes="30-foot cube centered on the melee.",
                )
            ],
            expectations={
                "must_include_save_targets": ["pc_aria", "cult_a", "cult_b"],
                "must_exclude_roll_targets": ["cult_outside"],
                "forbid_initial_save_effect_id": True,
                "expected_save_ability": "wis",
                "forbid_damage_records": True,
                "friendly_fire_targets": ["pc_aria"],
            },
            category="known_good",
        ),
        Scenario(
            name="fear_illusion_private_failed_ogres",
            summary=(
                "Fear is an Illusion spell against two ogres with weak Wisdom "
                "saves. Failed targets should become frightened and receive "
                "a private scoped fact for the phantasmal fear they perceive; "
                "the table should only see their outward panic."
            ),
            actor_id="pc_illusionist",
            intention=(
                "I cast Fear in a cone over the two ogres, shaping the image "
                "of the Pale Maw opening behind me. The battle-hardened "
                "captain outside the cone should not be affected."
            ),
            characters=[
                CharacterSpec(
                    "pc_illusionist",
                    "Sera Vale",
                    "adventurers",
                    ac=13,
                    hp=25,
                    playable=True,
                    abilities={"cha": 18, "dex": 14},
                    spellcasting=_spellcasting(
                        ability="cha",
                        attack_bonus=7,
                        save_dc=30,
                        spells=[fear],
                    ),
                ),
                CharacterSpec(
                    "ogre_vanguard",
                    "Ogre Vanguard",
                    "ogres",
                    ac=11,
                    hp=59,
                    abilities={"str": 19, "wis": 7, "int": 5},
                    actions=[claws],
                ),
                CharacterSpec(
                    "ogre_mauler",
                    "Ogre Mauler",
                    "ogres",
                    ac=11,
                    hp=59,
                    abilities={"str": 19, "wis": 7, "int": 5},
                    actions=[claws],
                ),
                CharacterSpec(
                    "hob_captain",
                    "Hobgoblin Captain",
                    "hobgoblins",
                    ac=17,
                    hp=39,
                    abilities={"wis": 12, "int": 12},
                    actions=[longsword],
                ),
            ],
            tokens=[
                ("pc_illusionist", 2, 5),
                ("ogre_vanguard", 5, 4),
                ("ogre_mauler", 6, 6),
                ("hob_captain", 10, 2),
            ],
            areas=[
                _area(
                    "fear_cone",
                    "Fear cone",
                    shape="cone",
                    x=2,
                    y=5,
                    width=6,
                    height=6,
                    notes=(
                        "30-foot cone projected east from Sera; the two ogres "
                        "are inside it, while the hobgoblin captain is outside."
                    ),
                )
            ],
            expectations={
                "must_include_save_targets": ["ogre_vanguard", "ogre_mauler"],
                "must_exclude_roll_targets": ["hob_captain"],
                "must_fail_save_targets": ["ogre_vanguard", "ogre_mauler"],
                "expected_save_ability": "wis",
                "forbid_damage_records": True,
                "require_effect_delta_if_failed": True,
                "require_effect_delta_matches": [
                    {
                        "operation": "start",
                        "target_id": "pc_illusionist",
                        "source_id": "fear",
                        "concentration": True,
                        "conditions_empty": True,
                    },
                    {
                        "operation": "start",
                        "target_id": "ogre_vanguard",
                        "source_id": "fear",
                        "conditions_include": ["frightened"],
                    },
                    {
                        "operation": "start",
                        "target_id": "ogre_mauler",
                        "source_id": "fear",
                        "conditions_include": ["frightened"],
                    },
                ],
                "require_private_fact_for_failed_save_targets": True,
                "private_fact_must_contain_any": [
                    "worst fear",
                    "phantasmal",
                    "pale maw",
                    "terror",
                ],
                "private_fact_forbid_visible_to_non_failed": True,
                "forbid_effect_conditions": ["concentrating"],
                "forbid_fact_contains": [
                    "maw",
                    "worst fear",
                    "phantasmal",
                    "phantasmal image",
                    "Pale Maw",
                ],
                "require_resource_spends": [{
                    "actor_id": "pc_illusionist",
                    "resource_id": "spell_slot_3",
                    "source_id": "fear",
                    "amount": 1,
                    "applied": True,
                }],
            },
        ),
        Scenario(
            name="fear_agent_forced_flight_overrides_attack",
            summary=(
                "An NPC ogre starts its turn under an existing Fear effect. "
                "Even if its submitted intention is to attack, Fear should "
                "force the mandatory flight behavior as a spell effect rather "
                "than letting the agent choose to stand and fight."
            ),
            actor_id="ogre_vanguard",
            intention=(
                "I ignore the terror and charge Sera with my claws, trying to "
                "tear her apart before she can keep casting."
            ),
            characters=[
                CharacterSpec(
                    "pc_illusionist",
                    "Sera Vale",
                    "adventurers",
                    ac=13,
                    hp=25,
                    playable=True,
                    abilities={"cha": 18, "dex": 14},
                    spellcasting=_spellcasting(
                        ability="cha",
                        attack_bonus=7,
                        save_dc=30,
                        spells=[fear],
                    ),
                    active_effects=[{
                        "effect_id": "fear_concentration_pc_illusionist",
                        "name": "Concentrating: Fear",
                        "slug": "concentration_fear",
                        "source_type": "spell",
                        "source_id": "fear",
                        "originator_id": "pc_illusionist",
                        "target_id": "pc_illusionist",
                        "conditions": [],
                        "concentration": True,
                        "duration_kind": "rounds",
                        "duration_amount": 10,
                        "remaining_rounds": 9,
                        "duration_text": "Concentration, up to 1 minute.",
                        "break_triggers": ["lost_concentration"],
                    }],
                ),
                CharacterSpec(
                    "ogre_vanguard",
                    "Ogre Vanguard",
                    "ogres",
                    ac=11,
                    hp=59,
                    abilities={"str": 19, "wis": 7, "int": 5},
                    actions=[claws],
                    conditions=["frightened"],
                    active_effects=[{
                        "effect_id": "fear_ogre_vanguard",
                        "name": "Frightened by Fear",
                        "slug": "fear_frightened",
                        "source_type": "spell",
                        "source_id": "fear",
                        "originator_id": "pc_illusionist",
                        "target_id": "ogre_vanguard",
                        "conditions": ["frightened"],
                        "concentration": False,
                        "duration_kind": "rounds",
                        "duration_amount": 10,
                        "remaining_rounds": 9,
                        "duration_text": (
                            "While frightened by Fear, the target must take "
                            "the Dash action and move away from the caster by "
                            "the safest available route on each of its turns "
                            "unless there is nowhere to move."
                        ),
                        "break_triggers": [
                            "caster_loses_concentration",
                            "successful_recurring_save",
                        ],
                        "recurring_save": {
                            "ability": "wis",
                            "dc": 30,
                            "timing": "end_of_turn",
                            "ends_on": "success",
                            "repeat": True,
                        },
                    }],
                ),
                CharacterSpec(
                    "pc_guard",
                    "Bram Flint",
                    "adventurers",
                    ac=17,
                    hp=38,
                    playable=True,
                    abilities={"str": 16},
                    actions=[longsword],
                ),
            ],
            tokens=[
                ("pc_illusionist", 2, 5),
                ("ogre_vanguard", 5, 5),
                ("pc_guard", 7, 8),
            ],
            expectations={
                "forbid_attack_rolls": True,
                "forbid_action_source_ids": ["claws"],
                "forbid_damage_records": True,
                "forbid_hp_change": True,
                "require_action_matches": {
                    "actor_id": "ogre_vanguard",
                    "source_type": "effect",
                    "source_id": "fear",
                    "effect_id": "fear_ogre_vanguard",
                    "use_mode": "sustain",
                },
                "require_spatial_delta_matches": {
                    "kind": "move_token",
                    "target_id": "ogre_vanguard",
                },
                "forbid_fact_contains": [
                    "chooses to flee",
                    "decides to flee",
                    "surrenders",
                ],
            },
        ),
        Scenario(
            name="commanded_player_flee_overrides_attack",
            summary=(
                "A player-controlled guard starts the turn under Command: "
                "Flee. The player intention tries to attack anyway, but the "
                "compulsory magical control should override the submitted "
                "action exactly as it would for an agent-controlled NPC."
            ),
            actor_id="pc_guard",
            intention=(
                "I refuse the command, plant my feet, and attack the cult "
                "enchanter with my longsword."
            ),
            characters=[
                CharacterSpec(
                    "pc_guard",
                    "Bram Flint",
                    "adventurers",
                    ac=17,
                    hp=38,
                    playable=True,
                    abilities={"str": 16, "wis": 10},
                    actions=[longsword],
                    active_effects=[{
                        "effect_id": "command_flee_pc_guard",
                        "name": "Command: Flee",
                        "slug": "command_flee",
                        "source_type": "spell",
                        "source_id": "command",
                        "originator_id": "cult_enchanter",
                        "target_id": "pc_guard",
                        "conditions": [],
                        "concentration": False,
                        "duration_kind": "rounds",
                        "duration_amount": 1,
                        "remaining_rounds": 1,
                        "duration_text": (
                            "Until the end of this turn. The target must spend "
                            "its action to Dash and move away from the caster "
                            "by the fastest available route, and it does "
                            "nothing else this turn."
                        ),
                        "break_triggers": ["end_of_target_turn"],
                    }],
                ),
                CharacterSpec(
                    "cult_enchanter",
                    "Cult Enchanter",
                    "ash_cult",
                    ac=13,
                    hp=24,
                    abilities={"cha": 16, "wis": 12},
                    actions=[dagger],
                ),
                CharacterSpec(
                    "pc_ally",
                    "Aria Venn",
                    "adventurers",
                    ac=17,
                    hp=32,
                    playable=True,
                    actions=[longsword],
                ),
            ],
            tokens=[
                ("cult_enchanter", 3, 4),
                ("pc_guard", 6, 4),
                ("pc_ally", 8, 4),
            ],
            expectations={
                "forbid_attack_rolls": True,
                "forbid_action_source_ids": ["longsword"],
                "forbid_damage_records": True,
                "forbid_hp_change": True,
                "require_action_matches": {
                    "actor_id": "pc_guard",
                    "source_type": "effect",
                    "source_id": "command",
                    "effect_id": "command_flee_pc_guard",
                    "use_mode": "sustain",
                },
                "require_spatial_delta_matches": {
                    "kind": "move_token",
                    "target_id": "pc_guard",
                },
                "forbid_fact_contains": [
                    "chooses to flee",
                    "decides to flee",
                    "keeps attacking",
                    "longsword connects",
                ],
            },
        ),
        Scenario(
            name="phantasmal_force_private_reality_no_forced_reaction",
            summary=(
                "Phantasmal Force is used creatively to make one orc perceive "
                "an iron portcullis as real. The failed target should receive "
                "that reality as a private scoped fact, while the combat "
                "manager should not force movement, a condition, damage, or "
                "a voluntary reaction."
            ),
            actor_id="pc_illusionist",
            intention=(
                "I cast Phantasmal Force on the orc raider, making it believe "
                "an iron portcullis has slammed down across the east arch. I "
                "am not forcing the orc to run or surrender; I just want that "
                "obstruction to be real to it."
            ),
            characters=[
                CharacterSpec(
                    "pc_illusionist",
                    "Sera Vale",
                    "adventurers",
                    ac=13,
                    hp=25,
                    playable=True,
                    abilities={"cha": 18, "dex": 14},
                    spellcasting=_spellcasting(
                        ability="cha",
                        attack_bonus=7,
                        save_dc=30,
                        slots={"2": {"current": 2, "max": 3}},
                        spells=[phantasmal_force],
                    ),
                ),
                CharacterSpec(
                    "orc_raider",
                    "Orc Raider",
                    "iron_orcs",
                    ac=13,
                    hp=24,
                    abilities={"int": 7, "wis": 11, "str": 16},
                    actions=[claws],
                ),
                CharacterSpec(
                    "orc_captain",
                    "Orc Captain",
                    "iron_orcs",
                    ac=15,
                    hp=45,
                    abilities={"int": 10, "wis": 12},
                    actions=[longsword],
                ),
            ],
            tokens=[
                ("pc_illusionist", 2, 5),
                ("orc_raider", 6, 5),
                ("orc_captain", 6, 7),
            ],
            expectations={
                "must_include_save_targets": ["orc_raider"],
                "must_exclude_roll_targets": ["orc_captain"],
                "must_fail_save_targets": ["orc_raider"],
                "expected_save_ability": "int",
                "forbid_damage_records": True,
                "forbid_hp_change": True,
                "require_effect_delta_if_failed": True,
                "require_effect_delta_matches": [
                    {
                        "operation": "start",
                        "target_id": "pc_illusionist",
                        "source_id": "phantasmal_force",
                        "concentration": True,
                        "conditions_empty": True,
                    },
                    {
                        "operation": "start",
                        "target_id": "orc_raider",
                        "source_id": "phantasmal_force",
                        "conditions_empty": True,
                    },
                ],
                "require_private_fact_for_targets": ["orc_raider"],
                "private_fact_must_contain_any": [
                    "iron portcullis",
                    "east arch",
                    "obstruction",
                ],
                "private_fact_forbid_visible_to_non_targets": True,
                "private_fact_forbid_contains": [
                    "illusion",
                    "illusory",
                    "phantasm",
                    "phantasmal",
                    "hallucination",
                    "perceive",
                    "perceived",
                    "seems",
                    "appears",
                    "appearing",
                    "looks",
                    "real to you",
                    "feels real",
                    "not real",
                    "fake",
                ],
                "forbid_fact_contains": [
                    "iron portcullis",
                    "portcullis",
                    "obstacle",
                    "blocked the passage",
                    "blocks the way",
                    "something blocks",
                    "obstructs",
                    "something now obstructs",
                    "as if",
                    "illusion",
                    "illusion spell",
                    "illusory",
                    "phantasmal",
                    "hallucination",
                    "not real",
                    "fake",
                    "runs away",
                    "surrenders",
                ],
                "forbid_spatial_delta_kinds": [
                    "move_token",
                    "add_area",
                    "remove_area",
                ],
                "forbid_effect_conditions": [
                    "frightened",
                    "restrained",
                    "charmed",
                    "incapacitated",
                    "concentrating",
                ],
                "forbid_router_observed_facts": True,
                "require_resource_spends": [{
                    "actor_id": "pc_illusionist",
                    "resource_id": "spell_slot_2",
                    "source_id": "phantasmal_force",
                    "amount": 1,
                    "applied": True,
                }],
            },
        ),
        Scenario(
            name="web_area_creation_and_restrain",
            summary=(
                "Web creates a persistent 20-foot cube on mixed terrain. It "
                "should request Dexterity saves for creatures in the cube and "
                "persist the web area for future turns."
            ),
            actor_id="pc_evoker",
            intention=(
                "I cast Web across the broken stair and rubble, pinning the two "
                "cultists in the cube but leaving Aria outside the edge."
            ),
            characters=[
                CharacterSpec(
                    "pc_evoker",
                    "Mira Evoker",
                    "adventurers",
                    ac=13,
                    hp=28,
                    playable=True,
                    abilities={"int": 18, "dex": 14},
                    spellcasting=_spellcasting(
                        slots={"2": {"current": 2, "max": 3}},
                        spells=[web],
                    ),
                ),
                CharacterSpec("pc_aria", "Aria Venn", "adventurers", ac=17, hp=32, playable=True),
                CharacterSpec("cult_a", "Cultist A", "ash_cult", ac=12, hp=18),
                CharacterSpec("cult_b", "Cultist B", "ash_cult", ac=12, hp=18),
                CharacterSpec("cult_edge", "Edge Cultist", "ash_cult", ac=12, hp=18),
            ],
            tokens=[
                ("pc_evoker", 1, 6),
                ("pc_aria", 4, 4),
                ("cult_a", 7, 4),
                ("cult_b", 8, 5),
                ("cult_edge", 11, 9),
            ],
            terrain=[
                _wall(
                    "broken_stair",
                    "Broken stair",
                    x=7,
                    y=4,
                    width=2,
                    height=2,
                    cover="half",
                    blocks_movement=False,
                    blocks_los=False,
                    notes="Anchor points and difficult footing.",
                )
            ],
            areas=[
                _area(
                    "web_cube",
                    "Intended Web cube",
                    shape="square",
                    x=6,
                    y=3,
                    width=4,
                    height=4,
                    notes="20-foot cube of Web anchored to rubble and stair.",
                )
            ],
            expectations={
                "must_include_save_targets": ["cult_a", "cult_b"],
                "must_exclude_roll_targets": ["pc_aria", "cult_edge"],
                "forbid_initial_save_effect_id": True,
                "expected_save_ability": "dex",
                "require_spatial_delta_kind": "add_area",
                "require_effect_delta_if_failed": True,
                "require_effect_delta_matches": [{
                    "operation": "start",
                    "target_id": "pc_evoker",
                    "source_id": "web",
                    "concentration": True,
                    "conditions_empty": True,
                }],
                "forbid_effect_delta_target_ids_from_spatial_areas": True,
                "forbid_effect_delta_recurring_save_source_ids": ["web"],
                "forbid_effect_conditions": ["concentrating"],
                "forbid_fact_contains": ["Web takes hold on Mira", "concentrating"],
                "require_resource_spends": [{
                    "actor_id": "pc_evoker",
                    "resource_id": "spell_slot_2",
                    "source_id": "web",
                    "amount": 1,
                    "applied": True,
                }],
            },
        ),
        Scenario(
            name="thunderwave_forced_movement_no_opportunity",
            summary=(
                "Thunderwave can push an enemy out of adjacent melee reach. "
                "The push is forced movement, so it should not generate "
                "opportunity attacks from nearby creatures."
            ),
            actor_id="pc_storm",
            intention=(
                "I cast Thunderwave so the orc raider is blasted east out of "
                "Bram's reach. Do not spare the raider if the push works."
            ),
            characters=[
                CharacterSpec(
                    "pc_storm",
                    "Nera Storm-Sage",
                    "adventurers",
                    ac=14,
                    hp=27,
                    playable=True,
                    abilities={"int": 18, "con": 14},
                    spellcasting=_spellcasting(
                        save_dc=30,
                        spells=[
                            {
                                **thunderwave,
                                "save": {"ability": "con", "dc": 30},
                            }
                        ],
                    ),
                ),
                CharacterSpec(
                    "pc_guard",
                    "Bram Flint",
                    "adventurers",
                    ac=17,
                    hp=38,
                    playable=True,
                    abilities={"str": 16},
                    actions=[longsword],
                ),
                CharacterSpec(
                    "orc_raider",
                    "Orc Raider",
                    "iron_orcs",
                    ac=13,
                    hp=24,
                    abilities={"con": 12, "str": 16},
                    actions=[claws],
                ),
            ],
            tokens=[
                ("pc_storm", 3, 4),
                ("orc_raider", 4, 4),
                ("pc_guard", 4, 5),
            ],
            areas=[
                _area(
                    "thunderwave_cube",
                    "Thunderwave cube",
                    shape="square",
                    x=4,
                    y=2,
                    width=3,
                    height=3,
                    notes=(
                        "15-foot cube originating from Nera and angled to "
                        "catch the orc without catching Bram."
                    ),
                )
            ],
            expectations={
                "forbid_attack_rolls": True,
                "forbid_opportunity_from": ["pc_guard", "pc_storm"],
                "forbid_initial_save_effect_id": True,
                "forbid_visible_damage_numbers": True,
                "must_include_save_targets": ["orc_raider"],
                "must_fail_save_targets": ["orc_raider"],
                "expected_save_ability": "con",
                "expected_save_damage_spell": True,
                "require_spatial_delta_if_failed": "move_token",
                "expected_move_if_failed": {
                    "target_id": "orc_raider",
                    "x": 6,
                    "y": 4,
                },
                "require_resource_spends": [{
                    "actor_id": "pc_storm",
                    "resource_id": "spell_slot_1",
                    "source_id": "thunderwave",
                    "amount": 1,
                    "applied": True,
                }],
            },
        ),
        Scenario(
            name="grapple_shove_contested_no_attack_roll",
            summary=(
                "A brawler tries to shove a guard prone and then grab him. "
                "Grapple and shove are special melee attacks resolved with "
                "contested Strength checks rather than attack rolls."
            ),
            actor_id="pc_brawler",
            intention=(
                "I use my Attack action to shove the Stone Guard prone, then "
                "grapple him with my free hand if I can."
            ),
            characters=[
                CharacterSpec(
                    "pc_brawler",
                    "Kessa Irongrip",
                    "adventurers",
                    ac=15,
                    hp=42,
                    playable=True,
                    abilities={"str": 18, "dex": 12},
                ),
                CharacterSpec(
                    "stone_guard",
                    "Stone Guard",
                    "stone_watch",
                    ac=16,
                    hp=38,
                    abilities={"str": 16, "dex": 10},
                    actions=[longsword],
                ),
            ],
            tokens=[
                ("pc_brawler", 4, 4),
                ("stone_guard", 5, 4),
            ],
            expectations={
                "forbid_attack_rolls": True,
                "must_include_any_roll_kinds": ["ability_check", "skill_check"],
                "require_opposed_rolls": True,
                "forbid_damage_records": True,
                "forbid_fact_contains": ["does not manage to secure a grapple"],
                "condition_fact_requires_delta": [{
                    "target_id": "stone_guard",
                    "condition": "prone",
                }],
            },
        ),
        Scenario(
            name="ready_spell_breaks_existing_concentration",
            summary=(
                "A wizard already concentrating on Web readies Magic Missile. "
                "Readying a spell requires concentration immediately, so the "
                "existing Web concentration should end even though no missile "
                "is released yet."
            ),
            actor_id="pc_evoker",
            intention=(
                "I ready Magic Missile for the instant the cult captain opens "
                "the bronze door. I keep the spell held until that trigger."
            ),
            characters=[
                CharacterSpec(
                    "pc_evoker",
                    "Mira Evoker",
                    "adventurers",
                    ac=13,
                    hp=28,
                    playable=True,
                    abilities={"int": 18, "dex": 14},
                    spellcasting=_spellcasting(
                        slots={
                            "1": {"current": 3, "max": 4},
                            "2": {"current": 1, "max": 3},
                        },
                        spells=[magic_missile, web],
                    ),
                ),
                CharacterSpec(
                    "cult_a",
                    "Restrained Cultist",
                    "ash_cult",
                    ac=12,
                    hp=18,
                    active_effects=[{
                        "effect_id": "web_existing_cult_a",
                        "name": "Web",
                        "slug": "web",
                        "source_type": "spell",
                        "source_id": "web",
                        "originator_id": "pc_evoker",
                        "target_id": "cult_a",
                        "conditions": ["restrained"],
                        "concentration": True,
                        "duration_kind": "hours",
                        "duration_amount": 1,
                        "remaining_rounds": 9,
                        "duration_text": "Concentration, up to 1 hour",
                        "break_triggers": ["concentration"],
                    }],
                    conditions=["restrained"],
                ),
                CharacterSpec("cult_captain", "Cult Captain", "ash_cult", ac=15, hp=34),
            ],
            tokens=[
                ("pc_evoker", 2, 5),
                ("cult_a", 7, 4),
                ("cult_captain", 9, 5),
            ],
            expectations={
                "forbid_rolls": True,
                "forbid_damage_records": True,
                "forbid_hp_change": True,
                "require_effect_end_slug": ["web"],
                "forbid_effect_conditions": ["concentrating"],
                "forbid_fact_contains": ["Magic Missile takes hold on Mira"],
                "require_resource_spends": [{
                    "actor_id": "pc_evoker",
                    "resource_id": "spell_slot_1",
                    "source_id": "magic_missile",
                    "amount": 1,
                    "applied": True,
                }],
            },
        ),
        Scenario(
            name="readied_magic_missile_release",
            summary=(
                "A cult captain opens the bronze door that a wizard named as a "
                "readied-spell trigger. The held Magic Missile should release "
                "from the wizard, spend her reaction, roll direct force damage, "
                "and end the held effect without spending another slot."
            ),
            actor_id="cult_captain",
            intention=(
                "I throw open the bronze door and step through, trusting the "
                "ashes to shield me."
            ),
            characters=[
                CharacterSpec(
                    "pc_evoker",
                    "Mira Evoker",
                    "adventurers",
                    ac=13,
                    hp=28,
                    playable=True,
                    abilities={"int": 18, "dex": 14},
                    spellcasting=_spellcasting(
                        slots={"1": {"current": 2, "max": 4}},
                        spells=[magic_missile],
                    ),
                    active_effects=[{
                        "effect_id": "ready_magic_missile",
                        "name": "Readied Magic Missile",
                        "slug": "readied_spell",
                        "source_type": "spell",
                        "source_id": "magic_missile",
                        "originator_id": "pc_evoker",
                        "target_id": "pc_evoker",
                        "conditions": [],
                        "concentration": True,
                        "duration_kind": "rounds",
                        "duration_amount": 1,
                        "remaining_rounds": 1,
                        "duration_text": "until the trigger or start of next turn",
                        "break_triggers": [
                            "the cult captain opens the bronze door",
                            "concentration ends",
                        ],
                        "metadata": {
                            "readied_action": {
                                "source_id": "magic_missile",
                                "source_type": "spell",
                                "readying_actor_id": "pc_evoker",
                                "trigger_text": (
                                    "when the cult captain opens the bronze door"
                                ),
                                "created_round": 1,
                                "created_turn_index": 0,
                                "requires_reaction": True,
                                "expires_at_start_of_actor_turn": True,
                            }
                        },
                    }],
                ),
                CharacterSpec(
                    "cult_captain",
                    "Cult Captain",
                    "ash_cult",
                    ac=15,
                    hp=34,
                    abilities={"dex": 12},
                    actions=[longsword],
                ),
            ],
            tokens=[
                ("pc_evoker", 3, 5),
                ("cult_captain", 8, 5),
            ],
            expectations={
                "must_include_roll_kinds": ["damage_roll"],
                "minimum_roll_kind_count": {"damage_roll": 1},
                "must_include_roll_targets": ["cult_captain"],
                "require_roll_effect_id": "ready_magic_missile",
                "require_effect_end_slug": ["readied_spell"],
                "require_reaction_spent": "pc_evoker",
                "forbid_resource_spends": True,
                "forbid_visible_damage_numbers": True,
            },
        ),
        Scenario(
            name="ranged_attack_in_melee_disadvantage",
            summary=(
                "An archer fires at a distant target while a hostile enemy is "
                "within 5 feet. Ranged attacks made in close combat should "
                "carry disadvantage."
            ),
            actor_id="pc_archer",
            intention=(
                "I ignore the wolf snapping beside me and shoot the cult "
                "marksman across the room with my shortbow."
            ),
            characters=[
                CharacterSpec(
                    "pc_archer",
                    "Tamsin Vale",
                    "adventurers",
                    ac=15,
                    hp=29,
                    playable=True,
                    abilities={"dex": 18},
                    actions=[shortbow],
                ),
                CharacterSpec(
                    "wolf",
                    "Iron Wolf",
                    "iron_orcs",
                    ac=13,
                    hp=18,
                    actions=[claws],
                ),
                CharacterSpec("cult_marksman", "Cult Marksman", "ash_cult", ac=14, hp=20),
            ],
            tokens=[
                ("pc_archer", 4, 4),
                ("wolf", 4, 5),
                ("cult_marksman", 9, 4),
            ],
            expectations={
                "must_include_roll_targets": ["cult_marksman"],
                "expected_advantage_by_target": {
                    "cult_marksman": "disadvantage",
                },
                "forbid_attack_modifier_bonus": True,
            },
        ),
        Scenario(
            name="invisible_attacker_advantage_and_breaks_effect",
            summary=(
                "An invisible attacker strikes from hiding. The attack should "
                "have advantage, and the Invisibility effect should end because "
                "the effect says it breaks when the target attacks."
            ),
            actor_id="pc_rogue",
            intention=(
                "Still invisible, I step in and stab the necromancer with my "
                "dagger. I am attacking now, so the invisibility should drop."
            ),
            characters=[
                CharacterSpec(
                    "pc_rogue",
                    "Rook",
                    "adventurers",
                    ac=15,
                    hp=26,
                    playable=True,
                    abilities={"dex": 18},
                    actions=[dagger],
                    conditions=["invisible"],
                    active_effects=[{
                        "effect_id": "invisibility_rook",
                        "name": "Invisibility",
                        "slug": "invisibility",
                        "source_type": "spell",
                        "source_id": "invisibility",
                        "originator_id": "pc_evoker",
                        "target_id": "pc_rogue",
                        "conditions": ["invisible"],
                        "concentration": True,
                        "duration_kind": "hours",
                        "duration_amount": 1,
                        "remaining_rounds": 7,
                        "duration_text": (
                            "Concentration, up to 1 hour; ends when the "
                            "target attacks or casts a spell"
                        ),
                        "break_triggers": ["attacks", "casts_spell"],
                    }],
                ),
                CharacterSpec("necromancer", "Necromancer", "bone_cabal", ac=13, hp=30),
            ],
            tokens=[
                ("pc_rogue", 5, 4),
                ("necromancer", 6, 4),
            ],
            expectations={
                "must_include_roll_targets": ["necromancer"],
                "expected_advantage_by_target": {
                    "necromancer": "advantage",
                },
                "forbid_attack_modifier_bonus": True,
                "require_effect_end_slug": ["invisibility"],
            },
        ),
        Scenario(
            name="narrative_surrender_mercy_concession",
            summary=(
                "A named watch captain voluntarily surrenders, orders the "
                "remaining watch to stand down, and offers the prison keys if "
                "the adventurers spare the wounded guards. This should create "
                "post-combat narrative continuity, not just tactical state."
            ),
            actor_id="watch_captain",
            intention=(
                "I drop my ember saber, raise both hands, and surrender to "
                "Seren. I order the remaining watch to stand down and promise "
                "to hand over the prison keys if she spares my wounded guards."
            ),
            characters=[
                CharacterSpec(
                    "watch_captain",
                    "Watch Captain Ardan",
                    "watch",
                    ac=16,
                    hp=9,
                    abilities={"str": 14, "cha": 12},
                    actions=[surrender_parley, longsword],
                    conditions=["bloodied"],
                ),
                CharacterSpec(
                    "pc_duelist",
                    "Seren Duelist",
                    "adventurers",
                    ac=16,
                    hp=34,
                    playable=True,
                    abilities={"str": 16, "dex": 14},
                    actions=[longsword],
                ),
                CharacterSpec(
                    "pc_guard",
                    "Bram Flint",
                    "adventurers",
                    ac=17,
                    hp=38,
                    playable=True,
                    abilities={"str": 16},
                    actions=[longsword],
                ),
                CharacterSpec(
                    "wounded_guard",
                    "Wounded Watch Guard",
                    "watch",
                    ac=14,
                    hp=3,
                    abilities={"str": 12},
                    actions=[longsword],
                    conditions=["prone", "bloodied"],
                ),
            ],
            tokens=[
                ("watch_captain", 5, 4),
                ("pc_duelist", 4, 4),
                ("pc_guard", 4, 5),
                ("wounded_guard", 6, 4),
            ],
            expectations={
                "forbid_rolls": True,
                "expected_combat_status": "ongoing",
                "minimum_router_observed_facts": 1,
                "router_fact_must_contain_any": [
                    "surrender",
                    "stand down",
                    "prison keys",
                    "spares",
                    "mercy",
                ],
                "router_reason_must_not_contain_any": [
                    "tactic",
                    "combat option",
                    "subsequent round",
                    "combat capabilities",
                ],
            },
        ),
        Scenario(
            name="narrative_misty_step_princess_rescue_vow",
            summary=(
                "A player uses Misty Step to protect a princess from an "
                "assassin and makes a public vow while taking the dangerous "
                "position. This should be a durable rescue/protected-objective "
                "fact if the action succeeds."
            ),
            actor_id="pc_warder",
            intention=(
                "I cast Misty Step into the empty space between the assassin "
                "and Princess Alen, raise my warding blade, and shout that the "
                "assassin has to go through me first. I am saving the princess, "
                "not attacking."
            ),
            characters=[
                CharacterSpec(
                    "pc_warder",
                    "Kael Warder",
                    "adventurers",
                    ac=15,
                    hp=31,
                    playable=True,
                    abilities={"int": 16, "dex": 14},
                    actions=[protective_interposition, longsword],
                    spellcasting=_spellcasting(
                        slots={"2": {"current": 2, "max": 3}},
                        spells=[misty_step],
                    ),
                ),
                CharacterSpec(
                    "princess_alen",
                    "Princess Alen",
                    "adventurers",
                    ac=12,
                    hp=18,
                    playable=True,
                    abilities={"dex": 12, "cha": 16},
                ),
                CharacterSpec(
                    "obsidian_assassin",
                    "Obsidian Assassin",
                    "obsidian_court",
                    ac=15,
                    hp=33,
                    abilities={"dex": 18},
                    actions=[dagger],
                ),
            ],
            tokens=[
                ("pc_warder", 2, 4),
                ("obsidian_assassin", 7, 4),
                ("princess_alen", 9, 4),
            ],
            expectations={
                "forbid_rolls": True,
                "expected_combat_status": "ongoing",
                "minimum_router_observed_facts": 1,
                "router_fact_must_contain_any": [
                    "Princess Alen",
                    "princess",
                    "saved",
                    "protect",
                    "vow",
                    "go through",
                ],
                "router_reason_must_not_contain_any": [
                    "tactic",
                    "combat option",
                    "subsequent round",
                    "combat capabilities",
                ],
                "require_spatial_delta_kind": "move_token",
                "forbid_effect_delta_source_ids": ["protective_interposition"],
                "forbid_fact_contains": [
                    "takes hold",
                    "did not make any attack",
                ],
                "require_resource_spends": [{
                    "actor_id": "pc_warder",
                    "resource_id": "spell_slot_2",
                    "source_id": "misty_step",
                    "amount": 1,
                    "applied": True,
                }],
            },
        ),
        Scenario(
            name="narrative_revealed_betrayal_unlocks_gate",
            summary=(
                "A supposed guide reveals a hidden allegiance to the Obsidian "
                "Court and unlocks the ritual gate mid-combat. The betrayal "
                "and opened gate should matter after initiative ends."
            ),
            actor_id="guide_valen",
            intention=(
                "I stop pretending to help Seren. I show the Obsidian Court "
                "brand under my glove, announce that I was their agent all "
                "along, and unlock the ritual gate for the cult captain."
            ),
            characters=[
                CharacterSpec(
                    "guide_valen",
                    "Valen the Guide",
                    "crown_guides",
                    ac=13,
                    hp=24,
                    abilities={"dex": 14, "cha": 16},
                    actions=[reveal_betrayal, dagger],
                ),
                CharacterSpec(
                    "pc_duelist",
                    "Seren Duelist",
                    "adventurers",
                    ac=16,
                    hp=34,
                    playable=True,
                    abilities={"str": 16, "dex": 14},
                    actions=[longsword],
                ),
                CharacterSpec(
                    "cult_captain",
                    "Obsidian Cult Captain",
                    "obsidian_court",
                    ac=15,
                    hp=34,
                    abilities={"dex": 12},
                    actions=[longsword],
                ),
            ],
            tokens=[
                ("guide_valen", 6, 5),
                ("pc_duelist", 4, 5),
                ("cult_captain", 8, 5),
            ],
            terrain=[
                _wall(
                    "ritual_gate",
                    "Locked ritual gate",
                    x=7,
                    y=4,
                    width=1,
                    height=3,
                    notes=(
                        "A locked ritual gate blocks the cult captain's route "
                        "until someone next to it opens the mechanism."
                    ),
                )
            ],
            expectations={
                "forbid_rolls": True,
                "expected_combat_status": "ongoing",
                "minimum_router_observed_facts": 1,
                "router_fact_must_contain_any": [
                    "betray",
                    "Obsidian Court",
                    "ritual gate",
                    "unlocked",
                    "allegiance",
                ],
                "router_reason_must_not_contain_any": [
                    "tactic",
                    "combat option",
                    "subsequent round",
                    "combat capabilities",
                ],
            },
        ),
        Scenario(
            name="dmg_optional_disarm_no_damage",
            summary=(
                "DMG optional Disarm is enabled for this scenario. Disarm uses "
                "a weapon attack contested by the target's Athletics or "
                "Acrobatics; success makes the target drop the named item and "
                "deals no weapon damage."
            ),
            actor_id="pc_duelist",
            intention=(
                "I hook my blade around the watch captain's ember saber and "
                "disarm him. I am not trying to cut him, just knock the saber "
                "out of his hand."
            ),
            characters=[
                CharacterSpec(
                    "pc_duelist",
                    "Seren Duelist",
                    "adventurers",
                    ac=16,
                    hp=34,
                    playable=True,
                    abilities={"str": 16, "dex": 14},
                    actions=[disarm, longsword],
                ),
                CharacterSpec(
                    "watch_captain",
                    "Watch Captain",
                    "watch",
                    ac=16,
                    hp=40,
                    abilities={"str": 14, "dex": 12},
                    actions=[longsword],
                ),
            ],
            tokens=[
                ("pc_duelist", 4, 4),
                ("watch_captain", 5, 4),
            ],
            expectations={
                "must_include_any_roll_kinds": [
                    "attack_roll",
                    "ability_check",
                    "skill_check",
                ],
                "must_include_roll_targets": ["watch_captain"],
                "require_opposed_rolls": True,
                "forbid_damage_records": True,
                "forbid_hp_change": True,
                "forbid_visible_damage_numbers": True,
                "forbid_fact_contains": [
                    "cuts",
                    "slashes",
                    "wounds",
                    "dealt no damage",
                    "deals no damage",
                    "no damage is dealt",
                    "suffers weapon damage",
                    "suffers damage",
                    "weapon damage",
                ],
            },
        ),
        Scenario(
            name="xgte_falling_onto_creature_split_damage",
            summary=(
                "Xanathar's falling-onto-a-creature optional rule is enabled. "
                "A falling creature entering another non-Tiny creature's space "
                "forces a DC 15 Dexterity save by the lower creature; on a "
                "failed save the falling damage is split evenly and both end "
                "prone."
            ),
            actor_id="pc_monk",
            intention=(
                "I drop from the broken balcony directly onto the ogre below, "
                "using my body as the impact. If he fails to get out of the "
                "way, we split the fall and both hit the ground."
            ),
            characters=[
                CharacterSpec(
                    "pc_monk",
                    "Taro Windstep",
                    "adventurers",
                    ac=17,
                    hp=38,
                    playable=True,
                    abilities={"dex": 18, "str": 12},
                    actions=[falling_collision],
                ),
                CharacterSpec(
                    "ogre",
                    "Ogre Below",
                    "ogres",
                    ac=11,
                    hp=59,
                    abilities={"dex": 8, "str": 19},
                    actions=[claws],
                ),
            ],
            tokens=[
                ("pc_monk", 5, 4),
                ("ogre", 5, 5),
            ],
            terrain=[
                _wall(
                    "broken_balcony",
                    "Broken balcony",
                    x=4,
                    y=3,
                    width=3,
                    height=1,
                    cover="none",
                    blocks_movement=False,
                    blocks_los=False,
                    notes=(
                        "The monk begins forty feet above the ogre and can "
                        "fall into the ogre's space this turn."
                    ),
                )
            ],
            expectations={
                "must_include_roll_kinds": ["saving_throw"],
                "must_include_save_targets": ["ogre"],
                "expected_save_ability": "dex",
                "require_hp_decrease_if_failed": ["pc_monk", "ogre"],
                "condition_fact_requires_delta": [
                    {"target_id": "pc_monk", "condition": "prone"},
                    {"target_id": "ogre", "condition": "prone"},
                ],
                "forbid_visible_damage_numbers": True,
                "forbid_fact_contains": ["DC 15", "4d6", "falling damage"],
            },
        ),
        Scenario(
            name="xgte_simultaneous_start_effects_before_movement",
            summary=(
                "Xanathar's simultaneous-effects timing is in force. At the "
                "start of the current actor's turn, Cloudkill and Sickening "
                "Radiance both affect the actor before movement; resolve both "
                "Constitution saves before the actor leaves the overlapping "
                "areas."
            ),
            actor_id="void_intruder",
            intention=(
                "I sprint out of the overlapping green fog and star-bright "
                "radiance toward the hatch before either effect can finish me."
            ),
            characters=[
                CharacterSpec(
                    "void_intruder",
                    "Void Intruder",
                    "raiders",
                    ac=14,
                    hp=120,
                    abilities={"con": 14, "dex": 14},
                    actions=[dagger],
                ),
                CharacterSpec(
                    "cloud_mage",
                    "Cloud Mage",
                    "adventurers",
                    ac=12,
                    hp=22,
                    playable=True,
                    abilities={"con": 12, "int": 18},
                    spellcasting=_spellcasting(
                        save_dc=30,
                        spells=[cloudkill_area],
                    ),
                ),
                CharacterSpec(
                    "radiance_cleric",
                    "Radiance Cleric",
                    "adventurers",
                    ac=18,
                    hp=30,
                    playable=True,
                    abilities={"con": 14, "wis": 18},
                    spellcasting=_spellcasting(
                        ability="wis",
                        save_dc=30,
                        spells=[sickening_radiance_area],
                    ),
                ),
            ],
            tokens=[
                ("void_intruder", 6, 5),
                ("cloud_mage", 2, 5),
                ("radiance_cleric", 3, 6),
            ],
            areas=[
                _area(
                    "cloudkill_area",
                    "Cloudkill",
                    shape="circle",
                    x=5,
                    y=5,
                    radius_squares=4,
                    notes=(
                        "Source cloud_mage; action_id cloudkill_area; DC 30 "
                        "Constitution save at the start of a creature's turn "
                        "in the area; 5d8 poison damage on a failure and no "
                        "damage on a success."
                    ),
                ),
                _area(
                    "sickening_radiance_area",
                    "Sickening Radiance",
                    shape="circle",
                    x=6,
                    y=5,
                    radius_squares=6,
                    notes=(
                        "Source radiance_cleric; action_id "
                        "sickening_radiance_area; DC 30 Constitution save "
                        "when a creature starts its turn in the area; 4d10 "
                        "radiant damage and one level of exhaustion on a "
                        "failure and no damage on a success."
                    ),
                ),
            ],
            expectations={
                "minimum_roll_kind_count": {"saving_throw": 2},
                "must_include_save_targets": ["void_intruder"],
                "require_roll_effect_id": "cloudkill_area",
                "expected_save_ability": "con",
                "expected_save_damage_spell": True,
                "forbid_visible_damage_numbers": True,
                "forbid_fact_contains": [
                    "Constitution save",
                    "succeeded",
                    "failed",
                    "takes radiant damage",
                    "suffers no poison damage",
                    "Dash",
                    "opportunity attack",
                    "hit points",
                    "HP",
                ],
            },
        ),
        Scenario(
            name="spelljammer_weightless_shove_drift_no_fall",
            summary=(
                "Spelljammer weightlessness is active. A creature shoved off "
                "the dorsal deck in Wildspace does not fall downward or take "
                "fall damage; it drifts away within the ship's air envelope "
                "until it grabs something, is rescued, or leaves the envelope."
            ),
            actor_id="pc_spacer",
            intention=(
                "I shove the void pirate over the rail and let him drift away "
                "from the ship. This is Wildspace weightlessness, so I am not "
                "trying to drop him for fall damage."
            ),
            characters=[
                CharacterSpec(
                    "pc_spacer",
                    "Nix Starhand",
                    "adventurers",
                    ac=15,
                    hp=33,
                    playable=True,
                    abilities={"str": 16, "dex": 14},
                ),
                CharacterSpec(
                    "void_pirate",
                    "Void Pirate",
                    "raiders",
                    ac=14,
                    hp=27,
                    abilities={"str": 12, "dex": 16},
                    actions=[dagger],
                ),
            ],
            tokens=[
                ("pc_spacer", 5, 4),
                ("void_pirate", 6, 4),
            ],
            terrain=[
                _wall(
                    "spelljammer_rail",
                    "Spelljammer rail",
                    x=7,
                    y=3,
                    width=1,
                    height=3,
                    cover="half",
                    blocks_movement=False,
                    blocks_los=False,
                    notes=(
                        "Beyond this rail is Wildspace inside the ship's air "
                        "envelope. Unsecured creatures drift; they do not fall."
                    ),
                )
            ],
            expectations={
                "must_include_any_roll_kinds": ["ability_check", "skill_check"],
                "require_opposed_rolls": True,
                "forbid_damage_records": True,
                "forbid_hp_change": True,
                "forbid_visible_damage_numbers": True,
                "forbid_opportunity_from": ["pc_spacer", "void_pirate"],
            },
        ),
    ]


async def _run_harness(selected: list[Scenario]) -> dict[str, Any]:
    load_dotenv()
    config = LLMConfig.from_env()
    missing = _preflight_api_keys(config, {"dnd_combat_manager"})
    if missing:
        raise SystemExit("Missing API key(s) for: " + ", ".join(missing))

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CALLS_PATH.write_text("", encoding="utf-8")
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    client = LLMClient(config)
    prompt_mgr = PromptManager(str(REPO_ROOT / "app/prompts"))
    resolver = CapturingDndCombatResolver(client, prompt_mgr)
    role_calls: list[dict[str, Any]] = []
    active_call_context: dict[str, Any] = {}
    plan_cache_reads_by_scenario: dict[str, int] = {}
    raw_call_index = 0
    real_complete = client.complete

    async def _recording_complete(*call_args, **kwargs):
        nonlocal raw_call_index
        role = kwargs.get("role") or (call_args[0] if call_args else "")
        response_model = kwargs.get("response_model")
        phase = _phase_label(response_model)
        raw_call_index += 1
        call_index = raw_call_index
        entry: dict[str, Any] = {
            "role": str(role),
            "phase": phase,
            "scenario": dict(active_call_context),
            "response_model": response_model.__name__ if response_model else "",
            "raw_call_index": call_index,
        }
        started = time.perf_counter()
        try:
            response = await real_complete(*call_args, **kwargs)
        except Exception as exc:
            entry["elapsed_s"] = round(time.perf_counter() - started, 3)
            entry["error"] = repr(exc)
            role_calls.append(entry)
            _append_raw_call({
                "call_index": call_index,
                "scenario": dict(active_call_context),
                "phase": phase,
                "role": str(role),
                "response_model": entry["response_model"],
                "elapsed_s": entry["elapsed_s"],
                "error": repr(exc),
            })
            raise
        entry["elapsed_s"] = round(time.perf_counter() - started, 3)
        entry["model"] = getattr(response, "model", "") or ""
        entry["usage"] = dict(getattr(response, "usage", {}) or {})
        entry["cache_watch"] = _cache_watch_for_call(
            active_call_context,
            phase=phase,
            usage=entry["usage"],
            plan_cache_reads_by_scenario=plan_cache_reads_by_scenario,
        )
        role_calls.append(entry)
        _append_raw_call({
            "call_index": call_index,
            "scenario": dict(active_call_context),
            "phase": phase,
            "role": str(role),
            "response_model": entry["response_model"],
            "elapsed_s": entry["elapsed_s"],
            "model": entry["model"],
            "usage": entry["usage"],
            "cache_watch": entry["cache_watch"],
            "content": getattr(response, "content", "") or "",
            "parsed": _jsonable(getattr(response, "parsed", None)),
            "reasoning_summaries": list(
                getattr(response, "reasoning_summaries", []) or []
            ),
            "assistant_content": _jsonable(
                getattr(response, "assistant_content", None)
            ),
            "raw_response": _jsonable(getattr(response, "raw_response", None)),
        })
        return response

    client.complete = _recording_complete  # type: ignore[method-assign]

    scenario_results: list[dict[str, Any]] = []
    error = ""
    try:
        for index, scenario in enumerate(selected, start=1):
            active_call_context = {
                "index": index,
                "name": scenario.name,
                "category": scenario.category,
                "actor_id": scenario.actor_id,
                "intention": scenario.intention,
            }
            call_start = len(role_calls)
            capture_start = len(resolver.captures)
            ckpt = _checkpoint(config, scenario)
            before_hp = combat_hp_by_id(ckpt)
            scenario_error = ""
            event_summary: dict[str, Any] = {}
            try:
                result = await resolver.resolve_combat_action(
                    ckpt=ckpt,
                    actor_id=scenario.actor_id,
                    intention=scenario.intention,
                )
                event_summary = _event_summary(result)
            except Exception:
                scenario_error = traceback.format_exc()
            capture = (
                resolver.captures[-1]
                if len(resolver.captures) > capture_start
                else None
            )
            result_payload = {
                "index": index,
                "name": scenario.name,
                "category": scenario.category,
                "summary": scenario.summary,
                "actor_id": scenario.actor_id,
                "intention": scenario.intention,
                "expectations": scenario.expectations,
                "event": event_summary,
                "capture": _capture_dump(capture),
                "before_hp": before_hp,
                "after_hp": combat_hp_by_id(ckpt),
                "after_reactions": combat_reaction_by_id(ckpt),
                "resource_spends": cat_ii_resource_spends(ckpt),
                "role_calls": role_calls[call_start:],
                "error": scenario_error,
            }
            result_payload["findings"] = _scenario_findings(result_payload)
            scenario_results.append(result_payload)
    except Exception:
        error = traceback.format_exc()
    finally:
        await client.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(RUN_DIR),
        "roles": {
            "dnd_combat_manager": _role_label(config, "dnd_combat_manager"),
        },
        "scenario_count": len(selected),
        "raw_calls_path": str(RAW_CALLS_PATH),
        "scenarios": scenario_results,
        "role_calls": role_calls,
        "usage_totals": _usage_totals(role_calls),
        "quality_findings": _quality_findings(scenario_results),
        "cache_watch_findings": _cache_watch_findings(role_calls),
        "router_observed_facts_by_salience": (
            _router_observed_facts_by_salience(scenario_results)
        ),
        "checks": _checks(scenario_results, error),
        "error": error,
    }
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(stress_report_markdown(report), encoding="utf-8")
    return report


def _append_raw_call(payload: dict[str, Any]) -> None:
    append_jsonl(RAW_CALLS_PATH, payload)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=[],
        help=(
            "Optional scenario names. Explicit names run even if they are "
            "known-good baselines."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=("active", "known-good", "all"),
        default="active",
        help=(
            "Scenario group to run when --scenarios is omitted. Defaults to "
            "active stress cases; known-good baselines are collapsed out."
        ),
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=0,
        help="Run only the first N selected scenarios.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List scenario names and categories without calling the model.",
    )
    args = parser.parse_args()
    scenarios = _scenarios()
    if args.list_scenarios:
        for scenario in scenarios:
            print(f"{scenario.name}\t{scenario.category}")
        return
    if args.scenarios:
        wanted = set(args.scenarios)
        scenarios = [scenario for scenario in scenarios if scenario.name in wanted]
        missing = wanted - {scenario.name for scenario in scenarios}
        if missing:
            raise SystemExit("Unknown scenario(s): " + ", ".join(sorted(missing)))
    elif args.suite == "active":
        scenarios = [
            scenario for scenario in scenarios
            if scenario.category == "active"
        ]
    elif args.suite == "known-good":
        scenarios = [
            scenario for scenario in scenarios
            if scenario.category == "known_good"
        ]
    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]
    if not scenarios:
        raise SystemExit("No scenarios selected.")

    report = await _run_harness(scenarios)
    print(JSON_PATH)
    print(MD_PATH)
    print(RAW_CALLS_PATH)
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status}: {check['name']}")
    if report.get("quality_findings"):
        print(f"QUALITY_FINDINGS: {len(report['quality_findings'])}")
        for finding in report["quality_findings"]:
            print(
                f"{finding.get('severity', 'info').upper()}: "
                f"{finding.get('scenario')} / {finding.get('name')}"
            )


if __name__ == "__main__":
    asyncio.run(main())
