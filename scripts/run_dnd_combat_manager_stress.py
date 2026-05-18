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
import re
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
from app.engine.dnd_combat_resolution import DndCombatResolver
from app.engine.model_config_sync import sync_checkpoint_runtime_models
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_cat_ii import (
    DndCombatManagerAdjudication,
    RollPlan,
)
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


REPORT_DIR = REPO_ROOT / "app/storage/playtest_reports"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = REPORT_DIR / f"dnd_combat_manager_stress_{TS}"
JSON_PATH = RUN_DIR / "report.json"
MD_PATH = RUN_DIR / "report.md"
LOG_PATH = RUN_DIR / "run.log"


@dataclass
class CombatCapture:
    actor_id: str
    intention: str
    packet: str = ""
    roll_plan: dict[str, Any] = field(default_factory=dict)
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
        "dnd5e_runtime": {
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
        private_state=PrivateState(
            goals=["Resolve the isolated stress-combat action."],
            current_objectives=["Act according to the stress scenario."],
            secrets=[],
        ),
        known_context=(
            "A focused D&D 5e combat-manager stress scenario is active."
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
                ("cult_far", 11, 9),
            ],
            terrain=[
                _wall(
                    "fallen_pillar",
                    "Fallen stone pillar",
                    x=6,
                    y=3,
                    width=1,
                    height=4,
                    notes=(
                        "Total cover and blocked sight, but Fireball's spell "
                        "text says the fire spreads around corners."
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
                "must_include_save_targets": ["pc_aria", "cult_a", "cult_b"],
                "must_exclude_roll_targets": ["cult_far"],
                "expected_save_damage_spell": True,
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
                "I fan Burning Hands north-east through the low-wall gap, not "
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
                    notes="15-foot cone aimed north-east.",
                )
            ],
            expectations={
                "forbid_attack_rolls": True,
                "must_include_save_targets": ["pc_aria", "pc_bram", "ash_acolyte"],
                "must_exclude_roll_targets": ["pc_rogue"],
                "expected_save_damage_spell": True,
                "cover_should_matter_for_dex_save_targets": ["pc_bram"],
                "friendly_fire_targets": ["ash_acolyte"],
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
                "must_include_save_targets": ["slit_guard", "brute"],
                "must_exclude_roll_targets": ["pc_aria"],
                "expected_save_damage_spell": True,
                "cover_should_matter_for_dex_save_targets": ["slit_guard"],
                "target_reason_contains": {"slit_guard": ["cover"]},
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
                "expected_save_ability": "wis",
                "forbid_damage_records": True,
                "friendly_fire_targets": ["pc_aria"],
            },
            category="known_good",
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
                ("cult_edge", 10, 7),
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
                "expected_save_ability": "dex",
                "require_spatial_delta_kind": "add_area",
                "require_effect_delta_if_failed": True,
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
                "I cast Thunderwave so the orc raider is blasted west out of "
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
                        save_dc=15,
                        spells=[thunderwave],
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
                ("pc_guard", 5, 4),
            ],
            areas=[
                _area(
                    "thunderwave_cube",
                    "Thunderwave cube",
                    shape="square",
                    x=2,
                    y=3,
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
                "must_include_save_targets": ["orc_raider"],
                "expected_save_ability": "con",
                "expected_save_damage_spell": True,
                "require_spatial_delta_if_failed": "move_token",
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
    ]


async def _run_harness(selected: list[Scenario]) -> dict[str, Any]:
    load_dotenv()
    config = LLMConfig.from_env()
    missing = _preflight_api_keys(config, {"dnd_combat_manager"})
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
    resolver = CapturingDndCombatResolver(client, prompt_mgr)
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

    scenario_results: list[dict[str, Any]] = []
    error = ""
    try:
        for index, scenario in enumerate(selected, start=1):
            call_start = len(role_calls)
            capture_start = len(resolver.captures)
            ckpt = _checkpoint(config, scenario)
            before_hp = _hp_by_id(ckpt)
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
                "after_hp": _hp_by_id(ckpt),
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
        "scenarios": scenario_results,
        "role_calls": role_calls,
        "usage_totals": _usage_totals(role_calls),
        "quality_findings": _quality_findings(scenario_results),
        "checks": _checks(scenario_results, error),
        "error": error,
    }
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(_markdown(report), encoding="utf-8")
    return report


def _event_summary(event: Any) -> dict[str, Any]:
    if event is None:
        return {}
    return {
        "event_id": getattr(event, "event_id", ""),
        "event_kind": getattr(event, "event_kind", ""),
        "decision_rationale": getattr(event, "decision_rationale", ""),
        "facts": [
            fact.text
            for fact in getattr(event.canonical_event, "observable_facts", [])
        ],
    }


def _capture_dump(capture: CombatCapture | None) -> dict[str, Any]:
    if capture is None:
        return {}
    return {
        "actor_id": capture.actor_id,
        "intention": capture.intention,
        "packet": json.loads(capture.packet) if capture.packet else {},
        "roll_plan": capture.roll_plan,
        "roll_ledger": capture.roll_ledger,
        "adjudication": capture.adjudication,
    }


def _hp_by_id(ckpt: CheckpointFile) -> dict[str, int]:
    combat = ckpt.session.active_combat
    if combat is None:
        return {}
    return {
        combatant.character_id: int(combatant.hit_points_current)
        for combatant in combat.combatants
    }


def _roll_requests(result: dict[str, Any]) -> list[dict[str, Any]]:
    return (
        ((result.get("capture") or {}).get("roll_plan") or {}).get("roll_requests")
        or []
    )


def _adjudication(result: dict[str, Any]) -> dict[str, Any]:
    return (result.get("capture") or {}).get("adjudication") or {}


def _packet(result: dict[str, Any]) -> dict[str, Any]:
    return (result.get("capture") or {}).get("packet") or {}


def _scenario_findings(result: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    expectations = result.get("expectations") or {}
    requests = _roll_requests(result)
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
            str(request.get("reason") or "").lower()
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
        damage_records = _damage_records(result)
        hp_changed = result.get("before_hp") != result.get("after_hp")
        if save_targets and not damage_records and not hp_changed:
            findings.append({
                "name": "save_damage_has_no_structured_application",
                "severity": "critical",
                "detail": {
                    "save_targets": sorted(save_targets),
                    "note": (
                        "Save-damage spells should produce structured "
                        "damage_records through the engine damage path."
                    ),
                },
            })
    if expectations.get("forbid_damage_records") and _damage_records(result):
        findings.append({
            "name": "unexpected_damage_records",
            "severity": "high",
            "detail": _damage_records(result),
        })
    if expectations.get("forbid_hp_change"):
        changed = {
            cid: {
                "before": (result.get("before_hp") or {}).get(cid),
                "after": (result.get("after_hp") or {}).get(cid),
            }
            for cid in sorted(
                set((result.get("before_hp") or {}).keys())
                | set((result.get("after_hp") or {}).keys())
            )
            if (result.get("before_hp") or {}).get(cid)
            != (result.get("after_hp") or {}).get(cid)
        }
        if changed:
            findings.append({
                "name": "unexpected_hp_change",
                "severity": "high",
                "detail": changed,
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
    for rule in expectations.get("condition_fact_requires_delta") or []:
        condition = str(rule.get("condition") or "").strip().lower()
        target_id = str(rule.get("target_id") or "").strip()
        if not condition or condition not in visible_text.lower():
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
    return findings


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


def _effect_delta_contains(delta: dict[str, Any], expected: str) -> bool:
    expected = expected.strip().lower()
    if not expected:
        return False
    haystack = " ".join(
        str(delta.get(key) or "").strip().lower()
        for key in ("effect_id", "name", "slug", "source_id", "reason")
    )
    return expected in haystack


def _failed_save_targets(result: dict[str, Any]) -> set[str]:
    failed: set[str] = set()
    requests_by_roll_id = {
        str(request.get("roll_id") or ""): request
        for request in _roll_requests(result)
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


def _dict_contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value:
            return True
        return any(_dict_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_dict_contains_key(item, key) for item in value)
    return False


def _quality_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        for finding in result.get("findings") or []:
            findings.append({
                "scenario": result.get("name"),
                **finding,
            })
    return findings


def _checks(results: list[dict[str, Any]], error: str) -> list[dict[str, Any]]:
    calls = [
        call
        for result in results
        for call in result.get("role_calls") or []
        if call.get("role") == "dnd_combat_manager"
    ]
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
            any(
                int((call.get("usage") or {}).get("cache_read_input_tokens", 0) or 0)
                > 0
                for call in calls[1:]
            ),
            [
                (call.get("usage") or {}).get("cache_read_input_tokens", 0)
                for call in calls
            ],
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
        "# D&D Combat Manager Targeted Stress Harness",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Run dir: `{report['run_dir']}`",
        f"Combat manager: `{report['roles']['dnd_combat_manager']}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks") or []:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {mark}: `{check['name']}`")
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
        lines.extend(["", "Roll plan:"])
        roll_plan = (scenario.get("capture") or {}).get("roll_plan") or {}
        lines.extend([
            "```json",
            json.dumps(roll_plan, indent=2),
            "```",
            "",
        ])
    if report.get("error"):
        lines.extend(["## Error", "", "```text", report["error"], "```", ""])
    return "\n".join(lines)


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
