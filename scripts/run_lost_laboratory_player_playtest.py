#!/usr/bin/env python3
"""Live player-style CLI playtest for Lost Laboratory imported content.

This replaces the older scripted demo scaffold. It still automates CLI input so
we can get repeatable evidence, but the story seed is built as a real party of
classed level 3 adventurers and the play flow goes through the same CLIState
command handlers used by scripts/play.py.
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
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.bot.engine_bridge import EngineBridge
from app.engine.content_pack_projections import (
    apply_checkpoint_projection,
    apply_field_start_projection,
    character_record_from_projection,
    content_pack_state_from_projection,
)
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
STORY_ID = "lost_laboratory_kwalish_class_party"
PLAYER_ID = "pc_marlowe_hexblade"
START_LOCATION = "loc.barrier_peaks_route"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = REPORT_ROOT / f"lost_laboratory_player_playtest_{TS}"
JSON_PATH = RUN_DIR / "report.json"
MD_PATH = RUN_DIR / "report.md"
LOG_PATH = RUN_DIR / "run.log"

INITIAL_CLI_COMMANDS = [
    f"/story start {STORY_ID}",
    "/characters",
    f"/join {PLAYER_ID}",
    "/sheet all",
    "/begin",
    (
        "I take point on the wooded ridge, keep the route folios behind me, "
        "with Tavi's cannon covering the open angle and Ilyra watching the "
        "soil and roots as we move toward the cave approach."
    ),
]

EXPLORATION_COMMANDS: list[str] = []

AMBUSH_COMMANDS = [
    (
        "At the cleft, armed figures step from the rocks and demand the route "
        "folios. I identify the apparent leader, invoke Hexblade's Curse on "
        "them, and cast Armor of Agathys while telling Tavi to light up any "
        "cluster with Faerie Fire and Ilyra to drop Entangle if they rush us."
    ),
    (
        "If the outlaws keep advancing or draw weapons, I fire Eldritch Blast "
        "at the leader and call for Tavi's Force Ballista and Ilyra's control "
        "magic to keep them away from the folios."
    ),
]

COMBAT_COMMANDS = [
    (
        "I cast Hex on the captain if the curse has not already taken hold, "
        "then fire Eldritch Blast at the most dangerous outlaw. I call for "
        "Tavi to use Faerie Fire if enemies are clustered, or Force Ballista "
        "if they are spread out, and for Ilyra to use Entangle if she can pin "
        "them without catching us."
    ),
    (
        "I keep pressure on the captain with Eldritch Blast and make room for "
        "Ilyra to use Halo of Spores or Healing Word if someone drops. Tavi "
        "should spend her cannon shot on whichever enemy threatens the folios."
    ),
    (
        "I fall back only far enough to keep the papers out of reach, then "
        "strike with my pact weapon if an outlaw closes. I tell Tavi and Ilyra "
        "to conserve slots unless a control spell will clearly swing the fight."
    ),
]


def _resource(
    resource_id: str,
    name: str,
    *,
    current: int,
    maximum: int,
    kind: str = "class_feature",
    recovery: str = "long_rest",
) -> dict[str, Any]:
    return {
        "id": resource_id,
        "name": name,
        "kind": kind,
        "current": current,
        "max": maximum,
        "recovery": recovery,
    }


def _consume(resource_id: str, amount: int = 1) -> list[dict[str, Any]]:
    return [{"resource_id": resource_id, "amount": amount}]


def _damage(formula: str, damage_type: str) -> list[dict[str, Any]]:
    return [{"formula": formula, "damage_type": damage_type}]


def _healing(formula: str) -> list[dict[str, Any]]:
    return [{"formula": formula}]


def _action(
    action_id: str,
    name: str,
    *,
    attack_bonus: int | None = None,
    damage: str = "",
    damage_type: str = "",
    attack_range: str = "5 ft",
    activation: str = "action",
    consumes: list[dict[str, Any]] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": action_id,
        "name": name,
        "kind": "attack" if attack_bonus is not None else "feature",
        "activation": {"type": activation},
        "attack": (
            {
                "bonus": attack_bonus,
                "damage": f"{damage} {damage_type}".strip(),
                "range": attack_range,
            }
            if attack_bonus is not None else {}
        ),
        "damage": _damage(damage, damage_type) if damage and damage_type else [],
        "healing": [],
        "consumes": list(consumes or []),
        "notes": notes,
    }


def _spell(
    spell_id: str,
    name: str,
    *,
    level: int,
    ability: str = "",
    attack_bonus: int | None = None,
    save_ability: str = "",
    dc: int = 0,
    damage: str = "",
    damage_type: str = "",
    healing: str = "",
    consumes: list[dict[str, Any]] | None = None,
    concentration: bool = False,
    activation: str = "action",
    duration_text: str = "",
    range_text: str = "",
    target_text: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": spell_id,
        "name": name,
        "level": level,
        "prepared": True,
        "always_prepared": level == 0,
        "concentration": concentration,
        "activation": {"type": activation},
        "duration": {"text": duration_text} if duration_text else {},
        "range": {"text": range_text} if range_text else {},
        "target": {"text": target_text} if target_text else {},
        "attack": (
            {"ability": ability, "bonus": attack_bonus}
            if attack_bonus is not None else {}
        ),
        "save": {"ability": save_ability, "dc": dc} if save_ability else {},
        "damage": _damage(damage, damage_type) if damage and damage_type else [],
        "healing": _healing(healing) if healing else [],
        "consumes": list(consumes or []),
        "notes": notes,
    }


def _feature(name: str, *, level: int, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "class",
        "level": level,
        "description": description,
    }


def _ability_scores(scores: dict[str, int]) -> dict[str, dict[str, int]]:
    return {
        ability: {"score": score, "modifier": (score - 10) // 2}
        for ability, score in scores.items()
    }


def _dnd_mechanics(
    *,
    name: str,
    species: str,
    background: str,
    class_name: str,
    subclass: str,
    ability_scores: dict[str, int],
    armor_class: int,
    hp: int,
    initiative: int,
    skills: dict[str, int],
    saves: dict[str, int],
    actions: list[dict[str, Any]],
    features: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    spellcasting: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ruleset_id": "dnd5e_basic",
        "level": 3,
        "proficiency_bonus": 2,
        "ability_scores": ability_scores,
        "skill_proficiencies": sorted(skills),
        "saving_throw_proficiencies": sorted(saves),
        "armor_class": armor_class,
        "hit_points": {"current": hp, "max": hp, "temporary": 0},
        "conditions": [],
        "resources": resources,
        "dnd5e_sheet": {
            "ruleset_id": "dnd5e_basic",
            "identity": {
                "name": name,
                "species": species,
                "background": background,
                "classes": [{
                    "name": class_name,
                    "subclass": subclass,
                    "level": 3,
                    "spellcasting_ability": (
                        spellcasting.get("profiles", [{}])[0].get("ability", "")
                        if spellcasting.get("profiles") else ""
                    ),
                }],
            },
            "statblock": {
                "proficiency_bonus": 2,
                "ability_scores": _ability_scores(ability_scores),
                "skills": {
                    skill: {"value": value, "proficiency_multiplier": 1}
                    for skill, value in skills.items()
                },
                "saves": {
                    ability: {"value": value, "proficiency_multiplier": 1}
                    for ability, value in saves.items()
                },
                "defenses": {
                    "armor_class": {"value": armor_class},
                    "hit_points": {
                        "current": hp,
                        "max": hp,
                        "temporary": 0,
                    },
                    "initiative": {"value": initiative},
                    "movement": {"walk": {"value": 30, "unit": "ft"}},
                },
                "resources": resources,
                "features": features,
                "actions": actions,
                "spellcasting": spellcasting,
            },
        },
        "raw": {},
        "dnd5e_runtime": {"active_effects": []},
    }


def _hexblade() -> CharacterRecord:
    resources = [
        _resource("pact_slot", "Pact Magic Slot", current=2, maximum=2, kind="spell_slot", recovery="short_rest"),
        _resource("hexblades_curse", "Hexblade's Curse", current=1, maximum=1, recovery="short_rest"),
    ]
    spells = [
        _spell(
            "eldritch_blast",
            "Eldritch Blast",
            level=0,
            ability="cha",
            attack_bonus=5,
            damage="1d10",
            damage_type="force",
            range_text="120 ft",
            target_text="one creature",
        ),
        _spell(
            "hex",
            "Hex",
            level=1,
            consumes=_consume("pact_slot"),
            concentration=True,
            duration_text="Concentration, up to 1 hour",
            range_text="90 ft",
            target_text="one creature",
            notes="Marks a target and adds necrotic damage when the caster hits it.",
        ),
        _spell(
            "armor_of_agathys",
            "Armor of Agathys",
            level=1,
            consumes=_consume("pact_slot"),
            duration_text="1 hour",
            range_text="self",
            target_text="self",
            notes="Grants 5 temporary hit points at level 1 and damages melee attackers while they last.",
        ),
        _spell(
            "hellish_rebuke",
            "Hellish Rebuke",
            level=1,
            save_ability="dex",
            dc=13,
            damage="2d10",
            damage_type="fire",
            consumes=_consume("pact_slot"),
            activation="reaction",
            range_text="60 ft",
            target_text="creature that damaged you",
        ),
    ]
    mechanics = _dnd_mechanics(
        name="Marlowe Vex",
        species="Tiefling",
        background="Failed cartographer",
        class_name="Warlock",
        subclass="Hexblade",
        ability_scores={"str": 8, "dex": 14, "con": 14, "int": 12, "wis": 10, "cha": 16},
        armor_class=15,
        hp=24,
        initiative=2,
        skills={"arcana": 3, "deception": 5, "investigation": 3, "survival": 2},
        saves={"wis": 2, "cha": 5},
        actions=[
            _action(
                "pact_rapier",
                "Pact Rapier",
                attack_bonus=5,
                damage="1d8+3",
                damage_type="piercing",
                notes="Hex Warrior weapon. Use Charisma for attack and damage.",
            ),
            _action(
                "hexblades_curse",
                "Hexblade's Curse",
                activation="bonus_action",
                consumes=_consume("hexblades_curse"),
                notes="Mark one target for 1 minute. Attacks against it crit on 19-20 and add proficiency bonus to damage.",
            ),
        ],
        features=[
            _feature(
                "Hex Warrior",
                level=1,
                description="Uses Charisma for the pact rapier's attack and damage.",
            ),
            _feature(
                "Hexblade's Curse",
                level=1,
                description="Bonus action mark once per short rest; adds pressure to one priority target.",
            ),
            _feature(
                "Pact Magic",
                level=1,
                description="Two pact slots, both refreshed on a short rest.",
            ),
        ],
        resources=resources,
        spellcasting={
            "profiles": [{
                "id": "warlock_pact_magic",
                "name": "Pact Magic",
                "ability": "cha",
                "spell_attack_bonus": 5,
                "spell_save_dc": 13,
            }],
            "pact_slots": {"level": 2, "current": 2, "max": 2},
            "spells": spells,
        },
    )
    return CharacterRecord(
        character_id=PLAYER_ID,
        name="Marlowe Vex",
        status=CharacterStatus.active,
        location=START_LOCATION,
        is_playable=True,
        agent_tier=CharacterAgentTier.standard,
        public_sheet=PublicSheet(
            role="level 3 Hexblade Warlock",
            appearance="A tiefling pathfinder in a dark travel coat, with a black-iron rapier and annotated route folios.",
            faction="Lost Laboratory expedition",
        ),
        private_state=PrivateState(
            goals=[
                "Keep the expedition alive long enough to reach the laboratory.",
                "Prove the curse-bound pact can be used with discipline rather than panic.",
            ],
            current_objectives=[
                "Protect the route folios.",
                "Identify the leader in any ambush and lock them down first.",
            ],
            intentions_enabled=True,
        ),
        backstory=(
            "Marlowe was hired after correcting two contradictions in the "
            "Cartophile's route notes. The pact is useful, but she treats it "
            "as a dangerous tool rather than a personality."
        ),
        personality=(
            "Precise, dry, and tactical. Speaks in short field instructions. "
            "Uses magic deliberately and worries about overcommitting pact slots."
        ),
        known_context=(
            "You accepted the Cartophile's return-documentation terms and are "
            "already on the Barrier Peaks route. Tavi and Ilyra are with you as "
            "full expedition members. Garret and Gearbox may be useful contacts, "
            "but this playtest party is responsible for the field decisions."
        ),
        mechanics=mechanics,
    )


def _artillerist() -> CharacterRecord:
    resources = [
        _resource("spell_slot_1", "Level 1 Spell Slot", current=3, maximum=3, kind="spell_slot"),
        _resource("eldritch_cannon_use", "Eldritch Cannon", current=1, maximum=1, recovery="long_rest"),
    ]
    spells = [
        _spell(
            "thorn_whip",
            "Thorn Whip",
            level=0,
            ability="int",
            attack_bonus=5,
            damage="1d6",
            damage_type="piercing",
            range_text="30 ft",
            target_text="one creature",
            notes="Can pull a target 10 feet on a hit.",
        ),
        _spell(
            "faerie_fire",
            "Faerie Fire",
            level=1,
            save_ability="dex",
            dc=13,
            consumes=_consume("spell_slot_1"),
            concentration=True,
            range_text="60 ft",
            target_text="20-foot cube",
            notes="Creatures that fail glow; attacks against them have advantage.",
        ),
        _spell(
            "catapult",
            "Catapult",
            level=1,
            save_ability="dex",
            dc=13,
            damage="3d8",
            damage_type="bludgeoning",
            consumes=_consume("spell_slot_1"),
            range_text="60 ft",
            target_text="one object hurled in a line",
        ),
        _spell(
            "cure_wounds",
            "Cure Wounds",
            level=1,
            healing="1d8+3",
            consumes=_consume("spell_slot_1"),
            range_text="touch",
            target_text="one creature",
        ),
    ]
    mechanics = _dnd_mechanics(
        name="Tavi Brasswake",
        species="Rock Gnome",
        background="Field sapper",
        class_name="Artificer",
        subclass="Artillerist",
        ability_scores={"str": 8, "dex": 14, "con": 14, "int": 16, "wis": 10, "cha": 10},
        armor_class=15,
        hp=24,
        initiative=2,
        skills={"arcana": 5, "investigation": 5, "perception": 2, "sleight_of_hand": 4},
        saves={"con": 4, "int": 5},
        actions=[
            _action(
                "light_crossbow",
                "Light Crossbow",
                attack_bonus=4,
                damage="1d8+2",
                damage_type="piercing",
                attack_range="80/320 ft",
            ),
            _action(
                "create_eldritch_cannon",
                "Create Eldritch Cannon",
                activation="action",
                consumes=_consume("eldritch_cannon_use"),
                notes="Creates a small magical cannon in an unoccupied space within 5 feet.",
            ),
            _action(
                "force_ballista",
                "Force Ballista",
                attack_bonus=5,
                damage="2d8",
                damage_type="force",
                attack_range="120 ft",
                activation="bonus_action",
                notes="Eldritch Cannon option. On a hit, the target is pushed 5 feet away from the cannon.",
            ),
        ],
        features=[
            _feature(
                "Infuse Item",
                level=2,
                description="Maintains practical field infusions rather than treasure-hunt wish lists.",
            ),
            _feature(
                "Eldritch Cannon",
                level=3,
                description="Can create and command a cannon; Force Ballista is the preferred route-security mode.",
            ),
            _feature(
                "Tool Expertise",
                level=3,
                description="Reads mechanisms, cave hardware, and improvised traps quickly.",
            ),
        ],
        resources=resources,
        spellcasting={
            "profiles": [{
                "id": "artificer_spellcasting",
                "name": "Artificer Spellcasting",
                "ability": "int",
                "spell_attack_bonus": 5,
                "spell_save_dc": 13,
            }],
            "slots": {"1": {"current": 3, "max": 3}},
            "spells": spells,
        },
    )
    return CharacterRecord(
        character_id="pc_tavi_artillerist",
        name="Tavi Brasswake",
        status=CharacterStatus.active,
        location=START_LOCATION,
        is_playable=True,
        agent_tier=CharacterAgentTier.standard,
        public_sheet=PublicSheet(
            role="level 3 Artillerist Artificer",
            appearance="A rock gnome engineer with a folding tripod cannon, blue lens goggles, and packed survey tools.",
            faction="Lost Laboratory expedition",
        ),
        private_state=PrivateState(
            goals=[
                "Keep the expedition equipment working under pressure.",
                "Get proof that the lost technology is reproducible, not just legendary.",
            ],
            current_objectives=[
                "Protect the route folios without wasting the cannon.",
                "Use control magic before damage when terrain favors it.",
            ],
            intentions_enabled=True,
        ),
        backstory=(
            "Tavi joined for access to Kwalish's technical trail. She is "
            "competent enough to be cautious and proud enough to test the "
            "cannon when the table gives her a clean angle."
        ),
        personality=(
            "Brisk, mechanically specific, and impatient with vague danger. "
            "She names tools, angles, and failure modes out loud."
        ),
        known_context=(
            "You are on the Barrier Peaks route with Marlowe and Ilyra. The "
            "Cartophile's documents are valuable, vulnerable, and technically "
            "interesting. Your cannon and control spells are expedition assets, "
            "not fireworks."
        ),
        mechanics=mechanics,
    )


def _spores_druid() -> CharacterRecord:
    resources = [
        _resource("spell_slot_1", "Level 1 Spell Slot", current=4, maximum=4, kind="spell_slot"),
        _resource("spell_slot_2", "Level 2 Spell Slot", current=2, maximum=2, kind="spell_slot"),
        _resource("wild_shape", "Wild Shape", current=2, maximum=2, recovery="short_rest"),
    ]
    spells = [
        _spell(
            "produce_flame",
            "Produce Flame",
            level=0,
            ability="wis",
            attack_bonus=5,
            damage="1d8",
            damage_type="fire",
            range_text="30 ft",
            target_text="one creature",
        ),
        _spell(
            "entangle",
            "Entangle",
            level=1,
            save_ability="str",
            dc=13,
            consumes=_consume("spell_slot_1"),
            concentration=True,
            range_text="90 ft",
            target_text="20-foot square",
            notes="Restrains creatures that fail their Strength save.",
        ),
        _spell(
            "healing_word",
            "Healing Word",
            level=1,
            healing="1d4+3",
            consumes=_consume("spell_slot_1"),
            activation="bonus_action",
            range_text="60 ft",
            target_text="one creature",
        ),
        _spell(
            "moonbeam",
            "Moonbeam",
            level=2,
            save_ability="con",
            dc=13,
            damage="2d10",
            damage_type="radiant",
            consumes=_consume("spell_slot_2"),
            concentration=True,
            range_text="120 ft",
            target_text="5-foot-radius cylinder",
        ),
    ]
    mechanics = _dnd_mechanics(
        name="Ilyra Mosswake",
        species="Half-Elf",
        background="Fungal surveyor",
        class_name="Druid",
        subclass="Circle of Spores",
        ability_scores={"str": 8, "dex": 14, "con": 14, "int": 11, "wis": 16, "cha": 10},
        armor_class=14,
        hp=24,
        initiative=2,
        skills={"medicine": 5, "nature": 2, "perception": 5, "survival": 5},
        saves={"int": 2, "wis": 5},
        actions=[
            _action(
                "quarterstaff",
                "Quarterstaff",
                attack_bonus=1,
                damage="1d6-1",
                damage_type="bludgeoning",
            ),
            _action(
                "symbiotic_entity",
                "Symbiotic Entity",
                activation="action",
                consumes=_consume("wild_shape"),
                notes="Spend Wild Shape to gain temporary hit points and empower spore damage for 10 minutes.",
            ),
            _action(
                "halo_of_spores",
                "Halo of Spores",
                activation="reaction",
                damage="1d4",
                damage_type="necrotic",
                notes="Reaction against a creature that starts its turn within 10 feet; target makes a Con save.",
            ),
        ],
        features=[
            _feature(
                "Halo of Spores",
                level=2,
                description="Reaction damage that rewards careful positioning.",
            ),
            _feature(
                "Symbiotic Entity",
                level=2,
                description="Uses Wild Shape as a combat stance instead of animal form.",
            ),
            _feature(
                "Circle Spells",
                level=3,
                description="Keeps fungal and control magic prepared for strange terrain.",
            ),
        ],
        resources=resources,
        spellcasting={
            "profiles": [{
                "id": "druid_spellcasting",
                "name": "Druid Spellcasting",
                "ability": "wis",
                "spell_attack_bonus": 5,
                "spell_save_dc": 13,
            }],
            "slots": {
                "1": {"current": 4, "max": 4},
                "2": {"current": 2, "max": 2},
            },
            "spells": spells,
        },
    )
    return CharacterRecord(
        character_id="pc_ilyra_spores",
        name="Ilyra Mosswake",
        status=CharacterStatus.active,
        location=START_LOCATION,
        is_playable=True,
        agent_tier=CharacterAgentTier.standard,
        public_sheet=PublicSheet(
            role="level 3 Circle of Spores Druid",
            appearance="A half-elf trail medic with spore-stained leather, seed charms, and a blackwood staff.",
            faction="Lost Laboratory expedition",
        ),
        private_state=PrivateState(
            goals=[
                "Read the living terrain before the expedition damages it.",
                "Keep the party alive without making every problem a wound.",
            ],
            current_objectives=[
                "Use Entangle or Moonbeam only when the line of effect is clean.",
                "Hold Healing Word for real emergencies.",
            ],
            intentions_enabled=True,
        ),
        backstory=(
            "Ilyra joined because the route's ecological oddities do not match "
            "ordinary mountain growth. She trusts fungi, tracks, and quiet signs "
            "more than people who call every ruin a treasure."
        ),
        personality=(
            "Calm, sensory, and unsentimental. She talks about spores and roots "
            "as practical witnesses."
        ),
        known_context=(
            "You are on the Barrier Peaks route with Marlowe and Tavi. Your job "
            "is to read living signs, prevent needless harm, and use control "
            "magic when it saves blood."
        ),
        mechanics=mechanics,
    )


def _class_party() -> list[CharacterRecord]:
    return [_hexblade(), _artillerist(), _spores_druid()]


def _park_imported_support_npcs(ckpt: CheckpointFile) -> None:
    for character in ckpt.characters:
        if character.character_id not in {"npc_garret", "npc_gearbox"}:
            continue
        character.status = CharacterStatus.dormant
        character.location = "loc.cartophile_collection"
        character.private_state.intentions_enabled = False
        character.known_context = (
            character.known_context.rstrip()
            + "\n\nFor this class-party playtest seed, you remained behind at "
            "the Cartophile's collection and are not present on the route."
        )


def _remove_imported_support_field_facts(ckpt: CheckpointFile) -> None:
    omitted = {
        "Garret is custodian of the route folios.",
        "Gearbox accompanies the expedition as technical support.",
    }
    ckpt.world_state.facts = [
        fact for fact in ckpt.world_state.facts if fact not in omitted
    ]


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
        mechanics={},
        agent_tier=CharacterAgentTier.standard,
    )


def _story_checkpoint() -> CheckpointFile:
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
                start_mode="field",
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
                    "A classed level 3 party follows incomplete route lore toward "
                    "a lost inventor's laboratory."
                ),
            ),
            lore=projection.checkpoint.world_lore,
        ),
        characters=npcs,
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    ckpt.session.config.settings.player_roll_mode = "auto"
    ckpt.session.config.models.agent_default = "claude-haiku-4-5"
    ckpt.session.config.models.agent_standard = "claude-haiku-4-5"
    ckpt.session.config.models.agent_convenience = "claude-haiku-4-5"
    apply_checkpoint_projection(ckpt, projection.checkpoint)
    apply_field_start_projection(ckpt, projection)
    _park_imported_support_npcs(ckpt)
    ckpt.characters.extend(_class_party())
    _remove_imported_support_field_facts(ckpt)
    field_primer = ckpt.player_primer.replace(
        " with Garret as map custodian and Gearbox as technical support",
        "",
    )
    ckpt.player_primer = (
        field_primer
        + "\n\nYou are testing a prebuilt level 3 D&D party already on the "
        "route. Pick one character, use normal in-character actions, and let "
        "the other party members respond as agents unless claimed."
    )
    ckpt.world_state.facts.extend([
        (
            "The active playtest party is Marlowe Vex, Tavi Brasswake, and "
            "Ilyra Mosswake: three level 3 adventurers already retained for "
            "the Lost Laboratory expedition."
        ),
        (
            "The classed party accepted the Cartophile's return-documentation "
            "terms and reached the wooded Barrier Peaks route with the route "
            "folios."
        ),
        (
            "Garret Levistusson and Gearbox remained behind at the Cartophile's "
            "collection for this class-party playtest; they are not present at "
            "the route cleft."
        ),
    ])
    ckpt.session.config.narrative_rules = (
        ckpt.session.config.narrative_rules
        + "\nFor this playtest seed, treat only Marlowe Vex, Tavi Brasswake, "
        "and Ilyra Mosswake as the active field party at the route cleft. "
        "Do not give Garret or Gearbox field turns unless the player sends "
        "for them from the Cartophile's collection."
    ).strip()
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


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def _role_label(config: LLMConfig, role: str) -> str:
    return f"{config.provider_for_role(role)}:{config.model_for_role(role)}"


def _class_summary(ckpt: CheckpointFile | None) -> list[dict[str, Any]]:
    if ckpt is None:
        return []
    summaries: list[dict[str, Any]] = []
    for character in ckpt.characters:
        if not character.character_id.startswith("pc_"):
            continue
        sheet = character.mechanics.get("dnd5e_sheet") or {}
        identity = sheet.get("identity") or {}
        statblock = sheet.get("statblock") or {}
        spellcasting = statblock.get("spellcasting") or {}
        summaries.append({
            "character_id": character.character_id,
            "name": character.name,
            "classes": identity.get("classes") or [],
            "resources": [
                item.get("id", "")
                for item in statblock.get("resources") or []
                if isinstance(item, dict)
            ],
            "actions": [
                item.get("id", "")
                for item in statblock.get("actions") or []
                if isinstance(item, dict)
            ],
            "spells": [
                item.get("id", "")
                for item in spellcasting.get("spells") or []
                if isinstance(item, dict)
            ],
        })
    return summaries


def _resource_state(ckpt: CheckpointFile | None) -> dict[str, Any]:
    if ckpt is None:
        return {}
    state: dict[str, Any] = {}
    for character in ckpt.characters:
        if not character.character_id.startswith("pc_"):
            continue
        sheet = character.mechanics.get("dnd5e_sheet") or {}
        statblock = sheet.get("statblock") or {}
        spellcasting = statblock.get("spellcasting") or {}
        state[character.character_id] = {
            "mechanics_resources": character.mechanics.get("resources") or [],
            "spell_slots": spellcasting.get("slots") or {},
            "pact_slots": spellcasting.get("pact_slots") or {},
            "conditions": character.mechanics.get("conditions") or [],
            "hp": character.mechanics.get("hit_points") or {},
        }
    return state


def _combat_state_summary(ckpt: CheckpointFile | None) -> dict[str, Any]:
    combat = getattr(getattr(ckpt, "session", None), "active_combat", None)
    if combat is None:
        return {"active": False}
    return {
        "active": True,
        "round": getattr(combat, "round", 0),
        "current_turn": getattr(combat, "current_turn", 0),
        "combatants": [
            {
                "id": getattr(combatant, "combatant_id", ""),
                "character_id": getattr(combatant, "character_id", ""),
                "name": getattr(combatant, "name", ""),
                "hp": getattr(combatant, "hit_points_current", 0),
                "max_hp": getattr(combatant, "hit_points_max", 0),
                "defeat_state": getattr(combatant, "defeat_state", ""),
                "conditions": list(getattr(combatant, "conditions", []) or []),
            }
            for combatant in getattr(combat, "combatants", []) or []
        ],
    }


def _usage_totals(role_calls: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {}
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
    for call in role_calls:
        role = str(call.get("role") or "unknown")
        usage = call.get("usage") or {}
        bucket = totals.setdefault(role, {key: 0 for key in keys})
        for key in keys:
            try:
                bucket[key] += int(usage.get(key, 0) or 0)
            except (TypeError, ValueError):
                pass
    return totals


def _log_warning_lines() -> list[str]:
    if not LOG_PATH.exists():
        return []
    lines: list[str] = []
    for line in LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if "WARNING" in line or "ERROR" in line:
            lines.append(line)
    return lines[-80:]


def _next_command(
    ckpt: CheckpointFile | None,
    state: CLIState,
    index: int,
) -> str:
    if state._joined_pending_roll_prompts():
        return "/roll all"
    combat = getattr(getattr(ckpt, "session", None), "active_combat", None)
    if combat is not None:
        return COMBAT_COMMANDS[index % len(COMBAT_COMMANDS)]
    if index < len(EXPLORATION_COMMANDS):
        return EXPLORATION_COMMANDS[index]
    return AMBUSH_COMMANDS[(index - len(EXPLORATION_COMMANDS)) % len(AMBUSH_COMMANDS)]


async def _run_playtest(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    os.environ.setdefault("LLM_MODEL_AGENT", "claude-haiku-4-5")
    os.environ.setdefault("LLM_MODEL_AGENT_STANDARD", "claude-haiku-4-5")
    os.environ.setdefault("LLM_MODEL_AGENT_CONVENIENCE", "claude-haiku-4-5")
    os.environ.setdefault("LLM_MODEL_CHARACTER_GEN", "claude-haiku-4-5")

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
        messages = kwargs.get("messages") or []
        prompt_text = _message_text(messages)
        record: dict[str, Any] = {
            "role": role,
            "message_count": len(messages),
            "prompt_chars": len(prompt_text),
            "approx_prompt_tokens": _approx_tokens(prompt_text),
            "contains_private_path": "private_extractions/" in prompt_text,
            "contains_class_party": (
                "Marlowe Vex" in prompt_text
                or "Tavi Brasswake" in prompt_text
                or "Ilyra Mosswake" in prompt_text
            ),
            "contains_spell_payload": any(
                marker in prompt_text
                for marker in (
                    "Hexblade's Curse",
                    "Eldritch Cannon",
                    "Symbiotic Entity",
                    "Pact Magic",
                    "Moonbeam",
                    "Faerie Fire",
                )
            ),
        }
        try:
            response = await real_complete(*call_args, **kwargs)
        except Exception:
            record["error"] = traceback.format_exc()
            role_calls.append(record)
            raise
        record["model"] = response.model
        record["usage"] = dict(response.usage or {})
        parsed = getattr(response, "parsed", None)
        if role == "event_router" and parsed is not None:
            record["router_output"] = {
                "interaction_mode": getattr(parsed, "interaction_mode", ""),
                "agent_responder_picks": list(
                    getattr(parsed, "agent_responder_picks", []) or []
                ),
                "decision_rationale": getattr(parsed, "decision_rationale", ""),
            }
        role_calls.append(record)
        return response

    engine.client.complete = _recording_complete  # type: ignore[method-assign]

    session_id = args.session or f"{STORY_ID}_{TS.lower()}"
    transcript: list[dict[str, str]] = []
    try:
        engine.create_empty_session(session_id)
        state = CLIState(engine, session_id, "")
        for line in INITIAL_CLI_COMMANDS:
            output = await _run_cli_line(state, line)
            transcript.append({"input": line, "output": output})
            if "error:" in output.lower():
                break

        dynamic_index = 0
        while (
            len(transcript) < args.max_commands
            and (not transcript or "error:" not in transcript[-1]["output"].lower())
        ):
            ckpt = engine.load_latest(session_id)
            line = _next_command(ckpt, state, dynamic_index)
            dynamic_index += 1
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
    cli_errors = [
        item for item in transcript if "error:" in item["output"].lower()
    ]
    agent_calls = [
        call for call in role_calls
        if call["role"] in {"agent", "agent_standard", "agent_convenience"}
    ]
    router_calls = [call for call in role_calls if call["role"] == "event_router"]
    content_calls = [call for call in role_calls if call["role"] == "content_manager"]
    combat_manager_calls = [
        call for call in role_calls if call["role"] == "dnd_combat_manager"
    ]
    class_summaries = _class_summary(checkpoint)
    transcript_text = "\n".join(
        f"$ {item['input']}\n{item['output']}" for item in transcript
    )
    turn_index = (
        int(getattr(checkpoint.session, "turn_index", 0) or 0)
        if checkpoint is not None else 0
    )
    checks = [
        _check("story_seed_validated", story_path.exists(), str(story_path)),
        _check("cli_completed_without_error_output", not cli_errors, cli_errors),
        _check(
            "classed_level_3_party_seeded",
            len(class_summaries) == 3
            and all(
                item["classes"]
                and item["classes"][0].get("level") == 3
                for item in class_summaries
            ),
            class_summaries,
        ),
        _check(
            "less_common_mechanics_visible_in_sheet_or_prompts",
            all(
                marker in transcript_text
                or any(call.get("contains_spell_payload") for call in role_calls)
                for marker in (
                    "Hexblade",
                    "Artillerist",
                    "Circle of Spores",
                )
            ),
            {
                "class_summaries": class_summaries,
                "prompt_hits": [
                    call for call in role_calls if call.get("contains_spell_payload")
                ][:5],
            },
        ),
        _check("content_manager_called", bool(content_calls), len(content_calls)),
        _check("event_router_called", bool(router_calls), len(router_calls)),
        _check(
            "character_agents_used_haiku_role",
            bool(agent_calls)
            and all(call["role"] != "agent" for call in agent_calls)
            and all(
                "haiku" in str(call.get("model", "")).lower()
                for call in agent_calls
            ),
            agent_calls,
        ),
        _check(
            "no_private_pack_paths_in_model_prompts",
            all(not call.get("contains_private_path") for call in role_calls),
            [call for call in role_calls if call.get("contains_private_path")],
        ),
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "session_id": session_id,
        "story_id": STORY_ID,
        "story_path": str(story_path.relative_to(REPO_ROOT)),
        "installed_story": bool(args.install_story),
        "max_commands": args.max_commands,
        "command_count": len(transcript),
        "roles": {
            role: _role_label(config, role)
            for role in (
                "content_manager",
                "event_router",
                "narrator",
                "dnd_combat_manager",
                "agent",
                "agent_standard",
                "agent_convenience",
            )
        },
        "turn_index": turn_index,
        "class_party": class_summaries,
        "resource_state": _resource_state(checkpoint),
        "combat": _combat_state_summary(checkpoint),
        "transcript": transcript,
        "role_calls": role_calls,
        "call_counts": {
            "content_manager": len(content_calls),
            "event_router": len(router_calls),
            "dnd_combat_manager": len(combat_manager_calls),
            "agent": len(agent_calls),
            "role_calls_total": len(role_calls),
        },
        "usage_totals": _usage_totals(role_calls),
        "log_warnings": _log_warning_lines(),
        "checks": checks,
        "error": error,
    }


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _markdown(report: dict[str, Any]) -> str:
    passed = sum(1 for check in report["checks"] if check["passed"])
    total = len(report["checks"])
    lines = [
        "# Lost Laboratory Player Playtest Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Run directory: `{report['run_dir']}`",
        f"Story: `{report['story_id']}`",
        f"Session: `{report['session_id']}`",
        f"Commands: `{report['command_count']}`",
        f"Turn index: `{report['turn_index']}`",
        f"Checks: `{passed}/{total}`",
        "",
        "## Party",
    ]
    for character in report["class_party"]:
        classes = ", ".join(
            " ".join(
                str(part)
                for part in (
                    entry.get("subclass") or "",
                    entry.get("name") or "",
                    entry.get("level") or "",
                )
                if str(part)
            )
            for entry in character["classes"]
        )
        lines.append(f"- `{character['character_id']}`: {classes}")
        lines.append(f"  - resources: {', '.join(character['resources'])}")
        lines.append(f"  - actions: {', '.join(character['actions'])}")
        lines.append(f"  - spells: {', '.join(character['spells'])}")

    lines.extend(["", "## Checks"])
    for check in report["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {marker} `{check['name']}`")

    lines.extend(["", "## Calls"])
    for role, count in report["call_counts"].items():
        lines.append(f"- `{role}`: {count}")

    lines.extend(["", "## Transcript"])
    for item in report["transcript"]:
        lines.append(f"### `$ {item['input']}`")
        output = item["output"].strip()
        lines.append(output if output else "(no output)")
        lines.append("")

    if report["error"]:
        lines.extend(["", "## Error", "```text", report["error"], "```"])
    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a classed Lost Laboratory party seed and run a live CLI "
            "playtest through the normal player command surface."
        )
    )
    parser.add_argument(
        "--max-commands",
        type=int,
        default=9,
        help="Maximum CLI commands to execute, including setup commands.",
    )
    parser.add_argument("--session", default="", help="Optional session id.")
    parser.add_argument(
        "--install-story",
        action="store_true",
        help=(
            "Also install the generated seed under app/storage/stories so it "
            "appears in normal /story list runs."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = asyncio.run(_run_playtest(args))
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {JSON_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {MD_PATH.relative_to(REPO_ROOT)}")
    failed = [check for check in report["checks"] if not check["passed"]]
    if report["error"]:
        print("Playtest errored; see report for traceback.")
        return 1
    if failed:
        print("Failed checks:")
        for check in failed:
            print(f"- {check['name']}")
        return 2
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
