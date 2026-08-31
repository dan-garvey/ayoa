#!/usr/bin/env python3
# ruff: noqa: E402 -- executable script adds the repository root to sys.path.
"""Run the frozen Mirelle depth-by-instruction dialogue factorial.

The runner is intentionally a thin adapter around the existing character
dialogue benchmark's request capture and production ``CharacterAgent``.  It
materializes the frozen manifest into ordinary ``BenchmarkCase``/``SceneSpec``
values, replaces the normalized treatment in memory, and writes opaque
coding-agent proxy handoffs.  It never edits production prompts, stories, or
profiles, and never makes provider-live calls.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.engine.character_agent import CharacterAgent, CharacterAgentOutputError
from app.engine.prompt_manager import PromptManager
from app.schemas.characters import ActorRecord, CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import ModelConfig, SessionConfig, SessionSettings, SessionState
from scripts.run_character_dialogue_benchmark import (
    BenchmarkCase,
    BenchmarkRequest,
    ModelCall,
    SceneSpec,
    _BenchmarkCallContext,
    _RecordingCharacterClient,
    _render_public_updates as _benchmark_render_public_updates,
    _raw_response_payload,
    _sha256_json,
    _sha256_text,
)


FACTORIAL_SCHEMA_VERSION = "mirelle_dialogue_factorial_manifest_v1"
LEDGER_SCHEMA_VERSION = "opaque_conversation_response_ledger_v1"
PENDING_SCHEMA_VERSION = "opaque_conversation_pending_request_v1"
ARTIFACT_SCHEMA_VERSION = "dialogue_factorial_artifact_v1"
REVIEW_SCHEMA_VERSION = "whole_conversation_blinded_review_v1"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "scripts" / "mirelle_dialogue_factorial_manifest.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "app"
    / "storage"
    / "playtest_reports"
    / "character-dialogue-benchmark"
    / "mirelle-dialogue-factorial"
)
MIRELLE_ID = "mirelle_voss"
PILOT_SCENARIO_IDS = ("seventh_stone_open_gap", "lost_authority")
PHASES = ("pilot", "confirmatory", "all")
PREREGISTERED_REPLICATES = {"pilot": 1, "confirmatory": 4}

POST_UNBLIND_AUDIT_FIELDS = ("unsupported_invented_history",)


class FactorialManifestError(ValueError):
    """The frozen factorial manifest is malformed or no longer reproducible."""


class FactorialTechnicalInvalidity(RuntimeError):
    """A harness, privacy, topology, or hash contract was violated."""


class ModelFailure(RuntimeError):
    """A provider/proxy response failed independently of the harness contract."""


class PendingRequest(RuntimeError):
    """The proxy ledger has no response for the next exact request."""

    def __init__(self, request: BenchmarkRequest, path: Path, sequence: int):
        self.request = request
        self.path = path
        self.sequence = sequence
        super().__init__(
            f"missing proxy response {sequence}; pending request written to {path}"
        )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _json_safe(dump(mode="json"))
        except TypeError:
            return _json_safe(dump())
    return repr(value)


def _clean(value: Any, label: str, *, allow_empty: bool = False) -> str:
    result = str(value or "").strip()
    if not result and not allow_empty:
        raise FactorialManifestError(f"{label} cannot be blank")
    if any(ord(char) < 32 and char not in "\n\t" for char in result):
        raise FactorialManifestError(f"{label} contains a control character")
    return result


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorialManifestError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FactorialManifestError(f"{label} must be a list")
    return value


@dataclass(frozen=True)
class FactorialScenario:
    scenario_id: str
    scenes: tuple[SceneSpec, ...]
    depth_relevance: str = ""
    scene_2_elapsed_s: int = 0

    @property
    def turn_order(self) -> tuple[str, ...]:
        return tuple(actor_id for scene in self.scenes for actor_id in scene.turn_order)

    def topology(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "depth_relevance": self.depth_relevance,
            "scene_2_elapsed_s": self.scene_2_elapsed_s,
            "scenes": [
                {
                    "scene_id": scene.scene_id,
                    "frame": scene.frame,
                    "between_scene_public_history": scene.between_scene_public_history,
                    "turn_order": list(scene.turn_order),
                    "actor_observations": {
                        actor_id: list(values)
                        for actor_id, values in sorted(scene.actor_observations.items())
                    },
                }
                for scene in self.scenes
            ],
        }


@dataclass(frozen=True)
class FactorialCell:
    cell_id: str
    depth: str
    instruction: str
    profile_condition: str
    prompt_condition: str
    case: BenchmarkCase
    checkpoint_template: CheckpointFile
    resolved_system_sha256: str
    source_provenance: Mapping[str, Any]
    normalized_boundaries: str = ""
    scenario_specs: tuple[FactorialScenario, ...] = ()

    def actor(self, actor_id: str) -> CharacterRecord:
        return self.case.actor(actor_id)

    @property
    def scenarios(self) -> tuple[FactorialScenario, ...]:
        if self.scenario_specs:
            return self.scenario_specs
        return tuple(
            FactorialScenario(
                scenario_id=scene_group[0].scene_id.removesuffix("/scene_1"),
                scenes=scene_group,
            )
            for scene_group in _group_case_scenes(self.case.scenes)
        )


@dataclass(frozen=True)
class FactorialManifest:
    manifest_sha256: str
    cells: tuple[FactorialCell, ...]
    prompt_sha256: str
    resolved_prompt_sha256: str
    seed_path: str
    source_provenance: Mapping[str, Any]
    scenarios: tuple[FactorialScenario, ...] = ()
    blind_review_contract: Mapping[str, Any] = ()
    decision_rule: Mapping[str, Any] = ()

    def cell(self, depth: str, instruction: str) -> FactorialCell:
        for cell in self.cells:
            if (cell.depth, cell.instruction) == (depth, instruction):
                return cell
        raise FactorialManifestError(f"missing cell {depth}/{instruction}")


@dataclass
class FactorialConversation:
    conversation_id: str
    conversation_token: str
    cell: FactorialCell
    scenario: FactorialScenario
    model: str
    checkpoint: CheckpointFile
    initial_checkpoint_sha256: str
    turns: list[dict[str, Any]]
    public_transcript: list[dict[str, Any]]
    phase: str = "confirmatory"
    proxy_agent_id: str = ""
    status: str = "valid"
    technical_invalidity: str = ""
    model_failure: str = ""

    def artifact(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "conversation_id": self.conversation_id,
            "opaque_conversation": self.conversation_token,
            "cell_id": self.cell.cell_id,
            "profile_condition": self.cell.profile_condition,
            "prompt_condition": self.cell.prompt_condition,
            "depth": self.cell.depth,
            "instruction": self.cell.instruction,
            "execution": {
                "mode": "coding_agent_proxy",
                "model": "gpt-5.6-luna",
                "provider_live_calls": False,
                "proxy_agent_id": self.proxy_agent_id,
            },
            "scenario": self.scenario.topology(),
            "phase": self.phase,
            "model": self.model,
            "status": self.status,
            "technical_invalidity": self.technical_invalidity,
            "model_failure": self.model_failure,
            "checkpoint": {
                "initial_sha256": self.initial_checkpoint_sha256,
                "final_sha256": _sha256_json(self.checkpoint.model_dump(mode="json")),
                "final": self.checkpoint.model_dump(mode="json"),
            },
            "public_transcript": copy.deepcopy(self.public_transcript),
            "turns": self.turns,
            "final_history_sha256": {
                actor_id: _sha256_json(
                    [message.model_dump(mode="json") for message in messages]
                )
                for actor_id, messages in sorted(
                    self.checkpoint.character_conversations.items()
                )
            },
        }


def _parse_actor_record(raw: Any, label: str) -> ActorRecord:
    value = dict(_object(raw, label))
    facts = _array(value.get("facts", []), f"{label}.facts")
    normalized_facts = []
    for index, fact in enumerate(facts):
        if isinstance(fact, str):
            normalized_facts.append({
                "origin": "lived",
                "text": _clean(fact, f"{label}.facts[{index}]")
            })
        else:
            normalized_facts.append(dict(_object(fact, f"{label}.facts[{index}]")))
    value["facts"] = normalized_facts
    try:
        return ActorRecord.model_validate(value)
    except ValidationError as error:
        raise FactorialManifestError(f"{label} is not an ActorRecord: {error}") from error


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    return repo_path if repo_path.exists() else manifest_path.parent / path


def _prompt_material(root: Mapping[str, Any]) -> tuple[str, str, str, str]:
    conditions = _object(root.get("prompt_conditions"), "prompt_conditions")
    lean = _object(conditions.get("lean_current"), "prompt_conditions.lean_current")
    legacy = _object(
        conditions.get("normalized_legacy_behavior"),
        "prompt_conditions.normalized_legacy_behavior",
    )
    replacement = _object(legacy.get("replacement"), "normalized legacy replacement")
    return (
        _clean(lean.get("template_sha256"), "lean template SHA256"),
        _clean(lean.get("template_path", "app/prompts/agent_turn.txt"), "lean template path"),
        _clean(replacement.get("normalized_boundaries_text"), "normalized boundaries"),
        _clean(replacement.get("resolved_template_sha256"), "resolved template SHA256"),
    )


def _resolved_prompt_text(base: str, boundaries: str) -> str:
    open_marker = "<boundaries>\n"
    close_marker = "\n</boundaries>"
    if open_marker not in base or close_marker not in base:
        raise FactorialManifestError("agent_turn template lacks stable boundaries markers")
    before, rest = base.split(open_marker, 1)
    _old, after = rest.split(close_marker, 1)
    return before + open_marker + boundaries + close_marker + after


def _parse_scenario(raw: Any, index: int, actor_ids: set[str]) -> FactorialScenario:
    value = _object(raw, f"scenarios[{index}]")
    scenario_id = _clean(value.get("scenario_id"), f"scenarios[{index}].scenario_id")
    if value.get("turn_count") not in (None, 12):
        raise FactorialManifestError(f"{scenario_id}.turn_count must be 12")
    first = _object(value.get("scene_1"), f"{scenario_id}.scene_1")
    second = _object(value.get("scene_2"), f"{scenario_id}.scene_2")
    first_order = tuple(_clean(item, f"{scenario_id}.scene_1.turn_order") for item in _array(first.get("turn_order"), f"{scenario_id}.scene_1.turn_order"))
    second_order = tuple(_clean(item, f"{scenario_id}.scene_2.turn_order") for item in _array(second.get("turn_order"), f"{scenario_id}.scene_2.turn_order"))
    if len(first_order) != 6 or len(second_order) != 6:
        raise FactorialManifestError(f"{scenario_id} needs two six-turn scenes")
    order = first_order + second_order
    if len(set(order)) != 2 or any(order[i] == order[i - 1] for i in range(1, 12)):
        raise FactorialManifestError(f"{scenario_id} must strictly alternate two actors")
    if not set(order).issubset(actor_ids):
        raise FactorialManifestError(f"{scenario_id} references an unknown actor")
    if MIRELLE_ID not in order:
        raise FactorialManifestError(f"{scenario_id} must include Mirelle")
    setup = _clean(first.get("setup_observation"), f"{scenario_id}.scene_1.setup_observation")
    bridge = _clean(second.get("bridge"), f"{scenario_id}.scene_2.bridge")
    try:
        scene_2_elapsed_s = int(value.get("scene_2_elapsed_s", 0))
    except (TypeError, ValueError) as error:
        raise FactorialManifestError(
            f"{scenario_id}.scene_2_elapsed_s must be an integer"
        ) from error
    if scene_2_elapsed_s < 0:
        raise FactorialManifestError(
            f"{scenario_id}.scene_2_elapsed_s cannot be negative"
        )
    forbidden = _array(value.get("forbidden_setup_content", []), f"{scenario_id}.forbidden_setup_content")
    combined = f"{setup}\n{bridge}".casefold()
    for item in forbidden:
        forbidden_text = _clean(item, f"{scenario_id}.forbidden_setup_content")
        if forbidden_text.casefold() in combined:
            raise FactorialManifestError(f"{scenario_id} contains forbidden setup text {forbidden_text!r}")
    scene_1 = SceneSpec(
        scene_id=f"{scenario_id}/scene_1",
        title=scenario_id,
        frame=setup,
        prior_public_exchange=(),
        turn_order=first_order,
        pressure_pulses=(),
        actor_observations={},
    )
    scene_2 = SceneSpec(
        scene_id=f"{scenario_id}/scene_2",
        title=scenario_id,
        frame="",
        prior_public_exchange=(),
        turn_order=second_order,
        pressure_pulses=(),
        between_scene_public_history=bridge,
        actor_observations={},
    )
    return FactorialScenario(
        scenario_id=scenario_id,
        scenes=(scene_1, scene_2),
        depth_relevance=str(value.get("depth_relevance", "") or "").strip(),
        scene_2_elapsed_s=scene_2_elapsed_s,
    )


def _group_case_scenes(scenes: Sequence[SceneSpec]) -> list[tuple[SceneSpec, ...]]:
    groups: dict[str, list[SceneSpec]] = {}
    for scene in scenes:
        root = scene.scene_id.rsplit("/", 1)[0]
        groups.setdefault(root, []).append(scene)
    return [tuple(groups[key]) for key in sorted(groups)]


def _profile_roster(seed: CheckpointFile, profile_raw: Any) -> tuple[CharacterRecord, ...]:
    profile = _object(profile_raw, "profile condition")
    actor_record = _parse_actor_record(profile.get("actor_record"), "profile.actor_record")
    roster = [character.model_copy(deep=True) for character in seed.characters]
    target = next((character for character in roster if character.character_id == MIRELLE_ID), None)
    if target is None:
        raise FactorialManifestError("seed checkpoint has no Mirelle")
    target.actor = actor_record
    for character in roster:
        character.pending_observations = []
    return tuple(roster)


def _actor_static(actor: CharacterRecord) -> dict[str, Any]:
    value = actor.model_dump(mode="json")
    private = dict(value.get("actor") or {})
    private.pop("facts", None)
    value["actor"] = private
    value["pending_observations"] = []
    return value


def _model_config(model: str) -> ModelConfig:
    return ModelConfig(
        event_router="",
        narrator="",
        image_director="",
        dnd_combat_manager="",
        agent_default=model,
        agent_standard=model,
        agent_convenience=model,
        character_manager="",
    )


def _fresh_checkpoint(cell: FactorialCell, model: str, conversation_id: str) -> CheckpointFile:
    checkpoint = cell.checkpoint_template.model_copy(deep=True)
    # The manifest case owns immutable condition material.  Every conversation
    # receives its own deep-copied roster so a pending-observation queue or
    # clock mutation can never cross cells, replicates, or parallel jobs.
    checkpoint.characters = [
        character.model_copy(deep=True) for character in cell.case.actors
    ]
    checkpoint.session = SessionState(
        session_id=f"mirelle-factorial:{conversation_id}",
        story_id="mirelle-dialogue-factorial",
        config=SessionConfig(
            models=_model_config(model),
            settings=SessionSettings(ruleset_id="narrative"),
        ),
    )
    checkpoint.player_primer = ""
    checkpoint.session_conversation = []
    checkpoint.narrator_conversations = {}
    checkpoint.character_conversations = {
        character.character_id: [] for character in checkpoint.characters
    }
    checkpoint.canonical_events = []
    checkpoint.visibility_log = []
    for character in checkpoint.characters:
        character.clock_at_s = 0
        character.last_agent_turn_at_s = None
        character.pending_observations = []
    return checkpoint


class FactorialPromptManager(PromptManager):
    """Apply the normalized behavior replacement in memory only."""

    def __init__(self, prompts_dir: Path, *, boundaries: str = "", resolved_sha256: str = ""):
        super().__init__(str(prompts_dir))
        self.boundaries = boundaries
        self.resolved_sha256 = resolved_sha256

    def _load_template_text(self, template_name: str) -> str:
        text = super()._load_template_text(template_name)
        if template_name != "agent_turn" or not self.boundaries:
            return text
        resolved = _resolved_prompt_text(text, self.boundaries)
        if self.resolved_sha256 and _sha256_text(resolved) != self.resolved_sha256:
            raise FactorialTechnicalInvalidity("resolved system prompt SHA256 mismatch")
        return resolved


def load_factorial_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> FactorialManifest:
    manifest_path = Path(path)
    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
        root = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as error:
        raise FactorialManifestError(f"cannot read manifest {manifest_path}: {error}") from error
    root = _object(root, "manifest")
    if root.get("schema_version") != FACTORIAL_SCHEMA_VERSION:
        raise FactorialManifestError(f"manifest schema_version must be {FACTORIAL_SCHEMA_VERSION}")
    prompt_sha, prompt_path_value, boundaries, resolved_sha = _prompt_material(root)
    prompt_path = _resolve_path(prompt_path_value, manifest_path)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if _sha256_text(prompt_text) != prompt_sha:
        raise FactorialManifestError("current agent_turn template SHA256 no longer matches manifest")
    if _sha256_text(_resolved_prompt_text(prompt_text, boundaries)) != resolved_sha:
        raise FactorialManifestError("resolved normalized prompt SHA256 no longer matches manifest")

    fixed = _object(root.get("fixed_runtime_contract"), "fixed_runtime_contract")
    generation = _object(fixed.get("generation"), "fixed_runtime_contract.generation")
    if generation.get("provider_live_calls") is not False:
        raise FactorialManifestError(
            "factorial generation must explicitly prohibit provider-live calls"
        )
    if "gpt-5.6-luna" not in str(generation.get("mode", "")):
        raise FactorialManifestError(
            "factorial generation must use the gpt-5.6-luna coding-agent proxy"
        )
    source = _object(root.get("source_revisions"), "source_revisions")
    current_source = _object(source.get("current"), "source_revisions.current")
    seed_info = _object(fixed.get("character_source", {}), "character_source")
    seed_path_value = seed_info.get("seed_path") or _object(current_source.get("seed"), "current seed").get("path")
    seed_path = _resolve_path(_clean(seed_path_value, "seed path"), manifest_path)
    try:
        seed = CheckpointFile.model_validate_json(seed_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise FactorialManifestError(f"cannot load current seed checkpoint: {error}") from error
    expected_seed_blob = str(_object(current_source.get("seed"), "current seed").get("blob_sha1", "") or "")
    if expected_seed_blob and _git_blob_sha1(seed_path) != expected_seed_blob:
        raise FactorialManifestError("current seed checkpoint blob SHA1 no longer matches manifest")

    profiles = _object(root.get("profile_conditions"), "profile_conditions")
    sparse = _profile_roster(seed, profiles.get("sparse_current"))
    rich = _profile_roster(seed, profiles.get("rich_legacy_augmentation"))
    sparse_mirelle = next(character for character in sparse if character.character_id == MIRELLE_ID)
    rich_mirelle = next(character for character in rich if character.character_id == MIRELLE_ID)
    seed_mirelle = next(character for character in seed.characters if character.character_id == MIRELLE_ID)
    if sparse_mirelle.actor is None or rich_mirelle.actor is None or seed_mirelle.actor is None:
        raise FactorialManifestError("profile conditions must include Mirelle actor records")
    if sparse_mirelle.actor.model_dump(mode="json") != seed_mirelle.actor.model_dump(mode="json"):
        raise FactorialManifestError(
            "sparse_current Mirelle actor record differs from the current seed"
        )
    if len(sparse_mirelle.actor.facts) != 4 or len(rich_mirelle.actor.facts) != 28:
        raise FactorialManifestError("Mirelle profile conditions must contain 4 and 28 facts")
    if [fact.text for fact in sparse_mirelle.actor.facts] != [fact.text for fact in rich_mirelle.actor.facts[:4]]:
        raise FactorialManifestError("rich profile must begin with sparse's exact four facts")
    scenarios_raw = _array(root.get("scenarios"), "scenarios")
    if len(scenarios_raw) != 3:
        raise FactorialManifestError("manifest must contain exactly three scenarios")
    scenarios = tuple(
        _parse_scenario(item, index, {character.character_id for character in seed.characters})
        for index, item in enumerate(scenarios_raw)
    )
    profile_lookup = {
        "sparse_current": ("sparse", sparse),
        "rich_legacy_augmentation": ("rich", rich),
    }
    prompt_lookup = {
        "lean_current": ("lean", "", prompt_sha),
        "normalized_legacy_behavior": ("normalized_legacy", boundaries, resolved_sha),
    }
    cells_raw = _array(root.get("cells"), "cells")
    if len(cells_raw) != 4:
        raise FactorialManifestError("manifest must contain exactly four cells")
    cells: list[FactorialCell] = []
    for index, raw_cell in enumerate(cells_raw):
        value = _object(raw_cell, f"cells[{index}]")
        cell_id = _clean(value.get("cell_id"), f"cells[{index}].cell_id")
        profile_condition = _clean(value.get("profile_condition"), f"{cell_id}.profile_condition")
        prompt_condition = _clean(value.get("prompt_condition"), f"{cell_id}.prompt_condition")
        if profile_condition not in profile_lookup or prompt_condition not in prompt_lookup:
            raise FactorialManifestError(f"{cell_id} has an unknown profile or prompt condition")
        depth, actors = profile_lookup[profile_condition]
        instruction, boundaries_for_cell, cell_prompt_sha = prompt_lookup[prompt_condition]
        case = BenchmarkCase(
            case_id=cell_id,
            title=cell_id,
            suite="ordinary_surface",
            scenes=tuple(scene for scenario in scenarios for scene in scenario.scenes),
            actors=actors,
            source_metadata={"profile_condition": profile_condition, "prompt_condition": prompt_condition},
        )
        cells.append(
            FactorialCell(
                cell_id=cell_id,
                depth=depth,
                instruction=instruction,
                profile_condition=profile_condition,
                prompt_condition=prompt_condition,
                case=case,
                checkpoint_template=seed.model_copy(deep=True),
                resolved_system_sha256=cell_prompt_sha,
                source_provenance={
                    "profile_condition": profile_condition,
                    "prompt_condition": prompt_condition,
                    "source_revisions": source.get("current", {}),
                },
                normalized_boundaries=boundaries_for_cell,
                scenario_specs=scenarios,
            )
        )
    keys = {(cell.depth, cell.instruction) for cell in cells}
    if keys != {("sparse", "lean"), ("rich", "lean"), ("sparse", "normalized_legacy"), ("rich", "normalized_legacy")}:
        raise FactorialManifestError("cells must define the complete sparse/rich by lean/normalized-legacy cross")
    if len({cell.cell_id for cell in cells}) != 4:
        raise FactorialManifestError("cell ids must be unique")
    reference = cells[0]
    for cell in cells:
        current_scenes = [(scene.scene_id, scene.frame, scene.turn_order, scene.between_scene_public_history) for scene in cell.case.scenes]
        expected_scenes = [(scene.scene_id, scene.frame, scene.turn_order, scene.between_scene_public_history) for scene in reference.case.scenes]
        if current_scenes != expected_scenes:
            raise FactorialManifestError("all cells must share identical scenario topology")
        if any(character.pending_observations for character in cell.case.actors):
            raise FactorialManifestError("cell actors cannot start with pending observations")
    for depth in ("sparse", "rich"):
        lean = next(cell for cell in cells if (cell.depth, cell.instruction) == (depth, "lean"))
        legacy = next(cell for cell in cells if (cell.depth, cell.instruction) == (depth, "normalized_legacy"))
        if [actor.model_dump(mode="json") for actor in lean.case.actors] != [actor.model_dump(mode="json") for actor in legacy.case.actors]:
            raise FactorialManifestError("instruction contrast changed actor material")
    sparse_cell = next(cell for cell in cells if cell.depth == "sparse")
    rich_cell = next(cell for cell in cells if cell.depth == "rich")
    for sparse_actor, rich_actor in zip(sparse_cell.case.actors, rich_cell.case.actors, strict=True):
        if _actor_static(sparse_actor) != _actor_static(rich_actor):
            raise FactorialManifestError("depth contrast changed non-fact actor material")
        sparse_facts = [fact.text for fact in sparse_actor.actor.facts] if sparse_actor.actor else []
        rich_facts = [fact.text for fact in rich_actor.actor.facts] if rich_actor.actor else []
        if sparse_actor.character_id == MIRELLE_ID and rich_facts[: len(sparse_facts)] != sparse_facts:
            raise FactorialManifestError("rich Mirelle facts are not sparse-plus-augmentation")
        if sparse_actor.character_id != MIRELLE_ID and sparse_facts != rich_facts:
            raise FactorialManifestError("depth contrast changed facts for a non-Mirelle actor")
    return FactorialManifest(
        manifest_sha256=_sha256_text(raw_text),
        cells=tuple(cells),
        prompt_sha256=prompt_sha,
        resolved_prompt_sha256=resolved_sha,
        seed_path=str(seed_path),
        source_provenance={"source_revisions": source, "profile_conditions": profiles},
        scenarios=scenarios,
        blind_review_contract=_object(
            root.get("blind_review_contract", {}), "blind_review_contract"
        ),
        decision_rule=_object(
            root.get("preregistered_decision_rule", {}),
            "preregistered_decision_rule",
        ),
    )


def _public_updates_render(updates: Sequence[Mapping[str, Any]]) -> str:
    """Use the benchmark's neutral witness rendering for public beats."""

    return _benchmark_render_public_updates(updates)


def _history_sha(checkpoint: CheckpointFile, actor_id: str) -> str:
    return _sha256_json([
        message.model_dump(mode="json")
        for message in checkpoint.character_conversations.get(actor_id, [])
    ])


def _assert_prompt_contract(
    request: BenchmarkRequest,
    actor: CharacterRecord,
    *,
    cell: FactorialCell,
    scenario: FactorialScenario,
) -> None:
    if request.compact is not False:
        raise FactorialTechnicalInvalidity("CharacterAgent compact must remain false")
    if request.cache is not True or request.temperature != 0.6 or request.max_tokens != 2000:
        raise FactorialTechnicalInvalidity("CharacterAgent request settings changed")
    if len(request.messages) < 2 or request.messages[0].get("role") != "system" or request.messages[-1].get("role") != "user":
        raise FactorialTechnicalInvalidity("request must be system/history/current-user")
    system = str(request.messages[0].get("content", ""))
    user = str(request.messages[-1].get("content", ""))
    if "<you>" not in user or "</you>" not in user or "<now>" not in user or "</now>" not in user:
        raise FactorialTechnicalInvalidity("current user turn lacks exact second-person packet")
    if f"You are {actor.name}." not in user:
        raise FactorialTechnicalInvalidity("actor identity is absent from current user packet")
    for fact in actor.actor.facts if actor.actor else ():
        if fact.text not in user:
            raise FactorialTechnicalInvalidity("actor fact is absent from current user packet")
        if fact.text in system:
            raise FactorialTechnicalInvalidity("actor fact leaked into system prefix")
    if actor.name in system or actor.character_id in system:
        raise FactorialTechnicalInvalidity("actor identity leaked into system prefix")
    if cell.instruction == "normalized_legacy":
        if "<boundaries>" not in system or "Let your voice come from" not in system:
            raise FactorialTechnicalInvalidity("normalized boundaries are absent from system")
    banned = ("AGENT-TURN", "PERCEPTION", "cell_id", "normalized_legacy", "benchmark", "checkpoint", "rubric", "provider")
    model_text = json.dumps(request.messages, ensure_ascii=False).casefold()
    for term in banned:
        if re.search(rf"(?<![a-z0-9_]){re.escape(term.casefold())}(?![a-z0-9_])", model_text):
            raise FactorialTechnicalInvalidity(f"model-facing request contains banned term {term!r}")
    for scene in scenario.scenes:
        if scene.frame and scene.frame in system:
            raise FactorialTechnicalInvalidity("turn-specific scene frame leaked into system")


def _default_responder(request: BenchmarkRequest) -> ModelCall:
    return ModelCall(
        content="I answer what is in front of us, then leave room for the other person.",
        model=request.model,
        provider="coding_agent_proxy",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


Responder = Callable[[BenchmarkRequest], ModelCall | str | Mapping[str, Any] | Awaitable[Any]]


def _provider_payload(request: BenchmarkRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "role": request.role,
        "messages": copy.deepcopy(_json_safe(request.messages)),
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "cache": request.cache,
        "compact": request.compact,
    }


def _response_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, ModelCall):
        return _raw_response_payload(value)
    if isinstance(value, str):
        return {"content": value}
    if isinstance(value, Mapping):
        if "response" in value and "content" not in value:
            nested = value["response"]
            if isinstance(nested, Mapping):
                value = nested
        if not isinstance(value.get("content"), str):
            raise FactorialTechnicalInvalidity("proxy response must contain string content")
        return copy.deepcopy(_json_safe(dict(value)))
    raise FactorialTechnicalInvalidity(f"unsupported proxy response type {type(value).__name__}")


def _proxy_response_parts(value: Any) -> tuple[str, dict[str, Any]]:
    """Extract the orchestrator's fresh-agent id without sending it to the model."""

    wrapper = value if isinstance(value, Mapping) else {}
    nested = wrapper.get("response") if isinstance(wrapper, Mapping) else None
    nested_mapping = nested if isinstance(nested, Mapping) else {}
    proxy_agent_id = (
        wrapper.get("proxy_agent_id")
        or wrapper.get("session_id")
        or nested_mapping.get("proxy_agent_id")
        or nested_mapping.get("session_id")
    )
    if not proxy_agent_id:
        raise FactorialTechnicalInvalidity(
            "imported proxy response must identify its fresh proxy agent"
        )
    try:
        cleaned_id = _clean(proxy_agent_id, "proxy_agent_id")
    except FactorialManifestError as error:
        raise FactorialTechnicalInvalidity(str(error)) from error
    if any(term in cleaned_id.casefold() for term in (
        "sparse", "rich", "lean", "legacy", "cell", "scenario", "pilot", "confirmatory"
    )):
        raise FactorialTechnicalInvalidity("proxy agent id must be opaque")
    response_data = _response_payload(value)
    response_data.pop("proxy_agent_id", None)
    response_data.pop("session_id", None)
    return cleaned_id, response_data


def export_pending_request(path: str | Path, request: BenchmarkRequest, *, conversation_token: str) -> Path:
    token = _clean(conversation_token, "opaque conversation token")
    if any(term in token.casefold() for term in ("sparse", "rich", "lean", "legacy", "cell")):
        raise FactorialTechnicalInvalidity("pending request token is not opaque")
    payload = _provider_payload(request)
    document = {
        "schema_version": PENDING_SCHEMA_VERSION,
        "conversation": token,
        "proxy_session_id": token,
        "sequence": request.turn_index - 1,
        "request": payload,
        "request_sha256": _sha256_json(payload),
    }
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resolved


def import_response(ledger_path: str | Path, pending_path: str | Path, response: Any) -> Path:
    pending_file = Path(pending_path)
    try:
        pending = json.loads(pending_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FactorialTechnicalInvalidity(f"cannot read pending request: {error}") from error
    pending = _object(pending, "pending request")
    request = _object(pending.get("request"), "pending request.request")
    if (
        pending.get("schema_version") != PENDING_SCHEMA_VERSION
        or pending.get("request_sha256") != _sha256_json(request)
        or pending.get("proxy_session_id") != pending.get("conversation")
    ):
        raise FactorialTechnicalInvalidity("pending request schema or hash is invalid")
    token = _clean(pending.get("conversation"), "pending conversation")
    ledger_file = Path(ledger_path)
    if ledger_file.exists():
        try:
            ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FactorialTechnicalInvalidity(f"cannot read response ledger: {error}") from error
    else:
        ledger = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "conversation": token,
            "responses": [],
        }
    ledger = dict(_object(ledger, "response ledger"))
    responses = ledger.get("responses")
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION or ledger.get("conversation") != token or not isinstance(responses, list):
        raise FactorialTechnicalInvalidity("response ledger identity or shape is invalid")
    sequence = int(pending.get("sequence", -1))
    if sequence != len(responses):
        raise FactorialTechnicalInvalidity("pending sequence is not the next ledger response")
    proxy_agent_id, response_data = _proxy_response_parts(response)
    stored_proxy_agent_id = ledger.get("proxy_agent_id")
    if responses and not stored_proxy_agent_id:
        raise FactorialTechnicalInvalidity(
            "response ledger lacks the proxy agent identity for existing responses"
        )
    if stored_proxy_agent_id and stored_proxy_agent_id != proxy_agent_id:
        raise FactorialTechnicalInvalidity(
            "proxy agent identity changed within one conversation"
        )
    ledger["proxy_agent_id"] = proxy_agent_id
    responses.append({
        "sequence": sequence,
        "request": copy.deepcopy(request),
        "request_sha256": pending["request_sha256"],
        "proxy_agent_id": proxy_agent_id,
        "response": response_data,
        "response_sha256": _sha256_json(response_data),
    })
    ledger_file.parent.mkdir(parents=True, exist_ok=True)
    ledger_file.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ledger_file


class ProxyResponder:
    """Consume exact opaque ledger entries in order, writing the next pending request."""

    def __init__(self, ledger_path: Path, pending_path: Path, token: str):
        self.ledger_path = ledger_path
        self.pending_path = pending_path
        self.token = token
        self.next_sequence = 0
        self.proxy_agent_id = ""

    def __call__(self, request: BenchmarkRequest) -> ModelCall:
        if self.ledger_path.exists():
            try:
                ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise FactorialTechnicalInvalidity(f"cannot read proxy ledger: {error}") from error
        else:
            ledger = {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "conversation": self.token,
                "responses": [],
            }
        ledger = _object(ledger, "response ledger")
        entries = ledger.get("responses")
        if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION or ledger.get("conversation") != self.token or not isinstance(entries, list):
            raise FactorialTechnicalInvalidity("proxy ledger identity or shape is invalid")
        stored_proxy_agent_id = ledger.get("proxy_agent_id")
        if entries and not isinstance(stored_proxy_agent_id, str):
            raise FactorialTechnicalInvalidity(
                "response ledger lacks its proxy agent identity"
            )
        if stored_proxy_agent_id:
            self.proxy_agent_id = stored_proxy_agent_id
        sequence = self.next_sequence
        if sequence >= len(entries):
            export_pending_request(self.pending_path, request, conversation_token=self.token)
            raise PendingRequest(request, self.pending_path, sequence)
        entry = _object(entries[sequence], "response ledger entry")
        if entry.get("sequence") != sequence:
            raise FactorialTechnicalInvalidity("proxy ledger response sequence is out of order")
        if entry.get("proxy_agent_id") != self.proxy_agent_id:
            raise FactorialTechnicalInvalidity(
                "proxy ledger response has a different agent identity"
            )
        stored_request = _object(entry.get("request"), "response ledger request")
        if entry.get("request_sha256") != _sha256_json(stored_request) or dict(stored_request) != _provider_payload(request):
            raise FactorialTechnicalInvalidity("proxy ledger request differs from exact current request")
        response = _object(entry.get("response"), "response ledger response")
        if entry.get("response_sha256") and entry["response_sha256"] != _sha256_json(response):
            raise FactorialTechnicalInvalidity("proxy ledger response hash is stale")
        content = response.get("content")
        if not isinstance(content, str):
            raise FactorialTechnicalInvalidity("proxy ledger response content is not a string")
        self.next_sequence += 1
        return ModelCall(
            content=content,
            model=str(response.get("model") or request.model),
            provider="coding_agent_proxy",
            usage=dict(response.get("usage", {}) or {}),
            raw_response=copy.deepcopy(response.get("raw_response")),
            assistant_content=copy.deepcopy(response.get("assistant_content")),
        )


async def run_conversation(
    cell: FactorialCell,
    scenario: FactorialScenario,
    *,
    model: str = DEFAULT_MODEL,
    conversation_id: str | None = None,
    conversation_token: str | None = None,
    responder: Responder | None = None,
    ledger_responder: ProxyResponder | None = None,
    prompt_dir: Path = REPO_ROOT / "app" / "prompts",
    phase: str = "confirmatory",
) -> FactorialConversation:
    if responder is not None and ledger_responder is not None:
        raise ValueError("responder and ledger_responder are mutually exclusive")
    selected_model = _clean(model, "model")
    if selected_model != DEFAULT_MODEL:
        raise FactorialTechnicalInvalidity(
            "factorial proxy generation is frozen to gpt-5.6-luna"
        )
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    run_id = conversation_id or f"conversation-{uuid.uuid4().hex}"
    token = conversation_token or f"opaque-{_sha256_text(run_id)[:24]}"
    checkpoint = _fresh_checkpoint(cell, selected_model, run_id)
    result = FactorialConversation(
        conversation_id=run_id,
        conversation_token=token,
        cell=cell,
        scenario=scenario,
        model=selected_model,
        checkpoint=checkpoint,
        initial_checkpoint_sha256=_sha256_json(checkpoint.model_dump(mode="json")),
        turns=[],
        public_transcript=[],
        phase=phase,
        proxy_agent_id=(
            ledger_responder.proxy_agent_id
            if ledger_responder is not None and ledger_responder.proxy_agent_id
            else f"offline-{_sha256_text(token)[:24]}"
        ),
    )
    client = _RecordingCharacterClient(
        model=selected_model,
        responder=ledger_responder or responder or _default_responder,
    )
    boundaries = "" if cell.instruction == "lean" else _normalized_boundaries_for_cell(cell)
    agent = CharacterAgent(
        client,  # type: ignore[arg-type]
        FactorialPromptManager(
            prompt_dir,
            boundaries=boundaries,
            resolved_sha256=cell.resolved_system_sha256 if boundaries else "",
        ),
    )
    seen_counts = {character.character_id: 0 for character in checkpoint.characters}
    public_transcript = result.public_transcript
    global_turn = 0
    try:
        for scene_index, scene in enumerate(scenario.scenes):
            if scene_index == 0:
                public_transcript.append({
                    "kind": "scene_start",
                    "scene_index": scene_index,
                    "scene_id": scene.scene_id,
                    "text": scene.frame,
                })
            else:
                # The bridge is a single public update, but its elapsed time
                # must advance the same session clock used by <now>.  Keep
                # this before the first scene-two packet so “next evening”,
                # “following afternoon”, and “two hours later” are true to
                # the relative-time block.
                checkpoint.session.leading_at_s += scenario.scene_2_elapsed_s
                public_transcript.append({
                    "kind": "scene_break",
                    "scene_index": scene_index,
                    "scene_id": scene.scene_id,
                    "text": scene.between_scene_public_history,
                })
            for scene_turn, actor_id in enumerate(scene.turn_order, start=1):
                global_turn += 1
                actor = next(
                    character
                    for character in checkpoint.characters
                    if character.character_id == actor_id
                )
                witnessed = _public_updates_render(public_transcript[seen_counts[actor_id] :])
                if witnessed:
                    actor.pending_observations.append(witnessed)
                history_before = len(checkpoint.character_conversations.get(actor_id, []))
                before_hash = _history_sha(checkpoint, actor_id)
                client.begin(_BenchmarkCallContext(
                    conversation_id=run_id,
                    case_id=cell.cell_id,
                    scene_id=scene.scene_id,
                    scene_index=scene_index,
                    scene_turn_index=scene_turn,
                    turn_index=global_turn,
                    actor_id=actor_id,
                    actor_name=actor.name,
                ))
                started = time.perf_counter()
                draft = await agent.draft_turn(actor, checkpoint, frame="foreground", local_context="")
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                request = client.last_request
                call = client.last_call
                if request is None or call is None:
                    raise FactorialTechnicalInvalidity("CharacterAgent completed without captured request")
                if ledger_responder is not None:
                    if not ledger_responder.proxy_agent_id:
                        raise FactorialTechnicalInvalidity(
                            "proxy response did not identify its agent"
                        )
                    if result.proxy_agent_id.startswith("offline-"):
                        result.proxy_agent_id = ledger_responder.proxy_agent_id
                    elif result.proxy_agent_id != ledger_responder.proxy_agent_id:
                        raise FactorialTechnicalInvalidity(
                            "proxy agent identity changed within one conversation"
                        )
                _assert_prompt_contract(request, actor, cell=cell, scenario=scenario)
                public_text = "<silence/>" if draft.output.is_silence else draft.output.public_text
                agent.commit_draft(actor, checkpoint, draft)
                history_after = len(checkpoint.character_conversations.get(actor_id, []))
                after_hash = _history_sha(checkpoint, actor_id)
                if history_after != history_before + 2:
                    raise FactorialTechnicalInvalidity("turn did not add exactly one user/assistant pair")
                stored_user = checkpoint.character_conversations[actor_id][-2]
                current_user = str(request.messages[-1].get("content", ""))
                if not isinstance(stored_user.content, str):
                    raise FactorialTechnicalInvalidity("stored user history is not plain text")
                if "<presentation_catalog" not in current_user and stored_user.content != current_user:
                    raise FactorialTechnicalInvalidity("history stripped content beyond presentation catalog")
                result.turns.append({
                    "sequence": global_turn - 1,
                    "scene_index": scene_index,
                    "scene_id": scene.scene_id,
                    "scene_turn": scene_turn,
                    "turn_index": global_turn,
                    "actor_id": actor_id,
                    "actor_name": actor.name,
                    "request": {
                        **_provider_payload(request),
                        "request_sha256": _sha256_json(_provider_payload(request)),
                        "system_prompt_sha256": _sha256_text(str(request.messages[0].get("content", ""))),
                    },
                    "response": {
                        **_raw_response_payload(call),
                        "response_sha256": _sha256_text(call.content),
                        "proxy_agent_id": result.proxy_agent_id,
                    },
                    "public_text": public_text,
                    "history": {
                        "before_sha256": before_hash,
                        "after_sha256": after_hash,
                        "message_count_before": history_before,
                        "message_count_after": history_after,
                    },
                    "elapsed_ms": elapsed_ms,
                })
                public_transcript.append({
                    "kind": "turn",
                    "scene_index": scene_index,
                    "scene_id": scene.scene_id,
                    "scene_turn_index": scene_turn,
                    "turn_index": global_turn,
                    "actor_id": actor_id,
                    "speaker_name": actor.name,
                    "text": public_text,
                })
                seen_counts[actor_id] = len(public_transcript)
                checkpoint.session.turn_index = global_turn
                # A normal turn advances the synthetic clock by one second;
                # unlike the benchmark's turn-index clock this preserves a
                # manifest-authored inter-scene interval once it has elapsed.
                checkpoint.session.leading_at_s += 1
        if len(result.turns) != 12:
            raise FactorialTechnicalInvalidity("conversation did not complete twelve turns")
        if ledger_responder is not None and ledger_responder.next_sequence != 12:
            raise FactorialTechnicalInvalidity("proxy ledger did not consume twelve responses")
        if not result.proxy_agent_id:
            raise FactorialTechnicalInvalidity("conversation has no proxy agent identity")
    except PendingRequest:
        raise
    except (FactorialTechnicalInvalidity,) as error:
        result.status = "technical_invalidity"
        result.technical_invalidity = str(error)
    except (ModelFailure, CharacterAgentOutputError) as error:
        result.status = "model_failure"
        result.model_failure = str(error)
    except Exception as error:
        result.status = "model_failure"
        result.model_failure = f"{type(error).__name__}: {error}"
    return result


def _normalized_boundaries_for_cell(cell: FactorialCell) -> str:
    return cell.normalized_boundaries


async def run_factorial(
    manifest: FactorialManifest,
    *,
    model: str = DEFAULT_MODEL,
    responder: Responder | None = None,
    replicates: int | None = None,
    phase: str = "pilot",
    parallelism: int | None = None,
    prompt_dir: Path = REPO_ROOT / "app" / "prompts",
) -> list[FactorialConversation]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if replicates is None:
        replicates = 4 if phase == "confirmatory" else 1
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    _validate_phase_replicates(phase, replicates)
    scenarios = _phase_scenarios(manifest, phase)
    jobs: list[tuple[FactorialCell, FactorialScenario, str, str]] = []
    for replicate in range(replicates):
        for cell in manifest.cells:
            for scenario in scenarios:
                slot = _conversation_slot(
                    manifest,
                    phase=phase,
                    cell=cell,
                    scenario=scenario,
                    replicate=replicate + 1,
                )
                digest = _sha256_text(slot)[:24]
                jobs.append((cell, scenario, f"conversation-{digest}", f"opaque-{digest}"))
    limit = len(jobs) if parallelism is None else parallelism
    if limit < 1:
        raise ValueError("parallelism must be at least one")
    semaphore = asyncio.Semaphore(limit)

    async def one(job: tuple[FactorialCell, FactorialScenario, str, str]) -> FactorialConversation:
        async with semaphore:
            return await run_conversation(
                job[0], job[1], model=model, conversation_id=job[2],
                conversation_token=job[3], responder=responder, prompt_dir=prompt_dir,
                phase=phase,
            )

    return list(await asyncio.gather(*(one(job) for job in jobs)))


def _manifest_scenarios(manifest: FactorialManifest) -> tuple[FactorialScenario, ...]:
    if manifest.scenarios:
        return manifest.scenarios
    first = manifest.cells[0]
    grouped = _group_case_scenes(first.case.scenes)
    return tuple(
        FactorialScenario(
            scenario_id=group[0].scene_id.rsplit("/", 1)[0],
            scenes=group,
        )
        for group in grouped
    )


def _phase_scenarios(
    manifest: FactorialManifest,
    phase: str,
) -> tuple[FactorialScenario, ...]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    scenarios = _manifest_scenarios(manifest)
    if phase != "pilot":
        return scenarios
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    missing = [scenario_id for scenario_id in PILOT_SCENARIO_IDS if scenario_id not in by_id]
    if missing:
        raise FactorialManifestError(
            "pilot scenarios missing from manifest: " + ", ".join(missing)
        )
    return tuple(by_id[scenario_id] for scenario_id in PILOT_SCENARIO_IDS)


def _validate_phase_replicates(phase: str, replicates: int) -> None:
    expected = PREREGISTERED_REPLICATES.get(phase)
    if expected is not None and replicates != expected:
        raise FactorialManifestError(
            f"{phase} is frozen to {expected} replicate(s); "
            "use phase='all' for exploratory counts"
        )


def _conversation_slot(
    manifest: FactorialManifest,
    *,
    phase: str,
    cell: FactorialCell,
    scenario: FactorialScenario,
    replicate: int,
) -> str:
    """Derive disjoint opaque ids for every preregistered phase slot."""

    return (
        f"{manifest.manifest_sha256}:{phase}:{cell.cell_id}:"
        f"{scenario.scenario_id}:r{replicate}"
    )


def _speaker_labels(conversation: FactorialConversation) -> dict[str, str]:
    labels: dict[str, str] = {}
    for actor_id in conversation.scenario.turn_order:
        if actor_id not in labels:
            labels[actor_id] = f"Speaker {chr(ord('A') + len(labels))}"
    return labels


def _empty_preregistered_scores(
    review_contract: Mapping[str, Any],
) -> dict[str, Any]:
    scores = _object(review_contract.get("scores"), "blind_review_contract.scores")
    form: dict[str, Any] = {}
    for score_id in ("B", "Q"):
        spec = _object(scores.get(score_id), f"blind_review_contract.scores.{score_id}")
        dimensions = _array(
            spec.get("dimensions"),
            f"blind_review_contract.scores.{score_id}.dimensions",
        )
        form[score_id] = {
            "range": list(_array(spec.get("range"), f"scores.{score_id}.range")),
            "aggregation": _clean(
                spec.get("aggregation"),
                f"scores.{score_id}.aggregation",
            ),
            "dimensions": {
                _clean(
                    _object(dimension, f"scores.{score_id}.dimensions[{index}]").get("id"),
                    f"scores.{score_id}.dimensions[{index}].id",
                ): {
                    "question": _clean(
                        _object(dimension, f"scores.{score_id}.dimensions[{index}]").get("question"),
                        f"scores.{score_id}.dimensions[{index}].question",
                    ),
                    "score": None,
                    "evidence": "",
                }
                for index, dimension in enumerate(dimensions)
            },
            "total": None,
        }
    evidence_requirement = review_contract.get("evidence_requirement")
    if evidence_requirement:
        form["evidence_requirement"] = _clean(
            evidence_requirement,
            "blind_review_contract.evidence_requirement",
        )
    return form


def _blind_name_aliases(
    conversation: FactorialConversation,
    labels: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    aliases: list[tuple[str, str]] = []
    stopwords = {"a", "an", "and", "of", "the"}
    for actor in conversation.cell.case.actors:
        actor_id = actor.character_id
        label = labels.get(actor_id, "someone")
        # Situation bridges use first names, while model prose can use a full
        # name, an id, or a possessive first name. Scrub every name component
        # on word boundaries so no identity survives in the blind packet.
        names = {actor.name, actor_id, *actor.name.split()}
        aliases.extend(
            (alias, label)
            for alias in names
            if len(alias) > 1 and alias.casefold() not in stopwords
        )
    return tuple(sorted(aliases, key=lambda item: len(item[0]), reverse=True))


def _blinded_reviews(
    conversations: Sequence[FactorialConversation],
    *,
    shuffle_seed: str = "",
    review_contract: Mapping[str, Any] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packets: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for conversation in conversations:
        blind_id = f"blind-{_sha256_text(conversation.conversation_token)[:16]}"
        labels = _speaker_labels(conversation)
        aliases = _blind_name_aliases(conversation, labels)
        transcript: list[dict[str, Any]] = []

        def anonymize(text: str) -> str:
            result = text
            for alias, label in aliases:
                result = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])",
                    label,
                    result,
                    flags=re.IGNORECASE,
                )
            return result

        for update in conversation.public_transcript:
            if update.get("kind") == "turn":
                label = labels.get(str(update.get("actor_id")), "Speaker A")
                transcript.append({
                    "kind": "turn",
                    "line": update.get("turn_index"),
                    "speaker": label,
                    "text": anonymize(str(update.get("text", ""))),
                })
            elif update.get("kind") == "scene_start":
                transcript.append({
                    "kind": "setup",
                    "line": 0,
                    "speaker": "Situation",
                    "text": anonymize(str(update.get("text", ""))),
                })
            elif update.get("kind") == "scene_break":
                transcript.append({
                    "kind": "scene_break",
                    "line": 0,
                    "speaker": "Situation",
                    "text": anonymize(str(update.get("text", ""))),
                })
        sheet = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "blind_id": blind_id,
            "unit": "whole_conversation",
            "transcript": transcript,
            "scores": _empty_preregistered_scores(review_contract),
            "outcome": {
                "human_quality_pass": None,
                "human_quality_fail": None,
                "invalid": None,
            },
        }
        key = {
            "blind_id": blind_id,
            "conversation_id": conversation.conversation_id,
            "cell_id": conversation.cell.cell_id,
            "depth": conversation.cell.depth,
            "instruction": conversation.cell.instruction,
            "scenario_id": conversation.scenario.scenario_id,
            "speaker_map": labels,
            "post_unblind_audit": {
                field: None for field in POST_UNBLIND_AUDIT_FIELDS
            },
        }
        packets.append((sheet, key))
    packets.sort(
        key=lambda packet: _sha256_text(
            f"{shuffle_seed}:{packet[0]['blind_id']}"
        )
    )
    return [packet[0] for packet in packets], [packet[1] for packet in packets]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_proxy_response(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise FactorialTechnicalInvalidity(
            f"cannot read proxy response {path}: {error}"
        ) from error
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _import_response_from_cli(
    pending_path: Path,
    response_path: Path,
    *,
    ledger_path: Path | None,
    ledger_dir: Path | None,
) -> Path:
    try:
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FactorialTechnicalInvalidity(
            f"cannot read pending request {pending_path}: {error}"
        ) from error
    pending = _object(pending, "pending request")
    token = _clean(pending.get("conversation"), "pending conversation")
    if not re.fullmatch(r"opaque-[0-9a-f]{24}", token):
        raise FactorialTechnicalInvalidity(
            "pending conversation is not a generated opaque slot token"
        )
    if ledger_path is not None and ledger_dir is not None:
        raise FactorialTechnicalInvalidity(
            "choose --ledger or --ledger-dir for response import, not both"
        )
    resolved_ledger = ledger_path
    if resolved_ledger is None:
        resolved_ledger = (
            ledger_dir / f"{token}.json"
            if ledger_dir is not None
            else pending_path.with_name(f"{token}.json")
        )
    return import_response(
        resolved_ledger,
        pending_path,
        _read_proxy_response(response_path),
    )


def write_artifacts(manifest: FactorialManifest, conversations: Sequence[FactorialConversation], output_dir: str | Path) -> Path:
    root = Path(output_dir)
    proxy_ids = [conversation.proxy_agent_id for conversation in conversations]
    if any(not proxy_id for proxy_id in proxy_ids):
        raise FactorialTechnicalInvalidity(
            "every completed conversation must identify its proxy agent"
        )
    if len(set(proxy_ids)) != len(proxy_ids):
        raise FactorialTechnicalInvalidity(
            "one fresh proxy agent is required per opaque conversation"
        )
    for conversation in conversations:
        _write_json(root / "raw" / f"{conversation.conversation_id}.json", conversation.artifact())
    sheets, answer_key = _blinded_reviews(
        conversations,
        shuffle_seed=manifest.manifest_sha256,
        review_contract=manifest.blind_review_contract,
    )
    _write_json(root / "review" / "whole_conversation_review.json", {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "model_judge": False,
        "auto_semantic_score": False,
        "review_contract": manifest.blind_review_contract,
        "sheets": sheets,
    })
    _write_json(root / "review" / "answer_key.json", {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "answer_key": answer_key,
    })
    _write_json(root / "manifest_provenance.json", {
        "manifest_sha256": manifest.manifest_sha256,
        "prompt_sha256": manifest.prompt_sha256,
        "resolved_prompt_sha256": manifest.resolved_prompt_sha256,
        "seed_path": manifest.seed_path,
        "source_provenance": manifest.source_provenance,
        "blind_review_contract": manifest.blind_review_contract,
        "preregistered_decision_rule": manifest.decision_rule,
    })
    _write_json(root / "report.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "manifest_sha256": manifest.manifest_sha256,
        "execution": {"mode": "coding_agent_proxy", "model": "gpt-5.6-luna", "provider_live_calls": False},
        "proxy_agent_ids": proxy_ids,
        "conversation_count": len(conversations),
        "turn_count": sum(len(conversation.turns) for conversation in conversations),
        "phases": sorted({conversation.phase for conversation in conversations}),
        "statuses": [conversation.status for conversation in conversations],
        "technical_invalidity_count": sum(conversation.status == "technical_invalidity" for conversation in conversations),
        "model_failure_count": sum(conversation.status == "model_failure" for conversation in conversations),
    })
    return root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="pilot",
        help="preregistered slot set (pilot is the safe default)",
    )
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--parallelism", type=int, default=None)
    parser.add_argument("--ledger-dir", type=Path, default=None)
    parser.add_argument("--pending-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--import-pending", type=Path, default=None)
    parser.add_argument("--import-response", type=Path, default=None)
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    manifest = load_factorial_manifest(args.manifest)
    if (args.import_pending is None) != (args.import_response is None):
        raise FactorialTechnicalInvalidity(
            "--import-pending and --import-response must be supplied together"
        )
    if args.import_pending is not None:
        _import_response_from_cli(
            args.import_pending,
            args.import_response,
            ledger_path=args.ledger,
            ledger_dir=args.ledger_dir,
        )
        return 0
    if args.ledger is not None:
        raise FactorialTechnicalInvalidity("--ledger is only valid with response import")
    replicates = args.replicates
    if replicates is None:
        replicates = 4 if args.phase == "confirmatory" else 1
    _validate_phase_replicates(args.phase, replicates)
    if args.ledger_dir:
        conversations: list[FactorialConversation] = []
        for replicate in range(replicates):
            for cell in manifest.cells:
                for scenario in _phase_scenarios(manifest, args.phase):
                    slot = _conversation_slot(
                        manifest,
                        phase=args.phase,
                        cell=cell,
                        scenario=scenario,
                        replicate=replicate + 1,
                    )
                    digest = _sha256_text(slot)[:24]
                    token = f"opaque-{digest}"
                    ledger = args.ledger_dir / f"{token}.json"
                    pending_dir = args.pending_dir or args.ledger_dir
                    pending = pending_dir / f"{token}.pending.json"
                    proxy = ProxyResponder(ledger, pending, token)
                    conversation = await run_conversation(
                        cell,
                        scenario,
                        model=args.model,
                        conversation_id=f"conversation-{digest}",
                        conversation_token=token,
                        ledger_responder=proxy,
                        phase=args.phase,
                    )
                    conversations.append(conversation)
        write_artifacts(manifest, conversations, args.output)
        return 0
    conversations = await run_factorial(
        manifest,
        model=args.model,
        replicates=replicates,
        phase=args.phase,
        parallelism=args.parallelism,
    )
    write_artifacts(manifest, conversations, args.output)
    return 0 if all(conversation.status == "valid" for conversation in conversations) else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(_main_async(build_arg_parser().parse_args(argv)))
    except PendingRequest as error:
        print(str(error), file=sys.stderr)
        return 75
    except (FactorialManifestError, FactorialTechnicalInvalidity, ModelFailure) as error:
        print(f"factorial failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
