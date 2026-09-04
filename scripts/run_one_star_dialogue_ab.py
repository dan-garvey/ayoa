#!/usr/bin/env python3
# ruff: noqa: E402 -- executable script adds the repository root to sys.path.
"""Blinded Luna/Terra dialogue harness for One-Star character work.

The harness has two deliberately separate paths.  The offline path accepts a
deterministic responder (and is what tests use); the live path calls the normal
LLM client only when a person explicitly asks for ``--live``.  Both paths save
the exact prompt, response, timing, and usage records before creating a blinded
review set.  The review set omits model and variant labels so a human can score
voice and dramatic usefulness without knowing which A/B produced a sample.

No provider call is made by importing this module or by the default command.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.engine.character_agent import CharacterAgent
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMResponse
from app.schemas.characters import CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import (
    CanonicalEventRecord,
    ObserverGroups,
)
from app.schemas.events import ObservableFact


SEED_CHECKPOINT_PATH = (
    REPO_ROOT
    / "app"
    / "storage"
    / "stories"
    / "one_star_ascension_s1"
    / "ckpt_0000.json"
)


LUNA_MODEL = "gpt-5.6-luna"
TERRA_MODEL = "gpt-5.6-terra"
SOL_MODEL = "gpt-5.6-sol"


@dataclass(frozen=True)
class DialogueVariant:
    """One blinded A/B model variant."""

    variant_id: str
    model: str


@dataclass(frozen=True)
class DialogueCase:
    """A fixed situation used by both variants."""

    case_id: str
    actor_id: str
    prompt: str
    contract: str
    location: str = "niflheim_lobby"
    canonical_fact: str = ""
    pending_observation: str = ""


DEFAULT_VARIANTS: tuple[DialogueVariant, ...] = (
    DialogueVariant("luna", LUNA_MODEL),
    DialogueVariant("terra", TERRA_MODEL),
)


DEFAULT_CASES: tuple[DialogueCase, ...] = (
    DialogueCase(
        "opening",
        "edren_marr",
        """Summon-light has just dropped you with Renna and Mirelle into a
        cramped Floor 1 ruin. Armed goblins are already rushing you at close
        range. You hold a laundry pole, not a soldier's weapon. Give Edren's
        first contribution.""",
        "One frightened literal response; panic rather than mission refusal, wry comparison, or competent defense; do not invent another character's words.",
        location="tower_floor_1_goblin_ambush",
        canonical_fact="Summon-light drops Edren, Renna, and Mirelle among armed goblins already rushing to attack.",
        pending_observation="evt_dialogue_opening: Summon-light drops Edren, Renna, and Mirelle among armed goblins already rushing to attack.",
    ),
    DialogueCase(
        "repeated_pressure",
        "renna_holt",
        """For a third time, a goblin uses the fallen bodies to dart around
        Mirelle's spear toward Renna. Renna has already named the obvious
        danger once. The party needs movement now; choose what she does and,
        only if useful, what she says.""",
        "Do not repeat the plan or objection; make a materially different action.",
        location="tower_floor_1_goblin_ambush",
        canonical_fact="A goblin uses the fallen bodies to dart around Mirelle's spear toward Renna for a third time.",
        pending_observation="evt_dialogue_repeated_pressure: A goblin uses the fallen bodies to dart around Mirelle's spear toward Renna for a third time.",
    ),
    DialogueCase(
        "ally_danger",
        "mirelle_voss",
        """Renna is pinned under fallen stone while the first-floor exit is
        within reach. Mirelle can pull her free only by giving up the safe route
        and exposing herself to the goblins closing behind them.""",
        "Make a consequential value-laden choice; do not decide Renna's response.",
        location="tower_floor_1_goblin_ambush",
        canonical_fact="Renna is pinned beneath fallen stone while the first-floor exit is within reach and goblins close behind the party.",
        pending_observation="evt_dialogue_ally_danger: Renna is pinned beneath fallen stone while the first-floor exit is within reach and goblins close behind the party.",
    ),
    DialogueCase(
        "lobby_favoritism",
        "castor_valebrand",
        """The Master equips Castor with the only polished weapon and leaves a
        frightened one-star with a cracked tool. Castor notices the attention
        and can either exploit it, challenge it, or redirect it.""",
        "Expose Castor's social defense and relationship to status through one specific choice.",
        canonical_fact="The Master equips Castor with the only polished weapon and leaves a frightened one-star with a cracked tool.",
        pending_observation="evt_dialogue_lobby_favoritism: The Master equips Castor with the only polished weapon and leaves a frightened one-star with a cracked tool.",
    ),
    DialogueCase(
        "post_mission_loss",
        "wren_thelantern",
        """The party returns with the objective complete, but a named Hero is
        absent and the System gives the Master a bright loss notice. Wren has
        witnesses nearby and no power to undo death.""",
        "Let grief alter a relationship or commitment; do not soften or reverse the canonical loss.",
        canonical_fact="The party returns with its objective complete, but a named Hero is absent and the System records the loss.",
        pending_observation="evt_dialogue_post_mission_loss: The party returns with its objective complete, but a named Hero is absent and the System records the loss.",
    ),
    DialogueCase(
        "iselle_control",
        "iselle_the_guide",
        """The live account state block supplies current Gold, Gems, building
        resources, facility levels, catalogue prices, prerequisites, reward
        sources, available management actions, and deployment stamina. The
        Master has just finished an opening roster choice. Address the Master
        with one accurate provocative next choice grounded in that state.""",
        "Use current supplied state; offer one choice, not a vague menu or stale claim.",
        canonical_fact="The Master has just completed the opening roster choice; current account and facility state is available through the System channel.",
        pending_observation="evt_dialogue_iselle_control: The Master has just completed the opening roster choice; current account and facility state is available through the System channel.",
    ),
)


BLINDED_RUBRIC: dict[str, Any] = {
    "version": "one_star_dialogue_v1",
    "blind_fields": ["variant_id", "model"],
    "dimensions": [
        {
            "id": "speakability",
            "label": "speakability and naturalness",
            "question": "Could a person plausibly say this aloud in this moment without sounding like polished story analysis?",
        },
        {
            "id": "subtext",
            "label": "subtext",
            "question": "Do public words and actions leave room for the private motive instead of restating or explaining it?",
        },
        {
            "id": "voice_without_schtick",
            "label": "voice without schtick",
            "question": "Is the character recognizable through choices and rhythm without leaning on a repeated catchphrase, metaphor, or gimmick?",
        },
        {
            "id": "chemistry",
            "label": "conversational chemistry",
            "question": "Does the beat notice what another person actually did and leave them something specific to answer, resist, or misread?",
        },
        {
            "id": "multi_turn_variety",
            "label": "multi-turn variety",
            "question": "Across this character's sequence, do sentence shapes, gestures, motifs, and kinds of response change with the moment?",
        },
    ],
    "scale": (
        "When an actor has multiple samples, score each dimension 1-5 after "
        "reading that actor's whole sequence and note exact repeated shapes or "
        "public/private echoes. For a lone sample, mark multi-turn variety not "
        "applicable and score the remaining dimensions."
    ),
    "variant_gate": (
        "A variant passes when every actor sequence scores at least 4 on each "
        "applicable dimension and no character-level sequence regresses "
        "materially; Iselle must have no material voice or control regression."
    ),
    "selection_rule": "Use Luna when both pass, Terra when only Terra passes, and evaluate Sol only when both A/B variants fail.",
}


def _load_seed_checkpoint() -> CheckpointFile:
    """Load the exact story checkpoint used by One-Star production calls."""
    return CheckpointFile.model_validate_json(SEED_CHECKPOINT_PATH.read_text())


def _fixture_event(case: DialogueCase) -> CanonicalEventRecord:
    """Create one canonical surface event for the pending-observation fixture."""
    fact = case.canonical_fact.strip() or case.prompt.strip()
    submission_id = f"submission_dialogue_{case.case_id}"
    return CanonicalEventRecord(
        event_id=f"evt_dialogue_{case.case_id}",
        causal_lane_id=f"lane_dialogue_{case.actor_id}",
        effective_at_s=45,
        duration_s=1,
        actor_ids=[],
        source_submission_ids=[submission_id],
        feasible_submission_ids=[submission_id],
        infeasible_submission_ids=[],
        observable_facts=[ObservableFact.all(fact)],
        observers=ObserverGroups(
            direct=[case.actor_id],
            indirect=[],
            inferred=[],
        ),
        spawn=[],
        dormant=[],
        cull=[],
        commitment_opens=[],
        commitment_resolutions=[],
        commitment_interrupts=[],
        location_updates=[],
        activate=[],
    )


def _case_checkpoint(case: DialogueCase) -> CheckpointFile:
    """Clone the production seed and stage a real actor observation frame."""
    checkpoint = _load_seed_checkpoint()
    actor = next(
        character
        for character in checkpoint.characters
        if character.character_id == case.actor_id
    )
    actor.status = CharacterStatus.active
    actor.location = case.location
    observation = (
        case.pending_observation.strip()
        or f"evt_dialogue_{case.case_id}: "
        f"{case.canonical_fact.strip() or case.prompt.strip()}"
    )
    actor.pending_observations = [observation]
    checkpoint.session.leading_at_s = 60
    checkpoint.canonical_events.append(_fixture_event(case))
    return checkpoint


class _CapturedClient:
    """Capture the exact CharacterAgent request while remaining offline by default."""

    def __init__(
        self,
        *,
        model: str,
        fixed_response: str = "",
        delegate: Any = None,
    ) -> None:
        self.model = model
        self.fixed_response = fixed_response
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        started = time.perf_counter()
        if self.delegate is None:
            response = LLMResponse(
                content=self.fixed_response,
                model=self.model,
                usage={},
                assistant_content=[{"type": "text", "text": self.fixed_response}],
            )
        else:
            response = await self.delegate.complete(
                role=role,
                messages=messages,
                **kwargs,
            )
        self.calls.append({
            "role": role,
            "model": response.model or self.model,
            "messages": messages,
            "request": kwargs,
            "response": response.content,
            "usage": response.usage,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        })
        return response


async def _production_case_call(
    variant: DialogueVariant,
    case: DialogueCase,
    *,
    client: _CapturedClient,
) -> dict[str, Any]:
    """Run the actual CharacterAgent prompt builder for one staged seed case."""
    checkpoint = _case_checkpoint(case)
    actor = next(
        character
        for character in checkpoint.characters
        if character.character_id == case.actor_id
    )
    agent = CharacterAgent(client, PromptManager())
    await agent.draft_turn(
        actor,
        checkpoint,
        frame="foreground",
        local_context=case.prompt,
    )
    if not client.calls:
        raise RuntimeError(f"CharacterAgent did not call the client for {case.case_id}")
    return {
        "variant_id": variant.variant_id,
        "model": variant.model,
        "case_id": case.case_id,
        "actor_id": case.actor_id,
        "prompt": client.calls[0]["messages"],
        "contract": case.contract,
        "response": client.calls[-1]["response"],
        "provider_calls": list(client.calls),
        "fixture": {
            "seed_checkpoint": str(SEED_CHECKPOINT_PATH),
            "canonical_event_id": f"evt_dialogue_{case.case_id}",
            "pending_observation": (
                case.pending_observation.strip()
                or f"evt_dialogue_{case.case_id}: "
                f"{case.canonical_fact.strip() or case.prompt.strip()}"
            ),
        },
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_jsonable(row), ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def blind_samples(raw_calls: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove model identity while preserving review-relevant evidence."""
    blinded: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls, start=1):
        blinded.append({
            "blind_id": f"sample-{index:03d}",
            "case_id": call["case_id"],
            "actor_id": call["actor_id"],
            "prompt": call["prompt"],
            "contract": call["contract"],
            "response": call["response"],
        })
    return blinded


def _offline_response(variant: DialogueVariant, case: DialogueCase) -> str:
    """Stable fixture-like output for smoke runs, never a provider call."""
    del variant
    return (
        f"[{case.actor_id}] I make one concrete choice in {case.case_id}: "
        "I move before I explain, and I leave the consequence visible."
    )


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "raw_calls": output_dir / "raw" / "dialogue_calls.jsonl",
        "blinded_samples": output_dir / "blinded_samples.json",
        "rubric": output_dir / "blinded_rubric.json",
        "report": output_dir / "report.json",
    }


def run_offline_ab(
    output_dir: Path,
    *,
    variants: tuple[DialogueVariant, ...] = DEFAULT_VARIANTS,
    cases: tuple[DialogueCase, ...] = DEFAULT_CASES,
    responder: Callable[[DialogueVariant, DialogueCase], str] = _offline_response,
) -> dict[str, Any]:
    """Run deterministic A/B samples and persist raw plus blinded artifacts."""
    async def _run() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for variant in variants:
            for case in cases:
                response = responder(variant, case)
                client = _CapturedClient(
                    model=variant.model,
                    fixed_response=response,
                )
                row = await _production_case_call(
                    variant,
                    case,
                    client=client,
                )
                row.update({
                    "elapsed_ms": round(
                        sum(call["elapsed_ms"] for call in client.calls),
                        3,
                    ),
                    "usage": {},
                    "mode": "offline",
                })
                rows.append(row)
        return rows

    raw_calls = asyncio.run(_run())

    paths = _artifact_paths(output_dir)
    _write_jsonl(paths["raw_calls"], raw_calls)
    blinded = blind_samples(raw_calls)
    _write_json(paths["blinded_samples"], blinded)
    _write_json(paths["rubric"], BLINDED_RUBRIC)
    report = {
        "mode": "offline",
        "variants": [variant.__dict__ for variant in variants],
        "cases": [case.case_id for case in cases],
        "sample_count": len(raw_calls),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "review": {
            "blinded": True,
            "rubric": str(paths["rubric"]),
            "selection_rule": BLINDED_RUBRIC["selection_rule"],
        },
    }
    _write_json(paths["report"], report)
    return report


async def run_live_ab(
    output_dir: Path,
    *,
    variants: tuple[DialogueVariant, ...] = DEFAULT_VARIANTS,
    cases: tuple[DialogueCase, ...] = DEFAULT_CASES,
) -> dict[str, Any]:
    """Run A/B calls through the normal client after explicit ``--live``."""
    from app.llm.client import LLMClient
    from app.llm.config import LLMConfig

    raw_calls: list[dict[str, Any]] = []
    for variant in variants:
        config = LLMConfig.from_env()
        for role in (
            "agent",
            "agent_standard",
            "agent_convenience",
            "character_manager",
        ):
            config.role_models[role] = variant.model
            config.role_providers[role] = "openai"
        client = _CapturedClient(
            model=variant.model,
            delegate=LLMClient(config=config),
        )
        for case in cases:
            started = time.perf_counter()
            row = await _production_case_call(
                variant,
                case,
                client=client,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            row.update({
                "elapsed_ms": elapsed_ms,
                "usage": client.calls[-1]["usage"] if client.calls else {},
                "mode": "live",
            })
            raw_calls.append(row)
            client.calls.clear()

    paths = _artifact_paths(output_dir)
    _write_jsonl(paths["raw_calls"], raw_calls)
    blinded = blind_samples(raw_calls)
    _write_json(paths["blinded_samples"], blinded)
    _write_json(paths["rubric"], BLINDED_RUBRIC)
    report = {
        "mode": "live",
        "variants": [variant.__dict__ for variant in variants],
        "cases": [case.case_id for case in cases],
        "sample_count": len(raw_calls),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "review": {
            "blinded": True,
            "rubric": str(paths["rubric"]),
            "selection_rule": BLINDED_RUBRIC["selection_rule"],
        },
    }
    _write_json(paths["report"], report)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="use deterministic fixtures (default)")
    mode.add_argument("--live", action="store_true", help="call the configured LLM client")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "app/storage/playtest_reports/one_star_dialogue_ab",
    )
    parser.add_argument(
        "--include-sol",
        action="store_true",
        help="include the Sol fallback variant; normally review it only after both A/B fail",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    variants = DEFAULT_VARIANTS
    if args.include_sol:
        variants += (DialogueVariant("sol_fallback", SOL_MODEL),)
    if args.live:
        report = asyncio.run(run_live_ab(args.output_dir, variants=variants))
    else:
        report = run_offline_ab(args.output_dir, variants=variants)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
