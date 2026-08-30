#!/usr/bin/env python3
# ruff: noqa: E402 -- executable script adds the repository root to sys.path.
"""Replay One-Star CharacterAgent dialogue without the router or narrator.

The fixed-context lane replays the committed Renna/Mirelle message lineages
from session ``42t24t``.  Each call keeps the persisted role/order/event
history through that turn while overlaying current production-seed profile
blocks.  The relay lane starts from the post-clear lobby checkpoint 0007 and
uses the production CharacterAgent prompt builder and commit path for four
alternating beats.  Each later relay prompt therefore sees the
candidate-authored public beat produced immediately before it.

Importing the module and the default command are offline.  ``--live`` is the
only path that calls the configured provider.  Every run saves raw prompts,
responses, timings, usage, and the historical comparison response.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.engine.character_agent import CharacterAgent
from app.engine.context_builder import (
    build_character_packet,
    build_character_state,
    build_world_context,
)
from app.engine.prompt_manager import PromptManager
from app.schemas.checkpoint import CheckpointFile
from scripts.run_one_star_dialogue_ab import (
    BLINDED_RUBRIC,
    TERRA_MODEL,
    _CapturedClient,
    _write_json,
    _write_jsonl,
)


PRODUCTION_SEED_PATH = (
    REPO_ROOT
    / "app"
    / "storage"
    / "stories"
    / "one_star_ascension_s1"
    / "ckpt_0000.json"
)


@dataclass(frozen=True)
class ReplayTurn:
    """One persisted CharacterAgent turn in global story order."""

    sequence_index: int
    actor_id: str
    actor_turn: int


REPLAY_TURNS: tuple[ReplayTurn, ...] = (
    ReplayTurn(1, "renna_holt", 1),
    ReplayTurn(2, "mirelle_voss", 1),
    ReplayTurn(3, "mirelle_voss", 2),
    ReplayTurn(4, "mirelle_voss", 3),
    ReplayTurn(5, "renna_holt", 2),
    ReplayTurn(6, "mirelle_voss", 4),
    ReplayTurn(7, "mirelle_voss", 5),
    ReplayTurn(8, "renna_holt", 3),
    ReplayTurn(9, "mirelle_voss", 6),
    ReplayTurn(10, "mirelle_voss", 7),
    ReplayTurn(11, "renna_holt", 4),
)

RELAY_ACTORS: tuple[str, ...] = (
    "renna_holt",
    "mirelle_voss",
    "renna_holt",
    "mirelle_voss",
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_TRAILING_INTENT_RE = re.compile(r"\((?P<intent>[^()]*)\)\s*$", re.DOTALL)
_IDENTITY_BLOCK_RE = re.compile(
    r"<character_identity>.*?</character_identity>",
    re.DOTALL,
)
_CURRENT_STATE_BLOCK_RE = re.compile(
    r"<current_state>.*?</current_state>",
    re.DOTALL,
)
_PRESENTATION_FOOTER_RE = re.compile(
    r"<presentation>.*?</presentation>",
    re.DOTALL,
)
_METRIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "is", "it", "its", "me", "my", "not", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "them",
    "they", "this", "to", "was", "we", "were", "will", "with", "you",
}


class _ReplayPromptManager(PromptManager):
    """Return one checkpoint-derived message list to CharacterAgent."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        super().__init__()
        self._messages = copy.deepcopy(messages)

    def render_conversation(
        self,
        template_name: str,
        history: list[Any],
        **variables: Any,
    ) -> list[dict[str, Any]]:
        del history, variables
        if template_name != "agent":
            raise ValueError(f"Replay only supports the agent template, got {template_name!r}")
        return copy.deepcopy(self._messages)


def load_checkpoint(path: Path) -> CheckpointFile:
    """Load one checkpoint without mutating the source artifact."""
    return CheckpointFile.model_validate_json(path.read_text(encoding="utf-8"))


def _character(checkpoint: CheckpointFile, actor_id: str) -> Any:
    return next(
        character
        for character in checkpoint.characters
        if character.character_id == actor_id
    )


def _render_profile_blocks(
    profile_checkpoint: CheckpointFile,
    actor_id: str,
    *,
    prompt_manager: PromptManager | None = None,
) -> tuple[str, str]:
    """Render identity/private-state blocks through the production template."""
    manager = prompt_manager or PromptManager()
    actor = _character(profile_checkpoint, actor_id)
    messages = manager.render_messages(
        "agent",
        agent_ruleset_system_addon=manager.render(
            "agent_ruleset_one_star"
        ).strip(),
        **build_character_packet(actor, profile_checkpoint),
        **build_character_state(actor, profile_checkpoint),
        world_context=build_world_context(actor, profile_checkpoint),
        pending_observations_block="",
        mode_header="## AGENT-TURN",
        mode_block="## Turn Frame\nforeground",
    )
    user_content = messages[-1]["content"]
    identity_match = _IDENTITY_BLOCK_RE.search(user_content)
    state_match = _CURRENT_STATE_BLOCK_RE.search(user_content)
    if identity_match is None or state_match is None:
        raise ValueError(f"Could not render current profile blocks for {actor_id}")
    return identity_match.group(0), state_match.group(0)


def _replace_profile_blocks(
    content: str,
    *,
    identity_block: str,
    state_block: str,
) -> str:
    """Replace blocks in-place without adding, dropping, or reordering messages."""
    updated, identity_count = _IDENTITY_BLOCK_RE.subn(identity_block, content)
    updated, state_count = _CURRENT_STATE_BLOCK_RE.subn(state_block, updated)
    if identity_count != 1 or state_count != 1:
        raise ValueError(
            "Persisted CharacterAgent user message must contain exactly one "
            "identity block and one current-state block"
        )
    return updated


def overlay_current_seed_profiles(
    checkpoint: CheckpointFile,
    profile_checkpoint: CheckpointFile,
    *,
    actor_ids: tuple[str, ...] = ("renna_holt", "mirelle_voss"),
) -> CheckpointFile:
    """Transplant current seed profiles while preserving runtime/history shape."""
    for actor_id in actor_ids:
        target = _character(checkpoint, actor_id)
        source = _character(profile_checkpoint, actor_id)
        target.public_sheet.role = source.public_sheet.role
        target.backstory = source.backstory
        target.personality = source.personality
        target.private_state.goals = copy.deepcopy(source.private_state.goals)
        target.private_state.current_objectives = copy.deepcopy(
            source.private_state.current_objectives
        )
        target.private_state.secrets = copy.deepcopy(
            source.private_state.secrets
        )

        identity_block, state_block = _render_profile_blocks(
            checkpoint,
            actor_id,
        )
        for message in checkpoint.character_conversations.get(actor_id, []):
            if message.role != "user":
                continue
            if not isinstance(message.content, str):
                raise ValueError("CharacterAgent user history must be plain text")
            message.content = _replace_profile_blocks(
                message.content,
                identity_block=identity_block,
                state_block=state_block,
            )
    return checkpoint


def build_persisted_replay_messages(
    checkpoint: CheckpointFile,
    turn: ReplayTurn,
    *,
    prompt_manager: PromptManager | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Build ``[system, *prior history, current user]`` from ``42t24t``.

    Production intentionally omits volatile One-Star state and presentation
    catalogs when committing the user message to character history.  This
    controlled replay uses that committed lineage exactly; it does not claim
    to reconstruct transient blocks that were never stored.
    """
    manager = prompt_manager or PromptManager()
    addon = manager.render("agent_ruleset_one_star").strip()
    system = manager.render_system_message(
        "agent",
        agent_ruleset_system_addon=addon,
    )
    conversation = checkpoint.character_conversations[turn.actor_id]
    user_index = (turn.actor_turn - 1) * 2
    assistant_index = user_index + 1
    if assistant_index >= len(conversation):
        raise IndexError(
            f"Missing {turn.actor_id} turn {turn.actor_turn} in checkpoint history"
        )
    if conversation[user_index].role != "user":
        raise ValueError(f"Expected user message at history index {user_index}")
    if conversation[assistant_index].role != "assistant":
        raise ValueError(f"Expected assistant message at history index {assistant_index}")
    messages: list[dict[str, Any]] = [system]
    messages.extend(
        {"role": item.role, "content": copy.deepcopy(item.content)}
        for item in conversation[: user_index + 1]
    )
    return messages, copy.deepcopy(conversation[assistant_index].content)


def split_public_intent(text: str) -> tuple[str, str]:
    """Split the final private intent for review metrics without engine mutation."""
    match = _TRAILING_INTENT_RE.search(text or "")
    if match is None:
        return (text or "").strip(), ""
    return (text or "")[: match.start()].rstrip(), match.group("intent").strip()


def public_private_overlap(text: str) -> dict[str, Any]:
    """Report lexical public/private echo as a review aid, not an auto-gate."""
    public, intent = split_public_intent(text)
    public = _PRESENTATION_FOOTER_RE.sub("", public)

    def significant_tokens(value: str) -> list[str]:
        return [
            token.lower()
            for token in _WORD_RE.findall(value)
            if token.lower() not in _METRIC_STOPWORDS
        ]

    public_tokens = significant_tokens(public)
    intent_tokens = significant_tokens(intent)
    public_counts = Counter(public_tokens)
    intent_counts = Counter(intent_tokens)
    shared_count = sum((public_counts & intent_counts).values())
    denominator = min(len(public_tokens), len(intent_tokens))
    return {
        "public_token_count": len(public_tokens),
        "intent_token_count": len(intent_tokens),
        "shared_token_count": shared_count,
        "overlap_of_shorter": round(shared_count / denominator, 3)
        if denominator
        else 0.0,
    }


def repeated_actor_trigrams(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Surface repeated three-word motifs across one actor's generated sequence."""
    by_actor: dict[str, Counter[tuple[str, str, str]]] = {}
    for row in rows:
        actor_id = str(row["actor_id"])
        public, _intent = split_public_intent(str(row["response"]))
        public = _PRESENTATION_FOOTER_RE.sub("", public)
        tokens = [token.lower() for token in _WORD_RE.findall(public)]
        trigrams = set(zip(tokens, tokens[1:], tokens[2:]))
        by_actor.setdefault(actor_id, Counter()).update(trigrams)
    return {
        actor_id: [
            {"phrase": " ".join(trigram), "turn_count": count}
            for trigram, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count > 1
        ]
        for actor_id, counts in by_actor.items()
    }


async def _run_fixed_calls(
    checkpoint: CheckpointFile,
    client: _CapturedClient,
    *,
    phase: str,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in REPLAY_TURNS:
        messages, historical_response = build_persisted_replay_messages(
            checkpoint,
            turn,
        )
        actor = _character(checkpoint, turn.actor_id)
        agent = CharacterAgent(client, _ReplayPromptManager(messages))
        started = time.perf_counter()
        draft = await agent.draft_turn(actor, checkpoint, frame="foreground")
        calls = copy.deepcopy(client.calls)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        raw_response = calls[-1]["response"]
        rows.append({
            "phase": phase,
            "sequence_index": turn.sequence_index,
            "actor_id": turn.actor_id,
            "actor_turn": turn.actor_turn,
            "checkpoint_path": str(checkpoint_path),
            "input_contract": (
                "persisted CharacterAgent role/order/event history with "
                "current production profile blocks"
            ),
            "prompt": messages,
            "historical_response": historical_response,
            "response": raw_response,
            "parsed": draft.output.model_dump(mode="json"),
            "public_private_overlap": public_private_overlap(raw_response),
            "elapsed_ms": elapsed_ms,
            "usage": agent.last_usage,
            "provider_calls": calls,
        })
        client.calls.clear()
    return rows


async def _run_sequential_relay(
    checkpoint: CheckpointFile,
    client: _CapturedClient,
    *,
    phase: str,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    agent = CharacterAgent(client, PromptManager())
    previous_actor_id = ""
    previous_public = ""
    for relay_turn, actor_id in enumerate(RELAY_ACTORS, start=1):
        actor = _character(checkpoint, actor_id)
        injected_observation = ""
        if previous_public:
            previous_actor = _character(checkpoint, previous_actor_id)
            injected_observation = (
                f"replay_relay_{relay_turn - 1}: "
                f"{previous_actor.name}'s immediately preceding observable beat: "
                f"{previous_public}"
            )
            actor.pending_observations.append(injected_observation)
        started = time.perf_counter()
        draft = await agent.draft_turn(actor, checkpoint, frame="foreground")
        calls = copy.deepcopy(client.calls)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        raw_response = calls[-1]["response"]
        agent.commit_draft(actor, checkpoint, draft)
        rows.append({
            "phase": phase,
            "relay_turn": relay_turn,
            "actor_id": actor_id,
            "checkpoint_path": str(checkpoint_path),
            "input_contract": "candidate-fed production CharacterAgent relay",
            "injected_observation": injected_observation,
            "prompt": calls[0]["messages"],
            "response": raw_response,
            "parsed": draft.output.model_dump(mode="json"),
            "public_private_overlap": public_private_overlap(raw_response),
            "elapsed_ms": elapsed_ms,
            "usage": agent.last_usage,
            "provider_calls": calls,
        })
        previous_actor_id = actor_id
        previous_public = draft.output.public_text
        client.calls.clear()
    return rows


def _artifact_paths(output_dir: Path, phase: str) -> dict[str, Path]:
    phase_dir = output_dir / phase
    return {
        "fixed_calls": phase_dir / "raw" / "fixed_context_calls.jsonl",
        "sequential_calls": phase_dir / "raw" / "sequential_relay_calls.jsonl",
        "review_samples": phase_dir / "review_samples.json",
        "rubric": phase_dir / "review_rubric.json",
        "report": phase_dir / "report.json",
    }


async def run_replay(
    output_dir: Path,
    *,
    phase: str,
    client: _CapturedClient,
    mode: str,
    checkpoint_path: Path,
    relay_checkpoint_path: Path,
    profile_checkpoint_path: Path = PRODUCTION_SEED_PATH,
) -> dict[str, Any]:
    """Run the fixed comparison and candidate-fed relay, then save evidence."""
    profile_checkpoint = load_checkpoint(profile_checkpoint_path)
    fixed_rows = await _run_fixed_calls(
        overlay_current_seed_profiles(
            load_checkpoint(checkpoint_path),
            profile_checkpoint,
        ),
        client,
        phase=phase,
        checkpoint_path=checkpoint_path,
    )
    sequential_rows = await _run_sequential_relay(
        overlay_current_seed_profiles(
            load_checkpoint(relay_checkpoint_path),
            profile_checkpoint,
        ),
        client,
        phase=phase,
        checkpoint_path=relay_checkpoint_path,
    )
    paths = _artifact_paths(output_dir, phase)
    _write_jsonl(paths["fixed_calls"], fixed_rows)
    _write_jsonl(paths["sequential_calls"], sequential_rows)
    review_samples = [
        {
            "sample_id": f"fixed-{row['sequence_index']:02d}",
            "lane": "fixed_context",
            "actor_id": row["actor_id"],
            "actor_turn": row["actor_turn"],
            "historical_response": row["historical_response"],
            "response": row["response"],
            "public_private_overlap": row["public_private_overlap"],
        }
        for row in fixed_rows
    ]
    review_samples.extend(
        {
            "sample_id": f"relay-{row['relay_turn']:02d}",
            "lane": "candidate_fed_relay",
            "actor_id": row["actor_id"],
            "response": row["response"],
            "public_private_overlap": row["public_private_overlap"],
        }
        for row in sequential_rows
    )
    _write_json(paths["review_samples"], review_samples)
    _write_json(paths["rubric"], BLINDED_RUBRIC)
    report = {
        "phase": phase,
        "mode": mode,
        "model": client.model,
        "profile_checkpoint_path": str(profile_checkpoint_path),
        "fixed_context_sample_count": len(fixed_rows),
        "sequential_relay_sample_count": len(sequential_rows),
        "fixed_context_contract": (
            "Each call preserves the persisted 42t24t CharacterAgent role, "
            "order, event input, and assistant history while overlaying the "
            "current production-seed identity/private-state blocks. Candidate "
            "output is not fed into later fixed-context calls."
        ),
        "sequential_relay_contract": (
            "Four alternating calls start from post-clear lobby ckpt_0007; "
            "every later prompt "
            "includes the previous candidate public beat and uses the production "
            "CharacterAgent commit path. No router or narrator is called."
        ),
        "repeated_public_trigrams": {
            "fixed_context": repeated_actor_trigrams(fixed_rows),
            "sequential_relay": repeated_actor_trigrams(sequential_rows),
        },
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["report"], report)
    return report


def _offline_client() -> _CapturedClient:
    return _CapturedClient(
        model=TERRA_MODEL,
        fixed_response=(
            "The character makes one observable choice without explaining it. "
            "(I keep the motive private and carry it into the next beat.)"
        ),
    )


def _live_client() -> _CapturedClient:
    from app.llm.client import LLMClient
    from app.llm.config import LLMConfig

    config = LLMConfig.from_env()
    for role in (
        "agent",
        "agent_standard",
        "agent_convenience",
        "character_manager",
    ):
        config.role_models[role] = TERRA_MODEL
        config.role_providers[role] = "openai"
    return _CapturedClient(
        model=TERRA_MODEL,
        delegate=LLMClient(config=config),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="use deterministic fixtures (default)")
    mode.add_argument("--live", action="store_true", help="call production Terra through the configured client")
    parser.add_argument("--phase", required=True, help="artifact subdirectory label, such as baseline or final")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="shared artifact root; phase outputs are written beneath it",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint containing the persisted character histories to replay",
    )
    parser.add_argument(
        "--relay-checkpoint",
        type=Path,
        required=True,
        help="checkpoint used to start the candidate-fed conversation",
    )
    parser.add_argument(
        "--profile-checkpoint",
        type=Path,
        default=PRODUCTION_SEED_PATH,
        help="current production seed whose identity/private-state fields overlay the replay",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    client = _live_client() if args.live else _offline_client()
    report = asyncio.run(
        run_replay(
            args.output_dir,
            phase=args.phase,
            client=client,
            mode="live" if args.live else "offline",
            checkpoint_path=args.checkpoint,
            relay_checkpoint_path=args.relay_checkpoint,
            profile_checkpoint_path=args.profile_checkpoint,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
