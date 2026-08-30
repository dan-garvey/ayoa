from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

from app.schemas.conversation import ConversationMessage
from scripts.run_one_star_character_dialogue_replay import (
    PRODUCTION_SEED_PATH,
    REPLAY_TURNS,
    ReplayTurn,
    _offline_client,
    _render_profile_blocks,
    build_persisted_replay_messages,
    load_checkpoint,
    overlay_current_seed_profiles,
    public_private_overlap,
    run_replay,
)


def _write_synthetic_checkpoint(
    path: Path,
    *,
    with_replay_history: bool,
) -> Path:
    checkpoint = load_checkpoint(PRODUCTION_SEED_PATH)
    if with_replay_history:
        turn_counts = {"renna_holt": 4, "mirelle_voss": 7}
        for actor_id, turn_count in turn_counts.items():
            identity, state = _render_profile_blocks(checkpoint, actor_id)
            user = ConversationMessage(
                role="user",
                content=f"{identity}\n\n{state}",
            )
            assistant = ConversationMessage(
                role="assistant",
                content=(
                    f"{actor_id} makes the historical choice for this turn. "
                    "(I carry a distinct private concern forward.)"
                ),
            )
            checkpoint.character_conversations[actor_id] = [
                item
                for _ in range(turn_count)
                for item in (copy.deepcopy(user), copy.deepcopy(assistant))
            ]
    path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
    return path


def test_persisted_replay_keeps_checkpoint_history_order_and_cache_split(
    tmp_path: Path,
) -> None:
    checkpoint = load_checkpoint(
        _write_synthetic_checkpoint(
            tmp_path / "fixed.json",
            with_replay_history=True,
        )
    )
    turn = ReplayTurn(4, "mirelle_voss", 3)

    messages, historical_response = build_persisted_replay_messages(
        checkpoint,
        turn,
    )
    stored = checkpoint.character_conversations[turn.actor_id]
    current_role = next(
        character.public_sheet.role
        for character in checkpoint.characters
        if character.character_id == turn.actor_id
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1:] == [
        {"role": message.role, "content": message.content}
        for message in stored[:5]
    ]
    assert historical_response == stored[5].content
    assert "Mirelle Voss" not in messages[0]["content"]
    assert "Renna Holt" not in messages[0]["content"]
    assert current_role not in messages[0]["content"]
    assert current_role in messages[-1]["content"]


def test_checkpoint_replay_covers_four_renna_and_seven_mirelle_turns() -> None:
    by_actor = {
        actor_id: [turn.actor_turn for turn in REPLAY_TURNS if turn.actor_id == actor_id]
        for actor_id in {turn.actor_id for turn in REPLAY_TURNS}
    }
    assert by_actor == {
        "renna_holt": [1, 2, 3, 4],
        "mirelle_voss": [1, 2, 3, 4, 5, 6, 7],
    }


def test_current_seed_profile_replaces_old_profile_in_every_persisted_user(
    tmp_path: Path,
) -> None:
    replay = load_checkpoint(
        _write_synthetic_checkpoint(
            tmp_path / "replay.json",
            with_replay_history=True,
        )
    )
    profile = load_checkpoint(PRODUCTION_SEED_PATH)
    replay_actor = next(
        character
        for character in replay.characters
        if character.character_id == "mirelle_voss"
    )
    profile_actor = next(
        character
        for character in profile.characters
        if character.character_id == "mirelle_voss"
    )
    old_personality = replay_actor.personality
    old_intentions_enabled = replay_actor.private_state.intentions_enabled
    old_known_context = replay_actor.known_context
    sentinel = "CURRENT-SEED-PERSONALITY-SENTINEL"
    profile_actor.personality = sentinel
    profile_actor.private_state.intentions_enabled = not old_intentions_enabled
    profile_actor.known_context = "PROFILE-KNOWLEDGE-SENTINEL"

    overlaid = overlay_current_seed_profiles(replay, profile)
    messages, _historical = build_persisted_replay_messages(
        overlaid,
        ReplayTurn(4, "mirelle_voss", 3),
    )
    user_messages = [
        message["content"]
        for message in messages
        if message["role"] == "user"
    ]

    assert user_messages
    assert all(sentinel in message for message in user_messages)
    assert all(old_personality not in message for message in user_messages)
    assert replay_actor.private_state.intentions_enabled is old_intentions_enabled
    assert replay_actor.known_context == old_known_context
    assert all("PROFILE-KNOWLEDGE-SENTINEL" not in message for message in user_messages)


def test_offline_replay_writes_fixed_and_candidate_fed_evidence(tmp_path: Path) -> None:
    fixed_checkpoint_path = _write_synthetic_checkpoint(
        tmp_path / "fixed.json",
        with_replay_history=True,
    )
    relay_checkpoint_path = _write_synthetic_checkpoint(
        tmp_path / "relay.json",
        with_replay_history=False,
    )
    report = asyncio.run(
        run_replay(
            tmp_path,
            phase="offline-check",
            client=_offline_client(),
            mode="offline",
            checkpoint_path=fixed_checkpoint_path,
            relay_checkpoint_path=relay_checkpoint_path,
        )
    )

    assert report["fixed_context_sample_count"] == 11
    assert report["sequential_relay_sample_count"] == 4
    fixed_rows = [
        json.loads(line)
        for line in (tmp_path / "offline-check/raw/fixed_context_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    relay_rows = [
        json.loads(line)
        for line in (tmp_path / "offline-check/raw/sequential_relay_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(fixed_rows) == 11
    assert len(relay_rows) == 4
    for previous, current in zip(relay_rows, relay_rows[1:]):
        previous_public = previous["parsed"]["public_text"]
        assert previous_public in current["injected_observation"]
        assert previous_public in current["prompt"][-1]["content"]
    assert all(row["provider_calls"] for row in fixed_rows + relay_rows)


def test_public_private_overlap_flags_direct_explanatory_echo() -> None:
    echoed = public_private_overlap(
        'She shuts the red door and says, "The red door stays shut." '
        "(I need the red door to stay shut.)"
    )
    subtextual = public_private_overlap(
        'She palms the latch. "Try me." '
        "(I cannot let him discover that the room is empty.)"
    )

    assert echoed["overlap_of_shorter"] > subtextual["overlap_of_shorter"]
