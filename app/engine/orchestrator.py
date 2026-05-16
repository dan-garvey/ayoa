"""Turn orchestrator — v11 beat loop binding.

The old v8 pipeline ran a single-pass turn (EventRouter → agents → Narrator)
and returned one rendered output. v11 shifts to a beat-cascading state
machine in `app.engine.turn_loop.run_beat`; this module is now a thin
adapter that:

  1. loads the checkpoint,
  2. resolves which character is acting,
  3. acquires the per-session act-slot lock so concurrent /acts serialize,
  4. validates the incoming /act against that slot,
  5. runs one beat to completion via `run_beat`,
  6. applies roster side-effects of every event that closed this beat,
  7. saves the checkpoint,
  8. returns a `TurnResponse` carrying per-POV renders.

The only LLM-facing object the orchestrator constructs directly is the
`LLMDispatcher` — the single adapter that binds the router, narrator,
and character_agent modules into the protocol `run_beat` depends on.

"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from app.engine.character_agent import (
    CharacterAgent,
    CharacterAgentTurnDraft,
    model_role_for_character,
)
from app.engine.context_builder import clear_character_inbox, collect_player_ids
from app.engine.character_manager import CharacterManager, _pinned_character_ids
from app.engine.checkpoint_manager import CheckpointManager
from app.engine import dnd_combat, dnd_inventory
from app.engine.dnd_cat_ii import (
    DndCatIIRollsPending,
    complete_pending_player_roll,
    pending_player_rolls,
    roll_transaction_source,
)
from app.engine.dnd_combat_access import (
    checkpoint_active_combat as _active_combat_state,
    combatant_character_id as _combatant_character_id,
    combatant_for_character as _combatant_for_character,
    combatant_name as _combatant_name,
    combatants as _combatants,
    current_combatant as _current_combatant_state,
    obj_get as _obj_get,
    obj_set as _obj_set,
)
from app.engine.dnd_roll_display import (
    completed_automatic_roll_keys,
    dice_roll_displays_since,
)
from app.engine.model_config_sync import sync_checkpoint_runtime_models
from app.engine.prompt_manager import PromptManager
from app.engine.text_safety import strip_terminal_control
from app.engine.turn_loop import (
    BeatResult,
    SessionLockManager,
    SlotConflict,
    broadcast_event,
    cat_ii_is_ready,
    check_act_slot,
    close_cat_ii,
    format_slot_rejection,
    release_character_slot,
    release_beat_slots,
    run_beat,
    _agent_intention_for_dispatch,
    align_cat_ii_resolution_time,
    _clear_pending_initiating_action,
    _end_beat,
    flush_combat_visible_facts,
    _filter_picks_for_dispatch,
)
from app.llm.client import LLMClient

# Imported at module level so tests can monkeypatch
# `app.engine.orchestrator.LLMDispatcher` without reaching into the
# adapter module directly. Hard import now that the Dispatcher has
# landed — a missing module is a real packaging error.
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterAgentTier, CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.requests import TurnRequest
from app.schemas.responses import TurnResponse
from app.schemas.state import CommitmentRevisionPrompt, PendingNarratorRender

logger = logging.getLogger(__name__)


# Commit 5: hard ceiling on background-tick concurrency, regardless of
# what `SessionSettings.tick_concurrency` is configured to. Sized for
# Anthropic's per-minute API limits on Haiku — a hollowstone-sized
# roster (~12 NPCs) all eligible at once would still fan out under
# this. Raise only after measuring rate-limit headroom.
TICK_CONCURRENCY_HARD_CAP = 16
TICK_BATCH_HARD_CAP = 8
TICK_CUE_FIRE_THRESHOLD = 10
TICK_PREMIUM_PER_BATCH = 1


@dataclass(frozen=True)
class TickCandidate:
    character: CharacterRecord
    score: float
    cue_score: float
    roster_index: int
    reason: str

_COMBAT_NO_ADVANCE_REASONS = {
    "slot_rejected",
    "cat_ii_pending",
    "cat_ii_pending_rolls",
    "cat_ii_stale",
    "combat_started",
    "combat_reaction_pending",
}


def _append_transcript_entry(
    ckpt: CheckpointFile,
    beat_result: BeatResult,
    preferred_actor_id: str,
) -> None:
    """v11-r7f: persist one POV's transcript entry to ckpt.transcript.

    Why: ckpt.transcript was a write-never field for the entire v11
    pipeline — pre-r7f the dispatcher returned only final_text from
    narrator.compose_pov_render and dropped the structured envelope's
    transcript_entry on the floor. /history rendered "(no turns yet)"
    after every play session and engine_bridge's recent-session
    summary stayed empty forever. Now that BeatResult carries a
    {character_id: TranscriptEntry} map, pick one entry per beat as
    the canonical session log line.

    Selection: prefer the acting actor's POV (the player who /act'd
    or whose Cat II is being adjudicated). Fall back to the first
    available render. No-op when no human rendered (Cat II pending,
    or beat-with-no-renderable-events) — ckpt.transcript stays at
    its prior length.

    Caveat: with multi-human play, only one POV's transcript_entry
    enters the global log per beat. The others' POV prose still
    lives in narrator_conversations[h] (their per-character rolling
    history), so nothing is lost — the global log just becomes the
    acting-player view. Multi-POV transcript layout is a separate
    schema decision deferred until we actually ship multi-human.
    """
    entries = beat_result.transcript_entries
    if not entries:
        return
    entry = entries.get(preferred_actor_id)
    if entry is None:
        # Fallback: first available POV's entry.
        entry = next(iter(entries.values()))
    ckpt.transcript.append(entry)


def _combatant_human_controlled(
    ckpt: CheckpointFile,
    combatant: Any,
) -> bool:
    from app.engine.context_builder import collect_player_ids

    return (
        _combatant_character_id(combatant) in collect_player_ids(ckpt)
        or bool(_obj_get(combatant, "player_controlled", False))
    )


def _current_combatant(ckpt: CheckpointFile, combat: Any) -> Any | None:
    return _current_combatant_state(combat)


def _combat_actor_is_human_controlled(
    ckpt: CheckpointFile,
    actor_id: str,
    combat: Any,
) -> bool:
    from app.engine.context_builder import collect_player_ids

    combatant = _combatant_for_character(combat, actor_id)
    if combatant is None:
        return False
    if actor_id in collect_player_ids(ckpt):
        return True
    return bool(_obj_get(combatant, "player_controlled", False))


def _combat_rejection_message(
    *,
    ckpt: CheckpointFile,
    acting_id: str,
    current: Any | None,
    attempted_text: str,
) -> str:
    current_name = (
        _combatant_name(current)
        if current is not None
        else "the current combatant"
    )
    actor_name = acting_id
    for char in ckpt.characters:
        if char.character_id == acting_id:
            actor_name = char.name
            break

    message = (
        f"It is **{current_name}**'s initiative turn. "
        f"Wait for **{current_name}** to finish before **{actor_name}** acts."
    )
    if attempted_text:
        preview = strip_terminal_control(attempted_text).strip()
        if len(preview) > 1500:
            preview = preview[:1497] + "..."
        message += f"\n\n> Your submitted text:\n> {preview}"
    return message


def _combat_turn_rejection(
    ckpt: CheckpointFile,
    acting_id: str,
    attempted_text: str,
) -> str | None:
    combat = _active_combat_state(ckpt)
    if combat is None:
        return None
    if not _combat_actor_is_human_controlled(ckpt, acting_id, combat):
        return None
    current = _current_combatant(ckpt, combat)
    current_id = _combatant_character_id(current) if current is not None else ""
    if current_id == acting_id:
        return None
    return _combat_rejection_message(
        ckpt=ckpt,
        acting_id=acting_id,
        current=current,
        attempted_text=attempted_text,
    )


def _begin_combat_turn(combatant: Any) -> None:
    _obj_set(combatant, "reaction_available", True)


def _set_pending_combat_advance(
    ckpt: CheckpointFile,
    actor_id: str,
) -> None:
    combat = _active_combat_state(ckpt)
    if combat is None:
        return
    _obj_set(combat, "pending_advance_actor_id", actor_id)


def _pending_combat_advance_actor_id(ckpt: CheckpointFile) -> str:
    combat = _active_combat_state(ckpt)
    if combat is None:
        return ""
    return str(_obj_get(combat, "pending_advance_actor_id", "") or "")


def _clear_pending_combat_advance(ckpt: CheckpointFile) -> None:
    combat = _active_combat_state(ckpt)
    if combat is not None:
        _obj_set(combat, "pending_advance_actor_id", "")


def _blocking_slots_open(ckpt: CheckpointFile) -> bool:
    return any(
        entry.reason in {
            "initiator",
            "cat_ii_responder",
            "cat_ii_roll",
            "combat_reaction",
        }
        for entry in ckpt.session.active_act_slots.values()
    )


def _append_combat_audit_line(ckpt: CheckpointFile, line: str) -> None:
    combat = _active_combat_state(ckpt)
    if isinstance(combat, dict):
        combat.setdefault("audit_lines", []).append(line)
        return

    appender = getattr(dnd_combat, "append_audit_line", None) or getattr(
        dnd_combat, "append_combat_audit", None
    )
    if appender is not None:
        try:
            appender(ckpt.session, line)
        except (AttributeError, TypeError, ValueError):
            appender(combat, line)
        return

    audit_lines = getattr(combat, "audit_lines", None)
    if isinstance(audit_lines, list):
        audit_lines.append(line)
        return

    logger.error("Unable to append D&D combat audit line: %s", line)


def _beat_should_advance_combat(
    ckpt: CheckpointFile,
    acting_id: str,
    beat_result: BeatResult,
) -> bool:
    combat = _active_combat_state(ckpt)
    if combat is None:
        return False
    if beat_result.ended_reason in _COMBAT_NO_ADVANCE_REASONS:
        return False
    current = _current_combatant(ckpt, combat)
    return _combatant_character_id(current) == acting_id


def _beat_can_delay_combat_advance(
    ckpt: CheckpointFile,
    acting_id: str,
    beat_result: BeatResult,
) -> bool:
    combat = _active_combat_state(ckpt)
    if combat is None or not beat_result.renders:
        return False
    current = _current_combatant(ckpt, combat)
    return _combatant_character_id(current) == acting_id


def _advance_combat_initiative_after_turn(
    ckpt: CheckpointFile,
) -> None:
    combat = _active_combat_state(ckpt)
    if combat is None:
        return
    try:
        current = dnd_combat.advance_turn_with_effects(
            ckpt.session,
            characters=ckpt.characters,
        )
    except ValueError:
        return
    dnd_combat.sync_combat_effects_to_characters(
        ckpt.session.active_combat,
        ckpt.characters,
    )
    _clear_pending_combat_advance(ckpt)
    current = _current_combatant(ckpt, combat) or current
    if current is not None:
        _append_combat_audit_line(
            ckpt,
            f"Initiative advanced to {_combatant_name(current)}.",
        )


def _advance_pending_combat_if_unblocked(ckpt: CheckpointFile) -> bool:
    pending_actor = _pending_combat_advance_actor_id(ckpt)
    if not pending_actor or _blocking_slots_open(ckpt):
        return False
    combat = _active_combat_state(ckpt)
    if combat is None:
        _clear_pending_combat_advance(ckpt)
        return False
    current = _current_combatant(ckpt, combat)
    if _combatant_character_id(current) != pending_actor:
        _clear_pending_combat_advance(ckpt)
        return False
    _advance_combat_initiative_after_turn(ckpt)
    return True


def _handle_combat_after_beat(
    ckpt: CheckpointFile,
    *,
    acting_id: str,
    beat_result: BeatResult,
    allow_new_pending: bool = True,
) -> None:
    reaction_prompts = beat_result.reaction_prompts or {}
    if reaction_prompts and allow_new_pending and _beat_can_delay_combat_advance(
        ckpt, acting_id, beat_result
    ):
        _set_pending_combat_advance(ckpt, acting_id)
        return
    if _beat_should_advance_combat(ckpt, acting_id, beat_result):
        _advance_combat_initiative_after_turn(ckpt)
        return
    _advance_pending_combat_if_unblocked(ckpt)


def _merge_render_maps(
    target: dict[str, str],
    source: dict[str, str],
) -> None:
    for cid, text in source.items():
        if not text:
            continue
        existing = target.get(cid, "")
        target[cid] = f"{existing}\n\n{text}" if existing else text


def _combine_beat_renders(results: list[BeatResult]) -> dict[str, str]:
    combined: dict[str, str] = {}
    for result in results:
        _merge_render_maps(combined, dict(result.renders or {}))
    return combined


def _combine_loot_prompts(results: list[BeatResult]) -> dict[str, list[str]]:
    combined: dict[str, list[str]] = {}
    for result in results:
        for cid, offer_ids in (result.loot_prompts or {}).items():
            if not offer_ids:
                continue
            bucket = combined.setdefault(cid, [])
            for offer_id in offer_ids:
                if offer_id not in bucket:
                    bucket.append(offer_id)
    return combined


def _drain_experience_awards(ckpt: CheckpointFile):
    return dnd_combat.drain_pending_experience_awards(ckpt.session)


def _automated_turn_snapshot(ckpt: CheckpointFile) -> dict[str, Any]:
    return {
        "canonical_events": len(ckpt.canonical_events),
        "render_buffers": {
            cid: len(buffer)
            for cid, buffer in ckpt.session.render_buffers.items()
        },
        "open_cat_ii_events": len(ckpt.session.open_cat_ii_events),
        "active_act_slots": dict(ckpt.session.active_act_slots),
        "session_conversation": len(ckpt.session_conversation),
        "narrator_conversations": {
            cid: len(history)
            for cid, history in ckpt.narrator_conversations.items()
        },
        "character_conversations": {
            cid: len(history)
            for cid, history in ckpt.character_conversations.items()
        },
    }


def _rollback_automated_turn_snapshot(
    ckpt: CheckpointFile,
    snapshot: dict[str, Any],
) -> None:
    del ckpt.canonical_events[snapshot["canonical_events"]:]

    render_lengths = snapshot["render_buffers"]
    for cid in list(ckpt.session.render_buffers):
        if cid not in render_lengths:
            del ckpt.session.render_buffers[cid]
            continue
        del ckpt.session.render_buffers[cid][render_lengths[cid]:]

    del ckpt.session.open_cat_ii_events[snapshot["open_cat_ii_events"]:]
    ckpt.session.active_act_slots = dict(snapshot["active_act_slots"])
    del ckpt.session_conversation[snapshot["session_conversation"]:]

    narrator_lengths = snapshot["narrator_conversations"]
    for cid in list(ckpt.narrator_conversations):
        if cid not in narrator_lengths:
            del ckpt.narrator_conversations[cid]
            continue
        del ckpt.narrator_conversations[cid][narrator_lengths[cid]:]

    character_lengths = snapshot["character_conversations"]
    for cid in list(ckpt.character_conversations):
        if cid not in character_lengths:
            del ckpt.character_conversations[cid]
            continue
        del ckpt.character_conversations[cid][character_lengths[cid]:]


def _commitment_revision_prompts(
    ckpt: CheckpointFile,
) -> dict[str, list[str]]:
    prompts: dict[str, list[str]] = {}
    for cid, prompt in (ckpt.session.pending_commitment_revisions or {}).items():
        if not prompt.commitment_id:
            continue
        prompts[cid] = [prompt.commitment_id]
    return prompts


def _combined_beat_reason(results: list[BeatResult]) -> str:
    if any(result.ended_reason == "combat_started" for result in results):
        return "combat_started"
    return results[-1].ended_reason if results else ""


def _turn_response_from_beat_results(
    *,
    session_id: str,
    ckpt: CheckpointFile,
    acting_id: str,
    beat_results: list[BeatResult],
    roll_keys_before: set[tuple[str, str]],
) -> TurnResponse | None:
    if not beat_results:
        return None
    per_player = _combine_beat_renders(beat_results)
    final_result = beat_results[-1]
    output_text = per_player.get(acting_id, "")
    return TurnResponse(
        session_id=session_id,
        checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
        turn_index=ckpt.session.turn_index,
        output_text=output_text,
        per_player_renders=per_player,
        beat_ended_reason=_combined_beat_reason(beat_results),
        reaction_prompts=final_result.reaction_prompts or {},
        loot_prompts=_combine_loot_prompts(beat_results),
        commitment_revision_prompts=_commitment_revision_prompts(ckpt),
        dice_rolls=dice_roll_displays_since(ckpt, roll_keys_before),
        experience_awards=_drain_experience_awards(ckpt),
    )


def _with_pre_turn_resolutions(
    response: TurnResponse,
    pre_turn: list[TurnResponse],
) -> TurnResponse:
    if pre_turn:
        response.pre_turn_resolutions = [
            *pre_turn,
            *(response.pre_turn_resolutions or []),
        ]
    return response


def _should_resume_automated_combat_before_act(
    ckpt: CheckpointFile,
    acting_id: str,
) -> bool:
    if ckpt.session.active_act_slots:
        return False
    combat = _active_combat_state(ckpt)
    if combat is None:
        return False
    if not _combat_actor_is_human_controlled(ckpt, acting_id, combat):
        return False
    current = _current_combatant(ckpt, combat)
    if current is None:
        return False
    return not _combatant_human_controlled(ckpt, current)


def _combat_roll_transaction(ckpt: CheckpointFile, event_id: str) -> Any | None:
    for txn in ckpt.session.cat_ii_roll_transactions:
        if txn.event_id == event_id and txn.source == "combat":
            return txn
    return None


def _active_combat_character_ids(ckpt: CheckpointFile) -> set[str]:
    combat = _active_combat_state(ckpt)
    if combat is None:
        return set()
    return {
        cid
        for cid in (
            _combatant_character_id(combatant)
            for combatant in _combatants(combat)
        )
        if cid
    }


def _canonized_tick_character_ids(
    routed: EventRouterOutput,
    drafts: list[tuple[CharacterRecord, CharacterAgentTurnDraft]],
) -> set[str]:
    """Return tick proposal ids represented in the canonical tick event."""
    observer_ids = {observer.character_id for observer in routed.observers}
    draft_ids = {char.character_id for char, _draft in drafts}
    return observer_ids & draft_ids


def _commit_tick_draft(
    ckpt: CheckpointFile,
    char: CharacterRecord,
    draft: CharacterAgentTurnDraft,
) -> None:
    clear_character_inbox(char)
    conv = ckpt.character_conversations.setdefault(char.character_id, [])
    conv.extend([draft.user_message, draft.assistant_message])


_WORD_RE = re.compile(r"[a-z0-9_']+")


def _words(text: str) -> set[str]:
    return {
        word
        for word in _WORD_RE.findall((text or "").lower())
        if len(word) >= 3
    }


def _recent_tick_signal_text(ckpt: CheckpointFile, *, event_limit: int = 8) -> str:
    parts: list[str] = []
    for event in ckpt.canonical_events[-event_limit:]:
        canonical = _obj_get(event, "canonical_event")
        for fact in list(_obj_get(canonical, "observable_facts", []) or []):
            text = str(_obj_get(fact, "text", "") or "").strip()
            if text:
                parts.append(text)
        for observer in list(_obj_get(event, "observers", []) or []):
            cid = str(_obj_get(observer, "character_id", "") or "").strip()
            if cid:
                parts.append(cid)
        for update in list(_obj_get(event, "location_updates", []) or []):
            loc = str(_obj_get(update, "location_label", "") or "").strip()
            if loc:
                parts.append(loc)
    parts.extend(str(line) for line in ckpt.session.pending_router_state_changes)
    return "\n".join(parts)


def _match_phrase_score(phrase: str, haystack: str, hay_words: set[str]) -> float:
    phrase = (phrase or "").strip().lower()
    if not phrase:
        return 0.0
    if phrase in haystack:
        return 14.0
    cue_words = _words(phrase)
    if not cue_words:
        return 0.0
    overlap = len(cue_words & hay_words)
    if len(cue_words) == 1:
        return 6.0 if overlap else 0.0
    required = max(2, int(len(cue_words) * 0.6 + 0.999))
    if overlap >= required:
        return 8.0 + min(4, overlap)
    return 0.0


def _agent_tier_weight(char: CharacterRecord) -> float:
    if char.agent_tier in {CharacterAgentTier.premium, CharacterAgentTier.plot}:
        return 8.0
    if char.agent_tier == CharacterAgentTier.standard:
        return 5.0
    return 2.0


def _is_premium_tick_tier(char: CharacterRecord) -> bool:
    return char.agent_tier in {CharacterAgentTier.premium, CharacterAgentTier.plot}


def _score_tick_candidate(
    ckpt: CheckpointFile,
    char: CharacterRecord,
    *,
    recent_text: str,
    recent_words: set[str],
    roster_index: int,
) -> TickCandidate:
    score = _agent_tier_weight(char)
    reasons: list[str] = [f"tier={char.agent_tier.value}"]

    cue_scores = [
        _match_phrase_score(cue, recent_text, recent_words)
        for cue in char.private_state.tick_cues
    ]
    cue_score = max(cue_scores) if cue_scores else 0.0
    if cue_score:
        score += cue_score
        reasons.append(f"cue={cue_score:g}")

    if char.pending_observations:
        score += 5.0
        reasons.append("pending_observations")

    for label in (
        char.character_id,
        char.name,
        char.public_sheet.faction,
        char.location,
    ):
        if not label:
            continue
        match = _match_phrase_score(label, recent_text, recent_words)
        if match:
            boost = min(4.0, match)
            score += boost
            reasons.append(f"recent_match={boost:g}")
            break

    for objective in char.private_state.current_objectives:
        overlap = _words(objective) & recent_words
        if overlap:
            boost = min(4.0, float(len(overlap)))
            score += boost
            reasons.append(f"objective_overlap={boost:g}")
            break

    leading = max(int(getattr(ckpt.session, "leading_at_s", 0) or 0), 0)
    previous = char.last_agent_turn_at_s
    if previous is None:
        score += 1.0
        reasons.append("never_ticked")
    else:
        debt = max(0, leading - previous)
        if debt:
            boost = min(6.0, debt / 300.0)
            score += boost
            reasons.append(f"debt={boost:g}")

    return TickCandidate(
        character=char,
        score=score,
        cue_score=cue_score,
        roster_index=roster_index,
        reason=", ".join(reasons),
    )


def _bounded_tick_batch_size(settings: Any) -> int:
    return min(max(1, int(settings.tick_batch_size)), TICK_BATCH_HARD_CAP)


def _select_tick_batch(
    candidates: list[TickCandidate],
    *,
    batch_size: int,
) -> list[TickCandidate]:
    selected: list[TickCandidate] = []
    premium_selected = 0
    non_premium_available = any(
        not _is_premium_tick_tier(c.character) for c in candidates
    )

    for candidate in candidates:
        if len(selected) >= batch_size:
            break
        is_premium = _is_premium_tick_tier(candidate.character)
        if (
            is_premium
            and non_premium_available
            and premium_selected >= TICK_PREMIUM_PER_BATCH
        ):
            continue
        selected.append(candidate)
        if is_premium:
            premium_selected += 1

    if not selected:
        selected = candidates[:batch_size]
    elif len(selected) < batch_size and not non_premium_available:
        seen = {c.character.character_id for c in selected}
        for candidate in candidates:
            if len(selected) >= batch_size:
                break
            if candidate.character.character_id in seen:
                continue
            selected.append(candidate)
    return selected


class Orchestrator:
    """Binds `turn_loop.run_beat` to the LLM/storage layers.

    One instance lives per `EngineBridge`; sessions in the same process
    share the `SessionLockManager` so concurrent /acts serialize against
    a single asyncio.Lock.
    """

    def __init__(
        self,
        client: LLMClient,
        checkpoint_mgr: CheckpointManager,
        prompt_mgr: PromptManager,
    ):
        self.client = client
        self.prompt_mgr = prompt_mgr
        self.checkpoint_mgr = checkpoint_mgr
        self.char_mgr = CharacterManager(client, prompt_mgr)
        # One manager per Orchestrator. /acts in the same session
        # serialize here; perception fan-out is observer-driven.
        self.session_locks = SessionLockManager()

    def _install_pending_narrator_render_saver(
        self,
        dispatcher: LLMDispatcher,
        *,
        acting_id: str,
        roll_keys_before: set[tuple[str, str]],
        revision_before: CommitmentRevisionPrompt | None,
    ) -> Callable[[], bool]:
        saved = False

        def _persist(ckpt: CheckpointFile) -> None:
            nonlocal saved
            if saved:
                return
            pending = ckpt.session.pending_narrator_render
            if pending is not None:
                pending.roll_keys_before = sorted(roll_keys_before)
                if revision_before is not None:
                    pending.commitment_revision_character_id = acting_id
                    pending.commitment_revision_id = (
                        revision_before.commitment_id
                    )
                    pending.commitment_revision_trigger_id = (
                        revision_before.trigger_event_id
                    )

            previous_turn = ckpt.session.turn_index
            ckpt.session.turn_index = previous_turn + 1
            try:
                self.checkpoint_mgr.save(ckpt)
            except Exception:
                ckpt.session.turn_index = previous_turn
                raise
            saved = True

        setattr(dispatcher, "persist_pending_narrator_render", _persist)
        return lambda: saved

    @staticmethod
    def _clear_pending_commitment_revision(
        ckpt: CheckpointFile,
        pending: PendingNarratorRender,
    ) -> None:
        character_id = pending.commitment_revision_character_id
        if not character_id:
            return
        current = ckpt.session.pending_commitment_revisions.get(character_id)
        if (
            current is not None
            and current.commitment_id == pending.commitment_revision_id
            and current.trigger_event_id
            == pending.commitment_revision_trigger_id
        ):
            ckpt.session.pending_commitment_revisions.pop(character_id, None)

    async def _resume_pending_narrator_render_locked(
        self,
        *,
        session_id: str,
        ckpt: CheckpointFile,
        dispatcher: LLMDispatcher,
    ) -> TurnResponse | None:
        pending = ckpt.session.pending_narrator_render
        if pending is None:
            return None

        roll_keys_before = {
            (str(transaction_id), str(roll_id))
            for transaction_id, roll_id in pending.roll_keys_before
        }
        beat_result = await _end_beat(
            ckpt,
            dispatcher,
            ended_reason=pending.ended_reason,
            events_closed=pending.events_closed,
            event_actor_ids=list(pending.event_actor_ids),
            release_slots=pending.release_slots,
            force_partial=pending.force_partial,
            acting_player_id=pending.acting_player_id,
            acting_player_input=pending.acting_player_input,
            suppress_reaction_prompts=pending.suppress_reaction_prompts,
        )
        self._clear_pending_commitment_revision(ckpt, pending)
        await self._apply_beat_roster_side_effects(
            ckpt, beat_result, log_label="Resumed BeatResult",
        )
        _append_transcript_entry(
            ckpt, beat_result, pending.acting_player_id,
        )
        reaction_prompts = beat_result.reaction_prompts or {}
        if not reaction_prompts:
            await self._run_ticks(
                ckpt,
                acted_this_turn=set(beat_result.event_actor_ids),
                acting_id=pending.acting_player_id,
            )
        _handle_combat_after_beat(
            ckpt,
            acting_id=pending.acting_player_id,
            beat_result=beat_result,
        )
        flush_combat_visible_facts(ckpt)
        self.checkpoint_mgr.save(ckpt)
        automated_results = await self._run_automated_combat_turns_locked(
            ckpt=ckpt,
            dispatcher=dispatcher,
        )

        beat_results = [beat_result, *automated_results]
        per_player = _combine_beat_renders(beat_results)
        output_text = per_player.get(pending.acting_player_id, "")
        final_result = beat_results[-1]
        return TurnResponse(
            session_id=session_id,
            checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
            turn_index=ckpt.session.turn_index,
            output_text=output_text,
            per_player_renders=per_player,
            beat_ended_reason=_combined_beat_reason(beat_results),
            reaction_prompts=final_result.reaction_prompts or {},
            loot_prompts=_combine_loot_prompts(beat_results),
            commitment_revision_prompts=_commitment_revision_prompts(ckpt),
            dice_rolls=dice_roll_displays_since(ckpt, roll_keys_before),
            experience_awards=_drain_experience_awards(ckpt),
        )

    async def _apply_beat_roster_side_effects(
        self,
        ckpt: CheckpointFile,
        beat_result: BeatResult,
        *,
        log_label: str,
    ) -> dict[str, list[str]]:
        if beat_result.events_closed <= 0:
            beat_result.loot_prompts = {}
            return {}
        closed_this_beat = ckpt.canonical_events[-beat_result.events_closed:]
        actors = beat_result.event_actor_ids
        if len(actors) != beat_result.events_closed:
            logger.warning(
                "%s BeatResult event_actor_ids length %d != events_closed %d; "
                "spawn location fallback may be empty.",
                log_label, len(actors), beat_result.events_closed,
            )
            actors = actors + [None] * (beat_result.events_closed - len(actors))
        for evt, evt_actor in zip(closed_this_beat, actors):
            self.char_mgr.apply_roster_updates(ckpt, evt)
            if evt.spawn:
                spawn_loc = (
                    self._resolve_location(ckpt, evt_actor)
                    if evt_actor else ""
                )
                await self.char_mgr.spawn_characters(
                    ckpt, evt.spawn,
                    acting_actor_location=spawn_loc,
                )
        loot_prompts = dnd_inventory.apply_loot_offers_from_events(
            ckpt,
            closed_this_beat,
        )
        beat_result.loot_prompts = loot_prompts
        return loot_prompts

    async def _run_automated_combat_turns_locked(
        self,
        *,
        ckpt: CheckpointFile,
        dispatcher: LLMDispatcher,
    ) -> list[BeatResult]:
        results: list[BeatResult] = []
        combat = _active_combat_state(ckpt)
        if combat is None:
            return results
        max_auto_turns = max(1, len(_combatants(combat)))
        turns_taken = 0

        while turns_taken < max_auto_turns:
            if _blocking_slots_open(ckpt):
                break
            combat = _active_combat_state(ckpt)
            if combat is None:
                break
            current = _current_combatant(ckpt, combat)
            if current is None:
                break
            actor_id = _combatant_character_id(current)
            if not actor_id or _combatant_human_controlled(ckpt, current):
                break

            try:
                intention = await _agent_intention_for_dispatch(
                    dispatcher, ckpt, actor_id,
                )
            except Exception:
                logger.exception(
                    "Automated combat agent intention failed for %s (%s); "
                    "skipping turn so initiative cannot wedge.",
                    _combatant_name(current), actor_id,
                )
                _append_combat_audit_line(
                    ckpt,
                    f"Automated combat turn for {_combatant_name(current)} "
                    "failed before an intention was produced; initiative "
                    "advances.",
                )
                _advance_combat_initiative_after_turn(ckpt)
                ckpt.session.turn_index += 1
                self.checkpoint_mgr.save(ckpt)
                turns_taken += 1
                continue
            if intention is None:
                _append_combat_audit_line(
                    ckpt,
                    f"No actionable combat intention from "
                    f"{_combatant_name(current)}; initiative advances.",
                )
                _advance_combat_initiative_after_turn(ckpt)
                ckpt.session.turn_index += 1
                self.checkpoint_mgr.save(ckpt)
                turns_taken += 1
                continue

            snapshot = _automated_turn_snapshot(ckpt)
            try:
                beat_result = await run_beat(
                    ckpt=ckpt,
                    dispatcher=dispatcher,
                    actor_id=actor_id,
                    intention=intention,
                )
            except Exception:
                logger.exception(
                    "Automated combat beat failed for %s (%s); aborting "
                    "partial NPC beat and advancing initiative.",
                    _combatant_name(current), actor_id,
                )
                _rollback_automated_turn_snapshot(ckpt, snapshot)
                _append_combat_audit_line(
                    ckpt,
                    f"Automated combat turn for {_combatant_name(current)} "
                    "failed during resolution; partial NPC beat was "
                    "aborted and initiative advances.",
                )
                _advance_combat_initiative_after_turn(ckpt)
                ckpt.session.turn_index += 1
                self.checkpoint_mgr.save(ckpt)
                turns_taken += 1
                continue
            await self._apply_beat_roster_side_effects(
                ckpt, beat_result, log_label="Combat automation",
            )
            _append_transcript_entry(ckpt, beat_result, actor_id)
            _handle_combat_after_beat(
                ckpt,
                acting_id=actor_id,
                beat_result=beat_result,
            )
            flush_combat_visible_facts(ckpt)
            ckpt.session.turn_index += 1
            self.checkpoint_mgr.save(ckpt)
            results.append(beat_result)
            turns_taken += 1

            if (
                _blocking_slots_open(ckpt)
                or beat_result.reaction_prompts
                or beat_result.ended_reason in _COMBAT_NO_ADVANCE_REASONS
            ):
                break

        if turns_taken >= max_auto_turns:
            combat = _active_combat_state(ckpt)
            current = (
                _current_combatant(ckpt, combat)
                if combat is not None else None
            )
            if (
                current is not None
                and not _combatant_human_controlled(ckpt, current)
            ):
                _append_combat_audit_line(
                    ckpt,
                    "Stopped automated NPC combat turn chain at safety cap.",
                )
                self.checkpoint_mgr.save(ckpt)

        return results

    async def process_turn(self, request: TurnRequest) -> TurnResponse:
        """Process a single turn end-to-end, v11-style.

        Steps: load checkpoint → resolve actor → acquire session lock →
        slot check → run beat → apply per-event roster side-effects →
        save → build response.
        """
        # 3. Acquire the session lock. Prevents two concurrent /acts
        # from both seeing FREE on their check_act_slot.
        lock = await self.session_locks.get(request.session_id)
        async with lock:
            ckpt = self.checkpoint_mgr.load_latest(request.session_id)
            sync_checkpoint_runtime_models(ckpt, self.client.config)
            dispatcher = LLMDispatcher(self.client, self.prompt_mgr)
            resumed = await self._resume_pending_narrator_render_locked(
                session_id=request.session_id,
                ckpt=ckpt,
                dispatcher=dispatcher,
            )
            if resumed is not None:
                return resumed

            # 1. Resolve the acting character.
            try:
                acting_id = self._resolve_acting_character(ckpt, request)
            except ValueError:
                return TurnResponse(
                    session_id=request.session_id,
                    checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                    turn_index=ckpt.session.turn_index,
                    output_text=(
                        "Choose a character before acting. Use /join to "
                        "claim one, or pass an acting character id."
                    ),
                    per_player_renders={},
                    beat_ended_reason="acting_character_required",
                )

            logger.info(
                "Turn %d for session %s (acting=%s)",
                ckpt.session.turn_index, request.session_id, acting_id,
            )

            pre_turn_resolutions: list[TurnResponse] = []
            if _should_resume_automated_combat_before_act(ckpt, acting_id):
                pre_roll_keys_before = completed_automatic_roll_keys(ckpt)
                automated_before = await self._run_automated_combat_turns_locked(
                    ckpt=ckpt,
                    dispatcher=dispatcher,
                )
                pre_response = _turn_response_from_beat_results(
                    session_id=request.session_id,
                    ckpt=ckpt,
                    acting_id=acting_id,
                    beat_results=automated_before,
                    roll_keys_before=pre_roll_keys_before,
                )
                if pre_response is not None:
                    pre_turn_resolutions.append(pre_response)
                    if (
                        pre_response.output_text
                        or pre_response.per_player_renders
                        or pre_response.dice_rolls
                    ):
                        return _with_pre_turn_resolutions(TurnResponse(
                            session_id=request.session_id,
                            checkpoint_id=(
                                f"ckpt_{ckpt.session.turn_index:04d}"
                            ),
                            turn_index=ckpt.session.turn_index,
                            output_text=(
                                "The scene changed before your submitted "
                                "action could be applied. Submit your next "
                                "action from the updated state."
                            ),
                            per_player_renders={},
                            beat_ended_reason="pre_turn_resolution",
                        ), pre_turn_resolutions)

            # 4. Validate against the session's active_act_slot.
            blocked_entry = ckpt.session.active_act_slots.get(acting_id)
            was_combat_blocked = (
                blocked_entry is not None
                and blocked_entry.reason == "combat_blocked"
            )
            check = check_act_slot(ckpt, acting_id)

            if (
                was_combat_blocked
                and request.user_input.strip().lower() == "(defer)"
            ):
                release_character_slot(ckpt, acting_id)
                self.checkpoint_mgr.save(ckpt)
                return _with_pre_turn_resolutions(TurnResponse(
                    session_id=request.session_id,
                    checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                    turn_index=ckpt.session.turn_index,
                    output_text=(
                        "The blocked combat-starting action was dropped. "
                        "You may act normally."
                    ),
                    per_player_renders={},
                    beat_ended_reason="combat_start_blocked_deferred",
                ), pre_turn_resolutions)

            if check.conflict in (SlotConflict.INITIATOR_HELD,
                                  SlotConflict.CAT_II_OTHER_HELD,
                                  SlotConflict.COMBAT_REACTION_OTHER_HELD,
                                  SlotConflict.CAT_II_SELF_ROLL,
                                  SlotConflict.SELF_BUSY):
                msg = format_slot_rejection(
                    check, ckpt, attempted_text=request.user_input,
                )
                # Reject early. Do NOT save — the checkpoint is unchanged.
                return _with_pre_turn_resolutions(TurnResponse(
                    session_id=request.session_id,
                    checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                    turn_index=ckpt.session.turn_index,
                    output_text=msg,
                    per_player_renders={},
                    beat_ended_reason="slot_rejected",
                ), pre_turn_resolutions)

            combat_reaction_event_id = (
                check.trigger_event_id
                if check.conflict == SlotConflict.COMBAT_REACTION_SELF
                else None
            )

            if combat_reaction_event_id and (
                request.user_input.strip().lower() == "(defer)"
            ):
                response = self._defer_combat_reaction_locked(
                    ckpt=ckpt,
                    session_id=request.session_id,
                    character_id=acting_id,
                    event_id=combat_reaction_event_id,
                )
                self.checkpoint_mgr.save(ckpt)
                return _with_pre_turn_resolutions(
                    response, pre_turn_resolutions,
                )

            if check.conflict not in (
                SlotConflict.CAT_II_SELF_RESPONDER,
                SlotConflict.COMBAT_REACTION_SELF,
            ):
                combat_rejection = _combat_turn_rejection(
                    ckpt, acting_id, request.user_input,
                )
                if combat_rejection is not None:
                    return _with_pre_turn_resolutions(TurnResponse(
                        session_id=request.session_id,
                        checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                        turn_index=ckpt.session.turn_index,
                        output_text=combat_rejection,
                        per_player_renders={},
                        beat_ended_reason="combat_turn_rejected",
                    ), pre_turn_resolutions)

            cat_ii_event_id = (
                check.cat_ii_event_id
                if check.conflict == SlotConflict.CAT_II_SELF_RESPONDER
                else None
            )

            if was_combat_blocked and check.conflict == SlotConflict.FREE:
                release_character_slot(ckpt, acting_id)

            # 5. Run the beat.
            roll_keys_before = completed_automatic_roll_keys(ckpt)
            revision_input_consumed = (
                cat_ii_event_id is None
                and combat_reaction_event_id is None
                and acting_id not in _active_combat_character_ids(ckpt)
            )
            revision_before = (
                ckpt.session.pending_commitment_revisions.get(acting_id)
                if revision_input_consumed else None
            )
            pending_render_saved = (
                self._install_pending_narrator_render_saver(
                    dispatcher,
                    acting_id=acting_id,
                    roll_keys_before=roll_keys_before,
                    revision_before=revision_before,
                )
            )
            beat_result = await run_beat(
                ckpt=ckpt,
                dispatcher=dispatcher,
                actor_id=acting_id,
                intention=request.user_input,
                cat_ii_event_id=cat_ii_event_id,
                combat_reaction_event_id=combat_reaction_event_id,
            )
            if revision_before is not None:
                current_revision = ckpt.session.pending_commitment_revisions.get(
                    acting_id
                )
                if (
                    current_revision is not None
                    and current_revision.commitment_id
                    == revision_before.commitment_id
                    and current_revision.trigger_event_id
                    == revision_before.trigger_event_id
                ):
                    ckpt.session.pending_commitment_revisions.pop(
                        acting_id,
                        None,
                    )

            # 6. Apply roster side-effects of every event that closed
            # this beat. `run_beat` broadcasts + renders but leaves
            # spawn / dormant / cull for the orchestrator to apply.
            await self._apply_beat_roster_side_effects(
                ckpt, beat_result, log_label="BeatResult",
            )

            # v11-r7f: persist the acting POV's transcript entry so
            # /history and the resume-display path see played turns.
            # ckpt.transcript is single-list session-global; with
            # multi-POV the other POVs' entries still live in
            # narrator_conversations[h] but we pick one canonical
            # speaker for the user-facing log.
            _append_transcript_entry(ckpt, beat_result, acting_id)

            # 6.5. Commits 5 + 6: background tick fan-out + unified-
            # router fan-in. Eligible off-stage NPCs run their
            # `.draft_tick()` after the on-stage beat closes. Successful
            # drafts are then bundled into ONE
            # router call in tick mode (the router's user message
            # gets a `## Off-Stage Tick` block listing each ticker's
            # public prose + location). The router emits one
            # canonical event capturing the off-stage developments,
            # plus any spawn/dormancy/cull changes; only canonized
            # drafts are committed to agent history. No narrator pass —
            # the player wasn't there.
            # Per-tick errors and the fan-in router error are both
            # logged-and-swallowed so a single LLM hiccup doesn't
            # drop the rest of the turn.
            reaction_prompts = beat_result.reaction_prompts or {}
            if not reaction_prompts:
                await self._run_ticks(
                    ckpt,
                    acted_this_turn=set(beat_result.event_actor_ids),
                    acting_id=acting_id,
                )

            _handle_combat_after_beat(
                ckpt,
                acting_id=acting_id,
                beat_result=beat_result,
                allow_new_pending=combat_reaction_event_id is None,
            )
            flush_combat_visible_facts(ckpt)

            # 7. Save. run_beat has already mutated active_act_slots,
            # open_cat_ii_events, render_buffers, canonical_events, and
            # (through the dispatcher) narrator_conversations. Ticks
            # (above) added rolling-conversation appends and last-intent
            # writes; one save covers both.
            if not pending_render_saved():
                ckpt.session.turn_index += 1
            self.checkpoint_mgr.save(ckpt)
            automated_results = await self._run_automated_combat_turns_locked(
                ckpt=ckpt,
                dispatcher=dispatcher,
            )

        # 8. Build the response.
        beat_results = [beat_result, *automated_results]
        per_player = _combine_beat_renders(beat_results)
        output_text = per_player.get(acting_id, "")
        final_result = beat_results[-1]
        return _with_pre_turn_resolutions(TurnResponse(
            session_id=request.session_id,
            checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
            turn_index=ckpt.session.turn_index,
            output_text=output_text,
            per_player_renders=per_player,
            beat_ended_reason=_combined_beat_reason(beat_results),
            reaction_prompts=final_result.reaction_prompts or {},
            loot_prompts=_combine_loot_prompts(beat_results),
            commitment_revision_prompts=_commitment_revision_prompts(ckpt),
            dice_rolls=dice_roll_displays_since(ckpt, roll_keys_before),
            experience_awards=_drain_experience_awards(ckpt),
        ), pre_turn_resolutions)

    def _defer_combat_reaction_locked(
        self,
        *,
        ckpt: CheckpointFile,
        session_id: str,
        character_id: str,
        event_id: str = "",
    ) -> TurnResponse:
        slot = ckpt.session.active_act_slots.get(character_id)
        trigger_id = ""
        if slot is not None:
            trigger_id = slot.trigger_event_id or slot.cat_ii_event_id or ""
        if (
            slot is None
            or slot.reason != "combat_reaction"
            or (event_id and trigger_id != event_id)
        ):
            return TurnResponse(
                session_id=session_id,
                checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                turn_index=ckpt.session.turn_index,
                output_text=(
                    "That reaction window is already closed. Use /combat "
                    "status to see the current state."
                ),
                per_player_renders={},
                beat_ended_reason="combat_reaction_stale",
            )

        release_character_slot(ckpt, character_id)
        advanced = _advance_pending_combat_if_unblocked(ckpt)
        if advanced:
            combat = _active_combat_state(ckpt)
            current = _current_combatant(ckpt, combat) if combat is not None else None
            current_name = (
                _combatant_name(current)
                if current is not None else "the next combatant"
            )
            message = (
                "No reaction recorded. "
                f"Initiative advances to **{current_name}**."
            )
        elif any(
            entry.reason == "combat_reaction"
            for entry in ckpt.session.active_act_slots.values()
        ):
            message = (
                "No reaction recorded. Waiting on another possible reaction."
            )
        else:
            message = "No reaction recorded."

        ckpt.session.turn_index += 1
        return TurnResponse(
            session_id=session_id,
            checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
            turn_index=ckpt.session.turn_index,
            output_text=message,
            per_player_renders={},
            beat_ended_reason="combat_reaction_deferred",
        )

    async def defer_combat_reaction(
        self,
        *,
        session_id: str,
        character_id: str,
        event_id: str = "",
    ) -> TurnResponse:
        lock = await self.session_locks.get(session_id)
        async with lock:
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            sync_checkpoint_runtime_models(ckpt, self.client.config)
            response = self._defer_combat_reaction_locked(
                ckpt=ckpt,
                session_id=session_id,
                character_id=character_id,
                event_id=event_id,
            )
            if response.beat_ended_reason != "combat_reaction_stale":
                self.checkpoint_mgr.save(ckpt)
            return response

    async def resolve_cat_ii(
        self, session_id: str, event_id: str,
    ) -> TurnResponse:
        """v11-r6b: adjudicate a Cat II event whose responders have all
        intended (typically after `sweep_stale_pins` synthesized AFK
        intentions). Used by EngineBridge.run_turn after sweep returns
        event ids, to close them out BEFORE the current /act processes.

        Acquires the session lock for the event, re-checks
        readiness, drives `route_intention` on the adjudication path,
        closes the event, broadcasts the canonical result, lets an NPC
        initiator take the first follow-up when applicable, fans renders
        out, applies roster side-effects, and saves.
        Returns a TurnResponse describing the resolution; if the event
        was already closed (race) returns an empty "cat_ii_stale"
        response.
        """
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        sync_checkpoint_runtime_models(ckpt, self.client.config)
        evt = next(
            (e for e in ckpt.session.open_cat_ii_events if e.event_id == event_id),
            None,
        )
        if evt is None:
            logger.warning(
                "resolve_cat_ii called for %s but event not open", event_id,
            )
            return TurnResponse(
                session_id=session_id,
                checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                turn_index=ckpt.session.turn_index,
                output_text="",
                per_player_renders={},
                beat_ended_reason="cat_ii_stale",
            )

        lock = await self.session_locks.get(session_id)
        async with lock:
            # Re-read: another task may have closed this event while we
            # were waiting for the lock.
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            sync_checkpoint_runtime_models(ckpt, self.client.config)
            roll_keys_before = completed_automatic_roll_keys(ckpt)
            evt_live = next(
                (e for e in ckpt.session.open_cat_ii_events
                 if e.event_id == event_id),
                None,
            )
            if evt_live is None:
                if roll_transaction_source(ckpt, event_id) == "combat":
                    if pending_player_rolls(ckpt, event_id=event_id):
                        return TurnResponse(
                            session_id=session_id,
                            checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                            turn_index=ckpt.session.turn_index,
                            output_text="",
                            per_player_renders={},
                            beat_ended_reason="cat_ii_pending_rolls",
                        )
                    return await self._resolve_ready_combat_after_rolls(
                        session_id=session_id,
                        ckpt=ckpt,
                        event_id=event_id,
                        output_actor_id=(
                            getattr(
                                _combat_roll_transaction(ckpt, event_id),
                                "actor_id",
                                "",
                            ) or ""
                        ),
                    )
                return TurnResponse(
                    session_id=session_id,
                    checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                    turn_index=ckpt.session.turn_index,
                    output_text="",
                    per_player_renders={},
                    beat_ended_reason="cat_ii_stale",
                )

            dispatcher = LLMDispatcher(self.client, self.prompt_mgr)

            if cat_ii_is_ready(evt_live):
                try:
                    resolved = await dispatcher.route_intention(
                        ckpt=ckpt,
                        actor_id=evt_live.initiator_id,
                        intention=evt_live.initiator_intention,
                        cat_ii_event=evt_live,
                    )
                except DndCatIIRollsPending:
                    beat_result = BeatResult(
                        renders={},
                        events_closed=0,
                        ended_reason="cat_ii_pending_rolls",
                        transcript_entries={},
                        event_actor_ids=[],
                        reaction_prompts={},
                    )
                    self.checkpoint_mgr.save(ckpt)
                    renders: dict[str, str] = {}
                    return TurnResponse(
                        session_id=session_id,
                        checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                        turn_index=ckpt.session.turn_index,
                        output_text="",
                        per_player_renders=renders,
                        beat_ended_reason=beat_result.ended_reason,
                        reaction_prompts=beat_result.reaction_prompts or {},
                        dice_rolls=dice_roll_displays_since(
                            ckpt, roll_keys_before,
                        ),
                    )
                close_cat_ii(ckpt, evt_live.event_id)
                release_beat_slots(ckpt)
                if resolved.requires_responders:
                    raise ValueError(
                        "Cat II resolution returned nested Cat II "
                        "(Part C invariant violated)."
                    )
                align_cat_ii_resolution_time(ckpt, evt_live, resolved)
                # A Cat II resolution is the adjudicated outcome of all
                # collected intentions. Every NPC observer, including the
                # initiator, needs the final result in their inbox for future
                # turns.
                broadcast_event(ckpt, resolved)
                initiator_pick = _filter_picks_for_dispatch(
                    ckpt, [evt_live.initiator_id],
                    event=resolved,
                )
                followup = None
                if initiator_pick:
                    followup = await _agent_intention_for_dispatch(
                        dispatcher, ckpt, evt_live.initiator_id,
                    )
                if followup is not None:
                    followup_result = await run_beat(
                        ckpt=ckpt,
                        dispatcher=dispatcher,
                        actor_id=evt_live.initiator_id,
                        intention=followup,
                    )
                    beat_result = BeatResult(
                        renders=followup_result.renders,
                        events_closed=1 + followup_result.events_closed,
                        ended_reason=followup_result.ended_reason,
                        transcript_entries=followup_result.transcript_entries,
                        event_actor_ids=[
                            evt_live.initiator_id,
                            *followup_result.event_actor_ids,
                        ],
                        reaction_prompts=followup_result.reaction_prompts or {},
                    )
                else:
                    beat_result = await _end_beat(
                        ckpt, dispatcher,
                        ended_reason="cat_ii_resolution",
                        events_closed=1,
                        event_actor_ids=[evt_live.initiator_id],
                    )
            else:
                # Still pending responders — nothing to adjudicate yet.
                beat_result = BeatResult(
                    renders={}, events_closed=0,
                    ended_reason="cat_ii_pending",
                    transcript_entries={},
                    event_actor_ids=[],
                    reaction_prompts={},
                )

            await self._apply_beat_roster_side_effects(
                ckpt,
                beat_result,
                log_label="Cat II BeatResult",
            )

            if beat_result.events_closed > 0:
                ckpt.session.turn_index += 1

            # v11-r7f: persist transcript entry for Cat II resolution
            # too — initiator's POV is the canonical speaker for the
            # adjudicated event.
            _append_transcript_entry(ckpt, beat_result, evt.initiator_id)
            _handle_combat_after_beat(
                ckpt,
                acting_id=evt.initiator_id,
                beat_result=beat_result,
            )
            flush_combat_visible_facts(ckpt)

            self.checkpoint_mgr.save(ckpt)
            automated_results = await self._run_automated_combat_turns_locked(
                ckpt=ckpt,
                dispatcher=dispatcher,
            )

        beat_results = [beat_result, *automated_results]
        renders = _combine_beat_renders(beat_results)
        final_result = beat_results[-1]
        actor_id = evt.initiator_id
        output_text = renders.get(actor_id, "") or (
            next(iter(renders.values()), "") if renders else ""
        )
        return TurnResponse(
            session_id=session_id,
            checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
            turn_index=ckpt.session.turn_index,
            output_text=output_text,
            per_player_renders=renders,
            beat_ended_reason=_combined_beat_reason(beat_results),
            reaction_prompts=final_result.reaction_prompts or {},
            loot_prompts=_combine_loot_prompts(beat_results),
            commitment_revision_prompts=_commitment_revision_prompts(ckpt),
            dice_rolls=dice_roll_displays_since(ckpt, roll_keys_before),
            experience_awards=_drain_experience_awards(ckpt),
        )

    async def submit_cat_ii_roll(
        self,
        *,
        session_id: str,
        event_id: str,
        roll_id: str,
        actor_id: str,
        user_id: str = "",
    ) -> TurnResponse:
        """Execute one pending player roll and finalize the Cat II if ready."""
        lock = await self.session_locks.get(session_id)
        async with lock:
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            sync_checkpoint_runtime_models(ckpt, self.client.config)
            evt_live = next(
                (e for e in ckpt.session.open_cat_ii_events
                 if e.event_id == event_id),
                None,
            )
            if evt_live is None:
                if roll_transaction_source(ckpt, event_id) == "combat":
                    transaction = _combat_roll_transaction(ckpt, event_id)
                    if (
                        transaction is None
                        or transaction.status == "cancelled"
                        or _active_combat_state(ckpt) is None
                    ):
                        return self._stale_combat_roll_response(
                            ckpt=ckpt,
                            session_id=session_id,
                            output_actor_id=actor_id,
                        )
                    pending_for_actor = pending_player_rolls(
                        ckpt, event_id=event_id, actor_id=actor_id,
                    )
                    if roll_id not in {
                        record.roll_id for record in pending_for_actor
                    }:
                        return self._stale_combat_roll_response(
                            ckpt=ckpt,
                            session_id=session_id,
                            output_actor_id=actor_id,
                        )
                    complete_pending_player_roll(
                        ckpt,
                        event_id=event_id,
                        roll_id=roll_id,
                        completed_by_user_id=user_id,
                    )
                    if pending_player_rolls(ckpt, event_id=event_id):
                        self.checkpoint_mgr.save(ckpt)
                        return TurnResponse(
                            session_id=session_id,
                            checkpoint_id=(
                                f"ckpt_{ckpt.session.turn_index:04d}"
                            ),
                            turn_index=ckpt.session.turn_index,
                            output_text="",
                            per_player_renders={},
                            beat_ended_reason="cat_ii_pending_rolls",
                        )
                    return await self._resolve_ready_combat_after_rolls(
                        session_id=session_id,
                        ckpt=ckpt,
                        event_id=event_id,
                        output_actor_id=actor_id,
                    )
                return TurnResponse(
                    session_id=session_id,
                    checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                    turn_index=ckpt.session.turn_index,
                    output_text="",
                    per_player_renders={},
                    beat_ended_reason="cat_ii_stale",
                )

            pending_for_actor = pending_player_rolls(
                ckpt, event_id=event_id, actor_id=actor_id,
            )
            if roll_id not in {record.roll_id for record in pending_for_actor}:
                return TurnResponse(
                    session_id=session_id,
                    checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                    turn_index=ckpt.session.turn_index,
                    output_text=(
                        "That roll is no longer pending for your character. "
                        "Use /combat status to see the current state."
                    ),
                    per_player_renders={},
                    beat_ended_reason="cat_ii_stale",
                )

            complete_pending_player_roll(
                ckpt,
                event_id=event_id,
                roll_id=roll_id,
                completed_by_user_id=user_id,
            )
            if pending_player_rolls(ckpt, event_id=event_id):
                self.checkpoint_mgr.save(ckpt)
                return TurnResponse(
                    session_id=session_id,
                    checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                    turn_index=ckpt.session.turn_index,
                    output_text="",
                    per_player_renders={},
                    beat_ended_reason="cat_ii_pending_rolls",
                )

            return await self._resolve_ready_cat_ii_after_rolls(
                session_id=session_id,
                ckpt=ckpt,
                evt_live=evt_live,
                output_actor_id=actor_id,
            )

    async def _resolve_ready_combat_after_rolls(
        self,
        *,
        session_id: str,
        ckpt: CheckpointFile,
        event_id: str,
        output_actor_id: str,
    ) -> TurnResponse:
        transaction = next(
            (
                txn for txn in ckpt.session.cat_ii_roll_transactions
                if txn.event_id == event_id
            ),
            None,
        )
        if (
            transaction is None
            or transaction.status == "cancelled"
            or transaction.status == "finalized"
            or _active_combat_state(ckpt) is None
        ):
            return self._stale_combat_roll_response(
                ckpt=ckpt,
                session_id=session_id,
                output_actor_id=output_actor_id,
            )
        dispatcher = LLMDispatcher(self.client, self.prompt_mgr)
        roll_keys_before = completed_automatic_roll_keys(ckpt)
        try:
            resolved = await dispatcher.continue_combat_transaction(
                ckpt=ckpt,
                event_id=event_id,
            )
        except DndCatIIRollsPending:
            self.checkpoint_mgr.save(ckpt)
            return TurnResponse(
                session_id=session_id,
                checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                turn_index=ckpt.session.turn_index,
                output_text="",
                per_player_renders={},
                beat_ended_reason="cat_ii_pending_rolls",
                dice_rolls=dice_roll_displays_since(ckpt, roll_keys_before),
            )

        if resolved.requires_responders:
            raise ValueError(
                "D&D combat roll continuation returned generic Cat II."
            )
        _clear_pending_initiating_action(ckpt, output_actor_id)
        broadcast_event(ckpt, resolved, actor_id=output_actor_id)
        beat_result = await _end_beat(
            ckpt,
            dispatcher,
            ended_reason=resolved.ends_beat_reason or "ruleset_resolution",
            events_closed=1,
            event_actor_ids=[output_actor_id],
            acting_player_id=output_actor_id,
        )

        await self._apply_beat_roster_side_effects(
            ckpt,
            beat_result,
            log_label="Combat roll BeatResult",
        )
        if beat_result.events_closed > 0:
            ckpt.session.turn_index += 1

        _append_transcript_entry(ckpt, beat_result, output_actor_id)
        _handle_combat_after_beat(
            ckpt,
            acting_id=output_actor_id,
            beat_result=beat_result,
        )
        if flush_combat_visible_facts(ckpt):
            self.checkpoint_mgr.save(ckpt)
        release_character_slot(ckpt, output_actor_id)
        self.checkpoint_mgr.save(ckpt)
        automated_results = await self._run_automated_combat_turns_locked(
            ckpt=ckpt,
            dispatcher=dispatcher,
        )

        beat_results = [beat_result, *automated_results]
        renders = _combine_beat_renders(beat_results)
        final_result = beat_results[-1]
        output_text = renders.get(output_actor_id, "") or (
            next(iter(renders.values()), "") if renders else ""
        )
        return TurnResponse(
            session_id=session_id,
            checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
            turn_index=ckpt.session.turn_index,
            output_text=output_text,
            per_player_renders=renders,
            beat_ended_reason=_combined_beat_reason(beat_results),
            reaction_prompts=final_result.reaction_prompts or {},
            loot_prompts=_combine_loot_prompts(beat_results),
            commitment_revision_prompts=_commitment_revision_prompts(ckpt),
            dice_rolls=dice_roll_displays_since(ckpt, roll_keys_before),
            experience_awards=_drain_experience_awards(ckpt),
        )

    def _stale_combat_roll_response(
        self,
        *,
        ckpt: CheckpointFile,
        session_id: str,
        output_actor_id: str,
    ) -> TurnResponse:
        if output_actor_id:
            release_character_slot(ckpt, output_actor_id)
        self.checkpoint_mgr.save(ckpt)
        return TurnResponse(
            session_id=session_id,
            checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
            turn_index=ckpt.session.turn_index,
            output_text=(
                "That combat roll is no longer active. Use /combat status "
                "to see the current state."
            ),
            per_player_renders={},
            beat_ended_reason="cat_ii_stale",
        )

    async def continue_cat_ii_after_roll(
        self,
        *,
        session_id: str,
        event_id: str,
        actor_id: str,
    ) -> TurnResponse:
        """Finalize a Cat II whose player roll was already persisted."""
        lock = await self.session_locks.get(session_id)
        async with lock:
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            sync_checkpoint_runtime_models(ckpt, self.client.config)
            evt_live = next(
                (e for e in ckpt.session.open_cat_ii_events
                 if e.event_id == event_id),
                None,
            )
            if evt_live is None:
                if roll_transaction_source(ckpt, event_id) == "combat":
                    if pending_player_rolls(ckpt, event_id=event_id):
                        return TurnResponse(
                            session_id=session_id,
                            checkpoint_id=(
                                f"ckpt_{ckpt.session.turn_index:04d}"
                            ),
                            turn_index=ckpt.session.turn_index,
                            output_text="",
                            per_player_renders={},
                            beat_ended_reason="cat_ii_pending_rolls",
                        )
                    return await self._resolve_ready_combat_after_rolls(
                        session_id=session_id,
                        ckpt=ckpt,
                        event_id=event_id,
                        output_actor_id=actor_id,
                    )
                return TurnResponse(
                    session_id=session_id,
                    checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                    turn_index=ckpt.session.turn_index,
                    output_text="",
                    per_player_renders={},
                    beat_ended_reason="cat_ii_stale",
                )
            if pending_player_rolls(ckpt, event_id=event_id):
                return TurnResponse(
                    session_id=session_id,
                    checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                    turn_index=ckpt.session.turn_index,
                    output_text="",
                    per_player_renders={},
                    beat_ended_reason="cat_ii_pending_rolls",
                )
            return await self._resolve_ready_cat_ii_after_rolls(
                session_id=session_id,
                ckpt=ckpt,
                evt_live=evt_live,
                output_actor_id=actor_id,
            )

    async def _resolve_ready_cat_ii_after_rolls(
        self,
        *,
        session_id: str,
        ckpt: CheckpointFile,
        evt_live,
        output_actor_id: str,
    ) -> TurnResponse:
        dispatcher = LLMDispatcher(self.client, self.prompt_mgr)
        roll_keys_before = completed_automatic_roll_keys(ckpt)
        try:
            resolved = await dispatcher.route_intention(
                ckpt=ckpt,
                actor_id=evt_live.initiator_id,
                intention=evt_live.initiator_intention,
                cat_ii_event=evt_live,
            )
        except DndCatIIRollsPending:
            self.checkpoint_mgr.save(ckpt)
            return TurnResponse(
                session_id=session_id,
                checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                turn_index=ckpt.session.turn_index,
                output_text="",
                per_player_renders={},
                beat_ended_reason="cat_ii_pending_rolls",
                dice_rolls=dice_roll_displays_since(ckpt, roll_keys_before),
            )

        close_cat_ii(ckpt, evt_live.event_id)
        release_beat_slots(ckpt)
        if resolved.requires_responders:
            raise ValueError(
                "Cat II resolution returned nested Cat II "
                "(Part C invariant violated)."
            )
        align_cat_ii_resolution_time(ckpt, evt_live, resolved)
        broadcast_event(ckpt, resolved)
        initiator_pick = _filter_picks_for_dispatch(
            ckpt, [evt_live.initiator_id],
            event=resolved,
        )
        followup = None
        if initiator_pick:
            followup = await _agent_intention_for_dispatch(
                dispatcher, ckpt, evt_live.initiator_id,
            )
        if followup is not None:
            followup_result = await run_beat(
                ckpt=ckpt,
                dispatcher=dispatcher,
                actor_id=evt_live.initiator_id,
                intention=followup,
            )
            beat_result = BeatResult(
                renders=followup_result.renders,
                events_closed=1 + followup_result.events_closed,
                ended_reason=followup_result.ended_reason,
                transcript_entries=followup_result.transcript_entries,
                event_actor_ids=[
                    evt_live.initiator_id,
                    *followup_result.event_actor_ids,
                ],
                reaction_prompts=followup_result.reaction_prompts or {},
            )
        else:
            beat_result = await _end_beat(
                ckpt, dispatcher,
                ended_reason="cat_ii_resolution",
                events_closed=1,
                event_actor_ids=[evt_live.initiator_id],
            )

        await self._apply_beat_roster_side_effects(
            ckpt,
            beat_result,
            log_label="Cat II roll BeatResult",
        )
        if beat_result.events_closed > 0:
            ckpt.session.turn_index += 1

        _append_transcript_entry(ckpt, beat_result, evt_live.initiator_id)
        _handle_combat_after_beat(
            ckpt,
            acting_id=evt_live.initiator_id,
            beat_result=beat_result,
        )
        flush_combat_visible_facts(ckpt)
        self.checkpoint_mgr.save(ckpt)
        automated_results = await self._run_automated_combat_turns_locked(
            ckpt=ckpt,
            dispatcher=dispatcher,
        )

        beat_results = [beat_result, *automated_results]
        renders = _combine_beat_renders(beat_results)
        final_result = beat_results[-1]
        output_text = renders.get(output_actor_id, "") or (
            next(iter(renders.values()), "") if renders else ""
        )
        return TurnResponse(
            session_id=session_id,
            checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
            turn_index=ckpt.session.turn_index,
            output_text=output_text,
            per_player_renders=renders,
            beat_ended_reason=_combined_beat_reason(beat_results),
            reaction_prompts=final_result.reaction_prompts or {},
            loot_prompts=_combine_loot_prompts(beat_results),
            commitment_revision_prompts=_commitment_revision_prompts(ckpt),
            dice_rolls=dice_roll_displays_since(ckpt, roll_keys_before),
            experience_awards=_drain_experience_awards(ckpt),
        )

    # ------------------------------------------------------------------ helpers

    def _resolve_acting_character(
        self, ckpt: CheckpointFile, request: TurnRequest
    ) -> str:
        """Pick the acting character id. Request-supplied wins; else fall
        back to the user's bound character; else session.player_character_id.
        Raises ValueError if nothing resolves."""
        if request.acting_character_id:
            return request.acting_character_id

        # Fall back to the session's creator binding. (Legacy single-
        # player call sites and CLI playtest sessions.)
        pid = ckpt.session.player_character_id
        if pid:
            return pid

        raise ValueError("Choose a character before acting.")

    def _resolve_location(
        self, ckpt: CheckpointFile, acting_id: str
    ) -> str:
        """The acting character's opaque location label, read from the
        roster. Returns "" when the character has no resolvable location."""
        for c in ckpt.characters:
            if c.character_id == acting_id and c.location:
                return c.location
        return ""

    # ---------------------------------------------------------- tick scheduler

    def _eligible_for_tick(
        self,
        ckpt: CheckpointFile,
        acted_this_turn: set[str],
    ) -> list[CharacterRecord]:
        """Filter the roster to characters that should run an off-stage
        tick on this beat.

        Five guards:
          - `private_state.intentions_enabled` is True — importer flag
            for "this character matters enough to advance off-screen"
          - `status == active` — dormant/culled don't tick
          - NOT in any player binding (`character_bindings` keys or
            `session.player_character_id`) — humans don't get auto-ticked
          - NOT in a current bound-player location — physically present
            scene participants stay on the normal on-stage routing path
          - NOT in `acted_this_turn` (the on-stage actor + any picked
            responders this beat) — they already had their say
          - NOT in active combat — combatants advance through initiative,
            not background ticks
          - NOT in `_pinned_character_ids(ckpt)` — pinned NPCs are
            mid-Cat-II, ticking races their pending resolution

        Order is roster order; that's also the order their tick
        outputs will reach the unified router.
        """
        pinned_ids = _pinned_character_ids(ckpt)
        player_ids = collect_player_ids(ckpt)
        combat_ids = _active_combat_character_ids(ckpt)
        player_locations = {
            char.location
            for char in ckpt.characters
            if char.character_id in player_ids and char.location
        }

        eligible: list[CharacterRecord] = []
        for char in ckpt.characters:
            if not char.private_state.intentions_enabled:
                continue
            if char.status != CharacterStatus.active:
                continue
            if char.character_id in player_ids:
                continue
            if char.location and char.location in player_locations:
                continue
            if char.character_id in acted_this_turn:
                continue
            if char.character_id in combat_ids:
                continue
            if char.character_id in pinned_ids:
                continue
            eligible.append(char)
        return eligible

    async def _run_ticks(
        self,
        ckpt: CheckpointFile,
        acted_this_turn: set[str],
        acting_id: str,
    ) -> list[tuple[CharacterRecord, CharacterAgentOutput]]:
        """Decide whether to fire a tick pass this beat, and if so, draft a
        bounded batch of high-value off-stage NPC actions.

        Trigger model:
          - `turns_since_last_tick` increments unconditionally each
            beat (even when no tick fires).
          - **Cue branch**: fires early when an eligible character's
            authored `private_state.tick_cues` match recent canonical
            events or queued state changes.
          - **Stagnation branch**: fires unconditionally after
            `tick_stagnation_max` idle beats so the world keeps
            moving even when the player stays in one conversation.
          - On any fire, reset `turns_since_last_tick` to 0.

        Concurrency: a fresh `asyncio.Semaphore` per call, sized at
        `min(settings.tick_concurrency, TICK_CONCURRENCY_HARD_CAP)`.
        Each tick uses its OWN `CharacterAgent` instance so concurrent
        completions don't race on `agent.last_usage`.
        The first selected character for each model role is awaited before
        parallelizing the rest of that role, giving provider cache prefixes a
        chance to warm per tier.

        After fan-out, Commit 6 bundles every successful draft's
        `public_text` into a single unified-router call in tick mode.
        Only drafts represented in the canonical event are committed to
        agent history and inbox clearing; uncanonized drafts are discarded.
        Returns the committed tick outputs primarily for tests and
        observability; the orchestrator caller doesn't otherwise use them.
        """
        sess = ckpt.session
        settings = sess.config.settings

        # Master kill switch (v11). Short-circuits BEFORE the trigger
        # counters mutate so flipping `ticks_enabled` back on later
        # resumes from where the trigger model left off rather than
        # firing a backlog. No counter increment, no eligibility
        # filter, no fan-out, no router fan-in, no canonical-event
        # append. This is intended for token-budget runs and for
        # diagnostics that want to isolate on-stage behavior.
        if not settings.ticks_enabled:
            logger.debug(
                "Tick scheduler: disabled via settings.ticks_enabled; "
                "skipping (turn=%d).",
                sess.turn_index,
            )
            return []

        sess.turns_since_last_tick += 1

        eligible = self._eligible_for_tick(ckpt, acted_this_turn)
        recent_text = _recent_tick_signal_text(ckpt).lower()
        recent_words = _words(recent_text)
        ranked = [
            _score_tick_candidate(
                ckpt,
                char,
                recent_text=recent_text,
                recent_words=recent_words,
                roster_index=idx,
            )
            for idx, char in enumerate(eligible)
        ]
        ranked.sort(key=lambda c: (-c.score, -c.cue_score, c.roster_index))

        stagnation_fires = (
            sess.turns_since_last_tick >= settings.tick_stagnation_max
        )
        cue_fires = bool(
            ranked and ranked[0].cue_score >= TICK_CUE_FIRE_THRESHOLD
        )

        if not (stagnation_fires or cue_fires):
            top = ranked[0] if ranked else None
            logger.debug(
                "Tick scheduler: no fire (turns_since_last_tick=%d, "
                "stagnation_ok=%s, cue_ok=%s, eligible=%d, top=%s)",
                sess.turns_since_last_tick,
                stagnation_fires,
                cue_fires,
                len(eligible),
                (
                    f"{top.character.character_id} score={top.score:g} "
                    f"cue={top.cue_score:g}"
                )
                if top is not None
                else "none",
            )
            return []

        if not eligible:
            # Still reset the counter — we DID try to fire, eligibility
            # was just empty (e.g. all NPCs are dormant, on-stage, or
            # already acted). Otherwise an all-on-stage beat would
            # never reset and the next turn would over-fire.
            if stagnation_fires:
                sess.turns_since_last_tick = 0
            logger.info(
                "Tick scheduler: %s fire but no eligible NPCs "
                "(roster=%d, acted=%d; pinned, player, dormant, "
                "or intentions_disabled filtered all out); "
                "counter %s.",
                "stagnation" if stagnation_fires else "cue",
                len(ckpt.characters),
                len(acted_this_turn),
                "reset" if stagnation_fires else "left unchanged",
            )
            return []

        reason = "stagnation" if stagnation_fires else "cue"
        candidate_pool = (
            [
                candidate for candidate in ranked
                if candidate.cue_score >= TICK_CUE_FIRE_THRESHOLD
            ]
            if cue_fires and not stagnation_fires
            else ranked
        )
        selected_candidates = _select_tick_batch(
            candidate_pool,
            batch_size=_bounded_tick_batch_size(settings),
        )
        selected = [candidate.character for candidate in selected_candidates]
        if not selected:
            if stagnation_fires:
                sess.turns_since_last_tick = 0
            logger.info(
                "Tick scheduler: %s fire had %d eligible NPC(s) but no "
                "candidate survived selection; counter %s.",
                reason,
                len(eligible),
                "reset" if stagnation_fires else "left unchanged",
            )
            return []

        cap = min(
            max(1, settings.tick_concurrency), TICK_CONCURRENCY_HARD_CAP,
        )
        semaphore = asyncio.Semaphore(cap)

        logger.info(
            "Tick scheduler: %s fire — %d eligible NPC(s), %d selected "
            "(batch cap %d, concurrency cap %d): %s",
            reason,
            len(eligible),
            len(selected),
            _bounded_tick_batch_size(settings),
            cap,
            "; ".join(
                f"{c.character.character_id}:score={c.score:g},"
                f"cue={c.cue_score:g}({c.reason})"
                for c in selected_candidates
            ),
        )

        async def _one(
            char: CharacterRecord,
        ) -> tuple[CharacterRecord, CharacterAgentTurnDraft] | None:
            async with semaphore:
                # Fresh agent per tick so concurrent ticks don't race
                # on `agent.last_usage`.
                agent = CharacterAgent(self.client, self.prompt_mgr)
                try:
                    draft = await agent.draft_tick(
                        character=char,
                        checkpoint=ckpt,
                        acting_character_id=acting_id,
                    )
                except Exception:
                    # Swallow per-tick failures. One agent's API hiccup
                    # shouldn't drop the whole beat — the on-stage
                    # render has already happened; ticks are
                    # best-effort world-advance.
                    logger.exception(
                        "Tick failed for %s (%s); skipping this character "
                        "and continuing the fan-out.",
                        char.name, char.character_id,
                    )
                    return None
                usage = dict(getattr(agent, "last_usage", {}) or {})
                logger.info(
                    "Tick %s (%s): public=%dch intent=%dch usage=%s",
                    char.name, char.character_id,
                    len(draft.output.public_text), len(draft.output.intent),
                    usage,
                )
                return (char, draft)

        result_by_id: dict[str, tuple[CharacterRecord, CharacterAgentTurnDraft]] = {}
        warmed_roles: set[str] = set()
        parallel_chars: list[CharacterRecord] = []

        for char in selected:
            role = model_role_for_character(char)
            if role in warmed_roles:
                parallel_chars.append(char)
                continue
            result = await _one(char)
            if result is None:
                continue
            result_by_id[char.character_id] = result
            warmed_roles.add(role)

        if parallel_chars:
            results = await asyncio.gather(*[_one(c) for c in parallel_chars])
            for result in results:
                if result is None:
                    continue
                result_by_id[result[0].character_id] = result

        drafts = [
            result_by_id[char.character_id]
            for char in selected
            if char.character_id in result_by_id
        ]

        sess.turns_since_last_tick = 0

        logger.info(
            "Tick scheduler: %s fire complete — %d/%d selected tick drafts "
            "succeeded (%d eligible).",
            reason, len(drafts), len(selected), len(eligible),
        )

        # Commit 6: tick fan-in to the unified router.
        # Bundle every successful draft's public prose (parenthetical
        # stripped — `draft.output.public_text` only) into one router call
        # in tick mode. The router emits a single canonical event
        # capturing off-stage developments. No narrator pass — the player
        # wasn't there. Drafts are committed only after that canonical
        # event identifies which proposed beats actually happened.
        if not drafts:
            return []

        tick_outputs = [
            (
                char.name,
                char.character_id,
                char.location or "",
                draft.output.public_text,
            )
            for char, draft in drafts
        ]
        dispatcher = LLMDispatcher(self.client, self.prompt_mgr)
        try:
            routed = await dispatcher.route_tick_intentions(
                ckpt=ckpt,
                tick_outputs=tick_outputs,
                acting_character_id=acting_id,
            )
        except Exception:
            # Same swallow philosophy as per-tick failures — the on-
            # stage render landed already; ticks are best-effort. If
            # the fan-in router call fails, the per-character drafts are
            # discarded; the next /act will go through the on-stage router
            # as normal. World state just doesn't pick up the off-stage
            # canonical event on this beat.
            logger.exception(
                "Tick fan-in router call failed; off-stage agent "
                "drafts were discarded and no canonical event lands "
                "this turn.",
            )
            return []

        if routed is None:
            return []

        canonized_ids = _canonized_tick_character_ids(routed, drafts)
        committed: list[tuple[CharacterRecord, CharacterAgentOutput]] = []
        for char, draft in drafts:
            if char.character_id not in canonized_ids:
                continue
            _commit_tick_draft(ckpt, char, draft)
            committed.append((char, draft.output))

        self.char_mgr.apply_roster_updates(ckpt, routed)
        if routed.spawn:
            # Tick-driven spawns have no single acting actor location.
            # Pass "" so spawns fall back to either router-supplied
            # seed.location or the spawn LLM's authored location.
            await self.char_mgr.spawn_characters(
                ckpt, routed.spawn, acting_actor_location="",
            )
        tick_actor_ids = [char.character_id for char, _ in committed]
        if routed.effective_at_s == 0 and tick_actor_ids:
            clocks = [
                getattr(char, "clock_at_s", 0)
                for char, _ in committed
            ]
            routed.effective_at_s = max(clocks) if clocks else 0
        routed.effective_at_s = max(
            routed.effective_at_s,
            ckpt.session.leading_at_s,
        )
        broadcast_event(
            ckpt,
            routed,
            actor_id=tick_actor_ids[0] if len(tick_actor_ids) == 1 else "",
        )
        tick_end_s = routed.effective_at_s + routed.duration_s
        ckpt.session.leading_at_s = max(ckpt.session.leading_at_s, tick_end_s)
        for char, _ in committed:
            char.clock_at_s = max(getattr(char, "clock_at_s", 0), tick_end_s)
            previous = char.last_agent_turn_at_s
            char.last_agent_turn_at_s = max(
                previous if previous is not None else 0,
                tick_end_s,
            )
        logger.info(
            "Tick fan-in routed: %d/%d draft(s) committed; %d spawn(s); "
            "off-stage canonical event appended.",
            len(committed), len(drafts), len(routed.spawn),
        )

        return committed
