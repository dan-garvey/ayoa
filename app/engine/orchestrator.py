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
from copy import deepcopy
from typing import Any, Callable

from app.engine.context_builder import collect_player_ids
from app.engine.character_manager import CharacterManager
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
    _cat_ii_resolution_followup_ids,
)
from app.llm.client import LLMClient

# Imported at module level so tests can monkeypatch
# `app.engine.orchestrator.LLMDispatcher` without reaching into the
# adapter module directly. Hard import now that the Dispatcher has
# landed — a missing module is a real packaging error.
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.schemas.checkpoint import CheckpointFile
from app.schemas.requests import TurnRequest
from app.schemas.responses import TurnResponse
from app.schemas.state import CommitmentRevisionPrompt, PendingNarratorRender

logger = logging.getLogger(__name__)


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
        "pending_engine_state_updates": list(
            ckpt.session.pending_engine_state_updates
        ),
        "content_state": deepcopy(ckpt.session.content_state),
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
    ckpt.session.pending_engine_state_updates = list(
        snapshot["pending_engine_state_updates"]
    )
    ckpt.session.content_state = deepcopy(snapshot["content_state"])

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
    fallback_to_first_render: bool = False,
) -> TurnResponse | None:
    if not beat_results:
        return None
    per_player = _combine_beat_renders(beat_results)
    final_result = beat_results[-1]
    output_text = per_player.get(acting_id, "")
    if fallback_to_first_render and not output_text:
        output_text = next(iter(per_player.values()), "")
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


def _cat_ii_pending_rolls_response(
    *,
    session_id: str,
    ckpt: CheckpointFile,
    roll_keys_before: set[tuple[str, str]] | None = None,
) -> TurnResponse:
    dice_rolls = (
        dice_roll_displays_since(ckpt, roll_keys_before)
        if roll_keys_before is not None
        else []
    )
    return TurnResponse(
        session_id=session_id,
        checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
        turn_index=ckpt.session.turn_index,
        output_text="",
        per_player_renders={},
        beat_ended_reason="cat_ii_pending_rolls",
        dice_rolls=dice_rolls,
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
        response = _turn_response_from_beat_results(
            session_id=session_id,
            ckpt=ckpt,
            acting_id=pending.acting_player_id,
            beat_results=beat_results,
            roll_keys_before=roll_keys_before,
        )
        assert response is not None
        return response

    async def retry_pending_narrator_render(
        self,
        session_id: str,
    ) -> TurnResponse:
        """Resume a failed narrator render without accepting a new action."""
        lock = await self.session_locks.get(session_id)
        async with lock:
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            sync_checkpoint_runtime_models(ckpt, self.client.config)
            dispatcher = LLMDispatcher(self.client, self.prompt_mgr)
            resumed = await self._resume_pending_narrator_render_locked(
                session_id=session_id,
                ckpt=ckpt,
                dispatcher=dispatcher,
            )
            if resumed is not None:
                return resumed

            return TurnResponse(
                session_id=session_id,
                checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                turn_index=ckpt.session.turn_index,
                output_text=(
                    "No failed narrator render is pending for this session."
                ),
                per_player_renders={},
                beat_ended_reason="no_pending_render",
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
                        or pre_response.asset_reveals
                        or pre_response.per_player_asset_reveals
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

            _handle_combat_after_beat(
                ckpt,
                acting_id=acting_id,
                beat_result=beat_result,
                allow_new_pending=combat_reaction_event_id is None,
            )
            flush_combat_visible_facts(ckpt)

            # 7. Save. run_beat has already mutated active_act_slots,
            # open_cat_ii_events, render_buffers, canonical_events, and
            # (through the dispatcher) narrator_conversations.
            if not pending_render_saved():
                ckpt.session.turn_index += 1
            self.checkpoint_mgr.save(ckpt)
            automated_results = await self._run_automated_combat_turns_locked(
                ckpt=ckpt,
                dispatcher=dispatcher,
            )

        # 8. Build the response.
        beat_results = [beat_result, *automated_results]
        response = _turn_response_from_beat_results(
            session_id=request.session_id,
            ckpt=ckpt,
            acting_id=acting_id,
            beat_results=beat_results,
            roll_keys_before=roll_keys_before,
        )
        assert response is not None
        return _with_pre_turn_resolutions(response, pre_turn_resolutions)

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

    async def _cat_ii_resolution_beat_result_locked(
        self,
        *,
        ckpt: CheckpointFile,
        dispatcher: LLMDispatcher,
        evt_live: Any,
    ) -> BeatResult:
        resolved = await dispatcher.route_intention(
            ckpt=ckpt,
            actor_id=evt_live.initiator_id,
            intention=evt_live.initiator_intention,
            cat_ii_event=evt_live,
        )
        close_cat_ii(ckpt, evt_live.event_id)
        release_beat_slots(ckpt)
        if resolved.requires_responders:
            raise ValueError(
                "Cat II resolution returned nested Cat II "
                "(Part C invariant violated)."
            )
        align_cat_ii_resolution_time(ckpt, evt_live, resolved)
        # A Cat II resolution is the adjudicated outcome of all collected
        # intentions. Every NPC observer, including the initiator, needs the
        # final result in their inbox for future turns.
        broadcast_event(ckpt, resolved)
        followup_ids = _cat_ii_resolution_followup_ids(
            ckpt,
            resolved,
            initiator_id=evt_live.initiator_id,
        )
        followup_actor_id = followup_ids[0] if followup_ids else ""
        followup = None
        if followup_actor_id:
            followup = await _agent_intention_for_dispatch(
                dispatcher, ckpt, followup_actor_id,
            )
        if followup is None:
            return await _end_beat(
                ckpt, dispatcher,
                ended_reason="cat_ii_resolution",
                events_closed=1,
                event_actor_ids=[evt_live.initiator_id],
            )

        followup_result = await run_beat(
            ckpt=ckpt,
            dispatcher=dispatcher,
            actor_id=followup_actor_id,
            intention=followup,
        )
        return BeatResult(
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

    async def _finalize_ready_cat_ii_locked(
        self,
        *,
        session_id: str,
        ckpt: CheckpointFile,
        dispatcher: LLMDispatcher,
        evt_live: Any,
        output_actor_id: str,
        roll_keys_before: set[tuple[str, str]],
        log_label: str,
    ) -> TurnResponse:
        try:
            beat_result = await self._cat_ii_resolution_beat_result_locked(
                ckpt=ckpt,
                dispatcher=dispatcher,
                evt_live=evt_live,
            )
        except DndCatIIRollsPending:
            self.checkpoint_mgr.save(ckpt)
            return _cat_ii_pending_rolls_response(
                session_id=session_id,
                ckpt=ckpt,
                roll_keys_before=roll_keys_before,
            )

        await self._apply_beat_roster_side_effects(
            ckpt,
            beat_result,
            log_label=log_label,
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
        response = _turn_response_from_beat_results(
            session_id=session_id,
            ckpt=ckpt,
            acting_id=output_actor_id,
            beat_results=beat_results,
            roll_keys_before=roll_keys_before,
            fallback_to_first_render=True,
        )
        assert response is not None
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
        closes the event, broadcasts the canonical result, lets selected
        follow-up NPCs act when applicable, fans renders
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
                        return _cat_ii_pending_rolls_response(
                            session_id=session_id,
                            ckpt=ckpt,
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
                return await self._finalize_ready_cat_ii_locked(
                    session_id=session_id,
                    ckpt=ckpt,
                    dispatcher=dispatcher,
                    evt_live=evt_live,
                    output_actor_id=evt_live.initiator_id,
                    roll_keys_before=roll_keys_before,
                    log_label="Cat II BeatResult",
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
        response = _turn_response_from_beat_results(
            session_id=session_id,
            ckpt=ckpt,
            acting_id=evt.initiator_id,
            beat_results=beat_results,
            roll_keys_before=roll_keys_before,
            fallback_to_first_render=True,
        )
        assert response is not None
        return response

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
                        return _cat_ii_pending_rolls_response(
                            session_id=session_id,
                            ckpt=ckpt,
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
                return _cat_ii_pending_rolls_response(
                    session_id=session_id,
                    ckpt=ckpt,
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
            return _cat_ii_pending_rolls_response(
                session_id=session_id,
                ckpt=ckpt,
                roll_keys_before=roll_keys_before,
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
            ended_reason=(
                resolved.event_kind
                if resolved.event_kind != "beat_continues"
                else "ruleset_resolution"
            ),
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
        response = _turn_response_from_beat_results(
            session_id=session_id,
            ckpt=ckpt,
            acting_id=output_actor_id,
            beat_results=beat_results,
            roll_keys_before=roll_keys_before,
            fallback_to_first_render=True,
        )
        assert response is not None
        return response

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
                        return _cat_ii_pending_rolls_response(
                            session_id=session_id,
                            ckpt=ckpt,
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
                return _cat_ii_pending_rolls_response(
                    session_id=session_id,
                    ckpt=ckpt,
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
        return await self._finalize_ready_cat_ii_locked(
            session_id=session_id,
            ckpt=ckpt,
            dispatcher=dispatcher,
            evt_live=evt_live,
            output_actor_id=output_actor_id,
            roll_keys_before=roll_keys_before,
            log_label="Cat II roll BeatResult",
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
