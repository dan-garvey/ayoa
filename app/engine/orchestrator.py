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

from app.engine.character_agent import CharacterAgent
from app.engine.character_manager import CharacterManager, _pinned_character_ids
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.model_config_sync import sync_checkpoint_runtime_models
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop import (
    BeatResult,
    SessionLockManager,
    SlotConflict,
    broadcast_event,
    cat_ii_is_ready,
    check_act_slot,
    close_cat_ii,
    format_slot_rejection,
    release_beat_slots,
    run_beat,
    _agent_intention_for_dispatch,
    _end_beat,
    _filter_picks_for_dispatch,
)
from app.llm.client import LLMClient

# Imported at module level so tests can monkeypatch
# `app.engine.orchestrator.LLMDispatcher` without reaching into the
# adapter module directly. Hard import now that the Dispatcher has
# landed — a missing module is a real packaging error.
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.requests import TurnRequest
from app.schemas.responses import TurnResponse

logger = logging.getLogger(__name__)


# Commit 5: hard ceiling on background-tick concurrency, regardless of
# what `SessionSettings.tick_concurrency` is configured to. Sized for
# Anthropic's per-minute API limits on Haiku — a hollowstone-sized
# roster (~12 NPCs) all eligible at once would still fan out under
# this. Raise only after measuring rate-limit headroom.
TICK_CONCURRENCY_HARD_CAP = 16


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

    async def process_turn(self, request: TurnRequest) -> TurnResponse:
        """Process a single turn end-to-end, v11-style.

        Steps: load checkpoint → resolve actor → acquire session lock →
        slot check → run beat → apply per-event roster side-effects →
        save → build response.
        """
        ckpt = self.checkpoint_mgr.load_latest(request.session_id)
        sync_checkpoint_runtime_models(ckpt, self.client.config)

        # 1. Resolve the acting character.
        acting_id = self._resolve_acting_character(ckpt, request)

        logger.info(
            "Turn %d for session %s (acting=%s)",
            ckpt.session.turn_index, request.session_id, acting_id,
        )

        # 3. Acquire the session lock. Prevents two concurrent /acts
        # from both seeing FREE on their check_act_slot.
        lock = await self.session_locks.get(request.session_id)
        async with lock:
            # 4. Validate against the session's active_act_slot.
            check = check_act_slot(ckpt, acting_id)

            if check.conflict in (SlotConflict.INITIATOR_HELD,
                                  SlotConflict.CAT_II_OTHER_HELD,
                                  SlotConflict.SELF_BUSY):
                msg = format_slot_rejection(
                    check, ckpt, attempted_text=request.user_input,
                )
                # Reject early. Do NOT save — the checkpoint is unchanged.
                return TurnResponse(
                    session_id=request.session_id,
                    checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                    turn_index=ckpt.session.turn_index,
                    output_text=msg,
                    per_player_renders={},
                    beat_ended_reason="slot_rejected",
                )

            cat_ii_event_id = (
                check.cat_ii_event_id
                if check.conflict == SlotConflict.CAT_II_SELF_RESPONDER
                else None
            )

            # 5. Run the beat.
            dispatcher = LLMDispatcher(self.client, self.prompt_mgr)
            beat_result = await run_beat(
                ckpt=ckpt,
                dispatcher=dispatcher,
                actor_id=acting_id,
                intention=request.user_input,
                cat_ii_event_id=cat_ii_event_id,
            )

            # 6. Apply roster side-effects of every event that closed
            # this beat. `run_beat` broadcasts + renders but leaves
            # spawn / dormant / cull for the orchestrator to apply.
            closed = beat_result.events_closed
            if closed > 0:
                closed_this_beat = ckpt.canonical_events[-closed:]
                actors = beat_result.event_actor_ids
                # Defensive: invariant says lengths match; if they don't
                # (e.g. a future refactor forgot to track an actor), pad
                # with None so spawn fallback logic stays conservative.
                if len(actors) != closed:
                    logger.warning(
                        "BeatResult event_actor_ids length %d != events_closed %d "
                        "; spawn location fallback may be empty.",
                        len(actors), closed,
                    )
                    actors = actors + [None] * (closed - len(actors))
                for evt, evt_actor in zip(closed_this_beat, actors):
                    self.char_mgr.apply_roster_updates(ckpt, evt)
                    # Spawns remain async LLM calls; if none declared,
                    # the helper is a no-op.
                    if evt.spawn:
                        spawn_loc = (
                            self._resolve_location(ckpt, evt_actor)
                            if evt_actor else ""
                        )
                        await self.char_mgr.spawn_characters(
                            ckpt, evt.spawn,
                            acting_actor_location=spawn_loc,
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
            # `.tick()` after the on-stage beat closes — each tick's
            # prose + trailing parenthetical lands in that agent's
            # rolling conversation (the parenthetical never leaves
            # the agent). Successful ticks are then bundled into ONE
            # router call in tick mode (the router's user message
            # gets a `## Off-Stage Tick` block listing each ticker's
            # public prose + location). The router emits one
            # canonical event capturing the off-stage developments,
            # plus any spawn/dormancy/cull changes; we apply those to the checkpoint
            # off-stage (no narrator pass — the player wasn't there).
            # Per-tick errors and the fan-in router error are both
            # logged-and-swallowed so a single LLM hiccup doesn't
            # drop the rest of the turn.
            await self._run_ticks(
                ckpt,
                acted_this_turn=set(beat_result.event_actor_ids),
                acting_id=acting_id,
            )

            # 7. Save. run_beat has already mutated active_act_slots,
            # open_cat_ii_events, render_buffers, canonical_events, and
            # (through the dispatcher) narrator_conversations. Ticks
            # (above) added rolling-conversation appends and last-intent
            # writes; one save covers both.
            ckpt.session.turn_index += 1
            self.checkpoint_mgr.save(ckpt)

        # 8. Build the response.
        per_player = dict(beat_result.renders)
        output_text = per_player.get(acting_id, "")
        return TurnResponse(
            session_id=request.session_id,
            checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
            turn_index=ckpt.session.turn_index,
            output_text=output_text,
            per_player_renders=per_player,
            beat_ended_reason=beat_result.ended_reason,
        )

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
            evt_live = next(
                (e for e in ckpt.session.open_cat_ii_events
                 if e.event_id == event_id),
                None,
            )
            if evt_live is None:
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
                )

            # Apply side-effects for each newly closed event.
            if beat_result.events_closed > 0:
                closed_this_beat = ckpt.canonical_events[
                    -beat_result.events_closed:
                ]
                actors = beat_result.event_actor_ids
                if len(actors) != beat_result.events_closed:
                    logger.warning(
                        "Cat II BeatResult event_actor_ids length %d != "
                        "events_closed %d; spawn location fallback may be empty.",
                        len(actors), beat_result.events_closed,
                    )
                    actors = actors + [None] * (
                        beat_result.events_closed - len(actors)
                    )
                for ev, ev_actor in zip(closed_this_beat, actors):
                    self.char_mgr.apply_roster_updates(ckpt, ev)
                    if ev.spawn:
                        spawn_loc = (
                            self._resolve_location(ckpt, ev_actor)
                            if ev_actor else ""
                        )
                        await self.char_mgr.spawn_characters(
                            ckpt, ev.spawn,
                            acting_actor_location=spawn_loc,
                        )

            if beat_result.events_closed > 0:
                ckpt.session.turn_index += 1

            # v11-r7f: persist transcript entry for Cat II resolution
            # too — initiator's POV is the canonical speaker for the
            # adjudicated event.
            _append_transcript_entry(ckpt, beat_result, evt.initiator_id)

            self.checkpoint_mgr.save(ckpt)

        renders = dict(beat_result.renders)
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
            beat_ended_reason=beat_result.ended_reason,
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

        raise ValueError(
            "process_turn: no acting_character_id resolvable — request did "
            "not supply one and session.player_character_id is empty. "
            "Callers must supply a character id for multi-player sessions."
        )

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
          - NOT in `acted_this_turn` (the on-stage actor + any picked
            responders this beat) — they already had their say
          - NOT in `_pinned_character_ids(ckpt)` — pinned NPCs are
            mid-Cat-II, ticking races their pending resolution

        Order is roster order; that's also the order their tick
        outputs will reach the unified router.
        """
        from app.engine.context_builder import collect_player_ids

        pinned_ids = _pinned_character_ids(ckpt)
        player_ids = collect_player_ids(ckpt)

        eligible: list[CharacterRecord] = []
        for char in ckpt.characters:
            if not char.private_state.intentions_enabled:
                continue
            if char.status != CharacterStatus.active:
                continue
            if char.character_id in player_ids:
                continue
            if char.character_id in acted_this_turn:
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
        """Decide whether to fire a tick pass this beat, and if so,
        fan out `CharacterAgent.tick()` for every eligible NPC under a
        bounded semaphore.

        Trigger model (Commit 5 / decision #9):
          - `turns_since_last_tick` increments unconditionally each
            beat (even when no tick fires).
          - **Stagnation branch**: fires unconditionally after
            `tick_stagnation_max` idle beats so the world keeps
            moving even when the player stays in one conversation.
          - On any fire, reset `turns_since_last_tick` to 0.

        Concurrency: a fresh `asyncio.Semaphore` per call, sized at
        `min(settings.tick_concurrency, TICK_CONCURRENCY_HARD_CAP)`.
        Each tick uses its OWN `CharacterAgent` instance so concurrent
        completions don't race on `agent.last_usage`.

        After fan-out, Commit 6 bundles every successful tick's
        `public_text` (parenthetical stripped — interior never leaves
        the agent) into a single unified-router call in tick mode.
        The router emits one canonical event capturing off-stage
        developments. We apply roster updates and append the canonical
        event to `ckpt.canonical_events`.
        Returns the per-character tick outputs primarily for tests
        and observability; the orchestrator caller doesn't otherwise
        use them.
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

        stagnation_fires = (
            sess.turns_since_last_tick >= settings.tick_stagnation_max
        )

        if not stagnation_fires:
            logger.debug(
                "Tick scheduler: no fire (turns_since_last_tick=%d, "
                "stagnation_ok=%s)",
                sess.turns_since_last_tick, stagnation_fires,
            )
            return []

        eligible = self._eligible_for_tick(ckpt, acted_this_turn)
        reason = "stagnation"
        if not eligible:
            # Still reset the counter — we DID try to fire, eligibility
            # was just empty (e.g. all NPCs are dormant, on-stage, or
            # already acted). Otherwise an all-on-stage beat would
            # never reset and the next turn would over-fire.
            sess.turns_since_last_tick = 0
            logger.info(
                "Tick scheduler: %s fire but no eligible NPCs "
                "(roster=%d, acted=%d; pinned, player, dormant, "
                "or intentions_disabled filtered all out); "
                "counter reset.",
                reason, len(ckpt.characters), len(acted_this_turn),
            )
            return []

        cap = min(
            max(1, settings.tick_concurrency), TICK_CONCURRENCY_HARD_CAP,
        )
        semaphore = asyncio.Semaphore(cap)

        logger.info(
            "Tick scheduler: %s fire — %d eligible NPC(s), "
            "concurrency cap %d",
            reason, len(eligible), cap,
        )

        async def _one(
            char: CharacterRecord,
        ) -> tuple[CharacterRecord, CharacterAgentOutput] | None:
            async with semaphore:
                # Fresh agent per tick so concurrent ticks don't race
                # on `agent.last_usage`.
                agent = CharacterAgent(self.client, self.prompt_mgr)
                try:
                    output = await agent.tick(
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
                    len(output.public_text), len(output.intent), usage,
                )
                return (char, output)

        results = await asyncio.gather(*[_one(c) for c in eligible])
        ticked = [r for r in results if r is not None]

        sess.turns_since_last_tick = 0

        logger.info(
            "Tick scheduler: %s fire complete — %d/%d ticks succeeded.",
            reason, len(ticked), len(eligible),
        )

        # Commit 6: tick fan-in to the unified router.
        # Bundle every successful tick's public prose (parenthetical
        # stripped — `output.public_text` only) into one router call
        # in tick mode. The router emits a single canonical event
        # capturing off-stage developments. No narrator pass — the
        # player wasn't there.
        if not ticked:
            return ticked

        tick_outputs = [
            (
                char.name,
                char.character_id,
                char.location or "",
                output.public_text,
            )
            for char, output in ticked
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
            # the fan-in router call fails, the per-character tick
            # outputs are still in the agents' rolling histories
            # (their future replays inherit the interior), and the
            # next /act will go through the on-stage router as
            # normal. World state just doesn't pick up the off-stage
            # canonical event on this beat.
            logger.exception(
                "Tick fan-in router call failed; off-stage agent "
                "outputs are in their own conversations but no "
                "canonical event lands this turn.",
            )
            return ticked

        if routed is None:
            return ticked

        self.char_mgr.apply_roster_updates(ckpt, routed)
        if routed.spawn:
            # Tick-driven spawns have no single acting actor location.
            # Pass "" so spawns fall back to either router-supplied
            # seed.location or the spawn LLM's authored location.
            await self.char_mgr.spawn_characters(
                ckpt, routed.spawn, acting_actor_location="",
            )
        # Append the tick canonical event to the world log so future
        # router calls + recap passes see it as part of session truth.
        # The narrator never composes off this entry (no human render
        # for tick events); this append exists for the router's
        # session_conversation continuity (which the dispatcher's
        # `route_tick_intentions` already handles).
        ckpt.canonical_events.append(routed)
        logger.info(
            "Tick fan-in routed: %d spawn(s); off-stage canonical event appended.",
            len(routed.spawn),
        )

        return ticked
