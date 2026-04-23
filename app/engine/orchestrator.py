"""Turn orchestrator — v11 beat loop binding.

The old v8 pipeline ran a single-pass turn (EventRouter → agents → Narrator)
and returned one rendered output. v11 shifts to a beat-cascading state
machine in `app.engine.turn_loop.run_beat`; this module is now a thin
adapter that:

  1. loads the checkpoint,
  2. resolves which character is acting,
  3. acquires the per-(session, scene) scene lock so two concurrent
     /acts on the same scene serialize,
  4. validates the incoming /act against the scene's active_act_slot,
  5. runs one beat to completion via `run_beat`,
  6. applies roster side-effects of every event that closed this beat,
  7. saves the checkpoint,
  8. returns a `TurnResponse` carrying per-POV renders.

The only LLM-facing object the orchestrator constructs directly is the
`LLMDispatcher` — the single adapter that binds the router, narrator,
and character_agent modules into the protocol `run_beat` depends on.

Helpers kept from v8:
  - `_apply_scene_creations` (scene-graph growth is still orchestrator-
    owned; `run_beat` broadcasts canonical events but doesn't mutate the
    scene graph).
  - `_log_cache_summary` (per-turn cache/spend readout; currently unused
    in the v11 wrapper because per-phase latencies aren't gathered yet —
    left intact for when they come back).
"""

from __future__ import annotations

import asyncio
import logging

from app.engine.character_agent import CharacterAgent
from app.engine.character_manager import CharacterManager, _pinned_character_ids
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop import (
    BeatResult,
    SceneLockManager,
    SlotConflict,
    broadcast_event,
    cat_ii_is_ready,
    check_act_slot,
    close_cat_ii,
    format_slot_rejection,
    run_beat,
    _end_beat,
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
from app.schemas.event_router import EventRouterOutput
from app.schemas.requests import TurnRequest
from app.schemas.responses import PhaseLatency, TurnResponse

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


def _apply_scene_creations(checkpoint: CheckpointFile, creations) -> None:
    """Grow the scene graph with router-created scenes.

    Two passes:
      1. Create each scene: validate id uniqueness, filter connected_to
         refs to scenes that exist (either pre-existing or elsewhere in
         this same batch), dedupe.
      2. Enforce bidirectionality: for every forward edge A → B in the
         graph, ensure B → A also exists. This covers both directions
         of batch-internal edges regardless of declaration order, and
         closes the reverse edge when a new scene connects to a
         pre-existing one.

    Runs after every closed event in a beat so later events can reference
    scenes introduced earlier in the same beat.
    """
    if not creations:
        return

    scene_graph = checkpoint.world_state.locations.scene_graph
    batch_ids = {s.scene_id for s in creations if s.scene_id}
    created_ids: list[str] = []

    # Pass 1: create each new scene with its declared (and filtered)
    # connected_to list.
    for scene in creations:
        if not scene.scene_id:
            logger.warning("Scene creation with empty scene_id — ignored")
            continue
        if scene.scene_id in scene_graph:
            logger.warning(
                "Router tried to create scene %r but it already exists — ignored",
                scene.scene_id,
            )
            continue

        valid_connections: list[str] = []
        for conn in scene.connected_to:
            if conn == scene.scene_id:
                continue  # self-reference is nonsense, drop silently
            if conn in scene_graph or conn in batch_ids:
                if conn not in valid_connections:
                    valid_connections.append(conn)
            else:
                logger.warning(
                    "New scene %r connects to unknown scene %r — edge dropped",
                    scene.scene_id, conn,
                )

        scene_graph[scene.scene_id] = {
            "name": scene.name,
            "description": scene.description,
            "connected_to": valid_connections,
            "properties": {},
        }
        created_ids.append(scene.scene_id)
        logger.info(
            "Scene created: %s (id: %s, connects to: %s)",
            scene.name, scene.scene_id,
            ", ".join(valid_connections) or "(none)",
        )

    # Pass 2: ensure bidirectionality for edges touching any newly-
    # created scene.
    for new_id in created_ids:
        new_scene = scene_graph.get(new_id)
        if not isinstance(new_scene, dict):
            continue

        for conn in new_scene.get("connected_to", []) or []:
            neighbor = scene_graph.get(conn)
            if not isinstance(neighbor, dict):
                continue
            n_conns = list(neighbor.get("connected_to", []) or [])
            if new_id not in n_conns:
                n_conns.append(new_id)
                neighbor["connected_to"] = n_conns

        for other_id, other in scene_graph.items():
            if other_id == new_id or not isinstance(other, dict):
                continue
            if new_id in (other.get("connected_to", []) or []):
                my_conns = list(new_scene.get("connected_to", []) or [])
                if other_id not in my_conns:
                    my_conns.append(other_id)
                    new_scene["connected_to"] = my_conns


class Orchestrator:
    """Binds `turn_loop.run_beat` to the LLM/storage layers.

    One instance lives per `EngineBridge`; sessions in the same process
    share the `SceneLockManager` so concurrent /acts targeting the same
    scene serialize against a single asyncio.Lock.
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
        # One manager per Orchestrator. Same-scene /acts serialize here;
        # different scenes acquire independent locks.
        self.scene_locks = SceneLockManager()

    async def process_turn(self, request: TurnRequest) -> TurnResponse:
        """Process a single turn end-to-end, v11-style.

        Steps: load checkpoint → resolve actor & scene → acquire scene
        lock → slot check → run beat → apply per-event roster side-
        effects → save → build response.
        """
        ckpt = self.checkpoint_mgr.load_latest(request.session_id)

        # 1. Resolve the acting character.
        acting_id = self._resolve_acting_character(ckpt, request)

        # 2. Determine which scene the acting character occupies.
        scene_id = self._resolve_scene_id(ckpt, acting_id)

        logger.info(
            "Turn %d for session %s (acting=%s, scene=%s)",
            ckpt.session.turn_index, request.session_id, acting_id, scene_id,
        )

        # 3. Acquire per-scene lock. Prevents two concurrent /acts on the
        # same scene from both seeing FREE on their check_act_slot.
        lock = await self.scene_locks.get(request.session_id, scene_id)
        async with lock:
            # 4. Validate against the scene's active_act_slot.
            check = check_act_slot(ckpt, scene_id, acting_id)

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
                scene_id=scene_id,
                cat_ii_event_id=cat_ii_event_id,
            )

            # 6. Apply roster side-effects of every event that closed
            # this beat. `run_beat` broadcasts + renders but leaves
            # scene_creations / roster_moves / spawn / dormant / cull
            # for the orchestrator to apply. Walk the tail of
            # canonical_events matching the count it reports, paired
            # with `event_actor_ids` so _apply_roster_moves can recognize
            # actor-self-moves (the v11-r7h unified movement path).
            closed = beat_result.events_closed
            if closed > 0:
                closed_this_beat = ckpt.canonical_events[-closed:]
                actors = beat_result.event_actor_ids
                # Defensive: invariant says lengths match; if they don't
                # (e.g. a future refactor forgot to track an actor), pad
                # with None so apply still runs but self-moves get blocked
                # by the player-bound/pin guards rather than misattributed.
                if len(actors) != closed:
                    logger.warning(
                        "BeatResult event_actor_ids length %d != events_closed %d "
                        "in scene %s; self-moves on this beat will be skipped.",
                        len(actors), closed, scene_id,
                    )
                    actors = actors + [None] * (closed - len(actors))
                for evt, evt_actor in zip(closed_this_beat, actors):
                    _apply_scene_creations(ckpt, evt.scenes_created)
                    self.char_mgr.apply_roster_updates(ckpt, evt)
                    self._apply_roster_moves(ckpt, evt, actor_id=evt_actor)
                    # Spawns remain async LLM calls; if none declared,
                    # the helper is a no-op.
                    if evt.spawn:
                        await self.char_mgr.spawn_characters(ckpt, evt.spawn)

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
            # plus any `roster_moves` / `scenes_created` the agents
            # implied; we apply those mutations to the checkpoint
            # off-stage (no narrator pass — the player wasn't there).
            # Per-tick errors and the fan-in router error are both
            # logged-and-swallowed so a single LLM hiccup doesn't
            # drop the rest of the turn. Picks up the actor's
            # POST-beat scene (which may differ from `scene_id`
            # when the actor self-moved during the beat) so the
            # scheduler tracks where the player ACTUALLY is now.
            post_beat_scene = self._resolve_scene_id(ckpt, acting_id)
            await self._run_ticks(
                ckpt,
                acted_this_turn=set(beat_result.event_actor_ids),
                acting_id=acting_id,
                current_scene=post_beat_scene,
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

        Acquires the scene lock for the event's scene, re-checks
        readiness, drives `route_intention` on the adjudication path,
        closes the event, broadcasts the canonical result, fans renders
        out via `_end_beat`, applies roster side-effects, and saves.
        Returns a TurnResponse describing the resolution; if the event
        was already closed (race) returns an empty "cat_ii_stale"
        response.
        """
        ckpt = self.checkpoint_mgr.load_latest(session_id)
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

        scene_id = evt.scene_id
        lock = await self.scene_locks.get(session_id, scene_id)
        async with lock:
            # Re-read: another task may have closed this event while we
            # were waiting for the lock.
            ckpt = self.checkpoint_mgr.load_latest(session_id)
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
                    scene_id=scene_id,
                    cat_ii_event=evt_live,
                )
                close_cat_ii(ckpt, evt_live.event_id)
                if resolved.requires_responders:
                    raise ValueError(
                        "Cat II resolution returned nested Cat II "
                        "(Part C invariant violated)."
                    )
                broadcast_event(ckpt, resolved, scene_id)
                beat_result = await _end_beat(
                    ckpt, dispatcher, scene_id,
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
                        "events_closed %d in scene %s; self-moves on this "
                        "beat will be skipped.",
                        len(actors), beat_result.events_closed, scene_id,
                    )
                    actors = actors + [None] * (
                        beat_result.events_closed - len(actors)
                    )
                for ev, ev_actor in zip(closed_this_beat, actors):
                    _apply_scene_creations(ckpt, ev.scenes_created)
                    self.char_mgr.apply_roster_updates(ckpt, ev)
                    self._apply_roster_moves(ckpt, ev, actor_id=ev_actor)
                    if ev.spawn:
                        await self.char_mgr.spawn_characters(ckpt, ev.spawn)

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

    def _resolve_scene_id(
        self, ckpt: CheckpointFile, acting_id: str
    ) -> str:
        """The scene the acting character is currently in. Prefers the
        roster's `location` field; falls back to
        world_state.locations.current_scene_id for characters that have
        no location set (legacy imports, newly-spawned)."""
        for c in ckpt.characters:
            if c.character_id == acting_id and c.location:
                return c.location
        return ckpt.world_state.locations.current_scene_id

    def _apply_roster_moves(
        self,
        ckpt: CheckpointFile,
        routed: EventRouterOutput,
        actor_id: str | None = None,
    ) -> None:
        """Apply router-directed character relocations.

        `roster_moves` is the engine's single movement mechanism: every
        location change this turn — NPCs walking in/out, the acting
        character self-moving as the result of their /act, opening-turn
        placement — flows through here.

        Three guards filter out moves that would corrupt state:
          - **Pinned characters** (mid-action, the engine is waiting on
            their intention) cannot be externally relocated; the pin
            would be stranded. EXCEPT when the pinned character IS the
            actor on the closing event — the move IS the resolution of
            their own action.
          - **Player-bound characters** cannot be externally relocated;
            humans enter only via /act. EXCEPT when they're the actor —
            they ARE acting, and the move is the outcome of that act.
          - Unknown scene_ids and unknown character_ids are skipped
            (nothing to point at).

        `actor_id` identifies the acting character on the event being
        applied; pass None for paths with no single actor (legacy
        callers only — the v11 pipeline always knows the actor). With
        None the actor exceptions can't fire and player/pinned guards
        block all moves on those characters.
        """
        if not routed.roster_moves:
            return

        scene_graph = ckpt.world_state.locations.scene_graph
        pinned_ids = _pinned_character_ids(ckpt)
        player_ids = set(ckpt.session.character_bindings or {})
        if ckpt.session.player_character_id:
            player_ids.add(ckpt.session.player_character_id)

        for move in routed.roster_moves:
            is_actor_self_move = (
                actor_id is not None and move.character_id == actor_id
            )
            if move.character_id in pinned_ids and not is_actor_self_move:
                logger.warning(
                    "Router tried to move pinned character %s; ignored. "
                    "Pinned characters must resolve their open event before "
                    "they can be relocated (unless the move IS the resolution "
                    "of their own action).",
                    move.character_id,
                )
                continue
            if move.character_id in player_ids and not is_actor_self_move:
                logger.warning(
                    "Router tried to move player-bound character %s; ignored. "
                    "Player-bound characters move only via their own /act "
                    "(self-move on the event they initiated).",
                    move.character_id,
                )
                continue
            if move.to_scene not in scene_graph:
                logger.warning(
                    "Router roster_move to unknown scene %r for %s; ignored",
                    move.to_scene, move.character_id,
                )
                continue
            target = next(
                (c for c in ckpt.characters if c.character_id == move.character_id),
                None,
            )
            if target is None:
                logger.warning(
                    "Router roster_move for unknown character %s; ignored",
                    move.character_id,
                )
                continue
            old = target.location
            target.location = move.to_scene
            kind = "self-move" if is_actor_self_move else "roster move"
            logger.info(
                "%s: %s %s -> %s (%s)",
                kind, target.name, old or "(unset)", move.to_scene,
                move.reason or "no reason given",
            )

    # ---------------------------------------------------------- tick scheduler

    def _eligible_for_tick(
        self,
        ckpt: CheckpointFile,
        acted_this_turn: set[str],
        active_scene: str = "",
    ) -> list[CharacterRecord]:
        """Filter the roster to characters that should run an off-stage
        tick on this beat.

        Tick is for OFF-STAGE characters only. Anyone in the acting
        player's current scene already had the opportunity to advance
        the world via the on-stage cascade this turn — if the router
        didn't pick them, that was the router's call, and ticking
        them off-stage would step on that decision.

        Six guards:
          - `private_state.intentions_enabled` is True — importer flag
            for "this character matters enough to advance off-screen"
          - `status == active` — dormant/culled don't tick
          - NOT in any player binding (`character_bindings` keys or
            `session.player_character_id`) — humans don't get auto-ticked
          - NOT in `acted_this_turn` (the on-stage actor + any picked
            responders this beat) — they already had their say
          - NOT in `_pinned_character_ids(ckpt)` — pinned NPCs are
            mid-Cat-II, ticking races their pending resolution
          - NOT in `active_scene` — they're on-stage for this turn,
            cascade is the right channel for them to act

        TODO (world-time coherence, see CLAUDE.md): in multi-player
        sessions with concurrent scenes, an NPC in *any* currently-
        bound player's scene is on-stage for someone, not just for
        the actor of this turn. Today we only exclude the acting
        player's scene; revisit when multi-scene/multi-player turn
        ordering gets a real model.

        Order is roster order; that's also the order their tick
        outputs will reach the unified router in Commit 6.
        """
        pinned_ids = _pinned_character_ids(ckpt)
        player_ids = set(ckpt.session.character_bindings or {})
        if ckpt.session.player_character_id:
            player_ids.add(ckpt.session.player_character_id)

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
            if active_scene and char.location == active_scene:
                continue
            eligible.append(char)
        return eligible

    async def _run_ticks(
        self,
        ckpt: CheckpointFile,
        acted_this_turn: set[str],
        acting_id: str,
        current_scene: str,
    ) -> list[tuple[CharacterRecord, CharacterAgentOutput]]:
        """Decide whether to fire a tick pass this beat, and if so,
        fan out `CharacterAgent.tick()` for every eligible NPC under a
        bounded semaphore.

        Trigger model (Commit 5 / decision #9):
          - `turns_since_last_tick` increments unconditionally each
            beat (even when no tick fires).
          - **Scene-change branch**: fires if the actor's post-beat
            scene differs from the previous tracked scene AND the
            cooldown has elapsed AND `ticks_on_scene_change` is on.
            Cooldown prevents a tick on every turn during quick
            scene-hopping.
          - **Stagnation branch**: fires unconditionally after
            `tick_stagnation_max` idle beats so the world keeps
            moving even when the player camps in one scene.
          - On any fire, reset `turns_since_last_tick` to 0.
          - Always update `tick_last_scene_id` to the post-beat scene
            so the next call has a baseline.

        Concurrency: a fresh `asyncio.Semaphore` per call, sized at
        `min(settings.tick_concurrency, TICK_CONCURRENCY_HARD_CAP)`.
        Each tick uses its OWN `CharacterAgent` instance so concurrent
        completions don't race on `agent.last_usage`.

        After fan-out, Commit 6 bundles every successful tick's
        `public_text` (parenthetical stripped — interior never leaves
        the agent) into a single unified-router call in tick mode.
        The router emits one canonical event capturing off-stage
        developments + any roster_moves / scenes_created the agents
        declared. We apply those mutations to the checkpoint and
        append the canonical event to `ckpt.canonical_events`.
        Returns the per-character tick outputs primarily for tests
        and observability; the orchestrator caller doesn't otherwise
        use them.
        """
        sess = ckpt.session
        settings = sess.config.settings

        sess.turns_since_last_tick += 1

        scene_changed = bool(
            sess.tick_last_scene_id
            and current_scene
            and current_scene != sess.tick_last_scene_id
        )
        cooldown_satisfied = (
            sess.turns_since_last_tick >= settings.tick_scene_change_cooldown
        )
        scene_change_fires = (
            scene_changed
            and cooldown_satisfied
            and settings.ticks_on_scene_change
        )
        stagnation_fires = (
            sess.turns_since_last_tick >= settings.tick_stagnation_max
        )

        sess.tick_last_scene_id = current_scene

        if not (scene_change_fires or stagnation_fires):
            logger.debug(
                "Tick scheduler: no fire (turns_since_last_tick=%d, "
                "scene_changed=%s, cooldown_ok=%s, stagnation_ok=%s, "
                "ticks_on_scene_change=%s)",
                sess.turns_since_last_tick, scene_changed,
                cooldown_satisfied, stagnation_fires,
                settings.ticks_on_scene_change,
            )
            return []

        eligible = self._eligible_for_tick(
            ckpt, acted_this_turn, active_scene=current_scene,
        )
        reason = (
            "scene_change" if scene_change_fires else "stagnation"
        )
        if not eligible:
            # Still reset the counter — we DID try to fire, eligibility
            # was just empty (e.g. all NPCs are dormant, on-stage, or
            # already acted). Otherwise an all-on-stage beat would
            # never reset and the next turn would over-fire.
            sess.turns_since_last_tick = 0
            logger.info(
                "Tick scheduler: %s fire but no eligible NPCs "
                "(roster=%d, acted=%d; pinned, player, dormant, "
                "in-scene, or intentions_disabled filtered all out); "
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
            "concurrency cap %d, post-beat scene %s",
            reason, len(eligible), cap, current_scene or "(none)",
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
        # capturing off-stage developments plus any roster_moves /
        # scenes_created the agents implied. We apply those mutations
        # off-stage. No narrator pass — the player wasn't there.
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
            # normal. World state just doesn't pick up off-stage
            # roster_moves on this beat.
            logger.exception(
                "Tick fan-in router call failed; off-stage agent "
                "outputs are in their own conversations but no "
                "canonical event lands this turn.",
            )
            return ticked

        if routed is None:
            return ticked

        _apply_scene_creations(ckpt, routed.scenes_created)
        self.char_mgr.apply_roster_updates(ckpt, routed)
        # actor_id=None — no single off-stage actor; the player-bound
        # and pinned guards in `_apply_roster_moves` will keep the
        # router from accidentally relocating any human or any
        # mid-Cat-II NPC even if it tries.
        self._apply_roster_moves(ckpt, routed, actor_id=None)
        if routed.spawn:
            await self.char_mgr.spawn_characters(ckpt, routed.spawn)
        # Append the tick canonical event to the world log so future
        # router calls + recap passes see it as part of session truth.
        # The narrator never composes off this entry (no human render
        # for tick events); this append exists for the router's
        # session_conversation continuity (which the dispatcher's
        # `route_tick_intentions` already handles) AND for any future
        # cross-scene observability path that walks the canonical
        # log.
        ckpt.canonical_events.append(routed)
        logger.info(
            "Tick fan-in routed: %d roster_move(s), %d scene(s) "
            "created, %d spawn(s); off-stage canonical event appended.",
            len(routed.roster_moves), len(routed.scenes_created),
            len(routed.spawn),
        )

        return ticked


def _log_cache_summary(latencies: list[PhaseLatency]) -> None:
    """Per-turn cache + spend readout at INFO. Retained from the v8
    pipeline; currently unused by the v11 wrapper because per-phase
    latencies aren't yet reconstructed from the dispatcher. Re-enable
    by feeding latencies collected inside LLMDispatcher/run_beat once
    that plumbing exists."""
    if not latencies:
        return

    def cost_units(l: PhaseLatency) -> float:
        return (
            l.input_tokens
            + 1.25 * l.cache_creation_input_tokens
            + 0.1 * l.cache_read_input_tokens
            + l.output_tokens
        )

    total_read = sum(l.cache_read_input_tokens for l in latencies)
    total_write = sum(l.cache_creation_input_tokens for l in latencies)
    total_in = sum(l.input_tokens for l in latencies)
    total_out = sum(l.output_tokens for l in latencies)
    total_prompt = total_read + total_write + total_in
    turn_rate = (total_read / total_prompt) if total_prompt else 0.0
    turn_cost = sum(cost_units(l) for l in latencies) or 1.0
    top = max(latencies, key=cost_units)
    logger.info(
        "Cache usage this turn: read=%d write=%d fresh=%d out=%d  "
        "hit_rate=%.1f%%  top_spender=%s (%.0f%%)",
        total_read, total_write, total_in, total_out,
        turn_rate * 100,
        top.phase, cost_units(top) / turn_cost * 100,
    )
    for l in latencies:
        prompt_tot = (
            l.cache_read_input_tokens
            + l.cache_creation_input_tokens
            + l.input_tokens
        )
        rate = (l.cache_read_input_tokens / prompt_tot) if prompt_tot else 0.0
        share = cost_units(l) / turn_cost
        logger.info(
            "  phase=%-16s read=%5d write=%5d fresh=%5d out=%5d  "
            "hit=%.1f%%  share=%.0f%%",
            l.phase,
            l.cache_read_input_tokens,
            l.cache_creation_input_tokens,
            l.input_tokens,
            l.output_tokens,
            rate * 100,
            share * 100,
        )
