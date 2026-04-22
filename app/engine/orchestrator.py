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

import logging

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
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.requests import TurnRequest
from app.schemas.responses import PhaseLatency, TurnResponse

logger = logging.getLogger(__name__)


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
            # canonical_events matching the count it reports.
            closed = beat_result.events_closed
            if closed > 0:
                closed_this_beat = ckpt.canonical_events[-closed:]
                for evt in closed_this_beat:
                    _apply_scene_creations(ckpt, evt.scenes_created)
                    self.char_mgr.apply_roster_updates(ckpt, evt)
                    self._apply_roster_moves(ckpt, evt)
                    # Spawns remain async LLM calls; if none declared,
                    # the helper is a no-op.
                    if evt.spawn:
                        await self.char_mgr.spawn_characters(ckpt, evt.spawn)

            # 7. Save. run_beat has already mutated active_act_slots,
            # open_cat_ii_events, render_buffers, canonical_events, and
            # (through the dispatcher) narrator_conversations.
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
                )
            else:
                # Still pending responders — nothing to adjudicate yet.
                beat_result = BeatResult(
                    renders={}, events_closed=0,
                    ended_reason="cat_ii_pending",
                )

            # Apply side-effects for each newly closed event.
            if beat_result.events_closed > 0:
                closed_this_beat = ckpt.canonical_events[
                    -beat_result.events_closed:
                ]
                for ev in closed_this_beat:
                    _apply_scene_creations(ckpt, ev.scenes_created)
                    self.char_mgr.apply_roster_updates(ckpt, ev)
                    self._apply_roster_moves(ckpt, ev)
                    if ev.spawn:
                        await self.char_mgr.spawn_characters(ckpt, ev.spawn)

            if beat_result.events_closed > 0:
                ckpt.session.turn_index += 1
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
        self, ckpt: CheckpointFile, routed: EventRouterOutput
    ) -> None:
        """Apply router-directed NPC relocations. Guards:
          - pinned characters (initiator or Cat II responder) are skipped,
            because moving them would strand their pin.
          - player-bound characters are skipped; they only move via their
            own /act.
          - unknown scene_ids are skipped.
          - unknown character_ids are skipped.
        """
        if not routed.roster_moves:
            return

        scene_graph = ckpt.world_state.locations.scene_graph
        pinned_ids = _pinned_character_ids(ckpt)
        player_ids = set(ckpt.session.character_bindings or {})
        if ckpt.session.player_character_id:
            player_ids.add(ckpt.session.player_character_id)

        for move in routed.roster_moves:
            if move.character_id in pinned_ids:
                logger.warning(
                    "Router tried to move pinned character %s; ignored. "
                    "Pinned characters must resolve their open event before "
                    "they can be relocated.",
                    move.character_id,
                )
                continue
            if move.character_id in player_ids:
                logger.warning(
                    "Router tried to move player-bound character %s; ignored",
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
            logger.info(
                "Roster move: %s %s -> %s (%s)",
                target.name, old or "(unset)", move.to_scene,
                move.reason or "no reason given",
            )


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
