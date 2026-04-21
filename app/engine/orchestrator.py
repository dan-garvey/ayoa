"""Turn orchestrator — wires the full turn pipeline.

Sequence: Load checkpoint -> EventRouter -> Agents -> Narrator -> Save checkpoint.

Every role maintains a rolling conversation on the checkpoint; nothing on the
wire goes stateless. The orchestrator's job is to sequence the roles, apply
state updates from their outputs, and persist the checkpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.engine.character_agent import CharacterAgent
from app.engine.character_manager import CharacterManager
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.context_builder import (
    collect_player_ids,
    iter_agent_beats,
    resolve_acting_character,
)
from app.engine.event_router import EventRouter
from app.engine.narrator import Narrator
from app.engine.prompt_manager import PromptManager
from app.engine.validators import validate_all_outputs
from app.llm.client import LLMClient
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord, IncomingDirective
from app.schemas.checkpoint import CheckpointFile
from app.schemas.narrator import TranscriptEntry
from app.schemas.requests import TurnRequest
from app.schemas.responses import DebugPayload, PhaseLatency, TurnResponse

logger = logging.getLogger(__name__)

# Per-agent timeout in seconds
AGENT_TIMEOUT = 60.0
# Safety cap above per-session settings: even if someone cranks
# max_responders or tick_concurrency to a ridiculous value, we don't
# want to dispatch hundreds of parallel LLM calls from one turn. Real
# orgs should lower session settings, not raise this ceiling.
RESPONDERS_HARD_CAP = 8
TICK_CONCURRENCY_HARD_CAP = 16
# Cap on how many pending observations any single character accumulates
# between their own response turns (older entries drop off the front).
MAX_PENDING_OBSERVATIONS = 10
# Directive delegation depth: warn above this, drop above the hard cap.
# Catches chains like A → B → C → D where intent gets laundered through
# too many hops. Depth 1 = directive sent by an agent acting spontaneously.
DIRECTIVE_DEPTH_WARN = 2
DIRECTIVE_DEPTH_CAP = 10
# Content-length at which we warn about a runaway directive. Not a hard
# cap — legitimate in-character messages might be a paragraph or two.
# But a 1KB directive is a signal the agent is drifting into monologue
# territory and we want it visible in logs. No truncation.
DIRECTIVE_LENGTH_WARN = 1000


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

    Runs before any movement logic on the turn so downstream code can
    treat the new scenes as real.
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
    # created scene. Iterating over both endpoints catches the case
    # where A was declared with connected_to=[B] but B was declared
    # with connected_to=[] — B still gains the reverse edge here.
    for new_id in created_ids:
        new_scene = scene_graph.get(new_id)
        if not isinstance(new_scene, dict):
            continue

        # Forward edges from the new scene: mirror onto each neighbor.
        for conn in new_scene.get("connected_to", []) or []:
            neighbor = scene_graph.get(conn)
            if not isinstance(neighbor, dict):
                continue
            n_conns = list(neighbor.get("connected_to", []) or [])
            if new_id not in n_conns:
                n_conns.append(new_id)
                neighbor["connected_to"] = n_conns

        # Incoming edges from any other scene: mirror onto the new one.
        for other_id, other in scene_graph.items():
            if other_id == new_id or not isinstance(other, dict):
                continue
            if new_id in (other.get("connected_to", []) or []):
                my_conns = list(new_scene.get("connected_to", []) or [])
                if other_id not in my_conns:
                    my_conns.append(other_id)
                    new_scene["connected_to"] = my_conns


class Orchestrator:
    """Orchestrates a full turn through the pipeline."""

    def __init__(
        self,
        client: LLMClient,
        checkpoint_mgr: CheckpointManager,
        prompt_mgr: PromptManager,
    ):
        self.client = client
        self.prompt_mgr = prompt_mgr
        self.narrator = Narrator(client, prompt_mgr)
        self.event_router = EventRouter(client, prompt_mgr)
        self.agent_engine = CharacterAgent(client, prompt_mgr)
        self.char_mgr = CharacterManager(client, prompt_mgr)
        self.checkpoint_mgr = checkpoint_mgr

    async def process_turn(self, request: TurnRequest) -> TurnResponse:
        """Process a single turn end-to-end."""
        turn_start = time.monotonic()
        latencies: list[PhaseLatency] = []

        checkpoint = self.checkpoint_mgr.load_latest(request.session_id)

        # Resolve the acting character — whose action drives this turn. Falls
        # back to the creator's binding for single-player / legacy call sites
        # that don't pass acting_character_id on the request.
        acting_character_id, _, _ = resolve_acting_character(
            checkpoint, request.acting_character_id,
        )
        logger.info(
            "Turn %d for session %s (acting=%s)",
            checkpoint.session.turn_index,
            request.session_id,
            acting_character_id or "(none)",
        )

        # --- EventRouter: adjudicate + route in one pass, appending to session_conversation ---
        t0 = time.monotonic()
        routed = await self.event_router.run(
            request.user_input, checkpoint, acting_character_id=acting_character_id,
        )
        event = routed.canonical_event
        latencies.append(self._phase_latency(
            "event_router",
            t0,
            self.client.config.model_for_role("event_router"),
            [self.event_router.last_usage],
        ))

        # --- Apply event consequences on state ---
        # Order matters: (1) create new scenes FIRST so downstream movement
        # and spawn logic can target them this same turn, then
        # (2) scene transition, (3) roster status updates, (4) NPC moves,
        # (5) spawns.
        _apply_scene_creations(checkpoint, routed.scenes_created)

        if event.scene_delta.new_scene_id:
            old_scene = checkpoint.world_state.locations.current_scene_id
            new_scene = event.scene_delta.new_scene_id
            if new_scene in checkpoint.world_state.locations.scene_graph:
                checkpoint.world_state.locations.current_scene_id = new_scene
                # Move the acting character along with the scene. Without
                # this, character.location stays at whatever was assigned
                # at import time while current_scene_id moves, so scene-
                # presence checks (characters_present, observer routing)
                # see the player as off-stage even when they're the only
                # one on it.
                acting_char = next(
                    (c for c in checkpoint.characters if c.character_id == acting_character_id),
                    None,
                )
                if acting_char is not None:
                    acting_char.location = new_scene
                logger.info("Scene transition: %s -> %s", old_scene, new_scene)
            else:
                logger.warning(
                    "Event analysis suggested scene %s but it's not in the "
                    "scene graph and the router did not include it in "
                    "scenes_created — likely a hallucinated id; movement "
                    "ignored",
                    new_scene,
                )

        self.char_mgr.apply_roster_updates(checkpoint, routed)

        # Collect player-controlled character_ids once and reuse. Bindings
        # don't mutate during a turn, so recomputing on each consumer would
        # just re-scan the whole roster N times.
        player_ids = collect_player_ids(checkpoint)

        # Apply router-directed NPC movement. Player-bound characters are
        # skipped defensively — the router prompt forbids moving them, but
        # we don't trust the model to respect that every turn.
        scene_graph = checkpoint.world_state.locations.scene_graph
        for move in routed.roster_moves:
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
                (c for c in checkpoint.characters if c.character_id == move.character_id),
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

        if routed.spawn:
            t0 = time.monotonic()
            await self.char_mgr.spawn_characters(checkpoint, routed.spawn)
            latencies.append(PhaseLatency(
                phase="character_spawn",
                duration_ms=(time.monotonic() - t0) * 1000,
                model=self.client.config.model_for_role("agent"),
            ))

        # --- Responder selection ---
        # Exclude every player-controlled character, not just the creator's.
        # Other bound characters are humans too and never get AI responses.
        candidates = sorted(
            [o for o in routed.observers if o.character_id not in player_ids],
            key=lambda o: o.response_priority,
            reverse=True,
        )
        # Two clamps stack: the session setting (operator tunable via
        # /settings) and the module hard cap (safety rail so a bad
        # setting can't fan out to 200 agents).
        settings = checkpoint.session.config.settings
        response_cap = min(
            max(0, settings.max_responders),
            RESPONDERS_HARD_CAP,
        )
        responding = [o for o in candidates if o.response_priority >= 3][:response_cap]
        observing_only = [o for o in candidates if o not in responding]

        agent_outputs: list[CharacterAgentOutput] = []
        base_depths: dict[str, int] = {}
        logger.info(
            "Response selection: %d candidates, %d responding (cap=%d), %d observing",
            len(candidates), len(responding), response_cap, len(observing_only),
        )

        # --- Short-circuit: unwitnessed failure — skip NP2 ---
        if not event.world_adjudication.feasible and not responding:
            logger.info("Short-circuit: unwitnessed failure, returning router outcome")

            self._push_pending_observations(
                checkpoint, observing_only,
                event.world_adjudication.resolved_outcome,
            )

            checkpoint.transcript.append(TranscriptEntry(
                user=request.user_input,
                assistant=event.world_adjudication.resolved_outcome,
            ))
            checkpoint.session.turn_index += 1
            self.checkpoint_mgr.save(checkpoint)

            total_ms = (time.monotonic() - turn_start) * 1000
            logger.info(
                "Turn %d complete (short-circuit): %.0fms total",
                checkpoint.session.turn_index, total_ms,
            )

            return TurnResponse(
                session_id=request.session_id,
                checkpoint_id=f"ckpt_{checkpoint.session.turn_index:04d}",
                turn_index=checkpoint.session.turn_index,
                output_text=event.world_adjudication.resolved_outcome,
                debug=self._build_debug(
                    event, routed, [], {}, latencies,
                    total_ms, [], short_circuit=True,
                ) if request.debug else None,
            )

        # --- Agent fan-out: primary then parallel secondaries ---
        if responding:
            t0 = time.monotonic()
            # Each respond() call overwrites agent_engine.last_usage, so snapshot
            # after each call to sum at the end for the phase latency.
            agent_usages: list[dict] = []

            # Capture base directive depths BEFORE fan-out, because respond()
            # clears each character's incoming_directives during flush.
            # Outgoing directives get stamped at max(flushed) + 1.
            for obs in responding:
                char = self.char_mgr.get_character(checkpoint, obs.character_id)
                if char:
                    base_depths[obs.character_id] = max(
                        (d.depth for d in char.incoming_directives), default=0,
                    )

            primary_obs = responding[0]
            primary_char = self.char_mgr.get_character(
                checkpoint, primary_obs.character_id
            )
            if primary_char:
                try:
                    primary_result = await asyncio.wait_for(
                        self.agent_engine.respond(
                            primary_char, event.observable_facts, checkpoint,
                            prior_responses=[],
                            acting_character_id=acting_character_id,
                        ),
                        timeout=AGENT_TIMEOUT,
                    )
                    agent_outputs.append(primary_result)
                    agent_usages.append(dict(self.agent_engine.last_usage))
                except Exception as e:
                    logger.warning(
                        "Primary agent %s failed: %s", primary_obs.character_id, e
                    )

            secondary_obs = responding[1:]
            if secondary_obs:
                secondary_tasks = []
                for obs in secondary_obs:
                    char = self.char_mgr.get_character(checkpoint, obs.character_id)
                    if char:
                        task = asyncio.wait_for(
                            self.agent_engine.respond(
                                char, event.observable_facts, checkpoint,
                                prior_responses=list(agent_outputs),
                                acting_character_id=acting_character_id,
                            ),
                            timeout=AGENT_TIMEOUT,
                        )
                        secondary_tasks.append(task)

                if secondary_tasks:
                    # Parallel calls share the engine's last_usage, so we can't
                    # snapshot per-call cleanly. Instead we instrument by doing
                    # serial awaits with snapshots. (Still concurrent — gather
                    # completes them; we just walk results afterwards.)
                    results = await asyncio.gather(
                        *secondary_tasks, return_exceptions=True
                    )
                    for result in results:
                        if isinstance(result, Exception):
                            logger.warning("Secondary agent failed: %s", result)
                        else:
                            agent_outputs.append(result)
                    # Best-effort: record the last_usage snapshot once. For
                    # precise per-agent accounting with parallel dispatch, we
                    # would need respond() to return usage directly.
                    if self.agent_engine.last_usage:
                        agent_usages.append(dict(self.agent_engine.last_usage))

            latencies.append(self._phase_latency(
                "agent_fanout",
                t0,
                self.client.config.model_for_role("agent"),
                agent_usages,
            ))

        # --- Validate agent outputs for knowledge leakage ---
        # Every responding observer saw the canonical event's observable
        # facts (the router no longer per-character-filters them).
        observer_facts = {
            o.character_id: event.observable_facts for o in routed.observers
        }
        validation_results = validate_all_outputs(
            agent_outputs, checkpoint, observer_facts
        )
        validation_entries = [
            {
                "character_id": v.character_id,
                "passed": v.passed,
                "flags": [
                    {"text": f.leaked_text, "reason": f.reason}
                    for f in v.flags
                ],
            }
            for v in validation_results
        ]
        checkpoint.visibility_log.append({
            "turn": checkpoint.session.turn_index,
            "event_id": f"evt_{checkpoint.session.turn_index:04d}",
            "validations": validation_entries,
        })

        # --- Detect character self-introduction from dialogue ---
        known = set(checkpoint.world_state.known_characters)
        for output in agent_outputs:
            if output.character_id in known:
                continue
            char = next(
                (c for c in checkpoint.characters if c.character_id == output.character_id),
                None,
            )
            if not char:
                continue
            first_name = char.name.split()[0] if char.name else ""
            if not first_name:
                continue
            for line in output.public_response.dialogue:
                if first_name in line:
                    checkpoint.world_state.known_characters.append(output.character_id)
                    logger.info(
                        "Character introduced: %s (%s)", char.name, output.character_id
                    )
                    break

        # --- Narrator composition: appends to narrator_conversation ---
        t0 = time.monotonic()
        final = await self.narrator.compose(
            request.user_input, event, agent_outputs, checkpoint, routed,
            acting_character_id=acting_character_id,
        )
        latencies.append(self._phase_latency(
            "narrator_compose",
            t0,
            self.client.config.model_for_role("narrator"),
            [self.narrator.last_usage],
        ))

        # --- Apply agent outputs (objectives + directives) ---
        for output in agent_outputs:
            char = self.char_mgr.get_character(checkpoint, output.character_id)
            if char:
                self._apply_agent_private_updates(
                    checkpoint, char, output,
                    base_depth=base_depths.get(output.character_id, 0),
                )

        # --- Push turn observations to silent observers' pending queues ---
        npc_summary = self._build_npc_turn_summary(
            request.user_input, agent_outputs, checkpoint,
            acting_character_id=acting_character_id,
        )
        self._push_pending_observations(checkpoint, observing_only, npc_summary)

        # --- Display transcript + turn bookkeeping ---
        checkpoint.transcript.append(final.transcript_entry)
        checkpoint.session.turn_index += 1

        # --- Turn recap (kicked off now, awaited after tick pass so the
        # two run in parallel when both fire). A terse Haiku summary of
        # narrator state-level deltas; consumed by next turn's router.
        from app.engine.turn_recap import summarize_turn
        recap_task = asyncio.create_task(
            summarize_turn(
                self.client, self.prompt_mgr, event, final.final_text,
            )
        )

        # --- Off-stage tick pass. Fires on cadence OR on scene change,
        # whichever comes first. Scene-change reset keeps the world moving
        # when the player jumps between spaces; cadence keeps it moving
        # when the player sits in one scene for a while. Tick effects
        # land on the FOLLOWING turn the player sees (no prose this turn).
        acted_this_turn = {o.character_id for o in agent_outputs}
        acted_this_turn.add(acting_character_id)
        t0 = time.monotonic()
        tick_usages = await self._run_ticks(
            checkpoint, acted_this_turn, acting_character_id,
            player_ids=player_ids,
        )
        if tick_usages:
            latencies.append(self._phase_latency(
                "off_stage_ticks",
                t0,
                self.client.config.model_for_role("agent"),
                tick_usages,
            ))

        # Await the recap that was kicked off before the tick pass.
        # Errors here shouldn't fail the turn — an empty recap just means
        # next turn's router runs without the delta note, same as today.
        try:
            recap_note = await recap_task
        except Exception:
            logger.exception("turn_recap failed; pending_recap left empty")
            recap_note = ""
        checkpoint.session.pending_recap = recap_note

        # Single save at the end of the turn. Previously we saved after
        # narrator compose AND after ticks, which left a crash window:
        # the turn persisted but tick effects didn't, and a replay on
        # restart would re-run ticks non-idempotently. One save here
        # means the turn is atomic — either the whole turn (including
        # tick state) lands on disk, or none of it does, and a retry
        # replays cleanly from the prior checkpoint.
        self.checkpoint_mgr.save(checkpoint)

        total_ms = (time.monotonic() - turn_start) * 1000
        logger.info(
            "Turn %d complete: %d chars output, %d agents responded, %.0fms total",
            checkpoint.session.turn_index, len(final.final_text),
            len(agent_outputs), total_ms,
        )
        _log_cache_summary(latencies)

        return TurnResponse(
            session_id=request.session_id,
            checkpoint_id=f"ckpt_{checkpoint.session.turn_index:04d}",
            turn_index=checkpoint.session.turn_index,
            output_text=final.final_text,
            debug=self._build_debug(
                event, routed, agent_outputs, final.world_updates,
                latencies, total_ms, validation_entries, short_circuit=False,
            ) if request.debug else None,
        )

    def _phase_latency(
        self,
        phase: str,
        start_mono: float,
        model: str,
        usages: list[dict],
    ) -> PhaseLatency:
        """Build a PhaseLatency record summing token usage across one or
        more LLM calls. Each role stuffs a `prompt_render_ms` key alongside
        token counts in its `last_usage` dict, which we sum here so
        render overhead is visible as its own column in debug logs."""
        def sum_int(key: str) -> int:
            return sum(int(u.get(key, 0) or 0) for u in usages if u)

        def sum_float(key: str) -> float:
            return sum(float(u.get(key, 0.0) or 0.0) for u in usages if u)

        return PhaseLatency(
            phase=phase,
            duration_ms=(time.monotonic() - start_mono) * 1000,
            model=model,
            input_tokens=sum_int("prompt_tokens"),
            output_tokens=sum_int("completion_tokens"),
            cache_read_input_tokens=sum_int("cache_read_input_tokens"),
            cache_creation_input_tokens=sum_int("cache_creation_input_tokens"),
            prompt_render_ms=sum_float("prompt_render_ms"),
        )

    async def _run_ticks(
        self,
        checkpoint: CheckpointFile,
        acted_this_turn: set[str],
        acting_character_id: str,
        player_ids: set[str] | None = None,
    ) -> list[dict]:
        """Off-stage tick pass. Returns the list of per-tick usage dicts
        so the caller can fold them into phase latencies; empty list when
        no ticks fired. `player_ids` may be passed by the caller to avoid
        re-scanning the roster; falls back to computing it here for tests
        that invoke _run_ticks directly."""
        session = checkpoint.session
        current_scene = checkpoint.world_state.locations.current_scene_id
        scene_changed_raw = (
            session.tick_last_scene_id != ""
            and current_scene != session.tick_last_scene_id
        )
        scene_changed = (
            scene_changed_raw
            and session.config.settings.ticks_on_scene_change
        )

        session.tick_turn_counter += 1
        cadence_hit = session.tick_turn_counter >= session.tick_cadence

        if not (cadence_hit or scene_changed):
            # Keep last_scene_id seeded on the very first turn so the
            # scene-change comparison has a baseline going forward.
            if not session.tick_last_scene_id:
                session.tick_last_scene_id = current_scene
            return []

        # Reset regardless of which trigger fired.
        session.tick_turn_counter = 0
        session.tick_last_scene_id = current_scene

        if player_ids is None:
            player_ids = collect_player_ids(checkpoint)
        eligible = [
            c for c in checkpoint.characters
            if c.private_state.intentions_enabled
            and c.status.value == "active"
            and c.character_id not in player_ids
            and c.character_id not in acted_this_turn
        ]
        if not eligible:
            logger.info(
                "Tick trigger fired (cadence=%s scene_changed=%s) but no eligible characters",
                cadence_hit, scene_changed,
            )
            return []

        logger.info(
            "Off-stage tick: %d eligible (cadence=%s scene_changed=%s)",
            len(eligible), cadence_hit, scene_changed,
        )

        # Capture base depths before tick() clears each character's
        # incoming_directives. Same pattern as the response fan-out.
        base_depths = {
            c.character_id: max(
                (d.depth for d in c.incoming_directives), default=0,
            )
            for c in eligible
        }

        # Concurrency cap: session setting, clamped to a hard ceiling
        # so a misconfigured setting doesn't fire dozens of parallel
        # agent calls. Min of 1 so the semaphore is never nonsensical.
        tick_conc = max(
            1,
            min(session.config.settings.tick_concurrency, TICK_CONCURRENCY_HARD_CAP),
        )
        semaphore = asyncio.Semaphore(tick_conc)

        async def _tick_one(char):
            async with semaphore:
                try:
                    return char, await asyncio.wait_for(
                        self.agent_engine.tick(
                            char, checkpoint,
                            acting_character_id=acting_character_id,
                        ),
                        timeout=AGENT_TIMEOUT,
                    ), dict(self.agent_engine.last_usage)
                except Exception as e:
                    logger.warning("Tick failed for %s: %s", char.character_id, e)
                    return char, None, {}

        results = await asyncio.gather(*[_tick_one(c) for c in eligible])

        # Whether the session has opted into agent-driven scene creation.
        # When True, tick outputs' scenes_created are applied (before the
        # per-character moved_to so the character can walk INTO the scene
        # they just invented). When False, non-empty scenes_created is
        # dropped with a warning — the agent shouldn't have emitted any.
        allow_agent_scenes = checkpoint.session.config.settings.agents_can_create_scenes

        usages: list[dict] = []
        for char, output, usage in results:
            if output is None:
                continue

            scenes = output.private_updates.scenes_created
            if scenes:
                if allow_agent_scenes:
                    _apply_scene_creations(checkpoint, scenes)
                else:
                    logger.warning(
                        "Tick from %s emitted %d scenes_created but the "
                        "agents_can_create_scenes setting is off — dropped",
                        char.character_id, len(scenes),
                    )

            self._apply_agent_private_updates(
                checkpoint, char, output,
                base_depth=base_depths.get(char.character_id, 0),
            )
            usages.append(usage)

        return usages

    def _apply_agent_private_updates(
        self,
        checkpoint: CheckpointFile,
        character: CharacterRecord,
        output: CharacterAgentOutput,
        base_depth: int,
    ) -> None:
        """Persist objectives and route directives after an agent response.

        - Authoritative whole-list replacement for current_objectives.
        - Routes outgoing directives to targets' incoming queues, stamping
          from/turn/depth. Depth = max(flushed incoming depth) + 1, so a
          directive A sends purely from its own initiative has depth 1; a
          directive A sends in response to receiving B's directive has
          depth B.depth + 1.
        - Warns above DIRECTIVE_DEPTH_WARN, drops above DIRECTIVE_DEPTH_CAP.
        """
        private = output.private_updates

        # Whole-list replacement. Agent emits the complete set of objectives
        # it's still pursuing; anything missing is considered dropped.
        character.private_state.current_objectives = list(private.current_objectives)

        # Self-initiated movement (primarily from ticks). Validated against
        # the scene graph; ignored if unknown to avoid corrupting location.
        if private.moved_to:
            scene_graph = checkpoint.world_state.locations.scene_graph
            if private.moved_to in scene_graph:
                old = character.location
                character.location = private.moved_to
                logger.info(
                    "Character %s moved self: %s -> %s",
                    character.character_id, old or "(unset)", private.moved_to,
                )
            else:
                logger.warning(
                    "Character %s tried to move to unknown scene %r; ignored",
                    character.character_id, private.moved_to,
                )

        if not private.directives_sent:
            return

        new_depth = base_depth + 1
        if new_depth > DIRECTIVE_DEPTH_CAP:
            logger.warning(
                "Directive chain from %s exceeded depth cap %d; dropping %d outgoing",
                character.character_id, DIRECTIVE_DEPTH_CAP,
                len(private.directives_sent),
            )
            return
        if new_depth > DIRECTIVE_DEPTH_WARN:
            logger.warning(
                "Deep directive chain from %s (depth=%d, cap=%d)",
                character.character_id, new_depth, DIRECTIVE_DEPTH_CAP,
            )

        turn_idx = checkpoint.session.turn_index
        for directive in private.directives_sent:
            target = self.char_mgr.get_character(checkpoint, directive.to)
            if target is None:
                logger.warning(
                    "Directive from %s to unknown character %s — dropped",
                    character.character_id, directive.to,
                )
                continue
            # Delivered either way; length cap is advisory, not enforcing.
            # A runaway agent producing 5KB messages is a signal worth
            # surfacing without blocking legitimate longer communications.
            if len(directive.content) > DIRECTIVE_LENGTH_WARN:
                logger.warning(
                    "Long directive from %s -> %s: %d chars "
                    "(advisory threshold %d); still delivered.",
                    character.character_id, directive.to,
                    len(directive.content), DIRECTIVE_LENGTH_WARN,
                )
            target.incoming_directives.append(IncomingDirective(
                from_character_id=character.character_id,
                content=directive.content,
                turn=turn_idx,
                depth=new_depth,
            ))
            logger.info(
                "Directive routed: %s -> %s (depth %d): %s",
                character.character_id, directive.to, new_depth,
                directive.content[:80],
            )

    def _push_pending_observations(
        self,
        checkpoint: CheckpointFile,
        observers: list,
        summary: str,
    ) -> None:
        """Append this turn's observations to each silent observer's pending list.

        On the character's next response turn, their agent flushes these into
        the user message.
        """
        turn_idx = checkpoint.session.turn_index
        for obs in observers:
            char = self.char_mgr.get_character(checkpoint, obs.character_id)
            if not char:
                continue

            if obs.observation_level == "d":
                entry = f"[Turn {turn_idx}] {summary}"
            elif obs.observation_level == "i":
                entry = f"[Turn {turn_idx}] [Heard nearby] {summary}"
            else:
                entry = f"[Turn {turn_idx}] [Sensed disturbance] Something happened nearby."

            char.pending_observations.append(entry)
            if len(char.pending_observations) > MAX_PENDING_OBSERVATIONS:
                char.pending_observations = char.pending_observations[
                    -MAX_PENDING_OBSERVATIONS:
                ]

    def _build_npc_turn_summary(
        self,
        user_input: str,
        agent_outputs: list[CharacterAgentOutput],
        checkpoint: CheckpointFile,
        acting_character_id: str = "",
    ) -> str:
        """Build a turn summary in NPC-perspective terms for observation queues."""
        acting_char = next(
            (c for c in checkpoint.characters if c.character_id == acting_character_id),
            None,
        )
        acting_name = (
            acting_char.name if acting_char
            else (checkpoint.session.player_name or "The player")
        )
        parts = [f"{acting_name}: {user_input}"]

        for output, char in iter_agent_beats(agent_outputs, checkpoint):
            name = char.name if char else output.character_id
            pieces = list(output.public_response.actions)
            pieces.extend(f'said "{line}"' for line in output.public_response.dialogue)
            if pieces:
                parts.append(f"{name}: {'; '.join(pieces)}")

        return " | ".join(parts)

    def _build_debug(
        self,
        event,
        routed,
        agent_outputs: list[CharacterAgentOutput],
        world_updates: dict,
        latencies: list[PhaseLatency],
        total_ms: float,
        validation_entries: list[dict],
        short_circuit: bool,
    ) -> DebugPayload:
        models_used = {
            "event_router": self.client.config.model_for_role("event_router"),
            "agent": self.client.config.model_for_role("agent"),
        }
        if not short_circuit:
            models_used["narrator"] = self.client.config.model_for_role("narrator")

        return DebugPayload(
            canonical_event=event.model_dump(),
            router_output=routed.model_dump(),
            agent_outputs=[o.model_dump() for o in agent_outputs],
            world_updates=world_updates,
            latencies=latencies,
            total_duration_ms=total_ms,
            models_used=models_used,
            prompt_versions=self.prompt_mgr.get_all_versions(),
            validations=validation_entries,
        )


def _log_cache_summary(latencies: list[PhaseLatency]) -> None:
    """Per-turn cache + spend readout at INFO. Fires every turn so both
    hit rate and where-the-dollars-went are visible in normal logs.

    `hit_rate` is cache_read / (cache_read + cache_write + input) — the
    fraction of total prompt tokens that were served from cache. `share`
    is each phase's cost-weighted contribution to the turn's total:
    input_tokens + 1.25*cache_write + 0.1*cache_read + output_tokens.
    The weighting reflects Anthropic's billing ratios (cache writes are
    1.25x, reads are ~0.1x), so `share` tells you where dollars actually
    went, not just where raw tokens went."""
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
