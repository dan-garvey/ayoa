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
from app.engine.context_builder import collect_player_ids
from app.engine.event_router import EventRouter
from app.engine.narrator import Narrator
from app.engine.prompt_manager import PromptManager
from app.engine.validators import validate_all_outputs
from app.llm.client import LLMClient
from app.schemas.agents import CharacterAgentOutput
from app.schemas.checkpoint import CheckpointFile
from app.schemas.narrator import TranscriptEntry
from app.schemas.requests import TurnRequest
from app.schemas.responses import DebugPayload, PhaseLatency, TurnResponse

logger = logging.getLogger(__name__)

# Per-agent timeout in seconds
AGENT_TIMEOUT = 60.0
# Maximum number of agents that can respond per turn
MAX_RESPONDERS = 3
# Cap on how many pending observations any single character accumulates
# between their own response turns (older entries drop off the front).
MAX_PENDING_OBSERVATIONS = 10


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
        acting_character_id = (
            request.acting_character_id
            or checkpoint.session.player_character_id
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
        disc_output = routed.to_discriminator_output()
        latencies.append(self._phase_latency(
            "event_router",
            t0,
            self.client.config.model_for_role("event_router"),
            [self.event_router.last_usage],
        ))

        # --- Apply event consequences on state (scene transitions, roster updates) ---
        if event.scene_delta.new_scene_id:
            old_scene = checkpoint.world_state.locations.current_scene_id
            new_scene = event.scene_delta.new_scene_id
            if new_scene in checkpoint.world_state.locations.scene_graph:
                checkpoint.world_state.locations.current_scene_id = new_scene
                logger.info("Scene transition: %s -> %s", old_scene, new_scene)
            else:
                logger.warning(
                    "Event analysis suggested scene %s but it's not in the scene graph",
                    new_scene,
                )

        self.char_mgr.apply_roster_updates(checkpoint, disc_output)

        if disc_output.spawn:
            t0 = time.monotonic()
            await self.char_mgr.spawn_characters(checkpoint, disc_output.spawn)
            latencies.append(PhaseLatency(
                phase="character_spawn",
                duration_ms=(time.monotonic() - t0) * 1000,
                model=self.client.config.model_for_role("agent"),
            ))

        # --- Responder selection ---
        # Exclude every player-controlled character, not just the creator's.
        # Other bound characters are humans too and never get AI responses.
        player_ids = collect_player_ids(checkpoint)
        candidates = sorted(
            [o for o in disc_output.observers if o.character_id not in player_ids],
            key=lambda o: o.response_priority,
            reverse=True,
        )
        response_cap = min(disc_output.suggested_response_cap, MAX_RESPONDERS)
        responding = [o for o in candidates if o.response_priority >= 3][:response_cap]
        observing_only = [o for o in candidates if o not in responding]

        agent_outputs: list[CharacterAgentOutput] = []
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
                    event, disc_output, [], {}, latencies,
                    total_ms, [], short_circuit=True,
                ) if request.debug else None,
            )

        # --- Agent fan-out: primary then parallel secondaries ---
        if responding:
            t0 = time.monotonic()
            # Each respond() call overwrites agent_engine.last_usage, so snapshot
            # after each call to sum at the end for the phase latency.
            agent_usages: list[dict] = []

            primary_obs = responding[0]
            primary_char = self.char_mgr.get_character(
                checkpoint, primary_obs.character_id
            )
            if primary_char:
                try:
                    primary_result = await asyncio.wait_for(
                        self.agent_engine.respond(
                            primary_char, primary_obs.facts, checkpoint,
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
                                char, obs.facts, checkpoint,
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
        observer_facts = {o.character_id: o.facts for o in disc_output.observers}
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
            "event_id": event.event_id,
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
            request.user_input, event, agent_outputs, checkpoint, disc_output,
            acting_character_id=acting_character_id,
        )
        latencies.append(self._phase_latency(
            "narrator_compose",
            t0,
            self.client.config.model_for_role("narrator"),
            [self.narrator.last_usage],
        ))

        # --- Apply agent outputs (attitude deltas) ---
        for output in agent_outputs:
            self.char_mgr.apply_agent_output(checkpoint, output)

        # --- Push turn observations to silent observers' pending queues ---
        npc_summary = self._build_npc_turn_summary(
            request.user_input, agent_outputs, checkpoint,
            acting_character_id=acting_character_id,
        )
        self._push_pending_observations(checkpoint, observing_only, npc_summary)

        # --- Display transcript + turn bookkeeping ---
        checkpoint.transcript.append(final.transcript_entry)
        checkpoint.session.turn_index += 1
        self.checkpoint_mgr.save(checkpoint)

        total_ms = (time.monotonic() - turn_start) * 1000
        logger.info(
            "Turn %d complete: %d chars output, %d agents responded, %.0fms total",
            checkpoint.session.turn_index, len(final.final_text),
            len(agent_outputs), total_ms,
        )

        return TurnResponse(
            session_id=request.session_id,
            checkpoint_id=f"ckpt_{checkpoint.session.turn_index:04d}",
            turn_index=checkpoint.session.turn_index,
            output_text=final.final_text,
            debug=self._build_debug(
                event, disc_output, agent_outputs, final.world_updates,
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
        """Build a PhaseLatency record summing token usage across one or more LLM calls."""
        def sum_field(key: str) -> int:
            return sum(int(u.get(key, 0) or 0) for u in usages if u)

        return PhaseLatency(
            phase=phase,
            duration_ms=(time.monotonic() - start_mono) * 1000,
            model=model,
            input_tokens=sum_field("prompt_tokens"),
            output_tokens=sum_field("completion_tokens"),
            cache_read_input_tokens=sum_field("cache_read_input_tokens"),
            cache_creation_input_tokens=sum_field("cache_creation_input_tokens"),
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

            if obs.observation_level == "direct":
                entry = f"[Turn {turn_idx}] {summary}"
            elif obs.observation_level == "indirect":
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

        for output in agent_outputs:
            char = self.char_mgr.get_character(checkpoint, output.character_id)
            name = char.name if char else output.character_id
            pieces = []
            for action in output.public_response.actions:
                pieces.append(action)
            for line in output.public_response.dialogue:
                pieces.append(f'said "{line}"')
            if pieces:
                parts.append(f"{name}: {'; '.join(pieces)}")

        return " | ".join(parts)

    def _build_debug(
        self,
        event,
        disc_output,
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
            discriminator=disc_output.model_dump(),
            agent_outputs=[o.model_dump() for o in agent_outputs],
            world_updates=world_updates,
            latencies=latencies,
            total_duration_ms=total_ms,
            models_used=models_used,
            prompt_versions=self.prompt_mgr.get_all_versions(),
            validations=validation_entries,
        )
