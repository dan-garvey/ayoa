"""Turn orchestrator — wires the full turn pipeline.

Sequence: Load checkpoint -> NP1+Discriminator or EventRouter -> Agents -> NP2 -> Save checkpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from app.engine.character_agent import CharacterAgent
from app.engine.character_manager import CharacterManager
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.discriminator import Discriminator
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
# Maximum observation queue entries per character
MAX_OBSERVATION_QUEUE = 10


class Orchestrator:
    """Orchestrates a full turn through the pipeline."""

    def __init__(
        self,
        client: LLMClient,
        checkpoint_mgr: CheckpointManager,
        prompt_mgr: PromptManager,
        merged_event_router: bool | None = None,
    ):
        self.client = client
        self.prompt_mgr = prompt_mgr
        self.narrator = Narrator(client, prompt_mgr)
        self.discriminator = Discriminator(client, prompt_mgr)
        self.event_router = EventRouter(client, prompt_mgr)
        self.agent_engine = CharacterAgent(client, prompt_mgr)
        self.char_mgr = CharacterManager(client, prompt_mgr)
        self.checkpoint_mgr = checkpoint_mgr
        self.use_merged_event_router = (
            merged_event_router
            if merged_event_router is not None
            else os.environ.get("INTFIC_MERGED_EVENT_ROUTER", "").lower()
            in {"1", "true", "yes", "on"}
        )

    async def process_turn(self, request: TurnRequest) -> TurnResponse:
        """Process a single turn through the full pipeline.

        Steps:
        1. Load checkpoint
        2. NP1: Adjudicate
        3. Discriminator: Perception gating
        4. Agents: Fan-out (parallel)
        5. NP2: Compose final prose
        6. Apply state updates
        7. Save checkpoint
        8. Return response
        """
        turn_start = time.monotonic()
        latencies: list[PhaseLatency] = []

        # Step 1: Load checkpoint
        checkpoint = self.checkpoint_mgr.load_latest(request.session_id)
        logger.info(
            "Turn %d for session %s",
            checkpoint.session.turn_index,
            request.session_id,
        )

        if self.use_merged_event_router:
            t0 = time.monotonic()
            routed = await self.event_router.run(request.user_input, checkpoint)
            event = routed.canonical_event
            disc_output = routed.to_discriminator_output()
            latencies.append(PhaseLatency(
                phase="event_router",
                duration_ms=(time.monotonic() - t0) * 1000,
                model=self.client.config.model_for_role("event_router"),
            ))
        else:
            # Step 2: NP1 Adjudicate
            t0 = time.monotonic()
            event = await self.narrator.phase_1(request.user_input, checkpoint)
            latencies.append(PhaseLatency(
                phase="np1_adjudicate",
                duration_ms=(time.monotonic() - t0) * 1000,
                model=self.client.config.model_for_role("narrator"),
            ))

            # Step 3: Discriminator
            t0 = time.monotonic()
            disc_output = await self.discriminator.run(event, checkpoint)
            latencies.append(PhaseLatency(
                phase="discriminator",
                duration_ms=(time.monotonic() - t0) * 1000,
                model=self.client.config.model_for_role("discriminator"),
            ))

        # Step 2.5 / merged equivalent: Apply scene transition before roster routing
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

        # Apply roster updates (dormancy, culling)
        self.char_mgr.apply_roster_updates(checkpoint, disc_output)

        # Step 3.5: Spawn new characters if requested
        if disc_output.spawn:
            t0 = time.monotonic()
            await self.char_mgr.spawn_characters(checkpoint, disc_output.spawn)
            latencies.append(PhaseLatency(
                phase="character_spawn",
                duration_ms=(time.monotonic() - t0) * 1000,
                model=self.client.config.model_for_role("agent"),
            ))

        # Step 4: Agent selection — priority-based with response cap
        player_id = checkpoint.session.player_character_id
        candidates = sorted(
            [o for o in disc_output.observers if o.character_id != player_id],
            key=lambda o: o.response_priority,
            reverse=True,
        )
        response_cap = min(
            disc_output.suggested_response_cap, MAX_RESPONDERS
        )
        responding = [
            o for o in candidates if o.response_priority >= 3
        ][:response_cap]
        observing_only = [o for o in candidates if o not in responding]

        agent_outputs: list[CharacterAgentOutput] = []

        logger.info(
            "Response selection: %d candidates, %d responding (cap=%d), %d observing",
            len(candidates), len(responding), response_cap, len(observing_only),
        )

        # Short-circuit: unwitnessed failure
        # If infeasible and no NPCs should respond, return NP1's outcome directly.
        if not event.world_adjudication.feasible and not responding:
            logger.info(
                "Short-circuit: unwitnessed failure, returning NP1 outcome directly"
            )

            # Populate observation queues for present-but-silent characters
            self._populate_observation_queues(
                checkpoint, observing_only,
                event.world_adjudication.resolved_outcome,
            )

            transcript_entry = TranscriptEntry(
                user=request.user_input,
                assistant=event.world_adjudication.resolved_outcome,
            )
            checkpoint.transcript.append(transcript_entry)
            checkpoint.session.turn_index += 1

            await self.narrator.compress_transcript(checkpoint)

            self.checkpoint_mgr.save(checkpoint)
            checkpoint_id = f"ckpt_{checkpoint.session.turn_index:04d}"

            total_ms = (time.monotonic() - turn_start) * 1000
            logger.info(
                "Turn %d complete (short-circuit): %.0fms total",
                checkpoint.session.turn_index,
                total_ms,
            )

            debug = None
            if request.debug:
                prompt_versions = self.prompt_mgr.get_all_versions()
                if self.use_merged_event_router:
                    models_used = {
                        "event_router": self.client.config.model_for_role("event_router"),
                        "agent": self.client.config.model_for_role("agent"),
                    }
                else:
                    models_used = {
                        "narrator": self.client.config.model_for_role("narrator"),
                        "discriminator": self.client.config.model_for_role("discriminator"),
                        "agent": self.client.config.model_for_role("agent"),
                    }
                debug = DebugPayload(
                    canonical_event=event.model_dump(),
                    discriminator=disc_output.model_dump(),
                    agent_outputs=[],
                    world_updates={},
                    latencies=latencies,
                    total_duration_ms=total_ms,
                    models_used=models_used,
                    prompt_versions=prompt_versions,
                    validations=[],
                )

            return TurnResponse(
                session_id=request.session_id,
                checkpoint_id=checkpoint_id,
                turn_index=checkpoint.session.turn_index,
                output_text=event.world_adjudication.resolved_outcome,
                debug=debug,
            )

        # Step 4a: Sequential agent execution
        # Phase 1: Primary responder (highest priority)
        # Phase 2: Secondary responders in parallel, each seeing primary's output
        if responding:
            t0 = time.monotonic()

            # Primary responder
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
                        ),
                        timeout=AGENT_TIMEOUT,
                    )
                    agent_outputs.append(primary_result)
                except Exception as e:
                    logger.warning("Primary agent %s failed: %s",
                                   primary_obs.character_id, e)

            # Secondary responders (see primary's output)
            secondary_obs = responding[1:]
            if secondary_obs:
                secondary_tasks = []
                for obs in secondary_obs:
                    char = self.char_mgr.get_character(
                        checkpoint, obs.character_id
                    )
                    if char:
                        task = asyncio.wait_for(
                            self.agent_engine.respond(
                                char, obs.facts, checkpoint,
                                prior_responses=list(agent_outputs),
                            ),
                            timeout=AGENT_TIMEOUT,
                        )
                        secondary_tasks.append(task)

                if secondary_tasks:
                    results = await asyncio.gather(
                        *secondary_tasks, return_exceptions=True
                    )
                    for result in results:
                        if isinstance(result, Exception):
                            logger.warning("Secondary agent failed: %s", result)
                        else:
                            agent_outputs.append(result)

            latencies.append(PhaseLatency(
                phase="agent_fanout",
                duration_ms=(time.monotonic() - t0) * 1000,
                model=self.client.config.model_for_role("agent"),
            ))

        # Step 4.5: Validate agent outputs for knowledge leakage
        observer_facts = {
            o.character_id: o.facts for o in disc_output.observers
        }
        validation_results = validate_all_outputs(
            agent_outputs, checkpoint, observer_facts
        )
        # Log to visibility log (v1: warn only, don't filter)
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
        visibility_entry = {
            "turn": checkpoint.session.turn_index,
            "event_id": event.event_id,
            "validations": validation_entries,
        }
        checkpoint.visibility_log.append(visibility_entry)

        # Step 4.9: Detect character introductions from dialogue
        # If a character says their own name, they've introduced themselves
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
            # Check if any dialogue line contains the character's first name
            first_name = char.name.split()[0]
            for line in output.public_response.dialogue:
                if first_name in line:
                    checkpoint.world_state.known_characters.append(output.character_id)
                    logger.info("Character introduced: %s (%s)", char.name, output.character_id)
                    break

        # Step 5: NP2 Compose
        t0 = time.monotonic()
        final = await self.narrator.phase_2(
            request.user_input, event, agent_outputs, checkpoint, disc_output
        )
        latencies.append(PhaseLatency(
            phase="np2_compose",
            duration_ms=(time.monotonic() - t0) * 1000,
            model=self.client.config.model_for_role("narrator"),
        ))

        # Step 6: Apply state updates
        for output in agent_outputs:
            self.char_mgr.apply_agent_output(checkpoint, output)

        # Step 6.1: Populate observation queues for non-responding observers
        # Build an NPC-perspective summary using character names (not player-
        # perspective descriptions from NP2's turn_summary).
        npc_summary = self._build_npc_turn_summary(
            request.user_input, agent_outputs, checkpoint,
        )
        self._populate_observation_queues(
            checkpoint, observing_only, npc_summary,
        )

        # Update transcript
        checkpoint.transcript.append(final.transcript_entry)

        # Advance turn
        checkpoint.session.turn_index += 1

        # Step 6.5: Compress transcript if needed
        compressed = await self.narrator.compress_transcript(checkpoint)
        if compressed:
            logger.info("Transcript was compressed")

        # Step 7: Save checkpoint
        self.checkpoint_mgr.save(checkpoint)
        checkpoint_id = f"ckpt_{checkpoint.session.turn_index:04d}"

        total_ms = (time.monotonic() - turn_start) * 1000
        logger.info(
            "Turn %d complete: %d chars output, %d agents responded, %.0fms total",
            checkpoint.session.turn_index,
            len(final.final_text),
            len(agent_outputs),
            total_ms,
        )

        # Step 8: Build response
        debug = None
        if request.debug:
            prompt_versions = self.prompt_mgr.get_all_versions()
            if self.use_merged_event_router:
                models_used = {
                    "event_router": self.client.config.model_for_role("event_router"),
                    "narrator": self.client.config.model_for_role("narrator"),
                    "agent": self.client.config.model_for_role("agent"),
                }
            else:
                models_used = {
                    "narrator": self.client.config.model_for_role("narrator"),
                    "discriminator": self.client.config.model_for_role("discriminator"),
                    "agent": self.client.config.model_for_role("agent"),
                }
            debug = DebugPayload(
                canonical_event=event.model_dump(),
                discriminator=disc_output.model_dump(),
                agent_outputs=[o.model_dump() for o in agent_outputs],
                world_updates=final.world_updates,
                latencies=latencies,
                total_duration_ms=total_ms,
                models_used=models_used,
                prompt_versions=prompt_versions,
                validations=validation_entries,
            )

        return TurnResponse(
            session_id=request.session_id,
            checkpoint_id=checkpoint_id,
            turn_index=checkpoint.session.turn_index,
            output_text=final.final_text,
            debug=debug,
        )

    def _populate_observation_queues(
        self,
        checkpoint: CheckpointFile,
        observers: list,
        summary: str,
    ) -> None:
        """Add turn observations to non-responding characters' queues."""
        turn_idx = checkpoint.session.turn_index
        queued_count = 0
        for obs in observers:
            char = self.char_mgr.get_character(checkpoint, obs.character_id)
            if not char:
                logger.debug("Queue skip: character %s not found", obs.character_id)
                continue

            if obs.observation_level == "direct":
                entry = f"[Turn {turn_idx}] {summary}"
            elif obs.observation_level == "indirect":
                entry = f"[Turn {turn_idx}] [Heard nearby] {summary}"
            else:
                entry = f"[Turn {turn_idx}] [Sensed disturbance] Something happened nearby."

            char.memory.observation_queue.append(entry)
            queued_count += 1

            # Cap the queue
            if len(char.memory.observation_queue) > MAX_OBSERVATION_QUEUE:
                char.memory.observation_queue = char.memory.observation_queue[
                    -MAX_OBSERVATION_QUEUE:
                ]

        if queued_count:
            logger.info(
                "Observation queues: %d characters updated", queued_count
            )

    def _build_npc_turn_summary(
        self,
        user_input: str,
        agent_outputs: list[CharacterAgentOutput],
        checkpoint: CheckpointFile,
    ) -> str:
        """Build a turn summary using character names for NPC consumption."""
        player_name = checkpoint.session.player_name or "The player"
        parts = [f"{player_name}: {user_input}"]

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
