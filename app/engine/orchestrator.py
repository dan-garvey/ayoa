"""Turn orchestrator — wires the full turn pipeline.

Sequence: Load checkpoint -> NP1 -> Discriminator -> Agents (parallel) -> NP2 -> Save checkpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.engine.character_agent import CharacterAgent
from app.engine.character_manager import CharacterManager
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.discriminator import Discriminator
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
        self.discriminator = Discriminator(client, prompt_mgr)
        self.agent_engine = CharacterAgent(client, prompt_mgr)
        self.char_mgr = CharacterManager(client, prompt_mgr)
        self.checkpoint_mgr = checkpoint_mgr

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

        # Step 2: NP1 Adjudicate
        t0 = time.monotonic()
        event = await self.narrator.phase_1(request.user_input, checkpoint)
        latencies.append(PhaseLatency(
            phase="np1_adjudicate",
            duration_ms=(time.monotonic() - t0) * 1000,
            model=self.client.config.model_for_role("narrator"),
        ))

        # Step 2.5: Apply scene transition if NP1 detected movement
        if event.scene_delta.new_scene_id:
            old_scene = checkpoint.world_state.locations.current_scene_id
            new_scene = event.scene_delta.new_scene_id
            if new_scene in checkpoint.world_state.locations.scene_graph:
                checkpoint.world_state.locations.current_scene_id = new_scene
                logger.info(
                    "Scene transition: %s -> %s", old_scene, new_scene
                )
            else:
                logger.warning(
                    "NP1 suggested scene %s but it's not in the scene graph",
                    new_scene,
                )

        # Step 3: Discriminator
        t0 = time.monotonic()
        disc_output = await self.discriminator.run(event, checkpoint)
        latencies.append(PhaseLatency(
            phase="discriminator",
            duration_ms=(time.monotonic() - t0) * 1000,
            model=self.client.config.model_for_role("discriminator"),
        ))

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

        # Step 4: Agent fan-out (parallel)
        # Exclude the player character — their actions come from user input, not agents
        player_id = checkpoint.session.player_character_id
        responding = [
            o for o in disc_output.observers
            if o.should_respond and o.character_id != player_id
        ]
        agent_outputs: list[CharacterAgentOutput] = []

        # Short-circuit: unwitnessed failure
        # If infeasible and no NPCs should respond, return NP1's outcome directly.
        if not event.world_adjudication.feasible and not responding:
            logger.info(
                "Short-circuit: unwitnessed failure, returning NP1 outcome directly"
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

        if responding:
            t0 = time.monotonic()
            agent_tasks = []
            for obs in responding:
                char = self.char_mgr.get_character(checkpoint, obs.character_id)
                if char:
                    task = asyncio.wait_for(
                        self.agent_engine.respond(char, obs.facts, checkpoint),
                        timeout=AGENT_TIMEOUT,
                    )
                    agent_tasks.append(task)

            results = await asyncio.gather(*agent_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Agent failed: %s", result)
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
