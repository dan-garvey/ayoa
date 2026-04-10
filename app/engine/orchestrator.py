"""Turn orchestrator — wires the full turn pipeline.

Sequence: Load checkpoint -> NP1 -> Discriminator -> Agents (parallel) -> NP2 -> Save checkpoint.
"""

from __future__ import annotations

import asyncio
import logging

from app.engine.character_agent import CharacterAgent
from app.engine.character_manager import CharacterManager
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.discriminator import Discriminator
from app.engine.narrator import Narrator
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.agents import CharacterAgentOutput
from app.schemas.checkpoint import CheckpointFile
from app.schemas.requests import TurnRequest
from app.schemas.responses import DebugPayload, TurnResponse

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
        # Step 1: Load checkpoint
        checkpoint = self.checkpoint_mgr.load_latest(request.session_id)
        logger.info(
            "Turn %d for session %s",
            checkpoint.session.turn_index,
            request.session_id,
        )

        # Step 2: NP1 Adjudicate
        event = await self.narrator.phase_1(request.user_input, checkpoint)

        # Step 3: Discriminator
        disc_output = await self.discriminator.run(event, checkpoint)

        # Apply roster updates (dormancy, culling)
        self.char_mgr.apply_roster_updates(checkpoint, disc_output)

        # Step 3.5: Spawn new characters if requested
        if disc_output.spawn:
            await self.char_mgr.spawn_characters(checkpoint, disc_output.spawn)

        # Step 4: Agent fan-out (parallel)
        responding = [o for o in disc_output.observers if o.should_respond]
        agent_outputs: list[CharacterAgentOutput] = []

        if responding:
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

        # Step 5: NP2 Compose
        final = await self.narrator.phase_2(
            request.user_input, event, agent_outputs, checkpoint
        )

        # Step 6: Apply state updates
        for output in agent_outputs:
            self.char_mgr.apply_agent_output(checkpoint, output)

        # Update transcript
        checkpoint.transcript.append(final.transcript_entry)

        # Advance turn
        checkpoint.session.turn_index += 1

        # Step 7: Save checkpoint
        self.checkpoint_mgr.save(checkpoint)
        checkpoint_id = f"ckpt_{checkpoint.session.turn_index:04d}"

        logger.info(
            "Turn %d complete: %d chars output, %d agents responded",
            checkpoint.session.turn_index,
            len(final.final_text),
            len(agent_outputs),
        )

        # Step 8: Build response
        debug = None
        if request.debug:
            debug = DebugPayload(
                canonical_event=event.model_dump(),
                discriminator=disc_output.model_dump(),
                agent_outputs=[o.model_dump() for o in agent_outputs],
                world_updates=final.world_updates,
            )

        return TurnResponse(
            session_id=request.session_id,
            checkpoint_id=checkpoint_id,
            turn_index=checkpoint.session.turn_index,
            output_text=final.final_text,
            debug=debug,
        )
