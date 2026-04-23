"""Character manager — registry operations, roster updates, spawning.

Handles character lookup, state mutations after agent responses,
roster changes from discriminator output, and LLM-powered character genesis.
"""

from __future__ import annotations

import json
import logging

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    PublicSheet,
    PrivateState,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, SpawnRequest

logger = logging.getLogger(__name__)

# Max spawns per turn to prevent latency blowups
MAX_SPAWNS_PER_TURN = 3


def _push_spawn_state_change(
    checkpoint: CheckpointFile,
    char: CharacterRecord,
    router_summary: str,
) -> None:
    """Surface a freshly-spawned character to the next router call.

    Writes one line to `session.pending_router_state_changes`. The next
    router call's "## State Changes Since Your Last Call" block drains
    the queue and surfaces it once; subsequent calls see this character
    only via router history.

    Prefers the LLM-authored `router_summary` from the spawn-generation
    prompt (Commit 4) — the model has full context on who the spawn
    is and writes a tight identity-and-intent line in third-person
    ledger prose. Falls back to a mechanical name+role+location+
    objectives line ONLY if the summary is missing or whitespace-only,
    which should not happen with v3+ of `character_gen` but guards the
    pipeline if a future prompt regression drops the field.
    """
    if router_summary and router_summary.strip():
        checkpoint.session.pending_router_state_changes.append(
            f"Spawned: {char.name} (id: {char.character_id}) — "
            f"{router_summary.strip()}"
        )
        return

    role = char.public_sheet.role or "unknown role"
    loc = char.location or "unknown"
    scene_graph = checkpoint.world_state.locations.scene_graph
    loc_name = scene_graph.get(loc, {}).get("name", loc)
    objs = [o for o in (char.private_state.current_objectives or []) if o]
    parts = [
        f"Spawned: {char.name} (id: {char.character_id})",
        f"role={role}",
        f"location={loc_name} ({loc})",
    ]
    if objs:
        parts.append("objectives=" + "; ".join(objs))
    checkpoint.session.pending_router_state_changes.append(", ".join(parts))
    logger.warning(
        "Spawn for %s landed without router_summary; surfaced mechanical "
        "fallback line. Check character_gen prompt rendering.",
        char.character_id,
    )


def _pinned_character_ids(checkpoint: CheckpointFile) -> set[str]:
    """v11: ids currently holding a scene's active_act_slot OR listed as a
    required responder on an open Cat II event. Used to guard roster
    changes that would incoherently move/dormant/cull a character the
    engine is waiting on."""
    pinned: set[str] = set()
    for slot in checkpoint.session.active_act_slots.values():
        pinned.update(slot.keys())
    for evt in checkpoint.session.open_cat_ii_events:
        pinned.add(evt.initiator_id)
        pinned.update(evt.required_responders)
    return pinned


class CharacterManager:
    """Manages character registry and state updates."""

    def __init__(
        self,
        client: LLMClient | None = None,
        prompt_manager: PromptManager | None = None,
    ):
        self.client = client
        self.prompt_manager = prompt_manager

    def get_character(self, checkpoint: CheckpointFile, character_id: str) -> CharacterRecord | None:
        """Look up a character by ID."""
        for char in checkpoint.characters:
            if char.character_id == character_id:
                return char
        return None

    def apply_roster_updates(
        self, checkpoint: CheckpointFile, routed: EventRouterOutput,
    ) -> None:
        """Apply router-directed roster status changes (dormancy, culling).

        v11-A5: culled characters also have their v11 slot/event/buffer
        state purged, so a character removed mid-beat cannot strand a pin
        or leave their id inside an open Cat II event. Dormant characters
        are NOT purged — they may return, and their state is still valid.
        """
        from app.engine.turn_loop import purge_character_state

        # v11: a character who is currently pinned in a scene (initiator
        # or Cat II responder) cannot coherently be dormanted or culled
        # mid-beat — the fiction has them actively engaged. The router
        # should not produce this shape; if it does, we skip the status
        # change and log loudly so prompt drift is visible.
        pinned_ids = _pinned_character_ids(checkpoint)

        for char_id in routed.dormant:
            if char_id in pinned_ids:
                logger.warning(
                    "Ignored dormant on %s: character is currently pinned in "
                    "a scene's active_act_slot or as a Cat II responder. "
                    "The router should resolve the open event before "
                    "dormanting them.",
                    char_id,
                )
                continue
            char = self.get_character(checkpoint, char_id)
            if char:
                char.status = CharacterStatus.dormant
                logger.info("Character %s set to dormant", char_id)

        for char_id in routed.cull:
            # Cull is terminal — unlike dormant, the character is gone
            # for good. If they were pinned, `purge_character_state`
            # already handles the cleanup (abandons open Cat II events
            # they initiated; removes them from responder lists;
            # clears their buffer). Warn but proceed; the alternative
            # would leave the character both dead-in-fiction AND
            # perpetually pinned, which is worse.
            if char_id in pinned_ids:
                logger.warning(
                    "Culling %s mid-pin: their open Cat II event will be "
                    "abandoned / they'll be removed from any responder "
                    "list. The router should normally resolve the event "
                    "before culling.",
                    char_id,
                )
            char = self.get_character(checkpoint, char_id)
            if char:
                char.status = CharacterStatus.culled
                logger.info("Character %s culled", char_id)
            # Purge v11 bookkeeping even if the character record is
            # already missing — cull + purge must be idempotent.
            purge_character_state(checkpoint, char_id)

    async def spawn_characters(
        self,
        checkpoint: CheckpointFile,
        spawn_requests: list[SpawnRequest],
    ) -> list[CharacterRecord]:
        """Generate new characters from discriminator spawn requests via LLM.

        Returns the list of newly created and registered characters.
        """
        if not self.client or not self.prompt_manager:
            logger.warning("CharacterManager has no LLM client; skipping spawns")
            return []

        # Limit spawns per turn
        requests = spawn_requests[:MAX_SPAWNS_PER_TURN]
        if len(spawn_requests) > MAX_SPAWNS_PER_TURN:
            logger.warning(
                "Capping spawns from %d to %d", len(spawn_requests), MAX_SPAWNS_PER_TURN
            )

        # Skip already-existing characters
        requests = [
            r for r in requests
            if self.get_character(checkpoint, r.character_id) is None
        ]

        if not requests:
            return []

        spawned = []
        for req in requests:
            try:
                char, router_summary = await self._spawn_one(checkpoint, req)
                checkpoint.characters.append(char)
                spawned.append(char)
                logger.info("Spawned character: %s (%s)", char.name, char.character_id)
                _push_spawn_state_change(checkpoint, char, router_summary)
            except Exception as e:
                logger.warning("Failed to spawn %s: %s", req.character_id, e)

        return spawned

    async def _spawn_one(
        self, checkpoint: CheckpointFile, req: SpawnRequest
    ) -> tuple[CharacterRecord, str]:
        """Generate a single character via LLM.

        Returns the freshly-built CharacterRecord plus the LLM-authored
        `router_summary` (one or two sentences for the next router
        call's State Changes block). The summary is NOT persisted on
        the record — it's an author-time scratch field consumed by
        `_push_spawn_state_change` and otherwise discarded.
        """
        setting = checkpoint.world_state.setting
        setting_summary = f"Genre: {setting.genre}\nEra: {setting.era}\nTone: {setting.tone}\nPremise: {setting.premise}"
        world_lore = checkpoint.world_state.lore or "No detailed lore."
        physics = checkpoint.world_state.physics_ruleset
        world_rules = f"Strength limits: {physics.strength_limits}\nMagic: {'enabled' if physics.magic_enabled else 'disabled'}"

        locations = checkpoint.world_state.locations
        scene_id = req.seed.get("location", locations.current_scene_id)
        scene = locations.scene_graph.get(scene_id, {})
        scene_context = f"Location: {scene.get('name', scene_id)}\n{scene.get('description', '')}"

        seed_lines = []
        for k, v in req.seed.items():
            seed_lines.append(f"{k}: {v}")
        spawn_seed = "\n".join(seed_lines) if seed_lines else "No specific seed provided."

        existing = ", ".join(c.name for c in checkpoint.characters)

        messages = self.prompt_manager.render_messages(
            "character_gen",
            setting_summary=setting_summary,
            world_lore=world_lore,
            world_rules=world_rules,
            scene_context=scene_context,
            character_id=req.character_id,
            spawn_seed=spawn_seed,
            existing_characters=existing,
            location=scene_id,
        )

        from app.schemas.takeover import AuthoredCharacter
        response = await self.client.complete(
            role="agent",
            messages=messages,
            response_model=AuthoredCharacter,
            temperature=0.6,
            max_tokens=3000,
        )
        authored: AuthoredCharacter = response.parsed
        char = authored.to_record(character_id=req.character_id)
        # Enforce location from request
        char.location = scene_id

        return char, authored.router_summary
