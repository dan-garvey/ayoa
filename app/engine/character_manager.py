"""Character manager — registry operations, roster updates, spawning.

Handles character lookup, state mutations after agent responses,
roster changes from event-router output, and LLM-powered character genesis.
"""

from __future__ import annotations

import logging

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterStatus,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, SpawnRequest

logger = logging.getLogger(__name__)

# Max spawns per turn to prevent latency blowups
MAX_SPAWNS_PER_TURN = 3


# Per-line cap for LLM-authored player-character summaries after newline
# normalization. Chosen to fit a tight 1-2 sentence ledger line plus generous
# slack; entries longer than this are usually a sign the LLM regressed into
# multi-paragraph backstory.
ROUTER_SUMMARY_MAX_CHARS = 600

DND_GENERATION_INSTRUCTIONS = """
11. Because this session uses D&D 5e rules, also emit `dnd_statblock`. Build it
    from this character's fiction: role, likely competence, physicality, gear,
    threat level, and reason for appearing. Keep it modest unless the Spawn
    Request clearly calls for a dangerous creature or trained combatant. Do not
    quote or copy protected source text. Use an empty action list only for a
    helpless or purely civilian character; otherwise include at least one
    simple action they could take in combat.""".strip()

DND_OUTPUT_SCHEMA_SUFFIX = r""",
  "dnd_statblock": {
    "size": "string - Tiny, Small, Medium, Large, Huge, or Gargantuan",
    "creature_type": "string - humanoid, beast, construct, undead, etc.",
    "alignment": "string - use unaligned if irrelevant",
    "armor_class": 10,
    "hit_points": 4,
    "hit_dice": "string - compact formula such as 1d8 or 3d8+3",
    "speed": "string - e.g. 30 ft.",
    "ability_scores": {
      "strength": 10,
      "dexterity": 10,
      "constitution": 10,
      "intelligence": 10,
      "wisdom": 10,
      "charisma": 10
    },
    "proficiency_bonus": 2,
    "skills": [{"name": "string", "value": 2}],
    "senses": ["string"],
    "passive_perception": 10,
    "languages": ["string"],
    "challenge_rating": "string - empty string if not a meaningful combat threat",
    "xp": 0,
    "traits": [{"name": "string", "description": "string"}],
    "actions": [{
      "action_id": "string",
      "name": "string",
      "attack_bonus": 0,
      "reach_ft": 5,
      "range_normal_ft": 0,
      "range_long_ft": 0,
      "target": "string",
      "damage": "string",
      "damage_type": "string",
      "description": "string"
    }]
  }"""


def _normalize_router_summary(summary: str) -> str:
    """Sanitize an LLM-authored one-line character summary."""
    s = " ".join(summary.split()).strip()
    if len(s) > ROUTER_SUMMARY_MAX_CHARS:
        s = s[: ROUTER_SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return s


def _pinned_character_ids(checkpoint: CheckpointFile) -> set[str]:
    """Ids currently holding a beat slot or listed on an open Cat II event."""
    pinned: set[str] = set()
    pinned.update(checkpoint.session.active_act_slots.keys())
    for evt in checkpoint.session.open_cat_ii_events:
        pinned.add(evt.initiator_id)
        pinned.update(evt.required_responders)
    return pinned


def _dnd_ruleset_enabled(checkpoint: CheckpointFile) -> bool:
    settings = getattr(checkpoint.session.config, "settings", None)
    return str(getattr(settings, "ruleset_id", "") or "") == "dnd5e_basic"


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

        # v11: a character who is currently pinned in a beat (initiator
        # or Cat II responder) cannot coherently be dormanted or culled
        # mid-beat — the fiction has them actively engaged. The router
        # should not produce this shape; if it does, we skip the status
        # change and log loudly so prompt drift is visible.
        pinned_ids = _pinned_character_ids(checkpoint)

        for char_id in routed.dormant:
            if char_id in pinned_ids:
                logger.warning(
                    "Ignored dormant on %s: character is currently pinned in "
                    "the active_act_slot or as a Cat II responder. "
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
        *,
        acting_actor_location: str = "",
    ) -> list[CharacterRecord]:
        """Generate new characters from router spawn requests via LLM.

        `acting_actor_location` is the location label of whoever's action
        triggered these spawns — initiator location for Cat I, post-beat
        actor location for the in-beat path, or a background actor's location
        for private/background routed-agent spawns.
        It's the fallback when the router omits `seed.location`, so a new
        character materializes near the action rather than at some
        unrelated default.

        Returns the list of newly created and registered characters.

        Every requested spawn must materialize exactly once. If the router
        emits duplicate ids, exceeds the per-turn cap, targets an existing
        id, or character generation fails, the beat is invalid and should
        fail loudly instead of hiding the mismatch from the next router call.
        """
        if not self.client or not self.prompt_manager:
            raise RuntimeError(
                "Router requested character spawn, but CharacterManager has "
                "no LLM client/prompt manager."
            )

        seen_ids: set[str] = set()
        duplicate_ids: list[str] = []
        for r in spawn_requests:
            if r.character_id in seen_ids:
                duplicate_ids.append(r.character_id)
                continue
            seen_ids.add(r.character_id)
        if duplicate_ids:
            raise ValueError(
                "Router requested duplicate character spawns: "
                + ", ".join(sorted(set(duplicate_ids)))
            )

        if len(spawn_requests) > MAX_SPAWNS_PER_TURN:
            raise ValueError(
                "Router requested "
                f"{len(spawn_requests)} character spawns; max per turn is "
                f"{MAX_SPAWNS_PER_TURN}."
            )

        existing_ids = [
            r.character_id
            for r in spawn_requests
            if self.get_character(checkpoint, r.character_id) is not None
        ]
        if existing_ids:
            raise ValueError(
                "Router requested character spawns for existing ids: "
                + ", ".join(existing_ids)
            )

        if not spawn_requests:
            return []

        spawned = []
        for req in spawn_requests:
            char, _router_summary = await self._spawn_one(
                checkpoint, req, default_location=acting_actor_location,
            )
            checkpoint.characters.append(char)
            spawned.append(char)
            logger.info(
                "Spawned character: %s (%s)", char.name, char.character_id,
            )

        return spawned

    async def _spawn_one(
        self, checkpoint: CheckpointFile, req: SpawnRequest,
        *, default_location: str = "",
    ) -> tuple[CharacterRecord, str]:
        """Generate a single character via LLM.

        Returns the freshly-built CharacterRecord plus the LLM-authored
        `router_summary`. Router-authored spawns are already present in
        compact router history, so the summary is not queued into the next
        router input.

        Spawn-location resolution chain:
          1. router-supplied `req.seed.location`
          2. `default_location` — the acting actor's current location label
          3. the LLM's `authored.location` — last-resort, only hit when
             both router and orchestrator omit a location.
        """
        from app.engine.context_builder import (
            build_setting_summary,
            build_world_rules,
        )
        setting_summary = build_setting_summary(checkpoint)
        world_lore = checkpoint.world_state.lore or "No detailed lore."
        world_rules = build_world_rules(checkpoint)

        seed_loc = req.seed.location.strip()
        location = seed_loc or default_location
        if not location:
            logger.warning(
                "Spawning %s with no resolvable location (no seed.location, "
                "no default_location passed by caller). Will trust the "
                "character_gen LLM's authored location.",
                req.character_id,
            )
        location_context = (
            f"Location: {location}" if location else "Location: (none supplied)"
        )

        seed_lines = [
            f"role: {req.seed.role}",
            f"reason: {req.seed.reason}",
            f"location: {req.seed.location}",
            f"objectives: {', '.join(req.seed.objectives)}",
        ]
        spawn_seed = "\n".join(seed_lines) if seed_lines else "No specific seed provided."

        existing = ", ".join(c.name for c in checkpoint.characters)
        dnd_enabled = _dnd_ruleset_enabled(checkpoint)

        messages = self.prompt_manager.render_messages(
            "character_gen",
            setting_summary=setting_summary,
            world_lore=world_lore,
            world_rules=world_rules,
            location_context=location_context,
            character_id=req.character_id,
            spawn_seed=spawn_seed,
            existing_characters=existing,
            location=location,
            dnd_generation_instructions=(
                DND_GENERATION_INSTRUCTIONS if dnd_enabled else ""
            ),
            dnd_output_schema_suffix=(
                DND_OUTPUT_SCHEMA_SUFFIX if dnd_enabled else ""
            ),
        )

        from app.schemas.takeover import AuthoredCharacter
        response_model = AuthoredCharacter
        if dnd_enabled:
            from app.schemas.dnd_character_gen import AuthoredDndCharacter

            response_model = AuthoredDndCharacter
        response = await self.client.complete(
            role="agent_convenience",
            messages=messages,
            response_model=response_model,
            temperature=0.6,
            max_tokens=3000,
        )
        authored: AuthoredCharacter = response.parsed
        char = authored.to_record(character_id=req.character_id)
        char.agent_tier = CharacterAgentTier.utility
        if dnd_enabled:
            self._attach_dnd_spawn_mechanics(char, authored, req=req)
        # Override the LLM's authored.location only when the router or
        # caller supplied a concrete location label. When neither is set,
        # trust the LLM.
        if location:
            char.location = location
        elif not char.location:
            logger.warning(
                "Spawn %s has no location (router omitted, caller omitted, "
                "LLM emitted empty). Character will be unsited.",
                req.character_id,
            )

        # Seed the location signal so the freshly-spawned NPC's first
        # dispatch knows where they are.
        # Players never read pending_observations, so spawned-as-playable
        # characters are skipped (rare, but possible).
        if not char.is_playable and char.location:
            char.pending_observations.append(
                f"[your own action] {char.name} at {char.location}."
            )

        return char, authored.router_summary

    def _attach_dnd_spawn_mechanics(
        self,
        char: CharacterRecord,
        authored: object,
        *,
        req: SpawnRequest,
    ) -> None:
        from app.engine import dnd_combat, dnd_monsters

        statblock = getattr(authored, "dnd_statblock", None)
        if statblock is None:
            dnd_combat.ensure_default_combatant_mechanics(
                char,
                source="character_gen_missing_statblock",
            )
            return

        char.mechanics = dnd_monsters.mechanics_from_statblock(
            statblock,
            monster_key=req.character_id,
            source="character_gen_dnd_statblock",
        )
        char.mechanics["character_gen_dnd_statblock"] = {
            "source": "character_gen",
        }
