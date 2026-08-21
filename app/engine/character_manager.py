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
    is_player_authored_slot,
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
12. Because this session uses D&D 5e rules, also emit `dnd_statblock`. Build it
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

ONE_STAR_OUTPUT_SCHEMA_SUFFIX = r""",
  "one_star_hero": {
    "level": 1,
    "experience_points": 0,
    "hp_current": 1,
    "hp_max": 1,
    "stats": {"string_stat_id": 0},
    "equipment": [{
      "item_id": "snake_case stable id",
      "name": "string",
      "slot": "string",
      "quantity": 1,
      "durability_current": 0,
      "durability_max": 0,
      "tags": ["string"],
      "visible": true
    }],
    "skills": [{
      "skill_id": "snake_case stable id",
      "name": "string",
      "rank": 1,
      "capability": "short concrete capability",
      "tags": ["string"],
      "visible": true
    }],
    "conditions": [],
    "persistent_injuries": [],
    "innate_system_sight": false,
    "hidden_capabilities": {},
    "private_potential": "string - private casting note; empty if none"
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


def _one_star_ruleset_enabled(checkpoint: CheckpointFile) -> bool:
    from app.schemas.one_star import ONE_STAR_RULESET_ID

    settings = getattr(checkpoint.session.config, "settings", None)
    return str(getattr(settings, "ruleset_id", "") or "") == ONE_STAR_RULESET_ID


def _assemble_knowledge_grant(
    checkpoint: CheckpointFile, tier: int,
) -> tuple[str, CharacterAgentTier | None]:
    """Build the authored character_gen contract for a spawn tier.

    Returns (grant_block, agent_tier). The block covers tiers 1..tier from
    `world_state.knowledge_tiers` (each rung's personal depth + unlocked
    world/plot knowledge), plus the exact target rung's optional non-cumulative
    generation guidance; agent_tier is the highest present rung's override, if
    any. Tier zero means no ladder knowledge when a story authored a ladder.
    Returns ("", None) only when the story has no ladder, so ordinary stories
    retain unrestricted generation.
    """
    tiers = list(getattr(checkpoint.world_state, "knowledge_tiers", None) or [])
    if not tiers:
        return "", None
    selected = sorted(
        (t for t in tiers if 1 <= t.tier <= tier), key=lambda t: t.tier,
    )
    lines = [
        "## Knowledge Budget (authoritative)",
    ]
    if selected:
        lines.append(
            f"This character is knowledge tier {tier}. Author their backstory, "
            "known_context, and secrets to EXACTLY this budget and no further: "
            "they know what follows and nothing beyond it. Keep spoiler-bearing "
            "world/plot knowledge in known_context/secrets, never in "
            "player-safe appearance or default_loadout."
        )
    else:
        lines.append(
            f"This character is knowledge tier {tier}, below the first authored "
            "rung. Give them no world or plot knowledge from the ladder. Build "
            "only from the immediate Entry Context and explicit Spawn Request; "
            "do not infer adjacent setting facts."
        )
    for t in selected:
        head = f"Tier {t.tier}" + (f" ({t.label})" if t.label else "")
        lines.append(f"\n{head}:")
        if t.personal_depth:
            lines.append(f"- Personal life they remember: {t.personal_depth}")
        if t.world_knowledge:
            lines.append(f"- World/plot they are aware of: {t.world_knowledge}")

    target = next((t for t in selected if t.tier == tier), None)
    guidance = (
        getattr(target, "generation_guidance", None)
        if target is not None
        else None
    )
    guidance_fields = (
        (
            ("Backstory depth", guidance.backstory_depth),
            ("Personality, voice, and contradiction depth", guidance.personality_depth),
            ("Public physical/visual detail", guidance.public_visual_detail),
            ("Loadout complexity and material finish", guidance.loadout_detail),
            ("Intended visual salience", guidance.visual_salience),
            ("Presentation guidance", guidance.presentation_guidance),
        )
        if guidance is not None
        else ()
    )
    if any(value.strip() for _label, value in guidance_fields):
        lines.extend(
            [
                "\n## Authored Generation Budget (authoritative)",
                (
                    f"Apply the Tier {tier} target below to the whole character. "
                    "Unlike the cumulative knowledge boundary, this is one "
                    "target budget: do not average it with lower tiers. Match "
                    "sparse/shared direction by withholding extra polish, and "
                    "match rich direction even when it overrides the prompt's "
                    "ordinary concision defaults. Keep every public visual "
                    "detail player-safe. Presentation guidance is story-local "
                    "casting and art direction, not a universal value judgment."
                ),
            ]
        )
        lines.extend(
            f"- {label}: {value}"
            for label, value in guidance_fields
            if value.strip()
        )

    agent_tier: CharacterAgentTier | None = None
    for t in selected:
        if t.agent_tier is not None:
            agent_tier = t.agent_tier
    return "\n".join(lines), agent_tier


def _generation_setting_summary(
    checkpoint: CheckpointFile,
    *,
    knowledge_isolated: bool,
) -> str:
    """Build style context without leaking a tiered story's premise.

    Genre, era, and tone establish casting and presentation without asserting
    that the generated character knows the premise. Untiered stories retain
    the established full setting summary, including premise.
    """
    from app.engine.context_builder import build_setting_summary

    if not knowledge_isolated:
        return build_setting_summary(checkpoint)

    setting = checkpoint.world_state.setting
    parts = [
        f"{label}: {value}"
        for label, value in (
            ("Genre", setting.genre),
            ("Era", setting.era),
            ("Tone", setting.tone),
        )
        if value
    ]
    return "\n".join(parts) if parts else "No style information available."


def _generation_world_context(
    checkpoint: CheckpointFile,
    *,
    knowledge_grant: str,
    world_lore: str,
    world_rules: str,
) -> str:
    """Project either full untiered lore or an exact authored tier budget."""
    if not checkpoint.world_state.knowledge_tiers:
        return (
            "## World Lore\n"
            f"{world_lore}\n\n"
            "## Physical Rules\n"
            f"{world_rules}"
        )

    magic = checkpoint.world_state.physics_ruleset.magic_enabled
    return (
        f"{knowledge_grant}\n\n"
        "## Physical Boundaries\n"
        f"- Magic exists in this story: {'yes' if magic else 'no'}. This does "
        "not itself grant knowledge of how magic works.\n"
        "- Give the character no ability, equipment, or physical capability "
        "beyond the Entry Context, Spawn Request, and Knowledge Budget."
    )


def _existing_character_generation_lines(
    characters: list[CharacterRecord],
    *,
    bound_character_ids: set[str] | None = None,
    local_active_location: str | None = None,
) -> str:
    lines: list[str] = []
    bound = bound_character_ids or set()
    for character in characters:
        if local_active_location is not None and (
            character.status != CharacterStatus.active
            or character.location != local_active_location
        ):
            continue
        if (
            is_player_authored_slot(character)
            and character.character_id not in bound
        ):
            continue
        parts = [
            character.name or character.character_id,
            f"id={character.character_id}",
        ]
        role = " ".join(character.public_sheet.role.split())
        appearance = " ".join(character.public_sheet.appearance.split())
        loadout = " ".join(character.visuals.default_loadout.split())
        if role:
            parts.append(f"role={role[:180]}")
        if appearance:
            parts.append(f"appearance={appearance[:240]}")
        if loadout:
            parts.append(f"loadout={loadout[:240]}")
        lines.append("- " + "; ".join(parts))
    return "\n".join(lines) if lines else "No existing characters."


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

        # Wake dormant characters back into play (inverse of dormant): a
        # benched Hero or a reserved off-stage persona the router is bringing
        # on stage. Flip to active and place them where they re-enter.
        for signal in getattr(routed, "activate", None) or []:
            char = self.get_character(checkpoint, signal.character_id)
            if char is None:
                logger.warning(
                    "Ignored activate on %s: no such character.",
                    signal.character_id,
                )
                continue
            if char.status == CharacterStatus.culled:
                logger.warning(
                    "Ignored activate on %s: character is culled; the dead do "
                    "not wake.", signal.character_id,
                )
                continue
            char.status = CharacterStatus.active
            if signal.location_label:
                char.location = signal.location_label
            logger.info(
                "Character %s woken to active at %s",
                signal.character_id, char.location,
            )

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
                if char.status == CharacterStatus.culled:
                    logger.warning(
                        "Ignored dormant on %s: character is culled; terminal "
                        "lifecycle cannot be downgraded.",
                        char_id,
                    )
                    continue
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
        one_star_hero_ids: set[str] | None = None,
    ) -> list[CharacterRecord]:
        """Generate new characters from router spawn requests via LLM.

        `acting_actor_location` is the location label of whoever's action
        triggered these spawns — initiator location for Cat I, post-beat
        actor location for the in-beat path, or a background actor's location
        for private/background routed-agent spawns.
        It's the fallback when the router omits `seed.location`, so a new
        character materializes near the action rather than at some
        unrelated default.

        `one_star_hero_ids` is the exact subset paired with a typed summon in
        the source event. Other spawns in that ruleset remain ordinary
        persistent characters and receive no Hero sheet.

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

        hero_spawn_ids = set(one_star_hero_ids or ())
        request_ids = {request.character_id for request in spawn_requests}
        unknown_hero_ids = hero_spawn_ids - request_ids
        if unknown_hero_ids:
            raise ValueError(
                "One-Star Hero generation targets absent spawn ids: "
                + ", ".join(sorted(unknown_hero_ids))
            )
        if hero_spawn_ids and not _one_star_ruleset_enabled(checkpoint):
            raise ValueError(
                "One-Star Hero generation was requested outside its ruleset"
            )
        if _one_star_ruleset_enabled(checkpoint):
            from app.engine.one_star_adapter import load_one_star_account

            _owner, account = load_one_star_account(checkpoint)
            ordinary_spawn_count = len(spawn_requests) - len(hero_spawn_ids)
            if len(hero_spawn_ids) > account.config.max_summon_batch:
                raise ValueError(
                    "Router requested "
                    f"{len(hero_spawn_ids)} One-Star Hero spawns; configured "
                    f"summon max is {account.config.max_summon_batch}."
                )
            if ordinary_spawn_count > MAX_SPAWNS_PER_TURN:
                raise ValueError(
                    "Router requested "
                    f"{ordinary_spawn_count} ordinary character spawns; max "
                    f"per turn is {MAX_SPAWNS_PER_TURN}."
                )
        elif len(spawn_requests) > MAX_SPAWNS_PER_TURN:
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
                checkpoint,
                req,
                default_location=acting_actor_location,
                one_star_hero=req.character_id in hero_spawn_ids,
            )
            checkpoint.characters.append(char)
            spawned.append(char)
            logger.info(
                "Spawned character: %s (%s)", char.name, char.character_id,
            )

        return spawned

    async def _spawn_one(
        self, checkpoint: CheckpointFile, req: SpawnRequest,
        *,
        default_location: str = "",
        one_star_hero: bool = False,
    ) -> tuple[CharacterRecord, str]:
        """Generate a single character via LLM.

        Returns the freshly-built CharacterRecord plus the LLM-authored
        `router_summary`. Router-authored spawns are already present in compact
        router history, so the summary is not queued into the next router
        input. The materialized display name is retained separately on that
        compact spawn record because it did not exist when the router authored
        the request.

        Spawn-location resolution chain:
          1. router-supplied `req.seed.location`
          2. `default_location` — the acting actor's current location label
          3. the LLM's `authored.location` — last-resort, only hit when
             both router and orchestrator omit a location.
        """
        from app.engine.context_builder import build_world_rules

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

        knowledge_grant, tier_agent_tier = _assemble_knowledge_grant(
            checkpoint, req.seed.knowledge_tier,
        )
        knowledge_isolated = bool(checkpoint.world_state.knowledge_tiers)
        setting_summary = _generation_setting_summary(
            checkpoint,
            knowledge_isolated=knowledge_isolated,
        )
        generation_context = _generation_world_context(
            checkpoint,
            knowledge_grant=knowledge_grant,
            world_lore=world_lore,
            world_rules=world_rules,
        )

        existing = _existing_character_generation_lines(
            checkpoint.characters,
            bound_character_ids=set(
                checkpoint.session.character_bindings or {}
            ),
            # A tier-gated generation call only needs visible local cast and
            # earlier members of the same wave. Dormant reserves and remote
            # plot figures are not character knowledge or necessary casting
            # context for this arrival.
            local_active_location=(location if knowledge_isolated else None),
        )
        dnd_enabled = _dnd_ruleset_enabled(checkpoint)
        one_star_enabled = _one_star_ruleset_enabled(checkpoint)
        if one_star_hero and not one_star_enabled:
            raise ValueError(
                "One-Star Hero generation was requested outside its ruleset"
            )
        if dnd_enabled and one_star_enabled:  # Defensive: ruleset_id is scalar.
            raise RuntimeError("multiple character-generation rulesets are active")

        ruleset_generation_instructions = ""
        ruleset_output_schema_suffix = ""
        ruleset_generation_context = ""
        if dnd_enabled:
            ruleset_generation_instructions = DND_GENERATION_INSTRUCTIONS
            ruleset_output_schema_suffix = DND_OUTPUT_SCHEMA_SUFFIX
        elif one_star_hero:
            if req.seed.knowledge_tier < 1:
                raise ValueError(
                    "One-Star generated Heroes require an authored birth-star "
                    "value in SpawnRequest.seed.knowledge_tier"
                )
            ruleset_generation_instructions = self.prompt_manager.render(
                "character_gen_ruleset_one_star",
            ).strip()
            ruleset_output_schema_suffix = ONE_STAR_OUTPUT_SCHEMA_SUFFIX
            ruleset_generation_context = (
                "## One-Star Mechanics Authority\n"
                f"Exact birth stars: {req.seed.knowledge_tier}. The structured "
                "starting sheet must express this character at that birth "
                "grade without teaching the character any premise facts."
            )

        messages = self.prompt_manager.render_messages(
            "character_gen",
            setting_summary=setting_summary,
            generation_context=generation_context,
            location_context=location_context,
            character_id=req.character_id,
            spawn_seed=spawn_seed,
            existing_characters=existing,
            location=location,
            ruleset_generation_instructions=ruleset_generation_instructions,
            ruleset_output_schema_suffix=ruleset_output_schema_suffix,
            ruleset_generation_context=ruleset_generation_context,
        )

        from app.schemas.takeover import AuthoredCharacter
        response_model = AuthoredCharacter
        if dnd_enabled:
            from app.schemas.dnd_character_gen import AuthoredDndCharacter

            response_model = AuthoredDndCharacter
        elif one_star_hero:
            from app.schemas.one_star_character_gen import (
                AuthoredOneStarCharacter,
            )

            response_model = AuthoredOneStarCharacter
        response = await self.client.complete(
            role="agent_convenience",
            messages=messages,
            response_model=response_model,
            temperature=0.6,
            max_tokens=3000,
        )
        authored: AuthoredCharacter = response.parsed
        char = authored.to_record(character_id=req.character_id)
        char.agent_tier = tier_agent_tier or CharacterAgentTier.utility
        char.knowledge_tier = req.seed.knowledge_tier
        if dnd_enabled:
            self._attach_dnd_spawn_mechanics(char, authored, req=req)
        elif one_star_hero:
            self._attach_one_star_spawn_mechanics(
                checkpoint,
                char,
                authored,
                birth_stars=req.seed.knowledge_tier,
            )
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

    @staticmethod
    def _attach_one_star_spawn_mechanics(
        checkpoint: CheckpointFile,
        char: CharacterRecord,
        authored: object,
        *,
        birth_stars: int,
    ) -> None:
        from app.engine.one_star_adapter import (
            load_one_star_account,
            validate_one_star_hero_state,
        )
        from app.schemas.one_star import ONE_STAR_HERO_KEY

        generated = getattr(authored, "one_star_hero", None)
        if generated is None:
            raise ValueError("One-Star character generation omitted Hero mechanics")
        hero = generated.to_hero_state(birth_stars=birth_stars)
        _owner, account = load_one_star_account(checkpoint)
        validate_one_star_hero_state(hero, account.config)
        char.mechanics = dict(char.mechanics)
        char.mechanics[ONE_STAR_HERO_KEY] = hero.model_dump(mode="json")

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
