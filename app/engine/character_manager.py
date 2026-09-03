"""Character manager — registry operations, roster updates, spawning.

Handles character lookup, state mutations after agent responses,
roster changes from event-router output, and LLM-powered character genesis.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, ConfigDict

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterStatus,
    is_player_authored_slot,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.event_router import EventRouterOutput, SpawnRequest

logger = logging.getLogger(__name__)

# Max spawns per turn to prevent latency blowups
MAX_SPAWNS_PER_TURN = 3

# Reasoning tokens and visible structured output share the same response
# ceiling. Character authoring and its sibling casting plan are one-time calls,
# and both need enough headroom for a max-reasoning model before emitting JSON.
# This is a ceiling, not a requested spend; completed responses stop earlier.
CHARACTER_MANAGER_MAX_TOKENS = 32_000
CASTING_PLAN_MAX_TOKENS = CHARACTER_MANAGER_MAX_TOKENS


# Per-line cap for LLM-authored player-character summaries after newline
# normalization. Chosen to fit a tight 1-2 sentence ledger line plus generous
# slack; entries longer than this are usually a sign the LLM regressed into
# multi-paragraph summary.
ROUTER_SUMMARY_MAX_CHARS = 600

# Character generation sees only another record's public identity anchors.
# Actor facts are private and must not become sibling casting material.
EXISTING_CHARACTER_PUBLIC_CONTEXT_MAX_CHARS = 240

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
    "strong_stat_id": "one configured stat id",
    "weak_stat_id": "a different configured stat id",
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
    "hidden_capabilities": [{
      "capability_id": "snake_case stable id",
      "description": "short private qualitative capability"
    }]
  }"""


class CastingBrief(BaseModel):
    """One compact, public-facing brief for a requested arrival."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    brief: str


class CastingPlan(BaseModel):
    """The complete cast plan returned before individual authoring calls."""

    model_config = ConfigDict(extra="forbid")

    briefs: list[CastingBrief]


def _immutable_checkpoint_copy(checkpoint: CheckpointFile) -> CheckpointFile:
    """Copy durable state without following process-local runtime handles."""

    return CheckpointFile.model_validate_json(
        checkpoint.model_dump_json(
            context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
        )
    )


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
    world/plot knowledge), plus the exact rung's optional non-cumulative
    authoring guidance; agent_tier is the highest present rung's override, if
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
        "## Knowledge Boundary (authoritative)",
    ]
    if selected:
        lines.append(
            f"This character is knowledge tier {tier}. Author only actor facts "
            "that this boundary supports: they know what follows and nothing "
            "beyond it. Keep spoiler-bearing world or plot material in private "
            "actor facts, never in player-safe appearance, public_context, or "
            "default_loadout."
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
            ("Actor-fact guidance", guidance.actor_fact_guidance),
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
                "\n## Tier Authoring Guidance (authoritative)",
                (
                    f"Use only the Tier {tier} guidance below; do not combine it "
                    "with another rung. It informs which concrete actor facts and "
                    "public details are warranted, but never requires a fact or "
                    "turns facts into categories. An empty or uneven fact list can "
                    "be correct. Keep every public visual detail player-safe. "
                    "Presentation guidance is story-local casting and art direction, "
                    "not a universal value judgment."
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
        "beyond the Entry Context, Spawn Request, and Knowledge Boundary."
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
        public_context = " ".join(character.public_sheet.public_context.split())
        loadout = " ".join(character.visuals.default_loadout.split())
        if role:
            parts.append(f"role={role[:180]}")
        if appearance:
            parts.append(f"appearance={appearance[:240]}")
        if public_context:
            parts.append(
                "public_context="
                f"{public_context[:EXISTING_CHARACTER_PUBLIC_CONTEXT_MAX_CHARS]}"
            )
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
            if char.status == CharacterStatus.active:
                # Activation is a dormant -> active lifecycle edge, not a
                # movement command. Replaying a previously applied event after
                # later canonical movement must therefore be a no-op; active
                # movement belongs exclusively in location_updates.
                logger.info(
                    "Ignored activate replay on %s: character is already active.",
                    signal.character_id,
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

    @staticmethod
    def _casting_plan_requests(
        spawn_requests: list[SpawnRequest],
    ) -> str:
        lines: list[str] = []
        for request in spawn_requests:
            seed = request.seed
            objectives = ", ".join(seed.objectives) or "none"
            lines.append(
                f"- id={request.character_id}; role={seed.role}; "
                f"reason={seed.reason}; location={seed.location}; "
                f"objectives={objectives}; knowledge_tier={seed.knowledge_tier}"
            )
        return "\n".join(lines)

    @staticmethod
    def _casting_plan_existing_characters(
        checkpoint: CheckpointFile,
    ) -> str:
        # A casting plan is shared with every generation branch. Keep it to
        # public identity anchors so a brief cannot smuggle actor facts into
        # another character's authoring call.
        return _existing_character_generation_lines(
            checkpoint.characters,
            bound_character_ids=set(
                checkpoint.session.character_bindings or {}
            ),
        )

    async def _make_casting_plan(
        self,
        checkpoint: CheckpointFile,
        spawn_requests: list[SpawnRequest],
    ) -> list[CastingBrief]:
        """Author one compact sibling-aware plan before character genesis."""
        if self.client is None or self.prompt_manager is None:
            raise RuntimeError("Character generation requires an LLM client")

        messages = self.prompt_manager.render_messages(
            "character_casting_plan",
            setting_summary=_generation_setting_summary(
                checkpoint,
                knowledge_isolated=bool(checkpoint.world_state.knowledge_tiers),
            ),
            spawn_requests=self._casting_plan_requests(spawn_requests),
            existing_characters=self._casting_plan_existing_characters(
                checkpoint
            ),
        )
        response = await self.client.complete(
            role="character_manager",
            messages=messages,
            response_model=CastingPlan,
            temperature=0.5,
            max_tokens=CASTING_PLAN_MAX_TOKENS,
            cache=True,
            compact=True,
        )
        parsed = response.parsed
        if parsed is None:
            try:
                parsed = CastingPlan.model_validate_json(response.content)
            except Exception as exc:
                raise ValueError(
                    "Character casting plan did not contain valid structured "
                    "output"
                ) from exc
        try:
            plan = (
                parsed
                if isinstance(parsed, CastingPlan)
                else CastingPlan.model_validate(parsed)
            )
        except Exception as exc:
            raise ValueError("Character casting plan is invalid") from exc

        request_ids = [request.character_id for request in spawn_requests]
        briefs_by_id: dict[str, CastingBrief] = {}
        for brief in plan.briefs:
            character_id = brief.character_id.strip()
            text = brief.brief.strip()
            if not character_id or not text:
                raise ValueError(
                    "Character casting plan entries require a non-empty id "
                    "and brief"
                )
            if character_id in briefs_by_id:
                raise ValueError(
                    "Character casting plan contains duplicate character ids: "
                    f"{character_id}"
                )
            briefs_by_id[character_id] = CastingBrief(
                character_id=character_id,
                brief=text,
            )
        if set(briefs_by_id) != set(request_ids) or len(briefs_by_id) != len(
            request_ids
        ):
            raise ValueError(
                "Character casting plan must contain exactly one brief for "
                "each requested character"
            )
        return [briefs_by_id[character_id] for character_id in request_ids]

    @staticmethod
    def _casting_briefs_block(briefs: list[CastingBrief]) -> str:
        if not briefs:
            return ""
        lines = [
            "## Sibling Casting Briefs",
            "Treat these as shared public casting direction for this arrival "
            "wave. Preserve each character's own request and knowledge tier; "
            "do not copy another brief's private facts.",
        ]
        lines.extend(
            f"- {brief.character_id}: {brief.brief}" for brief in briefs
        )
        return "\n".join(lines)

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

        casting_briefs = (
            await self._make_casting_plan(checkpoint, spawn_requests)
            if len(spawn_requests) > 1
            else []
        )
        briefs_block = self._casting_briefs_block(casting_briefs)

        # Every branch starts from the same immutable roster/context snapshot.
        # In particular, a fast first response must not become "existing
        # character" context for a slower sibling. Nothing is accepted into
        # the live roster until every branch has returned successfully.
        generation_snapshot = _immutable_checkpoint_copy(checkpoint)

        async def generate_one(
            request: SpawnRequest,
        ) -> tuple[CharacterRecord, str]:
            branch = _immutable_checkpoint_copy(generation_snapshot)
            return await self._spawn_one(
                branch,
                request,
                default_location=acting_actor_location,
                one_star_hero=request.character_id in hero_spawn_ids,
                casting_plan_block=briefs_block,
            )

        results = await asyncio.gather(
            *(generate_one(request) for request in spawn_requests),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            # Branches never touched the live checkpoint. Re-raise the first
            # failure after all siblings finish so no background generation is
            # left running against a supposedly accepted wave.
            raise failures[0]

        spawned = [result[0] for result in results]
        for char in spawned:
            checkpoint.characters.append(char)
            logger.info(
                "Spawned character: %s (%s)", char.name, char.character_id,
            )

        return spawned

    async def _spawn_one(
        self, checkpoint: CheckpointFile, req: SpawnRequest,
        *,
        default_location: str = "",
        one_star_hero: bool = False,
        casting_plan_block: str = "",
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
        if casting_plan_block:
            generation_context = (
                f"{generation_context}\n\n{casting_plan_block}"
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
            from app.engine.one_star_adapter import load_one_star_account

            _owner, account = load_one_star_account(checkpoint)
            stat_ids = ", ".join(account.config.progression.stat_ids)
            ruleset_generation_context = (
                "## One-Star Qualitative Affinities\n"
                f"Exact birth stars: {req.seed.knowledge_tier}. This is "
                "qualitative casting authority only. Choose one strong and one "
                "different weak affinity from these configured stat ids: "
                f"{stat_ids}. Do not emit numerical mechanics or reveal private "
                "casting information."
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
            role="character_manager",
            messages=messages,
            response_model=response_model,
            temperature=0.6,
            max_tokens=CHARACTER_MANAGER_MAX_TOKENS,
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
        from app.engine.one_star_adapter import load_one_star_account
        from app.engine.one_star_progression import build_generated_hero
        from app.schemas.one_star import ONE_STAR_HERO_KEY

        generated = getattr(authored, "one_star_hero", None)
        if generated is None:
            raise ValueError("One-Star character generation omitted Hero mechanics")
        _owner, account = load_one_star_account(checkpoint)
        hero = build_generated_hero(
            character_id=char.character_id,
            generated=generated,
            birth_stars=birth_stars,
            config=account.config,
        )
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
