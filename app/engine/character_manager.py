"""Character manager — registry operations, roster updates, spawning.

Handles character lookup, state mutations after agent responses,
roster changes from discriminator output, and LLM-powered character genesis.
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


# Per-line cap for state-change entries, applied to the LLM-authored
# `router_summary` after newline normalization. Chosen to fit a tight
# 1-2 sentence ledger line (the prompt says one or two sentences) plus
# generous slack; entries longer than this are usually a sign the LLM
# regressed into multi-paragraph backstory and would balloon the next
# router prompt.
ROUTER_SUMMARY_MAX_CHARS = 600


def _normalize_router_summary(summary: str) -> str:
    """Sanitize an LLM-authored `router_summary` for safe interpolation
    into `_build_state_changes_block`'s bullet list.

    The block renders each queued entry as a single markdown bullet
    (`- {entry}`), so an entry containing a literal newline would split
    the bullet across multiple lines and shatter the list for the
    router prompt. Collapse all whitespace runs to single spaces, and
    cap at `ROUTER_SUMMARY_MAX_CHARS` to defend against a regressed
    spawn LLM dumping a backstory paragraph into the field.
    """
    s = " ".join(summary.split()).strip()
    if len(s) > ROUTER_SUMMARY_MAX_CHARS:
        s = s[: ROUTER_SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return s


def _drop_pending_spawn_line(
    checkpoint: CheckpointFile, character_id: str,
) -> bool:
    """Strip any queued `pending_router_state_changes` line that
    references `character_id` as a freshly-spawned character.

    Called when a character is culled before the queue drains. Without
    this, the next router call would see "Spawned: X — ..."
    for a character who no longer exists on the roster — a ghost
    instruction that confuses adjudication and may prompt the router
    to re-spawn or hallucinate them.

    Matches both the current id-only form and the older name/id form so
    mixed live checkpoints remain cleanup-safe. Returns True if any line
    was removed.
    """
    queue = checkpoint.session.pending_router_state_changes or []
    new_needle = f"Spawned: {character_id}"
    old_needle = f"(id: {character_id})"
    kept = [
        line for line in queue
        if new_needle not in line and old_needle not in line
    ]
    if len(kept) == len(queue):
        return False
    removed = len(queue) - len(kept)
    checkpoint.session.pending_router_state_changes = kept
    logger.info(
        "Dropped %d pending state-change line(s) for culled character %s "
        "before they could surface to the router.",
        removed, character_id,
    )
    return True


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
    ledger prose. The summary is normalized (whitespace collapsed,
    over-long entries truncated) before interpolation. Falls back to
    a mechanical name+role+location+objectives line ONLY if the
    summary is missing or whitespace-only, which should not happen
    with v3+ of `character_gen` but guards the pipeline if a future
    prompt regression drops the field.
    """
    cleaned = _normalize_router_summary(router_summary or "")
    if cleaned:
        checkpoint.session.pending_router_state_changes.append(
            f"Spawned: {char.character_id} — {cleaned}"
        )
        return

    role = char.public_sheet.role or "unknown role"
    loc = char.location or "unknown"
    objs = [o for o in (char.private_state.current_objectives or []) if o]
    parts = [
        f"Spawned: {char.character_id}",
        f"role={role}",
        f"location={loc}",
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
    """Ids currently holding a beat slot or listed on an open Cat II event."""
    pinned: set[str] = set()
    pinned.update(checkpoint.session.active_act_slots.keys())
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
            # Commit-4b: also strip any not-yet-drained "Spawned: ..."
            # state-change line for this id, so the next router call
            # doesn't see a ghost spawn for a character who was killed
            # off in the same beat. Safe no-op if no line is queued.
            _drop_pending_spawn_line(checkpoint, char_id)

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
        actor location for the in-beat path, the off-stage actor's location
        for tick-driven spawns.
        It's the fallback when the router omits `seed.location`, so a new
        character materializes near the action rather than at some
        unrelated default.

        Returns the list of newly created and registered characters.

        v11-r7i: dedups within the batch (the router can emit two
        SpawnRequests with the same character_id when a single beat
        implies multiple instances of the same role — a v11 playtest
        spawned three `production_runner`s in one turn). Without
        dedup, the second-and-later entries overwrite the first via
        `checkpoint.characters.append` of the same id, leaving the
        roster with two records sharing one id and the rest of the
        engine confused about which one the canonical event refers to.
        """
        if not self.client or not self.prompt_manager:
            logger.warning("CharacterManager has no LLM client; skipping spawns")
            return []

        # Order is load-bearing: dedup BEFORE the per-turn cap.
        # Otherwise a router output of `[runner, runner, runner,
        # unique_villain]` would slice the first three (all dups),
        # collapse to a single `runner`, and silently drop
        # `unique_villain` despite us having only spawned one
        # character. By deduping first we count each distinct id once,
        # then cap on distinct ids — the unique villain survives, the
        # extra runners get logged-and-dropped.
        seen_ids: set[str] = set()
        deduped: list[SpawnRequest] = []
        for r in spawn_requests:
            if r.character_id in seen_ids:
                logger.warning(
                    "Dropped duplicate spawn for %s within the same router "
                    "batch — the router emitted multiple SpawnRequests with "
                    "this id; keeping the first.",
                    r.character_id,
                )
                continue
            seen_ids.add(r.character_id)
            deduped.append(r)

        requests = deduped[:MAX_SPAWNS_PER_TURN]
        if len(deduped) > MAX_SPAWNS_PER_TURN:
            logger.warning(
                "Capping distinct spawns from %d to %d (dropped: %s)",
                len(deduped), MAX_SPAWNS_PER_TURN,
                [r.character_id for r in deduped[MAX_SPAWNS_PER_TURN:]],
            )

        requests = [
            r for r in requests
            if self.get_character(checkpoint, r.character_id) is None
        ]

        if not requests:
            return []

        spawned = []
        for req in requests:
            try:
                char, router_summary = await self._spawn_one(
                    checkpoint, req, default_location=acting_actor_location,
                )
                checkpoint.characters.append(char)
                spawned.append(char)
                logger.info("Spawned character: %s (%s)", char.name, char.character_id)
                _push_spawn_state_change(checkpoint, char, router_summary)
            except Exception as e:
                logger.warning("Failed to spawn %s: %s", req.character_id, e)

        return spawned

    async def _spawn_one(
        self, checkpoint: CheckpointFile, req: SpawnRequest,
        *, default_location: str = "",
    ) -> tuple[CharacterRecord, str]:
        """Generate a single character via LLM.

        Returns the freshly-built CharacterRecord plus the LLM-authored
        `router_summary` (one or two sentences for the next router
        call's State Changes block). The summary is NOT persisted on
        the record — it's an author-time scratch field consumed by
        `_push_spawn_state_change` and otherwise discarded.

        Spawn-location resolution chain:
          1. router-supplied `req.seed["location"]`
          2. `default_location` — the acting actor's current location label
          3. the LLM's `authored.location` — last-resort, only hit when
             both router and orchestrator omit a location.
        """
        from app.engine.context_builder import build_setting_summary
        setting_summary = build_setting_summary(checkpoint)
        world_lore = checkpoint.world_state.lore or "No detailed lore."
        physics = checkpoint.world_state.physics_ruleset
        world_rules = f"Strength limits: {physics.strength_limits}\nMagic: {'enabled' if physics.magic_enabled else 'disabled'}"

        seed_loc = (req.seed.get("location") or "").strip()
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
            location_context=location_context,
            character_id=req.character_id,
            spawn_seed=spawn_seed,
            existing_characters=existing,
            location=location,
        )

        from app.schemas.takeover import AuthoredCharacter
        response = await self.client.complete(
            role="agent_convenience",
            messages=messages,
            response_model=AuthoredCharacter,
            temperature=0.6,
            max_tokens=3000,
        )
        authored: AuthoredCharacter = response.parsed
        char = authored.to_record(character_id=req.character_id)
        char.agent_tier = CharacterAgentTier.convenience
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
