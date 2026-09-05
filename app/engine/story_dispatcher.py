"""Production LLM boundary for batched story advancement.

The dispatcher prepares character-owned proposals, makes one strict router call
for a logical batch, applies optional rules-adapter transactions, and composes
per-POV narration. Scheduling, checkpoint writes, and delivery remain outside
this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from copy import deepcopy
from typing import Sequence

from pydantic import ValidationError

from app.engine import narrator as narrator_module
from app.engine.character_agent import (
    CharacterAgent,
    CharacterAgentTurnDraft,
    CharacterPerceptionDraft,
)
from app.engine.character_manager import CharacterManager
from app.engine.content_manager import (
    append_content_manager_router_records,
    content_manager_enabled,
)
from app.engine.content_lookup import append_router_content_lookup_records_with_llm
from app.engine.content_resolver import append_pending_router_content_records
from app.engine.event_runtime import (
    autonomous_character_is_eligible,
    autonomous_character_is_ready,
)
from app.engine.context_builder import (
    build_dnd_character_equipment_sentence,
    build_dnd_character_identity_sentence,
    build_hidden_facts,
    build_setting_summary,
    build_world_rules,
    collect_player_ids,
    is_unbound_player_authored_slot,
)
from app.engine.prompt_manager import PromptManager
from app.engine.router_batch import MaterializedRouterBatch, materialize_router_batch
from app.llm.client import LLMClient
from app.schemas.characters import CharacterStatus, is_non_social_hazard
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.delivery import NarratorEventRef
from app.schemas.event_router import (
    CanonicalEventRecord,
    DndRouterBatchOutput,
    RouterBatchOutput,
    RouterInputEnvelope,
)
from app.schemas.events import ObservableFact
from app.schemas.narrator import NarratorOutput, TranscriptEntry
from app.schemas.one_star import (
    ONE_STAR_RULESET_ID,
    OneStarCanonicalEventRecord,
    OneStarRouterBatchOutput,
    OneStarRouterEventDraft,
)


logger = logging.getLogger(__name__)
EVENT_ROUTER_MAX_TOKENS = 8_000
DND5E_BASIC_RULESET_ID = "dnd5e_basic"

_REFLECTIVE_APPEARANCE_PATTERNS = (
    re.compile(r"\bthe (?:look|expression|kind|sort|body|build|gaze) of "
               r"(?:someone|somebody|a person|a man|a woman|a child)\b", re.I),
    re.compile(r"\b(?:moves?|moving|speaking|looking) like "
               r"(?:someone|somebody|a person|a man|a woman|a child)\b", re.I),
    re.compile(r"\bthe way (?:someone|somebody|a person|a man|a woman|a child)\b", re.I),
)


def _sanitize_appearance_fragment(text: str) -> str:
    """Keep exterior sentences and drop reflective narrator-style similes."""

    kept = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", (text or "").strip())
        if sentence.strip()
        and not any(
            pattern.search(sentence)
            for pattern in _REFLECTIVE_APPEARANCE_PATTERNS
        )
    ]
    return " ".join(kept)


def _ruleset_id(checkpoint: CheckpointFile) -> str:
    return checkpoint.session.config.settings.ruleset_id


def _materialize_adapter_lifecycle(
    checkpoint: CheckpointFile,
    output: RouterBatchOutput,
) -> None:
    """Project adapter-owned lifecycle into the shared event contract.

    Adapter state can activate a character without asking the router to repeat
    the same mutation in generic fields. Resolve that deterministic projection
    before common batch validation so a next turn sourced from the event sees
    the roster state that event will actually establish.
    """

    if _ruleset_id(checkpoint) != ONE_STAR_RULESET_ID:
        return
    from app.engine.one_star_adapter import one_star_summon_lifecycle

    for draft in output.events:
        if not isinstance(draft, OneStarRouterEventDraft):
            raise RuntimeError("One-Star router returned the generic draft type")
        spawns, wakes = one_star_summon_lifecycle(
            checkpoint,
            draft.state_updates,
        )
        if not spawns and not wakes:
            continue
        if draft.spawn or draft.activate:
            raise RuntimeError(
                "One-Star summon lifecycle was duplicated by router output"
            )
        draft.spawn.extend(spawns)
        draft.activate.extend(wakes)


def eligible_autonomous_character_ids(
    checkpoint: CheckpointFile,
    *,
    excluded_ids: set[str] | None = None,
    resolving_ids: set[str] | None = None,
) -> list[str]:
    """Return every safe agent-owned actor; semantic separation is router-owned."""

    excluded = excluded_ids or set()
    resolving = resolving_ids or set()
    return [
        character.character_id
        for character in checkpoint.characters
        if character.status == CharacterStatus.active
        and (
            autonomous_character_is_ready(checkpoint, character.character_id)
            or (
                character.character_id in resolving
                and autonomous_character_is_eligible(
                    checkpoint,
                    character.character_id,
                )
            )
        )
        and character.character_id not in excluded
    ]


def _router_world_lore(checkpoint: CheckpointFile) -> str:
    parts: list[str] = []
    facts = [fact.strip() for fact in checkpoint.world_state.facts if fact.strip()]
    if facts:
        parts.append("Key world facts:\n" + "\n".join(f"- {fact}" for fact in facts))
    if checkpoint.world_state.lore.strip():
        parts.append(checkpoint.world_state.lore.strip())
    return "\n\n".join(parts) or "No detailed lore available."


def _is_router_context_record(message: ConversationMessage) -> bool:
    return bool(
        message.role == "assistant"
        and isinstance(message.content, str)
        and message.content.startswith((
            "roster_seed\n",
            "prior_event ",
            "content_known ",
            "location_card ",
            "front_signal ",
        ))
    )


def _initial_roster_record(checkpoint: CheckpointFile) -> str:
    if any(
        isinstance(message.content, str)
        and message.content.startswith("roster_seed\n")
        for message in checkpoint.session_conversation
    ):
        return ""
    if any(not _is_router_context_record(message) for message in checkpoint.session_conversation):
        raise RuntimeError("router history contains a noncanonical message")
    entries: list[str] = []
    for character in checkpoint.characters:
        if character.status != CharacterStatus.active:
            continue
        lines = [
            f"- {character.character_id}",
            f"  name={character.name}",
            f"  role={character.public_sheet.role or 'unknown'}",
            f"  location={character.location or 'unknown'}",
        ]
        if is_non_social_hazard(character):
            lines.append("  kind=non-social hazard")
        identity = build_dnd_character_identity_sentence(checkpoint, character)
        equipment = build_dnd_character_equipment_sentence(checkpoint, character)
        if identity:
            lines.append(f"  {identity}")
        if equipment:
            lines.append(f"  {equipment}")
        entries.append("\n".join(lines))
    return (
        "roster_seed\nInitial active fictional identities:\n"
        + "\n\n".join(entries)
        + "\n"
        if entries
        else ""
    )


def _one_star_opening_roster(
    checkpoint: CheckpointFile,
    participant_ids: set[str],
) -> str:
    if _ruleset_id(checkpoint) != ONE_STAR_RULESET_ID:
        return ""
    from app.engine.one_star_adapter import (
        one_star_opening_roster_pool_id,
        one_star_opening_roster_preview,
    )

    pool_id = one_star_opening_roster_pool_id(checkpoint, participant_ids)
    if not pool_id:
        return ""
    roster = {item.character_id: item for item in checkpoint.characters}
    draws = one_star_opening_roster_preview(checkpoint, pool_id)
    lines = [
        "Resolved One-Star opening roster:",
        f"pool={pool_id}",
        "Required opening summon state update (copy exactly): "
        + json.dumps(
            {
                "kind": "summon",
                "target_id": pool_id,
                "value": str(len(draws)),
                "details": [],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    ]
    for draw in draws:
        character = roster.get(draw.existing_character_id)
        if character is None:
            raise RuntimeError("opening roster references a missing Hero")
        lines.append(
            f"- slot={draw.slot} id={character.character_id} "
            f"name={character.name} birth_stars={draw.birth_stars} "
            f"role={character.public_sheet.role} "
            f"appearance={character.public_sheet.appearance}"
        )
    return "\n".join(lines)


def _opening_context(
    checkpoint: CheckpointFile,
    envelope: RouterInputEnvelope,
) -> str:
    normalized = envelope.payload.strip().casefold()
    if normalized not in {"(begin)", "(arrive)"}:
        return ""
    participant_ids = (
        collect_player_ids(checkpoint)
        if normalized == "(begin)"
        else set(envelope.actor_ids)
    )
    roster = {item.character_id: item for item in checkpoint.characters}
    lines = ["Authored opening participants:"]
    for character_id in participant_ids:
        character = roster.get(character_id)
        if character is None:
            continue
        lines.append(
            f"- id={character_id} name={character.name} "
            f"role={character.public_sheet.role} status={character.status.value} "
            f"location={character.location or 'unplaced'} "
            f"appearance={character.public_sheet.appearance or 'unspecified'}"
        )
    if normalized == "(begin)":
        selected = _one_star_opening_roster(checkpoint, participant_ids)
        if selected:
            lines.append(selected)
        policy = checkpoint.world_state.opening
        if policy is None:
            lines.append("New opening spawns are forbidden.")
        else:
            lines.append(
                "New opening spawns are "
                + ("allowed when explicitly required." if policy.allow_spawns else "forbidden.")
            )
            if policy.context.strip():
                lines.append(policy.context.strip())
    return "\n".join(lines)


def _drain_engine_updates(checkpoint: CheckpointFile) -> str:
    updates = list(checkpoint.session.pending_engine_state_updates)
    checkpoint.session.pending_engine_state_updates = []
    if not updates:
        return ""
    return (
        "Durable external state updates (do not replay as new action):\n"
        + "\n".join(f"- {item}" for item in updates)
    )


def _ruleset_prompt(
    prompt_manager: PromptManager,
    checkpoint: CheckpointFile,
) -> str:
    ruleset_id = _ruleset_id(checkpoint)
    if ruleset_id == ONE_STAR_RULESET_ID:
        from app.engine.one_star_router_context import render_one_star_router_static_config

        return prompt_manager.render(
            "event_router_ruleset_one_star",
            one_star_static_config=render_one_star_router_static_config(checkpoint),
        ).strip()
    if ruleset_id == DND5E_BASIC_RULESET_ID:
        return prompt_manager.render("event_router_ruleset_dnd5e").strip()
    return prompt_manager.render("event_router_ruleset_default").strip()


def _router_input_block(
    checkpoint: CheckpointFile,
    inputs: list[RouterInputEnvelope],
) -> str:
    resolving_ids = {
        character_id
        for envelope in inputs
        if envelope.kind == "cat_ii_resolution"
        for character_id in envelope.participant_ids
    }
    agent_owned = eligible_autonomous_character_ids(
        checkpoint,
        resolving_ids=resolving_ids,
    )
    blocks = [
        *(
            block
            for envelope in inputs
            if (block := _opening_context(checkpoint, envelope))
        ),
        _drain_engine_updates(checkpoint),
        "<inputs>\n"
        + json.dumps(
            [envelope.model_dump(mode="json") for envelope in inputs],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n</inputs>",
        "<autonomous_character_ids>\n"
        + (",".join(agent_owned) or "none")
        + "\n</autonomous_character_ids>",
    ]
    return "\n\n".join(block for block in blocks if block.strip())


def _router_snapshot(checkpoint: CheckpointFile) -> dict[str, object]:
    return {
        "pending_engine_state_updates": list(
            checkpoint.session.pending_engine_state_updates
        ),
        "content_state": deepcopy(checkpoint.session.content_state),
        "content_manager_preflight_cycle": (
            checkpoint.session.content_manager_preflight_cycle
        ),
        "content_manager_last_run_cycle": checkpoint.session.content_manager_last_run_cycle,
        "history_length": len(checkpoint.session_conversation),
    }


def _restore_router_snapshot(
    checkpoint: CheckpointFile,
    snapshot: dict[str, object],
) -> None:
    checkpoint.session.pending_engine_state_updates = list(
        snapshot["pending_engine_state_updates"]
    )
    checkpoint.session.content_state = deepcopy(snapshot["content_state"])
    checkpoint.session.content_manager_preflight_cycle = int(
        snapshot["content_manager_preflight_cycle"]
    )
    checkpoint.session.content_manager_last_run_cycle = int(
        snapshot["content_manager_last_run_cycle"]
    )
    del checkpoint.session_conversation[int(snapshot["history_length"]):]


def _compact(value: str, limit: int = 900) -> str:
    return " ".join(value.split())[:limit]


def router_history_record(
    checkpoint: CheckpointFile,
    event: CanonicalEventRecord,
) -> str:
    groups = event.observers
    lines = [
        f"prior_event {event.event_id} lane={event.causal_lane_id} "
        f"@{event.effective_at_s}+{event.duration_s} "
        f"actors={','.join(event.actor_ids) or '-'}",
        "outcomes "
        f"feasible={','.join(event.feasible_submission_ids) or '-'} "
        f"infeasible={','.join(event.infeasible_submission_ids) or '-'}",
        "observers "
        f"direct={','.join(groups.direct) or '-'} "
        f"indirect={','.join(groups.indirect) or '-'} "
        f"inferred={','.join(groups.inferred) or '-'}",
    ]
    for fact in event.observable_facts:
        audience = (
            "all" if fact.audience == "all_observers" else ",".join(fact.visible_to)
        )
        lines.append(
            f"fact +{fact.at_offset_s}/{fact.duration_s} to={audience}: "
            + _compact(fact.text)
        )
    roster = {item.character_id: item for item in checkpoint.characters}
    for request in event.spawn:
        character = roster.get(request.character_id)
        lines.append(
            f"spawn {request.character_id}"
            + (f" name={character.name}" if character is not None else "")
            + f" role={_compact(request.seed.role, 160)}"
        )
    for signal in event.activate:
        lines.append(f"activate {signal.character_id} at={signal.location_label}")
    if event.dormant:
        lines.append("dormant=" + ",".join(event.dormant))
    if event.cull:
        lines.append("cull=" + ",".join(event.cull))
    for update in event.location_updates:
        lines.append(f"location {update.character_id}={update.location_label}")
    for directive in event.commitment_opens:
        lines.append(
            "commitment_open actors="
            + ",".join(directive.actor_ids)
            + " description="
            + _compact(directive.description)
        )
    for signal in event.commitment_resolutions:
        target = signal.commitment_id or ",".join(signal.actor_ids)
        lines.append(f"commitment_{signal.reason} target={target}")
    for signal in event.commitment_interrupts:
        target = signal.commitment_id or ",".join(signal.actor_ids)
        lines.append(
            f"commitment_interrupted target={target} reason={_compact(signal.reason)}"
        )
    for update in getattr(event, "state_updates", ()):
        details = ",".join(update.details)
        lines.append(
            f"one_star_update {update.kind} {update.target_id}={update.value}"
            + (f" [{details}]" if details else "")
        )
    mode = getattr(event, "interaction_mode", "narrative")
    if mode != "narrative":
        lines.append(f"dnd_interaction={mode}")
    return "\n".join(lines)


class StoryDispatcher:
    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager
        self.character_agent = CharacterAgent(client, prompt_manager)
        self.character_manager = CharacterManager(client, prompt_manager)
        from app.engine.dnd_cat_ii import DndCatIIResolver
        from app.engine.dnd_combat_resolution import DndCombatResolver

        self.dnd_cat_ii = DndCatIIResolver(client, prompt_manager)
        self.dnd_combat = DndCombatResolver(client, prompt_manager)

    async def resolve_dnd_cat_ii(self, *, ckpt: CheckpointFile, opened):
        return await self.dnd_cat_ii.resolve_cat_ii(
            ckpt=ckpt,
            cat_ii_event=opened,
            content_context_records=append_pending_router_content_records(ckpt),
        )

    async def resolve_dnd_combat_action(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        intention: str,
    ):
        return await self.dnd_combat.resolve_combat_action(
            ckpt=ckpt,
            actor_id=actor_id,
            intention=intention,
            content_context_records=append_pending_router_content_records(ckpt),
        )

    async def continue_dnd_combat_transaction(
        self,
        *,
        ckpt: CheckpointFile,
        event_id: str,
    ):
        return await self.dnd_combat.continue_combat_transaction(
            ckpt=ckpt,
            event_id=event_id,
            content_context_records=append_pending_router_content_records(ckpt),
        )

    async def route_batch(
        self,
        *,
        ckpt: CheckpointFile,
        inputs: list[RouterInputEnvelope],
    ) -> MaterializedRouterBatch:
        if not inputs:
            raise ValueError("router batch cannot be empty")
        snapshot = _router_snapshot(ckpt)
        correlation = hashlib.sha256(
            "\x1f".join(item.submission_id for item in inputs).encode("utf-8")
        ).hexdigest()[:12]
        try:
            current_input = "\n".join(item.payload for item in inputs)
            actor_id = next(
                (
                    actor
                    for envelope in inputs
                    for actor in envelope.actor_ids
                ),
                "-",
            )
            if ckpt.session.active_combat is None:
                if content_manager_enabled(ckpt):
                    await append_content_manager_router_records(
                        ckpt,
                        actor_id=actor_id,
                        current_input=current_input,
                        client=self.client,
                        prompt_mgr=self.prompt_manager,
                    )
                else:
                    await append_router_content_lookup_records_with_llm(
                        ckpt,
                        actor_id=actor_id,
                        current_input=current_input,
                        client=self.client,
                        prompt_mgr=self.prompt_manager,
                    )
            seed = _initial_roster_record(ckpt)
            if seed:
                ckpt.session_conversation.append(
                    ConversationMessage(role="assistant", content=seed)
                )
            messages = self.prompt_manager.render_conversation(
                "event_router",
                history=ckpt.session_conversation,
                setting_summary=build_setting_summary(ckpt),
                world_lore=_router_world_lore(ckpt),
                world_rules=build_world_rules(ckpt),
                hidden_lore=ckpt.world_state.hidden_lore or "None.",
                hidden_facts=build_hidden_facts(ckpt, empty="None."),
                router_ruleset_addon=_ruleset_prompt(self.prompt_manager, ckpt),
                router_input_block=_router_input_block(ckpt, inputs),
            )
            ruleset = _ruleset_id(ckpt)
            response_model: type[RouterBatchOutput]
            if ruleset == ONE_STAR_RULESET_ID:
                response_model = OneStarRouterBatchOutput
            elif ruleset == DND5E_BASIC_RULESET_ID:
                response_model = DndRouterBatchOutput
            else:
                response_model = RouterBatchOutput
            logger.info(
                "router batch %s: %d input(s)",
                correlation,
                len(inputs),
            )
            response = await self.client.complete(
                role="event_router",
                messages=messages,
                response_model=response_model,
                temperature=0.35,
                max_tokens=EVENT_ROUTER_MAX_TOKENS,
                cache=True,
                compact=True,
            )
            _materialize_adapter_lifecycle(ckpt, response.parsed)
            materialized = materialize_router_batch(
                checkpoint=ckpt,
                inputs=inputs,
                output=response.parsed,
            )
            return materialized
        except Exception:
            _restore_router_snapshot(ckpt, snapshot)
            logger.exception("router batch %s failed", correlation)
            raise

    async def draft_character_turn(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        local_context: str,
    ) -> CharacterAgentTurnDraft:
        character = next(
            (item for item in ckpt.characters if item.character_id == character_id),
            None,
        )
        if character is None:
            raise ValueError(f"unknown character turn actor {character_id!r}")
        if is_unbound_player_authored_slot(ckpt, character):
            raise RuntimeError("an unclaimed player-authored character cannot act")
        return await self.character_agent.draft_turn(
            character=character,
            checkpoint=ckpt,
            frame="autonomous",
            local_context=local_context,
            include_location=True,
        )

    def commit_character_turn(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        draft: CharacterAgentTurnDraft,
        committed_at_s: int,
    ) -> None:
        character = next(
            (item for item in ckpt.characters if item.character_id == character_id),
            None,
        )
        if character is None:
            raise RuntimeError("prepared character disappeared before commit")
        self.character_agent.commit_draft(
            character,
            ckpt,
            draft,
            committed_at_s=committed_at_s,
        )

    async def prepare_batch(
        self,
        *,
        ckpt: CheckpointFile,
        batch: MaterializedRouterBatch,
        inputs: Sequence[RouterInputEnvelope],
        player_actor_ids: set[str],
    ) -> None:
        """Apply every fallible adapter and spawn operation before commit."""

        records = [item.record for item in batch.events]
        await self._prepare_appearance_harvests(ckpt, batch)
        if _ruleset_id(ckpt) == DND5E_BASIC_RULESET_ID:
            from app.engine.dnd_story_adapter import prepare_dnd_batch

            prepare_dnd_batch(ckpt, batch, inputs)
        one_star_hero_ids: set[str] = set()
        if _ruleset_id(ckpt) == ONE_STAR_RULESET_ID:
            from app.engine.one_star_adapter import one_star_summon_lifecycle

            for record in records:
                if not isinstance(record, OneStarCanonicalEventRecord):
                    raise RuntimeError("One-Star router returned the generic record type")
                spawns, wakes = one_star_summon_lifecycle(ckpt, record.state_updates)
                if (spawns or wakes) and (
                    list(record.spawn) != list(spawns)
                    or list(record.activate) != list(wakes)
                ):
                    raise RuntimeError(
                        "One-Star summon lifecycle diverged before preparation"
                    )
                one_star_hero_ids.update(item.character_id for item in spawns)

        spawn_requests = [request for record in records for request in record.spawn]
        if spawn_requests:
            spawned_characters = await self.character_manager.spawn_characters(
                ckpt,
                spawn_requests,
                acting_actor_location="",
                one_star_hero_ids=(
                    one_star_hero_ids
                    if _ruleset_id(ckpt) == ONE_STAR_RULESET_ID
                    else None
                ),
                name_derived_character_ids=(
                    one_star_hero_ids
                    if _ruleset_id(ckpt) == ONE_STAR_RULESET_ID
                    else None
                ),
            )
            if len(spawned_characters) != len(spawn_requests):
                raise RuntimeError(
                    "character generation did not return every requested spawn"
                )
            final_id_by_requested_id = {
                request.character_id: character.character_id
                for request, character in zip(
                    spawn_requests,
                    spawned_characters,
                    strict=True,
                )
            }
            for record in records:
                record.spawn = [
                    request.model_copy(update={
                        "character_id": final_id_by_requested_id[
                            request.character_id
                        ],
                    })
                    for request in record.spawn
                ]

        for record in records:
            if isinstance(record, OneStarCanonicalEventRecord):
                await self._prepare_one_star_event(
                    ckpt,
                    record,
                    player_actor_ids=player_actor_ids,
                )
                self.character_manager.apply_roster_updates(ckpt, record)
            else:
                self.character_manager.apply_roster_updates(ckpt, record)
        if _ruleset_id(ckpt) == DND5E_BASIC_RULESET_ID:
            from app.engine.dnd_inventory import apply_loot_offers_from_events

            apply_loot_offers_from_events(ckpt, records)
        # Adapter enrichment mutates staged records. Re-run the strict durable
        # schemas before any event or adapter state can reach the sole writer.
        for record in records:
            type(record).model_validate(record.model_dump())

    async def _prepare_appearance_harvests(
        self,
        checkpoint: CheckpointFile,
        batch: MaterializedRouterBatch,
    ) -> None:
        """Resolve router-requested current appearances concurrently.

        Each target runs against an immutable snapshot. Results and actor-owned
        presentation memory are merged into the staged checkpoint only after
        every requested call succeeds, so a partial harvest can never leak
        through a rejected router batch.
        """

        targets = list(dict.fromkeys(
            character_id
            for event in batch.events
            for character_id in event.appearance_target_ids
        ))
        if not targets:
            return
        allowed = set(eligible_autonomous_character_ids(checkpoint))
        invalid = set(targets) - allowed
        if invalid:
            raise RuntimeError(
                "router requested appearance from a non-agent-owned character: "
                + ", ".join(sorted(invalid))
            )
        frozen = checkpoint.model_dump_json(
            context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
        )

        async def _draft(character_id: str) -> tuple[str, CharacterPerceptionDraft]:
            snapshot = CheckpointFile.model_validate_json(frozen)
            character = next(
                item
                for item in snapshot.characters
                if item.character_id == character_id
            )
            return character_id, await self.character_agent.draft_perception(
                character,
                snapshot,
            )

        drafted = dict(await asyncio.gather(*(_draft(value) for value in targets)))
        roster = {item.character_id: item for item in checkpoint.characters}
        cleaned: dict[str, str] = {}
        for character_id in targets:
            draft = drafted[character_id]
            text = _sanitize_appearance_fragment(draft.output.public_text)
            if not text:
                raise RuntimeError(
                    f"appearance harvest returned no usable prose for {character_id}"
                )
            cleaned[character_id] = text

        for character_id in targets:
            self.character_agent.commit_perception(
                roster[character_id],
                checkpoint,
                drafted[character_id],
            )
        for event in batch.events:
            for character_id in event.appearance_target_ids:
                event.record.observable_facts.append(ObservableFact.all(
                    f"[loadout - {roster[character_id].name}] "
                    f"{cleaned[character_id]}",
                    at_offset_s=event.record.duration_s,
                ))

    async def _prepare_one_star_event(
        self,
        checkpoint: CheckpointFile,
        event: OneStarCanonicalEventRecord,
        *,
        player_actor_ids: set[str],
    ) -> None:
        from app.engine.one_star_adapter import (
            OneStarTransactionError,
            apply_one_star_prepared_mutation,
            one_star_event_fingerprint,
            one_star_state_updates_to_transaction,
            preflight_one_star_account_updates,
            prepare_one_star_transaction,
        )

        for actor_id in set(event.actor_ids).intersection(player_actor_ids):
            preflight_one_star_account_updates(
                checkpoint,
                event.state_updates,
                initiating_actor_id=actor_id,
                canonical_at_s=event.effective_at_s + event.duration_s,
            )
        try:
            fresh_summon_character_ids = (
                [request.character_id for request in event.spawn]
                if any(update.kind == "summon" for update in event.state_updates)
                else None
            )
            transaction = one_star_state_updates_to_transaction(
                checkpoint,
                event.state_updates,
                canonical_at_s=event.effective_at_s + event.duration_s,
                fresh_summon_character_ids=fresh_summon_character_ids,
            )
        except ValidationError as exc:
            raise OneStarTransactionError(
                "One-Star state update violates its typed bounds"
            ) from exc
        activation_locations = {
            item.character_id: item.location_label for item in event.activate
        }
        locations = {
            item.character_id: item.location_label for item in event.location_updates
        }
        fingerprint = one_star_event_fingerprint(event.model_dump(mode="json"))
        prepared = prepare_one_star_transaction(
            checkpoint,
            event_id=event.event_id,
            transaction=transaction,
            spawned_character_ids=[item.character_id for item in event.spawn],
            activated_character_ids=[item.character_id for item in event.activate],
            activated_character_locations=activation_locations,
            dormant_character_ids=event.dormant,
            generic_culled_character_ids=event.cull,
            location_updates=locations,
            canonical_at_s=event.effective_at_s + event.duration_s,
            event_fingerprint=fingerprint,
            initiating_actor_ids=event.actor_ids,
        )
        if prepared.already_applied:
            raise OneStarTransactionError("One-Star event was already applied")
        self._append_one_star_consequences(checkpoint, event, prepared.system_consequences)
        apply_one_star_prepared_mutation(checkpoint, prepared)

    @staticmethod
    def _append_one_star_consequences(
        checkpoint: CheckpointFile,
        event: OneStarCanonicalEventRecord,
        consequences: Sequence[object],
    ) -> None:
        if not consequences:
            return
        known = {item.character_id for item in checkpoint.characters}
        original_observers = list(event.observer_ids)
        added_observers: list[str] = []
        consequence_records: list[tuple[str, list[str]]] = []
        for consequence in consequences:
            text = str(getattr(consequence, "text", "") or "").strip()
            recipients = list(dict.fromkeys(
                str(value).strip()
                for value in getattr(consequence, "recipient_character_ids", ())
                if str(value).strip()
            ))
            if not text or not recipients or set(recipients) - known:
                raise RuntimeError("invalid One-Star deterministic consequence")
            added_observers.extend(
                value
                for value in recipients
                if value not in original_observers and value not in added_observers
            )
            consequence_records.append((text, recipients))
        if added_observers:
            for fact in event.observable_facts:
                if fact.audience == "all_observers":
                    fact.audience = "only"
                    fact.visible_to = list(original_observers)
            event.observers.indirect.extend(added_observers)
        from app.schemas.events import ObservableFact

        event.observable_facts.extend(
            ObservableFact.only(
                text,
                recipients,
                at_offset_s=event.duration_s,
            )
            for text, recipients in consequence_records
        )

    async def narrator_compose(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        event_refs: list[NarratorEventRef],
        partial_mode: bool,
        user_input: str,
        handoff_policy: str,
        handoff_context: str,
        narration_mode: str,
    ) -> tuple[NarratorOutput, TranscriptEntry]:
        return await narrator_module.compose_pov_render(
            client=self.client,
            prompt_mgr=self.prompt_manager,
            ckpt=ckpt,
            pov_character_id=character_id,
            buffered_events=event_refs,
            partial_mode=partial_mode,
            user_input=user_input,
            handoff_policy=handoff_policy,
            handoff_context=handoff_context,
            narration_mode=narration_mode,
        )


def append_router_history(
    checkpoint: CheckpointFile,
    events: Sequence[CanonicalEventRecord],
) -> None:
    checkpoint.session_conversation.extend(
        ConversationMessage(
            role="assistant",
            content=router_history_record(checkpoint, event),
        )
        for event in events
    )
