"""Single-writer orchestration for the batched canonical story runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

from app.engine.character_presentation import select_player_character_presentation
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.delivery_outbox import claim_deliveries
from app.engine.delivery_response import response_from_deliveries
from app.engine.event_runtime import release_action_obligation
from app.engine.model_config_sync import sync_checkpoint_runtime_models
from app.engine.narrator_delivery import process_narrator_lanes
from app.engine.prompt_manager import PromptManager
from app.engine.session_writer import SessionWriterLocks
from app.engine.story_contracts import AuthoritativeResultPlan
from app.engine.story_coordinator import (
    AdvanceResult,
    PreparedRouterInput,
    advance_story,
    commit_adapter_resolution,
    collect_player_contest_response,
    immutable_checkpoint,
    player_input,
    prepare_autonomous_contest_resolutions,
    prepare_ready_frontier_batch,
    ready_frontier_turns,
    release_frontier_gates_for_pov_action,
    replace_checkpoint_state,
)
from app.engine.story_dispatcher import StoryDispatcher
from app.llm.client import LLMClient
from app.schemas.characters import CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.delivery import DeliveryPayload
from app.schemas.requests import TurnRequest
from app.schemas.responses import TurnResponse


logger = logging.getLogger(__name__)


def _checkpoint_fingerprint(checkpoint: CheckpointFile) -> str:
    payload = checkpoint.model_dump_json(
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _preparation_fingerprint(checkpoint: CheckpointFile) -> str:
    """Hash story semantics while ignoring frontend delivery bookkeeping."""

    payload = checkpoint.model_dump(
        mode="json",
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
    )
    payload.pop("narrator_conversations", None)
    session = payload["session"]
    session.pop("turn_index", None)
    session.pop("narrator_render_jobs", None)
    session.pop("delivery_outbox", None)
    session.pop("last_acknowledged_event_sequence_by_pov", None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _resolve_actor(checkpoint: CheckpointFile, requested_id: str) -> str:
    actor_id = requested_id or checkpoint.session.player_character_id
    character = next(
        (item for item in checkpoint.characters if item.character_id == actor_id),
        None,
    )
    if character is None:
        raise ValueError("acting character does not exist")
    if character.status != CharacterStatus.active:
        raise ValueError("acting character is not active")
    if actor_id not in checkpoint.session.character_bindings:
        raise ValueError("acting character is not bound to a player")
    return actor_id


class Orchestrator:
    """Own one durable writer and one autonomous worker per session."""

    def __init__(
        self,
        client: LLMClient,
        checkpoint_mgr: CheckpointManager,
        prompt_mgr: PromptManager,
        *,
        image_sink: Any | None = None,
        image_generation: Any | None = None,
        spawn_authoring: Any | None = None,
    ) -> None:
        self.client = client
        self.checkpoint_mgr = checkpoint_mgr
        self.prompt_mgr = prompt_mgr
        self.dispatcher = StoryDispatcher(client, prompt_mgr)
        self.image_sink = image_sink
        self.image_generation = image_generation
        self.spawn_authoring = spawn_authoring
        self.session_locks = SessionWriterLocks()
        self._autonomous_tasks: dict[str, asyncio.Task[None]] = {}
        self._autonomous_phase: dict[str, str] = {}
        self._prepared_autonomous: dict[
            str,
            tuple[str, list[PreparedRouterInput]],
        ] = {}

    def _cached_prepared(
        self,
        checkpoint: CheckpointFile,
    ) -> list[PreparedRouterInput]:
        cached = self._prepared_autonomous.get(checkpoint.session.session_id)
        if cached is None:
            return []
        fingerprint, prepared = cached
        if fingerprint != _preparation_fingerprint(checkpoint):
            self._prepared_autonomous.pop(checkpoint.session.session_id, None)
            return []
        return list(prepared)

    def _remember_prepared(
        self,
        checkpoint: CheckpointFile,
        prepared: list[PreparedRouterInput],
    ) -> None:
        session_id = checkpoint.session.session_id
        if not prepared:
            self._prepared_autonomous.pop(session_id, None)
            return
        self._prepared_autonomous[session_id] = (
            _preparation_fingerprint(checkpoint),
            list(prepared),
        )

    def _forget_prepared(self, session_id: str) -> None:
        self._prepared_autonomous.pop(session_id, None)

    async def process_turn(self, request: TurnRequest) -> TurnResponse:
        lock = await self.session_locks.for_session(request.session_id)
        async with lock:
            return await self.process_turn_locked(request)

    async def process_turn_locked(self, request: TurnRequest) -> TurnResponse:
        checkpoint = self.checkpoint_mgr.load_latest(request.session_id)
        sync_checkpoint_runtime_models(checkpoint, self.client.config)
        actor_id = _resolve_actor(checkpoint, request.acting_character_id)
        if request.display_key:
            select_player_character_presentation(
                checkpoint,
                actor_id,
                request.display_key,
            )

        cached_prepared = self._cached_prepared(checkpoint)
        release_frontier_gates_for_pov_action(checkpoint, actor_id)
        self._cancel_uncommitted_autonomous(request.session_id)
        obligation = checkpoint.session.action_obligations.get(actor_id)
        if obligation is not None:
            if obligation.kind == "combat_reaction":
                result = await self._advance_dnd_combat_action(
                    checkpoint,
                    actor_id=actor_id,
                    intention=request.user_input,
                    is_reaction=True,
                    player_input=True,
                )
                return self._save_claim_and_schedule(
                    checkpoint,
                    acting_character_id=actor_id,
                    pause_reason=result.pause_reason,
                    result=result,
                )
            if obligation.kind != "cat_ii_response":
                raise ValueError(
                    f"character must resolve {obligation.kind} before acting"
                )
            prepared = collect_player_contest_response(
                checkpoint,
                character_id=actor_id,
                intention=request.user_input,
            )
            if not prepared:
                self._forget_prepared(request.session_id)
                checkpoint.session.turn_index += 1
                checkpoint_id = self.checkpoint_mgr.save(checkpoint)
                response = response_from_deliveries(
                    session_id=request.session_id,
                    checkpoint_id=checkpoint_id,
                    turn_index=checkpoint.session.turn_index,
                    acting_character_id=actor_id,
                    deliveries=[],
                    pause_reason="cat_ii_pending",
                )
                self._schedule_autonomous(request.session_id)
                return response
        else:
            from app.engine.dnd_story_adapter import combat_contains_character

            if combat_contains_character(checkpoint, actor_id):
                result = await self._advance_dnd_combat_action(
                    checkpoint,
                    actor_id=actor_id,
                    intention=request.user_input,
                    is_reaction=False,
                    player_input=True,
                )
                return self._save_claim_and_schedule(
                    checkpoint,
                    acting_character_id=actor_id,
                    pause_reason=result.pause_reason,
                    result=result,
                )
            prepared = [player_input(
                checkpoint,
                character_id=actor_id,
                payload=request.user_input,
            )]

        prepared = await prepare_ready_frontier_batch(
            checkpoint,
            self.dispatcher,
            initial=prepared,
            preprepared=cached_prepared,
        )
        result = await advance_story(
            checkpoint,
            self.dispatcher,
            prepared,
            user_input_by_pov={actor_id: request.user_input},
        )
        self._remember_prepared(checkpoint, result.prepared_followups)
        checkpoint_id = self.checkpoint_mgr.save(checkpoint)
        response = self._claim_response(
            checkpoint,
            checkpoint_id=checkpoint_id,
            acting_character_id=actor_id,
            pause_reason=result.pause_reason,
            consumer_id=f"inline:{request.session_id}:{checkpoint.session.turn_index}",
        )
        self.checkpoint_mgr.save(checkpoint)
        self._schedule_autonomous(request.session_id)
        return response

    async def _advance_dnd_combat_action(
        self,
        checkpoint: CheckpointFile,
        *,
        actor_id: str,
        intention: str,
        is_reaction: bool,
        player_input: bool,
        character_draft=None,
    ) -> AdvanceResult:
        from app.engine.dnd_cat_ii import DndCatIIRollsPending
        from app.engine.dnd_roll_display import (
            completed_automatic_roll_keys,
            dice_roll_displays_since,
        )
        from app.engine.dnd_story_adapter import (
            finalize_combat_action,
            validate_combat_action_actor,
        )

        if checkpoint.session.config.settings.ruleset_id != "dnd5e_basic":
            raise RuntimeError("active D&D combat requires the D&D rules adapter")
        if not intention.strip():
            raise ValueError("combat intention cannot be blank")
        validate_combat_action_actor(
            checkpoint,
            character_id=actor_id,
            is_reaction=is_reaction,
        )
        working = immutable_checkpoint(checkpoint)
        roll_keys_before = completed_automatic_roll_keys(working)
        try:
            resolution = await self.dispatcher.resolve_dnd_combat_action(
                ckpt=working,
                actor_id=actor_id,
                intention=intention,
            )
        except DndCatIIRollsPending:
            transaction = next(
                (
                    item
                    for item in reversed(working.session.cat_ii_roll_transactions)
                    if item.source == "combat" and item.actor_id == actor_id
                    and item.status != "finalized"
                ),
                None,
            )
            if transaction is None:
                raise RuntimeError("combat roll pause created no transaction")
            transaction.context["story_is_reaction"] = is_reaction
            if character_draft is not None:
                self.dispatcher.commit_character_turn(
                    ckpt=working,
                    character_id=actor_id,
                    draft=character_draft,
                    committed_at_s=next(
                        character.clock_at_s
                        for character in working.characters
                        if character.character_id == actor_id
                    ),
                )
            if player_input:
                working.session.autonomous_router_batches_since_player = 0
            else:
                working.session.autonomous_router_batches_since_player += 1
            replace_checkpoint_state(checkpoint, working)
            checkpoint.session.turn_index += 1
            return AdvanceResult(pause_reason="cat_ii_pending_rolls")

        event = resolution.event
        actor_clock = next(
            character.clock_at_s
            for character in working.characters
            if character.character_id == actor_id
        )
        event.effective_at_s = max(
            event.effective_at_s,
            actor_clock,
            working.session.leading_at_s,
        )
        installed = finalize_combat_action(
            working,
            event_id=event.event_id,
            actor_id=actor_id,
            reaction_candidate_ids=resolution.reaction_candidate_ids,
            is_reaction=is_reaction,
        )
        drafts = (
            [(actor_id, character_draft, event.effective_at_s + event.duration_s)]
            if character_draft is not None
            else []
        )
        return await commit_adapter_resolution(
            checkpoint,
            self.dispatcher,
            working=working,
            resolution=resolution,
            user_input_by_pov={actor_id: intention} if player_input else None,
            partial_pov_ids=set(installed),
            character_drafts=drafts,
            player_input_included=player_input,
            dice_rolls=[
                display.model_dump()
                for display in dice_roll_displays_since(working, roll_keys_before)
            ],
            pause_reason="combat_reaction_pending" if installed else "",
        )

    async def _continue_dnd_combat_transaction(
        self,
        checkpoint: CheckpointFile,
        *,
        event_id: str,
        player_input: bool,
    ) -> tuple[AdvanceResult, str]:
        from app.engine.dnd_roll_display import (
            completed_automatic_roll_keys,
            dice_roll_displays_since,
        )
        from app.engine.dnd_story_adapter import finalize_combat_action

        transaction = next(
            (
                item
                for item in checkpoint.session.cat_ii_roll_transactions
                if item.event_id == event_id and item.source == "combat"
            ),
            None,
        )
        if transaction is None or transaction.status in {"cancelled", "finalized"}:
            raise ValueError("combat roll transaction is no longer active")
        actor_id = transaction.actor_id
        is_reaction = bool(transaction.context.get("story_is_reaction", False))
        working = immutable_checkpoint(checkpoint)
        roll_keys_before = completed_automatic_roll_keys(working)
        resolution = await self.dispatcher.continue_dnd_combat_transaction(
            ckpt=working,
            event_id=event_id,
        )
        installed = finalize_combat_action(
            working,
            event_id=resolution.event.event_id,
            actor_id=actor_id,
            reaction_candidate_ids=resolution.reaction_candidate_ids,
            is_reaction=is_reaction,
        )
        result = await commit_adapter_resolution(
            checkpoint,
            self.dispatcher,
            working=working,
            resolution=resolution,
            partial_pov_ids=set(installed),
            player_input_included=player_input,
            dice_rolls=[
                display.model_dump()
                for display in dice_roll_displays_since(working, roll_keys_before)
            ],
            pause_reason="combat_reaction_pending" if installed else "",
        )
        return result, actor_id

    def _save_claim_and_schedule(
        self,
        checkpoint: CheckpointFile,
        *,
        acting_character_id: str,
        pause_reason: str,
        result: AdvanceResult | None = None,
    ) -> TurnResponse:
        if result is not None:
            self._remember_prepared(checkpoint, result.prepared_followups)
        else:
            self._forget_prepared(checkpoint.session.session_id)
        checkpoint_id = self.checkpoint_mgr.save(checkpoint)
        response = self._claim_response(
            checkpoint,
            checkpoint_id=checkpoint_id,
            acting_character_id=acting_character_id,
            pause_reason=pause_reason,
            consumer_id=(
                f"inline:{checkpoint.session.session_id}:"
                f"{checkpoint.session.turn_index}"
            ),
        )
        self.checkpoint_mgr.save(checkpoint)
        self._schedule_autonomous(checkpoint.session.session_id)
        return response

    async def process_authoritative_result(
        self,
        *,
        session_id: str,
        viewpoint_character_id: str,
        plan_builder: Callable[[CheckpointFile], AuthoritativeResultPlan],
    ) -> TurnResponse:
        lock = await self.session_locks.for_session(session_id)
        async with lock:
            checkpoint = self.checkpoint_mgr.load_latest(session_id)
            sync_checkpoint_runtime_models(checkpoint, self.client.config)
            viewpoint_id = _resolve_actor(checkpoint, viewpoint_character_id)
            cached_prepared = self._cached_prepared(checkpoint)
            release_frontier_gates_for_pov_action(checkpoint, viewpoint_id)
            self._cancel_uncommitted_autonomous(session_id)
            plan = plan_builder(immutable_checkpoint(checkpoint))

            frozen = immutable_checkpoint(checkpoint)
            contributions = await asyncio.gather(*(
                self.dispatcher.draft_character_turn(
                    ckpt=immutable_checkpoint(frozen),
                    character_id=request.character_id,
                    local_context=request.local_context,
                )
                for request in plan.contribution_requests
            ))
            contribution_payload = [
                {
                    "character_id": request.character_id,
                    "contribution": draft.output.public_text,
                }
                for request, draft in zip(
                    plan.contribution_requests,
                    contributions,
                    strict=True,
                )
                if not draft.output.is_silence
            ]
            payload = json.dumps(
                {
                    "authority": plan.authority_label,
                    "submitted_command": plan.submitted_command,
                    "fixed_result": plan.result_text,
                    "character_contributions": contribution_payload,
                    "fixed_location_updates": list(plan.location_updates),
                    "fixed_state_updates": list(plan.state_updates),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            base = player_input(
                checkpoint,
                character_id=viewpoint_id,
                payload=payload,
                kind="authoritative_result",
            )
            participant_ids = list(dict.fromkeys([
                viewpoint_id,
                plan.ruleset_actor_id,
                *(request.character_id for request in plan.contribution_requests),
            ]))
            participant_ids = [value for value in participant_ids if value]
            prepared = [PreparedRouterInput(
                envelope=base.envelope.model_copy(update={
                    "actor_ids": (
                        [plan.ruleset_actor_id]
                        if plan.ruleset_actor_id
                        else [viewpoint_id]
                    ),
                    "participant_ids": participant_ids,
                }),
                attached_character_drafts=tuple(
                    (request.character_id, draft)
                    for request, draft in zip(
                        plan.contribution_requests,
                        contributions,
                        strict=True,
                    )
                ),
            )]
            prepared = await prepare_ready_frontier_batch(
                checkpoint,
                self.dispatcher,
                initial=prepared,
                preprepared=cached_prepared,
            )
            result = await advance_story(
                checkpoint,
                self.dispatcher,
                prepared,
                user_input_by_pov={viewpoint_id: plan.submitted_command},
            )
            self._remember_prepared(checkpoint, result.prepared_followups)
            checkpoint_id = self.checkpoint_mgr.save(checkpoint)
            response = self._claim_response(
                checkpoint,
                checkpoint_id=checkpoint_id,
                acting_character_id=viewpoint_id,
                pause_reason=result.pause_reason,
                consumer_id=f"inline:{session_id}:{checkpoint.session.turn_index}",
            )
            self.checkpoint_mgr.save(checkpoint)
            self._schedule_autonomous(session_id)
            return response

    async def retry_pending_narrator_render(self, session_id: str) -> TurnResponse:
        lock = await self.session_locks.for_session(session_id)
        async with lock:
            checkpoint = self.checkpoint_mgr.load_latest(session_id)
            lane_ids = [
                item.lane_id
                for item in checkpoint.session.narrator_render_jobs
                if item.status in {"pending", "failed"}
            ]
            if not lane_ids:
                return response_from_deliveries(
                    session_id=session_id,
                    checkpoint_id=f"ckpt_{checkpoint.session.turn_index:04d}",
                    turn_index=checkpoint.session.turn_index,
                    acting_character_id=checkpoint.session.player_character_id,
                    deliveries=[],
                    pause_reason="no_pending_render",
                )
            outcomes = await process_narrator_lanes(
                checkpoint,
                self.dispatcher,
                lane_ids=lane_ids,
            )
            checkpoint.session.turn_index += 1
            checkpoint_id = self.checkpoint_mgr.save(checkpoint)
            response = self._claim_response(
                checkpoint,
                checkpoint_id=checkpoint_id,
                acting_character_id=checkpoint.session.player_character_id,
                pause_reason=(
                    "narrator_delivery_failed"
                    if any(item.failed_pov_ids for item in outcomes)
                    else ""
                ),
                consumer_id=f"retry:{session_id}:{checkpoint.session.turn_index}",
            )
            self.checkpoint_mgr.save(checkpoint)
            self._schedule_autonomous(session_id)
            return response

    async def resolve_cat_ii(
        self,
        session_id: str,
        event_id: str,
    ) -> TurnResponse:
        lock = await self.session_locks.for_session(session_id)
        async with lock:
            checkpoint = self.checkpoint_mgr.load_latest(session_id)
            prepared = await prepare_autonomous_contest_resolutions(
                checkpoint,
                self.dispatcher,
            )
            prepared = [
                item for item in prepared if item.contest_event_id == event_id
            ]
            if not prepared:
                raise ValueError("contested action is not ready")
            result = await advance_story(checkpoint, self.dispatcher, prepared)
            self._remember_prepared(checkpoint, result.prepared_followups)
            checkpoint_id = self.checkpoint_mgr.save(checkpoint)
            response = self._claim_response(
                checkpoint,
                checkpoint_id=checkpoint_id,
                acting_character_id=checkpoint.session.player_character_id,
                pause_reason=result.pause_reason,
                consumer_id=f"contest:{session_id}:{checkpoint.session.turn_index}",
            )
            self.checkpoint_mgr.save(checkpoint)
            self._schedule_autonomous(session_id)
            return response

    async def defer_combat_reaction(
        self,
        *,
        session_id: str,
        character_id: str,
        event_id: str = "",
    ) -> TurnResponse:
        lock = await self.session_locks.for_session(session_id)
        async with lock:
            checkpoint = self.checkpoint_mgr.load_latest(session_id)
            obligation = checkpoint.session.action_obligations.get(character_id)
            if obligation is None or obligation.kind != "combat_reaction":
                raise ValueError("character has no pending combat reaction")
            if event_id and obligation.source_event_id != event_id:
                raise ValueError("combat reaction is stale")
            release_frontier_gates_for_pov_action(checkpoint, character_id)
            release_action_obligation(checkpoint, character_id)
            self._forget_prepared(session_id)
            from app.engine.dnd_story_adapter import (
                advance_pending_combat_if_unblocked,
            )

            advance_pending_combat_if_unblocked(checkpoint)
            checkpoint.session.turn_index += 1
            checkpoint_id = self.checkpoint_mgr.save(checkpoint)
            self._schedule_autonomous(session_id)
            return response_from_deliveries(
                session_id=session_id,
                checkpoint_id=checkpoint_id,
                turn_index=checkpoint.session.turn_index,
                acting_character_id=character_id,
                deliveries=[],
                pause_reason="combat_reaction_deferred",
            )

    async def submit_cat_ii_roll(
        self,
        *,
        session_id: str,
        event_id: str,
        roll_id: str,
        actor_id: str,
        user_id: str = "",
    ) -> TurnResponse:
        from app.engine.dnd_cat_ii import (
            complete_pending_player_roll,
            pending_player_rolls,
            roll_transaction_source,
        )

        lock = await self.session_locks.for_session(session_id)
        async with lock:
            checkpoint = self.checkpoint_mgr.load_latest(session_id)
            sync_checkpoint_runtime_models(checkpoint, self.client.config)
            release_frontier_gates_for_pov_action(checkpoint, actor_id)
            pending = pending_player_rolls(
                checkpoint,
                event_id=event_id,
                actor_id=actor_id,
            )
            if roll_id not in {item.roll_id for item in pending}:
                raise ValueError("roll is no longer pending for this character")
            complete_pending_player_roll(
                checkpoint,
                event_id=event_id,
                roll_id=roll_id,
                completed_by_user_id=user_id,
            )
            if pending_player_rolls(checkpoint, event_id=event_id):
                checkpoint.session.turn_index += 1
                return self._save_claim_and_schedule(
                    checkpoint,
                    acting_character_id=actor_id,
                    pause_reason="cat_ii_pending_rolls",
                )
            if roll_transaction_source(checkpoint, event_id) == "combat":
                result, output_actor_id = await self._continue_dnd_combat_transaction(
                    checkpoint,
                    event_id=event_id,
                    player_input=True,
                )
            else:
                prepared = await prepare_autonomous_contest_resolutions(
                    checkpoint,
                    self.dispatcher,
                )
                prepared = [
                    item for item in prepared if item.contest_event_id == event_id
                ]
                if not prepared:
                    raise ValueError("contested roll transaction is no longer active")
                result = await advance_story(checkpoint, self.dispatcher, prepared)
                output_actor_id = actor_id
            return self._save_claim_and_schedule(
                checkpoint,
                acting_character_id=output_actor_id,
                pause_reason=result.pause_reason,
                result=result,
            )

    async def continue_cat_ii_after_roll(
        self,
        *,
        session_id: str,
        event_id: str,
        actor_id: str = "",
    ) -> TurnResponse:
        from app.engine.dnd_cat_ii import pending_player_rolls, roll_transaction_source

        lock = await self.session_locks.for_session(session_id)
        async with lock:
            checkpoint = self.checkpoint_mgr.load_latest(session_id)
            if pending_player_rolls(checkpoint, event_id=event_id):
                raise ValueError("contested action still has pending player rolls")
            if roll_transaction_source(checkpoint, event_id) == "combat":
                result, output_actor_id = await self._continue_dnd_combat_transaction(
                    checkpoint,
                    event_id=event_id,
                    player_input=False,
                )
            else:
                prepared = await prepare_autonomous_contest_resolutions(
                    checkpoint,
                    self.dispatcher,
                )
                prepared = [
                    item for item in prepared if item.contest_event_id == event_id
                ]
                if not prepared:
                    raise ValueError("contested action is not ready")
                result = await advance_story(checkpoint, self.dispatcher, prepared)
                output_actor_id = actor_id or checkpoint.session.player_character_id
            return self._save_claim_and_schedule(
                checkpoint,
                acting_character_id=output_actor_id,
                pause_reason=result.pause_reason,
                result=result,
            )

    def _claim_response(
        self,
        checkpoint: CheckpointFile,
        *,
        checkpoint_id: str,
        acting_character_id: str,
        pause_reason: str,
        consumer_id: str,
    ) -> TurnResponse:
        deliveries = [
            entry
            for pov_id in checkpoint.session.character_bindings
            for entry in claim_deliveries(
                checkpoint,
                pov_character_id=pov_id,
                consumer_id=consumer_id,
            )
        ]
        return response_from_deliveries(
            session_id=checkpoint.session.session_id,
            checkpoint_id=checkpoint_id,
            turn_index=checkpoint.session.turn_index,
            acting_character_id=acting_character_id,
            deliveries=deliveries,
            pause_reason=pause_reason,
        )

    def _cancel_uncommitted_autonomous(self, session_id: str) -> None:
        task = self._autonomous_tasks.get(session_id)
        if (
            task is not None
            and not task.done()
            and self._autonomous_phase.get(session_id) in {"preparing", "routing"}
        ):
            task.cancel()
            self._autonomous_tasks.pop(session_id, None)
            self._autonomous_phase.pop(session_id, None)

    def _schedule_autonomous(self, session_id: str) -> None:
        task = self._autonomous_tasks.get(session_id)
        if task is not None and not task.done():
            return
        self._autonomous_tasks[session_id] = asyncio.create_task(
            self._autonomous_worker(session_id),
            name=f"story-autonomous:{session_id}",
        )

    def schedule_autonomous(self, session_id: str) -> None:
        self._schedule_autonomous(session_id)

    def cancel_autonomous(self, session_id: str) -> None:
        task = self._autonomous_tasks.pop(session_id, None)
        self._autonomous_phase.pop(session_id, None)
        self._forget_prepared(session_id)
        if task is not None and not task.done():
            task.cancel()

    async def shutdown(self) -> None:
        """Cancel and await every session worker before provider shutdown."""

        tasks = list(self._autonomous_tasks.values())
        self._autonomous_tasks.clear()
        self._autonomous_phase.clear()
        self._prepared_autonomous.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _autonomous_combat_candidate(
        checkpoint: CheckpointFile,
    ) -> tuple[str, str] | None:
        from app.engine.dnd_combat_access import combatant_for_character
        from app.engine.dnd_story_adapter import current_combat_actor_id
        from app.schemas.characters import is_player_authored_slot

        combat = checkpoint.session.active_combat
        if combat is None or any(
            obligation.kind in {"combat_reaction", "cat_ii_roll"}
            for obligation in checkpoint.session.action_obligations.values()
        ):
            return None
        actor_id = current_combat_actor_id(checkpoint)
        combatant = combatant_for_character(combat, actor_id)
        if combatant is None:
            return None
        pending = combatant.pending_initiating_action.strip()
        if pending:
            return actor_id, pending
        if actor_id in checkpoint.session.character_bindings:
            return None
        character = next(
            (
                item
                for item in checkpoint.characters
                if item.character_id == actor_id
            ),
            None,
        )
        if (
            character is None
            or character.status != CharacterStatus.active
            or is_player_authored_slot(character)
        ):
            return None
        return actor_id, ""

    async def wait_for_autonomous(self, session_id: str) -> None:
        task = self._autonomous_tasks.get(session_id)
        if task is not None:
            await asyncio.shield(task)

    async def _autonomous_worker(self, session_id: str) -> None:
        try:
            while True:
                lock = await self.session_locks.for_session(session_id)
                async with lock:
                    live = self.checkpoint_mgr.load_latest(session_id)
                    limit = max(
                        1,
                        live.session.config.settings.max_router_batches_without_player_input,
                    )
                    if live.session.autonomous_router_batches_since_player >= limit:
                        return
                    combat_candidate = self._autonomous_combat_candidate(live)
                    if (
                        combat_candidate is None
                        and not live.session.open_cat_ii_events
                        and not ready_frontier_turns(live)
                    ):
                        return
                    cached_prepared = self._cached_prepared(live)
                    snapshot = immutable_checkpoint(live)
                    fingerprint = _checkpoint_fingerprint(live)

                self._autonomous_phase[session_id] = "preparing"
                if combat_candidate is not None:
                    actor_id, intention = combat_candidate
                    draft = None
                    if not intention:
                        draft = await self.dispatcher.draft_character_turn(
                            ckpt=immutable_checkpoint(snapshot),
                            character_id=actor_id,
                            local_context="It is this character's initiative turn.",
                        )
                        intention = (
                            draft.output.public_text
                            if not draft.output.is_silence
                            else "(defer)"
                        )
                    self._autonomous_phase[session_id] = "routing"
                    async with lock:
                        live = self.checkpoint_mgr.load_latest(session_id)
                        if _checkpoint_fingerprint(live) != fingerprint:
                            continue
                    result = await self._advance_dnd_combat_action(
                        snapshot,
                        actor_id=actor_id,
                        intention=intention,
                        is_reaction=False,
                        player_input=False,
                        character_draft=draft,
                    )
                    async with lock:
                        live = self.checkpoint_mgr.load_latest(session_id)
                        if _checkpoint_fingerprint(live) != fingerprint:
                            continue
                        self._remember_prepared(
                            snapshot,
                            result.prepared_followups,
                        )
                        self.checkpoint_mgr.save(snapshot)
                    continue
                prepared = await prepare_autonomous_contest_resolutions(
                    snapshot,
                    self.dispatcher,
                )
                if not prepared:
                    prepared = await prepare_ready_frontier_batch(
                        snapshot,
                        self.dispatcher,
                        preprepared=cached_prepared,
                    )
                if not prepared:
                    return

                self._autonomous_phase[session_id] = "routing"
                async with lock:
                    live = self.checkpoint_mgr.load_latest(session_id)
                    if _checkpoint_fingerprint(live) != fingerprint:
                        continue
                result = await advance_story(snapshot, self.dispatcher, prepared)
                async with lock:
                    live = self.checkpoint_mgr.load_latest(session_id)
                    if _checkpoint_fingerprint(live) != fingerprint:
                        continue
                    self._remember_prepared(snapshot, result.prepared_followups)
                    self.checkpoint_mgr.save(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("autonomous story worker failed for %s", session_id)
            await self._record_autonomous_error(session_id, exc)
        finally:
            if self._autonomous_tasks.get(session_id) is asyncio.current_task():
                self._autonomous_phase.pop(session_id, None)

    async def _record_autonomous_error(
        self,
        session_id: str,
        error: Exception,
    ) -> None:
        from app.engine.delivery_outbox import enqueue_delivery

        lock = await self.session_locks.for_session(session_id)
        async with lock:
            checkpoint = self.checkpoint_mgr.load_latest(session_id)
            message = (
                "Automatic story advancement paused after an error. Your saved "
                "story and pending character turns were preserved. "
                f"Owner detail: {type(error).__name__}: {error}"
            )[:1200]
            for pov_id in checkpoint.session.character_bindings:
                enqueue_delivery(
                    checkpoint,
                    pov_character_id=pov_id,
                    source_event_ids=[],
                    highest_event_sequence=len(checkpoint.canonical_events) - 1,
                    payload=DeliveryPayload(
                        prose="",
                        visual_novel=None,
                        asset_reveals=[],
                        reaction_prompt_event_id="",
                        loot_offer_ids=[],
                        commitment_revision_ids=[],
                        dice_rolls=[],
                        experience_awards=[],
                        owner_error=message,
                    ),
                )
            checkpoint.session.turn_index += 1
            self.checkpoint_mgr.save(checkpoint)


async def advance_pending_combat_if_unblocked(*args: Any, **kwargs: Any) -> None:
    """D&D adapter hook; initiative will enqueue its own sourced world turn."""

    return None
