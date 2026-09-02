"""Rules-neutral discovery and validation for autonomous scene ticks.

A scene tick is one bounded, self-directed character turn in a location that
has no human-controlled viewpoint present and no foreground frontier using it.
The turn loop may prepare different locations concurrently, but this module
keeps each request scoped tightly enough that their durable results can be
merged in a deterministic order.

Location labels are deliberately opaque.  They are the smallest canonical
scene boundary the current checkpoint owns; an empty label cannot prove
independence and is therefore never ticked.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    is_non_social_hazard,
    is_player_authored_slot,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput


@dataclass(frozen=True, slots=True)
class SceneTickRequest:
    """Volatile authority for one autonomous turn in one isolated scene."""

    location: str
    actor_id: str
    participant_ids: tuple[str, ...]
    canonical_at_s: int
    adapter_context: object | None = None


def _player_character_ids(checkpoint: CheckpointFile) -> set[str]:
    ids = set(checkpoint.session.character_bindings)
    if checkpoint.session.player_character_id:
        ids.add(checkpoint.session.player_character_id)
    return ids


def _normalized_location(character: CharacterRecord) -> str:
    return " ".join(str(character.location or "").split()).strip()


def _candidate_sort_key(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
    *,
    roster_index: int,
) -> tuple[int, int, int, str]:
    """Put the least-used actor first without adding a second cadence ledger."""

    conversation_turns = len(
        checkpoint.character_conversations.get(character.character_id, ())
    )
    last_turn = character.last_agent_turn_at_s
    return (
        conversation_turns,
        last_turn if last_turn is not None else -1,
        roster_index,
        character.character_id,
    )


def discover_scene_tick_requests(
    checkpoint: CheckpointFile,
    *,
    blocked_character_ids: set[str] | None = None,
    max_scenes: int,
) -> list[SceneTickRequest]:
    """Select at most one autonomous initiator per independent scene.

    Human locations and locations containing a current foreground participant
    are narrator-active for this boundary.  Open contests and active combat
    retain their existing sequential rules paths, so neither admits scene
    ticks.  Character conversation depth and last committed agent time provide
    a canonical fairness signal without persisting a parallel scene clock.
    """

    if max_scenes <= 0:
        return []
    session = checkpoint.session
    if session.open_cat_ii_events or session.active_combat is not None:
        return []

    blocked_character_ids = set(blocked_character_ids or ())
    player_ids = _player_character_ids(checkpoint)
    by_id = {
        character.character_id: character for character in checkpoint.characters
    }
    blocked_locations = {
        location
        for character_id in {*player_ids, *blocked_character_ids}
        if (character := by_id.get(character_id)) is not None
        if (location := _normalized_location(character))
    }
    unavailable_ids = {
        *player_ids,
        *session.active_act_slots,
        *session.pending_commitment_revisions,
        *(
            character_id
            for commitment in session.open_commitments
            for character_id in commitment.actor_ids
        ),
    }

    participants_by_location: dict[str, list[CharacterRecord]] = {}
    candidates_by_location: dict[str, list[tuple[int, CharacterRecord]]] = {}
    for roster_index, character in enumerate(checkpoint.characters):
        location = _normalized_location(character)
        if (
            not location
            or location in blocked_locations
            or character.status != CharacterStatus.active
            or character.character_id in player_ids
            or is_player_authored_slot(character)
            or is_non_social_hazard(character)
        ):
            continue
        participants_by_location.setdefault(location, []).append(character)
        if (
            character.character_id in unavailable_ids
            or character.actor is None
            or not character.actor.may_act_offstage
        ):
            continue
        candidates_by_location.setdefault(location, []).append(
            (roster_index, character)
        )

    one_star_context: object | None = None
    one_star_enabled = False
    from app.engine.one_star_adapter import (
        is_one_star_checkpoint,
        one_star_scene_tick_context,
    )

    if is_one_star_checkpoint(checkpoint):
        one_star_enabled = True
        one_star_context = one_star_scene_tick_context(checkpoint)
        if one_star_context is None:
            return []

    requests: list[tuple[tuple[int, int, int, str], SceneTickRequest]] = []
    for location, indexed_candidates in candidates_by_location.items():
        participants = participants_by_location.get(location, [])
        adapter_context: object | None = None
        if one_star_enabled:
            # One-Star owns which of its embodied scenes are disjoint from the
            # live mission.  The generic scheduler still owns cadence,
            # isolation, and merge.
            context = one_star_context
            allowed_ids = set(getattr(context, "hero_ids", ()))
            if location != getattr(context, "lobby_location", ""):
                continue
            indexed_candidates = [
                pair
                for pair in indexed_candidates
                if pair[1].character_id in allowed_ids
            ]
            participants = [
                character
                for character in participants
                if character.character_id in allowed_ids
            ]
            if not indexed_candidates:
                continue
            adapter_context = context

        indexed_candidates.sort(
            key=lambda pair: _candidate_sort_key(
                checkpoint,
                pair[1],
                roster_index=pair[0],
            )
        )
        selected_index, selected = indexed_candidates[0]
        fairness_key = _candidate_sort_key(
            checkpoint,
            selected,
            roster_index=selected_index,
        )
        participant_ids = tuple(
            character.character_id for character in participants
        )
        requests.append((
            (*fairness_key[:3], location),
            SceneTickRequest(
                location=location,
                actor_id=selected.character_id,
                participant_ids=participant_ids,
                canonical_at_s=max(0, int(session.leading_at_s)),
                adapter_context=adapter_context,
            ),
        ))

    requests.sort(key=lambda pair: pair[0])
    return [request for _key, request in requests[:max_scenes]]


def anchor_scene_tick_result(
    result: EventRouterOutput,
    *,
    request: SceneTickRequest,
) -> None:
    """Place independent ticks at their common source instant."""

    result.effective_at_s = request.canonical_at_s


def _adapter_side_effects_present(result: EventRouterOutput) -> bool:
    if getattr(result, "state_updates", ()):
        return True
    if getattr(result, "interaction_mode", "narrative") != "narrative":
        return True
    if (
        getattr(result, "combatant_ids", ())
        or getattr(result, "combatant_spawns", ())
    ):
        return True
    loot_offer = getattr(result, "loot_offer", None)
    if loot_offer is not None and bool(getattr(loot_offer, "present", False)):
        return True
    battle_map_seed = getattr(result, "battle_map_seed", None)
    return bool(
        battle_map_seed is not None
        and getattr(battle_map_seed, "present", False)
    )


def validate_scene_tick_result(
    checkpoint: CheckpointFile,
    *,
    request: SceneTickRequest,
    result: EventRouterOutput,
) -> None:
    """Prove that one routed tick can merge without cross-scene authority."""

    by_id = {
        character.character_id: character for character in checkpoint.characters
    }
    actor = by_id.get(request.actor_id)
    if (
        actor is None
        or actor.status != CharacterStatus.active
        or _normalized_location(actor) != request.location
        or actor.actor is None
        or not actor.actor.may_act_offstage
    ):
        raise ValueError("scene tick actor is no longer eligible in this scene")
    player_ids = _player_character_ids(checkpoint)
    participants = set(request.participant_ids)
    if not participants or request.actor_id not in participants:
        raise ValueError("scene tick requires its actor in a non-empty scene")
    for character_id in participants:
        character = by_id.get(character_id)
        if (
            character is None
            or character.status != CharacterStatus.active
            or _normalized_location(character) != request.location
            or character_id in player_ids
            or is_player_authored_slot(character)
        ):
            raise ValueError("scene tick participants must remain local autonomous actors")

    if result.effective_at_s != request.canonical_at_s or result.duration_s != 0:
        raise ValueError("scene tick must remain at its common canonical instant")
    if not result.canonical_event.world_adjudication.feasible:
        raise ValueError("scene tick must produce a feasible observable choice")
    if not result.canonical_event.observable_facts:
        raise ValueError("scene tick must produce at least one observable fact")
    if (
        result.requires_responders
        or result.required_responders
        or result.next_output_character_ids
        or result.perception_enrichment_character_ids
        or result.event_kind in {
            "beat_continues",
            "cat_ii_open",
            "ruleset_resolution",
            "ruleset_cat_ii_suppressed",
        }
    ):
        raise ValueError("scene tick must close without a response frontier")
    if (
        result.spawn
        or result.activate
        or result.dormant
        or result.cull
        or result.location_updates
        or result.commitment_resolutions
        or result.commitment_interrupts
        or _adapter_side_effects_present(result)
    ):
        raise ValueError("scene tick cannot mutate shared lifecycle or rules state")

    observer_ids = {observer.character_id for observer in result.observers}
    if request.actor_id not in observer_ids:
        raise ValueError("scene tick actor must observe the canonical result")
    if any(
        observer.character_id not in participants
        or observer.routing_role != "observe_only"
        for observer in result.observers
    ):
        raise ValueError("scene tick observers must be local and observe-only")

    structured_ids = {
        *result.required_responders,
        *observer_ids,
        *(
            character_id
            for fact in result.canonical_event.observable_facts
            for character_id in (*fact.visible_to, *fact.visual_subject_ids)
        ),
    }
    if structured_ids - participants:
        raise ValueError("scene tick cannot observe or depict another scene")

    if result.commitment_open.present:
        commitment_ids = set(result.commitment_open.actor_ids)
        if (
            request.actor_id not in commitment_ids
            or not commitment_ids
            or commitment_ids - participants
            or result.commitment_open.location_label
            not in {"", request.location}
        ):
            raise ValueError(
                "scene tick commitment must belong to actors in this scene"
            )

    if request.adapter_context is not None:
        from app.engine.one_star_adapter import (
            OneStarSceneTickContext,
            validate_one_star_scene_tick_result,
        )

        if not isinstance(request.adapter_context, OneStarSceneTickContext):
            raise ValueError("unknown scene tick adapter context")
        validate_one_star_scene_tick_result(
            checkpoint,
            context=request.adapter_context,
            request=request,
            result=result,
        )


def scene_tick_blocked_character_ids(
    result: EventRouterOutput,
    *,
    actor_id: str,
) -> set[str]:
    """Characters whose locations belong to the current causal frontier."""

    return {
        *([actor_id] if actor_id else []),
        *result.required_responders,
        *result.next_output_character_ids,
        *(
            character_id
            for fact in result.canonical_event.observable_facts
            for character_id in fact.visual_subject_ids
        ),
        *(update.character_id for update in result.location_updates),
    }
