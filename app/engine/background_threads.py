"""Selection and isolation contracts for router-authored background threads.

A background thread is one bounded autonomous character turn outside the
current focal event.  The router owns the semantic grouping: runtime never
infers it from ``CharacterRecord.location`` or persists a parallel scene
ledger.  Runtime only supplies the safe set of autonomous initiators, validates
the router's picks, and proves that independently prepared branches can merge.
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


MAX_CONCURRENT_BACKGROUND_THREADS = 4


class BackgroundThreadContractError(ValueError):
    """A router background-thread selection is unsafe or incomplete."""


@dataclass(frozen=True, slots=True)
class BackgroundThreadRequest:
    """Volatile authority for one independently prepared semantic thread."""

    actor_id: str
    participant_ids: tuple[str, ...]
    canonical_at_s: int


def _player_character_ids(checkpoint: CheckpointFile) -> set[str]:
    ids = set(checkpoint.session.character_bindings)
    if checkpoint.session.player_character_id:
        ids.add(checkpoint.session.player_character_id)
    return ids


def _unavailable_character_ids(checkpoint: CheckpointFile) -> set[str]:
    session = checkpoint.session
    ids = {
        *_player_character_ids(checkpoint),
        *session.active_act_slots,
        *session.pending_commitment_revisions,
    }
    for event in session.open_cat_ii_events:
        if event.initiator_id:
            ids.add(event.initiator_id)
        ids.update(event.required_responders)

    active_combat = session.active_combat
    if active_combat is not None:
        from app.engine.dnd_combat_access import (
            combatant_character_id,
            combatants,
        )

        ids.update(
            character_id
            for combatant in combatants(active_combat)
            if (character_id := combatant_character_id(combatant))
        )
    return ids


def _background_participant_records(
    checkpoint: CheckpointFile,
) -> list[CharacterRecord]:
    unavailable = _unavailable_character_ids(checkpoint)
    return [
        character
        for character in checkpoint.characters
        if character.status == CharacterStatus.active
        and character.character_id not in unavailable
        and character.actor is not None
        and not is_player_authored_slot(character)
        and not is_non_social_hazard(character)
    ]


def background_thread_candidate_ids(
    checkpoint: CheckpointFile,
) -> list[str]:
    """Return safe autonomous initiators in stable roster order.

    This is deliberately not a scene-discovery function.  It knows only which
    active records runtime may call as agents; the router decides which of them
    are semantically outside the current focal event and which belong together.
    """

    return [
        character.character_id
        for character in _background_participant_records(checkpoint)
        if character.actor is not None and character.actor.may_act_offstage
    ]


def format_background_thread_selection_contract(
    checkpoint: CheckpointFile,
    candidate_ids: list[str],
    *,
    max_threads: int = MAX_CONCURRENT_BACKGROUND_THREADS,
) -> str:
    """Render the fresh-turn selection contract in the volatile user tail."""

    names = {
        character.character_id: character.name
        for character in checkpoint.characters
    }
    candidates = ", ".join(
        f"{character_id} ({names.get(character_id, character_id)})"
        for character_id in candidate_ids
    ) or "none"
    return "\n".join([
        "<background_thread_selection>",
        "Eligible autonomous initiators: " + candidates,
        (
            "The observers, acting character, responders, and directly depicted "
            "characters of your current event form its focal thread. If any "
            "eligible initiator remains outside that thread, select at least one "
            "background thread now. Prefer the actor with the strongest immediate "
            "pressure from the newest events they personally witnessed."
        ),
        (
            "A witness left behind by a transition is part of that transition, "
            "not a background actor in the same output. On the first later event "
            "they do not witness, their last witnessed transition remains their "
            "context and their separate thread must be considered."
        ),
        (
            "For each selection, put its submitting actor in actor_id and list "
            "that actor plus only the other active autonomous characters who are "
            "semantically available in the same off-screen thread in "
            "participant_ids. A lone actor is valid. Separate selections must not "
            "share participants. Select at most "
            f"{max_threads}. These ids do not observe the current event and "
            "receive none of its facts."
        ),
        "If every eligible initiator belongs to the focal thread, emit background_threads=[].",
        "</background_thread_selection>",
    ])


def _focal_character_ids(
    result: EventRouterOutput,
    *,
    actor_id: str,
) -> set[str]:
    ids = {
        *([actor_id] if actor_id else []),
        *result.required_responders,
        *(observer.character_id for observer in result.observers),
        *result.dormant,
        *result.cull,
        *(signal.character_id for signal in result.activate),
        *(signal.character_id for signal in result.location_updates),
        *(
            character_id
            for fact in result.canonical_event.observable_facts
            for character_id in fact.visual_subject_ids
        ),
    }
    for signal in result.commitment_resolutions:
        ids.update(signal.actor_ids)
    for signal in result.commitment_interrupts:
        ids.update(signal.actor_ids)
    return {character_id for character_id in ids if character_id}


def validate_background_thread_selection(
    checkpoint: CheckpointFile,
    *,
    result: EventRouterOutput,
    actor_id: str,
    candidate_ids: list[str] | None = None,
    require_when_offscreen: bool,
    excluded_participant_ids: set[str] | None = None,
    max_threads: int = MAX_CONCURRENT_BACKGROUND_THREADS,
) -> None:
    """Validate router-owned semantic picks without inferring any location."""

    if not require_when_offscreen:
        if result.background_threads:
            raise BackgroundThreadContractError(
                "background_threads must be empty outside the fresh player-turn "
                "selection contract"
            )
        return

    excluded = set(excluded_participant_ids or ())
    raw_candidates = (
        candidate_ids
        if candidate_ids is not None
        else background_thread_candidate_ids(checkpoint)
    )
    candidates = [
        character_id
        for character_id in dict.fromkeys(raw_candidates)
        if character_id not in excluded
    ]
    offscreen_candidates = set(candidates) - _focal_character_ids(
        result,
        actor_id=actor_id,
    )
    picks = result.background_threads
    if offscreen_candidates and not picks:
        raise BackgroundThreadContractError(
            "at least one background_threads selection is required because "
            "eligible autonomous actors remain outside the focal event: "
            + ", ".join(
                character_id
                for character_id in candidates
                if character_id in offscreen_candidates
            )
        )
    if len(picks) > max_threads:
        raise BackgroundThreadContractError(
            "background_threads exceeds the remaining concurrency cap"
        )

    available_participants = {
        character.character_id
        for character in _background_participant_records(checkpoint)
    }
    used_participants: set[str] = set()
    for pick in picks:
        participants = set(pick.participant_ids)
        if participants & excluded:
            raise BackgroundThreadContractError(
                "background thread reuses a participant already selected this "
                "beat: " + ", ".join(sorted(participants & excluded))
            )
        if pick.actor_id not in offscreen_candidates:
            raise BackgroundThreadContractError(
                "background thread actor must be an eligible initiator outside "
                f"the focal event: {pick.actor_id!r}"
            )
        if participants - available_participants:
            raise BackgroundThreadContractError(
                "background thread contains unavailable participant ids: "
                + ", ".join(sorted(participants - available_participants))
            )
        focal_overlap = participants & _focal_character_ids(
            result,
            actor_id=actor_id,
        )
        if focal_overlap:
            raise BackgroundThreadContractError(
                "background thread overlaps the focal event: "
                + ", ".join(sorted(focal_overlap))
            )
        overlap = participants & used_participants
        if overlap:
            raise BackgroundThreadContractError(
                "parallel background threads share participant ids: "
                + ", ".join(sorted(overlap))
            )
        used_participants.update(participants)


def background_thread_requests(
    checkpoint: CheckpointFile,
    *,
    result: EventRouterOutput,
    actor_id: str,
    excluded_participant_ids: set[str] | None = None,
    max_threads: int = MAX_CONCURRENT_BACKGROUND_THREADS,
) -> list[BackgroundThreadRequest]:
    """Revalidate live picks and anchor them at one common causal instant."""

    # The selection requirement was already checked against the pre-event
    # roster used to build the router prompt.  Committing the focal event may
    # activate new characters; those arrivals were not eligible candidates for
    # this decision and must not retroactively turn an honest empty selection
    # into a contract failure.
    if not result.background_threads:
        return []
    candidates = background_thread_candidate_ids(checkpoint)
    validate_background_thread_selection(
        checkpoint,
        result=result,
        actor_id=actor_id,
        candidate_ids=candidates,
        require_when_offscreen=True,
        excluded_participant_ids=excluded_participant_ids,
        max_threads=max_threads,
    )
    canonical_at_s = max(
        0,
        int(checkpoint.session.leading_at_s),
        int(result.effective_at_s + result.duration_s),
    )
    return [
        BackgroundThreadRequest(
            actor_id=pick.actor_id,
            participant_ids=tuple(pick.participant_ids),
            canonical_at_s=canonical_at_s,
        )
        for pick in result.background_threads
    ]


def anchor_background_thread_result(
    result: EventRouterOutput,
    *,
    request: BackgroundThreadRequest,
) -> None:
    """Place independently prepared work at its shared source instant."""

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


def validate_background_thread_result(
    checkpoint: CheckpointFile,
    *,
    request: BackgroundThreadRequest,
    result: EventRouterOutput,
) -> None:
    """Prove that one routed background move has no cross-thread authority."""

    by_id = {
        character.character_id: character for character in checkpoint.characters
    }
    actor = by_id.get(request.actor_id)
    if (
        actor is None
        or actor.status != CharacterStatus.active
        or actor.actor is None
        or not actor.actor.may_act_offstage
        or request.actor_id in _unavailable_character_ids(checkpoint)
    ):
        raise ValueError("background thread actor is no longer eligible")

    participants = set(request.participant_ids)
    if not participants or request.actor_id not in participants:
        raise ValueError("background thread requires its actor as a participant")
    available_participants = {
        character.character_id
        for character in _background_participant_records(checkpoint)
    }
    if participants - available_participants:
        raise ValueError(
            "background thread participants are no longer autonomous and available"
        )

    if result.effective_at_s != request.canonical_at_s or result.duration_s != 0:
        raise ValueError("background thread must remain at its common canonical instant")
    if not result.canonical_event.world_adjudication.feasible:
        raise ValueError("background thread must produce a feasible observable choice")
    if not result.canonical_event.observable_facts:
        raise ValueError("background thread must produce at least one observable fact")
    if (
        result.requires_responders
        or result.required_responders
        or result.next_output_character_ids
        or result.perception_enrichment_character_ids
        or result.background_threads
        or result.event_kind in {
            "beat_continues",
            "cat_ii_open",
            "ruleset_resolution",
            "ruleset_cat_ii_suppressed",
        }
    ):
        raise ValueError("background thread must close without another frontier")
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
        raise ValueError(
            "background thread cannot mutate shared lifecycle, location, or rules state"
        )

    observer_ids = {observer.character_id for observer in result.observers}
    if request.actor_id not in observer_ids:
        raise ValueError("background thread actor must observe the canonical result")
    if any(
        observer.character_id not in participants
        or observer.routing_role != "observe_only"
        for observer in result.observers
    ):
        raise ValueError(
            "background thread observers must be participants and observe-only"
        )

    structured_ids = {
        *observer_ids,
        *(
            character_id
            for fact in result.canonical_event.observable_facts
            for character_id in (*fact.visible_to, *fact.visual_subject_ids)
        ),
    }
    if structured_ids - participants:
        raise ValueError("background thread cannot observe or depict another thread")

    if result.commitment_open.present:
        commitment_ids = set(result.commitment_open.actor_ids)
        if (
            request.actor_id not in commitment_ids
            or not commitment_ids
            or commitment_ids - participants
            or result.commitment_open.location_label
        ):
            raise ValueError(
                "background commitment must stay inside its semantic participants "
                "and carry no location authority"
            )
