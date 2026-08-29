"""v11-A3a: LLMDispatcher — concrete Dispatcher binding turn_loop to the
real router / agent / narrator modules.

`turn_loop.run_beat` talks to the world exclusively through the
`Dispatcher` Protocol. This module provides the production binding:
the three async methods call through to the `event_router` prompt
template, CharacterAgent.turn(), and
narrator.compose_pov_render() (per-POV entry point), constructing
their user-message blocks through the shared helpers in
`turn_loop_contracts` so prompt-code contracts stay in lockstep.

The legacy `EventRouter` engine class that wrapped this same prompt
template was murdered in v11-r7j; this dispatcher is the only
production caller of the `event_router` template now.

Tests should prefer passing fakes into `run_beat` directly; this
class is what the orchestrator constructs at wire-up time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy

from pydantic import ValidationError

from app.engine import narrator as narrator_module
from app.engine.character_agent import CharacterAgent, CharacterAgentTurnDraft
from app.engine.character_manager import CharacterManager
from app.engine.context_builder import (
    build_dnd_character_equipment_sentence,
    build_dnd_character_identity_sentence,
    build_hidden_facts,
    build_setting_summary,
    build_world_rules,
    collect_player_ids,
    resolve_acting_character,
)
from app.engine.prompt_manager import PromptManager
from app.engine.dnd_cat_ii import (
    DND5E_BASIC_RULESET_ID,
    DndCatIIRollsPending,
    DndCatIIResolver,
    dnd_cat_ii_router_enabled,
    dnd_combat_manager_enabled,
)
from app.engine.dnd_combat_resolution import DndCombatResolver
from app.engine.turn_loop_contracts import (
    AuthoritativeContributionRequest,
    AuthoritativeResultPlan,
    format_actor_submission,
    format_authoritative_result_block,
    format_cat_ii_resolution_block,
    format_router_continuation_block,
)
from app.llm.client import LLMClient
from app.schemas.characters import CharacterRecord, is_non_social_hazard
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import (
    ClosedEventRouterOutput,
    DndEventRouterOutput,
    EventRouterOutput,
    LocationUpdateSignal,
)
from app.schemas.one_star import (
    ClosedOneStarEventRouterOutput,
    OneStarAccountEnvelope,
    OneStarEventRouterOutput,
    ONE_STAR_RULESET_ID,
    OneStarStateUpdate,
    OneStarStateUpdateList,
    OneStarTransaction,
)
from app.schemas.narrator import NarratorOutput, TranscriptEntry
from app.schemas.state import OpenCatIIEvent, RenderBufferEntry

logger = logging.getLogger(__name__)


EVENT_ROUTER_MAX_TOKENS = 8000


def _session_ruleset_id(ckpt: CheckpointFile) -> str:
    return str(getattr(ckpt.session.config.settings, "ruleset_id", "") or "")


def _dnd_fresh_router_enabled(
    ckpt: CheckpointFile,
    cat_ii_event: OpenCatIIEvent | None,
) -> bool:
    return cat_ii_event is None and _session_ruleset_id(ckpt) == DND5E_BASIC_RULESET_ID


def _one_star_router_enabled(ckpt: CheckpointFile) -> bool:
    return _session_ruleset_id(ckpt) == ONE_STAR_RULESET_ID


_ONE_STAR_LOBBY_MANAGEMENT_OPERATIONS = frozenset(
    {
        "catalogue_apply",
        "summon",
        "inventory_delta",
        "equipment_move",
        "pending_open",
        "pending_resolve",
        "pending_cancel",
    }
)

_ONE_STAR_COMPACT_UPDATE_AUTHORITY = (
    'Compact state scalar authority: hero_delta, mission_update, '
    'pending_resolve, pending_cancel, and tutorial_delivery require value="". '
    "For pending_resolve and pending_cancel, put the pending operation id only "
    "in target_id and use details=[]. The mission_start update requires one or "
    "more unique counter.<nonempty_id>=<current>/<target> details."
)


def _append_one_star_system_consequences(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    consequences: Sequence[object],
) -> None:
    """Add adapter-authored facts before the event's ordinary fan-out.

    These facts describe deterministic state changes the router could not have
    authored, so they belong in the same canonical event rather than a second
    ledger snapshot on the next turn.  Their recipients receive a clear
    mediated observation without acquiring a response-routing role.
    """

    if not consequences:
        return

    from app.schemas.events import ObservableFact
    from app.schemas.event_router import ObserverEntry

    known_character_ids = {
        character.character_id
        for character in ckpt.characters
        if character.character_id
    }
    original_observer_ids = [
        observer.character_id for observer in result.observers if observer.character_id
    ]
    observer_ids = set(original_observer_ids)
    at_offset_s = max(0, int(result.duration_s))

    normalized: list[tuple[str, tuple[str, ...]]] = []
    for consequence in consequences:
        text = str(getattr(consequence, "text", "") or "").strip()
        recipients = tuple(
            dict.fromkeys(
                str(character_id).strip()
                for character_id in getattr(
                    consequence,
                    "recipient_character_ids",
                    (),
                )
                if str(character_id).strip()
            )
        )
        if not text or not recipients:
            raise ValueError(
                "One-Star adapter consequence requires text and recipients"
            )
        unknown = set(recipients) - known_character_ids
        if unknown:
            raise ValueError(
                "One-Star adapter consequence names unknown recipients: "
                + ", ".join(sorted(unknown))
            )
        normalized.append((text, recipients))

    missing_recipient_ids = {
        character_id
        for _text, recipients in normalized
        for character_id in recipients
        if character_id not in observer_ids
    }
    if missing_recipient_ids:
        # ``all_observers`` means the router's original observer envelope, not
        # every recipient of a later engine-origin private fact. Preserve that
        # envelope explicitly before adding mediated System recipients, or a
        # guide/status recipient would accidentally inherit unrelated scene
        # facts from an event they did not witness.
        if not original_observer_ids and any(
            fact.audience == "all_observers"
            for fact in result.canonical_event.observable_facts
        ):
            raise ValueError(
                "cannot add scoped One-Star recipients to an observerless public event"
            )
        for fact in result.canonical_event.observable_facts:
            if fact.audience != "all_observers":
                continue
            fact.audience = "only"
            fact.visible_to = list(original_observer_ids)
        for character_id in sorted(missing_recipient_ids):
            result.observers.append(
                ObserverEntry(
                    character_id=character_id,
                    observation_level="d",
                    routing_role="observe_only",
                )
            )
            observer_ids.add(character_id)

    existing_facts = {
        (
            fact.text,
            fact.audience,
            tuple(fact.visible_to),
            fact.at_offset_s,
            fact.duration_s,
        )
        for fact in result.canonical_event.observable_facts
    }
    for text, recipients in normalized:
        fact = ObservableFact.only(
            text,
            recipients,
            at_offset_s=at_offset_s,
        )
        fact_key = (
            fact.text,
            fact.audience,
            tuple(fact.visible_to),
            fact.at_offset_s,
            fact.duration_s,
        )
        if fact_key in existing_facts:
            continue
        result.canonical_event.observable_facts.append(fact)
        existing_facts.add(fact_key)


def _one_star_transaction_for_result(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
) -> OneStarTransaction | None:
    if not isinstance(
        result,
        (OneStarEventRouterOutput, ClosedOneStarEventRouterOutput),
    ):
        return None
    from app.engine.one_star_adapter import (
        OneStarTransactionError,
        one_star_state_updates_to_transaction,
    )

    try:
        return one_star_state_updates_to_transaction(
            ckpt,
            result.state_updates,
            canonical_at_s=result.effective_at_s + result.duration_s,
        )
    except ValidationError as exc:
        errors = exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        specific_issue = (
            str(errors[0].get("msg", ""))
            .removeprefix("Value error, ")
            .strip()
            if errors
            else ""
        )
        detail = f": {specific_issue}" if specific_issue else ""
        raise OneStarTransactionError(
            "a compact One-Star state update violates its typed value bounds"
            + detail
        ) from exc


def _one_star_summon_update_signature(
    updates: list[OneStarStateUpdate],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Fields a state-only repair may not change after lifecycle is staged."""
    return tuple(
        (update.target_id, update.value, tuple(update.details))
        for update in updates
        if update.kind == "summon"
    )


def _one_star_pending_resolution_count(
    updates: list[OneStarStateUpdate],
) -> int:
    """Return irreversible pending resolutions a repair may not erase."""

    return sum(update.kind == "pending_resolve" for update in updates)


def _validate_one_star_cat_ii_transaction(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
) -> None:
    """Keep a Cat II opening reversible until every intention is collected."""

    if not result.requires_responders:
        return
    transaction = _one_star_transaction_for_result(ckpt, result)
    if transaction is None or not transaction.present:
        return
    operations = transaction.operations
    if (
        len(operations) != 1
        or getattr(operations[0], "operation", "") != "pending_open"
    ):
        raise ValueError(
            "A One-Star Cat II opening may record only one pending_open "
            "selection; mechanical and irreversible operations must wait "
            "for the resolved event"
        )


def _validate_one_star_pending_operation_shapes(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
) -> None:
    """Reject kind-dependent pending fields before response routing begins."""

    transaction = _one_star_transaction_for_result(ckpt, result)
    if transaction is None or not transaction.present:
        return
    from app.engine.one_star_adapter import (
        validate_one_star_pending_operation_shape,
    )

    for operation in transaction.operations:
        if getattr(operation, "operation", "") == "pending_open":
            validate_one_star_pending_operation_shape(operation.pending)


def _one_star_initial_guided_roster(
    ckpt: CheckpointFile,
) -> tuple[str, OneStarAccountEnvelope, tuple[str, ...]] | None:
    """Return the seed-authorized initial roster in its acquisition order."""

    if not _one_star_router_enabled(ckpt) or not ckpt.canonical_events:
        return None

    from app.engine.one_star_adapter import (
        find_one_star_account_owner,
        load_one_star_account,
        load_one_star_hero,
    )
    from app.schemas.one_star import OneStarOpeningRosterSummonPool

    if find_one_star_account_owner(ckpt.characters) is None:
        return None
    owner, account = load_one_star_account(ckpt)
    if account.state.highest_cleared_floor != 0 or any(
        getattr(update, "kind", "") == "mission_start"
        for event in ckpt.canonical_events
        for update in getattr(event, "state_updates", ())
    ):
        return None
    guided_pools = [
        (pool_id, pool)
        for pool_id, pool in account.config.summon_pools.items()
        if isinstance(pool, OneStarOpeningRosterSummonPool)
        and pool.initial_deployment_requires_guide_handoff
    ]
    if not guided_pools:
        return None
    if len(guided_pools) != 1:
        raise ValueError(
            "One-Star initial deployment guide handoff requires exactly one "
            "configured opening-roster authority"
        )
    pool_id, pool = guided_pools[0]

    opening_event = ckpt.canonical_events[0]
    opening_summons = [
        update
        for update in getattr(opening_event, "state_updates", ())
        if getattr(update, "kind", "") == "summon"
    ]
    if (
        len(opening_summons) != 1
        or opening_summons[0].target_id != pool_id
        or opening_summons[0].value != str(len(pool.slots))
    ):
        return None

    acquired_ids = {
        character.character_id
        for character in ckpt.characters
        if (
            (hero := load_one_star_hero(character)) is not None
            and hero.owner_lobby_id == account.config.lobby_id
            and hero.acquisition_event_id == opening_event.event_id
        )
    }
    ordered_ids = tuple(
        signal.character_id
        for signal in opening_event.activate
        if signal.character_id in acquired_ids
    )
    if (
        len(acquired_ids) != len(pool.slots)
        or len(ordered_ids) != len(pool.slots)
        or len(set(ordered_ids)) != len(ordered_ids)
        or set(ordered_ids) != acquired_ids
    ):
        return None
    return owner.character_id, account, ordered_ids


def _validate_one_star_initial_deployment_responder_order(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    actor_id: str,
    cat_ii_event: OpenCatIIEvent | None,
) -> None:
    """Require the opted-in opening roster's independent response order."""

    if cat_ii_event is not None:
        return
    guided = _one_star_initial_guided_roster(ckpt)
    if guided is None:
        return
    owner_id, _account, ordered_ids = guided
    if actor_id != owner_id:
        return
    transaction = _one_star_transaction_for_result(ckpt, result)
    operations = (
        transaction.operations
        if transaction is not None and transaction.present
        else []
    )
    pending_opens = [
        operation
        for operation in operations
        if getattr(operation, "operation", "") == "pending_open"
        and getattr(operation.pending, "kind", "") == "deployment"
        and set(operation.pending.participant_ids) == set(ordered_ids)
    ]
    if not pending_opens:
        return
    if (
        len(pending_opens) != 1
        or not result.requires_responders
        or result.required_responders != list(ordered_ids)
    ):
        raise ValueError(
            "One-Star guided initial deployment must open Cat II with "
            "required_responders exactly equal to the opening-roster slot "
            "order and no guide or extra responder"
        )


def _validate_one_star_initial_deployment_guide_handoff(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    cat_ii_event: OpenCatIIEvent | None,
) -> None:
    """Keep the opted-in opening deployment response-owned by its guide.

    An opening-roster pool may require one guide turn after the independently
    collected Hero intentions and before any deployment side effect.  This
    check runs on the first Cat II resolution candidate, while a complete
    router-envelope correction is still possible.  The guide's ordinary later
    turn owns the physical crossing and pending/mission resolution.
    """

    if cat_ii_event is None:
        return
    guided = _one_star_initial_guided_roster(ckpt)
    if guided is None:
        return
    owner_id, account, ordered_ids = guided
    pending = account.state.pending_operation
    if (
        pending is None
        or pending.kind != "deployment"
        or cat_ii_event.initiator_id != owner_id
        or set(pending.participant_ids) != set(ordered_ids)
    ):
        return
    pending_open_event = next(
        (
            event
            for event in ckpt.canonical_events
            if event.event_id == cat_ii_event.opening_event_id
        ),
        None,
    )
    if pending_open_event is None or not any(
        getattr(update, "kind", "") == "pending_open"
        and getattr(update, "target_id", "") == pending.operation_id
        and getattr(update, "value", "") == "deployment"
        for update in getattr(pending_open_event, "state_updates", ())
    ):
        return

    characters = {
        character.character_id: character for character in ckpt.characters
    }
    active_guide_ids = [
        guide_id
        for guide_id in account.state.guide_character_ids
        if (
            (guide := characters.get(guide_id)) is not None
            and guide.status.value == "active"
        )
    ]
    errors: list[str] = []
    if len(active_guide_ids) != 1:
        errors.append(
            "exactly one configured guide must be active for the handoff"
        )
        guide_id = ""
    else:
        guide_id = active_guide_ids[0]

    if cat_ii_event.required_responders != list(ordered_ids):
        errors.append(
            "the collected responders must retain the opening-roster slot order"
        )
    if result.requires_responders or result.required_responders:
        errors.append("the post-collection bridge cannot request responders")
    if result.next_output_character_ids != ([guide_id] if guide_id else []):
        errors.append(
            "the sole next_output must be the active configured guide"
        )
    guide_observer = next(
        (
            observer
            for observer in result.observers
            if observer.character_id == guide_id
        ),
        None,
    )
    if (
        not guide_id
        or guide_observer is None
        or guide_observer.observation_level != "d"
        or guide_observer.routing_role != "next_output"
    ):
        errors.append(
            "the active configured guide must be a direct next_output observer"
        )
    if guide_id and not any(
        fact.is_visible_to(guide_id)
        for fact in result.canonical_event.observable_facts
    ):
        errors.append(
            "the guide must receive an observable fact carrying the collected "
            "response context"
        )
    if result.perception_enrichment_character_ids:
        errors.append("the post-collection bridge cannot request enrichment")
    if getattr(result, "state_updates", None):
        errors.append("the post-collection bridge must have empty state_updates")
    generic_side_effects = {
        "spawn": bool(result.spawn),
        "dormant": bool(result.dormant),
        "cull": bool(result.cull),
        "activate": bool(result.activate),
        "location_updates": bool(result.location_updates),
        "commitment_open": bool(result.commitment_open.present),
        "commitment_resolutions": bool(result.commitment_resolutions),
        "commitment_interrupts": bool(result.commitment_interrupts),
    }
    present_side_effects = [
        name for name, present in generic_side_effects.items() if present
    ]
    if present_side_effects:
        errors.append(
            "the post-collection bridge must have empty generic side effects: "
            + ", ".join(present_side_effects)
        )
    if errors:
        raise ValueError(
            "One-Star initial deployment requires a side-effect-free "
            "post-collection guide handoff:\n- "
            + "\n- ".join(errors)
        )


def _validate_one_star_pending_resolution_event_contract(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
) -> None:
    """Validate state/generic coupling while a full router retry is possible.

    Embodied pending resolutions depend on generic ``location_updates`` that
    the compact One-Star state-update repair cannot author. Deployment also
    couples that physical crossing to an ordered mission start. Reject those
    cross-envelope mistakes before router history or rules state mutates so
    the existing one-shot full-output correction can repair the whole event.
    """

    if not isinstance(
        result,
        (OneStarEventRouterOutput, ClosedOneStarEventRouterOutput),
    ):
        return

    location_updates: dict[str, str] = {}
    for update in result.location_updates:
        if update.character_id in location_updates:
            raise ValueError(
                "One-Star event has duplicate generic location_updates for "
                f"{update.character_id!r}"
            )
        location_updates[update.character_id] = update.location_label

    transaction = _one_star_transaction_for_result(ckpt, result)
    operations = (
        transaction.operations
        if transaction is not None and transaction.present
        else []
    )
    resolutions = [
        (index, operation)
        for index, operation in enumerate(operations)
        if getattr(operation, "operation", "") == "pending_resolve"
    ]
    mission_starts = [
        (index, operation)
        for index, operation in enumerate(operations)
        if getattr(operation, "operation", "") == "mission_start"
    ]

    from app.engine.one_star_adapter import (
        find_one_star_account_owner,
        load_one_star_account,
        load_one_star_hero,
    )

    if find_one_star_account_owner(ckpt.characters) is None:
        # Ruleset-only prompt/schema fixtures have no durable One-Star account,
        # hence cannot carry an open pending operation for this validator.
        return
    _owner, account = load_one_star_account(ckpt)
    pending = account.state.pending_operation
    if pending is None:
        if resolutions:
            raise ValueError(
                "One-Star pending_resolve requires one matching open pending "
                "operation"
            )
        return

    characters = {
        character.character_id: character for character in ckpt.characters
    }
    affected_ids = {
        *pending.participant_ids,
        *([pending.target_id] if pending.target_id else []),
    }
    final_locations = {
        character_id: location_updates.get(
            character_id,
            getattr(characters.get(character_id), "location", ""),
        )
        for character_id in affected_ids
    }

    if pending.kind == "deployment":
        matching_resolutions = [
            (index, operation)
            for index, operation in resolutions
            if getattr(operation, "operation_id", "") == pending.operation_id
        ]
        matching_mission_starts = [
            (index, operation)
            for index, operation in mission_starts
            if getattr(operation, "pending_operation_id", "")
            == pending.operation_id
        ]
        reaches_gate = any(
            getattr(characters.get(character_id), "location", "")
            == pending.destination
            or final_locations[character_id] == pending.destination
            for character_id in pending.participant_ids
        )
        if not (
            reaches_gate
            or matching_resolutions
            or matching_mission_starts
            or resolutions
        ):
            return

        errors: list[str] = []
        misplaced_ids = [
            character_id
            for character_id in sorted(affected_ids)
            if final_locations[character_id] != pending.destination
        ]
        if misplaced_ids:
            errors.append(
                "generic location_updates must place every affected Hero at "
                f"the exact pending destination {pending.destination!r}; "
                "missing or mismatched: " + ", ".join(misplaced_ids)
            )
        activation_locations = {
            signal.character_id: signal.location_label
            for signal in result.activate
        }
        unselected_crossers: list[str] = []
        selected_ids = set(pending.participant_ids)
        for character in ckpt.characters:
            if character.character_id in selected_ids:
                continue
            hero = load_one_star_hero(character)
            if hero is None or hero.owner_lobby_id != account.config.lobby_id:
                continue
            planned_location = location_updates.get(
                character.character_id,
                activation_locations.get(
                    character.character_id,
                    character.location,
                ),
            )
            currently_beyond_gate = (
                character.status.value == "active"
                and character.location == pending.destination
            )
            will_be_active = (
                character.status.value == "active"
                or character.character_id in activation_locations
            )
            if currently_beyond_gate or (
                will_be_active and planned_location == pending.destination
            ):
                unselected_crossers.append(character.character_id)
        if unselected_crossers:
            errors.append(
                "unselected local Heroes cannot reach the pending deployment "
                "destination through current, location-update, or activation "
                "state: " + ", ".join(sorted(unselected_crossers))
            )
        if len(resolutions) != 1 or len(matching_resolutions) != 1:
            errors.append(
                "exactly one pending_resolve must match the open deployment "
                f"{pending.operation_id!r}"
            )
        if (
            len(mission_starts) != 1
            or len(matching_mission_starts) != 1
            or not matching_resolutions
            or matching_mission_starts[0][0] <= matching_resolutions[0][0]
        ):
            errors.append(
                "exactly one matching mission_start must follow "
                "pending_resolve in the same event"
            )
        elif (
            set(matching_mission_starts[0][1].mission.party_ids)
            != set(pending.participant_ids)
            or matching_mission_starts[0][1].mission.destination
            != pending.destination
        ):
            errors.append(
                "mission party and destination must exactly match the "
                "pending deployment"
            )
        if errors:
            raise ValueError(
                "One-Star deployment crossing contract is incomplete:\n- "
                + "\n- ".join(errors)
            )
        return

    if not resolutions:
        # Synthesis and promotion allow physical chamber entry to precede
        # resolution. Only an explicit resolve couples their state update to
        # the generic location envelope.
        return
    if (
        len(resolutions) != 1
        or getattr(resolutions[0][1], "operation_id", "")
        != pending.operation_id
    ):
        raise ValueError(
            "One-Star pending_resolve must match the one open pending operation "
            f"{pending.operation_id!r}"
        )
    misplaced_ids = [
        character_id
        for character_id in sorted(affected_ids)
        if final_locations[character_id] != pending.destination
    ]
    if misplaced_ids:
        raise ValueError(
            "One-Star pending resolution requires generic location_updates "
            f"that place every affected Hero at the exact pending destination "
            f"{pending.destination!r}; missing or mismatched: "
            + ", ".join(misplaced_ids)
        )


def _validate_one_star_guide_routing(
    ckpt: CheckpointFile,
    *,
    actor_id: str,
    result: EventRouterOutput,
) -> None:
    """Require mediated guide delivery for an account-owner lobby mutation.

    A guide observer is delivery, not a forced character turn. The router must
    select ``next_output`` only when the guide has actual fictional pressure to
    respond; otherwise the guide receives the scoped System fact as an ordinary
    observer.
    """
    if not _one_star_router_enabled(ckpt):
        return

    transaction = _one_star_transaction_for_result(ckpt, result)
    operations = (
        transaction.operations
        if transaction is not None and transaction.present
        else []
    )
    operation_names = {getattr(operation, "operation", "") for operation in operations}
    lifecycle_ids = {
        *result.dormant,
        *(signal.character_id for signal in result.activate),
        *(update.character_id for update in result.location_updates),
    }
    if (
        not operation_names & _ONE_STAR_LOBBY_MANAGEMENT_OPERATIONS
        and "hero_delta" not in operation_names
        and not lifecycle_ids
    ):
        return

    from app.engine.one_star_adapter import (
        load_one_star_account,
        load_one_star_hero,
    )

    owner, account = load_one_star_account(ckpt)
    if actor_id != owner.character_id:
        return
    characters = {character.character_id: character for character in ckpt.characters}
    lobby_locations = {
        account.config.lobby_location_label,
        *(
            requirement.required_location
            for kind, requirement in account.config.operation_requirements.items()
            if kind != "deployment" and requirement.required_location
        ),
    }
    typed_hero_management = False
    for operation in operations:
        if getattr(operation, "operation", "") != "hero_delta":
            continue
        character = characters.get(getattr(operation, "hero_id", ""))
        hero = load_one_star_hero(character) if character is not None else None
        if (
            character is not None
            and hero is not None
            and hero.owner_lobby_id == account.config.lobby_id
            and character.location in lobby_locations
        ):
            typed_hero_management = True
            break
    lifecycle_management = False
    for character_id in lifecycle_ids:
        character = characters.get(character_id)
        hero = load_one_star_hero(character) if character is not None else None
        if hero is None or hero.owner_lobby_id != account.config.lobby_id:
            continue
        target_locations = {
            signal.location_label
            for signal in result.activate
            if signal.character_id == character_id
        } | {
            update.location_label
            for update in result.location_updates
            if update.character_id == character_id
        }
        if character.location in lobby_locations or target_locations & lobby_locations:
            lifecycle_management = True
            break
    if (
        not operation_names & _ONE_STAR_LOBBY_MANAGEMENT_OPERATIONS
        and not typed_hero_management
        and not lifecycle_management
    ):
        return

    guide_ids = tuple(account.state.guide_character_ids)
    if not guide_ids:
        return
    observers_by_id = {observer.character_id: observer for observer in result.observers}
    for guide_id in guide_ids:
        guide = characters.get(guide_id)
        if guide is None or guide.status.value != "active":
            raise ValueError(
                "One-Star configured guide is unavailable for mediated "
                f"lobby delivery: {guide_id!r}"
            )
        observer = observers_by_id.get(guide_id)
        if observer is None:
            raise ValueError(
                "One-Star account-owner lobby mutation must include configured "
                f"guide {guide_id!r} as a mediated observer"
            )
        if observer.observation_level != "d" or observer.routing_role not in {
            "observe_only",
            "next_output",
        }:
            raise ValueError(
                "One-Star guide delivery must use direct mediated observation "
                "and an ordinary observer or genuine next-output role"
            )
        if not any(
            fact.audience == "only" and guide_id in fact.visible_to
            for fact in result.canonical_event.observable_facts
        ):
            raise ValueError(
                "One-Star account-owner lobby mutation must deliver a scoped "
                f"System fact to configured guide {guide_id!r}"
            )


def _validate_one_star_pending_response_routing(
    ckpt: CheckpointFile,
    *,
    actor_id: str,
    result: EventRouterOutput,
) -> None:
    """Keep Master-selected embodied operations response-owned by Heroes."""

    if not _one_star_router_enabled(ckpt):
        return
    transaction = _one_star_transaction_for_result(ckpt, result)
    if transaction is None or not transaction.present:
        return

    from app.engine.one_star_adapter import (
        load_one_star_account,
        load_one_star_hero,
    )

    owner, account = load_one_star_account(ckpt)
    if actor_id != owner.character_id:
        return
    routed_ids = set(result.required_responders)
    summoned_ids = {
        character_id
        for operation in transaction.operations
        if getattr(operation, "operation", "") == "summon"
        for character_id in operation.hero_ids
    }
    characters = {character.character_id: character for character in ckpt.characters}
    for operation in transaction.operations:
        if getattr(operation, "operation", "") != "pending_open":
            continue
        pending = operation.pending
        affected_ids = {
            *pending.participant_ids,
            *([pending.target_id] if pending.target_id else []),
        }
        required_ids: set[str] = set()
        for character_id in affected_ids - {actor_id}:
            character = characters.get(character_id)
            if character is None or character.status.value != "active":
                continue
            hero = load_one_star_hero(character)
            if hero is not None and (
                hero.owner_lobby_id == account.config.lobby_id
                or character_id in summoned_ids
            ):
                required_ids.add(character_id)
        missing = required_ids - routed_ids
        if required_ids and not result.requires_responders:
            raise ValueError(
                "One-Star embodied selection affecting active Heroes must "
                "open Cat II and collect every affected intention"
            )
        if missing:
            raise ValueError(
                "One-Star embodied selection omitted affected Hero response "
                "routing: " + ", ".join(sorted(missing))
            )


def _include_one_star_synthesis_guide_responders(
    ckpt: CheckpointFile,
    *,
    actor_id: str,
    result: EventRouterOutput,
) -> None:
    """Give active configured wardens their own synthesis intention.

    The adapter chooses no coercive action here.  It only makes the guide a
    participant in the already-contested selection so their character agent
    can state what they do if a selected Hero refuses; the Cat II resolution
    remains router-authored from all collected intentions.
    """

    if not _one_star_router_enabled(ckpt) or not result.requires_responders:
        return
    transaction = _one_star_transaction_for_result(ckpt, result)
    if transaction is None or not transaction.present:
        return
    if not any(
        getattr(operation, "operation", "") == "pending_open"
        and getattr(operation.pending, "kind", "") == "synthesis"
        for operation in transaction.operations
    ):
        return

    from app.engine.one_star_adapter import load_one_star_account

    owner, account = load_one_star_account(ckpt)
    if actor_id != owner.character_id:
        return
    characters = {character.character_id: character for character in ckpt.characters}
    for guide_id in account.state.guide_character_ids:
        guide = characters.get(guide_id)
        if guide is None or guide.status.value != "active":
            continue
        if guide_id != actor_id and guide_id not in result.required_responders:
            result.required_responders.append(guide_id)


def _validate_one_star_tutorial_routing(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
) -> None:
    """Record teaching only when every named recipient actually receives it."""

    transaction = _one_star_transaction_for_result(ckpt, result)
    if transaction is None or not transaction.present:
        return
    observers = {observer.character_id: observer for observer in result.observers}
    for operation in transaction.operations:
        if getattr(operation, "operation", "") != "tutorial_delivery":
            continue
        for recipient_id in operation.delivered_to_ids:
            observer = observers.get(recipient_id)
            if observer is None or observer.observation_level != "d":
                raise ValueError(
                    "One-Star tutorial delivery requires each recipient to be "
                    "a direct observer of the teaching event"
                )
            if not any(
                fact.is_visible_to(recipient_id)
                for fact in result.canonical_event.observable_facts
            ):
                raise ValueError(
                    "One-Star tutorial delivery requires a canonical fact "
                    f"visible to recipient {recipient_id!r}"
                )


def _rollback_one_star_materialized_spawns(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
) -> None:
    """Remove an unaccepted One-Star spawn overlay after prepare failure."""

    if not result.spawn:
        return
    from app.engine.closed_event_runtime import closed_event_runtime_for

    runtime = closed_event_runtime_for(ckpt)
    if runtime is None:
        return
    removed = runtime.spawn_authoring.rollback_roster(
        checkpoint=ckpt,
        transaction_id=runtime.transaction_id,
    )
    runtime.applied_character_ids.difference_update(removed)


def _router_ruleset_template_vars(
    prompt_mgr: PromptManager,
    *,
    ruleset_id: str,
    dnd_fresh: bool,
    ckpt: CheckpointFile | None = None,
) -> dict[str, str]:
    if ruleset_id == ONE_STAR_RULESET_ID:
        if ckpt is None:
            raise ValueError("One-Star router rules require the active checkpoint")
        from app.engine.one_star_router_context import (
            render_one_star_router_static_config,
        )

        return {
            "router_ruleset_addon": prompt_mgr.render(
                "event_router_ruleset_one_star",
                one_star_static_config=render_one_star_router_static_config(ckpt),
            ).strip(),
        }
    if dnd_fresh:
        return {
            "router_ruleset_addon": prompt_mgr.render(
                "event_router_ruleset_dnd5e",
            ).strip(),
        }
    if ruleset_id == DND5E_BASIC_RULESET_ID:
        # Non-fresh D&D calls use a separate Cat II/combat resolver or are
        # canonicalizing already-committed output; the rules-neutral fresh
        # Cat II classifier would incorrectly reintroduce violence-as-Cat-II.
        return {"router_ruleset_addon": ""}
    return {
        "router_ruleset_addon": prompt_mgr.render(
            "event_router_ruleset_default",
        ).strip(),
    }


# v11-r7j note: a mirror copy of the legacy `EventRouter` engine
# class's private helpers used to live in app/engine/event_router.py
# and these were straight-ported during the v11 transition. The
# legacy class was murdered in v11-r7j; this dispatcher is now the
# only home for these helpers and the duplication concern is gone.


# `build_setting_summary` is now in `app/engine/context_builder.py`
# and imported at the top of this module — pre-v11-r7j three near-
# identical copies of the same helper lived in this module, narrator,
# and engine_bridge.


def _build_router_world_lore(checkpoint: CheckpointFile) -> str:
    """Stable common-knowledge world context for the router system prefix."""
    parts: list[str] = []
    facts = [fact for fact in (checkpoint.world_state.facts or []) if fact]
    if facts:
        parts.append("Key world facts:\n" + "\n".join(f"- {fact}" for fact in facts))
    if checkpoint.world_state.lore:
        parts.append(checkpoint.world_state.lore)
    if not parts:
        return "No detailed lore available."
    return "\n\n".join(parts)


def _build_initial_roster_block(checkpoint: CheckpointFile) -> str:
    """Render the binding-invariant durable roster seed on the first call.

    This is stored once as compact assistant-side router history. It is not a
    turn-specific user block: live bindings cannot alter fictional identity,
    and later stateless calls must retain the starting cast even though raw
    user turns are not replayed. Router-authored spawn, activation, location,
    and status mutations then extend this seed through compact prior events.

    Keep it small: id, name, role, initial location, D&D identity/equipment,
    and active goals. Appearance belongs to prose-facing roles and authored
    opening participant blocks. Returns "" once a roster seed or canonical
    router event already exists.

    Membership and rendering deliberately ignore live bindings. A character's
    controller cannot change the router's fictional roster. Dormant records,
    including an unintroduced player-authored slot, remain absent. Authored
    opening/arrival participant blocks carry a selected dormant character only
    for the transition that introduces them.

    This block does not carry private interior beyond seed-authored goals and
    objectives. Fresh private thought stays in character-local history.
    """
    if any(
        message.role == "assistant"
        and isinstance(message.content, str)
        and message.content.startswith("roster_seed\n")
        for message in checkpoint.session_conversation
    ):
        return ""
    if any(
        not _is_router_content_history_message(message)
        for message in checkpoint.session_conversation
    ):
        return ""
    if not checkpoint.characters:
        return ""

    entries: list[str] = []
    for char in checkpoint.characters:
        if char.status.value != "active":
            continue

        role = char.public_sheet.role or "unknown role"

        parts = [
            f"- {char.character_id}",
            f"  Name: {char.name}",
            f"  Role: {role}",
        ]
        if is_non_social_hazard(char):
            parts.append("  Kind: non-social hazard")
        location = char.location or "unknown location"
        parts.append(f"  Location: {location}")
        dnd_identity = build_dnd_character_identity_sentence(checkpoint, char)
        if dnd_identity:
            parts.append(f"  {dnd_identity}")
        dnd_equipment = build_dnd_character_equipment_sentence(checkpoint, char)
        if dnd_equipment:
            parts.append(f"  {dnd_equipment}")
        goals = [g for g in (char.private_state.goals or []) if g]
        if goals:
            parts.append("  Goals (long-term): " + "; ".join(goals))
        objs = [o for o in (char.private_state.current_objectives or []) if o]
        if objs:
            parts.append("  Current objectives (active pursuits): " + "; ".join(objs))
        entries.append("\n".join(parts))

    if not entries:
        return ""

    header = "roster_seed\nInitial active fictional identities and pursuits:\n"
    return header + "\n\n".join(entries) + "\n"


def _is_router_content_history_message(message: ConversationMessage) -> bool:
    if message.role != "assistant" or not isinstance(message.content, str):
        return False
    return message.content.startswith(
        ("content_known ", "location_card ", "front_signal ")
    )


def _build_engine_state_updates_block(checkpoint: CheckpointFile) -> str:
    """Drain non-router engine mutations for the next fresh router call.

    Exhaustive current producers:
    - player loot claim: character inventory changed from a D&D loot offer
    - player loot currency split: party inventories changed from a D&D loot offer
    - D&D sheet attached: character mechanics/ruleset state changed externally
    - an authored character left active fiction and became dormant
    - a custom character became ready for an authored arrival
    - an existing character received a replacement fictional identity

    Explicitly excluded:
    - router-authored spawn/dormant/cull/location/commitment changes
    - spawned-character summaries or interior; the generated display name is
      folded into the corresponding compact `prior_event` spawn line instead
    - combat deaths/effect expirations when they are already emitted through
      compact history or post-combat continuity updates
    - D&D Cat II resolver outputs, which are compact router history
    - routine D&D combat-manager outputs while initiative remains active
    - clocks/session leading time derived from canonical event timing
    - XP awards unless they become intentionally visible to fiction
    """
    queued = checkpoint.session.pending_engine_state_updates or []
    if not queued:
        return ""
    checkpoint.session.pending_engine_state_updates = []
    body = "\n".join(f"- {entry}" for entry in queued)
    return (
        "## Engine State Updates\n"
        "Durable state changed outside router-authored canonical events. "
        "Fold these updates into the next adjudication without replaying "
        "them as new visible action unless the current intention makes "
        "that visibility relevant.\n\n"
        f"{body}\n"
    )


def _build_router_input_block(*blocks: str) -> str:
    return "\n\n".join(block.strip() for block in blocks if block and block.strip())


def _is_begin_directive(intention: str) -> bool:
    return (intention or "").strip().casefold() == "(begin)"


def _build_one_star_opening_roster_block(
    checkpoint: CheckpointFile,
    opening_participant_ids: set[str],
) -> str:
    """Project an adapter-resolved opening roster into the fresh router turn."""

    if not _one_star_router_enabled(checkpoint):
        return ""

    from app.engine.one_star_adapter import (
        load_one_star_account,
        one_star_opening_roster_preview,
    )
    from app.schemas.one_star import (
        OneStarOpeningActorSummonPool,
        OneStarOpeningRosterSummonPool,
    )

    owner, account = load_one_star_account(checkpoint)
    opening_actor_ids = {
        pool.character_id
        for pool in account.config.summon_pools.values()
        if isinstance(pool, OneStarOpeningActorSummonPool)
    }
    if opening_actor_ids & opening_participant_ids:
        return ""
    if owner.character_id not in opening_participant_ids:
        return ""

    roster_pool_ids = [
        pool_id
        for pool_id, pool in account.config.summon_pools.items()
        if isinstance(pool, OneStarOpeningRosterSummonPool)
    ]
    if not roster_pool_ids:
        return ""
    if len(roster_pool_ids) != 1:
        raise ValueError(
            "an authored One-Star opening requires exactly one opening-roster pool"
        )

    pool_id = roster_pool_ids[0]
    characters = {
        character.character_id: character for character in checkpoint.characters
    }
    lines = [
        "## Resolved One-Star Opening Roster",
        "Use these existing Heroes in exactly this order. Their identities and "
        "birth grades are already resolved; do not substitute or reroll them.",
        f"Pool: {pool_id}",
    ]
    for draw in one_star_opening_roster_preview(checkpoint, pool_id):
        character_id = draw.existing_character_id
        if not character_id or character_id not in characters:
            raise ValueError(
                "resolved One-Star opening roster references a missing character"
            )
        character = characters[character_id]
        role = character.public_sheet.role or "unspecified role"
        appearance = (character.public_sheet.appearance or "not yet described").strip()
        lines.extend((
            f"{draw.slot}. {character.character_id}",
            f"   Name: {character.name}",
            f"   Birth stars: {draw.birth_stars}",
            f"   Role: {role}",
            f"   Appearance: {appearance}",
        ))
    return "\n".join(lines)


def _build_opening_context_block(
    checkpoint: CheckpointFile,
    intention: str,
    acting_character_id: str,
) -> str:
    """Render semantic participants plus story rules for begin/arrival.

    The opening participant selection is computed from interface bindings, but
    the router receives only its authorial meaning: these existing fictional
    records must enter this opening. For ``(arrive)``, the submitted actor is
    the sole semantic arrival participant. No controller/source metadata is
    exposed in either mode.
    """
    normalized = (intention or "").strip().casefold()
    is_begin = normalized == "(begin)"
    is_arrive = normalized == "(arrive)"
    if not is_begin and not is_arrive:
        return ""

    if is_begin:
        selected_ids = collect_player_ids(checkpoint)
        participant_heading = "## Authored Opening Participants"
        participant_purpose = (
            "Place these existing characters in the opening according to the "
            "authored context below."
        )
    else:
        selected_ids = {acting_character_id}
        participant_heading = "## Arriving Existing Character"
        participant_purpose = (
            "Bring this existing character into the current story frame now."
        )

    participant_lines: list[str] = []
    for character in checkpoint.characters:
        if character.character_id not in selected_ids:
            continue
        role = character.public_sheet.role or "unspecified role"
        appearance = (character.public_sheet.appearance or "not yet described").strip()
        location = character.location or "not yet placed"
        participant_lines.extend(
            [
                f"- {character.character_id}",
                f"  Name: {character.name}",
                f"  Role: {role}",
                f"  Appearance: {appearance}",
                f"  Current status: {character.status.value}",
                f"  Current location: {location}",
            ]
        )
        dnd_identity = build_dnd_character_identity_sentence(
            checkpoint,
            character,
        )
        if dnd_identity:
            participant_lines.append(f"  {dnd_identity}")
        dnd_equipment = build_dnd_character_equipment_sentence(
            checkpoint,
            character,
        )
        if dnd_equipment:
            participant_lines.append(f"  {dnd_equipment}")

    if participant_lines:
        participants = f"{participant_heading}\n{participant_purpose}\n" + "\n".join(
            participant_lines
        )
    else:
        participants = (
            f"{participant_heading}\n"
            "No existing character is authored to enter this transition."
        )

    if is_arrive:
        return participants

    resolved_roster = _build_one_star_opening_roster_block(
        checkpoint,
        selected_ids,
    )

    policy = checkpoint.world_state.opening
    if policy is None:
        opening_context = (
            "## Authored Opening Context\n"
            "New-character spawn requests: forbidden.\n"
            "No story-specific opening requirements were authored."
        )
    else:
        if policy.allow_spawns:
            spawn_rule = (
                "allowed only when the authored requirements below explicitly "
                "require newly generated characters"
            )
        else:
            spawn_rule = "forbidden"
        context = policy.context or "No story-specific opening requirements."
        opening_context = (
            "## Authored Opening Context\n"
            f"New-character spawn requests: {spawn_rule}.\n"
            f"{context}"
        )
    return _build_router_input_block(
        participants,
        resolved_roster,
        opening_context,
    )


def _validate_opening_spawn_authority(
    checkpoint: CheckpointFile,
    intention: str,
    result: EventRouterOutput,
) -> None:
    """Reject unauthorized or non-new spawn targets before history mutates."""
    if not _is_begin_directive(intention) or not result.spawn:
        return

    policy = checkpoint.world_state.opening
    if policy is None or not policy.allow_spawns:
        raise ValueError(
            "router emitted spawn requests for (begin), but this story has "
            "not authorized opening spawns"
        )

    existing_ids = {character.character_id for character in checkpoint.characters}
    conflicting_ids = [
        request.character_id
        for request in result.spawn
        if request.character_id in existing_ids
    ]
    if conflicting_ids:
        raise ValueError(
            "opening spawn requests must target genuinely new character ids; "
            "existing ids: " + ", ".join(dict.fromkeys(conflicting_ids))
        )


def _validate_projected_one_star_authored_opening_handoff(
    checkpoint: CheckpointFile,
    intention: str,
    result: EventRouterOutput,
) -> None:
    """Guard an adapter-authored opening lifecycle before history mutates.

    Opening summon identities are intentionally absent from the router's
    generic spawn/activate fields. Preview the adapter-owned lifecycle on a
    copy, then apply the same branch-aware handoff guard used after prepare.
    """

    if (
        not _is_begin_directive(intention)
        or not isinstance(
            result,
            (OneStarEventRouterOutput, ClosedOneStarEventRouterOutput),
        )
        or not (result.requires_responders or result.next_output_character_ids)
    ):
        return

    from app.engine.one_star_adapter import one_star_summon_lifecycle
    from app.engine.turn_loop import _validate_authored_opening_handoff

    adapter_spawns, adapter_wakes = one_star_summon_lifecycle(
        checkpoint,
        result.state_updates,
    )
    projected = result.model_copy(deep=True)
    projected.spawn.extend(adapter_spawns)
    projected.activate.extend(adapter_wakes)
    _validate_authored_opening_handoff(
        checkpoint,
        projected,
        submission=intention,
        events_closed=0,
        is_continuation=False,
    )


def _validate_one_star_guide_and_opening_envelope(
    checkpoint: CheckpointFile,
    *,
    actor_id: str,
    intention: str,
    result: EventRouterOutput,
) -> None:
    """Report coupled guide-delivery and authored-handoff defects together."""

    from app.engine.one_star_adapter import OneStarTransactionError

    errors: list[str] = []
    validators = (
        lambda: _validate_one_star_guide_routing(
            checkpoint,
            actor_id=actor_id,
            result=result,
        ),
        lambda: _validate_projected_one_star_authored_opening_handoff(
            checkpoint,
            intention,
            result,
        ),
    )
    for validate in validators:
        try:
            validate()
        except OneStarTransactionError:
            # State-shape failures keep their narrower repair path.
            raise
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError(
            "One-Star router envelope violates coupled routing contracts:\n- "
            + "\n- ".join(errors)
        )


def _router_call_snapshot(ckpt: CheckpointFile) -> dict[str, object]:
    """Capture prompt-side mutable state before a router call.

    Content lookup records are deterministic router-history deltas, just like
    compact `prior_event` records. They must not stay appended if the router call
    fails before producing a canonical event.
    """
    return {
        "pending_engine_state_updates": list(ckpt.session.pending_engine_state_updates),
        "content_state": deepcopy(getattr(ckpt.session, "content_state", {})),
        "content_manager_preflight_cycle": getattr(
            ckpt.session,
            "content_manager_preflight_cycle",
            0,
        ),
        "content_manager_last_run_cycle": getattr(
            ckpt.session,
            "content_manager_last_run_cycle",
            -1,
        ),
        "session_conversation_len": len(ckpt.session_conversation),
    }


def _restore_router_call_snapshot(
    ckpt: CheckpointFile,
    snapshot: dict[str, object],
) -> None:
    ckpt.session.pending_engine_state_updates = list(
        snapshot["pending_engine_state_updates"]
    )
    if hasattr(ckpt.session, "content_state"):
        ckpt.session.content_state = deepcopy(snapshot["content_state"])
    if hasattr(ckpt.session, "content_manager_preflight_cycle"):
        ckpt.session.content_manager_preflight_cycle = int(
            snapshot["content_manager_preflight_cycle"]
        )
    if hasattr(ckpt.session, "content_manager_last_run_cycle"):
        ckpt.session.content_manager_last_run_cycle = int(
            snapshot["content_manager_last_run_cycle"]
        )
    del ckpt.session_conversation[int(snapshot["session_conversation_len"]) :]


def _append_pending_router_content_records(ckpt: CheckpointFile) -> list[str]:
    """Append one-shot content lookup deltas to router history, if any."""
    from app.engine.content_resolver import append_pending_router_content_records

    return append_pending_router_content_records(ckpt)


async def _append_router_content_lookup_records(
    ckpt: CheckpointFile,
    *,
    actor_id: str,
    current_input: str,
    client: LLMClient,
    prompt_mgr: PromptManager,
) -> list[str]:
    """Append bounded preflight content records to router history."""
    from app.engine.content_manager import (
        append_content_manager_router_records,
        content_manager_enabled,
    )
    from app.engine.content_lookup import append_router_content_lookup_records_with_llm

    if getattr(ckpt.session, "active_combat", None) is not None:
        logger.info("Skipping router content preflight during active combat")
        return _append_pending_router_content_records(ckpt)

    if content_manager_enabled(ckpt):
        return await append_content_manager_router_records(
            ckpt,
            actor_id=actor_id,
            current_input=current_input,
            client=client,
            prompt_mgr=prompt_mgr,
        )
    return await append_router_content_lookup_records_with_llm(
        ckpt,
        actor_id=actor_id,
        current_input=current_input,
        client=client,
        prompt_mgr=prompt_mgr,
    )


def _transaction_waits_for_player_rolls(
    ckpt: CheckpointFile,
    event_id: str,
    *,
    source: str = "",
) -> bool:
    for transaction in ckpt.session.cat_ii_roll_transactions:
        if transaction.event_id != event_id:
            continue
        if source and transaction.source != source:
            continue
        return any(
            record.actor_control == "player" and record.status == "pending"
            for record in transaction.rolls
        )
    return False


def _compact_router_history_text(text: str) -> str:
    return " ".join((text or "").split())


def _compact_id_list(values: list[str]) -> str:
    return ",".join(value for value in values if value) or "-"


_MISSION_STATUS_RATIONALE_PREFIX = "mission_status:"
_MISSION_STATUS_HISTORY_PREFIX = "mission_status "


def _mission_status_history_line(decision_rationale: str) -> str:
    """Project one marked mission snapshot out of router diagnostics."""
    marked = [
        line.strip()
        for line in (decision_rationale or "").splitlines()
        if line.strip().startswith(_MISSION_STATUS_RATIONALE_PREFIX)
    ]
    if len(marked) > 1:
        raise ValueError("router emitted more than one mission_status line")
    if not marked:
        return ""
    status = _compact_router_history_text(
        marked[0][len(_MISSION_STATUS_RATIONALE_PREFIX) :]
    )
    if not status:
        raise ValueError("router emitted an empty mission_status line")
    return _MISSION_STATUS_HISTORY_PREFIX + status


def _drop_superseded_mission_status(
    conversation: list[ConversationMessage],
) -> None:
    """Keep one current mission snapshot across compact prior events."""
    for index, message in enumerate(conversation):
        if (
            message.role != "assistant"
            or not isinstance(message.content, str)
            or not message.content.startswith("prior_event ")
        ):
            continue
        lines = message.content.splitlines()
        retained = [
            line
            for line in lines
            if not line.startswith(_MISSION_STATUS_HISTORY_PREFIX)
        ]
        if len(retained) != len(lines):
            conversation[index] = ConversationMessage(
                role="assistant",
                content="\n".join(retained),
            )


def _defer_history_user_prompt(intention: str) -> str:
    """Return the compact user-history entry for defer, if applicable.

    Router history usually stores only deterministic assistant-side
    `prior_event` records. `(defer)` is the exception: repeated defers
    are pacing feedback, so the next router call needs to see that the
    player explicitly deferred rather than merely infer it from whatever
    event the previous router authored.
    """
    if (intention or "").strip().lower() == "(defer)":
        return "(defer)"
    return ""


def _one_star_engine_hero_history_line(character: CharacterRecord) -> str:
    """Project a newly engine-authored Hero sheet once into router history."""

    from app.engine.one_star_adapter import load_one_star_hero

    hero = load_one_star_hero(character)
    if hero is None:
        return ""
    payload = {
        "character_id": character.character_id,
        "name": character.name,
        "status": character.status.value,
        "location": character.location,
        "birth_stars": hero.birth_stars,
        "current_stars": hero.current_stars,
        "level": hero.level,
        "experience_points": hero.experience_points,
        "hp_current": hero.hp_current,
        "hp_max": hero.hp_max,
        "stats": hero.stats,
        "equipment": [item.model_dump(mode="json") for item in hero.equipment],
        "skills": [skill.model_dump(mode="json") for skill in hero.skills],
        "conditions": hero.conditions,
        "persistent_injuries": hero.persistent_injuries,
        "innate_system_sight": hero.innate_system_sight,
    }
    return "one_star_authority_hero " + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _router_history_record(
    *,
    acting_character_id: str,
    result: EventRouterOutput,
    mode: str = "intention",
    spawn_names: Mapping[str, str] | None = None,
    preserved_header: str = "",
    one_star_engine_characters: Sequence[CharacterRecord] = (),
    one_star_engine_updates: Sequence[str] = (),
    preserved_auxiliary_lines: Sequence[str] = (),
) -> str:
    """Compact assistant-side memory of a prior router output.

    Router history exists to carry canonical continuity, not to replay the
    full structured-output envelope. Store a deterministic event memory and
    omit the raw user message, broad `decision_rationale`, feasibility
    boilerplate, empty schema fields, and JSON punctuation. The one marked
    mission snapshot is retained because it is continuity rather than
    diagnostic explanation.
    """
    header = preserved_header
    if not header:
        header = (
            f"prior_event {result.event_id} @{result.effective_at_s}"
            f"+{result.duration_s} source={acting_character_id or '-'} mode={mode}"
        )
        header += f" kind={result.event_kind}"
        if result.requires_responders:
            header += f" requires={_compact_id_list(result.required_responders)}"
        if result.next_output_character_ids:
            header += f" next={_compact_id_list(result.next_output_character_ids)}"
        if result.perception_enrichment_character_ids:
            header += (
                " enrich="
                f"{_compact_id_list(result.perception_enrichment_character_ids)}"
            )

    lines = [header]
    # One-Star's compact updates own mission continuity. Do not replay a
    # parallel diagnostic ``mission_status:`` line.
    mission_status = (
        ""
        if isinstance(
            result,
            (OneStarEventRouterOutput, ClosedOneStarEventRouterOutput),
        )
        else _mission_status_history_line(result.decision_rationale)
    )
    if mission_status:
        lines.append(mission_status)

    for fact in result.canonical_event.observable_facts:
        text = _compact_router_history_text(fact.text)
        if not text:
            continue
        audience = (
            "all"
            if fact.audience == "all_observers"
            else f"only[{_compact_id_list(fact.visible_to)}]"
        )
        lines.append(f"fact {audience} @{fact.at_offset_s}+{fact.duration_s}: {text}")

    if result.observers:
        observer_bits = [
            f"{observer.character_id}:{observer.observation_level}:"
            f"{observer.routing_role}"
            for observer in result.observers
            if observer.character_id
        ]
        if observer_bits:
            lines.append("obs " + " ".join(observer_bits))

    if result.spawn:
        preserved_spawns = {
            parts[1]: line
            for line in preserved_auxiliary_lines
            if line.startswith("spawn ") and len(parts := line.split(" ", 2)) >= 2
        }
        for spawn in result.spawn:
            objectives = "; ".join(
                _compact_router_history_text(objective)
                for objective in spawn.seed.objectives
                if objective
            )
            seed_bits = [
                f"role={_compact_router_history_text(spawn.seed.role)}",
                f"reason={_compact_router_history_text(spawn.seed.reason)}",
                f"loc={_compact_router_history_text(spawn.seed.location)}",
            ]
            name = _compact_router_history_text(
                (spawn_names or {}).get(spawn.character_id, "")
            )
            if name:
                seed_bits.insert(0, f"name={name}")
            if objectives:
                seed_bits.append(f"objectives={objectives}")
            if not name and spawn.character_id in preserved_spawns:
                lines.append(preserved_spawns[spawn.character_id])
            else:
                lines.append(f"spawn {spawn.character_id} " + " ".join(seed_bits))
    if result.dormant:
        lines.append(f"dormant {_compact_id_list(result.dormant)}")
    if result.cull:
        lines.append(f"cull {_compact_id_list(result.cull)}")

    commitment_open = result.commitment_open
    if commitment_open.present:
        lines.append(
            "commit_open "
            f"actors={_compact_id_list(commitment_open.actor_ids)} "
            f"expected={commitment_open.expected_duration_s} "
            f"max={commitment_open.max_duration_s} "
            f"loc={_compact_router_history_text(commitment_open.location_label)} "
            f"desc={_compact_router_history_text(commitment_open.description)}"
        )
    for signal in result.commitment_resolutions:
        lines.append(
            "commit_resolve "
            f"id={signal.commitment_id or '-'} "
            f"actors={_compact_id_list(signal.actor_ids)} "
            f"at={signal.resolved_at_offset_s} "
            f"reason={_compact_router_history_text(signal.reason)}"
        )
    for signal in result.commitment_interrupts:
        lines.append(
            "commit_interrupt "
            f"id={signal.commitment_id or '-'} "
            f"actors={_compact_id_list(signal.actor_ids)} "
            f"at={signal.observed_at_offset_s} "
            f"reason={_compact_router_history_text(signal.reason)}"
        )
    for update in result.location_updates:
        lines.append(
            "loc "
            f"{update.character_id}={_compact_router_history_text(update.location_label)}"
        )

    if isinstance(
        result,
        (OneStarEventRouterOutput, ClosedOneStarEventRouterOutput),
    ):
        for update in result.state_updates:
            lines.append(
                "one_star_update "
                + json.dumps(
                    update.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        for update in one_star_engine_updates:
            compact = _compact_router_history_text(update)
            if compact:
                lines.append("one_star_authority_update " + compact)
        for character in one_star_engine_characters:
            line = _one_star_engine_hero_history_line(character)
            if line:
                lines.append(line)
        for line in preserved_auxiliary_lines:
            if (
                line.startswith(
                    (
                        "one_star_authority_hero ",
                        "one_star_authority_update ",
                    )
                )
                and line not in lines
            ):
                lines.append(line)

    interaction_mode = getattr(result, "interaction_mode", "")
    if interaction_mode:
        lines.append(f"dnd_mode {interaction_mode}")
    combatant_ids = getattr(result, "combatant_ids", [])
    if combatant_ids:
        lines.append(f"combatants {_compact_id_list(combatant_ids)}")

    return "\n".join(lines)


def _append_router_history_record(
    conversation: list[ConversationMessage],
    *,
    acting_character_id: str,
    result: EventRouterOutput,
    mode: str = "intention",
    user_prompt: str = "",
) -> None:
    record = _router_history_record(
        acting_character_id=acting_character_id,
        result=result,
        mode=mode,
    )
    if f"\n{_MISSION_STATUS_HISTORY_PREFIX}" in record:
        _drop_superseded_mission_status(conversation)
    if user_prompt:
        conversation.append(
            ConversationMessage(
                role="user",
                content=user_prompt,
            )
        )
    conversation.append(
        ConversationMessage(
            role="assistant",
            content=record,
        )
    )


def refresh_router_history_record(
    conversation: list[ConversationMessage],
    *,
    result: EventRouterOutput,
    acting_character_id: str | None = None,
    mode: str | None = None,
    spawned_characters: Sequence[CharacterRecord] = (),
    one_star_engine_characters: Sequence[CharacterRecord] = (),
    one_star_engine_updates: Sequence[str] = (),
    force: bool = False,
) -> bool:
    """Replace the compact memory for a router event after mutation.

    Harvest paths append authoritative perception facts after the router
    output has already been normalized and stored. Spawn authoring assigns
    public display names after the router output exists. Keep both mutations
    in the same durable `prior_event` projection without restoring generated
    summaries or a full roster replay.
    """
    spawn_names = {
        character.character_id: character.name
        for character in spawned_characters
        if character.character_id and character.name
    }
    identity_only = acting_character_id is None and mode is None
    if identity_only and not (
        (spawn_names and result.spawn)
        or one_star_engine_characters
        or one_star_engine_updates
        or force
    ):
        return False
    prefix = f"prior_event {result.event_id} "
    for index in range(len(conversation) - 1, -1, -1):
        message = conversation[index]
        if (
            message.role == "assistant"
            and isinstance(message.content, str)
            and message.content.startswith(prefix)
        ):
            preserved_header = ""
            preserved_auxiliary_lines: list[str] = []
            if acting_character_id is None or mode is None:
                preserved_header = message.content.split("\n", 1)[0]
            preserved_auxiliary_lines = [
                line
                for line in message.content.splitlines()[1:]
                if line.startswith(
                    (
                        "spawn ",
                        "one_star_authority_hero ",
                        "one_star_authority_update ",
                    )
                )
            ]
            conversation[index] = ConversationMessage(
                role="assistant",
                content=_router_history_record(
                    acting_character_id=acting_character_id or "",
                    result=result,
                    mode=mode or "intention",
                    spawn_names=spawn_names,
                    preserved_header=preserved_header,
                    one_star_engine_characters=one_star_engine_characters,
                    one_star_engine_updates=one_star_engine_updates,
                    preserved_auxiliary_lines=preserved_auxiliary_lines,
                ),
            )
            return True
    if not identity_only:
        logger.warning(
            "Router history refresh found no prior record for event %s",
            result.event_id,
        )
    return False


def _normalize_router_result_for_history(
    ckpt: CheckpointFile,
    *,
    result: EventRouterOutput,
    clock_anchor_character_id: str = "",
    cat_ii_event: OpenCatIIEvent | None = None,
) -> None:
    if clock_anchor_character_id:
        actor = next(
            (c for c in ckpt.characters if c.character_id == clock_anchor_character_id),
            None,
        )
        if actor is not None and cat_ii_event is None:
            result.effective_at_s = max(result.effective_at_s, actor.clock_at_s)
    if cat_ii_event is None:
        result.effective_at_s = max(
            result.effective_at_s,
            ckpt.session.leading_at_s,
        )
    if cat_ii_event is not None and cat_ii_event.opening_event_id:
        opening = next(
            (
                event
                for event in ckpt.canonical_events
                if event.event_id == cat_ii_event.opening_event_id
            ),
            None,
        )
        if opening is not None:
            result.effective_at_s = opening.effective_at_s


def _roll_transaction_actor_id(ckpt: CheckpointFile, event_id: str) -> str:
    for transaction in ckpt.session.cat_ii_roll_transactions:
        if transaction.event_id == event_id:
            return transaction.actor_id
    return ""


def _build_router_context(
    ckpt: CheckpointFile,
    acting_character_id: str,
    *,
    resolve_actor_fallback: bool = True,
    include_engine_state_updates: bool = True,
) -> dict[str, str]:
    """Collect every context variable the event_router template needs
    aside from the current router input block.

    Returns a dict ready to splat into `prompt_mgr.render_messages`
    after merging in `{router_input_block}`.
    """
    if resolve_actor_fallback:
        acting_id, _acting_char, _acting_name = resolve_acting_character(
            ckpt,
            acting_character_id,
        )
    else:
        acting_id = acting_character_id

    return {
        "setting_summary": build_setting_summary(ckpt),
        "world_lore": _build_router_world_lore(ckpt),
        "world_rules": build_world_rules(ckpt),
        "hidden_lore": ckpt.world_state.hidden_lore or "None.",
        "hidden_facts": build_hidden_facts(ckpt, empty="None."),
        "acting_character_id": acting_id,
        "initial_roster_block": _build_initial_roster_block(ckpt),
        "engine_state_updates_block": (
            _build_engine_state_updates_block(ckpt)
            if include_engine_state_updates
            else ""
        ),
    }


class LLMDispatcher:
    """Production Dispatcher implementation — binds `turn_loop.run_beat`
    to the real router / agent / narrator modules."""

    def __init__(self, client: LLMClient, prompt_mgr: PromptManager):
        self.client = client
        self.prompt_mgr = prompt_mgr
        # Character agent is stateless aside from `last_usage`; reusing one
        # instance avoids per-call allocation.
        self._agent = CharacterAgent(client, prompt_mgr)
        self._dnd_cat_ii = DndCatIIResolver(client, prompt_mgr)
        self._dnd_combat = DndCombatResolver(client, prompt_mgr)

    async def materialize_spawns(
        self,
        *,
        ckpt: CheckpointFile,
        result: EventRouterOutput,
        actor_id: str,
        character_ids: list[str],
    ) -> list[str]:
        """Materialize router spawns needed before in-beat dispatch.

        Most router spawns are applied by the orchestrator after `run_beat`.
        Cat II responder and same-event next-output dispatch happen earlier,
        so the event's full authored result is applied immediately in router
        order through the orchestrator-owned callback. The post-beat pass then
        observes the same records as already applied without creating
        characters a second time.
        """
        target_ids = {cid for cid in character_ids if cid}
        if not target_ids:
            return []
        requests = [spawn for spawn in result.spawn if spawn.character_id in target_ids]
        if not requests:
            return []

        from app.engine.closed_event_runtime import (
            closed_event_runtime_for,
        )

        runtime = closed_event_runtime_for(ckpt)
        if runtime is None:
            raise RuntimeError(
                "router spawn materialization requires the shared closed-event runtime"
            )
        # Start every spawn on the finalized router output from one immutable
        # snapshot, then await because an in-beat dispatch needs the records.
        records = await runtime.authored_records(
            checkpoint=ckpt,
            event=result,
            actor_id=actor_id,
        )
        applied = runtime.apply_records(
            ckpt,
            records,
        )
        refresh_router_history_record(
            ckpt.session_conversation,
            result=result,
            spawned_characters=records,
        )
        return applied

    async def prepare_ruleset_event(
        self,
        *,
        ckpt: CheckpointFile,
        result: EventRouterOutput,
        actor_id: str,
        authoritative_system_result: bool = False,
    ) -> None:
        """Validate and atomically apply compact opt-in ruleset updates.

        Generic narrative and D&D events have no work here.  One-Star first
        awaits every identity authored by this event, then validates the
        complete ledger/lifecycle transition on a durable copy. A single
        bounded router repair may correct only the compact update list; the
        already-authored fiction and routing envelope must remain identical.
        """

        if not _one_star_router_enabled(ckpt):
            return
        if not isinstance(
            result,
            (OneStarEventRouterOutput, ClosedOneStarEventRouterOutput),
        ):
            raise RuntimeError(
                "One-Star routing returned an output without its ruleset "
                "state-update contract"
            )

        from app.engine.one_star_adapter import (
            OneStarTransactionError,
            apply_one_star_prepared_mutation,
            one_star_event_already_applied,
            one_star_event_fingerprint,
            one_star_summon_lifecycle,
            preflight_one_star_account_updates,
            prepare_one_star_transaction,
        )

        player_submission = actor_id in collect_player_ids(ckpt)
        if player_submission:
            preflight_one_star_account_updates(
                ckpt,
                result.state_updates,
                initiating_actor_id=actor_id,
                canonical_at_s=(result.effective_at_s + result.duration_s),
            )
        adapter_spawns, adapter_wakes = one_star_summon_lifecycle(
            ckpt,
            result.state_updates,
        )
        existing_spawn_ids = {request.character_id for request in result.spawn}
        existing_wake_ids = {signal.character_id for signal in result.activate}
        generated_overlap = (
            existing_spawn_ids & {request.character_id for request in adapter_spawns}
        ) | (existing_wake_ids & {signal.character_id for signal in adapter_wakes})
        if generated_overlap:
            raise OneStarTransactionError(
                "summon lifecycle is adapter-authored and was duplicated: "
                + ", ".join(sorted(generated_overlap))
            )
        result.spawn.extend(adapter_spawns)
        result.activate.extend(adapter_wakes)

        spawned_ids = {request.character_id for request in result.spawn}
        activated_ids = {signal.character_id for signal in result.activate}
        lifecycle_overlap = spawned_ids & activated_ids
        if lifecycle_overlap:
            raise OneStarTransactionError(
                "One-Star event cannot both spawn and activate the same "
                "character: " + ", ".join(sorted(lifecycle_overlap))
            )

        if not authoritative_system_result:
            _include_one_star_synthesis_guide_responders(
                ckpt,
                actor_id=actor_id,
                result=result,
            )
        event_fingerprint = one_star_event_fingerprint(
            {
                "actor_id": actor_id,
                "event": result.model_dump(mode="json"),
            }
        )
        if one_star_event_already_applied(
            ckpt,
            event_id=result.event_id,
            event_fingerprint=event_fingerprint,
        ):
            raise OneStarTransactionError(
                "One-Star router event was already committed and cannot be "
                "broadcast a second time"
            )

        if not authoritative_system_result:
            _validate_one_star_cat_ii_transaction(ckpt, result)
            _validate_one_star_tutorial_routing(ckpt, result)
            _validate_one_star_pending_response_routing(
                ckpt,
                actor_id=actor_id,
                result=result,
            )
        if not authoritative_system_result:
            _validate_one_star_guide_routing(
                ckpt,
                actor_id=actor_id,
                result=result,
            )

        if result.spawn:
            try:
                await self.materialize_spawns(
                    ckpt=ckpt,
                    result=result,
                    actor_id=actor_id,
                    character_ids=[request.character_id for request in result.spawn],
                )
                _validate_one_star_pending_response_routing(
                    ckpt,
                    actor_id=actor_id,
                    result=result,
                )
            except Exception:
                _rollback_one_star_materialized_spawns(ckpt, result)
                raise

        location_updates = {
            update.character_id: update.location_label
            for update in result.location_updates
        }
        if len(location_updates) != len(result.location_updates):
            _rollback_one_star_materialized_spawns(ckpt, result)
            raise OneStarTransactionError(
                "One-Star event contains duplicate location updates"
            )
        activation_locations = {
            signal.character_id: signal.location_label for signal in result.activate
        }
        if len(activation_locations) != len(result.activate):
            _rollback_one_star_materialized_spawns(ckpt, result)
            raise OneStarTransactionError(
                "One-Star event contains duplicate activation signals"
            )

        def prepare():
            transaction = _one_star_transaction_for_result(ckpt, result)
            if transaction is None:
                raise OneStarTransactionError(
                    "One-Star event has no state-update contract"
                )
            return prepare_one_star_transaction(
                ckpt,
                event_id=result.event_id,
                transaction=transaction,
                spawned_character_ids=(
                    request.character_id for request in result.spawn
                ),
                activated_character_ids=(
                    signal.character_id for signal in result.activate
                ),
                activated_character_locations=activation_locations,
                dormant_character_ids=result.dormant,
                generic_culled_character_ids=result.cull,
                location_updates=location_updates,
                canonical_at_s=(result.effective_at_s + result.duration_s),
                event_fingerprint=one_star_event_fingerprint(
                    {
                        "actor_id": actor_id,
                        "event": result.model_dump(mode="json"),
                    }
                ),
                initiating_actor_id=actor_id,
                authoritative_system_result=authoritative_system_result,
            )

        try:
            try:
                prepared = prepare()
            except OneStarTransactionError as first_error:
                if authoritative_system_result:
                    raise
                summon_signature = _one_star_summon_update_signature(
                    result.state_updates,
                )
                pending_resolution_count = _one_star_pending_resolution_count(
                    result.state_updates,
                )
                repaired = await self._repair_one_star_transaction(
                    ckpt=ckpt,
                    result=result,
                    actor_id=actor_id,
                    validation_error=str(first_error),
                )
                if (
                    _one_star_summon_update_signature(
                        repaired.state_updates,
                    )
                    != summon_signature
                ):
                    raise OneStarTransactionError(
                        "state-update repair cannot change a summon after its "
                        "Hero lifecycle has been fixed"
                    ) from first_error
                if (
                    _one_star_pending_resolution_count(
                        repaired.state_updates,
                    )
                    != pending_resolution_count
                ):
                    raise OneStarTransactionError(
                        "state-update repair cannot erase a pending resolution "
                        "from the fixed canonical event"
                    ) from first_error
                result.state_updates = repaired.state_updates
                if player_submission:
                    preflight_one_star_account_updates(
                        ckpt,
                        result.state_updates,
                        initiating_actor_id=actor_id,
                        canonical_at_s=(result.effective_at_s + result.duration_s),
                    )
                _include_one_star_synthesis_guide_responders(
                    ckpt,
                    actor_id=actor_id,
                    result=result,
                )
                _validate_one_star_cat_ii_transaction(ckpt, result)
                _validate_one_star_tutorial_routing(ckpt, result)
                _validate_one_star_pending_response_routing(
                    ckpt,
                    actor_id=actor_id,
                    result=result,
                )
                _validate_one_star_guide_routing(
                    ckpt,
                    actor_id=actor_id,
                    result=result,
                )
                try:
                    prepared = prepare()
                except OneStarTransactionError as second_error:
                    raise OneStarTransactionError(
                        "One-Star state updates remained invalid after one repair: "
                        f"{second_error}"
                    ) from second_error
        except Exception:
            _rollback_one_star_materialized_spawns(ckpt, result)
            raise

        if prepared.already_applied:
            raise OneStarTransactionError(
                "One-Star prepared event was already committed and cannot be "
                "broadcast a second time"
            )
        from app.engine.closed_event_runtime import closed_event_runtime_for

        runtime = closed_event_runtime_for(ckpt)
        live_characters = ckpt.characters
        try:
            _append_one_star_system_consequences(
                ckpt,
                result,
                prepared.system_consequences,
            )
            apply_one_star_prepared_mutation(ckpt, prepared)

            # Generic lifecycle signals remain the identity/status authority.
            # The private mutation validates their One-Star meaning, then the
            # existing roster helper makes activated reserves dispatchable in
            # this same beat. Reapplying these signals post-beat is idempotent.
            CharacterManager().apply_roster_updates(ckpt, result)

            # One-Star resource mutation and generated identities are one
            # atomic canonical commit. Accept the speculative identity roster
            # only after every durable character mutation succeeds.
            if runtime is not None and result.spawn:
                accepted = runtime.spawn_authoring.accept_roster(
                    checkpoint=ckpt,
                    transaction_id=runtime.transaction_id,
                )
                runtime.applied_character_ids.update(accepted)

            acquired_characters = [
                character
                for character in ckpt.characters
                if character.character_id in prepared.newly_acquired_hero_ids
            ]
            refresh_router_history_record(
                ckpt.session_conversation,
                result=result,
                one_star_engine_characters=acquired_characters,
                one_star_engine_updates=prepared.engine_history_updates,
                force=True,
            )
        except Exception:
            ckpt.characters = live_characters
            _rollback_one_star_materialized_spawns(ckpt, result)
            raise

        if prepared.culled_character_ids:
            from app.engine.turn_loop import purge_character_state

            for character_id in prepared.culled_character_ids:
                purge_character_state(
                    ckpt,
                    character_id,
                    preserve_render_buffer=True,
                )

    async def _repair_one_star_transaction(
        self,
        *,
        ckpt: CheckpointFile,
        result: OneStarEventRouterOutput | ClosedOneStarEventRouterOutput,
        actor_id: str,
        validation_error: str,
    ) -> OneStarStateUpdateList:
        """Ask once for an update-list-only correction to a fixed event."""

        ctx = _build_router_context(
            ckpt,
            actor_id,
            include_engine_state_updates=False,
        )
        ctx.pop("initial_roster_block", "")
        ctx.pop("engine_state_updates_block", "")
        from app.engine.one_star_router_context import (
            render_one_star_repair_evidence,
        )

        conflict_evidence = render_one_star_repair_evidence(
            ckpt,
            state_updates=result.state_updates,
            canonical_at_s=result.effective_at_s + result.duration_s,
        )
        lifecycle = {
            "spawn": [
                {
                    "character_id": request.character_id,
                    "knowledge_tier": request.seed.knowledge_tier,
                }
                for request in result.spawn
            ],
            "dormant": result.dormant,
            "cull": result.cull,
            "activate": [signal.character_id for signal in result.activate],
            "location_updates": {
                update.character_id: update.location_label
                for update in result.location_updates
            },
        }
        repair_block = (
            "<one_star_state_update_repair>\n"
            "The following event fields are fixed and are not output here. "
            "Return only a repaired state_updates list that agrees with the "
            "supplied current One-Star state and this validation failure. "
            "The reported failure is mandatory: change the offending field "
            "instead of repeating the candidate.\n"
            "Pending field invariants: deployment repeats participant details "
            "for the complete party and omits target_id; synthesis repeats "
            "source Heroes as participant details and names a distinct "
            "target_id; promotion uses the same single Hero as its sole "
            "participant and target_id.\n"
            f"{_ONE_STAR_COMPACT_UPDATE_AUTHORITY}\n"
            f"Submitting actor id: {actor_id}\n"
            f"Validation failure: {validation_error}\n"
            "Current conflicting state:\n"
            f"{conflict_evidence}\n"
            "Fixed canonical event:\n"
            f"{result.canonical_event.model_dump_json()}\n"
            "Candidate state updates:\n"
            f"{json.dumps([update.model_dump(mode='json') for update in result.state_updates], ensure_ascii=False, separators=(',', ':'))}\n"
            "Fixed generic lifecycle:\n"
            f"{json.dumps(lifecycle, ensure_ascii=False, separators=(',', ':'))}\n"
            "</one_star_state_update_repair>"
        )
        messages = self.prompt_mgr.render_conversation(
            "event_router",
            history=ckpt.session_conversation,
            **ctx,
            **_router_ruleset_template_vars(
                self.prompt_mgr,
                ruleset_id=ONE_STAR_RULESET_ID,
                dnd_fresh=False,
                ckpt=ckpt,
            ),
            router_input_block=repair_block,
        )
        response = await self.client.complete(
            role="event_router",
            messages=messages,
            response_model=OneStarStateUpdateList,
            temperature=0.2,
            max_tokens=EVENT_ROUTER_MAX_TOKENS,
            cache=True,
            compact=True,
        )
        return response.parsed

    async def _retry_one_star_routing_contract(
        self,
        *,
        messages: list[dict[str, object]],
        result: OneStarEventRouterOutput | ClosedOneStarEventRouterOutput,
        validation_error: str,
        response_model: (
            type[OneStarEventRouterOutput]
            | type[ClosedOneStarEventRouterOutput]
        ) = OneStarEventRouterOutput,
    ) -> OneStarEventRouterOutput | ClosedOneStarEventRouterOutput:
        """Ask once for a complete replacement of an invalid routing envelope.

        State-update repair is intentionally narrower and runs only after the
        event's fiction and routing have been accepted.  Response ownership,
        observer delivery, and reversible Cat II selection are coupled to the
        whole event, so correcting one of those failures requires a fresh full
        output before any candidate reaches canonical history or rules state.
        """

        correction_messages = [
            *messages,
            {
                "role": "assistant",
                "content": result.model_dump_json(),
            },
            {
                "role": "user",
                "content": (
                    "<router_output_correction>\n"
                    "Return one complete replacement output. The candidate "
                    "above violates this story's routing contract:\n"
                    f"{validation_error}\n"
                    "First reconsider whether the candidate operation belongs. "
                    "A read-only System or status inspection has an empty "
                    "One-Star state updates and no responders; never turn such "
                    "an inspection into Cat II merely to satisfy this error. "
                    "Preserve any compatible fictional judgment, but make the "
                    "canonical event, Cat II classification, responder set, "
                    "observers, lifecycle, and One-Star state updates mutually "
                    "consistent. "
                    f"{_ONE_STAR_COMPACT_UPDATE_AUTHORITY} "
                    "Do not discuss the correction in the fiction "
                    "or rationale.\n"
                    "</router_output_correction>"
                ),
            },
        ]
        logger.warning(
            "Retrying invalid One-Star router output once: %s",
            validation_error,
        )
        response = await self.client.complete(
            role="event_router",
            messages=correction_messages,
            response_model=response_model,
            temperature=0.2,
            max_tokens=EVENT_ROUTER_MAX_TOKENS,
            cache=True,
            compact=True,
        )
        return response.parsed

    # ------------------------------------------------------------------
    # route_intention
    # ------------------------------------------------------------------

    async def route_intention(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        intention: str,
        cat_ii_event: OpenCatIIEvent | None = None,
    ) -> EventRouterOutput:
        """Classify + adjudicate one intention through event_router."""

        if cat_ii_event is not None and dnd_cat_ii_router_enabled(ckpt):
            logger.info(
                "LLMDispatcher.route_intention: actor=%s cat_ii=%s "
                "using dnd_cat_ii_router",
                actor_id,
                cat_ii_event.event_id,
            )
            dnd_snapshot = _router_call_snapshot(ckpt)
            try:
                content_context_records = (
                    []
                    if _transaction_waits_for_player_rolls(
                        ckpt,
                        cat_ii_event.event_id,
                    )
                    else _append_pending_router_content_records(ckpt)
                )
                result = await self._dnd_cat_ii.resolve_cat_ii(
                    ckpt=ckpt,
                    cat_ii_event=cat_ii_event,
                    content_context_records=content_context_records,
                )
            except DndCatIIRollsPending:
                raise
            except Exception:
                _restore_router_call_snapshot(ckpt, dnd_snapshot)
                raise
            _normalize_router_result_for_history(
                ckpt,
                result=result,
                clock_anchor_character_id=actor_id,
                cat_ii_event=cat_ii_event,
            )
            _append_router_history_record(
                ckpt.session_conversation,
                acting_character_id=actor_id,
                result=result,
                mode="cat_ii_resolution",
            )
            return result

        router_snapshot = _router_call_snapshot(ckpt)
        try:
            await _append_router_content_lookup_records(
                ckpt,
                actor_id=actor_id,
                current_input=intention,
                client=self.client,
                prompt_mgr=self.prompt_mgr,
            )
            ctx = _build_router_context(
                ckpt,
                actor_id,
            )
            initial_roster_record = ctx.pop("initial_roster_block", "")
            if initial_roster_record:
                ckpt.session_conversation.append(
                    ConversationMessage(
                        role="assistant",
                        content=initial_roster_record,
                    )
                )
            dnd_fresh = _dnd_fresh_router_enabled(ckpt, cat_ii_event)

            if cat_ii_event is None:
                intention_block = format_actor_submission(
                    actor_id,
                    intention,
                )
                cat_ii_resolution_block = ""
            else:
                evt = cat_ii_event
                responders: list[tuple[str, str]] = [
                    (rid, evt.collected_intentions[rid])
                    for rid in evt.required_responders
                    if rid in evt.collected_intentions
                ]
                cat_ii_resolution_block = format_cat_ii_resolution_block(
                    initiator_id=evt.initiator_id,
                    initiator_intention=evt.initiator_intention,
                    responders=responders,
                    swept_responders=list(evt.swept_responders),
                )
                intention_block = ""

            router_input_block = _build_router_input_block(
                _build_opening_context_block(ckpt, intention, actor_id),
                ctx.pop("engine_state_updates_block", ""),
                cat_ii_resolution_block,
                intention_block,
            )
            template_vars = {
                **ctx,
                **_router_ruleset_template_vars(
                    self.prompt_mgr,
                    ruleset_id=_session_ruleset_id(ckpt),
                    dnd_fresh=dnd_fresh,
                    ckpt=ckpt,
                ),
                "router_input_block": router_input_block,
            }

            # Use render_conversation so the rolling router ledger rides
            # along, and append this turn's exchange after the call so
            # continuity compounds across turns. The caller
            # (Orchestrator) persists the checkpoint after run_beat returns.
            messages = self.prompt_mgr.render_conversation(
                "event_router",
                history=ckpt.session_conversation,
                **template_vars,
            )

            logger.info(
                "LLMDispatcher.route_intention: actor=%s cat_ii=%s",
                actor_id,
                cat_ii_event.event_id if cat_ii_event else None,
            )

            response = await self.client.complete(
                role="event_router",
                messages=messages,
                response_model=(
                    OneStarEventRouterOutput
                    if _one_star_router_enabled(ckpt)
                    else DndEventRouterOutput
                    if dnd_fresh
                    else EventRouterOutput
                ),
                temperature=0.35,
                max_tokens=EVENT_ROUTER_MAX_TOKENS,
                cache=True,
                compact=True,
            )
            result: EventRouterOutput = response.parsed
            from app.engine.one_star_adapter import OneStarTransactionError

            def validate_candidate() -> None:
                _include_one_star_synthesis_guide_responders(
                    ckpt,
                    actor_id=actor_id,
                    result=result,
                )
                _validate_one_star_initial_deployment_responder_order(
                    ckpt,
                    result,
                    actor_id=actor_id,
                    cat_ii_event=cat_ii_event,
                )
                _validate_one_star_cat_ii_transaction(ckpt, result)
                _validate_one_star_pending_operation_shapes(ckpt, result)
                _validate_one_star_initial_deployment_guide_handoff(
                    ckpt,
                    result,
                    cat_ii_event=cat_ii_event,
                )
                _validate_one_star_pending_resolution_event_contract(
                    ckpt,
                    result,
                )
                _validate_one_star_tutorial_routing(ckpt, result)
                _validate_one_star_pending_response_routing(
                    ckpt,
                    actor_id=actor_id,
                    result=result,
                )
                _validate_one_star_guide_and_opening_envelope(
                    ckpt,
                    actor_id=actor_id,
                    intention=intention,
                    result=result,
                )
                _validate_opening_spawn_authority(ckpt, intention, result)

            try:
                validate_candidate()
            except OneStarTransactionError as first_error:
                repaired = await self._repair_one_star_transaction(
                    ckpt=ckpt,
                    result=result,
                    actor_id=actor_id,
                    validation_error=str(first_error),
                )
                result.state_updates = repaired.state_updates
                try:
                    validate_candidate()
                except ValueError as second_error:
                    result = await self._retry_one_star_routing_contract(
                        messages=messages,
                        result=result,
                        validation_error=str(second_error),
                    )
                    try:
                        validate_candidate()
                    except ValueError as final_error:
                        raise ValueError(
                            "One-Star router output remained invalid after state "
                            f"repair and routing correction: {final_error}"
                        ) from final_error
            except ValueError as first_error:
                if not _one_star_router_enabled(ckpt):
                    raise
                result = await self._retry_one_star_routing_contract(
                    messages=messages,
                    result=result,
                    validation_error=str(first_error),
                )
                try:
                    validate_candidate()
                except ValueError as second_error:
                    raise ValueError(
                        "One-Star router output remained invalid after one "
                        f"correction: {second_error}"
                    ) from second_error
        except Exception:
            _restore_router_call_snapshot(ckpt, router_snapshot)
            raise

        _normalize_router_result_for_history(
            ckpt,
            result=result,
            clock_anchor_character_id=actor_id,
            cat_ii_event=cat_ii_event,
        )
        # Persist only compact canonical memory. The current turn's raw
        # input already shaped this result; replaying it beside the result
        # duplicates context and delays cache benefit.
        _append_router_history_record(
            ckpt.session_conversation,
            acting_character_id=actor_id,
            result=result,
            mode="cat_ii_resolution" if cat_ii_event else "intention",
            user_prompt=(
                _defer_history_user_prompt(intention) if cat_ii_event is None else ""
            ),
        )
        return result

    async def route_combat_action(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        intention: str,
    ) -> EventRouterOutput:
        """Resolve one active D&D combat turn through the ruleset adapter."""
        if not dnd_combat_manager_enabled(ckpt):
            raise RuntimeError(
                "D&D combat routing requested outside active D&D combat."
            )
        logger.info(
            "LLMDispatcher.route_combat_action: actor=%s using dnd_combat_manager",
            actor_id,
        )
        dnd_snapshot = _router_call_snapshot(ckpt)
        try:
            content_context_records = _append_pending_router_content_records(ckpt)
            result = await self._dnd_combat.resolve_combat_action(
                ckpt=ckpt,
                actor_id=actor_id,
                intention=intention,
                content_context_records=content_context_records,
            )
        except DndCatIIRollsPending:
            raise
        except Exception:
            _restore_router_call_snapshot(ckpt, dnd_snapshot)
            raise
        _normalize_router_result_for_history(
            ckpt,
            result=result,
            clock_anchor_character_id=actor_id,
        )
        if ckpt.session.active_combat is None:
            _append_router_history_record(
                ckpt.session_conversation,
                acting_character_id=actor_id,
                result=result,
                mode="dnd_combat_end",
            )
        return result

    async def continue_combat_transaction(
        self,
        *,
        ckpt: CheckpointFile,
        event_id: str,
    ) -> EventRouterOutput:
        if not dnd_combat_manager_enabled(ckpt):
            raise RuntimeError(
                "D&D combat roll continuation requested outside active D&D combat."
            )
        logger.info(
            "LLMDispatcher.continue_combat_transaction: event=%s",
            event_id,
        )
        dnd_snapshot = _router_call_snapshot(ckpt)
        try:
            content_context_records = (
                []
                if _transaction_waits_for_player_rolls(
                    ckpt,
                    event_id,
                    source="combat",
                )
                else _append_pending_router_content_records(ckpt)
            )
            result = await self._dnd_combat.continue_combat_transaction(
                ckpt=ckpt,
                event_id=event_id,
                content_context_records=content_context_records,
            )
        except DndCatIIRollsPending:
            raise
        except Exception:
            _restore_router_call_snapshot(ckpt, dnd_snapshot)
            raise
        actor_id = _roll_transaction_actor_id(ckpt, event_id)
        _normalize_router_result_for_history(
            ckpt,
            result=result,
            clock_anchor_character_id=actor_id,
        )
        if ckpt.session.active_combat is None:
            _append_router_history_record(
                ckpt.session_conversation,
                acting_character_id=actor_id,
                result=result,
                mode="dnd_combat_end",
            )
        return result

    # ------------------------------------------------------------------
    # route_continuation
    # ------------------------------------------------------------------

    async def route_continuation(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        prior_result: EventRouterOutput,
        original_action: str = "",
    ) -> EventRouterOutput:
        """Ask the router for another event after a narrator continue handoff."""

        router_snapshot = _router_call_snapshot(ckpt)
        try:
            await _append_router_content_lookup_records(
                ckpt,
                actor_id=actor_id,
                current_input=prior_result.decision_rationale,
                client=self.client,
                prompt_mgr=self.prompt_mgr,
            )
            ctx = _build_router_context(
                ckpt,
                actor_id,
            )
            initial_roster_record = ctx.pop("initial_roster_block", "")
            if initial_roster_record:
                ckpt.session_conversation.append(
                    ConversationMessage(
                        role="assistant",
                        content=initial_roster_record,
                    )
                )
            continuation_block = format_router_continuation_block(
                prior_rationale=prior_result.decision_rationale,
                original_action=original_action,
            )

            router_input_block = _build_router_input_block(
                ctx.pop("engine_state_updates_block", ""),
                continuation_block,
            )
            template_vars = {
                **ctx,
                **_router_ruleset_template_vars(
                    self.prompt_mgr,
                    ruleset_id=_session_ruleset_id(ckpt),
                    dnd_fresh=False,
                    ckpt=ckpt,
                ),
                "router_input_block": router_input_block,
            }

            messages = self.prompt_mgr.render_conversation(
                "event_router",
                history=ckpt.session_conversation,
                **template_vars,
            )

            logger.info(
                "LLMDispatcher.route_continuation: actor=%s",
                actor_id,
            )

            response = await self.client.complete(
                role="event_router",
                messages=messages,
                response_model=(
                    ClosedOneStarEventRouterOutput
                    if _one_star_router_enabled(ckpt)
                    else ClosedEventRouterOutput
                ),
                temperature=0.35,
                max_tokens=EVENT_ROUTER_MAX_TOKENS,
                cache=True,
                compact=True,
            )
            result: EventRouterOutput = response.parsed

            def validate_candidate() -> None:
                if result.requires_responders or result.required_responders:
                    raise ValueError(
                        "One-Star continuation cannot open a new Cat II event"
                    )
                _include_one_star_synthesis_guide_responders(
                    ckpt,
                    actor_id=actor_id,
                    result=result,
                )
                _validate_one_star_cat_ii_transaction(ckpt, result)
                _validate_one_star_pending_operation_shapes(ckpt, result)
                _validate_one_star_pending_resolution_event_contract(
                    ckpt,
                    result,
                )
                _validate_one_star_tutorial_routing(ckpt, result)
                _validate_one_star_pending_response_routing(
                    ckpt,
                    actor_id=actor_id,
                    result=result,
                )
                _validate_one_star_guide_routing(
                    ckpt,
                    actor_id=actor_id,
                    result=result,
                )

            try:
                validate_candidate()
            except ValueError as first_error:
                if not _one_star_router_enabled(ckpt):
                    raise
                result = await self._retry_one_star_routing_contract(
                    messages=messages,
                    result=result,
                    validation_error=str(first_error),
                    response_model=ClosedOneStarEventRouterOutput,
                )
                try:
                    validate_candidate()
                except ValueError as second_error:
                    raise ValueError(
                        "One-Star continuation output remained invalid after "
                        f"one correction: {second_error}"
                    ) from second_error

            _normalize_router_result_for_history(
                ckpt,
                result=result,
                clock_anchor_character_id=actor_id,
            )
            _append_router_history_record(
                ckpt.session_conversation,
                acting_character_id=actor_id,
                result=result,
                mode="continuation",
            )
            return result
        except Exception:
            _restore_router_call_snapshot(ckpt, router_snapshot)
            raise

    # ------------------------------------------------------------------
    # route_authoritative_result
    # ------------------------------------------------------------------

    async def route_authoritative_result(
        self,
        *,
        ckpt: CheckpointFile,
        plan: AuthoritativeResultPlan,
        character_contributions: Sequence[tuple[str, str]] = (),
    ) -> EventRouterOutput:
        """Canonicalize fixed fictional authority without re-adjudicating it."""

        router_snapshot = _router_call_snapshot(ckpt)
        try:
            ctx = _build_router_context(
                ckpt,
                "-",
                resolve_actor_fallback=False,
                include_engine_state_updates=False,
            )
            initial_roster_record = ctx.pop("initial_roster_block", "")
            if initial_roster_record:
                ckpt.session_conversation.append(
                    ConversationMessage(
                        role="assistant",
                        content=initial_roster_record,
                    )
                )
            ctx.pop("engine_state_updates_block", "")
            authoritative_block = format_authoritative_result_block(
                plan,
                character_contributions=character_contributions,
            )
            messages = self.prompt_mgr.render_conversation(
                "event_router",
                history=ckpt.session_conversation,
                **ctx,
                **_router_ruleset_template_vars(
                    self.prompt_mgr,
                    ruleset_id=_session_ruleset_id(ckpt),
                    dnd_fresh=False,
                    ckpt=ckpt,
                ),
                router_input_block=authoritative_block,
            )
            logger.info(
                "LLMDispatcher.route_authoritative_result: authority=%s",
                plan.authority_label,
            )
            response = await self.client.complete(
                role="event_router",
                messages=messages,
                response_model=ClosedEventRouterOutput,
                temperature=0.2,
                max_tokens=EVENT_ROUTER_MAX_TOKENS,
                cache=True,
                compact=True,
            )
            routed: ClosedEventRouterOutput = response.parsed
            if not routed.canonical_event.world_adjudication.feasible:
                raise ValueError(
                    "authoritative result was incorrectly rejected as infeasible"
                )
            if (
                routed.spawn
                or routed.dormant
                or routed.cull
                or routed.activate
                or routed.commitment_open.present
                or routed.commitment_resolutions
                or routed.commitment_interrupts
            ):
                raise ValueError(
                    "authoritative result router output attempted to author "
                    "fixed side effects"
                )

            fixed_locations: dict[str, str] = {}
            for character_id, location in plan.location_updates:
                if character_id in fixed_locations:
                    raise ValueError(
                        "authoritative result contains duplicate location updates"
                    )
                fixed_locations[character_id] = location
            routed_location_ids: set[str] = set()
            for update in routed.location_updates:
                if (
                    update.character_id in routed_location_ids
                    or fixed_locations.get(update.character_id) != update.location_label
                ):
                    raise ValueError(
                        "authoritative result router output attempted to author "
                        "fixed side effects"
                    )
                routed_location_ids.add(update.character_id)

            location_updates = [
                LocationUpdateSignal(
                    character_id=character_id,
                    location_label=location,
                )
                for character_id, location in plan.location_updates
            ]
            payload = routed.model_dump(mode="python")
            payload.update(
                {
                    "event_kind": "ruleset_resolution",
                    "requires_responders": False,
                    "required_responders": [],
                    "location_updates": location_updates,
                }
            )
            if plan.state_updates:
                if not _one_star_router_enabled(ckpt):
                    raise ValueError(
                        "authoritative state updates require the One-Star ruleset"
                    )
                payload["state_updates"] = [
                    OneStarStateUpdate.model_validate(dict(update))
                    for update in plan.state_updates
                ]
                result: EventRouterOutput = (
                    ClosedOneStarEventRouterOutput.model_validate(payload)
                )
            else:
                result = ClosedEventRouterOutput.model_validate(payload)
            result.clear_routing_roles()

            viewpoint_observer = next(
                (
                    observer
                    for observer in result.observers
                    if observer.character_id == plan.viewpoint_character_id
                ),
                None,
            )
            if (
                viewpoint_observer is None
                or viewpoint_observer.observation_level != "d"
                or not any(
                    fact.is_visible_to(plan.viewpoint_character_id)
                    for fact in result.canonical_event.observable_facts
                )
            ):
                raise ValueError(
                    "authoritative result omitted the requesting viewpoint's "
                    "direct visible event"
                )
            observers_by_id = {
                observer.character_id: observer for observer in result.observers
            }
            for character_id, _location in plan.location_updates:
                observer = observers_by_id.get(character_id)
                if (
                    observer is None
                    or observer.observation_level != "d"
                    or not any(
                        fact.is_visible_to(character_id)
                        for fact in result.canonical_event.observable_facts
                    )
                ):
                    raise ValueError(
                        "authoritative result omitted a directly affected "
                        f"character: {character_id!r}"
                    )
        except Exception:
            _restore_router_call_snapshot(ckpt, router_snapshot)
            raise

        _normalize_router_result_for_history(ckpt, result=result)
        _append_router_history_record(
            ckpt.session_conversation,
            acting_character_id="-",
            result=result,
            mode="authoritative_result",
        )
        return result

    async def draft_authoritative_contributions(
        self,
        *,
        ckpt: CheckpointFile,
        requests: Sequence[AuthoritativeContributionRequest],
        location_updates: Sequence[tuple[str, str]] = (),
    ) -> list[tuple[str, CharacterAgentTurnDraft]]:
        """Draft independent character contributions without mutating state."""

        request_ids = [request.character_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("authoritative contribution requests must be unique")
        shadow = CheckpointFile.model_validate(
            ckpt.model_dump(
                mode="python",
                context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
            )
        )
        shadow_by_id = {
            character.character_id: character for character in shadow.characters
        }
        for character_id, location in location_updates:
            character = shadow_by_id.get(character_id)
            if character is None:
                raise ValueError(
                    "authoritative location update references unknown character "
                    f"{character_id!r}"
                )
            character.location = location

        async def _draft(
            request: AuthoritativeContributionRequest,
        ) -> tuple[str, CharacterAgentTurnDraft]:
            character = shadow_by_id.get(request.character_id)
            if character is None:
                raise ValueError(
                    "authoritative contribution references unknown character "
                    f"{request.character_id!r}"
                )
            draft = await self._agent.draft_turn(
                character=character,
                checkpoint=shadow,
                frame="foreground",
                local_context=request.local_context,
            )
            return request.character_id, draft

        return list(await asyncio.gather(*(_draft(request) for request in requests)))

    def commit_authoritative_contributions(
        self,
        *,
        ckpt: CheckpointFile,
        drafts: Sequence[tuple[str, CharacterAgentTurnDraft]],
    ) -> None:
        """Commit drafts only after the fixed event passes router/rules checks."""

        characters = {
            character.character_id: character for character in ckpt.characters
        }
        for character_id, draft in drafts:
            character = characters.get(character_id)
            if character is None:
                raise ValueError(
                    "authoritative contribution character disappeared before "
                    f"commit: {character_id!r}"
                )
            self._agent.commit_draft(character, ckpt, draft)

    # ------------------------------------------------------------------
    # agent_intend
    # ------------------------------------------------------------------

    async def agent_intend(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        frame: str = "foreground",
        local_context: str = "",
    ) -> str:
        """Invoke the character agent and return its prose for the router.

        The agent's prose IS the intention — no serialization layer
        needed. The trailing parenthetical (private intent) is stripped
        at parse time; what we return here is the public surface only,
        which is what the router reads as the acting character's intention.

        Three result shapes the caller (run_beat) must distinguish:
          - **non-empty prose** → real intention, route normally.
          - **`"(remains silent)"`** → the agent had a non-empty intent
            parenthetical but emitted no public prose (the agent
            prompt's "Sparse is valid" shared rule — paren-only
            output is in-character). The cascade MUST treat this as
            a real beat and route it; otherwise the prompt's promise
            that silence is a valid in-character choice gets quietly
            broken by `_is_agent_refusal` collapsing it to
            `cascade_exhausted`.
          - **`""`** → true refusal: no public prose AND no intent (or
            the parser logged a "missing trailing parenthetical"
            warning and we have nothing to route). The cascade ends.
        """
        character = next(
            (c for c in ckpt.characters if c.character_id == character_id),
            None,
        )
        if character is None:
            logger.warning(
                "agent_intend: unknown character_id %s",
                character_id,
            )
            return ""
        from app.engine.context_builder import is_unbound_player_authored_slot

        if is_unbound_player_authored_slot(ckpt, character):
            raise RuntimeError(
                "Cannot invoke a character agent for an unclaimed "
                f"player-authored seat: {character_id}"
            )

        output = await self._agent.turn(
            character=character,
            checkpoint=ckpt,
            frame=frame,
            local_context=local_context,
        )
        public = output.public_text.strip()
        if public:
            return public
        if output.intent.strip():
            # Agent chose deliberate silence (paren-only output). Surface
            # a fixed sentinel so the router can adjudicate a "watches
            # without speaking" beat instead of the cascade dying. The
            # sentinel is intentionally short, parenthesized, and
            # identical every time so the router can recognize it.
            logger.info(
                "Agent %s emitted silent beat (intent=%d chars); routing via sentinel.",
                character.name,
                len(output.intent),
            )
            return "(remains silent)"
        return ""

    # ------------------------------------------------------------------
    # harvest_perceptions  (v11-r8a: observation_harvest fork)
    # ------------------------------------------------------------------

    async def harvest_perceptions(
        self,
        *,
        ckpt: CheckpointFile,
        character_ids: list[str],
    ) -> list[str]:
        """Fan out CharacterAgent.perceive() across `character_ids` in
        parallel.

        Returns one string per id in input order — empty string for any
        character whose perception call failed (unknown id, LLM error,
        empty output). The harvest fork in `run_beat` filters empties
        and appends the non-empty fragments to the canonical event's
        `observable_facts` block.

        Per-character exceptions are absorbed locally rather than
        bubbled. The harvest is a UX enrichment, not the beat's
        critical path; one failed perception out of three should
        leave the other two on the player's screen instead of taking
        the whole render down. The caller logs dropped fragments at
        WARN so test playthroughs still surface the failure.

        Cache lineage: every `perceive` call shares the SAME system
        prompt as normal agent turns under the same ruleset (single
        unified `agent` template). Character identity lives in the
        per-call user message, so parallel fan-out compounds well with
        this — a 3-character harvest bills three Luna calls in roughly
        one round-trip wall time, all hitting the cached system prefix.
        """
        if not character_ids:
            return []

        by_id = {c.character_id: c for c in ckpt.characters}
        from app.engine.context_builder import is_unbound_player_authored_slot

        absent_ids = [
            character_id
            for character_id in character_ids
            if is_unbound_player_authored_slot(ckpt, by_id.get(character_id))
        ]
        if absent_ids:
            raise RuntimeError(
                "Cannot harvest perceptions for unclaimed player-authored "
                "seats: " + ", ".join(absent_ids)
            )

        async def _one(cid: str) -> str:
            character = by_id.get(cid)
            if character is None:
                logger.warning(
                    "harvest_perceptions: unknown character_id %s",
                    cid,
                )
                return ""
            try:
                result = await self._agent.perceive(
                    character=character,
                    checkpoint=ckpt,
                )
                return result.public_text
            except Exception as exc:  # noqa: BLE001 — see docstring
                logger.warning(
                    "harvest_perceptions: perceive() failed for %s: %s",
                    cid,
                    exc,
                )
                return ""

        logger.info(
            "harvest_perceptions: firing %d parallel perceive calls",
            len(character_ids),
        )
        return list(await asyncio.gather(*(_one(c) for c in character_ids)))

    # ------------------------------------------------------------------
    # narrator_compose
    # ------------------------------------------------------------------

    async def narrator_compose(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        buffered_events: list[RenderBufferEntry],
        partial_mode_override: bool | None = None,
        user_input: str = "",
        handoff_policy: str = "forced",
        handoff_context: str = "",
    ) -> tuple[NarratorOutput, "TranscriptEntry"]:
        """Render per-POV prose via narrator.compose_pov_render.

        Returns `(NarratorFinalOutput, TranscriptEntry)` so run_beat can
        populate the parallel BeatResult.transcript_entries response map. The
        transient entry is
        constructed engine-side from `user_input` (the real player
        utterance for the acting POV; "" for incidental POVs in a
        multi-human beat) and the rendered prose. Pre-r7j the LLM
        owned the transcript entry and emitted `"{name} — "` for the
        user field every time.

        `partial_mode` defaults to True iff this character is currently
        pinned as a Cat II responder in the current beat — the narrator renders
        a partial view because the beat still has outstanding resolution
        work. `partial_mode_override`, when not None, wins over the slot
        scan (v11-r6a: Cat II-open render path sets this True for the
        initiator + pinned humans so they see the mid-attempt cliffhanger
        even though the initiator isn't pinned themselves).
        """
        if partial_mode_override is not None:
            partial_mode = partial_mode_override
        else:
            partial_mode = _is_pinned_as_cat_ii_responder(ckpt, character_id)

        envelope, entry = await narrator_module.compose_pov_render(
            client=self.client,
            prompt_mgr=self.prompt_mgr,
            ckpt=ckpt,
            pov_character_id=character_id,
            buffered_events=buffered_events,
            partial_mode=partial_mode,
            user_input=user_input,
            handoff_policy=handoff_policy,
            handoff_context=handoff_context,
        )
        return envelope, entry


def _is_pinned_as_cat_ii_responder(
    ckpt: CheckpointFile,
    character_id: str,
) -> bool:
    """True iff `character_id` is pinned as a Cat II responder this beat."""
    entry = ckpt.session.active_act_slots.get(character_id)
    return bool(entry is not None and entry.reason == "cat_ii_responder")
