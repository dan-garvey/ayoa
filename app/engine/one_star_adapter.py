"""Pure ledger preparation for the opt-in One-Star Ascension adapter.

The router still arbitrates fiction. This module translates its one compact
state-update list into private typed bookkeeping, validates it, and applies it
atomically. Weighted and authored opening summons are resolved here without
exposing identities or future draws to the router. The adapter deliberately
has no combat resolver, story id, or facility/economy constants.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping

from pydantic import ValidationError

from app.engine.action_rejection import PlayerActionRejected
from app.engine.one_star_progression import (
    apply_experience,
    apply_promotion_banked_experience,
    experience_to_reach_level,
    preview_experience,
    rebalance_hero,
    scaled_by_grade,
)
from app.schemas.characters import (
    ActorFact,
    ActorFactOrigin,
    ActorRecord,
    CharacterRecord,
    CharacterStatus,
    is_player_authored_slot,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.event_router import (
    EventRouterOutput,
    ObserverEntry,
    SpawnRequest,
    WakeSignal,
)
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_COMBATANT_KEY,
    ONE_STAR_GACHA_WEIGHT_TOTAL,
    ONE_STAR_HERO_KEY,
    ONE_STAR_RULESET_ID,
    OneStarAccountEnvelope,
    OneStarAccountState,
    OneStarCatalogueApplyOperation,
    OneStarCombatantState,
    OneStarCost,
    OneStarEquipmentEntry,
    OneStarGemPurchaseConfig,
    OneStarGemPurchaseOperation,
    OneStarHeroDeltaOperation,
    OneStarHeroState,
    OneStarInventoryDeltaOperation,
    OneStarMissionCounter,
    OneStarMissionEndOperation,
    OneStarMissionStartOperation,
    OneStarMissionUpdateOperation,
    OneStarOperation,
    OneStarOpeningRosterBoundPlayerActorSlot,
    OneStarOpeningRosterFixedSlot,
    OneStarOpeningRosterSummonPool,
    OneStarPendingOperation,
    OneStarPendingOperationSelection,
    OneStarPendingCancelOperation,
    OneStarPendingOpenOperation,
    OneStarPendingResolveOperation,
    OneStarResources,
    OneStarRulesConfig,
    OneStarSkillEntry,
    OneStarSkillRankUpdate,
    OneStarStateUpdate,
    OneStarStandardSummonPool,
    OneStarSynthesisPreview,
    OneStarSummonOperation,
    OneStarTransaction,
    OneStarTutorialDeliveryOperation,
    OneStarDurabilityUpdate,
    OneStarEquipmentMoveOperation,
)

class OneStarTransactionError(ValueError):
    """A compact One-Star update cannot be committed safely."""


_ACCOUNT_RESOURCE_FIELDS = frozenset({
    "gold",
    "gems",
    "building_resources",
})


def validate_one_star_pending_operation_shape(pending: object) -> None:
    """Validate kind-dependent pending fields without reading mutable state."""

    kind = getattr(pending, "kind", "")
    participant_ids = set(getattr(pending, "participant_ids", ()))
    target_id = str(getattr(pending, "target_id", "") or "")
    if kind == "deployment" and target_id:
        raise OneStarTransactionError("deployment has no separate target Hero")
    if kind == "synthesis":
        if not target_id:
            raise OneStarTransactionError("synthesis requires a target Hero")
        if target_id in participant_ids:
            raise OneStarTransactionError(
                "synthesis sources cannot include the target"
            )
    if kind == "promotion":
        if not target_id:
            raise OneStarTransactionError("promotion requires a target Hero")
        if participant_ids != {target_id}:
            raise OneStarTransactionError(
                "promotion participants must contain exactly the target Hero"
            )


def _durable_checkpoint_copy(checkpoint: CheckpointFile) -> CheckpointFile:
    """Copy durable checkpoint fields without traversing live runtime objects."""

    return CheckpointFile.model_validate_json(checkpoint.model_dump_json(
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
    ))


@dataclass(frozen=True, slots=True)
class OneStarPreparedMutation:
    """A fully validated after-checkpoint ready for one deterministic apply.

    ``hero_initializations`` reports the exact mechanics attached to generic
    spawn/activation records.  Generic records must be staged before prepare;
    the prepared checkpoint already carries these mechanics directly.
    """

    event_id: str
    event_fingerprint: str
    after_checkpoint: CheckpointFile
    hero_initializations: dict[str, OneStarHeroState] = field(default_factory=dict)
    culled_character_ids: tuple[str, ...] = ()
    newly_acquired_hero_ids: tuple[str, ...] = ()
    touched_hero_ids: tuple[str, ...] = ()
    engine_history_updates: tuple[str, ...] = ()
    system_consequences: tuple["OneStarSystemConsequence", ...] = ()
    already_applied: bool = False


@dataclass(frozen=True, slots=True)
class OneStarSystemConsequence:
    """A compact adapter-owned System fact for the just-committed event."""

    text: str
    recipient_character_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OneStarSummonDraw:
    """One authoritative slot in the next unconsumed standard-pool draw."""

    slot: int
    birth_stars: int
    existing_character_id: str = ""


def is_one_star_checkpoint(checkpoint: CheckpointFile) -> bool:
    return checkpoint.session.config.settings.ruleset_id == ONE_STAR_RULESET_ID


def one_star_active_mission_has_bound_party_member(
    checkpoint: CheckpointFile,
) -> bool:
    """Return whether a live One-Star mission includes a human-held Hero."""

    if not is_one_star_checkpoint(checkpoint):
        return False
    try:
        _owner, account = load_one_star_account(checkpoint)
    except (OneStarTransactionError, ValidationError):
        return False
    mission = account.state.active_mission
    return bool(
        mission is not None
        and set(mission.party_ids) & set(checkpoint.session.character_bindings)
    )


def one_star_master_has_human_led_mission(
    checkpoint: CheckpointFile,
    *,
    actor_id: str,
) -> bool:
    """Return whether this Master turn must stay off a human-led floor."""

    if not is_one_star_checkpoint(checkpoint):
        return False
    try:
        owner, account = load_one_star_account(checkpoint)
    except (OneStarTransactionError, ValidationError):
        return False
    mission = account.state.active_mission
    return bool(
        mission is not None
        and actor_id.strip() == owner.character_id
        and set(mission.party_ids) & set(checkpoint.session.character_bindings)
    )


def one_star_master_may_act_while_mission_responder_pinned(
    checkpoint: CheckpointFile,
    *,
    actor_id: str,
) -> bool:
    """Admit the Master beside only live human mission-response pins."""

    if not one_star_master_has_human_led_mission(
        checkpoint,
        actor_id=actor_id,
    ):
        return False
    try:
        _owner, account = load_one_star_account(checkpoint)
    except (OneStarTransactionError, ValidationError):
        return False
    mission = account.state.active_mission
    if mission is None:  # Guarded above; retain a narrow type-safe boundary.
        return False
    conflicting_ids = set(checkpoint.session.active_act_slots) - {actor_id}
    if not conflicting_ids:
        return False
    bindings = checkpoint.session.character_bindings
    open_events = {
        event.event_id: event for event in checkpoint.session.open_cat_ii_events
    }
    for character_id in conflicting_ids:
        slot = checkpoint.session.active_act_slots[character_id]
        open_event = open_events.get(slot.cat_ii_event_id or "")
        if (
            character_id not in mission.party_ids
            or character_id not in bindings
            or slot.reason != "cat_ii_responder"
            or open_event is None
            or character_id not in open_event.required_responders
            or character_id in open_event.collected_intentions
        ):
            return False
    return True


def _one_star_lobby_safe_locations(
    config: OneStarRulesConfig,
) -> set[str]:
    return {
        config.lobby_location_label,
        *(
            requirement.required_location
            for kind, requirement in config.operation_requirements.items()
            if kind != "deployment" and requirement.required_location
        ),
    }


def _one_star_mission_watch_requested(user_input: str) -> bool:
    normalized = " ".join((user_input or "").lower().split())
    if normalized in {"(defer)", "/defer", "defer"}:
        return True
    words = set(re.findall(r"[a-z]+", normalized))
    return bool(
        words
        & {
            "mission",
            "tower",
            "floor",
            "deployed",
            "deployment",
            "party",
        }
        and words & {"watch", "observe", "monitor", "view", "status", "check"}
    )


def _one_star_human_led_lobby_hero_is_safe(
    checkpoint: CheckpointFile,
    *,
    character_id: str,
    account: OneStarAccountEnvelope,
    allowed_locations: set[str],
) -> bool:
    mission = account.state.active_mission
    if mission is None or character_id in mission.party_ids:
        return False
    character = next(
        (
            candidate
            for candidate in checkpoint.characters
            if candidate.character_id == character_id
        ),
        None,
    )
    hero = load_one_star_hero(character) if character is not None else None
    return bool(
        character is not None
        and character.status == CharacterStatus.active
        and hero is not None
        and hero.owner_lobby_id == account.config.lobby_id
        and character.location in allowed_locations
    )


def _one_star_human_led_lobby_guide_is_safe(
    checkpoint: CheckpointFile,
    *,
    character_id: str,
    account: OneStarAccountEnvelope,
    allowed_locations: set[str],
) -> bool:
    mission = account.state.active_mission
    if (
        mission is None
        or character_id in mission.party_ids
        or character_id not in account.state.guide_character_ids
    ):
        return False
    character = next(
        (
            candidate
            for candidate in checkpoint.characters
            if candidate.character_id == character_id
        ),
        None,
    )
    return bool(
        character is not None
        and character.status == CharacterStatus.active
        and character.location in allowed_locations
    )


def _one_star_human_led_lobby_recipient_is_safe(
    checkpoint: CheckpointFile,
    *,
    character_id: str,
    account: OneStarAccountEnvelope,
    allowed_locations: set[str],
) -> bool:
    mission = account.state.active_mission
    if mission is None or character_id in mission.party_ids:
        return False
    character = next(
        (
            candidate
            for candidate in checkpoint.characters
            if candidate.character_id == character_id
        ),
        None,
    )
    return bool(
        character is not None
        and character.status == CharacterStatus.active
        and character.location in allowed_locations
    )


def _one_star_human_led_watch_query_is_safe(
    result: EventRouterOutput,
    *,
    actor_id: str,
    owner_id: str,
) -> bool:
    """Allow a zero-time Master view without creating floor-side fiction."""

    owner_observer_only = bool(
        len(result.observers) == 1
        and result.observers[0].character_id == owner_id
        and result.observers[0].routing_role == "observe_only"
    )
    owner_facts_only = all(
        fact.audience == "only"
        and fact.visible_to == [owner_id]
        for fact in result.canonical_event.observable_facts
    )

    return bool(
        actor_id == owner_id
        and result.event_kind == "query_response"
        and result.duration_s == 0
        and owner_observer_only
        and owner_facts_only
        and not getattr(result, "state_updates", ())
        and not result.requires_responders
        and not result.required_responders
        and not result.next_output_character_ids
        and not result.perception_enrichment_character_ids
        and not result.spawn
        and not result.activate
        and not result.dormant
        and not result.cull
        and not result.commitment_open.present
        and not result.commitment_resolutions
        and not result.commitment_interrupts
        and not result.location_updates
    )


def validate_one_star_human_led_mission_result(
    checkpoint: CheckpointFile,
    *,
    actor_id: str,
    result: EventRouterOutput,
) -> None:
    """Keep every event in a Master-started beat off a human-led floor.

    The guard lasts for the whole active mission, not only while a remote player
    owns a Cat II responder slot. ``actor_id`` is the actual producer of this
    event, so a routed guide may deliver a lobby tutorial without acquiring the
    Master's broader management authority. Ordinary transaction preparation
    still owns detailed costs and operation semantics; this helper proves that
    every event in the beat remains on the disjoint lobby side. A zero-duration
    Master query is the sole exception: it may observe or depict the party but
    cannot route it or author any fictional or durable change.
    """

    if not is_one_star_checkpoint(checkpoint):
        return
    owner, account = load_one_star_account(checkpoint)
    mission = account.state.active_mission
    if (
        mission is None
        or not set(mission.party_ids) & set(checkpoint.session.character_bindings)
    ):
        raise OneStarTransactionError(
            "human-led mission guard requires an active bound mission party"
        )
    party_ids = set(mission.party_ids)
    safe_locations = _one_star_lobby_safe_locations(account.config)
    clean_actor_id = actor_id.strip()
    actor_is_owner = clean_actor_id == owner.character_id
    actor_is_guide = _one_star_human_led_lobby_guide_is_safe(
        checkpoint,
        character_id=clean_actor_id,
        account=account,
        allowed_locations=safe_locations,
    )
    actor_is_lobby_hero = _one_star_human_led_lobby_hero_is_safe(
        checkpoint,
        character_id=clean_actor_id,
        account=account,
        allowed_locations=safe_locations,
    )
    if not (actor_is_owner or actor_is_guide or actor_is_lobby_hero):
        raise OneStarTransactionError(
            "human-led mission beat produced an event from a mission-party or "
            "non-lobby actor"
        )

    if result.event_kind == "query_response":
        if _one_star_human_led_watch_query_is_safe(
            result,
            actor_id=clean_actor_id,
            owner_id=owner.character_id,
        ):
            return
        raise OneStarTransactionError(
            "human-led mission watch queries must be zero-duration, Master-only, "
            "and free of routing, lifecycle, commitment, location, and state changes"
        )

    def require_safe_hero(character_id: str) -> None:
        if not _one_star_human_led_lobby_hero_is_safe(
            checkpoint,
            character_id=character_id,
            account=account,
            allowed_locations=safe_locations,
        ):
            raise OneStarTransactionError(
                "human-led mission lobby event targets a mission-party or "
                "non-lobby Hero"
            )

    structured_character_ids = {
        *result.required_responders,
        *(observer.character_id for observer in result.observers),
        *(
            character_id
            for fact in result.canonical_event.observable_facts
            for character_id in (
                *fact.visible_to,
                *fact.visual_subject_ids,
            )
        ),
    }
    if party_ids & structured_character_ids:
        raise OneStarTransactionError(
            "human-led mission lobby event cannot observe, route, or depict "
            "mission-party Heroes"
        )
    for character_id in result.required_responders:
        require_safe_hero(character_id)
    for character_id in result.next_output_character_ids:
        if _one_star_human_led_lobby_guide_is_safe(
            checkpoint,
            character_id=character_id,
            account=account,
            allowed_locations=safe_locations,
        ):
            continue
        require_safe_hero(character_id)

    if (
        result.spawn
        or result.activate
        or result.dormant
        or result.cull
        or result.commitment_open.present
        or result.commitment_resolutions
        or result.commitment_interrupts
    ):
        raise OneStarTransactionError(
            "human-led mission lobby event cannot author generic lifecycle or "
            "commitment changes"
        )
    for location_update in result.location_updates:
        require_safe_hero(location_update.character_id)
        if (
            location_update.location_label not in safe_locations
            or location_update.location_label == mission.destination
        ):
            raise OneStarTransactionError(
                "human-led mission lobby movement must remain inside configured "
                "lobby facilities"
            )

    for update in getattr(result, "state_updates", ()):
        if update.kind == "tutorial_delivery":
            if not actor_is_guide:
                raise OneStarTransactionError(
                    "human-led mission tutorial delivery must be authored by an "
                    "active configured lobby guide"
                )
            details = _state_update_details(update)
            recipient_ids = details.get("recipient", [])
            if not recipient_ids or any(
                not _one_star_human_led_lobby_recipient_is_safe(
                    checkpoint,
                    character_id=character_id,
                    account=account,
                    allowed_locations=safe_locations,
                )
                for character_id in recipient_ids
            ):
                raise OneStarTransactionError(
                    "human-led mission tutorial recipients must be active "
                    "non-party characters in configured lobby facilities"
                )
            continue
        if not actor_is_owner:
            raise OneStarTransactionError(
                "human-led mission autonomous lobby followers cannot author "
                "account or mission state changes"
            )
        if update.kind in {"catalogue_apply", "gem_purchase"}:
            continue
        if update.kind == "summon":
            pool = account.config.summon_pools.get(update.target_id.strip())
            if not isinstance(pool, OneStarStandardSummonPool):
                raise OneStarTransactionError(
                    "human-led mission lobby turns may use only standard summon pools"
                )
            continue
        if update.kind == "equipment_move":
            destination = update.value.strip()
            if destination != "account":
                require_safe_hero(destination)
            holders = []
            for character in checkpoint.characters:
                hero = load_one_star_hero(character)
                if hero is not None and any(
                    item.item_id == update.target_id.strip()
                    for item in hero.equipment
                ):
                    holders.append(character.character_id)
            for holder_id in holders:
                require_safe_hero(holder_id)
            continue
        if update.kind == "pending_open":
            if update.value.strip() not in {"synthesis", "promotion"}:
                raise OneStarTransactionError(
                    "human-led mission lobby turns cannot open a deployment"
                )
            details = _state_update_details(update)
            affected = {
                *details.get("participant", []),
                *details.get("target_id", []),
            }
            if not affected:
                raise OneStarTransactionError(
                    "human-led mission lobby operation must identify its affected Heroes"
                )
            for character_id in affected:
                require_safe_hero(character_id)
            continue
        if update.kind in {"pending_resolve", "pending_cancel"}:
            pending = account.state.pending_operation
            if (
                pending is None
                or pending.operation_id != update.target_id.strip()
                or pending.kind not in {"synthesis", "promotion"}
            ):
                raise OneStarTransactionError(
                    "human-led mission lobby resolution must target the open "
                    "lobby operation"
                )
            for character_id in {
                *pending.participant_ids,
                *([pending.target_id] if pending.target_id else []),
            }:
                require_safe_hero(character_id)
            continue
        raise OneStarTransactionError(
            f"One-Star {update.kind} cannot commit during a human-led mission "
            "Master beat"
        )


def _one_star_mission_floor_character_ids(
    checkpoint: CheckpointFile,
    *,
    owner_character_id: str,
    party_ids: set[str],
    destination: str,
) -> set[str]:
    return {
        character.character_id
        for character in checkpoint.characters
        if (
            character.status == CharacterStatus.active
            and character.character_id != owner_character_id
            and (
                character.character_id in party_ids
                or character.location == destination
            )
        )
    }


def _active_one_star_system_observer_ids(
    checkpoint: CheckpointFile,
    state: OneStarAccountState,
) -> tuple[str, ...]:
    characters = {
        character.character_id: character
        for character in checkpoint.characters
        if character.character_id
    }
    return tuple(
        character_id
        for character_id in dict.fromkeys(
            getattr(state, "system_observer_ids", ())
        )
        if (
            character_id in characters
            and characters[character_id].status == CharacterStatus.active
        )
    )


def prepare_one_star_live_mission_observers(
    checkpoint: CheckpointFile,
    *,
    actor_id: str,
    result: EventRouterOutput,
) -> tuple[str, ...]:
    """Materialize configured live System mission observers.

    The established event observer list is the sole delivery path.  A clear
    System feed is direct perception even when its recipient remains in the
    lobby, so broad mission facts may flow through ordinary event broadcast.
    The feed grants knowledge only: before the terminal event it never grants
    response ownership, physical presence, or authority over the floor.

    The configured feed is engine authority, not a routing choice the model
    can weaken or turn into physical presence. Normalize its observation and
    presentation metadata deterministically: remote viewers receive direct
    observation, remain passive before mission end, and never enter a floor
    sprite roster merely because the router repeated their id in
    ``visual_subject_ids``. Genuine responder or perception targeting remains
    invalid below because that changes event semantics rather than metadata.

    Returns the active remote System observer ids when ``result`` is a live
    mission-floor event, otherwise an empty tuple.  Repeated calls are
    idempotent so router validation and commit-time validation share the same
    normalization.
    """

    if not is_one_star_checkpoint(checkpoint):
        return ()
    try:
        owner, account = load_one_star_account(checkpoint)
    except (OneStarTransactionError, ValidationError):
        # Schema-only router fixtures can select the ruleset without carrying
        # an account. The ordinary One-Star transaction path remains
        # responsible for rejecting a malformed real checkpoint.
        return ()
    mission = getattr(account.state, "active_mission", None)
    if mission is None:
        return ()

    party_ids = set(mission.party_ids)
    floor_character_ids = _one_star_mission_floor_character_ids(
        checkpoint,
        owner_character_id=owner.character_id,
        party_ids=party_ids,
        destination=mission.destination,
    )
    clean_actor_id = actor_id.strip()
    if clean_actor_id:
        if clean_actor_id not in floor_character_ids:
            return ()
    elif one_star_active_mission_has_bound_party_member(checkpoint):
        # Router-owned autonomous continuations exist only for NPC-only
        # missions. Human-led continuations retain their floor actor id.
        return ()

    remote_system_observer_ids = tuple(
        character_id
        for character_id in _active_one_star_system_observer_ids(
            checkpoint,
            account.state,
        )
        if (
            character_id != owner.character_id
            and character_id not in floor_character_ids
        )
    )
    if not remote_system_observer_ids:
        return ()

    observers_by_id = {
        observer.character_id: observer
        for observer in result.observers
        if observer.character_id
    }
    for character_id in remote_system_observer_ids:
        if character_id not in observers_by_id:
            observer = ObserverEntry(
                character_id=character_id,
                observation_level="d",
                routing_role="observe_only",
            )
            result.observers.append(observer)
            observers_by_id[character_id] = observer

    updates = tuple(getattr(result, "state_updates", ()))
    terminal_result = any(
        update.kind == "mission_end"
        and update.target_id.strip() == mission.mission_id
        for update in updates
    )
    terminal_guide_ids = (
        set(remote_system_observer_ids)
        & set(getattr(account.state, "guide_character_ids", ()))
        if terminal_result
        else set()
    )
    remote_ids = set(remote_system_observer_ids)
    for character_id in remote_system_observer_ids:
        observer = observers_by_id[character_id]
        if (
            terminal_result
            and observer.routing_role == "next_output"
            and character_id not in terminal_guide_ids
        ):
            raise OneStarTransactionError(
                "live One-Star System observers must remain observe_only unless "
                "selected as an eligible guide at mission end: "
                f"{character_id}"
            )
        observer.observation_level = "d"
        if not (
            observer.routing_role == "next_output"
            and character_id in terminal_guide_ids
        ):
            observer.routing_role = "observe_only"

    for fact in result.canonical_event.observable_facts:
        fact.visual_subject_ids = [
            character_id
            for character_id in fact.visual_subject_ids
            if character_id not in remote_ids
        ]

    forbidden_routed_ids = remote_ids & {
        *result.required_responders,
        *result.perception_enrichment_character_ids,
    }
    if forbidden_routed_ids:
        raise OneStarTransactionError(
            "live One-Star System observers cannot become responders or "
            "perception targets: "
            + ", ".join(sorted(forbidden_routed_ids))
        )
    remote_next_output_ids = remote_ids & set(result.next_output_character_ids)
    if remote_next_output_ids - terminal_guide_ids:
        raise OneStarTransactionError(
            "live One-Star System observers cannot receive midmission output: "
            + ", ".join(sorted(remote_next_output_ids - terminal_guide_ids))
        )
    if len(remote_next_output_ids) > 1:
        raise OneStarTransactionError(
            "mission end may hand off to at most one live System guide"
        )

    return remote_system_observer_ids


def validate_one_star_autonomous_mission_batch_result(
    checkpoint: CheckpointFile,
    *,
    actor_id: str,
    result: EventRouterOutput,
) -> None:
    """Keep a post-admission NPC mission batch on its active floor.

    The initiating owner event is validated and committed before the batch is
    admitted, so this guard applies only to subsequent agent turns and
    router-owned continuations.  The owner and configured System observers may
    keep watching without becoming fictional floor actors or account-operation
    authorities. System observers remain ``observe_only`` before mission end;
    a terminal event may hand off once to a configured guide. Adapter-authored
    terminal System notices may add other scoped recipients without widening
    the router-authored floor event.
    """

    if not is_one_star_checkpoint(checkpoint):
        return
    owner, account = load_one_star_account(checkpoint)
    mission = account.state.active_mission
    if (
        mission is None
        or set(mission.party_ids) & set(checkpoint.session.character_bindings)
    ):
        raise OneStarTransactionError(
            "autonomous mission batch requires an active NPC-only party"
        )

    party_ids = set(mission.party_ids)
    floor_character_ids = _one_star_mission_floor_character_ids(
        checkpoint,
        owner_character_id=owner.character_id,
        party_ids=party_ids,
        destination=mission.destination,
    )
    known_character_ids = {
        character.character_id
        for character in checkpoint.characters
        if character.character_id
    }
    clean_actor_id = actor_id.strip()
    if clean_actor_id and clean_actor_id not in floor_character_ids:
        raise OneStarTransactionError(
            "autonomous mission batch event actor must be router-owned or an "
            "active floor character"
        )

    system_remote_ids = set(prepare_one_star_live_mission_observers(
        checkpoint,
        actor_id=clean_actor_id,
        result=result,
    ))
    updates = tuple(getattr(result, "state_updates", ()))
    terminal_result = any(
        update.kind == "mission_end"
        and update.target_id.strip() == mission.mission_id
        for update in updates
    )
    terminal_system_guide_ids = (
        system_remote_ids & set(account.state.guide_character_ids)
        if terminal_result
        else set()
    )
    passive_remote_ids = {owner.character_id, *system_remote_ids}
    if terminal_result:
        passive_remote_ids.update(
            one_star_terminal_system_recipient_ids(checkpoint)
        )
    allowed_observer_ids = floor_character_ids | passive_remote_ids
    observer_ids = {
        observer.character_id
        for observer in result.observers
        if observer.character_id
    }
    unrelated_observers = observer_ids - allowed_observer_ids
    if unrelated_observers:
        raise OneStarTransactionError(
            "autonomous mission batch event names non-floor observers: "
            + ", ".join(sorted(unrelated_observers))
        )
    for observer in result.observers:
        if (
            observer.character_id in passive_remote_ids - floor_character_ids
            and not (
                observer.routing_role == "observe_only"
                or (
                    observer.character_id in terminal_system_guide_ids
                    and observer.routing_role == "next_output"
                )
            )
        ):
            raise OneStarTransactionError(
                "autonomous mission batch remote viewers must remain "
                "observe_only until an eligible terminal guide handoff"
            )

    always_floor_routed_ids = {
        *result.required_responders,
        *result.perception_enrichment_character_ids,
    }
    non_floor_routing = always_floor_routed_ids - floor_character_ids
    non_floor_routing.update(
        set(result.next_output_character_ids)
        - floor_character_ids
        - terminal_system_guide_ids
    )
    if non_floor_routing:
        raise OneStarTransactionError(
            "autonomous mission batch cannot route non-floor characters: "
            + ", ".join(sorted(non_floor_routing))
        )

    remote_system_result_ids = (
        passive_remote_ids
        - floor_character_ids
        - {owner.character_id}
        - system_remote_ids
    )
    for fact in result.canonical_event.observable_facts:
        if fact.audience == "only":
            unrelated_recipients = set(fact.visible_to) - allowed_observer_ids
            if unrelated_recipients:
                raise OneStarTransactionError(
                    "autonomous mission batch fact names non-floor recipients: "
                    + ", ".join(sorted(unrelated_recipients))
                )
        elif remote_system_result_ids:
            result_observers = observer_ids & remote_system_result_ids
            if result_observers:
                raise OneStarTransactionError(
                    "autonomous mission batch terminal System recipients cannot "
                    "inherit broad floor facts"
                )
        non_floor_subjects = (
            set(fact.visual_subject_ids) - floor_character_ids
        )
        if non_floor_subjects:
            labels = sorted(
                character_id
                for character_id in non_floor_subjects
                if character_id in known_character_ids
            ) or sorted(non_floor_subjects)
            raise OneStarTransactionError(
                "autonomous mission batch cannot depict non-floor characters: "
                + ", ".join(labels)
            )

    if (
        result.spawn
        or result.activate
        or result.dormant
        or result.cull
        or result.commitment_open.present
        or result.commitment_resolutions
        or result.commitment_interrupts
        or result.location_updates
    ):
        raise OneStarTransactionError(
            "autonomous mission batch cannot author generic lifecycle, "
            "commitment, or location changes"
        )

    for update in updates:
        target_id = update.target_id.strip()
        if update.kind in {"mission_update", "mission_end"}:
            if target_id != mission.mission_id:
                raise OneStarTransactionError(
                    "autonomous mission batch update targets another mission"
                )
            continue
        if update.kind == "hero_delta":
            if target_id not in party_ids:
                raise OneStarTransactionError(
                    "autonomous mission batch Hero update targets a non-party "
                    "character"
                )
            continue
        raise OneStarTransactionError(
            f"One-Star {update.kind} cannot commit during an autonomous "
            "mission batch"
        )


def one_star_should_autonomous_mission_batch_after_result(
    checkpoint: CheckpointFile,
    *,
    actor_id: str,
    result: EventRouterOutput,
    user_input: str = "",
) -> bool:
    """Start one NPC mission cascade after an owner mutation or watch."""

    if not is_one_star_checkpoint(checkpoint):
        return False
    try:
        owner, account = load_one_star_account(checkpoint)
    except (OneStarTransactionError, ValidationError):
        return False
    mission = account.state.active_mission
    if (
        mission is None
        or actor_id.strip() != owner.character_id
        or one_star_active_mission_has_bound_party_member(checkpoint)
    ):
        return False

    # An ordinary nonparty handoff keeps ownership of the current cascade.
    # Do not turn that same event into a floor batch as well.
    if set(result.next_output_character_ids) - set(mission.party_ids):
        return False
    if getattr(result, "state_updates", ()):
        return True
    if set(result.next_output_character_ids) & set(mission.party_ids):
        return True

    return _one_star_mission_watch_requested(user_input)


def one_star_standard_summon_guide_handoff_authority(
    checkpoint: CheckpointFile,
    *,
    result: EventRouterOutput,
) -> tuple[str, str] | None:
    """Return owner and guide ids for an ordinary-summon handoff.

    ``None`` means this event is outside the ordinary-summon contract.  Guide
    availability is validated before commit; returning the configured target
    rather than re-reading its post-event status lets beat pacing fail loudly
    if that accepted target somehow becomes undispatchable.  Routing,
    induction validation, and pacing therefore share one One-Star authority.
    """

    if not is_one_star_checkpoint(checkpoint):
        return None
    summon_pool_ids = [
        str(getattr(update, "target_id", "") or "")
        for update in getattr(result, "state_updates", ())
        if getattr(update, "kind", "") == "summon"
    ]
    if not summon_pool_ids:
        return None

    owner, account = load_one_star_account(checkpoint)
    if not any(
        getattr(account.config.summon_pools.get(pool_id), "usage", "")
        == "standard"
        for pool_id in summon_pool_ids
    ):
        return None
    if not account.state.guide_character_ids:
        return None
    return owner.character_id, account.state.guide_character_ids[0]


def one_star_opening_account_owner_actor_id(
    checkpoint: CheckpointFile,
    participant_ids: Iterable[str],
) -> str:
    """Return the bound owner when the selected opening uses owner authority.

    The user who presses ``/begin`` is only the interface trigger.  A mixed
    opening roster containing authored non-player slots remains an account
    acquisition and therefore uses the bound account owner as its semantic
    actor.  Bound-player-only openings keep their triggering actor.
    """

    pool_id = one_star_opening_roster_pool_id(
        checkpoint,
        participant_ids,
    )
    if not pool_id:
        return ""
    owner, account = load_one_star_account(checkpoint)
    pool = account.config.summon_pools[pool_id]
    if any(
        not isinstance(slot, OneStarOpeningRosterBoundPlayerActorSlot)
        for slot in pool.slots
    ):
        return owner.character_id
    return ""


def one_star_opening_roster_pool_id(
    checkpoint: CheckpointFile,
    participant_ids: Iterable[str],
) -> str:
    """Select the one authored opening pool for the live bound roster."""

    if not is_one_star_checkpoint(checkpoint):
        return ""
    owner, account = load_one_star_account(checkpoint)
    participants = {
        character_id.strip()
        for character_id in participant_ids
        if character_id.strip()
    }
    owner_present = owner.character_id in participants
    bound_participant_ids = participants - {owner.character_id}
    matching_pool_ids = [
        pool_id
        for pool_id, pool in account.config.summon_pools.items()
        if isinstance(pool, OneStarOpeningRosterSummonPool)
        and {
            slot.character_id
            for slot in pool.slots
            if isinstance(slot, OneStarOpeningRosterBoundPlayerActorSlot)
        }
        == bound_participant_ids
        and (
            owner_present
            == any(
                not isinstance(
                    slot,
                    OneStarOpeningRosterBoundPlayerActorSlot,
                )
                for slot in pool.slots
            )
        )
    ]
    if len(matching_pool_ids) > 1:
        raise OneStarTransactionError(
            "the live opening participants match more than one authored "
            "One-Star opening roster"
        )
    return matching_pool_ids[0] if matching_pool_ids else ""


def find_one_star_account_owner(
    characters: Iterable[CharacterRecord],
) -> CharacterRecord | None:
    """Find the unique account owner by state marker, never by story/name."""

    owners = [
        character
        for character in characters
        if isinstance(character.mechanics, dict)
        and ONE_STAR_ACCOUNT_KEY in character.mechanics
    ]
    if len(owners) > 1:
        raise OneStarTransactionError("multiple One-Star account owners exist")
    return owners[0] if owners else None


def load_one_star_account(
    checkpoint: CheckpointFile,
) -> tuple[CharacterRecord, OneStarAccountEnvelope]:
    """Return the typed account marker for an active One-Star checkpoint."""

    if not is_one_star_checkpoint(checkpoint):
        raise OneStarTransactionError("One-Star adapter is inactive for this checkpoint")
    owner = find_one_star_account_owner(checkpoint.characters)
    if owner is None:
        raise OneStarTransactionError("active One-Star checkpoint has no account owner")
    try:
        return owner, OneStarAccountEnvelope.model_validate(
            owner.mechanics[ONE_STAR_ACCOUNT_KEY]
        )
    except ValidationError as exc:
        raise OneStarTransactionError("invalid One-Star account state") from exc


def load_one_star_hero(character: CharacterRecord) -> OneStarHeroState | None:
    """Read a Hero overlay without treating a non-Hero as an error."""

    if not isinstance(character.mechanics, dict):
        return None
    raw = character.mechanics.get(ONE_STAR_HERO_KEY)
    if raw is None:
        return None
    try:
        return OneStarHeroState.model_validate(raw)
    except ValidationError as exc:
        raise OneStarTransactionError(
            f"character {character.character_id!r} has invalid One-Star Hero state"
        ) from exc


def load_one_star_combatant(
    character: CharacterRecord,
) -> OneStarCombatantState | None:
    """Read a non-Hero combat overlay without broadening Hero semantics."""

    if not isinstance(character.mechanics, dict):
        return None
    raw = character.mechanics.get(ONE_STAR_COMBATANT_KEY)
    if raw is None:
        return None
    if ONE_STAR_HERO_KEY in character.mechanics:
        raise OneStarTransactionError(
            f"character {character.character_id!r} cannot be both a One-Star "
            "Hero and a non-Hero combatant"
        )
    try:
        return OneStarCombatantState.model_validate(raw)
    except ValidationError as exc:
        raise OneStarTransactionError(
            f"character {character.character_id!r} has invalid One-Star "
            "combatant state"
        ) from exc


def one_star_birth_stars_for_ticket(
    pool: OneStarStandardSummonPool,
    ticket: int,
) -> int:
    """Map a zero-based 10,000-point ticket through configured pool weights."""

    if ticket < 0 or ticket >= ONE_STAR_GACHA_WEIGHT_TOTAL:
        raise OneStarTransactionError(
            "summon ticket falls outside the configured weight scale"
        )
    upper_bound = 0
    for birth_stars, weight in sorted(pool.star_weights.items()):
        upper_bound += weight
        if ticket < upper_bound:
            return birth_stars
    raise OneStarTransactionError(
        "summon pool weights do not cover the configured scale"
    )


def _stable_bounded_draw(
    *,
    session_id: str,
    pool_id: str,
    draw_index: int,
    stream: str,
    upper_bound: int,
) -> int:
    """Return an unbiased replay-stable integer below ``upper_bound``."""

    if upper_bound < 1:
        raise OneStarTransactionError("summon draw requires a positive bound")
    full_range = 1 << 256
    rejection_limit = full_range - (full_range % upper_bound)
    attempt = 0
    while True:
        payload = json.dumps(
            [
                "one-star-gacha-v1",
                session_id,
                pool_id,
                draw_index,
                stream,
                attempt,
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        if value < rejection_limit:
            return value % upper_bound
        attempt += 1


def _one_star_summon_draw_preview(
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
    state: OneStarAccountState,
    *,
    pool_id: str,
    count: int,
) -> tuple[OneStarSummonDraw, ...]:
    pool = config.summon_pools.get(pool_id)
    if pool is None:
        raise OneStarTransactionError(
            "summon preview references an unknown configured pool"
        )
    if pool.usage != "standard":
        raise OneStarTransactionError(
            "only standard summon pools have weighted draw previews"
        )
    if count < 1 or count > config.max_summon_batch:
        raise OneStarTransactionError(
            "summon preview count exceeds the configured batch range"
        )

    available_by_star: dict[int, list[str]] = {
        birth_stars: [] for birth_stars in pool.star_weights
    }
    for character_id in pool.eligible_existing_ids:
        character = _require_character(checkpoint, character_id)
        hero = load_one_star_hero(character)
        if hero is None:
            raise OneStarTransactionError(
                "eligible summon reserve does not carry a One-Star Hero sheet"
            )
        if hero.birth_stars not in pool.star_weights:
            raise OneStarTransactionError(
                "eligible summon reserve falls outside its configured pool weights"
            )
        if (
            character.status == CharacterStatus.dormant
            and not hero.owner_lobby_id
            and not hero.acquisition_event_id
        ):
            available_by_star[hero.birth_stars].append(character_id)
    for candidates in available_by_star.values():
        candidates.sort()

    start_index = state.summon_draw_counters.get(pool_id, 0)
    draws: list[OneStarSummonDraw] = []
    for slot_offset in range(count):
        draw_index = start_index + slot_offset
        ticket = _stable_bounded_draw(
            session_id=checkpoint.session.session_id,
            pool_id=pool_id,
            draw_index=draw_index,
            stream="birth-stars",
            upper_bound=ONE_STAR_GACHA_WEIGHT_TOTAL,
        )
        birth_stars = one_star_birth_stars_for_ticket(pool, ticket)
        candidates = available_by_star[birth_stars]
        existing_character_id = ""
        if candidates:
            candidate_index = _stable_bounded_draw(
                session_id=checkpoint.session.session_id,
                pool_id=pool_id,
                draw_index=draw_index,
                stream="existing-reserve",
                upper_bound=len(candidates),
            )
            existing_character_id = candidates.pop(candidate_index)
        elif not pool.fresh_generation_allowed:
            raise OneStarTransactionError(
                "summon pool has no eligible reserve for its next weighted result"
            )
        draws.append(
            OneStarSummonDraw(
                slot=slot_offset + 1,
                birth_stars=birth_stars,
                existing_character_id=existing_character_id,
            )
        )
    return tuple(draws)


def one_star_summon_draw_preview(
    checkpoint: CheckpointFile,
    pool_id: str,
    *,
    count: int | None = None,
) -> tuple[OneStarSummonDraw, ...]:
    """Return the exact next standard-pool slots without consuming them."""

    _owner, account = load_one_star_account(checkpoint)
    return _one_star_summon_draw_preview(
        checkpoint,
        account.config,
        account.state,
        pool_id=pool_id,
        count=account.config.max_summon_batch if count is None else count,
    )


def _opening_existing_hero(
    checkpoint: CheckpointFile,
    character_id: str,
    *,
    exclude_player_authored: bool,
) -> tuple[CharacterRecord, OneStarHeroState]:
    """Return one acquisition-eligible authored opening participant."""

    character = _require_character(checkpoint, character_id)
    hero = load_one_star_hero(character)
    if hero is None:
        raise OneStarTransactionError(
            f"opening character {character_id!r} has no One-Star Hero sheet"
        )
    if character.status != CharacterStatus.dormant:
        raise OneStarTransactionError(
            f"opening character {character_id!r} is not dormant"
        )
    if hero.owner_lobby_id or hero.acquisition_event_id:
        raise OneStarTransactionError(
            f"opening character {character_id!r} is already owned or acquired"
        )
    if hero.terminal_event_id:
        raise OneStarTransactionError(
            f"opening character {character_id!r} is terminal"
        )
    if exclude_player_authored and is_player_authored_slot(character):
        raise OneStarTransactionError(
            f"opening roster character {character_id!r} is player-authored"
        )
    return character, hero


def _one_star_opening_roster_preview(
    checkpoint: CheckpointFile,
    *,
    pool_id: str,
    pool: OneStarOpeningRosterSummonPool,
) -> tuple[OneStarSummonDraw, ...]:
    exact_ids = {
        slot.character_id
        for slot in pool.slots
        if isinstance(
            slot,
            (
                OneStarOpeningRosterFixedSlot,
                OneStarOpeningRosterBoundPlayerActorSlot,
            ),
        )
    }
    exact_heroes: dict[str, OneStarHeroState] = {}
    for character_id in sorted(exact_ids):
        slot = next(
            candidate
            for candidate in pool.slots
            if getattr(candidate, "character_id", "") == character_id
        )
        is_bound_actor = isinstance(
            slot,
            OneStarOpeningRosterBoundPlayerActorSlot,
        )
        _character, hero = _opening_existing_hero(
            checkpoint,
            character_id,
            exclude_player_authored=not is_bound_actor,
        )
        if is_bound_actor:
            if not is_player_authored_slot(_character):
                raise OneStarTransactionError(
                    f"bound opening actor {character_id!r} is not a "
                    "player-authored slot"
                )
            if character_id not in checkpoint.session.character_bindings:
                raise OneStarTransactionError(
                    f"bound opening actor {character_id!r} has no live player binding"
                )
        exact_heroes[character_id] = hero

    chosen_ids: set[str] = set()
    draws: list[OneStarSummonDraw] = []
    for slot_index, slot in enumerate(pool.slots):
        if isinstance(
            slot,
            (
                OneStarOpeningRosterFixedSlot,
                OneStarOpeningRosterBoundPlayerActorSlot,
            ),
        ):
            character_id = slot.character_id
            hero = exact_heroes[character_id]
        else:
            candidates: list[tuple[str, OneStarHeroState]] = []
            for character in checkpoint.characters:
                if character.character_id in exact_ids | chosen_ids:
                    continue
                hero = load_one_star_hero(character)
                if hero is None or hero.birth_stars != slot.birth_stars:
                    continue
                if (
                    character.status != CharacterStatus.dormant
                    or hero.owner_lobby_id
                    or hero.acquisition_event_id
                    or hero.terminal_event_id
                    or is_player_authored_slot(character)
                ):
                    continue
                candidates.append((character.character_id, hero))
            candidates.sort(key=lambda item: item[0])
            if not candidates:
                raise OneStarTransactionError(
                    "opening roster has no eligible dormant unowned "
                    f"birth-{slot.birth_stars} Hero for slot {slot_index + 1}"
                )
            candidate_index = _stable_bounded_draw(
                session_id=checkpoint.session.session_id,
                pool_id=pool_id,
                draw_index=slot_index,
                stream=f"opening-roster-birth-{slot.birth_stars}",
                upper_bound=len(candidates),
            )
            character_id, hero = candidates[candidate_index]
        if character_id in chosen_ids:
            raise OneStarTransactionError(
                f"opening roster selects character {character_id!r} more than once"
            )
        chosen_ids.add(character_id)
        draws.append(OneStarSummonDraw(
            slot=slot_index + 1,
            birth_stars=hero.birth_stars,
            existing_character_id=character_id,
        ))
    return tuple(draws)


def one_star_opening_roster_preview(
    checkpoint: CheckpointFile,
    pool_id: str,
) -> tuple[OneStarSummonDraw, ...]:
    """Resolve a free authored opening roster without mutating the checkpoint."""

    _owner, account = load_one_star_account(checkpoint)
    pool = account.config.summon_pools.get(pool_id)
    if not isinstance(pool, OneStarOpeningRosterSummonPool):
        raise OneStarTransactionError(
            "opening roster preview requires an opening-roster summon pool"
        )
    return _one_star_opening_roster_preview(
        checkpoint,
        pool_id=pool_id,
        pool=pool,
    )


def _one_star_summon_result_preview(
    checkpoint: CheckpointFile,
    account: OneStarAccountEnvelope,
    *,
    pool_id: str,
    count: int,
) -> tuple[OneStarSummonDraw, ...]:
    pool = account.config.summon_pools.get(pool_id)
    if pool is None:
        raise OneStarTransactionError(
            "summon state update references an unknown configured pool"
        )
    if isinstance(pool, OneStarStandardSummonPool):
        return _one_star_summon_draw_preview(
            checkpoint,
            account.config,
            account.state,
            pool_id=pool_id,
            count=count,
        )
    if count != len(pool.slots):
        raise OneStarTransactionError(
            "opening-roster summon count must exactly match its configured slots"
        )
    return _one_star_opening_roster_preview(
        checkpoint,
        pool_id=pool_id,
        pool=pool,
    )


def _state_update_details(update: OneStarStateUpdate) -> dict[str, list[str]]:
    details: dict[str, list[str]] = {}
    for raw_entry in update.details:
        key, separator, value = raw_entry.partition("=")
        key = key.strip()
        if not separator or not key:
            raise OneStarTransactionError(
                "One-Star state-update details must use non-empty key=value entries"
            )
        details.setdefault(key, []).append(value.strip())
    return details


def _validate_state_update_detail_keys(
    update: OneStarStateUpdate,
    details: Mapping[str, list[str]],
) -> None:
    """Reject misspelled compact fields instead of silently dropping them."""

    exact_by_kind: dict[str, frozenset[str]] = {
        "catalogue_apply": frozenset(),
        "summon": frozenset(),
        "inventory_delta": frozenset(),
        "gem_purchase": frozenset(),
        "hero_delta": frozenset({
            "hp_current",
            "equipment_remove",
            "skill_remove",
            "condition",
            "persistent_injury",
            "terminal_action",
            "death_cause",
        }),
        "mission_start": frozenset({
            "pending_operation_id",
            "party",
            "destination",
            "completion",
            "failure",
            "duration_s",
        }),
        "mission_update": frozenset(),
        "mission_end": frozenset({
            "return_destination",
            "escape_authority_id",
        }),
        "pending_open": frozenset({
            "participant",
            "target_id",
            "destination",
        }),
        "pending_resolve": frozenset(),
        "equipment_move": frozenset(),
        "pending_cancel": frozenset(),
        "tutorial_delivery": frozenset({"recipient"}),
    }
    prefix_by_kind: dict[str, tuple[str, ...]] = {
        "hero_delta": (
            "durability.",
            "skill_rank.",
            "equipment_add.",
            "skill_add.",
        ),
        "mission_start": ("counter.",),
        "mission_update": ("counter.",),
    }
    exact = exact_by_kind[update.kind]
    prefixes = prefix_by_kind.get(update.kind, ())
    unknown = sorted(
        key
        for key in details
        if key not in exact and not any(key.startswith(prefix) for prefix in prefixes)
    )
    if unknown:
        raise OneStarTransactionError(
            f"One-Star {update.kind} state update has unsupported details: "
            + ", ".join(unknown)
        )

    if update.kind == "hero_delta":
        equipment_fields = {
            "name",
            "slot",
            "quantity",
            "durability_current",
            "durability_max",
            "tag",
            "visible",
        }
        skill_fields = {"name", "rank", "capability", "tag", "visible"}
        for key in details:
            for prefix, fields in (
                ("equipment_add.", equipment_fields),
                ("skill_add.", skill_fields),
            ):
                if not key.startswith(prefix):
                    continue
                remainder = key.removeprefix(prefix)
                identity, separator, field_name = remainder.rpartition(".")
                if not separator or not identity or field_name not in fields:
                    raise OneStarTransactionError(
                        f"One-Star hero_delta state update has unsupported detail {key!r}"
                    )
                break
            if key.startswith(("durability.", "skill_rank.")):
                if not key.partition(".")[2]:
                    raise OneStarTransactionError(
                        f"One-Star hero_delta state update has empty detail id {key!r}"
                    )

    for key in details:
        if key.startswith("counter.") and not key.removeprefix("counter."):
            raise OneStarTransactionError(
                f"One-Star {update.kind} state update has empty detail id {key!r}"
            )


def _validate_state_update_scalar_shape(update: OneStarStateUpdate) -> None:
    empty_value_kinds = {
        "hero_delta",
        "mission_update",
        "pending_resolve",
        "pending_cancel",
        "tutorial_delivery",
    }
    if update.kind in empty_value_kinds and update.value:
        raise OneStarTransactionError(
            f"One-Star {update.kind} state update does not use value"
        )


def _single_detail(
    details: Mapping[str, list[str]],
    key: str,
    *,
    default: str | None = None,
) -> str:
    values = details.get(key, [])
    if not values:
        if default is not None:
            return default
        raise OneStarTransactionError(
            f"One-Star state update requires detail {key!r}"
        )
    if len(values) != 1:
        raise OneStarTransactionError(
            f"One-Star state update detail {key!r} must appear exactly once"
        )
    return values[0]


def _integer_update_value(value: str, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise OneStarTransactionError(
            f"One-Star state update {label} must be an integer"
        ) from exc


def _boolean_update_value(value: str, *, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise OneStarTransactionError(
        f"One-Star state update {label} must be true or false"
    )


def _counter_from_detail(key: str, value: str) -> OneStarMissionCounter:
    current_text, separator, target_text = value.partition("/")
    if not separator:
        raise OneStarTransactionError(
            "One-Star mission counter updates must use current/target"
        )
    return OneStarMissionCounter(
        counter_id=key.removeprefix("counter."),
        current=_integer_update_value(current_text, label="counter current"),
        target=_integer_update_value(target_text, label="counter target"),
    )


def _fresh_summon_character_id(
    *,
    lobby_id: str,
    pool_id: str,
    draw_index: int,
) -> str:
    safe_pool = "".join(
        character if character.isalnum() else "_" for character in pool_id.lower()
    ).strip("_") or "summon"
    return f"{lobby_id}_{safe_pool}_{draw_index + 1:04d}"


def one_star_state_updates_to_transaction(
    checkpoint: CheckpointFile,
    state_updates: Iterable[OneStarStateUpdate],
    *,
    canonical_at_s: int,
) -> OneStarTransaction:
    """Translate the router's one compact update list into private typed work.

    The durable operation models remain an adapter implementation detail.  In
    particular, configured gacha results and canonical timestamps are filled
    here rather than repeated by the model.
    """

    updates = list(state_updates)
    if sum(update.kind == "summon" for update in updates) > 1:
        raise OneStarTransactionError(
            "a compact One-Star update list may contain only one summon"
        )
    account: OneStarAccountEnvelope | None = None
    operations: list[OneStarOperation] = []
    for update in updates:
        details = _state_update_details(update)
        _validate_state_update_detail_keys(update, details)
        _validate_state_update_scalar_shape(update)
        kind = update.kind
        target_id = update.target_id.strip()

        if kind == "catalogue_apply":
            operations.append(OneStarCatalogueApplyOperation(
                operation=kind,
                catalogue_id=target_id,
                quantity=_integer_update_value(update.value, label="quantity"),
            ))
            continue

        if kind == "summon":
            if account is None:
                _owner, account = load_one_star_account(checkpoint)
            count = _integer_update_value(update.value, label="summon count")
            pool = account.config.summon_pools.get(target_id)
            if pool is None:
                raise OneStarTransactionError(
                    "summon state update references an unknown configured pool"
                )
            if details:
                raise OneStarTransactionError(
                    "summon state updates must not name adapter-owned results"
                )
            draws = _one_star_summon_result_preview(
                checkpoint,
                account,
                pool_id=target_id,
                count=count,
            )
            start_index = account.state.summon_draw_counters.get(target_id, 0)
            hero_ids = [
                draw.existing_character_id
                or _fresh_summon_character_id(
                    lobby_id=account.config.lobby_id,
                    pool_id=target_id,
                    draw_index=start_index + offset,
                )
                for offset, draw in enumerate(draws)
            ]
            birth_stars = [draw.birth_stars for draw in draws]
            operations.append(OneStarSummonOperation(
                operation=kind,
                pool_id=target_id,
                hero_ids=hero_ids,
                birth_stars=birth_stars,
            ))
            continue

        if kind == "inventory_delta":
            operations.append(OneStarInventoryDeltaOperation(
                operation=kind,
                item_id=target_id,
                quantity_delta=_integer_update_value(
                    update.value,
                    label="inventory delta",
                ),
            ))
            continue

        if kind == "gem_purchase":
            if target_id != "gems":
                raise OneStarTransactionError(
                    "Gem-purchase state updates must target the Gems balance"
                )
            operations.append(OneStarGemPurchaseOperation(
                operation=kind,
                gem_quantity=_integer_update_value(
                    update.value,
                    label="Gem-purchase quantity",
                ),
            ))
            continue

        if kind == "equipment_move":
            operations.append(OneStarEquipmentMoveOperation(
                operation=kind,
                item_id=target_id,
                destination=update.value,
            ))
            continue

        if kind == "hero_delta":
            equipment_add: list[OneStarEquipmentEntry] = []
            equipment_ids = {
                key.split(".", 2)[1]
                for key in details
                if key.startswith("equipment_add.") and key.count(".") >= 2
            }
            for item_id in sorted(equipment_ids):
                prefix = f"equipment_add.{item_id}."
                equipment_add.append(OneStarEquipmentEntry(
                    item_id=item_id,
                    name=_single_detail(details, prefix + "name"),
                    slot=_single_detail(details, prefix + "slot"),
                    quantity=_integer_update_value(
                        _single_detail(details, prefix + "quantity"),
                        label=prefix + "quantity",
                    ),
                    durability_current=_integer_update_value(
                        _single_detail(details, prefix + "durability_current"),
                        label=prefix + "durability_current",
                    ),
                    durability_max=_integer_update_value(
                        _single_detail(details, prefix + "durability_max"),
                        label=prefix + "durability_max",
                    ),
                    tags=details.get(prefix + "tag", []),
                    visible=_boolean_update_value(
                        _single_detail(details, prefix + "visible"),
                        label=prefix + "visible",
                    ),
                ))
            skills_add: list[OneStarSkillEntry] = []
            skill_ids = {
                key.split(".", 2)[1]
                for key in details
                if key.startswith("skill_add.") and key.count(".") >= 2
            }
            for skill_id in sorted(skill_ids):
                prefix = f"skill_add.{skill_id}."
                skills_add.append(OneStarSkillEntry(
                    skill_id=skill_id,
                    name=_single_detail(details, prefix + "name"),
                    rank=_integer_update_value(
                        _single_detail(details, prefix + "rank"),
                        label=prefix + "rank",
                    ),
                    capability=_single_detail(
                        details,
                        prefix + "capability",
                        default="",
                    ),
                    tags=details.get(prefix + "tag", []),
                    visible=_boolean_update_value(
                        _single_detail(details, prefix + "visible"),
                        label=prefix + "visible",
                    ),
                ))
            operations.append(OneStarHeroDeltaOperation(
                operation=kind,
                hero_id=target_id,
                hp_current=(
                    _integer_update_value(
                        _single_detail(details, "hp_current"),
                        label="hp_current",
                    )
                    if "hp_current" in details else None
                ),
                equipment_add=equipment_add,
                equipment_remove_ids=details.get("equipment_remove", []),
                skills_add=skills_add,
                skills_remove_ids=details.get("skill_remove", []),
                equipment_durability=[
                    OneStarDurabilityUpdate(
                        item_id=key.removeprefix("durability."),
                        durability_current=_integer_update_value(
                            _single_detail(details, key),
                            label=key,
                        ),
                    )
                    for key, values in details.items()
                    if key.startswith("durability.")
                ],
                skill_rank_updates=[
                    OneStarSkillRankUpdate(
                        skill_id=key.removeprefix("skill_rank."),
                        rank=_integer_update_value(
                            _single_detail(details, key),
                            label=key,
                        ),
                    )
                    for key, values in details.items()
                    if key.startswith("skill_rank.")
                ],
                conditions=(
                    [value for value in details["condition"] if value]
                    if "condition" in details else None
                ),
                persistent_injuries=(
                    [value for value in details["persistent_injury"] if value]
                    if "persistent_injury" in details else None
                ),
                terminal_action=_single_detail(
                    details,
                    "terminal_action",
                    default="none",
                ),
                death_cause=_single_detail(details, "death_cause", default=""),
            ))
            continue

        if kind == "mission_start":
            counters = [
                _counter_from_detail(key, _single_detail(details, key))
                for key, values in details.items()
                if key.startswith("counter.")
            ]
            operations.append(OneStarMissionStartOperation(
                operation=kind,
                pending_operation_id=_single_detail(
                    details,
                    "pending_operation_id",
                    default="",
                ),
                mission={
                    "mission_id": target_id,
                    "floor": _integer_update_value(update.value, label="mission floor"),
                    "party_ids": details.get("party", []),
                    "destination": _single_detail(details, "destination"),
                    "completion_declaration": _single_detail(details, "completion"),
                    "failure_declaration": _single_detail(details, "failure"),
                    "counters": counters,
                    "started_at_s": canonical_at_s,
                    "deadline_at_s": (
                        canonical_at_s
                        + _integer_update_value(
                            _single_detail(details, "duration_s"),
                            label="mission duration",
                        )
                        if "duration_s" in details
                        else 0
                    ),
                },
            ))
            continue

        if kind == "mission_update":
            operations.append(OneStarMissionUpdateOperation(
                operation=kind,
                mission_id=target_id,
                counters=[
                    _counter_from_detail(key, _single_detail(details, key))
                    for key, values in details.items()
                    if key.startswith("counter.")
                ],
            ))
            continue

        if kind == "mission_end":
            operations.append(OneStarMissionEndOperation(
                operation=kind,
                mission_id=target_id,
                outcome=update.value,
                return_destination=_single_detail(
                    details,
                    "return_destination",
                    default="",
                ),
                escape_authority_id=_single_detail(
                    details,
                    "escape_authority_id",
                    default="",
                ),
            ))
            continue

        if kind == "pending_open":
            operations.append(OneStarPendingOpenOperation(
                operation=kind,
                pending=OneStarPendingOperationSelection(
                    operation_id=target_id,
                    kind=update.value,
                    participant_ids=details.get("participant", []),
                    target_id=_single_detail(details, "target_id", default=""),
                    destination=_single_detail(details, "destination", default=""),
                    opened_at_s=canonical_at_s,
                ),
            ))
            continue

        if kind == "pending_resolve":
            operations.append(OneStarPendingResolveOperation(
                operation=kind,
                operation_id=target_id,
            ))
            continue

        if kind == "pending_cancel":
            operations.append(OneStarPendingCancelOperation(
                operation=kind,
                operation_id=target_id,
            ))
            continue

        if kind == "tutorial_delivery":
            operations.append(OneStarTutorialDeliveryOperation(
                operation=kind,
                tutorial_key=target_id,
                delivered_to_ids=details.get("recipient", []),
            ))
            continue

        raise OneStarTransactionError(
            f"unsupported One-Star state update kind {kind!r}"
        )

    return OneStarTransaction(
        present=bool(operations),
        operations=operations,
    )


def one_star_summon_lifecycle(
    checkpoint: CheckpointFile,
    state_updates: Iterable[OneStarStateUpdate],
) -> tuple[tuple[SpawnRequest, ...], tuple[WakeSignal, ...]]:
    """Materialize every configured summon without router-authored identities."""

    all_updates = list(state_updates)
    updates = [update for update in all_updates if update.kind == "summon"]
    if not updates:
        return (), ()
    if len(updates) > 1:
        raise OneStarTransactionError(
            "a compact One-Star update list may contain only one summon"
        )
    _owner, account = load_one_star_account(checkpoint)
    spawn_requests: list[SpawnRequest] = []
    wake_signals: list[WakeSignal] = []
    for update in updates:
        pool_id = update.target_id.strip()
        pool = account.config.summon_pools.get(pool_id)
        if pool is None:
            raise OneStarTransactionError(
                "summon lifecycle references an unknown configured pool"
            )
        if update.details:
            raise OneStarTransactionError(
                "summon state updates must not name adapter-owned results"
            )
        count = _integer_update_value(update.value, label="summon count")
        draws = _one_star_summon_result_preview(
            checkpoint,
            account,
            pool_id=pool_id,
            count=count,
        )
        arrival_location = account.config.lobby_location_label
        if isinstance(pool, OneStarOpeningRosterSummonPool):
            transaction = one_star_state_updates_to_transaction(
                checkpoint,
                all_updates,
                canonical_at_s=checkpoint.session.leading_at_s,
            )
            if (
                len(transaction.operations) != 2
                or not isinstance(
                    transaction.operations[0],
                    OneStarSummonOperation,
                )
                or not isinstance(
                    transaction.operations[1],
                    OneStarMissionStartOperation,
                )
            ):
                raise OneStarTransactionError(
                    "an opening-roster summon must be followed by exactly one "
                    "direct mission start in the same event"
                )
            mission_start = transaction.operations[1]
            expected_ids = [draw.existing_character_id for draw in draws]
            if (
                mission_start.pending_operation_id
                or mission_start.mission.party_ids != expected_ids
            ):
                raise OneStarTransactionError(
                    "a direct opening mission must omit pending_operation_id "
                    "and preserve the opening-roster order as its exact party"
                )
            arrival_location = mission_start.mission.destination
        start_index = account.state.summon_draw_counters.get(pool_id, 0)
        for offset, draw in enumerate(draws):
            if draw.existing_character_id:
                wake_signals.append(WakeSignal(
                    character_id=draw.existing_character_id,
                    location_label=arrival_location,
                ))
                continue
            if not isinstance(pool, OneStarStandardSummonPool):
                raise OneStarTransactionError(
                    "authored opening summons cannot create fresh identities"
                )
            character_id = _fresh_summon_character_id(
                lobby_id=account.config.lobby_id,
                pool_id=pool_id,
                draw_index=start_index + offset,
            )
            spawn_requests.append(SpawnRequest.model_validate({
                "character_id": character_id,
                "seed": {
                    "role": "newly summoned Hero",
                    "reason": f"configured {pool_id} summon result",
                    "location": account.config.lobby_location_label,
                    "objectives": [],
                    "knowledge_tier": draw.birth_stars,
                },
            }))
    return tuple(spawn_requests), tuple(wake_signals)


def _require_character(
    checkpoint: CheckpointFile, character_id: str,
) -> CharacterRecord:
    character = next(
        (item for item in checkpoint.characters if item.character_id == character_id),
        None,
    )
    if character is None:
        raise OneStarTransactionError(f"unknown character id {character_id!r}")
    return character


def _require_active_hero(checkpoint: CheckpointFile, character_id: str) -> tuple[CharacterRecord, OneStarHeroState]:
    character = _require_character(checkpoint, character_id)
    if character.status != CharacterStatus.active:
        raise OneStarTransactionError(f"Hero {character_id!r} is not active")
    hero = load_one_star_hero(character)
    if hero is None:
        raise OneStarTransactionError(f"character {character_id!r} is not a One-Star Hero")
    return character, hero


def _require_local_active_hero(
    checkpoint: CheckpointFile, character_id: str, config: OneStarRulesConfig,
) -> tuple[CharacterRecord, OneStarHeroState]:
    character, hero = _require_active_hero(checkpoint, character_id)
    if hero.owner_lobby_id != config.lobby_id:
        raise OneStarTransactionError("transaction cannot affect a Hero owned by another lobby")
    return character, hero


def _store_hero(character: CharacterRecord, hero: OneStarHeroState) -> None:
    character.mechanics = dict(character.mechanics)
    character.mechanics[ONE_STAR_HERO_KEY] = hero.model_dump(mode="json")


def _store_account(owner: CharacterRecord, account: OneStarAccountEnvelope) -> None:
    try:
        validated = OneStarAccountEnvelope.model_validate(
            account.model_dump(mode="python")
        )
    except ValidationError as exc:
        raise OneStarTransactionError(
            "One-Star transaction produced an invalid account state"
        ) from exc
    owner.mechanics = dict(owner.mechanics)
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = validated.model_dump(mode="json")


def one_star_synthesis_source_experience(
    source: OneStarHeroState,
    config: OneStarRulesConfig,
) -> int:
    """Return one selected source's exact, replay-stable synthesis XP offer."""

    potential = scaled_by_grade(
        config.progression.synthesis_source_base_xp,
        source.birth_stars,
        config,
    )
    return source.experience_points // 2 + potential


def _synthesis_sources_and_target(
    pending: OneStarPendingOperationSelection | OneStarPendingOperation,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
) -> tuple[
    CharacterRecord,
    OneStarHeroState,
    list[tuple[CharacterRecord, OneStarHeroState]],
]:
    target_character, target = _require_local_active_hero(
        checkpoint,
        pending.target_id,
        config,
    )
    sources: list[tuple[CharacterRecord, OneStarHeroState]] = []
    for source_id in pending.participant_ids:
        if source_id == pending.target_id:
            raise OneStarTransactionError("synthesis target cannot be a source")
        sources.append(_require_local_active_hero(checkpoint, source_id, config))
    return target_character, target, sources


def _synthesis_input_state_fingerprint(
    pending: OneStarPendingOperationSelection | OneStarPendingOperation,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
) -> str:
    """Fingerprint only inputs that can change the promised synthesis result.

    HP, conditions, and injuries may legitimately change while selected Heroes
    respond and physically reach the chamber. They do not alter the previewed
    XP, returned gear, or skill-transfer pool, so they must not stale an
    otherwise valid operation.
    """

    _target_character, target, sources = _synthesis_sources_and_target(
        pending,
        checkpoint,
        config,
    )
    payload = [
        "ayoa.one_star.synthesis.input-state.v1",
        {
            "role": "target",
            "character_id": pending.target_id,
            "birth_stars": target.birth_stars,
            "current_stars": target.current_stars,
            "level": target.level,
            "experience_points": target.experience_points,
            "hp_max": target.hp_max,
            "stats": target.stats,
            "progression_seed": target.progression_seed,
            "strong_stat_id": target.strong_stat_id,
            "weak_stat_id": target.weak_stat_id,
            "potential_grade": target.potential_grade,
            "skills": [skill.model_dump(mode="json") for skill in target.skills],
        },
        *[
            {
                "role": "source",
                "character_id": source_character.character_id,
                "birth_stars": source.birth_stars,
                "experience_points": source.experience_points,
                "skills": [
                    skill.model_dump(mode="json") for skill in source.skills
                ],
            }
            for source_character, source in sources
        ],
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _synthesis_preview(
    pending: OneStarPendingOperationSelection | OneStarPendingOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
) -> OneStarSynthesisPreview:
    _target_character, target, sources = _synthesis_sources_and_target(
        pending,
        checkpoint,
        config,
    )
    offered_xp = sum(
        one_star_synthesis_source_experience(source, config)
        for _character, source in sources
    )
    try:
        xp_preview = preview_experience(
            hero=target,
            experience_delta=offered_xp,
            config=config,
        )
    except ValueError as exc:
        raise OneStarTransactionError("synthesis target has invalid progression state") from exc
    if xp_preview.applied_xp < 1:
        raise OneStarTransactionError(
            "synthesis target cannot accept any offered experience"
        )
    returned_equipment = [
        item.model_copy(deep=True)
        for _character, source in sources
        for item in source.equipment
    ]
    known_item_ids = {item.item_id for item in state.stored_equipment}
    for item in returned_equipment:
        if item.item_id in known_item_ids:
            raise OneStarTransactionError(
                "synthesis source equipment conflicts with durable storage"
            )
        known_item_ids.add(item.item_id)
    return OneStarSynthesisPreview(
        offered_xp=xp_preview.offered_xp,
        applied_xp=xp_preview.applied_xp,
        wasted_xp=xp_preview.wasted_xp,
        returned_equipment=returned_equipment,
        skill_transfer_chance_basis_points=(
            config.progression.synthesis_skill_chance_basis_points
        ),
        input_state_fingerprint=_synthesis_input_state_fingerprint(
            pending,
            checkpoint,
            config,
        ),
    )


def _system_recipients(*character_ids: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(character_id for character_id in character_ids if character_id))


def _system_window_recipients(
    checkpoint: CheckpointFile,
    state: OneStarAccountState,
    config: OneStarRulesConfig,
    owner_character_id: str,
    eligible_hero_ids: Iterable[str] = (),
    *,
    include_management_observers: bool = True,
) -> tuple[str, ...]:
    characters = {character.character_id: character for character in checkpoint.characters}
    eligible_hero_id_set = set(eligible_hero_ids)
    recipients = [owner_character_id]
    if include_management_observers:
        recipients.extend(state.guide_character_ids)
        recipients.extend(state.system_observer_ids)
    research_visible = bool(
        config.hero_system_visibility_research_key
        and state.research_levels.get(
            config.hero_system_visibility_research_key,
            0,
        )
    )
    for character in checkpoint.characters:
        hero = load_one_star_hero(character)
        if (
            character.character_id in eligible_hero_id_set
            and character.status == CharacterStatus.active
            and hero is not None
            and hero.owner_lobby_id == config.lobby_id
            and (hero.innate_system_sight or research_visible)
        ):
            recipients.append(character.character_id)
    return tuple(
        character_id
        for character_id in _system_recipients(*recipients)
        if (
            character_id == owner_character_id
            or (
                character_id in characters
                and characters[character_id].status == CharacterStatus.active
            )
        )
    )


def one_star_terminal_system_recipient_ids(
    checkpoint: CheckpointFile,
) -> tuple[str, ...]:
    """Return active POVs entitled to terminal mission System facts."""

    if not is_one_star_checkpoint(checkpoint):
        return ()
    owner, account = load_one_star_account(checkpoint)
    local_active_hero_ids = tuple(
        character.character_id
        for character in checkpoint.characters
        if (
            character.status == CharacterStatus.active
            and (hero := load_one_star_hero(character)) is not None
            and hero.owner_lobby_id == account.config.lobby_id
        )
    )
    ordinary_recipients = _system_window_recipients(
        checkpoint,
        account.state,
        account.config,
        owner.character_id,
        local_active_hero_ids,
        include_management_observers=False,
    )
    return _system_recipients(
        owner.character_id,
        *_active_one_star_system_observer_ids(checkpoint, account.state),
        *(
            character_id
            for character_id in ordinary_recipients
            if character_id != owner.character_id
        ),
    )


def _format_basis_points(basis_points: int) -> str:
    whole, fractional = divmod(basis_points, 100)
    if not fractional:
        return f"{whole}%"
    return f"{whole}.{fractional:02d}".rstrip("0") + "%"


def _progression_consequence(
    *,
    character: CharacterRecord,
    checkpoint: CheckpointFile,
    state: OneStarAccountState,
    config: OneStarRulesConfig,
    owner_character_id: str,
    report: object,
    label: str,
    include_management_observers: bool = True,
) -> tuple[OneStarSystemConsequence, ...]:
    applied_xp = int(getattr(report, "applied_xp", 0))
    levels_gained = int(getattr(report, "levels_gained", 0))
    stat_gains = dict(getattr(report, "stat_gains", {}))
    hp_max_gain = int(getattr(report, "hp_max_gain", 0))
    details: list[str] = []
    if applied_xp:
        details.append(f"{applied_xp} XP applied")
    if levels_gained:
        details.append(
            f"{levels_gained} level{'s' if levels_gained != 1 else ''} gained"
        )
    if stat_gains:
        details.append(
            "stats "
            + ", ".join(
                f"{stat_id} +{amount}"
                for stat_id, amount in sorted(stat_gains.items())
            )
        )
    if hp_max_gain:
        details.append(f"max HP +{hp_max_gain}")
    if not details:
        return ()
    recipients = _system_window_recipients(
        checkpoint,
        state,
        config,
        owner_character_id,
        (character.character_id,),
        include_management_observers=include_management_observers,
    )
    consequences = [OneStarSystemConsequence(
        text=f"System: {character.name} {label}: " + "; ".join(details) + ".",
        recipient_character_ids=recipients,
    )]
    if character.character_id not in recipients:
        consequences.append(OneStarSystemConsequence(
            text=(
                f"A tangible surge of hard-won strength settles through "
                f"{character.name}."
            ),
            recipient_character_ids=(character.character_id,),
        ))
    return tuple(consequences)


def _synthesis_preview_consequence(
    *,
    pending: OneStarPendingOperation,
    checkpoint: CheckpointFile,
    state: OneStarAccountState,
    config: OneStarRulesConfig,
    owner_character_id: str,
) -> OneStarSystemConsequence:
    preview = pending.synthesis_preview
    if preview is None:  # Defensive: durable schema requires it for synthesis.
        raise OneStarTransactionError("synthesis pending operation has no preview")
    target = _require_character(checkpoint, pending.target_id)
    source_names = [
        _require_character(checkpoint, source_id).name
        for source_id in pending.participant_ids
    ]
    returned = (
        ", ".join(
            f"{item.name} [{item.item_id}]"
            for item in preview.returned_equipment
        )
        or "none"
    )
    return OneStarSystemConsequence(
        text=(
            f"System: synthesis preview for {target.name}: "
            f"{preview.offered_xp} XP offered; {preview.applied_xp} XP applies; "
            f"{preview.wasted_xp} XP wasted at the current cap; return {returned}; "
            f"each selected source ({', '.join(source_names)}) has an independent "
            f"{_format_basis_points(preview.skill_transfer_chance_basis_points)} "
            "skill-transfer chance."
        ),
        recipient_character_ids=_system_recipients(
            *_system_window_recipients(
                checkpoint,
                state,
                config,
                owner_character_id,
                (pending.target_id, *pending.participant_ids),
            ),
        ),
    )


def _add_resources(resources: OneStarCost, amount: OneStarCost) -> None:
    resources.gold += amount.gold
    resources.gems += amount.gems
    resources.building_resources += amount.building_resources
    for material_id, quantity in amount.materials.items():
        resources.materials[material_id] = resources.materials.get(material_id, 0) + quantity


def _spend_resources(resources: OneStarCost, cost: OneStarCost) -> None:
    insufficient = (
        resources.gold < cost.gold
        or resources.gems < cost.gems
        or resources.building_resources < cost.building_resources
        or any(resources.materials.get(key, 0) < quantity for key, quantity in cost.materials.items())
    )
    if insufficient:
        raise OneStarTransactionError("insufficient configured One-Star resources")
    resources.gold -= cost.gold
    resources.gems -= cost.gems
    resources.building_resources -= cost.building_resources
    for material_id, quantity in cost.materials.items():
        resources.materials[material_id] -= quantity


def _multiply_cost(cost: OneStarCost, quantity: int) -> OneStarCost:
    return OneStarCost(
        gold=cost.gold * quantity,
        gems=cost.gems * quantity,
        building_resources=cost.building_resources * quantity,
        materials={key: value * quantity for key, value in cost.materials.items()},
    )


def effective_one_star_stamina(
    state: OneStarAccountState,
    config: OneStarRulesConfig,
    now_s: int,
) -> tuple[int, int]:
    """Project stamina and its anchor at canonical ``now_s`` without mutation."""

    if now_s < state.stamina_recovery_anchor_s:
        raise OneStarTransactionError("canonical time cannot precede stamina recovery anchor")
    if state.stamina_current >= config.maximum_stamina:
        return config.maximum_stamina, now_s
    recovered = (now_s - state.stamina_recovery_anchor_s) // config.stamina_recovery_seconds
    if recovered <= 0:
        return state.stamina_current, state.stamina_recovery_anchor_s
    recovered = min(recovered, config.maximum_stamina - state.stamina_current)
    current = state.stamina_current + recovered
    if current >= config.maximum_stamina:
        return config.maximum_stamina, now_s
    return (
        current,
        state.stamina_recovery_anchor_s
        + recovered * config.stamina_recovery_seconds,
    )


def _recover_stamina(state: OneStarAccountState, config: OneStarRulesConfig, now_s: int) -> None:
    current, anchor = effective_one_star_stamina(state, config, now_s)
    state.stamina_current = current
    state.stamina_recovery_anchor_s = anchor


def effective_one_star_discretionary_funds(
    state: OneStarAccountState,
    config: OneStarRulesConfig,
    now_s: int,
) -> tuple[int, int]:
    """Project configured periodic external funds without mutating state."""

    authority = config.gem_purchase
    if authority is None:
        return state.discretionary_funds, state.funds_accrual_anchor_s
    if now_s < state.funds_accrual_anchor_s:
        raise OneStarTransactionError(
            "canonical time cannot precede discretionary-funds accrual anchor"
        )
    periods = (
        now_s - state.funds_accrual_anchor_s
    ) // authority.income_interval_seconds
    if periods <= 0:
        return state.discretionary_funds, state.funds_accrual_anchor_s
    return (
        state.discretionary_funds + periods * authority.periodic_income,
        state.funds_accrual_anchor_s
        + periods * authority.income_interval_seconds,
    )


def _recover_discretionary_funds(
    state: OneStarAccountState,
    config: OneStarRulesConfig,
    now_s: int,
) -> None:
    current, anchor = effective_one_star_discretionary_funds(
        state,
        config,
        now_s,
    )
    state.discretionary_funds = current
    state.funds_accrual_anchor_s = anchor


def format_one_star_discretionary_funds(
    authority: OneStarGemPurchaseConfig,
    amount: int,
) -> str:
    """Render one configured external-funds amount for player-facing text."""

    if authority.funds_label in {"$", "£", "€", "¥"}:
        return f"{authority.funds_label}{amount}"
    return f"{amount} {authority.funds_label}"


def _validate_hero_progression_state(
    hero: OneStarHeroState,
    config: OneStarRulesConfig,
) -> None:
    if hero.current_stars not in config.star_level_caps:
        raise OneStarTransactionError("Hero current stars have no configured level cap")
    if hero.birth_stars not in config.star_level_caps:
        raise OneStarTransactionError("Hero birth stars have no configured level cap")
    if hero.potential_grade not in config.star_level_caps:
        raise OneStarTransactionError("Hero potential grade has no configured level cap")
    if hero.level > config.star_level_caps[hero.current_stars]:
        raise OneStarTransactionError("Hero level exceeds configured current-star cap")
    if hero.experience_points < experience_to_reach_level(hero.level, config):
        raise OneStarTransactionError("Hero XP falls below its reached level threshold")
    if (
        hero.level < config.star_level_caps[hero.current_stars]
        and hero.experience_points >= experience_to_reach_level(hero.level + 1, config)
    ):
        raise OneStarTransactionError("Hero XP must immediately realize reachable levels")
    if hero.level == config.star_level_caps[hero.current_stars] and (
        hero.experience_points
        > experience_to_reach_level(hero.level + config.progression.cap_bank_extra_levels, config)
    ):
        raise OneStarTransactionError("Hero XP exceeds its configured cap bank")
    if set(hero.stats) != set(config.progression.stat_ids):
        raise OneStarTransactionError("Hero stats must match configured progression stats")
    if hero.strong_stat_id not in hero.stats or hero.weak_stat_id not in hero.stats:
        raise OneStarTransactionError("Hero affinities must match configured progression stats")
    if hero.strong_stat_id == hero.weak_stat_id:
        raise OneStarTransactionError("Hero affinities must differ")
    if any(value <= 0 for value in hero.stats.values()):
        raise OneStarTransactionError("Hero progression stats must remain positive")
    middle_stat_id = next(
        stat_id
        for stat_id in config.progression.stat_ids
        if stat_id not in {hero.strong_stat_id, hero.weak_stat_id}
    )
    if not (
        hero.stats[hero.strong_stat_id]
        > hero.stats[middle_stat_id]
        > hero.stats[hero.weak_stat_id]
    ):
        raise OneStarTransactionError(
            "Hero progression stats must preserve strong, middle, weak order"
        )
    if hero.hp_current < 0 or hero.hp_current > hero.hp_max:
        raise OneStarTransactionError("Hero current HP violates its maximum")
    if not all(item.item_id for item in hero.equipment) or len(
        {item.item_id for item in hero.equipment}
    ) != len(hero.equipment):
        raise OneStarTransactionError(
            "Hero equipment ids must be non-empty and unique"
        )
    if not all(skill.skill_id for skill in hero.skills) or len(
        {skill.skill_id for skill in hero.skills}
    ) != len(hero.skills):
        raise OneStarTransactionError(
            "Hero skill ids must be non-empty and unique"
        )
    expected = hero.model_copy(deep=True)
    try:
        rebalance_hero(hero=expected, config=config)
    except ValueError as exc:
        raise OneStarTransactionError(
            "Hero deterministic progression inputs are invalid"
        ) from exc
    if expected.stats != hero.stats or expected.hp_max != hero.hp_max:
        raise OneStarTransactionError(
            "Hero stats and maximum HP do not match deterministic progression authority"
        )


def _validate_all_hero_progression_states(
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
) -> None:
    for character in checkpoint.characters:
        hero = load_one_star_hero(character)
        if hero is not None:
            _validate_hero_progression_state(hero, config)


def validate_one_star_hero_state(
    hero: OneStarHeroState,
    config: OneStarRulesConfig,
) -> None:
    """Validate a generated or seeded Hero against story-authored bounds."""

    _validate_hero_progression_state(hero, config)


def one_star_transaction_cull_ids(
    checkpoint: CheckpointFile,
    *,
    event_id: str,
) -> tuple[str, ...]:
    """Return Hero identities first terminally culled by this committed event."""

    return tuple(
        character.character_id
        for character in checkpoint.characters
        if character.status == CharacterStatus.culled
        and (hero := load_one_star_hero(character)) is not None
        and hero.terminal_event_id == event_id
    )


def _apply_inventory_delta(
    operation: OneStarInventoryDeltaOperation,
    state: OneStarAccountState,
) -> None:
    if operation.item_id in _ACCOUNT_RESOURCE_FIELDS:
        if operation.item_id == "gems" and operation.quantity_delta > 0:
            raise OneStarTransactionError(
                "positive Gem changes require a gem_purchase or a configured "
                "adapter reward operation"
            )
        current = int(getattr(state.resources, operation.item_id))
        updated = current + operation.quantity_delta
        if updated < 0:
            raise OneStarTransactionError(
                "resource delta would consume unavailable "
                f"{operation.item_id}"
            )
        setattr(state.resources, operation.item_id, updated)
        return

    current = state.inventory.get(operation.item_id, 0)
    updated = current + operation.quantity_delta
    if updated < 0:
        raise OneStarTransactionError(
            "inventory delta would consume unavailable items"
        )
    if updated:
        state.inventory[operation.item_id] = updated
    else:
        state.inventory.pop(operation.item_id, None)


def _apply_gem_purchase(
    operation: OneStarGemPurchaseOperation,
    state: OneStarAccountState,
    config: OneStarRulesConfig,
) -> None:
    authority = config.gem_purchase
    if authority is None:
        raise OneStarTransactionError(
            "Gem purchases are not configured for this One-Star account"
        )
    packs, remainder = divmod(operation.gem_quantity, authority.gems_granted)
    if remainder:
        raise OneStarTransactionError(
            "Gem purchase quantity must be an exact configured pack multiple"
        )
    funds_cost = packs * authority.funds_cost
    if state.discretionary_funds < funds_cost:
        raise OneStarTransactionError(
            "Gem purchase exceeds available discretionary funds"
        )
    state.discretionary_funds -= funds_cost
    state.resources.gems += operation.gem_quantity


def _apply_catalogue_effects(entry: object, state: OneStarAccountState) -> None:
    """Apply only seed-authored structural effects after their price is paid."""

    # Kept structural rather than story-named so the same catalogue shape can
    # author an expansion, a workshop, or another setting's equivalent.
    resulting_lobby_floor = getattr(entry, "resulting_lobby_floor", 0)
    resulting_capacity = getattr(entry, "resulting_capacity", 0)
    if resulting_lobby_floor:
        if resulting_lobby_floor != state.lobby_floor + 1:
            raise OneStarTransactionError("catalogue lobby floor effect must advance exactly one floor")
        state.lobby_floor = resulting_lobby_floor
    if resulting_capacity:
        if resulting_capacity < state.capacity:
            raise OneStarTransactionError("catalogue effect cannot reduce lobby capacity")
        state.capacity = resulting_capacity


def _apply_catalogue(
    operation: OneStarCatalogueApplyOperation,
    state: OneStarAccountState,
    config: OneStarRulesConfig,
) -> None:
    entry = config.catalogue.get(operation.catalogue_id)
    if entry is None:
        raise OneStarTransactionError("catalogue operation references an unknown entry")
    if state.lobby_floor < entry.required_lobby_floor:
        raise OneStarTransactionError(
            "catalogue operation requires a higher lobby floor"
        )
    if state.highest_cleared_floor < entry.required_cleared_floor:
        raise OneStarTransactionError(
            "catalogue operation requires a cleared Tower floor"
        )
    if entry.kind == "purchase":
        _spend_resources(
            state.resources,
            _multiply_cost(entry.cost, operation.quantity),
        )
        state.inventory[entry.inventory_item_id] = (
            state.inventory.get(entry.inventory_item_id, 0) + operation.quantity
        )
        return
    if operation.quantity != 1:
        raise OneStarTransactionError(
            "non-purchase catalogue operations require quantity one"
        )

    if entry.kind == "facility_build":
        if state.facilities.get(entry.facility_id, 0) != 0:
            raise OneStarTransactionError(
                "facility build targets an existing facility"
            )
        _spend_resources(state.resources, entry.cost)
        state.facilities[entry.facility_id] = entry.target_level
        _apply_catalogue_effects(entry, state)
        return

    if entry.kind == "facility_upgrade":
        if entry.resulting_lobby_floor:
            if (
                entry.resulting_lobby_floor != state.lobby_floor + 1
                or entry.target_level != entry.resulting_lobby_floor
            ):
                raise OneStarTransactionError(
                    "lobby upgrade must advance exactly one configured floor"
                )
        else:
            current = state.facilities.get(entry.facility_id, 0)
            if current <= 0 or entry.target_level != current + 1:
                raise OneStarTransactionError(
                    "facility upgrade must advance exactly one configured level"
                )
        _spend_resources(state.resources, entry.cost)
        if not entry.resulting_lobby_floor:
            state.facilities[entry.facility_id] = entry.target_level
        _apply_catalogue_effects(entry, state)
        return

    if not entry.research_key:
        raise OneStarTransactionError(
            "research catalogue entry has no configured research key"
        )
    current = state.research_levels.get(entry.research_key, 0)
    if entry.research_level != current + 1:
        raise OneStarTransactionError(
            "research must advance exactly one configured level"
        )
    _spend_resources(state.resources, entry.cost)
    state.research_levels[entry.research_key] = entry.research_level


def _resource_amount_text(cost: OneStarCost) -> str:
    labels = {
        "gold": "Gold",
        "gems": "Gems",
        "building_resources": "Building Resources",
    }
    pieces = [
        f"{int(getattr(cost, field))} {label}"
        for field, label in labels.items()
        if int(getattr(cost, field))
    ]
    pieces.extend(
        f"{quantity} {material_id.replace('_', ' ').title()}"
        for material_id, quantity in sorted(cost.materials.items())
        if quantity
    )
    if not pieces:
        return "no resources"
    if len(pieces) == 1:
        return pieces[0]
    return ", ".join(pieces[:-1]) + f" and {pieces[-1]}"


def _available_amount_text(
    resources: OneStarResources,
    cost: OneStarCost,
) -> str:
    labels = {
        "gold": "Gold",
        "gems": "Gems",
        "building_resources": "Building Resources",
    }
    pieces = [
        f"{int(getattr(resources, field))} {label}"
        for field, label in labels.items()
        if int(getattr(cost, field))
    ]
    pieces.extend(
        f"{resources.materials.get(material_id, 0)} "
        f"{material_id.replace('_', ' ').title()}"
        for material_id, quantity in sorted(cost.materials.items())
        if quantity
    )
    if len(pieces) == 1:
        return pieces[0]
    return ", ".join(pieces[:-1]) + f" and {pieces[-1]}"


def preflight_one_star_account_updates(
    checkpoint: CheckpointFile,
    state_updates: Iterable[OneStarStateUpdate],
    *,
    initiating_actor_id: str,
    canonical_at_s: int | None = None,
) -> None:
    """Reject known-impossible player account controls before side effects.

    This simulates only deterministic Gem purchases, catalogue/inventory
    changes, and a following standard summon. Other semantics fall through to
    normal atomic validation rather than risking a false player rejection.
    """

    updates = list(state_updates)
    summon_count = sum(update.kind == "summon" for update in updates)
    if summon_count > 1 or not any(
        update.kind in {"gem_purchase", "summon"}
        or (
            update.kind == "inventory_delta"
            and update.target_id.strip() == "gems"
        )
        for update in updates
    ):
        return
    try:
        owner, account = load_one_star_account(checkpoint)
    except (OneStarTransactionError, ValidationError, ValueError):
        # Malformed model output still follows the ordinary validation/repair
        # path. Only an established, deterministic player constraint belongs
        # on the player-facing rejection surface.
        return

    if initiating_actor_id.strip() != owner.character_id:
        return

    state = account.state.model_copy(deep=True)
    ledger_now_s = max(
        checkpoint.session.leading_at_s,
        canonical_at_s or 0,
        state.funds_accrual_anchor_s,
    )
    _recover_discretionary_funds(state, account.config, ledger_now_s)
    for update in updates:
        if update.kind in {"inventory_delta", "catalogue_apply"}:
            if (
                update.kind == "inventory_delta"
                and update.target_id.strip() == "gems"
            ):
                try:
                    gem_delta = _integer_update_value(
                        update.value,
                        label="Gem delta",
                    )
                except OneStarTransactionError:
                    return
                if gem_delta > 0:
                    authority = account.config.gem_purchase
                    if authority is None:
                        message = (
                            "A direct Gem grant is not an available account "
                            "action. Nothing was charged and no Gems were "
                            "added."
                        )
                    else:
                        message = (
                            "Gems are sold only in packs of "
                            f"{authority.gems_granted} for "
                            f"{format_one_star_discretionary_funds(authority, authority.funds_cost)}. "
                            "Nothing was charged and no Gems were added."
                        )
                    raise PlayerActionRejected(
                        message,
                        reason="one_star_gem_purchase_rejected",
                    )
            try:
                operation = one_star_state_updates_to_transaction(
                    checkpoint,
                    [update],
                    canonical_at_s=checkpoint.session.leading_at_s,
                ).operations[0]
                if isinstance(operation, OneStarInventoryDeltaOperation):
                    _apply_inventory_delta(operation, state)
                elif isinstance(operation, OneStarCatalogueApplyOperation):
                    _apply_catalogue(operation, state, account.config)
                else:  # Defensive: each update translates to one operation.
                    return
            except (
                IndexError,
                OneStarTransactionError,
                ValidationError,
                ValueError,
            ):
                return
            continue
        if update.kind == "gem_purchase":
            authority = account.config.gem_purchase
            if update.target_id.strip() != "gems" or update.details:
                return
            try:
                gem_quantity = _integer_update_value(
                    update.value,
                    label="Gem-purchase quantity",
                )
            except OneStarTransactionError:
                return
            if authority is None:
                raise PlayerActionRejected(
                    "Gem purchases are not available for this System account. "
                    "Nothing was charged and no Gems were added.",
                    reason="one_star_gem_purchase_rejected",
                )
            if gem_quantity < 1 or gem_quantity % authority.gems_granted:
                raise PlayerActionRejected(
                    "Gems are sold in packs of "
                    f"{authority.gems_granted} for "
                    f"{format_one_star_discretionary_funds(authority, authority.funds_cost)}; "
                    f"{gem_quantity} Gems cannot be purchased. Nothing was "
                    "charged and no Gems were added.",
                    reason="one_star_gem_purchase_rejected",
                )
            packs = gem_quantity // authority.gems_granted
            funds_cost = packs * authority.funds_cost
            if state.discretionary_funds < funds_cost:
                raise PlayerActionRejected(
                    f"{gem_quantity} Gems cost "
                    f"{format_one_star_discretionary_funds(authority, funds_cost)}, "
                    "but only "
                    f"{format_one_star_discretionary_funds(authority, state.discretionary_funds)} "
                    "is available. Nothing was charged and no Gems were added.",
                    reason="one_star_gem_purchase_rejected",
                )
            _apply_gem_purchase(
                OneStarGemPurchaseOperation(
                    operation="gem_purchase",
                    gem_quantity=gem_quantity,
                ),
                state,
                account.config,
            )
            continue
        if update.kind != "summon":
            return

        pool_id = update.target_id.strip()
        pool = account.config.summon_pools.get(pool_id)
        if (
            pool is None
            or pool.usage != "standard"
            or update.details
        ):
            return

        try:
            count = _integer_update_value(update.value, label="summon count")
        except OneStarTransactionError:
            return
        if count < 1:
            return
        pool_label = pool_id.replace("_", " ").strip().title()
        if count > account.config.max_summon_batch:
            raise PlayerActionRejected(
                f"{pool_label} summons allow at most "
                f"{account.config.max_summon_batch} pulls at once; "
                f"{count} were requested. Nothing was spent and no Heroes "
                "were summoned.",
                reason="one_star_summon_rejected",
            )
        total_cost = _multiply_cost(pool.cost, count)
        insufficient = (
            state.resources.gold < total_cost.gold
            or state.resources.gems < total_cost.gems
            or state.resources.building_resources
            < total_cost.building_resources
            or any(
                state.resources.materials.get(material_id, 0) < quantity
                for material_id, quantity in total_cost.materials.items()
            )
        )
        if insufficient:
            raise PlayerActionRejected(
                f"{pool_label} summon rejected: {count} "
                f"{'pull' if count == 1 else 'pulls'} cost "
                f"{_resource_amount_text(total_cost)}, but only "
                f"{_available_amount_text(state.resources, total_cost)} "
                "are available. "
                "Nothing was spent and no Heroes were summoned.",
                reason="one_star_summon_rejected",
            )

        occupied = sum(
            1
            for character in checkpoint.characters
            if character.status != CharacterStatus.culled
            and (hero := load_one_star_hero(character)) is not None
            and hero.owner_lobby_id == account.config.lobby_id
        )
        open_slots = max(0, state.capacity - occupied)
        if count > open_slots:
            raise PlayerActionRejected(
                f"{pool_label} summon rejected: {count} "
                f"{'Hero was' if count == 1 else 'Heroes were'} requested, "
                f"but the lobby has {open_slots} open "
                f"{'slot' if open_slots == 1 else 'slots'}. Nothing was "
                "spent and no Heroes were summoned.",
                reason="one_star_summon_rejected",
            )
        return


def _apply_summon(
    operation: OneStarSummonOperation,
    state: OneStarAccountState,
    config: OneStarRulesConfig,
    checkpoint: CheckpointFile,
    event_id: str,
    hero_initializations: dict[str, OneStarHeroState],
    expected_summon_ids: set[str],
    spawned_ids: set[str],
    activated_ids: set[str],
    activated_locations: Mapping[str, str],
    initiating_actor_id: str,
    configured_arrival_location: str,
) -> None:
    pool = config.summon_pools.get(operation.pool_id)
    if pool is None:
        raise OneStarTransactionError("summon references an unknown configured pool")
    if len(operation.hero_ids) > config.max_summon_batch:
        raise OneStarTransactionError("summon exceeds configured maximum batch size")
    if set(operation.hero_ids) != expected_summon_ids:
        raise OneStarTransactionError(
            "summon ids must exactly match fresh spawns and unowned Hero activations"
        )
    if isinstance(pool, OneStarStandardSummonPool):
        expected_draws = _one_star_summon_draw_preview(
            checkpoint,
            config,
            state,
            pool_id=operation.pool_id,
            count=len(operation.hero_ids),
        )
    else:
        expected_draws = _one_star_opening_roster_preview(
            checkpoint,
            pool_id=operation.pool_id,
            pool=pool,
        )
        account_owner = find_one_star_account_owner(checkpoint.characters)
        bound_actor_ids = {
            slot.character_id
            for slot in pool.slots
            if isinstance(slot, OneStarOpeningRosterBoundPlayerActorSlot)
        }
        has_authored_non_bound_slot = any(
            not isinstance(slot, OneStarOpeningRosterBoundPlayerActorSlot)
            for slot in pool.slots
        )
        authorized_actor = (
            (
                account_owner is not None
                and initiating_actor_id == account_owner.character_id
            )
            if has_authored_non_bound_slot
            else initiating_actor_id in bound_actor_ids
        )
        if (
            set(operation.hero_ids) != set(expected_summon_ids)
            or not set(operation.hero_ids).issubset(activated_ids)
            or spawned_ids
            or state.applied_event_fingerprints
            or not authorized_actor
        ):
            raise OneStarTransactionError(
                "opening-roster summon must be its authorized actor's first "
                "event and activate only its configured existing Heroes"
            )
        if any(
            character.status != CharacterStatus.culled
            and (hero := load_one_star_hero(character)) is not None
            and hero.owner_lobby_id == config.lobby_id
            for character in checkpoint.characters
        ):
            raise OneStarTransactionError(
                "opening-roster summon requires an account with no acquired Heroes"
            )

    expected_ids = [
        draw.existing_character_id
        or _fresh_summon_character_id(
            lobby_id=config.lobby_id,
            pool_id=operation.pool_id,
            draw_index=(
                state.summon_draw_counters.get(operation.pool_id, 0) + offset
            ),
        )
        for offset, draw in enumerate(expected_draws)
    ]
    expected_birth_stars = [draw.birth_stars for draw in expected_draws]
    if (
        operation.hero_ids != expected_ids
        or operation.birth_stars != expected_birth_stars
    ):
        raise OneStarTransactionError(
            "summon identities and birth stars must match the exact adapter preview"
        )
    for hero_id, draw in zip(operation.hero_ids, expected_draws, strict=True):
        if draw.existing_character_id:
            if hero_id not in activated_ids or hero_id in spawned_ids:
                raise OneStarTransactionError(
                    "summon must activate the exact existing Hero selected by "
                    "the adapter"
                )
        elif hero_id not in spawned_ids or hero_id in activated_ids:
            raise OneStarTransactionError(
                "a fresh weighted summon result requires a matching new Hero spawn"
            )

    if isinstance(pool, OneStarStandardSummonPool):
        _spend_resources(
            state.resources,
            _multiply_cost(pool.cost, len(operation.hero_ids)),
        )
    occupied = sum(
        1
        for character in checkpoint.characters
        if character.status != CharacterStatus.culled
        and (hero := load_one_star_hero(character)) is not None
        and hero.owner_lobby_id == config.lobby_id
    )
    if occupied + len(operation.hero_ids) > state.capacity:
        raise OneStarTransactionError("summon exceeds configured lobby capacity")
    for hero_id, birth_stars in zip(operation.hero_ids, operation.birth_stars, strict=True):
        existing = next((item for item in checkpoint.characters if item.character_id == hero_id), None)
        if existing is None:
            raise OneStarTransactionError("generic summon records must be staged before One-Star preparation")
        existing_hero = load_one_star_hero(existing)
        if existing_hero is None:
            raise OneStarTransactionError(
                "summoned records must carry a generated or reserve Hero sheet"
            )
        hero = existing_hero
        if hero_id in spawned_ids:
            if (
                not isinstance(pool, OneStarStandardSummonPool)
                or not pool.fresh_generation_allowed
            ):
                raise OneStarTransactionError(
                    "summon pool does not allow fresh generation"
                )
            arrival_location = existing.location
        elif hero_id in activated_ids:
            if (
                isinstance(pool, OneStarStandardSummonPool)
                and hero_id not in pool.eligible_existing_ids
            ):
                raise OneStarTransactionError(
                    "existing summon identity is not eligible for this pool"
                )
            if existing.status != CharacterStatus.dormant:
                raise OneStarTransactionError(
                    "existing summon reserves must be dormant before activation"
                )
            arrival_location = activated_locations[hero_id] or existing.location
        else:  # Defensive: exact set equality above makes this unreachable.
            raise OneStarTransactionError(
                "summon identity lacks a matching spawn or activation signal"
            )
        expected_arrival_location = (
            configured_arrival_location
            if isinstance(pool, OneStarOpeningRosterSummonPool)
            else config.lobby_location_label
        )
        if arrival_location != expected_arrival_location:
            if isinstance(pool, OneStarOpeningRosterSummonPool):
                raise OneStarTransactionError(
                    "opening-roster Heroes must arrive at the configured "
                    "mission destination"
                )
            raise OneStarTransactionError(
                "summoned Heroes must arrive at the configured lobby"
            )
        if hero.owner_lobby_id or hero.acquisition_event_id:
            raise OneStarTransactionError(
                "summon result must be an unowned, not-yet-acquired Hero"
            )
        if hero.birth_stars != birth_stars:
            raise OneStarTransactionError(
                "summon must preserve the Hero's immutable birth stars"
            )
        hero.owner_lobby_id = config.lobby_id
        hero.acquisition_event_id = event_id
        # A newly acquired Hero is eligible for normal autonomous follow-up
        # once the summon has made them part of this lobby.  This is engine
        # scheduling policy, not an actor-authored intention.
        if existing.actor is None:
            existing.actor = ActorRecord(may_act_offstage=True)
        else:
            existing.actor.may_act_offstage = True
        _validate_hero_progression_state(hero, config)
        hero_initializations[hero_id] = hero
        _store_hero(existing, hero)
    if isinstance(pool, OneStarStandardSummonPool):
        state.summon_draw_counters[operation.pool_id] = (
            state.summon_draw_counters.get(operation.pool_id, 0)
            + len(expected_draws)
        )


def _direct_opening_operations(
    transaction: OneStarTransaction,
    config: OneStarRulesConfig,
) -> tuple[OneStarSummonOperation, OneStarMissionStartOperation] | None:
    """Resolve the one atomic opening-roster acquisition contract."""

    opening_summons = [
        operation
        for operation in transaction.operations
        if isinstance(operation, OneStarSummonOperation)
        and isinstance(
            config.summon_pools.get(operation.pool_id),
            OneStarOpeningRosterSummonPool,
        )
    ]
    direct_mission_starts = [
        operation
        for operation in transaction.operations
        if isinstance(operation, OneStarMissionStartOperation)
        and not operation.pending_operation_id
    ]
    if not opening_summons and not direct_mission_starts:
        return None
    if (
        len(transaction.operations) != 2
        or len(opening_summons) != 1
        or len(direct_mission_starts) != 1
        or transaction.operations[0] is not opening_summons[0]
        or transaction.operations[1] is not direct_mission_starts[0]
    ):
        raise OneStarTransactionError(
            "an opening-roster summon and direct mission start must be the "
            "only operations in their event, in that order"
        )
    summon = opening_summons[0]
    mission_start = direct_mission_starts[0]
    if mission_start.mission.party_ids != summon.hero_ids:
        raise OneStarTransactionError(
            "a direct opening mission party must preserve the exact "
            "opening-roster order"
        )
    return summon, mission_start


def _apply_hero_delta(
    operation: OneStarHeroDeltaOperation,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
    event_id: str,
) -> str:
    character, hero = _require_local_active_hero(checkpoint, operation.hero_id, config)
    if operation.hp_current is not None:
        hero.hp_current = operation.hp_current
    remove_equipment = set(operation.equipment_remove_ids)
    if len(remove_equipment) != len(operation.equipment_remove_ids):
        raise OneStarTransactionError("Hero equipment removal ids must be unique")
    if not remove_equipment.issubset({item.item_id for item in hero.equipment}):
        raise OneStarTransactionError("Hero equipment removal references an unknown item")
    hero.equipment = [item for item in hero.equipment if item.item_id not in remove_equipment]
    added_equipment_ids = [item.item_id for item in operation.equipment_add]
    if len(added_equipment_ids) != len(set(added_equipment_ids)):
        raise OneStarTransactionError("Hero equipment additions must be unique")
    if {item.item_id for item in hero.equipment} & set(added_equipment_ids):
        raise OneStarTransactionError("Hero equipment addition duplicates an item id")
    hero.equipment.extend(operation.equipment_add)
    remove_skills = set(operation.skills_remove_ids)
    if len(remove_skills) != len(operation.skills_remove_ids):
        raise OneStarTransactionError("Hero skill removal ids must be unique")
    if not remove_skills.issubset({skill.skill_id for skill in hero.skills}):
        raise OneStarTransactionError("Hero skill removal references an unknown skill")
    hero.skills = [skill for skill in hero.skills if skill.skill_id not in remove_skills]
    added_skill_ids = [skill.skill_id for skill in operation.skills_add]
    if len(added_skill_ids) != len(set(added_skill_ids)):
        raise OneStarTransactionError("Hero skill additions must be unique")
    if {skill.skill_id for skill in hero.skills} & set(added_skill_ids):
        raise OneStarTransactionError("Hero skill addition duplicates a skill id")
    hero.skills.extend(operation.skills_add)
    equipment_by_id = {item.item_id: item for item in hero.equipment}
    if len({update.item_id for update in operation.equipment_durability}) != len(
        operation.equipment_durability
    ):
        raise OneStarTransactionError("durability updates must be unique")
    for update in operation.equipment_durability:
        item = equipment_by_id.get(update.item_id)
        if item is None:
            raise OneStarTransactionError("durability update references an unknown equipment item")
        if item.durability_max and update.durability_current > item.durability_max:
            raise OneStarTransactionError("durability update exceeds item maximum")
        item.durability_current = update.durability_current
    skills_by_id = {skill.skill_id: skill for skill in hero.skills}
    if len({update.skill_id for update in operation.skill_rank_updates}) != len(
        operation.skill_rank_updates
    ):
        raise OneStarTransactionError("skill rank updates must be unique")
    for update in operation.skill_rank_updates:
        skill = skills_by_id.get(update.skill_id)
        if skill is None:
            raise OneStarTransactionError("skill rank update references an unknown skill")
        skill.rank = update.rank
    if operation.conditions is not None:
        hero.conditions = operation.conditions
    if operation.persistent_injuries is not None:
        hero.persistent_injuries = operation.persistent_injuries
    if hero.hp_current == 0 and operation.terminal_action != "death":
        raise OneStarTransactionError("zero HP requires explicit death semantics")
    if operation.terminal_action == "death":
        if hero.hp_current != 0:
            raise OneStarTransactionError("death semantics require zero HP")
        if not operation.death_cause.strip():
            raise OneStarTransactionError("death requires a cause")
        hero.terminal_cause = operation.death_cause.strip()
        hero.terminal_event_id = event_id
        character.status = CharacterStatus.culled
    elif operation.death_cause.strip():
        raise OneStarTransactionError(
            "death cause requires an explicit death action"
        )
    try:
        validated_hero = OneStarHeroState.model_validate(
            hero.model_dump(mode="python")
        )
    except ValidationError as exc:
        raise OneStarTransactionError(
            "Hero delta produced an invalid durable Hero state"
        ) from exc
    _validate_hero_progression_state(validated_hero, config)
    _store_hero(character, validated_hero)
    return character.name if operation.terminal_action == "death" else ""


def _validate_global_equipment_ids(
    checkpoint: CheckpointFile,
    state: OneStarAccountState,
) -> None:
    """Require a stable item id to identify exactly one durable record."""

    owners: dict[str, str] = {}
    for item in state.stored_equipment:
        if item.item_id in owners:
            raise OneStarTransactionError(
                f"equipment item id {item.item_id!r} is duplicated in account storage"
            )
        owners[item.item_id] = "account storage"
    for character in checkpoint.characters:
        hero = load_one_star_hero(character)
        if hero is None:
            continue
        for item in hero.equipment:
            previous = owners.get(item.item_id)
            if previous is not None:
                raise OneStarTransactionError(
                    f"equipment item id {item.item_id!r} is shared by {previous} "
                    f"and Hero {character.character_id!r}"
                )
            owners[item.item_id] = f"Hero {character.character_id!r}"


def _apply_equipment_move(
    operation: OneStarEquipmentMoveOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
) -> None:
    """Move one exact record between local lobby Heroes and account storage."""

    item_id = operation.item_id
    matches: list[tuple[str, CharacterRecord | None, OneStarEquipmentEntry]] = []
    for item in state.stored_equipment:
        if item.item_id == item_id:
            matches.append(("account", None, item))
    for character in checkpoint.characters:
        hero = load_one_star_hero(character)
        if hero is None:
            continue
        for item in hero.equipment:
            if item.item_id == item_id:
                matches.append(("hero", character, item))
    if len(matches) != 1:
        raise OneStarTransactionError(
            "equipment move must identify exactly one durable item record"
        )
    source_kind, source_character, item = matches[0]
    destination = operation.destination
    if destination == "account":
        if source_kind == "account":
            raise OneStarTransactionError("equipment move cannot leave an item in place")
        if source_character is None:  # pragma: no cover - source_kind guards it.
            raise OneStarTransactionError("equipment move has no Hero source")
        source_character, source = _require_local_active_hero(
            checkpoint,
            source_character.character_id,
            config,
        )
        if source_character.location != config.lobby_location_label:
            raise OneStarTransactionError("equipment moves require a lobby source Hero")
        if (
            state.active_mission is not None
            and source_character.character_id in state.active_mission.party_ids
        ):
            raise OneStarTransactionError(
                "active-mission equipment controls cannot affect mission party Heroes"
            )
        source.equipment = [entry for entry in source.equipment if entry.item_id != item_id]
        state.stored_equipment.append(item.model_copy(deep=True))
        _store_hero(source_character, source)
        return

    destination_character, destination_hero = _require_local_active_hero(
        checkpoint,
        destination,
        config,
    )
    if destination_character.location != config.lobby_location_label:
        raise OneStarTransactionError("equipment moves require a lobby destination Hero")
    if (
        state.active_mission is not None
        and destination_character.character_id in state.active_mission.party_ids
    ):
        raise OneStarTransactionError(
            "active-mission equipment controls cannot affect mission party Heroes"
        )
    if source_kind == "hero":
        if source_character is None:  # pragma: no cover - source_kind guards it.
            raise OneStarTransactionError("equipment move has no Hero source")
        source_character, source = _require_local_active_hero(
            checkpoint,
            source_character.character_id,
            config,
        )
        if source_character.location != config.lobby_location_label:
            raise OneStarTransactionError("equipment moves require a lobby source Hero")
        if (
            state.active_mission is not None
            and source_character.character_id in state.active_mission.party_ids
        ):
            raise OneStarTransactionError(
                "active-mission equipment controls cannot affect mission party Heroes"
            )
        if source_character.character_id == destination_character.character_id:
            raise OneStarTransactionError("equipment move cannot leave an item in place")
        source.equipment = [entry for entry in source.equipment if entry.item_id != item_id]
        _store_hero(source_character, source)
    else:
        state.stored_equipment = [
            entry for entry in state.stored_equipment if entry.item_id != item_id
        ]
    if any(entry.item_id == item_id for entry in destination_hero.equipment):
        raise OneStarTransactionError("equipment move duplicates a destination item id")
    destination_hero.equipment.append(item.model_copy(deep=True))
    _store_hero(destination_character, destination_hero)


def _apply_mission_start(
    operation: OneStarMissionStartOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
    now_s: int,
    resolved_deployments: Mapping[str, object],
    direct_opening_party_ids: tuple[str, ...] | None = None,
    direct_opening_locations: Mapping[str, str] | None = None,
) -> None:
    if state.active_mission is not None:
        raise OneStarTransactionError("cannot start a mission while another is active")
    if operation.mission.started_at_s != now_s:
        raise OneStarTransactionError("mission start must use the canonical event time")
    direct_opening = direct_opening_party_ids is not None
    if direct_opening:
        if (
            operation.pending_operation_id
            or state.pending_operation is not None
            or state.highest_cleared_floor != 0
            or operation.mission.floor != state.highest_unlocked_floor
        ):
            raise OneStarTransactionError(
                "a direct opening mission must be the account's first unlocked "
                "floor and cannot use or bypass a pending operation"
            )
        pending_party = set(direct_opening_party_ids)
        expected_destination = operation.mission.destination
    else:
        pending = resolved_deployments.get(operation.pending_operation_id)
        if pending is None:
            raise OneStarTransactionError(
                "mission start requires a deployment resolved in this event"
            )
        pending_party = set(getattr(pending, "participant_ids", ()))
        expected_destination = getattr(pending, "destination", "")
    if set(operation.mission.party_ids) != pending_party:
        raise OneStarTransactionError(
            "mission party must exactly match its deployment authority"
        )
    if operation.mission.destination != expected_destination:
        raise OneStarTransactionError(
            "mission destination must match its deployment authority"
        )
    if operation.mission.floor > state.highest_unlocked_floor:
        raise OneStarTransactionError("mission targets a locked Tower floor")
    scenario = config.floor_scenarios.get(operation.mission.floor)
    if scenario is None:
        raise OneStarTransactionError(
            "mission targets a floor with no reviewed scenario authority"
        )
    if operation.mission.floor not in config.floor_rewards:
        raise OneStarTransactionError(
            "mission targets a floor with no reviewed reward authority"
        )
    if (
        operation.mission.mission_id != scenario.mission_id
        or operation.mission.destination != scenario.destination
        or operation.mission.completion_declaration
        != scenario.completion_declaration
        or operation.mission.failure_declaration != scenario.failure_declaration
        or operation.mission.counters != scenario.counters
    ):
        raise OneStarTransactionError(
            "mission start must exactly match its reviewed floor scenario"
        )
    for hero_id in operation.mission.party_ids:
        if direct_opening:
            character = _require_character(checkpoint, hero_id)
            hero = load_one_star_hero(character)
            location = (direct_opening_locations or {}).get(hero_id, "")
            valid_party_member = (
                hero is not None
                and hero.owner_lobby_id == config.lobby_id
                and location == operation.mission.destination
            )
        else:
            character, _hero = _require_local_active_hero(
                checkpoint, hero_id, config
            )
            location = character.location
            valid_party_member = location == operation.mission.destination
        if not valid_party_member:
            raise OneStarTransactionError(
                "mission party has not physically entered its destination"
            )
    party_ids = set(operation.mission.party_ids)
    for character in checkpoint.characters:
        if (
            character.character_id in party_ids
            or character.status != CharacterStatus.active
            or (
                (direct_opening_locations or {}).get(
                    character.character_id,
                    character.location,
                )
                != operation.mission.destination
            )
        ):
            continue
        hero = load_one_star_hero(character)
        if hero is not None and hero.owner_lobby_id == config.lobby_id:
            raise OneStarTransactionError(
                "a local Hero at the mission destination must belong to the "
                "resolved deployment party"
            )
    if state.stamina_current < config.deployment_stamina_cost:
        raise OneStarTransactionError("insufficient stamina for mission start")
    state.stamina_current -= config.deployment_stamina_cost
    state.active_mission = operation.mission


def _apply_mission_update(
    operation: OneStarMissionUpdateOperation,
    state: OneStarAccountState,
    now_s: int,
) -> None:
    mission = state.active_mission
    if mission is None or mission.mission_id != operation.mission_id:
        raise OneStarTransactionError("mission update does not target the active mission")
    if mission.deadline_at_s and now_s > mission.deadline_at_s:
        raise OneStarTransactionError(
            "mission counters cannot advance after the canonical deadline"
        )
    current_counters = {counter.counter_id: counter for counter in mission.counters}
    updated_counters = {counter.counter_id: counter for counter in operation.counters}
    if set(updated_counters) != set(current_counters):
        raise OneStarTransactionError("mission update must retain the declared counter set")
    for key, counter in updated_counters.items():
        declared = current_counters[key]
        if counter.target != declared.target or counter.current < declared.current:
            raise OneStarTransactionError("mission counters cannot change targets or regress")
    mission.counters = list(operation.counters)


def _mission_xp_award(
    hero: OneStarHeroState,
    *,
    floor: int,
    config: OneStarRulesConfig,
) -> int:
    overlevel = max(0, hero.level - floor)
    percentage = config.progression.overlevel_xp_percentages[
        min(overlevel, len(config.progression.overlevel_xp_percentages) - 1)
    ]
    return (
        config.progression.floor_xp_per_floor
        * floor
        * percentage
        // 100
    )


def _apply_mission_end(
    operation: OneStarMissionEndOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
    now_s: int,
    pre_event_escape_authorities: Mapping[str, set[str]],
    owner_character_id: str,
) -> tuple[OneStarSystemConsequence, ...]:
    mission = state.active_mission
    if mission is None or mission.mission_id != operation.mission_id:
        raise OneStarTransactionError("mission end does not target the active mission")
    if (
        operation.outcome == "completed"
        and mission.deadline_at_s
        and now_s > mission.deadline_at_s
    ):
        raise OneStarTransactionError(
            "a timed mission cannot complete after its canonical deadline"
        )
    survivors: list[tuple[CharacterRecord, OneStarHeroState]] = []
    for hero_id in mission.party_ids:
        character = _require_character(checkpoint, hero_id)
        hero = load_one_star_hero(character)
        if hero is None or hero.owner_lobby_id != config.lobby_id:
            raise OneStarTransactionError(
                "mission party contains a non-local Hero"
            )
        if character.status == CharacterStatus.active:
            survivors.append((character, hero))
    if operation.outcome == "escaped":
        authority_id = operation.escape_authority_id.strip()
        if not authority_id:
            raise OneStarTransactionError("escape requires an explicit authority id")
        has_authority = any(
            authority_id
            in pre_event_escape_authorities.get(character.character_id, set())
            for character, _hero in survivors
        )
        if not has_authority:
            raise OneStarTransactionError("escape authority is not established on the mission party")
    elif operation.escape_authority_id.strip():
        raise OneStarTransactionError("only an escaped mission may cite escape authority")
    if operation.outcome == "failed" and survivors:
        raise OneStarTransactionError(
            "a failed mission with living Heroes cannot return them through the sealed gate"
        )
    system_consequences: list[OneStarSystemConsequence] = []
    result_recipient_ids = one_star_terminal_system_recipient_ids(checkpoint)
    if operation.outcome == "completed":
        incomplete_counters = [
            counter.counter_id
            for counter in mission.counters
            if counter.current < counter.target
        ]
        if incomplete_counters:
            raise OneStarTransactionError(
                "a completed mission must satisfy every declared counter: "
                + ", ".join(sorted(incomplete_counters))
            )
        reward = config.floor_rewards.get(mission.floor)
        if reward is None:
            raise OneStarTransactionError("completed floor has no configured fixed reward")
        first_clear = mission.floor > state.highest_cleared_floor
        unlocked_floor = 0
        if first_clear:
            reward_delta = OneStarCost.model_validate(reward.model_dump())
            _add_resources(state.resources, reward_delta)
            state.highest_cleared_floor = mission.floor
            next_floor = mission.floor + 1
            if next_floor in config.floor_scenarios:
                unlocked_floor = next_floor
                state.highest_unlocked_floor = max(
                    state.highest_unlocked_floor,
                    next_floor,
                )
        else:
            gold = reward.gold * config.repeat_gold_numerator // config.repeat_gold_denominator
            if reward.gold:
                gold = max(config.repeat_gold_minimum, gold)
            reward_delta = OneStarCost(
                gold=gold,
                gems=0,
                building_resources=0,
                materials={},
            )
            _add_resources(state.resources, reward_delta)
        balance = _resource_amount_text(OneStarCost.model_validate(
            state.resources.model_dump()
        ))
        system_consequences.append(OneStarSystemConsequence(
            text=(
                f"System: Floor {mission.floor} "
                f"{'first-clear' if first_clear else 'repeat-clear'} reward "
                f"applied: {_resource_amount_text(reward_delta)}. "
                f"Account resources now: {balance}."
            ),
            recipient_character_ids=result_recipient_ids,
        ))
        if unlocked_floor:
            system_consequences.append(OneStarSystemConsequence(
                text=f"System: Tower floor {unlocked_floor} unlocked.",
                recipient_character_ids=result_recipient_ids,
            ))
    else:
        system_consequences.append(OneStarSystemConsequence(
            text=(
                f"System: Floor {mission.floor} ended as "
                f"{operation.outcome}; "
                "no floor reward or unlock was granted."
            ),
            recipient_character_ids=result_recipient_ids,
        ))
    if operation.outcome == "completed":
        for character, hero in survivors:
            award = _mission_xp_award(
                hero,
                floor=mission.floor,
                config=config,
            )
            try:
                report = apply_experience(
                    hero=hero,
                    experience_delta=award,
                    config=config,
                )
            except ValueError as exc:
                raise OneStarTransactionError(
                    "mission XP could not be applied to a surviving Hero"
                ) from exc
            _validate_hero_progression_state(hero, config)
            consequence = _progression_consequence(
                character=character,
                checkpoint=checkpoint,
                state=state,
                config=config,
                owner_character_id=owner_character_id,
                report=report,
                label="mission reward",
                include_management_observers=False,
            )
            system_consequences.extend(consequence)
    destination = operation.return_destination.strip()
    returning = operation.outcome in {"completed", "escaped"}
    if returning and destination != config.lobby_location_label:
        raise OneStarTransactionError(
            "mission end must return survivors to the configured lobby"
        )
    if not returning and destination:
        raise OneStarTransactionError(
            "failed missions with no survivors have no return destination"
        )
    for character, hero in survivors:
        character.location = destination
        if config.lobby_return_healing and hero.hp_current > 0:
            hero.hp_current = hero.hp_max
            hero.conditions = []
        _store_hero(character, hero)
    state.active_mission = None
    return tuple(system_consequences)


def _apply_pending_open(
    operation: OneStarPendingOpenOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
    now_s: int,
) -> OneStarPendingOperation:
    if state.active_mission is not None and operation.pending.kind == "deployment":
        raise OneStarTransactionError(
            "cannot open a second deployment while a Tower mission is active"
        )
    selection = operation.pending
    validate_one_star_pending_operation_shape(selection)
    if selection.opened_at_s != now_s:
        raise OneStarTransactionError("pending operation must use canonical event time")
    for hero_id in selection.participant_ids:
        character, _hero = _require_local_active_hero(
            checkpoint,
            hero_id,
            config,
        )
        if state.active_mission is not None and (
            hero_id in state.active_mission.party_ids
            or character.location != config.lobby_location_label
        ):
            raise OneStarTransactionError(
                "active-mission lobby operations require nonparty Heroes "
                "physically in the lobby"
            )
    if selection.kind in {"synthesis", "promotion"}:
        target_character, _hero = _require_local_active_hero(
            checkpoint,
            selection.target_id,
            config,
        )
        if state.active_mission is not None and (
            selection.target_id in state.active_mission.party_ids
            or target_character.location != config.lobby_location_label
        ):
            raise OneStarTransactionError(
                "active-mission lobby operations require a nonparty target "
                "physically in the lobby"
            )
    if not selection.destination:
        raise OneStarTransactionError(
            "embodied operations require a physical destination"
        )
    lobby_operation_locations = {
        requirement.required_location
        for kind, requirement in config.operation_requirements.items()
        if kind != "deployment" and requirement.required_location
    }
    if selection.kind == "deployment" and selection.destination in {
        config.lobby_location_label,
        *lobby_operation_locations,
    }:
        raise OneStarTransactionError(
            "deployment destination must cross beyond the configured lobby"
        )
    _require_operational_facility(selection, state, config)
    pending = OneStarPendingOperation(
        **selection.model_dump(mode="python"),
        synthesis_preview=(
            _synthesis_preview(selection, state, checkpoint, config)
            if selection.kind == "synthesis"
            else None
        ),
    )
    # A pending selection is reversible metadata, not an irreversible part of
    # the operation.  The account therefore has one current selection rather
    # than a queue: a later router-authored selection supersedes the earlier
    # one atomically while leaving every character's physical state untouched.
    state.pending_operation = pending
    return pending


def _require_operational_facility(
    pending: object,
    state: OneStarAccountState,
    config: OneStarRulesConfig,
) -> None:
    kind = getattr(pending, "kind", "")
    requirement = config.operation_requirements.get(kind)
    if requirement is None:
        raise OneStarTransactionError(
            "embodied operation has no configured physical requirement"
        )
    if state.facilities.get(requirement.facility_id, 0) < 1:
        raise OneStarTransactionError(
            "embodied operation requires an operational configured facility"
        )
    destination = str(getattr(pending, "destination", "") or "").strip()
    if (
        requirement.required_location
        and destination != requirement.required_location
    ):
        raise OneStarTransactionError(
            "embodied operation destination does not match its configured facility"
        )


def _restore_promotion_knowledge(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
    promoted_tier: int,
) -> None:
    rung = next(
        (
            tier
            for tier in checkpoint.world_state.knowledge_tiers
            if tier.tier == promoted_tier
        ),
        None,
    )
    if rung is None:
        return
    restored_facts = [
        text.strip()
        for text in (rung.personal_depth, rung.world_knowledge)
        if text.strip()
    ]
    # Promotion can grant bounded lore or recovered memory, but it never
    # rewrites the person's existing understanding. Keep each newly available
    # source string as its own told fact so later tiers can append only the
    # authority they add.
    if character.actor is None:
        character.actor = ActorRecord()
    known_facts = {fact.text.casefold() for fact in character.actor.facts}
    for restored in restored_facts:
        if restored.casefold() in known_facts:
            continue
        character.actor.facts.append(
            ActorFact(origin=ActorFactOrigin.told, text=restored)
        )
        known_facts.add(restored.casefold())
    character.knowledge_tier = promoted_tier
    if rung.agent_tier is not None:
        character.agent_tier = rung.agent_tier


def _synthesis_skill_roll_succeeds(
    *,
    checkpoint: CheckpointFile,
    operation_id: str,
    source_id: str,
    chance_basis_points: int,
) -> bool:
    """Use one namespaced cryptographic stream per selected source Hero."""

    if chance_basis_points < 0 or chance_basis_points > 10_000:
        raise OneStarTransactionError("synthesis skill chance is outside basis-point bounds")
    payload = json.dumps(
        [
            "ayoa.one_star.synthesis.skill-transfer.v1",
            checkpoint.session.session_id,
            operation_id,
            source_id,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return draw < ((1 << 256) * chance_basis_points // 10_000)


def _synthesis_skill_consequences(
    *,
    skill: OneStarSkillEntry,
    target: CharacterRecord,
    checkpoint: CheckpointFile,
    state: OneStarAccountState,
    config: OneStarRulesConfig,
    owner_character_id: str,
) -> tuple[OneStarSystemConsequence, ...]:
    recipients = _system_window_recipients(
        checkpoint,
        state,
        config,
        owner_character_id,
        (target.character_id,),
    )
    if skill.visible:
        system_text = (
            f"System: {target.name} received transferred skill {skill.name} at rank 1."
        )
    else:
        system_text = f"System: {target.name} received a latent transferred capability."
    consequences = [OneStarSystemConsequence(
        text=system_text,
        recipient_character_ids=recipients,
    )]
    if target.character_id not in recipients:
        consequences.append(OneStarSystemConsequence(
            text=f"An unfamiliar pattern settles quietly within {target.name}.",
            recipient_character_ids=(target.character_id,),
        ))
    return tuple(consequences)


def _apply_pending_resolve(
    operation: OneStarPendingResolveOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
    *,
    event_id: str,
    owner_character_id: str,
    engine_history_updates: list[str],
) -> tuple[OneStarPendingOperation, tuple[OneStarSystemConsequence, ...]]:
    pending = state.pending_operation
    if pending is None or pending.operation_id != operation.operation_id:
        raise OneStarTransactionError("pending resolution does not match the open operation")
    _require_operational_facility(pending, state, config)
    system_consequences: list[OneStarSystemConsequence] = []
    if pending.kind == "deployment":
        for hero_id in pending.participant_ids:
            character, _hero = _require_local_active_hero(checkpoint, hero_id, config)
            if character.location != pending.destination:
                raise OneStarTransactionError(
                    "deployment cannot resolve before every Hero physically enters the gate"
                )
    elif pending.kind == "synthesis":
        target_character, target, sources = _synthesis_sources_and_target(
            pending,
            checkpoint,
            config,
        )
        if target_character.location != pending.destination:
            raise OneStarTransactionError(
                "synthesis target has not physically entered the chamber"
            )
        for source_character, _source in sources:
            if source_character.location != pending.destination:
                raise OneStarTransactionError(
                    "synthesis source has not physically entered the chamber"
                )
        expected_preview = _synthesis_preview(pending, state, checkpoint, config)
        if pending.synthesis_preview != expected_preview:
            raise OneStarTransactionError(
                "synthesis sources, target capacity, or returned equipment changed since selection"
            )
        preview = pending.synthesis_preview
        if preview is None:  # Defensive: durable schema requires it for synthesis.
            raise OneStarTransactionError("synthesis pending operation has no preview")
        known_item_ids = {item.item_id for item in state.stored_equipment}
        for item in preview.returned_equipment:
            if item.item_id in known_item_ids:
                raise OneStarTransactionError(
                    "synthesis returned equipment conflicts with durable storage"
                )
            known_item_ids.add(item.item_id)
        # Move each exact record before culling its source.  The after-checkpoint
        # is prepared as one unit, so any later validation failure rolls this back.
        state.stored_equipment.extend(
            item.model_copy(deep=True) for item in preview.returned_equipment
        )
        for source_character, source in sources:
            source.equipment = []
            _store_hero(source_character, source)
        try:
            report = apply_experience(
                hero=target,
                experience_delta=preview.offered_xp,
                config=config,
            )
        except ValueError as exc:
            raise OneStarTransactionError("synthesis XP could not be applied") from exc
        if (
            report.applied_xp != preview.applied_xp
            or report.wasted_xp != preview.wasted_xp
        ):
            raise OneStarTransactionError("synthesis XP preview no longer matches resolution")
        for source_character, source in sources:
            if not _synthesis_skill_roll_succeeds(
                checkpoint=checkpoint,
                operation_id=pending.operation_id,
                source_id=source_character.character_id,
                chance_basis_points=preview.skill_transfer_chance_basis_points,
            ):
                continue
            known_skill_ids = {skill.skill_id for skill in target.skills}
            eligible = sorted(
                (
                    skill
                    for skill in source.skills
                    if skill.rank > 0 and skill.skill_id not in known_skill_ids
                ),
                key=lambda skill: skill.skill_id,
            )
            if not eligible:
                continue
            transferred = eligible[0].model_copy(
                update={"rank": 1},
                deep=True,
            )
            target.skills.append(transferred)
            engine_history_updates.append(
                "synthesis_skill_transfer "
                + json.dumps(
                    {
                        "target_character_id": target_character.character_id,
                        "source_character_id": source_character.character_id,
                        "skill": transferred.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            system_consequences.extend(_synthesis_skill_consequences(
                skill=transferred,
                target=target_character,
                checkpoint=checkpoint,
                state=state,
                config=config,
                owner_character_id=owner_character_id,
            ))
        progression_consequences = _progression_consequence(
            character=target_character,
            checkpoint=checkpoint,
            state=state,
            config=config,
            owner_character_id=owner_character_id,
            report=report,
            label="synthesis result",
        )
        system_consequences.extend(progression_consequences)
        for item in preview.returned_equipment:
            system_consequences.append(OneStarSystemConsequence(
                text=(
                    f"System: {item.name} [{item.item_id}] returned to durable "
                    "equipment storage."
                ),
                recipient_character_ids=_system_window_recipients(
                    checkpoint,
                    state,
                    config,
                    owner_character_id,
                    tuple(character.character_id for character, _source in sources),
                ),
            ))
        for source_character, source in sources:
            source.terminal_cause = "synthesized"
            source.terminal_event_id = event_id
            source_character.status = CharacterStatus.culled
            _store_hero(source_character, source)
        _validate_hero_progression_state(target, config)
        _store_hero(target_character, target)
        state.synthesis_resolution_count += 1
    else:  # promotion
        target_character, target = _require_local_active_hero(checkpoint, pending.target_id, config)
        if target_character.location != pending.destination:
            raise OneStarTransactionError(
                "promotion target has not physically entered the chamber"
            )
        previous_stars = target.current_stars
        next_stars = previous_stars + 1
        if next_stars not in config.star_level_caps:
            raise OneStarTransactionError("promotion target has no configured star cap")
        if target.level != config.star_level_caps[target.current_stars]:
            raise OneStarTransactionError(
                "promotion requires the Hero to reach the current-star level cap"
            )
        _spend_resources(state.resources, config.promotion_cost)
        target.current_stars = next_stars
        if config.visual_novel_presentation is not None:
            from app.engine.one_star_visuals import (
                one_star_identity_reveal_stars,
            )

            reveal_stars = one_star_identity_reveal_stars(
                checkpoint,
                target_character,
            )
            if (
                reveal_stars is not None
                and previous_stars < reveal_stars <= next_stars
            ):
                introduced = checkpoint.session.visual_introductions.get(
                    owner_character_id,
                    [],
                )
                checkpoint.session.visual_introductions[
                    owner_character_id
                ] = [
                    character_id
                    for character_id in introduced
                    if character_id != target_character.character_id
                ]
        try:
            report = apply_promotion_banked_experience(hero=target, config=config)
        except ValueError as exc:
            raise OneStarTransactionError(
                "promotion could not apply the Hero's banked experience"
            ) from exc
        _restore_promotion_knowledge(
            checkpoint,
            target_character,
            next_stars,
        )
        _validate_hero_progression_state(target, config)
        _store_hero(target_character, target)
        system_consequences.extend(_progression_consequence(
            character=target_character,
            checkpoint=checkpoint,
            state=state,
            config=config,
            owner_character_id=owner_character_id,
            report=report,
            label="promotion result",
        ))
    state.pending_operation = None
    return pending, tuple(system_consequences)


def _apply_pending_cancel(operation: OneStarPendingCancelOperation, state: OneStarAccountState) -> None:
    if state.pending_operation is None or state.pending_operation.operation_id != operation.operation_id:
        raise OneStarTransactionError("pending cancellation does not match the open operation")
    state.pending_operation = None


def _apply_tutorial(
    operation: OneStarTutorialDeliveryOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
) -> None:
    key = operation.tutorial_key.strip()
    recipients = list(dict.fromkeys(cid.strip() for cid in operation.delivered_to_ids if cid.strip()))
    if not key or not recipients:
        raise OneStarTransactionError("tutorial delivery requires a key and recipients")
    for character_id in recipients:
        character = _require_character(checkpoint, character_id)
        if character.status != CharacterStatus.active:
            raise OneStarTransactionError(
                "tutorial delivery recipients must be active characters"
            )
    delivered = state.tutorial_deliveries.setdefault(key, [])
    duplicates = set(delivered) & set(recipients)
    if duplicates:
        raise OneStarTransactionError(
            "tutorial delivery must occur exactly once per recipient"
        )
    state.tutorial_deliveries[key] = list(dict.fromkeys([*delivered, *recipients]))


def one_star_event_fingerprint(payload: Mapping[str, object]) -> str:
    """Return a stable fingerprint for one complete router event payload."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _transaction_lifecycle_fingerprint(
    *,
    transaction: OneStarTransaction,
    spawned_character_ids: Iterable[str],
    activated_character_ids: Iterable[str],
    activated_character_locations: Mapping[str, str],
    dormant_character_ids: Iterable[str],
    generic_culled_character_ids: Iterable[str],
    location_updates: Mapping[str, str],
    initiating_actor_id: str,
) -> str:
    return one_star_event_fingerprint(
        {
            "transaction": transaction.model_dump(mode="json"),
            "spawn": sorted(spawned_character_ids),
            "activate": sorted(activated_character_ids),
            "activate_locations": dict(sorted(activated_character_locations.items())),
            "dormant": sorted(dormant_character_ids),
            "generic_cull": sorted(generic_culled_character_ids),
            "locations": dict(sorted(location_updates.items())),
            "initiating_actor_id": initiating_actor_id,
        }
    )


def one_star_event_already_applied(
    checkpoint: CheckpointFile,
    *,
    event_id: str,
    event_fingerprint: str,
) -> bool:
    """Check exact replay idempotency, rejecting event-id payload drift."""

    _owner, account = load_one_star_account(checkpoint)
    existing = account.state.applied_event_fingerprints.get(event_id)
    if existing is None:
        return False
    if existing != event_fingerprint:
        raise OneStarTransactionError(
            "One-Star event id was reused with a different canonical payload"
        )
    return True


def prepare_one_star_transaction(
    checkpoint: CheckpointFile,
    *,
    event_id: str,
    transaction: OneStarTransaction,
    spawned_character_ids: Iterable[str] = (),
    activated_character_ids: Iterable[str] = (),
    activated_character_locations: Mapping[str, str] | None = None,
    dormant_character_ids: Iterable[str] = (),
    generic_culled_character_ids: Iterable[str] = (),
    location_updates: Mapping[str, str] | None = None,
    canonical_at_s: int | None = None,
    event_fingerprint: str = "",
    initiating_actor_id: str = "",
    authoritative_system_result: bool = False,
) -> OneStarPreparedMutation:
    """Validate a One-Star transaction against a deep copy and prepare apply.

    ``spawned_character_ids`` and ``activated_character_ids`` are the exact
    generic event lifecycle IDs. A summon must cover exactly the fresh or
    activated records carrying an unowned Hero sheet; ordinary non-Hero spawns
    remain on the generic narrative path.
    """

    if not event_id.strip():
        raise OneStarTransactionError("One-Star transaction requires a stable event id")
    if not is_one_star_checkpoint(checkpoint):
        if transaction.present:
            raise OneStarTransactionError("One-Star transaction emitted outside active ruleset")
        return OneStarPreparedMutation(
            event_id=event_id,
            event_fingerprint="",
            after_checkpoint=_durable_checkpoint_copy(checkpoint),
        )

    spawned_ids = {
        character_id.strip()
        for character_id in spawned_character_ids
        if character_id.strip()
    }
    activated_ids = {
        character_id.strip()
        for character_id in activated_character_ids
        if character_id.strip()
    }
    normalized_activation_locations = {
        character_id.strip(): location.strip()
        for character_id, location in (activated_character_locations or {}).items()
        if character_id.strip()
    }
    if set(normalized_activation_locations) != activated_ids:
        raise OneStarTransactionError(
            "activation locations must exactly match activation ids"
        )
    dormant_ids = {
        character_id.strip()
        for character_id in dormant_character_ids
        if character_id.strip()
    }
    generic_cull_ids = {
        character_id.strip()
        for character_id in generic_culled_character_ids
        if character_id.strip()
    }
    normalized_locations = {
        character_id.strip(): location.strip()
        for character_id, location in (location_updates or {}).items()
    }
    pending_open_operations = [
        operation
        for operation in transaction.operations
        if isinstance(operation, OneStarPendingOpenOperation)
    ]
    authoritative_synthesis_pair = False
    if authoritative_system_result:
        authoritative_synthesis_pair = (
            len(transaction.operations) == 2
            and isinstance(
                transaction.operations[0],
                OneStarPendingOpenOperation,
            )
            and transaction.operations[0].pending.kind == "synthesis"
            and isinstance(
                transaction.operations[1],
                OneStarPendingResolveOperation,
            )
            and transaction.operations[1].operation_id
            == transaction.operations[0].pending.operation_id
        )
        if not authoritative_synthesis_pair:
            raise OneStarTransactionError(
                "authoritative System results currently require one exact "
                "synthesis pending_open followed by its pending_resolve"
            )
    if pending_open_operations:
        if len(transaction.operations) != 1 and not authoritative_synthesis_pair:
            raise OneStarTransactionError(
                "pending_open must be the only One-Star operation in its event"
            )
        pending = pending_open_operations[0].pending
        affected_ids = {
            *pending.participant_ids,
            *([pending.target_id] if pending.target_id else []),
        }
        changed_while_opening = affected_ids & (
            spawned_ids
            | activated_ids
            | dormant_ids
            | generic_cull_ids
            | (
                set()
                if authoritative_synthesis_pair
                else set(normalized_locations)
            )
        )
        if changed_while_opening:
            raise OneStarTransactionError(
                "opening an embodied operation cannot move or change the "
                "lifecycle of affected Heroes: "
                + ", ".join(sorted(changed_while_opening))
            )
        if authoritative_synthesis_pair:
            wrong_locations = {
                character_id
                for character_id in affected_ids
                if normalized_locations.get(character_id) != pending.destination
            }
            if wrong_locations:
                raise OneStarTransactionError(
                    "authoritative synthesis must move every source and target "
                    "to the configured chamber: "
                    + ", ".join(sorted(wrong_locations))
                )
    fingerprint = event_fingerprint.strip() or _transaction_lifecycle_fingerprint(
        transaction=transaction,
        spawned_character_ids=spawned_ids,
        activated_character_ids=activated_ids,
        activated_character_locations=normalized_activation_locations,
        dormant_character_ids=dormant_ids,
        generic_culled_character_ids=generic_cull_ids,
        location_updates=normalized_locations,
        initiating_actor_id=initiating_actor_id.strip(),
    )
    if one_star_event_already_applied(
        checkpoint,
        event_id=event_id,
        event_fingerprint=fingerprint,
    ):
        return OneStarPreparedMutation(
            event_id=event_id,
            event_fingerprint=fingerprint,
            after_checkpoint=_durable_checkpoint_copy(checkpoint),
            already_applied=True,
        )

    after = _durable_checkpoint_copy(checkpoint)
    owner, account = load_one_star_account(after)
    state = account.state
    config = account.config
    direct_opening = _direct_opening_operations(transaction, config)
    _validate_global_equipment_ids(after, state)
    _validate_all_hero_progression_states(after, config)
    if any(
        isinstance(operation, OneStarSummonOperation)
        and len(operation.hero_ids) > config.max_summon_batch
        for operation in transaction.operations
    ):
        raise OneStarTransactionError(
            "summon exceeds configured maximum batch size"
        )
    preexisting_mission = state.active_mission
    preexisting_pending = state.pending_operation
    if preexisting_pending is not None and preexisting_pending.kind == "deployment":
        deployment_participants = set(preexisting_pending.participant_ids)
        deployment_destination = preexisting_pending.destination
        resolves_deployment = any(
            isinstance(operation, OneStarPendingResolveOperation)
            and operation.operation_id == preexisting_pending.operation_id
            for operation in transaction.operations
        )
        starts_deployment = any(
            isinstance(operation, OneStarMissionStartOperation)
            and operation.pending_operation_id == preexisting_pending.operation_id
            for operation in transaction.operations
        )
        for participant_id in preexisting_pending.participant_ids:
            participant = _require_character(after, participant_id)
            destination = preexisting_pending.destination
            next_location = normalized_locations.get(
                participant_id,
                participant.location,
            )
            crossed_before = participant.location == destination
            crosses_now = (
                participant.location != destination
                and next_location == destination
            )
            if crossed_before and next_location != destination:
                raise OneStarTransactionError(
                    "a Hero who crossed the deployment gate cannot return while "
                    "the deployment is still pending"
                )
            if (crossed_before or crosses_now) and not (
                resolves_deployment and starts_deployment
            ):
                raise OneStarTransactionError(
                    "deployment gate crossing, pending resolution, and mission "
                    "start must commit in the same canonical event"
                )
        for character in after.characters:
            if character.character_id in deployment_participants:
                continue
            hero = load_one_star_hero(character)
            if hero is None or hero.owner_lobby_id != config.lobby_id:
                continue
            if (
                character.status == CharacterStatus.active
                and character.location == deployment_destination
            ):
                raise OneStarTransactionError(
                    "a local Hero beyond the pending deployment gate must be "
                    "a selected participant"
                )
            planned_location = normalized_locations.get(
                character.character_id,
                normalized_activation_locations.get(
                    character.character_id,
                    character.location,
                ),
            )
            will_be_active = (
                character.status == CharacterStatus.active
                or character.character_id in activated_ids
            )
            if will_be_active and planned_location == deployment_destination:
                raise OneStarTransactionError(
                    "an unselected local Hero cannot cross a pending deployment gate"
                )
    pre_event_escape_authorities: dict[str, set[str]] = {}
    if preexisting_mission is not None:
        for character in after.characters:
            if character.character_id not in preexisting_mission.party_ids:
                continue
            hero = load_one_star_hero(character)
            if hero is None:
                continue
            pre_event_escape_authorities[character.character_id] = {
                *(
                    item.item_id
                    for item in hero.equipment
                    if "tower_escape" in item.tags
                ),
                *(
                    skill.skill_id
                    for skill in hero.skills
                    if "tower_escape" in skill.tags
                ),
            }
    matching_return_outcomes = {
        operation.outcome
        for operation in transaction.operations
        if isinstance(operation, OneStarMissionEndOperation)
        and preexisting_mission is not None
        and operation.mission_id == preexisting_mission.mission_id
    }
    before_character_state = {
        character.character_id: (
            character.status,
            character.location,
            character.knowledge_tier,
            character.mechanics.get(ONE_STAR_HERO_KEY) if isinstance(character.mechanics, dict) else None,
        )
        for character in after.characters
    }
    for normalized_id, normalized_location in normalized_locations.items():
        if not normalized_id or not normalized_location:
            raise OneStarTransactionError(
                "location updates require non-empty character ids and locations"
            )
        moving_character = _require_character(after, normalized_id)
        if (
            preexisting_mission is not None
            and normalized_id in preexisting_mission.party_ids
            and normalized_location != preexisting_mission.destination
        ):
            if (
                normalized_location != config.lobby_location_label
                or not matching_return_outcomes.intersection({"completed", "escaped"})
            ):
                raise OneStarTransactionError(
                    "an active mission Hero cannot cross the sealed Tower boundary "
                    "without a completed or authorized escaped mission end"
                )
        if (
            preexisting_mission is not None
            and normalized_id not in preexisting_mission.party_ids
            and normalized_location == preexisting_mission.destination
            and moving_character.location != normalized_location
        ):
            moving_hero = load_one_star_hero(moving_character)
            if (
                moving_hero is not None
                and moving_hero.owner_lobby_id == config.lobby_id
            ):
                raise OneStarTransactionError(
                    "a local Hero cannot join an active mission without a "
                    "deployment transition"
                )
        moving_character.location = normalized_location

    now_s = after.session.leading_at_s if canonical_at_s is None else canonical_at_s
    if now_s < 0:
        raise OneStarTransactionError("canonical event time cannot be negative")
    if spawned_ids & activated_ids:
        raise OneStarTransactionError(
            "the same character cannot be both spawned and activated"
        )
    expected_summon_ids: set[str] = set()
    for character_id in spawned_ids:
        character = _require_character(after, character_id)
        hero = load_one_star_hero(character)
        if hero is None:
            continue
        if hero.owner_lobby_id:
            raise OneStarTransactionError(
                "freshly spawned Hero records must begin unowned"
            )
        expected_summon_ids.add(character_id)
    for character_id in activated_ids:
        character = _require_character(after, character_id)
        if character.status == CharacterStatus.culled:
            raise OneStarTransactionError(
                "a culled character cannot be activated"
            )
        hero = load_one_star_hero(character)
        if hero is not None and not hero.owner_lobby_id:
            expected_summon_ids.add(character_id)
    for character_id in generic_cull_ids:
        character = _require_character(after, character_id)
        hero = load_one_star_hero(character)
        if hero is not None and hero.owner_lobby_id == config.lobby_id:
            raise OneStarTransactionError(
                "local One-Star Hero culls belong only in One-Star state updates"
            )
    pinned_ids = set(after.session.active_act_slots)
    for open_event in after.session.open_cat_ii_events:
        pinned_ids.add(open_event.initiator_id)
        pinned_ids.update(open_event.required_responders)
    for character_id in dormant_ids:
        character = _require_character(after, character_id)
        if character.status == CharacterStatus.culled:
            raise OneStarTransactionError(
                "a culled character cannot become dormant"
            )
        if character_id in pinned_ids:
            raise OneStarTransactionError(
                "a pinned character cannot become dormant before its open "
                "event resolves"
            )
    if preexisting_mission is not None:
        mission_lifecycle_overlap = set(preexisting_mission.party_ids) & (
            dormant_ids | activated_ids
        )
        if mission_lifecycle_overlap:
            raise OneStarTransactionError(
                "active mission party lifecycle cannot change through generic "
                "dormant or activate signals: "
                + ", ".join(sorted(mission_lifecycle_overlap))
            )
        for character_id in activated_ids - set(preexisting_mission.party_ids):
            character = _require_character(after, character_id)
            hero = load_one_star_hero(character)
            arrival_location = (
                normalized_activation_locations[character_id]
                or character.location
            )
            if (
                hero is not None
                and hero.owner_lobby_id == config.lobby_id
                and arrival_location == preexisting_mission.destination
            ):
                raise OneStarTransactionError(
                    "a local Hero cannot activate into an active mission "
                    "without a deployment transition"
                )
    # Cat II resolution retains the opening's fictional timestamp even when
    # another slot has since advanced the session. Stamina is a serialized
    # account ledger, so recover it at a monotonic ledger clock while leaving
    # mission deadlines and authored event timestamps on ``now_s``.
    stamina_now_s = max(
        now_s,
        after.session.leading_at_s,
        state.stamina_recovery_anchor_s,
    )
    stamina_before_recovery = state.stamina_current
    _recover_stamina(state, config, stamina_now_s)
    engine_history_updates = [
        (
            f"stamina_recovered current={state.stamina_current} "
            f"recovery_anchor_s={state.stamina_recovery_anchor_s}"
        )
    ] if state.stamina_current > stamina_before_recovery else []
    funds_now_s = max(
        now_s,
        after.session.leading_at_s,
        state.funds_accrual_anchor_s,
    )
    funds_before_accrual = state.discretionary_funds
    _recover_discretionary_funds(state, config, funds_now_s)
    if state.discretionary_funds > funds_before_accrual:
        engine_history_updates.append(
            "discretionary_funds_accrued "
            f"current={state.discretionary_funds} "
            f"accrual_anchor_s={state.funds_accrual_anchor_s}"
        )
    system_consequences: list[OneStarSystemConsequence] = []
    if not transaction.present:
        if expected_summon_ids:
            raise OneStarTransactionError(
                "fresh spawns or unowned Hero activations require a matching summon"
            )
        state.applied_event_fingerprints[event_id] = fingerprint
        _store_account(owner, account)
        return OneStarPreparedMutation(
            event_id=event_id,
            event_fingerprint=fingerprint,
            after_checkpoint=after,
            engine_history_updates=tuple(engine_history_updates),
        )

    hero_initializations: dict[str, OneStarHeroState] = {}
    preexisting_pending_operation_id = (
        state.pending_operation.operation_id
        if state.pending_operation is not None
        else ""
    )
    summon_ids: set[str] = set()
    resolved_deployments: dict[str, object] = {}
    started_deployment_ids: set[str] = set()
    actor_id = initiating_actor_id.strip()

    def require_account_owner(operation_name: str) -> None:
        if actor_id != owner.character_id:
            raise OneStarTransactionError(
                f"only the account owner may initiate {operation_name}"
            )

    for operation in transaction.operations:
        if isinstance(operation, OneStarCatalogueApplyOperation):
            require_account_owner("a catalogue operation")
            _apply_catalogue(operation, state, config)
        elif isinstance(operation, OneStarSummonOperation):
            if summon_ids:
                raise OneStarTransactionError("a transaction may contain only one summon operation")
            pool = config.summon_pools.get(operation.pool_id)
            if pool is None:
                raise OneStarTransactionError(
                    "summon references an unknown configured pool"
                )
            if pool.usage == "standard":
                require_account_owner("an account summon")
            summon_ids.update(operation.hero_ids)
            _apply_summon(
                operation,
                state,
                config,
                after,
                event_id,
                hero_initializations,
                expected_summon_ids,
                spawned_ids,
                activated_ids,
                normalized_activation_locations,
                initiating_actor_id.strip(),
                (
                    direct_opening[1].mission.destination
                    if direct_opening is not None
                    and operation is direct_opening[0]
                    else config.lobby_location_label
                ),
            )
        elif isinstance(operation, OneStarInventoryDeltaOperation):
            _apply_inventory_delta(operation, state)
        elif isinstance(operation, OneStarGemPurchaseOperation):
            require_account_owner("a Gem purchase")
            _apply_gem_purchase(operation, state, config)
        elif isinstance(operation, OneStarHeroDeltaOperation):
            death_name = _apply_hero_delta(operation, after, config, event_id)
            if (
                death_name
                and state.active_mission is not None
                and operation.hero_id in state.active_mission.party_ids
            ):
                system_consequences.append(OneStarSystemConsequence(
                    text=f"System: {death_name} died.",
                    recipient_character_ids=(
                        one_star_terminal_system_recipient_ids(after)
                    ),
                ))
        elif isinstance(operation, OneStarMissionStartOperation):
            _apply_mission_start(
                operation,
                state,
                after,
                config,
                now_s,
                resolved_deployments,
                (
                    tuple(direct_opening[0].hero_ids)
                    if direct_opening is not None
                    and operation is direct_opening[1]
                    else None
                ),
                (
                    normalized_activation_locations
                    if direct_opening is not None
                    and operation is direct_opening[1]
                    else None
                ),
            )
            if operation.pending_operation_id:
                started_deployment_ids.add(operation.pending_operation_id)
        elif isinstance(operation, OneStarMissionUpdateOperation):
            _apply_mission_update(operation, state, now_s)
        elif isinstance(operation, OneStarMissionEndOperation):
            system_consequences.extend(_apply_mission_end(
                operation,
                state,
                after,
                config,
                now_s,
                pre_event_escape_authorities,
                owner.character_id,
            ))
        elif isinstance(operation, OneStarPendingOpenOperation):
            require_account_owner("an embodied operation selection")
            pending = _apply_pending_open(operation, state, after, config, now_s)
            if pending.kind == "synthesis" and not authoritative_synthesis_pair:
                system_consequences.append(_synthesis_preview_consequence(
                    pending=pending,
                    checkpoint=after,
                    state=state,
                    config=config,
                    owner_character_id=owner.character_id,
                ))
        elif isinstance(operation, OneStarPendingResolveOperation):
            if (
                operation.operation_id != preexisting_pending_operation_id
                and not (
                    authoritative_synthesis_pair
                    and state.pending_operation is not None
                    and state.pending_operation.operation_id
                    == operation.operation_id
                )
            ):
                raise OneStarTransactionError(
                    "an embodied operation cannot resolve in the event that opened it"
                )
            resolved, resolved_consequences = _apply_pending_resolve(
                operation,
                state,
                after,
                config,
                event_id=event_id,
                owner_character_id=owner.character_id,
                engine_history_updates=engine_history_updates,
            )
            system_consequences.extend(resolved_consequences)
            if getattr(resolved, "kind", "") == "deployment":
                resolved_deployments[operation.operation_id] = resolved
        elif isinstance(operation, OneStarPendingCancelOperation):
            if operation.operation_id != preexisting_pending_operation_id:
                raise OneStarTransactionError(
                    "an embodied operation cannot cancel in the event that opened it"
                )
            _apply_pending_cancel(operation, state)
        elif isinstance(operation, OneStarEquipmentMoveOperation):
            require_account_owner("an equipment move")
            _apply_equipment_move(operation, state, after, config)
        elif isinstance(operation, OneStarTutorialDeliveryOperation):
            if actor_id not in state.guide_character_ids:
                raise OneStarTransactionError(
                    "tutorial delivery must originate from a configured guide"
                )
            _apply_tutorial(operation, state, after)
        else:  # pragma: no cover - discriminated union prevents this at parse time.
            raise OneStarTransactionError(f"unsupported One-Star operation {operation!r}")
    if summon_ids != expected_summon_ids:
        raise OneStarTransactionError(
            "fresh spawns or unowned Hero activations require exactly one matching summon"
        )
    if set(resolved_deployments) != started_deployment_ids:
        raise OneStarTransactionError(
            "every resolved deployment must start exactly one mission in the same event"
        )
    _validate_global_equipment_ids(after, state)
    _validate_all_hero_progression_states(after, config)

    state.applied_event_fingerprints[event_id] = fingerprint
    _store_account(owner, account)
    culled_character_ids = tuple(
        character.character_id
        for character in after.characters
        if before_character_state.get(character.character_id, (CharacterStatus.culled, "", 0, None))[0] != CharacterStatus.culled
        and character.status == CharacterStatus.culled
    )
    dormant_terminal_overlap = set(culled_character_ids) & dormant_ids
    if dormant_terminal_overlap:
        raise OneStarTransactionError(
            "terminal One-Star Hero culls cannot also use generic dormancy: "
            + ", ".join(sorted(dormant_terminal_overlap))
        )
    touched_hero_ids = tuple(
        character.character_id
        for character in after.characters
        if load_one_star_hero(character) is not None
        and before_character_state.get(character.character_id)
        != (
            character.status,
            character.location,
            character.knowledge_tier,
            character.mechanics.get(ONE_STAR_HERO_KEY),
        )
    )
    return OneStarPreparedMutation(
        event_id=event_id,
        event_fingerprint=fingerprint,
        after_checkpoint=after,
        hero_initializations=hero_initializations,
        culled_character_ids=culled_character_ids,
        newly_acquired_hero_ids=tuple(hero_initializations),
        touched_hero_ids=touched_hero_ids,
        engine_history_updates=tuple(engine_history_updates),
        system_consequences=tuple(system_consequences),
    )


def apply_one_star_prepared_mutation(
    checkpoint: CheckpointFile, prepared: OneStarPreparedMutation,
) -> bool:
    """Replace the checkpoint with a previously prepared, fully valid state.

    Preparation has already parsed and validated the complete after-state, so
    this does no piecemeal resource/lifecycle mutation.  Reapplying an already
    committed event is a deliberate no-op.
    """

    _owner, live_account = load_one_star_account(checkpoint)
    existing_fingerprint = live_account.state.applied_event_fingerprints.get(
        prepared.event_id
    )
    if existing_fingerprint is not None:
        if existing_fingerprint != prepared.event_fingerprint:
            raise OneStarTransactionError(
                "prepared One-Star event id conflicts with committed payload"
            )
        return False
    if prepared.already_applied:
        raise OneStarTransactionError(
            "prepared One-Star replay was not present in the live checkpoint"
        )
    # The adapter owns character-attached ledger/mechanics and embodied
    # character fields only. Replacing the whole checkpoint would detach live
    # generic runtime objects such as OpenCatIIEvent references held by the
    # turn loop. Swap the fully validated character list in one assignment and
    # leave session, router history, buffers, world state, and runtime handles
    # on their existing object graph.
    checkpoint.characters = [
        character.model_copy(deep=True)
        for character in prepared.after_checkpoint.characters
    ]
    return True
