"""D&D lifecycle effects around the rules-neutral canonical event path."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.engine import (
    dnd_combat,
    dnd_monsters,
    dnd_spatial,
    imported_encounters,
    imported_statblocks,
)
from app.engine.dnd_combat_access import combatant_for_character
from app.engine.event_runtime import set_action_obligation
from app.engine.router_batch import MaterializedRouterBatch
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import DndCanonicalEventRecord, RouterInputEnvelope
from app.schemas.events import ObservableFact


def prepare_dnd_batch(
    checkpoint: CheckpointFile,
    batch: MaterializedRouterBatch,
    inputs: Sequence[RouterInputEnvelope],
) -> None:
    """Apply fallible D&D start/end effects to an isolated checkpoint."""

    inputs_by_id = {item.submission_id: item for item in inputs}
    for materialized in batch.events:
        event = materialized.record
        if not isinstance(event, DndCanonicalEventRecord):
            raise RuntimeError("D&D router returned a non-D&D canonical record")
        if event.interaction_mode == "dnd_combat_start":
            _prepare_combat_start(checkpoint, event, inputs_by_id)
        elif event.interaction_mode == "dnd_combat_end":
            _prepare_combat_end(checkpoint, event)


def install_dnd_reactions(
    checkpoint: CheckpointFile,
    batch: MaterializedRouterBatch,
) -> None:
    """Turn transient router reaction picks into typed player obligations."""

    for materialized in batch.events:
        if not materialized.dnd_reaction_ids:
            continue
        install_dnd_reaction_candidates(
            checkpoint,
            event_id=materialized.record.event_id,
            candidate_ids=materialized.dnd_reaction_ids,
        )


def install_dnd_reaction_candidates(
    checkpoint: CheckpointFile,
    *,
    event_id: str,
    candidate_ids: Sequence[str],
    excluded_ids: Sequence[str] = (),
) -> list[str]:
    """Install optional player reactions and return the characters gated."""

    if not candidate_ids:
        return []
    combat = checkpoint.session.active_combat
    if combat is None:
        raise RuntimeError("D&D reaction candidates require active combat")
    bound = set(checkpoint.session.character_bindings)
    excluded = set(excluded_ids)
    installed: list[str] = []
    for character_id in dict.fromkeys(candidate_ids):
        if character_id in excluded:
            continue
        combatant = combatant_for_character(combat, character_id)
        if combatant is None or not _combatant_can_react(combatant):
            raise RuntimeError(
                f"D&D reaction candidate is ineligible: {character_id}"
            )
        if character_id not in bound:
            continue
        set_action_obligation(
            checkpoint,
            character_id=character_id,
            kind="combat_reaction",
            source_event_id=event_id,
        )
        installed.append(character_id)
    return installed


def current_combat_actor_id(checkpoint: CheckpointFile) -> str:
    combat = checkpoint.session.active_combat
    if combat is None:
        return ""
    try:
        current = dnd_combat.current_combatant(combat)
    except ValueError:
        return ""
    return current.character_id or current.combatant_id


def combat_contains_character(
    checkpoint: CheckpointFile,
    character_id: str,
) -> bool:
    combat = checkpoint.session.active_combat
    return combat is not None and combatant_for_character(combat, character_id) is not None


def validate_combat_action_actor(
    checkpoint: CheckpointFile,
    *,
    character_id: str,
    is_reaction: bool,
) -> None:
    combat = checkpoint.session.active_combat
    if combat is None or combatant_for_character(combat, character_id) is None:
        raise ValueError("character is not participating in active combat")
    if is_reaction:
        obligation = checkpoint.session.action_obligations.get(character_id)
        if obligation is None or obligation.kind != "combat_reaction":
            raise ValueError("character has no pending combat reaction")
        return
    if any(
        obligation.kind in {"combat_reaction", "cat_ii_roll"}
        for obligation in checkpoint.session.action_obligations.values()
    ):
        raise ValueError("combat is waiting for a reaction or player roll")
    current_id = current_combat_actor_id(checkpoint)
    if current_id != character_id:
        raise ValueError(f"it is {current_id or 'another combatant'}'s initiative turn")


def finalize_combat_action(
    checkpoint: CheckpointFile,
    *,
    event_id: str,
    actor_id: str,
    reaction_candidate_ids: Sequence[str],
    is_reaction: bool,
) -> list[str]:
    """Apply reaction gates and initiative motion after canonical resolution."""

    combat = checkpoint.session.active_combat
    if combat is None:
        return []
    actor = combatant_for_character(combat, actor_id)
    if actor is not None:
        actor.pending_initiating_action = ""
        actor.pending_initiating_event_id = ""
    if is_reaction:
        checkpoint.session.action_obligations.pop(actor_id, None)
        installed = install_dnd_reaction_candidates(
            checkpoint,
            event_id=event_id,
            candidate_ids=reaction_candidate_ids,
            excluded_ids=[actor_id],
        )
        advance_pending_combat_if_unblocked(checkpoint)
        return installed

    installed = install_dnd_reaction_candidates(
        checkpoint,
        event_id=event_id,
        candidate_ids=reaction_candidate_ids,
        excluded_ids=[actor_id],
    )
    if installed:
        combat.pending_advance_actor_id = actor_id
    else:
        _advance_combat(checkpoint)
    return installed


def advance_pending_combat_if_unblocked(checkpoint: CheckpointFile) -> bool:
    combat = checkpoint.session.active_combat
    if combat is None or not combat.pending_advance_actor_id:
        return False
    if any(
        obligation.kind in {"combat_reaction", "cat_ii_roll"}
        for obligation in checkpoint.session.action_obligations.values()
    ):
        return False
    current_id = current_combat_actor_id(checkpoint)
    if current_id != combat.pending_advance_actor_id:
        combat.pending_advance_actor_id = ""
        return False
    _advance_combat(checkpoint)
    return True


def _advance_combat(checkpoint: CheckpointFile) -> None:
    combat = checkpoint.session.active_combat
    if combat is None:
        return
    next_combatant = dnd_combat.advance_turn_with_effects(
        checkpoint.session,
        characters=checkpoint.characters,
    )
    combat.pending_advance_actor_id = ""
    dnd_combat.sync_combat_effects_to_characters(combat, checkpoint.characters)
    dnd_combat.append_audit_line(
        combat,
        f"Initiative advanced to {next_combatant.name or next_combatant.character_id}.",
    )


def _combatant_can_react(combatant: object) -> bool:
    if str(getattr(combatant, "defeat_state", "active")) != "active":
        return False
    if bool(getattr(combatant, "removed", False)):
        return False
    if not bool(getattr(combatant, "reaction_available", True)):
        return False
    conditions = {
        str(value).strip().lower()
        for value in getattr(combatant, "conditions", ())
    }
    return not conditions.intersection({
        "incapacitated",
        "paralyzed",
        "stunned",
        "unconscious",
        "dead",
    })


def _event_actor_and_intention(
    event: DndCanonicalEventRecord,
    inputs_by_id: dict[str, RouterInputEnvelope],
) -> tuple[str, str]:
    for submission_id in event.source_submission_ids:
        envelope = inputs_by_id.get(submission_id)
        if envelope is not None and envelope.actor_ids:
            return envelope.actor_ids[0], envelope.payload
    if event.actor_ids:
        return event.actor_ids[0], ""
    raise RuntimeError("D&D combat start requires an acting character")


def _character_location(checkpoint: CheckpointFile, character_id: str) -> str:
    character = next(
        (
            item
            for item in checkpoint.characters
            if item.character_id == character_id
        ),
        None,
    )
    return character.location if character is not None else ""


def _clean_character_id(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(
        r"[^a-z0-9_]+",
        "_",
        value.strip().lower(),
    )).strip("_")


def _unique_character_id(existing: set[str], preferred: str) -> str:
    root = _clean_character_id(preferred) or "combatant"
    if root not in existing:
        return root
    index = 2
    while f"{root}_{index}" in existing:
        index += 1
    return f"{root}_{index}"


def _mark_combat_spawn(
    character: CharacterRecord,
    *,
    event_id: str,
    monster_key: str,
    statblock_ref: str,
) -> None:
    mechanics = dict(character.mechanics or {})
    mechanics["combat_spawn"] = {
        key: value
        for key, value in {
            "spawned": True,
            "source_event_id": event_id,
            "monster_key": monster_key,
            "statblock_ref": statblock_ref,
        }.items()
        if value or key == "spawned"
    }
    mechanics.setdefault("source", "router_combatant_spawn")
    character.mechanics = mechanics


def _materialize_combatant_spawns(
    checkpoint: CheckpointFile,
    event: DndCanonicalEventRecord,
    *,
    actor_id: str,
) -> None:
    existing = {item.character_id for item in checkpoint.characters}
    default_location = _character_location(checkpoint, actor_id)
    materialized = []
    combatant_ids = list(event.combatant_ids)
    for original in event.combatant_spawns:
        spawn = original.model_copy(deep=True)
        character_id = _unique_character_id(
            existing,
            spawn.character_id or spawn.monster_key or spawn.name,
        )
        if character_id != spawn.character_id:
            combatant_ids = [
                character_id if value == spawn.character_id else value
                for value in combatant_ids
            ]
            spawn = spawn.model_copy(update={"character_id": character_id})
        character = imported_statblocks.resolve_spawn_character_from_content_state(
            spawn,
            content_state=checkpoint.session.content_state,
            default_location=default_location,
        )
        if character is None:
            character = dnd_monsters.character_from_combatant_spawn(
                spawn,
                default_location=default_location,
            )
        _mark_combat_spawn(
            character,
            event_id=event.event_id,
            monster_key=spawn.monster_key,
            statblock_ref=spawn.statblock_ref,
        )
        checkpoint.characters.append(character)
        existing.add(character.character_id)
        combatant_ids.append(character.character_id)
        materialized.append(spawn)
    event.combatant_spawns = materialized
    event.combatant_ids = list(dict.fromkeys(
        value for value in combatant_ids if value
    ))


def _combat_participants(
    checkpoint: CheckpointFile,
    *,
    actor_id: str,
    combatant_ids: Sequence[str],
) -> list[CharacterRecord]:
    roster = {item.character_id: item for item in checkpoint.characters}
    explicit = set(combatant_ids)
    participants: list[CharacterRecord] = []
    for character_id in dict.fromkeys((actor_id, *combatant_ids)):
        character = roster.get(character_id)
        if character is None or character.status == CharacterStatus.culled:
            continue
        if character.status == CharacterStatus.dormant:
            if character_id not in explicit:
                continue
            character.status = CharacterStatus.active
        if character.status == CharacterStatus.active:
            participants.append(character)
    return participants


def _ensure_direct_observers(
    event: DndCanonicalEventRecord,
    character_ids: Sequence[str],
) -> None:
    promoted = {character_id for character_id in character_ids if character_id}
    event.observers.indirect = [
        character_id
        for character_id in event.observers.indirect
        if character_id not in promoted
    ]
    event.observers.inferred = [
        character_id
        for character_id in event.observers.inferred
        if character_id not in promoted
    ]
    event.observers.direct = list(dict.fromkeys([
        *event.observers.direct,
        *character_ids,
    ]))


def _append_public_fact(event: DndCanonicalEventRecord, text: str) -> None:
    if any(fact.text.strip() == text for fact in event.observable_facts):
        return
    event.observable_facts.append(ObservableFact.all(
        text,
        at_offset_s=event.duration_s,
    ))


def _prepare_combat_start(
    checkpoint: CheckpointFile,
    event: DndCanonicalEventRecord,
    inputs_by_id: dict[str, RouterInputEnvelope],
) -> None:
    if checkpoint.session.active_combat is not None:
        raise RuntimeError("router attempted to start combat while combat is active")
    actor_id, intention = _event_actor_and_intention(event, inputs_by_id)
    imported = imported_encounters.resolve_combat_start_from_content_state(
        checkpoint.session.content_state,
        location_ref=_character_location(checkpoint, actor_id),
    )
    imported_encounters.apply_resolved_encounter_to_router_output(event, imported)
    _materialize_combatant_spawns(checkpoint, event, actor_id=actor_id)
    participants = _combat_participants(
        checkpoint,
        actor_id=actor_id,
        combatant_ids=event.combatant_ids,
    )
    if len(participants) < 2:
        raise RuntimeError("D&D combat start resolved fewer than two combatants")
    combat = dnd_combat.start_combat(
        checkpoint.session,
        participants,
        combat_id=f"combat_{event.event_id}",
    )
    battle_map = dnd_spatial.normalize_battle_map_seed(
        event.battle_map_seed,
        combat.combatants,
    )
    if battle_map is not None:
        combat.battle_map = battle_map
    current = combatant_for_character(combat, actor_id)
    if current is not None:
        current.pending_initiating_action = intention.strip()
        current.pending_initiating_event_id = event.event_id
    participant_ids = [item.character_id for item in participants]
    _ensure_direct_observers(event, participant_ids)
    _append_public_fact(event, "D&D combat begins.")
    order = ", ".join(
        f"{item.name or item.character_id} {item.initiative_total}"
        for item in combat.combatants
    )
    dnd_combat.append_audit_line(
        combat,
        f"Combat started from canonical event {event.event_id}. "
        f"Initiative order: {order}.",
    )


def _prepare_combat_end(
    checkpoint: CheckpointFile,
    event: DndCanonicalEventRecord,
) -> None:
    combat = checkpoint.session.active_combat
    if combat is None:
        raise RuntimeError("router attempted to end combat when none is active")
    participant_ids = [
        item.character_id or item.combatant_id
        for item in combat.combatants
        if item.character_id or item.combatant_id
    ]
    _ensure_direct_observers(event, participant_ids)
    for fact in dnd_combat.drain_pending_visible_facts(combat):
        _append_public_fact(event, fact)
    dnd_combat.queue_router_observed_fact_updates(checkpoint.session, combat)
    dnd_combat.end_combat(
        checkpoint.session,
        characters=checkpoint.characters,
    )
    _append_public_fact(event, "D&D combat ends.")
