from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from app.engine import dice, mechanics
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.state import (
    DndCombatantState,
    DndCombatState,
    DndRuntimeEffect,
    SessionState,
)


DND_RUNTIME_KEY = "dnd5e_runtime"
DND_ACTIVE_EFFECTS_KEY = "active_effects"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_combat(
    session: SessionState,
    characters: Iterable[CharacterRecord],
    *,
    combat_id: str = "combat",
    include_inactive: bool = False,
) -> DndCombatState:
    """Start active D&D combat on the session and roll initiative."""
    if session.active_combat is not None:
        raise ValueError("Combat is already active.")

    combatants = build_combatants(
        characters,
        session=session,
        include_inactive=include_inactive,
    )
    if not combatants:
        raise ValueError("Cannot start combat without combatants.")

    combat = DndCombatState(
        combat_id=combat_id,
        status="active",
        round_number=1,
        turn_index=0,
        combatants=combatants,
        started_at_turn_index=session.turn_index,
    )
    roll_initiative(combat)
    _move_to_next_available(combat, starting_at=0, count_round_wrap=False)
    session.active_combat = combat
    return combat


def end_combat(
    session: SessionState,
    *,
    characters: Iterable[CharacterRecord] | None = None,
) -> DndCombatState:
    """Clear active combat and return the ended snapshot."""
    combat = _require_combat(session)
    if characters is not None:
        sync_combat_effects_to_characters(combat, characters)
    combat.status = "ended"
    combat.ended_at_turn_index = session.turn_index
    combat.pending_advance_actor_id = ""
    _cancel_combat_roll_transactions(session)
    _clear_combat_slots(session)
    session.active_combat = None
    return combat


def build_combatants(
    characters: Iterable[CharacterRecord],
    *,
    session: SessionState | None = None,
    include_inactive: bool = False,
) -> list[DndCombatantState]:
    return [
        build_combatant(character, session=session)
        for character in characters
        if include_inactive or character.status == CharacterStatus.active
    ]


def build_combatant(
    character: CharacterRecord,
    *,
    session: SessionState | None = None,
    combatant_id: str | None = None,
) -> DndCombatantState:
    mechanics_state = character.mechanics or {}
    hp = _hit_points(mechanics_state)
    hp_current = _safe_int(hp.get("current"), 0)
    hp_max = _safe_int(hp.get("max"), hp_current)
    hp_temp = _safe_int(hp.get("temporary"), 0)
    active_effects = runtime_effects_for_character(character)
    combatant = DndCombatantState(
        combatant_id=combatant_id or character.character_id,
        character_id=character.character_id,
        name=character.name,
        player_controlled=(
            session is not None
            and character.character_id in session.character_bindings
        ),
        armor_class=_armor_class(mechanics_state),
        hit_points_current=max(0, hp_current),
        hit_points_max=max(0, hp_max),
        hit_points_temporary=max(0, hp_temp),
        initiative_modifier=_initiative_modifier(mechanics_state),
        initiative_advantage_state=_initiative_advantage_state_from_mechanics(
            mechanics_state
        ),
        conditions=_merge_conditions(
            _conditions(mechanics_state),
            _effect_conditions(active_effects),
        ),
        active_effects=active_effects,
        death_save_successes=_death_save_count(mechanics_state, "successes"),
        death_save_failures=_death_save_count(mechanics_state, "failures"),
    )
    _sync_defeat_state(
        combatant,
        uses_death_saves=_uses_death_saves(session, combatant),
    )
    return combatant


def roll_initiative(combat: DndCombatState) -> DndCombatState:
    """Roll initiative for every non-removed combatant and stable-sort order."""
    for combatant in combat.combatants:
        if combatant.removed:
            continue
        result = dice.roll_d20_check(
            roll_id=f"initiative_{combatant.combatant_id}",
            modifier=combatant.initiative_modifier,
            actor_id=combatant.character_id or combatant.combatant_id,
            reason="initiative",
            advantage_state=_initiative_advantage_state(combatant),
        )
        combatant.initiative_roll = _kept_d20_value(result)
        combatant.initiative_total = result.total
        combatant.initiative_detail = result.detail

    current_id = (
        current_combatant(combat).combatant_id
        if combat.combatants else ""
    )
    sort_turn_order(combat)
    if current_id:
        _set_turn_to_combatant(combat, current_id)
    return combat


def append_audit_line(combat: DndCombatState | SessionState, line: str) -> None:
    """Append a durable combat audit line to the active combat snapshot."""
    text = line.strip()
    if not text:
        return
    _active_from(combat).audit_lines.append(text)


def _append_pending_visible_fact(combat: DndCombatState, fact: str) -> None:
    text = fact.strip()
    if text:
        combat.pending_visible_facts.append(text)


def drain_pending_visible_facts(
    combat: DndCombatState | SessionState | None,
) -> list[str]:
    if isinstance(combat, SessionState):
        active = combat.active_combat
    else:
        active = combat
    if active is None:
        return []
    if isinstance(active, dict):
        pending = active.get("pending_visible_facts") or []
        facts = [
            str(fact).strip()
            for fact in pending
            if str(fact).strip()
        ]
        active["pending_visible_facts"] = []
        return facts
    facts = [
        str(fact).strip()
        for fact in active.pending_visible_facts
        if str(fact).strip()
    ]
    active.pending_visible_facts = []
    return facts


def runtime_effects_for_character(
    character: CharacterRecord | None,
) -> list[DndRuntimeEffect]:
    """Read D&D runtime effects from the adapter-owned mechanics bag."""
    if character is None:
        return []
    runtime = (character.mechanics or {}).get(DND_RUNTIME_KEY) or {}
    raw_effects = runtime.get(DND_ACTIVE_EFFECTS_KEY) or []
    if not isinstance(raw_effects, list):
        return []
    out: list[DndRuntimeEffect] = []
    for raw in raw_effects:
        try:
            effect = DndRuntimeEffect.model_validate(raw)
        except Exception:
            continue
        if not effect.target_id:
            effect.target_id = character.character_id
        if not effect.effect_id:
            effect.effect_id = _new_effect_id()
        out.append(effect)
    return out


def set_runtime_effects_for_character(
    character: CharacterRecord,
    effects: Iterable[DndRuntimeEffect],
) -> None:
    mechanics_state = dict(character.mechanics or {})
    runtime = dict(mechanics_state.get(DND_RUNTIME_KEY) or {})
    runtime[DND_ACTIVE_EFFECTS_KEY] = [
        effect.model_dump()
        for effect in effects
        if effect.target_id in {"", character.character_id}
    ]
    mechanics_state[DND_RUNTIME_KEY] = runtime
    character.mechanics = mechanics_state


def sync_combat_effects_to_characters(
    combat: DndCombatState | SessionState | None,
    characters: Iterable[CharacterRecord],
) -> None:
    active = combat.active_combat if isinstance(combat, SessionState) else combat
    if active is None:
        return
    by_id = {character.character_id: character for character in characters}
    for combatant in active.combatants:
        cid = combatant.character_id or combatant.combatant_id
        character = by_id.get(cid)
        if character is None:
            continue
        effects = [
            effect for effect in combatant.active_effects
            if effect.target_id in {"", cid, combatant.combatant_id}
        ]
        set_runtime_effects_for_character(character, effects)


def start_effect(
    combat: DndCombatState | SessionState,
    effect: DndRuntimeEffect,
) -> DndRuntimeEffect:
    active = _active_from(combat)
    target = _find_combatant(active, effect.target_id)
    if not effect.effect_id:
        effect.effect_id = _new_effect_id()
    if not effect.target_id:
        effect.target_id = target.character_id or target.combatant_id
    if not effect.slug:
        effect.slug = _slug(effect.name or effect.effect_id)
    if effect.duration_kind == "minutes" and not effect.remaining_rounds:
        effect.remaining_rounds = max(0, effect.duration_amount * 10)
    elif effect.duration_kind == "rounds" and not effect.remaining_rounds:
        effect.remaining_rounds = max(0, effect.duration_amount)
    if effect.concentration and effect.originator_id:
        end_concentration_effects(
            active,
            effect.originator_id,
            except_effect_id=effect.effect_id,
            reason="new concentration",
        )
    target.active_effects = [
        existing for existing in target.active_effects
        if existing.effect_id != effect.effect_id
    ]
    target.active_effects.append(effect)
    target.conditions = _merge_conditions(target.conditions, effect.conditions)
    _append_pending_visible_fact(
        active,
        f"{_combatant_label(target)} is affected by {effect.name or effect.slug}.",
    )
    append_audit_line(
        active,
        f"Effect started on {_combatant_label(target)}: "
        f"{effect.name or effect.slug} ({effect.effect_id}).",
    )
    return effect


def end_effect(
    combat: DndCombatState | SessionState,
    *,
    effect_id: str = "",
    target_id: str = "",
    slug: str = "",
    reason: str = "",
) -> list[DndRuntimeEffect]:
    active = _active_from(combat)
    ended: list[DndRuntimeEffect] = []
    slug = slug.strip().lower()
    for combatant in active.combatants:
        if target_id:
            ids = {combatant.combatant_id, combatant.character_id}
            if target_id not in ids:
                continue
        remaining: list[DndRuntimeEffect] = []
        ended_here: list[DndRuntimeEffect] = []
        for effect in combatant.active_effects:
            matches = (
                (effect_id and effect.effect_id == effect_id)
                or (slug and effect.slug == slug)
                or (not effect_id and not slug and target_id)
            )
            if matches:
                ended.append(effect)
                ended_here.append(effect)
            else:
                remaining.append(effect)
        if len(remaining) != len(combatant.active_effects):
            ended_conditions = [
                condition for effect in ended_here for condition in effect.conditions
            ]
            combatant.active_effects = remaining
            _reconcile_effect_conditions(
                combatant, ended_conditions=ended_conditions,
            )
    for effect in ended:
        target = _find_combatant(active, effect.target_id)
        _append_pending_visible_fact(
            active,
            f"{effect.name or effect.slug} ends on {_combatant_label(target)}.",
        )
        note = f"Effect ended: {effect.name or effect.slug} ({effect.effect_id})"
        if reason:
            note += f"; reason={reason}"
        append_audit_line(active, note + ".")
    return ended


def end_concentration_effects(
    combat: DndCombatState | SessionState,
    originator_id: str,
    *,
    except_effect_id: str = "",
    reason: str = "",
) -> list[DndRuntimeEffect]:
    active = _active_from(combat)
    ended: list[DndRuntimeEffect] = []
    for combatant in active.combatants:
        remaining: list[DndRuntimeEffect] = []
        ended_here: list[DndRuntimeEffect] = []
        for effect in combatant.active_effects:
            if (
                effect.concentration
                and effect.originator_id == originator_id
                and effect.effect_id != except_effect_id
            ):
                ended.append(effect)
                ended_here.append(effect)
            else:
                remaining.append(effect)
        if len(remaining) != len(combatant.active_effects):
            ended_conditions = [
                condition for effect in ended_here for condition in effect.conditions
            ]
            combatant.active_effects = remaining
            _reconcile_effect_conditions(
                combatant, ended_conditions=ended_conditions,
            )
    for effect in ended:
        target = _find_combatant(active, effect.target_id)
        _append_pending_visible_fact(
            active,
            f"{effect.name or effect.slug} ends on {_combatant_label(target)}.",
        )
        append_audit_line(
            active,
            f"Concentration effect ended for {originator_id}: "
            f"{effect.name or effect.slug} ({effect.effect_id}); {reason}.",
        )
    return ended


def apply_action_tags(
    combat: DndCombatState | SessionState,
    actor_id: str,
    tags: Iterable[str],
) -> list[DndRuntimeEffect]:
    active = _active_from(combat)
    normalized = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
    if not normalized:
        return []
    combatant = _find_combatant(active, actor_id)
    remaining: list[DndRuntimeEffect] = []
    ended: list[DndRuntimeEffect] = []
    for effect in combatant.active_effects:
        triggers = set(effect.break_triggers or [])
        if triggers & normalized:
            ended.append(effect)
        else:
            remaining.append(effect)
    if not ended:
        return []
    combatant.active_effects = remaining
    _reconcile_effect_conditions(
        combatant,
        ended_conditions=[
            condition for effect in ended for condition in effect.conditions
        ],
    )
    for effect in ended:
        _append_pending_visible_fact(
            active,
            f"{effect.name or effect.slug} ends on {_combatant_label(combatant)}.",
        )
        append_audit_line(
            active,
            f"Effect ended by action tag {sorted(normalized)}: "
            f"{effect.name or effect.slug} ({effect.effect_id}).",
        )
    return ended


def _cancel_combat_roll_transactions(session: SessionState) -> set[str]:
    event_ids: set[str] = set()
    for transaction in session.cat_ii_roll_transactions:
        if transaction.source != "combat":
            continue
        if transaction.status in {"finalized", "cancelled"}:
            continue
        transaction.status = "cancelled"
        transaction.updated_at = _utcnow_iso()
        event_ids.add(transaction.event_id)
        for record in transaction.rolls:
            if record.status == "pending":
                record.status = "cancelled"
    return event_ids


def _clear_combat_slots(session: SessionState) -> None:
    cancelled_event_ids = {
        transaction.event_id
        for transaction in session.cat_ii_roll_transactions
        if transaction.source == "combat" and transaction.status == "cancelled"
    }
    session.active_act_slots = {
        cid: slot for cid, slot in session.active_act_slots.items()
        if not (
            slot.reason in {"combat_reaction", "combat_blocked"}
            or (
                slot.reason == "cat_ii_roll"
                and (slot.cat_ii_event_id or "") in cancelled_event_ids
            )
        )
    }


def sort_turn_order(combat: DndCombatState) -> DndCombatState:
    indexed = list(enumerate(combat.combatants))
    indexed.sort(key=lambda item: (-item[1].initiative_total, item[0]))
    combat.combatants = [combatant for _, combatant in indexed]
    for index, combatant in enumerate(combat.combatants):
        combatant.initiative_order = index + 1
    combat.turn_index = _clamp_turn_index(combat, combat.turn_index)
    return combat


def current_combatant(combat: DndCombatState | SessionState) -> DndCombatantState:
    active = combat.active_combat if isinstance(combat, SessionState) else combat
    if active is None or not active.combatants:
        raise ValueError("Combat is not active.")
    active.turn_index = _clamp_turn_index(active, active.turn_index)
    return active.combatants[active.turn_index]


def advance_turn(combat: DndCombatState | SessionState) -> DndCombatantState:
    return advance_turn_with_effects(combat)


def advance_turn_with_effects(
    combat: DndCombatState | SessionState,
    *,
    characters: Iterable[CharacterRecord] | None = None,
) -> DndCombatantState:
    active = combat.active_combat if isinstance(combat, SessionState) else combat
    if active is None or not active.combatants:
        raise ValueError("Combat is not active.")
    if not _has_turn_candidates(active):
        raise ValueError("Combat has no available combatants.")

    active.turn_index = _clamp_turn_index(active, active.turn_index)
    current = active.combatants[active.turn_index]
    _end_turn(active, current, characters=characters)
    start = active.turn_index + 1
    _move_to_next_available(
        active,
        starting_at=start,
        count_round_wrap=True,
        characters=characters,
    )
    return current_combatant(active)


def add_combatant(
    combat: DndCombatState | SessionState,
    character: CharacterRecord,
    *,
    combatant_id: str | None = None,
    session: SessionState | None = None,
) -> DndCombatantState:
    active = _active_from(combat)
    new_combatant_id = combatant_id or character.character_id
    if any(c.combatant_id == new_combatant_id for c in active.combatants):
        raise ValueError("Combatant is already in combat.")

    owning_session = combat if isinstance(combat, SessionState) else session
    current_id = current_combatant(active).combatant_id if active.combatants else ""
    combatant = build_combatant(
        character,
        session=owning_session if isinstance(owning_session, SessionState) else None,
        combatant_id=combatant_id,
    )
    active.combatants.append(combatant)
    _roll_single_initiative(combatant)
    sort_turn_order(active)
    if current_id:
        _set_turn_to_combatant(active, current_id)
    else:
        _move_to_next_available(active, starting_at=0, count_round_wrap=False)
    return combatant


def remove_combatant(
    combat: DndCombatState | SessionState,
    combatant_id: str,
    *,
    hard: bool = False,
) -> DndCombatantState:
    active = _active_from(combat)
    combatant = _find_combatant(active, combatant_id)
    active.turn_index = _clamp_turn_index(active, active.turn_index)
    canonical_id = combatant.combatant_id
    was_current = active.combatants[active.turn_index].combatant_id == canonical_id
    if hard:
        active.combatants = [
            candidate
            for candidate in active.combatants
            if candidate.combatant_id != canonical_id
        ]
        sort_turn_order(active)
    else:
        combatant.removed = True
    if active.combatants and was_current and _has_available_combatants(active):
        _move_to_next_available(
            active,
            starting_at=active.turn_index,
            count_round_wrap=False,
        )
    return combatant


def apply_damage(
    combat: DndCombatState | SessionState,
    combatant_id: str,
    amount: int,
    *,
    characters: Iterable[CharacterRecord] | None = None,
) -> DndCombatantState:
    if amount < 0:
        raise ValueError("Damage amount must be non-negative.")
    active = _active_from(combat)
    combatant = _find_combatant(active, combatant_id)
    uses_death_saves = _uses_death_saves(combat, combatant)
    if amount == 0:
        return combatant

    if combatant.hit_points_current <= 0 and uses_death_saves:
        if _defeat_state(combatant) in {"down", "stable"}:
            if combatant.hit_points_max > 0 and amount >= combatant.hit_points_max:
                _set_dead(combatant)
                _append_pending_visible_fact(
                    active,
                    f"{combatant.name or combatant.combatant_id} dies.",
                )
            else:
                _add_death_save_failures(combatant, 1)
        _process_concentration_damage(
            active, combatant, amount, characters=characters,
        )
        return combatant

    previous_hp = combatant.hit_points_current
    remaining = amount
    temp_loss = min(combatant.hit_points_temporary, remaining)
    combatant.hit_points_temporary -= temp_loss
    remaining -= temp_loss
    combatant.hit_points_current = max(0, combatant.hit_points_current - remaining)
    if combatant.hit_points_current <= 0:
        overkill = max(0, remaining - max(previous_hp, 0))
        if uses_death_saves:
            if combatant.hit_points_max > 0 and overkill >= combatant.hit_points_max:
                _set_dead(combatant)
                _append_pending_visible_fact(
                    active,
                    f"{combatant.name or combatant.combatant_id} dies.",
                )
            else:
                _set_down(combatant, reset_saves=True)
        else:
            _set_defeated(combatant)
    else:
        _set_active(combatant)
    _process_concentration_damage(
        active, combatant, amount, characters=characters,
    )
    return combatant


def apply_healing(
    combat: DndCombatState | SessionState,
    combatant_id: str,
    amount: int,
) -> DndCombatantState:
    if amount < 0:
        raise ValueError("Healing amount must be non-negative.")
    combatant = _find_combatant(_active_from(combat), combatant_id)
    cap = combatant.hit_points_max
    if cap <= 0:
        cap = combatant.hit_points_current + amount
    combatant.hit_points_current = min(cap, combatant.hit_points_current + amount)
    if combatant.hit_points_current > 0:
        _set_active(combatant)
    return combatant


def roll_death_save(
    combat: DndCombatState | SessionState,
    combatant_id: str,
) -> dice.RollResult:
    active = _active_from(combat)
    combatant = _find_combatant(active, combatant_id)
    if _defeat_state(combatant) != "down":
        raise ValueError(f"Combatant {combatant_id!r} is not down.")

    result = dice.roll_d20_check(
        roll_id=f"death_save_{combatant.combatant_id}",
        modifier=0,
        actor_id=combatant.character_id or combatant.combatant_id,
        reason="death save",
        advantage_state="normal",
    )
    natural = _kept_d20_value(result)
    if natural == 20:
        combatant.hit_points_current = 1
        _set_active(combatant)
        _append_pending_visible_fact(
            active,
            f"{combatant.name or combatant.combatant_id} regains consciousness.",
        )
        append_audit_line(
            active,
            f"Death save for {combatant.name or combatant.combatant_id}: "
            "natural 20; they regain 1 HP.",
        )
    elif natural == 1:
        _add_death_save_failures(combatant, 2)
        if _defeat_state(combatant) == "dead":
            _append_pending_visible_fact(
                active,
                f"{combatant.name or combatant.combatant_id} dies.",
            )
            append_audit_line(
                active,
                f"Death save for {combatant.name or combatant.combatant_id}: "
                f"{result.detail}; natural 1; they die.",
            )
        else:
            append_audit_line(
                active,
                f"Death save for {combatant.name or combatant.combatant_id}: "
                f"{result.detail}; two failures "
                f"({combatant.death_save_failures}/3).",
            )
    elif result.total >= 10:
        _add_death_save_success(combatant)
        if _defeat_state(combatant) == "stable":
            _append_pending_visible_fact(
                active,
                f"{combatant.name or combatant.combatant_id} stabilizes.",
            )
            append_audit_line(
                active,
                f"Death save for {combatant.name or combatant.combatant_id}: "
                f"{result.detail}; third success; they are stable.",
            )
        else:
            append_audit_line(
                active,
                f"Death save for {combatant.name or combatant.combatant_id}: "
                f"{result.detail}; success "
                f"({combatant.death_save_successes}/3).",
            )
    else:
        _add_death_save_failures(combatant, 1)
        if _defeat_state(combatant) == "dead":
            _append_pending_visible_fact(
                active,
                f"{combatant.name or combatant.combatant_id} dies.",
            )
            append_audit_line(
                active,
                f"Death save for {combatant.name or combatant.combatant_id}: "
                f"{result.detail}; third failure; they die.",
            )
        else:
            append_audit_line(
                active,
                f"Death save for {combatant.name or combatant.combatant_id}: "
                f"{result.detail}; failure "
                f"({combatant.death_save_failures}/3).",
            )
    return result


def public_status(combat: DndCombatState | SessionState) -> dict[str, Any]:
    active = _active_from(combat)
    current = current_combatant(active)
    return {
        "combat_id": active.combat_id,
        "round": active.round_number,
        "current": _public_combatant(current),
        "turn_order": [_public_combatant(c) for c in active.combatants],
    }


def private_status(combat: DndCombatState | SessionState) -> dict[str, Any]:
    active = _active_from(combat)
    current = current_combatant(active)
    return {
        "combat_id": active.combat_id,
        "round": active.round_number,
        "turn_index": active.turn_index,
        "current": _private_combatant(current),
        "turn_order": [_private_combatant(c) for c in active.combatants],
    }


def _roll_single_initiative(combatant: DndCombatantState) -> None:
    result = dice.roll_d20_check(
        roll_id=f"initiative_{combatant.combatant_id}",
        modifier=combatant.initiative_modifier,
        actor_id=combatant.character_id or combatant.combatant_id,
        reason="initiative",
        advantage_state=_initiative_advantage_state(combatant),
    )
    combatant.initiative_roll = _kept_d20_value(result)
    combatant.initiative_total = result.total
    combatant.initiative_detail = result.detail


def _active_from(combat: DndCombatState | SessionState) -> DndCombatState:
    if isinstance(combat, SessionState):
        return _require_combat(combat)
    return combat


def _require_combat(session: SessionState) -> DndCombatState:
    if session.active_combat is None:
        raise ValueError("Combat is not active.")
    return session.active_combat


def _move_to_next_available(
    combat: DndCombatState,
    *,
    starting_at: int,
    count_round_wrap: bool,
    characters: Iterable[CharacterRecord] | None = None,
) -> None:
    total = len(combat.combatants)
    if total == 0:
        combat.turn_index = 0
        return
    for offset in range(total):
        raw_index = starting_at + offset
        index = raw_index % total
        if count_round_wrap and raw_index >= total and index == 0:
            combat.round_number += 1
        candidate = combat.combatants[index]
        if _defeat_state(candidate) == "down":
            roll_death_save(combat, candidate.combatant_id)
        if _available(candidate):
            combat.turn_index = index
            _begin_turn(combat, candidate, characters=characters)
            return
    combat.turn_index = _clamp_turn_index(combat, combat.turn_index)


def _begin_turn(
    combat: DndCombatState,
    combatant: DndCombatantState,
    *,
    characters: Iterable[CharacterRecord] | None = None,
) -> None:
    combatant.reaction_available = True
    _process_recurring_saves(
        combat,
        combatant,
        characters=characters,
        timing="start_of_turn",
    )


def _end_turn(
    combat: DndCombatState,
    combatant: DndCombatantState,
    *,
    characters: Iterable[CharacterRecord] | None = None,
) -> None:
    _process_recurring_saves(
        combat,
        combatant,
        characters=characters,
        timing="end_of_turn",
    )
    _tick_effect_durations(combat, combatant)


def _process_recurring_saves(
    combat: DndCombatState,
    combatant: DndCombatantState,
    *,
    characters: Iterable[CharacterRecord] | None = None,
    timing: str,
) -> None:
    if _defeat_state(combatant) not in {"active", "down"}:
        return
    remaining: list[DndRuntimeEffect] = []
    ended: list[DndRuntimeEffect] = []
    by_id = _characters_by_id(characters)
    character = by_id.get(combatant.character_id)
    for effect in combatant.active_effects:
        save = effect.recurring_save
        if save is None or not save.repeat or save.timing != timing:
            remaining.append(effect)
            continue
        if not save.ability or save.dc <= 0:
            remaining.append(effect)
            continue
        modifier = _saving_throw_modifier(character, save.ability)
        result = dice.roll_d20_check(
            roll_id=f"effect_save_{effect.effect_id}",
            modifier=modifier,
            actor_id=combatant.character_id or combatant.combatant_id,
            reason=f"Recurring save for {effect.name or effect.slug}",
            advantage_state="normal",
        )
        success = result.total >= save.dc
        should_end = (
            (save.ends_on == "success" and success)
            or (save.ends_on == "failure" and not success)
        )
        append_audit_line(
            combat,
            f"Recurring save for {_combatant_label(combatant)} "
            f"against {effect.name or effect.slug}: {result.detail} "
            f"vs DC {save.dc}; {'ends' if should_end else 'continues'}.",
        )
        if should_end:
            ended.append(effect)
        else:
            remaining.append(effect)
            _append_pending_visible_fact(
                combat,
                f"{effect.name or effect.slug} remains on "
                f"{_combatant_label(combatant)}.",
            )
    if ended:
        combatant.active_effects = remaining
        _reconcile_effect_conditions(
            combatant,
            ended_conditions=[
                condition for effect in ended for condition in effect.conditions
            ],
        )
        for effect in ended:
            _append_pending_visible_fact(
                combat,
                f"{effect.name or effect.slug} ends on "
                f"{_combatant_label(combatant)}.",
            )


def _tick_effect_durations(
    combat: DndCombatState,
    combatant: DndCombatantState,
) -> None:
    remaining: list[DndRuntimeEffect] = []
    expired: list[DndRuntimeEffect] = []
    for effect in combatant.active_effects:
        if effect.remaining_rounds <= 0:
            remaining.append(effect)
            continue
        effect.remaining_rounds -= 1
        if effect.remaining_rounds <= 0:
            expired.append(effect)
        else:
            remaining.append(effect)
    if not expired:
        return
    combatant.active_effects = remaining
    _reconcile_effect_conditions(
        combatant,
        ended_conditions=[
            condition for effect in expired for condition in effect.conditions
        ],
    )
    for effect in expired:
        _append_pending_visible_fact(
            combat,
            f"{effect.name or effect.slug} ends on {_combatant_label(combatant)}.",
        )
        append_audit_line(
            combat,
            f"Effect expired on {_combatant_label(combatant)}: "
            f"{effect.name or effect.slug} ({effect.effect_id}).",
        )


def _has_available_combatants(combat: DndCombatState) -> bool:
    return any(_available(combatant) for combatant in combat.combatants)


def _has_turn_candidates(combat: DndCombatState) -> bool:
    return any(
        not combatant.removed
        and _defeat_state(combatant) in {"active", "down"}
        for combatant in combat.combatants
    )


def _available(combatant: DndCombatantState) -> bool:
    return (
        _defeat_state(combatant) == "active"
        and not combatant.removed
    )


def _find_combatant(
    combat: DndCombatState,
    combatant_id: str,
) -> DndCombatantState:
    for combatant in combat.combatants:
        if (
            combatant.combatant_id == combatant_id
            or combatant.character_id == combatant_id
        ):
            return combatant
    raise ValueError(f"No combatant {combatant_id!r} in combat.")


def _set_turn_to_combatant(combat: DndCombatState, combatant_id: str) -> None:
    for index, combatant in enumerate(combat.combatants):
        if combatant.combatant_id == combatant_id:
            combat.turn_index = index
            return


def _clamp_turn_index(combat: DndCombatState, index: int) -> int:
    if not combat.combatants:
        return 0
    return max(0, min(index, len(combat.combatants) - 1))


def _public_combatant(combatant: DndCombatantState) -> dict[str, Any]:
    return {
        "combatant_id": combatant.combatant_id,
        "character_id": combatant.character_id,
        "name": combatant.name,
        "hp": {
            "current": combatant.hit_points_current,
            "max": combatant.hit_points_max,
            "temporary": combatant.hit_points_temporary,
        },
        "defeat_state": _defeat_state(combatant),
        "death_saves": {
            "successes": combatant.death_save_successes,
            "failures": combatant.death_save_failures,
        },
        "active_effects": [_effect_public_summary(e) for e in combatant.active_effects],
        "removed": combatant.removed,
        "pending_initiating_action": combatant.pending_initiating_action,
        "pending_initiating_event_id": combatant.pending_initiating_event_id,
    }


def _private_combatant(combatant: DndCombatantState) -> dict[str, Any]:
    data = _public_combatant(combatant)
    data.update({
        "player_controlled": combatant.player_controlled,
        "armor_class": combatant.armor_class,
        "initiative": {
            "roll": combatant.initiative_roll,
            "modifier": combatant.initiative_modifier,
            "total": combatant.initiative_total,
            "detail": combatant.initiative_detail,
            "order": combatant.initiative_order,
        },
        "conditions": list(combatant.conditions),
        "reaction_available": combatant.reaction_available,
        "notes": combatant.notes,
    })
    return data


def _effect_public_summary(effect: DndRuntimeEffect) -> dict[str, Any]:
    return {
        "effect_id": effect.effect_id,
        "name": effect.name,
        "slug": effect.slug,
        "conditions": list(effect.conditions),
        "concentration": effect.concentration,
        "remaining_rounds": effect.remaining_rounds,
        "duration_text": effect.duration_text,
        "recurring_save": bool(effect.recurring_save),
    }


def _hit_points(mechanics_state: dict[str, Any]) -> dict[str, Any]:
    hp = mechanics_state.get("hit_points")
    if isinstance(hp, dict):
        return hp
    sheet_hp = (
        ((mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {})
        .get("defenses", {})
        .get("hit_points")
    )
    return sheet_hp if isinstance(sheet_hp, dict) else {}


def _armor_class(mechanics_state: dict[str, Any]) -> int:
    direct = mechanics_state.get("armor_class")
    if direct is not None:
        return _safe_int(direct, 10)
    sheet_ac = (
        ((mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {})
        .get("defenses", {})
        .get("armor_class")
    )
    if isinstance(sheet_ac, dict):
        return _safe_int(sheet_ac.get("value"), 10)
    return 10


def _conditions(mechanics_state: dict[str, Any]) -> list[str]:
    conditions = mechanics_state.get("conditions")
    if conditions is None:
        conditions = (
            ((mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {})
            .get("defenses", {})
            .get("conditions")
        )
    if not isinstance(conditions, list):
        return []
    out: list[str] = []
    for condition in conditions:
        if isinstance(condition, dict):
            name = str(condition.get("name") or condition.get("id") or "").strip()
        else:
            name = str(condition).strip()
        if name:
            out.append(name)
    return out


def _effect_conditions(effects: Iterable[DndRuntimeEffect]) -> list[str]:
    out: list[str] = []
    for effect in effects:
        out.extend(effect.conditions)
    return out


def _merge_conditions(
    existing: Iterable[str],
    added: Iterable[str],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for condition in [*existing, *added]:
        text = str(condition).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _reconcile_effect_conditions(
    combatant: DndCombatantState,
    ended_conditions: Iterable[str] = (),
) -> None:
    effect_conditions = {
        condition.strip().lower()
        for effect in combatant.active_effects
        for condition in effect.conditions
        if condition.strip()
    }
    removed_effect_conditions = {
        condition.strip().lower() for condition in ended_conditions
        if condition.strip()
    }
    # The flat conditions list has no provenance, so effect removal owns the
    # condition names attached to the ended effect unless another active effect
    # still grants the same condition.
    current = _merge_conditions(combatant.conditions, [])
    if removed_effect_conditions:
        combatant.conditions = [
            condition for condition in current
            if (
                condition.strip().lower() not in removed_effect_conditions
                or condition.strip().lower() in effect_conditions
            )
        ]
    else:
        combatant.conditions = current
    combatant.conditions = _merge_conditions(
        combatant.conditions, _effect_conditions(combatant.active_effects),
    )


def _characters_by_id(
    characters: Iterable[CharacterRecord] | None,
) -> dict[str, CharacterRecord]:
    if characters is None:
        return {}
    return {character.character_id: character for character in characters}


def _saving_throw_modifier(
    character: CharacterRecord | None,
    ability: str,
) -> int:
    if character is None:
        return 0
    try:
        from app.schemas.dnd_cat_ii import PlannedRoll

        request = PlannedRoll(
            roll_id="effect_save",
            actor_id=character.character_id,
            kind="saving_throw",
            ability=ability,  # type: ignore[arg-type]
            skill="",
            dc=0,
            opposed_by="",
            advantage_state="normal",
            reason="effect save",
            action_id="",
            target_id="",
            effect_id="",
        )
        return mechanics.roll_modifier(character, request)
    except Exception:
        return 0


def _process_concentration_damage(
    combat: DndCombatState,
    combatant: DndCombatantState,
    amount: int,
    *,
    characters: Iterable[CharacterRecord] | None = None,
) -> None:
    if amount <= 0:
        return
    owner_id = combatant.character_id or combatant.combatant_id
    if not _has_concentration_effect(combat, owner_id):
        return
    dc = max(10, amount // 2)
    modifier = _saving_throw_modifier(_characters_by_id(characters).get(owner_id), "con")
    result = dice.roll_d20_check(
        roll_id=f"concentration_{owner_id}",
        modifier=modifier,
        actor_id=owner_id,
        reason="concentration check",
        advantage_state="normal",
    )
    append_audit_line(
        combat,
        f"Concentration check for {_combatant_label(combatant)}: "
        f"{result.detail} vs DC {dc}.",
    )
    if result.total < dc:
        end_concentration_effects(
            combat,
            owner_id,
            reason="failed concentration save",
        )


def _has_concentration_effect(combat: DndCombatState, owner_id: str) -> bool:
    for combatant in combat.combatants:
        for effect in combatant.active_effects:
            if effect.concentration and effect.originator_id == owner_id:
                return True
    return False


def _new_effect_id() -> str:
    return f"eff_{uuid.uuid4().hex[:12]}"


def _slug(value: object) -> str:
    text = str(value or "").strip().lower()
    out = []
    prev_dash = False
    for char in text:
        if char.isalnum():
            out.append(char)
            prev_dash = False
        elif not prev_dash:
            out.append("_")
            prev_dash = True
    return "".join(out).strip("_")


def _combatant_label(combatant: DndCombatantState) -> str:
    return combatant.name or combatant.character_id or combatant.combatant_id


def _death_save_count(mechanics_state: dict[str, Any], key: str) -> int:
    direct = mechanics_state.get("death_saves")
    if not isinstance(direct, dict):
        direct = (
            ((mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {})
            .get("defenses", {})
            .get("death_saves")
        )
    if not isinstance(direct, dict):
        return 0
    aliases = {
        "successes": ("successes", "success_count", "successCount"),
        "failures": ("failures", "fail_count", "failCount"),
    }
    for name in aliases.get(key, (key,)):
        if name in direct:
            return max(0, min(3, _safe_int(direct.get(name), 0)))
    return 0


def _initiative_modifier(mechanics_state: dict[str, Any]) -> int:
    initiative = (
        ((mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {})
        .get("defenses", {})
        .get("initiative")
    )
    if isinstance(initiative, dict) and "value" in initiative:
        return _safe_int(initiative.get("value"), 0)

    scores = mechanics_state.get("ability_scores") or {}
    if not isinstance(scores, dict):
        return 0
    dex = scores.get("dex", 10)
    if isinstance(dex, dict):
        if "modifier" in dex:
            return _safe_int(dex.get("modifier"), 0)
        dex = dex.get("score", 10)
    return mechanics.ability_modifier(_safe_int(dex, 10))


def _initiative_advantage_state(combatant: DndCombatantState) -> dice.AdvantageState:
    state = combatant.initiative_advantage_state
    if state in {"advantage", "disadvantage"}:
        return state
    return "normal"


def _initiative_advantage_state_from_mechanics(
    mechanics_state: dict[str, Any],
) -> dice.AdvantageState:
    initiative = (
        ((mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {})
        .get("defenses", {})
        .get("initiative")
    )
    if isinstance(initiative, dict):
        state = str(initiative.get("advantage_state") or "normal")
        if state in {"advantage", "disadvantage"}:
            return state
    return "normal"


def _kept_d20_value(result: dice.RollResult) -> int:
    for die in result.dice:
        if die.kept and die.values:
            return die.values[0]
    if result.dice and result.dice[0].values:
        return result.dice[0].values[0]
    return result.total


def _uses_death_saves(
    combat: DndCombatState | SessionState | None,
    combatant: DndCombatantState,
) -> bool:
    if combatant.player_controlled:
        return True
    if isinstance(combat, SessionState):
        ids = {combatant.character_id, combatant.combatant_id}
        return bool(ids & set(combat.character_bindings or {}))
    return False


def _defeat_state(combatant: DndCombatantState) -> str:
    state = str(getattr(combatant, "defeat_state", "") or "")
    if state:
        return state
    return "active"


def _sync_defeat_state(
    combatant: DndCombatantState,
    *,
    uses_death_saves: bool,
) -> None:
    if combatant.removed:
        return
    if combatant.hit_points_max <= 0 or combatant.hit_points_current > 0:
        _set_active(combatant)
        return
    if not uses_death_saves:
        _set_defeated(combatant)
        return
    _clamp_death_saves(combatant)
    if combatant.death_save_failures >= 3:
        _set_dead(combatant)
    elif combatant.death_save_successes >= 3:
        _set_stable(combatant)
    else:
        _set_down(combatant, reset_saves=False)


def _set_active(combatant: DndCombatantState) -> None:
    combatant.defeat_state = "active"
    combatant.death_save_successes = 0
    combatant.death_save_failures = 0
    _remove_condition(combatant, "unconscious")


def _set_down(
    combatant: DndCombatantState,
    *,
    reset_saves: bool,
) -> None:
    combatant.hit_points_current = 0
    combatant.defeat_state = "down"
    if reset_saves:
        combatant.death_save_successes = 0
        combatant.death_save_failures = 0
    _add_condition(combatant, "unconscious")


def _set_stable(combatant: DndCombatantState) -> None:
    combatant.hit_points_current = 0
    combatant.defeat_state = "stable"
    combatant.death_save_successes = 0
    combatant.death_save_failures = 0
    _add_condition(combatant, "unconscious")


def _set_dead(combatant: DndCombatantState) -> None:
    combatant.hit_points_current = 0
    combatant.defeat_state = "dead"
    combatant.death_save_failures = 3
    _remove_condition(combatant, "unconscious")


def _set_defeated(combatant: DndCombatantState) -> None:
    combatant.hit_points_current = max(0, combatant.hit_points_current)
    combatant.defeat_state = "defeated"
    combatant.death_save_successes = 0
    combatant.death_save_failures = 0


def _add_death_save_success(combatant: DndCombatantState) -> None:
    combatant.death_save_successes = min(3, combatant.death_save_successes + 1)
    combatant.death_save_failures = min(2, combatant.death_save_failures)
    if combatant.death_save_successes >= 3:
        _set_stable(combatant)


def _add_death_save_failures(
    combatant: DndCombatantState,
    count: int,
) -> None:
    combatant.hit_points_current = 0
    if _defeat_state(combatant) == "stable":
        combatant.death_save_successes = 0
        combatant.death_save_failures = 0
    combatant.death_save_failures = min(
        3, combatant.death_save_failures + max(0, count)
    )
    if combatant.death_save_failures >= 3:
        _set_dead(combatant)
    else:
        combatant.defeat_state = "down"
        _add_condition(combatant, "unconscious")


def _clamp_death_saves(combatant: DndCombatantState) -> None:
    combatant.death_save_successes = max(
        0, min(3, int(combatant.death_save_successes or 0))
    )
    combatant.death_save_failures = max(
        0, min(3, int(combatant.death_save_failures or 0))
    )


def _add_condition(combatant: DndCombatantState, condition: str) -> None:
    names = {existing.strip().lower() for existing in combatant.conditions}
    if condition.strip().lower() not in names:
        combatant.conditions.append(condition)


def _remove_condition(combatant: DndCombatantState, condition: str) -> None:
    target = condition.strip().lower()
    combatant.conditions = [
        existing for existing in combatant.conditions
        if existing.strip().lower() != target
    ]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
