from __future__ import annotations

from typing import Any, Iterable

from app.engine import dice, mechanics
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.state import DndCombatantState, DndCombatState, SessionState


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


def end_combat(session: SessionState) -> DndCombatState:
    """Clear active combat and return the ended snapshot."""
    combat = _require_combat(session)
    combat.status = "ended"
    combat.ended_at_turn_index = session.turn_index
    combat.pending_advance_actor_id = ""
    session.active_act_slots = {
        cid: slot for cid, slot in session.active_act_slots.items()
        if slot.reason != "combat_reaction"
    }
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
        conditions=_conditions(mechanics_state),
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
    active = combat.active_combat if isinstance(combat, SessionState) else combat
    if active is None or not active.combatants:
        raise ValueError("Combat is not active.")
    if not _has_turn_candidates(active):
        raise ValueError("Combat has no available combatants.")

    start = _clamp_turn_index(active, active.turn_index) + 1
    _move_to_next_available(active, starting_at=start, count_round_wrap=True)
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
            else:
                _add_death_save_failures(combatant, 1)
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
            else:
                _set_down(combatant, reset_saves=True)
        else:
            _set_defeated(combatant)
    else:
        _set_active(combatant)
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
        append_audit_line(
            active,
            f"Death save for {combatant.name or combatant.combatant_id}: "
            "natural 20; they regain 1 HP.",
        )
    elif natural == 1:
        _add_death_save_failures(combatant, 2)
        if _defeat_state(combatant) == "dead":
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
            _begin_turn(candidate)
            return
    combat.turn_index = _clamp_turn_index(combat, combat.turn_index)


def _begin_turn(combatant: DndCombatantState) -> None:
    combatant.reaction_available = True


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
        and not combatant.defeated
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
        "defeated": combatant.defeated,
        "removed": combatant.removed,
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
    if combatant.defeated and state in {"", "active"}:
        return "defeated"
    if state:
        return state
    return "defeated" if combatant.defeated else "active"


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
    combatant.defeated = False
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
    combatant.defeated = True
    if reset_saves:
        combatant.death_save_successes = 0
        combatant.death_save_failures = 0
    _add_condition(combatant, "unconscious")


def _set_stable(combatant: DndCombatantState) -> None:
    combatant.hit_points_current = 0
    combatant.defeat_state = "stable"
    combatant.defeated = True
    combatant.death_save_successes = 0
    combatant.death_save_failures = 0
    _add_condition(combatant, "unconscious")


def _set_dead(combatant: DndCombatantState) -> None:
    combatant.hit_points_current = 0
    combatant.defeat_state = "dead"
    combatant.defeated = True
    combatant.death_save_failures = 3
    _remove_condition(combatant, "unconscious")


def _set_defeated(combatant: DndCombatantState) -> None:
    combatant.hit_points_current = max(0, combatant.hit_points_current)
    combatant.defeat_state = "defeated"
    combatant.defeated = True
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
        combatant.defeated = True
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
