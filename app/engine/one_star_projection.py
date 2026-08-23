"""POV-safe projections of the optional One-Star mechanics ledger.

This module is deliberately read-only. The adapter owns validation and
persistence; projections select only fields appropriate to a particular
fictional audience and never expose a raw mechanics mapping.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.engine.one_star_adapter import (
    ONE_STAR_RULESET_ID,
    effective_one_star_stamina,
    load_one_star_account,
    load_one_star_hero,
)
from app.engine.one_star_progression import (
    banked_experience_at_current_cap,
    experience_to_reach_level,
)
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import (
    OneStarAccountEnvelope,
    OneStarEquipmentEntry,
    OneStarHeroState,
    OneStarMissionState,
    OneStarPendingOperation,
    OneStarSkillEntry,
)


def _ruleset_id(checkpoint: CheckpointFile) -> str:
    settings = getattr(
        getattr(checkpoint.session, "config", None), "settings", None
    )
    return str(getattr(settings, "ruleset_id", "") or "")


def _account(
    checkpoint: CheckpointFile,
) -> tuple[CharacterRecord, OneStarAccountEnvelope] | None:
    if _ruleset_id(checkpoint) != ONE_STAR_RULESET_ID:
        return None
    return load_one_star_account(checkpoint)


def _visible_equipment(
    hero: OneStarHeroState,
) -> list[OneStarEquipmentEntry]:
    return [item for item in hero.equipment if bool(item.visible)]


def _visible_skills(hero: OneStarHeroState) -> list[OneStarSkillEntry]:
    return [skill for skill in hero.skills if bool(skill.visible)]


def _belongs_to_account(
    hero: OneStarHeroState,
    envelope: OneStarAccountEnvelope,
) -> bool:
    return bool(
        hero.owner_lobby_id
        and hero.owner_lobby_id == envelope.config.lobby_id
    )


def _join_values(values: Iterable[str], *, empty: str = "none") -> str:
    rendered = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(rendered) if rendered else empty


def _summon_rate_text(weights: dict[int, int]) -> str:
    rendered: list[str] = []
    for birth_stars, weight in sorted(weights.items()):
        whole, fractional = divmod(weight, 100)
        percent = (
            f"{whole}%"
            if not fractional
            else f"{whole}.{fractional:02d}".rstrip("0") + "%"
        )
        rendered.append(f"{birth_stars}-star {percent}")
    return ", ".join(rendered)


def _equipment_line(hero: OneStarHeroState, *, exact: bool) -> str:
    parts: list[str] = []
    for item in _visible_equipment(hero):
        label = f"{item.name} ({item.slot})"
        if item.quantity != 1:
            label += f" x{item.quantity}"
        if exact and item.durability_max:
            label += (
                f" durability {item.durability_current}/"
                f"{item.durability_max}"
            )
        parts.append(label)
    return _join_values(parts)


def _skills_line(hero: OneStarHeroState, *, exact: bool) -> str:
    parts: list[str] = []
    for skill in _visible_skills(hero):
        label = skill.name
        if exact:
            label += f" (rank {skill.rank})"
        capability = str(skill.capability or "").strip()
        if capability:
            label += f": {capability}"
        parts.append(label)
    return _join_values(parts)


def _qualitative_condition(hero: OneStarHeroState) -> str:
    if hero.hp_max <= 0 or hero.hp_current <= 0:
        condition = "unconscious or unable to continue"
    else:
        fraction = hero.hp_current / hero.hp_max
        if fraction <= 0.25:
            condition = "critically hurt"
        elif fraction <= 0.5:
            condition = "badly hurt"
        elif fraction < 1:
            condition = "hurt"
        else:
            condition = "physically steady"
    embodied = [*hero.conditions, *hero.persistent_injuries]
    if embodied:
        condition += "; " + _join_values(embodied)
    return condition


def _experience_progress_line(
    hero: OneStarHeroState,
    envelope: OneStarAccountEnvelope,
) -> str:
    config = envelope.config
    current_cap = config.star_level_caps[hero.current_stars]
    if hero.level < current_cap:
        next_threshold = experience_to_reach_level(hero.level + 1, config)
        return (
            f"XP {hero.experience_points}/{next_threshold} "
            f"to level {hero.level + 1}"
        )
    cap_threshold = experience_to_reach_level(current_cap, config)
    bank_limit = (
        experience_to_reach_level(
            current_cap + config.progression.cap_bank_extra_levels,
            config,
        )
        - cap_threshold
    )
    return (
        f"XP {hero.experience_points}; cap-bank "
        f"{banked_experience_at_current_cap(hero, config)}/{bank_limit}"
    )


def _exact_hero_lines(
    name: str,
    hero: OneStarHeroState,
    envelope: OneStarAccountEnvelope,
) -> list[str]:
    stats = _join_values(
        (f"{key} {value}" for key, value in sorted(hero.stats.items()))
    )
    return [
        f"{name}: {hero.current_stars}-star (born {hero.birth_stars}-star), "
        f"level {hero.level}, {_experience_progress_line(hero, envelope)}, "
        f"HP {hero.hp_current}/{hero.hp_max}",
        f"Stats: {stats}",
        f"Skills: {_skills_line(hero, exact=True)}",
        f"Equipment: {_equipment_line(hero, exact=True)}",
        f"Conditions: {_join_values([*hero.conditions, *hero.persistent_injuries])}",
    ]


def _own_hero_lines(
    character: CharacterRecord,
    hero: OneStarHeroState,
    envelope: OneStarAccountEnvelope,
    *,
    exact: bool,
) -> list[str]:
    if exact:
        return _exact_hero_lines(character.name, hero, envelope)
    return [
        f"Your bodily condition: {_qualitative_condition(hero)}.",
        f"What you visibly carry or wear: {_equipment_line(hero, exact=False)}.",
        f"Usable embodied skills: {_skills_line(hero, exact=False)}.",
    ]


def _mission_lines(mission: OneStarMissionState | None) -> list[str]:
    if mission is None:
        return ["Active mission: none"]
    counters = _join_values(
        (
            f"{counter.counter_id} {counter.current}/{counter.target}"
            for counter in sorted(mission.counters, key=lambda entry: entry.counter_id)
        )
    )
    return [
        f"Active mission: {mission.mission_id}; floor {mission.floor}; "
        f"destination {mission.destination}",
        f"Completion: {mission.completion_declaration}",
        f"Failure: {mission.failure_declaration}",
        f"Mission party: {_join_values(mission.party_ids)}; formation "
        + _join_values(
            f"{entry.character_id}={entry.label}"
            for entry in sorted(
                mission.formation_labels,
                key=lambda formation: formation.character_id,
            )
        ),
        "Mission counters: "
        f"{counters}; deadline "
        f"{mission.deadline_at_s if mission.deadline_at_s else 'untimed'}",
    ]


def _pending_operation_lines(
    operation: OneStarPendingOperation | None,
) -> list[str]:
    if operation is None:
        return ["Pending management operation: none"]
    return [
        f"Pending management operation: {operation.kind} "
        f"({operation.operation_id}); participants "
        f"{_join_values(operation.participant_ids)}; target "
        f"{operation.target_id or 'none'}; destination "
        f"{operation.destination or 'none'}; opened at "
        f"{operation.opened_at_s}s"
    ]


def _management_lines(
    envelope: OneStarAccountEnvelope,
    *,
    include_active_feed: bool,
    include_stored_equipment: bool = False,
    canonical_now_s: int,
) -> list[str]:
    state = envelope.state
    facilities = _join_values(
        (f"{key} {value}" for key, value in sorted(state.facilities.items()))
    )
    inventory = _join_values(
        (f"{key} x{value}" for key, value in sorted(state.inventory.items()))
    )
    materials = _join_values(
        (
            f"{key} x{value}"
            for key, value in sorted(state.resources.materials.items())
        )
    )
    summon_pools = _join_values(
        (
            f"{pool_id}: {pool.cost.gold} Gold, {pool.cost.gems} Gems, "
            f"{pool.cost.building_resources} Building Resources; "
            f"birth stars {pool.minimum_birth_stars}-"
            f"{pool.maximum_birth_stars}; rates {_summon_rate_text(pool.star_weights)}"
            for pool_id, pool in sorted(envelope.config.summon_pools.items())
            if pool.usage == "standard"
        )
    )
    research = _join_values(
        (
            f"{key} {level}"
            for key, level in sorted(state.research_levels.items())
        )
    )
    stamina_current, stamina_anchor = effective_one_star_stamina(
        state,
        envelope.config,
        canonical_now_s,
    )
    lines = [
        "Wallet: "
        f"Gold {state.resources.gold}; Gems {state.resources.gems}; "
        f"Building Resources {state.resources.building_resources}",
        f"Materials: {materials}",
        f"Facilities: {facilities}; lobby floor {state.lobby_floor}; "
        f"highest cleared floor {state.highest_cleared_floor}; "
        f"highest unlocked floor {state.highest_unlocked_floor}",
        f"Hero capacity: {state.capacity}; stamina "
        f"{stamina_current}/{envelope.config.maximum_stamina}",
        f"Stamina recovery: anchor {stamina_anchor}s; "
        f"one per {envelope.config.stamina_recovery_seconds}s",
        f"Summon pools (maximum batch {envelope.config.max_summon_batch}): "
        f"{summon_pools}",
        f"Research: {research}",
        f"Account inventory: {inventory}",
    ]
    if include_stored_equipment:
        stored_equipment = _join_values(
            (
                f"{item.name} [{item.item_id}] ({item.slot})"
                + (f" x{item.quantity}" if item.quantity != 1 else "")
                + (
                    f" durability {item.durability_current}/{item.durability_max}"
                    if item.durability_max
                    else ""
                )
                for item in state.stored_equipment
            )
        )
        lines.append(f"Stored equipment: {stored_equipment}")
    if include_active_feed:
        lines.append(f"Active feed: {state.active_master_feed_id or 'none'}")
    return lines


def _public_roster_lines(
    checkpoint: CheckpointFile,
    envelope: OneStarAccountEnvelope,
) -> list[str]:
    lines = ["Public Hero roster:"]
    found = False
    for character in checkpoint.characters:
        hero = load_one_star_hero(character)
        if hero is None or not _belongs_to_account(hero, envelope):
            continue
        found = True
        lifecycle = str(
            character.status.value
            if hasattr(character.status, "value")
            else character.status
        )
        hero_lines = _exact_hero_lines(character.name, hero, envelope)
        summary = hero_lines[0]
        terminal = (
            f"; terminal cause {hero.terminal_cause}"
            if hero.terminal_cause
            else ""
        )
        lines.append(
            f"- {summary}; lifecycle {lifecycle}; "
            f"location {character.location or 'unknown'}{terminal}"
        )
        lines.extend(f"  {line}" for line in hero_lines[1:])
    if not found:
        lines.append("- none")
    return lines


def _owned_hero_records(
    checkpoint: CheckpointFile,
    envelope: OneStarAccountEnvelope,
) -> list[tuple[CharacterRecord, OneStarHeroState]]:
    records: list[tuple[CharacterRecord, OneStarHeroState]] = []
    for character in checkpoint.characters:
        hero = load_one_star_hero(character)
        if hero is not None and _belongs_to_account(hero, envelope):
            records.append((character, hero))
    return records


def _master_status_lines(
    checkpoint: CheckpointFile,
    envelope: OneStarAccountEnvelope,
) -> tuple[str, ...]:
    state = envelope.state
    occupied = sum(
        character.status != CharacterStatus.culled
        for character, _hero in _owned_hero_records(checkpoint, envelope)
    )
    lines = _management_lines(
        envelope,
        include_active_feed=True,
        include_stored_equipment=True,
        canonical_now_s=checkpoint.session.leading_at_s,
    )
    lines.append(f"Occupied Hero slots: {occupied}/{state.capacity}")
    lines.extend(_mission_lines(state.active_mission))
    lines.extend(_pending_operation_lines(state.pending_operation))
    return tuple(lines)


def _master_hero_roster_lines(
    checkpoint: CheckpointFile,
    envelope: OneStarAccountEnvelope,
) -> tuple[str, ...]:
    records = _owned_hero_records(checkpoint, envelope)
    if not records:
        return ("Owned Heroes: none",)
    lines = ["Owned Heroes:"]
    for index, (character, hero) in enumerate(records, start=1):
        lifecycle = str(
            character.status.value
            if hasattr(character.status, "value")
            else character.status
        )
        exact = _exact_hero_lines(character.name, hero, envelope)
        lines.append(
            f"{index}. {exact[0]}; lifecycle {lifecycle}; "
            f"location {character.location or 'unknown'}"
        )
        lines.append(f"   {exact[1]}")
    lines.append("Use /master hero <name|id|#> for skills and equipment.")
    return tuple(lines)


def _resolve_owned_hero(
    records: list[tuple[CharacterRecord, OneStarHeroState]],
    hero_ref: str,
) -> tuple[CharacterRecord, OneStarHeroState]:
    token = hero_ref.strip()
    numbered = token.removeprefix("#")
    if numbered.isdigit():
        index = int(numbered)
        if 1 <= index <= len(records):
            return records[index - 1]
        raise ValueError(
            f"Hero number must be between 1 and {len(records)}."
        )
    folded = token.casefold()
    matches = [
        record
        for record in records
        if folded in {
            record[0].character_id.casefold(),
            record[0].name.strip().casefold(),
        }
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"More than one owned Hero is named {token!r}; use an id or #."
        )
    raise ValueError(f"No owned Hero matches {token!r}.")


def _master_hero_lines(
    checkpoint: CheckpointFile,
    envelope: OneStarAccountEnvelope,
    hero_ref: str,
) -> tuple[str, ...]:
    records = _owned_hero_records(checkpoint, envelope)
    if not records:
        raise ValueError("This System account does not own any Heroes.")
    if not hero_ref.strip():
        raise ValueError("Choose a Hero by name, id, or roster number.")
    character, hero = _resolve_owned_hero(records, hero_ref)
    lifecycle = str(
        character.status.value
        if hasattr(character.status, "value")
        else character.status
    )
    lines = [f"{character.name} [{character.character_id}]"]
    lines.extend(_exact_hero_lines(character.name, hero, envelope))
    lines.append(
        f"Lifecycle: {lifecycle}; location "
        f"{character.location or 'unknown'}"
    )
    if hero.terminal_cause:
        lines.append(f"Terminal cause: {hero.terminal_cause}")
    return tuple(lines)


def _has_exact_own_sheet(
    character_id: str,
    hero: OneStarHeroState,
    envelope: OneStarAccountEnvelope,
) -> bool:
    research_key = envelope.config.hero_system_visibility_research_key
    research_level = (
        envelope.state.research_levels.get(research_key, 0)
        if research_key
        else 0
    )
    return bool(
        hero.innate_system_sight
        or character_id in envelope.state.system_observer_ids
        or research_level > 0
    )


def one_star_agent_state_block(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
) -> str:
    """Return dynamic user-tail mechanics visible to one character agent."""
    loaded = _account(checkpoint)
    if loaded is None:
        return ""
    owner, envelope = loaded
    state = envelope.state
    lines: list[str] = []
    if character.character_id == owner.character_id:
        lines = ["## Current System Account State"]
        lines.extend(_management_lines(
            envelope,
            include_active_feed=True,
            include_stored_equipment=True,
            canonical_now_s=checkpoint.session.leading_at_s,
        ))
        lines.extend(_mission_lines(state.active_mission))
        lines.extend(_public_roster_lines(checkpoint, envelope))
        lines.extend(_pending_operation_lines(state.pending_operation))
    elif character.character_id in state.guide_character_ids:
        lines = [
            "## Authored System Channel",
            "Lobby management and tutorial state:",
        ]
        lines.extend(_management_lines(
            envelope,
            include_active_feed=False,
            canonical_now_s=checkpoint.session.leading_at_s,
        ))
        lines.extend(_pending_operation_lines(state.pending_operation))
        lines.append(
            "Tutorial deliveries: " + _join_values(
                f"{tutorial}: {_join_values(recipients)}"
                for tutorial, recipients
                in sorted(state.tutorial_deliveries.items())
            )
        )
    else:
        hero = load_one_star_hero(character)
        if hero is None:
            return ""
        exact = (
            _has_exact_own_sheet(character.character_id, hero, envelope)
            if _belongs_to_account(hero, envelope)
            else hero.innate_system_sight
        )
        lines = ["## Your Current Mechanics"]
        lines.extend(_own_hero_lines(
            character,
            hero,
            envelope,
            exact=exact,
        ))
    return "\n".join(lines)


def one_star_status_lines(
    checkpoint: CheckpointFile,
    viewpoint_character_id: str,
) -> tuple[str, ...]:
    """Return shared CLI/Discord status lines for the selected fictional POV."""
    if (
        _ruleset_id(checkpoint) != ONE_STAR_RULESET_ID
        or not viewpoint_character_id
    ):
        return ()
    character = next(
        (
            candidate
            for candidate in checkpoint.characters
            if candidate.character_id == viewpoint_character_id
        ),
        None,
    )
    if character is None:
        return ()
    block = one_star_agent_state_block(checkpoint, character)
    if not block:
        return ()
    return tuple(
        line[3:] + ":" if line.startswith("## ") else line
        for line in block.splitlines()
    )


def one_star_master_command_lines(
    checkpoint: CheckpointFile,
    viewpoint_character_id: str,
    command: str,
    *,
    hero_ref: str = "",
) -> tuple[str, ...]:
    """Project one read-only Master command from the account owner's POV."""

    loaded = _account(checkpoint)
    if loaded is None:
        raise ValueError(
            "Master commands are available only in a One-Star Ascension "
            "session."
        )
    owner, envelope = loaded
    if viewpoint_character_id != owner.character_id:
        raise ValueError(
            "Master commands are available only to the character that owns "
            "this System account."
        )
    normalized = command.strip().casefold()
    if normalized == "status":
        return _master_status_lines(checkpoint, envelope)
    if normalized == "heroes":
        return _master_hero_roster_lines(checkpoint, envelope)
    if normalized == "hero":
        return _master_hero_lines(
            checkpoint,
            envelope,
            hero_ref,
        )
    raise ValueError("Master command must be status, heroes, or hero.")


def one_star_synthesis_command_intention(
    checkpoint: CheckpointFile,
    viewpoint_character_id: str,
    *,
    target_ref: str,
    source_refs: Sequence[str],
) -> str:
    """Build one exact Master submission without mutating adapter state.

    The returned text goes through the ordinary router, character-response,
    and narrator path.  This helper resolves user-facing roster references and
    rejects impossible command preconditions; it does not open or resolve the
    durable synthesis operation itself.
    """

    loaded = _account(checkpoint)
    if loaded is None:
        raise ValueError(
            "Synthesis commands are available only in a One-Star Ascension "
            "session."
        )
    owner, envelope = loaded
    if viewpoint_character_id != owner.character_id:
        raise ValueError(
            "Synthesis commands are available only to the character that "
            "owns this System account."
        )
    if envelope.state.active_mission is not None:
        raise ValueError(
            "Synthesis cannot be selected while a Tower mission is active."
        )
    if envelope.state.pending_operation is not None:
        raise ValueError(
            "Finish or cancel the current pending management operation "
            "before selecting synthesis."
        )

    requirement = envelope.config.operation_requirements.get("synthesis")
    if (
        requirement is None
        or not requirement.facility_id
        or not requirement.required_location
        or envelope.state.facilities.get(requirement.facility_id, 0) < 1
    ):
        raise ValueError("The Synthesis Chamber is not operational.")

    records = _owned_hero_records(checkpoint, envelope)
    if not records:
        raise ValueError("This System account does not own any Heroes.")
    if not target_ref.strip():
        raise ValueError("Choose the surviving target Hero.")
    clean_source_refs = [value.strip() for value in source_refs if value.strip()]
    if not clean_source_refs:
        raise ValueError("Choose at least one source Hero to consume.")

    target_character, _target = _resolve_owned_hero(records, target_ref)
    resolved_sources = [
        _resolve_owned_hero(records, source_ref)
        for source_ref in clean_source_refs
    ]
    source_characters = [character for character, _hero in resolved_sources]
    source_ids = [character.character_id for character in source_characters]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Each synthesis source Hero may be selected only once.")
    if target_character.character_id in source_ids:
        raise ValueError("The surviving target Hero cannot also be a source.")

    for character in [target_character, *source_characters]:
        if character.status != CharacterStatus.active:
            raise ValueError(
                f"{character.name} is not an active Hero and cannot be "
                "selected for synthesis."
            )

    characters = {
        character.character_id: character for character in checkpoint.characters
    }
    guide_characters: list[CharacterRecord] = []
    for guide_id in envelope.state.guide_character_ids:
        guide = characters.get(guide_id)
        if guide is None or guide.status != CharacterStatus.active:
            raise ValueError(
                "The configured lobby guide is unavailable to carry out "
                "synthesis."
            )
        guide_characters.append(guide)

    target_label = (
        f"{target_character.name} [{target_character.character_id}]"
    )
    source_labels = ", ".join(
        f"{character.name} [{character.character_id}]"
        for character in source_characters
    )
    guide_labels = _join_values(
        (
            f"{character.name} [{character.character_id}]"
            for character in guide_characters
        ),
        empty="the lobby warden",
    )
    return (
        "Through the System interface, I select "
        f"{source_labels} as synthesis source"
        f"{'s' if len(source_characters) != 1 else ''} for the surviving "
        f"target {target_label}, at {requirement.required_location}. "
        "The System openly notifies every selected Hero and "
        f"{guide_labels}. This opens the selection only: it does not assert "
        "consent, movement into the chamber, or completed synthesis. Each "
        "selected Hero gets to cooperate, refuse, bargain, flee, or resist. "
        f"{guide_labels} must answer in their own voice as warden; any "
        "physical enforcement is the warden's action, never consent invented "
        "for a Hero."
    )


def visible_equipped_item_description(character: CharacterRecord) -> str:
    """Describe only visible current equipment for image-direction callers."""
    hero = load_one_star_hero(character)
    if hero is None:
        return ""
    return _join_values(
        (
            f"{item.name} worn or carried in the {item.slot} slot"
            for item in _visible_equipment(hero)
        ),
        empty="",
    )
