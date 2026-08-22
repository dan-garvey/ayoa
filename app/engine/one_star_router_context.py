"""Minimal read-only One-Star router context projections.

The cached packet contains reviewed configuration; the volatile packet contains
current balances, operation state, compact roster mechanics, and full sheets
only for scene-relevant Heroes. Adapter-private draw sources, fingerprints, and
hidden potential never enter model context.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.engine.one_star_adapter import (
    effective_one_star_stamina,
    is_one_star_checkpoint,
    load_one_star_account,
    load_one_star_hero,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import OneStarCost, OneStarHeroState


def _render_resources(resources: OneStarCost) -> str:
    pieces = [
        f"gold={resources.gold}",
        f"gems={resources.gems}",
        f"building_resources={resources.building_resources}",
    ]
    pieces.extend(
        f"{material}={amount}"
        for material, amount in sorted(resources.materials.items())
    )
    return ", ".join(pieces)


def _render_nonzero_resources(resources: OneStarCost) -> str:
    pieces = []
    if resources.gold:
        pieces.append(f"gold={resources.gold}")
    if resources.gems:
        pieces.append(f"gems={resources.gems}")
    if resources.building_resources:
        pieces.append(f"building_resources={resources.building_resources}")
    pieces.extend(
        f"{material}={amount}"
        for material, amount in sorted(resources.materials.items())
        if amount
    )
    return ",".join(pieces) if pieces else "free"


def _render_ids(values: Iterable[str]) -> str:
    rendered = [value for value in values if value]
    return ",".join(rendered) if rendered else "none"


def _render_weight_percent(weight: int) -> str:
    whole, fractional = divmod(weight, 100)
    if not fractional:
        return f"{whole}%"
    return f"{whole}.{fractional:02d}".rstrip("0") + "%"


def _render_star_weights(weights: dict[int, int]) -> str:
    return ",".join(
        f"{birth_stars}={_render_weight_percent(weight)}"
        for birth_stars, weight in sorted(weights.items())
    )


def render_one_star_router_static_config(checkpoint: CheckpointFile) -> str:
    """Render immutable seed rules for the router's cached system prefix."""
    if not is_one_star_checkpoint(checkpoint):
        return ""

    _owner, account = load_one_star_account(checkpoint)
    config = account.config
    lines = [
        "<one_star_rules_config>",
        f"lobby={config.lobby_id}; location={config.lobby_location_label}",
        f"summon_batch_max={config.max_summon_batch}",
        "summon_pools:",
    ]
    for pool_id, pool in sorted(config.summon_pools.items()):
        lines.append(
            f"- {pool_id}: cost[{_render_nonzero_resources(pool.cost)}]; "
            f"stars={pool.minimum_birth_stars}-{pool.maximum_birth_stars}; "
            f"rates[{_render_star_weights(pool.star_weights)}]; usage={pool.usage}"
        )

    lines.append("catalogue:")
    for catalogue_id, entry in sorted(config.catalogue.items()):
        effects = [f"kind={entry.kind}"]
        for label, value in (
            ("item", entry.inventory_item_id),
            ("facility", entry.facility_id),
            ("research", entry.research_key),
        ):
            if value:
                effects.append(f"{label}={value}")
        for label, value in (
            ("level", entry.target_level),
            ("lobby_floor", entry.resulting_lobby_floor),
            ("capacity", entry.resulting_capacity),
            ("research_level", entry.research_level),
        ):
            if value:
                effects.append(f"{label}={value}")
        prerequisites = []
        if entry.required_cleared_floor:
            prerequisites.append(f"cleared>={entry.required_cleared_floor}")
        if entry.required_lobby_floor:
            prerequisites.append(f"lobby>={entry.required_lobby_floor}")
        lines.append(
            f"- {catalogue_id}: cost[{_render_nonzero_resources(entry.cost)}]; "
            + (f"requires[{','.join(prerequisites)}]; " if prerequisites else "")
            + f"effect[{','.join(effects)}]"
        )

    lines.append(
        "star_level_caps: " + ", ".join(
            f"{stars}={cap}" for stars, cap in sorted(config.star_level_caps.items())
        )
    )
    constraints = config.hero_constraints
    lines.append(
        "hero_bounds: "
        f"hp_max={constraints.minimum_hp_max}-{constraints.maximum_hp_max}; "
        f"xp<={constraints.maximum_xp}; "
        f"abs_stat<={constraints.maximum_stat_value}; "
        f"equipment<={constraints.maximum_equipment_entries}; "
        f"skills<={constraints.maximum_skill_entries}"
    )
    lines.append(
        f"deployment: stamina_cost={config.deployment_stamina_cost}; "
        f"stamina_max={config.maximum_stamina}; "
        f"stamina_recovery_seconds={config.stamina_recovery_seconds}"
    )
    lines.append("embodied_operation_requirements:")
    for operation_kind, requirement in sorted(config.operation_requirements.items()):
        lines.append(
            f"- kind={operation_kind}; facility_id={requirement.facility_id}; "
            f"required_location={requirement.required_location or 'none'}"
        )
    lines.append(
        f"repeat_clear_gold: first_clear_gold*{config.repeat_gold_numerator}/"
        f"{config.repeat_gold_denominator}; minimum={config.repeat_gold_minimum}; "
        "no repeated non-Gold floor rewards"
    )
    lines.append(
        f"promotion_cost: {_render_nonzero_resources(config.promotion_cost)}"
    )
    lines.append(f"lobby_return_healing: {str(config.lobby_return_healing).lower()}")
    lines.append("</one_star_rules_config>")
    return "\n".join(lines)


def _render_hero(hero: OneStarHeroState) -> list[str]:
    lines = [
        f"birth_stars={hero.birth_stars}; current_stars={hero.current_stars}; "
        f"level={hero.level}; experience_points={hero.experience_points}; "
        f"hp={hero.hp_current}/{hero.hp_max}; "
        f"innate_system_sight={str(hero.innate_system_sight).lower()}",
        "stats: " + ", ".join(
            f"{key}={value}" for key, value in sorted(hero.stats.items())
        ),
        "equipment: " + "; ".join(
            f"id={item.item_id},name={item.name},slot={item.slot},"
            f"quantity={item.quantity},durability={item.durability_current}/"
            f"{item.durability_max},tags={_render_ids(item.tags)},"
            f"visible={str(item.visible).lower()}"
            for item in hero.equipment
        ),
        "skills: " + "; ".join(
            f"id={skill.skill_id},name={skill.name},rank={skill.rank},"
            f"capability={skill.capability or 'none'},tags={_render_ids(skill.tags)},"
            f"visible={str(skill.visible).lower()}"
            for skill in hero.skills
        ),
        f"conditions={_render_ids(hero.conditions)}; "
        f"persistent_injuries={_render_ids(hero.persistent_injuries)}; "
        f"terminal_cause={hero.terminal_cause or 'none'}",
    ]
    return lines


def render_one_star_router_ledger(
    checkpoint: CheckpointFile,
    *,
    acting_character_id: str = "",
) -> str:
    """Render only current adapter state relevant to the next adjudication."""
    if not is_one_star_checkpoint(checkpoint):
        return ""

    owner, account = load_one_star_account(checkpoint)
    state = account.state
    config = account.config
    effective_stamina, effective_anchor = effective_one_star_stamina(
        state,
        config,
        checkpoint.session.leading_at_s,
    )
    lines = [
        "<one_star_current_ledger>",
        f"account_owner_id={owner.character_id}",
        f"canonical_clock_s={checkpoint.session.leading_at_s}",
        f"resources: {_render_resources(state.resources)}",
        "inventory: " + ", ".join(
            f"{item_id}={quantity}" for item_id, quantity in sorted(state.inventory.items())
        ),
        "facilities: " + ", ".join(
            f"{facility_id}=L{level}"
            for facility_id, level in sorted(state.facilities.items())
        ),
        "research_levels: " + ", ".join(
            f"{key}={level}" for key, level in sorted(state.research_levels.items())
        ),
        f"lobby_progression: floor={state.lobby_floor}; capacity={state.capacity}; "
        f"highest_unlocked_tower_floor={state.highest_unlocked_floor}; "
        f"highest_cleared_tower_floor={state.highest_cleared_floor}",
        f"stamina: current={effective_stamina}; "
        f"recovery_anchor_s={effective_anchor}",
        f"active_master_feed_id={state.active_master_feed_id or 'none'}",
        "guide_character_ids: " + _render_ids(state.guide_character_ids),
        "tutorial_deliveries: " + "; ".join(
            f"{key}={_render_ids(character_ids)}"
            for key, character_ids in sorted(state.tutorial_deliveries.items())
        ),
    ]
    if state.active_mission is None:
        lines.append("active_mission: none")
        floor = state.highest_unlocked_floor
    else:
        mission = state.active_mission
        floor = mission.floor
        lines.extend([
            f"active_mission: id={mission.mission_id}; floor={mission.floor}; "
            f"destination={mission.destination}; "
            f"started_at_s={mission.started_at_s}; deadline_at_s={mission.deadline_at_s}",
            f"mission_completion={mission.completion_declaration}; "
            f"mission_failure={mission.failure_declaration}",
            "mission_party: " + _render_ids(mission.party_ids),
            "mission_formation: " + ", ".join(
                f"{entry.character_id}={entry.label}"
                for entry in sorted(
                    mission.formation_labels,
                    key=lambda formation: formation.character_id,
                )
            ),
            "mission_counters: " + ", ".join(
                f"{counter.counter_id}={counter.current}/{counter.target}"
                for counter in sorted(
                    mission.counters,
                    key=lambda entry: entry.counter_id,
                )
            ),
        ])
    reward = config.floor_rewards.get(floor)
    if reward is not None:
        lines.append(
            f"floor_{floor}_first_clear_reward: "
            f"{_render_nonzero_resources(reward)}"
        )
    if state.pending_operation is None:
        lines.append("pending_operation: none")
    else:
        pending = state.pending_operation
        lines.append(
            f"pending_operation: id={pending.operation_id}; kind={pending.kind}; "
            f"participants={_render_ids(pending.participant_ids)}; "
            f"target={pending.target_id or 'none'}; "
            f"destination={pending.destination or 'none'}; "
            f"opened_at_s={pending.opened_at_s}"
        )

    local_heroes: list[tuple[object, OneStarHeroState]] = []
    for character in checkpoint.characters:
        hero = load_one_star_hero(character)
        if hero is None or hero.owner_lobby_id != config.lobby_id:
            continue
        local_heroes.append((character, hero))
    lines.append("hero_summaries:")
    if not local_heroes:
        lines.append("- none")
    for character, hero in local_heroes:
        lines.append(
            f"- id={character.character_id}; name={character.name}; "
            f"status={character.status.value}; location={character.location}; "
            f"stars={hero.current_stars}; level={hero.level}; "
            f"xp={hero.experience_points}; hp={hero.hp_current}/{hero.hp_max}; "
            f"conditions={_render_ids(hero.conditions)}; "
            f"equipment={_render_ids(item.item_id for item in hero.equipment)}; "
            f"skills={_render_ids(skill.skill_id for skill in hero.skills)}"
        )

    relevant_ids = {acting_character_id, state.active_master_feed_id}
    if state.active_mission is not None:
        relevant_ids.update(state.active_mission.party_ids)
    if state.pending_operation is not None:
        relevant_ids.update(state.pending_operation.participant_ids)
        relevant_ids.add(state.pending_operation.target_id)
    acting_character = next(
        (
            character for character in checkpoint.characters
            if character.character_id == acting_character_id
        ),
        None,
    )
    if acting_character is not None and acting_character_id != owner.character_id:
        relevant_ids.update(
            character.character_id
            for character, _hero in local_heroes
            if character.location == acting_character.location
        )

    detailed = [
        (character, hero)
        for character, hero in local_heroes
        if character.character_id in relevant_ids
    ]
    if detailed:
        lines.append("scene_relevant_hero_details:")
    for character, hero in detailed:
        lines.append(f"- id={character.character_id}")
        lines.extend(f"  {line}" for line in _render_hero(hero))
    lines.append("</one_star_current_ledger>")
    return "\n".join(lines)
