"""Read-only One-Star router context projections.

The router needs one immutable rules packet in its cached prefix and one full
current ledger packet in its per-turn tail.  This is deliberately separate
from player/character projections: the router is the adjudicator, not a
fictional viewpoint.
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


def _render_ids(values: Iterable[str]) -> str:
    rendered = [value for value in values if value]
    return ",".join(rendered) if rendered else "none"


def render_one_star_router_static_config(checkpoint: CheckpointFile) -> str:
    """Render immutable seed rules for the router's cached system prefix."""
    if not is_one_star_checkpoint(checkpoint):
        return ""

    _owner, account = load_one_star_account(checkpoint)
    config = account.config
    lines = [
        "<one_star_rules_config>",
        f"lobby: id={config.lobby_id}; location={config.lobby_location_label}; "
        f"starting_floor={config.starting_lobby_floor}; "
        f"starting_capacity={config.starting_capacity}",
        f"starting_resources: {_render_resources(config.starting_resources)}",
        f"summon_batch_max: {config.max_summon_batch}",
        "summon_pools:",
    ]
    for pool_id, pool in sorted(config.summon_pools.items()):
        lines.append(
            f"- id={pool_id}; cost[{_render_resources(pool.cost)}]; "
            f"birth_stars={pool.minimum_birth_stars}-{pool.maximum_birth_stars}; "
            f"usage={pool.usage}; "
            f"eligible_existing_ids={_render_ids(pool.eligible_existing_ids)}; "
            f"fresh_generation_allowed={str(pool.fresh_generation_allowed).lower()}"
        )

    lines.append("catalogue:")
    for catalogue_id, entry in sorted(config.catalogue.items()):
        effects = [
            f"kind={entry.kind}",
            f"inventory_item_id={entry.inventory_item_id or 'none'}",
            f"facility_id={entry.facility_id or 'none'}",
            f"target_level={entry.target_level}",
            f"resulting_lobby_floor={entry.resulting_lobby_floor}",
            f"resulting_capacity={entry.resulting_capacity}",
            f"research_key={entry.research_key or 'none'}",
            f"research_level={entry.research_level}",
        ]
        prerequisites = [
            f"cleared_tower_floor>={entry.required_cleared_floor}",
            f"lobby_floor>={entry.required_lobby_floor}",
        ]
        lines.append(
            f"- id={catalogue_id}; cost[{_render_resources(entry.cost)}]; "
            f"prerequisites[{'; '.join(prerequisites)}]; "
            f"effects[{'; '.join(effects)}]"
        )

    lines.append(
        "star_level_caps: " + ", ".join(
            f"{stars}={cap}" for stars, cap in sorted(config.star_level_caps.items())
        )
    )
    constraints = config.hero_constraints
    lines.append(
        "hero_constraints: "
        f"hp_max={constraints.minimum_hp_max}-{constraints.maximum_hp_max}; "
        f"experience_points<={constraints.maximum_xp}; "
        f"abs(stat_value)<={constraints.maximum_stat_value}; "
        f"equipment_entries<={constraints.maximum_equipment_entries}; "
        f"skill_entries<={constraints.maximum_skill_entries}"
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
    lines.append("first_clear_floor_rewards:")
    for floor, reward in sorted(config.floor_rewards.items()):
        lines.append(f"- floor={floor}; reward[{_render_resources(reward)}]")
    lines.append(
        f"repeat_clear_gold: first_clear_gold*{config.repeat_gold_numerator}/"
        f"{config.repeat_gold_denominator}; minimum={config.repeat_gold_minimum}; "
        "no repeated non-Gold floor rewards"
    )
    lines.append(f"promotion_cost: {_render_resources(config.promotion_cost)}")
    lines.append(f"lobby_return_healing: {str(config.lobby_return_healing).lower()}")
    lines.append(
        "hero_system_visibility_research_key: "
        f"{config.hero_system_visibility_research_key or 'none'}"
    )
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
        f"terminal_cause={hero.terminal_cause or 'none'}; "
        f"hidden_capabilities={hero.hidden_capabilities}; "
        f"private_potential={hero.private_potential or 'none'}",
    ]
    return lines


def render_one_star_router_ledger(checkpoint: CheckpointFile) -> str:
    """Render all current adapter state relevant to the next adjudication."""
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
        "system_observer_ids: " + _render_ids(state.system_observer_ids),
        "tutorial_deliveries: " + "; ".join(
            f"{key}={_render_ids(character_ids)}"
            for key, character_ids in sorted(state.tutorial_deliveries.items())
        ),
    ]
    if state.active_mission is None:
        lines.append("active_mission: none")
    else:
        mission = state.active_mission
        lines.extend([
            f"active_mission: id={mission.mission_id}; floor={mission.floor}; "
            f"destination={mission.destination}; "
            f"started_at_s={mission.started_at_s}; deadline_at_s={mission.deadline_at_s}",
            f"mission_completion={mission.completion_declaration}; "
            f"mission_failure={mission.failure_declaration}",
            "mission_party: " + _render_ids(mission.party_ids),
            "mission_formation: " + ", ".join(
                f"{character_id}={label}"
                for character_id, label in sorted(mission.formation_labels.items())
            ),
            "mission_counters: " + ", ".join(
                f"{counter_id}={counter.current}/{counter.target}"
                for counter_id, counter in sorted(mission.counters.items())
            ),
        ])
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

    lines.append("heroes:")
    eligible_reserve_ids = {
        character_id
        for pool in config.summon_pools.values()
        for character_id in pool.eligible_existing_ids
    }
    reserve_rows: list[str] = []
    for character in checkpoint.characters:
        if character.character_id not in eligible_reserve_ids:
            continue
        hero = load_one_star_hero(character)
        if hero is None or hero.owner_lobby_id:
            continue
        reserve_rows.append(
            f"id={character.character_id}; birth_stars={hero.birth_stars}; "
            f"status={character.status.value}"
        )
    lines.append("eligible_unowned_reserves: " + _render_ids(reserve_rows))
    found_hero = False
    for character in checkpoint.characters:
        hero = load_one_star_hero(character)
        if hero is None or hero.owner_lobby_id != config.lobby_id:
            continue
        found_hero = True
        lines.append(
            f"- id={character.character_id}; name={character.name}; "
            f"status={character.status.value}; location={character.location}"
        )
        lines.extend(f"  {line}" for line in _render_hero(hero))
    if not found_hero:
        lines.append("- none")
    lines.append("</one_star_current_ledger>")
    return "\n".join(lines)
