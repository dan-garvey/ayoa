"""Minimal One-Star router configuration and failure evidence projections."""

from __future__ import annotations

from app.engine.one_star_adapter import (
    effective_one_star_stamina,
    is_one_star_checkpoint,
    load_one_star_account,
    load_one_star_combatant,
    load_one_star_hero,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import OneStarCost, OneStarStateUpdate


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
    combatants: list[str] = []
    for character in sorted(
        checkpoint.characters,
        key=lambda item: item.character_id,
    ):
        combatant = load_one_star_combatant(character)
        if combatant is None:
            continue
        stats = ",".join(
            f"{stat_id}={value}"
            for stat_id, value in sorted(combatant.stats.items())
        )
        combatants.append(
            f"- {character.character_id}: hp={combatant.hp_current}/"
            f"{combatant.hp_max}; stats[{stats}]"
        )
    if combatants:
        lines.append("non_hero_combat_authority:")
        lines.extend(combatants)
    lines.append("</one_star_rules_config>")
    return "\n".join(lines)


def render_one_star_repair_evidence(
    checkpoint: CheckpointFile,
    *,
    state_updates: list[OneStarStateUpdate],
    canonical_at_s: int | None = None,
) -> str:
    """Render only state rows implicated by a rejected compact update list."""
    if not is_one_star_checkpoint(checkpoint):
        return ""

    _owner, account = load_one_star_account(checkpoint)
    state = account.state
    config = account.config
    lines = ["<one_star_conflict_evidence>"]
    seen: set[str] = set()

    def add(line: str) -> None:
        if line not in seen:
            seen.add(line)
            lines.append(line)

    def add_resources() -> None:
        add(f"current_resources: {_render_resources(state.resources)}")

    def render_values(values: list[str]) -> str:
        return ",".join(value for value in values if value) or "none"

    def add_hero(
        character_id: str,
        detail_keys: set[str],
        *,
        include_progression: bool = False,
    ) -> None:
        character = next(
            (item for item in checkpoint.characters if item.character_id == character_id),
            None,
        )
        if character is None:
            add(f"hero {character_id}: nonexistent")
            return
        hero = load_one_star_hero(character)
        if hero is None:
            add(
                f"hero {character_id}: status={character.status.value}; "
                f"location={character.location}; no One-Star Hero state"
            )
            return
        add(
            f"hero {character_id}: status={character.status.value}; "
            f"location={character.location}; hp={hero.hp_current}/{hero.hp_max}"
        )
        if include_progression:
            from app.engine.one_star_progression import (
                banked_experience_at_current_cap,
                experience_to_reach_level,
            )

            current_cap = config.star_level_caps[hero.current_stars]
            if hero.level < current_cap:
                next_threshold = experience_to_reach_level(hero.level + 1, config)
                add(
                    f"hero {character_id} progression: level={hero.level}/{current_cap}; "
                    f"xp={hero.experience_points}/{next_threshold}"
                )
            else:
                cap_threshold = experience_to_reach_level(current_cap, config)
                banked = banked_experience_at_current_cap(hero, config)
                add(
                    f"hero {character_id} progression: level={hero.level}/{current_cap}; "
                    f"cap_threshold={cap_threshold}; cap_banked_xp={banked}"
                )
        if any(key.startswith(("condition", "persistent_injury")) for key in detail_keys):
            add(
                f"hero {character_id} conditions: conditions={render_values(hero.conditions)}; "
                f"injuries={render_values(hero.persistent_injuries)}"
            )
        if any("equipment" in key or key.startswith("durability.") for key in detail_keys):
            add(
                f"hero {character_id} equipment: "
                + "; ".join(
                    f"{item.item_id} qty={item.quantity} durability="
                    f"{item.durability_current}/{item.durability_max}"
                    for item in hero.equipment
                )
            )
        if any("skill" in key for key in detail_keys):
            add(
                f"hero {character_id} skills: "
                + "; ".join(
                    f"{skill.skill_id} rank={skill.rank}" for skill in hero.skills
                )
            )

    def add_equipment_holder(item_id: str) -> None:
        matches: list[tuple[str, object]] = [
            ("account", item)
            for item in getattr(state, "stored_equipment", ())
            if item.item_id == item_id
        ]
        for character in checkpoint.characters:
            hero = load_one_star_hero(character)
            if hero is None:
                continue
            matches.extend(
                (character.character_id, item)
                for item in hero.equipment
                if item.item_id == item_id
            )
        if not matches:
            add(f"equipment {item_id}: nonexistent")
            return
        if len(matches) != 1:
            add(f"equipment {item_id}: holder is ambiguous")
            return
        holder, item = matches[0]
        add(
            f"equipment {item_id}: holder={holder}; name={item.name}; "
            f"slot={item.slot}; quantity={item.quantity}; durability="
            f"{item.durability_current}/{item.durability_max}"
        )

    for update in state_updates:
        detail_keys = {
            detail.partition("=")[0].strip()
            for detail in update.details
            if detail.partition("=")[0].strip()
        }
        if update.kind == "catalogue_apply":
            add_resources()
            entry = config.catalogue.get(update.target_id)
            if entry is None:
                add(f"catalogue {update.target_id}: nonexistent")
            else:
                add(
                    f"catalogue {update.target_id}: cost="
                    f"{_render_nonzero_resources(entry.cost)}; kind={entry.kind}; "
                    f"current_lobby_floor={state.lobby_floor}; "
                    f"highest_cleared_floor={state.highest_cleared_floor}; "
                    f"facility_level={state.facilities.get(entry.facility_id, 0)}; "
                    f"research_level={state.research_levels.get(entry.research_key, 0)}; "
                    f"inventory={state.inventory.get(entry.inventory_item_id, 0)}"
                )
        elif update.kind == "summon":
            add_resources()
            pool = config.summon_pools.get(update.target_id)
            if pool is None:
                add(f"summon_pool {update.target_id}: nonexistent")
            else:
                owned_count = sum(
                    1
                    for character in checkpoint.characters
                    if (hero := load_one_star_hero(character)) is not None
                    and hero.owner_lobby_id == config.lobby_id
                    and character.status.value != "culled"
                )
                add(
                    f"summon_pool {update.target_id}: cost_per_pull="
                    f"{_render_nonzero_resources(pool.cost)}; "
                    f"maximum_batch={config.max_summon_batch}; "
                    f"occupied={owned_count}/{state.capacity}"
                )
        elif update.kind == "inventory_delta":
            if update.target_id in {
                "gold",
                "gems",
                "building_resources",
            }:
                add_resources()
            else:
                add(
                    f"inventory {update.target_id}: "
                    f"current={state.inventory.get(update.target_id, 0)}"
                )
        elif update.kind == "equipment_move":
            add_equipment_holder(update.target_id)
        elif update.kind == "hero_delta":
            add_hero(update.target_id, detail_keys)
        elif update.kind.startswith("mission_"):
            mission = state.active_mission
            stamina_at_s = max(
                checkpoint.session.leading_at_s,
                state.stamina_recovery_anchor_s,
                canonical_at_s or 0,
            )
            effective_stamina, _effective_anchor = effective_one_star_stamina(
                state,
                config,
                stamina_at_s,
            )
            add(
                "active_mission: none"
                if mission is None
                else (
                    f"active_mission: id={mission.mission_id}; floor={mission.floor}; "
                    f"party={','.join(mission.party_ids)}; destination={mission.destination}; "
                    "counters=" + ",".join(
                        f"{counter.counter_id}={counter.current}/{counter.target}"
                        for counter in mission.counters
                    )
                )
            )
            add(
                f"mission_account: stamina={effective_stamina}; "
                f"highest_unlocked={state.highest_unlocked_floor}; "
                f"highest_cleared={state.highest_cleared_floor}"
            )
        elif update.kind.startswith("pending_"):
            pending = state.pending_operation
            add(
                "pending_operation: none"
                if pending is None
                else (
                    f"pending_operation: id={pending.operation_id}; kind={pending.kind}; "
                    f"participants={','.join(pending.participant_ids)}; "
                    f"target={pending.target_id or 'none'}; "
                    f"destination={pending.destination or 'none'}"
                )
            )
            detail_values = _detail_values(update)
            implicated_ids = [
                *detail_values.get("participant", []),
                *detail_values.get("target_id", []),
            ]
            if pending is not None and update.target_id == pending.operation_id:
                implicated_ids.extend(pending.participant_ids)
                if pending.target_id:
                    implicated_ids.append(pending.target_id)
                preview = getattr(pending, "synthesis_preview", None)
                if preview is not None:
                    add(
                        "synthesis_preview: "
                        f"offered_xp={preview.offered_xp}; "
                        f"applied_xp={preview.applied_xp}; "
                        f"wasted_xp={preview.wasted_xp}; "
                        "returned_equipment="
                        f"{','.join(item.item_id for item in preview.returned_equipment) or 'none'}; "
                        "skill_transfer_chance="
                        f"{preview.skill_transfer_chance_basis_points / 100:g}%"
                    )
            for character_id in implicated_ids:
                add_hero(
                    character_id,
                    set(),
                    include_progression=bool(
                        pending is not None
                        and pending.kind in {"synthesis", "promotion"}
                        and character_id == pending.target_id
                    ),
                )
            if update.value == "promotion":
                add_resources()
                add(f"promotion_cost: {_render_nonzero_resources(config.promotion_cost)}")
        elif update.kind == "tutorial_delivery":
            add(
                f"tutorial {update.target_id}: already_delivered_to="
                f"{','.join(state.tutorial_deliveries.get(update.target_id, [])) or 'none'}"
            )
        elif update.kind == "active_feed":
            add(f"active_master_feed_id={state.active_master_feed_id or 'none'}")
            if update.target_id:
                add_hero(update.target_id, set())

    lines.append("</one_star_conflict_evidence>")
    return "\n".join(lines)


def _detail_values(update: OneStarStateUpdate) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for detail in update.details:
        key, separator, value = detail.partition("=")
        if separator and key.strip():
            values.setdefault(key.strip(), []).append(value.strip())
    return values
