"""Replay-stable, adapter-owned One-Star Hero progression.

This module deliberately has no router or projection dependency.  It turns a
qualitative generated identity into durable numerical mechanics and applies
later XP with integer fixed-point arithmetic only.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import TYPE_CHECKING

from app.schemas.one_star import (
    OneStarEquipmentEntry,
    OneStarHeroState,
    OneStarRulesConfig,
)

if TYPE_CHECKING:
    from app.schemas.one_star_character_gen import AuthoredOneStarHeroMechanics


_MILLI = 1_000
_BASIS_POINTS = 10_000
_PROGRESSION_NAMESPACE = "ayoa.one_star.progression.v1"
_NON_IDENTIFIER = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class OneStarExperienceReport:
    """Visible numerical result of one deterministic XP application."""

    offered_xp: int
    applied_xp: int
    wasted_xp: int
    levels_gained: int
    stat_gains: dict[str, int]
    hp_max_gain: int


def _round_half_up(numerator: int, denominator: int) -> int:
    """Return a non-negative rational rounded half-up without floats."""

    if numerator < 0 or denominator <= 0:
        raise ValueError("fixed-point rounding requires a non-negative ratio")
    return (numerator * 2 + denominator) // (denominator * 2)


def _scaled_by_grade(value: int, grade: int, multiplier_milli: int) -> int:
    if value < 0 or grade < 1 or multiplier_milli < 1:
        raise ValueError("invalid fixed-point grade scaling input")
    exponent = grade - 1
    return _round_half_up(
        value * multiplier_milli**exponent,
        _MILLI**exponent,
    )


def scaled_by_grade(value: int, grade: int, config: OneStarRulesConfig) -> int:
    """Scale an integer by the configured grade multiplier with half-up rounding."""

    _validate_grade(grade, config, label="grade")
    return _scaled_by_grade(
        value,
        grade,
        config.progression.grade_multiplier_milli,
    )


def _slug(value: str, *, fallback: str) -> str:
    normalized = _NON_IDENTIFIER.sub("_", value.strip().lower()).strip("_")
    return normalized or fallback


def derive_progression_seed(*, character_id: str, birth_stars: int) -> str:
    """Return the immutable, cryptographically-derived stream seed for a Hero."""

    if birth_stars < 1:
        raise ValueError("birth stars must be positive")
    character_key = character_id.strip()
    if not character_key:
        raise ValueError("character id must be non-empty")
    material = (
        f"{_PROGRESSION_NAMESPACE}.seed\0{character_key}\0{birth_stars}"
    ).encode("utf-8")
    return sha256(material).hexdigest()


def _stable_int(*, seed: str, namespace: str, minimum: int, maximum: int) -> int:
    """Draw a stable inclusive integer from an explicitly namespaced digest."""

    if not seed or not namespace or maximum < minimum:
        raise ValueError("invalid deterministic draw bounds")
    width = maximum - minimum + 1
    material = f"{_PROGRESSION_NAMESPACE}.draw\0{namespace}\0{seed}".encode(
        "utf-8"
    )
    return minimum + int.from_bytes(sha256(material).digest(), "big") % width


def _validate_grade(grade: int, config: OneStarRulesConfig, *, label: str) -> None:
    if grade not in config.star_level_caps:
        raise ValueError(f"{label} has no configured One-Star grade")


def _validate_affinities(
    *, strong_stat_id: str, weak_stat_id: str, config: OneStarRulesConfig
) -> None:
    allowed = set(config.progression.stat_ids)
    if strong_stat_id not in allowed or weak_stat_id not in allowed:
        raise ValueError("Hero affinities must use configured progression stat ids")
    if strong_stat_id == weak_stat_id:
        raise ValueError("Hero strong and weak stat ids must differ")


def birth_stat_total_mean(birth_stars: int, config: OneStarRulesConfig) -> int:
    """Mean level-one total before the Hero's stable five-percent variation."""

    _validate_grade(birth_stars, config, label="birth stars")
    return scaled_by_grade(config.progression.birth_stat_total, birth_stars, config)


def birth_hp_mean(birth_stars: int, config: OneStarRulesConfig) -> int:
    """Mean level-one maximum HP before the Hero's stable variation."""

    _validate_grade(birth_stars, config, label="birth stars")
    return scaled_by_grade(config.progression.birth_hp_max, birth_stars, config)


def experience_to_reach_level(level: int, config: OneStarRulesConfig) -> int:
    """Cumulative XP required to have reached ``level``."""

    if level < 1:
        raise ValueError("level must be positive")
    return config.progression.xp_threshold_factor * level * (level - 1)


def _variation(mean: int, config: OneStarRulesConfig) -> int:
    if config.progression.variance_basis_points == 0:
        return 0
    return max(
        1,
        _round_half_up(
            mean * config.progression.variance_basis_points,
            _BASIS_POINTS,
        ),
    )


def _initial_stat_total(hero: OneStarHeroState, config: OneStarRulesConfig) -> int:
    mean = birth_stat_total_mean(hero.birth_stars, config)
    variation = _variation(mean, config)
    return mean + _stable_int(
        seed=hero.progression_seed,
        namespace="birth_stat_total",
        minimum=-variation,
        maximum=variation,
    )


def _initial_hp_max(hero: OneStarHeroState, config: OneStarRulesConfig) -> int:
    mean = birth_hp_mean(hero.birth_stars, config)
    variation = _variation(mean, config)
    return max(
        1,
        mean
        + _stable_int(
            seed=hero.progression_seed,
            namespace="birth_hp_max",
            minimum=-variation,
            maximum=variation,
        ),
    )


def _allocate_birth_stats(
    *, total: int, strong_stat_id: str, weak_stat_id: str, config: OneStarRulesConfig
) -> dict[str, int]:
    """Split a total across three stats with strict strong-high/weak-low order."""

    stat_ids = config.progression.stat_ids
    other_stat_id = next(
        stat_id
        for stat_id in stat_ids
        if stat_id not in {strong_stat_id, weak_stat_id}
    )
    if total < len(stat_ids) + 3:
        raise ValueError("birth stat total is too small to express affinities")
    base, remainder = divmod(total, len(stat_ids))
    values = {
        weak_stat_id: base - 1,
        other_stat_id: base,
        strong_stat_id: base + 1,
    }
    if remainder >= 1:
        values[strong_stat_id] += 1
    if remainder >= 2:
        values[other_stat_id] += 1
    return {stat_id: values[stat_id] for stat_id in stat_ids}


def _growth_milli(hero: OneStarHeroState, config: OneStarRulesConfig, *, base: int) -> int:
    _validate_grade(hero.potential_grade, config, label="potential grade")
    return _scaled_by_grade(
        base,
        hero.potential_grade,
        config.progression.grade_multiplier_milli,
    )


def _total_growth_points(level: int, *, growth_milli: int) -> int:
    return ((level - 1) * growth_milli) // _MILLI


def _apply_stat_growth(
    stats: dict[str, int], hero: OneStarHeroState, config: OneStarRulesConfig
) -> None:
    growth_points = _total_growth_points(
        hero.level,
        growth_milli=_growth_milli(
            hero,
            config,
            base=config.progression.stat_growth_per_level_milli,
        ),
    )
    other_stat_id = next(
        stat_id
        for stat_id in config.progression.stat_ids
        if stat_id not in {hero.strong_stat_id, hero.weak_stat_id}
    )
    # This repeating allocation preserves the authored affinity order while
    # retaining the exact fixed-point total growth over replay.
    allocation_order = (
        hero.strong_stat_id,
        other_stat_id,
        hero.strong_stat_id,
        hero.weak_stat_id,
    )
    full_cycles, remainder = divmod(growth_points, len(allocation_order))
    if full_cycles:
        stats[hero.strong_stat_id] += full_cycles * 2
        stats[other_stat_id] += full_cycles
        stats[hero.weak_stat_id] += full_cycles
    for stat_id in allocation_order[:remainder]:
        stats[stat_id] += 1


def normalise_equipment_item_id(
    *, character_id: str, item_id: str, occurrence: int = 1
) -> str:
    """Prefix an authored item id with its Hero id and resolve list duplicates."""

    if occurrence < 1:
        raise ValueError("equipment occurrence must be positive")
    character_prefix = _slug(character_id, fallback="hero")
    item_suffix = _slug(item_id, fallback="item")
    base = (
        item_suffix
        if item_suffix == character_prefix or item_suffix.startswith(character_prefix + "_")
        else f"{character_prefix}_{item_suffix}"
    )
    return base if occurrence == 1 else f"{base}_{occurrence}"


def normalise_generated_equipment(
    *, character_id: str, equipment: list[OneStarEquipmentEntry]
) -> list[OneStarEquipmentEntry]:
    """Return copies whose globally stable ids belong to ``character_id``."""

    occurrences: dict[str, int] = {}
    normalised: list[OneStarEquipmentEntry] = []
    for entry in equipment:
        base = normalise_equipment_item_id(
            character_id=character_id,
            item_id=entry.item_id,
        )
        occurrences[base] = occurrences.get(base, 0) + 1
        normalised.append(
            entry.model_copy(
                update={
                    "item_id": normalise_equipment_item_id(
                        character_id=character_id,
                        item_id=entry.item_id,
                        occurrence=occurrences[base],
                    )
                },
                deep=True,
            )
        )
    return normalised


def build_generated_hero(
    *,
    character_id: str,
    generated: AuthoredOneStarHeroMechanics,
    birth_stars: int,
    config: OneStarRulesConfig,
) -> OneStarHeroState:
    """Construct a full numeric Hero state from qualitative character output."""

    _validate_grade(birth_stars, config, label="birth stars")
    _validate_affinities(
        strong_stat_id=generated.strong_stat_id,
        weak_stat_id=generated.weak_stat_id,
        config=config,
    )
    seed = derive_progression_seed(
        character_id=character_id,
        birth_stars=birth_stars,
    )
    hero = OneStarHeroState(
        birth_stars=birth_stars,
        current_stars=birth_stars,
        level=1,
        experience_points=0,
        hp_current=1,
        hp_max=1,
        stats={config.progression.stat_ids[0]: 1},
        equipment=normalise_generated_equipment(
            character_id=character_id,
            equipment=generated.equipment,
        ),
        skills=[entry.model_copy(deep=True) for entry in generated.skills],
        conditions=list(generated.conditions),
        persistent_injuries=list(generated.persistent_injuries),
        innate_system_sight=generated.innate_system_sight,
        generated_for_summon=True,
        acquisition_event_id="",
        owner_lobby_id="",
        terminal_cause="",
        terminal_event_id="",
        hidden_capabilities={
            entry.capability_id: entry.description
            for entry in generated.hidden_capabilities
        },
        progression_seed=seed,
        strong_stat_id=generated.strong_stat_id,
        weak_stat_id=generated.weak_stat_id,
        potential_grade=birth_stars,
    )
    rebalance_hero(hero=hero, config=config, restore_full_hp=True)
    return hero


def rebalance_hero(
    *,
    hero: OneStarHeroState,
    config: OneStarRulesConfig,
    restore_full_hp: bool = False,
) -> OneStarHeroState:
    """Rebuild deterministic stats and HP from immutable Hero progression data."""

    _validate_grade(hero.birth_stars, config, label="birth stars")
    _validate_grade(hero.current_stars, config, label="current stars")
    _validate_grade(hero.potential_grade, config, label="potential grade")
    _validate_affinities(
        strong_stat_id=hero.strong_stat_id,
        weak_stat_id=hero.weak_stat_id,
        config=config,
    )
    level_cap = config.star_level_caps[hero.current_stars]
    if hero.level > level_cap:
        raise ValueError("Hero level exceeds its current-star cap")
    hero.experience_points = max(
        hero.experience_points,
        experience_to_reach_level(hero.level, config),
    )
    hero.experience_points = min(
        hero.experience_points,
        _maximum_experience_at_current_stars(hero, config),
    )
    stats = _allocate_birth_stats(
        total=_initial_stat_total(hero, config),
        strong_stat_id=hero.strong_stat_id,
        weak_stat_id=hero.weak_stat_id,
        config=config,
    )
    _apply_stat_growth(stats, hero, config)
    hp_max = _initial_hp_max(hero, config) + _total_growth_points(
        hero.level,
        growth_milli=_growth_milli(
            hero,
            config,
            base=config.progression.hp_growth_per_level_milli,
        ),
    )
    hp_deficit = max(0, hero.hp_max - hero.hp_current)
    hero.stats = stats
    hero.hp_max = hp_max
    hero.hp_current = hp_max if restore_full_hp else max(0, hp_max - hp_deficit)
    return hero


def _maximum_experience_at_current_stars(
    hero: OneStarHeroState, config: OneStarRulesConfig
) -> int:
    _validate_grade(hero.current_stars, config, label="current stars")
    cap = config.star_level_caps[hero.current_stars]
    return experience_to_reach_level(
        cap + config.progression.cap_bank_extra_levels,
        config,
    )


def _validate_starting_experience(
    *, hero: OneStarHeroState, config: OneStarRulesConfig, allow_banked_release: bool
) -> None:
    cap = config.star_level_caps[hero.current_stars]
    if hero.experience_points < experience_to_reach_level(hero.level, config):
        raise ValueError("Hero XP is below its reached level threshold")
    if (
        not allow_banked_release
        and hero.level < cap
        and hero.experience_points >= experience_to_reach_level(hero.level + 1, config)
    ):
        raise ValueError("Hero XP is sufficient for an unrealized next level")
    if hero.experience_points > _maximum_experience_at_current_stars(hero, config):
        raise ValueError("Hero XP exceeds its current-star capacity")


def banked_experience_at_current_cap(
    hero: OneStarHeroState, config: OneStarRulesConfig
) -> int:
    """Return retained XP above a reached current-star cap, without mutation."""

    cap = config.star_level_caps[hero.current_stars]
    if hero.level != cap:
        return 0
    cap_threshold = experience_to_reach_level(cap, config)
    return max(
        0,
        min(hero.experience_points, _maximum_experience_at_current_stars(hero, config))
        - cap_threshold,
    )


def remaining_experience_capacity(
    hero: OneStarHeroState, config: OneStarRulesConfig
) -> int:
    """Return how much additional XP this Hero can retain before promotion."""

    maximum = _maximum_experience_at_current_stars(hero, config)
    return max(0, maximum - hero.experience_points)


def _experience_report(
    *,
    offered_xp: int,
    before_level: int,
    before_experience: int,
    before_stats: dict[str, int],
    before_hp_max: int,
    hero: OneStarHeroState,
) -> OneStarExperienceReport:
    applied_xp = hero.experience_points - before_experience
    return OneStarExperienceReport(
        offered_xp=offered_xp,
        applied_xp=applied_xp,
        wasted_xp=offered_xp - applied_xp,
        levels_gained=hero.level - before_level,
        stat_gains={
            stat_id: value - before_stats.get(stat_id, 0)
            for stat_id, value in hero.stats.items()
            if value > before_stats.get(stat_id, 0)
        },
        hp_max_gain=hero.hp_max - before_hp_max,
    )


def preview_experience(
    *, hero: OneStarHeroState, experience_delta: int, config: OneStarRulesConfig
) -> OneStarExperienceReport:
    """Calculate an XP result without changing the supplied Hero."""

    preview = hero.model_copy(deep=True)
    return _apply_experience(
        hero=preview,
        experience_delta=experience_delta,
        config=config,
        allow_banked_release=False,
    )


def apply_experience(
    *, hero: OneStarHeroState, experience_delta: int, config: OneStarRulesConfig
) -> OneStarExperienceReport:
    """Apply non-negative cumulative XP, level growth, and current-cap banking."""

    return _apply_experience(
        hero=hero,
        experience_delta=experience_delta,
        config=config,
        allow_banked_release=False,
    )


def _apply_experience(
    *,
    hero: OneStarHeroState,
    experience_delta: int,
    config: OneStarRulesConfig,
    allow_banked_release: bool,
) -> OneStarExperienceReport:

    if experience_delta < 0:
        raise ValueError("experience delta cannot be negative")
    _validate_grade(hero.current_stars, config, label="current stars")
    cap = config.star_level_caps[hero.current_stars]
    if hero.level > cap:
        raise ValueError("Hero level exceeds its current-star cap")
    maximum_experience = _maximum_experience_at_current_stars(hero, config)
    _validate_starting_experience(
        hero=hero,
        config=config,
        allow_banked_release=allow_banked_release,
    )
    before_level = hero.level
    before_experience = hero.experience_points
    before_stats = dict(hero.stats)
    before_hp_max = hero.hp_max
    hero.experience_points += experience_delta
    while (
        hero.level < cap
        and hero.experience_points
        >= experience_to_reach_level(hero.level + 1, config)
    ):
        hero.level += 1
    if hero.level == cap:
        hero.experience_points = min(
            hero.experience_points,
            maximum_experience,
        )
    rebalance_hero(hero=hero, config=config)
    return _experience_report(
        offered_xp=experience_delta,
        before_level=before_level,
        before_experience=before_experience,
        before_stats=before_stats,
        before_hp_max=before_hp_max,
        hero=hero,
    )


def apply_promotion_banked_experience(
    *, hero: OneStarHeroState, config: OneStarRulesConfig
) -> OneStarExperienceReport:
    """Apply XP retained at the former cap after the caller promotes the Hero."""

    return _apply_experience(
        hero=hero,
        experience_delta=0,
        config=config,
        allow_banked_release=True,
    )
