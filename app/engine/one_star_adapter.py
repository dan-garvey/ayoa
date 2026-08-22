"""Pure ledger preparation for the opt-in One-Star Ascension adapter.

The router still arbitrates fiction. This module translates its one compact
state-update list into private typed bookkeeping, validates it, and applies it
atomically. Standard weighted summons are resolved here without exposing future
draws to the router. The adapter deliberately has no combat resolver, stat
formula, XP curve, story id, or facility/economy constants.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping

from pydantic import ValidationError

from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import SpawnRequest, WakeSignal
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_GACHA_WEIGHT_TOTAL,
    ONE_STAR_HERO_KEY,
    ONE_STAR_RULESET_ID,
    OneStarAccountEnvelope,
    OneStarAccountState,
    OneStarActiveFeedOperation,
    OneStarCatalogueApplyOperation,
    OneStarCost,
    OneStarEquipmentEntry,
    OneStarFormationEntry,
    OneStarHeroDeltaOperation,
    OneStarHeroState,
    OneStarInventoryDeltaOperation,
    OneStarMissionCounter,
    OneStarMissionEndOperation,
    OneStarMissionStartOperation,
    OneStarMissionUpdateOperation,
    OneStarOperation,
    OneStarPendingOperation,
    OneStarPendingCancelOperation,
    OneStarPendingOpenOperation,
    OneStarPendingResolveOperation,
    OneStarRulesConfig,
    OneStarSkillEntry,
    OneStarSkillRankUpdate,
    OneStarStateUpdate,
    OneStarStatDelta,
    OneStarSummonPool,
    OneStarSummonOperation,
    OneStarTransaction,
    OneStarTutorialDeliveryOperation,
    OneStarDurabilityUpdate,
)


class OneStarTransactionError(ValueError):
    """A compact One-Star update cannot be committed safely."""


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

    return CheckpointFile.model_validate_json(checkpoint.model_dump_json())


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
    already_applied: bool = False


@dataclass(frozen=True, slots=True)
class OneStarSummonDraw:
    """One authoritative slot in the next unconsumed standard-pool draw."""

    slot: int
    birth_stars: int
    existing_character_id: str = ""


def is_one_star_checkpoint(checkpoint: CheckpointFile) -> bool:
    return checkpoint.session.config.settings.ruleset_id == ONE_STAR_RULESET_ID


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


def one_star_birth_stars_for_ticket(
    pool: OneStarSummonPool,
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
        "summon": frozenset({"hero_id"}),
        "inventory_delta": frozenset(),
        "hero_delta": frozenset({
            "hp_current",
            "hp_max",
            "level",
            "experience_delta",
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
        "pending_resolve": frozenset({"cull"}),
        "pending_cancel": frozenset(),
        "tutorial_delivery": frozenset({"recipient"}),
        "active_feed": frozenset(),
    }
    prefix_by_kind: dict[str, tuple[str, ...]] = {
        "hero_delta": (
            "stat.",
            "durability.",
            "skill_rank.",
            "equipment_add.",
            "skill_add.",
        ),
        "mission_start": ("counter.", "formation."),
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
            if key.startswith(("stat.", "durability.", "skill_rank.")):
                if not key.partition(".")[2]:
                    raise OneStarTransactionError(
                        f"One-Star hero_delta state update has empty detail id {key!r}"
                    )

    for prefix in ("counter.", "formation."):
        for key in details:
            if key.startswith(prefix) and not key.removeprefix(prefix):
                raise OneStarTransactionError(
                    f"One-Star {update.kind} state update has empty detail id {key!r}"
                )


def _validate_state_update_scalar_shape(update: OneStarStateUpdate) -> None:
    empty_value_kinds = {
        "hero_delta",
        "mission_update",
        "pending_cancel",
        "tutorial_delivery",
        "active_feed",
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
            supplied_ids = [value for value in details.get("hero_id", []) if value]
            if pool.usage == "standard":
                if details:
                    raise OneStarTransactionError(
                        "standard summon state updates must not name hidden draw results"
                    )
                draws = one_star_summon_draw_preview(
                    checkpoint,
                    target_id,
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
                if supplied_ids and supplied_ids != hero_ids:
                    raise OneStarTransactionError(
                        "standard summon ids are adapter-authored and must not be replaced"
                    )
                birth_stars = [draw.birth_stars for draw in draws]
            else:
                if len(supplied_ids) != count:
                    raise OneStarTransactionError(
                        "authored opening summon must name each exact Hero id"
                    )
                hero_ids = supplied_ids
                birth_stars = []
                for hero_id in hero_ids:
                    character = next(
                        (
                            item for item in checkpoint.characters
                            if item.character_id == hero_id
                        ),
                        None,
                    )
                    hero = load_one_star_hero(character) if character else None
                    birth_stars.append(
                        hero.birth_stars if hero is not None
                        else pool.minimum_birth_stars
                    )
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

        if kind == "hero_delta":
            stat_deltas = [
                OneStarStatDelta(
                    stat_id=key.removeprefix("stat."),
                    delta=_integer_update_value(
                        _single_detail(details, key),
                        label=key,
                    ),
                )
                for key, values in details.items()
                if key.startswith("stat.")
            ]
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
                hp_max=(
                    _integer_update_value(
                        _single_detail(details, "hp_max"),
                        label="hp_max",
                    )
                    if "hp_max" in details else None
                ),
                level=(
                    _integer_update_value(
                        _single_detail(details, "level"),
                        label="level",
                    )
                    if "level" in details else None
                ),
                experience_delta=_integer_update_value(
                    _single_detail(details, "experience_delta", default="0"),
                    label="experience_delta",
                ),
                stats_delta=stat_deltas,
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
            formations = [
                OneStarFormationEntry(
                    character_id=key.removeprefix("formation."),
                    label=_single_detail(details, key),
                )
                for key, values in details.items()
                if key.startswith("formation.")
            ]
            operations.append(OneStarMissionStartOperation(
                operation=kind,
                pending_operation_id=_single_detail(
                    details,
                    "pending_operation_id",
                ),
                mission={
                    "mission_id": target_id,
                    "floor": _integer_update_value(update.value, label="mission floor"),
                    "party_ids": details.get("party", []),
                    "formation_labels": formations,
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
                pending=OneStarPendingOperation(
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
                cull_ids=details.get("cull", []),
                promotion_target_stars=(
                    _integer_update_value(update.value, label="promotion target stars")
                    if update.value else None
                ),
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

        if kind == "active_feed":
            operations.append(OneStarActiveFeedOperation(
                operation=kind,
                hero_id=target_id,
            ))
            continue

        raise OneStarTransactionError(
            f"unsupported One-Star state update kind {kind!r}"
        )

    return OneStarTransaction(
        present=bool(operations),
        operations=operations,
    )


def one_star_standard_summon_lifecycle(
    checkpoint: CheckpointFile,
    state_updates: Iterable[OneStarStateUpdate],
) -> tuple[tuple[SpawnRequest, ...], tuple[WakeSignal, ...]]:
    """Materialize standard weighted draws without exposing them to the router."""

    updates = [update for update in state_updates if update.kind == "summon"]
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
        if pool is None or pool.usage != "standard":
            continue
        if update.details:
            raise OneStarTransactionError(
                "standard summon state updates must not name hidden draw results"
            )
        count = _integer_update_value(update.value, label="summon count")
        draws = one_star_summon_draw_preview(checkpoint, pool_id, count=count)
        start_index = account.state.summon_draw_counters.get(pool_id, 0)
        for offset, draw in enumerate(draws):
            if draw.existing_character_id:
                wake_signals.append(WakeSignal(
                    character_id=draw.existing_character_id,
                    location_label=account.config.lobby_location_label,
                ))
                continue
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


def _validate_hero_constraints(hero: OneStarHeroState, config: OneStarRulesConfig) -> None:
    bounds = config.hero_constraints
    if hero.current_stars not in config.star_level_caps:
        raise OneStarTransactionError("Hero current stars have no configured level cap")
    if hero.birth_stars not in config.star_level_caps:
        raise OneStarTransactionError("Hero birth stars have no configured level cap")
    if hero.level > config.star_level_caps[hero.current_stars]:
        raise OneStarTransactionError("Hero level exceeds configured current-star cap")
    if not bounds.minimum_hp_max <= hero.hp_max <= bounds.maximum_hp_max:
        raise OneStarTransactionError("Hero HP maximum violates configured bounds")
    if hero.hp_current < 0 or hero.hp_current > hero.hp_max:
        raise OneStarTransactionError("Hero current HP violates its maximum")
    if hero.experience_points > bounds.maximum_xp:
        raise OneStarTransactionError("Hero XP exceeds configured bound")
    if any(abs(value) > bounds.maximum_stat_value for value in hero.stats.values()):
        raise OneStarTransactionError("Hero stat value exceeds configured bound")
    if len(hero.equipment) > bounds.maximum_equipment_entries:
        raise OneStarTransactionError("Hero equipment exceeds configured bound")
    if len(hero.skills) > bounds.maximum_skill_entries:
        raise OneStarTransactionError("Hero skills exceed configured bound")
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


def validate_one_star_hero_state(
    hero: OneStarHeroState,
    config: OneStarRulesConfig,
) -> None:
    """Validate a generated or seeded Hero against story-authored bounds."""

    _validate_hero_constraints(hero, config)


def one_star_transaction_cull_ids(
    transaction: OneStarTransaction,
) -> tuple[str, ...]:
    """Project terminal identities without inventing a generic cull signal."""

    character_ids: list[str] = []
    for operation in transaction.operations:
        if (
            isinstance(operation, OneStarHeroDeltaOperation)
            and operation.terminal_action == "death"
        ):
            character_ids.append(operation.hero_id)
        elif isinstance(operation, OneStarPendingResolveOperation):
            character_ids.extend(operation.cull_ids)
    return tuple(dict.fromkeys(cid for cid in character_ids if cid))


def _has_synthesis_advancement(
    before: OneStarHeroState,
    after: OneStarHeroState,
) -> bool:
    """Recognize a concrete target gain without deciding its fictional cause."""

    if after.level > before.level or after.experience_points > before.experience_points:
        return True
    if after.hp_max > before.hp_max:
        return True
    if any(
        value > before.stats.get(stat_id, 0)
        for stat_id, value in after.stats.items()
    ):
        return True
    before_equipment = {item.item_id for item in before.equipment}
    if any(item.item_id not in before_equipment for item in after.equipment):
        return True
    before_skills = {skill.skill_id: skill.rank for skill in before.skills}
    if any(
        skill.skill_id not in before_skills
        or skill.rank > before_skills[skill.skill_id]
        for skill in after.skills
    ):
        return True
    if set(after.conditions) < set(before.conditions):
        return True
    return set(after.persistent_injuries) < set(before.persistent_injuries)


def _has_synthesis_regression(
    before: OneStarHeroState,
    after: OneStarHeroState,
) -> bool:
    """Reject paying permanent sources for a target state that got worse."""

    if after.hp_current < before.hp_current or after.hp_max < before.hp_max:
        return True
    if any(
        after.stats.get(stat_id, 0) < before.stats.get(stat_id, 0)
        for stat_id in set(before.stats) | set(after.stats)
    ):
        return True
    after_equipment = {item.item_id: item for item in after.equipment}
    for item in before.equipment:
        updated = after_equipment.get(item.item_id)
        if updated is None or updated.quantity < item.quantity:
            return True
        if updated.durability_current < item.durability_current:
            return True
    after_skills = {skill.skill_id: skill for skill in after.skills}
    for skill in before.skills:
        updated = after_skills.get(skill.skill_id)
        if updated is None or updated.rank < skill.rank:
            return True
    if not set(after.conditions).issubset(before.conditions):
        return True
    return not set(after.persistent_injuries).issubset(before.persistent_injuries)


def _apply_inventory_delta(
    operation: OneStarInventoryDeltaOperation,
    state: OneStarAccountState,
) -> None:
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
    if pool.usage == "opening_actor":
        if (
            len(operation.hero_ids) != 1
            or operation.hero_ids[0] != initiating_actor_id
            or set(operation.hero_ids) != activated_ids
            or spawned_ids
            or state.applied_event_fingerprints
        ):
            raise OneStarTransactionError(
                "opening-actor summon must be the first event and activate only its actor"
            )
        if any(
            character.status != CharacterStatus.culled
            and (hero := load_one_star_hero(character)) is not None
            and hero.owner_lobby_id == config.lobby_id
            for character in checkpoint.characters
        ):
            raise OneStarTransactionError(
                "opening-actor summon requires an account with no acquired Heroes"
            )
        weighted_draws: tuple[OneStarSummonDraw, ...] = ()
    elif pool.usage == "opening_wave":
        account_owner = find_one_star_account_owner(checkpoint.characters)
        if (
            set(operation.hero_ids) != spawned_ids
            or activated_ids
            or state.applied_event_fingerprints
            or account_owner is None
            or initiating_actor_id != account_owner.character_id
        ):
            raise OneStarTransactionError(
                "opening-wave summon must be the account owner's first event "
                "and contain only fresh Hero spawns"
            )
        if any(
            character.status != CharacterStatus.culled
            and (hero := load_one_star_hero(character)) is not None
            and hero.owner_lobby_id == config.lobby_id
            for character in checkpoint.characters
        ):
            raise OneStarTransactionError(
                "opening-wave summon requires an account with no acquired Heroes"
            )
        weighted_draws = ()
    else:
        weighted_draws = _one_star_summon_draw_preview(
            checkpoint,
            config,
            state,
            pool_id=operation.pool_id,
            count=len(operation.hero_ids),
        )
        expected_birth_stars = [draw.birth_stars for draw in weighted_draws]
        if operation.birth_stars != expected_birth_stars:
            raise OneStarTransactionError(
                "summon birth stars must consume the exact next weighted draw prefix"
            )
        for hero_id, draw in zip(
            operation.hero_ids,
            weighted_draws,
            strict=True,
        ):
            if draw.existing_character_id:
                if (
                    hero_id != draw.existing_character_id
                    or hero_id not in activated_ids
                    or hero_id in spawned_ids
                ):
                    raise OneStarTransactionError(
                        "summon must activate the exact reserve selected by "
                        "its weighted draw"
                    )
            elif hero_id not in spawned_ids or hero_id in activated_ids:
                raise OneStarTransactionError(
                    "a fresh weighted summon result requires a matching new Hero spawn"
                )
    if any(
        stars < pool.minimum_birth_stars or stars > pool.maximum_birth_stars
        for stars in operation.birth_stars
    ):
        raise OneStarTransactionError("summon birth stars fall outside the selected pool")
    _spend_resources(state.resources, _multiply_cost(pool.cost, len(operation.hero_ids)))
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
            if not pool.fresh_generation_allowed:
                raise OneStarTransactionError(
                    "summon pool does not allow fresh generation"
                )
            arrival_location = existing.location
        elif hero_id in activated_ids:
            if hero_id not in pool.eligible_existing_ids:
                raise OneStarTransactionError("existing summon identity is not eligible for this pool")
            if existing.status != CharacterStatus.dormant:
                raise OneStarTransactionError(
                    "existing summon reserves must be dormant before activation"
                )
            arrival_location = activated_locations[hero_id] or existing.location
        else:  # Defensive: exact set equality above makes this unreachable.
            raise OneStarTransactionError(
                "summon identity lacks a matching spawn or activation signal"
            )
        if arrival_location != config.lobby_location_label:
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
        _validate_hero_constraints(hero, config)
        hero_initializations[hero_id] = hero
        _store_hero(existing, hero)
    if weighted_draws:
        state.summon_draw_counters[operation.pool_id] = (
            state.summon_draw_counters.get(operation.pool_id, 0)
            + len(weighted_draws)
        )


def _apply_hero_delta(
    operation: OneStarHeroDeltaOperation,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
) -> None:
    character, hero = _require_local_active_hero(checkpoint, operation.hero_id, config)
    prior_level = hero.level
    if operation.hp_current is not None:
        hero.hp_current = operation.hp_current
    if operation.hp_max is not None:
        hero.hp_max = operation.hp_max
    if operation.level is not None:
        if operation.level < prior_level:
            raise OneStarTransactionError("Hero level cannot regress")
        hero.level = operation.level
    hero.experience_points += operation.experience_delta
    if hero.experience_points < 0:
        raise OneStarTransactionError("Hero XP cannot become negative")
    for stat_delta in operation.stats_delta:
        key = stat_delta.stat_id
        hero.stats[key] = hero.stats.get(key, 0) + stat_delta.delta
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
    _validate_hero_constraints(validated_hero, config)
    _store_hero(character, validated_hero)


def _apply_mission_start(
    operation: OneStarMissionStartOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
    now_s: int,
    resolved_deployments: Mapping[str, object],
) -> None:
    if state.active_mission is not None:
        raise OneStarTransactionError("cannot start a mission while another is active")
    if operation.mission.started_at_s != now_s:
        raise OneStarTransactionError("mission start must use the canonical event time")
    pending = resolved_deployments.get(operation.pending_operation_id)
    if pending is None:
        raise OneStarTransactionError(
            "mission start requires a deployment resolved in this event"
        )
    pending_party = set(getattr(pending, "participant_ids", ()))
    if set(operation.mission.party_ids) != pending_party:
        raise OneStarTransactionError(
            "mission party must exactly match the resolved deployment"
        )
    if operation.mission.destination != getattr(pending, "destination", ""):
        raise OneStarTransactionError(
            "mission destination must match the resolved deployment"
        )
    if operation.mission.floor > state.highest_unlocked_floor:
        raise OneStarTransactionError("mission targets a locked Tower floor")
    if operation.mission.floor not in config.floor_rewards:
        raise OneStarTransactionError(
            "mission targets a floor with no reviewed reward authority"
        )
    for hero_id in operation.mission.party_ids:
        character, _hero = _require_local_active_hero(
            checkpoint, hero_id, config
        )
        if character.location != operation.mission.destination:
            raise OneStarTransactionError(
                "mission party has not physically entered its destination"
            )
    party_ids = set(operation.mission.party_ids)
    for character in checkpoint.characters:
        if (
            character.character_id in party_ids
            or character.status != CharacterStatus.active
            or character.location != operation.mission.destination
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


def _apply_mission_end(
    operation: OneStarMissionEndOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
    now_s: int,
    pre_event_escape_authorities: Mapping[str, set[str]],
) -> None:
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
        if first_clear:
            _add_resources(state.resources, OneStarCost.model_validate(reward.model_dump()))
            state.highest_cleared_floor = mission.floor
            next_floor = mission.floor + 1
            if next_floor in config.floor_rewards:
                state.highest_unlocked_floor = max(
                    state.highest_unlocked_floor,
                    next_floor,
                )
        else:
            gold = reward.gold * config.repeat_gold_numerator // config.repeat_gold_denominator
            if reward.gold:
                gold = max(config.repeat_gold_minimum, gold)
            _add_resources(state.resources, OneStarCost(gold=gold, gems=0, building_resources=0, materials={}))
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
    state.active_master_feed_id = ""


def _apply_pending_open(
    operation: OneStarPendingOpenOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
    now_s: int,
) -> None:
    if state.active_mission is not None:
        raise OneStarTransactionError(
            "cannot open a lobby operation while a Tower mission is active"
        )
    if state.pending_operation is not None:
        raise OneStarTransactionError("only one embodied One-Star operation may be pending")
    pending = operation.pending
    validate_one_star_pending_operation_shape(pending)
    if pending.opened_at_s != now_s:
        raise OneStarTransactionError("pending operation must use canonical event time")
    for hero_id in pending.participant_ids:
        _require_local_active_hero(checkpoint, hero_id, config)
    if pending.kind in {"synthesis", "promotion"}:
        _require_local_active_hero(checkpoint, pending.target_id, config)
    if not pending.destination:
        raise OneStarTransactionError(
            "embodied operations require a physical destination"
        )
    lobby_operation_locations = {
        requirement.required_location
        for kind, requirement in config.operation_requirements.items()
        if kind != "deployment" and requirement.required_location
    }
    if pending.kind == "deployment" and pending.destination in {
        config.lobby_location_label,
        *lobby_operation_locations,
    }:
        raise OneStarTransactionError(
            "deployment destination must cross beyond the configured lobby"
        )
    _require_operational_facility(pending, state, config)
    state.pending_operation = pending


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
    restored_lines = [
        f"What promotion has returned at memory tier {promoted_tier}:"
    ]
    if rung.personal_depth.strip():
        restored_lines.append(rung.personal_depth.strip())
    if rung.world_knowledge.strip():
        restored_lines.append(
            "What you now remember about this world: "
            + rung.world_knowledge.strip()
        )
    restored = "\n".join(restored_lines)
    character.known_context = "\n\n".join(
        value
        for value in (character.known_context.strip(), restored)
        if value
    )
    character.knowledge_tier = promoted_tier
    if rung.agent_tier is not None:
        character.agent_tier = rung.agent_tier


def _apply_pending_resolve(
    operation: OneStarPendingResolveOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
) -> object:
    pending = state.pending_operation
    if pending is None or pending.operation_id != operation.operation_id:
        raise OneStarTransactionError("pending resolution does not match the open operation")
    _require_operational_facility(pending, state, config)
    cull_ids = list(dict.fromkeys(cid.strip() for cid in operation.cull_ids if cid.strip()))
    if len(cull_ids) != len(operation.cull_ids):
        raise OneStarTransactionError("pending resolution cull ids must be unique and non-empty")
    if pending.kind == "deployment":
        if cull_ids or operation.promotion_target_stars is not None:
            raise OneStarTransactionError("deployment resolution cannot cull or promote Heroes")
        for hero_id in pending.participant_ids:
            character, _hero = _require_local_active_hero(checkpoint, hero_id, config)
            if character.location != pending.destination:
                raise OneStarTransactionError(
                    "deployment cannot resolve before every Hero physically enters the gate"
                )
        state.active_master_feed_id = pending.participant_ids[0]
    elif pending.kind == "synthesis":
        if set(cull_ids) != set(pending.participant_ids):
            raise OneStarTransactionError("synthesis resolution must cull exactly all selected sources")
        if operation.promotion_target_stars is not None:
            raise OneStarTransactionError("synthesis resolution cannot promote Heroes")
        target_character, target = _require_local_active_hero(checkpoint, pending.target_id, config)
        if target_character.location != pending.destination:
            raise OneStarTransactionError(
                "synthesis target has not physically entered the chamber"
            )
        for source_id in cull_ids:
            source_character, source = _require_local_active_hero(checkpoint, source_id, config)
            if source_id == pending.target_id:
                raise OneStarTransactionError("synthesis target cannot be culled")
            if source_character.location != pending.destination:
                raise OneStarTransactionError(
                    "synthesis source has not physically entered the chamber"
                )
            source.terminal_cause = "synthesized"
            source_character.status = CharacterStatus.culled
            _store_hero(source_character, source)
        _validate_hero_constraints(target, config)
    else:  # promotion
        if cull_ids:
            raise OneStarTransactionError("promotion resolution cannot cull Heroes")
        target_character, target = _require_local_active_hero(checkpoint, pending.target_id, config)
        if target_character.location != pending.destination:
            raise OneStarTransactionError(
                "promotion target has not physically entered the chamber"
            )
        next_stars = operation.promotion_target_stars
        if next_stars is None or next_stars != target.current_stars + 1:
            raise OneStarTransactionError("promotion must advance exactly one star")
        if next_stars not in config.star_level_caps:
            raise OneStarTransactionError("promotion target has no configured star cap")
        if target.level != config.star_level_caps[target.current_stars]:
            raise OneStarTransactionError(
                "promotion requires the Hero to reach the current-star level cap"
            )
        _spend_resources(state.resources, config.promotion_cost)
        target.current_stars = next_stars
        _restore_promotion_knowledge(
            checkpoint,
            target_character,
            next_stars,
        )
        _validate_hero_constraints(target, config)
        _store_hero(target_character, target)
    state.pending_operation = None
    return pending


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


def _apply_active_feed(
    operation: OneStarActiveFeedOperation,
    state: OneStarAccountState,
    checkpoint: CheckpointFile,
    config: OneStarRulesConfig,
) -> None:
    if operation.hero_id:
        _require_local_active_hero(checkpoint, operation.hero_id, config)
    state.active_master_feed_id = operation.hero_id


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
    if pending_open_operations:
        if len(transaction.operations) != 1:
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
            | set(normalized_locations)
        )
        if changed_while_opening:
            raise OneStarTransactionError(
                "opening an embodied operation cannot move or change the "
                "lifecycle of affected Heroes: "
                + ", ".join(sorted(changed_while_opening))
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
    _recover_stamina(state, config, stamina_now_s)
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
    synthesis_targets: set[str] = set()
    promotion_targets: set[str] = set()
    actor_id = initiating_actor_id.strip()

    def require_account_owner(operation_name: str) -> None:
        if actor_id != owner.character_id:
            raise OneStarTransactionError(
                f"only the account owner may initiate {operation_name}"
            )

    for operation in transaction.operations:
        if isinstance(operation, OneStarCatalogueApplyOperation):
            require_account_owner("a catalogue operation")
            if state.active_mission is not None:
                raise OneStarTransactionError(
                    "account catalogue controls are unavailable during an active mission"
                )
            _apply_catalogue(operation, state, config)
        elif isinstance(operation, OneStarSummonOperation):
            if summon_ids:
                raise OneStarTransactionError("a transaction may contain only one summon operation")
            pool = config.summon_pools.get(operation.pool_id)
            if pool is None:
                raise OneStarTransactionError(
                    "summon references an unknown configured pool"
                )
            if pool.usage in {"standard", "opening_wave"}:
                require_account_owner("an account summon")
                if state.active_mission is not None:
                    raise OneStarTransactionError(
                        "account summons are unavailable during an active mission"
                    )
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
            )
        elif isinstance(operation, OneStarInventoryDeltaOperation):
            _apply_inventory_delta(operation, state)
        elif isinstance(operation, OneStarHeroDeltaOperation):
            _apply_hero_delta(operation, after, config)
        elif isinstance(operation, OneStarMissionStartOperation):
            _apply_mission_start(
                operation,
                state,
                after,
                config,
                now_s,
                resolved_deployments,
            )
            started_deployment_ids.add(operation.pending_operation_id)
        elif isinstance(operation, OneStarMissionUpdateOperation):
            _apply_mission_update(operation, state, now_s)
        elif isinstance(operation, OneStarMissionEndOperation):
            _apply_mission_end(
                operation,
                state,
                after,
                config,
                now_s,
                pre_event_escape_authorities,
            )
        elif isinstance(operation, OneStarPendingOpenOperation):
            require_account_owner("an embodied operation selection")
            _apply_pending_open(operation, state, after, config, now_s)
        elif isinstance(operation, OneStarPendingResolveOperation):
            if operation.operation_id != preexisting_pending_operation_id:
                raise OneStarTransactionError(
                    "an embodied operation cannot resolve in the event that opened it"
                )
            resolved = _apply_pending_resolve(
                operation, state, after, config
            )
            if getattr(resolved, "kind", "") == "deployment":
                resolved_deployments[operation.operation_id] = resolved
            elif getattr(resolved, "kind", "") == "synthesis":
                synthesis_targets.add(getattr(resolved, "target_id", ""))
            elif getattr(resolved, "kind", "") == "promotion":
                promotion_targets.add(getattr(resolved, "target_id", ""))
        elif isinstance(operation, OneStarPendingCancelOperation):
            if operation.operation_id != preexisting_pending_operation_id:
                raise OneStarTransactionError(
                    "an embodied operation cannot cancel in the event that opened it"
                )
            _apply_pending_cancel(operation, state)
        elif isinstance(operation, OneStarTutorialDeliveryOperation):
            if actor_id not in state.guide_character_ids:
                raise OneStarTransactionError(
                    "tutorial delivery must originate from a configured guide"
                )
            _apply_tutorial(operation, state, after)
        elif isinstance(operation, OneStarActiveFeedOperation):
            require_account_owner("an active-feed selection")
            _apply_active_feed(operation, state, after, config)
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
    for target_id in synthesis_targets:
        before = before_character_state.get(target_id)
        after_character = _require_character(after, target_id)
        after_hero = load_one_star_hero(after_character)
        try:
            before_hero = (
                OneStarHeroState.model_validate(before[3])
                if before is not None and before[3] is not None
                else None
            )
        except ValidationError as exc:
            raise OneStarTransactionError(
                "synthesis target had invalid pre-event Hero state"
            ) from exc
        if (
            before_hero is None
            or after_hero is None
            or after_character.status != CharacterStatus.active
            or not _has_synthesis_advancement(before_hero, after_hero)
            or _has_synthesis_regression(before_hero, after_hero)
        ):
            raise OneStarTransactionError(
                "synthesis must leave its active target with a concrete mechanical advancement"
            )
    for target_id in promotion_targets:
        before = before_character_state.get(target_id)
        after_hero = load_one_star_hero(_require_character(after, target_id))
        before_hero = (
            OneStarHeroState.model_validate(before[3])
            if before is not None and before[3] is not None
            else None
        )
        if (
            before_hero is None
            or after_hero is None
            or after_hero.level != before_hero.level
            or after_hero.experience_points != before_hero.experience_points
        ):
            raise OneStarTransactionError(
                "promotion must retain the target's level and experience"
            )

    if state.active_master_feed_id in dormant_ids:
        state.active_master_feed_id = ""
    if state.active_master_feed_id:
        feed_character = _require_character(after, state.active_master_feed_id)
        feed_hero = load_one_star_hero(feed_character)
        if (
            feed_character.status != CharacterStatus.active
            or feed_hero is None
            or feed_hero.owner_lobby_id != config.lobby_id
        ):
            state.active_master_feed_id = ""
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
