"""Frontend-neutral result and view DTOs for engine-facing UI callers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas.narrator import TranscriptEntry
from app.schemas.responses import TurnResponse


@dataclass(frozen=True)
class CharacterSummary:
    """Spoiler-free summary of a character for roster, join, and story views.

    Public-sheet fields plus status / name / binding only. `bound_user_id` is
    populated when reading a session checkpoint and empty on pristine story
    lookups.
    """

    character_id: str
    name: str
    role: str
    faction: str
    appearance: str
    status: str
    is_playable: bool
    bound_user_id: str = ""
    player_slot_kind: str = "standard"
    player_guidance: str = ""


@dataclass(frozen=True)
class StorySummary:
    story_id: str
    title: str
    genre: str
    premise: str
    player_primer: str
    recommended_players: str
    play_guidance: str
    playable_seat_count: int


@dataclass(frozen=True)
class RewindResult:
    """Result of rewinding a session to an earlier checkpoint."""

    session_id: str
    target_turn: int
    previous_latest: int
    new_latest: int
    deleted_turns: list[int]
    location: str = ""
    actor_character_id: str = ""


@dataclass(frozen=True)
class RetryRenderResult:
    response: TurnResponse
    actor_character_id: str = ""
    actor_user_id: str = ""


@dataclass(frozen=True)
class TurnHistoryEntry:
    turn_index: int
    entry: TranscriptEntry


@dataclass(frozen=True)
class PlayerJoinResult:
    character_id: str
    character_name: str
    pre_play: bool
    response: TurnResponse | None = None


@dataclass(frozen=True)
class OpeningLobbyView:
    requires_confirmation: bool
    claimed_seat_names: tuple[str, ...]
    open_seat_names: tuple[str, ...]


@dataclass(frozen=True)
class SessionActivityView:
    session_id: str
    story_id: str
    turn_index: int
    state: str
    viewpoint_name: str = ""
    location: str = ""
    joined_seat_names: tuple[str, ...] = ()
    nearby_character_names: tuple[str, ...] = ()
    requested_next_names: tuple[str, ...] = ()
    last_visible_update: str = ""
    ruleset_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class PendingRollPrompt:
    session_id: str
    event_id: str
    roll_id: str
    actor_id: str
    user_id: str
    label: str
    reason: str


@dataclass(frozen=True)
class CompletedPendingRoll:
    session_id: str
    event_id: str
    roll_id: str
    actor_id: str
    user_id: str
    label: str
    reason: str
    expression: str
    total: int
    detail: str
    crit: str
    remaining_pending_rolls: int
    die_values: tuple[int, ...] = ()
    kept_die_values: tuple[int, ...] = ()
    modifier: int = 0
    dc: int = 0
    outcome: str = ""
    target_id: str = ""
    target_name: str = ""
    damage_total: int = 0
    damage_type: str = ""
    damage_detail: str = ""


@dataclass(frozen=True)
class DndSheetAttachmentSummary:
    character_id: str
    character_name: str
    imported_name: str
    ruleset_id: str
    session_ruleset_id: str
    player_roll_mode: str
    source_type: str
    total_level: int
    classes: list[str]
    armor_class: int
    hit_points_current: int
    hit_points_max: int
    hit_points_temporary: int
    skills_count: int
    actions_count: int
    spells_count: int
    resources_count: int
    name_overridden: bool


@dataclass(frozen=True)
class DndInventoryView:
    character_id: str
    character_name: str
    items: list[dict[str, Any]]
    currency: dict[str, int]


@dataclass(frozen=True)
class DndLootClaimResult:
    offer_id: str
    character_id: str
    claimed_items: list[dict[str, Any]]
    claimed_currency: dict[str, int]
    shares: dict[str, dict[str, int]]
    offer_closed: bool
    message: str


@dataclass(frozen=True)
class DndExperienceAwardResult:
    character_id: str
    character_name: str
    amount: int
    before: int
    after: int
    total_level: int
    next_level: int
    xp_to_next_level: int
    level_available: bool
    eligible_level: int


@dataclass(frozen=True)
class DndCombatParticipantView:
    character_id: str
    name: str
    current: bool = False
    initiative: int | None = None
    hp_current: int | None = None
    hp_max: int | None = None
    hp_temporary: int = 0
    armor_class: int | None = None
    conditions: tuple[str, ...] = ()
    active_effects: tuple[str, ...] = ()
    defeat_state: str = "active"
    death_save_successes: int = 0
    death_save_failures: int = 0
    pending_initiating_action: str = ""


@dataclass(frozen=True)
class DndCombatView:
    session_id: str
    active: bool
    round_number: int = 0
    turn_number: int = 0
    current_participant_id: str = ""
    participants: tuple[DndCombatParticipantView, ...] = ()
    map_lines: tuple[str, ...] = ()
    message: str = ""
