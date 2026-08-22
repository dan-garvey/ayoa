"""Typed, opt-in state contracts for the One-Star Ascension rules adapter.

The generic narrative engine deliberately does not import this module.  The
adapter owns its state below ``CharacterRecord.mechanics`` and only uses the
normal character lifecycle/location fields for embodied state.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.event_router import ClosedEventRouterOutput, EventRouterOutput


ONE_STAR_RULESET_ID = "one_star_ascension"
ONE_STAR_ACCOUNT_KEY = "one_star_account"
ONE_STAR_HERO_KEY = "one_star_hero"
ONE_STAR_GACHA_WEIGHT_TOTAL = 10_000


class OneStarResources(BaseModel):
    """Currency and material amounts.  All fields are seed-authored values."""

    model_config = ConfigDict(extra="forbid")

    gold: int = Field(ge=0)
    gems: int = Field(ge=0)
    building_resources: int = Field(ge=0)
    materials: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_materials(self) -> "OneStarResources":
        if any(not key.strip() or amount < 0 for key, amount in self.materials.items()):
            raise ValueError("materials must have non-empty ids and non-negative amounts")
        self.materials = {
            key.strip(): amount
            for key, amount in self.materials.items()
            if key.strip()
        }
        return self


class OneStarCost(OneStarResources):
    """A non-negative catalogue price or fixed reward amount."""


class OneStarCatalogueEntry(BaseModel):
    """A story-authored fixed-price catalogue item.

    ``kind`` says which transaction operation may spend this price; the
    adapter intentionally does not attach any story-specific facility names to
    it.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["purchase", "facility_build", "facility_upgrade", "research"]
    cost: OneStarCost
    inventory_item_id: str = ""
    facility_id: str = ""
    target_level: int = Field(default=0, ge=0)
    required_cleared_floor: int = Field(default=0, ge=0)
    required_lobby_floor: int = Field(default=0, ge=0)
    resulting_lobby_floor: int = Field(default=0, ge=0)
    resulting_capacity: int = Field(default=0, ge=0)
    research_key: str = ""
    research_level: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_entry(self) -> "OneStarCatalogueEntry":
        self.inventory_item_id = self.inventory_item_id.strip()
        self.facility_id = self.facility_id.strip()
        self.research_key = self.research_key.strip()
        if self.kind == "purchase" and not self.inventory_item_id:
            raise ValueError("purchase catalogue entries require an item id")
        if self.kind in {"facility_build", "facility_upgrade"} and (
            not self.facility_id or self.target_level < 1
        ):
            raise ValueError(
                "facility catalogue entries require an id and target level"
            )
        if self.kind == "research" and (
            not self.research_key or self.research_level < 1
        ):
            raise ValueError("research entries require a key and level")
        return self


class OneStarSummonPool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cost: OneStarCost
    minimum_birth_stars: int = Field(ge=1)
    maximum_birth_stars: int = Field(ge=1)
    star_weights: dict[int, int]
    eligible_existing_ids: list[str] = Field(default_factory=list)
    fresh_generation_allowed: bool = False
    usage: Literal["standard", "opening_actor", "opening_wave"]

    @model_validator(mode="after")
    def _valid_star_range(self) -> "OneStarSummonPool":
        if self.maximum_birth_stars < self.minimum_birth_stars:
            raise ValueError("maximum_birth_stars must be at least minimum_birth_stars")
        expected_stars = set(
            range(
                self.minimum_birth_stars,
                self.maximum_birth_stars + 1,
            )
        )
        if set(self.star_weights) != expected_stars:
            raise ValueError(
                "star_weights must define every configured birth-star grade exactly"
            )
        if any(weight <= 0 for weight in self.star_weights.values()):
            raise ValueError("summon star weights must be positive")
        if sum(self.star_weights.values()) != ONE_STAR_GACHA_WEIGHT_TOTAL:
            raise ValueError(
                f"summon star weights must total {ONE_STAR_GACHA_WEIGHT_TOTAL}"
            )
        self.eligible_existing_ids = list(dict.fromkeys(
            value.strip() for value in self.eligible_existing_ids if value.strip()
        ))
        if self.usage == "opening_actor":
            if self.fresh_generation_allowed:
                raise ValueError(
                    "opening-actor summon pools cannot generate a substitute"
                )
            if any((
                self.cost.gold,
                self.cost.gems,
                self.cost.building_resources,
                *self.cost.materials.values(),
            )):
                raise ValueError("opening-actor summon pools must be free")
        if self.usage == "opening_wave":
            if (
                not self.fresh_generation_allowed
                or self.eligible_existing_ids
                or self.minimum_birth_stars != self.maximum_birth_stars
            ):
                raise ValueError(
                    "opening-wave summon pools require one fixed fresh grade and no reserves"
                )
        return self


class OneStarFloorReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gold: int = Field(ge=0)
    gems: int = Field(ge=0)
    building_resources: int = Field(ge=0)
    materials: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_materials(self) -> "OneStarFloorReward":
        if any(not key.strip() or amount < 0 for key, amount in self.materials.items()):
            raise ValueError("floor reward materials must have non-empty ids and non-negative amounts")
        return self


class OneStarHeroConstraints(BaseModel):
    """Seed-authored hard bounds, not a stat formula or allocation budget."""

    model_config = ConfigDict(extra="forbid")

    minimum_hp_max: int = Field(ge=1)
    maximum_hp_max: int = Field(ge=1)
    maximum_xp: int = Field(ge=0)
    maximum_stat_value: int = Field(ge=0)
    maximum_equipment_entries: int = Field(ge=0)
    maximum_skill_entries: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "OneStarHeroConstraints":
        if self.maximum_hp_max < self.minimum_hp_max:
            raise ValueError("maximum_hp_max must be at least minimum_hp_max")
        return self


OneStarEmbodiedOperationKind = Literal["deployment", "synthesis", "promotion"]


class OneStarOperationRequirement(BaseModel):
    """Seed-authored physical prerequisite for an embodied operation."""

    model_config = ConfigDict(extra="forbid")

    facility_id: str
    required_location: str

    @model_validator(mode="after")
    def _clean(self) -> "OneStarOperationRequirement":
        self.facility_id = self.facility_id.strip()
        self.required_location = self.required_location.strip()
        if not self.facility_id:
            raise ValueError("embodied operation requirement needs a facility id")
        return self


class OneStarRulesConfig(BaseModel):
    """The complete seed-authored fixed ledger configuration for one story."""

    model_config = ConfigDict(extra="forbid")

    starting_resources: OneStarResources
    lobby_id: str
    lobby_location_label: str
    catalogue: dict[str, OneStarCatalogueEntry]
    summon_pools: dict[str, OneStarSummonPool]
    star_level_caps: dict[int, int]
    starting_lobby_floor: int = Field(ge=1)
    starting_capacity: int = Field(ge=1)
    maximum_stamina: int = Field(ge=0)
    stamina_recovery_seconds: int = Field(ge=1)
    deployment_stamina_cost: int = Field(ge=0)
    max_summon_batch: int = Field(ge=1, le=20)
    hero_constraints: OneStarHeroConstraints
    floor_rewards: dict[int, OneStarFloorReward]
    repeat_gold_numerator: int = Field(ge=0)
    repeat_gold_denominator: int = Field(ge=1)
    repeat_gold_minimum: int = Field(ge=0)
    promotion_cost: OneStarCost
    operation_requirements: dict[
        OneStarEmbodiedOperationKind, OneStarOperationRequirement
    ]
    lobby_return_healing: bool
    hero_system_visibility_research_key: str = ""

    @model_validator(mode="after")
    def _validate_config(self) -> "OneStarRulesConfig":
        self.catalogue = {
            key.strip(): value
            for key, value in self.catalogue.items()
            if key.strip()
        }
        self.summon_pools = {
            key.strip(): value
            for key, value in self.summon_pools.items()
            if key.strip()
        }
        self.lobby_id = self.lobby_id.strip()
        self.lobby_location_label = self.lobby_location_label.strip()
        if not self.lobby_id or not self.lobby_location_label:
            raise ValueError(
                "One-Star config requires lobby id and location label"
            )
        if not self.summon_pools:
            raise ValueError("One-Star config requires at least one summon pool")
        if not self.star_level_caps or any(
            stars < 1 or cap < 1 for stars, cap in self.star_level_caps.items()
        ):
            raise ValueError("star_level_caps must contain positive star and level values")
        if any(floor < 1 for floor in self.floor_rewards):
            raise ValueError("floor reward keys must be positive")
        if self.repeat_gold_numerator > self.repeat_gold_denominator:
            raise ValueError("repeat Gold fraction cannot exceed first-clear Gold")
        required_operation_kinds = {"deployment", "synthesis", "promotion"}
        if set(self.operation_requirements) != required_operation_kinds:
            raise ValueError(
                "operation_requirements must define deployment, synthesis, and promotion"
            )
        return self


class OneStarEquipmentEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    name: str
    slot: str
    quantity: int = Field(ge=1)
    durability_current: int = Field(ge=0)
    durability_max: int = Field(ge=0)
    tags: list[str]
    visible: bool

    @model_validator(mode="after")
    def _clean(self) -> "OneStarEquipmentEntry":
        self.item_id = self.item_id.strip()
        self.name = self.name.strip()
        self.slot = self.slot.strip()
        self.tags = list(dict.fromkeys(tag.strip() for tag in self.tags if tag.strip()))
        if not self.item_id or not self.name or not self.slot:
            raise ValueError("equipment requires non-empty item id, name, and slot")
        if self.durability_max == 0 and self.durability_current != 0:
            raise ValueError(
                "untracked equipment durability must use current/max 0"
            )
        if self.durability_max and self.durability_current > self.durability_max:
            raise ValueError("equipment durability_current cannot exceed durability_max")
        return self


class OneStarSkillEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str
    name: str
    rank: int = Field(ge=0)
    capability: str
    tags: list[str]
    visible: bool

    @model_validator(mode="after")
    def _clean(self) -> "OneStarSkillEntry":
        self.skill_id = self.skill_id.strip()
        self.name = self.name.strip()
        self.capability = self.capability.strip()
        self.tags = list(dict.fromkeys(tag.strip() for tag in self.tags if tag.strip()))
        if not self.skill_id or not self.name:
            raise ValueError("skills require non-empty id and name")
        return self


class OneStarHeroState(BaseModel):
    """Durable mechanics for an embodied Hero.

    Private potential is deliberately a separate field so projections can
    include ordinary mechanics without accidentally exposing spoilers.
    """

    model_config = ConfigDict(extra="forbid")

    birth_stars: int = Field(ge=1)
    current_stars: int = Field(ge=1)
    level: int = Field(default=1, ge=1)
    experience_points: int = Field(default=0, ge=0)
    hp_current: int = Field(default=1, ge=0)
    hp_max: int = Field(default=1, ge=1)
    stats: dict[str, int] = Field(default_factory=dict)
    equipment: list[OneStarEquipmentEntry] = Field(default_factory=list)
    skills: list[OneStarSkillEntry] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    persistent_injuries: list[str] = Field(default_factory=list)
    innate_system_sight: bool = False
    # True only when character generation created this identity for a summon.
    # Seeded dormant reserves keep their authored model tier instead.
    generated_for_summon: bool = False
    acquisition_event_id: str = ""
    owner_lobby_id: str = ""
    terminal_cause: str = ""
    hidden_capabilities: dict[str, str] = Field(default_factory=dict)
    private_potential: str = ""

    @model_validator(mode="after")
    def _validate_hero(self) -> "OneStarHeroState":
        if self.current_stars < self.birth_stars:
            raise ValueError("current_stars cannot be below immutable birth_stars")
        if self.hp_current > self.hp_max:
            raise ValueError("hp_current cannot exceed hp_max")
        if any(not key.strip() for key in self.stats):
            raise ValueError("Hero stat ids must be non-empty")
        self.stats = {key.strip(): value for key, value in self.stats.items()}
        self.conditions = list(dict.fromkeys(value.strip() for value in self.conditions if value.strip()))
        self.persistent_injuries = list(
            dict.fromkeys(value.strip() for value in self.persistent_injuries if value.strip())
        )
        self.acquisition_event_id = self.acquisition_event_id.strip()
        self.owner_lobby_id = self.owner_lobby_id.strip()
        self.terminal_cause = self.terminal_cause.strip()
        self.private_potential = self.private_potential.strip()
        if len({entry.item_id for entry in self.equipment}) != len(self.equipment):
            raise ValueError("equipment item_ids must be unique")
        if len({entry.skill_id for entry in self.skills}) != len(self.skills):
            raise ValueError("skill_ids must be unique")
        return self


class OneStarMissionCounter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counter_id: str
    current: int = Field(ge=0)
    target: int = Field(ge=1)

    @model_validator(mode="after")
    def _valid_counter(self) -> "OneStarMissionCounter":
        self.counter_id = self.counter_id.strip()
        if not self.counter_id:
            raise ValueError("mission counter id cannot be empty")
        if self.current > self.target:
            raise ValueError("mission counter current cannot exceed target")
        return self


class OneStarFormationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    label: str

    @model_validator(mode="after")
    def _clean(self) -> "OneStarFormationEntry":
        self.character_id = self.character_id.strip()
        self.label = self.label.strip()
        if not self.character_id or not self.label:
            raise ValueError("mission formation entries require character id and label")
        return self


class OneStarMissionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mission_id: str
    floor: int = Field(ge=1)
    party_ids: list[str]
    formation_labels: list[OneStarFormationEntry]
    destination: str
    completion_declaration: str
    failure_declaration: str
    counters: list[OneStarMissionCounter]
    started_at_s: int = Field(ge=0)
    deadline_at_s: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_mission(self) -> "OneStarMissionState":
        self.mission_id = self.mission_id.strip()
        self.party_ids = list(dict.fromkeys(cid.strip() for cid in self.party_ids if cid.strip()))
        self.destination = self.destination.strip()
        if not self.mission_id or not self.party_ids or not self.destination:
            raise ValueError("mission requires id, party, and destination")
        if not self.completion_declaration.strip() or not self.failure_declaration.strip():
            raise ValueError("mission requires immutable completion and failure declarations")
        if self.deadline_at_s and self.deadline_at_s < self.started_at_s:
            raise ValueError("mission deadline cannot precede start")
        formation_ids = [entry.character_id for entry in self.formation_labels]
        if len(formation_ids) != len(set(formation_ids)):
            raise ValueError("mission formation character ids must be unique")
        if not set(formation_ids).issubset(self.party_ids):
            raise ValueError("mission formation ids must belong to the party")
        counter_ids = [entry.counter_id for entry in self.counters]
        if not counter_ids or len(counter_ids) != len(set(counter_ids)):
            raise ValueError("mission counter ids must be non-empty and unique")
        return self


class OneStarStatDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stat_id: str
    delta: int

    @model_validator(mode="after")
    def _clean(self) -> "OneStarStatDelta":
        self.stat_id = self.stat_id.strip()
        if not self.stat_id:
            raise ValueError("Hero stat delta id cannot be empty")
        return self


class OneStarPendingOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str
    kind: OneStarEmbodiedOperationKind
    participant_ids: list[str]
    target_id: str
    destination: str
    opened_at_s: int = Field(ge=0)

    @model_validator(mode="after")
    def _clean(self) -> "OneStarPendingOperation":
        self.operation_id = self.operation_id.strip()
        self.participant_ids = list(dict.fromkeys(
            cid.strip() for cid in self.participant_ids if cid.strip()
        ))
        self.target_id = self.target_id.strip()
        self.destination = self.destination.strip()
        if not self.operation_id or not self.participant_ids:
            raise ValueError("pending operation requires an id and participants")
        return self


class OneStarAccountState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resources: OneStarResources
    inventory: dict[str, int] = Field(default_factory=dict)
    facilities: dict[str, int] = Field(default_factory=dict)
    research_levels: dict[str, int] = Field(default_factory=dict)
    lobby_floor: int = Field(ge=1)
    capacity: int = Field(ge=1)
    highest_unlocked_floor: int = Field(default=0, ge=0)
    highest_cleared_floor: int = Field(default=0, ge=0)
    stamina_current: int = Field(ge=0)
    stamina_recovery_anchor_s: int = Field(default=0, ge=0)
    active_mission: OneStarMissionState | None = None
    pending_operation: OneStarPendingOperation | None = None
    guide_character_ids: list[str] = Field(default_factory=list)
    system_observer_ids: list[str] = Field(default_factory=list)
    tutorial_deliveries: dict[str, list[str]] = Field(default_factory=dict)
    summon_draw_counters: dict[str, int] = Field(default_factory=dict)
    applied_event_fingerprints: dict[str, str] = Field(default_factory=dict)
    active_master_feed_id: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "OneStarAccountState":
        if any(not key.strip() or value < 0 for key, value in self.inventory.items()):
            raise ValueError("inventory must have non-empty ids and non-negative amounts")
        if any(not key.strip() or value < 0 for key, value in self.facilities.items()):
            raise ValueError("facilities must have non-empty ids and non-negative levels")
        if any(not key.strip() or value < 0 for key, value in self.research_levels.items()):
            raise ValueError("research levels must have non-empty ids and non-negative levels")
        self.inventory = {key.strip(): value for key, value in self.inventory.items() if key.strip()}
        self.facilities = {key.strip(): value for key, value in self.facilities.items() if key.strip()}
        self.research_levels = {key.strip(): value for key, value in self.research_levels.items() if key.strip()}
        self.guide_character_ids = list(dict.fromkeys(cid.strip() for cid in self.guide_character_ids if cid.strip()))
        self.system_observer_ids = list(dict.fromkeys(cid.strip() for cid in self.system_observer_ids if cid.strip()))
        self.tutorial_deliveries = {
            key.strip(): list(
                dict.fromkeys(
                    character_id.strip()
                    for character_id in character_ids
                    if character_id.strip()
                )
            )
            for key, character_ids in self.tutorial_deliveries.items()
            if key.strip()
        }
        if any(
            not pool_id.strip() or draw_count < 0
            for pool_id, draw_count in self.summon_draw_counters.items()
        ):
            raise ValueError(
                "summon draw counters require non-empty pool ids and non-negative values"
            )
        self.summon_draw_counters = {
            pool_id.strip(): draw_count
            for pool_id, draw_count in self.summon_draw_counters.items()
        }
        if any(
            not event_id.strip() or not fingerprint.strip()
            for event_id, fingerprint in self.applied_event_fingerprints.items()
        ):
            raise ValueError(
                "applied event fingerprints require non-empty ids and values"
            )
        self.applied_event_fingerprints = {
            event_id.strip(): fingerprint.strip()
            for event_id, fingerprint in self.applied_event_fingerprints.items()
        }
        return self


class OneStarAccountEnvelope(BaseModel):
    """Value stored at ``mechanics[ONE_STAR_ACCOUNT_KEY]``."""

    model_config = ConfigDict(extra="forbid")

    config: OneStarRulesConfig
    state: OneStarAccountState

    @model_validator(mode="after")
    def _validate_state_against_config(self) -> "OneStarAccountEnvelope":
        if self.state.lobby_floor < self.config.starting_lobby_floor:
            raise ValueError("lobby floor cannot be below its configured start")
        if self.state.capacity < self.config.starting_capacity:
            raise ValueError("Hero capacity cannot be below its configured start")
        if self.state.stamina_current > self.config.maximum_stamina:
            raise ValueError("stamina exceeds its configured maximum")
        if self.state.highest_cleared_floor > self.state.highest_unlocked_floor:
            raise ValueError("cleared Tower floor cannot exceed unlocked floor")
        if (
            self.state.active_mission is not None
            and self.state.pending_operation is not None
        ):
            raise ValueError(
                "an active mission and embodied pending operation cannot coexist"
            )
        standard_pool_ids = {
            pool_id
            for pool_id, pool in self.config.summon_pools.items()
            if pool.usage == "standard"
        }
        unknown_draw_counters = (
            set(self.state.summon_draw_counters) - standard_pool_ids
        )
        if unknown_draw_counters:
            raise ValueError(
                "summon draw counters reference non-standard pools: "
                + ", ".join(sorted(unknown_draw_counters))
            )
        available_facility_ids = {
            *self.state.facilities,
            *(
                entry.facility_id
                for entry in self.config.catalogue.values()
                if entry.facility_id
            ),
        }
        missing = {
            requirement.facility_id
            for requirement in self.config.operation_requirements.values()
            if requirement.facility_id not in available_facility_ids
        }
        if missing:
            raise ValueError(
                "operation requirements reference unavailable facilities: "
                + ", ".join(sorted(missing))
            )
        return self


class OneStarCatalogueApplyOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["catalogue_apply"]
    catalogue_id: str
    quantity: int = Field(ge=1)


class OneStarSummonOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["summon"]
    pool_id: str
    hero_ids: list[str]
    birth_stars: list[int]

    @model_validator(mode="after")
    def _validate_summon(self) -> "OneStarSummonOperation":
        self.hero_ids = [value.strip() for value in self.hero_ids if value.strip()]
        if not self.hero_ids or len(self.hero_ids) != len(set(self.hero_ids)):
            raise ValueError("summon hero_ids must be non-empty and unique")
        if len(self.hero_ids) != len(self.birth_stars):
            raise ValueError("summon hero_ids and birth_stars must align")
        return self


class OneStarInventoryDeltaOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["inventory_delta"]
    item_id: str
    quantity_delta: int

    @model_validator(mode="after")
    def _clean(self) -> "OneStarInventoryDeltaOperation":
        self.item_id = self.item_id.strip()
        if not self.item_id or self.quantity_delta == 0:
            raise ValueError(
                "inventory delta requires a non-empty item id and non-zero delta"
            )
        return self


class OneStarHeroDeltaOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["hero_delta"]
    hero_id: str
    hp_current: int | None = Field(ge=0)
    hp_max: int | None = Field(ge=1)
    level: int | None = Field(ge=1)
    experience_delta: int = Field(ge=0)
    stats_delta: list[OneStarStatDelta]
    equipment_add: list[OneStarEquipmentEntry]
    equipment_remove_ids: list[str]
    skills_add: list[OneStarSkillEntry]
    skills_remove_ids: list[str]
    equipment_durability: list["OneStarDurabilityUpdate"]
    skill_rank_updates: list["OneStarSkillRankUpdate"]
    conditions: list[str] | None
    persistent_injuries: list[str] | None
    terminal_action: Literal["none", "death"]
    death_cause: str

    @model_validator(mode="after")
    def _validate_stat_deltas(self) -> "OneStarHeroDeltaOperation":
        stat_ids = [entry.stat_id for entry in self.stats_delta]
        if len(stat_ids) != len(set(stat_ids)):
            raise ValueError("Hero stat delta ids must be unique")
        return self


class OneStarDurabilityUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str
    durability_current: int = Field(ge=0)


class OneStarSkillRankUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_id: str
    rank: int = Field(ge=0)


class OneStarMissionStartOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["mission_start"]
    pending_operation_id: str
    mission: OneStarMissionState


class OneStarMissionUpdateOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["mission_update"]
    mission_id: str
    counters: list[OneStarMissionCounter]

    @model_validator(mode="after")
    def _validate_counters(self) -> "OneStarMissionUpdateOperation":
        counter_ids = [entry.counter_id for entry in self.counters]
        if not counter_ids or len(counter_ids) != len(set(counter_ids)):
            raise ValueError("mission update counter ids must be non-empty and unique")
        return self


class OneStarMissionEndOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["mission_end"]
    mission_id: str
    outcome: Literal["completed", "failed", "escaped"]
    return_destination: str
    escape_authority_id: str


class OneStarPendingOpenOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["pending_open"]
    pending: OneStarPendingOperation


class OneStarPendingResolveOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["pending_resolve"]
    operation_id: str
    cull_ids: list[str]
    promotion_target_stars: int | None


class OneStarPendingCancelOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["pending_cancel"]
    operation_id: str


class OneStarTutorialDeliveryOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["tutorial_delivery"]
    tutorial_key: str
    delivered_to_ids: list[str]


class OneStarActiveFeedOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["active_feed"]
    hero_id: str


OneStarOperation = (
    OneStarCatalogueApplyOperation
    | OneStarSummonOperation
    | OneStarInventoryDeltaOperation
    | OneStarHeroDeltaOperation
    | OneStarMissionStartOperation
    | OneStarMissionUpdateOperation
    | OneStarMissionEndOperation
    | OneStarPendingOpenOperation
    | OneStarPendingResolveOperation
    | OneStarPendingCancelOperation
    | OneStarTutorialDeliveryOperation
    | OneStarActiveFeedOperation
)


class OneStarTransaction(BaseModel):
    """Private typed mutation batch produced from compact router updates."""

    model_config = ConfigDict(extra="forbid")

    present: bool
    operations: list[OneStarOperation]

    @model_validator(mode="after")
    def _validate_presence(self) -> "OneStarTransaction":
        if self.present != bool(self.operations):
            raise ValueError("mutation-batch presence must exactly match operations")
        return self


OneStarStateUpdateKind = Literal[
    "catalogue_apply",
    "summon",
    "inventory_delta",
    "hero_delta",
    "mission_start",
    "mission_update",
    "mission_end",
    "pending_open",
    "pending_resolve",
    "pending_cancel",
    "tutorial_delivery",
    "active_feed",
]


class OneStarStateUpdate(BaseModel):
    """Compact router-authored semantic update.

    The provider sees this one fixed record instead of the adapter's durable
    mutation models. ``target_id`` identifies the primary account, Hero,
    mission, pending operation, tutorial, or pool. ``value`` carries the
    operation's primary scalar. Additional non-empty ``key=value`` entries
    live in ``details``; repeated keys represent lists. The adapter parses and
    validates those entries into its private typed transaction before commit.
    """

    model_config = ConfigDict(extra="forbid")

    kind: OneStarStateUpdateKind
    target_id: str
    value: str
    details: list[str]


class OneStarStateUpdateList(BaseModel):
    """Narrow repair response for only the adapter update list."""

    model_config = ConfigDict(extra="forbid")

    state_updates: list[OneStarStateUpdate]


class OneStarEventRouterOutput(EventRouterOutput):
    """One-Star router response; imported by the ruleset router dispatcher."""

    state_updates: list[OneStarStateUpdate]


class ClosedOneStarEventRouterOutput(ClosedEventRouterOutput):
    """Closed continuation response carrying the same compact update list."""

    state_updates: list[OneStarStateUpdate]
