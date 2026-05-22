from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ReviewStatus = Literal[
    "unreviewed",
    "needs_review",
    "reviewed",
    "approved",
    "blocked",
    "rejected",
]
SpoilerClass = Literal["none", "low", "moderate", "high"]
CoverageGateStatus = Literal["runtime_ready", "flagged", "blocked"]
AssetRevealAudience = Literal["all_observers", "only"]
AssetPresentation = Literal["inline", "attachment", "reference", "map_overlay"]
ContentVisibility = Literal[
    "hidden",
    "router_hidden",
    "player_visible_after_reveal",
    "host_only",
    "player_safe",
]
CrossReferenceRelation = Literal[
    "contains",
    "connects_to",
    "depends_on",
    "foreshadows",
    "mentions",
    "reveals",
    "unlocks",
    "uses",
]
RevealRelation = Literal["reveals", "requires", "foreshadows", "unlocks", "blocks"]
RevealAudience = Literal["router", "players", "character", "host"]
SectionKind = Literal["book", "chapter", "section", "appendix", "span_group"]
SpanRole = Literal[
    "heading",
    "body",
    "table",
    "caption",
    "sidebar",
    "boxed_text",
    "map_label",
    "statblock",
]
LocationKind = Literal["region", "site", "level", "keyed_area", "room", "route"]
HandoutKind = Literal["document", "image", "map", "symbol", "object", "other"]
AdventureTableKind = Literal[
    "random_encounter",
    "treasure",
    "rumor",
    "reaction",
    "weather",
    "other",
]
TacticalMapKind = Literal["battle_map", "exploration_map", "region_map", "theater"]
TacticalFeatureKind = Literal[
    "wall",
    "door",
    "window",
    "pit",
    "cliff",
    "stairs",
    "balcony",
    "furniture",
    "water",
    "cover",
    "difficult_ground",
    "blocked_movement",
    "line_of_sight_blocker",
    "secret_feature",
    "other",
]
SpawnAnchorKind = Literal[
    "players",
    "enemies",
    "reinforcements",
    "exit",
    "retreat",
    "fallback",
    "other",
]
FrontActionKind = Literal[
    "spy",
    "threaten",
    "relocate",
    "trap",
    "negotiate",
    "attack",
    "rumor",
    "environment",
    "other",
]
AutomationScope = Literal["combat", "noncombat_lookup", "blocked"]
DndEconomyKind = Literal[
    "action",
    "bonus_action",
    "reaction",
    "legendary_action",
    "lair_action",
    "none",
]
TrapHazardKind = Literal["trap", "hazard", "secret", "environmental_hazard"]
TreasureKind = Literal["hoard", "container", "loose", "reward", "item"]
EncounterDifficulty = Literal["trivial", "easy", "medium", "hard", "deadly", "unknown"]


class ContentProvenance(BaseModel):
    """Compiled source pointer without raw paths or protected excerpts."""

    model_config = ConfigDict(extra="forbid")

    source_asset_id: str = ""
    page_id: str = ""
    span_id: str = ""
    image_id: str = ""
    bbox: list[float] = Field(default_factory=list)
    section_id: str = ""
    method: str = ""
    confidence: float = 1.0
    importer_version: str = ""
    human_review_status: ReviewStatus = "unreviewed"

    @model_validator(mode="after")
    def _clean(self) -> "ContentProvenance":
        self.source_asset_id = self.source_asset_id.strip()
        self.page_id = self.page_id.strip()
        self.span_id = self.span_id.strip()
        self.image_id = self.image_id.strip()
        self.section_id = self.section_id.strip()
        self.method = self.method.strip()
        self.importer_version = self.importer_version.strip()
        self.confidence = _clamp_confidence(self.confidence)
        self.bbox = [float(value) for value in self.bbox]
        return self


class ContentPackDomainRecord(BaseModel):
    """Common review/gate envelope for manually authored module records."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    ref: str
    content_hash: str = ""
    record_kind: str = "domain"
    visibility: ContentVisibility = "router_hidden"
    title: str = ""
    summary: str = ""
    body: str = ""
    spoiler_class: SpoilerClass = "none"
    reveal_trigger: str = ""
    confidence: float = 1.0
    review_status: ReviewStatus = "unreviewed"
    gate_status: CoverageGateStatus = "flagged"
    gate_reasons: list[str] = Field(default_factory=list)
    provenance: list[ContentProvenance] = Field(default_factory=list)
    coverage_notes: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "ContentPackDomainRecord":
        self.pack_id = self.pack_id.strip()
        self.ref = self.ref.strip()
        self.content_hash = self.content_hash.strip()
        self.record_kind = self.record_kind.strip().lower() or "domain"
        self.title = self.title.strip()
        self.summary = self.summary.strip()
        self.body = self.body.strip()
        self.reveal_trigger = self.reveal_trigger.strip()
        self.coverage_notes = self.coverage_notes.strip()
        self.confidence = _clamp_confidence(self.confidence)
        self.gate_reasons = _clean_unique_strings(self.gate_reasons)
        if self.gate_status == "runtime_ready" and not self.content_hash:
            raise ValueError("runtime_ready records need content_hash")
        if self.gate_status == "runtime_ready" and self.review_status not in {
            "reviewed",
            "approved",
        }:
            raise ValueError("runtime_ready records must be reviewed or approved")
        if (
            self.gate_status == "runtime_ready"
            and self.spoiler_class == "high"
            and not self.reveal_trigger
        ):
            raise ValueError("high spoiler runtime-ready records need reveal_trigger")
        return self


class ContentCrossReference(ContentPackDomainRecord):
    """Stable graph edge between compiled domain records."""

    record_kind: Literal["cross_ref"] = "cross_ref"
    record_ref: str
    target_ref: str
    relation: CrossReferenceRelation = "mentions"
    target_kind: str = ""
    required: bool = True
    external: bool = False
    note: str = ""

    @model_validator(mode="after")
    def _clean_cross_ref(self) -> "ContentCrossReference":
        self.record_ref = self.record_ref.strip()
        self.target_ref = self.target_ref.strip()
        self.target_kind = self.target_kind.strip()
        self.note = self.note.strip()
        return self


class ContentSectionRecord(ContentPackDomainRecord):
    """Reviewed module section boundary without raw source text."""

    record_kind: Literal["section"] = "section"
    section_kind: SectionKind = "section"
    parent_section_ref: str = ""
    ordinal: int = 0
    page_refs: list[str] = Field(default_factory=list)
    child_section_refs: list[str] = Field(default_factory=list)
    span_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean_section(self) -> "ContentSectionRecord":
        self.parent_section_ref = self.parent_section_ref.strip()
        self.ordinal = max(0, int(self.ordinal or 0))
        self.page_refs = _clean_unique_strings(self.page_refs)
        self.child_section_refs = _clean_unique_strings(self.child_section_refs)
        self.span_refs = _clean_unique_strings(self.span_refs)
        return self


class ContentSpanRecord(ContentPackDomainRecord):
    """Reviewed span pointer plus redacted summary."""

    record_kind: Literal["span"] = "span"
    section_ref: str = ""
    page_id: str
    source_span_id: str = ""
    span_role: SpanRole = "body"
    ordinal: int = 0
    bbox: list[float] = Field(default_factory=list)
    redacted_summary: str = ""

    @model_validator(mode="after")
    def _clean_span(self) -> "ContentSpanRecord":
        self.section_ref = self.section_ref.strip()
        self.page_id = self.page_id.strip()
        self.source_span_id = self.source_span_id.strip()
        self.ordinal = max(0, int(self.ordinal or 0))
        self.bbox = [float(value) for value in self.bbox]
        self.redacted_summary = self.redacted_summary.strip()
        return self


class LocationExit(BaseModel):
    """Navigable edge from a location or keyed area."""

    model_config = ConfigDict(extra="forbid")

    exit_id: str
    to_ref: str
    label: str = ""
    travel_mode: str = "walk"
    visible: bool = True
    locked: bool = False
    secret: bool = False
    one_way: bool = False
    requirements: list[str] = Field(default_factory=list)
    reveal_trigger: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "LocationExit":
        self.exit_id = self.exit_id.strip()
        self.to_ref = self.to_ref.strip()
        self.label = self.label.strip()
        self.travel_mode = self.travel_mode.strip().lower() or "walk"
        self.requirements = _clean_unique_strings(self.requirements)
        self.reveal_trigger = self.reveal_trigger.strip()
        if self.secret and not self.reveal_trigger:
            raise ValueError("secret exits need reveal_trigger")
        return self


class LocationRecord(ContentPackDomainRecord):
    """Rules-neutral location record with typed navigation and module links."""

    record_kind: Literal["location"] = "location"
    location_kind: LocationKind = "site"
    keyed_label: str = ""
    parent_location_ref: str = ""
    section_ref: str = ""
    player_arrival_summary: str = ""
    exits: list[LocationExit] = Field(default_factory=list)
    clue_refs: list[str] = Field(default_factory=list)
    secret_refs: list[str] = Field(default_factory=list)
    trap_refs: list[str] = Field(default_factory=list)
    hazard_refs: list[str] = Field(default_factory=list)
    treasure_refs: list[str] = Field(default_factory=list)
    encounter_template_refs: list[str] = Field(default_factory=list)
    handout_refs: list[str] = Field(default_factory=list)
    table_refs: list[str] = Field(default_factory=list)
    map_template_refs: list[str] = Field(default_factory=list)
    front_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean_location(self) -> "LocationRecord":
        self.keyed_label = self.keyed_label.strip()
        self.parent_location_ref = self.parent_location_ref.strip()
        self.section_ref = self.section_ref.strip()
        self.player_arrival_summary = self.player_arrival_summary.strip()
        for field_name in _LOCATION_REF_LIST_FIELDS:
            setattr(self, field_name, _clean_unique_strings(getattr(self, field_name)))
        return self


class KeyedAreaRecord(LocationRecord):
    """Location record for a keyed module/map area."""

    record_kind: Literal["keyed_area"] = "keyed_area"
    location_kind: Literal["keyed_area"] = "keyed_area"
    keyed_label: str

    @model_validator(mode="after")
    def _clean_keyed_area(self) -> "KeyedAreaRecord":
        self.keyed_label = self.keyed_label.strip()
        if not self.keyed_label:
            raise ValueError("keyed areas need keyed_label")
        return self


class RevealGraphEdge(ContentPackDomainRecord):
    """Reveal dependency edge over authored refs."""

    record_kind: Literal["reveal_edge"] = "reveal_edge"
    from_ref: str
    to_ref: str
    relation: RevealRelation = "reveals"
    trigger: str = ""
    audience: RevealAudience = "router"
    required: bool = True

    @model_validator(mode="after")
    def _clean_reveal_edge(self) -> "RevealGraphEdge":
        self.from_ref = self.from_ref.strip()
        self.to_ref = self.to_ref.strip()
        self.trigger = self.trigger.strip()
        return self


class HandoutRevealBlock(BaseModel):
    """Player-safe partial handout reveal."""

    model_config = ConfigDict(extra="forbid")

    reveal_id: str
    trigger: str
    safe_text: str = ""
    safe_asset_ids: list[str] = Field(default_factory=list)
    audience: RevealAudience = "players"

    @model_validator(mode="after")
    def _clean(self) -> "HandoutRevealBlock":
        self.reveal_id = self.reveal_id.strip()
        self.trigger = self.trigger.strip()
        self.safe_text = self.safe_text.strip()
        self.safe_asset_ids = _clean_unique_strings(self.safe_asset_ids)
        return self


class HandoutRecord(ContentPackDomainRecord):
    """Reviewed handout/document record with safe reveal projections."""

    record_kind: Literal["handout"] = "handout"
    handout_kind: HandoutKind = "document"
    safe_asset_ids: list[str] = Field(default_factory=list)
    player_safe_text: str = ""
    player_safe_caption: str = ""
    player_safe_alt_text: str = ""
    possession_ref: str = ""
    reading_constraints: list[str] = Field(default_factory=list)
    partial_reveals: list[HandoutRevealBlock] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean_handout(self) -> "HandoutRecord":
        self.safe_asset_ids = _clean_unique_strings(self.safe_asset_ids)
        self.player_safe_text = self.player_safe_text.strip()
        self.player_safe_caption = self.player_safe_caption.strip()
        self.player_safe_alt_text = self.player_safe_alt_text.strip()
        self.possession_ref = self.possession_ref.strip()
        self.reading_constraints = _clean_unique_strings(self.reading_constraints)
        return self


class AdventureTableRow(BaseModel):
    """One reviewed random/lookup table row."""

    model_config = ConfigDict(extra="forbid")

    row_id: str = ""
    range_start: int | None = None
    range_end: int | None = None
    weight: int = 1
    result_ref: str = ""
    result_summary: str = ""
    reroll: bool = False

    @model_validator(mode="after")
    def _clean(self) -> "AdventureTableRow":
        self.row_id = self.row_id.strip()
        self.result_ref = self.result_ref.strip()
        self.result_summary = self.result_summary.strip()
        self.weight = max(0, int(self.weight or 0))
        if (
            self.range_start is not None
            and self.range_end is not None
            and self.range_end < self.range_start
        ):
            raise ValueError("table row range_end must be >= range_start")
        return self


class AdventureTableRecord(ContentPackDomainRecord):
    """Reviewed table usable by router or adapter lookup."""

    record_kind: Literal["table"] = "table"
    table_kind: AdventureTableKind = "other"
    roll_formula: str = ""
    rows: list[AdventureTableRow] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean_table(self) -> "AdventureTableRecord":
        self.roll_formula = self.roll_formula.strip()
        return self


class GridPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int

    @model_validator(mode="after")
    def _clean(self) -> "GridPoint":
        self.x = max(0, int(self.x or 0))
        self.y = max(0, int(self.y or 0))
        return self


class GridRect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int
    width: int
    height: int

    @model_validator(mode="after")
    def _clean(self) -> "GridRect":
        self.x = max(0, int(self.x or 0))
        self.y = max(0, int(self.y or 0))
        self.width = _positive_int(self.width, "width")
        self.height = _positive_int(self.height, "height")
        return self


class TacticalMapSpawnAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_id: str
    anchor_kind: SpawnAnchorKind = "other"
    cells: list[GridPoint] = Field(default_factory=list)
    label: str = ""
    linked_ref: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "TacticalMapSpawnAnchor":
        self.anchor_id = self.anchor_id.strip()
        self.label = self.label.strip()
        self.linked_ref = self.linked_ref.strip()
        return self


class TacticalMapFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_id: str
    feature_kind: TacticalFeatureKind
    cells: list[GridPoint] = Field(default_factory=list)
    bounds: GridRect | None = None
    label: str = ""
    blocks_movement: bool = False
    blocks_line_of_sight: bool = False
    difficult_terrain: bool = False
    cover: str = ""
    secret: bool = False
    reveal_trigger: str = ""
    linked_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "TacticalMapFeature":
        self.feature_id = self.feature_id.strip()
        self.label = self.label.strip()
        self.cover = self.cover.strip()
        self.reveal_trigger = self.reveal_trigger.strip()
        self.linked_refs = _clean_unique_strings(self.linked_refs)
        if self.secret and not self.reveal_trigger:
            raise ValueError("secret map features need reveal_trigger")
        return self


class TacticalMapAreaLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area_id: str
    location_ref: str
    cells: list[GridPoint] = Field(default_factory=list)
    bounds: GridRect | None = None

    @model_validator(mode="after")
    def _clean(self) -> "TacticalMapAreaLink":
        self.area_id = self.area_id.strip()
        self.location_ref = self.location_ref.strip()
        return self


class TacticalMapTemplateRecord(ContentPackDomainRecord):
    """D&D adapter-owned tactical geometry derived from reviewed map assets."""

    record_kind: Literal["tactical_map_template"] = "tactical_map_template"
    ruleset_id: Literal["dnd5e_basic"] = "dnd5e_basic"
    target_runtime_schema: Literal["DndBattleMapRuntimeState"] = (
        "DndBattleMapRuntimeState"
    )
    map_kind: TacticalMapKind = "battle_map"
    derived_from_map_asset_id: str = ""
    grid_width: int
    grid_height: int
    square_size_ft: int = 5
    origin_notes: str = ""
    orientation: str = ""
    spawn_anchors: list[TacticalMapSpawnAnchor] = Field(default_factory=list)
    terrain_features: list[TacticalMapFeature] = Field(default_factory=list)
    area_links: list[TacticalMapAreaLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean_map_template(self) -> "TacticalMapTemplateRecord":
        self.derived_from_map_asset_id = self.derived_from_map_asset_id.strip()
        self.grid_width = _positive_int(self.grid_width, "grid_width")
        self.grid_height = _positive_int(self.grid_height, "grid_height")
        self.square_size_ft = _positive_int(self.square_size_ft, "square_size_ft")
        self.origin_notes = self.origin_notes.strip()
        self.orientation = self.orientation.strip()
        return self


class FrontClock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clock_id: str
    label: str = ""
    current: int = 0
    maximum: int
    thresholds: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "FrontClock":
        self.clock_id = self.clock_id.strip()
        self.label = self.label.strip()
        self.maximum = _positive_int(self.maximum, "maximum")
        self.current = min(max(0, int(self.current or 0)), self.maximum)
        self.thresholds = {
            str(key).strip(): str(value).strip()
            for key, value in self.thresholds.items()
            if str(key).strip() and str(value).strip()
        }
        return self


class FrontKnowledgeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    trigger: str
    knowledge_delta: str
    source_ref: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "FrontKnowledgeRule":
        self.rule_id = self.rule_id.strip()
        self.trigger = self.trigger.strip()
        self.knowledge_delta = self.knowledge_delta.strip()
        self.source_ref = self.source_ref.strip()
        return self


class FrontActionPaletteEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    action_kind: FrontActionKind
    trigger: str = ""
    cooldown: str = ""
    target_scope: str = ""
    summary: str = ""
    consequence_refs: list[str] = Field(default_factory=list)
    encounter_template_refs: list[str] = Field(default_factory=list)
    statblock_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "FrontActionPaletteEntry":
        self.action_id = self.action_id.strip()
        self.trigger = self.trigger.strip()
        self.cooldown = self.cooldown.strip()
        self.target_scope = self.target_scope.strip()
        self.summary = self.summary.strip()
        self.consequence_refs = _clean_unique_strings(self.consequence_refs)
        self.encounter_template_refs = _clean_unique_strings(
            self.encounter_template_refs
        )
        self.statblock_refs = _clean_unique_strings(self.statblock_refs)
        return self


class FrontDossierRecord(ContentPackDomainRecord):
    """Immutable front/villain pressure dossier."""

    record_kind: Literal["front_dossier"] = "front_dossier"
    villain_refs: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    domain_refs: list[str] = Field(default_factory=list)
    initial_knowledge: list[str] = Field(default_factory=list)
    knowledge_rules: list[FrontKnowledgeRule] = Field(default_factory=list)
    clocks: list[FrontClock] = Field(default_factory=list)
    action_palette: list[FrontActionPaletteEntry] = Field(default_factory=list)
    foreshadowing_refs: list[str] = Field(default_factory=list)
    hard_spoiler_boundaries: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean_front(self) -> "FrontDossierRecord":
        for field_name in (
            "villain_refs",
            "goals",
            "constraints",
            "resources",
            "domain_refs",
            "initial_knowledge",
            "foreshadowing_refs",
            "hard_spoiler_boundaries",
        ):
            setattr(self, field_name, _clean_unique_strings(getattr(self, field_name)))
        return self


class DndAbilityScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strength: int
    dexterity: int
    constitution: int
    intelligence: int
    wisdom: int
    charisma: int

    @model_validator(mode="after")
    def _clean(self) -> "DndAbilityScores":
        for field_name in (
            "strength",
            "dexterity",
            "constitution",
            "intelligence",
            "wisdom",
            "charisma",
        ):
            value = int(getattr(self, field_name) or 10)
            setattr(self, field_name, min(30, max(1, value)))
        return self


class DndModifier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: int

    @model_validator(mode="after")
    def _clean(self) -> "DndModifier":
        self.name = self.name.strip().lower().replace("_", " ")
        self.value = int(self.value or 0)
        return self


class DndDamageExpression(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str
    damage_type: str = ""
    condition: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "DndDamageExpression":
        self.expression = self.expression.strip()
        self.damage_type = self.damage_type.strip().lower()
        self.condition = self.condition.strip()
        return self


class DndRulesFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_id: str
    name: str
    economy: DndEconomyKind = "none"
    attack_bonus: int | None = None
    save_dc: int | None = None
    save_ability: str = ""
    range_ft: int = 0
    reach_ft: int = 0
    target: str = ""
    damage: list[DndDamageExpression] = Field(default_factory=list)
    description: str = ""
    resource_cost: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "DndRulesFeature":
        self.feature_id = self.feature_id.strip()
        self.name = self.name.strip()
        self.save_ability = self.save_ability.strip().lower()
        self.range_ft = max(0, int(self.range_ft or 0))
        self.reach_ft = max(0, int(self.reach_ft or 0))
        self.target = self.target.strip()
        self.description = self.description.strip()
        self.resource_cost = self.resource_cost.strip()
        return self


class DndSpellcastingProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ability: str = ""
    save_dc: int = 0
    attack_bonus: int = 0
    caster_level: int = 0
    spell_slots_by_level: dict[str, int] = Field(default_factory=dict)
    spells: list[str] = Field(default_factory=list)
    at_will: list[str] = Field(default_factory=list)
    limited_uses: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "DndSpellcastingProfile":
        self.ability = self.ability.strip().lower()
        self.save_dc = max(0, int(self.save_dc or 0))
        self.attack_bonus = int(self.attack_bonus or 0)
        self.caster_level = max(0, int(self.caster_level or 0))
        self.spell_slots_by_level = {
            str(key).strip(): max(0, int(value or 0))
            for key, value in self.spell_slots_by_level.items()
            if str(key).strip()
        }
        self.spells = _clean_unique_strings(self.spells)
        self.at_will = _clean_unique_strings(self.at_will)
        self.limited_uses = {
            str(key).strip(): str(value).strip()
            for key, value in self.limited_uses.items()
            if str(key).strip() and str(value).strip()
        }
        return self


class DndStatBlockRecord(ContentPackDomainRecord):
    """D&D adapter-owned statblock catalog row."""

    record_kind: Literal["dnd_statblock"] = "dnd_statblock"
    ruleset_id: Literal["dnd5e_basic"] = "dnd5e_basic"
    automation_scope: AutomationScope = "noncombat_lookup"
    size: str = ""
    creature_type: str = ""
    alignment: str = ""
    armor_class: int = 10
    hit_points: int = 1
    hit_dice: str = ""
    speed_ft_by_mode: dict[str, int] = Field(default_factory=dict)
    ability_scores: DndAbilityScores
    proficiency_bonus: int = 0
    saves: list[DndModifier] = Field(default_factory=list)
    skills: list[DndModifier] = Field(default_factory=list)
    senses: list[str] = Field(default_factory=list)
    passive_perception: int = 10
    languages: list[str] = Field(default_factory=list)
    challenge_rating: str = ""
    xp: int = 0
    damage_resistances: list[str] = Field(default_factory=list)
    damage_immunities: list[str] = Field(default_factory=list)
    damage_vulnerabilities: list[str] = Field(default_factory=list)
    condition_immunities: list[str] = Field(default_factory=list)
    traits: list[DndRulesFeature] = Field(default_factory=list)
    actions: list[DndRulesFeature] = Field(default_factory=list)
    bonus_actions: list[DndRulesFeature] = Field(default_factory=list)
    reactions: list[DndRulesFeature] = Field(default_factory=list)
    legendary_actions: list[DndRulesFeature] = Field(default_factory=list)
    lair_actions: list[DndRulesFeature] = Field(default_factory=list)
    spellcasting: DndSpellcastingProfile | None = None
    parse_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean_statblock(self) -> "DndStatBlockRecord":
        self.size = self.size.strip().lower()
        self.creature_type = self.creature_type.strip().lower()
        self.alignment = self.alignment.strip().lower()
        self.armor_class = max(0, int(self.armor_class or 0))
        self.hit_points = max(1, int(self.hit_points or 1))
        self.hit_dice = self.hit_dice.strip()
        self.speed_ft_by_mode = {
            str(key).strip().lower(): max(0, int(value or 0))
            for key, value in self.speed_ft_by_mode.items()
            if str(key).strip()
        }
        self.proficiency_bonus = int(self.proficiency_bonus or 0)
        self.senses = _clean_unique_strings(self.senses)
        self.passive_perception = max(0, int(self.passive_perception or 0))
        self.languages = _clean_unique_strings(self.languages)
        self.challenge_rating = self.challenge_rating.strip()
        self.xp = max(0, int(self.xp or 0))
        for field_name in (
            "damage_resistances",
            "damage_immunities",
            "damage_vulnerabilities",
            "condition_immunities",
            "parse_warnings",
        ):
            setattr(self, field_name, _clean_unique_strings(getattr(self, field_name)))
        if self.automation_scope == "combat" and not self.actions:
            raise ValueError("combat statblocks need at least one action")
        return self


class TrapHazardMechanics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ruleset_id: Literal["dnd5e_basic"] = "dnd5e_basic"
    detection_dc: int | None = None
    disarm_dc: int | None = None
    save_dc: int | None = None
    save_ability: str = ""
    attack_bonus: int | None = None
    damage: list[DndDamageExpression] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    reset_policy: str = ""
    depletion_ref: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "TrapHazardMechanics":
        self.save_ability = self.save_ability.strip().lower()
        self.conditions = _clean_unique_strings(self.conditions)
        self.reset_policy = self.reset_policy.strip()
        self.depletion_ref = self.depletion_ref.strip()
        return self


class TrapHazardRecord(ContentPackDomainRecord):
    """Trap, hazard, or secret feature with optional D&D mechanics."""

    record_kind: Literal["trap_hazard"] = "trap_hazard"
    trap_hazard_kind: TrapHazardKind = "trap"
    trigger: str = ""
    detection: str = ""
    countermeasures: list[str] = Field(default_factory=list)
    linked_location_refs: list[str] = Field(default_factory=list)
    linked_map_feature_refs: list[str] = Field(default_factory=list)
    mechanics: TrapHazardMechanics | None = None

    @model_validator(mode="after")
    def _clean_trap_hazard(self) -> "TrapHazardRecord":
        self.trigger = self.trigger.strip()
        self.detection = self.detection.strip()
        self.countermeasures = _clean_unique_strings(self.countermeasures)
        self.linked_location_refs = _clean_unique_strings(self.linked_location_refs)
        self.linked_map_feature_refs = _clean_unique_strings(
            self.linked_map_feature_refs
        )
        return self


class TreasureCurrency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    denomination: str
    amount: int

    @model_validator(mode="after")
    def _clean(self) -> "TreasureCurrency":
        self.denomination = self.denomination.strip().lower()
        self.amount = max(0, int(self.amount or 0))
        return self


class TreasureItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_ref: str = ""
    name: str
    quantity: int = 1
    item_type: str = ""
    rarity: str = ""
    value_gp: float = 0.0
    weight_lb: float = 0.0
    requires_attunement: bool = False
    consumable: bool = False
    identified: bool = True

    @model_validator(mode="after")
    def _clean(self) -> "TreasureItem":
        self.item_ref = self.item_ref.strip()
        self.name = self.name.strip()
        self.quantity = max(0, int(self.quantity or 0))
        self.item_type = self.item_type.strip().lower()
        self.rarity = self.rarity.strip().lower()
        self.value_gp = max(0.0, float(self.value_gp or 0.0))
        self.weight_lb = max(0.0, float(self.weight_lb or 0.0))
        return self


class TreasureRecord(ContentPackDomainRecord):
    """Loot or treasure definition with stable depletion refs."""

    record_kind: Literal["treasure"] = "treasure"
    treasure_kind: TreasureKind = "container"
    container_ref: str = ""
    availability: str = ""
    depletion_ref: str = ""
    currency: list[TreasureCurrency] = Field(default_factory=list)
    items: list[TreasureItem] = Field(default_factory=list)
    eligible_character_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean_treasure(self) -> "TreasureRecord":
        self.container_ref = self.container_ref.strip()
        self.availability = self.availability.strip()
        self.depletion_ref = self.depletion_ref.strip()
        self.eligible_character_refs = _clean_unique_strings(
            self.eligible_character_refs
        )
        return self


class EncounterParticipant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str
    statblock_ref: str = ""
    npc_ref: str = ""
    count: int = 1
    role: str = ""
    starting_anchor_ref: str = ""
    tactics: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "EncounterParticipant":
        self.participant_id = self.participant_id.strip()
        self.statblock_ref = self.statblock_ref.strip()
        self.npc_ref = self.npc_ref.strip()
        self.count = _positive_int(self.count, "count")
        self.role = self.role.strip()
        self.starting_anchor_ref = self.starting_anchor_ref.strip()
        self.tactics = self.tactics.strip()
        return self


class EncounterTemplateRecord(ContentPackDomainRecord):
    """Reusable D&D encounter seed without mutable combat state."""

    record_kind: Literal["encounter_template"] = "encounter_template"
    ruleset_id: Literal["dnd5e_basic"] = "dnd5e_basic"
    difficulty: EncounterDifficulty = "unknown"
    trigger: str = ""
    location_refs: list[str] = Field(default_factory=list)
    participants: list[EncounterParticipant] = Field(default_factory=list)
    map_template_refs: list[str] = Field(default_factory=list)
    trap_refs: list[str] = Field(default_factory=list)
    treasure_refs: list[str] = Field(default_factory=list)
    handout_refs: list[str] = Field(default_factory=list)
    front_refs: list[str] = Field(default_factory=list)
    noncombat_resolution: str = ""
    xp_policy: str = ""

    @model_validator(mode="after")
    def _clean_encounter(self) -> "EncounterTemplateRecord":
        self.trigger = self.trigger.strip()
        for field_name in (
            "location_refs",
            "map_template_refs",
            "trap_refs",
            "treasure_refs",
            "handout_refs",
            "front_refs",
        ):
            setattr(self, field_name, _clean_unique_strings(getattr(self, field_name)))
        self.noncombat_resolution = self.noncombat_resolution.strip()
        self.xp_policy = self.xp_policy.strip()
        return self


class ContentPackDomainCatalog(BaseModel):
    """Typed pre-compiler catalog for reviewed manually authored module records."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str
    pack_version: str = ""
    schema_version: str = "content-pack-domain-v1"
    source_fingerprint: str = ""
    build_hash: str = ""
    sections: list[ContentSectionRecord] = Field(default_factory=list)
    spans: list[ContentSpanRecord] = Field(default_factory=list)
    locations: list[LocationRecord] = Field(default_factory=list)
    keyed_areas: list[KeyedAreaRecord] = Field(default_factory=list)
    reveal_edges: list[RevealGraphEdge] = Field(default_factory=list)
    handouts: list[HandoutRecord] = Field(default_factory=list)
    tables: list[AdventureTableRecord] = Field(default_factory=list)
    tactical_map_templates: list[TacticalMapTemplateRecord] = Field(
        default_factory=list
    )
    front_dossiers: list[FrontDossierRecord] = Field(default_factory=list)
    statblocks: list[DndStatBlockRecord] = Field(default_factory=list)
    trap_hazards: list[TrapHazardRecord] = Field(default_factory=list)
    treasures: list[TreasureRecord] = Field(default_factory=list)
    encounter_templates: list[EncounterTemplateRecord] = Field(default_factory=list)
    cross_refs: list[ContentCrossReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean_catalog(self) -> "ContentPackDomainCatalog":
        self.pack_id = self.pack_id.strip()
        self.pack_version = self.pack_version.strip()
        self.schema_version = (
            self.schema_version.strip() or "content-pack-domain-v1"
        )
        self.source_fingerprint = self.source_fingerprint.strip()
        self.build_hash = self.build_hash.strip()

        known_refs: set[str] = set()
        for record in self._domain_records():
            if not record.pack_id:
                record.pack_id = self.pack_id
            if record.ref in known_refs:
                raise ValueError(f"duplicate content pack ref: {record.ref}")
            known_refs.add(record.ref)

        for cross_ref in self.cross_refs:
            if not cross_ref.pack_id:
                cross_ref.pack_id = self.pack_id
            if cross_ref.gate_status == "blocked":
                continue
            if cross_ref.record_ref not in known_refs:
                raise ValueError(
                    f"cross-reference source is not authored: {cross_ref.record_ref}"
                )
            if (
                cross_ref.required
                and not cross_ref.external
                and cross_ref.target_ref not in known_refs
            ):
                raise ValueError(
                    f"required cross-reference target is not authored: "
                    f"{cross_ref.target_ref}"
                )

        for edge in self.reveal_edges:
            if edge.gate_status == "blocked":
                continue
            if edge.from_ref not in known_refs:
                raise ValueError(f"reveal edge source is not authored: {edge.from_ref}")
            if edge.to_ref not in known_refs:
                raise ValueError(f"reveal edge target is not authored: {edge.to_ref}")

        for location in [*self.locations, *self.keyed_areas]:
            if location.gate_status == "blocked":
                continue
            for exit_record in location.exits:
                if exit_record.to_ref not in known_refs:
                    raise ValueError(
                        f"location exit target is not authored: {exit_record.to_ref}"
                    )

        return self

    def _domain_records(self) -> list[ContentPackDomainRecord]:
        return [
            *self.sections,
            *self.spans,
            *self.locations,
            *self.keyed_areas,
            *self.reveal_edges,
            *self.handouts,
            *self.tables,
            *self.tactical_map_templates,
            *self.front_dossiers,
            *self.statblocks,
            *self.trap_hazards,
            *self.treasures,
            *self.encounter_templates,
            *self.cross_refs,
        ]


class PageInventoryRecord(BaseModel):
    """One logical source page in the compiled private pack inventory."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    page_id: str
    source_asset_id: str
    pdf_page_index: int = 0
    printed_page_label: str = ""
    source_sha256: str = ""
    section_id: str = ""
    alignment_status: str = ""
    confidence: float = 1.0
    review_status: ReviewStatus = "unreviewed"
    coverage_status: CoverageGateStatus = "runtime_ready"
    notes: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "PageInventoryRecord":
        self.pack_id = self.pack_id.strip()
        self.page_id = self.page_id.strip()
        self.source_asset_id = self.source_asset_id.strip()
        self.printed_page_label = self.printed_page_label.strip()
        self.source_sha256 = self.source_sha256.strip()
        self.section_id = self.section_id.strip()
        self.alignment_status = self.alignment_status.strip()
        self.notes = self.notes.strip()
        if self.pdf_page_index < 0:
            self.pdf_page_index = 0
        self.confidence = _clamp_confidence(self.confidence)
        return self


class CompiledContentCard(BaseModel):
    """Runtime-readable compiled card, already redacted for the pack layer."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    ref: str
    content_hash: str = ""
    card_kind: str = "content"
    visibility: str = "hidden"
    title: str = ""
    summary: str = ""
    body: str = ""
    spoiler_class: SpoilerClass = "none"
    reveal_trigger: str = ""
    confidence: float = 1.0
    review_status: ReviewStatus = "unreviewed"
    gate_status: CoverageGateStatus = "flagged"
    gate_reasons: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    provenance: list[ContentProvenance] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "CompiledContentCard":
        self.pack_id = self.pack_id.strip()
        self.ref = self.ref.strip()
        self.content_hash = self.content_hash.strip()
        self.card_kind = self.card_kind.strip().lower() or "content"
        self.visibility = self.visibility.strip().lower() or "hidden"
        self.title = self.title.strip()
        self.summary = self.summary.strip()
        self.body = self.body.strip()
        self.reveal_trigger = self.reveal_trigger.strip()
        self.confidence = _clamp_confidence(self.confidence)
        self.gate_reasons = [
            reason.strip()
            for reason in dict.fromkeys(self.gate_reasons)
            if reason.strip()
        ]
        self.aliases = [
            alias.strip()
            for alias in dict.fromkeys(self.aliases)
            if alias.strip()
        ]
        return self


class ContentAliasRecord(BaseModel):
    """Search/catalog alias pointing at a compiled card ref."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    alias: str
    ref: str
    kind: str = "alias"
    confidence: float = 1.0
    review_status: ReviewStatus = "unreviewed"

    @model_validator(mode="after")
    def _clean(self) -> "ContentAliasRecord":
        self.pack_id = self.pack_id.strip()
        self.alias = self.alias.strip()
        self.ref = self.ref.strip()
        self.kind = self.kind.strip().lower() or "alias"
        self.confidence = _clamp_confidence(self.confidence)
        return self


class CoverageGateResult(BaseModel):
    """A deterministic preflight result for serving a compiled record."""

    model_config = ConfigDict(extra="forbid")

    status: CoverageGateStatus
    allowed: bool
    reasons: list[str] = Field(default_factory=list)


class CoverageManifest(BaseModel):
    """Pack-level import coverage summary persisted beside compiled records."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str
    pack_version: str = ""
    source_fingerprint: str = ""
    importer_version: str = ""
    schema_version: str = "content-pack-v1"
    source_page_count: int = 0
    compiled_page_count: int = 0
    card_count: int = 0
    alias_count: int = 0
    ready_count: int = 0
    flagged_count: int = 0
    blocked_count: int = 0
    low_confidence_count: int = 0
    high_spoiler_count: int = 0
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "CoverageManifest":
        self.pack_id = self.pack_id.strip()
        self.pack_version = self.pack_version.strip()
        self.source_fingerprint = self.source_fingerprint.strip()
        self.importer_version = self.importer_version.strip()
        self.schema_version = self.schema_version.strip() or "content-pack-v1"
        self.warnings = [
            warning.strip()
            for warning in dict.fromkeys(self.warnings)
            if warning.strip()
        ]
        for field_name in (
            "source_page_count",
            "compiled_page_count",
            "card_count",
            "alias_count",
            "ready_count",
            "flagged_count",
            "blocked_count",
            "low_confidence_count",
            "high_spoiler_count",
        ):
            if getattr(self, field_name) < 0:
                setattr(self, field_name, 0)
        return self


class ContentImageAsset(BaseModel):
    """Private asset catalog row addressed by stable ids, not file paths."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    asset_id: str
    kind: str = "image"
    title: str = ""
    mime_type: str = ""
    width: int = 0
    height: int = 0
    sha256: str = ""
    source_ref: str = ""
    review_status: ReviewStatus = "unreviewed"
    spoiler_class: SpoilerClass = "none"
    player_safe_alt_text: str = ""
    player_safe_caption: str = ""
    delivery_ref: str = ""
    safe_for_players: bool = False
    safe_for_llm: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "ContentImageAsset":
        self.pack_id = self.pack_id.strip()
        self.asset_id = self.asset_id.strip()
        self.kind = self.kind.strip().lower() or "image"
        self.title = self.title.strip()
        self.mime_type = self.mime_type.strip()
        self.sha256 = self.sha256.strip()
        self.source_ref = self.source_ref.strip()
        self.player_safe_alt_text = self.player_safe_alt_text.strip()
        self.player_safe_caption = self.player_safe_caption.strip()
        self.delivery_ref = self.delivery_ref.strip()
        if self.width < 0:
            self.width = 0
        if self.height < 0:
            self.height = 0
        return self


class AssetReveal(BaseModel):
    """Router-owned image/map reveal request using observable-fact visibility."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    asset_id: str
    audience: AssetRevealAudience = "all_observers"
    visible_to_character_ids: list[str] = Field(default_factory=list)
    visible_to_user_ids: list[str] = Field(default_factory=list)
    presentation: AssetPresentation = "reference"
    caption: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "AssetReveal":
        self.pack_id = self.pack_id.strip()
        self.asset_id = self.asset_id.strip()
        self.visible_to_character_ids = [
            value.strip()
            for value in dict.fromkeys(self.visible_to_character_ids)
            if value.strip()
        ]
        self.visible_to_user_ids = [
            value.strip()
            for value in dict.fromkeys(self.visible_to_user_ids)
            if value.strip()
        ]
        self.caption = self.caption.strip()
        return self


class SafeAssetRevealPayload(BaseModel):
    """Player/LLM-safe reveal payload. No source paths, notes, or bytes."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str
    asset_id: str
    kind: str
    title: str = ""
    mime_type: str = ""
    width: int = 0
    height: int = 0
    sha256: str = ""
    delivery_ref: str
    presentation: AssetPresentation = "reference"
    caption: str = ""
    alt_text: str = ""


_LOCATION_REF_LIST_FIELDS = (
    "clue_refs",
    "secret_refs",
    "trap_refs",
    "hazard_refs",
    "treasure_refs",
    "encounter_template_refs",
    "handout_refs",
    "table_refs",
    "map_template_refs",
    "front_refs",
)


def _clean_unique_strings(values: list[str]) -> list[str]:
    return [value for value in dict.fromkeys(item.strip() for item in values) if value]


def _positive_int(value: int, field_name: str) -> int:
    try:
        cleaned = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be positive") from exc
    if cleaned <= 0:
        raise ValueError(f"{field_name} must be positive")
    return cleaned


def _clamp_confidence(value: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence
