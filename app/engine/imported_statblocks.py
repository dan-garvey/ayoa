from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Any

from app.engine import dnd_combat, dnd_monsters
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterVisuals,
    PublicSheet,
)
from app.schemas.content_pack import (
    ContentPackDomainCatalog,
    DndDamageExpression,
    DndModifier,
    DndRulesFeature,
    DndSpellcastingProfile,
    DndStatBlockRecord,
)
from app.schemas.content_privacy import (
    redact_imported_asset_text,
    sanitize_player_safe_text,
)
from app.schemas.dnd_monsters import (
    DndCombatantSpawn,
    DndMonsterAction,
    DndMonsterStatBlock,
)
from app.schemas.state import DndCombatantState, SessionState


APPROVED_REVIEW_STATUSES = {"reviewed", "approved"}
COMBAT_REQUIRED_EXPLICIT_FIELDS = frozenset({
    "armor_class",
    "hit_points",
    "speed_ft_by_mode",
    "ability_scores",
    "actions",
})
ABILITY_KEYS = {
    "strength": "str",
    "dexterity": "dex",
    "constitution": "con",
    "intelligence": "int",
    "wisdom": "wis",
    "charisma": "cha",
}


class ImportedStatBlockError(ValueError):
    """Base error for reviewed imported statblock resolution."""


class ImportedStatBlockNotFoundError(
    ImportedStatBlockError,
    dnd_monsters.StatblockResolutionError,
):
    """Raised when a requested statblock ref is not in the catalog."""


class ImportedStatBlockValidationError(
    ImportedStatBlockError,
    dnd_monsters.StatblockResolutionError,
):
    """Raised when an imported statblock is not safe to use for combat."""


@dataclass(frozen=True)
class ImportedStatBlockSpawnSpec:
    """Ref-based request to materialize a reviewed imported combatant."""

    statblock_ref: str
    character_id: str
    name: str = ""
    location: str = ""
    description: str = ""
    tactics_refs: tuple[str, ...] = field(default_factory=tuple)


class ImportedStatBlockCatalog:
    """In-memory index of reviewed D&D statblock pack records."""

    def __init__(
        self,
        records: Iterable[DndStatBlockRecord | Mapping[str, Any]],
    ) -> None:
        self._records: dict[str, DndStatBlockRecord] = {}
        self._aliases: dict[str, str] = {}
        self._explicit_fields: dict[str, frozenset[str]] = {}
        for raw in records:
            record, explicit_fields = _coerce_record_with_fields(raw)
            ref = record.ref.strip()
            if not ref:
                raise ImportedStatBlockValidationError(
                    "Imported D&D statblock record is missing ref."
                )
            if ref in self._records:
                raise ImportedStatBlockValidationError(
                    f"Duplicate imported D&D statblock ref: {ref}"
                )
            self._records[ref] = record
            self._explicit_fields[ref] = explicit_fields
            for alias in (ref, _clean_id(ref)):
                if not alias:
                    continue
                existing = self._aliases.get(alias)
                if existing and existing != ref:
                    raise ImportedStatBlockValidationError(
                        "Imported D&D statblock aliases collide: "
                        f"{alias} maps to both {existing} and {ref}."
                    )
                self._aliases[alias] = ref

    @classmethod
    def from_domain_catalog(
        cls,
        catalog: ContentPackDomainCatalog | Mapping[str, Any],
    ) -> "ImportedStatBlockCatalog":
        if isinstance(catalog, Mapping):
            pack_id = str(catalog.get("pack_id") or "").strip()
            records = []
            for raw in catalog.get("statblocks") or ():
                if isinstance(raw, Mapping) and pack_id and not raw.get("pack_id"):
                    records.append({**raw, "pack_id": pack_id})
                else:
                    records.append(raw)
            return cls(records)
        return cls(catalog.statblocks)

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(self._records)

    def get(self, ref: str) -> DndStatBlockRecord:
        cleaned = str(ref or "").strip()
        canonical_ref = self._aliases.get(cleaned) or self._aliases.get(
            _clean_id(cleaned)
        )
        if not cleaned or canonical_ref not in self._records:
            raise ImportedStatBlockNotFoundError(
                f"Imported D&D statblock ref is not authored: {cleaned or '<empty>'}"
            )
        return self._records[canonical_ref]

    def resolve_monster_statblock(self, ref: str) -> DndMonsterStatBlock:
        canonical_ref = self._canonical_ref(ref)
        return monster_statblock_from_record(
            self._records[canonical_ref],
            explicit_fields=self._explicit_fields[canonical_ref],
        )

    def resolve_character(
        self,
        spec: ImportedStatBlockSpawnSpec,
        *,
        default_location: str = "",
    ) -> CharacterRecord:
        canonical_ref = self._canonical_ref(spec.statblock_ref)
        return character_from_statblock_record(
            self._records[canonical_ref],
            character_id=spec.character_id,
            name=spec.name,
            location=spec.location or default_location,
            description=spec.description,
            tactics_refs=spec.tactics_refs,
            explicit_fields=self._explicit_fields[canonical_ref],
        )

    def resolve_combatant(
        self,
        spec: ImportedStatBlockSpawnSpec,
        *,
        session: SessionState | None = None,
        default_location: str = "",
    ) -> DndCombatantState:
        character = self.resolve_character(spec, default_location=default_location)
        return dnd_combat.build_combatant(character, session=session)

    def _canonical_ref(self, ref: str) -> str:
        cleaned = str(ref or "").strip()
        canonical_ref = self._aliases.get(cleaned) or self._aliases.get(
            _clean_id(cleaned)
        )
        if not cleaned or canonical_ref not in self._records:
            raise ImportedStatBlockNotFoundError(
                f"Imported D&D statblock ref is not authored: {cleaned or '<empty>'}"
            )
        return canonical_ref


def statblock_override_provider(
    catalog: ImportedStatBlockCatalog,
) -> Callable[[DndCombatantSpawn], DndMonsterStatBlock | None]:
    """Return an existing dnd_monsters provider keyed by spawn ref."""

    def _provider(spawn: DndCombatantSpawn) -> DndMonsterStatBlock | None:
        ref = str(spawn.statblock_ref or spawn.monster_key or "").strip()
        if not ref:
            return None
        try:
            catalog.get(ref)
        except ImportedStatBlockNotFoundError:
            return None
        return catalog.resolve_monster_statblock(ref)

    return _provider


def catalog_from_content_state(content_state: Any) -> ImportedStatBlockCatalog | None:
    """Build a statblock catalog from checkpoint content-pack runtime metadata."""

    if not isinstance(content_state, Mapping):
        return None
    records: list[DndStatBlockRecord | Mapping[str, Any]] = []
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        metadata = _pack_metadata(pack_state)
        records.extend(_metadata_statblock_records(metadata, pack_id=pack_id))
    if not records:
        return None
    return ImportedStatBlockCatalog(records)


def resolve_spawn_character_from_content_state(
    spawn: DndCombatantSpawn,
    *,
    content_state: Any,
    default_location: str = "",
) -> CharacterRecord | None:
    """Resolve a ref-only combatant spawn from checkpoint content state.

    Returns None for ordinary inline-statblock spawns. Ref-bearing spawns are
    required to resolve through a reviewed imported statblock catalog.
    """

    ref = str(spawn.statblock_ref or "").strip()
    if not ref:
        return None
    catalog = catalog_from_content_state(content_state)
    if catalog is None:
        raise ImportedStatBlockNotFoundError(
            "Imported D&D statblock ref cannot be resolved because no runtime "
            f"statblock catalog is available: {ref}"
        )
    return catalog.resolve_character(
        ImportedStatBlockSpawnSpec(
            statblock_ref=ref,
            character_id=spawn.character_id,
            name=spawn.name,
            location=spawn.location,
            description=spawn.description,
        ),
        default_location=default_location,
    )


def character_from_statblock_record(
    record: DndStatBlockRecord | Mapping[str, Any],
    *,
    character_id: str,
    name: str = "",
    location: str = "",
    description: str = "",
    tactics_refs: Sequence[str] = (),
    explicit_fields: Set[str] | None = None,
) -> CharacterRecord:
    statblock = _validated_combat_record(record, explicit_fields=explicit_fields)
    role = " ".join(
        part for part in (statblock.size, statblock.creature_type) if part
    ).strip()
    safe_name = (
        _player_safe_text(name)
        or _player_safe_text(statblock.title)
        or statblock.ref
    )
    visible_description = (
        _player_safe_text(description)
        or _player_safe_text(statblock.summary)
        or role
        or statblock.creature_type
    )
    return CharacterRecord(
        character_id=_clean_id(character_id),
        name=safe_name,
        location=_player_safe_text(location),
        is_playable=False,
        agent_tier=CharacterAgentTier.utility,
        public_sheet=PublicSheet(
            role=role,
            public_context=visible_description,
        ),
        visuals=CharacterVisuals(default_loadout=visible_description),
        mechanics=mechanics_from_statblock_record(
            statblock,
            tactics_refs=tactics_refs,
        ),
    )


def mechanics_from_statblock_record(
    record: DndStatBlockRecord | Mapping[str, Any],
    *,
    tactics_refs: Sequence[str] = (),
    explicit_fields: Set[str] | None = None,
) -> dict[str, Any]:
    statblock = _validated_combat_record(record, explicit_fields=explicit_fields)
    scores = _flat_scores(statblock)
    detailed_scores = {
        key: {"score": value, "modifier": _ability_modifier(value)}
        for key, value in scores.items()
    }
    defenses = _defenses(statblock, detailed_scores)
    actions = _action_records(statblock.actions, default_economy="action")
    bonus_actions = _action_records(
        statblock.bonus_actions,
        default_economy="bonus_action",
    )
    reactions = _action_records(statblock.reactions, default_economy="reaction")
    legendary_actions = _action_records(
        statblock.legendary_actions,
        default_economy="legendary_action",
    )
    lair_actions = _action_records(
        statblock.lair_actions,
        default_economy="lair_action",
    )
    all_actions = [
        *actions,
        *bonus_actions,
        *reactions,
        *legendary_actions,
        *lair_actions,
    ]
    imported_metadata = {
        "ref": statblock.ref,
        "pack_id": statblock.pack_id,
        "content_hash": statblock.content_hash,
        "automation_scope": statblock.automation_scope,
        "tactics_refs": _clean_refs(tactics_refs),
    }
    sheet = {
        "ruleset_id": "dnd5e_basic",
        "identity": {
            "name": _safe_text(statblock.title),
            "total_level": 0,
            "experience_points": 0,
            "classes": [],
        },
        "statblock": {
            "source": "imported_statblock_catalog",
            "imported": imported_metadata,
            "size": statblock.size,
            "creature_type": statblock.creature_type,
            "alignment": statblock.alignment,
            "speed": _speed_text(statblock.speed_ft_by_mode),
            "speed_ft_by_mode": dict(statblock.speed_ft_by_mode),
            "ability_scores": detailed_scores,
            "proficiency_bonus": statblock.proficiency_bonus,
            "skills": _modifier_map(statblock.skills),
            "saves": _modifier_map(statblock.saves, ability_aliases=True),
            "defenses": defenses,
            "senses": [_safe_text(sense) for sense in statblock.senses],
            "passive_perception": statblock.passive_perception,
            "languages": [_safe_text(language) for language in statblock.languages],
            "challenge": {
                "rating": statblock.challenge_rating,
                "xp": statblock.xp,
            },
            "challenge_rating": statblock.challenge_rating,
            "xp": statblock.xp,
            "features": [
                _feature_record(trait, kind="trait")
                for trait in statblock.traits
                if trait.name or trait.description
            ],
            "traits": [
                _feature_record(trait, kind="trait")
                for trait in statblock.traits
                if trait.name or trait.description
            ],
            "actions": all_actions,
            "bonus_actions": bonus_actions,
            "reactions": reactions,
            "legendary_actions": legendary_actions,
            "lair_actions": lair_actions,
            "spellcasting": _spellcasting(statblock.spellcasting),
        },
    }
    hit_points = defenses["hit_points"]
    return {
        "ruleset_id": "dnd5e_basic",
        "source": "imported_statblock_catalog",
        "imported_statblock": imported_metadata,
        "monster_key": statblock.ref,
        "ability_scores": scores,
        "proficiency_bonus": statblock.proficiency_bonus,
        "armor_class": statblock.armor_class,
        "hit_points": hit_points,
        "conditions": [],
        "defenses": defenses,
        "challenge_rating": statblock.challenge_rating,
        "xp_value": statblock.xp,
        "dnd5e_sheet": sheet,
    }


def monster_statblock_from_record(
    record: DndStatBlockRecord | Mapping[str, Any],
    *,
    explicit_fields: Set[str] | None = None,
) -> DndMonsterStatBlock:
    """Project an imported record into the legacy router-spawn schema."""

    statblock = _validated_combat_record(record, explicit_fields=explicit_fields)
    actions = [
        _monster_action_from_feature(feature)
        for feature in statblock.actions
        if feature.name or feature.description
    ]
    return DndMonsterStatBlock(
        size=statblock.size,
        creature_type=statblock.creature_type,
        alignment=statblock.alignment,
        armor_class=statblock.armor_class,
        hit_points=statblock.hit_points,
        hit_dice=statblock.hit_dice,
        speed=_speed_text(statblock.speed_ft_by_mode),
        ability_scores={
            "strength": statblock.ability_scores.strength,
            "dexterity": statblock.ability_scores.dexterity,
            "constitution": statblock.ability_scores.constitution,
            "intelligence": statblock.ability_scores.intelligence,
            "wisdom": statblock.ability_scores.wisdom,
            "charisma": statblock.ability_scores.charisma,
        },
        proficiency_bonus=statblock.proficiency_bonus,
        skills=[
            {"name": modifier.name, "value": modifier.value}
            for modifier in statblock.skills
        ],
        senses=list(statblock.senses),
        passive_perception=statblock.passive_perception,
        languages=list(statblock.languages),
        challenge_rating=statblock.challenge_rating,
        xp=statblock.xp,
        traits=[
            {"name": trait.name, "description": _safe_text(trait.description)}
            for trait in statblock.traits
        ],
        actions=actions,
    )


def _validated_combat_record(
    record: DndStatBlockRecord | Mapping[str, Any],
    *,
    explicit_fields: Set[str] | None = None,
) -> DndStatBlockRecord:
    if explicit_fields is None and isinstance(record, Mapping):
        explicit_fields = frozenset(str(key) for key in record)
    statblock = _coerce_record(record)
    problems = _combat_validation_problems(statblock, explicit_fields=explicit_fields)
    if problems:
        joined = ", ".join(problems)
        raise ImportedStatBlockValidationError(
            f"Imported D&D statblock {statblock.ref!r} is not combat-ready: "
            f"{joined}."
        )
    return statblock


def _combat_validation_problems(
    statblock: DndStatBlockRecord,
    *,
    explicit_fields: Set[str] | None = None,
) -> list[str]:
    problems: list[str] = []
    if statblock.automation_scope != "combat":
        problems.append(f"automation_scope:{statblock.automation_scope}")
    if statblock.gate_status != "runtime_ready":
        problems.append(f"gate_status:{statblock.gate_status}")
    if statblock.review_status not in APPROVED_REVIEW_STATUSES:
        problems.append(f"review_status:{statblock.review_status}")
    if not statblock.content_hash:
        problems.append("content_hash")
    fields_set = set(explicit_fields or statblock.model_fields_set)
    for field_name in sorted(COMBAT_REQUIRED_EXPLICIT_FIELDS - fields_set):
        problems.append(field_name)
    if statblock.armor_class <= 0:
        problems.append("armor_class")
    if statblock.hit_points <= 0:
        problems.append("hit_points")
    if not statblock.speed_ft_by_mode:
        problems.append("speed_ft_by_mode")
    if not statblock.actions:
        problems.append("actions")
    return list(dict.fromkeys(problems))


def _coerce_record(
    record: DndStatBlockRecord | Mapping[str, Any],
) -> DndStatBlockRecord:
    if isinstance(record, DndStatBlockRecord):
        return record
    return DndStatBlockRecord.model_validate(record)


def _coerce_record_with_fields(
    record: DndStatBlockRecord | Mapping[str, Any],
) -> tuple[DndStatBlockRecord, frozenset[str]]:
    if isinstance(record, DndStatBlockRecord):
        return record, frozenset(record.model_fields_set)
    return (
        DndStatBlockRecord.model_validate(record),
        frozenset(str(key) for key in record),
    )


def _pack_metadata(pack_state: Any) -> Mapping[str, Any]:
    metadata = (
        pack_state.get("metadata")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "metadata", None)
    )
    return metadata if isinstance(metadata, Mapping) else {}


def _pack_id(pack_key: Any, pack_state: Any) -> str:
    raw = (
        pack_state.get("pack_id")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "pack_id", "")
    )
    return str(raw or pack_key or "").strip()


def _metadata_statblock_records(
    metadata: Mapping[str, Any],
    *,
    pack_id: str,
) -> list[DndStatBlockRecord | Mapping[str, Any]]:
    records: list[DndStatBlockRecord | Mapping[str, Any]] = []
    for key in ("domain_catalog", "content_pack_domain_catalog"):
        raw_catalog = metadata.get(key)
        if raw_catalog is None:
            continue
        if isinstance(raw_catalog, ContentPackDomainCatalog):
            records.extend(raw_catalog.statblocks)
        elif isinstance(raw_catalog, Mapping):
            raw_records = raw_catalog.get("statblocks") or ()
            records.extend(_records_with_pack(raw_records, pack_id=pack_id))
    for key in ("statblocks", "dnd_statblocks", "imported_statblocks"):
        records.extend(_records_with_pack(metadata.get(key) or (), pack_id=pack_id))
    return records


def _records_with_pack(
    raw_records: Any,
    *,
    pack_id: str,
) -> list[DndStatBlockRecord | Mapping[str, Any]]:
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        return []
    records: list[DndStatBlockRecord | Mapping[str, Any]] = []
    for raw in raw_records:
        if isinstance(raw, Mapping) and pack_id and not raw.get("pack_id"):
            records.append({**raw, "pack_id": pack_id})
        elif isinstance(raw, (DndStatBlockRecord, Mapping)):
            records.append(raw)
    return records


def _defenses(
    statblock: DndStatBlockRecord,
    detailed_scores: dict[str, dict[str, int]],
) -> dict[str, Any]:
    dex = detailed_scores["dex"]["modifier"]
    return {
        "armor_class": {
            "value": statblock.armor_class,
            "calculation": "imported_statblock",
        },
        "hit_points": {
            "current": statblock.hit_points,
            "max": statblock.hit_points,
            "temporary": 0,
            "formula": statblock.hit_dice,
        },
        "initiative": {
            "ability": "dex",
            "value": dex,
            "advantage_state": "normal",
        },
        "movement": {
            mode: {"value": value, "unit": "ft"}
            for mode, value in statblock.speed_ft_by_mode.items()
        },
        "senses": {
            "special": [
                {"id": _slug(sense), "name": _safe_text(sense)}
                for sense in statblock.senses
            ],
        },
        "damage_resistances": _defense_entries(statblock.damage_resistances),
        "damage_immunities": _defense_entries(statblock.damage_immunities),
        "damage_vulnerabilities": _defense_entries(statblock.damage_vulnerabilities),
        "condition_immunities": _defense_entries(statblock.condition_immunities),
        "conditions": [],
    }


def _defense_entries(values: Sequence[str]) -> list[dict[str, str]]:
    return [
        {"id": _slug(value), "name": _safe_text(value)}
        for value in values
        if _safe_text(value)
    ]


def _flat_scores(statblock: DndStatBlockRecord) -> dict[str, int]:
    return {
        "str": statblock.ability_scores.strength,
        "dex": statblock.ability_scores.dexterity,
        "con": statblock.ability_scores.constitution,
        "int": statblock.ability_scores.intelligence,
        "wis": statblock.ability_scores.wisdom,
        "cha": statblock.ability_scores.charisma,
    }


def _modifier_map(
    modifiers: Sequence[DndModifier],
    *,
    ability_aliases: bool = False,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for modifier in modifiers:
        names = [modifier.name]
        if ability_aliases:
            alias = ABILITY_KEYS.get(modifier.name)
            if alias:
                names.append(alias)
        for name in names:
            if name:
                out[name] = {"value": modifier.value}
    return out


def _action_records(
    features: Sequence[DndRulesFeature],
    *,
    default_economy: str,
) -> list[dict[str, Any]]:
    return [
        _action_record(feature, default_economy=default_economy)
        for feature in features
        if feature.name or feature.description
    ]


def _action_record(
    feature: DndRulesFeature,
    *,
    default_economy: str,
) -> dict[str, Any]:
    economy = feature.economy if feature.economy != "none" else default_economy
    damage = _damage_records(feature.damage)
    attack: dict[str, Any] = {
        "reach_ft": feature.reach_ft,
        "range_normal_ft": feature.range_ft,
        "range_long_ft": feature.range_ft,
        "target": _safe_text(feature.target),
        "damage": _damage_summary(feature.damage),
    }
    if feature.attack_bonus is not None:
        attack["bonus"] = feature.attack_bonus
    save: dict[str, Any] = {}
    if feature.save_dc is not None:
        save["dc"] = feature.save_dc
    if feature.save_ability:
        save["ability"] = feature.save_ability
    consumes = []
    if feature.resource_cost:
        consumes.append({
            "resource_id": _slug(feature.resource_cost),
            "amount": 1,
            "text": _safe_text(feature.resource_cost),
        })
    return {
        "id": _slug(feature.feature_id or feature.name),
        "name": _safe_text(feature.name),
        "kind": "attack" if feature.attack_bonus is not None else economy,
        "activation": {"type": economy},
        "attack": attack,
        "save": save,
        "range": {
            "normal_ft": feature.range_ft,
            "long_ft": feature.range_ft,
            "reach_ft": feature.reach_ft,
        },
        "target": {"text": _safe_text(feature.target)},
        "damage": damage,
        "healing": [],
        "consumes": consumes,
        "description": _safe_text(feature.description),
        "source_refs": [],
    }


def _feature_record(feature: DndRulesFeature, *, kind: str) -> dict[str, Any]:
    return {
        "id": _slug(feature.feature_id or feature.name),
        "name": _safe_text(feature.name),
        "kind": kind,
        "activation": {"type": feature.economy},
        "description": _safe_text(feature.description),
        "source_refs": [],
    }


def _monster_action_from_feature(feature: DndRulesFeature) -> DndMonsterAction:
    first_damage = feature.damage[0] if feature.damage else None
    return DndMonsterAction(
        action_id=_slug(feature.feature_id or feature.name),
        name=_safe_text(feature.name),
        attack_bonus=int(feature.attack_bonus or 0),
        reach_ft=feature.reach_ft,
        range_normal_ft=feature.range_ft,
        range_long_ft=feature.range_ft,
        target=_safe_text(feature.target),
        damage=_damage_summary(feature.damage),
        damage_type=(first_damage.damage_type if first_damage else ""),
        description=_safe_text(feature.description),
    )


def _damage_records(
    damage: Sequence[DndDamageExpression],
) -> list[dict[str, str]]:
    return [
        {
            "formula": _safe_text(component.expression),
            "damage_type": _safe_text(component.damage_type),
            "condition": _safe_text(component.condition),
        }
        for component in damage
        if component.expression
    ]


def _damage_summary(damage: Sequence[DndDamageExpression]) -> str:
    parts = []
    for component in damage:
        expression = _safe_text(component.expression)
        if not expression:
            continue
        damage_type = _safe_text(component.damage_type)
        parts.append(f"{expression} {damage_type}".strip())
    return " + ".join(parts)


def _spellcasting(profile: DndSpellcastingProfile | None) -> dict[str, Any]:
    if profile is None:
        return {}
    spells = list(dict.fromkeys([*profile.spells, *profile.at_will]))
    return {
        "profiles": [
            {
                "id": "imported_spellcasting",
                "name": "Spellcasting",
                "ability": profile.ability,
                "spell_attack_bonus": profile.attack_bonus,
                "spell_save_dc": profile.save_dc,
                "caster_level": profile.caster_level,
            }
        ],
        "slots": {
            level: {"current": count, "max": count}
            for level, count in profile.spell_slots_by_level.items()
        },
        "pact_slots": None,
        "spells": [
            {
                "id": _slug(spell),
                "name": _safe_text(spell),
                "level": 0,
                "prepared": True,
                "always_prepared": spell in profile.at_will,
                "concentration": False,
                "attack": {},
                "save": {},
                "damage": [],
                "healing": [],
                "consumes": [],
            }
            for spell in spells
            if _safe_text(spell)
        ],
        "at_will": [_safe_text(spell) for spell in profile.at_will],
        "limited_uses": {
            _safe_text(name): _safe_text(value)
            for name, value in profile.limited_uses.items()
        },
    }


def _speed_text(speed_ft_by_mode: Mapping[str, int]) -> str:
    parts = [
        f"{mode} {value} ft."
        for mode, value in speed_ft_by_mode.items()
        if value > 0
    ]
    if not parts and speed_ft_by_mode:
        parts = [
            f"{mode} {value} ft."
            for mode, value in speed_ft_by_mode.items()
        ]
    return ", ".join(parts)


def _clean_refs(values: Sequence[str]) -> list[str]:
    return [
        ref
        for ref in dict.fromkeys(str(value or "").strip() for value in values)
        if ref
    ]


def _clean_id(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _safe_text(value: object) -> str:
    return redact_imported_asset_text(str(value or ""))


def _player_safe_text(value: object) -> str:
    return sanitize_player_safe_text(str(value or ""))


def _ability_modifier(score: int) -> int:
    return (int(score) - 10) // 2
