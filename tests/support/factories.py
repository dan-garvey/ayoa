from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.llm.client import LLMResponse
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import (
    CanonicalEventRecord,
    DndCanonicalEventRecord,
    DndRouterEventDraft,
    ObserverGroups,
    RouterBatchOutput,
    RouterEventDraft,
)
from app.schemas.dnd_inventory import DndCurrency, DndLootOfferSignal
from app.schemas.dnd_spatial import DndBattleMapSeed
from app.schemas.delivery import NarratorEventRef
from app.schemas.events import ObservableFact
from app.schemas.narrator import NarratorFinalOutput
from app.schemas.state import SessionState, StorySetting, WorldState


def text_block(text: str = "{}") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    block.model_dump = lambda: {"type": "text", "text": text}
    return block


def llm_response(
    parsed: Any = None,
    *,
    content: str | None = None,
    model: str = "gpt-5.2",
    raw_text: str | None = None,
) -> LLMResponse:
    response_text = content
    if response_text is None:
        response_text = (
            parsed.model_dump_json()
            if hasattr(parsed, "model_dump_json")
            else "{}"
        )
    raw = MagicMock()
    raw.model = model
    raw.content = [] if raw_text is None else [text_block(raw_text)]
    return LLMResponse(
        parsed=parsed,
        raw_response=raw,
        content=response_text,
        model=model,
    )


def text_llm_response(
    text: str,
    *,
    model: str = "gpt-5.6-luna",
) -> LLMResponse:
    return llm_response(
        parsed=None,
        content=text,
        model=model,
        raw_text=text,
    )


def narrator_llm_response(
    final_text: str = "RENDERED",
    *,
    handoff: str = "render",
    handoff_reason: str = "The visible sequence is ready.",
    model: str = "claude-sonnet-5",
    raw_text: str = "{}",
) -> LLMResponse:
    return llm_response(
        parsed=NarratorFinalOutput(
            handoff=handoff,
            handoff_reason=handoff_reason,
            final_text=final_text,
        ),
        content=raw_text,
        model=model,
        raw_text=raw_text,
    )


def narrator_event_ref(
    *,
    event_id: str,
    observation_level: str = "direct",
    visible_at_s: int = 0,
    event_sequence: int = 0,
) -> NarratorEventRef:
    return NarratorEventRef(
        event_id=event_id,
        observation_level=observation_level,
        visible_at_s=visible_at_s,
        event_sequence=event_sequence,
        sprite_variant_keys_by_character_id={},
    )


def character_record(
    character_id: str,
    *,
    name: str | None = None,
    role: str = "npc",
    location: str = "gatehouse",
    is_playable: bool = False,
    public_sheet: PublicSheet | None = None,
    mechanics: dict | None = None,
    **overrides: Any,
) -> CharacterRecord:
    data = dict(
        character_id=character_id,
        name=name or character_id.title(),
        public_sheet=public_sheet or PublicSheet(role=role),
        location=location,
        is_playable=is_playable,
    )
    if mechanics is not None:
        data["mechanics"] = mechanics
    data.update(overrides)
    return CharacterRecord(**data)


def checkpoint(
    *,
    session_id: str = "s",
    turn_index: int = 0,
    bindings: dict[str, str] | None = None,
    player_character_id: str | None = None,
    characters: list[CharacterRecord] | None = None,
    world_state: WorldState | None = None,
    **overrides: Any,
) -> CheckpointFile:
    session_data: dict[str, Any] = dict(
        session_id=session_id,
        turn_index=turn_index,
        character_bindings=bindings or {},
    )
    if player_character_id is not None:
        session_data["player_character_id"] = player_character_id
    return CheckpointFile(
        session=SessionState(**session_data),
        world_state=world_state or WorldState(),
        characters=characters or [],
        **overrides,
    )


def gatehouse_checkpoint(
    *,
    bindings: dict[str, str] | None = None,
    player_character_id: str | None = None,
    turn_index: int = 0,
    pip_role: str = "npc",
    world_setting: StorySetting | None = None,
) -> CheckpointFile:
    return checkpoint(
        turn_index=turn_index,
        bindings=bindings,
        player_character_id=player_character_id,
        world_state=WorldState(setting=world_setting) if world_setting else None,
        characters=[
            character_record(
                "alice",
                name="Alice",
                role="player",
                is_playable=True,
            ),
            character_record(
                "bob",
                name="Bob",
                role="player",
                is_playable=True,
            ),
            character_record(
                "pip",
                name="Pip",
                role=pip_role,
                is_playable=False,
            ),
        ],
    )


def router_event_draft(
    *,
    feasible_input_indexes: list[int] | None = None,
    infeasible_input_indexes: list[int] | None = None,
    observer_ids: list[str] | None = None,
    facts: list[ObservableFact] | None = None,
    duration_s: int = 0,
    required_responders: list[str] | None = None,
    **overrides: Any,
) -> RouterEventDraft:
    observers = ["alice"] if observer_ids is None else observer_ids
    data: dict[str, Any] = {
        "feasible_input_indexes": (
            [0] if feasible_input_indexes is None else feasible_input_indexes
        ),
        "infeasible_input_indexes": (
            [] if infeasible_input_indexes is None else infeasible_input_indexes
        ),
        "duration_s": duration_s,
        "observable_facts": (
            [ObservableFact.all("Something happens.")] if facts is None else facts
        ),
        "observers": ObserverGroups(direct=observers, indirect=[], inferred=[]),
        "required_responders": (
            [] if required_responders is None else required_responders
        ),
        "appearance_target_ids": [],
        "spawn": [],
        "dormant": [],
        "cull": [],
        "commitment_opens": [],
        "commitment_resolutions": [],
        "commitment_interrupts": [],
        "location_updates": [],
        "activate": [],
    }
    data.update(overrides)
    return RouterEventDraft.model_validate(data)


def router_batch_output(
    *,
    events: list[RouterEventDraft] | None = None,
    next_turns: list[dict[str, Any]] | None = None,
) -> RouterBatchOutput:
    return RouterBatchOutput(
        events=events or [router_event_draft()],
        next_turns=next_turns or [],
    )


def dnd_router_event_draft(**overrides: Any) -> DndRouterEventDraft:
    data = router_event_draft().model_dump()
    data.update({
        "interaction_mode": "narrative",
        "combatant_ids": [],
        "combatant_spawns": [],
        "loot_offer": DndLootOfferSignal(
            present=False,
            source_kind="other",
            source_label="",
            visibility="table",
            eligible_character_ids=[],
            items=[],
            currency=DndCurrency(cp=0, sp=0, ep=0, gp=0, pp=0),
            notes="",
        ),
        "battle_map_seed": DndBattleMapSeed.model_validate({}),
        "dnd_reaction_ids": [],
    })
    data.update(overrides)
    return DndRouterEventDraft.model_validate(data)


def canonical_event(
    *,
    event_id: str = "evt_test",
    lane_id: str = "lane_test",
    actor_ids: list[str] | None = None,
    observer_ids: list[str] | None = None,
    facts: list[ObservableFact | dict[str, Any]] | None = None,
    effective_at_s: int = 0,
    duration_s: int = 0,
    **overrides: Any,
) -> CanonicalEventRecord:
    observers = ["alice"] if observer_ids is None else observer_ids
    normalized_facts = (
        [ObservableFact.all("Something happens.")]
        if facts is None
        else [
            fact
            if isinstance(fact, ObservableFact)
            else ObservableFact.all(
                str(fact.get("text", "")),
                visual_subject_ids=fact.get("visual_subject_ids", ()),
                at_offset_s=int(fact.get("at_offset_s", 0)),
                duration_s=int(fact.get("duration_s", 0)),
            )
            for fact in facts
        ]
    )
    data: dict[str, Any] = {
        "event_id": event_id,
        "causal_lane_id": lane_id,
        "effective_at_s": effective_at_s,
        "duration_s": duration_s,
        "actor_ids": [] if actor_ids is None else actor_ids,
        "source_submission_ids": ["submission_test"],
        "feasible_submission_ids": ["submission_test"],
        "infeasible_submission_ids": [],
        "observable_facts": normalized_facts,
        "observers": ObserverGroups(direct=observers, indirect=[], inferred=[]),
        "spawn": [],
        "dormant": [],
        "cull": [],
        "commitment_opens": [],
        "commitment_resolutions": [],
        "commitment_interrupts": [],
        "location_updates": [],
        "activate": [],
    }
    data.update(overrides)
    return CanonicalEventRecord.model_validate(data)


def dnd_canonical_event(
    *,
    event_id: str = "evt_test",
    lane_id: str = "lane_test",
    actor_ids: list[str] | None = None,
    observer_ids: list[str] | None = None,
    facts: list[ObservableFact | dict[str, Any]] | None = None,
    effective_at_s: int = 0,
    duration_s: int = 0,
    **overrides: Any,
) -> DndCanonicalEventRecord:
    data = canonical_event(
        event_id=event_id,
        lane_id=lane_id,
        actor_ids=actor_ids,
        observer_ids=observer_ids,
        facts=facts,
        effective_at_s=effective_at_s,
        duration_s=duration_s,
    ).model_dump()
    data.update({
        "interaction_mode": "narrative",
        "combatant_ids": [],
        "combatant_spawns": [],
        "loot_offer": DndLootOfferSignal(
            present=False,
            source_kind="other",
            source_label="",
            visibility="table",
            eligible_character_ids=[],
            items=[],
            currency=DndCurrency(cp=0, sp=0, ep=0, gp=0, pp=0),
            notes="",
        ),
        "battle_map_seed": DndBattleMapSeed.model_validate({}),
    })
    data.update(overrides)
    return DndCanonicalEventRecord.model_validate(data)



def dnd5e_mechanics(
    *,
    xp: int = 0,
    hp: int = 10,
    ac: int = 12,
    ability_scores: dict[str, int] | None = None,
    proficiency_bonus: int = 2,
    skill_proficiencies: list[str] | None = None,
    name: str = "Alice",
) -> dict:
    abilities = {
        "str": 10,
        "dex": 10,
        "con": 10,
        "int": 10,
        "wis": 10,
        "cha": 10,
    }
    if ability_scores:
        abilities.update(ability_scores)
    mechanics = {
        "ruleset_id": "dnd5e_basic",
        "experience_points": xp,
        "armor_class": ac,
        "hit_points": {"current": hp, "max": hp, "temporary": 0},
        "ability_scores": abilities,
        "proficiency_bonus": proficiency_bonus,
        "dnd5e_sheet": {
            "identity": {
                "name": name,
                "total_level": 1,
                "experience_points": xp,
                "classes": [{"name": "Fighter", "level": 1}],
            },
            "statblock": {},
        },
    }
    if skill_proficiencies is not None:
        mechanics["skill_proficiencies"] = skill_proficiencies
    return mechanics
