from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.llm.client import LLMResponse
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import (
    DndEventRouterOutput,
    EventRouterOutput,
    ObserverEntry,
    SpawnRequest,
    empty_commitment_open_signal,
)
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import SessionState, StorySetting, WorldState


def _attach_legacy_beat_attrs(event: EventRouterOutput) -> EventRouterOutput:
    legacy_ends_beat = event.event_kind != "beat_continues"
    legacy_reason = event.event_kind if legacy_ends_beat else ""
    object.__setattr__(event, "ends_beat", legacy_ends_beat)
    object.__setattr__(event, "ends_beat_reason", legacy_reason)
    return event


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
    model: str = "claude-haiku-4-5",
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
    model: str = "claude-sonnet-4-6",
    raw_text: str = "{}",
) -> LLMResponse:
    return llm_response(
        parsed=NarratorFinalOutput(final_text=final_text),
        content=raw_text,
        model=model,
        raw_text=raw_text,
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


def router_output(
    *,
    event_id: str = "",
    requires_responders: bool = False,
    required_responders: list[str] | None = None,
    agent_ids: list[str] | None = None,
    observer_ids: list[str] | None = None,
    event_kind: str | None = None,
    ends_beat: bool = True,
    ends_beat_reason: str | None = "directed_at_player",
    facts: list[ObservableFact] | None = None,
    effective_at_s: int = 0,
    duration_s: int = 0,
    location_updates: list[dict] | None = None,
    spawn: list[SpawnRequest] | None = None,
    dormant: list[str] | None = None,
    cull: list[str] | None = None,
) -> EventRouterOutput:
    picks = agent_ids or []
    required = required_responders or []
    observer_ids = observer_ids if observer_ids is not None else [
        "alice", *picks, *required
    ]
    observers: list[ObserverEntry] = []
    seen: set[str] = set()
    for cid in observer_ids:
        if cid in seen:
            continue
        seen.add(cid)
        observers.append(
            ObserverEntry(
                character_id=cid,
                observation_level="d",
                routing_role="next_output" if cid in picks else "observe_only",
            )
        )

    resolved_event_kind = event_kind or (
        "cat_ii_open"
        if requires_responders
        else "beat_continues"
        if not ends_beat
        else (ends_beat_reason or "directed_at_player")
    )
    data: dict[str, Any] = dict(
        event_id=event_id,
        effective_at_s=effective_at_s,
        duration_s=duration_s,
        decision_rationale="test fixture",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(feasible=True),
            observable_facts=facts if facts is not None else [
                ObservableFact.all("Something happens.")
            ],
        ),
        event_kind=resolved_event_kind,
        observers=observers,
        requires_responders=requires_responders,
        required_responders=required,
        spawn=spawn or [],
        dormant=dormant or [],
        cull=cull or [],
        commitment_open=empty_commitment_open_signal(),
        commitment_resolutions=[],
        commitment_interrupts=[],
        location_updates=location_updates or [],
    )
    return _attach_legacy_beat_attrs(EventRouterOutput(**data))


def dnd_router_output(
    *,
    interaction_mode: str = "cat_i",
    combatant_ids: list[str] | None = None,
    combatant_spawns: list[dict] | None = None,
    loot_offer: dict | None = None,
    battle_map_seed: dict | None = None,
    **kwargs: Any,
) -> DndEventRouterOutput:
    data = router_output(**kwargs).model_dump()
    data["interaction_mode"] = interaction_mode
    data["combatant_ids"] = combatant_ids or []
    data["combatant_spawns"] = combatant_spawns or []
    if loot_offer is not None:
        data["loot_offer"] = loot_offer
    if battle_map_seed is not None:
        data["battle_map_seed"] = battle_map_seed
    return _attach_legacy_beat_attrs(DndEventRouterOutput(**data))


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


class InstanceFakeDispatcher:
    def __init__(self):
        self.route_calls: list[dict] = []
        self.continuation_calls: list[dict] = []
        self.agent_output_calls: list[dict] = []
        self.agent_calls: list[dict] = []
        self.agent_character_exists: list[bool] = []
        self.narrator_calls: list[dict] = []
        self.harvest_calls: list[dict] = []
        self.combat_calls: list[dict] = []
        self.materialize_calls: list[dict] = []
        self._route_responses: list[EventRouterOutput] = []
        self._combat_responses: list[EventRouterOutput] = []
        self._agent_responses: list[str] = []
        self._harvest_responses: list[list[str]] = []
        self._narrator_response = "RENDER"

    def queue_route(self, response: EventRouterOutput) -> None:
        self._route_responses.append(response)

    def queue_combat(self, response: EventRouterOutput) -> None:
        self._combat_responses.append(response)

    def queue_agent(self, intention: str) -> None:
        self._agent_responses.append(intention)

    def queue_harvest(self, fragments: list[str]) -> None:
        self._harvest_responses.append(fragments)

    async def route_intention(self, **kw) -> EventRouterOutput:
        self.route_calls.append(kw)
        return self._route_responses.pop(0)

    async def route_continuation(self, **kw) -> EventRouterOutput:
        self.continuation_calls.append(kw)
        return self._route_responses.pop(0)

    async def route_agent_output(self, **kw) -> EventRouterOutput:
        self.agent_output_calls.append(kw)
        return self._route_responses.pop(0)

    async def route_combat_action(self, **kw) -> EventRouterOutput:
        self.combat_calls.append(kw)
        return self._combat_responses.pop(0)

    async def continue_combat_transaction(self, **kw) -> EventRouterOutput:
        self.combat_calls.append(kw)
        return self._combat_responses.pop(0)

    async def agent_intend(self, **kw) -> str:
        self.agent_calls.append(kw)
        character_id = kw.get("character_id", "")
        ckpt = kw.get("ckpt")
        self.agent_character_exists.append(
            any(
                char.character_id == character_id
                for char in getattr(ckpt, "characters", [])
            )
        )
        return self._agent_responses.pop(0)

    async def materialize_spawns(self, **kw) -> list[str]:
        self.materialize_calls.append(kw)
        ckpt = kw["ckpt"]
        result = kw["result"]
        target_ids = set(kw.get("character_ids", []))
        spawned_ids: list[str] = []
        remaining: list[SpawnRequest] = []
        for spawn in result.spawn:
            if spawn.character_id not in target_ids:
                remaining.append(spawn)
                continue
            ckpt.characters.append(
                character_record(
                    spawn.character_id,
                    name=spawn.seed.role or spawn.character_id,
                    role=spawn.seed.role,
                    location=spawn.seed.location or "gatehouse",
                )
            )
            spawned_ids.append(spawn.character_id)
        result.spawn = remaining
        return spawned_ids

    async def harvest_perceptions(self, **kw) -> list[str]:
        self.harvest_calls.append(kw)
        if self._harvest_responses:
            return self._harvest_responses.pop(0)
        return ["" for _ in kw.get("character_ids", [])]

    async def narrator_compose(self, **kw):
        self.narrator_calls.append(kw)
        envelope = NarratorFinalOutput(final_text=self._narrator_response)
        entry = TranscriptEntry(
            user=kw.get("user_input", ""),
            assistant=self._narrator_response,
        )
        return envelope, entry


class ClassFakeDispatcher:
    _route_responses: list[EventRouterOutput] = []
    _agent_responses: list[str] = []
    _narrator_errors: list[Exception] = []
    _narrator_text: str = "POV_RENDER"
    route_calls: list[dict] = []
    agent_calls: list[dict] = []
    narrator_calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def reset(cls) -> None:
        cls._route_responses = []
        cls._agent_responses = []
        cls._narrator_errors = []
        cls._narrator_text = "POV_RENDER"
        cls.route_calls = []
        cls.agent_calls = []
        cls.narrator_calls = []

    @classmethod
    def queue_route(cls, response: EventRouterOutput) -> None:
        cls._route_responses.append(response)

    @classmethod
    def queue_agent(cls, intention: str) -> None:
        cls._agent_responses.append(intention)

    @classmethod
    def queue_narrator_error(cls, error: Exception) -> None:
        cls._narrator_errors.append(error)

    async def route_intention(self, **kw) -> EventRouterOutput:
        type(self).route_calls.append(kw)
        return type(self)._route_responses.pop(0)

    async def route_continuation(self, **kw) -> EventRouterOutput:
        type(self).route_calls.append(kw)
        return type(self)._route_responses.pop(0)

    async def route_combat_action(self, **kw) -> EventRouterOutput:
        type(self).route_calls.append(kw)
        return type(self)._route_responses.pop(0)

    async def continue_combat_transaction(self, **kw) -> EventRouterOutput:
        type(self).route_calls.append(kw)
        return type(self)._route_responses.pop(0)

    async def agent_intend(self, **kw) -> str:
        type(self).agent_calls.append(kw)
        return type(self)._agent_responses.pop(0)

    async def narrator_compose(self, **kw):
        type(self).narrator_calls.append(kw)
        if type(self)._narrator_errors:
            raise type(self)._narrator_errors.pop(0)
        envelope = NarratorFinalOutput(final_text=type(self)._narrator_text)
        entry = TranscriptEntry(
            user=kw.get("user_input", ""),
            assistant=type(self)._narrator_text,
        )
        return envelope, entry
