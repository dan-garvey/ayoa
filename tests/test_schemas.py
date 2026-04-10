"""Tests for all Pydantic schemas — construct from design doc examples, round-trip, reject invalid."""

import pytest
from pydantic import ValidationError

from app.schemas.state import (
    ModelConfig, SessionConfig, SessionState, TimeState,
    LocationState, PhysicsRuleset, WorldState,
)
from app.schemas.characters import (
    CharacterRecord, CharacterStatus, PublicSheet, PrivateState, CharacterMemory,
)
from app.schemas.events import CanonicalEvent, WorldAdjudication, SceneDelta
from app.schemas.discriminator import DiscriminatorOutput, ObserverEntry, SpawnRequest
from app.schemas.agents import CharacterAgentOutput, PublicResponse, PrivateUpdates
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.requests import TurnRequest, DebugFlags
from app.schemas.responses import TurnResponse, DebugPayload
from app.schemas.checkpoint import CheckpointFile


# --- Design doc example data (sections 7.1-7.7) ---

WORLD_STATE_EXAMPLE = {
    "time": {"scene_time": "2026-04-10T15:21:00Z", "turn_count": 42},
    "locations": {"current_scene_id": "estate_courtyard", "scene_graph": {}},
    "facts": ["The courtyard is wet from earlier rain.", "The main building is made of stone."],
    "physics_ruleset": {"strength_limits": "human_baseline", "magic_enabled": False},
    "global_flags": {},
}

CHARACTER_EXAMPLE = {
    "character_id": "guard_17",
    "name": "Captain Vero",
    "status": "active",
    "location": "estate_courtyard",
    "public_sheet": {
        "role": "guard captain",
        "traits": ["disciplined", "dry humor"],
        "voice": "clipped and formal",
    },
    "private_state": {
        "goals": ["maintain order"],
        "attitudes": {"user": -0.1},
        "intentions_enabled": True,
    },
    "memory": {"episodic": [], "summaries": []},
}

CANONICAL_EVENT_EXAMPLE = {
    "event_id": "evt_0042",
    "user_intent": "lift the building with bare hands",
    "world_adjudication": {
        "attempted_action": "user attempts impossible feat",
        "feasible": False,
        "resolved_outcome": "visible failed strain with no structural movement",
    },
    "scene_delta": {"time_advanced_seconds": 6},
    "observable_facts": [
        "The user braces against the building.",
        "The building does not move.",
        "The user visibly strains.",
    ],
}

DISCRIMINATOR_EXAMPLE = {
    "event_id": "evt_0042",
    "observers": [
        {
            "character_id": "guard_17",
            "observation_level": "direct",
            "facts": [
                "The user braces against the building.",
                "The building does not move.",
                "The user visibly strains.",
            ],
            "should_respond": True,
        }
    ],
    "spawn": [],
    "dormant": [],
    "cull": [],
}

AGENT_OUTPUT_EXAMPLE = {
    "character_id": "guard_17",
    "public_response": {
        "actions": ["takes one step closer"],
        "dialogue": ["Need a lever, not a miracle."],
        "expression": "one brow lifts",
    },
    "private_updates": {
        "intentions": ["monitor the user more closely"],
        "attitude_delta": {"user": -0.05},
    },
    "memory_writes": ["Observed the user attempt an impossible physical feat and fail."],
}

NARRATOR_FINAL_EXAMPLE = {
    "final_text": (
        "You plant both palms against the stone and drive upward until your arms shake. "
        'Nothing gives. Rainwater slicks beneath your boots. Captain Vero steps closer, '
        'one brow raised. "Need a lever, not a miracle."'
    ),
    "world_updates": {"time_advanced_seconds": 6},
    "transcript_entry": {
        "user": "I lift the building with my bare hands.",
        "assistant": "You plant both palms...",
    },
    "turn_summary": "User attempted impossible physical action; failed publicly; nearby guard observed and commented.",
}


# --- Construction from design doc examples ---

class TestWorldState:
    def test_construct(self):
        ws = WorldState(**WORLD_STATE_EXAMPLE)
        assert ws.locations.current_scene_id == "estate_courtyard"
        assert ws.physics_ruleset.magic_enabled is False
        assert len(ws.facts) == 2

    def test_round_trip(self):
        ws = WorldState(**WORLD_STATE_EXAMPLE)
        rebuilt = WorldState(**ws.model_dump())
        assert rebuilt == ws

    def test_defaults(self):
        ws = WorldState()
        assert ws.facts == []
        assert ws.physics_ruleset.strength_limits == "human_baseline"


class TestCharacterRecord:
    def test_construct(self):
        cr = CharacterRecord(**CHARACTER_EXAMPLE)
        assert cr.name == "Captain Vero"
        assert cr.status == CharacterStatus.active
        assert "dry humor" in cr.public_sheet.traits
        assert cr.private_state.attitudes["user"] == pytest.approx(-0.1)

    def test_round_trip(self):
        cr = CharacterRecord(**CHARACTER_EXAMPLE)
        rebuilt = CharacterRecord(**cr.model_dump())
        assert rebuilt == cr

    def test_invalid_status(self):
        data = {**CHARACTER_EXAMPLE, "status": "exploded"}
        with pytest.raises(ValidationError):
            CharacterRecord(**data)


class TestCanonicalEvent:
    def test_construct(self):
        ce = CanonicalEvent(**CANONICAL_EVENT_EXAMPLE)
        assert ce.user_intent == "lift the building with bare hands"
        assert ce.world_adjudication.feasible is False
        assert ce.scene_delta.time_advanced_seconds == 6
        assert len(ce.observable_facts) == 3

    def test_round_trip(self):
        ce = CanonicalEvent(**CANONICAL_EVENT_EXAMPLE)
        rebuilt = CanonicalEvent(**ce.model_dump())
        assert rebuilt == ce

    def test_rejects_extra_fields(self):
        data = {**CANONICAL_EVENT_EXAMPLE, "rogue_field": "surprise"}
        with pytest.raises(ValidationError):
            CanonicalEvent(**data)


class TestDiscriminatorOutput:
    def test_construct(self):
        do = DiscriminatorOutput(**DISCRIMINATOR_EXAMPLE)
        assert len(do.observers) == 1
        assert do.observers[0].character_id == "guard_17"
        assert do.observers[0].should_respond is True

    def test_round_trip(self):
        do = DiscriminatorOutput(**DISCRIMINATOR_EXAMPLE)
        rebuilt = DiscriminatorOutput(**do.model_dump())
        assert rebuilt == do

    def test_rejects_extra_fields_on_observer(self):
        bad_observer = {**DISCRIMINATOR_EXAMPLE["observers"][0], "secret": "leaked"}
        data = {**DISCRIMINATOR_EXAMPLE, "observers": [bad_observer]}
        with pytest.raises(ValidationError):
            DiscriminatorOutput(**data)

    def test_spawn_request(self):
        data = {
            **DISCRIMINATOR_EXAMPLE,
            "spawn": [{"character_id": "stablehand_03", "seed": {"role": "stablehand"}}],
        }
        do = DiscriminatorOutput(**data)
        assert do.spawn[0].character_id == "stablehand_03"


class TestCharacterAgentOutput:
    def test_construct(self):
        ao = CharacterAgentOutput(**AGENT_OUTPUT_EXAMPLE)
        assert ao.character_id == "guard_17"
        assert "Need a lever, not a miracle." in ao.public_response.dialogue
        assert ao.private_updates.attitude_delta["user"] == pytest.approx(-0.05)

    def test_round_trip(self):
        ao = CharacterAgentOutput(**AGENT_OUTPUT_EXAMPLE)
        rebuilt = CharacterAgentOutput(**ao.model_dump())
        assert rebuilt == ao

    def test_rejects_extra_fields(self):
        data = {**AGENT_OUTPUT_EXAMPLE, "hidden_thoughts": "I know everything"}
        with pytest.raises(ValidationError):
            CharacterAgentOutput(**data)


class TestNarratorFinalOutput:
    def test_construct(self):
        nfo = NarratorFinalOutput(**NARRATOR_FINAL_EXAMPLE)
        assert "Captain Vero" in nfo.final_text
        assert nfo.transcript_entry.user == "I lift the building with my bare hands."
        assert nfo.turn_summary != ""

    def test_round_trip(self):
        nfo = NarratorFinalOutput(**NARRATOR_FINAL_EXAMPLE)
        rebuilt = NarratorFinalOutput(**nfo.model_dump())
        assert rebuilt == nfo

    def test_rejects_extra_fields(self):
        data = {**NARRATOR_FINAL_EXAMPLE, "chain_of_thought": "secret"}
        with pytest.raises(ValidationError):
            NarratorFinalOutput(**data)


class TestTurnRequest:
    def test_construct(self):
        tr = TurnRequest(session_id="abc", user_input="I look around.")
        assert tr.stream is False
        assert tr.debug is False
        assert tr.debug_flags.include_discriminator is False

    def test_full(self):
        tr = TurnRequest(
            session_id="abc",
            checkpoint_id="ckpt_0001",
            user_input="I try the door.",
            stream=True,
            debug=True,
            debug_flags=DebugFlags(include_discriminator=True, include_agent_outputs=True),
        )
        assert tr.stream is True
        assert tr.debug_flags.include_discriminator is True


class TestTurnResponse:
    def test_normal_mode(self):
        tr = TurnResponse(session_id="abc", output_text="You look around.")
        assert tr.debug is None

    def test_debug_mode(self):
        tr = TurnResponse(
            session_id="abc",
            output_text="You look around.",
            debug=DebugPayload(canonical_event={"event_id": "evt_001"}),
        )
        assert tr.debug is not None
        assert tr.debug.canonical_event["event_id"] == "evt_001"


class TestCheckpointFile:
    def test_construct(self):
        ckpt = CheckpointFile(
            session=SessionState(session_id="test-session"),
            world_state=WorldState(**WORLD_STATE_EXAMPLE),
            characters=[CharacterRecord(**CHARACTER_EXAMPLE)],
        )
        assert ckpt.schema_version == "1.0"
        assert ckpt.session.session_id == "test-session"
        assert len(ckpt.characters) == 1
        assert ckpt.prompt_versions["narrator"] == "v1"

    def test_round_trip(self):
        ckpt = CheckpointFile(
            session=SessionState(session_id="test-session"),
            world_state=WorldState(**WORLD_STATE_EXAMPLE),
            characters=[CharacterRecord(**CHARACTER_EXAMPLE)],
            transcript=[TranscriptEntry(user="hello", assistant="world")],
        )
        rebuilt = CheckpointFile(**ckpt.model_dump())
        assert rebuilt == ckpt

    def test_defaults(self):
        ckpt = CheckpointFile(session=SessionState(session_id="minimal"))
        assert ckpt.characters == []
        assert ckpt.transcript == []
        assert ckpt.visibility_log == []
