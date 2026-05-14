"""Tests for all Pydantic schemas — construct from design doc examples, round-trip, reject invalid."""

import pytest
from pydantic import ValidationError

from app.schemas.state import (
    ModelConfig, SessionConfig, SessionState, TimeState,
    LocationState, PhysicsRuleset, StorySetting, WorldState,
)
from app.schemas.characters import (
    CharacterRecord, CharacterStatus, PublicSheet, PrivateState,
)
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    WorldAdjudication,
    visible_fact_texts,
)
from app.schemas.event_router import (
    DndEventRouterOutput,
    EventRouterOutput,
    ObserverEntry,
    SpawnRequest,
)
from app.schemas.dnd_cat_ii import RulesAdjudication
from app.schemas.agents import CharacterAgentOutput
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.requests import TurnRequest
from app.schemas.responses import TurnResponse
from app.schemas.checkpoint import CheckpointFile


# --- Design doc example data (sections 7.1-7.7) ---

WORLD_STATE_EXAMPLE = {
    "time": {"story_time": "2026-04-10T15:21:00Z", "turn_count": 42},
    "locations": {},
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
    },
    "private_state": {
        "goals": ["maintain order"],
        "intentions_enabled": True,
    },
    "pending_observations": [],
}

CANONICAL_EVENT_EXAMPLE = {
    "world_adjudication": {
        "feasible": False,
    },
    "observable_facts": [
        "The user braces against the building.",
        "The building does not move.",
        "The user visibly strains.",
    ],
}

# Every field is REQUIRED on EventRouterOutput / nested types — see
# the schema-shape policy in app/schemas/event_router.py. The LLM
# emits explicit empty values; this example mirrors that contract.
ROUTER_OUTPUT_EXAMPLE = {
    "event_id": "",
    "effective_at_s": 0,
    "duration_s": 0,
    "decision_rationale": "(test fixture)",
    "canonical_event": {
        "world_adjudication": {
            "feasible": False,
        },
        "observable_facts": [],
    },
    "requires_responders": False,
    "required_responders": [],
    "agent_responder_picks": [],
    "ends_beat": True,
    "ends_beat_reason": "",
    "observers": [
        {
            "character_id": "guard_17",
            "observation_level": "d",
            "response_priority": 5,
        }
    ],
    "spawn": [],
    "dormant": [],
    "cull": [],
    "commitment_open": {
        "present": False,
        "actor_ids": [],
        "description": "",
        "expected_duration_s": 0,
        "max_duration_s": 0,
        "location_label": "",
    },
    "commitment_resolutions": [],
    "commitment_interrupts": [],
    "location_updates": [],
}

AGENT_OUTPUT_EXAMPLE = {
    "character_id": "guard_17",
    "public_text": (
        'He takes one step closer, one brow lifting. '
        '"Need a lever, not a miracle."'
    ),
    "intent": "Monitor the user more closely; this attempt was theatrical.",
}

NARRATOR_FINAL_EXAMPLE = {
    "final_text": (
        "You plant both palms against the stone and drive upward until your arms shake. "
        'Nothing gives. Rainwater slicks beneath your boots. Captain Vero steps closer, '
        'one brow raised. "Need a lever, not a miracle."'
    ),
}


# --- Construction from design doc examples ---

class TestWorldState:
    def test_construct(self):
        ws = WorldState(**WORLD_STATE_EXAMPLE)
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
        assert ws.lore == ""
        assert ws.setting.genre == ""

    def test_rich_world(self):
        ws = WorldState(
            setting=StorySetting(
                genre="fantasy",
                era="post-war covenant era",
                tone="dark political intrigue",
                premise="Academy for future leaders of warring races",
            ),
            lore="Three centuries ago, the major races nearly destroyed each other...",
            facts=["Article Nineteen prohibits cross-racial procreation."],
            physics_ruleset=PhysicsRuleset(magic_enabled=True),
        )
        assert ws.setting.genre == "fantasy"
        assert "Article Nineteen" in ws.facts[0]
        assert ws.physics_ruleset.magic_enabled is True
        assert "Three centuries" in ws.lore


class TestCharacterRecord:
    def test_construct(self):
        cr = CharacterRecord(**CHARACTER_EXAMPLE)
        assert cr.name == "Captain Vero"
        assert cr.status == CharacterStatus.active
        assert cr.public_sheet.role == "guard captain"
        assert "maintain order" in cr.private_state.goals

    def test_round_trip(self):
        cr = CharacterRecord(**CHARACTER_EXAMPLE)
        rebuilt = CharacterRecord(**cr.model_dump())
        assert rebuilt == cr

    def test_invalid_status(self):
        data = {**CHARACTER_EXAMPLE, "status": "exploded"}
        with pytest.raises(ValidationError):
            CharacterRecord(**data)

    def test_rich_character(self):
        cr = CharacterRecord(
            character_id="ashara_01",
            name="Ashara vel Kothren",
            location="garvey_house",
            public_sheet=PublicSheet(
                role="demon seat heir-designate",
                appearance="Tall—6'1\" before the horns. Deep red skin, molten gold eyes.",
                faction="House vel Kothren",
            ),
            private_state=PrivateState(
                goals=["become the greatest demon seat-holder", "lift demon restrictions"],
                secrets=["grandmother was involved in the human collapse"],
            ),
            backstory="Born to House vel Kothren. Won the Trials of Ascension at seventeen...",
            personality="Confident without cruelty. Meritocratic to a fault. Her tail is her honest voice — flicks when irritated, curls when pleased.",
        )
        assert "grandmother" in cr.private_state.secrets[0]
        assert "Trials of Ascension" in cr.backstory
        assert "tail" in cr.personality
        assert cr.public_sheet.faction == "House vel Kothren"

    def test_legacy_is_player_alias_migrated_to_is_playable(self):
        """playable-2 renamed `is_player` to `is_playable`. Old saves on
        disk were written under the previous name — the model_validator
        on CharacterRecord must map them on load so legacy checkpoints
        still hydrate. This pins the back-compat contract called out in
        the field docstring."""
        legacy = {**CHARACTER_EXAMPLE, "is_player": True}
        cr = CharacterRecord(**legacy)
        assert cr.is_playable is True
        # The old key is consumed; it does not survive on the model.
        assert not hasattr(cr, "is_player")

    def test_legacy_is_player_false_also_migrates(self):
        """The False case matters too — without the alias it would
        silently default to False and look indistinguishable from a
        broken load."""
        legacy = {**CHARACTER_EXAMPLE, "is_player": False}
        cr = CharacterRecord(**legacy)
        assert cr.is_playable is False

    def test_explicit_is_playable_wins_over_legacy_alias(self, caplog):
        """If both keys appear and disagree (shouldn't happen in real
        checkpoints, but a hand-edited save could do it), the new name
        wins and the bridge logs a warning so the operator knows the
        legacy key was discarded."""
        import logging
        conflicting = {**CHARACTER_EXAMPLE, "is_player": False, "is_playable": True}
        with caplog.at_level(logging.WARNING, logger="app.schemas.characters"):
            cr = CharacterRecord(**conflicting)
        assert cr.is_playable is True
        assert any("is_player" in rec.message and "is_playable" in rec.message
                   for rec in caplog.records)


class TestCanonicalEvent:
    def test_construct(self):
        ce = CanonicalEvent(**CANONICAL_EVENT_EXAMPLE)
        assert ce.world_adjudication.feasible is False
        assert len(ce.observable_facts) == 3
        assert all(f.audience == "all_observers" for f in ce.observable_facts)

    def test_round_trip(self):
        ce = CanonicalEvent(**CANONICAL_EVENT_EXAMPLE)
        rebuilt = CanonicalEvent(**ce.model_dump())
        assert rebuilt == ce

    def test_legacy_audit_fields_are_dropped(self):
        data = {
            **CANONICAL_EVENT_EXAMPLE,
            "world_adjudication": {
                **CANONICAL_EVENT_EXAMPLE["world_adjudication"],
                "attempted_action": "legacy intent frame",
                "resolved_outcome": "legacy audit line",
            },
        }
        ce = CanonicalEvent(**data)
        dumped = ce.model_dump()
        assert "attempted_action" not in dumped["world_adjudication"]
        assert "resolved_outcome" not in dumped["world_adjudication"]

    def test_rejects_extra_fields(self):
        data = {**CANONICAL_EVENT_EXAMPLE, "rogue_field": "surprise"}
        with pytest.raises(ValidationError):
            CanonicalEvent(**data)

    def test_fact_level_visibility_filters_by_character(self):
        facts = [
            ObservableFact.all("The public question lands at the table."),
            ObservableFact.only(
                "Dan's foot touches Ashara's boot under the table.",
                ["ashara"],
            ),
        ]

        assert visible_fact_texts(facts, "ashara") == [
            "The public question lands at the table.",
            "Dan's foot touches Ashara's boot under the table.",
        ]
        assert visible_fact_texts(facts, "aldric") == [
            "The public question lands at the table.",
        ]
        assert visible_fact_texts(
            facts, "ashara", include_all_observers=False,
        ) == [
            "Dan's foot touches Ashara's boot under the table.",
        ]

    def test_scoped_fact_requires_visible_recipient(self):
        with pytest.raises(ValidationError):
            ObservableFact(text="quiet signal", audience="only", visible_to=[])


class TestEventRouterOutput:
    def test_json_schema_is_openai_strict_object_compatible(self):
        """OpenAI strict structured outputs reject nested object schemas
        unless every object explicitly sets additionalProperties=false."""
        schema = EventRouterOutput.model_json_schema()
        missing = []

        def walk(node, path=()):
            if isinstance(node, dict):
                if (
                    node.get("type") == "object"
                    and node.get("additionalProperties") is not False
                ):
                    missing.append(path)
                for key, value in node.items():
                    walk(value, path + (key,))
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, path + (str(i),))

        walk(schema)
        assert missing == []

    def test_construct(self):
        r = EventRouterOutput(**ROUTER_OUTPUT_EXAMPLE)
        assert len(r.observers) == 1
        assert r.observers[0].character_id == "guard_17"
        assert r.observers[0].response_priority == 5
        assert r.observers[0].observation_level == "d"

    def test_round_trip(self):
        r = EventRouterOutput(**ROUTER_OUTPUT_EXAMPLE)
        rebuilt = EventRouterOutput(**r.model_dump())
        assert rebuilt == r

    def test_rejects_extra_fields_on_observer(self):
        bad_observer = {**ROUTER_OUTPUT_EXAMPLE["observers"][0], "secret": "leaked"}
        data = {**ROUTER_OUTPUT_EXAMPLE, "observers": [bad_observer]}
        with pytest.raises(ValidationError):
            EventRouterOutput(**data)

    def test_scoped_fact_visible_to_must_be_observer(self):
        data = {
            **ROUTER_OUTPUT_EXAMPLE,
            "canonical_event": {
                **ROUTER_OUTPUT_EXAMPLE["canonical_event"],
                "observable_facts": [
                    {
                        "text": "Dan's foot touches Ashara's boot.",
                        "audience": "only",
                        "visible_to": ["ashara"],
                    }
                ],
            },
        }
        with pytest.raises(ValidationError):
            EventRouterOutput(**data)

    def test_spawn_request(self):
        data = {
            **ROUTER_OUTPUT_EXAMPLE,
            "spawn": [{"character_id": "stablehand_03", "seed": {"role": "stablehand"}}],
        }
        r = EventRouterOutput(**data)
        assert r.spawn[0].character_id == "stablehand_03"

    def test_empty_internal_time_commitment_signals_are_clamped_not_rejected(self):
        data = {
            **ROUTER_OUTPUT_EXAMPLE,
            "effective_at_s": -5,
            "duration_s": -10,
            "commitment_open": {
                "present": True,
                "actor_ids": [],
                "description": "wait",
                "expected_duration_s": -1,
                "max_duration_s": -1,
                "location_label": "",
            },
            "commitment_resolutions": [
                {
                    "commitment_id": "",
                    "actor_ids": [],
                    "reason": "",
                    "resolved_at_offset_s": -1,
                }
            ],
            "commitment_interrupts": [
                {
                    "commitment_id": "",
                    "actor_ids": [],
                    "observed_at_offset_s": -1,
                    "reason": "",
                }
            ],
            "location_updates": [
                {"character_id": "", "location_label": "gatehouse"},
                {"character_id": "alice", "location_label": ""},
            ],
        }

        rebuilt = EventRouterOutput.model_validate(data)

        assert rebuilt.effective_at_s == 0
        assert rebuilt.duration_s == 0
        assert rebuilt.commitment_open.expected_duration_s == 0
        assert rebuilt.commitment_open.max_duration_s == 0
        assert rebuilt.commitment_resolutions[0].resolved_at_offset_s == 0
        assert rebuilt.commitment_interrupts[0].observed_at_offset_s == 0
        assert rebuilt.location_updates == []

        data["duration_s"] = 10
        data["commitment_resolutions"][0]["resolved_at_offset_s"] = 999
        data["commitment_interrupts"][0]["observed_at_offset_s"] = 999
        rebuilt = EventRouterOutput.model_validate(data)
        assert rebuilt.commitment_resolutions[0].resolved_at_offset_s == 10
        assert rebuilt.commitment_interrupts[0].observed_at_offset_s == 10

    def test_cat_ii_open_does_not_advance_duration(self):
        data = {
            **ROUTER_OUTPUT_EXAMPLE,
            "duration_s": 20,
            "requires_responders": True,
            "required_responders": ["guard_17"],
            "ends_beat_reason": "cat_ii_open",
            "canonical_event": {
                **ROUTER_OUTPUT_EXAMPLE["canonical_event"],
                "observable_facts": [
                    {
                        "text": "Alice's fist drives toward Pip.",
                        "audience": "all_observers",
                        "visible_to": [],
                        "at_offset_s": 10,
                        "duration_s": 5,
                    }
                ],
            },
        }

        rebuilt = EventRouterOutput.model_validate(data)

        assert rebuilt.duration_s == 0
        fact = rebuilt.canonical_event.observable_facts[0]
        assert fact.at_offset_s == 0
        assert fact.duration_s == 0


class TestDndEventRouterOutput:
    def test_json_schema_is_openai_strict_object_compatible(self):
        schema = DndEventRouterOutput.model_json_schema()
        missing = []

        def walk(node, path=()):
            if isinstance(node, dict):
                if (
                    node.get("type") == "object"
                    and node.get("additionalProperties") is not False
                ):
                    missing.append(path)
                for key, value in node.items():
                    walk(value, path + (key,))
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, path + (str(i),))

        walk(schema)
        assert missing == []

    def test_combat_start_clamps_cat_ii_fields(self):
        data = {
            **ROUTER_OUTPUT_EXAMPLE,
            "interaction_mode": "dnd_combat_start",
            "combatant_ids": ["alice", "pip"],
            "requires_responders": True,
            "required_responders": ["pip"],
        }

        out = DndEventRouterOutput(**data)

        assert out.requires_responders is False
        assert out.required_responders == []
        assert out.combatant_ids == ["alice", "pip"]
        assert out.loot_offer.present is False

    def test_dnd_loot_offer_contract(self):
        data = {
            **ROUTER_OUTPUT_EXAMPLE,
            "interaction_mode": "cat_i",
            "combatant_ids": [],
            "loot_offer": {
                "present": True,
                "source_kind": "container",
                "source_label": "iron chest",
                "visibility": "table",
                "eligible_character_ids": ["alice"],
                "items": [
                    {
                        "item_id": "healing_potion",
                        "name": "Potion of Healing",
                        "kind": "consumable",
                        "quantity": 1,
                        "identified": True,
                        "requires_identification": False,
                        "requires_attunement": False,
                        "consumable": True,
                        "value_gp": 50,
                        "weight": 0.5,
                        "notes": "",
                    }
                ],
                "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 5, "pp": 0},
                "notes": "",
            },
        }

        out = DndEventRouterOutput(**data)

        assert out.loot_offer.present is True
        assert out.loot_offer.items[0].item_id == "healing_potion"
        assert out.loot_offer.currency.gp == 5

    def test_dnd_loot_offer_clamps_unknown_literals(self):
        data = {
            **ROUTER_OUTPUT_EXAMPLE,
            "interaction_mode": "cat_i",
            "combatant_ids": [],
            "loot_offer": {
                "present": True,
                "source_kind": "dragon_lair",
                "source_label": "strange cache",
                "visibility": "everyone_everywhere",
                "eligible_character_ids": ["alice"],
                "items": [],
                "currency": {"gp": 1},
                "notes": "",
            },
        }

        out = DndEventRouterOutput(**data)

        assert out.loot_offer.source_kind == "other"
        assert out.loot_offer.visibility == "table"
        assert out.loot_offer.currency.gp == 1

    def test_non_combat_start_modes_clear_battle_map_seed(self):
        data = {
            **ROUTER_OUTPUT_EXAMPLE,
            "interaction_mode": "cat_i",
            "combatant_ids": [],
            "battle_map_seed": {
                "present": True,
                "map_name": "Leaky Seed",
                "width": 8,
                "height": 8,
                "square_size_ft": 5,
                "tokens": [
                    {
                        "token_id": "alice",
                        "character_id": "alice",
                        "label": "Alice",
                        "x": 1,
                        "y": 1,
                        "size_squares": 1,
                    }
                ],
                "terrain": [],
                "areas": [],
                "notes": "",
            },
        }

        out = DndEventRouterOutput(**data)

        assert out.battle_map_seed.present is False
        assert out.battle_map_seed.tokens == []

    def test_combat_start_allows_engine_to_validate_participant_count(self):
        data = {
            **ROUTER_OUTPUT_EXAMPLE,
            "interaction_mode": "dnd_combat_start",
            "combatant_ids": ["alice", "alice", ""],
        }

        out = DndEventRouterOutput(**data)

        assert out.combatant_ids == ["alice"]

    def test_cat_ii_requires_responders(self):
        data = {
            **ROUTER_OUTPUT_EXAMPLE,
            "interaction_mode": "cat_ii",
            "combatant_ids": [],
            "required_responders": [],
        }

        with pytest.raises(ValidationError):
            DndEventRouterOutput(**data)


class TestDndRulesAdjudication:
    def test_spatial_delta_schema_is_strict_object_compatible(self):
        schema = RulesAdjudication.model_json_schema()
        spatial_delta = schema["$defs"]["DndSpatialDelta"]
        assert "spatial_deltas" in schema["required"]
        assert spatial_delta["additionalProperties"] is False
        assert set(spatial_delta["required"]) == {
            "kind",
            "target_id",
            "character_id",
            "x",
            "y",
            "size_squares",
            "label",
            "shape",
            "radius_squares",
            "width",
            "height",
            "duration_rounds",
            "reason",
        }


class TestCharacterAgentOutput:
    def test_construct(self):
        ao = CharacterAgentOutput(**AGENT_OUTPUT_EXAMPLE)
        assert ao.character_id == "guard_17"
        assert "Need a lever, not a miracle." in ao.public_text
        assert "Monitor the user" in ao.intent

    def test_round_trip(self):
        ao = CharacterAgentOutput(**AGENT_OUTPUT_EXAMPLE)
        rebuilt = CharacterAgentOutput(**ao.model_dump())
        assert rebuilt == ao

    def test_rejects_extra_fields(self):
        # Commit 1: extra="forbid" is intentional. If someone reintroduces
        # a structured field on the agent output (e.g. a fresh
        # `current_objectives`) without updating the prompt + downstream
        # consumers, the construction MUST fail loudly here rather than
        # silently leak a parallel intent surface.
        data = {**AGENT_OUTPUT_EXAMPLE, "hidden_thoughts": "I know everything"}
        with pytest.raises(ValidationError):
            CharacterAgentOutput(**data)

    def test_rejects_legacy_structured_fields(self):
        """The pre-Commit-1 schema had `public_response` + `private_updates`
        nested models. Re-introducing either by accident must fail the
        same way an unknown extra would, because downstream code (router
        framing, narrator legacy shim, leakage validator) was rewired to
        read `public_text` only and would silently misbehave."""
        for legacy_key in ("public_response", "private_updates"):
            data = {**AGENT_OUTPUT_EXAMPLE, legacy_key: {}}
            with pytest.raises(ValidationError):
                CharacterAgentOutput(**data)


class TestNarratorFinalOutput:
    def test_construct(self):
        nfo = NarratorFinalOutput(**NARRATOR_FINAL_EXAMPLE)
        assert "Captain Vero" in nfo.final_text

    def test_round_trip(self):
        nfo = NarratorFinalOutput(**NARRATOR_FINAL_EXAMPLE)
        rebuilt = NarratorFinalOutput(**nfo.model_dump())
        assert rebuilt == nfo

    def test_rejects_extra_fields(self):
        # v11-r7j: transcript_entry is no longer a NarratorFinalOutput
        # field; the engine builds it. Schema must reject the legacy
        # name as an unknown extra.
        data = {**NARRATOR_FINAL_EXAMPLE, "transcript_entry": {"user": "x", "assistant": "y"}}
        with pytest.raises(ValidationError):
            NarratorFinalOutput(**data)


class TestTurnRequest:
    def test_construct(self):
        tr = TurnRequest(session_id="abc", user_input="I look around.")
        assert tr.stream is False
        assert tr.acting_character_id == ""

    def test_full(self):
        tr = TurnRequest(
            session_id="abc",
            checkpoint_id="ckpt_0001",
            user_input="I try the door.",
            stream=True,
            acting_character_id="hero",
        )
        assert tr.stream is True
        assert tr.acting_character_id == "hero"

    def test_legacy_debug_fields_silently_dropped(self):
        # v11-r7j murdered `debug` and `debug_flags`. Pre-v11-r7j on-the-
        # wire turn requests sometimes set them to keep per-phase
        # latency reporting alive on the (also-murdered) TurnResponse
        # debug payload. Pydantic v2's default `extra='ignore'` drops
        # them silently; this test pins that contract so a future
        # `extra='forbid'` flip wouldn't quietly break old replay
        # harnesses.
        tr = TurnRequest(
            session_id="abc",
            user_input="I look around.",
            debug=True,
            debug_flags={"include_discriminator": True},
        )
        assert tr.session_id == "abc"
        assert not hasattr(tr, "debug")
        assert not hasattr(tr, "debug_flags")


class TestTurnResponse:
    def test_normal_mode(self):
        tr = TurnResponse(session_id="abc", output_text="You look around.")
        assert tr.session_id == "abc"
        assert tr.output_text == "You look around."
        assert tr.per_player_renders == {}
        assert tr.beat_ended_reason == ""
        assert tr.loot_prompts == {}
        assert tr.commitment_revision_prompts == {}

    def test_legacy_debug_payload_silently_dropped(self):
        tr = TurnResponse(
            session_id="abc",
            output_text="You look around.",
            debug={"canonical_event": {"event_id": "evt_001"}},
        )
        assert tr.session_id == "abc"
        assert not hasattr(tr, "debug")


class TestCheckpointFile:
    def test_construct(self):
        ckpt = CheckpointFile(
            session=SessionState(session_id="test-session"),
            world_state=WorldState(**WORLD_STATE_EXAMPLE),
            characters=[CharacterRecord(**CHARACTER_EXAMPLE)],
        )
        assert ckpt.schema_version == "4.0"  # relative-time hard break
        assert ckpt.session.session_id == "test-session"
        assert len(ckpt.characters) == 1
        # Pre-versioning / hand-built checkpoints have empty importer_version;
        # the importer stamps it on build.
        assert ckpt.importer_version == ""

    def test_legacy_prompt_versions_field_ignored(self):
        # Older checkpoints stamped a `prompt_versions` dict (when
        # prompt files carried `_v#` suffixes). Git is the version
        # log now and the field is gone — but pydantic must still
        # tolerate it on load so old saves don't hard-break.
        legacy_payload = {
            "session": {"session_id": "legacy"},
            "prompt_versions": {"event_router": "v9", "agent": "v11"},
        }
        ckpt = CheckpointFile(**legacy_payload)
        assert ckpt.session.session_id == "legacy"
        assert not hasattr(ckpt, "prompt_versions")

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
        # ux-primer-4: Pre-v8 (and freshly hand-built) checkpoints have
        # no player primer — render_briefing falls back to a stub.
        assert ckpt.player_primer == ""

    def test_player_primer_round_trip(self):
        """Importer Call 6 stamps a 1-2 paragraph world primer onto the
        checkpoint; render_briefing reads it directly. Round-trip
        through JSON to make sure the field survives save → load."""
        primer = (
            "You're a contestant on a sun-bleached dating show. Cameras "
            "everywhere; a dozen rivals; one rose left. Last thing you "
            "remember was a truck and the smell of ozone. Now you're here."
        )
        ckpt = CheckpointFile(
            session=SessionState(session_id="primer-test"),
            player_primer=primer,
        )
        rebuilt = CheckpointFile(**ckpt.model_dump())
        assert rebuilt.player_primer == primer
