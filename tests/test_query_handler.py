"""Tests for the out-of-character /query handler.

Covers:
- QueryResponse schema basics + happy-path and gated-path round-trip
- answer_query LLM dispatch (mocked) — role, response_model, returned shape
- Internal block formatters (scene, recent events, pending observations)
- Trust boundary — hidden_lore / hidden_facts must NEVER reach the prompt
- Prompt template renders cleanly with all required variables
- Recent-events filter — only includes events the asking character observed
- Recent-events cap — bounded by max_events
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.prompt_manager import PromptManager
from app.engine.query_handler import (
    _format_pending,
    _format_recent_events,
    _format_scene,
    answer_query,
)
from app.schemas.characters import CharacterRecord, CharacterStatus, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    SceneDelta,
    WorldAdjudication,
)
from app.schemas.query import QueryResponse
from app.schemas.state import LocationState, SessionState, WorldState


# ---- helpers -----------------------------------------------------------------


def _llm_response(parsed):
    """Mirror the test_turn_recap helper — a fake LLMResponse carrying a
    parsed Pydantic model. The raw_response is a MagicMock with the
    minimum block shape callers inspect."""
    from app.llm.client import LLMResponse
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "{}"
    text_block.model_dump = lambda: {"type": "text", "text": "{}"}
    raw.content = [text_block]
    raw.model = "claude-haiku-4-5"
    return LLMResponse(
        parsed=parsed,
        raw_response=raw,
        content="{}",
        model="claude-haiku-4-5",
    )


def _build_event(
    *,
    outcome: str,
    observers: list[tuple[str, str]],
    facts: list[object] | None = None,
) -> EventRouterOutput:
    """Build a minimal EventRouterOutput with the v11-required all-fields-set
    shape. `observers` is a list of (character_id, level) pairs."""
    return EventRouterOutput(
        event_id="",  # validator mints one
        decision_rationale="test",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                attempted_action="t",
                feasible=True,
                resolved_outcome=outcome,
            ),
            scene_delta=SceneDelta(time_advanced_seconds=0),
            observable_facts=list(facts or []),
        ),
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="",
        observers=[
            ObserverEntry(
                character_id=cid,
                observation_level=lvl,
                response_priority=3,
            )
            for cid, lvl in observers
        ],
        spawn=[],
        dormant=[],
        cull=[],
        roster_moves=[],
        scenes_created=[],
    )


def _build_checkpoint(
    *,
    extra_characters: list[CharacterRecord] | None = None,
    canonical_events: list[EventRouterOutput] | None = None,
    hidden_lore: str = "",
    hidden_facts: list[str] | None = None,
    pending_observations: list[str] | None = None,
) -> CheckpointFile:
    """Build a checkpoint with one player character (`alice`) in
    `garden`, plus an NPC (`bob`) co-located. Test variants pass extra
    characters / events / hidden state via kwargs."""
    alice = CharacterRecord(
        character_id="alice",
        name="Alice",
        status=CharacterStatus.active,
        location="garden",
        is_playable=True,
        public_sheet=PublicSheet(
            role="Detective",
            appearance="weathered coat",
            faction="city watch",
        ),
        backstory="grew up here",
        personality="terse",
        known_context="You know the streets and the local rumors.",
        pending_observations=list(pending_observations or []),
    )
    bob = CharacterRecord(
        character_id="bob",
        name="Bob",
        status=CharacterStatus.active,
        location="garden",
        is_playable=False,
        public_sheet=PublicSheet(
            role="Gardener",
            appearance="muddy apron",
        ),
    )
    chars = [alice, bob] + list(extra_characters or [])

    locations = LocationState(scene_graph={
        "garden": {
            "name": "The Garden",
            "description": "Ivy crawling over a stone wall.",
        },
        "library": {
            "name": "Old Library",
            "description": "Dust and shelves.",
        },
    })
    world = WorldState(
        locations=locations,
        hidden_lore=hidden_lore,
        hidden_facts=list(hidden_facts or []),
    )
    return CheckpointFile(
        session=SessionState(
            session_id="test_session",
            character_bindings={"alice": "111"},
        ),
        world_state=world,
        characters=chars,
        canonical_events=list(canonical_events or []),
    )


# ---- schema ------------------------------------------------------------------


class TestQueryResponseSchema:
    def test_defaults(self):
        r = QueryResponse()
        assert r.answer == ""
        assert r.knowledge_gated is False
        assert r.gate_reason == ""

    def test_happy_round_trip(self):
        r = QueryResponse(answer="You see the garden.", knowledge_gated=False)
        roundtripped = QueryResponse.model_validate_json(r.model_dump_json())
        assert roundtripped == r

    def test_gated_round_trip(self):
        r = QueryResponse(
            answer="You can't see — you're blindfolded.",
            knowledge_gated=True,
            gate_reason="blindfolded",
        )
        roundtripped = QueryResponse.model_validate_json(r.model_dump_json())
        assert roundtripped == r

    def test_extra_field_rejected(self):
        """`extra='forbid'` keeps the structured-output grammar tight; an
        accidental extra field on the LLM side should fail loud."""
        with pytest.raises(Exception):
            QueryResponse.model_validate({"answer": "x", "bogus": True})


# ---- internal helpers --------------------------------------------------------


class TestFormatScene:
    def test_with_others_present(self):
        ckpt = _build_checkpoint()
        out = _format_scene(ckpt, "alice")
        assert "Location: The Garden" in out
        assert "Ivy crawling over a stone wall." in out
        assert "Bob" in out
        assert "Gardener" in out

    def test_alone_in_scene(self):
        """Bob is removed; alice is the only one in the garden."""
        ckpt = _build_checkpoint()
        ckpt.characters = [c for c in ckpt.characters if c.character_id != "bob"]
        out = _format_scene(ckpt, "alice")
        assert "Location: The Garden" in out
        assert "No one else is present" in out

    def test_unsited_character(self):
        ckpt = _build_checkpoint()
        ckpt.characters[0].location = ""
        out = _format_scene(ckpt, "alice")
        assert "unknown" in out

    def test_unknown_character_id(self):
        ckpt = _build_checkpoint()
        out = _format_scene(ckpt, "ghost")
        assert "unknown" in out

    def test_culled_character_excluded_from_present(self):
        ckpt = _build_checkpoint()
        bob = next(c for c in ckpt.characters if c.character_id == "bob")
        bob.status = CharacterStatus.culled
        out = _format_scene(ckpt, "alice")
        assert "Bob" not in out
        assert "No one else is present" in out


class TestFormatRecentEvents:
    def test_no_events_yields_no_events_marker(self):
        ckpt = _build_checkpoint()
        out = _format_recent_events(ckpt, "alice")
        assert "no events on record" in out

    def test_only_observer_events_kept(self):
        """Two events fire; alice observes the first, bob observes the
        second. Alice's recent events block must NOT contain the second."""
        e1 = _build_event(
            outcome="A messenger arrives.",
            observers=[("alice", "d")],
            facts=["a messenger steps up to the gate", "envelope on the table"],
        )
        e2 = _build_event(
            outcome="A bell tolls in the distance.",
            observers=[("bob", "i")],
            facts=["a deep bell tolls somewhere east of the courtyard"],
        )
        ckpt = _build_checkpoint(canonical_events=[e1, e2])
        out = _format_recent_events(ckpt, "alice")
        assert "messenger steps up to the gate" in out
        assert "envelope on the table" in out
        assert "bell tolls" not in out

    def test_observation_level_label(self):
        e = _build_event(
            outcome="Lights flicker.",
            observers=[("alice", "i")],
            facts=["the chandelier dims, brightens, dims again"],
        )
        ckpt = _build_checkpoint(canonical_events=[e])
        out = _format_recent_events(ckpt, "alice")
        assert "indirectly perceived" in out

    def test_max_events_cap(self):
        """When more than max_events match, only the freshest are kept,
        and they're returned chronologically (oldest of the kept first)."""
        events = [
            _build_event(
                outcome=f"Event {i}.",
                observers=[("alice", "d")],
                facts=[f"event {i} fact"],
            )
            for i in range(20)
        ]
        ckpt = _build_checkpoint(canonical_events=events)
        out = _format_recent_events(ckpt, "alice", max_events=3)
        # Last three events were #17, #18, #19; rendered chronologically.
        assert "event 19 fact" in out
        assert "event 18 fact" in out
        assert "event 17 fact" in out
        assert "event 16 fact" not in out
        idx_17 = out.index("event 17 fact")
        idx_19 = out.index("event 19 fact")
        assert idx_17 < idx_19, "kept events should render chronologically"

    def test_unobserved_character_yields_no_events_marker(self):
        e = _build_event(
            outcome="Bob does something private.",
            observers=[("bob", "d")],
            facts=["bob shifts on his stool"],
        )
        ckpt = _build_checkpoint(canonical_events=[e])
        out = _format_recent_events(ckpt, "alice")
        assert "hasn't observed any events" in out

    def test_resolved_outcome_NOT_surfaced_to_asker(self):
        """Option B: `resolved_outcome` is audit-only and must NOT
        leak into the asking character's introspection. The router's
        outcome string regularly contains narrator-grade interpretive
        prose ("the strain of speaking close to the edge of what she
        is permitted") that the asker has no perceptual basis for
        knowing — surfacing it would let an agent "remember" interior
        details they could not have observed. Only `observable_facts`
        is rendered to the asker."""
        e = _build_event(
            outcome=(
                "Seraphel recites a fractured plague verse, the strain "
                "of speaking close to the edge of what she is permitted "
                "showing in her wings."
            ),
            observers=[("alice", "d")],
            facts=[
                "Seraphel recites: 'The plague that fell on human "
                "ground / Killed only those who could be found'",
                "her wings draw tight against her back, then flutter sharply",
            ],
        )
        ckpt = _build_checkpoint(canonical_events=[e])
        out = _format_recent_events(ckpt, "alice")
        assert "wings draw tight" in out
        assert "what she is permitted" not in out
        assert "the strain of speaking" not in out

    def test_scoped_facts_filter_by_asking_character(self):
        e = _build_event(
            outcome="Dan questions Thessaly and signals Ashara.",
            observers=[("alice", "d"), ("ashara", "d")],
            facts=[
                ObservableFact.only(
                    "Dan's foot touches Ashara's boot under the table.",
                    ["ashara"],
                ),
                ObservableFact.all(
                    "Dan asks Thessaly whether she knows curses.",
                ),
            ],
        )
        ckpt = _build_checkpoint(canonical_events=[e])

        alice_out = _format_recent_events(ckpt, "alice")
        ashara_out = _format_recent_events(ckpt, "ashara")

        assert "foot touches Ashara's boot" not in alice_out
        assert "knows curses" in alice_out
        assert "foot touches Ashara's boot" in ashara_out
        assert "knows curses" in ashara_out

    def test_event_with_no_observable_facts_renders_marker(self):
        """An event with empty observable_facts shouldn't crash or
        leak the outcome — it renders a clean placeholder so the
        asker sees the event happened without any sensory contents."""
        e = _build_event(
            outcome="Something happens.",
            observers=[("alice", "d")],
            facts=[],
        )
        ckpt = _build_checkpoint(canonical_events=[e])
        out = _format_recent_events(ckpt, "alice")
        assert "no observable surface" in out
        assert "Something happens" not in out


class TestFormatPending:
    def test_empty(self):
        ckpt = _build_checkpoint()
        out = _format_pending(ckpt, "alice")
        assert "none" in out

    def test_renders_entries_as_bullets(self):
        ckpt = _build_checkpoint(pending_observations=[
            "[off-scene] you hear a window break upstairs",
            "[off-scene] footsteps recede toward the hall",
        ])
        out = _format_pending(ckpt, "alice")
        assert "- [off-scene] you hear a window break upstairs" in out
        assert "- [off-scene] footsteps recede toward the hall" in out


# ---- answer_query LLM dispatch ----------------------------------------------


class TestAnswerQuery:
    def _run(self, coro):
        return asyncio.run(coro)

    def _client(self, parsed: QueryResponse):
        client = MagicMock()
        client.config = MagicMock()
        client.config.model_for_role = MagicMock(return_value="claude-haiku-4-5")
        client.complete = AsyncMock(return_value=_llm_response(parsed))
        return client

    def test_happy_path_returns_answer(self):
        client = self._client(QueryResponse(
            answer="You see ivy crawling over a stone wall.",
            knowledge_gated=False,
        ))
        pm = PromptManager("app/prompts")
        ckpt = _build_checkpoint()
        result = self._run(answer_query(
            client, pm, ckpt, "alice", "What do I see?",
        ))
        assert result.answer == "You see ivy crawling over a stone wall."
        assert result.knowledge_gated is False
        client.complete.assert_awaited_once()

    def test_gated_path_returns_refusal(self):
        client = self._client(QueryResponse(
            answer="You can't see — you're blindfolded.",
            knowledge_gated=True,
            gate_reason="blindfolded",
        ))
        pm = PromptManager("app/prompts")
        ckpt = _build_checkpoint()
        result = self._run(answer_query(
            client, pm, ckpt, "alice", "What's on the desk?",
        ))
        assert result.knowledge_gated is True
        assert result.gate_reason == "blindfolded"
        assert "blindfolded" in result.answer

    def test_whitespace_stripped(self):
        client = self._client(QueryResponse(
            answer="  Padded answer.\n\n",
            knowledge_gated=False,
            gate_reason="  ",
        ))
        pm = PromptManager("app/prompts")
        ckpt = _build_checkpoint()
        result = self._run(answer_query(
            client, pm, ckpt, "alice", "anything",
        ))
        assert result.answer == "Padded answer."
        assert result.gate_reason == ""

    def test_dispatches_under_query_handler_role(self):
        """Critical for the LLMConfig wiring — query_handler picks Haiku
        by default, swappable per role independent of /act."""
        client = self._client(QueryResponse(answer="ok"))
        pm = PromptManager("app/prompts")
        ckpt = _build_checkpoint()
        self._run(answer_query(client, pm, ckpt, "alice", "x"))
        kwargs = client.complete.await_args.kwargs
        assert kwargs["role"] == "query_handler"
        assert kwargs["response_model"] is QueryResponse

    def test_empty_question_falls_back_to_placeholder(self):
        """The bot guards empty questions before calling, but if the
        engine layer is hit directly with whitespace-only it shouldn't
        crash on the prompt format step."""
        client = self._client(QueryResponse(answer="?"))
        pm = PromptManager("app/prompts")
        ckpt = _build_checkpoint()
        self._run(answer_query(client, pm, ckpt, "alice", "   "))
        msgs = client.complete.await_args.kwargs["messages"]
        user_blob = "\n".join(m["content"] for m in msgs if m["role"] == "user")
        assert "(empty question)" in user_blob

    def test_unknown_character_renders_without_crashing(self):
        """If the caller hands a character_id not in the roster (race
        condition / orphaned binding), the helpers should degrade
        gracefully instead of throwing."""
        client = self._client(QueryResponse(answer="?"))
        pm = PromptManager("app/prompts")
        ckpt = _build_checkpoint()
        self._run(answer_query(client, pm, ckpt, "ghost", "where am I?"))
        client.complete.assert_awaited_once()


# ---- trust boundary ----------------------------------------------------------


class TestTrustBoundary:
    """The query handler must NEVER receive hidden world knowledge in
    its prompt. The asking character's POV is the boundary; hidden_lore
    and hidden_facts are router/discriminator-only context."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_hidden_lore_not_in_prompt(self):
        client = MagicMock()
        client.config = MagicMock()
        client.config.model_for_role = MagicMock(return_value="claude-haiku-4-5")
        client.complete = AsyncMock(return_value=_llm_response(
            QueryResponse(answer="ok"),
        ))
        pm = PromptManager("app/prompts")
        ckpt = _build_checkpoint(
            hidden_lore=(
                "SPOILER_TOKEN_HIDDEN_LORE: the real killer is the "
                "gardener's twin."
            ),
            hidden_facts=[
                "SPOILER_TOKEN_HIDDEN_FACT: the door has a false bottom",
            ],
        )
        self._run(answer_query(client, pm, ckpt, "alice", "what's going on?"))
        msgs = client.complete.await_args.kwargs["messages"]
        full_prompt = "\n".join(m["content"] for m in msgs)
        assert "SPOILER_TOKEN_HIDDEN_LORE" not in full_prompt
        assert "SPOILER_TOKEN_HIDDEN_FACT" not in full_prompt


# ---- prompt template contract ------------------------------------------------


class TestPromptContract:
    def test_template_renders_with_all_required_variables(self):
        """Smoke test that the prompt file's `{var}` slots match what
        answer_query passes. If a slot is added to the .txt without a
        corresponding builder kwarg this test catches it loudly rather
        than at runtime in the bot."""
        pm = PromptManager("app/prompts")
        rendered = pm.render(
            "query_handler",
            setting_summary="SENTINEL_SETTING",
            character_identity_block="SENTINEL_IDENTITY",
            known_context_block="SENTINEL_KNOWN_CONTEXT",
            scene_block="SENTINEL_SCENE",
            player_characters_block="SENTINEL_PLAYERS",
            recent_events_block="SENTINEL_RECENT_EVENTS",
            pending_observations_block="SENTINEL_PENDING",
            question="SENTINEL_QUESTION",
        )
        for sentinel in (
            "SENTINEL_SETTING",
            "SENTINEL_IDENTITY",
            "SENTINEL_KNOWN_CONTEXT",
            "SENTINEL_SCENE",
            "SENTINEL_PLAYERS",
            "SENTINEL_RECENT_EVENTS",
            "SENTINEL_PENDING",
            "SENTINEL_QUESTION",
        ):
            assert sentinel in rendered

    def test_template_rejects_missing_required_variable(self):
        pm = PromptManager("app/prompts")
        kwargs = dict(
            setting_summary="x",
            character_identity_block="x",
            known_context_block="x",
            scene_block="x",
            player_characters_block="x",
            recent_events_block="x",
            pending_observations_block="x",
        )
        with pytest.raises(KeyError):
            pm.render("query_handler", **kwargs)
