"""Tests for the Orchestrator — turn pipeline integration.

v11-A3c: the TestOrchestrator class tests reach into the v8
process_turn body (router → agent → narrator LLM call sequence,
per-phase latencies, pending_recap, turn_recap summarizer) that the
v11 rewrite deleted. Coverage for the new thin-wrapper shape lives in
tests/test_orchestrator_v11.py. The TestCharacterManager and
TestCharacterSpawn classes survive — those exercise CharacterManager
directly and don't depend on the orchestrator path.

v11-r7j: the legacy `EventRouter` engine class was murdered. Skipped
tests below still mention "EventRouter" in their docstrings as the
historical name; that's accurate flavor for a class that no longer
exists.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.engine.character_manager import CharacterManager
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.engine.turn_recap import _TurnRecapOutput
from app.llm.client import LLMClient, LLMResponse
from app.llm.config import LLMConfig
from app.schemas.agents import CharacterAgentOutput  # noqa: F401  (skipped tests construct fixtures lazily)
from app.schemas.characters import CharacterRecord, PublicSheet, PrivateState
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry, SpawnRequest
from app.schemas.events import CanonicalEvent, WorldAdjudication, SceneDelta
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.requests import TurnRequest
from app.schemas.state import LocationState, SessionState, WorldState

# --- Fixtures ---

@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock()
    client.config = LLMConfig()
    return client

@pytest.fixture
def sample_checkpoint():
    return CheckpointFile(
        session=SessionState(session_id="test-session", turn_index=0),
        world_state=WorldState(
            locations=LocationState(
                scene_graph={
                    "courtyard": {
                        "name": "Courtyard",
                        "description": "A stone courtyard.",
                        "connected_to": [],
                    },
                },
            ),
        ),
        characters=[
            CharacterRecord(
                character_id="guard_17",
                name="Captain Vero",
                location="courtyard",
                public_sheet=PublicSheet(role="guard captain"),
                private_state=PrivateState(),
            ),
        ],
    )

@pytest.fixture
def mock_checkpoint_mgr(sample_checkpoint):
    mgr = MagicMock()
    mgr.load_latest.return_value = sample_checkpoint
    mgr.save.return_value = None
    return mgr

def _llm_response(parsed) -> LLMResponse:
    """Shape the LLMResponse. Character spawn parses JSON from
    response.content (structured output disabled — see benchmark), so
    content round-trips through the parsed model."""
    text = parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else "{}"
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    text_block.model_dump = lambda: {"type": "text", "text": text}
    raw.content = [text_block]
    raw.model = "claude-sonnet-4-6"
    return LLMResponse(parsed=parsed, raw_response=raw, content=text, model="claude-sonnet-4-6")

@pytest.fixture
def prompt_manager():
    return PromptManager("app/prompts")

# --- Tests ---

class TestCharacterManager:
    def test_get_character(self, sample_checkpoint):
        mgr = CharacterManager()
        char = mgr.get_character(sample_checkpoint, "guard_17")
        assert char is not None
        assert char.name == "Captain Vero"

    def test_get_missing_character(self, sample_checkpoint):
        mgr = CharacterManager()
        char = mgr.get_character(sample_checkpoint, "nonexistent")
        assert char is None

    def test_apply_roster_dormant(self, sample_checkpoint):
        mgr = CharacterManager()
        routed = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="wait", feasible=True, resolved_outcome="",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[],
            ),
            observers=[],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="",
            spawn=[],
            dormant=["guard_17"],
            cull=[],
            roster_moves=[],
            scenes_created=[],
        )
        mgr.apply_roster_updates(sample_checkpoint, routed)
        char = mgr.get_character(sample_checkpoint, "guard_17")
        assert char.status.value == "dormant"

@pytest.mark.skip(
    reason="v11: legacy orchestrator path superseded; rewrite coverage in "
    "test_orchestrator_v11.py"
)
class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_full_turn(self, mock_client, mock_checkpoint_mgr, prompt_manager):
        """Test a complete turn with mocked LLM responses (canonical merged path)."""
        # 3 LLM calls: EventRouter (merged NP1+Disc), Agent, NP2.
        merged = EventRouterOutput(
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="survey area",
                    feasible=True,
                    resolved_outcome="Player looks around.",
                ),
                observable_facts=["Player looks around."],
            ),
            observers=[
                ObserverEntry(
                    character_id="guard_17",
                    response_priority=5,
                ),
            ],
        )

        agent_out = CharacterAgentOutput(
            character_id="guard_17",
            public_text='"Everything alright?"',
            intent="",
        )

        narrator_out = NarratorFinalOutput(
            final_text="You look around. \"Everything alright?\" Captain Vero asks.",
            transcript_entry=TranscriptEntry(
                user="I look around.",
                assistant="You look around. \"Everything alright?\" Captain Vero asks.",
            ),
        )

        mock_client.complete.side_effect = [
            _llm_response(merged),
            _llm_response(agent_out),
            _llm_response(narrator_out),
            _llm_response(_TurnRecapOutput(note="Captain Vero watched.")),
        ]

        orchestrator = Orchestrator(mock_client, mock_checkpoint_mgr, prompt_manager)
        request = TurnRequest(session_id="test-session", user_input="I look around.")

        response = await orchestrator.process_turn(request)

        assert response.session_id == "test-session"
        assert "look around" in response.output_text.lower()
        assert response.turn_index == 1
        # 4 LLM calls: router + agent + narrator + turn-recap summarizer.
        assert mock_client.complete.call_count == 4
        mock_checkpoint_mgr.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_turn_with_no_responders(
        self, mock_client, mock_checkpoint_mgr, prompt_manager
    ):
        """Turn where no characters respond — EventRouter returns no observers."""
        merged = EventRouterOutput(
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="internal reflection",
                    feasible=True,
                    resolved_outcome="Player thinks.",
                ),
            ),
            observers=[],
        )

        narrator_out = NarratorFinalOutput(
            final_text="You stand quietly, collecting your thoughts.",
            transcript_entry=TranscriptEntry(
                user="I think about things.",
                assistant="You stand quietly, collecting your thoughts.",
            ),
        )

        mock_client.complete.side_effect = [
            _llm_response(merged),
            _llm_response(narrator_out),
            _llm_response(_TurnRecapOutput(note="")),
        ]

        orchestrator = Orchestrator(mock_client, mock_checkpoint_mgr, prompt_manager)
        request = TurnRequest(session_id="test-session", user_input="I think about things.")

        response = await orchestrator.process_turn(request)

        assert "quietly" in response.output_text.lower()
        # 2 LLM calls: EventRouter, NP2 (no agents)
        # router + narrator + turn-recap (no agents this turn).
        assert mock_client.complete.call_count == 3

    # NOTE: a `test_debug_mode` lived here that asserted `response.debug`
    # was populated when `request.debug=True`. v11-r7j murdered both
    # `TurnRequest.debug` and `TurnResponse.debug` (the orchestrator
    # never wrote the response payload — the test was a green lie
    # because the surrounding class is `@pytest.mark.skip`'d on the
    # legacy orchestrator path). The deletion is intentional.


class TestCharacterSpawn:
    @pytest.mark.asyncio
    async def test_spawn_character(self, mock_client, sample_checkpoint):
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Tom the Stablehand",
            location="courtyard",
            role="stablehand",
            appearance="", faction="", backstory="",
            personality="Nervous, avoids eye contact.",
            known_context="", goals=[], current_objectives=[],
            secrets=[], intentions_enabled=False,
            router_summary="Nervous stablehand at the courtyard, watching the gate.",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="stablehand_01", seed={"role": "stablehand"})],
        )

        assert len(spawned) == 1
        assert spawned[0].name == "Tom the Stablehand"
        # Should be added to checkpoint
        assert mgr.get_character(sample_checkpoint, "stablehand_01") is not None

    @pytest.mark.asyncio
    async def test_skip_existing_character(self, mock_client, sample_checkpoint):
        mock_client.complete = AsyncMock()
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        # guard_17 already exists
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="guard_17", seed={"role": "guard"})],
        )

        assert len(spawned) == 0
        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_spawn_limit(self, mock_client, sample_checkpoint):
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="NPC", location="courtyard",
            role="", appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False,
            router_summary="Generic walk-on at the courtyard.",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        requests = [
            SpawnRequest(character_id=f"npc_{i}", seed={"role": f"npc{i}"})
            for i in range(5)
        ]
        spawned = await mgr.spawn_characters(sample_checkpoint, requests)

        # Capped at MAX_SPAWNS_PER_TURN = 3
        assert len(spawned) == 3

    def test_spawn_without_client(self, sample_checkpoint):
        mgr = CharacterManager()
        import asyncio
        spawned = asyncio.run(
            mgr.spawn_characters(
                sample_checkpoint,
                [SpawnRequest(character_id="test", seed={})],
            )
        )
        assert len(spawned) == 0

    @pytest.mark.asyncio
    async def test_spawn_writes_router_summary_then_drains(
        self, mock_client, sample_checkpoint,
    ):
        """Commit 4: spawn flow produces a router_summary that lands in
        `pending_router_state_changes` exactly once.

        Verifies the full wiring: the LLM-authored summary makes it
        through `_spawn_one`, `_push_spawn_state_change` formats it
        as a "Spawned: ..." line, and the subsequent router-context
        builder drains the queue (one-shot semantics).
        """
        from app.engine.turn_loop_dispatcher import _build_state_changes_block
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Sera the Cartographer",
            location="courtyard",
            role="cartographer",
            appearance="", faction="", backstory="",
            personality="Quiet, watchful.",
            known_context="",
            goals=[], current_objectives=[],
            secrets=[], intentions_enabled=False,
            router_summary=(
                "Exiled cartographer who has just stepped into the courtyard, "
                "looking for the steward to plead her family's case."
            ),
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="sera_01", seed={"role": "cartographer"})],
        )
        assert len(spawned) == 1

        queue = sample_checkpoint.session.pending_router_state_changes
        assert len(queue) == 1
        line = queue[0]
        assert "Spawned: Sera the Cartographer" in line
        assert "sera_01" in line
        assert "Exiled cartographer" in line
        assert "steward" in line

        block = _build_state_changes_block(sample_checkpoint)
        assert "## State Changes Since Your Last Call" in block
        assert "Spawned: Sera the Cartographer" in block
        assert sample_checkpoint.session.pending_router_state_changes == []

        block_again = _build_state_changes_block(sample_checkpoint)
        assert block_again == ""

    @pytest.mark.asyncio
    async def test_spawn_falls_back_when_router_summary_blank(
        self, mock_client, sample_checkpoint,
    ):
        """Commit 4 fallback: if the LLM regresses and emits an empty
        `router_summary`, the engine writes the mechanical
        name+role+location+objectives line so the spawn still surfaces.
        """
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Anon",
            location="courtyard",
            role="messenger",
            appearance="", faction="", backstory="",
            personality="",
            known_context="",
            goals=[], current_objectives=["deliver the writ"],
            secrets=[], intentions_enabled=False,
            router_summary="   ",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="anon_01", seed={"role": "messenger"})],
        )
        queue = sample_checkpoint.session.pending_router_state_changes
        assert len(queue) == 1
        line = queue[0]
        assert "Spawned: Anon" in line
        assert "role=messenger" in line
        assert "objectives=deliver the writ" in line

    @pytest.mark.asyncio
    async def test_spawn_summary_normalizes_newlines_and_truncates(
        self, mock_client, sample_checkpoint,
    ):
        """Commit 4b: a router_summary with embedded newlines or
        gigantic length is normalized so it can't shatter the next
        router prompt's bullet list or balloon the user message.
        """
        from app.engine.character_manager import (
            ROUTER_SUMMARY_MAX_CHARS,
            _normalize_router_summary,
        )
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        long_summary = (
            "First line.\nSecond line with extra   whitespace.\n\n"
            + ("padding " * 200)
        )
        authored = AuthoredCharacter(
            name="Verbose Visitor",
            location="courtyard", role="bard",
            appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False,
            router_summary=long_summary,
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="verb_01", seed={"role": "bard"})],
        )
        queue = sample_checkpoint.session.pending_router_state_changes
        assert len(queue) == 1
        line = queue[0]
        assert "\n" not in line
        prefix_overhead = len("Spawned: Verbose Visitor (id: verb_01) — ")
        assert len(line) <= prefix_overhead + ROUTER_SUMMARY_MAX_CHARS

        cleaned = _normalize_router_summary(long_summary)
        assert "\n" not in cleaned
        assert len(cleaned) <= ROUTER_SUMMARY_MAX_CHARS
        assert "  " not in cleaned

    @pytest.mark.asyncio
    async def test_cull_drops_pending_spawn_line(
        self, mock_client, sample_checkpoint,
    ):
        """Commit 4b: spawning a character and then culling them in the
        same beat must NOT leave a "Spawned: ..." ghost line in the
        next router call's State Changes block.
        """
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Doomed",
            location="courtyard", role="messenger",
            appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False,
            router_summary="Messenger who arrived with bad news.",
        )
        mock_client.complete.return_value = _llm_response(authored)
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="doomed_01", seed={"role": "messenger"})],
        )
        assert any(
            "doomed_01" in line
            for line in sample_checkpoint.session.pending_router_state_changes
        )

        cull_routed = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(
                    attempted_action="The messenger collapses",
                    feasible=True,
                    resolved_outcome="The messenger dies on the spot.",
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=[],
            ),
            observers=[],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="",
            spawn=[],
            dormant=[],
            cull=["doomed_01"],
            roster_moves=[],
            scenes_created=[],
        )
        mgr.apply_roster_updates(sample_checkpoint, cull_routed)

        assert all(
            "doomed_01" not in line
            for line in sample_checkpoint.session.pending_router_state_changes
        )

    @pytest.mark.asyncio
    async def test_spawn_dedups_within_batch(
        self, mock_client, sample_checkpoint,
    ):
        """v11-r7i: a single router call can emit multiple SpawnRequests
        sharing one character_id (regression observed in dating_villa
        playtest where the router emitted three `production_runner`
        spawns in one turn). Engine must keep the first and silently
        drop the rest, otherwise `checkpoint.characters.append` of
        the same id would leave the roster with duplicate records
        sharing one id and the canonical event ambiguous."""
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Production Runner",
            location="courtyard", role="runner",
            appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False,
            router_summary="A runner.",
        )
        mock_client.complete.return_value = _llm_response(authored)

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [
                SpawnRequest(character_id="runner_01", seed={"role": "runner"}),
                SpawnRequest(character_id="runner_01", seed={"role": "runner"}),
                SpawnRequest(character_id="runner_01", seed={"role": "runner"}),
            ],
        )
        assert len(spawned) == 1
        assert mock_client.complete.call_count == 1
        ids = [c.character_id for c in sample_checkpoint.characters]
        assert ids.count("runner_01") == 1

    @pytest.mark.asyncio
    async def test_spawn_dedup_runs_before_cap(
        self, mock_client, sample_checkpoint,
    ):
        """Dedup MUST run before MAX_SPAWNS_PER_TURN truncation, not
        after. Otherwise an LLM that spams `[runner, runner, runner,
        unique_villain]` gets sliced to `[runner, runner, runner]`
        first, then collapsed to `[runner]` — the unique villain is
        silently dropped despite us spawning only 1 actual character.
        Correct order: dedup → 2 distinct ids → both spawn."""
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        # Generic authored body; we only care about call count + ids.
        authored = AuthoredCharacter(
            name="X", location="courtyard", role="r",
            appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False, router_summary="x",
        )
        mock_client.complete.return_value = _llm_response(authored)
        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))

        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [
                SpawnRequest(character_id="runner_01", seed={"role": "r"}),
                SpawnRequest(character_id="runner_01", seed={"role": "r"}),
                SpawnRequest(character_id="runner_01", seed={"role": "r"}),
                SpawnRequest(character_id="unique_villain", seed={"role": "v"}),
            ],
        )
        ids = sorted(c.character_id for c in spawned)
        assert ids == ["runner_01", "unique_villain"], (
            "unique_villain must survive dedup-then-cap; "
            "got " + repr(ids)
        )

    @pytest.mark.asyncio
    async def test_spawn_uses_acting_actor_location_when_seed_omits(
        self, mock_client, sample_checkpoint,
    ):
        """v11-r7i: when the router omits `seed.location`, the new
        character materializes at the acting actor's scene (passed by
        the orchestrator). Pre-r7i the fallback was the global
        `current_scene_id` which was stale; new behavior drops the
        spawn at the actor's actual scene."""
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        # The LLM authors location="ignored" — the engine should
        # OVERRIDE this with the resolved scene_id and the spawn
        # should land in "library" (acting_actor_location), not the
        # LLM's choice.
        authored = AuthoredCharacter(
            name="Courier",
            location="ignored_by_engine",
            role="messenger",
            appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False,
            router_summary="A courier.",
        )
        mock_client.complete.return_value = _llm_response(authored)
        sample_checkpoint.world_state.locations.scene_graph["library"] = {
            "name": "Library", "description": "Quiet stacks.",
        }

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(character_id="courier_01", seed={})],
            acting_actor_location="library",
        )
        assert len(spawned) == 1
        assert spawned[0].location == "library"

    @pytest.mark.asyncio
    async def test_spawn_seed_location_beats_actor_location(
        self, mock_client, sample_checkpoint,
    ):
        """When the router DOES specify `seed.location`, that wins over
        the acting actor's scene — the router knows where the spawn
        should appear (e.g. recipient's scene for a courier, witness
        location for a bystander). Actor-location is only the
        fallback when seed is silent."""
        from app.schemas.takeover import AuthoredCharacter
        mock_client.complete = AsyncMock()
        authored = AuthoredCharacter(
            name="Steward", location="ignored_by_engine", role="steward",
            appearance="", faction="", backstory="",
            personality="", known_context="",
            goals=[], current_objectives=[], secrets=[],
            intentions_enabled=False, router_summary="Steward.",
        )
        mock_client.complete.return_value = _llm_response(authored)
        sample_checkpoint.world_state.locations.scene_graph["chapel"] = {
            "name": "Chapel", "description": "",
        }
        sample_checkpoint.world_state.locations.scene_graph["library"] = {
            "name": "Library", "description": "",
        }

        mgr = CharacterManager(mock_client, PromptManager("app/prompts"))
        spawned = await mgr.spawn_characters(
            sample_checkpoint,
            [SpawnRequest(
                character_id="steward_01",
                seed={"location": "chapel", "role": "steward"},
            )],
            acting_actor_location="library",
        )
        assert spawned[0].location == "chapel"
