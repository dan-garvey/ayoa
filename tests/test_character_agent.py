"""Tests for the Character Agent engine (rolling-conversation architecture)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.character_agent import CharacterAgent
from app.engine.context_builder import (
    build_character_packet,
    build_character_state,
    build_characters_present,
    build_scene_context,
    build_world_context,
    format_observed_facts,
    format_pending_observations_block,
    resolve_scene_for_character,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.state import (
    LocationState,
    SessionState,
    StorySetting,
    WorldState,
)


# --- Fixtures ---

@pytest.fixture
def prompt_manager():
    return PromptManager("app/prompts")


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock()
    return client


def _llm_response(text: str) -> LLMResponse:
    """Build an LLMResponse for the prose-output agent (Commit 1).

    `text` is the raw assistant prose ending in a trailing parenthetical;
    the engine parses it into `(public_text, intent)` via
    `_extract_parenthetical`. We mirror that on `raw_response.content` so
    `append_turn_to_conversation` can persist the verbatim text.
    """
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    text_block.model_dump = lambda: {"type": "text", "text": text}
    raw.content = [text_block]
    raw.model = "claude-haiku-4-5"
    return LLMResponse(parsed=None, raw_response=raw, content=text, model="claude-haiku-4-5")


@pytest.fixture
def guard_character():
    return CharacterRecord(
        character_id="guard_17",
        name="Captain Vero",
        location="courtyard",
        public_sheet=PublicSheet(
            role="guard captain",
            appearance="Tall, scarred, in polished armor",
            faction="City Watch",
        ),
        private_state=PrivateState(
            goals=["maintain order", "protect the estate"],
            secrets=["knows about the hidden passage"],
            intentions_enabled=True,
        ),
        backstory="Served the estate for twenty years. Rose from foot soldier to captain.",
        personality="Disciplined guard captain with dry humor, clipped and formal speech. Stoic exterior hiding genuine care for those under his protection. His right hand twitches when he's lying. Respects competence.",
    )


@pytest.fixture
def sample_checkpoint():
    return CheckpointFile(
        session=SessionState(session_id="test"),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="courtyard",
                scene_graph={
                    "courtyard": {
                        "name": "Estate Courtyard",
                        "description": "A wide stone courtyard with a dry fountain.",
                    },
                },
            ),
            setting=StorySetting(
                genre="fantasy",
                tone="dark intrigue",
                premise="A young lord navigates court politics.",
            ),
            facts=["The courtyard fountain is dry.", "A storm approaches."],
        ),
    )


@pytest.fixture
def sample_agent_text():
    """Raw prose-+-parenthetical the LLM returns (Commit 1 contract).

    Intentionally exercises every shape the engine cares about:
    - Mid-prose action ("steps closer, hand resting on...") that must
      stay in `public_text`.
    - Inline dialogue verbatim.
    - Trailing parenthetical `(...)` that must be split off as `intent`
      and never reach other agents or the narrator.
    """
    return (
        'He steps closer, hand resting on sword pommel. His eyes narrow '
        'slightly, scanning the perimeter. "You\'ll want to head inside. '
        "Storm's coming.\" "
        "(Watch this newcomer more closely — the timing is wrong.)"
    )


# --- Context builder tests ---

class TestContextBuilder:
    def test_build_character_packet_identity(self, guard_character):
        packet = build_character_packet(guard_character)
        assert packet["character_id"] == "guard_17"
        assert packet["character_name"] == "Captain Vero"
        assert "twenty years" in packet["character_backstory"]
        assert "right hand twitches" in packet["character_personality"]

    def test_build_character_state_dynamic(self, guard_character):
        state = build_character_state(guard_character)
        assert "maintain order" in state["character_goals"]
        assert "hidden passage" in state["character_secrets"]

    def test_build_character_state_empty(self):
        char = CharacterRecord(character_id="minimal", name="Nobody")
        state = build_character_state(char)
        assert state["character_goals"] == "None specified."
        assert state["character_secrets"] == "None."

    def test_build_scene_context(self, sample_checkpoint):
        context = build_scene_context(sample_checkpoint)
        assert "Estate Courtyard" in context
        assert "dry fountain" in context

    def test_build_scene_context_keyed_to_character_location(self, sample_checkpoint, guard_character):
        """v11-r7h: build_scene_context honors the character's actual
        location, not the importer's pivot. After the guard moves to the
        archive, his agent prompt must show the archive — not the
        starting courtyard."""
        sample_checkpoint.world_state.locations.scene_graph["archive"] = {
            "name": "Sealed Archive",
            "description": "Iron-banded shelves stretching into the dark.",
        }
        guard_character.location = "archive"
        sample_checkpoint.characters = [guard_character]
        context = build_scene_context(
            sample_checkpoint, guard_character.character_id,
        )
        assert "Sealed Archive" in context
        assert "Iron-banded" in context
        # Importer pivot (courtyard) must NOT leak in.
        assert "Estate Courtyard" not in context

    def test_build_scene_context_no_character_falls_back(self, sample_checkpoint):
        """Pre-r7h behavior preserved: callers that don't have an actor
        binding (legacy paths, /scene UI surfaces) keep getting the
        importer's pivot scene."""
        context = build_scene_context(sample_checkpoint)
        assert "Estate Courtyard" in context
        context_none = build_scene_context(sample_checkpoint, None)
        assert "Estate Courtyard" in context_none

    def test_build_world_context_legacy_fallback(self, sample_checkpoint, guard_character):
        """Pre-v2 characters (known_context=="") fall back to global lore/facts."""
        assert guard_character.known_context == ""
        context = build_world_context(guard_character, sample_checkpoint)
        assert "fantasy" in context
        assert "fountain is dry" in context

    def test_build_world_context_uses_envelope(self, sample_checkpoint, guard_character):
        """When the character carries a known_context envelope, that IS the
        world context — global lore doesn't bleed in."""
        guard_character.known_context = "The courtyard is wet. You heard shouting earlier."
        context = build_world_context(guard_character, sample_checkpoint)
        assert context == guard_character.known_context
        # Global lore/premise absent
        assert "fantasy" not in context

    def test_format_observed_facts(self):
        facts = ["Player looks around.", "Player touches the wall."]
        formatted = format_observed_facts(facts)
        assert "Player looks around" in formatted
        assert "Player touches the wall" in formatted

    def test_format_observed_facts_empty(self):
        formatted = format_observed_facts([])
        assert "nothing unusual" in formatted

    def test_format_pending_empty(self, guard_character):
        assert format_pending_observations_block(guard_character) == ""

    def test_format_pending_nonempty(self, guard_character):
        guard_character.pending_observations = [
            "[Turn 3] A shout in the hall.",
            "[Turn 4] Footsteps receding.",
        ]
        block = format_pending_observations_block(guard_character)
        assert "Since your last response" in block
        assert "shout in the hall" in block
        assert "Footsteps receding" in block


class TestSceneResolution:
    """v11-r7h: actor-keyed scene context. Pre-r7h, every router /
    narrator / agent context block read `world_state.locations.
    current_scene_id` directly — meaning the LLM was told "you are at
    the starting scene" forever, regardless of where the actor had
    actually moved. Combined with the dead `scene_delta.new_scene_id`
    field (which structurally stranded actors at their starting
    location), this produced the desync that surfaced in the playtest
    where Mira was narrated as moving but her checkpoint said she
    hadn't.

    These tests pin the new helper + the two builder functions that
    consume it. The dispatcher / narrator wrappers (which delegate
    here) are exercised via the integration paths in
    test_orchestrator_v11.py."""

    def _ckpt(self, *, current="courtyard", graph=None):
        graph = graph or {
            "courtyard": {"name": "Estate Courtyard", "description": "Wide stones."},
            "archive": {"name": "Sealed Archive", "description": "Iron-banded shelves."},
        }
        return CheckpointFile(
            session=SessionState(session_id="t"),
            world_state=WorldState(
                locations=LocationState(
                    current_scene_id=current, scene_graph=graph,
                ),
                setting=StorySetting(),
            ),
        )

    def _char(self, cid: str, *, location: str = ""):
        return CharacterRecord(
            character_id=cid,
            name=cid.title(),
            location=location,
            public_sheet=PublicSheet(role="role"),
        )

    def test_resolve_uses_character_location(self):
        ckpt = self._ckpt()
        ckpt.characters = [self._char("guard", location="archive")]
        assert resolve_scene_for_character(ckpt, "guard") == "archive"

    def test_resolve_falls_back_when_character_unset(self):
        """Character has `location=""` (schema default — legacy import
        path or pre-spawn race). Fall back to `current_scene_id` so the
        prompt isn't empty."""
        ckpt = self._ckpt()
        ckpt.characters = [self._char("guard", location="")]
        assert resolve_scene_for_character(ckpt, "guard") == "courtyard"

    def test_resolve_falls_back_when_character_missing(self):
        """Unknown character_id (typo, race) — fall back, don't raise."""
        ckpt = self._ckpt()
        ckpt.characters = []
        assert resolve_scene_for_character(ckpt, "ghost") == "courtyard"

    def test_resolve_none_character_id_returns_pivot(self):
        """Legacy callers that don't pass a character_id keep the old
        importer-pivot answer."""
        ckpt = self._ckpt()
        assert resolve_scene_for_character(ckpt, None) == "courtyard"

    def test_characters_present_keyed_to_actor_location(self):
        """Actor moved to the archive; `build_characters_present` must
        list who's IN the archive (only the actor) — not who's at the
        importer's pivot. Pre-r7h: this returned the courtyard's roster
        and the actor saw "you're alone" in the place they'd just
        physically left."""
        ckpt = self._ckpt()
        actor = self._char("guard", location="archive")
        bystander_at_pivot = self._char("steward", location="courtyard")
        bystander_at_archive = self._char("scribe", location="archive")
        ckpt.characters = [actor, bystander_at_pivot, bystander_at_archive]

        present = build_characters_present(actor, ckpt)
        # The scribe (same scene as actor) is named.
        assert "Scribe" in present
        # The steward (back at the pivot) is NOT.
        assert "Steward" not in present

class TestCharacterAgent:
    @pytest.mark.asyncio
    async def test_basic_response(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        result = await agent.respond(
            guard_character,
            ["Player looks around the courtyard."],
            sample_checkpoint,
        )

        assert result.character_id == "guard_17"
        # Dialogue + actions live in public_text; intent is split off
        # into the private parenthetical and MUST not bleed into public.
        assert "storm" in result.public_text.lower()
        assert "steps closer" in result.public_text
        assert "Watch this newcomer" in result.intent
        assert "Watch this newcomer" not in result.public_text

    @pytest.mark.asyncio
    async def test_character_id_always_actor(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        """The engine stamps `character_id` from the actor record, not from
        the LLM. Even if a misbehaving model emitted a name in the prose
        that resembled another character, the schema's `character_id` field
        is set by the engine, not parsed from prose. Smoke that contract."""
        mock_client.complete.return_value = _llm_response(
            'A flicker of irritation. "Storm\'s coming." (Stalling.)'
        )
        agent = CharacterAgent(mock_client, prompt_manager)
        result = await agent.respond(
            guard_character, ["fact"], sample_checkpoint,
        )
        assert result.character_id == "guard_17"

    @pytest.mark.asyncio
    async def test_missing_trailing_parenthetical_yields_empty_intent(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, caplog,
    ):
        """Misbehaving model: omits the trailing parenthetical entirely.
        Engine must NOT crash — it logs a warning, returns the raw prose
        as `public_text`, and writes an empty `intent`. Empty intent is
        what Commit 3's `last_intent` writeback uses to short-circuit a
        spurious overwrite of the prior turn's interior."""
        import logging
        mock_client.complete.return_value = _llm_response(
            'He nods curtly. "Move along."'
        )
        agent = CharacterAgent(mock_client, prompt_manager)
        with caplog.at_level(logging.WARNING):
            result = await agent.respond(
                guard_character, ["fact"], sample_checkpoint,
            )
        assert result.public_text  # full prose preserved
        assert result.intent == ""
        assert any(
            "missing trailing parenthetical" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_prompt_contains_character_context(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(
            guard_character, ["Player looks around."], sample_checkpoint,
        )

        call_args = mock_client.complete.call_args
        prompt = "\n".join(
            m["content"] for m in call_args.kwargs["messages"]
            if isinstance(m["content"], str)
        )
        assert "Captain Vero" in prompt
        assert "guard captain" in prompt
        assert "clipped and formal" in prompt
        assert "hidden passage" in prompt
        assert "twenty years" in prompt
        assert "right hand twitches" in prompt

    @pytest.mark.asyncio
    async def test_prompt_contains_observed_facts(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(
            guard_character,
            ["Player picks up a rock.", "Player throws the rock at the wall."],
            sample_checkpoint,
        )

        call_args = mock_client.complete.call_args
        prompt = "\n".join(
            m["content"] for m in call_args.kwargs["messages"]
            if isinstance(m["content"], str)
        )
        assert "picks up a rock" in prompt
        assert "throws the rock" in prompt

    @pytest.mark.asyncio
    async def test_uses_agent_role(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, ["fact"], sample_checkpoint)

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["role"] == "agent"
        # Commit 1: the agent emits prose + parenthetical, NOT structured
        # JSON. response_model is intentionally absent so the LLM is free
        # to write natural language.
        assert "response_model" not in call_args.kwargs
        assert call_args.kwargs["compact"] is True

    @pytest.mark.asyncio
    async def test_appends_to_rolling_conversation(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        assert sample_checkpoint.character_conversations == {}

        await agent.respond(
            guard_character, ["fact1"], sample_checkpoint,
        )

        convo = sample_checkpoint.character_conversations["guard_17"]
        assert len(convo) == 2
        assert convo[0].role == "user"
        assert convo[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_pending_observations_flushed_and_cleared(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        guard_character.pending_observations = [
            "[Turn 2] A door slammed.",
            "[Turn 3] Footsteps in the hall.",
        ]
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(
            guard_character, ["current fact"], sample_checkpoint,
        )

        # User message should contain the flushed observations.
        call_args = mock_client.complete.call_args
        user_msg = call_args.kwargs["messages"][-1]["content"]
        user_text = user_msg if isinstance(user_msg, str) else user_msg[0]["text"]
        assert "door slammed" in user_text
        assert "Footsteps" in user_text
        # And the pending list is cleared on the character.
        assert guard_character.pending_observations == []

    @pytest.mark.asyncio
    async def test_sends_prior_conversation_as_history(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        sample_checkpoint.character_conversations["guard_17"] = [
            ConversationMessage(role="user", content="prior user content"),
            ConversationMessage(
                role="assistant",
                content=[{"type": "text", "text": "prior response"}],
            ),
        ]
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, ["fact"], sample_checkpoint)

        # Expect: system, prior user, prior assistant, current user
        messages = mock_client.complete.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "prior user content"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
