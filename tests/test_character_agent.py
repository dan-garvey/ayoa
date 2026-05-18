"""Tests for the Character Agent engine (rolling-conversation architecture)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.character_agent import CharacterAgent, _extract_parenthetical
from app.engine.context_builder import (
    build_character_packet,
    build_character_state,
    build_world_context,
    format_elapsed_agent_turn_block,
    format_pending_observations_block,
    resolve_location_for_character,
)
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import (
    AGENT_PERCEPTION_HEADER,
    AGENT_TURN_HEADER,
)
from app.llm.client import LLMClient, LLMResponse
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.dnd_spatial import DndBattleMapState, DndBattleMapToken
from app.schemas.state import (
    DndCombatantState,
    DndCombatState,
    DndRuntimeEffect,
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

    def test_format_elapsed_turn_empty_without_prior_agent_turn(
        self, sample_checkpoint, guard_character,
    ):
        assert (
            format_elapsed_agent_turn_block(
                guard_character, sample_checkpoint,
            ) == ""
        )

    def test_format_elapsed_turn_uses_session_leading_time(
        self, sample_checkpoint, guard_character,
    ):
        guard_character.last_agent_turn_at_s = 10
        guard_character.clock_at_s = 50
        sample_checkpoint.session.leading_at_s = 80

        block = format_elapsed_agent_turn_block(
            guard_character, sample_checkpoint,
        )

        assert "Time Since Your Last Turn" in block
        assert "1 minute and 10 seconds" in block


class TestLocationResolution:
    """Actor-keyed location context.

    Runtime no longer keeps a scene graph or global current scene. The
    helper returns the character's own location label and otherwise stays
    silent.
    """

    def _ckpt(self):
        return CheckpointFile(
            session=SessionState(session_id="t"),
            world_state=WorldState(
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
        assert resolve_location_for_character(ckpt, "guard") == "archive"

    def test_resolve_returns_empty_when_character_unset(self):
        """Character has `location=""`; callers must handle the empty case."""
        ckpt = self._ckpt()
        ckpt.characters = [self._char("guard", location="")]
        assert resolve_location_for_character(ckpt, "guard") == ""

    def test_resolve_returns_empty_when_character_missing(self):
        """Unknown character_id (typo, race) — return "", don't raise."""
        ckpt = self._ckpt()
        ckpt.characters = []
        assert resolve_location_for_character(ckpt, "ghost") == ""

    def test_resolve_none_character_id_returns_empty(self):
        """Callers that don't pass a character_id get ""."""
        ckpt = self._ckpt()
        assert resolve_location_for_character(ckpt, None) == ""


class TestPovLocationForUser:
    """Player-facing location resolution. The CLI status line, Discord
    embeds, and the takeover prompt all read `pov_location_for_user`.
    The function must (a) prefer the asking
    user's bound character, (b) fall through cleanly to the creator
    binding and then "first is_playable," and (c) NEVER hand back a
    culled character's last-known location — that's a stale ghost
    reading."""

    def _ckpt(self):
        return CheckpointFile(
            session=SessionState(session_id="t"),
            world_state=WorldState(
                setting=StorySetting(),
            ),
        )

    def _player(self, cid: str, *, location: str = "", status: str = "active"):
        c = CharacterRecord(
            character_id=cid,
            name=cid.title(),
            location=location,
            public_sheet=PublicSheet(role="protagonist"),
        )
        c.is_playable = True
        c.status = status
        return c

    def test_user_id_binding_wins(self):
        from app.engine.context_builder import pov_location_for_user

        ckpt = self._ckpt()
        ckpt.characters = [
            self._player("p1", location="courtyard"),
            self._player("p2", location="archive"),
        ]
        ckpt.session.character_bindings = {"p1": "11", "p2": "22"}
        ckpt.session.player_character_id = "p1"

        assert pov_location_for_user(ckpt, user_id="22") == "archive"
        assert pov_location_for_user(ckpt, user_id="11") == "courtyard"

    def test_falls_back_to_creator_binding(self):
        from app.engine.context_builder import pov_location_for_user

        ckpt = self._ckpt()
        ckpt.characters = [self._player("p1", location="archive")]
        ckpt.session.player_character_id = "p1"
        assert pov_location_for_user(ckpt) == "archive"

    def test_skips_culled_player(self):
        """A culled player's last-known location must not surface as
        "where the action is." Bug-3: takeover prompts and CLI status
        lines were rendering a dead player's location as the active one
        because pov_location_for_user only checked is_playable + location,
        never status."""
        from app.engine.context_builder import pov_location_for_user

        ckpt = self._ckpt()
        dead = self._player("ghost", location="archive", status="culled")
        live = self._player("hero", location="courtyard", status="active")
        ckpt.characters = [dead, live]
        ckpt.session.player_character_id = "ghost"

        # Creator binding points at the dead one — must skip and find
        # the live `is_playable`.
        assert pov_location_for_user(ckpt) == "courtyard"

    def test_skips_culled_via_user_id_lookup(self):
        from app.engine.context_builder import pov_location_for_user

        ckpt = self._ckpt()
        dead = self._player("dead_pc", location="archive", status="culled")
        ckpt.characters = [dead]
        ckpt.session.character_bindings = {"dead_pc": "99"}

        # User's bound character was culled; nothing else to fall
        # back to -> "" (UI renders "(no active location)").
        assert pov_location_for_user(ckpt, user_id="99") == ""

    def test_returns_empty_when_no_player(self):
        from app.engine.context_builder import pov_location_for_user

        ckpt = self._ckpt()
        npc = CharacterRecord(
            character_id="npc",
            name="NPC",
            location="courtyard",
            public_sheet=PublicSheet(role="r"),
        )
        npc.is_playable = False
        ckpt.characters = [npc]
        assert pov_location_for_user(ckpt) == ""


class TestCharacterAgent:
    @pytest.mark.asyncio
    async def test_basic_response(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        result = await agent.respond(guard_character, sample_checkpoint)

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
        result = await agent.respond(guard_character, sample_checkpoint)
        assert result.character_id == "guard_17"

    @pytest.mark.asyncio
    async def test_missing_trailing_parenthetical_yields_empty_intent(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, caplog,
    ):
        """Misbehaving model: omits the trailing parenthetical entirely.
        Engine must NOT crash — it logs a warning, returns the raw
        prose as `public_text`, and writes an empty `intent`. Routing
        downstream still works on `public_text`; the lost parse only
        means this turn's parenthetical-vs-prose split is fuzzy for
        the few consumers that strip the trailing paren."""
        import logging
        mock_client.complete.return_value = _llm_response(
            'He nods curtly. "Move along."'
        )
        agent = CharacterAgent(mock_client, prompt_manager)
        with caplog.at_level(logging.WARNING):
            result = await agent.respond(guard_character, sample_checkpoint)
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

        await agent.respond(guard_character, sample_checkpoint)

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
    async def test_elapsed_turn_context_is_user_tail_only(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        guard_character.last_agent_turn_at_s = 15
        guard_character.clock_at_s = 45
        sample_checkpoint.session.leading_at_s = 75
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, sample_checkpoint)

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_text = messages[0]["content"]
        user_text = messages[-1]["content"]
        assert "Time Since Your Last Turn" not in system_text
        assert "Time Since Your Last Turn" in user_text
        assert "1 minute" in user_text

    @pytest.mark.asyncio
    async def test_respond_commit_updates_last_agent_turn_time(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        guard_character.clock_at_s = 20
        sample_checkpoint.session.leading_at_s = 35
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, sample_checkpoint)

        assert guard_character.last_agent_turn_at_s == 35

    @pytest.mark.asyncio
    async def test_dnd_ruleset_addon_is_cached_system_prefix(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        sample_checkpoint.session.config.settings.ruleset_id = "dnd5e_basic"
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, sample_checkpoint)

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_text = messages[0]["content"]
        user_text = messages[-1]["content"]
        assert "<ruleset_addon>" in system_text
        assert system_text.index("<ruleset_addon>") < system_text.index("<role>")
        assert "Captain Vero" not in system_text
        assert "Captain Vero" in user_text
        assert "Active D&D 5e initiative is running" not in system_text
        assert "Active D&D 5e initiative is running" not in user_text

    @pytest.mark.asyncio
    async def test_dnd_combat_marker_is_live_user_context(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        sample_checkpoint.session.config.settings.ruleset_id = "dnd5e_basic"
        guard_character.mechanics = {
            "dnd5e_sheet": {
                "statblock": {
                    "actions": [
                        {
                            "id": "blade",
                            "name": "Blade",
                            "attack": {
                                "bonus": 5,
                                "damage": "1d8+3 slashing",
                                "range": "5 ft",
                            },
                        }
                    ]
                }
            }
        }
        sample_checkpoint.session.active_combat = DndCombatState(
            round_number=2,
            turn_index=0,
            battle_map=DndBattleMapState(
                present=True,
                map_name="Gatehouse",
                width=8,
                height=6,
                tokens=[
                    DndBattleMapToken(
                        token_id="guard_17",
                        character_id="guard_17",
                        label="Captain Vero",
                        x=1,
                        y=1,
                    ),
                    DndBattleMapToken(
                        token_id="raider",
                        character_id="raider",
                        label="Raider",
                        x=4,
                        y=1,
                    ),
                ],
            ),
            combatants=[
                DndCombatantState(
                    combatant_id="guard_17",
                    character_id="guard_17",
                    name="Captain Vero",
                    armor_class=16,
                    hit_points_current=22,
                    hit_points_max=31,
                    initiative_roll=17,
                    initiative_total=99,
                    initiative_detail="d20(17) + 82 = 99",
                    death_save_successes=2,
                    death_save_failures=1,
                    pending_initiating_action="I cut down the raider.",
                    conditions=["grappled"],
                    active_effects=[
                        DndRuntimeEffect(
                            effect_id="eff_bless",
                            name="Bless",
                            slug="bless",
                            target_id="guard_17",
                            conditions=["blessed"],
                            remaining_rounds=4,
                        )
                    ],
                ),
                DndCombatantState(
                    combatant_id="raider",
                    character_id="raider",
                    name="Raider",
                ),
            ],
        )
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, sample_checkpoint)

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_text = messages[0]["content"]
        user_text = messages[-1]["content"]
        assert "## D&D Combat" not in system_text
        assert "## D&D Combat" in user_text
        assert "Round: 2." in user_text
        assert "It is your initiative turn." in user_text
        assert (
            "AC 16; HP 22/31; conditions: grappled; effects: Bless"
            in user_text
        )
        assert "Available combat actions:" in user_text
        assert "Blade; id blade; attack +5; damage 1d8+3 slashing; range 5 ft" in user_text
        assert "Before initiative, you declared this pending intent: I cut down the raider." in user_text
        assert "## Tactical Map" in user_text
        assert "Gatehouse" in user_text
        assert "raider 15 ft" in user_text
        assert "Gatehouse" not in system_text
        assert "d20" not in user_text
        assert "99" not in user_text
        assert "remaining_rounds" not in user_text
        assert "death save" not in user_text.lower()
        assert "successes" not in user_text
        assert "failures" not in user_text
        persisted_user = sample_checkpoint.character_conversations[
            "guard_17"
        ][0].content
        assert isinstance(persisted_user, str)
        assert "## D&D Combat" in persisted_user
        assert "## Tactical Map" not in persisted_user
        assert "Gatehouse" not in persisted_user

    @pytest.mark.asyncio
    async def test_foreground_local_context_is_live_only_not_saved(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(
            guard_character,
            sample_checkpoint,
            frame="foreground",
            local_context=(
                "Immediate combat instruction: choose one listed action and "
                "name a target."
            ),
        )

        live_user = mock_client.complete.call_args.kwargs["messages"][-1]["content"]
        assert "## Local Context" in live_user
        assert "choose one listed action" in live_user

        saved_user = sample_checkpoint.character_conversations[
            "guard_17"
        ][0].content
        assert "## Local Context" not in saved_user
        assert "choose one listed action" not in saved_user
        assert "## Turn Frame\nforeground" in saved_user

    @pytest.mark.asyncio
    async def test_pending_observations_carry_in_scene_perception(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        """In-scene perception now reaches the agent through the
        `pending_observations` inbox (populated by `broadcast_event`)
        rather than the dead `## What You Observe This Turn` block.
        Smoke that the inbox path delivers facts into the user
        message verbatim — that's the only path left for "what just
        happened in your scene this beat" to reach the agent."""
        guard_character.pending_observations = [
            "Player picks up a rock.",
            "Player throws the rock at the wall.",
        ]
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, sample_checkpoint)

        call_args = mock_client.complete.call_args
        user_msg = call_args.kwargs["messages"][-1]["content"]
        user_text = user_msg if isinstance(user_msg, str) else user_msg[0]["text"]
        assert "picks up a rock" in user_text
        assert "throws the rock" in user_text

    @pytest.mark.asyncio
    async def test_uses_agent_role(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, sample_checkpoint)

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["role"] == "agent"
        # Commit 1: the agent emits prose + parenthetical, NOT structured
        # JSON. response_model is intentionally absent so the LLM is free
        # to write natural language.
        assert "response_model" not in call_args.kwargs
        assert call_args.kwargs["compact"] is True

    @pytest.mark.asyncio
    async def test_standard_agent_uses_haiku_role(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        guard_character.agent_tier = CharacterAgentTier.standard
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, sample_checkpoint)

        assert mock_client.complete.call_args.kwargs["role"] == "agent_standard"

    @pytest.mark.asyncio
    async def test_utility_agent_uses_convenience_role(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        guard_character.agent_tier = CharacterAgentTier.utility
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, sample_checkpoint)

        assert mock_client.complete.call_args.kwargs["role"] == "agent_convenience"

    @pytest.mark.asyncio
    async def test_legacy_convenience_agent_uses_cheaper_role(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        guard_character.agent_tier = CharacterAgentTier.convenience
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, sample_checkpoint)

        assert mock_client.complete.call_args.kwargs["role"] == "agent_convenience"

    @pytest.mark.asyncio
    async def test_appends_to_rolling_conversation(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        assert sample_checkpoint.character_conversations == {}

        await agent.respond(guard_character, sample_checkpoint)

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

        await agent.respond(guard_character, sample_checkpoint)

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

        await agent.respond(guard_character, sample_checkpoint)

        # Expect: system, prior user, prior assistant, current user
        messages = mock_client.complete.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "prior user content"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"


class TestExtractParenthetical:
    """Direct unit coverage of the parser the engine uses to split agent
    prose from its trailing parenthetical.

    The agent's freshest interior is exactly what the parenthetical
    encloses, and a wrong split is one of two failure shapes:
      - intent leaks into public_text → router/narrator/other agents
        see another character's interior. Information-asymmetry
        regression (see DESIGN.md §4.5 "Agents Author Intentions, Not
        State" — the per-character interior asymmetry rule).
      - public prose gets eaten as intent → the cascade or narrator
        sees a silent beat from a character that actually spoke.

    These tests pin the corner cases so neither failure mode can
    silently re-enter via a parser tweak.
    """

    def test_simple_trailing_parenthetical(self):
        public, intent = _extract_parenthetical(
            'He nods. "Move along." (Watching the gate.)'
        )
        assert public == 'He nods. "Move along."'
        assert intent == "Watching the gate."

    def test_no_trailing_paren_returns_text_and_empty(self):
        # Misbehaving model: omits the parenthetical entirely. Engine
        # gets the prose back as `public_text`; intent is "" so no
        # downstream consumer sees fabricated interior.
        public, intent = _extract_parenthetical("He nods curtly.")
        assert public == "He nods curtly."
        assert intent == ""

    def test_empty_string(self):
        public, intent = _extract_parenthetical("")
        assert public == ""
        assert intent == ""

    def test_whitespace_only(self):
        # Trailing whitespace is trimmed before the )-detection runs;
        # an all-whitespace string still has no closing paren so it's
        # treated as a missing trailing parenthetical.
        public, intent = _extract_parenthetical("   \n  \t ")
        assert intent == ""
        # We don't promise an exact public_text shape for this
        # degenerate input — just that no parse explosion happens.

    def test_mid_prose_paren_not_treated_as_intent(self):
        # Stage directions inside prose ("she pauses (just long enough
        # to be noticed)") must NOT be split off — only the FINAL
        # group at the very end of the trimmed text counts as intent.
        text = (
            "She pauses (just long enough to be noticed) and turns her "
            'head. "Yes?" (Trying to seem unbothered.)'
        )
        public, intent = _extract_parenthetical(text)
        assert intent == "Trying to seem unbothered."
        # The mid-prose stage direction stays in public_text.
        assert "(just long enough to be noticed)" in public
        # And the trailing parenthetical's contents are stripped from
        # public_text — no double-render.
        assert "Trying to seem unbothered" not in public

    def test_nested_parens_in_intent(self):
        # Balanced nesting at the end. The parser walks ) depth so a
        # nested pair inside the trailing group must be preserved
        # verbatim rather than truncating intent at the first '('.
        text = (
            'He shrugs. (Plan: stall (until the bell rings) and then '
            "slip out the side door.)"
        )
        public, intent = _extract_parenthetical(text)
        assert public == "He shrugs."
        # Whole nested expression survives as intent.
        assert intent == (
            "Plan: stall (until the bell rings) and then slip out "
            "the side door."
        )

    def test_unbalanced_trailing_paren_warns_and_returns_text(self):
        # `)` at the end with no matching `(` upstream — the parser
        # walks back, never finds depth==0, logs a warning, and
        # returns the raw text. The model's malformed output doesn't
        # crash the engine and doesn't fabricate an empty intent
        # cluster.
        public, intent = _extract_parenthetical(
            'He looks up. "Strange weather, that.")'
        )
        assert intent == ""
        # Public text is the raw original (un-stripped of the dangling
        # `)`) — losing the prose would be a worse failure than
        # leaving the malformed char in place.
        assert public.endswith(")")

    def test_multiline_trailing_parenthetical(self):
        # Trailing paren can span newlines (e.g. agents writing out
        # a multi-clause interior). The split point must still be the
        # final `(` matching the closing `)` at end of the trimmed
        # text — newlines are not balance markers.
        text = (
            'He bows his head. "As you wish."\n'
            "(Two thoughts at once: keep her placated,\n"
            "and find the steward before sundown.)"
        )
        public, intent = _extract_parenthetical(text)
        assert public == 'He bows his head. "As you wish."'
        assert "Two thoughts at once" in intent
        assert "find the steward" in intent
        # Parenthetical body's leading newline is stripped (the
        # parser uses `.strip()` on the inner span).
        assert not intent.startswith("\n")

    def test_trailing_whitespace_after_paren_is_tolerated(self):
        # Some models append a stray newline or trailing space after
        # the closing `)`. The parser rstrips before the `endswith`
        # check, so the parse still succeeds.
        text = 'He nods. (Stalling for time.)   \n'
        public, intent = _extract_parenthetical(text)
        assert public == "He nods."
        assert intent == "Stalling for time."

    def test_empty_parenthetical_yields_empty_intent(self):
        # `(...)` that contains nothing → intent is "", which short-
        # circuits any "agent had no interior this turn" downstream
        # check without misclassifying as a missing trailing paren.
        text = 'He shrugs. ()'
        public, intent = _extract_parenthetical(text)
        assert public == "He shrugs."
        assert intent == ""


# `TestPriorResponsesLeakGuard` was deleted in v11-r10 along with
# `format_prior_responses`. The on-stage agent body no longer carries
# a "## Other Characters' Responses This Turn" block — production
# always passed `prior_responses=None` because cascade NPCs already
# see prior responses through their own `pending_observations` inbox
# (each cascade event broadcasts to scene-mates). The cross-agent
# intent-leak chokepoint moved to `_extract_parenthetical` (which
# splits public_text from intent at the source) and the router-
# intention block in turn_loop_dispatcher (which forwards only
# `output.public_text`). See `tests/test_turn_loop.py` for the
# broadcast-event coverage that replaced this suite's role.


class TestUnifiedAgentCacheLineage:
    """v11 cache-trail invariant: agent turn frames share ONE system
    prompt, while rolling histories remain per character.

    Pre-v11 the engine had two separate templates (an `agent` and
    separate prompt pair) for foreground and background calls. Both
    rendered into the same `character_conversations[id]` history,
    but each had a DIFFERENT system prefix — so every mode switch
    invalidated the Anthropic prompt cache for that character. On a
    long session the same character pays for two cache lineages and
    eats a cache-write per mode flip.

    The v11 fix is a single unified prompt with a first-token
    turn marker plus a user-tail frame. These
    tests pin that fix:

      - **Same template name**: both frames load `agent`.
      - **Identical system prefix**: byte-for-byte equality between
        modes and between characters under the same ruleset. This is
        THE invariant — if it regresses, the cache trail re-splits and
        the bug is back.
      - **Mode header is the first user-message line**: the prompt's
        "Mode Routing" section keys off the first token of the user
        message; if the marker drifts off line 1 the agent's mode
        signal is buried mid-message.
    """

    @pytest.mark.asyncio
    async def test_foreground_and_background_share_same_system_prefix(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        # Run respond, capture system message.
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.respond(guard_character, sample_checkpoint)
        respond_messages = mock_client.complete.call_args.kwargs["messages"]
        respond_system = respond_messages[0]
        assert respond_system["role"] == "system"

        # Reset call captures and run a background turn on the SAME character
        # + checkpoint. The path must produce a byte-identical
        # system message — that's the cache-trail invariant.
        mock_client.complete.reset_mock()
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        await agent.turn(guard_character, sample_checkpoint, frame="background")
        background_messages = mock_client.complete.call_args.kwargs["messages"]
        background_system = background_messages[0]
        assert background_system["role"] == "system"

        # Byte-equality: the system prompts MUST match across modes.
        # Any divergence (a stray newline, a mode-conditional line)
        # invalidates the Anthropic prompt cache and resurrects the
        # cache-trail proliferation bug.
        assert background_system["content"] == respond_system["content"]

    @pytest.mark.asyncio
    async def test_different_characters_share_same_system_prefix(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        other = guard_character.model_copy(deep=True)
        other.character_id = "mistress_vale"
        other.name = "Mistress Vale"
        other.public_sheet = PublicSheet(
            role="estate spymaster",
            appearance="Silver mask and black gloves",
            faction="House Vale",
        )
        other.private_state = PrivateState(
            goals=["control the household intelligence network"],
            current_objectives=["identify who bribed the footman"],
            secrets=["keeps a second ledger in the chapel wall"],
            intentions_enabled=True,
        )
        other.backstory = "Raised in the archive rooms and trusted by no one."
        other.personality = "Soft-spoken, precise, and impossible to hurry."

        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.respond(guard_character, sample_checkpoint)
        guard_messages = mock_client.complete.call_args.kwargs["messages"]

        mock_client.complete.reset_mock()
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        await agent.respond(other, sample_checkpoint)
        other_messages = mock_client.complete.call_args.kwargs["messages"]

        guard_system = guard_messages[0]["content"]
        other_system = other_messages[0]["content"]
        assert guard_system == other_system
        assert "Captain Vero" not in guard_system
        assert "Mistress Vale" not in other_system
        assert "Captain Vero" in guard_messages[-1]["content"]
        assert "Mistress Vale" in other_messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_respond_user_message_starts_with_agent_turn_header(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.respond(guard_character, sample_checkpoint)
        messages = mock_client.complete.call_args.kwargs["messages"]
        user_content = messages[-1]["content"]
        # First non-empty line must be the mode header — the
        # prompt's "Mode Routing" section keys off this exact
        # first-token signal.
        first_line = next(
            ln for ln in user_content.splitlines() if ln.strip()
        )
        assert first_line == AGENT_TURN_HEADER
        assert "## Turn Frame\nforeground" in user_content

    @pytest.mark.asyncio
    async def test_background_user_message_uses_background_frame(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.turn(guard_character, sample_checkpoint, frame="background")
        messages = mock_client.complete.call_args.kwargs["messages"]
        user_content = messages[-1]["content"]
        first_line = next(
            ln for ln in user_content.splitlines() if ln.strip()
        )
        assert first_line == AGENT_TURN_HEADER
        assert "## Turn Frame\nbackground" in user_content

    @pytest.mark.asyncio
    async def test_background_local_context_is_live_only_not_saved(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(
            guard_character,
            sample_checkpoint,
            frame="background",
            local_context=(
                "Location: courtyard\n"
                "Nearby active characters: Steward Lysa (steward_lysa)"
            ),
        )

        live_user = mock_client.complete.call_args.kwargs["messages"][-1]["content"]
        assert "## Local Context" in live_user
        assert "Steward Lysa" in live_user

        saved_user = sample_checkpoint.character_conversations[
            "guard_17"
        ][0].content
        assert "## Local Context" not in saved_user
        assert "Steward Lysa" not in saved_user
        assert "## Turn Frame\nbackground" in saved_user

    @pytest.mark.asyncio
    async def test_background_turn_appends_to_same_rolling_conversation_as_respond(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        # Single rolling history per character — foreground and background
        # both write into `character_conversations[character_id]`.
        # If they ever split (separate history per mode), the
        # agent's interior memory desyncs across modes.
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.respond(guard_character, sample_checkpoint)
        mock_client.complete.return_value = _llm_response(
            'He stands by the window. (Watching the gate.)'
        )
        await agent.turn(guard_character, sample_checkpoint, frame="background")

        # Exactly ONE rolling-history key for this character; both
        # the foreground pair AND the background pair are appended to it.
        keys = list(sample_checkpoint.character_conversations.keys())
        assert keys == ["guard_17"]
        convo = sample_checkpoint.character_conversations["guard_17"]
        # 2 user/assistant pairs = 4 messages total.
        assert len(convo) == 4
        # Sequence: foreground user, foreground asst, background user,
        # background asst.
        assert convo[0].role == "user"
        assert AGENT_TURN_HEADER in convo[0].content
        assert "## Turn Frame\nforeground" in convo[0].content
        assert convo[2].role == "user"
        assert AGENT_TURN_HEADER in convo[2].content
        assert "## Turn Frame\nbackground" in convo[2].content


class TestPerceptionMode:
    """v11-r8a: PERCEPTION mode — observer-agnostic visual loadout.

    Fired by the observation-harvest fork in run_beat (and reachable
    later from /query for "what does X look like?" questions). Three
    load-bearing properties distinguish perception from normal agent turns:

      1. The user-message FIRST LINE is `## PERCEPTION`. The agent
         prompt's "Mode Routing" section keys off this exact token to
         flip into Perception Mode rules; if the marker drifts off
         line 1 the agent's mode signal is buried mid-message.
      2. Perception calls append to the same rolling conversation.
         A character should remember what they established about their
         visual presentation in the scene.
      3. Cache lineage with normal agent turns is preserved: the system
         prompt is byte-identical across all three modes for the
         same ruleset.
    """

    def _llm_text_only(self, text: str) -> LLMResponse:
        # Perception output is plain prose, no parenthetical.
        raw = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        text_block.model_dump = lambda: {"type": "text", "text": text}
        raw.content = [text_block]
        raw.model = "claude-haiku-4-5"
        return LLMResponse(
            parsed=None, raw_response=raw, content=text,
            model="claude-haiku-4-5",
        )

    @pytest.mark.asyncio
    async def test_perceive_returns_text_from_llm(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        loadout = (
            "Polished armor over a clean undertunic, the city watch sigil "
            "centered. He stands at parade rest, hands behind his back."
        )
        mock_client.complete.return_value = self._llm_text_only(loadout)
        agent = CharacterAgent(mock_client, prompt_manager)
        result = await agent.perceive(guard_character, sample_checkpoint)
        assert result == loadout

    @pytest.mark.asyncio
    async def test_perceive_user_message_starts_with_perception_header(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        mock_client.complete.return_value = self._llm_text_only("loadout")
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.perceive(guard_character, sample_checkpoint)
        messages = mock_client.complete.call_args.kwargs["messages"]
        user_content = messages[-1]["content"]
        first_line = next(
            ln for ln in user_content.splitlines() if ln.strip()
        )
        assert first_line == AGENT_PERCEPTION_HEADER
        # Other mode markers must not also appear (mutually exclusive).
        assert AGENT_TURN_HEADER not in user_content
        assert "Hard prose constraint" in user_content
        assert "with the [quality] of someone/people who" in user_content

    @pytest.mark.asyncio
    async def test_perceive_appends_to_rolling_history(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        # Perception is not an on-stage action, but the character should
        # remember the visual loadout they authored for this scene.
        mock_client.complete.return_value = self._llm_text_only(
            "Polished armor, parade-rest posture."
        )
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.perceive(guard_character, sample_checkpoint)
        convo = sample_checkpoint.character_conversations["guard_17"]
        assert len(convo) == 2
        assert convo[0].role == "user"
        assert AGENT_PERCEPTION_HEADER in convo[0].content
        assert convo[1].role == "assistant"
        assert "Polished armor" in convo[1].content[0]["text"]

    @pytest.mark.asyncio
    async def test_perceive_does_not_drain_pending_observations(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        # The pending_observations queue belongs to the next on-stage
        # normal agent turn. Draining it here would silently swallow
        # off-scene perceptions the next on-stage turn needs to react
        # to. The perception render also passes an EMPTY pending
        # block so the loadout isn't primed by "react to these
        # incoming events."
        guard_character.pending_observations = [
            "[off-scene perception] A shout in the courtyard.",
            "[off-scene perception] Bells ring at the gate.",
        ]
        mock_client.complete.return_value = self._llm_text_only("loadout")
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.perceive(guard_character, sample_checkpoint)
        # Inbox preserved for the next normal agent turn.
        assert len(guard_character.pending_observations) == 2
        # And the perception's user message did NOT carry the inbox
        # contents (no priming).
        messages = mock_client.complete.call_args.kwargs["messages"]
        user_content = messages[-1]["content"]
        assert "shout in the courtyard" not in user_content
        assert "Bells ring" not in user_content

    @pytest.mark.asyncio
    async def test_perceive_shares_system_prefix_with_respond(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        # Cache-lineage invariant: respond and perceive must yield
        # byte-identical system prompts so the Anthropic prompt cache
        # hits across modes.
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.respond(guard_character, sample_checkpoint)
        respond_system = mock_client.complete.call_args.kwargs["messages"][0]

        mock_client.complete.reset_mock()
        mock_client.complete.return_value = self._llm_text_only("loadout")
        await agent.perceive(guard_character, sample_checkpoint)
        perceive_system = mock_client.complete.call_args.kwargs["messages"][0]

        assert perceive_system["role"] == "system"
        assert perceive_system["content"] == respond_system["content"]

    @pytest.mark.asyncio
    async def test_perceive_does_not_render_or_update_elapsed_turn_context(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        guard_character.last_agent_turn_at_s = 12
        guard_character.clock_at_s = 40
        sample_checkpoint.session.leading_at_s = 90
        mock_client.complete.return_value = self._llm_text_only("loadout")
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.perceive(guard_character, sample_checkpoint)

        messages = mock_client.complete.call_args.kwargs["messages"]
        user_text = messages[-1]["content"]
        assert "Time Since Your Last Turn" not in user_text
        assert guard_character.last_agent_turn_at_s == 12

    @pytest.mark.asyncio
    async def test_perceive_uses_lower_max_tokens_than_respond(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        # Perception is capped at 3 sentences; the call site uses a
        # smaller token budget than normal agent turns. Pinning the budget
        # so a future "let's give the agent more room" tweak doesn't
        # silently regress to 2000 tokens per perception (which would
        # bloat the cost of a 3-target harvest by 6x).
        mock_client.complete.return_value = self._llm_text_only("loadout")
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.perceive(guard_character, sample_checkpoint)
        max_tokens = mock_client.complete.call_args.kwargs["max_tokens"]
        assert max_tokens <= 1000

    @pytest.mark.asyncio
    async def test_perceive_strips_whitespace(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        # Some models emit a leading/trailing newline; the harvest
        # path appends fragments verbatim into observable_facts so
        # whitespace around the loadout looks ugly in the narrator
        # render.
        mock_client.complete.return_value = self._llm_text_only(
            "\n\n  Polished armor.  \n",
        )
        agent = CharacterAgent(mock_client, prompt_manager)
        result = await agent.perceive(guard_character, sample_checkpoint)
        assert result == "Polished armor."
