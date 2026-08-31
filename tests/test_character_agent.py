"""Tests for the Character Agent engine (rolling-conversation architecture)."""

import re
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.character_agent import (
    CharacterAgent,
    CharacterAgentOutputError,
    _parse_agent_turn_response,
    sanitize_character_public_text,
)
from app.engine.context_builder import (
    build_character_self_packet,
    build_visible_self_packet,
    format_elapsed_agent_turn_block,
    format_pending_observations_block,
    resolve_location_for_character,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.agents import (
    CharacterPresentationChoice,
)
from app.schemas.content_privacy import REDACTED_IMPORT_SENTINEL
from app.schemas.characters import (
    ActorFact,
    ActorRecord,
    CharacterAgentTier,
    CharacterRecord,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.content import ContentKnowledgeEntityState, ContentPackState
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


def _llm_response(
    text: str,
    *,
    usage: dict[str, int] | None = None,
    parsed: object | None = None,
) -> LLMResponse:
    """Build an LLMResponse for the free-form character-agent output."""
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    text_block.model_dump = lambda: {"type": "text", "text": text}
    raw.content = [text_block]
    raw.model = "gpt-5.6-luna"
    return LLMResponse(
        parsed=parsed,
        raw_response=raw,
        content=text,
        model="gpt-5.6-luna",
        usage=usage or {},
    )


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
        actor=ActorRecord(
            may_act_offstage=True,
            facts=[
                ActorFact(
                    text=(
                        "You served the estate for twenty years and rose "
                        "from foot soldier to captain."
                    )
                ),
                ActorFact(
                    text=(
                        "You maintain order and protect the estate. You use "
                        "dry humor, speak formally, and your right hand "
                        "twitches when you lie."
                    )
                ),
                ActorFact(text="You know about the hidden passage."),
            ],
        ),
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
    """Ordinary observable character prose."""
    return (
        'He steps closer, hand resting on sword pommel. His eyes narrow '
        'slightly, scanning the perimeter. "You\'ll want to head inside. '
        "Storm's coming.\""
    )


def _message_text(message: dict) -> str:
    """Flatten one model message for boundary assertions."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _assert_no_legacy_character_markers(messages: list[dict]) -> None:
    """Routing labels must never reach any model-facing message."""
    rendered = "\n".join(_message_text(message) for message in messages)
    for marker in ("## AGENT-TURN", "## PERCEPTION", "## Turn Frame"):
        assert marker not in rendered
    assert not re.search(
        r"(?im)^\s*(?:foreground|background|private)\s*$", rendered
    )


# --- Context builder tests ---

class TestContextBuilder:
    def test_build_character_self_packet_is_second_person(self, guard_character):
        rendered = build_character_self_packet(guard_character)
        assert "You are Captain Vero." in rendered
        for expected in (
            "guard captain",
            "twenty years",
            "right hand twitches",
            "maintain order",
            "hidden passage",
        ):
            assert expected in rendered

    def test_build_visible_self_packet_excludes_private_life(self, guard_character):
        rendered = build_visible_self_packet(guard_character)
        assert "You are Captain Vero." in rendered
        assert "polished armor" in rendered
        assert "right hand twitches" not in rendered
        assert "twenty years" not in rendered
        assert "hidden passage" not in rendered

    def test_build_character_self_packet_keeps_identity_without_facts(self):
        char = CharacterRecord(character_id="minimal", name="Nobody")
        assert build_character_self_packet(char) == "You are Nobody."

    def test_format_pending_empty(self, guard_character):
        assert format_pending_observations_block(guard_character) == ""

    def test_format_pending_nonempty(self, guard_character):
        guard_character.pending_observations = [
            "[Turn 3] A shout in the hall.",
            "[Turn 4] Footsteps receding.",
        ]
        block = format_pending_observations_block(guard_character)
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

        result = await agent.turn(guard_character, sample_checkpoint)

        assert result.character_id == "guard_17"
        # Dialogue and actions remain observable prose in both the public
        # result and the actor's assistant history.
        assert "storm" in result.public_text.lower()
        assert "steps closer" in result.public_text
        assert result.is_silence is False
        saved = sample_checkpoint.character_conversations["guard_17"][1]
        saved_text = saved.content[0]["text"]
        assert saved_text == result.public_text

    @pytest.mark.asyncio
    async def test_deliberate_silence_commits_without_answering_a_demand(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        guard_character.pending_observations = [
            'The captain points at the empty chair. "Answer me now."'
        ]
        mock_client.complete.return_value = _llm_response("<silence/>")
        agent = CharacterAgent(mock_client, prompt_manager)

        result = await agent.turn(guard_character, sample_checkpoint)

        assert result.public_text == ""
        assert result.is_silence is True
        assert guard_character.pending_observations == []
        history = sample_checkpoint.character_conversations["guard_17"]
        assert history[1].content[0]["text"] == "<silence/>"
        assert guard_character.last_agent_turn_at_s is not None

    @pytest.mark.asyncio
    async def test_visual_novel_presentation_is_character_owned_and_stripped_from_history(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        sample_checkpoint.session.config.settings.presentation_mode = (
            "visual_novel"
        )
        response = (
            'He tilts his head. "That does not follow."\n'
            '<presentation>{"use":"skeptical","request":""}</presentation>'
        )
        mock_client.complete.return_value = _llm_response(response)
        agent = CharacterAgent(mock_client, prompt_manager)

        result = await agent.turn(guard_character, sample_checkpoint)

        assert result.public_text == (
            'He tilts his head. "That does not follow."'
        )
        assert result.presentation.use == "skeptical"
        assert (
            guard_character.visuals.visual_novel_presentation.current_variant_key
            == "skeptical"
        )
        live_user = mock_client.complete.await_args.kwargs["messages"][-1][
            "content"
        ]
        assert '<presentation_catalog current="neutral">' in live_user
        saved = sample_checkpoint.character_conversations["guard_17"]
        catalog_start = live_user.index("<presentation_catalog")
        catalog_end = live_user.index("</presentation_catalog>") + len(
            "</presentation_catalog>"
        )
        catalog = live_user[catalog_start:catalog_end]
        assert catalog not in saved[0].content
        assert saved[0].content == live_user.replace(catalog, "", 1)
        for value in (
            "Captain Vero",
            "twenty years",
            "hidden passage",
        ):
            assert value in live_user
            assert value in saved[0].content
        _assert_no_legacy_character_markers(mock_client.complete.await_args.kwargs["messages"])
        saved_assistant = saved[1].content[0]["text"]
        assert "<presentation>" not in saved_assistant
        assert "skeptical" not in saved_assistant
        assert saved_assistant == result.public_text

    @pytest.mark.asyncio
    async def test_visual_novel_footer_is_stripped_without_leak(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        sample_checkpoint.session.config.settings.presentation_mode = (
            "visual_novel"
        )
        response = (
            "He braces beside the gate.\n"
            '<presentation>{"use":"tense","request":""}</presentation>'
        )
        mock_client.complete.return_value = _llm_response(response)

        result = await CharacterAgent(mock_client, prompt_manager).turn(
            guard_character,
            sample_checkpoint,
        )

        assert result.public_text == "He braces beside the gate."
        saved = sample_checkpoint.character_conversations["guard_17"][1]
        assert saved.content[0]["text"] == result.public_text
        assert "presentation" not in result.public_text
        assert (
            guard_character.visuals.visual_novel_presentation.current_variant_key
            == "tense"
        )

    @pytest.mark.asyncio
    async def test_imported_asset_source_sentinels_do_not_reach_prompt(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
        sample_agent_text,
    ):
        sentinels = [
            "source_ref=raw-row",
            "delivery_ref=asset://synthetic/hidden-map",
            "/private/table/source-map.png",
            "raw_ocr=PROTECTED_SOURCE_EXCERPT",
        ]
        guard_character.pending_observations = ["Visible surface. " + " ".join(sentinels)]
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint)

        mock_client.complete.assert_awaited_once()
        messages = mock_client.complete.await_args.kwargs["messages"]
        flat = "\n".join(
            message["content"]
            for message in messages
            if isinstance(message.get("content"), str)
        )
        for sentinel in sentinels:
            assert sentinel not in flat
        assert REDACTED_IMPORT_SENTINEL in flat
        assert "Visible surface." in flat

    @pytest.mark.asyncio
    async def test_imported_content_metadata_does_not_reach_agent_prompt(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
        sample_agent_text,
    ):
        pack_id = "lost_laboratory_kwalish_full_reviewed_v1"
        content_hash = (
            "sha256:0123456789abcdef0123456789abcdef"
            "0123456789abcdef0123456789abcdef"
        )
        compact_ref = (
            f"{pack_id}:agent_context.garret.strategic@{content_hash}"
        )
        sample_checkpoint.session.content_state = {
            pack_id: ContentPackState(
                pack_id=pack_id,
                knowledge_map={
                    "guard_17": ContentKnowledgeEntityState(
                        entity_id="guard_17",
                        known_refs=[compact_ref],
                    )
                },
                metadata={
                    "source_fingerprint": content_hash,
                },
            )
        }
        guard_character.actor = ActorRecord(
            may_act_offstage=True,
            facts=[
                ActorFact(
                    text=(
                        "You are a careful expedition custodian. Pack marker "
                        f"{pack_id}; stat.garret_levistusson."
                    )
                ),
                ActorFact(
                    text=f"You use reviewed notes, not {compact_ref}, to stay grounded."
                ),
                ActorFact(
                    text=(
                        "Garret knows the party route. Reviewed content refs known "
                        f"to you: {compact_ref}. Start near loc.barrier_peaks_route."
                    )
                ),
                ActorFact(text=f"Protect the folios tied to {pack_id}."),
                ActorFact(
                    text="Use area.c2_enhanced_sphinx only when the route reaches it."
                ),
                ActorFact(text=f"The source fingerprint is {content_hash}."),
            ],
        )
        guard_character.location = "loc.barrier_peaks_route"
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint, frame="background")

        messages = mock_client.complete.await_args.kwargs["messages"]
        flat = "\n".join(
            message["content"]
            for message in messages
            if isinstance(message.get("content"), str)
        )
        assert "Garret knows the party route" in flat
        assert "Barrier Peaks Route" in flat
        forbidden = [
            pack_id,
            compact_ref,
            content_hash,
            "sha256:",
            "agent_context.garret.strategic",
            "stat.garret_levistusson",
            "loc.barrier_peaks_route",
            "area.c2_enhanced_sphinx",
            "Reviewed content refs",
        ]
        leaks = [term for term in forbidden if term in flat]
        assert not leaks, "Imported metadata leaked into agent prompt: " + ", ".join(leaks)

    @pytest.mark.asyncio
    async def test_character_id_always_actor(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        """The engine stamps `character_id` from the actor record, not from
        the LLM. Even if a misbehaving model emitted a name in the prose
        that resembled another character, the schema's `character_id` field
        is set by the engine, not parsed from prose. Smoke that contract."""
        mock_client.complete.return_value = _llm_response(
            'A flicker of irritation. "Storm\'s coming."'
        )
        agent = CharacterAgent(mock_client, prompt_manager)
        result = await agent.turn(guard_character, sample_checkpoint)
        assert result.character_id == "guard_17"
        assert "format_repairs" not in agent.last_usage

    @pytest.mark.asyncio
    async def test_ordinary_prose_is_one_call(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint,
    ):
        mock_client.complete.return_value = _llm_response(
            'He nods curtly. "Move along."',
        )
        agent = CharacterAgent(mock_client, prompt_manager)

        result = await agent.turn(guard_character, sample_checkpoint)

        assert result.public_text == 'He nods curtly. "Move along."'
        assert result.is_silence is False
        assert mock_client.complete.await_count == 1
        assert "response_model" not in mock_client.complete.await_args.kwargs
        assert "format_repairs" not in agent.last_usage

    @pytest.mark.asyncio
    async def test_retired_private_marker_fails_before_commit(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint,
    ):
        mock_client.complete.return_value = _llm_response(
            'He nods curtly. <private_carry>I will watch the gate.'
        )
        agent = CharacterAgent(mock_client, prompt_manager)

        with pytest.raises(CharacterAgentOutputError, match="retired"):
            await agent.turn(guard_character, sample_checkpoint)

        assert mock_client.complete.await_count == 1
        assert not sample_checkpoint.character_conversations.get(
            guard_character.character_id
        )
        assert guard_character.last_agent_turn_at_s is None

    @pytest.mark.asyncio
    async def test_retired_private_marker_fails_before_history_or_inbox_mutation(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint,
    ):
        guard_character.pending_observations = ["A bell rings outside."]
        mock_client.complete.return_value = _llm_response(
            "<private_carry>I will ask who rang it.</private_carry>"
        )
        agent = CharacterAgent(mock_client, prompt_manager)

        with pytest.raises(CharacterAgentOutputError, match="retired"):
            await agent.turn(guard_character, sample_checkpoint)

        assert not sample_checkpoint.character_conversations.get(
            guard_character.character_id
        )
        assert guard_character.pending_observations == ["A bell rings outside."]
        assert guard_character.last_agent_turn_at_s is None

    @pytest.mark.asyncio
    async def test_retired_private_marker_after_presentation_fails_before_commit(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint,
    ):
        response = (
            'He nods curtly.\n'
            '<presentation>{"use":"stern","request":""}</presentation>\n'
            '<private_carry>I will watch the gate.</private_carry>'
        )
        sample_checkpoint.session.config.settings.presentation_mode = (
            "visual_novel"
        )
        mock_client.complete.return_value = _llm_response(response)

        with pytest.raises(CharacterAgentOutputError, match="retired"):
            await CharacterAgent(mock_client, prompt_manager).turn(
                guard_character,
                sample_checkpoint,
            )

        assert not sample_checkpoint.character_conversations.get(
            guard_character.character_id
        )
        assert guard_character.last_agent_turn_at_s is None

    @pytest.mark.asyncio
    async def test_misplaced_retired_private_marker_never_leaks(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint,
    ):
        mock_client.complete.return_value = _llm_response(
            '<private_carry>secret</private_carry>\nHe says this publicly.'
        )
        agent = CharacterAgent(mock_client, prompt_manager)

        with pytest.raises(CharacterAgentOutputError, match="retired"):
            await agent.turn(guard_character, sample_checkpoint)

        assert not sample_checkpoint.character_conversations.get(
            guard_character.character_id
        )
        assert guard_character.last_agent_turn_at_s is None

    @pytest.mark.asyncio
    async def test_prompt_contains_character_context(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint)

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_text = _message_text(messages[0])
        user_text = _message_text(messages[-1])
        for value in (
            "Captain Vero",
            "guard captain",
            "dry humor, speak formally",
            "hidden passage",
            "twenty years",
            "right hand twitches",
        ):
            assert value not in system_text
            assert value in user_text
        assert re.search(r"\bYou are\b[^\n]*Captain Vero", user_text)
        _assert_no_legacy_character_markers(messages)

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

        await agent.turn(guard_character, sample_checkpoint)

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_text = messages[0]["content"]
        user_text = messages[-1]["content"]
        assert "Time Since Your Last Turn" not in system_text
        assert "1 minute" in user_text

    @pytest.mark.asyncio
    async def test_turn_commit_updates_last_agent_turn_time(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        guard_character.clock_at_s = 20
        sample_checkpoint.session.leading_at_s = 35
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint)

        assert guard_character.last_agent_turn_at_s == 35

    @pytest.mark.asyncio
    async def test_dnd_ruleset_addon_stays_in_stable_system_prefix(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        sample_checkpoint.session.config.settings.ruleset_id = "dnd5e_basic"
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint)

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_text = _message_text(messages[0])
        user_text = _message_text(messages[-1])
        ruleset = prompt_manager.render("agent_ruleset_dnd5e").strip()
        assert ruleset in system_text
        assert ruleset not in user_text
        for value in (
            "Captain Vero",
            "twenty years",
            "hidden passage",
        ):
            assert value not in system_text
            assert value in user_text
        assert "Active D&D 5e initiative is running" not in system_text
        _assert_no_legacy_character_markers(messages)

    @pytest.mark.asyncio
    async def test_dnd_player_identity_species_is_live_user_context(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        sample_checkpoint.session.config.settings.ruleset_id = "dnd5e_basic"
        lyra = CharacterRecord(
            character_id="lyra",
            name="Lyra",
            public_sheet=PublicSheet(role="cleric"),
            mechanics={
                "ruleset_id": "dnd5e_basic",
                "dnd5e_sheet": {
                    "identity": {
                        "species": "Hill Dwarf",
                        "classes": [{"name": "Cleric", "level": 3}],
                    },
                    "statblock": {},
                },
            },
        )
        sample_checkpoint.characters = [guard_character, lyra]
        sample_checkpoint.session.character_bindings = {"lyra": "discord_1"}
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint)

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_text = messages[0]["content"]
        user_text = messages[-1]["content"]
        assert "D&D Player Character Identities" not in system_text
        assert "Hill Dwarf" not in system_text
        assert "Lyra: Hill Dwarf; Cleric 3" in user_text

    @pytest.mark.asyncio
    async def test_dnd_player_identity_absent_outside_dnd_ruleset(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        lyra = CharacterRecord(
            character_id="lyra",
            name="Lyra",
            public_sheet=PublicSheet(role="cleric"),
            mechanics={
                "ruleset_id": "dnd5e_basic",
                "dnd5e_sheet": {
                    "identity": {"species": "Hill Dwarf"},
                    "statblock": {},
                },
            },
        )
        sample_checkpoint.characters = [guard_character, lyra]
        sample_checkpoint.session.character_bindings = {"lyra": "discord_1"}
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint)

        messages = mock_client.complete.call_args.kwargs["messages"]
        flat = "\n".join(
            message["content"] for message in messages
            if isinstance(message.get("content"), str)
        )
        assert "D&D Player Character Identities" not in flat
        assert "Hill Dwarf" not in flat

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

        await agent.turn(guard_character, sample_checkpoint)

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
        assert "## Tactical Map" in persisted_user
        assert "Gatehouse" in persisted_user

    @pytest.mark.asyncio
    async def test_foreground_local_context_is_preserved_in_saved_history(
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
        assert "choose one listed action" in live_user
        messages = mock_client.complete.call_args.kwargs["messages"]
        assert "choose one listed action" not in messages[0]["content"]
        assert "Captain Vero" in live_user
        assert "Captain Vero" not in messages[0]["content"]

        saved_user = sample_checkpoint.character_conversations[
            "guard_17"
        ][0].content
        assert "choose one listed action" in saved_user
        assert "Captain Vero" in saved_user
        _assert_no_legacy_character_markers(messages)

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

        await agent.turn(guard_character, sample_checkpoint)

        call_args = mock_client.complete.call_args
        user_msg = call_args.kwargs["messages"][-1]["content"]
        user_text = user_msg if isinstance(user_msg, str) else user_msg[0]["text"]
        system_text = call_args.kwargs["messages"][0]["content"]
        assert "picks up a rock" in user_text
        assert "throws the rock" in user_text
        assert "picks up a rock" not in system_text
        assert "throws the rock" not in system_text
        _assert_no_legacy_character_markers(call_args.kwargs["messages"])

    @pytest.mark.asyncio
    async def test_uses_agent_role(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint)

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["role"] == "agent"
        # Character turns are free-form prose with optional terminal markers,
        # not structured JSON; provider compaction is deliberately disabled.
        assert "response_model" not in call_args.kwargs
        assert call_args.kwargs["compact"] is False

    @pytest.mark.asyncio
    async def test_standard_agent_uses_luna_role(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        guard_character.agent_tier = CharacterAgentTier.standard
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint)

        assert mock_client.complete.call_args.kwargs["role"] == "agent_standard"

    @pytest.mark.asyncio
    async def test_utility_agent_uses_convenience_role(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        guard_character.agent_tier = CharacterAgentTier.utility
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint)

        assert mock_client.complete.call_args.kwargs["role"] == "agent_convenience"

    @pytest.mark.asyncio
    async def test_appends_to_rolling_conversation(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        assert sample_checkpoint.character_conversations == {}

        await agent.turn(guard_character, sample_checkpoint)

        convo = sample_checkpoint.character_conversations["guard_17"]
        assert len(convo) == 2
        assert convo[0].role == "user"
        assert convo[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_sanitizes_text_block_while_preserving_provider_blocks(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        response = _llm_response(sample_agent_text)
        response.assistant_content = [
            {"type": "text", "text": sample_agent_text},
            {"type": "compaction", "id": "provider-state"},
        ]
        mock_client.complete.return_value = response
        agent = CharacterAgent(mock_client, prompt_manager)

        result = await agent.turn(guard_character, sample_checkpoint)

        saved = sample_checkpoint.character_conversations["guard_17"][1]
        assert saved.content == [
            {
                "type": "text",
                "text": result.public_text,
            },
            {"type": "compaction", "id": "provider-state"},
        ]

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

        await agent.turn(guard_character, sample_checkpoint)

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

        await agent.turn(guard_character, sample_checkpoint)

        # Expect: system, prior user, prior assistant, current user
        messages = mock_client.complete.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "prior user content"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"

    @pytest.mark.asyncio
    async def test_historical_user_projection_keeps_turn_evidence_without_dossiers(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        first = (
            'He looks toward the fountain. "I heard that crack."'
        )
        second = 'He takes one step back. "Not again."'
        mock_client.complete.side_effect = [
            _llm_response(first),
            _llm_response(second),
        ]
        agent = CharacterAgent(mock_client, prompt_manager)

        guard_character.pending_observations = ["The fountain cracks again."]
        await agent.turn(guard_character, sample_checkpoint)
        first_messages = mock_client.complete.await_args_list[0].kwargs[
            "messages"
        ]
        first_user = first_messages[-1]["content"]
        guard_character.pending_observations = ["Rain starts falling."]
        await agent.turn(guard_character, sample_checkpoint)

        messages = mock_client.complete.await_args_list[1].kwargs["messages"]
        system_text = _message_text(messages[0])
        historical_user = _message_text(messages[1])
        current_user = _message_text(messages[-1])
        for value in (
            "Captain Vero",
            "twenty years",
            "right hand twitches",
            "hidden passage",
        ):
            assert value not in system_text
            assert value in first_user
            assert value in historical_user
            assert value in current_user
        assert "The fountain cracks again." in first_user
        assert "The fountain cracks again." in historical_user
        assert "Rain starts falling." in current_user
        assert historical_user == first_user
        assert "<private_state>" not in system_text + "\n" + current_user
        assert "<your_life>" not in system_text + "\n" + current_user
        historical_user = sample_checkpoint.character_conversations[
            "guard_17"
        ][0].content
        assert historical_user == first_user
        _assert_no_legacy_character_markers(messages)
        assert "## Current System Account State" not in historical_user


class TestCharacterAgentTurnParser:
    def test_plain_prose_is_public_and_has_no_private_surface(self):
        public, is_silence, presentation = _parse_agent_turn_response(
            'He says, "The gate is open."'
        )
        assert public == 'He says, "The gate is open."'
        assert is_silence is False
        assert presentation == CharacterPresentationChoice()

    def test_parenthetical_remains_observable_prose(self):
        public, is_silence, _presentation = _parse_agent_turn_response(
            'He says, "The gate is open." (He keeps one hand on the latch.)'
        )
        assert public == (
            'He says, "The gate is open." (He keeps one hand on the latch.)'
        )
        assert is_silence is False

    def test_public_sanitizer_removes_presentation(self):
        text = (
            'He says, "The gate is open."\n'
            '<presentation>{"use":"stern","request":""}</presentation>'
        )
        assert sanitize_character_public_text(text) == (
            'He says, "The gate is open."'
        )

    @pytest.mark.parametrize(
        "text",
        [
            "<private_carry>secret</private_carry> after",
            "before <private_carry>secret</private_carry> after",
            "before <private_carry>secret",
            "before </private_carry>",
            "before <PRIVATE_CARRY>secret</PRIVATE_CARRY>",
            "before <private_carry></private_carry>",
            (
                'before\n'
                '<presentation>{"use":"stern","request":""}</presentation>\n'
                '<private_carry>secret</private_carry>'
            ),
        ],
    )
    def test_retired_private_marker_is_rejected(self, text):
        with pytest.raises(CharacterAgentOutputError, match="retired"):
            _parse_agent_turn_response(text)

    def test_exact_silence_commits_but_empty_is_rejected(self):
        silent = _parse_agent_turn_response("<silence/>")
        assert silent[0] == ""
        assert silent[1] is True
        assert silent[2] == CharacterPresentationChoice()
        with pytest.raises(CharacterAgentOutputError, match="observable"):
            _parse_agent_turn_response("")

    @pytest.mark.asyncio
    async def test_empty_turn_does_not_consume_inbox_or_actor_time(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        guard_character.pending_observations = ["A key turns in the outer door."]
        mock_client.complete.return_value = _llm_response("")

        with pytest.raises(CharacterAgentOutputError, match="observable"):
            await CharacterAgent(mock_client, prompt_manager).turn(
                guard_character,
                sample_checkpoint,
            )

        assert guard_character.pending_observations == [
            "A key turns in the outer door."
        ]
        assert guard_character.last_agent_turn_at_s is None
        assert guard_character.character_id not in (
            sample_checkpoint.character_conversations
        )

    def test_silence_with_presentation_is_stored_without_footer(self):
        public, is_silence, presentation = _parse_agent_turn_response(
            '<silence/>\n<presentation>{"use":"quiet","request":""}</presentation>'
        )
        assert public == ""
        assert is_silence is True
        assert presentation.use == "quiet"

    @pytest.mark.parametrize(
        "text",
        [
            "<silence />",
            "<silence/> then speaks",
            "He speaks <silence/>",
            "<SILENCE/>",
        ],
    )
    def test_silence_marker_must_be_exact(self, text):
        with pytest.raises(CharacterAgentOutputError, match="exact"):
            _parse_agent_turn_response(text)


class TestCharacterAgentTurnHistory:
    """Character turn frames share one turn prompt and one actor history.

    Every user turn is a complete actor/current-input packet.  It stays in the
    rolling conversation verbatim (apart from the disposable presentation
    catalog), and provider-side context compaction is intentionally disabled.
    """

    @pytest.mark.asyncio
    async def test_full_actor_packet_repeats_without_compaction_or_self_stripping(
        self,
        mock_client,
        prompt_manager,
        guard_character,
        sample_checkpoint,
        sample_agent_text,
    ):
        actor_fact = "You suspect the footman accepted a bribe before dusk."
        guard_character.actor.facts.append(ActorFact(text=actor_fact))
        mock_client.complete.side_effect = [
            _llm_response(sample_agent_text),
            _llm_response(sample_agent_text),
        ]
        agent = CharacterAgent(mock_client, prompt_manager)

        guard_character.pending_observations = ["First unique observation."]
        await agent.turn(guard_character, sample_checkpoint)
        first_messages = mock_client.complete.await_args_list[0].kwargs[
            "messages"
        ]
        first_system = _message_text(first_messages[0])
        first_user = _message_text(first_messages[-1])
        assert actor_fact not in first_system
        assert actor_fact in first_user
        assert "Captain Vero" not in first_system
        assert "Captain Vero" in first_user
        assert "First unique observation." not in first_system
        assert "First unique observation." in first_user
        assert re.search(r"\bYou are\b[^\n]*Captain Vero", first_user)
        assert mock_client.complete.await_args_list[0].kwargs["compact"] is False
        _assert_no_legacy_character_markers(first_messages)

        guard_character.pending_observations = ["Second unique observation."]
        await agent.turn(guard_character, sample_checkpoint)
        second_messages = mock_client.complete.await_args_list[1].kwargs[
            "messages"
        ]
        second_system = _message_text(second_messages[0])
        historical_user = _message_text(second_messages[1])
        second_user = _message_text(second_messages[-1])
        assert actor_fact not in second_system
        for user_text in (historical_user, second_user):
            assert actor_fact in user_text
            assert "Captain Vero" in user_text
        assert "First unique observation." in historical_user
        assert "Second unique observation." in second_user
        assert historical_user == first_user
        assert mock_client.complete.await_args_list[1].kwargs["compact"] is False
        _assert_no_legacy_character_markers(second_messages)

    @pytest.mark.asyncio
    async def test_foreground_and_background_share_same_system_prefix(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        # Run a turn, capture the stable generic contract.
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.turn(guard_character, sample_checkpoint)
        turn_messages = mock_client.complete.call_args.kwargs["messages"]
        turn_system = turn_messages[0]
        assert turn_system["role"] == "system"

        # A background frame uses the same stable turn contract; only the
        # current user packet differs.
        mock_client.complete.reset_mock()
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        await agent.turn(
            guard_character,
            sample_checkpoint,
            frame="background",
            local_context="Background-only location cue.",
        )
        background_messages = mock_client.complete.call_args.kwargs["messages"]
        background_system = background_messages[0]
        assert background_system["role"] == "system"

        # The actor packet, frame-specific cue, and legacy-marker retirement
        # all belong to the user-facing boundary.
        assert background_system["content"] == turn_system["content"]
        assert "Captain Vero" not in turn_system["content"]
        assert "Captain Vero" not in background_system["content"]
        assert "Background-only location cue." in background_messages[-1]["content"]
        assert "Background-only location cue." not in background_system["content"]
        _assert_no_legacy_character_markers(turn_messages)
        _assert_no_legacy_character_markers(background_messages)

    @pytest.mark.asyncio
    async def test_different_characters_keep_stable_system_and_full_user_identity(
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
        other.actor = ActorRecord(
            may_act_offstage=True,
            facts=[
                ActorFact(
                    text=(
                        "You were raised in the archive rooms and trusted by "
                        "no one."
                    )
                ),
                ActorFact(
                    text=(
                        "You control the household intelligence network and "
                        "keep a second ledger in the chapel wall."
                    )
                ),
                ActorFact(
                    text="You speak softly, precisely, and without hurry."
                ),
            ],
        )

        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.turn(guard_character, sample_checkpoint)
        guard_messages = mock_client.complete.call_args.kwargs["messages"]

        mock_client.complete.reset_mock()
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        await agent.turn(other, sample_checkpoint)
        other_messages = mock_client.complete.call_args.kwargs["messages"]

        guard_system = guard_messages[0]["content"]
        other_system = other_messages[0]["content"]
        guard_user = guard_messages[-1]["content"]
        other_user = other_messages[-1]["content"]
        assert guard_system == other_system
        assert "Captain Vero" not in guard_system
        assert "Mistress Vale" not in other_system
        assert "Captain Vero" in guard_user
        assert "Mistress Vale" in other_user
        assert "You control the household intelligence network" not in guard_user
        assert "You know about the hidden passage." not in other_user
        assert re.search(r"\bYou are\b[^\n]*Captain Vero", guard_user)
        assert re.search(r"\bYou are\b[^\n]*Mistress Vale", other_user)
        _assert_no_legacy_character_markers(guard_messages)
        _assert_no_legacy_character_markers(other_messages)

    @pytest.mark.asyncio
    async def test_turn_user_message_contains_full_packet_without_legacy_labels(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        guard_character.pending_observations = ["A unique current observation."]
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.turn(guard_character, sample_checkpoint)
        messages = mock_client.complete.call_args.kwargs["messages"]
        user_content = messages[-1]["content"]
        system_content = messages[0]["content"]
        assert "Captain Vero" not in system_content
        assert "Captain Vero" in user_content
        assert "A unique current observation." not in system_content
        assert "A unique current observation." in user_content
        assert re.search(r"\bYou are\b[^\n]*Captain Vero", user_content)
        assert "## Scene" not in user_content
        assert "## What You Observe This Turn" not in user_content
        assert "## Other Characters' Responses This Turn" not in user_content
        _assert_no_legacy_character_markers(messages)

    @pytest.mark.asyncio
    async def test_background_user_message_keeps_frame_data_out_of_system(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.turn(
            guard_character,
            sample_checkpoint,
            frame="background",
            local_context="Background-only location cue.",
        )
        messages = mock_client.complete.call_args.kwargs["messages"]
        user_content = messages[-1]["content"]
        system_content = messages[0]["content"]
        assert "Background-only location cue." in user_content
        assert "Background-only location cue." not in system_content
        assert "Captain Vero" in user_content
        assert "Captain Vero" not in system_content
        _assert_no_legacy_character_markers(messages)

    @pytest.mark.asyncio
    async def test_background_local_context_is_preserved_in_saved_history(
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

        messages = mock_client.complete.call_args.kwargs["messages"]
        live_user = mock_client.complete.call_args.kwargs["messages"][-1]["content"]
        assert "Steward Lysa" in live_user

        saved_user = sample_checkpoint.character_conversations[
            "guard_17"
        ][0].content
        assert "Steward Lysa" in saved_user
        assert "Captain Vero" in saved_user
        _assert_no_legacy_character_markers(messages)

    @pytest.mark.asyncio
    async def test_background_turn_appends_to_same_rolling_conversation_as_foreground(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        # Single rolling history per character — foreground and background
        # both write into `character_conversations[character_id]`.
        # If they ever split (separate history per mode), the
        # agent's interior memory desyncs across modes.
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)

        await agent.turn(guard_character, sample_checkpoint)
        mock_client.complete.return_value = _llm_response(
            "He stands by the window."
        )
        await agent.turn(guard_character, sample_checkpoint, frame="background")

        # Exactly ONE rolling-history key for this character; both
        # the foreground pair AND the background pair are appended to it.
        keys = list(sample_checkpoint.character_conversations.keys())
        assert keys == ["guard_17"]
        convo = sample_checkpoint.character_conversations["guard_17"]
        # 2 user/assistant pairs = 4 messages total.
        assert len(convo) == 4
        # Sequence: first user, first assistant, background user,
        # background assistant.
        assert convo[0].role == "user"
        assert "Captain Vero" in convo[0].content
        assert convo[2].role == "user"
        assert "Captain Vero" in convo[2].content
        assert convo[0].content != convo[2].content


class TestPerceptionMode:
    """Observer-agnostic visual loadout with a public-only user packet.

    Fired by the observation-harvest fork in run_beat (and reachable
    later from /query for "what does X look like?" questions). Three
    load-bearing properties distinguish perception from normal agent turns:

      1. The user message carries only the public identity and visible
         presentation input; actor facts and turn observations stay out.
      2. Perception calls append to the same rolling conversation.
         A character should remember what they established about their
         visual presentation in the scene.
      3. Perception has a separate system contract because it has a
         different output job from a normal agent turn.
    """

    def _llm_text_only(self, text: str) -> LLMResponse:
        # Perception output is plain exterior prose.
        raw = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        text_block.model_dump = lambda: {"type": "text", "text": text}
        raw.content = [text_block]
        raw.model = "gpt-5.6-luna"
        return LLMResponse(
            parsed=None, raw_response=raw, content=text,
            model="gpt-5.6-luna",
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
        assert result.public_text == loadout

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response, error_match",
        (
            ("Visible coat. <private_carry>hidden</private_carry>", "retired"),
            ("<silence/>", "perception"),
        ),
    )
    async def test_perception_rejects_turn_controls_before_history_mutation(
        self,
        response,
        error_match,
        mock_client,
        prompt_manager,
        guard_character,
        sample_checkpoint,
    ):
        mock_client.complete.return_value = self._llm_text_only(response)

        with pytest.raises(CharacterAgentOutputError, match=error_match):
            await CharacterAgent(mock_client, prompt_manager).perceive(
                guard_character,
                sample_checkpoint,
            )

        assert guard_character.character_id not in (
            sample_checkpoint.character_conversations
        )

    @pytest.mark.asyncio
    async def test_perception_commits_private_visual_novel_choice(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        sample_checkpoint.session.config.settings.presentation_mode = (
            "visual_novel"
        )
        response = (
            "His shoulders ease, and a small smile reaches his eyes.\n"
            '<presentation>{"use":"happy","request":""}</presentation>'
        )
        mock_client.complete.return_value = self._llm_text_only(response)

        result = await CharacterAgent(mock_client, prompt_manager).perceive(
            guard_character,
            sample_checkpoint,
        )

        assert result.public_text == (
            "His shoulders ease, and a small smile reaches his eyes."
        )
        assert result.presentation.use == "happy"
        assert (
            guard_character.visuals.visual_novel_presentation.current_variant_key
            == "happy"
        )
        live_messages = mock_client.complete.await_args.kwargs["messages"]
        live_user = live_messages[-1]["content"]
        catalog_start = live_user.index("<presentation_catalog")
        catalog_end = live_user.index("</presentation_catalog>") + len(
            "</presentation_catalog>"
        )
        catalog = live_user[catalog_start:catalog_end]
        saved = sample_checkpoint.character_conversations["guard_17"]
        assert catalog not in saved[0].content
        assert saved[0].content == live_user.replace(catalog, "", 1)
        assert "Captain Vero" in saved[0].content
        assert "guard captain" in saved[0].content
        assert "hidden passage" not in saved[0].content
        _assert_no_legacy_character_markers(live_messages)
        assert "presentation" not in saved[1].content[0]["text"]

    @pytest.mark.asyncio
    async def test_perceive_user_message_contains_only_public_surface(
        self, mock_client, prompt_manager, guard_character, sample_checkpoint,
    ):
        guard_character.pending_observations = [
            "A private incoming observation that perception must not receive."
        ]
        mock_client.complete.return_value = self._llm_text_only("loadout")
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.perceive(guard_character, sample_checkpoint)
        messages = mock_client.complete.call_args.kwargs["messages"]
        system_content = _message_text(messages[0])
        user_content = _message_text(messages[-1])
        public_values = (
            "Captain Vero",
            "guard captain",
            "Tall, scarred, in polished armor",
            "City Watch",
        )
        private_values = (
            "twenty years",
            "dry humor, speak formally",
            "hidden passage",
            "A private incoming observation",
        )
        for value in public_values:
            assert value not in system_content
            assert value in user_content
        for value in private_values:
            assert value not in system_content
            assert value not in user_content
        assert re.search(r"\bYou are\b[^\n]*Captain Vero", user_content)
        _assert_no_legacy_character_markers(messages)

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
        assert "Captain Vero" in convo[0].content
        assert "guard captain" in convo[0].content
        assert "hidden passage" not in convo[0].content
        _assert_no_legacy_character_markers(
            [{"role": message.role, "content": message.content} for message in convo]
        )
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
        system_content = messages[0]["content"]
        assert "shout in the courtyard" not in user_content
        assert "Bells ring" not in user_content
        assert "shout in the courtyard" not in system_content
        assert "Bells ring" not in system_content
        _assert_no_legacy_character_markers(messages)

    @pytest.mark.asyncio
    async def test_perceive_shares_system_prefix_with_turn(
        self, mock_client, prompt_manager, guard_character,
        sample_checkpoint, sample_agent_text,
    ):
        # Turn and perception have separate stable contracts.  The D&D rules
        # addon belongs to the turn's stable system contract; perception stays
        # narrower and receives only its public surface.  Both calls explicitly
        # avoid provider compaction.
        sample_checkpoint.session.config.settings.ruleset_id = "dnd5e_basic"
        mock_client.complete.return_value = _llm_response(sample_agent_text)
        agent = CharacterAgent(mock_client, prompt_manager)
        await agent.turn(guard_character, sample_checkpoint)
        turn_system = mock_client.complete.call_args.kwargs["messages"][0]
        turn_user = mock_client.complete.call_args.kwargs["messages"][-1]

        mock_client.complete.reset_mock()
        mock_client.complete.return_value = self._llm_text_only("loadout")
        await agent.perceive(guard_character, sample_checkpoint)
        perceive_system = mock_client.complete.call_args.kwargs["messages"][0]
        perceive_user = mock_client.complete.call_args.kwargs["messages"][-1]

        assert perceive_system["role"] == "system"
        assert perceive_system["content"] != turn_system["content"]
        ruleset = prompt_manager.render("agent_ruleset_dnd5e").strip()
        assert ruleset in turn_system["content"]
        assert ruleset not in turn_user["content"]
        assert ruleset not in perceive_system["content"]
        assert ruleset not in perceive_user["content"]
        for system in (turn_system["content"], perceive_system["content"]):
            assert "Captain Vero" not in system
            assert "twenty years" not in system
        _assert_no_legacy_character_markers(
            [turn_system, turn_user, perceive_system, perceive_user]
        )
        assert mock_client.complete.await_args.kwargs["compact"] is False

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
    async def test_perceive_uses_lower_max_tokens_than_turn(
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
        assert result.public_text == "Polished armor."
