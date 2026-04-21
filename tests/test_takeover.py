"""Tests for the takeover (/join_custom) flows on EngineBridge.

Covers three engine methods:
- `takeover` (plain — bind + flip is_player on an existing character)
- `create_custom_character` (mode='describe' — spawn a new authored char)
- `suggest_replacement_targets` (mode='suggest' — surface candidates)
- `replace_with_custom` (mode='replace' — graft authored identity onto NPC)

The LLM is mocked via `client.complete` so tests never hit the network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.engine_bridge import EngineBridge
from app.llm.client import LLMResponse
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    IncomingDirective,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.state import LocationState, SessionState, WorldState
from app.schemas.takeover import (
    ReplacementCandidate,
    TakeoverAuthoredOutput,
    TakeoverSuggestOutput,
)


SESSION_ID = "test_session"


def _llm_response(parsed) -> LLMResponse:
    """Shape an LLMResponse the way EngineBridge expects."""
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "{}"
    text_block.model_dump = lambda: {"type": "text", "text": "{}"}
    raw.content = [text_block]
    raw.model = "claude-sonnet-4-6"
    return LLMResponse(
        parsed=parsed,
        raw_response=raw,
        content="{}",
        model="claude-sonnet-4-6",
    )


def _make_checkpoint(characters: list[CharacterRecord] | None = None,
                     bindings: dict[str, str] | None = None) -> CheckpointFile:
    if characters is None:
        characters = [
            CharacterRecord(
                character_id="npc1",
                name="Guard Vero",
                status=CharacterStatus.active,
                is_player=False,
                location="courtyard",
                public_sheet=PublicSheet(role="guard"),
            ),
        ]
    return CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            turn_index=1,
            character_bindings=bindings or {},
        ),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="courtyard",
                scene_graph={
                    "courtyard": {
                        "name": "Courtyard",
                        "description": "A stone courtyard.",
                        "connected_to": [],
                    },
                },
            ),
        ),
        characters=characters,
    )


@pytest.fixture
def bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    """EngineBridge backed by a temp saves dir.

    EngineBridge builds an LLMClient during __init__ from env; we set a
    harmless API key so it initializes, then the tests replace
    `bridge.client.complete` with an AsyncMock before any async call.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return EngineBridge(saves_dir=str(tmp_path), prompts_dir="app/prompts")


def _seed(bridge: EngineBridge, ckpt: CheckpointFile) -> None:
    bridge.checkpoint_mgr.save(ckpt)


# ---- plain takeover -----------------------------------------------------


class TestTakeoverPlain:
    def test_takeover_binds_and_flips_is_player(self, bridge: EngineBridge):
        ckpt = _make_checkpoint()
        _seed(bridge, ckpt)

        result = bridge.takeover(SESSION_ID, "npc1", user_id=42)

        assert result.session.character_bindings == {"npc1": "42"}
        npc = next(c for c in result.characters if c.character_id == "npc1")
        assert npc.is_player is True

        # Persisted
        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        loaded_npc = next(c for c in loaded.characters if c.character_id == "npc1")
        assert loaded_npc.is_player is True
        assert loaded.session.character_bindings == {"npc1": "42"}

    def test_takeover_rejects_culled(self, bridge: EngineBridge):
        ckpt = _make_checkpoint(
            characters=[
                CharacterRecord(
                    character_id="dead_one",
                    name="Ghost",
                    status=CharacterStatus.culled,
                ),
            ],
        )
        _seed(bridge, ckpt)

        with pytest.raises(ValueError, match="culled"):
            bridge.takeover(SESSION_ID, "dead_one", user_id=42)

    def test_takeover_rejects_already_claimed_by_other(self, bridge: EngineBridge):
        ckpt = _make_checkpoint(bindings={"npc1": "99"})
        _seed(bridge, ckpt)

        with pytest.raises(ValueError, match="already bound"):
            bridge.takeover(SESSION_ID, "npc1", user_id=42)

    def test_takeover_rejects_missing_character(self, bridge: EngineBridge):
        ckpt = _make_checkpoint()
        _seed(bridge, ckpt)

        with pytest.raises(ValueError, match="No character"):
            bridge.takeover(SESSION_ID, "does_not_exist", user_id=42)


# ---- create_custom_character (mode='describe') --------------------------


class TestCreateCustomCharacter:
    @pytest.mark.asyncio
    async def test_spawns_and_binds(self, bridge: EngineBridge):
        ckpt = _make_checkpoint()
        _seed(bridge, ckpt)

        from app.schemas.takeover import AuthoredCharacter
        authored = AuthoredCharacter(
            name="Tessa",
            role="scout",
            traits=["quick"],
            backstory="Trained in the hills.",
            personality="Wary.",
            goals=["find the informant"],
            secrets=["carries a forged seal"],
            location="courtyard",
        )
        out = TakeoverAuthoredOutput(character=authored, session_note="")

        bridge.client.complete = AsyncMock(return_value=_llm_response(out))

        new_char = await bridge.create_custom_character(
            SESSION_ID, user_id=7, description="a scout with a bad past",
        )

        assert new_char.name == "Tessa"
        assert new_char.character_id == "tessa"
        assert new_char.is_player is True

        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        ids = {c.character_id for c in loaded.characters}
        assert "tessa" in ids
        assert loaded.session.character_bindings.get("tessa") == "7"

        tessa = next(c for c in loaded.characters if c.character_id == "tessa")
        assert tessa.is_player is True
        assert tessa.private_state.goals == ["find the informant"]

    @pytest.mark.asyncio
    async def test_disambiguates_existing_id(self, bridge: EngineBridge):
        # Roster already has "tessa".
        existing = CharacterRecord(
            character_id="tessa",
            name="Tessa (the original)",
            public_sheet=PublicSheet(role="local"),
        )
        ckpt = _make_checkpoint(characters=[existing])
        _seed(bridge, ckpt)

        from app.schemas.takeover import AuthoredCharacter
        authored = AuthoredCharacter(name="Tessa", role="scout")
        out = TakeoverAuthoredOutput(character=authored, session_note="")
        bridge.client.complete = AsyncMock(return_value=_llm_response(out))

        new_char = await bridge.create_custom_character(
            SESSION_ID, user_id=7, description="another Tessa",
        )

        assert new_char.character_id == "tessa_2"
        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        ids = {c.character_id for c in loaded.characters}
        assert "tessa" in ids
        assert "tessa_2" in ids
        assert loaded.session.character_bindings.get("tessa_2") == "7"


# ---- suggest_replacement_targets (mode='suggest') -----------------------


class TestSuggestReplacementTargets:
    @pytest.mark.asyncio
    async def test_returns_candidates_without_mutation(self, bridge: EngineBridge):
        ckpt = _make_checkpoint(
            characters=[
                CharacterRecord(
                    character_id="npc_a", name="A",
                    public_sheet=PublicSheet(role="merchant"),
                ),
                CharacterRecord(
                    character_id="npc_b", name="B",
                    public_sheet=PublicSheet(role="guard"),
                ),
                CharacterRecord(
                    character_id="npc_c", name="C",
                    public_sheet=PublicSheet(role="scribe"),
                ),
            ],
        )
        _seed(bridge, ckpt)

        # Snapshot to verify no mutation
        before_json = bridge.checkpoint_mgr.load_latest(SESSION_ID).model_dump_json()

        candidates = [
            ReplacementCandidate(
                character_id="npc_a", name="A",
                fit_rationale="A is pliable.",
            ),
            ReplacementCandidate(
                character_id="npc_c", name="C",
                fit_rationale="C is mysterious.",
            ),
        ]
        out = TakeoverSuggestOutput(candidates=candidates, preamble="note")
        bridge.client.complete = AsyncMock(return_value=_llm_response(out))

        result = await bridge.suggest_replacement_targets(
            SESSION_ID, "a trickster",
        )

        assert isinstance(result, dict)
        assert result["preamble"] == "note"
        assert len(result["candidates"]) == 2
        ids = [c["character_id"] for c in result["candidates"]]
        assert ids == ["npc_a", "npc_c"]
        assert all("fit_rationale" in c for c in result["candidates"])

        # No mutation
        after_json = bridge.checkpoint_mgr.load_latest(SESSION_ID).model_dump_json()
        assert before_json == after_json


# ---- replace_with_custom (mode='replace') -------------------------------


class TestReplaceWithCustom:
    @pytest.mark.asyncio
    async def test_preserves_circumstances_and_overwrites_identity(
        self, bridge: EngineBridge,
    ):
        target = CharacterRecord(
            character_id="rival_1",
            name="Original Rival",
            status=CharacterStatus.active,
            is_player=False,
            location="garden",
            public_sheet=PublicSheet(role="old role", faction="old faction"),
            backstory="old backstory",
            personality="old personality",
            narrative_notes="old notes",
            known_context="old context",
            private_state=PrivateState(
                goals=["old goal"],
                current_objectives=["win the duel"],
                secrets=["old secret"],
                intentions_enabled=False,
            ),
            pending_observations=["saw the flash"],
            incoming_directives=[
                IncomingDirective(
                    from_character_id="playerX",
                    content="Meet me at dawn.",
                    turn=3,
                ),
            ],
        )
        ckpt = _make_checkpoint(characters=[target])
        # Seed a prior character_conversations entry for this target; it
        # should be cleared by replace.
        ckpt.character_conversations["rival_1"] = [
            ConversationMessage(role="user", content="prior turn context"),
        ]
        _seed(bridge, ckpt)

        from app.schemas.takeover import AuthoredCharacter
        authored = AuthoredCharacter(
            name="Brooding Rival",
            location="somewhere_else",  # ignored — location preserved
            role="brooder",
            faction="loners",
            traits=["quiet"],
            backstory="A new, grim history.",
            personality="Cold and calculating.",
            narrative_notes="Portray with long silences.",
            known_context="Knows the new truth.",
            goals=["new vendetta"],
            secrets=["hides a dagger"],
            intentions_enabled=True,
            # Ignored — the authored value for current_objectives
            # shouldn't overwrite the target's.
            current_objectives=["unused"],
        )
        out = TakeoverAuthoredOutput(
            character=authored, session_note="The rival has changed.",
        )
        bridge.client.complete = AsyncMock(return_value=_llm_response(out))

        updated = await bridge.replace_with_custom(
            SESSION_ID, user_id=5,
            target_character_id="rival_1",
            description="a brooding rival",
        )

        # character_id preserved
        assert updated.character_id == "rival_1"

        # Identity overwritten
        assert updated.name == "Brooding Rival"
        assert updated.public_sheet.role == "brooder"
        assert updated.public_sheet.faction == "loners"
        assert updated.backstory == "A new, grim history."
        assert updated.personality == "Cold and calculating."
        assert updated.narrative_notes == "Portray with long silences."
        assert updated.known_context == "Knows the new truth."
        assert updated.private_state.goals == ["new vendetta"]
        assert updated.private_state.secrets == ["hides a dagger"]
        assert updated.private_state.intentions_enabled is True

        # Circumstances preserved
        assert updated.location == "garden"
        assert updated.private_state.current_objectives == ["win the duel"]
        assert updated.pending_observations == ["saw the flash"]
        assert len(updated.incoming_directives) == 1
        assert updated.incoming_directives[0].content == "Meet me at dawn."

        # Flag + binding
        assert updated.is_player is True

        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        assert loaded.session.character_bindings.get("rival_1") == "5"
        # Rolling conversation cleared
        assert "rival_1" not in loaded.character_conversations

        # Persisted target got the new identity
        reloaded_target = next(
            c for c in loaded.characters if c.character_id == "rival_1"
        )
        assert reloaded_target.name == "Brooding Rival"
        assert reloaded_target.location == "garden"
        assert reloaded_target.is_player is True

    @pytest.mark.asyncio
    async def test_rejects_player_target(self, bridge: EngineBridge):
        player_char = CharacterRecord(
            character_id="player_slot",
            name="Hero",
            is_player=True,
            public_sheet=PublicSheet(role="hero"),
        )
        ckpt = _make_checkpoint(characters=[player_char])
        _seed(bridge, ckpt)

        # client.complete should never be called
        bridge.client.complete = AsyncMock(side_effect=AssertionError(
            "LLM should not be called for a rejected target",
        ))

        with pytest.raises(ValueError, match="already a player"):
            await bridge.replace_with_custom(
                SESSION_ID, user_id=5,
                target_character_id="player_slot",
                description="a new voice",
            )

    @pytest.mark.asyncio
    async def test_rejects_claim_by_other_user(self, bridge: EngineBridge):
        npc = CharacterRecord(
            character_id="shared_npc",
            name="Shared",
            is_player=False,
            public_sheet=PublicSheet(role="mystery"),
        )
        ckpt = _make_checkpoint(
            characters=[npc],
            bindings={"shared_npc": "99"},
        )
        _seed(bridge, ckpt)

        bridge.client.complete = AsyncMock(side_effect=AssertionError(
            "LLM should not be called for a rejected target",
        ))

        with pytest.raises(ValueError, match="already bound"):
            await bridge.replace_with_custom(
                SESSION_ID, user_id=5,
                target_character_id="shared_npc",
                description="a new voice",
            )

    @pytest.mark.asyncio
    async def test_rejects_missing_target(self, bridge: EngineBridge):
        ckpt = _make_checkpoint()
        _seed(bridge, ckpt)

        bridge.client.complete = AsyncMock(side_effect=AssertionError(
            "LLM should not be called for a missing target",
        ))

        with pytest.raises(ValueError, match="No character"):
            await bridge.replace_with_custom(
                SESSION_ID, user_id=5,
                target_character_id="does_not_exist",
                description="a new voice",
            )
