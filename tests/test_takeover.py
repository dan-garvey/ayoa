"""Tests for the takeover flows on EngineBridge.

Under playable-2 semantics:
- `takeover` (plain) just binds a Discord user to an existing character;
  it does NOT toggle `is_playable` (which is an authoring-time flag).
- `create_custom_character` and `replace_with_custom` still mark the
  resulting record `is_playable=True` since the bot path explicitly
  authored a player slot.
- `replace_with_custom` no longer rejects `is_playable=True` targets;
  the only rejection is "already bound to another user" (handled by
  the explicit binding check in the bridge).

These engine methods stay live for the play-CLI takeover path; the
Discord /join_custom + /pick_replacement commands that used to surface
them in chat were murdered as part of the UX overhaul.

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
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.state import SessionState, WorldState
from app.schemas.takeover import (
    ReplacementCandidate,
    TakeoverAuthoredOutput,
    TakeoverSuggestOutput,
)


SESSION_ID = "test_session"


def _llm_response(parsed) -> LLMResponse:
    """Shape an LLMResponse. Takeover paths parse JSON from response.content
    (structured output disabled — see benchmark), so content must contain
    the pydantic model's JSON."""
    text = parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else "{}"
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    text_block.model_dump = lambda: {"type": "text", "text": text}
    raw.content = [text_block]
    raw.model = "claude-sonnet-4-6"
    return LLMResponse(
        parsed=parsed,
        raw_response=raw,
        content=text,
        model="claude-sonnet-4-6",
    )


def _authored(**overrides):
    """Build an AuthoredCharacter with every required field satisfied. Tests
    override only the ones they care about; the rest default to empty."""
    from app.schemas.takeover import AuthoredCharacter
    defaults = dict(
        name="default", location="", role="", appearance="",
        default_loadout="", faction="",
        backstory="", personality="", known_context="",
        goals=[], current_objectives=[], secrets=[], intentions_enabled=False,
        router_summary="",
    )
    defaults.update(overrides)
    return AuthoredCharacter(**defaults)


def _make_checkpoint(characters: list[CharacterRecord] | None = None,
                     bindings: dict[str, str] | None = None) -> CheckpointFile:
    if characters is None:
        characters = [
            CharacterRecord(
                character_id="npc1",
                name="Guard Vero",
                status=CharacterStatus.active,
                is_playable=True,
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
        world_state=WorldState(),
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
    def test_takeover_binds_user_without_touching_is_playable(
        self, bridge: EngineBridge,
    ):
        """Under playable-2, takeover binds the user but leaves the
        authored `is_playable` flag alone — it's an authoring-time
        property, not a runtime toggle."""
        ckpt = _make_checkpoint()
        _seed(bridge, ckpt)

        result = bridge.takeover(SESSION_ID, "npc1", user_id=42)

        assert result.session.character_bindings == {"npc1": "42"}
        npc = next(c for c in result.characters if c.character_id == "npc1")
        # Was authored with is_playable=True — still True after takeover.
        assert npc.is_playable is True

        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        loaded_npc = next(
            c for c in loaded.characters if c.character_id == "npc1"
        )
        assert loaded_npc.is_playable is True
        assert loaded.session.character_bindings == {"npc1": "42"}

    def test_takeover_of_non_playable_succeeds_with_warning(
        self, bridge: EngineBridge, caplog,
    ):
        """If a user takes over a character the importer didn't mark
        playable, the binding still applies (explicit user intent wins)
        but the bridge logs a warning so the operator can fix the
        importer authoring."""
        ckpt = _make_checkpoint(
            characters=[
                CharacterRecord(
                    character_id="npc1",
                    name="Guard Vero",
                    status=CharacterStatus.active,
                    is_playable=False,
                    location="courtyard",
                    public_sheet=PublicSheet(role="guard"),
                ),
            ],
        )
        _seed(bridge, ckpt)

        import logging
        with caplog.at_level(logging.WARNING, logger="app.bot.engine_bridge"):
            result = bridge.takeover(SESSION_ID, "npc1", user_id=42)

        assert result.session.character_bindings == {"npc1": "42"}
        # is_playable stays False — flag tracks authorship, not binding.
        npc = next(c for c in result.characters if c.character_id == "npc1")
        assert npc.is_playable is False
        assert any(
            "not marked is_playable" in rec.message for rec in caplog.records
        )

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

        authored = _authored(
            name="Tessa",
            role="scout",
            default_loadout="Weathered green cloak and hill-road boots.",
            backstory="Trained in the hills.",
            personality="Wary, quick on her feet.",
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
        assert new_char.visuals.default_loadout == (
            "Weathered green cloak and hill-road boots."
        )
        # Bot-authored player slots are marked is_playable=True so the
        # roster reflects "this is a human-authored character" even
        # before binding is recorded.
        assert new_char.is_playable is True

        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        ids = {c.character_id for c in loaded.characters}
        assert "tessa" in ids
        assert loaded.session.character_bindings.get("tessa") == "7"

        tessa = next(c for c in loaded.characters if c.character_id == "tessa")
        assert tessa.is_playable is True
        assert tessa.private_state.goals == ["find the informant"]
        assert tessa.visuals.default_loadout.startswith("Weathered green cloak")

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

        authored = _authored(name="Tessa", role="scout")
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


# ---- create_player_character_simple (LLM-free /join custom-create) ------


class TestCreatePlayerCharacterSimple:
    """LLM-free spawn used by the /join "Create your own character"
    modal. Asserts the path: (1) authors a record from raw user input
    only, (2) binds the user, (3) leaves location empty so the router
    can place them via the (arrive) directive that fires next, and
    (4) does NOT call the LLM at any point."""

    def test_happy_path_spawns_binds_and_emits_state_change(
        self, bridge: EngineBridge,
    ):
        ckpt = _make_checkpoint()
        _seed(bridge, ckpt)
        bridge.client.complete = AsyncMock(side_effect=AssertionError(
            "create_player_character_simple must not call the LLM"
        ))

        new_char = bridge.create_player_character_simple(
            SESSION_ID, user_id=42,
            name="Akari Tanaka",
            appearance="short, dark hair, hoodie over a school uniform",
            backstory="A college student dragged here by a freak storm.",
        )

        assert new_char.name == "Akari Tanaka"
        assert new_char.character_id == "akari_tanaka"
        assert new_char.is_playable is True
        # Router places via (arrive) directive — leave the slot empty
        # rather than guessing.
        assert new_char.location == ""
        assert new_char.public_sheet.appearance.startswith("short, dark hair")
        assert new_char.visuals.default_loadout.startswith("short, dark hair")
        assert "freak storm" in new_char.backstory

        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        ids = {c.character_id for c in loaded.characters}
        assert "akari_tanaka" in ids
        assert loaded.session.character_bindings.get("akari_tanaka") == "42"

        # Router needs a heads-up about the new arrival.
        changes = loaded.session.pending_engine_state_updates
        arrival_change = next(
            line for line in changes
            if "akari_tanaka" in line and "[player-bound]" in line
        )
        assert "appearance:" in arrival_change
        assert "player-supplied backstory:" in arrival_change
        assert "sparse player-authored arrival" in arrival_change
        assert "observable_facts" in arrival_change

    def test_backstory_optional(self, bridge: EngineBridge):
        ckpt = _make_checkpoint()
        _seed(bridge, ckpt)
        bridge.client.complete = AsyncMock(side_effect=AssertionError(
            "must not call the LLM"
        ))

        new_char = bridge.create_player_character_simple(
            SESSION_ID, user_id=7,
            name="Mira",
            appearance="freckles, red braid, satchel of seed packets",
        )
        assert new_char.backstory == ""
        assert new_char.visuals.default_loadout == (
            "freckles, red braid, satchel of seed packets"
        )
        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        assert loaded.session.character_bindings.get("mira") == "7"

    def test_empty_name_rejected(self, bridge: EngineBridge):
        ckpt = _make_checkpoint()
        _seed(bridge, ckpt)
        before = bridge.checkpoint_mgr.load_latest(SESSION_ID).model_dump_json()

        with pytest.raises(ValueError, match="name"):
            bridge.create_player_character_simple(
                SESSION_ID, user_id=1,
                name="   ", appearance="someone",
            )
        # No partial state on validation failure.
        after = bridge.checkpoint_mgr.load_latest(SESSION_ID).model_dump_json()
        assert before == after

    def test_empty_appearance_rejected(self, bridge: EngineBridge):
        ckpt = _make_checkpoint()
        _seed(bridge, ckpt)
        before = bridge.checkpoint_mgr.load_latest(SESSION_ID).model_dump_json()

        with pytest.raises(ValueError, match="appearance"):
            bridge.create_player_character_simple(
                SESSION_ID, user_id=1,
                name="Akari", appearance="",
            )
        after = bridge.checkpoint_mgr.load_latest(SESSION_ID).model_dump_json()
        assert before == after

    def test_disambiguates_existing_id(self, bridge: EngineBridge):
        existing = CharacterRecord(
            character_id="akari", name="Akari (the original)",
            public_sheet=PublicSheet(role="local"),
        )
        ckpt = _make_checkpoint(characters=[existing])
        _seed(bridge, ckpt)

        new_char = bridge.create_player_character_simple(
            SESSION_ID, user_id=11,
            name="Akari", appearance="modern clothes, phone in hand",
        )
        assert new_char.character_id == "akari_2"

        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        ids = {c.character_id for c in loaded.characters}
        assert {"akari", "akari_2"}.issubset(ids)
        # Original character's binding is unchanged; new one bound.
        assert loaded.session.character_bindings.get("akari_2") == "11"
        assert "akari" not in loaded.session.character_bindings


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
            is_playable=False,
            location="garden",
            public_sheet=PublicSheet(role="old role", faction="old faction"),
            backstory="old backstory",
            personality="old personality",
            known_context="old context",
            private_state=PrivateState(
                goals=["old goal"],
                current_objectives=["win the duel"],
                secrets=["old secret"],
                intentions_enabled=False,
            ),
            pending_observations=["saw the flash"],
        )
        ckpt = _make_checkpoint(characters=[target])
        # Seed a prior character_conversations entry for this target; it
        # should be cleared by replace.
        ckpt.character_conversations["rival_1"] = [
            ConversationMessage(role="user", content="prior turn context"),
        ]
        ckpt.session.visual_introductions = {
            "alice": ["rival_1", "witness"],
            "rival_1": ["alice"],
            "bob": ["rival_1"],
        }
        _seed(bridge, ckpt)

        authored = _authored(
            name="Brooding Rival",
            location="somewhere_else",  # ignored — location preserved
            role="brooder",
            default_loadout="Black officer's coat and a hard, steady stare.",
            faction="loners",
            backstory="A new, grim history.",
            personality="Cold and calculating. Speaks in long silences.",
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
        assert updated.visuals.default_loadout == (
            "Black officer's coat and a hard, steady stare."
        )
        assert updated.backstory == "A new, grim history."
        assert updated.personality == "Cold and calculating. Speaks in long silences."
        assert updated.known_context == "Knows the new truth."
        assert updated.private_state.goals == ["new vendetta"]
        assert updated.private_state.secrets == ["hides a dagger"]
        assert updated.private_state.intentions_enabled is True

        # Circumstances preserved
        assert updated.location == "garden"
        assert updated.private_state.current_objectives == ["win the duel"]
        assert updated.pending_observations == ["saw the flash"]

        # Flag + binding
        assert updated.is_playable is True

        loaded = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        assert loaded.session.character_bindings.get("rival_1") == "5"
        assert loaded.session.visual_introductions == {
            "alice": ["witness"],
        }
        # Rolling conversation cleared
        assert "rival_1" not in loaded.character_conversations

        # Persisted target got the new identity
        reloaded_target = next(
            c for c in loaded.characters if c.character_id == "rival_1"
        )
        assert reloaded_target.name == "Brooding Rival"
        assert reloaded_target.location == "garden"
        assert reloaded_target.is_playable is True

    @pytest.mark.asyncio
    async def test_replace_allowed_for_unbound_playable(
        self, bridge: EngineBridge,
    ):
        """playable-2: an `is_playable=True` slot that ISN'T bound to a
        human is just an agent NPC and is fair game for replacement.
        Pre-rename code rejected this path because the old `is_player`
        field doubled as both authorship and binding."""
        playable_unbound = CharacterRecord(
            character_id="player_slot",
            name="Hero",
            is_playable=True,
            location="hub",
            public_sheet=PublicSheet(role="hero"),
        )
        ckpt = _make_checkpoint(characters=[playable_unbound])
        _seed(bridge, ckpt)

        authored = _authored(name="New Hero", role="rogue", location="hub")
        out = TakeoverAuthoredOutput(character=authored, session_note="")
        bridge.client.complete = AsyncMock(return_value=_llm_response(out))

        updated = await bridge.replace_with_custom(
            SESSION_ID, user_id=5,
            target_character_id="player_slot",
            description="a fresh take",
        )

        assert updated.character_id == "player_slot"
        assert updated.name == "New Hero"
        assert updated.is_playable is True

    @pytest.mark.asyncio
    async def test_rejects_claim_by_other_user(self, bridge: EngineBridge):
        npc = CharacterRecord(
            character_id="shared_npc",
            name="Shared",
            is_playable=False,
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
