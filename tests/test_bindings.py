"""Tests for multiplayer bindings — EngineBridge bind/unbind/dossier/roster."""

from __future__ import annotations

import pytest
from pathlib import Path

from app.bot.engine_bridge import (
    CharacterSummary,
    EngineBridge,
    _summaries_from_checkpoint,
)
from app.engine.context_builder import (
    build_player_characters_block,
    collect_player_ids,
)
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import SessionState, WorldState


SESSION_ID = "test_session"


def _make_checkpoint() -> CheckpointFile:
    """Three characters: one player slot (Aldric), one claimable NPC (Sera),
    one dormant NPC (Thane), one culled NPC (Vex)."""
    return CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            player_name="Aldric",
            player_character_id="aldric",
        ),
        world_state=WorldState(
            hidden_lore="The throne is cursed.",
            hidden_facts=["Thane poisoned the last king."],
        ),
        characters=[
            CharacterRecord(
                character_id="aldric",
                name="Aldric",
                is_player=True,
                public_sheet=PublicSheet(role="wanderer", appearance="tall"),
                backstory="Raised by wolves.",
                personality="Quiet.",
                private_state=PrivateState(
                    goals=["survive"],
                    secrets=["knows the royal sigil"],
                ),
            ),
            CharacterRecord(
                character_id="sera",
                name="Sera Vance",
                public_sheet=PublicSheet(role="thief", appearance="wiry"),
                backstory="Grew up on the docks.",
                private_state=PrivateState(secrets=["owes the guild"]),
            ),
            CharacterRecord(
                character_id="thane",
                name="Thane",
                status=CharacterStatus.dormant,
                public_sheet=PublicSheet(role="assassin"),
            ),
            CharacterRecord(
                character_id="vex",
                name="Vex",
                status=CharacterStatus.culled,
                public_sheet=PublicSheet(role="former rival"),
            ),
        ],
    )


@pytest.fixture
def bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    """EngineBridge with a temp saves dir and a pre-seeded checkpoint."""
    # EngineBridge builds an LLMClient from env; avoid that by stubbing the
    # env-derived config to point at a harmless model.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    b = EngineBridge(saves_dir=str(tmp_path), prompts_dir="app/prompts")
    ckpt = _make_checkpoint()
    ckpt.session.turn_index = 1
    b.checkpoint_mgr.save(ckpt)
    return b


class TestSummaries:
    def test_surfaces_only_public_fields(self):
        summaries = _summaries_from_checkpoint(_make_checkpoint())
        by_id = {s.character_id: s for s in summaries}
        # All roster entries surfaced.
        assert set(by_id) == {"aldric", "sera", "thane", "vex"}
        # Public fields only — no secrets field on CharacterSummary.
        assert not hasattr(by_id["sera"], "secrets")
        assert not hasattr(by_id["sera"], "backstory")
        # Dormant/culled status comes through.
        assert by_id["thane"].status == "dormant"
        assert by_id["vex"].status == "culled"
        # Player-slot flag comes through.
        assert by_id["aldric"].is_player is True
        assert by_id["sera"].is_player is False


class TestBindUnbind:
    def test_bind_user_happy_path(self, bridge: EngineBridge):
        bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        assert bridge.get_user_binding(SESSION_ID, 42) == "sera"

    def test_bind_allows_dormant(self, bridge: EngineBridge):
        bridge.bind_user(SESSION_ID, user_id=42, character_id="thane")
        assert bridge.get_user_binding(SESSION_ID, 42) == "thane"

    def test_bind_rejects_culled(self, bridge: EngineBridge):
        with pytest.raises(ValueError, match="culled"):
            bridge.bind_user(SESSION_ID, user_id=42, character_id="vex")

    def test_bind_rejects_unknown_character(self, bridge: EngineBridge):
        with pytest.raises(ValueError, match="No character"):
            bridge.bind_user(SESSION_ID, user_id=42, character_id="ghost")

    def test_bind_rejects_second_user_on_same_char(self, bridge: EngineBridge):
        bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        with pytest.raises(ValueError, match="already bound"):
            bridge.bind_user(SESSION_ID, user_id=99, character_id="sera")

    def test_bind_rejects_user_double_binding(self, bridge: EngineBridge):
        bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        with pytest.raises(ValueError, match="already bound"):
            bridge.bind_user(SESSION_ID, user_id=42, character_id="thane")

    def test_bind_idempotent_for_same_pair(self, bridge: EngineBridge):
        # Binding the same user to the same character is a no-op (not an error).
        bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        assert bridge.get_user_binding(SESSION_ID, 42) == "sera"

    def test_unbind_frees_character(self, bridge: EngineBridge):
        bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        freed = bridge.unbind_user(SESSION_ID, 42)
        assert freed == "sera"
        assert bridge.get_user_binding(SESSION_ID, 42) is None
        # Another user can now claim sera.
        bridge.bind_user(SESSION_ID, user_id=99, character_id="sera")
        assert bridge.get_user_binding(SESSION_ID, 99) == "sera"

    def test_unbind_no_binding_returns_none(self, bridge: EngineBridge):
        assert bridge.unbind_user(SESSION_ID, 999) is None


class TestDossier:
    def test_includes_character_interior(self, bridge: EngineBridge):
        """Dossier surfaces what the CHARACTER knows about themselves."""
        dossier = bridge.build_character_dossier(SESSION_ID, "aldric")
        assert "Raised by wolves" in dossier  # backstory
        assert "survive" in dossier           # goal
        assert "royal sigil" in dossier       # secret THIS character keeps

    def test_excludes_world_hidden_content(self, bridge: EngineBridge):
        """World-wide hidden lore/facts are engine secrets, not per-character
        knowledge. Including them spoils the plot for players whose
        characters wouldn't actually know them."""
        dossier = bridge.build_character_dossier(SESSION_ID, "aldric")
        assert "throne is cursed" not in dossier
        assert "Thane poisoned" not in dossier

    def test_unknown_character_raises(self, bridge: EngineBridge):
        with pytest.raises(ValueError, match="No character"):
            bridge.build_character_dossier(SESSION_ID, "ghost")


class TestCollectPlayerIds:
    def test_unions_bindings_creator_and_is_player(self):
        ckpt = _make_checkpoint()
        ckpt.session.character_bindings = {"sera": "42"}
        ids = collect_player_ids(ckpt)
        # aldric (is_player + creator binding) + sera (explicit binding)
        assert ids == {"aldric", "sera"}

    def test_empty_bindings_still_includes_is_player(self):
        ckpt = _make_checkpoint()
        ids = collect_player_ids(ckpt)
        assert "aldric" in ids


class TestPlayerCharactersBlock:
    def test_marks_acting_character(self):
        ckpt = _make_checkpoint()
        ckpt.session.character_bindings = {"aldric": "1", "sera": "2"}
        block = build_player_characters_block(ckpt, "aldric")
        assert "acting this turn" in block
        # Acting marker on Aldric, not Sera.
        aldric_line = next(l for l in block.splitlines() if "Aldric" in l)
        sera_line = next(l for l in block.splitlines() if "Sera" in l)
        assert "acting this turn" in aldric_line
        assert "acting this turn" not in sera_line

    def test_falls_back_to_is_player_when_no_bindings(self):
        ckpt = _make_checkpoint()
        block = build_player_characters_block(ckpt, "aldric")
        assert "Aldric" in block

    def test_includes_character_id_annotation(self):
        """The id is what the router needs in `roster_moves[].character_id`
        when the actor moves themselves. Without it visible alongside the
        name, the router has no scaffolded id for player characters (NPCs
        get one in the registry; players don't appear there)."""
        ckpt = _make_checkpoint()
        ckpt.session.character_bindings = {"aldric": "1", "sera": "2"}
        block = build_player_characters_block(ckpt, "aldric")
        assert "(id: aldric)" in block
        assert "(id: sera)" in block
