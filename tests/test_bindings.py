"""Tests for multiplayer bindings — EngineBridge bind/unbind/dossier/roster."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from app.bot.engine_bridge import (
    EngineBridge,
    joinable_character_summaries,
    _summaries_from_checkpoint,
)
from app.engine.frontend_views import CharacterSummary
from app.engine.context_builder import (
    collect_player_ids,
)
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    PlayerSlotKind,
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
                is_playable=True,
                public_sheet=PublicSheet(role="wanderer", appearance="tall"),
                player_guidance=(
                    "Control Aldric's body, speech, and choices. "
                    "You can read the trail signs he recognizes."
                ),
                backstory="Raised by wolves.",
                personality="MODEL-ONLY: Keep his voice clipped and quiet.",
                known_context="The north trail is safe at dawn.",
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
    b = EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )
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
        assert by_id["aldric"].is_playable is True
        assert by_id["sera"].is_playable is False

    def test_joinable_filter_matches_join_picker_contract(self):
        summaries = [
            CharacterSummary(
                character_id="open",
                name="Open",
                role="hero",
                faction="",
                appearance="",
                status="active",
                is_playable=True,
                bound_user_id="",
            ),
            CharacterSummary(
                character_id="claimed",
                name="Claimed",
                role="hero",
                faction="",
                appearance="",
                status="active",
                is_playable=True,
                bound_user_id="42",
            ),
            CharacterSummary(
                character_id="npc",
                name="NPC",
                role="guide",
                faction="",
                appearance="",
                status="active",
                is_playable=False,
                bound_user_id="",
            ),
            CharacterSummary(
                character_id="culled",
                name="Culled",
                role="gone",
                faction="",
                appearance="",
                status="culled",
                is_playable=True,
                bound_user_id="",
            ),
        ]

        joinable = joinable_character_summaries(summaries)

        assert [summary.character_id for summary in joinable] == ["open"]

    def test_player_authored_slot_is_selectable_with_claim_guidance(self):
        ckpt = _make_checkpoint()
        ckpt.session.player_character_id = ""
        ckpt.session.character_bindings = {}
        ckpt.characters.append(CharacterRecord(
            character_id="blank_arrival",
            name="the Newcomer",
            status=CharacterStatus.dormant,
            location="not_yet_fictional",
            is_playable=True,
            player_slot_kind=PlayerSlotKind.player_authored,
            player_guidance="Choose this character's identity when joining.",
        ))

        summaries = _summaries_from_checkpoint(ckpt)
        joinable = joinable_character_summaries(summaries)

        summary = next(
            item for item in joinable
            if item.character_id == "blank_arrival"
        )
        assert summary.player_slot_kind == "player_authored"
        assert "identity" in summary.player_guidance


class TestBindUnbind:
    async def test_bind_user_happy_path(self, bridge: EngineBridge):
        await bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        assert bridge.get_user_binding(SESSION_ID, 42) == "sera"
        assert bridge.load_latest(
            SESSION_ID,
        ).session.pending_engine_state_updates == []

    async def test_bind_allows_dormant(self, bridge: EngineBridge):
        await bridge.bind_user(SESSION_ID, user_id=42, character_id="thane")
        assert bridge.get_user_binding(SESSION_ID, 42) == "thane"

    async def test_bind_rejects_culled(self, bridge: EngineBridge):
        with pytest.raises(ValueError, match="culled"):
            await bridge.bind_user(SESSION_ID, user_id=42, character_id="vex")

    async def test_bind_rejects_unknown_character(self, bridge: EngineBridge):
        with pytest.raises(ValueError, match="No character"):
            await bridge.bind_user(SESSION_ID, user_id=42, character_id="ghost")

    async def test_bind_rejects_second_user_on_same_char(self, bridge: EngineBridge):
        await bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        with pytest.raises(ValueError, match="already bound"):
            await bridge.bind_user(SESSION_ID, user_id=99, character_id="sera")

    async def test_bind_rejects_user_double_binding(self, bridge: EngineBridge):
        await bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        with pytest.raises(ValueError, match="already bound"):
            await bridge.bind_user(SESSION_ID, user_id=42, character_id="thane")

    async def test_bind_idempotent_for_same_pair(self, bridge: EngineBridge):
        # Binding the same user to the same character is a no-op (not an error).
        await bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        await bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        assert bridge.get_user_binding(SESSION_ID, 42) == "sera"

    async def test_unbind_frees_character(self, bridge: EngineBridge):
        await bridge.bind_user(SESSION_ID, user_id=42, character_id="sera")
        freed = await bridge.unbind_user(SESSION_ID, 42)
        assert freed == "sera"
        assert bridge.get_user_binding(SESSION_ID, 42) is None
        assert bridge.load_latest(
            SESSION_ID,
        ).session.pending_engine_state_updates == []
        # Another user can now claim sera.
        await bridge.bind_user(SESSION_ID, user_id=99, character_id="sera")
        assert bridge.get_user_binding(SESSION_ID, 99) == "sera"

    async def test_unbind_no_binding_returns_none(self, bridge: EngineBridge):
        assert await bridge.unbind_user(SESSION_ID, 999) is None


class TestStrictPlayerJoin:
    @staticmethod
    def _add_player_authored_slot(bridge: EngineBridge) -> None:
        ckpt = bridge.load_latest(SESSION_ID)
        ckpt.characters.append(CharacterRecord(
            character_id="blank_arrival",
            name="the Newcomer",
            status=CharacterStatus.dormant,
            location="not_yet_fictional",
            is_playable=True,
            player_slot_kind=PlayerSlotKind.player_authored,
            public_sheet=PublicSheet(role="new arrival"),
        ))
        bridge.checkpoint_mgr.save(ckpt)

    async def test_strict_claim_rejects_nonplayable_character(
        self, bridge: EngineBridge,
    ):
        with pytest.raises(ValueError, match="not an available player seat"):
            await bridge.claim_player_character(
                SESSION_ID,
                "sera",
                42,
            )

    async def test_player_authored_claim_is_atomic_on_missing_identity(
        self, bridge: EngineBridge, monkeypatch: pytest.MonkeyPatch,
    ):
        self._add_player_authored_slot(bridge)
        save = MagicMock(wraps=bridge.checkpoint_mgr.save)
        monkeypatch.setattr(bridge.checkpoint_mgr, "save", save)

        with pytest.raises(ValueError, match="describe.*appearance"):
            await bridge.claim_player_character(
                SESSION_ID,
                "blank_arrival",
                42,
                name="Mara Vale",
            )

        save.assert_not_called()
        current = bridge.load_latest(SESSION_ID)
        slot = next(
            character
            for character in current.characters
            if character.character_id == "blank_arrival"
        )
        assert current.session.character_bindings.get("blank_arrival") is None
        assert slot.name == "the Newcomer"
        assert slot.public_sheet.appearance == ""

    async def test_player_authored_claim_writes_identity_and_binding_together(
        self, bridge: EngineBridge, monkeypatch: pytest.MonkeyPatch,
    ):
        self._add_player_authored_slot(bridge)
        save = MagicMock(wraps=bridge.checkpoint_mgr.save)
        monkeypatch.setattr(bridge.checkpoint_mgr, "save", save)

        claimed = await bridge.claim_player_character(
            SESSION_ID,
            "blank_arrival",
            42,
            name="  Mara   Vale  ",
            appearance="scarlet coat and iron-gray braid",
        )

        save.assert_called_once()
        slot = next(
            character
            for character in claimed.characters
            if character.character_id == "blank_arrival"
        )
        assert claimed.session.character_bindings["blank_arrival"] == "42"
        assert slot.name == "Mara Vale"
        assert slot.public_sheet.appearance == (
            "scarlet coat and iron-gray braid"
        )
        assert slot.visuals.default_loadout == slot.public_sheet.appearance

    async def test_player_authored_leave_goes_off_frame_without_agent_handoff(
        self, bridge: EngineBridge, monkeypatch: pytest.MonkeyPatch,
    ):
        self._add_player_authored_slot(bridge)
        await bridge.claim_player_character(
            SESSION_ID,
            "blank_arrival",
            42,
            name="Mara Vale",
            appearance="scarlet coat and iron-gray braid",
        )
        ckpt = bridge.load_latest(SESSION_ID)
        slot = next(
            character
            for character in ckpt.characters
            if character.character_id == "blank_arrival"
        )
        slot.status = CharacterStatus.active
        slot.location = "gatehouse"
        bridge.checkpoint_mgr.save(ckpt)
        synthesize = AsyncMock()
        monkeypatch.setattr(bridge, "synthesize_personality", synthesize)

        freed = await bridge.leave_character(SESSION_ID, 42)

        assert freed == "blank_arrival"
        synthesize.assert_not_awaited()
        current = bridge.load_latest(SESSION_ID)
        slot = next(
            character
            for character in current.characters
            if character.character_id == "blank_arrival"
        )
        assert "blank_arrival" not in current.session.character_bindings
        assert slot.status == CharacterStatus.dormant
        assert slot.location == "outside_active_fiction"
        assert len(current.session.pending_engine_state_updates) == 1
        lifecycle_update = current.session.pending_engine_state_updates[0]
        assert "left active fiction" in lifecycle_update
        assert "now dormant" in lifecycle_update
        for forbidden in (
            "human",
            "player",
            "binding",
            "ai control",
            "agent",
            "protagonist",
        ):
            assert forbidden not in lifecycle_update.lower()

    async def test_standard_playable_leave_retains_agent_handoff(
        self, bridge: EngineBridge, monkeypatch: pytest.MonkeyPatch,
    ):
        ckpt = bridge.load_latest(SESSION_ID)
        sera = next(c for c in ckpt.characters if c.character_id == "sera")
        sera.is_playable = True
        bridge.checkpoint_mgr.save(ckpt)
        await bridge.claim_player_character(SESSION_ID, "sera", 42)
        synthesize = AsyncMock(return_value=bridge.load_latest(SESSION_ID))
        monkeypatch.setattr(bridge, "synthesize_personality", synthesize)

        await bridge.leave_character(SESSION_ID, 42)

        synthesize.assert_awaited_once_with(SESSION_ID, "sera")


class TestDossier:
    def test_projects_player_contract_and_character_interior_only(
        self,
        bridge: EngineBridge,
    ):
        """The player sees control guidance and character-known material,
        never the model's portrayal direction."""
        dossier = bridge.build_character_dossier(SESSION_ID, "aldric")
        assert "## Your Control & Perspective" in dossier
        assert "Control Aldric's body" in dossier
        assert "Raised by wolves" in dossier  # backstory
        assert "north trail" in dossier        # known context
        assert "survive" in dossier           # goal
        assert "royal sigil" in dossier       # secret THIS character keeps
        assert "MODEL-ONLY" not in dossier
        assert "clipped and quiet" not in dossier

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
    def test_unions_bindings_and_creator(self):
        """Under playable-2 semantics, `collect_player_ids` returns
        EXACTLY characters currently human-controlled — bindings
        keys + the creator (`player_character_id`). It deliberately
        does NOT pull in `is_playable=True` slots that nobody has
        claimed; those still run as agent NPCs."""
        ckpt = _make_checkpoint()
        ckpt.session.character_bindings = {"sera": "42"}
        ids = collect_player_ids(ckpt)
        # aldric (creator binding) + sera (explicit binding)
        assert ids == {"aldric", "sera"}

    def test_empty_bindings_still_includes_creator(self):
        """With no `character_bindings`, the creator is still
        considered controlled (legacy single-player checkpoints)."""
        ckpt = _make_checkpoint()
        ids = collect_player_ids(ckpt)
        assert "aldric" in ids

    def test_unbound_playable_is_not_player(self):
        """An is_playable=True character with no binding is an
        agent NPC, not a human-controlled player. Verifies the
        playable-2 semantic: 'playable' is an authoring flag,
        binding is the runtime state."""
        ckpt = _make_checkpoint()
        # Wipe creator binding so the only signal left is is_playable.
        ckpt.session.player_character_id = ""
        ckpt.session.character_bindings = {}
        # Aldric is is_playable=True but unbound. They should NOT
        # be in player_ids — that would block their agent ticks.
        ids = collect_player_ids(ckpt)
        assert "aldric" not in ids
