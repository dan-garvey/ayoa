"""Tests for user-tunable settings exposed via /set on EngineBridge."""

from __future__ import annotations

from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.bot.engine_bridge import EngineBridge
from app.engine.settings import (
    SETTINGS,
    SETTINGS_BY_KEY,
    UnknownSettingError,
    _parse_bool,
    get_setting,
    list_settings_view,
    set_setting,
)
from app.llm.config import LLMConfig
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import LocationState, SessionState, WorldState


# ---- fixtures ---------------------------------------------------------------


SESSION_ID = "test_settings"


def _ckpt() -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            player_character_id="aldric",
        ),
        world_state=WorldState(locations=LocationState()),
        characters=[
            CharacterRecord(
                character_id="aldric",
                name="Aldric",
                is_playable=True,
                location="hall",
                public_sheet=PublicSheet(role="envoy"),
            ),
            CharacterRecord(
                character_id="regent",
                name="The Regent",
                location="hall",
                public_sheet=PublicSheet(role="regent"),
                private_state=PrivateState(intentions_enabled=True),
            ),
        ],
    )


@pytest.fixture
def bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    b = EngineBridge(saves_dir=str(tmp_path), prompts_dir="app/prompts")
    ckpt = _ckpt()
    ckpt.session.turn_index = 1
    b.checkpoint_mgr.save(ckpt)
    return b


# ---- registry internals -----------------------------------------------------


class TestSettingsRegistry:
    def test_parse_bool_accepts_common_spellings(self):
        for raw in ("true", "True", "TRUE", "yes", "on", "1", "enabled"):
            assert _parse_bool(raw) is True
        for raw in ("false", "no", "off", "0", "disabled"):
            assert _parse_bool(raw) is False

    def test_parse_bool_rejects_garbage(self):
        with pytest.raises(ValueError, match="boolean"):
            _parse_bool("maybe")

    def test_registry_is_non_empty(self):
        # If this breaks the /settings surface is empty — likely a refactor
        # removed the only setting without replacing it. Add one or remove
        # the UI.
        assert SETTINGS, "no settings registered"
        assert "ticks_enabled" in SETTINGS_BY_KEY

    def test_every_spec_round_trips_its_default(self):
        """Every setting's default value should render cleanly back through
        its parser (proves parse + render are consistent)."""
        for spec in SETTINGS:
            rendered = spec.render(spec.default)
            reparsed = spec.parse(rendered)
            assert reparsed == spec.default, (
                f"{spec.key}: default {spec.default!r} renders as "
                f"{rendered!r} but re-parsing yields {reparsed!r}"
            )


class TestSettingsHelpers:
    def test_get_setting_returns_default_on_fresh_ckpt(self):
        ckpt = _ckpt()
        assert get_setting(ckpt, "ticks_enabled") is True

    def test_get_unknown_raises(self):
        ckpt = _ckpt()
        with pytest.raises(UnknownSettingError):
            get_setting(ckpt, "not_a_real_setting")

    def test_set_setting_applies_parsed_value(self):
        ckpt = _ckpt()
        new = set_setting(ckpt, "ticks_enabled", "off")
        assert new is False
        assert ckpt.session.config.settings.ticks_enabled is False

    def test_set_player_roll_mode(self):
        ckpt = _ckpt()
        new = set_setting(ckpt, "player_roll_mode", "interactive")
        assert new == "interactive"
        assert ckpt.session.config.settings.player_roll_mode == "interactive"

    def test_set_setting_unknown_raises(self):
        ckpt = _ckpt()
        with pytest.raises(UnknownSettingError):
            set_setting(ckpt, "phantom", "true")

    def test_set_setting_bad_value_raises(self):
        ckpt = _ckpt()
        with pytest.raises(ValueError):
            set_setting(ckpt, "ticks_enabled", "kinda")

    def test_list_view_shape(self):
        ckpt = _ckpt()
        view = list_settings_view(ckpt)
        assert len(view) == len(SETTINGS)
        keys = {row["key"] for row in view}
        assert "ticks_enabled" in keys
        row = next(r for r in view if r["key"] == "ticks_enabled")
        for field in (
            "key", "value", "rendered_value",
            "default", "rendered_default", "description",
        ):
            assert field in row


# ---- EngineBridge passthrough ----------------------------------------------


class TestEngineBridgeSettings:
    def test_get_returns_default(self, bridge: EngineBridge):
        assert bridge.get_setting(SESSION_ID, "ticks_enabled") is True

    def test_set_persists_across_reloads(self, bridge: EngineBridge):
        bridge.set_setting(SESSION_ID, "ticks_enabled", "false")
        # New handle, fresh disk read.
        assert bridge.get_setting(SESSION_ID, "ticks_enabled") is False

    def test_list_has_rows(self, bridge: EngineBridge):
        rows = bridge.list_settings(SESSION_ID)
        assert len(rows) >= 1
        assert all("key" in r and "value" in r for r in rows)

    def test_known_setting_keys(self, bridge: EngineBridge):
        keys = bridge.known_setting_keys()
        assert keys == [spec.key for spec in SETTINGS]
        assert {
            "ticks_enabled",
            "ruleset_id",
            "player_roll_mode",
        }.issubset(keys)
