"""Tests for user-tunable settings exposed via /set on EngineBridge."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.bot.engine_bridge import EngineBridge
from app.engine.settings import (
    SETTINGS,
    SETTINGS_BY_KEY,
    UnknownSettingError,
    get_setting,
    list_settings_view,
    set_setting,
)
from app.schemas.characters import (
    ActorRecord,
    CharacterRecord,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import SessionState, WorldState


# ---- fixtures ---------------------------------------------------------------


SESSION_ID = "test_settings"


def _ckpt() -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            player_character_id="aldric",
        ),
        world_state=WorldState(),
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
                actor=ActorRecord(may_act_offstage=True),
            ),
        ],
    )


@pytest.fixture
def bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    b = EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )
    ckpt = _ckpt()
    ckpt.session.turn_index = 1
    b.checkpoint_mgr.save(ckpt)
    return b


# ---- registry internals -----------------------------------------------------


class TestSettingsRegistry:
    def test_registry_is_non_empty(self):
        # If this breaks the /settings surface is empty — likely a refactor
        # removed the only setting without replacing it. Add one or remove
        # the UI.
        assert SETTINGS, "no settings registered"
        assert "max_router_batches_without_player_input" in SETTINGS_BY_KEY

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
        assert get_setting(
            ckpt,
            "max_router_batches_without_player_input",
        ) == 12

    def test_get_unknown_raises(self):
        ckpt = _ckpt()
        with pytest.raises(UnknownSettingError):
            get_setting(ckpt, "not_a_real_setting")

    def test_set_setting_applies_parsed_value(self):
        ckpt = _ckpt()
        new = set_setting(
            ckpt,
            "max_router_batches_without_player_input",
            "7",
        )
        assert new == 7
        assert (
            ckpt.session.config.settings.max_router_batches_without_player_input
            == 7
        )

    def test_set_player_roll_mode(self):
        ckpt = _ckpt()
        new = set_setting(ckpt, "player_roll_mode", "interactive")
        assert new == "interactive"
        assert ckpt.session.config.settings.player_roll_mode == "interactive"

    @pytest.mark.parametrize("raw", ["visual_novel", "visual-novel", "vn"])
    def test_set_visual_novel_presentation_mode(self, raw):
        ckpt = _ckpt()

        new = set_setting(ckpt, "presentation_mode", raw)

        assert new == "visual_novel"
        assert ckpt.session.config.settings.presentation_mode == "visual_novel"

    def test_reject_unknown_presentation_mode(self):
        ckpt = _ckpt()
        with pytest.raises(ValueError, match="prose or visual_novel"):
            set_setting(ckpt, "presentation_mode", "cinematic")

    @pytest.mark.parametrize(
        "retired_key",
        ["image_generation_mode", "image_generation_every_n_beats"],
    )
    def test_actor_cadence_image_settings_are_retired(self, retired_key):
        ckpt = _ckpt()
        with pytest.raises(UnknownSettingError):
            get_setting(ckpt, retired_key)

    def test_autonomous_batch_cap_must_be_positive(self):
        ckpt = _ckpt()
        with pytest.raises(ValueError, match="at least 1"):
            set_setting(
                ckpt,
                "max_router_batches_without_player_input",
                "0",
            )

    def test_set_setting_unknown_raises(self):
        ckpt = _ckpt()
        with pytest.raises(UnknownSettingError):
            set_setting(ckpt, "phantom", "true")

    def test_set_setting_bad_value_raises(self):
        ckpt = _ckpt()
        with pytest.raises(ValueError):
            set_setting(
                ckpt,
                "max_router_batches_without_player_input",
                "zero",
            )

    def test_list_view_shape(self):
        ckpt = _ckpt()
        view = list_settings_view(ckpt)
        assert len(view) == len(SETTINGS)
        keys = {row["key"] for row in view}
        assert "max_router_batches_without_player_input" in keys
        row = next(
            r
            for r in view
            if r["key"] == "max_router_batches_without_player_input"
        )
        for field in (
            "key", "value", "rendered_value",
            "default", "rendered_default", "description",
        ):
            assert field in row


# ---- EngineBridge passthrough ----------------------------------------------


class TestEngineBridgeSettings:
    def test_get_returns_default(self, bridge: EngineBridge):
        assert bridge.get_setting(
            SESSION_ID,
            "max_router_batches_without_player_input",
        ) == 12

    def test_set_persists_across_reloads(self, bridge: EngineBridge):
        asyncio.run(
            bridge.set_setting(
                SESSION_ID,
                "max_router_batches_without_player_input",
                "9",
            )
        )
        # New handle, fresh disk read.
        assert bridge.get_setting(
            SESSION_ID,
            "max_router_batches_without_player_input",
        ) == 9

    def test_set_waits_for_running_turn_then_persists(
        self,
        bridge: EngineBridge,
        monkeypatch: pytest.MonkeyPatch,
    ):
        observed_modes: list[str] = []

        async def exercise() -> None:
            turn_started = asyncio.Event()
            finish_turn = asyncio.Event()

            async def fake_run_turn_locked(**_kwargs):
                ckpt = bridge.checkpoint_mgr.load_latest(SESSION_ID)
                observed_modes.append(
                    ckpt.session.config.settings.presentation_mode
                )
                turn_started.set()
                await finish_turn.wait()
                ckpt.session.turn_index += 1
                bridge.checkpoint_mgr.save(ckpt)
                return None

            monkeypatch.setattr(
                bridge,
                "_run_turn_locked",
                fake_run_turn_locked,
            )
            turn_task = asyncio.create_task(bridge.run_turn(
                session_id=SESSION_ID,
                user_input="I wait.",
                acting_character_id="aldric",
            ))
            await turn_started.wait()
            setting_task = asyncio.create_task(bridge.set_setting(
                SESSION_ID,
                "presentation_mode",
                "visual_novel",
            ))
            await asyncio.sleep(0)
            assert setting_task.done() is False

            finish_turn.set()
            await turn_task
            assert await setting_task == "visual_novel"

        asyncio.run(exercise())

        latest = bridge.checkpoint_mgr.load_latest(SESSION_ID)
        assert observed_modes == ["prose"]
        assert latest.session.turn_index == 2
        assert latest.session.config.settings.presentation_mode == "visual_novel"

    def test_story_start_preserves_runtime_defaults(
        self, bridge: EngineBridge,
    ):
        story_id = "runtime_defaults_story"
        story_dir = bridge.stories_dir / story_id
        story_dir.mkdir(parents=True)
        ckpt = _ckpt()
        ckpt.session.session_id = story_id
        ckpt.session.story_id = story_id
        ckpt.session.config.settings.max_router_batches_without_player_input = 8
        (story_dir / "ckpt_0000.json").write_text(
            ckpt.model_dump_json(indent=2)
        )

        bridge.create_empty_session("loaded_ticks_default")
        loaded = bridge.load_story_into_session(
            "loaded_ticks_default", story_id,
        )

        assert (
            loaded.session.config.settings.max_router_batches_without_player_input
            == 8
        )
        assert not hasattr(loaded, "config")

    def test_list_has_rows(self, bridge: EngineBridge):
        rows = bridge.list_settings(SESSION_ID)
        assert len(rows) >= 1
        assert all("key" in r and "value" in r for r in rows)

    def test_known_setting_keys(self, bridge: EngineBridge):
        keys = bridge.known_setting_keys()
        assert keys == [spec.key for spec in SETTINGS]
        assert {
            "max_router_batches_without_player_input",
            "ruleset_id",
            "player_roll_mode",
            "presentation_mode",
        }.issubset(keys)
