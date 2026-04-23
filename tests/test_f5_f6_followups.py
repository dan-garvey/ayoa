"""Targeted tests for the F5/F6 cleanup batch:

- F5.2: max_responders + tick_concurrency settings are honored and
  clamped against module hard caps.
- F5.4: prompt_render_ms on PhaseLatency sums across call usages.
- F6.2: long directives (>1000 chars) log an advisory warning but are
  still delivered.

Other items in this batch are structural / documentation-only:
  F5.1 (save collapse) — behavior is covered by the existing orchestrator
       turn tests; saving once vs twice is observable only on crash.
  F5.7 (import experiment TODO) — docs only.
  F6.1 (MIME validation) — Discord-interaction-heavy; covered by manual
       inspection of commands.py and the ephemeral rejection path.
"""

from __future__ import annotations

import pytest

# v11: the v8 orchestrator is gone; these follow-up tests reached into
# module-level constants (DIRECTIVE_LENGTH_WARN, RESPONDERS_HARD_CAP,
# TICK_CONCURRENCY_HARD_CAP) + per-phase latency plumbing that no longer
# exist. Re-port to the v11 pipeline when ticks + directives features
# are rebuilt on top of run_beat; skip the whole module for now.
pytestmark = pytest.mark.skip(
    reason="v11: legacy v8 orchestrator constants removed; re-port against run_beat",
)

import asyncio  # noqa: E402
import logging  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

# Stub names so the module still parses under skip.
DIRECTIVE_LENGTH_WARN = 1000
RESPONDERS_HARD_CAP = 8
TICK_CONCURRENCY_HARD_CAP = 8


class Orchestrator:  # type: ignore[no-redef]
    pass
from app.engine.settings import (
    SETTINGS_BY_KEY,
    _parse_int_nonneg,
    _parse_int_positive,
    set_setting,
)
from app.llm.config import LLMConfig
from app.schemas.agents import CharacterAgentOutput  # noqa: F401

# Module-level skip is set above. Stubs let the file parse without the
# legacy schema names that Commit 1 removed; the actual skipped tests
# never execute these paths.
class _Stub:
    def __init__(self, **kw): pass

DirectiveSend = _Stub  # type: ignore[assignment]
PrivateUpdates = _Stub  # type: ignore[assignment]
PublicResponse = _Stub  # type: ignore[assignment]
from app.schemas.characters import (
    CharacterRecord,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.responses import PhaseLatency
from app.schemas.state import LocationState, SessionState, WorldState


def _ckpt() -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id="test_f5_f6",
            player_character_id="aldric",
            turn_index=3,
        ),
        world_state=WorldState(
            locations=LocationState(
                current_scene_id="hall",
                scene_graph={
                    "hall": {"name": "Hall", "description": "", "connected_to": [], "properties": {}},
                },
            ),
        ),
        characters=[
            CharacterRecord(
                character_id="aldric",
                name="Aldric",
                is_player=True,
                location="hall",
                public_sheet=PublicSheet(role="player"),
            ),
            CharacterRecord(
                character_id="regent",
                name="Regent",
                location="hall",
                public_sheet=PublicSheet(role="regent"),
                private_state=PrivateState(intentions_enabled=True),
            ),
        ],
    )


def _orch() -> Orchestrator:
    client = MagicMock()
    client.complete = AsyncMock()
    client.config = LLMConfig()
    return Orchestrator(client, MagicMock(), MagicMock())


# ---- F5.2: caps in settings -------------------------------------------------


class TestCapsSettings:
    def test_registry_exposes_both_caps(self):
        assert "max_responders" in SETTINGS_BY_KEY
        assert "tick_concurrency" in SETTINGS_BY_KEY

    def test_max_responders_defaults_to_3(self):
        ckpt = _ckpt()
        assert ckpt.session.config.settings.max_responders == 3

    def test_tick_concurrency_defaults_to_4(self):
        ckpt = _ckpt()
        assert ckpt.session.config.settings.tick_concurrency == 4

    def test_max_responders_accepts_zero(self):
        """0 is a legal diagnostic value — silences all NPC responses."""
        ckpt = _ckpt()
        set_setting(ckpt, "max_responders", "0")
        assert ckpt.session.config.settings.max_responders == 0

    def test_tick_concurrency_rejects_zero(self):
        ckpt = _ckpt()
        with pytest.raises(ValueError):
            set_setting(ckpt, "tick_concurrency", "0")

    def test_both_reject_negatives(self):
        ckpt = _ckpt()
        with pytest.raises(ValueError):
            set_setting(ckpt, "max_responders", "-1")
        with pytest.raises(ValueError):
            set_setting(ckpt, "tick_concurrency", "-1")

    def test_both_reject_non_integers(self):
        with pytest.raises(ValueError):
            _parse_int_nonneg("3.5")
        with pytest.raises(ValueError):
            _parse_int_positive("lots")


class TestHardCaps:
    """Module-level hard caps exist as a safety rail above session settings.
    If a bad setting somehow slipped through, the hard cap prevents us
    from fanning out to hundreds of agents."""

    def test_hard_caps_exist(self):
        assert RESPONDERS_HARD_CAP >= 1
        assert TICK_CONCURRENCY_HARD_CAP >= 1

    def test_hard_cap_bounds_are_sane(self):
        # Defaults should be well below the hard caps — if a default
        # hits or exceeds the hard cap, one of the two is wrong.
        ckpt = _ckpt()
        assert ckpt.session.config.settings.max_responders <= RESPONDERS_HARD_CAP
        assert ckpt.session.config.settings.tick_concurrency <= TICK_CONCURRENCY_HARD_CAP


# ---- F5.4: prompt_render_ms surfaces on PhaseLatency ------------------------


class TestPromptRenderTime:
    def test_field_defaults_to_zero(self):
        p = PhaseLatency(phase="x", duration_ms=100.0)
        assert p.prompt_render_ms == 0.0

    def test_phase_latency_sums_render_time_across_calls(self):
        orch = _orch()
        usages = [
            {"prompt_tokens": 100, "prompt_render_ms": 12.5},
            {"prompt_tokens": 200, "prompt_render_ms": 7.5},
            {"prompt_tokens": 50, "prompt_render_ms": 5.0},
        ]
        p = orch._phase_latency(
            phase="agent_fanout",
            start_mono=0.0,
            model="claude-sonnet-4-6",
            usages=usages,
        )
        assert p.input_tokens == 350
        assert p.prompt_render_ms == pytest.approx(25.0)

    def test_phase_latency_handles_missing_render_field(self):
        """Old usage dicts (before F5.4) don't carry prompt_render_ms.
        Summing should silently treat them as 0."""
        orch = _orch()
        p = orch._phase_latency(
            phase="x", start_mono=0.0, model="",
            usages=[{"prompt_tokens": 100}],
        )
        assert p.prompt_render_ms == 0.0


# ---- F6.2: long-directive warning -------------------------------------------


class TestLongDirectiveWarning:
    def test_warn_threshold_is_1000(self):
        assert DIRECTIVE_LENGTH_WARN == 1000

    def test_long_directive_warns_but_delivers(self, caplog):
        orch = _orch()
        ckpt = _ckpt()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")

        payload = "x" * (DIRECTIVE_LENGTH_WARN + 50)
        output = CharacterAgentOutput(
            character_id="regent",
            public_response=PublicResponse(
                actions=[], dialogue=[], expression="",
            ),
            private_updates=PrivateUpdates(
                current_objectives=[],
                directives_sent=[DirectiveSend(to="aldric", content=payload)],
                moved_to="",
                scenes_created=[],
            ),
        )

        with caplog.at_level(logging.WARNING):
            orch._apply_agent_private_updates(ckpt, regent, output, base_depth=0)

        # Warning fired.
        assert any("Long directive" in r.message for r in caplog.records)
        # But the directive was still delivered.
        aldric = next(c for c in ckpt.characters if c.character_id == "aldric")
        assert len(aldric.incoming_directives) == 1
        assert aldric.incoming_directives[0].content == payload

    def test_short_directive_does_not_warn(self, caplog):
        orch = _orch()
        ckpt = _ckpt()
        regent = next(c for c in ckpt.characters if c.character_id == "regent")

        output = CharacterAgentOutput(
            character_id="regent",
            public_response=PublicResponse(
                actions=[], dialogue=[], expression="",
            ),
            private_updates=PrivateUpdates(
                current_objectives=[],
                directives_sent=[DirectiveSend(to="aldric", content="short note")],
                moved_to="",
                scenes_created=[],
            ),
        )

        with caplog.at_level(logging.WARNING):
            orch._apply_agent_private_updates(ckpt, regent, output, base_depth=0)

        assert not any("Long directive" in r.message for r in caplog.records)
        aldric = next(c for c in ckpt.characters if c.character_id == "aldric")
        assert aldric.incoming_directives[0].content == "short note"
