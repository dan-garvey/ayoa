"""Tests for the router `activate` (wake) channel.

Waking is the inverse of `dormant`: the router flips a dormant character back
to active and places them where they re-enter. It lets a benched Hero or a
reserved off-stage persona return using their authored self, and it is how the
One-Star Ascension summon pool is meant to enter play (rather than the router
narrating a phantom arrival with no record).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.engine.character_manager import CharacterManager
from app.schemas.characters import CharacterStatus
from app.schemas.event_router import EventRouterOutput
from app.schemas.checkpoint import CheckpointFile
from tests.support.factories import router_output


SEED_PATH = (
    Path(__file__).resolve().parent.parent
    / "app" / "storage" / "stories" / "one_star_ascension_s1" / "ckpt_0000.json"
)


def _seed() -> CheckpointFile:
    return CheckpointFile.model_validate(
        json.loads(SEED_PATH.read_text(encoding="utf-8"))
    )


def test_activate_defaults_to_empty_when_omitted() -> None:
    # Constructions/fixtures that predate `activate` still build (the
    # mode="before" validator fills it), and it survives a model_dump round-trip.
    routed = router_output()
    assert routed.activate == []
    assert EventRouterOutput(**routed.model_dump()).activate == []


def test_activate_wakes_dormant_character_and_places_them() -> None:
    checkpoint = _seed()
    wren = next(c for c in checkpoint.characters if c.character_id == "wren_thelantern")
    assert wren.status == CharacterStatus.dormant

    routed = router_output(activate=[
        {"character_id": "wren_thelantern", "location_label": "niflheim_lobby"},
    ])
    CharacterManager().apply_roster_updates(checkpoint, routed)

    assert wren.status == CharacterStatus.active
    assert wren.location == "niflheim_lobby"


def test_activate_cannot_wake_the_culled() -> None:
    checkpoint = _seed()
    wren = next(c for c in checkpoint.characters if c.character_id == "wren_thelantern")
    wren.status = CharacterStatus.culled

    routed = router_output(activate=[
        {"character_id": "wren_thelantern", "location_label": "niflheim_lobby"},
    ])
    CharacterManager().apply_roster_updates(checkpoint, routed)

    assert wren.status == CharacterStatus.culled  # the dead do not wake


def test_activate_ignores_unknown_character() -> None:
    checkpoint = _seed()
    routed = router_output(activate=[
        {"character_id": "nope_404", "location_label": "somewhere"},
    ])
    # Must not raise; unknown ids are logged and skipped.
    CharacterManager().apply_roster_updates(checkpoint, routed)


def test_seed_summon_contract_prefers_wake_over_phantom() -> None:
    checkpoint = _seed()
    hidden = (
        checkpoint.world_state.hidden_lore
        + "\n"
        + "\n".join(checkpoint.world_state.hidden_facts)
    ).lower()
    # The wake channel is the documented way to bring a reserved persona in,
    # and a described arrival with no activate/spawn is called out as a phantom.
    assert "activate" in hidden
    assert "phantom" in hidden
