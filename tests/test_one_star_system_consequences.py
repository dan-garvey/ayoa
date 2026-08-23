from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.engine.turn_loop import broadcast_event
from app.engine.turn_loop_dispatcher import (
    _append_one_star_system_consequences,
    _router_history_record,
)
from tests.support.factories import character_record, checkpoint, router_output


def test_adapter_consequence_joins_current_event_history_and_fanout() -> None:
    ckpt = checkpoint(
        bindings={"alice": "human-1"},
        characters=[
            character_record("alice", name="Alice", is_playable=True),
            character_record("pip", name="Pip", is_playable=False),
        ],
    )
    result = router_output(
        event_id="ev_progression",
        observer_ids=["pip"],
        duration_s=4,
    )
    consequence = SimpleNamespace(
        text="[ LEVEL UP: Pip, Lv.2 ]",
        recipient_character_ids=("alice", "pip"),
    )

    _append_one_star_system_consequences(ckpt, result, [consequence])

    assert [observer.character_id for observer in result.observers] == [
        "pip",
        "alice",
    ]
    original = result.canonical_event.observable_facts[0]
    assert original.audience == "only"
    assert original.visible_to == ["pip"]
    added = result.canonical_event.observable_facts[-1]
    assert added.text == "[ LEVEL UP: Pip, Lv.2 ]"
    assert added.audience == "only"
    assert added.visible_to == ["alice", "pip"]
    assert added.at_offset_s == 4
    record = _router_history_record(
        acting_character_id="pip",
        result=result,
    )
    assert record.count("[ LEVEL UP: Pip, Lv.2 ]") == 1

    visible_humans = broadcast_event(ckpt, result, actor_id="pip")

    assert visible_humans == ["alice"]
    pip = next(character for character in ckpt.characters if character.character_id == "pip")
    assert any("[ LEVEL UP: Pip, Lv.2 ]" in item for item in pip.pending_observations)
    assert ckpt.session.render_buffers["alice"][-1].event_id == "ev_progression"


def test_adapter_consequence_does_not_widen_existing_scene_facts() -> None:
    ckpt = checkpoint(
        characters=[
            character_record("alice", name="Alice"),
            character_record("pip", name="Pip"),
        ],
    )
    result = router_output(event_id="ev_scoped", observer_ids=["pip"])

    _append_one_star_system_consequences(
        ckpt,
        result,
        [SimpleNamespace(
            text="A private System notice.",
            recipient_character_ids=("alice",),
        )],
    )
    broadcast_event(ckpt, result, actor_id="pip")

    alice = next(
        character
        for character in ckpt.characters
        if character.character_id == "alice"
    )
    delivered = "\n".join(alice.pending_observations)
    assert "A private System notice." in delivered
    assert "Something happens." not in delivered


def test_adapter_consequence_rejects_an_unknown_recipient() -> None:
    ckpt = checkpoint(
        characters=[character_record("alice", name="Alice")],
    )
    result = router_output(event_id="ev_unknown", observer_ids=["alice"])

    with pytest.raises(ValueError, match="unknown recipients: missing"):
        _append_one_star_system_consequences(
            ckpt,
            result,
            [SimpleNamespace(
                text="A private System notice.",
                recipient_character_ids=("missing",),
            )],
        )
