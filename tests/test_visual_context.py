from app.engine.visual_context import (
    mark_visual_introductions,
    plan_render_visual_introductions,
)
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterVisuals,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication
from app.schemas.state import RenderBufferEntry, SessionState


def _event(text: str) -> EventRouterOutput:
    return EventRouterOutput(
        event_id="evt_group",
        decision_rationale="test fixture",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(feasible=True),
            observable_facts=[ObservableFact.all(text)],
        ),
        observers=[
            ObserverEntry(
                character_id="alice",
                observation_level="d",
                routing_role="observe_only",
            ),
        ],
        requires_responders=False,
        required_responders=[],
        ends_beat=True,
        ends_beat_reason="directed_at_player",
        spawn=[],
        dormant=[],
        cull=[],
    )


def _character(
    character_id: str,
    name: str,
    tier: CharacterAgentTier,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        agent_tier=tier,
        visuals=CharacterVisuals(default_loadout=f"{name} loadout."),
    )


def test_first_meeting_plan_caps_by_tier_and_leaves_overflow_unintroduced():
    ckpt = CheckpointFile(
        session=SessionState(session_id="s", character_bindings={"alice": "1"}),
        characters=[
            CharacterRecord(character_id="alice", name="Alice"),
            _character("utility_a", "Utility A", CharacterAgentTier.utility),
            _character("standard_a", "Standard A", CharacterAgentTier.standard),
            _character("premium_a", "Premium A", CharacterAgentTier.premium),
            _character("utility_b", "Utility B", CharacterAgentTier.utility),
            _character("standard_b", "Standard B", CharacterAgentTier.standard),
        ],
    )
    event = _event(
        "Utility A, Standard A, Premium A, Utility B, and Standard B enter."
    )
    resolved = [(RenderBufferEntry(event_id=event.event_id), event)]

    first = plan_render_visual_introductions(
        ckpt,
        viewer_id="alice",
        resolved=resolved,
        max_loadouts=3,
    )

    assert [intro.character_id for intro in first.loadouts] == [
        "premium_a",
        "standard_a",
        "standard_b",
    ]
    mark_visual_introductions(ckpt, "alice", first.mark_character_ids)

    second = plan_render_visual_introductions(
        ckpt,
        viewer_id="alice",
        resolved=resolved,
        max_loadouts=3,
    )

    assert [intro.character_id for intro in second.loadouts] == [
        "utility_a",
        "utility_b",
    ]
    assert ckpt.session.visual_introductions["alice"] == [
        "premium_a",
        "standard_a",
        "standard_b",
    ]


def test_first_meeting_plan_uses_explicit_loadout_not_raw_appearance():
    ckpt = CheckpointFile(
        session=SessionState(session_id="s", character_bindings={"alice": "1"}),
        characters=[
            CharacterRecord(character_id="alice", name="Alice"),
            CharacterRecord(
                character_id="korva",
                name="Korva",
                public_sheet={"appearance": "Raw private-sheet appearance."},
                visuals=CharacterVisuals(default_loadout=""),
            ),
        ],
    )
    event = _event("Korva enters the room.")

    plan = plan_render_visual_introductions(
        ckpt,
        viewer_id="alice",
        resolved=[(RenderBufferEntry(event_id=event.event_id), event)],
        max_loadouts=3,
    )

    assert plan.loadouts == []
    assert plan.mark_character_ids == []
