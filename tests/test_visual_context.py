from app.engine.visual_context import (
    mark_visual_introductions,
    plan_event_visual_introductions,
    plan_render_visual_introductions,
)
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterVisuals,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.events import ObservableFact
from app.schemas.state import RenderBufferEntry, SessionState
from tests.support.factories import router_output


def _event(text: str) -> EventRouterOutput:
    return router_output(
        event_id="evt_group",
        facts=[ObservableFact.all(text)],
        observer_ids=["alice"],
        event_kind="directed_at_player",
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


def test_agent_event_plan_uses_speaker_not_quoted_mentions():
    ckpt = CheckpointFile(
        session=SessionState(session_id="s"),
        characters=[
            CharacterRecord(character_id="bob", name="Bob"),
            _character("alice", "Alice", CharacterAgentTier.standard),
            _character("pip", "Pip", CharacterAgentTier.standard),
        ],
    )
    event = _event("Alice says, 'Pip is coming later.'")

    plan = plan_event_visual_introductions(
        ckpt,
        viewer_id="bob",
        event=event,
        observation_level="direct",
        max_loadouts=3,
    )

    assert [intro.character_id for intro in plan.loadouts] == ["alice"]
    assert plan.mark_character_ids == ["alice"]


def test_agent_event_plan_ignores_plain_name_mentions():
    ckpt = CheckpointFile(
        session=SessionState(session_id="s"),
        characters=[
            CharacterRecord(character_id="bob", name="Bob"),
            _character("alice", "Alice", CharacterAgentTier.standard),
            _character("pip", "Pip", CharacterAgentTier.standard),
        ],
    )
    event = _event("Alice points toward Pip's empty chair.")

    plan = plan_event_visual_introductions(
        ckpt,
        viewer_id="bob",
        event=event,
        observation_level="direct",
        max_loadouts=3,
    )

    assert plan.loadouts == []
    assert plan.mark_character_ids == []
