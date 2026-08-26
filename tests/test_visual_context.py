from app.engine.visual_context import (
    format_narrator_visual_introductions,
    mark_visual_introductions,
    plan_event_visual_introductions,
    plan_render_visual_introductions,
)
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterDescriptions,
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
        event_kind="cascade_exhausted",
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


def test_narrator_first_meeting_excludes_public_bio_without_exterior():
    ckpt = CheckpointFile(
        session=SessionState(session_id="s", character_bindings={"alice": "1"}),
        characters=[
            CharacterRecord(character_id="alice", name="Alice"),
            CharacterRecord(
                character_id="korva",
                name="Korva",
                descriptions=CharacterDescriptions(
                    public="Korva is an S-rank guild quartermaster."
                ),
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
    assert format_narrator_visual_introductions(plan.loadouts) == ""


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


def test_agent_remote_voice_does_not_consume_intro_before_direct_speech():
    ckpt = CheckpointFile(
        session=SessionState(session_id="s"),
        characters=[
            CharacterRecord(character_id="bob", name="Bob"),
            _character("pip", "Pip", CharacterAgentTier.standard),
        ],
    )

    remote = plan_event_visual_introductions(
        ckpt,
        viewer_id="bob",
        event=_event("Pip says over the radio, 'I will arrive later.'"),
        observation_level="direct",
        max_loadouts=3,
    )
    mark_visual_introductions(ckpt, "bob", remote.mark_character_ids)

    assert remote.loadouts == []
    assert ckpt.session.visual_introductions == {}

    meeting = plan_event_visual_introductions(
        ckpt,
        viewer_id="bob",
        event=_event("Pip holds a sealed letter and says, 'I made it.'"),
        observation_level="direct",
        max_loadouts=3,
    )

    assert [intro.character_id for intro in meeting.loadouts] == ["pip"]
    assert meeting.mark_character_ids == ["pip"]


def test_narrator_remote_references_do_not_consume_intro_before_copresence():
    ckpt = CheckpointFile(
        session=SessionState(session_id="s", character_bindings={"alice": "1"}),
        characters=[
            CharacterRecord(character_id="alice", name="Alice"),
            _character("pip", "Pip", CharacterAgentTier.standard),
        ],
    )
    remote_texts = [
        "The radio crackles: 'Pip will arrive later.'",
        "Pip's message says he will arrive later.",
        "A wall feed shows Pip walking through the outer courtyard.",
        "Alice points toward Pip's empty chair.",
        "Alice mentions Pip and smiles.",
    ]
    for index, text in enumerate(remote_texts):
        event = _event(text)
        event.event_id = f"evt_remote_{index}"
        plan = plan_render_visual_introductions(
            ckpt,
            viewer_id="alice",
            resolved=[(
                RenderBufferEntry(
                    event_id=event.event_id,
                    observation_level="direct",
                ),
                event,
            )],
            max_loadouts=3,
        )
        mark_visual_introductions(ckpt, "alice", plan.mark_character_ids)
        assert plan.loadouts == []

    assert ckpt.session.visual_introductions == {}

    meeting = _event("Pip is now beside Alice in the gatehouse.")
    plan = plan_render_visual_introductions(
        ckpt,
        viewer_id="alice",
        resolved=[(
            RenderBufferEntry(
                event_id=meeting.event_id,
                observation_level="direct",
            ),
            meeting,
        )],
        max_loadouts=3,
    )

    assert [intro.character_id for intro in plan.loadouts] == ["pip"]
    assert plan.mark_character_ids == ["pip"]


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
