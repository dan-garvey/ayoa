from pathlib import Path

from app.engine.one_star_visuals import VEILED_FIRST_LOOK
from app.engine.visual_context import (
    format_narrator_visual_introductions,
    mark_visual_introductions,
    physically_present_character_ids,
    plan_event_visual_introductions,
    plan_render_visual_introductions,
    visually_staged_character_ids,
)
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterDescriptions,
    CharacterRecord,
    CharacterVisuals,
    PlayerSlotKind,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.events import ObservableFact
from app.schemas.state import RenderBufferEntry, SessionState
from tests.support.factories import router_output


ONE_STAR_CHECKPOINT = Path(
    "app/storage/stories/one_star_ascension_s1/ckpt_0000.json"
)


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


def test_presence_excludes_unbound_player_authored_slot_until_claimed():
    checkpoint = CheckpointFile(
        session=SessionState(
            session_id="s",
            character_bindings={"guide": "guide-player"},
        ),
        characters=[
            CharacterRecord(
                character_id="guide",
                name="Guide",
                location="lobby",
            ),
            CharacterRecord(
                character_id="newcomer",
                name="the Newcomer",
                location="not_yet_fictional",
                player_slot_kind=PlayerSlotKind.player_authored,
            ),
        ],
    )
    text = "Guide waits with the newcomer in the lobby courtyard."

    assert physically_present_character_ids(checkpoint, [text]) == {"guide"}

    checkpoint.session.character_bindings["newcomer"] = "newcomer-player"
    assert physically_present_character_ids(checkpoint, [text]) == {
        "guide",
        "newcomer",
    }


def test_presence_recognizes_common_vn_motion_without_remote_mentions():
    checkpoint = CheckpointFile(
        session=SessionState(session_id="s"),
        characters=[
            CharacterRecord(character_id="iselle", name="Iselle"),
            CharacterRecord(character_id="mara", name="Mara Venn"),
            CharacterRecord(character_id="edda", name="Edda Brin"),
        ],
    )

    assert physically_present_character_ids(
        checkpoint,
        ["Iselle flits up before Mara Venn and Edda Brin."],
    ) == {"iselle", "mara", "edda"}
    assert physically_present_character_ids(
        checkpoint,
        ["Mara Venn shifts closer to Edda Brin."],
    ) == {"mara", "edda"}
    assert physically_present_character_ids(
        checkpoint,
        ["Iselle tilts her head while a radio report mentions Mara Venn."],
    ) == {"iselle"}
    assert physically_present_character_ids(
        checkpoint,
        [
            "Mara Venn says, 'I will wait.' Mara Venn looks to Edda Brin."
        ],
    ) == {"mara"}
    assert physically_present_character_ids(
        checkpoint,
        ["Iselle’s smile holds while Mara Venn's empty chair stays vacant."],
    ) == {"iselle"}


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


def test_one_star_master_first_look_is_veiled_until_reveal_threshold():
    ckpt = CheckpointFile.model_validate_json(ONE_STAR_CHECKPOINT.read_text())
    renna = next(
        character
        for character in ckpt.characters
        if character.character_id == "renna_holt"
    )
    renna.status = "active"
    event = _event("Renna Holt steps into the lobby courtyard.")
    resolved = [(RenderBufferEntry(event_id=event.event_id), event)]

    veiled = plan_render_visual_introductions(
        ckpt,
        viewer_id="the_master",
        resolved=resolved,
    )
    assert veiled.loadouts[0].default_loadout == VEILED_FIRST_LOOK
    assert veiled.loadouts[0].public_context == ""
    assert renna.visuals.default_loadout not in VEILED_FIRST_LOOK

    renna.mechanics["one_star_hero"]["current_stars"] = 2
    revealed = plan_render_visual_introductions(
        ckpt,
        viewer_id="the_master",
        resolved=resolved,
    )
    assert revealed.loadouts[0].default_loadout == renna.visuals.default_loadout


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


def test_channel_scoping_is_per_subject_and_clause_for_both_consumers():
    ckpt = CheckpointFile(
        session=SessionState(session_id="s"),
        characters=[
            CharacterRecord(character_id="bob", name="Bob"),
            _character("pip", "Pip", CharacterAgentTier.standard),
            _character("alice", "Alice", CharacterAgentTier.standard),
        ],
    )
    cases = (
        ("Pip says Alice waits at the gate.", {"pip"}),
        ("Alice reports Pip waits at the gate.", {"alice"}),
        ("Bob reports Pip waits and Alice stands guard.", set()),
        ("A photograph shows Pip standing by the gate.", set()),
        ("A sketch shows Pip standing by the gate.", set()),
        ("A sketch shows Pip standing and Alice kneeling.", set()),
        ("Pip steps into the room while the radio crackles.", {"pip"}),
        ("Pip steps into the room and speaks over the radio.", {"pip"}),
        ("Pip speaks over the radio and steps into the room.", {"pip"}),
        ("Pip speaks over the radio and stands beside Alice.", {"pip", "alice"}),
        ("Alice says hello and Pip enters the room.", {"alice", "pip"}),
        ("Alice asks a question and Pip enters the room.", {"alice", "pip"}),
        ("Alice asks whether Pip waits and Pip enters the room.", {"alice"}),
        ("Pip reports in a message.", set()),
        ("Pip reports in writing.", set()),
        ("Pip reports by letter.", set()),
        ("Pip reports via text.", set()),
        ("Pip speaks over the radio and eventually enters the room.", set()),
        ("Tomorrow, Pip speaks over the radio and enters the room.", set()),
        ("Pip arrives and Alice enters the room.", {"pip", "alice"}),
        ("Pip and Alice step into the room.", {"pip", "alice"}),
        ("Pip stands beside Alice.", {"pip", "alice"}),
    )

    for text, expected_ids in cases:
        event = _event(text)
        agent_plan = plan_event_visual_introductions(
            ckpt,
            viewer_id="bob",
            event=event,
            observation_level="direct",
            max_loadouts=3,
        )
        narrator_plan = plan_render_visual_introductions(
            ckpt,
            viewer_id="bob",
            resolved=[(
                RenderBufferEntry(
                    event_id=event.event_id,
                    observation_level="direct",
                ),
                event,
            )],
            max_loadouts=3,
        )

        assert {
            introduction.character_id for introduction in agent_plan.loadouts
        } == expected_ids
        assert {
            introduction.character_id
            for introduction in narrator_plan.loadouts
        } == expected_ids


def test_remote_references_preserve_later_physical_introduction():
    for consumer in ("agent", "narrator"):
        for remote_text, expected_remote_ids, meeting_id, meeting_text in (
            (
                "Alice reports Pip waits at the gate.",
                {"alice"},
                "pip",
                "Pip steps into the gatehouse.",
            ),
            (
                "A sketch shows Pip standing by the gate.",
                set(),
                "pip",
                "Pip steps into the gatehouse.",
            ),
            (
                "Pip reports in a message.",
                set(),
                "pip",
                "Pip steps into the gatehouse.",
            ),
            (
                "Pip reports in writing.",
                set(),
                "pip",
                "Pip steps into the gatehouse.",
            ),
            (
                "Pip reports by letter.",
                set(),
                "pip",
                "Pip steps into the gatehouse.",
            ),
            (
                "Pip reports via text.",
                set(),
                "pip",
                "Pip steps into the gatehouse.",
            ),
            (
                "Pip speaks over the radio and eventually enters the room.",
                set(),
                "pip",
                "Pip steps into the gatehouse.",
            ),
            (
                "Tomorrow, Pip speaks over the radio and enters the room.",
                set(),
                "pip",
                "Pip steps into the gatehouse.",
            ),
            (
                "Alice asks whether Pip waits and Korva enters the room.",
                {"alice"},
                "korva",
                "Korva steps into the gatehouse.",
            ),
        ):
            ckpt = CheckpointFile(
                session=SessionState(session_id="s"),
                characters=[
                    CharacterRecord(character_id="bob", name="Bob"),
                    _character("alice", "Alice", CharacterAgentTier.standard),
                    _character("pip", "Pip", CharacterAgentTier.standard),
                    _character("korva", "Korva", CharacterAgentTier.standard),
                ],
            )
            remote_event = _event(remote_text)
            if consumer == "agent":
                remote = plan_event_visual_introductions(
                    ckpt,
                    viewer_id="bob",
                    event=remote_event,
                    observation_level="direct",
                    max_loadouts=3,
                )
            else:
                remote = plan_render_visual_introductions(
                    ckpt,
                    viewer_id="bob",
                    resolved=[(
                        RenderBufferEntry(
                            event_id=remote_event.event_id,
                            observation_level="direct",
                        ),
                        remote_event,
                    )],
                    max_loadouts=3,
                )
            mark_visual_introductions(
                ckpt,
                "bob",
                remote.mark_character_ids,
            )

            assert set(remote.mark_character_ids) == expected_remote_ids
            assert meeting_id not in remote.mark_character_ids

            meeting_event = _event(meeting_text)
            if consumer == "agent":
                meeting = plan_event_visual_introductions(
                    ckpt,
                    viewer_id="bob",
                    event=meeting_event,
                    observation_level="direct",
                    max_loadouts=3,
                )
            else:
                meeting = plan_render_visual_introductions(
                    ckpt,
                    viewer_id="bob",
                    resolved=[(
                        RenderBufferEntry(
                            event_id=meeting_event.event_id,
                            observation_level="direct",
                        ),
                        meeting_event,
                    )],
                    max_loadouts=3,
                )

            assert [intro.character_id for intro in meeting.loadouts] == [meeting_id]
            assert meeting.mark_character_ids == [meeting_id]


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
        "A voice from Pip says he will arrive later.",
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
    assert visually_staged_character_ids(
        ckpt,
        ["Alice points toward Pip's empty chair."],
    ) == {"alice"}
