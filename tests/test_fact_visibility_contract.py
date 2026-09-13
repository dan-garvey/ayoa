"""Explicit fact recipients are the one disclosure boundary for every reader."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.engine.event_runtime import commit_event_batch
from app.engine.image_director import build_projection_groups
from app.engine.narrator import _format_visible_events_block, _visual_novel_sprite_roster
from app.engine.router_batch import materialize_router_batch
from app.engine.story_coordinator import (
    advance_story,
    immutable_checkpoint,
    player_input,
    prepare_autonomous_contest_resolutions,
)
from app.engine.story_dispatcher import StoryDispatcher, append_router_history
from app.engine.router_prompt_projection import router_prompt_history
from app.engine.visual_context import plan_render_visual_introductions
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import RouterBatchOutput
from app.schemas.events import ObservableFact
from tests.support.factories import canonical_event, character_record, checkpoint, router_event_draft
from tests.test_story_coordinator import FakeDispatcher


def test_whisper_split_survives_materialization_fanout_narration_and_reload():
    guests = [f"guest_{i}" for i in range(6)]
    ids = ["rowan", "caelindra", *guests]
    ckpt = checkpoint(
        characters=[character_record(cid) for cid in ids],
        bindings={"rowan": "1", "caelindra": "2", "guest_0": "3"},
    )
    gesture = "Rowan leans close and whispers to Caelindra."
    invitation = "Meet me in the west garden at midnight. Password: silver-fern."
    output = RouterBatchOutput(events=[router_event_draft(facts=[
        ObservableFact.only(gesture, ids, visual_subject_ids=["rowan", "caelindra"]),
        ObservableFact.only(invitation, ["rowan", "caelindra"]),
    ])], next_turns=[])
    batch = materialize_router_batch(
        checkpoint=ckpt,
        inputs=[player_input(ckpt, character_id="rowan", payload="Whisper an invitation.").envelope],
        output=output,
    )
    commit_event_batch(ckpt, [batch.events[0].record])
    saved = immutable_checkpoint(ckpt)
    event = saved.canonical_events[0]
    assert event.observer_ids == ids
    for job in saved.session.narrator_render_jobs:
        rendered = _format_visible_events_block(
            [(job.event_refs[0], event)], pov_character_id=job.pov_character_id,
        )
        assert gesture in rendered
        assert (invitation in rendered) == (job.pov_character_id != "guest_0")
    for guest in saved.characters[3:]:
        observed = "\n".join(guest.pending_observations)
        assert gesture in observed
        assert invitation not in observed


@pytest.mark.asyncio
async def test_autonomous_contest_draft_does_not_receive_other_observers_private_fact():
    ckpt = checkpoint(
        characters=[character_record("alice"), character_record("bob")],
        bindings={"alice": "1"},
    )
    attempt = "Alice reaches for Bob's hand."
    secret = "Only Alice sees the guard leave through the hidden door."
    output = RouterBatchOutput(events=[router_event_draft(
        facts=[ObservableFact.only(attempt, ["alice", "bob"]), ObservableFact.only(secret, ["alice"])],
        required_responders=["bob"],
    )], next_turns=[])
    dispatcher = FakeDispatcher([output])
    await advance_story(ckpt, dispatcher, [player_input(ckpt, character_id="alice", payload="Take his hand.")])
    assert secret in ckpt.session.open_cat_ii_events[0].opening_observable_facts
    await prepare_autonomous_contest_resolutions(immutable_checkpoint(ckpt), dispatcher)
    assert dispatcher.draft_contexts["bob"] == attempt


def test_audio_fact_does_not_grant_visuals_even_when_it_names_the_actor():
    ckpt = checkpoint(
        characters=[
            character_record("alice"), character_record("bob"),
            character_record("pip", visuals={"default_loadout": "SECRET red coat"}),
        ],
        bindings={"alice": "1", "bob": "2"},
    )
    event = canonical_event(facts=[
        ObservableFact.only("Pip speaks beside Alice.", ["alice"], visual_subject_ids=["pip"]),
        ObservableFact.only("Bob hears Pip say hello through the radio.", ["bob"]),
    ])
    commit_event_batch(ckpt, [event])
    by_pov = {job.pov_character_id: job for job in ckpt.session.narrator_render_jobs}
    for pov, job in by_pov.items():
        resolved = [(job.event_refs[0], event)]
        assert _visual_novel_sprite_roster(ckpt, viewer_id=pov, resolved=resolved) == (("Pip",) if pov == "alice" else ())
        intros = plan_render_visual_introductions(ckpt, viewer_id=pov, resolved=resolved)
        assert [item.character_id for item in intros.loadouts] == (["pip"] if pov == "alice" else [])
    projections = build_projection_groups(checkpoint=ckpt, event=event, event_sequence=0,
        transaction_id="tx", source_turn_index=1, actor_id="pip")
    by_viewer = {cid: projection for projection in projections for cid in projection.viewer_character_ids}
    assert [item.character_id for item in by_viewer["alice"].characters] == ["pip"]
    assert by_viewer["bob"].characters == ()
    assert "SECRET" not in str(by_viewer["bob"])


def test_router_history_stores_references_and_projects_fiction_in_original_order():
    ckpt = checkpoint(characters=[character_record("alice")])
    first = canonical_event(event_id="first", facts=["Unique first fact."])
    second = canonical_event(event_id="second", facts=["Unique second fact."])
    ckpt.canonical_events.append(first)
    append_router_history(ckpt, [first])
    ckpt.session_conversation.append(ConversationMessage(role="assistant", content="content_known external arrival"))
    ckpt.canonical_events.append(second)
    append_router_history(ckpt, [second])
    restored = immutable_checkpoint(ckpt)
    assert [item.content for item in restored.session_conversation] == [
        "prior_event sequence=0", "content_known external arrival", "prior_event sequence=1",
    ]
    projected = router_prompt_history(restored)
    assert "Unique first fact." in projected[0].content
    assert projected[1].content == "content_known external arrival"
    assert "Unique second fact." in projected[2].content
    assert restored.model_dump_json().count("Unique first fact.") == 1
    for field in ("source_submission_ids", "feasible_submission_ids", "infeasible_submission_ids", "visibility_log", "observers"):
        assert field not in restored.model_dump_json()


@pytest.mark.asyncio
async def test_appearance_harvest_adds_detail_only_to_visual_recipients():
    ckpt = checkpoint(characters=[character_record(cid) for cid in ["alice", "bob", "pip"]])
    output = RouterBatchOutput(events=[router_event_draft(facts=[
        ObservableFact.only("Pip waves to Alice.", ["alice"], visual_subject_ids=["pip"]),
        ObservableFact.only("Bob hears Pip's hello on the radio.", ["bob"]),
    ], appearance_target_ids=["pip"])], next_turns=[])
    batch = materialize_router_batch(checkpoint=ckpt,
        inputs=[player_input(ckpt, character_id="pip", payload="Say hello.").envelope], output=output)
    agent = SimpleNamespace(
        draft_perception=AsyncMock(return_value=SimpleNamespace(output=SimpleNamespace(public_text="A bright red coat."))),
        commit_perception=MagicMock(),
    )
    await StoryDispatcher._prepare_appearance_harvests(SimpleNamespace(character_agent=agent), ckpt, batch)
    appearance = batch.events[0].record.observable_facts[-1]
    assert appearance.visible_to == ["alice"]
    assert appearance.visual_subject_ids == ["pip"]
    assert "bright red coat" in appearance.text
    assert agent.commit_perception.call_count == 1


def test_appearance_harvest_requires_some_visual_recipient():
    with pytest.raises(ValidationError, match="appearance targets require visual recipients"):
        router_event_draft(facts=[ObservableFact.only("Bob hears Pip on the radio.", ["bob"])],
            appearance_target_ids=["pip"])
