from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.engine.checkpoint_manager import CheckpointManager
from app.engine.delivery_outbox import claim_deliveries
from app.engine.delivery_response import response_from_deliveries
from app.engine.event_runtime import commit_event_batch
from app.engine.narrator import compose_pov_render
from app.engine.narrator_delivery import process_narrator_lanes
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.events import ObservableFact
from app.schemas.narrator import (
    NarratorFinalOutput,
    VisualNovelBeatPages,
    VisualNovelNarratorOutput,
    VisualNovelPage,
)
from tests.support.factories import (
    canonical_event,
    character_record,
    checkpoint,
    llm_response,
    narrator_event_ref,
)


@pytest.fixture
def stored_sample():
    return json.loads(
        (
            Path(__file__).parent / "fixtures/narrator_order_and_controls.json"
        ).read_text()
    )


def _narrator_dispatcher(text: str, *, visual_novel: bool = False):
    client = MagicMock(spec=LLMClient)

    async def complete(**_kwargs):
        if visual_novel:
            parsed = VisualNovelNarratorOutput(
                handoff="render",
                handoff_reason="Visible result is ready.",
                beats=[
                    VisualNovelBeatPages(
                        pages=[
                            VisualNovelPage(kind="narration", text=text),
                        ]
                    )
                ],
            )
        else:
            parsed = NarratorFinalOutput(
                handoff="render",
                handoff_reason="Visible result is ready.",
                final_text=text,
            )
        return llm_response(parsed)

    client.complete = AsyncMock(side_effect=complete)

    async def compose(**kwargs):
        return await compose_pov_render(
            client=client,
            prompt_mgr=PromptManager("app/prompts"),
            pov_character_id=kwargs.pop("character_id"),
            buffered_events=kwargs.pop("event_refs"),
            **kwargs,
        )

    return SimpleNamespace(narrator_compose=compose), client


@pytest.mark.asyncio
@pytest.mark.parametrize("pov_id", ["the_master", "davan"])
async def test_stored_kills_keep_source_ties_and_chronological_pov_batch(
    stored_sample,
    pov_id,
):
    ckpt = checkpoint(
        characters=[
            character_record("the_master", name="Master"),
            character_record("davan", name="Davan"),
        ]
    )
    refs = []
    expected_facts = []
    for sequence, source in enumerate(stored_sample["events"]):
        facts = [
            ObservableFact.model_validate(fact) for fact in source["observable_facts"]
        ]
        event = canonical_event(
            event_id=source["event_id"],
            effective_at_s=source["effective_at_s"],
            duration_s=source["duration_s"],
            observer_ids=["the_master", "davan"],
            facts=facts,
        )
        ckpt.canonical_events.append(event)
        refs.append(
            narrator_event_ref(
                event_id=event.event_id,
                event_sequence=sequence,
                visible_at_s=event.effective_at_s,
            )
        )
        expected_facts.extend(
            fact.text
            for fact in sorted(facts, key=lambda fact: fact.at_offset_s)
            if fact.is_visible_to(pov_id)
        )
    dispatcher, client = _narrator_dispatcher("The result is visible.")
    await dispatcher.narrator_compose(
        ckpt=ckpt,
        character_id=pov_id,
        event_refs=list(reversed(refs)),
        partial_mode=False,
        user_input="",
        handoff_policy="forced",
        handoff_context="The motion has ended.",
        narration_mode="event_aligned",
    )
    rendered_input = client.complete.call_args.kwargs["messages"][-1]["content"]
    positions = [rendered_input.index(text) for text in expected_facts]
    assert positions == sorted(positions)
    for source in stored_sample["events"]:
        for fact in source["observable_facts"]:
            if fact["audience"] == "only" and pov_id not in fact["visible_to"]:
                assert fact["text"] not in rendered_input


@pytest.mark.asyncio
@pytest.mark.parametrize("visual_novel", [False, True])
async def test_repaired_prose_is_identical_in_history_outbox_and_frontend_reload(
    stored_sample,
    tmp_path,
    visual_novel,
):
    raw = stored_sample["corrupted_prose"]
    assert sum(ord(char) < 32 and char not in "\t\r\n" for char in raw) == 8
    expected = raw.replace("\x19", "’").replace("\x1c", "“").replace("\x1d", "”")
    raw += "\n\nA note names /private/plot.txt."
    expected += "\n\nA note names [redacted private module material]."
    ckpt = checkpoint(
        bindings={"the_master": "1"},
        characters=[character_record("the_master", name="Master")],
    )
    if visual_novel:
        ckpt.session.config.settings.presentation_mode = "visual_novel"
    event = canonical_event(observer_ids=["the_master"])
    commit_event_batch(ckpt, [event])
    dispatcher, _client = _narrator_dispatcher(raw, visual_novel=visual_novel)
    outcome = await process_narrator_lanes(
        ckpt, dispatcher, lane_ids=[event.causal_lane_id]
    )
    assert outcome[0].rendered_pov_ids == ["the_master"]
    history = json.loads(
        ckpt.narrator_conversations["the_master"][-1].content[0]["text"]
    )
    assert (
        history["pages"][0]["text"] if visual_novel else history["final_text"]
    ) == expected
    assert ckpt.session.delivery_outbox[0].payload.prose == expected
    manager = CheckpointManager(str(tmp_path / "sessions"))
    checkpoint_id = manager.save(ckpt)
    for consumer in ("cli", "discord"):
        restored = manager.load_latest(ckpt.session.session_id)
        claims = claim_deliveries(
            restored, pov_character_id="the_master", consumer_id=consumer
        )
        response = response_from_deliveries(
            session_id=restored.session.session_id,
            checkpoint_id=checkpoint_id,
            turn_index=restored.session.turn_index,
            acting_character_id="the_master",
            deliveries=claims,
        )
        assert response.output_text.encode() == expected.encode()
        assert response.per_player_renders["the_master"] == expected
        assert response.deliveries[0].payload.prose == expected
        if visual_novel:
            render = response.per_player_visual_novel_renders["the_master"]
            assert render.segments[0].pages[0].text == expected


@pytest.mark.parametrize(
    "control",
    [chr(code) for code in [*range(32), 127] if code not in {9, 10, 13, 25, 28, 29}],
)
@pytest.mark.parametrize("visual_novel", [False, True])
def test_unknown_controls_fail_before_whitespace_stripping(control, visual_novel):
    with pytest.raises(ValidationError, match="forbidden control"):
        if visual_novel:
            VisualNovelPage(kind="narration", text=control + "Visible." + control)
        else:
            NarratorFinalOutput(
                handoff="render",
                handoff_reason="Ready.",
                final_text=control + "Visible." + control,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("visual_novel", [False, True])
async def test_unsafe_render_preserves_pending_work_without_committing_prose(
    tmp_path,
    visual_novel,
):
    ckpt = checkpoint(bindings={"alice": "1"}, characters=[character_record("alice")])
    if visual_novel:
        ckpt.session.config.settings.presentation_mode = "visual_novel"
    event = canonical_event()
    commit_event_batch(ckpt, [event])
    dispatcher, _client = _narrator_dispatcher(
        "Never commit\x00this.", visual_novel=visual_novel
    )
    result = await process_narrator_lanes(
        ckpt, dispatcher, lane_ids=[event.causal_lane_id]
    )
    assert result[0].failed_pov_ids == ["alice"]
    assert not ckpt.narrator_conversations
    assert not ckpt.session.delivery_outbox
    assert not ckpt.session.visual_introductions
    manager = CheckpointManager(str(tmp_path / "sessions"))
    manager.save(ckpt)
    restored = manager.load_latest(ckpt.session.session_id)
    assert restored.session.narrator_render_jobs[0].status == "failed"
    assert len(restored.canonical_events) == 1
    assert not restored.narrator_conversations
    assert not restored.session.delivery_outbox
    dispatcher, _client = _narrator_dispatcher(
        "Safe first paragraph.\n\nSafe second paragraph.", visual_novel=visual_novel
    )
    result = await process_narrator_lanes(
        restored, dispatcher, lane_ids=[event.causal_lane_id]
    )
    assert result[0].rendered_pov_ids == ["alice"]
    assert len(restored.canonical_events) == len(restored.session.delivery_outbox) == 1
    assert (
        restored.session.delivery_outbox[0].payload.prose
        == "Safe first paragraph.\n\nSafe second paragraph."
    )
