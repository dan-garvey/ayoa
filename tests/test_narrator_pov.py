"""Tests for the per-POV narrator entry point (`compose_pov_render`).

Exercises the new function against a mocked LLMClient so we can verify:
- Buffered events resolve against ckpt.canonical_events by event_id and
  the resolved prose is returned unchanged from the LLM parsed output.
- Per-POV rolling history stores assistant messages only.
- partial_mode puts the stop-before-resolution instruction in the user payload.
- A buffer entry missing from canonical_events fails before an incomplete
  player-visible sequence can be rendered or flushed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.narrator import commit_pov_render, compose_pov_render
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import PARTIAL_MODE_MARKER
from app.llm.client import LLMClient
from app.schemas.content_privacy import REDACTED_IMPORT_SENTINEL
from app.schemas.characters import (
    CharacterDescriptions,
    CharacterRecord,
    CharacterStatus,
    CharacterVisuals,
    PublicSheet,
)
from app.schemas.event_router import EventRouterOutput
from app.schemas.events import ObservableFact
from app.schemas.narrator import (
    NarratorFinalOutput,
    VisualNovelNarratorOutput,
    VisualNovelPage,
)
from app.schemas.state import (
    RenderBufferEntry,
    SessionConfig,
    StorySetting,
    WorldState,
)
from tests.support.factories import (
    character_record,
    checkpoint,
    narrator_llm_response,
    llm_response,
    router_output,
)


# ---- helpers --------------------------------------------------------------


def _ckpt():
    """Minimal checkpoint with two canonical events and a
    single human-bound character."""
    ckpt = checkpoint(
        bindings={"alice": "1"},
        player_character_id="alice",
        world_state=WorldState(
            setting=StorySetting(genre="fantasy", tone="quiet"),
        ),
        characters=[
            character_record(
                "alice",
                name="Alice",
                public_sheet=PublicSheet(role="player", appearance="dark-haired"),
                is_playable=True,
            ),
            character_record("pip", name="Pip", role="npc"),
        ],
    )
    ckpt.session.config = SessionConfig(narrative_rules="Concise prose.")
    # Seed two canonical events into the log.
    ev1 = _router_event(
        "evt_alpha",
        [ObservableFact.all("The arch is weathered.")],
    )
    ev2 = _router_event(
        "evt_beta",
        [ObservableFact.all("Pip nods.")],
    )
    ckpt.canonical_events.extend([ev1, ev2])
    return ckpt


def _llm_response(final_text: str = "RENDERED"):
    """Minimal LLMResponse that can pass through
    `serialize_assistant_content` when the narrator appends history."""
    return narrator_llm_response(final_text)


def _router_event(
    event_id: str,
    facts: list[ObservableFact],
    *,
    observers: list[str] | None = None,
    duration_s: int = 0,
) -> EventRouterOutput:
    event = router_output(
        event_id=event_id,
        duration_s=duration_s,
        facts=facts,
        event_kind="cascade_exhausted",
        observer_ids=observers or ["alice"],
    )
    event.commitment_open = {
        "present": False,
        "actor_ids": [],
        "description": "",
        "expected_duration_s": 0,
        "max_duration_s": 0,
        "location_label": "",
    }
    event.commitment_resolutions = []
    event.commitment_interrupts = []
    return event


@pytest.fixture
def prompt_manager() -> PromptManager:
    return PromptManager("app/prompts")


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=_llm_response("RENDERED"))
    return client


# ---- tests ----------------------------------------------------------------


class TestComposePovRender:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("presentation_mode", ["prose", "visual_novel"])
    async def test_forced_handoff_rejects_continue_without_composition_mutation(
        self,
        mock_client,
        prompt_manager,
        presentation_mode,
    ):
        ckpt = _ckpt()
        ckpt.session.config.settings.presentation_mode = presentation_mode
        next(c for c in ckpt.characters if c.character_id == "pip").visuals = (
            CharacterVisuals(default_loadout="Patched red coat.")
        )
        result = (
            VisualNovelNarratorOutput(
                handoff="continue",
                handoff_reason="Motion continues.",
                pages=[],
            )
            if presentation_mode == "visual_novel"
            else NarratorFinalOutput(
                handoff="continue",
                handoff_reason="Motion continues.",
                final_text="",
            )
        )
        mock_client.complete = AsyncMock(return_value=llm_response(result))

        with pytest.raises(ValueError, match="forced handoff policy"):
            await compose_pov_render(
                client=mock_client,
                prompt_mgr=prompt_manager,
                ckpt=ckpt,
                pov_character_id="alice",
                buffered_events=[
                    RenderBufferEntry(
                        event_id="evt_beta",
                        observation_level="direct",
                    )
                ],
                partial_mode=False,
                handoff_policy="forced",
            )

        assert mock_client.complete.await_count == 1
        assert ckpt.narrator_conversations == {}
        assert ckpt.session.visual_introductions == {}

    @pytest.mark.asyncio
    async def test_visual_novel_mode_uses_structured_pages_and_plain_history(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        ckpt.session.config.settings.presentation_mode = "visual_novel"
        mock_client.complete = AsyncMock(return_value=llm_response(
            VisualNovelNarratorOutput(
                handoff="render",
                handoff_reason="Pip's question returns control.",
                pages=[
                    VisualNovelPage(
                        kind="narration",
                        text="Rain beads on the weathered arch.",
                    ),
                    VisualNovelPage(
                        kind="dialogue",
                        speaker="Pip",
                        text="Are you coming?",
                    ),
                ],
            )
        ))
        buffered = [
            RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
        ]

        result, entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=False,
            user_input="I wait under the arch.",
        )

        assert isinstance(result, VisualNovelNarratorOutput)
        assert entry.assistant == (
            "Rain beads on the weathered arch.\n\nPip: Are you coming?"
        )
        call = mock_client.complete.await_args.kwargs
        assert call["response_model"] is VisualNovelNarratorOutput
        assert "I wait under the arch." not in call["messages"][0]["content"]
        assert "I wait under the arch." in call["messages"][-1]["content"]
        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            result=result,
            user_input=entry.user,
        )
        stored = json.loads(
            ckpt.narrator_conversations["alice"][-1].content[0]["text"]
        )
        assert "final_text" not in stored
        assert stored["pages"][1] == {
            "kind": "dialogue",
            "speaker": "Pip",
            "text": "Are you coming?",
        }

    @pytest.mark.asyncio
    async def test_visual_novel_source_identifier_gets_one_transient_correction(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        ckpt.session.config.settings.presentation_mode = "visual_novel"
        unsafe = VisualNovelNarratorOutput(
            handoff="render",
            handoff_reason="The reply returns control.",
            pages=[
                VisualNovelPage(
                    kind="dialogue",
                    speaker="pip",
                    text="Are you coming?",
                ),
            ],
        )
        safe = VisualNovelNarratorOutput(
            handoff="render",
            handoff_reason="The reply returns control.",
            pages=[
                VisualNovelPage(
                    kind="dialogue",
                    speaker="the small courier",
                    text="Are you coming?",
                ),
            ],
        )
        mock_client.complete = AsyncMock(
            side_effect=[llm_response(unsafe), llm_response(safe)]
        )
        buffered = [
            RenderBufferEntry(event_id="evt_beta", observation_level="direct"),
        ]

        result, entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=False,
        )

        assert mock_client.complete.await_count == 2
        correction_messages = mock_client.complete.await_args_list[1].kwargs[
            "messages"
        ]
        assert correction_messages[-2] == {
            "role": "assistant",
            "content": unsafe.model_dump_json(),
        }
        assert correction_messages[-1]["role"] == "user"
        assert result == safe
        assert entry.assistant == "the small courier: Are you coming?"

        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            result=result,
            user_input=entry.user,
        )
        stored_history = json.dumps(
            [item.model_dump() for item in ckpt.narrator_conversations["alice"]]
        )
        assert '"speaker": "pip"' not in stored_history
        assert "the small courier" in stored_history

    @pytest.mark.asyncio
    async def test_visual_novel_correction_may_change_only_the_unsafe_text_field(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        ckpt.session.config.settings.presentation_mode = "visual_novel"
        unsafe = VisualNovelNarratorOutput(
            handoff="render",
            handoff_reason="The motion is visible.",
            pages=[
                VisualNovelPage(
                    kind="dialogue",
                    speaker="Pip",
                    text="pip waits beneath the arch.",
                ),
            ],
        )
        safe = VisualNovelNarratorOutput(
            handoff="render",
            handoff_reason="A revised diagnostic reason is allowed.",
            pages=[
                VisualNovelPage(
                    kind="dialogue",
                    speaker="Pip",
                    text="The courier waits beneath the arch.",
                ),
            ],
        )
        mock_client.complete = AsyncMock(
            side_effect=[llm_response(unsafe), llm_response(safe)]
        )

        result, _entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(event_id="evt_beta", observation_level="direct")
            ],
            partial_mode=False,
        )

        assert result == safe
        assert mock_client.complete.await_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mutation", "error"),
        [
            ("handoff", "changed the handoff decision"),
            ("page_count", "changed the page count"),
            ("kind_order", "changed page kind/order"),
            ("safe_text", "changed an already-safe text field"),
        ],
    )
    async def test_visual_novel_correction_rejects_semantic_drift(
        self,
        mock_client,
        prompt_manager,
        mutation,
        error,
    ):
        ckpt = _ckpt()
        ckpt.session.config.settings.presentation_mode = "visual_novel"
        unsafe = VisualNovelNarratorOutput(
            handoff="render",
            handoff_reason="The reply returns control.",
            pages=[
                VisualNovelPage(kind="narration", text="Rain marks the arch."),
                VisualNovelPage(kind="dialogue", speaker="pip", text="Ready?"),
            ],
        )
        corrected_pages = [
            VisualNovelPage(kind="narration", text="Rain marks the arch."),
            VisualNovelPage(
                kind="dialogue",
                speaker="the small courier",
                text="Ready?",
            ),
        ]
        corrected_handoff = "render"
        if mutation == "handoff":
            corrected_handoff = "continue"
            corrected_pages = []
        elif mutation == "page_count":
            corrected_pages = corrected_pages[:1]
        elif mutation == "kind_order":
            corrected_pages = list(reversed(corrected_pages))
        elif mutation == "safe_text":
            corrected_pages[0] = VisualNovelPage(
                kind="narration",
                text="Rain now sheets across the arch.",
            )
        corrected = VisualNovelNarratorOutput(
            handoff=corrected_handoff,
            handoff_reason="Correction attempted.",
            pages=corrected_pages,
        )
        mock_client.complete = AsyncMock(
            side_effect=[llm_response(unsafe), llm_response(corrected)]
        )

        with pytest.raises(ValueError, match=error):
            await compose_pov_render(
                client=mock_client,
                prompt_mgr=prompt_manager,
                ckpt=ckpt,
                pov_character_id="alice",
                buffered_events=[
                    RenderBufferEntry(
                        event_id="evt_beta",
                        observation_level="direct",
                    )
                ],
                partial_mode=False,
                handoff_policy="candidate",
            )

        assert mock_client.complete.await_count == 2
        assert ckpt.narrator_conversations == {}
        assert ckpt.session.visual_introductions == {}

    @pytest.mark.asyncio
    async def test_visual_novel_exact_id_check_includes_culled_roster_records(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        ckpt.session.config.settings.presentation_mode = "visual_novel"
        ckpt.characters.append(CharacterRecord(
            character_id="retiredguard",
            name="Old Guard",
            status=CharacterStatus.culled,
        ))
        unsafe = VisualNovelNarratorOutput(
            handoff="render",
            handoff_reason="The reply returns control.",
            pages=[
                VisualNovelPage(kind="dialogue", speaker="retiredguard", text="Halt.")
            ],
        )
        safe = VisualNovelNarratorOutput(
            handoff="render",
            handoff_reason="The reply returns control.",
            pages=[VisualNovelPage(kind="dialogue", speaker="Old Guard", text="Halt.")],
        )
        mock_client.complete = AsyncMock(
            side_effect=[llm_response(unsafe), llm_response(safe)]
        )

        result, _entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(event_id="evt_alpha", observation_level="direct")
            ],
            partial_mode=False,
        )

        assert result == safe
        assert mock_client.complete.await_count == 2

    @pytest.mark.asyncio
    async def test_visual_novel_second_identifier_failure_rolls_back_state(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        ckpt.session.config.settings.presentation_mode = "visual_novel"
        next(c for c in ckpt.characters if c.character_id == "pip").visuals = (
            CharacterVisuals(default_loadout="Patched red coat.")
        )
        first = VisualNovelNarratorOutput(
            handoff="render",
            handoff_reason="The reply returns control.",
            pages=[VisualNovelPage(kind="dialogue", speaker="pip", text="Ready?")],
        )
        second = VisualNovelNarratorOutput(
            handoff="render",
            handoff_reason="The reply returns control.",
            pages=[
                VisualNovelPage(
                    kind="narration",
                    text="off_roster_id waits beneath the arch.",
                )
            ],
        )
        mock_client.complete = AsyncMock(
            side_effect=[llm_response(first), llm_response(second)]
        )

        with pytest.raises(ValueError, match="source identifier"):
            await compose_pov_render(
                client=mock_client,
                prompt_mgr=prompt_manager,
                ckpt=ckpt,
                pov_character_id="alice",
                buffered_events=[
                    RenderBufferEntry(
                        event_id="evt_beta",
                        observation_level="direct",
                    )
                ],
                partial_mode=False,
            )

        assert mock_client.complete.await_count == 2
        assert ckpt.narrator_conversations == {}
        assert ckpt.session.visual_introductions == {}

    def test_visual_novel_commit_reasserts_identifier_safety_before_mutation(
        self,
    ):
        ckpt = _ckpt()
        next(c for c in ckpt.characters if c.character_id == "pip").visuals = (
            CharacterVisuals(default_loadout="Patched red coat.")
        )
        unsafe = VisualNovelNarratorOutput(
            handoff="render",
            handoff_reason="The reply returns control.",
            pages=[VisualNovelPage(kind="dialogue", speaker="pip", text="Ready?")],
        )

        with pytest.raises(ValueError, match="source identifier"):
            commit_pov_render(
                ckpt,
                pov_character_id="alice",
                buffered_events=[
                    RenderBufferEntry(
                        event_id="evt_beta",
                        observation_level="direct",
                    )
                ],
                result=unsafe,
                user_input="",
            )

        assert ckpt.narrator_conversations == {}
        assert ckpt.session.visual_introductions == {}

    @pytest.mark.asyncio
    async def test_basic_render_commits_history_only_when_accepted(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        buffered = [
            RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
            RenderBufferEntry(event_id="evt_beta", observation_level="indirect"),
        ]

        result, entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=False,
            user_input="I look around.",
        )

        # The narrator emits a delivery judgment and prose; the engine builds the entry
        # from the real player input (passed in) and the rendered prose.
        assert isinstance(result, NarratorFinalOutput)
        assert result.final_text == "RENDERED"
        assert entry.user == "I look around."
        assert entry.assistant == "RENDERED"
        assert ckpt.narrator_conversations.get("alice", []) == []
        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            result=result,
            user_input=entry.user,
        )
        # Per-POV history preserves the exact accepted player/narrator turn.
        alice_hist = ckpt.narrator_conversations["alice"]
        assert len(alice_hist) == 2
        assert alice_hist[0].role == "user"
        assert alice_hist[0].content == "I look around."
        assert alice_hist[1].role == "assistant"

        # Visible details made it into the rendered prompt.
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["max_tokens"] == 8000
        flat = "\n".join(
            m["content"] for m in call_kwargs["messages"]
            if isinstance(m.get("content"), str)
        )
        assert "evt_alpha" not in flat
        assert "evt_beta" not in flat
        assert "The arch is weathered" in flat
        assert "Pip nods" in flat
        # Audit/framing fields are dropped from the narrator input.
        assert "Alice sees the arch" not in flat
        assert "Pip dips his chin" not in flat
        user_msg = call_kwargs["messages"][-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], str)
        assert PARTIAL_MODE_MARKER not in user_msg["content"]
        assert user_msg["content"].index("I look around.") < user_msg[
            "content"
        ].index("The arch is weathered")

    @pytest.mark.asyncio
    async def test_accepted_player_submission_is_replayed_on_next_render(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        buffered = [
            RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
        ]
        first_result, first_entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=False,
            user_input="I wait until the bell rings, then open the gate.",
        )
        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            result=first_result,
            user_input=first_entry.user,
        )

        await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=False,
            user_input="I listen for it.",
        )

        messages = mock_client.complete.await_args.kwargs["messages"]
        assert [message["role"] for message in messages] == [
            "system", "user", "assistant", "user",
        ]
        assert messages[1]["content"] == (
            "I wait until the bell rings, then open the gate."
        )
        assert "I listen for it." in messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_dnd_player_species_reaches_narrator_user_context(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        alice = next(char for char in ckpt.characters if char.character_id == "alice")
        alice.mechanics = {
            "ruleset_id": "dnd5e_basic",
            "dnd5e_sheet": {
                "identity": {
                    "species": "Hill Dwarf",
                    "classes": [{"name": "Cleric", "level": 3}],
                },
                "statblock": {},
            },
        }

        result, _entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
            ],
            partial_mode=False,
            user_input="I look around.",
        )

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_text = messages[0]["content"]
        user_text = messages[-1]["content"]
        assert "Hill Dwarf" not in system_text
        assert "Alice (you) — Hill Dwarf; Cleric 3" in user_text

    @pytest.mark.asyncio
    async def test_imported_asset_source_sentinels_do_not_reach_prompt(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        sentinels = [
            "delivery_ref=asset://synthetic/hidden-map",
            "source_ref=raw-row",
            "/private/table/source-map.png",
            "raw_ocr=PROTECTED_SOURCE_EXCERPT",
            "data:image/png;base64,AAAA",
        ]
        ckpt.canonical_events = [
            _router_event(
                "evt_leak",
                [ObservableFact.all("Visible surface. " + " ".join(sentinels))],
            ),
        ]

        await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(
                    event_id="evt_leak",
                    observation_level="direct",
                )
            ],
            partial_mode=False,
            user_input="I look around.",
        )

        mock_client.complete.assert_awaited_once()
        messages = mock_client.complete.await_args.kwargs["messages"]
        flat = "\n".join(
            message["content"]
            for message in messages
            if isinstance(message.get("content"), str)
        )
        for sentinel in sentinels:
            assert sentinel not in flat
        assert REDACTED_IMPORT_SENTINEL in flat
        assert "Visible surface." in flat

    @pytest.mark.asyncio
    async def test_render_strips_unmatched_trailing_brace_from_final_text(
        self, mock_client, prompt_manager,
    ):
        mock_client.complete = AsyncMock(return_value=_llm_response(
            "She says, 'entirely human?'}",
        ))
        ckpt = _ckpt()

        result, entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
            ],
            partial_mode=False,
            user_input="I listen.",
        )

        assert result.final_text == "She says, 'entirely human?'"
        assert entry.assistant == "She says, 'entirely human?'"
        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(
                    event_id="evt_alpha", observation_level="direct",
                ),
            ],
            result=result,
            user_input=entry.user,
        )
        assistant = ckpt.narrator_conversations["alice"][-1]
        assert assistant.role == "assistant"
        assert isinstance(assistant.content, list)
        stored = json.loads(assistant.content[0]["text"])
        assert stored["final_text"] == "She says, 'entirely human?'"

    @pytest.mark.asyncio
    async def test_new_character_context_is_user_tail_only(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        ckpt.session.character_bindings["sora_kageyama"] = "2"
        ckpt.characters.append(CharacterRecord(
            character_id="sora_kageyama",
            name="Sora Kageyama",
            public_sheet=PublicSheet(
                role="Hero of the Realm; private authorial role should not leak",
                appearance=(
                    "Japanese, tall, quick posture. Wears the Crown's "
                    "blue Hero livery over a close-fitting white shirt."
                ),
            ),
            descriptions=CharacterDescriptions(
                public=(
                    "Sora is the cohort's informal leader; his blue "
                    "sun-crest tabard marks Crown Hero livery."
                ),
                private="Sora is also quietly watching the defective summon.",
            ),
            visuals=CharacterVisuals(
                default_loadout=(
                    "LOADOUT ORDER SENTINEL: Blue sun-crest tabard, quick posture."
                ),
            ),
            location="gatehouse",
        ))
        ckpt.canonical_events.append(_router_event(
            "evt_sora",
            [ObservableFact.all(
                "VISIBLE RESULT ORDER SENTINEL: "
                "sora_kageyama adjusts the blue tabard."
            )],
        ))

        result, _entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(
                    event_id="evt_sora", observation_level="direct",
                ),
            ],
            partial_mode=False,
            user_input="PLAYER ATTEMPT ORDER SENTINEL",
        )

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_content = messages[0]["content"]
        user_content = messages[-1]["content"]
        flat = "\n".join(
            m["content"] for m in messages
            if isinstance(m.get("content"), str)
        )

        assert "Sora Kageyama: visible exterior" not in system_content
        assert "blue sun-crest tabard" not in system_content
        assert "quietly watching" not in system_content
        assert "private authorial role" not in system_content
        assert "Crown's blue Hero livery" not in flat

        assert "- Sora Kageyama: visible exterior" in user_content
        assert "cohort's informal leader" not in user_content
        assert "Blue sun-crest tabard" in user_content
        assert "private authorial role" not in user_content
        assert (
            user_content.index("LOADOUT ORDER SENTINEL")
            < user_content.index("PLAYER ATTEMPT ORDER SENTINEL")
            < user_content.index("VISIBLE RESULT ORDER SENTINEL")
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("presentation_mode", ["prose", "visual_novel"])
    async def test_imported_visual_metadata_is_safe_before_provider_input(
        self,
        mock_client,
        prompt_manager,
        presentation_mode,
    ):
        ckpt = _ckpt()
        ckpt.session.config.settings.presentation_mode = presentation_mode
        ckpt.world_state.setting.premise = (
            "SAFE PREMISE SENTINEL. source_path=/private/story/outline.md "
            "actor.hidden /secret.env https://example.com/public/premise.png "
            "\x1b[31m"
        )
        ckpt.session.config.narrative_rules = (
            "SAFE NARRATIVE RULE SENTINEL. "
            "tests/fixtures/private/rules.txt /secret.pem "
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
        beta_event = next(
            event for event in ckpt.canonical_events
            if event.event_id == "evt_beta"
        )
        beta_event.canonical_event.observable_facts = [ObservableFact.all(
            "Pip nods in a scarlet coat. actor.hidden "
            "app/storage/stories/private/pip.png"
        )]
        next(c for c in ckpt.characters if c.character_id == "pip").visuals = (
            CharacterVisuals(default_loadout=(
                "Scarlet coat. source_path=/private/module/source-map.png "
                "private_extractions/page-07.txt actor.hidden "
                r"C:\Users\dan\ayoa\private\pip.png "
                "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef "
                "\x1b[31mbrass clasp\x1b[0m\x07"
            ))
        )
        provider_result = (
            VisualNovelNarratorOutput(
                handoff="render",
                handoff_reason="The arrival is visible.",
                pages=[VisualNovelPage(kind="narration", text="Pip arrives.")],
            )
            if presentation_mode == "visual_novel"
            else NarratorFinalOutput(
                handoff="render",
                handoff_reason="The arrival is visible.",
                final_text="Pip arrives.",
            )
        )
        mock_client.complete = AsyncMock(
            return_value=llm_response(provider_result)
        )

        await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(event_id="evt_beta", observation_level="direct")
            ],
            partial_mode=False,
        )

        messages = mock_client.complete.await_args.kwargs["messages"]
        provider_input = "\n".join(
            message["content"]
            for message in messages
            if isinstance(message.get("content"), str)
        )
        assert "Scarlet coat" in provider_input
        assert "brass clasp" in provider_input
        assert "SAFE PREMISE SENTINEL" in provider_input
        assert "SAFE NARRATIVE RULE SENTINEL" in provider_input
        assert "https://example.com/public/premise.png" in provider_input
        assert REDACTED_IMPORT_SENTINEL in provider_input
        assert "/private/story/outline.md" not in provider_input
        assert "/private/module/source-map.png" not in provider_input
        assert "/secret.env" not in provider_input
        assert "/secret.pem" not in provider_input
        assert "private_extractions" not in provider_input
        assert r"C:\Users\dan\ayoa" not in provider_input
        assert "app/storage/stories" not in provider_input
        assert "actor.hidden" not in provider_input
        assert "0123456789abcdef" not in provider_input
        assert "aaaaaaaaaaaaaaaa" not in provider_input
        assert "\x1b" not in provider_input
        assert "\x07" not in provider_input

    @pytest.mark.asyncio
    @pytest.mark.parametrize("presentation_mode", ["prose", "visual_novel"])
    async def test_remote_reference_preserves_first_meeting_for_accepted_arrival(
        self,
        mock_client,
        prompt_manager,
        presentation_mode,
    ):
        ckpt = _ckpt()
        ckpt.session.config.settings.presentation_mode = presentation_mode
        next(c for c in ckpt.characters if c.character_id == "pip").visuals = (
            CharacterVisuals(default_loadout=(
                "LATER MEETING EXTERIOR SENTINEL: patched red coat."
            ))
        )
        remote_event = _router_event(
            "evt_remote_pip",
            [ObservableFact.all(
                "The radio crackles: 'Pip will arrive later.'"
            )],
        )
        meeting_event = _router_event(
            "evt_meeting_pip",
            [ObservableFact.all("Pip steps into the room.")],
        )
        ckpt.canonical_events.extend([remote_event, meeting_event])
        provider_results = (
            [
                VisualNovelNarratorOutput(
                    handoff="render",
                    handoff_reason="The radio message is complete.",
                    pages=[VisualNovelPage(
                        kind="narration",
                        text="A radio crackles.",
                    )],
                ),
                VisualNovelNarratorOutput(
                    handoff="render",
                    handoff_reason="The arrival is visible.",
                    pages=[VisualNovelPage(
                        kind="narration",
                        text="Pip steps into the room.",
                    )],
                ),
            ]
            if presentation_mode == "visual_novel"
            else [
                NarratorFinalOutput(
                    handoff="render",
                    handoff_reason="The radio message is complete.",
                    final_text="A radio crackles.",
                ),
                NarratorFinalOutput(
                    handoff="render",
                    handoff_reason="The arrival is visible.",
                    final_text="Pip steps into the room.",
                ),
            ]
        )
        mock_client.complete = AsyncMock(side_effect=[
            llm_response(result) for result in provider_results
        ])

        remote_buffer = [RenderBufferEntry(
            event_id=remote_event.event_id,
            observation_level="direct",
        )]
        remote_result, remote_entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=remote_buffer,
            partial_mode=False,
        )
        remote_messages = mock_client.complete.await_args_list[0].kwargs["messages"]
        assert all(
            "LATER MEETING EXTERIOR SENTINEL" not in message["content"]
            for message in remote_messages
            if isinstance(message.get("content"), str)
        )
        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=remote_buffer,
            result=remote_result,
            user_input=remote_entry.user,
        )
        assert ckpt.session.visual_introductions == {}

        meeting_buffer = [RenderBufferEntry(
            event_id=meeting_event.event_id,
            observation_level="direct",
        )]
        meeting_result, meeting_entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=meeting_buffer,
            partial_mode=False,
        )
        meeting_messages = mock_client.complete.await_args_list[1].kwargs["messages"]
        assert "LATER MEETING EXTERIOR SENTINEL" not in (
            meeting_messages[0]["content"]
        )
        assert "LATER MEETING EXTERIOR SENTINEL" in (
            meeting_messages[-1]["content"]
        )
        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=meeting_buffer,
            result=meeting_result,
            user_input=meeting_entry.user,
        )

        assert ckpt.session.visual_introductions == {"alice": ["pip"]}

    @pytest.mark.asyncio
    async def test_public_context_does_not_use_raw_sheet_or_private_description(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        ckpt.characters.append(CharacterRecord(
            character_id="korva_sahl",
            name="Korva Sahl",
            public_sheet=PublicSheet(
                role="quartermaster; privately the demon heir",
                appearance=(
                    "Plain travel leathers. Hidden horns are tucked under "
                    "her hair."
                ),
                faction="Public Guild. Private demonic court.",
            ),
            descriptions=CharacterDescriptions(
                public="Korva is an S-rank Guild adventurer usually found near the contract board.",
                private="Korva is the Demon Lord's daughter with hidden horns.",
            ),
            location="gatehouse",
        ))
        ckpt.canonical_events.append(_router_event(
            "evt_korva",
            [ObservableFact.all("korva_sahl stands near the notice board.")],
        ))

        await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(
                    event_id="evt_korva", observation_level="direct",
                ),
            ],
            partial_mode=False,
        )

        messages = mock_client.complete.call_args.kwargs["messages"]
        flat = "\n".join(
            m["content"] for m in messages
            if isinstance(m.get("content"), str)
        )
        assert "Korva is an S-rank Guild adventurer" not in flat
        assert "quartermaster" not in flat
        assert "privately the demon heir" not in flat
        assert "Plain travel leathers" not in flat
        assert "Hidden horns" not in flat
        assert "demonic court" not in flat
        assert "Demon Lord's daughter" not in flat

    @pytest.mark.asyncio
    async def test_partial_mode_includes_stop_instruction(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        buffered = [
            RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
        ]

        result, _entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=True,
        )
        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            result=result,
            user_input="",
        )

        call_kwargs = mock_client.complete.call_args.kwargs
        messages = call_kwargs["messages"]
        # The stop-before-resolution instruction lives in the volatile
        # user message, not the cached system prefix.
        last = messages[-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], str)
        assert PARTIAL_MODE_MARKER in last["content"]
        assert PARTIAL_MODE_MARKER not in messages[0]["content"]

        # Engine-only partial-mode instructions are not persisted as dialogue.
        alice_hist = ckpt.narrator_conversations["alice"]
        assert len(alice_hist) == 1
        assert alice_hist[0].role == "assistant"
        assert PARTIAL_MODE_MARKER not in json.dumps(alice_hist[0].content)

    @pytest.mark.asyncio
    async def test_no_partial_marker_when_not_partial(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        buffered = [
            RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
        ]

        result, _entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=False,
        )
        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            result=result,
            user_input="",
        )

        call_kwargs = mock_client.complete.call_args.kwargs
        user_msg = call_kwargs["messages"][-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], str)
        assert PARTIAL_MODE_MARKER not in user_msg["content"]
        # With no player submission, only the assistant output is stored.
        alice_hist = ckpt.narrator_conversations["alice"]
        assert len(alice_hist) == 1
        assert alice_hist[0].role == "assistant"
        assert PARTIAL_MODE_MARKER not in json.dumps(alice_hist[0].content)

    @pytest.mark.asyncio
    async def test_first_meeting_loadout_is_user_tail_only_and_marks_pair(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        pip.visuals = CharacterVisuals(
            default_loadout=(
                "PIP LOADOUT SENTINEL: Patched red coat, brass buttons, "
                "ink-dark braid."
            ),
        )

        buffered = [
            RenderBufferEntry(event_id="evt_beta", observation_level="direct"),
        ]
        result, _entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=False,
        )
        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            result=result,
            user_input="",
        )

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_content = messages[0]["content"]
        user_content = messages[-1]["content"]
        assert "Patched red coat" not in system_content
        assert "PIP LOADOUT SENTINEL" not in system_content
        assert "PIP LOADOUT SENTINEL" in user_content
        assert "Pip: visible exterior: PIP LOADOUT SENTINEL" in user_content
        assert ckpt.session.visual_introductions["alice"] == ["pip"]

    @pytest.mark.asyncio
    async def test_harvested_loadout_marks_without_default_duplicate(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        pip.visuals = CharacterVisuals(default_loadout="Default red coat.")
        ckpt.canonical_events.append(_router_event(
            "evt_query_loadout",
            [ObservableFact.all(
                "[loadout — Pip] Pip wears a blue cloak. actor.hidden "
                "app/storage/stories/private/pip.png"
            )],
        ))

        buffered = [
            RenderBufferEntry(
                event_id="evt_query_loadout",
                observation_level="direct",
            ),
        ]
        result, _entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=False,
        )
        commit_pov_render(
            ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            result=result,
            user_input="",
        )

        user_content = mock_client.complete.call_args.kwargs["messages"][-1]["content"]
        assert "Pip wears a blue cloak." in user_content
        assert "actor.hidden" not in user_content
        assert "app/storage/stories" not in user_content
        assert REDACTED_IMPORT_SENTINEL in user_content
        assert "Default red coat" not in user_content
        assert "Newly introduced character context" not in user_content
        assert ckpt.session.visual_introductions["alice"] == ["pip"]

    @pytest.mark.asyncio
    async def test_missing_event_id_fails_before_narrator_call(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        buffered = [
            RenderBufferEntry(event_id="evt_ghost", observation_level="direct"),
            RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
        ]

        with pytest.raises(
            RuntimeError,
            match="missing canonical event.*evt_ghost",
        ):
            await compose_pov_render(
                client=mock_client,
                prompt_mgr=prompt_manager,
                ckpt=ckpt,
                pov_character_id="alice",
                buffered_events=buffered,
                partial_mode=False,
            )

        mock_client.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_structured_result_fails_loudly(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        response = _llm_response()
        response.parsed = None
        mock_client.complete = AsyncMock(return_value=response)

        with pytest.raises(
            RuntimeError,
            match="Narrator returned no structured result",
        ):
            await compose_pov_render(
                client=mock_client,
                prompt_mgr=prompt_manager,
                ckpt=ckpt,
                pov_character_id="alice",
                buffered_events=[
                    RenderBufferEntry(
                        event_id="evt_alpha",
                        observation_level="direct",
                    ),
                ],
                partial_mode=False,
            )


class TestFormatVisibleEventsBlock:
    """The narrator reads only visible surface details from each event.
    Audit/framing fields are not part of the render input."""

    def _resolved(
        self, *, event_id: str, facts: list[ObservableFact],
        level: str = "direct", observers: list[str] | None = None,
        duration_s: int = 0,
    ):
        ev = _router_event(
            event_id,
            list(facts),
            observers=observers or [],
            duration_s=duration_s,
        )
        entry = RenderBufferEntry(
            event_id=event_id,
            observation_level=level,
        )
        return [(entry, ev)]

    def test_facts_surface_audit_fields_do_not(self):
        from app.engine.narrator import _format_visible_events_block

        resolved = self._resolved(
            event_id="evt_x",
            facts=[
                ObservableFact.all(
                    "Seraphel recites: 'The plague that fell on human ground'"
                ),
                ObservableFact.all("her wings draw tight against her back"),
            ],
        )
        out = _format_visible_events_block(resolved)
        assert "Seen directly:" in out
        assert "wings draw tight" in out
        assert "The plague that fell" in out
        # Audit line must NOT appear — that's the whole point.
        assert "resolved_outcome:" not in out
        assert "what she is permitted" not in out
        assert "the strain of speaking" not in out

    def test_empty_facts_renders_none_marker(self):
        from app.engine.narrator import _format_visible_events_block

        resolved = self._resolved(
            event_id="evt_y",
            facts=[],
        )
        out = _format_visible_events_block(resolved)
        assert "Nothing concrete is visible." in out
        assert "(audit-only)" not in out

    def test_loadout_tags_are_removed_before_narrator_sees_them(self):
        from app.engine.narrator import _format_visible_events_block

        resolved = self._resolved(
            event_id="evt_loadout",
            facts=[
                ObservableFact.all("[loadout — Pip] Pip wears a red coat."),
                ObservableFact.all(
                    "[loadout - Vex] Vex keeps a hand on the doorframe."
                ),
            ],
        )
        out = _format_visible_events_block(resolved)
        assert "Pip wears a red coat." in out
        assert "Vex keeps a hand on the doorframe." in out
        assert "[loadout" not in out

    def test_router_ids_render_as_names_for_narrator(self):
        from app.engine.narrator import _format_visible_events_block

        ckpt = _ckpt()
        resolved = self._resolved(
            event_id="evt_ids",
            facts=[ObservableFact.all("alice sets pip's ledger on the table.")],
        )

        out = _format_visible_events_block(resolved, ckpt=ckpt)

        assert "Alice sets Pip's ledger on the table." in out
        assert "alice sets pip" not in out

    def test_scoped_facts_filter_by_pov_before_narrator_sees_them(self):
        from app.engine.narrator import _format_visible_events_block
        resolved = self._resolved(
            event_id="evt_private",
            facts=[
                ObservableFact.only(
                    "Dan's foot touches Ashara's boot under the table.",
                    ["ashara"],
                ),
                ObservableFact.all(
                    "Dan asks Thessaly whether she knows curses.",
                ),
            ],
            observers=["ashara", "aldric"],
        )

        as_ashara = _format_visible_events_block(resolved, "ashara")
        as_aldric = _format_visible_events_block(resolved, "aldric")

        assert "foot touches Ashara's boot" in as_ashara
        assert "knows curses" in as_ashara
        assert "foot touches Ashara's boot" not in as_aldric
        assert "knows curses" in as_aldric

    def test_resolved_buffers_sort_by_visible_time(self):
        from app.engine.narrator import _resolve_buffered_events

        ckpt = _ckpt()
        buffered = [
            RenderBufferEntry(
                event_id="evt_alpha",
                observation_level="direct",
                visible_at_s=20,
                event_sequence=0,
            ),
            RenderBufferEntry(
                event_id="evt_beta",
                observation_level="direct",
                visible_at_s=10,
                event_sequence=1,
            ),
        ]

        resolved = _resolve_buffered_events(ckpt, buffered)

        assert [event.event_id for _, event in resolved] == [
            "evt_beta",
            "evt_alpha",
        ]

    def test_visible_facts_sort_by_fact_time(self):
        from app.engine.narrator import _format_visible_events_block

        resolved = self._resolved(
            event_id="evt_timed",
            facts=[
                ObservableFact.all("Second visible beat.", at_offset_s=5),
                ObservableFact.all("First visible beat.", at_offset_s=1),
            ],
            duration_s=10,
        )

        out = _format_visible_events_block(resolved)

        assert out.index("First visible beat.") < out.index("Second visible beat.")

    def test_distinct_signal_details_reach_narrator_in_order(self):
        from app.engine.narrator import _format_visible_events_block

        resolved = self._resolved(
            event_id="evt_signal",
            facts=[
                ObservableFact.all(
                    "Bob raises his index and middle fingers toward Alice.",
                    at_offset_s=1,
                ),
                ObservableFact.all(
                    "Alice answers with only her middle finger.",
                    at_offset_s=2,
                ),
                ObservableFact.all(
                    "Bob lowers his hand and waits.",
                    at_offset_s=3,
                ),
            ],
            duration_s=3,
        )

        out = _format_visible_events_block(resolved)

        assert out.index("index and middle fingers") < out.index(
            "only her middle finger"
        ) < out.index("lowers his hand and waits")
