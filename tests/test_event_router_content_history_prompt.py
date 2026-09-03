import re

from app.engine.prompt_manager import PromptManager
from app.schemas.conversation import ConversationMessage


_CONTRACT_RE = re.compile(
    r"<assistant_history_records>(.*?)</assistant_history_records>",
    re.DOTALL,
)


def _render_router_conversation(
    *,
    history: list[ConversationMessage] | None = None,
    router_input_block: str = "I wait.",
) -> list[dict]:
    mgr = PromptManager(prompts_dir="app/prompts")
    return mgr.render_conversation(
        "event_router",
        history=history or [],
        setting_summary="Genre: gothic fantasy",
        world_lore="The village knows the old road.",
        world_rules="Ordinary doors open when unbarred.",
        hidden_lore="The cellar has a hidden exit.",
        hidden_facts="- The baron spies through the servant.",
        acting_character_id="alice",
        router_ruleset_addon=mgr.render(
            "event_router_ruleset_default",
        ).strip(),
        router_output_schema_addon="",
        router_input_block=router_input_block,
    )


def _contract_section(system_content: str) -> str:
    match = _CONTRACT_RE.search(system_content)
    assert match is not None
    return match.group(1)


def test_event_router_defines_content_history_record_contract():
    messages = _render_router_conversation()

    contract = _contract_section(messages[0]["content"])

    for record_kind in (
        "content_known",
        "location_card",
        "front_signal",
        "turn_hint",
    ):
        assert record_kind in contract
    assert "visibility=hidden" in contract
    assert "scope=router" in contract
    assert "analysis/adjudication context only" in contract
    assert "canonical_event.observable_facts" in contract
    assert "visible consequence" in contract
    assert "existing visibility rules" in contract
    assert "non-binding attention hint" in contract
    assert "spawn authority" in contract
    assert "Never route or spawn" in contract


def test_content_history_records_stay_assistant_side_not_current_turn_packet():
    hidden_location = (
        'location_card ref=crypt/entry exits=["stairs"] '
        'hazards=["blade trap"] clues=["raven seal"] visibility=hidden '
        'hash=hash-loc pack=strahd summary="Trapdoor under the altar."'
    )
    hidden_front = (
        'front_signal ref=front/strahd actor=strahd knows=["ireena"] '
        'pressure="summon wolves" visibility=hidden hash=hash-front '
        'pack=strahd summary="Secret hunt plan."'
    )
    messages = _render_router_conversation(
        history=[
            ConversationMessage(role="assistant", content=hidden_location),
            ConversationMessage(role="assistant", content=hidden_front),
        ],
        router_input_block="I inspect the altar.",
    )

    assert messages[1] == {"role": "assistant", "content": hidden_location}
    assert messages[2] == {"role": "assistant", "content": hidden_front}

    current_user = messages[-1]["content"]
    assert "I inspect the altar." in current_user
    assert "location_card ref=crypt/entry" not in current_user
    assert "front_signal ref=front/strahd" not in current_user
    assert "Trapdoor under the altar" not in current_user
    assert "summon wolves" not in current_user


def test_turn_hint_records_stay_assistant_side_not_current_turn_packet():
    turn_hint = (
        "turn_hint scope=attention_hint character=strahd priority=high "
        'refs=strahd:front/hunt facts=f03 reason="May react."'
    )
    messages = _render_router_conversation(
        history=[ConversationMessage(role="assistant", content=turn_hint)],
        router_input_block="I leave town before dawn.",
    )

    assert messages[1] == {"role": "assistant", "content": turn_hint}

    current_user = messages[-1]["content"]
    assert "I leave town before dawn." in current_user
    assert "turn_hint" not in current_user
    assert "front/hunt" not in current_user


def test_event_router_without_content_history_has_no_current_turn_content_packet():
    messages = _render_router_conversation(
        history=[],
        router_input_block="I open the north door.",
    )

    current_user = messages[-1]["content"]
    assert "I open the north door." in current_user
    assert "content_known ref=" not in current_user
    assert "location_card ref=" not in current_user
    assert "front_signal ref=" not in current_user
