import copy

import pytest
from run import verify_fork, wire_text


def test_external_advice_is_distinct_from_action_and_sent_only_when_supplied():
    text = wire_text("I sit down.", "Private suggestion.")
    assert (
        text.index("Private suggestion.")
        < text.index("Return the next passage")
        < text.index("I sit down.")
    )
    assert text.count("<backstage_guidance>") == 1
    following = wire_text("I leave.")
    assert (
        "Private suggestion." not in following and "backstage_guidance" not in following
    )


def test_fork_rejects_wrong_prefix_text_or_later_turns():
    checkpoint = {
        "native_turns": [
            {"turn_id": "cut", "output": "Published.", "wire_text": "Action."}
        ]
    }
    result = {
        "model": "gpt-5.6-sol",
        "reasoningEffort": "max",
        "instructionSources": [],
        "thread": {
            "turns": [
                {
                    "id": "cut",
                    "items": [
                        {
                            "type": "userMessage",
                            "content": [{"type": "text", "text": "Action."}],
                        },
                        {
                            "type": "agentMessage",
                            "text": "Published.",
                            "phase": "final_answer",
                        },
                    ],
                }
            ]
        },
    }
    verify_fork(result, checkpoint)
    later = copy.deepcopy(result)
    later["thread"]["turns"].append({"id": "future", "items": []})
    with pytest.raises(AssertionError):
        verify_fork(later, checkpoint)
    altered = copy.deepcopy(result)
    altered["thread"]["turns"][0]["items"][1]["text"] = "Unpublished alternative."
    with pytest.raises(AssertionError):
        verify_fork(altered, checkpoint)
