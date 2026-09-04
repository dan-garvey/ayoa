"""Rendered contracts for the separate character turn/perception prompts.

These tests deliberately assert placement and forbidden surfaces rather than
freezing the prose of either prompt.  The character engine owns wiring and
parsing; this file only protects the prompt boundary it consumes.
"""

import re
from pathlib import Path

import pytest

from app.engine.prompt_manager import PromptManager


@pytest.fixture
def manager():
    return PromptManager("app/prompts")


def _turn_values(
    *,
    request_packet: str | None = None,
) -> dict[str, str]:
    actor_context = (
        "Other people know you as a tide archivist with ink-stained cuffs.\n\n"
        "You spent seven winters copying names from wreckage.\n\n"
        "You are patient with records and impatient with evasions.\n\n"
        "The harbor bell rings twice for an unclaimed body.\n\n"
        "You want to recover the missing ledger.\n\n"
        "You kept one page instead of filing it."
    )
    catalog = (
        '<presentation_catalog current="neutral">\n'
        "- neutral: standing still\n"
        "</presentation_catalog>"
    )
    packet = request_packet or (
        "<identity>\n"
        "You are Juniper Unique Actor.\n"
        f"{actor_context}\n"
        "</identity>\n\n"
        "<current_input>\n"
        "A unique witness coughed behind the door.\n"
        "The exchange is waiting for your next observable choice.\n"
        f"{catalog}\n"
        "</current_input>"
    )
    return {
        "ruleset_guidance": "",
        "request_packet": packet,
    }


def test_turn_prompt_puts_full_actor_packet_in_user_after_delimiter(manager):
    messages = manager.render_messages("agent_turn", **_turn_values())
    system = messages[0]["content"]
    user = messages[1]["content"]

    for value in (
        "Juniper Unique Actor",
        "You spent seven winters copying names from wreckage.",
        "recover the missing ledger",
        "You kept one page instead of filing it.",
    ):
        assert value not in system
        assert value in user
    assert "unique witness" not in system
    assert "unique witness" in user
    assert re.search(r"\bYou are\b[^\n]*Juniper Unique Actor", user)
    assert "<private_carry>" not in system
    assert '<presentation_catalog current="neutral">' in user
    assert '<presentation_catalog current="neutral">' not in system

    for legacy_marker in (
        "## AGENT-TURN",
        "## PERCEPTION",
        "## Turn Frame",
        "\nforeground\n",
    ):
        assert legacy_marker not in system + "\n" + user


def test_perception_prompt_receives_only_public_surface_in_user(manager):
    catalog = (
        '<presentation_catalog current="neutral">\n'
        "- neutral: standing still\n"
        "</presentation_catalog>"
    )
    values = _turn_values(
        request_packet=(
            "<public_identity>\n"
            "You are Juniper Unique Actor.\n"
            "Your public role is tide archivist.\n"
            "Your established appearance is ink-stained cuffs.\n"
            "Your visible affiliation is the harbor archive.\n"
            f"</public_identity>\n\n{catalog}"
        )
    )
    messages = manager.render_messages("agent_perception", **values)
    system = messages[0]["content"]
    user = messages[1]["content"]

    private_values = (
        "You spent seven winters copying names from wreckage.",
        "You are patient with records and impatient with evasions.",
        "The harbor bell rings twice for an unclaimed body.",
        "You want to recover the missing ledger.",
        "You kept one page instead of filing it.",
    )
    for value in private_values:
        assert value not in system
        assert value not in user
    assert "Juniper Unique Actor" not in system
    assert "Juniper Unique Actor" in user
    assert "tide archivist" not in system
    assert "ink-stained cuffs" not in system
    assert "harbor archive" not in system
    assert "tide archivist" in user
    assert "ink-stained cuffs" in user
    assert "harbor archive" in user
    assert "This should be ignored." not in user
    assert '<presentation_catalog current="neutral">' in user
    for legacy_marker in (
        "## AGENT-TURN",
        "## PERCEPTION",
        "## Turn Frame",
        "\nforeground\n",
    ):
        assert legacy_marker not in system + "\n" + user


def test_turn_system_is_stable_when_actor_packet_changes(manager):
    first = manager.render_messages("agent_turn", **_turn_values())
    second = manager.render_messages(
        "agent_turn",
        ruleset_guidance="",
        request_packet=(
            "<identity>\n"
            "You are Aster Different Actor.\n"
            "You repair clocks and distrust compliments.\n"
            "</identity>\n\n"
            "A different current observation reached you."
        ),
    )

    assert first[0]["content"] == second[0]["content"]
    assert "Juniper Unique Actor" not in first[0]["content"]
    assert "Aster Different Actor" not in second[0]["content"]
    assert "You spent seven winters" not in first[0]["content"]
    assert "You repair clocks" not in second[0]["content"]
    assert "Juniper Unique Actor" in first[1]["content"]
    assert "Aster Different Actor" in second[1]["content"]
    assert "different current observation" in second[1]["content"]


def test_runtime_character_prompts_do_not_contain_evaluator_vocabulary():
    forbidden = (
        r"\bepistemology\b",
        r"\binterpersonal objective\b",
        r"\bswappability\b",
        r"\brubric\b",
        r"\bstatus negotiation\b",
        r"\bdramatic debt\b",
        r"\bsubtext analysis\b",
        r"\bengine\b",
        r"\bprovider\b",
        r"\bbenchmark\b",
        r"\bevaluator\b",
        r"\bevaluation\b",
        r"\bcompaction\b",
        r"\bcontext window\b",
        r"\brolling history\b",
        r"\bAGENT-TURN\b",
        r"\bPERCEPTION\b",
    )
    for path in ("app/prompts/agent_turn.txt", "app/prompts/agent_perception.txt"):
        text = Path(path).read_text()
        for pattern in forbidden:
            assert re.search(pattern, text, flags=re.IGNORECASE) is None, (
                path,
                pattern,
            )


def test_turn_prompt_does_not_restore_fixed_or_clipped_response_shape():
    prompt = Path("app/prompts/agent_turn.txt").read_text()
    frame_source = Path("app/engine/story_contracts.py").read_text()

    forbidden = (
        r"\bone observable contribution\b",
        r"\b(?:one|two|three|four|\d+)(?:\s+to\s+(?:one|two|three|four|\d+))?\s+sentences?\b",
        r"\bReact in real time to what reached you\b",
        r"\bstate (?:your|the) (?:theme|lesson)\b",
    )
    for pattern in forbidden:
        assert re.search(pattern, prompt, flags=re.IGNORECASE) is None
        assert re.search(pattern, frame_source, flags=re.IGNORECASE) is None


def test_one_star_addon_is_mechanics_only_at_the_generic_prompt_boundary():
    text = Path("app/prompts/agent_ruleset_one_star.txt").read_text().lower()

    for forbidden in (
        "private_intent",
        "trailing parenthetical",
        "one-star dialogue",
        "subtext",
        "epistemology",
        "status negotiation",
    ):
        assert forbidden not in text


def test_retired_unified_and_repair_templates_are_not_resolvable(manager):
    for template_name in ("agent", "agent_format_repair"):
        with pytest.raises(FileNotFoundError):
            manager._find_template(template_name)
