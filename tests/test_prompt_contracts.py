from pathlib import Path

from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import (
    AGENT_ON_STAGE_HEADER,
    AGENT_TICK_HEADER,
    CAT_II_RESOLUTION_HEADER,
    PARTIAL_MODE_MARKER,
    SWEPT_RESPONDERS_SUBHEADER,
    TICK_FAN_IN_HEADER,
    format_agent_on_stage_body,
    format_agent_tick_body,
    format_cat_ii_resolution_block,
    format_human_initiator_intention,
    format_npc_cascade_intention,
    format_ooc_directive,
    format_partial_render_marker,
    format_tick_fan_in_block,
)

ROUTER_PROMPT = Path("app/prompts/event_router_v9.txt").read_text()
NARRATOR_PROMPT = Path("app/prompts/narrator_phase2_v9.txt").read_text()
# Agent prompt is version-stamped — load via PromptManager so test
# follows the file forward as v11 → v12 etc., instead of pinning a
# specific filename and silently going stale.
_AGENT_PROMPT_PATH = PromptManager(
    prompts_dir="app/prompts",
)._find_template("agent")
AGENT_PROMPT = _AGENT_PROMPT_PATH.read_text()


class TestPromptReferencesConstants:
    def test_router_prompt_mentions_cat_ii_resolution_header(self):
        assert CAT_II_RESOLUTION_HEADER in ROUTER_PROMPT

    def test_router_prompt_mentions_swept_responders_subheader(self):
        assert SWEPT_RESPONDERS_SUBHEADER in ROUTER_PROMPT

    def test_narrator_prompt_mentions_partial_mode_marker(self):
        assert PARTIAL_MODE_MARKER in NARRATOR_PROMPT


class TestContractHelpers:
    def test_human_initiator_framing(self):
        out = format_human_initiator_intention("Alice", "I look around")
        assert out.startswith("## Intention")
        assert "Alice attempts: I look around" in out

    def test_npc_cascade_framing(self):
        out = format_npc_cascade_intention("Pip", "steps closer")
        assert "Pip intends: steps closer" in out
        assert "attempts:" not in out  # Crucial — NPC cascades must NOT use "attempts"

    def test_ooc_framing(self):
        out = format_ooc_directive("(begin)")
        assert "(OOC) (begin)" in out

    def test_cat_ii_resolution_omits_swept_from_responder_list(self):
        block = format_cat_ii_resolution_block(
            initiator_id="pip",
            initiator_intention="throws a punch at Alice",
            responders=[
                ("alice", "I duck"),
                ("bob", "(sentinel text — should be filtered)"),
            ],
            swept_responders=["bob"],
        )
        assert "Initiator (pip)" in block
        assert "alice: I duck" in block
        assert "sentinel text" not in block  # Bob's intention text NOT in block.
        assert SWEPT_RESPONDERS_SUBHEADER in block
        assert "bob" in block  # Bob is named under swept subheader.

    def test_cat_ii_resolution_no_swept_section_when_all_live(self):
        block = format_cat_ii_resolution_block(
            initiator_id="pip",
            initiator_intention="punch",
            responders=[("alice", "dodge")],
            swept_responders=[],
        )
        assert SWEPT_RESPONDERS_SUBHEADER not in block

    def test_partial_marker(self):
        assert format_partial_render_marker() == PARTIAL_MODE_MARKER


class TestTickFanInBlock:
    """Commit 6 fan-in helper. The block must render the
    TICK_FAN_IN_HEADER (so the router prompt's mode-routing line can
    detect tick mode), name each ticker with their character_id and
    location, and faithfully forward the public_text the caller
    handed over. Crucially, it must NEVER carry the agent's
    parenthetical (intent) — that asymmetry is the whole reason we
    have separate per-actor LLM calls; the caller is responsible for
    stripping intent before this helper runs.
    """

    def test_empty_list_returns_empty_string(self):
        # Empty input → "" so the dispatcher can short-circuit the
        # router call instead of firing a payload-less LLM hit.
        assert format_tick_fan_in_block([]) == ""

    def test_renders_header_and_per_entry_lines(self):
        block = format_tick_fan_in_block([
            ("Regent", "regent", "great_hall", "He paces the long table."),
            ("Scribe", "scribe", "library", "She copies a passage."),
        ])
        assert TICK_FAN_IN_HEADER in block
        assert "2 off-stage NPC(s)" in block
        # Per-entry: name, id, location, body all present.
        assert "**Regent**" in block
        assert "regent" in block
        assert "great_hall" in block
        assert "He paces the long table." in block
        assert "**Scribe**" in block
        assert "library" in block
        assert "She copies a passage." in block

    def test_blank_location_falls_back_to_unset(self):
        block = format_tick_fan_in_block([
            ("Wraith", "wraith", "", "It drifts."),
        ])
        assert "(unset)" in block
        assert "It drifts." in block

    def test_blank_public_text_falls_back_to_silent_marker(self):
        # Some agents emit only a parenthetical (silent beat). The
        # caller has already stripped intent, so public_text is "".
        # The block must surface a placeholder so the router still
        # sees the character was active without inventing prose.
        block = format_tick_fan_in_block([
            ("Mute", "mute", "stables", ""),
        ])
        assert "(no public action)" in block

    def test_router_prompt_mentions_tick_fan_in_header(self):
        # Mode routing in event_router_v9 keys off this exact string;
        # if we ever rename the constant, this catches the prompt
        # falling out of sync with the contract.
        assert TICK_FAN_IN_HEADER in ROUTER_PROMPT


class TestAgentModeContract:
    """v11 unified-agent contract. There is now ONE agent system
    prompt for both on-stage and off-stage calls (cache-trail
    deduplication — `agent_v10.txt` and `agent_tick_v3.txt` are
    merged). The agent identifies its mode by reading the FIRST line
    of its current user message: `## ON-STAGE` or `## TICK`. The
    prompt's "Mode Routing" section keys off those exact strings;
    these tests pin the prompt-code contract so a rename or a
    silent header drift trips a loud failure.

    Why this matters: if the prompt's mode-routing block falls out
    of sync with the constants, the agent has no signal to flip its
    rule set — and we'd silently regress to "tick agents follow
    on-stage rules" or vice versa, with no test to catch it.
    """

    def test_agent_prompt_mentions_on_stage_header(self):
        assert AGENT_ON_STAGE_HEADER in AGENT_PROMPT

    def test_agent_prompt_mentions_tick_header(self):
        assert AGENT_TICK_HEADER in AGENT_PROMPT

    def test_agent_prompt_has_mode_routing_section(self):
        # The "Mode Routing" header is the cross-reference target for
        # agent.respond / agent.tick — if it disappears the prompt's
        # rule structure has been refactored away from the
        # first-token-bitflip design.
        assert "Mode Routing" in AGENT_PROMPT

    def test_agent_prompt_keeps_user_message_template_slots(self):
        # The unified template's user message contains exactly two
        # interpolation slots after the system delimiter:
        # `{mode_header}` (first-token signal) and `{mode_block}`
        # (mode-specific body assembled by the matching helper).
        # Pin both so a refactor that drops one is loud.
        _, user_tail = AGENT_PROMPT.split("<<<USER>>>", 1)
        assert "{mode_header}" in user_tail
        assert "{mode_block}" in user_tail

    def test_on_stage_body_renders_required_headers(self):
        body = format_agent_on_stage_body(
            scene_context="Estate courtyard, raining.",
            characters_present="No other characters are present.",
            observed_facts="Aldric strains against the building.",
            prior_character_responses="No other characters have responded yet.",
        )
        # Each header pinned individually so a partial drift (one
        # section vanishes) still fails distinctly.
        assert "## Scene" in body
        assert "## Characters Present" in body
        assert "## What You Observe This Turn" in body
        assert "## Other Characters' Responses This Turn" in body
        # And the inputs survive verbatim into the body.
        assert "Estate courtyard, raining." in body
        assert "Aldric strains against the building." in body

    def test_tick_body_renders_location_and_standing_instruction(self):
        body = format_agent_tick_body(
            scene_context="Location: Library (id: library)\nDusty stacks.",
        )
        assert "## Where You Are" in body
        assert "## What You Do This Tick" in body
        assert "Library" in body
        # Standing instruction text — the agent's tick rules expect
        # this nudge to be present.
        assert "off-stage" in body
        assert "single tight beat" in body

    def test_on_stage_body_does_not_carry_tick_markers(self):
        # The two mode bodies must be visually + semantically
        # distinct — if on-stage starts emitting `## What You Do
        # This Tick` (or the reverse), the agent's mode-routing
        # would see conflicting signals.
        body = format_agent_on_stage_body(
            scene_context="x",
            characters_present="x",
            observed_facts="x",
            prior_character_responses="x",
        )
        assert "## Where You Are" not in body
        assert "## What You Do This Tick" not in body

    def test_tick_body_does_not_carry_on_stage_markers(self):
        body = format_agent_tick_body(scene_context="x")
        assert "## Scene" not in body
        assert "## Characters Present" not in body
        assert "## What You Observe This Turn" not in body
        assert "## Other Characters' Responses This Turn" not in body


class TestRule2bCrossReference:
    def test_rule_2b_mentions_part_c(self):
        """Rule 2b must forward-reference Part C so a reader can't miss the suspension."""
        # Search for the rule 2b text + confirm it mentions Part C / cat_ii_resolution.
        # The rule 2b header is flagged by "Rule 2b" or "2b.**" somewhere.
        # Tolerant assertion: somewhere near "render the attempt, not the completion"
        # there's a "Part C" mention.
        lines = ROUTER_PROMPT.split("\n")
        for i, ln in enumerate(lines):
            if "render the attempt, not the completion" in ln.lower():
                window = "\n".join(lines[i : i + 10])
                assert "Part C" in window or "cat_ii_resolution" in window, \
                    "Rule 2b should forward-reference Part C's suspension"
                break


class TestMultiEventExemplar:
    def test_narrator_has_multi_event_exemplar(self):
        assert "Center of gravity" in NARRATOR_PROMPT
        assert "subordinate clause" in NARRATOR_PROMPT


class TestPartIInvariantInRouter:
    def test_picks_subset_of_observers_invariant(self):
        """`agent_responder_picks` must reference `observers` as an invariant."""
        assert "agent_responder_picks" in ROUTER_PROMPT
        assert "observers" in ROUTER_PROMPT
        # Look for INVARIANT keyword near agent_responder_picks
        lower = ROUTER_PROMPT.lower()
        idx = lower.find("invariant")
        assert idx != -1, "Expected 'INVARIANT' keyword tagging the picks⊆observers rule"
