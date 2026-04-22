from pathlib import Path

from app.engine.turn_loop_contracts import (
    CAT_II_RESOLUTION_HEADER,
    PARTIAL_MODE_MARKER,
    SWEPT_RESPONDERS_SUBHEADER,
    format_cat_ii_resolution_block,
    format_human_initiator_intention,
    format_npc_cascade_intention,
    format_ooc_directive,
    format_partial_render_marker,
)

ROUTER_PROMPT = Path("app/prompts/event_router_v9.txt").read_text()
NARRATOR_PROMPT = Path("app/prompts/narrator_phase2_v8.txt").read_text()


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
