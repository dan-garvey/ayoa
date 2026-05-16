from app.engine.turn_loop_contracts import (
    SWEPT_RESPONDERS_SUBHEADER,
    ROUTER_CONTINUATION_HEADER,
    ROUTER_FRONTIER_RESULTS_HEADER,
    format_agent_turn_body,
    format_agent_on_stage_body,
    format_cat_ii_resolution_block,
    format_frontier_results_block,
    format_human_initiator_intention,
    format_npc_cascade_intention,
    format_router_continuation_block,
)


class TestContractHelpers:
    def test_human_initiator_framing(self):
        out = format_human_initiator_intention("alice", "I look around")
        assert out == "I look around"
        assert "## Intention" not in out
        assert "attempts:" not in out

    def test_npc_cascade_framing(self):
        out = format_npc_cascade_intention("pip", "steps closer")
        assert out == "steps closer"
        assert "intends:" not in out
        assert "attempts:" not in out

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


class TestRouterContinuationBlock:
    def test_continuation_block_is_not_framed_as_character_intention(self):
        block = format_router_continuation_block(
            prior_rationale="Router left the beat open.",
        )
        assert block.startswith(ROUTER_CONTINUATION_HEADER)
        assert "attempts:" not in block
        assert "intends:" not in block
        assert "Router left the beat open." in block


class TestFrontierResultsBlock:
    """Frontier helper. The block must render the frontier header, name
    each completed target with character id/frame/result kind, and forward
    only public_text. Agent parentheticals stay in the agent's own history.
    """

    def test_empty_list_returns_empty_string(self):
        assert format_frontier_results_block([]) == ""

    def test_renders_header_and_per_entry_lines(self):
        block = format_frontier_results_block([
            ("agent_turn", "regent", "background", "He paces the long table."),
            ("agent_turn", "scribe", "foreground", "She copies a passage."),
        ])
        assert ROUTER_FRONTIER_RESULTS_HEADER in block
        assert "2 selected frontier target(s)" in block
        assert "regent" in block
        assert "background" in block
        assert "He paces the long table." in block
        assert "foreground" in block
        assert "She copies a passage." in block

    def test_blank_public_text_falls_back_to_silent_marker(self):
        block = format_frontier_results_block([
            ("agent_turn", "mute", "background", ""),
        ])
        assert "(no public action)" in block


class TestAgentModeContract:
    """Contract helpers that build the user-message bodies for agent modes."""

    def test_on_stage_body_is_empty(self):
        """v11-r10: the on-stage body has no per-turn content. The
        three blocks it once carried — `## Scene`, `## What You
        Observe This Turn`, `## Other Characters' Responses This
        Turn` — were all removed because the agent already learns
        the same information through their `pending_observations`
        inbox (location seeded at importer/spawn, perception pushed by
        `broadcast_event`). Pin the empty-body shape so a future
        edit that re-adds a per-turn block here is loud."""
        body = format_agent_on_stage_body()
        assert body == ""

    def test_background_turn_body_renders_location_and_instruction(self):
        body = format_agent_turn_body(
            frame="background",
            location_context="Location: Library (id: library)\nDusty stacks.",
        )
        assert "background" in body
        assert "## Where You Are" in body
        assert "## What You Do" in body
        assert "Library" in body
        assert "single tight beat" in body

    def test_agent_turn_body_does_not_carry_removed_context_blocks(self):
        body = format_agent_turn_body(frame="background", location_context="x")
        assert "## Scene" not in body
        assert "## What You Observe This Turn" not in body
        assert "## Other Characters' Responses This Turn" not in body
