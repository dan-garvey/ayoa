from app.engine.turn_loop_contracts import (
    ACTOR_SUBMISSION_HEADER,
    ROUTER_CONTINUATION_HEADER,
    UNANSWERED_RESPONDERS_SUBHEADER,
    format_agent_turn_body,
    format_actor_submission,
    format_cat_ii_resolution_block,
    format_router_continuation_block,
)


class TestContractHelpers:
    def test_actor_submission_names_character_and_preserves_proposed_motion(self):
        out = format_actor_submission("alice", "I look around")

        assert out == (
            f"{ACTOR_SUBMISSION_HEADER}\n\n"
            "submitted_actor_id: alice\n"
            "submission_text:\n"
            "I look around"
        )

    def test_actor_submission_blank_text_has_explicit_no_action_marker(self):
        out = format_actor_submission("pip", "")

        assert out.endswith("submission_text:\n(no public action)")

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
        assert UNANSWERED_RESPONDERS_SUBHEADER in block
        assert "bob" in block  # Bob is named under swept subheader.
        assert "AFK" not in block
        assert "player" not in block.lower()

    def test_cat_ii_resolution_no_swept_section_when_all_live(self):
        block = format_cat_ii_resolution_block(
            initiator_id="pip",
            initiator_intention="punch",
            responders=[("alice", "dodge")],
            swept_responders=[],
        )
        assert UNANSWERED_RESPONDERS_SUBHEADER not in block


class TestRouterContinuationBlock:
    def test_continuation_block_is_not_framed_as_character_intention(self):
        block = format_router_continuation_block(
            prior_rationale="Router left the beat open.",
            original_action="I wait until the lift arrives.",
        )
        assert block.startswith(ROUTER_CONTINUATION_HEADER)
        assert "attempts:" not in block
        assert "intends:" not in block
        assert "Router left the beat open." in block
        assert "I wait until the lift arrives." in block
        assert "Pending motion:" not in block
        assert "Cat II" in block
        assert "autonomous" not in block.lower()
        assert "player" not in block.lower()
        assert "agent" not in block.lower()


class TestAgentModeContract:
    """Contract helpers that build the user-message bodies for agent modes."""

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

    def test_foreground_turn_body_does_not_carry_removed_context_blocks(self):
        body = format_agent_turn_body(frame="foreground", location_context="x")
        assert "## Scene" not in body
        assert "## What You Observe This Turn" not in body
        assert "## Other Characters' Responses This Turn" not in body
