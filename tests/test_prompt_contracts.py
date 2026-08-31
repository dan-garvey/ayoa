from app.engine.turn_loop_contracts import (
    ACTOR_SUBMISSION_HEADER,
    ROUTER_CONTINUATION_HEADER,
    UNANSWERED_RESPONDERS_SUBHEADER,
    format_actor_submission,
    format_cat_ii_resolution_block,
    format_character_moment,
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
    """Contract helpers that build volatile character input."""

    def test_background_moment_renders_location_and_local_context(self):
        body = format_character_moment(
            frame="background",
            location="Library",
            local_context="Dusty stacks.",
        )
        assert "Library" in body
        assert "Dusty stacks." in body
        assert "## AGENT-TURN" not in body
        assert "## PERCEPTION" not in body
        assert "## Turn Frame" not in body
        assert not any(
            line.strip() in {"foreground", "background", "private"}
            for line in body.splitlines()
        )

    def test_foreground_moment_contains_only_immediate_local_context(self):
        body = format_character_moment(
            frame="foreground",
            local_context="Aldric strains against the building.",
        )
        assert "Aldric strains against the building." in body
        assert "## Scene" not in body
        assert "## What You Observe This Turn" not in body
        assert "## Other Characters' Responses This Turn" not in body
        assert "## AGENT-TURN" not in body
        assert "## PERCEPTION" not in body
        assert "## Turn Frame" not in body
        assert not any(
            line.strip() in {"foreground", "background", "private"}
            for line in body.splitlines()
        )
