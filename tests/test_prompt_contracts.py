from app.engine.turn_loop_contracts import (
    SWEPT_RESPONDERS_SUBHEADER,
    ROUTER_CONTINUATION_HEADER,
    TICK_FAN_IN_HEADER,
    format_agent_on_stage_body,
    format_agent_tick_body,
    format_cat_ii_resolution_block,
    format_human_initiator_intention,
    format_npc_cascade_intention,
    format_router_continuation_block,
    format_tick_fan_in_block,
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
        # Per-entry: id, location, body all present; name is not repeated.
        assert "**Regent**" not in block
        assert "regent" in block
        assert "great_hall" in block
        assert "He paces the long table." in block
        assert "**Scribe**" not in block
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

    def test_tick_body_renders_location_and_standing_instruction(self):
        body = format_agent_tick_body(
            location_context="Location: Library (id: library)\nDusty stacks.",
        )
        assert "## Where You Are" in body
        assert "## What You Do This Tick" in body
        assert "Library" in body
        # Standing instruction text — the agent's tick rules expect
        # this nudge to be present.
        assert "off-stage" in body
        assert "single tight beat" in body

    def test_tick_body_does_not_carry_on_stage_markers(self):
        # Tick mode keeps the only remaining headered block
        # (`## Where You Are` for location framing); on-stage's
        # historical headers (`## Scene`, `## What You Observe This
        # Turn`, `## Other Characters' Responses This Turn`) are
        # gone everywhere now and must not resurface in tick body
        # either.
        body = format_agent_tick_body(location_context="x")
        assert "## Scene" not in body
        assert "## What You Observe This Turn" not in body
        assert "## Other Characters' Responses This Turn" not in body
