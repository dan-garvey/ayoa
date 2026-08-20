"""Tests for the prompt manager — template loading, rendering, partials.

Versioning policy: prompts are versioned in git, not in their filenames.
A template named `event_router` is the file `app/prompts/event_router.txt`;
revisions are normal commits. The PromptManager is therefore a thin
wrapper around `(name -> {prompts_dir}/{name}.txt)`, plus include
expansion and `<<<USER>>>` system/user splitting.
"""

import re

import pytest

from app.engine.prompt_manager import PromptManager


@pytest.fixture
def prompts_dir(tmp_path):
    """Create a temp prompts directory with test templates."""
    (tmp_path / "greeting.txt").write_text("Hello {name}, welcome to {place}.")
    (tmp_path / "simple.txt").write_text("No variables here.")
    return tmp_path


@pytest.fixture
def mgr(prompts_dir):
    return PromptManager(prompts_dir=str(prompts_dir))


class TestPromptManagerRender:
    def test_render_template(self, mgr):
        result = mgr.render("greeting", name="Alice", place="the courtyard")
        assert result == "Hello Alice, welcome to the courtyard."

    def test_render_no_variables(self, mgr):
        result = mgr.render("simple")
        assert result == "No variables here."

    def test_render_missing_variable_raises(self, mgr):
        with pytest.raises(KeyError, match="name"):
            mgr.render("greeting", place="somewhere")

    def test_render_missing_template_raises(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.render("nonexistent", foo="bar")

    def test_render_does_not_glob_versioned_files(self, tmp_path):
        # Defensive against the pre-cleanup behavior: a directory with
        # ONLY `greeting_v1.txt` (no bare `greeting.txt`) must not
        # satisfy `render("greeting")`. The pre-cleanup manager
        # globbed `greeting_v*.txt` and silently picked the highest;
        # post-cleanup, git is the version store and the engine must
        # call its templates by their actual stem.
        (tmp_path / "greeting_v1.txt").write_text("Hello {name}")
        mgr = PromptManager(prompts_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            mgr.render("greeting", name="Alice")


class TestPromptManagerInclude:
    def test_expand_include_loads_partial(self, tmp_path):
        partials = tmp_path / "_partials"
        partials.mkdir()
        (partials / "greet.txt").write_text("Hello {name}")
        (tmp_path / "wrap.txt").write_text('Preamble {include "greet"} <<<USER>>>\nAfter')
        mgr = PromptManager(prompts_dir=str(tmp_path))
        out = mgr.render("wrap", name="Vero")
        assert out == "Preamble Hello Vero <<<USER>>>\nAfter"

    def test_nested_include(self, tmp_path):
        partials = tmp_path / "_partials"
        partials.mkdir()
        (partials / "inner.txt").write_text("{x}")
        (partials / "outer.txt").write_text('a {include "inner"} b')
        (tmp_path / "t.txt").write_text('{include "outer"} <<<USER>>>')
        mgr = PromptManager(prompts_dir=str(tmp_path))
        assert mgr.render("t", x="ok") == "a ok b <<<USER>>>"

    def test_include_missing_raises(self, tmp_path):
        (tmp_path / "t.txt").write_text('{include "nope"} <<<USER>>>')
        mgr = PromptManager(prompts_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Include not found"):
            mgr.render("t")


class TestPromptManagerInit:
    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            PromptManager(prompts_dir="/nonexistent/path")


class TestPromptManagerWithRealTemplates:
    """Verify the actual project prompt templates load and render."""

    def test_legacy_template_names_rejected(self):
        # The old `_v#`-suffixed names are gone; calling code that still
        # tries to render them must fail loudly so the regression is
        # obvious rather than silently picking up a stale stem.
        mgr = PromptManager(prompts_dir="app/prompts")
        for legacy in (
            "event_router_v9",
            "narrator_phase2_v9",
            "agent_v11",
            "character_gen_v3",
            "takeover_v1",
        ):
            with pytest.raises(FileNotFoundError):
                mgr._find_template(legacy)

    def test_event_router_renders(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        result = mgr.render(
            "event_router",
            setting_summary="Genre: fantasy\nTone: dark",
            world_lore="The kingdom has been at war.",
            world_rules="No magic. Human baseline strength.",
            hidden_lore="Secret conspiracy details.",
            hidden_facts="- Hidden fact one",
            acting_character_name="Aldric",
            acting_character_id="aldric",
            fresh_intention_classifier=mgr.render(
                "event_router_ruleset_default",
            ).strip(),
            router_input_block="I try to lift the building.",
        )
        assert "I try to lift the building" in result
        assert "aldric" in result
        assert "No magic" in result

    def test_dnd_cat_ii_router_renders(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        result = mgr.render(
            "dnd_cat_ii_router",
            phase="PLAN_ROLLS",
            contested_action_packet='{"initiator_id": "alice"}',
            roll_ledger_block="No rolls have been made yet.",
        )
        assert "PLAN_ROLLS" in result
        assert "alice" in result

    def test_dnd_combat_manager_renders_action_turn_plan(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        result = mgr.render(
            "dnd_combat_manager",
            phase="PLAN_TURN",
            combat_action_packet='{"current_turn": {"actor_id": "alice"}}',
            planned_actions_block="No planned actions yet.",
            roll_ledger_block="No rolls have been made yet.",
        )
        assert "PLAN_TURN" in result
        assert "Planned Actions" in result
        assert "alice" in result

    def test_agent_renders(self):
        # v11: unified on-stage + tick template. The mode-specific
        # body lives in `mode_block` (caller-assembled string from
        # `format_agent_*_body`) and the first-token mode signal
        # lives in `mode_header`. The on-stage-specific location /
        # presence / observed-facts / prior-responders surfaces moved
        # OUT of the template's variable list and INTO mode_block;
        # the system prefix is now identical between respond and
        # tick so a single cache lineage covers both modes.
        #
        # v11-r10 (2026-04): the agent template no longer surfaces a
        # `## Player Characters` block. That section explicitly told
        # the model "these are the human-played characters at the
        # keyboard, treat them as protagonists" — a sycophancy primer
        # that fought the entire reason we run agents (authentic NPC
        # POV). The render kwargs `acting_character_name` and
        # `player_characters_block` are no longer required by the
        # template; this test stops asserting on the dropped block.
        mgr = PromptManager(prompts_dir="app/prompts")
        on_stage_body = (
            "## What Reached You This Turn\n"
            "Aldric strains against the building.\n\n"
            "## Other Characters' Responses This Turn\n"
            "No other characters have responded yet."
        )
        result = mgr.render(
            "agent",
            agent_ruleset_system_addon="",
            character_name="Captain Vero",
            character_role="guard captain",
            character_appearance="Tall, scarred",
            character_faction="City Watch",
            character_backstory="Twenty years of service.",
            character_personality="Stoic but caring.",
            character_goals="- maintain order",
            character_current_objectives="- monitor the new arrival",
            character_secrets="- knows the hidden passage",
            world_context="Genre: fantasy",
            character_id="guard_17",
            pending_observations_block="",
            mode_header="## ON-STAGE",
            mode_block=on_stage_body,
        )
        assert "Captain Vero" in result
        # The character_id kwarg is silently ignored — v11 dropped the
        # surface (it was a debug echo, never the LLM's hook) but the
        # engine still passes the kwarg for symmetry with the rest of
        # the character packet.
        assert "hidden passage" in result
        # Mode header AND body markers both present; the agent's
        # "Mode Routing" section keys off the header line.
        assert "## ON-STAGE" in result
        assert "## What Reached You This Turn" in result
        # Sycophancy guard: the dropped block must not reappear.
        assert "## Player Characters" not in result
        assert "human-played" not in result
        assert "human at the keyboard" not in result

    def test_render_messages_requires_delimiter(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        # All canonical templates have the delimiter, so this should succeed.
        messages = mgr.render_messages(
            "event_router",
            setting_summary="x", world_lore="x", world_rules="x",
            hidden_lore="x", hidden_facts="x",
            acting_character_name="x",
            acting_character_id="x",
            player_characters_block="x",
            fresh_intention_classifier=mgr.render(
                "event_router_ruleset_default",
            ).strip(),
            router_input_block="",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_render_system_message_formats_only_system_prefix(self, tmp_path):
        (tmp_path / "router.txt").write_text(
            "System {stable} <<<USER>>> User {volatile}"
        )
        mgr = PromptManager(prompts_dir=str(tmp_path))

        message = mgr.render_system_message("router", stable="cacheable")

        assert message == {"role": "system", "content": "System cacheable"}

    def test_render_system_message_still_requires_system_variables(
        self, tmp_path,
    ):
        (tmp_path / "router.txt").write_text(
            "System {stable} <<<USER>>> User {volatile}"
        )
        mgr = PromptManager(prompts_dir=str(tmp_path))

        with pytest.raises(KeyError, match="stable"):
            mgr.render_system_message("router", volatile="turn")

    def test_event_router_keeps_actor_context_out_of_system_prefix(self):
        """Actor state belongs in the volatile user tail, not the cache."""
        mgr = PromptManager(prompts_dir="app/prompts")
        messages = mgr.render_messages(
            "event_router",
            setting_summary="Genre: fantasy",
            world_lore="The kingdom has been at war.",
            world_rules="No magic. Human baseline strength.",
            hidden_lore="Secret conspiracy details.",
            hidden_facts="- Hidden fact one",
            acting_character_name="Aldric UniqueActor",
            acting_character_id="aldric_unique_actor",
            fresh_intention_classifier=mgr.render(
                "event_router_ruleset_default",
            ).strip(),
            router_input_block="I wait.",
        )

        system = messages[0]["content"]
        user = messages[1]["content"]

        assert "Aldric UniqueActor" not in system
        assert "aldric_unique_actor" not in system

        assert "Aldric UniqueActor" not in user
        assert "aldric_unique_actor" in user
        assert "Player Characters" not in user
        assert "human" not in user.lower()

    def test_router_templates_exclude_controller_metadata(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        rendered = {
            "event_router": mgr.render(
                "event_router",
                setting_summary="Genre: fantasy",
                world_lore="A bell marks the hour.",
                world_rules="Ordinary physical constraints apply.",
                hidden_lore="None.",
                hidden_facts="None.",
                acting_character_id="alice",
                fresh_intention_classifier="",
                router_input_block=(
                    "## Actor Submission\n\n"
                    "submitted_actor_id: alice\n"
                    "submission_text:\nI listen."
                ),
            ),
            "event_router_ruleset_dnd5e": mgr.render(
                "event_router_ruleset_dnd5e",
            ),
        }
        forbidden = (
            r"\bhuman[-_ ](?:bound|controlled|played|player)\b",
            r"\bplayer[-_ ](?:owned|controlled|bound|characters?)\b",
            r"\bnpcs?\b",
            r"\bagent[-_ ]output\b",
            r"\broute_agent_output\b",
            r"\bcharacter_bindings?\b",
            r"\bplayer_controlled\b",
            r"\bplayer_roll_mode\b",
            r"\bbinding metadata\b",
        )

        for template_name, text in rendered.items():
            for pattern in forbidden:
                assert re.search(pattern, text, flags=re.IGNORECASE) is None, (
                    template_name,
                    pattern,
                )

    def test_narrator_keeps_pov_context_out_of_system_prefix(self):
        """Narrator cache efficiency depends on POV-specific render inputs
        living in the volatile user tail, not the cached system prefix."""
        from app.engine.turn_loop_contracts import PARTIAL_MODE_MARKER

        mgr = PromptManager(prompts_dir="app/prompts")
        messages = mgr.render_messages(
            "narrator_phase2",
            setting_summary="Genre: fantasy",
            narrative_rules="Concise prose.",
            pov_character_name="Aldric UniquePOV",
            player_characters_block=(
                "- Aldric UniquePOV (you) - scholar. "
                "Unique player-block appearance."
            ),
            rendering_note=PARTIAL_MODE_MARKER,
            visible_events=(
                "Seen directly:\n"
                "- Unique event fact.\n\n"
                "Newly introduced character context:\n"
                "- Pip UniqueKnown: player-safe context: stable public glossary marker."
            ),
            user_input="Unique submitted action.",
            handoff_policy="Unique candidate policy.",
            handoff_context="Unique unresolved motion.",
        )

        system = messages[0]["content"]
        user = messages[1]["content"]

        assert "Aldric UniquePOV" not in system
        assert "Unique player-block appearance" not in system
        assert "Pip UniqueKnown" not in system
        assert "Unique event fact" not in system
        assert "stable public glossary marker" not in system
        assert "Unique submitted action" not in system
        assert "Unique candidate policy" not in system
        assert "Unique unresolved motion" not in system
        assert PARTIAL_MODE_MARKER not in system

        assert "Genre: fantasy" in system
        assert "Concise prose." in system
        assert "Aldric UniquePOV" in user
        assert "Unique player-block appearance" in user
        assert "Pip UniqueKnown" in user
        assert "stable public glossary marker" in user
        assert "Unique event fact" in user
        assert "Unique submitted action" in user
        assert "Unique candidate policy" in user
        assert "Unique unresolved motion" in user
        assert PARTIAL_MODE_MARKER in user

    def test_render_messages_rejects_missing_delimiter(self, mgr):
        # The tmp-path `greeting` fixture has no <<<USER>>> delimiter.
        with pytest.raises(ValueError, match="<<<USER>>>"):
            mgr.render_messages("greeting", name="a", place="b")

    def test_render_conversation_inserts_history(self):
        from app.schemas.conversation import ConversationMessage

        mgr = PromptManager(prompts_dir="app/prompts")
        history = [
            ConversationMessage(role="user", content="prev user"),
            ConversationMessage(role="assistant", content="prev assistant"),
        ]
        messages = mgr.render_conversation(
            "event_router",
            history=history,
            setting_summary="x", world_lore="x", world_rules="x",
            hidden_lore="x", hidden_facts="x",
            acting_character_name="x",
            acting_character_id="x",
            player_characters_block="x",
            fresh_intention_classifier=mgr.render(
                "event_router_ruleset_default",
            ).strip(),
            router_input_block="",
        )
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "prev user"
        assert messages[2]["content"] == "prev assistant"
        assert messages[3]["role"] == "user"
