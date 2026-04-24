"""Tests for the prompt manager — template loading, rendering, version extraction."""

import pytest

from app.engine.prompt_manager import PromptManager


@pytest.fixture
def prompts_dir(tmp_path):
    """Create a temp prompts directory with test templates."""
    (tmp_path / "greeting_v1.txt").write_text("Hello {name}, welcome to {place}.")
    (tmp_path / "greeting_v2.txt").write_text("Welcome, {name}! You are in {place}.")
    (tmp_path / "simple_v1.txt").write_text("No variables here.")
    return tmp_path


@pytest.fixture
def mgr(prompts_dir):
    return PromptManager(prompts_dir=str(prompts_dir))


class TestPromptManagerRender:
    def test_render_exact_name(self, mgr):
        result = mgr.render("greeting_v1", name="Alice", place="the courtyard")
        assert result == "Hello Alice, welcome to the courtyard."

    def test_render_base_name_uses_latest(self, mgr):
        result = mgr.render("greeting", name="Bob", place="the hall")
        assert result == "Welcome, Bob! You are in the hall."

    def test_render_no_variables(self, mgr):
        result = mgr.render("simple_v1")
        assert result == "No variables here."

    def test_render_missing_variable_raises(self, mgr):
        with pytest.raises(KeyError, match="name"):
            mgr.render("greeting_v1", place="somewhere")

    def test_render_missing_template_raises(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.render("nonexistent", foo="bar")


class TestPromptManagerVersions:
    def test_get_version_exact(self, mgr):
        assert mgr.get_version("greeting_v1") == "v1"
        assert mgr.get_version("greeting_v2") == "v2"

    def test_get_version_base_name(self, mgr):
        # Base name resolves to latest
        assert mgr.get_version("greeting") == "v2"

    def test_get_all_versions(self, mgr):
        versions = mgr.get_all_versions()
        assert versions["greeting"] == "v2"  # latest wins
        assert versions["simple"] == "v1"

    def test_get_version_missing_raises(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.get_version("nonexistent")


class TestPromptManagerInclude:
    def test_expand_include_loads_partial(self, tmp_path):
        partials = tmp_path / "_partials"
        partials.mkdir()
        (partials / "greet.txt").write_text("Hello {name}")
        (tmp_path / "wrap_v1.txt").write_text('Preamble {include "greet"} <<<USER>>>\nAfter')
        mgr = PromptManager(prompts_dir=str(tmp_path))
        out = mgr.render("wrap", name="Vero")
        assert out == "Preamble Hello Vero <<<USER>>>\nAfter"

    def test_nested_include(self, tmp_path):
        partials = tmp_path / "_partials"
        partials.mkdir()
        (partials / "inner.txt").write_text("{x}")
        (partials / "outer.txt").write_text('a {include "inner"} b')
        (tmp_path / "t_v1.txt").write_text('{include "outer"} <<<USER>>>')
        mgr = PromptManager(prompts_dir=str(tmp_path))
        assert mgr.render("t", x="ok") == "a ok b <<<USER>>>"

    def test_include_missing_raises(self, tmp_path):
        (tmp_path / "t_v1.txt").write_text('{include "nope"} <<<USER>>>')
        mgr = PromptManager(prompts_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Include not found"):
            mgr.render("t")


class TestPromptManagerInit:
    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            PromptManager(prompts_dir="/nonexistent/path")


class TestPromptManagerWithRealTemplates:
    """Verify the actual project prompt templates load and render."""

    def test_all_templates_load(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        versions = mgr.get_all_versions()
        assert "narrator_phase2" in versions
        assert "agent" in versions
        assert "character_gen" in versions
        assert "event_router" in versions
        # Legacy templates removed:
        assert "narrator_phase1" not in versions
        assert "discriminator" not in versions
        assert "transcript_summary" not in versions

    def test_event_router_renders(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        result = mgr.render(
            "event_router",
            setting_summary="Genre: fantasy\nTone: dark",
            world_lore="The kingdom has been at war.",
            world_rules="No magic. Human baseline strength.",
            scene_graph="- courtyard (id: courtyard)",
            current_scene="A stone courtyard in the rain.",
            characters_present="- Captain Vero (guard): disciplined",
            hidden_lore="Secret conspiracy details.",
            hidden_facts="- Hidden fact one",
            user_input="I try to lift the building.",
            acting_character_name="Aldric",
            acting_character_id="aldric",
            player_characters_block="- **Aldric** (acting this turn) (id: aldric) — scholar. Tall, broad-shouldered, grey-streaked hair.",
            since_last_turn_block="",
            opening_directive="",
            recent_turn_recap="",
            # Commit-3: dropped `world_facts` (full) + `character_registry`
            # from per-turn context. Replaced with three new optional
            # blocks; on most calls these are empty strings.
            world_facts_delta_block="",
            initial_roster_block="",
            state_changes_block="",
            cat_ii_resolution_block="",
            tick_fan_in_block="",
            intention_block="## Intention\nAldric attempts: I try to lift the building.",
        )
        assert "I try to lift the building" in result
        assert "Aldric" in result
        assert "No magic" in result
        assert "Captain Vero" in result
        assert "Tall, broad-shouldered" in result

    def test_agent_renders(self):
        # v11: unified on-stage + tick template. The mode-specific
        # body lives in `mode_block` (caller-assembled string from
        # `format_agent_*_body`) and the first-token mode signal
        # lives in `mode_header`. The on-stage-specific scene /
        # presence / observed-facts / prior-responders surfaces moved
        # OUT of the template's variable list and INTO mode_block;
        # the system prefix is now identical between respond and
        # tick so a single cache lineage covers both modes.
        mgr = PromptManager(prompts_dir="app/prompts")
        on_stage_body = (
            "## Scene\nEstate courtyard, raining.\n\n"
            "## Characters Present\nNo other characters are present.\n\n"
            "## What You Observe This Turn\n"
            "Aldric strains against the building.\n\n"
            "## Other Characters' Responses This Turn\n"
            "No other characters have responded yet."
        )
        result = mgr.render(
            "agent",
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
            acting_character_name="Aldric",
            player_characters_block=(
                "- **Aldric** (acting this turn) — scholar. Tall, "
                "in rain-darkened traveling clothes."
            ),
            mode_header="## ON-STAGE",
            mode_block=on_stage_body,
        )
        assert "Captain Vero" in result
        # The character_id kwarg is silently ignored — v11 dropped the
        # surface (it was a debug echo, never the LLM's hook) but the
        # engine still passes the kwarg for symmetry with the rest of
        # the character packet.
        assert "hidden passage" in result
        assert "rain-darkened traveling clothes" in result
        # Mode header AND body markers both present; the agent's
        # "Mode Routing" section keys off the header line.
        assert "## ON-STAGE" in result
        assert "## Scene" in result
        assert "Estate courtyard, raining" in result

    def test_render_messages_requires_delimiter(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        # All canonical templates have the delimiter, so this should succeed.
        messages = mgr.render_messages(
            "event_router",
            setting_summary="x", world_lore="x", world_rules="x",
            scene_graph="x", current_scene="x", characters_present="x",
            hidden_lore="x", hidden_facts="x",
            user_input="x",
            acting_character_name="x",
            acting_character_id="x",
            player_characters_block="x",
            since_last_turn_block="",
            opening_directive="",
            recent_turn_recap="",
            world_facts_delta_block="",
            initial_roster_block="",
            state_changes_block="",
            cat_ii_resolution_block="",
            tick_fan_in_block="",
            intention_block="",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_render_messages_rejects_missing_delimiter(self, mgr):
        # The tmp-path `greeting` fixture has no <<<USER>>> delimiter.
        with pytest.raises(ValueError, match="<<<USER>>>"):
            mgr.render_messages("greeting_v1", name="a", place="b")

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
            scene_graph="x", current_scene="x", characters_present="x",
            hidden_lore="x", hidden_facts="x",
            user_input="x",
            acting_character_name="x",
            acting_character_id="x",
            player_characters_block="x",
            since_last_turn_block="",
            opening_directive="",
            recent_turn_recap="",
            world_facts_delta_block="",
            initial_roster_block="",
            state_changes_block="",
            cat_ii_resolution_block="",
            tick_fan_in_block="",
            intention_block="",
        )
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "prev user"
        assert messages[2]["content"] == "prev assistant"
        assert messages[3]["role"] == "user"
