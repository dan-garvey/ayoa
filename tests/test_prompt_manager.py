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
            world_facts="The courtyard is wet.",
            hidden_lore="Secret conspiracy details.",
            hidden_facts="- Hidden fact one",
            character_registry="- guard_17 Captain Vero",
            user_input="I try to lift the building.",
            acting_character_name="Aldric",
            player_characters_block="- **Aldric** (acting this turn) — scholar. Tall, broad-shouldered, grey-streaked hair.",
            since_last_turn_block="",
        )
        assert "I try to lift the building" in result
        assert "Aldric" in result
        assert "No magic" in result
        assert "Captain Vero" in result
        assert "Tall, broad-shouldered" in result

    def test_agent_renders(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        result = mgr.render(
            "agent",
            character_name="Captain Vero",
            character_role="guard captain",
            character_traits="disciplined, dry humor",
            character_voice="clipped and formal",
            character_appearance="Tall, scarred",
            character_faction="City Watch",
            character_backstory="Twenty years of service.",
            character_personality="Stoic but caring.",
            character_goals="- maintain order",
            character_current_objectives="- monitor the new arrival",
            character_attitudes="- aldric: -0.1 (neutral)",
            character_secrets="- knows the hidden passage",
            character_narrative_notes="Right hand twitches when lying.",
            world_context="Genre: fantasy",
            observed_facts="Aldric strains against the building.",
            scene_context="Estate courtyard, raining.",
            characters_present="No other characters are present.",
            character_id="guard_17",
            prior_character_responses="No other characters have responded yet.",
            pending_observations_block="",
            acting_character_name="Aldric",
            player_characters_block="- **Aldric** (acting this turn) — scholar. Tall, in rain-darkened traveling clothes.",
        )
        assert "Captain Vero" in result
        assert "guard_17" in result
        assert "hidden passage" in result
        assert "rain-darkened traveling clothes" in result

    def test_render_messages_requires_delimiter(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        # All canonical templates have the delimiter, so this should succeed.
        messages = mgr.render_messages(
            "event_router",
            setting_summary="x", world_lore="x", world_rules="x",
            scene_graph="x", current_scene="x", characters_present="x",
            world_facts="x", hidden_lore="x", hidden_facts="x",
            character_registry="x", user_input="x",
            acting_character_name="x", player_characters_block="x",
            since_last_turn_block="",
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
            world_facts="x", hidden_lore="x", hidden_facts="x",
            character_registry="x", user_input="x",
            acting_character_name="x", player_characters_block="x",
            since_last_turn_block="",
        )
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "prev user"
        assert messages[2]["content"] == "prev assistant"
        assert messages[3]["role"] == "user"
