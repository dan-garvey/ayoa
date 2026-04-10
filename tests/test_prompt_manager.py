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
        assert "narrator_phase1" in versions
        assert "narrator_phase2" in versions
        assert "discriminator" in versions
        assert "agent" in versions
        assert "character_gen" in versions

    def test_narrator_phase1_renders(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        result = mgr.render(
            "narrator_phase1",
            setting_summary="Genre: fantasy\nTone: dark",
            world_lore="The kingdom has been at war.",
            world_rules="No magic. Human baseline strength.",
            scene_context="A stone courtyard in the rain.",
            characters_present="- Captain Vero (guard): disciplined",
            recent_transcript="(none)",
            world_facts="The courtyard is wet.",
            narrative_rules="Write concise prose.",
            user_input="I try to lift the building.",
        )
        assert "I try to lift the building" in result
        assert "No magic" in result
        assert "Captain Vero" in result

    def test_agent_renders(self):
        mgr = PromptManager(prompts_dir="app/prompts")
        result = mgr.render(
            "agent",
            character_name="Captain Vero",
            character_role="guard captain",
            character_sheet="Disciplined, dry humor",
            character_memories="(none)",
            observed_facts="The user strains against the building.",
            scene_context="Estate courtyard, raining.",
            character_voice="clipped and formal",
            character_id="guard_17",
        )
        assert "Captain Vero" in result
        assert "guard_17" in result
