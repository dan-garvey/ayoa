"""Tests for knowledge isolation validators."""

import pytest

from app.engine.validators import (
    extract_entities,
    extract_text_from_output,
    build_known_entities,
    validate_agent_output,
    validate_all_outputs,
    LeakageFlag,
)
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import (
    CharacterRecord,
    PublicSheet,
    PrivateState,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import LocationState, SessionState, StorySetting, WorldState


# --- Fixtures ---

@pytest.fixture
def checkpoint():
    return CheckpointFile(
        session=SessionState(session_id="test"),
        world_state=WorldState(
            locations=LocationState(current_scene_id="courtyard"),
            facts=[
                "The Covenant banned cross-racial procreation.",
                "Article Nineteen is enforced by death.",
            ],
            setting=StorySetting(
                premise="Human heir navigates a magical academy.",
            ),
        ),
        characters=[
            CharacterRecord(
                character_id="guard_17",
                name="Captain Vero",
                location="courtyard",
                public_sheet=PublicSheet(role="guard captain"),
                private_state=PrivateState(
                    secrets=["knows about the hidden passage beneath the armory"],
                ),
                backstory="Served the estate for twenty years.",
            ),
            CharacterRecord(
                character_id="spy_01",
                name="Lady Nightshade",
                location="tower",
                public_sheet=PublicSheet(role="visiting diplomat"),
                private_state=PrivateState(
                    secrets=["is actually an assassin sent to kill the heir"],
                ),
                backstory="Born in the southern kingdoms.",
            ),
        ],
    )


@pytest.fixture
def guard_character(checkpoint):
    return checkpoint.characters[0]  # Captain Vero


@pytest.fixture
def spy_character(checkpoint):
    return checkpoint.characters[1]  # Lady Nightshade


# --- Entity extraction tests ---

class TestExtractEntities:
    def test_basic_entities(self):
        entities = extract_entities("Captain Vero stepped into the Great Hall.")
        assert "vero" in entities or "captain vero" in entities
        assert "captain" in entities

    def test_multi_word(self):
        entities = extract_entities("She mentioned Article Nineteen casually.")
        assert "article" in entities

    def test_no_entities(self):
        entities = extract_entities("the quick brown fox jumped over the lazy dog")
        assert len(entities) == 0


class TestExtractTextFromOutput:
    def test_returns_public_text(self):
        output = CharacterAgentOutput(
            character_id="test",
            public_text='He steps forward. "Hello there." A warm smile.',
            intent="Probing the visitor's intent.",
        )
        text = extract_text_from_output(output)
        assert "steps forward" in text
        assert "Hello there" in text
        assert "warm smile" in text
        # Intent (the trailing parenthetical contents) MUST NOT bleed into
        # the validator's input — leakage detection runs on the public
        # surface only.
        assert "Probing" not in text


# --- Known entities tests ---

class TestBuildKnownEntities:
    def test_includes_own_name(self, guard_character, checkpoint):
        known = build_known_entities(guard_character, [], checkpoint)
        assert "vero" in known
        assert "captain" in known

    def test_includes_observed_facts(self, guard_character, checkpoint):
        facts = ["The player looks at Captain Vero."]
        known = build_known_entities(guard_character, facts, checkpoint)
        assert "player" in known

    def test_includes_world_facts(self, guard_character, checkpoint):
        known = build_known_entities(guard_character, [], checkpoint)
        assert "covenant" in known or "article" in known


# --- Validation tests ---

class TestValidateAgentOutput:
    def test_clean_output_passes(self, guard_character, checkpoint):
        output = CharacterAgentOutput(
            character_id="guard_17",
            public_text='He adjusts his sword belt. "Move along, nothing to see here."',
            intent="Keep order; routine deflection.",
        )
        result = validate_agent_output(
            output, guard_character, ["Player stands in the courtyard."], checkpoint
        )
        assert result.passed is True
        assert len(result.flags) == 0

    def test_references_unobserved_character(self, guard_character, checkpoint):
        """Guard references Nightshade by name without observing her."""
        output = CharacterAgentOutput(
            character_id="guard_17",
            public_text='"I saw Nightshade lurking by the tower."',
            intent="",
        )
        result = validate_agent_output(
            output, guard_character,
            ["Player looks around."],
            checkpoint,
        )
        # Nightshade is not in guard's observation set
        assert result.passed is False
        assert any("nightshade" in f.leaked_text.lower() for f in result.flags)


class TestValidateAllOutputs:
    def test_multiple_outputs(self, checkpoint):
        outputs = [
            CharacterAgentOutput(
                character_id="guard_17",
                public_text='"All clear."',
                intent="",
            ),
            CharacterAgentOutput(
                character_id="spy_01",
                public_text='"Good evening."',
                intent="",
            ),
        ]
        observer_facts = {
            "guard_17": ["Player enters courtyard."],
            "spy_01": ["Player enters courtyard."],
        }
        results = validate_all_outputs(outputs, checkpoint, observer_facts)
        assert len(results) == 2
