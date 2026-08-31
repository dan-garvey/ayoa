"""Contracts for sparse actor-fact character authoring."""

import re

from app.engine.prompt_manager import PromptManager
from app.schemas.characters import ActorFact, ActorRecord, CharacterRecord
from app.schemas.takeover import AuthoredCharacter


def _generation_values() -> dict[str, str]:
    return {
        "setting_summary": "A salt-marsh city under a long eclipse.",
        "location_context": "A locked archive below the tide line.",
        "character_id": "archive_actor_unique",
        "location": "tide_archive",
        "spawn_seed": (
            "role: salvage clerk\n"
            "reason: the sealed index was found under the archive\n"
            "objectives: find who removed the last page"
        ),
        "generation_context": "GENERATION_CONTEXT_UNIQUE",
        "ruleset_generation_context": "RULESET_CONTEXT_UNIQUE",
        "existing_characters": "EXISTING_CAST_UNIQUE",
        "ruleset_generation_instructions": "",
        "ruleset_output_schema_suffix": "",
    }


def _authored(*, facts: list[ActorFact] | None = None) -> AuthoredCharacter:
    return AuthoredCharacter(
        name="Iria Venn",
        location="tide_archive",
        role="salvage clerk",
        appearance="A narrow-faced adult with gray braids and a copper tooth.",
        public_context="Archive staff recognize her tide-marked ledger.",
        default_loadout="Oilskin coat, tide-marked ledger, and a cracked blue lens.",
        faction="Archive of Low Water",
        actor=ActorRecord(
            may_act_offstage=True,
            facts=facts
            if facts is not None
            else [
                ActorFact(
                    origin="lived",
                    text=(
                        "You found your brother's boat empty beneath the west "
                        "sluice and kept its cracked blue lens."
                    ),
                ),
                ActorFact(
                    origin="told",
                    text=(
                        "You were told by Mara's crew that the west sluice was "
                        "inspected before the tide turned."
                    ),
                ),
                ActorFact(
                    origin="inferred",
                    text=(
                        "You suspect salt on the binding means the missing page "
                        "was removed before the inspection."
                    ),
                ),
            ],
        ),
        router_summary="",
    )


def test_generation_prompt_keeps_spawn_inputs_in_the_volatile_tail():
    messages = PromptManager("app/prompts").render_messages(
        "character_gen", **_generation_values()
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "archive_actor_unique" not in system
    assert "GENERATION_CONTEXT_UNIQUE" not in system
    assert "RULESET_CONTEXT_UNIQUE" not in system
    assert "EXISTING_CAST_UNIQUE" not in system
    assert "archive_actor_unique" in user
    assert "GENERATION_CONTEXT_UNIQUE" in user
    assert "RULESET_CONTEXT_UNIQUE" in user
    assert "EXISTING_CAST_UNIQUE" in user


def test_authored_actor_record_preserves_public_identity_and_provenance():
    authored = _authored()
    record = authored.to_record(character_id="archive_actor_unique")

    assert isinstance(record, CharacterRecord)
    assert record.public_sheet.public_context == authored.public_context
    assert record.visuals.default_loadout == authored.default_loadout
    assert record.actor is not None
    assert record.actor.may_act_offstage is True
    assert [fact.origin.value for fact in record.actor.facts] == [
        "lived",
        "told",
        "inferred",
    ]
    assert all(fact.text.startswith("You ") for fact in record.actor.facts)


def test_zero_actor_facts_are_valid_and_stay_empty():
    authored = _authored(facts=[])
    record = authored.to_record(character_id="archive_actor_unique")

    assert record.actor is not None
    assert record.actor.facts == []


def test_character_generation_prompt_retires_profile_and_review_vocabulary():
    text = PromptManager("app/prompts").render(
        "character_gen",
        **_generation_values(),
    )
    forbidden = (
        r"\bbackstory\b",
        r"\bpersonality\b",
        r"\bknown_context\b",
        r"\bprivate_state\b",
        r"\bcurrent_objectives\b",
        r"\bintentions_enabled\b",
        r"\brubric\b",
        r"\bepistemology\b",
        r"\binterpersonal objective\b",
        r"\bswappability\b",
        r"\bstatus negotiation\b",
        r"\bdramatic debt\b",
        r"\bsubtext analysis\b",
    )
    for pattern in forbidden:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None
