from scripts.run_dnd_combat_manager_stress import (
    _fact_asserts_condition,
    _scenario_findings,
)


def test_fact_asserts_condition_ignores_negated_condition_claims():
    assert not _fact_asserts_condition(
        "Neither creature is hurt by the fall, and neither is knocked prone.",
        "prone",
    )
    assert not _fact_asserts_condition(
        "The target avoids being restrained and slips clear.",
        "restrained",
    )


def test_forbid_hp_change_ignores_missing_after_hp_when_combat_ends():
    result = {
        "expectations": {"forbid_hp_change": True},
        "before_hp": {"pc_spacer": 33, "void_pirate": 27},
        "after_hp": {},
        "event": {"facts": ["D&D combat ends."]},
        "capture": {"adjudication": {"visible_outcome_facts": []}},
    }

    assert not [
        finding
        for finding in _scenario_findings(result)
        if finding["name"] == "unexpected_hp_change"
    ]


def test_condition_fact_requires_delta_ignores_negated_condition_fact():
    result = {
        "expectations": {
            "condition_fact_requires_delta": [
                {"target_id": "pc_monk", "condition": "prone"},
                {"target_id": "ogre", "condition": "prone"},
            ]
        },
        "event": {
            "facts": [
                "Neither creature is hurt by the fall, and neither is knocked prone."
            ]
        },
        "capture": {"adjudication": {"combat_state_deltas": []}},
    }

    assert not [
        finding
        for finding in _scenario_findings(result)
        if finding["name"] == "condition_fact_missing_state_delta"
    ]
