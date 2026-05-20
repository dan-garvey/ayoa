from app.engine.dnd_combat_harness import (
    _event_summary,
    _fact_asserts_condition,
    _scenario_findings,
)
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication


class _Event:
    event_id = "evt"
    event_kind = "ruleset_resolution"
    decision_rationale = ""
    canonical_event = CanonicalEvent(
        world_adjudication=WorldAdjudication(feasible=True),
        observable_facts=[
            ObservableFact.all("Everyone sees the ogre panic."),
            ObservableFact.only("You see your worst fear.", ["ogre"]),
        ],
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


def test_event_summary_keeps_private_facts_scoped():
    summary = _event_summary(_Event())

    assert summary["facts"] == ["Everyone sees the ogre panic."]
    assert summary["private_facts"] == [
        {"text": "You see your worst fear.", "visible_to": ["ogre"]}
    ]


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


def test_private_fact_check_accepts_failed_save_targets_only():
    result = {
        "expectations": {
            "require_private_fact_for_failed_save_targets": True,
            "private_fact_must_contain_any": ["worst fear"],
            "private_fact_forbid_visible_to_non_failed": True,
        },
        "capture": {
            "flattened_rolls": [
                {
                    "roll_id": "fear_ogre_a",
                    "kind": "saving_throw",
                    "target_id": "ogre_a",
                },
                {
                    "roll_id": "fear_ogre_b",
                    "kind": "saving_throw",
                    "target_id": "ogre_b",
                },
            ],
            "roll_ledger": [
                "saving_throw fear_ogre_a: Wisdom = 8, DC 30",
                "saving_throw fear_ogre_b: Wisdom = 11, DC 30",
            ],
            "adjudication": {
                "private_outcome_facts": [
                    {
                        "text": "Each ogre sees its own worst fear in the spell.",
                        "visible_to": ["ogre_a", "ogre_b"],
                    }
                ],
                "visible_outcome_facts": [],
            },
        },
        "event": {"facts": []},
    }

    assert not [
        finding
        for finding in _scenario_findings(result)
        if finding["name"] in {
            "missing_required_private_outcome_facts",
            "failed_save_targets_missing_private_facts",
            "private_fact_visible_to_non_failed_targets",
        }
    ]


def test_private_fact_check_rejects_non_failed_recipients():
    result = {
        "expectations": {
            "require_private_fact_for_failed_save_targets": True,
            "private_fact_must_contain_any": ["worst fear"],
            "private_fact_forbid_visible_to_non_failed": True,
        },
        "capture": {
            "flattened_rolls": [
                {
                    "roll_id": "fear_ogre",
                    "kind": "saving_throw",
                    "target_id": "ogre",
                },
                {
                    "roll_id": "fear_captain",
                    "kind": "saving_throw",
                    "target_id": "captain",
                },
            ],
            "roll_ledger": [
                "saving_throw fear_ogre: Wisdom = 8, DC 30",
                "saving_throw fear_captain: Wisdom = 31, DC 30",
            ],
            "adjudication": {
                "private_outcome_facts": [
                    {
                        "text": "The victims see their worst fear.",
                        "visible_to": ["ogre", "captain"],
                    }
                ],
                "visible_outcome_facts": [],
            },
        },
        "event": {"facts": []},
    }

    findings = _scenario_findings(result)

    assert [
        finding
        for finding in findings
        if finding["name"] == "private_fact_visible_to_non_failed_targets"
    ]


def test_effect_delta_match_supports_required_conditions():
    result = {
        "expectations": {
            "require_effect_delta_matches": [
                {
                    "operation": "start",
                    "target_id": "ogre",
                    "source_id": "fear",
                    "conditions_include": ["frightened"],
                }
            ]
        },
        "capture": {
            "adjudication": {
                "effect_deltas": [
                    {
                        "operation": "start",
                        "target_id": "ogre",
                        "source_id": "fear",
                        "conditions": ["frightened"],
                    }
                ],
            },
        },
        "event": {"facts": []},
    }

    assert not [
        finding
        for finding in _scenario_findings(result)
        if finding["name"] == "missing_required_effect_delta_match"
    ]
