from app.engine.dnd_combat_harness import (
    _cache_watch_findings,
    _cache_watch_for_call,
    _event_summary,
    _fact_asserts_condition,
    _scenario_findings,
)
from app.schemas.events import ObservableFact
from tests.support.factories import canonical_event


def _event():
    return canonical_event(
        event_id="evt",
        observer_ids=["ogre"],
        facts=[
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
    summary = _event_summary(_event())

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


def test_cache_watch_allows_one_small_cache_block_variance():
    plan_cache_reads_by_scenario = {}
    context = {"index": 1, "name": "sample"}

    _cache_watch_for_call(
        context,
        phase="plan_turn",
        usage={"cache_read_input_tokens": 6016},
        plan_cache_reads_by_scenario=plan_cache_reads_by_scenario,
    )
    watch = _cache_watch_for_call(
        context,
        phase="finalize_outcome",
        usage={"cache_read_input_tokens": 5888},
        plan_cache_reads_by_scenario=plan_cache_reads_by_scenario,
    )

    assert watch["finalize_cache_read_delta_from_plan"] == -128
    assert watch["finalize_below_plan_cache_read"] is False
    assert _cache_watch_findings([{"cache_watch": watch}]) == []


def test_cache_watch_flags_large_finalize_cache_drop():
    plan_cache_reads_by_scenario = {}
    context = {"index": 1, "name": "sample"}

    _cache_watch_for_call(
        context,
        phase="plan_turn",
        usage={"cache_read_input_tokens": 6016},
        plan_cache_reads_by_scenario=plan_cache_reads_by_scenario,
    )
    watch = _cache_watch_for_call(
        context,
        phase="finalize_outcome",
        usage={"cache_read_input_tokens": 5500},
        plan_cache_reads_by_scenario=plan_cache_reads_by_scenario,
    )

    assert watch["finalize_below_plan_cache_read"] is True
    assert _cache_watch_findings([{
        "raw_call_index": 2,
        "phase": "finalize_outcome",
        "scenario": {"name": "sample"},
        "cache_watch": watch,
    }])


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


def test_private_fact_target_check_accepts_scoped_real_fiction():
    result = {
        "expectations": {
            "require_private_fact_for_targets": ["orc"],
            "private_fact_must_contain_any": ["iron portcullis"],
            "private_fact_forbid_visible_to_non_targets": True,
            "private_fact_forbid_contains": ["illusion"],
            "forbid_fact_contains": ["iron portcullis"],
        },
        "capture": {
            "adjudication": {
                "private_outcome_facts": [
                    {
                        "text": "An iron portcullis seals the east arch.",
                        "visible_to": ["orc"],
                    }
                ],
                "visible_outcome_facts": ["Sera focuses magic toward the arch."],
            },
        },
        "event": {"facts": ["Sera focuses magic toward the arch."]},
    }

    assert not [
        finding
        for finding in _scenario_findings(result)
        if finding["name"] in {
            "missing_required_private_outcome_facts",
            "private_fact_targets_missing",
            "private_fact_visible_to_unexpected_targets",
            "forbidden_private_fact",
            "forbidden_visible_fact",
        }
    ]


def test_private_fact_target_check_rejects_leaked_illusion_language():
    result = {
        "expectations": {
            "require_private_fact_for_targets": ["orc"],
            "private_fact_must_contain_any": ["iron portcullis"],
            "private_fact_forbid_visible_to_non_targets": True,
            "private_fact_forbid_contains": ["illusion"],
        },
        "capture": {
            "adjudication": {
                "private_outcome_facts": [
                    {
                        "text": "An illusion of an iron portcullis seals the arch.",
                        "visible_to": ["orc", "captain"],
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
        if finding["name"] == "private_fact_visible_to_unexpected_targets"
    ]
    assert [
        finding for finding in findings
        if finding["name"] == "forbidden_private_fact"
    ]


def test_action_and_spatial_match_checks_catch_control_contract():
    result = {
        "expectations": {
            "forbid_action_source_ids": ["longsword"],
            "require_action_matches": {
                "actor_id": "pc_guard",
                "source_type": "effect",
                "source_id": "command",
                "effect_id": "command_flee_pc_guard",
                "use_mode": "sustain",
            },
            "require_spatial_delta_matches": {
                "kind": "move_token",
                "target_id": "pc_guard",
            },
            "forbid_spatial_delta_kinds": ["add_area"],
        },
        "capture": {
            "turn_plan": {
                "actions": [
                    {
                        "actor_id": "pc_guard",
                        "source_type": "effect",
                        "source_id": "command",
                        "effect_id": "command_flee_pc_guard",
                        "use_mode": "sustain",
                    }
                ],
            },
            "adjudication": {
                "spatial_deltas": [
                    {"kind": "move_token", "target_id": "pc_guard"}
                ],
            },
        },
        "event": {"facts": []},
    }

    assert not [
        finding
        for finding in _scenario_findings(result)
        if finding["name"] in {
            "forbidden_action_source_used",
            "missing_required_action_match",
            "missing_required_spatial_delta_match",
            "forbidden_spatial_delta_kind",
        }
    ]


def test_forbid_router_observed_facts_rejects_private_illusion_duplication():
    result = {
        "expectations": {"forbid_router_observed_facts": True},
        "capture": {
            "adjudication": {
                "router_observed_facts": [
                    {
                        "fact": "Sera created a private illusion.",
                        "salience": "notable",
                        "reason": "It may matter after combat.",
                    }
                ],
            },
        },
        "event": {"facts": []},
    }

    assert [
        finding for finding in _scenario_findings(result)
        if finding["name"] == "unexpected_router_observed_facts"
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
