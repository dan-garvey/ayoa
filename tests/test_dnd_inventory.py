from __future__ import annotations

import pytest

from app.engine import dnd_inventory
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import DndEventRouterOutput
from app.schemas.state import SessionState


def _ckpt() -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id="s",
            character_bindings={"alice": "1", "bob": "2"},
        ),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                mechanics={
                    "ruleset_id": "dnd5e_basic",
                    "dnd5e_sheet": {
                        "statblock": {
                            "inventory": {
                                "items": [
                                    {
                                        "id": "ddb_item_rope",
                                        "name": "Rope",
                                        "kind": "gear",
                                        "quantity": 1,
                                    }
                                ],
                                "currency": {"gp": 2},
                            }
                        }
                    },
                },
            ),
            CharacterRecord(character_id="bob", name="Bob"),
        ],
    )


def _loot_event() -> DndEventRouterOutput:
    return DndEventRouterOutput(
        event_id="evt_loot",
        decision_rationale="test",
        canonical_event={
            "world_adjudication": {"feasible": True},
            "observable_facts": [
                {
                    "text": "Alice opens the chest.",
                    "audience": "all_observers",
                    "visible_to": [],
                }
            ],
        },
        observers=[
            {
                "character_id": "alice",
                "observation_level": "d",
                "response_priority": 1,
            }
        ],
        spawn=[],
        dormant=[],
        cull=[],
        interaction_mode="cat_i",
        combatant_ids=[],
        loot_offer={
            "present": True,
            "source_kind": "container",
            "source_label": "iron chest",
            "visibility": "table",
            "eligible_character_ids": ["alice", "bob"],
            "items": [
                {
                    "item_id": "healing_potion",
                    "name": "Potion of Healing",
                    "kind": "consumable",
                    "quantity": 1,
                    "identified": True,
                    "requires_identification": False,
                    "requires_attunement": False,
                    "consumable": True,
                    "value_gp": 50,
                    "weight": 0.5,
                    "notes": "",
                }
            ],
            "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 12, "pp": 0},
            "notes": "",
        },
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="state_change",
    )


def test_apply_loot_offer_is_idempotent_and_prompts_bound_characters():
    ckpt = _ckpt()
    event = _loot_event()

    prompts = dnd_inventory.apply_loot_offers_from_events(ckpt, [event])
    prompts_again = dnd_inventory.apply_loot_offers_from_events(ckpt, [event])

    assert prompts == {"alice": ["loot_evt_loot"], "bob": ["loot_evt_loot"]}
    assert prompts_again == {}
    assert len(ckpt.session.dnd_inventory_offers) == 1


def test_claim_loot_creates_runtime_overlay_without_mutating_import():
    ckpt = _ckpt()
    dnd_inventory.apply_loot_offers_from_events(ckpt, [_loot_event()])

    result = dnd_inventory.claim_loot(
        ckpt,
        character_id="alice",
        offer_id="loot_evt_loot",
        item_ids=["healing_potion"],
        take_currency=True,
    )

    alice = ckpt.characters[0]
    imported = (
        alice.mechanics["dnd5e_sheet"]["statblock"]["inventory"]["items"]
    )
    runtime = alice.mechanics["dnd5e_runtime"]["inventory"]
    assert imported == [
        {
            "id": "ddb_item_rope",
            "name": "Rope",
            "kind": "gear",
            "quantity": 1,
        }
    ]
    assert [item["name"] for item in runtime["items"]] == [
        "Rope",
        "Potion of Healing",
    ]
    assert runtime["currency"]["gp"] == 14
    assert result["offer_closed"] is True


def test_stale_claim_fails_cleanly():
    ckpt = _ckpt()
    dnd_inventory.apply_loot_offers_from_events(ckpt, [_loot_event()])
    dnd_inventory.claim_loot(
        ckpt,
        character_id="alice",
        offer_id="loot_evt_loot",
        item_ids=["healing_potion"],
        take_currency=False,
    )

    with pytest.raises(ValueError, match="no longer available"):
        dnd_inventory.claim_loot(
            ckpt,
            character_id="bob",
            offer_id="loot_evt_loot",
            item_ids=["healing_potion"],
            take_currency=False,
        )


def test_split_currency_divides_by_bound_eligible_characters():
    ckpt = _ckpt()
    dnd_inventory.apply_loot_offers_from_events(ckpt, [_loot_event()])

    result = dnd_inventory.split_loot_currency(
        ckpt,
        offer_id="loot_evt_loot",
        actor_id="alice",
    )

    assert result["shares"]["alice"]["gp"] == 6
    assert result["shares"]["bob"]["gp"] == 6
    assert ckpt.characters[0].mechanics["dnd5e_runtime"]["inventory"][
        "currency"
    ]["gp"] == 8
    assert ckpt.characters[1].mechanics["dnd5e_runtime"]["inventory"][
        "currency"
    ]["gp"] == 6
