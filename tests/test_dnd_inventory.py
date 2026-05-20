from __future__ import annotations

import pytest

from app.engine import dnd_inventory, dnd_presentation
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_inventory import DndLootOfferItem
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
            CharacterRecord(character_id="pip", name="Pip"),
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
                "routing_role": "observe_only",
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


def test_narrative_mundane_gear_handoff_becomes_claimable_loot():
    ckpt = _ckpt()
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    data = _loot_event().model_dump()
    data["event_id"] = "evt_gear_handoff"
    data["canonical_event"]["observable_facts"] = [
        {
            "text": (
                "Korva returns from the ledger counter carrying a small bundle "
                "of Hall-stock travel gear: coiled rope, two bedrolls, ration "
                "wraps, a torch bundle, and a folded rough map sheet. She sets "
                "the bundle down within reach of Alice and Bob; a healer's kit "
                "in a leather roll lands on top."
            ),
            "audience": "all_observers",
            "visible_to": [],
        }
    ]
    data["loot_offer"] = {
        "present": False,
        "source_kind": "other",
        "source_label": "",
        "visibility": "table",
        "eligible_character_ids": [],
        "items": [],
        "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 0, "pp": 0},
        "notes": "",
    }

    prompts = dnd_inventory.apply_loot_offers_from_events(
        ckpt,
        [DndEventRouterOutput(**data)],
    )

    offer = ckpt.session.dnd_inventory_offers[0]
    assert prompts == {"alice": ["loot_evt_gear_handoff"], "bob": ["loot_evt_gear_handoff"]}
    assert offer.source_kind == "handoff"
    assert offer.source_label == "Hall stock"
    items = {item.item_id: item for item in offer.items}
    assert set(items) == {
        "healers_kit",
        "rope",
        "bedroll",
        "rations",
        "torch_bundle",
        "rough_map",
    }
    assert items["bedroll"].quantity == 2

    result = dnd_inventory.claim_loot(
        ckpt,
        character_id="alice",
        offer_id="loot_evt_gear_handoff",
        item_ids=[],
        take_all_available=True,
    )

    assert [item["name"] for item in result["claimed_items"]] == [
        "Healer's Kit",
        "Rope",
        "Bedroll",
        "Rations",
        "Torch Bundle",
        "Rough Map",
    ]
    runtime_items = ckpt.characters[0].mechanics["dnd5e_runtime"]["inventory"][
        "items"
    ]
    assert "Healer's Kit" in {item["name"] for item in runtime_items}


def test_narrative_gear_handoff_inference_is_dnd_only():
    ckpt = _ckpt()
    data = _loot_event().model_dump()
    data["event_id"] = "evt_gear_handoff"
    data["canonical_event"]["observable_facts"] = [
        {
            "text": "Korva sets a rope and healer's kit down within reach.",
            "audience": "all_observers",
            "visible_to": [],
        }
    ]
    data["loot_offer"] = {
        "present": False,
        "source_kind": "other",
        "source_label": "",
        "visibility": "table",
        "eligible_character_ids": [],
        "items": [],
        "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 0, "pp": 0},
        "notes": "",
    }

    prompts = dnd_inventory.apply_loot_offers_from_events(
        ckpt,
        [DndEventRouterOutput(**data)],
    )

    assert prompts == {}
    assert ckpt.session.dnd_inventory_offers == []


def test_narrative_handoff_does_not_offer_animal_tack():
    ckpt = _ckpt()
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    data = _loot_event().model_dump()
    data["event_id"] = "evt_mule_rope"
    data["canonical_event"]["observable_facts"] = [
        {
            "text": (
                "Korva gathers the supply mule's lead rope, passes the reins "
                "to Alice, and keeps the animal steady by the stable door."
            ),
            "audience": "all_observers",
            "visible_to": [],
        }
    ]
    data["loot_offer"] = {
        "present": False,
        "source_kind": "other",
        "source_label": "",
        "visibility": "table",
        "eligible_character_ids": [],
        "items": [],
        "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 0, "pp": 0},
        "notes": "",
    }

    prompts = dnd_inventory.apply_loot_offers_from_events(
        ckpt,
        [DndEventRouterOutput(**data)],
    )

    assert prompts == {}
    assert ckpt.session.dnd_inventory_offers == []


def test_narrative_handoff_does_not_offer_owned_gear():
    ckpt = _ckpt()
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    data = _loot_event().model_dump()
    data["event_id"] = "evt_owned_rope"
    data["canonical_event"]["observable_facts"] = [
        {
            "text": (
                "Garrick pulls rope from his own pack, passes the loose end "
                "around Dace's wrists, and ties it off."
            ),
            "audience": "all_observers",
            "visible_to": [],
        }
    ]
    data["loot_offer"] = {
        "present": False,
        "source_kind": "other",
        "source_label": "",
        "visibility": "table",
        "eligible_character_ids": [],
        "items": [],
        "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 0, "pp": 0},
        "notes": "",
    }

    prompts = dnd_inventory.apply_loot_offers_from_events(
        ckpt,
        [DndEventRouterOutput(**data)],
    )

    assert prompts == {}
    assert ckpt.session.dnd_inventory_offers == []


def test_loot_and_inventory_lines_do_not_duplicate_kind_suffix():
    item = DndLootOfferItem(
        item_id="scaled_vest",
        name="scaled vest (armor)",
        kind="armor",
        quantity=1,
        identified=True,
        requires_identification=False,
        requires_attunement=False,
        consumable=False,
        value_gp=0,
        weight=20,
        notes="",
    )

    assert dnd_presentation.loot_item_line(item) == "scaled vest (armor)"
    assert dnd_presentation.inventory_item_line(
        {
            "id": "scaled_vest",
            "name": "scaled vest (armor)",
            "kind": "armor",
            "quantity": 1,
        },
        include_id=False,
    ) == "scaled vest (armor)"


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

    with pytest.raises(ValueError, match="Use /loot list"):
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


def test_take_all_computes_remaining_items_at_claim_time():
    ckpt = _ckpt()
    event = _loot_event()
    data = event.model_dump()
    data["loot_offer"]["items"].append({
        "item_id": "silver_ring",
        "name": "Silver Ring",
        "kind": "gear",
        "quantity": 1,
        "identified": True,
        "requires_identification": False,
        "requires_attunement": False,
        "consumable": False,
        "value_gp": 5,
        "weight": 0,
        "notes": "",
    })
    dnd_inventory.apply_loot_offers_from_events(
        ckpt,
        [DndEventRouterOutput(**data)],
    )
    dnd_inventory.claim_loot(
        ckpt,
        character_id="bob",
        offer_id="loot_evt_loot",
        item_ids=["healing_potion"],
        take_currency=False,
    )

    result = dnd_inventory.claim_loot(
        ckpt,
        character_id="alice",
        offer_id="loot_evt_loot",
        item_ids=[],
        take_currency=True,
        take_all_available=True,
    )

    assert [item["name"] for item in result["claimed_items"]] == ["Silver Ring"]
    assert result["claimed_currency"]["gp"] == 12


def test_offer_eligibility_filters_to_bound_player_characters():
    ckpt = _ckpt()
    event = _loot_event()
    data = event.model_dump()
    data["loot_offer"]["eligible_character_ids"] = ["alice", "pip"]

    prompts = dnd_inventory.apply_loot_offers_from_events(
        ckpt,
        [DndEventRouterOutput(**data)],
    )

    offer = ckpt.session.dnd_inventory_offers[0]
    assert offer.eligible_character_ids == ["alice"]
    assert prompts == {"alice": ["loot_evt_loot"]}


def test_departed_character_closes_orphaned_offer():
    ckpt = _ckpt()
    event = _loot_event()
    data = event.model_dump()
    data["loot_offer"]["eligible_character_ids"] = ["alice"]
    dnd_inventory.apply_loot_offers_from_events(
        ckpt,
        [DndEventRouterOutput(**data)],
    )

    del ckpt.session.character_bindings["alice"]
    changed = dnd_inventory.remove_character_from_loot_offers(ckpt, "alice")

    assert changed >= 1
    assert ckpt.session.dnd_inventory_offers[0].status == "closed"


def test_prune_inventory_offers_keeps_bounded_closed_tail():
    ckpt = _ckpt()
    for idx in range(5):
        offer = _loot_event()
        data = offer.model_dump()
        data["event_id"] = f"evt_loot_{idx}"
        dnd_inventory.apply_loot_offers_from_events(
            ckpt,
            [DndEventRouterOutput(**data)],
        )
        ckpt.session.dnd_inventory_offers[-1].status = "closed"

    removed = dnd_inventory.prune_inventory_offers(ckpt, max_closed=2)

    assert removed == 3
    assert [offer.offer_id for offer in ckpt.session.dnd_inventory_offers] == [
        "loot_evt_loot_3",
        "loot_evt_loot_4",
    ]
