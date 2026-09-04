from __future__ import annotations

import pytest

from app.engine import dnd_inventory
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.imported_treasure import ImportedTreasureValidationError
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content import (
    ContentOverlayState,
    ContentPackState,
    ContentTreasureOverlayState,
)
from app.schemas.events import ObservableFact
from app.schemas.event_router import DndCanonicalEventRecord
from app.schemas.state import SessionState
from tests.support.factories import dnd_canonical_event


def _ckpt(
    *,
    treasure: dict | None = None,
    overlay: ContentTreasureOverlayState | None = None,
) -> CheckpointFile:
    ckpt = CheckpointFile(
        session=SessionState(
            session_id="s",
            character_bindings={"alice": "u1", "bob": "u2"},
        ),
        characters=[
            CharacterRecord(character_id="alice", name="Alice"),
            CharacterRecord(character_id="bob", name="Bob"),
            CharacterRecord(character_id="pip", name="Pip"),
        ],
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    treasures = [treasure or _treasure()]
    overlay_records = {}
    if overlay is not None:
        overlay_records[overlay.overlay_key()] = overlay
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            metadata={"treasures": treasures},
            overlay=ContentOverlayState(treasures=overlay_records),
        )
    }
    return ckpt


def _treasure(**overrides: object) -> dict:
    data: dict[str, object] = {
        "ref": "treasure.cache",
        "content_hash": "hash-treasure-cache",
        "title": "Synthetic Cache",
        "summary": "A reviewed synthetic cache.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "treasure_kind": "container",
        "container_ref": "container.cache",
        "depletion_ref": "depleted.treasure.cache",
        "currency": [{"denomination": "gp", "amount": 12}],
        "items": [
            {
                "item_ref": "item.synthetic_key",
                "name": "Synthetic key",
                "quantity": 1,
                "item_type": "key",
                "value_gp": 0,
            }
        ],
    }
    data.update(overrides)
    return data


def _revealed_overlay(**overrides: object) -> ContentTreasureOverlayState:
    data = {
        "treasure_id": "treasure.cache",
        "content_hash": "hash-treasure-cache",
        "revealed": True,
    }
    data.update(overrides)
    return ContentTreasureOverlayState(**data)


def _no_loot_event() -> DndCanonicalEventRecord:
    event = dnd_canonical_event(
        event_id="evt_search",
        facts=[ObservableFact.all("Alice opens the chest and sees a key.")],
        observer_ids=["alice"],
        interaction_mode="narrative",
        combatant_ids=[],
        loot_offer={
            "present": False,
            "source_kind": "other",
            "source_label": "",
            "visibility": "table",
            "eligible_character_ids": [],
            "items": [],
            "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 0, "pp": 0},
            "notes": "",
        },
    )
    return event


def test_hidden_imported_treasure_does_not_create_loot_offer_from_prose() -> None:
    ckpt = _ckpt()

    prompts = dnd_inventory.apply_loot_offers_from_events(ckpt, [_no_loot_event()])

    assert prompts == {}
    assert ckpt.session.dnd_inventory_offers == []


def test_revealed_imported_treasure_noops_outside_dnd_ruleset() -> None:
    ckpt = _ckpt(overlay=_revealed_overlay())
    ckpt.session.config.settings.ruleset_id = "rules_neutral"

    prompts = dnd_inventory.apply_loot_offers_from_events(ckpt, [])

    assert prompts == {}
    assert ckpt.session.dnd_inventory_offers == []


def test_revealed_imported_treasure_creates_structured_offer_with_refs() -> None:
    ckpt = _ckpt(overlay=_revealed_overlay())

    prompts = dnd_inventory.apply_loot_offers_from_events(ckpt, [])

    offer = ckpt.session.dnd_inventory_offers[0]
    assert prompts == {
        "alice": [offer.offer_id],
        "bob": [offer.offer_id],
    }
    assert offer.source_pack_id == "pack"
    assert offer.source_ref == "treasure.cache"
    assert offer.source_content_hash == "hash-treasure-cache"
    assert offer.source_depletion_ref == "depleted.treasure.cache"
    assert offer.source_kind == "container"
    assert offer.items[0].item_id == "item.synthetic_key"
    assert offer.items[0].name == "Synthetic key"
    assert offer.currency.gp == 12
    overlay = ckpt.session.content_state["pack"].overlay.treasures[
        "treasure.cache::hash-treasure-cache"
    ]
    assert overlay.remaining_ref_ids == ["item.synthetic_key", "currency/gp"]
    assert overlay.depleted is False


def test_revealed_body_and_reward_treasure_choose_source_kind() -> None:
    body_ckpt = _ckpt(
        treasure=_treasure(container_ref="body/guardian"),
        overlay=_revealed_overlay(),
    )
    reward_ckpt = _ckpt(
        treasure=_treasure(
            ref="treasure.quest_reward",
            content_hash="hash-quest-reward",
            treasure_kind="reward",
        ),
        overlay=_revealed_overlay(
            treasure_id="treasure.quest_reward",
            content_hash="hash-quest-reward",
        ),
    )

    dnd_inventory.apply_loot_offers_from_events(body_ckpt, [])
    dnd_inventory.apply_loot_offers_from_events(reward_ckpt, [])

    assert body_ckpt.session.dnd_inventory_offers[0].source_kind == "body"
    assert reward_ckpt.session.dnd_inventory_offers[0].source_kind == "reward"


def test_claiming_revealed_imported_treasure_persists_and_rewinds_overlay(
    tmp_path,
) -> None:
    ckpt = _ckpt(overlay=_revealed_overlay())
    dnd_inventory.apply_loot_offers_from_events(ckpt, [])
    manager = CheckpointManager(str(tmp_path / "sessions"))
    ckpt.session.turn_index = 1
    manager.save(ckpt)

    offer_id = ckpt.session.dnd_inventory_offers[0].offer_id
    result = dnd_inventory.claim_loot(
        ckpt,
        character_id="alice",
        offer_id=offer_id,
        item_ids=[],
        take_currency=True,
        take_all_available=True,
    )
    ckpt.session.turn_index = 2
    manager.save(ckpt)

    assert result["offer_closed"] is True
    claimed = ckpt.session.content_state["pack"].overlay.treasures[
        "treasure.cache::hash-treasure-cache"
    ]
    assert claimed.looted is True
    assert claimed.depleted is True
    assert claimed.claimed_ref_ids == ["item.synthetic_key", "currency/gp"]
    assert claimed.remaining_ref_ids == []

    before_claim = manager.load("s", "ckpt_0001")
    after_claim = manager.load("s", "ckpt_0002")
    before_overlay = before_claim.session.content_state["pack"].overlay.treasures[
        "treasure.cache::hash-treasure-cache"
    ]
    after_overlay = after_claim.session.content_state["pack"].overlay.treasures[
        "treasure.cache::hash-treasure-cache"
    ]
    assert after_claim.session.dnd_inventory_offers[0].source_ref == "treasure.cache"
    assert before_overlay.depleted is False
    assert before_overlay.remaining_ref_ids == ["item.synthetic_key", "currency/gp"]
    assert after_overlay.depleted is True
    assert after_overlay.claimed_ref_ids == ["item.synthetic_key", "currency/gp"]


def test_revealed_unreviewed_treasure_blocks_loudly() -> None:
    ckpt = _ckpt(
        treasure=_treasure(review_status="needs_review", gate_status="flagged"),
        overlay=_revealed_overlay(),
    )

    with pytest.raises(ImportedTreasureValidationError, match="review_status"):
        dnd_inventory.apply_loot_offers_from_events(ckpt, [])
