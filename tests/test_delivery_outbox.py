from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.engine.delivery_outbox import (
    acknowledge_delivery,
    claim_deliveries,
    enqueue_delivery,
    prune_delivery_after_revision,
)
from app.engine.delivery_response import response_from_deliveries
from app.schemas.delivery import DeliveryPayload
from tests.support.factories import checkpoint


def _payload(text: str = "Rendered") -> DeliveryPayload:
    return DeliveryPayload(
        prose=text,
        visual_novel=None,
        asset_reveals=[],
        reaction_prompt_event_id="",
        loot_offer_ids=[],
        commitment_revision_ids=[],
        dice_rolls=[],
        experience_awards=[],
        owner_error="",
    )


def test_delivery_claim_ack_and_knowledge_cutoff() -> None:
    ckpt = checkpoint(session_id="delivery", turn_index=4)
    entry = enqueue_delivery(
        ckpt,
        pov_character_id="alice",
        source_event_ids=["evt_a"],
        highest_event_sequence=7,
        payload=_payload(),
    )
    claimed = claim_deliveries(
        ckpt,
        pov_character_id="alice",
        consumer_id="discord:1",
    )
    assert [item.delivery_id for item in claimed] == [entry.delivery_id]
    assert claimed[0].claim_token

    acknowledged = acknowledge_delivery(
        ckpt,
        delivery_id=entry.delivery_id,
        claim_token=claimed[0].claim_token,
        consumer_id="discord:1",
    )
    assert acknowledged.status == "acknowledged"
    assert ckpt.session.last_acknowledged_event_sequence_by_pov == {"alice": 7}
    assert claim_deliveries(
        ckpt,
        pov_character_id="alice",
        consumer_id="discord:1",
    ) == []


def test_expired_claim_is_reissued_with_stable_delivery_id() -> None:
    ckpt = checkpoint(session_id="delivery")
    entry = enqueue_delivery(
        ckpt,
        pov_character_id="alice",
        source_event_ids=["evt_a"],
        highest_event_sequence=0,
        payload=_payload(),
    )
    now = datetime.now(timezone.utc)
    first = claim_deliveries(
        ckpt,
        pov_character_id="alice",
        consumer_id="cli:first",
        now=now,
    )[0]
    assert claim_deliveries(
        ckpt,
        pov_character_id="alice",
        consumer_id="cli:second",
        now=now + timedelta(seconds=30),
    ) == []
    second = claim_deliveries(
        ckpt,
        pov_character_id="alice",
        consumer_id="cli:second",
        now=now + timedelta(seconds=121),
    )[0]
    assert second.delivery_id == entry.delivery_id
    assert second.claim_token != first.claim_token


def test_ack_rejects_wrong_claim() -> None:
    ckpt = checkpoint()
    entry = enqueue_delivery(
        ckpt,
        pov_character_id="alice",
        source_event_ids=[],
        highest_event_sequence=-1,
        payload=_payload("worker error"),
    )
    claim_deliveries(
        ckpt,
        pov_character_id="alice",
        consumer_id="cli",
    )
    with pytest.raises(ValueError, match="live claim"):
        acknowledge_delivery(
            ckpt,
            delivery_id=entry.delivery_id,
            claim_token="wrong",
            consumer_id="cli",
        )


def test_rewind_prunes_later_delivery_and_ack_cutoff() -> None:
    ckpt = checkpoint(session_id="delivery", turn_index=1)
    first = enqueue_delivery(
        ckpt,
        pov_character_id="alice",
        source_event_ids=["evt_a"],
        highest_event_sequence=1,
        payload=_payload("first"),
    )
    claim = claim_deliveries(
        ckpt,
        pov_character_id="alice",
        consumer_id="cli",
    )[0]
    acknowledge_delivery(
        ckpt,
        delivery_id=first.delivery_id,
        claim_token=claim.claim_token,
        consumer_id="cli",
    )
    ckpt.session.turn_index = 5
    enqueue_delivery(
        ckpt,
        pov_character_id="alice",
        source_event_ids=["evt_b"],
        highest_event_sequence=8,
        payload=_payload("later"),
    )
    assert prune_delivery_after_revision(ckpt, revision=2) == 1
    assert [entry.delivery_id for entry in ckpt.session.delivery_outbox] == [
        first.delivery_id
    ]
    assert ckpt.session.last_acknowledged_event_sequence_by_pov == {"alice": 1}


def test_multi_pov_response_deduplicates_shared_dice_and_xp() -> None:
    ckpt = checkpoint(session_id="delivery", turn_index=2)
    shared_roll = {
        "transaction_id": "tx_a",
        "event_id": "evt_a",
        "roll_id": "attack",
        "actor_id": "alice",
        "total": 17,
    }
    shared_award = {
        "character_id": "alice",
        "amount": 50,
        "source": "goblin",
        "experience_points": 50,
    }
    claims = []
    for pov_id in ("alice", "bob"):
        payload = _payload(f"Rendered for {pov_id}").model_copy(update={
            "dice_rolls": [shared_roll],
            "experience_awards": [shared_award],
        })
        enqueue_delivery(
            ckpt,
            pov_character_id=pov_id,
            source_event_ids=["evt_a"],
            highest_event_sequence=0,
            payload=payload,
        )
        claims.extend(claim_deliveries(
            ckpt,
            pov_character_id=pov_id,
            consumer_id="discord:shared",
        ))

    response = response_from_deliveries(
        session_id="delivery",
        checkpoint_id="ckpt_0002",
        turn_index=2,
        acting_character_id="alice",
        deliveries=claims,
    )

    assert [item.roll_id for item in response.dice_rolls] == ["attack"]
    assert [item.amount for item in response.experience_awards] == [50]
