"""Durable per-POV delivery claims shared by every frontend."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Iterable

from app.schemas.checkpoint import CheckpointFile
from app.schemas.delivery import DeliveryOutboxEntry, DeliveryPayload


DEFAULT_CLAIM_LEASE_SECONDS = 120


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def delivery_id_for(
    *,
    session_id: str,
    pov_character_id: str,
    source_event_ids: Iterable[str],
    created_revision: int,
) -> str:
    basis = "\x1f".join((
        session_id,
        pov_character_id,
        str(created_revision),
        *source_event_ids,
    ))
    return "delivery_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def enqueue_delivery(
    checkpoint: CheckpointFile,
    *,
    pov_character_id: str,
    source_event_ids: list[str],
    highest_event_sequence: int,
    payload: DeliveryPayload,
) -> DeliveryOutboxEntry:
    """Append one idempotent pending delivery after a successful POV render."""

    delivery_id = delivery_id_for(
        session_id=checkpoint.session.session_id,
        pov_character_id=pov_character_id,
        source_event_ids=source_event_ids,
        created_revision=checkpoint.session.turn_index + 1,
    )
    existing = next(
        (
            entry
            for entry in checkpoint.session.delivery_outbox
            if entry.delivery_id == delivery_id
        ),
        None,
    )
    if existing is not None:
        if existing.payload != payload:
            raise RuntimeError(
                "stable delivery id was reused for a different payload"
            )
        return existing
    entry = DeliveryOutboxEntry(
        delivery_id=delivery_id,
        pov_character_id=pov_character_id,
        source_event_ids=source_event_ids,
        highest_event_sequence=highest_event_sequence,
        created_revision=checkpoint.session.turn_index + 1,
        payload=payload,
        status="pending",
        claim_token="",
        claimed_by="",
        claimed_at="",
        attempts=0,
        acknowledged_at="",
    )
    checkpoint.session.delivery_outbox.append(entry)
    return entry


def claim_deliveries(
    checkpoint: CheckpointFile,
    *,
    pov_character_id: str,
    consumer_id: str,
    limit: int = 20,
    now: datetime | None = None,
    lease_seconds: int = DEFAULT_CLAIM_LEASE_SECONDS,
) -> list[DeliveryOutboxEntry]:
    """Lease pending or expired entries in canonical delivery order."""

    if not consumer_id.strip():
        raise ValueError("delivery consumer id must not be blank")
    instant = (now or _utcnow()).astimezone(timezone.utc)
    expiry = instant - timedelta(seconds=max(1, lease_seconds))
    claimed: list[DeliveryOutboxEntry] = []
    candidates = sorted(
        checkpoint.session.delivery_outbox,
        key=lambda entry: (
            entry.created_revision,
            entry.highest_event_sequence,
            entry.delivery_id,
        ),
    )
    for entry in candidates:
        if len(claimed) >= max(1, limit):
            break
        if entry.pov_character_id != pov_character_id:
            continue
        if entry.status == "acknowledged":
            continue
        if entry.status == "claimed":
            claimed_at = _parse_time(entry.claimed_at)
            if claimed_at is not None and claimed_at > expiry:
                continue
        entry.status = "claimed"
        entry.claim_token = secrets.token_urlsafe(18)
        entry.claimed_by = consumer_id
        entry.claimed_at = instant.isoformat()
        entry.attempts += 1
        claimed.append(entry.model_copy(deep=True))
    return claimed


def acknowledge_delivery(
    checkpoint: CheckpointFile,
    *,
    delivery_id: str,
    claim_token: str,
    consumer_id: str,
    now: datetime | None = None,
) -> DeliveryOutboxEntry:
    entry = next(
        (
            candidate
            for candidate in checkpoint.session.delivery_outbox
            if candidate.delivery_id == delivery_id
        ),
        None,
    )
    if entry is None:
        raise ValueError("unknown delivery id")
    if entry.status == "acknowledged":
        return entry
    if (
        entry.status != "claimed"
        or not claim_token
        or entry.claim_token != claim_token
        or entry.claimed_by != consumer_id
    ):
        raise ValueError("delivery acknowledgement does not match its live claim")
    entry.status = "acknowledged"
    entry.acknowledged_at = (now or _utcnow()).astimezone(timezone.utc).isoformat()
    current = checkpoint.session.last_acknowledged_event_sequence_by_pov.get(
        entry.pov_character_id,
        -1,
    )
    checkpoint.session.last_acknowledged_event_sequence_by_pov[
        entry.pov_character_id
    ] = max(current, entry.highest_event_sequence)
    return entry


def release_consumer_claims(
    checkpoint: CheckpointFile,
    *,
    consumer_id: str,
) -> int:
    released = 0
    for entry in checkpoint.session.delivery_outbox:
        if entry.status != "claimed" or entry.claimed_by != consumer_id:
            continue
        entry.status = "pending"
        entry.claim_token = ""
        entry.claimed_by = ""
        entry.claimed_at = ""
        released += 1
    return released


def prune_delivery_after_revision(
    checkpoint: CheckpointFile,
    *,
    revision: int,
) -> int:
    before = len(checkpoint.session.delivery_outbox)
    checkpoint.session.delivery_outbox = [
        entry
        for entry in checkpoint.session.delivery_outbox
        if entry.created_revision <= revision
    ]
    checkpoint.session.narrator_render_jobs = [
        job
        for job in checkpoint.session.narrator_render_jobs
        if job.created_revision <= revision
    ]
    surviving_sequences: dict[str, int] = {}
    for entry in checkpoint.session.delivery_outbox:
        if entry.status != "acknowledged":
            continue
        surviving_sequences[entry.pov_character_id] = max(
            surviving_sequences.get(entry.pov_character_id, -1),
            entry.highest_event_sequence,
        )
    checkpoint.session.last_acknowledged_event_sequence_by_pov = surviving_sequences
    return before - len(checkpoint.session.delivery_outbox)
