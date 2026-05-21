from __future__ import annotations

import hashlib
import json
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.schemas.content import (
    ContentFrontState,
    ContentPackState,
    ContentVillainState,
    PendingContentSignal,
)


FRONT_RUNTIME_METADATA_KEY = "front_runtime"
_PUBLIC_SIGNAL_VISIBILITIES = {"public", "semi_public"}


@dataclass(frozen=True)
class FrontSignalUpdate:
    queued_signal: PendingContentSignal | None
    suppressed_reason: str = ""

    @property
    def queued(self) -> bool:
        return self.queued_signal is not None


def queue_front_signal_from_consequence(
    content_state: MutableMapping[str, ContentPackState],
    *,
    pack_id: str,
    front_id: str,
    source_event_id: str,
    villain_id: str = "",
    actor_id: str = "",
    known: str | Sequence[str] = (),
    pressure: str = "",
    summary: str = "",
    consequence_visibility: str = "public",
    now_s: int = 0,
    cooldown_until_s: int | None = None,
    restraint: str = "",
    restraint_until_s: int | None = None,
    active_plan: str = "",
    hidden_plan: str = "",
    priority: int = 0,
) -> FrontSignalUpdate:
    """Record front knowledge and queue one compact router-only signal.

    The helper deliberately stores durable runtime state in the pack metadata
    instead of adding generic story-engine schema fields for front mechanics.
    `hidden_plan` is accepted as a guardrail for callers working from rich
    content sources, but is never persisted or formatted.
    """

    pack_key = _clean_text(pack_id)
    front_key = _clean_text(front_id)
    event_key = _clean_text(source_event_id)
    if not pack_key:
        raise ValueError("pack_id is required")
    if not front_key:
        raise ValueError("front_id is required")
    if not event_key:
        raise ValueError("source_event_id is required")

    visibility = _clean_token(consequence_visibility).replace("-", "_")
    if visibility not in _PUBLIC_SIGNAL_VISIBILITIES:
        return FrontSignalUpdate(
            queued_signal=None,
            suppressed_reason="visibility",
        )

    pack_state = content_state.setdefault(
        pack_key,
        ContentPackState(pack_id=pack_key),
    )
    if not pack_state.pack_id:
        pack_state.pack_id = pack_key

    front_state = pack_state.fronts.setdefault(
        front_key,
        ContentFrontState(front_id=front_key),
    )
    if not front_state.status:
        front_state.status = "active"

    villain_key = _clean_text(villain_id)
    if villain_key:
        villain_state = pack_state.villains.setdefault(
            villain_key,
            ContentVillainState(villain_id=villain_key),
        )
        if not villain_state.status:
            villain_state.status = "active"
        _append_unique(front_state.villain_ids, [villain_key])
        _append_unique(villain_state.front_ids, [front_key])

    runtime = _front_runtime(pack_state)
    front_runtime = _runtime_entry(runtime, "fronts", front_key)
    villain_runtime = (
        _runtime_entry(runtime, "villains", villain_key) if villain_key else None
    )

    knowledge = _clean_items(known)
    _record_knowledge(
        front_runtime,
        source_event_id=event_key,
        known=knowledge,
        visibility=visibility,
        now_s=now_s,
    )
    if villain_runtime is not None:
        _record_knowledge(
            villain_runtime,
            source_event_id=event_key,
            known=knowledge,
            visibility=visibility,
            now_s=now_s,
        )

    safe_active_plan = _clean_text(active_plan)
    if safe_active_plan:
        _upsert_active_plan(
            front_runtime,
            plan=safe_active_plan,
            source_event_id=event_key,
            actor_id=_clean_text(actor_id) or villain_key,
        )
        if villain_runtime is not None:
            _upsert_active_plan(
                villain_runtime,
                plan=safe_active_plan,
                source_event_id=event_key,
                actor_id=_clean_text(actor_id) or villain_key,
            )

    suppressed = _active_suppression(
        (front_runtime, villain_runtime),
        now_s=now_s,
    )
    if suppressed:
        _append_unique(
            front_runtime.setdefault("suppressed_source_event_ids", []),
            [event_key],
        )
        if villain_runtime is not None:
            _append_unique(
                villain_runtime.setdefault("suppressed_source_event_ids", []),
                [event_key],
            )
        return FrontSignalUpdate(
            queued_signal=None,
            suppressed_reason=suppressed,
        )

    cooldown_until = _clean_nonnegative_int(cooldown_until_s)
    if cooldown_until is not None:
        front_runtime["cooldown_until_s"] = cooldown_until
        if villain_runtime is not None:
            villain_runtime["cooldown_until_s"] = cooldown_until

    restraint_text = _clean_text(restraint)
    restraint_until = _clean_nonnegative_int(restraint_until_s)
    if restraint_text or restraint_until is not None:
        restraint_metadata = {
            key: value
            for key, value in {
                "reason": restraint_text,
                "until_s": restraint_until,
                "source_event_id": event_key,
            }.items()
            if value not in ("", None)
        }
        front_runtime["restraint"] = restraint_metadata
        if villain_runtime is not None:
            villain_runtime["restraint"] = dict(restraint_metadata)

    actor = _clean_text(actor_id) or villain_key
    allowed_summary = _clean_text(summary)
    allowed_pressure = _clean_text(pressure)
    signal_payload: dict[str, Any] = {
        "kind": "front_signal",
        "visibility": "hidden",
        "actor": actor,
        "knows": knowledge,
        "pressure": allowed_pressure,
        "summary": allowed_summary,
        "front_id": front_key,
        "villain_id": villain_key,
        "source_event_ids": [event_key],
        "consequence_visibility": visibility,
    }
    signal_payload = {
        key: value
        for key, value in signal_payload.items()
        if value not in ("", [], None)
    }
    signal = PendingContentSignal(
        signal_id=_signal_id(
            pack_id=pack_key,
            front_id=front_key,
            villain_id=villain_key,
            source_event_id=event_key,
        ),
        pack_id=pack_key,
        ref_id=_front_ref(front_key),
        content_hash=_signal_hash(
            source_event_id=event_key,
            ref_id=_front_ref(front_key),
            actor=actor,
            known=knowledge,
            pressure=allowed_pressure,
            summary=allowed_summary,
        ),
        reason=allowed_summary or allowed_pressure or "Front pressure changed.",
        source_event_id=event_key,
        status="pending",
        priority=max(0, priority),
        created_at_s=max(0, now_s),
        requested_fields=["front_signal"],
        metadata=signal_payload,
    )
    pack_state.pending_signals[signal.signal_id] = signal
    return FrontSignalUpdate(queued_signal=signal)


def _front_runtime(pack_state: ContentPackState) -> dict[str, Any]:
    runtime = pack_state.metadata.get(FRONT_RUNTIME_METADATA_KEY)
    if not isinstance(runtime, dict):
        runtime = {}
        pack_state.metadata[FRONT_RUNTIME_METADATA_KEY] = runtime
    runtime.setdefault("fronts", {})
    runtime.setdefault("villains", {})
    return runtime


def _runtime_entry(
    runtime: dict[str, Any],
    group: str,
    key: str,
) -> dict[str, Any]:
    values = runtime.setdefault(group, {})
    if not isinstance(values, dict):
        values = {}
        runtime[group] = values
    entry = values.setdefault(key, {})
    if not isinstance(entry, dict):
        entry = {}
        values[key] = entry
    return entry


def _record_knowledge(
    runtime_entry: dict[str, Any],
    *,
    source_event_id: str,
    known: Sequence[str],
    visibility: str,
    now_s: int,
) -> None:
    _append_unique(runtime_entry.setdefault("known_facts", []), known)
    _append_unique(runtime_entry.setdefault("source_event_ids", []), [source_event_id])
    runtime_entry["last_visibility"] = visibility
    runtime_entry["last_known_at_s"] = max(0, now_s)


def _upsert_active_plan(
    runtime_entry: dict[str, Any],
    *,
    plan: str,
    source_event_id: str,
    actor_id: str,
) -> None:
    plans = runtime_entry.setdefault("active_plans", [])
    if not isinstance(plans, list):
        plans = []
        runtime_entry["active_plans"] = plans
    payload = {
        "plan": plan,
        "source_event_id": source_event_id,
        "status": "active",
    }
    if actor_id:
        payload["actor"] = actor_id
    for index, existing in enumerate(plans):
        if (
            isinstance(existing, dict)
            and existing.get("source_event_id") == source_event_id
        ):
            plans[index] = payload
            return
    plans.append(payload)


def _active_suppression(
    entries: Sequence[dict[str, Any] | None],
    *,
    now_s: int,
) -> str:
    now = max(0, now_s)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if _clean_nonnegative_int(entry.get("cooldown_until_s")) not in (None, 0):
            cooldown_until = _clean_nonnegative_int(entry.get("cooldown_until_s")) or 0
            if cooldown_until > now:
                return "cooldown"
        restraint = entry.get("restraint")
        if isinstance(restraint, dict):
            restraint_until = _clean_nonnegative_int(restraint.get("until_s")) or 0
            if restraint_until > now:
                return "restraint"
    return ""


def _front_ref(front_id: str) -> str:
    if front_id.startswith("front/"):
        return front_id
    return f"front/{front_id}"


def _signal_id(
    *,
    pack_id: str,
    front_id: str,
    villain_id: str,
    source_event_id: str,
) -> str:
    raw = json.dumps(
        {
            "pack": pack_id,
            "front": front_id,
            "villain": villain_id,
            "source": source_event_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "front_sig_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _signal_hash(
    *,
    source_event_id: str,
    ref_id: str,
    actor: str,
    known: Sequence[str],
    pressure: str,
    summary: str,
) -> str:
    raw = json.dumps(
        {
            "source_event_id": source_event_id,
            "ref_id": ref_id,
            "actor": actor,
            "known": list(known),
            "pressure": pressure,
            "summary": summary,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _append_unique(target: list[Any], values: Sequence[str]) -> None:
    seen = {_clean_text(value) for value in target if _clean_text(value)}
    for value in values:
        item = _clean_text(value)
        if item and item not in seen:
            target.append(item)
            seen.add(item)


def _clean_items(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [_clean_text(value)] if _clean_text(value) else []
    return [
        item
        for item in dict.fromkeys(_clean_text(item) for item in value)
        if item
    ]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _clean_token(value: Any) -> str:
    return _clean_text(value).lower()


def _clean_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)
