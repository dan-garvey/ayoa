"""v11 turn loop — DM-paced, beat-based, Cat I/II intention pipeline.

This module codifies the Plan-v11 state machine for the turn pipeline.
It replaces the one-shot `process_turn → single narrator render` shape
with a beat-cascading loop: intentions flow in, are classified Cat I
(self-closing) or Cat II (contested responder-collecting), canonical
events are adjudicated, broadcast to observers in the shared perceptual
frame, and the beat
continues through agent reactions until the router signals ends_beat
or the backstop fires.

## State machine summary

  enter(intention, actor_id):
    validate against session active_act_slots — claim, reject, or
    interpret-as-Cat-II-responder.

  classify(intention) → Cat I | Cat II
    Cat I → canonicalize, broadcast, check ends_beat.
    Cat II → open event, collect required responders (agents intend
      immediately; humans pin slot and wait), adjudicate on close,
      then give the initiator the first follow-up if they are an NPC.

  broadcast(event):
    Append to every local human's render buffer, plus mediated humans
    explicitly named by visible facts.
    Feed local agents, plus mediated agents explicitly named by visible
    facts, through their observation context.

  beat_end?:
    If true → render each human with a queued buffer, flush
    buffers, release beat slots, park loop.
    If false → pick next actor (agent-only, by pressure or
      responder_picks from the event), call agent.intend(),
      recurse.

## What's wired vs. TODO

This module is the state-machine skeleton. Key integration points are
marked with TODO(v11-wireup) comments; they'll be filled in as the
other engine modules (event_router, character_agent, narrator) are
brought into the v11 shape. The existing orchestrator.py still runs
the old v8 pipeline — the v11 path is gated until it's fully wired.

## Terminology

  Beat — a stretch of canonical events that closes on ends_beat.
  active_act_slot — the session's beat gate, holding 0..N character-slot entries.
  Cat I — self-closing intention (dialogue, passive, unambiguous).
  Cat II — contested intention, collects responder intentions.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol


def _utcnow_iso() -> str:
    """v11-r3c: timezone-aware UTC ISO-8601 timestamp. Never use
    datetime.utcnow() (deprecated in 3.12, naive, and miscomputes
    staleness under non-UTC hosts)."""
    return datetime.now(timezone.utc).isoformat()


def _is_agent_refusal(text: str) -> bool:
    """v11-r5: empty-output guard only.

    Previous rounds tried to detect LLM refusal phrasings via substring
    matching so the orchestrator could skip the pick rather than routing
    "I cannot help with that" as an intention. That was the wrong layer:
    pattern-matching produced false positives on legitimate terse
    in-character dialogue ("I can't see them from here," "I cannot allow
    that, my lord.") and false negatives on real refusals with stylistic
    variations (em-dashed preambles, quoted wrappers, sorry-prefixes).
    The correct fix is PROMPT-level: the agent prompt has a rule
    forbidding refusals / frame-breaks and tells the agent that an empty
    output is valid in-character silence. The engine then only needs to
    guard against literal empty output — a real agent failure. If the
    agent emits a refusal anyway, the adjudicator treats it as dialogue
    and the beat reveals the bug loudly in playtest rather than masking
    it with a drop.

    Returns True iff the text is empty or whitespace-only.
    """
    return not (text or "").strip()

from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    WorldAdjudication,
    visible_fact_texts,
)
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import (
    CommitmentRevisionPrompt,
    OpenCatIIEvent,
    OpenCommitment,
    RenderBufferEntry,
    SlotEntry,
)
from app.engine import dnd_combat
from app.engine.dnd_cat_ii import DndCatIIRollsPending

logger = logging.getLogger(__name__)


_HUMAN_CLASS_NOUNS = (
    "someone|somebody|people|those|men|women|folk|anyone|"
    "a person|a man|a woman|a child"
)

_HARVEST_REFLECTIVE_PATTERNS = (
    re.compile(
        rf"\bwith (?:a|an|the) [^.!?;]{{0,140}}\bof "
        rf"(?:{_HUMAN_CLASS_NOUNS})\b(?:\s+(?:who|whose)\b|\s+\w+ing\b)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bthe (?:look|expression|kind|sort|sound|shape|register|voice|"
        rf"body|build|gaze) of (?:{_HUMAN_CLASS_NOUNS})\b"
        rf"(?:\s+(?:who|whose)\b|\s+\w+ing\b)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:moves?|moving|speaking|looking)\s+like "
        rf"(?:{_HUMAN_CLASS_NOUNS})\b(?:\s+(?:who|whose)\b|\s+\w+ing\b)?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bthe way (?:{_HUMAN_CLASS_NOUNS})\b",
        re.IGNORECASE,
    ),
)


def _sanitize_harvest_fragment(text: str) -> str:
    """Drop perception-loadout sentences that violate narrative rules.

    Observation harvest bypasses the router's canonicalization pass:
    CharacterAgent.perceive() returns prose and run_beat appends it
    straight into `observable_facts`. That means the normal router
    repair rule cannot protect the narrator from agent-originated
    reflective similes. Perception is enrichment rather than critical
    event state, so the safest deterministic repair is to omit only
    the violating sentence and keep the remaining visible surface.
    """
    text = (text or "").strip()
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    dropped = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if any(pattern.search(sentence) for pattern in _HARVEST_REFLECTIVE_PATTERNS):
            dropped += 1
            continue
        kept.append(sentence)

    if dropped:
        logger.warning(
            "Harvest: dropped %d perception sentence(s) with banned "
            "reflective-simile frames",
            dropped,
        )
    return " ".join(kept).strip()


class SessionLockManager:
    """v11: one asyncio.Lock per session.

    The orchestrator holds an instance; every /act acquires the session
    lock before running check_act_slot → run_beat. Prevents
    the race where two concurrent /acts both see FREE and both claim
    the initiator slot.

    Locks are created lazily. They are NOT persisted; a process restart
    starts with no locks held.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._mutex = asyncio.Lock()

    async def get(self, session_id: str) -> asyncio.Lock:
        async with self._mutex:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            return lock


class SlotConflict(Enum):
    """v11 error taxonomy for /act rejection. See docs in `check_act_slot`.

    The orchestrator maps these to user-facing messages. Four distinct
    failure modes, each with a specific explanation:
    """
    # Someone else holds the beat's initiator slot.
    INITIATOR_HELD = "initiator_held"
    # A Cat II is pinned on another human — this user can't act.
    CAT_II_OTHER_HELD = "cat_ii_other_held"
    # This user is pinned as a Cat II responder. Their
    # /act IS accepted but interpreted as their responder intention,
    # not as a fresh initiator. (This is a marker, not an error.)
    CAT_II_SELF_RESPONDER = "cat_ii_self_responder"
    # This user owes a player-facing dice roll for a D&D Cat II event.
    # /act is rejected; the Discord roll UI must submit the stored roll.
    CAT_II_SELF_ROLL = "cat_ii_self_roll"
    # This user is holding an out-of-turn combat reaction window. Their
    # next /act is accepted as that reaction.
    COMBAT_REACTION_SELF = "combat_reaction_self"
    # A combat reaction prompt is pending on another player.
    COMBAT_REACTION_OTHER_HELD = "combat_reaction_other_held"
    # This user already holds the initiator slot — their previous /act
    # is still mid-beat.
    SELF_BUSY = "self_busy"
    # No conflict — slot is claimable.
    FREE = "free"


@dataclass
class SlotCheck:
    """Outcome of validating an incoming /act against the beat slots.

    If `conflict == FREE` → orchestrator proceeds to claim and process.
    If `conflict == CAT_II_SELF_RESPONDER` → the /act is a responder
      intention for `cat_ii_event_id`, not a fresh initiator.
    Any other value → reject the /act with the explanation pointing at
      `holder_id` (who's currently holding the slot) and `reason`.
    """
    conflict: SlotConflict
    # character_id of the holder (if any). Empty when conflict==FREE.
    holder_id: str = ""
    # For CAT_II_SELF_RESPONDER, the open event id this /act fills.
    cat_ii_event_id: str | None = None
    # For COMBAT_REACTION_SELF, the closed canonical event id this /act
    # reacts to.
    trigger_event_id: str | None = None


def check_act_slot(
    ckpt: CheckpointFile,
    acting_character_id: str,
) -> SlotCheck:
    """Validate an incoming /act against the session's active act slot.

    Returns a SlotCheck indicating whether to accept, reject, or
    interpret-as-Cat-II-response.
    """
    if _active_combat(ckpt) is None and any(
        entry.reason in {"combat_reaction", "combat_blocked"}
        for entry in ckpt.session.active_act_slots.values()
    ):
        ckpt.session.active_act_slots = {
            cid: entry
            for cid, entry in ckpt.session.active_act_slots.items()
            if entry.reason not in {"combat_reaction", "combat_blocked"}
        }
    slot = ckpt.session.active_act_slots
    if not slot:
        return SlotCheck(conflict=SlotConflict.FREE)

    # If this character is a Cat II responder here, their /act is
    # their response intention.
    my_entry = slot.get(acting_character_id)
    if my_entry and my_entry.reason == "cat_ii_responder":
        return SlotCheck(
            conflict=SlotConflict.CAT_II_SELF_RESPONDER,
            holder_id=acting_character_id,
            cat_ii_event_id=my_entry.cat_ii_event_id,
        )

    if my_entry and my_entry.reason == "cat_ii_roll":
        return SlotCheck(
            conflict=SlotConflict.CAT_II_SELF_ROLL,
            holder_id=acting_character_id,
            cat_ii_event_id=my_entry.cat_ii_event_id,
        )

    if my_entry and my_entry.reason == "combat_reaction":
        return SlotCheck(
            conflict=SlotConflict.COMBAT_REACTION_SELF,
            holder_id=acting_character_id,
            trigger_event_id=my_entry.trigger_event_id
                or my_entry.cat_ii_event_id,
        )

    # If this character holds the initiator slot, they're double-acting.
    if my_entry and my_entry.reason == "initiator":
        return SlotCheck(
            conflict=SlotConflict.SELF_BUSY,
            holder_id=acting_character_id,
        )

    # Someone else holds slot(s). Find out who and why.
    # Prefer reporting a Cat II responder pin (more specific) over an
    # initiator pin.
    for holder_id, entry in slot.items():
        if entry.reason in {"cat_ii_responder", "cat_ii_roll"}:
            return SlotCheck(
                conflict=SlotConflict.CAT_II_OTHER_HELD,
                holder_id=holder_id,
            )
    for holder_id, entry in slot.items():
        if entry.reason == "combat_reaction":
            return SlotCheck(
                conflict=SlotConflict.COMBAT_REACTION_OTHER_HELD,
                holder_id=holder_id,
                trigger_event_id=entry.trigger_event_id
                    or entry.cat_ii_event_id,
            )
    for holder_id, entry in slot.items():
        if entry.reason == "initiator":
            return SlotCheck(
                conflict=SlotConflict.INITIATOR_HELD,
                holder_id=holder_id,
            )

    # Fallback — shouldn't happen, but don't crash.
    return SlotCheck(conflict=SlotConflict.FREE)


def claim_initiator_slot(
    ckpt: CheckpointFile,
    character_id: str,
) -> None:
    """Claim the session's initiator slot for a human about to run a beat."""
    ckpt.session.active_act_slots[character_id] = SlotEntry(
        reason="initiator",
        cat_ii_event_id=None,
        claimed_at=_utcnow_iso(),
    )


def pin_cat_ii_responder(
    ckpt: CheckpointFile,
    character_id: str,
    cat_ii_event_id: str,
) -> None:
    """Pin a human as a Cat II responder. The session cannot accept
    unrelated /acts from other humans until the event closes."""
    ckpt.session.active_act_slots[character_id] = SlotEntry(
        reason="cat_ii_responder",
        cat_ii_event_id=cat_ii_event_id,
        claimed_at=_utcnow_iso(),
    )


def pin_combat_reaction(
    ckpt: CheckpointFile,
    character_id: str,
    trigger_event_id: str,
) -> bool:
    """Pin a human for a possible D&D combat reaction.

    Returns False if the character already has a live slot. Cat II and roll
    pins are higher priority than optional reactions, so reaction prompting
    never overwrites them.
    """
    if character_id in ckpt.session.active_act_slots:
        return False
    ckpt.session.active_act_slots[character_id] = SlotEntry(
        reason="combat_reaction",
        trigger_event_id=trigger_event_id,
        claimed_at=_utcnow_iso(),
    )
    return True


def pin_combat_start_blocked(
    ckpt: CheckpointFile,
    character_id: str,
    trigger_event_id: str,
) -> bool:
    existing = ckpt.session.active_act_slots.get(character_id)
    if existing is not None and existing.reason != "initiator":
        return False
    ckpt.session.active_act_slots[character_id] = SlotEntry(
        reason="combat_blocked",
        trigger_event_id=trigger_event_id,
        claimed_at=_utcnow_iso(),
    )
    return True


def release_beat_slots(ckpt: CheckpointFile) -> None:
    """Release slot entries at beat end.

    CRITICAL: this does NOT clobber pins tied to still-open Cat II
    events. If a beat ends while a different Cat II is still awaiting
    human responders, the pins for those responders stay. Only initiator
    slots and responder slots whose owning event has already closed are
    released.
    """
    slot = ckpt.session.active_act_slots
    if not slot:
        return
    open_evt_ids = {e.event_id for e in ckpt.session.open_cat_ii_events}
    keep: dict[str, SlotEntry] = {}
    for cid, entry in slot.items():
        if (
            entry.reason in {"cat_ii_responder", "cat_ii_roll"}
            and entry.cat_ii_event_id in open_evt_ids
        ):
            keep[cid] = entry
        elif entry.reason == "combat_reaction":
            keep[cid] = entry
        elif entry.reason == "combat_blocked" and _active_combat(ckpt) is not None:
            keep[cid] = entry
    if keep:
        ckpt.session.active_act_slots = keep
    else:
        ckpt.session.active_act_slots = {}


def purge_character_state(ckpt: CheckpointFile, character_id: str) -> None:
    """Sweep all v11 bookkeeping when a character leaves/culls/unbinds.

    Called by the roster-move / /leave / cull code paths so state that
    would otherwise strand (pins, open events they're required in,
    render buffers) doesn't leak.

    What this does:
    - Drops the character's entry from the active act slots.
    - Removes the character from every open Cat II event's required
      responders AND collected intentions. If the event has no
      remaining required responders missing (all filled / none left),
      it stays open for the orchestrator to resolve normally. If the
      initiator is the one leaving, the event is abandoned (closed
      without resolution).
    - Empties their render buffer.
    - Leaves canonical_events + narrator_conversations alone — those
      are historical, not live state.
    """
    # 1. Slots.
    ckpt.session.active_act_slots.pop(character_id, None)

    # 2. Open Cat II events.
    remaining_events: list[OpenCatIIEvent] = []
    for evt in ckpt.session.open_cat_ii_events:
        if evt.initiator_id == character_id:
            # Initiator left — abandon the event.
            logger.info(
                "Cat II event %s abandoned: initiator %s removed",
                evt.event_id, character_id,
            )
            continue
        if character_id in evt.required_responders:
            evt.required_responders = [
                r for r in evt.required_responders if r != character_id
            ]
        evt.collected_intentions.pop(character_id, None)
        # If all required responders are now gone (removed as they left),
        # the event is trivially ready — leave it for the orchestrator.
        remaining_events.append(evt)
    ckpt.session.open_cat_ii_events = remaining_events

    # 3. Render buffer.
    ckpt.session.render_buffers.pop(character_id, None)


def sweep_stale_cat_ii_pins(
    ckpt: CheckpointFile,
    now_iso: str | None = None,
) -> list[str]:
    """v11: walk all open Cat II events; for any whose human responders
    have been pinned longer than `cat_ii_human_timeout_seconds`, auto-
    resolve them as "stays out" by synthesizing an intention on the
    human's behalf (visible in the adjudicated event — the fallback is
    not invisible LLM substitution; the prose will say the character
    did not act).

    Returns the event_ids of any events that had intentions synthesized.
    The orchestrator then re-enters run_beat for each (using the Cat II
    event path) to trigger adjudication. Zero-length list is the normal
    case.

    Set `cat_ii_human_timeout_seconds = 0` in settings to disable.
    """
    timeout = ckpt.session.config.settings.cat_ii_human_timeout_seconds
    if timeout <= 0:
        return []

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc) if now_iso is None else _parse_iso(now_iso)
    bindings = ckpt.session.character_bindings or {}

    to_adjudicate: list[str] = []
    for evt in ckpt.session.open_cat_ii_events:
        # Compute how long the event has been open.
        try:
            opened = _parse_iso(evt.opened_at)
        except Exception:
            continue
        if (now - opened).total_seconds() < timeout:
            continue

        # Find pinned humans whose intention is still missing.
        stale_humans = [
            r for r in evt.required_responders
            if r in bindings and r not in evt.collected_intentions
        ]
        if not stale_humans:
            continue

        mutated = False
        for h in stale_humans:
            # Structured marker, not a magic string. Part C of the router
            # prompt skips swept responders entirely — they don't end up
            # in the adjudication input, so no meta text can leak into
            # canonical event facts. The render will show the
            # character as present-but-non-reactive via narrator
            # guidance, never as "does not act — away from the action".
            if h not in evt.swept_responders:
                evt.swept_responders.append(h)
                mutated = True
            # Sentinel string still recorded in collected_intentions for
            # debug inspection, but adjudication will filter these out
            # via swept_responders.
            evt.collected_intentions[h] = "[AFK-swept: no player intention]"
            logger.warning(
                "Cat II event %s: auto-resolved pin on %s after timeout",
                evt.event_id, h,
            )
        # v11-r5: only append to to_adjudicate when this call ACTUALLY
        # mutated the event's swept_responders. Prevents a second call
        # (e.g. concurrent sweep within the lock, or a retry) from
        # double-returning the same event id.
        if mutated and evt.event_id not in to_adjudicate:
            to_adjudicate.append(evt.event_id)
    return to_adjudicate


def _parse_iso(s: str):
    """Tolerant ISO-8601 parse: strips trailing 'Z', accepts naive or
    aware timestamps."""
    from datetime import datetime, timezone
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def abort_beat(ckpt: CheckpointFile) -> int:
    """Admin-escape: force-release every slot, abandon every open Cat II
    event, and flush every queued render buffer. Returns the number of
    events abandoned.

    Used by the `/abort_beat` admin command when a pin gets wedged.
    Preserves the canonical_events log (we keep history) but the
    mid-flight Cat II is treated as if it never resolved. Render buffers
    are flushed so mid-beat aborts don't leak stale events into the next
    beat's render fan-out.
    """
    buffers_cleared = 0
    for h, existing in list(ckpt.session.render_buffers.items()):
        existing = ckpt.session.render_buffers.get(h)
        if existing:
            buffers_cleared += 1
        ckpt.session.render_buffers[h] = []

    ckpt.session.active_act_slots = {}
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is not None:
        if isinstance(combat, dict):
            combat["pending_advance_actor_id"] = ""
        else:
            setattr(combat, "pending_advance_actor_id", "")
    before = len(ckpt.session.open_cat_ii_events)
    ckpt.session.open_cat_ii_events = []
    dropped = before - len(ckpt.session.open_cat_ii_events)
    logger.warning(
        "abort_beat: slots released; %d open Cat II events abandoned; "
        "%d render buffers cleared",
        dropped, buffers_cleared,
    )
    return dropped


def release_character_slot(
    ckpt: CheckpointFile,
    character_id: str,
) -> None:
    """Release one character's slot entry while leaving other entries
    intact. Used when a single Cat II responder closes their intention
    but others are still pending."""
    ckpt.session.active_act_slots.pop(character_id, None)


def new_event_id() -> str:
    """Fresh event id for canonical events and open Cat II events."""
    return f"evt_{uuid.uuid4().hex[:12]}"


def open_cat_ii(
    ckpt: CheckpointFile,
    initiator_id: str,
    initiator_intention: str,
    required_responders: list[str],
    opening_event: EventRouterOutput | None = None,
) -> OpenCatIIEvent:
    """Open a Cat II event and register it on the checkpoint. Humans
    among required_responders are pinned into the beat slot."""
    evt = OpenCatIIEvent(
        event_id=new_event_id(),
        initiator_id=initiator_id,
        initiator_intention=initiator_intention,
        required_responders=list(required_responders),
        collected_intentions={},
        opening_event_id=opening_event.event_id if opening_event else "",
        opening_observer_ids=[
            observer.character_id for observer in opening_event.observers
        ] if opening_event else [],
        opening_observable_facts=[
            fact.text for fact in opening_event.canonical_event.observable_facts
        ] if opening_event else [],
        opened_at=_utcnow_iso(),
    )
    ckpt.session.open_cat_ii_events.append(evt)
    return evt


def collect_cat_ii_intention(
    ckpt: CheckpointFile,
    event_id: str,
    responder_id: str,
    intention: str,
) -> OpenCatIIEvent | None:
    """Record a responder's intention into its open Cat II event.
    Returns the open event, or None if the event was not found (e.g.
    already closed)."""
    for evt in ckpt.session.open_cat_ii_events:
        if evt.event_id == event_id:
            evt.collected_intentions[responder_id] = intention
            return evt
    return None


def cat_ii_is_ready(evt: OpenCatIIEvent) -> bool:
    """True iff every required responder has submitted an intention."""
    return set(evt.collected_intentions.keys()) >= set(evt.required_responders)


def close_cat_ii(ckpt: CheckpointFile, event_id: str) -> None:
    """Remove a Cat II event from the open-events list once adjudicated."""
    ckpt.session.open_cat_ii_events = [
        e for e in ckpt.session.open_cat_ii_events if e.event_id != event_id
    ]


def align_cat_ii_resolution_time(
    ckpt: CheckpointFile,
    open_event: OpenCatIIEvent,
    resolved: EventRouterOutput,
) -> None:
    """Pin a Cat II resolution to the opening event's fiction time."""
    if not open_event.opening_event_id:
        return
    opening = next(
        (
            event for event in ckpt.canonical_events
            if event.event_id == open_event.opening_event_id
        ),
        None,
    )
    if opening is None:
        return
    resolved.effective_at_s = opening.effective_at_s


# ---- Broadcast + render buffering ------------------------------------------


def append_to_render_buffer(
    ckpt: CheckpointFile,
    character_id: str,
    event_id: str,
    observation_level: str = "direct",
    *,
    visible_at_s: int = 0,
    event_sequence: int = 0,
) -> None:
    """Queue a canonical event for a human's next render."""
    buf = ckpt.session.render_buffers.setdefault(character_id, [])
    buf.append(
        RenderBufferEntry(
            event_id=event_id,
            observation_level=observation_level,
            visible_at_s=max(0, visible_at_s),
            event_sequence=max(0, event_sequence),
        )
    )


def flush_render_buffer(
    ckpt: CheckpointFile, character_id: str
) -> list[RenderBufferEntry]:
    """Pop a human's render buffer for composition. Called right before
    the narrator renders their POV at beat end."""
    out = ckpt.session.render_buffers.get(character_id, [])
    ckpt.session.render_buffers[character_id] = []
    return out


def _log_router_rationale(
    result: EventRouterOutput, actor_id: str,
    *, kind: str = "route",
) -> None:
    """v11-r7g (TEMPORARY DIAGNOSTIC): surface the router's one-sentence
    rationale at INFO so playtest logs show "why" alongside "what".

    The decision_rationale field on EventRouterOutput is a temporary
    addition to solidify v11 prompt engineering. We log it here from
    every route_intention site (Cat I, Cat II open, Cat II resolve)
    rather than scattering log calls inline. Remove this helper, the
    schema field, the prompt rule, and every call to it together when
    the diagnostic is no longer worth the per-turn token cost.

    `kind` distinguishes the routing context — "route" (normal Cat I
    / Cat II open call), "cat_ii_resolve" (final adjudication of a
    closed Cat II), so log readers can tell them apart.
    """
    rationale = (result.decision_rationale or "").strip()
    if not rationale:
        rationale = "(no rationale emitted)"
    logger.info(
        "router[%s] actor=%s cat=%s ends_beat=%s reason=%r picks=%s :: %s",
        kind, actor_id,
        "II" if result.requires_responders else "I",
        result.ends_beat, result.ends_beat_reason,
        result.agent_responder_picks, rationale,
    )


def _filter_picks_for_dispatch(
    ckpt: CheckpointFile,
    picks: list[str],
    event: EventRouterOutput | None = None,
) -> list[str]:
    """Drop human-controlled picks before NPC dispatch.

    Humans do not cascade via the router; they only enter through
    `/act`. Using `collect_player_ids` here (rather than the bare
    bindings dict) is the load-bearing fix for the playable-2 era:
    `session.player_character_id` may name the creator's bound
    character without there being a corresponding entry in
    `character_bindings` (older saves; CLI single-player flows).
    Filtering on bindings alone let those creator-bound characters
    slip into NPC dispatch and produced the "router tried to make my
    own character speak" symptom.

    NPC picks are intentionally not filtered by physical location or
    fact-level visibility here. Whether an off-location NPC should act
    is a router decision; if it picks a remote producer, caller, spirit,
    watcher, or other mediated participant, we dispatch that NPC and let
    prompt/schema tuning decide whether the pick was good.

    Returns the filtered list preserving router order.
    """
    del event
    # Local import to avoid an engine-package import cycle on module
    # load (context_builder pulls some of the same schemas turn_loop
    # exports). Cheap — the function is tiny and the import is cached.
    from app.engine.context_builder import collect_player_ids

    humans = collect_player_ids(ckpt)
    return [rid for rid in picks if rid not in humans]


def _character_by_id(ckpt: CheckpointFile) -> dict[str, Any]:
    return {c.character_id: c for c in ckpt.characters}


def _visible_facts_for_character(
    event: EventRouterOutput,
    character_id: str,
) -> list[ObservableFact]:
    out: list[ObservableFact] = []
    for fact in event.canonical_event.observable_facts:
        if fact.audience == "all_observers" or fact.is_visible_to(character_id):
            if fact.text.strip():
                out.append(fact)
    return out


def _visible_at_for_facts(
    event: EventRouterOutput,
    facts: list[ObservableFact],
) -> int:
    if not facts:
        return event.effective_at_s + event.duration_s
    return max(
        event.effective_at_s + fact.at_offset_s + fact.duration_s
        for fact in facts
    )


def _advance_character_clock(
    ckpt: CheckpointFile,
    character_id: str,
    at_s: int,
    by_id: dict[str, Any],
) -> None:
    if at_s < 0:
        at_s = 0
    char = by_id.get(character_id)
    if char is None:
        return
    char.clock_at_s = max(getattr(char, "clock_at_s", 0), at_s)
    ckpt.session.leading_at_s = max(ckpt.session.leading_at_s, char.clock_at_s)


def _commitment_id_for_event(
    event: EventRouterOutput,
    actor_ids: list[str],
) -> str:
    suffix = "_".join(actor_ids) if actor_ids else "open"
    suffix = re.sub(r"[^a-zA-Z0-9_]+", "_", suffix).strip("_") or "open"
    return f"commit_{event.event_id}_{suffix}"[:96]


def _matching_commitments(
    ckpt: CheckpointFile,
    *,
    commitment_id: str = "",
    actor_ids: list[str] | None = None,
) -> list[OpenCommitment]:
    actors = {cid for cid in (actor_ids or []) if cid}
    matches: list[OpenCommitment] = []
    for commitment in ckpt.session.open_commitments:
        if commitment_id:
            if commitment.commitment_id == commitment_id:
                matches.append(commitment)
            continue
        if actors and actors.intersection(commitment.actor_ids):
            matches.append(commitment)
    return matches


def _drop_commitments(
    ckpt: CheckpointFile,
    commitments: list[OpenCommitment],
) -> None:
    drop_ids = {commitment.commitment_id for commitment in commitments}
    if not drop_ids:
        return
    affected_actor_ids = {
        actor_id
        for commitment in commitments
        for actor_id in commitment.actor_ids
    }
    ckpt.session.open_commitments = [
        commitment
        for commitment in ckpt.session.open_commitments
        if commitment.commitment_id not in drop_ids
    ]
    for actor_id in affected_actor_ids:
        prompt = ckpt.session.pending_commitment_revisions.get(actor_id)
        if prompt and prompt.commitment_id in drop_ids:
            ckpt.session.pending_commitment_revisions.pop(actor_id, None)


def _apply_location_updates(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    by_id: dict[str, Any],
) -> None:
    for update in event.location_updates:
        char = by_id.get(update.character_id)
        if char is None:
            logger.warning(
                "location_updates referenced unknown character_id %s",
                update.character_id,
            )
            continue
        char.location = update.location_label


def _apply_commitment_open(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    *,
    actor_id: str,
    by_id: dict[str, Any],
) -> None:
    signal = event.commitment_open
    if not signal.present:
        return
    actor_ids = list(signal.actor_ids)
    if not actor_ids and actor_id:
        actor_ids = [actor_id]
    actor_ids = [cid for cid in dict.fromkeys(actor_ids) if cid]
    if not actor_ids:
        logger.warning("commitment_open ignored for %s: no actor_ids", event.event_id)
        return
    overlapping = _matching_commitments(ckpt, actor_ids=actor_ids)
    if overlapping:
        _drop_commitments(ckpt, overlapping)

    expected_duration_s = signal.expected_duration_s
    max_duration_s = signal.max_duration_s or expected_duration_s
    if expected_duration_s == 0 and event.duration_s > 0:
        expected_duration_s = event.duration_s
    if max_duration_s == 0:
        max_duration_s = expected_duration_s
    location_label = signal.location_label
    if not location_label:
        first_actor = by_id.get(actor_ids[0])
        location_label = getattr(first_actor, "location", "") if first_actor else ""

    ckpt.session.open_commitments.append(
        OpenCommitment(
            commitment_id=_commitment_id_for_event(event, actor_ids),
            actor_ids=actor_ids,
            description=signal.description,
            trigger_event_id=event.event_id,
            started_at_s=event.effective_at_s,
            expected_end_s=event.effective_at_s + expected_duration_s,
            max_end_s=event.effective_at_s + max_duration_s,
            location_label=location_label,
        )
    )


def _apply_commitment_resolutions(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    *,
    by_id: dict[str, Any],
) -> None:
    for signal in event.commitment_resolutions:
        matches = _matching_commitments(
            ckpt,
            commitment_id=signal.commitment_id,
            actor_ids=signal.actor_ids,
        )
        if not matches:
            logger.info(
                "commitment resolution on %s matched no open commitments",
                event.event_id,
            )
            continue
        resolved_at_s = event.effective_at_s + signal.resolved_at_offset_s
        for commitment in matches:
            for actor_id in commitment.actor_ids:
                _advance_character_clock(ckpt, actor_id, resolved_at_s, by_id)
        _drop_commitments(ckpt, matches)


def _apply_commitment_interrupts(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    *,
    by_id: dict[str, Any],
) -> None:
    if not event.commitment_interrupts:
        return
    from app.engine.context_builder import collect_player_ids

    player_ids = collect_player_ids(ckpt)
    observer_ids = {observer.character_id for observer in event.observers}
    for signal in event.commitment_interrupts:
        matches = _matching_commitments(
            ckpt,
            commitment_id=signal.commitment_id,
            actor_ids=signal.actor_ids,
        )
        if not matches:
            logger.info(
                "commitment interrupt on %s matched no open commitments",
                event.event_id,
            )
            continue
        for commitment in matches:
            target_ids = signal.actor_ids or commitment.actor_ids
            for character_id in target_ids:
                if character_id not in player_ids or character_id not in observer_ids:
                    continue
                facts = _visible_facts_for_character(event, character_id)
                if not facts:
                    continue
                observed_at_s = event.effective_at_s + signal.observed_at_offset_s
                observed_at_s = max(observed_at_s, _visible_at_for_facts(event, facts))
                _advance_character_clock(ckpt, character_id, observed_at_s, by_id)
                ckpt.session.pending_commitment_revisions[character_id] = (
                    CommitmentRevisionPrompt(
                        character_id=character_id,
                        commitment_id=commitment.commitment_id,
                        trigger_event_id=event.event_id,
                        observed_at_s=observed_at_s,
                        reason=signal.reason,
                        previous_description=commitment.description,
                    )
                )


def _auto_commitment_revision_for_visible_event(
    ckpt: CheckpointFile,
    *,
    character_id: str,
    event: EventRouterOutput,
    visible_at_s: int,
    actor_id: str,
) -> None:
    if character_id == actor_id:
        return
    commitments = _matching_commitments(ckpt, actor_ids=[character_id])
    if not commitments:
        return
    for commitment in commitments:
        if commitment.trigger_event_id == event.event_id:
            continue
        existing = ckpt.session.pending_commitment_revisions.get(character_id)
        if existing is not None and existing.trigger_event_id == event.event_id:
            continue
        ckpt.session.pending_commitment_revisions[character_id] = (
            CommitmentRevisionPrompt(
                character_id=character_id,
                commitment_id=commitment.commitment_id,
                trigger_event_id=event.event_id,
                observed_at_s=visible_at_s,
                reason=(
                    "new visible information arrived before this commitment "
                    "resolved"
                ),
                previous_description=commitment.description,
            )
        )
        return


def _apply_time_and_private_state(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    *,
    actor_id: str,
    by_id: dict[str, Any],
) -> None:
    if event.effective_at_s < 0:
        event.effective_at_s = 0
    if event.duration_s < 0:
        event.duration_s = 0
    if actor_id:
        actor_clock = getattr(by_id.get(actor_id), "clock_at_s", 0)
        event.effective_at_s = max(event.effective_at_s, actor_clock)

    end_at_s = event.effective_at_s + event.duration_s
    if actor_id:
        _advance_character_clock(ckpt, actor_id, end_at_s, by_id)
    ckpt.session.leading_at_s = max(ckpt.session.leading_at_s, end_at_s)

    _apply_location_updates(ckpt, event, by_id)
    _apply_commitment_resolutions(ckpt, event, by_id=by_id)
    _apply_commitment_open(ckpt, event, actor_id=actor_id, by_id=by_id)
    _apply_commitment_interrupts(ckpt, event, by_id=by_id)


async def _agent_intention_for_dispatch(
    dispatcher: Dispatcher,
    ckpt: CheckpointFile,
    character_id: str,
) -> str | None:
    """Fetch one NPC intention and normalize empty/refusal outputs."""
    raw = await dispatcher.agent_intend(
        ckpt=ckpt,
        character_id=character_id,
    )
    if raw and raw.strip() and not _is_agent_refusal(raw):
        return raw
    logger.warning(
        "Dropped empty/refusal intention from agent %s", character_id,
    )
    return None


async def _append_harvest_fragments(
    dispatcher: Dispatcher,
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    picks: list[str],
    current_actor: str,
    log_label: str,
) -> None:
    """Append perception fragments to a just-broadcast event.

    Used by normal observation harvest and by private query answers
    that need an NPC's current visual self-presentation. Mutating the
    event after broadcast is intentional: render buffers store event ids
    and narrator composition resolves the live canonical event by id.
    """
    if not picks:
        logger.warning(
            "%s requested perception harvest but no harvestable picks "
            "remained after filtering.",
            log_label,
        )
        return

    fragments = await dispatcher.harvest_perceptions(
        ckpt=ckpt,
        character_ids=picks,
        acting_character_id=current_actor,
    )
    by_id = {c.character_id: c for c in ckpt.characters}
    appended = 0
    for cid, fragment in zip(picks, fragments):
        text = _sanitize_harvest_fragment(fragment)
        if not text:
            logger.warning(
                "%s: empty fragment for %s; dropped", log_label, cid,
            )
            continue
        name = by_id.get(cid)
        name = name.name if name else cid
        result.canonical_event.observable_facts.append(
            ObservableFact.all(f"[loadout — {name}] {text}")
        )
        appended += 1
    logger.info(
        "%s: %d/%d fragments appended to event %s",
        log_label, appended, len(picks), result.event_id,
    )


def broadcast_event(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    actor_id: str = "",
) -> list[str]:
    """Append a closed canonical event to the log and fan it out to
    every human observer's render buffer and every NPC observer's
    `pending_observations` queue, except the event's own actor.

    Perception is structural: the router declares event observers and
    fact-level visibility packets. Location is not a fallback and does
    not create implicit observation.

    `actor_id` is the character whose intention produced this event
    (the player who /act'd, or the cascade NPC whose intention the
    router just adjudicated). Excluded from the inbox push because
    the actor's own action lives in their rolling history (the
    assistant message they just produced); pushing it onto their
    inbox would surface as "you observed yourself doing the thing
    you just did" on their next on-stage turn. Default `""` is the
    backward-compatible no-op (no character is ever excluded by
    that id) but every production caller in `turn_loop.run_beat` /
    orchestrator passes the real actor.

    The event's stable `event_id` is what lands in human render buffers
    (not Python object identity) so checkpoints remain resolvable
    across process restarts.

    The NPC inbox path is the engine implementation of the perception
    channel the router describes in `observable_facts`. When an NPC is
    picked as `agent_responder` for this same beat, their `respond` call
    drains the queue and they see the just-broadcast event as the most
    recent entry.

    Pre-r8b the local observer path was a bug-hatchery: cascade NPCs got
    `observed_facts=[]` from `LLMDispatcher.agent_intend`, so they
    reacted to whatever stale entries already sat in their queue
    (typically stale perceptions from when the location opened).
    The router then saw an off-topic intention and fabricated
    plausible dialogue to fit the cascade slot — the "narrator
    summarized dialogue" symptom the v11-r8b playtest caught was
    the visible end of that chain.

    Note: the `actor_id` of the event is also surfaced via
    BeatResult.event_actor_ids for callers that need actor-aware
    event application.
    """
    obs_level_by_char: dict[str, str] = {}
    for o in event.observers:
        # Legacy: observation_level is a single-char code ("d"|"i"|"f").
        obs_level_by_char[o.character_id] = {
            "d": "direct", "i": "indirect", "f": "inferred",
        }.get(o.observation_level, "direct")

    by_id = _character_by_id(ckpt)
    _apply_time_and_private_state(
        ckpt,
        event,
        actor_id=actor_id,
        by_id=by_id,
    )
    event_sequence = len(ckpt.canonical_events)
    ckpt.canonical_events.append(event)

    from app.engine.context_builder import collect_player_ids

    player_ids = collect_player_ids(ckpt)
    # NPC perception payload. Pre-v11-r10 this was the router's
    # one-line event summary — narrator-grade prose with interior
    # interpretation woven in ("the strain of speaking close to the
    # edge of what she is permitted"), which leaked author-voice
    # interior into local NPCs as their own perception. r10 moved
    # the channel to `observable_facts`. r11 adds fact-level
    # visibility: public facts go to every observer, but private
    # facts (`audience="only"`) are filtered per recipient before
    # anything reaches an NPC inbox.
    visible_humans: list[str] = []
    for o in event.observers:
        visible_facts = _visible_facts_for_character(event, o.character_id)
        facts = [
            fact.text.strip()
            for fact in visible_facts
            if fact.text.strip()
        ]
        if not facts:
            continue
        visible_at_s = _visible_at_for_facts(event, visible_facts)
        _advance_character_clock(ckpt, o.character_id, visible_at_s, by_id)
        if o.character_id in player_ids:
            level = obs_level_by_char.get(o.character_id, "direct")
            append_to_render_buffer(
                ckpt,
                o.character_id,
                event.event_id,
                level,
                visible_at_s=visible_at_s,
                event_sequence=event_sequence,
            )
            _auto_commitment_revision_for_visible_event(
                ckpt,
                character_id=o.character_id,
                event=event,
                visible_at_s=visible_at_s,
                actor_id=actor_id,
            )
            visible_humans.append(o.character_id)

        if o.character_id in player_ids:
            continue
        if o.character_id == actor_id:
            # Actor of the event — their own action is already in their
            # rolling history (and for cascade NPCs, the next thing
            # they'll see in user-message context is what someone ELSE
            # said in response, not what they themselves just did).
            continue
        recipient = by_id.get(o.character_id)
        if recipient is None or recipient.status == "culled":
            continue
        if facts:
            if len(facts) == 1:
                payload = facts[0]
            else:
                payload = "\n".join(f"  - {f}" for f in facts)
        else:
            payload = ""
        if not payload:
            # No observable_facts to push — silence is the right
            # outcome. (We deliberately do NOT fall back to any
            # summary string; that's the leak this fix closes.)
            continue
        from app.engine.context_builder import replace_character_ids_with_names

        payload = replace_character_ids_with_names(payload, ckpt)
        recipient.pending_observations.append(payload)

    return visible_humans


_COMBAT_REACTION_MIN_PRIORITY = 5
_COMBAT_REACTION_EXCLUDED_REASONS = {
    "cat_ii_open",
    "cat_ii_resolution",
    "cat_ii_pending",
    "cat_ii_pending_rolls",
    "cat_ii_stale",
    "ruleset_cat_ii_suppressed",
    "ruleset_resolution",
    "query_response",
    "observation_harvest",
    "off_stage_tick",
}
_REACTION_BLOCKING_CONDITIONS = {
    "incapacitated",
    "paralyzed",
    "stunned",
    "unconscious",
}


def _obj_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _obj_set(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _active_combat(ckpt: CheckpointFile) -> Any | None:
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is None:
        return None
    status = _obj_get(combat, "status")
    if status is not None and status != "active":
        return None
    combatants = list(_obj_get(combat, "combatants", []) or [])
    return combat if combatants else None


def _combatant_character_id(combatant: Any) -> str:
    return (
        str(_obj_get(combatant, "character_id", "") or "")
        or str(_obj_get(combatant, "combatant_id", "") or "")
    )


def _combatant_defeat_state(combatant: Any) -> str:
    state = str(_obj_get(combatant, "defeat_state", "") or "")
    if state:
        return state
    return "active"


def _combatant_for_character(combat: Any, character_id: str) -> Any | None:
    for combatant in list(_obj_get(combat, "combatants", []) or []):
        if _combatant_character_id(combatant) == character_id:
            return combatant
    return None


def _character_in_active_combat(
    ckpt: CheckpointFile,
    character_id: str,
) -> bool:
    combat = _active_combat(ckpt)
    return combat is not None and (
        _combatant_for_character(combat, character_id) is not None
    )


def _character_in_dnd_active_combat(
    ckpt: CheckpointFile,
    character_id: str,
) -> bool:
    settings = getattr(
        getattr(ckpt.session, "config", None),
        "settings",
        None,
    )
    ruleset_id = str(getattr(settings, "ruleset_id", "") or "")
    return (
        ruleset_id == "dnd5e_basic"
        and _character_in_active_combat(ckpt, character_id)
    )


def _dnd_ruleset_enabled(ckpt: CheckpointFile) -> bool:
    settings = getattr(ckpt.session.config, "settings", None)
    ruleset_id = str(getattr(settings, "ruleset_id", "") or "")
    return ruleset_id == "dnd5e_basic"


def _character_status_value(character: Any) -> str:
    status = getattr(character, "status", "")
    return str(getattr(status, "value", status) or "")


def _dnd_interaction_mode(result: EventRouterOutput) -> str:
    return str(getattr(result, "interaction_mode", "") or "")


def _dnd_combat_start_participants(
    ckpt: CheckpointFile,
    actor_id: str,
    combatant_ids: list[str],
) -> list[Any]:
    seed_ids = list(dict.fromkeys([
        *([actor_id] if actor_id else []),
        *[cid for cid in combatant_ids if cid],
    ]))
    by_id = {
        char.character_id: char
        for char in ckpt.characters
        if _character_status_value(char) == "active"
    }
    selected: list[Any] = []
    seen: set[str] = set()
    for cid in seed_ids:
        char = by_id.get(cid)
        if char is None or cid in seen:
            continue
        selected.append(char)
        seen.add(cid)
    return selected


def _ensure_combatant_observers(
    result: EventRouterOutput,
    participants: list[Any],
) -> None:
    existing = {observer.character_id for observer in result.observers}
    for character in participants:
        cid = str(getattr(character, "character_id", "") or "")
        if not cid or cid in existing:
            continue
        result.observers.append(ObserverEntry(
            character_id=cid,
            observation_level="d",
            response_priority=3,
        ))
        existing.add(cid)


def _set_pending_initiating_action(
    combat: Any,
    *,
    actor_id: str,
    event_id: str,
    intention: str,
) -> None:
    for combatant in list(_obj_get(combat, "combatants", []) or []):
        if _combatant_character_id(combatant) != actor_id:
            continue
        _obj_set(combatant, "pending_initiating_action", intention.strip())
        _obj_set(combatant, "pending_initiating_event_id", event_id)
        return


def _clear_pending_initiating_action(ckpt: CheckpointFile, actor_id: str) -> None:
    combat = _active_combat(ckpt)
    if combat is None:
        return
    combatant = _combatant_for_character(combat, actor_id)
    if combatant is None:
        return
    _obj_set(combatant, "pending_initiating_action", "")
    _obj_set(combatant, "pending_initiating_event_id", "")


def flush_combat_visible_facts(ckpt: CheckpointFile) -> int:
    combat = _active_combat(ckpt)
    if combat is None:
        return 0
    facts = dnd_combat.drain_pending_visible_facts(combat)
    if not facts:
        return 0
    observers: list[ObserverEntry] = []
    seen: set[str] = set()
    for combatant in list(_obj_get(combat, "combatants", []) or []):
        cid = _combatant_character_id(combatant)
        if not cid or cid in seen:
            continue
        observers.append(ObserverEntry(
            character_id=cid,
            observation_level="d",
            response_priority=3,
        ))
        seen.add(cid)
    if not observers:
        return 0
    event = EventRouterOutput(
        event_id="",
        decision_rationale="code-owned combat state change",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(feasible=True),
            observable_facts=[ObservableFact.all(fact) for fact in facts],
        ),
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="state_change",
        observers=observers,
        spawn=[],
        dormant=[],
        cull=[],
    )
    broadcast_event(ckpt, event)
    return 1


def _start_dnd_combat_from_router_signal(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    actor_id: str,
    intention: str,
) -> bool:
    if not _dnd_ruleset_enabled(ckpt) or _active_combat(ckpt) is not None:
        return False
    combatant_ids = list(getattr(result, "combatant_ids", []) or [])
    participants = _dnd_combat_start_participants(
        ckpt, actor_id, combatant_ids,
    )
    if len(participants) < 2:
        result.requires_responders = False
        result.required_responders = []
        result.agent_responder_picks = []
        result.ends_beat = True
        result.ends_beat_reason = "state_change"
        return False

    combat_id = f"combat_{result.event_id or uuid.uuid4().hex[:8]}"
    combat = dnd_combat.start_combat(
        ckpt.session,
        participants,
        combat_id=combat_id,
    )
    order = ", ".join(
        f"{c.name or c.character_id} {c.initiative_total}"
        for c in combat.combatants
    )
    result.requires_responders = False
    result.required_responders = []
    result.agent_responder_picks = []
    result.ends_beat = True
    result.ends_beat_reason = "state_change"
    _ensure_combatant_observers(result, participants)
    _set_pending_initiating_action(
        combat,
        actor_id=actor_id,
        event_id=result.event_id,
        intention=intention,
    )
    result.canonical_event.observable_facts.append(ObservableFact.all(
        "D&D combat begins."
    ))
    dnd_combat.append_audit_line(
        combat,
        "Combat started from router D&D interaction signal: "
        f"{actor_id}; combatants={', '.join(c.character_id for c in participants)}. "
        f"Initiative order: {order}.",
    )
    return True


def _block_dnd_combat_start_from_router_signal(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    actor_id: str,
) -> None:
    result.requires_responders = False
    result.required_responders = []
    result.agent_responder_picks = []
    result.ends_beat = True
    result.ends_beat_reason = "state_change"
    if actor_id:
        existing_observers = {
            observer.character_id for observer in result.observers
        }
        if actor_id not in existing_observers:
            result.observers.append(ObserverEntry(
                character_id=actor_id,
                observation_level="d",
                response_priority=3,
            ))
        result.canonical_event.observable_facts.append(ObservableFact.all(
            "The hostile action stops short; no attack, spell, or injury "
            "takes effect."
        ))
        pin_combat_start_blocked(ckpt, actor_id, result.event_id)


def _end_dnd_combat_from_router_signal(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    actor_id: str,
) -> bool:
    combat = _active_combat(ckpt)
    if not _dnd_ruleset_enabled(ckpt) or combat is None:
        return False
    if _combatant_for_character(combat, actor_id) is None:
        result.requires_responders = False
        result.required_responders = []
        result.agent_responder_picks = []
        result.ends_beat = True
        result.ends_beat_reason = "state_change"
        return False
    dnd_combat.append_audit_line(
        combat,
        f"Combat ended from router D&D interaction signal: {result.event_id}.",
    )
    for fact in dnd_combat.drain_pending_visible_facts(combat):
        result.canonical_event.observable_facts.append(ObservableFact.all(fact))
    dnd_combat.end_combat(ckpt.session, characters=ckpt.characters)
    result.requires_responders = False
    result.required_responders = []
    result.agent_responder_picks = []
    result.ends_beat = True
    result.ends_beat_reason = "state_change"
    result.canonical_event.observable_facts.append(ObservableFact.all(
        "D&D combat ends."
    ))
    return True


def _combatant_can_react(combatant: Any) -> bool:
    if _combatant_defeat_state(combatant) != "active":
        return False
    if bool(_obj_get(combatant, "removed", False)):
        return False
    if not bool(_obj_get(combatant, "reaction_available", True)):
        return False
    conditions = {
        str(condition).strip().lower()
        for condition in (_obj_get(combatant, "conditions", []) or [])
    }
    return not (conditions & _REACTION_BLOCKING_CONDITIONS)


def _mark_combat_reaction_spent(
    ckpt: CheckpointFile,
    character_id: str,
) -> None:
    combat = _active_combat(ckpt)
    if combat is None:
        return
    combatant = _combatant_for_character(combat, character_id)
    if combatant is None:
        return
    if isinstance(combatant, dict):
        combatant["reaction_available"] = False
    else:
        setattr(combatant, "reaction_available", False)


def _combat_reaction_intention(
    ckpt: CheckpointFile,
    *,
    actor_id: str,
    trigger_event_id: str,
    intention: str,
) -> str:
    trigger = next(
        (
            event for event in ckpt.canonical_events
            if event.event_id == trigger_event_id
        ),
        None,
    )
    if trigger is None:
        return f"Combat reaction: {intention.strip()}"

    facts = visible_fact_texts(
        trigger.canonical_event.observable_facts,
        actor_id,
        include_all_observers=True,
    )
    summary = " ".join(facts[:3]).strip()
    if len(summary) > 700:
        summary = summary[:697].rstrip() + "..."
    if not summary:
        return f"Combat reaction: {intention.strip()}"
    return (
        f"Combat reaction to this event: {summary}\n"
        f"Reaction intention: {intention.strip()}"
    )


def _current_combat_character_id(combat: Any) -> str:
    combatants = list(_obj_get(combat, "combatants", []) or [])
    if not combatants:
        return ""
    try:
        idx = int(_obj_get(combat, "turn_index", 0) or 0) % len(combatants)
    except (TypeError, ValueError):
        idx = 0
    for offset in range(len(combatants)):
        candidate = combatants[(idx + offset) % len(combatants)]
        if (
            _combatant_defeat_state(candidate) == "active"
            and not bool(_obj_get(candidate, "removed", False))
        ):
            return _combatant_character_id(candidate)
    return ""


def _eligible_combat_reaction_prompts(
    ckpt: CheckpointFile,
    *,
    events_closed: int,
    event_actor_ids: list[str],
    suppress_reaction_prompts: bool,
) -> dict[str, str]:
    """Return character_id -> triggering event id for combat reaction UI.

    The router's response_priority is treated as a conservative UX signal,
    not as mechanical proof that a D&D reaction is legal.
    """
    if suppress_reaction_prompts or events_closed <= 0:
        return {}
    combat = _active_combat(ckpt)
    if combat is None:
        return {}

    from app.engine.context_builder import collect_player_ids

    player_ids = collect_player_ids(ckpt)
    if not player_ids:
        return {}

    closed_events = ckpt.canonical_events[-events_closed:]
    actors = event_actor_ids
    if len(actors) != len(closed_events):
        actors = actors + [""] * (len(closed_events) - len(actors))

    prompts: dict[str, str] = {}
    # Latest eligible trigger wins for each character, keeping one slot per
    # character and avoiding multiple prompts from a single dense beat.
    for event, actor_id in reversed(list(zip(closed_events, actors))):
        if event.ends_beat_reason in _COMBAT_REACTION_EXCLUDED_REASONS:
            continue
        for observer in event.observers:
            cid = observer.character_id
            if cid in prompts or cid == actor_id or cid not in player_ids:
                continue
            if observer.observation_level != "d":
                continue
            if observer.response_priority < _COMBAT_REACTION_MIN_PRIORITY:
                continue
            if cid in ckpt.session.active_act_slots:
                continue
            combatant = _combatant_for_character(combat, cid)
            if combatant is None or not _combatant_can_react(combatant):
                continue
            if not pin_combat_reaction(ckpt, cid, event.event_id):
                continue
            prompts[cid] = event.event_id
    return prompts


def _beat_cap_overrun_cause(
    ckpt: CheckpointFile,
    events_closed: int,
    ended_reason: str,
) -> tuple[bool, bool, bool]:
    """Best-effort cause flags for passive beat-cap overrun telemetry."""
    if events_closed <= 0:
        return False, False, False

    closed_events = ckpt.canonical_events[-events_closed:]
    closed_reasons = [event.ends_beat_reason for event in closed_events]
    cat_ii_open = (
        ended_reason == "cat_ii_pending"
        or any(reason == "cat_ii_open" for reason in closed_reasons)
    )
    cat_ii_resolution = ended_reason == "cat_ii_resolution" or (
        cat_ii_open and events_closed >= 2
    )
    cat_ii_followup = cat_ii_resolution and events_closed >= 3
    return cat_ii_open, cat_ii_resolution, cat_ii_followup


# ---- The beat loop ---------------------------------------------------------
#
# This is the heart of v11. `run_beat` is the orchestrator entry point for
# one player's /act or Cat II responder intention. It cascades through
# agent reactions until the router ends the beat, then fans the narrator
# out per human with a queued perception and releases the beat slots.
#
# TODO(v11-wireup): the calls into the router, narrator, and character_agent
# are sketched as protocol methods; these need to be bound to the real
# engine modules (event_router.run, narrator.compose, character_agent.
# run). See `Dispatcher` below.
# ----------------------------------------------------------------------------


class Dispatcher(Protocol):
    """Adapter for the async callables `run_beat` depends on. Keeps the
    loop testable in isolation from the concrete router/narrator/agent
    modules.

    The orchestrator's setup code constructs a Dispatcher binding real
    implementations; tests can pass in fakes.
    """

    async def route_intention(
        self,
        ckpt: CheckpointFile,
        actor_id: str,
        intention: str,
        cat_ii_event: OpenCatIIEvent | None = None,
    ) -> EventRouterOutput:
        """Classify + adjudicate one intention. Returns a canonical
        event (with Cat I/II decision + agent_responder_picks +
        ends_beat populated).

        If `cat_ii_event` is provided, the router is composing the final
        adjudication of an open Cat II event — it gets the full
        required-responders + collected-intentions bundle to resolve
        into one canonical event.
        """
        ...

    async def route_continuation(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        prior_result: EventRouterOutput,
    ) -> EventRouterOutput:
        """Advance an open beat when the prior router output supplied
        no dispatchable next actor despite `ends_beat=false`."""
        ...

    async def route_combat_action(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        intention: str,
    ) -> EventRouterOutput:
        """Resolve one active D&D combat turn through a ruleset adapter."""
        ...

    async def continue_combat_transaction(
        self,
        *,
        ckpt: CheckpointFile,
        event_id: str,
    ) -> EventRouterOutput:
        """Finalize an active-combat roll transaction after player dice."""
        ...

    async def agent_intend(
        self,
        ckpt: CheckpointFile,
        character_id: str,
    ) -> str:
        """Ask an agent for their next intention (as free-form text).
        Returns the intention string; the orchestrator re-routes it
        through route_intention as a fresh intention."""
        ...

    async def harvest_perceptions(
        self,
        ckpt: CheckpointFile,
        character_ids: list[str],
        acting_character_id: str,
    ) -> list[str]:
        """Fire CharacterAgent.perceive() across `character_ids` in
        parallel; return per-character "visual loadout" fragments in
        the same order as the input ids.

        Used by the observation-harvest fork in `run_beat` when the
        router classifies an action as `ends_beat_reason=
        "observation_harvest"`. Empty / failed-perception entries are
        included as empty strings so the caller can zip with the
        input ids; callers SHOULD filter those out before composing
        the canonical event's `observable_facts`.

        `acting_character_id` is the looker (the player who /act'd
        the observation). It is NOT the observer for filtering
        purposes — perception is observer-agnostic — but is plumbed
        for context-builder helpers that frame "the player who is
        currently doing things" in the agent's identity envelope.

        Per-call exceptions (LLM failure, agent prompt error) are
        absorbed into an empty fragment for that character rather
        than failing the whole harvest. One bad perception should
        not hide three good ones.
        """
        ...

    async def narrator_compose(
        self,
        ckpt: CheckpointFile,
        character_id: str,
        buffered_events: list[RenderBufferEntry],
        partial_mode_override: bool | None = None,
        user_input: str = "",
    ) -> tuple[NarratorFinalOutput, TranscriptEntry]:
        """Render this human's POV prose for the beat. Input: their
        buffered events since last render, observation levels tagged.
        Returns `(NarratorFinalOutput, TranscriptEntry)`.

        Returning the transcript entry alongside the envelope lets
        run_beat propagate the per-POV transcript entry up to the
        orchestrator, which appends to ckpt.transcript so /history and
        the resume-display path see the actual played turns. The entry
        is constructed engine-side from `user_input` (passed by the
        acting-POV caller; "" for incidental POVs) and the rendered
        `final_text`.

        `partial_mode_override`, when not None, wins over the dispatcher's
        default slot-scan detection — the v11-r6a Cat II-open render path
        sets this True for pinned humans + initiator so all of them see
        the mid-attempt cliffhanger even though the initiator isn't pinned
        themselves.
        """
        ...


@dataclass
class BeatResult:
    """Summary of a single beat's run, returned to the orchestrator for
    delivery. `renders` is keyed by character_id; orchestrator posts each
    to that player's delivery channel (thread / ephemeral / DM).

    `transcript_entries` is the parallel per-POV TranscriptEntry from
    each narrator render — orchestrator picks the acting player's entry
    (or first available) and appends to ckpt.transcript so /history and
    resume-display see the actual played turns. Empty when no human
    rendered this beat (Cat II pending, or beat-with-no-renderable-events).

    `event_actor_ids` is a list parallel to the tail of
    `ckpt.canonical_events` (length == `events_closed`, in beat-order):
    `event_actor_ids[i]` is the character_id whose intention produced
    `canonical_events[-events_closed + i]`. Empty when
    `events_closed == 0`.

    `reaction_prompts` is runtime UI state: character_id -> closed canonical
    event id for combat reaction windows created by this beat.

    `loot_prompts` is D&D-adapter UI state filled by the orchestrator after
    closed events are inspected for loot offers.
    """
    renders: dict[str, str]
    events_closed: int
    ended_reason: str  # "ends_beat" | "max_events_cap" | "cat_ii_pending" | ...
    transcript_entries: dict[str, TranscriptEntry]
    event_actor_ids: list[str]
    reaction_prompts: dict[str, str] | None = None
    loot_prompts: dict[str, list[str]] | None = None


async def run_beat(
    ckpt: CheckpointFile,
    dispatcher: Dispatcher,
    actor_id: str,
    intention: str,
    cat_ii_event_id: str | None = None,
    combat_reaction_event_id: str | None = None,
) -> BeatResult:
    """Run one beat to completion.

    Entry paths:
    1. Fresh /act from a human — `cat_ii_event_id=None`, actor's intention
       is a fresh initiator. Claim slot, route, cascade.
    2. Cat II responder intention — `cat_ii_event_id` set. Collect the
       intention into the open event. If all required responders are in,
       adjudicate the Cat II (calling dispatcher.route_intention with
       the full open_event); else return immediately with ended_reason
       "cat_ii_pending" (nothing to render yet).
    3. Combat reaction intention — `combat_reaction_event_id` set. The
       actor's live reaction slot is cleared, their reaction is marked spent,
       and the intention is routed as an out-of-turn response.

    Termination:
    - Router's ends_beat=true → render fan-out, slot release.
    - Cat II event adjudicates; NPC initiators get first follow-up,
      otherwise render fan-out and slot release.
    - max_events_per_beat reached → forced render + slot release.
    """
    max_events = ckpt.session.config.settings.max_events_per_beat
    events_closed = 0
    event_actor_ids: list[str] = []
    current_intention = intention
    current_actor = actor_id
    continuation_rescue_used = False
    pending_result: EventRouterOutput | None = None
    suppress_reaction_prompts = combat_reaction_event_id is not None

    async def _queue_router_continuation(
        prior_result: EventRouterOutput,
    ) -> None:
        nonlocal continuation_rescue_used, pending_result
        if continuation_rescue_used:
            raise RuntimeError(
                "Router kept beat open without a dispatchable continuation: "
                "ends_beat=false and no agent_responder_picks after the "
                "continuation rescue."
            )
        continuation_rescue_used = True
        pending_result = await dispatcher.route_continuation(
            ckpt=ckpt,
            actor_id=actor_id,
            prior_result=prior_result,
        )
        _log_router_rationale(
            pending_result, actor_id, kind="continuation",
        )

    async def _pause_for_pending_rolls() -> BeatResult:
        return await _end_beat(
            ckpt, dispatcher,
            ended_reason="cat_ii_pending_rolls",
            events_closed=events_closed,
            event_actor_ids=event_actor_ids,
            release_slots=False,
            force_partial=True,
            acting_player_id=actor_id,
            acting_player_input=intention,
            suppress_reaction_prompts=suppress_reaction_prompts,
        )

    # --- Step 1: handle entry path ------------------------------------------

    if cat_ii_event_id is not None:
        # Responder intention path.
        evt = collect_cat_ii_intention(
            ckpt, cat_ii_event_id, actor_id, intention
        )
        if evt is None:
            # Event already gone — race condition. Treat as no-op.
            return BeatResult(
                renders={}, events_closed=0, ended_reason="cat_ii_stale",
                transcript_entries={}, event_actor_ids=[],
                reaction_prompts={},
            )
        # Free this specific character's slot — others may still be pinned.
        release_character_slot(ckpt, actor_id)
        if not cat_ii_is_ready(evt):
            # Still waiting on other responders. Beat stays paused.
            return BeatResult(
                renders={},
                events_closed=0,
                ended_reason="cat_ii_pending",
                transcript_entries={}, event_actor_ids=[],
                reaction_prompts={},
            )
        # All responders in — adjudicate.
        try:
            resolved = await dispatcher.route_intention(
                ckpt=ckpt,
                actor_id=evt.initiator_id,
                intention=evt.initiator_intention,
                cat_ii_event=evt,
            )
        except DndCatIIRollsPending:
            return await _pause_for_pending_rolls()
        _log_router_rationale(
            resolved, evt.initiator_id, kind="cat_ii_resolve",
        )
        close_cat_ii(ckpt, evt.event_id)
        # Defensive guard: if the router's resolution call ever returns
        # requires_responders=true (contradicting Part C of the prompt),
        # we refuse the nesting rather than silently dropping it. Raise
        # loudly so prompt drift gets caught in playtest.
        if resolved.requires_responders:
            raise ValueError(
                "Router returned Cat II nesting on an adjudication call; "
                "Part C invariant violated."
            )
        align_cat_ii_resolution_time(ckpt, evt, resolved)
        # A Cat II resolution is not a single actor's self-action; it is the
        # adjudicated outcome of every collected intention. Broadcast it to all
        # NPC observers, including the initiator, so agents retain the final
        # result instead of only the player render seeing it.
        broadcast_event(ckpt, resolved)
        events_closed = 1
        event_actor_ids.append(evt.initiator_id)

        initiator_pick = _filter_picks_for_dispatch(
            ckpt, [evt.initiator_id], event=resolved,
        )
        followup = None
        if initiator_pick:
            followup = await _agent_intention_for_dispatch(
                dispatcher, ckpt, evt.initiator_id,
            )
        if followup is None:
            return await _end_beat(
                ckpt, dispatcher,
                ended_reason="cat_ii_resolution",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=suppress_reaction_prompts,
            )

        # The Cat II initiator gets first follow-up after the adjudication.
        # Human initiators cannot be dispatched here, so they naturally fall
        # through to the player render above.
        current_actor = evt.initiator_id
        current_intention = followup

    if combat_reaction_event_id is not None:
        release_character_slot(ckpt, actor_id)
        _mark_combat_reaction_spent(ckpt, actor_id)
        current_intention = _combat_reaction_intention(
            ckpt,
            actor_id=actor_id,
            trigger_event_id=combat_reaction_event_id,
            intention=intention,
        )
        if _character_in_dnd_active_combat(ckpt, actor_id):
            try:
                resolved = await dispatcher.route_combat_action(
                    ckpt=ckpt,
                    actor_id=actor_id,
                    intention=current_intention,
                )
            except DndCatIIRollsPending:
                return await _pause_for_pending_rolls()
            _log_router_rationale(
                resolved, actor_id, kind="dnd_combat_reaction_resolve",
            )
            if resolved.requires_responders:
                raise ValueError(
                    "D&D combat reaction returned generic Cat II; combat "
                    "resolver must close the reaction or pause for player "
                    "rolls."
                )
            broadcast_event(ckpt, resolved, actor_id=actor_id)
            event_actor_ids.append(actor_id)
            events_closed += 1
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason=resolved.ends_beat_reason
                or "ruleset_resolution",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=True,
            )

    # Fresh initiator path.
    if cat_ii_event_id is None and combat_reaction_event_id is None:
        claim_initiator_slot(ckpt, actor_id)
        if _character_in_dnd_active_combat(ckpt, actor_id):
            try:
                resolved = await dispatcher.route_combat_action(
                    ckpt=ckpt,
                    actor_id=actor_id,
                    intention=intention,
                )
            except DndCatIIRollsPending:
                return await _pause_for_pending_rolls()
            _log_router_rationale(
                resolved, actor_id, kind="dnd_combat_resolve",
            )
            if resolved.requires_responders:
                raise ValueError(
                    "D&D combat resolution returned generic Cat II; combat "
                    "resolver must close the turn or pause for player rolls."
                )
            _clear_pending_initiating_action(ckpt, actor_id)
            broadcast_event(ckpt, resolved, actor_id=actor_id)
            event_actor_ids.append(actor_id)
            events_closed += 1
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason=resolved.ends_beat_reason
                or "ruleset_resolution",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
            )

    while True:
        if pending_result is None:
            # Route the current intention.
            result = await dispatcher.route_intention(
                ckpt=ckpt,
                actor_id=current_actor,
                intention=current_intention,
                cat_ii_event=None,
            )
            result_actor_id = current_actor
            result_is_continuation = False
            _log_router_rationale(
                result, current_actor, kind="route",
            )
        else:
            result = pending_result
            pending_result = None
            result_actor_id = ""
            result_is_continuation = True

        interaction_mode = _dnd_interaction_mode(result)
        if interaction_mode == "dnd_combat_start":
            if result_is_continuation:
                raise RuntimeError(
                    "Router continuation tried to start D&D combat; only a "
                    "fresh intention can start initiative."
                )
            signal_actor_id = result_actor_id or current_actor or actor_id
            if _active_combat(ckpt) is not None:
                _block_dnd_combat_start_from_router_signal(
                    ckpt, result, actor_id=signal_actor_id,
                )
                broadcast_event(ckpt, result, actor_id=signal_actor_id)
                event_actor_ids.append(signal_actor_id)
                events_closed += 1
                return await _end_beat(
                    ckpt,
                    dispatcher,
                    ended_reason="combat_start_blocked",
                    events_closed=events_closed,
                    event_actor_ids=event_actor_ids,
                    acting_player_id=actor_id,
                    acting_player_input=intention,
                    suppress_reaction_prompts=True,
                )
            started = _start_dnd_combat_from_router_signal(
                ckpt,
                result,
                actor_id=signal_actor_id,
                intention=current_intention,
            )
            broadcast_event(ckpt, result, actor_id=signal_actor_id)
            event_actor_ids.append(signal_actor_id)
            events_closed += 1
            if not started:
                return await _end_beat(
                    ckpt,
                    dispatcher,
                    ended_reason="state_change",
                    events_closed=events_closed,
                    event_actor_ids=event_actor_ids,
                    acting_player_id=actor_id,
                    acting_player_input=intention,
                    suppress_reaction_prompts=True,
                )
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason="combat_started",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=True,
            )

        if interaction_mode == "dnd_combat_end":
            signal_actor_id = result_actor_id or current_actor or actor_id
            _end_dnd_combat_from_router_signal(
                ckpt, result, actor_id=signal_actor_id,
            )
            broadcast_event(ckpt, result, actor_id=signal_actor_id)
            event_actor_ids.append(signal_actor_id)
            events_closed += 1
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason=result.ends_beat_reason or "state_change",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=True,
            )

        if result.requires_responders:
            suppressed_actor_id = result_actor_id or current_actor or actor_id
            required = [
                r for r in result.required_responders if r != result_actor_id
            ]
            if _character_in_active_combat(ckpt, suppressed_actor_id):
                logger.warning(
                    "Router emitted generic Cat II during active combat; "
                    "suppressing responder flow for actor=%s required=%s",
                    suppressed_actor_id,
                    result.required_responders,
                )
                result.requires_responders = False
                result.required_responders = []
                result.agent_responder_picks = []
                result.ends_beat = True
                result.ends_beat_reason = "ruleset_cat_ii_suppressed"
                broadcast_event(ckpt, result, actor_id=suppressed_actor_id)
                event_actor_ids.append(suppressed_actor_id)
                events_closed += 1
                return await _end_beat(
                    ckpt, dispatcher,
                    ended_reason="ruleset_cat_ii_suppressed",
                    events_closed=events_closed,
                    event_actor_ids=event_actor_ids,
                    acting_player_id=actor_id,
                    acting_player_input=intention,
                    suppress_reaction_prompts=True,
                )
            if result_is_continuation:
                raise RuntimeError(
                    "Router continuation opened Cat II; continuation mode "
                    "must create a closed cue or pick a dispatchable NPC."
                )
            # Cat II: open the event, pin humans, request agent
            # responder intentions immediately (they're fast).
            # Filter: the initiator can never be their own responder,
            # even if the router hallucinates it — would either double-
            # pin them or overwrite the initiator slot.
            if not required:
                # The only "responder" was the initiator themselves; treat
                # this as Cat I — there's nothing to contest. Broadcast the
                # canonical event as-is and continue.
                broadcast_event(ckpt, result, actor_id=result_actor_id)
                event_actor_ids.append(result_actor_id)
                events_closed += 1
                if events_closed >= max_events:
                    return await _end_beat(
                        ckpt, dispatcher,
                        ended_reason="max_events_cap",
                        events_closed=events_closed,
                        event_actor_ids=event_actor_ids,
                        acting_player_id=actor_id,
                        acting_player_input=intention,
                        suppress_reaction_prompts=suppress_reaction_prompts,
                    )
                # Fall through to the standard Cat I ends_beat check.
                if result.ends_beat:
                    return await _end_beat(
                        ckpt, dispatcher,
                        ended_reason=result.ends_beat_reason
                            or "cascade_exhausted",
                        events_closed=events_closed,
                        event_actor_ids=event_actor_ids,
                        acting_player_id=actor_id,
                        acting_player_input=intention,
                        suppress_reaction_prompts=suppress_reaction_prompts,
                    )
                # Keep cascading via picks — reuse the normal path by
                # letting the loop iterate. Break out of the inner if
                # and continue. The helper strips human-controlled ids;
                # NPC eligibility stays with the router.
                picks = _filter_picks_for_dispatch(
                    ckpt, result.agent_responder_picks,
                    event=result,
                )
                if not picks:
                    await _queue_router_continuation(result)
                    continue
                next_actor = picks[0]
                next_intention = await dispatcher.agent_intend(
                    ckpt=ckpt, character_id=next_actor,
                )
                current_actor = next_actor
                current_intention = next_intention
                continue

            evt = open_cat_ii(
                ckpt=ckpt,
                initiator_id=result_actor_id,
                initiator_intention=current_intention,
                required_responders=required,
                opening_event=result,
            )

            # The router's Cat II-open output is already the canonical
            # attempt-in-progress: visible setup, dialogue, gestures,
            # and fact-level visibility are all in `observable_facts`.
            # Broadcast it before collecting responder intentions so
            # every observer receives the same public attempt that the
            # responder is reacting to. Pre-r12 synthesized an
            # observerless stub here and dropped the router's facts.
            result.ends_beat = True
            result.ends_beat_reason = "cat_ii_open"
            broadcast_event(ckpt, result, actor_id=result_actor_id)
            event_actor_ids.append(result_actor_id)
            events_closed += 1

            # Agents among required_responders intend immediately; their
            # intentions are collected into the same Cat II event.
            bindings = ckpt.session.character_bindings or {}
            for rid in required:
                if rid in bindings:
                    # Human — pin slot, wait.
                    pin_cat_ii_responder(ckpt, rid, evt.event_id)
                else:
                    # Agent — intend inline.
                    ai_intent = await dispatcher.agent_intend(
                        ckpt=ckpt,
                        character_id=rid,
                    )
                    collect_cat_ii_intention(
                        ckpt, evt.event_id, rid, ai_intent
                    )

            if cat_ii_is_ready(evt):
                # No humans in required list — resolve immediately.
                try:
                    resolved = await dispatcher.route_intention(
                        ckpt=ckpt,
                        actor_id=evt.initiator_id,
                        intention=evt.initiator_intention,
                        cat_ii_event=evt,
                    )
                except DndCatIIRollsPending:
                    return await _pause_for_pending_rolls()
                _log_router_rationale(
                    resolved, evt.initiator_id,
                    kind="cat_ii_resolve_inline",
                )
                close_cat_ii(ckpt, evt.event_id)
                if resolved.requires_responders:
                    raise ValueError(
                        "Router returned Cat II nesting on an adjudication "
                        "call; Part C invariant violated."
                    )
                align_cat_ii_resolution_time(ckpt, evt, resolved)
                # Cat II resolution belongs to every participant in the
                # contest, so do not exclude the initiator from the observer
                # inbox fan-out.
                broadcast_event(ckpt, resolved)
                event_actor_ids.append(evt.initiator_id)
                events_closed += 1
                initiator_pick = _filter_picks_for_dispatch(
                    ckpt, [evt.initiator_id], event=resolved,
                )
                followup = None
                if initiator_pick:
                    followup = await _agent_intention_for_dispatch(
                        dispatcher, ckpt, evt.initiator_id,
                    )
                if followup is not None:
                    current_actor = evt.initiator_id
                    current_intention = followup
                    continue
                return await _end_beat(
                    ckpt, dispatcher,
                    ended_reason="cat_ii_resolution",
                    events_closed=events_closed,
                    event_actor_ids=event_actor_ids,
                    acting_player_id=actor_id,
                    acting_player_input=intention,
                    suppress_reaction_prompts=suppress_reaction_prompts,
                )
            # Humans are pinned — pause the beat here. Their /acts will
            # re-enter run_beat with their cat_ii_event_id.
            #
            # v11-r6a/r12: render the already-broadcast router event in
            # PARTIAL mode and keep pins alive. All humans with buffered
            # observations render, not just the responder and initiator;
            # NPC observers already received their visible facts through
            # `broadcast_event`.
            return await _end_beat(
                ckpt, dispatcher,
                ended_reason="cat_ii_pending",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                release_slots=False,
                force_partial=True,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=suppress_reaction_prompts,
            )

        # Cat I path — canonical event closes immediately.
        broadcast_event(ckpt, result, actor_id=result_actor_id)
        event_actor_ids.append(result_actor_id)
        events_closed += 1

        # v11-r8a: observation_harvest fork. The router signals
        # `ends_beat_reason="observation_harvest"` when the actor is
        # purely observing perceptually available NPCs (looking,
        # studying, scanning
        # without dialogue or contact). We bypass the cascade and
        # instead fire each pick's `perceive()` in parallel to harvest
        # one self-presentation fragment per target. Fragments are
        # appended to the just-broadcast event's `observable_facts`
        # so the narrator's render reads them naturally as part of
        # the event's perceptual surface.
        #
        # Mutating the canonical event after broadcast is safe:
        # `broadcast_event` only takes references (event_id into render
        # buffers) and the narrator reads the live
        # `observable_facts` list at compose time. The harvest path uses
        # the same human-only pick guard as the cascade path.
        if result.ends_beat_reason == "observation_harvest":
            harvest_picks = _filter_picks_for_dispatch(
                ckpt, result.agent_responder_picks,
                event=result,
            )
            await _append_harvest_fragments(
                dispatcher, ckpt, result,
                picks=harvest_picks,
                current_actor=current_actor,
                log_label="Observation harvest",
            )
        elif result.ends_beat_reason == "query_response":
            query_picks = _filter_picks_for_dispatch(
                ckpt, result.agent_responder_picks,
                event=result,
            )
            if query_picks:
                await _append_harvest_fragments(
                    dispatcher, ckpt, result,
                    picks=query_picks,
                    current_actor=current_actor,
                    log_label="Query harvest",
                )

        # Ends-beat decision.
        if result.ends_beat or events_closed >= max_events:
            reason = result.ends_beat_reason or "max_events_cap"
            if events_closed >= max_events and not result.ends_beat:
                reason = "max_events_cap"
            return await _end_beat(
                ckpt, dispatcher,
                ended_reason=reason,
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=suppress_reaction_prompts,
            )

        # Pick the next actor for the cascade. Agents only — humans
        # don't continue a beat unless they /act fresh. NPC location /
        # perception eligibility is owned by the router; the engine only
        # strips human-controlled ids here.
        picks = _filter_picks_for_dispatch(
            ckpt, result.agent_responder_picks,
            event=result,
        )
        if not picks:
            await _queue_router_continuation(result)
            continue
        # v11 first cut: chain one pick at a time. Multi-pick fan-out can
        # come later; for now, the first pick acts, we re-route, the
        # router can emit fresh picks on its next output.
        # Try picks in order; skip any that return an empty/refusal
        # intention (agent failure mode: an unconfigured model emits "",
        # a whitespace string, or a refusal sentinel). Failed picks are
        # silently dropped; if all fail, the beat ends as cascade_
        # exhausted rather than routing garbage through the adjudicator.
        next_actor = None
        next_intention: str | None = None
        for candidate in picks:
            raw = await _agent_intention_for_dispatch(
                dispatcher, ckpt, candidate,
            )
            if raw is not None:
                next_actor = candidate
                next_intention = raw
                break
        if next_actor is None or next_intention is None:
            await _queue_router_continuation(result)
            continue
        current_actor = next_actor
        current_intention = next_intention


async def _end_beat(
    ckpt: CheckpointFile,
    dispatcher: Dispatcher,
    ended_reason: str,
    events_closed: int,
    event_actor_ids: list[str],
    *,
    release_slots: bool = True,
    force_partial: bool = False,
    render_only: set[str] | None = None,
    acting_player_id: str | None = None,
    acting_player_input: str = "",
    suppress_reaction_prompts: bool = False,
) -> BeatResult:
    """Compose per-human renders, flush buffers, and optionally release
    beat slots.

    v11-r6a params:
    - `release_slots=False`: used by the Cat II-open render path so the
      pins we JUST created stay live until responders /act. The normal
      end-of-beat path leaves this True.
    - `force_partial=True`: every rendered POV gets partial_mode=True
      regardless of the dispatcher's default slot-scan detection. Used
      by the Cat II-open render path so the initiator (who isn't
      pinned) also sees the cliffhanger.
    - `render_only=set(...)`: only render POVs whose character_id is in
      this set. Reserved for callers that need narrower fan-out; the
      Cat II-open path deliberately leaves this unset so every human
      observer with a buffered event sees the attempted action.

    v11-r7j params:
    - `acting_player_id` + `acting_player_input`: the actual /act'd
      player and their verbatim utterance. Threaded into the matching
      POV's narrator call so the engine-built TranscriptEntry carries
      the real input. Other POVs (incidental human observers) get
      `user_input=""`. Pre-r7j the LLM owned the transcript entry and
      no real input ever made it that far.
    """
    renders: dict[str, str] = {}
    transcript_entries: dict[str, TranscriptEntry] = {}
    from app.engine.context_builder import collect_player_ids
    player_ids = collect_player_ids(ckpt)
    reaction_prompts = _eligible_combat_reaction_prompts(
        ckpt,
        events_closed=events_closed,
        event_actor_ids=event_actor_ids,
        suppress_reaction_prompts=suppress_reaction_prompts,
    )
    candidates = [
        h for h, buf in ckpt.session.render_buffers.items()
        if h in player_ids and buf
    ]
    if render_only is not None:
        candidates = [h for h in candidates if h in render_only]
    partial_override: bool | None = True if force_partial else None

    # v11-r6c: fan out narrator calls in parallel. Independent POVs; no
    # shared state to mutate mid-call. Trims a 3-human render from 3×
    # narrator latency to 1× (slowest POV). Buffers are flushed up front
    # so the concurrent tasks see consistent inputs and no task observes
    # a race against another's flush.
    targets: list[tuple[str, list[RenderBufferEntry]]] = []
    for h in candidates:
        buf = flush_render_buffer(ckpt, h)
        if not buf:
            continue  # Human had no perceivable events this beat.
        targets.append((h, buf))

    async def _render_one(
        h: str, buf: list[RenderBufferEntry],
    ) -> tuple[str, NarratorFinalOutput, TranscriptEntry]:
        pov_user_input = (
            acting_player_input if h == acting_player_id else ""
        )
        envelope, entry = await dispatcher.narrator_compose(
            ckpt=ckpt,
            character_id=h,
            buffered_events=buf,
            partial_mode_override=partial_override,
            user_input=pov_user_input,
        )
        return h, envelope, entry

    if targets:
        results = await asyncio.gather(
            *(_render_one(h, buf) for h, buf in targets)
        )
        for h, envelope, entry in results:
            renders[h] = envelope.final_text
            transcript_entries[h] = entry

    if release_slots:
        release_beat_slots(ckpt)
    max_events = ckpt.session.config.settings.max_events_per_beat
    if events_closed > max_events:
        (
            cat_ii_open,
            cat_ii_resolution,
            cat_ii_followup,
        ) = _beat_cap_overrun_cause(ckpt, events_closed, ended_reason)
        logger.warning(
            "Beat cap overrun: configured_cap=%d events_rendered=%d "
            "ended_reason=%s cat_ii_open_likely=%s "
            "cat_ii_resolution_likely=%s cat_ii_followup_likely=%s",
            max_events,
            events_closed,
            ended_reason,
            cat_ii_open,
            cat_ii_resolution,
            cat_ii_followup,
        )
    final_reason = (
        "combat_reaction_pending" if reaction_prompts else ended_reason
    )
    logger.info(
        "Beat closed: events=%d reason=%s renders=%d "
        "release_slots=%s force_partial=%s reaction_prompts=%d",
        events_closed, final_reason, len(renders),
        release_slots, force_partial, len(reaction_prompts),
    )
    return BeatResult(
        renders=renders, events_closed=events_closed, ended_reason=final_reason,
        transcript_entries=transcript_entries,
        event_actor_ids=event_actor_ids,
        reaction_prompts=reaction_prompts,
    )


# ---- Error-message shaping for rejected /acts ------------------------------


def format_slot_rejection(
    check: SlotCheck,
    ckpt: CheckpointFile,
    attempted_text: str = "",
) -> str:
    """Render a user-facing message explaining why an /act was rejected.

    The SlotCheck carries the holder's character_id; we look up their
    display name from the roster to make the message friendly. If
    `attempted_text` is provided, the rejection echoes it back so the
    player can copy-paste on retry — solves the "Discord ate my three
    sentences" pain point.
    """
    holder_name = check.holder_id
    for c in ckpt.characters:
        if c.character_id == check.holder_id:
            holder_name = c.name
            break

    base: str
    if check.conflict == SlotConflict.INITIATOR_HELD:
        base = (
            f"**{holder_name}** is taking their turn — your /act didn't "
            f"go through. You'll see a new render appear when the beat "
            f"re-opens; try again then."
        )
    elif check.conflict == SlotConflict.CAT_II_OTHER_HELD:
        base = (
            f"The beat is paused on **{holder_name}** — someone's action "
            f"is waiting on their response. Your /act didn't go through. "
            f"You'll see a new render when the beat closes."
        )
    elif check.conflict == SlotConflict.COMBAT_REACTION_OTHER_HELD:
        base = (
            f"The beat is paused on **{holder_name}**'s possible reaction. "
            f"Your /act didn't go through. They'll use /act to react or "
            f"press **No reaction** to pass."
        )
    elif check.conflict == SlotConflict.SELF_BUSY:
        base = (
            "Your previous /act is still processing. Give the beat a "
            "moment before submitting again."
        )
    elif check.conflict == SlotConflict.CAT_II_SELF_ROLL:
        base = (
            "The beat is waiting on your dice roll, not another /act. "
            "Use the roll prompt for the current contested action."
        )
    elif check.conflict == SlotConflict.CAT_II_SELF_RESPONDER:
        # Not an error — this is the "your /act was accepted as your Cat II
        # response" case. Callers usually don't render this path, but
        # provide a friendly confirmation if they do.
        base = (
            "Your /act was accepted as your response to the current contested "
            "action. The beat will resolve once all responders have moved."
        )
    elif check.conflict == SlotConflict.COMBAT_REACTION_SELF:
        base = (
            "Your /act was accepted as your combat reaction. The beat will "
            "continue after the reaction resolves."
        )
    else:
        base = "Your /act could not be accepted right now."

    if attempted_text:
        # Echo the player's text so they don't lose it. Truncate long
        # inputs so the rejection doesn't run off the screen; blockquote
        # so they can easily copy-paste. Discord's raw message limit is
        # ~2000 chars; 1500 leaves headroom for the rejection preamble.
        preview = attempted_text.strip()
        if len(preview) > 1500:
            preview = preview[:1497] + "..."
        base += f"\n\n> Your submitted text:\n> {preview}"
    return base
