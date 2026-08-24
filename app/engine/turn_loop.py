"""v11 turn loop — DM-paced, beat-based, Cat I/II intention pipeline.

This module codifies the Plan-v11 state machine for the turn pipeline.
It replaces the one-shot `process_turn → single narrator render` shape
with a beat-cascading loop: intentions flow in, are classified Cat I
(self-closing) or Cat II (contested responder-collecting), canonical
events are adjudicated, broadcast to observers in the shared perceptual
frame, and the narrator controls whether the player regains control before a
speculatively prepared autonomous reaction becomes canonical.

## State machine summary

  enter(intention, actor_id):
    validate against session active_act_slots — claim, reject, or
    interpret-as-Cat-II-responder.

  classify(intention) → Cat I | Cat II
    Cat I → canonicalize, broadcast, check `event_kind`.
    Cat II → open event, collect required responders (agents intend
      immediately; humans pin slot and wait), adjudicate on close,
      then route the resolution's selected follow-up NPCs, falling back to the
      initiator only when the resolution did not select anyone.

  broadcast(event):
    Append to every local human's render buffer, plus mediated humans
    explicitly named by visible facts.
    Feed local agents, plus mediated agents explicitly named by visible
    facts, through their observation context.

  presentation gate:
    After each closed event, ask the narrator whether the player should regain
    control while speculatively preparing any autonomous `next_output` in
    parallel. A render discards that speculative state while the canonical
    event retains the semantic handoff for a possible player `(defer)`. A
    continue judgment commits the prepared branch and resumes the cascade.

## Runtime wiring

The orchestrator binds this state machine to the concrete router,
character-agent, and narrator modules through the `Dispatcher` protocol.
Tests can pass fakes through the same protocol, which keeps the beat loop
isolated without implying a second runtime path.

## Terminology

  Beat — a stretch of canonical events that closes on terminal `event_kind`.
  active_act_slot — the session's beat gate, holding 0..N character-slot entries.
  Cat I — self-closing intention (dialogue, passive, unambiguous).
  Cat II — contested intention, collects responder intentions.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from app.engine import (
    dnd_combat,
    dnd_monsters,
    dnd_spatial,
    imported_encounters,
    imported_statblocks,
)
from app.engine.dnd_cat_ii import DndCatIIRollsPending
from app.engine.narrator import commit_pov_render
from app.engine.dnd_combat_access import (
    checkpoint_active_combat,
    combatant_character_id as _combatant_character_id,
    combatant_defeat_state as _combatant_defeat_state,
    combatant_for_character as _combatant_for_character,
    combatants as _combatants,
    current_combatant as _current_combatant,
    obj_get as _obj_get,
    obj_set as _obj_set,
)
from app.engine.text_safety import strip_terminal_control
from app.engine.visual_context import (
    AGENT_FIRST_MEETING_CAP,
    format_visual_introductions,
    mark_visual_introductions,
    plan_event_visual_introductions,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    is_non_social_hazard,
)
from app.schemas.event_router import (
    EventRouterOutput,
    ObserverEntry,
    empty_commitment_open_signal,
)
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    WorldAdjudication,
    visible_fact_texts,
)
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.router_targets import (
    RouterOutputTarget,
    targets_from_router_output,
)
from app.schemas.state import (
    CommitmentRevisionPrompt,
    OpenCatIIEvent,
    OpenCommitment,
    PendingNarratorRender,
    RenderBufferEntry,
    SlotEntry,
)

logger = logging.getLogger(__name__)
narrator_handoff_logger = logging.getLogger(
    f"{__name__}.narrator_handoff"
)
# Handoff reasons are the narrator's durable pacing diagnostic. The CLI keeps
# its ordinary module loggers at ERROR unless --verbose is enabled, so give
# this narrow child logger an explicit level and let it propagate to the
# configured persistent file handler.
narrator_handoff_logger.setLevel(logging.INFO)

MAX_BACKGROUND_THREADS_PER_BEAT = 4

FORCED_HANDOFF_EVENT_KINDS = {
    "cat_ii_open",
    "cat_ii_resolution",
    "max_events_cap",
    "observation_harvest",
    "query_response",
    "ruleset_resolution",
    "ruleset_cat_ii_suppressed",
}


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


def purge_character_state(
    ckpt: CheckpointFile,
    character_id: str,
    *,
    preserve_render_buffer: bool = False,
) -> None:
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
    - Empties their render buffer unless a pre-presentation terminal adapter
      commit explicitly preserves already-queued POV events for narration.
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

    # 3. Render buffer. A character who dies during a multi-event beat still
    # receives the final lossless POV rendering; ordinary leave/cull paths keep
    # the historical behavior of clearing their undeliverable queue.
    if not preserve_render_buffer:
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
        # An empty or malformed `opened_at` (hand-edited or migration-authored
        # checkpoints; the schema default is "") is treated as fully stale so
        # the event auto-resolves rather than wedging every other player's
        # /act behind a CAT_II_OTHER_HELD pin forever.
        if not _pin_stamp_is_stale(evt.opened_at, now, timeout):
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


def sweep_stale_combat_reaction_pins(
    ckpt: CheckpointFile,
    now_iso: str | None = None,
) -> list[str]:
    """Auto-pass human D&D combat-reaction pins that have outlived the
    human AFK timeout.

    A `combat_reaction` slot pins a human for an OPTIONAL reaction. If that
    human never answers (`/act` or `/defer`), `_blocking_slots_open` stays
    true and any delayed initiative advance never fires — the table wedges
    on someone who may simply be AFK. This mirrors the Cat II AFK sweep:
    once a reaction pin is older than `cat_ii_human_timeout_seconds` (the
    shared human-AFK knob), release it as "no reaction recorded." Advancing
    initiative afterward is the caller's job — it lives in the orchestrator
    and must not be imported here.

    Returns the character ids whose reaction pins were released. An empty or
    unparseable `claimed_at` is treated as fully stale so a malformed pin
    cannot wedge the table forever. Set the timeout to 0 to disable.
    """
    timeout = ckpt.session.config.settings.cat_ii_human_timeout_seconds
    if timeout <= 0:
        return []

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc) if now_iso is None else _parse_iso(now_iso)

    released: list[str] = []
    for character_id, entry in list(ckpt.session.active_act_slots.items()):
        if entry.reason != "combat_reaction":
            continue
        if not _pin_stamp_is_stale(entry.claimed_at, now, timeout):
            continue
        release_character_slot(ckpt, character_id)
        released.append(character_id)
        logger.warning(
            "Combat reaction pin on %s auto-passed after AFK timeout",
            character_id,
        )
    return released


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


def _pin_stamp_is_stale(stamp: str, now, timeout_seconds: float) -> bool:
    """Shared AFK-sweep predicate for Cat II `opened_at` and combat-reaction
    `claimed_at`. A stamp is stale when it is empty or unparseable (we cannot
    prove the pin is fresh, and a malformed stamp must never wedge the table)
    or when its age has reached the timeout."""
    if not stamp:
        return True
    try:
        stamped = _parse_iso(stamp)
    except Exception:
        return True
    return (now - stamped).total_seconds() >= timeout_seconds


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
    ckpt: CheckpointFile | None = None,
    resolved_cat_ii: OpenCatIIEvent | None = None,
) -> None:
    """Surface the router's terse rationale at INFO.

    `decision_rationale` is part of the router output contract so playtest
    logs show "why" alongside "what" for every route_intention site (Cat I,
    Cat II open, Cat II resolve) without scattering log calls inline.

    `kind` distinguishes the routing context — "route" (normal Cat I
    / Cat II open call), "cat_ii_resolve" (final adjudication of a
    closed Cat II), so log readers can tell them apart.
    """
    rationale = (result.decision_rationale or "").strip()
    if not rationale:
        rationale = "(no rationale emitted)"
    forced = _event_requires_forced_handoff(
        result,
        ckpt=ckpt,
        resolved_cat_ii=resolved_cat_ii,
    )
    logger.info(
        "router[%s] actor=%s cat=%s forced_handoff=%s kind=%r next=%s "
        "enrich=%s :: %s",
        kind, actor_id,
        "II" if result.requires_responders else "I",
        forced, result.event_kind,
        result.next_output_character_ids,
        result.perception_enrichment_character_ids,
        rationale,
    )


def _event_requires_forced_handoff(
    result: EventRouterOutput,
    *,
    ckpt: CheckpointFile | None = None,
    resolved_cat_ii: OpenCatIIEvent | None = None,
) -> bool:
    """Whether this closed event is an unconditional presentation boundary.

    A Cat II involving a bound participant remains mandatory. When every
    participant in the resolved contest is autonomous, a bound observer is
    only watching the exchange and gets the ordinary narrator pacing gate.
    Callers without the resolved contest retain the conservative forced
    behavior.
    """
    if result.event_kind not in FORCED_HANDOFF_EVENT_KINDS:
        return False
    if (
        result.event_kind != "cat_ii_resolution"
        or ckpt is None
        or resolved_cat_ii is None
    ):
        return True

    from app.engine.context_builder import collect_player_ids

    participant_ids = {
        resolved_cat_ii.initiator_id,
        *resolved_cat_ii.required_responders,
    }
    return not participant_ids.isdisjoint(collect_player_ids(ckpt))


def _event_handoff_reason(result: EventRouterOutput) -> str:
    return result.event_kind if result.event_kind != "beat_continues" else ""


def _filter_routed_agents_for_dispatch(
    ckpt: CheckpointFile,
    character_ids: list[str],
    event: EventRouterOutput | None = None,
) -> list[str]:
    """Drop routed character ids the engine must never dispatch as agent turns.

    Humans do not cascade via the router; they only enter through
    `/act`. Using `collect_player_ids` here (rather than the bare
    bindings dict) is the load-bearing fix for the playable-2 era:
    `session.player_character_id` may name the creator's bound
    character without there being a corresponding entry in
    `character_bindings` (older saves; CLI single-player flows).
    Filtering on bindings alone let those creator-bound characters
    slip into NPC dispatch and produced the "router tried to make my
    own character speak" symptom.

    Routed NPCs are intentionally not filtered by physical location or
    fact-level visibility here. Whether an off-location NPC should act
    is a router decision; if it routes a remote producer, caller, spirit,
    watcher, or other mediated participant, we dispatch that NPC.

    The hard filters here are runtime safety constraints, not pacing:
    humans, inactive characters, disabled-agent records, pinned actors,
    and active combatants cannot be advanced by router-picked agent output
    turns.

    Returns the filtered list preserving router order.
    """
    # Local import to avoid an engine-package import cycle on module
    # load (context_builder pulls some of the same schemas turn_loop
    # exports). Cheap — the function is tiny and the import is cached.
    from app.engine.context_builder import (
        collect_player_ids,
        is_unbound_player_authored_slot,
    )

    humans = collect_player_ids(ckpt)
    by_id = _character_by_id(ckpt)
    observer_ids = {
        observer.character_id
        for observer in (event.observers if event is not None else [])
    }
    combat = _active_combat(ckpt)
    combat_ids = (
        {
            _combatant_character_id(combatant)
            for combatant in _combatants(combat)
        }
        if combat is not None else set()
    )
    blocked_slot_ids = {
        cid
        for cid, entry in ckpt.session.active_act_slots.items()
        if entry.reason in {
            "cat_ii_responder",
            "cat_ii_roll",
            "combat_reaction",
            "combat_blocked",
        }
    }
    out: list[str] = []
    for rid in character_ids:
        if rid in out or rid in humans or rid in blocked_slot_ids:
            continue
        char = by_id.get(rid)
        if char is None:
            logger.warning("router picked unknown agent id %s; dropped", rid)
            continue
        if is_non_social_hazard(char):
            logger.error(
                "router picked non-social hazard %s for a character-agent "
                "turn; dropped",
                rid,
            )
            continue
        if is_unbound_player_authored_slot(ckpt, char):
            logger.error(
                "router picked unbound player-authored slot %s; dropped",
                rid,
            )
            continue
        raw_status = getattr(char, "status", "")
        status = str(getattr(raw_status, "value", raw_status))
        if status != "active":
            continue
        if (
            event is not None
            and rid not in observer_ids
            and event.event_kind not in {
                "query_response",
                "observation_harvest",
            }
            and not getattr(
                getattr(char, "private_state", None),
                "intentions_enabled",
                False,
            )
        ):
            continue
        if rid in combat_ids:
            continue
        out.append(rid)
    return out


def _validate_non_social_hazard_routing(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
) -> None:
    """Reject character-owned work assigned to a patterned scene hazard.

    A non-social hazard may remain an ``observe_only`` recipient so the
    canonical ledger can record what its sensors or triggers registered. It
    never produces an intention, self-presentation fragment, player-facing
    reaction, or private commitment revision. Its established behavior is
    authored directly as environmental pressure.
    """
    hazard_ids = {
        character.character_id
        for character in ckpt.characters
        if is_non_social_hazard(character)
    }
    if not hazard_ids:
        return

    violations: list[str] = []
    for character_id in event.required_responders:
        if character_id in hazard_ids:
            violations.append(f"required_responder={character_id}")
    for observer in event.observers:
        if (
            observer.character_id in hazard_ids
            and observer.routing_role != "observe_only"
        ):
            violations.append(
                f"routing_role={observer.routing_role}:"
                f"{observer.character_id}"
            )
    for character_id in event.commitment_open.actor_ids:
        if character_id in hazard_ids:
            violations.append(f"commitment_open={character_id}")
    for signal in event.commitment_resolutions:
        for character_id in signal.actor_ids:
            if character_id in hazard_ids:
                violations.append(f"commitment_resolution={character_id}")
    for signal in event.commitment_interrupts:
        for character_id in signal.actor_ids:
            if character_id in hazard_ids:
                violations.append(f"commitment_interrupt={character_id}")

    if violations:
        raise RuntimeError(
            "Router event assigned character-owned work to a non-social "
            "hazard: " + ", ".join(dict.fromkeys(violations))
        )


async def _materialize_router_spawns_for_dispatch(
    dispatcher: Dispatcher,
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    actor_id: str,
    character_ids: list[str],
    contract_label: str,
    require_all: bool,
) -> None:
    """Materialize same-output spawns needed before an in-beat dispatch.

    Router-authored `spawn` is normally a post-beat orchestrator side effect,
    but both Cat II responder collection and next-output cascades happen inside
    `run_beat`. Materialize any selected ids backed by this output's spawn
    requests before broadcasting the event, so the new character receives its
    visible facts before the character agent is called.

    Cat II requires every missing responder to be backed by a spawn request.
    An unrelated unknown next-output id is not materializable. A matching
    spawn that fails to materialize is a loud runtime contract violation in
    either mode.
    """
    known_ids = {char.character_id for char in ckpt.characters}
    missing = [cid for cid in character_ids if cid not in known_ids]
    if not missing:
        return

    spawn_ids = {
        spawn.character_id
        for spawn in result.spawn
        if spawn.character_id
    }
    materializable = [cid for cid in missing if cid in spawn_ids]
    if materializable:
        materializer = getattr(dispatcher, "materialize_spawns", None)
        if materializer is None:
            raise RuntimeError(
                f"{contract_label} are not in the roster and the dispatcher "
                "cannot materialize router spawns before dispatch: "
                + ", ".join(materializable)
            )
        await materializer(
            ckpt=ckpt,
            result=result,
            actor_id=actor_id,
            character_ids=materializable,
        )

    known_ids = {char.character_id for char in ckpt.characters}
    must_exist = missing if require_all else materializable
    still_missing = [cid for cid in must_exist if cid not in known_ids]
    if still_missing:
        raise RuntimeError(
            f"{contract_label} are not in the roster and were not spawned by "
            "the router output: "
            + ", ".join(still_missing)
        )


async def _materialize_required_responder_spawns(
    dispatcher: Dispatcher,
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    actor_id: str,
    required: list[str],
) -> None:
    await _materialize_router_spawns_for_dispatch(
        dispatcher,
        ckpt,
        result,
        actor_id=actor_id,
        character_ids=required,
        contract_label="Cat II required responders",
        require_all=True,
    )


async def _materialize_next_output_spawns(
    dispatcher: Dispatcher,
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    actor_id: str,
) -> None:
    await _materialize_router_spawns_for_dispatch(
        dispatcher,
        ckpt,
        result,
        actor_id=actor_id,
        character_ids=result.next_output_character_ids,
        contract_label="Router next-output characters",
        require_all=False,
    )


def _binding_aware_next_output_targets(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
) -> list[tuple[str, str]]:
    """Resolve semantic next-output ids to runtime handoff kinds in order."""
    from app.engine.context_builder import collect_player_ids

    bound_ids = collect_player_ids(ckpt)
    targets: list[tuple[str, str]] = []
    for character_id in event.next_output_character_ids:
        if character_id in bound_ids:
            targets.append(("bound", character_id))
            continue
        autonomous = _filter_routed_agents_for_dispatch(
            ckpt,
            [character_id],
            event=event,
        )
        if autonomous:
            targets.append(("autonomous", autonomous[0]))
    return targets


def _validate_cat_ii_open_participant_state(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    *,
    initiator_id: str,
    responder_ids: list[str],
) -> None:
    """Reject durable participant outcomes before a contest resolves.

    A Cat II opening is canonical attempt-in-progress surface. Its facts can
    establish posture, contact, and immediate position for responders, but it
    cannot durably move, activate, remove, or kill either side before their
    collected intentions are adjudicated. Non-participant side effects and
    responder spawns remain valid.
    """
    participant_ids = {initiator_id, *responder_ids}
    current_locations = {
        character.character_id: character.location
        for character in ckpt.characters
        if character.character_id in participant_ids
    }
    changed_ids = {
        update.character_id
        for update in event.location_updates
        if update.character_id in participant_ids
        and current_locations.get(update.character_id) != update.location_label
    }
    changed_ids.update(
        character_id
        for character_id in (*event.dormant, *event.cull)
        if character_id in participant_ids
    )
    changed_ids.update(
        update.character_id
        for update in event.activate
        if update.character_id in participant_ids
    )
    if changed_ids:
        raise ValueError(
            "Cat II open applied a durable state change to unresolved "
            "participant(s): " + ", ".join(sorted(changed_ids))
        )


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
    *,
    frame: str = "foreground",
    local_context: str = "",
) -> str | None:
    """Fetch one NPC intention and normalize empty/refusal outputs."""
    from app.engine.context_builder import is_unbound_player_authored_slot

    character = _character_by_id(ckpt).get(character_id)
    if is_unbound_player_authored_slot(ckpt, character):
        raise ValueError(
            "Cannot dispatch an unbound player-authored slot as an agent: "
            f"{character_id}"
        )
    raw = await dispatcher.agent_intend(
        ckpt=ckpt,
        character_id=character_id,
        frame=frame,
        local_context=local_context,
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
    targets: list[str],
    log_label: str,
) -> int:
    """Append perception fragments before an event is broadcast.

    Used by normal observation harvest and by private query answers
    that need an NPC's current visual self-presentation. Finalize the
    canonical fact list first so human render buffers, NPC inboxes, and
    compact router history all project the same event.
    """
    if not targets:
        logger.warning(
            "%s requested perception harvest but no harvestable targets "
            "remained after filtering.",
            log_label,
        )
        return 0

    fragments = await dispatcher.harvest_perceptions(
        ckpt=ckpt,
        character_ids=targets,
    )
    by_id = {c.character_id: c for c in ckpt.characters}
    appended = 0
    for cid, fragment in zip(targets, fragments):
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
        log_label, appended, len(targets), result.event_id,
    )
    return appended


def _loadout_fact_count(result: EventRouterOutput) -> int:
    return sum(
        1
        for fact in result.canonical_event.observable_facts
        if fact.text.lstrip().startswith("[loadout")
    )


def _keep_only_harvest_loadout_facts(result: EventRouterOutput) -> None:
    result.canonical_event.observable_facts = [
        fact for fact in result.canonical_event.observable_facts
        if fact.text.lstrip().startswith("[loadout")
    ]


def _refresh_router_history_after_mutation(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    acting_character_id: str,
    mode: str = "intention",
) -> None:
    from app.engine.turn_loop_dispatcher import refresh_router_history_record

    refresh_router_history_record(
        ckpt.session_conversation,
        acting_character_id=acting_character_id,
        result=result,
        mode=mode,
    )


def broadcast_event(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    actor_id: str = "",
    *,
    close_for_presentation: bool = True,
    preflighted: bool = False,
) -> list[str]:
    """Append a closed canonical event to the log and fan it out to
    every human observer's render buffer and every NPC observer's
    `pending_observations` queue.

    Perception is structural: the router declares event observers and
    fact-level visibility packets. Location is not a fallback and does
    not create implicit observation.

    `actor_id` is the character whose intention produced this event
    (the player who /act'd, or the cascade NPC whose intention the
    router just adjudicated). It does not make every fact in the resulting
    event the actor's own action: adjudication can add another character's
    response, an environmental change, or an attack outcome. The actor
    therefore receives the same visible canonical facts as any other NPC
    observer. Their rolling history already contains their submitted action,
    but the canonical event is the only complete account of what actually
    happened around and because of it.

    The event's stable `event_id` is what lands in human render buffers
    (not Python object identity) so checkpoints remain resolvable
    across process restarts.

    The NPC inbox path is the engine implementation of the perception
    channel the router describes in `observable_facts`. When an NPC is
    routed for next output in this same beat, their agent turn drains
    the queue and they see the just-broadcast event as the most recent
    entry.

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
    if not preflighted:
        _preflight_broadcast_event(ckpt, event, actor_id=actor_id)
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
    from app.engine.closed_event_runtime import closed_event_runtime_for

    if close_for_presentation:
        closed_event_runtime = closed_event_runtime_for(ckpt)
        if closed_event_runtime is not None:
            closed_event_runtime.close_event(
                checkpoint=ckpt,
                event=event,
                event_sequence=event_sequence,
                actor_id=actor_id,
            )
    from app.engine.content_fronts import queue_front_signals_from_public_event

    queue_front_signals_from_public_event(ckpt, event, actor_id=actor_id)

    from app.engine.context_builder import collect_player_ids

    player_ids = collect_player_ids(ckpt)
    if actor_id and actor_id not in player_ids:
        actor = by_id.get(actor_id)
        if (
            actor is not None
            and actor.status != "culled"
            and not is_non_social_hazard(actor)
        ):
            end_at_s = max(0, event.effective_at_s + event.duration_s)
            previous = actor.last_agent_turn_at_s
            actor.last_agent_turn_at_s = max(
                previous if previous is not None else 0,
                end_at_s,
            )

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
        recipient = by_id.get(o.character_id)
        if recipient is None or recipient.status == "culled":
            continue
        if is_non_social_hazard(recipient):
            # The event ledger already preserves any fact scoped to this
            # hazard. There is no future character-agent call that could
            # consume an inbox or a first-meeting self-presentation.
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
        intro_plan = plan_event_visual_introductions(
            ckpt,
            viewer_id=o.character_id,
            event=event,
            observation_level=obs_level_by_char.get(o.character_id, "direct"),
            priority_target_ids=[actor_id],
            max_loadouts=AGENT_FIRST_MEETING_CAP,
        )
        intro_block = format_visual_introductions(intro_plan.loadouts)
        if intro_block:
            recipient.pending_observations.append(intro_block)
        if intro_plan.mark_character_ids:
            mark_visual_introductions(
                ckpt,
                o.character_id,
                intro_plan.mark_character_ids,
            )

    return visible_humans


def _preflight_broadcast_event(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    *,
    actor_id: str,
) -> None:
    """Run every fallible structural guard before adapter state commits."""

    _validate_non_social_hazard_routing(ckpt, event)
    _reject_unbound_player_authored_references(
        ckpt,
        event,
        actor_id=actor_id,
    )


async def prepare_event_for_broadcast(
    dispatcher: Dispatcher,
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    *,
    actor_id: str = "",
) -> None:
    """Apply an optional ruleset transaction after generic preflight.

    The hook is deliberately optional so every generic/D&D fake dispatcher and
    production path remains byte-for-byte behaviorally unchanged when no
    adapter transaction exists.
    """

    _preflight_broadcast_event(ckpt, event, actor_id=actor_id)
    preparer = getattr(dispatcher, "prepare_ruleset_event", None)
    if callable(preparer):
        await preparer(
            ckpt=ckpt,
            result=event,
            actor_id=actor_id,
        )


def _reject_unbound_player_authored_references(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    *,
    actor_id: str,
) -> None:
    """Fail loudly if a blank authored seat reaches canonical runtime state."""
    from app.engine.context_builder import is_unbound_player_authored_slot

    absent_ids = {
        character.character_id
        for character in ckpt.characters
        if is_unbound_player_authored_slot(ckpt, character)
    }
    if not absent_ids:
        return

    references: dict[str, set[str]] = {
        "actor": {actor_id} if actor_id else set(),
        "observer": {observer.character_id for observer in event.observers},
        "required responder": set(event.required_responders),
        "dormant": set(event.dormant),
        "cull": set(event.cull),
        "activate": {signal.character_id for signal in event.activate},
        "location update": {
            signal.character_id for signal in event.location_updates
        },
        "commitment open": set(event.commitment_open.actor_ids),
        "commitment resolution": {
            character_id
            for signal in event.commitment_resolutions
            for character_id in signal.actor_ids
        },
        "commitment interrupt": {
            character_id
            for signal in event.commitment_interrupts
            for character_id in signal.actor_ids
        },
        "private fact": {
            character_id
            for fact in event.canonical_event.observable_facts
            for character_id in fact.visible_to
        },
        "fact text": {
            character_id
            for fact in event.canonical_event.observable_facts
            for character_id in absent_ids
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(character_id)}"
                rf"(?![A-Za-z0-9_])",
                fact.text,
            )
        },
        "spawn": {request.character_id for request in event.spawn},
        "combatant": set(getattr(event, "combatant_ids", []) or []),
    }
    violations = [
        f"{label}={character_id}"
        for label, character_ids in references.items()
        for character_id in sorted(character_ids & absent_ids)
    ]
    if violations:
        raise RuntimeError(
            "Router event referenced an unclaimed player-authored seat: "
            + ", ".join(violations)
        )


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
}
_REACTION_BLOCKING_CONDITIONS = {
    "incapacitated",
    "paralyzed",
    "stunned",
    "unconscious",
}


def _active_combat(ckpt: CheckpointFile) -> Any | None:
    combat = checkpoint_active_combat(ckpt)
    return combat if combat is not None and _combatants(combat) else None


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


def _one_star_ruleset_enabled(ckpt: CheckpointFile) -> bool:
    settings = getattr(ckpt.session.config, "settings", None)
    ruleset_id = str(getattr(settings, "ruleset_id", "") or "")
    return ruleset_id == "one_star_ascension"


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
    by_id = {char.character_id: char for char in ckpt.characters}
    selected: list[Any] = []
    seen: set[str] = set()
    explicitly_selected = set(combatant_ids)
    for cid in seed_ids:
        char = by_id.get(cid)
        if char is None or cid in seen:
            continue
        status = _character_status_value(char)
        if status == "culled":
            continue
        if status == "dormant":
            if cid not in explicitly_selected:
                continue
            if _dnd_character_defeated_by_mechanics(char):
                if _dnd_character_is_combat_spawn(char):
                    char.status = CharacterStatus.culled
                continue
            char.status = CharacterStatus.active
        elif status != "active":
            continue
        selected.append(char)
        seen.add(cid)
    return selected


def _dnd_character_defeated_by_mechanics(character: Any) -> bool:
    mechanics = getattr(character, "mechanics", {}) or {}
    if not isinstance(mechanics, dict):
        return False
    hp = _dnd_character_hit_points(mechanics)
    if hp is None:
        return False
    max_hp = _safe_int(hp.get("max"), 0)
    if max_hp <= 0:
        return False
    return _safe_int(hp.get("current"), max_hp) <= 0


def _dnd_character_hit_points(mechanics: dict[str, Any]) -> dict[str, Any] | None:
    hp = mechanics.get("hit_points")
    if isinstance(hp, dict):
        return hp
    sheet_hp = (
        ((mechanics.get("dnd5e_sheet") or {}).get("statblock") or {})
        .get("defenses", {})
        .get("hit_points")
    )
    return sheet_hp if isinstance(sheet_hp, dict) else None


def _dnd_character_is_combat_spawn(character: Any) -> bool:
    mechanics = getattr(character, "mechanics", {}) or {}
    if not isinstance(mechanics, dict):
        return False
    marker = mechanics.get("combat_spawn")
    if isinstance(marker, dict) and bool(marker.get("spawned")):
        return True
    return str(mechanics.get("source") or "") == "router_combatant_spawn"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _character_location(ckpt: CheckpointFile, character_id: str) -> str:
    for character in ckpt.characters:
        if character.character_id == character_id:
            return str(getattr(character, "location", "") or "")
    return ""


def _router_target_needs_local_context(
    prior_result: EventRouterOutput,
    target: RouterOutputTarget,
) -> bool:
    if target.frame != "background":
        return False
    if prior_result.event_kind == "public_fact":
        return True
    return any(
        update.character_id == target.character_id
        for update in prior_result.location_updates
    )


def _router_target_local_context(
    ckpt: CheckpointFile,
    character_id: str,
) -> str:
    actor = next(
        (
            character for character in ckpt.characters
            if character.character_id == character_id
        ),
        None,
    )
    if actor is None:
        return ""
    location = str(getattr(actor, "location", "") or "").strip()
    location_label = location or "Off-screen / unspecified location."
    nearby: list[str] = []
    if location:
        from app.engine.context_builder import is_unbound_player_authored_slot

        for character in ckpt.characters:
            if character.character_id == character_id:
                continue
            if _character_status_value(character) != "active":
                continue
            if is_unbound_player_authored_slot(ckpt, character):
                continue
            if str(getattr(character, "location", "") or "").strip() != location:
                continue
            nearby.append(f"{character.name} ({character.character_id})")
    nearby_text = ", ".join(nearby) if nearby else "none known"
    return "\n".join([
        f"Location: {location_label}",
        f"Nearby active characters: {nearby_text}",
    ])


def _materialize_dnd_combatant_spawns(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
    *,
    actor_id: str,
) -> tuple[list[str], list[str]]:
    prior_combatant_ids = list(getattr(result, "combatant_ids", []) or [])
    spawns = list(getattr(result, "combatant_spawns", []) or [])
    if not spawns:
        return [], prior_combatant_ids
    by_id = {character.character_id: character for character in ckpt.characters}
    combatant_ids = list(prior_combatant_ids)
    default_location = _character_location(ckpt, actor_id)
    spawned_ids: list[str] = []
    materialized_spawns: list[Any] = []
    for spawn in spawns:
        spawn_id = str(getattr(spawn, "character_id", "") or "")
        if not spawn_id:
            continue
        if spawn_id in by_id:
            combatant_ids = [cid for cid in combatant_ids if cid != spawn_id]
            base = (
                str(getattr(spawn, "monster_key", "") or "")
                or str(getattr(spawn, "name", "") or "")
                or spawn_id
                or "combatant"
            )
            spawn_id = _unique_spawn_character_id(by_id, base)
            spawn = spawn.model_copy(update={"character_id": spawn_id})
        character = imported_statblocks.resolve_spawn_character_from_content_state(
            spawn,
            content_state=getattr(ckpt.session, "content_state", {}),
            default_location=default_location,
        )
        if character is None:
            character = dnd_monsters.character_from_combatant_spawn(
                spawn,
                default_location=default_location,
            )
        _mark_dnd_combat_spawn_character(
            character,
            spawn=spawn,
            source_event_id=str(getattr(result, "event_id", "") or ""),
        )
        ckpt.characters.append(character)
        by_id[character.character_id] = character
        combatant_ids.append(character.character_id)
        spawned_ids.append(character.character_id)
        materialized_spawns.append(spawn)
    if spawned_ids:
        result.combatant_ids = [
            cid for cid in dict.fromkeys(combatant_ids) if cid
        ]
        result.combatant_spawns = materialized_spawns
        logger.info(
            "Materialized D&D combatant spawn(s): %s",
            ", ".join(spawned_ids),
        )
    return spawned_ids, prior_combatant_ids


def _mark_dnd_combat_spawn_character(
    character: Any,
    *,
    spawn: Any,
    source_event_id: str,
) -> None:
    mechanics = dict(getattr(character, "mechanics", {}) or {})
    marker = {
        "spawned": True,
        "source_event_id": source_event_id,
        "monster_key": str(getattr(spawn, "monster_key", "") or ""),
        "statblock_ref": str(getattr(spawn, "statblock_ref", "") or ""),
    }
    mechanics["combat_spawn"] = {
        key: value for key, value in marker.items() if value or key == "spawned"
    }
    if not mechanics.get("source"):
        mechanics["source"] = "router_combatant_spawn"
    character.mechanics = mechanics


def _unique_spawn_character_id(
    existing: dict[str, Any],
    base: str,
) -> str:
    candidate = _clean_character_id(base) or "combatant"
    if candidate not in existing:
        return candidate
    index = 2
    while f"{candidate}_{index}" in existing:
        index += 1
    return f"{candidate}_{index}"


def _clean_character_id(value: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


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
            routing_role="observe_only",
        ))
        existing.add(cid)


def _set_pending_initiating_action(
    combat: Any,
    *,
    actor_id: str,
    event_id: str,
    intention: str,
) -> None:
    for combatant in _combatants(combat):
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
    for combatant in _combatants(combat):
        cid = _combatant_character_id(combatant)
        if not cid or cid in seen:
            continue
        observers.append(ObserverEntry(
            character_id=cid,
            observation_level="d",
            routing_role="observe_only",
        ))
        seen.add(cid)
    if not observers:
        return 0
    event = EventRouterOutput(
        event_id="",
        effective_at_s=0,
        duration_s=0,
        decision_rationale="code-owned combat state change",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(feasible=True),
            observable_facts=[ObservableFact.all(fact) for fact in facts],
        ),
        event_kind="state_change",
        requires_responders=False,
        required_responders=[],
        observers=observers,
        spawn=[],
        dormant=[],
        cull=[],
        commitment_open=empty_commitment_open_signal(),
        commitment_resolutions=[],
        commitment_interrupts=[],
        location_updates=[],
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
    imported_encounter = imported_encounters.resolve_combat_start_from_content_state(
        getattr(ckpt.session, "content_state", {}),
        location_ref=_character_location(ckpt, actor_id),
    )
    imported_encounters.apply_resolved_encounter_to_router_output(
        result,
        imported_encounter,
    )
    spawned_ids, prior_combatant_ids = _materialize_dnd_combatant_spawns(
        ckpt,
        result,
        actor_id=actor_id,
    )
    if spawned_ids and getattr(result, "spawn", None):
        spawned_set = set(spawned_ids)
        result.spawn = [
            spawn for spawn in result.spawn
            if spawn.character_id not in spawned_set
        ]
    combatant_ids = list(getattr(result, "combatant_ids", []) or [])
    participants = _dnd_combat_start_participants(
        ckpt, actor_id, combatant_ids,
    )
    if len(participants) < 2:
        if spawned_ids:
            spawned_set = set(spawned_ids)
            ckpt.characters = [
                character for character in ckpt.characters
                if character.character_id not in spawned_set
            ]
            result.combatant_spawns = []
            result.combatant_ids = [
                cid for cid in prior_combatant_ids if cid not in spawned_set
            ]
        result.requires_responders = False
        result.required_responders = []
        result.clear_routing_roles()
        result.event_kind = "state_change"
        return False

    combat_id = f"combat_{result.event_id or uuid.uuid4().hex[:8]}"
    combat = dnd_combat.start_combat(
        ckpt.session,
        participants,
        combat_id=combat_id,
    )
    battle_map_seed = getattr(result, "battle_map_seed", None)
    battle_map = dnd_spatial.normalize_battle_map_seed(
        battle_map_seed,
        combat.combatants,
    )
    if battle_map is not None:
        combat.battle_map = battle_map
    order = ", ".join(
        f"{c.name or c.character_id} {c.initiative_total}"
        for c in combat.combatants
    )
    result.requires_responders = False
    result.required_responders = []
    result.clear_routing_roles()
    result.event_kind = "state_change"
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
    if combat.battle_map is not None:
        dnd_combat.append_audit_line(
            combat,
            "Battle map seeded: "
            f"{combat.battle_map.map_name} "
            f"{combat.battle_map.width}x{combat.battle_map.height}.",
        )
    if imported_encounter is not None:
        dnd_combat.append_audit_line(
            combat,
            "Imported encounter template applied: "
            f"{imported_encounter.encounter_ref}.",
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
    result.clear_routing_roles()
    result.event_kind = "state_change"
    if actor_id:
        existing_observers = {
            observer.character_id for observer in result.observers
        }
        if actor_id not in existing_observers:
            result.observers.append(ObserverEntry(
                character_id=actor_id,
                observation_level="d",
                routing_role="observe_only",
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
        result.clear_routing_roles()
        result.event_kind = "state_change"
        return False
    dnd_combat.append_audit_line(
        combat,
        f"Combat ended from router D&D interaction signal: {result.event_id}.",
    )
    for fact in dnd_combat.drain_pending_visible_facts(combat):
        result.canonical_event.observable_facts.append(ObservableFact.all(fact))
    dnd_combat.queue_router_observed_fact_updates(ckpt.session, combat)
    dnd_combat.end_combat(ckpt.session, characters=ckpt.characters)
    result.requires_responders = False
    result.required_responders = []
    result.clear_routing_roles()
    result.event_kind = "state_change"
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
    combatant = _current_combatant(combat, skip_defeated=True)
    return _combatant_character_id(combatant) if combatant is not None else ""


def _eligible_combat_reaction_prompts(
    ckpt: CheckpointFile,
    *,
    events_closed: int,
    event_actor_ids: list[str],
    suppress_reaction_prompts: bool,
) -> dict[str, str]:
    """Return character_id -> triggering event id for combat reaction UI.

    D&D extends observer routing with an explicit reaction signal. That signal
    is a UI affordance, not mechanical proof that a reaction is legal.
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
        if event.event_kind in _COMBAT_REACTION_EXCLUDED_REASONS:
            continue
        for observer in event.observers:
            cid = observer.character_id
            if cid in prompts or cid == actor_id or cid not in player_ids:
                continue
            if observer.observation_level != "d":
                continue
            if observer.routing_role != "dnd_reaction":
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
    closed_reasons = [_event_handoff_reason(event) for event in closed_events]
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
# Router, narrator, and character-agent calls are abstracted behind the
# `Dispatcher` protocol. The production orchestrator binds that protocol to
# the concrete LLM-backed modules; tests bind fakes.
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
        event (with Cat I/II decision, observer routing roles, and
        event_kind populated).

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
        original_action: str = "",
    ) -> EventRouterOutput:
        """Author another grounded event after a narrator continuation
        handoff keeps the current visible batch open."""
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
        frame: str = "foreground",
        local_context: str = "",
    ) -> str:
        """Ask an agent for their next intention (as free-form text).
        Returns the intention string; the orchestrator re-routes it
        through route_intention as a fresh intention."""
        ...

    async def harvest_perceptions(
        self,
        ckpt: CheckpointFile,
        character_ids: list[str],
    ) -> list[str]:
        """Fire CharacterAgent.perceive() across `character_ids` in
        parallel; return per-character "visual loadout" fragments in
        the same order as the input ids.

        Used by the observation-harvest fork in `run_beat` when the
        router classifies an action as `event_kind="observation_harvest"`.
        Empty / failed-perception entries are
        included as empty strings so the caller can zip with the
        input ids; callers SHOULD filter those out before composing
        the canonical event's `observable_facts`.

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
        handoff_policy: str = "forced",
        handoff_context: str = "",
    ) -> tuple[NarratorFinalOutput, TranscriptEntry]:
        """Render this human's POV prose for the beat. Input: their
        buffered events since last render, observation levels tagged.
        Returns `(NarratorFinalOutput, TranscriptEntry)`.

        The transient entry is constructed engine-side from `user_input`
        (passed by the acting-POV caller; "" for incidental POVs) and the
        rendered `final_text`. Durable `/history` reads the per-character
        narrator conversations rather than a second checkpoint transcript.

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

    `transcript_entries` is the transient parallel per-POV response record
    from each narrator render. It is empty when no human rendered this beat
    (Cat II pending, or a beat with no renderable events).

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
    ended_reason: str  # Semantic event kind, safety cap, or pending Cat II boundary.
    transcript_entries: dict[str, TranscriptEntry]
    event_actor_ids: list[str]
    reaction_prompts: dict[str, str] | None = None
    loot_prompts: dict[str, list[str]] | None = None
    rendered_event_ids_by_pov: dict[str, list[str]] = field(
        default_factory=dict
    )
    continue_requested: bool = False


@dataclass(slots=True)
class _SpeculativeNextOutput:
    """One isolated autonomous handoff prepared beside narrator pacing.

    Agent memory and compact router history stay on ``checkpoint`` until the
    narrator explicitly defers presentation. A discarded branch therefore
    cannot teach a character that they said something which never became
    canonical fiction.
    """

    outcome: str
    checkpoint: CheckpointFile | None = None
    result: EventRouterOutput | None = None
    actor_id: str = ""
    submission: str = ""
    touched_actor_ids: list[str] = field(default_factory=list)


def _durable_checkpoint_copy(ckpt: CheckpointFile) -> CheckpointFile:
    """Clone model state without copying live tasks or runtime handles."""

    return CheckpointFile.model_validate_json(ckpt.model_dump_json(
        context={"include_private_runtime_metadata": True},
    ))


def _adopt_speculative_router_state(
    ckpt: CheckpointFile,
    speculative: CheckpointFile,
    *,
    actor_ids: list[str],
) -> None:
    """Commit only state an agent submission and router call may mutate."""

    speculative_characters = {
        character.character_id: character
        for character in speculative.characters
    }
    for actor_id in dict.fromkeys(actor_ids):
        speculative_actor = speculative_characters.get(actor_id)
        if speculative_actor is not None:
            for character in ckpt.characters:
                if character.character_id == actor_id:
                    for field_name in CharacterRecord.model_fields:
                        setattr(
                            character,
                            field_name,
                            deepcopy(getattr(speculative_actor, field_name)),
                        )
                    break

        if actor_id in speculative.character_conversations:
            ckpt.character_conversations[actor_id] = deepcopy(
                speculative.character_conversations[actor_id]
            )
        else:
            ckpt.character_conversations.pop(actor_id, None)

    ckpt.session_conversation = deepcopy(speculative.session_conversation)
    ckpt.session.pending_engine_state_updates = list(
        speculative.session.pending_engine_state_updates
    )
    ckpt.session.content_state = deepcopy(speculative.session.content_state)
    ckpt.session.content_manager_preflight_cycle = (
        speculative.session.content_manager_preflight_cycle
    )
    ckpt.session.content_manager_last_run_cycle = (
        speculative.session.content_manager_last_run_cycle
    )


async def run_beat(
    ckpt: CheckpointFile,
    dispatcher: Dispatcher,
    actor_id: str,
    intention: str,
    cat_ii_event_id: str | None = None,
    combat_reaction_event_id: str | None = None,
    resume_after_handoff: EventRouterOutput | None = None,
    resume_events_closed: int = 0,
    resume_event_actor_ids: list[str] | None = None,
) -> BeatResult:
    """Run one beat to completion.

    Entry paths:
    1. Fresh actor submission — `cat_ii_event_id=None`; claim the initiator
       slot, route the proposed motion, and cascade.
    2. Cat II responder intention — `cat_ii_event_id` set. Collect the
       intention into the open event. If all required responders are in,
       adjudicate the Cat II (calling dispatcher.route_intention with
       the full open_event); else return immediately with ended_reason
       "cat_ii_pending" (nothing to render yet).
    3. Combat reaction intention — `combat_reaction_event_id` set. The
       actor's live reaction slot is cleared, their reaction is marked spent,
       and the intention is routed as an out-of-turn response.

    Handoff:
    - Every closed narrative event becomes a narrator handoff candidate.
      Autonomous `next_output` work is prepared in parallel on isolated state;
      narrator render discards the prepared state but leaves the canonical
      semantic handoff resumable by `(defer)`, while narrator continue commits
      it immediately.
    - Bound `next_output` targets and forced safety/rules boundaries render
      without speculative autonomous work.
    - Targetless events may request a router continuation when the narrator
      keeps established motion or a submitted wait condition unresolved.
    - Rules, query, Cat II opening and bound-participant resolution, harvest,
      and safety-cap event kinds force render fan-out and slot release.
    - Cat II event adjudicates; selected semantic `next_output` targets yield
      for bound characters or dispatch eligible autonomous characters. If no
      selected character can act, render fan-out and slot release.
    - max_events_per_beat reached → forced render + slot release.
    - max_agent_cascades_per_beat reached → forced render + slot release.
    """
    settings = ckpt.session.config.settings
    max_events = max(1, int(settings.max_events_per_beat))
    max_agent_cascades = max(
        1,
        int(getattr(settings, "max_agent_cascades_per_beat", 35)),
    )
    events_closed = max(0, int(resume_events_closed))
    agent_cascade_attempts = 0
    background_thread_attempts = 0
    event_actor_ids = list(resume_event_actor_ids or [])
    current_intention = intention
    current_actor = actor_id
    pending_result: EventRouterOutput | None = None
    pending_result_is_continuation = False
    pending_result_actor_id: str = ""
    pending_result_submission: str = ""
    suppress_reaction_prompts = combat_reaction_event_id is not None

    async def _commit_event(
        event: EventRouterOutput,
        *,
        event_actor_id: str = "",
        ruleset_actor_id: str | None = None,
        close_for_presentation: bool = True,
    ) -> list[str]:
        await prepare_event_for_broadcast(
            dispatcher,
            ckpt,
            event,
            actor_id=(
                event_actor_id
                if ruleset_actor_id is None
                else ruleset_actor_id
            ),
        )
        return broadcast_event(
            ckpt,
            event,
            actor_id=event_actor_id,
            close_for_presentation=close_for_presentation,
            preflighted=True,
        )

    async def _queue_router_continuation(
        prior_result: EventRouterOutput,
    ) -> None:
        nonlocal pending_result, pending_result_is_continuation
        nonlocal pending_result_actor_id
        nonlocal pending_result_submission
        pending_result = await dispatcher.route_continuation(
            ckpt=ckpt,
            actor_id=actor_id,
            prior_result=prior_result,
            original_action=intention,
        )
        pending_result_is_continuation = True
        pending_result_actor_id = actor_id
        pending_result_submission = ""
        _log_router_rationale(
            pending_result, actor_id, kind="continuation",
        )

    async def _prepare_speculative_next_output(
        prior_result: EventRouterOutput,
        ordered_targets: list[tuple[str, str]],
    ) -> _SpeculativeNextOutput:
        """Prepare one autonomous response without touching live state."""

        nonlocal agent_cascade_attempts, background_thread_attempts
        speculative_ckpt = _durable_checkpoint_copy(ckpt)
        from app.engine.context_builder import collect_player_ids

        touched_actor_ids: list[str] = []
        for control_kind, character_id in ordered_targets:
            if control_kind == "bound":
                return _SpeculativeNextOutput(
                    outcome="human",
                    checkpoint=speculative_ckpt,
                    touched_actor_ids=touched_actor_ids,
                )
            if agent_cascade_attempts >= max_agent_cascades:
                return _SpeculativeNextOutput(
                    outcome="cap",
                    checkpoint=speculative_ckpt,
                    touched_actor_ids=touched_actor_ids,
                )

            agent_targets = targets_from_router_output(
                prior_result,
                player_ids=collect_player_ids(speculative_ckpt),
                agent_ids=[character_id],
            )
            if not agent_targets:
                continue

            target = agent_targets[0]
            if target.frame == "background":
                if (
                    background_thread_attempts
                    >= MAX_BACKGROUND_THREADS_PER_BEAT
                ):
                    return _SpeculativeNextOutput(
                        outcome="cap",
                        checkpoint=speculative_ckpt,
                        touched_actor_ids=touched_actor_ids,
                    )
                background_thread_attempts += 1
            agent_cascade_attempts += 1
            touched_actor_ids.append(target.character_id)

            local_context = (
                _router_target_local_context(
                    speculative_ckpt, target.character_id,
                )
                if _router_target_needs_local_context(prior_result, target)
                else ""
            )
            agent_output = await _agent_intention_for_dispatch(
                dispatcher,
                speculative_ckpt,
                target.character_id,
                frame=target.frame,
                local_context=local_context,
            )
            if agent_output is None:
                continue

            routed = await dispatcher.route_intention(
                ckpt=speculative_ckpt,
                actor_id=target.character_id,
                intention=agent_output,
                cat_ii_event=None,
            )
            _log_router_rationale(
                routed,
                target.character_id,
                kind="speculative_route",
            )
            return _SpeculativeNextOutput(
                outcome="queued",
                checkpoint=speculative_ckpt,
                result=routed,
                actor_id=target.character_id,
                submission=agent_output,
                touched_actor_ids=touched_actor_ids,
            )

        return _SpeculativeNextOutput(
            outcome="exhausted",
            checkpoint=speculative_ckpt,
            touched_actor_ids=touched_actor_ids,
        )

    def _adopt_speculative_next_output(
        prepared: _SpeculativeNextOutput,
    ) -> None:
        nonlocal pending_result, pending_result_is_continuation
        nonlocal pending_result_actor_id, pending_result_submission
        if prepared.checkpoint is None:
            raise RuntimeError(
                "Speculative next_output has no prepared checkpoint."
            )
        _adopt_speculative_router_state(
            ckpt,
            prepared.checkpoint,
            actor_ids=prepared.touched_actor_ids,
        )
        if prepared.outcome != "queued":
            return
        if prepared.result is None:
            raise RuntimeError(
                "Queued speculative next_output has no routed result."
            )
        pending_result = prepared.result
        pending_result_is_continuation = False
        pending_result_actor_id = prepared.actor_id
        pending_result_submission = prepared.submission

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

    async def _end_for_cascade_cap() -> BeatResult:
        logger.warning(
            "Beat cascade cap reached: configured_cap=%d events_rendered=%d",
            max_agent_cascades,
            events_closed,
        )
        return await _end_beat(
            ckpt, dispatcher,
            ended_reason="cascade_cap",
            events_closed=events_closed,
            event_actor_ids=event_actor_ids,
            acting_player_id=actor_id,
            acting_player_input=intention,
            suppress_reaction_prompts=suppress_reaction_prompts,
        )

    async def _advance_or_render(
        result: EventRouterOutput,
        *,
        default_ended_reason: str = "",
        resolved_cat_ii: OpenCatIIEvent | None = None,
    ) -> BeatResult | None:
        """Race narrator pacing against isolated autonomous next-output work."""
        if (
            events_closed >= max_events
            and default_ended_reason != "cat_ii_resolution"
        ):
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason="max_events_cap",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=suppress_reaction_prompts,
            )
        if _event_requires_forced_handoff(
            result,
            ckpt=ckpt,
            resolved_cat_ii=resolved_cat_ii,
        ):
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason=_event_handoff_reason(result),
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=suppress_reaction_prompts,
            )

        targets = _binding_aware_next_output_targets(ckpt, result)
        if targets and targets[0][0] == "bound":
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason="awaiting_player_turn",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=suppress_reaction_prompts,
            )

        from app.engine.context_builder import collect_player_ids

        player_ids = collect_player_ids(ckpt)
        has_narrator_target = any(
            character_id in player_ids and bool(buffer)
            for character_id, buffer in ckpt.session.render_buffers.items()
        )
        if targets and not has_narrator_target:
            prepared = await _prepare_speculative_next_output(result, targets)
            if prepared.checkpoint is not None:
                _adopt_speculative_next_output(prepared)
            if prepared.outcome == "queued":
                return None
            if prepared.outcome == "cap":
                return await _end_for_cascade_cap()
            if prepared.outcome == "human":
                return await _end_beat(
                    ckpt,
                    dispatcher,
                    ended_reason="awaiting_player_turn",
                    events_closed=events_closed,
                    event_actor_ids=event_actor_ids,
                    acting_player_id=actor_id,
                    acting_player_input=intention,
                    suppress_reaction_prompts=suppress_reaction_prompts,
                )
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason=(
                    default_ended_reason
                    or _event_handoff_reason(result)
                    or "cascade_exhausted"
                ),
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=suppress_reaction_prompts,
            )

        speculative_task: asyncio.Task[_SpeculativeNextOutput] | None = None
        if targets:
            if agent_cascade_attempts >= max_agent_cascades:
                return await _end_for_cascade_cap()
            speculative_task = asyncio.create_task(
                _prepare_speculative_next_output(result, targets)
            )

        try:
            handoff = await _end_beat(
                ckpt, dispatcher,
                ended_reason=(
                    default_ended_reason
                    or _event_handoff_reason(result)
                    or "cascade_exhausted"
                ),
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=suppress_reaction_prompts,
                soft_handoff_candidate=True,
            )
        except BaseException:
            if speculative_task is not None:
                if not speculative_task.done():
                    speculative_task.cancel()
                await asyncio.gather(speculative_task, return_exceptions=True)
            raise
        if not handoff.continue_requested:
            if speculative_task is not None:
                if not speculative_task.done():
                    speculative_task.cancel()
                await asyncio.gather(speculative_task, return_exceptions=True)
            return handoff

        # A narrator defer discards its candidate prose and preserves the
        # current beat. Restore any staged spawn records that the candidate
        # render temporarily rolled back before adopting speculative work.
        _restore_speculative_spawn_roster(ckpt)
        if speculative_task is not None:
            try:
                prepared = await speculative_task
            except BaseException:
                _rollback_speculative_spawn_roster(ckpt)
                raise
            if prepared.checkpoint is not None:
                _adopt_speculative_next_output(prepared)
            if prepared.outcome == "queued":
                return None
            if prepared.outcome == "cap":
                return await _end_for_cascade_cap()
            if prepared.outcome == "human":
                return await _end_beat(
                    ckpt,
                    dispatcher,
                    ended_reason="awaiting_player_turn",
                    events_closed=events_closed,
                    event_actor_ids=event_actor_ids,
                    acting_player_id=actor_id,
                    acting_player_input=intention,
                    suppress_reaction_prompts=suppress_reaction_prompts,
                )

        if handoff.continue_requested:
            try:
                await _queue_router_continuation(result)
            except Exception:
                _rollback_speculative_spawn_roster(ckpt)
                raise
            return None
        raise AssertionError("unreachable narrator handoff state")

    # --- Step 1: handle entry path ------------------------------------------

    if resume_after_handoff is None and cat_ii_event_id is not None:
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
            resolved,
            evt.initiator_id,
            kind="cat_ii_resolve",
            ckpt=ckpt,
            resolved_cat_ii=evt,
        )
        _validate_non_social_hazard_routing(ckpt, resolved)
        one_star_resolution = _one_star_ruleset_enabled(ckpt)
        if not one_star_resolution:
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
        await _commit_event(
            resolved,
            ruleset_actor_id=evt.initiator_id,
        )
        if one_star_resolution:
            close_cat_ii(ckpt, evt.event_id)
        events_closed = 1
        event_actor_ids.append(evt.initiator_id)

        completed = await _advance_or_render(
            resolved,
            default_ended_reason="cat_ii_resolution",
            resolved_cat_ii=evt,
        )
        if completed is not None:
            return completed

    if resume_after_handoff is None and combat_reaction_event_id is not None:
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
            await _commit_event(resolved, event_actor_id=actor_id)
            event_actor_ids.append(actor_id)
            events_closed += 1
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason=_event_handoff_reason(resolved)
                or "ruleset_resolution",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
                suppress_reaction_prompts=True,
            )

    # Fresh initiator path.
    if (
        resume_after_handoff is None
        and cat_ii_event_id is None
        and combat_reaction_event_id is None
    ):
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
            await _commit_event(resolved, event_actor_id=actor_id)
            event_actor_ids.append(actor_id)
            events_closed += 1
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason=_event_handoff_reason(resolved)
                or "ruleset_resolution",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
            )

    if resume_after_handoff is not None:
        resume_targets = _binding_aware_next_output_targets(
            ckpt, resume_after_handoff,
        )
        if resume_targets:
            prepared = await _prepare_speculative_next_output(
                resume_after_handoff,
                resume_targets,
            )
            if prepared.checkpoint is not None:
                _adopt_speculative_next_output(prepared)
            if prepared.outcome == "cap":
                return await _end_for_cascade_cap()
            if prepared.outcome == "human":
                return await _end_beat(
                    ckpt,
                    dispatcher,
                    ended_reason="awaiting_player_turn",
                    events_closed=events_closed,
                    event_actor_ids=event_actor_ids,
                    acting_player_id=actor_id,
                    acting_player_input=intention,
                    suppress_reaction_prompts=suppress_reaction_prompts,
                )
            if prepared.outcome != "queued":
                await _queue_router_continuation(resume_after_handoff)
        else:
            await _queue_router_continuation(resume_after_handoff)

    while True:
        if pending_result is None:
            # Route the current actor submission.
            result = await dispatcher.route_intention(
                ckpt=ckpt,
                actor_id=current_actor,
                intention=current_intention,
                cat_ii_event=None,
            )
            result_actor_id = current_actor
            result_submission = current_intention
            result_is_continuation = False
            _log_router_rationale(
                result, current_actor, kind="route",
            )
        else:
            result = pending_result
            pending_result = None
            result_actor_id = pending_result_actor_id
            pending_result_actor_id = ""
            result_submission = pending_result_submission
            pending_result_submission = ""
            result_is_continuation = pending_result_is_continuation
            pending_result_is_continuation = False
            if not result_is_continuation:
                current_actor = result_actor_id
                current_intention = result_submission

        if (
            result.event_kind == "beat_continues"
            and not result.next_output_character_ids
        ):
            raise RuntimeError(
                "Router contract violation: event_kind=beat_continues "
                "requires at least one next_output character."
            )

        # Validate the semantic frontier before materializing spawns, opening
        # Cat II, pinning slots, or otherwise mutating beat state.
        _validate_non_social_hazard_routing(ckpt, result)

        interaction_mode = _dnd_interaction_mode(result)
        if interaction_mode == "dnd_combat_start":
            if result_is_continuation:
                raise RuntimeError(
                    "Router continuation tried to start D&D combat; only an "
                    "actor submission can start initiative."
                )
            signal_actor_id = result_actor_id or current_actor or actor_id
            if _active_combat(ckpt) is not None:
                _block_dnd_combat_start_from_router_signal(
                    ckpt, result, actor_id=signal_actor_id,
                )
                await _commit_event(result, event_actor_id=signal_actor_id)
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
                intention=result_submission,
            )
            await _commit_event(result, event_actor_id=signal_actor_id)
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
            await _commit_event(result, event_actor_id=signal_actor_id)
            event_actor_ids.append(signal_actor_id)
            events_closed += 1
            return await _end_beat(
                ckpt,
                dispatcher,
                ended_reason=_event_handoff_reason(result) or "state_change",
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
                result.clear_routing_roles()
                result.event_kind = "ruleset_cat_ii_suppressed"
                await _commit_event(
                    result,
                    event_actor_id=suppressed_actor_id,
                )
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
            await _materialize_required_responder_spawns(
                dispatcher,
                ckpt,
                result,
                actor_id=result_actor_id,
                required=required,
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
                await _materialize_next_output_spawns(
                    dispatcher,
                    ckpt,
                    result,
                    actor_id=result_actor_id,
                )
                await _commit_event(result, event_actor_id=result_actor_id)
                event_actor_ids.append(result_actor_id)
                events_closed += 1
                completed = await _advance_or_render(result)
                if completed is not None:
                    return completed
                continue

            _validate_cat_ii_open_participant_state(
                ckpt,
                result,
                initiator_id=result_actor_id,
                responder_ids=required,
            )

            evt = open_cat_ii(
                ckpt=ckpt,
                initiator_id=result_actor_id,
                initiator_intention=result_submission,
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
            result.event_kind = "cat_ii_open"
            await _commit_event(result, event_actor_id=result_actor_id)
            event_actor_ids.append(result_actor_id)
            events_closed += 1

            # Autonomous required responders intend immediately; their
            # intentions are collected into the same Cat II event.
            from app.engine.context_builder import is_unbound_player_authored_slot

            bindings = ckpt.session.character_bindings or {}
            by_id = _character_by_id(ckpt)
            for rid in required:
                if rid in bindings:
                    # Human — pin slot, wait.
                    pin_cat_ii_responder(ckpt, rid, evt.event_id)
                else:
                    if is_unbound_player_authored_slot(ckpt, by_id.get(rid)):
                        raise ValueError(
                            "Router required an absent player-authored slot "
                            f"as a responder: {rid}"
                        )
                    # Agent — intend inline.
                    ai_intent = await dispatcher.agent_intend(
                        ckpt=ckpt,
                        character_id=rid,
                    )
                    collect_cat_ii_intention(
                        ckpt, evt.event_id, rid, ai_intent
                    )

            if cat_ii_is_ready(evt):
                # No bound responders in the required list — resolve immediately.
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
                    resolved,
                    evt.initiator_id,
                    kind="cat_ii_resolve_inline",
                    ckpt=ckpt,
                    resolved_cat_ii=evt,
                )
                _validate_non_social_hazard_routing(ckpt, resolved)
                one_star_resolution = _one_star_ruleset_enabled(ckpt)
                if not one_star_resolution:
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
                await _commit_event(
                    resolved,
                    ruleset_actor_id=evt.initiator_id,
                )
                if one_star_resolution:
                    close_cat_ii(ckpt, evt.event_id)
                event_actor_ids.append(evt.initiator_id)
                events_closed += 1
                completed = await _advance_or_render(
                    resolved,
                    default_ended_reason="cat_ii_resolution",
                    resolved_cat_ii=evt,
                )
                if completed is not None:
                    return completed
                continue
            # Bound responders are pinned — pause the beat here. Their /acts will
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

        if result.next_output_character_ids:
            await _materialize_next_output_spawns(
                dispatcher,
                ckpt,
                result,
                actor_id=result_actor_id,
            )

        # v11-r8a: observation_harvest fork. The router signals
        # `event_kind="observation_harvest"` when the actor is
        # purely observing perceptually available NPCs (looking,
        # studying, scanning
        # without dialogue or contact). We bypass the cascade and
        # instead fire each pick's `perceive()` in parallel to harvest
        # one self-presentation fragment per target. Finalize those facts
        # before broadcast so NPC inboxes receive the same enriched event
        # that human render buffers and compact router history receive. The
        # harvest path uses the same human-only pick guard as the cascade path.
        if result.event_kind == "observation_harvest":
            harvest_picks = _filter_routed_agents_for_dispatch(
                ckpt, result.perception_enrichment_character_ids,
                event=result,
            )
            appended = await _append_harvest_fragments(
                dispatcher, ckpt, result,
                targets=harvest_picks,
                log_label="Observation harvest",
            )
            if appended:
                _refresh_router_history_after_mutation(
                    ckpt, result,
                    acting_character_id=result_actor_id or current_actor,
                    mode="continuation" if result_is_continuation else "intention",
                )
        elif result.event_kind == "query_response":
            query_picks = _filter_routed_agents_for_dispatch(
                ckpt, result.perception_enrichment_character_ids,
                event=result,
            )
            if query_picks:
                before_loadouts = _loadout_fact_count(result)
                appended = await _append_harvest_fragments(
                    dispatcher, ckpt, result,
                    targets=query_picks,
                    log_label="Query harvest",
                )
                if appended and _loadout_fact_count(result) > before_loadouts:
                    _keep_only_harvest_loadout_facts(result)
                    _refresh_router_history_after_mutation(
                        ckpt, result,
                        acting_character_id=result_actor_id or current_actor,
                        mode=(
                            "continuation" if result_is_continuation else "intention"
                        ),
                    )

        await _commit_event(
            result,
            event_actor_id=result_actor_id,
        )
        event_actor_ids.append(result_actor_id)
        events_closed += 1

        completed = await _advance_or_render(result)
        if completed is not None:
            return completed


async def _stage_closed_event_spawns_for_render(
    ckpt: CheckpointFile,
    *,
    events_closed: int,
    event_actor_ids: list[str],
) -> tuple[CharacterRecord, ...]:
    """Await and expose every spawn in the candidate render batch."""

    if events_closed <= 0:
        return ()
    closed_events = ckpt.canonical_events[-events_closed:]
    spawned_events = [event for event in closed_events if event.spawn]
    if not spawned_events:
        return ()

    from app.engine.closed_event_runtime import closed_event_runtime_for
    from app.engine.turn_loop_dispatcher import refresh_router_history_record

    runtime = closed_event_runtime_for(ckpt)
    if runtime is None:
        raise RuntimeError(
            "narrator spawn materialization requires the shared "
            "closed-event runtime"
        )

    pending = ckpt.session.pending_narrator_render
    durable_records = list(
        pending.pending_spawn_records if pending is not None else []
    )
    durable_introductions = dict(
        pending.pending_spawn_introductions if pending is not None else {}
    )
    if durable_introductions and not durable_records:
        raise RuntimeError(
            "pending narrator spawn introductions require durable records"
        )
    durable_by_id = {
        record.character_id: record for record in durable_records
    }
    if len(durable_by_id) != len(durable_records):
        raise RuntimeError(
            "pending narrator spawn payload contains duplicate character ids"
        )
    requested_ids = {
        request.character_id
        for event in spawned_events
        for request in event.spawn
        if request.character_id
    }
    unexpected_ids = set(durable_by_id) - requested_ids
    if unexpected_ids:
        raise RuntimeError(
            "pending narrator spawn payload contains records outside the "
            "render batch: " + ", ".join(sorted(unexpected_ids))
        )

    staged_by_id: dict[str, CharacterRecord] = {}
    for index, event in enumerate(closed_events):
        if not event.spawn:
            continue
        actor_id = (
            event_actor_ids[index]
            if index < len(event_actor_ids)
            else ""
        )
        event_spawn_ids = [
            request.character_id
            for request in event.spawn
            if request.character_id
        ]
        event_durable = tuple(
            durable_by_id[character_id]
            for character_id in event_spawn_ids
            if character_id in durable_by_id
        )
        if event_durable and len(event_durable) != len(event_spawn_ids):
            raise RuntimeError(
                "pending narrator spawn payload is incomplete for event "
                f"{event.event_id}"
            )
        records = event_durable or await runtime.authored_records(
            checkpoint=ckpt,
            event=event,
            actor_id=actor_id,
        )
        existing_by_id = {
            character.character_id: character
            for character in ckpt.characters
        }
        existing_records = tuple(
            existing_by_id[record.character_id]
            for record in records
            if record.character_id in existing_by_id
        )
        if existing_records and len(existing_records) != len(records):
            raise RuntimeError(
                "spawn render batch is only partially present in the "
                f"accepted roster for event {event.event_id}"
            )
        if existing_records:
            # Ruleset transactions may atomically accept identity records
            # before presentation.  Reuse the durable, mechanics-enriched
            # records instead of trying to stage the authoring copies again.
            records = existing_records
        else:
            runtime.apply_records(ckpt, records)
        for record in records:
            staged_by_id.setdefault(record.character_id, record)
        refresh_router_history_record(
            ckpt.session_conversation,
            result=event,
            spawned_characters=records,
        )
    if durable_introductions:
        runtime.spawn_authoring.load_pending_introductions(
            checkpoint=ckpt,
            transaction_id=runtime.transaction_id,
            introductions=durable_introductions,
        )
    return tuple(staged_by_id.values())


def _rollback_speculative_spawn_roster(
    ckpt: CheckpointFile,
) -> dict[str, list[str]]:
    from app.engine.closed_event_runtime import closed_event_runtime_for

    runtime = closed_event_runtime_for(ckpt)
    if runtime is None:
        return {}
    rolled_back = runtime.spawn_authoring.rollback_roster(
        checkpoint=ckpt,
        transaction_id=runtime.transaction_id,
    )
    runtime.applied_character_ids.difference_update(rolled_back)
    return runtime.spawn_authoring.pending_introductions(
        runtime.transaction_id
    )


def _restore_speculative_spawn_roster(ckpt: CheckpointFile) -> None:
    from app.engine.closed_event_runtime import closed_event_runtime_for

    runtime = closed_event_runtime_for(ckpt)
    if runtime is None:
        return
    restored = runtime.spawn_authoring.restore_roster(
        checkpoint=ckpt,
        transaction_id=runtime.transaction_id,
    )
    runtime.applied_character_ids.update(restored)


def _accept_speculative_spawn_roster(ckpt: CheckpointFile) -> None:
    from app.engine.closed_event_runtime import closed_event_runtime_for

    runtime = closed_event_runtime_for(ckpt)
    if runtime is None:
        return
    accepted = runtime.spawn_authoring.accept_roster(
        checkpoint=ckpt,
        transaction_id=runtime.transaction_id,
    )
    runtime.applied_character_ids.update(accepted)


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
    soft_handoff_candidate: bool = False,
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
    rendered_event_ids_by_pov: dict[str, list[str]] = {}
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
    # narrator latency to 1× (slowest POV). Keep the buffers intact until
    # every narrator call succeeds; if the provider fails mid-render, the
    # orchestrator can persist and retry exactly this render without
    # replaying the router or character-agent calls that produced it.
    targets: list[tuple[str, list[RenderBufferEntry]]] = []
    for h in candidates:
        buf = list(ckpt.session.render_buffers.get(h, []))
        if not buf:
            continue  # Human had no perceivable events this beat.
        targets.append((h, buf))

    staged_spawn_records: tuple[CharacterRecord, ...] = ()
    if targets:
        staged_spawn_records = await _stage_closed_event_spawns_for_render(
            ckpt,
            events_closed=events_closed,
            event_actor_ids=event_actor_ids,
        )

    image_runtime = None
    image_transaction_id: str | None = None
    if targets:
        from app.engine.closed_event_runtime import closed_event_runtime_for

        image_runtime = closed_event_runtime_for(ckpt)
        if image_runtime is not None:
            closed_events = (
                ckpt.canonical_events[-events_closed:]
                if events_closed > 0
                else []
            )
            actor_ids_by_event_id = {
                event.event_id: actor
                for event, actor in zip(closed_events, event_actor_ids)
            }
            image_transaction_id = await image_runtime.start_render_candidate(
                checkpoint=ckpt,
                buffered_events_by_pov=dict(targets),
                actor_ids_by_event_id=actor_ids_by_event_id,
            )

    gate_id = ""
    gate_buffer: list[RenderBufferEntry] = []
    if soft_handoff_candidate and targets:
        target_ids = {character_id for character_id, _buf in targets}
        gate_id = (
            acting_player_id
            if acting_player_id in target_ids
            else targets[0][0]
        )
        gate_buffer = next(
            buf for character_id, buf in targets if character_id == gate_id
        )

    persist_pending = None
    if targets:
        persist_pending = getattr(
            dispatcher, "persist_pending_narrator_render", None,
        )
        if callable(persist_pending):
            ckpt.session.pending_narrator_render = PendingNarratorRender(
                ended_reason=ended_reason,
                events_closed=events_closed,
                event_actor_ids=list(event_actor_ids),
                acting_player_id=acting_player_id or "",
                acting_player_input=acting_player_input,
                release_slots=release_slots,
                force_partial=force_partial,
                suppress_reaction_prompts=suppress_reaction_prompts,
                soft_handoff_candidate=soft_handoff_candidate,
                handoff_event_id=(
                    gate_buffer[-1].event_id
                    if soft_handoff_candidate and gate_buffer
                    else ""
                ),
                pending_spawn_records=list(staged_spawn_records),
            )

    async def _render_one(
        h: str, buf: list[RenderBufferEntry],
    ) -> tuple[str, NarratorFinalOutput, TranscriptEntry]:
        pov_user_input = (
            acting_player_input if h == acting_player_id else ""
        )
        commitments = [
            commitment.description
            for commitment in ckpt.session.open_commitments
            if h in commitment.actor_ids and commitment.description
        ]
        handoff_context = (
            "Unresolved submitted activity: " + "; ".join(commitments)
            if commitments
            else "No unresolved submitted activity."
        )
        envelope, entry = await dispatcher.narrator_compose(
            ckpt=ckpt,
            character_id=h,
            buffered_events=buf,
            partial_mode_override=partial_override,
            user_input=pov_user_input,
            handoff_policy=(
                "candidate" if h == gate_id else "forced"
            ),
            handoff_context=handoff_context,
        )
        return h, envelope, entry

    if targets:
        narrator_lengths = {
            h: len(ckpt.narrator_conversations.get(h, []))
            for h, _buf in targets
        }
        visual_intro_snapshot = {
            viewer_id: list(character_ids)
            for viewer_id, character_ids in ckpt.session.visual_introductions.items()
        }
        results = await asyncio.gather(
            *(_render_one(h, buf) for h, buf in targets),
            return_exceptions=True,
        )
        errors = [
            result for result in results
            if isinstance(result, BaseException)
        ]
        if errors:
            try:
                if image_runtime is not None:
                    await image_runtime.reject_render_candidate(
                        image_transaction_id
                    )
            finally:
                for h, length in narrator_lengths.items():
                    history = ckpt.narrator_conversations.get(h)
                    if history is not None:
                        del history[length:]
                ckpt.session.visual_introductions = visual_intro_snapshot
                pending_spawn_introductions = (
                    _rollback_speculative_spawn_roster(ckpt)
                )
                if ckpt.session.pending_narrator_render is not None:
                    ckpt.session.pending_narrator_render.pending_spawn_introductions = (
                        pending_spawn_introductions
                    )
                if callable(persist_pending):
                    persist_pending(ckpt)
            raise errors[0]
        for h, envelope, _entry in results:
            narrator_handoff_logger.info(
                "Narrator handoff judgment: pov=%s decision=%s reason=%s "
                "events=%d",
                h,
                envelope.handoff,
                envelope.handoff_reason,
                len(dict(targets)[h]),
            )
        if soft_handoff_candidate:
            gate_result = next(
                envelope
                for h, envelope, _entry in results
                if h == gate_id
            )
            if gate_result.handoff == "continue":
                try:
                    if image_runtime is not None:
                        await image_runtime.reject_render_candidate(
                            image_transaction_id
                        )
                finally:
                    for h, length in narrator_lengths.items():
                        history = ckpt.narrator_conversations.get(h)
                        if history is not None:
                            del history[length:]
                    ckpt.session.visual_introductions = visual_intro_snapshot
                    _rollback_speculative_spawn_roster(ckpt)
                return BeatResult(
                    renders={},
                    events_closed=events_closed,
                    ended_reason="narrator_continue",
                    transcript_entries={},
                    event_actor_ids=event_actor_ids,
                    reaction_prompts={},
                    continue_requested=True,
                )
        if image_runtime is not None:
            image_runtime.accept_render_candidate(image_transaction_id)
        for h, buf in targets:
            current = ckpt.session.render_buffers.get(h, [])
            if current[:len(buf)] == buf:
                ckpt.session.render_buffers[h] = current[len(buf):]
            else:
                ckpt.session.render_buffers[h] = []
        for h, envelope, entry in results:
            commit_pov_render(
                ckpt,
                pov_character_id=h,
                buffered_events=dict(targets)[h],
                result=envelope,
                user_input=entry.user,
            )
            renders[h] = envelope.final_text
            transcript_entries[h] = entry
        rendered_event_ids_by_pov = {
            h: [entry.event_id for entry in buf]
            for h, buf in targets
        }

    _accept_speculative_spawn_roster(ckpt)

    if ckpt.session.pending_narrator_render is not None:
        ckpt.session.pending_narrator_render = None

    if release_slots:
        release_beat_slots(ckpt)
    max_events = max(1, int(ckpt.session.config.settings.max_events_per_beat))
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
        rendered_event_ids_by_pov=rendered_event_ids_by_pov,
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
        preview = strip_terminal_control(attempted_text).strip()
        if len(preview) > 1500:
            preview = preview[:1497] + "..."
        base += f"\n\n> Your submitted text:\n> {preview}"
    return base
