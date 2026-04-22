"""v11 turn loop — DM-paced, beat-based, Cat I/II intention pipeline.

This module codifies the Plan-v11 state machine for the turn pipeline.
It replaces the one-shot `process_turn → single narrator render` shape
with a beat-cascading loop: intentions flow in, are classified Cat I
(self-closing) or Cat II (contested responder-collecting), canonical
events are adjudicated, broadcast to in-scene observers, and the beat
continues through agent reactions until the router signals ends_beat
or the backstop fires.

## State machine summary

  enter(intention, actor_id):
    validate against scene.active_act_slot — claim, reject, or
    interpret-as-Cat-II-responder.

  classify(intention) → Cat I | Cat II
    Cat I → canonicalize, broadcast, check ends_beat.
    Cat II → open event, collect required responders (agents intend
      immediately; humans pin slot and wait), adjudicate on close,
      ends_beat=true implicit.

  broadcast(event):
    Append to every in-scene human's render buffer.
    Feed every in-scene agent's observation context.

  beat_end?:
    If true → render each in-scene human via their buffer, flush
    buffers, release slots in scene, park loop.
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
  active_act_slot — a scene's lock, holding 0..N character-slot entries.
  Cat I — self-closing intention (dialogue, passive, unambiguous).
  Cat II — contested intention, collects responder intentions.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable, Protocol


def _utcnow_iso() -> str:
    """v11-r3c: timezone-aware UTC ISO-8601 timestamp. Never use
    datetime.utcnow() (deprecated in 3.12, naive, and miscomputes
    staleness under non-UTC hosts)."""
    return datetime.now(timezone.utc).isoformat()


_REFUSAL_PATTERNS = (
    # Refusal+meta shapes: specific enough that they don't collide with
    # legitimate in-character dialogue like "I can't see them from here."
    "i can't help", "i can't assist", "i can't comply", "i can't generate",
    "i cannot help", "i cannot assist", "i cannot comply",
    "i cannot generate", "i cannot provide",
    "i'm unable to", "i am unable to",
    "i'm not able to", "i am not able to",
    "sorry, i can't", "sorry, i cannot", "sorry, but i",
    "as an ai", "as a language model",
    "i don't have access", "i do not have access",
    "i need to decline",
)


def _is_agent_refusal(text: str) -> bool:
    """v11-r4c: detect stock LLM refusal phrasings so we can skip the
    pick rather than routing "I cannot help with that" as an intention
    through the adjudicator.

    Narrowed rules (v11-r4c):

    1. **Sentence-start anchor only.** A refusal phrase must appear at
       the very start of the output or at a sentence boundary (after
       ``. ``, ``! ``, ``? ``, or a newline). Mid-sentence occurrences
       of "i can't" in dialogue (e.g. "I can't see them from here.")
       are NOT refusals.

    2. **Refusal+meta patterns only.** The pattern list is specific
       enough ("i can't help", "i'm unable to", "as an ai", ...) that
       in-character terse dialogue like "I can't allow that, my lord."
       slips through unflagged.

    3. **Length cap.** Outputs longer than 200 chars are never flagged
       — long narratives that happen to contain "I can't" as dialogue
       are preserved.

    4. **Empty/whitespace** is always a refusal (agent failure).
    """
    t = (text or "").strip().lower()
    if not t:
        return True
    if len(t) > 200:
        return False  # Long outputs are not refusals.
    # Build the set of sentence-start offsets: position 0, plus every
    # offset immediately following ". ", "! ", "? ", or a newline.
    starts = [0]
    for i in range(len(t) - 1):
        ch = t[i]
        nxt = t[i + 1]
        if ch == "\n":
            starts.append(i + 1)
        elif ch in ".!?" and nxt == " ":
            starts.append(i + 2)
    for pat in _REFUSAL_PATTERNS:
        for s in starts:
            if t.startswith(pat, s):
                return True
    return False

from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.state import OpenCatIIEvent, RenderBufferEntry, SlotEntry

logger = logging.getLogger(__name__)


class SceneLockManager:
    """v11: per-scene asyncio.Lock indexed by (session_id, scene_id).

    The orchestrator holds an instance; every /act acquires the lock
    for its scene before running check_act_slot → run_beat. Prevents
    the race where two concurrent /acts both see FREE and both claim
    the initiator slot.

    Locks are created lazily. They are NOT persisted; a process restart
    starts with no locks held (all scenes implicitly released on
    startup).
    """

    def __init__(self):
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._mutex = asyncio.Lock()

    async def get(self, session_id: str, scene_id: str) -> asyncio.Lock:
        key = (session_id, scene_id)
        async with self._mutex:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock


class SlotConflict(Enum):
    """v11 error taxonomy for /act rejection. See docs in `check_act_slot`.

    The orchestrator maps these to user-facing messages. Four distinct
    failure modes, each with a specific explanation:
    """
    # Someone else holds the scene's initiator slot.
    INITIATOR_HELD = "initiator_held"
    # A Cat II is pinned on another human — this user can't act.
    CAT_II_OTHER_HELD = "cat_ii_other_held"
    # This user is pinned as a Cat II responder in this scene. Their
    # /act IS accepted but interpreted as their responder intention,
    # not as a fresh initiator. (This is a marker, not an error.)
    CAT_II_SELF_RESPONDER = "cat_ii_self_responder"
    # This user already holds the initiator slot — their previous /act
    # is still mid-beat.
    SELF_BUSY = "self_busy"
    # No conflict — slot is claimable.
    FREE = "free"


@dataclass
class SlotCheck:
    """Outcome of validating an incoming /act against the scene's slots.

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


def check_act_slot(
    ckpt: CheckpointFile,
    scene_id: str,
    acting_character_id: str,
) -> SlotCheck:
    """Validate an incoming /act against the scene's active_act_slot.

    Returns a SlotCheck indicating whether to accept, reject, or
    interpret-as-Cat-II-response.
    """
    slot = ckpt.session.active_act_slots.get(scene_id, {})
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
        if entry.reason == "cat_ii_responder":
            return SlotCheck(
                conflict=SlotConflict.CAT_II_OTHER_HELD,
                holder_id=holder_id,
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
    scene_id: str,
    character_id: str,
) -> None:
    """Claim the scene's initiator slot for a human about to run a beat."""
    slot = ckpt.session.active_act_slots.setdefault(scene_id, {})
    slot[character_id] = SlotEntry(
        reason="initiator",
        cat_ii_event_id=None,
        claimed_at=_utcnow_iso(),
    )


def pin_cat_ii_responder(
    ckpt: CheckpointFile,
    scene_id: str,
    character_id: str,
    cat_ii_event_id: str,
) -> None:
    """Pin a human as a Cat II responder. The scene cannot accept
    unrelated /acts from other humans until the event closes."""
    slot = ckpt.session.active_act_slots.setdefault(scene_id, {})
    slot[character_id] = SlotEntry(
        reason="cat_ii_responder",
        cat_ii_event_id=cat_ii_event_id,
        claimed_at=_utcnow_iso(),
    )


def release_scene_slots(ckpt: CheckpointFile, scene_id: str) -> None:
    """Release slot entries for a scene at beat end.

    CRITICAL: this does NOT clobber pins tied to still-open Cat II
    events. If a beat ends while a different Cat II is still awaiting
    human responders, the pins for those responders stay. Only initiator
    slots and responder slots whose owning event has already closed are
    released.
    """
    slot = ckpt.session.active_act_slots.get(scene_id)
    if not slot:
        return
    open_evt_ids = {e.event_id for e in ckpt.session.open_cat_ii_events}
    keep: dict[str, SlotEntry] = {}
    for cid, entry in slot.items():
        if entry.reason == "cat_ii_responder" and entry.cat_ii_event_id in open_evt_ids:
            keep[cid] = entry
    if keep:
        ckpt.session.active_act_slots[scene_id] = keep
    else:
        ckpt.session.active_act_slots.pop(scene_id, None)


def purge_character_state(ckpt: CheckpointFile, character_id: str) -> None:
    """Sweep all v11 bookkeeping when a character leaves/culls/unbinds.

    Called by the roster-move / /leave / cull code paths so state that
    would otherwise strand (pins, open events they're required in,
    render buffers) doesn't leak.

    What this does:
    - Drops the character's entry from every scene's active_act_slots.
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
    for scene_id, slot in list(ckpt.session.active_act_slots.items()):
        if character_id in slot:
            slot.pop(character_id, None)
            if not slot:
                ckpt.session.active_act_slots.pop(scene_id, None)

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

        for h in stale_humans:
            # Structured marker, not a magic string. Part C of the router
            # prompt skips swept responders entirely — they don't end up
            # in the adjudication input, so no meta text can leak into
            # `resolved_outcome`. The rendered scene will show the
            # character as present-but-non-reactive via narrator
            # guidance, never as "does not act — away from the scene".
            if h not in evt.swept_responders:
                evt.swept_responders.append(h)
            # Sentinel string still recorded in collected_intentions for
            # debug inspection, but adjudication will filter these out
            # via swept_responders.
            evt.collected_intentions[h] = "[AFK-swept: no player intention]"
            logger.warning(
                "Cat II event %s: auto-resolved pin on %s after timeout",
                evt.event_id, h,
            )
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


def abort_scene(ckpt: CheckpointFile, scene_id: str) -> int:
    """Admin-escape: force-release every slot in a scene, abandon every
    open Cat II event in that scene, AND flush render buffers for every
    human currently in the scene. Returns the number of events
    abandoned.

    Used by the `/abort_beat` admin command when a pin gets wedged.
    Preserves the canonical_events log (we keep history) but the
    mid-flight Cat II is treated as if it never resolved. Render buffers
    are flushed so mid-beat aborts don't leak stale events into the next
    beat's render fan-out.
    """
    # Snapshot in-scene humans BEFORE slots are popped — humans_in_scene
    # derives from character location + bindings, so slot state doesn't
    # affect it, but pinning this ordering keeps intent clear.
    in_scene_humans = humans_in_scene(ckpt, scene_id)
    buffers_cleared = 0
    for h in in_scene_humans:
        existing = ckpt.session.render_buffers.get(h)
        if existing:
            buffers_cleared += 1
        ckpt.session.render_buffers[h] = []

    ckpt.session.active_act_slots.pop(scene_id, None)
    before = len(ckpt.session.open_cat_ii_events)
    ckpt.session.open_cat_ii_events = [
        e for e in ckpt.session.open_cat_ii_events if e.scene_id != scene_id
    ]
    dropped = before - len(ckpt.session.open_cat_ii_events)
    logger.warning(
        "abort_scene: %s released; %d open Cat II events abandoned; "
        "%d render buffers cleared",
        scene_id, dropped, buffers_cleared,
    )
    return dropped


def release_character_slot(
    ckpt: CheckpointFile,
    scene_id: str,
    character_id: str,
) -> None:
    """Release one character's slot entry while leaving the scene's
    other entries intact. Used when a single Cat II responder closes
    their intention but others are still pending."""
    slot = ckpt.session.active_act_slots.get(scene_id)
    if slot is None:
        return
    slot.pop(character_id, None)
    if not slot:
        ckpt.session.active_act_slots.pop(scene_id, None)


def new_event_id() -> str:
    """Fresh event id for canonical events and open Cat II events."""
    return f"evt_{uuid.uuid4().hex[:12]}"


def open_cat_ii(
    ckpt: CheckpointFile,
    scene_id: str,
    initiator_id: str,
    initiator_intention: str,
    required_responders: list[str],
) -> OpenCatIIEvent:
    """Open a Cat II event and register it on the checkpoint. Humans
    among required_responders are pinned into the scene's slot."""
    evt = OpenCatIIEvent(
        event_id=new_event_id(),
        scene_id=scene_id,
        initiator_id=initiator_id,
        initiator_intention=initiator_intention,
        required_responders=list(required_responders),
        collected_intentions={},
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


# ---- Broadcast + render buffering ------------------------------------------


def append_to_render_buffer(
    ckpt: CheckpointFile,
    character_id: str,
    event_id: str,
    observation_level: str = "direct",
) -> None:
    """Queue a canonical event for a human's next render."""
    buf = ckpt.session.render_buffers.setdefault(character_id, [])
    buf.append(
        RenderBufferEntry(
            event_id=event_id,
            observation_level=observation_level,
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


def humans_in_scene(ckpt: CheckpointFile, scene_id: str) -> list[str]:
    """Return character_ids of human-bound characters currently in the
    given scene. A 'human-bound' character is one with an entry in
    session.character_bindings.
    """
    bindings = ckpt.session.character_bindings or {}
    return [
        c.character_id
        for c in ckpt.characters
        if c.location == scene_id and c.character_id in bindings
    ]


def broadcast_event(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    scene_id: str,
) -> list[str]:
    """Append a closed canonical event to the log and fan it out to
    every in-scene human's render buffer.

    `scene_id` is the caller-authoritative scene — always the beat's
    scene, never derived from session state, so concurrent scenes
    can't leak buffers into each other.

    The event's stable `event_id` is what lands in render buffers
    (not Python object identity) so checkpoints remain resolvable
    across process restarts.
    """
    ckpt.canonical_events.append(event)
    humans = humans_in_scene(ckpt, scene_id)

    obs_level_by_char: dict[str, str] = {}
    for o in event.observers:
        # Legacy: observation_level is a single-char code ("d"|"i"|"f").
        obs_level_by_char[o.character_id] = {
            "d": "direct", "i": "indirect", "f": "inferred",
        }.get(o.observation_level, "direct")

    for h in humans:
        level = obs_level_by_char.get(h, "direct")
        append_to_render_buffer(ckpt, h, event.event_id, level)

    return humans


# ---- The beat loop ---------------------------------------------------------
#
# This is the heart of v11. `run_beat` is the orchestrator entry point for
# one player's /act or Cat II responder intention. It cascades through
# agent reactions until the router ends the beat, then fans the narrator
# out per in-scene human and releases the scene's slots.
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
        scene_id: str,
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

    async def agent_intend(
        self,
        ckpt: CheckpointFile,
        character_id: str,
        scene_id: str,
    ) -> str:
        """Ask an agent for their next intention (as free-form text).
        Returns the intention string; the orchestrator re-routes it
        through route_intention as a fresh intention."""
        ...

    async def narrator_compose(
        self,
        ckpt: CheckpointFile,
        character_id: str,
        buffered_events: list[RenderBufferEntry],
    ) -> str:
        """Render this human's POV prose for the beat. Input: their
        buffered events since last render, observation levels tagged.
        Output: second-person prose per narrator_phase2_v7."""
        ...


@dataclass
class BeatResult:
    """Summary of a single beat's run, returned to the orchestrator for
    delivery. `renders` is keyed by character_id; orchestrator posts each
    to that player's delivery channel (thread / ephemeral / DM)."""
    renders: dict[str, str]
    events_closed: int
    ended_reason: str  # "ends_beat" | "max_events_cap" | "cat_ii_pending" | ...


async def run_beat(
    ckpt: CheckpointFile,
    dispatcher: Dispatcher,
    actor_id: str,
    intention: str,
    scene_id: str,
    cat_ii_event_id: str | None = None,
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

    Termination:
    - Router's ends_beat=true → render fan-out, slot release.
    - Cat II event adjudicates (always implicit ends_beat) → same.
    - max_events_per_beat reached → forced render + slot release.
    """
    max_events = ckpt.session.config.settings.max_events_per_beat

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
            )
        # Free this specific character's slot — others may still be pinned.
        release_character_slot(ckpt, scene_id, actor_id)
        if not cat_ii_is_ready(evt):
            # Still waiting on other responders. Beat stays paused.
            return BeatResult(
                renders={},
                events_closed=0,
                ended_reason="cat_ii_pending",
            )
        # All responders in — adjudicate.
        # Use evt.scene_id (where the event opened) rather than the
        # caller's scene_id — the responder may have moved scenes
        # between pin and /act (unlikely but possible via admin or
        # roster_move), and the canonical event belongs to the scene
        # where it originated.
        resolution_scene = evt.scene_id
        resolved = await dispatcher.route_intention(
            ckpt=ckpt,
            actor_id=evt.initiator_id,
            intention=evt.initiator_intention,
            scene_id=resolution_scene,
            cat_ii_event=evt,
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
        broadcast_event(ckpt, resolved, resolution_scene)
        # Cat II adjudication always ends the beat.
        return await _end_beat(
            ckpt, dispatcher, scene_id,
            ended_reason="cat_ii_resolution",
            events_closed=1,
        )

    # Fresh initiator path.
    claim_initiator_slot(ckpt, scene_id, actor_id)

    events_closed = 0
    current_intention = intention
    current_actor = actor_id

    while True:
        # Route the current intention.
        result = await dispatcher.route_intention(
            ckpt=ckpt,
            actor_id=current_actor,
            intention=current_intention,
            scene_id=scene_id,
            cat_ii_event=None,
        )

        if result.requires_responders:
            # Cat II: open the event, pin humans, request agent
            # responder intentions immediately (they're fast).
            # Filter: the initiator can never be their own responder,
            # even if the router hallucinates it — would either double-
            # pin them or overwrite the initiator slot.
            required = [
                r for r in result.required_responders if r != current_actor
            ]
            if not required:
                # The only "responder" was the initiator themselves; treat
                # this as Cat I — there's nothing to contest. Broadcast the
                # canonical event as-is and continue.
                broadcast_event(ckpt, result, scene_id)
                events_closed += 1
                if events_closed >= max_events:
                    return await _end_beat(
                        ckpt, dispatcher, scene_id,
                        ended_reason="max_events_cap",
                        events_closed=events_closed,
                    )
                # Fall through to the standard Cat I ends_beat check.
                if result.ends_beat:
                    return await _end_beat(
                        ckpt, dispatcher, scene_id,
                        ended_reason=result.ends_beat_reason
                            or "cascade_exhausted",
                        events_closed=events_closed,
                    )
                # Keep cascading via picks — reuse the normal path by
                # letting the loop iterate. Break out of the inner if
                # and continue.
                picks = [
                    rid for rid in result.agent_responder_picks
                    if rid not in (ckpt.session.character_bindings or {})
                ]
                if not picks:
                    return await _end_beat(
                        ckpt, dispatcher, scene_id,
                        ended_reason="cascade_exhausted",
                        events_closed=events_closed,
                    )
                next_actor = picks[0]
                next_intention = await dispatcher.agent_intend(
                    ckpt=ckpt, character_id=next_actor, scene_id=scene_id,
                )
                current_actor = next_actor
                current_intention = next_intention
                continue

            evt = open_cat_ii(
                ckpt=ckpt,
                scene_id=scene_id,
                initiator_id=current_actor,
                initiator_intention=current_intention,
                required_responders=required,
            )

            # Agents among required_responders intend immediately; their
            # intentions are collected into the same Cat II event.
            bindings = ckpt.session.character_bindings or {}
            for rid in required:
                if rid in bindings:
                    # Human — pin slot, wait.
                    pin_cat_ii_responder(ckpt, scene_id, rid, evt.event_id)
                else:
                    # Agent — intend inline.
                    ai_intent = await dispatcher.agent_intend(
                        ckpt=ckpt,
                        character_id=rid,
                        scene_id=scene_id,
                    )
                    collect_cat_ii_intention(
                        ckpt, evt.event_id, rid, ai_intent
                    )

            if cat_ii_is_ready(evt):
                # No humans in required list — resolve immediately.
                resolved = await dispatcher.route_intention(
                    ckpt=ckpt,
                    actor_id=evt.initiator_id,
                    intention=evt.initiator_intention,
                    scene_id=scene_id,
                    cat_ii_event=evt,
                )
                close_cat_ii(ckpt, evt.event_id)
                if resolved.requires_responders:
                    raise ValueError(
                        "Router returned Cat II nesting on an adjudication "
                        "call; Part C invariant violated."
                    )
                broadcast_event(ckpt, resolved, scene_id)
                events_closed += 1
                return await _end_beat(
                    ckpt, dispatcher, scene_id,
                    ended_reason="cat_ii_resolution",
                    events_closed=events_closed,
                )
            # Humans are pinned — pause the beat here. Their /acts will
            # re-enter run_beat with their cat_ii_event_id.
            return BeatResult(
                renders={},
                events_closed=events_closed,
                ended_reason="cat_ii_pending",
            )

        # Cat I path — canonical event closes immediately.
        broadcast_event(ckpt, result, scene_id)
        events_closed += 1

        # Ends-beat decision.
        if result.ends_beat or events_closed >= max_events:
            reason = result.ends_beat_reason or "max_events_cap"
            if events_closed >= max_events and not result.ends_beat:
                reason = "max_events_cap"
            return await _end_beat(
                ckpt, dispatcher, scene_id,
                ended_reason=reason,
                events_closed=events_closed,
            )

        # Pick the next actor for the cascade. Agents only — humans
        # don't continue a beat unless they /act fresh.
        picks = [
            rid for rid in result.agent_responder_picks
            if rid not in (ckpt.session.character_bindings or {})
        ]
        if not picks:
            # Router signaled continue but gave no pick — treat as
            # cascade_exhausted and end beat.
            return await _end_beat(
                ckpt, dispatcher, scene_id,
                ended_reason="cascade_exhausted",
                events_closed=events_closed,
            )
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
            raw = await dispatcher.agent_intend(
                ckpt=ckpt,
                character_id=candidate,
                scene_id=scene_id,
            )
            if raw and raw.strip() and not _is_agent_refusal(raw):
                next_actor = candidate
                next_intention = raw
                break
            logger.warning(
                "Dropped empty/refusal intention from agent %s", candidate,
            )
        if next_actor is None or next_intention is None:
            return await _end_beat(
                ckpt, dispatcher, scene_id,
                ended_reason="cascade_exhausted",
                events_closed=events_closed,
            )
        current_actor = next_actor
        current_intention = next_intention


async def _end_beat(
    ckpt: CheckpointFile,
    dispatcher: Dispatcher,
    scene_id: str,
    ended_reason: str,
    events_closed: int,
) -> BeatResult:
    """Compose per-human renders, flush buffers, release scene slots."""
    renders: dict[str, str] = {}
    for h in humans_in_scene(ckpt, scene_id):
        buf = flush_render_buffer(ckpt, h)
        if not buf:
            continue  # Human had no perceivable events this beat.
        prose = await dispatcher.narrator_compose(
            ckpt=ckpt,
            character_id=h,
            buffered_events=buf,
        )
        renders[h] = prose
    release_scene_slots(ckpt, scene_id)
    logger.info(
        "Beat closed in %s: events=%d reason=%s renders=%d",
        scene_id, events_closed, ended_reason, len(renders),
    )
    return BeatResult(
        renders=renders, events_closed=events_closed, ended_reason=ended_reason,
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
            f"go through. You'll see a new render appear when the scene "
            f"re-opens; try again then."
        )
    elif check.conflict == SlotConflict.CAT_II_OTHER_HELD:
        base = (
            f"The scene is paused on **{holder_name}** — someone's action "
            f"is waiting on their response. Your /act didn't go through. "
            f"You'll see a new render when the beat closes."
        )
    elif check.conflict == SlotConflict.SELF_BUSY:
        base = (
            "Your previous /act is still processing. Give the scene a "
            "moment before submitting again."
        )
    elif check.conflict == SlotConflict.CAT_II_SELF_RESPONDER:
        # Not an error — this is the "your /act was accepted as your Cat II
        # response" case. Callers usually don't render this path, but
        # provide a friendly confirmation if they do.
        base = (
            "Your /act was accepted as your response to the current contested "
            "action. The scene will resolve once all responders have moved."
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
