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
import re
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
from app.schemas.event_router import EventRouterOutput
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    SceneDelta,
    WorldAdjudication,
    visible_fact_texts,
)
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import OpenCatIIEvent, RenderBufferEntry, SlotEntry

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

        mutated = False
        for h in stale_humans:
            # Structured marker, not a magic string. Part C of the router
            # prompt skips swept responders entirely — they don't end up
            # in the adjudication input, so no meta text can leak into
            # `resolved_outcome`. The rendered scene will show the
            # character as present-but-non-reactive via narrator
            # guidance, never as "does not act — away from the scene".
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


def _scene_member_ids(ckpt: CheckpointFile, scene_id: str) -> set[str]:
    """v11-r7g: set of character_ids currently in `scene_id`, used to
    filter the router's agent_responder_picks before dispatch.

    Pre-r7g the engine accepted any pick the router emitted; the
    router's prompt asked for in-scene picks but didn't enforce it,
    and the playtest log showed routine over-reaches: Mira /act'd at
    archive_main_hall, the router picked `pip` and `nyx` (both at
    bell_of_arrivals), agent_intend() returned empty/refusal because
    they had nothing to react to, and the engine logged a warning per
    pick. Same root cause as the spawn-cap (router over-eager) but
    cheaper to fix at the engine boundary because the pick set is
    small and the location data is local.

    Set rather than list because membership tests dominate the call
    site (filter comprehension); with ~20-character rosters either
    is fine but set documents intent.
    """
    return {
        c.character_id for c in ckpt.characters if c.location == scene_id
    }


def _log_router_rationale(
    result: EventRouterOutput, actor_id: str, scene_id: str,
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
        "router[%s] actor=%s scene=%s cat=%s ends_beat=%s reason=%r picks=%s :: %s",
        kind, actor_id, scene_id,
        "II" if result.requires_responders else "I",
        result.ends_beat, result.ends_beat_reason,
        result.agent_responder_picks, rationale,
    )


def _filter_picks_for_dispatch(
    ckpt: CheckpointFile,
    scene_id: str,
    picks: list[str],
) -> list[str]:
    """v11-r7g: drop picks the engine refuses to dispatch.

    Two filters, in order:
      1. Humans — anyone the engine considers human-controlled,
         per `collect_player_ids`. Humans don't cascade via the
         router; they only enter through `/act`. Using
         `collect_player_ids` here (rather than the bare bindings
         dict) is the load-bearing fix for the playable-2 era:
         `session.player_character_id` may name the creator's bound
         character without there being a corresponding entry in
         `character_bindings` (older saves; CLI single-player
         flows). Filtering on bindings alone let those creator-
         bound characters slip into NPC dispatch and produced the
         "router tried to make my own character speak" symptom.
      2. Out-of-scene NPCs — picks must be in the beat's scene to
         have any perceptual context to react to. Out-of-scene picks
         routinely produced empty/refusal intentions in the playtest;
         filtering at the engine boundary saves an LLM call AND
         removes the WARNING noise pre-r7g produced for legitimate
         router over-reach.

    Returns the filtered list preserving router order.
    """
    # Local import to avoid an engine-package import cycle on module
    # load (context_builder pulls some of the same schemas turn_loop
    # exports). Cheap — the function is tiny and the import is cached.
    from app.engine.context_builder import collect_player_ids

    humans = collect_player_ids(ckpt)
    in_scene = _scene_member_ids(ckpt, scene_id)
    return [
        rid for rid in picks
        if rid not in humans and rid in in_scene
    ]


def broadcast_event(
    ckpt: CheckpointFile,
    event: EventRouterOutput,
    scene_id: str,
    actor_id: str = "",
) -> list[str]:
    """Append a closed canonical event to the log and fan it out to
    (a) every in-scene human's render buffer, and (b) every NPC observer's
    `pending_observations` queue (in-scene and off-scene alike, except the
    event's own actor).

    `scene_id` is the caller-authoritative scene — always the beat's
    scene, never derived from session state, so concurrent scenes
    can't leak buffers into each other.

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
    channel router rule 13 promises. As of v11-r9b, fan-out is
    strictly in-scene:

    - **In-scene observers** (NPCs co-located with the event but not
      the actor): receive their visible observable_facts as inbox entries
      (no `[in-scene perception]` tag — the entries are the agent's
      live sensorium for this scene and don't need a routing
      label). These are the cascade pool. When one of them is picked
      as `agent_responder` for this same beat, their `respond` call
      drains the queue and they see the just-broadcast event as the
      most recent entry — which IS the live event they're being asked
      to react to.
      In-scene NPCs who AREN'T picked this beat keep accumulating
      events silently and drain on whatever future beat picks them.
      This is the only viable channel: dispatching a per-event LLM
      call to every in-scene non-actor would burn budget on agents
      who have no opening to speak this beat. The user message they
      receive when they finally fire carries the full set in one shot.

    - **Off-scene observers** are NOT pushed (v11-r9b). Pre-r9b the
      router's `observers` list could include characters in other
      scenes (a courier delivery to a remote NPC, an ambient world
      event), and `broadcast_event` would push the same one-line
      summary to all of them tagged `[off-scene perception]`. That
      worked when summaries were narrow ("a courier carries the
      writ to Marcus") but failed catastrophically when an event
      fused private and public sub-beats into one omniscient
      `resolved_outcome` line: the t8 playtest caught Ashara
      receiving "Dan Garvey pulls on clothes from the wardrobe he
      didn't choose and walks downstairs ..." as an off-scene
      perception in the dining hall — a privacy violation and an
      attribution-confusion seed. Off-scene awareness now requires
      a separate event whose `scene_id` is where the news actually
      lands.

    Pre-r8b the in-scene path was a bug-hatchery: cascade NPCs got
    `observed_facts=[]` from `LLMDispatcher.agent_intend`, so they
    reacted to whatever stale entries already sat in their queue
    (typically off-scene perceptions from when the scene opened).
    The router then saw an off-topic intention and fabricated
    plausible dialogue to fit the cascade slot — the "narrator
    summarized dialogue" symptom the v11-r8b playtest caught was
    the visible end of that chain.

    Note: the `actor_id` of the event is also surfaced via
    BeatResult.event_actor_ids by callers that need actor-aware
    event-application downstream (e.g. _apply_roster_moves for
    the v11-r7h self-move path).
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

    bindings = ckpt.session.character_bindings or {}
    in_scene_ids = _scene_member_ids(ckpt, scene_id)
    by_id = {c.character_id: c for c in ckpt.characters}
    # NPC perception payload. Pre-v11-r10 this was the router's
    # `resolved_outcome` — narrator-grade prose with interior
    # interpretation woven in ("the strain of speaking close to the
    # edge of what she is permitted"), which leaked author-voice
    # interior into in-scene NPCs as their own perception. r10 moved
    # the channel to `observable_facts`. r11 adds fact-level
    # visibility: public facts go to every observer, but private
    # facts (`audience="only"`) are filtered per recipient before
    # anything reaches an NPC inbox.
    canonical = event.canonical_event

    for o in event.observers:
        if o.character_id in bindings:
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
        facts = visible_fact_texts(canonical.observable_facts, o.character_id)
        if facts:
            if len(facts) == 1:
                payload = facts[0]
            else:
                payload = "\n".join(f"  - {f}" for f in facts)
        else:
            payload = ""
        if not payload:
            # No observable_facts to push — silence is the right
            # outcome. (We deliberately do NOT fall back to the
            # outcome string; that's the leak this fix closes.)
            continue
        # v11-r9b: off-scene observers no longer get the summary
        # pushed. Pre-r9b every router-listed observer received the
        # full `resolved_outcome` regardless of where they were
        # standing — the t8 playtest caught this surfacing Dan
        # Garvey's private bedroom wardrobe choice on Ashara's queue
        # because the router fused his transit-and-arrival into one
        # outcome string and listed her as an observer of his
        # arrival. The fan-out is now strictly in-scene: a character
        # only gets an inbox push for events that actually fired in
        # the room they were standing in. Real cross-scene
        # awareness (a courier carrying a note, a public
        # announcement, magic) belongs in a separate event whose
        # `scene_id` is wherever the news lands, not piggy-backed
        # on an in-scene event's omniscient summary.
        if o.character_id not in in_scene_ids:
            continue
        recipient.pending_observations.append(payload)

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


def _build_open_attempt_event(
    initiator_name: str,
    initiator_intention: str,
    scene_id: str,
) -> EventRouterOutput:
    """v11-r6a: synthesize a transient 'open Cat II attempt' canonical
    event for rendering PARTIAL-mode cliffhangers to pinned humans (and
    the initiator if human).

    The attempt is IN PROGRESS. The audit string in `resolved_outcome`
    and the seed observable fact both describe the attempt mid-flight
    ('{initiator_name} attempts: ...') — the narrator's PARTIAL mode
    ends the prose on that moment. Pre-Option-B the narrator read
    `resolved_outcome` as the prose seed; post-Option-B it reads
    `observable_facts`, so the seed must live in both fields (the
    audit line for `/history` and tooling, the fact for the narrator
    render). This event IS appended to canonical_events so the
    narrator's event-id lookup resolves; on resolution the adjudicated
    event lands separately in the log. The resulting "two events per
    Cat II" shape reflects historical truth — an attempt WAS made
    pre-resolution — at the cost of one extra event entry per Cat II.

    `initiator_name` MUST be the display name (e.g. "Pip"), NOT the
    character_id (e.g. "char_0dab1f"). The narrator reads the seed
    fact as prose; passing the id leaks engine structure into the
    cliffhanger.
    """
    # Use contracts helper so the seed string is consistent with the
    # router's own Part C framing in both fields.
    from app.engine.turn_loop_contracts import format_open_attempt_outcome
    seed = format_open_attempt_outcome(initiator_name, initiator_intention)
    return EventRouterOutput(
        event_id="",  # _assign_event_id validator mints a fresh one
        decision_rationale=(
            "Synthetic open-attempt event for Cat II partial-mode "
            "render; no LLM call was made for this routing."
        ),
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                attempted_action=initiator_intention,
                feasible=True,
                resolved_outcome=seed,
            ),
            scene_delta=SceneDelta(time_advanced_seconds=0),
            observable_facts=[ObservableFact.all(seed)],
        ),
        observers=[],
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="cat_ii_open",
        spawn=[],
        dormant=[],
        cull=[],
        roster_moves=[],
        scenes_created=[],
    )


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
    `canonical_events[-events_closed + i]`. Required by the orchestrator's
    apply loop in v11-r7h so `_apply_roster_moves` can recognize self-
    moves (the actor moving themselves) and apply them despite the
    player-bound / pinned guards. Empty when `events_closed == 0`.
    """
    renders: dict[str, str]
    events_closed: int
    ended_reason: str  # "ends_beat" | "max_events_cap" | "cat_ii_pending" | ...
    transcript_entries: dict[str, TranscriptEntry]
    event_actor_ids: list[str]


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
                transcript_entries={}, event_actor_ids=[],
            )
        # Free this specific character's slot — others may still be pinned.
        release_character_slot(ckpt, scene_id, actor_id)
        if not cat_ii_is_ready(evt):
            # Still waiting on other responders. Beat stays paused.
            return BeatResult(
                renders={},
                events_closed=0,
                ended_reason="cat_ii_pending",
                transcript_entries={}, event_actor_ids=[],
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
        _log_router_rationale(
            resolved, evt.initiator_id, resolution_scene, kind="cat_ii_resolve",
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
        broadcast_event(
            ckpt, resolved, resolution_scene, actor_id=evt.initiator_id,
        )
        # Cat II adjudication always ends the beat.
        return await _end_beat(
            ckpt, dispatcher, scene_id,
            ended_reason="cat_ii_resolution",
            events_closed=1,
            event_actor_ids=[evt.initiator_id],
            acting_player_id=actor_id,
            acting_player_input=intention,
        )

    # Fresh initiator path.
    claim_initiator_slot(ckpt, scene_id, actor_id)

    events_closed = 0
    event_actor_ids: list[str] = []
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
        _log_router_rationale(result, current_actor, scene_id, kind="route")

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
                broadcast_event(ckpt, result, scene_id, actor_id=current_actor)
                event_actor_ids.append(current_actor)
                events_closed += 1
                if events_closed >= max_events:
                    return await _end_beat(
                        ckpt, dispatcher, scene_id,
                        ended_reason="max_events_cap",
                        events_closed=events_closed,
                        event_actor_ids=event_actor_ids,
                        acting_player_id=actor_id,
                        acting_player_input=intention,
                    )
                # Fall through to the standard Cat I ends_beat check.
                if result.ends_beat:
                    return await _end_beat(
                        ckpt, dispatcher, scene_id,
                        ended_reason=result.ends_beat_reason
                            or "cascade_exhausted",
                        events_closed=events_closed,
                        event_actor_ids=event_actor_ids,
                        acting_player_id=actor_id,
                        acting_player_input=intention,
                    )
                # Keep cascading via picks — reuse the normal path by
                # letting the loop iterate. Break out of the inner if
                # and continue. v11-r7g: filter humans + out-of-scene
                # in one helper.
                picks = _filter_picks_for_dispatch(
                    ckpt, scene_id, result.agent_responder_picks,
                )
                if not picks:
                    return await _end_beat(
                        ckpt, dispatcher, scene_id,
                        ended_reason="cascade_exhausted",
                        events_closed=events_closed,
                        event_actor_ids=event_actor_ids,
                        acting_player_id=actor_id,
                        acting_player_input=intention,
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
                _log_router_rationale(
                    resolved, evt.initiator_id, scene_id,
                    kind="cat_ii_resolve_inline",
                )
                close_cat_ii(ckpt, evt.event_id)
                if resolved.requires_responders:
                    raise ValueError(
                        "Router returned Cat II nesting on an adjudication "
                        "call; Part C invariant violated."
                    )
                broadcast_event(
                    ckpt, resolved, scene_id, actor_id=evt.initiator_id,
                )
                event_actor_ids.append(evt.initiator_id)
                events_closed += 1
                return await _end_beat(
                    ckpt, dispatcher, scene_id,
                    ended_reason="cat_ii_resolution",
                    events_closed=events_closed,
                    event_actor_ids=event_actor_ids,
                    acting_player_id=actor_id,
                    acting_player_input=intention,
                )
            # Humans are pinned — pause the beat here. Their /acts will
            # re-enter run_beat with their cat_ii_event_id.
            #
            # v11-r6a: before returning, render a PARTIAL-mode cliffhanger
            # for the pinned humans (and the initiator if human) so they
            # see the mid-attempt moment instead of waiting blind. Synth
            # an "open attempt" canonical event, buffer it for the render
            # targets at observation_level "direct", and reuse _end_beat
            # — but with release_slots=False (keep pins alive) and
            # force_partial=True (every render gets partial_mode=True
            # even though the initiator isn't pinned themselves).
            initiator_name = current_actor
            for c in ckpt.characters:
                if c.character_id == current_actor:
                    initiator_name = c.name
                    break
            open_attempt = _build_open_attempt_event(
                initiator_name=initiator_name,
                initiator_intention=current_intention,
                scene_id=scene_id,
            )
            ckpt.canonical_events.append(open_attempt)
            # The synthetic open-attempt event is initiator-authored;
            # tracking the actor for it keeps event_actor_ids parallel
            # with the canonical_events tail. No roster_moves are
            # attached to the open attempt (it's a render-only stub),
            # so this is bookkeeping rather than functional, but the
            # invariant must hold for the orchestrator's apply zip.
            event_actor_ids.append(current_actor)

            render_targets: set[str] = set()
            for rid in required:
                if rid in bindings:
                    render_targets.add(rid)
            if current_actor in bindings:
                render_targets.add(current_actor)

            for cid in render_targets:
                append_to_render_buffer(
                    ckpt, cid, open_attempt.event_id,
                    observation_level="direct",
                )

            events_closed += 1
            return await _end_beat(
                ckpt, dispatcher, scene_id,
                ended_reason="cat_ii_pending",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                release_slots=False,
                force_partial=True,
                render_only=render_targets,
                acting_player_id=actor_id,
                acting_player_input=intention,
            )

        # Cat I path — canonical event closes immediately.
        broadcast_event(ckpt, result, scene_id, actor_id=current_actor)
        event_actor_ids.append(current_actor)
        events_closed += 1

        # v11-r8a: observation_harvest fork. The router signals
        # `ends_beat_reason="observation_harvest"` when the actor is
        # purely observing in-scene NPCs (looking, studying, scanning
        # without dialogue or contact). We bypass the cascade and
        # instead fire each pick's `perceive()` in parallel to harvest
        # one self-presentation fragment per target. Fragments are
        # appended to the just-broadcast event's `observable_facts`
        # so the narrator's render reads them naturally as part of
        # the scene's perceptual surface.
        #
        # Mutating the canonical event after broadcast is safe:
        # `broadcast_event` only takes references (event_id into render
        # buffers; off-scene inbox uses `resolved_outcome` not
        # observable_facts), and the narrator reads the live
        # `observable_facts` list at compose time. The harvest path
        # respects the same human / out-of-scene filter as the
        # cascade picks (a router that picked a human or an off-scene
        # character should not cause us to fire a perception call on
        # them).
        if result.ends_beat_reason == "observation_harvest":
            harvest_picks = _filter_picks_for_dispatch(
                ckpt, scene_id, result.agent_responder_picks,
            )
            if harvest_picks:
                fragments = await dispatcher.harvest_perceptions(
                    ckpt=ckpt,
                    character_ids=harvest_picks,
                    acting_character_id=current_actor,
                )
                # Build "<Name>: <fragment>" lines so the narrator can
                # tell whose loadout each line describes. Empty
                # fragments (LLM failure for that pick) are dropped
                # silently — one bad perception should not poison the
                # render. The order matches harvest_picks (preserves
                # router intent).
                by_id = {c.character_id: c for c in ckpt.characters}
                appended = 0
                for cid, fragment in zip(harvest_picks, fragments):
                    text = _sanitize_harvest_fragment(fragment)
                    if not text:
                        logger.warning(
                            "Harvest: empty fragment for %s; dropped", cid,
                        )
                        continue
                    name = by_id.get(cid)
                    name = name.name if name else cid
                    result.canonical_event.observable_facts.append(
                        ObservableFact.all(f"[loadout — {name}] {text}")
                    )
                    appended += 1
                logger.info(
                    "Observation harvest: %d/%d fragments appended to "
                    "event %s",
                    appended, len(harvest_picks), result.event_id,
                )
            else:
                logger.warning(
                    "ends_beat_reason='observation_harvest' but no "
                    "harvestable picks after filtering; falling through "
                    "as a sparse Cat I close.",
                )

        # Ends-beat decision.
        if result.ends_beat or events_closed >= max_events:
            reason = result.ends_beat_reason or "max_events_cap"
            if events_closed >= max_events and not result.ends_beat:
                reason = "max_events_cap"
            return await _end_beat(
                ckpt, dispatcher, scene_id,
                ended_reason=reason,
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
            )

        # Pick the next actor for the cascade. Agents only — humans
        # don't continue a beat unless they /act fresh. v11-r7g:
        # also filter out-of-scene picks (the router routinely
        # over-reaches to NPCs at other locations who then return
        # empty/refusal intentions).
        picks = _filter_picks_for_dispatch(
            ckpt, scene_id, result.agent_responder_picks,
        )
        if not picks:
            # Router signaled continue but gave no pick — treat as
            # cascade_exhausted and end beat.
            return await _end_beat(
                ckpt, dispatcher, scene_id,
                ended_reason="cascade_exhausted",
                events_closed=events_closed,
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
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
                event_actor_ids=event_actor_ids,
                acting_player_id=actor_id,
                acting_player_input=intention,
            )
        current_actor = next_actor
        current_intention = next_intention


async def _end_beat(
    ckpt: CheckpointFile,
    dispatcher: Dispatcher,
    scene_id: str,
    ended_reason: str,
    events_closed: int,
    event_actor_ids: list[str],
    *,
    release_slots: bool = True,
    force_partial: bool = False,
    render_only: set[str] | None = None,
    acting_player_id: str | None = None,
    acting_player_input: str = "",
) -> BeatResult:
    """Compose per-human renders, flush buffers, (optionally) release
    scene slots.

    v11-r6a params:
    - `release_slots=False`: used by the Cat II-open render path so the
      pins we JUST created stay live until responders /act. The normal
      end-of-beat path leaves this True.
    - `force_partial=True`: every rendered POV gets partial_mode=True
      regardless of the dispatcher's default slot-scan detection. Used
      by the Cat II-open render path so the initiator (who isn't
      pinned) also sees the cliffhanger.
    - `render_only=set(...)`: only render POVs whose character_id is in
      this set. Used by the Cat II-open render path to fan-out only to
      pinned humans + initiator-if-human, not to other in-scene humans
      who aren't participants in the open event.

    v11-r7j params:
    - `acting_player_id` + `acting_player_input`: the actual /act'd
      player and their verbatim utterance. Threaded into the matching
      POV's narrator call so the engine-built TranscriptEntry carries
      the real input. Other POVs (incidental humans in the scene) get
      `user_input=""`. Pre-r7j the LLM owned the transcript entry and
      no real input ever made it that far.
    """
    renders: dict[str, str] = {}
    transcript_entries: dict[str, TranscriptEntry] = {}
    candidates = humans_in_scene(ckpt, scene_id)
    if render_only is not None:
        candidates = [h for h in candidates if h in render_only]
    partial_override: bool | None = True if force_partial else None

    # v11-r6c: fan out narrator calls in parallel. Independent POVs; no
    # shared state to mutate mid-call. Trims a 3-human scene from 3×
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
        release_scene_slots(ckpt, scene_id)
    logger.info(
        "Beat closed in %s: events=%d reason=%s renders=%d "
        "release_slots=%s force_partial=%s",
        scene_id, events_closed, ended_reason, len(renders),
        release_slots, force_partial,
    )
    return BeatResult(
        renders=renders, events_closed=events_closed, ended_reason=ended_reason,
        transcript_entries=transcript_entries,
        event_actor_ids=event_actor_ids,
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
