from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.events import visible_fact_texts
from app.schemas.state import RenderBufferEntry

AGENT_FIRST_MEETING_CAP = 4
NARRATOR_FIRST_MEETING_CAP = 6

_LOADOUT_TAG_RE = re.compile(r"^\[loadout\s+[—–-]\s*([^\]]+)\]", re.IGNORECASE)
_SPEECH_VERB_RE = (
    r"says|said|asks|asked|replies|replied|whispers|whispered|"
    r"shouts|shouted|speaks|spoke|calls|called|murmurs|murmured|"
    r"mutters|muttered|tells|told"
)


@dataclass(frozen=True)
class VisualIntroduction:
    character_id: str
    name: str
    default_loadout: str
    public_context: str


@dataclass(frozen=True)
class VisualIntroductionPlan:
    loadouts: list[VisualIntroduction]
    mark_character_ids: list[str]


def _character_status_value(character: CharacterRecord) -> str:
    status = getattr(character, "status", "")
    return str(getattr(status, "value", status) or "")


def _tier_rank(character: CharacterRecord) -> int:
    tier = str(getattr(character.agent_tier, "value", character.agent_tier))
    if tier == "premium":
        return 0
    if tier == "standard":
        return 1
    if tier == "utility":
        return 2
    return 3


def _default_loadout(character: CharacterRecord) -> str:
    visuals = getattr(character, "visuals", None)
    return str(getattr(visuals, "default_loadout", "") or "").strip()


def _public_context(character: CharacterRecord) -> str:
    descriptions = getattr(character, "descriptions", None)
    return str(getattr(descriptions, "public", "") or "").strip()


def _is_redundant_context(loadout: str, public_context: str) -> bool:
    left = " ".join((loadout or "").casefold().split())
    right = " ".join((public_context or "").casefold().split())
    if not left or not right:
        return False
    return left in right or right in left


def _active_roster_characters(ckpt: CheckpointFile) -> list[CharacterRecord]:
    return [
        character for character in ckpt.characters
        if _character_status_value(character) != "culled"
    ]


def _character_maps(
    ckpt: CheckpointFile,
) -> tuple[dict[str, CharacterRecord], dict[str, str]]:
    by_id = {
        character.character_id: character
        for character in _active_roster_characters(ckpt)
    }
    by_label: dict[str, str] = {}
    for character in by_id.values():
        if character.character_id:
            by_label[character.character_id.casefold()] = character.character_id
        if character.name:
            by_label[character.name.casefold()] = character.character_id
    return by_id, by_label


def _mentioned_character_ids(
    ckpt: CheckpointFile,
    visible_texts: Iterable[str],
) -> set[str]:
    text = "\n".join(t for t in visible_texts if t).strip()
    if not text:
        return set()
    mentioned: set[str] = set()
    for character in _active_roster_characters(ckpt):
        probes = [character.character_id, character.name]
        for probe in probes:
            if not probe:
                continue
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(probe)}(?![A-Za-z0-9_])",
                text,
                flags=re.IGNORECASE,
            ):
                mentioned.add(character.character_id)
                break
    return mentioned


def _speaking_character_ids(
    ckpt: CheckpointFile,
    visible_texts: Iterable[str],
) -> set[str]:
    texts = [t for t in visible_texts if t and t.strip()]
    if not texts:
        return set()
    speaking: set[str] = set()
    for character in _active_roster_characters(ckpt):
        probes = [character.character_id, character.name]
        for probe in probes:
            if not probe:
                continue
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(probe)}(?![A-Za-z0-9_])"
                rf"(?=[^.?!\n]{{0,120}}\b(?:{_SPEECH_VERB_RE})\b)",
                re.IGNORECASE,
            )
            if any(pattern.search(text) for text in texts):
                speaking.add(character.character_id)
                break
    return speaking


def _loadout_tag_character_ids(
    ckpt: CheckpointFile,
    visible_texts: Iterable[str],
) -> set[str]:
    _by_id, by_label = _character_maps(ckpt)
    tagged: set[str] = set()
    for text in visible_texts:
        match = _LOADOUT_TAG_RE.match((text or "").strip())
        if not match:
            continue
        label = match.group(1).strip().casefold()
        character_id = by_label.get(label)
        if character_id:
            tagged.add(character_id)
    return tagged


def _introduced_ids(ckpt: CheckpointFile, viewer_id: str) -> set[str]:
    return set(ckpt.session.visual_introductions.get(viewer_id, []))


def mark_visual_introductions(
    ckpt: CheckpointFile,
    viewer_id: str,
    character_ids: Iterable[str],
) -> None:
    if not viewer_id:
        return
    bucket = ckpt.session.visual_introductions.setdefault(viewer_id, [])
    seen = set(bucket)
    for character_id in character_ids:
        if not character_id or character_id == viewer_id or character_id in seen:
            continue
        bucket.append(character_id)
        seen.add(character_id)


def forget_visual_introductions_for_character(
    ckpt: CheckpointFile,
    character_id: str,
) -> None:
    """Clear first-meeting ledger entries tied to a replaced character id."""
    character_id = (character_id or "").strip()
    if not character_id:
        return

    ckpt.session.visual_introductions.pop(character_id, None)
    for viewer_id, character_ids in list(ckpt.session.visual_introductions.items()):
        filtered = [cid for cid in character_ids if cid != character_id]
        if filtered:
            ckpt.session.visual_introductions[viewer_id] = filtered
        else:
            ckpt.session.visual_introductions.pop(viewer_id, None)


def _plan_visual_introductions(
    ckpt: CheckpointFile,
    *,
    viewer_id: str,
    visible_texts: Iterable[str],
    candidate_ids: Iterable[str] | None = None,
    priority_target_ids: Iterable[str] = (),
    max_loadouts: int,
) -> VisualIntroductionPlan:
    texts = [text for text in visible_texts if text and text.strip()]
    if not viewer_id or max_loadouts <= 0 or not texts:
        return VisualIntroductionPlan(loadouts=[], mark_character_ids=[])

    by_id, _by_label = _character_maps(ckpt)
    introduced = _introduced_ids(ckpt, viewer_id)
    tagged_ids = _loadout_tag_character_ids(ckpt, texts)
    priority_ids = [cid for cid in priority_target_ids if cid]
    priority_index = {
        cid: index for index, cid in enumerate(priority_ids)
    }
    candidate_set = (
        set(candidate_ids)
        if candidate_ids is not None
        else _mentioned_character_ids(ckpt, texts)
    ) | tagged_ids
    candidate_set.discard(viewer_id)

    mark_ids: list[str] = []
    loadout_candidates: list[tuple[tuple[int, int, int], VisualIntroduction]] = []
    roster_index = {
        character.character_id: index
        for index, character in enumerate(ckpt.characters)
    }
    for character_id in candidate_set:
        character = by_id.get(character_id)
        if character is None or character_id in introduced:
            continue
        if character_id in tagged_ids:
            mark_ids.append(character_id)
            continue
        loadout = _default_loadout(character)
        public_context = _public_context(character)
        if _is_redundant_context(loadout, public_context):
            public_context = ""
        if not loadout and not public_context:
            continue
        sort_key = (
            priority_index.get(character_id, 10_000),
            _tier_rank(character),
            roster_index.get(character_id, 10_000),
        )
        loadout_candidates.append((
            sort_key,
            VisualIntroduction(
                character_id=character_id,
                name=character.name or character_id,
                default_loadout=loadout,
                public_context=public_context,
            ),
        ))

    loadouts = [
        introduction
        for _key, introduction in sorted(loadout_candidates, key=lambda item: item[0])
    ][:max_loadouts]
    mark_ids.extend(introduction.character_id for introduction in loadouts)
    return VisualIntroductionPlan(
        loadouts=loadouts,
        mark_character_ids=list(dict.fromkeys(mark_ids)),
    )


def plan_event_visual_introductions(
    ckpt: CheckpointFile,
    *,
    viewer_id: str,
    event: EventRouterOutput,
    observation_level: str,
    priority_target_ids: Iterable[str] = (),
    max_loadouts: int = AGENT_FIRST_MEETING_CAP,
) -> VisualIntroductionPlan:
    if observation_level != "direct":
        return VisualIntroductionPlan(loadouts=[], mark_character_ids=[])
    texts = visible_fact_texts(
        event.canonical_event.observable_facts,
        viewer_id,
        include_all_observers=True,
    )
    return _plan_visual_introductions(
        ckpt,
        viewer_id=viewer_id,
        visible_texts=texts,
        candidate_ids=_speaking_character_ids(ckpt, texts),
        priority_target_ids=priority_target_ids,
        max_loadouts=max_loadouts,
    )


def plan_render_visual_introductions(
    ckpt: CheckpointFile,
    *,
    viewer_id: str,
    resolved: list[tuple[RenderBufferEntry, EventRouterOutput]],
    max_loadouts: int = NARRATOR_FIRST_MEETING_CAP,
) -> VisualIntroductionPlan:
    visible_texts: list[str] = []
    for entry, event in resolved:
        if entry.observation_level != "direct":
            continue
        visible_texts.extend(
            visible_fact_texts(
                event.canonical_event.observable_facts,
                viewer_id,
                include_all_observers=True,
            )
        )
    return _plan_visual_introductions(
        ckpt,
        viewer_id=viewer_id,
        visible_texts=visible_texts,
        max_loadouts=max_loadouts,
    )


def format_visual_introductions(
    introductions: Iterable[VisualIntroduction],
) -> str:
    entries = list(introductions)
    if not entries:
        return ""
    lines = ["Newly introduced character context:"]
    for introduction in entries:
        parts: list[str] = []
        if introduction.default_loadout:
            parts.append(f"first visible impression: {introduction.default_loadout}")
        if introduction.public_context:
            parts.append(f"player-safe context: {introduction.public_context}")
        if parts:
            lines.append(f"- {introduction.name}: {' '.join(parts)}")
    return "\n".join(lines)
