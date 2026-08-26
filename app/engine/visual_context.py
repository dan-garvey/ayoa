from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.engine.text_safety import strip_terminal_control
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import redact_imported_content_metadata_text
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
_QUOTED_SPAN_RE = re.compile(
    r'"[^"\n]*"|“[^”\n]*”|\'[^\'\n]*\'|‘[^’\n]*’'
)
_MEDIATED_CHANNEL_NOUNS = (
    r"radio|intercom|telephone|phone|voicemail|recording|broadcast|screen|"
    r"monitor|camera|video|feed|speaker|projection|hologram"
)
_MEDIATED_VISUAL_NOUNS = (
    r"screen|monitor|camera|video|feed|projection|hologram|recording|"
    r"image|portrait|photograph|photo|likeness"
)
_MEDIATED_SUBJECT_PREFIX_RE = re.compile(
    rf"\b(?:{_MEDIATED_CHANNEL_NOUNS}|image|portrait|photograph|photo|"
    rf"likeness|voice|message|mail|letter|note|report|rumou?r|news)\b"
    rf"[^.?!;\n]{{0,96}}\b(?:shows?|displays?|depicts?|captures?|carries|"
    rf"relays?|plays?|crackles|reports?|says?|announces?|mentions?|from|"
    rf"about|of)\b[^.?!;\n]*$|"
    rf"\b(?:over|through|via|on|from)\s+(?:a|an|the)?\s*"
    rf"(?:{_MEDIATED_CHANNEL_NOUNS})\b[^.?!;\n]*$|"
    rf"\b(?:on|in)\s+(?:a|an|the)?\s*(?:{_MEDIATED_VISUAL_NOUNS})\b"
    rf"[^.?!;\n]*$",
    re.IGNORECASE,
)
_MEDIATED_CHANNEL_BINDING_RE = re.compile(
    rf"\b(?:over|through|via|on|from)\s+(?:a|an|the)?\s*"
    rf"(?:{_MEDIATED_CHANNEL_NOUNS})\b|"
    rf"\b(?:on|in)\s+(?:a|an|the)?\s*(?:{_MEDIATED_VISUAL_NOUNS})\b",
    re.IGNORECASE,
)
_PRIOR_REPORTING_RE = re.compile(
    rf"\b(?:{_SPEECH_VERB_RE})\b|\baccording\s+to\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?:[,;]\s*|\s+)\b(?:while|whereas|although|though|meanwhile|then|"
    r"afterwards?|but|as)\b(?:\s+|,\s*)",
    re.IGNORECASE,
)
_PHYSICAL_PRESENCE_VERB_RE = re.compile(
    rf"\b(?:(?:{_SPEECH_VERB_RE})|"
    r"adjust(?:s|ed|ing)?|approach(?:es|ed|ing)?|arriv(?:e|es|ed|ing)|"
    r"bow(?:s|ed|ing)?|come(?:s)?|came|cross(?:es|ed|ing)?|"
    r"crouch(?:es|ed|ing)?|depart(?:s|ed|ing)?|duck(?:s|ed|ing)?|"
    r"emerg(?:e|es|ed|ing)|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|"
    r"face(?:s|d|ing)?|follow(?:s|ed|ing)?|gesture(?:s|d|ing)?|"
    r"grip(?:s|ped|ping)?|hold(?:s|ing)?|held|kneel(?:s|ed|ing)?|"
    r"lean(?:s|ed|ing)?|look(?:s|ed|ing)?|move(?:s|d|ing)?|"
    r"nod(?:s|ded|ding)?|pass(?:es|ed|ing)?|reach(?:es|ed|ing)?|"
    r"rise(?:s|n)?|rose|run(?:s|ning)?|ran|sit(?:s|ting)?|sat|"
    r"smil(?:e|es|ed|ing)|stand(?:s|ing)?|stood|step(?:s|ped|ping)?|"
    r"stop(?:s|ped|ping)?|turn(?:s|ed|ing)?|wait(?:s|ed|ing)?|"
    r"walk(?:s|ed|ing)?|wave(?:s|d|ing)?|wear(?:s|ing)?|wore"
    r")\b",
    re.IGNORECASE,
)
_COPRESENCE_PREDICATE_RE = re.compile(
    r"\b(?:am|is|are|was|were|remain(?:s|ed|ing)?)\b"
    r"(?:\s+(?:still|already|now|right|just)){0,2}\s+"
    r"(?:here|there|present|visible|nearby|inside|outside|at|in|by|beside|"
    r"near|behind|before|among|with)\b",
    re.IGNORECASE,
)
_NON_CURRENT_PRESENCE_RE = re.compile(
    r"\b(?:will|would|shall|should|could|may|might|plans?|planned|"
    r"intends?|intended|expects?|expected|scheduled|due)\b",
    re.IGNORECASE,
)
_FUTURE_TIME_RE = re.compile(
    r"\b(?:later|tomorrow|eventually|next\s+(?:day|week|month|year))\b",
    re.IGNORECASE,
)
_COPRESENCE_RELATION_RE = re.compile(
    r"\b(?:alongside|beside|near|with|among|by|behind|before|next\s+to)\b",
    re.IGNORECASE,
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


@dataclass(frozen=True)
class _PresenceEvidence:
    character_id: str
    evidence_start: int
    clause_start: int
    clause_end: int


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
    return _safe_visual_context(
        getattr(visuals, "default_loadout", "") or ""
    )


def _public_context(character: CharacterRecord) -> str:
    descriptions = getattr(character, "descriptions", None)
    return _safe_visual_context(getattr(descriptions, "public", "") or "")


def _safe_visual_context(value: object) -> str:
    return redact_imported_content_metadata_text(
        strip_terminal_control(str(value or ""))
    ).strip()


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


def _presence_gap_is_bounded_subject_context(
    ckpt: CheckpointFile,
    *,
    character_id: str,
    gap: str,
) -> bool:
    """Accept only modifiers or an explicit coordinated roster subject list."""

    cleaned = gap
    contains_other_character = False
    for character in _active_roster_characters(ckpt):
        for probe in (character.character_id, character.name):
            if not probe:
                continue
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(probe)}(?![A-Za-z0-9_])",
                re.IGNORECASE,
            )
            cleaned, count = pattern.subn(" ", cleaned)
            if count and character.character_id != character_id:
                contains_other_character = True

    has_coordination = bool(re.search(r"\b(?:and|or)\b", cleaned, re.IGNORECASE))
    if contains_other_character != has_coordination:
        return False
    cleaned = re.sub(
        r"\b(?:and|or|both|all|each|together|is|are|was|were|has|have|had|"
        r"begins?|began|starts?|started|keeps?|kept|to|just|now|still|already)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b[A-Za-z]+ly\b", " ", cleaned)
    return not re.search(r"[A-Za-z0-9_]", cleaned)


def _clause_bounds(sentence: str, position: int) -> tuple[int, int]:
    start = 0
    end = len(sentence)
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(sentence):
        if boundary.end() <= position:
            start = boundary.end()
            continue
        if boundary.start() >= position:
            end = boundary.start()
            break
    return start, end


def _presence_evidence_for_match(
    ckpt: CheckpointFile,
    *,
    sentence: str,
    character_id: str,
    subject_match: re.Match[str],
) -> _PresenceEvidence | None:
    if re.match(
        r"\s*[’']s\b",
        sentence[subject_match.end():],
        re.IGNORECASE,
    ):
        return None

    clause_start, clause_end = _clause_bounds(sentence, subject_match.start())
    bounded_end = min(clause_end, subject_match.end() + 180)
    suffix = sentence[subject_match.end():bounded_end]
    evidence_matches = [
        evidence
        for matcher in (
            _PHYSICAL_PRESENCE_VERB_RE,
            _COPRESENCE_PREDICATE_RE,
        )
        if (evidence := matcher.search(suffix)) is not None
    ]
    if not evidence_matches:
        return None
    evidence_match = min(evidence_matches, key=lambda item: item.start())
    gap = suffix[:evidence_match.start()]
    if not _presence_gap_is_bounded_subject_context(
        ckpt,
        character_id=character_id,
        gap=gap,
    ):
        return None
    if _NON_CURRENT_PRESENCE_RE.search(gap):
        return None

    subject_prefix = sentence[clause_start:subject_match.start()]
    if _MEDIATED_SUBJECT_PREFIX_RE.search(subject_prefix):
        return None
    if _PRIOR_REPORTING_RE.search(subject_prefix):
        return None

    evidence_start = subject_match.end() + evidence_match.start()
    evidence_end = subject_match.end() + evidence_match.end()
    if _FUTURE_TIME_RE.search(
        sentence[evidence_end:min(clause_end, evidence_end + 32)]
    ):
        return None
    if _MEDIATED_CHANNEL_BINDING_RE.search(
        sentence[subject_match.end():clause_end]
    ):
        return None
    return _PresenceEvidence(
        character_id=character_id,
        evidence_start=evidence_start,
        clause_start=clause_start,
        clause_end=clause_end,
    )


def _copresent_object_ids(
    ckpt: CheckpointFile,
    *,
    sentence: str,
    evidence_records: Iterable[_PresenceEvidence],
) -> set[str]:
    object_ids: set[str] = set()
    for evidence in evidence_records:
        relation_end = min(
            evidence.clause_end,
            evidence.evidence_start + 120,
        )
        for relation in _COPRESENCE_RELATION_RE.finditer(
            sentence,
            evidence.evidence_start,
            relation_end,
        ):
            for character in _active_roster_characters(ckpt):
                if character.character_id == evidence.character_id:
                    continue
                seen_probes: set[str] = set()
                for probe in (character.character_id, character.name):
                    if not probe or probe.casefold() in seen_probes:
                        continue
                    seen_probes.add(probe.casefold())
                    pattern = re.compile(
                        rf"(?<![A-Za-z0-9_]){re.escape(probe)}"
                        rf"(?![A-Za-z0-9_])",
                        re.IGNORECASE,
                    )
                    for match in pattern.finditer(
                        sentence,
                        relation.end(),
                        evidence.clause_end,
                    ):
                        if re.match(
                            r"\s*[’']s\b",
                            sentence[match.end():],
                            re.IGNORECASE,
                        ):
                            continue
                        if not _presence_gap_is_bounded_subject_context(
                            ckpt,
                            character_id=character.character_id,
                            gap=sentence[relation.end():match.start()],
                        ):
                            continue
                        if _MEDIATED_SUBJECT_PREFIX_RE.search(
                            sentence[evidence.clause_start:match.start()]
                        ):
                            continue
                        object_ids.add(character.character_id)
                        break
                    if character.character_id in object_ids:
                        break
    return object_ids


def _physically_present_character_ids(
    ckpt: CheckpointFile,
    visible_texts: Iterable[str],
) -> set[str]:
    texts = [text for text in visible_texts if text and text.strip()]
    if not texts:
        return set()
    physical: set[str] = set()
    roster = _active_roster_characters(ckpt)
    for text in texts:
        unquoted = _QUOTED_SPAN_RE.sub(" ", text)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", unquoted):
            evidence_records: list[_PresenceEvidence] = []
            for character in roster:
                seen_probes: set[str] = set()
                found_for_sentence = False
                for probe in (character.character_id, character.name):
                    if not probe or probe.casefold() in seen_probes:
                        continue
                    seen_probes.add(probe.casefold())
                    probe_pattern = re.compile(
                        rf"(?<![A-Za-z0-9_]){re.escape(probe)}"
                        rf"(?![A-Za-z0-9_])",
                        re.IGNORECASE,
                    )
                    for match in probe_pattern.finditer(sentence):
                        evidence = _presence_evidence_for_match(
                            ckpt,
                            sentence=sentence,
                            character_id=character.character_id,
                            subject_match=match,
                        )
                        if evidence is not None:
                            evidence_records.append(evidence)
                            physical.add(character.character_id)
                            found_for_sentence = True
                            break
                    if found_for_sentence:
                        break
            physical.update(_copresent_object_ids(
                ckpt,
                sentence=sentence,
                evidence_records=evidence_records,
            ))
    return physical


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
    candidates = [
        character_id
        for character_id in character_ids
        if character_id and character_id != viewer_id
    ]
    if not candidates:
        return
    bucket = ckpt.session.visual_introductions.setdefault(viewer_id, [])
    seen = set(bucket)
    for character_id in candidates:
        if character_id in seen:
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
    include_public_context: bool = True,
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
        else _physically_present_character_ids(ckpt, texts)
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
        public_context = (
            _public_context(character) if include_public_context else ""
        )
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
        candidate_ids=_physically_present_character_ids(ckpt, texts),
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
        include_public_context=False,
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


def format_narrator_visual_introductions(
    introductions: Iterable[VisualIntroduction],
) -> str:
    """Format first-look exterior vocabulary for either narrator mode.

    Narrators need a bounded visual impression, not the broader public
    biography used by character agents.  Labels bind each exterior to source
    context but do not themselves grant the viewpoint knowledge of a name.
    """

    lines: list[str] = []
    for introduction in introductions:
        if introduction.default_loadout:
            lines.append(
                f"- {introduction.name}: visible exterior: "
                f"{introduction.default_loadout}"
            )
    return "\n".join(lines)
