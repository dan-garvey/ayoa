from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.engine.text_safety import strip_terminal_control
from app.engine.one_star_visuals import first_look_override_for_viewer
from app.schemas.characters import CharacterRecord, is_player_authored_slot
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import redact_imported_content_metadata_text
from app.schemas.delivery import NarratorEventRef
from app.schemas.event_router import CanonicalEventRecord
from app.schemas.events import visible_fact_texts

AGENT_FIRST_MEETING_CAP = 4
NARRATOR_FIRST_MEETING_CAP = 6

_LOADOUT_TAG_RE = re.compile(r"^\[loadout\s+[—–-]\s*([^\]]+)\]", re.IGNORECASE)
_SPEECH_VERB_RE = (
    r"says|said|asks|asked|replies|replied|whispers|whispered|"
    r"shouts|shouted|speaks|spoke|calls|called|murmurs|murmured|"
    r"mutters|muttered|tells|told|reports|reported|reporting|"
    r"announces|announced|announcing|mentions|mentioned|mentioning|"
    r"states|stated|stating|describes|described|describing"
)
_QUOTED_SPAN_RE = re.compile(r'"[^"\n]*"|“[^”\n]*”|\'[^\'\n]*\'|‘[^’\n]*’')
_EMBODIED_POSSESSIVE_RE = re.compile(
    r"\s*[’']s\s+(?:smile|eyes?|face|head|brows?|mouth|ears?|"
    r"hands?|arms?|shoulders?|legs?|feet|wings?|tail|stance|posture)\b",
    re.IGNORECASE,
)
_MEDIATED_CHANNEL_NOUNS = (
    r"radio|intercom|telephone|phone|voicemail|recording|broadcast|screen|"
    r"monitor|camera|video|feed|speaker|projection|hologram"
)
_MEDIATED_VISUAL_NOUNS = (
    r"screen|monitor|camera|video|feed|projection|hologram|recording|"
    r"image|portrait|photograph|photo|likeness|sketch|drawing|painting|"
    r"illustration|poster|mural|statue|sculpture|scrying\s+mirror"
)
_TEXTUAL_CHANNEL_NOUNS = (
    r"voice\s+message|message|e-?mail|mail|letter|note|"
    r"text(?:\s+message)?|writing|written\s+report"
)
_MEDIATED_SUBJECT_PREFIX_RE = re.compile(
    rf"\b(?:{_MEDIATED_CHANNEL_NOUNS}|{_MEDIATED_VISUAL_NOUNS}|voice|"
    rf"message|mail|letter|note|report|rumou?r|news)\b"
    rf"[^.?!;\n]{{0,96}}\b(?:shows?|displays?|depicts?|portrays?|"
    rf"represents?|renders?|captures?|carries|relays?|plays?|crackles|"
    rf"reports?|says?|announces?|mentions?|from|about|of)\b[^.?!;\n]*$|"
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
_MEDIATED_REPORT_BINDING_RE = re.compile(
    rf"\b(?:{_SPEECH_VERB_RE})\b[^.?!;\n]{{0,48}}\b(?:"
    rf"(?:in|via|through|on|from)\s+(?:a|an|the)?\s*"
    rf"(?:{_TEXTUAL_CHANNEL_NOUNS})|"
    rf"by\s+(?:{_TEXTUAL_CHANNEL_NOUNS})"
    rf")\b",
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
_PREDICATE_COORDINATOR_RE = re.compile(r"\b(?:and|or)\b", re.IGNORECASE)
_PHYSICAL_PRESENCE_VERB_RE = re.compile(
    rf"\b(?:(?:{_SPEECH_VERB_RE})|"
    r"adjust(?:s|ed|ing)?|approach(?:es|ed|ing)?|arriv(?:e|es|ed|ing)|"
    r"bow(?:s|ed|ing)?|come(?:s)?|came|cross(?:es|ed|ing)?|"
    r"crouch(?:es|ed|ing)?|depart(?:s|ed|ing)?|duck(?:s|ed|ing)?|"
    r"emerg(?:e|es|ed|ing)|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|"
    r"face(?:s|d|ing)?|follow(?:s|ed|ing)?|gesture(?:s|d|ing)?|"
    r"flit(?:s|ted|ting)?|"
    r"grip(?:s|ped|ping)?|hold(?:s|ing)?|held|kneel(?:s|ed|ing)?|"
    r"lean(?:s|ed|ing)?|look(?:s|ed|ing)?|move(?:s|d|ing)?|"
    r"nod(?:s|ded|ding)?|pass(?:es|ed|ing)?|reach(?:es|ed|ing)?|"
    r"rise(?:s|n)?|rose|run(?:s|ning)?|ran|sit(?:s|ting)?|sat|"
    r"shift(?:s|ed|ing)?|smil(?:e|es|ed|ing)|stand(?:s|ing)?|stood|"
    r"step(?:s|ped|ping)?|stop(?:s|ped|ping)?|tilt(?:s|ed|ing)?|"
    r"turn(?:s|ed|ing)?|wait(?:s|ed|ing)?|"
    r"walk(?:s|ed|ing)?|wave(?:s|d|ing)?|wear(?:s|ing)?|wore"
    r")\b",
    re.IGNORECASE,
)
_GENERIC_VISIBLE_ACTION_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z'’\-]{1,}(?:s|ed|ing))\b",
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
_ABSENT_PRESENCE_RE = re.compile(
    r"^\s*(?:am|is|are|was|were|remain(?:s|ed|ing)?)\s+"
    r"(?:still\s+)?(?:absent|missing|gone|nowhere|not\s+(?:here|there|"
    r"present|visible|nearby|inside|outside))\b",
    re.IGNORECASE,
)
_FUTURE_TIME_RE = re.compile(
    r"\b(?:later|tomorrow|eventually|next\s+(?:day|week|month|year))\b",
    re.IGNORECASE,
)
_COPRESENCE_RELATION_RE = re.compile(
    r"\b(?:alongside|beside|near|with|among|by|behind|before|closer\s+to|"
    r"next\s+to)\b",
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
    return _safe_visual_context(getattr(visuals, "default_loadout", "") or "")


def _public_context(character: CharacterRecord) -> str:
    public_sheet = getattr(character, "public_sheet", None)
    return _safe_visual_context(
        getattr(public_sheet, "public_context", "") or ""
    )


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
        character
        for character in ckpt.characters
        if _character_status_value(character) != "culled"
        and not (
            is_player_authored_slot(character)
            and character.character_id not in (ckpt.session.character_bindings or {})
            and character.character_id != ckpt.session.player_character_id
        )
    ]


def _character_presence_probes(
    ckpt: CheckpointFile,
    character: CharacterRecord,
) -> tuple[str, ...]:
    """Return exact identity probes plus an unambiguous spoken first name."""

    probes = [character.character_id, character.name]
    name_parts = (character.name or "").split()
    if len(name_parts) > 1:
        first_name = name_parts[0]
        first_folded = first_name.casefold()
        if (
            first_folded not in {"a", "an", "the", "lady", "lord", "sir", "dame"}
            and re.fullmatch(r"[A-Za-z][A-Za-z'’\-]*", first_name)
            and sum(
                1
                for candidate in _active_roster_characters(ckpt)
                if (candidate.name or "").split()
                and candidate.name.split()[0].casefold() == first_folded
            )
            == 1
        ):
            probes.append(first_name)
    return tuple(
        sorted(
            dict.fromkeys(probe for probe in probes if probe),
            key=len,
            reverse=True,
        )
    )


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
        for probe in _character_presence_probes(ckpt, character):
            if not probe:
                continue
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(probe)}(?![A-Za-z0-9_])",
                re.IGNORECASE,
            )
            cleaned, count = pattern.subn(" ", cleaned)
            if count and character.character_id != character_id:
                contains_other_character = True

    has_coordination = bool(
        re.search(
            r",|\b(?:and|or)\b",
            cleaned,
            re.IGNORECASE,
        )
    )
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


def _subject_scope_start(
    ckpt: CheckpointFile,
    sentence: str,
    *,
    clause_start: int,
    subject_start: int,
) -> int:
    """Start at an independent coordinated subject, not an earlier speaker."""

    scope_start = clause_start
    for coordinator in _PREDICATE_COORDINATOR_RE.finditer(
        sentence,
        clause_start,
        subject_start,
    ):
        prior_predicate = sentence[scope_start : coordinator.start()]
        if _PHYSICAL_PRESENCE_VERB_RE.search(
            prior_predicate
        ) or _COPRESENCE_PREDICATE_RE.search(prior_predicate):
            report = _PRIOR_REPORTING_RE.search(prior_predicate)
            reported_text = (
                prior_predicate[report.end() :] if report is not None else ""
            )
            reported_roster_subject = any(
                re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(probe)}"
                    rf"(?![A-Za-z0-9_])",
                    reported_text,
                    re.IGNORECASE,
                )
                for character in _active_roster_characters(ckpt)
                for probe in _character_presence_probes(ckpt, character)
                if probe
            )
            if (
                _MEDIATED_SUBJECT_PREFIX_RE.search(prior_predicate)
                or reported_roster_subject
            ):
                continue
            scope_start = coordinator.end()
    return scope_start


def _presence_evidence_matches(
    suffix: str,
    *,
    include_generic_actions: bool,
) -> list[re.Match[str]]:
    matches: dict[tuple[int, int], re.Match[str]] = {}
    matchers = [
        _PHYSICAL_PRESENCE_VERB_RE,
        _COPRESENCE_PREDICATE_RE,
    ]
    if include_generic_actions:
        matchers.append(_GENERIC_VISIBLE_ACTION_RE)
    for matcher in matchers:
        for match in matcher.finditer(suffix):
            matches.setdefault((match.start(), match.end()), match)
    return sorted(matches.values(), key=lambda item: (item.start(), item.end()))


def _presence_evidence_for_match(
    ckpt: CheckpointFile,
    *,
    sentence: str,
    character_id: str,
    subject_match: re.Match[str],
    include_generic_actions: bool,
) -> _PresenceEvidence | None:
    possessive = re.match(
        r"\s*[’']s\b",
        sentence[subject_match.end() :],
        re.IGNORECASE,
    )
    embodied_possessive = _EMBODIED_POSSESSIVE_RE.match(sentence[subject_match.end() :])
    if possessive is not None and embodied_possessive is None:
        return None
    subject_end = subject_match.end() + (
        embodied_possessive.end() if embodied_possessive is not None else 0
    )

    clause_start, clause_end = _clause_bounds(sentence, subject_match.start())
    subject_scope_start = _subject_scope_start(
        ckpt,
        sentence,
        clause_start=clause_start,
        subject_start=subject_match.start(),
    )
    bounded_end = min(clause_end, subject_end + 180)
    suffix = sentence[subject_end:bounded_end]
    if _ABSENT_PRESENCE_RE.match(suffix):
        return None
    evidence_matches = _presence_evidence_matches(
        suffix,
        include_generic_actions=include_generic_actions,
    )
    if not evidence_matches:
        return None

    subject_prefix = sentence[subject_scope_start : subject_match.start()]
    if _MEDIATED_SUBJECT_PREFIX_RE.search(subject_prefix):
        return None
    if _PRIOR_REPORTING_RE.search(subject_prefix):
        return None

    absolute_matches = [
        (
            subject_end + match.start(),
            subject_end + match.end(),
        )
        for match in evidence_matches
    ]
    for index, (evidence_start, evidence_end) in enumerate(absolute_matches):
        predicate_start = subject_end
        if index:
            previous_end = absolute_matches[index - 1][1]
            coordinators = list(
                _PREDICATE_COORDINATOR_RE.finditer(
                    sentence,
                    previous_end,
                    evidence_start,
                )
            )
            if coordinators:
                predicate_start = coordinators[-1].end()

        predicate_end = clause_end
        if index + 1 < len(absolute_matches):
            next_start = absolute_matches[index + 1][0]
            next_coordinator = _PREDICATE_COORDINATOR_RE.search(
                sentence,
                evidence_end,
                next_start,
            )
            if next_coordinator:
                predicate_end = next_coordinator.start()

        gap = sentence[predicate_start:evidence_start]
        if not _presence_gap_is_bounded_subject_context(
            ckpt,
            character_id=character_id,
            gap=gap,
        ):
            continue
        predicate_lead = sentence[subject_end:evidence_start]
        if _NON_CURRENT_PRESENCE_RE.search(
            predicate_lead
        ) or _NON_CURRENT_PRESENCE_RE.search(sentence[evidence_start:evidence_end]):
            continue
        future_tail = sentence[evidence_end : min(predicate_end, evidence_end + 32)]
        if (
            _FUTURE_TIME_RE.search(
                sentence[subject_scope_start : subject_match.start()]
            )
            or _FUTURE_TIME_RE.search(predicate_lead)
            or _FUTURE_TIME_RE.search(future_tail)
        ):
            continue
        predicate = sentence[predicate_start:predicate_end]
        if _MEDIATED_CHANNEL_BINDING_RE.search(
            predicate
        ) or _MEDIATED_REPORT_BINDING_RE.search(predicate):
            continue
        return _PresenceEvidence(
            character_id=character_id,
            evidence_start=evidence_start,
            clause_start=predicate_start,
            clause_end=predicate_end,
        )
    return None


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
                for probe in _character_presence_probes(ckpt, character):
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
                            sentence[match.end() :],
                            re.IGNORECASE,
                        ):
                            continue
                        if not _presence_gap_is_bounded_subject_context(
                            ckpt,
                            character_id=character.character_id,
                            gap=sentence[relation.end() : match.start()],
                        ):
                            continue
                        if _MEDIATED_SUBJECT_PREFIX_RE.search(
                            sentence[evidence.clause_start : match.start()]
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
    *,
    include_generic_actions: bool = False,
) -> set[str]:
    texts = [text for text in visible_texts if text and text.strip()]
    if not texts:
        return set()
    physical: set[str] = set()
    roster = _active_roster_characters(ckpt)
    for text in texts:
        unquoted = _QUOTED_SPAN_RE.sub(
            lambda match: (
                match.group(0)[-2]
                if len(match.group(0)) >= 2 and match.group(0)[-2] in ".!?…"
                else " "
            ),
            text,
        )
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", unquoted):
            evidence_records: list[_PresenceEvidence] = []
            for character in roster:
                seen_probes: set[str] = set()
                found_for_sentence = False
                for probe in _character_presence_probes(ckpt, character):
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
                            include_generic_actions=include_generic_actions,
                        )
                        if evidence is not None:
                            evidence_records.append(evidence)
                            physical.add(character.character_id)
                            found_for_sentence = True
                            break
                    if found_for_sentence:
                        break
            physical.update(
                _copresent_object_ids(
                    ckpt,
                    sentence=sentence,
                    evidence_records=evidence_records,
                )
            )
    return physical


def physically_present_character_ids(
    ckpt: CheckpointFile,
    visible_texts: Iterable[str],
) -> set[str]:
    """Return roster ids directly embodied in the supplied visible facts.

    This is the shared person-channel classifier used for first-meeting state
    and noncanonical visual staging.  Keeping both consumers on one contract
    prevents a reported name, screen image, or future arrival from selecting
    the named person's remote location as the current scene.
    """

    return _physically_present_character_ids(ckpt, visible_texts)


_LIVE_VIEW_FOREGROUND_RE = re.compile(
    r"\b(?:(?:the|your)\s+)?(?:[A-Za-z0-9'_-]+\s+){0,4}"
    r"(?:view|feed|camera)\s+(?:holds?|frames?|shows?|tracks?|follows?|"
    r"(?:centers?|focuses?|rests?|settles?)(?:\s+on)?|"
    r"remains?(?:\s+on)?|"
    r"(?:is\s+)?held\s+on|"
    r"shifts?\s+(?:from\s+[^.;]{1,120}?\s+)?to)\s+"
    r"(?P<subjects>.+?)"
    r"(?=\s+(?:together\s+)?(?:in|at|beside|before|behind|beneath|under|"
    r"near|inside|outside|within|against|with)\b|[.;]|$)",
    re.IGNORECASE,
)

_LIVE_VIEW_SCENE_RE = re.compile(
    r"\b(?:(?:the|your)\s+)?(?:[A-Za-z0-9'_-]+\s+){0,4}"
    r"(?:view|feed|camera)\s+(?:shows?|captures?)\s+"
    r"(?P<scene>[^.;]+)",
    re.IGNORECASE,
)


def _leading_live_view_subject_ids(
    ckpt: CheckpointFile,
    scene: str,
) -> set[str]:
    """Resolve roster subjects named at the start of an explicit live view.

    A live-feed fact may put an embodied predicate immediately after its
    subject (for example, ``Alice speaks ... and enters ...``).  The general
    physical-presence classifier deliberately treats speech conservatively,
    while this surrounding live-view clause already establishes that its
    leading subject is what the camera depicts.  Keep the exception narrow:
    accept only a leading roster name/list and reject possessives such as
    ``Alice's empty chair`` or ``Alice's status record``.
    """

    probes = sorted(
        (
            (probe, character.character_id)
            for character in _active_roster_characters(ckpt)
            for probe in _character_presence_probes(ckpt, character)
            if probe
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    offset = 0
    matched_ids: set[str] = set()
    while True:
        subject_match: re.Match[str] | None = None
        subject_id = ""
        for probe, character_id in probes:
            candidate = re.match(
                rf"\s*{re.escape(probe)}(?![A-Za-z0-9_])",
                scene[offset:],
                re.IGNORECASE,
            )
            if candidate is not None:
                subject_match = candidate
                subject_id = character_id
                break
        if subject_match is None:
            break
        offset += subject_match.end()
        if re.match(r"\s*[’']s\b", scene[offset:], re.IGNORECASE):
            return set()
        matched_ids.add(subject_id)
        separator = re.match(
            r"\s*(?:,\s*(?:and\s+)?|(?:and|or|&)\s+)",
            scene[offset:],
            re.IGNORECASE,
        )
        if separator is None:
            break
        next_offset = offset + separator.end()
        if not any(
            re.match(
                rf"\s*{re.escape(probe)}(?![A-Za-z0-9_])",
                scene[next_offset:],
                re.IGNORECASE,
            )
            for probe, _character_id in probes
        ):
            break
        offset = next_offset
    return matched_ids


def _live_view_foreground_character_ids(
    ckpt: CheckpointFile,
    visible_texts: Iterable[str],
) -> set[str]:
    """Resolve a roster-only subject list explicitly framed by a view."""

    staged: set[str] = set()
    for text in visible_texts:
        for scene_match in _LIVE_VIEW_SCENE_RE.finditer(text or ""):
            staged.update(
                _leading_live_view_subject_ids(
                    ckpt,
                    scene_match.group("scene"),
                )
            )
            staged.update(
                _physically_present_character_ids(
                    ckpt,
                    [scene_match.group("scene")],
                    include_generic_actions=True,
                )
            )
        for view_match in _LIVE_VIEW_FOREGROUND_RE.finditer(text or ""):
            subject_text = view_match.group("subjects")
            cleaned = subject_text
            matched_ids: set[str] = set()
            probes = sorted(
                (
                    (probe, character.character_id)
                    for character in _active_roster_characters(ckpt)
                    for probe in _character_presence_probes(ckpt, character)
                    if probe
                ),
                key=lambda item: len(item[0]),
                reverse=True,
            )
            for probe, character_id in probes:
                pattern = re.compile(
                    rf"(?<![A-Za-z0-9_]){re.escape(probe)}"
                    rf"(?![A-Za-z0-9_])",
                    re.IGNORECASE,
                )
                cleaned, count = pattern.subn(" ", cleaned)
                if count:
                    matched_ids.add(character_id)
            cleaned = re.sub(
                r"\b(?:and|or|both|all|each|together)\b|[,/&+]",
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )
            if matched_ids and not re.search(r"[A-Za-z0-9_]", cleaned):
                staged.update(matched_ids)
    return staged


def visually_staged_character_ids(
    ckpt: CheckpointFile,
    visible_texts: Iterable[str],
) -> set[str]:
    """Return embodied subjects suitable for transient VN foreground cues.

    Unlike the persistent first-meeting ledger, a page foreground must handle
    arbitrary visible action verbs authored by the router. The same mediated,
    reported, future, possessive, and quoted-content guards still apply; the
    broader predicate matcher changes only transient staging and never marks a
    character as visually introduced.
    """

    visible_text_list = list(visible_texts)
    staged = _physically_present_character_ids(
        ckpt,
        visible_text_list,
        include_generic_actions=True,
    )
    staged.update(_live_view_foreground_character_ids(ckpt, visible_text_list))
    return staged


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
    priority_index = {cid: index for index, cid in enumerate(priority_ids)}
    candidate_set = (
        set(candidate_ids)
        if candidate_ids is not None
        else _physically_present_character_ids(ckpt, texts)
    ) | tagged_ids
    candidate_set.discard(viewer_id)

    mark_ids: list[str] = []
    loadout_candidates: list[tuple[tuple[int, int, int], VisualIntroduction]] = []
    roster_index = {
        character.character_id: index for index, character in enumerate(ckpt.characters)
    }
    for character_id in candidate_set:
        character = by_id.get(character_id)
        if character is None or character_id in introduced:
            continue
        if character_id in tagged_ids:
            mark_ids.append(character_id)
            continue
        first_look_override = first_look_override_for_viewer(
            ckpt,
            viewer_character_id=viewer_id,
            character=character,
        )
        loadout = first_look_override or _default_loadout(character)
        public_context = (
            _public_context(character)
            if include_public_context and first_look_override is None
            else ""
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
        loadout_candidates.append(
            (
                sort_key,
                VisualIntroduction(
                    character_id=character_id,
                    name=character.name or character_id,
                    default_loadout=loadout,
                    public_context=public_context,
                ),
            )
        )

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
    event: CanonicalEventRecord,
    observation_level: str,
    priority_target_ids: Iterable[str] = (),
    max_loadouts: int = AGENT_FIRST_MEETING_CAP,
) -> VisualIntroductionPlan:
    if observation_level != "direct":
        return VisualIntroductionPlan(loadouts=[], mark_character_ids=[])
    texts = visible_fact_texts(
        event.observable_facts,
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
    resolved: list[tuple[NarratorEventRef, CanonicalEventRecord]],
    max_loadouts: int = NARRATOR_FIRST_MEETING_CAP,
) -> VisualIntroductionPlan:
    visible_texts: list[str] = []
    for entry, event in resolved:
        if entry.observation_level != "direct":
            continue
        visible_texts.extend(
            visible_fact_texts(
                event.observable_facts,
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
