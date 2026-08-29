"""Canonical-event-derived One-Star mission reports and deterministic boards."""

from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from app.engine.one_star_adapter import (
    is_one_star_checkpoint,
    load_one_star_account,
    one_star_mission_report_recipient_ids,
)
from app.engine.player_media import ResolvedPlayerMedia
from app.schemas.checkpoint import CheckpointFile
from app.schemas.one_star import OneStarCost, OneStarStateUpdate
from app.schemas.responses import VisualNovelRender


REPORT_BOARD_SIZE = (1024, 576)
_SERIF_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")
_SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
_SANS_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

MissionReportKind = Literal["critical", "boss_kill", "dialogue"]


class OneStarMissionReportError(RuntimeError):
    """A committed mission-report presentation contract is inconsistent."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"One-Star mission report failed ({code}).")


@dataclass(frozen=True)
class OneStarMissionReportHighlight:
    event_id: str
    kind: MissionReportKind
    credited_character_ids: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class OneStarMissionReportDeath:
    event_id: str
    character_id: str
    character_name: str
    cause: str


@dataclass(frozen=True)
class OneStarMissionReportReward:
    first_clear: bool
    resources: OneStarCost
    unlocked_floor: int = 0


@dataclass(frozen=True)
class OneStarMissionReport:
    event_id: str
    mission_id: str
    floor: int
    outcome: Literal["completed", "failed", "escaped"]
    party_character_ids: tuple[str, ...]
    deaths: tuple[OneStarMissionReportDeath, ...]
    boss_kills: tuple[OneStarMissionReportHighlight, ...]
    critical_highlights: tuple[OneStarMissionReportHighlight, ...]
    dialogue_highlights: tuple[OneStarMissionReportHighlight, ...]
    reward: OneStarMissionReportReward | None
    mvp_character_id: str
    mvp_evidence_event_id: str


@dataclass(frozen=True)
class OneStarMissionReportBoard:
    media: ResolvedPlayerMedia
    accessible_text: str
    event_id: str
    page_number: int
    page_count: int


def _detail_values(update: OneStarStateUpdate) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for raw_detail in update.details:
        key, separator, value = raw_detail.partition("=")
        if separator and key.strip():
            values.setdefault(key.strip(), []).append(value.strip())
    return values


def _one_detail(
    details: dict[str, list[str]],
    key: str,
    *,
    default: str = "",
) -> str:
    values = details.get(key, [])
    if not values:
        return default
    if len(values) != 1:
        raise OneStarMissionReportError("duplicate_detail")
    return values[0]


def _event_report_text(
    event: object,
    credited_character_ids: tuple[str, ...],
) -> str:
    """Project only public or unanimously credited-party-visible facts."""

    credited = set(credited_character_ids)
    facts = []
    for fact in getattr(
        getattr(event, "canonical_event", None),
        "observable_facts",
        (),
    ):
        if fact.audience == "all_observers" or (
            credited and credited.issubset(fact.visible_to)
        ):
            text = fact.text.strip()
            if text:
                facts.append(text)
    return " ".join(facts)


def _mission_start_for_end(
    checkpoint: CheckpointFile,
    *,
    mission_id: str,
    end_index: int,
) -> tuple[int, OneStarStateUpdate]:
    for index in range(end_index, -1, -1):
        event = checkpoint.canonical_events[index]
        starts = [
            update
            for update in getattr(event, "state_updates", ())
            if update.kind == "mission_start"
            and update.target_id.strip() == mission_id
        ]
        if len(starts) > 1:
            raise OneStarMissionReportError("ambiguous_mission_start")
        if starts:
            return index, starts[0]
    raise OneStarMissionReportError("mission_start_missing")


def _reward_for_completed_mission(
    checkpoint: CheckpointFile,
    *,
    mission_id: str,
    floor: int,
    end_index: int,
) -> OneStarMissionReportReward:
    _owner, account = load_one_star_account(checkpoint)
    reward = account.config.floor_rewards.get(floor)
    if reward is None:
        raise OneStarMissionReportError("floor_reward_missing")
    prior_completion = any(
        update.kind == "mission_end"
        and update.target_id.strip() == mission_id
        and update.value.strip() == "completed"
        for event in checkpoint.canonical_events[:end_index]
        for update in getattr(event, "state_updates", ())
    )
    if not prior_completion:
        resources = OneStarCost.model_validate(reward.model_dump())
        unlocked_floor = (
            floor + 1 if floor + 1 in account.config.floor_scenarios else 0
        )
        return OneStarMissionReportReward(
            first_clear=True,
            resources=resources,
            unlocked_floor=unlocked_floor,
        )
    gold = (
        reward.gold
        * account.config.repeat_gold_numerator
        // account.config.repeat_gold_denominator
    )
    if reward.gold:
        gold = max(account.config.repeat_gold_minimum, gold)
    return OneStarMissionReportReward(
        first_clear=False,
        resources=OneStarCost(
            gold=gold,
            gems=0,
            building_resources=0,
            materials={},
        ),
    )


def committed_one_star_mission_report(
    checkpoint: CheckpointFile,
    event_id: str,
) -> OneStarMissionReport | None:
    """Project one report from a committed mission-end event."""

    if not is_one_star_checkpoint(checkpoint):
        return None
    _owner, account = load_one_star_account(checkpoint)
    clean_event_id = event_id.strip()
    if clean_event_id not in account.state.applied_event_fingerprints:
        return None
    matches = [
        (index, event)
        for index, event in enumerate(checkpoint.canonical_events)
        if event.event_id == clean_event_id
    ]
    if len(matches) > 1:
        raise OneStarMissionReportError("duplicate_canonical_event")
    if not matches:
        return None
    end_index, end_event = matches[0]
    end_updates = [
        update
        for update in getattr(end_event, "state_updates", ())
        if update.kind == "mission_end"
    ]
    if not end_updates:
        return None
    if len(end_updates) != 1:
        raise OneStarMissionReportError("ambiguous_mission_end")
    end_update = end_updates[0]
    mission_id = end_update.target_id.strip()
    if not mission_id:
        raise OneStarMissionReportError("mission_id_missing")
    start_index, start_update = _mission_start_for_end(
        checkpoint,
        mission_id=mission_id,
        end_index=end_index,
    )
    start_details = _detail_values(start_update)
    party_ids = tuple(start_details.get("party", []))
    if not party_ids or len(party_ids) != len(set(party_ids)):
        raise OneStarMissionReportError("mission_party_invalid")
    try:
        floor = int(start_update.value)
    except (TypeError, ValueError) as exc:
        raise OneStarMissionReportError("mission_floor_invalid") from exc

    character_names = {
        character.character_id: character.name
        for character in checkpoint.characters
    }
    highlights: list[OneStarMissionReportHighlight] = []
    deaths: list[OneStarMissionReportDeath] = []
    for event in checkpoint.canonical_events[start_index:end_index + 1]:
        for update in getattr(event, "state_updates", ()):
            details = _detail_values(update)
            if (
                update.kind == "mission_update"
                and update.target_id.strip() == mission_id
                and "report_kind" in details
            ):
                report_kind = _one_detail(details, "report_kind")
                if report_kind not in {"critical", "boss_kill", "dialogue"}:
                    raise OneStarMissionReportError("report_kind_invalid")
                credits = tuple(details.get("report_credit", []))
                if not credits or not set(credits).issubset(party_ids):
                    raise OneStarMissionReportError("report_credit_invalid")
                report_text = _event_report_text(event, credits)
                highlights.append(OneStarMissionReportHighlight(
                    event_id=event.event_id,
                    kind=report_kind,
                    credited_character_ids=credits,
                    text=report_text or report_kind.replace("_", " ").title(),
                ))
            if (
                update.kind == "hero_delta"
                and update.target_id.strip() in party_ids
                and _one_detail(details, "terminal_action", default="none")
                == "death"
            ):
                character_id = update.target_id.strip()
                deaths.append(OneStarMissionReportDeath(
                    event_id=event.event_id,
                    character_id=character_id,
                    character_name=character_names.get(
                        character_id,
                        "Unknown Hero",
                    ),
                    cause=_one_detail(
                        details,
                        "death_cause",
                        default="cause not recorded",
                    ),
                ))

    end_details = _detail_values(end_update)
    mvp_character_id = _one_detail(end_details, "mvp_character_id")
    mvp_evidence_event_id = _one_detail(
        end_details,
        "mvp_evidence_event_id",
    )
    marked_event_ids = {
        highlight.event_id
        for highlight in highlights
        if mvp_character_id in highlight.credited_character_ids
    }
    if (
        mvp_character_id not in party_ids
        or mvp_evidence_event_id not in marked_event_ids
    ):
        raise OneStarMissionReportError("mvp_evidence_invalid")
    outcome = end_update.value.strip()
    if outcome not in {"completed", "failed", "escaped"}:
        raise OneStarMissionReportError("mission_outcome_invalid")
    reward = (
        _reward_for_completed_mission(
            checkpoint,
            mission_id=mission_id,
            floor=floor,
            end_index=end_index,
        )
        if outcome == "completed"
        else None
    )
    return OneStarMissionReport(
        event_id=clean_event_id,
        mission_id=mission_id,
        floor=floor,
        outcome=outcome,
        party_character_ids=party_ids,
        deaths=tuple(deaths),
        boss_kills=tuple(
            highlight for highlight in highlights if highlight.kind == "boss_kill"
        ),
        critical_highlights=tuple(
            highlight for highlight in highlights if highlight.kind == "critical"
        ),
        dialogue_highlights=tuple(
            highlight for highlight in highlights if highlight.kind == "dialogue"
        ),
        reward=reward,
        mvp_character_id=mvp_character_id,
        mvp_evidence_event_id=mvp_evidence_event_id,
    )


def new_one_star_mission_reports(
    checkpoint: CheckpointFile,
    previous_checkpoint: CheckpointFile | None,
) -> tuple[OneStarMissionReport, ...]:
    """Return newly committed mission reports in canonical order."""

    if not is_one_star_checkpoint(checkpoint):
        return ()
    _owner, account = load_one_star_account(checkpoint)
    prior_ids: set[str] = set()
    if previous_checkpoint is not None and is_one_star_checkpoint(previous_checkpoint):
        _prior_owner, prior_account = load_one_star_account(previous_checkpoint)
        prior_ids = set(prior_account.state.applied_event_fingerprints)
    new_ids = set(account.state.applied_event_fingerprints) - prior_ids
    reports: list[OneStarMissionReport] = []
    seen: set[str] = set()
    for event in checkpoint.canonical_events:
        if event.event_id in seen or event.event_id not in new_ids:
            continue
        seen.add(event.event_id)
        report = committed_one_star_mission_report(checkpoint, event.event_id)
        if report is not None:
            reports.append(report)
    return tuple(reports)


def one_star_mission_reports_for_render(
    *,
    checkpoint: CheckpointFile,
    previous_checkpoint: CheckpointFile | None,
    viewer_character_id: str,
    render: VisualNovelRender,
) -> tuple[OneStarMissionReport, ...]:
    """Select new owner reports in rendered-segment order."""

    if not is_one_star_checkpoint(checkpoint):
        return ()
    if viewer_character_id not in one_star_mission_report_recipient_ids(
        checkpoint
    ):
        return ()
    required = {
        report.event_id: report
        for report in new_one_star_mission_reports(
            checkpoint,
            previous_checkpoint,
        )
    }
    rendered_ids = [segment.rendered_event_id for segment in render.segments]
    if set(required) - set(rendered_ids):
        raise OneStarMissionReportError("owner_render_missing_mission_end")
    ordered: list[OneStarMissionReport] = []
    seen: set[str] = set()
    for event_id in rendered_ids:
        if event_id in required and event_id not in seen:
            ordered.append(required[event_id])
            seen.add(event_id)
    return tuple(ordered)


def _resource_text(resources: OneStarCost) -> str:
    entries = [
        f"{resources.gold} Gold",
        f"{resources.gems} Gems",
        f"{resources.building_resources} Building Resources",
    ]
    entries.extend(
        f"{amount} {material_id.replace('_', ' ').title()}"
        for material_id, amount in sorted(resources.materials.items())
    )
    return ", ".join(entry for entry in entries if not entry.startswith("0 ")) or "None"


def _credited_names(
    report: OneStarMissionReport,
    checkpoint: CheckpointFile,
    ids: tuple[str, ...],
) -> str:
    names = {
        character.character_id: character.name
        for character in checkpoint.characters
    }
    return ", ".join(
        names.get(character_id, "Unknown Hero")
        for character_id in ids
    )


def render_one_star_mission_report_accessibility(
    *,
    checkpoint: CheckpointFile,
    report: OneStarMissionReport,
) -> str:
    """Render the complete deterministic text counterpart of one report."""

    names = {
        character.character_id: character.name
        for character in checkpoint.characters
    }
    lines = [
        f"System mission report — Floor {report.floor}",
        f"Outcome: {report.outcome.title()}",
    ]
    if report.deaths:
        lines.append("Deaths: " + "; ".join(
            f"{death.character_name} — {death.cause}"
            for death in report.deaths
        ))
    else:
        lines.append("Deaths: None")
    for title, highlights in (
        ("Boss kills", report.boss_kills),
        ("Critical highlights", report.critical_highlights),
        ("Dialogue highlights", report.dialogue_highlights),
    ):
        if highlights:
            lines.append(title + ": " + "; ".join(
                f"{highlight.text} [credit: "
                f"{_credited_names(report, checkpoint, highlight.credited_character_ids)}]"
                for highlight in highlights
            ))
        else:
            lines.append(title + ": None")
    if report.reward is None:
        lines.append("Rewards: None")
        lines.append("Unlocks: None")
    else:
        reward_label = "First clear" if report.reward.first_clear else "Repeat clear"
        lines.append(
            f"Rewards: {reward_label} — {_resource_text(report.reward.resources)}"
        )
        lines.append(
            "Unlocks: "
            + (
                f"Floor {report.reward.unlocked_floor}"
                if report.reward.unlocked_floor
                else "None"
            )
        )
    lines.append(
        f"MVP: {names.get(report.mvp_character_id, 'Unknown Hero')}"
    )
    return "\n".join(lines)


def render_one_star_mission_report_boards(
    *,
    checkpoint: CheckpointFile,
    report: OneStarMissionReport,
) -> tuple[OneStarMissionReportBoard, ...]:
    """Render local, deterministic PNG boards with matching accessibility."""

    accessible = render_one_star_mission_report_accessibility(
        checkpoint=checkpoint,
        report=report,
    )
    wrapped: list[str] = []
    for line in accessible.splitlines():
        wrapped.extend(textwrap.wrap(line, width=82) or [""])
    pages = [wrapped[index:index + 13] for index in range(0, len(wrapped), 13)]
    page_count = len(pages)
    boards: list[OneStarMissionReportBoard] = []
    for page_number, lines in enumerate(pages, start=1):
        image = Image.new("RGB", REPORT_BOARD_SIZE, (5, 13, 26))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1023, 575), outline=(160, 126, 65), width=3)
        draw.text(
            (42, 30),
            "MISSION REPORT",
            font=ImageFont.truetype(str(_SERIF_BOLD), 32),
            fill=(239, 235, 222),
        )
        if page_count > 1:
            draw.text(
                (822, 42),
                f"{page_number} / {page_count}",
                font=ImageFont.truetype(str(_SANS_BOLD), 17),
                fill=(164, 187, 204),
            )
        y = 96
        for line in lines:
            draw.text(
                (48, y),
                line,
                font=ImageFont.truetype(str(_SANS), 24),
                fill=(220, 228, 233),
            )
            y += 33
        output = BytesIO()
        image.save(output, format="PNG", optimize=False)
        data = output.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        boards.append(OneStarMissionReportBoard(
            media=ResolvedPlayerMedia(
                filename=f"one-star-mission-report-{page_number:02d}.png",
                mime_type="image/png",
                data=data,
                sha256=digest,
                byte_count=len(data),
                width=REPORT_BOARD_SIZE[0],
                height=REPORT_BOARD_SIZE[1],
            ),
            accessible_text="\n".join(lines),
            event_id=report.event_id,
            page_number=page_number,
            page_count=page_count,
        ))
    return tuple(boards)
