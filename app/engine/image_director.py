from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.engine.text_safety import strip_terminal_control
from app.llm.client import LLMClient
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    CharacterVisuals,
    PrivateState,
    PublicSheet,
    is_player_authored_slot,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import redact_imported_asset_text
from app.schemas.event_router import EventRouterOutput
from app.schemas.image_director import ImageDirectorOutput, ImageGenerationMode
from app.schemas.state import (
    RenderBufferEntry,
    SessionConfig,
    SessionState,
    SessionSettings,
    StorySetting,
    WorldState,
)


_ONE_STAR_RULESET_ID = "one_star_ascension"
_ONE_STAR_HERO_KEY = "one_star_hero"


_OBSERVATION_LEVELS = {
    "d": "direct",
    "i": "indirect",
    "f": "inferred",
}
_FORBIDDEN_RENDERED_TEXT_RE = re.compile(
    r"\b(?:caption|lettering|logo|watermark|speech bubble|text line|"
    r"readable text|written words?|visible words?|"
    r"status window|system window|interface (?:panel|window|overlay|screen|text)|"
    r"card art|panel border|split panel|comic page|collage|character sheet|"
    r"lineup|poster|promotional splash)\b|[★☆]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PublicCharacterVisual:
    character_id: str
    name: str
    appearance: str
    default_loadout: str
    depiction_policy: str
    is_new_character: bool
    has_identity_reference: bool
    public_role: str = ""
    is_playable: bool = False
    recurring_actor: bool = False

    def prompt_line(self) -> str:
        fields = [
            f"id={self.character_id}",
            f"name={self.name or self.character_id}",
            f"role={self.public_role or '(unspecified)'}",
            f"depiction_policy={self.depiction_policy}",
            f"new_character={'yes' if self.is_new_character else 'no'}",
            f"player_controlled={'yes' if self.is_playable else 'no'}",
            f"recurring_actor={'yes' if self.recurring_actor else 'no'}",
            (
                "has_identity_reference=yes"
                if self.has_identity_reference
                else "has_identity_reference=no"
            ),
        ]
        if self.appearance:
            fields.append(f"appearance={self.appearance}")
        if self.default_loadout:
            fields.append(f"loadout={self.default_loadout}")
        return "- " + "; ".join(fields)


@dataclass(frozen=True)
class SelectableVisualReference:
    """Text-only authored option; image metadata stays engine-private."""

    reference_id: str
    scope: str
    scope_id: str
    selection_hint: str

    def prompt_line(self) -> str:
        applies_to = (
            self.scope_id if self.scope == "character" else "visible_location"
        )
        return (
            f"- id={self.reference_id}; applies_to={applies_to}; "
            f"use={self.selection_hint}"
        )


@dataclass(frozen=True)
class VisibleEventProjection:
    session_id: str
    transaction_id: str
    source_turn_index: int
    event_id: str
    event_sequence: int
    event_fingerprint: str
    viewer_character_ids: tuple[str, ...]
    perception_level: str
    effective_at_s: int
    duration_s: int
    visible_facts: tuple[tuple[str, int, int], ...]
    characters: tuple[PublicCharacterVisual, ...]
    story_genre: str
    story_era: str
    story_tone: str
    story_premise: str
    canonical_event_count: int
    active_roster_count: int
    total_roster_count: int
    reference_options: tuple[SelectableVisualReference, ...] = ()
    engine_visual_style: str = ""
    # Engine-only opaque key used after the director returns. The label is
    # deliberately omitted from rendered LLM messages; the director sees only
    # has_location_reference plus public observable facts.
    engine_location_label: str = ""
    has_location_reference: bool = False
    presentation_mode: str = "prose"

    def grouping_key(self) -> str:
        """Identity of model and diffusion input, excluding audiences."""

        payload = {
            "perception_level": self.perception_level,
            "effective_at_s": self.effective_at_s,
            "duration_s": self.duration_s,
            "visible_facts": self.visible_facts,
            "characters": [
                character.__dict__ for character in self.characters
            ],
            "reference_options": [
                reference.__dict__ for reference in self.reference_options
            ],
            "story": (
                self.story_genre,
                self.story_era,
                self.story_tone,
                self.story_premise,
            ),
            "progress": (
                self.canonical_event_count,
                self.active_roster_count,
                self.total_roster_count,
            ),
            "engine_location_label": self.engine_location_label,
            "has_location_reference": self.has_location_reference,
            "presentation_mode": self.presentation_mode,
        }
        return _json_hash(payload)

    def to_storage_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "transaction_id": self.transaction_id,
            "source_turn_index": self.source_turn_index,
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "event_fingerprint": self.event_fingerprint,
            "viewer_character_ids": list(self.viewer_character_ids),
            "perception_level": self.perception_level,
            "effective_at_s": self.effective_at_s,
            "duration_s": self.duration_s,
            "visible_facts": [list(item) for item in self.visible_facts],
            "characters": [character.__dict__ for character in self.characters],
            "reference_options": [
                reference.__dict__ for reference in self.reference_options
            ],
            "story_genre": self.story_genre,
            "story_era": self.story_era,
            "story_tone": self.story_tone,
            "story_premise": self.story_premise,
            "canonical_event_count": self.canonical_event_count,
            "active_roster_count": self.active_roster_count,
            "total_roster_count": self.total_roster_count,
            "engine_visual_style": self.engine_visual_style,
            "engine_location_label": self.engine_location_label,
            "has_location_reference": self.has_location_reference,
            "presentation_mode": self.presentation_mode,
        }

    @classmethod
    def from_storage_dict(
        cls,
        value: dict[str, object],
    ) -> "VisibleEventProjection":
        return cls(
            session_id=str(value["session_id"]),
            transaction_id=str(value["transaction_id"]),
            source_turn_index=int(value["source_turn_index"]),
            event_id=str(value["event_id"]),
            event_sequence=int(value["event_sequence"]),
            event_fingerprint=str(value["event_fingerprint"]),
            viewer_character_ids=tuple(
                str(item) for item in value["viewer_character_ids"]  # type: ignore[index]
            ),
            perception_level=str(value["perception_level"]),
            effective_at_s=int(value["effective_at_s"]),
            duration_s=int(value["duration_s"]),
            visible_facts=tuple(
                (str(item[0]), int(item[1]), int(item[2]))
                for item in value["visible_facts"]  # type: ignore[index]
            ),
            characters=tuple(
                PublicCharacterVisual(**item)
                for item in value["characters"]  # type: ignore[arg-type,index]
            ),
            reference_options=tuple(
                SelectableVisualReference(**item)
                for item in value.get("reference_options", [])  # type: ignore[arg-type,union-attr]
            ),
            story_genre=str(value["story_genre"]),
            story_era=str(value["story_era"]),
            story_tone=str(value["story_tone"]),
            story_premise=str(value["story_premise"]),
            canonical_event_count=int(value["canonical_event_count"]),
            active_roster_count=int(value["active_roster_count"]),
            total_roster_count=int(value["total_roster_count"]),
            engine_visual_style=str(value.get("engine_visual_style") or ""),
            engine_location_label=str(
                value.get("engine_location_label") or ""
            ),
            has_location_reference=bool(
                value.get("has_location_reference", False)
            ),
            presentation_mode=str(value.get("presentation_mode") or "prose"),
        )


@dataclass(frozen=True)
class DurableDirectorRun:
    run_id: str
    projection: VisibleEventProjection
    status: str
    output: ImageDirectorOutput | None
    error_code: str
    attempts: int
    created_at: float
    updated_at: float


def source_event_fingerprint(event: EventRouterOutput) -> str:
    return hashlib.sha256(
        event.model_dump_json(
            exclude_none=False,
            by_alias=True,
        ).encode("utf-8")
    ).hexdigest()


def projection_checkpoint_snapshot(
    checkpoint: CheckpointFile,
    *,
    event_ids: Iterable[str] = (),
) -> CheckpointFile:
    """Copy only bounded public state needed by asynchronous projection."""

    setting = checkpoint.world_state.setting
    settings = getattr(checkpoint.session.config, "settings", None)
    ruleset_id = str(
        getattr(
            settings,
            "ruleset_id",
            "",
        )
        or ""
    )
    selected_event_ids = set(event_ids)
    return CheckpointFile(
        session=SessionState(
            session_id=checkpoint.session.session_id,
            player_character_id=checkpoint.session.player_character_id,
            character_bindings=dict(
                checkpoint.session.character_bindings or {}
            ),
            config=SessionConfig(
                settings=SessionSettings(
                    ruleset_id=ruleset_id,
                    presentation_mode=str(
                        getattr(settings, "presentation_mode", "prose")
                        or "prose"
                    ),
                ),
            ),
        ),
        world_state=WorldState(
            setting=StorySetting(
                genre=_safe_text(setting.genre, 300),
                era=_safe_text(setting.era, 300),
                tone=_safe_text(setting.tone, 300),
                premise=_safe_text(setting.premise, 1_000),
                visual_style=_safe_text(setting.visual_style, 800),
            )
        ),
        characters=[
            CharacterRecord(
                character_id=_safe_identifier(character.character_id),
                name=_safe_text(character.name, 200),
                entity_kind=character.entity_kind,
                status=character.status,
                location=_safe_identifier(character.location),
                public_sheet=PublicSheet(
                    role=_safe_text(character.public_sheet.role, 300),
                    appearance=_safe_text(
                        character.public_sheet.appearance,
                        600,
                    )
                ),
                visuals=CharacterVisuals(
                    default_loadout=_safe_text(
                        image_loadout_for_character(checkpoint, character),
                        700,
                    ),
                    depiction_policy=character.visuals.depiction_policy,
                    identity_reference_id=(
                        character.visuals.identity_reference_id
                    ),
                ),
                private_state=PrivateState(
                    intentions_enabled=bool(
                        character.private_state.intentions_enabled
                    ),
                ),
                is_playable=bool(character.is_playable),
            )
            for character in checkpoint.characters
            if not (
                is_player_authored_slot(character)
                and character.character_id
                not in (checkpoint.session.character_bindings or {})
            )
        ],
        reviewed_visual_references=[
            reference.model_copy(deep=True)
            for reference in checkpoint.reviewed_visual_references
        ],
        location_visual_reference_ids={
            label: list(reference_ids)
            for label, reference_ids in (
                checkpoint.location_visual_reference_ids.items()
            )
        },
        canonical_events=[
            event.model_copy(deep=True)
            for event in checkpoint.canonical_events
            if event.event_id in selected_event_ids
        ],
    )


def build_projection_groups(
    *,
    checkpoint: CheckpointFile,
    event: EventRouterOutput,
    event_sequence: int,
    transaction_id: str,
    source_turn_index: int,
    spawned_records: Sequence[CharacterRecord] = (),
    actor_id: str = "",
    active_identity_character_ids: Iterable[str] = (),
    active_location_labels: Iterable[str] = (),
) -> list[VisibleEventProjection]:
    """Project one finalized event and merge equivalent human audiences."""

    human_ids = {
        character_id
        for character_id, user_id in (
            checkpoint.session.character_bindings or {}
        ).items()
        if str(user_id).strip()
    }
    if checkpoint.session.player_character_id:
        human_ids.add(checkpoint.session.player_character_id)
    if not human_ids:
        return []
    active_references = set(active_identity_character_ids)
    active_locations = set(active_location_labels)
    spawned_by_id = {
        character.character_id: character for character in spawned_records
    }
    by_id = {
        character.character_id: character for character in checkpoint.characters
    }
    by_id.update(spawned_by_id)
    new_ids = {
        request.character_id
        for request in event.spawn
        if request.character_id
    }
    setting = checkpoint.world_state.setting
    event_hash = source_event_fingerprint(event)
    base = {
        "session_id": checkpoint.session.session_id,
        "transaction_id": transaction_id,
        "source_turn_index": source_turn_index,
        "event_id": event.event_id,
        "event_sequence": event_sequence,
        "event_fingerprint": event_hash,
        "story_genre": _safe_text(setting.genre, 300),
        "story_era": _safe_text(setting.era, 300),
        "story_tone": _safe_text(setting.tone, 300),
        "story_premise": _safe_text(setting.premise, 1_000),
        "canonical_event_count": event_sequence + 1,
        "active_roster_count": sum(
            character.status == CharacterStatus.active
            for character in by_id.values()
        ),
        "total_roster_count": sum(
            character.status != CharacterStatus.culled
            for character in by_id.values()
        ),
        "engine_visual_style": _safe_text(setting.visual_style, 800),
    }

    grouped: dict[str, VisibleEventProjection] = {}
    for observer in event.observers:
        viewer_id = observer.character_id
        if viewer_id not in human_ids:
            continue
        facts = tuple(
            (
                _safe_text(fact.text, 2_000),
                max(0, int(fact.at_offset_s)),
                max(0, int(fact.duration_s)),
            )
            for fact in event.canonical_event.observable_facts
            if fact.is_visible_to(viewer_id) and _safe_text(fact.text, 2_000)
        )
        if not facts:
            continue
        public_characters = _public_character_projection(
            checkpoint=checkpoint,
            facts=facts,
            actor_id=actor_id,
            viewer_character_id=viewer_id,
            by_id=by_id,
            new_ids=new_ids,
            active_identity_character_ids=active_references,
        )
        location_label = _visible_location_label(
            checkpoint=checkpoint,
            event=event,
            viewer_character_id=viewer_id,
        )
        has_location_reference = location_label in active_locations
        projection = VisibleEventProjection(
            **base,
            characters=public_characters,
            reference_options=_selectable_reference_options(
                checkpoint=checkpoint,
                characters=public_characters,
                location_label=location_label,
                active_identity_character_ids=active_references,
                active_location_labels=active_locations,
            ),
            viewer_character_ids=(viewer_id,),
            perception_level=_OBSERVATION_LEVELS.get(
                observer.observation_level,
                "direct",
            ),
            effective_at_s=max(0, int(event.effective_at_s)),
            duration_s=max(0, int(event.duration_s)),
            visible_facts=facts,
            engine_location_label=(
                location_label if has_location_reference else ""
            ),
            has_location_reference=has_location_reference,
        )
        key = projection.grouping_key()
        prior = grouped.get(key)
        if prior is None:
            grouped[key] = projection
        else:
            grouped[key] = VisibleEventProjection(
                **{
                    **prior.__dict__,
                    "viewer_character_ids": tuple(
                        dict.fromkeys(
                            (*prior.viewer_character_ids, viewer_id)
                        )
                    ),
                }
            )
    return list(grouped.values())


def build_render_batch_projection_groups(
    *,
    checkpoint: CheckpointFile,
    buffered_events_by_pov: dict[str, Sequence[RenderBufferEntry]],
    eligible_viewer_ids: set[str],
    transaction_id: str,
    source_turn_index: int,
    spawned_records: Sequence[CharacterRecord] = (),
    actor_ids_by_event_id: dict[str, str] | None = None,
    active_identity_character_ids: Iterable[str] = (),
    active_location_labels: Iterable[str] = (),
) -> list[VisibleEventProjection]:
    """Project a complete pending render and merge equivalent POV batches."""
    actor_ids = actor_ids_by_event_id or {}
    event_sequences = {
        entry.event_id: entry.event_sequence
        for viewer_id, entries in buffered_events_by_pov.items()
        if viewer_id in eligible_viewer_ids
        for entry in entries
    }
    events_by_id = {
        event.event_id: event for event in checkpoint.canonical_events
    }
    needed_event_ids = {
        entry.event_id
        for viewer_id, entries in buffered_events_by_pov.items()
        if viewer_id in eligible_viewer_ids
        for entry in entries
    }
    parts_by_viewer_event: dict[
        tuple[str, str],
        VisibleEventProjection,
    ] = {}
    for event_id in sorted(
        needed_event_ids,
        key=lambda value: event_sequences.get(value, 10**9),
    ):
        event = events_by_id.get(event_id)
        event_sequence = event_sequences.get(event_id)
        if event is None or event_sequence is None:
            continue
        for projection in build_projection_groups(
            checkpoint=checkpoint,
            event=event,
            event_sequence=event_sequence,
            transaction_id=transaction_id,
            source_turn_index=source_turn_index,
            spawned_records=spawned_records,
            actor_id=actor_ids.get(event_id, ""),
            active_identity_character_ids=active_identity_character_ids,
            active_location_labels=active_location_labels,
        ):
            for viewer_id in projection.viewer_character_ids:
                if viewer_id in eligible_viewer_ids:
                    parts_by_viewer_event[(viewer_id, event_id)] = projection

    grouped: dict[str, VisibleEventProjection] = {}
    perception_rank = {"direct": 0, "indirect": 1, "inferred": 2}
    for viewer_id, entries in buffered_events_by_pov.items():
        if viewer_id not in eligible_viewer_ids:
            continue
        ordered_entries = sorted(
            entries,
            key=lambda entry: (entry.visible_at_s, entry.event_sequence),
        )
        parts = [
            part
            for entry in ordered_entries
            if (part := parts_by_viewer_event.get((viewer_id, entry.event_id)))
            is not None
        ]
        if not parts:
            continue
        start_s = min(part.effective_at_s for part in parts)
        end_s = max(
            part.effective_at_s + part.duration_s for part in parts
        )
        facts = tuple(
            (
                text,
                max(0, part.effective_at_s - start_s + offset),
                duration,
            )
            for part in parts
            for text, offset, duration in part.visible_facts
        )
        characters = {
            character.character_id: character
            for part in parts
            for character in part.characters
        }
        references = {
            reference.reference_id: reference
            for part in parts
            for reference in part.reference_options
        }
        anchor = parts[-1]
        perception_level = max(
            (part.perception_level for part in parts),
            key=lambda level: perception_rank.get(level, 2),
        )
        projection = VisibleEventProjection(
            **{
                **anchor.__dict__,
                "viewer_character_ids": (viewer_id,),
                "perception_level": perception_level,
                "effective_at_s": start_s,
                "duration_s": max(0, end_s - start_s),
                "visible_facts": facts,
                "characters": tuple(characters.values()),
                "reference_options": tuple(references.values()),
                "presentation_mode": (
                    checkpoint.session.config.settings.presentation_mode
                ),
            }
        )
        key = projection.grouping_key()
        prior = grouped.get(key)
        if prior is None:
            grouped[key] = projection
            continue
        grouped[key] = VisibleEventProjection(
            **{
                **prior.__dict__,
                "viewer_character_ids": tuple(dict.fromkeys((
                    *prior.viewer_character_ids,
                    viewer_id,
                ))),
            }
        )
    return list(grouped.values())


class ImageDirector:
    def __init__(
        self,
        client: LLMClient,
        prompt_manager: object,
        *,
        max_requests: int = 6,
        max_subjects: int = 4,
        max_scene_prompt_chars: int = 2_000,
        max_references: int = 4,
        generation_modes: Sequence[ImageGenerationMode] = ("compose",),
    ) -> None:
        self.client = client
        self.prompt_manager = prompt_manager
        self.max_requests = max(0, max_requests)
        self.max_subjects = max(1, max_subjects)
        self.max_scene_prompt_chars = max(1, max_scene_prompt_chars)
        self.max_references = max(0, max_references)
        self.generation_modes = tuple(dict.fromkeys(generation_modes))
        if not self.generation_modes:
            raise ValueError("image director needs at least one generation mode")

    async def decide(
        self,
        projection: VisibleEventProjection,
        *,
        stage_context: Sequence[str] = (),
    ) -> ImageDirectorOutput:
        visual_novel = projection.presentation_mode == "visual_novel"
        messages = self.prompt_manager.render_messages(
            "image_director_visual_novel" if visual_novel else "image_director",
            story_block=_story_block(projection),
            visible_event_block=_visible_event_block(projection),
            perception_level=projection.perception_level,
            public_characters_block=(
                "\n".join(
                    character.prompt_line()
                    for character in projection.characters
                )
                or "No named character has usable public visual metadata."
            ),
            visual_references_block=(
                "\n".join(
                    reference.prompt_line()
                    for reference in projection.reference_options
                )
                or "None."
            ),
            generation_modes_block="\n".join(
                f"- {mode}" for mode in self.generation_modes
            ),
            stage_context_block=(
                "\n".join(
                    f"- {_safe_text(item, 500)}"
                    for item in stage_context
                    if _safe_text(item, 500)
                )
                or "None."
            ),
            max_requests=self.max_requests,
            max_subjects=self.max_subjects,
            max_references=self.max_references,
        )
        identity_retry_rule = (
            "For a visual-novel stage, use reuse or clear with no request, "
            "or replace with exactly one non-portrait request whose subjects "
            "already have identity references."
            if visual_novel
            else (
                "New named characters without identity references need "
                "individual portrait requests before any group, action, "
                "establishing, or detail request includes them as subjects."
            )
        )
        for attempt in range(2):
            response = await self.client.complete(
                role="image_director",
                messages=messages,
                response_model=ImageDirectorOutput,
                temperature=0.3,
                max_tokens=2_000,
                cache=True,
            )
            output: ImageDirectorOutput = response.parsed
            try:
                self.validate_output(projection, output)
            except ValueError as exc:
                if attempt:
                    raise
                messages = [
                    *messages,
                    {
                        "role": "assistant",
                        "content": output.model_dump_json(),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Return corrected JSON only. The previous response "
                            f"violated this visual contract: {exc}. Respect the "
                            "request/subject limits, select only allowed ids, "
                            "generation modes, and visual reference ids, "
                            "and never name an anonymous or omitted character "
                            "in either subjects or scene_prompt. Do not request "
                            "visible text, symbols, cards, windows, HUD, "
                            "interface overlays, speech bubbles, panel borders, "
                            "collages, lineups, or chibi substitutions. "
                            f"{identity_retry_rule}"
                        ),
                    },
                ]
                continue
            return output
        raise RuntimeError("unreachable image-director retry state")

    def validate_output(
        self,
        projection: VisibleEventProjection,
        output: ImageDirectorOutput,
    ) -> None:
        if projection.presentation_mode == "visual_novel":
            self._validate_visual_novel_stage(output)
        elif output.stage_action != "independent":
            raise ValueError(
                "non-visual-novel direction must use independent stage_action"
            )
        if len(output.requests) > self.max_requests:
            raise ValueError(
                f"image director returned {len(output.requests)} requests; "
                f"maximum is {self.max_requests}"
            )
        allowed = {
            character.character_id: character
            for character in projection.characters
            if character.depiction_policy == "normal"
        }
        allowed_references = {
            reference.reference_id: reference
            for reference in projection.reference_options
        }
        new_unanchored = {
            character.character_id
            for character in allowed.values()
            if character.is_new_character and not character.has_identity_reference
        }
        portrait_subjects = {
            request.subject_character_ids[0]
            for request in output.requests
            if request.kind == "portrait"
            and len(request.subject_character_ids) == 1
        }
        if (
            projection.presentation_mode != "visual_novel"
            and len(new_unanchored) <= self.max_requests
        ):
            missing_portraits = sorted(new_unanchored - portrait_subjects)
            if missing_portraits:
                raise ValueError(
                    "new named characters require individual first portraits: "
                    + ", ".join(missing_portraits)
                )
        anchored_this_output: set[str] = set()
        restricted = [
            character
            for character in projection.characters
            if character.depiction_policy != "normal"
        ]
        for request in output.requests:
            if request.generation_mode not in self.generation_modes:
                raise ValueError(
                    "image director selected unavailable generation mode: "
                    + request.generation_mode
                )
            if len(request.reference_ids) > self.max_references:
                raise ValueError("image director reference count exceeds limit")
            unknown_references = sorted(
                set(request.reference_ids) - set(allowed_references)
            )
            if unknown_references:
                raise ValueError(
                    "image director selected unavailable visual reference ids: "
                    + ", ".join(unknown_references)
                )
            if request.generation_mode == "edit" and not request.reference_ids:
                raise ValueError("edit generation requires a selected reference")
            if request.generation_mode == "edit" and len(
                request.reference_ids
            ) > 3:
                raise ValueError("edit generation accepts at most 3 references")
            unrelated = [
                reference_id
                for reference_id in request.reference_ids
                if allowed_references[reference_id].scope == "character"
                and allowed_references[reference_id].scope_id
                not in request.subject_character_ids
            ]
            if unrelated:
                raise ValueError(
                    "selected character references must belong to request subjects: "
                    + ", ".join(unrelated)
                )
            if len(request.scene_prompt) > self.max_scene_prompt_chars:
                raise ValueError("image director scene_prompt exceeds limit")
            if len(request.subject_character_ids) > self.max_subjects:
                raise ValueError("image director subject count exceeds limit")
            if _FORBIDDEN_RENDERED_TEXT_RE.search(request.scene_prompt):
                raise ValueError(
                    "scene_prompt requests rendered text, UI, or card imagery"
                )
            unknown = [
                character_id
                for character_id in request.subject_character_ids
                if character_id not in allowed
            ]
            if unknown:
                raise ValueError(
                    "image director selected unavailable or non-depictable "
                    f"character ids: {', '.join(unknown)}"
                )
            named_restricted = [
                character.character_id
                for character in restricted
                if text_names_public_character(
                    request.scene_prompt,
                    character,
                )
            ]
            if named_restricted:
                raise ValueError(
                    "image director named anonymous or omitted characters: "
                    + ", ".join(named_restricted)
                )
            if request.kind == "portrait" and len(
                request.subject_character_ids
            ) != 1:
                raise ValueError("portrait requests require exactly one subject")
            unanchored_subjects = [
                character_id
                for character_id in request.subject_character_ids
                if character_id in new_unanchored
                and character_id not in anchored_this_output
                and request.kind != "portrait"
            ]
            if unanchored_subjects:
                raise ValueError(
                    "new named characters must be portrait-anchored before "
                    "scene or group requests include them: "
                    + ", ".join(unanchored_subjects)
                )
            if (
                request.kind == "portrait"
                and len(request.subject_character_ids) == 1
            ):
                anchored_this_output.add(request.subject_character_ids[0])
            if request.kind == "group_portrait" and len(
                request.subject_character_ids
            ) < 2:
                raise ValueError(
                    "group_portrait requests require at least two subjects"
                )

    @staticmethod
    def _validate_visual_novel_stage(output: ImageDirectorOutput) -> None:
        if output.stage_action == "independent":
            raise ValueError(
                "visual-novel direction requires reuse, replace, or clear"
            )
        if output.stage_action in {"reuse", "clear"}:
            if output.requests:
                raise ValueError(
                    f"{output.stage_action} stage transitions require no requests"
                )
            return
        if len(output.requests) != 1:
            raise ValueError(
                "replace stage transitions require exactly one scene request"
            )
        if output.requests[0].kind == "portrait":
            raise ValueError("visual-novel stage replacements cannot be portraits")


def _selectable_reference_options(
    *,
    checkpoint: CheckpointFile,
    characters: Sequence[PublicCharacterVisual],
    location_label: str,
    active_identity_character_ids: set[str],
    active_location_labels: set[str],
) -> tuple[SelectableVisualReference, ...]:
    visible_character_ids = {
        character.character_id for character in characters
    }
    selected_location_ids = set(
        checkpoint.location_visual_reference_ids.get(location_label, [])
    )
    result: list[SelectableVisualReference] = []
    for reference in checkpoint.reviewed_visual_references:
        if not reference.diffusion_authorized:
            continue
        if reference.scope == "character":
            if (
                reference.scope_id not in visible_character_ids
                or reference.scope_id not in active_identity_character_ids
            ):
                continue
        elif (
            location_label not in active_location_labels
            or reference.reference_id not in selected_location_ids
        ):
            continue
        result.append(
            SelectableVisualReference(
                reference_id=_safe_identifier(reference.reference_id),
                scope=reference.scope,
                scope_id=_safe_identifier(reference.scope_id),
                selection_hint=_safe_text(reference.selection_hint, 500),
            )
        )
    return tuple(result)


def _public_character_projection(
    *,
    checkpoint: CheckpointFile,
    facts: Sequence[tuple[str, int, int]],
    actor_id: str,
    viewer_character_id: str,
    by_id: dict[str, CharacterRecord],
    new_ids: set[str],
    active_identity_character_ids: set[str],
) -> tuple[PublicCharacterVisual, ...]:
    visible_text = " ".join(text for text, _, _ in facts)
    # The acting character is safe implicit context only for their own POV.
    # Other observers get that identity only when their visible facts name it;
    # an indirect or anonymous fact must not disclose the engine-known actor.
    relevant_ids = (
        [actor_id]
        if actor_id and actor_id == viewer_character_id
        else []
    )
    for character in by_id.values():
        if _text_names_character(visible_text, character):
            relevant_ids.append(character.character_id)
    result: list[PublicCharacterVisual] = []
    for character_id in dict.fromkeys(relevant_ids):
        character = by_id.get(character_id)
        if character is None:
            continue
        result.append(PublicCharacterVisual(
            character_id=_safe_identifier(character.character_id),
            name=_safe_text(character.name, 200),
            public_role=_safe_text(character.public_sheet.role, 300),
            appearance=_safe_text(character.public_sheet.appearance, 600),
            default_loadout=image_loadout_for_character(checkpoint, character),
            depiction_policy=str(character.visuals.depiction_policy),
            is_new_character=character_id in new_ids,
            has_identity_reference=(
                character_id in active_identity_character_ids
            ),
            is_playable=bool(character.is_playable),
            recurring_actor=bool(
                character.private_state.intentions_enabled
            ),
        ))
    return tuple(result)


def image_loadout_for_character(
    checkpoint: CheckpointFile,
    character: CharacterRecord,
) -> str:
    """Return the current loadout allowed in image-direction context."""
    ruleset_id = str(
        getattr(
            getattr(checkpoint.session.config, "settings", None),
            "ruleset_id",
            "",
        )
        or ""
    )
    mechanics = character.mechanics
    if (
        ruleset_id == _ONE_STAR_RULESET_ID
        and isinstance(mechanics, dict)
        and mechanics.get(_ONE_STAR_HERO_KEY) is not None
    ):
        # The One-Star projection helper filters visible=false entries.  Do
        # not fall back to the stale authored loadout when the live Hero has
        # no visible equipment: that would reintroduce retired/hidden gear.
        # Keep the optional rules adapter out of generic narrative imports.
        from app.engine.one_star_projection import (
            visible_equipped_item_description,
        )

        return _safe_text(visible_equipped_item_description(character), 700)
    return _safe_text(character.visuals.default_loadout, 700)


def _text_names_character(text: str, character: CharacterRecord) -> bool:
    for value in (character.character_id, character.name):
        cleaned = str(value or "").strip()
        if cleaned and re.search(
            rf"(?<!\w){re.escape(cleaned)}(?!\w)",
            text,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def text_names_public_character(
    text: str,
    character: PublicCharacterVisual,
) -> bool:
    for value in (character.character_id, character.name):
        cleaned = str(value or "").strip()
        if cleaned and re.search(
            rf"(?<!\w){re.escape(cleaned)}(?!\w)",
            text,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _story_block(projection: VisibleEventProjection) -> str:
    lines = [
        f"genre: {projection.story_genre or '(unspecified)'}",
        f"era: {projection.story_era or '(unspecified)'}",
        f"tone: {projection.story_tone or '(unspecified)'}",
        f"premise: {projection.story_premise or '(unspecified)'}",
        f"canonical events so far: {projection.canonical_event_count}",
        f"active roster count: {projection.active_roster_count}",
        f"total roster count: {projection.total_roster_count}",
    ]
    return "\n".join(lines)


def _visible_event_block(projection: VisibleEventProjection) -> str:
    lines = [
        f"event time: {projection.effective_at_s}s",
        f"event duration: {projection.duration_s}s",
        (
            "has_location_reference=yes"
            if projection.has_location_reference
            else "has_location_reference=no"
        ),
    ]
    for text, offset, duration in projection.visible_facts:
        lines.append(
            f"- +{offset}s, duration {duration}s: {text}"
        )
    return "\n".join(lines)


def _visible_location_label(
    *,
    checkpoint: CheckpointFile,
    event: EventRouterOutput,
    viewer_character_id: str,
) -> str:
    for update in reversed(event.location_updates):
        if update.character_id == viewer_character_id:
            return _safe_identifier(update.location_label)
    for wake in reversed(event.activate):
        if wake.character_id == viewer_character_id:
            return _safe_identifier(wake.location_label)
    character = next(
        (
            item
            for item in checkpoint.characters
            if item.character_id == viewer_character_id
        ),
        None,
    )
    if character is None:
        return ""
    return _safe_identifier(character.location)


def _safe_text(value: object, max_chars: int) -> str:
    text = strip_terminal_control(
        redact_imported_asset_text(str(value or ""))
    )
    text = " ".join(text.split())
    return text[:max_chars].rstrip()


def _safe_identifier(value: object) -> str:
    return _safe_text(value, 200)


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
