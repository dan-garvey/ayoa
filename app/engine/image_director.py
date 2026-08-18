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
from app.schemas.image_director import ImageDirectorOutput
from app.schemas.state import SessionState, StorySetting, WorldState


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
    engine_visual_style: str = ""
    delivery_kind: str = "discord"
    viewer_delivery_bindings: tuple[tuple[str, str], ...] = ()
    # Engine-only opaque key used after the director returns. The label is
    # deliberately omitted from rendered LLM messages; the director sees only
    # has_location_reference plus public observable facts.
    engine_location_label: str = ""
    has_location_reference: bool = False

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
            "story_genre": self.story_genre,
            "story_era": self.story_era,
            "story_tone": self.story_tone,
            "story_premise": self.story_premise,
            "canonical_event_count": self.canonical_event_count,
            "active_roster_count": self.active_roster_count,
            "total_roster_count": self.total_roster_count,
            "engine_visual_style": self.engine_visual_style,
            "delivery_kind": self.delivery_kind,
            "viewer_delivery_bindings": [
                list(item) for item in self.viewer_delivery_bindings
            ],
            "engine_location_label": self.engine_location_label,
            "has_location_reference": self.has_location_reference,
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
            story_genre=str(value["story_genre"]),
            story_era=str(value["story_era"]),
            story_tone=str(value["story_tone"]),
            story_premise=str(value["story_premise"]),
            canonical_event_count=int(value["canonical_event_count"]),
            active_roster_count=int(value["active_roster_count"]),
            total_roster_count=int(value["total_roster_count"]),
            engine_visual_style=str(value.get("engine_visual_style") or ""),
            delivery_kind=str(value.get("delivery_kind") or "discord"),
            viewer_delivery_bindings=tuple(
                (str(item[0]), str(item[1]))
                for item in value.get("viewer_delivery_bindings", [])  # type: ignore[union-attr]
            ),
            engine_location_label=str(
                value.get("engine_location_label") or ""
            ),
            has_location_reference=bool(
                value.get("has_location_reference", False)
            ),
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
) -> CheckpointFile:
    """Copy only bounded public state needed by asynchronous projection."""

    setting = checkpoint.world_state.setting
    return CheckpointFile(
        session=SessionState(
            session_id=checkpoint.session.session_id,
            player_character_id=checkpoint.session.player_character_id,
            character_bindings=dict(
                checkpoint.session.character_bindings or {}
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
                        character.visuals.default_loadout,
                        700,
                    ),
                    depiction_policy=character.visuals.depiction_policy,
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
    delivery_kind: str = "discord",
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
        "delivery_kind": delivery_kind,
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
            viewer_character_ids=(viewer_id,),
            perception_level=_OBSERVATION_LEVELS.get(
                observer.observation_level,
                "direct",
            ),
            effective_at_s=max(0, int(event.effective_at_s)),
            duration_s=max(0, int(event.duration_s)),
            visible_facts=facts,
            viewer_delivery_bindings=(
                (
                    viewer_id,
                    str(
                        checkpoint.session.character_bindings.get(
                            viewer_id,
                            "",
                        )
                    ),
                ),
            ),
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
                    "viewer_delivery_bindings": tuple(
                        dict.fromkeys(
                            (
                                *prior.viewer_delivery_bindings,
                                (
                                    viewer_id,
                                    str(
                                        checkpoint.session.character_bindings.get(
                                            viewer_id,
                                            "",
                                        )
                                    ),
                                ),
                            )
                        )
                    ),
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
    ) -> None:
        self.client = client
        self.prompt_manager = prompt_manager
        self.max_requests = max(0, max_requests)
        self.max_subjects = max(1, max_subjects)
        self.max_scene_prompt_chars = max(1, max_scene_prompt_chars)

    async def decide(
        self,
        projection: VisibleEventProjection,
        *,
        recent_illustrations: Sequence[str] = (),
    ) -> ImageDirectorOutput:
        messages = self.prompt_manager.render_messages(
            "image_director",
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
            recent_illustrations_block=(
                "\n".join(
                    f"- {_safe_text(item, 500)}"
                    for item in recent_illustrations
                    if _safe_text(item, 500)
                )
                or "None."
            ),
            max_requests=self.max_requests,
            max_subjects=self.max_subjects,
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
                            "and never name an anonymous or omitted character "
                            "in either subjects or scene_prompt. Do not request "
                            "visible text, symbols, cards, windows, HUD, "
                            "interface overlays, speech bubbles, panel borders, "
                            "collages, lineups, or chibi substitutions. New "
                            "named characters without identity references need "
                            "individual portrait requests before any group, "
                            "action, establishing, or detail request includes "
                            "them as subjects."
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
        if len(new_unanchored) <= self.max_requests:
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


def _public_character_projection(
    *,
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
            default_loadout=_safe_text(
                character.visuals.default_loadout,
                700,
            ),
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
