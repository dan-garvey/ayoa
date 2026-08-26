from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.content_privacy import (
    redact_imported_asset_text,
    sanitize_player_safe_text,
)
from app.schemas.content_pack import SafeAssetRevealPayload
from app.schemas.narrator import (
    VisualNovelPage,
    visual_novel_pages_contain_source_identifiers,
)
from app.schemas.state import DndExperienceAwardDisplay


_ASSET_DELIVERY_REF_RE = re.compile(
    r"^asset://(?P<pack_id>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<asset_id>[A-Za-z0-9][A-Za-z0-9_.-]*)$"
)


class DiceRollDisplay(BaseModel):
    """Runtime UI payload for an already executed D&D roll.

    This is intentionally presentation-shaped rather than prompt-shaped:
    durable mechanics details stay in checkpoint roll transactions, and LLM
    context continues to receive only canonical outcome facts.
    """

    transaction_id: str = ""
    event_id: str = ""
    source: str = ""
    roll_id: str = ""
    actor_id: str = ""
    actor_name: str = ""
    target_id: str = ""
    target_name: str = ""
    label: str = ""
    reason: str = ""
    kind: str = ""
    ability: str = ""
    skill: str = ""
    expression: str = ""
    detail: str = ""
    die_faces: int = 20
    die_values: list[int] = Field(default_factory=list)
    kept_die_values: list[int] = Field(default_factory=list)
    modifier: int = 0
    total: int = 0
    dc: int = 0
    outcome: str = ""
    crit: str = "none"
    damage_raw_total: int = 0
    damage_total: int = 0
    damage_type: str = ""
    damage_expression: str = ""
    damage_detail: str = ""
    target_hp_before: int = 0
    target_hp_after: int = 0
    target_hp_max: int = 0
    target_temp_hp_before: int = 0
    target_temp_hp_after: int = 0
    target_defeat_state: str = ""
    automatic: bool = True


class VisualNovelRenderSegment(BaseModel):
    """One accepted POV beat and its canonical stage provenance."""

    pages: list[VisualNovelPage] = Field(min_length=1)
    rendered_event_ids: list[str] = Field(min_length=1)

    @field_validator("rendered_event_ids")
    @classmethod
    def _require_nonempty_event_ids(cls, values: list[str]) -> list[str]:
        cleaned = [str(value).strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("rendered_event_ids cannot contain blank ids")
        return cleaned

    @model_validator(mode="after")
    def _reject_source_shaped_page_fields(self) -> "VisualNovelRenderSegment":
        if visual_novel_pages_contain_source_identifiers(self.pages):
            raise ValueError(
                "visual-novel response pages cannot expose source-shaped ids"
            )
        return self


class VisualNovelRender(BaseModel):
    """Ordered player-safe beat segments before card composition."""

    segments: list[VisualNovelRenderSegment] = Field(min_length=1)


class TurnResponse(BaseModel):
    session_id: str
    checkpoint_id: str = ""
    turn_index: int = 0
    # Back-compat single-POV render. For v11 beats this mirrors
    # `per_player_renders[acting_character_id]` so legacy callers (the
    # Discord bot's default-channel post, the CLI REPL) keep working
    # unchanged. New multi-POV callers should walk `per_player_renders`
    # directly to deliver each player their own prose.
    output_text: str = ""
    # v11: per-POV beat renders, keyed by character_id. Populated by
    # `run_beat`'s fan-out through `Dispatcher.narrator_compose`; one
    # entry per human POV with at least one observed event this beat,
    # whether local or connected by explicit live perception. Empty when
    # the beat paused mid-Cat-II (see
    # `beat_ended_reason`) or nobody was present to observe.
    per_player_renders: dict[str, str] = Field(default_factory=dict)
    # Structured ADV beat segments for POVs rendered while the session is in
    # visual_novel mode. Each segment carries the canonical event ids used to
    # resolve its own stage plate. ``per_player_renders`` remains the
    # deterministic accessibility/transcript projection of the same pages.
    per_player_visual_novel_renders: dict[str, VisualNovelRender] = Field(
        default_factory=dict
    )
    # Player-safe asset reveal payloads for legacy single-POV callers. For
    # multi-POV delivery this should mirror
    # `per_player_asset_reveals[acting_character_id]`; new callers should use
    # the per-player map so private reveals stay scoped to the intended viewer.
    asset_reveals: list[SafeAssetRevealPayload] = Field(default_factory=list)
    # Authoritative per-POV asset reveal payloads, keyed by character_id. These
    # payloads contain only delivery-safe references and public presentation
    # metadata; private source refs, raw bytes, and DM/source notes never belong
    # on this response surface.
    per_player_asset_reveals: dict[str, list[SafeAssetRevealPayload]] = Field(
        default_factory=dict
    )
    # v11: why the beat stopped. Values come from `BeatResult.ended_reason`
    # (e.g. "awaiting_player_turn", "cat_ii_resolution", "cat_ii_pending",
    # "max_events_cap", "cascade_exhausted"). Callers can detect the
    # Cat II pending state with `not per_player_renders` +
    # `beat_ended_reason == "cat_ii_pending"`.
    beat_ended_reason: str = ""
    # Runtime-only UI affordances keyed by character_id. A value is the
    # canonical event id that opened that character's possible combat reaction
    # window. The event itself stays in checkpoint state; this map tells
    # frontends which rendered POVs should receive a no-LLM "No reaction"
    # button.
    reaction_prompts: dict[str, str] = Field(default_factory=dict)
    # Runtime-only D&D inventory affordances keyed by character_id. Values are
    # pending loot offer ids this character can inspect and claim.
    loot_prompts: dict[str, list[str]] = Field(default_factory=dict)
    # Runtime-only revision affordances keyed by character_id. Values are open
    # commitment ids whose owning player should revise or continue the activity.
    commitment_revision_prompts: dict[str, list[str]] = Field(default_factory=dict)
    # Runtime-only D&D dice-display payloads for rolls completed while building
    # this response. Frontends may animate these before rendering narrator prose.
    dice_rolls: list[DiceRollDisplay] = Field(default_factory=list)
    # Runtime-only D&D XP notices for automatic combat awards. These are
    # frontend display data, not prompt context.
    experience_awards: list[DndExperienceAwardDisplay] = Field(default_factory=list)
    # v11-r7a+: pre-turn resolutions. Stale Cat II closure and resumed
    # automated combat can each produce TurnResponse objects that should be
    # delivered before the actor's own /act result so display order matches
    # story time. Empty in the common case.
    pre_turn_resolutions: list["TurnResponse"] = Field(default_factory=list)
    # NOTE: a `debug: DebugPayload | None` field lived here through
    # v11-r7i. The orchestrator never wrote it, every consumer
    # (Discord latency log, CLI status, playtest summary) was guarded
    # by `if response.debug is not None:` and silently no-op'd. v11-r7j
    # murdered the field per the vestigial-field destruction policy
    # in DESIGN.md §19.1. Per-turn diagnostics live in the engine
    # logger (`turn_loop.router[route]` lines) and per-turn checkpoint
    # files.

    @model_validator(mode="after")
    def _sanitize_player_output_surfaces(self) -> "TurnResponse":
        self.output_text = redact_imported_asset_text(self.output_text)
        self.per_player_renders = {
            str(cid): redact_imported_asset_text(text)
            for cid, text in (self.per_player_renders or {}).items()
        }
        for render in (self.per_player_visual_novel_renders or {}).values():
            for segment in render.segments:
                for page in segment.pages:
                    page.speaker = sanitize_player_safe_text(page.speaker)
                    page.text = redact_imported_asset_text(page.text)
        self.asset_reveals = _safe_asset_payloads(self.asset_reveals)
        self.per_player_asset_reveals = {
            str(cid): _safe_asset_payloads(payloads)
            for cid, payloads in (self.per_player_asset_reveals or {}).items()
        }
        return self


def _safe_asset_payloads(
    payloads: list[SafeAssetRevealPayload],
) -> list[SafeAssetRevealPayload]:
    safe: list[SafeAssetRevealPayload] = []
    for payload in payloads or []:
        if not _safe_delivery_ref(payload):
            continue
        payload.title = sanitize_player_safe_text(payload.title)
        payload.caption = sanitize_player_safe_text(payload.caption)
        payload.alt_text = sanitize_player_safe_text(payload.alt_text)
        safe.append(payload)
    return safe


def _safe_delivery_ref(payload: SafeAssetRevealPayload) -> bool:
    match = _ASSET_DELIVERY_REF_RE.fullmatch((payload.delivery_ref or "").strip())
    return bool(
        match
        and match.group("pack_id") == payload.pack_id.strip()
        and match.group("asset_id") == payload.asset_id.strip()
    )
