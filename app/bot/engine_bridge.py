"""Thin async wrapper around the engine for the Discord bot.

Responsibilities:
- Holds the shared Orchestrator, LLMClient, CheckpointManager, PromptManager.
- Creates a fresh session from a story seed (copies ckpt_0000 into the
  session dir; no auto-binding — players pick a character via /join).
- Runs turns behind a per-session asyncio.Lock so concurrent /act commands
  on the same channel serialize cleanly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from app.engine.character_agent import sanitize_character_public_text
from app.engine.character_manager import CharacterManager, _normalize_router_summary
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.context_builder import (
    is_unbound_player_authored_slot,
    resolve_location_for_character,
)
from app.engine.dnd_combat_access import (
    combatant_name,
    current_combatant,
    checkpoint_active_combat,
)
from app.engine.dnd_cat_ii import (
    complete_pending_player_roll,
    pending_player_rolls,
    roll_transaction_source,
)
from app.engine.dnd_roll_display import dice_roll_display_for_record
from app.engine.dnd_character_import import (
    mechanics_from_snapshot,
    normalize_dndbeyond_export,
)
from app.engine import dnd_experience, dnd_inventory, dnd_runtime, dnd_spatial
from app.engine.frontend_views import (
    CharacterSummary,
    CompletedPendingRoll,
    DndCombatParticipantView,
    DndCombatView,
    DndExperienceAwardResult,
    DndInventoryView,
    DndLootClaimResult,
    DndSheetAttachmentSummary,
    PendingRollPrompt,
    PreparedStoryOnboarding,
    PlayerJoinResult,
    RetryRenderResult,
    RewindResult,
    StorySummary,
    StoryOnboardingChoiceView,
    SessionActivityView,
    TurnHistoryEntry,
)
from app.engine.image_generation import (
    ImageDeliveryTarget,
    ImageGenerationConfig,
    ImageGenerationCoordinator,
)
from app.engine.image_director import ImageDirector
from app.engine.event_image_sidecar import EventImageSidecar
from app.engine.spawn_authoring import SpawnAuthoringCoordinator
from app.engine.model_config_sync import sync_checkpoint_runtime_models
from app.engine.orchestrator import Orchestrator
from app.engine.one_star_hero_cards import (
    OneStarHeroCardError,
    generated_portrait_prewarm_character_ids,
    new_one_star_hero_card_events,
    one_star_hero_card_events_for_render,
    render_one_star_hero_card_boards,
    render_one_star_summon_card_boards,
    render_one_star_summon_reveals,
)
from app.engine.one_star_adapter import (
    one_star_mission_report_recipient_ids,
    one_star_opening_account_owner_actor_id,
)
from app.engine.one_star_mission_reports import (
    OneStarMissionReportError,
    new_one_star_mission_reports,
    one_star_mission_reports_for_render,
    render_one_star_mission_report_accessibility,
    render_one_star_mission_report_boards,
)
from app.engine.prompt_manager import PromptManager
from app.engine.reviewed_visual_references import (
    freeze_story_visual_references,
    load_frozen_visual_references,
    resolve_frozen_visual_reference_media,
    validate_story_visual_references,
)
from app.engine.settings import (
    SETTINGS_BY_KEY,
    get_setting,
    list_settings_view,
    set_setting,
)
from app.engine.turn_loop import broadcast_event, flush_combat_visible_facts
from app.engine.visual_context import forget_visual_introductions_for_character
from app.engine.visual_novel_presentation import (
    VisualNovelCardRenderer,
    VisualNovelDeck,
    VisualNovelDeckSection,
)
from app.engine.visual_novel_sprites import (
    resolve_visual_novel_identity_transition_placement,
    resolve_visual_novel_sprite_placements,
    visual_novel_sprite_identity_transitions,
)
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.characters import (
    ActorFact,
    ActorFactOrigin,
    ActorRecord,
    CharacterRecord,
    CharacterStatus,
    CharacterVisuals,
    PublicSheet,
    is_player_authored_slot,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.dnd_inventory import DndLootOffer
from app.schemas.event_router import (
    EventRouterOutput,
    ObserverEntry,
    empty_commitment_open_signal,
)
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication
from app.schemas.image_generation import ImageDeliveryKind
from app.schemas.narrator import (
    TranscriptEntry,
    VisualNovelPage,
    visual_novel_pages_plain_text,
)
from app.schemas.requests import TurnRequest
from app.schemas.responses import TurnResponse, VisualNovelRender
from app.schemas.state import SlotEntry


def _visual_novel_character_variant_key(
    checkpoint: CheckpointFile,
    character_id: str,
) -> str:
    character = next(
        (
            item
            for item in checkpoint.characters
            if item.character_id == character_id
        ),
        None,
    )
    if character is None:
        return "neutral"
    return (
        character.visuals.visual_novel_presentation.current_variant_key
        or "neutral"
    )


def _loot_claim_message(result: dict[str, Any]) -> str:
    parts: list[str] = []
    item_names = [
        str(item.get("name") or "Item")
        for item in result.get("claimed_items") or []
        if isinstance(item, dict)
    ]
    if item_names:
        parts.append("Claimed " + ", ".join(item_names) + ".")
    currency = _coin_text(result.get("claimed_currency") or {})
    if currency:
        parts.append(f"Claimed {currency}.")
    if result.get("offer_closed"):
        parts.append("The loot offer is now closed.")
    return " ".join(parts) or "Nothing was claimed."


def _loot_split_message(result: dict[str, Any]) -> str:
    shares = result.get("shares") or {}
    lines = []
    for cid, currency in shares.items():
        text = _coin_text(currency)
        if text:
            lines.append(f"{cid}: {text}")
    if not lines:
        return "No currency was split."
    suffix = " The loot offer is now closed." if result.get("offer_closed") else ""
    return "Split currency: " + "; ".join(lines) + "." + suffix


def _loot_offer_by_id(
    ckpt: CheckpointFile,
    offer_id: str,
) -> DndLootOffer | None:
    for offer in ckpt.session.dnd_inventory_offers or []:
        if offer.offer_id == offer_id:
            return offer
    return None


def _loot_source_text(offer: DndLootOffer | None, offer_id: str) -> str:
    if offer is None:
        return f"loot offer {offer_id}"
    if offer.source_label:
        return offer.source_label
    if offer.source_kind:
        return f"{offer.source_kind} loot offer {offer.offer_id}"
    return f"loot offer {offer.offer_id}"


def _loot_item_claim_text(item: dict[str, Any]) -> str:
    name = str(item.get("name") or item.get("source_item_id") or "Item").strip()
    try:
        quantity = int(item.get("quantity", 1) or 1)
    except (TypeError, ValueError):
        quantity = 1
    if quantity > 1:
        return f"{name} x{quantity}"
    return name


def _join_claim_parts(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _loot_claim_router_update(
    *,
    character_id: str,
    offer: DndLootOffer | None,
    offer_id: str,
    result: dict[str, Any],
) -> str:
    claimed: list[str] = []
    for item in result.get("claimed_items") or []:
        if isinstance(item, dict):
            claimed.append(_loot_item_claim_text(item))
    currency = _coin_text(result.get("claimed_currency") or {})
    if currency:
        claimed.append(currency)
    if not claimed:
        return ""
    source = _loot_source_text(offer, offer_id)
    return (
        "Inventory update before the next action: "
        f"{character_id} took {_join_claim_parts(claimed)} from {source}. "
        "Preserve this as established inventory continuity unless the current "
        "input reverses it."
    )


def _loot_split_router_update(
    *,
    character_id: str,
    offer: DndLootOffer | None,
    offer_id: str,
    result: dict[str, Any],
) -> str:
    shares = result.get("shares") or {}
    share_bits: list[str] = []
    for cid, currency in shares.items():
        text = _coin_text(currency)
        if text:
            share_bits.append(f"{cid} received {text}")
    if not share_bits:
        return ""
    source = _loot_source_text(offer, offer_id)
    return (
        "Inventory update before the next action: "
        f"{character_id} split currency from {source}; "
        f"{'; '.join(share_bits)}. Preserve this as established inventory "
        "continuity unless the current input reverses it."
    )


def _coin_text(currency: dict[str, Any]) -> str:
    parts = []
    for key in ("pp", "gp", "ep", "sp", "cp"):
        try:
            value = int(currency.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value:
            parts.append(f"{value} {key}")
    return ", ".join(parts)


logger = logging.getLogger(__name__)


# Dynamic stage work is best-effort presentation. A committed story turn must
# not remain hidden behind a slow or unreachable image worker; director and
# diffusion jobs are durable and continue after this player-facing budget.
_VISUAL_NOVEL_PRESENTATION_WAIT_SECONDS = 20.0


class EngineBridge:
    """Shared engine state for all Discord interactions.

    Stories (synthetic story seed ckpt_0000 files) live under
    `stories_dir`. Player sessions (one dir per session_id, ckpt_NNNN
    grows with turn_index) live under `sessions_dir`. Keeping them in
    separate namespaces means creating a new session from a story does not
    mutate the canonical source seed.

    """

    def __init__(
        self,
        *,
        stories_dir: str | None = None,
        sessions_dir: str | None = None,
        prompts_dir: str = "app/prompts",
        llm_config: LLMConfig | None = None,
        image_generation: ImageGenerationCoordinator | None = None,
        image_sidecar: EventImageSidecar | None = None,
    ):
        self.stories_dir = Path(stories_dir or "app/storage/stories")
        self.sessions_dir = Path(sessions_dir or "app/storage/sessions")
        self.stories_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.client = LLMClient(config=llm_config or LLMConfig.from_env())
        self.checkpoint_mgr = CheckpointManager(save_dir=str(self.sessions_dir))
        self.prompt_mgr = PromptManager(prompts_dir=prompts_dir)
        image_runtime_root = self.sessions_dir.parent / "runtime" / "image_generation"
        self.image_generation = image_generation or ImageGenerationCoordinator(
            sessions_dir=self.sessions_dir,
            config=ImageGenerationConfig.from_environment(
                runtime_root=image_runtime_root,
            ),
            repo_root=Path.cwd(),
        )
        self.visual_novel_renderer = VisualNovelCardRenderer(
            self.sessions_dir.parent / "runtime" / "visual_novel_presentation"
        )
        self._reviewed_visual_binding_signatures: dict[str, str] = {}
        self.checkpoint_mgr.set_load_validator(self._validate_loaded_visual_references)
        self.spawn_authoring = SpawnAuthoringCoordinator(
            CharacterManager(self.client, self.prompt_mgr)
        )
        self.image_sidecar = image_sidecar or EventImageSidecar(
            director=ImageDirector(
                self.client,
                self.prompt_mgr,
                max_requests=self.image_generation.config.max_requests,
                max_subjects=self.image_generation.config.max_subjects,
                max_scene_prompt_chars=(
                    self.image_generation.config.max_scene_prompt_chars
                ),
                max_references=self.image_generation.config.max_references,
                generation_modes=(self.image_generation.supported_generation_modes),
            ),
            generation=self.image_generation,
            spawn_authoring=self.spawn_authoring,
        )
        self.orchestrator = Orchestrator(
            self.client,
            self.checkpoint_mgr,
            self.prompt_mgr,
            image_sink=self.image_sidecar,
            image_generation=self.image_generation,
            spawn_authoring=self.spawn_authoring,
        )
        # One lock per session_id; created lazily.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_mutex = asyncio.Lock()

    async def start(self) -> None:
        await self.image_generation.start()
        await self.image_sidecar.start()

    async def close(self) -> None:
        await self.image_sidecar.close()
        await self.image_generation.close()
        await self.client.close()

    async def prepare_visual_novel_deck(
        self,
        *,
        session_id: str,
        checkpoint_id: str,
        pov_character_id: str,
        render: VisualNovelRender,
    ) -> VisualNovelDeck:
        """Build one ordered deck while preserving each beat's stage plate."""

        checkpoint = self.load_checkpoint(session_id, checkpoint_id)
        previous_checkpoint = self._previous_visual_novel_checkpoint(
            session_id=session_id,
            checkpoint_id=checkpoint_id,
        )
        card_events = one_star_hero_card_events_for_render(
            checkpoint=checkpoint,
            previous_checkpoint=previous_checkpoint,
            viewer_character_id=pov_character_id,
            render=render,
        )
        mission_reports = one_star_mission_reports_for_render(
            checkpoint=checkpoint,
            previous_checkpoint=previous_checkpoint,
            viewer_character_id=pov_character_id,
            render=render,
        )
        required_portrait_ids = generated_portrait_prewarm_character_ids(
            checkpoint=checkpoint,
            viewer_character_id=pov_character_id,
            events=card_events,
        )
        loop = asyncio.get_running_loop()
        presentation_deadline = (
            loop.time() + _VISUAL_NOVEL_PRESENTATION_WAIT_SECONDS
        )
        try:
            await asyncio.wait_for(
                self._prewarm_visual_novel_sprites(
                    session_id=session_id,
                    checkpoint=checkpoint,
                    required_visible_character_ids=required_portrait_ids,
                    await_required=True,
                ),
                timeout=_VISUAL_NOVEL_PRESENTATION_WAIT_SECONDS,
            )
        except TimeoutError:
            logger.info(
                "visual-novel sprite prewarm exceeded presentation budget "
                "session=%s",
                session_id,
            )
        remaining_presentation_wait = max(
            0.0,
            presentation_deadline - loop.time(),
        )
        await self.wait_for_visual_novel_stage_work(
            session_id=session_id,
            renders_by_pov={pov_character_id: render},
            timeout=remaining_presentation_wait,
        )
        pages = tuple(
            page
            for segment in render.segments
            for page in segment.pages
        )
        identity_transitions = (
            visual_novel_sprite_identity_transitions(
                before_checkpoint=previous_checkpoint,
                after_checkpoint=checkpoint,
                viewer_character_id=pov_character_id,
                pages=pages,
            )
            if previous_checkpoint is not None
            else ()
        )
        prior_sprite_set_ids = {
            transition.character_id: transition.before_sprite_set_id
            for transition in identity_transitions
        }
        sections: list[VisualNovelDeckSection] = []
        card_events_by_id = {event.event_id: event for event in card_events}
        inserted_card_event_ids: set[str] = set()
        mission_reports_by_id = {
            report.event_id: report for report in mission_reports
        }
        inserted_mission_report_ids: set[str] = set()
        final_stage_media = None
        for segment_index, segment in enumerate(render.segments, start=1):
            resolution, stage_media = self.image_generation.resolve_visual_novel_stage(
                session_id=session_id,
                pov_character_id=pov_character_id,
                rendered_event_ids=segment.rendered_event_ids,
            )
            if resolution.fallback_reason:
                logger.info(
                    "visual-novel neutral stage session=%s pov=%s segment=%s reason=%s",
                    session_id,
                    pov_character_id,
                    segment_index,
                    resolution.fallback_reason,
                )
            final_stage_media = stage_media
            for page in segment.pages:
                sections.append(
                    VisualNovelDeckSection(
                        pages=(page,),
                        stage_media=stage_media,
                        sprite_placements=(
                            resolve_visual_novel_sprite_placements(
                                checkpoint=checkpoint,
                                viewer_character_id=pov_character_id,
                                page=page,
                                generation=self.image_generation,
                                sprite_set_id_overrides=prior_sprite_set_ids,
                                variant_keys_by_label=(
                                    segment.sprite_variant_keys_by_label
                                ),
                            )
                        ),
                    )
                )
            for rendered_event_id in segment.rendered_event_ids:
                card_event = card_events_by_id.get(rendered_event_id)
                if (
                    card_event is not None
                    and card_event.event_id not in inserted_card_event_ids
                ):
                    reveals = render_one_star_summon_reveals(
                        checkpoint=checkpoint,
                        viewer_character_id=pov_character_id,
                        event=card_event,
                        generation=self.image_generation,
                    )
                    if reveals:
                        summon_boards = render_one_star_summon_card_boards(
                            checkpoint=checkpoint,
                            viewer_character_id=pov_character_id,
                            event=card_event,
                            generation=self.image_generation,
                        )
                        if len(reveals) != len(
                            summon_boards.individual_boards
                        ):
                            raise OneStarHeroCardError(
                                "summon_reveal_card_count_mismatch"
                            )
                        ordered_panels = []
                        for reveal, board in zip(
                            reveals,
                            summon_boards.individual_boards,
                            strict=True,
                        ):
                            ordered_panels.extend((reveal, board))
                        ordered_panels.extend(summon_boards.group_boards)
                    else:
                        ordered_panels = render_one_star_hero_card_boards(
                            checkpoint=checkpoint,
                            viewer_character_id=pov_character_id,
                            event=card_event,
                            generation=self.image_generation,
                        )
                    for panel in ordered_panels:
                        sections.append(VisualNovelDeckSection(
                            pages=(VisualNovelPage(
                                kind="narration",
                                text=panel.accessible_text,
                            ),),
                            stage_media=panel.media,
                            card_style="system_panel",
                        ))
                    inserted_card_event_ids.add(card_event.event_id)
                mission_report = mission_reports_by_id.get(
                    rendered_event_id
                )
                if (
                    mission_report is not None
                    and mission_report.event_id
                    not in inserted_mission_report_ids
                ):
                    for board in render_one_star_mission_report_boards(
                        checkpoint=checkpoint,
                        report=mission_report,
                    ):
                        sections.append(VisualNovelDeckSection(
                            pages=(VisualNovelPage(
                                kind="narration",
                                text=board.accessible_text,
                            ),),
                            stage_media=board.media,
                            card_style="system_panel",
                        ))
                    inserted_mission_report_ids.add(
                        mission_report.event_id
                    )

        if previous_checkpoint is not None:
            for transition in identity_transitions:
                before_placement = (
                    resolve_visual_novel_identity_transition_placement(
                        checkpoint=previous_checkpoint,
                        viewer_character_id=pov_character_id,
                        character_id=transition.character_id,
                        sprite_set_id=transition.before_sprite_set_id,
                        generation=self.image_generation,
                        variant_key=_visual_novel_character_variant_key(
                            previous_checkpoint,
                            transition.character_id,
                        ),
                    )
                )
                after_placement = resolve_visual_novel_identity_transition_placement(
                    checkpoint=checkpoint,
                    viewer_character_id=pov_character_id,
                    character_id=transition.character_id,
                    sprite_set_id=transition.after_sprite_set_id,
                    generation=self.image_generation,
                    variant_key=_visual_novel_character_variant_key(
                        checkpoint,
                        transition.character_id,
                    ),
                )
                if before_placement is None or after_placement is None:
                    logger.warning(
                        "VN identity reveal unavailable session=%s pov=%s character=%s",
                        session_id,
                        pov_character_id,
                        transition.character_id,
                    )
                    continue
                sections.extend((
                    VisualNovelDeckSection(
                        pages=(VisualNovelPage(
                            kind="narration",
                            text="A new identity comes into focus.",
                        ),),
                        stage_media=final_stage_media,
                        sprite_placements=(before_placement,),
                        card_style="identity_flash",
                    ),
                    VisualNovelDeckSection(
                        pages=(VisualNovelPage(
                            kind="narration",
                            text=transition.character_name,
                        ),),
                        stage_media=final_stage_media,
                        sprite_placements=(after_placement,),
                        card_style="identity_reveal",
                    ),
                ))
        return self.visual_novel_renderer.render_deck(sections)

    async def prepare_story_onboarding_deck(
        self,
        session_id: str,
    ) -> PreparedStoryOnboarding | None:
        """Render a story-authored VN tutorial without any model calls."""

        checkpoint = self.load_latest(session_id)
        onboarding = checkpoint.visual_novel_onboarding
        if (
            onboarding is None
            or checkpoint.session.config.settings.presentation_mode
            != "visual_novel"
        ):
            return None

        frozen_stage = self.image_generation.store.reviewed_reference(
            session_id=session_id,
            reference_id=onboarding.stage_reference_id,
        )
        if frozen_stage is None:
            raise RuntimeError(
                "The reviewed visual-novel onboarding stage is unavailable."
            )
        stage_media = resolve_frozen_visual_reference_media(
            frozen_stage,
            runtime_root=self.image_generation.config.runtime_root,
        )

        sections: list[VisualNovelDeckSection] = []
        for authored in onboarding.pages:
            placements = resolve_visual_novel_sprite_placements(
                checkpoint=checkpoint,
                viewer_character_id="",
                page=authored.page,
                generation=self.image_generation,
                variant_keys_by_label=authored.sprite_variant_keys_by_label,
            )
            if len(placements) != len(authored.page.sprites):
                raise RuntimeError(
                    "A reviewed visual-novel onboarding sprite is unavailable."
                )
            sections.append(
                VisualNovelDeckSection(
                    pages=(authored.page,),
                    stage_media=stage_media,
                    sprite_placements=placements,
                )
            )

        characters = {
            character.character_id: character
            for character in checkpoint.characters
        }
        choices = tuple(
            StoryOnboardingChoiceView(
                label=choice.label,
                character_id=choice.character_id,
                character_name=characters[choice.character_id].name,
                player_authored=is_player_authored_slot(
                    characters[choice.character_id]
                ),
            )
            for choice in onboarding.join_choices
        )
        return PreparedStoryOnboarding(
            deck=self.visual_novel_renderer.render_deck(sections),
            join_choices=choices,
        )

    def _previous_visual_novel_checkpoint(
        self,
        *,
        session_id: str,
        checkpoint_id: str,
    ) -> CheckpointFile | None:
        """Load the exact prior committed turn, if one exists."""

        match = re.fullmatch(r"ckpt_(\d+)", checkpoint_id)
        if match is None:
            return None
        current_turn = int(match.group(1))
        prior_turns = [
            turn
            for turn in self.checkpoint_mgr.list_turn_indices(session_id)
            if turn < current_turn
        ]
        if not prior_turns:
            return None
        return self.checkpoint_mgr.load(
            session_id,
            f"ckpt_{max(prior_turns):04d}",
        )

    async def _prewarm_visual_novel_sprites(
        self,
        *,
        session_id: str,
        checkpoint: CheckpointFile | None = None,
        required_visible_character_ids: Iterable[str] = (),
        await_required: bool = False,
    ) -> None:
        """Start optional candidates and await only required Hero-card neutrals."""

        try:
            source = checkpoint or self.checkpoint_mgr.load_latest(session_id)
            required_ids = tuple(dict.fromkeys(
                character_id.strip()
                for character_id in required_visible_character_ids
                if character_id.strip()
            ))
            if required_ids:
                admitted = (
                    await self.image_generation.ensure_visual_novel_sprite_prewarm(
                        source,
                        required_visible_character_ids=required_ids,
                    )
                )
            else:
                admitted = (
                    await self.image_generation.ensure_visual_novel_sprite_prewarm(
                        source
                    )
                )
            if await_required and required_ids:
                required_id_set = set(required_ids)
                terminal_jobs = []
                for job_id in admitted:
                    job = self.image_generation.store.get(job_id)
                    if (
                        job is not None
                        and job.request.character_id in required_id_set
                        and job.request.sprite_variant_key == "neutral"
                    ):
                        terminal_jobs.append(
                            self.image_generation.wait_for_terminal(job_id)
                        )
                if terminal_jobs:
                    await asyncio.gather(*terminal_jobs)
        except Exception:
            logger.exception(
                "visual-novel sprite prewarm failed session=%s",
                session_id,
            )

    def _validate_one_star_hero_card_routing(
        self,
        response: TurnResponse,
    ) -> None:
        if not response.checkpoint_id:
            return
        try:
            checkpoint = self.load_checkpoint(
                response.session_id,
                response.checkpoint_id,
            )
        except FileNotFoundError:
            # Some transport-only responses (and offline callers) may carry a
            # checkpoint label without owning its persisted state. There is no
            # One-Star commit to validate in that case.
            return
        if checkpoint.session.config.settings.presentation_mode != "visual_novel":
            return
        previous = self._previous_visual_novel_checkpoint(
            session_id=response.session_id,
            checkpoint_id=response.checkpoint_id,
        )
        required = new_one_star_hero_card_events(checkpoint, previous)
        if not required:
            return
        from app.engine.one_star_visuals import one_star_visual_novel_config

        configured = one_star_visual_novel_config(checkpoint)
        if configured is None:
            return
        owner_id, _config = configured
        render = response.per_player_visual_novel_renders.get(owner_id)
        if render is None:
            raise OneStarHeroCardError("master_render_missing_card_event")
        one_star_hero_card_events_for_render(
            checkpoint=checkpoint,
            previous_checkpoint=previous,
            viewer_character_id=owner_id,
            render=render,
        )

    def _append_one_star_mission_report_prose(
        self,
        response: TurnResponse,
        *,
        acting_character_id: str = "",
    ) -> None:
        """Attach the deterministic terminal report to bound eligible POVs."""

        if not response.checkpoint_id:
            return
        try:
            checkpoint = self.load_checkpoint(
                response.session_id,
                response.checkpoint_id,
            )
        except FileNotFoundError:
            return
        previous = self._previous_visual_novel_checkpoint(
            session_id=response.session_id,
            checkpoint_id=response.checkpoint_id,
        )
        reports = new_one_star_mission_reports(checkpoint, previous)
        if not reports:
            return
        report_text = "\n\n".join(
            render_one_star_mission_report_accessibility(
                checkpoint=checkpoint,
                report=report,
            )
            for report in reports
        )
        bound_recipient_ids = (
            set(one_star_mission_report_recipient_ids(checkpoint))
            & set(checkpoint.session.character_bindings)
        )
        for character_id in bound_recipient_ids:
            existing = response.per_player_renders.get(
                character_id,
                "",
            ).rstrip()
            response.per_player_renders[character_id] = "\n\n".join(
                value for value in (existing, report_text) if value
            )
        if acting_character_id in bound_recipient_ids:
            response.output_text = response.per_player_renders[
                acting_character_id
            ]

    def _validate_one_star_mission_report_routing(
        self,
        response: TurnResponse,
    ) -> None:
        """Require each bound System POV to carry the report's end segment."""

        if not response.checkpoint_id:
            return
        try:
            checkpoint = self.load_checkpoint(
                response.session_id,
                response.checkpoint_id,
            )
        except FileNotFoundError:
            return
        if checkpoint.session.config.settings.presentation_mode != "visual_novel":
            return
        previous = self._previous_visual_novel_checkpoint(
            session_id=response.session_id,
            checkpoint_id=response.checkpoint_id,
        )
        if not new_one_star_mission_reports(checkpoint, previous):
            return
        bound_recipient_ids = (
            set(one_star_mission_report_recipient_ids(checkpoint))
            & set(checkpoint.session.character_bindings)
        )
        for character_id in bound_recipient_ids:
            render = response.per_player_visual_novel_renders.get(character_id)
            if render is None:
                raise OneStarMissionReportError(
                    "recipient_render_missing_mission_end"
                )
            one_star_mission_reports_for_render(
                checkpoint=checkpoint,
                previous_checkpoint=previous,
                viewer_character_id=character_id,
                render=render,
            )

    async def wait_for_visual_novel_stage_work(
        self,
        *,
        session_id: str,
        renders_by_pov: dict[str, VisualNovelRender],
        timeout: float | None = _VISUAL_NOVEL_PRESENTATION_WAIT_SECONDS,
    ) -> bool:
        """Wait briefly for stage work without blocking committed prose.

        The sidecar's director and diffusion jobs are durable background work.
        Timing out this presentation wait therefore does not cancel or discard
        them; the current deck resolves against whichever reviewed or generated
        stage is already available.
        """

        rendered_event_ids_by_pov: dict[str, list[str]] = {}
        for pov_character_id, render in renders_by_pov.items():
            seen: set[str] = set()
            event_ids: list[str] = []
            for segment in render.segments:
                for event_id in segment.rendered_event_ids:
                    if event_id in seen:
                        continue
                    seen.add(event_id)
                    event_ids.append(event_id)
            rendered_event_ids_by_pov[pov_character_id] = event_ids

        timeout = None if timeout is None else max(0.0, float(timeout))
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        try:
            await asyncio.wait_for(
                self.image_sidecar.wait_for_stage_discovery(session_id),
                timeout=timeout,
            )
        except TimeoutError:
            logger.info(
                "visual-novel stage discovery exceeded presentation budget "
                "session=%s",
                session_id,
            )
            return False

        remaining = None if deadline is None else max(0.0, deadline - loop.time())
        ready = await self.image_generation.wait_for_render_images(
            session_id=session_id,
            rendered_event_ids_by_pov=rendered_event_ids_by_pov,
            timeout=remaining,
        )
        if not ready:
            logger.info(
                "visual-novel stage generation exceeded presentation budget "
                "session=%s",
                session_id,
            )
        return ready

    def load_visual_novel_deck(self, deck_id: str) -> VisualNovelDeck | None:
        return self.visual_novel_renderer.load_deck(deck_id)

    def _validate_loaded_visual_references(
        self,
        checkpoint: CheckpointFile,
        checkpoint_path: Path,
    ) -> None:
        checkpoint_ids = self.checkpoint_mgr.list_checkpoints(
            checkpoint.session.session_id
        )
        if not checkpoint_ids or checkpoint_path.stem != checkpoint_ids[-1]:
            # Historical reads must not replace the live session's runtime
            # bindings. Rewind validates its target before deleting anything.
            return
        signature = self._reviewed_visual_binding_signature(checkpoint)
        if (
            self._reviewed_visual_binding_signatures.get(checkpoint.session.session_id)
            == signature
        ):
            return
        frozen_references = load_frozen_visual_references(
            checkpoint,
            runtime_root=self.image_generation.config.runtime_root,
        )
        self.image_generation.register_reviewed_visual_references(
            checkpoint=checkpoint,
            frozen_references=frozen_references,
        )
        self._reviewed_visual_binding_signatures[checkpoint.session.session_id] = (
            signature
        )

    @staticmethod
    def _reviewed_visual_binding_signature(
        checkpoint: CheckpointFile,
    ) -> str:
        payload = {
            "registry": [
                reference.model_dump(mode="json")
                for reference in checkpoint.reviewed_visual_references
            ],
            "identities": [
                (
                    character.character_id,
                    character.visuals.identity_reference_id,
                )
                for character in checkpoint.characters
            ],
            "locations": checkpoint.location_visual_reference_ids,
            "sprite_sets": [
                sprite_set.model_dump(mode="json")
                for sprite_set in checkpoint.reviewed_visual_novel_sprite_sets
            ],
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    async def lock_image_identity(
        self,
        *,
        session_id: str,
        candidate_id: str,
    ):
        async with await self._lock_for(session_id):
            candidate = self.image_generation.lock_identity_candidate(
                session_id=session_id,
                candidate_id=candidate_id,
            )
            ckpt = self.load_latest(session_id)
            character = next(
                (
                    item
                    for item in ckpt.characters
                    if item.character_id == candidate.character_id
                ),
                None,
            )
            if character is None:
                raise ValueError(
                    "identity candidate character is no longer in the roster"
                )
            character.visuals.identity_reference_id = candidate.candidate_id
            self.checkpoint_mgr.save(ckpt)
            return candidate

    async def reroll_image_identity(
        self,
        *,
        session_id: str,
        reference_id: str,
        pov_character_id: str,
        delivery_kind: ImageDeliveryKind,
        delivery: dict[str, object],
    ):
        async with await self._lock_for(session_id):
            ckpt = self.load_latest(session_id)
            reference_id = reference_id.strip()
            if not reference_id:
                active = self.image_generation.active_identity_candidate(
                    session_id=session_id,
                    character_id=pov_character_id,
                )
                character = next(
                    (
                        item
                        for item in ckpt.characters
                        if item.character_id == pov_character_id
                    ),
                    None,
                )
                reference_id = (
                    active.candidate_id
                    if active is not None
                    else (
                        character.visuals.identity_reference_id
                        if character is not None
                        else ""
                    )
                )
            if not reference_id:
                raise ValueError("Your character has no identity reference to reroll.")
            checkpoint_path = (
                self.sessions_dir
                / session_id
                / f"ckpt_{ckpt.session.turn_index:04d}.json"
            )
            checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            return await self.image_generation.reroll_identity_reference(
                session_id=session_id,
                reference_id=reference_id,
                delivery_targets=[
                    ImageDeliveryTarget(
                        pov_character_id=pov_character_id,
                        delivery_kind=delivery_kind,
                        delivery=delivery,
                    )
                ],
                checkpoint=ckpt,
                source_checkpoint_sha256=checkpoint_hash,
            )

    # ---- session lifecycle ---------------------------------------------------

    def list_story_ids(self) -> list[str]:
        """Return available story IDs — directories under stories_dir that
        contain a ckpt_0000.json."""
        if not self.stories_dir.exists():
            return []
        return sorted(
            child.name
            for child in self.stories_dir.iterdir()
            if child.is_dir() and (child / "ckpt_0000.json").exists()
        )

    def load_story_ckpt(self, story_id: str) -> CheckpointFile:
        """Load a story seed ckpt_0000, not a live session checkpoint."""
        path = self.stories_dir / story_id / "ckpt_0000.json"
        if not path.exists():
            raise FileNotFoundError(f"Story '{story_id}' not found at {path}")
        checkpoint = CheckpointFile.model_validate_json(path.read_text())
        validate_story_visual_references(
            checkpoint,
            story_dir=path.parent,
        )
        return checkpoint

    def story_summary(self, story_id: str) -> StorySummary:
        checkpoint = self.load_story_ckpt(story_id)
        setting = checkpoint.world_state.setting
        title = setting.title.strip() or story_id.replace("_", " ").title()
        playable_seat_count = sum(
            1
            for character in checkpoint.characters
            if character.is_playable and character.status != CharacterStatus.culled
        )
        return StorySummary(
            story_id=story_id,
            title=title,
            genre=setting.genre,
            premise=setting.premise,
            player_primer=checkpoint.player_primer,
            recommended_players=setting.recommended_players,
            play_guidance=setting.play_guidance,
            playable_seat_count=playable_seat_count,
        )

    def list_story_summaries(
        self,
        *,
        discoverable_only: bool = True,
    ) -> list[StorySummary]:
        summaries: list[StorySummary] = []
        for story_id in self.list_story_ids():
            checkpoint = self.load_story_ckpt(story_id)
            if discoverable_only and not checkpoint.world_state.setting.discoverable:
                continue
            setting = checkpoint.world_state.setting
            summaries.append(
                StorySummary(
                    story_id=story_id,
                    title=setting.title.strip() or story_id.replace("_", " ").title(),
                    genre=setting.genre,
                    premise=setting.premise,
                    player_primer=checkpoint.player_primer,
                    recommended_players=setting.recommended_players,
                    play_guidance=setting.play_guidance,
                    playable_seat_count=sum(
                        1
                        for character in checkpoint.characters
                        if character.is_playable
                        and character.status != CharacterStatus.culled
                    ),
                )
            )
        return summaries

    # ---- session primitives --------------------------------------------------

    def list_session_ids(self) -> list[str]:
        """Return directory names under sessions_dir. Each entry is a
        named save the user can resume."""
        if not self.sessions_dir.exists():
            return []
        return sorted(
            child.name for child in self.sessions_dir.iterdir() if child.is_dir()
        )

    def create_empty_session(self, session_id: str) -> None:
        """Create a session directory with no checkpoint. The caller
        runs /story start to load content into it afterward.
        Raises FileExistsError if the session already exists with ckpts."""
        dst = self.sessions_dir / session_id
        if dst.exists() and any(dst.glob("ckpt_*.json")):
            raise FileExistsError(
                f"Session '{session_id}' already exists with checkpoints. "
                f"Run /session resume or pick a different name."
            )
        dst.mkdir(parents=True, exist_ok=True)
        logger.info("Created empty session %s at %s", session_id, dst)

    def load_story_into_session(
        self,
        session_id: str,
        story_id: str,
    ) -> CheckpointFile:
        """Copy a story seed ckpt_0000 into the named session dir,
        rewriting session_id. No personalize, no auto-bind — the player
        picks characters via /join after. Refuses if the session already
        has a story loaded (run /story delete first)."""
        src = self.stories_dir / story_id / "ckpt_0000.json"
        if not src.exists():
            raise FileNotFoundError(f"Story '{story_id}' not found at {src}")

        dst_dir = self.sessions_dir / session_id
        if not dst_dir.exists():
            raise FileNotFoundError(
                f"Session '{session_id}' does not exist. Run /session start first."
            )
        if any(dst_dir.glob("ckpt_*.json")):
            raise FileExistsError(
                f"Session '{session_id}' already has a story loaded. "
                f"Run /story delete first to unload it."
            )

        self._reviewed_visual_binding_signatures.pop(session_id, None)
        data = json.loads(src.read_text())
        from app.schemas.checkpoint import CURRENT_SCHEMA_VERSION

        version = str(data.get("schema_version", "")).strip()
        if version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Story '{story_id}' has schema_version={version!r}, "
                f"expected {CURRENT_SCHEMA_VERSION!r}. Regenerate the story "
                f"before starting a new session."
            )
        data["session"]["session_id"] = session_id
        data["session"]["story_id"] = story_id
        data.pop("importer_version", None)
        data.pop("import_analysis", None)
        ckpt = CheckpointFile.model_validate(data)
        freeze_story_visual_references(
            ckpt,
            story_dir=src.parent,
            runtime_root=self.image_generation.config.runtime_root,
        )
        sync_checkpoint_runtime_models(ckpt, self.client.config)
        (dst_dir / "ckpt_0000.json").write_text(
            ckpt.model_dump_json(
                indent=2,
                context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
            )
        )
        ckpt = self.checkpoint_mgr.load(session_id, "ckpt_0000")
        if ckpt.session.turn_index == 0:
            ckpt.session.turn_index = 1
        sync_checkpoint_runtime_models(ckpt, self.client.config)
        self.checkpoint_mgr.save(ckpt)
        logger.info("Loaded story %s into session %s", story_id, session_id)
        return ckpt

    def unload_story_from_session(self, session_id: str) -> int:
        """Wipe all checkpoints from a session dir. Returns files removed.
        After this, /story start may be run again to load a different
        story. Bindings and character state are gone — the session is
        an empty container."""
        dst = self.sessions_dir / session_id
        if not dst.exists():
            raise FileNotFoundError(f"Session '{session_id}' does not exist.")
        self.image_generation.store.cancel_session(session_id)
        self._reviewed_visual_binding_signatures.pop(session_id, None)
        removed = 0
        for ckpt in dst.glob("ckpt_*.json"):
            ckpt.unlink()
            removed += 1
        logger.info("Unloaded story from session %s (%d files)", session_id, removed)
        return removed

    def load_latest(self, session_id: str) -> CheckpointFile:
        return self.checkpoint_mgr.load_latest(session_id)

    def load_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
    ) -> CheckpointFile:
        return self.checkpoint_mgr.load(session_id, checkpoint_id)

    def list_checkpoint_turns(self, session_id: str) -> list[int]:
        """Return the integer turn indices of every saved checkpoint for
        this session, sorted ascending. Used by the /rewind command to
        validate the user's target and show the playable range."""
        return self.checkpoint_mgr.list_turn_indices(session_id)

    def turn_history(
        self,
        session_id: str,
        pov_character_id: str,
    ) -> list[TurnHistoryEntry]:
        """Return only the narrator history generated for one player POV.

        Per-POV narrator conversations are already the authoritative rendered
        camera streams. Reconstruct turn ids from checkpoint deltas instead of
        reading the legacy session-global selected-POV transcript.
        """
        if not pov_character_id.strip():
            raise ValueError("History requires a selected character POV.")
        history: list[TurnHistoryEntry] = []
        previous_len = 0
        for turn in self.list_checkpoint_turns(session_id):
            ckpt = self.checkpoint_mgr.load(session_id, f"ckpt_{turn:04d}")
            messages = list(ckpt.narrator_conversations.get(pov_character_id, []) or [])
            if len(messages) < previous_len:
                previous_len = 0
            pending_user = ""
            for message in messages[previous_len:]:
                if message.role == "user":
                    pending_user = (
                        message.content if isinstance(message.content, str) else ""
                    )
                    continue
                if message.role != "assistant":
                    continue
                text = _narrator_history_message_text(message.content)
                if not text:
                    continue
                history.append(
                    TurnHistoryEntry(
                        turn_index=turn,
                        entry=TranscriptEntry(
                            user=pending_user,
                            assistant=text,
                        ),
                    )
                )
                pending_user = ""
            previous_len = len(messages)
        return history

    def preview_rewind(
        self,
        session_id: str,
        target_turn: int,
    ) -> RewindResult:
        """Same validation as `rewind_session`, without mutating disk.
        Used by the Discord confirmation flow: show the user exactly
        what would be deleted before they click Confirm. The
        `deleted_turns` list in the returned result is the *would-be*
        deletion list; `new_latest` is what the latest WOULD be after
        the cull (i.e. == target_turn). On confirm, the frontend calls
        `rewind_session` and the actual cull happens.

        Raises the same exceptions as `rewind_session` for invalid
        inputs, so the frontend's confirmation prompt never advances
        past validation if the target is bogus.
        """
        turns = self.list_checkpoint_turns(session_id)
        if not turns:
            raise FileNotFoundError(
                f"Session '{session_id}' has no checkpoints to rewind."
            )
        latest = turns[-1]
        if target_turn < 0:
            raise ValueError(f"Cannot rewind to turn {target_turn}: must be >= 0.")
        if target_turn not in turns:
            raise ValueError(
                f"Turn {target_turn} has no checkpoint for session "
                f"'{session_id}'. Available: {turns[0]}..{latest}."
            )
        if target_turn >= latest:
            raise ValueError(
                f"Cannot rewind to turn {target_turn}: that's already the "
                f"current state (latest is turn {latest}). Pick an "
                f"earlier turn."
            )
        would_delete = [t for t in turns if t > target_turn]
        ckpt = self.checkpoint_mgr.load(session_id, f"ckpt_{target_turn:04d}")
        actor_id = self._actor_id_after_rewind(ckpt)
        return RewindResult(
            session_id=session_id,
            target_turn=target_turn,
            previous_latest=latest,
            new_latest=target_turn,
            deleted_turns=would_delete,
            location=self._actor_location_after_rewind(ckpt),
            actor_character_id=actor_id,
        )

    async def rewind_session(
        self,
        session_id: str,
        target_turn: int,
    ) -> RewindResult:
        """Rewind a session to the state captured by `ckpt_<target_turn>.json`.

        Mechanism: cull every ckpt_NNNN.json with NNNN > target_turn from
        the session directory. The next call to load_latest then returns
        the target checkpoint, and the orchestrator resumes from there
        as if the deleted turns never happened. No engine-level state
        replay needed — checkpoints are atomic per-turn snapshots
        capturing canonical_events, all rolling conversations,
        world_state, every CharacterRecord (with its pending_observations
        queue), render_buffers, slot/Cat-II state, content_state overlays,
        recap, and bindings.

        What this does NOT touch:

        - SessionMap (Discord channel→session, pov_threads cache). Those
          are session-stable and orthogonal to per-turn state. A user
          who joined between target_turn and now will lose their
          binding (the loaded checkpoint predates their /join), but
          their cached pov_thread row is fine — they just need to
          /join again.
        - The per-session asyncio Lock. We acquire it for the cull so
          a concurrent /act cannot save into the gap, but the lock
          object itself is ephemeral and survives.
        - LLM/Anthropic prompt cache. The first post-rewind LLM call
          will miss cache (different conversation prefix); subsequent
          calls re-warm normally. Cost, not correctness.

        Validation:
        - Target must be >= 0 and present in `list_checkpoint_turns`.
        - Target must be < the current latest (refusing to "rewind to
          the present" prevents a noisy no-op and a stray confirmation
          on the user side).
        - If the session has no checkpoints at all, raise FileNotFoundError.

        Returns a `RewindResult` with the deleted turn list and a
        human-readable summary suitable for confirmation embeds.
        """
        turns = self.list_checkpoint_turns(session_id)
        if not turns:
            raise FileNotFoundError(
                f"Session '{session_id}' has no checkpoints to rewind."
            )
        latest = turns[-1]
        if target_turn < 0:
            raise ValueError(f"Cannot rewind to turn {target_turn}: must be >= 0.")
        if target_turn not in turns:
            raise ValueError(
                f"Turn {target_turn} has no checkpoint for session "
                f"'{session_id}'. Available: {turns[0]}..{latest}."
            )
        if target_turn >= latest:
            raise ValueError(
                f"Cannot rewind to turn {target_turn}: that's already the "
                f"current state (latest is turn {latest}). Pick an "
                f"earlier turn."
            )

        lock = await self._lock_for(session_id)
        async with lock:
            # Re-check inside the lock — a concurrent /act may have
            # advanced past our snapshot's `latest` between read and cull.
            # Re-validating against the current latest catches the
            # rare race where rewind and /act arrive simultaneously.
            current_turns = self.list_checkpoint_turns(session_id)
            if target_turn not in current_turns:
                raise ValueError(
                    f"Turn {target_turn} is no longer available "
                    f"(checkpoint disappeared between validation and cull)."
                )
            current_latest = current_turns[-1]
            if target_turn >= current_latest:
                raise ValueError(
                    f"Cannot rewind to turn {target_turn}: latest is now "
                    f"{current_latest}."
                )
            target_checkpoint = self.checkpoint_mgr.load(
                session_id,
                f"ckpt_{target_turn:04d}",
            )
            target_frozen_references = load_frozen_visual_references(
                target_checkpoint,
                runtime_root=self.image_generation.config.runtime_root,
            )
            target_visual_signature = self._reviewed_visual_binding_signature(
                target_checkpoint
            )
            await self.image_generation.cancel_after(session_id, target_turn)
            deleted = self.checkpoint_mgr.delete_checkpoints_after(
                session_id,
                target_turn,
            )
            new_latest = self.list_checkpoint_turns(session_id)[-1]
            self.image_generation.register_reviewed_visual_references(
                checkpoint=target_checkpoint,
                frozen_references=target_frozen_references,
            )
            self._reviewed_visual_binding_signatures[session_id] = (
                target_visual_signature
            )

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        actor_id = self._actor_id_after_rewind(ckpt)
        return RewindResult(
            session_id=session_id,
            target_turn=target_turn,
            previous_latest=current_latest,
            new_latest=new_latest,
            deleted_turns=deleted,
            location=self._actor_location_after_rewind(ckpt),
            actor_character_id=actor_id,
        )

    def _actor_id_after_rewind(self, ckpt: CheckpointFile) -> str:
        cid = ckpt.session.player_character_id
        if cid:
            return cid
        bindings = ckpt.session.character_bindings or {}
        if len(bindings) == 1:
            return next(iter(bindings.keys()))
        return ""

    def _actor_location_after_rewind(self, ckpt: CheckpointFile) -> str:
        """Best-effort: return the bound actor's location after rewind.

        Empty string if we can't resolve it (no bound character, unknown
        character_id) — the embed just omits the location line.
        """
        cid = self._actor_id_after_rewind(ckpt)
        if not cid:
            return ""
        for c in ckpt.characters:
            if c.character_id == cid:
                return c.location or ""
        return ""

    async def set_character_identity(
        self,
        session_id: str,
        character_id: str,
        *,
        name: str | None = None,
        appearance: str | None = None,
    ) -> CheckpointFile:
        """Update a character's name and/or appearance, serialized on the
        per-session lock so it cannot lose its write to a concurrent /act."""
        async with await self._lock_for(session_id):
            return self._set_character_identity_locked(
                session_id,
                character_id,
                name=name,
                appearance=appearance,
            )

    def _set_character_identity_locked(
        self,
        session_id: str,
        character_id: str,
        *,
        name: str | None = None,
        appearance: str | None = None,
    ) -> CheckpointFile:
        """Update a character's name and/or appearance. Used by /describe
        after takeover so the player's name and look land on the record
        without changing the private actor record. Assumes the per-session
        lock is held."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        target = next(
            (c for c in ckpt.characters if c.character_id == character_id), None
        )
        if target is None:
            raise ValueError(
                f"Character '{character_id}' not found in session '{session_id}'"
            )
        if name is not None and name.strip():
            target.name = name.strip()
        if appearance is not None:
            target.public_sheet.appearance = appearance.strip()
            target.visuals.default_loadout = appearance.strip()
        self.checkpoint_mgr.save(ckpt)
        return ckpt

    async def leave_character(
        self,
        session_id: str,
        user_id: int,
    ) -> str | None:
        """Unified leave endpoint shared by CLI and Discord.

        Synthesizes a sparse actor record from observable rolling conversation
        only when this is an ordinary character without one, then unbinds the
        user. An existing actor record, including one with zero facts, is
        already authoritative and is not replaced.

        Returns the freed character_id, or None if the user had no
        binding. Synthesis errors are logged and swallowed — don't
        block the unbind on a synthesis failure.
        """
        binding = self.get_user_binding(session_id, user_id)
        if binding:
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            target = next(
                (
                    character
                    for character in ckpt.characters
                    if character.character_id == binding
                ),
                None,
            )
            if not is_player_authored_slot(target):
                try:
                    await self.synthesize_actor_record(session_id, binding)
                except Exception:
                    logger.exception(
                        "actor-record synthesis failed for %s; unbinding anyway",
                        binding,
                    )
        return await self.unbind_user(session_id, user_id)

    async def synthesize_actor_record(
        self,
        session_id: str,
        character_id: str,
    ) -> CheckpointFile:
        """Create a missing actor record from observable player-side history.

        This is the handoff path when a player-created ordinary character is
        released. An extant record, including ``facts=[]``, is deliberate and
        remains untouched. The engine owns the offstage scheduling default;
        synthesis authors facts only.
        """
        from pydantic import BaseModel

        class _ActorFactsOutput(BaseModel):
            facts: list[ActorFact]

        # Serialize on the per-session lock, held across the synthesis LLM
        # call (same pattern as the custom-character spawn). Without it this
        # load->await->mutate->save races a concurrent /act or operator edit
        # at the same turn_index and silently drops the actor record or
        # clobbers the committed mutation.
        async with await self._lock_for(session_id):
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            target = next(
                (c for c in ckpt.characters if c.character_id == character_id),
                None,
            )
            if target is None:
                raise ValueError(f"No character '{character_id}' in session.")
            if target.actor is not None:
                logger.info(
                    "Actor record already set on %s; skipping synthesis",
                    character_id,
                )
                return ckpt

            # Historical user messages are prompt snapshots and can contain
            # private observations, adapter state, local context, or account
            # data. Only this actor's assistant-side observable contributions
            # may support a newly synthesized private fact.
            history = ckpt.character_conversations.get(character_id, [])
            convo_snippets = []
            for msg in history[-20:]:
                content = (
                    msg.content if hasattr(msg, "content") else msg.get("content", "")
                )
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if content:
                    role = msg.role if hasattr(msg, "role") else msg.get("role")
                    if role != "assistant":
                        continue
                    public = sanitize_character_public_text(content)
                    snippet = public.strip()
                    if snippet:
                        convo_snippets.append(f"[{role}] {snippet[:500]}")
            history_block = "\n".join(convo_snippets) or "(no rolling conversation yet)"

            messages = [
                {
                    "role": "system",
                    "content": (
                        "<role>\n"
                        "You preserve sparse self-knowledge for a person after "
                        "their time in a scene.\n"
                        "</role>\n\n"
                        "<instructions>\n"
                        "Return only concrete facts supported by the public "
                        "identity and observable conversation. Every fact text "
                        "must be a second-person sentence, such as \"You ...\". "
                        "Use origin lived, witnessed, told, or inferred for how "
                        "the fact entered this person's experience. Keep an "
                        "inference uncertain in its text. Empty facts are valid; "
                        "do not fill space with a behavioral profile, a diagnosis, "
                        "or a complete life story. Do not infer a private fact "
                        "from silence. No commentary outside the JSON.\n"
                        "</instructions>\n\n"
                        "<output_schema>\n"
                        "Respond with ONLY valid JSON: "
                        "{\"facts\":[{\"origin\":\"lived|witnessed|told|inferred\","
                        "\"text\":\"<second-person fact>\"}]}\n"
                        "</output_schema>"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "<character_context>\n"
                        "<public_identity>\n"
                        f"Name: {target.name}\n"
                        f"Role: {target.public_sheet.role}\n"
                        f"Appearance: {target.public_sheet.appearance}\n"
                        f"Faction: {target.public_sheet.faction}\n"
                        f"Public context: {target.public_sheet.public_context}\n"
                        "</public_identity>\n"
                        "</character_context>\n\n"
                        "<recent_conversation>\n"
                        f"{history_block}\n"
                        "</recent_conversation>\n\n"
                        "<task>\n"
                        "Write the sparse actor-facts JSON now.\n"
                        "</task>"
                    ),
                },
            ]
            response = await self.client.complete(
                role="narrator",
                messages=messages,
                temperature=0.5,
                max_tokens=1000,
            )
            out = _parse_model_json(_ActorFactsOutput, response.content)
            target.actor = ActorRecord(
                may_act_offstage=False,
                facts=out.facts,
            )
            self.checkpoint_mgr.save(ckpt)
            logger.info(
                "Synthesized actor record for %s (%d facts)",
                character_id,
                len(target.actor.facts),
            )
            return ckpt

    # ---- character catalog (spoiler-free) ------------------------------------

    def list_session_characters(self, session_id: str) -> list[CharacterSummary]:
        """Spoiler-free roster for the session's current checkpoint, annotated
        with binding state. Used by /story characters when a session exists."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        return _summaries_from_checkpoint(ckpt)

    def session_activity(
        self,
        session_id: str,
        pov_character_id: str = "",
    ) -> SessionActivityView:
        """Return the shared, player-safe activity/status projection."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        characters = {c.character_id: c for c in ckpt.characters}
        bindings = ckpt.session.character_bindings or {}
        viewpoint = characters.get(pov_character_id)
        location = resolve_location_for_character(ckpt, pov_character_id)

        joined_names = tuple(
            character.name
            for character in ckpt.characters
            if character.character_id in bindings
            and character.status != CharacterStatus.culled
        )
        nearby_names = tuple(
            character.name
            for character in ckpt.characters
            if character.character_id != pov_character_id
            and character.status == CharacterStatus.active
            and bool(location)
            and character.location == location
            and not is_unbound_player_authored_slot(ckpt, character)
        )

        requested_ids: list[str] = []
        if ckpt.canonical_events:
            requested_ids = list(
                ckpt.canonical_events[-1].next_output_character_ids or []
            )
        requested_names = tuple(
            characters[character_id].name
            for character_id in requested_ids
            if character_id in characters and character_id in bindings
        )

        if ckpt.session.pending_narrator_render is not None:
            state = "A story update is waiting to finish rendering."
        else:
            combat = checkpoint_active_combat(ckpt)
            combatant = current_combatant(combat) if combat is not None else None
            if combatant is not None:
                state = f"D&D combat: {combatant_name(combatant)} is acting."
            else:
                waiting_names = tuple(
                    characters[character_id].name
                    for character_id, entry in ckpt.session.active_act_slots.items()
                    if entry.reason != "initiator" and character_id in characters
                )
                if waiting_names:
                    state = "Waiting on " + ", ".join(waiting_names) + "."
                elif requested_names:
                    state = (
                        "Requested next: "
                        + ", ".join(requested_names)
                        + " (advisory; any joined player may act)."
                    )
                else:
                    state = "Open table: any joined player may act."

        last_visible_update = ""
        if viewpoint is not None:
            for message in reversed(
                ckpt.narrator_conversations.get(pov_character_id, []) or []
            ):
                if message.role != "assistant":
                    continue
                last_visible_update = _narrator_history_message_text(
                    message.content
                ).strip()
                if last_visible_update:
                    last_visible_update = last_visible_update[:400]
                    break

        ruleset_lines: tuple[str, ...] = ()
        if ckpt.session.config.settings.ruleset_id == "one_star_ascension":
            from app.engine.one_star_projection import one_star_status_lines

            ruleset_lines = one_star_status_lines(
                ckpt,
                viewpoint.character_id if viewpoint is not None else "",
            )

        return SessionActivityView(
            session_id=session_id,
            story_id=ckpt.session.story_id,
            turn_index=ckpt.session.turn_index,
            state=state,
            viewpoint_name=viewpoint.name if viewpoint is not None else "",
            location=location,
            joined_seat_names=joined_names,
            nearby_character_names=nearby_names,
            requested_next_names=requested_names,
            last_visible_update=last_visible_update,
            ruleset_lines=ruleset_lines,
        )

    def one_star_master_command(
        self,
        session_id: str,
        viewpoint_character_id: str,
        command: str,
        *,
        hero_ref: str = "",
    ) -> tuple[str, ...]:
        """Return one adapter-gated, read-only Master ledger projection."""

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        from app.engine.one_star_projection import (
            one_star_master_command_lines,
        )

        return one_star_master_command_lines(
            ckpt,
            viewpoint_character_id,
            command,
            hero_ref=hero_ref,
        )

    async def run_one_star_synthesis_command(
        self,
        session_id: str,
        viewpoint_character_id: str,
        *,
        target_ref: str,
        source_refs: Iterable[str],
    ) -> TurnResponse:
        """Resolve one Master synthesis through the fixed System-result path."""

        clean_source_refs = tuple(
            value.strip() for value in source_refs if value.strip()
        )
        lock = await self._lock_for(session_id)
        async with lock:
            from app.engine.one_star_projection import (
                one_star_synthesis_authoritative_plan,
            )

            pre_turn = await self._resolve_swept_events_locked(session_id)
            response = await self.orchestrator.process_authoritative_result(
                session_id=session_id,
                viewpoint_character_id=viewpoint_character_id,
                plan_builder=lambda checkpoint: (
                    one_star_synthesis_authoritative_plan(
                        checkpoint,
                        viewpoint_character_id,
                        target_ref=target_ref,
                        source_refs=clean_source_refs,
                    )
                ),
            )
            response.pre_turn_resolutions = [
                *pre_turn,
                *(response.pre_turn_resolutions or []),
            ]
            return response

    def list_joinable_characters(self, session_id: str) -> list[CharacterSummary]:
        """Open pre-authored slots surfaced by `/join`.

        Keep this as the single filter used by Discord and CLI entrypoints so
        the interactive picker and text-mode roster do not drift apart.
        """
        return joinable_character_summaries(self.list_session_characters(session_id))

    def list_story_characters(self, story_id: str) -> list[CharacterSummary]:
        """Spoiler-free roster from the story seed (no session needed)."""
        ckpt = self.load_story_ckpt(story_id)
        return _summaries_from_checkpoint(ckpt)

    # ---- bindings ------------------------------------------------------------

    def get_user_binding(self, session_id: str, user_id: int) -> str | None:
        """Return the character_id bound to this Discord user, or None."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        uid = str(user_id)
        for char_id, bound in ckpt.session.character_bindings.items():
            if bound == uid:
                return char_id
        return None

    def get_bound_character_record(
        self,
        session_id: str,
        user_id: int,
        character_id: str | None = None,
    ) -> CharacterRecord:
        """Return a character this user currently controls.

        `character_id` is optional because ordinary Discord users have
        exactly one live binding. If supplied, it must still be one of
        the invoker's bindings; D&D sheets are player-private enough that
        the generic `/sheet` path should not become a roster browser.
        """
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        target_id = self._bound_character_id_for_user(
            ckpt,
            user_id=user_id,
            character_id=character_id,
        )
        target = next((c for c in ckpt.characters if c.character_id == target_id), None)
        if target is None:
            raise ValueError(f"No character '{target_id}' in this session.")
        return target

    def list_inventory(
        self,
        session_id: str,
        user_id: int,
        character_id: str | None = None,
    ) -> DndInventoryView:
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        target_id = self._bound_character_id_for_user(
            ckpt,
            user_id=user_id,
            character_id=character_id,
        )
        target = next((c for c in ckpt.characters if c.character_id == target_id), None)
        if target is None:
            raise ValueError(f"No character '{target_id}' in this session.")
        inventory = dnd_inventory.inventory_view(target)
        return DndInventoryView(
            character_id=target.character_id,
            character_name=target.name,
            items=[
                item
                for item in (inventory.get("items") or [])
                if isinstance(item, dict)
            ],
            currency=inventory.get("currency") or {},
        )

    def award_dnd_experience(
        self,
        session_id: str,
        target: str,
        amount: int,
        *,
        source: str = "",
    ) -> list[DndExperienceAwardResult]:
        if amount <= 0:
            raise ValueError("Experience award must be positive.")
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        target_ids = _dnd_experience_target_ids(ckpt, target)

        results: list[DndExperienceAwardResult] = []
        for target_id in target_ids:
            character = next(
                (c for c in ckpt.characters if c.character_id == target_id),
                None,
            )
            if character is None:
                raise ValueError(f"No character '{target_id}' in this session.")
            view = dnd_experience.award_experience(
                character,
                amount,
                source=source,
                turn_index=ckpt.session.turn_index,
            )
            results.append(_dnd_experience_award_result(character, view))

        self.checkpoint_mgr.save(ckpt)
        return results

    async def award_dnd_experience_locked(
        self,
        session_id: str,
        target: str,
        amount: int,
        *,
        source: str = "",
    ) -> list[DndExperienceAwardResult]:
        bridge_lock = await self._lock_for(session_id)
        async with bridge_lock:
            orchestrator_lock = await self.orchestrator.session_locks.get(session_id)
            async with orchestrator_lock:
                return self.award_dnd_experience(
                    session_id,
                    target,
                    amount,
                    source=source,
                )

    def list_loot_offers(
        self,
        session_id: str,
        user_id: int,
        character_id: str | None = None,
    ) -> list[DndLootOffer]:
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        if dnd_inventory.prune_inventory_offers(ckpt):
            self.checkpoint_mgr.save(ckpt)
        target_id = self._bound_character_id_for_user(
            ckpt,
            user_id=user_id,
            character_id=character_id,
        )
        return dnd_inventory.open_loot_offers_for_character(ckpt, target_id)

    async def claim_loot(
        self,
        *,
        session_id: str,
        user_id: int,
        character_id: str | None,
        offer_id: str,
        item_ids: list[str],
        take_currency: bool = False,
        take_all_available: bool = False,
    ) -> DndLootClaimResult:
        lock = await self._lock_for(session_id)
        async with lock:
            orchestrator_lock = await self.orchestrator.session_locks.get(session_id)
            async with orchestrator_lock:
                ckpt = self.checkpoint_mgr.load_latest(session_id)
                target_id = self._bound_character_id_for_user(
                    ckpt,
                    user_id=user_id,
                    character_id=character_id,
                )
                offer = _loot_offer_by_id(ckpt, offer_id)
                result = dnd_inventory.claim_loot(
                    ckpt,
                    character_id=target_id,
                    offer_id=offer_id,
                    item_ids=item_ids,
                    take_currency=take_currency,
                    take_all_available=take_all_available,
                )
                router_update = _loot_claim_router_update(
                    character_id=target_id,
                    offer=offer,
                    offer_id=offer_id,
                    result=result,
                )
                if router_update:
                    ckpt.session.pending_engine_state_updates.append(
                        router_update,
                    )
                self.checkpoint_mgr.save(ckpt)
        return DndLootClaimResult(
            offer_id=offer_id,
            character_id=result["character_id"],
            claimed_items=result["claimed_items"],
            claimed_currency=result["claimed_currency"],
            shares={},
            offer_closed=bool(result["offer_closed"]),
            message=_loot_claim_message(result),
        )

    async def split_loot_currency(
        self,
        *,
        session_id: str,
        user_id: int,
        offer_id: str,
        character_id: str | None = None,
    ) -> DndLootClaimResult:
        lock = await self._lock_for(session_id)
        async with lock:
            orchestrator_lock = await self.orchestrator.session_locks.get(session_id)
            async with orchestrator_lock:
                ckpt = self.checkpoint_mgr.load_latest(session_id)
                target_id = self._bound_character_id_for_user(
                    ckpt,
                    user_id=user_id,
                    character_id=character_id,
                )
                offer = _loot_offer_by_id(ckpt, offer_id)
                result = dnd_inventory.split_loot_currency(
                    ckpt,
                    offer_id=offer_id,
                    actor_id=target_id,
                )
                router_update = _loot_split_router_update(
                    character_id=target_id,
                    offer=offer,
                    offer_id=offer_id,
                    result=result,
                )
                if router_update:
                    ckpt.session.pending_engine_state_updates.append(
                        router_update,
                    )
                self.checkpoint_mgr.save(ckpt)
        return DndLootClaimResult(
            offer_id=offer_id,
            character_id=target_id,
            claimed_items=[],
            claimed_currency={},
            shares=result["shares"],
            offer_closed=bool(result["offer_closed"]),
            message=_loot_split_message(result),
        )

    async def decline_loot(
        self,
        *,
        session_id: str,
        user_id: int,
        offer_id: str,
        character_id: str | None = None,
    ) -> DndLootClaimResult:
        lock = await self._lock_for(session_id)
        async with lock:
            orchestrator_lock = await self.orchestrator.session_locks.get(session_id)
            async with orchestrator_lock:
                ckpt = self.checkpoint_mgr.load_latest(session_id)
                target_id = self._bound_character_id_for_user(
                    ckpt,
                    user_id=user_id,
                    character_id=character_id,
                )
                result = dnd_inventory.decline_loot(
                    ckpt,
                    character_id=target_id,
                    offer_id=offer_id,
                )
                self.checkpoint_mgr.save(ckpt)
        return DndLootClaimResult(
            offer_id=offer_id,
            character_id=target_id,
            claimed_items=[],
            claimed_currency={},
            shares={},
            offer_closed=bool(result["offer_closed"]),
            message=(
                "Declined the loot offer."
                if not result["offer_closed"]
                else "Declined the loot offer; it is now closed."
            ),
        )

    async def attach_dndbeyond_character_export(
        self,
        session_id: str,
        user_id: int,
        export: dict[str, Any],
        *,
        character_id: str | None = None,
        name_override: str | None = None,
    ) -> DndSheetAttachmentSummary:
        """Attach a D&D Beyond JSON export to the invoker's bound character.

        This is deliberately an attachment, not a character replacement.
        The story-facing `CharacterRecord.name`, appearance, role,
        location, actor record, and conversations stay as they are
        unless `name_override` is explicitly provided. The imported
        D&D identity remains visible inside `mechanics.dnd5e_sheet` for
        sheet display and rules arbitration.
        """
        if not isinstance(export, dict):
            raise ValueError("D&D Beyond export must be a JSON object.")

        override = (name_override or "").strip()
        snapshot = normalize_dndbeyond_export(
            export,
            include_raw_source=True,
        )
        mechanics = mechanics_from_snapshot(snapshot)

        lock = await self._lock_for(session_id)
        async with lock:
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            target_id = self._bound_character_id_for_user(
                ckpt,
                user_id=user_id,
                character_id=character_id,
            )
            target = next(
                (c for c in ckpt.characters if c.character_id == target_id),
                None,
            )
            if target is None:
                raise ValueError(f"No character '{target_id}' in this session.")

            target.mechanics = mechanics
            if override:
                target.name = override

            settings = ckpt.session.config.settings
            settings.ruleset_id = "dnd5e_basic"

            summary = _dnd_attachment_summary(
                character=target,
                snapshot=snapshot,
                settings=settings,
                name_overridden=bool(override),
            )
            class_line = ", ".join(summary.classes) or "unknown class"
            imported = summary.imported_name or "unnamed D&D character"
            equipment_line = _dnd_attachment_equipment_sentence(mechanics)
            ckpt.session.pending_engine_state_updates.append(
                f"D&D sheet attached: {target.character_id} now has "
                f"imported D&D mechanics "
                f"from {imported}; {class_line}, level {summary.total_level}, "
                f"AC {summary.armor_class}, HP "
                f"{summary.hit_points_current}/{summary.hit_points_max}. "
                "D&D rules now govern relevant adjudication. "
                "Use these mechanics for D&D adjudication; preserve the "
                "story identity unless fiction explicitly changes it. "
                f"{equipment_line}"
            )
            self.checkpoint_mgr.save(ckpt)
            logger.info(
                "Attached D&D sheet in %s for user %s: %s <- %s",
                session_id,
                user_id,
                target.character_id,
                imported,
            )
            return summary

    def _bound_character_id_for_user(
        self,
        ckpt: CheckpointFile,
        *,
        user_id: int,
        character_id: str | None = None,
    ) -> str:
        uid = str(user_id)
        bound_ids = [
            cid
            for cid, bound in (ckpt.session.character_bindings or {}).items()
            if bound == uid
        ]
        if not bound_ids:
            raise ValueError("You are not bound to a character in this session.")

        requested = (character_id or "").strip()
        if requested:
            if requested not in bound_ids:
                raise ValueError(
                    "You can only use this with a character you currently control."
                )
            return requested

        if len(bound_ids) > 1:
            raise ValueError(
                "Multiple character bindings found. Provide `character_id`."
            )
        return bound_ids[0]

    def pending_roll_prompts(
        self,
        session_id: str,
        *,
        user_id: int | None = None,
    ) -> list[PendingRollPrompt]:
        """Return Discord-addressable pending player rolls for a session."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        bindings = ckpt.session.character_bindings or {}
        wanted_uid = str(user_id) if user_id is not None else ""
        prompts: list[PendingRollPrompt] = []
        for transaction in ckpt.session.cat_ii_roll_transactions:
            for record in pending_player_rolls(ckpt, event_id=transaction.event_id):
                bound_uid = bindings.get(record.actor_id, "")
                if not bound_uid:
                    continue
                if wanted_uid and bound_uid != wanted_uid:
                    continue
                prompts.append(
                    PendingRollPrompt(
                        session_id=session_id,
                        event_id=transaction.event_id,
                        roll_id=record.roll_id,
                        actor_id=record.actor_id,
                        user_id=bound_uid,
                        label=record.label or "Roll",
                        reason=record.reason,
                    )
                )
        return prompts

    async def complete_pending_roll(
        self,
        *,
        session_id: str,
        event_id: str,
        roll_id: str,
        user_id: int,
    ) -> CompletedPendingRoll:
        """Execute one pending player roll and persist the dice result only.

        This intentionally does not run the router finalize call. Discord can
        show the mechanical result immediately, then continue the Cat II
        resolution as a separate slower step.
        """
        bridge_lock = await self._lock_for(session_id)
        async with bridge_lock:
            return await self._complete_pending_roll_locked(
                session_id=session_id,
                event_id=event_id,
                roll_id=roll_id,
                user_id=user_id,
            )

    async def _complete_pending_roll_locked(
        self,
        *,
        session_id: str,
        event_id: str,
        roll_id: str,
        user_id: int,
    ) -> CompletedPendingRoll:
        """Complete a roll while the EngineBridge session lock is held."""
        actor_id = self.get_user_binding(session_id, user_id)
        if actor_id is None:
            raise ValueError("This Discord user is not bound to a character.")

        orchestrator_lock = await self.orchestrator.session_locks.get(session_id)
        async with orchestrator_lock:
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            sync_checkpoint_runtime_models(ckpt, self.client.config)
            bindings = ckpt.session.character_bindings or {}
            if bindings.get(actor_id, "") != str(user_id):
                raise ValueError(
                    "That roll is not pending for your character. Use "
                    "/combat status to see the current state."
                )
            source = roll_transaction_source(ckpt, event_id)
            if (
                source == "combat"
                and getattr(ckpt.session, "active_combat", None) is None
            ):
                raise ValueError(
                    "That combat roll is no longer active. Use /combat "
                    "status to see the current state."
                )
            if source != "combat" and not any(
                evt.event_id == event_id for evt in ckpt.session.open_cat_ii_events
            ):
                raise ValueError("That contested action is no longer open.")

            pending_for_actor = pending_player_rolls(
                ckpt,
                event_id=event_id,
                actor_id=actor_id,
            )
            record = next(
                (r for r in pending_for_actor if r.roll_id == roll_id),
                None,
            )
            if record is None:
                if source == "combat":
                    raise ValueError(
                        "That combat roll is no longer active. Use /combat "
                        "status to see the current state."
                    )
                raise ValueError(
                    "That roll is not pending for your character. Use /combat "
                    "status to see the current state."
                )

            completed = complete_pending_player_roll(
                ckpt,
                event_id=event_id,
                roll_id=roll_id,
                completed_by_user_id=str(user_id),
            )
            remaining = pending_player_rolls(ckpt, event_id=event_id)
            if not remaining:
                ckpt.session.active_act_slots[actor_id] = SlotEntry(
                    reason="cat_ii_roll",
                    cat_ii_event_id=event_id,
                    claimed_at=datetime.now(timezone.utc).isoformat(),
                )
            self.checkpoint_mgr.save(ckpt)

        result = completed.result or {}
        transaction = next(
            (
                txn
                for txn in ckpt.session.cat_ii_roll_transactions
                if txn.event_id == event_id
                and any(r.roll_id == completed.roll_id for r in txn.rolls)
            ),
            None,
        )
        display = (
            dice_roll_display_for_record(ckpt, transaction, completed)
            if transaction is not None
            else None
        )
        return CompletedPendingRoll(
            session_id=session_id,
            event_id=event_id,
            roll_id=completed.roll_id,
            actor_id=completed.actor_id,
            user_id=str(user_id),
            label=completed.label or "Roll",
            reason=completed.reason,
            expression=str(result.get("expression", "")),
            total=int(result.get("total", 0)),
            detail=str(result.get("detail", "")),
            crit=str(result.get("crit", "none")),
            remaining_pending_rolls=len(remaining),
            die_values=tuple(display.die_values if display else ()),
            kept_die_values=tuple(
                display.kept_die_values if display else (),
            ),
            modifier=display.modifier if display else 0,
            dc=display.dc if display else 0,
            outcome=display.outcome if display else "",
            target_id=display.target_id if display else "",
            target_name=display.target_name if display else "",
            damage_total=display.damage_total if display else 0,
            damage_type=display.damage_type if display else "",
            damage_detail=display.damage_detail if display else "",
        )

    async def continue_pending_roll(
        self,
        *,
        session_id: str,
        event_id: str,
        actor_id: str,
    ) -> TurnResponse:
        """Finalize a Cat II after a prior complete_pending_roll call."""
        bridge_lock = await self._lock_for(session_id)
        async with bridge_lock:
            return await self.orchestrator.continue_cat_ii_after_roll(
                session_id=session_id,
                event_id=event_id,
                actor_id=actor_id,
            )

    async def bind_user(
        self,
        session_id: str,
        user_id: int,
        character_id: str,
    ) -> CheckpointFile:
        """Bind a Discord user to a roster character.

        Serialized on the per-session lock (the same lock `run_turn` holds)
        so a concurrent /act cannot clobber the binding with a stale
        checkpoint write, and two simultaneous /joins cannot lose a binding.
        """
        async with await self._lock_for(session_id):
            return self._bind_user_locked(session_id, user_id, character_id)

    def _bind_user_locked(
        self,
        session_id: str,
        user_id: int,
        character_id: str,
    ) -> CheckpointFile:
        """Bind a Discord user to a roster character. Assumes the caller
        holds the per-session lock.

        Refuses if the user already has a different binding, if the character
        doesn't exist, if the character is culled, or if another user is
        already bound to it. Dormant characters are allowed — binding doesn't
        wake them; the fiction decides reactivation.
        """
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        self._bind_user_in_checkpoint(ckpt, user_id, character_id)
        self.checkpoint_mgr.save(ckpt)
        return ckpt

    @staticmethod
    def _bind_user_in_checkpoint(
        ckpt: CheckpointFile,
        user_id: int,
        character_id: str,
    ) -> CharacterRecord:
        """Validate and apply a binding without saving the checkpoint."""
        uid = str(user_id)

        target = next(
            (c for c in ckpt.characters if c.character_id == character_id), None
        )
        if target is None:
            raise ValueError(f"No character '{character_id}' in this session.")
        if target.status.value == "culled":
            raise ValueError(
                f"Character '{target.name}' is no longer in the story (culled)."
            )

        existing_for_char = ckpt.session.character_bindings.get(character_id)
        if existing_for_char and existing_for_char != uid:
            raise ValueError(
                f"Character '{target.name}' is already bound to another player."
            )

        existing_for_user = next(
            (
                cid
                for cid, bound in ckpt.session.character_bindings.items()
                if bound == uid and cid != character_id
            ),
            None,
        )
        if existing_for_user:
            raise ValueError(
                f"You are already bound to '{existing_for_user}'. "
                f"Run /leave first if you want to switch."
            )

        ckpt.session.character_bindings[character_id] = uid
        return target

    async def claim_player_character(
        self,
        session_id: str,
        character_id: str,
        user_id: int,
        *,
        name: str = "",
        appearance: str = "",
    ) -> CheckpointFile:
        """Strict player-facing claim shared by Discord and the CLI.

        Unlike the internal takeover helper, this accepts only authored
        playable seats. Player-authored seats require identity input and apply
        identity plus binding in one checkpoint write, so a failed modal or
        command can never leave a half-authored bound character behind.
        """
        async with await self._lock_for(session_id):
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            target = next(
                (
                    character
                    for character in ckpt.characters
                    if character.character_id == character_id
                ),
                None,
            )
            if target is None:
                raise ValueError(f"No character '{character_id}' in this session.")
            if not target.is_playable:
                raise ValueError(
                    f"Character '{target.name}' is not an available player seat."
                )

            chosen_name = " ".join(str(name or "").split()).strip()
            chosen_appearance = str(appearance or "").strip()
            if len(chosen_name) > 80:
                raise ValueError("Character name must be 80 characters or fewer.")
            if len(chosen_appearance) > 600:
                raise ValueError("Appearance must be 600 characters or fewer.")
            if is_player_authored_slot(target):
                if not chosen_name:
                    raise ValueError(
                        f"'{target.name}' is a player-authored seat; choose "
                        "your character's name before joining."
                    )
                if chosen_name.casefold() == target.name.strip().casefold():
                    raise ValueError(
                        f"Rename the player-authored seat '{target.name}' "
                        "before joining."
                    )
                if not chosen_appearance:
                    raise ValueError(
                        f"'{target.name}' is a player-authored seat; describe "
                        "your character's appearance before joining."
                    )

            self._bind_user_in_checkpoint(ckpt, user_id, character_id)
            if chosen_name:
                target.name = chosen_name
            if chosen_appearance:
                target.public_sheet.appearance = chosen_appearance
                target.visuals.default_loadout = chosen_appearance
            self.checkpoint_mgr.save(ckpt)
            return ckpt

    async def join_player_character(
        self,
        session_id: str,
        character_id: str,
        user_id: int,
        *,
        name: str = "",
        appearance: str = "",
    ) -> PlayerJoinResult:
        """Claim a seat, then use the shared lobby/arrival lifecycle."""
        ckpt = await self.claim_player_character(
            session_id,
            character_id,
            user_id,
            name=name,
            appearance=appearance,
        )
        target = next(
            character
            for character in ckpt.characters
            if character.character_id == character_id
        )
        pre_play = not any(ckpt.narrator_conversations.values())
        response = None
        if not pre_play:
            response = await self.run_arrival_turn(
                session_id=session_id,
                acting_character_id=character_id,
            )
        return PlayerJoinResult(
            character_id=character_id,
            character_name=target.name,
            pre_play=pre_play,
            response=response,
        )

    async def unbind_user(self, session_id: str, user_id: int) -> str | None:
        """Remove this user's binding, serialized on the per-session lock so
        it cannot race a concurrent /act. Returns the freed character_id."""
        async with await self._lock_for(session_id):
            return self._unbind_user_locked(session_id, user_id)

    def _unbind_user_locked(self, session_id: str, user_id: int) -> str | None:
        """Remove this user's binding. Returns the freed character_id, or None
        if they had no binding. Assumes the caller holds the per-session lock.

        v11-A5: purges any v11 state (active_act_slots entries, open Cat II
        event responder lists/collected intentions, render buffers) the
        character held before removing the binding. Prevents stranded pins
        from freezing the beat when a player /leave's mid-beat.
        """
        from app.engine.turn_loop import purge_character_state

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        uid = str(user_id)
        freed = None
        for char_id, bound in list(ckpt.session.character_bindings.items()):
            if bound == uid:
                freed = char_id
        if freed is not None:
            # v11-r5: purge v11 state BEFORE removing the binding so no
            # reader can observe an unbound character with a stale pin.
            # Purge is a pure in-memory mutation; del + save follow
            # atomically on the same checkpoint write.
            purge_character_state(ckpt, freed)
            del ckpt.session.character_bindings[freed]
            dnd_inventory.remove_character_from_loot_offers(ckpt, freed)
            target = next(
                (
                    character
                    for character in ckpt.characters
                    if character.character_id == freed
                ),
                None,
            )
            if is_player_authored_slot(target):
                target.status = CharacterStatus.dormant
                target.location = "outside_active_fiction"
                ckpt.session.pending_engine_state_updates.append(
                    f"Character lifecycle update: {freed} left active fiction "
                    "and is now dormant at outside_active_fiction. Do not "
                    "route, observe, or activate this existing record unless "
                    "a later authored arrival explicitly reintroduces it."
                )
            if ckpt.session.player_character_id == freed:
                ckpt.session.player_character_id = ""
            self.checkpoint_mgr.save(ckpt)
        return freed

    # ---- takeover -----------------------------------------------------------

    async def takeover(
        self,
        session_id: str,
        character_id: str,
        user_id: int,
    ) -> CheckpointFile:
        """Bind the user to an existing character, serialized on the
        per-session lock (shares the lock with /act)."""
        async with await self._lock_for(session_id):
            return self._takeover_locked(session_id, character_id, user_id)

    def _takeover_locked(
        self,
        session_id: str,
        character_id: str,
        user_id: int,
    ) -> CheckpointFile:
        """Plain takeover: bind the user to an existing character. Assumes
        the caller holds the per-session lock.

        Name, appearance, identity, everything else stays as-authored.
        This is the default `/join` path — the user becomes the
        character as the story seed wrote them. Refuses if already
        claimed, culled, or nonexistent (via `bind_user`).

        Under playable-2 semantics the binding IS the takeover —
        `is_playable` is an authoring-time flag describing which slots
        a human can step into, not a runtime flag toggled by binding.
        We log a warning if the story seed did not mark this character
        playable but a player is taking them over anyway (most likely
        a custom override via /character path); the binding still
        applies because explicit user intent wins.
        """
        ckpt = self._bind_user_locked(session_id, user_id, character_id)
        target = next(
            (c for c in ckpt.characters if c.character_id == character_id),
            None,
        )
        if target is None:
            raise ValueError(f"No character '{character_id}' in this session.")
        if not target.is_playable:
            logger.warning(
                "takeover: %s (%s) was not marked is_playable=true in the "
                "story seed, but user %s bound to them anyway. Binding stands; "
                "the seed probably should have flagged them playable.",
                target.name,
                character_id,
                user_id,
            )
        return ckpt

    async def create_custom_character(
        self,
        session_id: str,
        user_id: int,
        description: str,
    ) -> CharacterRecord:
        """Mode='describe' custom-character spawn, serialized on the
        per-session lock (held across the takeover LLM call, exactly like
        /act) so the spawn+binding cannot be clobbered by a concurrent turn."""
        async with await self._lock_for(session_id):
            return await self._create_custom_character_locked(
                session_id,
                user_id,
                description,
            )

    async def _create_custom_character_locked(
        self,
        session_id: str,
        user_id: int,
        description: str,
    ) -> CharacterRecord:
        """Mode='describe': router authors a full new character from the
        player's concept, lands them in the world, binds to the user. The
        returned record has its engine-assigned character_id,
        is_playable=True, and is already written to the checkpoint. Assumes
        the per-session lock is held."""
        from app.schemas.takeover import TakeoverAuthoredOutput

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        out: TakeoverAuthoredOutput = await self._call_takeover(
            ckpt,
            mode="describe",
            description=description,
            invoking_user_id=str(user_id),
        )
        new_id = _pick_unused_character_id(ckpt, out.character.name)
        new_char = out.character.to_record(character_id=new_id)
        new_char.is_playable = True
        ckpt.characters.append(new_char)
        ckpt.session.character_bindings[new_id] = str(user_id)
        # Prefer the LLM-authored router_summary (it has full omniscient
        # context). Fall back to a mechanical line if missing — the
        # describe prompt is supposed to always emit a non-empty
        # summary, but the engine should never silently swallow a
        # spawn just because the LLM regressed. The summary is
        # normalized before interpolation (whitespace collapsed,
        # over-long entries truncated) using the same helper the spawn
        # path uses, so an LLM that drops a multi-paragraph explanation
        # into the field can't shatter the next router prompt.
        summary = _normalize_router_summary(out.character.router_summary or "")
        if summary:
            ckpt.session.pending_engine_state_updates.append(
                f"Existing character ready for arrival: {new_id} — "
                f"{summary} Status=active; location=not yet placed."
            )
        else:
            role = new_char.public_sheet.role or "unknown role"
            loc = new_char.location or "unknown"
            ckpt.session.pending_engine_state_updates.append(
                f"Existing character ready for arrival: {new_id}, "
                f"role={role}, location={loc}."
            )
            logger.warning(
                "Custom-character spawn for %s landed without "
                "router_summary; surfaced mechanical fallback line.",
                new_id,
            )

        if out.session_note:
            _append_session_note(ckpt, out.session_note)

        self.checkpoint_mgr.save(ckpt)
        logger.info(
            "Custom character spawned in %s: %s (%s)",
            session_id,
            new_char.name,
            new_char.character_id,
        )
        return new_char

    async def create_player_character_simple(
        self,
        session_id: str,
        user_id: int,
        *,
        name: str,
        appearance: str,
        lived_fact: str = "",
    ) -> CharacterRecord:
        """LLM-free player-character spawn, serialized on the per-session
        lock so the spawn+binding cannot be clobbered by a concurrent /act."""
        async with await self._lock_for(session_id):
            return self._create_player_character_simple_locked(
                session_id,
                user_id,
                name=name,
                appearance=appearance,
                lived_fact=lived_fact,
            )

    def _create_player_character_simple_locked(
        self,
        session_id: str,
        user_id: int,
        *,
        name: str,
        appearance: str,
        lived_fact: str = "",
    ) -> CharacterRecord:
        """LLM-free player-character spawn from raw user inputs. Assumes the
        per-session lock is held.

        This is the fast path behind /join's "Create your own character"
        option. It does not call takeover authoring: it builds a record from
        the player's own public identity and, optionally, one lived fact.
        The router places the character through the immediate `(arrive)`
        directive. If the player supplies no fact, the record remains
        actor-less until ordinary release synthesis has observable material.
        - `name` and `appearance` are required.
        - `lived_fact` is optional and becomes one second-person actor fact.

        Returns the freshly-created record (already saved to the
        checkpoint).
        """
        name = name.strip()
        appearance = appearance.strip()
        lived_fact = lived_fact.strip()
        if not name:
            raise ValueError("Character name cannot be empty.")
        if not appearance:
            raise ValueError("Character appearance cannot be empty.")

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        new_id = _pick_unused_character_id(ckpt, name)
        new_char = CharacterRecord(
            character_id=new_id,
            name=name,
            status=CharacterStatus.active,
            location="",  # router places via (arrive) directive
            is_playable=True,
            public_sheet=PublicSheet(appearance=appearance),
            visuals=CharacterVisuals(default_loadout=appearance),
            actor=(
                ActorRecord(
                    may_act_offstage=False,
                    facts=[
                        ActorFact(
                            origin=ActorFactOrigin.lived,
                            text=lived_fact,
                        )
                    ],
                )
                if lived_fact
                else None
            ),
        )
        ckpt.characters.append(new_char)
        ckpt.session.character_bindings[new_id] = str(user_id)

        # Surface the spawn to the router. The (arrive) turn that fires
        # right after this will pick the line up via the standard
        # state-changes block and decide where to drop them in. Keep
        # the line tight — actor-local facts do not belong in the per-turn
        # router context.
        bits = [f"appearance: {appearance[:200]}"]
        bits.append(
            "sparse arrival: infer a concrete story role "
            "and immediate on-ramp from the premise; surface that on-ramp "
            "as in-fiction observable_facts to the characters who would know"
        )
        ckpt.session.pending_engine_state_updates.append(
            f"Existing character ready for arrival: {new_id} — {'; '.join(bits)}."
        )

        self.checkpoint_mgr.save(ckpt)
        logger.info(
            "Custom player character (LLM-free) spawned in %s by user %s: %s (%s)",
            session_id,
            user_id,
            name,
            new_id,
        )
        return new_char

    async def suggest_replacement_targets(
        self,
        session_id: str,
        description: str,
        *,
        invoking_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Mode='suggest': router surveys the roster for NPCs worth
        replacing with the player's concept. Returns the candidate list
        and an optional preamble. No mutation.

        `invoking_user_id` is forwarded to the prompt's POV-location
        resolver so multi-player sessions don't see another player's
        location as "the action" in the suggest context."""
        from app.schemas.takeover import TakeoverSuggestOutput

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        out: TakeoverSuggestOutput = await self._call_takeover(
            ckpt,
            mode="suggest",
            description=description,
            invoking_user_id=invoking_user_id,
        )
        by_id = {c.character_id: c for c in ckpt.characters}
        candidates = []
        for candidate in out.candidates:
            data = candidate.model_dump()
            character = by_id.get(candidate.character_id)
            if character is not None:
                data["name"] = character.name
            candidates.append(data)
        return {
            "candidates": candidates,
            "preamble": out.preamble,
        }

    async def replace_with_custom(
        self,
        session_id: str,
        user_id: int,
        target_character_id: str,
        description: str,
    ) -> CharacterRecord:
        """Mode='replace' custom-character graft, serialized on the
        per-session lock (held across the takeover LLM call, like /act)."""
        async with await self._lock_for(session_id):
            return await self._replace_with_custom_locked(
                session_id,
                user_id,
                target_character_id,
                description,
            )

    async def _replace_with_custom_locked(
        self,
        session_id: str,
        user_id: int,
        target_character_id: str,
        description: str,
    ) -> CharacterRecord:
        """Mode='replace': graft a player-authored character onto an
        existing NPC's slot. Assumes the per-session lock is held. Preserves
        circumstances (location, status, pending observations) and overwrites
        public identity and the private actor record. Clears the target's
        rolling character conversation so the new self starts fresh.

        Binds user, marks the slot is_playable=true, saves."""
        from app.schemas.takeover import TakeoverAuthoredOutput

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        target = next(
            (c for c in ckpt.characters if c.character_id == target_character_id),
            None,
        )
        if target is None:
            raise ValueError(f"No character '{target_character_id}' in this session.")
        claimed_by = ckpt.session.character_bindings.get(target_character_id)
        if claimed_by and claimed_by != str(user_id):
            raise ValueError(f"'{target.name}' is already bound to another player.")

        out: TakeoverAuthoredOutput = await self._call_takeover(
            ckpt,
            mode="replace",
            description=description,
            picked_target=target,
            invoking_user_id=str(user_id),
        )
        authored = out.character

        # Identity overwrite from authored (flat shape); circumstances
        # preserved from target.
        target.name = authored.name
        target.public_sheet = PublicSheet(
            role=authored.role,
            appearance=authored.appearance,
            faction=authored.faction,
            public_context=authored.public_context,
        )
        self.image_generation.retire_character_identity(
            session_id=session_id,
            character_id=target_character_id,
            source_turn_index=ckpt.session.turn_index,
        )
        target.visuals = CharacterVisuals(
            default_loadout=authored.default_loadout or authored.appearance,
        )
        self.image_generation.suppress_reviewed_identity_binding(
            session_id=session_id,
            character_id=target_character_id,
        )
        self.image_generation.allow_character_identity_after(
            session_id=session_id,
            character_id=target_character_id,
            minimum_source_turn=ckpt.session.turn_index + 1,
        )
        target.actor = authored.actor.model_copy(deep=True)
        # Keep target's location, status, and pending observations as the
        # circumstances the player inherits.
        target.is_playable = True
        forget_visual_introductions_for_character(ckpt, target_character_id)

        # Drop rolling character conversation — the voice has changed.
        # The agent's prior parentheticals (their interior continuity)
        # lived in this conversation; popping it gives the new authored
        # character a clean slate.
        ckpt.character_conversations.pop(target_character_id, None)
        target.agent_identity_seed_sha256 = ""

        ckpt.session.character_bindings[target_character_id] = str(user_id)
        # Same router_summary preference as create_custom_character. The
        # replacement summary describes only the external fictional identity
        # change; interface ownership is irrelevant to router adjudication.
        summary = _normalize_router_summary(out.character.router_summary or "")
        if summary:
            ckpt.session.pending_engine_state_updates.append(
                f"Character replacement: id {target_character_id} "
                f"identity overwritten — {summary} Treat this as a new "
                "identity with the same established body and circumstances."
            )
        else:
            ckpt.session.pending_engine_state_updates.append(
                f"Character replacement: identity of {target_character_id} "
                f"has been overwritten — "
                f"role={target.public_sheet.role or 'unknown role'}. "
                "The private actor record is different from the prior version; "
                f"treat this as a new actor with the "
                f"same body."
            )
            logger.warning(
                "Character replacement for %s landed without "
                "router_summary; surfaced mechanical fallback line.",
                target_character_id,
            )

        if out.session_note:
            _append_session_note(ckpt, out.session_note)

        self.checkpoint_mgr.save(ckpt)
        logger.info(
            "Character replaced in %s: %s grafted onto %s",
            session_id,
            target.name,
            target_character_id,
        )
        return target

    async def _call_takeover(
        self,
        ckpt: CheckpointFile,
        *,
        mode: str,
        description: str,
        picked_target: CharacterRecord | None = None,
        invoking_user_id: str | None = None,
    ):
        """Render the takeover prompt, dispatch to the LLM with the
        mode-appropriate response model, and return the parsed output.

        mode: 'describe' | 'suggest' | 'replace'
        picked_target: required only for mode='replace'.
        invoking_user_id: the Discord/CLI user running this takeover.
            Forwarded to `_build_takeover_context` so the prompt's
            current location block tracks the invoking user's position
            instead of falling through to the creator binding.
        """
        from app.schemas.takeover import (
            TakeoverAuthoredOutput,
            TakeoverSuggestOutput,
        )

        if mode not in {"describe", "suggest", "replace"}:
            raise ValueError(f"Unknown takeover mode: {mode}")
        if mode == "replace" and picked_target is None:
            raise ValueError("mode='replace' requires picked_target")

        context = _build_takeover_context(
            ckpt,
            description,
            picked_target,
            invoking_user_id=invoking_user_id,
        )

        response_model = (
            TakeoverSuggestOutput if mode == "suggest" else TakeoverAuthoredOutput
        )
        messages = self.prompt_mgr.render_messages(
            "takeover",
            mode=mode,
            player_description=description,
            **context,
        )
        response = await self.client.complete(
            role="event_router",
            messages=messages,
            response_model=response_model,
            temperature=0.7,
            max_tokens=4000,
        )
        return response.parsed

    def build_character_dossier(
        self,
        session_id: str,
        character_id: str,
    ) -> str:
        """Build the private DM a joining player needs to play this character.

        It combines public identity, a concise human control contract, and
        the actor's own sparse facts. Global hidden lore remains outside this
        character-specific view.
        """
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        char = next(
            (c for c in ckpt.characters if c.character_id == character_id), None
        )
        if char is None:
            raise ValueError(f"No character '{character_id}' in this session.")

        lines: list[str] = [
            f"# Dossier · {char.name}",
            "_This is what your character knows about themselves, wants, and "
            "keeps private. The world beyond this page is for you to discover "
            "through play._",
        ]

        sheet = char.public_sheet
        sheet_bits: list[str] = []
        if sheet.role:
            sheet_bits.append(f"**Role** — {sheet.role}")
        if sheet.faction:
            sheet_bits.append(f"**Faction** — {sheet.faction}")
        if sheet.appearance:
            sheet_bits.append(f"**Appearance** — {sheet.appearance}")
        if sheet.public_context:
            sheet_bits.append(f"**Known publicly** — {sheet.public_context}")
        if sheet_bits:
            lines.append("\n".join(sheet_bits))

        if char.player_guidance:
            lines.append(f"## Your Control & Perspective\n{char.player_guidance}")
        if char.actor is not None and char.actor.facts:
            lines.append(
                "## What You Know Of Yourself\n"
                + "\n".join(
                    f"- {fact.text}"
                    for fact in char.actor.facts
                )
            )

        return "\n\n".join(lines)

    # ---- settings -----------------------------------------------------------

    def list_settings(self, session_id: str) -> list[dict[str, Any]]:
        """Return the full settings view (keys, current + default values,
        descriptions) for the /settings list command."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        return list_settings_view(ckpt)

    def get_setting(self, session_id: str, key: str) -> Any:
        """Return the current value of a single setting by key.
        Raises UnknownSettingError if the key isn't registered."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        return get_setting(ckpt, key)

    async def set_setting(
        self,
        session_id: str,
        key: str,
        raw_value: str,
    ) -> Any:
        """Update and persist one setting under the session mutation lock.

        A turn loads, mutates, and saves the same checkpoint. Serializing this
        read-modify-write path with turns ensures an already-running turn
        finishes with its original configuration and the requested setting is
        then applied to that completed checkpoint instead of being overwritten.
        """

        lock = await self._lock_for(session_id)
        async with lock:
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            new_value = set_setting(ckpt, key, raw_value)
            self.checkpoint_mgr.save(ckpt)
            logger.info(
                "Setting updated in %s: %s = %r",
                session_id,
                key,
                new_value,
            )
            return new_value

    def known_setting_keys(self) -> list[str]:
        """Exposed so bot command autocomplete and CLI can surface the
        valid keys without each frontend importing the registry."""
        return list(SETTINGS_BY_KEY.keys())

    # ---- D&D combat tracker -------------------------------------------------

    def _dnd_combat_module(self) -> Any:
        """Import the optional core combat tracker integration.

        The bot layer deliberately does not define combat state itself. The
        contract lives in app.engine.dnd_combat; until that module is present,
        these helpers raise a clean runtime error instead of making Discord
        command import fail.
        """
        try:
            from app.engine import dnd_combat
        except ImportError as e:
            raise RuntimeError(
                "D&D combat tracker core is not available "
                "(expected app.engine.dnd_combat)."
            ) from e
        return dnd_combat

    def _default_combat_participant_ids(self, ckpt: CheckpointFile) -> list[str]:
        """Default roster: active bound characters plus active same-location NPCs.

        There is no Discord user context on /combat begin, so "visible" is
        interpreted conservatively as the current bound party's active
        location(s). If no bound active character has a location, the default
        is just the active bound characters.
        """
        active_by_id = {
            c.character_id: c for c in ckpt.characters if c.status.value == "active"
        }
        bound_ids = [
            cid for cid in ckpt.session.character_bindings if cid in active_by_id
        ]
        locations = {
            active_by_id[cid].location
            for cid in bound_ids
            if active_by_id[cid].location
        }
        selected: list[str] = []
        for char in ckpt.characters:
            if char.status.value != "active":
                continue
            is_bound = char.character_id in bound_ids
            is_same_location_npc = (
                bool(locations)
                and char.character_id not in ckpt.session.character_bindings
                and char.location in locations
            )
            if is_bound or is_same_location_npc:
                selected.append(char.character_id)
        return selected

    def _resolve_combat_participant_refs(
        self,
        ckpt: CheckpointFile,
        refs: Iterable[str],
    ) -> list[str]:
        by_id = {c.character_id: c for c in ckpt.characters}
        resolved: list[str] = []
        missing: list[str] = []
        for raw_ref in refs:
            ref = str(raw_ref or "").strip()
            if not ref:
                continue
            if ref in by_id:
                resolved.append(ref)
                continue
            ref_slug = _participant_ref_slug(ref)
            matches = [
                character
                for character in ckpt.characters
                if _combat_participant_ref_matches(character, ref_slug)
            ]
            if len(matches) == 1:
                resolved.append(matches[0].character_id)
                continue
            if len(matches) > 1:
                choices = ", ".join(
                    _combat_participant_choice_text(character)
                    for character in matches[:8]
                )
                raise ValueError(
                    f"Ambiguous combat participant {ref!r}. "
                    f"Use one of these character IDs: {choices}."
                )
            missing.append(ref)
        if missing:
            options = _combat_participant_options_text(ckpt.characters)
            suffix = f" Available participants: {options}." if options else ""
            raise ValueError(
                "Unknown combat participant(s): "
                + ", ".join(missing)
                + ". Use character IDs or unambiguous names."
                + suffix
            )
        return resolved

    def _combat_save_and_view(
        self,
        ckpt: CheckpointFile,
        result: Any,
    ) -> DndCombatView:
        if isinstance(result, CheckpointFile):
            ckpt = result
            result = None
        elif isinstance(result, tuple):
            for item in result:
                if isinstance(item, CheckpointFile):
                    ckpt = item
                else:
                    result = item
                    break
        self.checkpoint_mgr.save(ckpt)
        return self._combat_view(ckpt, result)

    def _combat_state_source(self, ckpt: CheckpointFile, result: Any = None) -> Any:
        if result is not None:
            return result
        candidates = [
            getattr(ckpt.session, "active_combat", None),
            getattr(ckpt.session, "dnd_combat", None),
            getattr(ckpt.session, "combat", None),
            ckpt.world_state.global_flags.get("dnd_combat"),
            ckpt.world_state.global_flags.get("combat"),
        ]
        for candidate in candidates:
            if candidate:
                return candidate
        return None

    def _combat_get(self, source: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(source, dict) and name in source:
                return source[name]
            if hasattr(source, name):
                return getattr(source, name)
        return default

    def _combat_effect_labels(self, effects: Any) -> tuple[str, ...]:
        labels: list[str] = []
        for effect in effects or []:
            if isinstance(effect, str):
                name = effect.strip()
                detail_parts: list[str] = []
            else:
                name = str(
                    self._combat_get(
                        effect,
                        "name",
                        "slug",
                        "source_id",
                        default="",
                    )
                    or ""
                ).strip()
                if "_" in name or "-" in name:
                    name = name.replace("_", " ").replace("-", " ").title()
                detail_parts = []
                if bool(self._combat_get(effect, "concentration", default=False)):
                    detail_parts.append("concentration")
                remaining = self._optional_int(
                    self._combat_get(effect, "remaining_rounds", default=None)
                )
                if remaining:
                    noun = "round" if remaining == 1 else "rounds"
                    detail_parts.append(f"{remaining} {noun}")
                elif self._combat_get(effect, "duration_text", default=""):
                    detail_parts.append(
                        str(
                            self._combat_get(
                                effect,
                                "duration_text",
                                default="",
                            )
                        ).strip()
                    )
            if not name:
                continue
            detail = "; ".join(part for part in detail_parts if part)
            labels.append(f"{name} ({detail})" if detail else name)
        return tuple(labels)

    def _combat_map_lines(self, source: Any) -> tuple[str, ...]:
        raw_lines = self._combat_get(source, "battle_map_summary", default=None)
        if raw_lines:
            return tuple(str(line) for line in raw_lines if str(line).strip())
        return tuple(dnd_spatial.render_battle_map_summary(source))

    def _combat_participant_view(
        self,
        raw: Any,
        characters: dict[str, CharacterRecord],
        current_id: str,
    ) -> DndCombatParticipantView:
        cid = str(
            self._combat_get(
                raw,
                "combatant_id",
                "character_id",
                "participant_id",
                "id",
                default="",
            )
        )
        character_id = str(self._combat_get(raw, "character_id", default=cid) or cid)
        char = characters.get(character_id)
        mechanics = char.mechanics if char else {}
        hp = mechanics.get("hit_points") if isinstance(mechanics, dict) else {}
        if not isinstance(hp, dict):
            hp = {}
        raw_hp = self._combat_get(raw, "hp", "hit_points", default={})
        if isinstance(raw_hp, dict):
            hp = {**hp, **raw_hp}
        raw_initiative = self._combat_get(
            raw,
            "initiative",
            "initiative_total",
            default=None,
        )
        if isinstance(raw_initiative, dict):
            raw_initiative = raw_initiative.get("total")
        name = str(
            self._combat_get(raw, "name", default="") or (char.name if char else cid)
        )
        conditions = self._combat_get(raw, "conditions", default=None)
        if conditions is None:
            conditions = (
                mechanics.get("conditions", []) if isinstance(mechanics, dict) else []
            )
        active_effects = self._combat_get(raw, "active_effects", default=None)
        if active_effects is None and isinstance(mechanics, dict):
            active_effects = dnd_runtime.get_dnd_runtime(mechanics).get(
                "active_effects", []
            )
        raw_death_saves = self._combat_get(raw, "death_saves", default={})
        if not isinstance(raw_death_saves, dict):
            raw_death_saves = {}
        defeat_state = str(
            self._combat_get(raw, "defeat_state", default="active") or "active"
        )
        return DndCombatParticipantView(
            character_id=cid,
            name=name,
            current=bool(
                self._combat_get(raw, "current", "is_current", default=False)
                or (cid and cid == current_id)
            ),
            initiative=self._optional_int(raw_initiative),
            hp_current=self._optional_int(
                self._combat_get(
                    raw,
                    "hp_current",
                    "current_hp",
                    "hit_points_current",
                    default=hp.get("current"),
                )
            ),
            hp_max=self._optional_int(
                self._combat_get(
                    raw,
                    "hp_max",
                    "max_hp",
                    "hit_points_max",
                    default=hp.get("max"),
                )
            ),
            hp_temporary=int(
                self._combat_get(
                    raw,
                    "hp_temporary",
                    "temp_hp",
                    "hit_points_temporary",
                    default=hp.get("temporary", 0),
                )
                or 0
            ),
            armor_class=self._optional_int(
                self._combat_get(
                    raw,
                    "armor_class",
                    "ac",
                    default=(
                        mechanics.get("armor_class")
                        if isinstance(mechanics, dict)
                        else None
                    ),
                )
            ),
            conditions=tuple(str(c) for c in (conditions or []) if str(c)),
            active_effects=self._combat_effect_labels(active_effects),
            defeat_state=defeat_state,
            death_save_successes=self._optional_int(
                self._combat_get(
                    raw,
                    "death_save_successes",
                    default=raw_death_saves.get("successes", 0),
                )
            )
            or 0,
            death_save_failures=self._optional_int(
                self._combat_get(
                    raw,
                    "death_save_failures",
                    default=raw_death_saves.get("failures", 0),
                )
            )
            or 0,
            pending_initiating_action=str(
                self._combat_get(raw, "pending_initiating_action", default="") or ""
            ),
        )

    def _optional_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _combat_view(
        self,
        ckpt: CheckpointFile,
        result: Any = None,
    ) -> DndCombatView:
        source = self._combat_state_source(ckpt, result)
        if source is None:
            return DndCombatView(
                session_id=ckpt.session.session_id,
                active=False,
                message="No active combat.",
            )
        characters = {c.character_id: c for c in ckpt.characters}
        current_id = str(
            self._combat_get(
                source,
                "current_participant_id",
                "current_character_id",
                "active_participant_id",
                default="",
            )
            or ""
        )
        current_raw = self._combat_get(source, "current", default=None)
        if current_raw is not None and not current_id:
            current_id = str(
                self._combat_get(
                    current_raw,
                    "combatant_id",
                    "character_id",
                    default="",
                )
                or ""
            )
        raw_participants = (
            self._combat_get(
                source,
                "participants",
                "turn_order",
                "combatants",
                default=[],
            )
            or []
        )
        turn_index = (
            self._optional_int(self._combat_get(source, "turn_index", default=0)) or 0
        )
        if (
            not current_id
            and raw_participants
            and 0 <= turn_index < len(raw_participants)
        ):
            current_raw = raw_participants[turn_index]
            current_id = str(
                self._combat_get(
                    current_raw,
                    "combatant_id",
                    "character_id",
                    default="",
                )
                or ""
            )
        participants = tuple(
            self._combat_participant_view(raw, characters, current_id)
            for raw in raw_participants
        )
        if not current_id:
            current = next((p.character_id for p in participants if p.current), "")
            current_id = current
        raw_turn_number = self._combat_get(source, "turn_number", default=None)
        if raw_turn_number is None:
            raw_turn_index = self._combat_get(source, "turn_index", default=None)
            parsed_turn_index = self._optional_int(raw_turn_index)
            turn_number = (
                parsed_turn_index + 1
                if parsed_turn_index is not None and raw_participants
                else 0
            )
        else:
            turn_number = self._optional_int(raw_turn_number) or 0
        active = bool(self._combat_get(source, "active", "is_active", default=True))
        return DndCombatView(
            session_id=ckpt.session.session_id,
            active=active,
            round_number=(
                self._optional_int(
                    self._combat_get(
                        source,
                        "round_number",
                        "round",
                        default=0,
                    )
                )
                or 0
            ),
            turn_number=turn_number,
            current_participant_id=current_id,
            participants=participants,
            map_lines=self._combat_map_lines(source),
            message=str(self._combat_get(source, "message", default="") or ""),
        )

    def begin_combat(
        self,
        session_id: str,
        participant_ids: list[str] | None = None,
    ) -> DndCombatView:
        module = self._dnd_combat_module()
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        participants = participant_ids or self._default_combat_participant_ids(ckpt)
        if not participants:
            raise ValueError("No active combat participants found.")
        participants = self._resolve_combat_participant_refs(ckpt, participants)
        by_id = {c.character_id: c for c in ckpt.characters}
        selected = [by_id[cid] for cid in participants if cid in by_id]
        result = module.start_combat(ckpt.session, selected)
        return self._combat_save_and_view(ckpt, result)

    def combat_status(
        self,
        session_id: str,
        private: bool = False,
    ) -> DndCombatView:
        module = self._dnd_combat_module()
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        if getattr(ckpt.session, "active_combat", None) is None:
            return self._combat_view(ckpt)
        if private:
            result = module.private_status(ckpt.session)
        else:
            result = module.public_status(ckpt.session)
        return self._combat_view(ckpt, result)

    def combat_next(self, session_id: str) -> DndCombatView:
        module = self._dnd_combat_module()
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        module.advance_turn_with_effects(ckpt.session, characters=ckpt.characters)
        module.sync_combat_effects_to_characters(
            ckpt.session.active_combat,
            ckpt.characters,
        )
        flush_combat_visible_facts(ckpt)
        self.checkpoint_mgr.save(ckpt)
        return self._combat_view(ckpt)

    def combat_end(self, session_id: str) -> DndCombatView:
        module = self._dnd_combat_module()
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        combat = getattr(ckpt.session, "active_combat", None)
        combatants = list(getattr(combat, "combatants", []) or []) if combat else []
        observers = []
        seen: set[str] = set()
        for combatant in combatants:
            cid = str(
                getattr(combatant, "character_id", "")
                or getattr(combatant, "combatant_id", "")
                or ""
            )
            if not cid or cid in seen:
                continue
            observers.append(
                ObserverEntry(
                    character_id=cid,
                    observation_level="d",
                    routing_role="observe_only",
                )
            )
            seen.add(cid)
        pending_facts = []
        if combat is not None:
            pending_facts = list(module.drain_pending_visible_facts(combat))
            module.queue_router_observed_fact_updates(ckpt.session, combat)
        module.end_combat(ckpt.session, characters=ckpt.characters)
        if observers:
            observable_facts = [
                ObservableFact.all(fact) for fact in pending_facts if str(fact).strip()
            ]
            observable_facts.append(ObservableFact.all("D&D combat ends."))
            broadcast_event(
                ckpt,
                EventRouterOutput(
                    event_id="",
                    effective_at_s=0,
                    duration_s=0,
                    decision_rationale="manual combat end",
                    canonical_event=CanonicalEvent(
                        world_adjudication=WorldAdjudication(feasible=True),
                        observable_facts=observable_facts,
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
                ),
            )
        self.checkpoint_mgr.save(ckpt)
        return DndCombatView(
            session_id=ckpt.session.session_id,
            active=False,
            message="Combat ended.",
        )

    def combat_add(self, session_id: str, character_id: str) -> DndCombatView:
        module = self._dnd_combat_module()
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        character = next(
            (c for c in ckpt.characters if c.character_id == character_id),
            None,
        )
        if character is None:
            raise ValueError(f"Unknown combat participant: {character_id}")
        module.add_combatant(
            ckpt.session,
            character,
            session=ckpt.session,
        )
        self.checkpoint_mgr.save(ckpt)
        return self._combat_view(ckpt)

    def combat_remove(
        self,
        session_id: str,
        combatant_id: str,
        hard: bool = False,
    ) -> DndCombatView:
        module = self._dnd_combat_module()
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        module.remove_combatant(ckpt.session, combatant_id, hard=hard)
        self.checkpoint_mgr.save(ckpt)
        return self._combat_view(ckpt)

    def combat_damage(
        self,
        session_id: str,
        target_id: str,
        amount: int,
    ) -> DndCombatView:
        if amount <= 0:
            raise ValueError("Damage amount must be positive.")
        module = self._dnd_combat_module()
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        module.apply_damage(
            ckpt.session,
            target_id,
            amount,
            characters=ckpt.characters,
        )
        flush_combat_visible_facts(ckpt)
        self.checkpoint_mgr.save(ckpt)
        return self._combat_view(ckpt)

    def combat_heal(
        self,
        session_id: str,
        target_id: str,
        amount: int,
    ) -> DndCombatView:
        if amount <= 0:
            raise ValueError("Heal amount must be positive.")
        module = self._dnd_combat_module()
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        module.apply_healing(
            ckpt.session,
            target_id,
            amount,
            characters=ckpt.characters,
        )
        flush_combat_visible_facts(ckpt)
        self.checkpoint_mgr.save(ckpt)
        return self._combat_view(ckpt)

    async def begin_combat_locked(
        self,
        session_id: str,
        participant_ids: list[str] | None = None,
    ) -> DndCombatView:
        return await self._run_combat_mutation_locked(
            session_id,
            lambda: self.begin_combat(session_id, participant_ids),
        )

    async def combat_next_locked(self, session_id: str) -> DndCombatView:
        return await self._run_combat_mutation_locked(
            session_id,
            lambda: self.combat_next(session_id),
        )

    async def combat_add_locked(
        self,
        session_id: str,
        character_id: str,
    ) -> DndCombatView:
        return await self._run_combat_mutation_locked(
            session_id,
            lambda: self.combat_add(session_id, character_id),
        )

    async def combat_remove_locked(
        self,
        session_id: str,
        combatant_id: str,
        *,
        hard: bool = False,
    ) -> DndCombatView:
        return await self._run_combat_mutation_locked(
            session_id,
            lambda: self.combat_remove(session_id, combatant_id, hard=hard),
        )

    async def combat_end_locked(self, session_id: str) -> DndCombatView:
        return await self._run_combat_mutation_locked(
            session_id,
            lambda: self.combat_end(session_id),
        )

    async def combat_damage_locked(
        self,
        session_id: str,
        target_id: str,
        amount: int,
    ) -> DndCombatView:
        return await self._run_combat_mutation_locked(
            session_id,
            lambda: self.combat_damage(session_id, target_id, amount),
        )

    async def combat_heal_locked(
        self,
        session_id: str,
        target_id: str,
        amount: int,
    ) -> DndCombatView:
        return await self._run_combat_mutation_locked(
            session_id,
            lambda: self.combat_heal(session_id, target_id, amount),
        )

    async def _run_combat_mutation_locked(
        self,
        session_id: str,
        mutate: Callable[[], DndCombatView],
    ) -> DndCombatView:
        bridge_lock = await self._lock_for(session_id)
        async with bridge_lock:
            orchestrator_lock = await self.orchestrator.session_locks.get(session_id)
            async with orchestrator_lock:
                return mutate()

    def combat_reaction_prompt_event(
        self,
        session_id: str,
        character_id: str,
    ) -> str:
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        slot = ckpt.session.active_act_slots.get(character_id)
        if slot is None or slot.reason != "combat_reaction":
            return ""
        return slot.trigger_event_id or slot.cat_ii_event_id or ""

    async def defer_combat_reaction(
        self,
        *,
        session_id: str,
        character_id: str,
        event_id: str = "",
        user_id: int | None = None,
    ) -> TurnResponse:
        if user_id is not None:
            bound = self.get_user_binding(session_id, user_id)
            if bound != character_id:
                raise ValueError("That reaction belongs to another character.")
        lock = await self._lock_for(session_id)
        async with lock:
            return await self.orchestrator.defer_combat_reaction(
                session_id=session_id,
                character_id=character_id,
                event_id=event_id,
            )

    # ---- turn execution ------------------------------------------------------

    async def _lock_for(self, session_id: str) -> asyncio.Lock:
        async with self._locks_mutex:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    # v11-A5 lazy sweep hook.
    # Invoked from the /act hot path (run_turn) so stale Cat II pins are
    # surfaced without a background scheduler. It also releases stale optional
    # combat-reaction pins. `_run_turn_locked` consumes the returned Cat II
    # event IDs and delivers any pre-turn resolutions before the actor's own
    # turn response.
    def sweep_stale_pins(self, session_id: str) -> list[str]:
        """Sweep stale Cat II and combat-reaction pins, saving iff state changed.

        Returns Cat II event IDs that now need re-adjudication. Released combat
        reactions may advance combat immediately but do not return event IDs.
        Safe to call with no open events (returns []).
        """
        from app.engine.orchestrator import advance_pending_combat_if_unblocked
        from app.engine.turn_loop import (
            sweep_stale_cat_ii_pins,
            sweep_stale_combat_reaction_pins,
        )

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        swept = sweep_stale_cat_ii_pins(ckpt)
        released = sweep_stale_combat_reaction_pins(ckpt)
        if released:
            # A released reaction pin may unblock delayed initiative; the
            # whole sweep runs inside the per-session lock so this advance
            # cannot race a concurrent turn.
            advance_pending_combat_if_unblocked(ckpt)
        if swept or released:
            self.checkpoint_mgr.save(ckpt)
        if swept:
            logger.info(
                "v11 sweep: auto-resolved %d Cat II event(s) pre-turn: %s",
                len(swept),
                swept,
            )
        if released:
            logger.info(
                "AFK sweep: auto-passed %d combat-reaction pin(s) pre-turn: %s",
                len(released),
                released,
            )
        return swept

    async def run_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        acting_character_id: str = "",
        display_key: str = "",
    ) -> TurnResponse:
        """Process one turn under a per-session lock. Subsequent concurrent calls
        for the same session_id queue and run in order.

        v11-A5 / r5: `sweep_stale_pins` runs INSIDE the per-session lock
        so it cannot race with a concurrent orchestrator turn. Two /acts
        arriving at the same moment serialize: the lock-holder sweeps
        (no-op if nothing stale) then runs the orchestrator; the second
        /act then sees the already-swept state and its own sweep is a
        no-op. This is a narrow path — sweeps usually don't mutate —
        but moving it inside the lock eliminates the clobber race
        flagged by the round-4 edge review.
        """
        lock = await self._lock_for(session_id)
        async with lock:
            return await self._run_turn_locked(
                session_id=session_id,
                user_input=user_input,
                acting_character_id=acting_character_id,
                display_key=display_key,
            )

    async def retry_failed_render(
        self,
        *,
        session_id: str,
    ) -> RetryRenderResult:
        """Retry a pending narrator render without submitting a new action."""
        lock = await self._lock_for(session_id)
        async with lock:
            ckpt = self.checkpoint_mgr.load_latest(session_id)
            pending = ckpt.session.pending_narrator_render
            if pending is None:
                return RetryRenderResult(
                    response=TurnResponse(
                        session_id=session_id,
                        checkpoint_id=f"ckpt_{ckpt.session.turn_index:04d}",
                        turn_index=ckpt.session.turn_index,
                        output_text=(
                            "No failed narrator render is pending for this session."
                        ),
                        per_player_renders={},
                        beat_ended_reason="no_pending_render",
                    )
                )

            actor_id = pending.acting_player_id
            actor_user_id = str(
                (ckpt.session.character_bindings or {}).get(actor_id, "")
            )
            response = await self.orchestrator.retry_pending_narrator_render(session_id)
            return RetryRenderResult(
                response=response,
                actor_character_id=actor_id,
                actor_user_id=actor_user_id,
            )

    async def run_arrival_turn(
        self,
        *,
        session_id: str,
        acting_character_id: str,
    ) -> TurnResponse:
        """Run an `(arrive)` turn for a player joining a story that's
        already underway. Always fires `(arrive)` — the canonical
        opening (`(begin)`) is now driven exclusively by `/begin`
        through `run_begin_turn`, which the caller invokes BEFORE this
        method becomes the right tool.

        Pre-r9d this method picked `(begin)` vs `(arrive)` itself by
        inspecting `narrator_conversations`. That branch went away
        with the lobby split: `/join` now binds-only when the story
        hasn't begun, and the dedicated `/begin` command opens the
        story. By the time anyone calls `run_arrival_turn`, the
        opening has already happened and `(arrive)` is unambiguous.
        """
        lock = await self._lock_for(session_id)
        async with lock:
            logger.info(
                "run_arrival_turn: session=%s actor=%s directive=(arrive)",
                session_id,
                acting_character_id,
            )
            return await self._run_turn_locked(
                session_id=session_id,
                user_input="(arrive)",
                acting_character_id=acting_character_id,
            )

    async def run_begin_turn(
        self,
        *,
        session_id: str,
        triggering_character_id: str = "",
    ) -> TurnResponse:
        """Fire the canonical opening turn (`(begin)`) for the session.

        Driven by `/begin` after one or more players have `/join`'d
        into the lobby (pre-play). The router sees `(begin)` and
        composes the opening beat from `world_state` + the
        `## Initial Roster`, placing EVERY bound player at the chosen
        starting location so each gets their own POV render through the
        normal `_end_beat` per-POV fan-out.

        Args:
            session_id: the session to open.
            triggering_character_id: the player who typed `/begin`.
                Used as the default `acting_character_id` in the
                router's per-turn context. A rules adapter may instead
                select a bound semantic owner when the opening consumes
                that owner's authority. May be empty — the helper falls
                back to a deterministic pick from the bound roster
                (sorted by id) so two simultaneous `/begin`s converge
                on the same actor.

        Raises:
            ValueError: if no players are bound, or if the story has
                already begun (any narrator history present). Both
                are pre-checked under the per-session lock so two
                racing `/begin`s can't both fire the opener.

        TODO(multi-location-opening): this function currently asks the
        router to converge all bound players on one shared starting
        location (see event_router.txt OOC `(begin)` rules). When we
        want distinct starting locations per player, run the opening as
        N parallel `(begin)` calls — one per bound player, each
        seeded with their own intended location — and merge the
        resulting per-POV renders. The current single-location path is
        the simplest correct first step; the multi-location shape can
        be additive on top of it.
        """
        lock = await self._lock_for(session_id)
        async with lock:
            ckpt = self.checkpoint_mgr.load_latest(session_id)

            # Bound players by id, sorted for deterministic
            # actor selection when triggering_character_id is unset
            # or has been kicked between dispatch and lock acquisition.
            bound_ids = sorted((ckpt.session.character_bindings or {}).keys())
            if not bound_ids:
                raise ValueError(
                    "Cannot /begin: no players are bound to this session. "
                    "Have at least one player run /join first."
                )

            if any(ckpt.narrator_conversations.values()):
                raise ValueError(
                    "Cannot /begin: this story has already started. "
                    "Late joiners use /join — it now fires (arrive) "
                    "for any player binding after the opening."
                )

            triggering_actor_id = (
                triggering_character_id
                if triggering_character_id in bound_ids
                else bound_ids[0]
            )
            actor_id = (
                one_star_opening_account_owner_actor_id(ckpt, bound_ids)
                or triggering_actor_id
            )
            logger.info(
                "run_begin_turn: session=%s actor=%s bound=%s",
                session_id,
                actor_id,
                bound_ids,
            )
            return await self._run_turn_locked(
                session_id=session_id,
                user_input="(begin)",
                acting_character_id=actor_id,
            )

    async def _resolve_swept_events_locked(
        self,
        session_id: str,
    ) -> list[TurnResponse]:
        """Close sweep-populated events before accepting a new command."""

        try:
            event_ids = self.sweep_stale_pins(session_id)
        except Exception:
            # Best-effort — never let an AFK sweep error crash a turn.
            logger.exception(
                "v11 sweep_stale_pins failed for %s",
                session_id,
            )
            event_ids = []

        # v11-r6b: drive adjudication of any events the sweep filled.
        # Without this, a beat pinned on an AFK human sits open
        # indefinitely — the next /act would bounce off the pin. By
        # closing the sweep-populated events first, the hot path
        # clears their state before the player's /act runs.
        #
        # v11-r7a: capture each resolution's TurnResponse so the
        # frontend can fan its per-POV renders out. Previously the
        # responses were dropped and pinned humans never saw their
        # AFK-resolved beats.
        pre_turn: list[TurnResponse] = []
        for event_id in event_ids:
            try:
                resp = await self.orchestrator.resolve_cat_ii(
                    session_id,
                    event_id,
                )
                if resp.per_player_renders:
                    pre_turn.append(resp)
            except Exception:
                logger.exception(
                    "resolve_cat_ii failed for session=%s event=%s",
                    session_id,
                    event_id,
                )
        return pre_turn

    async def _run_turn_locked(
        self,
        *,
        session_id: str,
        user_input: str,
        acting_character_id: str,
        display_key: str = "",
    ) -> TurnResponse:
        """Body of `run_turn` — caller MUST hold the per-session lock."""

        pre_turn = await self._resolve_swept_events_locked(session_id)

        response = await self.orchestrator.process_turn(
            TurnRequest(
                session_id=session_id,
                user_input=user_input,
                acting_character_id=acting_character_id,
                display_key=display_key,
            )
        )
        response.pre_turn_resolutions = [
            *pre_turn,
            *(response.pre_turn_resolutions or []),
        ]
        for resolved_response in response.pre_turn_resolutions:
            self._append_one_star_mission_report_prose(resolved_response)
            self._validate_one_star_hero_card_routing(resolved_response)
            self._validate_one_star_mission_report_routing(resolved_response)
        self._append_one_star_mission_report_prose(
            response,
            acting_character_id=acting_character_id,
        )
        self._validate_one_star_hero_card_routing(response)
        self._validate_one_star_mission_report_routing(response)
        await self._prewarm_visual_novel_sprites(session_id=session_id)
        return response

    async def run_query(
        self,
        *,
        session_id: str,
        character_id: str,
        question: str,
    ) -> TurnResponse:
        """Answer /query through the router/narrator path.

        A query is no longer a separate read-only LLM role. It enters
        the normal turn loop as a fully parenthesized OOC directive, so
        the router emits a private observable fact and the narrator
        renders it from the querying character's POV. This mutates the
        checkpoint like any other private beat.
        """
        return await self.run_turn(
            session_id=session_id,
            user_input=f"(query: {question.strip()})",
            acting_character_id=character_id,
        )


# ---- player-flow helpers ----------------------------------------------------
# Helpers that support the bot's player-facing flows. Under playable-2
# semantics, every player→character lookup routes through
# `session.character_bindings` (the canonical live state). `is_playable`
# is now an authoring-time flag describing which slots a human MAY step
# into via /join — not a runtime flag toggled by binding.
#
# * `build_character_dossier` / `_summaries_from_checkpoint` — bindings-
#   driven; used by takeover prompts and roster summaries.
# * `_pick_unused_character_id` — pure slug helper.


def _pick_unused_character_id(
    ckpt: CheckpointFile,
    name: str,
) -> str:
    """Slugify `name` into a snake_case character_id; disambiguate with
    a numeric suffix if the slug is already in use."""
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "character"
    taken = {c.character_id for c in ckpt.characters}
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def _participant_ref_slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _combat_participant_ref_matches(
    character: CharacterRecord,
    ref_slug: str,
) -> bool:
    if not ref_slug:
        return False
    character_id = str(character.character_id or "")
    name = str(character.name or "")
    id_slug = _participant_ref_slug(character_id)
    name_slug = _participant_ref_slug(name)
    if ref_slug in {id_slug, name_slug}:
        return True
    return ref_slug in id_slug.split("_") or ref_slug in name_slug.split("_")


def _combat_participant_choice_text(character: CharacterRecord) -> str:
    name = str(character.name or "").strip()
    if name and name != character.character_id:
        return f"{name} ({character.character_id})"
    return str(character.character_id)


def _combat_participant_options_text(
    characters: Iterable[CharacterRecord],
) -> str:
    active = [
        character
        for character in characters
        if str(getattr(character.status, "value", character.status)) == "active"
    ]
    choices = [_combat_participant_choice_text(character) for character in active[:12]]
    suffix = "" if len(active) <= 12 else ", ..."
    return ", ".join(choices) + suffix


def _build_takeover_context(
    ckpt: CheckpointFile,
    description: str,
    picked_target: "CharacterRecord | None",
    invoking_user_id: str | None = None,
) -> dict[str, str]:
    """Render the context blocks the takeover prompt expects. Stays
    router-style: omniscient world state, full roster (spoiler-safe is
    NOT a constraint here — the router is authoring for itself).

    `invoking_user_id`: the Discord user (or CLI session) running the
    takeover. Threaded into `pov_location_for_user` so the prompt's
    location block reflects where this user is, not another binding.
    """
    from app.engine.context_builder import (
        build_hidden_facts,
        build_setting_summary,
        build_world_rules,
        pov_location_for_user,
    )

    setting_summary = build_setting_summary(ckpt)
    world_lore = ckpt.world_state.lore or "No detailed lore."
    hidden_lore = ckpt.world_state.hidden_lore or "(none)"
    hidden_facts = build_hidden_facts(ckpt, empty="(none)")
    world_rules = build_world_rules(ckpt)

    location = pov_location_for_user(ckpt, user_id=invoking_user_id)
    current_location_context = location or "(no active location)"

    registry_lines = []
    bindings = ckpt.session.character_bindings or {}
    for c in ckpt.characters:
        if c.status.value == "culled":
            continue
        if is_player_authored_slot(c) and c.character_id not in bindings:
            continue
        if c.character_id in bindings:
            marker = " [player-bound]"
        elif c.is_playable:
            marker = " [playable]"
        else:
            marker = ""
        role = f" — {c.public_sheet.role}" if c.public_sheet.role else ""
        fac = f" ({c.public_sheet.faction})" if c.public_sheet.faction else ""
        loc = f" @ {c.location}" if c.location else ""
        registry_lines.append(f"- {c.character_id}{role}{fac}{loc}{marker}")
    character_registry = "\n".join(registry_lines) or "(empty)"

    viewer_character_id = next(
        (
            character_id
            for character_id, user_id in bindings.items()
            if user_id == str(invoking_user_id)
        ),
        "",
    )
    recent_messages = (
        ckpt.narrator_conversations.get(viewer_character_id, [])
        if viewer_character_id
        else []
    )
    recent_bits = [
        text
        for message in recent_messages
        if message.role == "assistant"
        if (text := _narrator_history_message_text(message.content))
    ][-6:]
    recent_session_summary = (
        "\n\n".join(recent_bits) if recent_bits else "(no POV-safe turns available yet)"
    )

    if picked_target is not None:
        actor_facts = "\n".join(
            f"- {fact.origin.value}: {fact.text}"
            for fact in (picked_target.actor.facts if picked_target.actor else ())
        ) or "(none)"
        picked_target_block = (
            "## Picked Target\n"
            f"character_id: {picked_target.character_id}\n"
            f"role: {picked_target.public_sheet.role}\n"
            f"faction: {picked_target.public_sheet.faction}\n"
            f"public_context: {picked_target.public_sheet.public_context}\n"
            f"location: {picked_target.location}\n"
            f"actor facts:\n{actor_facts}\n\n"
        )
    else:
        picked_target_block = ""

    return {
        "setting_summary": setting_summary,
        "world_lore": world_lore,
        "hidden_lore": hidden_lore,
        "hidden_facts": hidden_facts,
        "world_rules": world_rules,
        "current_location_context": current_location_context,
        "character_registry": character_registry,
        "recent_session_summary": recent_session_summary,
        "picked_target_block": picked_target_block,
    }


def _append_session_note(ckpt: CheckpointFile, note: str) -> None:
    """Post a router-authored in-fiction note into the session_conversation
    so the next turn's router sees the new character's arrival. Stored
    as an assistant-like entry; non-authoritative — purely context."""
    from app.schemas.conversation import ConversationMessage

    if not note.strip():
        return
    ckpt.session_conversation.append(
        ConversationMessage(
            role="assistant",
            content=f'{{"takeover_note": {json.dumps(note)}}}',
        )
    )


def _dnd_attachment_summary(
    *,
    character: CharacterRecord,
    snapshot: dict[str, Any],
    settings: Any,
    name_overridden: bool,
) -> DndSheetAttachmentSummary:
    identity = snapshot.get("identity") or {}
    statblock = snapshot.get("statblock") or {}
    defenses = statblock.get("defenses") or {}
    hp = defenses.get("hit_points") or {}
    ac = defenses.get("armor_class") or {}
    source = snapshot.get("source") or {}

    classes: list[str] = []
    total_level = 0
    for entry in identity.get("classes") or []:
        if not isinstance(entry, dict):
            continue
        level = _safe_int(entry.get("level"), 0)
        total_level += level
        name = str(entry.get("name") or "").strip()
        subclass = str(entry.get("subclass") or "").strip()
        label = name
        if subclass and subclass.lower() != name.lower():
            label = f"{subclass} {name}" if name else subclass
        if label:
            classes.append(f"{label} {level}".strip())

    spellcasting = statblock.get("spellcasting") or {}
    return DndSheetAttachmentSummary(
        character_id=character.character_id,
        character_name=character.name,
        imported_name=str(identity.get("name") or ""),
        ruleset_id=str(snapshot.get("ruleset_id") or ""),
        session_ruleset_id=str(getattr(settings, "ruleset_id", "")),
        player_roll_mode=str(getattr(settings, "player_roll_mode", "")),
        source_type=str(source.get("type") or ""),
        total_level=total_level,
        classes=classes,
        armor_class=_safe_int(ac.get("value"), 0),
        hit_points_current=_safe_int(hp.get("current"), 0),
        hit_points_max=_safe_int(hp.get("max"), 0),
        hit_points_temporary=_safe_int(hp.get("temporary"), 0),
        skills_count=len(statblock.get("skills") or {}),
        actions_count=len(statblock.get("actions") or []),
        spells_count=len(spellcasting.get("spells") or []),
        resources_count=len(statblock.get("resources") or []),
        name_overridden=name_overridden,
    )


def _dnd_attachment_equipment_sentence(mechanics: dict[str, Any]) -> str:
    sheet = mechanics.get("dnd5e_sheet") if isinstance(mechanics, dict) else {}
    statblock = sheet.get("statblock") if isinstance(sheet, dict) else {}
    inventory = statblock.get("inventory") if isinstance(statblock, dict) else {}
    raw_items = inventory.get("items") if isinstance(inventory, dict) else []
    items = [item for item in raw_items or [] if isinstance(item, dict)]
    if not items:
        return "Their D&D equipment has no listed gear."

    labels: list[str] = []
    for item in items[:10]:
        name = str(item.get("name") or "Item").strip() or "Item"
        try:
            quantity = int(item.get("quantity") or 1)
        except (TypeError, ValueError):
            quantity = 1
        labels.append(f"{quantity}x {name}" if quantity != 1 else name)
    if len(items) > 10:
        labels.append(f"{len(items) - 10} more item(s)")
    return (
        "Their listed D&D equipment is physically present and available now "
        f"unless later fiction changes it: {', '.join(labels)}."
    )


def _dnd_experience_target_ids(
    ckpt: CheckpointFile,
    target: str,
) -> list[str]:
    normalized = (target or "").strip()
    if not normalized:
        raise ValueError("Choose a character id, or `all` for all bound players.")

    characters = {c.character_id: c for c in ckpt.characters}
    if normalized.lower() in {"all", "party", "players"}:
        target_ids = [
            cid for cid in (ckpt.session.character_bindings or {}) if cid in characters
        ]
        if not target_ids:
            raise ValueError("No bound player characters are available for XP.")
    else:
        if normalized not in characters:
            raise ValueError(f"No character '{normalized}' in this session.")
        target_ids = [normalized]

    missing = [
        cid
        for cid in target_ids
        if not dnd_experience.has_dnd_mechanics(characters[cid])
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "D&D experience can only be awarded to characters with attached "
            f"D&D mechanics. Missing: {joined}."
        )

    culled = [
        cid
        for cid in target_ids
        if str(getattr(characters[cid].status, "value", characters[cid].status))
        == "culled"
    ]
    if culled:
        joined = ", ".join(culled)
        raise ValueError(f"Cannot award XP to culled character(s): {joined}.")
    return target_ids


def _dnd_experience_award_result(
    character: CharacterRecord,
    view: dict[str, Any],
) -> DndExperienceAwardResult:
    return DndExperienceAwardResult(
        character_id=character.character_id,
        character_name=character.name,
        amount=_safe_int(view.get("amount"), 0),
        before=_safe_int(view.get("before"), 0),
        after=_safe_int(view.get("after"), 0),
        total_level=_safe_int(view.get("total_level"), 0),
        next_level=_safe_int(view.get("next_level"), 0),
        xp_to_next_level=_safe_int(view.get("xp_to_next_level"), 0),
        level_available=bool(view.get("level_available")),
        eligible_level=_safe_int(view.get("eligible_level"), 0),
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _narrator_history_message_text(content: Any) -> str:
    if isinstance(content, str):
        candidates = [content]
    elif isinstance(content, list):
        candidates = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
    else:
        return ""
    for candidate in reversed(candidates):
        text = candidate.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
            final_text = payload.get("final_text")
            if isinstance(final_text, str):
                return final_text
            raw_pages = payload.get("pages")
            if isinstance(raw_pages, list):
                pages = [VisualNovelPage.model_validate(page) for page in raw_pages]
                return visual_novel_pages_plain_text(pages)
            raise ValueError("narrator history has no supported render payload")
        except Exception:
            logger.warning(
                "Skipping malformed narrator history envelope",
                exc_info=True,
            )
    return ""


def _parse_model_json(model_cls, content: str):
    """Parse a Pydantic model from the LLM's free-form JSON output.

    Used when we can't enforce structured output via output_format. A
    live benchmark (scripts/structured_output_benchmark.py) showed
    the enforced path 400s with "Schema is too complex" 0/3 times for
    AuthoredCharacter after ~180s of hanging, while raw JSON + this
    helper succeeded 3/3 times in ~29s. output_format is also marked
    deprecated by the SDK, so we're on the sunset path using it.

    Strips common LLM envelope noise (markdown fences) and parses with
    Pydantic. Raises ValueError on unparseable output with a 500-char
    snippet so call sites surface something diagnostic.
    """
    import re

    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return model_cls.model_validate_json(text)
    except Exception as e:
        snippet = text[:500]
        raise ValueError(
            f"Failed to parse {model_cls.__name__} from LLM output: {e} "
            f"— first 500 chars: {snippet!r}"
        ) from e


def _summaries_from_checkpoint(ckpt: CheckpointFile) -> list[CharacterSummary]:
    """Render a roster's spoiler-free summaries from any checkpoint."""
    bindings = ckpt.session.character_bindings or {}
    summaries: list[CharacterSummary] = []
    for char in ckpt.characters:
        summaries.append(
            CharacterSummary(
                character_id=char.character_id,
                name=char.name,
                role=char.public_sheet.role or "",
                faction=char.public_sheet.faction or "",
                appearance=char.public_sheet.appearance or "",
                status=char.status.value,
                is_playable=char.is_playable,
                bound_user_id=bindings.get(char.character_id, ""),
                player_slot_kind=char.player_slot_kind.value,
                player_guidance=char.player_guidance,
            )
        )
    return summaries


def joinable_character_summaries(
    summaries: Iterable[CharacterSummary],
) -> list[CharacterSummary]:
    return [
        summary
        for summary in summaries
        if summary.is_playable
        and summary.status != "culled"
        and not summary.bound_user_id
    ]
