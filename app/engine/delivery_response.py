"""Project leased outbox entries onto the current frontend response shape."""

from __future__ import annotations

from app.schemas.delivery import DeliveryOutboxEntry, DeliveryVisualNovelRender
from app.schemas.responses import DiceRollDisplay, TurnResponse
from app.schemas.state import DndExperienceAwardDisplay


def response_from_deliveries(
    *,
    session_id: str,
    checkpoint_id: str,
    turn_index: int,
    acting_character_id: str,
    deliveries: list[DeliveryOutboxEntry],
    pause_reason: str = "",
) -> TurnResponse:
    prose_by_pov: dict[str, list[str]] = {}
    visual_by_pov: dict[str, DeliveryVisualNovelRender] = {}
    assets_by_pov: dict[str, list[object]] = {}
    reaction_prompts: dict[str, str] = {}
    loot_prompts: dict[str, list[str]] = {}
    commitment_prompts: dict[str, list[str]] = {}
    dice_rolls: list[DiceRollDisplay] = []
    experience_awards: list[DndExperienceAwardDisplay] = []
    seen_dice_rolls: set[str] = set()
    seen_experience_awards: set[str] = set()

    for entry in deliveries:
        pov_id = entry.pov_character_id
        payload = entry.payload
        if payload.prose.strip():
            prose_by_pov.setdefault(pov_id, []).append(payload.prose.strip())
        if payload.visual_novel is not None:
            existing = visual_by_pov.get(pov_id)
            if existing is None:
                visual_by_pov[pov_id] = payload.visual_novel.model_copy(deep=True)
            else:
                existing.segments.extend(
                    segment.model_copy(deep=True)
                    for segment in payload.visual_novel.segments
                )
        assets_by_pov.setdefault(pov_id, []).extend(payload.asset_reveals)
        if payload.reaction_prompt_event_id:
            reaction_prompts[pov_id] = payload.reaction_prompt_event_id
        if payload.loot_offer_ids:
            loot_prompts.setdefault(pov_id, []).extend(payload.loot_offer_ids)
        if payload.commitment_revision_ids:
            commitment_prompts.setdefault(pov_id, []).extend(
                payload.commitment_revision_ids
            )
        for item in payload.dice_rolls:
            display = DiceRollDisplay.model_validate(item)
            key = display.model_dump_json()
            if key not in seen_dice_rolls:
                seen_dice_rolls.add(key)
                dice_rolls.append(display)
        for item in payload.experience_awards:
            award = DndExperienceAwardDisplay.model_validate(item)
            key = award.model_dump_json()
            if key not in seen_experience_awards:
                seen_experience_awards.add(key)
                experience_awards.append(award)
        if payload.owner_error:
            prose_by_pov.setdefault(pov_id, []).append(payload.owner_error)

    per_player_renders = {
        pov_id: "\n\n".join(parts)
        for pov_id, parts in prose_by_pov.items()
        if parts
    }
    per_player_assets = {
        pov_id: list(items) for pov_id, items in assets_by_pov.items() if items
    }
    return TurnResponse(
        session_id=session_id,
        checkpoint_id=checkpoint_id,
        turn_index=turn_index,
        output_text=per_player_renders.get(acting_character_id, ""),
        per_player_renders=per_player_renders,
        per_player_visual_novel_renders=visual_by_pov,
        asset_reveals=list(per_player_assets.get(acting_character_id, [])),
        per_player_asset_reveals=per_player_assets,
        pause_reason=pause_reason,
        deliveries=deliveries,
        reaction_prompts=reaction_prompts,
        loot_prompts=loot_prompts,
        commitment_revision_prompts=commitment_prompts,
        dice_rolls=dice_rolls,
        experience_awards=experience_awards,
    )
