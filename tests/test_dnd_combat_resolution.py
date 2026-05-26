import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine import dice, dnd_cat_ii as cat, dnd_combat
from app.engine.dnd_roll_display import dice_roll_displays_since
from app.engine.dnd_combat_resolution import (
    COMBAT_MANAGER_FINALIZE_MAX_TOKENS,
    COMBAT_MANAGER_PLAN_MAX_TOKENS,
    DndCombatResolver,
    _merge_content_context_records,
    _scrub_visible_bookkeeping,
    _scrub_private_outcome_leaks,
)
from app.llm.client import LLMResponse
from app.schemas.characters import CharacterRecord, CharacterStatus, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_cat_ii import (
    DndCombatActionUse,
    DndCombatCasting,
    DndCombatManagerAdjudication,
    DndCombatTurnPlan,
    DndPlannedActionRoll,
    DndPlannedResourceSpend,
    PlannedRoll,
    RulesAdjudication,
)
from app.schemas.content_privacy import REDACTED_IMPORT_SENTINEL
from app.schemas.dnd_spatial import (
    DndAreaTemplate,
    DndBattleMapState,
    DndBattleMapToken,
)
from app.schemas.state import (
    CatIIRollTransaction,
    DndCombatantState,
    DndCombatState,
    DndRuntimeEffect,
    SessionState,
    SlotEntry,
    WorldState,
)


def _llm_response(parsed) -> LLMResponse:
    raw = MagicMock()
    raw.content = []
    raw.model = "gpt-5.2"
    return LLMResponse(
        parsed=parsed,
        raw_response=raw,
        content="{}",
        model="gpt-5.2",
    )


_SPELL_SOURCE_IDS = {
    "acid_splash",
    "burning_hands",
    "call_lightning",
    "cone_of_cold",
    "fireball",
    "fire_bolt",
    "guiding_bolt",
    "hold_person",
    "magic_missile",
    "misty_step",
    "scorching_ray",
}


def test_combat_content_context_merge_redacts_imported_asset_sentinels():
    sentinels = [
        "delivery_ref=asset://synthetic/hidden-map",
        "/private/table/source-map.png",
        "raw_ocr=PROTECTED_SOURCE_EXCERPT",
    ]
    context = {
        "content_context": [
            "content_known ref=room summary=\"Existing surface\" "
            + " ".join(sentinels[:1])
        ]
    }

    _merge_content_context_records(
        context,
        ["front_signal ref=front summary=\"Visible pressure\" " + " ".join(sentinels)],
    )
    flat = json.dumps(context, sort_keys=True)

    for sentinel in sentinels:
        assert sentinel not in flat
    assert REDACTED_IMPORT_SENTINEL in flat
    assert "Existing surface" in flat
    assert "Visible pressure" in flat


def test_scrub_private_outcome_leaks_removes_public_and_router_duplicates():
    adjudication = DndCombatManagerAdjudication(
        feasible=True,
        combat_status="ongoing",
        mechanical_summary="The private perception takes hold.",
        visible_outcome_facts=[
            "Sera casts a spell at the Orc Raider.",
            (
                "The Orc Raider turns as if an iron portcullis blocks the "
                "east arch."
            ),
            "The Orc Raider acts as if something obstructs the way.",
        ],
        private_outcome_facts=[
            {
                "text": "You see an iron portcullis block the east arch.",
                "visible_to": ["orc_raider"],
            }
        ],
        state_deltas=[],
        combat_state_deltas=[],
        effect_deltas=[],
        spatial_deltas=[],
        rules_notes=[],
        fallback_reason="",
        router_observed_facts=[
            {
                "fact": "Sera made the orc perceive the portcullis.",
                "salience": "notable",
                "reason": "The private belief may matter later.",
            }
        ],
    )

    _scrub_private_outcome_leaks(adjudication)

    assert adjudication.visible_outcome_facts == [
        "Sera casts a spell at the Orc Raider."
    ]
    assert adjudication.router_observed_facts == []


def test_scrub_visible_bookkeeping_removes_concentration_fact():
    adjudication = DndCombatManagerAdjudication(
        feasible=True,
        combat_status="ongoing",
        mechanical_summary="The action resolves.",
        visible_outcome_facts=[
            "Sera begins concentrating to maintain the effect.",
            "Sera gestures toward the Orc Raider.",
        ],
        private_outcome_facts=[],
        state_deltas=[],
        combat_state_deltas=[],
        effect_deltas=[],
        spatial_deltas=[],
        rules_notes=[],
        fallback_reason="",
        router_observed_facts=[],
    )

    _scrub_visible_bookkeeping(adjudication)

    assert adjudication.visible_outcome_facts == [
        "Sera gestures toward the Orc Raider."
    ]


def test_scrub_visible_bookkeeping_rewrites_save_and_dash_terms():
    adjudication = DndCombatManagerAdjudication(
        feasible=True,
        combat_status="ongoing",
        mechanical_summary="The action resolves.",
        visible_outcome_facts=[
            (
                "The Void Intruder fails the Sickening Radiance Constitution "
                "save and is seared by radiant light."
            ),
            (
                "The Void Intruder succeeds on the Cloudkill Constitution "
                "save and resists the cloud's poison."
            ),
            "Hurt but moving, the Void Intruder dashes toward the hatch.",
        ],
        private_outcome_facts=[],
        state_deltas=[],
        combat_state_deltas=[],
        effect_deltas=[],
        spatial_deltas=[],
        rules_notes=[],
        fallback_reason="",
        router_observed_facts=[],
    )

    _scrub_visible_bookkeeping(adjudication)

    assert adjudication.visible_outcome_facts == [
        "The Void Intruder is seared by radiant light.",
        "The Void Intruder resists the cloud's poison.",
        "Hurt but moving, the Void Intruder rushes toward the hatch.",
    ]


def test_scrub_private_outcome_leaks_does_not_match_character_name_substrings():
    adjudication = DndCombatManagerAdjudication(
        feasible=True,
        combat_status="ongoing",
        mechanical_summary="The private perception takes hold.",
        visible_outcome_facts=[
            "Sera Illusionist casts a spell at the Orc Raider.",
        ],
        private_outcome_facts=[
            {
                "text": "You see an iron portcullis block the east arch.",
                "visible_to": ["orc_raider"],
            }
        ],
        state_deltas=[],
        combat_state_deltas=[],
        effect_deltas=[],
        spatial_deltas=[],
        rules_notes=[],
        fallback_reason="",
        router_observed_facts=[],
    )

    _scrub_private_outcome_leaks(adjudication)

    assert adjudication.visible_outcome_facts == [
        "Sera Illusionist casts a spell at the Orc Raider."
    ]


def _turn_plan(
    *,
    needs_rolls: bool,
    roll_requests: list[PlannedRoll],
    no_roll_reason: str,
    actor_id: str = "alice",
) -> DndCombatTurnPlan:
    _ = needs_rolls
    actions_by_key: dict[tuple[str, str, str, str], DndCombatActionUse] = {}
    if not roll_requests:
        return DndCombatTurnPlan(
            feasible=True,
            actions=[
                DndCombatActionUse(
                    actor_id=actor_id,
                    source_type="speech",
                    source_id="",
                    use_mode="speak",
                    economy="free",
                    rolls=[],
                    reason=no_roll_reason or "No roll needed.",
                )
            ],
            no_action_reason=no_roll_reason,
        )
    for request in roll_requests:
        action = _action_from_planned_roll(request)
        key = (
            action.actor_id,
            action.source_type,
            action.source_id,
            action.effect_id,
        )
        if key not in actions_by_key:
            actions_by_key[key] = action
        actions_by_key[key].rolls.append(_action_roll_from_planned_roll(request))
    return DndCombatTurnPlan(
        feasible=True,
        actions=list(actions_by_key.values()),
        no_action_reason=no_roll_reason,
    )


def _action_from_planned_roll(request: PlannedRoll) -> DndCombatActionUse:
    source_id = request.action_id
    source_type = "action"
    use_mode = "attack" if request.kind == "attack_roll" else "activate"
    economy = "action"
    actor_id = request.actor_id
    if request.effect_id:
        source_type = "effect"
        use_mode = "release"
        economy = "reaction"
    elif source_id == "spell":
        source_type = "speech"
    elif source_id in _SPELL_SOURCE_IDS:
        source_type = "spell"
        use_mode = "cast"
    elif not source_id and request.kind == "saving_throw":
        source_type = "speech"
        source_id = (
            "hold_person"
            if "hold person" in request.reason.lower()
            else "spell"
        )
        actor_id = "alice"
    elif not source_id and request.kind == "attack_roll":
        reason_text = request.reason.lower()
        if "shortbow" in reason_text:
            source_id = "shortbow"
        elif "longsword" in reason_text:
            source_id = "longsword"
        else:
            source_id = "blade"
    if "opportunity" in request.reason.lower():
        economy = "none"
    return DndCombatActionUse(
        actor_id=actor_id,
        source_type=source_type,
        source_id=source_id,
        effect_id=request.effect_id,
        use_mode=use_mode,
        economy=economy,
        rolls=[],
        reason=request.reason,
    )


def _action_roll_from_planned_roll(request: PlannedRoll) -> DndPlannedActionRoll:
    roller_id = (
        request.target_id
        if request.kind == "saving_throw" and request.target_id
        else request.actor_id
    )
    return DndPlannedActionRoll(
        roll_id=request.roll_id,
        kind=request.kind,
        roller_id=roller_id,
        target_id=request.target_id,
        ability=request.ability,
        skill=request.skill,
        dc=request.dc,
        opposed_by=request.opposed_by,
        advantage_state=request.advantage_state,
        modifier_bonus=request.modifier_bonus,
        modifier_bonus_reason=request.modifier_bonus_reason,
        damage_on_save_success=request.damage_on_save_success,
        damage_adjustments=list(request.damage_adjustments),
        reason=request.reason,
    )


def _spell_turn_plan(
    *,
    source_id: str,
    cast_level: int,
    resource_id: str,
    roll_requests: list[PlannedRoll] | None = None,
    economy: str = "action",
    reason: str = "",
) -> DndCombatTurnPlan:
    spends = (
        [
            DndPlannedResourceSpend(
                resource_id=resource_id,
                amount=1,
                reason=reason or f"cast {source_id}",
            )
        ]
        if resource_id else []
    )
    return DndCombatTurnPlan(
        feasible=True,
        actions=[
            DndCombatActionUse(
                actor_id="alice",
                source_type="spell",
                source_id=source_id,
                use_mode="cast",
                economy=economy,
                casting=DndCombatCasting(cast_level=cast_level),
                resource_spends=spends,
                rolls=[
                    _action_roll_from_planned_roll(request)
                    for request in roll_requests or []
                ],
                reason=reason or f"cast {source_id}",
            )
        ],
        no_action_reason="",
    )


def _character(
    character_id: str,
    name: str,
    *,
    attack_bonus: int = 0,
    damage: str = "",
    actions: list[dict] | None = None,
    defenses: dict | None = None,
    spellcasting: dict | None = None,
    faction: str = "",
) -> CharacterRecord:
    if actions is None:
        actions = []
    if damage and not actions:
        actions.append({
            "id": "blade",
            "name": "Blade",
            "attack": {"bonus": attack_bonus, "damage": damage},
        })
    return CharacterRecord(
        character_id=character_id,
        name=name,
        public_sheet=PublicSheet(role="combatant", faction=faction),
        mechanics={
            "ruleset_id": "dnd5e_basic",
            "ability_scores": {
                "str": 10,
                "dex": 10,
                "con": 10,
                "int": 10,
                "wis": 10,
                "cha": 10,
            },
            "armor_class": 12,
            "hit_points": {"current": 13, "max": 13, "temporary": 0},
            "dnd5e_sheet": {
                "statblock": {
                    "actions": actions,
                    "defenses": defenses or {},
                    "spellcasting": spellcasting or {},
                }
            },
        },
    )


def _spell(
    spell_id: str,
    name: str,
    *,
    level: int,
    attack_bonus: int | None = None,
    save_ability: str = "",
    dc: int = 0,
    damage: str = "",
    healing: str = "",
    consumes_level: int | None = None,
    concentration: bool = False,
) -> dict:
    consumes = []
    if consumes_level is not None:
        consumes.append({
            "resource_id": f"spell_slot_{consumes_level}",
            "amount": 1,
        })
    return {
        "id": spell_id,
        "name": name,
        "level": level,
        "prepared": True,
        "always_prepared": False,
        "concentration": concentration,
        "attack": {"bonus": attack_bonus} if attack_bonus is not None else {},
        "save": {"ability": save_ability, "dc": dc} if save_ability else {},
        "damage": [{"formula": damage}] if damage else [],
        "healing": [{"formula": healing}] if healing else [],
        "consumes": consumes,
    }


def _ckpt() -> CheckpointFile:
    ckpt = CheckpointFile(
        session=SessionState(
            session_id="s",
            character_bindings={"alice": "1"},
        ),
        world_state=WorldState(),
        characters=[
            _character("alice", "Alice", attack_bonus=5, damage="1d8+3 slashing"),
            _character("bob", "Bob"),
        ],
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    ckpt.session.active_combat = DndCombatState(
        combat_id="test",
        round_number=1,
        turn_index=0,
        combatants=[
            DndCombatantState(
                combatant_id="alice",
                character_id="alice",
                name="Alice",
                player_controlled=True,
                armor_class=12,
                hit_points_current=13,
                hit_points_max=13,
            ),
            DndCombatantState(
                combatant_id="bob",
                character_id="bob",
                name="Bob",
                armor_class=12,
                hit_points_current=13,
                hit_points_max=13,
            ),
        ],
    )
    return ckpt


def _give_alice_readied_magic_missile(ckpt: CheckpointFile) -> None:
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{
            "id": "class_1",
            "name": "Wizard",
            "ability": "int",
            "spell_attack_bonus": 5,
        }],
        "slots": {"1": {"current": 2, "max": 4}},
        "spells": [
            _spell(
                "magic_missile",
                "Magic Missile",
                level=1,
                damage="3 darts, each 1d4+1 force",
                consumes_level=1,
            )
        ],
    }
    alice = ckpt.session.active_combat.combatants[0]
    alice.reaction_available = True
    alice.active_effects.append(DndRuntimeEffect(
        effect_id="ready_magic_missile",
        name="Readied Magic Missile",
        slug="readied_spell",
        source_type="spell",
        source_id="magic_missile",
        originator_id="alice",
        target_id="alice",
        concentration=True,
        duration_kind="rounds",
        duration_amount=1,
        remaining_rounds=1,
        duration_text="until the trigger or start of next turn",
        metadata={
            "readied_action": {
                "source_id": "magic_missile",
                "source_type": "spell",
                "readying_actor_id": "alice",
                "trigger_text": "when Bob opens the door",
                "created_round": 1,
                "created_turn_index": 0,
                "requires_reaction": True,
                "expires_at_start_of_actor_turn": True,
            },
        },
    ))


def _planned_attack(
    *,
    action_id: str = "blade",
    target_id: str = "bob",
    reason: str = "Alice attacks Bob with a blade.",
    damage_adjustments: list[dict] | None = None,
) -> PlannedRoll:
    data = {
        "roll_id": "attack_alice",
        "actor_id": "alice",
        "kind": "attack_roll",
        "ability": "str",
        "skill": "",
        "dc": 12,
        "opposed_by": "",
        "advantage_state": "normal",
        "reason": reason,
        "action_id": action_id,
        "target_id": target_id,
    }
    if damage_adjustments is not None:
        data["damage_adjustments"] = damage_adjustments
    return PlannedRoll(**data)


def _basic_attack_mocks(request: PlannedRoll) -> tuple[MagicMock, MagicMock]:
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[request],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            mechanical_summary="Alice's attack hits Bob.",
            visible_outcome_facts=["Alice's blade hits Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]
    return client, prompt_mgr


def test_combat_resolver_rolls_attack_damage_and_applies_hp(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters.append(_character("charlie", "Charlie"))
    ckpt.session.active_combat.combatants.append(
        DndCombatantState(
            combatant_id="charlie",
            character_id="charlie",
            name="Charlie",
            armor_class=12,
            hit_points_current=0,
            hit_points_max=13,
            defeat_state="defeated",
        )
    )
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )

    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="attack_alice",
                    actor_id="alice",
                    kind="attack_roll",
                    ability="str",
                    skill="",
                    dc=12,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Alice attacks Bob with a blade.",
                    action_id="",
                    target_id="bob",
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            mechanical_summary="Alice's attack hits Bob.",
            visible_outcome_facts=["Alice cuts Bob across the guard."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    bob = ckpt.session.active_combat.combatants[1]
    assert bob.hit_points_current == 6
    assert routed.event_kind == "ruleset_resolution"
    assert routed.canonical_event.observable_facts[0].text == (
        "Alice cuts Bob across the guard."
    )
    assert "charlie" not in {observer.character_id for observer in routed.observers}
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.source == "combat"
    assert transaction.status == "finalized"
    assert transaction.rolls[0].modifier == 5
    assert transaction.damage_records[0].roll_id == "attack_alice"
    assert transaction.damage_records[0].raw_amount == 7
    assert transaction.damage_records[0].amount == 7
    assert transaction.damage_records[0].damage_type == "slashing"
    assert transaction.damage_records[0].adjustments == []
    assert transaction.damage_records[0].target_hp_before == 13
    assert transaction.damage_records[0].target_hp_after == 6
    assert transaction.damage_records[0].target_hp_max == 13
    assert transaction.damage_records[0].target_defeat_state_after == "active"
    assert transaction.damage_records[0].applied is True
    assert any("damage_for=attack_alice" in line for line in transaction.ledger_lines)
    assert ckpt.session.pending_engine_state_updates == []
    assert ckpt.session_conversation == []
    assert [
        call.args[0] for call in prompt_mgr.render_messages.call_args_list
    ] == ["dnd_combat_manager", "dnd_combat_manager"]
    assert prompt_mgr.render_messages.call_args_list[0].kwargs["phase"] == "PLAN_TURN"
    assert "planned_actions_block" in prompt_mgr.render_messages.call_args_list[1].kwargs
    final_packet = json.loads(
        prompt_mgr.render_messages.call_args_list[1].kwargs[
            "combat_action_packet"
        ]
    )
    assert "mechanics" not in final_packet["combatants"][0]
    assert final_packet["used_sources"][0]["source"]["id"] == "blade"
    assert [
        call.kwargs["role"] for call in client.complete.await_args_list
    ] == ["dnd_combat_manager", "dnd_combat_manager"]
    assert client.complete.await_args_list[0].kwargs["max_tokens"] == (
        COMBAT_MANAGER_PLAN_MAX_TOKENS
    )
    assert client.complete.await_args_list[1].kwargs["max_tokens"] == (
        COMBAT_MANAGER_FINALIZE_MAX_TOKENS
    )
    assert (
        client.complete.await_args_list[1].kwargs["response_model"]
        is DndCombatManagerAdjudication
    )


def test_combat_resolver_dedupes_router_attack_damage_roll(monkeypatch):
    ckpt = _ckpt()
    values = iter([9, 4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )

    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(DndCombatTurnPlan(
            feasible=True,
            actions=[
                DndCombatActionUse(
                    actor_id="alice",
                    source_type="action",
                    source_id="blade",
                    use_mode="attack",
                    economy="action",
                    resource_spends=[],
                    rolls=[
                        DndPlannedActionRoll(
                            roll_id="attack_alice",
                            kind="attack_roll",
                            roller_id="alice",
                            target_id="bob",
                            ability="str",
                            skill="",
                            dc=12,
                            opposed_by="",
                            advantage_state="normal",
                            reason="Alice attacks Bob with a blade.",
                        ),
                        DndPlannedActionRoll(
                            roll_id="damage_alice",
                            kind="damage_roll",
                            roller_id="alice",
                            target_id="bob",
                            ability="str",
                            skill="",
                            dc=0,
                            opposed_by="",
                            advantage_state="normal",
                            reason="Blade damage if the attack hits.",
                        ),
                    ],
                    reason="Alice attacks Bob with a blade.",
                )
            ],
            no_action_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice's attack hits Bob.",
            visible_outcome_facts=["Alice cuts Bob across the guard."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    bob = ckpt.session.active_combat.combatants[1]
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert bob.hit_points_current == 5
    assert [record.roll_id for record in transaction.rolls] == ["attack_alice"]
    assert [damage.roll_id for damage in transaction.damage_records] == [
        "attack_alice"
    ]
    assert transaction.damage_records[0].amount == 8
    assert any(
        "skipped conditional damage_roll 'damage_alice'" in line
        for line in transaction.ledger_lines
    )
    assert not any(
        "damage_for=damage_alice" in line for line in transaction.ledger_lines
    )
    planned_actions = prompt_mgr.render_messages.call_args_list[1].kwargs[
        "planned_actions_block"
    ]
    assert "damage_alice" not in planned_actions


def test_combat_resolver_drops_router_damage_roll_when_attack_misses(monkeypatch):
    ckpt = _ckpt()
    values = iter([1])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )

    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(DndCombatTurnPlan(
            feasible=True,
            actions=[
                DndCombatActionUse(
                    actor_id="alice",
                    source_type="action",
                    source_id="blade",
                    use_mode="attack",
                    economy="action",
                    resource_spends=[],
                    rolls=[
                        DndPlannedActionRoll(
                            roll_id="attack_alice",
                            kind="attack_roll",
                            roller_id="alice",
                            target_id="bob",
                            ability="str",
                            skill="",
                            dc=12,
                            opposed_by="",
                            advantage_state="normal",
                            reason="Alice attacks Bob with a blade.",
                        ),
                        DndPlannedActionRoll(
                            roll_id="damage_alice",
                            kind="damage_roll",
                            roller_id="alice",
                            target_id="bob",
                            ability="str",
                            skill="",
                            dc=0,
                            opposed_by="",
                            advantage_state="normal",
                            reason="Blade damage if the attack hits.",
                        ),
                    ],
                    reason="Alice attacks Bob with a blade.",
                )
            ],
            no_action_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice's attack misses Bob.",
            visible_outcome_facts=["Alice's cut goes wide of Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    bob = ckpt.session.active_combat.combatants[1]
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert bob.hit_points_current == 13
    assert [record.roll_id for record in transaction.rolls] == ["attack_alice"]
    assert transaction.damage_records == []
    assert any(
        "skipped conditional damage_roll 'damage_alice'" in line
        for line in transaction.ledger_lines
    )
    assert any(
        "attack total 7 vs AC 12 -> miss" in line
        for line in transaction.ledger_lines
    )
    assert not any("damage_for=" in line for line in transaction.ledger_lines)


@pytest.mark.parametrize(
    ("source_id", "use_mode", "intention"),
    [
        ("dash", "activate", "I take the Dash action."),
        ("disengage", "activate", "I take the Disengage action."),
        ("dodge", "activate", "I take the Dodge action."),
        ("help", "activate", "I take the Help action."),
        ("hide", "activate", "I take the Hide action."),
        ("ready", "activate", "I take the Ready action."),
        ("search", "activate", "I take the Search action."),
        ("use_an_object", "interact", "I take the Use an Object action."),
    ],
)
def test_combat_resolver_accepts_universal_no_roll_actions(
    source_id: str,
    use_mode: str,
    intention: str,
):
    ckpt = _ckpt()
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(DndCombatTurnPlan(
            feasible=True,
            actions=[
                DndCombatActionUse(
                    actor_id="alice",
                    source_type="action",
                    source_id=source_id,
                    use_mode=use_mode,
                    economy="action",
                    rolls=[],
                    resource_spends=[],
                    reason=intention,
                )
            ],
            no_action_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            mechanical_summary="Alice focuses entirely on defense.",
            visible_outcome_facts=["Alice squares up behind her shield."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention=intention,
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.status == "finalized"
    assert transaction.rolls == []
    assert transaction.resource_spends == []
    ledger_source = "use_an_object" if source_id == "use_an_object" else source_id
    assert any(ledger_source in line for line in transaction.ledger_lines)
    assert routed.canonical_event.observable_facts[0].text == (
        "Alice squares up behind her shield."
    )


def test_combat_saving_throw_uses_target_modifier_and_cover_bonus(monkeypatch):
    ckpt = _ckpt()
    ckpt.session.character_bindings["bob"] = "2"
    ckpt.characters[1].mechanics["ability_scores"]["dex"] = 18
    values = iter([4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="save_bob_spell",
                    actor_id="alice",
                    kind="saving_throw",
                    ability="dex",
                    skill="",
                    dc=13,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Bob dodges Alice's spell.",
                    action_id="spell",
                    target_id="bob",
                    modifier_bonus=5,
                    modifier_bonus_reason="three-quarters cover",
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Bob rolls a Dexterity save.",
            visible_outcome_facts=["Bob ducks behind the slit as the spell hits."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast a spell at Bob.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    roll = transaction.rolls[0]
    assert roll.actor_id == "bob"
    assert roll.actor_control == "player"
    assert roll.modifier == 9
    assert "save_bob_spell: bob saving_throw from alice" in transaction.ledger_lines[0]
    assert "modifier bonus +5 (three-quarters cover)" in transaction.ledger_lines[0]


def test_combat_save_damage_rolls_once_and_applies_per_target(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        spellcasting={
            "profiles": [{
                "id": "class_1",
                "name": "Wizard",
                "ability": "int",
                "spell_save_dc": 10,
            }],
            "spells": [{
                "id": "fireball",
                "name": "Fireball",
                "level": 3,
                "save": {"ability": "dex", "dc": 10},
                "damage": [{"formula": "2d6 fire"}],
                "target": {"text": "20-foot-radius sphere"},
            }],
        },
    )
    ckpt.characters.append(_character("charlie", "Charlie"))
    ckpt.session.active_combat.combatants.append(
        DndCombatantState(
            combatant_id="charlie",
            character_id="charlie",
            name="Charlie",
            armor_class=12,
            hit_points_current=13,
            hit_points_max=13,
        )
    )
    values = iter([14, 4, 2, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="save_bob_fireball",
                    actor_id="alice",
                    kind="saving_throw",
                    ability="dex",
                    skill="",
                    dc=10,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Bob saves against Alice's Fireball.",
                    action_id="fireball",
                    target_id="bob",
                    damage_on_save_success="half",
                ),
                PlannedRoll(
                    roll_id="save_charlie_fireball",
                    actor_id="alice",
                    kind="saving_throw",
                    ability="dex",
                    skill="",
                    dc=10,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Charlie saves against Alice's Fireball.",
                    action_id="fireball",
                    target_id="charlie",
                    damage_on_save_success="half",
                ),
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice's Fireball detonates.",
            visible_outcome_facts=["Alice's Fireball catches Bob and Charlie."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Fireball on Bob and Charlie.",
        )
    )

    bob = next(c for c in ckpt.session.active_combat.combatants if c.character_id == "bob")
    charlie = next(
        c for c in ckpt.session.active_combat.combatants
        if c.character_id == "charlie"
    )
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert bob.hit_points_current == 10
    assert charlie.hit_points_current == 6
    assert len(transaction.damage_records) == 2
    assert {damage.raw_amount for damage in transaction.damage_records} == {7}
    assert {
        (damage.roll_id, damage.target_id, damage.amount, damage.applied)
        for damage in transaction.damage_records
    } == {
        ("save_bob_fireball", "bob", 3, True),
        ("save_charlie_fireball", "charlie", 7, True),
    }
    assert sum(
        1 for line in transaction.ledger_lines
        if line.startswith("damage_for=")
    ) == 2


def test_damage_engine_owns_defeat_condition_deltas(monkeypatch):
    ckpt = _ckpt()
    bob = ckpt.session.active_combat.combatants[1]
    bob.hit_points_current = 5
    values = iter([9, 7])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[_planned_attack()],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice drops Bob with a blade.",
            visible_outcome_facts=["Alice's blade drops Bob."],
            state_deltas=[],
            combat_state_deltas=[{
                "kind": "condition_add",
                "target_id": "bob",
                "amount": 0,
                "condition": "unconscious",
                "reason": "Bob reached 0 HP.",
            }],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    assert bob.hit_points_current == 0
    assert bob.defeat_state == "defeated"
    assert "unconscious" not in bob.conditions
    assert any(
        "damage engine owns 'unconscious'" in line
        for line in ckpt.session.active_combat.audit_lines
    )


def test_combat_packet_includes_battle_map_and_spatial_deltas_apply():
    ckpt = _ckpt()
    ckpt.session.active_combat.battle_map = DndBattleMapState(
        present=True,
        map_name="Bridge",
        width=8,
        height=5,
        square_size_ft=5,
        tokens=[
            DndBattleMapToken(
                token_id="alice",
                character_id="alice",
                label="Alice",
                x=0,
                y=0,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id="bob",
                character_id="bob",
                label="Bob",
                x=5,
                y=0,
                size_squares=1,
            ),
        ],
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=False,
            roll_requests=[],
            no_roll_reason="No roll.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            mechanical_summary="Alice moves closer.",
            visible_outcome_facts=["Alice closes the distance."],
            state_deltas=[],
            combat_state_deltas=[],
            spatial_deltas=[
                {
                    "kind": "move_token",
                    "target_id": "alice",
                    "character_id": "alice",
                    "x": 2,
                    "y": 1,
                    "size_squares": 1,
                    "label": "Alice",
                    "shape": "",
                    "radius_squares": 0,
                    "width": 1,
                    "height": 1,
                    "duration_rounds": 0,
                    "reason": "Alice moved.",
                }
            ],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(DndCombatResolver(client, prompt_mgr).resolve_combat_action(
        ckpt=ckpt,
        actor_id="alice",
        intention="I move toward Bob.",
    ))

    packet = json.loads(
        prompt_mgr.render_messages.call_args_list[0].kwargs[
            "combat_action_packet"
        ]
    )
    assert packet["tactical_map"]["map_name"] == "Bridge"
    assert packet["tactical_map"]["targets"][0]["character_id"] == "bob"
    assert packet["tactical_map"]["targets"][0]["distance_ft"] == 25
    alice = next(
        token for token in ckpt.session.active_combat.battle_map.tokens
        if token.character_id == "alice"
    )
    assert (alice.x, alice.y) == (2, 1)


def test_combat_packet_trims_non_actor_inventory_resources_and_raw():
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["resources"] = {"superiority_dice": 4}
    ckpt.characters[0].mechanics["raw"] = {"actor_secret": True}
    ckpt.characters[1].mechanics["resources"] = {"spell_slots": {"1": 2}}
    ckpt.characters[1].mechanics["raw"] = {"enemy_secret": True}
    ckpt.characters[1].mechanics["dnd5e_sheet"]["statblock"]["inventory"] = {
        "items": [{"id": "amulet", "name": "Hidden Amulet"}],
        "currency": {"gp": 99},
    }
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=False,
            roll_requests=[],
            no_roll_reason="No roll needed.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            mechanical_summary="Alice waits.",
            visible_outcome_facts=["Alice waits."],
            state_deltas=[],
            combat_state_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(DndCombatResolver(client, prompt_mgr).resolve_combat_action(
        ckpt=ckpt,
        actor_id="alice",
        intention="I wait.",
    ))

    packet = json.loads(
        prompt_mgr.render_messages.call_args_list[0].kwargs[
            "combat_action_packet"
        ]
    )
    by_id = {
        combatant["character_id"]: combatant
        for combatant in packet["combatants"]
    }
    assert {action["id"] for action in packet["standard_combat_actions"]} >= {
        "dash",
        "dodge",
        "grapple",
    }
    assert "dash" not in {action["id"] for action in by_id["alice"]["actions"]}
    assert "dash" not in {action["id"] for action in by_id["bob"]["actions"]}
    assert "resources" in by_id["alice"]["mechanics"]
    assert "inventory" in by_id["alice"]["mechanics"]
    assert "raw" in by_id["alice"]["mechanics"]
    assert "resources" not in by_id["bob"]["mechanics"]
    assert "inventory" not in by_id["bob"]["mechanics"]
    assert "raw" not in by_id["bob"]["mechanics"]
    assert by_id["bob"]["mechanics"]["defenses"] == {}


def test_combat_packet_marks_current_actor_relationships():
    ckpt = _ckpt()
    ckpt.characters[1] = _character("bob", "Bob", faction="ash_cult")
    ckpt.characters.append(_character("eve", "Eve", faction="ash_cult"))
    ckpt.session.active_combat.turn_index = 1
    ckpt.session.active_combat.combatants[1].name = "Bob"
    ckpt.session.active_combat.combatants.append(DndCombatantState(
        combatant_id="eve",
        character_id="eve",
        name="Eve",
        armor_class=12,
        hit_points_current=13,
        hit_points_max=13,
    ))
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=False,
            roll_requests=[],
            no_roll_reason="No roll needed.",
            actor_id="bob",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            mechanical_summary="Bob waits.",
            visible_outcome_facts=["Bob waits."],
            state_deltas=[],
            combat_state_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(DndCombatResolver(client, prompt_mgr).resolve_combat_action(
        ckpt=ckpt,
        actor_id="bob",
        intention="I wait.",
    ))

    packet = json.loads(
        prompt_mgr.render_messages.call_args_list[0].kwargs[
            "combat_action_packet"
        ]
    )
    by_id = {
        combatant["character_id"]: combatant
        for combatant in packet["combatants"]
    }
    assert by_id["bob"]["relationship_to_current_actor"] == "self"
    assert by_id["alice"]["relationship_to_current_actor"] == "enemy"
    assert by_id["alice"]["enemy_to_current_actor"] is True
    assert by_id["eve"]["relationship_to_current_actor"] == "ally"
    assert by_id["eve"]["enemy_to_current_actor"] is False
    assert by_id["eve"]["faction"] == "ash_cult"
    assert "reaction_available" not in by_id["eve"]
    assert (
        "Automatic opportunity attacks do not require or spend combat reactions."
        in packet["house_rules"]
    )


def test_combat_resolver_observes_target_dropped_by_same_event(monkeypatch):
    ckpt = _ckpt()
    ckpt.session.character_bindings["bob"] = "2"
    bob = ckpt.session.active_combat.combatants[1]
    bob.player_controlled = True
    bob.hit_points_current = 5
    bob.hit_points_max = 13
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack())

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    assert bob.hit_points_current == 0
    assert bob.defeat_state == "down"
    assert "bob" in {observer.character_id for observer in routed.observers}


def test_combat_damage_applies_sheet_resistance_after_roll(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[1] = _character(
        "bob",
        "Bob",
        defenses={
            "damage_resistances": [
                {"id": "slashing", "name": "Slashing", "condition": ""}
            ],
        },
    )
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack())

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    bob = ckpt.session.active_combat.combatants[1]
    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    assert bob.hit_points_current == 10
    assert damage.raw_amount == 7
    assert damage.amount == 3
    assert damage.damage_type == "slashing"
    assert damage.adjustments[0].source == "sheet"
    assert damage.adjustments[0].kind == "resistance"
    assert damage.adjustments[0].amount_before == 7
    assert damage.adjustments[0].amount_after == 3


def test_combat_damage_applies_sheet_immunity(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[1] = _character(
        "bob",
        "Bob",
        defenses={
            "damage_immunities": [
                {"id": "slashing", "name": "Slashing", "condition": ""}
            ],
        },
    )
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack())

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    bob = ckpt.session.active_combat.combatants[1]
    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    assert bob.hit_points_current == 13
    assert damage.raw_amount == 7
    assert damage.amount == 0
    assert damage.adjustments[0].kind == "immunity"
    assert damage.adjustments[0].amount_before == 7
    assert damage.adjustments[0].amount_after == 0
    assert damage.applied is True


def test_combat_damage_applies_router_vulnerability(monkeypatch):
    ckpt = _ckpt()
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        damage_adjustments=[
            {
                "kind": "vulnerability",
                "damage_type": "slashing",
                "reason": "Bob is exposed by a temporary effect.",
            }
        ],
    ))

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    bob = ckpt.session.active_combat.combatants[1]
    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    assert bob.hit_points_current == 0
    assert damage.raw_amount == 7
    assert damage.amount == 14
    assert damage.adjustments[0].source == "router"
    assert damage.adjustments[0].kind == "vulnerability"
    assert damage.adjustments[0].amount_before == 7
    assert damage.adjustments[0].amount_after == 14


def test_combat_damage_applies_router_halving(monkeypatch):
    ckpt = _ckpt()
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        damage_adjustments=[
            {
                "kind": "halve",
                "damage_type": "slashing",
                "reason": "Bob uses a damage-halving reaction.",
            }
        ],
    ))

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    bob = ckpt.session.active_combat.combatants[1]
    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    assert bob.hit_points_current == 10
    assert damage.raw_amount == 7
    assert damage.amount == 3
    assert damage.adjustments[0].source == "router"
    assert damage.adjustments[0].kind == "halve"
    assert damage.adjustments[0].amount_before == 7
    assert damage.adjustments[0].amount_after == 3


def test_falling_collision_split_damage_applies_to_falling_actor(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[{
            "id": "falling_collision",
            "name": "Falling Collision",
            "attack": {"bonus": 0, "damage": "4d6 bludgeoning"},
        }],
    )
    ckpt.session.active_combat.combatants[0].hit_points_current = 13
    ckpt.session.active_combat.combatants[1].hit_points_current = 13
    values = iter([3, 3, 3, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="fall_damage",
                    actor_id="alice",
                    kind="damage_roll",
                    ability="str",
                    skill="",
                    dc=0,
                    opposed_by="",
                    advantage_state="normal",
                    reason="falling damage divided between Alice and Bob",
                    action_id="falling_collision",
                    target_id="bob",
                    damage_adjustments=[{
                        "kind": "halve",
                        "damage_type": "bludgeoning",
                        "reason": (
                            "Falling damage is divided evenly between the "
                            "falling creature and the lower creature."
                        ),
                    }],
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice falls onto Bob.",
            visible_outcome_facts=["Alice crashes into Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I fall onto Bob.",
        )
    )

    alice = ckpt.session.active_combat.combatants[0]
    bob = ckpt.session.active_combat.combatants[1]
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert alice.hit_points_current == 5
    assert bob.hit_points_current == 5
    assert [
        (damage.roll_id, damage.target_id, damage.amount, damage.applied)
        for damage in transaction.damage_records
    ] == [
        ("fall_damage", "bob", 8, True),
        ("fall_damage_split_alice", "alice", 8, True),
    ]


def test_failed_falling_collision_save_rolls_split_damage(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[{
            "id": "falling_collision",
            "name": "Falling Collision",
            "attack": {"bonus": 0, "damage": "4d6 bludgeoning"},
        }],
    )
    ckpt.session.active_combat.combatants[0].hit_points_current = 13
    ckpt.session.active_combat.combatants[1].hit_points_current = 13
    values = iter([0, 3, 3, 3, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="fall_save",
                    actor_id="alice",
                    kind="saving_throw",
                    ability="dex",
                    skill="",
                    dc=15,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Bob avoids Alice falling into him.",
                    action_id="falling_collision",
                    target_id="bob",
                    damage_on_save_success="none",
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice falls onto Bob.",
            visible_outcome_facts=["Alice crashes into Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I fall onto Bob.",
        )
    )

    alice = ckpt.session.active_combat.combatants[0]
    bob = ckpt.session.active_combat.combatants[1]
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert alice.hit_points_current == 5
    assert bob.hit_points_current == 5
    assert [
        (damage.roll_id, damage.target_id, damage.amount, damage.applied)
        for damage in transaction.damage_records
    ] == [
        ("fall_save", "bob", 8, True),
        ("fall_save_split_alice", "alice", 8, True),
    ]


def test_combat_crit_damage_is_doubled_before_resistance(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[1] = _character(
        "bob",
        "Bob",
        defenses={
            "damage_resistances": [
                {"id": "slashing", "name": "Slashing", "condition": ""}
            ],
        },
    )
    values = iter([19, 3, 4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack())

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    bob = ckpt.session.active_combat.combatants[1]
    assert damage.expression == "2d8+3"
    assert damage.raw_amount == 12
    assert damage.amount == 6
    assert damage.adjustments[0].amount_before == 12
    assert damage.adjustments[0].amount_after == 6
    assert bob.hit_points_current == 7


def test_combat_damage_rolls_and_adjusts_multiple_damage_types(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "flame_blade",
                "name": "Flame Blade",
                "attack": {
                    "bonus": 5,
                    "damage": "1d8+3 slashing + 2d6 fire",
                },
            },
        ],
    )
    ckpt.characters[1] = _character(
        "bob",
        "Bob",
        defenses={
            "damage_resistances": ["slashing"],
            "damage_vulnerabilities": ["fire"],
        },
    )
    values = iter([9, 3, 1, 2])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        action_id="flame_blade",
        reason="Alice attacks Bob with a flame blade.",
    ))

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I strike Bob with the flame blade.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    bob = ckpt.session.active_combat.combatants[1]
    assert damage.raw_amount == 12
    assert damage.amount == 13
    assert damage.damage_type == "slashing, fire"
    assert [component.damage_type for component in damage.components] == [
        "slashing",
        "fire",
    ]
    assert [component.raw_amount for component in damage.components] == [7, 5]
    assert [component.amount for component in damage.components] == [3, 10]
    assert damage.components[0].adjustments[0].kind == "resistance"
    assert damage.components[1].adjustments[0].kind == "vulnerability"
    assert bob.hit_points_current == 0


def test_combat_damage_groups_same_type_before_resistance(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "fangs",
                "name": "Fangs",
                "attack": {"bonus": 5, "damage": "1d4 piercing + 1d4 piercing"},
            },
        ],
    )
    ckpt.characters[1] = _character(
        "bob",
        "Bob",
        defenses={"damage_resistances": ["piercing"]},
    )
    values = iter([9, 0, 0])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        action_id="fangs",
        reason="Alice bites Bob.",
    ))

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I bite Bob.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    bob = ckpt.session.active_combat.combatants[1]
    assert damage.raw_amount == 2
    assert damage.amount == 1
    assert damage.components[0].expression == "1d4+1d4"
    assert damage.components[0].raw_amount == 2
    assert damage.components[0].amount == 1
    assert bob.hit_points_current == 12


def test_combat_damage_parses_trailing_same_type_and_ignores_alternative(
    monkeypatch,
):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "blade",
                "name": "Blade",
                "attack": {"bonus": 5, "damage": "1d8 + 1d6 + 3 slashing"},
            },
        ],
    )
    values = iter([9, 3, 2])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack())

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    assert damage.expression == "1d8+1d6+3"
    assert damage.raw_amount == 10
    assert damage.damage_type == "slashing"

    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "blade",
                "name": "Blade",
                "attack": {
                    "bonus": 5,
                    "damage": (
                        "1d8 slashing or 1d10 slashing if used with two hands"
                    ),
                },
            },
        ],
    )
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack())

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    assert damage.expression == "1d8"
    assert damage.raw_amount == 4


def test_combat_damage_parses_static_monster_damage(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "bite",
                "name": "Bite",
                "attack": {"bonus": 5, "damage": "1 piercing"},
            }
        ],
    )
    values = iter([14])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        action_id="bite",
        reason="Alice bites Bob.",
    ))

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I bite Bob.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    bob = ckpt.session.active_combat.combatants[1]
    assert damage.raw_amount == 1
    assert damage.amount == 1
    assert damage.components[0].expression == "1"
    assert damage.components[0].damage_type == "piercing"
    assert bob.hit_points_current == 12


def test_combat_crit_doubles_every_damage_component(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "flame_blade",
                "name": "Flame Blade",
                "attack": {"bonus": 5, "damage": "1d8 slashing + 1d6 fire"},
            },
        ],
    )
    values = iter([19, 3, 4, 1, 2])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        action_id="flame_blade",
        reason="Alice attacks Bob with a flame blade.",
    ))

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I strike Bob with the flame blade.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    assert damage.expression == "2d8 + 2d6"
    assert damage.raw_amount == 14
    assert [component.expression for component in damage.components] == [
        "2d8",
        "2d6",
    ]


def test_combat_attack_total_adjustment_applies_after_components(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "flame_blade",
                "name": "Flame Blade",
                "attack": {"bonus": 5, "damage": "1d8 slashing + 1d6 fire"},
            },
        ],
    )
    values = iter([9, 0, 0])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        action_id="flame_blade",
        damage_adjustments=[
            {
                "kind": "halve",
                "damage_type": "all",
                "reason": "Uncanny Dodge.",
                "scope": "attack_total",
            }
        ],
    ))

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I strike Bob with the flame blade.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    assert [component.raw_amount for component in damage.components] == [1, 1]
    assert [component.amount for component in damage.components] == [1, 1]
    assert damage.raw_amount == 2
    assert damage.amount == 1
    assert damage.adjustments[0].source == "router"
    assert damage.adjustments[0].kind == "halve"
    assert damage.adjustments[0].amount_before == 2
    assert damage.adjustments[0].amount_after == 1


def test_combat_damage_ignores_empty_router_adjustment_type(monkeypatch):
    ckpt = _ckpt()
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        damage_adjustments=[
            {
                "kind": "halve",
                "damage_type": "",
                "reason": "model omitted the damage type.",
            }
        ],
    ))

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    bob = ckpt.session.active_combat.combatants[1]
    assert damage.raw_amount == 7
    assert damage.amount == 7
    assert damage.adjustments == []
    assert bob.hit_points_current == 6


def test_combat_damage_supports_string_defenses_and_skips_qualified_defenses(
    monkeypatch,
):
    ckpt = _ckpt()
    ckpt.characters[1] = _character(
        "bob",
        "Bob",
        defenses={"damage_resistances": ["slashing", "nonmagical fire"]},
    )
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack())

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    assert damage.amount == 3
    assert damage.adjustments[0].reason == "slashing"

    ckpt = _ckpt()
    ckpt.characters[1] = _character(
        "bob",
        "Bob",
        defenses={"damage_resistances": ["nonmagical slashing"]},
    )
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack())

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    damage = ckpt.session.cat_ii_roll_transactions[0].damage_records[0]
    assert damage.amount == 7
    assert damage.adjustments == []


def test_combat_resolver_can_end_combat_from_adjudication():
    ckpt = _ckpt()
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=False,
            roll_requests=[],
            no_roll_reason="Bob surrenders.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ended",
            mechanical_summary="Bob surrenders and Alice accepts.",
            visible_outcome_facts=["Bob drops his blade and Alice lowers hers."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I accept Bob's surrender.",
        )
    )

    assert ckpt.session.active_combat is None
    facts = [fact.text for fact in routed.canonical_event.observable_facts]
    assert "Bob drops his blade and Alice lowers hers." in facts
    assert any("D&D combat ends" in fact for fact in facts)


def test_combat_resolver_auto_ends_when_spawned_hostile_is_defeated(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        attack_bonus=5,
        damage="1d8+3 slashing",
        faction="expedition",
    )
    panther = _character("mon_panther_1", "Panther")
    panther.mechanics["combat_spawn"] = {
        "spawned": True,
        "monster_key": "panther",
    }
    ckpt.characters[1] = panther
    ckpt.session.active_combat = DndCombatState(
        combat_id="test",
        round_number=1,
        turn_index=0,
        combatants=[
            DndCombatantState(
                combatant_id="alice",
                character_id="alice",
                name="Alice",
                player_controlled=True,
                armor_class=12,
                hit_points_current=13,
                hit_points_max=13,
            ),
            DndCombatantState(
                combatant_id="mon_panther_1",
                character_id="mon_panther_1",
                name="Panther",
                armor_class=12,
                hit_points_current=5,
                hit_points_max=5,
            ),
        ],
    )
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        target_id="mon_panther_1",
        reason="Alice attacks the panther with a blade.",
    ))

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cut down the panther.",
        )
    )

    assert ckpt.session.active_combat is None
    assert panther.status == CharacterStatus.culled
    facts = [fact.text for fact in routed.canonical_event.observable_facts]
    assert "D&D combat ends." in facts
    assert any(
        "All hostile combat-spawned monsters are defeated" in note
        for note in routed.decision_rationale.split("; ")
    )


def test_combat_resolver_rolls_healing_and_syncs_character_hp(monkeypatch):
    values = iter([0])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        spellcasting={
            "profiles": [{"id": "cleric", "ability": "wis"}],
            "slots": {"1": {"current": 1, "max": 1}},
            "spells": [
                _spell(
                    "healing_word",
                    "Healing Word",
                    level=1,
                    healing="1d4+3",
                    consumes_level=1,
                )
            ],
        },
    )
    ckpt.session.config.settings.player_roll_mode = "interactive"
    bob = ckpt.characters[1]
    bob.mechanics["hit_points"] = {"current": 5, "max": 13, "temporary": 0}
    bob.mechanics["dnd5e_sheet"]["statblock"]["defenses"] = {
        "hit_points": {"current": 5, "max": 13, "temporary": 0}
    }
    ckpt.session.active_combat.combatants[1].hit_points_current = 5

    plan = DndCombatTurnPlan.model_validate({
        "feasible": True,
        "actions": [{
            "actor_id": "alice",
            "source_type": "spell",
            "source_id": "healing_word",
            "use_mode": "cast",
            "economy": "bonus_action",
            "casting": {"cast_level": 1},
            "resource_spends": [{
                "resource_id": "spell_slot_1",
                "amount": 1,
                "reason": "Healing Word spends a 1st-level slot.",
            }],
            "targeting": {"mode": "targets", "target_ids": ["bob"]},
            "rolls": [],
            "reason": "Alice heals Bob with Healing Word.",
        }],
        "no_action_reason": "",
    })
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(plan),
        _llm_response(DndCombatManagerAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice heals Bob.",
            visible_outcome_facts=["Alice speaks a quick healing word to Bob."],
            state_deltas=[],
            combat_state_deltas=[{
                "kind": "healing",
                "target_id": "bob",
                "amount": 99,
                "condition": "",
                "reason": "The manager restates the healing.",
            }],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
            router_observed_facts=[],
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Healing Word on Bob.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert [roll.request["kind"] for roll in transaction.rolls] == [
        "healing_roll"
    ]
    assert transaction.healing_records[0].amount == 4
    bob_combatant = ckpt.session.active_combat.combatants[1]
    assert bob_combatant.hit_points_current == 9
    assert bob.mechanics["hit_points"]["current"] == 9
    assert (
        bob.mechanics["dnd5e_sheet"]["statblock"]["defenses"]["hit_points"]
        ["current"]
        == 9
    )
    assert ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"][
        "spellcasting"
    ]["slots"]["1"]["current"] == 0
    displays = dice_roll_displays_since(ckpt, set())
    assert displays[0].kind == "healing_roll"
    assert displays[0].total == 4
    assert displays[0].target_hp_before == 5
    assert displays[0].target_hp_after == 9


def test_combat_resolver_ends_when_hostile_disengages_and_party_does_not_pursue():
    ckpt = _ckpt()
    ckpt.session.active_combat.turn_index = 1
    ckpt.characters[0].public_sheet.faction = "expedition"
    ckpt.characters[1].public_sheet.faction = "bandits"

    plan = DndCombatTurnPlan.model_validate({
        "feasible": True,
        "actions": [{
            "actor_id": "bob",
            "source_type": "movement",
            "source_id": "move",
            "use_mode": "move",
            "economy": "movement",
            "targeting": {"mode": "none", "target_ids": []},
            "rolls": [],
            "reason": "Bob flees out of sight down the passage.",
        }],
        "no_action_reason": "",
    })
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(plan),
        _llm_response(DndCombatManagerAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Bob flees out of sight.",
            visible_outcome_facts=["Bob flees out of sight down the passage."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
            router_observed_facts=[],
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="bob",
            intention="Alice holds fire and does not pursue while Bob flees.",
        )
    )

    assert ckpt.session.active_combat is None
    facts = [fact.text for fact in routed.canonical_event.observable_facts]
    assert "D&D combat ends." in facts
    assert any(
        "party is not pursuing" in note
        for note in routed.decision_rationale.split("; ")
    )


def test_group_withdrawal_auto_ends_when_party_declines_pursuit():
    ckpt = _ckpt()
    ckpt.characters[0].public_sheet.faction = "expedition"
    ckpt.characters[1].public_sheet.faction = "bandits"
    ckpt.characters.append(
        _character("bandit_ambusher_01", "Bandit Ambusher", faction="bandits")
    )
    ckpt.session.active_combat.combatants = [
        DndCombatantState(
            combatant_id="alice",
            character_id="alice",
            name="Alice",
            player_controlled=True,
            armor_class=12,
            hit_points_current=13,
            hit_points_max=13,
        ),
        DndCombatantState(
            combatant_id="bob",
            character_id="bob",
            name="Bandit Leader",
            armor_class=12,
            hit_points_current=13,
            hit_points_max=13,
        ),
        DndCombatantState(
            combatant_id="bandit_ambusher_01",
            character_id="bandit_ambusher_01",
            name="Bandit Ambusher",
            armor_class=12,
            hit_points_current=11,
            hit_points_max=11,
        ),
        DndCombatantState(
            combatant_id="bandit_ambusher_02",
            character_id="bandit_ambusher_02",
            name="Bandit Ambusher",
            armor_class=12,
            hit_points_current=0,
            hit_points_max=11,
            defeat_state="defeated",
        ),
    ]
    transaction = CatIIRollTransaction(
        transaction_id="txn",
        event_id="evt",
        source="combat",
        actor_id="bandit_ambusher_02",
        intention=(
            "Marlowe holds fire and does not pursue while the bandits withdrew "
            "toward the tree line."
        ),
    )
    adjudication = DndCombatManagerAdjudication(
        feasible=True,
        combat_status="ongoing",
        mechanical_summary="The last ambusher falls after an opportunity attack.",
        visible_outcome_facts=[
            (
                "The remaining bandits have withdrawn toward the tree line and "
                "none of them have doubled back."
            )
        ],
        state_deltas=[],
        combat_state_deltas=[],
        effect_deltas=[],
        spatial_deltas=[],
        rules_notes=[],
        fallback_reason="",
        router_observed_facts=[],
    )

    cat._auto_end_if_hostiles_disengaged(ckpt, transaction, adjudication)

    assert adjudication.combat_status == "ended"
    active_hostiles = [
        combatant for combatant in ckpt.session.active_combat.combatants
        if combatant.defeat_state == "active" and not combatant.player_controlled
    ]
    assert active_hostiles
    assert all(cat._combatant_is_marked_disengaged(c) for c in active_hostiles)
    assert any("party is not pursuing" in note for note in adjudication.rules_notes)


def test_current_withdrawal_ignores_negated_attack_pressure():
    ckpt = _ckpt()
    ckpt.session.active_combat.turn_index = 1
    ckpt.characters[0].public_sheet.faction = "expedition"
    ckpt.characters[1].public_sheet.faction = "bandits"
    transaction = CatIIRollTransaction(
        transaction_id="txn",
        event_id="evt",
        source="combat",
        actor_id="bob",
        intention="Alice tells everyone to hold position and not fire as Bob pulls back.",
    )
    adjudication = DndCombatManagerAdjudication(
        feasible=True,
        combat_status="ongoing",
        mechanical_summary=(
            "Bob spends his movement to withdraw into the cleft, breaking clear "
            "line of sight. No attacks or attack rolls were required."
        ),
        visible_outcome_facts=[
            "Bob backs into the shadowed corridor and vanishes from clear sight."
        ],
        state_deltas=[],
        combat_state_deltas=[],
        effect_deltas=[],
        spatial_deltas=[],
        rules_notes=[],
        fallback_reason="",
        router_observed_facts=[],
    )

    cat._auto_end_if_hostiles_disengaged(ckpt, transaction, adjudication)

    assert adjudication.combat_status == "ended"
    assert cat._combatant_is_marked_disengaged(
        ckpt.session.active_combat.combatants[1]
    )


def test_combat_end_queues_router_observed_continuity():
    ckpt = _ckpt()
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=False,
            roll_requests=[],
            no_roll_reason="Alice spares Bob.",
        )),
        _llm_response(DndCombatManagerAdjudication(
            feasible=True,
            combat_status="ended",
            mechanical_summary="Alice accepts Bob's surrender.",
            visible_outcome_facts=["Alice accepts Bob's surrender."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
            router_observed_facts=[
                {
                    "fact": "Alice publicly spared Bob after he surrendered.",
                    "salience": "major",
                    "reason": (
                        "This mercy changes how survivors are likely to treat "
                        "Alice after initiative ends."
                    ),
                },
            ],
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I spare Bob if he yields.",
        )
    )

    assert ckpt.session.active_combat is None
    assert ckpt.session.pending_engine_state_updates == [
        "Combat continuity [major]: Alice publicly spared Bob after he surrendered. "
        "Reason: This mercy changes how survivors are likely to treat Alice after "
        "initiative ends."
    ]
    facts = [fact.text for fact in routed.canonical_event.observable_facts]
    assert "Alice accepts Bob's surrender." in facts
    assert "D&D combat ends." in facts


def test_combat_end_includes_queued_death_fact(monkeypatch):
    ckpt = _ckpt()
    ckpt.session.character_bindings["bob"] = "2"
    ckpt.characters.append(_character("charlie", "Charlie"))
    ckpt.session.active_act_slots["charlie"] = SlotEntry(
        reason="combat_blocked",
        trigger_event_id="evt_blocked",
    )
    bob = ckpt.session.active_combat.combatants[1]
    bob.hit_points_current = 1
    bob.hit_points_max = 1
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )

    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="attack_alice",
                    actor_id="alice",
                    kind="attack_roll",
                    ability="str",
                    skill="",
                    dc=12,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Alice attacks Bob with a blade.",
                    action_id="blade",
                    target_id="bob",
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ended",
            mechanical_summary="Alice's attack ends the fight.",
            visible_outcome_facts=["Alice's blade drops Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )

    facts = [fact.text for fact in routed.canonical_event.observable_facts]
    assert "Alice's blade drops Bob." in facts
    assert "Bob dies." in facts
    assert "D&D combat ends." in facts
    assert "charlie" not in {observer.character_id for observer in routed.observers}
    assert ckpt.session.active_combat is None
    assert ckpt.session.active_act_slots == {}
    assert ckpt.session.pending_engine_state_updates == [
        "Combat continuity [major]: Bob died during the combat. Reason: "
        "Character death is durable post-combat state."
    ]


def test_combat_damage_waits_for_successful_finalization(monkeypatch):
    ckpt = _ckpt()
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )

    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="attack_alice",
                    actor_id="alice",
                    kind="attack_roll",
                    ability="str",
                    skill="",
                    dc=12,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Alice attacks Bob with a blade.",
                    action_id="blade",
                    target_id="bob",
                )
            ],
            no_roll_reason="",
        )),
        RuntimeError("finalizer failed"),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    with pytest.raises(RuntimeError, match="finalizer failed"):
        asyncio.run(
            DndCombatResolver(client, prompt_mgr).resolve_combat_action(
                ckpt=ckpt,
                actor_id="alice",
                intention="I slash Bob with my blade.",
            )
        )

    bob = ckpt.session.active_combat.combatants[1]
    assert bob.hit_points_current == 13
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert any("damage_for=attack_alice" in line for line in transaction.ledger_lines)
    assert transaction.damage_records[0].amount == 7
    assert transaction.damage_records[0].applied is False


def test_combat_damage_retry_applies_persisted_damage_once(monkeypatch):
    ckpt = _ckpt()
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )

    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="attack_alice",
                    actor_id="alice",
                    kind="attack_roll",
                    ability="str",
                    skill="",
                    dc=12,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Alice attacks Bob with a blade.",
                    action_id="blade",
                    target_id="bob",
                )
            ],
            no_roll_reason="",
        )),
        RuntimeError("finalizer failed"),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]
    resolver = DndCombatResolver(client, prompt_mgr)

    with pytest.raises(RuntimeError, match="finalizer failed"):
        asyncio.run(
            resolver.resolve_combat_action(
                ckpt=ckpt,
                actor_id="alice",
                intention="I slash Bob with my blade.",
            )
        )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    bob = ckpt.session.active_combat.combatants[1]
    assert bob.hit_points_current == 13
    assert len(transaction.damage_records) == 1
    assert transaction.damage_records[0].amount == 7
    assert transaction.damage_records[0].applied is False

    def _fail_if_rerolled(_):
        raise AssertionError("retry should reuse persisted damage")

    monkeypatch.setattr(dice.d20.expression.random, "randrange", _fail_if_rerolled)
    client.complete = AsyncMock(return_value=_llm_response(RulesAdjudication(
        feasible=True,
        mechanical_summary="Alice's persisted hit resolves.",
        visible_outcome_facts=["Alice's blade bites into Bob."],
        state_deltas=[],
        combat_state_deltas=[],
        rules_notes=[],
        fallback_reason="",
    )))
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        resolver.continue_combat_transaction(
            ckpt=ckpt,
            event_id=transaction.event_id,
        )
    )

    assert bob.hit_points_current == 6
    assert len(transaction.damage_records) == 1
    assert transaction.damage_records[0].applied is True


def test_combat_state_delta_rejects_damage_kind():
    with pytest.raises(ValueError):
        RulesAdjudication(
            feasible=True,
            mechanical_summary="Bad damage delta.",
            visible_outcome_facts=["Alice hits Bob."],
            state_deltas=[],
            combat_state_deltas=[
                {
                    "kind": "damage",
                    "target_id": "bob",
                    "amount": 4,
                    "condition": "",
                    "reason": "duplicate damage",
                }
            ],
            rules_notes=[],
            fallback_reason="",
        )


def test_combat_packet_exposes_actions_and_empty_action_id_matches_reason(
    monkeypatch,
):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "longsword",
                "name": "Longsword",
                "attack": {"bonus": 6, "damage": "1d8+3 slashing"},
            },
            {
                "id": "shortbow",
                "name": "Shortbow",
                "attack": {
                    "bonus": 7,
                    "damage": "1d6+4 piercing",
                    "range": "80/320 ft",
                },
                "notes": "Ranged weapon attack.",
            },
        ],
    )
    values = iter([14, 2])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )

    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="attack_alice",
                    actor_id="alice",
                    kind="attack_roll",
                    ability="dex",
                    skill="",
                    dc=12,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Alice attacks Bob with a shortbow.",
                    action_id="",
                    target_id="bob",
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            mechanical_summary="Alice's shortbow hits Bob.",
            visible_outcome_facts=["Alice's arrow hits Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I shoot Bob with my shortbow.",
        )
    )

    first_packet = json.loads(
        prompt_mgr.render_messages.call_args_list[0].kwargs[
            "combat_action_packet"
        ]
    )
    alice = next(
        combatant for combatant in first_packet["combatants"]
        if combatant["character_id"] == "alice"
    )
    shortbow = next(
        action for action in alice["actions"]
        if action["id"] == "shortbow"
    )
    assert shortbow["damage"] == "1d6+4 piercing"
    assert shortbow["range"] == "80/320 ft"
    assert shortbow["notes"] == "Ranged weapon attack."
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.rolls[0].modifier == 7
    assert transaction.damage_records[0].expression == "1d6+4"


def test_attack_ledger_reports_effective_dc_when_cover_changes_threshold(
    monkeypatch,
):
    ckpt = _ckpt()
    values = iter([14, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    request = PlannedRoll(
        roll_id="attack_alice_cover",
        actor_id="alice",
        kind="attack_roll",
        ability="str",
        skill="",
        dc=14,
        opposed_by="",
        advantage_state="normal",
        reason="Bob has half cover from the low wall.",
        action_id="blade",
        target_id="bob",
    )
    client, prompt_mgr = _basic_attack_mocks(request)

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I attack Bob through cover.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert any(
        "vs effective DC 14 (base AC 12)" in line
        for line in transaction.ledger_lines
    )


def test_runtime_inventory_weapon_becomes_combat_action(monkeypatch):
    ckpt = _ckpt()
    alice = ckpt.characters[0]
    alice.mechanics["ability_scores"]["str"] = 16
    alice.mechanics["proficiency_bonus"] = 2
    statblock = alice.mechanics["dnd5e_sheet"]["statblock"]
    statblock["proficiencies"] = {"weapons": ["Martial Weapons"]}
    alice.mechanics["dnd5e_runtime"] = {
        "inventory": {
            "items": [
                {
                    "id": "loot_evt_club_iron_capped_club",
                    "item_id": "loot_evt_club_iron_capped_club",
                    "source_item_id": "iron_capped_club",
                    "source_offer_id": "loot_evt_club",
                    "name": "Iron-capped club",
                    "kind": "weapon",
                    "quantity": 1,
                    "equipped": False,
                    "attuned": False,
                    "identified": True,
                }
            ],
            "currency": {},
        }
    }
    values = iter([14, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )

    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                _planned_attack(
                    action_id="runtime_item_attack_iron_capped_club",
                    reason="Alice attacks Bob with the iron-capped club.",
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            mechanical_summary="Alice hits Bob with the club.",
            visible_outcome_facts=["Alice hits Bob with the club."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I swing the iron-capped club.",
        )
    )

    first_packet = prompt_mgr.render_messages.call_args_list[0].kwargs[
        "combat_action_packet"
    ]
    assert "Iron-capped club" in first_packet
    assert "runtime_item_attack_iron_capped_club" in first_packet
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.rolls[0].modifier == 5
    assert transaction.rolls[0].label == "Attack (Iron-capped club)"
    assert transaction.damage_records[0].damage_type == "bludgeoning"


def test_combat_packet_exposes_defenses_and_effect_break_triggers():
    ckpt = _ckpt()
    ckpt.characters[1] = _character(
        "bob",
        "Bob",
        defenses={"damage_resistances": ["nonmagical slashing"]},
    )
    alice = ckpt.session.active_combat.combatants[0]
    alice.active_effects.append(DndRuntimeEffect(
        effect_id="eff_invisible",
        name="Invisibility",
        slug="invisibility",
        target_id="alice",
        originator_id="alice",
        conditions=["invisible"],
        break_triggers=["attack", "cast_spell"],
    ))
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=False,
            roll_requests=[],
            no_roll_reason="probe",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="No effect.",
            visible_outcome_facts=["Alice waits."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I wait.",
        )
    )

    packet = json.loads(
        prompt_mgr.render_messages.call_args_list[0].kwargs[
            "combat_action_packet"
        ]
    )
    by_id = {
        combatant["character_id"]: combatant
        for combatant in packet["combatants"]
    }
    assert by_id["bob"]["mechanics"]["defenses"]["damage_resistances"] == [
        "nonmagical slashing"
    ]
    assert by_id["alice"]["active_effects"][0]["break_triggers"] == [
        "attack",
        "cast_spell",
    ]


def test_invalid_action_id_does_not_fall_back_to_reason_weapon(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "longsword",
                "name": "Longsword",
                "attack": {"bonus": 6, "damage": "1d8+3 slashing"},
            },
            {
                "id": "shortbow",
                "name": "Shortbow",
                "attack": {"bonus": 7, "damage": "1d6+4 piercing"},
            },
        ],
    )
    values = iter([14])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        action_id="blade",
        reason="Alice attacks Bob with a shortbow.",
    ))

    with pytest.raises(ValueError, match="source is not available"):
        asyncio.run(
            DndCombatResolver(client, prompt_mgr).resolve_combat_action(
                ckpt=ckpt,
                actor_id="alice",
                intention="I shoot Bob with my shortbow.",
            )
        )
    assert ckpt.session.cat_ii_roll_transactions == []
    assert ckpt.session.active_combat.combatants[1].hit_points_current == 13


def test_missing_action_source_fails_before_dice(
    monkeypatch,
):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "longsword",
                "name": "Longsword",
                "attack": {"bonus": 6, "damage": "1d8+3 slashing"},
            },
            {
                "id": "shortbow",
                "name": "Shortbow",
                "attack": {"bonus": 7, "damage": "1d6+4 piercing"},
            },
        ],
    )
    values = iter([14])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        action_id="",
        reason="Alice attacks Bob with a weapon.",
    ))

    with pytest.raises(ValueError, match="source is not available"):
        asyncio.run(
            DndCombatResolver(client, prompt_mgr).resolve_combat_action(
                ckpt=ckpt,
                actor_id="alice",
                intention="I attack Bob with a weapon.",
            )
        )
    assert ckpt.session.cat_ii_roll_transactions == []
    assert ckpt.session.active_combat.combatants[1].hit_points_current == 13


def test_known_no_damage_attack_action_does_not_emit_damage_marker(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        actions=[
            {
                "id": "disarm",
                "name": "Disarm",
                "attack": {"bonus": 5, "damage": ""},
                "notes": "DMG optional Disarm; deals no damage.",
            }
        ],
    )
    values = iter([19])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack(
        action_id="disarm",
        reason="Alice tries to disarm Bob.",
    ))

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I disarm Bob.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.damage_records == []
    assert not any(line.startswith("damage_for=") for line in transaction.ledger_lines)
    assert any(
        "disarm has no damage expression; no damage is rolled" in line
        for line in transaction.ledger_lines
    )


def test_combat_packet_exposes_current_actor_spellcasting():
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        spellcasting={
            "profiles": [
                {
                    "id": "class_1",
                    "name": "Wizard",
                    "ability": "int",
                    "spell_attack_bonus": 5,
                    "spell_save_dc": 13,
                }
            ],
            "slots": {"2": {"current": 1, "max": 2}},
            "spells": [
                {
                    "id": "hold_person",
                    "name": "Hold Person",
                    "level": 2,
                    "prepared": True,
                    "always_prepared": False,
                    "concentration": True,
                    "duration": {
                        "kind": "minutes",
                        "amount": 1,
                        "text": "Concentration, up to 1 minute",
                    },
                    "save": {"ability": "wis", "dc": 13},
                    "damage": [],
                    "healing": [],
                    "consumes": [{"resource_id": "spell_slot_2", "amount": 1}],
                },
                {
                    "id": "cone_of_cold",
                    "name": "Cone of Cold",
                    "level": 5,
                    "prepared": True,
                    "always_prepared": False,
                    "target": {"text": "60-foot cone"},
                    "save": {"ability": "con", "dc": 13},
                    "damage": [{"formula": "8d8 cold"}],
                    "healing": [],
                    "consumes": [{"resource_id": "spell_slot_5", "amount": 1}],
                }
            ],
        },
    )
    ckpt.session.active_combat.battle_map = DndBattleMapState(
        present=True,
        map_name="Hall",
        width=8,
        height=5,
        tokens=[
            DndBattleMapToken(
                token_id="alice",
                character_id="alice",
                label="Alice",
                x=1,
                y=2,
            ),
            DndBattleMapToken(
                token_id="bob",
                character_id="bob",
                label="Bob",
                x=3,
                y=2,
            ),
        ],
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=False,
            roll_requests=[],
            no_roll_reason="probe",
        )),
        _llm_response(RulesAdjudication(
            feasible=False,
            combat_status="ongoing",
            mechanical_summary="No effect.",
            visible_outcome_facts=["Alice cannot complete the spell."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Hold Person on Bob.",
        )
    )

    first_packet = json.loads(
        prompt_mgr.render_messages.call_args_list[0].kwargs[
            "combat_action_packet"
        ]
    )
    alice = next(
        combatant for combatant in first_packet["combatants"]
        if combatant["character_id"] == "alice"
    )
    spell_names = {spell["name"] for spell in alice["spells"]}
    assert "Hold Person" in spell_names
    assert alice["spellcasting"]["profiles"][0]["spell_save_dc"] == 13
    assert alice["spellcasting"]["slots"] == {"2": {"current": 1, "max": 2}}
    hold_person = next(
        spell for spell in alice["spells"]
        if spell["id"] == "hold_person"
    )
    assert hold_person["duration"]["text"] == "Concentration, up to 1 minute"
    assert first_packet["tactical_map"]["area_targeting"][0]["action_id"] == (
        "cone_of_cold"
    )
    standard_ids = {
        action["id"] for action in first_packet["standard_combat_actions"]
    }
    assert {"dash", "shove", "grapple"}.issubset(standard_ids)


def test_concentration_only_self_effect_does_not_publish_condition_fact():
    ckpt = _ckpt()
    alice = ckpt.session.active_combat.combatants[0]

    effect = DndRuntimeEffect(
        effect_id="web_concentration",
        name="Web",
        slug="web",
        source_type="spell",
        source_id="web",
        originator_id="alice",
        target_id="alice",
        conditions=["concentrating"],
        concentration=True,
        duration_kind="hours",
        duration_amount=1,
        remaining_rounds=10,
    )
    dnd_combat.start_effect(ckpt.session, effect, replace_concentration=False)

    assert alice.active_effects[0].effect_id == "web_concentration"
    assert alice.active_effects[0].conditions == []
    assert "concentrating" not in alice.conditions
    assert dnd_combat.drain_pending_visible_facts(ckpt.session.active_combat) == []

    dnd_combat.start_effect(
        ckpt.session,
        DndRuntimeEffect(
            effect_id="web_restrained",
            name="Web",
            slug="web",
            source_type="spell",
            source_id="web",
            originator_id="alice",
            target_id="bob",
            conditions=["restrained"],
            concentration=True,
            duration_kind="hours",
            duration_amount=1,
            remaining_rounds=10,
            metadata={"reason": "failed initial save."},
        ),
        replace_concentration=False,
    )
    assert dnd_combat.drain_pending_visible_facts(ckpt.session.active_combat) == [
        "Web takes hold on Bob after the initial save fails."
    ]


def test_combat_resolver_consumes_spell_slot_once_for_multi_target_save_spell(
    monkeypatch,
):
    ckpt = _ckpt()
    ckpt.characters.append(_character("charlie", "Charlie"))
    ckpt.session.active_combat.combatants.append(
        DndCombatantState(
            combatant_id="charlie",
            character_id="charlie",
            name="Charlie",
            armor_class=12,
            hit_points_current=13,
            hit_points_max=13,
        )
    )
    ckpt.characters[0].mechanics["resources"] = [
        {
            "id": "spell_slot_1",
            "name": "Level 1 Spell Slot",
            "current": 1,
            "max": 1,
        }
    ]
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{
            "id": "class_1",
            "name": "Wizard",
            "ability": "int",
            "spell_save_dc": 13,
        }],
        "slots": {"1": {"current": 1, "max": 1}},
        "spells": [
            _spell(
                "burning_hands",
                "Burning Hands",
                level=1,
                save_ability="dex",
                dc=13,
                damage="3d6 fire",
                consumes_level=1,
            )
        ],
    }
    monkeypatch.setattr(dice.d20.expression.random, "randrange", lambda _: 4)
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_spell_turn_plan(
            source_id="burning_hands",
            cast_level=1,
            resource_id="spell_slot_1",
            roll_requests=[
                PlannedRoll(
                    roll_id="save_bob_fire",
                    actor_id="alice",
                    kind="saving_throw",
                    ability="dex",
                    skill="",
                    dc=13,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Bob resists Burning Hands.",
                    action_id="burning_hands",
                    target_id="bob",
                    effect_id="",
                    damage_on_save_success="half",
                ),
                PlannedRoll(
                    roll_id="save_charlie_fire",
                    actor_id="alice",
                    kind="saving_throw",
                    ability="dex",
                    skill="",
                    dc=13,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Charlie resists Burning Hands.",
                    action_id="burning_hands",
                    target_id="charlie",
                    effect_id="",
                    damage_on_save_success="half",
                ),
            ],
            reason="Alice casts Burning Hands.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice's fire catches Bob and Charlie.",
            visible_outcome_facts=["Alice's fire washes over Bob and Charlie."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Burning Hands at Bob and Charlie.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert [
        (spend.resource_id, spend.source_id, spend.amount, spend.applied)
        for spend in transaction.resource_spends
    ] == [("spell_slot_1", "burning_hands", 1, True)]
    resources = ckpt.characters[0].mechanics["resources"]
    assert resources[0]["current"] == 0
    slots = ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"][
        "spellcasting"
    ]["slots"]
    assert slots["1"]["current"] == 0


def test_combat_resolver_consumes_no_roll_spell_named_in_intention():
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{
            "id": "class_1",
            "name": "Wizard",
            "ability": "int",
            "spell_attack_bonus": 5,
        }],
        "slots": {"2": {"current": 2, "max": 3}},
        "spells": [
            _spell(
                "misty_step",
                "Misty Step",
                level=2,
                consumes_level=2,
            )
        ],
    }
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_spell_turn_plan(
            source_id="misty_step",
            cast_level=2,
            resource_id="spell_slot_2",
            economy="bonus_action",
            reason="Misty Step needs no roll.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice casts Misty Step and teleports.",
            visible_outcome_facts=["Alice vanishes and reappears nearby."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Misty Step to cross the room.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert [
        (spend.resource_id, spend.source_id, spend.amount, spend.applied)
        for spend in transaction.resource_spends
    ] == [("spell_slot_2", "misty_step", 1, True)]
    slots = ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"][
        "spellcasting"
    ]["slots"]
    assert slots["2"]["current"] == 1


def test_combat_resolver_spell_attack_spends_slot_and_uses_spell_damage(
    monkeypatch,
):
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{
            "id": "class_1",
            "name": "Wizard",
            "ability": "int",
            "spell_attack_bonus": 6,
        }],
        "slots": {"2": {"current": 1, "max": 1}},
        "spells": [
            _spell(
                "scorching_ray",
                "Scorching Ray",
                level=2,
                attack_bonus=6,
                damage="2d6 fire",
                consumes_level=2,
            )
        ],
    }
    values = iter([9, 2, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_spell_turn_plan(
            source_id="scorching_ray",
            cast_level=2,
            resource_id="spell_slot_2",
            roll_requests=[
                PlannedRoll(
                    roll_id="ray_1",
                    actor_id="alice",
                    kind="attack_roll",
                    ability="int",
                    skill="",
                    dc=12,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Alice hurls a ray at Bob.",
                    action_id="scorching_ray",
                    target_id="bob",
                )
            ],
            reason="Alice casts Scorching Ray.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="The ray hits.",
            visible_outcome_facts=["Alice's ray scorches Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Scorching Ray at Bob.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.rolls[0].modifier == 6
    assert transaction.damage_records[0].damage_type == "fire"
    assert transaction.resource_spends[0].source_id == "scorching_ray"
    assert ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"][
        "spellcasting"
    ]["slots"]["2"]["current"] == 0


def test_combat_resolver_magic_missile_spends_one_slot_for_many_damage_rolls(
    monkeypatch,
):
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{"id": "class_1", "name": "Wizard", "ability": "int"}],
        "slots": {"1": {"current": 1, "max": 1}},
        "spells": [
            _spell(
                "magic_missile",
                "Magic Missile",
                level=1,
                damage="3 darts, each 1d4+1 force",
                consumes_level=1,
            )
        ],
    }
    values = iter([0, 1, 2])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_spell_turn_plan(
            source_id="magic_missile",
            cast_level=1,
            resource_id="spell_slot_1",
            roll_requests=[
                PlannedRoll(
                    roll_id=f"dart_{index}",
                    actor_id="alice",
                    kind="damage_roll",
                    ability="str",
                    skill="",
                    dc=0,
                    opposed_by="",
                    advantage_state="normal",
                    reason=f"Magic Missile dart {index} hits Bob.",
                    action_id="magic_missile",
                    target_id="bob",
                )
                for index in range(1, 4)
            ],
            reason="Alice casts Magic Missile.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="The darts hit.",
            visible_outcome_facts=["Alice's missiles strike Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Magic Missile at Bob.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert [damage.amount for damage in transaction.damage_records] == [2, 3, 4]
    assert [
        (spend.resource_id, spend.source_id, spend.amount, spend.applied)
        for spend in transaction.resource_spends
    ] == [("spell_slot_1", "magic_missile", 1, True)]
    assert ckpt.session.active_combat.combatants[1].hit_points_current == 4


def test_combat_resolver_cantrip_attack_spends_no_slot(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{
            "id": "class_1",
            "name": "Wizard",
            "ability": "int",
            "spell_attack_bonus": 5,
        }],
        "slots": {"1": {"current": 1, "max": 1}},
        "spells": [
            _spell(
                "fire_bolt",
                "Fire Bolt",
                level=0,
                attack_bonus=5,
                damage="1d10 fire",
            )
        ],
    }
    values = iter([9, 4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(DndCombatTurnPlan(
            feasible=True,
            actions=[
                DndCombatActionUse(
                    actor_id="alice",
                    source_type="spell",
                    source_id="fire_bolt",
                    use_mode="cast",
                    economy="action",
                    resource_spends=[],
                    rolls=[
                        DndPlannedActionRoll(
                            roll_id="fire_bolt_attack",
                            kind="attack_roll",
                            roller_id="alice",
                            target_id="bob",
                            ability="int",
                            skill="",
                            dc=12,
                            opposed_by="",
                            advantage_state="normal",
                            reason="Alice casts Fire Bolt at Bob.",
                        )
                    ],
                    reason="cantrip attack",
                )
            ],
            no_action_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="The cantrip hits.",
            visible_outcome_facts=["Alice's fire bolt strikes Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Fire Bolt at Bob.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.resource_spends == []
    assert transaction.damage_records[0].damage_type == "fire"
    assert ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"][
        "spellcasting"
    ]["slots"]["1"]["current"] == 1


def test_combat_resolver_ritual_spell_spends_no_slot():
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{"id": "class_1", "name": "Wizard", "ability": "int"}],
        "slots": {"1": {"current": 1, "max": 1}},
        "spells": [
            _spell("alarm", "Alarm", level=1, consumes_level=1)
        ],
    }
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(DndCombatTurnPlan(
            feasible=True,
            actions=[
                DndCombatActionUse(
                    actor_id="alice",
                    source_type="spell",
                    source_id="alarm",
                    use_mode="cast",
                    economy="action",
                    casting=DndCombatCasting(cast_level=1, ritual=True),
                    resource_spends=[],
                    rolls=[],
                    reason="Alice casts Alarm as a ritual.",
                )
            ],
            no_action_reason="Alarm needs no roll.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice sets the ward.",
            visible_outcome_facts=["Alice completes a quiet warding rite."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Alarm as a ritual.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.resource_spends == []
    assert ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"][
        "spellcasting"
    ]["slots"]["1"]["current"] == 1


def test_combat_resolver_pact_slot_spell_spends_pact_slot():
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{"id": "pact", "name": "Warlock", "ability": "cha"}],
        "pact_slots": {"current": 1, "max": 1, "level": 2},
        "spells": [{
            **_spell("hex", "Hex", level=1, concentration=True),
            "consumes": [{"resource_id": "pact_slot", "amount": 1}],
        }],
    }
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_spell_turn_plan(
            source_id="hex",
            cast_level=0,
            resource_id="pact_slot",
            reason="Alice casts Hex.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice curses Bob.",
            visible_outcome_facts=["Alice marks Bob with a curse."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Hex on Bob.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert [
        (spend.resource_id, spend.source_id, spend.applied)
        for spend in transaction.resource_spends
    ] == [("pact_slot", "hex", True)]
    assert ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"][
        "spellcasting"
    ]["pact_slots"]["current"] == 0


def test_combat_resolver_rejects_invalid_or_missing_resource_spends():
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{"id": "class_1", "name": "Wizard", "ability": "int"}],
        "slots": {"2": {"current": 0, "max": 1}},
        "spells": [
            _spell("misty_step", "Misty Step", level=2, consumes_level=2)
        ],
    }
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.return_value = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "plan"},
    ]
    client = MagicMock()
    client.complete = AsyncMock(return_value=_llm_response(_spell_turn_plan(
        source_id="misty_step",
        cast_level=2,
        resource_id="spell_slot_2",
        economy="bonus_action",
        reason="Alice casts Misty Step.",
    )))

    with pytest.raises(ValueError, match="exceeds the available resource"):
        asyncio.run(
            DndCombatResolver(client, prompt_mgr).resolve_combat_action(
                ckpt=ckpt,
                actor_id="alice",
                intention="I cast Misty Step.",
            )
        )

    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"][
        "slots"
    ]["2"]["current"] = 1
    client.complete = AsyncMock(return_value=_llm_response(DndCombatTurnPlan(
        feasible=True,
        actions=[
            DndCombatActionUse(
                actor_id="alice",
                source_type="spell",
                source_id="misty_step",
                use_mode="cast",
                economy="bonus_action",
                casting=DndCombatCasting(cast_level=2),
                resource_spends=[],
                rolls=[],
                reason="Alice casts Misty Step.",
            )
        ],
        no_action_reason="",
    )))

    with pytest.raises(ValueError, match="must declare resource_spends"):
        asyncio.run(
            DndCombatResolver(client, prompt_mgr).resolve_combat_action(
                ckpt=ckpt,
                actor_id="alice",
                intention="I cast Misty Step.",
            )
        )


def test_combat_resolver_rejects_upcast_slot_mismatch():
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{"id": "class_1", "name": "Wizard", "ability": "int"}],
        "slots": {"3": {"current": 1, "max": 1}},
        "spells": [
            _spell("burning_hands", "Burning Hands", level=1, consumes_level=1)
        ],
    }
    client = MagicMock()
    client.complete = AsyncMock(return_value=_llm_response(_spell_turn_plan(
        source_id="burning_hands",
        cast_level=2,
        resource_id="spell_slot_3",
        reason="Alice upcasts Burning Hands.",
    )))
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.return_value = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "plan"},
    ]

    with pytest.raises(ValueError, match="must match casting.cast_level"):
        asyncio.run(
            DndCombatResolver(client, prompt_mgr).resolve_combat_action(
                ckpt=ckpt,
                actor_id="alice",
                intention="I cast Burning Hands with a third-level slot.",
            )
        )


def test_combat_resolver_ongoing_effect_rolls_without_original_slot(
    monkeypatch,
):
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{
            "id": "class_1",
            "name": "Druid",
            "ability": "wis",
            "spell_save_dc": 13,
        }],
        "slots": {"3": {"current": 0, "max": 1}},
        "spells": [
            _spell(
                "call_lightning",
                "Call Lightning",
                level=3,
                save_ability="dex",
                dc=13,
                damage="3d10 lightning",
                consumes_level=3,
                concentration=True,
            )
        ],
    }
    alice = ckpt.session.active_combat.combatants[0]
    alice.active_effects.append(DndRuntimeEffect(
        effect_id="eff_call_lightning",
        name="Call Lightning",
        slug="call_lightning",
        source_type="spell",
        source_id="call_lightning",
        originator_id="alice",
        target_id="alice",
        concentration=True,
        duration_kind="minutes",
        duration_amount=10,
        remaining_rounds=10,
    ))
    values = iter([4, 0, 1, 2])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(DndCombatTurnPlan(
            feasible=True,
            actions=[
                DndCombatActionUse(
                    actor_id="alice",
                    source_type="effect",
                    source_id="call_lightning",
                    effect_id="eff_call_lightning",
                    use_mode="sustain",
                    economy="action",
                    resource_spends=[],
                    rolls=[
                        DndPlannedActionRoll(
                            roll_id="call_lightning_bob",
                            kind="saving_throw",
                            roller_id="bob",
                            target_id="bob",
                            ability="dex",
                            skill="",
                            dc=13,
                            opposed_by="",
                            advantage_state="normal",
                            damage_on_save_success="half",
                            reason="Bob dodges the called lightning.",
                        )
                    ],
                    reason="Alice calls lightning down again.",
                )
            ],
            no_action_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="The storm bolt lands.",
            visible_outcome_facts=["Lightning crashes into Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I call lightning down on Bob again.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.resource_spends == []
    assert transaction.damage_records[0].damage_type == "lightning"
    assert ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"][
        "spellcasting"
    ]["slots"]["3"]["current"] == 0


def test_combat_resolver_battle_map_area_effect_rolls_from_source_spell(
    monkeypatch,
):
    ckpt = _ckpt()
    ckpt.session.active_combat.turn_index = 1
    ckpt.session.active_combat.battle_map = DndBattleMapState(
        present=True,
        map_name="Toxic chamber",
        width=10,
        height=10,
        square_size_ft=5,
        tokens=[
            DndBattleMapToken(
                token_id="alice",
                character_id="alice",
                label="Alice",
                x=1,
                y=1,
                size_squares=1,
            ),
            DndBattleMapToken(
                token_id="bob",
                character_id="bob",
                label="Bob",
                x=5,
                y=5,
                size_squares=1,
            ),
        ],
        terrain=[],
        areas=[
            DndAreaTemplate(
                template_id="cloudkill_area",
                label="Cloudkill",
                shape="circle",
                x=5,
                y=5,
                radius_squares=4,
                width=1,
                height=1,
                duration_rounds=10,
                notes=(
                    "Source alice; action_id cloudkill_area; creatures in "
                    "the cloud make a Constitution save at turn start."
                ),
            )
        ],
        notes="",
    )
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{
            "id": "class_1",
            "name": "Wizard",
            "ability": "int",
            "spell_save_dc": 13,
        }],
        "slots": {"5": {"current": 0, "max": 1}},
        "spells": [
            _spell(
                "cloudkill_area",
                "Cloudkill",
                level=5,
                save_ability="con",
                dc=13,
                damage="5d8 poison",
                concentration=True,
            )
        ],
    }
    values = iter([0, 0, 0, 0, 0, 0])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(DndCombatTurnPlan(
            feasible=True,
            actions=[
                DndCombatActionUse(
                    actor_id="bob",
                    source_type="effect",
                    source_id="cloudkill_area",
                    effect_id="cloudkill_area",
                    use_mode="sustain",
                    economy="none",
                    resource_spends=[],
                    rolls=[
                        DndPlannedActionRoll(
                            roll_id="cloudkill_bob",
                            kind="saving_throw",
                            roller_id="bob",
                            target_id="bob",
                            ability="con",
                            skill="",
                            dc=13,
                            opposed_by="",
                            advantage_state="normal",
                            damage_on_save_success="none",
                            reason="Bob resists the cloudkill area.",
                        )
                    ],
                    reason="Resolve Cloudkill at the start of Bob's turn.",
                )
            ],
            no_action_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="The poisonous cloud bites.",
            visible_outcome_facts=["The green vapor burns Bob's lungs."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            spatial_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="bob",
            intention="I run out of the cloudkill.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.resource_spends == []
    assert transaction.damage_records[0].target_id == "bob"
    assert transaction.damage_records[0].damage_type == "poison"
    assert ckpt.session.active_combat.combatants[1].hit_points_current == 8


def test_combat_resolver_rejects_area_effect_without_effect_id():
    ckpt = _ckpt()
    ckpt.session.active_combat.turn_index = 1
    ckpt.session.active_combat.battle_map = DndBattleMapState(
        present=True,
        map_name="Toxic chamber",
        width=10,
        height=10,
        square_size_ft=5,
        tokens=[
            DndBattleMapToken(
                token_id="bob",
                character_id="bob",
                label="Bob",
                x=5,
                y=5,
                size_squares=1,
            ),
        ],
        terrain=[],
        areas=[
            DndAreaTemplate(
                template_id="cloudkill_area",
                label="Cloudkill",
                shape="circle",
                x=5,
                y=5,
                radius_squares=4,
                width=1,
                height=1,
                duration_rounds=10,
                notes="Creatures in the cloud save at turn start.",
            )
        ],
        notes="",
    )
    client = MagicMock()
    client.complete = AsyncMock(return_value=_llm_response(DndCombatTurnPlan(
        feasible=True,
        actions=[
            DndCombatActionUse(
                actor_id="bob",
                source_type="effect",
                source_id="cloudkill_area",
                effect_id="",
                use_mode="sustain",
                economy="none",
                resource_spends=[],
                rolls=[
                    DndPlannedActionRoll(
                        roll_id="cloudkill_bob",
                        kind="saving_throw",
                        roller_id="bob",
                        target_id="bob",
                        ability="con",
                        skill="",
                        dc=13,
                        opposed_by="",
                        advantage_state="normal",
                        damage_on_save_success="none",
                        reason="Bob resists the cloudkill area.",
                    )
                ],
                reason="Resolve Cloudkill at the start of Bob's turn.",
            )
        ],
        no_action_reason="",
    )))
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.return_value = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "plan"},
    ]

    with pytest.raises(ValueError, match="must declare effect_id"):
        asyncio.run(
            DndCombatResolver(client, prompt_mgr).resolve_combat_action(
                ckpt=ckpt,
                actor_id="bob",
                intention="I run out of the cloudkill.",
            )
        )


def test_combat_resolver_consumes_readied_no_roll_spell_once():
    ckpt = _ckpt()
    ckpt.characters[0].mechanics["resources"] = {
        "spell_slot_1": {"current": 3, "max": 4}
    }
    ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"]["spellcasting"] = {
        "profiles": [{
            "id": "class_1",
            "name": "Wizard",
            "ability": "int",
            "spell_attack_bonus": 5,
        }],
        "slots": {"1": {"current": 3, "max": 4}},
        "spells": [
            _spell(
                "magic_missile",
                "Magic Missile",
                level=1,
                damage="3 darts, each 1d4+1 force",
                consumes_level=1,
            )
        ],
    }
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_spell_turn_plan(
            source_id="magic_missile",
            cast_level=1,
            resource_id="spell_slot_1",
            reason="Readying the spell needs no roll.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice readies Magic Missile.",
            visible_outcome_facts=["Alice readies a spell for Bob's move."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[{
                "operation": "start",
                "target_id": "alice",
                "effect_id": "ready_magic_missile",
                "name": "Readied Magic Missile",
                "slug": "readied_spell",
                "source_type": "spell",
                "source_id": "magic_missile",
                "originator_id": "alice",
                "conditions": ["concentrating"],
                "concentration": True,
                "duration_kind": "until_removed",
                "duration_amount": 0,
                "remaining_rounds": 0,
                "duration_text": "",
                "break_triggers": [],
                "reason": "spell readied",
            }],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I ready Magic Missile for Bob's move.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert [
        (spend.resource_id, spend.source_id, spend.amount, spend.applied)
        for spend in transaction.resource_spends
    ] == [("spell_slot_1", "magic_missile", 1, True)]
    assert ckpt.characters[0].mechanics["resources"]["spell_slot_1"]["current"] == 2
    slots = ckpt.characters[0].mechanics["dnd5e_sheet"]["statblock"][
        "spellcasting"
    ]["slots"]
    assert slots["1"]["current"] == 2
    cat._apply_combat_resource_spends(ckpt, transaction)
    assert ckpt.characters[0].mechanics["resources"]["spell_slot_1"]["current"] == 2
    facts = [fact.text for fact in routed.canonical_event.observable_facts]
    assert not any("Magic Missile takes hold on Alice" in fact for fact in facts)
    readied_effect = ckpt.session.active_combat.combatants[0].active_effects[0]
    assert readied_effect.conditions == []
    assert readied_effect.duration_kind == "rounds"
    assert readied_effect.duration_amount == 1
    assert readied_effect.remaining_rounds == 1
    assert "start of the readying actor's next turn" in readied_effect.duration_text
    assert readied_effect.break_triggers == [
        "concentration ends",
        "start of readying actor's next turn",
        "trigger occurs",
    ]
    assert readied_effect.metadata["readied_action"]["source_id"] == "magic_missile"
    assert readied_effect.metadata["readied_action"]["readying_actor_id"] == "alice"
    assert "Bob's move" in readied_effect.metadata["readied_action"]["trigger_text"]


def test_combat_resolver_releases_readied_damage_rolls(monkeypatch):
    ckpt = _ckpt()
    _give_alice_readied_magic_missile(ckpt)
    ckpt.session.active_combat.turn_index = 1
    values = iter([0, 0, 0])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id=f"magic_missile_{index}",
                    actor_id="alice",
                    kind="damage_roll",
                    ability="str",
                    skill="",
                    dc=0,
                    opposed_by="",
                    advantage_state="normal",
                    reason=f"Magic Missile dart {index} hits Bob.",
                    action_id="magic_missile",
                    target_id="bob",
                    effect_id="ready_magic_missile",
                )
                for index in range(1, 4)
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice releases the held spell as Bob moves.",
            visible_outcome_facts=[
                "Alice's held missiles slam into Bob as he opens the door."
            ],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[{
                "operation": "end",
                "target_id": "alice",
                "effect_id": "ready_magic_missile",
                "slug": "readied_spell",
                "reason": "trigger occurred",
            }],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="bob",
            intention="I open the door.",
        )
    )

    alice = ckpt.session.active_combat.combatants[0]
    bob = ckpt.session.active_combat.combatants[1]
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert bob.hit_points_current == 7
    assert alice.active_effects == []
    assert alice.reaction_available is False
    assert transaction.resource_spends == []
    assert [damage.amount for damage in transaction.damage_records] == [2, 2, 2]
    assert all(damage.applied for damage in transaction.damage_records)
    assert any("readied_release=ready_magic_missile" in line for line in transaction.ledger_lines)
    facts = [fact.text for fact in routed.canonical_event.observable_facts]
    assert facts == ["Alice's held missiles slam into Bob as he opens the door."]


def test_readied_release_requires_explicit_effect_end(monkeypatch):
    ckpt = _ckpt()
    _give_alice_readied_magic_missile(ckpt)
    ckpt.session.active_combat.turn_index = 1
    values = iter([0])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="magic_missile_1",
                    actor_id="alice",
                    kind="damage_roll",
                    ability="str",
                    skill="",
                    dc=0,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Magic Missile dart hits Bob.",
                    action_id="magic_missile",
                    target_id="bob",
                    effect_id="ready_magic_missile",
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice releases the held spell.",
            visible_outcome_facts=["Alice's missile strikes Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    with pytest.raises(ValueError, match="must end the held effect"):
        asyncio.run(
            DndCombatResolver(client, prompt_mgr).resolve_combat_action(
                ckpt=ckpt,
                actor_id="bob",
                intention="I open the door.",
            )
        )

    bob = ckpt.session.active_combat.combatants[1]
    assert bob.hit_points_current == 13
    assert ckpt.session.active_combat.combatants[0].reaction_available is True


def test_combat_resolver_starts_sustained_effect_from_adjudication(monkeypatch):
    ckpt = _ckpt()
    values = iter([4])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="save_bob_hold",
                    actor_id="bob",
                    kind="saving_throw",
                    ability="wis",
                    skill="",
                    dc=13,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Bob resists Hold Person.",
                    action_id="",
                    target_id="bob",
                    effect_id="",
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Bob fails the initial save.",
            visible_outcome_facts=["Bob locks in place under Alice's spell."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[
                {
                    "operation": "start",
                    "target_id": "bob",
                    "effect_id": "eff_hold",
                    "name": "Hold Person",
                    "slug": "hold_person",
                    "source_type": "spell",
                    "source_id": "hold_person",
                    "originator_id": "alice",
                    "conditions": ["paralyzed"],
                    "concentration": True,
                    "duration_kind": "minutes",
                    "duration_amount": 1,
                    "remaining_rounds": 10,
                    "duration_text": "Concentration, up to 1 minute",
                    "recurring_save": {
                        "ability": "wis",
                        "dc": 13,
                        "timing": "end_of_turn",
                        "ends_on": "success",
                        "repeat": True,
                    },
                    "reason": "failed initial save",
                }
            ],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Hold Person on Bob.",
        )
    )

    bob = ckpt.session.active_combat.combatants[1]
    assert bob.active_effects[0].effect_id == "eff_hold"
    assert bob.active_effects[0].recurring_save.dc == 13
    assert "paralyzed" in bob.conditions
    stored = ckpt.characters[1].mechanics["dnd5e_runtime"]["active_effects"]
    assert stored[0]["effect_id"] == "eff_hold"


def test_combat_resolver_batches_same_spell_concentration_effects():
    ckpt = _ckpt()
    ckpt.characters.append(_character("charlie", "Charlie"))
    ckpt.session.active_combat.combatants.append(
        DndCombatantState(
            combatant_id="charlie",
            character_id="charlie",
            name="Charlie",
            armor_class=12,
            hit_points_current=13,
            hit_points_max=13,
        )
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=False,
            roll_requests=[],
            no_roll_reason="No roll.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice starts one concentration spell.",
            visible_outcome_facts=["Alice's pattern catches Bob and Charlie."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[
                {
                    "operation": "start",
                    "target_id": "bob",
                    "effect_id": "hypnotic_pattern:alice:bob",
                    "name": "Hypnotic Pattern - Bob",
                    "slug": "hypnotic_pattern",
                    "source_type": "spell",
                    "source_id": "hypnotic_pattern",
                    "originator_id": "alice",
                    "conditions": ["charmed", "incapacitated"],
                    "concentration": True,
                    "duration_kind": "rounds",
                    "duration_amount": 10,
                    "remaining_rounds": 10,
                    "reason": "failed initial save",
                },
                {
                    "operation": "start",
                    "target_id": "charlie",
                    "effect_id": "hypnotic_pattern:alice:charlie",
                    "name": "Hypnotic Pattern - Charlie",
                    "slug": "hypnotic_pattern",
                    "source_type": "spell",
                    "source_id": "hypnotic_pattern",
                    "originator_id": "alice",
                    "conditions": ["charmed", "incapacitated"],
                    "concentration": True,
                    "duration_kind": "rounds",
                    "duration_amount": 10,
                    "remaining_rounds": 10,
                    "reason": "failed initial save",
                },
            ],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Hypnotic Pattern.",
        )
    )

    bob = next(c for c in ckpt.session.active_combat.combatants if c.character_id == "bob")
    charlie = next(
        c for c in ckpt.session.active_combat.combatants
        if c.character_id == "charlie"
    )
    assert [effect.effect_id for effect in bob.active_effects] == [
        "hypnotic_pattern:alice:bob"
    ]
    assert [effect.effect_id for effect in charlie.active_effects] == [
        "hypnotic_pattern:alice:charlie"
    ]
    facts = [fact.text for fact in routed.canonical_event.observable_facts]
    assert not any("concentration shifts to a new effect" in fact for fact in facts)


def test_combat_resolver_notes_skipped_effect_delta_for_missing_target():
    ckpt = _ckpt()
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=False,
            roll_requests=[],
            no_roll_reason="No roll needed.",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice's spell tries to bind Bob.",
            visible_outcome_facts=["The spell fails to catch."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[
                {
                    "operation": "start",
                    "target_id": "bobb",
                    "effect_id": "eff_hold",
                    "name": "Hold Person",
                    "slug": "hold_person",
                    "originator_id": "alice",
                    "conditions": ["paralyzed"],
                    "reason": "failed initial save",
                }
            ],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I cast Hold Person on Bob.",
        )
    )

    combatants = ckpt.session.active_combat.combatants
    assert all(not combatant.active_effects for combatant in combatants)
    assert "Effect start skipped for Hold Person" in routed.decision_rationale
    assert "target_id='bobb'" in routed.decision_rationale


def test_finalized_combat_transaction_cannot_be_continued(monkeypatch):
    ckpt = _ckpt()
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client, prompt_mgr = _basic_attack_mocks(_planned_attack())
    resolver = DndCombatResolver(client, prompt_mgr)

    asyncio.run(
        resolver.resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob with my blade.",
        )
    )
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.status == "finalized"

    with pytest.raises(ValueError, match="finalized"):
        asyncio.run(
            resolver.continue_combat_transaction(
                ckpt=ckpt,
                event_id=transaction.event_id,
            )
        )
    assert client.complete.await_count == 2


def test_combat_resolver_explicit_effect_delta_breaks_existing_effect(monkeypatch):
    ckpt = _ckpt()
    alice = ckpt.session.active_combat.combatants[0]
    alice.active_effects.append(DndRuntimeEffect(
        effect_id="eff_invisible",
        name="Invisibility",
        slug="invisibility",
        target_id="alice",
        originator_id="alice",
        conditions=["invisible"],
        concentration=True,
        duration_kind="minutes",
        duration_amount=1,
        remaining_rounds=10,
    ))
    alice.conditions.append("invisible")
    values = iter([9, 3])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="attack_alice",
                    actor_id="alice",
                    kind="attack_roll",
                    ability="str",
                    skill="",
                    dc=12,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Alice attacks Bob with a blade.",
                    action_id="blade",
                    target_id="bob",
                    effect_id="",
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            combat_status="ongoing",
            mechanical_summary="Alice attacks from invisibility.",
            visible_outcome_facts=["Alice's blade catches Bob."],
            state_deltas=[],
            combat_state_deltas=[],
            effect_deltas=[
                {
                    "operation": "end",
                    "target_id": "alice",
                    "effect_id": "eff_invisible",
                    "slug": "invisibility",
                    "reason": "when Alice attacks",
                }
            ],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I attack Bob.",
        )
    )

    assert alice.active_effects == []
    assert "invisible" not in alice.conditions
    facts = [fact.text for fact in routed.canonical_event.observable_facts]
    assert "Invisibility ends on Alice when Alice attacks." in facts
    assert ckpt.session.active_combat.pending_visible_facts == []


def test_combat_resolver_executes_opportunity_attack_roll(monkeypatch):
    ckpt = _ckpt()
    ckpt.characters[1] = _character(
        "bob",
        "Bob",
        attack_bonus=4,
        damage="1d6+2 slashing",
    )
    values = iter([11, 13, 3, 2])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )

    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="attack_alice",
                    actor_id="alice",
                    kind="attack_roll",
                    ability="str",
                    skill="",
                    dc=12,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Alice attacks Bob with a blade.",
                    action_id="blade",
                    target_id="bob",
                ),
                PlannedRoll(
                    roll_id="oa_bob",
                    actor_id="bob",
                    kind="attack_roll",
                    ability="str",
                    skill="",
                    dc=12,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Bob makes an opportunity attack with a blade.",
                    action_id="blade",
                    target_id="alice",
                ),
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            mechanical_summary="Both attacks hit.",
            visible_outcome_facts=["Alice and Bob trade cuts."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I slash Bob and move away.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert {roll.roll_id for roll in transaction.rolls} == {
        "attack_alice",
        "oa_bob",
    }
    assert all(roll.status == "completed" for roll in transaction.rolls)
    assert {damage.roll_id for damage in transaction.damage_records} == {
        "attack_alice",
        "oa_bob",
    }


def test_combat_resolver_allows_hostile_spawn_opportunity_against_party_npc(
    monkeypatch,
):
    ckpt = _ckpt()
    ckpt.characters[0] = _character("alice", "Alice", faction="expedition")
    ckpt.characters[1] = _character(
        "npc_gearbox",
        "Gearbox",
        faction="expedition",
    )
    panther = _character(
        "mon_panther_1",
        "Panther",
        attack_bonus=4,
        damage="1d4+2 slashing",
    )
    panther.mechanics["combat_spawn"] = {
        "spawned": True,
        "monster_key": "panther",
    }
    ckpt.characters.append(panther)
    ckpt.session.active_combat = DndCombatState(
        combat_id="test",
        round_number=1,
        turn_index=1,
        combatants=[
            DndCombatantState(
                combatant_id="alice",
                character_id="alice",
                name="Alice",
                player_controlled=True,
                armor_class=12,
                hit_points_current=13,
                hit_points_max=13,
            ),
            DndCombatantState(
                combatant_id="npc_gearbox",
                character_id="npc_gearbox",
                name="Gearbox",
                armor_class=12,
                hit_points_current=13,
                hit_points_max=13,
            ),
            DndCombatantState(
                combatant_id="mon_panther_1",
                character_id="mon_panther_1",
                name="Panther",
                armor_class=12,
                hit_points_current=13,
                hit_points_max=13,
            ),
        ],
    )
    values = iter([13, 2])
    monkeypatch.setattr(
        dice.d20.expression.random,
        "randrange",
        lambda _: next(values),
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_turn_plan(
            needs_rolls=True,
            roll_requests=[
                PlannedRoll(
                    roll_id="oa_panther",
                    actor_id="mon_panther_1",
                    kind="attack_roll",
                    ability="str",
                    skill="",
                    dc=12,
                    opposed_by="",
                    advantage_state="normal",
                    reason="Panther makes an opportunity attack with a claw.",
                    action_id="blade",
                    target_id="npc_gearbox",
                )
            ],
            no_roll_reason="",
        )),
        _llm_response(RulesAdjudication(
            feasible=True,
            mechanical_summary="Panther lashes out as Gearbox leaves reach.",
            visible_outcome_facts=["The panther claws at Gearbox."],
            state_deltas=[],
            combat_state_deltas=[],
            rules_notes=[],
            fallback_reason="",
        )),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    asyncio.run(
        DndCombatResolver(client, prompt_mgr).resolve_combat_action(
            ckpt=ckpt,
            actor_id="npc_gearbox",
            intention="Gearbox backs away from the panther.",
        )
    )

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert [roll.roll_id for roll in transaction.rolls] == ["oa_panther"]
    assert {damage.roll_id for damage in transaction.damage_records} == {
        "oa_panther",
    }


def test_combat_resolver_rejects_same_faction_opportunity_attack():
    ckpt = _ckpt()
    ckpt.characters[0] = _character(
        "alice",
        "Alice",
        attack_bonus=5,
        damage="1d8+3 slashing",
        faction="expedition",
    )
    ckpt.characters[1] = _character("npc_gearbox", "Gearbox", faction="expedition")
    ckpt.session.active_combat = DndCombatState(
        combat_id="test",
        round_number=1,
        turn_index=1,
        combatants=[
            DndCombatantState(
                combatant_id="alice",
                character_id="alice",
                name="Alice",
                player_controlled=True,
                armor_class=12,
                hit_points_current=13,
                hit_points_max=13,
            ),
            DndCombatantState(
                combatant_id="npc_gearbox",
                character_id="npc_gearbox",
                name="Gearbox",
                armor_class=12,
                hit_points_current=13,
                hit_points_max=13,
            ),
        ],
    )
    client = MagicMock()
    client.complete = AsyncMock(return_value=_llm_response(_turn_plan(
        needs_rolls=True,
        roll_requests=[
            PlannedRoll(
                roll_id="friendly_oa",
                actor_id="alice",
                kind="attack_roll",
                ability="str",
                skill="",
                dc=12,
                opposed_by="",
                advantage_state="normal",
                reason="Alice makes an opportunity attack with a blade.",
                action_id="blade",
                target_id="npc_gearbox",
            ),
        ],
        no_roll_reason="",
    )))
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.return_value = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "plan"},
    ]

    with pytest.raises(ValueError, match="legal readied release"):
        asyncio.run(
            DndCombatResolver(client, prompt_mgr).resolve_combat_action(
                ckpt=ckpt,
                actor_id="npc_gearbox",
                intention="Gearbox withdraws toward Garret.",
            )
        )
