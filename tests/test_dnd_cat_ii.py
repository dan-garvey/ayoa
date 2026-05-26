import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from app.engine import dnd_cat_ii as cat
from app.engine.dnd_cat_ii import (
    DndCatIIRollsPending,
    DndCatIIResolver,
    _combat_visible_facts,
    complete_pending_player_roll,
)
from app.schemas.dnd_cat_ii import (
    DndCombatActionUse,
    DndCombatCasting,
    DndCombatTurnPlan,
    DndPlannedActionRoll,
    DndPlannedResourceSpend,
    PlannedRoll,
    PrivateOutcomeFact,
    RollPlan,
    RulesAdjudication,
)
from app.schemas.content_privacy import REDACTED_IMPORT_SENTINEL
from app.schemas.state import (
    CatIIRollRecord,
    CatIIRollTransaction,
    DndCombatState,
    OpenCatIIEvent,
)
from tests.support.factories import (
    character_record,
    checkpoint,
    dnd5e_mechanics,
    llm_response,
)


def _ckpt():
    return checkpoint(
        characters=[
            character_record(
                "alice",
                name="Alice",
                role="fighter",
                mechanics=dnd5e_mechanics(
                    ability_scores={"str": 16, "dex": 12},
                    skill_proficiencies=["athletics"],
                ),
            ),
            character_record(
                "pip",
                name="Pip",
                role="goblin",
                mechanics=dnd5e_mechanics(
                    ability_scores={"str": 8, "dex": 14},
                    skill_proficiencies=["acrobatics"],
                    name="Pip",
                ),
            ),
        ],
    )


def _llm_response(parsed):
    return llm_response(parsed, content="{}", model="gpt-5.2")


def test_combat_visible_facts_suppress_private_illusion_effect_note():
    combat = DndCombatState(
        pending_visible_facts=[
            (
                "Phantasmal Force — illusory portcullis takes hold on "
                "Orc Raider."
            )
        ]
    )
    adjudication = RulesAdjudication(
        feasible=True,
        combat_status="ongoing",
        mechanical_summary="The private illusion is sustained.",
        visible_outcome_facts=["Sera gestures toward the Orc Raider."],
        private_outcome_facts=[
            PrivateOutcomeFact(
                text="An iron portcullis blocks the east arch.",
                visible_to=["orc_raider"],
            )
        ],
        state_deltas=[],
        combat_state_deltas=[],
        effect_deltas=[{"operation": "start", "target_id": "orc_raider"}],
        spatial_deltas=[],
        rules_notes=[],
        fallback_reason="",
    )

    assert _combat_visible_facts(
        combat,
        manager_facts=adjudication.visible_outcome_facts,
        adjudication=adjudication,
    ) == ["Sera gestures toward the Orc Raider."]


def test_combat_visible_facts_keep_public_condition_effect_note():
    combat = DndCombatState(
        pending_visible_facts=[
            "Web takes hold on Bob after the initial save fails."
        ]
    )
    adjudication = RulesAdjudication(
        feasible=True,
        combat_status="ongoing",
        mechanical_summary="Web restrains Bob.",
        visible_outcome_facts=["Bob is held in place."],
        private_outcome_facts=[
            PrivateOutcomeFact(
                text="The portcullis is real to you.",
                visible_to=["orc_raider"],
            )
        ],
        state_deltas=[],
        combat_state_deltas=[],
        effect_deltas=[{"operation": "start", "target_id": "bob"}],
        spatial_deltas=[],
        rules_notes=[],
        fallback_reason="",
    )

    assert _combat_visible_facts(
        combat,
        manager_facts=adjudication.visible_outcome_facts,
        adjudication=adjudication,
    ) == [
        "Bob is held in place.",
        "Web takes hold on Bob after the initial save fails.",
    ]


def _opposed_plan() -> RollPlan:
    return RollPlan(
        needs_rolls=True,
        roll_requests=[
            PlannedRoll(
                roll_id="roll_alice",
                actor_id="alice",
                kind="skill_check",
                ability="str",
                skill="athletics",
                dc=0,
                opposed_by="roll_pip",
                advantage_state="normal",
                reason="Alice tries to shove Pip away from the door.",
            ),
            PlannedRoll(
                roll_id="roll_pip",
                actor_id="pip",
                kind="skill_check",
                ability="dex",
                skill="acrobatics",
                dc=0,
                opposed_by="roll_alice",
                advantage_state="normal",
                reason="Pip tries to keep his feet.",
            ),
        ],
        no_roll_reason="",
    )


def _social_plan() -> RollPlan:
    return RollPlan(
        needs_rolls=True,
        roll_requests=[
            PlannedRoll(
                roll_id="roll_alice",
                actor_id="alice",
                kind="skill_check",
                ability="cha",
                skill="intimidation",
                dc=0,
                opposed_by="roll_pip",
                advantage_state="normal",
                reason="Alice presses Pip with a credible threat.",
            ),
            PlannedRoll(
                roll_id="roll_pip",
                actor_id="pip",
                kind="skill_check",
                ability="wis",
                skill="insight",
                dc=0,
                opposed_by="roll_alice",
                advantage_state="normal",
                reason="Pip tries to read how dangerous Alice really is.",
            ),
        ],
        no_roll_reason="",
    )


def _open_event() -> OpenCatIIEvent:
    return OpenCatIIEvent(
        event_id="evt_open",
        initiator_id="alice",
        initiator_intention="I shove Pip away from the door",
        required_responders=["pip"],
        collected_intentions={"pip": "I twist aside"},
        opening_observer_ids=["alice", "pip"],
        opening_observable_facts=[
            "Alice lunges toward Pip at the doorway.",
        ],
    )


def test_dnd_cat_ii_content_context_redacts_imported_asset_sentinels():
    sentinels = [
        "delivery_ref=asset://synthetic/hidden-map",
        "/private/table/source-map.png",
        "raw_ocr=PROTECTED_SOURCE_EXCERPT",
    ]

    packet = cat._build_contested_packet(
        _ckpt(),
        _open_event(),
        content_context_records=[
            "content_known ref=room summary=\"Visible surface\" "
            + " ".join(sentinels)
        ],
    )
    decoded = json.loads(packet)
    flat = json.dumps(decoded, sort_keys=True)

    for sentinel in sentinels:
        assert sentinel not in flat
    assert REDACTED_IMPORT_SENTINEL in flat
    assert "Visible surface" in flat


def test_dnd_cat_ii_packet_includes_identity_without_training_spar_hints():
    lyra = character_record(
        "lyra",
        name="Lyra",
        role="cleric",
        mechanics={
            "ruleset_id": "dnd5e_basic",
            "dnd5e_sheet": {
                "identity": {
                    "species": "Hill Dwarf",
                    "classes": [{"name": "Cleric", "level": 3}],
                    "background": "Acolyte",
                },
                "statblock": {},
            },
        },
    )
    herrik = character_record("herrik", name="Herrik", role="trainer")
    ckpt = checkpoint(
        bindings={"lyra": "discord_1"},
        characters=[lyra, herrik],
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    evt = OpenCatIIEvent(
        event_id="evt_spar",
        initiator_id="lyra",
        initiator_intention=(
            "I ask Herrik for a training spar to test my footing and read "
            "an opening."
        ),
        required_responders=["herrik"],
        collected_intentions={"herrik": "I give Lyra one clean exchange."},
        opening_observer_ids=["lyra", "herrik"],
        opening_observable_facts=[
            "Lyra and Herrik square up for a training exchange."
        ],
    )

    packet = json.loads(cat._build_contested_packet(ckpt, evt))

    lyra_packet = next(
        item for item in packet["participants"]
        if item["character_id"] == "lyra"
    )
    assert lyra_packet["mechanics"]["identity"] == {
        "species": "Hill Dwarf",
        "classes": "Cleric 3",
        "background": "Acolyte",
    }
    assert "training_spar" not in packet.get("adjudication_hints", {})


def test_hidden_observer_who_beats_player_stealth_gets_next_output():
    ckpt = checkpoint(
        bindings={"alice": "discord_1"},
        characters=[
            character_record("alice", name="Alice", role="rogue"),
            character_record("dace", name="Dace", role="hidden lookout"),
        ],
    )
    evt = OpenCatIIEvent(
        event_id="evt_stealth",
        initiator_id="alice",
        initiator_intention="I sneak along the hedge to recon the lookout.",
        required_responders=["dace"],
        collected_intentions={"dace": "I stay hidden and watch for movement."},
        opening_observer_ids=["alice", "dace"],
        opening_observable_facts=[
            "Alice moves quietly along the hedge while an unseen lookout watches."
        ],
    )
    alice_roll = PlannedRoll(
        roll_id="roll_alice",
        actor_id="alice",
        kind="skill_check",
        ability="dex",
        skill="stealth",
        dc=0,
        opposed_by="roll_dace",
        advantage_state="normal",
        reason="Alice tries to move unseen along the hedge.",
    )
    dace_roll = PlannedRoll(
        roll_id="roll_dace",
        actor_id="dace",
        kind="skill_check",
        ability="wis",
        skill="perception",
        dc=0,
        opposed_by="roll_alice",
        advantage_state="normal",
        reason="Dace watches the hedge for movement.",
    )
    transaction = CatIIRollTransaction(
        transaction_id="rolltxn_stealth",
        event_id="evt_stealth",
        ruleset_id="dnd5e_basic",
        status="ready_to_finalize",
        rolls=[
            CatIIRollRecord(
                roll_id="roll_alice",
                actor_id="alice",
                status="completed",
                request=alice_roll.model_dump(),
                result={"total": 12, "detail": "d20(9) + 3"},
            ),
            CatIIRollRecord(
                roll_id="roll_dace",
                actor_id="dace",
                status="completed",
                request=dace_roll.model_dump(),
                result={"total": 13, "detail": "d20(11) + 2"},
            ),
        ],
    )
    adjudication = RulesAdjudication(
        feasible=True,
        combat_status="ongoing",
        mechanical_summary="Dace beats Alice's Stealth.",
        visible_outcome_facts=[
            "Alice reaches the hedge without drawing a shout."
        ],
        private_outcome_facts=[
            PrivateOutcomeFact(
                text="Dace catches the hedge movement and can track Alice.",
                visible_to=["dace"],
            )
        ],
        state_deltas=[],
        combat_state_deltas=[],
        effect_deltas=[],
        spatial_deltas=[],
        rules_notes=[],
        fallback_reason="",
    )

    routed = cat._compile_event_router_output(ckpt, evt, transaction, adjudication)

    assert routed.event_kind == "beat_continues"
    assert routed.next_output_character_ids == ["dace"]
    dace_observer = next(
        observer for observer in routed.observers
        if observer.character_id == "dace"
    )
    assert dace_observer.routing_role == "next_output"


def test_dnd_combat_turn_plan_accepts_nested_no_roll_action():
    plan = DndCombatTurnPlan(
        feasible=True,
        actions=[
            DndCombatActionUse(
                actor_id="alice",
                source_type="spell",
                source_id="misty_step",
                use_mode="cast",
                economy="bonus_action",
                casting=DndCombatCasting(cast_level=2),
                resource_spends=[
                    DndPlannedResourceSpend(
                        resource_id="spell_slot_2",
                        amount=1,
                        reason="cast Misty Step",
                    )
                ],
                rolls=[],
                reason="Alice teleports.",
            )
        ],
        no_action_reason="",
    )

    dumped = plan.model_dump()
    assert dumped["actions"][0]["source_id"] == "misty_step"
    assert dumped["actions"][0]["rolls"] == []
    assert dumped["actions"][0]["resource_spends"][0]["resource_id"] == (
        "spell_slot_2"
    )


def test_dnd_combat_turn_plan_accepts_nested_multi_roll_spell_action():
    plan = DndCombatTurnPlan(
        feasible=True,
        actions=[
            DndCombatActionUse(
                actor_id="alice",
                source_type="spell",
                source_id="scorching_ray",
                use_mode="cast",
                economy="action",
                casting=DndCombatCasting(cast_level=2),
                resource_spends=[
                    DndPlannedResourceSpend(resource_id="spell_slot_2")
                ],
                rolls=[
                    DndPlannedActionRoll(
                        roll_id="ray_1",
                        kind="attack_roll",
                        roller_id="alice",
                        target_id="pip",
                        ability="int",
                        skill="",
                        dc=13,
                        opposed_by="",
                        advantage_state="normal",
                        reason="first ray",
                    ),
                    DndPlannedActionRoll(
                        roll_id="ray_2",
                        kind="attack_roll",
                        roller_id="alice",
                        target_id="pip",
                        ability="int",
                        skill="",
                        dc=13,
                        opposed_by="",
                        advantage_state="normal",
                        reason="second ray",
                    ),
                ],
                reason="Alice casts Scorching Ray.",
            )
        ],
        no_action_reason="",
    )

    assert [roll.roll_id for roll in plan.actions[0].rolls] == ["ray_1", "ray_2"]


def test_dnd_combat_turn_plan_fills_missing_action_roll_ids():
    plan = DndCombatTurnPlan.model_validate({
        "feasible": True,
        "actions": [
            {
                "actor_id": "pc_marlowe_hexblade",
                "source_type": "spell",
                "source_id": "eldritch_blast",
                "use_mode": "cast",
                "economy": "action",
                "rolls": [
                    {
                        "roll_id": "",
                        "kind": "attack_roll",
                        "roller_id": "pc_marlowe_hexblade",
                        "target_id": "bandit_ambusher_03",
                        "ability": "cha",
                        "skill": "",
                        "dc": 12,
                        "opposed_by": "",
                        "advantage_state": "normal",
                        "reason": "Marlowe blasts the wounded ambusher.",
                    }
                ],
                "reason": "Marlowe casts Eldritch Blast.",
            }
        ],
        "no_action_reason": "",
    })

    assert plan.actions[0].rolls[0].roll_id == (
        "roll_pc_marlowe_hexblade_eldritch_blast_attack_roll_"
        "bandit_ambusher_03_1_1"
    )


def test_dnd_combat_action_roll_reuses_planned_roll_shape():
    action_roll = DndPlannedActionRoll(
        roll_id=" save_bob ",
        kind="saving_throw",
        roller_id=" bob ",
        target_id=" bob ",
        ability="dex",
        skill=" Acrobatics ",
        dc=-1,
        opposed_by=" ",
        advantage_state="normal",
        modifier_bonus=2,
        modifier_bonus_reason=" cover ",
        damage_on_save_success="HALF",
        reason=" Bob dives away. ",
    )

    request = action_roll.as_planned_roll(
        actor_id=" alice ",
        action_id=" Fireball ",
        effect_id=" effect_1 ",
    )

    assert request == PlannedRoll(
        roll_id="save_bob",
        actor_id="alice",
        kind="saving_throw",
        ability="dex",
        skill="acrobatics",
        dc=0,
        opposed_by="",
        advantage_state="normal",
        reason="Bob dives away.",
        action_id="fireball",
        target_id="bob",
        effect_id="effect_1",
        modifier_bonus=2,
        modifier_bonus_reason="cover",
        damage_on_save_success="half",
        damage_adjustments=[],
    )


def test_dnd_cat_ii_executes_roll_plan_and_compiles_router_output(monkeypatch):
    from app.engine import dice

    ckpt = _ckpt()
    values = iter([9, 12])
    monkeypatch.setattr(
        dice.d20.expression.random, "randrange", lambda _: next(values)
    )

    adjudication = RulesAdjudication(
        feasible=True,
        mechanical_summary="Alice beats Pip's opposed check.",
        visible_outcome_facts=[
            "Alice drives Pip back from the doorway.",
        ],
        state_deltas=[],
        rules_notes=["Opposed Athletics versus Acrobatics."],
        fallback_reason="",
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_opposed_plan()),
        _llm_response(adjudication),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCatIIResolver(client, prompt_mgr).resolve_cat_ii(
            ckpt=ckpt,
            cat_ii_event=_open_event(),
        )
    )

    assert [call.kwargs["role"] for call in client.complete.await_args_list] == [
        "event_router",
        "event_router",
    ]
    assert routed.requires_responders is False
    assert routed.event_kind == "cat_ii_resolution"
    assert routed.next_output_character_ids == []
    assert [o.character_id for o in routed.observers] == ["alice", "pip"]
    assert routed.canonical_event.observable_facts[0].text == (
        "Alice drives Pip back from the doorway."
    )
    assert "roll_alice" not in routed.decision_rationale
    assert "roll_pip" not in routed.decision_rationale

    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.status == "finalized"
    assert [r.roll_id for r in transaction.rolls] == ["roll_alice", "roll_pip"]
    assert "roll_alice" in transaction.ledger_lines[0]
    assert ckpt.session.pending_engine_state_updates == []
    assert ckpt.session_conversation == []


def test_dnd_cat_ii_scopes_private_outcome_facts(monkeypatch):
    from app.engine import dice

    ckpt = _ckpt()
    values = iter([15, 8])
    monkeypatch.setattr(
        dice.d20.expression.random, "randrange", lambda _: next(values)
    )

    adjudication = RulesAdjudication(
        feasible=True,
        mechanical_summary="Alice's warning gets through to Pip.",
        visible_outcome_facts=[
            "Alice says, 'Step away from the door before this gets worse.'",
        ],
        private_outcome_facts=[
            PrivateOutcomeFact(
                text=(
                    "Alice's threat feels immediate enough that staying in "
                    "place feels dangerous."
                ),
                visible_to=["pip"],
            )
        ],
        state_deltas=[],
        rules_notes=["Intimidation pressure resolved privately."],
        fallback_reason="",
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_social_plan()),
        _llm_response(adjudication),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]

    routed = asyncio.run(
        DndCatIIResolver(client, prompt_mgr).resolve_cat_ii(
            ckpt=ckpt,
            cat_ii_event=_open_event(),
        )
    )

    public, private = routed.canonical_event.observable_facts
    assert public.audience == "all_observers"
    assert public.text == (
        "Alice says, 'Step away from the door before this gets worse.'"
    )
    assert private.audience == "only"
    assert private.visible_to == ["pip"]
    assert private.text == (
        "Alice's threat feels immediate enough that staying in place feels "
        "dangerous."
    )
    assert routed.requires_responders is False
    assert routed.required_responders == []
    assert routed.event_kind == "beat_continues"
    assert routed.next_output_character_ids == ["pip"]
    assert {
        observer.character_id: observer.routing_role
        for observer in routed.observers
    } == {
        "alice": "observe_only",
        "pip": "next_output",
    }


def test_dnd_cat_ii_interactive_player_roll_pauses_until_roll_submitted(
    monkeypatch,
):
    from app.engine import dice

    ckpt = _ckpt()
    ckpt.session.character_bindings["alice"] = "discord_1"
    ckpt.session.config.settings.player_roll_mode = "interactive"
    values = iter([12, 9])
    monkeypatch.setattr(
        dice.d20.expression.random, "randrange", lambda _: next(values)
    )

    adjudication = RulesAdjudication(
        feasible=False,
        mechanical_summary="Pip beats Alice's opposed check.",
        visible_outcome_facts=["Alice fails to move Pip from the doorway."],
        state_deltas=[],
        rules_notes=[],
        fallback_reason="",
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(_opposed_plan()),
        _llm_response(adjudication),
    ])
    prompt_mgr = MagicMock()
    prompt_mgr.render_messages.side_effect = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "plan"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "final"}],
    ]
    evt = _open_event()
    resolver = DndCatIIResolver(client, prompt_mgr)

    try:
        asyncio.run(resolver.resolve_cat_ii(ckpt=ckpt, cat_ii_event=evt))
    except DndCatIIRollsPending as exc:
        transaction = exc.transaction
    else:
        raise AssertionError("interactive player roll did not pause")

    assert transaction.status == "awaiting_player_rolls"
    assert transaction.rolls[0].status == "pending"
    assert transaction.rolls[1].status == "completed"
    assert ckpt.session.active_act_slots["alice"].reason == "cat_ii_roll"
    assert client.complete.await_count == 1

    complete_pending_player_roll(
        ckpt,
        event_id=evt.event_id,
        roll_id="roll_alice",
        completed_by_user_id="discord_1",
    )
    routed = asyncio.run(resolver.resolve_cat_ii(ckpt=ckpt, cat_ii_event=evt))

    assert routed.canonical_event.observable_facts[0].text == (
        "Alice fails to move Pip from the doorway."
    )
    assert transaction.status == "finalized"
    assert ckpt.session.active_act_slots == {}
    assert [call.kwargs["role"] for call in client.complete.await_args_list] == [
        "event_router",
        "event_router",
    ]
