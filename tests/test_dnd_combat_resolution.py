import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine import dice
from app.engine.dnd_cat_ii import DndCombatResolver
from app.llm.client import LLMResponse
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_cat_ii import PlannedRoll, RollPlan, RulesAdjudication
from app.schemas.state import (
    DndCombatantState,
    DndCombatState,
    SessionState,
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


def _character(
    character_id: str,
    name: str,
    *,
    attack_bonus: int = 0,
    damage: str = "",
    actions: list[dict] | None = None,
    spellcasting: dict | None = None,
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
        public_sheet=PublicSheet(role="combatant"),
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
                    "spellcasting": spellcasting or {},
                }
            },
        },
    )


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
        _llm_response(RollPlan(
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
    assert routed.ends_beat_reason == "ruleset_resolution"
    assert routed.canonical_event.observable_facts[0].text == (
        "Alice cuts Bob across the guard."
    )
    assert "charlie" not in {observer.character_id for observer in routed.observers}
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.source == "combat"
    assert transaction.status == "finalized"
    assert transaction.rolls[0].modifier == 5
    assert transaction.damage_records[0].roll_id == "attack_alice"
    assert transaction.damage_records[0].amount == 7
    assert transaction.damage_records[0].applied is True
    assert any("damage_for=attack_alice" in line for line in transaction.ledger_lines)
    assert ckpt.session.pending_router_state_changes[0].startswith(
        "D&D combat resolved:"
    )
    assert ckpt.session_conversation == []


def test_combat_resolver_can_end_combat_from_adjudication():
    ckpt = _ckpt()
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(RollPlan(
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
        _llm_response(RollPlan(
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
        _llm_response(RollPlan(
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
                "attack": {"bonus": 7, "damage": "1d6+4 piercing"},
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
        _llm_response(RollPlan(
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

    first_packet = prompt_mgr.render_messages.call_args_list[0].kwargs[
        "combat_action_packet"
    ]
    assert '"id": "shortbow"' in first_packet
    assert '"damage": "1d6+4 piercing"' in first_packet
    transaction = ckpt.session.cat_ii_roll_transactions[0]
    assert transaction.rolls[0].modifier == 7
    assert transaction.damage_records[0].expression == "1d6+4"


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
                    "save": {"ability": "wis", "dc": 13},
                    "damage": [],
                    "healing": [],
                    "consumes": [{"resource_id": "spell_slot_2", "amount": 1}],
                }
            ],
        },
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        _llm_response(RollPlan(
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

    first_packet = prompt_mgr.render_messages.call_args_list[0].kwargs[
        "combat_action_packet"
    ]
    assert '"name": "Hold Person"' in first_packet
    assert '"spell_save_dc": 13' in first_packet
    assert '"slots": {' in first_packet


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
        _llm_response(RollPlan(
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
