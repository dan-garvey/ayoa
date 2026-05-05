import pytest

from app.engine import dice


def test_roll_expression_returns_auditable_result():
    result = dice.roll_expression(
        dice.RollRequest(
            roll_id="roll_damage",
            expression="2d6+3",
            actor_id="gundren",
            reason="shortsword damage",
        )
    )

    assert result.roll_id == "roll_damage"
    assert result.actor_id == "gundren"
    assert result.reason == "shortsword damage"
    assert result.expression == "2d6+3"
    assert 5 <= result.total <= 15
    assert result.detail
    assert result.crit == "none"
    assert len(result.dice) == 2
    assert all(d.size == 6 for d in result.dice)


def test_roll_d20_check_applies_positive_and_negative_modifiers(monkeypatch):
    values = iter([14, 2])
    monkeypatch.setattr(dice.d20.expression.random, "randrange", lambda _: next(values))

    positive = dice.roll_d20_check(roll_id="roll_pos", modifier=4)
    negative = dice.roll_d20_check(roll_id="roll_neg", modifier=-1)

    assert positive.expression == "1d20+4"
    assert positive.total == 19
    assert positive.dice[0].values == [15]
    assert negative.expression == "1d20-1"
    assert negative.total == 2
    assert negative.dice[0].values == [3]


def test_roll_d20_check_supports_advantage_and_disadvantage(monkeypatch):
    values = iter([4, 17, 4, 17])
    monkeypatch.setattr(dice.d20.expression.random, "randrange", lambda _: next(values))

    advantage = dice.roll_d20_check(
        roll_id="roll_adv",
        modifier=2,
        advantage_state="advantage",
    )
    disadvantage = dice.roll_d20_check(
        roll_id="roll_dis",
        modifier=2,
        advantage_state="disadvantage",
    )

    assert advantage.total == 20
    assert advantage.advantage_state == "advantage"
    assert [d.values for d in advantage.dice] == [[5], [18]]
    assert [d.kept for d in advantage.dice] == [False, True]
    assert disadvantage.total == 7
    assert disadvantage.advantage_state == "disadvantage"
    assert [d.values for d in disadvantage.dice] == [[5], [18]]
    assert [d.kept for d in disadvantage.dice] == [True, False]


def test_roll_d20_check_reports_crit_and_fail(monkeypatch):
    values = iter([19, 0])
    monkeypatch.setattr(dice.d20.expression.random, "randrange", lambda _: next(values))

    crit = dice.roll_d20_check(roll_id="roll_crit")
    fail = dice.roll_d20_check(roll_id="roll_fail")

    assert crit.total == 20
    assert crit.crit == "crit"
    assert fail.total == 1
    assert fail.crit == "fail"


def test_roll_ledger_preserves_request_metadata():
    ledger = dice.roll_ledger(
        [
            dice.RollRequest(
                roll_id="roll_1",
                expression="1d4",
                actor_id="alice",
                reason="first",
            ),
            dice.RollRequest(
                roll_id="roll_2",
                expression="1d4+1",
                actor_id="bob",
                reason="second",
            ),
        ]
    )

    assert [r.roll_id for r in ledger.rolls] == ["roll_1", "roll_2"]
    assert [r.actor_id for r in ledger.rolls] == ["alice", "bob"]
    assert [r.reason for r in ledger.rolls] == ["first", "second"]


def test_invalid_and_excessive_expressions_raise_dice_roll_error():
    with pytest.raises(dice.DiceRollError):
        dice.roll_expression(dice.RollRequest(roll_id="bad", expression="1d"))

    with pytest.raises(dice.DiceRollError):
        dice.roll_expression(
            dice.RollRequest(roll_id="too_many", expression="201d20")
        )


def test_roll_expression_rejects_advantage_state():
    with pytest.raises(dice.DiceRollError):
        dice.roll_expression(
            dice.RollRequest(
                roll_id="adv",
                expression="1d20",
                advantage_state="advantage",
            )
        )
