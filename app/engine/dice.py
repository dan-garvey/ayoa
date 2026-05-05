from __future__ import annotations

from typing import Literal

import d20
from pydantic import BaseModel, Field


AdvantageState = Literal["normal", "advantage", "disadvantage"]
CritState = Literal["none", "crit", "fail"]


class DiceRollError(ValueError):
    """Raised when a dice expression cannot be parsed or safely evaluated."""


class RollRequest(BaseModel):
    roll_id: str
    expression: str
    actor_id: str = ""
    reason: str = ""
    advantage_state: AdvantageState = "normal"


class DiceValue(BaseModel):
    size: int | str
    values: list[int] = Field(default_factory=list)
    kept: bool
    total: int


class RollResult(BaseModel):
    roll_id: str
    expression: str
    total: int
    detail: str
    crit: CritState
    dice: list[DiceValue] = Field(default_factory=list)
    actor_id: str = ""
    reason: str = ""
    advantage_state: AdvantageState = "normal"


class RollLedger(BaseModel):
    rolls: list[RollResult] = Field(default_factory=list)


_ROLLER = d20.Roller(d20.RollContext(max_rolls=200))


def roll_expression(request: RollRequest) -> RollResult:
    """Roll an engine-built expression without advantage/disadvantage."""
    if request.advantage_state != "normal":
        raise DiceRollError(
            "roll_expression does not accept advantage_state; "
            "use roll_d20_check for advantage or disadvantage."
        )
    return _roll(request)


def roll_d20_check(
    *,
    roll_id: str,
    modifier: int = 0,
    actor_id: str = "",
    reason: str = "",
    advantage_state: AdvantageState = "normal",
) -> RollResult:
    """Roll a D&D-style d20 check, save, or attack with an integer modifier."""
    expression = _format_d20_expression(modifier)
    request = RollRequest(
        roll_id=roll_id,
        expression=expression,
        actor_id=actor_id,
        reason=reason,
        advantage_state=advantage_state,
    )
    return _roll(request)


def roll_ledger(requests: list[RollRequest]) -> RollLedger:
    """Roll several engine-built requests into one auditable ledger."""
    return RollLedger(rolls=[roll_expression(request) for request in requests])


def _roll(request: RollRequest) -> RollResult:
    expression = request.expression.strip()
    if not expression:
        raise DiceRollError("Dice expression cannot be empty.")

    try:
        rolled = _ROLLER.roll(
            expression,
            allow_comments=False,
            advantage=_to_d20_advantage(request.advantage_state),
        )
    except d20.RollError as exc:
        raise DiceRollError(
            f"Could not roll expression {expression!r}: {exc}"
        ) from exc

    return RollResult(
        roll_id=request.roll_id,
        expression=expression,
        total=rolled.total,
        detail=str(rolled),
        crit=_crit_state(rolled.crit),
        dice=_extract_dice_values(rolled.expr),
        actor_id=request.actor_id,
        reason=request.reason,
        advantage_state=request.advantage_state,
    )


def _format_d20_expression(modifier: int) -> str:
    if modifier > 0:
        return f"1d20+{modifier}"
    if modifier < 0:
        return f"1d20{modifier}"
    return "1d20"


def _to_d20_advantage(advantage_state: AdvantageState) -> d20.AdvType:
    if advantage_state == "advantage":
        return d20.AdvType.ADV
    if advantage_state == "disadvantage":
        return d20.AdvType.DIS
    return d20.AdvType.NONE


def _crit_state(crit: d20.CritType) -> CritState:
    name = getattr(crit, "name", "NONE").lower()
    if name == "crit":
        return "crit"
    if name in {"fail", "failure"}:
        return "fail"
    return "none"


def _extract_dice_values(root: object) -> list[DiceValue]:
    dice_values: list[DiceValue] = []

    def visit(node: object) -> None:
        if isinstance(node, d20.Dice):
            for die in node.values:
                dice_values.append(
                    DiceValue(
                        size=die.size,
                        values=[_literal_value(v) for v in die.values],
                        kept=bool(die.kept),
                        total=int(die.total),
                    )
                )
            return
        for child in getattr(node, "children", []) or []:
            visit(child)

    visit(root)
    return dice_values


def _literal_value(value: object) -> int:
    number = getattr(value, "number", None)
    if number is None:
        number = getattr(value, "total", None)
    return int(number)
