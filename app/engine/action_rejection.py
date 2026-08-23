from __future__ import annotations


class PlayerActionRejected(ValueError):
    """An expected, side-effect-free rejection safe to show to the player."""

    def __init__(self, message: str, *, reason: str = "action_rejected") -> None:
        super().__init__(message)
        self.reason = reason
