from __future__ import annotations

from app.llm.client import TransientLLMError


def player_safe_error_message(
    exc: Exception,
    *,
    operation: str = "that action",
) -> str:
    """Return frontend-safe error text for a non-debug player surface."""
    if isinstance(exc, TransientLLMError):
        return str(exc)
    return (
        f"The engine hit an internal error while processing {operation}. "
        "Your last saved checkpoint is intact. Try again, or use /rewind "
        "to inspect recent checkpoints if the scene looks wrong."
    )
