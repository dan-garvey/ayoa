from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel


class ConversationMessage(BaseModel):
    """One turn in a rolling conversation with a role (EventRouter, character agent, narrator).

    `content` is either a plain string (typical for user messages we build
    ourselves) or a list of content-block dicts (for assistant messages we
    persist verbatim from the API — required to preserve compaction blocks
    that the API needs to see again on subsequent requests).
    """
    role: Literal["user", "assistant"]
    content: Union[str, list[dict]]
