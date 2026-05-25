from __future__ import annotations

from pydantic import ConfigDict

from app.schemas.dnd_monsters import DndMonsterStatBlock
from app.schemas.takeover import AuthoredCharacter


class AuthoredDndCharacter(AuthoredCharacter):
    """Character-gen output for D&D sessions.

    The fiction fields stay identical to generic character generation, while
    D&D mechanics are an adapter-owned extension used only when the session
    ruleset asks for them.
    """

    model_config = ConfigDict(extra="forbid")

    dnd_statblock: DndMonsterStatBlock
