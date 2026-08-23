from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from app.schemas.one_star import (
    OneStarEquipmentEntry,
    OneStarSkillEntry,
)
from app.schemas.takeover import AuthoredCharacter


class AuthoredOneStarHiddenCapability(BaseModel):
    """A fixed-shape qualitative capability supplied by character generation."""

    model_config = ConfigDict(extra="forbid")

    capability_id: str
    description: str

    @model_validator(mode="after")
    def _clean(self) -> "AuthoredOneStarHiddenCapability":
        self.capability_id = self.capability_id.strip()
        self.description = self.description.strip()
        if not self.capability_id or not self.description:
            raise ValueError("hidden capabilities require an id and description")
        return self


class AuthoredOneStarHeroMechanics(BaseModel):
    """Character-gen's qualitative Hero identity for a newly drawn Hero.

    Birth/current stars, lobby ownership, and acquisition provenance are
    transaction facts, and numerical progression is adapter-owned.  The
    character generator only chooses affinities and authored possessions.
    """

    model_config = ConfigDict(extra="forbid")

    strong_stat_id: str
    weak_stat_id: str
    equipment: list[OneStarEquipmentEntry]
    skills: list[OneStarSkillEntry]
    conditions: list[str]
    persistent_injuries: list[str]
    innate_system_sight: bool
    hidden_capabilities: list[AuthoredOneStarHiddenCapability]

    @model_validator(mode="after")
    def _validate_starting_sheet(self) -> "AuthoredOneStarHeroMechanics":
        self.strong_stat_id = self.strong_stat_id.strip()
        self.weak_stat_id = self.weak_stat_id.strip()
        if not self.strong_stat_id or not self.weak_stat_id:
            raise ValueError("strong and weak stat ids must be non-empty")
        if self.strong_stat_id == self.weak_stat_id:
            raise ValueError("strong and weak stat ids must differ")
        capability_ids = [entry.capability_id for entry in self.hidden_capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("hidden capability ids must be unique")
        return self


class AuthoredOneStarCharacter(AuthoredCharacter):
    """Generic authored identity plus the opt-in One-Star Hero overlay."""

    model_config = ConfigDict(extra="forbid")

    one_star_hero: AuthoredOneStarHeroMechanics
