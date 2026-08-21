from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.one_star import (
    OneStarEquipmentEntry,
    OneStarHeroState,
    OneStarSkillEntry,
)
from app.schemas.takeover import AuthoredCharacter


class AuthoredOneStarHeroMechanics(BaseModel):
    """Character-gen's fiction-led starting sheet for a newly drawn Hero.

    Birth/current stars, lobby ownership, and acquisition provenance are
    transaction facts, so they are deliberately absent from this model.  The
    character generator authors only the lightweight mechanical expression of
    the person it just created.
    """

    model_config = ConfigDict(extra="forbid")

    level: int = Field(ge=1)
    experience_points: int = Field(ge=0)
    hp_current: int = Field(ge=1)
    hp_max: int = Field(ge=1)
    stats: dict[str, int]
    equipment: list[OneStarEquipmentEntry]
    skills: list[OneStarSkillEntry]
    conditions: list[str]
    persistent_injuries: list[str]
    innate_system_sight: bool
    hidden_capabilities: dict[str, str]
    private_potential: str

    @model_validator(mode="after")
    def _validate_starting_sheet(self) -> "AuthoredOneStarHeroMechanics":
        if self.hp_current > self.hp_max:
            raise ValueError("starting HP cannot exceed maximum HP")
        return self

    def to_hero_state(self, *, birth_stars: int) -> OneStarHeroState:
        return OneStarHeroState(
            birth_stars=birth_stars,
            current_stars=birth_stars,
            level=self.level,
            experience_points=self.experience_points,
            hp_current=self.hp_current,
            hp_max=self.hp_max,
            stats=dict(self.stats),
            equipment=[entry.model_copy(deep=True) for entry in self.equipment],
            skills=[entry.model_copy(deep=True) for entry in self.skills],
            conditions=list(self.conditions),
            persistent_injuries=list(self.persistent_injuries),
            innate_system_sight=self.innate_system_sight,
            hidden_capabilities=dict(self.hidden_capabilities),
            private_potential=self.private_potential,
        )


class AuthoredOneStarCharacter(AuthoredCharacter):
    """Generic authored identity plus the opt-in One-Star Hero overlay."""

    model_config = ConfigDict(extra="forbid")

    one_star_hero: AuthoredOneStarHeroMechanics
