"""Result of a fire-danger calculation."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.fire_danger_level import FireDangerLevel


@dataclass(frozen=True)
class FireDangerCalculation:
    """Calculated fire-weather danger score and application classification."""

    score: float
    level: FireDangerLevel

    def __post_init__(self) -> None:
        if not isinstance(self.score, (int, float)) or isinstance(self.score, bool):
            raise ValueError(f"score must be numeric, got {self.score!r}")
        if not isinstance(self.level, FireDangerLevel):
            raise ValueError(f"level must be a FireDangerLevel, got {self.level!r}")
