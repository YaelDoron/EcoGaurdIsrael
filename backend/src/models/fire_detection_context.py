"""Contextual (non-evidence) information available for one Fire Detection candidate."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.calculators.fire_danger.ffwi_config import FFWI_MAX_SCORE, FFWI_MIN_SCORE
from src.models.fire_danger_level import FireDangerLevel


@dataclass(frozen=True)
class FireDetectionContext:
    """Fire Danger context for one candidate. Context, never evidence of a wildfire.

    `fire_danger_available=False` means no usable Fire Danger assessment was
    found (none exists, stale, no geographic match, or INSUFFICIENT_DATA); all
    other fields are then None. A missing value is never represented by a
    substitute score: an FFWI score of 0 is a real, very-low-danger
    observation, which is different from "unknown".

    The fields a future ML feature extractor is expected to consume are
    `fire_danger_available`, `fire_danger_score` and `fire_danger_age_minutes`.
    `fire_danger_level` is derived directly from the score and is kept only for
    traceability - it should not be used as a separate ML feature.
    """

    fire_danger_available: bool
    fire_danger_score: float | None = None
    fire_danger_age_minutes: float | None = None
    fire_danger_level: FireDangerLevel | None = None
    fire_danger_assessment_id: int | None = None
    fire_danger_assessed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fire_danger_available, bool):
            raise ValueError(f"fire_danger_available must be a bool, got {self.fire_danger_available!r}")

        traceability_fields = (
            self.fire_danger_score,
            self.fire_danger_age_minutes,
            self.fire_danger_level,
            self.fire_danger_assessment_id,
            self.fire_danger_assessed_at,
        )
        if not self.fire_danger_available:
            if any(value is not None for value in traceability_fields):
                raise ValueError("Unavailable Fire Detection context must not carry Fire Danger values.")
            return

        if any(value is None for value in traceability_fields):
            raise ValueError("Available Fire Detection context requires every Fire Danger field.")
        self._validate_finite_number("fire_danger_score", self.fire_danger_score)
        if not FFWI_MIN_SCORE <= self.fire_danger_score <= FFWI_MAX_SCORE:
            raise ValueError(
                f"fire_danger_score must be within [{FFWI_MIN_SCORE}, {FFWI_MAX_SCORE}], "
                f"got {self.fire_danger_score!r}"
            )
        self._validate_finite_number("fire_danger_age_minutes", self.fire_danger_age_minutes)
        if self.fire_danger_age_minutes < 0:
            raise ValueError(f"fire_danger_age_minutes must be >= 0, got {self.fire_danger_age_minutes!r}")
        if not isinstance(self.fire_danger_level, FireDangerLevel):
            raise ValueError(f"fire_danger_level must be a FireDangerLevel, got {self.fire_danger_level!r}")
        if (
            isinstance(self.fire_danger_assessment_id, bool)
            or not isinstance(self.fire_danger_assessment_id, int)
            or self.fire_danger_assessment_id <= 0
        ):
            raise ValueError(
                f"fire_danger_assessment_id must be a positive integer, got {self.fire_danger_assessment_id!r}"
            )
        if not isinstance(self.fire_danger_assessed_at, datetime) or self.fire_danger_assessed_at.tzinfo is None:
            raise ValueError(
                f"fire_danger_assessed_at must be a timezone-aware datetime, got {self.fire_danger_assessed_at!r}"
            )

    @classmethod
    def unavailable(cls) -> FireDetectionContext:
        """Context for a candidate with no usable Fire Danger information."""
        return cls(fire_danger_available=False)

    @staticmethod
    def _validate_finite_number(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
