"""Domain model for a fire-danger assessment result."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.calculators.fire_danger.ffwi_config import FFWI_MAX_SCORE, FFWI_MIN_SCORE
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel


@dataclass(frozen=True)
class FireDangerAssessment:
    """Fire-danger assessment for an area at a specific time."""

    area_id: str
    area_name: str
    area_latitude: float
    area_longitude: float
    area_radius_km: float
    assessed_at: datetime
    status: FireDangerAssessmentStatus
    score: float | None
    level: FireDangerLevel | None
    methodology: str
    methodology_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.area_id, str) or not self.area_id.strip():
            raise ValueError(f"area_id must be a non-empty string, got {self.area_id!r}")
        if not isinstance(self.area_name, str) or not self.area_name.strip():
            raise ValueError(f"area_name must be a non-empty string, got {self.area_name!r}")
        self._validate_finite_number("area_latitude", self.area_latitude)
        self._validate_finite_number("area_longitude", self.area_longitude)
        self._validate_finite_number("area_radius_km", self.area_radius_km)
        if not -90 <= self.area_latitude <= 90:
            raise ValueError(f"area_latitude must be within [-90, 90], got {self.area_latitude!r}")
        if not -180 <= self.area_longitude <= 180:
            raise ValueError(f"area_longitude must be within [-180, 180], got {self.area_longitude!r}")
        if self.area_radius_km <= 0:
            raise ValueError(f"area_radius_km must be greater than 0, got {self.area_radius_km!r}")
        if not isinstance(self.assessed_at, datetime) or self.assessed_at.tzinfo is None:
            raise ValueError(f"assessed_at must be a timezone-aware datetime, got {self.assessed_at!r}")
        if not isinstance(self.status, FireDangerAssessmentStatus):
            raise ValueError(f"status must be a FireDangerAssessmentStatus, got {self.status!r}")
        if not isinstance(self.methodology, str) or not self.methodology.strip():
            raise ValueError(f"methodology must be a non-empty string, got {self.methodology!r}")
        if not isinstance(self.methodology_version, str) or not self.methodology_version.strip():
            raise ValueError(
                f"methodology_version must be a non-empty string, got {self.methodology_version!r}"
            )

        if self.status is FireDangerAssessmentStatus.VALID:
            self._validate_valid_result()
        elif self.status is FireDangerAssessmentStatus.INSUFFICIENT_DATA:
            self._validate_insufficient_data_result()

    def _validate_valid_result(self) -> None:
        if self.score is None:
            raise ValueError("VALID fire-danger assessments must include score.")
        if self.level is None:
            raise ValueError("VALID fire-danger assessments must include level.")
        self._validate_finite_number("score", self.score)
        if not FFWI_MIN_SCORE <= self.score <= FFWI_MAX_SCORE:
            raise ValueError(
                f"score must be within [{FFWI_MIN_SCORE}, {FFWI_MAX_SCORE}], got {self.score!r}"
            )
        if not isinstance(self.level, FireDangerLevel):
            raise ValueError(f"level must be a FireDangerLevel, got {self.level!r}")

    def _validate_insufficient_data_result(self) -> None:
        if self.score is not None:
            raise ValueError("INSUFFICIENT_DATA fire-danger assessments must not include score.")
        if self.level is not None:
            raise ValueError("INSUFFICIENT_DATA fire-danger assessments must not include level.")

    @staticmethod
    def _validate_finite_number(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value!r}")
