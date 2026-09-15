"""Domain model for a persisted active wildfire severity assessment."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.calculators.fire_severity.fire_severity_config import (
    MAX_SEVERITY_SCORE,
    MIN_SEVERITY_SCORE,
)
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel


@dataclass(frozen=True)
class FireSeverityAssessment:
    """Severity assessment result and compact vegetation snapshot for a FireEvent."""

    fire_event_id: int
    assessed_at: datetime
    status: FireSeverityAssessmentStatus
    score: float | None
    level: FireSeverityLevel | None
    methodology: str
    methodology_version: str
    vegetation_source: str | None = None
    vegetation_dataset_year: int | None = None
    vegetation_radius_km: float | None = None
    vegetation_dominant_land_cover: str | None = None
    vegetation_fuel_score: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")
        if not isinstance(self.assessed_at, datetime) or self.assessed_at.tzinfo is None:
            raise ValueError(f"assessed_at must be a timezone-aware datetime, got {self.assessed_at!r}")
        if not isinstance(self.status, FireSeverityAssessmentStatus):
            raise ValueError(f"status must be a FireSeverityAssessmentStatus, got {self.status!r}")
        if not isinstance(self.methodology, str) or not self.methodology.strip():
            raise ValueError(f"methodology must be a non-empty string, got {self.methodology!r}")
        if not isinstance(self.methodology_version, str) or not self.methodology_version.strip():
            raise ValueError(f"methodology_version must be a non-empty string, got {self.methodology_version!r}")

        if self.status is FireSeverityAssessmentStatus.VALID:
            self._validate_valid_result()
        elif self.status in (
            FireSeverityAssessmentStatus.INSUFFICIENT_DATA,
            FireSeverityAssessmentStatus.INACTIVE_EVENT,
        ):
            self._validate_not_calculated_result()

        self._validate_optional_text("vegetation_source", self.vegetation_source)
        self._validate_optional_text("vegetation_dominant_land_cover", self.vegetation_dominant_land_cover)
        if self.vegetation_dataset_year is not None and (
            isinstance(self.vegetation_dataset_year, bool)
            or not isinstance(self.vegetation_dataset_year, int)
            or self.vegetation_dataset_year <= 0
        ):
            raise ValueError(
                "vegetation_dataset_year must be a positive integer or None, "
                f"got {self.vegetation_dataset_year!r}"
            )
        if self.vegetation_radius_km is not None:
            self._validate_finite_number("vegetation_radius_km", self.vegetation_radius_km)
            if self.vegetation_radius_km <= 0:
                raise ValueError(
                    f"vegetation_radius_km must be greater than 0, got {self.vegetation_radius_km!r}"
                )
        if self.vegetation_fuel_score is not None:
            self._validate_finite_number("vegetation_fuel_score", self.vegetation_fuel_score)
            if not 0 <= self.vegetation_fuel_score <= 1:
                raise ValueError(
                    "vegetation_fuel_score must be within [0, 1], "
                    f"got {self.vegetation_fuel_score!r}"
                )

    def _validate_valid_result(self) -> None:
        if self.score is None:
            raise ValueError("VALID fire-severity assessments must include score.")
        if self.level is None:
            raise ValueError("VALID fire-severity assessments must include level.")
        self._validate_finite_number("score", self.score)
        if not MIN_SEVERITY_SCORE <= self.score <= MAX_SEVERITY_SCORE:
            raise ValueError(
                f"score must be within [{MIN_SEVERITY_SCORE}, {MAX_SEVERITY_SCORE}], got {self.score!r}"
            )
        if not isinstance(self.level, FireSeverityLevel):
            raise ValueError(f"level must be a FireSeverityLevel, got {self.level!r}")

    def _validate_not_calculated_result(self) -> None:
        if self.score is not None:
            raise ValueError(f"{self.status.value} fire-severity assessments must not include score.")
        if self.level is not None:
            raise ValueError(f"{self.status.value} fire-severity assessments must not include level.")

    @staticmethod
    def _validate_finite_number(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value!r}")

    @staticmethod
    def _validate_optional_text(field_name: str, value: object | None) -> None:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{field_name} must be a non-empty string or None, got {value!r}")
