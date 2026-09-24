"""Domain models for one deterministic wildfire-spread prediction run.

These models reference upstream persisted records (FireEvent,
FireSeverityAssessment) by id and must not duplicate their fields, and they
must not compute spread probabilities themselves -- the PROPAGATOR-style CA
algorithm is implemented separately. See backend/docs/fire_spread_prediction.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.fire_spread_effective_state_fingerprint import validate_effective_state_fingerprint
from src.models.fire_spread_insufficient_data_reason import FireSpreadInsufficientDataReason
from src.models.fire_spread_prediction_status import FireSpreadPredictionStatus

CA_TIME_STEP_MINUTES = 5
SUPPORTED_HORIZON_MINUTES = (30, 60)


@dataclass(frozen=True)
class FireSpreadPredictionCell:
    """One predicted grid cell produced by a wildfire-spread prediction run."""

    latitude: float
    longitude: float
    spread_probability: float
    spread_risk_score: float
    reached_step: int
    reached_minutes: int

    def __post_init__(self) -> None:
        self._validate_finite_range("latitude", self.latitude, -90, 90)
        self._validate_finite_range("longitude", self.longitude, -180, 180)
        self._validate_finite_range("spread_probability", self.spread_probability, 0.0, 1.0)
        self._validate_finite_range("spread_risk_score", self.spread_risk_score, 0.0, 100.0)

        expected_risk_score = self.spread_probability * 100
        if not math.isclose(self.spread_risk_score, expected_risk_score, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "spread_risk_score must equal spread_probability * 100, got "
                f"spread_probability={self.spread_probability!r} and spread_risk_score={self.spread_risk_score!r}"
            )

        self._validate_non_negative_int("reached_step", self.reached_step)
        self._validate_non_negative_int("reached_minutes", self.reached_minutes)

        expected_reached_minutes = self.reached_step * CA_TIME_STEP_MINUTES
        if self.reached_minutes != expected_reached_minutes:
            raise ValueError(
                f"reached_minutes must equal reached_step * {CA_TIME_STEP_MINUTES}, got "
                f"reached_step={self.reached_step!r} and reached_minutes={self.reached_minutes!r}"
            )

    @staticmethod
    def _validate_finite_range(field_name: str, value: object, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not minimum <= value <= maximum:
            raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")

    @staticmethod
    def _validate_non_negative_int(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")


@dataclass(frozen=True)
class FireSpreadPrediction:
    """One deterministic wildfire-spread prediction run for an active FireEvent.

    `insufficient_data_reason` may only be set for INSUFFICIENT_DATA. It is
    None for VALID/INACTIVE_EVENT, and may also be None for INSUFFICIENT_DATA
    rows persisted before the reason existed (unknown historical reason).
    """

    fire_event_id: int
    severity_assessment_id: int | None
    predicted_at: datetime
    horizon_minutes: int
    status: FireSpreadPredictionStatus
    methodology: str
    methodology_version: str
    cells: tuple[FireSpreadPredictionCell, ...] = ()
    effective_state_fingerprint: str | None = None
    insufficient_data_reason: FireSpreadInsufficientDataReason | None = None

    def __post_init__(self) -> None:
        self._validate_positive_int("fire_event_id", self.fire_event_id)
        self._validate_optional_positive_int("severity_assessment_id", self.severity_assessment_id)

        if not isinstance(self.predicted_at, datetime) or self.predicted_at.tzinfo is None:
            raise ValueError(f"predicted_at must be a timezone-aware datetime, got {self.predicted_at!r}")

        self._validate_positive_int("horizon_minutes", self.horizon_minutes)
        if self.horizon_minutes not in SUPPORTED_HORIZON_MINUTES:
            raise ValueError(
                f"horizon_minutes must be one of {SUPPORTED_HORIZON_MINUTES}, got {self.horizon_minutes!r}"
            )

        if not isinstance(self.status, FireSpreadPredictionStatus):
            raise ValueError(f"status must be a FireSpreadPredictionStatus, got {self.status!r}")

        self._validate_non_empty_string("methodology", self.methodology)
        self._validate_non_empty_string("methodology_version", self.methodology_version)
        validate_effective_state_fingerprint(self.effective_state_fingerprint)

        if not isinstance(self.cells, tuple) or not all(
            isinstance(cell, FireSpreadPredictionCell) for cell in self.cells
        ):
            raise ValueError(f"cells must be a tuple of FireSpreadPredictionCell, got {self.cells!r}")

        if self.status is not FireSpreadPredictionStatus.VALID and self.cells:
            raise ValueError(f"{self.status.value} spread predictions must not include cells.")

        if self.status is FireSpreadPredictionStatus.VALID and self.severity_assessment_id is None:
            raise ValueError("VALID spread predictions must include severity_assessment_id.")

        if self.insufficient_data_reason is not None:
            if not isinstance(self.insufficient_data_reason, FireSpreadInsufficientDataReason):
                raise ValueError(
                    "insufficient_data_reason must be a FireSpreadInsufficientDataReason or None, "
                    f"got {self.insufficient_data_reason!r}"
                )
            if self.status is not FireSpreadPredictionStatus.INSUFFICIENT_DATA:
                raise ValueError(f"{self.status.value} spread predictions must not include insufficient_data_reason.")

    @staticmethod
    def _validate_positive_int(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer, got {value!r}")

    @staticmethod
    def _validate_optional_positive_int(field_name: str, value: object) -> None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{field_name} must be a positive integer or None, got {value!r}")

    @staticmethod
    def _validate_non_empty_string(field_name: str, value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
