"""Normalized inputs for pure response-target generation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.predicted_risk_target_candidate import PredictedRiskTargetCandidate


@dataclass(frozen=True)
class ResponseTargetInput:
    """Complete normalized input required by ResponseTargetCalculator."""

    fire_event_id: int
    fire_latitude: float
    fire_longitude: float
    severity_score: float | None
    predicted_candidates: tuple[PredictedRiskTargetCandidate, ...] = ()

    def __post_init__(self) -> None:
        self._validate_positive_int("fire_event_id", self.fire_event_id)
        self._validate_finite_range("fire_latitude", self.fire_latitude, -90.0, 90.0)
        self._validate_finite_range("fire_longitude", self.fire_longitude, -180.0, 180.0)

        if self.severity_score is not None:
            self._validate_finite_range("severity_score", self.severity_score, 0.0, 100.0)

        try:
            candidates = tuple(self.predicted_candidates)
        except TypeError as exc:
            raise ValueError("predicted_candidates must be iterable.") from exc
        for candidate in candidates:
            if not isinstance(candidate, PredictedRiskTargetCandidate):
                raise ValueError(
                    "predicted_candidates must contain PredictedRiskTargetCandidate, "
                    f"got {candidate!r}"
                )
            if candidate.fire_event_id != self.fire_event_id:
                raise ValueError(
                    "predicted candidate fire_event_id must match input fire_event_id, got "
                    f"{candidate.fire_event_id!r} for input {self.fire_event_id!r}"
                )

        object.__setattr__(
            self,
            "predicted_candidates",
            tuple(sorted(candidates, key=_candidate_sort_key)),
        )

    @staticmethod
    def _validate_positive_int(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer, got {value!r}")

    @staticmethod
    def _validate_finite_range(field_name: str, value: object, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not minimum <= value <= maximum:
            raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")


def _candidate_sort_key(candidate: PredictedRiskTargetCandidate) -> tuple[int, int, int, float, float]:
    return (
        candidate.prediction_horizon_minutes,
        candidate.spread_prediction_id,
        candidate.spread_prediction_cell_id,
        candidate.latitude,
        candidate.longitude,
    )
