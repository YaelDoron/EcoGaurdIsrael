"""Pure input model for predicted locations considered as response targets."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class PredictedRiskTargetCandidate:
    """One already-computed spread prediction cell eligible for target generation."""

    fire_event_id: int
    latitude: float
    longitude: float
    risk_score: float
    prediction_horizon_minutes: int
    spread_prediction_id: int
    spread_prediction_cell_id: int

    def __post_init__(self) -> None:
        self._validate_positive_int("fire_event_id", self.fire_event_id)
        self._validate_finite_range("latitude", self.latitude, -90.0, 90.0)
        self._validate_finite_range("longitude", self.longitude, -180.0, 180.0)
        self._validate_finite_range("risk_score", self.risk_score, 0.0, 100.0)
        self._validate_positive_int("prediction_horizon_minutes", self.prediction_horizon_minutes)
        self._validate_positive_int("spread_prediction_id", self.spread_prediction_id)
        self._validate_positive_int("spread_prediction_cell_id", self.spread_prediction_cell_id)

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
