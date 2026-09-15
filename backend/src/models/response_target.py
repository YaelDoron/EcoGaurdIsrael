"""Pure operational response target domain model."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.response_target_type import ResponseTargetType


@dataclass(frozen=True)
class ResponseTarget:
    """Location and priority for response consideration, independent of routing."""

    fire_event_id: int
    target_type: ResponseTargetType
    latitude: float
    longitude: float
    priority_score: float
    prediction_horizon_minutes: int | None = None
    spread_prediction_id: int | None = None
    spread_prediction_cell_id: int | None = None

    def __post_init__(self) -> None:
        self._validate_positive_int("fire_event_id", self.fire_event_id)
        if not isinstance(self.target_type, ResponseTargetType):
            raise ValueError(f"target_type must be a ResponseTargetType, got {self.target_type!r}")

        self._validate_finite_range("latitude", self.latitude, -90.0, 90.0)
        self._validate_finite_range("longitude", self.longitude, -180.0, 180.0)
        self._validate_finite_number("priority_score", self.priority_score)

        if self.target_type is ResponseTargetType.ACTIVE_FIRE:
            self._validate_active_fire_metadata()
        elif self.target_type is ResponseTargetType.PREDICTED_RISK:
            self._validate_predicted_risk_metadata()

    def _validate_active_fire_metadata(self) -> None:
        if self.prediction_horizon_minutes is not None:
            raise ValueError("ACTIVE_FIRE targets must not include prediction_horizon_minutes.")
        if self.spread_prediction_id is not None:
            raise ValueError("ACTIVE_FIRE targets must not include spread_prediction_id.")
        if self.spread_prediction_cell_id is not None:
            raise ValueError("ACTIVE_FIRE targets must not include spread_prediction_cell_id.")

    def _validate_predicted_risk_metadata(self) -> None:
        self._validate_positive_int("prediction_horizon_minutes", self.prediction_horizon_minutes)
        self._validate_positive_int("spread_prediction_id", self.spread_prediction_id)
        self._validate_positive_int("spread_prediction_cell_id", self.spread_prediction_cell_id)

    @staticmethod
    def _validate_positive_int(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer, got {value!r}")

    @staticmethod
    def _validate_finite_number(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")

    @staticmethod
    def _validate_finite_range(field_name: str, value: object, minimum: float, maximum: float) -> None:
        ResponseTarget._validate_finite_number(field_name, value)
        if not minimum <= value <= maximum:
            raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")
