"""Domain model for a persisted wildfire event."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.fire_event_status import FireEventStatus


@dataclass(frozen=True)
class FireEvent:
    """Internal business representation of a wildfire event."""

    latitude: float
    longitude: float
    detected_at: datetime
    updated_at: datetime
    status: FireEventStatus
    detection_confidence: float
    methodology: str
    methodology_version: str

    def __post_init__(self) -> None:
        self._validate_coordinate("latitude", self.latitude, -90, 90)
        self._validate_coordinate("longitude", self.longitude, -180, 180)
        self._validate_aware_datetime("detected_at", self.detected_at)
        self._validate_aware_datetime("updated_at", self.updated_at)
        if self.updated_at < self.detected_at:
            raise ValueError("updated_at must be greater than or equal to detected_at.")
        if not isinstance(self.status, FireEventStatus):
            raise ValueError(f"status must be a FireEventStatus, got {self.status!r}.")
        if isinstance(self.detection_confidence, bool) or not isinstance(self.detection_confidence, Real):
            raise ValueError(f"detection_confidence must be numeric, got {self.detection_confidence!r}.")
        if not math.isfinite(self.detection_confidence) or not 0 <= self.detection_confidence <= 1:
            raise ValueError(f"detection_confidence must be finite within [0, 1], got {self.detection_confidence!r}.")
        self._validate_non_empty_string("methodology", self.methodology)
        self._validate_non_empty_string("methodology_version", self.methodology_version)

    @staticmethod
    def _validate_coordinate(field_name: str, value: object, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}.")
        if not minimum <= value <= maximum:
            raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}.")

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}.")

    @staticmethod
    def _validate_non_empty_string(field_name: str, value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string, got {value!r}.")
