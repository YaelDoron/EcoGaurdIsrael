"""Domain model for a persisted wildfire event."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.fire_event_status import FireEventStatus


@dataclass(frozen=True)
class FireEvent:
    """Internal business representation of a wildfire event.

    `location_name` is optional, trustworthy provenance - never a read-side
    geography guess. It is set at creation time (by FireDetectionAgent, from
    its correlated evidence's own `location_name` - see
    FireDetectionEvidence) and never changed afterward by
    FireEventRepository.update_event. `None` means no trustworthy label was
    available at creation (e.g. real, non-simulation evidence) - callers may
    still apply a safe read-side fallback (see
    ActiveFireEventsService/resolve_nearest_containing_area_name), but must
    never overwrite this persisted value with that fallback.
    """

    latitude: float
    longitude: float
    detected_at: datetime
    updated_at: datetime
    status: FireEventStatus
    detection_confidence: float
    methodology: str
    methodology_version: str
    location_name: str | None = None

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
        if self.location_name is not None and not isinstance(self.location_name, str):
            raise ValueError(f"location_name must be a string or None, got {self.location_name!r}.")

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
