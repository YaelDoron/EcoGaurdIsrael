"""Read-only active-FireEvents dashboard DTOs for US 6.1.

Pure presentation models over data already persisted by Fire Detection
(FireEvent) and Fire Severity Assessment - no detection, no severity
calculation, and no agent invocation happens here. A FireEvent with no
persisted severity assessment yet is represented as `severity=None` rather
than guessed or defaulted, matching the "latest persisted state, honestly
reported" principle already used for Fire Spread's effective-state reads.

There is deliberately no `area_name` field: FireEvent does not persist a
trustworthy area/location label (see src/models/fire_event.py), and this
layer does not fabricate one via geocoding or manual coordinate mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.optimization_validation import validate_positive_int


@dataclass(frozen=True)
class ActiveFireEventSeveritySummary:
    """Presentation view of one FireEvent's latest persisted severity assessment."""

    assessment_id: int
    status: FireSeverityAssessmentStatus
    score: float | None
    level: FireSeverityLevel | None
    assessed_at: datetime

    def __post_init__(self) -> None:
        validate_positive_int("assessment_id", self.assessment_id)
        if not isinstance(self.status, FireSeverityAssessmentStatus):
            raise ValueError(f"status must be a FireSeverityAssessmentStatus, got {self.status!r}")
        if self.score is not None:
            _validate_finite_number("score", self.score)
        if self.level is not None and not isinstance(self.level, FireSeverityLevel):
            raise ValueError(f"level must be a FireSeverityLevel or None, got {self.level!r}")
        _validate_aware_datetime("assessed_at", self.assessed_at)


@dataclass(frozen=True)
class ActiveFireEventSummary:
    """Presentation view of one currently-active FireEvent for the dashboard."""

    fire_event_id: int
    status: FireEventStatus
    latitude: float
    longitude: float
    detection_confidence: float
    detected_at: datetime
    updated_at: datetime
    severity: ActiveFireEventSeveritySummary | None

    def __post_init__(self) -> None:
        validate_positive_int("fire_event_id", self.fire_event_id)
        if not isinstance(self.status, FireEventStatus):
            raise ValueError(f"status must be a FireEventStatus, got {self.status!r}")
        _validate_coordinate("latitude", self.latitude, -90, 90)
        _validate_coordinate("longitude", self.longitude, -180, 180)
        _validate_finite_number("detection_confidence", self.detection_confidence)
        if not 0 <= self.detection_confidence <= 1:
            raise ValueError(f"detection_confidence must be within [0, 1], got {self.detection_confidence!r}")
        _validate_aware_datetime("detected_at", self.detected_at)
        _validate_aware_datetime("updated_at", self.updated_at)
        if self.severity is not None and not isinstance(self.severity, ActiveFireEventSeveritySummary):
            raise ValueError(
                f"severity must be an ActiveFireEventSeveritySummary or None, got {self.severity!r}"
            )


@dataclass(frozen=True)
class ActiveFireEventsResult:
    """Root read model: the active-FireEvents dashboard snapshot at one instant."""

    as_of: datetime
    items: tuple[ActiveFireEventSummary, ...]

    def __post_init__(self) -> None:
        _validate_aware_datetime("as_of", self.as_of)
        items = _coerce_tuple("items", self.items)
        for item in items:
            if not isinstance(item, ActiveFireEventSummary):
                raise ValueError(f"items must contain ActiveFireEventSummary entries, got {item!r}")
        object.__setattr__(self, "items", items)


def _coerce_tuple(field_name: str, value: object) -> tuple:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable, got {value!r}") from exc


def _validate_finite_number(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")


def _validate_coordinate(field_name: str, value: object, minimum: float, maximum: float) -> None:
    _validate_finite_number(field_name, value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")


def _validate_aware_datetime(field_name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")
