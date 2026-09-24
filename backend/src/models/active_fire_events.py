"""Read-only active-FireEvents dashboard DTOs for US 6.1.

Pure presentation models over data already persisted by Fire Detection
(FireEvent) and Fire Severity Assessment - no detection, no severity
calculation, and no agent invocation happens here. A FireEvent with no
persisted severity assessment yet is represented as `severity=None` rather
than guessed or defaulted, matching the "latest persisted state, honestly
reported" principle already used for Fire Spread's effective-state reads.

`location_name` below reflects `FireEvent.location_name` when the persisted
FireEvent itself carries trustworthy provenance (e.g. a simulation's own
canonical scenario location - see src/models/fire_event.py and
FireDetectionAgent._resolve_location_name), falling back to the read-side
Fire Danger area-containment lookup (same helper satellite hotspots use)
only for historical FireEvents predating that provenance, and finally to
`None` - never geocoded, never a coordinate-to-name dictionary, and a
trusted persisted value is never overwritten by the read-side fallback (see
ActiveFireEventsService._resolve_location_name for the exact priority).

`created_at` is the FireEvent row's own DB-insert timestamp (when EcoGuard
actually opened/persisted this event) - distinct from `detected_at` (the
earliest correlated evidence's own, possibly-earlier, observation time).
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
class ActiveFireEventMLSummary:
    """Lightweight presentation view of one FireEvent's ML assessment, for
    the Active Fire dashboard cards (ML Task 7).

    Deliberately trimmed to the two fields the dashboard card actually
    displays - `available` and `model_score` - unlike Event Details'
    FireEventMLAssessmentResponse, which exposes the full persisted trace.
    `model_score` is a model-estimated score from the synthetic-trained
    Logistic Regression V3 classifier, never a calibrated real-world
    probability of wildfire occurrence.
    """

    available: bool
    model_score: float | None
    # Which Fire Detection decision mode produced the assessment (e.g. "ai_hybrid_v5"); lets the dashboard label the scores correctly.
    mode: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise ValueError(f"available must be a bool, got {self.available!r}")
        if self.available:
            if self.model_score is None:
                raise ValueError("model_score must not be None when available.")
            _validate_finite_number("model_score", self.model_score)
            if not 0 <= self.model_score <= 1:
                raise ValueError(f"model_score must be within [0, 1], got {self.model_score!r}")
        elif self.model_score is not None:
            raise ValueError("model_score must be None when not available.")


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
    created_at: datetime
    severity: ActiveFireEventSeveritySummary | None
    location_name: str | None = None
    ml_summary: ActiveFireEventMLSummary | None = None

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
        _validate_aware_datetime("created_at", self.created_at)
        if self.severity is not None and not isinstance(self.severity, ActiveFireEventSeveritySummary):
            raise ValueError(
                f"severity must be an ActiveFireEventSeveritySummary or None, got {self.severity!r}"
            )
        if self.location_name is not None and not isinstance(self.location_name, str):
            raise ValueError(f"location_name must be a string or None, got {self.location_name!r}")
        if self.ml_summary is not None and not isinstance(self.ml_summary, ActiveFireEventMLSummary):
            raise ValueError(f"ml_summary must be an ActiveFireEventMLSummary or None, got {self.ml_summary!r}")


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
