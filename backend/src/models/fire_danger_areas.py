"""Read-only Fire Danger areas/latest dashboard DTOs (Task A4).

Pure presentation models over data FireDangerAssessmentAgent has already
persisted - no FFWI calculation, no assessment execution, and no agent
invocation happens here.

Task A4 audit finding: the persistence model has no independent
assessment-area registry (no "areas" table) - `AssessmentArea` (see
src/models/assessment_area.py) is only ever a transient construction input
to FireDangerAssessmentAgent.assess(), and each persisted
FireDangerAssessment already embeds its own area identity (area_id/
area_name/area_latitude/area_longitude/area_radius_km) directly. As a
result, "known area" in this read model means exactly "has at least one
persisted FireDangerAssessment" - there is no way to represent an area that
is expected/configured but has never been assessed. See
FireDangerQueryService for how this shapes get_latest_for_area's None
return (unknown area_id) versus a populated snapshot (known area, which -
under the current architecture - always carries a non-None `assessment`).
`assessment` stays Optional on FireDangerAreaSnapshot for API/contract
honesty (e.g. if an independent area registry is added later), not because
it is reachable today.

A FireDangerAssessment with status=INSUFFICIENT_DATA is a real, persisted
assessment attempt (score/level are None because no trustworthy result
could be calculated) - it is represented here with `assessment` populated
and `score`/`level` both None, which is a different thing from "area has
never been assessed" (assessment is None entirely). Collapsing these two
would make "no assessment" indistinguishable from "assessed, insufficient
data", and both indistinguishable from "assessed LOW" if `status` were
dropped - so `status` is kept on the summary, mirroring
ActiveFireEventSeveritySummary's own precedent
(src/models/active_fire_events.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.optimization_validation import validate_non_empty_string, validate_positive_int


@dataclass(frozen=True)
class FireDangerAreaAssessmentSummary:
    """Presentation view of one area's latest persisted fire-danger assessment."""

    assessment_id: int
    status: FireDangerAssessmentStatus
    score: float | None
    level: FireDangerLevel | None
    assessed_at: datetime
    methodology: str
    methodology_version: str

    def __post_init__(self) -> None:
        validate_positive_int("assessment_id", self.assessment_id)
        if not isinstance(self.status, FireDangerAssessmentStatus):
            raise ValueError(f"status must be a FireDangerAssessmentStatus, got {self.status!r}")
        if self.score is not None:
            _validate_finite_number("score", self.score)
        if self.level is not None and not isinstance(self.level, FireDangerLevel):
            raise ValueError(f"level must be a FireDangerLevel or None, got {self.level!r}")
        _validate_aware_datetime("assessed_at", self.assessed_at)
        validate_non_empty_string("methodology", self.methodology)
        validate_non_empty_string("methodology_version", self.methodology_version)


@dataclass(frozen=True)
class FireDangerAreaSnapshot:
    """Presentation view of one Fire Danger assessment area for the dashboard."""

    area_id: str
    area_name: str
    area_latitude: float
    area_longitude: float
    area_radius_km: float
    assessment: FireDangerAreaAssessmentSummary | None

    def __post_init__(self) -> None:
        validate_non_empty_string("area_id", self.area_id)
        validate_non_empty_string("area_name", self.area_name)
        _validate_coordinate("area_latitude", self.area_latitude, -90, 90)
        _validate_coordinate("area_longitude", self.area_longitude, -180, 180)
        _validate_finite_number("area_radius_km", self.area_radius_km)
        if self.area_radius_km <= 0:
            raise ValueError(f"area_radius_km must be greater than 0, got {self.area_radius_km!r}")
        if self.assessment is not None and not isinstance(self.assessment, FireDangerAreaAssessmentSummary):
            raise ValueError(
                f"assessment must be a FireDangerAreaAssessmentSummary or None, got {self.assessment!r}"
            )


@dataclass(frozen=True)
class FireDangerAreasResult:
    """Root read model: the Fire Danger areas/latest dashboard snapshot at one instant."""

    as_of: datetime
    areas: tuple[FireDangerAreaSnapshot, ...]

    def __post_init__(self) -> None:
        _validate_aware_datetime("as_of", self.as_of)
        areas = _coerce_tuple("areas", self.areas)
        for area in areas:
            if not isinstance(area, FireDangerAreaSnapshot):
                raise ValueError(f"areas must contain FireDangerAreaSnapshot entries, got {area!r}")
        object.__setattr__(self, "areas", areas)


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
