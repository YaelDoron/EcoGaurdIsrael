"""API transport contract for Task A4 (Fire Danger read endpoints).

These Pydantic models are the HTTP response shape only - distinct from the
read-layer dataclasses in `src.models.fire_danger_areas`/
`src.models.fire_danger_assessment_detail`, which `FireDangerQueryService`
(src/services/fire_danger/fire_danger_query_service.py) already returns.
`to_*_response` functions below are the only place that convert between the
two - a pure field-by-field copy (tuple -> list, enum -> its `.value` via
Pydantic's own enum serialization, plus one derived field: `age_seconds`),
never a recalculation of score/level.

`age_seconds` (elapsed wall-clock seconds since `assessed_at`) is computed
here, not in the query service, because it depends on request/response time
rather than persisted data - see fire_danger_query_service's module
docstring for why no "stale"/"expired" label is derived alongside it (no
canonical assessment-freshness rule exists in this project to reuse; do not
add one here).

`Optional[X]` is used instead of `X | None` for the same reason as
`src.api.schemas.response_plans`: Pydantic resolves annotations at
class-definition time and `X | None` (PEP 604) is not evaluable on this
project's Python 3.9 runtime for arbitrary types.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from src.models.fire_danger_areas import (
    FireDangerAreaAssessmentSummary,
    FireDangerAreaSnapshot,
    FireDangerAreasResult,
)
from src.models.fire_danger_assessment_detail import (
    FireDangerAssessmentDetail,
    FireDangerAssessmentWeatherInputSummary,
)
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel


class FireDangerAreaCenterResponse(BaseModel):
    """An area's center point, as sent over HTTP - no Leaflet-specific shape."""

    latitude: float
    longitude: float


class FireDangerAssessmentSummaryResponse(BaseModel):
    """An area's latest persisted fire-danger assessment, as sent over HTTP.

    `score`/`level` are `None` exactly when `status` is `insufficient_data`
    (a real, persisted assessment attempt that could not produce a
    trustworthy result) - never fabricated as `0`/`"low"`. This object being
    present at all already means an assessment exists; see
    FireDangerAreaLatestResponse.assessment for the "no assessment yet"
    case (a `None` at that level, not this one).
    """

    assessment_id: int
    status: FireDangerAssessmentStatus
    score: Optional[float]
    level: Optional[FireDangerLevel]
    assessed_at: datetime
    age_seconds: float
    methodology: str
    methodology_version: str


class FireDangerAreaLatestResponse(BaseModel):
    """One Fire Danger assessment area with its latest assessment, as sent over HTTP."""

    area_id: str
    area_name: str
    center: FireDangerAreaCenterResponse
    radius_km: float
    assessment: Optional[FireDangerAssessmentSummaryResponse]


class FireDangerAreasLatestResponse(BaseModel):
    """Response body for `GET /api/v1/fire-danger/areas/latest`."""

    as_of: datetime
    areas: list[FireDangerAreaLatestResponse]


class FireDangerAssessmentWeatherInputResponse(BaseModel):
    """One persisted weather observation traced as input to an assessment, as sent over HTTP."""

    observation_id: int
    station_external_id: int
    station_name: str
    observed_at: datetime


class FireDangerAssessmentDetailResponse(BaseModel):
    """Response body for `GET /api/v1/fire-danger/assessments/{assessment_id}`.

    Describes one historical persisted assessment exactly as stored - no
    recalculation, no live weather fetch.
    """

    assessment_id: int
    area_id: str
    area_name: str
    center: FireDangerAreaCenterResponse
    radius_km: float
    status: FireDangerAssessmentStatus
    score: Optional[float]
    level: Optional[FireDangerLevel]
    assessed_at: datetime
    age_seconds: float
    methodology: str
    methodology_version: str
    weather_inputs: list[FireDangerAssessmentWeatherInputResponse]


def _age_seconds(assessed_at: datetime, now: datetime) -> float:
    return (now - assessed_at).total_seconds()


def _to_assessment_summary_response(
    assessment: FireDangerAreaAssessmentSummary, now: datetime
) -> FireDangerAssessmentSummaryResponse:
    return FireDangerAssessmentSummaryResponse(
        assessment_id=assessment.assessment_id,
        status=assessment.status,
        score=assessment.score,
        level=assessment.level,
        assessed_at=assessment.assessed_at,
        age_seconds=_age_seconds(assessment.assessed_at, now),
        methodology=assessment.methodology,
        methodology_version=assessment.methodology_version,
    )


def _to_area_response(area: FireDangerAreaSnapshot, now: datetime) -> FireDangerAreaLatestResponse:
    return FireDangerAreaLatestResponse(
        area_id=area.area_id,
        area_name=area.area_name,
        center=FireDangerAreaCenterResponse(latitude=area.area_latitude, longitude=area.area_longitude),
        radius_km=area.area_radius_km,
        assessment=(
            _to_assessment_summary_response(area.assessment, now) if area.assessment is not None else None
        ),
    )


def to_fire_danger_areas_latest_response(result: FireDangerAreasResult) -> FireDangerAreasLatestResponse:
    """Map the Task A4 read model to the areas/latest endpoint's response contract.

    Uses `result.as_of` (the query snapshot time) as the reference instant
    for every area's `age_seconds`, so all ages in one response are mutually
    consistent even though computing them takes nonzero time.
    """
    return FireDangerAreasLatestResponse(
        as_of=result.as_of,
        areas=[_to_area_response(area, result.as_of) for area in result.areas],
    )


def to_fire_danger_area_latest_response(
    area: FireDangerAreaSnapshot, *, now: datetime | None = None
) -> FireDangerAreaLatestResponse:
    """Map one area snapshot to the single-area endpoint's response contract."""
    return _to_area_response(area, now if now is not None else datetime.now(timezone.utc))


def to_fire_danger_assessment_detail_response(
    detail: FireDangerAssessmentDetail, *, now: datetime | None = None
) -> FireDangerAssessmentDetailResponse:
    """Map one assessment detail read model to the assessment-detail endpoint's response contract."""
    reference_now = now if now is not None else datetime.now(timezone.utc)
    return FireDangerAssessmentDetailResponse(
        assessment_id=detail.assessment_id,
        area_id=detail.area_id,
        area_name=detail.area_name,
        center=FireDangerAreaCenterResponse(latitude=detail.area_latitude, longitude=detail.area_longitude),
        radius_km=detail.area_radius_km,
        status=detail.status,
        score=detail.score,
        level=detail.level,
        assessed_at=detail.assessed_at,
        age_seconds=_age_seconds(detail.assessed_at, reference_now),
        methodology=detail.methodology,
        methodology_version=detail.methodology_version,
        weather_inputs=[_to_weather_input_response(item) for item in detail.weather_inputs],
    )


def _to_weather_input_response(
    item: FireDangerAssessmentWeatherInputSummary,
) -> FireDangerAssessmentWeatherInputResponse:
    return FireDangerAssessmentWeatherInputResponse(
        observation_id=item.observation_id,
        station_external_id=item.station_external_id,
        station_name=item.station_name,
        observed_at=item.observed_at,
    )
