"""API transport contract for Task A5 (Operations Activity detail endpoint).

These Pydantic models are the HTTP response shape only - distinct from the
read-layer dataclasses in `src.models.operations_activity`, which
`OperationsActivityQueryService` (src/services/operations/
operations_activity_query_service.py) already returns. `to_*_response`
functions below are the only place that convert between the two - a pure
field-by-field copy, never a recalculation.

`OperationsActivityDetailResponse` is a discriminated union on
`activity_type` (a fixed `Literal` per subtype) - the frontend switches on
that one field rather than probing which nullable block is populated. The
FIRE_DANGER variant's `details` reuses A4's own
`FireDangerAssessmentDetailResponse`/`to_fire_danger_assessment_detail_response`
unchanged (src.api.schemas.fire_danger) - never a re-derivation of that
contract.

`Optional[X]` is used instead of `X | None` for the same reason as
`src.api.schemas.response_plans`: Pydantic resolves annotations at
class-definition time and `X | None` (PEP 604) is not evaluable on this
project's Python 3.9 runtime for arbitrary types.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from src.api.schemas.fire_danger import (
    FireDangerAssessmentDetailResponse,
    to_fire_danger_assessment_detail_response,
)
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.operations_activity import (
    FireDangerActivityDetail,
    FireEventActivityDetail,
    FireSeverityActivityDetail,
    GlobalPlanningRunActivityDetail,
    NewsReportActivityDetail,
    OperationsActivityDetail,
    OperationsActivityLocation,
    SatelliteHotspotActivityDetail,
    WeatherConditionsActivityDetail,
)


class OperationsActivityLocationResponse(BaseModel):
    """A point location for the drawer/map, as sent over HTTP - never fabricated."""

    latitude: float
    longitude: float


# ---------------------------------------------------------------------------
# FIRE_DANGER
# ---------------------------------------------------------------------------


class FireDangerActivityDetailResponse(BaseModel):
    activity_type: Literal["fire_danger"] = "fire_danger"
    entity_id: int
    occurred_at: datetime
    title: str
    location: Optional[OperationsActivityLocationResponse]
    details: FireDangerAssessmentDetailResponse


# ---------------------------------------------------------------------------
# SATELLITE_HOTSPOT
# ---------------------------------------------------------------------------


class SatelliteHotspotDetailsResponse(BaseModel):
    """Persisted satellite evidence, as sent over HTTP - never a confirmed-fire claim.

    `confidence` is the raw FIRMS confidence string exactly as persisted
    (never normalized/reclassified here - no canonical EcoGuard mapping for
    it exists yet, see src/mappers/satellite_hotspot_mapper.py).
    """

    hotspot_id: int
    detected_at: datetime
    latitude: float
    longitude: float
    confidence: Optional[str]
    frp: Optional[float]
    brightness: Optional[float]
    satellite: Optional[str]
    instrument: Optional[str]
    day_night: Optional[str]


class SatelliteHotspotActivityDetailResponse(BaseModel):
    activity_type: Literal["satellite_hotspot"] = "satellite_hotspot"
    entity_id: int
    occurred_at: datetime
    title: str
    location: Optional[OperationsActivityLocationResponse]
    details: SatelliteHotspotDetailsResponse


# ---------------------------------------------------------------------------
# NEWS_REPORT
# ---------------------------------------------------------------------------


class NewsReportDetailsResponse(BaseModel):
    """Persisted wildfire news report content, exactly as stored - never re-fetched."""

    report_id: int
    source_url: str
    source_feed: str
    title: str
    summary: str
    location_name: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    published_at: Optional[datetime]
    fetched_at: datetime


class NewsReportActivityDetailResponse(BaseModel):
    activity_type: Literal["news_report"] = "news_report"
    entity_id: int
    occurred_at: datetime
    title: str
    location: Optional[OperationsActivityLocationResponse]
    details: NewsReportDetailsResponse


# ---------------------------------------------------------------------------
# FIRE_EVENT
# ---------------------------------------------------------------------------


class FireEventEvidenceRefsResponse(BaseModel):
    satellite_hotspot_ids: List[int]
    news_report_ids: List[int]


class FireEventSeverityReferenceResponse(BaseModel):
    """A lightweight pointer to a FireEvent's latest severity - see FIRE_SEVERITY for the full detail."""

    assessment_id: int
    status: FireSeverityAssessmentStatus
    score: Optional[float]
    level: Optional[FireSeverityLevel]
    assessed_at: datetime


class FireEventDetailsResponse(BaseModel):
    """Persisted FireEvent state, as sent over HTTP.

    `location_name` reflects the persisted FireEvent's own trustworthy
    provenance when available (never a fabricated area/place name) -
    `None` when no trusted label exists. `created_at` ("Opened") is the
    row's own DB-insert timestamp, distinct from `detected_at` (source
    evidence time, which may be earlier).
    """

    fire_event_id: int
    status: FireEventStatus
    detection_confidence: float
    detected_at: datetime
    updated_at: datetime
    created_at: datetime
    latitude: float
    longitude: float
    methodology: str
    methodology_version: str
    location_name: Optional[str] = None
    evidence: FireEventEvidenceRefsResponse
    latest_severity: Optional[FireEventSeverityReferenceResponse]


class FireEventActivityDetailResponse(BaseModel):
    activity_type: Literal["fire_event"] = "fire_event"
    entity_id: int
    occurred_at: datetime
    title: str
    location: Optional[OperationsActivityLocationResponse]
    details: FireEventDetailsResponse


# ---------------------------------------------------------------------------
# FIRE_SEVERITY
# ---------------------------------------------------------------------------


class FireSeverityDetailsResponse(BaseModel):
    """Persisted severity assessment and its input trace - never a recalculated result."""

    assessment_id: int
    fire_event_id: int
    status: FireSeverityAssessmentStatus
    score: Optional[float]
    level: Optional[FireSeverityLevel]
    assessed_at: datetime
    methodology: str
    methodology_version: str
    vegetation_source: Optional[str]
    vegetation_dataset_year: Optional[int]
    vegetation_radius_km: Optional[float]
    vegetation_dominant_land_cover: Optional[str]
    vegetation_fuel_score: Optional[float]
    weather_observation_ids: List[int]
    satellite_hotspot_ids: List[int]
    selected_frp_hotspot_id: Optional[int]


class FireSeverityActivityDetailResponse(BaseModel):
    activity_type: Literal["fire_severity"] = "fire_severity"
    entity_id: int
    occurred_at: datetime
    title: str
    location: Optional[OperationsActivityLocationResponse]
    details: FireSeverityDetailsResponse


# ---------------------------------------------------------------------------
# GLOBAL_PLANNING_RUN
# ---------------------------------------------------------------------------


class GlobalPlanningRunMemberResponse(BaseModel):
    """One FireEvent's membership/outcome within a GlobalPlanningRun."""

    fire_event_id: int
    event_order: int
    result_status: Optional[GlobalPlanningRunEventStatus]
    response_plan_id: Optional[int]
    severity_level: Optional[FireSeverityLevel]
    assigned_resources: Optional[int]
    coverage_score: Optional[float]
    average_eta_seconds: Optional[float]


class GlobalPlanningRunDetailsResponse(BaseModel):
    """The global multi-fire optimization run - never a single-event projection.

    `fire_event_ids`/`response_plan_ids` are derived from `members` (every
    FireEvent this cycle actually considered, and every child ResponsePlan
    it produced) - a run covering 2+ active fires shows all of them here.
    """

    global_planning_run_id: int
    status: GlobalPlanningRunStatus
    trigger: str
    started_at: datetime
    completed_at: Optional[datetime]
    methodology: str
    methodology_version: str
    fire_event_ids: List[int]
    response_plan_ids: List[int]
    coverage_score: Optional[float]
    average_eta_seconds: Optional[float]
    shortage_total_required: Optional[int]
    shortage_total_desired: Optional[int]
    shortage_total_assigned: Optional[int]
    shortage_unmet_required: Optional[int]
    shortage_unmet_desired: Optional[int]
    ga_population_size: Optional[int]
    ga_generation_count: Optional[int]
    ga_mutation_rate: Optional[float]
    ga_crossover_rate: Optional[float]
    members: List[GlobalPlanningRunMemberResponse]


class GlobalPlanningRunActivityDetailResponse(BaseModel):
    activity_type: Literal["global_planning_run"] = "global_planning_run"
    entity_id: int
    occurred_at: datetime
    title: str
    location: Optional[OperationsActivityLocationResponse]
    details: GlobalPlanningRunDetailsResponse


# ---------------------------------------------------------------------------
# WEATHER_CONDITIONS
# ---------------------------------------------------------------------------


class WeatherConditionsStationReadingResponse(BaseModel):
    """One station's raw weather reading that fed a HIGH+ Fire Danger assessment."""

    station_id: int
    station_name: str
    observation_id: int
    observed_at: datetime
    temperature: Optional[float]
    relative_humidity: Optional[float]
    wind_speed: Optional[float]
    wind_gust: Optional[float]


class WeatherConditionsDetailsResponse(BaseModel):
    """The persisted weather inputs already traced for a HIGH+ Fire Danger
    assessment - never a recalculated FFWI score."""

    fire_danger_assessment_id: int
    area_name: str
    fire_danger_level: FireDangerLevel
    assessed_at: datetime
    readings: List[WeatherConditionsStationReadingResponse]


class WeatherConditionsActivityDetailResponse(BaseModel):
    activity_type: Literal["weather_conditions"] = "weather_conditions"
    entity_id: int
    occurred_at: datetime
    title: str
    location: Optional[OperationsActivityLocationResponse]
    details: WeatherConditionsDetailsResponse


OperationsActivityDetailResponse = Annotated[
    Union[
        FireDangerActivityDetailResponse,
        SatelliteHotspotActivityDetailResponse,
        NewsReportActivityDetailResponse,
        FireEventActivityDetailResponse,
        FireSeverityActivityDetailResponse,
        GlobalPlanningRunActivityDetailResponse,
        WeatherConditionsActivityDetailResponse,
    ],
    Field(discriminator="activity_type"),
]


def _to_location_response(location: OperationsActivityLocation | None) -> OperationsActivityLocationResponse | None:
    if location is None:
        return None
    return OperationsActivityLocationResponse(latitude=location.latitude, longitude=location.longitude)


def to_operations_activity_detail_response(
    detail: OperationsActivityDetail, *, now: datetime | None = None
) -> "OperationsActivityDetailResponse":
    """Map one Task A5 domain read model to its discriminated HTTP response variant."""
    if isinstance(detail, FireDangerActivityDetail):
        reference_now = now if now is not None else datetime.now(timezone.utc)
        return FireDangerActivityDetailResponse(
            entity_id=detail.entity_id,
            occurred_at=detail.occurred_at,
            title=detail.title,
            location=_to_location_response(detail.location),
            details=to_fire_danger_assessment_detail_response(detail.details, now=reference_now),
        )

    if isinstance(detail, SatelliteHotspotActivityDetail):
        hotspot = detail.details
        return SatelliteHotspotActivityDetailResponse(
            entity_id=detail.entity_id,
            occurred_at=detail.occurred_at,
            title=detail.title,
            location=_to_location_response(detail.location),
            details=SatelliteHotspotDetailsResponse(
                hotspot_id=detail.entity_id,
                detected_at=hotspot.detected_at,
                latitude=hotspot.latitude,
                longitude=hotspot.longitude,
                confidence=hotspot.confidence,
                frp=hotspot.frp,
                brightness=hotspot.brightness,
                satellite=hotspot.satellite,
                instrument=hotspot.instrument,
                day_night=hotspot.day_night,
            ),
        )

    if isinstance(detail, NewsReportActivityDetail):
        report = detail.details
        return NewsReportActivityDetailResponse(
            entity_id=detail.entity_id,
            occurred_at=detail.occurred_at,
            title=detail.title,
            location=_to_location_response(detail.location),
            details=NewsReportDetailsResponse(
                report_id=detail.entity_id,
                source_url=report.source_url,
                source_feed=report.source_feed,
                title=report.title,
                summary=report.summary,
                location_name=report.location_name,
                latitude=report.latitude,
                longitude=report.longitude,
                published_at=report.published_at,
                fetched_at=report.fetched_at,
            ),
        )

    if isinstance(detail, FireEventActivityDetail):
        event = detail.details.fire_event
        evidence = detail.details.evidence
        severity_ref = detail.details.latest_severity
        return FireEventActivityDetailResponse(
            entity_id=detail.entity_id,
            occurred_at=detail.occurred_at,
            title=detail.title,
            location=_to_location_response(detail.location),
            details=FireEventDetailsResponse(
                fire_event_id=detail.entity_id,
                status=event.status,
                detection_confidence=event.detection_confidence,
                detected_at=event.detected_at,
                updated_at=event.updated_at,
                created_at=detail.details.created_at,
                latitude=event.latitude,
                longitude=event.longitude,
                methodology=event.methodology,
                methodology_version=event.methodology_version,
                location_name=event.location_name,
                evidence=FireEventEvidenceRefsResponse(
                    satellite_hotspot_ids=list(evidence.satellite_hotspot_ids),
                    news_report_ids=list(evidence.news_report_ids),
                ),
                latest_severity=(
                    FireEventSeverityReferenceResponse(
                        assessment_id=severity_ref.assessment_id,
                        status=severity_ref.status,
                        score=severity_ref.score,
                        level=severity_ref.level,
                        assessed_at=severity_ref.assessed_at,
                    )
                    if severity_ref is not None
                    else None
                ),
            ),
        )

    if isinstance(detail, FireSeverityActivityDetail):
        assessment = detail.details.assessment
        return FireSeverityActivityDetailResponse(
            entity_id=detail.entity_id,
            occurred_at=detail.occurred_at,
            title=detail.title,
            location=_to_location_response(detail.location),
            details=FireSeverityDetailsResponse(
                assessment_id=detail.entity_id,
                fire_event_id=assessment.fire_event_id,
                status=assessment.status,
                score=assessment.score,
                level=assessment.level,
                assessed_at=assessment.assessed_at,
                methodology=assessment.methodology,
                methodology_version=assessment.methodology_version,
                vegetation_source=assessment.vegetation_source,
                vegetation_dataset_year=assessment.vegetation_dataset_year,
                vegetation_radius_km=assessment.vegetation_radius_km,
                vegetation_dominant_land_cover=assessment.vegetation_dominant_land_cover,
                vegetation_fuel_score=assessment.vegetation_fuel_score,
                weather_observation_ids=list(detail.details.weather_observation_ids),
                satellite_hotspot_ids=list(detail.details.satellite_hotspot_ids),
                selected_frp_hotspot_id=detail.details.selected_frp_hotspot_id,
            ),
        )

    if isinstance(detail, GlobalPlanningRunActivityDetail):
        run = detail.details.run
        members = detail.details.members
        return GlobalPlanningRunActivityDetailResponse(
            entity_id=detail.entity_id,
            occurred_at=detail.occurred_at,
            title=detail.title,
            location=_to_location_response(detail.location),
            details=GlobalPlanningRunDetailsResponse(
                global_planning_run_id=detail.entity_id,
                status=run.status,
                trigger=run.trigger,
                started_at=run.started_at,
                completed_at=run.completed_at,
                methodology=run.methodology,
                methodology_version=run.methodology_version,
                fire_event_ids=list(detail.details.fire_event_ids),
                response_plan_ids=list(detail.details.response_plan_ids),
                coverage_score=run.coverage_score,
                average_eta_seconds=run.average_eta_seconds,
                shortage_total_required=run.shortage_total_required,
                shortage_total_desired=run.shortage_total_desired,
                shortage_total_assigned=run.shortage_total_assigned,
                shortage_unmet_required=run.shortage_unmet_required,
                shortage_unmet_desired=run.shortage_unmet_desired,
                ga_population_size=run.ga_population_size,
                ga_generation_count=run.ga_generation_count,
                ga_mutation_rate=run.ga_mutation_rate,
                ga_crossover_rate=run.ga_crossover_rate,
                members=[
                    GlobalPlanningRunMemberResponse(
                        fire_event_id=member.fire_event_id,
                        event_order=member.event_order,
                        result_status=member.result_status,
                        response_plan_id=member.response_plan_id,
                        severity_level=member.severity_level,
                        assigned_resources=member.assigned_resources,
                        coverage_score=member.coverage_score,
                        average_eta_seconds=member.average_eta_seconds,
                    )
                    for member in members
                ],
            ),
        )

    if isinstance(detail, WeatherConditionsActivityDetail):
        details = detail.details
        return WeatherConditionsActivityDetailResponse(
            entity_id=detail.entity_id,
            occurred_at=detail.occurred_at,
            title=detail.title,
            location=_to_location_response(detail.location),
            details=WeatherConditionsDetailsResponse(
                fire_danger_assessment_id=details.fire_danger_assessment_id,
                area_name=details.area_name,
                fire_danger_level=details.fire_danger_level,
                assessed_at=details.assessed_at,
                readings=[
                    WeatherConditionsStationReadingResponse(
                        station_id=reading.station_id,
                        station_name=reading.station_name,
                        observation_id=reading.observation_id,
                        observed_at=reading.observed_at,
                        temperature=reading.temperature,
                        relative_humidity=reading.relative_humidity,
                        wind_speed=reading.wind_speed,
                        wind_gust=reading.wind_gust,
                    )
                    for reading in details.readings
                ],
            ),
        )

    raise TypeError(f"Unsupported OperationsActivityDetail variant: {detail!r}")
