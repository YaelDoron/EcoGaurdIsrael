"""API transport contract for US 6.2 (`GET /api/v1/fire-events/{fire_event_id}/details`).

`EventDetailsResult` is this endpoint's response model. Unlike
`src.api.schemas.active_fire_events` (US 6.1), there is no separate
`src.models` read dataclass this maps from: `EventDetailsService`
(`src/services/fire_event_read/event_details_service.py`) builds these
Pydantic DTOs directly from repository-layer data, since this feature's
scope is deliberately limited to schemas + service + router + dependency
wiring. All the usual EventDetails invariants still hold at the boundary:
strictly Pydantic DTOs (no SQLAlchemy ORM objects), ISO-8601 timezone-aware
timestamps, and missing data represented as `None` / `[]` rather than a
sentinel string.

`danger` is always `None` today: `FireDangerAssessment` is keyed by an
area-level `area_id` (a named regional fire-weather zone) with no foreign
key to `FireEvent`, and no repository query resolves one from the other.
The field is kept in the contract (shaped like `severity`) so the frontend
has a stable place to read it from once a real FireEvent<->area
relationship exists - it is not populated by guessing or spatial-nearest
matching.

`Optional[X]` is used instead of `X | None` for the same reason as
`active_fire_events.py`: Pydantic resolves annotations at class-definition
time and PEP 604 syntax is not evaluable on this project's older supported
Python runtime.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.fire_spread_prediction_status import FireSpreadPredictionStatus
from src.models.resource_status import ResourceStatus
from src.models.response_target_type import ResponseTargetType


class FireEventSummaryResponse(BaseModel):
    """Core identity and lifecycle fields for the FireEvent being viewed."""

    fire_event_id: int
    status: FireEventStatus
    latitude: float
    longitude: float
    detection_confidence: float
    detected_at: datetime
    updated_at: datetime
    methodology: str
    methodology_version: str


class SeverityAssessmentResponse(BaseModel):
    """The FireEvent's latest persisted severity assessment."""

    assessment_id: int
    status: FireSeverityAssessmentStatus
    score: Optional[float]
    level: Optional[FireSeverityLevel]
    assessed_at: datetime


class DangerAssessmentResponse(BaseModel):
    """A fire-weather danger assessment, shaped for a future FireEvent<->area link.

    See this module's docstring - `EventDetailsResult.danger` is always
    `None` today; this schema exists only to give that field a stable shape.
    """

    assessment_id: int
    status: FireDangerAssessmentStatus
    score: Optional[float]
    level: Optional[FireDangerLevel]
    assessed_at: datetime


class SpreadPredictionCellResponse(BaseModel):
    """One predicted grid cell within a wildfire-spread prediction run."""

    latitude: float
    longitude: float
    spread_probability: float
    spread_risk_score: float
    reached_step: int
    reached_minutes: int


class SpreadPredictionResponse(BaseModel):
    """The latest persisted spread prediction for one supported horizon.

    `cells` is `[]` whenever `status` is not `valid` (insufficient_data or
    inactive_event) - this is never a fallback to an older valid run, it is
    always this horizon's true latest persisted state.
    """

    horizon_minutes: int
    status: FireSpreadPredictionStatus
    predicted_at: datetime
    cells: list[SpreadPredictionCellResponse]


class ResponseTargetResponse(BaseModel):
    """One response target from the FireEvent's latest persisted target set."""

    target_order: int
    target_type: ResponseTargetType
    latitude: float
    longitude: float
    priority_score: float
    prediction_horizon_minutes: Optional[int]


class SatelliteEvidenceResponse(BaseModel):
    """A persisted satellite hotspot detection cited as evidence for the FireEvent."""

    id: int
    detected_at: datetime
    latitude: float
    longitude: float
    confidence: Optional[str]
    frp: Optional[float]
    brightness: Optional[float]
    satellite: Optional[str]
    instrument: Optional[str]
    day_night: Optional[str]


class NewsEvidenceResponse(BaseModel):
    """A persisted wildfire news report cited as evidence for the FireEvent."""

    id: int
    title: str
    summary: str
    source: str
    observed_at: datetime
    location_name: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]


class DetectionEvidenceResponse(BaseModel):
    """The direct evidence (satellite + news) that supports the FireEvent's detection."""

    satellite: list[SatelliteEvidenceResponse]
    news: list[NewsEvidenceResponse]


class FireStationResponse(BaseModel):
    """A known fire station, for map display."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    station_type: Optional[str]
    address: Optional[str]


class FirefightingResourceResponse(BaseModel):
    """A firefighting resource attached to a known fire station."""

    resource_id: str
    station_id: str
    status: ResourceStatus


class StationAllocationResponse(BaseModel):
    """One resource from a station allocated to the event's current response plan."""

    resource_id: str
    fire_event_id: int
    response_plan_id: int


class StationSummaryResponse(BaseModel):
    """Per-station resource counts and current-plan allocations, for the station popup."""

    station_id: str
    total_resources: int
    available: int
    assigned_status: int
    unavailable: int
    current_global_plan_allocations: list[StationAllocationResponse]


class ResponseActionResponse(BaseModel):
    """One assigned resource within the current response plan."""

    resource_id: str
    station_id: str
    response_target_id: int
    target_type: str
    target_priority: float
    eta_seconds: Optional[float]
    route_distance_meters: Optional[float]
    node_path: Optional[list[int]]


class BaselineComparisonResponse(BaseModel):
    """Optimized-vs-baseline comparison for the current response plan, if computed."""

    baseline_score: float
    baseline_coverage_score: float
    baseline_average_eta_seconds: Optional[float]
    score_difference: float
    improvement_percentage: Optional[float]


class CurrentResponsePlanResponse(BaseModel):
    """The FireEvent's current planning-safe response plan, if one exists."""

    plan_id: int
    generated_at: datetime
    methodology: str
    methodology_version: str
    plan_score: float
    coverage_score: float
    average_eta_seconds: Optional[float]
    actions: list[ResponseActionResponse]
    uncovered_target_ids: list[int]
    baseline_comparison: Optional[BaselineComparisonResponse]


class EventDetailsResult(BaseModel):
    """Response body for `GET /api/v1/fire-events/{fire_event_id}/details`."""

    as_of: datetime
    fire_event: FireEventSummaryResponse
    severity: Optional[SeverityAssessmentResponse]
    danger: Optional[DangerAssessmentResponse]
    detection_evidence: DetectionEvidenceResponse
    spread_predictions: list[SpreadPredictionResponse]
    targets: list[ResponseTargetResponse]
    stations: list[FireStationResponse]
    resources: list[FirefightingResourceResponse]
    station_summaries: list[StationSummaryResponse]
    current_response_plan: Optional[CurrentResponsePlanResponse]
