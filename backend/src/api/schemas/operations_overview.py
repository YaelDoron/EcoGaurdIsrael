"""API transport contract for Task A6 (Operations Overview endpoint).

These Pydantic models are the HTTP response shape only - distinct from the
read-layer dataclasses in `src.models.operations_overview`, which
`OperationsOverviewQueryService` (src/services/operations/
operations_overview_query_service.py) already returns.
`to_operations_overview_response` is the only place that converts between
the two - a pure field-by-field copy, never a recalculation.

Maximal reuse of prior tasks' own response schemas - never a duplicate
parallel contract:
    fire_danger_areas -> A4's `FireDangerAreaLatestResponse` /
                          `to_fire_danger_areas_latest_response`
                          (src.api.schemas.fire_danger), by wrapping this
                          snapshot's already-fetched areas in a throwaway
                          `FireDangerAreasResult` just to reuse that mapper.
    active_fires        -> the existing US 6.1 `ActiveFireEventResponse` /
                          `to_active_fire_events_response`
                          (src.api.schemas.active_fire_events), via the same
                          throwaway-wrapper trick with `ActiveFireEventsResult`.
    simulation.run     -> A3's `SimulationRunStatusResponse` /
                          `to_simulation_run_status_response`
                          (src.api.schemas.simulation_control), unchanged.

`Optional[X]` is used instead of `X | None` for the same reason as
`src.api.schemas.response_plans`: Pydantic resolves annotations at
class-definition time and `X | None` (PEP 604) is not evaluable on this
project's Python 3.9 runtime for arbitrary types.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Union

from pydantic import BaseModel

from src.api.schemas.active_fire_events import ActiveFireEventResponse, to_active_fire_events_response
from src.api.schemas.fire_danger import FireDangerAreaLatestResponse, to_fire_danger_areas_latest_response
from src.api.schemas.operations_activity import OperationsActivityLocationResponse
from src.api.schemas.simulation_control import SimulationRunStatusResponse, to_simulation_run_status_response
from src.models.active_fire_events import ActiveFireEventsResult
from src.models.fire_danger_areas import FireDangerAreasResult
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.operations_activity import OperationsActivityType
from src.models.operations_overview import (
    FireDangerActivityPreview,
    FireEventActivityPreview,
    FireSeverityActivityPreview,
    GlobalPlanningRunActivityPreview,
    NewsReportActivityPreview,
    OperationsActivityFeed,
    OperationsActivityFeedItem,
    OperationsOverviewSnapshot,
    OperationsSimulationSummary,
    SatelliteHotspotActivityPreview,
    WeatherConditionsActivityPreview,
)


class OperationsSimulationSummaryResponse(BaseModel):
    """The A3 simulation-control state, as sent over HTTP.

    `run` is `None` when the control API is disabled (`enabled=False`) or
    enabled with no run ever started - never an error either way.
    """

    enabled: bool
    run: Optional[SimulationRunStatusResponse]


# ---------------------------------------------------------------------------
# Activity Feed item previews - small, typed, per-type projections
# ---------------------------------------------------------------------------


class FireDangerActivityPreviewResponse(BaseModel):
    area_name: str
    status: FireDangerAssessmentStatus
    level: Optional[FireDangerLevel]
    score: Optional[float]


class SatelliteHotspotActivityPreviewResponse(BaseModel):
    confidence: Optional[str]
    frp: Optional[float]
    location_name: Optional[str] = None


class NewsReportActivityPreviewResponse(BaseModel):
    source: str
    headline: str


class FireEventActivityPreviewResponse(BaseModel):
    status: FireEventStatus
    confidence: float


class FireSeverityActivityPreviewResponse(BaseModel):
    fire_event_id: int
    level: Optional[FireSeverityLevel]
    score: Optional[float]


class GlobalPlanningRunActivityPreviewResponse(BaseModel):
    status: GlobalPlanningRunStatus
    fire_event_count: int


class WeatherConditionsActivityPreviewResponse(BaseModel):
    area_name: str
    fire_danger_level: FireDangerLevel
    fire_danger_assessment_id: int
    temperature_c: float
    relative_humidity_pct: float
    wind_speed_kmh: float
    wind_gust_kmh: Optional[float] = None


ActivityPreviewResponse = Union[
    FireDangerActivityPreviewResponse,
    SatelliteHotspotActivityPreviewResponse,
    NewsReportActivityPreviewResponse,
    FireEventActivityPreviewResponse,
    FireSeverityActivityPreviewResponse,
    GlobalPlanningRunActivityPreviewResponse,
    WeatherConditionsActivityPreviewResponse,
]


class OperationsActivityFeedItemResponse(BaseModel):
    """One lightweight Activity Feed entry, as sent over HTTP.

    Never the full A5 detail payload - `activity_type` + `entity_id`
    together resolve through `GET /api/v1/operations/activity/{activity_type}/{entity_id}`
    (Task A5) for the full detail; `activity_id` is the stable composite
    "{activity_type}:{entity_id}" identity (entity_id alone collides across
    the six source tables).
    """

    activity_id: str
    activity_type: OperationsActivityType
    entity_id: int
    occurred_at: datetime
    available_at: datetime
    title: str
    location: Optional[OperationsActivityLocationResponse]
    preview: ActivityPreviewResponse


class OperationsActivityFeedResponse(BaseModel):
    """The bounded, globally-sorted Activity Feed, as sent over HTTP."""

    items: List[OperationsActivityFeedItemResponse]
    limit: int


class OperationsOverviewResponse(BaseModel):
    """Response body for `GET /api/v1/operations/overview`."""

    generated_at: datetime
    simulation: OperationsSimulationSummaryResponse
    fire_danger_areas: List[FireDangerAreaLatestResponse]
    active_fires: List[ActiveFireEventResponse]
    activity_feed: OperationsActivityFeedResponse


def _to_preview_response(preview: object) -> ActivityPreviewResponse:
    if isinstance(preview, FireDangerActivityPreview):
        return FireDangerActivityPreviewResponse(
            area_name=preview.area_name, status=preview.status, level=preview.level, score=preview.score
        )
    if isinstance(preview, SatelliteHotspotActivityPreview):
        return SatelliteHotspotActivityPreviewResponse(
            confidence=preview.confidence, frp=preview.frp, location_name=preview.location_name
        )
    if isinstance(preview, NewsReportActivityPreview):
        return NewsReportActivityPreviewResponse(source=preview.source, headline=preview.headline)
    if isinstance(preview, FireEventActivityPreview):
        return FireEventActivityPreviewResponse(status=preview.status, confidence=preview.confidence)
    if isinstance(preview, FireSeverityActivityPreview):
        return FireSeverityActivityPreviewResponse(
            fire_event_id=preview.fire_event_id, level=preview.level, score=preview.score
        )
    if isinstance(preview, GlobalPlanningRunActivityPreview):
        return GlobalPlanningRunActivityPreviewResponse(
            status=preview.status, fire_event_count=preview.fire_event_count
        )
    if isinstance(preview, WeatherConditionsActivityPreview):
        return WeatherConditionsActivityPreviewResponse(
            area_name=preview.area_name,
            fire_danger_level=preview.fire_danger_level,
            fire_danger_assessment_id=preview.fire_danger_assessment_id,
            temperature_c=preview.temperature_c,
            relative_humidity_pct=preview.relative_humidity_pct,
            wind_speed_kmh=preview.wind_speed_kmh,
            wind_gust_kmh=preview.wind_gust_kmh,
        )
    raise TypeError(f"Unsupported activity preview variant: {preview!r}")


def _to_location_response(
    location,
) -> Optional[OperationsActivityLocationResponse]:
    if location is None:
        return None
    return OperationsActivityLocationResponse(latitude=location.latitude, longitude=location.longitude)


def _to_feed_item_response(item: OperationsActivityFeedItem) -> OperationsActivityFeedItemResponse:
    return OperationsActivityFeedItemResponse(
        activity_id=item.activity_id,
        activity_type=item.activity_type,
        entity_id=item.entity_id,
        occurred_at=item.occurred_at,
        available_at=item.available_at,
        title=item.title,
        location=_to_location_response(item.location),
        preview=_to_preview_response(item.preview),
    )


def _to_activity_feed_response(feed: OperationsActivityFeed) -> OperationsActivityFeedResponse:
    return OperationsActivityFeedResponse(
        items=[_to_feed_item_response(item) for item in feed.items],
        limit=feed.limit,
    )


def _to_simulation_summary_response(
    simulation: OperationsSimulationSummary,
) -> OperationsSimulationSummaryResponse:
    return OperationsSimulationSummaryResponse(
        enabled=simulation.enabled,
        run=(to_simulation_run_status_response(simulation.run) if simulation.run is not None else None),
    )


def to_operations_overview_response(snapshot: OperationsOverviewSnapshot) -> OperationsOverviewResponse:
    """Map the Task A6 read model to the overview endpoint's response contract."""
    fire_danger_response = to_fire_danger_areas_latest_response(
        FireDangerAreasResult(as_of=snapshot.generated_at, areas=snapshot.fire_danger_areas)
    )
    active_fires_response = to_active_fire_events_response(
        ActiveFireEventsResult(as_of=snapshot.generated_at, items=snapshot.active_fires)
    )
    return OperationsOverviewResponse(
        generated_at=snapshot.generated_at,
        simulation=_to_simulation_summary_response(snapshot.simulation),
        fire_danger_areas=fire_danger_response.areas,
        active_fires=active_fires_response.items,
        activity_feed=_to_activity_feed_response(snapshot.activity_feed),
    )
