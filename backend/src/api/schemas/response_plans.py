"""API transport contract for US 6.3 (view the recommended wildfire response plan).

These Pydantic models are the HTTP response shape only - distinct from the
read-layer dataclasses in `src.models.response_plan_details`
(`ResponsePlanDetails` etc.), which US 5.5's `ResponsePlanDetailsService`
already assembles. They are pure data containers - no enrichment logic
lives in this module; that belongs to the separate, read-only
`src.api.response_plan_presenter.ResponsePlanPresenter` (US 6.3, Task 4),
which turns a `ResponsePlanDetails` plus read-only lookups against the
exact persisted ResponsePlan/ResponseTargetSet/RoutePlanningRun/GraphNode
rows it references into a `ResponsePlanDetailResponse`.

The Task 1-3 flat contract (`ResponseActionResponse`, `ResponsePlanResponse`,
`to_response_plan_response`) that used to live here has been retired (Task 5):
both endpoints in `src.api.routers.response_plans` now return the enriched
`ResponsePlanDetailResponse` below via `ResponsePlanPresenter`, and nothing
else referenced the flat contract.

`Optional[X]` is used instead of `X | None` because Pydantic resolves
annotations at class-definition time and the `X | None` (PEP 604) syntax is
not evaluable on this project's Python 3.9 runtime for arbitrary types
(unlike the plain dataclasses in src.models, whose annotations are never
resolved at runtime).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

from src.models.response_plan_status import ResponsePlanStatus
from src.models.routing import RouteStatus


class ResponsePlanMetricsResponse(BaseModel):
    """Aggregate scoring metrics for one response plan, as sent over HTTP."""

    plan_score: float
    coverage_score: float
    average_eta_seconds: Optional[float]


class BaselineComparisonResponse(BaseModel):
    """A persisted optimized-vs-baseline comparison, as sent over HTTP."""

    baseline_score: float
    baseline_coverage_score: float
    baseline_average_eta_seconds: Optional[float]
    score_difference: float
    improvement_percentage: Optional[float]


class OptimizationConfigResponse(BaseModel):
    """The exact GA configuration snapshot for one response plan, as sent over HTTP."""

    population_size: int
    generation_count: int
    mutation_rate: float
    crossover_rate: float
    eta_reference_seconds: float
    initial_assignment_probability: float
    tournament_size: int
    elitism_count: int


class CoordinateResponse(BaseModel):
    """A latitude/longitude pair, as sent over HTTP."""

    latitude: float
    longitude: float


class ResponsePlanResourceResponse(BaseModel):
    """The assigned resource and its station origin, as sent over HTTP.

    `station_name`/`origin` are `None` only when the action's `station_id`
    no longer resolves against persisted FireStation data - never fabricated.
    """

    resource_id: str
    station_id: str
    station_name: Optional[str]
    origin: Optional[CoordinateResponse]


class ResponsePlanTargetResponse(BaseModel):
    """One response target's location and priority, as sent over HTTP.

    `target_type`/`priority_score`/`latitude`/`longitude` are `None` only
    when `response_target_id` no longer resolves against the plan's exact
    persisted ResponseTargetSet snapshot - never fabricated or recalculated.
    """

    response_target_id: int
    target_type: Optional[str]
    priority_score: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]


class ResponsePlanRouteResponse(BaseModel):
    """The persisted route for one action, as sent over HTTP.

    `status` is `None` only when no persisted RouteResult could be found for
    this action at all. `path_coordinates` is populated only when every node
    in `node_path` resolves to a persisted GraphNode - otherwise it is
    `None`, never a fabricated or partial line.
    """

    status: Optional[RouteStatus]
    eta_seconds: Optional[float]
    distance_meters: Optional[float]
    node_path: Optional[list[int]]
    path_coordinates: Optional[list[CoordinateResponse]]


class ResponsePlanActionResponse(BaseModel):
    """One assigned resource within a response plan, fully enriched for display."""

    resource: ResponsePlanResourceResponse
    target: ResponsePlanTargetResponse
    route: ResponsePlanRouteResponse


class ResponsePlanDetailResponse(BaseModel):
    """The complete, enriched Response Plan UI contract for one response plan."""

    plan_id: int
    fire_event_id: int
    response_target_set_id: int
    route_planning_run_id: int
    generated_at: datetime
    methodology: str
    methodology_version: str
    random_seed: int
    status: ResponsePlanStatus
    is_current: bool
    metrics: ResponsePlanMetricsResponse
    actions: list[ResponsePlanActionResponse]
    uncovered_targets: list[ResponsePlanTargetResponse]
    baseline_comparison: Optional[BaselineComparisonResponse]
    optimization_config: Optional[OptimizationConfigResponse]
    no_resources_during_planning: bool


class ResponsePlanEnvelopeResponse(BaseModel):
    """Shared Epic 6 top-level envelope: `{"plan": ...}`, or `{"plan": null}` when none exists.

    `plan_status` (current-plan endpoint only) tells a missing plan apart:
    `generating` - the FireEvent is CONFIRMED/response-eligible, so its plan
    is being produced by the confirmed-fire pipeline; `not_applicable` - no
    plan is expected (e.g. a SUSPECTED, monitoring-only event);
    `available` - `plan` is set. None on the plan-by-id endpoint.
    """

    plan: Optional[ResponsePlanDetailResponse]
    plan_status: Optional[Literal["available", "generating", "not_applicable"]] = None
