"""API transport contract for the Global Response Plan endpoints (Epic 6
UI/API Rework, Tasks B-BE-4/5/6/7).

`GlobalResponsePlanResponse` is the response model for
`GET /api/v1/global-response-plan/current`. Like `src.api.schemas.event_details`
and `src.api.schemas.response_plans`, there is no separate `src.models` read
dataclass this maps from directly: `GlobalResponsePlanReadService`
(`src/services/global_planning/global_response_plan_read_service.py`) builds
these Pydantic DTOs from `GlobalPlanningRunRepository` read data plus
per-event enrichment reused from `ResponsePlanDetailsService`/
`ResponsePlanPresenter` (US 6.3) - no route/target hydration logic is
duplicated here or in that service.

`actions`/`uncovered_targets` on `GlobalEventPlan` reuse
`src.api.schemas.response_plans.ResponsePlanActionResponse`/
`ResponsePlanTargetResponse` verbatim (the "EnrichedResponseAction"/
"UncoveredTargetResponse" shapes from the task brief) rather than redefining
an equivalent contract, since `ResponsePlanPresenter` already produces
exactly these DTOs for one plan.

Station summaries are deliberately NOT duplicated onto this contract:
`EventDetailsResult.station_summaries` (Task B-BE-2) already gives the UI
per-station allocation data keyed by `fire_event_id`, so a Global Response
Map screen composes that existing per-event endpoint rather than this one
carrying a second copy of the same station/resource read.

`Optional[X]` is used instead of `X | None` for the same reason as this
project's other schema modules: Pydantic resolves annotations at
class-definition time and PEP 604 syntax is not evaluable on this project's
older supported Python runtime.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

from src.api.schemas.response_plans import ResponsePlanActionResponse, ResponsePlanTargetResponse
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run_status import GlobalPlanningRunStatus


class GlobalPlanMetrics(BaseModel):
    """Aggregate GA scoring metrics for one GlobalPlanningRun, as sent over HTTP.

    `None` for a run whose optimization metadata was never recorded (Stage 6:
    `GlobalPlanningRunRepository.record_global_optimization_metadata` runs
    only after the GA actually produces a result) - never fabricated.
    """

    fitness_score: Optional[float]
    coverage_score: Optional[float]
    average_eta_seconds: Optional[float]


class GlobalPlanShortage(BaseModel):
    """Aggregate resource demand-vs-supply totals across every FireEvent in
    the generation, summed live from each member's own persisted demand
    snapshot (Task B-BE-4) rather than reused from the run's own persisted
    shortage_* columns, which capture a different, earlier accounting
    (candidate/committed/unavailable resource counts as of planning time)."""

    total_required: int
    total_desired: int
    total_assigned: int
    unmet_required: int
    unmet_desired: int


class GlobalOptimizationConfig(BaseModel):
    """The exact GA configuration snapshot for one GlobalPlanningRun, as sent over HTTP."""

    random_seed: int
    population_size: int
    generation_count: int
    mutation_rate: float
    crossover_rate: float


class GlobalEventPlan(BaseModel):
    """One FireEvent's materialized child plan within the global generation.

    Only emitted for a membership row that actually has a `response_plan_id`
    (Task B-BE-7) - a NO_OP/FAILED/INSUFFICIENT_DATA/SKIPPED_INACTIVE member
    never appears here (though it still contributes to `GlobalPlanShortage`
    when it has a demand snapshot). `severity_level`/`severity_score`/
    `minimum_resources`/`desired_resources`/`assigned_resources` are this
    FireEvent's exact demand snapshot as of this run (`None` when this
    member's cycle never reached demand computation); `actions`/
    `uncovered_targets` are the same enriched DTOs
    `GET /api/v1/response-plans/{plan_id}` returns for this exact
    `response_plan_id`, reused via `ResponsePlanDetailsService` +
    `ResponsePlanPresenter` - never recomputed here.
    """

    fire_event_id: int
    response_plan_id: int
    severity_level: Optional[FireSeverityLevel]
    severity_score: Optional[float]
    minimum_resources: Optional[int]
    desired_resources: Optional[int]
    assigned_resources: Optional[int]
    coverage_score: Optional[float]
    average_eta_seconds: Optional[float]
    actions: list[ResponsePlanActionResponse]
    uncovered_targets: list[ResponsePlanTargetResponse]


class GlobalPlanResponse(BaseModel):
    """The latest materialized GlobalPlanningRun, fully assembled for the
    Global Response Map."""

    run_id: int
    started_at: datetime
    completed_at: Optional[datetime]
    status: GlobalPlanningRunStatus
    metrics: GlobalPlanMetrics
    shortage: GlobalPlanShortage
    optimization_config: Optional[GlobalOptimizationConfig]
    events: list[GlobalEventPlan]


class GlobalPlanCoverage(BaseModel):
    """How far the latest Global Response Plan covers the CURRENTLY response-eligible fires.

    Derived on every read from persisted state - never a stored counter:
    `eligible` = CONFIRMED (response-eligible) FireEvents now; `covered` =
    those with a ResponsePlan in the latest materialized generation;
    `unplannable` = eligible, uncovered fires the latest finished planning
    cycle explicitly could not plan (FAILED / INSUFFICIENT_DATA); `pending` =
    the rest, i.e. still being planned. `state`:

    - `none`: no eligible fire and no plan -> "No response plan available"
    - `generating`: eligible fires are pending and no plan covers any of them yet
    - `updating`: a plan exists, but at least one eligible fire is still pending
    - `current`: every eligible fire is covered (or explicitly unplannable)
    """

    state: Literal["none", "generating", "updating", "current"]
    eligible_fire_event_ids: list[int]
    covered_fire_event_ids: list[int]
    pending_fire_event_ids: list[int]
    unplannable_fire_event_ids: list[int]
    # Active but NOT response-eligible (SUSPECTED, monitoring-only) fires. They
    # never count towards eligible/pending; reported so the UI can say why no
    # plan is being generated for them.
    monitoring_fire_event_ids: list[int] = []
    eligible_count: int
    covered_count: int


class GlobalResponsePlanResponse(BaseModel):
    """Response body for `GET /api/v1/global-response-plan/current`.

    `plan` is `None` only when no GlobalPlanningRun has ever materialized a
    ResponsePlan (`GlobalPlanningRunRepository.get_latest_materialized_generation`
    returns `None`) - e.g. a brand-new deployment with no completed cycle
    yet, or every cycle so far having been a NO_OP.
    """

    as_of: datetime
    plan: Optional[GlobalPlanResponse]
    # None only when the read service was built without a FireEvent repository
    # (unit tests); the API always reports it.
    coverage: Optional[GlobalPlanCoverage] = None
