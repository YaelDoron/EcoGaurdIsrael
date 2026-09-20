"""Read-only Global Response Plan aggregation service (Epic 6 UI/API Rework,
Tasks B-BE-4 / B-BE-7).

`GlobalResponsePlanReadService` assembles the `GlobalResponsePlanResponse`
DTO purely by reading the latest materialized `GlobalPlanningRun` (Task
B-BE-3: `GlobalPlanningRunRepository.get_latest_materialized_generation`)
and its membership rows, then enriching each materialized child plan by
reusing US 5.5/US 6.3's existing collaborators:

    GlobalPlanningRunRepository -> StoredGlobalPlanningRun(+members)
    ResponsePlanDetailsService  -> ResponsePlanDetails (per response_plan_id)
    ResponsePlanPresenter       -> ResponsePlanDetailResponse (actions/uncovered_targets)

No route hydration, target/resource lookup, or DTO-building for one child
plan happens here directly (Task B-BE-7's "do not duplicate route hydration
logic") - this service only orchestrates calls into those two existing
collaborators and copies their output into `GlobalEventPlan`. It contains
no GA, Dijkstra, or any other optimization/calculation logic of its own,
and it never writes to any repository - a strict read/aggregate/map step,
matching `EventDetailsService`'s (US 6.2) own read-only precedent.

The aggregate `GlobalPlanShortage` (Task B-BE-4, point 3) is summed live
from each membership row's own persisted demand snapshot
(`minimum_resources`/`desired_resources`/`assigned_resources`/
`unmet_required`/`unmet_desired` - the last two already computed as
properties on `GlobalPlanningRunEvent`), across EVERY member of the run
(materialized or not) - not read from `GlobalPlanningRun.shortage_*`, which
is a different, earlier accounting (candidate/committed/unavailable
resource counts as of planning time, not a sum of these per-event fields).
A member whose cycle never reached demand computation (NO_OP/FAILED/
INSUFFICIENT_DATA/SKIPPED_INACTIVE with no snapshot) contributes nothing to
the sum rather than a fabricated 0-as-real-data value.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging

from src.api.response_plan_presenter import ResponsePlanPresenter
from src.api.schemas.global_response_plan import (
    GlobalEventPlan,
    GlobalOptimizationConfig,
    GlobalPlanMetrics,
    GlobalPlanResponse,
    GlobalPlanShortage,
    GlobalResponsePlanResponse,
)
from src.models.global_planning_run import GlobalPlanningRun
from src.models.global_planning_run_event import GlobalPlanningRunEvent
from src.repositories.global_planning_run_repository import (
    GlobalPlanningRunRepository,
    StoredGlobalPlanningRun,
    StoredGlobalPlanningRunEvent,
)
from src.services.response_planning.response_plan_details_service import ResponsePlanDetailsService

logger = logging.getLogger(__name__)


class GlobalResponsePlanReadService:
    """Assemble `GlobalResponsePlanResponse` from the latest materialized GlobalPlanningRun."""

    def __init__(
        self,
        *,
        global_planning_run_repository: GlobalPlanningRunRepository | None = None,
        response_plan_details_service: ResponsePlanDetailsService | None = None,
        response_plan_presenter: ResponsePlanPresenter | None = None,
    ) -> None:
        self._global_planning_run_repository = global_planning_run_repository or GlobalPlanningRunRepository()
        self._response_plan_details_service = response_plan_details_service or ResponsePlanDetailsService()
        self._response_plan_presenter = response_plan_presenter or ResponsePlanPresenter()

    def get_current(self, *, as_of: datetime | None = None) -> GlobalResponsePlanResponse:
        """Return the latest materialized generation, or `plan=None` if none exists yet.

        `as_of` only stamps the returned snapshot's `as_of` field (matching
        `EventDetailsService.get_event_details`'s own contract) - it is not
        a cutoff for which run is considered latest.
        """
        snapshot_time = as_of if as_of is not None else datetime.now(timezone.utc)
        self._validate_aware_datetime("as_of", snapshot_time)

        stored_run = self._global_planning_run_repository.get_latest_materialized_generation()
        if stored_run is None:
            return GlobalResponsePlanResponse(as_of=snapshot_time, plan=None)

        members = self._global_planning_run_repository.get_members(stored_run.id)
        return GlobalResponsePlanResponse(as_of=snapshot_time, plan=self._build_plan(stored_run, members))

    def _build_plan(
        self,
        stored_run: StoredGlobalPlanningRun,
        members: tuple[StoredGlobalPlanningRunEvent, ...],
    ) -> GlobalPlanResponse:
        run = stored_run.run
        return GlobalPlanResponse(
            run_id=stored_run.id,
            started_at=run.started_at,
            completed_at=run.completed_at,
            status=run.status,
            metrics=GlobalPlanMetrics(
                fitness_score=run.fitness_score,
                coverage_score=run.coverage_score,
                average_eta_seconds=run.average_eta_seconds,
            ),
            shortage=self._aggregate_shortage(members),
            optimization_config=self._to_optimization_config(run),
            events=self._build_events(members),
        )

    @staticmethod
    def _to_optimization_config(run: GlobalPlanningRun) -> GlobalOptimizationConfig | None:
        if run.random_seed is None:
            return None
        return GlobalOptimizationConfig(
            random_seed=run.random_seed,
            population_size=run.ga_population_size,
            generation_count=run.ga_generation_count,
            mutation_rate=run.ga_mutation_rate,
            crossover_rate=run.ga_crossover_rate,
        )

    @staticmethod
    def _aggregate_shortage(members: tuple[StoredGlobalPlanningRunEvent, ...]) -> GlobalPlanShortage:
        total_required = 0
        total_desired = 0
        total_assigned = 0
        unmet_required = 0
        unmet_desired = 0
        for stored_member in members:
            member = stored_member.member
            if member.minimum_resources is not None:
                total_required += member.minimum_resources
            if member.desired_resources is not None:
                total_desired += member.desired_resources
            if member.assigned_resources is not None:
                total_assigned += member.assigned_resources
            if member.unmet_required is not None:
                unmet_required += member.unmet_required
            if member.unmet_desired is not None:
                unmet_desired += member.unmet_desired
        return GlobalPlanShortage(
            total_required=total_required,
            total_desired=total_desired,
            total_assigned=total_assigned,
            unmet_required=unmet_required,
            unmet_desired=unmet_desired,
        )

    def _build_events(self, members: tuple[StoredGlobalPlanningRunEvent, ...]) -> list[GlobalEventPlan]:
        events: list[GlobalEventPlan] = []
        for stored_member in members:
            member = stored_member.member
            if member.response_plan_id is None:
                continue
            event_plan = self._build_event_plan(member)
            if event_plan is not None:
                events.append(event_plan)
        return events

    def _build_event_plan(self, member: GlobalPlanningRunEvent) -> GlobalEventPlan | None:
        details = self._response_plan_details_service.get_plan_details_by_id(member.response_plan_id)
        if details is None:
            logger.warning(
                "GlobalPlanningRunEvent for FireEvent %s references missing ResponsePlan %s",
                member.fire_event_id,
                member.response_plan_id,
            )
            return None
        presented = self._response_plan_presenter.present(details)
        return GlobalEventPlan(
            fire_event_id=member.fire_event_id,
            response_plan_id=member.response_plan_id,
            severity_level=member.severity_level,
            severity_score=member.severity_score,
            minimum_resources=member.minimum_resources,
            desired_resources=member.desired_resources,
            assigned_resources=member.assigned_resources,
            coverage_score=member.coverage_score,
            average_eta_seconds=member.average_eta_seconds,
            actions=presented.actions,
            uncovered_targets=presented.uncovered_targets,
        )

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")
