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

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timezone
import logging
from threading import Lock

from src.api.response_plan_presenter import ResponsePlanPresenter
from src.api.schemas.global_response_plan import (
    GlobalEventPlan,
    GlobalOptimizationConfig,
    GlobalPlanCoverage,
    GlobalPlanMetrics,
    GlobalPlanResponse,
    GlobalPlanShortage,
    GlobalResponsePlanResponse,
)
from src.models.global_planning_run import GlobalPlanningRun
from src.models.global_planning_run_event import GlobalPlanningRunEvent
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.repositories.global_planning_run_repository import (
    GlobalPlanningRunRepository,
    StoredGlobalPlanningRun,
    StoredGlobalPlanningRunEvent,
)
from src.repositories.fire_event_repository import FireEventRepository
from src.services.response_planning.response_plan_details_service import ResponsePlanDetailsService

logger = logging.getLogger(__name__)

_PlanParts = tuple  # (actions, uncovered_targets) exactly as ResponsePlanPresenter.present() returns them


class PersistedPlanPartsCache:
    """Process-wide cache of each materialized plan's presented actions/uncovered targets.

    Everything `GlobalEventPlan.actions`/`uncovered_targets` is built from is
    immutable once a ResponsePlan is persisted: its actions and uncovered
    targets, the referenced ResponseTargetSet and RoutePlanningRun (new
    planning always writes NEW rows), each resource's home station and the
    road-graph node coordinates. Database ids are never reused (a demo reset
    deletes rows, it does not restart sequences), so a hit can never belong
    to a different plan. Viewing a persisted plan therefore needs no repeated
    hydration round trips - it never recomputes anything either way. The
    per-member demand/assignment snapshot and the run header are NOT cached;
    they are read fresh on every request.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max_entries
        self._entries: dict[int, _PlanParts] = {}
        self._lock = Lock()

    def get(self, response_plan_id: int) -> _PlanParts | None:
        with self._lock:
            return self._entries.get(response_plan_id)

    def put(self, response_plan_id: int, parts: _PlanParts) -> None:
        with self._lock:
            if response_plan_id not in self._entries and len(self._entries) >= self._max_entries:
                self._entries.pop(next(iter(self._entries)))
            self._entries[response_plan_id] = parts


class GlobalResponsePlanReadService:
    """Assemble `GlobalResponsePlanResponse` from the latest materialized GlobalPlanningRun."""

    def __init__(
        self,
        *,
        global_planning_run_repository: GlobalPlanningRunRepository | None = None,
        response_plan_details_service: ResponsePlanDetailsService | None = None,
        response_plan_presenter: ResponsePlanPresenter | None = None,
        plan_parts_cache: PersistedPlanPartsCache | None = None,
        max_parallel_plan_loads: int = 1,
        fire_event_repository: FireEventRepository | None = None,
    ) -> None:
        """`plan_parts_cache`/`max_parallel_plan_loads` are opt-in (the API
        dependency passes a process-wide cache and loads uncached member
        plans concurrently, since each is an independent chain of remote
        round trips); the defaults keep the plain sequential, uncached read."""
        self._global_planning_run_repository = global_planning_run_repository or GlobalPlanningRunRepository()
        self._response_plan_details_service = response_plan_details_service or ResponsePlanDetailsService()
        self._response_plan_presenter = response_plan_presenter or ResponsePlanPresenter()
        self._plan_parts_cache = plan_parts_cache
        self._max_parallel_plan_loads = max(1, max_parallel_plan_loads)
        # Opt-in (the API passes one): enables the plan coverage/lifecycle report.
        self._fire_event_repository = fire_event_repository

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
            return GlobalResponsePlanResponse(as_of=snapshot_time, plan=None, coverage=self._coverage(None, ()))

        members = self._global_planning_run_repository.get_members(stored_run.id)
        return GlobalResponsePlanResponse(
            as_of=snapshot_time,
            plan=self._build_plan(stored_run, members),
            coverage=self._coverage(stored_run, members),
        )

    def _coverage(
        self,
        stored_run: StoredGlobalPlanningRun | None,
        members: tuple[StoredGlobalPlanningRunEvent, ...],
    ) -> GlobalPlanCoverage | None:
        """Compare the currently eligible fires with the ones the latest plan covers (see GlobalPlanCoverage)."""
        if self._fire_event_repository is None:
            return None
        # Strictly CONFIRMED (RESPONSE_ELIGIBLE_STATUSES) FireEvents - never all
        # active events: a SUSPECTED fire is monitoring-only and is never planned.
        eligible = list(self._fire_event_repository.get_response_eligible_fire_event_ids())
        eligible_set = set(eligible)
        monitoring = [
            fire_event_id
            for fire_event_id in self._fire_event_repository.get_active_fire_event_ids()
            if fire_event_id not in eligible_set
        ]
        planned = {m.member.fire_event_id for m in members if m.member.response_plan_id is not None}
        covered = [fire_event_id for fire_event_id in eligible if fire_event_id in planned]
        uncovered = [fire_event_id for fire_event_id in eligible if fire_event_id not in planned]
        unplannable = self._unplannable_in_latest_finished_cycle(uncovered) if uncovered else []
        pending = [fire_event_id for fire_event_id in uncovered if fire_event_id not in unplannable]

        if pending:
            state = "updating" if stored_run is not None and covered else "generating"
        elif stored_run is None and not covered:
            state = "none"
        else:
            state = "current"
        return GlobalPlanCoverage(
            state=state,
            eligible_fire_event_ids=eligible,
            covered_fire_event_ids=covered,
            pending_fire_event_ids=pending,
            unplannable_fire_event_ids=unplannable,
            monitoring_fire_event_ids=monitoring,
            eligible_count=len(eligible),
            covered_count=len(covered),
        )

    def _unplannable_in_latest_finished_cycle(self, fire_event_ids: list[int]) -> list[int]:
        """Uncovered fires the most recent FINISHED planning cycle explicitly could not plan.

        Without this, a fire whose planning failed (e.g. no road access) would
        look "still being planned" forever. A cycle still RUNNING says nothing
        yet, so its fires stay pending.
        """
        latest = self._global_planning_run_repository.get_latest()
        if latest is None or latest.run.status is GlobalPlanningRunStatus.RUNNING:
            return []
        terminal = {GlobalPlanningRunEventStatus.FAILED, GlobalPlanningRunEventStatus.INSUFFICIENT_DATA}
        failed = {
            m.member.fire_event_id
            for m in self._global_planning_run_repository.get_members(latest.id)
            if m.member.result_status in terminal
        }
        return [fire_event_id for fire_event_id in fire_event_ids if fire_event_id in failed]

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
        planned = [stored_member.member for stored_member in members if stored_member.member.response_plan_id is not None]
        parts_by_plan_id = self._load_plan_parts(planned)
        events: list[GlobalEventPlan] = []
        for member in planned:
            parts = parts_by_plan_id.get(member.response_plan_id)
            if parts is not None:
                events.append(self._to_event_plan(member, parts))
        return events

    def _load_plan_parts(self, planned: list[GlobalPlanningRunEvent]) -> dict[int, _PlanParts]:
        found: dict[int, _PlanParts] = {}
        missing: list[GlobalPlanningRunEvent] = []
        for member in planned:
            cached = self._plan_parts_cache.get(member.response_plan_id) if self._plan_parts_cache is not None else None
            if cached is not None:
                found[member.response_plan_id] = cached
            else:
                missing.append(member)

        if len(missing) > 1 and self._max_parallel_plan_loads > 1:
            # Each worker gets its own copy of the request context, so a
            # read-only database scope set for this request still applies.
            with ThreadPoolExecutor(max_workers=min(self._max_parallel_plan_loads, len(missing))) as executor:
                futures = [executor.submit(copy_context().run, self._present_plan_parts, member) for member in missing]
                loaded = [future.result() for future in futures]
        else:
            loaded = [self._present_plan_parts(member) for member in missing]

        for member, parts in zip(missing, loaded):
            if parts is None:
                continue
            found[member.response_plan_id] = parts
            if self._plan_parts_cache is not None:
                self._plan_parts_cache.put(member.response_plan_id, parts)
        return found

    def _present_plan_parts(self, member: GlobalPlanningRunEvent) -> _PlanParts | None:
        details = self._response_plan_details_service.get_plan_details_by_id(member.response_plan_id)
        if details is None:
            logger.warning(
                "GlobalPlanningRunEvent for FireEvent %s references missing ResponsePlan %s",
                member.fire_event_id,
                member.response_plan_id,
            )
            return None
        presented = self._response_plan_presenter.present(details)
        return (presented.actions, presented.uncovered_targets)

    @staticmethod
    def _to_event_plan(member: GlobalPlanningRunEvent, parts: _PlanParts) -> GlobalEventPlan:
        actions, uncovered_targets = parts
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
            actions=actions,
            uncovered_targets=uncovered_targets,
        )

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")
