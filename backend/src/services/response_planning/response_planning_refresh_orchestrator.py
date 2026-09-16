"""Refresh POLICY orchestration for Epic 5, User Story 5.4, Task 5.

ResponsePlanningRefreshOrchestrator decides whether a FireEvent is eligible
for active planning, whether current planning inputs are sufficient,
whether the effective planning state changed, and whether a full new
planning cycle - or only a missing baseline comparison - must be run. It
contains no Dijkstra, no GA, no severity/spread/detection logic, no
target-priority or resource-availability calculation, and never builds
ResponseOptimizationInput: routing, optimization, and baseline comparison
are injected collaborators behind response_planning_refresh_ports.py's
Protocols. Wiring real adapters for those against the production
RoutePlanningAgent/ResponseOptimizationAgent/BaselineComparisonService is
Task 6's job.

A missing FireEvent (FireEventRepository.get_by_id returns None) is treated
as an orchestration failure (PlanningRefreshStatus.FAILED), not
INSUFFICIENT_DATA or INACTIVE_EVENT: neither status means "no such record",
and this mirrors ResponseTargetInputService.prepare_input's own precedent of
raising for a FireEvent that does not exist. The raise is caught by this
orchestrator's single top-level boundary, matching
OperationalRefreshOrchestrator's established exception-to-FAILED pattern.
"""
from __future__ import annotations

from datetime import datetime
import logging

from src.calculators.response_optimization.response_optimization_config import DEFAULT_RANDOM_SEED
from src.models.fire_event_status import FireEventStatus
from src.models.planning_effective_state_status import PlanningEffectiveStateStatus
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.plan_comparison_repository import PlanComparisonRepository, StoredPlanComparison
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository, StoredResponsePlan
from src.services.response_planning.planning_effective_state_builder import PlanningEffectiveStateBuilder
from src.services.response_planning.planning_refresh_result import PlanningRefreshResult, PlanningRefreshStatus
from src.services.response_planning.response_planning_refresh_ports import (
    BaselineComparisonCollaborator,
    OptimizationCollaborator,
    RoutingCollaborator,
)

logger = logging.getLogger(__name__)

_INACTIVE_EVENT_STATUSES = frozenset({FireEventStatus.RESOLVED, FireEventStatus.DISMISSED})


class ResponsePlanningRefreshOrchestrator:
    """Decide whether US 5.4 needs a new planning cycle, and sequence it via injected collaborators."""

    def __init__(
        self,
        *,
        planning_state_builder: PlanningEffectiveStateBuilder,
        routing_collaborator: RoutingCollaborator,
        optimization_collaborator: OptimizationCollaborator,
        baseline_collaborator: BaselineComparisonCollaborator,
        fire_event_repository: FireEventRepository | None = None,
        response_plan_repository: ResponsePlanRepository | None = None,
        response_plan_planning_state_repository: ResponsePlanPlanningStateRepository | None = None,
        plan_comparison_repository: PlanComparisonRepository | None = None,
        optimization_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        self._planning_state_builder = planning_state_builder
        self._routing_collaborator = routing_collaborator
        self._optimization_collaborator = optimization_collaborator
        self._baseline_collaborator = baseline_collaborator
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._response_plan_repository = response_plan_repository or ResponsePlanRepository()
        self._response_plan_planning_state_repository = (
            response_plan_planning_state_repository or ResponsePlanPlanningStateRepository()
        )
        self._plan_comparison_repository = plan_comparison_repository or PlanComparisonRepository()
        self._optimization_seed = optimization_seed

    def refresh(self, *, fire_event_id: int, as_of: datetime) -> PlanningRefreshResult:
        """Decide and, if needed, drive one US 5.4 planning-refresh cycle for one FireEvent."""
        self._validate_request(fire_event_id, as_of)
        try:
            return self._refresh(fire_event_id, as_of)
        except Exception as exc:  # noqa: BLE001 - conservative orchestration failure boundary.
            logger.exception("Planning refresh failed for FireEvent %s", fire_event_id)
            return PlanningRefreshResult(
                status=PlanningRefreshStatus.FAILED,
                fire_event_id=fire_event_id,
                route_planning_run_id=None,
                response_plan_id=None,
                comparison_id=None,
                error=str(exc) or "Planning refresh failed.",
            )

    def _refresh(self, fire_event_id: int, as_of: datetime) -> PlanningRefreshResult:
        stored_event = self._fire_event_repository.get_by_id(fire_event_id)
        if stored_event is None:
            raise ValueError(f"FireEvent {fire_event_id!r} was not found.")
        if stored_event.event.status in _INACTIVE_EVENT_STATUSES:
            return self._bare_result(PlanningRefreshStatus.INACTIVE_EVENT, fire_event_id)

        state_result = self._planning_state_builder.build(fire_event_id=fire_event_id, as_of=as_of)
        if state_result.status is PlanningEffectiveStateStatus.NO_CURRENT_TARGETS:
            return self._bare_result(PlanningRefreshStatus.INSUFFICIENT_DATA, fire_event_id)

        current_fingerprint = state_result.state.fingerprint

        latest_plan = self._response_plan_repository.get_latest_for_fire_event(fire_event_id)
        stored_sidecar = (
            self._response_plan_planning_state_repository.get_for_plan(latest_plan.id)
            if latest_plan is not None
            else None
        )

        if (
            latest_plan is not None
            and stored_sidecar is not None
            and stored_sidecar.planning_effective_state_fingerprint == current_fingerprint
        ):
            return self._handle_same_fingerprint(fire_event_id, latest_plan)

        return self._run_full_planning_cycle(fire_event_id, as_of, current_fingerprint)

    def _handle_same_fingerprint(
        self,
        fire_event_id: int,
        latest_plan: StoredResponsePlan,
    ) -> PlanningRefreshResult:
        route_planning_run_id = latest_plan.plan.route_planning_run_id
        existing_comparison = self._find_comparison_for_plan(fire_event_id, latest_plan.id)
        if existing_comparison is not None:
            return PlanningRefreshResult(
                status=PlanningRefreshStatus.NO_OP,
                fire_event_id=fire_event_id,
                route_planning_run_id=route_planning_run_id,
                response_plan_id=latest_plan.id,
                comparison_id=existing_comparison.id,
                error=None,
            )

        try:
            stored_comparison = self._baseline_collaborator.compare(response_plan_id=latest_plan.id)
        except Exception as exc:  # noqa: BLE001 - injected collaborator's documented failure mode.
            return PlanningRefreshResult(
                status=PlanningRefreshStatus.FAILED,
                fire_event_id=fire_event_id,
                route_planning_run_id=route_planning_run_id,
                response_plan_id=latest_plan.id,
                comparison_id=None,
                error=str(exc) or "Baseline comparison failed.",
            )
        return PlanningRefreshResult(
            status=PlanningRefreshStatus.REFRESHED,
            fire_event_id=fire_event_id,
            route_planning_run_id=route_planning_run_id,
            response_plan_id=latest_plan.id,
            comparison_id=stored_comparison.id,
            error=None,
        )

    def _run_full_planning_cycle(
        self,
        fire_event_id: int,
        as_of: datetime,
        current_fingerprint: str,
    ) -> PlanningRefreshResult:
        routing_result = self._routing_collaborator.plan(fire_event_id=fire_event_id, as_of=as_of)
        if not routing_result.success:
            return PlanningRefreshResult(
                status=PlanningRefreshStatus.FAILED,
                fire_event_id=fire_event_id,
                route_planning_run_id=routing_result.run_id,
                response_plan_id=None,
                comparison_id=None,
                error=routing_result.error_message or "Routing failed.",
            )
        route_planning_run_id = routing_result.run_id

        optimization_result = self._optimization_collaborator.optimize(
            route_planning_run_id=route_planning_run_id,
            as_of=as_of,
            seed=self._optimization_seed,
        )
        if not optimization_result.success:
            return PlanningRefreshResult(
                status=PlanningRefreshStatus.FAILED,
                fire_event_id=fire_event_id,
                route_planning_run_id=route_planning_run_id,
                response_plan_id=optimization_result.response_plan_id,
                comparison_id=None,
                error=optimization_result.error_message or "Optimization failed.",
            )
        response_plan_id = optimization_result.response_plan_id

        try:
            self._response_plan_planning_state_repository.save(
                response_plan_id=response_plan_id,
                planning_effective_state_fingerprint=current_fingerprint,
            )
        except Exception as exc:  # noqa: BLE001 - sidecar persistence's documented failure mode.
            return PlanningRefreshResult(
                status=PlanningRefreshStatus.FAILED,
                fire_event_id=fire_event_id,
                route_planning_run_id=route_planning_run_id,
                response_plan_id=response_plan_id,
                comparison_id=None,
                error=str(exc) or "Planning-state sidecar persistence failed.",
            )

        try:
            stored_comparison = self._baseline_collaborator.compare(response_plan_id=response_plan_id)
        except Exception as exc:  # noqa: BLE001 - injected collaborator's documented failure mode.
            return PlanningRefreshResult(
                status=PlanningRefreshStatus.FAILED,
                fire_event_id=fire_event_id,
                route_planning_run_id=route_planning_run_id,
                response_plan_id=response_plan_id,
                comparison_id=None,
                error=str(exc) or "Baseline comparison failed.",
            )

        return PlanningRefreshResult(
            status=PlanningRefreshStatus.REFRESHED,
            fire_event_id=fire_event_id,
            route_planning_run_id=route_planning_run_id,
            response_plan_id=response_plan_id,
            comparison_id=stored_comparison.id,
            error=None,
        )

    def _find_comparison_for_plan(self, fire_event_id: int, response_plan_id: int) -> StoredPlanComparison | None:
        # If more than one comparison exists for this exact plan (e.g. historical/manual
        # data), pick the highest/newest persisted id - the same deterministic
        # tie-break BaselineComparisonCollaboratorAdapter already uses (Task 6.1).
        matches = [
            stored_comparison
            for stored_comparison in self._plan_comparison_repository.list_for_fire_event(fire_event_id)
            if stored_comparison.comparison.optimized_plan_id == response_plan_id
        ]
        if not matches:
            return None
        return max(matches, key=lambda stored_comparison: stored_comparison.id)

    @staticmethod
    def _bare_result(status: PlanningRefreshStatus, fire_event_id: int) -> PlanningRefreshResult:
        return PlanningRefreshResult(
            status=status,
            fire_event_id=fire_event_id,
            route_planning_run_id=None,
            response_plan_id=None,
            comparison_id=None,
            error=None,
        )

    @staticmethod
    def _validate_request(fire_event_id: int, as_of: datetime) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
