"""GlobalPlanningOrchestrator: one audit boundary around all active
FireEvents' existing per-event planning (Stage 2 of the Global
Multi-Incident Optimizer refactor).

This is NOT global optimization. It orchestrates the existing, completely
unchanged ResponsePlanningRefreshOrchestrator sequentially, one active
FireEvent at a time, in a deterministic neutral order (fire_event_id ASC -
Task 9; NOT a severity/priority policy). It contains no routing, no GA, no
target-priority or resource-availability calculation of its own - every one
of those remains the child orchestrator's responsibility, exactly as
before this stage existed.

Flow (Task 8):
    capture active-event snapshot (once, Task 6)
        -> create GlobalPlanningRun(status=RUNNING) + one membership row
           per snapshot member (Task 3, atomic with the run itself)
        -> for each snapshot member, in order:
               invoke the existing child orchestrator
               record its outcome onto that member's row (Task 12)
               if it created a genuinely NEW ResponsePlan, stamp that
               plan with this run's id (Task 11) - never for NO_OP/
               baseline-only-recovery reuse of an existing plan
        -> finalize the run's aggregate status (Task 13)

Resource-commitment concurrency correctness is entirely Stage 1's: the
snapshot here is an orchestration/audit boundary, not a concurrency
boundary. Each child cycle re-validates current ResourceCommitments/status
against the live database at its own activation time (Task 17) -
sequential order can still influence WHICH resource an event ends up with
(this stage does not change that), but never lets two events end up
believing they own the same resource.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging

from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.planning_effective_state_status import PlanningEffectiveStateStatus
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository, StoredGlobalPlanningRun
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.services.global_planning.global_planning_config import METHODOLOGY, METHODOLOGY_VERSION
from src.services.global_planning.global_planning_result import GlobalPlanningEventResult, GlobalPlanningResult
from src.services.global_planning.global_planning_snapshot import GlobalPlanningSnapshot
from src.services.response_planning.planning_effective_state_builder import PlanningEffectiveStateBuilder
from src.services.response_planning.planning_refresh_result import PlanningRefreshStatus
from src.services.response_planning.response_planning_refresh_orchestrator import ResponsePlanningRefreshOrchestrator

logger = logging.getLogger(__name__)

# Task 12: a fixed, safe code recorded for every FAILED member - never the
# raw PlanningRefreshResult.error text (which may embed exception details).
CHILD_PLANNING_FAILED_ERROR_CODE = "child_planning_failed"

# PlanningRefreshStatus -> GlobalPlanningRunEventStatus. REFRESHED maps to
# PLANNED regardless of whether the child created a brand-new ResponsePlan
# or only added a missing baseline comparison to an already-current plan
# (a "baseline-only recovery") - that distinction is orthogonal to this
# status (see set_global_planning_run_id's own before/after-plan-id check
# in _run_one_member, which is what actually decides Task 11's stamping).
_RESULT_STATUS_MAP = {
    PlanningRefreshStatus.REFRESHED: GlobalPlanningRunEventStatus.PLANNED,
    PlanningRefreshStatus.NO_OP: GlobalPlanningRunEventStatus.NO_OP,
    PlanningRefreshStatus.INACTIVE_EVENT: GlobalPlanningRunEventStatus.SKIPPED_INACTIVE,
    PlanningRefreshStatus.INSUFFICIENT_DATA: GlobalPlanningRunEventStatus.INSUFFICIENT_DATA,
    PlanningRefreshStatus.FAILED: GlobalPlanningRunEventStatus.FAILED,
}


class GlobalPlanningOrchestrator:
    """Sequence the existing per-event refresh orchestrator across all active FireEvents, once, with audit bookkeeping."""

    def __init__(
        self,
        *,
        child_orchestrator: ResponsePlanningRefreshOrchestrator,
        fire_event_repository: FireEventRepository | None = None,
        resource_commitment_repository: ResourceCommitmentRepository | None = None,
        response_plan_repository: ResponsePlanRepository | None = None,
        planning_state_builder: PlanningEffectiveStateBuilder | None = None,
        global_planning_run_repository: GlobalPlanningRunRepository | None = None,
        methodology: str = METHODOLOGY,
        methodology_version: str = METHODOLOGY_VERSION,
    ) -> None:
        self._child_orchestrator = child_orchestrator
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._resource_commitment_repository = resource_commitment_repository or ResourceCommitmentRepository()
        self._response_plan_repository = response_plan_repository or ResponsePlanRepository()
        self._planning_state_builder = planning_state_builder or PlanningEffectiveStateBuilder()
        self._global_planning_run_repository = global_planning_run_repository or GlobalPlanningRunRepository()
        self._methodology = methodology
        self._methodology_version = methodology_version

    def run(self, *, as_of: datetime, trigger: str) -> GlobalPlanningResult:
        """Run one global planning cycle over every currently-active FireEvent."""
        self._validate_request(as_of, trigger)
        snapshot = self._capture_snapshot(as_of)

        stored_run = self._global_planning_run_repository.create_run(
            started_at=as_of,
            trigger=trigger,
            methodology=self._methodology,
            methodology_version=self._methodology_version,
            input_fingerprint=snapshot.fingerprint(),
            fire_event_ids=snapshot.active_fire_event_ids,
        )
        run_id = stored_run.id

        if not snapshot.active_fire_event_ids:
            completed = self._global_planning_run_repository.complete_run(
                run_id, status=GlobalPlanningRunStatus.NO_ACTIVE_EVENTS, completed_at=as_of
            )
            return self._to_global_result(completed, ())

        event_results = tuple(
            self._run_one_member(run_id, fire_event_id, as_of, snapshot)
            for fire_event_id in snapshot.active_fire_event_ids
        )

        final_status = self._determine_final_status(tuple(result.planning_status for result in event_results))
        completed = self._global_planning_run_repository.complete_run(
            run_id, status=final_status, completed_at=datetime.now(timezone.utc)
        )
        return self._to_global_result(completed, event_results)

    def _capture_snapshot(self, as_of: datetime) -> GlobalPlanningSnapshot:
        """Capture the RESPONSE-ELIGIBLE event set and its per-event/commitment context exactly once (Task 6).

        Task 9A: only CONFIRMED events are planned for; SUSPECTED events are active for monitoring but never
        consume routing / allocation / commitment work."""
        active_fire_event_ids = self._fire_event_repository.get_response_eligible_fire_event_ids()  # id-ASC

        fingerprints: dict[int, str | None] = {}
        for fire_event_id in active_fire_event_ids:
            fingerprints[fire_event_id] = self._safe_local_fingerprint(fire_event_id, as_of)

        commitments = self._resource_commitment_repository.get_for_fire_events(active_fire_event_ids)

        return GlobalPlanningSnapshot(
            as_of=as_of,
            active_fire_event_ids=active_fire_event_ids,
            resource_commitments=commitments,
            per_event_effective_state_fingerprints=fingerprints,
            methodology=self._methodology,
            methodology_version=self._methodology_version,
        )

    def _safe_local_fingerprint(self, fire_event_id: int, as_of: datetime) -> str | None:
        """Best-effort per-event fingerprint for the snapshot's own traceability payload.

        Audit-only (Task 7); a malformed single event's targets must never
        prevent the snapshot itself - and therefore the whole run - from
        being captured. Never raises.
        """
        try:
            state_result = self._planning_state_builder.build(fire_event_id=fire_event_id, as_of=as_of)
        except Exception:  # noqa: BLE001 - see docstring: audit-only, must not abort snapshot capture.
            logger.exception("Could not compute local effective-state fingerprint for FireEvent %s", fire_event_id)
            return None
        if state_result.status is not PlanningEffectiveStateStatus.BUILT:
            return None
        return state_result.state.fingerprint

    def _run_one_member(
        self,
        run_id: int,
        fire_event_id: int,
        as_of: datetime,
        snapshot: GlobalPlanningSnapshot,
    ) -> GlobalPlanningEventResult:
        before_plan = self._response_plan_repository.get_latest_for_fire_event(fire_event_id)
        before_plan_id = before_plan.id if before_plan is not None else None

        try:
            refresh_result = self._child_orchestrator.refresh(fire_event_id=fire_event_id, as_of=as_of)
        except Exception:  # noqa: BLE001 - refresh() already catches everything internally and
            # returns FAILED; this is a last-resort net so one member can never abort the
            # rest of the cycle (Task 14), even if that internal guarantee were ever broken.
            logger.exception("Unexpected exception from child refresh for FireEvent %s", fire_event_id)
            member_status = GlobalPlanningRunEventStatus.FAILED
            response_plan_id = None
            error_code = CHILD_PLANNING_FAILED_ERROR_CODE
        else:
            member_status = _RESULT_STATUS_MAP[refresh_result.status]
            response_plan_id = refresh_result.response_plan_id
            error_code = (
                CHILD_PLANNING_FAILED_ERROR_CODE if member_status is GlobalPlanningRunEventStatus.FAILED else None
            )

        if (
            member_status is GlobalPlanningRunEventStatus.PLANNED
            and response_plan_id is not None
            and response_plan_id != before_plan_id
        ):
            # A genuinely NEW ResponsePlan was created this cycle (Task 11) -
            # as opposed to REFRESHED-but-same-plan (a baseline-only recovery).
            self._response_plan_repository.set_global_planning_run_id(response_plan_id, run_id)

        self._global_planning_run_repository.record_member_result(
            run_id,
            fire_event_id,
            result_status=member_status,
            response_plan_id=response_plan_id,
            local_state_fingerprint=snapshot.per_event_effective_state_fingerprints.get(fire_event_id),
            error_code=error_code,
        )
        return GlobalPlanningEventResult(
            fire_event_id=fire_event_id,
            planning_status=member_status,
            response_plan_id=response_plan_id,
        )

    @staticmethod
    def _determine_final_status(
        member_statuses: tuple[GlobalPlanningRunEventStatus, ...],
    ) -> GlobalPlanningRunStatus:
        """Task 13's exact aggregation rule, kept as one small pure function so it can be unit-tested directly."""
        if not member_statuses:
            return GlobalPlanningRunStatus.NO_ACTIVE_EVENTS
        failed_count = sum(1 for status in member_statuses if status is GlobalPlanningRunEventStatus.FAILED)
        if failed_count == 0:
            return GlobalPlanningRunStatus.COMPLETED
        if failed_count == len(member_statuses):
            return GlobalPlanningRunStatus.FAILED
        return GlobalPlanningRunStatus.PARTIAL

    @staticmethod
    def _to_global_result(
        stored_run: StoredGlobalPlanningRun,
        event_results: tuple[GlobalPlanningEventResult, ...],
    ) -> GlobalPlanningResult:
        return GlobalPlanningResult(
            global_planning_run_id=stored_run.id,
            status=stored_run.run.status,
            started_at=stored_run.run.started_at,
            completed_at=stored_run.run.completed_at,
            input_fingerprint=stored_run.run.input_fingerprint,
            event_results=event_results,
        )

    @staticmethod
    def _validate_request(as_of: object, trigger: object) -> None:
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
        if not isinstance(trigger, str) or not trigger.strip():
            raise ValueError(f"trigger must be a non-empty string, got {trigger!r}")
