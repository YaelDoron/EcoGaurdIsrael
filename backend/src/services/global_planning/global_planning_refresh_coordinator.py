"""GlobalPlanningRefreshCoordinator: the ONE production entry point for the
Global Multi-Incident Optimizer refactor's authoritative planning cycle
(Stage 6, Tasks 18-19, 33).

    material operational change
        -> capture ALL active FireEvents
        -> create GlobalPlanningRun
        -> GlobalPlanningInputBuilder -> stable GlobalPlanningInput
        -> global NO_OP check (input + policy fingerprint)
        -> GlobalResponseOptimizationService (Global GA)
        -> GlobalResponsePlanActivationService (atomic activation, with its
           own stale-input revalidation)
        -> finalize GlobalPlanningRun
        -> return one structured GlobalPlanningRefreshResult

No per-event optimizer invocation anywhere in this flow - ONE global
optimization problem, exactly the target architecture (Task 1).

Bounded retry (Task 33): GlobalPlanningInputUnstable or a stale-input
rejection from activation rebuilds the ENTIRE cycle (a fresh GlobalPlanningRun,
a fresh GlobalPlanningInput, a fresh GA run) up to MAX_GLOBAL_STALE_RETRIES
times - never a per-event patch, never manually edited chromosome. If
retries are exhausted, the previous authoritative generation's ResponsePlans/
commitments/dispatch locks remain fully intact and current; this cycle is
recorded as failed and STALE_RETRY_EXHAUSTED is returned.

Cheap NO_OP precheck (performance pass, "Implement the first
performance-optimization pass" task, Optimization 3): profiling showed
NO_OP cycles were paying for the full road-network load + route matrix +
Dijkstra before the existing input+policy fingerprint comparison below
ever got a chance to say "nothing changed." `_run_one_cycle` now asks
`GlobalPlanningInputBuilder.compute_pre_routing_bundle()` for a cheap
signature FIRST - everything the real fingerprint hashes except route
content (see that method's docstring for why omitting routes cannot
produce a false NO_OP) - and compares it against `_settled_pre_routing_signature`,
this COORDINATOR instance's own record of the last cycle that reached a
genuine terminal success (ACTIVATED, or NO_OP by either path). A match
skips build()/optimize()/activate() entirely. This precheck is instance-
scoped state (reset to None on a fresh coordinator, e.g. a new process/
simulation run) and is updated ONLY at the two points below marked
"settle" - never on GlobalPlanningInputUnstable, an activation failure, or
any other exception, so a failed attempt can never poison a future cheap
check into skipping planning it should not have.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from enum import Enum

from src.calculators.global_response_optimization.global_assignment_change_calculator import GlobalAssignmentChange
from src.calculators.global_response_optimization.global_assignment_stability_policy import (
    GlobalAssignmentStabilityPolicy,
)
from src.calculators.global_response_optimization.global_demand_scoring_policy import GlobalDemandScoringPolicy
from src.calculators.global_response_optimization.global_optimization_policy_fingerprint import (
    compute_optimization_policy_fingerprint,
)
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_service import (
    GlobalResponseOptimizationService,
)
from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy
from src.database.connection import get_session_factory
from src.models.global_event_optimization_result import GlobalEventOptimizationResult
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.global_resource_shortage import GlobalResourceShortage
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.services.global_planning.global_planning_input_builder import GlobalPlanningInputBuilder
from src.services.global_planning.global_planning_input_unstable import GlobalPlanningInputUnstable
from src.services.global_planning.global_response_plan_activation_service import (
    GlobalPlanningStaleInput,
    GlobalResponsePlanActivationService,
    GlobalRunHistoryContext,
)
from src.services.resource_reservation.resource_commitment_conflict import ResourceCommitmentConflict

MAX_GLOBAL_STALE_RETRIES = 1

_REFRESH_METHODOLOGY = "global_planning_refresh_coordinator"
_REFRESH_METHODOLOGY_VERSION = "1.0"


class GlobalPlanningRefreshStatus(Enum):
    """Outcome of one refresh() call - richer than GlobalPlanningRunStatus
    (Stage 2's own DB-level status), which has no NO_OP/STALE_RETRY_EXHAUSTED
    distinction of its own."""

    ACTIVATED = "activated"
    NO_OP = "no_op"
    NO_ACTIVE_EVENTS = "no_active_events"
    FAILED = "failed"
    STALE_RETRY_EXHAUSTED = "stale_retry_exhausted"


@dataclass(frozen=True)
class GlobalPlanningRefreshResult:
    """One structured, auditable outcome of a global refresh cycle."""

    status: GlobalPlanningRefreshStatus
    trigger: str
    as_of: datetime
    global_planning_run_id: int | None = None
    input_fingerprint: str | None = None
    optimization_policy_fingerprint: str | None = None
    response_plan_ids_by_event: dict[int, int] = field(default_factory=dict)
    assignment_changes: tuple[GlobalAssignmentChange, ...] = ()
    shortage: GlobalResourceShortage | None = None
    event_results: tuple[GlobalEventOptimizationResult, ...] = ()
    retry_count: int = 0


class GlobalPlanningRefreshCoordinator:
    """The single production orchestrator for the Global GA's authoritative planning cycle."""

    def __init__(
        self,
        fire_event_repository: FireEventRepository | None = None,
        global_planning_run_repository: GlobalPlanningRunRepository | None = None,
        input_builder: GlobalPlanningInputBuilder | None = None,
        optimization_service: GlobalResponseOptimizationService | None = None,
        activation_service: GlobalResponsePlanActivationService | None = None,
        config: GlobalResponseOptimizationConfig | None = None,
        demand_scoring_policy: GlobalDemandScoringPolicy | None = None,
        severity_demand_policy: SeverityDemandPolicy | None = None,
        stability_policy: GlobalAssignmentStabilityPolicy | None = None,
        session_factory=None,
    ) -> None:
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._global_planning_run_repository = global_planning_run_repository or GlobalPlanningRunRepository()
        self._input_builder = input_builder or GlobalPlanningInputBuilder()
        self._optimization_service = optimization_service or GlobalResponseOptimizationService()
        self._activation_service = activation_service or GlobalResponsePlanActivationService()
        self._config = config or GlobalResponseOptimizationConfig()
        self._demand_scoring_policy = demand_scoring_policy or GlobalDemandScoringPolicy()
        self._severity_demand_policy = severity_demand_policy or SeverityDemandPolicy()
        self._stability_policy = stability_policy or GlobalAssignmentStabilityPolicy()
        # Cheap NO_OP precheck settled state (Optimization 3) - instance-
        # scoped, starts empty so the very first cycle on a fresh
        # coordinator always fails open to a full build. Never touched
        # except at the two "settle" points in _run_one_cycle.
        self._settled_pre_routing_signature: str | None = None
        self._settled_input_fingerprint: str | None = None
        self._settled_combined_fingerprint: str | None = None

    def refresh(self, *, trigger: str, as_of: datetime) -> GlobalPlanningRefreshResult:
        self._validate_request(trigger, as_of)

        active_fire_event_ids = self._fire_event_repository.get_active_fire_event_ids()
        if not active_fire_event_ids:
            return GlobalPlanningRefreshResult(
                status=GlobalPlanningRefreshStatus.NO_ACTIVE_EVENTS, trigger=trigger, as_of=as_of
            )

        optimization_policy_fingerprint = compute_optimization_policy_fingerprint(
            self._config, self._demand_scoring_policy, self._severity_demand_policy, self._stability_policy
        )

        retry_count = 0
        for attempt in range(MAX_GLOBAL_STALE_RETRIES + 1):
            result = self._run_one_cycle(
                trigger, as_of, active_fire_event_ids, optimization_policy_fingerprint, retry_count
            )
            if result is not None:
                return result
            retry_count += 1
            if attempt == MAX_GLOBAL_STALE_RETRIES:
                return GlobalPlanningRefreshResult(
                    status=GlobalPlanningRefreshStatus.STALE_RETRY_EXHAUSTED,
                    trigger=trigger,
                    as_of=as_of,
                    retry_count=retry_count,
                )
            # Bounded retry (Task 33): the active set may itself have
            # changed between attempts (a new fire, a resolve) - re-capture
            # it fresh for the next full cycle rather than reusing the
            # stale snapshot.
            active_fire_event_ids = self._fire_event_repository.get_active_fire_event_ids()
            if not active_fire_event_ids:
                return GlobalPlanningRefreshResult(
                    status=GlobalPlanningRefreshStatus.NO_ACTIVE_EVENTS,
                    trigger=trigger,
                    as_of=as_of,
                    retry_count=retry_count,
                )

    def _run_one_cycle(
        self,
        trigger: str,
        as_of: datetime,
        active_fire_event_ids: tuple[int, ...],
        optimization_policy_fingerprint: str,
        retry_count: int,
    ) -> GlobalPlanningRefreshResult | None:
        """Returns a terminal GlobalPlanningRefreshResult, or None to signal
        "retry the whole cycle" (caller decides whether a retry remains)."""
        stored_run = self._global_planning_run_repository.create_run(
            started_at=as_of,
            trigger=trigger,
            methodology=_REFRESH_METHODOLOGY,
            methodology_version=_REFRESH_METHODOLOGY_VERSION,
            input_fingerprint=None,
            fire_event_ids=active_fire_event_ids,
        )

        # Safety net: a run row is persisted as RUNNING above, before any of
        # the steps below. Two exception types are an expected, retryable
        # outcome of stale input (handled inline, returning None so the
        # caller retries a fresh cycle); anything else - including a defect
        # or infrastructure failure inside input building itself, such as
        # the road-network bbox query - must still mark this run FAILED
        # before propagating, so it never stays stuck RUNNING forever. The
        # original exception is always re-raised, never swallowed.
        try:
            pre_routing_bundle = self._input_builder.compute_pre_routing_bundle(
                global_planning_run_id=stored_run.id, as_of=as_of
            )

            if (
                self._settled_pre_routing_signature is not None
                and self._settled_pre_routing_signature == pre_routing_bundle.pre_routing_signature
            ):
                # Cheap NO_OP (Optimization 3): never touches the road
                # network, route matrix, Dijkstra, or GA. Persist the last
                # SETTLED combined fingerprint forward (not a freshly
                # computed one - we deliberately never computed one this
                # cycle) so get_latest_activated() keeps returning a valid,
                # self-consistent value for any future cycle that falls
                # back to the full fingerprint comparison below.
                for fire_event_id in active_fire_event_ids:
                    self._global_planning_run_repository.record_member_result(
                        stored_run.id,
                        fire_event_id,
                        result_status=GlobalPlanningRunEventStatus.NO_OP,
                        response_plan_id=None,
                        local_state_fingerprint=pre_routing_bundle.pre_routing_signature,
                        error_code=None,
                    )
                if self._settled_combined_fingerprint is not None:
                    self._global_planning_run_repository.set_input_fingerprint(
                        stored_run.id, self._settled_combined_fingerprint
                    )
                self._global_planning_run_repository.complete_run(
                    stored_run.id, status=GlobalPlanningRunStatus.COMPLETED, completed_at=as_of
                )
                return GlobalPlanningRefreshResult(
                    status=GlobalPlanningRefreshStatus.NO_OP,
                    trigger=trigger,
                    as_of=as_of,
                    global_planning_run_id=stored_run.id,
                    input_fingerprint=self._settled_input_fingerprint,
                    optimization_policy_fingerprint=optimization_policy_fingerprint,
                    retry_count=retry_count,
                )

            global_planning_input = self._input_builder.build(
                global_planning_run_id=stored_run.id, as_of=as_of, precomputed=pre_routing_bundle
            )

            combined_fingerprint = self._combined_fingerprint(
                global_planning_input.input_fingerprint, optimization_policy_fingerprint
            )
            self._global_planning_run_repository.set_input_fingerprint(stored_run.id, combined_fingerprint)

            latest_activated_fingerprint = self._latest_activated_combined_fingerprint(exclude_run_id=stored_run.id)
            if latest_activated_fingerprint is not None and latest_activated_fingerprint == combined_fingerprint:
                for fire_event_id in active_fire_event_ids:
                    self._global_planning_run_repository.record_member_result(
                        stored_run.id,
                        fire_event_id,
                        result_status=GlobalPlanningRunEventStatus.NO_OP,
                        response_plan_id=None,
                        local_state_fingerprint=combined_fingerprint,
                        error_code=None,
                    )
                self._global_planning_run_repository.complete_run(
                    stored_run.id, status=GlobalPlanningRunStatus.COMPLETED, completed_at=as_of
                )
                # Settle (path 2 of 2): reached NO_OP via the full fingerprint
                # comparison (e.g. the cheap precheck above had no settled
                # baseline yet). Future cycles can now use the cheap path.
                self._settle(pre_routing_bundle, global_planning_input.input_fingerprint, combined_fingerprint)
                return GlobalPlanningRefreshResult(
                    status=GlobalPlanningRefreshStatus.NO_OP,
                    trigger=trigger,
                    as_of=as_of,
                    global_planning_run_id=stored_run.id,
                    input_fingerprint=global_planning_input.input_fingerprint,
                    optimization_policy_fingerprint=optimization_policy_fingerprint,
                    retry_count=retry_count,
                )

            ga_result = self._optimization_service.optimize(
                global_planning_input,
                self._config,
                self._demand_scoring_policy,
                self._severity_demand_policy,
                self._stability_policy,
            )

            # Final-closure atomicity fix: the required GlobalPlanningRunEvent
            # member results, the run's optimization metadata, and its final
            # COMPLETED/PARTIAL status are written by activate() itself, in the
            # SAME transaction as the ResponsePlan/ResourceCommitment writes -
            # never as separate, later, independently-committed calls. This
            # closes a crash window where activation could commit (making a
            # generation authoritative) while a later, separate write of its
            # required history never happens.
            run_history = GlobalRunHistoryContext(
                combined_fingerprint=combined_fingerprint,
                optimization_policy_fingerprint=optimization_policy_fingerprint,
                demand_scoring_policy_methodology=self._demand_scoring_policy.methodology,
                demand_scoring_policy_version=self._demand_scoring_policy.methodology_version,
                severity_demand_policy_methodology=self._severity_demand_policy.methodology,
                severity_demand_policy_version=self._severity_demand_policy.methodology_version,
                stability_policy_methodology=self._stability_policy.methodology,
                stability_policy_version=self._stability_policy.methodology_version,
            )
            try:
                activation = self._activation_service.activate(
                    global_planning_input=global_planning_input,
                    global_optimization_result=ga_result,
                    as_of=as_of,
                    run_history=run_history,
                )
            except (GlobalPlanningStaleInput, ResourceCommitmentConflict):
                self._global_planning_run_repository.complete_run(
                    stored_run.id, status=GlobalPlanningRunStatus.FAILED, completed_at=as_of
                )
                return None

            # Settle (path 1 of 2): a genuinely new plan was activated -
            # this cycle's inputs are now the current baseline.
            self._settle(pre_routing_bundle, global_planning_input.input_fingerprint, combined_fingerprint)
            return GlobalPlanningRefreshResult(
                status=GlobalPlanningRefreshStatus.ACTIVATED,
                trigger=trigger,
                as_of=as_of,
                global_planning_run_id=stored_run.id,
                input_fingerprint=global_planning_input.input_fingerprint,
                optimization_policy_fingerprint=optimization_policy_fingerprint,
                response_plan_ids_by_event=dict(activation.response_plan_ids_by_event),
                assignment_changes=ga_result.assignment_changes,
                shortage=ga_result.shortage,
                event_results=ga_result.event_results,
                retry_count=retry_count,
            )
        except GlobalPlanningInputUnstable:
            self._global_planning_run_repository.complete_run(
                stored_run.id, status=GlobalPlanningRunStatus.FAILED, completed_at=as_of
            )
            return None
        except Exception:
            self._global_planning_run_repository.complete_run(
                stored_run.id, status=GlobalPlanningRunStatus.FAILED, completed_at=as_of
            )
            raise

    def _latest_activated_combined_fingerprint(self, *, exclude_run_id: int) -> str | None:
        """The combined (input+policy) fingerprint of the most recently
        successfully-activated generation - reuses GlobalPlanningRunDB's
        existing `input_fingerprint` column to store the COMBINED value
        rather than adding a new column: both are 64-char SHA-256 hex
        digests, so this is a semantic reuse, not a schema change (Task 44:
        no schema migration expected for this piece)."""
        latest = self._global_planning_run_repository.get_latest_activated(exclude_run_id=exclude_run_id)
        if latest is None:
            return None
        return latest.run.input_fingerprint

    def _settle(
        self,
        pre_routing_bundle,
        input_fingerprint: str,
        combined_fingerprint: str,
    ) -> None:
        """Record this cycle's inputs as the new baseline for future cheap
        NO_OP prechecks (Optimization 3). Call ONLY from a genuinely
        successful terminal outcome (ACTIVATED or NO_OP) - never from a
        failure/exception path, or a failed attempt could establish a
        signature that was never actually planned against."""
        self._settled_pre_routing_signature = pre_routing_bundle.pre_routing_signature
        self._settled_input_fingerprint = input_fingerprint
        self._settled_combined_fingerprint = combined_fingerprint

    @staticmethod
    def _combined_fingerprint(input_fingerprint: str, optimization_policy_fingerprint: str) -> str:
        return hashlib.sha256(f"{input_fingerprint}|{optimization_policy_fingerprint}".encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_request(trigger: object, as_of: object) -> None:
        if not isinstance(trigger, str) or not trigger.strip():
            raise ValueError(f"trigger must be a non-empty string, got {trigger!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
