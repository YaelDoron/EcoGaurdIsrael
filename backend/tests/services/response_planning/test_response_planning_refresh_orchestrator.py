"""Tests for ResponsePlanningRefreshOrchestrator (Epic 5, US 5.4, Task 5), using fakes only."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.analysis.response_optimization_result import ResponseOptimizationResult, ResponseOptimizationStatus
from src.agents.routing.route_planning_result import RoutePlanningResult, RoutePlanningStatus
from src.calculators.baseline_plan.baseline_plan_comparison_calculator import PlanComparison
from src.models import (
    PlanningEffectiveState,
    PlanningEffectiveStateResult,
    PlanningEffectiveStateStatus,
    PlanningResourceState,
    PlanningTargetState,
    ResponseAction,
    ResponsePlan,
    ResponsePlanStatus,
    ResponseTargetType,
)
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.routing import RouteResult, RoutePlanningRun, RouteStatus, StoredRouteResult
from src.repositories.fire_event_repository import StoredFireEvent
from src.repositories.plan_comparison_repository import StoredPlanComparison
from src.repositories.response_plan_planning_state_repository import StoredResponsePlanPlanningState
from src.repositories.response_plan_repository import StoredResponsePlan
from src.repositories.route_planning_repository import StoredRoutePlanningRun
from src.services.response_planning import (
    PlanningRefreshResult,
    PlanningRefreshStatus,
    ResponsePlanningRefreshOrchestrator,
)

FIRE_EVENT_ID = 42
OTHER_FIRE_EVENT_ID = 99
AS_OF = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeFireEventRepository:
    def __init__(self, stored_event=None):
        self.stored_event = stored_event
        self.calls = []

    def get_by_id(self, fire_event_id):
        self.calls.append(fire_event_id)
        return self.stored_event


class FakePlanningStateBuilder:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def build(self, *, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        return self.result


class FakeResponsePlanRepository:
    def __init__(self, latest_plan=None):
        self.latest_plan = latest_plan
        self.calls = []

    def get_latest_for_fire_event(self, fire_event_id):
        self.calls.append(fire_event_id)
        return self.latest_plan


class FakeResponsePlanPlanningStateRepository:
    def __init__(self, sidecar=None, save_exc=None, call_log=None):
        self.sidecar = sidecar
        self.save_exc = save_exc
        self.get_calls = []
        self.save_calls = []
        self._call_log = call_log

    def get_for_plan(self, response_plan_id):
        self.get_calls.append(response_plan_id)
        return self.sidecar

    def save(self, *, response_plan_id, planning_effective_state_fingerprint):
        self.save_calls.append(
            {
                "response_plan_id": response_plan_id,
                "planning_effective_state_fingerprint": planning_effective_state_fingerprint,
            }
        )
        if self._call_log is not None:
            self._call_log.append("sidecar")
        if self.save_exc is not None:
            raise self.save_exc
        return StoredResponsePlanPlanningState(
            id=1,
            response_plan_id=response_plan_id,
            planning_effective_state_fingerprint=planning_effective_state_fingerprint,
            created_at=AS_OF,
        )


class FakePlanComparisonRepository:
    def __init__(self, comparisons=()):
        self.comparisons = comparisons
        self.calls = []

    def list_for_fire_event(self, fire_event_id):
        self.calls.append(fire_event_id)
        return self.comparisons


class FakeRoutingCollaborator:
    def __init__(self, result, call_log=None):
        self.result = result
        self.calls = []
        self._call_log = call_log

    def plan(self, *, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        if self._call_log is not None:
            self._call_log.append("routing")
        return self.result


class FakeOptimizationCollaborator:
    def __init__(self, result, call_log=None):
        self.result = result
        self.calls = []
        self._call_log = call_log

    def optimize(self, *, route_planning_run_id, as_of, seed):
        self.calls.append({"route_planning_run_id": route_planning_run_id, "as_of": as_of, "seed": seed})
        if self._call_log is not None:
            self._call_log.append("optimization")
        return self.result


class FakeBaselineCollaborator:
    def __init__(self, result=None, exc=None, call_log=None):
        self.result = result
        self.exc = exc
        self.calls = []
        self._call_log = call_log

    def compare(self, *, response_plan_id):
        self.calls.append(response_plan_id)
        if self._call_log is not None:
            self._call_log.append("baseline")
        if self.exc is not None:
            raise self.exc
        return self.result


# ---------------------------------------------------------------------------
# Domain object builders
# ---------------------------------------------------------------------------


def make_stored_event(status=FireEventStatus.CONFIRMED, fire_event_id=FIRE_EVENT_ID) -> StoredFireEvent:
    event = FireEvent(
        latitude=32.731,
        longitude=35.046,
        detected_at=AS_OF - timedelta(hours=1),
        updated_at=AS_OF - timedelta(minutes=5),
        status=status,
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    return StoredFireEvent(id=fire_event_id, event=event)


def make_planning_state(fire_event_id=FIRE_EVENT_ID, resources=None) -> PlanningEffectiveState:
    default_resources = (
        PlanningResourceState(
            resource_id="truck-1",
            station_id="station-1",
            station_latitude=32.7,
            station_longitude=35.0,
        ),
    )
    return PlanningEffectiveState(
        fire_event_id=fire_event_id,
        targets=(
            PlanningTargetState(
                target_type=ResponseTargetType.ACTIVE_FIRE,
                latitude=32.731,
                longitude=35.046,
                priority_score=150.0,
            ),
        ),
        resources=default_resources if resources is None else resources,
    )


def make_built_result(fire_event_id=FIRE_EVENT_ID, resources=None) -> PlanningEffectiveStateResult:
    return PlanningEffectiveStateResult(
        status=PlanningEffectiveStateStatus.BUILT,
        state=make_planning_state(fire_event_id=fire_event_id, resources=resources),
        fire_event_id=fire_event_id,
    )


def make_no_targets_result(fire_event_id=FIRE_EVENT_ID) -> PlanningEffectiveStateResult:
    return PlanningEffectiveStateResult(
        status=PlanningEffectiveStateStatus.NO_CURRENT_TARGETS,
        state=None,
        fire_event_id=fire_event_id,
    )


def make_stored_plan(plan_id=501, route_planning_run_id=601, fire_event_id=FIRE_EVENT_ID) -> StoredResponsePlan:
    plan = ResponsePlan(
        fire_event_id=fire_event_id,
        response_target_set_id=1,
        route_planning_run_id=route_planning_run_id,
        generated_at=AS_OF - timedelta(minutes=10),
        status=ResponsePlanStatus.COMPLETE,
        methodology="GENETIC_RESOURCE_ALLOCATION",
        methodology_version="1.0",
        random_seed=42,
        actions=(ResponseAction("truck-1", 10, 100),),
        uncovered_target_ids=(),
        plan_score=90.0,
        coverage_score=100.0,
        average_eta_seconds=200.0,
    )
    return StoredResponsePlan(id=plan_id, plan=plan)


def make_sidecar(plan_id=501, fingerprint=FINGERPRINT_A) -> StoredResponsePlanPlanningState:
    return StoredResponsePlanPlanningState(
        id=1,
        response_plan_id=plan_id,
        planning_effective_state_fingerprint=fingerprint,
        created_at=AS_OF,
    )


def make_comparison(comparison_id=701, plan_id=501, fire_event_id=FIRE_EVENT_ID) -> StoredPlanComparison:
    comparison = PlanComparison(
        fire_event_id=fire_event_id,
        optimized_plan_id=plan_id,
        route_planning_run_id=601,
        response_target_set_id=1,
        optimized_score=850.0,
        baseline_score=700.0,
        optimized_coverage_score=90.0,
        baseline_coverage_score=80.0,
        optimized_average_eta_seconds=300.0,
        baseline_average_eta_seconds=340.0,
        score_difference=150.0,
        improvement_percentage=21.4,
    )
    return StoredPlanComparison(id=comparison_id, comparison=comparison)


def make_stored_routing_run(run_id=601) -> StoredRoutePlanningRun:
    run = RoutePlanningRun(
        fire_event_id=FIRE_EVENT_ID,
        response_target_set_id=1,
        planned_at=AS_OF,
        methodology="ECOGUARD_ROUTING_DIJKSTRA",
        methodology_version="1.0",
        resource_ids=("truck-1",),
        routes=(
            RouteResult(
                resource_id="truck-1",
                response_target_id=10,
                status=RouteStatus.REACHABLE,
                source_node_id=1,
                target_node_id=2,
                node_path=(1, 2),
                distance_meters=1000.0,
                travel_time_seconds=90.0,
            ),
        ),
    )
    stored_routes = (StoredRouteResult(id=1, route_result=run.routes[0]),)
    return StoredRoutePlanningRun(id=run_id, run=run, routes=stored_routes)


def make_routing_success(run_id=601) -> RoutePlanningResult:
    return RoutePlanningResult(
        success=True,
        fire_event_id=FIRE_EVENT_ID,
        status=RoutePlanningStatus.PLANNED,
        run_id=run_id,
        run=make_stored_routing_run(run_id),
        route_count=1,
        error_message=None,
    )


def make_routing_failure(error_message="Routing failed.") -> RoutePlanningResult:
    return RoutePlanningResult(
        success=False,
        fire_event_id=FIRE_EVENT_ID,
        status=RoutePlanningStatus.FAILED,
        run_id=None,
        run=None,
        route_count=0,
        error_message=error_message,
    )


def make_optimization_success(plan_id=501) -> ResponseOptimizationResult:
    return ResponseOptimizationResult(
        success=True,
        fire_event_id=FIRE_EVENT_ID,
        response_target_set_id=1,
        route_planning_run_id=601,
        status=ResponseOptimizationStatus.OPTIMIZED,
        plan_status=ResponsePlanStatus.COMPLETE,
        response_plan_id=plan_id,
        action_count=1,
        uncovered_target_count=0,
        plan_score=90.0,
        coverage_score=100.0,
        average_eta_seconds=200.0,
        error_message=None,
    )


def make_optimization_failure(error_message="Optimization failed.") -> ResponseOptimizationResult:
    return ResponseOptimizationResult(
        success=False,
        fire_event_id=FIRE_EVENT_ID,
        response_target_set_id=1,
        route_planning_run_id=601,
        status=ResponseOptimizationStatus.FAILED,
        plan_status=None,
        response_plan_id=None,
        action_count=0,
        uncovered_target_count=0,
        error_message=error_message,
    )


def make_orchestrator(
    *,
    fire_event_repository=None,
    planning_state_builder=None,
    response_plan_repository=None,
    response_plan_planning_state_repository=None,
    plan_comparison_repository=None,
    routing_collaborator=None,
    optimization_collaborator=None,
    baseline_collaborator=None,
) -> ResponsePlanningRefreshOrchestrator:
    return ResponsePlanningRefreshOrchestrator(
        planning_state_builder=planning_state_builder or FakePlanningStateBuilder(make_built_result()),
        routing_collaborator=routing_collaborator or FakeRoutingCollaborator(make_routing_success()),
        optimization_collaborator=optimization_collaborator
        or FakeOptimizationCollaborator(make_optimization_success()),
        baseline_collaborator=baseline_collaborator or FakeBaselineCollaborator(make_comparison()),
        fire_event_repository=fire_event_repository or FakeFireEventRepository(make_stored_event()),
        response_plan_repository=response_plan_repository or FakeResponsePlanRepository(None),
        response_plan_planning_state_repository=(
            response_plan_planning_state_repository or FakeResponsePlanPlanningStateRepository(None)
        ),
        plan_comparison_repository=plan_comparison_repository or FakePlanComparisonRepository(()),
    )


# ---------------------------------------------------------------------------
# 1-3. FireEvent gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_inactive_event_returns_inactive_event_status(status):
    result = make_orchestrator(
        fire_event_repository=FakeFireEventRepository(make_stored_event(status=status))
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.INACTIVE_EVENT
    assert result.route_planning_run_id is None
    assert result.response_plan_id is None
    assert result.comparison_id is None
    assert result.error is None


def test_inactive_event_does_not_build_planning_state():
    planning_state_builder = FakePlanningStateBuilder(make_built_result())

    make_orchestrator(
        fire_event_repository=FakeFireEventRepository(make_stored_event(status=FireEventStatus.RESOLVED)),
        planning_state_builder=planning_state_builder,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert planning_state_builder.calls == []


# ---------------------------------------------------------------------------
# 4-5. Planning-state sufficiency
# ---------------------------------------------------------------------------


def test_no_current_targets_returns_insufficient_data():
    result = make_orchestrator(
        planning_state_builder=FakePlanningStateBuilder(make_no_targets_result())
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.INSUFFICIENT_DATA
    assert result.route_planning_run_id is None
    assert result.response_plan_id is None
    assert result.comparison_id is None


def test_zero_resources_still_proceeds_to_full_planning_cycle():
    result = make_orchestrator(
        planning_state_builder=FakePlanningStateBuilder(make_built_result(resources=())),
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.REFRESHED


# ---------------------------------------------------------------------------
# 6-8. Missing/changed fingerprint -> full planning cycle
# ---------------------------------------------------------------------------


def test_no_prior_plan_triggers_full_planning_cycle():
    routing_collaborator = FakeRoutingCollaborator(make_routing_success())
    optimization_collaborator = FakeOptimizationCollaborator(make_optimization_success())
    baseline_collaborator = FakeBaselineCollaborator(make_comparison())

    result = make_orchestrator(
        response_plan_repository=FakeResponsePlanRepository(None),
        routing_collaborator=routing_collaborator,
        optimization_collaborator=optimization_collaborator,
        baseline_collaborator=baseline_collaborator,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.REFRESHED
    assert len(routing_collaborator.calls) == 1
    assert len(optimization_collaborator.calls) == 1
    assert len(baseline_collaborator.calls) == 1


def test_prior_plan_without_sidecar_triggers_full_planning_cycle():
    routing_collaborator = FakeRoutingCollaborator(make_routing_success())

    result = make_orchestrator(
        response_plan_repository=FakeResponsePlanRepository(make_stored_plan()),
        response_plan_planning_state_repository=FakeResponsePlanPlanningStateRepository(None),
        routing_collaborator=routing_collaborator,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.REFRESHED
    assert len(routing_collaborator.calls) == 1


def test_different_fingerprint_triggers_full_planning_cycle():
    routing_collaborator = FakeRoutingCollaborator(make_routing_success())

    result = make_orchestrator(
        response_plan_repository=FakeResponsePlanRepository(make_stored_plan()),
        response_plan_planning_state_repository=FakeResponsePlanPlanningStateRepository(
            make_sidecar(fingerprint=FINGERPRINT_B)
        ),
        routing_collaborator=routing_collaborator,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.REFRESHED
    assert len(routing_collaborator.calls) == 1


# ---------------------------------------------------------------------------
# 9-13. Same fingerprint + comparison exists -> true NO_OP
# ---------------------------------------------------------------------------


def _same_fingerprint_orchestrator(**overrides):
    stored_plan = make_stored_plan()
    fingerprint = make_built_result().state.fingerprint
    defaults = dict(
        response_plan_repository=FakeResponsePlanRepository(stored_plan),
        response_plan_planning_state_repository=FakeResponsePlanPlanningStateRepository(
            make_sidecar(plan_id=stored_plan.id, fingerprint=fingerprint)
        ),
        plan_comparison_repository=FakePlanComparisonRepository((make_comparison(plan_id=stored_plan.id),)),
    )
    defaults.update(overrides)
    return make_orchestrator(**defaults)


def test_same_fingerprint_with_existing_comparison_is_true_no_op():
    stored_plan = make_stored_plan()
    result = _same_fingerprint_orchestrator().refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.NO_OP
    assert result.route_planning_run_id == stored_plan.plan.route_planning_run_id
    assert result.response_plan_id == stored_plan.id
    assert result.comparison_id == 701
    assert result.error is None


def test_no_op_picks_highest_id_among_duplicate_comparisons_for_the_same_plan():
    """Task 7 regression: multiple PlanComparison rows for one plan must not pick
    an arbitrary/oldest match - this mirrors BaselineComparisonCollaboratorAdapter's
    own newest-id tie-break (Task 6.1) so both duplicate-resolution paths agree."""
    stored_plan = make_stored_plan()
    result = _same_fingerprint_orchestrator(
        plan_comparison_repository=FakePlanComparisonRepository(
            (
                make_comparison(comparison_id=701, plan_id=stored_plan.id),
                make_comparison(comparison_id=705, plan_id=stored_plan.id),
                make_comparison(comparison_id=703, plan_id=stored_plan.id),
            )
        )
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.NO_OP
    assert result.comparison_id == 705


def test_true_no_op_invokes_no_routing():
    routing_collaborator = FakeRoutingCollaborator(make_routing_success())
    _same_fingerprint_orchestrator(routing_collaborator=routing_collaborator).refresh(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )
    assert routing_collaborator.calls == []


def test_true_no_op_invokes_no_optimization():
    optimization_collaborator = FakeOptimizationCollaborator(make_optimization_success())
    _same_fingerprint_orchestrator(optimization_collaborator=optimization_collaborator).refresh(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )
    assert optimization_collaborator.calls == []


def test_true_no_op_invokes_no_baseline():
    baseline_collaborator = FakeBaselineCollaborator(make_comparison())
    _same_fingerprint_orchestrator(baseline_collaborator=baseline_collaborator).refresh(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )
    assert baseline_collaborator.calls == []


def test_true_no_op_creates_no_sidecar():
    sidecar_repository = FakeResponsePlanPlanningStateRepository(
        make_sidecar(plan_id=501, fingerprint=make_built_result().state.fingerprint)
    )
    _same_fingerprint_orchestrator(response_plan_planning_state_repository=sidecar_repository).refresh(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )
    assert sidecar_repository.save_calls == []


# ---------------------------------------------------------------------------
# 14-18. Same fingerprint + comparison missing -> baseline-only recovery
# ---------------------------------------------------------------------------


def _baseline_only_orchestrator(**overrides):
    stored_plan = make_stored_plan()
    fingerprint = make_built_result().state.fingerprint
    defaults = dict(
        response_plan_repository=FakeResponsePlanRepository(stored_plan),
        response_plan_planning_state_repository=FakeResponsePlanPlanningStateRepository(
            make_sidecar(plan_id=stored_plan.id, fingerprint=fingerprint)
        ),
        plan_comparison_repository=FakePlanComparisonRepository(()),
    )
    defaults.update(overrides)
    return make_orchestrator(**defaults)


def test_same_fingerprint_with_missing_comparison_retries_baseline_only():
    stored_plan = make_stored_plan()
    routing_collaborator = FakeRoutingCollaborator(make_routing_success())
    optimization_collaborator = FakeOptimizationCollaborator(make_optimization_success())
    baseline_collaborator = FakeBaselineCollaborator(make_comparison(comparison_id=801, plan_id=stored_plan.id))

    result = _baseline_only_orchestrator(
        routing_collaborator=routing_collaborator,
        optimization_collaborator=optimization_collaborator,
        baseline_collaborator=baseline_collaborator,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.REFRESHED
    assert routing_collaborator.calls == []
    assert optimization_collaborator.calls == []
    assert baseline_collaborator.calls == [stored_plan.id]


def test_baseline_only_recovery_success_preserves_old_run_and_plan_ids():
    stored_plan = make_stored_plan(plan_id=501, route_planning_run_id=601)
    baseline_collaborator = FakeBaselineCollaborator(make_comparison(comparison_id=802, plan_id=501))

    result = _baseline_only_orchestrator(baseline_collaborator=baseline_collaborator).refresh(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    assert result.status is PlanningRefreshStatus.REFRESHED
    assert result.route_planning_run_id == 601
    assert result.response_plan_id == 501
    assert result.comparison_id == 802


def test_baseline_only_recovery_failure_returns_failed_and_preserves_plan():
    baseline_collaborator = FakeBaselineCollaborator(exc=RuntimeError("baseline exploded"))
    sidecar_repository = FakeResponsePlanPlanningStateRepository(
        make_sidecar(plan_id=501, fingerprint=make_built_result().state.fingerprint)
    )

    result = _baseline_only_orchestrator(
        baseline_collaborator=baseline_collaborator,
        response_plan_planning_state_repository=sidecar_repository,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.FAILED
    assert result.route_planning_run_id == 601
    assert result.response_plan_id == 501
    assert result.comparison_id is None
    assert result.error
    assert sidecar_repository.save_calls == []


# ---------------------------------------------------------------------------
# 19-22. Full-cycle failure handling
# ---------------------------------------------------------------------------


def test_routing_failure_returns_failed_without_optimization_sidecar_or_baseline():
    optimization_collaborator = FakeOptimizationCollaborator(make_optimization_success())
    sidecar_repository = FakeResponsePlanPlanningStateRepository(None)
    baseline_collaborator = FakeBaselineCollaborator(make_comparison())

    result = make_orchestrator(
        routing_collaborator=FakeRoutingCollaborator(make_routing_failure("no resources")),
        optimization_collaborator=optimization_collaborator,
        response_plan_planning_state_repository=sidecar_repository,
        baseline_collaborator=baseline_collaborator,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.FAILED
    assert result.route_planning_run_id is None
    assert result.response_plan_id is None
    assert result.comparison_id is None
    assert result.error == "no resources"
    assert optimization_collaborator.calls == []
    assert sidecar_repository.save_calls == []
    assert baseline_collaborator.calls == []


def test_optimization_failure_returns_failed_and_preserves_routing_run_id():
    sidecar_repository = FakeResponsePlanPlanningStateRepository(None)
    baseline_collaborator = FakeBaselineCollaborator(make_comparison())

    result = make_orchestrator(
        routing_collaborator=FakeRoutingCollaborator(make_routing_success(run_id=601)),
        optimization_collaborator=FakeOptimizationCollaborator(make_optimization_failure("GA blew up")),
        response_plan_planning_state_repository=sidecar_repository,
        baseline_collaborator=baseline_collaborator,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.FAILED
    assert result.route_planning_run_id == 601
    assert result.response_plan_id is None
    assert result.comparison_id is None
    assert result.error == "GA blew up"
    assert sidecar_repository.save_calls == []
    assert baseline_collaborator.calls == []


def test_sidecar_save_failure_returns_failed_without_baseline():
    baseline_collaborator = FakeBaselineCollaborator(make_comparison())

    result = make_orchestrator(
        routing_collaborator=FakeRoutingCollaborator(make_routing_success(run_id=601)),
        optimization_collaborator=FakeOptimizationCollaborator(make_optimization_success(plan_id=501)),
        response_plan_planning_state_repository=FakeResponsePlanPlanningStateRepository(
            None, save_exc=RuntimeError("sidecar write failed")
        ),
        baseline_collaborator=baseline_collaborator,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.FAILED
    assert result.route_planning_run_id == 601
    assert result.response_plan_id == 501
    assert result.comparison_id is None
    assert result.error
    assert baseline_collaborator.calls == []


def test_baseline_failure_after_new_plan_retains_new_run_and_plan_ids():
    result = make_orchestrator(
        routing_collaborator=FakeRoutingCollaborator(make_routing_success(run_id=601)),
        optimization_collaborator=FakeOptimizationCollaborator(make_optimization_success(plan_id=501)),
        baseline_collaborator=FakeBaselineCollaborator(exc=RuntimeError("baseline exploded")),
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.FAILED
    assert result.route_planning_run_id == 601
    assert result.response_plan_id == 501
    assert result.comparison_id is None
    assert result.error


# ---------------------------------------------------------------------------
# 23-26. Full success sequence, exact values passed downstream
# ---------------------------------------------------------------------------


def test_full_success_sequence_returns_refreshed_in_correct_order():
    call_log: list[str] = []
    routing_collaborator = FakeRoutingCollaborator(make_routing_success(run_id=601), call_log=call_log)
    optimization_collaborator = FakeOptimizationCollaborator(
        make_optimization_success(plan_id=501), call_log=call_log
    )
    sidecar_repository = FakeResponsePlanPlanningStateRepository(None, call_log=call_log)
    baseline_collaborator = FakeBaselineCollaborator(make_comparison(comparison_id=701), call_log=call_log)

    result = make_orchestrator(
        routing_collaborator=routing_collaborator,
        optimization_collaborator=optimization_collaborator,
        response_plan_planning_state_repository=sidecar_repository,
        baseline_collaborator=baseline_collaborator,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result == PlanningRefreshResult(
        status=PlanningRefreshStatus.REFRESHED,
        fire_event_id=FIRE_EVENT_ID,
        route_planning_run_id=601,
        response_plan_id=501,
        comparison_id=701,
        error=None,
    )
    assert call_log == ["routing", "optimization", "sidecar", "baseline"]


def test_sidecar_receives_exact_current_fingerprint():
    planning_state = make_planning_state()
    sidecar_repository = FakeResponsePlanPlanningStateRepository(None)

    make_orchestrator(
        planning_state_builder=FakePlanningStateBuilder(
            PlanningEffectiveStateResult(
                status=PlanningEffectiveStateStatus.BUILT, state=planning_state, fire_event_id=FIRE_EVENT_ID
            )
        ),
        response_plan_planning_state_repository=sidecar_repository,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert sidecar_repository.save_calls[0]["planning_effective_state_fingerprint"] == planning_state.fingerprint


def test_sidecar_receives_exact_response_plan_id_from_optimizer():
    sidecar_repository = FakeResponsePlanPlanningStateRepository(None)

    make_orchestrator(
        optimization_collaborator=FakeOptimizationCollaborator(make_optimization_success(plan_id=999)),
        response_plan_planning_state_repository=sidecar_repository,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert sidecar_repository.save_calls[0]["response_plan_id"] == 999


def test_baseline_receives_exact_same_response_plan_id():
    baseline_collaborator = FakeBaselineCollaborator(make_comparison(plan_id=999))

    make_orchestrator(
        optimization_collaborator=FakeOptimizationCollaborator(make_optimization_success(plan_id=999)),
        baseline_collaborator=baseline_collaborator,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert baseline_collaborator.calls == [999]


# ---------------------------------------------------------------------------
# 27-29. Isolation, immutability, zero artifacts on true NO_OP
# ---------------------------------------------------------------------------


def test_event_a_does_not_use_event_b_latest_plan_or_comparison():
    response_plan_repository = FakeResponsePlanRepository(make_stored_plan(fire_event_id=OTHER_FIRE_EVENT_ID))

    make_orchestrator(response_plan_repository=response_plan_repository).refresh(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    assert response_plan_repository.calls == [FIRE_EVENT_ID]


def test_historical_plan_and_sidecar_are_never_mutated_on_true_no_op():
    sidecar_repository = FakeResponsePlanPlanningStateRepository(
        make_sidecar(plan_id=501, fingerprint=make_built_result().state.fingerprint)
    )
    for forbidden in ("update", "overwrite", "delete", "upsert"):
        assert not hasattr(sidecar_repository, forbidden)

    _same_fingerprint_orchestrator(response_plan_planning_state_repository=sidecar_repository).refresh(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    assert sidecar_repository.save_calls == []


def test_true_no_op_produces_zero_new_planning_artifacts():
    routing_collaborator = FakeRoutingCollaborator(make_routing_success())
    optimization_collaborator = FakeOptimizationCollaborator(make_optimization_success())
    sidecar_repository = FakeResponsePlanPlanningStateRepository(
        make_sidecar(plan_id=501, fingerprint=make_built_result().state.fingerprint)
    )
    baseline_collaborator = FakeBaselineCollaborator(make_comparison())

    _same_fingerprint_orchestrator(
        routing_collaborator=routing_collaborator,
        optimization_collaborator=optimization_collaborator,
        response_plan_planning_state_repository=sidecar_repository,
        baseline_collaborator=baseline_collaborator,
    ).refresh(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert routing_collaborator.calls == []
    assert optimization_collaborator.calls == []
    assert sidecar_repository.save_calls == []
    assert baseline_collaborator.calls == []


# ---------------------------------------------------------------------------
# Missing FireEvent
# ---------------------------------------------------------------------------


def test_missing_fire_event_returns_failed():
    result = make_orchestrator(fire_event_repository=FakeFireEventRepository(None)).refresh(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    assert result.status is PlanningRefreshStatus.FAILED
    assert result.route_planning_run_id is None
    assert result.response_plan_id is None
    assert result.comparison_id is None
    assert result.error


# ---------------------------------------------------------------------------
# 30-31. Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_fire_event_id", [0, -1, True, "42"])
def test_invalid_fire_event_id_rejected(invalid_fire_event_id):
    with pytest.raises(ValueError):
        make_orchestrator().refresh(fire_event_id=invalid_fire_event_id, as_of=AS_OF)


def test_naive_as_of_rejected():
    with pytest.raises(ValueError):
        make_orchestrator().refresh(fire_event_id=FIRE_EVENT_ID, as_of=datetime(2026, 9, 16, 12, 0))


# ---------------------------------------------------------------------------
# PlanningRefreshResult validation
# ---------------------------------------------------------------------------


def test_result_rejects_failed_status_without_error():
    with pytest.raises(ValueError):
        PlanningRefreshResult(
            status=PlanningRefreshStatus.FAILED,
            fire_event_id=FIRE_EVENT_ID,
            route_planning_run_id=None,
            response_plan_id=None,
            comparison_id=None,
            error=None,
        )


def test_result_rejects_no_op_missing_ids():
    with pytest.raises(ValueError):
        PlanningRefreshResult(
            status=PlanningRefreshStatus.NO_OP,
            fire_event_id=FIRE_EVENT_ID,
            route_planning_run_id=601,
            response_plan_id=None,
            comparison_id=701,
            error=None,
        )


def test_result_rejects_inactive_event_with_ids():
    with pytest.raises(ValueError):
        PlanningRefreshResult(
            status=PlanningRefreshStatus.INACTIVE_EVENT,
            fire_event_id=FIRE_EVENT_ID,
            route_planning_run_id=601,
            response_plan_id=None,
            comparison_id=None,
            error=None,
        )
