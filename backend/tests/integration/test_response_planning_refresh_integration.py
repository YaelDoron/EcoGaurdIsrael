"""End-to-end Task 6 integration tests: real repositories, real GA optimizer, real baseline
pipeline, against an in-memory SQLite database - no external IMS/NASA/Copernicus/OSM/network
calls.

Routing is the one piece stood in for: genuinely running Dijkstra needs a live/cached road
network, which is out of scope here. `PersistingFakeRoutingCollaborator` below skips Dijkstra
but performs REAL persistence via the real RoutePlanningRepository, producing an authentic
persisted RoutePlanningRun - so the actual Task 6 boundary under test (the optimization adapter
consuming the EXACT persisted routing snapshot) is exercised for real.

The GA/optimization and baseline-comparison steps are fully real: a real
ResponseOptimizationAgent (real GeneticResponsePlanOptimizer) and a real BaselineComparisonService
(real BaselinePlanCalculator/BaselinePlanEvaluator/BaselinePlanComparisonCalculator/
ResponsePlanScorer, reached through the real, production ResponsePlanBaselineScorerAdapter) both
run - as of Task 6.1, no test-only scorer bridge is needed any more.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.analysis.response_optimization_agent import ResponseOptimizationAgent
from src.agents.routing.route_planning_result import RoutePlanningResult, RoutePlanningStatus
from src.calculators.response_optimization.response_optimization_config import DEFAULT_RANDOM_SEED
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models import (
    ResponseTarget,
    ResponseTargetSet,
    ResponseTargetType,
    ResourceStatus,
)
from src.models.routing import RoutePlanningRun, RouteResult, RouteStatus
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.plan_comparison_repository import PlanComparisonRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.baseline_comparison.baseline_comparison_production_readers import (
    ResponsePlanOptimizedPlanReaderAdapter,
    RoutePlanningRunReaderAdapter,
)
from src.services.baseline_comparison.baseline_comparison_service import BaselineComparisonService
from src.services.baseline_comparison.response_plan_baseline_scorer_adapter import ResponsePlanBaselineScorerAdapter
from src.services.operational.operational_context_service import OperationalContextService
from src.services.response_planning.baseline_comparison_collaborator_adapter import (
    BaselineComparisonCollaboratorAdapter,
)
from src.services.response_planning.planning_effective_state_builder import PlanningEffectiveStateBuilder
from src.services.response_planning.planning_refresh_result import PlanningRefreshStatus
from src.services.response_planning.response_optimization_collaborator_adapter import (
    ResponseOptimizationCollaboratorAdapter,
)
from src.services.response_planning.response_planning_refresh_orchestrator import (
    ResponsePlanningRefreshOrchestrator,
)
from src.services.resource_reservation import ResponsePlanActivationService

AS_OF = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
LATER = AS_OF + timedelta(hours=1)


# ---------------------------------------------------------------------------
# Test-only collaborators
# ---------------------------------------------------------------------------


class PersistingFakeRoutingCollaborator:
    """Skips Dijkstra/OSM but performs REAL persistence via RoutePlanningRepository."""

    def __init__(self, response_target_repository, route_planning_repository, resource_ids):
        self._response_target_repository = response_target_repository
        self._route_planning_repository = route_planning_repository
        self._resource_ids = resource_ids
        self.calls = []

    def plan(self, *, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        stored_target_set = self._response_target_repository.get_latest_for_event_as_of(fire_event_id, as_of)
        active_target = next(
            t for t in stored_target_set.targets if t.target.target_type is ResponseTargetType.ACTIVE_FIRE
        )
        routes = tuple(
            RouteResult(
                resource_id=resource_id,
                response_target_id=active_target.id,
                status=RouteStatus.UNMAPPABLE,
                source_node_id=None,
                target_node_id=None,
                node_path=(),
                distance_meters=None,
                travel_time_seconds=None,
            )
            for resource_id in self._resource_ids
        )
        run = RoutePlanningRun(
            fire_event_id=fire_event_id,
            response_target_set_id=stored_target_set.id,
            planned_at=as_of,
            methodology="ECOGUARD_ROUTING_DIJKSTRA",
            methodology_version="1.0",
            resource_ids=self._resource_ids,
            routes=routes,
        )
        stored_run = self._route_planning_repository.save_run(run)
        return RoutePlanningResult(
            success=True,
            fire_event_id=fire_event_id,
            status=RoutePlanningStatus.PLANNED,
            run_id=stored_run.id,
            run=stored_run,
            route_count=len(stored_run.routes),
            error_message=None,
        )


class SpyRoutingCollaborator:
    def __init__(self, delegate):
        self._delegate = delegate
        self.calls = []

    def plan(self, *, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        return self._delegate.plan(fire_event_id=fire_event_id, as_of=as_of)


class SpyOptimizationCollaborator:
    def __init__(self, delegate):
        self._delegate = delegate
        self.calls = []

    def optimize(self, *, route_planning_run_id, as_of, seed):
        self.calls.append({"route_planning_run_id": route_planning_run_id, "as_of": as_of, "seed": seed})
        return self._delegate.optimize(route_planning_run_id=route_planning_run_id, as_of=as_of, seed=seed)


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def seed_fire_event(session_factory, status="confirmed") -> int:
    session = session_factory()
    event = FireEventDB(
        latitude=32.731,
        longitude=35.046,
        detected_at=AS_OF - timedelta(hours=1),
        updated_at=AS_OF - timedelta(minutes=5),
        status=status,
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.commit()
    fire_event_id = event.id
    session.close()
    return fire_event_id


def seed_station_and_resources(session_factory, resource_ids, station_id="station-1"):
    session = session_factory()
    station = FireStationDB(id=station_id, name="Station 1", latitude=32.7, longitude=35.0)
    session.add(station)
    session.flush()
    for resource_id in resource_ids:
        session.add(FirefightingResourceDB(id=resource_id, station_id=station_id, status=ResourceStatus.AVAILABLE))
    session.commit()
    session.close()


def seed_target_set(response_target_repository, fire_event_id, as_of=AS_OF, priority_score=150.0):
    return response_target_repository.save_target_set(
        ResponseTargetSet(
            fire_event_id=fire_event_id,
            generated_at=as_of,
            methodology="TEST_TARGETS",
            methodology_version="1.0",
            targets=(
                ResponseTarget(
                    fire_event_id=fire_event_id,
                    target_type=ResponseTargetType.ACTIVE_FIRE,
                    latitude=32.731,
                    longitude=35.046,
                    priority_score=priority_score,
                ),
            ),
        )
    )


class Wiring:
    """Bundles every repository + collaborator needed for one real orchestrator instance."""

    def __init__(self, session_factory, resource_ids=("truck-1",)):
        self.response_target_repository = ResponseTargetRepository(session_factory)
        self.route_planning_repository = RoutePlanningRepository(session_factory)
        self.response_plan_repository = ResponsePlanRepository(session_factory)
        self.plan_comparison_repository = PlanComparisonRepository(session_factory)
        self.sidecar_repository = ResponsePlanPlanningStateRepository(session_factory)
        self.fire_event_repository = FireEventRepository(session_factory)
        self.resource_commitment_repository = ResourceCommitmentRepository(session_factory)
        self.activation_service = ResponsePlanActivationService(
            response_plan_repository=self.response_plan_repository,
            resource_commitment_repository=self.resource_commitment_repository,
            session_factory=session_factory,
        )

        self.routing_collaborator = PersistingFakeRoutingCollaborator(
            self.response_target_repository, self.route_planning_repository, resource_ids
        )
        self.optimization_collaborator = ResponseOptimizationCollaboratorAdapter(
            optimization_agent=ResponseOptimizationAgent(repository=self.response_plan_repository),
            route_planning_repository=self.route_planning_repository,
            response_target_repository=self.response_target_repository,
        )

    def baseline_collaborator(self, real_scorer):
        baseline_service = BaselineComparisonService(
            optimized_plan_reader=ResponsePlanOptimizedPlanReaderAdapter(self.response_plan_repository),
            route_planning_run_reader=RoutePlanningRunReaderAdapter(self.route_planning_repository),
            scorer=ResponsePlanBaselineScorerAdapter(real_scorer),
            response_target_set_reader=self.response_target_repository,
            plan_comparison_repository=self.plan_comparison_repository,
        )
        return BaselineComparisonCollaboratorAdapter(baseline_service, self.plan_comparison_repository)

    def orchestrator(self, operational_context_service, baseline_collaborator) -> ResponsePlanningRefreshOrchestrator:
        return ResponsePlanningRefreshOrchestrator(
            planning_state_builder=PlanningEffectiveStateBuilder(
                response_target_repository=self.response_target_repository,
                operational_context_service=operational_context_service,
            ),
            routing_collaborator=self.routing_collaborator,
            optimization_collaborator=self.optimization_collaborator,
            baseline_collaborator=baseline_collaborator,
            fire_event_repository=self.fire_event_repository,
            response_plan_repository=self.response_plan_repository,
            response_plan_planning_state_repository=self.sidecar_repository,
            plan_comparison_repository=self.plan_comparison_repository,
            optimization_seed=DEFAULT_RANDOM_SEED,
            activation_collaborator=self.activation_service,
            resource_commitment_repository=self.resource_commitment_repository,
        )


@pytest.fixture
def real_scorer():
    from src.calculators.response_optimization.response_plan_scorer import ResponsePlanScorer

    return ResponsePlanScorer()


# ---------------------------------------------------------------------------
# 1. Full changed-state cycle -> REFRESHED
# ---------------------------------------------------------------------------


def test_changed_planning_state_runs_full_real_cycle_and_returns_refreshed(sqlite_session_factory, real_scorer):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory, resource_ids=("truck-1",))
    stored_target_set = seed_target_set(wiring.response_target_repository, fire_event_id)
    baseline_collaborator = wiring.baseline_collaborator(real_scorer)
    orchestrator = wiring.orchestrator(OperationalContextService(), baseline_collaborator)

    result = orchestrator.refresh(fire_event_id=fire_event_id, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.REFRESHED
    assert result.route_planning_run_id is not None
    assert result.response_plan_id is not None
    assert result.comparison_id is not None


# ---------------------------------------------------------------------------
# 2. Same state -> NO_OP, zero duplicate artifacts
# ---------------------------------------------------------------------------


def test_same_state_second_refresh_is_no_op_with_no_duplicate_artifacts(sqlite_session_factory, real_scorer):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory, resource_ids=("truck-1",))
    stored_target_set = seed_target_set(wiring.response_target_repository, fire_event_id)
    baseline_collaborator = wiring.baseline_collaborator(real_scorer)
    orchestrator = wiring.orchestrator(OperationalContextService(), baseline_collaborator)

    first = orchestrator.refresh(fire_event_id=fire_event_id, as_of=AS_OF)
    assert first.status is PlanningRefreshStatus.REFRESHED

    runs_after_first = wiring.route_planning_repository.get_history_for_event(fire_event_id)
    plans_after_first = wiring.response_plan_repository.get_for_fire_event(fire_event_id)
    comparisons_after_first = wiring.plan_comparison_repository.list_for_fire_event(fire_event_id)

    second = orchestrator.refresh(fire_event_id=fire_event_id, as_of=LATER)

    assert second.status is PlanningRefreshStatus.NO_OP
    assert second.route_planning_run_id == first.route_planning_run_id
    assert second.response_plan_id == first.response_plan_id
    assert second.comparison_id == first.comparison_id
    assert wiring.route_planning_repository.get_history_for_event(fire_event_id) == runs_after_first
    assert wiring.response_plan_repository.get_for_fire_event(fire_event_id) == plans_after_first
    assert wiring.plan_comparison_repository.list_for_fire_event(fire_event_id) == comparisons_after_first


# ---------------------------------------------------------------------------
# 3. Resource snapshot consistency
# ---------------------------------------------------------------------------


def test_optimization_consumes_exact_resource_snapshot_despite_later_availability_change(
    sqlite_session_factory,
):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1", "truck-2"])
    response_target_repository = ResponseTargetRepository(sqlite_session_factory)
    route_planning_repository = RoutePlanningRepository(sqlite_session_factory)
    seed_target_set(response_target_repository, fire_event_id)

    routing_collaborator = PersistingFakeRoutingCollaborator(
        response_target_repository, route_planning_repository, ("truck-1", "truck-2")
    )
    routing_result = routing_collaborator.plan(fire_event_id=fire_event_id, as_of=AS_OF)

    # Availability changes after routing: truck-2 becomes ASSIGNED.
    from src.repositories.firefighting_resource_repository import FirefightingResourceRepository

    FirefightingResourceRepository(sqlite_session_factory).update_status("truck-2", ResourceStatus.ASSIGNED)

    class SpyOptimizationAgent:
        def __init__(self):
            self.calls = []

        def optimize_from_input(self, optimization_input, *, as_of, config):
            self.calls.append(optimization_input)
            raise RuntimeError("not needed - inspecting the constructed input is enough")

    spy_agent = SpyOptimizationAgent()
    adapter = ResponseOptimizationCollaboratorAdapter(
        optimization_agent=spy_agent,
        route_planning_repository=route_planning_repository,
        response_target_repository=response_target_repository,
    )

    with pytest.raises(RuntimeError):
        adapter.optimize(route_planning_run_id=routing_result.run_id, as_of=LATER, seed=DEFAULT_RANDOM_SEED)

    resource_ids = {r.resource_id for r in spy_agent.calls[0].resources}
    assert resource_ids == {"truck-1", "truck-2"}


# ---------------------------------------------------------------------------
# 4. Exact target-set consistency
# ---------------------------------------------------------------------------


def test_optimization_consumes_exact_target_set_despite_newer_target_set(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    response_target_repository = ResponseTargetRepository(sqlite_session_factory)
    route_planning_repository = RoutePlanningRepository(sqlite_session_factory)

    target_set_a = seed_target_set(response_target_repository, fire_event_id, as_of=AS_OF, priority_score=150.0)

    routing_collaborator = PersistingFakeRoutingCollaborator(
        response_target_repository, route_planning_repository, ("truck-1",)
    )
    routing_result = routing_collaborator.plan(fire_event_id=fire_event_id, as_of=AS_OF)

    # A newer target set (B) becomes latest before optimization runs.
    seed_target_set(response_target_repository, fire_event_id, as_of=LATER, priority_score=999.0)

    class SpyOptimizationAgent:
        def __init__(self):
            self.calls = []

        def optimize_from_input(self, optimization_input, *, as_of, config):
            self.calls.append(optimization_input)
            raise RuntimeError("not needed - inspecting the constructed input is enough")

    spy_agent = SpyOptimizationAgent()
    adapter = ResponseOptimizationCollaboratorAdapter(
        optimization_agent=spy_agent,
        route_planning_repository=route_planning_repository,
        response_target_repository=response_target_repository,
    )

    with pytest.raises(RuntimeError):
        adapter.optimize(route_planning_run_id=routing_result.run_id, as_of=LATER, seed=DEFAULT_RANDOM_SEED)

    optimization_input = spy_agent.calls[0]
    assert optimization_input.response_target_set_id == target_set_a.id
    assert optimization_input.targets[0].priority_score == 150.0


# ---------------------------------------------------------------------------
# 5. Baseline-only recovery
# ---------------------------------------------------------------------------


def test_baseline_only_recovery_skips_routing_and_optimization(sqlite_session_factory, real_scorer):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory, resource_ids=("truck-1",))
    stored_target_set = seed_target_set(wiring.response_target_repository, fire_event_id)
    real_baseline_collaborator = wiring.baseline_collaborator(real_scorer)
    orchestrator = wiring.orchestrator(OperationalContextService(), real_baseline_collaborator)

    # First cycle creates the plan + sidecar, but strip the comparison it also created
    # so the DB state matches "missing comparison" for this scenario.
    first = orchestrator.refresh(fire_event_id=fire_event_id, as_of=AS_OF)
    assert first.status is PlanningRefreshStatus.REFRESHED

    from src.database.models.plan_comparison_db import PlanComparisonDB

    session = sqlite_session_factory()
    session.query(PlanComparisonDB).delete()
    session.commit()
    session.close()
    assert wiring.plan_comparison_repository.list_for_fire_event(fire_event_id) == ()

    spy_routing = SpyRoutingCollaborator(wiring.routing_collaborator)
    spy_optimization = SpyOptimizationCollaborator(wiring.optimization_collaborator)
    recovery_orchestrator = ResponsePlanningRefreshOrchestrator(
        planning_state_builder=PlanningEffectiveStateBuilder(
            response_target_repository=wiring.response_target_repository,
            operational_context_service=OperationalContextService(),
        ),
        routing_collaborator=spy_routing,
        optimization_collaborator=spy_optimization,
        baseline_collaborator=real_baseline_collaborator,
        fire_event_repository=wiring.fire_event_repository,
        response_plan_repository=wiring.response_plan_repository,
        response_plan_planning_state_repository=wiring.sidecar_repository,
        plan_comparison_repository=wiring.plan_comparison_repository,
        optimization_seed=DEFAULT_RANDOM_SEED,
        activation_collaborator=wiring.activation_service,
        resource_commitment_repository=wiring.resource_commitment_repository,
    )

    result = recovery_orchestrator.refresh(fire_event_id=fire_event_id, as_of=LATER)

    assert result.status is PlanningRefreshStatus.REFRESHED
    assert result.route_planning_run_id == first.route_planning_run_id
    assert result.response_plan_id == first.response_plan_id
    assert result.comparison_id is not None
    assert spy_routing.calls == []
    assert spy_optimization.calls == []


# ---------------------------------------------------------------------------
# Acceptance traceability
# ---------------------------------------------------------------------------


def test_refreshed_plan_is_fully_traceable(sqlite_session_factory, real_scorer):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory, resource_ids=("truck-1",))
    stored_target_set = seed_target_set(wiring.response_target_repository, fire_event_id)
    baseline_collaborator = wiring.baseline_collaborator(real_scorer)
    orchestrator = wiring.orchestrator(OperationalContextService(), baseline_collaborator)

    result = orchestrator.refresh(fire_event_id=fire_event_id, as_of=AS_OF)
    assert result.status is PlanningRefreshStatus.REFRESHED

    stored_plan = wiring.response_plan_repository.get_by_id(result.response_plan_id)
    assert stored_plan.plan.fire_event_id == fire_event_id
    assert stored_plan.plan.response_target_set_id == stored_target_set.id
    assert stored_plan.plan.route_planning_run_id == result.route_planning_run_id

    stored_run = wiring.route_planning_repository.get_by_id(result.route_planning_run_id)
    assert set(stored_run.run.resource_ids) == {"truck-1"}
    assert len(stored_run.routes) == 1

    sidecar = wiring.sidecar_repository.get_for_plan(result.response_plan_id)
    assert sidecar is not None
    assert sidecar.response_plan_id == result.response_plan_id

    stored_comparison = wiring.plan_comparison_repository.get_by_id(result.comparison_id)
    assert stored_comparison.comparison.optimized_plan_id == result.response_plan_id
    assert stored_comparison.comparison.route_planning_run_id == result.route_planning_run_id
    assert stored_comparison.comparison.response_target_set_id == stored_target_set.id
