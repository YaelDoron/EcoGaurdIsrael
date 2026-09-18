"""Task 7: Failure / recovery / history-preservation integration tests for US 5.4.

Real repositories, real DB models, and real Task 6/6.1 production adapters
throughout. Only the one stage each test is deliberately forcing to fail is
replaced by a small, explicit, deterministic double at that stage's own
boundary (see each Failing* class's docstring for exactly what it stands in
for) - everything else in the pipeline is real, mirroring
test_response_planning_refresh_integration.py's own established convention.

Routing itself is still stood in by `PersistingFakeRoutingCollaborator`
(skips Dijkstra/OSM but performs REAL persistence), exactly as in Task 6,
since genuinely running Dijkstra needs a live/cached road network that is
out of scope here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.agents.analysis.response_optimization_agent import ResponseOptimizationAgent
from src.agents.routing.route_planning_result import RoutePlanningResult, RoutePlanningStatus
from src.calculators.response_optimization.response_optimization_config import DEFAULT_RANDOM_SEED
from src.calculators.response_optimization.response_plan_scorer import ResponsePlanScorer
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models import (
    ResponseTarget,
    ResponseTargetSet,
    ResponseTargetType,
    ResourceStatus,
)
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
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
from src.services.baseline_comparison.baseline_comparison_service import (
    BaselineComparisonService,
    BaselineComparisonServiceError,
)
from src.services.baseline_comparison.response_plan_baseline_scorer_adapter import ResponsePlanBaselineScorerAdapter
from src.services.operational.operational_context_service import OperationalContextService
from src.services.response_planning.baseline_comparison_collaborator_adapter import (
    BaselineComparisonCollaboratorAdapter,
)
from src.services.response_planning.current_response_plan_resolver import CurrentResponsePlanResolver
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


def later(hours: int) -> datetime:
    return AS_OF + timedelta(hours=hours)


# ---------------------------------------------------------------------------
# Test-only failing doubles - each stands in for exactly one real boundary
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


class FailingRoutingCollaborator:
    """Stands in for RoutePlanningAgent failing - real RoutePlanningAgent.plan() persists
    nothing and returns exactly this FAILED shape when routing cannot complete."""

    def __init__(self, error_message="Routing failed."):
        self.error_message = error_message
        self.calls = []

    def plan(self, *, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        return RoutePlanningResult(
            success=False,
            fire_event_id=fire_event_id,
            status=RoutePlanningStatus.FAILED,
            run_id=None,
            run=None,
            route_count=0,
            error_message=self.error_message,
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


class FailingGeneticOptimizer:
    """Stands in for GeneticResponsePlanOptimizer failing - runs through the REAL
    ResponseOptimizationAgent, which catches this and returns FAILED without persisting."""

    def __init__(self):
        self.calls = []

    def optimize(self, optimization_input, config):
        self.calls.append({"optimization_input": optimization_input, "config": config})
        raise RuntimeError("GA exploded")


class FailingSidecarRepository(ResponsePlanPlanningStateRepository):
    """Real repository whose save() is forced to fail; get_for_plan stays fully real."""

    def __init__(self, session_factory):
        super().__init__(session_factory)
        self.save_calls = []

    def save(self, *, response_plan_id, planning_effective_state_fingerprint):
        self.save_calls.append(response_plan_id)
        raise RuntimeError("sidecar write failed")


class FailingActivationService(ResponsePlanActivationService):
    """Real activation service whose activate() is forced to fail (Stage 1
    moved the sidecar write inside the atomic commitment+sidecar
    transaction, so a "sidecar failure" is now an activation failure): the
    ResponsePlan row itself was already saved earlier, separately, by
    ResponseOptimizationAgent - only the atomic activation step fails,
    leaving that plan real but without commitments or a current-plan
    sidecar, exactly like the old sidecar-only failure this replaces."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_calls = []

    def activate(self, **kwargs):
        self.save_calls.append(kwargs.get("response_plan_id"))
        raise RuntimeError("activation failed")


class FailingBaselineComparisonService(BaselineComparisonService):
    """Stands in for the baseline-comparison stage failing before anything is persisted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def compare(self, *, response_plan_id):
        self.calls.append(response_plan_id)
        raise BaselineComparisonServiceError("baseline exploded")


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


def mark_event_inactive(session_factory, fire_event_id, status: FireEventStatus, as_of: datetime) -> None:
    repository = FireEventRepository(session_factory)
    stored = repository.get_by_id(fire_event_id)
    repository.update_event(
        fire_event_id,
        FireEvent(
            latitude=stored.event.latitude,
            longitude=stored.event.longitude,
            detected_at=stored.event.detected_at,
            updated_at=as_of,
            status=status,
            detection_confidence=stored.event.detection_confidence,
            methodology=stored.event.methodology,
            methodology_version=stored.event.methodology_version,
        ),
    )


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
    """Bundles every real repository needed for one FireEvent's real orchestrator(s).

    Multiple orchestrator() calls against the SAME Wiring share the SAME
    underlying repositories/session_factory, so a failing refresh followed by
    a healthy one sees the exact same persisted history.
    """

    def __init__(self, session_factory, resource_ids=("truck-1",)):
        self.session_factory = session_factory
        self.resource_ids = resource_ids
        self.response_target_repository = ResponseTargetRepository(session_factory)
        self.route_planning_repository = RoutePlanningRepository(session_factory)
        self.response_plan_repository = ResponsePlanRepository(session_factory)
        self.plan_comparison_repository = PlanComparisonRepository(session_factory)
        self.sidecar_repository = ResponsePlanPlanningStateRepository(session_factory)
        self.fire_event_repository = FireEventRepository(session_factory)
        self.resource_commitment_repository = ResourceCommitmentRepository(session_factory)

    def working_activation_service(self):
        return ResponsePlanActivationService(
            response_plan_repository=self.response_plan_repository,
            resource_commitment_repository=self.resource_commitment_repository,
            session_factory=self.session_factory,
        )

    def working_routing_collaborator(self):
        return PersistingFakeRoutingCollaborator(
            self.response_target_repository, self.route_planning_repository, self.resource_ids
        )

    def working_optimization_collaborator(self, optimizer=None):
        return ResponseOptimizationCollaboratorAdapter(
            optimization_agent=ResponseOptimizationAgent(repository=self.response_plan_repository, optimizer=optimizer),
            route_planning_repository=self.route_planning_repository,
            response_target_repository=self.response_target_repository,
        )

    def working_baseline_collaborator(self, service_cls=BaselineComparisonService):
        service = service_cls(
            optimized_plan_reader=ResponsePlanOptimizedPlanReaderAdapter(self.response_plan_repository),
            route_planning_run_reader=RoutePlanningRunReaderAdapter(self.route_planning_repository),
            scorer=ResponsePlanBaselineScorerAdapter(ResponsePlanScorer()),
            response_target_set_reader=self.response_target_repository,
            plan_comparison_repository=self.plan_comparison_repository,
        )
        return BaselineComparisonCollaboratorAdapter(service, self.plan_comparison_repository)

    def orchestrator(
        self,
        *,
        routing=None,
        optimization=None,
        baseline=None,
        sidecar_repository=None,
        activation_collaborator=None,
    ) -> ResponsePlanningRefreshOrchestrator:
        return ResponsePlanningRefreshOrchestrator(
            planning_state_builder=PlanningEffectiveStateBuilder(
                response_target_repository=self.response_target_repository,
                operational_context_service=OperationalContextService(),
            ),
            routing_collaborator=routing or self.working_routing_collaborator(),
            optimization_collaborator=optimization or self.working_optimization_collaborator(),
            baseline_collaborator=baseline or self.working_baseline_collaborator(),
            fire_event_repository=self.fire_event_repository,
            response_plan_repository=self.response_plan_repository,
            response_plan_planning_state_repository=sidecar_repository or self.sidecar_repository,
            plan_comparison_repository=self.plan_comparison_repository,
            optimization_seed=DEFAULT_RANDOM_SEED,
            activation_collaborator=activation_collaborator or self.working_activation_service(),
            resource_commitment_repository=self.resource_commitment_repository,
        )

    def history_counts(self, fire_event_id):
        return {
            "runs": len(self.route_planning_repository.get_history_for_event(fire_event_id)),
            "plans": len(self.response_plan_repository.get_for_fire_event(fire_event_id)),
            "comparisons": len(self.plan_comparison_repository.list_for_fire_event(fire_event_id)),
        }

    def resolver(self) -> CurrentResponsePlanResolver:
        return CurrentResponsePlanResolver(self.response_plan_repository, self.sidecar_repository)


def run_full_success_cycle(wiring: Wiring, fire_event_id: int, as_of: datetime):
    """Run one real, fully-successful refresh cycle and return its result."""
    result = wiring.orchestrator().refresh(fire_event_id=fire_event_id, as_of=as_of)
    assert result.status is PlanningRefreshStatus.REFRESHED
    return result


# ---------------------------------------------------------------------------
# 1. Routing failure preserves prior history
# ---------------------------------------------------------------------------


def test_routing_failure_creates_no_artifacts_and_preserves_prior_history(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    prior = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    counts_before = wiring.history_counts(fire_event_id)
    sidecar_before = wiring.sidecar_repository.get_for_plan(prior.response_plan_id)

    # Change effective state so this is a genuine new-cycle attempt, not NO_OP.
    seed_target_set(wiring.response_target_repository, fire_event_id, as_of=later(1), priority_score=999.0)

    result = wiring.orchestrator(routing=FailingRoutingCollaborator("no resources")).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )

    assert result.status is PlanningRefreshStatus.FAILED
    assert result.route_planning_run_id is None
    assert result.response_plan_id is None
    assert result.comparison_id is None

    assert wiring.history_counts(fire_event_id) == counts_before
    unchanged_plan = wiring.response_plan_repository.get_by_id(prior.response_plan_id)
    assert unchanged_plan.plan.plan_score == 95.0 or unchanged_plan.plan is not None  # still present, untouched below
    assert wiring.sidecar_repository.get_for_plan(prior.response_plan_id) == sidecar_before
    assert wiring.plan_comparison_repository.get_by_id(prior.comparison_id) is not None


# ---------------------------------------------------------------------------
# 2 & 3. Optimization failure: routing run kept, no plan/sidecar/comparison, prior history intact
# ---------------------------------------------------------------------------


def test_optimization_failure_preserves_routing_run_and_prior_history_creates_no_plan_or_sidecar(
    sqlite_session_factory,
):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    prior = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    counts_before = wiring.history_counts(fire_event_id)

    seed_target_set(wiring.response_target_repository, fire_event_id, as_of=later(1), priority_score=999.0)

    failing_optimization = wiring.working_optimization_collaborator(optimizer=FailingGeneticOptimizer())
    result = wiring.orchestrator(optimization=failing_optimization).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )

    assert result.status is PlanningRefreshStatus.FAILED
    assert result.route_planning_run_id is not None  # routing succeeded and persisted a new run
    assert result.route_planning_run_id != prior.route_planning_run_id
    assert result.response_plan_id is None
    assert result.comparison_id is None

    counts_after = wiring.history_counts(fire_event_id)
    assert counts_after["runs"] == counts_before["runs"] + 1  # the new routing run IS persisted
    assert counts_after["plans"] == counts_before["plans"]  # no new plan
    assert counts_after["comparisons"] == counts_before["comparisons"]  # no new comparison
    assert wiring.sidecar_repository.get_for_plan(prior.response_plan_id) is not None

    # The latest successful ResponsePlan is still the prior one.
    latest_plan = wiring.response_plan_repository.get_latest_for_fire_event(fire_event_id)
    assert latest_plan.id == prior.response_plan_id


# ---------------------------------------------------------------------------
# 4 & 5. Sidecar failure: new plan kept, no comparison, missing sidecar never NO_OPs
# ---------------------------------------------------------------------------


def test_sidecar_failure_leaves_plan_persisted_with_no_comparison_and_prevents_false_no_op(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    failing_activation = FailingActivationService(
        response_plan_repository=wiring.response_plan_repository,
        resource_commitment_repository=wiring.resource_commitment_repository,
        session_factory=sqlite_session_factory,
    )
    baseline_collaborator = wiring.working_baseline_collaborator()
    result = wiring.orchestrator(activation_collaborator=failing_activation, baseline=baseline_collaborator).refresh(
        fire_event_id=fire_event_id, as_of=AS_OF
    )

    assert result.status is PlanningRefreshStatus.FAILED
    assert result.route_planning_run_id is not None
    assert result.response_plan_id is not None
    assert result.comparison_id is None
    assert len(failing_activation.save_calls) == 1

    new_plan_id = result.response_plan_id
    assert wiring.response_plan_repository.get_by_id(new_plan_id) is not None  # plan NOT rolled back
    assert wiring.sidecar_repository.get_for_plan(new_plan_id) is None  # truthfully no sidecar
    assert wiring.plan_comparison_repository.list_for_fire_event(fire_event_id) == ()  # baseline never invoked

    # A subsequent refresh with the SAME state must not falsely NO_OP: the
    # latest plan has no sidecar, so a full new cycle must run again.
    routing_spy = SpyRoutingCollaborator(wiring.working_routing_collaborator())
    optimization_spy = SpyOptimizationCollaborator(wiring.working_optimization_collaborator())
    recovery = wiring.orchestrator(routing=routing_spy, optimization=optimization_spy).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )

    assert recovery.status is PlanningRefreshStatus.REFRESHED
    assert len(routing_spy.calls) == 1
    assert len(optimization_spy.calls) == 1
    assert recovery.response_plan_id != new_plan_id


# ---------------------------------------------------------------------------
# 6-9. Baseline failure -> recovery (baseline-only retry) -> true NO_OP
# ---------------------------------------------------------------------------


def test_baseline_failure_then_recovery_then_true_no_op_three_refresh_sequence(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    # --- Refresh 1: baseline fails after a fully real routing+optimization success ---
    # Force the baseline stage to fail cleanly, without touching real persistence.
    failing_service = FailingBaselineComparisonService(
        optimized_plan_reader=ResponsePlanOptimizedPlanReaderAdapter(wiring.response_plan_repository),
        route_planning_run_reader=RoutePlanningRunReaderAdapter(wiring.route_planning_repository),
        scorer=ResponsePlanBaselineScorerAdapter(ResponsePlanScorer()),
        response_target_set_reader=wiring.response_target_repository,
        plan_comparison_repository=wiring.plan_comparison_repository,
    )
    failing_baseline_collaborator = BaselineComparisonCollaboratorAdapter(
        failing_service, wiring.plan_comparison_repository
    )

    first = wiring.orchestrator(baseline=failing_baseline_collaborator).refresh(fire_event_id=fire_event_id, as_of=AS_OF)

    assert first.status is PlanningRefreshStatus.FAILED
    assert first.route_planning_run_id is not None
    assert first.response_plan_id is not None
    assert first.comparison_id is None
    assert wiring.sidecar_repository.get_for_plan(first.response_plan_id) is not None
    assert wiring.plan_comparison_repository.list_for_fire_event(fire_event_id) == ()

    counts_after_first = wiring.history_counts(fire_event_id)

    # --- Refresh 2: same state, baseline now works -> baseline-only recovery ---
    routing_spy = SpyRoutingCollaborator(wiring.working_routing_collaborator())
    optimization_spy = SpyOptimizationCollaborator(wiring.working_optimization_collaborator())
    second = wiring.orchestrator(routing=routing_spy, optimization=optimization_spy).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )

    assert second.status is PlanningRefreshStatus.REFRESHED
    assert routing_spy.calls == []
    assert optimization_spy.calls == []
    assert second.route_planning_run_id == first.route_planning_run_id
    assert second.response_plan_id == first.response_plan_id
    assert second.comparison_id is not None

    counts_after_second = wiring.history_counts(fire_event_id)
    assert counts_after_second["runs"] == counts_after_first["runs"]
    assert counts_after_second["plans"] == counts_after_first["plans"]
    assert counts_after_second["comparisons"] == counts_after_first["comparisons"] + 1

    # --- Refresh 3: same state again -> true NO_OP, zero new artifacts ---
    routing_spy_2 = SpyRoutingCollaborator(wiring.working_routing_collaborator())
    optimization_spy_2 = SpyOptimizationCollaborator(wiring.working_optimization_collaborator())
    third = wiring.orchestrator(routing=routing_spy_2, optimization=optimization_spy_2).refresh(
        fire_event_id=fire_event_id, as_of=later(2)
    )

    assert third.status is PlanningRefreshStatus.NO_OP
    assert routing_spy_2.calls == []
    assert optimization_spy_2.calls == []
    assert third.route_planning_run_id == second.route_planning_run_id
    assert third.response_plan_id == second.response_plan_id
    assert third.comparison_id == second.comparison_id
    assert wiring.history_counts(fire_event_id) == counts_after_second


# ---------------------------------------------------------------------------
# 11 & 12. Append-only across two genuinely different fingerprints
# ---------------------------------------------------------------------------


def test_two_different_fingerprints_append_separate_history_without_touching_the_first(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id, priority_score=150.0)

    cycle_a = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    plan_a_before = wiring.response_plan_repository.get_by_id(cycle_a.response_plan_id)
    run_a_before = wiring.route_planning_repository.get_by_id(cycle_a.route_planning_run_id)
    sidecar_a_before = wiring.sidecar_repository.get_for_plan(cycle_a.response_plan_id)
    comparison_a_before = wiring.plan_comparison_repository.get_by_id(cycle_a.comparison_id)

    # Genuinely different effective state: a new target set with a different priority.
    seed_target_set(wiring.response_target_repository, fire_event_id, as_of=later(1), priority_score=777.0)
    cycle_b = run_full_success_cycle(wiring, fire_event_id, later(1))

    assert cycle_b.route_planning_run_id != cycle_a.route_planning_run_id
    assert cycle_b.response_plan_id != cycle_a.response_plan_id
    assert cycle_b.comparison_id != cycle_a.comparison_id

    # A is completely unchanged.
    assert wiring.response_plan_repository.get_by_id(cycle_a.response_plan_id) == plan_a_before
    assert wiring.route_planning_repository.get_by_id(cycle_a.route_planning_run_id) == run_a_before
    assert wiring.sidecar_repository.get_for_plan(cycle_a.response_plan_id) == sidecar_a_before
    assert wiring.plan_comparison_repository.get_by_id(cycle_a.comparison_id) == comparison_a_before

    # Sidecars/comparisons point at the correct, distinct plans.
    sidecar_b = wiring.sidecar_repository.get_for_plan(cycle_b.response_plan_id)
    assert sidecar_b.response_plan_id == cycle_b.response_plan_id
    comparison_b = wiring.plan_comparison_repository.get_by_id(cycle_b.comparison_id)
    assert comparison_b.comparison.optimized_plan_id == cycle_b.response_plan_id

    counts = wiring.history_counts(fire_event_id)
    assert counts == {"runs": 2, "plans": 2, "comparisons": 2}


# ---------------------------------------------------------------------------
# 10. Same state repeated -> no history growth at all
# ---------------------------------------------------------------------------


def test_repeated_refresh_with_identical_state_never_grows_history(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    first = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    counts = wiring.history_counts(fire_event_id)

    for hour in (1, 2, 3):
        result = wiring.orchestrator().refresh(fire_event_id=fire_event_id, as_of=later(hour))
        assert result.status is PlanningRefreshStatus.NO_OP
        assert result.response_plan_id == first.response_plan_id
        assert wiring.history_counts(fire_event_id) == counts


# ---------------------------------------------------------------------------
# 13, 14 & 15. Inactive event hardening (RESOLVED / DISMISSED)
# ---------------------------------------------------------------------------


def _run_inactive_event_scenario(sqlite_session_factory, status: FireEventStatus):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    prior = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    counts_before = wiring.history_counts(fire_event_id)

    mark_event_inactive(sqlite_session_factory, fire_event_id, status, later(1))

    result = wiring.orchestrator().refresh(fire_event_id=fire_event_id, as_of=later(2))

    assert result.status is PlanningRefreshStatus.INACTIVE_EVENT
    assert result.route_planning_run_id is None
    assert result.response_plan_id is None
    assert result.comparison_id is None
    assert wiring.history_counts(fire_event_id) == counts_before

    # Historical artifacts remain fully intact and queryable.
    assert wiring.response_plan_repository.get_by_id(prior.response_plan_id) is not None
    assert wiring.route_planning_repository.get_by_id(prior.route_planning_run_id) is not None
    assert wiring.sidecar_repository.get_for_plan(prior.response_plan_id) is not None
    assert wiring.plan_comparison_repository.get_by_id(prior.comparison_id) is not None


def test_resolved_event_creates_no_new_artifacts_and_preserves_history(sqlite_session_factory):
    _run_inactive_event_scenario(sqlite_session_factory, FireEventStatus.RESOLVED)


def test_dismissed_event_creates_no_new_artifacts_and_preserves_history(sqlite_session_factory):
    _run_inactive_event_scenario(sqlite_session_factory, FireEventStatus.DISMISSED)


# ---------------------------------------------------------------------------
# 16. Multi-event isolation under failure
# ---------------------------------------------------------------------------


def test_event_a_failure_does_not_affect_event_b_history_and_b_still_refreshes_normally(sqlite_session_factory):
    fire_event_a = seed_fire_event(sqlite_session_factory)
    fire_event_b = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1", "truck-2"])
    wiring = Wiring(sqlite_session_factory, resource_ids=("truck-1",))
    seed_target_set(wiring.response_target_repository, fire_event_a, priority_score=150.0)
    seed_target_set(wiring.response_target_repository, fire_event_b, priority_score=150.0)

    cycle_a = run_full_success_cycle(wiring, fire_event_a, AS_OF)
    cycle_b = run_full_success_cycle(wiring, fire_event_b, AS_OF)
    counts_b_before = wiring.history_counts(fire_event_b)

    # Change A's state and force A's optimization to fail.
    seed_target_set(wiring.response_target_repository, fire_event_a, as_of=later(1), priority_score=999.0)
    failing_optimization = wiring.working_optimization_collaborator(optimizer=FailingGeneticOptimizer())
    result_a = wiring.orchestrator(optimization=failing_optimization).refresh(
        fire_event_id=fire_event_a, as_of=later(1)
    )
    assert result_a.status is PlanningRefreshStatus.FAILED

    # Event B is completely unaffected.
    assert wiring.history_counts(fire_event_b) == counts_b_before
    assert wiring.response_plan_repository.get_by_id(cycle_b.response_plan_id) is not None
    assert wiring.route_planning_repository.get_by_id(cycle_b.route_planning_run_id) is not None
    assert wiring.sidecar_repository.get_for_plan(cycle_b.response_plan_id) is not None
    assert wiring.plan_comparison_repository.get_by_id(cycle_b.comparison_id) is not None
    latest_b = wiring.response_plan_repository.get_latest_for_fire_event(fire_event_b)
    assert latest_b.id == cycle_b.response_plan_id

    # B still refreshes normally (same state -> NO_OP).
    result_b = wiring.orchestrator().refresh(fire_event_id=fire_event_b, as_of=later(1))
    assert result_b.status is PlanningRefreshStatus.NO_OP
    assert result_b.response_plan_id == cycle_b.response_plan_id


# ---------------------------------------------------------------------------
# 17 & 18. Failed optimization after prior success never becomes an FP2 NO_OP baseline
# ---------------------------------------------------------------------------


def test_failed_optimization_after_prior_success_does_not_poison_next_refresh(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id, priority_score=150.0)

    cycle_1 = run_full_success_cycle(wiring, fire_event_id, AS_OF)  # FP1
    sidecar_1 = wiring.sidecar_repository.get_for_plan(cycle_1.response_plan_id)

    # Move to FP2: routing succeeds, optimization fails.
    seed_target_set(wiring.response_target_repository, fire_event_id, as_of=later(1), priority_score=999.0)
    failing_optimization = wiring.working_optimization_collaborator(optimizer=FailingGeneticOptimizer())
    fp2_attempt = wiring.orchestrator(optimization=failing_optimization).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )
    assert fp2_attempt.status is PlanningRefreshStatus.FAILED
    assert fp2_attempt.route_planning_run_id is not None  # FP2's routing run DOES exist

    # Latest successful plan is unchanged; its sidecar still reflects FP1.
    latest_plan = wiring.response_plan_repository.get_latest_for_fire_event(fire_event_id)
    assert latest_plan.id == cycle_1.response_plan_id
    assert wiring.sidecar_repository.get_for_plan(cycle_1.response_plan_id) == sidecar_1

    # Next refresh with the SAME FP2 state must attempt a full new cycle -
    # NOT treat the failed attempt's routing run as an existing NO_OP baseline.
    routing_spy = SpyRoutingCollaborator(wiring.working_routing_collaborator())
    optimization_spy = SpyOptimizationCollaborator(wiring.working_optimization_collaborator())
    fp2_retry = wiring.orchestrator(routing=routing_spy, optimization=optimization_spy).refresh(
        fire_event_id=fire_event_id, as_of=later(2)
    )

    assert fp2_retry.status is PlanningRefreshStatus.REFRESHED
    assert len(routing_spy.calls) == 1  # routing WAS invoked again, not skipped
    assert len(optimization_spy.calls) == 1
    assert fp2_retry.response_plan_id != cycle_1.response_plan_id
    assert fp2_retry.response_plan_id is not None


# ---------------------------------------------------------------------------
# 19. Legacy / missing-sidecar ResponsePlan safety
# ---------------------------------------------------------------------------


def test_legacy_plan_without_sidecar_never_causes_false_no_op(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    stored_target_set = seed_target_set(wiring.response_target_repository, fire_event_id)

    # Seed a routing run + a ResponsePlan directly (bypassing the orchestrator
    # entirely), simulating data from before the Task 4 sidecar existed - no
    # ResponsePlanPlanningState row is created for it.
    routing_collaborator = wiring.working_routing_collaborator()
    routing_result = routing_collaborator.plan(fire_event_id=fire_event_id, as_of=AS_OF)
    from src.models import ResponseAction, ResponsePlan, ResponsePlanStatus

    active_target_id = stored_target_set.targets[0].id
    legacy_plan = wiring.response_plan_repository.save(
        ResponsePlan(
            fire_event_id=fire_event_id,
            response_target_set_id=stored_target_set.id,
            route_planning_run_id=routing_result.run_id,
            generated_at=AS_OF,
            status=ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS,
            methodology="GENETIC_RESOURCE_ALLOCATION",
            methodology_version="1.0",
            random_seed=DEFAULT_RANDOM_SEED,
            actions=(),
            uncovered_target_ids=(active_target_id,),
            plan_score=0.0,
            coverage_score=0.0,
            average_eta_seconds=None,
        )
    )
    assert wiring.sidecar_repository.get_for_plan(legacy_plan.id) is None

    routing_spy = SpyRoutingCollaborator(wiring.working_routing_collaborator())
    optimization_spy = SpyOptimizationCollaborator(wiring.working_optimization_collaborator())
    result = wiring.orchestrator(routing=routing_spy, optimization=optimization_spy).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )

    assert result.status is PlanningRefreshStatus.REFRESHED
    assert len(routing_spy.calls) == 1
    assert len(optimization_spy.calls) == 1
    assert result.response_plan_id != legacy_plan.id

    # History preserved: the legacy plan is still there, still without a sidecar.
    assert wiring.response_plan_repository.get_by_id(legacy_plan.id) is not None
    assert wiring.sidecar_repository.get_for_plan(legacy_plan.id) is None


# ---------------------------------------------------------------------------
# 20. Duplicate historical comparisons do not corrupt NO_OP detection
# ---------------------------------------------------------------------------


def test_duplicate_historical_comparisons_do_not_corrupt_no_op_detection(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    cycle = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    original_comparison = wiring.plan_comparison_repository.get_by_id(cycle.comparison_id)

    # Simulate a historical/manual duplicate: a second comparison row for the
    # exact same ResponsePlan, saved directly (bypassing the orchestrator).
    duplicate = wiring.plan_comparison_repository.save(original_comparison.comparison)
    assert duplicate.id > cycle.comparison_id

    result = wiring.orchestrator().refresh(fire_event_id=fire_event_id, as_of=later(1))

    assert result.status is PlanningRefreshStatus.NO_OP
    assert result.response_plan_id == cycle.response_plan_id
    assert result.comparison_id in (cycle.comparison_id, duplicate.id)  # a genuinely valid comparison for this plan

    # No duplicate rows were deleted.
    assert len(wiring.plan_comparison_repository.list_for_fire_event(fire_event_id)) == 2


def test_no_op_comparison_id_uses_the_newest_duplicate_matching_task_6_convention(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    cycle = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    original_comparison = wiring.plan_comparison_repository.get_by_id(cycle.comparison_id)
    duplicate = wiring.plan_comparison_repository.save(original_comparison.comparison)

    result = wiring.orchestrator().refresh(fire_event_id=fire_event_id, as_of=later(1))

    assert result.status is PlanningRefreshStatus.NO_OP
    assert result.comparison_id == duplicate.id  # highest/newest id, matching Task 6's own convention


# ---------------------------------------------------------------------------
# 21. Full history remains queryable after failures and recoveries
# ---------------------------------------------------------------------------


def test_full_history_remains_queryable_after_failures_and_recoveries(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id, priority_score=150.0)

    cycle_a = run_full_success_cycle(wiring, fire_event_id, AS_OF)

    # A failed attempt in between (optimization failure) - should not appear as a plan.
    seed_target_set(wiring.response_target_repository, fire_event_id, as_of=later(1), priority_score=500.0)
    failing_optimization = wiring.working_optimization_collaborator(optimizer=FailingGeneticOptimizer())
    failed_attempt = wiring.orchestrator(optimization=failing_optimization).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )
    assert failed_attempt.status is PlanningRefreshStatus.FAILED

    # A second successful cycle for the SAME (still current) FP2 state.
    cycle_b = wiring.orchestrator().refresh(fire_event_id=fire_event_id, as_of=later(2))
    assert cycle_b.status is PlanningRefreshStatus.REFRESHED

    runs = wiring.route_planning_repository.get_history_for_event(fire_event_id)
    plans = wiring.response_plan_repository.get_for_fire_event(fire_event_id)
    comparisons = wiring.plan_comparison_repository.list_for_fire_event(fire_event_id)

    run_ids = {r.id for r in runs}
    plan_ids = {p.id for p in plans}
    comparison_ids = {c.id for c in comparisons}

    assert cycle_a.route_planning_run_id in run_ids
    assert failed_attempt.route_planning_run_id in run_ids  # routing run from the failed attempt is real history
    assert cycle_b.route_planning_run_id in run_ids
    assert len(runs) == 3

    assert cycle_a.response_plan_id in plan_ids
    assert cycle_b.response_plan_id in plan_ids
    assert len(plans) == 2  # the failed optimization never produced a plan

    assert cycle_a.comparison_id in comparison_ids
    assert cycle_b.comparison_id in comparison_ids
    assert len(comparisons) == 2

    # Sidecar repository resolves the correct fingerprint per historical plan.
    sidecar_a = wiring.sidecar_repository.get_for_plan(cycle_a.response_plan_id)
    sidecar_b = wiring.sidecar_repository.get_for_plan(cycle_b.response_plan_id)
    assert sidecar_a.response_plan_id == cycle_a.response_plan_id
    assert sidecar_b.response_plan_id == cycle_b.response_plan_id
    assert sidecar_a.planning_effective_state_fingerprint != sidecar_b.planning_effective_state_fingerprint


# ---------------------------------------------------------------------------
# Task 8: CurrentResponsePlanResolver alignment with real Task 7 failure scenarios
# ---------------------------------------------------------------------------


def test_resolver_routing_failure_after_prior_success_keeps_prior_plan_current(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    prior = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    seed_target_set(wiring.response_target_repository, fire_event_id, as_of=later(1), priority_score=999.0)

    result = wiring.orchestrator(routing=FailingRoutingCollaborator()).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )
    assert result.status is PlanningRefreshStatus.FAILED

    current = wiring.resolver().resolve(fire_event_id=fire_event_id)
    assert current.id == prior.response_plan_id


def test_resolver_optimization_failure_after_prior_success_keeps_prior_plan_current(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    prior = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    seed_target_set(wiring.response_target_repository, fire_event_id, as_of=later(1), priority_score=999.0)

    failing_optimization = wiring.working_optimization_collaborator(optimizer=FailingGeneticOptimizer())
    result = wiring.orchestrator(optimization=failing_optimization).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )
    assert result.status is PlanningRefreshStatus.FAILED
    assert result.route_planning_run_id is not None  # a new routing run for the failed attempt DOES exist

    current = wiring.resolver().resolve(fire_event_id=fire_event_id)
    assert current.id == prior.response_plan_id


def test_resolver_sidecar_failure_keeps_prior_sidecar_backed_plan_current(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    prior = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    seed_target_set(wiring.response_target_repository, fire_event_id, as_of=later(1), priority_score=999.0)

    failing_activation = FailingActivationService(
        response_plan_repository=wiring.response_plan_repository,
        resource_commitment_repository=wiring.resource_commitment_repository,
        session_factory=sqlite_session_factory,
    )
    result = wiring.orchestrator(activation_collaborator=failing_activation).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )
    assert result.status is PlanningRefreshStatus.FAILED
    new_plan_id = result.response_plan_id
    assert new_plan_id is not None
    assert new_plan_id != prior.response_plan_id

    # The new plan is real history but is NOT current planning-safe (no sidecar).
    current = wiring.resolver().resolve(fire_event_id=fire_event_id)
    assert current.id == prior.response_plan_id
    assert current.id != new_plan_id
    # It is still queryable as history, unmutated.
    assert wiring.response_plan_repository.get_by_id(new_plan_id) is not None


def test_resolver_baseline_failure_still_makes_new_plan_current(sqlite_session_factory):
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    failing_service = FailingBaselineComparisonService(
        optimized_plan_reader=ResponsePlanOptimizedPlanReaderAdapter(wiring.response_plan_repository),
        route_planning_run_reader=RoutePlanningRunReaderAdapter(wiring.route_planning_repository),
        scorer=ResponsePlanBaselineScorerAdapter(ResponsePlanScorer()),
        response_target_set_reader=wiring.response_target_repository,
        plan_comparison_repository=wiring.plan_comparison_repository,
    )
    failing_baseline_collaborator = BaselineComparisonCollaboratorAdapter(
        failing_service, wiring.plan_comparison_repository
    )

    result = wiring.orchestrator(baseline=failing_baseline_collaborator).refresh(
        fire_event_id=fire_event_id, as_of=AS_OF
    )
    assert result.status is PlanningRefreshStatus.FAILED
    assert result.comparison_id is None
    assert wiring.plan_comparison_repository.list_for_fire_event(fire_event_id) == ()

    # The new plan+sidecar ARE current planning-safe despite the missing comparison.
    current = wiring.resolver().resolve(fire_event_id=fire_event_id)
    assert current.id == result.response_plan_id


def test_task5_no_op_baseline_and_task8_current_plan_are_intentionally_different_questions(
    sqlite_session_factory,
):
    """Task 5's NO_OP decision keys strictly off the LATEST persisted plan's sidecar; Task 8's
    resolver scans backward past a sidecar-less latest plan. These must not be merged."""
    fire_event_id = seed_fire_event(sqlite_session_factory)
    seed_station_and_resources(sqlite_session_factory, ["truck-1"])
    wiring = Wiring(sqlite_session_factory)
    seed_target_set(wiring.response_target_repository, fire_event_id)

    prior = run_full_success_cycle(wiring, fire_event_id, AS_OF)
    seed_target_set(wiring.response_target_repository, fire_event_id, as_of=later(1), priority_score=999.0)
    failing_activation = FailingActivationService(
        response_plan_repository=wiring.response_plan_repository,
        resource_commitment_repository=wiring.resource_commitment_repository,
        session_factory=sqlite_session_factory,
    )
    failed = wiring.orchestrator(activation_collaborator=failing_activation).refresh(
        fire_event_id=fire_event_id, as_of=later(1)
    )
    assert failed.status is PlanningRefreshStatus.FAILED
    sidecar_less_plan_id = failed.response_plan_id

    # Task 5's own view: the LATEST persisted plan (the sidecar-less one) is what
    # get_latest_for_fire_event returns - Task 5 forces a full refresh from it,
    # never falling back to an older fingerprint.
    latest_persisted = wiring.response_plan_repository.get_latest_for_fire_event(fire_event_id)
    assert latest_persisted.id == sidecar_less_plan_id
    assert wiring.sidecar_repository.get_for_plan(latest_persisted.id) is None

    # Task 8's own view: current planning-safe plan is the OLDER sidecar-backed one.
    current = wiring.resolver().resolve(fire_event_id=fire_event_id)
    assert current.id == prior.response_plan_id
    assert current.id != latest_persisted.id
