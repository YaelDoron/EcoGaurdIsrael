"""Tests for GlobalPlanningOrchestrator (Stage 2 of the Global
Multi-Incident Optimizer refactor).

Routing is stood in for (genuinely running Dijkstra needs a live/cached
road network); the GA/optimization, activation, and baseline-comparison
steps are fully real, mirroring tests/integration/test_response_planning_
refresh_integration.py's established pattern - so the actual thing Stage 2
adds (audit bookkeeping around the existing pipeline) is exercised against
a real, unmodified ResponsePlanningRefreshOrchestrator.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.agents.analysis.response_optimization_agent import ResponseOptimizationAgent
from src.agents.routing.route_planning_result import RoutePlanningResult, RoutePlanningStatus
from src.calculators.baseline_plan.baseline_plan_evaluator import BaselinePlanScorer
from src.calculators.response_optimization.response_optimization_config import DEFAULT_RANDOM_SEED
from src.calculators.response_optimization.response_plan_scorer import ResponsePlanScorer
from src.database.base import Base
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.models import ResourceStatus, ResponseTarget, ResponseTargetSet, ResponseTargetType
from src.models.fire_event_status import FireEventStatus
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.response_plan import ResponsePlan
from src.models.response_plan_status import ResponsePlanStatus
from src.models.routing import RoutePlanningRun, RouteResult, RouteStatus
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.plan_comparison_repository import PlanComparisonRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.baseline_comparison.baseline_comparison_production_readers import (
    ResponsePlanOptimizedPlanReaderAdapter,
    RoutePlanningRunReaderAdapter,
)
from src.services.baseline_comparison.baseline_comparison_service import BaselineComparisonService
from src.services.baseline_comparison.response_plan_baseline_scorer_adapter import ResponsePlanBaselineScorerAdapter
from src.services.fire_event_lifecycle import FireEventLifecycleService
from src.services.global_planning.global_planning_orchestrator import GlobalPlanningOrchestrator
from src.services.operational.operational_context_service import OperationalContextService
from src.services.response_planning.baseline_comparison_collaborator_adapter import (
    BaselineComparisonCollaboratorAdapter,
)
from src.services.response_planning.planning_effective_state_builder import PlanningEffectiveStateBuilder
from src.services.response_planning.planning_refresh_result import PlanningRefreshResult, PlanningRefreshStatus
from src.services.response_planning.response_optimization_collaborator_adapter import (
    ResponseOptimizationCollaboratorAdapter,
)
from src.services.response_planning.response_planning_refresh_orchestrator import (
    ResponsePlanningRefreshOrchestrator,
)
from src.services.resource_reservation import CrossEventReservedResourceResolver, ResponsePlanActivationService

AS_OF = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
LATER = AS_OF + timedelta(hours=1)


def _build_sqlite_session_factory() -> sessionmaker:
    """A fresh, independent in-memory SQLite database (mirrors conftest.py's
    sqlite_session_factory fixture) - built as a plain function, not a
    fixture, so a single test can hold TWO fully independent databases at
    once (Task 21's parity test needs this)."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Test-only routing collaborator (per-fire-event resource pools)
# ---------------------------------------------------------------------------


class PersistingFakeRoutingCollaborator:
    """Skips Dijkstra/OSM but performs REAL persistence via RoutePlanningRepository."""

    def __init__(self, response_target_repository, route_planning_repository, resource_ids_by_fire_event):
        self._response_target_repository = response_target_repository
        self._route_planning_repository = route_planning_repository
        self._resource_ids_by_fire_event = resource_ids_by_fire_event
        self.calls = []

    def plan(self, *, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        stored_target_set = self._response_target_repository.get_latest_for_event_as_of(fire_event_id, as_of)
        active_target = next(
            t for t in stored_target_set.targets if t.target.target_type is ResponseTargetType.ACTIVE_FIRE
        )
        resource_ids = self._resource_ids_by_fire_event[fire_event_id]
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
            for resource_id in resource_ids
        )
        run = RoutePlanningRun(
            fire_event_id=fire_event_id,
            response_target_set_id=stored_target_set.id,
            planned_at=as_of,
            methodology="ECOGUARD_ROUTING_DIJKSTRA",
            methodology_version="1.0",
            resource_ids=resource_ids,
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


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def seed_fire_event(session_factory, *, latitude=32.731, longitude=35.046, status="confirmed") -> int:
    session = session_factory()
    event_row = FireEventDB(
        latitude=latitude,
        longitude=longitude,
        detected_at=AS_OF - timedelta(hours=1),
        updated_at=AS_OF - timedelta(minutes=5),
        status=status,
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event_row)
    session.commit()
    fire_event_id = event_row.id
    session.close()
    return fire_event_id


def seed_station_and_resources(session_factory, resource_ids, station_id) -> None:
    session = session_factory()
    if session.get(FireStationDB, station_id) is None:
        session.add(FireStationDB(id=station_id, name=station_id, latitude=32.7, longitude=35.0))
        session.flush()
    for resource_id in resource_ids:
        session.add(FirefightingResourceDB(id=resource_id, station_id=station_id, status=ResourceStatus.AVAILABLE))
    session.commit()
    session.close()


def seed_target_set(response_target_repository, fire_event_id, as_of=AS_OF, priority_score=150.0, latitude=32.731, longitude=35.046):
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
                    latitude=latitude,
                    longitude=longitude,
                    priority_score=priority_score,
                ),
            ),
        )
    )


def persist_minimal_plan(session_factory, response_plan_repository, fire_event_id) -> tuple[int, int]:
    """A real, zero-action ResponsePlan (+ its target-set/route-run parents)
    for tests that need a genuinely existing plan id to reference in a
    canned PlanningRefreshResult. Returns (response_plan_id, route_planning_run_id)."""
    session = session_factory()
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id, generated_at=AS_OF, methodology="m", methodology_version="1.0"
    )
    session.add(target_set)
    session.flush()
    route_run = RoutePlanningRunDB(
        fire_event_id=fire_event_id,
        response_target_set_id=target_set.id,
        planned_at=AS_OF,
        methodology="m",
        methodology_version="1.0",
        resource_ids=[],
    )
    session.add(route_run)
    session.commit()
    target_set_id, route_run_id = target_set.id, route_run.id
    session.close()

    stored = response_plan_repository.save(
        ResponsePlan(
            fire_event_id=fire_event_id,
            response_target_set_id=target_set_id,
            route_planning_run_id=route_run_id,
            generated_at=AS_OF,
            status=ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS,
            methodology="m",
            methodology_version="1.0",
            random_seed=1,
            actions=(),
            uncovered_target_ids=(),
            plan_score=0.0,
            coverage_score=0.0,
            average_eta_seconds=None,
        )
    )
    return stored.id, route_run_id


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class Wiring:
    """Bundles every repository/collaborator needed for one real child
    orchestrator plus one real GlobalPlanningOrchestrator wrapping it."""

    def __init__(self, session_factory, resource_ids_by_fire_event):
        self.session_factory = session_factory
        self.response_target_repository = ResponseTargetRepository(session_factory)
        self.route_planning_repository = RoutePlanningRepository(session_factory)
        self.response_plan_repository = ResponsePlanRepository(session_factory)
        self.plan_comparison_repository = PlanComparisonRepository(session_factory)
        self.sidecar_repository = ResponsePlanPlanningStateRepository(session_factory)
        self.fire_event_repository = FireEventRepository(session_factory)
        self.resource_commitment_repository = ResourceCommitmentRepository(session_factory)
        self.global_planning_run_repository = GlobalPlanningRunRepository(session_factory)

        self.routing_collaborator = PersistingFakeRoutingCollaborator(
            self.response_target_repository, self.route_planning_repository, resource_ids_by_fire_event
        )
        self.optimization_collaborator = ResponseOptimizationCollaboratorAdapter(
            optimization_agent=ResponseOptimizationAgent(repository=self.response_plan_repository),
            route_planning_repository=self.route_planning_repository,
            response_target_repository=self.response_target_repository,
        )
        self.activation_service = ResponsePlanActivationService(
            response_plan_repository=self.response_plan_repository,
            resource_commitment_repository=self.resource_commitment_repository,
            session_factory=session_factory,
        )
        self.operational_context_service = OperationalContextService(
            cross_event_reserved_resource_resolver=CrossEventReservedResourceResolver(
                fire_event_repository=self.fire_event_repository,
                response_plan_repository=self.response_plan_repository,
                resource_commitment_repository=self.resource_commitment_repository,
            ),
        )
        self.planning_state_builder = PlanningEffectiveStateBuilder(
            response_target_repository=self.response_target_repository,
            operational_context_service=self.operational_context_service,
        )

    def baseline_collaborator(self):
        service = BaselineComparisonService(
            optimized_plan_reader=ResponsePlanOptimizedPlanReaderAdapter(self.response_plan_repository),
            route_planning_run_reader=RoutePlanningRunReaderAdapter(self.route_planning_repository),
            scorer=ResponsePlanBaselineScorerAdapter(ResponsePlanScorer()),
            response_target_set_reader=self.response_target_repository,
            plan_comparison_repository=self.plan_comparison_repository,
        )
        return BaselineComparisonCollaboratorAdapter(service, self.plan_comparison_repository)

    def child_orchestrator(self) -> ResponsePlanningRefreshOrchestrator:
        return ResponsePlanningRefreshOrchestrator(
            planning_state_builder=self.planning_state_builder,
            routing_collaborator=self.routing_collaborator,
            optimization_collaborator=self.optimization_collaborator,
            baseline_collaborator=self.baseline_collaborator(),
            activation_collaborator=self.activation_service,
            fire_event_repository=self.fire_event_repository,
            response_plan_repository=self.response_plan_repository,
            response_plan_planning_state_repository=self.sidecar_repository,
            resource_commitment_repository=self.resource_commitment_repository,
            plan_comparison_repository=self.plan_comparison_repository,
            optimization_seed=DEFAULT_RANDOM_SEED,
        )

    def global_orchestrator(self, child_orchestrator=None) -> GlobalPlanningOrchestrator:
        return GlobalPlanningOrchestrator(
            child_orchestrator=child_orchestrator or self.child_orchestrator(),
            fire_event_repository=self.fire_event_repository,
            resource_commitment_repository=self.resource_commitment_repository,
            response_plan_repository=self.response_plan_repository,
            planning_state_builder=self.planning_state_builder,
            global_planning_run_repository=self.global_planning_run_repository,
        )


class FakeChildOrchestrator:
    """Canned PlanningRefreshResult per fire_event_id - for tests needing
    deterministic FAILED/NO_OP/PLANNED outcomes without a real GA run.

    A value may be a PlanningRefreshResult (returned as-is) or a zero-arg
    callable returning one - the callable form lets a test create the
    "new" ResponsePlan row at refresh() time, i.e. AFTER
    GlobalPlanningOrchestrator has already captured that event's
    "before" latest-plan-id, exactly mirroring how a real child
    orchestrator only creates the row during its own refresh() call.
    """

    def __init__(self, results_by_fire_event_id):
        self._results_by_fire_event_id = results_by_fire_event_id
        self.calls = []

    def refresh(self, *, fire_event_id, as_of):
        self.calls.append(fire_event_id)
        result_or_factory = self._results_by_fire_event_id[fire_event_id]
        return result_or_factory() if callable(result_or_factory) else result_or_factory


class SideEffectBeforeRefreshChildOrchestrator:
    """Wraps a REAL child orchestrator; runs a side effect just before
    delegating to it for one specific fire_event_id (Tasks 15/16's
    mid-run scenarios need something to actually happen mid-cycle)."""

    def __init__(self, real_child_orchestrator, side_effects_by_fire_event_id):
        self._real = real_child_orchestrator
        self._side_effects = side_effects_by_fire_event_id

    def refresh(self, *, fire_event_id, as_of):
        side_effect = self._side_effects.get(fire_event_id)
        if side_effect is not None:
            side_effect()
        return self._real.refresh(fire_event_id=fire_event_id, as_of=as_of)


# ---------------------------------------------------------------------------
# _determine_final_status (Task 13 - pure function, tested directly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "member_statuses,expected",
    [
        ((), GlobalPlanningRunStatus.NO_ACTIVE_EVENTS),
        ((GlobalPlanningRunEventStatus.PLANNED,), GlobalPlanningRunStatus.COMPLETED),
        (
            (GlobalPlanningRunEventStatus.PLANNED, GlobalPlanningRunEventStatus.NO_OP),
            GlobalPlanningRunStatus.COMPLETED,
        ),
        (
            (GlobalPlanningRunEventStatus.PLANNED, GlobalPlanningRunEventStatus.SKIPPED_INACTIVE),
            GlobalPlanningRunStatus.COMPLETED,
        ),
        (
            (GlobalPlanningRunEventStatus.PLANNED, GlobalPlanningRunEventStatus.FAILED),
            GlobalPlanningRunStatus.PARTIAL,
        ),
        (
            (GlobalPlanningRunEventStatus.FAILED, GlobalPlanningRunEventStatus.NO_OP, GlobalPlanningRunEventStatus.PLANNED),
            GlobalPlanningRunStatus.PARTIAL,
        ),
        ((GlobalPlanningRunEventStatus.FAILED,), GlobalPlanningRunStatus.FAILED),
        (
            (GlobalPlanningRunEventStatus.FAILED, GlobalPlanningRunEventStatus.FAILED),
            GlobalPlanningRunStatus.FAILED,
        ),
    ],
)
def test_determine_final_status(member_statuses, expected):
    assert GlobalPlanningOrchestrator._determine_final_status(member_statuses) is expected


# ---------------------------------------------------------------------------
# Task 25 - Empty run
# ---------------------------------------------------------------------------


def test_empty_run_persists_no_active_events_status_with_no_members():
    session_factory = _build_sqlite_session_factory()
    wiring = Wiring(session_factory, resource_ids_by_fire_event={})

    result = wiring.global_orchestrator().run(as_of=AS_OF, trigger="manual")

    assert result.status is GlobalPlanningRunStatus.NO_ACTIVE_EVENTS
    assert result.event_results == ()
    stored_run = wiring.global_planning_run_repository.get_by_id(result.global_planning_run_id)
    assert stored_run.run.status is GlobalPlanningRunStatus.NO_ACTIVE_EVENTS
    assert wiring.global_planning_run_repository.get_members(result.global_planning_run_id) == ()


# ---------------------------------------------------------------------------
# Task 21 - Single-incident parity
# ---------------------------------------------------------------------------


def test_single_incident_parity_with_direct_child_orchestrator_call():
    # Database 1: call the child orchestrator directly.
    direct_session_factory = _build_sqlite_session_factory()
    direct_wiring = Wiring(direct_session_factory, resource_ids_by_fire_event={})
    direct_event_id = seed_fire_event(direct_session_factory)
    seed_station_and_resources(direct_session_factory, ["truck-1"], "station-1")
    direct_wiring.routing_collaborator._resource_ids_by_fire_event[direct_event_id] = ["truck-1"]
    seed_target_set(direct_wiring.response_target_repository, direct_event_id)

    direct_result = direct_wiring.child_orchestrator().refresh(fire_event_id=direct_event_id, as_of=AS_OF)
    direct_plan = direct_wiring.response_plan_repository.get_by_id(direct_result.response_plan_id)

    # Database 2: identical setup, but driven through GlobalPlanningOrchestrator.
    global_session_factory = _build_sqlite_session_factory()
    global_wiring = Wiring(global_session_factory, resource_ids_by_fire_event={})
    global_event_id = seed_fire_event(global_session_factory)
    seed_station_and_resources(global_session_factory, ["truck-1"], "station-1")
    global_wiring.routing_collaborator._resource_ids_by_fire_event[global_event_id] = ["truck-1"]
    seed_target_set(global_wiring.response_target_repository, global_event_id)

    global_result = global_wiring.global_orchestrator().run(as_of=AS_OF, trigger="manual")

    assert global_result.status is GlobalPlanningRunStatus.COMPLETED
    (member,) = global_result.event_results
    assert member.planning_status is GlobalPlanningRunEventStatus.PLANNED
    global_plan = global_wiring.response_plan_repository.get_by_id(member.response_plan_id)

    # Same targets/routes/GA/scoring/commitment semantics -> byte-identical
    # plan CONTENT (actions, score, coverage) between the two paths.
    assert direct_plan.plan.actions == global_plan.plan.actions
    assert direct_plan.plan.plan_score == global_plan.plan.plan_score
    assert direct_plan.plan.coverage_score == global_plan.plan.coverage_score
    assert direct_plan.plan.status == global_plan.plan.status
    assert direct_result.status is PlanningRefreshStatus.REFRESHED


# ---------------------------------------------------------------------------
# Task 22 - Multi-incident
# ---------------------------------------------------------------------------


def test_multi_incident_run_creates_one_run_two_members_and_disjoint_commitments():
    session_factory = _build_sqlite_session_factory()
    wiring = Wiring(session_factory, resource_ids_by_fire_event={})
    event_a = seed_fire_event(session_factory, latitude=32.7, longitude=35.0)
    event_b = seed_fire_event(session_factory, latitude=32.8, longitude=35.1)
    seed_station_and_resources(session_factory, ["truck-a"], "station-a")
    seed_station_and_resources(session_factory, ["truck-b"], "station-b")
    wiring.routing_collaborator._resource_ids_by_fire_event[event_a] = ["truck-a"]
    wiring.routing_collaborator._resource_ids_by_fire_event[event_b] = ["truck-b"]
    seed_target_set(wiring.response_target_repository, event_a, latitude=32.7, longitude=35.0)
    seed_target_set(wiring.response_target_repository, event_b, latitude=32.8, longitude=35.1)

    result = wiring.global_orchestrator().run(as_of=AS_OF, trigger="manual")

    assert result.status is GlobalPlanningRunStatus.COMPLETED
    members = wiring.global_planning_run_repository.get_members(result.global_planning_run_id)
    assert len(members) == 2
    assert [m.member.fire_event_id for m in members] == [event_a, event_b]
    assert all(m.member.result_status is GlobalPlanningRunEventStatus.PLANNED for m in members)

    plan_a_id = next(er.response_plan_id for er in result.event_results if er.fire_event_id == event_a)
    plan_b_id = next(er.response_plan_id for er in result.event_results if er.fire_event_id == event_b)
    assert wiring.response_plan_repository.get_global_planning_run_id(plan_a_id) == result.global_planning_run_id
    assert wiring.response_plan_repository.get_global_planning_run_id(plan_b_id) == result.global_planning_run_id

    committed_a = {c.resource_id for c in wiring.resource_commitment_repository.get_for_fire_event(event_a)}
    committed_b = {c.resource_id for c in wiring.resource_commitment_repository.get_for_fire_event(event_b)}
    assert committed_a.isdisjoint(committed_b)


# ---------------------------------------------------------------------------
# Task 23 - NO_OP
# ---------------------------------------------------------------------------


def test_no_op_run_leaves_existing_plan_and_its_global_run_id_untouched():
    session_factory = _build_sqlite_session_factory()
    wiring = Wiring(session_factory, resource_ids_by_fire_event={})
    event_id = seed_fire_event(session_factory)
    seed_station_and_resources(session_factory, ["truck-1"], "station-1")
    wiring.routing_collaborator._resource_ids_by_fire_event[event_id] = ["truck-1"]
    seed_target_set(wiring.response_target_repository, event_id)

    first_result = wiring.global_orchestrator().run(as_of=AS_OF, trigger="manual")
    (first_member,) = first_result.event_results
    original_plan_id = first_member.response_plan_id
    original_run_id = wiring.response_plan_repository.get_global_planning_run_id(original_plan_id)
    assert original_run_id == first_result.global_planning_run_id

    second_result = wiring.global_orchestrator().run(as_of=LATER, trigger="manual")

    assert second_result.status is GlobalPlanningRunStatus.COMPLETED
    (second_member,) = second_result.event_results
    assert second_member.planning_status is GlobalPlanningRunEventStatus.NO_OP
    assert second_member.response_plan_id == original_plan_id  # no duplicate plan created
    # The plan's ORIGINAL global_planning_run_id is unchanged - never re-stamped by the NO_OP run.
    assert wiring.response_plan_repository.get_global_planning_run_id(original_plan_id) == original_run_id
    assert original_run_id != second_result.global_planning_run_id
    assert len(wiring.response_plan_repository.get_for_fire_event(event_id)) == 1


# ---------------------------------------------------------------------------
# Task 24 - Partial failure
# ---------------------------------------------------------------------------


def test_partial_failure_run_status_is_partial_with_all_members_persisted():
    session_factory = _build_sqlite_session_factory()
    wiring = Wiring(session_factory, resource_ids_by_fire_event={})
    event_a = seed_fire_event(session_factory)
    event_b = seed_fire_event(session_factory)
    event_c = seed_fire_event(session_factory)

    # C's plan already exists BEFORE this run starts (simulating a NO_OP
    # that reuses it); A's plan is created lazily, inside the fake's
    # refresh() call, so its id genuinely differs from A's "before" state
    # (which is None - no prior plan) - see FakeChildOrchestrator's docstring.
    plan_c_id, run_c_id = persist_minimal_plan(session_factory, wiring.response_plan_repository, event_c)
    plan_a_ids: dict[str, int] = {}

    def _create_plan_a_and_return_refreshed():
        plan_a_id, run_a_id = persist_minimal_plan(session_factory, wiring.response_plan_repository, event_a)
        plan_a_ids["id"] = plan_a_id
        return PlanningRefreshResult(
            status=PlanningRefreshStatus.REFRESHED,
            fire_event_id=event_a,
            route_planning_run_id=run_a_id,
            response_plan_id=plan_a_id,
            comparison_id=1,
        )

    fake_child = FakeChildOrchestrator(
        {
            event_a: _create_plan_a_and_return_refreshed,
            event_b: PlanningRefreshResult(
                status=PlanningRefreshStatus.FAILED,
                fire_event_id=event_b,
                route_planning_run_id=None,
                response_plan_id=None,
                comparison_id=None,
                error="synthetic failure for this test",
            ),
            event_c: PlanningRefreshResult(
                status=PlanningRefreshStatus.NO_OP,
                fire_event_id=event_c,
                route_planning_run_id=run_c_id,
                response_plan_id=plan_c_id,
                comparison_id=2,
            ),
        }
    )

    result = wiring.global_orchestrator(child_orchestrator=fake_child).run(as_of=AS_OF, trigger="manual")

    assert result.status is GlobalPlanningRunStatus.PARTIAL
    members = wiring.global_planning_run_repository.get_members(result.global_planning_run_id)
    assert len(members) == 3
    statuses_by_event = {m.member.fire_event_id: m.member.result_status for m in members}
    assert statuses_by_event[event_a] is GlobalPlanningRunEventStatus.PLANNED
    assert statuses_by_event[event_b] is GlobalPlanningRunEventStatus.FAILED
    assert statuses_by_event[event_c] is GlobalPlanningRunEventStatus.NO_OP

    failed_member = next(m for m in members if m.member.fire_event_id == event_b)
    assert failed_member.member.error_code == "child_planning_failed"
    assert "synthetic failure" not in (failed_member.member.error_code or "")  # never the raw message

    # A's plan was genuinely NEW this run (no prior plan existed) -> stamped.
    assert (
        wiring.response_plan_repository.get_global_planning_run_id(plan_a_ids["id"])
        == result.global_planning_run_id
    )
    # C's plan was NOT created this run (its before/after plan id are identical) -> untouched.
    assert wiring.response_plan_repository.get_global_planning_run_id(plan_c_id) is None


def test_failed_run_status_when_every_member_fails():
    session_factory = _build_sqlite_session_factory()
    wiring = Wiring(session_factory, resource_ids_by_fire_event={})
    event_a = seed_fire_event(session_factory)
    event_b = seed_fire_event(session_factory)
    fake_child = FakeChildOrchestrator(
        {
            event_a: PlanningRefreshResult(
                status=PlanningRefreshStatus.FAILED, fire_event_id=event_a,
                route_planning_run_id=None, response_plan_id=None, comparison_id=None, error="x",
            ),
            event_b: PlanningRefreshResult(
                status=PlanningRefreshStatus.FAILED, fire_event_id=event_b,
                route_planning_run_id=None, response_plan_id=None, comparison_id=None, error="y",
            ),
        }
    )

    result = wiring.global_orchestrator(child_orchestrator=fake_child).run(as_of=AS_OF, trigger="manual")

    assert result.status is GlobalPlanningRunStatus.FAILED


# ---------------------------------------------------------------------------
# Task 15 - New event mid-run
# ---------------------------------------------------------------------------


def test_new_event_created_mid_run_is_not_retroactively_added_to_the_snapshot():
    session_factory = _build_sqlite_session_factory()
    wiring = Wiring(session_factory, resource_ids_by_fire_event={})
    event_a = seed_fire_event(session_factory)
    event_b = seed_fire_event(session_factory)
    seed_station_and_resources(session_factory, ["truck-a", "truck-b"], "station-1")
    wiring.routing_collaborator._resource_ids_by_fire_event[event_a] = ["truck-a"]
    wiring.routing_collaborator._resource_ids_by_fire_event[event_b] = ["truck-b"]
    seed_target_set(wiring.response_target_repository, event_a)
    seed_target_set(wiring.response_target_repository, event_b, latitude=32.8, longitude=35.1)

    new_event_holder = {}

    def _create_event_c():
        new_event_holder["id"] = seed_fire_event(session_factory, latitude=32.9, longitude=35.2)

    side_effecting_child = SideEffectBeforeRefreshChildOrchestrator(
        wiring.child_orchestrator(), {event_a: _create_event_c}
    )

    first_result = wiring.global_orchestrator(child_orchestrator=side_effecting_child).run(
        as_of=AS_OF, trigger="manual"
    )

    member_event_ids = {
        m.member.fire_event_id
        for m in wiring.global_planning_run_repository.get_members(first_result.global_planning_run_id)
    }
    assert member_event_ids == {event_a, event_b}
    assert new_event_holder["id"] not in member_event_ids

    # The NEXT run picks up C.
    seed_station_and_resources(session_factory, ["truck-c"], "station-2")
    wiring.routing_collaborator._resource_ids_by_fire_event[new_event_holder["id"]] = ["truck-c"]
    seed_target_set(wiring.response_target_repository, new_event_holder["id"], latitude=32.9, longitude=35.2)

    second_result = wiring.global_orchestrator().run(as_of=LATER, trigger="manual")
    second_member_ids = {
        m.member.fire_event_id
        for m in wiring.global_planning_run_repository.get_members(second_result.global_planning_run_id)
    }
    assert second_member_ids == {event_a, event_b, new_event_holder["id"]}


# ---------------------------------------------------------------------------
# Task 16 - Event resolves mid-run
# ---------------------------------------------------------------------------


def test_event_resolved_mid_run_is_recorded_as_skipped_inactive():
    session_factory = _build_sqlite_session_factory()
    wiring = Wiring(session_factory, resource_ids_by_fire_event={})
    event_a = seed_fire_event(session_factory)
    event_b = seed_fire_event(session_factory)
    seed_station_and_resources(session_factory, ["truck-a", "truck-b"], "station-1")
    wiring.routing_collaborator._resource_ids_by_fire_event[event_a] = ["truck-a"]
    wiring.routing_collaborator._resource_ids_by_fire_event[event_b] = ["truck-b"]
    seed_target_set(wiring.response_target_repository, event_a)
    seed_target_set(wiring.response_target_repository, event_b, latitude=32.8, longitude=35.1)

    lifecycle_service = FireEventLifecycleService(
        fire_event_repository=wiring.fire_event_repository,
        resource_commitment_repository=wiring.resource_commitment_repository,
        session_factory=session_factory,
    )

    def _resolve_b():
        lifecycle_service.resolve_event(event_b, as_of=AS_OF)

    side_effecting_child = SideEffectBeforeRefreshChildOrchestrator(
        wiring.child_orchestrator(), {event_b: _resolve_b}
    )

    result = wiring.global_orchestrator(child_orchestrator=side_effecting_child).run(as_of=AS_OF, trigger="manual")

    # SKIPPED_INACTIVE is not a failure (Task 13's rule counts FAILED
    # members only) - zero FAILED members among A/B means COMPLETED.
    assert result.status is GlobalPlanningRunStatus.COMPLETED
    members = {
        m.member.fire_event_id: m.member.result_status
        for m in wiring.global_planning_run_repository.get_members(result.global_planning_run_id)
    }
    assert members[event_a] is GlobalPlanningRunEventStatus.PLANNED
    assert members[event_b] is GlobalPlanningRunEventStatus.SKIPPED_INACTIVE
    # B is still in the audit trail even though it was active at snapshot time.
    assert event_b in members


# ---------------------------------------------------------------------------
# Task 26 - Traceability
# ---------------------------------------------------------------------------


def test_traceability_queries_do_not_require_timestamp_inference():
    session_factory = _build_sqlite_session_factory()
    wiring = Wiring(session_factory, resource_ids_by_fire_event={})
    event_id = seed_fire_event(session_factory)
    seed_station_and_resources(session_factory, ["truck-1"], "station-1")
    wiring.routing_collaborator._resource_ids_by_fire_event[event_id] = ["truck-1"]
    seed_target_set(wiring.response_target_repository, event_id)

    result = wiring.global_orchestrator().run(as_of=AS_OF, trigger="manual")
    (member,) = result.event_results

    # "Which global cycle created ResponsePlan #X?"
    assert wiring.response_plan_repository.get_global_planning_run_id(member.response_plan_id) == (
        result.global_planning_run_id
    )
    # "Which FireEvents were considered in GlobalPlanningRun #Y?"
    members = wiring.global_planning_run_repository.get_members(result.global_planning_run_id)
    assert {m.member.fire_event_id for m in members} == {event_id}
    # "What happened to each event in that run?"
    assert members[0].member.result_status is GlobalPlanningRunEventStatus.PLANNED
