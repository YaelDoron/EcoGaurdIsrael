"""Tests for ResponsePlanDetailsService (Epic 5, US 5.5, Task 2), using fakes only.

All injected dependencies are hand-written fakes exposing only the read
methods the service actually calls - no SQLite, no real database, and (by
construction) no save/update/delete method for the service to accidentally
invoke.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from src.calculators.baseline_plan.baseline_plan_comparison_calculator import PlanComparison
from src.models import (
    FirefightingResource,
    ResourceStatus,
    ResponseAction,
    ResponsePlan,
    ResponsePlanStatus,
    ResponseTarget,
    ResponseTargetSet,
    ResponseTargetType,
)
from src.models.routing import RouteResult, RouteStatus
from src.repositories.plan_comparison_repository import StoredPlanComparison
from src.repositories.response_plan_repository import StoredResponsePlan
from src.repositories.response_target_repository import StoredResponseTarget, StoredResponseTargetSet
from src.repositories.route_planning_repository import StoredRoutePlanningRun
from src.services.response_planning.response_plan_details_service import ResponsePlanDetailsService

FIRE_EVENT_ID = 42
AS_OF = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fakes (read-only by construction)
# ---------------------------------------------------------------------------


class FakeCurrentResponsePlanResolver:
    def __init__(self, current_plan: StoredResponsePlan | None = None):
        self.current_plan = current_plan
        self.calls: list[int] = []

    def resolve(self, *, fire_event_id: int) -> StoredResponsePlan | None:
        self.calls.append(fire_event_id)
        if self.current_plan is not None and self.current_plan.plan.fire_event_id == fire_event_id:
            return self.current_plan
        return None


class FakeResponsePlanRepository:
    def __init__(self, plans: tuple[StoredResponsePlan, ...] = ()):
        self.plans = {p.id: p for p in plans}

    def get_by_id(self, plan_id: int) -> StoredResponsePlan | None:
        return self.plans.get(plan_id)


class FakeRoutePlanningRepository:
    def __init__(self, runs: tuple[StoredRoutePlanningRun, ...] = ()):
        self.runs = {r.id: r for r in runs}

    def get_by_id(self, run_id: int) -> StoredRoutePlanningRun | None:
        return self.runs.get(run_id)


class FakeResponseTargetRepository:
    def __init__(self, target_sets: tuple[StoredResponseTargetSet, ...] = ()):
        self.target_sets = {t.id: t for t in target_sets}

    def get_by_id(self, target_set_id: int) -> StoredResponseTargetSet | None:
        return self.target_sets.get(target_set_id)


class FakeFirefightingResourceRepository:
    def __init__(self, resources: tuple[FirefightingResource, ...] = ()):
        self.resources = {str(r.id): r for r in resources}

    def get_by_id(self, resource_id) -> FirefightingResource | None:
        return self.resources.get(str(resource_id))


@dataclass(frozen=True)
class FakeStationRow:
    id: str


class FakeFireStationRepository:
    def __init__(self, station_ids: tuple[str, ...] = ()):
        self.station_ids = station_ids

    def get_all_stations(self) -> list[FakeStationRow]:
        return [FakeStationRow(id=station_id) for station_id in self.station_ids]


class FakePlanComparisonRepository:
    def __init__(self, comparisons: tuple[StoredPlanComparison, ...] = ()):
        self.comparisons = comparisons

    def list_for_fire_event(self, fire_event_id: int) -> tuple[StoredPlanComparison, ...]:
        return tuple(c for c in self.comparisons if c.comparison.fire_event_id == fire_event_id)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_target(target_id: int, response_target_set_id: int = 200) -> StoredResponseTarget:
    return StoredResponseTarget(
        id=target_id,
        target_order=0,
        target=ResponseTarget(
            fire_event_id=FIRE_EVENT_ID,
            target_type=ResponseTargetType.ACTIVE_FIRE,
            latitude=32.0,
            longitude=34.8,
            priority_score=150.0,
        ),
    )


def make_target_set(target_set_id: int = 200, target_ids: tuple[int, ...] = (10,)) -> StoredResponseTargetSet:
    stored_targets = tuple(make_target(target_id, target_set_id) for target_id in target_ids)
    target_set = ResponseTargetSet(
        fire_event_id=FIRE_EVENT_ID,
        generated_at=AS_OF,
        methodology="PREDICTIVE_ACTIVE_FIRE",
        methodology_version="1.0",
        targets=tuple(st.target for st in stored_targets),
    )
    return StoredResponseTargetSet(id=target_set_id, target_set=target_set, targets=stored_targets)


def make_reachable_route(resource_id: str = "TRUCK-A", target_id: int = 10) -> RouteResult:
    return RouteResult(
        resource_id=resource_id,
        response_target_id=target_id,
        status=RouteStatus.REACHABLE,
        source_node_id=1,
        target_node_id=2,
        node_path=(1, 2),
        distance_meters=2500.0,
        travel_time_seconds=300.0,
    )


def make_unreachable_route(resource_id: str = "TRUCK-A", target_id: int = 10) -> RouteResult:
    return RouteResult(
        resource_id=resource_id,
        response_target_id=target_id,
        status=RouteStatus.UNREACHABLE,
        source_node_id=1,
        target_node_id=2,
        node_path=(),
        distance_meters=None,
        travel_time_seconds=None,
    )


def make_route_run(run_id: int = 300, routes: tuple[RouteResult, ...] = ()) -> StoredRoutePlanningRun:
    from src.models.routing import RoutePlanningRun, StoredRouteResult

    run = RoutePlanningRun(
        fire_event_id=FIRE_EVENT_ID,
        response_target_set_id=200,
        planned_at=AS_OF,
        methodology="DIJKSTRA",
        methodology_version="1.0",
        resource_ids=tuple({route.resource_id for route in routes}),
        routes=routes,
    )
    stored_routes = tuple(
        StoredRouteResult(id=1000 + i, route_result=route) for i, route in enumerate(routes)
    )
    return StoredRoutePlanningRun(id=run_id, run=run, routes=stored_routes)


def make_resource(resource_id: str = "TRUCK-A", station_id: str = "STATION-1") -> FirefightingResource:
    return FirefightingResource(id=resource_id, station_id=station_id, status=ResourceStatus.ASSIGNED)


def make_plan(
    plan_id: int = 1,
    *,
    actions: tuple[ResponseAction, ...] = (ResponseAction("TRUCK-A", 10, 100),),
    uncovered_target_ids: tuple[int, ...] = (),
    response_target_set_id: int = 200,
    route_planning_run_id: int = 300,
) -> StoredResponsePlan:
    if not uncovered_target_ids:
        status = ResponsePlanStatus.COMPLETE
    elif actions:
        status = ResponsePlanStatus.PARTIAL
    else:
        status = ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS
    plan = ResponsePlan(
        fire_event_id=FIRE_EVENT_ID,
        response_target_set_id=response_target_set_id,
        route_planning_run_id=route_planning_run_id,
        generated_at=AS_OF,
        status=status,
        methodology="GENETIC_RESOURCE_ALLOCATION",
        methodology_version="1.0",
        random_seed=42,
        actions=actions,
        uncovered_target_ids=uncovered_target_ids,
        plan_score=90.0,
        coverage_score=1.0,
        average_eta_seconds=300.0 if actions else None,
    )
    return StoredResponsePlan(id=plan_id, plan=plan)


def make_comparison(plan_id: int = 1) -> StoredPlanComparison:
    comparison = PlanComparison(
        fire_event_id=FIRE_EVENT_ID,
        optimized_plan_id=plan_id,
        route_planning_run_id=300,
        response_target_set_id=200,
        optimized_score=90.0,
        baseline_score=70.0,
        optimized_coverage_score=1.0,
        baseline_coverage_score=0.8,
        optimized_average_eta_seconds=300.0,
        baseline_average_eta_seconds=400.0,
        score_difference=20.0,
        improvement_percentage=28.5,
    )
    return StoredPlanComparison(id=1, comparison=comparison)


def make_service(
    *,
    current_plan: StoredResponsePlan | None = None,
    plans: tuple[StoredResponsePlan, ...] = (),
    target_sets: tuple[StoredResponseTargetSet, ...] = (),
    route_runs: tuple[StoredRoutePlanningRun, ...] = (),
    resources: tuple[FirefightingResource, ...] = (),
    station_ids: tuple[str, ...] = ("STATION-1",),
    comparisons: tuple[StoredPlanComparison, ...] = (),
) -> ResponsePlanDetailsService:
    return ResponsePlanDetailsService(
        current_response_plan_resolver=FakeCurrentResponsePlanResolver(current_plan),
        response_plan_repository=FakeResponsePlanRepository(plans),
        route_planning_repository=FakeRoutePlanningRepository(route_runs),
        response_target_repository=FakeResponseTargetRepository(target_sets),
        firefighting_resource_repository=FakeFirefightingResourceRepository(resources),
        fire_station_repository=FakeFireStationRepository(station_ids),
        plan_comparison_repository=FakePlanComparisonRepository(comparisons),
    )


# ---------------------------------------------------------------------------
# 1. Current plan, full data
# ---------------------------------------------------------------------------


def test_get_current_plan_details_returns_full_assembled_details():
    plan = make_plan()
    service = make_service(
        current_plan=plan,
        plans=(plan,),
        target_sets=(make_target_set(),),
        route_runs=(make_route_run(routes=(make_reachable_route(),)),),
        resources=(make_resource(),),
        comparisons=(make_comparison(plan.id),),
    )

    details = service.get_current_plan_details(FIRE_EVENT_ID)

    assert details is not None
    assert details.plan_id == plan.id
    assert details.fire_event_id == FIRE_EVENT_ID
    assert details.is_current is True
    assert details.uncovered_target_ids == ()
    assert len(details.actions) == 1

    action = details.actions[0]
    assert action.resource_id == "TRUCK-A"
    assert action.station_id == "STATION-1"
    assert action.response_target_id == 10
    assert action.target_type == ResponseTargetType.ACTIVE_FIRE.value
    assert action.target_priority == 150.0
    assert action.eta_seconds == 300.0
    assert action.route_distance_meters == 2500.0
    assert action.node_path == (1, 2)

    assert details.baseline_comparison is not None
    assert details.baseline_comparison.baseline_score == 70.0
    assert details.baseline_comparison.improvement_percentage == 28.5


# ---------------------------------------------------------------------------
# 2. Historical/superseded plan by id
# ---------------------------------------------------------------------------


def test_get_plan_details_by_id_marks_superseded_plan_as_not_current():
    historical = make_plan(plan_id=1)
    current = make_plan(plan_id=2)
    service = make_service(
        current_plan=current,
        plans=(historical, current),
        target_sets=(make_target_set(),),
        route_runs=(make_route_run(routes=(make_reachable_route(),)),),
        resources=(make_resource(),),
    )

    details = service.get_plan_details_by_id(historical.id)

    assert details is not None
    assert details.plan_id == historical.id
    assert details.is_current is False


def test_get_plan_details_by_id_marks_current_plan_as_current():
    current = make_plan(plan_id=2)
    service = make_service(
        current_plan=current,
        plans=(current,),
        target_sets=(make_target_set(),),
        route_runs=(make_route_run(routes=(make_reachable_route(),)),),
        resources=(make_resource(),),
    )

    details = service.get_plan_details_by_id(current.id)

    assert details is not None
    assert details.is_current is True


# ---------------------------------------------------------------------------
# 3. Missing plan / event
# ---------------------------------------------------------------------------


def test_get_current_plan_details_returns_none_when_no_current_plan_exists():
    service = make_service(current_plan=None)

    assert service.get_current_plan_details(FIRE_EVENT_ID) is None


def test_get_plan_details_by_id_returns_none_when_plan_does_not_exist():
    service = make_service(plans=())

    assert service.get_plan_details_by_id(999) is None


# ---------------------------------------------------------------------------
# 4. Missing baseline comparison
# ---------------------------------------------------------------------------


def test_missing_baseline_comparison_yields_none_not_a_crash():
    plan = make_plan()
    service = make_service(
        current_plan=plan,
        plans=(plan,),
        target_sets=(make_target_set(),),
        route_runs=(make_route_run(routes=(make_reachable_route(),)),),
        resources=(make_resource(),),
        comparisons=(),
    )

    details = service.get_current_plan_details(FIRE_EVENT_ID)

    assert details is not None
    assert details.baseline_comparison is None


# ---------------------------------------------------------------------------
# 5. Unreachable route metrics
# ---------------------------------------------------------------------------


def test_unreachable_route_yields_none_eta_distance_and_path():
    plan = make_plan()
    service = make_service(
        current_plan=plan,
        plans=(plan,),
        target_sets=(make_target_set(),),
        route_runs=(make_route_run(routes=(make_unreachable_route(),)),),
        resources=(make_resource(),),
    )

    details = service.get_current_plan_details(FIRE_EVENT_ID)

    assert details is not None
    action = details.actions[0]
    assert action.eta_seconds is None
    assert action.route_distance_meters is None
    assert action.node_path is None


# ---------------------------------------------------------------------------
# Graceful degradation for missing snapshots
# ---------------------------------------------------------------------------


def test_missing_response_target_set_yields_empty_actions_not_a_crash():
    plan = make_plan()
    service = make_service(
        current_plan=plan,
        plans=(plan,),
        target_sets=(),
        route_runs=(make_route_run(routes=(make_reachable_route(),)),),
        resources=(make_resource(),),
    )

    details = service.get_current_plan_details(FIRE_EVENT_ID)

    assert details is not None
    assert details.actions == ()


def test_missing_route_planning_run_still_returns_actions_with_none_metrics():
    plan = make_plan()
    service = make_service(
        current_plan=plan,
        plans=(plan,),
        target_sets=(make_target_set(),),
        route_runs=(),
        resources=(make_resource(),),
    )

    details = service.get_current_plan_details(FIRE_EVENT_ID)

    assert details is not None
    assert len(details.actions) == 1
    assert details.actions[0].eta_seconds is None
    assert details.actions[0].node_path is None


def test_missing_resource_skips_that_action_without_crashing():
    plan = make_plan()
    service = make_service(
        current_plan=plan,
        plans=(plan,),
        target_sets=(make_target_set(),),
        route_runs=(make_route_run(routes=(make_reachable_route(),)),),
        resources=(),
    )

    details = service.get_current_plan_details(FIRE_EVENT_ID)

    assert details is not None
    assert details.actions == ()


def test_missing_response_target_skips_that_action_without_crashing():
    plan = make_plan()
    service = make_service(
        current_plan=plan,
        plans=(plan,),
        target_sets=(make_target_set(target_ids=(999,)),),
        route_runs=(make_route_run(routes=(make_reachable_route(),)),),
        resources=(make_resource(),),
    )

    details = service.get_current_plan_details(FIRE_EVENT_ID)

    assert details is not None
    assert details.actions == ()


def test_uncovered_target_ids_are_carried_through_verbatim():
    plan = make_plan(actions=(), uncovered_target_ids=(10, 20))
    service = make_service(
        current_plan=plan,
        plans=(plan,),
        target_sets=(make_target_set(),),
        route_runs=(),
        resources=(),
    )

    details = service.get_current_plan_details(FIRE_EVENT_ID)

    assert details is not None
    assert details.uncovered_target_ids == (10, 20)
    assert details.actions == ()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_id", [0, -1, True, "1"])
def test_get_current_plan_details_rejects_invalid_fire_event_id(invalid_id):
    service = make_service()

    with pytest.raises(ValueError):
        service.get_current_plan_details(invalid_id)


@pytest.mark.parametrize("invalid_id", [0, -1, True, "1"])
def test_get_plan_details_by_id_rejects_invalid_plan_id(invalid_id):
    service = make_service()

    with pytest.raises(ValueError):
        service.get_plan_details_by_id(invalid_id)


# ---------------------------------------------------------------------------
# Architecture guard: no calculators, no writes
# ---------------------------------------------------------------------------


def test_service_source_never_calls_a_write_operation():
    import inspect

    source = inspect.getsource(ResponsePlanDetailsService)
    for forbidden in (".save(", ".update_status(", ".set_statuses(", ".delete("):
        assert forbidden not in source


def test_service_module_does_not_import_calculation_engines():
    import ast
    from pathlib import Path

    forbidden_fragments = (
        "genetic_optimizer",
        "response_plan_scorer",
        "chromosome_decoder",
        "initial_population_generator",
        "feasible_route_lookup",
        "Dijkstra",
        "dijkstra",
        "NodeMapping",
        "node_mapping",
        "baseline_plan_calculator",
        "baseline_plan_evaluator",
    )
    path = Path("backend/src/services/response_planning/response_plan_details_service.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)

    assert violations == []
