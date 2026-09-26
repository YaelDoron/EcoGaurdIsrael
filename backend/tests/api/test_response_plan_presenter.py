"""Tests for the US 6.3 Task 4 presentation assembler (`src.api.response_plan_presenter`).

Exercises `ResponsePlanPresenter.present` directly against fakes for every
repository dependency - no real DB/Neon access happens in these tests. Each
fake records the exact id(s) it was called with, so tests can prove the
presenter always reads the *exact* persisted snapshot a `ResponsePlanDetails`
references (its own `plan_id`/`response_target_set_id`/`route_planning_run_id`)
rather than "latest" or "current" data - the whole point of Task 4's
exact-snapshot rule.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

from src.api.response_plan_presenter import ResponsePlanPresenter
from src.api.schemas.response_plans import ResponsePlanActionResponse, ResponsePlanDetailResponse
from src.database.models.fire_station_db import FireStationDB
from src.database.models.graph_node_db import GraphNodeDB
from src.models.response_plan_details import (
    BaselineComparisonDetails,
    OptimizationConfigDetails,
    ResponseActionDetails,
    ResponsePlanDetails,
)
from src.models.response_plan_status import ResponsePlanStatus
from src.models.response_target import ResponseTarget
from src.models.response_target_type import ResponseTargetType
from src.models.routing import RouteResult, RouteStatus

GENERATED_AT = datetime(2026, 9, 17, 13, 20, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fakes - duck-typed stand-ins for the repositories the presenter depends on.
# ---------------------------------------------------------------------------


class FakePlanRow:
    def __init__(self, status: ResponsePlanStatus):
        self.status = status


class FakeStoredResponsePlan:
    def __init__(self, status: ResponsePlanStatus):
        self.plan = FakePlanRow(status)


class FakeResponsePlanRepository:
    def __init__(self, plans_by_id: dict[int, ResponsePlanStatus]):
        self._plans_by_id = plans_by_id
        self.calls: list[int] = []

    def get_by_id(self, plan_id: int):
        self.calls.append(plan_id)
        status = self._plans_by_id.get(plan_id)
        return FakeStoredResponsePlan(status) if status is not None else None


class FakeStoredTarget:
    def __init__(self, id: int, target: ResponseTarget):
        self.id = id
        self.target = target


class FakeStoredTargetSet:
    def __init__(self, targets):
        self.targets = targets


class FakeResponseTargetRepository:
    def __init__(self, target_sets_by_id: dict[int, list[FakeStoredTarget]]):
        self._target_sets_by_id = target_sets_by_id
        self.calls: list[int] = []

    def get_by_id(self, response_target_set_id: int):
        self.calls.append(response_target_set_id)
        targets = self._target_sets_by_id.get(response_target_set_id)
        return FakeStoredTargetSet(targets) if targets is not None else None


class FakeRun:
    def __init__(self, routes, resource_ids):
        self.routes = routes
        self.resource_ids = resource_ids


class FakeStoredRun:
    def __init__(self, run: FakeRun):
        self.run = run


class FakeRoutePlanningRepository:
    def __init__(self, runs_by_id: dict[int, FakeRun]):
        self._runs_by_id = runs_by_id
        self.calls: list[int] = []

    def get_by_id(self, route_planning_run_id: int):
        self.calls.append(route_planning_run_id)
        run = self._runs_by_id.get(route_planning_run_id)
        return FakeStoredRun(run) if run is not None else None


class FakeFireStationRepository:
    def __init__(self, stations: list[FireStationDB]):
        self._stations = stations
        self.calls = 0

    def get_all_stations(self):
        self.calls += 1
        return self._stations


class FakeGraphNodeReadRepository:
    def __init__(self, nodes: dict[int, GraphNodeDB]):
        self._nodes = nodes
        self.calls: list[frozenset] = []

    def get_by_ids(self, node_ids):
        node_ids = list(node_ids)
        self.calls.append(frozenset(node_ids))
        return {node_id: self._nodes[node_id] for node_id in node_ids if node_id in self._nodes}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_action(**overrides) -> ResponseActionDetails:
    values = dict(
        resource_id="engine-1",
        station_id="station-1",
        response_target_id=1,
        target_type="active_fire",
        target_priority=0.75,
        eta_seconds=120.0,
        route_distance_meters=800.0,
        node_path=(1, 2, 3),
    )
    values.update(overrides)
    return ResponseActionDetails(**values)


def make_plan(**overrides) -> ResponsePlanDetails:
    values = dict(
        plan_id=7,
        fire_event_id=3,
        response_target_set_id=9,
        route_planning_run_id=11,
        generated_at=GENERATED_AT,
        methodology="genetic_algorithm",
        methodology_version="1.0.0",
        random_seed=42,
        is_current=True,
        plan_score=95.5,
        coverage_score=0.9,
        average_eta_seconds=150.0,
        actions=(make_action(),),
        uncovered_target_ids=(),
        baseline_comparison=None,
        optimization_config=None,
    )
    values.update(overrides)
    return ResponsePlanDetails(**values)


def make_target(**overrides) -> ResponseTarget:
    values = dict(
        fire_event_id=3,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.5,
        longitude=35.5,
        priority_score=0.85,
    )
    values.update(overrides)
    return ResponseTarget(**values)


def make_route_result(**overrides) -> RouteResult:
    values = dict(
        resource_id="engine-1",
        response_target_id=1,
        status=RouteStatus.REACHABLE,
        source_node_id=1,
        target_node_id=3,
        node_path=(1, 2, 3),
        distance_meters=800.0,
        travel_time_seconds=120.0,
    )
    values.update(overrides)
    return RouteResult(**values)


def make_station(**overrides) -> FireStationDB:
    values = dict(id="station-1", name="Central Station", latitude=32.0, longitude=35.0)
    values.update(overrides)
    return FireStationDB(**values)


def make_node(node_id: int, latitude: float, longitude: float) -> GraphNodeDB:
    return GraphNodeDB(id=node_id, latitude=latitude, longitude=longitude)


def make_presenter(
    *,
    plan_status: ResponsePlanStatus = ResponsePlanStatus.COMPLETE,
    plan_id: int = 7,
    target_set_id: int = 9,
    targets: list[FakeStoredTarget] | None = None,
    run_id: int = 11,
    routes: tuple[RouteResult, ...] = (),
    resource_ids: tuple[str, ...] = ("engine-1",),
    stations: list[FireStationDB] | None = None,
    nodes: dict[int, GraphNodeDB] | None = None,
) -> tuple[ResponsePlanPresenter, dict]:
    plan_repo = FakeResponsePlanRepository({plan_id: plan_status})
    target_repo = FakeResponseTargetRepository({target_set_id: targets if targets is not None else []})
    route_repo = FakeRoutePlanningRepository({run_id: FakeRun(routes=routes, resource_ids=resource_ids)})
    station_repo = FakeFireStationRepository(stations if stations is not None else [make_station()])
    node_repo = FakeGraphNodeReadRepository(nodes if nodes is not None else {})

    presenter = ResponsePlanPresenter(
        response_plan_repository=plan_repo,
        response_target_repository=target_repo,
        route_planning_repository=route_repo,
        fire_station_repository=station_repo,
        graph_node_read_repository=node_repo,
    )
    fakes = {
        "plan_repo": plan_repo,
        "target_repo": target_repo,
        "route_repo": route_repo,
        "station_repo": station_repo,
        "node_repo": node_repo,
    }
    return presenter, fakes


# ---------------------------------------------------------------------------
# 1-2: plan-level status vs is_current
# ---------------------------------------------------------------------------


def test_persisted_plan_status_is_copied_exactly():
    presenter, _ = make_presenter(plan_status=ResponsePlanStatus.PARTIAL)
    details = make_plan(is_current=True)

    response = presenter.present(details)

    assert isinstance(response, ResponsePlanDetailResponse)
    assert response.status == ResponsePlanStatus.PARTIAL


def test_is_current_is_preserved_independently_of_status():
    presenter, _ = make_presenter(plan_status=ResponsePlanStatus.PARTIAL)
    details = make_plan(is_current=False)

    response = presenter.present(details)

    assert response.status == ResponsePlanStatus.PARTIAL
    assert response.is_current is False


# ---------------------------------------------------------------------------
# 3: target coordinates/priority from the exact ResponseTargetSet
# ---------------------------------------------------------------------------


def test_target_coordinates_and_priority_come_from_referenced_target_set():
    target = make_target(latitude=31.111, longitude=34.222, priority_score=0.63)
    presenter, _ = make_presenter(targets=[FakeStoredTarget(id=1, target=target)])
    details = make_plan(actions=(make_action(response_target_id=1),))

    response = presenter.present(details)

    assert isinstance(response.actions[0], ResponsePlanActionResponse)
    mapped_target = response.actions[0].target
    assert mapped_target.response_target_id == 1
    assert mapped_target.latitude == 31.111
    assert mapped_target.longitude == 34.222
    assert mapped_target.priority_score == 0.63
    assert mapped_target.target_type == "active_fire"


# ---------------------------------------------------------------------------
# 4: station name/origin from persisted FireStation
# ---------------------------------------------------------------------------


def test_station_name_and_origin_come_from_persisted_fire_station():
    station = make_station(id="station-9", name="North Station", latitude=33.1, longitude=35.9)
    target = make_target()
    presenter, _ = make_presenter(
        targets=[FakeStoredTarget(id=1, target=target)],
        stations=[station],
    )
    details = make_plan(actions=(make_action(station_id="station-9", response_target_id=1),))

    response = presenter.present(details)

    resource = response.actions[0].resource
    assert resource.station_id == "station-9"
    assert resource.station_name == "North Station"
    assert resource.origin.latitude == 33.1
    assert resource.origin.longitude == 35.9


# ---------------------------------------------------------------------------
# 5-8: route enrichment (ETA/distance/status/node order/coordinate order)
# ---------------------------------------------------------------------------


def test_eta_and_distance_are_exactly_the_persisted_route_result_values():
    route = make_route_result(distance_meters=1234.5, travel_time_seconds=321.0)
    target = make_target()
    presenter, _ = make_presenter(targets=[FakeStoredTarget(id=1, target=target)], routes=(route,))
    details = make_plan(actions=(make_action(response_target_id=1),))

    response = presenter.present(details)

    route_response = response.actions[0].route
    assert route_response.eta_seconds == 321.0
    assert route_response.distance_meters == 1234.5


def test_route_status_is_preserved():
    route = make_route_result(status=RouteStatus.REACHABLE)
    target = make_target()
    presenter, _ = make_presenter(targets=[FakeStoredTarget(id=1, target=target)], routes=(route,))
    details = make_plan(actions=(make_action(response_target_id=1),))

    response = presenter.present(details)

    assert response.actions[0].route.status == RouteStatus.REACHABLE


def test_node_path_order_is_preserved():
    route = make_route_result(node_path=(10, 20, 30), source_node_id=10, target_node_id=30)
    target = make_target()
    presenter, _ = make_presenter(targets=[FakeStoredTarget(id=1, target=target)], routes=(route,))
    details = make_plan(actions=(make_action(response_target_id=1),))

    response = presenter.present(details)

    assert response.actions[0].route.node_path == [10, 20, 30]


def test_graph_node_coordinates_returned_in_exact_node_path_order():
    route = make_route_result(node_path=(10, 20, 30), source_node_id=10, target_node_id=30)
    target = make_target()
    nodes = {
        10: make_node(10, 1.0, 1.0),
        20: make_node(20, 2.0, 2.0),
        30: make_node(30, 3.0, 3.0),
    }
    # Station colocated with node 10 (the route's own origin) - keeps this
    # test about node_path ORDER only; the First-Mile Heuristic Fallback
    # (a station far from its snapped node) has its own dedicated tests below.
    station = make_station(latitude=1.0, longitude=1.0)
    presenter, _ = make_presenter(
        targets=[FakeStoredTarget(id=1, target=target)], routes=(route,), nodes=nodes, stations=[station]
    )
    details = make_plan(actions=(make_action(response_target_id=1),))

    response = presenter.present(details)

    coordinates = response.actions[0].route.path_coordinates
    assert [(c.latitude, c.longitude) for c in coordinates] == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


# ---------------------------------------------------------------------------
# First-Mile Heuristic Fallback: bridge path_coordinates with the station's
# real coordinate when the route's own origin node is a genuine, non-trivial
# distance away (Step 4's station-micro fetch never found close-enough real
# road data) - never touches node_path, eta_seconds, or distance_meters,
# which are always the exact persisted values (GlobalRouteMatrixBuilder
# already applied its own numeric correction to those at build time - the
# presenter must never apply a second one).
# ---------------------------------------------------------------------------


def test_first_mile_bridge_prepends_the_stations_real_coordinate_when_origin_snapped_far():
    # ~223m from node 10 - comfortably past FIRST_MILE_PENALTY_THRESHOLD_KM (100m).
    route = make_route_result(node_path=(10, 20, 30), source_node_id=10, target_node_id=30)
    target = make_target()
    nodes = {10: make_node(10, 1.0, 1.0), 20: make_node(20, 2.0, 2.0), 30: make_node(30, 3.0, 3.0)}
    station = make_station(latitude=1.002, longitude=1.0)
    presenter, _ = make_presenter(
        targets=[FakeStoredTarget(id=1, target=target)], routes=(route,), nodes=nodes, stations=[station]
    )
    details = make_plan(actions=(make_action(response_target_id=1),))

    response = presenter.present(details)

    action = response.actions[0]
    coordinates = [(c.latitude, c.longitude) for c in action.route.path_coordinates]
    assert coordinates == [(1.002, 1.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    # node_path itself is untouched - the bridge only ever affects path_coordinates.
    assert action.route.node_path == [10, 20, 30]


def test_no_first_mile_bridge_when_the_snap_gap_is_within_the_ordinary_threshold():
    # ~11m from node 10 - well under FIRST_MILE_PENALTY_THRESHOLD_KM (100m).
    route = make_route_result(node_path=(10, 20, 30), source_node_id=10, target_node_id=30)
    target = make_target()
    nodes = {10: make_node(10, 1.0, 1.0), 20: make_node(20, 2.0, 2.0), 30: make_node(30, 3.0, 3.0)}
    station = make_station(latitude=1.0001, longitude=1.0)
    presenter, _ = make_presenter(
        targets=[FakeStoredTarget(id=1, target=target)], routes=(route,), nodes=nodes, stations=[station]
    )
    details = make_plan(actions=(make_action(response_target_id=1),))

    response = presenter.present(details)

    coordinates = [(c.latitude, c.longitude) for c in response.actions[0].route.path_coordinates]
    assert coordinates == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


def test_first_mile_bridge_does_not_fire_when_the_station_cannot_be_resolved():
    """A station_id that no longer resolves against persisted FireStation
    data (see ResponsePlanResourceResponse's own docstring) means the real
    coordinate to bridge from is unknown - never guessed at."""
    route = make_route_result(node_path=(10, 20, 30), source_node_id=10, target_node_id=30)
    target = make_target()
    nodes = {10: make_node(10, 1.0, 1.0), 20: make_node(20, 2.0, 2.0), 30: make_node(30, 3.0, 3.0)}
    presenter, _ = make_presenter(
        targets=[FakeStoredTarget(id=1, target=target)], routes=(route,), nodes=nodes, stations=[]
    )
    details = make_plan(actions=(make_action(response_target_id=1, station_id="unknown-station"),))

    response = presenter.present(details)

    coordinates = [(c.latitude, c.longitude) for c in response.actions[0].route.path_coordinates]
    assert coordinates == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


def test_first_mile_bridge_never_alters_eta_or_distance():
    """GlobalRouteMatrixBuilder already applies its own numeric first-mile
    correction to eta_seconds/distance_meters at build time (before
    persistence) - the presenter's geometry bridge must never apply a
    second correction on top of the already-persisted values."""
    route = make_route_result(
        node_path=(10, 20, 30), source_node_id=10, target_node_id=30,
        distance_meters=800.0, travel_time_seconds=120.0,
    )
    target = make_target()
    nodes = {10: make_node(10, 1.0, 1.0), 20: make_node(20, 2.0, 2.0), 30: make_node(30, 3.0, 3.0)}
    station = make_station(latitude=1.002, longitude=1.0)  # far - the bridge fires
    presenter, _ = make_presenter(
        targets=[FakeStoredTarget(id=1, target=target)], routes=(route,), nodes=nodes, stations=[station]
    )
    details = make_plan(actions=(make_action(response_target_id=1),))

    response = presenter.present(details)

    route_response = response.actions[0].route
    assert route_response.distance_meters == 800.0
    assert route_response.eta_seconds == 120.0


# ---------------------------------------------------------------------------
# 9: bulk graph-node lookup, not one query per node
# ---------------------------------------------------------------------------


def test_graph_nodes_are_retrieved_in_a_single_bulk_call():
    first_route = make_route_result(
        resource_id="engine-1", response_target_id=1, node_path=(1, 2, 3), source_node_id=1, target_node_id=3
    )
    second_route = make_route_result(
        resource_id="engine-2", response_target_id=2, node_path=(4, 5), source_node_id=4, target_node_id=5
    )
    target = make_target()
    nodes = {i: make_node(i, float(i), float(i)) for i in range(1, 6)}
    presenter, fakes = make_presenter(
        targets=[FakeStoredTarget(id=1, target=target), FakeStoredTarget(id=2, target=target)],
        routes=(first_route, second_route),
        resource_ids=("engine-1", "engine-2"),
        nodes=nodes,
    )
    details = make_plan(
        actions=(
            make_action(resource_id="engine-1", response_target_id=1, node_path=(1, 2, 3)),
            make_action(resource_id="engine-2", response_target_id=2, node_path=(4, 5)),
        )
    )

    presenter.present(details)

    assert len(fakes["node_repo"].calls) == 1
    assert fakes["node_repo"].calls[0] == frozenset({1, 2, 3, 4, 5})


# ---------------------------------------------------------------------------
# 10-11: missing/unreachable route data is never fabricated
# ---------------------------------------------------------------------------


def test_missing_graph_node_makes_path_coordinates_unavailable():
    route = make_route_result(node_path=(1, 2, 3), source_node_id=1, target_node_id=3)
    target = make_target()
    nodes = {1: make_node(1, 1.0, 1.0), 3: make_node(3, 3.0, 3.0)}  # node 2 missing
    presenter, _ = make_presenter(targets=[FakeStoredTarget(id=1, target=target)], routes=(route,), nodes=nodes)
    details = make_plan(actions=(make_action(response_target_id=1),))

    response = presenter.present(details)

    route_response = response.actions[0].route
    assert route_response.node_path == [1, 2, 3]
    assert route_response.path_coordinates is None


def test_unreachable_route_does_not_fabricate_a_path():
    route = make_route_result(
        status=RouteStatus.UNREACHABLE,
        node_path=(),
        distance_meters=None,
        travel_time_seconds=None,
    )
    target = make_target()
    presenter, _ = make_presenter(targets=[FakeStoredTarget(id=1, target=target)], routes=(route,))
    details = make_plan(
        actions=(make_action(response_target_id=1, eta_seconds=None, route_distance_meters=None, node_path=None),)
    )

    response = presenter.present(details)

    route_response = response.actions[0].route
    assert route_response.status == RouteStatus.UNREACHABLE
    assert route_response.eta_seconds is None
    assert route_response.distance_meters is None
    assert route_response.node_path is None
    assert route_response.path_coordinates is None


def test_missing_route_result_represents_route_as_none_fields_not_fabricated():
    target = make_target()
    presenter, _ = make_presenter(targets=[FakeStoredTarget(id=1, target=target)], routes=())
    details = make_plan(actions=(make_action(response_target_id=1),))

    response = presenter.present(details)

    route_response = response.actions[0].route
    assert route_response.status is None
    assert route_response.eta_seconds is None
    assert route_response.distance_meters is None
    assert route_response.node_path is None
    assert route_response.path_coordinates is None


# ---------------------------------------------------------------------------
# 12: uncovered targets enriched, order preserved
# ---------------------------------------------------------------------------


def test_uncovered_targets_are_enriched_and_preserve_order():
    target_a = make_target(latitude=10.0, longitude=20.0, priority_score=0.1)
    target_b = make_target(latitude=11.0, longitude=21.0, priority_score=0.2)
    presenter, _ = make_presenter(
        targets=[FakeStoredTarget(id=5, target=target_a), FakeStoredTarget(id=2, target=target_b)]
    )
    details = make_plan(actions=(), uncovered_target_ids=(5, 2))

    response = presenter.present(details)

    assert [t.response_target_id for t in response.uncovered_targets] == [5, 2]
    assert response.uncovered_targets[0].latitude == 10.0
    assert response.uncovered_targets[1].latitude == 11.0


def test_uncovered_target_missing_from_target_set_is_not_fabricated():
    presenter, _ = make_presenter(targets=[])
    details = make_plan(actions=(), uncovered_target_ids=(99,))

    response = presenter.present(details)

    uncovered = response.uncovered_targets[0]
    assert uncovered.response_target_id == 99
    assert uncovered.target_type is None
    assert uncovered.priority_score is None
    assert uncovered.latitude is None
    assert uncovered.longitude is None


# ---------------------------------------------------------------------------
# 13-14: no_resources_during_planning
# ---------------------------------------------------------------------------


def test_zero_resource_ids_means_no_resources_during_planning_true():
    presenter, _ = make_presenter(resource_ids=())
    details = make_plan(actions=())

    response = presenter.present(details)

    assert response.no_resources_during_planning is True


def test_non_empty_resource_ids_means_no_resources_during_planning_false():
    presenter, _ = make_presenter(resource_ids=("engine-1", "engine-2"))
    details = make_plan(actions=())

    response = presenter.present(details)

    assert response.no_resources_during_planning is False


# ---------------------------------------------------------------------------
# 15: exact-snapshot rule for a superseded plan
# ---------------------------------------------------------------------------


def test_superseded_plan_is_enriched_from_its_own_exact_snapshot_ids():
    target = make_target(latitude=40.0, longitude=50.0, priority_score=0.9)
    presenter, fakes = make_presenter(
        plan_id=123,
        plan_status=ResponsePlanStatus.COMPLETE,
        target_set_id=456,
        targets=[FakeStoredTarget(id=1, target=target)],
        run_id=789,
        resource_ids=("engine-1",),
    )
    details = make_plan(
        plan_id=123,
        response_target_set_id=456,
        route_planning_run_id=789,
        is_current=False,
        actions=(make_action(response_target_id=1),),
    )

    response = presenter.present(details)

    assert fakes["plan_repo"].calls == [123]
    assert fakes["target_repo"].calls == [456]
    assert fakes["route_repo"].calls == [789]
    assert response.is_current is False
    assert response.actions[0].target.latitude == 40.0


# ---------------------------------------------------------------------------
# 16: no forbidden imports/invocations
# ---------------------------------------------------------------------------


def test_presenter_module_does_not_import_forbidden_dependencies():
    forbidden_fragments = (
        "src.calculators",
        "src.agents",
        "src.external",
        "src.simulation",
        "dijkstra",
        "Dijkstra",
        "genetic",
        "response_optimization",
        "response_planning_refresh",
        "planning_orchestrator",
        "current_response_plan_resolver",
    )
    path = (Path(__file__).resolve().parents[2] / "src/api/response_plan_presenter.py")
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


def test_presenter_source_calls_no_write_or_mutation_operation():
    import inspect

    source = inspect.getsource(ResponsePlanPresenter).lower()
    for forbidden in (".save(", ".update_status(", ".delete(", "mutate", "refresh_plan", "optimize", "dijkstra"):
        assert forbidden not in source
