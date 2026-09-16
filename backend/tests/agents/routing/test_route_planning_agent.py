"""Tests for RoutePlanningAgent orchestration, using fakes for every dependency (no real DB)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.agents.routing import RoutePlanningAgent, RoutePlanningResult, RoutePlanningStatus
from src.calculators.response_target.response_target_config import (
    RESPONSE_TARGET_METHODOLOGY_NAME,
    RESPONSE_TARGET_METHODOLOGY_VERSION,
)
from src.calculators.routing.dijkstra_calculator import DijkstraResult
from src.calculators.routing.routing_config import ROUTING_METHODOLOGY_NAME, ROUTING_METHODOLOGY_VERSION
from src.models import GraphEdge, GraphNode, ResponseTarget, ResponseTargetSet, ResponseTargetType, RouteStatus
from src.models.routing import StoredRouteResult
from src.repositories.response_target_repository import StoredResponseTarget, StoredResponseTargetSet
from src.repositories.route_planning_repository import StoredRoutePlanningRun

AS_OF = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
FIRE_EVENT_ID = 42

ACTIVE_TARGET = ResponseTarget(
    fire_event_id=FIRE_EVENT_ID,
    target_type=ResponseTargetType.ACTIVE_FIRE,
    latitude=32.731,
    longitude=35.046,
    priority_score=150.0,
)
PREDICTED_TARGET = ResponseTarget(
    fire_event_id=FIRE_EVENT_ID,
    target_type=ResponseTargetType.PREDICTED_RISK,
    latitude=32.75,
    longitude=35.07,
    priority_score=80.0,
    prediction_horizon_minutes=30,
    spread_prediction_id=10,
    spread_prediction_cell_id=100,
)

ROAD_NODES = [
    GraphNode(id=1001, latitude=32.700, longitude=35.000),
    GraphNode(id=1002, latitude=32.700, longitude=35.010),
]
ROAD_EDGES = [GraphEdge(source_node_id=1001, target_node_id=1002, distance_meters=1000.0, travel_time_seconds=90.0)]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSession:
    def close(self) -> None:
        pass


class FakeResponseTargetRepository:
    def __init__(self, stored_set: StoredResponseTargetSet | None, exc: Exception | None = None) -> None:
        self.stored_set = stored_set
        self.exc = exc
        self.calls = []

    def get_latest_for_event_as_of(self, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        if self.exc is not None:
            raise self.exc
        return self.stored_set


class FakeOperationalContextService:
    def __init__(self, context=None, exc: Exception | None = None) -> None:
        self.context = context
        self.exc = exc
        self.calls = []

    def build_context(self, db, fire_latitude, fire_longitude, min_resources: int = 1):
        self.calls.append({"fire_latitude": fire_latitude, "fire_longitude": fire_longitude})
        if self.exc is not None:
            raise self.exc
        return self.context


class FakeNodeMappingService:
    def __init__(self, resource_map: dict | None = None, target_map: dict | None = None) -> None:
        self._resource_map = resource_map or {}
        self._target_map = target_map or {}
        self.map_resources_calls = []
        self.map_targets_calls = []

    def map_resources(self, resources, nodes, edges):
        self.map_resources_calls.append(resources)
        return {resource.resource_id: self._resource_map.get(resource.resource_id) for resource in resources}

    def map_targets(self, targets, nodes, edges):
        self.map_targets_calls.append(targets)
        return {target.response_target_id: self._target_map.get(target.response_target_id) for target in targets}


class FakeDijkstraCalculator:
    def __init__(self, results: dict | None = None) -> None:
        self._results = results or {}
        self.calls = []

    def calculate_shortest_path(self, source_node_id, target_node_id, edges):
        self.calls.append((source_node_id, target_node_id))
        key = (source_node_id, target_node_id)
        if key in self._results:
            return self._results[key]
        return DijkstraResult(
            status=RouteStatus.REACHABLE,
            node_path=(source_node_id, target_node_id),
            distance_meters=1000.0,
            travel_time_seconds=90.0,
        )


class FakeRoutePlanningRepository:
    def __init__(self, first_id: int = 601, exc: Exception | None = None) -> None:
        self._next_id = first_id
        self.exc = exc
        self.calls = []

    def save_run(self, run):
        self.calls.append(run)
        if self.exc is not None:
            raise self.exc
        stored_run_id = self._next_id
        self._next_id += 1
        stored_routes = tuple(
            StoredRouteResult(id=1000 + index, route_result=route) for index, route in enumerate(run.routes)
        )
        return StoredRoutePlanningRun(id=stored_run_id, run=run, routes=stored_routes)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_target_set(targets=(ACTIVE_TARGET,), set_id: int = 501) -> StoredResponseTargetSet:
    domain_set = ResponseTargetSet(
        fire_event_id=FIRE_EVENT_ID,
        generated_at=AS_OF - timedelta(minutes=5),
        methodology=RESPONSE_TARGET_METHODOLOGY_NAME,
        methodology_version=RESPONSE_TARGET_METHODOLOGY_VERSION,
        targets=targets,
    )
    stored_targets = tuple(
        StoredResponseTarget(id=900 + index, target_order=index, target=target)
        for index, target in enumerate(targets)
    )
    return StoredResponseTargetSet(id=set_id, target_set=domain_set, targets=stored_targets)


def make_station(station_id="station-1", latitude=32.700, longitude=35.000):
    return SimpleNamespace(id=station_id, latitude=latitude, longitude=longitude)


def make_resource(resource_id="truck-1", station_id="station-1"):
    return SimpleNamespace(id=resource_id, station_id=station_id)


def make_context(stations=None, resources=None, road_nodes=None, road_edges=None):
    return SimpleNamespace(
        stations=stations if stations is not None else [make_station()],
        available_resources=resources if resources is not None else [make_resource()],
        road_nodes=road_nodes if road_nodes is not None else ROAD_NODES,
        road_edges=road_edges if road_edges is not None else ROAD_EDGES,
    )


def make_agent(
    response_target_repository=None,
    operational_context_service=None,
    node_mapping_service=None,
    dijkstra_calculator=None,
    route_planning_repository=None,
) -> RoutePlanningAgent:
    return RoutePlanningAgent(
        response_target_repository=response_target_repository or FakeResponseTargetRepository(make_target_set()),
        operational_context_service=operational_context_service or FakeOperationalContextService(make_context()),
        node_mapping_service=node_mapping_service
        or FakeNodeMappingService(resource_map={"truck-1": 1001}, target_map={900: 1002}),
        dijkstra_calculator=dijkstra_calculator or FakeDijkstraCalculator(),
        route_planning_repository=route_planning_repository or FakeRoutePlanningRepository(),
        session_factory=lambda: FakeSession(),
    )


# ---------------------------------------------------------------------------
# Successful run: multiple resources and targets
# ---------------------------------------------------------------------------


def test_successful_run_computes_full_cross_product_of_resources_and_targets():
    response_target_repository = FakeResponseTargetRepository(make_target_set((ACTIVE_TARGET, PREDICTED_TARGET)))
    context = make_context(
        stations=[make_station("station-1"), make_station("station-2", latitude=32.71, longitude=35.02)],
        resources=[make_resource("truck-1", "station-1"), make_resource("truck-2", "station-2")],
    )
    node_mapping_service = FakeNodeMappingService(
        resource_map={"truck-1": 1001, "truck-2": 1002},
        target_map={900: 1001, 901: 1002},
    )
    route_planning_repository = FakeRoutePlanningRepository()

    result = make_agent(
        response_target_repository=response_target_repository,
        operational_context_service=FakeOperationalContextService(context),
        node_mapping_service=node_mapping_service,
        route_planning_repository=route_planning_repository,
    ).plan(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.success is True
    assert result.status is RoutePlanningStatus.PLANNED
    assert result.route_count == 4
    assert result.run_id is not None
    assert len(route_planning_repository.calls) == 1
    saved_run = route_planning_repository.calls[0]
    assert saved_run.resource_ids == ("truck-1", "truck-2")
    pairs = {(route.resource_id, route.response_target_id) for route in saved_run.routes}
    assert pairs == {
        ("truck-1", 900),
        ("truck-1", 901),
        ("truck-2", 900),
        ("truck-2", 901),
    }
    assert all(route.status is RouteStatus.REACHABLE for route in saved_run.routes)


def test_successful_run_saves_correct_methodology_and_run_metadata():
    route_planning_repository = FakeRoutePlanningRepository()

    make_agent(route_planning_repository=route_planning_repository).plan(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    saved_run = route_planning_repository.calls[0]
    assert saved_run.fire_event_id == FIRE_EVENT_ID
    assert saved_run.response_target_set_id == 501
    assert saved_run.planned_at == AS_OF
    assert saved_run.methodology == ROUTING_METHODOLOGY_NAME
    assert saved_run.methodology_version == ROUTING_METHODOLOGY_VERSION


def test_operational_context_is_built_from_active_fire_target_coordinates():
    operational_context_service = FakeOperationalContextService(make_context())
    target_set = make_target_set((ACTIVE_TARGET, PREDICTED_TARGET))

    make_agent(
        response_target_repository=FakeResponseTargetRepository(target_set),
        operational_context_service=operational_context_service,
    ).plan(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert operational_context_service.calls == [
        {"fire_latitude": ACTIVE_TARGET.latitude, "fire_longitude": ACTIVE_TARGET.longitude}
    ]


def test_reachable_dijkstra_result_fields_are_mapped_onto_route_result():
    dijkstra_calculator = FakeDijkstraCalculator(
        results={
            (1001, 1002): DijkstraResult(
                status=RouteStatus.REACHABLE,
                node_path=(1001, 1500, 1002),
                distance_meters=2500.0,
                travel_time_seconds=180.0,
            )
        }
    )
    route_planning_repository = FakeRoutePlanningRepository()

    make_agent(dijkstra_calculator=dijkstra_calculator, route_planning_repository=route_planning_repository).plan(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    route = route_planning_repository.calls[0].routes[0]
    assert route.status is RouteStatus.REACHABLE
    assert route.source_node_id == 1001
    assert route.target_node_id == 1002
    assert route.node_path == (1001, 1500, 1002)
    assert route.distance_meters == 2500.0
    assert route.travel_time_seconds == 180.0


def test_dijkstra_unreachable_result_produces_unreachable_route():
    dijkstra_calculator = FakeDijkstraCalculator(
        results={
            (1001, 1002): DijkstraResult(
                status=RouteStatus.UNREACHABLE,
                node_path=(),
                distance_meters=None,
                travel_time_seconds=None,
            )
        }
    )
    route_planning_repository = FakeRoutePlanningRepository()

    make_agent(dijkstra_calculator=dijkstra_calculator, route_planning_repository=route_planning_repository).plan(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    route = route_planning_repository.calls[0].routes[0]
    assert route.status is RouteStatus.UNREACHABLE
    assert route.source_node_id == 1001
    assert route.target_node_id == 1002
    assert route.node_path == ()
    assert route.distance_meters is None
    assert route.travel_time_seconds is None


def test_unmapped_resource_produces_unmappable_route_without_calling_dijkstra():
    node_mapping_service = FakeNodeMappingService(resource_map={}, target_map={900: 1002})
    dijkstra_calculator = FakeDijkstraCalculator()
    route_planning_repository = FakeRoutePlanningRepository()

    make_agent(
        node_mapping_service=node_mapping_service,
        dijkstra_calculator=dijkstra_calculator,
        route_planning_repository=route_planning_repository,
    ).plan(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    route = route_planning_repository.calls[0].routes[0]
    assert route.status is RouteStatus.UNMAPPABLE
    assert route.source_node_id is None
    assert route.target_node_id == 1002
    assert dijkstra_calculator.calls == []


def test_unmapped_target_produces_unmappable_route_without_calling_dijkstra():
    node_mapping_service = FakeNodeMappingService(resource_map={"truck-1": 1001}, target_map={})
    dijkstra_calculator = FakeDijkstraCalculator()
    route_planning_repository = FakeRoutePlanningRepository()

    make_agent(
        node_mapping_service=node_mapping_service,
        dijkstra_calculator=dijkstra_calculator,
        route_planning_repository=route_planning_repository,
    ).plan(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    route = route_planning_repository.calls[0].routes[0]
    assert route.status is RouteStatus.UNMAPPABLE
    assert route.source_node_id == 1001
    assert route.target_node_id is None
    assert dijkstra_calculator.calls == []


# ---------------------------------------------------------------------------
# Graceful handling: no available resources
# ---------------------------------------------------------------------------


def test_no_available_resources_persists_empty_run_and_returns_no_resources_status():
    context = make_context(stations=[], resources=[])
    node_mapping_service = FakeNodeMappingService()
    dijkstra_calculator = FakeDijkstraCalculator()
    route_planning_repository = FakeRoutePlanningRepository()

    result = make_agent(
        operational_context_service=FakeOperationalContextService(context),
        node_mapping_service=node_mapping_service,
        dijkstra_calculator=dijkstra_calculator,
        route_planning_repository=route_planning_repository,
    ).plan(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.success is True
    assert result.status is RoutePlanningStatus.NO_RESOURCES_AVAILABLE
    assert result.route_count == 0
    assert result.run_id is not None
    assert result.run.run.resource_ids == ()
    assert result.run.run.routes == ()
    assert node_mapping_service.map_resources_calls == []
    assert node_mapping_service.map_targets_calls == []
    assert dijkstra_calculator.calls == []
    assert len(route_planning_repository.calls) == 1


# ---------------------------------------------------------------------------
# Graceful handling: no current ResponseTargetSet
# ---------------------------------------------------------------------------


def test_no_target_set_returns_no_targets_status_without_further_calls():
    response_target_repository = FakeResponseTargetRepository(stored_set=None)
    operational_context_service = FakeOperationalContextService(make_context())
    route_planning_repository = FakeRoutePlanningRepository()

    result = make_agent(
        response_target_repository=response_target_repository,
        operational_context_service=operational_context_service,
        route_planning_repository=route_planning_repository,
    ).plan(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.success is True
    assert result.status is RoutePlanningStatus.NO_TARGETS
    assert result.run_id is None
    assert result.run is None
    assert result.route_count == 0
    assert operational_context_service.calls == []
    assert route_planning_repository.calls == []


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_response_target_repository_exception_returns_failed():
    response_target_repository = FakeResponseTargetRepository(None, exc=RuntimeError("db exploded"))
    operational_context_service = FakeOperationalContextService(make_context())

    result = make_agent(
        response_target_repository=response_target_repository,
        operational_context_service=operational_context_service,
    ).plan(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.success is False
    assert result.status is RoutePlanningStatus.FAILED
    assert result.run_id is None
    assert result.run is None
    assert result.error_message
    assert operational_context_service.calls == []


def test_operational_context_exception_returns_failed_without_persisting():
    operational_context_service = FakeOperationalContextService(exc=RuntimeError("context blew up"))
    route_planning_repository = FakeRoutePlanningRepository()

    result = make_agent(
        operational_context_service=operational_context_service,
        route_planning_repository=route_planning_repository,
    ).plan(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is RoutePlanningStatus.FAILED
    assert route_planning_repository.calls == []


def test_route_planning_repository_exception_returns_failed():
    route_planning_repository = FakeRoutePlanningRepository(exc=RuntimeError("persistence exploded"))

    result = make_agent(route_planning_repository=route_planning_repository).plan(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    assert result.status is RoutePlanningStatus.FAILED
    assert result.run_id is None
    assert len(route_planning_repository.calls) == 1


def test_resource_referencing_unknown_station_is_caught_as_failed():
    context = make_context(stations=[make_station("station-1")], resources=[make_resource("truck-1", "station-ghost")])

    result = make_agent(operational_context_service=FakeOperationalContextService(context)).plan(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    assert result.status is RoutePlanningStatus.FAILED
    assert result.success is False


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_fire_event_id", [0, -1, True, "42"])
def test_invalid_fire_event_id_rejected(invalid_fire_event_id):
    with pytest.raises(ValueError):
        make_agent().plan(fire_event_id=invalid_fire_event_id, as_of=AS_OF)


def test_naive_as_of_rejected():
    with pytest.raises(ValueError):
        make_agent().plan(fire_event_id=FIRE_EVENT_ID, as_of=datetime(2026, 9, 16, 12, 0))


def test_result_model_rejects_inconsistent_status_and_fields():
    with pytest.raises(ValueError):
        RoutePlanningResult(
            success=True,
            fire_event_id=FIRE_EVENT_ID,
            status=RoutePlanningStatus.PLANNED,
            run_id=None,
            run=None,
            route_count=1,
        )
