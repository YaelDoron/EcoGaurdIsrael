"""Tests for shared Epic 5 routing contracts."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import math

import pytest

from src.models.response_target_type import ResponseTargetType
from src.models.routing import (
    RoutePlanningRun,
    RouteResult,
    RouteStatus,
    RoutingResource,
    RoutingTarget,
    StoredRouteResult,
)


def make_resource(**overrides) -> RoutingResource:
    defaults = dict(resource_id="truck-1", station_id="station-1", latitude=32.731, longitude=35.046)
    defaults.update(overrides)
    return RoutingResource(**defaults)


def make_target(**overrides) -> RoutingTarget:
    defaults = dict(
        response_target_id=1,
        target_order=0,
        target_type=ResponseTargetType.ACTIVE_FIRE.value,
        latitude=32.74,
        longitude=35.06,
        priority_score=150.0,
    )
    defaults.update(overrides)
    return RoutingTarget(**defaults)


def make_reachable_result(**overrides) -> RouteResult:
    defaults = dict(
        resource_id="truck-1",
        response_target_id=1,
        status=RouteStatus.REACHABLE,
        source_node_id=10,
        target_node_id=20,
        node_path=(10, 15, 20),
        distance_meters=1200.0,
        travel_time_seconds=90.0,
    )
    defaults.update(overrides)
    return RouteResult(**defaults)


# --- RoutingResource ---------------------------------------------------


def test_valid_routing_resource():
    resource = make_resource()

    assert resource.resource_id == "truck-1"
    assert resource.station_id == "station-1"


def test_routing_resource_is_immutable():
    resource = make_resource()

    with pytest.raises(FrozenInstanceError):
        resource.resource_id = "truck-2"


@pytest.mark.parametrize("resource_id", ["", "   ", 1, None])
def test_invalid_resource_id_rejected(resource_id):
    with pytest.raises(ValueError):
        make_resource(resource_id=resource_id)


@pytest.mark.parametrize("station_id", ["", "   ", 1, None])
def test_invalid_station_id_rejected(station_id):
    with pytest.raises(ValueError):
        make_resource(station_id=station_id)


@pytest.mark.parametrize("latitude", [-90.1, 90.1, math.nan, math.inf, -math.inf, True])
def test_routing_resource_invalid_latitude_rejected(latitude):
    with pytest.raises(ValueError):
        make_resource(latitude=latitude)


@pytest.mark.parametrize("longitude", [-180.1, 180.1, math.nan, math.inf, -math.inf, True])
def test_routing_resource_invalid_longitude_rejected(longitude):
    with pytest.raises(ValueError):
        make_resource(longitude=longitude)


# --- RoutingTarget -------------------------------------------------------


def test_valid_routing_target():
    target = make_target()

    assert target.response_target_id == 1
    assert target.target_type == ResponseTargetType.ACTIVE_FIRE.value


@pytest.mark.parametrize("response_target_id", [0, -1, True, "1"])
def test_invalid_response_target_id_rejected(response_target_id):
    with pytest.raises(ValueError):
        make_target(response_target_id=response_target_id)


@pytest.mark.parametrize("target_order", [-1, True, "0"])
def test_invalid_target_order_rejected(target_order):
    with pytest.raises(ValueError):
        make_target(target_order=target_order)


def test_target_order_zero_is_valid():
    target = make_target(target_order=0)

    assert target.target_order == 0


@pytest.mark.parametrize("target_type", ["", "not_a_real_type", 1, None])
def test_invalid_target_type_rejected(target_type):
    with pytest.raises(ValueError):
        make_target(target_type=target_type)


@pytest.mark.parametrize("priority_score", [math.nan, math.inf, -math.inf, True])
def test_invalid_priority_score_rejected(priority_score):
    with pytest.raises(ValueError):
        make_target(priority_score=priority_score)


# --- RouteResult: REACHABLE ---------------------------------------------


def test_valid_reachable_route_result():
    result = make_reachable_result()

    assert result.status is RouteStatus.REACHABLE
    assert result.node_path == (10, 15, 20)


def test_route_result_is_immutable():
    result = make_reachable_result()

    with pytest.raises(FrozenInstanceError):
        result.distance_meters = 0.0


def test_reachable_result_requires_source_and_target_node_ids():
    with pytest.raises(ValueError):
        make_reachable_result(source_node_id=None)
    with pytest.raises(ValueError):
        make_reachable_result(target_node_id=None)


def test_reachable_result_requires_node_path_matching_endpoints():
    with pytest.raises(ValueError):
        make_reachable_result(node_path=())
    with pytest.raises(ValueError):
        make_reachable_result(node_path=(11, 15, 20))
    with pytest.raises(ValueError):
        make_reachable_result(node_path=(10, 15, 21))


@pytest.mark.parametrize("distance_meters", [None, -1.0, math.nan, math.inf])
def test_reachable_result_requires_valid_distance(distance_meters):
    with pytest.raises(ValueError):
        make_reachable_result(distance_meters=distance_meters)


@pytest.mark.parametrize("travel_time_seconds", [None, -1.0, math.nan, math.inf])
def test_reachable_result_requires_valid_travel_time(travel_time_seconds):
    with pytest.raises(ValueError):
        make_reachable_result(travel_time_seconds=travel_time_seconds)


# --- RouteResult: UNREACHABLE / UNMAPPABLE -------------------------------


def test_valid_unreachable_route_result():
    result = make_reachable_result(
        status=RouteStatus.UNREACHABLE,
        node_path=(),
        distance_meters=None,
        travel_time_seconds=None,
    )

    assert result.status is RouteStatus.UNREACHABLE
    assert result.source_node_id == 10
    assert result.target_node_id == 20


def test_unreachable_result_requires_both_endpoints_mapped():
    with pytest.raises(ValueError):
        make_reachable_result(
            status=RouteStatus.UNREACHABLE,
            source_node_id=None,
            node_path=(),
            distance_meters=None,
            travel_time_seconds=None,
        )


def test_valid_unmappable_route_result():
    result = make_reachable_result(
        status=RouteStatus.UNMAPPABLE,
        source_node_id=None,
        target_node_id=None,
        node_path=(),
        distance_meters=None,
        travel_time_seconds=None,
    )

    assert result.status is RouteStatus.UNMAPPABLE


def test_unmappable_result_rejects_both_endpoints_mapped():
    with pytest.raises(ValueError):
        make_reachable_result(
            status=RouteStatus.UNMAPPABLE,
            node_path=(),
            distance_meters=None,
            travel_time_seconds=None,
        )


@pytest.mark.parametrize(
    "status",
    [RouteStatus.UNREACHABLE, RouteStatus.UNMAPPABLE],
)
def test_non_reachable_result_rejects_non_empty_node_path(status):
    with pytest.raises(ValueError):
        make_reachable_result(
            status=status,
            source_node_id=None,
            distance_meters=None,
            travel_time_seconds=None,
        )


@pytest.mark.parametrize(
    "status",
    [RouteStatus.UNREACHABLE, RouteStatus.UNMAPPABLE],
)
def test_non_reachable_result_rejects_distance_or_time(status):
    with pytest.raises(ValueError):
        make_reachable_result(status=status, source_node_id=None, node_path=(), travel_time_seconds=None)
    with pytest.raises(ValueError):
        make_reachable_result(status=status, source_node_id=None, node_path=(), distance_meters=None)


# --- RoutePlanningRun ------------------------------------------------------


def make_run(**overrides) -> RoutePlanningRun:
    defaults = dict(
        fire_event_id=1,
        response_target_set_id=1,
        planned_at=datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc),
        methodology="ECOGUARD_ROUTING_DIJKSTRA",
        methodology_version="1.0",
        resource_ids=("truck-1",),
        routes=(make_reachable_result(),),
    )
    defaults.update(overrides)
    return RoutePlanningRun(**defaults)


def test_valid_route_planning_run():
    run = make_run()

    assert run.resource_ids == ("truck-1",)
    assert run.routes == (make_reachable_result(),)


def test_route_planning_run_is_immutable():
    run = make_run()

    with pytest.raises(FrozenInstanceError):
        run.methodology = "OTHER"


def test_route_planning_run_coerces_iterables_to_tuples():
    run = make_run(resource_ids=["truck-1"], routes=[make_reachable_result()])

    assert run.resource_ids == ("truck-1",)
    assert isinstance(run.routes, tuple)


@pytest.mark.parametrize("fire_event_id", [0, -1, True, "1"])
def test_run_invalid_fire_event_id_rejected(fire_event_id):
    with pytest.raises(ValueError):
        make_run(fire_event_id=fire_event_id)


@pytest.mark.parametrize("response_target_set_id", [0, -1, True, "1"])
def test_run_invalid_response_target_set_id_rejected(response_target_set_id):
    with pytest.raises(ValueError):
        make_run(response_target_set_id=response_target_set_id)


def test_run_naive_planned_at_rejected():
    with pytest.raises(ValueError):
        make_run(planned_at=datetime(2026, 9, 16, 12, 0))


@pytest.mark.parametrize("methodology", ["", "   ", 1, None])
def test_run_invalid_methodology_rejected(methodology):
    with pytest.raises(ValueError):
        make_run(methodology=methodology)


def test_run_duplicate_resource_ids_rejected():
    with pytest.raises(ValueError):
        make_run(resource_ids=("truck-1", "truck-1"), routes=())


def test_run_empty_resource_ids_and_routes_is_valid():
    run = make_run(resource_ids=(), routes=())

    assert run.resource_ids == ()
    assert run.routes == ()


def test_run_rejects_non_route_result_items():
    with pytest.raises(ValueError):
        make_run(routes=("not-a-route-result",))


def test_run_rejects_route_resource_id_not_in_resource_ids():
    with pytest.raises(ValueError):
        make_run(
            resource_ids=("truck-1",),
            routes=(make_reachable_result(resource_id="truck-2"),),
        )


def test_run_rejects_duplicate_resource_target_pair():
    duplicate = make_reachable_result()
    with pytest.raises(ValueError):
        make_run(resource_ids=("truck-1",), routes=(duplicate, duplicate))


# --- StoredRouteResult -------------------------------------------------


def test_valid_stored_route_result():
    stored = StoredRouteResult(id=1, route_result=make_reachable_result())

    assert stored.id == 1
    assert stored.route_result.status is RouteStatus.REACHABLE


def test_stored_route_result_is_immutable():
    stored = StoredRouteResult(id=1, route_result=make_reachable_result())

    with pytest.raises(FrozenInstanceError):
        stored.id = 2


@pytest.mark.parametrize("id_value", [0, -1, True, "1"])
def test_stored_route_result_invalid_id_rejected(id_value):
    with pytest.raises(ValueError):
        StoredRouteResult(id=id_value, route_result=make_reachable_result())


def test_stored_route_result_requires_route_result_instance():
    with pytest.raises(ValueError):
        StoredRouteResult(id=1, route_result="not-a-route-result")
