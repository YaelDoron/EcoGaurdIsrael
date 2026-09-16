"""Tests for building optimization input from persisted routing snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.models import ResponseTarget, ResponseTargetSet, ResponseTargetType
from src.models.routing import RoutePlanningRun, RouteResult, RouteStatus, StoredRouteResult
from src.repositories.response_target_repository import StoredResponseTarget, StoredResponseTargetSet
from src.repositories.route_planning_repository import StoredRoutePlanningRun
from src.services.response_optimization import ResponseOptimizationInputService, ResponseOptimizationInputServiceError

AS_OF = datetime(2026, 9, 16, 16, 0, tzinfo=timezone.utc)
_DEFAULT = object()


class FakeRoutePlanningRepository:
    def __init__(self, stored_run) -> None:
        self._stored_run = stored_run

    def get_by_id(self, run_id: int):  # noqa: ANN001
        return self._stored_run if run_id == 10 else None


class FakeResponseTargetRepository:
    def __init__(self, stored_target_set) -> None:
        self._stored_target_set = stored_target_set

    def get_by_id(self, target_set_id: int):  # noqa: ANN001
        return self._stored_target_set if target_set_id == 20 else None


def target(
    target_id: int,
    order: int,
    priority: float = 100.0,
    *,
    fire_event_id: int = 1,
    target_type: ResponseTargetType | None = None,
) -> StoredResponseTarget:
    target_type = target_type or (
        ResponseTargetType.ACTIVE_FIRE if order == 0 else ResponseTargetType.PREDICTED_RISK
    )
    metadata = {}
    if target_type is ResponseTargetType.PREDICTED_RISK:
        metadata = {
            "prediction_horizon_minutes": 30,
            "spread_prediction_id": 1000 + target_id,
            "spread_prediction_cell_id": 2000 + target_id,
        }
    return StoredResponseTarget(
        id=target_id,
        target_order=order,
        target=ResponseTarget(
            fire_event_id=fire_event_id,
            target_type=target_type,
            latitude=32.7,
            longitude=35.0,
            priority_score=priority,
            **metadata,
        ),
    )


def target_set(targets=(target(101, 0, 90.0), target(102, 1, 40.0))):
    fire_event_id = targets[0].target.fire_event_id
    return StoredResponseTargetSet(
        id=20,
        target_set=ResponseTargetSet(
            fire_event_id=fire_event_id,
            generated_at=AS_OF,
            methodology="TARGETS",
            methodology_version="1.0",
            targets=tuple(stored.target for stored in targets),
        ),
        targets=tuple(targets),
    )


def route_result(
    resource_id: str,
    target_id: int,
    status: RouteStatus = RouteStatus.REACHABLE,
    *,
    eta: float | None = 12.5,
    distance: float | None = 100.0,
) -> RouteResult:
    return RouteResult(
        resource_id=resource_id,
        response_target_id=target_id,
        status=status,
        source_node_id=1 if status is not RouteStatus.UNMAPPABLE else None,
        target_node_id=2 if status is not RouteStatus.UNMAPPABLE else None,
        node_path=(1, 2) if status is RouteStatus.REACHABLE else (),
        distance_meters=distance if status is RouteStatus.REACHABLE else None,
        travel_time_seconds=eta if status is RouteStatus.REACHABLE else None,
    )


def stored_run(routes=None, resource_ids=("R1", "R2")):
    routes = (
        StoredRouteResult(501, route_result("R1", 101, RouteStatus.REACHABLE, eta=12.5, distance=450.0)),
        StoredRouteResult(502, route_result("R2", 102, RouteStatus.UNREACHABLE, eta=None, distance=None)),
    ) if routes is None else routes
    return StoredRoutePlanningRun(
        id=10,
        run=RoutePlanningRun(
            fire_event_id=1,
            response_target_set_id=20,
            planned_at=AS_OF,
            methodology="ROUTING",
            methodology_version="1.0",
            resource_ids=resource_ids,
            routes=tuple(stored.route_result for stored in routes),
        ),
        routes=tuple(routes),
    )


def service(run=_DEFAULT, targets=_DEFAULT) -> ResponseOptimizationInputService:
    return ResponseOptimizationInputService(
        route_planning_repository=FakeRoutePlanningRepository(stored_run() if run is _DEFAULT else run),
        response_target_repository=FakeResponseTargetRepository(target_set() if targets is _DEFAULT else targets),
    )


def test_exact_run_target_resource_and_route_mapping():
    input_data = service().build_from_route_planning_run(10)

    assert input_data.fire_event_id == 1
    assert input_data.response_target_set_id == 20
    assert input_data.route_planning_run_id == 10
    assert [(target.response_target_id, target.target_order, target.priority_score) for target in input_data.targets] == [
        (101, 0, 90.0),
        (102, 1, 40.0),
    ]
    assert [resource.resource_id for resource in input_data.resources] == ["R1", "R2"]
    assert [(route.route_result_id, route.resource_id, route.response_target_id) for route in input_data.route_options] == [
        (501, "R1", 101),
        (502, "R2", 102),
    ]


def test_route_status_eta_and_distance_mapping():
    routes = (
        StoredRouteResult(501, route_result("R1", 101, RouteStatus.REACHABLE, eta=33.0, distance=700.0)),
        StoredRouteResult(502, route_result("R1", 102, RouteStatus.UNREACHABLE, eta=None, distance=None)),
        StoredRouteResult(503, route_result("R2", 102, RouteStatus.UNMAPPABLE, eta=None, distance=None)),
    )

    input_data = service(stored_run(routes=routes)).build_from_route_planning_run(10)

    by_id = {route.route_result_id: route for route in input_data.route_options}
    assert by_id[501].is_reachable is True
    assert by_id[501].travel_time_seconds == 33.0
    assert by_id[501].distance_meters == 700.0
    assert by_id[502].is_reachable is False
    assert by_id[502].travel_time_seconds is None
    assert by_id[503].is_reachable is False
    assert by_id[503].distance_meters is None


def test_resource_snapshot_does_not_use_current_resource_state():
    input_data = service(stored_run(routes=(), resource_ids=("R1", "R2"))).build_from_route_planning_run(10)

    assert [resource.resource_id for resource in input_data.resources] == ["R1", "R2"]


def test_missing_route_planning_run_and_target_set_fail_clearly():
    with pytest.raises(ResponseOptimizationInputServiceError):
        service().build_from_route_planning_run(999)
    with pytest.raises(ResponseOptimizationInputServiceError):
        service(targets=None).build_from_route_planning_run(10)


def test_inconsistent_snapshots_are_rejected():
    bad_target_set = StoredResponseTargetSet(
        id=20,
        target_set=ResponseTargetSet(
            fire_event_id=999,
            generated_at=AS_OF,
            methodology="TARGETS",
            methodology_version="1.0",
            targets=(target(101, 0, fire_event_id=999).target,),
        ),
        targets=(target(101, 0, fire_event_id=999),),
    )

    with pytest.raises(ResponseOptimizationInputServiceError):
        service(targets=bad_target_set).build_from_route_planning_run(10)


def test_route_references_unknown_resource_or_target_rejected():
    valid_set = target_set()
    unknown_resource_run = SimpleNamespace(
        id=10,
        run=SimpleNamespace(fire_event_id=1, response_target_set_id=20, resource_ids=("R1",)),
        routes=(StoredRouteResult(501, route_result("R2", 101)),),
    )
    unknown_target_run = SimpleNamespace(
        id=10,
        run=SimpleNamespace(fire_event_id=1, response_target_set_id=20, resource_ids=("R1",)),
        routes=(StoredRouteResult(501, route_result("R1", 999)),),
    )

    with pytest.raises(ResponseOptimizationInputServiceError):
        service(run=unknown_resource_run, targets=valid_set).build_from_route_planning_run(10)
    with pytest.raises(ResponseOptimizationInputServiceError):
        service(run=unknown_target_run, targets=valid_set).build_from_route_planning_run(10)


def test_duplicate_route_ids_or_pairs_rejected():
    duplicate_ids = SimpleNamespace(
        id=10,
        run=SimpleNamespace(fire_event_id=1, response_target_set_id=20, resource_ids=("R1", "R2")),
        routes=(
            StoredRouteResult(501, route_result("R1", 101)),
            StoredRouteResult(501, route_result("R2", 102)),
        ),
    )
    duplicate_pairs = SimpleNamespace(
        id=10,
        run=SimpleNamespace(fire_event_id=1, response_target_set_id=20, resource_ids=("R1",)),
        routes=(
            StoredRouteResult(501, route_result("R1", 101)),
            StoredRouteResult(502, route_result("R1", 101)),
        ),
    )

    with pytest.raises(ResponseOptimizationInputServiceError):
        service(run=duplicate_ids).build_from_route_planning_run(10)
    with pytest.raises(ResponseOptimizationInputServiceError):
        service(run=duplicate_pairs).build_from_route_planning_run(10)
