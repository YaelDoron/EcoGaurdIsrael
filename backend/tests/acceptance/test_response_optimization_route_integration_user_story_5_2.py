"""Acceptance tests for real US 5.1 routing to US 5.2 optimization integration."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.analysis.response_optimization_agent import ResponseOptimizationAgent
from src.calculators.response_optimization import ResponseOptimizationConfig
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.fire_spread_prediction_cell_db import FireSpreadPredictionCellDB
from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.graph_node_db import GraphNodeDB
from src.models import ResponsePlanStatus, ResponseTarget, ResponseTargetSet, ResponseTargetType
from src.models.resource_status import ResourceStatus
from src.models.routing import RoutePlanningRun, RouteResult, RouteStatus
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.response_optimization import ResponseOptimizationInputService

AS_OF = datetime(2026, 9, 16, 17, 0, tzinfo=timezone.utc)
SOURCE_NODE_ID = 91001
TARGET_NODE_ID = 91002


@pytest.fixture
def stack(sqlite_session_factory):
    session = sqlite_session_factory()
    session.add_all(
        [
            GraphNodeDB(id=SOURCE_NODE_ID, latitude=32.7, longitude=35.0),
            GraphNodeDB(id=TARGET_NODE_ID, latitude=32.71, longitude=35.01),
        ]
    )
    event = FireEventDB(
        latitude=32.7,
        longitude=35.0,
        detected_at=AS_OF - timedelta(minutes=30),
        updated_at=AS_OF - timedelta(minutes=5),
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    spread_prediction = FireSpreadPredictionDB(
        fire_event_id=event.id,
        severity_assessment_id=None,
        predicted_at=AS_OF - timedelta(minutes=10),
        horizon_minutes=30,
        status="insufficient_data",
        methodology="TEST_SPREAD",
        methodology_version="1.0",
    )
    session.add(spread_prediction)
    session.flush()
    spread_cell = FireSpreadPredictionCellDB(
        prediction_id=spread_prediction.id,
        latitude=32.701,
        longitude=35.0,
        spread_probability=0.8,
        spread_risk_score=50.0,
        reached_step=1,
        reached_minutes=30,
    )
    session.add(spread_cell)
    session.flush()
    session.add(FireStationDB(id="S-1", name="Station 1", latitude=32.7, longitude=35.0))
    session.flush()
    session.add_all(
        [
            FirefightingResourceDB(id="R1", station_id="S-1", status=ResourceStatus.AVAILABLE),
            FirefightingResourceDB(id="R2", station_id="S-1", status=ResourceStatus.UNAVAILABLE),
            FirefightingResourceDB(id="R3", station_id="S-1", status=ResourceStatus.AVAILABLE),
        ]
    )
    session.commit()
    session.close()

    target_repository = ResponseTargetRepository(sqlite_session_factory)
    route_repository = RoutePlanningRepository(sqlite_session_factory)
    plan_repository = ResponsePlanRepository(sqlite_session_factory)
    service = ResponseOptimizationInputService(
        route_planning_repository=route_repository,
        response_target_repository=target_repository,
    )
    agent = ResponseOptimizationAgent(plan_repository, input_service=service)
    return {
        "event_id": event.id,
        "spread_prediction_id": spread_prediction.id,
        "spread_cell_id": spread_cell.id,
        "targets": target_repository,
        "routes": route_repository,
        "plans": plan_repository,
        "agent": agent,
        "session_factory": sqlite_session_factory,
    }


def target_set(stack, priorities=(100.0, 50.0)):
    stored = stack["targets"].save_target_set(
        ResponseTargetSet(
            fire_event_id=stack["event_id"],
            generated_at=AS_OF - timedelta(minutes=2),
            methodology="TARGETS",
            methodology_version="1.0",
            targets=tuple(
                ResponseTarget(
                    fire_event_id=stack["event_id"],
                    target_type=(
                        ResponseTargetType.ACTIVE_FIRE
                        if index == 0
                        else ResponseTargetType.PREDICTED_RISK
                    ),
                    latitude=32.7 + index * 0.001,
                    longitude=35.0,
                    priority_score=priority,
                    prediction_horizon_minutes=30 if index != 0 else None,
                    spread_prediction_id=stack["spread_prediction_id"] if index != 0 else None,
                    spread_prediction_cell_id=stack["spread_cell_id"] if index != 0 else None,
                )
                for index, priority in enumerate(priorities)
            ),
        )
    )
    return stored


def reachable(resource_id: str, target_id: int, eta: float) -> RouteResult:
    return RouteResult(
        resource_id=resource_id,
        response_target_id=target_id,
        status=RouteStatus.REACHABLE,
        source_node_id=SOURCE_NODE_ID,
        target_node_id=TARGET_NODE_ID,
        node_path=(SOURCE_NODE_ID, TARGET_NODE_ID),
        distance_meters=eta * 10.0,
        travel_time_seconds=eta,
    )


def unavailable(resource_id: str, target_id: int, status: RouteStatus) -> RouteResult:
    return RouteResult(
        resource_id=resource_id,
        response_target_id=target_id,
        status=status,
        source_node_id=SOURCE_NODE_ID if status is RouteStatus.UNREACHABLE else None,
        target_node_id=TARGET_NODE_ID if status is RouteStatus.UNREACHABLE else None,
        node_path=(),
        distance_meters=None,
        travel_time_seconds=None,
    )


def save_run(stack, stored_target_set, routes, resource_ids=("R1", "R2")):
    return stack["routes"].save_run(
        RoutePlanningRun(
            fire_event_id=stack["event_id"],
            response_target_set_id=stored_target_set.id,
            planned_at=AS_OF - timedelta(minutes=1),
            methodology="ECOGUARD_ROUTING_DIJKSTRA",
            methodology_version="1.0",
            resource_ids=resource_ids,
            routes=tuple(routes),
        )
    )


def logical_plan(plan):
    return (
        plan.status,
        plan.actions,
        plan.uncovered_target_ids,
        plan.plan_score,
        plan.coverage_score,
        plan.average_eta_seconds,
        plan.methodology,
        plan.methodology_version,
        plan.random_seed,
    )


def test_real_route_planning_run_to_response_plan_traceability_and_status_nonmutation(stack):
    stored_targets = target_set(stack)
    t1, t2 = [stored_target.id for stored_target in stored_targets.targets]
    stored_run = save_run(
        stack,
        stored_targets,
        (reachable("R1", t1, 100.0), reachable("R2", t2, 200.0)),
    )

    result = stack["agent"].optimize_from_route_planning_run(
        stored_run.id,
        as_of=AS_OF,
        config=ResponseOptimizationConfig(population_size=4, generation_count=2, random_seed=5),
    )
    plan = stack["plans"].get_by_id(result.response_plan_id).plan

    assert plan.fire_event_id == stored_run.run.fire_event_id
    assert plan.response_target_set_id == stored_run.run.response_target_set_id
    assert plan.route_planning_run_id == stored_run.id
    assert plan.generated_at == AS_OF
    assert plan.status is ResponsePlanStatus.COMPLETE
    assert {action.route_result_id for action in plan.actions} == {stored.id for stored in stored_run.routes}
    assert plan.methodology == "GENETIC_RESOURCE_ALLOCATION"
    assert plan.methodology_version == "1.0"
    session = stack["session_factory"]()
    statuses = {row.id: row.status for row in session.query(FirefightingResourceDB).all()}
    session.close()
    assert statuses == {"R1": ResourceStatus.AVAILABLE, "R2": ResourceStatus.UNAVAILABLE, "R3": ResourceStatus.AVAILABLE}


def test_unreachable_and_unmappable_routes_are_not_assigned(stack):
    stored_targets = target_set(stack)
    t1, t2 = [stored_target.id for stored_target in stored_targets.targets]
    stored_run = save_run(
        stack,
        stored_targets,
        (
            reachable("R1", t1, 100.0),
            unavailable("R1", t2, RouteStatus.UNREACHABLE),
            unavailable("R2", t1, RouteStatus.UNMAPPABLE),
            unavailable("R2", t2, RouteStatus.UNREACHABLE),
        ),
    )

    result = stack["agent"].optimize_from_route_planning_run(
        stored_run.id,
        as_of=AS_OF,
        config=ResponseOptimizationConfig(population_size=4, generation_count=2, random_seed=7),
    )
    plan = stack["plans"].get_by_id(result.response_plan_id).plan

    assert [action.response_target_id for action in plan.actions] == [t1]
    assert plan.uncovered_target_ids == (t2,)
    assert all(action.route_result_id == stored_run.routes[0].id for action in plan.actions)


def test_no_resource_routing_run_is_successful_no_feasible_plan(stack):
    stored_targets = target_set(stack)
    stored_run = save_run(stack, stored_targets, (), resource_ids=())

    result = stack["agent"].optimize_from_route_planning_run(
        stored_run.id,
        as_of=AS_OF,
        config=ResponseOptimizationConfig(population_size=4, generation_count=1, random_seed=3),
    )
    plan = stack["plans"].get_by_id(result.response_plan_id).plan

    assert result.success is True
    assert plan.status is ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS
    assert plan.actions == ()
    assert plan.uncovered_target_ids == tuple(stored_target.id for stored_target in stored_targets.targets)


def test_route_planning_run_integration_is_deterministic_logically(stack):
    stored_targets = target_set(stack)
    t1, t2 = [stored_target.id for stored_target in stored_targets.targets]
    stored_run = save_run(
        stack,
        stored_targets,
        (reachable("R1", t1, 100.0), reachable("R2", t2, 200.0)),
    )
    config = ResponseOptimizationConfig(population_size=4, generation_count=2, random_seed=11)

    first = stack["agent"].optimize_from_route_planning_run(stored_run.id, as_of=AS_OF, config=config)
    second = stack["agent"].optimize_from_route_planning_run(stored_run.id, as_of=AS_OF, config=config)

    assert first.response_plan_id != second.response_plan_id
    assert logical_plan(stack["plans"].get_by_id(first.response_plan_id).plan) == logical_plan(
        stack["plans"].get_by_id(second.response_plan_id).plan
    )
