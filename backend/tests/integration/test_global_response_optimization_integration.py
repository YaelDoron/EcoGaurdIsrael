"""Integration test (Task 42): real Stage-3 GlobalPlanningInputBuilder output
fed into the Stage-4 GlobalResponseOptimizationService. SQLite-backed (a
deterministic, directly-persisted road-network fixture - no live OSM
access), matching Task 42's explicit "SQLite is acceptable for pure
integration if the road-network fixture is deterministic" guidance.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_service import (
    GlobalResponseOptimizationService,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.models import GraphEdge, GraphNode
from src.models.resource_status import ResourceStatus
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.global_planning.global_candidate_collector import GlobalCandidateCollector
from src.services.global_planning.global_planning_input_builder import GlobalPlanningInputBuilder
from src.services.global_planning.global_route_matrix_builder import GlobalRouteMatrixBuilder
from src.services.operational.operational_context_service import OperationalContextService

AS_OF = datetime(2026, 9, 23, 9, 0, tzinfo=timezone.utc)
NODE_STATION_A = 1
NODE_STATION_B = 2
NODE_TARGET_A1 = 3
NODE_TARGET_B1 = 4


def _persist_fire_event(session, latitude, longitude) -> int:
    event = FireEventDB(
        latitude=latitude, longitude=longitude, detected_at=AS_OF, updated_at=AS_OF, status="confirmed",
        detection_confidence=0.9, methodology="m", methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    return event.id


def _persist_target_set(session, fire_event_id, latitude, longitude, priority_score=100.0) -> int:
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id, generated_at=AS_OF, methodology="m", methodology_version="1.0"
    )
    session.add(target_set)
    session.flush()
    session.add(
        ResponseTargetDB(
            response_target_set_id=target_set.id, fire_event_id=fire_event_id, target_order=1,
            target_type="active_fire", latitude=latitude, longitude=longitude, priority_score=priority_score,
        )
    )
    session.flush()
    return target_set.id


def _persist_station_and_resources(session, station_id, latitude, longitude, resource_ids, status=ResourceStatus.AVAILABLE) -> None:
    session.add(FireStationDB(id=station_id, name=station_id, latitude=latitude, longitude=longitude))
    session.flush()
    for resource_id in resource_ids:
        session.add(FirefightingResourceDB(id=resource_id, station_id=station_id, status=status))


def _persist_road_network(sqlite_session_factory) -> None:
    session = sqlite_session_factory()
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=NODE_STATION_A, latitude=32.70, longitude=35.00),
            GraphNode(id=NODE_STATION_B, latitude=32.90, longitude=35.20),
            GraphNode(id=NODE_TARGET_A1, latitude=32.701, longitude=35.001),
            GraphNode(id=NODE_TARGET_B1, latitude=32.901, longitude=35.201),
        ],
        edges=[
            GraphEdge(source_node_id=NODE_STATION_A, target_node_id=NODE_TARGET_A1, distance_meters=200.0, travel_time_seconds=30.0),
            GraphEdge(source_node_id=NODE_STATION_A, target_node_id=NODE_TARGET_B1, distance_meters=20000.0, travel_time_seconds=1800.0),
            GraphEdge(source_node_id=NODE_STATION_B, target_node_id=NODE_TARGET_A1, distance_meters=20000.0, travel_time_seconds=1800.0),
            GraphEdge(source_node_id=NODE_STATION_B, target_node_id=NODE_TARGET_B1, distance_meters=200.0, travel_time_seconds=30.0),
        ],
    )
    session.commit()
    session.close()


def _persist_commitment(sqlite_session_factory, fire_event_id, target_set_id, resource_id) -> int:
    session = sqlite_session_factory()
    route_run = RoutePlanningRunDB(
        fire_event_id=fire_event_id, response_target_set_id=target_set_id, planned_at=AS_OF,
        methodology="m", methodology_version="1.0", resource_ids=[],
    )
    session.add(route_run)
    session.flush()
    plan = ResponsePlanDB(
        fire_event_id=fire_event_id, response_target_set_id=target_set_id, route_planning_run_id=route_run.id,
        generated_at=AS_OF, status="complete", methodology="m", methodology_version="1.0", random_seed=1,
    )
    session.add(plan)
    session.flush()
    plan_id = plan.id
    session.commit()
    session.close()

    session2 = sqlite_session_factory()
    ResourceCommitmentRepository(sqlite_session_factory).replace_commitments_for_plan(
        session2, fire_event_id=fire_event_id, response_plan_id=plan_id, resource_ids=(resource_id,), committed_at=AS_OF,
    )
    session2.commit()
    session2.close()
    return plan_id


def _make_input_builder(sqlite_session_factory) -> GlobalPlanningInputBuilder:
    fire_station_repository = FireStationRepository(sqlite_session_factory)
    firefighting_resource_repository = FirefightingResourceRepository(sqlite_session_factory)
    return GlobalPlanningInputBuilder(
        global_planning_run_repository=GlobalPlanningRunRepository(sqlite_session_factory),
        response_target_repository=ResponseTargetRepository(sqlite_session_factory),
        candidate_collector=GlobalCandidateCollector(
            fire_station_repository=fire_station_repository,
            firefighting_resource_repository=firefighting_resource_repository,
            resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
            operational_context_service=OperationalContextService(
                fire_station_repository=fire_station_repository,
                firefighting_resource_repository=firefighting_resource_repository,
            ),
        ),
        route_matrix_builder=GlobalRouteMatrixBuilder(),
        road_network_repository=RoadNetworkRepository(),
        session_factory=sqlite_session_factory,
    )


@pytest.fixture
def two_event_scenario(sqlite_session_factory):
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    event_b = _persist_fire_event(session, 32.90, 35.20)
    target_set_a = _persist_target_set(session, event_a, 32.701, 35.001)
    target_set_b = _persist_target_set(session, event_b, 32.901, 35.201)
    _persist_station_and_resources(session, "STATION-A", 32.70, 35.00, ["R1"])
    _persist_station_and_resources(session, "STATION-B", 32.90, 35.20, ["R2"])
    session.commit()
    session.close()
    _persist_road_network(sqlite_session_factory)
    plan_a_id = _persist_commitment(sqlite_session_factory, event_a, target_set_a, "R1")

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event_a, event_b),
    )
    return {"event_a": event_a, "event_b": event_b, "run_id": stored_run.id, "plan_a_id": plan_a_id}


def _row_counts(sqlite_session_factory):
    session = sqlite_session_factory()
    try:
        return (
            len(session.execute(select(ResponsePlanDB)).scalars().all()),
            len(session.execute(select(ResponseTargetSetDB)).scalars().all()),
            len(session.execute(select(FirefightingResourceDB)).scalars().all()),
        )
    finally:
        session.close()


def test_real_stage3_input_flows_into_a_genuine_global_optimization(two_event_scenario, sqlite_session_factory):
    input_builder = _make_input_builder(sqlite_session_factory)
    before_counts = _row_counts(sqlite_session_factory)

    global_input = input_builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)
    result = GlobalResponseOptimizationService().optimize(
        global_input, GlobalResponseOptimizationConfig(population_size=30, generation_count=40, random_seed=42)
    )

    after_counts = _row_counts(sqlite_session_factory)
    assert before_counts == after_counts  # zero DB writes from optimization itself

    # One global optimization spanning both events.
    fire_event_ids_in_actions = {action.fire_event_id for action in result.actions}
    assert fire_event_ids_in_actions == {two_event_scenario["event_a"], two_event_scenario["event_b"]}

    # No resource appears twice.
    resource_ids = [action.resource_id for action in result.actions]
    assert len(resource_ids) == len(set(resource_ids))

    # Every action corresponds to a real Stage-3 route matrix entry.
    for action in result.actions:
        route = global_input.route_matrix.get(action.resource_id, action.response_target_id)
        assert route is not None
        assert route.eta_seconds == action.eta_seconds
        assert route.route_distance_meters == action.route_distance_meters

    # Event projections partition the global action set exactly.
    projected_actions = [action for event_result in result.event_results for action in event_result.actions]
    assert sorted(projected_actions, key=lambda a: a.resource_id) == sorted(result.actions, key=lambda a: a.resource_id)
    assert {er.fire_event_id for er in result.event_results} == {
        two_event_scenario["event_a"], two_event_scenario["event_b"]
    }
