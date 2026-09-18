"""Live integration test for Stage 0 cross-event resource reservation
(Global Multi-Incident Optimizer refactor) against Neon.

Mirrors tests/services/resource_reservation/test_multi_incident_resource_
collision_regression.py's SQLite scenario against the real database: two
active FireEvents share one station/resource candidate pool (unlike
test_two_incident_scenario_keeps_fire_event_planning_fully_isolated in
test_simulation_response_planning_e2e.py, whose fixture uses geographically
isolated stations and so never exercises the overlap path), and the same
resource must never be eligible for both events' current plans at once. No
FIRMS/IMS/Copernicus/agent access is required - only this project's own
Neon/PostgreSQL database. No DB mutation survives past this test's cleanup.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from src.config.settings import settings
from src.database.connection import get_engine, get_session_factory, init_db
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.route_result_db import RouteResultDB
from src.models import GraphNode, ResponseAction, ResponsePlan, ResponsePlanStatus
from src.models.resource_status import ResourceStatus
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.operational.operational_context_service import OperationalContextService
from src.services.resource_reservation import CrossEventReservedResourceResolver

pytestmark = pytest.mark.integration

METHODOLOGY_VERSION = "stage0-cross-event-reservation-integration"
DETECTED_AT = datetime(2026, 5, 5, 5, 5, 5, tzinfo=timezone.utc)
LATITUDE_A = 31.55555
LONGITUDE_A = 34.55555
LATITUDE_B = 31.55600
LONGITUDE_B = 34.55600
STATION_ID = "STAGE0-IT-STATION"
RESOURCE_R1 = "STAGE0-IT-R1"
RESOURCE_R2 = "STAGE0-IT-R2"
SOURCE_NODE_ID = 910030001
TARGET_NODE_ID = 910030002


@pytest.fixture(autouse=True)
def _require_database_url() -> None:
    if not settings.DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured; skipping live Neon integration test.")


@pytest.fixture(autouse=True)
def _clean_test_rows(_require_database_url):
    init_db()
    _delete_test_rows()
    yield
    _delete_test_rows()


def _delete_test_rows() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM response_plan_planning_states WHERE response_plan_id IN ("
                "SELECT id FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version"
                "))"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_actions WHERE response_plan_id IN ("
                "SELECT id FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version"
                "))"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM route_results WHERE route_planning_run_id IN ("
                "SELECT id FROM route_planning_runs WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version"
                "))"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM route_planning_runs WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_targets WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_target_sets WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text("DELETE FROM fire_events WHERE methodology_version = :methodology_version"),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text("DELETE FROM firefighting_resources WHERE station_id = :station_id"),
            {"station_id": STATION_ID},
        )
        connection.execute(text("DELETE FROM fire_stations WHERE id = :station_id"), {"station_id": STATION_ID})
        connection.execute(
            text("DELETE FROM graph_nodes WHERE id IN (:source_id, :target_id)"),
            {"source_id": SOURCE_NODE_ID, "target_id": TARGET_NODE_ID},
        )


def test_overlapping_resource_pool_does_not_double_book_across_two_active_incidents_neon():
    session = get_session_factory()()
    station = FireStationDB(id=STATION_ID, name="Stage 0 IT Station", latitude=LATITUDE_A, longitude=LONGITUDE_A)
    session.add(station)
    session.add(FirefightingResourceDB(id=RESOURCE_R1, station_id=STATION_ID, status=ResourceStatus.AVAILABLE))
    session.add(FirefightingResourceDB(id=RESOURCE_R2, station_id=STATION_ID, status=ResourceStatus.AVAILABLE))
    event_a = FireEventDB(
        latitude=LATITUDE_A,
        longitude=LONGITUDE_A,
        detected_at=DETECTED_AT,
        updated_at=DETECTED_AT,
        status="confirmed",
        detection_confidence=0.9,
        methodology="STAGE0_IT_DETECTION",
        methodology_version=METHODOLOGY_VERSION,
    )
    event_b = FireEventDB(
        latitude=LATITUDE_B,
        longitude=LONGITUDE_B,
        detected_at=DETECTED_AT,
        updated_at=DETECTED_AT,
        status="confirmed",
        detection_confidence=0.9,
        methodology="STAGE0_IT_DETECTION",
        methodology_version=METHODOLOGY_VERSION,
    )
    session.add(event_a)
    session.add(event_b)
    session.commit()
    event_a_id, event_b_id = event_a.id, event_b.id
    session.close()

    response_plan_repository = ResponsePlanRepository()
    response_plan_planning_state_repository = ResponsePlanPlanningStateRepository()
    firefighting_resource_repository = FirefightingResourceRepository()
    fire_event_repository = FireEventRepository()
    operational_context_service = OperationalContextService(
        firefighting_resource_repository=firefighting_resource_repository,
        cross_event_reserved_resource_resolver=CrossEventReservedResourceResolver(
            fire_event_repository=fire_event_repository,
            response_plan_repository=response_plan_repository,
        ),
    )

    session = get_session_factory()()
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=SOURCE_NODE_ID, latitude=LATITUDE_A, longitude=LONGITUDE_A),
            GraphNode(id=TARGET_NODE_ID, latitude=LATITUDE_A, longitude=LONGITUDE_A),
        ],
        edges=[],
    )
    target_set_a = ResponseTargetSetDB(
        fire_event_id=event_a_id, generated_at=DETECTED_AT, methodology="TEST_TARGETS", methodology_version="1.0"
    )
    target_set_b = ResponseTargetSetDB(
        fire_event_id=event_b_id, generated_at=DETECTED_AT, methodology="TEST_TARGETS", methodology_version="1.0"
    )
    session.add(target_set_a)
    session.add(target_set_b)
    session.flush()
    route_run_a = RoutePlanningRunDB(
        fire_event_id=event_a_id,
        response_target_set_id=target_set_a.id,
        planned_at=DETECTED_AT,
        methodology="TEST_ROUTING",
        methodology_version="1.0",
        resource_ids=[],
    )
    route_run_b = RoutePlanningRunDB(
        fire_event_id=event_b_id,
        response_target_set_id=target_set_b.id,
        planned_at=DETECTED_AT,
        methodology="TEST_ROUTING",
        methodology_version="1.0",
        resource_ids=[],
    )
    session.add(route_run_a)
    session.add(route_run_b)
    session.flush()
    target_a = ResponseTargetDB(
        response_target_set_id=target_set_a.id,
        fire_event_id=event_a_id,
        target_order=1,
        target_type="active_fire",
        latitude=LATITUDE_A,
        longitude=LONGITUDE_A,
        priority_score=100.0,
    )
    target_b = ResponseTargetDB(
        response_target_set_id=target_set_b.id,
        fire_event_id=event_b_id,
        target_order=1,
        target_type="active_fire",
        latitude=LATITUDE_B,
        longitude=LONGITUDE_B,
        priority_score=100.0,
    )
    session.add(target_a)
    session.add(target_b)
    session.flush()
    route_result_a = RouteResultDB(
        route_planning_run_id=route_run_a.id,
        resource_id=RESOURCE_R1,
        response_target_id=target_a.id,
        status="reachable",
        source_node_id=SOURCE_NODE_ID,
        target_node_id=TARGET_NODE_ID,
        node_path=[SOURCE_NODE_ID, TARGET_NODE_ID],
        distance_meters=500.0,
        travel_time_seconds=60.0,
    )
    route_result_b = RouteResultDB(
        route_planning_run_id=route_run_b.id,
        resource_id=RESOURCE_R2,
        response_target_id=target_b.id,
        status="reachable",
        source_node_id=SOURCE_NODE_ID,
        target_node_id=TARGET_NODE_ID,
        node_path=[SOURCE_NODE_ID, TARGET_NODE_ID],
        distance_meters=500.0,
        travel_time_seconds=60.0,
    )
    session.add(route_result_a)
    session.add(route_result_b)
    session.commit()
    target_a_id, target_b_id = target_a.id, target_b.id
    target_set_a_id, target_set_b_id = target_set_a.id, target_set_b.id
    route_run_a_id, route_run_b_id = route_run_a.id, route_run_b.id
    session.close()

    # FireEvent A is planned; its current ResponsePlan uses R1.
    plan_a = response_plan_repository.save(
        ResponsePlan(
            fire_event_id=event_a_id,
            response_target_set_id=target_set_a_id,
            route_planning_run_id=route_run_a_id,
            generated_at=DETECTED_AT,
            status=ResponsePlanStatus.COMPLETE,
            methodology="GENETIC_RESOURCE_ALLOCATION",
            methodology_version="1.0",
            random_seed=1,
            actions=(ResponseAction(RESOURCE_R1, target_a_id, route_result_a.id),),
            uncovered_target_ids=(),
            plan_score=90.0,
            coverage_score=100.0,
            average_eta_seconds=60.0,
        )
    )
    response_plan_planning_state_repository.save(
        response_plan_id=plan_a.id, planning_effective_state_fingerprint="a" * 64
    )

    # FireEvent B is planned afterward - its candidate pool must exclude R1.
    eligible_for_b = operational_context_service.get_available_resources(
        [station], excluded_fire_event_id=event_b_id
    )
    eligible_resource_ids = {resource.id for resource in eligible_for_b}

    assert eligible_resource_ids == {RESOURCE_R2}
    assert RESOURCE_R1 not in eligible_resource_ids

    plan_b = response_plan_repository.save(
        ResponsePlan(
            fire_event_id=event_b_id,
            response_target_set_id=target_set_b_id,
            route_planning_run_id=route_run_b_id,
            generated_at=DETECTED_AT,
            status=ResponsePlanStatus.COMPLETE,
            methodology="GENETIC_RESOURCE_ALLOCATION",
            methodology_version="1.0",
            random_seed=1,
            actions=(ResponseAction(RESOURCE_R2, target_b_id, route_result_b.id),),
            uncovered_target_ids=(),
            plan_score=90.0,
            coverage_score=100.0,
            average_eta_seconds=60.0,
        )
    )
    response_plan_planning_state_repository.save(
        response_plan_id=plan_b.id, planning_effective_state_fingerprint="b" * 64
    )

    all_current_plan_resource_ids = response_plan_repository.get_current_plan_resource_ids_for_fire_events(
        [event_a_id, event_b_id]
    )
    assert all_current_plan_resource_ids == {RESOURCE_R1, RESOURCE_R2}
