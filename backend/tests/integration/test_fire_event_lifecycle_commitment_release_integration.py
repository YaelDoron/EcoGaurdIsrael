"""Live integration test for FireEventLifecycleService (Stage 1.1 of the
Global Multi-Incident Optimizer refactor) against Neon: resolving/dismissing
a FireEvent through the real production lifecycle path must immediately
release its resource commitments while leaving its historical ResponsePlan/
ResponseAction rows untouched. No FIRMS/IMS/Copernicus/agent access is
required - only this project's own Neon/PostgreSQL database. No DB mutation
survives past this test's cleanup.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
from src.models.fire_event_status import FireEventStatus
from src.models.resource_status import ResourceStatus
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.fire_event_lifecycle import FireEventLifecycleService
from src.services.resource_reservation import ResponsePlanActivationService

pytestmark = pytest.mark.integration

METHODOLOGY_VERSION = "stage1-1-lifecycle-release-integration"
DETECTED_AT = datetime(2026, 9, 19, 9, 0, tzinfo=timezone.utc)
STATION_ID = "STAGE1-1-LC-STATION"
RESOURCE_R1 = "STAGE1-1-LC-R1"
SOURCE_NODE_ID = 910050001
TARGET_NODE_ID = 910050002


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
                "DELETE FROM resource_commitments WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
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


def _build_active_event_with_committed_resource(resource_id: str) -> tuple[int, int]:
    """Persist a station+resource+active FireEvent, activate a ResponsePlan
    that commits `resource_id` to it, and return (fire_event_id, response_plan_id)."""
    session = get_session_factory()()
    station = FireStationDB(id=STATION_ID, name="Stage 1.1 LC Station", latitude=31.5, longitude=34.5)
    session.add(station)
    session.add(FirefightingResourceDB(id=resource_id, station_id=STATION_ID, status=ResourceStatus.AVAILABLE))
    event = FireEventDB(
        latitude=31.5,
        longitude=34.5,
        detected_at=DETECTED_AT,
        updated_at=DETECTED_AT,
        status="confirmed",
        detection_confidence=0.9,
        methodology="STAGE1_1_LC_DETECTION",
        methodology_version=METHODOLOGY_VERSION,
    )
    session.add(event)
    session.commit()
    fire_event_id = event.id
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=SOURCE_NODE_ID, latitude=31.5, longitude=34.5),
            GraphNode(id=TARGET_NODE_ID, latitude=31.5, longitude=34.5),
        ],
        edges=[],
    )
    session.commit()
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id, generated_at=DETECTED_AT, methodology="m", methodology_version="1.0"
    )
    session.add(target_set)
    session.flush()
    route_run = RoutePlanningRunDB(
        fire_event_id=fire_event_id,
        response_target_set_id=target_set.id,
        planned_at=DETECTED_AT,
        methodology="m",
        methodology_version="1.0",
        resource_ids=[],
    )
    session.add(route_run)
    session.flush()
    target = ResponseTargetDB(
        response_target_set_id=target_set.id,
        fire_event_id=fire_event_id,
        target_order=1,
        target_type="active_fire",
        latitude=31.5,
        longitude=34.5,
        priority_score=100.0,
    )
    session.add(target)
    session.flush()
    route_result = RouteResultDB(
        route_planning_run_id=route_run.id,
        resource_id=resource_id,
        response_target_id=target.id,
        status="reachable",
        source_node_id=SOURCE_NODE_ID,
        target_node_id=TARGET_NODE_ID,
        node_path=[SOURCE_NODE_ID, TARGET_NODE_ID],
        distance_meters=500.0,
        travel_time_seconds=60.0,
    )
    session.add(route_result)
    session.commit()
    target_set_id, run_id, target_id, route_result_id = target_set.id, route_run.id, target.id, route_result.id
    session.close()

    response_plan_repository = ResponsePlanRepository()
    stored_plan = response_plan_repository.save(
        ResponsePlan(
            fire_event_id=fire_event_id,
            response_target_set_id=target_set_id,
            route_planning_run_id=run_id,
            generated_at=DETECTED_AT,
            status=ResponsePlanStatus.COMPLETE,
            methodology="GENETIC_RESOURCE_ALLOCATION",
            methodology_version="1.0",
            random_seed=1,
            actions=(ResponseAction(resource_id, target_id, route_result_id),),
            uncovered_target_ids=(),
            plan_score=90.0,
            coverage_score=100.0,
            average_eta_seconds=60.0,
        )
    )
    activation_service = ResponsePlanActivationService(
        response_plan_repository=response_plan_repository,
        resource_commitment_repository=ResourceCommitmentRepository(),
    )
    activation_service.activate(
        response_plan_id=stored_plan.id, planning_effective_state_fingerprint="a" * 64, as_of=DETECTED_AT
    )
    return fire_event_id, stored_plan.id


def test_resolve_through_real_lifecycle_path_releases_commitment_and_preserves_history_neon():
    resource_id = RESOURCE_R1
    fire_event_id, response_plan_id = _build_active_event_with_committed_resource(resource_id)
    resource_commitment_repository = ResourceCommitmentRepository()
    assert resource_commitment_repository.get_by_resource_id(resource_id) is not None

    lifecycle_service = FireEventLifecycleService()
    result = lifecycle_service.resolve_event(fire_event_id, as_of=DETECTED_AT + timedelta(hours=1))

    assert result.event.status is FireEventStatus.RESOLVED

    engine = get_engine()
    with engine.connect() as connection:
        db_status = connection.execute(
            text("SELECT status FROM fire_events WHERE id = :id"), {"id": fire_event_id}
        ).scalar()
        commitment_count = connection.execute(
            text("SELECT COUNT(*) FROM resource_commitments WHERE resource_id = :rid"), {"rid": resource_id}
        ).scalar()
        plan_count = connection.execute(
            text("SELECT COUNT(*) FROM response_plans WHERE id = :id"), {"id": response_plan_id}
        ).scalar()
        action_count = connection.execute(
            text("SELECT COUNT(*) FROM response_actions WHERE response_plan_id = :id"), {"id": response_plan_id}
        ).scalar()

    assert db_status == "resolved"
    assert commitment_count == 0
    assert plan_count == 1
    assert action_count == 1


def test_dismiss_through_real_lifecycle_path_releases_commitment_neon():
    resource_id = f"{RESOURCE_R1}-DISMISS"
    fire_event_id, _ = _build_active_event_with_committed_resource(resource_id)
    resource_commitment_repository = ResourceCommitmentRepository()
    assert resource_commitment_repository.get_by_resource_id(resource_id) is not None

    lifecycle_service = FireEventLifecycleService()
    result = lifecycle_service.dismiss_event(fire_event_id, as_of=DETECTED_AT + timedelta(hours=1))

    assert result.event.status is FireEventStatus.DISMISSED
    assert resource_commitment_repository.get_by_resource_id(resource_id) is None
