"""Live integration test for GlobalPlanningInputBuilder (Stage 3 of the
Global Multi-Incident Optimizer refactor) against Neon: two active
FireEvents, two current ResponseTargetSets, two stations, multiple
resources, and one existing commitment, driven through the real production
factory. No FIRMS/IMS/Copernicus/agent access is required - the road
network is persisted directly (no live OSM fetch) so this test never
depends on network access beyond Neon itself. No DB mutation survives past
this test's cleanup.
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
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.models import GraphEdge, GraphNode
from src.models.resource_status import ResourceStatus
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.global_planning.global_planning_input_production_factory import (
    build_global_planning_input_builder,
)

pytestmark = pytest.mark.integration

METHODOLOGY_VERSION = "stage3-global-planning-input-integration"
AS_OF = datetime(2026, 9, 21, 11, 0, tzinfo=timezone.utc)
STATION_A_ID = "STAGE3-IT-STATION-A"
STATION_B_ID = "STAGE3-IT-STATION-B"
NODE_STATION_A = 910060001
NODE_STATION_B = 910060002
NODE_TARGET_A1 = 910060003
NODE_TARGET_B1 = 910060004


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
        run_ids = connection.execute(
            text(
                "SELECT global_planning_run_id FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :v"
                ") AND global_planning_run_id IS NOT NULL"
            ),
            {"v": METHODOLOGY_VERSION},
        ).scalars().all()
        run_ids_from_membership = connection.execute(
            text(
                "SELECT id FROM global_planning_runs WHERE id IN ("
                "SELECT global_planning_run_id FROM global_planning_run_events WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :v))"
            ),
            {"v": METHODOLOGY_VERSION},
        ).scalars().all()
        connection.execute(
            text(
                "DELETE FROM global_planning_run_events WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :v)"
            ),
            {"v": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM resource_commitments WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :v)"
            ),
            {"v": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :v)"
            ),
            {"v": METHODOLOGY_VERSION},
        )
        for run_id in set(list(run_ids) + list(run_ids_from_membership)):
            connection.execute(text("DELETE FROM global_planning_runs WHERE id = :id"), {"id": run_id})
        connection.execute(
            text(
                "DELETE FROM route_planning_runs WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :v)"
            ),
            {"v": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_targets WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :v)"
            ),
            {"v": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_target_sets WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :v)"
            ),
            {"v": METHODOLOGY_VERSION},
        )
        connection.execute(text("DELETE FROM fire_events WHERE methodology_version = :v"), {"v": METHODOLOGY_VERSION})
        connection.execute(
            text("DELETE FROM firefighting_resources WHERE station_id IN (:a, :b)"),
            {"a": STATION_A_ID, "b": STATION_B_ID},
        )
        connection.execute(
            text("DELETE FROM fire_stations WHERE id IN (:a, :b)"), {"a": STATION_A_ID, "b": STATION_B_ID}
        )
        connection.execute(
            text(
                "DELETE FROM graph_edges WHERE source_node_id IN (:n1, :n2, :n3, :n4) "
                "OR target_node_id IN (:n1, :n2, :n3, :n4)"
            ),
            {"n1": NODE_STATION_A, "n2": NODE_STATION_B, "n3": NODE_TARGET_A1, "n4": NODE_TARGET_B1},
        )
        connection.execute(
            text("DELETE FROM graph_nodes WHERE id IN (:n1, :n2, :n3, :n4)"),
            {"n1": NODE_STATION_A, "n2": NODE_STATION_B, "n3": NODE_TARGET_A1, "n4": NODE_TARGET_B1},
        )


def test_two_event_global_planning_input_neon():
    session = get_session_factory()()
    event_a = FireEventDB(
        latitude=31.5, longitude=34.5, detected_at=AS_OF, updated_at=AS_OF, status="confirmed",
        detection_confidence=0.9, methodology="STAGE3_IT_DETECTION", methodology_version=METHODOLOGY_VERSION,
    )
    event_b = FireEventDB(
        latitude=31.7, longitude=34.7, detected_at=AS_OF, updated_at=AS_OF, status="confirmed",
        detection_confidence=0.9, methodology="STAGE3_IT_DETECTION", methodology_version=METHODOLOGY_VERSION,
    )
    session.add(event_a)
    session.add(event_b)
    session.flush()
    event_a_id, event_b_id = event_a.id, event_b.id

    station_a = FireStationDB(id=STATION_A_ID, name="Stage 3 IT Station A", latitude=31.5, longitude=34.5)
    station_b = FireStationDB(id=STATION_B_ID, name="Stage 3 IT Station B", latitude=31.7, longitude=34.7)
    session.add(station_a)
    session.add(station_b)
    session.add(FirefightingResourceDB(id="STAGE3-IT-R1", station_id=STATION_A_ID, status=ResourceStatus.AVAILABLE))
    session.add(FirefightingResourceDB(id="STAGE3-IT-R2", station_id=STATION_B_ID, status=ResourceStatus.AVAILABLE))

    target_set_a = ResponseTargetSetDB(
        fire_event_id=event_a_id, generated_at=AS_OF, methodology="m", methodology_version="1.0"
    )
    target_set_b = ResponseTargetSetDB(
        fire_event_id=event_b_id, generated_at=AS_OF, methodology="m", methodology_version="1.0"
    )
    session.add(target_set_a)
    session.add(target_set_b)
    session.flush()
    session.add(
        ResponseTargetDB(
            response_target_set_id=target_set_a.id, fire_event_id=event_a_id, target_order=1,
            target_type="active_fire", latitude=31.501, longitude=34.501, priority_score=100.0,
        )
    )
    session.add(
        ResponseTargetDB(
            response_target_set_id=target_set_b.id, fire_event_id=event_b_id, target_order=1,
            target_type="active_fire", latitude=31.701, longitude=34.701, priority_score=100.0,
        )
    )
    session.commit()
    target_set_a_id, target_set_b_id = target_set_a.id, target_set_b.id
    session.close()

    RoadNetworkRepository().save_network(
        get_session_factory()(),
        nodes=[
            GraphNode(id=NODE_STATION_A, latitude=31.5, longitude=34.5),
            GraphNode(id=NODE_STATION_B, latitude=31.7, longitude=34.7),
            GraphNode(id=NODE_TARGET_A1, latitude=31.501, longitude=34.501),
            GraphNode(id=NODE_TARGET_B1, latitude=31.701, longitude=34.701),
        ],
        edges=[
            GraphEdge(source_node_id=NODE_STATION_A, target_node_id=NODE_TARGET_A1, distance_meters=200.0, travel_time_seconds=30.0),
            GraphEdge(source_node_id=NODE_STATION_A, target_node_id=NODE_TARGET_B1, distance_meters=20000.0, travel_time_seconds=1800.0),
            GraphEdge(source_node_id=NODE_STATION_B, target_node_id=NODE_TARGET_A1, distance_meters=20000.0, travel_time_seconds=1800.0),
            GraphEdge(source_node_id=NODE_STATION_B, target_node_id=NODE_TARGET_B1, distance_meters=200.0, travel_time_seconds=30.0),
        ],
    )

    # One existing commitment: R1 -> event A.
    route_run = RoutePlanningRunDB(
        fire_event_id=event_a_id, response_target_set_id=target_set_a_id, planned_at=AS_OF,
        methodology="m", methodology_version="1.0", resource_ids=[],
    )
    session2 = get_session_factory()()
    session2.add(route_run)
    session2.flush()
    plan = ResponsePlanDB(
        fire_event_id=event_a_id, response_target_set_id=target_set_a_id, route_planning_run_id=route_run.id,
        generated_at=AS_OF, status="complete", methodology="m", methodology_version="1.0", random_seed=1,
    )
    session2.add(plan)
    session2.commit()
    plan_id = plan.id
    session2.close()

    session3 = get_session_factory()()
    ResourceCommitmentRepository().replace_commitments_for_plan(
        session3, fire_event_id=event_a_id, response_plan_id=plan_id, resource_ids=("STAGE3-IT-R1",), committed_at=AS_OF,
    )
    session3.commit()
    session3.close()

    run_repository = GlobalPlanningRunRepository()
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event_a_id, event_b_id),
    )

    # No DB writes caused by input building: snapshot response_plans/route_planning_runs counts before/after.
    engine = get_engine()
    with engine.connect() as connection:
        plan_count_before = connection.execute(text("SELECT COUNT(*) FROM response_plans")).scalar()
        run_count_before = connection.execute(text("SELECT COUNT(*) FROM route_planning_runs")).scalar()

    builder = build_global_planning_input_builder()
    result = builder.build(global_planning_run_id=stored_run.id, as_of=AS_OF)

    with engine.connect() as connection:
        plan_count_after = connection.execute(text("SELECT COUNT(*) FROM response_plans")).scalar()
        run_count_after = connection.execute(text("SELECT COUNT(*) FROM route_planning_runs")).scalar()

    assert plan_count_after == plan_count_before
    assert run_count_after == run_count_before

    assert set(result.active_fire_event_ids) == {event_a_id, event_b_id}
    assert {t.fire_event_id for t in result.targets} == {event_a_id, event_b_id}
    assert {r.resource_id for r in result.resources} == {"STAGE3-IT-R1", "STAGE3-IT-R2"}

    r1 = next(r for r in result.resources if r.resource_id == "STAGE3-IT-R1")
    assert r1.current_commitment_fire_event_id == event_a_id
    assert r1.current_commitment_response_plan_id == plan_id

    target_b1 = next(t for t in result.targets if t.fire_event_id == event_b_id)
    cross_route = result.route_matrix.get("STAGE3-IT-R1", target_b1.response_target_id)
    assert cross_route is not None
    assert cross_route.fire_event_id == event_b_id
