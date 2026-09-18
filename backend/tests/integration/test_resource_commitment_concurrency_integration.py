"""Live concurrency integration tests for ResponsePlanActivationService
(Stage 1 of the Global Multi-Incident Optimizer refactor) against Neon.

These use real OS threads plus a barrier to force two activation calls to
race for real, inside real overlapping DB transactions - a sequential call
pattern (call A, then call B) cannot exercise SELECT ... FOR UPDATE's
actual blocking/serialization behavior or the resource_commitments PK as a
genuine concurrent-write safety net, only its logic in isolation. No FIRMS/
IMS/Copernicus/agent access is required - only this project's own Neon/
PostgreSQL database. No DB mutation survives past this test's cleanup.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
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
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.resource_reservation import ResourceCommitmentConflict, ResponsePlanActivationService

pytestmark = pytest.mark.integration

METHODOLOGY_VERSION = "stage1-commitment-concurrency-integration"
DETECTED_AT = datetime(2026, 6, 6, 6, 6, 6, tzinfo=timezone.utc)
STATION_ID = "STAGE1-CC-STATION"
SOURCE_NODE_ID = 910040001
TARGET_NODE_ID = 910040002


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


def _persist_fire_event(session, latitude: float, longitude: float) -> int:
    event = FireEventDB(
        latitude=latitude,
        longitude=longitude,
        detected_at=DETECTED_AT,
        updated_at=DETECTED_AT,
        status="confirmed",
        detection_confidence=0.9,
        methodology="STAGE1_CC_DETECTION",
        methodology_version=METHODOLOGY_VERSION,
    )
    session.add(event)
    session.flush()
    return event.id


def _persist_plan_prerequisites(session, fire_event_id: int) -> tuple[int, int]:
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
    return target_set.id, route_run.id


def _persist_target_and_route_result(
    session, target_set_id: int, fire_event_id: int, target_order: int, resource_id: str, route_run_id: int
) -> tuple[int, int]:
    target = ResponseTargetDB(
        response_target_set_id=target_set_id,
        fire_event_id=fire_event_id,
        target_order=target_order,
        target_type="active_fire",
        latitude=32.7,
        longitude=35.0,
        priority_score=100.0,
    )
    session.add(target)
    session.flush()
    route_result = RouteResultDB(
        route_planning_run_id=route_run_id,
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
    session.flush()
    return target.id, route_result.id


def _make_plan(response_plan_repository, fire_event_id, target_set_id, run_id, resource_ids) -> int:
    session = get_session_factory()()
    existing_max_order = (
        session.query(ResponseTargetDB.target_order)
        .filter(ResponseTargetDB.response_target_set_id == target_set_id)
        .order_by(ResponseTargetDB.target_order.desc())
        .first()
    )
    order_start = (existing_max_order[0] + 1) if existing_max_order else 1
    actions = []
    for offset, resource_id in enumerate(resource_ids):
        target_id, route_result_id = _persist_target_and_route_result(
            session, target_set_id, fire_event_id, order_start + offset, resource_id, run_id
        )
        actions.append(ResponseAction(resource_id, target_id, route_result_id))
    session.commit()
    session.close()

    stored = response_plan_repository.save(
        ResponsePlan(
            fire_event_id=fire_event_id,
            response_target_set_id=target_set_id,
            route_planning_run_id=run_id,
            generated_at=DETECTED_AT,
            status=ResponsePlanStatus.COMPLETE,
            methodology="GENETIC_RESOURCE_ALLOCATION",
            methodology_version="1.0",
            random_seed=1,
            actions=tuple(actions),
            uncovered_target_ids=(),
            plan_score=90.0,
            coverage_score=100.0,
            average_eta_seconds=60.0,
        )
    )
    return stored.id


def _run_concurrently(fn_a, fn_b):
    """Release both callables at (as close to) the same instant via a barrier,
    each on its own thread, and return their (result, exception) outcomes."""
    barrier = threading.Barrier(2)
    outcomes = {}

    def _call(name, fn):
        barrier.wait()
        try:
            outcomes[name] = ("ok", fn())
        except Exception as exc:  # noqa: BLE001 - capturing for assertion, not swallowing
            outcomes[name] = ("error", exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_call, "a", fn_a)
        future_b = executor.submit(_call, "b", fn_b)
        future_a.result(timeout=60)
        future_b.result(timeout=60)
    return outcomes


def test_two_concurrent_fire_events_cannot_both_commit_the_same_resource_neon():
    session = get_session_factory()()
    station = FireStationDB(id=STATION_ID, name="Stage 1 CC Station", latitude=31.5, longitude=34.5)
    session.add(station)
    session.add(FirefightingResourceDB(id="STAGE1-CC-R1", station_id=STATION_ID, status=ResourceStatus.AVAILABLE))
    event_a_id = _persist_fire_event(session, 31.5, 34.5)
    event_b_id = _persist_fire_event(session, 31.51, 34.51)
    session.commit()
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=SOURCE_NODE_ID, latitude=31.5, longitude=34.5),
            GraphNode(id=TARGET_NODE_ID, latitude=31.5, longitude=34.5),
        ],
        edges=[],
    )
    session.commit()
    target_set_a_id, run_a_id = _persist_plan_prerequisites(session, event_a_id)
    target_set_b_id, run_b_id = _persist_plan_prerequisites(session, event_b_id)
    session.commit()
    session.close()

    response_plan_repository = ResponsePlanRepository()
    resource_commitment_repository = ResourceCommitmentRepository()
    activation_service = ResponsePlanActivationService(
        response_plan_repository=response_plan_repository,
        resource_commitment_repository=resource_commitment_repository,
    )

    plan_a_id = _make_plan(response_plan_repository, event_a_id, target_set_a_id, run_a_id, ("STAGE1-CC-R1",))
    plan_b_id = _make_plan(response_plan_repository, event_b_id, target_set_b_id, run_b_id, ("STAGE1-CC-R1",))

    outcomes = _run_concurrently(
        lambda: activation_service.activate(
            response_plan_id=plan_a_id, planning_effective_state_fingerprint="a" * 64, as_of=DETECTED_AT
        ),
        lambda: activation_service.activate(
            response_plan_id=plan_b_id, planning_effective_state_fingerprint="b" * 64, as_of=DETECTED_AT
        ),
    )

    statuses = {name: kind for name, (kind, _) in outcomes.items()}
    assert sorted(statuses.values()) == ["error", "ok"], (
        f"expected exactly one winner and one conflict, got {outcomes}"
    )
    (loser_name,) = [name for name, kind in statuses.items() if kind == "error"]
    assert isinstance(outcomes[loser_name][1], ResourceCommitmentConflict)

    engine = get_engine()
    with engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM resource_commitments WHERE resource_id = 'STAGE1-CC-R1'")
        ).scalar()
    assert count == 1

    committed = resource_commitment_repository.get_by_resource_id("STAGE1-CC-R1")
    winner_event_id = event_a_id if statuses["a"] == "ok" else event_b_id
    assert committed.fire_event_id == winner_event_id


def test_two_concurrent_activations_for_the_same_fire_event_serialize_without_a_mixed_commitment_set_neon():
    session = get_session_factory()()
    station = FireStationDB(id=STATION_ID, name="Stage 1 CC Station", latitude=31.5, longitude=34.5)
    session.add(station)
    for resource_id in ("STAGE1-CC-R1", "STAGE1-CC-R2", "STAGE1-CC-R3"):
        session.add(FirefightingResourceDB(id=resource_id, station_id=STATION_ID, status=ResourceStatus.AVAILABLE))
    event_a_id = _persist_fire_event(session, 31.5, 34.5)
    session.commit()
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=SOURCE_NODE_ID, latitude=31.5, longitude=34.5),
            GraphNode(id=TARGET_NODE_ID, latitude=31.5, longitude=34.5),
        ],
        edges=[],
    )
    session.commit()
    target_set_id, run_id = _persist_plan_prerequisites(session, event_a_id)
    session.commit()
    session.close()

    response_plan_repository = ResponsePlanRepository()
    resource_commitment_repository = ResourceCommitmentRepository()
    activation_service = ResponsePlanActivationService(
        response_plan_repository=response_plan_repository,
        resource_commitment_repository=resource_commitment_repository,
    )

    plan_1_resources = ("STAGE1-CC-R1", "STAGE1-CC-R2")
    plan_2_resources = ("STAGE1-CC-R2", "STAGE1-CC-R3")
    plan_1_id = _make_plan(response_plan_repository, event_a_id, target_set_id, run_id, plan_1_resources)
    plan_2_id = _make_plan(response_plan_repository, event_a_id, target_set_id, run_id, plan_2_resources)

    outcomes = _run_concurrently(
        lambda: activation_service.activate(
            response_plan_id=plan_1_id, planning_effective_state_fingerprint="1" * 64, as_of=DETECTED_AT
        ),
        lambda: activation_service.activate(
            response_plan_id=plan_2_id, planning_effective_state_fingerprint="2" * 64, as_of=DETECTED_AT
        ),
    )

    # Same FireEvent, no cross-event resource overlap issue possible - the
    # FireEvent-row lock only serializes these two writers, it does not make
    # either of them fail; both must succeed.
    assert {name: kind for name, (kind, _) in outcomes.items()} == {"a": "ok", "b": "ok"}, outcomes

    committed = {c.resource_id for c in resource_commitment_repository.get_for_fire_event(event_a_id)}
    assert committed in (set(plan_1_resources), set(plan_2_resources)), (
        f"final commitment set must match exactly one activation's resource set, not a mix: {committed}"
    )
