"""Multi-incident resource-collision regression test (Global Multi-Incident
Optimizer refactor, Stage 0).

Reproduces the exact defect the architecture audit identified: two active
FireEvents with an OVERLAPPING candidate resource pool (a shared nearby
station) could previously both select the same AVAILABLE resource into
their current ResponsePlans. Unlike the pre-existing
test_two_incident_scenario_keeps_fire_event_planning_fully_isolated in
tests/integration/test_simulation_response_planning_e2e.py - whose fixture
uses geographically isolated stations and so never actually exercises the
collision path - this test deliberately shares one station between two
active incidents and proves Stage 0's cross-event exclusion prevents the
collision, using real repositories against SQLite.
"""
from __future__ import annotations

from datetime import datetime, timezone

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
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.operational.operational_context_service import OperationalContextService
from src.services.resource_reservation import CrossEventReservedResourceResolver

DETECTED_AT = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
SHARED_STATION_ID = "SHARED-STATION"
FIXTURE_SOURCE_NODE_ID = 9001
FIXTURE_TARGET_NODE_ID = 9002


def _station() -> FireStationDB:
    return FireStationDB(id=SHARED_STATION_ID, name="Shared Station", latitude=32.733, longitude=35.048)


def _persist_fire_event(session, **overrides) -> FireEventDB:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        detected_at=DETECTED_AT,
        updated_at=DETECTED_AT,
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    defaults.update(overrides)
    event = FireEventDB(**defaults)
    session.add(event)
    session.flush()
    return event


def _persist_planning_prerequisites(session, fire_event_id: int) -> tuple[int, int]:
    """Return (response_target_set_id, route_planning_run_id)."""
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id,
        generated_at=DETECTED_AT,
        methodology="TEST_TARGETS",
        methodology_version="1.0",
    )
    session.add(target_set)
    session.flush()
    route_run = RoutePlanningRunDB(
        fire_event_id=fire_event_id,
        response_target_set_id=target_set.id,
        planned_at=DETECTED_AT,
        methodology="TEST_ROUTING",
        methodology_version="1.0",
        resource_ids=[],
    )
    session.add(route_run)
    session.flush()
    return target_set.id, route_run.id


def _persist_response_target(session, response_target_set_id: int, fire_event_id: int, target_id: int) -> None:
    session.add(
        ResponseTargetDB(
            id=target_id,
            response_target_set_id=response_target_set_id,
            fire_event_id=fire_event_id,
            target_order=target_id,
            target_type="active_fire",
            latitude=32.731,
            longitude=35.046,
            priority_score=100.0,
        )
    )


def _persist_road_network(session) -> None:
    """Two fixed nodes reused as every route result's source/target - only
    their existence (for the FK) matters here, not real path geometry."""
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=FIXTURE_SOURCE_NODE_ID, latitude=32.700, longitude=35.000),
            GraphNode(id=FIXTURE_TARGET_NODE_ID, latitude=32.700, longitude=35.010),
        ],
        edges=[],
    )


def _persist_route_result(
    session,
    route_planning_run_id: int,
    resource_id: str,
    response_target_id: int,
    result_id: int,
) -> None:
    session.add(
        RouteResultDB(
            id=result_id,
            route_planning_run_id=route_planning_run_id,
            resource_id=resource_id,
            response_target_id=response_target_id,
            status="reachable",
            source_node_id=FIXTURE_SOURCE_NODE_ID,
            target_node_id=FIXTURE_TARGET_NODE_ID,
            node_path=[FIXTURE_SOURCE_NODE_ID, FIXTURE_TARGET_NODE_ID],
            distance_meters=500.0,
            travel_time_seconds=60.0,
        )
    )


def _make_current_plan(
    response_plan_repository: ResponsePlanRepository,
    response_plan_planning_state_repository: ResponsePlanPlanningStateRepository,
    *,
    fire_event_id: int,
    response_target_set_id: int,
    route_planning_run_id: int,
    resource_id: str,
    response_target_id: int,
    route_result_id: int,
):
    plan = ResponsePlan(
        fire_event_id=fire_event_id,
        response_target_set_id=response_target_set_id,
        route_planning_run_id=route_planning_run_id,
        generated_at=DETECTED_AT,
        status=ResponsePlanStatus.COMPLETE,
        methodology="GENETIC_RESOURCE_ALLOCATION",
        methodology_version="1.0",
        random_seed=1,
        actions=(ResponseAction(resource_id, response_target_id, route_result_id),),
        uncovered_target_ids=(),
        plan_score=90.0,
        coverage_score=100.0,
        average_eta_seconds=60.0,
    )
    stored = response_plan_repository.save(plan)
    response_plan_planning_state_repository.save(
        response_plan_id=stored.id, planning_effective_state_fingerprint="a" * 64
    )
    return stored


def _build_wired_service(sqlite_session_factory):
    fire_event_repository = FireEventRepository(sqlite_session_factory)
    firefighting_resource_repository = FirefightingResourceRepository(sqlite_session_factory)
    response_plan_repository = ResponsePlanRepository(sqlite_session_factory)
    response_plan_planning_state_repository = ResponsePlanPlanningStateRepository(sqlite_session_factory)
    resource_commitment_repository = ResourceCommitmentRepository(sqlite_session_factory)
    operational_context_service = OperationalContextService(
        firefighting_resource_repository=firefighting_resource_repository,
        cross_event_reserved_resource_resolver=CrossEventReservedResourceResolver(
            fire_event_repository=fire_event_repository,
            response_plan_repository=response_plan_repository,
            resource_commitment_repository=resource_commitment_repository,
        ),
    )
    return operational_context_service, response_plan_repository, response_plan_planning_state_repository


def test_overlapping_resource_pool_does_not_double_book_across_two_active_incidents(sqlite_session_factory):
    operational_context_service, response_plan_repository, response_plan_planning_state_repository = (
        _build_wired_service(sqlite_session_factory)
    )

    session = sqlite_session_factory()
    _persist_road_network(session)
    # 1. R1 (and an alternative, R2) are AVAILABLE at a station both incidents can reach.
    event_a = _persist_fire_event(session, latitude=32.731, longitude=35.046)
    event_b = _persist_fire_event(session, latitude=32.735, longitude=35.050)
    session.add(_station())
    session.add(FirefightingResourceDB(id="R1", station_id=SHARED_STATION_ID, status=ResourceStatus.AVAILABLE))
    session.add(FirefightingResourceDB(id="R2", station_id=SHARED_STATION_ID, status=ResourceStatus.AVAILABLE))
    target_set_a_id, route_run_a_id = _persist_planning_prerequisites(session, event_a.id)
    target_set_b_id, route_run_b_id = _persist_planning_prerequisites(session, event_b.id)
    _persist_response_target(session, target_set_a_id, event_a.id, target_id=10)
    _persist_response_target(session, target_set_b_id, event_b.id, target_id=20)
    session.flush()
    _persist_route_result(session, route_run_a_id, "R1", response_target_id=10, result_id=100)
    _persist_route_result(session, route_run_b_id, "R2", response_target_id=20, result_id=200)
    session.commit()
    session.close()

    # 2 & 3. FireEvent A is planned; its current ResponsePlan uses R1.
    _make_current_plan(
        response_plan_repository,
        response_plan_planning_state_repository,
        fire_event_id=event_a.id,
        response_target_set_id=target_set_a_id,
        route_planning_run_id=route_run_a_id,
        resource_id="R1",
        response_target_id=10,
        route_result_id=100,
    )

    # 4 & 5. FireEvent B is planned afterward - its candidate pool must exclude R1.
    eligible_for_b = operational_context_service.get_available_resources(
        [_station()], excluded_fire_event_id=event_b.id
    )
    eligible_resource_ids = {resource.id for resource in eligible_for_b}
    assert eligible_resource_ids == {"R2"}
    assert "R1" not in eligible_resource_ids

    # 6. B's ResponsePlan forms using the remaining alternative, never R1.
    _make_current_plan(
        response_plan_repository,
        response_plan_planning_state_repository,
        fire_event_id=event_b.id,
        response_target_set_id=target_set_b_id,
        route_planning_run_id=route_run_b_id,
        resource_id="R2",
        response_target_id=20,
        route_result_id=200,
    )

    # R1 appears in at most one current plan across the two active incidents.
    all_current_plan_resource_ids = response_plan_repository.get_current_plan_resource_ids_for_fire_events(
        [event_a.id, event_b.id]
    )
    assert all_current_plan_resource_ids == {"R1", "R2"}
    plan_a = response_plan_repository.get_latest_for_fire_event(event_a.id)
    plan_b = response_plan_repository.get_latest_for_fire_event(event_b.id)
    resource_ids_in_a = {action.resource_id for action in plan_a.plan.actions}
    resource_ids_in_b = {action.resource_id for action in plan_b.plan.actions}
    assert resource_ids_in_a == {"R1"}
    assert resource_ids_in_b == {"R2"}
    assert resource_ids_in_a.isdisjoint(resource_ids_in_b)


def test_same_fire_event_replanning_keeps_its_own_current_plan_resource_eligible(sqlite_session_factory):
    """Task 10: FireEvent A's own current plan uses R1; when A itself
    replans, R1 must remain eligible - only OTHER active events' current
    plans are excluded."""
    operational_context_service, response_plan_repository, response_plan_planning_state_repository = (
        _build_wired_service(sqlite_session_factory)
    )

    session = sqlite_session_factory()
    _persist_road_network(session)
    event_a = _persist_fire_event(session)
    session.add(_station())
    session.add(FirefightingResourceDB(id="R1", station_id=SHARED_STATION_ID, status=ResourceStatus.AVAILABLE))
    target_set_id, route_run_id = _persist_planning_prerequisites(session, event_a.id)
    _persist_response_target(session, target_set_id, event_a.id, target_id=10)
    session.flush()
    _persist_route_result(session, route_run_id, "R1", response_target_id=10, result_id=100)
    session.commit()
    session.close()

    _make_current_plan(
        response_plan_repository,
        response_plan_planning_state_repository,
        fire_event_id=event_a.id,
        response_target_set_id=target_set_id,
        route_planning_run_id=route_run_id,
        resource_id="R1",
        response_target_id=10,
        route_result_id=100,
    )

    eligible_for_a_replan = operational_context_service.get_available_resources(
        [_station()], excluded_fire_event_id=event_a.id
    )

    assert {resource.id for resource in eligible_for_a_replan} == {"R1"}


def test_resolved_event_does_not_reserve_its_former_resource(sqlite_session_factory):
    """Task 7: a RESOLVED FireEvent's old current plan must not block its
    resource for other active events."""
    operational_context_service, response_plan_repository, response_plan_planning_state_repository = (
        _build_wired_service(sqlite_session_factory)
    )

    session = sqlite_session_factory()
    _persist_road_network(session)
    event_resolved = _persist_fire_event(session, status="confirmed")  # active while planned...
    event_b = _persist_fire_event(session, latitude=32.735, longitude=35.050)
    session.add(_station())
    session.add(FirefightingResourceDB(id="R1", station_id=SHARED_STATION_ID, status=ResourceStatus.AVAILABLE))
    target_set_id, route_run_id = _persist_planning_prerequisites(session, event_resolved.id)
    _persist_response_target(session, target_set_id, event_resolved.id, target_id=10)
    session.flush()
    _persist_route_result(session, route_run_id, "R1", response_target_id=10, result_id=100)
    session.commit()
    session.close()

    _make_current_plan(
        response_plan_repository,
        response_plan_planning_state_repository,
        fire_event_id=event_resolved.id,
        response_target_set_id=target_set_id,
        route_planning_run_id=route_run_id,
        resource_id="R1",
        response_target_id=10,
        route_result_id=100,
    )

    # ...then the event resolves.
    session = sqlite_session_factory()
    db_event = session.get(FireEventDB, event_resolved.id)
    db_event.status = "resolved"
    session.commit()
    session.close()

    eligible_for_b = operational_context_service.get_available_resources(
        [_station()], excluded_fire_event_id=event_b.id
    )

    assert {resource.id for resource in eligible_for_b} == {"R1"}
