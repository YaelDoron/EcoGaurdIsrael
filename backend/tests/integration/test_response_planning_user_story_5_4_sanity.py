"""US 5.4 final end-to-end sanity check: the fully REAL production pipeline, wired
exclusively through build_response_planning_refresh_orchestrator() (Task 6/6.1) -
real RoutePlanningAgent + real Dijkstra (over a tiny pre-cached road network, so
no OSM network call is made), real GeneticResponsePlanOptimizer, real
BaselineComparisonService, real ResponsePlanScorer. Nothing in this file is a
fake/test-double collaborator.

Scenario 1: first refresh -> REFRESHED, full traceability chain.
Scenario 2: identical second refresh -> NO_OP, zero new artifacts.
Scenario 3: a real resource-status change (AVAILABLE -> UNAVAILABLE via the real
FirefightingResourceRepository) -> REFRESHED again, cycle 1 fully preserved.
Then CurrentResponsePlanResolver sanity over the resulting two-cycle history.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.graph_edge_db import GraphEdgeDB
from src.database.models.graph_node_db import GraphNodeDB
from src.models import ResourceStatus, ResponseTarget, ResponseTargetSet, ResponseTargetType
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.plan_comparison_repository import PlanComparisonRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.response_planning import CurrentResponsePlanResolver, PlanningRefreshStatus
from src.services.response_planning.response_planning_production_factory import (
    build_response_planning_refresh_orchestrator,
)

FIRE_LAT, FIRE_LON = 32.7300, 35.0450
STATION_LAT, STATION_LON = 32.7290, 35.0440
STATION_ID = "station-1"
RESOURCE_ID = "truck-1"
STATION_NODE_ID = 1001
FIRE_NODE_ID = 1002

AS_OF = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def later(hours: int) -> datetime:
    return AS_OF + timedelta(hours=hours)


def seed_environment(session_factory) -> int:
    """Seed FireEvent, station+resource, and a tiny pre-cached road network (no OSM call)."""
    session = session_factory()
    event = FireEventDB(
        latitude=FIRE_LAT,
        longitude=FIRE_LON,
        detected_at=AS_OF - timedelta(hours=1),
        updated_at=AS_OF - timedelta(minutes=5),
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    fire_event_id = event.id

    session.add(FireStationDB(id=STATION_ID, name="Station 1", latitude=STATION_LAT, longitude=STATION_LON))
    session.flush()
    session.add(FirefightingResourceDB(id=RESOURCE_ID, station_id=STATION_ID, status=ResourceStatus.AVAILABLE))

    session.add_all(
        [
            GraphNodeDB(id=STATION_NODE_ID, latitude=STATION_LAT, longitude=STATION_LON),
            GraphNodeDB(id=FIRE_NODE_ID, latitude=FIRE_LAT, longitude=FIRE_LON),
        ]
    )
    session.flush()
    session.add(
        GraphEdgeDB(
            source_node_id=STATION_NODE_ID,
            target_node_id=FIRE_NODE_ID,
            distance_meters=500.0,
            travel_time_seconds=60.0,
        )
    )
    session.commit()
    session.close()
    return fire_event_id


def seed_target_set(session_factory, fire_event_id, as_of, priority_score=150.0):
    return ResponseTargetRepository(session_factory).save_target_set(
        ResponseTargetSet(
            fire_event_id=fire_event_id,
            generated_at=as_of,
            methodology="TEST_TARGETS",
            methodology_version="1.0",
            targets=(
                ResponseTarget(
                    fire_event_id=fire_event_id,
                    target_type=ResponseTargetType.ACTIVE_FIRE,
                    latitude=FIRE_LAT,
                    longitude=FIRE_LON,
                    priority_score=priority_score,
                ),
            ),
        )
    )


def counts(session_factory, fire_event_id):
    return {
        "runs": len(RoutePlanningRepository(session_factory).get_history_for_event(fire_event_id)),
        "plans": len(ResponsePlanRepository(session_factory).get_for_fire_event(fire_event_id)),
        "comparisons": len(PlanComparisonRepository(session_factory).list_for_fire_event(fire_event_id)),
    }


def test_full_real_production_pipeline_end_to_end(sqlite_session_factory):
    fire_event_id = seed_environment(sqlite_session_factory)
    seed_target_set(sqlite_session_factory, fire_event_id, AS_OF)

    orchestrator = build_response_planning_refresh_orchestrator(session_factory=sqlite_session_factory)

    # -----------------------------------------------------------------
    # Scenario 1: first complete refresh
    # -----------------------------------------------------------------
    first = orchestrator.refresh(fire_event_id=fire_event_id, as_of=AS_OF)

    assert first.status is PlanningRefreshStatus.REFRESHED, first.error
    assert first.route_planning_run_id is not None
    assert first.response_plan_id is not None
    assert first.comparison_id is not None

    response_plan_repository = ResponsePlanRepository(sqlite_session_factory)
    route_planning_repository = RoutePlanningRepository(sqlite_session_factory)
    sidecar_repository = ResponsePlanPlanningStateRepository(sqlite_session_factory)
    plan_comparison_repository = PlanComparisonRepository(sqlite_session_factory)

    stored_plan = response_plan_repository.get_by_id(first.response_plan_id)
    stored_run = route_planning_repository.get_by_id(first.route_planning_run_id)
    stored_sidecar = sidecar_repository.get_for_plan(first.response_plan_id)
    stored_comparison = plan_comparison_repository.get_by_id(first.comparison_id)

    # Traceability
    assert stored_plan.plan.fire_event_id == fire_event_id
    assert stored_plan.plan.response_target_set_id == stored_run.run.response_target_set_id
    assert stored_plan.plan.route_planning_run_id == stored_run.id
    assert RESOURCE_ID in stored_run.run.resource_ids
    assert stored_sidecar.response_plan_id == first.response_plan_id
    assert stored_comparison.comparison.optimized_plan_id == first.response_plan_id
    assert stored_comparison.comparison.route_planning_run_id == first.route_planning_run_id
    assert stored_comparison.comparison.response_target_set_id == stored_run.run.response_target_set_id

    counts_after_cycle_1 = counts(sqlite_session_factory, fire_event_id)
    assert counts_after_cycle_1 == {"runs": 1, "plans": 1, "comparisons": 1}
    fingerprint_1 = stored_sidecar.planning_effective_state_fingerprint

    # -----------------------------------------------------------------
    # Scenario 2: identical second refresh -> NO_OP, zero new artifacts
    # -----------------------------------------------------------------
    second = orchestrator.refresh(fire_event_id=fire_event_id, as_of=later(1))

    assert second.status is PlanningRefreshStatus.NO_OP
    assert second.route_planning_run_id == first.route_planning_run_id
    assert second.response_plan_id == first.response_plan_id
    assert second.comparison_id == first.comparison_id
    assert counts(sqlite_session_factory, fire_event_id) == counts_after_cycle_1

    # -----------------------------------------------------------------
    # Scenario 3: real resource-status change -> REFRESHED, append-only history
    # -----------------------------------------------------------------
    FirefightingResourceRepository(sqlite_session_factory).update_status(RESOURCE_ID, ResourceStatus.UNAVAILABLE)

    third = orchestrator.refresh(fire_event_id=fire_event_id, as_of=later(2))

    assert third.status is PlanningRefreshStatus.REFRESHED, third.error
    assert third.route_planning_run_id != first.route_planning_run_id
    assert third.response_plan_id != first.response_plan_id
    assert third.comparison_id != first.comparison_id

    new_run = route_planning_repository.get_by_id(third.route_planning_run_id)
    assert RESOURCE_ID not in new_run.run.resource_ids  # excluded now that it's UNAVAILABLE

    new_sidecar = sidecar_repository.get_for_plan(third.response_plan_id)
    assert new_sidecar.planning_effective_state_fingerprint != fingerprint_1

    counts_after_cycle_2 = counts(sqlite_session_factory, fire_event_id)
    assert counts_after_cycle_2 == {"runs": 2, "plans": 2, "comparisons": 2}

    # Cycle 1 artifacts are fully preserved, byte-identical.
    assert response_plan_repository.get_by_id(first.response_plan_id) == stored_plan
    assert route_planning_repository.get_by_id(first.route_planning_run_id) == stored_run
    assert sidecar_repository.get_for_plan(first.response_plan_id) == stored_sidecar
    assert plan_comparison_repository.get_by_id(first.comparison_id) == stored_comparison

    # -----------------------------------------------------------------
    # CurrentResponsePlanResolver sanity
    # -----------------------------------------------------------------
    resolver = CurrentResponsePlanResolver(response_plan_repository, sidecar_repository)
    current = resolver.resolve(fire_event_id=fire_event_id)

    assert current.id == third.response_plan_id  # newest sidecar-backed plan
    # Cycle 1 remains queryable as history, not deleted/mutated.
    history = response_plan_repository.get_for_fire_event(fire_event_id)
    assert {p.id for p in history} == {first.response_plan_id, third.response_plan_id}
