"""Blocker E, Task 11: the full deterministic dynamic multi-incident
scenario - one continuous chain of five real GlobalPlanningRefreshCoordinator
cycles against a real (SQLite) database, verifying active event set,
GlobalPlanningRun id, per-event/global demand and shortage, current
ResponsePlans, ResourceCommitments, dispatch locks, and assignment changes
at every step. No resource may ever appear assigned to two FireEvents at
once.

    T+0   Fire A active -> R1 dispatched to A.
    T+40  Fire B appears -> ONE global replan; A's lock on R1 is preserved;
          B is not yet reachable by anything free (nothing to assign).
    T+60  A's severity worsens to HIGH (minimum=2, desired=3) -> demand
          increases; R1 stays with A (never yanked just because demand
          grew); the one spare resource (R3) reinforces A.
    T+80  R2 (the only resource near B) goes UNAVAILABLE -> no spare
          resource remains (R1/R3 are both committed to A), so B correctly
          gets NO replacement - never fabricated.
    T+100 A resolves through FireEventLifecycleService -> R1/R3's locks are
          released; ONE global replan for the remaining active event (B
          only) follows; a freed resource reinforces B.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.models import GraphEdge, GraphNode
from src.models.dispatch_state import DispatchState
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.resource_status import ResourceStatus
from src.models.satellite_hotspot import SatelliteHotspot
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_event_lifecycle.fire_event_lifecycle_service import FireEventLifecycleService
from src.services.global_planning.current_global_assignment_loader import CurrentGlobalAssignmentLoader
from src.services.global_planning.global_candidate_collector import GlobalCandidateCollector
from src.services.global_planning.global_incident_demand_builder import GlobalIncidentDemandBuilder
from src.services.global_planning.global_planning_input_builder import GlobalPlanningInputBuilder
from src.services.global_planning.global_planning_refresh_coordinator import (
    GlobalPlanningRefreshCoordinator,
    GlobalPlanningRefreshStatus,
)
from src.services.global_planning.global_response_plan_activation_service import GlobalResponsePlanActivationService
from src.services.global_planning.global_route_matrix_builder import GlobalRouteMatrixBuilder
from src.services.operational.operational_context_service import OperationalContextService
from src.services.operational_refresh.resource_status_update_service import ResourceStatusUpdateService
from src.services.response_planning.current_response_plan_resolver import CurrentResponsePlanResolver

T0 = datetime(2026, 9, 25, 8, 0, tzinfo=timezone.utc)
T40 = T0 + timedelta(minutes=40)
T60 = T0 + timedelta(minutes=60)
T80 = T0 + timedelta(minutes=80)
T100 = T0 + timedelta(minutes=100)

# Node layout: 1=STATION-A, 2=fire A target, 3=STATION-B, 4=fire B target,
# 5=STATION-C (spare, reachable to both).
NODE_STATION_A, NODE_TARGET_A = 1, 2
NODE_STATION_B, NODE_TARGET_B = 3, 4
NODE_STATION_C = 5


def _persist_fire_event(session, latitude, longitude, detected_at=T0) -> int:
    event = FireEventDB(
        latitude=latitude, longitude=longitude, detected_at=detected_at, updated_at=detected_at,
        status="confirmed", detection_confidence=0.9, methodology="m", methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    return event.id


def _persist_target_set(session, fire_event_id, latitude, longitude, generated_at=T0) -> int:
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id, generated_at=generated_at, methodology="m", methodology_version="1.0"
    )
    session.add(target_set)
    session.flush()
    session.add(
        ResponseTargetDB(
            response_target_set_id=target_set.id, fire_event_id=fire_event_id, target_order=1,
            target_type="active_fire", latitude=latitude, longitude=longitude, priority_score=100.0,
        )
    )
    session.flush()
    return target_set.id


def _persist_station_and_resource(session, station_id, latitude, longitude, resource_id) -> None:
    session.add(FireStationDB(id=station_id, name=station_id, latitude=latitude, longitude=longitude))
    session.flush()
    session.add(FirefightingResourceDB(id=resource_id, station_id=station_id, status=ResourceStatus.AVAILABLE))


def _persist_severity(session_factory, fire_event_id, latitude, longitude, level, score, assessed_at) -> None:
    """Minimal VALID severity assessment (weather + satellite trace rows
    required by FireSeverityAssessmentRepository), mirroring
    test_global_planning_history_persistence_integration.py's own helper."""
    weather_repository = WeatherRepository(session_factory)
    station = WeatherStation(
        external_station_id=930000 + fire_event_id, name=f"WStation-{fire_event_id}",
        latitude=latitude, longitude=longitude,
    )
    weather_repository.save_station(station)
    weather_repository.save_observation(
        WeatherObservation(
            station_external_id=station.external_station_id, timestamp=assessed_at - timedelta(minutes=5),
            temperature=30.0, relative_humidity=25.0, wind_speed=20.0,
        )
    )
    weather_id = next(
        record.observation_id
        for record in weather_repository.get_recent_observations_for_area_candidates(
            latitude=latitude, longitude=longitude, radius_km=5.0,
            start_time=assessed_at - timedelta(minutes=30), end_time=assessed_at,
        )
        if record.observation.station_external_id == station.external_station_id
    )
    satellite_repository = SatelliteHotspotRepository(session_factory)
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=latitude, longitude=longitude, detected_at=assessed_at - timedelta(minutes=20),
            confidence="h", frp=72.0, satellite="N20",
        )
    )
    hotspot_id = satellite_repository.get_recent_hotspots(as_of=assessed_at, lookback_minutes=360)[0].id
    FireSeverityAssessmentRepository(session_factory).save_assessment(
        FireSeverityAssessment(
            fire_event_id=fire_event_id, assessed_at=assessed_at, status=FireSeverityAssessmentStatus.VALID,
            score=score, level=level, methodology="m", methodology_version="1.0",
        ),
        weather_observation_ids=(weather_id,), satellite_hotspot_ids=(hotspot_id,), selected_frp_hotspot_id=hotspot_id,
    )


def _make_coordinator(sqlite_session_factory) -> GlobalPlanningRefreshCoordinator:
    fire_station_repository = FireStationRepository(sqlite_session_factory)
    firefighting_resource_repository = FirefightingResourceRepository(sqlite_session_factory)
    resource_commitment_repository = ResourceCommitmentRepository(sqlite_session_factory)
    candidate_collector = GlobalCandidateCollector(
        fire_station_repository=fire_station_repository,
        firefighting_resource_repository=firefighting_resource_repository,
        resource_commitment_repository=resource_commitment_repository,
        operational_context_service=OperationalContextService(
            fire_station_repository=fire_station_repository,
            firefighting_resource_repository=firefighting_resource_repository,
        ),
    )
    current_global_assignment_loader = CurrentGlobalAssignmentLoader(
        current_response_plan_resolver=CurrentResponsePlanResolver(
            ResponsePlanRepository(sqlite_session_factory), ResponsePlanPlanningStateRepository(sqlite_session_factory)
        ),
        resource_commitment_repository=resource_commitment_repository,
    )
    input_builder = GlobalPlanningInputBuilder(
        global_planning_run_repository=GlobalPlanningRunRepository(sqlite_session_factory),
        response_target_repository=ResponseTargetRepository(sqlite_session_factory),
        candidate_collector=candidate_collector,
        incident_demand_builder=GlobalIncidentDemandBuilder(
            fire_severity_assessment_repository=FireSeverityAssessmentRepository(sqlite_session_factory)
        ),
        current_global_assignment_loader=current_global_assignment_loader,
        route_matrix_builder=GlobalRouteMatrixBuilder(),
        road_network_repository=RoadNetworkRepository(),
        session_factory=sqlite_session_factory,
    )
    activation_service = GlobalResponsePlanActivationService(
        response_plan_repository=ResponsePlanRepository(sqlite_session_factory),
        response_plan_planning_state_repository=ResponsePlanPlanningStateRepository(sqlite_session_factory),
        route_planning_repository=RoutePlanningRepository(sqlite_session_factory),
        resource_commitment_repository=resource_commitment_repository,
        session_factory=sqlite_session_factory,
    )
    return GlobalPlanningRefreshCoordinator(
        fire_event_repository=FireEventRepository(sqlite_session_factory),
        global_planning_run_repository=GlobalPlanningRunRepository(sqlite_session_factory),
        input_builder=input_builder,
        optimization_service=GlobalResponseOptimizationService(),
        activation_service=activation_service,
        config=GlobalResponseOptimizationConfig(population_size=24, generation_count=24, random_seed=42),
    )


def _committed_resource_ids_for_event(sqlite_session_factory, fire_event_id) -> set[str]:
    session = sqlite_session_factory()
    try:
        rows = session.execute(
            select(ResourceCommitmentDB).where(ResourceCommitmentDB.fire_event_id == fire_event_id)
        ).scalars().all()
        return {row.resource_id for row in rows}
    finally:
        session.close()


def _all_committed_resource_ids(sqlite_session_factory) -> list[str]:
    session = sqlite_session_factory()
    try:
        rows = session.execute(select(ResourceCommitmentDB)).scalars().all()
        return [row.resource_id for row in rows]
    finally:
        session.close()


def test_full_five_step_dynamic_multi_incident_scenario(sqlite_session_factory):
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00, detected_at=T0)
    _persist_target_set(session, event_a, 32.70, 35.00, generated_at=T0)
    _persist_station_and_resource(session, "STATION-A", 32.70, 35.001, "R1")
    _persist_station_and_resource(session, "STATION-C", 32.80, 35.10, "R3")  # spare, present from the start
    session.commit()
    session.close()
    _persist_road_network_for_scenario(sqlite_session_factory)

    coordinator = _make_coordinator(sqlite_session_factory)

    # ================= T+0: Fire A active -> R1 dispatched to A. =================
    step_0 = coordinator.refresh(trigger="fire_event_confirmed", as_of=T0)
    assert step_0.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert sorted(step_0.response_plan_ids_by_event) == [event_a]
    assert _committed_resource_ids_for_event(sqlite_session_factory, event_a) == {"R1"}
    run_id_0 = step_0.global_planning_run_id

    # ================= T+40: Fire B appears -> ONE global replan; A's lock preserved. =================
    session = sqlite_session_factory()
    event_b = _persist_fire_event(session, 32.90, 35.20, detected_at=T40)
    _persist_target_set(session, event_b, 32.90, 35.20, generated_at=T40)
    _persist_station_and_resource(session, "STATION-B", 32.90, 35.201, "R2")
    session.commit()
    session.close()

    step_40 = coordinator.refresh(trigger="fire_event_confirmed", as_of=T40)
    assert step_40.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert sorted(step_40.response_plan_ids_by_event) == sorted([event_a, event_b])
    assert step_40.global_planning_run_id != run_id_0  # a distinct, new GlobalPlanningRun
    assert _committed_resource_ids_for_event(sqlite_session_factory, event_a) == {"R1"}  # never yanked
    assert _committed_resource_ids_for_event(sqlite_session_factory, event_b) == {"R2"}
    run_id_40 = step_40.global_planning_run_id

    # No resource ever appears committed to two FireEvents.
    all_committed = _all_committed_resource_ids(sqlite_session_factory)
    assert len(all_committed) == len(set(all_committed))

    # ================= T+60: A's severity worsens to HIGH -> demand increases; R1 preserved, R3 reinforces. =================
    _persist_severity(
        sqlite_session_factory, event_a, 32.70, 35.00, FireSeverityLevel.HIGH, 88.0, T60 - timedelta(minutes=5)
    )

    step_60 = coordinator.refresh(trigger="severity_update", as_of=T60)
    assert step_60.status is GlobalPlanningRefreshStatus.ACTIVATED
    (event_a_result,) = [r for r in step_60.event_results if r.fire_event_id == event_a]
    assert event_a_result.demand_result.minimum_resources == 2
    assert event_a_result.demand_result.desired_resources == 3
    assert "R1" in _committed_resource_ids_for_event(sqlite_session_factory, event_a)  # never yanked
    assert _committed_resource_ids_for_event(sqlite_session_factory, event_a) == {"R1", "R3"}  # spare reinforces
    assert _committed_resource_ids_for_event(sqlite_session_factory, event_b) == {"R2"}  # B untouched
    assert step_60.shortage.total_required == 3  # 2 (A) + 1 (B)
    run_id_60 = step_60.global_planning_run_id
    assert run_id_60 not in (run_id_0, run_id_40)

    all_committed = _all_committed_resource_ids(sqlite_session_factory)
    assert len(all_committed) == len(set(all_committed))

    # ================= T+80: R2 (B's only resource) goes UNAVAILABLE -> no spare left; no replacement fabricated. =================
    status_service = ResourceStatusUpdateService(FirefightingResourceRepository(sqlite_session_factory))
    status_result = status_service.update_status(resource_id="R2", new_status=ResourceStatus.UNAVAILABLE)
    assert status_result.status.name == "UPDATED"

    step_80 = coordinator.refresh(trigger="resource_status_update", as_of=T80)
    assert step_80.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert _committed_resource_ids_for_event(sqlite_session_factory, event_a) == {"R1", "R3"}  # A still untouched
    assert _committed_resource_ids_for_event(sqlite_session_factory, event_b) == set()  # no spare exists to replace R2
    (event_b_result_80,) = [r for r in step_80.event_results if r.fire_event_id == event_b]
    assert event_b_result_80.demand_result.suppression_resources_assigned == 0
    assert event_b_result_80.demand_result.unmet_minimum == 1  # honestly reported, never fabricated

    all_committed = _all_committed_resource_ids(sqlite_session_factory)
    assert len(all_committed) == len(set(all_committed))

    # ================= T+100: A resolves -> locks released; ONE global replan for B; a freed resource reinforces B. =================
    lifecycle_service = FireEventLifecycleService(
        fire_event_repository=FireEventRepository(sqlite_session_factory),
        resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
        session_factory=sqlite_session_factory,
    )
    lifecycle_service.resolve_event(event_a, as_of=T100)
    assert _committed_resource_ids_for_event(sqlite_session_factory, event_a) == set()  # released immediately

    step_100 = coordinator.refresh(trigger="fire_event_resolved", as_of=T100)
    assert step_100.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert sorted(step_100.response_plan_ids_by_event) == [event_b]  # A no longer active - never replanned
    final_b_resources = _committed_resource_ids_for_event(sqlite_session_factory, event_b)
    assert final_b_resources, "a freed resource (R1 or R3) must reinforce B"
    assert final_b_resources <= {"R1", "R3"}
    assert len(final_b_resources) == 1  # B's desired demand is 1

    # Final invariant: no resource anywhere is committed to more than one FireEvent.
    all_committed = _all_committed_resource_ids(sqlite_session_factory)
    assert len(all_committed) == len(set(all_committed))
    session = sqlite_session_factory()
    a_status = session.get(FireEventDB, event_a).status
    session.close()
    assert a_status == "resolved"


def _persist_road_network_for_scenario(sqlite_session_factory) -> None:
    session = sqlite_session_factory()
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=NODE_STATION_A, latitude=32.70, longitude=35.001),
            GraphNode(id=NODE_TARGET_A, latitude=32.70, longitude=35.00),
            GraphNode(id=NODE_STATION_B, latitude=32.90, longitude=35.201),
            GraphNode(id=NODE_TARGET_B, latitude=32.90, longitude=35.20),
            GraphNode(id=NODE_STATION_C, latitude=32.80, longitude=35.10),
        ],
        edges=[
            GraphEdge(source_node_id=NODE_STATION_A, target_node_id=NODE_TARGET_A, distance_meters=100.0, travel_time_seconds=15.0),
            GraphEdge(source_node_id=NODE_STATION_B, target_node_id=NODE_TARGET_B, distance_meters=100.0, travel_time_seconds=15.0),
            GraphEdge(source_node_id=NODE_STATION_C, target_node_id=NODE_TARGET_A, distance_meters=3000.0, travel_time_seconds=240.0),
            GraphEdge(source_node_id=NODE_STATION_C, target_node_id=NODE_TARGET_B, distance_meters=3000.0, travel_time_seconds=240.0),
        ],
    )
    session.commit()
    session.close()
