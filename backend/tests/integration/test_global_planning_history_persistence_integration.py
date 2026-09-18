"""Integration test for Stage 6's demand/shortage historical persistence
(GlobalPlanningRunRepository.record_member_result's `incident_demand`/
`event_optimization_result` parameters and
.record_global_optimization_metadata) - driven end-to-end through the real
GlobalPlanningRefreshCoordinator against a real (SQLite) database, matching
test_global_planning_refresh_coordinator_integration.py's own harness.

Scenario: Run #1 plans FireEvent A alone at HIGH severity; Run #2 (after A's
severity worsens to CRITICAL and FireEvent B appears at MODERATE) plans both
together. The test proves the DB history distinguishes the two generations'
per-event and global snapshots, and that a THIRD severity change for A
(recorded after Run #2) never mutates either run's already-persisted
history - the defining guarantee of this feature.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_service import (
    GlobalResponseOptimizationService,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.models import GraphEdge, GraphNode
from src.models.demand_source import DemandSource
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
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
from src.services.response_planning.current_response_plan_resolver import CurrentResponsePlanResolver

AS_OF_1 = datetime(2026, 9, 23, 9, 0, tzinfo=timezone.utc)
AS_OF_2 = AS_OF_1 + timedelta(minutes=40)
AS_OF_3 = AS_OF_1 + timedelta(minutes=80)


def _persist_fire_event(session, latitude, longitude) -> int:
    event = FireEventDB(
        latitude=latitude, longitude=longitude, detected_at=AS_OF_1, updated_at=AS_OF_1, status="confirmed",
        detection_confidence=0.9, methodology="m", methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    return event.id


def _persist_target_set(session, fire_event_id, latitude, longitude) -> int:
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id, generated_at=AS_OF_1, methodology="m", methodology_version="1.0"
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


def _persist_station_and_resources(session, station_id, latitude, longitude, resource_ids) -> None:
    session.add(FireStationDB(id=station_id, name=station_id, latitude=latitude, longitude=longitude))
    session.flush()
    for resource_id in resource_ids:
        session.add(FirefightingResourceDB(id=resource_id, station_id=station_id, status=ResourceStatus.AVAILABLE))


def _persist_road_network(sqlite_session_factory, nodes, edges) -> None:
    session = sqlite_session_factory()
    RoadNetworkRepository().save_network(session, nodes=nodes, edges=edges)
    session.commit()
    session.close()


_STATION_OFFSET_COUNTER = [0]


def _persist_severity(session_factory, fire_event_id, latitude, longitude, level, score, assessed_at) -> None:
    """Persist one VALID severity assessment, plus the minimal weather
    observation + satellite hotspot trace rows FireSeverityAssessmentRepository
    requires for VALID status (see test_fire_severity_assessment_repository.py's
    own persist_weather/persist_hotspot precedent)."""
    _STATION_OFFSET_COUNTER[0] += 1
    offset = _STATION_OFFSET_COUNTER[0]

    weather_repository = WeatherRepository(session_factory)
    station = WeatherStation(
        external_station_id=920000 + offset, name=f"Station {offset}", latitude=latitude, longitude=longitude
    )
    weather_repository.save_station(station)
    weather_repository.save_observation(
        WeatherObservation(
            station_external_id=station.external_station_id,
            timestamp=assessed_at - timedelta(minutes=5),
            temperature=30.0,
            relative_humidity=25.0,
            wind_speed=20.0,
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
            fire_event_id=fire_event_id,
            assessed_at=assessed_at,
            status=FireSeverityAssessmentStatus.VALID,
            score=score,
            level=level,
            methodology="m",
            methodology_version="1.0",
        ),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
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
        config=GlobalResponseOptimizationConfig(population_size=16, generation_count=12, random_seed=42),
    )


def test_two_global_replans_leave_independently_readable_and_immutable_history(sqlite_session_factory):
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    _persist_target_set(session, event_a, 32.701, 35.001)
    _persist_station_and_resources(session, "STATION-A", 32.70, 35.00, ["R1"])
    session.commit()
    session.close()
    _persist_road_network(
        sqlite_session_factory,
        nodes=[
            GraphNode(id=1, latitude=32.70, longitude=35.00),
            GraphNode(id=2, latitude=32.701, longitude=35.001),
            GraphNode(id=3, latitude=32.90, longitude=35.20),
            GraphNode(id=4, latitude=32.901, longitude=35.201),
        ],
        edges=[
            GraphEdge(source_node_id=1, target_node_id=2, distance_meters=200.0, travel_time_seconds=30.0),
            GraphEdge(source_node_id=1, target_node_id=4, distance_meters=20000.0, travel_time_seconds=1800.0),
            GraphEdge(source_node_id=3, target_node_id=2, distance_meters=20000.0, travel_time_seconds=1800.0),
            GraphEdge(source_node_id=3, target_node_id=4, distance_meters=200.0, travel_time_seconds=30.0),
        ],
    )
    _persist_severity(
        sqlite_session_factory, event_a, 32.70, 35.00, FireSeverityLevel.HIGH, 65.0, AS_OF_1 - timedelta(minutes=5)
    )

    coordinator = _make_coordinator(sqlite_session_factory)
    run_1 = coordinator.refresh(trigger="manual", as_of=AS_OF_1)
    assert run_1.status is GlobalPlanningRefreshStatus.ACTIVATED

    # --- Run #1 readback: A alone, HIGH (minimum=2, desired=3). ---
    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run_1 = run_repository.get_by_id(run_1.global_planning_run_id)
    assert stored_run_1.run.fitness_score is not None
    assert stored_run_1.run.random_seed == 42
    assert stored_run_1.run.ga_population_size == 16
    assert stored_run_1.run.shortage_total_required == 2
    assert stored_run_1.run.shortage_total_desired == 3

    (member_a_run_1,) = run_repository.get_members(run_1.global_planning_run_id)
    assert member_a_run_1.member.fire_event_id == event_a
    assert member_a_run_1.member.result_status is GlobalPlanningRunEventStatus.PLANNED
    assert member_a_run_1.member.severity_level is FireSeverityLevel.HIGH
    assert member_a_run_1.member.demand_source is DemandSource.SEVERITY_ASSESSMENT
    assert member_a_run_1.member.minimum_resources == 2
    assert member_a_run_1.member.desired_resources == 3
    assert member_a_run_1.member.assigned_resources == 1  # only R1 exists at this point
    assert member_a_run_1.member.unmet_required == 1

    # --- A worsens to CRITICAL; B appears at MODERATE. ---
    session = sqlite_session_factory()
    event_b = _persist_fire_event(session, 32.90, 35.20)
    _persist_target_set(session, event_b, 32.901, 35.201)
    _persist_station_and_resources(session, "STATION-B", 32.90, 35.20, ["R2"])
    session.commit()
    session.close()
    _persist_severity(
        sqlite_session_factory, event_a, 32.70, 35.00, FireSeverityLevel.CRITICAL, 90.0,
        AS_OF_1 + timedelta(minutes=10),
    )
    _persist_severity(
        sqlite_session_factory, event_b, 32.90, 35.20, FireSeverityLevel.MODERATE, 40.0,
        AS_OF_1 + timedelta(minutes=10),
    )

    run_2 = coordinator.refresh(trigger="manual", as_of=AS_OF_2)
    assert run_2.status is GlobalPlanningRefreshStatus.ACTIVATED

    # --- Run #2 readback: A CRITICAL (min=3, desired=4) + B MODERATE (min=1, desired=2). ---
    stored_run_2 = run_repository.get_by_id(run_2.global_planning_run_id)
    assert stored_run_2.run.shortage_total_required == 4  # 3 (A, CRITICAL) + 1 (B, MODERATE)
    assert stored_run_2.run.shortage_total_desired == 6  # 4 (A) + 2 (B)

    members_run_2 = {m.member.fire_event_id: m.member for m in run_repository.get_members(run_2.global_planning_run_id)}
    assert set(members_run_2) == {event_a, event_b}
    assert members_run_2[event_a].severity_level is FireSeverityLevel.CRITICAL
    assert members_run_2[event_a].minimum_resources == 3
    assert members_run_2[event_a].desired_resources == 4
    assert members_run_2[event_b].severity_level is FireSeverityLevel.MODERATE
    assert members_run_2[event_b].minimum_resources == 1
    assert members_run_2[event_b].desired_resources == 2

    # --- History distinguishes the two generations for the SAME FireEvent. ---
    assert member_a_run_1.member.severity_level != members_run_2[event_a].severity_level
    assert member_a_run_1.member.minimum_resources != members_run_2[event_a].minimum_resources
    assert stored_run_1.run.shortage_total_required != stored_run_2.run.shortage_total_required

    # --- A THIRD severity change for A, recorded AFTER Run #2, must never
    # mutate either run's already-persisted history (no reconstruction from
    # "current" severity). No third refresh is even needed to prove this -
    # the guarantee is that nothing about a persisted GlobalPlanningRun/
    # GlobalPlanningRunEvent row is ever rewritten by a later event. ---
    _persist_severity(sqlite_session_factory, event_a, 32.70, 35.00, FireSeverityLevel.LOW, 10.0, AS_OF_3)

    (member_a_run_1_reread,) = [
        m.member for m in run_repository.get_members(run_1.global_planning_run_id) if m.member.fire_event_id == event_a
    ]
    member_a_run_2_reread = {
        m.member.fire_event_id: m.member for m in run_repository.get_members(run_2.global_planning_run_id)
    }[event_a]

    assert member_a_run_1_reread.severity_level is FireSeverityLevel.HIGH
    assert member_a_run_1_reread.minimum_resources == 2
    assert member_a_run_2_reread.severity_level is FireSeverityLevel.CRITICAL
    assert member_a_run_2_reread.minimum_resources == 3
    stored_run_1_reread = run_repository.get_by_id(run_1.global_planning_run_id)
    stored_run_2_reread = run_repository.get_by_id(run_2.global_planning_run_id)
    assert stored_run_1_reread.run.shortage_total_required == 2
    assert stored_run_2_reread.run.shortage_total_required == 4
