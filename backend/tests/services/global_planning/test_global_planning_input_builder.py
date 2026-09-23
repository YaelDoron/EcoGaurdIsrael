"""Tests for GlobalPlanningInputBuilder (Stage 3 of the Global
Multi-Incident Optimizer refactor, Tasks 19-22, 28-30).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.services.global_planning.global_candidate_collector import GlobalCandidateCollector
from src.services.global_planning.global_incident_demand_builder import GlobalIncidentDemandBuilder
from src.services.global_planning.global_planning_input_builder import GlobalPlanningInputBuilder
from src.services.global_planning.global_route_matrix_builder import GlobalRouteMatrixBuilder
from src.services.operational.operational_context_service import OperationalContextService

AS_OF = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)

NODE_STATION_A = 1
NODE_STATION_B = 2
NODE_TARGET_A1 = 3
NODE_TARGET_B1 = 4


def _persist_fire_event(session, latitude, longitude) -> int:
    event = FireEventDB(
        latitude=latitude, longitude=longitude, detected_at=AS_OF, updated_at=AS_OF,
        status="confirmed", detection_confidence=0.9, methodology="m", methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    return event.id


def _persist_target_set(session, fire_event_id, latitude, longitude) -> int:
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id, generated_at=AS_OF, methodology="m", methodology_version="1.0"
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


class _CountingRoadNetworkRepository(RoadNetworkRepository):
    """Spies on get_network_in_bbox() calls so tests can assert whether the
    builder's road-network cache (Optimization 2) actually avoided a DB
    round trip, without depending on timing."""

    def __init__(self):
        super().__init__()
        self.get_network_in_bbox_calls = 0

    def get_network_in_bbox(self, db, min_lat, max_lat, min_lon, max_lon):
        self.get_network_in_bbox_calls += 1
        return super().get_network_in_bbox(db, min_lat, max_lat, min_lon, max_lon)


def _make_builder(sqlite_session_factory, road_network_repository=None) -> GlobalPlanningInputBuilder:
    from src.repositories.fire_station_repository import FireStationRepository
    from src.repositories.firefighting_resource_repository import FirefightingResourceRepository

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
        incident_demand_builder=GlobalIncidentDemandBuilder(
            fire_severity_assessment_repository=FireSeverityAssessmentRepository(sqlite_session_factory)
        ),
        route_matrix_builder=GlobalRouteMatrixBuilder(),
        road_network_repository=road_network_repository or RoadNetworkRepository(),
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
    return {
        "event_a": event_a, "event_b": event_b, "target_set_a": target_set_a, "target_set_b": target_set_b,
        "run_id": stored_run.id, "plan_a_id": plan_a_id,
    }


# ---------------------------------------------------------------------------
# Task 28 - integration scenario
# ---------------------------------------------------------------------------


def test_two_event_input_contains_exactly_the_run_members(two_event_scenario, sqlite_session_factory):
    builder = _make_builder(sqlite_session_factory)

    result = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert set(result.active_fire_event_ids) == {two_event_scenario["event_a"], two_event_scenario["event_b"]}


def test_targets_from_both_events_appear(two_event_scenario, sqlite_session_factory):
    builder = _make_builder(sqlite_session_factory)

    result = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    fire_event_ids_with_targets = {target.fire_event_id for target in result.targets}
    assert fire_event_ids_with_targets == {two_event_scenario["event_a"], two_event_scenario["event_b"]}
    assert len(result.targets) == 2


def test_resources_are_globally_deduplicated(two_event_scenario, sqlite_session_factory):
    builder = _make_builder(sqlite_session_factory)

    result = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    resource_ids = [resource.resource_id for resource in result.resources]
    assert sorted(resource_ids) == ["R1", "R2"]
    assert len(resource_ids) == len(set(resource_ids))


def test_commitment_ownership_appears(two_event_scenario, sqlite_session_factory):
    builder = _make_builder(sqlite_session_factory)

    result = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    r1 = next(resource for resource in result.resources if resource.resource_id == "R1")
    assert r1.current_commitment_fire_event_id == two_event_scenario["event_a"]
    assert r1.current_commitment_response_plan_id == two_event_scenario["plan_a_id"]
    r2 = next(resource for resource in result.resources if resource.resource_id == "R2")
    assert r2.current_commitment_fire_event_id is None


def test_cross_event_routes_exist(two_event_scenario, sqlite_session_factory):
    builder = _make_builder(sqlite_session_factory)

    result = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    target_b1 = next(target for target in result.targets if target.fire_event_id == two_event_scenario["event_b"])
    cross_route = result.route_matrix.get("R1", target_b1.response_target_id)
    assert cross_route is not None
    assert cross_route.fire_event_id == two_event_scenario["event_b"]


def test_no_db_writes_caused_by_input_building(two_event_scenario, sqlite_session_factory):
    from src.database.models.response_plan_db import ResponsePlanDB as PlanRow
    from sqlalchemy import select

    builder = _make_builder(sqlite_session_factory)
    session = sqlite_session_factory()
    plan_count_before = len(session.execute(select(PlanRow)).scalars().all())
    session.close()

    builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    session = sqlite_session_factory()
    plan_count_after = len(session.execute(select(PlanRow)).scalars().all())
    session.close()
    assert plan_count_after == plan_count_before


def test_matrix_size_instrumentation(two_event_scenario, sqlite_session_factory):
    """Task 30: resources, targets, potential pairs, feasible pairs are all
    obtainable from the finished GlobalPlanningInput for the integration
    scenario - GlobalRouteMatrixBuilder itself is unit-tested separately
    for the exact Dijkstra-call-count instrumentation."""
    builder = _make_builder(sqlite_session_factory)

    result = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    resource_count = len(result.resources)
    target_count = len(result.targets)
    potential_pairs = resource_count * target_count
    feasible_pairs = len(result.route_matrix)
    assert resource_count == 2
    assert target_count == 2
    assert potential_pairs == 4
    assert feasible_pairs == 4  # every resource has a real (if expensive) route to every target here


# ---------------------------------------------------------------------------
# Task 29 - single-incident compatibility
# ---------------------------------------------------------------------------


def test_single_incident_input_reduces_to_resources_times_that_events_targets(sqlite_session_factory):
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    target_set_a = _persist_target_set(session, event_a, 32.701, 35.001)
    _persist_station_and_resources(session, "STATION-A", 32.70, 35.00, ["R1", "R2"])
    session.commit()
    session.close()
    _persist_road_network(sqlite_session_factory)

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event_a,),
    )

    builder = _make_builder(sqlite_session_factory)
    result = builder.build(global_planning_run_id=stored_run.id, as_of=AS_OF)

    assert result.active_fire_event_ids == (event_a,)
    assert {t.fire_event_id for t in result.targets} == {event_a}
    assert {r.resource_id for r in result.resources} == {"R1", "R2"}
    # every (resource, target) pair for this one event is present and feasible
    (target,) = result.targets
    for resource_id in ("R1", "R2"):
        option = result.route_matrix.get(resource_id, target.response_target_id)
        assert option is not None
        assert option.eta_seconds == 30.0
        assert option.route_distance_meters == 200.0


# ---------------------------------------------------------------------------
# Task 5/6 - event with no usable target set
# ---------------------------------------------------------------------------


def test_active_event_with_no_target_set_contributes_zero_targets_honestly(sqlite_session_factory):
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    _persist_target_set(session, event_a, 32.701, 35.001)
    event_b = _persist_fire_event(session, 32.90, 35.20)  # no target set at all
    _persist_station_and_resources(session, "STATION-A", 32.70, 35.00, ["R1"])
    session.commit()
    session.close()
    _persist_road_network(sqlite_session_factory)

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event_a, event_b),
    )

    builder = _make_builder(sqlite_session_factory)
    result = builder.build(global_planning_run_id=stored_run.id, as_of=AS_OF)

    assert event_b in result.active_fire_event_ids
    assert event_b not in result.event_target_set_ids
    assert all(target.fire_event_id != event_b for target in result.targets)


# ---------------------------------------------------------------------------
# Task 20 - honors exact Stage 2 run membership
# ---------------------------------------------------------------------------


def test_builder_uses_run_membership_not_current_active_events(sqlite_session_factory):
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    _persist_target_set(session, event_a, 32.701, 35.001)
    _persist_station_and_resources(session, "STATION-A", 32.70, 35.00, ["R1"])
    session.commit()
    session.close()
    _persist_road_network(sqlite_session_factory)

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event_a,),
    )

    # A new active FireEvent appears AFTER the run was captured.
    session = sqlite_session_factory()
    event_c = _persist_fire_event(session, 33.0, 35.5)
    session.commit()
    session.close()

    builder = _make_builder(sqlite_session_factory)
    result = builder.build(global_planning_run_id=stored_run.id, as_of=AS_OF)

    assert result.active_fire_event_ids == (event_a,)
    assert event_c not in result.active_fire_event_ids


# ---------------------------------------------------------------------------
# Fingerprint determinism
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_across_repeated_builds(two_event_scenario, sqlite_session_factory):
    builder = _make_builder(sqlite_session_factory)

    first = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)
    second = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert first.input_fingerprint == second.input_fingerprint


# ---------------------------------------------------------------------------
# Stage 5, Task 6 - a severity/demand change must change the fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_a_severity_assessment_appears(two_event_scenario, sqlite_session_factory):
    from src.calculators.fire_severity.fire_severity_config import (
        FIRE_SEVERITY_METHODOLOGY_NAME,
        FIRE_SEVERITY_METHODOLOGY_VERSION,
    )
    from src.models import FireSeverityAssessment, FireSeverityAssessmentStatus, FireSeverityLevel, SatelliteHotspot
    from src.models.weather_observation import WeatherObservation
    from src.models.weather_station import WeatherStation
    from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
    from src.repositories.weather_repository import WeatherRepository

    builder = _make_builder(sqlite_session_factory)
    before = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    weather_repository = WeatherRepository(sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(sqlite_session_factory)
    station = WeatherStation(external_station_id=930000, name="Station", latitude=32.70, longitude=35.00)
    weather_repository.save_station(station)
    weather_repository.save_observation(
        WeatherObservation(
            station_external_id=station.external_station_id, timestamp=AS_OF - timedelta(minutes=5),
            temperature=30.0, relative_humidity=25.0, wind_speed=20.0,
        )
    )
    weather_id = next(
        record.observation_id
        for record in weather_repository.get_recent_observations_for_area_candidates(
            latitude=32.70, longitude=35.00, radius_km=5.0,
            start_time=AS_OF - timedelta(minutes=30), end_time=AS_OF,
        )
        if record.observation.station_external_id == station.external_station_id
    )
    satellite_repository.save_hotspot(
        SatelliteHotspot(latitude=32.70, longitude=35.00, detected_at=AS_OF - timedelta(minutes=20), confidence="h", frp=72.0, satellite="N20")
    )
    hotspot_id = satellite_repository.get_recent_hotspots(as_of=AS_OF, lookback_minutes=360)[0].id

    FireSeverityAssessmentRepository(sqlite_session_factory).save_assessment(
        FireSeverityAssessment(
            fire_event_id=two_event_scenario["event_a"],
            assessed_at=AS_OF,
            status=FireSeverityAssessmentStatus.VALID,
            score=95.0,
            level=FireSeverityLevel.CRITICAL,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        ),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )

    after = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert before.input_fingerprint != after.input_fingerprint
    before_demand = next(d for d in before.incident_demands if d.fire_event_id == two_event_scenario["event_a"])
    after_demand = next(d for d in after.incident_demands if d.fire_event_id == two_event_scenario["event_a"])
    assert before_demand.severity_level is None
    assert after_demand.severity_level is FireSeverityLevel.CRITICAL
    assert (before_demand.minimum_resources, before_demand.desired_resources) != (
        after_demand.minimum_resources,
        after_demand.desired_resources,
    )


# ---------------------------------------------------------------------------
# Optimization 3 (performance pass) - cheap pre-routing signature
# ---------------------------------------------------------------------------


def test_pre_routing_signature_is_stable_across_repeated_calls(two_event_scenario, sqlite_session_factory):
    builder = _make_builder(sqlite_session_factory)

    first = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)
    second = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert first.pre_routing_signature == second.pre_routing_signature


def test_pre_routing_signature_unaffected_by_build_and_still_stable(two_event_scenario, sqlite_session_factory):
    """The cheap signature must remain exactly as stable as the full
    fingerprint (see test_fingerprint_is_stable_across_repeated_builds)
    for the identical, unchanged scenario - including across a real
    build() call in between (build() must not itself mutate any state the
    signature depends on)."""
    builder = _make_builder(sqlite_session_factory)

    before = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)
    builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)
    after = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert before.pre_routing_signature == after.pre_routing_signature


def test_pre_routing_signature_changes_when_active_event_added(sqlite_session_factory):
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    _persist_target_set(session, event_a, 32.701, 35.001)
    _persist_station_and_resources(session, "STATION-A", 32.70, 35.00, ["R1"])
    session.commit()
    session.close()
    _persist_road_network(sqlite_session_factory)

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    run_before = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(event_a,),
    )
    builder = _make_builder(sqlite_session_factory)
    before = builder.compute_pre_routing_bundle(global_planning_run_id=run_before.id, as_of=AS_OF)

    session = sqlite_session_factory()
    event_b = _persist_fire_event(session, 32.90, 35.20)
    _persist_target_set(session, event_b, 32.901, 35.201)
    _persist_station_and_resources(session, "STATION-B", 32.90, 35.20, ["R2"])
    session.commit()
    session.close()

    run_after = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(event_a, event_b),
    )
    after = builder.compute_pre_routing_bundle(global_planning_run_id=run_after.id, as_of=AS_OF)

    assert before.pre_routing_signature != after.pre_routing_signature


def test_pre_routing_signature_changes_when_active_event_removed(two_event_scenario, sqlite_session_factory):
    builder = _make_builder(sqlite_session_factory)
    both = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    run_one_event = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(two_event_scenario["event_a"],),
    )
    one = builder.compute_pre_routing_bundle(global_planning_run_id=run_one_event.id, as_of=AS_OF)

    assert both.pre_routing_signature != one.pre_routing_signature


def test_pre_routing_signature_changes_when_target_priority_changes(two_event_scenario, sqlite_session_factory):
    builder = _make_builder(sqlite_session_factory)
    before = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    session = sqlite_session_factory()
    target_set = ResponseTargetSetDB(
        fire_event_id=two_event_scenario["event_a"], generated_at=AS_OF + timedelta(seconds=1),
        methodology="m", methodology_version="1.0",
    )
    session.add(target_set)
    session.flush()
    session.add(
        ResponseTargetDB(
            response_target_set_id=target_set.id, fire_event_id=two_event_scenario["event_a"], target_order=1,
            target_type="active_fire", latitude=32.701, longitude=35.001, priority_score=250.0,
        )
    )
    session.commit()
    session.close()

    after = builder.compute_pre_routing_bundle(
        global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF + timedelta(seconds=2)
    )

    assert before.pre_routing_signature != after.pre_routing_signature


def test_pre_routing_signature_changes_when_resource_availability_changes(two_event_scenario, sqlite_session_factory):
    from src.database.models.firefighting_resource_db import FirefightingResourceDB

    builder = _make_builder(sqlite_session_factory)
    before = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    session = sqlite_session_factory()
    resource = session.get(FirefightingResourceDB, "R2")
    resource.status = ResourceStatus.UNAVAILABLE
    session.commit()
    session.close()

    after = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert before.pre_routing_signature != after.pre_routing_signature


def test_pre_routing_signature_changes_when_commitment_changes(two_event_scenario, sqlite_session_factory):
    builder = _make_builder(sqlite_session_factory)
    before = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    _persist_commitment(
        sqlite_session_factory, two_event_scenario["event_b"], two_event_scenario["target_set_b"], "R2"
    )

    after = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert before.pre_routing_signature != after.pre_routing_signature


def test_pre_routing_signature_changes_when_demand_changes(two_event_scenario, sqlite_session_factory):
    """Same severity-appears scenario as test_fingerprint_changes_when_a_severity_assessment_appears -
    the cheap signature must agree with the full fingerprint that this is
    a real change (it must never omit incident_demands)."""
    from src.calculators.fire_severity.fire_severity_config import (
        FIRE_SEVERITY_METHODOLOGY_NAME,
        FIRE_SEVERITY_METHODOLOGY_VERSION,
    )
    from src.models import FireSeverityAssessment, FireSeverityAssessmentStatus, FireSeverityLevel, SatelliteHotspot
    from src.models.weather_observation import WeatherObservation
    from src.models.weather_station import WeatherStation
    from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
    from src.repositories.weather_repository import WeatherRepository

    builder = _make_builder(sqlite_session_factory)
    before = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    weather_repository = WeatherRepository(sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(sqlite_session_factory)
    station = WeatherStation(external_station_id=930001, name="Station", latitude=32.70, longitude=35.00)
    weather_repository.save_station(station)
    weather_repository.save_observation(
        WeatherObservation(
            station_external_id=station.external_station_id, timestamp=AS_OF - timedelta(minutes=5),
            temperature=30.0, relative_humidity=25.0, wind_speed=20.0,
        )
    )
    weather_id = next(
        record.observation_id
        for record in weather_repository.get_recent_observations_for_area_candidates(
            latitude=32.70, longitude=35.00, radius_km=5.0,
            start_time=AS_OF - timedelta(minutes=30), end_time=AS_OF,
        )
        if record.observation.station_external_id == station.external_station_id
    )
    satellite_repository.save_hotspot(
        SatelliteHotspot(latitude=32.70, longitude=35.00, detected_at=AS_OF - timedelta(minutes=20), confidence="h", frp=72.0, satellite="N20")
    )
    hotspot_id = satellite_repository.get_recent_hotspots(as_of=AS_OF, lookback_minutes=360)[0].id

    FireSeverityAssessmentRepository(sqlite_session_factory).save_assessment(
        FireSeverityAssessment(
            fire_event_id=two_event_scenario["event_a"],
            assessed_at=AS_OF,
            status=FireSeverityAssessmentStatus.VALID,
            score=95.0,
            level=FireSeverityLevel.CRITICAL,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        ),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )

    after = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert before.pre_routing_signature != after.pre_routing_signature


def test_build_with_precomputed_bundle_matches_build_without_it(two_event_scenario, sqlite_session_factory):
    """build(precomputed=...) must produce an input identical (same
    fingerprint) to build() computing everything itself - the precomputed
    path is a pure optimization, never a different code path semantically."""
    builder = _make_builder(sqlite_session_factory)

    direct = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    bundle = builder.compute_pre_routing_bundle(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)
    via_bundle = builder.build(
        global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF, precomputed=bundle
    )

    assert direct.input_fingerprint == via_bundle.input_fingerprint


# ---------------------------------------------------------------------------
# Optimization 2 (performance pass) - road-network subgraph reuse
# ---------------------------------------------------------------------------


def test_load_road_network_reuses_cached_subgraph_for_same_bbox(two_event_scenario, sqlite_session_factory):
    counting_repo = _CountingRoadNetworkRepository()
    builder = _make_builder(sqlite_session_factory, road_network_repository=counting_repo)

    builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)
    calls_after_first = counting_repo.get_network_in_bbox_calls
    builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)
    calls_after_second = counting_repo.get_network_in_bbox_calls

    assert calls_after_first >= 1
    assert calls_after_second == calls_after_first  # second build reused the cache - no new DB query


def test_load_road_network_reloads_when_bbox_changes(two_event_scenario, sqlite_session_factory):
    counting_repo = _CountingRoadNetworkRepository()
    builder = _make_builder(sqlite_session_factory, road_network_repository=counting_repo)
    two_event_result = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)
    calls_after_first = counting_repo.get_network_in_bbox_calls

    # A run over just event_a alone spans a smaller/different bbox.
    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    run_single = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(two_event_scenario["event_a"],),
    )
    single_event_result = builder.build(global_planning_run_id=run_single.id, as_of=AS_OF)

    assert counting_repo.get_network_in_bbox_calls > calls_after_first
    # The reload actually reflects the DB, not a stale/blindly-reused copy:
    # dropping event_b/STATION-B out of scope must be visible in the result.
    assert len(single_event_result.route_matrix) < len(two_event_result.route_matrix)


def test_road_network_cache_does_not_leak_across_builder_instances(two_event_scenario, sqlite_session_factory):
    counting_repo_1 = _CountingRoadNetworkRepository()
    counting_repo_2 = _CountingRoadNetworkRepository()
    builder_1 = _make_builder(sqlite_session_factory, road_network_repository=counting_repo_1)
    builder_2 = _make_builder(sqlite_session_factory, road_network_repository=counting_repo_2)

    builder_1.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)
    builder_2.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert counting_repo_1.get_network_in_bbox_calls >= 1
    # builder_2 must NOT have inherited builder_1's cache - a fresh builder
    # instance always starts with an empty cache (Optimization 2 is never
    # global/module-level state).
    assert counting_repo_2.get_network_in_bbox_calls >= 1


def test_load_road_network_returns_empty_without_touching_cache_when_no_anchors(sqlite_session_factory):
    counting_repo = _CountingRoadNetworkRepository()
    builder = _make_builder(sqlite_session_factory, road_network_repository=counting_repo)

    nodes, edges = builder._load_road_network((), ())

    assert nodes == []
    assert edges == []
    assert counting_repo.get_network_in_bbox_calls == 0
    assert builder._road_network_cache is None


def test_load_road_network_cache_hit_is_not_corrupted_by_mutating_returned_objects(
    two_event_scenario, sqlite_session_factory
):
    """Defends against a consumer (route-matrix building, node mapping)
    mutating the list/objects `_load_road_network` returns - that must
    never leak into the cached copy or a later cache hit (Optimization 2)."""
    builder = _make_builder(sqlite_session_factory)
    anchors = (
        (two_event_scenario["event_a"], 32.701, 35.001),
        (two_event_scenario["event_b"], 32.901, 35.201),
    )
    resources = ()

    nodes_1, edges_1 = builder._load_road_network(anchors, resources)
    original_node_count = len(nodes_1)
    original_edge_count = len(edges_1)
    assert original_node_count > 0 and original_edge_count > 0

    for node in nodes_1:
        node.latitude = -999.0
    for edge in edges_1:
        edge.distance_meters = -1.0

    nodes_2, edges_2 = builder._load_road_network(anchors, resources)  # cache hit (same bbox)

    assert len(nodes_2) == original_node_count
    assert len(edges_2) == original_edge_count
    assert all(node.latitude != -999.0 for node in nodes_2)
    assert all(edge.distance_meters != -1.0 for edge in edges_2)
