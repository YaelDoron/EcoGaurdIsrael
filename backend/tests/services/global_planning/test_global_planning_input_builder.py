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
# Per-anchor road-network coverage fallback (infrastructure fix): the
# combined-bbox fetch used to only trigger a live OSM fetch when EVERY
# active event's combined bbox was completely empty, so one event with
# dense coverage (e.g. Carmel) silently masked another event in the same
# build whose own local area had NO road data at all - that event's targets
# then simply never got a feasible route, indistinguishable from "the GA
# assigned 0 resources" from the outside.
# ---------------------------------------------------------------------------

FAR_EVENT_STATION_NODE = 101
FAR_EVENT_TARGET_NODE = 102


class FakeRoadNetworkFetcher:
    """Records every bbox it's asked to fetch and returns configured
    nodes/edges only for bboxes containing `serves_point` - lets a test
    assert exactly which anchor triggered a dedicated fetch, without
    touching a real network."""

    def __init__(self, serves_point: tuple[float, float], nodes, edges):
        self._serves_point = serves_point
        self._nodes = nodes
        self._edges = edges
        self.calls: list[tuple[float, float, float, float]] = []

    def fetch_network_in_bbox(self, min_lat, max_lat, min_lon, max_lon, *, custom_filter=None):
        self.calls.append((min_lat, max_lat, min_lon, max_lon))
        lat, lon = self._serves_point
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return self._nodes, self._edges
        return [], []

    # These builder-level tests exercise per-anchor concurrency/dedup, not
    # tiling itself (see test_road_network_fetcher.py for that, against the
    # real RoadNetworkFetcher/_tile_bbox) - a thin passthrough keeps this
    # fake's existing single-bbox recording/matching behavior exactly as is.
    def fetch_network_in_bbox_tiled(self, min_lat, max_lat, min_lon, max_lon, **_kwargs):
        return self.fetch_network_in_bbox(min_lat, max_lat, min_lon, max_lon)


@pytest.fixture
def far_uncovered_event_scenario(sqlite_session_factory):
    """One event (A) near the seeded road network (as in two_event_scenario),
    plus a second event (C) ~190km away with NO seeded road-network nodes
    anywhere nearby - a genuinely uncovered region, unlike two_event_scenario's
    event_b which is ~29km from event_a and well inside both the combined
    bbox and the 15km per-anchor coverage check."""
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    event_c = _persist_fire_event(session, 31.00, 34.00)
    target_set_a = _persist_target_set(session, event_a, 32.701, 35.001)
    target_set_c = _persist_target_set(session, event_c, 31.001, 34.001)
    _persist_station_and_resources(session, "STATION-A", 32.70, 35.00, ["R1"])
    _persist_station_and_resources(session, "STATION-C", 31.01, 34.01, ["R3"])
    session.commit()
    session.close()
    _persist_road_network(sqlite_session_factory)  # only covers event_a's area

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event_a, event_c),
    )
    return {
        "event_a": event_a, "event_c": event_c,
        "target_set_a": target_set_a, "target_set_c": target_set_c,
        "run_id": stored_run.id,
    }


def _make_builder_with_fetcher(
    sqlite_session_factory, road_network_fetcher, incident_demand_builder=None
) -> GlobalPlanningInputBuilder:
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
        incident_demand_builder=incident_demand_builder
        or GlobalIncidentDemandBuilder(
            fire_severity_assessment_repository=FireSeverityAssessmentRepository(sqlite_session_factory)
        ),
        route_matrix_builder=GlobalRouteMatrixBuilder(),
        road_network_repository=RoadNetworkRepository(),
        road_network_fetcher=road_network_fetcher,
        session_factory=sqlite_session_factory,
    )


def test_uncovered_anchor_triggers_a_dedicated_osm_fetch_for_its_own_area(
    far_uncovered_event_scenario, sqlite_session_factory
):
    fetcher = FakeRoadNetworkFetcher(
        serves_point=(31.00, 34.00),
        nodes=[
            GraphNode(id=FAR_EVENT_STATION_NODE, latitude=31.01, longitude=34.01),
            GraphNode(id=FAR_EVENT_TARGET_NODE, latitude=31.001, longitude=34.001),
        ],
        edges=[
            GraphEdge(
                source_node_id=FAR_EVENT_STATION_NODE, target_node_id=FAR_EVENT_TARGET_NODE,
                distance_meters=500.0, travel_time_seconds=60.0,
            ),
        ],
    )
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher)

    result = builder.build(global_planning_run_id=far_uncovered_event_scenario["run_id"], as_of=AS_OF)

    # Only event_c's anchor was uncovered (event_a's own bbox already had
    # nodes), and progressive expansion stops at Step 1: the local fetch
    # alone finds R3 (its own station) locally-reachable, which already
    # meets desired_resources=1, so Step 3's wide fetch is never attempted.
    assert len(fetcher.calls) == 1
    target_c = next(t for t in result.targets if t.fire_event_id == far_uncovered_event_scenario["event_c"])
    route = result.route_matrix.get("R3", target_c.response_target_id)
    assert route is not None
    assert route.eta_seconds == pytest.approx(60.0)


def test_dedicated_fetch_result_is_persisted_for_the_next_build(
    far_uncovered_event_scenario, sqlite_session_factory
):
    """The fetched nodes/edges are saved via RoadNetworkRepository.save_network
    (same as the existing combined-bbox path), so a LATER build - even one
    using a fetcher that returns nothing - can still route event_c from the
    now-persisted data instead of needing another live fetch."""
    fetcher = FakeRoadNetworkFetcher(
        serves_point=(31.00, 34.00),
        nodes=[
            GraphNode(id=FAR_EVENT_STATION_NODE, latitude=31.01, longitude=34.01),
            GraphNode(id=FAR_EVENT_TARGET_NODE, latitude=31.001, longitude=34.001),
        ],
        edges=[
            GraphEdge(
                source_node_id=FAR_EVENT_STATION_NODE, target_node_id=FAR_EVENT_TARGET_NODE,
                distance_meters=500.0, travel_time_seconds=60.0,
            ),
        ],
    )
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher)
    builder.build(global_planning_run_id=far_uncovered_event_scenario["run_id"], as_of=AS_OF)

    empty_fetcher = FakeRoadNetworkFetcher(serves_point=(31.00, 34.00), nodes=[], edges=[])
    second_builder = _make_builder_with_fetcher(sqlite_session_factory, empty_fetcher)
    result = second_builder.build(global_planning_run_id=far_uncovered_event_scenario["run_id"], as_of=AS_OF)

    assert len(empty_fetcher.calls) == 0  # already covered from the DB now; no fetch needed at all
    target_c = next(t for t in result.targets if t.fire_event_id == far_uncovered_event_scenario["event_c"])
    assert result.route_matrix.get("R3", target_c.response_target_id) is not None


def test_a_still_uncovered_anchor_degrades_gracefully_without_crashing(
    far_uncovered_event_scenario, sqlite_session_factory
):
    """If OSM genuinely has nothing for the area (or the fetch times out /
    fails, per RoadNetworkFetcher's own always-degrade-to-empty contract),
    the build must still complete - that event's targets are simply left
    unroutable, exactly like today's combined-bbox-empty case, not a
    crash or an infinite retry. With progressive expansion, an empty local
    (Step 1) fetch finds 0 locally-reachable resources - short of R3's
    desired_resources=1 - so Step 3's wide fetch is also attempted (also
    empty), and R3's own station STILL lacks coverage afterward, so Step 4's
    station-micro fetch also runs (also empty) - 3 total calls."""
    fetcher = FakeRoadNetworkFetcher(serves_point=(31.00, 34.00), nodes=[], edges=[])
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher)

    result = builder.build(global_planning_run_id=far_uncovered_event_scenario["run_id"], as_of=AS_OF)

    assert len(fetcher.calls) == 3  # local (Step 1) + wide (Step 3) + station-micro (Step 4)
    target_c = next(t for t in result.targets if t.fire_event_id == far_uncovered_event_scenario["event_c"])
    assert result.route_matrix.get("R3", target_c.response_target_id) is None
    target_a = next(t for t in result.targets if t.fire_event_id == far_uncovered_event_scenario["event_a"])
    assert result.route_matrix.get("R1", target_a.response_target_id) is not None  # event_a is unaffected


def test_covered_anchor_never_triggers_a_dedicated_fetch(two_event_scenario, sqlite_session_factory):
    """Regression guard for the common case: when the existing combined-bbox
    fetch already covers every anchor (as in two_event_scenario), the new
    per-anchor pass must not call the fetcher at all."""
    fetcher = FakeRoadNetworkFetcher(serves_point=(0.0, 0.0), nodes=[], edges=[])
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher)

    builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert fetcher.calls == []


class RaisingRoadNetworkFetcher:
    """Simulates a dedicated per-anchor fetch that raises outright, rather
    than degrading to ([], []) the way the real RoadNetworkFetcher's own
    fail-safe contract does - e.g. a DB write against a session that went
    stale while this fetch was in flight. Used to prove one anchor's
    fetch/save failure can never crash the whole build."""

    def fetch_network_in_bbox(self, min_lat, max_lat, min_lon, max_lon, **_kwargs):
        raise TimeoutError("OSM fetch/conversion exceeded the bound for this bounding box.")

    def fetch_network_in_bbox_tiled(self, min_lat, max_lat, min_lon, max_lon, **_kwargs):
        return self.fetch_network_in_bbox(min_lat, max_lat, min_lon, max_lon)


def test_a_failing_dedicated_fetch_never_crashes_the_whole_build(
    far_uncovered_event_scenario, sqlite_session_factory
):
    """Regression guard for the observed production incident: a per-anchor
    fetch/save failure used to propagate all the way out of build(),
    leaving the whole GlobalPlanningRun uncompleted and abandoning every
    OTHER active event in the same cycle too. It must instead be logged and
    skipped, leaving only that one anchor's targets unroutable."""
    builder = _make_builder_with_fetcher(sqlite_session_factory, RaisingRoadNetworkFetcher())

    result = builder.build(global_planning_run_id=far_uncovered_event_scenario["run_id"], as_of=AS_OF)

    target_c = next(t for t in result.targets if t.fire_event_id == far_uncovered_event_scenario["event_c"])
    assert result.route_matrix.get("R3", target_c.response_target_id) is None
    target_a = next(t for t in result.targets if t.fire_event_id == far_uncovered_event_scenario["event_a"])
    assert result.route_matrix.get("R1", target_a.response_target_id) is not None  # event_a's own plan is unaffected


class RaisingRoadNetworkRepository:
    """Simulates the observed production incident one level further up: the
    very first combined-bbox lookup itself fails outright (e.g. the real
    PostgreSQL 65535-bind-parameter error hit on a large multi-event bbox),
    before the per-anchor fallback even gets a chance to run."""

    def get_network_in_bbox(self, session, min_lat, max_lat, min_lon, max_lon):
        raise Exception("number of parameters must be between 0 and 65535")

    def save_network(self, session, nodes, edges):
        raise AssertionError("save_network should never be reached in this scenario")


def test_a_totally_failed_road_network_load_still_includes_every_event(
    two_event_scenario, sqlite_session_factory
):
    """Regression guard for the broader outer safety net: even when road-
    network loading fails so completely that there is no fallback data to
    fall back to, build() must still return every active event's targets
    and resources (simply with an empty route matrix) rather than raising
    and aborting the whole GlobalPlanningInput - the Global Response Plan
    must never silently lose an event because of an infrastructure hiccup,
    however severe."""
    from src.repositories.fire_station_repository import FireStationRepository
    from src.repositories.firefighting_resource_repository import FirefightingResourceRepository

    fire_station_repository = FireStationRepository(sqlite_session_factory)
    firefighting_resource_repository = FirefightingResourceRepository(sqlite_session_factory)
    builder = GlobalPlanningInputBuilder(
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
        road_network_repository=RaisingRoadNetworkRepository(),
        session_factory=sqlite_session_factory,
    )

    result = builder.build(global_planning_run_id=two_event_scenario["run_id"], as_of=AS_OF)

    assert set(result.active_fire_event_ids) == {two_event_scenario["event_a"], two_event_scenario["event_b"]}
    assert len(result.targets) == 2
    assert len(result.route_matrix) == 0  # no road network data at all this cycle - honest, not fabricated


# ---------------------------------------------------------------------------
# Performance fix: multiple anchors needing a dedicated OSM fetch in the same
# build now fetch CONCURRENTLY (a thread pool) instead of strictly one at a
# time. These tests prove the fetches genuinely overlap in wall-clock time
# (not just that the end result happens to be correct), and that the
# coordinate-based pre-dedup still collapses two nearby uncovered anchors
# into one fetch the way the old sequential version did.
# ---------------------------------------------------------------------------

FAR_EVENT_D_STATION_NODE = 201
FAR_EVENT_D_TARGET_NODE = 202


class SlowRecordingRoadNetworkFetcher:
    """Sleeps for `delay_seconds` per call and records each call's own
    (start, end) wall-clock window - used to prove fetches for different
    anchors genuinely run concurrently (their windows overlap), not just
    that the final merged result happens to be correct either way."""

    def __init__(self, delay_seconds: float, *, nodes_by_point: dict[tuple[float, float], tuple[list, list]]):
        self._delay_seconds = delay_seconds
        self._nodes_by_point = nodes_by_point
        self.call_windows: list[tuple[float, float]] = []

    def fetch_network_in_bbox(self, min_lat, max_lat, min_lon, max_lon, *, custom_filter=None):
        import time

        start = time.perf_counter()
        time.sleep(self._delay_seconds)
        for (lat, lon), result in self._nodes_by_point.items():
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                self.call_windows.append((start, time.perf_counter()))
                return result
        self.call_windows.append((start, time.perf_counter()))
        return [], []

    def fetch_network_in_bbox_tiled(self, min_lat, max_lat, min_lon, max_lon, **_kwargs):
        return self.fetch_network_in_bbox(min_lat, max_lat, min_lon, max_lon)


@pytest.fixture
def two_far_uncovered_events_scenario(sqlite_session_factory):
    """Two events (C, D), each ~190km+ from the seeded network AND from
    each other - both genuinely uncovered, and far enough apart that the
    coordinate-based pre-dedup must NOT collapse them into one fetch (unlike
    two anchors that happen to be close together).

    Also seeds one edgeless "filler" node inside the COMBINED bbox (roughly
    midway between C and D, but >5km from each) purely so
    get_network_in_bbox's initial combined-bbox read is non-empty - without
    it, that read returns zero rows and _load_road_network_inner's OWN
    (pre-existing, non-per-anchor) "if not road_nodes: fetch" fallback
    fires first and can satisfy whichever anchor happens to fall inside
    that wide combined bbox by coincidence, which would silently turn this
    into a single-fetch test instead of the two-concurrent-fetches test it
    needs to be. far_uncovered_event_scenario avoids this same trap only
    because its one far event's combined bbox happens to reach the default
    seeded network; with two mutually far, unrelated events, nothing does
    unless seeded explicitly.
    """
    session = sqlite_session_factory()
    event_c = _persist_fire_event(session, 31.00, 34.00)
    event_d = _persist_fire_event(session, 29.50, 33.00)  # >150km from event_c
    target_set_c = _persist_target_set(session, event_c, 31.001, 34.001)
    target_set_d = _persist_target_set(session, event_d, 29.501, 33.001)
    _persist_station_and_resources(session, "STATION-C", 31.01, 34.01, ["R3"])
    _persist_station_and_resources(session, "STATION-D", 29.51, 33.01, ["R4"])
    session.commit()
    session.close()
    filler_session = sqlite_session_factory()
    RoadNetworkRepository().save_network(
        filler_session,
        nodes=[GraphNode(id=999, latitude=30.25, longitude=33.50)],  # >5km from both C and D
        edges=[],
    )
    filler_session.commit()
    filler_session.close()

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event_c, event_d),
    )
    return {
        "event_c": event_c, "event_d": event_d,
        "target_set_c": target_set_c, "target_set_d": target_set_d,
        "run_id": stored_run.id,
    }


def test_two_distant_uncovered_anchors_fetch_concurrently_not_sequentially(
    two_far_uncovered_events_scenario, sqlite_session_factory
):
    """The central performance claim, proven directly rather than inferred
    from total wall-clock time alone: two independent OSM fetches' own
    (start, end) windows must overlap, not run back-to-back."""
    fetcher = SlowRecordingRoadNetworkFetcher(
        delay_seconds=0.3,
        nodes_by_point={
            (31.00, 34.00): (
                [GraphNode(id=FAR_EVENT_STATION_NODE, latitude=31.01, longitude=34.01),
                 GraphNode(id=FAR_EVENT_TARGET_NODE, latitude=31.001, longitude=34.001)],
                [GraphEdge(source_node_id=FAR_EVENT_STATION_NODE, target_node_id=FAR_EVENT_TARGET_NODE,
                           distance_meters=500.0, travel_time_seconds=60.0)],
            ),
            (29.50, 33.00): (
                [GraphNode(id=FAR_EVENT_D_STATION_NODE, latitude=29.51, longitude=33.01),
                 GraphNode(id=FAR_EVENT_D_TARGET_NODE, latitude=29.501, longitude=33.001)],
                [GraphEdge(source_node_id=FAR_EVENT_D_STATION_NODE, target_node_id=FAR_EVENT_D_TARGET_NODE,
                           distance_meters=500.0, travel_time_seconds=60.0)],
            ),
        },
    )
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher)

    import time

    wall_clock_start = time.perf_counter()
    result = builder.build(global_planning_run_id=two_far_uncovered_events_scenario["run_id"], as_of=AS_OF)
    wall_clock_elapsed = time.perf_counter() - wall_clock_start

    assert len(fetcher.call_windows) == 2
    (start_1, end_1), (start_2, end_2) = fetcher.call_windows
    # Overlap check: each call must have started before the OTHER finished -
    # impossible if they ran strictly one after another.
    assert start_1 < end_2 and start_2 < end_1, (
        f"expected overlapping fetch windows, got {fetcher.call_windows} - looks sequential, not concurrent"
    )
    # The overlap check above is the real proof; this is only a loose sanity
    # check against a full regression to strictly-sequential fetching -
    # build() does real non-fetch work too (candidate collection, target
    # loading, route matrix construction), so this bound is generous rather
    # than tight against the fetches' own 0.3s duration.
    assert wall_clock_elapsed < 3.0, f"build() took {wall_clock_elapsed:.2f}s - unexpectedly slow"

    target_c = next(t for t in result.targets if t.fire_event_id == two_far_uncovered_events_scenario["event_c"])
    target_d = next(t for t in result.targets if t.fire_event_id == two_far_uncovered_events_scenario["event_d"])
    assert result.route_matrix.get("R3", target_c.response_target_id) is not None
    assert result.route_matrix.get("R4", target_d.response_target_id) is not None


def test_two_nearby_uncovered_anchors_still_dedup_to_one_fetch(sqlite_session_factory):
    """Regression guard for the coordinate-based pre-dedup: two anchors
    within _PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM of EACH OTHER (not just of
    existing coverage) must still only trigger one fetch between them, the
    same guarantee the old sequential "check the growing node pool" version
    provided - now decided up front by comparing anchor coordinates
    directly, before any fetch runs."""
    session = sqlite_session_factory()
    event_c = _persist_fire_event(session, 31.00, 34.00)
    event_e = _persist_fire_event(session, 31.001, 34.001)  # ~150m from event_c - well within snap distance
    _persist_target_set(session, event_c, 31.001, 34.001)
    _persist_target_set(session, event_e, 31.002, 34.002)
    _persist_station_and_resources(session, "STATION-C", 31.01, 34.01, ["R3"])
    session.commit()
    session.close()
    _persist_road_network(sqlite_session_factory)  # covers neither event

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event_c, event_e),
    )

    fetcher = FakeRoadNetworkFetcher(
        serves_point=(31.00, 34.00),
        nodes=[GraphNode(id=FAR_EVENT_STATION_NODE, latitude=31.01, longitude=34.01),
               GraphNode(id=FAR_EVENT_TARGET_NODE, latitude=31.001, longitude=34.001)],
        edges=[GraphEdge(source_node_id=FAR_EVENT_STATION_NODE, target_node_id=FAR_EVENT_TARGET_NODE,
                          distance_meters=500.0, travel_time_seconds=60.0)],
    )
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher)

    builder.build(global_planning_run_id=stored_run.id, as_of=AS_OF)

    assert len(fetcher.calls) == 1


# ---------------------------------------------------------------------------
# Architecture guard: the Global Multi-Incident Optimizer path must never
# fabricate a route from air (straight-line) distance - a target with no
# real Dijkstra path over the actual road graph must stay uncovered, not
# get a fake ETA. HaversineFallbackCalculator exists (used only by the
# single-event RoutePlanningAgent) but must never reach this module or
# GlobalRouteMatrixBuilder. `haversine_distance_km` itself is a different,
# legitimate use - plain distance MEASUREMENT (snap-distance validation,
# per-anchor coverage checks), not a fabricated route/ETA - so only the
# fallback-route class/module is checked here, not that utility function.
# ---------------------------------------------------------------------------


def test_input_builder_never_imports_the_haversine_route_fallback():
    import ast
    from pathlib import Path

    forbidden_fragments = ("HaversineFallbackCalculator", "haversine_fallback_calculator")
    for relative_path in (
        "src/services/global_planning/global_planning_input_builder.py",
        "src/services/global_planning/global_route_matrix_builder.py",
    ):
        path = Path(relative_path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        violations = []
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            if any(fragment in module for fragment in forbidden_fragments):
                violations.append(module)
        assert violations == [], f"{relative_path} imports the Haversine route fallback: {violations}"


# ---------------------------------------------------------------------------
# Progressive expansion (lazy loading): a small local fetch first, expanding
# to the expensive wide/tiled fetch only when local candidate supply falls
# short of the fire's own desired_resources - not for every uncovered
# anchor unconditionally.
# ---------------------------------------------------------------------------


class FakeIncidentDemandBuilder:
    """Returns a fixed, caller-supplied demand per fire_event_id, bypassing
    real severity-assessment computation (which needs a full weather +
    satellite evidence chain just to produce a level) - these tests only
    care about the resulting desired_resources number."""

    def __init__(self, demand_by_event: dict):
        self._demand_by_event = demand_by_event

    def build(self, fire_event_ids, as_of):
        from tests.calculators.global_response_optimization.helpers import make_demand

        return tuple(
            make_demand(fire_event_id, desired_resources=self._demand_by_event[fire_event_id])
            for fire_event_id in fire_event_ids
            if fire_event_id in self._demand_by_event
        )


def test_local_fetch_alone_satisfies_desired_resources_skips_the_wide_fetch(
    far_uncovered_event_scenario, sqlite_session_factory
):
    """desired_resources=1 and R3 (event_c's own station) is well within
    _LOCAL_FETCH_RADIUS_KM - the local (Step 1) fetch alone must satisfy it,
    so only ONE call happens and its bbox must be the SMALL local one, not
    the wide one."""
    from src.services.global_planning.global_planning_input_builder import _LOCAL_FETCH_RADIUS_KM, _KM_PER_DEGREE

    fetcher = FakeRoadNetworkFetcher(
        serves_point=(31.00, 34.00),
        nodes=[
            GraphNode(id=FAR_EVENT_STATION_NODE, latitude=31.01, longitude=34.01),
            GraphNode(id=FAR_EVENT_TARGET_NODE, latitude=31.001, longitude=34.001),
        ],
        edges=[
            GraphEdge(source_node_id=FAR_EVENT_STATION_NODE, target_node_id=FAR_EVENT_TARGET_NODE,
                      distance_meters=500.0, travel_time_seconds=60.0),
        ],
    )
    demand_builder = FakeIncidentDemandBuilder(
        {far_uncovered_event_scenario["event_a"]: 1, far_uncovered_event_scenario["event_c"]: 1}
    )
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher, demand_builder)

    builder.build(global_planning_run_id=far_uncovered_event_scenario["run_id"], as_of=AS_OF)

    assert len(fetcher.calls) == 1
    min_lat, max_lat, min_lon, max_lon = fetcher.calls[0]
    local_degrees = _LOCAL_FETCH_RADIUS_KM / _KM_PER_DEGREE
    assert (max_lat - min_lat) == pytest.approx(2 * local_degrees, rel=1e-6)
    assert (max_lon - min_lon) == pytest.approx(2 * local_degrees, rel=1e-6)


class FilterRecordingRoadNetworkFetcher:
    """Records the exact `custom_filter` value passed to every call, so a
    test can assert local and wide fetches use the SAME filter (parity),
    not just that each individually completes."""

    def __init__(self, serves_point, nodes, edges):
        self._serves_point = serves_point
        self._nodes = nodes
        self._edges = edges
        self.filters_seen: list[object] = []

    def fetch_network_in_bbox(self, min_lat, max_lat, min_lon, max_lon, *, custom_filter=None):
        self.filters_seen.append(custom_filter)
        lat, lon = self._serves_point
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return self._nodes, self._edges
        return [], []

    def fetch_network_in_bbox_tiled(self, min_lat, max_lat, min_lon, max_lon, *, custom_filter=None, **_kwargs):
        return self.fetch_network_in_bbox(min_lat, max_lat, min_lon, max_lon, custom_filter=custom_filter)


def test_local_wide_and_station_micro_fetches_use_the_identical_graph_fidelity_filter(
    far_uncovered_event_scenario, sqlite_session_factory
):
    """Parity guard: local (Step 1), wide (Step 3), and station-micro
    (Step 4) fetches must all request the SAME road-class fidelity - they
    are only ever meant to differ in the AREA covered, never in which road
    types are visible once fetched."""
    from src.services.operational.road_network_fetcher import GRAPH_FIDELITY_CUSTOM_FILTER

    fetcher = FilterRecordingRoadNetworkFetcher(serves_point=(31.00, 34.00), nodes=[], edges=[])
    demand_builder = FakeIncidentDemandBuilder(
        {far_uncovered_event_scenario["event_a"]: 1, far_uncovered_event_scenario["event_c"]: 2}
    )
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher, demand_builder)

    builder.build(global_planning_run_id=far_uncovered_event_scenario["run_id"], as_of=AS_OF)

    # Empty results force escalation through Step 3 and Step 4, so all three phases run.
    assert len(fetcher.filters_seen) == 3
    assert all(seen == GRAPH_FIDELITY_CUSTOM_FILTER for seen in fetcher.filters_seen)


def test_local_shortage_escalates_to_the_wide_tiled_fetch(far_uncovered_event_scenario, sqlite_session_factory):
    """desired_resources=2 but only R3 (1 resource) is locally reachable -
    a genuine shortage - so Step 3's wide fetch must also run, and its bbox
    must be the LARGE (_PER_ANCHOR_FETCH_RADIUS_KM) one, not the local one."""
    from src.services.global_planning.global_planning_input_builder import (
        _LOCAL_FETCH_RADIUS_KM,
        _PER_ANCHOR_FETCH_RADIUS_KM,
        _KM_PER_DEGREE,
    )

    fetcher = FakeRoadNetworkFetcher(
        serves_point=(31.00, 34.00),
        nodes=[
            GraphNode(id=FAR_EVENT_STATION_NODE, latitude=31.01, longitude=34.01),
            GraphNode(id=FAR_EVENT_TARGET_NODE, latitude=31.001, longitude=34.001),
        ],
        edges=[
            GraphEdge(source_node_id=FAR_EVENT_STATION_NODE, target_node_id=FAR_EVENT_TARGET_NODE,
                      distance_meters=500.0, travel_time_seconds=60.0),
        ],
    )
    demand_builder = FakeIncidentDemandBuilder(
        {far_uncovered_event_scenario["event_a"]: 1, far_uncovered_event_scenario["event_c"]: 2}
    )
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher, demand_builder)

    builder.build(global_planning_run_id=far_uncovered_event_scenario["run_id"], as_of=AS_OF)

    assert len(fetcher.calls) == 2
    bbox_spans = sorted((max_lat - min_lat) for min_lat, max_lat, _min_lon, _max_lon in fetcher.calls)
    local_span = 2 * (_LOCAL_FETCH_RADIUS_KM / _KM_PER_DEGREE)
    wide_span = 2 * (_PER_ANCHOR_FETCH_RADIUS_KM / _KM_PER_DEGREE)
    assert bbox_spans[0] == pytest.approx(local_span, rel=1e-6)
    assert bbox_spans[1] == pytest.approx(wide_span, rel=1e-6)


class TestLocallyReachableAssignableCount:
    """Direct unit tests for the sufficiency-check helper itself, isolated
    from any fetch/threading concern."""

    def _resource(self, resource_id, lat, lon, status=ResourceStatus.AVAILABLE):
        from src.models.global_planning_resource import GlobalPlanningResource

        return GlobalPlanningResource(
            resource_id=resource_id, station_id=f"station-{resource_id}", station_name=f"Station {resource_id}",
            station_latitude=lat, station_longitude=lon, operational_status=status,
        )

    def test_counts_a_resource_within_radius_and_with_nearby_coverage(self):
        resource = self._resource("R1", 31.005, 34.005)
        nodes = [GraphNode(id=1, latitude=31.005, longitude=34.005)]

        count = GlobalPlanningInputBuilder._locally_reachable_assignable_count(31.00, 34.00, (resource,), nodes)

        assert count == 1

    def test_excludes_a_resource_beyond_the_local_radius(self):
        from src.services.global_planning.global_planning_input_builder import _LOCAL_FETCH_RADIUS_KM

        far_lat = 31.00 + (_LOCAL_FETCH_RADIUS_KM * 2) / 111.32  # well beyond the local radius
        resource = self._resource("R1", far_lat, 34.00)
        nodes = [GraphNode(id=1, latitude=far_lat, longitude=34.00)]

        count = GlobalPlanningInputBuilder._locally_reachable_assignable_count(31.00, 34.00, (resource,), nodes)

        assert count == 0

    def test_excludes_an_unavailable_resource_even_if_locally_covered(self):
        resource = self._resource("R1", 31.005, 34.005, status=ResourceStatus.UNAVAILABLE)
        nodes = [GraphNode(id=1, latitude=31.005, longitude=34.005)]

        count = GlobalPlanningInputBuilder._locally_reachable_assignable_count(31.00, 34.00, (resource,), nodes)

        assert count == 0

    def test_excludes_a_resource_with_no_nearby_fetched_node(self):
        """Within radius geographically, but the local fetch didn't actually
        return a node near ITS station - not really reachable yet."""
        resource = self._resource("R1", 31.005, 34.005)

        count = GlobalPlanningInputBuilder._locally_reachable_assignable_count(31.00, 34.00, (resource,), [])

        assert count == 0


# ---------------------------------------------------------------------------
# Step 4 (targeted origin fetching): a candidate resource's OWN station gets
# a guaranteed micro-fetch if it lacks coverage, INDEPENDENT of whether the
# anchor itself was already deemed "sufficient" and Step 3 was skipped -
# the exact "station sits at the edge of an otherwise-covered area" case
# reported live.
# ---------------------------------------------------------------------------


def test_station_micro_fetch_fires_even_when_the_anchor_itself_skipped_the_wide_fetch(sqlite_session_factory):
    """Two resources near one anchor, BOTH within GlobalCandidateCollector's
    own first (5km) search tier so both genuinely become candidates:
    R-COVERED's station is where Step 1's local fetch returns its one node
    (satisfies desired_resources=1 alone, so Step 3 is skipped for the
    anchor), but R-EDGE's station - ~8km from that node, on the opposite
    side of the anchor, so still within 5km of the anchor itself but
    outside the 5km station-coverage check - never gets covered by Step 1.
    Step 4 must still fetch R-EDGE's own station micro-radius, even though
    the anchor-level decision above it never escalated to Step 3."""
    session = sqlite_session_factory()
    event = _persist_fire_event(session, 31.00, 34.00)
    _persist_target_set(session, event, 31.001, 34.001)
    # ~4km north and ~4km south of the anchor - both within the collector's
    # own 5km first search tier, but ~8km from EACH OTHER (beyond the 5km
    # node-coverage check).
    _persist_station_and_resources(session, "STATION-COVERED", 31.0359, 34.00, ["R-COVERED"])
    _persist_station_and_resources(session, "STATION-EDGE", 30.9641, 34.00, ["R-EDGE"])
    session.commit()
    session.close()

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event,),
    )

    # The fetcher's local (Step 1) call only ever returns a node right at
    # STATION-COVERED's own coordinate - never anything near STATION-EDGE -
    # so STATION-EDGE remains genuinely uncovered after Step 1, regardless
    # of bbox size, until its own dedicated Step 4 fetch runs.
    fetcher = FilterRecordingRoadNetworkFetcher(
        serves_point=(31.0359, 34.00),
        nodes=[GraphNode(id=301, latitude=31.0359, longitude=34.00)],
        edges=[],
    )
    demand_builder = FakeIncidentDemandBuilder({event: 1})
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher, demand_builder)

    builder.build(global_planning_run_id=stored_run.id, as_of=AS_OF)

    # 2 calls: Step 1's local fetch (finds R-COVERED, satisfies desired_resources=1,
    # so Step 3 never runs) + Step 4's dedicated micro-fetch for STATION-EDGE.
    assert len(fetcher.filters_seen) == 2


class WidthDispatchingRoadNetworkFetcher:
    """Returns different canned (nodes, edges) depending on the QUERIED
    bbox's WIDTH (i.e. which radius produced it), not merely whether a
    point falls inside it - lets a test give Step 1's local (10km) fetch
    and Step 4's station-micro (2km) fetch genuinely DIFFERENT results, the
    same way real Overpass coverage can differ by area size (a wide fetch
    finding the fire's own roads while a specific station's short access
    road is still missing until its own dedicated micro-fetch runs)."""

    def __init__(self, results_by_radius_km: dict[float, tuple[list, list]]):
        self._results_by_radius_km = results_by_radius_km
        self.calls: list[tuple[float, float, float, float]] = []

    def fetch_network_in_bbox(self, min_lat, max_lat, min_lon, max_lon, *, custom_filter=None):
        from src.services.global_planning.global_planning_input_builder import _KM_PER_DEGREE

        self.calls.append((min_lat, max_lat, min_lon, max_lon))
        width_km = (max_lat - min_lat) / 2 * _KM_PER_DEGREE
        for radius_km, (nodes, edges) in self._results_by_radius_km.items():
            if width_km == pytest.approx(radius_km, rel=0.05):
                return nodes, edges
        return [], []

    def fetch_network_in_bbox_tiled(self, min_lat, max_lat, min_lon, max_lon, **_kwargs):
        return self.fetch_network_in_bbox(min_lat, max_lat, min_lon, max_lon)


def test_station_micro_fetch_makes_dijkstra_snap_to_the_genuinely_near_station_node(sqlite_session_factory):
    """Integration-level reproduction of the live Gush Etzion symptom
    ("route starts 1-2km from the station's blue dot"): Steps 1-3 leave
    only a node ~1.2km from the station's real coordinate - well within the
    OLD 5km station-coverage threshold (_PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM),
    so the pre-fix code wrongly considered the station already "covered"
    and Step 4 never ran, leaving NodeMappingService/Dijkstra no choice but
    to snap to that 1.2km-distant node. With the tightened
    _STATION_COVERAGE_CHECK_RADIUS_KM, Step 4 must fire, supply a node
    essentially AT the station via its own dedicated micro-fetch, and the
    resulting route (built through the FULL build() -> route_matrix
    pipeline, not a unit test of _stations_needing_fetch in isolation) must
    actually START from that near node - not the distant one."""
    from src.services.global_planning.global_planning_input_builder import (
        _LOCAL_FETCH_RADIUS_KM,
        _STATION_MICRO_FETCH_RADIUS_KM,
    )

    session = sqlite_session_factory()
    event = _persist_fire_event(session, 31.00, 34.00)
    _persist_target_set(session, event, 31.001, 34.001)
    # ~1.2km east of the anchor: inside the collector's first (5km) search
    # tier (so it's genuinely discovered as a candidate) and inside the OLD
    # 5km station-coverage check against the anchor node below, but outside
    # the new tight one.
    _persist_station_and_resources(session, "STATION-NEAR-EDGE", 31.00, 34.013, ["R1"])
    session.commit()
    session.close()

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event,),
    )

    node_anchor_distant = 501
    node_target = 502
    node_station_local = 503

    fetcher = WidthDispatchingRoadNetworkFetcher(
        {
            _LOCAL_FETCH_RADIUS_KM: (
                [
                    GraphNode(id=node_anchor_distant, latitude=31.00, longitude=34.00),
                    GraphNode(id=node_target, latitude=31.001, longitude=34.001),
                ],
                [
                    GraphEdge(
                        source_node_id=node_anchor_distant, target_node_id=node_target,
                        distance_meters=1500.0, travel_time_seconds=180.0,
                    ),
                ],
            ),
            _STATION_MICRO_FETCH_RADIUS_KM: (
                [GraphNode(id=node_station_local, latitude=31.00, longitude=34.013)],
                [
                    GraphEdge(
                        source_node_id=node_station_local, target_node_id=node_target,
                        distance_meters=300.0, travel_time_seconds=40.0,
                    ),
                ],
            ),
        }
    )
    demand_builder = FakeIncidentDemandBuilder({event: 1})
    builder = _make_builder_with_fetcher(sqlite_session_factory, fetcher, demand_builder)

    result = builder.build(global_planning_run_id=stored_run.id, as_of=AS_OF)

    (target,) = result.targets
    option = result.route_matrix.get("R1", target.response_target_id)
    assert option is not None, "R1 should have a feasible route once Step 4 supplies its own local node."
    assert option.node_path[0] == node_station_local, (
        "Dijkstra must start R1's route from its own station-local node (Step 4's "
        "micro-fetch), not the ~1.2km-distant anchor node - snapping to the distant "
        "node is exactly the reported 'route starts far from the station's blue "
        f"dot' bug. Got node_path={option.node_path!r}."
    )
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
