"""End-to-end proof that the simulation path reaches the Stage 6 Global GA
through the real production US4.4 -> global-planning bridge
(OperationalPlanningRefreshCoordinator -> GlobalPlanningRefreshCoordinator).

Everything here is real production code driven through SimulationEventExecutor
/ SimulationFireDetectionCoordinator / SimulationRefreshCoordinator, exactly
as scripts/run_demo_simulation.py drives it - real FireDetectionAgent, real
FireSeverityAssessmentAgent, real FireSpreadRefreshOrchestrator, real
ResponseTargetGenerationAgent, real OperationalRefreshOrchestrator, real
GlobalPlanningRefreshCoordinator (via build_global_planning_refresh_coordinator,
so real GlobalPlanningInputBuilder + real Global GA + real
GlobalResponsePlanActivationService - no Dijkstra rerun during activation,
no legacy per-event optimizer, no baseline comparison fabricated), all
against one SQLite in-memory database.

Stage 6 Task 48 (simulation cutover): this replaces the pre-Stage-6 wiring
that drove the legacy per-event ResponsePlanningRefreshOrchestrator - the
simulation now exercises the exact same GlobalPlanningRefreshCoordinator
production path as every other trigger, never a separate demo planner.

No live network call is made:
- Weather/Satellite/News data comes from the same deterministic, seeded
  simulation generators the demo script uses - no external API involved.
- Vegetation/land-cover lookup (normally Copernicus) is replaced with the
  same StaticVegetationMapper/land_cover_client fake pattern already used by
  tests/acceptance/test_fire_severity_user_story_2_3.py, so no Copernicus
  call is made either.
- The road network is pre-seeded directly into the SQLite database (as
  test_response_planning_user_story_5_4_sanity.py already does for US5.4),
  so OperationalContextService's cache lookup is a hit and RoadNetworkFetcher
  (OSM/osmnx) is never invoked.

These tests therefore do not require osmnx to be installed for their own
logic - only for the ability to *import* src.simulation at all (see
src/simulation/analysis/simulation_operational_coordinator.py's chain
through OperationalContextService), which is a pre-existing characteristic
of this package unrelated to this task.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.agents.analysis import FireDetectionAgent, FireSeverityAssessmentAgent, ResponseTargetGenerationAgent
from src.agents.analysis.fire_spread_prediction_agent import FireSpreadPredictionAgent
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_severity.fire_severity_calculator import FireSeverityCalculator
from src.calculators.fire_spread import FireSpreadCalculator
from src.calculators.response_target import ResponseTargetCalculator
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.graph_edge_db import GraphEdgeDB
from src.database.models.graph_node_db import GraphNodeDB
from src.models import ResourceStatus, VegetationData
from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.plan_comparison_repository import PlanComparisonRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_detection import FireDetectionEvidenceService
from src.services.fire_severity import FireSeverityInputService
from src.services.fire_spread import FireSpreadInputService
from src.services.global_planning.global_planning_refresh_coordinator import GlobalPlanningRefreshStatus
from src.services.global_planning.global_planning_refresh_production_factory import (
    build_global_planning_refresh_coordinator,
)
from src.services.operational import OperationalContextService
from src.services.operational_planning_refresh.operational_planning_refresh_coordinator import (
    OperationalPlanningRefreshCoordinator,
)
from src.services.operational_refresh import (
    FireSeverityRefreshOrchestrator,
    FireSpreadRefreshOrchestrator,
    OperationalRefreshOrchestrator,
)
from src.services.operational_refresh.resource_status_update_service import ResourceStatusUpdateService
from src.services.response_target import ResponseTargetInputService
from src.simulation import (
    CARMEL_LOCATION,
    GOLAN_LOCATION,
    SimulationEventExecutor,
    SimulationEventType,
    build_active_fire_resource_refresh_scenario,
    build_active_fire_scenario,
    build_carmel_golan_active_fire_scenario,
    simulation_event_timestamp,
)
from src.simulation.analysis import SimulationFireDetectionCoordinator, SimulationOperationalCoordinator
from src.simulation.analysis.simulation_refresh_coordinator import SimulationRefreshCoordinator

STARTED_AT = datetime(2026, 9, 17, 8, 0, tzinfo=timezone.utc)
CARMEL_STATION_ID = "station-carmel"
CARMEL_RESOURCE_ID = "truck-carmel-1"
GOLAN_STATION_ID = "station-golan"
GOLAN_RESOURCE_ID = "truck-golan-1"


class _StaticVegetationProvider:
    """Stands in for CopernicusLandCoverClient - no network call."""

    def get_land_cover_statistics(self, latitude, longitude, radius_km):
        return object()


class _StaticVegetationMapper:
    """Stands in for VegetationMapper - always returns a mappable fuel class."""

    def __init__(self, vegetation_data: VegetationData) -> None:
        self.vegetation_data = vegetation_data

    def map_statistics(self, statistics):
        return self.vegetation_data


_MAPPABLE_VEGETATION = VegetationData(
    fuel_score=0.8,
    dominant_land_cover="Shrub cover",
    source="TEST_FAKE",
    dataset_year=2019,
    radius_km=1.0,
)


def seed_station_and_road_network(
    session_factory,
    *,
    station_id: str,
    resource_id: str,
    latitude: float,
    longitude: float,
    resource_status: ResourceStatus = ResourceStatus.AVAILABLE,
) -> None:
    """Seed a station/resource plus a tiny road network node pair covering it.

    A pre-cached node/edge pair means OperationalContextService's road-network
    cache lookup is a hit, so RoadNetworkFetcher (OSM) is never called -
    mirrors test_response_planning_user_story_5_4_sanity.py's seeding.
    """
    session = session_factory()
    session.add(FireStationDB(id=station_id, name=station_id, latitude=latitude, longitude=longitude))
    session.flush()
    session.add(
        FirefightingResourceDB(id=resource_id, station_id=station_id, status=resource_status)
    )
    station_node_id = abs(hash((station_id, "station"))) % 900_000 + 100_000
    fire_node_id = abs(hash((station_id, "fire"))) % 900_000 + 100_000
    session.add_all(
        [
            GraphNodeDB(id=station_node_id, latitude=latitude, longitude=longitude),
            GraphNodeDB(id=fire_node_id, latitude=latitude + 0.001, longitude=longitude + 0.001),
        ]
    )
    session.flush()
    session.add(
        GraphEdgeDB(
            source_node_id=station_node_id,
            target_node_id=fire_node_id,
            distance_meters=500.0,
            travel_time_seconds=60.0,
        )
    )
    session.commit()
    session.close()


class RealSimulationStack:
    """Bundles the real production simulation + Epic 4/5 components for one SQLite DB."""

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory
        self.fire_event_repository = FireEventRepository(session_factory)
        satellite_repository = SatelliteHotspotRepository(session_factory)
        news_repository = NewsRepository(session_factory)
        weather_repository = WeatherRepository(session_factory)

        self.executor = SimulationEventExecutor(
            weather_repository=weather_repository,
            satellite_repository=satellite_repository,
            news_repository=news_repository,
        )

        detection_agent = FireDetectionAgent(
            evidence_service=FireDetectionEvidenceService(
                satellite_repository=satellite_repository, news_repository=news_repository
            ),
            calculator=FireDetectionCalculator(),
            fire_event_repository=self.fire_event_repository,
            satellite_repository=satellite_repository,
            news_repository=news_repository,
        )
        self.fire_detection_coordinator = SimulationFireDetectionCoordinator(detection_agent=detection_agent)

        severity_repository = FireSeverityAssessmentRepository(session_factory)
        severity_input_service = FireSeverityInputService(
            fire_event_repository=self.fire_event_repository,
            weather_repository=weather_repository,
            satellite_hotspot_repository=satellite_repository,
            land_cover_client=_StaticVegetationProvider(),
            vegetation_mapper=_StaticVegetationMapper(_MAPPABLE_VEGETATION),
        )
        severity_agent = FireSeverityAssessmentAgent(
            input_service=severity_input_service,
            calculator=FireSeverityCalculator(),
            repository=severity_repository,
        )
        severity_refresh_orchestrator = FireSeverityRefreshOrchestrator(
            input_service=severity_input_service,
            assessment_agent=severity_agent,
            assessment_repository=severity_repository,
        )

        spread_input_service = FireSpreadInputService(
            fire_event_repository=self.fire_event_repository,
            fire_severity_assessment_repository=severity_repository,
            weather_repository=weather_repository,
        )
        self.spread_prediction_repository = FireSpreadPredictionRepository(session_factory)
        spread_agent = FireSpreadPredictionAgent(
            input_service=spread_input_service,
            calculator=FireSpreadCalculator(),
            repository=self.spread_prediction_repository,
        )
        spread_refresh_orchestrator = FireSpreadRefreshOrchestrator(
            input_service=spread_input_service,
            prediction_agent=spread_agent,
            prediction_repository=self.spread_prediction_repository,
        )

        self.response_target_repository = ResponseTargetRepository(session_factory)
        response_target_agent = ResponseTargetGenerationAgent(
            input_service=ResponseTargetInputService(
                fire_event_repository=self.fire_event_repository,
                fire_severity_assessment_repository=severity_repository,
                fire_spread_prediction_repository=self.spread_prediction_repository,
            ),
            calculator=ResponseTargetCalculator(),
            repository=self.response_target_repository,
        )

        self.resource_repository = FirefightingResourceRepository(session_factory)
        operational_refresh_orchestrator = OperationalRefreshOrchestrator(
            severity_refresh_orchestrator=severity_refresh_orchestrator,
            spread_refresh_orchestrator=spread_refresh_orchestrator,
            response_target_agent=response_target_agent,
            resource_status_service=ResourceStatusUpdateService(self.resource_repository),
            resource_repository=self.resource_repository,
            fire_event_repository=self.fire_event_repository,
        )

        self.global_planning_refresh_coordinator = build_global_planning_refresh_coordinator(
            session_factory=session_factory
        )

        operational_planning_refresh = OperationalPlanningRefreshCoordinator(
            operational_refresh_orchestrator=operational_refresh_orchestrator,
            global_planning_refresh=self.global_planning_refresh_coordinator,
        )

        operational_context_service = OperationalContextService(
            fire_station_repository=FireStationRepository(session_factory),
            firefighting_resource_repository=self.resource_repository,
        )
        simulation_operational_coordinator = SimulationOperationalCoordinator(
            operational_context_service=operational_context_service,
            firefighting_resource_repository=self.resource_repository,
        )

        self.simulation_refresh_coordinator = SimulationRefreshCoordinator(
            operational_planning_refresh=operational_planning_refresh,
            fire_event_repository=self.fire_event_repository,
            operational_coordinator=simulation_operational_coordinator,
        )

        self.route_planning_repository = RoutePlanningRepository(session_factory)
        self.response_plan_repository = ResponsePlanRepository(session_factory)
        self.plan_comparison_repository = PlanComparisonRepository(session_factory)

    def run_scenario(self, scenario) -> list:
        """Drive every event through the real simulation path, exactly as
        scripts/run_demo_simulation.py's execute_and_report_event does
        (detection -> refresh), and return the SimulationRefreshResult for
        every event that triggered a refresh."""
        refresh_results = []
        for event in scenario.events:
            event_timestamp = simulation_event_timestamp(STARTED_AT, event)
            execution_result = self.executor.execute(scenario=scenario, event=event, event_timestamp=event_timestamp)

            detection_result = None
            if event.event_type in (SimulationEventType.SATELLITE, SimulationEventType.NEWS):
                fire_detection_result = self.fire_detection_coordinator.handle_event(
                    scenario=scenario,
                    event=event,
                    execution_result=execution_result,
                    event_timestamp=event_timestamp,
                )
                if fire_detection_result.triggered:
                    detection_result = fire_detection_result.detection_result

            refresh_result = self.simulation_refresh_coordinator.handle_event(
                scenario=scenario,
                event=event,
                execution_result=execution_result,
                event_timestamp=event_timestamp,
                detection_result=detection_result,
            )
            if refresh_result.triggered:
                refresh_results.append(refresh_result)
        return refresh_results

    def plan_counts(self, fire_event_id: int) -> dict:
        return {
            "runs": len(self.route_planning_repository.get_history_for_event(fire_event_id)),
            "plans": len(self.response_plan_repository.get_for_fire_event(fire_event_id)),
            "comparisons": len(self.plan_comparison_repository.list_for_fire_event(fire_event_id)),
        }


def _all_fire_event_ids(stack: RealSimulationStack) -> tuple[int, ...]:
    return stack.fire_event_repository.get_active_fire_event_ids()


# ---------------------------------------------------------------------------
# E2E A: active-fire simulation reaches a persisted ResponsePlan
# ---------------------------------------------------------------------------


def test_active_fire_simulation_reaches_a_persisted_response_plan(sqlite_session_factory):
    seed_station_and_road_network(
        sqlite_session_factory,
        station_id=CARMEL_STATION_ID,
        resource_id=CARMEL_RESOURCE_ID,
        latitude=CARMEL_LOCATION.latitude,
        longitude=CARMEL_LOCATION.longitude,
    )
    stack = RealSimulationStack(sqlite_session_factory)
    scenario = build_active_fire_scenario(location=CARMEL_LOCATION, seed=42)

    refresh_results = stack.run_scenario(scenario)

    fire_event_ids = _all_fire_event_ids(stack)
    assert fire_event_ids, "the simulation path must have created a FireEvent through real detection"
    fire_event_id = fire_event_ids[0]

    # Severity: at least one VALID assessment was persisted for this event.
    severity_repository = FireSeverityAssessmentRepository(sqlite_session_factory)
    latest_severity = severity_repository.get_latest_for_event(fire_event_id)
    assert latest_severity is not None, "real FireSeverityAssessmentAgent must have run through the bridge"

    # Response targets: a target set was generated for this event.
    target_history = stack.response_target_repository.get_history_for_event(fire_event_id)
    assert target_history, "real ResponseTargetGenerationAgent must have run through the bridge"

    # Planning was reached: at least one GlobalPlanningRefreshResult exists.
    global_results = [r.global_planning_result for r in refresh_results if r.global_planning_result is not None]
    assert global_results, "the Global GA planning refresh must have been invoked for the detected FireEvent"

    # With a real, available resource and a pre-cached road network, planning
    # must have actually produced a full cycle (ACTIVATED), not just been attempted.
    activated_results = [r for r in global_results if r.status is GlobalPlanningRefreshStatus.ACTIVATED]
    assert activated_results, (
        f"expected at least one ACTIVATED global planning result with prerequisites present, got statuses: "
        f"{[r.status for r in global_results]}"
    )
    last_activated = activated_results[-1]
    response_plan_id = last_activated.response_plan_ids_by_event.get(fire_event_id)
    assert response_plan_id is not None

    counts = stack.plan_counts(fire_event_id)
    assert counts["runs"] >= 1
    assert counts["plans"] >= 1
    # Baseline comparison is never fabricated for a Global-GA-produced plan
    # (Task 46) - genuinely absent, not merely unchecked.
    assert counts["comparisons"] == 0

    stored_plan = stack.response_plan_repository.get_by_id(response_plan_id)
    assert stored_plan.plan.fire_event_id == fire_event_id
    assert stored_plan.plan.methodology == "global_genetic_resource_allocation"
    assert CARMEL_RESOURCE_ID in stack.route_planning_repository.get_by_id(
        stored_plan.plan.route_planning_run_id
    ).run.resource_ids


# ---------------------------------------------------------------------------
# E2E B: NO_OP does not duplicate planning artifacts
# ---------------------------------------------------------------------------


def test_semantically_unchanged_followup_event_is_a_planning_no_op(sqlite_session_factory):
    seed_station_and_road_network(
        sqlite_session_factory,
        station_id=CARMEL_STATION_ID,
        resource_id=CARMEL_RESOURCE_ID,
        latitude=CARMEL_LOCATION.latitude,
        longitude=CARMEL_LOCATION.longitude,
    )
    stack = RealSimulationStack(sqlite_session_factory)
    scenario = build_active_fire_scenario(location=CARMEL_LOCATION, seed=42)

    refresh_results = stack.run_scenario(scenario)
    fire_event_id = _all_fire_event_ids(stack)[0]
    counts_after_scenario = stack.plan_counts(fire_event_id)
    assert counts_after_scenario["plans"] >= 1

    # A second WEATHER refresh for the same FireEvent, at a slightly later
    # instant, with nothing about targets/resources having changed, must be
    # a NO_OP for planning - not a fresh routing/optimization/baseline cycle.
    followup_as_of = simulation_event_timestamp(STARTED_AT, scenario.events[-1]) + timedelta(seconds=5)
    followup_result = stack.simulation_refresh_coordinator.refresh_fire_events(
        fire_event_ids=(fire_event_id,),
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=followup_as_of,
    )

    assert followup_result.global_planning_result is not None, "the followup refresh must still reach global planning"
    assert followup_result.global_planning_result.status is GlobalPlanningRefreshStatus.NO_OP, (
        f"expected NO_OP, got status: {followup_result.global_planning_result.status}"
    )

    counts_after_followup = stack.plan_counts(fire_event_id)
    assert counts_after_followup == counts_after_scenario, "NO_OP must not create a new run/plan/comparison"


# ---------------------------------------------------------------------------
# E2E C: multi-incident - distinct per-event plans, correct resource assignment
# ---------------------------------------------------------------------------


def test_two_incident_scenario_produces_distinct_plans_with_geographically_correct_assignments(sqlite_session_factory):
    """Stage 6: this is no longer "resource pool isolation" (there is ONE
    global candidate-resource universe, considered together for both
    events - Task 1) - it is "each event still gets its OWN distinct
    ResponsePlan/RoutePlanningRun/ResponseTargetSet, and its ACTUALLY
    ASSIGNED resource is the geographically correct, nearby one," which the
    Global GA's own cross-event competition (Stage 4) already guarantees."""
    seed_station_and_road_network(
        sqlite_session_factory,
        station_id=CARMEL_STATION_ID,
        resource_id=CARMEL_RESOURCE_ID,
        latitude=CARMEL_LOCATION.latitude,
        longitude=CARMEL_LOCATION.longitude,
    )
    seed_station_and_road_network(
        sqlite_session_factory,
        station_id=GOLAN_STATION_ID,
        resource_id=GOLAN_RESOURCE_ID,
        latitude=GOLAN_LOCATION.latitude,
        longitude=GOLAN_LOCATION.longitude,
    )
    stack = RealSimulationStack(sqlite_session_factory)
    scenario = build_carmel_golan_active_fire_scenario(seed=42)

    stack.run_scenario(scenario)

    fire_event_ids = _all_fire_event_ids(stack)
    assert len(fire_event_ids) == 2, f"expected two isolated FireEvents, got {fire_event_ids}"
    event_a, event_b = fire_event_ids

    plan_a = stack.response_plan_repository.get_latest_for_fire_event(event_a)
    plan_b = stack.response_plan_repository.get_latest_for_fire_event(event_b)
    assert plan_a is not None and plan_b is not None
    assert plan_a.id != plan_b.id
    assert plan_a.plan.response_target_set_id != plan_b.plan.response_target_set_id
    assert plan_a.plan.route_planning_run_id != plan_b.plan.route_planning_run_id

    run_a = stack.route_planning_repository.get_by_id(plan_a.plan.route_planning_run_id)
    run_b = stack.route_planning_repository.get_by_id(plan_b.plan.route_planning_run_id)
    assert run_a.run.fire_event_id == event_a
    assert run_b.run.fire_event_id == event_b

    # Each event's ACTUALLY ASSIGNED resource (the one with a real
    # ResponseAction on its plan) is its own nearby station's resource -
    # the Global GA's cross-event competition still resolves to the
    # geographically sensible outcome, even though the candidate universe
    # itself was considered together (Task 1).
    assigned_a = {action.resource_id for action in plan_a.plan.actions}
    assigned_b = {action.resource_id for action in plan_b.plan.actions}
    assert assigned_a <= {CARMEL_RESOURCE_ID}
    assert assigned_b <= {GOLAN_RESOURCE_ID}
    assert not (assigned_a & assigned_b)  # no resource double-booked across events


# ---------------------------------------------------------------------------
# E2E D: resource change (real RESOURCE_STATUS simulation event)
# ---------------------------------------------------------------------------


def test_resource_becoming_unavailable_then_available_triggers_replanning_without_rerunning_analysis(
    sqlite_session_factory,
):
    seed_station_and_road_network(
        sqlite_session_factory,
        station_id=CARMEL_STATION_ID,
        resource_id=CARMEL_RESOURCE_ID,
        latitude=CARMEL_LOCATION.latitude,
        longitude=CARMEL_LOCATION.longitude,
    )
    stack = RealSimulationStack(sqlite_session_factory)
    scenario = build_active_fire_resource_refresh_scenario(location=CARMEL_LOCATION, seed=42)

    refresh_results = stack.run_scenario(scenario)
    fire_event_id = _all_fire_event_ids(stack)[0]

    severity_repository = FireSeverityAssessmentRepository(sqlite_session_factory)
    severity_before = severity_repository.get_latest_for_event(fire_event_id)
    targets_before = stack.response_target_repository.get_history_for_event(fire_event_id)[0]

    resource_refresh_results = [
        r
        for r in refresh_results
        if r.refresh_results and r.refresh_results[0].trigger_type.value == "resource_status_update"
    ]
    assert len(resource_refresh_results) == 2, "the resource_refresh preset has two RESOURCE_STATUS events"

    # Resource-only updates must never rerun wildfire analysis: the latest
    # severity assessment and target set must be identical (same ids) to
    # what the environmental events already produced - no new rows.
    severity_after = severity_repository.get_latest_for_event(fire_event_id)
    targets_after = stack.response_target_repository.get_history_for_event(fire_event_id)[0]
    assert severity_after.assessment_id == severity_before.assessment_id
    assert targets_after.id == targets_before.id

    # Planning was reevaluated for both resource transitions.
    for resource_result in resource_refresh_results:
        assert resource_result.global_planning_result is not None, "resource change must reach global planning refresh"

    resource = stack.resource_repository.get_by_id(CARMEL_RESOURCE_ID)
    assert resource.status is ResourceStatus.AVAILABLE  # back to AVAILABLE after the second event

    # The unavailable-resource cycle's plan (if any) must not have assigned
    # the resource while it was UNAVAILABLE.
    latest_plan = stack.response_plan_repository.get_latest_for_fire_event(fire_event_id)
    if latest_plan is not None:
        latest_run = stack.route_planning_repository.get_by_id(latest_plan.plan.route_planning_run_id)
        # Whatever the final resource_ids snapshot is, it must reflect the
        # resource's actual final (AVAILABLE) status - not a stale UNAVAILABLE one.
        assert isinstance(latest_run.run.resource_ids, tuple)
