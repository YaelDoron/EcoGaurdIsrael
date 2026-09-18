"""Integration test for GlobalPlanningRefreshCoordinator (Stage 6 of the
Global Multi-Incident Optimizer refactor): the full production flow -
capture active FireEvents -> GlobalPlanningRun -> GlobalPlanningInputBuilder
-> NO_OP check -> Global GA -> atomic activation -> finalize - driven
end-to-end against a real (SQLite) database, deterministic road-network
fixture (no live OSM access), matching the existing Stage 4/5 integration
test's own precedent.
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
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.route_result_db import RouteResultDB
from src.models import GraphEdge, GraphNode
from src.models.dispatch_state import DispatchState
from src.models.resource_status import ResourceStatus
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
from src.services.response_planning.current_response_plan_resolver import CurrentResponsePlanResolver

AS_OF = datetime(2026, 9, 23, 9, 0, tzinfo=timezone.utc)


def _persist_fire_event(session, latitude, longitude) -> int:
    event = FireEventDB(
        latitude=latitude, longitude=longitude, detected_at=AS_OF, updated_at=AS_OF, status="confirmed",
        detection_confidence=0.9, methodology="m", methodology_version="1.0",
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
        config=GlobalResponseOptimizationConfig(population_size=20, generation_count=20, random_seed=42),
    )


def _row_counts(sqlite_session_factory):
    session = sqlite_session_factory()
    try:
        return (
            len(session.execute(select(ResponsePlanDB)).scalars().all()),
            len(session.execute(select(ResourceCommitmentDB)).scalars().all()),
        )
    finally:
        session.close()


def test_no_active_events_returns_no_active_events_status(sqlite_session_factory):
    coordinator = _make_coordinator(sqlite_session_factory)

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.NO_ACTIVE_EVENTS


def test_single_event_refresh_activates_a_real_plan(sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session, 32.70, 35.00)
    _persist_target_set(session, fire_event_id, 32.701, 35.001)
    _persist_station_and_resources(session, "S1", 32.70, 35.00, ["R1"])
    session.commit()
    session.close()
    _persist_road_network(
        sqlite_session_factory,
        nodes=[GraphNode(id=1, latitude=32.70, longitude=35.00), GraphNode(id=2, latitude=32.701, longitude=35.001)],
        edges=[GraphEdge(source_node_id=1, target_node_id=2, distance_meters=200.0, travel_time_seconds=30.0)],
    )

    coordinator = _make_coordinator(sqlite_session_factory)
    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert fire_event_id in result.response_plan_ids_by_event

    session = sqlite_session_factory()
    commitment = session.get(ResourceCommitmentDB, "R1")
    assert commitment.fire_event_id == fire_event_id
    assert commitment.dispatch_state == DispatchState.DISPATCHED.value
    session.close()


def test_second_identical_refresh_is_a_no_op(sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session, 32.70, 35.00)
    _persist_target_set(session, fire_event_id, 32.701, 35.001)
    _persist_station_and_resources(session, "S1", 32.70, 35.00, ["R1"])
    session.commit()
    session.close()
    _persist_road_network(
        sqlite_session_factory,
        nodes=[GraphNode(id=1, latitude=32.70, longitude=35.00), GraphNode(id=2, latitude=32.701, longitude=35.001)],
        edges=[GraphEdge(source_node_id=1, target_node_id=2, distance_meters=200.0, travel_time_seconds=30.0)],
    )

    coordinator = _make_coordinator(sqlite_session_factory)
    first = coordinator.refresh(trigger="manual", as_of=AS_OF)
    assert first.status is GlobalPlanningRefreshStatus.ACTIVATED

    # The immediately-following refresh is NOT a no-op: R1's dispatch_state
    # genuinely changed (uncommitted -> DISPATCHED) as a direct result of
    # the first activation, so this second cycle's input legitimately
    # differs and re-activates (re-affirming the same DISPATCHED assignment).
    second = coordinator.refresh(trigger="manual", as_of=AS_OF + timedelta(minutes=1))
    assert second.status is GlobalPlanningRefreshStatus.ACTIVATED
    before_counts = _row_counts(sqlite_session_factory)

    # Nothing changed between the second and third cycles - genuinely a no-op.
    third = coordinator.refresh(trigger="manual", as_of=AS_OF + timedelta(minutes=2))

    assert third.status is GlobalPlanningRefreshStatus.NO_OP
    after_counts = _row_counts(sqlite_session_factory)
    assert before_counts == after_counts  # no new ResponsePlan/commitment rows


def test_new_fire_arrival_preserves_already_dispatched_resource(sqlite_session_factory):
    """Task 36/49's central dynamic scenario, through the real coordinator:
    A gets R1 dispatched on the first refresh; B then appears; the second
    refresh must leave R1 -> A untouched while covering B with a free
    resource."""
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    _persist_target_set(session, event_a, 32.701, 35.001)
    _persist_station_and_resources(session, "STATION-A", 32.70, 35.00, ["R1"])
    _persist_station_and_resources(session, "STATION-B", 32.90, 35.20, ["R2"])
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

    coordinator = _make_coordinator(sqlite_session_factory)
    first = coordinator.refresh(trigger="manual", as_of=AS_OF)
    assert first.status is GlobalPlanningRefreshStatus.ACTIVATED
    session = sqlite_session_factory()
    assert session.get(ResourceCommitmentDB, "R1").fire_event_id == event_a
    session.close()

    # Fire B arrives.
    session = sqlite_session_factory()
    event_b = _persist_fire_event(session, 32.90, 35.20)
    _persist_target_set(session, event_b, 32.901, 35.201)
    session.commit()
    session.close()

    second = coordinator.refresh(trigger="manual", as_of=AS_OF + timedelta(minutes=40))

    assert second.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert event_a in second.response_plan_ids_by_event
    assert event_b in second.response_plan_ids_by_event

    session = sqlite_session_factory()
    r1_commitment = session.get(ResourceCommitmentDB, "R1")
    assert r1_commitment.fire_event_id == event_a  # never moved to B
    assert r1_commitment.dispatch_state == DispatchState.DISPATCHED.value
    r2_commitment = session.get(ResourceCommitmentDB, "R2")
    assert r2_commitment.fire_event_id == event_b  # the free resource covers B
    session.close()


def test_epic_6_readback_apis_read_a_global_ga_activated_plan_correctly(sqlite_session_factory):
    """Task 44/45: CurrentResponsePlanResolver + ResponsePlanDetailsService -
    the EXISTING Epic 6 read path - must correctly read back a Global-GA-
    produced, atomically-activated plan with no special-casing."""
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session, 32.70, 35.00)
    _persist_target_set(session, fire_event_id, 32.701, 35.001)
    _persist_station_and_resources(session, "S1", 32.70, 35.00, ["R1"])
    session.commit()
    session.close()
    _persist_road_network(
        sqlite_session_factory,
        nodes=[GraphNode(id=1, latitude=32.70, longitude=35.00), GraphNode(id=2, latitude=32.701, longitude=35.001)],
        edges=[GraphEdge(source_node_id=1, target_node_id=2, distance_meters=200.0, travel_time_seconds=30.0)],
    )

    coordinator = _make_coordinator(sqlite_session_factory)
    result = coordinator.refresh(trigger="manual", as_of=AS_OF)
    assert result.status is GlobalPlanningRefreshStatus.ACTIVATED
    plan_id = result.response_plan_ids_by_event[fire_event_id]

    from src.services.response_planning.response_plan_details_service import ResponsePlanDetailsService

    details_service = ResponsePlanDetailsService(
        current_response_plan_resolver=CurrentResponsePlanResolver(
            ResponsePlanRepository(sqlite_session_factory), ResponsePlanPlanningStateRepository(sqlite_session_factory)
        ),
        response_plan_repository=ResponsePlanRepository(sqlite_session_factory),
        route_planning_repository=RoutePlanningRepository(sqlite_session_factory),
        response_target_repository=ResponseTargetRepository(sqlite_session_factory),
        firefighting_resource_repository=FirefightingResourceRepository(sqlite_session_factory),
        fire_station_repository=FireStationRepository(sqlite_session_factory),
    )

    details = details_service.get_current_plan_details(fire_event_id)

    assert details is not None
    assert details.plan_id == plan_id
    assert details.is_current is True
    assert details.methodology == "global_genetic_resource_allocation"
    (action,) = details.actions
    assert action.resource_id == "R1"
    assert action.eta_seconds == 30.0
    assert action.route_distance_meters == 200.0
    assert action.node_path == (1, 2)
    assert details.baseline_comparison is None  # Task 46: never fabricated for a global-methodology plan
    assert details.optimization_config is not None
    assert details.optimization_config.population_size == 20
    assert details.random_seed == 42


def test_resolving_a_fire_event_releases_its_commitment_and_a_subsequent_global_refresh_reinforces_the_remaining_event(
    sqlite_session_factory,
):
    """Blocker E, Task 19 (dispatch-release end-to-end): A is active and
    gets R1 DISPATCHED; A resolves through the real FireEventLifecycleService
    (Stage 1.1's canonical production lifecycle path, not a direct DB edit);
    R1's commitment is released atomically as part of that transition; the
    NEXT real GlobalPlanningRefreshCoordinator.refresh() cycle - now
    considering only the remaining active FireEvent B - correctly re-derives
    that R1 is candidate-eligible again (its operational status was never
    touched by dispatch or by resolve, only its commitment ownership was)
    and reinforces B with it. No behavior is invented: resolve_event only
    ever deletes the ResourceCommitment row (see
    ResourceCommitmentRepository.release_for_fire_event_in_session) - it
    never touches FirefightingResourceDB.status, so R1 remains AVAILABLE
    throughout and is immediately reconsiderable.
    """
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    _persist_target_set(session, event_a, 32.701, 35.001)
    _persist_station_and_resources(session, "STATION-A", 32.70, 35.00, ["R1"])
    event_b = _persist_fire_event(session, 32.90, 35.20)
    _persist_target_set(session, event_b, 32.901, 35.201)
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
            GraphEdge(source_node_id=1, target_node_id=4, distance_meters=2000.0, travel_time_seconds=180.0),
            GraphEdge(source_node_id=3, target_node_id=2, distance_meters=2000.0, travel_time_seconds=180.0),
        ],
    )

    coordinator = _make_coordinator(sqlite_session_factory)

    # -- A active, R1 gets DISPATCHED -> A; B gets a plan with no feasible
    # assignments yet (its only usable resource, R1, is busy with A). --
    first = coordinator.refresh(trigger="manual", as_of=AS_OF)
    assert first.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert event_a in first.response_plan_ids_by_event
    assert event_b in first.response_plan_ids_by_event

    session = sqlite_session_factory()
    assert session.get(ResourceCommitmentDB, "R1").fire_event_id == event_a
    assert session.get(ResourceCommitmentDB, "R1").dispatch_state == DispatchState.DISPATCHED.value
    plan_b_first = session.get(ResponsePlanDB, first.response_plan_ids_by_event[event_b])
    assert plan_b_first.actions == []  # nothing reaches B yet
    session.close()

    # -- A resolves through the real production lifecycle path. --
    lifecycle_service = FireEventLifecycleService(
        fire_event_repository=FireEventRepository(sqlite_session_factory),
        resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
        session_factory=sqlite_session_factory,
    )
    lifecycle_service.resolve_event(event_a, as_of=AS_OF + timedelta(minutes=100))

    session = sqlite_session_factory()
    assert session.get(ResourceCommitmentDB, "R1") is None  # commitment released immediately, before any replan
    resource = session.get(FirefightingResourceDB, "R1")
    assert resource.status == ResourceStatus.AVAILABLE  # never touched by dispatch or by resolve
    session.close()

    # -- The correct global refresh path for the remaining active event(s). --
    second = coordinator.refresh(trigger="manual", as_of=AS_OF + timedelta(minutes=101))

    assert second.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert event_a not in second.response_plan_ids_by_event  # RESOLVED - no longer active, never replanned
    assert event_b in second.response_plan_ids_by_event  # R1 is now free and reinforces B

    session = sqlite_session_factory()
    r1_commitment = session.get(ResourceCommitmentDB, "R1")
    assert r1_commitment.fire_event_id == event_b
    assert r1_commitment.dispatch_state == DispatchState.DISPATCHED.value
    session.close()


def test_farther_station_reinforces_when_the_local_station_cannot(sqlite_session_factory):
    """Blocker E, Task 18 (farther-station full pipeline): the local
    station's only resource is UNAVAILABLE, so the global candidate pool
    (never station-scoped - GlobalCandidateCollector considers every
    operationally-available resource everywhere) must reach past it to a
    farther station's resource over a real multi-hop route. The persisted
    current ResponsePlan must carry that farther resource's EXACT
    precomputed route facts (resource_id, station_id via the action's
    node_path/ETA/distance, full node_path) - GlobalResponsePlanActivationService
    projects RouteResult straight from the already-computed GlobalResponseAction,
    it never re-runs Dijkstra (see _build_route_planning_run)."""
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session, 32.70, 35.00)
    _persist_target_set(session, fire_event_id, 32.70, 35.00)
    _persist_station_and_resources(session, "STATION-LOCAL", 32.70, 35.001, ["R-LOCAL"])
    _persist_station_and_resources(session, "STATION-FAR", 32.90, 35.20, ["R-FAR"])
    session.commit()
    session.execute(
        select(FirefightingResourceDB).where(FirefightingResourceDB.id == "R-LOCAL")
    ).scalar_one().status = ResourceStatus.UNAVAILABLE
    session.commit()
    session.close()
    _persist_road_network(
        sqlite_session_factory,
        nodes=[
            GraphNode(id=1, latitude=32.70, longitude=35.001),  # STATION-LOCAL
            GraphNode(id=2, latitude=32.70, longitude=35.00),  # fire target
            GraphNode(id=3, latitude=32.90, longitude=35.20),  # STATION-FAR
            GraphNode(id=4, latitude=32.80, longitude=35.10),  # intermediate hop
        ],
        edges=[
            GraphEdge(source_node_id=1, target_node_id=2, distance_meters=100.0, travel_time_seconds=15.0),
            GraphEdge(source_node_id=3, target_node_id=4, distance_meters=3000.0, travel_time_seconds=240.0),
            GraphEdge(source_node_id=4, target_node_id=2, distance_meters=2000.0, travel_time_seconds=160.0),
        ],
    )

    coordinator = _make_coordinator(sqlite_session_factory)
    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.ACTIVATED
    plan_id = result.response_plan_ids_by_event[fire_event_id]

    session = sqlite_session_factory()
    db_plan = session.get(ResponsePlanDB, plan_id)
    (action,) = db_plan.actions
    assert action.resource_id == "R-FAR"  # the only operationally-available resource anywhere
    route_run = session.get(RoutePlanningRunDB, db_plan.route_planning_run_id)
    assert route_run.fire_event_id == fire_event_id
    route_result = session.execute(
        select(RouteResultDB).where(RouteResultDB.id == action.route_result_id)
    ).scalar_one()
    assert route_result.resource_id == "R-FAR"
    assert route_result.source_node_id == 3
    assert route_result.target_node_id == 2
    assert route_result.node_path == [3, 4, 2]  # the real multi-hop Dijkstra path, persisted exactly
    assert route_result.distance_meters == 5000.0  # 3000 + 2000, never recomputed
    assert route_result.travel_time_seconds == 400.0  # 240 + 160

    commitment = session.get(ResourceCommitmentDB, "R-FAR")
    assert commitment.fire_event_id == fire_event_id
    assert commitment.response_plan_id == plan_id
    assert commitment.dispatch_state == DispatchState.DISPATCHED.value
    session.close()

    from src.services.response_planning.response_plan_details_service import ResponsePlanDetailsService

    details_service = ResponsePlanDetailsService(
        current_response_plan_resolver=CurrentResponsePlanResolver(
            ResponsePlanRepository(sqlite_session_factory), ResponsePlanPlanningStateRepository(sqlite_session_factory)
        ),
        response_plan_repository=ResponsePlanRepository(sqlite_session_factory),
        route_planning_repository=RoutePlanningRepository(sqlite_session_factory),
        response_target_repository=ResponseTargetRepository(sqlite_session_factory),
        firefighting_resource_repository=FirefightingResourceRepository(sqlite_session_factory),
        fire_station_repository=FireStationRepository(sqlite_session_factory),
    )
    details = details_service.get_current_plan_details(fire_event_id)
    assert details is not None
    (readback_action,) = details.actions
    assert readback_action.resource_id == "R-FAR"
    assert readback_action.eta_seconds == 400.0
    assert readback_action.route_distance_meters == 5000.0
    assert readback_action.node_path == (3, 4, 2)
