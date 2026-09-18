"""Shared pure test builders for Stage 4 Global GA tests. GlobalPlanningInput
and everything the GA consumes are plain dataclasses, so tests never need a
database - the whole scenario is built in memory.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.models.current_global_assignment import CurrentGlobalAssignment
from src.models.demand_source import DemandSource
from src.models.dispatch_state import DispatchState
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_incident_demand import GlobalIncidentDemand
from src.models.global_planning_input import GlobalPlanningInput
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.global_planning_target import GlobalPlanningTarget
from src.models.resource_status import ResourceStatus
from src.models.response_target_type import ResponseTargetType
from src.models.global_route_matrix import GlobalRouteMatrix
from src.models.global_route_option import GlobalRouteOption

AS_OF = datetime(2026, 9, 23, 8, 0, tzinfo=timezone.utc)


def make_target(
    fire_event_id: int,
    response_target_id: int,
    *,
    target_order: int = 0,
    priority_score: float = 100.0,
    latitude: float = 32.7,
    longitude: float = 35.0,
    target_type: ResponseTargetType = ResponseTargetType.ACTIVE_FIRE,
    prediction_horizon_minutes: int | None = None,
) -> GlobalPlanningTarget:
    return GlobalPlanningTarget(
        fire_event_id=fire_event_id,
        response_target_id=response_target_id,
        target_order=target_order,
        target_type=target_type,
        latitude=latitude,
        longitude=longitude,
        priority_score=priority_score,
        prediction_horizon_minutes=prediction_horizon_minutes,
    )


def make_resource(
    resource_id: str,
    *,
    station_id: str | None = None,
    station_latitude: float = 32.7,
    station_longitude: float = 35.0,
    operational_status: ResourceStatus = ResourceStatus.AVAILABLE,
    current_commitment_fire_event_id: int | None = None,
    current_commitment_response_plan_id: int | None = None,
) -> GlobalPlanningResource:
    return GlobalPlanningResource(
        resource_id=resource_id,
        station_id=station_id or f"STATION-{resource_id}",
        station_name=station_id or f"STATION-{resource_id}",
        station_latitude=station_latitude,
        station_longitude=station_longitude,
        operational_status=operational_status,
        current_commitment_fire_event_id=current_commitment_fire_event_id,
        current_commitment_response_plan_id=current_commitment_response_plan_id,
    )


def make_route(
    resource_id: str,
    fire_event_id: int,
    response_target_id: int,
    *,
    eta_seconds: float,
    route_distance_meters: float | None = None,
    node_path: tuple[int, ...] = (1, 2),
) -> GlobalRouteOption:
    return GlobalRouteOption(
        resource_id=resource_id,
        fire_event_id=fire_event_id,
        response_target_id=response_target_id,
        eta_seconds=eta_seconds,
        route_distance_meters=route_distance_meters if route_distance_meters is not None else eta_seconds * 10,
        node_path=node_path,
    )


def make_demand(
    fire_event_id: int,
    *,
    minimum_resources: int = 0,
    desired_resources: int = 1,
    severity_level: FireSeverityLevel | None = None,
    severity_score: float | None = None,
    severity_assessment_id: int | None = None,
    demand_source: DemandSource = DemandSource.INSUFFICIENT_SEVERITY,
) -> GlobalIncidentDemand:
    """Default: minimum=0/desired=1, i.e. Stage 4's original "one optional
    slot per target" behavior - most Stage 4 tests never need to think
    about demand at all and can keep using make_input() unchanged."""
    return GlobalIncidentDemand(
        fire_event_id=fire_event_id,
        severity_assessment_id=severity_assessment_id,
        severity_level=severity_level,
        severity_score=severity_score,
        minimum_resources=minimum_resources,
        desired_resources=desired_resources,
        demand_source=demand_source,
        policy_methodology="test-demand-policy",
        policy_version="1.0",
    )


def make_assignment(
    resource_id: str,
    fire_event_id: int,
    response_target_id: int,
    *,
    response_plan_id: int = 1,
    dispatch_state: DispatchState = DispatchState.DISPATCHED,
) -> CurrentGlobalAssignment:
    return CurrentGlobalAssignment(
        resource_id=resource_id,
        fire_event_id=fire_event_id,
        response_target_id=response_target_id,
        response_plan_id=response_plan_id,
        dispatch_state=dispatch_state,
    )


def make_input(
    *,
    global_planning_run_id: int = 1,
    as_of: datetime = AS_OF,
    active_fire_event_ids: tuple[int, ...],
    targets: tuple[GlobalPlanningTarget, ...],
    resources: tuple[GlobalPlanningResource, ...],
    routes: tuple[GlobalRouteOption, ...],
    event_target_set_ids: dict[int, int] | None = None,
    incident_demands: tuple[GlobalIncidentDemand, ...] | None = None,
    current_assignments: tuple[CurrentGlobalAssignment, ...] = (),
    input_fingerprint: str = "f" * 64,
) -> GlobalPlanningInput:
    if event_target_set_ids is None:
        event_target_set_ids = {
            fire_event_id: 100 + fire_event_id
            for fire_event_id in {target.fire_event_id for target in targets}
        }
    if incident_demands is None:
        incident_demands = tuple(make_demand(fire_event_id) for fire_event_id in active_fire_event_ids)
    return GlobalPlanningInput(
        global_planning_run_id=global_planning_run_id,
        as_of=as_of,
        active_fire_event_ids=active_fire_event_ids,
        targets=targets,
        resources=resources,
        route_matrix=GlobalRouteMatrix(options=routes),
        event_target_set_ids=event_target_set_ids,
        incident_demands=incident_demands,
        current_assignments=current_assignments,
        input_fingerprint=input_fingerprint,
    )
