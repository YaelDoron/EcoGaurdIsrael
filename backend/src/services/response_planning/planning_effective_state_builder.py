"""Build the current PlanningEffectiveState from already-persisted operational data.

PlanningEffectiveStateBuilder reads the latest persisted ResponseTargetSet via
the same call US 5.1's RoutePlanningAgent uses
(ResponseTargetRepository.get_latest_for_event_as_of) and the currently
AVAILABLE firefighting resources for the same operational context
(OperationalContextService.get_available_operational_context, anchored on
that set's ACTIVE_FIRE target coordinates - the same anchor RoutePlanningAgent
uses to build routing's operational context). It does not recalculate fire
detection, severity, spread, response-target priority, or resource
availability - US 4.4/5.1 already keep those current; this builder only reads
and normalizes them into the Task 1 domain model.
"""
from __future__ import annotations

from datetime import datetime

from src.calculators.response_optimization.response_optimization_config import (
    METHODOLOGY as OPTIMIZATION_METHODOLOGY_NAME,
    METHODOLOGY_VERSION as OPTIMIZATION_METHODOLOGY_VERSION,
)
from src.calculators.routing.routing_config import ROUTING_METHODOLOGY_NAME, ROUTING_METHODOLOGY_VERSION
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models.planning_effective_state import PlanningEffectiveState, PlanningResourceState, PlanningTargetState
from src.models.planning_effective_state_result import PlanningEffectiveStateResult
from src.models.planning_effective_state_status import PlanningEffectiveStateStatus
from src.models.response_target import ResponseTarget
from src.models.response_target_type import ResponseTargetType
from src.repositories.response_target_repository import ResponseTargetRepository, StoredResponseTargetSet
from src.services.operational.operational_context_service import OperationalContextService


class PlanningEffectiveStateBuilder:
    """Build the current semantic planning state for one FireEvent."""

    def __init__(
        self,
        response_target_repository: ResponseTargetRepository | None = None,
        operational_context_service: OperationalContextService | None = None,
    ) -> None:
        self._response_target_repository = response_target_repository or ResponseTargetRepository()
        self._operational_context_service = operational_context_service or OperationalContextService()

    def build(self, fire_event_id: int, as_of: datetime) -> PlanningEffectiveStateResult:
        """Build the current PlanningEffectiveState for one FireEvent at a timezone-aware instant."""
        _validate_fire_event_id(fire_event_id)
        _validate_aware_datetime("as_of", as_of)

        stored_target_set = self._response_target_repository.get_latest_for_event_as_of(fire_event_id, as_of)
        if stored_target_set is None:
            return PlanningEffectiveStateResult(
                status=PlanningEffectiveStateStatus.NO_CURRENT_TARGETS,
                state=None,
                fire_event_id=fire_event_id,
            )
        if stored_target_set.target_set.fire_event_id != fire_event_id:
            raise ValueError(
                "response target set fire_event_id must match requested fire_event_id, got "
                f"{stored_target_set.target_set.fire_event_id!r} for request {fire_event_id!r}"
            )

        fire_target = _find_active_fire_target(stored_target_set)
        stations, available_resources = self._operational_context_service.get_available_operational_context(
            fire_target.latitude, fire_target.longitude
        )

        state = PlanningEffectiveState(
            fire_event_id=fire_event_id,
            targets=_to_planning_targets(stored_target_set),
            resources=_to_planning_resources(stations, available_resources),
            routing_methodology=ROUTING_METHODOLOGY_NAME,
            routing_methodology_version=ROUTING_METHODOLOGY_VERSION,
            optimization_methodology=OPTIMIZATION_METHODOLOGY_NAME,
            optimization_methodology_version=OPTIMIZATION_METHODOLOGY_VERSION,
        )
        return PlanningEffectiveStateResult(
            status=PlanningEffectiveStateStatus.BUILT,
            state=state,
            fire_event_id=fire_event_id,
        )


def _to_planning_targets(stored_target_set: StoredResponseTargetSet) -> tuple[PlanningTargetState, ...]:
    targets = tuple(
        PlanningTargetState(
            target_type=stored_target.target.target_type,
            latitude=stored_target.target.latitude,
            longitude=stored_target.target.longitude,
            priority_score=stored_target.target.priority_score,
            prediction_horizon_minutes=stored_target.target.prediction_horizon_minutes,
        )
        for stored_target in stored_target_set.targets
    )
    return tuple(sorted(targets, key=_target_sort_key))


def _to_planning_resources(
    stations: list[FireStationDB],
    available_resources: list[FirefightingResourceDB],
) -> tuple[PlanningResourceState, ...]:
    stations_by_id = {station.id: station for station in stations}
    resources = []
    for resource in available_resources:
        station = stations_by_id.get(resource.station_id)
        if station is None:
            raise ValueError(
                f"Firefighting resource {resource.id!r} references station {resource.station_id!r}, "
                "which is not part of this operational context."
            )
        resources.append(
            PlanningResourceState(
                resource_id=resource.id,
                station_id=resource.station_id,
                station_latitude=station.latitude,
                station_longitude=station.longitude,
            )
        )
    return tuple(sorted(resources, key=lambda resource: (resource.station_id, resource.resource_id)))


def _find_active_fire_target(stored_target_set: StoredResponseTargetSet) -> ResponseTarget:
    for stored_target in stored_target_set.targets:
        if stored_target.target.target_type is ResponseTargetType.ACTIVE_FIRE:
            return stored_target.target
    raise ValueError(
        f"ResponseTargetSet {stored_target_set.id!r} has no ACTIVE_FIRE target; this should be impossible."
    )


def _target_sort_key(target: PlanningTargetState) -> tuple[str, float, float, float, int]:
    return (
        target.target_type.value,
        target.latitude,
        target.longitude,
        target.priority_score,
        -1 if target.prediction_horizon_minutes is None else target.prediction_horizon_minutes,
    )


def _validate_fire_event_id(fire_event_id: object) -> None:
    if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
        raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")


def _validate_aware_datetime(field_name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")
