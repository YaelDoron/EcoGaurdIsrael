"""GlobalPlanningInput: the single, Stage-4/5/6-ready input object (Stage 3
of the Global Multi-Incident Optimizer refactor; extended in Stage 5 with
severity-driven per-incident demand, and in Stage 6 with each resource's
current operational assignment/dispatch state).

Everything the future global GA needs, computed once and handed over as an
immutable snapshot: no repository queries, no Dijkstra, no DB writes happen
during Stage 4/5/6's fitness evaluation - it all already happened here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.models.current_global_assignment import CurrentGlobalAssignment
from src.models.global_incident_demand import GlobalIncidentDemand
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.global_planning_target import GlobalPlanningTarget
from src.models.global_route_matrix import GlobalRouteMatrix


@dataclass(frozen=True)
class GlobalPlanningInput:
    """The complete, validated, fingerprinted input to the future global GA."""

    global_planning_run_id: int
    as_of: datetime
    active_fire_event_ids: tuple[int, ...]
    targets: tuple[GlobalPlanningTarget, ...]
    resources: tuple[GlobalPlanningResource, ...]
    route_matrix: GlobalRouteMatrix
    event_target_set_ids: dict[int, int]
    incident_demands: tuple[GlobalIncidentDemand, ...]
    input_fingerprint: str
    current_assignments: tuple[CurrentGlobalAssignment, ...] = ()

    def __post_init__(self) -> None:
        _validate_positive_int("global_planning_run_id", self.global_planning_run_id)
        if not isinstance(self.as_of, datetime) or self.as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {self.as_of!r}")

        active_fire_event_ids = _coerce_tuple("active_fire_event_ids", self.active_fire_event_ids)
        for fire_event_id in active_fire_event_ids:
            _validate_positive_int("active_fire_event_ids", fire_event_id)
        if len(set(active_fire_event_ids)) != len(active_fire_event_ids):
            raise ValueError("active_fire_event_ids must not contain duplicates.")
        object.__setattr__(self, "active_fire_event_ids", active_fire_event_ids)
        active_fire_event_id_set = set(active_fire_event_ids)

        targets = _coerce_tuple("targets", self.targets)
        for target in targets:
            if not isinstance(target, GlobalPlanningTarget):
                raise ValueError(f"targets must contain GlobalPlanningTarget items, got {target!r}")
        object.__setattr__(self, "targets", targets)

        resources = _coerce_tuple("resources", self.resources)
        for resource in resources:
            if not isinstance(resource, GlobalPlanningResource):
                raise ValueError(f"resources must contain GlobalPlanningResource items, got {resource!r}")
        object.__setattr__(self, "resources", resources)

        if not isinstance(self.route_matrix, GlobalRouteMatrix):
            raise ValueError(f"route_matrix must be a GlobalRouteMatrix, got {self.route_matrix!r}")

        if not isinstance(self.event_target_set_ids, dict):
            raise ValueError(f"event_target_set_ids must be a dict, got {self.event_target_set_ids!r}")

        incident_demands = _coerce_tuple("incident_demands", self.incident_demands)
        for demand in incident_demands:
            if not isinstance(demand, GlobalIncidentDemand):
                raise ValueError(f"incident_demands must contain GlobalIncidentDemand items, got {demand!r}")
        demand_fire_event_ids = [demand.fire_event_id for demand in incident_demands]
        if len(set(demand_fire_event_ids)) != len(demand_fire_event_ids):
            raise ValueError("incident_demands must not contain duplicate fire_event_id entries.")
        if set(demand_fire_event_ids) != active_fire_event_id_set:
            raise ValueError(
                "incident_demands must contain exactly one record per active_fire_event_id, got demands for "
                f"{sorted(set(demand_fire_event_ids))!r} but active events {sorted(active_fire_event_id_set)!r}."
            )
        object.__setattr__(self, "incident_demands", incident_demands)

        if not isinstance(self.input_fingerprint, str) or not self.input_fingerprint.strip():
            raise ValueError(f"input_fingerprint must be a non-empty string, got {self.input_fingerprint!r}")

        current_assignments = _coerce_tuple("current_assignments", self.current_assignments)
        for assignment in current_assignments:
            if not isinstance(assignment, CurrentGlobalAssignment):
                raise ValueError(
                    f"current_assignments must contain CurrentGlobalAssignment items, got {assignment!r}"
                )
        assignment_resource_ids = [assignment.resource_id for assignment in current_assignments]
        if len(set(assignment_resource_ids)) != len(assignment_resource_ids):
            raise ValueError("current_assignments must not contain duplicate resource_id entries.")
        object.__setattr__(self, "current_assignments", current_assignments)

        self._validate_cross_references(active_fire_event_id_set, targets, resources)

    def _validate_cross_references(
        self,
        active_fire_event_id_set: set[int],
        targets: tuple[GlobalPlanningTarget, ...],
        resources: tuple[GlobalPlanningResource, ...],
    ) -> None:
        resource_ids = [resource.resource_id for resource in resources]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("resources must not contain duplicate resource_id values.")
        resource_id_set = set(resource_ids)

        target_ids = [target.response_target_id for target in targets]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("targets must not contain duplicate response_target_id values.")
        target_by_id = {target.response_target_id: target for target in targets}

        for target in targets:
            if target.fire_event_id not in active_fire_event_id_set:
                raise ValueError(
                    f"target {target.response_target_id!r} belongs to fire_event_id "
                    f"{target.fire_event_id!r}, which is not one of active_fire_event_ids."
                )

        for fire_event_id, response_target_set_id in self.event_target_set_ids.items():
            if fire_event_id not in active_fire_event_id_set:
                raise ValueError(
                    f"event_target_set_ids key {fire_event_id!r} is not one of active_fire_event_ids."
                )
            if isinstance(response_target_set_id, bool) or not isinstance(response_target_set_id, int) or response_target_set_id <= 0:
                raise ValueError(
                    f"event_target_set_ids values must be positive integers, got {response_target_set_id!r}"
                )
        events_with_targets = {target.fire_event_id for target in targets}
        missing_target_set_ids = events_with_targets - set(self.event_target_set_ids)
        if missing_target_set_ids:
            raise ValueError(
                f"targets exist for fire_event_ids {sorted(missing_target_set_ids)!r} with no matching "
                "entry in event_target_set_ids."
            )

        for option in self.route_matrix:
            if option.resource_id not in resource_id_set:
                raise ValueError(
                    f"route option references resource_id {option.resource_id!r}, which is not in resources."
                )
            target = target_by_id.get(option.response_target_id)
            if target is None:
                raise ValueError(
                    f"route option references response_target_id {option.response_target_id!r}, "
                    "which is not in targets."
                )
            if target.fire_event_id != option.fire_event_id:
                raise ValueError(
                    f"route option fire_event_id {option.fire_event_id!r} does not match target "
                    f"{option.response_target_id!r}'s owning fire_event_id {target.fire_event_id!r}."
                )

        for assignment in self.current_assignments:
            if assignment.resource_id not in resource_id_set:
                raise ValueError(
                    f"current_assignments references resource_id {assignment.resource_id!r}, "
                    "which is not in resources."
                )
            if assignment.fire_event_id not in active_fire_event_id_set:
                raise ValueError(
                    f"current_assignments entry for resource_id {assignment.resource_id!r} references "
                    f"fire_event_id {assignment.fire_event_id!r}, which is not one of active_fire_event_ids."
                )
            target = target_by_id.get(assignment.response_target_id)
            if target is None:
                raise ValueError(
                    f"current_assignments entry for resource_id {assignment.resource_id!r} references "
                    f"response_target_id {assignment.response_target_id!r}, which is not in targets."
                )
            if target.fire_event_id != assignment.fire_event_id:
                raise ValueError(
                    f"current_assignments entry for resource_id {assignment.resource_id!r} has fire_event_id "
                    f"{assignment.fire_event_id!r} not matching target {assignment.response_target_id!r}'s "
                    f"owning fire_event_id {target.fire_event_id!r}."
                )


def _coerce_tuple(field_name: str, value: object) -> tuple:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable, got {value!r}") from exc


def _validate_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
