"""Pure assignment-change auditing (Stage 6 of the Global Multi-Incident
Optimizer refactor, Task 25).

Compares each resource's PREVIOUS current_assignment (loaded into
GlobalPlanningInput before the GA ran) against its NEW fire_event_id in the
winning GlobalOptimizationResult's actions - pure data-to-data comparison,
no DB access, so it can be computed here (or re-derived identically by any
caller) without a second query.

A hard-dispatched resource can only ever show UNCHANGED or (if its own
FireEvent gained/lost a slot it happens to occupy) still UNCHANGED - never
REASSIGNED, by construction (see global_hard_dispatch_lock.py): ordinary
global replanning structurally cannot move it to a different fire_event_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.models.current_global_assignment import CurrentGlobalAssignment
from src.models.global_response_action import GlobalResponseAction


class AssignmentChangeType(Enum):
    """One resource's assignment-change classification for one global generation."""

    UNCHANGED = "unchanged"
    NEW_ASSIGNMENT = "new_assignment"
    REASSIGNED = "reassigned"
    RELEASED = "released"


@dataclass(frozen=True)
class GlobalAssignmentChange:
    """One resource's before/after FireEvent ownership for one global generation."""

    resource_id: str
    previous_fire_event_id: int | None
    new_fire_event_id: int | None
    change_type: AssignmentChangeType

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id.strip():
            raise ValueError(f"resource_id must be a non-empty string, got {self.resource_id!r}")
        for field_name, value in (
            ("previous_fire_event_id", self.previous_fire_event_id),
            ("new_fire_event_id", self.new_fire_event_id),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{field_name} must be a positive integer or None, got {value!r}")
        if not isinstance(self.change_type, AssignmentChangeType):
            raise ValueError(f"change_type must be an AssignmentChangeType, got {self.change_type!r}")

        expected = _classify(self.previous_fire_event_id, self.new_fire_event_id)
        if self.change_type is not expected:
            raise ValueError(
                f"change_type {self.change_type!r} is inconsistent with previous_fire_event_id="
                f"{self.previous_fire_event_id!r}, new_fire_event_id={self.new_fire_event_id!r} "
                f"(expected {expected!r})."
            )


def _classify(previous_fire_event_id: int | None, new_fire_event_id: int | None) -> AssignmentChangeType:
    if previous_fire_event_id is None and new_fire_event_id is not None:
        return AssignmentChangeType.NEW_ASSIGNMENT
    if previous_fire_event_id is not None and new_fire_event_id is None:
        return AssignmentChangeType.RELEASED
    if previous_fire_event_id == new_fire_event_id:
        return AssignmentChangeType.UNCHANGED
    return AssignmentChangeType.REASSIGNED


def compute_assignment_changes(
    current_assignments: tuple[CurrentGlobalAssignment, ...],
    actions: tuple[GlobalResponseAction, ...],
) -> tuple[GlobalAssignmentChange, ...]:
    """One GlobalAssignmentChange per resource that was either previously
    assigned, newly assigned, or both - ordered by resource_id."""
    previous_by_resource = {assignment.resource_id: assignment.fire_event_id for assignment in current_assignments}
    new_by_resource = {action.resource_id: action.fire_event_id for action in actions}

    all_resource_ids = sorted(set(previous_by_resource) | set(new_by_resource))
    changes = []
    for resource_id in all_resource_ids:
        previous_fire_event_id = previous_by_resource.get(resource_id)
        new_fire_event_id = new_by_resource.get(resource_id)
        changes.append(
            GlobalAssignmentChange(
                resource_id=resource_id,
                previous_fire_event_id=previous_fire_event_id,
                new_fire_event_id=new_fire_event_id,
                change_type=_classify(previous_fire_event_id, new_fire_event_id),
            )
        )
    return tuple(changes)
