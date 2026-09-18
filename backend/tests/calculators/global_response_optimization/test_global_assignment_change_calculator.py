"""Tests for compute_assignment_changes (Stage 6 of the Global
Multi-Incident Optimizer refactor, Task 25)."""
from __future__ import annotations

import pytest

from src.calculators.global_response_optimization.global_assignment_change_calculator import (
    AssignmentChangeType,
    GlobalAssignmentChange,
    compute_assignment_changes,
)
from src.models.current_global_assignment import CurrentGlobalAssignment
from src.models.dispatch_state import DispatchState
from src.models.global_response_action import GlobalResponseAction
from src.models.response_target_type import ResponseTargetType


def _assignment(resource_id: str, fire_event_id: int, response_target_id: int = 10) -> CurrentGlobalAssignment:
    return CurrentGlobalAssignment(
        resource_id=resource_id,
        fire_event_id=fire_event_id,
        response_target_id=response_target_id,
        response_plan_id=1,
        dispatch_state=DispatchState.PLANNED,
    )


def _action(resource_id: str, fire_event_id: int, response_target_id: int = 10) -> GlobalResponseAction:
    return GlobalResponseAction(
        resource_id=resource_id,
        station_id="S1",
        fire_event_id=fire_event_id,
        response_target_id=response_target_id,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        target_priority=100.0,
        eta_seconds=10.0,
        route_distance_meters=100.0,
        node_path=(1, 2),
    )


def test_new_assignment_for_a_previously_uncommitted_resource():
    changes = compute_assignment_changes((), (_action("R1", 1),))
    (change,) = changes
    assert change.change_type is AssignmentChangeType.NEW_ASSIGNMENT
    assert change.previous_fire_event_id is None
    assert change.new_fire_event_id == 1


def test_released_when_a_previously_committed_resource_gets_no_new_action():
    changes = compute_assignment_changes((_assignment("R1", 1),), ())
    (change,) = changes
    assert change.change_type is AssignmentChangeType.RELEASED
    assert change.previous_fire_event_id == 1
    assert change.new_fire_event_id is None


def test_unchanged_when_the_fire_event_stays_the_same():
    changes = compute_assignment_changes((_assignment("R1", 1),), (_action("R1", 1),))
    (change,) = changes
    assert change.change_type is AssignmentChangeType.UNCHANGED


def test_reassigned_when_the_fire_event_differs():
    changes = compute_assignment_changes((_assignment("R1", 1),), (_action("R1", 2),))
    (change,) = changes
    assert change.change_type is AssignmentChangeType.REASSIGNED
    assert change.previous_fire_event_id == 1
    assert change.new_fire_event_id == 2


def test_multiple_resources_are_each_classified_independently():
    changes = compute_assignment_changes(
        (_assignment("R1", 1), _assignment("R2", 1)),
        (_action("R1", 1), _action("R3", 2)),
    )
    by_resource = {change.resource_id: change for change in changes}
    assert by_resource["R1"].change_type is AssignmentChangeType.UNCHANGED
    assert by_resource["R2"].change_type is AssignmentChangeType.RELEASED
    assert by_resource["R3"].change_type is AssignmentChangeType.NEW_ASSIGNMENT


def test_results_are_ordered_by_resource_id():
    changes = compute_assignment_changes((), (_action("R2", 1), _action("R1", 1)))
    assert [change.resource_id for change in changes] == ["R1", "R2"]


def test_rejects_change_type_inconsistent_with_before_after_state():
    with pytest.raises(ValueError):
        GlobalAssignmentChange(
            resource_id="R1",
            previous_fire_event_id=1,
            new_fire_event_id=2,
            change_type=AssignmentChangeType.UNCHANGED,
        )
