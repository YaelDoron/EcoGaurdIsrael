"""Stage 6 (dispatch/en-route hard locking + soft assignment stability)
full-pipeline acceptance tests for GlobalResponseOptimizationService - the
central scenarios the Stage 6 spec requires direct evidence for (Tasks
36-42): new-fire arrival while resources are en route, severity increase
with dispatched resources preserved, a second fire's severity increase,
hard lock + resource becoming UNAVAILABLE, and soft anti-thrashing for
still-reassignable (PLANNED) resources.
"""
from __future__ import annotations

from src.calculators.global_response_optimization.global_assignment_change_calculator import AssignmentChangeType
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_service import (
    GlobalResponseOptimizationService,
)
from src.models.demand_source import DemandSource
from src.models.dispatch_state import DispatchState
from src.models.fire_severity_level import FireSeverityLevel
from src.models.resource_status import ResourceStatus
from src.models.response_target_type import ResponseTargetType
from tests.calculators.global_response_optimization.helpers import (
    make_assignment,
    make_demand,
    make_input,
    make_resource,
    make_route,
    make_target,
)

RELIABLE_CONFIG = GlobalResponseOptimizationConfig(population_size=60, generation_count=80, random_seed=42)


def _service() -> GlobalResponseOptimizationService:
    return GlobalResponseOptimizationService()


# ---------------------------------------------------------------------------
# Task 36 - new fire arrival while resources are already dispatched
# ---------------------------------------------------------------------------


def test_new_fire_arrival_never_pulls_a_dispatched_resource_off_its_event():
    demand_a = make_demand(
        1, minimum_resources=2, desired_resources=2,
        severity_level=FireSeverityLevel.HIGH, severity_score=65.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    demand_b = make_demand(
        2, minimum_resources=1, desired_resources=2,
        severity_level=FireSeverityLevel.CRITICAL, severity_score=95.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    resources = (make_resource("R1"), make_resource("R2"), make_resource("R3"), make_resource("R4"))
    routes = (
        make_route("R1", 1, 10, eta_seconds=20.0),
        make_route("R2", 1, 10, eta_seconds=25.0),
        # R3/R4 are free and, despite B being CRITICAL, ALSO happen to be
        # excellent for A - a naive optimizer might want to steal R1/R2's
        # spot, but R1/R2 are hard-locked and must never move.
        make_route("R1", 2, 20, eta_seconds=5.0),
        make_route("R2", 2, 20, eta_seconds=5.0),
        make_route("R3", 1, 10, eta_seconds=200.0),
        make_route("R3", 2, 20, eta_seconds=30.0),
        make_route("R4", 1, 10, eta_seconds=200.0),
        make_route("R4", 2, 20, eta_seconds=35.0),
    )
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=165.0), make_target(2, 20, priority_score=195.0)),
        resources=resources,
        routes=routes,
        incident_demands=(demand_a, demand_b),
        current_assignments=(make_assignment("R1", 1, 10), make_assignment("R2", 1, 10)),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assignment = {action.resource_id: action.fire_event_id for action in result.actions}
    assert assignment["R1"] == 1
    assert assignment["R2"] == 1
    # B is genuinely served by the free resources.
    assert {"R3", "R4"} & {rid for rid, fid in assignment.items() if fid == 2}
    by_resource = {change.resource_id: change for change in result.assignment_changes}
    assert by_resource["R1"].change_type is AssignmentChangeType.UNCHANGED
    assert by_resource["R2"].change_type is AssignmentChangeType.UNCHANGED


# ---------------------------------------------------------------------------
# Task 37 - severity increase with already-dispatched resources preserved
# ---------------------------------------------------------------------------


def test_severity_increase_preserves_dispatched_resources_and_adds_reinforcement():
    # HIGH (min=2/desired=3) already dispatched with 3 resources -> severity
    # increases to CRITICAL (min=3/desired=4): R1,R2,R3 must stay; a 4th
    # free resource may be added.
    demand_critical = make_demand(
        1, minimum_resources=3, desired_resources=4,
        severity_level=FireSeverityLevel.CRITICAL, severity_score=95.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    resources = (make_resource("R1"), make_resource("R2"), make_resource("R3"), make_resource("R4"))
    routes = tuple(make_route(f"R{i}", 1, 10, eta_seconds=10.0 * i) for i in range(1, 5))
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=195.0),),
        resources=resources,
        routes=routes,
        incident_demands=(demand_critical,),
        current_assignments=(
            make_assignment("R1", 1, 10),
            make_assignment("R2", 1, 10),
            make_assignment("R3", 1, 10),
        ),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    resource_ids = {action.resource_id for action in result.actions}
    assert {"R1", "R2", "R3"}.issubset(resource_ids)
    assert resource_ids == {"R1", "R2", "R3", "R4"}
    by_resource = {change.resource_id: change for change in result.assignment_changes}
    for locked in ("R1", "R2", "R3"):
        assert by_resource[locked].change_type is AssignmentChangeType.UNCHANGED
    assert by_resource["R4"].change_type is AssignmentChangeType.NEW_ASSIGNMENT


def test_demand_decrease_never_recalls_an_already_dispatched_resource():
    """Task 8: desired demand DROPS below the already-dispatched count -
    ordinary replanning must never fabricate an implicit recall."""
    demand_low = make_demand(
        1, minimum_resources=1, desired_resources=2,
        severity_level=FireSeverityLevel.MODERATE, severity_score=30.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    resources = tuple(make_resource(f"R{i}") for i in range(1, 5))
    routes = tuple(make_route(f"R{i}", 1, 10, eta_seconds=10.0 * i) for i in range(1, 5))
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=130.0),),
        resources=resources,
        routes=routes,
        incident_demands=(demand_low,),
        current_assignments=tuple(make_assignment(f"R{i}", 1, 10) for i in range(1, 5)),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    resource_ids = {action.resource_id for action in result.actions}
    assert resource_ids == {"R1", "R2", "R3", "R4"}
    (event_result,) = result.event_results
    assert event_result.demand_result.locked_resources_preserved == 2
    assert event_result.demand_result.desired_resources == 2  # never inflated by the overflow


# ---------------------------------------------------------------------------
# Task 38 - second fire arrives then its severity increases
# ---------------------------------------------------------------------------


def test_second_fire_severity_increase_preserves_both_events_dispatched_resources():
    demand_a = make_demand(
        1, minimum_resources=2, desired_resources=2,
        severity_level=FireSeverityLevel.HIGH, severity_score=65.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    demand_b_critical = make_demand(
        2, minimum_resources=2, desired_resources=3,
        severity_level=FireSeverityLevel.CRITICAL, severity_score=95.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    resources = (make_resource("R1"), make_resource("R2"), make_resource("R3"), make_resource("R4"))
    routes = (
        make_route("R1", 1, 10, eta_seconds=10.0),
        make_route("R2", 1, 10, eta_seconds=15.0),
        make_route("R3", 2, 20, eta_seconds=10.0),
        make_route("R4", 2, 20, eta_seconds=20.0),
        make_route("R4", 1, 10, eta_seconds=500.0),
    )
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=165.0), make_target(2, 20, priority_score=195.0)),
        resources=resources,
        routes=routes,
        incident_demands=(demand_a, demand_b_critical),
        current_assignments=(
            make_assignment("R1", 1, 10),
            make_assignment("R2", 1, 10),
            make_assignment("R3", 2, 20),
        ),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assignment = {action.resource_id: action.fire_event_id for action in result.actions}
    assert assignment["R1"] == 1
    assert assignment["R2"] == 1
    assert assignment["R3"] == 2
    assert assignment.get("R4") == 2  # the only free resource reinforces B


# ---------------------------------------------------------------------------
# Task 40 - hard lock + resource becomes UNAVAILABLE
# ---------------------------------------------------------------------------


def test_dispatched_resource_going_unavailable_is_replaced_not_reassigned():
    demand_a = make_demand(1, minimum_resources=1, desired_resources=1)
    now_unavailable = make_resource("R1", operational_status=ResourceStatus.UNAVAILABLE)
    replacement = make_resource("R2")
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10),),
        resources=(now_unavailable, replacement),
        routes=(make_route("R1", 1, 10, eta_seconds=5.0), make_route("R2", 1, 10, eta_seconds=50.0)),
        incident_demands=(demand_a,),
        # R1's stale commitment metadata still says it's dispatched to fire
        # 1, even though its operational status has since flipped.
        current_assignments=(make_assignment("R1", 1, 10),),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    resource_ids = {action.resource_id for action in result.actions}
    assert "R1" not in resource_ids
    assert resource_ids == {"R2"}


# ---------------------------------------------------------------------------
# Task 42 - soft anti-thrashing for still-reassignable (PLANNED) resources
# ---------------------------------------------------------------------------


def test_planned_resource_is_preserved_when_alternative_improvement_is_trivial():
    demand_a = make_demand(1, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=100.0),),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=20.0),
            make_route("R2", 1, 10, eta_seconds=19.0),  # only marginally better
        ),
        incident_demands=(demand_a,),
        current_assignments=(make_assignment("R1", 1, 10, dispatch_state=DispatchState.PLANNED),),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    resource_ids = {action.resource_id for action in result.actions}
    assert resource_ids == {"R1"}


def test_planned_resource_is_replaced_when_a_genuinely_closer_resource_is_available():
    """Regression test for the reported dispatch bug: a resource already
    PLANNED to a target from far away (20 minutes) must lose its slot to a
    genuinely closer free resource (13 minutes - a real ~7-minute advantage,
    not the trivial ~1-minute gap covered by the test above). The soft
    stability bonus exists only to damp flickering between near-identical
    candidates, never to entrench a materially farther one over a closer
    available alternative."""
    demand_a = make_demand(1, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=100.0),),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=1200.0),  # 20 min - already PLANNED here
            make_route("R2", 1, 10, eta_seconds=780.0),  # 13 min - free, genuinely closer
        ),
        incident_demands=(demand_a,),
        current_assignments=(make_assignment("R1", 1, 10, dispatch_state=DispatchState.PLANNED),),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    resource_ids = {action.resource_id for action in result.actions}
    assert resource_ids == {"R2"}


def test_planned_resource_moves_when_improvement_is_a_genuine_emergency():
    """A PLANNED (soft) commitment must NEVER outrank Stage 5's own
    required-suppression-under-severity-shortage hierarchy."""
    demand_low = make_demand(
        1, minimum_resources=0, desired_resources=1,
        severity_level=FireSeverityLevel.LOW, severity_score=5.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    demand_critical = make_demand(
        2, minimum_resources=1, desired_resources=1,
        severity_level=FireSeverityLevel.CRITICAL, severity_score=95.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=105.0), make_target(2, 20, priority_score=195.0)),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0), make_route("R1", 2, 20, eta_seconds=10.0)),
        incident_demands=(demand_low, demand_critical),
        current_assignments=(make_assignment("R1", 1, 10, dispatch_state=DispatchState.PLANNED),),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assignment = {action.resource_id: action.fire_event_id for action in result.actions}
    assert assignment["R1"] == 2


def test_stability_bonus_never_beats_a_feasible_required_suppression_slot():
    """Section 13's explicit invariant, exercised end to end: an optional
    (predicted-risk) slot's stability bonus must never outscore a feasible
    REQUIRED suppression slot on another event."""
    demand_required = make_demand(1, minimum_resources=1, desired_resources=1)
    predicted_target = make_target(2, 21, target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30)
    demand_none = make_demand(2, minimum_resources=0, desired_resources=0)
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10), predicted_target),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0), make_route("R1", 2, 21, eta_seconds=10.0)),
        incident_demands=(demand_required, demand_none),
        current_assignments=(make_assignment("R1", 2, 21, dispatch_state=DispatchState.PLANNED),),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assignment = {action.resource_id: action.fire_event_id for action in result.actions}
    assert assignment["R1"] == 1
