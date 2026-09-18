"""Tests for the hard dispatch lock mechanics (Stage 6 of the Global
Multi-Incident Optimizer refactor, Tasks 3/6/16): build_global_optimization_problem's
feasible-slot restriction + GlobalHardDispatchLockInfeasible, and
enforce_hard_locks' repair guarantee.
"""
from __future__ import annotations

import pytest

from src.calculators.global_response_optimization.global_hard_dispatch_lock import (
    GlobalHardDispatchLockInfeasible,
    enforce_hard_locks,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    build_global_optimization_problem,
)
from src.models.dispatch_state import DispatchState
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome
from src.models.response_target_type import ResponseTargetType
from tests.calculators.global_response_optimization.helpers import (
    make_assignment,
    make_demand,
    make_input,
    make_resource,
    make_route,
    make_target,
)


def test_locked_resource_feasible_slots_restricted_to_its_own_event():
    demand_a = make_demand(1, minimum_resources=1, desired_resources=1)
    demand_b = make_demand(2, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10), make_target(2, 20)),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0), make_route("R1", 2, 20, eta_seconds=5.0)),
        incident_demands=(demand_a, demand_b),
        current_assignments=(make_assignment("R1", 1, 10),),
    )

    problem = build_global_optimization_problem(global_input)

    assert problem.locked_resource_ids == frozenset({"R1"})
    assert problem.feasible_slot_ids_by_resource["R1"] == ("10#0",)


def test_locked_resource_with_no_feasible_route_to_its_own_event_raises():
    demand_a = make_demand(1, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10),),
        resources=(make_resource("R1"),),
        routes=(),  # R1 has no feasible route to its own locked event at all
        incident_demands=(demand_a,),
        current_assignments=(make_assignment("R1", 1, 10),),
    )

    with pytest.raises(GlobalHardDispatchLockInfeasible):
        build_global_optimization_problem(global_input)


def test_locked_resource_count_beyond_desired_generates_overflow_slots_it_can_use():
    demand_a = make_demand(1, minimum_resources=1, desired_resources=2)
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10),),
        resources=(make_resource("R1"), make_resource("R2"), make_resource("R3")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=10.0),
            make_route("R2", 1, 10, eta_seconds=20.0),
            make_route("R3", 1, 10, eta_seconds=30.0),
        ),
        incident_demands=(demand_a,),
        current_assignments=(
            make_assignment("R1", 1, 10),
            make_assignment("R2", 1, 10),
            make_assignment("R3", 1, 10),
        ),
    )

    problem = build_global_optimization_problem(global_input)

    # desired=2, but 3 resources are locked - one overflow slot must exist.
    assert len({slot.slot_id for slot in problem.slots}) == 3
    for resource_id in ("R1", "R2", "R3"):
        assert len(problem.feasible_slot_ids_by_resource[resource_id]) == 3


def test_planned_resource_is_not_locked():
    demand_a = make_demand(1, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10),),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0),),
        incident_demands=(demand_a,),
        current_assignments=(make_assignment("R1", 1, 10, dispatch_state=DispatchState.PLANNED),),
    )

    problem = build_global_optimization_problem(global_input)

    assert problem.locked_resource_ids == frozenset()
    assert "R1" in problem.current_assignment_by_resource


# ---------------------------------------------------------------------------
# enforce_hard_locks
# ---------------------------------------------------------------------------


def _problem_with_two_events_one_locked_resource():
    demand_a = make_demand(1, minimum_resources=1, desired_resources=1)
    demand_b = make_demand(2, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10), make_target(2, 20)),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0), make_route("R1", 2, 20, eta_seconds=5.0)),
        incident_demands=(demand_a, demand_b),
        current_assignments=(make_assignment("R1", 1, 10),),
    )
    return build_global_optimization_problem(global_input)


def test_enforce_hard_locks_fixes_an_idle_locked_resource():
    problem = _problem_with_two_events_one_locked_resource()
    idle = GlobalResponsePlanChromosome(problem.resource_ids, (None,))

    fixed = enforce_hard_locks(problem, idle)

    assert dict(zip(fixed.resource_ids, fixed.genes))["R1"] == "10#0"


def test_enforce_hard_locks_evicts_a_competitor_from_the_only_feasible_slot():
    demand_a = make_demand(1, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10),),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0), make_route("R2", 1, 10, eta_seconds=20.0)),
        incident_demands=(demand_a,),
        current_assignments=(make_assignment("R1", 1, 10),),
    )
    problem = build_global_optimization_problem(global_input)
    # R2 illegally squatting on R1's only feasible (and only existing) slot.
    illegal = GlobalResponsePlanChromosome(problem.resource_ids, (None, "10#0"))

    fixed = enforce_hard_locks(problem, illegal)

    gene_by_resource = dict(zip(fixed.resource_ids, fixed.genes))
    assert gene_by_resource["R1"] == "10#0"
    assert gene_by_resource["R2"] is None


def test_enforce_hard_locks_leaves_already_legal_chromosome_untouched():
    problem = _problem_with_two_events_one_locked_resource()
    legal = GlobalResponsePlanChromosome(problem.resource_ids, ("10#0",))

    fixed = enforce_hard_locks(problem, legal)

    assert fixed.genes == legal.genes


def test_enforce_hard_locks_is_a_no_op_when_nothing_is_locked():
    demand_a = make_demand(1, minimum_resources=0, desired_resources=1)
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10),),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0),),
        incident_demands=(demand_a,),
    )
    problem = build_global_optimization_problem(global_input)
    idle = GlobalResponsePlanChromosome(problem.resource_ids, (None,))

    fixed = enforce_hard_locks(problem, idle)

    assert fixed.genes == idle.genes


def test_enforce_hard_locks_corrects_a_resource_pointing_at_an_infeasible_slot():
    """Defensive: a gene value outside the restricted feasible set (should
    never arise from a legitimate operator, but defended anyway) is treated
    the same as idle."""
    demand_a = make_demand(1, minimum_resources=1, desired_resources=1)
    demand_b = make_demand(2, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10), make_target(2, 20)),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0), make_route("R1", 2, 20, eta_seconds=5.0)),
        incident_demands=(demand_a, demand_b),
        current_assignments=(make_assignment("R1", 1, 10),),
    )
    problem = build_global_optimization_problem(global_input)
    illegal = GlobalResponsePlanChromosome(problem.resource_ids, ("20#0",))

    fixed = enforce_hard_locks(problem, illegal)

    assert dict(zip(fixed.resource_ids, fixed.genes))["R1"] == "10#0"
