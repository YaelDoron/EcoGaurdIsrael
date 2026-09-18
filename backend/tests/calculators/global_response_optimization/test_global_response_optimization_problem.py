"""Tests for GlobalResponseOptimizationProblem (Stage 4, Task 8)."""
from __future__ import annotations

import pytest

from src.calculators.global_response_optimization.global_response_optimization_problem import (
    build_global_optimization_problem,
)
from src.models.resource_status import ResourceStatus
from src.models.response_target_type import ResponseTargetType
from tests.calculators.global_response_optimization.helpers import make_input, make_resource, make_route, make_target


def test_problem_indexes_only_assignable_resources():
    targets = (make_target(1, 10),)
    resources = (
        make_resource("R1", operational_status=ResourceStatus.AVAILABLE),
        make_resource("R2", operational_status=ResourceStatus.UNAVAILABLE),
    )
    routes = (make_route("R1", 1, 10, eta_seconds=60), make_route("R2", 1, 10, eta_seconds=60))
    global_input = make_input(active_fire_event_ids=(1,), targets=targets, resources=resources, routes=routes)

    problem = build_global_optimization_problem(global_input)

    assert problem.resource_ids == ("R1",)
    assert "R2" not in problem.resources_by_id


def test_problem_resource_ordering_is_deterministic():
    targets = (make_target(1, 10),)
    resources = (make_resource("R2"), make_resource("R1"), make_resource("R3"))
    global_input = make_input(active_fire_event_ids=(1,), targets=targets, resources=resources, routes=())

    problem = build_global_optimization_problem(global_input)

    assert problem.resource_ids == ("R1", "R2", "R3")


def test_problem_builds_one_slot_per_target():
    targets = (
        make_target(1, 10, target_type=ResponseTargetType.ACTIVE_FIRE),
        make_target(1, 11, target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30),
    )
    global_input = make_input(active_fire_event_ids=(1,), targets=targets, resources=(), routes=())

    problem = build_global_optimization_problem(global_input)

    assert len(problem.slots) == 2
    assert set(problem.slots_by_id) == {slot.slot_id for slot in problem.slots}


def test_feasible_slot_ids_by_resource_reflects_route_matrix():
    targets = (make_target(1, 10), make_target(2, 20))
    resources = (make_resource("R1"),)
    routes = (make_route("R1", 1, 10, eta_seconds=60),)  # only R1->target10 is feasible
    global_input = make_input(active_fire_event_ids=(1, 2), targets=targets, resources=resources, routes=routes)

    problem = build_global_optimization_problem(global_input)

    slot_for_10 = next(s.slot_id for s in problem.slots if s.response_target_id == 10)
    slot_for_20 = next(s.slot_id for s in problem.slots if s.response_target_id == 20)
    assert problem.feasible_slot_ids_by_resource["R1"] == (slot_for_10,)
    assert slot_for_20 not in problem.feasible_slot_ids_by_resource["R1"]


def test_route_for_returns_none_for_infeasible_pair():
    targets = (make_target(1, 10),)
    resources = (make_resource("R1"),)
    global_input = make_input(active_fire_event_ids=(1,), targets=targets, resources=resources, routes=())

    problem = build_global_optimization_problem(global_input)

    slot_id = problem.slots[0].slot_id
    assert problem.route_for("R1", slot_id) is None


def test_route_for_returns_the_real_route_option():
    targets = (make_target(1, 10),)
    resources = (make_resource("R1"),)
    routes = (make_route("R1", 1, 10, eta_seconds=42),)
    global_input = make_input(active_fire_event_ids=(1,), targets=targets, resources=resources, routes=routes)

    problem = build_global_optimization_problem(global_input)

    slot_id = problem.slots[0].slot_id
    route = problem.route_for("R1", slot_id)
    assert route is not None
    assert route.eta_seconds == 42


def test_rejects_non_global_planning_input():
    with pytest.raises(ValueError):
        build_global_optimization_problem("not-an-input")
