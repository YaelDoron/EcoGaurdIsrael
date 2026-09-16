"""Tests for the pure deterministic greedy baseline allocation calculator."""
from __future__ import annotations

import random

import pytest

from src.calculators.baseline_plan import (
    BaselinePlanAllocation,
    BaselinePlanCalculator,
    RouteCandidate,
    TargetOrder,
)
from src.models import ResponseTargetType


def make_candidate(**overrides) -> RouteCandidate:
    defaults = dict(
        route_result_id=1,
        resource_id="R1",
        response_target_id=1,
        status="reachable",
        travel_time_seconds=100.0,
        distance_meters=1000.0,
    )
    defaults.update(overrides)
    return RouteCandidate(**defaults)


def make_target(**overrides) -> TargetOrder:
    defaults = dict(
        response_target_id=1,
        target_order=0,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        priority_score=100.0,
    )
    defaults.update(overrides)
    return TargetOrder(**defaults)


def allocate(*, targets, route_candidates) -> BaselinePlanAllocation:
    return BaselinePlanCalculator().allocate(targets=targets, route_candidates=route_candidates)


# ---------------------------------------------------------------------------
# Lowest ETA selection
# ---------------------------------------------------------------------------


def test_lowest_travel_time_resource_is_selected():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=300.0),
        make_candidate(route_result_id=2, resource_id="R2", travel_time_seconds=100.0),
        make_candidate(route_result_id=3, resource_id="R3", travel_time_seconds=200.0),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert len(result.assignments) == 1
    assert result.assignments[0].resource_id == "R2"
    assert result.assignments[0].route_result_id == 2
    assert result.uncovered_response_target_ids == ()


# ---------------------------------------------------------------------------
# Target priority/order
# ---------------------------------------------------------------------------


def test_targets_processed_in_persisted_target_order_and_earlier_use_blocks_later():
    # Resource R1 is the fastest for both target 1 and target 2. Target 1
    # has the earlier target_order, so it claims resource R1 first.
    targets = (
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
    )
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, travel_time_seconds=50.0),
        make_candidate(route_result_id=2, resource_id="R2", response_target_id=1, travel_time_seconds=999.0),
        make_candidate(route_result_id=3, resource_id="R1", response_target_id=2, travel_time_seconds=10.0),
        make_candidate(route_result_id=4, resource_id="R2", response_target_id=2, travel_time_seconds=20.0),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    by_target = {a.response_target_id: a for a in result.assignments}
    assert by_target[1].resource_id == "R1"
    assert by_target[2].resource_id == "R2"


def test_target_processing_is_deterministic_from_target_order_not_input_order():
    # target_order: 102 -> 1, 101 -> 2, 103 -> 3. Resource "R1" is the
    # fastest candidate for both 101 and 102, so which one claims it
    # depends only on target_order (102 first), never on the incidental
    # order `targets` is supplied in.
    targets_by_order = (
        make_target(response_target_id=101, target_order=2),
        make_target(response_target_id=102, target_order=1),
        make_target(response_target_id=103, target_order=3),
    )
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=101, travel_time_seconds=50.0),
        make_candidate(route_result_id=2, resource_id="R2", response_target_id=101, travel_time_seconds=999.0),
        make_candidate(route_result_id=3, resource_id="R1", response_target_id=102, travel_time_seconds=10.0),
        make_candidate(route_result_id=4, resource_id="R2", response_target_id=102, travel_time_seconds=20.0),
        make_candidate(route_result_id=5, resource_id="R3", response_target_id=103, travel_time_seconds=5.0),
    )
    expected_assignment_order = (102, 101, 103)
    expected_resources = {102: "R1", 101: "R2", 103: "R3"}

    incidental_orderings = [
        (targets_by_order[0], targets_by_order[1], targets_by_order[2]),  # 101, 102, 103
        (targets_by_order[1], targets_by_order[0], targets_by_order[2]),  # 102, 101, 103
        (targets_by_order[2], targets_by_order[0], targets_by_order[1]),  # 103, 101, 102
    ]

    for ordering in incidental_orderings:
        result = allocate(targets=ordering, route_candidates=candidates)

        assert tuple(a.response_target_id for a in result.assignments) == expected_assignment_order
        assert {a.response_target_id: a.resource_id for a in result.assignments} == expected_resources
        assert result.uncovered_response_target_ids == ()


# ---------------------------------------------------------------------------
# Resource uniqueness
# ---------------------------------------------------------------------------


def test_same_resource_never_assigned_to_two_targets():
    targets = (
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
    )
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, travel_time_seconds=50.0),
        make_candidate(route_result_id=2, resource_id="R1", response_target_id=2, travel_time_seconds=10.0),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assigned_resource_ids = [a.resource_id for a in result.assignments]
    assert len(assigned_resource_ids) == len(set(assigned_resource_ids))
    assert result.uncovered_response_target_ids == (2,)


# ---------------------------------------------------------------------------
# reachable / unreachable / unmappable
# ---------------------------------------------------------------------------


def test_unreachable_route_is_never_selected():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=1.0, status="unreachable"),
        make_candidate(route_result_id=2, resource_id="R2", travel_time_seconds=500.0, status="reachable"),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert len(result.assignments) == 1
    assert result.assignments[0].resource_id == "R2"


def test_unmappable_route_is_never_selected():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=1.0, status="unmappable"),
        make_candidate(route_result_id=2, resource_id="R2", travel_time_seconds=500.0, status="reachable"),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert len(result.assignments) == 1
    assert result.assignments[0].resource_id == "R2"


# ---------------------------------------------------------------------------
# Uncovered targets
# ---------------------------------------------------------------------------


def test_target_with_no_eligible_resource_is_uncovered():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", status="unreachable", travel_time_seconds=None),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert result.assignments == ()
    assert result.uncovered_response_target_ids == (1,)


# ---------------------------------------------------------------------------
# Resource/target count mismatches
# ---------------------------------------------------------------------------


def test_fewer_resources_than_targets_produces_valid_partial_coverage():
    targets = (
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
        make_target(response_target_id=3, target_order=3),
    )
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, travel_time_seconds=50.0),
        make_candidate(route_result_id=2, resource_id="R1", response_target_id=2, travel_time_seconds=10.0),
        make_candidate(route_result_id=3, resource_id="R1", response_target_id=3, travel_time_seconds=5.0),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert len(result.assignments) == 1
    assert result.assignments[0].response_target_id == 1
    assert result.uncovered_response_target_ids == (2, 3)


def test_more_resources_than_targets_produces_no_duplicate_assignments():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, travel_time_seconds=50.0),
        make_candidate(route_result_id=2, resource_id="R2", response_target_id=1, travel_time_seconds=10.0),
        make_candidate(route_result_id=3, resource_id="R3", response_target_id=1, travel_time_seconds=5.0),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert len(result.assignments) == 1
    assert result.assignments[0].resource_id == "R3"
    assert result.uncovered_response_target_ids == ()


# ---------------------------------------------------------------------------
# Deterministic tie-breaking
# ---------------------------------------------------------------------------


def test_equal_eta_tie_break_is_deterministic_by_lowest_resource_id():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=1, resource_id="R5", travel_time_seconds=100.0),
        make_candidate(route_result_id=2, resource_id="R2", travel_time_seconds=100.0),
        make_candidate(route_result_id=3, resource_id="R9", travel_time_seconds=100.0),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert result.assignments[0].resource_id == "R2"
    assert result.assignments[0].route_result_id == 2


def test_equal_eta_tie_break_uses_lexicographic_string_ordering():
    # "truck-001" < "truck-002" lexicographically, matching the frozen
    # resource_id: str contract's tie-break rule.
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=1, resource_id="truck-002", travel_time_seconds=100.0),
        make_candidate(route_result_id=2, resource_id="truck-001", travel_time_seconds=100.0),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert result.assignments[0].resource_id == "truck-001"


def test_equal_eta_tie_break_repeated_runs_pick_same_resource():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=1, resource_id="engine-b", travel_time_seconds=100.0),
        make_candidate(route_result_id=2, resource_id="engine-a", travel_time_seconds=100.0),
    )

    first = allocate(targets=targets, route_candidates=candidates)
    second = allocate(targets=targets, route_candidates=candidates)

    assert first == second
    assert first.assignments[0].resource_id == "engine-a"


# ---------------------------------------------------------------------------
# Determinism under incidental reordering
# ---------------------------------------------------------------------------


def test_result_is_identical_regardless_of_route_candidate_input_order():
    targets = (
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
    )
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, travel_time_seconds=50.0),
        make_candidate(route_result_id=2, resource_id="R2", response_target_id=1, travel_time_seconds=10.0),
        make_candidate(route_result_id=3, resource_id="R1", response_target_id=2, travel_time_seconds=5.0),
        make_candidate(route_result_id=4, resource_id="R3", response_target_id=2, travel_time_seconds=5.0),
    )

    forward = allocate(targets=targets, route_candidates=candidates)
    reversed_order = allocate(targets=targets, route_candidates=tuple(reversed(candidates)))
    shuffled = list(candidates)
    random.Random(42).shuffle(shuffled)
    shuffled_result = allocate(targets=targets, route_candidates=tuple(shuffled))

    assert forward == reversed_order == shuffled_result


# ---------------------------------------------------------------------------
# RouteResult traceability
# ---------------------------------------------------------------------------


def test_every_assignment_traces_to_the_exact_route_result_id_selected():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=42, resource_id="R1", response_target_id=1, travel_time_seconds=50.0),
        make_candidate(route_result_id=99, resource_id="R2", response_target_id=1, travel_time_seconds=999.0),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert result.assignments[0].route_result_id == 42


# ---------------------------------------------------------------------------
# Empty / no-eligible-route edge cases
# ---------------------------------------------------------------------------


def test_no_eligible_routes_leaves_all_targets_uncovered_without_crashing():
    targets = (
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
    )
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, status="unreachable"),
        make_candidate(route_result_id=2, resource_id="R2", response_target_id=2, status="unmappable"),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert result.assignments == ()
    assert result.uncovered_response_target_ids == (1, 2)


def test_empty_targets_returns_valid_empty_result():
    result = allocate(targets=(), route_candidates=())

    assert result.assignments == ()
    assert result.uncovered_response_target_ids == ()


def test_empty_routes_leaves_targets_uncovered_without_crashing():
    targets = (
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
    )

    result = allocate(targets=targets, route_candidates=())

    assert result.assignments == ()
    assert result.uncovered_response_target_ids == (1, 2)


# ---------------------------------------------------------------------------
# Later target processing continues after an uncovered target
# ---------------------------------------------------------------------------


def test_later_target_still_covered_after_earlier_target_is_uncovered():
    targets = (
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
    )
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, status="unreachable"),
        make_candidate(route_result_id=2, resource_id="R1", response_target_id=2, travel_time_seconds=30.0),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert result.uncovered_response_target_ids == (1,)
    assert len(result.assignments) == 1
    assert result.assignments[0].response_target_id == 2
    assert result.assignments[0].resource_id == "R1"


# ---------------------------------------------------------------------------
# Reachable candidate without a valid travel time is not eligible
# ---------------------------------------------------------------------------


def test_reachable_candidate_missing_travel_time_is_not_eligible():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", status="reachable", travel_time_seconds=None),
        make_candidate(route_result_id=2, resource_id="R2", status="reachable", travel_time_seconds=75.0),
    )

    result = allocate(targets=targets, route_candidates=candidates)

    assert len(result.assignments) == 1
    assert result.assignments[0].resource_id == "R2"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_status_rejected():
    with pytest.raises(ValueError):
        make_candidate(status="not_a_real_status")


def test_uppercase_status_values_are_rejected():
    # The frozen RouteStatus contract's persisted values are lowercase;
    # this guards against silently accepting the old uppercase shape.
    with pytest.raises(ValueError):
        make_candidate(status="REACHABLE")


def test_non_positive_route_result_id_rejected():
    with pytest.raises(ValueError):
        make_candidate(route_result_id=0)


def test_non_string_resource_id_rejected():
    # The frozen contract is resource_id: str; int resource ids must be
    # rejected at the boundary rather than silently accepted.
    with pytest.raises(ValueError):
        make_candidate(resource_id=1)


def test_empty_resource_id_rejected():
    with pytest.raises(ValueError):
        make_candidate(resource_id="")


def test_negative_target_order_rejected():
    with pytest.raises(ValueError):
        make_target(target_order=-1)


def test_non_positive_response_target_id_rejected_in_target_order():
    with pytest.raises(ValueError):
        make_target(response_target_id=0)
