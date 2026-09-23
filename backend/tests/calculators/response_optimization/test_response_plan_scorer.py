"""Tests for pure response-plan scoring."""
from __future__ import annotations

import math

import pytest

from src.calculators.response_optimization import (
    ETA_REFERENCE_SECONDS,
    ResponseOptimizationConfig,
    ResponsePlanScorer,
    ResponsePlanScoringError,
    eta_factor,
)
from src.models import (
    OptimizationResource,
    OptimizationRouteOption,
    OptimizationTarget,
    ResponseAction,
    ResponseOptimizationInput,
    ResponsePlanStatus,
    ResponseTargetType,
)


def target(target_id: int, order: int, priority: float) -> OptimizationTarget:
    return OptimizationTarget(
        response_target_id=target_id,
        target_order=order,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        priority_score=priority,
    )


def resource(resource_id: str) -> OptimizationResource:
    return OptimizationResource(resource_id)


def route(
    route_id: int,
    resource_id: str,
    target_id: int,
    eta: float | None,
    *,
    reachable: bool = True,
) -> OptimizationRouteOption:
    return OptimizationRouteOption(
        route_result_id=route_id,
        resource_id=resource_id,
        response_target_id=target_id,
        is_reachable=reachable,
        travel_time_seconds=eta,
        distance_meters=1000.0 if reachable else None,
    )


def action(resource_id: str, target_id: int, route_id: int) -> ResponseAction:
    return ResponseAction(resource_id=resource_id, response_target_id=target_id, route_result_id=route_id)


def optimization_input(
    *,
    targets=(target(10, 0, 100.0),),
    resources=(resource("R1"),),
    route_options=(route(100, "R1", 10, 0.0),),
) -> ResponseOptimizationInput:
    return ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=targets,
        resources=resources,
        route_options=route_options,
    )


def score(input_data: ResponseOptimizationInput, actions: tuple[ResponseAction, ...]):
    return ResponsePlanScorer().evaluate(input_data, actions)


def test_eta_factor_reference_points_and_monotonicity():
    # Quadratic in (eta / eta_reference_seconds): eta_factor(0) == 1 and
    # eta_factor(eta_reference_seconds) == 0.5 are unchanged, but it falls
    # off much faster beyond the reference point than a linear factor would
    # (1/(1+2**2) == 0.2, not the old linear formula's 1/3) - this is what
    # heavily penalizes travel time/distance so the closest available
    # resources are favored over farther ones.
    assert eta_factor(0.0) == pytest.approx(1.0)
    assert eta_factor(ETA_REFERENCE_SECONDS) == pytest.approx(0.5)
    assert eta_factor(1800.0) == pytest.approx(0.2)
    assert eta_factor(300.0) > eta_factor(900.0) > eta_factor(1800.0)


@pytest.mark.parametrize(
    ("eta", "expected_raw", "expected_score", "expected_average"),
    [(0.0, 100.0, 100.0, 0.0), (900.0, 50.0, 50.0, 900.0)],
)
def test_single_target_scoring_separates_coverage_from_eta(eta, expected_raw, expected_score, expected_average):
    input_data = optimization_input(route_options=(route(100, "R1", 10, eta),))

    result = score(input_data, (action("R1", 10, 100),))

    assert result.raw_fitness == pytest.approx(expected_raw)
    assert result.total_score == pytest.approx(expected_score)
    assert result.coverage_score == pytest.approx(100.0)
    assert result.average_eta_seconds == pytest.approx(expected_average)
    assert result.status is ResponsePlanStatus.COMPLETE


def test_multi_target_scoring_uses_priority_and_eta_utility():
    input_data = optimization_input(
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R1"), resource("R2")),
        route_options=(route(100, "R1", 10, 900.0), route(200, "R2", 20, 0.0)),
    )

    result = score(input_data, (action("R1", 10, 100), action("R2", 20, 200)))

    assert result.raw_fitness == pytest.approx(100.0)
    assert result.total_priority == pytest.approx(150.0)
    assert result.total_score == pytest.approx(100.0 * 100.0 / 150.0)
    assert result.coverage_score == pytest.approx(100.0)
    assert result.average_eta_seconds == pytest.approx(450.0)
    assert result.status is ResponsePlanStatus.COMPLETE


def test_partial_coverage_priority_weighted_scores_and_ordered_uncovered_targets():
    input_data = optimization_input(
        targets=(target(20, 1, 50.0), target(10, 0, 100.0)),
        resources=(resource("R1"),),
        route_options=(route(100, "R1", 10, 0.0),),
    )

    result = score(input_data, (action("R1", 10, 100),))

    assert result.raw_fitness == pytest.approx(100.0)
    assert result.total_score == pytest.approx(100.0 * 100.0 / 150.0)
    assert result.coverage_score == pytest.approx(100.0 * 100.0 / 150.0)
    assert result.average_eta_seconds == pytest.approx(0.0)
    assert result.status is ResponsePlanStatus.PARTIAL
    assert result.uncovered_target_ids == (20,)


def test_empty_plan_is_valid_no_feasible_assignments():
    input_data = optimization_input(
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R1"),),
        route_options=(route(100, "R1", 10, 0.0),),
    )

    result = score(input_data, ())

    assert result.total_score == pytest.approx(0.0)
    assert result.raw_fitness == pytest.approx(0.0)
    assert result.coverage_score == pytest.approx(0.0)
    assert result.average_eta_seconds is None
    assert result.covered_target_count == 0
    assert result.uncovered_target_ids == (10, 20)
    assert result.status is ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS


def test_priority_influence_higher_priority_target_scores_higher_with_equal_eta():
    high_input = optimization_input(
        targets=(target(10, 0, 100.0), target(20, 1, 10.0)),
        route_options=(route(100, "R1", 10, 300.0),),
    )
    low_input = optimization_input(
        targets=(target(10, 0, 100.0), target(20, 1, 10.0)),
        route_options=(route(200, "R1", 20, 300.0),),
    )

    high = score(high_input, (action("R1", 10, 100),))
    low = score(low_input, (action("R1", 20, 200),))

    assert high.total_score > low.total_score


def test_eta_influence_shorter_eta_scores_higher_for_same_target_priority():
    short_input = optimization_input(route_options=(route(100, "R1", 10, 300.0),))
    long_input = optimization_input(route_options=(route(200, "R1", 10, 1200.0),))

    short = score(short_input, (action("R1", 10, 100),))
    long = score(long_input, (action("R1", 10, 200),))

    assert short.total_score > long.total_score


def test_quadratic_eta_penalty_prefers_closer_lower_priority_target_over_farther_higher_priority_target():
    """Regression test for the dispatch-priority bug: a target with 2x the
    priority but 4x the ETA must no longer outscore a closer, lower-priority
    target - this is what makes the closest available resources get
    exhausted before farther ones are considered.

    Under the old linear-in-eta formula this pair would have scored
    100 * eta_factor(300) == 75.0 for the near/low-priority target vs
    200 * eta_factor(1200) == 85.71 for the far/high-priority one - i.e. the
    far target would have won, favoring distance-blind priority chasing. The
    quadratic formula flips it to 90.0 vs 72.0.
    """
    near_low_priority = optimization_input(
        targets=(target(10, 0, 100.0),),
        route_options=(route(100, "R1", 10, 300.0),),
    )
    far_high_priority = optimization_input(
        targets=(target(20, 0, 200.0),),
        route_options=(route(200, "R1", 20, 1200.0),),
    )

    near_result = score(near_low_priority, (action("R1", 10, 100),))
    far_result = score(far_high_priority, (action("R1", 20, 200),))

    assert near_result.raw_fitness == pytest.approx(90.0)
    assert far_result.raw_fitness == pytest.approx(72.0)
    assert near_result.raw_fitness > far_result.raw_fitness


def test_resource_conflict_rejected():
    input_data = optimization_input(
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R1"),),
        route_options=(route(100, "R1", 10, 0.0), route(200, "R1", 20, 0.0)),
    )

    with pytest.raises(ResponsePlanScoringError):
        score(input_data, (action("R1", 10, 100), action("R1", 20, 200)))


def test_target_conflict_rejected():
    input_data = optimization_input(
        targets=(target(10, 0, 100.0),),
        resources=(resource("R1"), resource("R2")),
        route_options=(route(100, "R1", 10, 0.0), route(200, "R2", 10, 0.0)),
    )

    with pytest.raises(ResponsePlanScoringError):
        score(input_data, (action("R1", 10, 100), action("R2", 10, 200)))


@pytest.mark.parametrize(
    "bad_action",
    [
        action("R2", 10, 100),
        action("R1", 20, 100),
    ],
)
def test_route_mismatch_rejected(bad_action):
    input_data = optimization_input(
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R1"), resource("R2")),
        route_options=(route(100, "R1", 10, 0.0), route(200, "R2", 20, 0.0)),
    )

    with pytest.raises(ResponsePlanScoringError):
        score(input_data, (bad_action,))


def test_unreachable_route_rejected():
    input_data = optimization_input(route_options=(route(100, "R1", 10, None, reachable=False),))

    with pytest.raises(ResponsePlanScoringError):
        score(input_data, (action("R1", 10, 100),))


@pytest.mark.parametrize(
    "bad_action",
    [
        action("MISSING", 10, 100),
        action("R1", 999, 100),
        action("R1", 10, 999),
    ],
)
def test_unknown_ids_rejected(bad_action):
    input_data = optimization_input()

    with pytest.raises(ResponsePlanScoringError):
        score(input_data, (bad_action,))


def test_determinism_for_equivalent_action_order():
    input_data = optimization_input(
        targets=(target(20, 1, 50.0), target(10, 0, 100.0), target(30, 2, 25.0)),
        resources=(resource("R2"), resource("R1")),
        route_options=(route(200, "R2", 20, 600.0), route(100, "R1", 10, 300.0)),
    )

    first = score(input_data, (action("R2", 20, 200), action("R1", 10, 100)))
    second = score(input_data, (action("R1", 10, 100), action("R2", 20, 200)))

    assert first == second
    assert first.uncovered_target_ids == (30,)


def test_zero_total_priority_no_division_by_zero_or_non_finite_results():
    input_data = optimization_input(
        targets=(target(10, 0, 0.0),),
        route_options=(route(100, "R1", 10, 0.0),),
    )

    result = score(input_data, (action("R1", 10, 100),))

    assert result.raw_fitness == pytest.approx(0.0)
    assert result.total_score == pytest.approx(0.0)
    assert result.coverage_score == pytest.approx(0.0)
    assert math.isfinite(result.total_score)
    assert math.isfinite(result.coverage_score)
    assert result.status is ResponsePlanStatus.COMPLETE


def test_full_coverage_clamps_floating_point_drift_instead_of_raising():
    """Regression test: raw_fitness/covered_priority sum over actions sorted
    by resource_id (see _normalize_actions), while total_priority sums over
    targets in input order - a different addition order for the same
    underlying values. Float addition is not associative, so a fully
    covering plan can land a few ULPs above the mathematical ceiling of 100
    (e.g. 100.00000000000003) instead of exactly 100.0. Before the clamp in
    ResponsePlanScorer.evaluate, this raised inside PlanScoreBreakdown's
    range validation and crashed the whole GA run for an otherwise-valid,
    fully-covering candidate - this exact priority/eta data set (found via a
    realistic-scale GA run) reproduced that crash pre-fix.
    """
    targets = (
        target(100, 0, 85.99796663725434),
        target(101, 1, 78.21589626462722),
        target(102, 2, 47.85144227477605),
        target(103, 3, 33.302507526366696),
        target(104, 4, 56.01472492317477),
        target(105, 5, 46.444072370537285),
    )
    resources = tuple(resource(f"R{i}") for i in range(8))
    route_options = (
        route(1, "R0", 100, 131.3519435613909),
        route(2, "R6", 101, 112.4059027727733),
        route(3, "R5", 102, 2104.435478937752),
        route(4, "R2", 103, 590.4196924156784),
        route(5, "R4", 104, 1514.7041110197113),
        route(6, "R7", 105, 292.80218418690185),
    )
    input_data = optimization_input(targets=targets, resources=resources, route_options=route_options)
    actions = (
        action("R0", 100, 1),
        action("R6", 101, 2),
        action("R5", 102, 3),
        action("R2", 103, 4),
        action("R4", 104, 5),
        action("R7", 105, 6),
    )

    result = score(input_data, actions)

    assert result.coverage_score == 100.0
    assert result.total_score <= 100.0
    assert result.status is ResponsePlanStatus.COMPLETE


def test_config_eta_reference_valid_and_rejects_invalid_values():
    assert ResponseOptimizationConfig(eta_reference_seconds=1200.0).eta_reference_seconds == 1200.0

    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ResponseOptimizationConfig(eta_reference_seconds=value)
