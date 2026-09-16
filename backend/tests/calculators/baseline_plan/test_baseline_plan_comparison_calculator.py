"""Tests for Task 3: pure optimized-vs-baseline plan comparison.

Optimized-side fixtures here are contract-compatible test doubles only
(Company 2's real `ResponsePlan`/GA/scorer may not be merged yet -- see
`baseline_plan_comparison_calculator.py`). They are not a fake GA, fake
optimization search, or fake scoring implementation: they simply provide
already-evaluated metrics for the comparison calculator to reuse.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.calculators.baseline_plan import (
    BaselinePlanComparisonCalculator,
    BaselinePlanResult,
    OptimizedPlanEvaluation,
    PlanComparison,
)


@dataclass(frozen=True)
class FakePlanScoreBreakdown:
    """Test-only stand-in shaped like the frozen `PlanScoreBreakdown` contract."""

    total_score: float
    coverage_score: float
    average_eta_seconds: float | None
    covered_target_count: int
    total_target_count: int
    uncovered_target_ids: tuple[int, ...]


def make_optimized(**overrides) -> OptimizedPlanEvaluation:
    defaults = dict(
        optimized_plan_id=1,
        fire_event_id=1,
        route_planning_run_id=1,
        response_target_set_id=1,
        score=FakePlanScoreBreakdown(
            total_score=850.0,
            coverage_score=90.0,
            average_eta_seconds=300.0,
            covered_target_count=3,
            total_target_count=3,
            uncovered_target_ids=(),
        ),
    )
    defaults.update(overrides)
    return OptimizedPlanEvaluation(**defaults)


def make_baseline(**overrides) -> BaselinePlanResult:
    defaults = dict(
        route_planning_run_id=1,
        response_target_set_id=1,
        actions=(),
        score=FakePlanScoreBreakdown(
            total_score=700.0,
            coverage_score=80.0,
            average_eta_seconds=340.0,
            covered_target_count=2,
            total_target_count=3,
            uncovered_target_ids=(103,),
        ),
    )
    defaults.update(overrides)
    return BaselinePlanResult(**defaults)


def compare(*, optimized, baseline) -> PlanComparison:
    return BaselinePlanComparisonCalculator().compare(optimized=optimized, baseline=baseline)


# ---------------------------------------------------------------------------
# 1 & 2. score_difference / improvement_percentage -- positive
# ---------------------------------------------------------------------------


def test_score_difference_positive():
    optimized = make_optimized(score=FakePlanScoreBreakdown(850.0, 90.0, 300.0, 3, 3, ()))
    baseline = make_baseline(score=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 3, (103,)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.score_difference == 150.0


def test_improvement_percentage_positive():
    optimized = make_optimized(score=FakePlanScoreBreakdown(850.0, 90.0, 300.0, 3, 3, ()))
    baseline = make_baseline(score=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 3, (103,)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.improvement_percentage == pytest.approx(((850.0 - 700.0) / 700.0) * 100)


# ---------------------------------------------------------------------------
# 3. Non-improving optimized plan -- must not be altered
# ---------------------------------------------------------------------------


def test_non_improving_optimized_plan_is_preserved_accurately():
    optimized = make_optimized(score=FakePlanScoreBreakdown(650.0, 70.0, 400.0, 2, 3, (103,)))
    baseline = make_baseline(score=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 3, (103,)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.score_difference == -50.0
    assert result.improvement_percentage < 0
    assert result.improvement_percentage == pytest.approx(((650.0 - 700.0) / 700.0) * 100)


# ---------------------------------------------------------------------------
# 4. Equal scores
# ---------------------------------------------------------------------------


def test_equal_scores_produce_zero_difference_and_zero_improvement():
    optimized = make_optimized(score=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 3, (103,)))
    baseline = make_baseline(score=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 3, (103,)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.score_difference == 0.0
    assert result.improvement_percentage == 0.0


# ---------------------------------------------------------------------------
# 5 & 6. baseline_score == 0
# ---------------------------------------------------------------------------


def test_baseline_score_zero_yields_none_improvement_percentage():
    optimized = make_optimized(score=FakePlanScoreBreakdown(500.0, 50.0, 300.0, 1, 3, (102, 103)))
    baseline = make_baseline(score=FakePlanScoreBreakdown(0.0, 0.0, None, 0, 3, (101, 102, 103)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.improvement_percentage is None
    assert result.score_difference == 500.0


def test_zero_baseline_and_zero_optimized():
    optimized = make_optimized(score=FakePlanScoreBreakdown(0.0, 0.0, None, 0, 3, (101, 102, 103)))
    baseline = make_baseline(score=FakePlanScoreBreakdown(0.0, 0.0, None, 0, 3, (101, 102, 103)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.score_difference == 0.0
    assert result.improvement_percentage is None


# ---------------------------------------------------------------------------
# 7 & 8. Same planning snapshot required
# ---------------------------------------------------------------------------


def test_differing_route_planning_run_id_fails_explicitly():
    optimized = make_optimized(route_planning_run_id=1)
    baseline = make_baseline(route_planning_run_id=2)

    with pytest.raises(ValueError):
        compare(optimized=optimized, baseline=baseline)


def test_differing_response_target_set_id_fails_explicitly():
    optimized = make_optimized(response_target_set_id=1)
    baseline = make_baseline(response_target_set_id=2)

    with pytest.raises(ValueError):
        compare(optimized=optimized, baseline=baseline)


# ---------------------------------------------------------------------------
# 9. Exact identifiers preserved
# ---------------------------------------------------------------------------


def test_exact_identifiers_preserved_without_substitution():
    optimized = make_optimized(
        optimized_plan_id=42,
        fire_event_id=7,
        route_planning_run_id=13,
        response_target_set_id=99,
    )
    baseline = make_baseline(route_planning_run_id=13, response_target_set_id=99)

    result = compare(optimized=optimized, baseline=baseline)

    assert result.fire_event_id == 7
    assert result.optimized_plan_id == 42
    assert result.route_planning_run_id == 13
    assert result.response_target_set_id == 99


# ---------------------------------------------------------------------------
# 10 & 11. Scores / coverage are reused, not recalculated
# ---------------------------------------------------------------------------


def test_scores_are_copied_directly_from_source_data():
    optimized = make_optimized(score=FakePlanScoreBreakdown(850.5, 90.0, 300.0, 3, 3, ()))
    baseline = make_baseline(score=FakePlanScoreBreakdown(700.25, 80.0, 340.0, 2, 3, (103,)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.optimized_score == optimized.score.total_score == 850.5
    assert result.baseline_score == baseline.score.total_score == 700.25


def test_coverage_scores_are_copied_directly_without_local_recalculation():
    optimized = make_optimized(score=FakePlanScoreBreakdown(850.0, 91.7, 300.0, 3, 3, ()))
    baseline = make_baseline(score=FakePlanScoreBreakdown(700.0, 63.2, 340.0, 2, 3, (103,)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.optimized_coverage_score == optimized.score.coverage_score == 91.7
    assert result.baseline_coverage_score == baseline.score.coverage_score == 63.2


# ---------------------------------------------------------------------------
# 12 & 13. Average ETA reused, None preserved
# ---------------------------------------------------------------------------


def test_average_eta_values_are_copied_unchanged():
    optimized = make_optimized(score=FakePlanScoreBreakdown(850.0, 90.0, 275.5, 3, 3, ()))
    baseline = make_baseline(score=FakePlanScoreBreakdown(700.0, 80.0, 333.25, 2, 3, (103,)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.optimized_average_eta_seconds == 275.5
    assert result.baseline_average_eta_seconds == 333.25


def test_average_eta_none_is_preserved_on_either_side():
    optimized = make_optimized(score=FakePlanScoreBreakdown(0.0, 0.0, None, 0, 3, (101, 102, 103)))
    baseline = make_baseline(score=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 3, (103,)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.optimized_average_eta_seconds is None
    assert result.baseline_average_eta_seconds == 340.0


# ---------------------------------------------------------------------------
# 14. No score normalization/rounding
# ---------------------------------------------------------------------------


def test_non_round_score_values_are_preserved_exactly():
    optimized = make_optimized(score=FakePlanScoreBreakdown(853.14159, 90.6667, 301.333, 3, 3, ()))
    baseline = make_baseline(score=FakePlanScoreBreakdown(701.2345, 80.111, 341.777, 2, 3, (103,)))

    result = compare(optimized=optimized, baseline=baseline)

    assert result.optimized_score == 853.14159
    assert result.baseline_score == 701.2345
    assert result.score_difference == 853.14159 - 701.2345


# ---------------------------------------------------------------------------
# 15. Determinism
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_identical_comparison():
    optimized = make_optimized()
    baseline = make_baseline()

    first = compare(optimized=optimized, baseline=baseline)
    second = compare(optimized=optimized, baseline=baseline)

    assert first == second


# ---------------------------------------------------------------------------
# 16 & 17. No mutation
# ---------------------------------------------------------------------------


def test_baseline_result_is_not_mutated():
    optimized = make_optimized()
    baseline = make_baseline()
    baseline_before = baseline

    compare(optimized=optimized, baseline=baseline)

    assert baseline == baseline_before


def test_optimized_input_is_not_mutated():
    optimized = make_optimized()
    baseline = make_baseline()
    optimized_before = optimized

    compare(optimized=optimized, baseline=baseline)

    assert optimized == optimized_before


# ---------------------------------------------------------------------------
# 18. No dependency on action order
# ---------------------------------------------------------------------------


def test_comparison_metrics_do_not_depend_on_action_order():
    from src.calculators.baseline_plan import BaselineAssignment

    optimized = make_optimized()
    shared_score = FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 3, (103,))
    actions_forward = (
        BaselineAssignment(resource_id="R1", response_target_id=101, route_result_id=1),
        BaselineAssignment(resource_id="R2", response_target_id=102, route_result_id=2),
    )
    baseline_forward = make_baseline(score=shared_score, actions=actions_forward)
    baseline_reversed = make_baseline(score=shared_score, actions=tuple(reversed(actions_forward)))

    result_forward = compare(optimized=optimized, baseline=baseline_forward)
    result_reversed = compare(optimized=optimized, baseline=baseline_reversed)

    assert result_forward == result_reversed


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_non_positive_optimized_plan_id_rejected():
    with pytest.raises(ValueError):
        make_optimized(optimized_plan_id=0)


def test_non_positive_fire_event_id_rejected():
    with pytest.raises(ValueError):
        make_optimized(fire_event_id=0)
