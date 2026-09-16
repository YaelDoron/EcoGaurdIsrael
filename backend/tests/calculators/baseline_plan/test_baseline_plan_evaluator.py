"""Tests for Task 2: connecting the Task 1 baseline to shared scoring.

The scorer used here is a test-only double (see `MOCK / TEST DOUBLE RULE`
in the US 5.3 Task 2 brief): it exists purely to prove the evaluator calls
the injected scoring dependency, calls it exactly once, passes it the
exact Task 1 allocation, and preserves whatever it returns unchanged. It
is not a fitness/scoring implementation and must never be treated as one.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.calculators.baseline_plan import (
    BaselinePlanAllocation,
    BaselinePlanCalculator,
    BaselinePlanEvaluator,
    BaselinePlanResult,
    BaselinePlanScoringContext,
    RouteCandidate,
    TargetOrder,
)
from src.calculators.baseline_plan.baseline_plan_config import (
    BASELINE_PLAN_METHODOLOGY,
    BASELINE_PLAN_METHODOLOGY_VERSION,
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


class RecordingScorer:
    """Test double: records every call and returns a predetermined score."""

    def __init__(self, *, result: FakePlanScoreBreakdown) -> None:
        self._result = result
        self.call_count = 0
        self.received_contexts: list[BaselinePlanScoringContext] = []

    def evaluate(self, context: BaselinePlanScoringContext) -> FakePlanScoreBreakdown:
        self.call_count += 1
        self.received_contexts.append(context)
        return self._result


class RaisingScorer:
    """Test double that simulates the shared scorer failing."""

    def evaluate(self, context: BaselinePlanScoringContext) -> FakePlanScoreBreakdown:
        raise RuntimeError("shared scorer failed")


KNOWN_SCORE = FakePlanScoreBreakdown(
    total_score=720.0,
    coverage_score=80.0,
    average_eta_seconds=340.0,
    covered_target_count=2,
    total_target_count=3,
    uncovered_target_ids=(103,),
)


def make_candidate(**overrides) -> RouteCandidate:
    defaults = dict(
        route_result_id=1,
        resource_id="R1",
        response_target_id=1,
        status="reachable",
        travel_time_seconds=100.0,
    )
    defaults.update(overrides)
    return RouteCandidate(**defaults)


def make_target(**overrides) -> TargetOrder:
    defaults = dict(response_target_id=1, target_order=0)
    defaults.update(overrides)
    return TargetOrder(**defaults)


def evaluate(
    *,
    route_planning_run_id=1,
    response_target_set_id=1,
    targets,
    route_candidates,
    scorer,
) -> BaselinePlanResult:
    return BaselinePlanEvaluator().evaluate(
        route_planning_run_id=route_planning_run_id,
        response_target_set_id=response_target_set_id,
        targets=targets,
        route_candidates=route_candidates,
        scorer=scorer,
    )


# ---------------------------------------------------------------------------
# 1 & 2. Shared scorer is used, exactly once
# ---------------------------------------------------------------------------


def test_shared_scorer_is_used():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=50.0),)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    evaluate(targets=targets, route_candidates=candidates, scorer=scorer)

    assert scorer.call_count >= 1


def test_scorer_called_exactly_once():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=50.0),)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    evaluate(targets=targets, route_candidates=candidates, scorer=scorer)

    assert scorer.call_count == 1


# ---------------------------------------------------------------------------
# 3. Task 1 allocation is used, unchanged, in the scoring context
# ---------------------------------------------------------------------------


def test_task1_allocation_is_passed_to_scorer_unchanged():
    targets = (
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
    )
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, travel_time_seconds=50.0),
        make_candidate(route_result_id=2, resource_id="R2", response_target_id=1, travel_time_seconds=10.0),
        make_candidate(route_result_id=3, resource_id="R1", response_target_id=2, travel_time_seconds=5.0),
    )
    expected_allocation = BaselinePlanCalculator().allocate(targets=targets, route_candidates=candidates)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    evaluate(targets=targets, route_candidates=candidates, scorer=scorer)

    assert scorer.received_contexts[0].allocation == expected_allocation


# ---------------------------------------------------------------------------
# 4. Score pass-through
# ---------------------------------------------------------------------------


def test_score_is_preserved_exactly_without_recalculation():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=50.0),)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    result = evaluate(targets=targets, route_candidates=candidates, scorer=scorer)

    assert result.score is KNOWN_SCORE
    assert result.score == KNOWN_SCORE


# ---------------------------------------------------------------------------
# 5 & 6. Identifiers preserved
# ---------------------------------------------------------------------------


def test_route_planning_run_id_preserved():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=50.0),)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    result = evaluate(
        route_planning_run_id=77,
        targets=targets,
        route_candidates=candidates,
        scorer=scorer,
    )

    assert result.route_planning_run_id == 77
    assert scorer.received_contexts[0].route_planning_run_id == 77


def test_response_target_set_id_preserved():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=50.0),)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    result = evaluate(
        response_target_set_id=55,
        targets=targets,
        route_candidates=candidates,
        scorer=scorer,
    )

    assert result.response_target_set_id == 55
    assert scorer.received_contexts[0].response_target_set_id == 55


# ---------------------------------------------------------------------------
# 7. Exact route traceability preserved
# ---------------------------------------------------------------------------


def test_exact_route_result_id_traceability_preserved():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (
        make_candidate(route_result_id=42, resource_id="R1", travel_time_seconds=50.0),
        make_candidate(route_result_id=99, resource_id="R2", travel_time_seconds=999.0),
    )
    scorer = RecordingScorer(result=KNOWN_SCORE)

    result = evaluate(targets=targets, route_candidates=candidates, scorer=scorer)

    assert len(result.actions) == 1
    assert result.actions[0].route_result_id == 42
    assert result.actions[0].resource_id == "R1"
    assert result.actions[0].response_target_id == 1


# ---------------------------------------------------------------------------
# 8 & 9. Methodology / methodology version
# ---------------------------------------------------------------------------


def test_methodology_is_greedy_nearest_available():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=50.0),)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    result = evaluate(targets=targets, route_candidates=candidates, scorer=scorer)

    assert result.methodology == "GREEDY_NEAREST_AVAILABLE"
    assert result.methodology == BASELINE_PLAN_METHODOLOGY


def test_methodology_version_is_1_0():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=50.0),)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    result = evaluate(targets=targets, route_candidates=candidates, scorer=scorer)

    assert result.methodology_version == "1.0"
    assert result.methodology_version == BASELINE_PLAN_METHODOLOGY_VERSION


# ---------------------------------------------------------------------------
# 10. Uncovered targets reach the scoring boundary
# ---------------------------------------------------------------------------


def test_uncovered_targets_reach_the_scoring_context():
    targets = (
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
    )
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, status="unreachable"),
        make_candidate(route_result_id=2, resource_id="R1", response_target_id=2, travel_time_seconds=30.0),
    )
    scorer = RecordingScorer(result=KNOWN_SCORE)

    result = evaluate(targets=targets, route_candidates=candidates, scorer=scorer)

    assert scorer.received_contexts[0].allocation.uncovered_response_target_ids == (1,)
    # The result's score is whatever the scorer decided -- preserved as-is.
    assert result.score.uncovered_target_ids == KNOWN_SCORE.uncovered_target_ids


# ---------------------------------------------------------------------------
# 11. Zero feasible actions
# ---------------------------------------------------------------------------


def test_zero_actions_does_not_crash_and_scorer_still_decides_score():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", status="unreachable"),)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    result = evaluate(targets=targets, route_candidates=candidates, scorer=scorer)

    assert result.actions == ()
    assert scorer.call_count == 1
    assert result.score is KNOWN_SCORE


# ---------------------------------------------------------------------------
# 12. Empty target set
# ---------------------------------------------------------------------------


def test_empty_target_set_handled_deterministically():
    scorer = RecordingScorer(result=KNOWN_SCORE)

    result = evaluate(targets=(), route_candidates=(), scorer=scorer)

    assert result.actions == ()
    assert scorer.call_count == 1
    assert scorer.received_contexts[0].allocation == BaselinePlanAllocation(
        assignments=(), uncovered_response_target_ids=()
    )
    assert result.score is KNOWN_SCORE


# ---------------------------------------------------------------------------
# 13. No mutation of inputs
# ---------------------------------------------------------------------------


def test_input_targets_and_routes_are_not_mutated():
    targets = [
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
    ]
    candidates = [
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, travel_time_seconds=50.0),
        make_candidate(route_result_id=2, resource_id="R1", response_target_id=2, travel_time_seconds=10.0),
    ]
    targets_snapshot = list(targets)
    candidates_snapshot = list(candidates)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    evaluate(targets=targets, route_candidates=candidates, scorer=scorer)

    assert targets == targets_snapshot
    assert candidates == candidates_snapshot


# ---------------------------------------------------------------------------
# 14. Determinism
# ---------------------------------------------------------------------------


def test_identical_inputs_and_scorer_result_produce_identical_output():
    targets = (
        make_target(response_target_id=1, target_order=1),
        make_target(response_target_id=2, target_order=2),
    )
    candidates = (
        make_candidate(route_result_id=1, resource_id="R1", response_target_id=1, travel_time_seconds=50.0),
        make_candidate(route_result_id=2, resource_id="R1", response_target_id=2, travel_time_seconds=10.0),
    )

    first = evaluate(targets=targets, route_candidates=candidates, scorer=RecordingScorer(result=KNOWN_SCORE))
    second = evaluate(targets=targets, route_candidates=candidates, scorer=RecordingScorer(result=KNOWN_SCORE))

    assert first == second


# ---------------------------------------------------------------------------
# 15. Scorer error propagates
# ---------------------------------------------------------------------------


def test_scorer_exception_propagates_without_inventing_a_score():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=50.0),)

    with pytest.raises(RuntimeError, match="shared scorer failed"):
        evaluate(targets=targets, route_candidates=candidates, scorer=RaisingScorer())


# ---------------------------------------------------------------------------
# Input / result validation
# ---------------------------------------------------------------------------


def test_non_positive_route_planning_run_id_rejected():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=50.0),)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    with pytest.raises(ValueError):
        evaluate(route_planning_run_id=0, targets=targets, route_candidates=candidates, scorer=scorer)
    assert scorer.call_count == 0


def test_non_positive_response_target_set_id_rejected():
    targets = (make_target(response_target_id=1, target_order=0),)
    candidates = (make_candidate(route_result_id=1, resource_id="R1", travel_time_seconds=50.0),)
    scorer = RecordingScorer(result=KNOWN_SCORE)

    with pytest.raises(ValueError):
        evaluate(response_target_set_id=0, targets=targets, route_candidates=candidates, scorer=scorer)
    assert scorer.call_count == 0


def test_baseline_plan_result_rejects_wrong_methodology():
    with pytest.raises(ValueError):
        BaselinePlanResult(
            route_planning_run_id=1,
            response_target_set_id=1,
            actions=(),
            score=KNOWN_SCORE,
            methodology="SOMETHING_ELSE",
        )
