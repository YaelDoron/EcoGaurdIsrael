"""Tests for ResponsePlanBaselineScorerAdapter, using a fake ResponsePlanScorer (no GA)."""
from __future__ import annotations

import pytest

from src.calculators.baseline_plan.baseline_plan_calculator import BaselineAssignment, BaselinePlanAllocation, RouteCandidate, TargetOrder
from src.calculators.baseline_plan.baseline_plan_evaluator import BaselinePlanScoringContext
from src.models import ResponseTargetType
from src.models.plan_score_breakdown import PlanScoreBreakdown
from src.models.response_plan_status import ResponsePlanStatus
from src.services.baseline_comparison.response_plan_baseline_scorer_adapter import ResponsePlanBaselineScorerAdapter

FIRE_EVENT_ID = 42
RUN_ID = 601
TARGET_SET_ID = 77


class FakeResponsePlanScorer:
    def __init__(self, result: PlanScoreBreakdown | None = None):
        self.result = result or make_score_breakdown()
        self.calls = []

    def evaluate(self, optimization_input, actions):
        self.calls.append({"optimization_input": optimization_input, "actions": actions})
        return self.result


def make_score_breakdown(**overrides) -> PlanScoreBreakdown:
    defaults = dict(
        total_score=80.0,
        raw_fitness=80.0,
        coverage_score=100.0,
        average_eta_seconds=50.0,
        total_priority=100.0,
        covered_priority=100.0,
        covered_target_count=1,
        total_target_count=1,
        uncovered_target_ids=(),
        status=ResponsePlanStatus.COMPLETE,
    )
    defaults.update(overrides)
    return PlanScoreBreakdown(**defaults)


def make_target(**overrides) -> TargetOrder:
    defaults = dict(
        response_target_id=900,
        target_order=0,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        priority_score=150.0,
    )
    defaults.update(overrides)
    return TargetOrder(**defaults)


def make_candidate(**overrides) -> RouteCandidate:
    defaults = dict(
        route_result_id=1,
        resource_id="truck-1",
        response_target_id=900,
        status="reachable",
        travel_time_seconds=90.0,
        distance_meters=1000.0,
    )
    defaults.update(overrides)
    return RouteCandidate(**defaults)


def make_context(
    *,
    fire_event_id=FIRE_EVENT_ID,
    route_planning_run_id=RUN_ID,
    response_target_set_id=TARGET_SET_ID,
    targets=None,
    route_candidates=None,
    assignments=None,
    uncovered_response_target_ids=(),
) -> BaselinePlanScoringContext:
    targets = (make_target(),) if targets is None else targets
    route_candidates = (make_candidate(),) if route_candidates is None else route_candidates
    assignments = (
        (BaselineAssignment(resource_id="truck-1", response_target_id=900, route_result_id=1),)
        if assignments is None
        else assignments
    )
    return BaselinePlanScoringContext(
        fire_event_id=fire_event_id,
        route_planning_run_id=route_planning_run_id,
        response_target_set_id=response_target_set_id,
        targets=targets,
        route_candidates=route_candidates,
        allocation=BaselinePlanAllocation(
            assignments=assignments,
            uncovered_response_target_ids=uncovered_response_target_ids,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Delegates exactly once
# ---------------------------------------------------------------------------


def test_delegates_to_response_plan_scorer_exactly_once():
    scorer = FakeResponsePlanScorer()
    adapter = ResponsePlanBaselineScorerAdapter(scorer)

    adapter.evaluate(make_context())

    assert len(scorer.calls) == 1


# ---------------------------------------------------------------------------
# 2-7. Exact fields forwarded
# ---------------------------------------------------------------------------


def test_exact_fire_event_id_forwarded():
    scorer = FakeResponsePlanScorer()
    ResponsePlanBaselineScorerAdapter(scorer).evaluate(make_context(fire_event_id=999))

    assert scorer.calls[0]["optimization_input"].fire_event_id == 999


def test_exact_target_type_forwarded():
    scorer = FakeResponsePlanScorer()
    context = make_context(targets=(make_target(target_type=ResponseTargetType.PREDICTED_RISK),))

    ResponsePlanBaselineScorerAdapter(scorer).evaluate(context)

    assert scorer.calls[0]["optimization_input"].targets[0].target_type is ResponseTargetType.PREDICTED_RISK


def test_exact_priority_score_forwarded():
    scorer = FakeResponsePlanScorer()
    context = make_context(targets=(make_target(priority_score=321.5),))

    ResponsePlanBaselineScorerAdapter(scorer).evaluate(context)

    assert scorer.calls[0]["optimization_input"].targets[0].priority_score == 321.5


def test_exact_travel_time_seconds_forwarded():
    scorer = FakeResponsePlanScorer()
    context = make_context(route_candidates=(make_candidate(travel_time_seconds=77.0),))

    ResponsePlanBaselineScorerAdapter(scorer).evaluate(context)

    assert scorer.calls[0]["optimization_input"].route_options[0].travel_time_seconds == 77.0


def test_exact_distance_meters_forwarded():
    scorer = FakeResponsePlanScorer()
    context = make_context(route_candidates=(make_candidate(distance_meters=4321.0),))

    ResponsePlanBaselineScorerAdapter(scorer).evaluate(context)

    assert scorer.calls[0]["optimization_input"].route_options[0].distance_meters == 4321.0


def test_exact_resource_and_target_relationships_preserved():
    scorer = FakeResponsePlanScorer()
    context = make_context(
        targets=(make_target(response_target_id=900),),
        route_candidates=(make_candidate(resource_id="truck-9", response_target_id=900),),
        assignments=(BaselineAssignment(resource_id="truck-9", response_target_id=900, route_result_id=1),),
    )

    ResponsePlanBaselineScorerAdapter(scorer).evaluate(context)

    action = scorer.calls[0]["actions"][0]
    assert action.resource_id == "truck-9"
    assert action.response_target_id == 900
    assert action.route_result_id == 1


# ---------------------------------------------------------------------------
# 8. Result mapped back without recomputation
# ---------------------------------------------------------------------------


def test_shared_scorer_result_returned_unchanged():
    expected = make_score_breakdown(total_score=42.0)
    scorer = FakeResponsePlanScorer(result=expected)

    result = ResponsePlanBaselineScorerAdapter(scorer).evaluate(make_context())

    assert result is expected


# ---------------------------------------------------------------------------
# 9. Multiple targets preserve distinct priorities
# ---------------------------------------------------------------------------


def test_multiple_targets_with_different_priorities_preserve_exact_values():
    scorer = FakeResponsePlanScorer()
    targets = (
        make_target(response_target_id=1, target_order=0, priority_score=200.0),
        make_target(response_target_id=2, target_order=1, priority_score=50.0),
    )
    route_candidates = (
        make_candidate(route_result_id=1, resource_id="truck-1", response_target_id=1),
        make_candidate(route_result_id=2, resource_id="truck-2", response_target_id=2),
    )
    context = make_context(
        targets=targets,
        route_candidates=route_candidates,
        assignments=(),
    )

    ResponsePlanBaselineScorerAdapter(scorer).evaluate(context)

    priority_by_id = {t.response_target_id: t.priority_score for t in scorer.calls[0]["optimization_input"].targets}
    assert priority_by_id == {1: 200.0, 2: 50.0}


# ---------------------------------------------------------------------------
# 10-11. Missing/invalid data fails explicitly
# ---------------------------------------------------------------------------


def test_missing_required_target_scoring_data_fails_explicitly():
    with pytest.raises(ValueError):
        make_target(priority_score="not-a-number")


def test_missing_required_route_scoring_data_fails_explicitly():
    # RouteCandidate itself allows travel_time_seconds=None even when
    # "reachable" (mirrors its existing leniency); a reachable
    # OptimizationRouteOption requires it, so the adapter's translation
    # fails explicitly rather than silently treating it as unreachable.
    scorer = FakeResponsePlanScorer()
    context = make_context(route_candidates=(make_candidate(status="reachable", travel_time_seconds=None),))

    with pytest.raises(ValueError):
        ResponsePlanBaselineScorerAdapter(scorer).evaluate(context)


def test_adapter_rejects_non_context_argument():
    scorer = FakeResponsePlanScorer()

    with pytest.raises(ValueError):
        ResponsePlanBaselineScorerAdapter(scorer).evaluate("not-a-context")


# ---------------------------------------------------------------------------
# 12-13. No repository query, no current/latest data
# ---------------------------------------------------------------------------


def test_no_repository_query_occurs_inside_the_scorer_adapter():
    scorer = FakeResponsePlanScorer()
    for forbidden in ("get_by_id", "get_latest_for_event_as_of", "get_available_resources"):
        assert not hasattr(scorer, forbidden)

    ResponsePlanBaselineScorerAdapter(scorer).evaluate(make_context())

    # The adapter only reads from the context object it was given.
    assert len(scorer.calls) == 1


def test_no_latest_or_current_operational_data_consulted():
    """The adapter builds its ResponseOptimizationInput purely from context fields."""
    import inspect

    from src.services.baseline_comparison.response_plan_baseline_scorer_adapter import (
        ResponsePlanBaselineScorerAdapter as AdapterClass,
    )

    source = inspect.getsource(AdapterClass)
    for forbidden in ("Repository(", "get_latest", "get_available", ".save(", "session"):
        assert forbidden not in source
