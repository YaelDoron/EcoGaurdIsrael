"""Tests for GlobalResponsePlanScorer (Stage 4 Task 10; demand-aware tiered
scoring since Stage 5, Tasks 14-16, 27-28)."""
from __future__ import annotations

import pytest

from src.calculators.global_response_optimization.global_demand_scoring_policy import GlobalDemandScoringPolicy
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    build_global_optimization_problem,
)
from src.calculators.global_response_optimization.global_response_plan_scorer import (
    GlobalResponsePlanScorer,
    GlobalResponsePlanScoringError,
)
from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy
from src.models.demand_source import DemandSource
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome
from src.models.response_target_type import ResponseTargetType
from tests.calculators.global_response_optimization.helpers import make_demand, make_input, make_resource, make_route, make_target


def _problem(targets, resources, routes, demands=None, active_fire_event_ids=(1,)):
    return build_global_optimization_problem(
        make_input(
            active_fire_event_ids=active_fire_event_ids,
            targets=targets,
            resources=resources,
            routes=routes,
            incident_demands=demands,
        )
    )


def test_all_idle_chromosome_scores_zero_fitness_and_coverage():
    problem = _problem((make_target(1, 10, priority_score=100.0),), (make_resource("R1"),), ())
    chromosome = GlobalResponsePlanChromosome(problem.resource_ids, (None,))

    score = GlobalResponsePlanScorer().evaluate(problem, chromosome)

    assert score.fitness_score == 0.0
    assert score.coverage_score == 0.0
    assert score.average_eta_seconds is None
    assert score.covered_slot_count == 0


def test_full_coverage_scores_full_coverage_percentage():
    problem = _problem(
        (make_target(1, 10, priority_score=100.0),),
        (make_resource("R1"),),
        (make_route("R1", 1, 10, eta_seconds=0.0),),
    )
    chromosome = GlobalResponsePlanChromosome(problem.resource_ids, (problem.slots[0].slot_id,))

    score = GlobalResponsePlanScorer().evaluate(problem, chromosome)

    assert score.coverage_score == 100.0
    assert score.average_eta_seconds == 0.0
    assert score.fitness_score > 0.0


def test_lower_eta_produces_higher_fitness_for_the_same_target():
    problem = _problem(
        (make_target(1, 10, priority_score=100.0),),
        (make_resource("R1"),),
        (make_route("R1", 1, 10, eta_seconds=60.0),),
    )
    fast_score = GlobalResponsePlanScorer().evaluate(
        problem, GlobalResponsePlanChromosome(problem.resource_ids, (problem.slots[0].slot_id,))
    )

    slow_problem = _problem(
        (make_target(1, 10, priority_score=100.0),),
        (make_resource("R1"),),
        (make_route("R1", 1, 10, eta_seconds=6000.0),),
    )
    slow_score = GlobalResponsePlanScorer().evaluate(
        slow_problem, GlobalResponsePlanChromosome(slow_problem.resource_ids, (slow_problem.slots[0].slot_id,))
    )

    assert fast_score.fitness_score > slow_score.fitness_score


def test_uncovered_slot_ids_lists_the_missing_slot():
    problem = _problem(
        (
            make_target(1, 10, target_type=ResponseTargetType.ACTIVE_FIRE),
            make_target(1, 11, target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30),
        ),
        (make_resource("R1"),),
        (make_route("R1", 1, 10, eta_seconds=60.0),),
    )
    slot_10 = next(s for s in problem.slots if s.response_target_id == 10).slot_id
    chromosome = GlobalResponsePlanChromosome(problem.resource_ids, (slot_10,))

    score = GlobalResponsePlanScorer().evaluate(problem, chromosome)

    assert score.covered_slot_count == 1
    assert score.total_slot_count == 2
    slot_11 = next(s for s in problem.slots if s.response_target_id == 11).slot_id
    assert score.uncovered_slot_ids == (slot_11,)


def test_zero_priority_target_still_scores_a_positive_tier_weight():
    """Task 14/16: coverage of a tier is valuable on its own - a zero
    target.priority_score only zeroes out the WITHIN-tier component, never
    the whole value (unlike Stage 4's old priority-normalized formula)."""
    problem = _problem(
        (make_target(1, 10, priority_score=0.0),),
        (make_resource("R1"),),
        (make_route("R1", 1, 10, eta_seconds=0.0),),
    )
    chromosome = GlobalResponsePlanChromosome(problem.resource_ids, (problem.slots[0].slot_id,))

    score = GlobalResponsePlanScorer().evaluate(problem, chromosome)

    assert score.coverage_score == 100.0
    assert score.fitness_score > 0.0  # tier weight alone is positive


def test_rejects_chromosome_with_mismatched_resource_ids():
    problem = _problem((make_target(1, 10),), (make_resource("R1"),), ())
    chromosome = GlobalResponsePlanChromosome(("R2",), (None,))

    with pytest.raises(GlobalResponsePlanScoringError):
        GlobalResponsePlanScorer().evaluate(problem, chromosome)


def test_eta_factor_uses_configured_reference_seconds():
    scorer_default = GlobalResponsePlanScorer(GlobalResponseOptimizationConfig(eta_reference_seconds=900.0))
    scorer_short = GlobalResponsePlanScorer(GlobalResponseOptimizationConfig(eta_reference_seconds=100.0))
    assert scorer_default.eta_factor(900.0) == pytest.approx(0.5)
    assert scorer_short.eta_factor(100.0) == pytest.approx(0.5)
    assert scorer_short.eta_factor(900.0) < scorer_default.eta_factor(900.0)


# ---------------------------------------------------------------------------
# Task 14-16 - required > desired > predicted-risk tier ordering
# ---------------------------------------------------------------------------


def _slot_value(problem, scorer, response_target_id, eta_seconds):
    slot = next(s for s in problem.slots if s.response_target_id == response_target_id)
    return scorer.slot_value(problem, slot, eta_seconds)


def test_required_slot_always_outscores_desired_slot_regardless_of_eta():
    """Task 27: even with a terrible ETA, a REQUIRED slot must outscore a
    DESIRED slot with a perfect ETA."""
    demand = make_demand(1, minimum_resources=1, desired_resources=2)
    problem = _problem(
        (make_target(1, 10, priority_score=200.0),),
        (make_resource("R1"), make_resource("R2")),
        (make_route("R1", 1, 10, eta_seconds=100000.0), make_route("R2", 1, 10, eta_seconds=0.0)),
        demands=(demand,),
    )
    scorer = GlobalResponsePlanScorer()
    required_slot_id, desired_slot_id = (
        problem.slots[0].slot_id if problem.slots[0].required else problem.slots[1].slot_id,
        problem.slots[1].slot_id if not problem.slots[1].required else problem.slots[0].slot_id,
    )

    worst_required_value = scorer.slot_value(problem, problem.slots_by_id[required_slot_id], eta_seconds=100000.0)
    best_desired_value = scorer.slot_value(problem, problem.slots_by_id[desired_slot_id], eta_seconds=0.0)

    assert worst_required_value > best_desired_value


def test_desired_slot_always_outscores_predicted_risk_slot():
    demand = make_demand(1, minimum_resources=0, desired_resources=1)
    problem = _problem(
        (
            make_target(1, 10, target_type=ResponseTargetType.ACTIVE_FIRE, priority_score=200.0),
            make_target(1, 11, target_type=ResponseTargetType.PREDICTED_RISK, priority_score=200.0, prediction_horizon_minutes=30),
        ),
        (make_resource("R1"),),
        (),
        demands=(demand,),
    )
    scorer = GlobalResponsePlanScorer()

    worst_desired_value = _slot_value(problem, scorer, 10, eta_seconds=100000.0)
    best_predicted_risk_value = _slot_value(problem, scorer, 11, eta_seconds=0.0)

    assert worst_desired_value > best_predicted_risk_value


def test_required_before_optional_acceptance():
    """Task 27 acceptance: Fire A has an uncovered required slot, Fire B has
    a possible desired slot; one resource can serve either - the configured
    policy must prefer covering the REQUIRED slot."""
    demand_a = make_demand(1, minimum_resources=1, desired_resources=1)
    demand_b = make_demand(2, minimum_resources=0, desired_resources=1)
    problem = _problem(
        (make_target(1, 10, priority_score=100.0), make_target(2, 20, priority_score=100.0)),
        (make_resource("R1"),),
        (make_route("R1", 1, 10, eta_seconds=500.0), make_route("R1", 2, 20, eta_seconds=1.0)),
        demands=(demand_a, demand_b),
        active_fire_event_ids=(1, 2),
    )
    scorer = GlobalResponsePlanScorer()

    required_value = _slot_value(problem, scorer, 10, eta_seconds=500.0)
    desired_value = _slot_value(problem, scorer, 20, eta_seconds=1.0)

    assert required_value > desired_value


# ---------------------------------------------------------------------------
# Task 15/28 - severity breaks ties under required-slot scarcity
# ---------------------------------------------------------------------------


def test_critical_required_slot_outranks_low_required_slot_under_comparable_eta():
    demand_critical = make_demand(
        1, minimum_resources=1, desired_resources=1, severity_level=FireSeverityLevel.CRITICAL,
        demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    demand_low = make_demand(
        2, minimum_resources=1, desired_resources=1, severity_level=FireSeverityLevel.LOW,
        demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    problem = _problem(
        (make_target(1, 10, priority_score=100.0), make_target(2, 20, priority_score=100.0)),
        (make_resource("R1"),),
        (make_route("R1", 1, 10, eta_seconds=50.0), make_route("R1", 2, 20, eta_seconds=50.0)),
        demands=(demand_critical, demand_low),
        active_fire_event_ids=(1, 2),
    )
    scorer = GlobalResponsePlanScorer()

    critical_value = _slot_value(problem, scorer, 10, eta_seconds=50.0)
    low_value = _slot_value(problem, scorer, 20, eta_seconds=50.0)

    assert critical_value > low_value


def test_severity_ordering_is_deterministic_and_explainable_via_policy():
    policy = SeverityDemandPolicy()
    ranks = [policy.severity_rank(level) for level in (
        FireSeverityLevel.LOW, FireSeverityLevel.MODERATE, FireSeverityLevel.HIGH, FireSeverityLevel.CRITICAL
    )]
    assert ranks == sorted(ranks)  # strictly increasing with severity
    assert len(set(ranks)) == 4


# ---------------------------------------------------------------------------
# Task 16 - GlobalDemandScoringPolicy validates its own safety margins
# ---------------------------------------------------------------------------


def test_scoring_policy_rejects_a_gap_too_small_to_preserve_ordering():
    with pytest.raises(ValueError):
        GlobalDemandScoringPolicy(required_coverage_weight=100.0, desired_coverage_weight=99.0)
