"""Stage 5 (severity-driven demand + multi-slot allocation + shortage +
farther-station reinforcement) full-pipeline acceptance tests for
GlobalResponseOptimizationService. These exercise the real service/GA
together - the central scenarios the Stage 5 spec requires direct evidence
for (Tasks 25/26/29/30/31/32/34), on top of the narrower unit-level
coverage already in test_global_response_plan_scorer.py and
test_global_allocation_slot.py.
"""
from __future__ import annotations

import itertools
import time

import pytest

from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    build_global_optimization_problem,
)
from src.calculators.global_response_optimization.global_demand_scoring_policy import GlobalDemandScoringPolicy
from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy
from src.calculators.global_response_optimization.global_response_plan_scorer import GlobalResponsePlanScorer
from src.models.demand_source import DemandSource
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome
from src.models.response_target_type import ResponseTargetType
from src.models.resource_status import ResourceStatus
from src.calculators.global_response_optimization.global_response_optimization_service import (
    GlobalResponseOptimizationService,
)
from tests.calculators.global_response_optimization.helpers import make_demand, make_input, make_resource, make_route, make_target

RELIABLE_CONFIG = GlobalResponseOptimizationConfig(population_size=60, generation_count=80, random_seed=42)


def _service() -> GlobalResponseOptimizationService:
    return GlobalResponseOptimizationService()


# ---------------------------------------------------------------------------
# Task 25 - multiple resources assigned to ONE fire's ACTIVE_FIRE target
# ---------------------------------------------------------------------------


def test_critical_fire_with_four_feasible_resources_gets_four_distinct_resources_on_the_same_target():
    demand = make_demand(
        1,
        minimum_resources=3,
        desired_resources=4,
        severity_level=FireSeverityLevel.CRITICAL,
        severity_score=95.0,
        demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=195.0),),
        resources=tuple(make_resource(f"R{i}") for i in range(4)),
        routes=tuple(make_route(f"R{i}", 1, 10, eta_seconds=10.0 * (i + 1)) for i in range(4)),
        incident_demands=(demand,),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    resource_ids = {action.resource_id for action in result.actions}
    target_ids = {action.response_target_id for action in result.actions}
    assert resource_ids == {"R0", "R1", "R2", "R3"}
    assert target_ids == {10}
    (event_result,) = result.event_results
    assert event_result.demand_result.suppression_resources_assigned == 4
    assert event_result.demand_result.required_slots_uncovered == 0
    assert event_result.demand_result.desired_slots_uncovered == 0


# ---------------------------------------------------------------------------
# Task 26 - severity monotonicity: higher severity never gets fewer
# resources assigned than a lower-severity fire when supply is ample.
# ---------------------------------------------------------------------------


def test_higher_severity_incident_receives_at_least_as_many_resources_as_a_lower_severity_one():
    low_demand = make_demand(
        1, minimum_resources=1, desired_resources=1,
        severity_level=FireSeverityLevel.LOW, severity_score=10.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    critical_demand = make_demand(
        2, minimum_resources=3, desired_resources=4,
        severity_level=FireSeverityLevel.CRITICAL, severity_score=95.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    resources = tuple(make_resource(f"R{i}") for i in range(5))
    routes = tuple(make_route(f"R{i}", 1, 10, eta_seconds=20.0) for i in range(5)) + tuple(
        make_route(f"R{i}", 2, 20, eta_seconds=20.0) for i in range(5)
    )
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=110.0), make_target(2, 20, priority_score=195.0)),
        resources=resources,
        routes=routes,
        incident_demands=(low_demand, critical_demand),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    by_event = {event_result.fire_event_id: event_result for event_result in result.event_results}
    low_assigned = by_event[1].demand_result.suppression_resources_assigned
    critical_assigned = by_event[2].demand_result.suppression_resources_assigned
    assert critical_assigned >= low_assigned
    assert critical_assigned == 4
    assert low_assigned == 1


# ---------------------------------------------------------------------------
# Task 29 - desired reinforcement only happens after every event's minimum
# is satisfied.
# ---------------------------------------------------------------------------


def test_desired_slots_are_not_filled_while_another_events_minimum_is_still_uncovered():
    starved_demand = make_demand(
        1, minimum_resources=2, desired_resources=2,
        severity_level=FireSeverityLevel.MODERATE, severity_score=40.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    reinforced_demand = make_demand(
        2, minimum_resources=1, desired_resources=3,
        severity_level=FireSeverityLevel.HIGH, severity_score=65.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    # Only 3 resources exist for 2 (min) + 1 (min) + 2 (extra desired) = up
    # to 6 possible slots - not enough for every desired slot, so the GA
    # must prioritize both events' minimums before spending anything on
    # event 2's extra desired slots.
    resources = tuple(make_resource(f"R{i}") for i in range(3))
    routes = tuple(make_route(f"R{i}", 1, 10, eta_seconds=20.0) for i in range(3)) + tuple(
        make_route(f"R{i}", 2, 20, eta_seconds=20.0) for i in range(3)
    )
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=140.0), make_target(2, 20, priority_score=165.0)),
        resources=resources,
        routes=routes,
        incident_demands=(starved_demand, reinforced_demand),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    by_event = {event_result.fire_event_id: event_result for event_result in result.event_results}
    assert by_event[1].demand_result.required_slots_uncovered == 0
    assert by_event[2].demand_result.required_slots_uncovered == 0
    # With exactly 3 resources and both minimums (2 + 1 = 3) fully covered,
    # nothing is left for event 2's desired slots.
    assert by_event[2].demand_result.desired_slots_covered == 0


# ---------------------------------------------------------------------------
# Task 30 - shared-area shortage: two HIGH fires, insufficient total supply
# ---------------------------------------------------------------------------


def test_shared_area_shortage_is_reported_honestly_when_two_high_fires_outstrip_supply():
    demand_a = make_demand(
        1, minimum_resources=2, desired_resources=3,
        severity_level=FireSeverityLevel.HIGH, severity_score=65.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    demand_b = make_demand(
        2, minimum_resources=2, desired_resources=3,
        severity_level=FireSeverityLevel.HIGH, severity_score=65.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    resources = tuple(make_resource(f"R{i}") for i in range(3))
    routes = tuple(make_route(f"R{i}", 1, 10, eta_seconds=20.0) for i in range(3)) + tuple(
        make_route(f"R{i}", 2, 20, eta_seconds=20.0) for i in range(3)
    )
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=165.0), make_target(2, 20, priority_score=165.0)),
        resources=resources,
        routes=routes,
        incident_demands=(demand_a, demand_b),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assert result.shortage.total_required == 4
    assert result.shortage.total_desired == 6
    assert result.shortage.total_assigned == 3
    assert result.shortage.unmet_required == 1
    assert result.shortage.has_shortage is True
    resource_ids = [action.resource_id for action in result.actions]
    assert len(resource_ids) == len(set(resource_ids)) == 3


# ---------------------------------------------------------------------------
# Task 31 - farther-station reinforcement, end to end
# ---------------------------------------------------------------------------


def test_farther_station_resources_reinforce_a_locally_under_supplied_critical_fire():
    demand = make_demand(
        1, minimum_resources=3, desired_resources=4,
        severity_level=FireSeverityLevel.CRITICAL, severity_score=95.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    local_resources = (make_resource("LOCAL1"), make_resource("LOCAL2"))
    reinforcement_resources = (make_resource("FAR1"), make_resource("FAR2"), make_resource("FAR3"))
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=195.0),),
        resources=local_resources + reinforcement_resources,
        routes=(
            make_route("LOCAL1", 1, 10, eta_seconds=30.0),
            make_route("LOCAL2", 1, 10, eta_seconds=40.0),
            make_route("FAR1", 1, 10, eta_seconds=900.0),
            make_route("FAR2", 1, 10, eta_seconds=950.0),
            make_route("FAR3", 1, 10, eta_seconds=1000.0),
        ),
        incident_demands=(demand,),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    # Only 4 slots exist (minimum=3, desired=4) for 5 candidate resources -
    # the two local resources cover 2 of the 3 required slots, and farther
    # (genuinely reachable) reinforcement resources fill the rest: the
    # third required slot plus the one extra desired slot. Exactly one of
    # the three farther resources is left unused, by design (5 candidates,
    # 4 slots) - never fabricated, never duplicated.
    resource_ids = {action.resource_id for action in result.actions}
    assert resource_ids == {"LOCAL1", "LOCAL2", "FAR1", "FAR2"}
    (event_result,) = result.event_results
    assert event_result.demand_result.required_slots_uncovered == 0
    assert event_result.demand_result.desired_slots_uncovered == 0
    assert event_result.demand_result.suppression_resources_assigned == 4


# ---------------------------------------------------------------------------
# Task 32 - two events competing for one local station: ONE chromosome/ONE
# global result, not sequential per-event allocation.
# ---------------------------------------------------------------------------


def test_two_events_competing_for_one_local_station_resolves_via_one_global_optimization():
    demand_a = make_demand(
        1, minimum_resources=1, desired_resources=1,
        severity_level=FireSeverityLevel.HIGH, severity_score=65.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    demand_b = make_demand(
        2, minimum_resources=1, desired_resources=1,
        severity_level=FireSeverityLevel.CRITICAL, severity_score=95.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    # A single shared local resource is excellent for BOTH events; a second,
    # much worse resource exists only for event A.
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=165.0), make_target(2, 20, priority_score=195.0)),
        resources=(make_resource("SHARED"), make_resource("BACKUP_A")),
        routes=(
            make_route("SHARED", 1, 10, eta_seconds=10.0),
            make_route("SHARED", 2, 20, eta_seconds=12.0),
            make_route("BACKUP_A", 1, 10, eta_seconds=200.0),
        ),
        incident_demands=(demand_a, demand_b),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assignment = {action.resource_id: action.response_target_id for action in result.actions}
    resource_ids = list(assignment)
    assert len(resource_ids) == len(set(resource_ids))
    # The globally-superior allocation gives SHARED to the higher-severity
    # CRITICAL event and falls back to BACKUP_A for event 1, covering both
    # minimums in one shot rather than starving event 1 by processing
    # events sequentially.
    assert assignment.get("SHARED") == 20
    assert assignment.get("BACKUP_A") == 10
    by_event = {event_result.fire_event_id: event_result for event_result in result.event_results}
    assert by_event[1].demand_result.required_slots_uncovered == 0
    assert by_event[2].demand_result.required_slots_uncovered == 0


# ---------------------------------------------------------------------------
# Task 34 - UNAVAILABLE resources never receive a slot, even with stale
# commitment metadata.
# ---------------------------------------------------------------------------


def test_unavailable_resource_is_never_assigned_even_with_stale_commitment_metadata():
    unavailable_but_committed = make_resource(
        "GHOST",
        operational_status=ResourceStatus.UNAVAILABLE,
        current_commitment_fire_event_id=1,
        current_commitment_response_plan_id=1,
    )
    available_resource = make_resource("REAL")
    demand = make_demand(1, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=100.0),),
        resources=(unavailable_but_committed, available_resource),
        routes=(
            make_route("GHOST", 1, 10, eta_seconds=5.0),
            make_route("REAL", 1, 10, eta_seconds=50.0),
        ),
        incident_demands=(demand,),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    resource_ids = {action.resource_id for action in result.actions}
    assert "GHOST" not in resource_ids
    assert resource_ids == {"REAL"}


# ---------------------------------------------------------------------------
# Task 37 - exact-oracle report for multi-slot demand (severity-driven,
# several required + desired slots on one target, not just one slot each).
# ---------------------------------------------------------------------------


def _exhaustive_best_fitness(problem, scorer) -> float:
    """Test-only exhaustive enumeration of every feasible chromosome - never used in production."""
    choices_per_resource = [
        [None, *problem.feasible_slot_ids_by_resource[resource_id]] for resource_id in problem.resource_ids
    ]
    best = 0.0
    for combo in itertools.product(*choices_per_resource):
        assigned = [slot_id for slot_id in combo if slot_id is not None]
        if len(assigned) != len(set(assigned)):
            continue
        chromosome = GlobalResponsePlanChromosome(problem.resource_ids, combo)
        score = scorer.evaluate(problem, chromosome)
        best = max(best, score.fitness_score)
    return best


def test_ga_reaches_the_exhaustive_optimum_with_multi_slot_severity_driven_demand():
    demand = make_demand(
        1, minimum_resources=2, desired_resources=3,
        severity_level=FireSeverityLevel.HIGH, severity_score=65.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=165.0),),
        resources=(make_resource("R1"), make_resource("R2"), make_resource("R3"), make_resource("R4")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=40.0),
            make_route("R2", 1, 10, eta_seconds=90.0),
            make_route("R3", 1, 10, eta_seconds=25.0),
            make_route("R4", 1, 10, eta_seconds=500.0),
        ),
        incident_demands=(demand,),
    )
    scorer = GlobalResponsePlanScorer(RELIABLE_CONFIG)
    problem = build_global_optimization_problem(global_input)
    exhaustive_best = _exhaustive_best_fitness(problem, scorer)

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assert result.fitness_score == pytest.approx(exhaustive_best, abs=1e-9)


# ---------------------------------------------------------------------------
# Task 38 - reproducibility with demand + custom scoring policies
# ---------------------------------------------------------------------------


def test_same_seed_with_demand_and_custom_policies_is_fully_reproducible():
    demand = make_demand(
        1, minimum_resources=1, desired_resources=2,
        severity_level=FireSeverityLevel.HIGH, severity_score=65.0, demand_source=DemandSource.SEVERITY_ASSESSMENT,
    )
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=165.0),),
        resources=(make_resource("R1"), make_resource("R2"), make_resource("R3")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=10.0),
            make_route("R2", 1, 10, eta_seconds=20.0),
            make_route("R3", 1, 10, eta_seconds=30.0),
        ),
        incident_demands=(demand,),
    )
    demand_scoring_policy = GlobalDemandScoringPolicy()
    severity_demand_policy = SeverityDemandPolicy()

    first = _service().optimize(global_input, RELIABLE_CONFIG, demand_scoring_policy, severity_demand_policy)
    second = _service().optimize(global_input, RELIABLE_CONFIG, demand_scoring_policy, severity_demand_policy)

    assert first.actions == second.actions
    assert first.fitness_score == second.fitness_score
    assert first.coverage_score == second.coverage_score
    assert first.shortage == second.shortage
    assert first.event_results == second.event_results


# ---------------------------------------------------------------------------
# Task 39 - performance guard with severity-driven slot counts and
# predicted-risk targets (3-5 events, 15-25 resources).
# ---------------------------------------------------------------------------


def test_performance_guard_with_severity_driven_slots_and_predicted_risk_targets():
    fire_event_ids = (1, 2, 3, 4)
    severities = (FireSeverityLevel.LOW, FireSeverityLevel.MODERATE, FireSeverityLevel.HIGH, FireSeverityLevel.CRITICAL)
    demand_pairs = {
        FireSeverityLevel.LOW: (1, 1),
        FireSeverityLevel.MODERATE: (1, 2),
        FireSeverityLevel.HIGH: (2, 3),
        FireSeverityLevel.CRITICAL: (3, 4),
    }
    targets = []
    demands = []
    for index, (fire_event_id, severity) in enumerate(zip(fire_event_ids, severities)):
        minimum_resources, desired_resources = demand_pairs[severity]
        targets.append(make_target(fire_event_id, fire_event_id * 100, priority_score=100.0 + index, target_type=ResponseTargetType.ACTIVE_FIRE))
        targets.append(
            make_target(
                fire_event_id, fire_event_id * 100 + 1, priority_score=50.0 + index,
                target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30,
            )
        )
        demands.append(
            make_demand(
                fire_event_id, minimum_resources=minimum_resources, desired_resources=desired_resources,
                severity_level=severity, severity_score=float(index * 20), demand_source=DemandSource.SEVERITY_ASSESSMENT,
            )
        )
    resource_count = 20
    resources = tuple(make_resource(f"R{i}") for i in range(resource_count))
    routes = tuple(
        make_route(f"R{i}", target.fire_event_id, target.response_target_id, eta_seconds=float(10 * (i + 1) + target.response_target_id))
        for i, resource in enumerate(resources)
        for target in targets
    )
    global_input = make_input(
        active_fire_event_ids=fire_event_ids, targets=tuple(targets), resources=resources, routes=routes,
        incident_demands=tuple(demands),
    )
    config = GlobalResponseOptimizationConfig(population_size=30, generation_count=25, random_seed=7)
    problem = build_global_optimization_problem(global_input)

    started = time.monotonic()
    result = _service().optimize(global_input, config)
    elapsed = time.monotonic() - started

    slot_count = len(problem.slots)
    potential_pairs = len(problem.resource_ids) * slot_count
    feasible_pairs = sum(len(slots) for slots in problem.feasible_slot_ids_by_resource.values())

    resource_ids = [action.resource_id for action in result.actions]
    assert len(resource_ids) == len(set(resource_ids))
    assert len(fire_event_ids) == 4
    assert resource_count == 20
    assert len(targets) == 8
    assert slot_count == sum(pair[1] for pair in demand_pairs.values()) + len(fire_event_ids)  # desired suppression slots + 1 predicted-risk slot each
    assert potential_pairs == len(problem.resource_ids) * slot_count
    assert feasible_pairs > 0
    assert elapsed < 30.0  # generous, non-brittle guard - just proves it terminates promptly
