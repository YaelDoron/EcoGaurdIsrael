"""Tests for GlobalResponsePlanChromosomeDecoder (Stage 4, Task 20, 25)."""
from __future__ import annotations

import pytest

from src.calculators.global_response_optimization.global_response_optimization_problem import (
    build_global_optimization_problem,
)
from src.calculators.global_response_optimization.global_response_plan_chromosome_decoder import (
    GlobalResponsePlanChromosomeDecodeError,
    GlobalResponsePlanChromosomeDecoder,
)
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome
from src.models.resource_status import ResourceStatus
from tests.calculators.global_response_optimization.helpers import make_input, make_resource, make_route, make_target


def _problem(targets, resources, routes, active_fire_event_ids=(1,)):
    return build_global_optimization_problem(
        make_input(active_fire_event_ids=active_fire_event_ids, targets=targets, resources=resources, routes=routes)
    )


def test_decodes_feasible_assignment_with_route_facts_from_the_matrix():
    problem = _problem(
        (make_target(1, 10, priority_score=77.0),),
        (make_resource("R1", station_id="S1"),),
        (make_route("R1", 1, 10, eta_seconds=45.0, route_distance_meters=900.0, node_path=(1, 2, 3)),),
    )
    chromosome = GlobalResponsePlanChromosome(problem.resource_ids, (problem.slots[0].slot_id,))

    (action,) = GlobalResponsePlanChromosomeDecoder.decode(problem, chromosome)

    assert action.resource_id == "R1"
    assert action.station_id == "S1"
    assert action.fire_event_id == 1
    assert action.response_target_id == 10
    assert action.target_priority == 77.0
    assert action.eta_seconds == 45.0
    assert action.route_distance_meters == 900.0
    assert action.node_path == (1, 2, 3)


def test_idle_gene_produces_no_action():
    problem = _problem((make_target(1, 10),), (make_resource("R1"),), ())
    chromosome = GlobalResponsePlanChromosome(problem.resource_ids, (None,))

    actions = GlobalResponsePlanChromosomeDecoder.decode(problem, chromosome)

    assert actions == ()


def test_never_decodes_an_unreachable_pair():
    """Task 25: the route matrix lacks (R1, target 10) -> the chromosome
    must never decode to that assignment, even if somehow constructed."""
    problem = _problem((make_target(1, 10),), (make_resource("R1"),), ())  # no routes at all
    # Bypass normal construction paths (which would never produce this) to
    # directly prove the decoder itself refuses an infeasible chromosome.
    chromosome = GlobalResponsePlanChromosome(problem.resource_ids, (problem.slots[0].slot_id,))

    with pytest.raises(GlobalResponsePlanChromosomeDecodeError):
        GlobalResponsePlanChromosomeDecoder.decode(problem, chromosome)


def test_rejects_unavailable_resource_even_if_gene_present():
    problem = _problem(
        (make_target(1, 10),),
        (make_resource("R1", operational_status=ResourceStatus.UNAVAILABLE),),
        (make_route("R1", 1, 10, eta_seconds=10.0),),
    )
    # R1 is UNAVAILABLE -> excluded from problem.resource_ids entirely, so a
    # chromosome cannot even reference it structurally. Confirm directly.
    assert "R1" not in problem.resource_ids


def test_rejects_chromosome_with_mismatched_resource_ids():
    problem = _problem((make_target(1, 10),), (make_resource("R1"),), ())
    chromosome = GlobalResponsePlanChromosome(("OTHER",), (None,))

    with pytest.raises(GlobalResponsePlanChromosomeDecodeError):
        GlobalResponsePlanChromosomeDecoder.decode(problem, chromosome)


def test_multiple_assignments_decode_independently():
    problem = _problem(
        (make_target(1, 10), make_target(2, 11)),
        (make_resource("R1"), make_resource("R2")),
        (make_route("R1", 1, 10, eta_seconds=10.0), make_route("R2", 2, 11, eta_seconds=20.0)),
        active_fire_event_ids=(1, 2),
    )
    slot_10 = next(s for s in problem.slots if s.response_target_id == 10).slot_id
    slot_11 = next(s for s in problem.slots if s.response_target_id == 11).slot_id
    genes = tuple(slot_10 if rid == "R1" else slot_11 for rid in problem.resource_ids)
    chromosome = GlobalResponsePlanChromosome(problem.resource_ids, genes)

    actions = GlobalResponsePlanChromosomeDecoder.decode(problem, chromosome)

    assert len(actions) == 2
    assert {a.resource_id for a in actions} == {"R1", "R2"}
