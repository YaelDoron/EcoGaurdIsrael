"""Tests for response-plan chromosome decoding."""
from __future__ import annotations

import pytest

from src.calculators.response_optimization import (
    ResponsePlanChromosomeDecoder,
    ResponsePlanChromosomeDecodeError,
    ResponsePlanScorer,
)
from src.models import (
    OptimizationResource,
    OptimizationRouteOption,
    OptimizationTarget,
    ResponseOptimizationInput,
    ResponsePlanChromosome,
    ResponsePlanStatus,
    ResponseTargetType,
)


def target(target_id: int, order: int = 0, priority: float = 100.0) -> OptimizationTarget:
    return OptimizationTarget(target_id, order, ResponseTargetType.ACTIVE_FIRE, priority)


def resource(resource_id: int | str) -> OptimizationResource:
    return OptimizationResource(resource_id)


def route(
    route_id: int,
    resource_id: int | str,
    target_id: int,
    *,
    reachable: bool = True,
) -> OptimizationRouteOption:
    return OptimizationRouteOption(
        route_result_id=route_id,
        resource_id=resource_id,
        response_target_id=target_id,
        is_reachable=reachable,
        travel_time_seconds=300.0 if reachable else None,
        distance_meters=1000.0 if reachable else None,
    )


def optimization_input(
    *,
    targets=(target(10, 0),),
    resources=(resource("R1"),),
    route_options=(route(100, "R1", 10),),
) -> ResponseOptimizationInput:
    return ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=targets,
        resources=resources,
        route_options=route_options,
    )


def test_decodes_target_indexed_chromosome_to_actions():
    input_data = optimization_input(
        targets=(target(10, 0), target(20, 1), target(30, 2)),
        resources=(resource("R1"), resource("R2")),
        route_options=(route(100, "R2", 10), route(200, "R1", 30)),
    )

    actions = ResponsePlanChromosomeDecoder.decode(
        input_data,
        ResponsePlanChromosome(("R2", None, "R1")),
    )

    assert [(action.resource_id, action.response_target_id, action.route_result_id) for action in actions] == [
        ("R2", 10, 100),
        ("R1", 30, 200),
    ]


def test_all_none_decodes_to_empty_actions():
    input_data = optimization_input(targets=(target(10, 0), target(20, 1)), route_options=())

    assert ResponsePlanChromosomeDecoder.decode(input_data, ResponsePlanChromosome((None, None))) == ()


def test_wrong_chromosome_length_rejected():
    with pytest.raises(ResponsePlanChromosomeDecodeError):
        ResponsePlanChromosomeDecoder.decode(optimization_input(), ResponsePlanChromosome((None, None)))


def test_unknown_resource_rejected():
    with pytest.raises(ResponsePlanChromosomeDecodeError):
        ResponsePlanChromosomeDecoder.decode(optimization_input(), ResponsePlanChromosome(("MISSING",)))


def test_no_route_option_for_pair_rejected_without_substitution():
    input_data = optimization_input(
        targets=(target(10, 0),),
        resources=(resource("R1"), resource("R2")),
        route_options=(route(100, "R2", 10),),
    )

    with pytest.raises(ResponsePlanChromosomeDecodeError):
        ResponsePlanChromosomeDecoder.decode(input_data, ResponsePlanChromosome(("R1",)))


def test_unreachable_route_rejected():
    input_data = optimization_input(route_options=(route(100, "R1", 10, reachable=False),))

    with pytest.raises(ResponsePlanChromosomeDecodeError):
        ResponsePlanChromosomeDecoder.decode(input_data, ResponsePlanChromosome(("R1",)))


def test_decoded_actions_are_accepted_by_scorer():
    input_data = optimization_input(
        targets=(target(10, 0), target(20, 1)),
        resources=(resource("R1"),),
        route_options=(route(100, "R1", 10),),
    )
    actions = ResponsePlanChromosomeDecoder.decode(input_data, ResponsePlanChromosome(("R1", None)))

    result = ResponsePlanScorer().evaluate(input_data, actions)

    assert result.status is ResponsePlanStatus.PARTIAL
    assert result.uncovered_target_ids == (20,)
