"""Decode response-plan chromosomes into scorer-ready actions."""
from __future__ import annotations

from src.calculators.response_optimization.feasible_route_lookup import route_options_by_pair
from src.models.response_action import ResponseAction
from src.models.response_optimization_input import ResponseOptimizationInput
from src.models.response_plan_chromosome import ResponsePlanChromosome


class ResponsePlanChromosomeDecodeError(ValueError):
    """Raised when a chromosome is infeasible for the supplied optimization input."""


class ResponsePlanChromosomeDecoder:
    """Convert target-indexed chromosomes into response actions."""

    @staticmethod
    def decode(
        optimization_input: ResponseOptimizationInput,
        chromosome: ResponsePlanChromosome,
    ) -> tuple[ResponseAction, ...]:
        if not isinstance(optimization_input, ResponseOptimizationInput):
            raise ResponsePlanChromosomeDecodeError(
                f"optimization_input must be a ResponseOptimizationInput, got {optimization_input!r}"
            )
        if not isinstance(chromosome, ResponsePlanChromosome):
            raise ResponsePlanChromosomeDecodeError(
                f"chromosome must be a ResponsePlanChromosome, got {chromosome!r}"
            )
        if len(chromosome.genes) != len(optimization_input.targets):
            raise ResponsePlanChromosomeDecodeError(
                "chromosome gene count must match optimization target count, got "
                f"{len(chromosome.genes)} genes for {len(optimization_input.targets)} targets."
            )

        resource_ids = {resource.resource_id for resource in optimization_input.resources}
        assigned_resources: set[int | str] = set()
        route_by_pair = route_options_by_pair(optimization_input)
        actions: list[ResponseAction] = []

        for target, resource_id in zip(optimization_input.targets, chromosome.genes):
            if resource_id is None:
                continue
            if resource_id not in resource_ids:
                raise ResponsePlanChromosomeDecodeError(
                    f"chromosome references unknown resource_id {resource_id!r}."
                )
            if resource_id in assigned_resources:
                raise ResponsePlanChromosomeDecodeError(
                    f"chromosome assigns resource_id {resource_id!r} more than once."
                )

            pair = (resource_id, target.response_target_id)
            route = route_by_pair.get(pair)
            if route is None:
                raise ResponsePlanChromosomeDecodeError(
                    f"no route option exists for resource-target pair {pair!r}."
                )
            if not route.is_reachable:
                raise ResponsePlanChromosomeDecodeError(
                    f"route option {route.route_result_id!r} for resource-target pair {pair!r} is unreachable."
                )

            assigned_resources.add(resource_id)
            actions.append(
                ResponseAction(
                    resource_id=resource_id,
                    response_target_id=target.response_target_id,
                    route_result_id=route.route_result_id,
                )
            )

        return tuple(actions)
