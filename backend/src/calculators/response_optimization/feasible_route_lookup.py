"""Deterministic route lookup helpers for response-plan chromosomes."""
from __future__ import annotations

from src.models.optimization_route_option import OptimizationRouteOption
from src.models.optimization_validation import resource_sort_key
from src.models.response_optimization_input import ResponseOptimizationInput


def route_options_by_pair(
    optimization_input: ResponseOptimizationInput,
) -> dict[tuple[int | str, int], OptimizationRouteOption]:
    """Return the authoritative route option for each resource-target pair."""
    if not isinstance(optimization_input, ResponseOptimizationInput):
        raise ValueError(f"optimization_input must be a ResponseOptimizationInput, got {optimization_input!r}")
    return {route.resource_target_key: route for route in optimization_input.route_options}


def feasible_resource_ids_by_target(
    optimization_input: ResponseOptimizationInput,
) -> dict[int, tuple[int | str, ...]]:
    """Return reachable resources per target in stable resource ordering."""
    if not isinstance(optimization_input, ResponseOptimizationInput):
        raise ValueError(f"optimization_input must be a ResponseOptimizationInput, got {optimization_input!r}")

    target_ids = {target.response_target_id for target in optimization_input.targets}
    feasible: dict[int, list[int | str]] = {target_id: [] for target_id in target_ids}
    for route in optimization_input.route_options:
        if route.is_reachable:
            feasible[route.response_target_id].append(route.resource_id)

    return {
        target_id: tuple(sorted(resource_ids, key=resource_sort_key))
        for target_id, resource_ids in feasible.items()
    }
