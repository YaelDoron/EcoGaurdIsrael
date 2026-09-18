"""Decode resource-indexed chromosomes into GlobalResponseAction (Stage 4, Task 20).

All route facts (eta_seconds, route_distance_meters, node_path) come
straight from Stage 3's GlobalRouteMatrix - never recalculated.
"""
from __future__ import annotations

from src.calculators.global_response_optimization.global_response_optimization_problem import (
    GlobalResponseOptimizationProblem,
)
from src.models.global_response_action import GlobalResponseAction
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome


class GlobalResponsePlanChromosomeDecodeError(ValueError):
    """Raised when a chromosome is infeasible for the supplied problem."""


class GlobalResponsePlanChromosomeDecoder:
    """Convert resource-indexed chromosomes into global response actions."""

    @staticmethod
    def decode(
        problem: GlobalResponseOptimizationProblem,
        chromosome: GlobalResponsePlanChromosome,
    ) -> tuple[GlobalResponseAction, ...]:
        if not isinstance(problem, GlobalResponseOptimizationProblem):
            raise GlobalResponsePlanChromosomeDecodeError(
                f"problem must be a GlobalResponseOptimizationProblem, got {problem!r}"
            )
        if not isinstance(chromosome, GlobalResponsePlanChromosome):
            raise GlobalResponsePlanChromosomeDecodeError(
                f"chromosome must be a GlobalResponsePlanChromosome, got {chromosome!r}"
            )
        if chromosome.resource_ids != problem.resource_ids:
            raise GlobalResponsePlanChromosomeDecodeError(
                "chromosome resource_ids must exactly match the problem's resource_ids."
            )

        used_slot_ids: set[str] = set()
        actions: list[GlobalResponseAction] = []

        for resource_id, slot_id in zip(chromosome.resource_ids, chromosome.genes):
            if slot_id is None:
                continue
            if slot_id in used_slot_ids:
                raise GlobalResponsePlanChromosomeDecodeError(f"slot_id {slot_id!r} is assigned more than once.")
            resource = problem.resources_by_id.get(resource_id)
            if resource is None:
                raise GlobalResponsePlanChromosomeDecodeError(f"chromosome references unknown resource_id {resource_id!r}.")
            if not resource.is_assignable:
                raise GlobalResponsePlanChromosomeDecodeError(
                    f"resource {resource_id!r} is not assignable (operational_status={resource.operational_status!r})."
                )
            slot = problem.slots_by_id.get(slot_id)
            if slot is None:
                raise GlobalResponsePlanChromosomeDecodeError(f"chromosome references unknown slot_id {slot_id!r}.")
            route = problem.route_matrix.get(resource_id, slot.response_target_id)
            if route is None:
                raise GlobalResponsePlanChromosomeDecodeError(
                    f"no feasible route exists for resource {resource_id!r} -> slot {slot_id!r}."
                )
            target = problem.targets_by_id[slot.response_target_id]

            used_slot_ids.add(slot_id)
            actions.append(
                GlobalResponseAction(
                    resource_id=resource_id,
                    station_id=resource.station_id,
                    fire_event_id=slot.fire_event_id,
                    response_target_id=slot.response_target_id,
                    target_type=target.target_type,
                    target_priority=target.priority_score,
                    eta_seconds=route.eta_seconds,
                    route_distance_meters=route.route_distance_meters,
                    node_path=route.node_path,
                )
            )

        return tuple(actions)
