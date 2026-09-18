"""Hard dispatch lock: makes "an already-dispatched resource crosses to
another FireEvent" a STRUCTURALLY IMPOSSIBLE chromosome state (Stage 6 of
the Global Multi-Incident Optimizer refactor, Tasks 3/6/16), not merely a
large fitness penalty.

Two cooperating pieces:

1. `build_global_optimization_problem` (see global_response_optimization_problem.py)
   restricts a locked resource's `feasible_slot_ids_by_resource` entry to
   ONLY slots belonging to its locked FireEvent - so no allele-generating
   code path (initial population, crossover, mutation) can ever even
   *consider* assigning it to another event's slot. If a locked resource
   has zero feasible slots for its own event (a genuinely unreachable
   truck, or a target that no longer exists), `GlobalHardDispatchLockInfeasible`
   is raised rather than silently dropping the lock or guessing a target.

2. `enforce_hard_locks` (this module) is the final repair step every
   chromosome-producing code path runs before returning a chromosome: any
   locked resource left unassigned (or, defensively, pointing at a slot
   outside its restricted feasible set) is given one of its own feasible
   slots, evicting whichever OTHER resource currently holds it if every
   feasible slot is already claimed. A locked resource's claim on its own
   FireEvent's slots is unconditional - it always wins that contention.

This never relies on a fitness penalty a mutation/crossover step might
statistically fail to correct - the illegal region of the search space is
simply never reachable.
"""
from __future__ import annotations

from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome


class GlobalHardDispatchLockInfeasible(ValueError):
    """Raised when a hard-dispatched resource has no feasible slot left for
    its own locked FireEvent (Task 16) - never silently dropped or guessed."""


def enforce_hard_locks(
    problem: "GlobalResponseOptimizationProblem",  # noqa: F821 - avoid import cycle; duck-typed on purpose
    chromosome: GlobalResponsePlanChromosome,
) -> GlobalResponsePlanChromosome:
    """Return a chromosome identical to `chromosome` except every locked
    resource is guaranteed one of its own feasible slots, in deterministic
    (resource_id-ordered) priority."""
    if not problem.locked_resource_ids:
        return chromosome

    gene_by_resource: dict[str, str | None] = dict(zip(chromosome.resource_ids, chromosome.genes))
    slot_owner: dict[str, str] = {
        slot_id: resource_id for resource_id, slot_id in gene_by_resource.items() if slot_id is not None
    }

    for resource_id in sorted(problem.locked_resource_ids):
        feasible = problem.feasible_slot_ids_by_resource.get(resource_id, ())
        current = gene_by_resource.get(resource_id)
        if current is not None and current in feasible:
            continue  # already legally satisfying its own lock

        target_slot = next((slot_id for slot_id in feasible if slot_id not in slot_owner), None)
        if target_slot is None:
            if not feasible:
                raise GlobalHardDispatchLockInfeasible(
                    f"hard-dispatched resource {resource_id!r} has no feasible slot for its locked FireEvent."
                )
            # Every feasible slot is already claimed - a locked resource's
            # claim on its own FireEvent's slots is unconditional, so evict
            # the current (deterministic, first-by-slot-ordering) claimant.
            target_slot = feasible[0]
            evicted_resource = slot_owner[target_slot]
            gene_by_resource[evicted_resource] = None

        if current is not None:
            slot_owner.pop(current, None)
        gene_by_resource[resource_id] = target_slot
        slot_owner[target_slot] = resource_id

    genes = tuple(gene_by_resource[resource_id] for resource_id in problem.resource_ids)
    return GlobalResponsePlanChromosome(problem.resource_ids, genes)
