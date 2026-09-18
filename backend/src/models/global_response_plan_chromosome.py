"""Resource-indexed chromosome for the Global GA (Stage 4 of the Global
Multi-Incident Optimizer refactor, Task 2/5/6).

Unlike the legacy, per-event ResponsePlanChromosome (target-indexed: one
gene per ResponseTarget, valued by resource id), this chromosome is
RESOURCE-indexed: one gene per physical resource, valued by the
GlobalAllocationSlot it is assigned to (or None for idle). This makes

    R1 -> Fire A
    R1 -> Fire B

structurally impossible: resource_ids has no duplicates and each resource
gets exactly one gene, so it can hold at most one assignment, globally,
by construction - never by a rule the GA has to remember to enforce.

`resource_ids` is carried inside the chromosome itself (not merely implied
by an external input's ordering, unlike the legacy chromosome) so each
instance is fully self-describing and self-validating.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_validation import validate_non_empty_string


@dataclass(frozen=True)
class GlobalResponsePlanChromosome:
    """genes[i] is the slot_id resource_ids[i] is assigned to, or None (idle)."""

    resource_ids: tuple[str, ...]
    genes: tuple[str | None, ...]

    def __post_init__(self) -> None:
        try:
            resource_ids = tuple(self.resource_ids)
        except TypeError as exc:
            raise ValueError("resource_ids must be iterable.") from exc
        for resource_id in resource_ids:
            validate_non_empty_string("resource_ids", resource_id)
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("resource_ids must not contain duplicates.")
        object.__setattr__(self, "resource_ids", resource_ids)

        try:
            genes = tuple(self.genes)
        except TypeError as exc:
            raise ValueError("genes must be iterable.") from exc
        if len(genes) != len(resource_ids):
            raise ValueError(
                f"genes length ({len(genes)}) must equal resource_ids length ({len(resource_ids)})."
            )

        assigned_slot_ids: list[str] = []
        for gene in genes:
            if gene is None:
                continue
            validate_non_empty_string("gene (slot_id)", gene)
            assigned_slot_ids.append(gene)
        if len(assigned_slot_ids) != len(set(assigned_slot_ids)):
            raise ValueError("genes must not assign the same slot_id more than once.")
        object.__setattr__(self, "genes", genes)
