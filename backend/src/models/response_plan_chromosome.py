"""Pure chromosome representation for response-plan optimization."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_validation import normalize_resource_id


@dataclass(frozen=True)
class ResponsePlanChromosome:
    """Target-indexed resource-allocation genome.

    Gene positions correspond to the deterministic target ordering in the
    matching ResponseOptimizationInput. A None gene leaves that target
    uncovered; a resource id gene assigns that resource to the target at the
    same position.
    """

    genes: tuple[int | str | None, ...]

    def __post_init__(self) -> None:
        try:
            genes = tuple(self.genes)
        except TypeError as exc:
            raise ValueError("genes must be iterable.") from exc

        normalized_genes: list[int | str | None] = []
        resource_genes: list[int | str] = []
        for gene in genes:
            if gene is None:
                normalized_genes.append(None)
                continue
            resource_id = normalize_resource_id(gene)
            normalized_genes.append(resource_id)
            resource_genes.append(resource_id)

        if len(resource_genes) != len(set(resource_genes)):
            raise ValueError("genes must not assign the same resource_id more than once.")

        object.__setattr__(self, "genes", tuple(normalized_genes))
