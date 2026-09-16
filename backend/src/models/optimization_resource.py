"""Pure response-optimization resource input model."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_validation import normalize_resource_id, resource_sort_key


@dataclass(frozen=True)
class OptimizationResource:
    """Eligible firefighting resource normalized for assignment optimization."""

    resource_id: int | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_id", normalize_resource_id(self.resource_id))

    @property
    def ordering_key(self) -> tuple[str, str]:
        return resource_sort_key(self.resource_id)
