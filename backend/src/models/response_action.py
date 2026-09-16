"""Pure response-plan assignment action model."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_validation import normalize_resource_id, resource_sort_key, validate_positive_int


@dataclass(frozen=True)
class ResponseAction:
    """Recommended assignment of one resource to one response target."""

    resource_id: int | str
    response_target_id: int
    route_result_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_id", normalize_resource_id(self.resource_id))
        validate_positive_int("response_target_id", self.response_target_id)
        validate_positive_int("route_result_id", self.route_result_id)

    @property
    def ordering_key(self) -> tuple[tuple[str, str], int, int]:
        return (resource_sort_key(self.resource_id), self.response_target_id, self.route_result_id)
