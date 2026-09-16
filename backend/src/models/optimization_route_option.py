"""Pure route facts consumed by response optimization."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_validation import (
    normalize_resource_id,
    validate_finite_non_negative_number,
    validate_positive_int,
)


@dataclass(frozen=True)
class OptimizationRouteOption:
    """Normalized routing facts from future US 5.1 output.

    This is not a US 5.1 RouteResult. It contains only the trace id and
    reachability/metric facts required by US 5.2 optimization.
    """

    route_result_id: int
    resource_id: int | str
    response_target_id: int
    is_reachable: bool
    travel_time_seconds: float | None
    distance_meters: float | None

    def __post_init__(self) -> None:
        validate_positive_int("route_result_id", self.route_result_id)
        object.__setattr__(self, "resource_id", normalize_resource_id(self.resource_id))
        validate_positive_int("response_target_id", self.response_target_id)
        if not isinstance(self.is_reachable, bool):
            raise ValueError(f"is_reachable must be a bool, got {self.is_reachable!r}")

        if self.is_reachable:
            if self.travel_time_seconds is None:
                raise ValueError("reachable route options require travel_time_seconds.")
            if self.distance_meters is None:
                raise ValueError("reachable route options require distance_meters.")
            validate_finite_non_negative_number("travel_time_seconds", self.travel_time_seconds)
            validate_finite_non_negative_number("distance_meters", self.distance_meters)
        else:
            if self.travel_time_seconds is not None:
                raise ValueError("non-reachable route options must not include travel_time_seconds.")
            if self.distance_meters is not None:
                raise ValueError("non-reachable route options must not include distance_meters.")

    @property
    def resource_target_key(self) -> tuple[int | str, int]:
        return (self.resource_id, self.response_target_id)
