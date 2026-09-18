"""Global optimizer output action (Stage 4 of the Global Multi-Incident
Optimizer refactor, Task 20). All route facts come from Stage 3's
GlobalRouteMatrix - never recalculated here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.optimization_validation import validate_non_empty_string, validate_positive_int
from src.models.response_target_type import ResponseTargetType


@dataclass(frozen=True)
class GlobalResponseAction:
    """One resource -> target assignment in a GlobalOptimizationResult, carrying its own route facts."""

    resource_id: str
    station_id: str
    fire_event_id: int
    response_target_id: int
    target_type: ResponseTargetType
    target_priority: float
    eta_seconds: float
    route_distance_meters: float
    node_path: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_non_empty_string("resource_id", self.resource_id)
        validate_non_empty_string("station_id", self.station_id)
        validate_positive_int("fire_event_id", self.fire_event_id)
        validate_positive_int("response_target_id", self.response_target_id)
        if not isinstance(self.target_type, ResponseTargetType):
            raise ValueError(f"target_type must be a ResponseTargetType, got {self.target_type!r}")
        self._validate_finite("target_priority", self.target_priority)
        self._validate_non_negative_finite("eta_seconds", self.eta_seconds)
        self._validate_non_negative_finite("route_distance_meters", self.route_distance_meters)

        node_path = tuple(self.node_path)
        if not node_path:
            raise ValueError("node_path must not be empty.")
        for node_id in node_path:
            if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id <= 0:
                raise ValueError(f"node_path must contain only positive ints, got {node_id!r}")
        object.__setattr__(self, "node_path", node_path)

    @staticmethod
    def _validate_finite(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")

    @staticmethod
    def _validate_non_negative_finite(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative finite number, got {value!r}")
