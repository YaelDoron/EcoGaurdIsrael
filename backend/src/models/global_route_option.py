"""Global optimizer-facing route model (Stage 3 of the Global Multi-Incident
Optimizer refactor).

A GlobalRouteOption exists only for a FEASIBLE (REACHABLE) resource-target
pairing - infeasible pairs simply have no GlobalRouteOption (see
GlobalRouteMatrix), which the future GA must treat as an illegal
assignment. `fire_event_id` is included even though it is derivable from
the target's own ownership, to let a route be validated/traced without a
second lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class GlobalRouteOption:
    """One feasible resource -> target route, with its ETA/distance/path."""

    resource_id: str
    fire_event_id: int
    response_target_id: int
    eta_seconds: float
    route_distance_meters: float
    node_path: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string("resource_id", self.resource_id)
        _validate_positive_int("fire_event_id", self.fire_event_id)
        _validate_positive_int("response_target_id", self.response_target_id)
        _validate_non_negative_finite("eta_seconds", self.eta_seconds)
        _validate_non_negative_finite("route_distance_meters", self.route_distance_meters)

        node_path = _coerce_tuple("node_path", self.node_path)
        if not node_path:
            raise ValueError("node_path must not be empty for a feasible GlobalRouteOption.")
        for node_id in node_path:
            if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id <= 0:
                raise ValueError(f"node_path must contain only positive ints, got {node_id!r} in {node_path!r}")
        object.__setattr__(self, "node_path", node_path)


def _coerce_tuple(field_name: str, value: object) -> tuple:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable, got {value!r}") from exc


def _validate_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def _validate_non_empty_string(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")


def _validate_non_negative_finite(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")
