"""Shared contracts for Epic 5 routing (User Story 5.1 and beyond).

These are pure dataclasses/enums with no persistence or pathfinding logic of
their own - they are the shared vocabulary that NodeMappingService (Task 0/1)
and the later pathfinding step (e.g. Dijkstra) build from and pass between
each other. Keeping them here, independent of any single service, lets both
sides of routing agree on shapes without importing each other.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from numbers import Real

from src.models.response_target_type import ResponseTargetType


class RouteStatus(str, Enum):
    """Outcome of attempting to route one resource to one response target."""

    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    UNMAPPABLE = "unmappable"


@dataclass(frozen=True)
class RoutingResource:
    """A firefighting resource's identity and location, as routing needs it."""

    resource_id: str
    station_id: str
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        _validate_non_empty_str("resource_id", self.resource_id)
        _validate_non_empty_str("station_id", self.station_id)
        _validate_latitude(self.latitude)
        _validate_longitude(self.longitude)


@dataclass(frozen=True)
class RoutingTarget:
    """A response target's identity, location, and priority, as routing needs it."""

    response_target_id: int
    target_order: int
    target_type: str  # ResponseTargetType
    latitude: float
    longitude: float
    priority_score: float

    def __post_init__(self) -> None:
        _validate_positive_int("response_target_id", self.response_target_id)
        _validate_non_negative_int("target_order", self.target_order)
        _validate_target_type(self.target_type)
        _validate_latitude(self.latitude)
        _validate_longitude(self.longitude)
        _validate_finite_number("priority_score", self.priority_score)


@dataclass(frozen=True)
class RouteResult:
    """The outcome of routing one RoutingResource to one RoutingTarget.

    `node_path` is the ordered sequence of GraphNode ids from source to
    target (inclusive). Field combinations are validated per `status`:

    - REACHABLE: source_node_id, target_node_id, a non-empty node_path
      running between them, and non-negative distance/time are all required.
    - UNREACHABLE: both endpoints mapped to a node, but no path connects
      them - node_path/distance/time stay empty/None.
    - UNMAPPABLE: at least one endpoint's coordinate could not be mapped to
      any road-network node at all - node_path/distance/time stay
      empty/None.
    """

    resource_id: str
    response_target_id: int
    status: RouteStatus
    source_node_id: int | None
    target_node_id: int | None
    node_path: tuple[int, ...]
    distance_meters: float | None
    travel_time_seconds: float | None

    def __post_init__(self) -> None:
        _validate_non_empty_str("resource_id", self.resource_id)
        _validate_positive_int("response_target_id", self.response_target_id)
        if not isinstance(self.status, RouteStatus):
            raise ValueError(f"status must be a RouteStatus, got {self.status!r}")
        _validate_optional_node_id("source_node_id", self.source_node_id)
        _validate_optional_node_id("target_node_id", self.target_node_id)
        _validate_node_path(self.node_path)

        if self.status is RouteStatus.REACHABLE:
            self._validate_reachable()
        else:
            self._validate_unreachable_or_unmappable()

    def _validate_reachable(self) -> None:
        if self.source_node_id is None or self.target_node_id is None:
            raise ValueError("REACHABLE routes must have both source_node_id and target_node_id set.")
        if (
            not self.node_path
            or self.node_path[0] != self.source_node_id
            or self.node_path[-1] != self.target_node_id
        ):
            raise ValueError(
                "REACHABLE routes must have a node_path starting at source_node_id and ending at "
                f"target_node_id, got node_path={self.node_path!r} for source_node_id="
                f"{self.source_node_id!r}, target_node_id={self.target_node_id!r}."
            )
        _validate_non_negative_finite("distance_meters", self.distance_meters)
        _validate_non_negative_finite("travel_time_seconds", self.travel_time_seconds)

    def _validate_unreachable_or_unmappable(self) -> None:
        if self.node_path:
            raise ValueError(f"{self.status.value} routes must have an empty node_path, got {self.node_path!r}.")
        if self.distance_meters is not None:
            raise ValueError(
                f"{self.status.value} routes must not have distance_meters, got {self.distance_meters!r}."
            )
        if self.travel_time_seconds is not None:
            raise ValueError(
                f"{self.status.value} routes must not have travel_time_seconds, "
                f"got {self.travel_time_seconds!r}."
            )

        if self.status is RouteStatus.UNMAPPABLE:
            if self.source_node_id is not None and self.target_node_id is not None:
                raise ValueError(
                    "UNMAPPABLE routes must be missing source_node_id and/or target_node_id, got both set."
                )
        elif self.source_node_id is None or self.target_node_id is None:
            raise ValueError("UNREACHABLE routes must have both source_node_id and target_node_id set.")


@dataclass(frozen=True)
class RoutePlanningRun:
    """One complete routing pass: every resource considered and the RouteResult for each attempted pairing."""

    fire_event_id: int
    response_target_set_id: int
    planned_at: datetime
    methodology: str
    methodology_version: str
    resource_ids: tuple[str, ...]
    routes: tuple[RouteResult, ...]

    def __post_init__(self) -> None:
        _validate_positive_int("fire_event_id", self.fire_event_id)
        _validate_positive_int("response_target_set_id", self.response_target_set_id)
        if not isinstance(self.planned_at, datetime) or self.planned_at.tzinfo is None:
            raise ValueError(f"planned_at must be a timezone-aware datetime, got {self.planned_at!r}")
        _validate_non_empty_str("methodology", self.methodology)
        _validate_non_empty_str("methodology_version", self.methodology_version)

        resource_ids = _coerce_tuple("resource_ids", self.resource_ids)
        for resource_id in resource_ids:
            _validate_non_empty_str("resource_ids", resource_id)
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError(f"resource_ids must not contain duplicates, got {resource_ids!r}")
        object.__setattr__(self, "resource_ids", resource_ids)

        routes = _coerce_tuple("routes", self.routes)
        seen_pairs: set[tuple[str, int]] = set()
        for route in routes:
            if not isinstance(route, RouteResult):
                raise ValueError(f"routes must contain RouteResult items, got {route!r}")
            if route.resource_id not in resource_ids:
                raise ValueError(
                    "route resource_id must be one of the run's resource_ids, got "
                    f"{route.resource_id!r} not in {resource_ids!r}"
                )
            pair = (route.resource_id, route.response_target_id)
            if pair in seen_pairs:
                raise ValueError(f"routes must not contain a duplicate (resource_id, response_target_id) pair, got {pair!r}")
            seen_pairs.add(pair)
        object.__setattr__(self, "routes", routes)


@dataclass(frozen=True)
class StoredRouteResult:
    """A persisted RouteResult plus its database identity."""

    id: int
    route_result: RouteResult

    def __post_init__(self) -> None:
        _validate_positive_int("id", self.id)
        if not isinstance(self.route_result, RouteResult):
            raise ValueError(f"route_result must be a RouteResult, got {self.route_result!r}")


def _coerce_tuple(field_name: str, value: object) -> tuple:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable, got {value!r}") from exc


def _validate_non_empty_str(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")


def _validate_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def _validate_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")


def _validate_finite_number(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")


def _validate_non_negative_finite(field_name: str, value: object) -> None:
    if value is None:
        raise ValueError(f"{field_name} must be a finite number, got None")
    _validate_finite_number(field_name, value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")


def _validate_latitude(value: object) -> None:
    _validate_finite_number("latitude", value)
    if not -90.0 <= value <= 90.0:
        raise ValueError(f"latitude must be within [-90, 90], got {value!r}")


def _validate_longitude(value: object) -> None:
    _validate_finite_number("longitude", value)
    if not -180.0 <= value <= 180.0:
        raise ValueError(f"longitude must be within [-180, 180], got {value!r}")


def _validate_target_type(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError(f"target_type must be a string, got {value!r}")
    valid_values = {member.value for member in ResponseTargetType}
    if value not in valid_values:
        raise ValueError(f"target_type must be one of {sorted(valid_values)!r}, got {value!r}")


def _validate_optional_node_id(field_name: str, value: object) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"{field_name} must be a positive integer or None, got {value!r}")


def _validate_node_path(value: object) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"node_path must be a tuple of positive ints, got {value!r}")
    for node_id in value:
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id <= 0:
            raise ValueError(f"node_path must contain only positive ints, got {node_id!r} in {value!r}")
