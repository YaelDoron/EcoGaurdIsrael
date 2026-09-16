"""Pure Planning Effective State domain models (Epic 5, User Story 5.4, Tasks 1 and 3).

These are semantic representations of the current operational planning state,
used by later tasks to decide whether a ResponsePlan needs to be recalculated.
They deliberately exclude provenance/persistence identifiers (response_target_id,
response_target_set_id, spread_prediction_id, spread_prediction_cell_id,
route_planning_run_id, response_plan_id, timestamps) so that equality - and the
fingerprint below - reflect only what materially affects planning output, not
persistence history.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real

from src.calculators.response_optimization.response_optimization_config import (
    METHODOLOGY as OPTIMIZATION_METHODOLOGY_NAME,
    METHODOLOGY_VERSION as OPTIMIZATION_METHODOLOGY_VERSION,
)
from src.calculators.routing.routing_config import ROUTING_METHODOLOGY_NAME, ROUTING_METHODOLOGY_VERSION
from src.models.response_target_type import ResponseTargetType


@dataclass(frozen=True)
class PlanningTargetState:
    """Target properties that materially affect planning, independent of provenance."""

    target_type: ResponseTargetType
    latitude: float
    longitude: float
    priority_score: float
    prediction_horizon_minutes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_type, ResponseTargetType):
            raise ValueError(f"target_type must be a ResponseTargetType, got {self.target_type!r}")
        _validate_latitude(self.latitude)
        _validate_longitude(self.longitude)
        _validate_finite_number("priority_score", self.priority_score)
        if self.prediction_horizon_minutes is not None:
            _validate_non_negative_int("prediction_horizon_minutes", self.prediction_horizon_minutes)

    def _canonical_payload(self) -> tuple[tuple[str, object], ...]:
        return (
            ("target_type", self.target_type.value),
            ("latitude", self.latitude),
            ("longitude", self.longitude),
            ("priority_score", self.priority_score),
            ("prediction_horizon_minutes", self.prediction_horizon_minutes),
        )


@dataclass(frozen=True)
class PlanningResourceState:
    """An AVAILABLE firefighting resource's identity and station origin, as planning needs it."""

    resource_id: str
    station_id: str
    station_latitude: float
    station_longitude: float

    def __post_init__(self) -> None:
        _validate_non_empty_string("resource_id", self.resource_id)
        _validate_non_empty_string("station_id", self.station_id)
        _validate_latitude(self.station_latitude)
        _validate_longitude(self.station_longitude)

    def _canonical_payload(self) -> tuple[tuple[str, object], ...]:
        return (
            ("resource_id", self.resource_id),
            ("station_id", self.station_id),
            ("station_latitude", self.station_latitude),
            ("station_longitude", self.station_longitude),
        )


@dataclass(frozen=True)
class PlanningEffectiveState:
    """The effective operational planning state for one FireEvent.

    Represents current targets, available resources, and the methodology
    identity that will act on them - not which persisted records produced
    them.
    """

    fire_event_id: int
    targets: tuple[PlanningTargetState, ...]
    resources: tuple[PlanningResourceState, ...]
    routing_methodology: str = ROUTING_METHODOLOGY_NAME
    routing_methodology_version: str = ROUTING_METHODOLOGY_VERSION
    optimization_methodology: str = OPTIMIZATION_METHODOLOGY_NAME
    optimization_methodology_version: str = OPTIMIZATION_METHODOLOGY_VERSION

    def __post_init__(self) -> None:
        _validate_positive_int("fire_event_id", self.fire_event_id)

        targets = _coerce_tuple("targets", self.targets)
        for target in targets:
            if not isinstance(target, PlanningTargetState):
                raise ValueError(f"targets must contain PlanningTargetState items, got {target!r}")
        object.__setattr__(self, "targets", targets)

        resources = _coerce_tuple("resources", self.resources)
        for resource in resources:
            if not isinstance(resource, PlanningResourceState):
                raise ValueError(f"resources must contain PlanningResourceState items, got {resource!r}")
        object.__setattr__(self, "resources", resources)

        _validate_non_empty_string("routing_methodology", self.routing_methodology)
        _validate_non_empty_string("routing_methodology_version", self.routing_methodology_version)
        _validate_non_empty_string("optimization_methodology", self.optimization_methodology)
        _validate_non_empty_string("optimization_methodology_version", self.optimization_methodology_version)

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 hex digest of the canonical effective planning state.

        Order-independent over `targets`/`resources`, so a newly persisted
        snapshot with identical semantic content produces the same
        fingerprint as an earlier one, regardless of collection order.
        """
        payload = json.dumps(self._canonical_payload(), separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _canonical_payload(self) -> tuple[tuple[str, object], ...]:
        sorted_targets = sorted(
            (target._canonical_payload() for target in self.targets),
            key=_target_payload_sort_key,
        )
        sorted_resources = sorted(
            (resource._canonical_payload() for resource in self.resources),
            key=_resource_payload_sort_key,
        )
        return (
            ("fire_event_id", self.fire_event_id),
            ("targets", tuple(sorted_targets)),
            ("resources", tuple(sorted_resources)),
            ("routing_methodology", self.routing_methodology),
            ("routing_methodology_version", self.routing_methodology_version),
            ("optimization_methodology", self.optimization_methodology),
            ("optimization_methodology_version", self.optimization_methodology_version),
        )


def _target_payload_sort_key(payload: tuple[tuple[str, object], ...]) -> tuple[str, float, float, float, int]:
    values = dict(payload)
    horizon = values["prediction_horizon_minutes"]
    return (
        values["target_type"],
        values["latitude"],
        values["longitude"],
        values["priority_score"],
        -1 if horizon is None else horizon,
    )


def _resource_payload_sort_key(payload: tuple[tuple[str, object], ...]) -> tuple[str, str, float, float]:
    values = dict(payload)
    return (
        values["station_id"],
        values["resource_id"],
        values["station_latitude"],
        values["station_longitude"],
    )


def _coerce_tuple(field_name: str, value: object) -> tuple:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable, got {value!r}") from exc


def _validate_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def _validate_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")


def _validate_finite_number(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")


def _validate_latitude(value: object) -> None:
    _validate_finite_number("latitude", value)
    if not -90.0 <= value <= 90.0:
        raise ValueError(f"latitude must be within [-90, 90], got {value!r}")


def _validate_longitude(value: object) -> None:
    _validate_finite_number("longitude", value)
    if not -180.0 <= value <= 180.0:
        raise ValueError(f"longitude must be within [-180, 180], got {value!r}")


def _validate_non_empty_string(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
