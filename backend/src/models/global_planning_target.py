"""Global optimizer-facing target model (Stage 3 of the Global Multi-Incident
Optimizer refactor).

A GlobalPlanningTarget is a ResponseTarget as the future global GA needs it:
flat, provenance-light routing/priority data, but ALWAYS keeping its owning
`fire_event_id` - the future GA assigns resources globally while still
needing to reconstruct per-FireEvent ResponsePlans afterward. Deliberately
does not carry severity/demand semantics yet (Stage 5).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.response_target_type import ResponseTargetType


@dataclass(frozen=True)
class GlobalPlanningTarget:
    """One response target, as the global optimizer needs it, with its owning FireEvent."""

    fire_event_id: int
    response_target_id: int
    target_order: int
    target_type: ResponseTargetType
    latitude: float
    longitude: float
    priority_score: float
    prediction_horizon_minutes: int | None = None

    def __post_init__(self) -> None:
        _validate_positive_int("fire_event_id", self.fire_event_id)
        _validate_positive_int("response_target_id", self.response_target_id)
        _validate_non_negative_int("target_order", self.target_order)
        if not isinstance(self.target_type, ResponseTargetType):
            raise ValueError(f"target_type must be a ResponseTargetType, got {self.target_type!r}")
        _validate_latitude(self.latitude)
        _validate_longitude(self.longitude)
        _validate_finite_number("priority_score", self.priority_score)
        if self.prediction_horizon_minutes is not None:
            _validate_non_negative_int("prediction_horizon_minutes", self.prediction_horizon_minutes)


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
