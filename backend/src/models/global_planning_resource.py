"""Global optimizer-facing resource model (Stage 3 of the Global
Multi-Incident Optimizer refactor).

One physical firefighting resource appears exactly once here, regardless of
how many active FireEvents it is geographically near - this is the "global
resource universe" Stage 3 exists to build, replacing per-event candidate
pools.

Eligibility policy (Task 4), deliberately explicit rather than collapsed
into one boolean the caller has to reverse-engineer:

- `operational_status is AVAILABLE` and `current_commitment_fire_event_id
  is None`: freely available - a genuinely unowned resource.
- `current_commitment_fire_event_id is not None`: already committed to an
  active FireEvent (Stage 1's ResourceCommitment) - still `is_assignable`
  (Stage 4 must be able to evaluate preserving OR reassigning it under a
  future anti-thrashing policy - Stage 3 provides information, not that
  policy). Its raw `operational_status` for a committed resource is
  typically ASSIGNED, but Stage 1.1 documents that a resource can also be
  UNAVAILABLE while still holding a commitment (deliberately not released
  prematurely) - see `is_assignable` below for how that case is handled.
- `operational_status is UNAVAILABLE`: NOT assignable, regardless of any
  commitment - the one exclusion rule. Still represented (not dropped from
  the resource list) so the global input stays a complete, honest picture,
  but `is_assignable` is False so Stage 4 cannot select it.

This is the one policy Stage 3 needs; Stage 3 never stops here to decide
who SHOULD get a resource - only who COULD.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.resource_status import ResourceStatus


@dataclass(frozen=True)
class GlobalPlanningResource:
    """One physical firefighting resource, globally deduplicated, with commitment visibility."""

    resource_id: str
    station_id: str
    station_name: str
    station_latitude: float
    station_longitude: float
    operational_status: ResourceStatus
    current_commitment_fire_event_id: int | None = None
    current_commitment_response_plan_id: int | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string("resource_id", self.resource_id)
        _validate_non_empty_string("station_id", self.station_id)
        _validate_non_empty_string("station_name", self.station_name)
        _validate_latitude(self.station_latitude)
        _validate_longitude(self.station_longitude)
        if not isinstance(self.operational_status, ResourceStatus):
            raise ValueError(f"operational_status must be a ResourceStatus, got {self.operational_status!r}")

        if self.current_commitment_fire_event_id is not None:
            _validate_positive_int("current_commitment_fire_event_id", self.current_commitment_fire_event_id)
        if self.current_commitment_response_plan_id is not None:
            _validate_positive_int(
                "current_commitment_response_plan_id", self.current_commitment_response_plan_id
            )
        has_event = self.current_commitment_fire_event_id is not None
        has_plan = self.current_commitment_response_plan_id is not None
        if has_event != has_plan:
            raise ValueError(
                "current_commitment_fire_event_id and current_commitment_response_plan_id must be "
                f"both set or both None, got fire_event_id={self.current_commitment_fire_event_id!r}, "
                f"response_plan_id={self.current_commitment_response_plan_id!r}"
            )

    @property
    def is_committed(self) -> bool:
        """True when a ResourceCommitment currently owns this resource."""
        return self.current_commitment_fire_event_id is not None

    @property
    def is_assignable(self) -> bool:
        """False only for UNAVAILABLE (Task 4's one exclusion rule) - never
        excludes a resource merely for already being committed."""
        return self.operational_status is not ResourceStatus.UNAVAILABLE


def _validate_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def _validate_non_empty_string(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")


def _validate_finite_number(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")


def _validate_latitude(value: object) -> None:
    _validate_finite_number("station_latitude", value)
    if not -90.0 <= value <= 90.0:
        raise ValueError(f"station_latitude must be within [-90, 90], got {value!r}")


def _validate_longitude(value: object) -> None:
    _validate_finite_number("station_longitude", value)
    if not -180.0 <= value <= 180.0:
        raise ValueError(f"station_longitude must be within [-180, 180], got {value!r}")
