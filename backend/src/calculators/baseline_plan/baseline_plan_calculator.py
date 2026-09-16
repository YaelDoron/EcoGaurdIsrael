"""Pure deterministic greedy baseline resource-to-target allocation.

Implements User Story 5.3 Task 1: a deterministic "nearest available
resource" baseline used later to compare against the optimized (GA) plan.

This module is a pure function of its inputs: no database access, no
routing/Dijkstra calculation, no ETA/priority/severity/spread calculation,
no agent invocation, no GA/fitness logic, and no persistence or dispatch.
It only consumes already-prepared target order and already-computed route
results and deterministically assigns resources to targets.

Epic 5 Task 0 froze shared contracts (`RouteResult`, `RouteStatus`,
`StoredRouteResult`, `ResponseAction`) that are owned by other companies'
User Stories (routing: US 5.1; GA/scoring: US 5.2). As of Task 1, none of
those models exist yet in this codebase (no `RouteResult`, `RouteStatus`,
or `ResponseAction` anywhere under `src/`). Rather than fabricate
production copies of contracts this team does not own, this calculator
defines its own minimal, additive, locally-scoped pure input/output types
below (`RouteCandidate`, `TargetOrder`, `BaselineAssignment`,
`BaselinePlanAllocation`) that structurally mirror the fields Task 1 needs
from those frozen contracts:

- `RouteCandidate.status` uses the frozen `RouteStatus` string values
  ("reachable" / "unreachable" / "unmappable") directly, as a `Literal`
  rather than a new `Enum` class, so this module never defines a competing
  type identity for the real `RouteStatus` enum once US 5.1 merges it.
- `RouteCandidate.resource_id` / `BaselineAssignment.resource_id` are
  `str`, matching the frozen `resource_id: str` contract.
- `TargetOrder` mirrors the persisted `response_target_id` /
  `target_order` fields (see `ResponseTargetDB`) needed to reproduce the
  persisted operational processing order without recalculating
  `priority_score`.

Task 2 (once US 5.1/5.2 are merged) is expected to translate between these
and the real shared models -- e.g. build a `RouteCandidate` from each
`StoredRouteResult`, and a `ResponseAction` from each `BaselineAssignment`.

Epic 5 Task 6.1 note: `TargetOrder.target_type`/`.priority_score` and
`RouteCandidate.distance_meters` were added once US 5.1/5.2's real models
existed, purely to carry already-loaded snapshot data through to the shared
`ResponsePlanScorer` (see `baseline_comparison_service.py`). None of them are
read by `BaselinePlanCalculator.allocate()` itself - target processing
order, candidate eligibility, and tie-breaking are unchanged; only the
information carried alongside each target/candidate was enriched.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, Sequence

from src.models.response_target_type import ResponseTargetType

# Local mirror of the frozen `RouteStatus` contract's persisted string
# values (lowercase: "reachable" / "unreachable" / "unmappable"). A
# `Literal` is used instead of a new `Enum` class so this module never
# defines a competing type identity for the real `RouteStatus` enum once
# US 5.1 merges it -- callers can pass `route_status.value`.
RouteCandidateStatus = Literal["reachable", "unreachable", "unmappable"]
_VALID_STATUSES = ("reachable", "unreachable", "unmappable")


@dataclass(frozen=True)
class RouteCandidate:
    """Pure input mirroring one persisted route result for one resource/target pair.

    Structurally equivalent to a `StoredRouteResult` + its `RouteResult`
    (`route_result_id`, `resource_id`, `response_target_id`, `status`,
    `travel_time_seconds`), scoped to only the fields this baseline
    calculator needs.
    """

    route_result_id: int
    resource_id: str
    response_target_id: int
    status: RouteCandidateStatus
    travel_time_seconds: float | None
    distance_meters: float | None

    def __post_init__(self) -> None:
        _validate_positive_int("route_result_id", self.route_result_id)
        _validate_resource_id(self.resource_id)
        _validate_positive_int("response_target_id", self.response_target_id)
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {_VALID_STATUSES}, got {self.status!r}")
        if self.travel_time_seconds is not None:
            _validate_finite_non_negative("travel_time_seconds", self.travel_time_seconds)
        if self.distance_meters is not None:
            _validate_finite_non_negative("distance_meters", self.distance_meters)


@dataclass(frozen=True)
class TargetOrder:
    """Persisted processing-order identity for one response target.

    Structurally mirrors the frozen `response_target_id` / `target_order`
    fields (see `ResponseTargetDB`). The calculator sorts by this
    contract, never by the incidental order targets are supplied in.
    """

    response_target_id: int
    target_order: int
    target_type: ResponseTargetType
    priority_score: float

    def __post_init__(self) -> None:
        _validate_positive_int("response_target_id", self.response_target_id)
        _validate_non_negative_int("target_order", self.target_order)
        if not isinstance(self.target_type, ResponseTargetType):
            raise ValueError(f"target_type must be a ResponseTargetType, got {self.target_type!r}")
        _validate_finite("priority_score", self.priority_score)


@dataclass(frozen=True)
class BaselineAssignment:
    """One deterministic resource-to-target assignment produced by the baseline.

    Structurally mirrors the frozen `ResponseAction` contract
    (`resource_id`, `response_target_id`, `route_result_id`) so Task 2 can
    build a real `ResponseAction` from it once that shared model is merged.
    """

    resource_id: str
    response_target_id: int
    route_result_id: int

    def __post_init__(self) -> None:
        _validate_resource_id(self.resource_id)
        _validate_positive_int("response_target_id", self.response_target_id)
        _validate_positive_int("route_result_id", self.route_result_id)


@dataclass(frozen=True)
class BaselinePlanAllocation:
    """Pure result of the greedy baseline allocation for one route-planning run."""

    assignments: tuple[BaselineAssignment, ...]
    uncovered_response_target_ids: tuple[int, ...]


class BaselinePlanCalculator:
    """Deterministic nearest-available-resource greedy baseline allocation."""

    def allocate(
        self,
        *,
        targets: Sequence[TargetOrder],
        route_candidates: Sequence[RouteCandidate],
    ) -> BaselinePlanAllocation:
        """Assign at most one resource per target, in persisted target order.

        Targets are processed by ascending `target_order` (ties broken by
        `response_target_id`), never by the incidental order `targets` is
        supplied in. For each target, selects the "reachable", unused
        candidate with the lowest `travel_time_seconds`, breaking ties by
        `resource_id` (ascending lexicographic). Targets with no eligible
        candidate are left uncovered; processing continues.
        """
        ordered_target_ids = _sort_targets(targets)
        candidates_by_target = _group_by_target(route_candidates)

        assigned_resource_ids: set[str] = set()
        assignments: list[BaselineAssignment] = []
        uncovered_response_target_ids: list[int] = []

        for target_id in ordered_target_ids:
            eligible = [
                candidate
                for candidate in candidates_by_target.get(target_id, ())
                if candidate.status == "reachable"
                and candidate.travel_time_seconds is not None
                and candidate.resource_id not in assigned_resource_ids
            ]
            if not eligible:
                uncovered_response_target_ids.append(target_id)
                continue

            chosen = min(eligible, key=_tie_break_key)
            assigned_resource_ids.add(chosen.resource_id)
            assignments.append(
                BaselineAssignment(
                    resource_id=chosen.resource_id,
                    response_target_id=target_id,
                    route_result_id=chosen.route_result_id,
                )
            )

        return BaselinePlanAllocation(
            assignments=tuple(assignments),
            uncovered_response_target_ids=tuple(uncovered_response_target_ids),
        )


def _sort_targets(targets: Sequence[TargetOrder]) -> tuple[int, ...]:
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise ValueError(f"targets must be a sequence, got {targets!r}")

    for target in targets:
        if not isinstance(target, TargetOrder):
            raise ValueError(f"targets must contain TargetOrder items, got {target!r}")

    # Explicitly sort by the persisted target_order contract (ties broken by
    # response_target_id for full determinism) -- processing order must
    # never depend on the incidental order `targets` was supplied in.
    ordered = sorted(targets, key=lambda target: (target.target_order, target.response_target_id))
    return tuple(target.response_target_id for target in ordered)


def _group_by_target(route_candidates: Sequence[RouteCandidate]) -> dict[int, list[RouteCandidate]]:
    if not isinstance(route_candidates, Sequence) or isinstance(route_candidates, (str, bytes)):
        raise ValueError(f"route_candidates must be a sequence, got {route_candidates!r}")

    grouped: dict[int, list[RouteCandidate]] = {}
    for candidate in route_candidates:
        if not isinstance(candidate, RouteCandidate):
            raise ValueError(f"route_candidates must contain RouteCandidate items, got {candidate!r}")
        grouped.setdefault(candidate.response_target_id, []).append(candidate)
    return grouped


def _tie_break_key(candidate: RouteCandidate) -> tuple[float, str]:
    # Primary: lowest travel_time_seconds. Secondary (equal ETA): lowest
    # resource_id via a stable ascending lexicographic comparison, per the
    # frozen `resource_id: str` contract (e.g. "R1" < "R2",
    # "truck-001" < "truck-002").
    return (candidate.travel_time_seconds, candidate.resource_id)


def _validate_resource_id(resource_id: object) -> None:
    if not isinstance(resource_id, str) or not resource_id.strip():
        raise ValueError(f"resource_id must be a non-empty string, got {resource_id!r}")


def _validate_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def _validate_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")


def _validate_finite_non_negative(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number, got {value!r}")


def _validate_finite(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")
