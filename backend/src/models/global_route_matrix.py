"""Immutable, lookup-optimized cross-event route matrix (Stage 3 of the
Global Multi-Incident Optimizer refactor).

Built once by GlobalRouteMatrixBuilder and handed to the future global GA
(Stage 4) as-is: the GA must be able to evaluate fitness purely from this
in-memory structure, never re-querying repositories or re-running Dijkstra.
Cross-referential validation against a resource/target universe (every
route's resource/target actually exist) happens one level up, in
GlobalPlanningInput, which is the only place that has both lists - this
matrix only guarantees internal consistency: no duplicate (resource_id,
response_target_id) pair, deterministic iteration order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.models.global_route_option import GlobalRouteOption


@dataclass(frozen=True)
class GlobalRouteMatrix:
    """Deduplicated, order-deterministic collection of feasible GlobalRouteOptions with O(1) lookup."""

    options: tuple[GlobalRouteOption, ...]
    _lookup: dict[tuple[str, int], GlobalRouteOption] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        options = _coerce_tuple("options", self.options)
        for option in options:
            if not isinstance(option, GlobalRouteOption):
                raise ValueError(f"options must contain GlobalRouteOption items, got {option!r}")

        ordered = tuple(sorted(options, key=lambda option: (option.resource_id, option.response_target_id)))
        lookup: dict[tuple[str, int], GlobalRouteOption] = {}
        for option in ordered:
            key = (option.resource_id, option.response_target_id)
            if key in lookup:
                raise ValueError(f"options must not contain a duplicate (resource_id, response_target_id) pair, got {key!r}")
            lookup[key] = option

        object.__setattr__(self, "options", ordered)
        object.__setattr__(self, "_lookup", lookup)

    def get(self, resource_id: str, response_target_id: int) -> GlobalRouteOption | None:
        """Return the feasible route for (resource_id, response_target_id), or None if infeasible/absent."""
        return self._lookup.get((resource_id, response_target_id))

    def __len__(self) -> int:
        return len(self.options)

    def __iter__(self):
        return iter(self.options)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, tuple) and len(key) == 2 and key in self._lookup


def _coerce_tuple(field_name: str, value: object) -> tuple:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable, got {value!r}") from exc
