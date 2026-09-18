"""Configuration for WHICH resources are eligible to be considered by global
planning (Stage 3 of the Global Multi-Incident Optimizer refactor) - never
how many a fire "needs" or which should be assigned. That remains Stage 4+.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalCandidateResourceConfig:
    """Candidate-geography knobs for GlobalCandidateCollector.

    `search_radii_km[0]` is the entry radius passed to
    OperationalContextService.get_stations_in_operational_area per active
    FireEvent - that existing method already owns its own progressive
    widening (5 -> 20 -> 50 km, matching this config's own default) and an
    unconditional closest-station fallback if nothing is found at any
    radius (see Task 1: reused unchanged, not reimplemented). Elements of
    `search_radii_km` beyond index 0 and `include_closest_fallback`
    therefore currently describe behavior Stage 3 INHERITS from that
    method rather than independently enforces; they are kept as an
    explicit config surface (not silently hidden) for a future
    OperationalContextService enhancement that takes the full sequence.

    `max_fallback_stations`, unlike the above, IS independently enforced
    here: if the per-event unioned station set is smaller than this after
    the per-event searches, GlobalCandidateCollector adds next-nearest
    stations (by distance to any active event) up to this cap. `None`
    disables this extra top-up.

    `max_demand_driven_stations` (Stage 5, Tasks 11-12): a SEPARATE,
    independent cap on how many ADDITIONAL stations may be pulled in
    specifically to satisfy total desired suppression demand (as opposed to
    `max_fallback_stations`, which only tops up an otherwise-empty/too-small
    candidate pool regardless of demand). `None` (the default) means no cap
    - GlobalCandidateCollector keeps adding next-nearest stations until
    assignable supply meets desired demand or no eligible stations remain.
    """

    search_radii_km: tuple[float, ...] = (5.0, 20.0, 50.0)
    include_closest_fallback: bool = True
    max_fallback_stations: int | None = None
    max_demand_driven_stations: int | None = None

    def __post_init__(self) -> None:
        radii = tuple(self.search_radii_km)
        if not radii:
            raise ValueError("search_radii_km must not be empty.")
        for radius_km in radii:
            if isinstance(radius_km, bool) or not isinstance(radius_km, (int, float)) or radius_km <= 0:
                raise ValueError(f"search_radii_km entries must be positive numbers, got {radius_km!r}")
        object.__setattr__(self, "search_radii_km", radii)

        if not isinstance(self.include_closest_fallback, bool):
            raise ValueError(
                f"include_closest_fallback must be a bool, got {self.include_closest_fallback!r}"
            )

        if self.max_fallback_stations is not None:
            if isinstance(self.max_fallback_stations, bool) or not isinstance(self.max_fallback_stations, int) or self.max_fallback_stations <= 0:
                raise ValueError(
                    f"max_fallback_stations must be a positive integer or None, got {self.max_fallback_stations!r}"
                )

        if self.max_demand_driven_stations is not None:
            if (
                isinstance(self.max_demand_driven_stations, bool)
                or not isinstance(self.max_demand_driven_stations, int)
                or self.max_demand_driven_stations <= 0
            ):
                raise ValueError(
                    "max_demand_driven_stations must be a positive integer or None, got "
                    f"{self.max_demand_driven_stations!r}"
                )
