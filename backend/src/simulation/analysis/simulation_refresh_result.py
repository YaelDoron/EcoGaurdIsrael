"""Result for simulation-triggered central operational + global planning refresh."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.operational_refresh.operational_refresh_result import OperationalRefreshResult

if TYPE_CHECKING:
    from src.services.global_planning.global_planning_refresh_coordinator import GlobalPlanningRefreshResult


@dataclass(frozen=True)
class SimulationRefreshResult:
    """Whether a simulation event triggered production operational + global planning refresh.

    `refresh_results` holds the US 4.4 operational-refresh outcome (one per
    FireEvent, or one for a resource update), unchanged in shape from before
    the US 4.4 -> Stage 6 bridge existed. `global_planning_result` is the
    Stage 6 GlobalPlanningRefreshResult the bridge triggered ONCE for the
    whole batch - None whenever no operational refresh in the batch
    succeeded, since global planning never runs in that case (see
    OperationalPlanningRefreshCoordinator).

    Stage 6 Task 48 (simulation cutover): this replaces the pre-Stage-6
    `planning_results` tuple of per-FireEvent PlanningRefreshResult items -
    the simulation now exercises the exact same GlobalPlanningRefreshCoordinator
    production path as every other trigger, never a separate demo planner.
    """

    triggered: bool
    refresh_results: tuple[OperationalRefreshResult, ...] = ()
    global_planning_result: "GlobalPlanningRefreshResult | None" = None
    fire_event_ids: tuple[int, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.triggered, bool):
            raise ValueError(f"triggered must be a bool, got {self.triggered!r}.")

        refresh_results = tuple(self.refresh_results)
        for refresh_result in refresh_results:
            if not isinstance(refresh_result, OperationalRefreshResult):
                raise ValueError(
                    "refresh_results must contain OperationalRefreshResult items, "
                    f"got {refresh_result!r}."
                )
        object.__setattr__(self, "refresh_results", refresh_results)

        fire_event_ids = tuple(sorted(set(self.fire_event_ids)))
        for fire_event_id in fire_event_ids:
            if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
                raise ValueError(f"fire_event_ids must contain positive integer ids, got {fire_event_id!r}.")
        object.__setattr__(self, "fire_event_ids", fire_event_ids)

        if self.triggered:
            if self.reason is not None:
                raise ValueError("triggered simulation refresh results must not include reason.")
            if not refresh_results:
                raise ValueError("triggered simulation refresh results must include refresh_results.")
        else:
            if refresh_results:
                raise ValueError("non-triggered simulation refresh results must not include refresh_results.")
            if self.global_planning_result is not None:
                raise ValueError("non-triggered simulation refresh results must not include global_planning_result.")
            if fire_event_ids:
                raise ValueError("non-triggered simulation refresh results must not include fire_event_ids.")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("non-triggered simulation refresh results must include reason.")

    @property
    def refreshes_requested(self) -> int:
        return len(self.refresh_results)

    @property
    def failures(self) -> int:
        return sum(1 for result in self.refresh_results if not result.success)
