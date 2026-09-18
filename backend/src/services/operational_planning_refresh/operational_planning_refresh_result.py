"""Aggregate result for one OperationalPlanningRefreshCoordinator call."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.operational_refresh.operational_refresh_result import OperationalRefreshResult

if TYPE_CHECKING:
    from src.services.global_planning.global_planning_refresh_coordinator import GlobalPlanningRefreshResult


@dataclass(frozen=True)
class OperationalPlanningRefreshResult:
    """Pass-through aggregate of one US 4.4 result and the Stage 6 global
    planning-refresh result it triggered.

    Does not re-summarize or reinterpret either nested result - callers
    inspect `operational_result` and `global_planning_result` directly,
    exactly as OperationalRefreshOrchestrator/GlobalPlanningRefreshCoordinator
    already validated and returned them. In particular, whether the global
    cycle actually reran or was a NO_OP is never decided here; that
    information is already on GlobalPlanningRefreshResult.status.

    `global_planning_result` is:
    - None when the operational refresh did not succeed (a global refresh
      must never run after an operational failure - see __post_init__);
    - otherwise exactly one GlobalPlanningRefreshResult, covering every
      currently active FireEvent together (Stage 6, Task 47) - never one
      result per FireEvent, and never a fan-out loop.

    Stage 6 (Task 47): this replaces the pre-Stage-6 `planning_results`
    tuple of per-FireEvent PlanningRefreshResult items, which used to fan
    out refresh(A); refresh(B); refresh(C) through independent legacy
    per-event planners. That per-event port/result type (PlanningRefreshPort/
    PlanningRefreshResult) remains in source and still works for direct/
    historical use - it is simply no longer what this coordinator calls.

    Not isinstance-checked against the concrete GlobalPlanningRefreshResult
    class: that type lives in a package with real DB dependencies, and it
    was already validated at its own construction.
    """

    operational_result: OperationalRefreshResult
    global_planning_result: "GlobalPlanningRefreshResult | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.operational_result, OperationalRefreshResult):
            raise ValueError(
                f"operational_result must be an OperationalRefreshResult, got {self.operational_result!r}."
            )
        if not self.operational_result.success and self.global_planning_result is not None:
            raise ValueError("global_planning_result must be None when the operational refresh did not succeed.")


@dataclass(frozen=True)
class OperationalPlanningRefreshBatchResult:
    """Aggregate result for one BATCH of FireEvents sharing a single logical
    operational update (Stage 6, Task 34: "avoid duplicate global replans
    for one logical batch - trigger one global refresh after the logical
    update batch completes").

    `operational_results` holds one US 4.4 result per FireEvent in the
    batch (severity/spread/target regeneration genuinely is per-FireEvent).
    `global_planning_result` is the single global refresh triggered ONCE
    after the whole batch's operational refreshes complete - None only when
    NONE of the batch's operational refreshes succeeded.
    """

    operational_results: tuple[OperationalRefreshResult, ...]
    global_planning_result: "GlobalPlanningRefreshResult | None" = None

    def __post_init__(self) -> None:
        try:
            operational_results = tuple(self.operational_results)
        except TypeError as exc:
            raise ValueError(f"operational_results must be iterable, got {self.operational_results!r}.") from exc
        for result in operational_results:
            if not isinstance(result, OperationalRefreshResult):
                raise ValueError(f"operational_results must contain OperationalRefreshResult items, got {result!r}.")
        object.__setattr__(self, "operational_results", operational_results)

        any_success = any(result.success for result in operational_results)
        if not any_success and self.global_planning_result is not None:
            raise ValueError(
                "global_planning_result must be None when no operational refresh in the batch succeeded."
            )
