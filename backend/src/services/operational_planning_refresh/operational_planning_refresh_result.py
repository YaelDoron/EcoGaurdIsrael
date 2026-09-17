"""Aggregate result for one OperationalPlanningRefreshCoordinator call."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.operational_refresh.operational_refresh_result import OperationalRefreshResult

if TYPE_CHECKING:
    from src.services.response_planning.planning_refresh_result import PlanningRefreshResult


@dataclass(frozen=True)
class OperationalPlanningRefreshResult:
    """Pass-through aggregate of one US 4.4 result and the US 5.4 result(s) it triggered.

    Does not re-summarize or reinterpret either nested result - callers
    inspect `operational_result` and `planning_results` directly, exactly as
    OperationalRefreshOrchestrator/ResponsePlanningRefreshOrchestrator
    already validated and returned them. In particular, whether a planning
    cycle actually reran or was a NO_OP is never decided here; that
    information is already on each PlanningRefreshResult.

    `planning_results` holds:
    - zero entries when the operational refresh did not succeed (planning
      must never run after an operational failure - see __post_init__);
    - exactly one entry for a FireEvent/environmental refresh;
    - zero or more entries for a resource refresh (one per currently active
      FireEvent checked as a candidate).

    Items are not isinstance-checked against the concrete PlanningRefreshResult
    class: that type lives in a package with an optional osmnx dependency
    (see operational_planning_refresh_ports.py), and every item reaching this
    aggregate was already validated at its own construction by
    ResponsePlanningRefreshOrchestrator/PlanningRefreshResult itself.
    """

    operational_result: OperationalRefreshResult
    planning_results: tuple["PlanningRefreshResult", ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operational_result, OperationalRefreshResult):
            raise ValueError(
                f"operational_result must be an OperationalRefreshResult, got {self.operational_result!r}."
            )
        try:
            planning_results = tuple(self.planning_results)
        except TypeError as exc:
            raise ValueError(f"planning_results must be iterable, got {self.planning_results!r}.") from exc
        object.__setattr__(self, "planning_results", planning_results)
        if not self.operational_result.success and planning_results:
            raise ValueError("planning_results must be empty when the operational refresh did not succeed.")
