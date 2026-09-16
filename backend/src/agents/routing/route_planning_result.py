"""Result object for a RoutePlanningAgent run."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.repositories.route_planning_repository import StoredRoutePlanningRun


class RoutePlanningStatus(Enum):
    """Business/operational outcome of one route-planning orchestration run."""

    PLANNED = "planned"
    NO_TARGETS = "no_targets"
    NO_RESOURCES_AVAILABLE = "no_resources_available"
    FAILED = "failed"


@dataclass(frozen=True)
class RoutePlanningResult:
    """Outcome of one RoutePlanningAgent orchestration run."""

    success: bool
    fire_event_id: int
    status: RoutePlanningStatus
    run_id: int | None
    run: StoredRoutePlanningRun | None
    route_count: int = 0
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError(f"success must be a bool, got {self.success!r}.")
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}.")
        if not isinstance(self.status, RoutePlanningStatus):
            raise ValueError(f"status must be a RoutePlanningStatus, got {self.status!r}.")
        if isinstance(self.route_count, bool) or not isinstance(self.route_count, int) or self.route_count < 0:
            raise ValueError(f"route_count must be a non-negative integer, got {self.route_count!r}.")
        if self.run is not None and not isinstance(self.run, StoredRoutePlanningRun):
            raise ValueError(f"run must be a StoredRoutePlanningRun or None, got {self.run!r}.")

        if self.status is RoutePlanningStatus.PLANNED:
            self._validate_planned()
        elif self.status is RoutePlanningStatus.NO_RESOURCES_AVAILABLE:
            self._validate_no_resources_available()
        elif self.status is RoutePlanningStatus.NO_TARGETS:
            self._validate_no_targets()
        elif self.status is RoutePlanningStatus.FAILED:
            self._validate_failed()

    def _validate_planned(self) -> None:
        if not self.success:
            raise ValueError("PLANNED route-planning results must be successful.")
        if isinstance(self.run_id, bool) or not isinstance(self.run_id, int) or self.run_id <= 0:
            raise ValueError("PLANNED route-planning results must include a positive run_id.")
        if self.run is None:
            raise ValueError("PLANNED route-planning results must include the saved run.")
        if self.run.id != self.run_id:
            raise ValueError("PLANNED route-planning results must have run.id equal to run_id.")
        if self.route_count == 0:
            raise ValueError("PLANNED route-planning results must include at least one route.")
        if self.route_count != len(self.run.routes):
            raise ValueError("route_count must equal len(run.routes).")
        if self.error_message is not None:
            raise ValueError("PLANNED route-planning results must not include error_message.")

    def _validate_no_resources_available(self) -> None:
        if not self.success:
            raise ValueError("NO_RESOURCES_AVAILABLE route-planning results must be successful.")
        if isinstance(self.run_id, bool) or not isinstance(self.run_id, int) or self.run_id <= 0:
            raise ValueError("NO_RESOURCES_AVAILABLE route-planning results must include a positive run_id.")
        if self.run is None:
            raise ValueError("NO_RESOURCES_AVAILABLE route-planning results must include the saved run.")
        if self.route_count != 0:
            raise ValueError("NO_RESOURCES_AVAILABLE route-planning results must have zero routes.")
        if self.error_message is not None:
            raise ValueError("NO_RESOURCES_AVAILABLE route-planning results must not include error_message.")

    def _validate_no_targets(self) -> None:
        if not self.success:
            raise ValueError("NO_TARGETS route-planning results must be successful.")
        if self.run_id is not None:
            raise ValueError("NO_TARGETS route-planning results must not include run_id.")
        if self.run is not None:
            raise ValueError("NO_TARGETS route-planning results must not include run.")
        if self.route_count != 0:
            raise ValueError("NO_TARGETS route-planning results must have zero routes.")
        if self.error_message is not None:
            raise ValueError("NO_TARGETS route-planning results must not include error_message.")

    def _validate_failed(self) -> None:
        if self.success:
            raise ValueError("FAILED route-planning results must not be successful.")
        if self.run_id is not None:
            raise ValueError("FAILED route-planning results must not include run_id.")
        if self.run is not None:
            raise ValueError("FAILED route-planning results must not include run.")
        if self.route_count != 0:
            raise ValueError("FAILED route-planning results must have zero routes.")
        if not isinstance(self.error_message, str) or not self.error_message.strip():
            raise ValueError("FAILED route-planning results must include error_message.")
