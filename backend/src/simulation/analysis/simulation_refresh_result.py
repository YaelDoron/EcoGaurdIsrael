"""Result for simulation-triggered central operational refresh."""
from __future__ import annotations

from dataclasses import dataclass

from src.services.operational_refresh.operational_refresh_result import OperationalRefreshResult


@dataclass(frozen=True)
class SimulationRefreshResult:
    """Whether a simulation event triggered production operational refresh."""

    triggered: bool
    refresh_results: tuple[OperationalRefreshResult, ...] = ()
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
