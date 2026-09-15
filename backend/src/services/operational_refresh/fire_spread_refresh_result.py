"""Result models for operational fire-spread refresh orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType


class FireSpreadRefreshHorizonStatus(Enum):
    """Outcome of refreshing one fire-spread prediction horizon."""

    REFRESHED = "refreshed"
    NO_OP = "no_op"
    INSUFFICIENT_DATA = "insufficient_data"
    INACTIVE_EVENT = "inactive_event"
    FAILED = "failed"


@dataclass(frozen=True)
class FireSpreadRefreshHorizonResult:
    """Refresh outcome for one FireEvent/horizon pair."""

    fire_event_id: int
    horizon_minutes: int
    status: FireSpreadRefreshHorizonStatus
    prediction_id: int | None = None
    previous_prediction_id: int | None = None
    effective_state_fingerprint: str | None = None
    error_message: str | None = None

    @property
    def prediction_created(self) -> bool:
        return self.prediction_id is not None and self.status is not FireSpreadRefreshHorizonStatus.NO_OP


@dataclass(frozen=True)
class FireSpreadRefreshResult:
    """Aggregate result for one fire-spread refresh request."""

    fire_event_id: int
    trigger_type: OperationalRefreshTriggerType
    as_of: datetime
    reevaluation_required: bool
    horizon_results: tuple[FireSpreadRefreshHorizonResult, ...] = ()

    @property
    def predictions_created(self) -> int:
        return sum(1 for result in self.horizon_results if result.prediction_created)

    @property
    def no_ops(self) -> int:
        return sum(1 for result in self.horizon_results if result.status is FireSpreadRefreshHorizonStatus.NO_OP)

    @property
    def insufficient(self) -> int:
        return sum(
            1 for result in self.horizon_results if result.status is FireSpreadRefreshHorizonStatus.INSUFFICIENT_DATA
        )

    @property
    def inactive(self) -> int:
        return sum(1 for result in self.horizon_results if result.status is FireSpreadRefreshHorizonStatus.INACTIVE_EVENT)

    @property
    def failed(self) -> int:
        return sum(1 for result in self.horizon_results if result.status is FireSpreadRefreshHorizonStatus.FAILED)
