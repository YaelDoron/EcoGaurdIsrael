"""Tests for OperationalPlanningRefreshResult's structural invariants."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.services.operational_planning_refresh.operational_planning_refresh_result import (
    OperationalPlanningRefreshResult,
)
from src.services.operational_refresh.operational_refresh_result import (
    OperationalRefreshResult,
    OperationalRefreshStatus,
)

AS_OF = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class FakePlanningRefreshResult:
    status: str
    fire_event_id: int


def success_result() -> OperationalRefreshResult:
    return OperationalRefreshResult(
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        status=OperationalRefreshStatus.REFRESHED,
        success=True,
        fire_event_id=1,
        as_of=AS_OF,
    )


def failure_result() -> OperationalRefreshResult:
    return OperationalRefreshResult(
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        status=OperationalRefreshStatus.FAILED,
        success=False,
        fire_event_id=1,
        as_of=AS_OF,
        error_message="boom",
    )


def test_successful_operational_result_may_carry_planning_results():
    planning_result = FakePlanningRefreshResult(status="refreshed", fire_event_id=1)

    result = OperationalPlanningRefreshResult(operational_result=success_result(), planning_results=(planning_result,))

    assert result.planning_results == (planning_result,)


def test_successful_operational_result_may_carry_zero_planning_results():
    result = OperationalPlanningRefreshResult(operational_result=success_result(), planning_results=())

    assert result.planning_results == ()


def test_failed_operational_result_defaults_to_no_planning_results():
    result = OperationalPlanningRefreshResult(operational_result=failure_result())

    assert result.planning_results == ()


def test_failed_operational_result_with_planning_results_is_rejected():
    planning_result = FakePlanningRefreshResult(status="refreshed", fire_event_id=1)

    with pytest.raises(ValueError):
        OperationalPlanningRefreshResult(operational_result=failure_result(), planning_results=(planning_result,))


def test_planning_results_is_normalized_to_a_tuple():
    planning_result = FakePlanningRefreshResult(status="refreshed", fire_event_id=1)

    result = OperationalPlanningRefreshResult(
        operational_result=success_result(), planning_results=[planning_result]
    )

    assert result.planning_results == (planning_result,)


def test_non_operational_refresh_result_is_rejected():
    with pytest.raises(ValueError):
        OperationalPlanningRefreshResult(operational_result="not a result")
