"""Tests for OperationalPlanningRefreshResult / OperationalPlanningRefreshBatchResult's
structural invariants (Stage 6, Task 47: single global_planning_result, not
a per-FireEvent planning_results tuple)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.services.operational_planning_refresh.operational_planning_refresh_result import (
    OperationalPlanningRefreshBatchResult,
    OperationalPlanningRefreshResult,
)
from src.services.operational_refresh.operational_refresh_result import (
    OperationalRefreshResult,
    OperationalRefreshStatus,
)

AS_OF = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class FakeGlobalPlanningRefreshResult:
    status: str


def success_result(fire_event_id: int = 1) -> OperationalRefreshResult:
    return OperationalRefreshResult(
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        status=OperationalRefreshStatus.REFRESHED,
        success=True,
        fire_event_id=fire_event_id,
        as_of=AS_OF,
    )


def failure_result(fire_event_id: int = 1) -> OperationalRefreshResult:
    return OperationalRefreshResult(
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        status=OperationalRefreshStatus.FAILED,
        success=False,
        fire_event_id=fire_event_id,
        as_of=AS_OF,
        error_message="boom",
    )


# ---------------------------------------------------------------------------
# OperationalPlanningRefreshResult
# ---------------------------------------------------------------------------


def test_successful_operational_result_may_carry_a_global_planning_result():
    global_planning_result = FakeGlobalPlanningRefreshResult(status="activated")

    result = OperationalPlanningRefreshResult(
        operational_result=success_result(), global_planning_result=global_planning_result
    )

    assert result.global_planning_result == global_planning_result


def test_successful_operational_result_may_carry_no_global_planning_result():
    result = OperationalPlanningRefreshResult(operational_result=success_result(), global_planning_result=None)

    assert result.global_planning_result is None


def test_failed_operational_result_defaults_to_no_global_planning_result():
    result = OperationalPlanningRefreshResult(operational_result=failure_result())

    assert result.global_planning_result is None


def test_failed_operational_result_with_a_global_planning_result_is_rejected():
    global_planning_result = FakeGlobalPlanningRefreshResult(status="activated")

    with pytest.raises(ValueError):
        OperationalPlanningRefreshResult(
            operational_result=failure_result(), global_planning_result=global_planning_result
        )


def test_non_operational_refresh_result_is_rejected():
    with pytest.raises(ValueError):
        OperationalPlanningRefreshResult(operational_result="not a result")


# ---------------------------------------------------------------------------
# OperationalPlanningRefreshBatchResult
# ---------------------------------------------------------------------------


def test_batch_result_with_at_least_one_success_may_carry_a_global_planning_result():
    global_planning_result = FakeGlobalPlanningRefreshResult(status="activated")

    result = OperationalPlanningRefreshBatchResult(
        operational_results=(success_result(1), failure_result(2)), global_planning_result=global_planning_result
    )

    assert result.global_planning_result == global_planning_result
    assert len(result.operational_results) == 2


def test_batch_result_with_no_successes_rejects_a_global_planning_result():
    global_planning_result = FakeGlobalPlanningRefreshResult(status="activated")

    with pytest.raises(ValueError):
        OperationalPlanningRefreshBatchResult(
            operational_results=(failure_result(1), failure_result(2)),
            global_planning_result=global_planning_result,
        )


def test_batch_result_with_no_successes_defaults_to_no_global_planning_result():
    result = OperationalPlanningRefreshBatchResult(operational_results=(failure_result(1), failure_result(2)))

    assert result.global_planning_result is None


def test_batch_result_operational_results_is_normalized_to_a_tuple():
    result = OperationalPlanningRefreshBatchResult(operational_results=[success_result(1)])

    assert result.operational_results == (success_result(1),)


def test_batch_result_rejects_non_operational_refresh_result_items():
    with pytest.raises(ValueError):
        OperationalPlanningRefreshBatchResult(operational_results=("not a result",))
