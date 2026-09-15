"""Tests for SimulationFireSpreadCoordinator trigger policy.

Mirrors test_simulation_fire_severity_coordinator.py's fake-agent approach:
the real FireSpreadPredictionAgent (and everything inside it -- input
selection, PROPAGATOR calculation, persistence) is already exhaustively
proven correct in Tasks 4B-9's own unit and live-Neon integration suites.
This file verifies only the coordinator's own trigger/orchestration logic:
which FireEvents it calls the agent for, at which horizons, and how it
records success/failure -- using a fake agent, never a live one.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
from src.models import (
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    FireSpreadPrediction,
    FireSpreadPredictionStatus,
)
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
from src.repositories.fire_spread_prediction_repository import StoredFireSpreadPrediction
from src.simulation.analysis import SimulationFireSpreadCoordinator
from src.simulation.analysis.simulation_fire_spread_coordinator import (
    NO_SEVERITY_ASSESSMENTS_REASON,
    SEVERITY_NOT_TRIGGERED_REASON,
)
from src.simulation.analysis.simulation_fire_severity_result import SimulationFireSeverityResult
from src.simulation.analysis.simulation_fire_spread_result import SimulationFireSpreadResult

STARTED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
T0 = STARTED_AT
T65 = STARTED_AT + timedelta(seconds=65)


class FakeSpreadAgent:
    def __init__(self, fail_keys=()) -> None:
        # fail_keys: set of (fire_event_id, horizon_minutes) tuples that raise.
        self.fail_keys = set(fail_keys)
        self.calls = []
        self._next_id = 1000

    def predict(self, fire_event_id, as_of, horizon_minutes):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of, "horizon_minutes": horizon_minutes})
        if (fire_event_id, horizon_minutes) in self.fail_keys:
            raise RuntimeError(f"spread failed for {fire_event_id}/{horizon_minutes}")
        self._next_id += 1
        return StoredFireSpreadPrediction(
            id=self._next_id,
            prediction=FireSpreadPrediction(
                fire_event_id=fire_event_id,
                severity_assessment_id=fire_event_id + 5000,
                predicted_at=as_of,
                horizon_minutes=horizon_minutes,
                status=FireSpreadPredictionStatus.VALID,
                methodology=METHODOLOGY_NAME,
                methodology_version=METHODOLOGY_VERSION,
                cells=(),
            ),
            weather_observation_id=fire_event_id + 9000,
        )


def make_stored_assessment(fire_event_id: int, assessed_at: datetime = T0) -> StoredFireSeverityAssessment:
    return StoredFireSeverityAssessment(
        assessment_id=fire_event_id + 1000,
        assessment=FireSeverityAssessment(
            fire_event_id=fire_event_id,
            assessed_at=assessed_at,
            status=FireSeverityAssessmentStatus.VALID,
            score=72.5,
            level=FireSeverityLevel.HIGH,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        ),
        weather_observation_ids=(1,),
        satellite_hotspot_ids=(2,),
        selected_frp_hotspot_id=2,
    )


def make_severity_result(
    *,
    triggered: bool = True,
    fire_event_ids: tuple[int, ...] = (55,),
    failed_fire_event_ids: tuple[int, ...] = (),
    reason: str | None = None,
) -> SimulationFireSeverityResult:
    if not triggered:
        return SimulationFireSeverityResult(triggered=False, reason=reason or "not_triggered")
    return SimulationFireSeverityResult(
        triggered=True,
        assessment_results=tuple(make_stored_assessment(event_id) for event_id in fire_event_ids),
        failed_fire_event_ids=failed_fire_event_ids,
        error_messages=tuple(f"severity failed for {event_id}" for event_id in failed_fire_event_ids),
    )


# ---------------------------------------------------------------------------
# A. Active incident -- both horizons predicted and persisted
# ---------------------------------------------------------------------------


def test_triggered_severity_predicts_both_horizons_for_the_event():
    agent = FakeSpreadAgent()
    coordinator = SimulationFireSpreadCoordinator(agent)
    severity_result = make_severity_result(fire_event_ids=(55,))

    result = coordinator.handle_severity_result(severity_result, T65)

    assert result.triggered is True
    assert agent.calls == [
        {"fire_event_id": 55, "as_of": T65, "horizon_minutes": 30},
        {"fire_event_id": 55, "as_of": T65, "horizon_minutes": 60},
    ]
    assert len(result.prediction_results) == 2
    assert {p.prediction.horizon_minutes for p in result.prediction_results} == {30, 60}
    assert all(p.prediction.fire_event_id == 55 for p in result.prediction_results)
    assert all(p.prediction.predicted_at == T65 for p in result.prediction_results)
    assert all(p.prediction.methodology == METHODOLOGY_NAME for p in result.prediction_results)
    assert all(p.prediction.methodology_version == METHODOLOGY_VERSION for p in result.prediction_results)


def test_severity_assessment_id_is_not_fabricated_by_coordinator():
    """The coordinator must not invent traceability -- whatever the (fake,
    here standing in for the real) agent decided is what gets returned."""
    agent = FakeSpreadAgent()
    coordinator = SimulationFireSpreadCoordinator(agent)

    result = coordinator.handle_severity_result(make_severity_result(fire_event_ids=(55,)), T65)

    assert all(p.prediction.severity_assessment_id == 55 + 5000 for p in result.prediction_results)


# ---------------------------------------------------------------------------
# Multi-incident: predictions stay associated with the correct FireEvent
# ---------------------------------------------------------------------------


def test_multiple_fire_events_are_predicted_independently_without_cross_contamination():
    agent = FakeSpreadAgent()
    coordinator = SimulationFireSpreadCoordinator(agent)
    severity_result = make_severity_result(fire_event_ids=(10, 20))

    result = coordinator.handle_severity_result(severity_result, T65)

    assert agent.calls == [
        {"fire_event_id": 10, "as_of": T65, "horizon_minutes": 30},
        {"fire_event_id": 10, "as_of": T65, "horizon_minutes": 60},
        {"fire_event_id": 20, "as_of": T65, "horizon_minutes": 30},
        {"fire_event_id": 20, "as_of": T65, "horizon_minutes": 60},
    ]
    by_event = {}
    for stored in result.prediction_results:
        by_event.setdefault(stored.prediction.fire_event_id, set()).add(stored.prediction.horizon_minutes)
    assert by_event == {10: {30, 60}, 20: {30, 60}}


def test_duplicate_fire_event_ids_in_severity_result_are_deduplicated():
    agent = FakeSpreadAgent()
    coordinator = SimulationFireSpreadCoordinator(agent)
    severity_result = SimulationFireSeverityResult(
        triggered=True,
        assessment_results=(make_stored_assessment(55, T0), make_stored_assessment(55, T0 + timedelta(minutes=1))),
    )

    coordinator.handle_severity_result(severity_result, T65)

    assert agent.calls == [
        {"fire_event_id": 55, "as_of": T65, "horizon_minutes": 30},
        {"fire_event_id": 55, "as_of": T65, "horizon_minutes": 60},
    ]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_call_order_is_deterministic_sorted_by_event_then_horizon():
    agent = FakeSpreadAgent()
    coordinator = SimulationFireSpreadCoordinator(agent)
    severity_result = make_severity_result(fire_event_ids=(30, 10, 20))

    coordinator.handle_severity_result(severity_result, T65)

    assert [call["fire_event_id"] for call in agent.calls] == [10, 10, 20, 20, 30, 30]
    assert [call["horizon_minutes"] for call in agent.calls] == [30, 60, 30, 60, 30, 60]


def test_same_severity_result_and_timestamp_produce_equivalent_calls():
    severity_result = make_severity_result(fire_event_ids=(55,))

    first_agent = FakeSpreadAgent()
    SimulationFireSpreadCoordinator(first_agent).handle_severity_result(severity_result, T65)
    second_agent = FakeSpreadAgent()
    SimulationFireSpreadCoordinator(second_agent).handle_severity_result(severity_result, T65)

    assert first_agent.calls == second_agent.calls


# ---------------------------------------------------------------------------
# D. Inactive-event passthrough: coordinator never inspects/overrides status
# ---------------------------------------------------------------------------


def test_coordinator_passes_through_whatever_status_the_agent_produces():
    """Simulates what the real agent returns for a RESOLVED/DISMISSED
    FireEvent: an INACTIVE_EVENT-status prediction with zero cells. The
    coordinator must record it as-is, not fabricate or filter it."""

    class InactiveEventAgent(FakeSpreadAgent):
        def predict(self, fire_event_id, as_of, horizon_minutes):
            self.calls.append(
                {"fire_event_id": fire_event_id, "as_of": as_of, "horizon_minutes": horizon_minutes}
            )
            return StoredFireSpreadPrediction(
                id=1,
                prediction=FireSpreadPrediction(
                    fire_event_id=fire_event_id,
                    severity_assessment_id=None,
                    predicted_at=as_of,
                    horizon_minutes=horizon_minutes,
                    status=FireSpreadPredictionStatus.INACTIVE_EVENT,
                    methodology=METHODOLOGY_NAME,
                    methodology_version=METHODOLOGY_VERSION,
                    cells=(),
                ),
                weather_observation_id=None,
            )

    agent = InactiveEventAgent()
    coordinator = SimulationFireSpreadCoordinator(agent)

    result = coordinator.handle_severity_result(make_severity_result(fire_event_ids=(55,)), T65)

    assert result.triggered is True
    assert all(p.prediction.status is FireSpreadPredictionStatus.INACTIVE_EVENT for p in result.prediction_results)
    assert all(p.prediction.cells == () for p in result.prediction_results)


# ---------------------------------------------------------------------------
# Not triggered
# ---------------------------------------------------------------------------


def test_severity_not_triggered_produces_no_spread_trigger():
    agent = FakeSpreadAgent()
    coordinator = SimulationFireSpreadCoordinator(agent)

    result = coordinator.handle_severity_result(make_severity_result(triggered=False), T65)

    assert result.triggered is False
    assert result.reason == SEVERITY_NOT_TRIGGERED_REASON
    assert agent.calls == []


def test_severity_triggered_with_only_failures_produces_no_spread_trigger():
    agent = FakeSpreadAgent()
    coordinator = SimulationFireSpreadCoordinator(agent)
    severity_result = SimulationFireSeverityResult(
        triggered=True,
        failed_fire_event_ids=(55,),
        error_messages=("severity failed for 55",),
    )

    result = coordinator.handle_severity_result(severity_result, T65)

    assert result.triggered is False
    assert result.reason == NO_SEVERITY_ASSESSMENTS_REASON
    assert agent.calls == []


# ---------------------------------------------------------------------------
# Partial failure
# ---------------------------------------------------------------------------


def test_agent_failure_for_one_horizon_does_not_stop_the_other():
    agent = FakeSpreadAgent(fail_keys={(55, 60)})
    coordinator = SimulationFireSpreadCoordinator(agent)

    result = coordinator.handle_severity_result(make_severity_result(fire_event_ids=(55,)), T65)

    assert result.triggered is True
    assert len(result.prediction_results) == 1
    assert result.prediction_results[0].prediction.horizon_minutes == 30
    assert result.failed_fire_event_ids == (55,)
    assert result.error_messages == ("spread failed for 55/60",)


def test_one_event_failing_does_not_stop_another_event():
    agent = FakeSpreadAgent(fail_keys={(10, 30), (10, 60)})
    coordinator = SimulationFireSpreadCoordinator(agent)

    result = coordinator.handle_severity_result(make_severity_result(fire_event_ids=(10, 20)), T65)

    assert tuple(sorted({p.prediction.fire_event_id for p in result.prediction_results})) == (20,)
    assert result.failed_fire_event_ids == (10,)
    assert len(result.error_messages) == 2


# ---------------------------------------------------------------------------
# Result model validation
# ---------------------------------------------------------------------------


def test_simulation_fire_spread_result_is_immutable():
    result = SimulationFireSpreadResult(
        triggered=True,
        prediction_results=(FakeSpreadAgent().predict(55, T0, 30),),
    )

    with pytest.raises(FrozenInstanceError):
        result.triggered = False


def test_non_triggered_result_requires_reason():
    with pytest.raises(ValueError):
        SimulationFireSpreadResult(triggered=False)


def test_triggered_result_without_predictions_or_failures_rejected():
    with pytest.raises(ValueError):
        SimulationFireSpreadResult(triggered=True)


def test_non_triggered_result_with_predictions_rejected():
    with pytest.raises(ValueError):
        SimulationFireSpreadResult(
            triggered=False,
            reason="x",
            prediction_results=(FakeSpreadAgent().predict(55, T0, 30),),
        )
